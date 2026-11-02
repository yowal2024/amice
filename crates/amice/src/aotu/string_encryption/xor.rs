use crate::aotu::string_encryption::{EncryptedGlobalValue, StringEncryptionAlgo};
use crate::config::{StringDecryptTiming as DecryptTiming, StringEncryptionConfig};
use amice_llvm::inkwell2::{BuilderExt, ModuleExt};
use amice_llvm::ptr_type;
use amice_plugin::inkwell;
use amice_plugin::inkwell::AddressSpace;
use amice_plugin::inkwell::AtomicOrdering;
use amice_plugin::inkwell::attributes::{Attribute, AttributeLoc};
use amice_plugin::inkwell::module::Linkage;
use amice_plugin::inkwell::values::BasicValue;
use inkwell::module::Module;
use inkwell::values::FunctionValue;
use log::{Level, debug, log_enabled, warn};
use rand::Rng;

#[derive(Default)]
pub(super) struct XorAlgo;

impl StringEncryptionAlgo for XorAlgo {
    fn initialize(&mut self, _cfg: &StringEncryptionConfig, _module: &mut Module<'_>) -> anyhow::Result<()> {
        Ok(())
    }

    fn do_string_encrypt(&mut self, cfg: &StringEncryptionConfig, module: &mut Module<'_>) -> anyhow::Result<()> {
        do_handle(cfg, module)
    }
}

fn do_handle<'a>(cfg: &StringEncryptionConfig, module: &mut Module<'a>) -> anyhow::Result<()> {
    let candidates = super::collect_string_candidates(module, cfg);
    let ctx = module.get_context();
    let i32_ty = ctx.i32_type();
    let is_lazy_mode = matches!(cfg.timing, DecryptTiming::Lazy);

    let mut strings = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let global = candidate.global;
        let name = global.get_name().to_string_lossy();
        let mask = rand::rng().random_range(1..=u8::MAX);
        let encoded: Vec<_> = candidate.bytes.iter().map(|byte| byte ^ mask).collect();
        let flag = is_lazy_mode.then(|| {
            let flag = module.add_global(i32_ty, None, &format!("dec_flag_{name}"));
            flag.set_initializer(&i32_ty.const_zero());
            flag.set_linkage(Linkage::Private);
            flag
        });
        global.set_initializer(&super::rebuild_const_string(ctx, candidate.elem_size, &encoded));
        global.set_constant(false);
        let mut string =
            EncryptedGlobalValue::new_array_string(global, encoded.len() as u32, flag, false, candidate.users);
        string.xor_mask = mask;
        strings.push(string);
    }
    let decrypt_fn = add_decrypt_function(
        module,
        &format!("__amice__decrypt_strings_{}", rand::random::<u32>()),
        is_lazy_mode,
        cfg.inline_decrypt,
    )?;
    match cfg.timing {
        DecryptTiming::Lazy => {
            emit_decrypt_before_inst(module, strings.clone(), decrypt_fn, false, false)?;
        },
        DecryptTiming::Global => emit_global_string_decryptor_ctor(module, &strings, decrypt_fn)?,
    }
    for string in strings {
        string.free();
    }
    Ok(())
}
fn emit_decrypt_before_inst<'a>(
    module: &mut Module<'a>,
    strings: Vec<EncryptedGlobalValue<'a>>,
    decrypt_fn: FunctionValue<'a>,
    _stack_alloc: bool,
    _allow_non_entry_stack_alloc: bool,
) -> anyhow::Result<()> {
    let ctx = module.get_context();
    let builder = ctx.create_builder();
    for string in strings {
        if string.user_slice().is_empty() {
            anyhow::bail!("lazy string has no guarded uses; constructor fallback is forbidden");
        }
        let flag = string
            .flag
            .ok_or_else(|| anyhow::anyhow!("lazy string has no once flag"))?;
        let ptr = string.global.as_pointer_value();
        let len = ctx.i32_type().const_int(string.str_len as u64, false);
        for (inst, _) in string.user_slice() {
            let insertion = super::prepare_string_insertion(*inst)?;
            builder.position_before(&insertion);
            builder.build_call(
                decrypt_fn,
                &[
                    ptr.into(),
                    len.into(),
                    flag.as_pointer_value().into(),
                    ptr.into(),
                    ctx.i8_type().const_int(string.xor_mask as u64, false).into(),
                ],
                "",
            )?;
        }
    }
    Ok(())
}
fn emit_global_string_decryptor_ctor<'a>(
    module: &mut Module<'a>,
    global_strings: &Vec<EncryptedGlobalValue<'a>>,
    decrypt_fn: FunctionValue<'a>,
) -> anyhow::Result<()> {
    let ctx = module.get_context();
    let i32_ty = ctx.i32_type();
    let i32_ptr = ptr_type!(ctx, i32_type);

    let decrypt_stub_ty = ctx.void_type().fn_type(&[], false);
    let decrypt_stub = module.add_function("decrypt_strings_stub", decrypt_stub_ty, None);
    decrypt_stub.set_linkage(Linkage::Internal);

    let entry = ctx.append_basic_block(decrypt_stub, "entry");
    let builder = ctx.create_builder();

    builder.position_at_end(entry);
    for string in global_strings {
        assert!(!string.use_stack_alloc);

        let ptr = if string.is_array_string() {
            // 如果是c风格的基于ArrayValue的字符串，那么全局常量本身就是这个字符串
            string.global.as_pointer_value()
        } else {
            assert!(string.is_struct_string());
            let Some(struct_value) = &string.struct_value else {
                panic!("string.struct_value must be Some(StructValue)");
            };
            let Some(field_idx) = string.field_index else {
                panic!("string.field_index must be Some(u32)");
            };
            let global_ptr = string.global.as_pointer_value();
            let value_type = string.global.get_value_type().into_struct_type();

            if log_enabled!(Level::Debug) {
                debug!(
                    "StructString found, type = {:?}, value = {:?}, idx = {}",
                    value_type, struct_value, field_idx
                );
            }

            // 生成 GEP: getelementptr inbounds %struct..., ptr @global, i32 0, i32 field_idx
            builder.build_struct_gep2(value_type, global_ptr, field_idx, "field_gep")?
        };

        let len_val = i32_ty.const_int(string.str_len as u64, false);
        let flag_ptr = i32_ptr.const_null();
        let dst = ptr;
        builder.build_call(
            decrypt_fn,
            &[
                ptr.into(),
                len_val.into(),
                flag_ptr.into(),
                dst.into(),
                ctx.i8_type().const_int(string.xor_mask as u64, false).into(),
            ],
            "",
        )?;
    }

    builder.build_return(None)?;

    let priority = 0; // Default priority
    module.append_to_global_ctors(decrypt_stub, priority);

    Ok(())
}

fn add_decrypt_function<'a>(
    module: &mut Module<'a>,
    name: &str,
    has_flag: bool,
    inline_fn: bool,
) -> anyhow::Result<FunctionValue<'a>> {
    let ctx = module.get_context();
    let i8_ty = ctx.i8_type();
    let i32_ty = ctx.i32_type();
    let i8_ptr = ptr_type!(ctx, i8_type);
    let i32_ptr = ptr_type!(ctx, i32_type);

    // void decrypt_strings(i8* str, i32 len, i32* flag, i8* dst)
    let fn_ty = ctx.void_type().fn_type(
        &[
            i8_ptr.into(),
            i32_ty.into(),
            i32_ptr.into(),
            i8_ptr.into(),
            i8_ty.into(),
        ],
        false,
    );

    let decrypt_fn = module.add_function(name, fn_ty, None);
    decrypt_fn.set_linkage(Linkage::Internal);
    if inline_fn {
        warn!("(strenc) using inline decryption function, this may increase binary size.");
        let inlinehint_attr = ctx.create_enum_attribute(Attribute::get_named_enum_kind_id("alwaysinline"), 0);
        decrypt_fn.add_attribute(AttributeLoc::Function, inlinehint_attr);
    }

    let prepare = ctx.append_basic_block(decrypt_fn, "prepare");
    let prepare_has_flags = if has_flag {
        ctx.append_basic_block(decrypt_fn, "prepare_has_flags").into()
    } else {
        None
    };
    let spin_wait = if has_flag {
        ctx.append_basic_block(decrypt_fn, "spin_wait").into()
    } else {
        None
    };
    let do_mark_done = if has_flag {
        ctx.append_basic_block(decrypt_fn, "do_mark_done").into()
    } else {
        None
    };
    let entry = ctx.append_basic_block(decrypt_fn, "entry");
    let body = ctx.append_basic_block(decrypt_fn, "body");
    let next = ctx.append_basic_block(decrypt_fn, "next");
    let exit = ctx.append_basic_block(decrypt_fn, "exit");

    let ptr = decrypt_fn
        .get_nth_param(0)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow::anyhow!("Failed to get pointer parameter"))?;
    let len = decrypt_fn
        .get_nth_param(1)
        .map(|param| param.into_int_value())
        .ok_or_else(|| anyhow::anyhow!("Failed to get length parameter"))?;
    let flag_ptr = decrypt_fn
        .get_nth_param(2)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow::anyhow!("Failed to get flag parameter"))?;
    let dst_ptr = decrypt_fn
        .get_nth_param(3)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow::anyhow!("Failed to get source pointer parameter"))?;
    let xor_mask = decrypt_fn
        .get_nth_param(4)
        .ok_or_else(|| anyhow::anyhow!("missing per-string mask"))?
        .into_int_value();

    let builder = ctx.create_builder();
    builder.position_at_end(prepare);

    if has_flag
        && let Some(prepare_has_flags) = prepare_has_flags
        && let Some(spin_wait) = spin_wait
        && let Some(do_mark_done) = do_mark_done
    {
        // Check if flag_ptr is NULL (stack allocation case)
        let flag_is_null = builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            flag_ptr,
            flag_ptr.get_type().const_null(),
            "flag_is_null",
        )?;
        // If flag_ptr is NULL, go directly to decrypt (no synchronization needed)
        // If flag_ptr is not NULL, go to prepare_has_flags for synchronization
        builder.build_conditional_branch(flag_is_null, entry, prepare_has_flags)?;

        builder.position_at_end(prepare_has_flags);
        // Three-state protocol for thread safety:
        // 0 = not decrypted, 1 = decrypting (in progress), 2 = decrypted (complete)
        // Try to atomically change flag from 0 to 1 (claim the decryption task)
        let cmpxchg_result = builder.build_cmpxchg(
            flag_ptr,
            i32_ty.const_zero(),        // expected: 0 (not decrypted)
            i32_ty.const_int(1, false), // new value: 1 (decrypting)
            AtomicOrdering::AcquireRelease,
            AtomicOrdering::Acquire,
        )?;
        // Extract the success flag (second element of the result struct)
        let cmpxchg_success = builder
            .build_extract_value(cmpxchg_result, 1, "cmpxchg_success")?
            .into_int_value();
        // If cmpxchg succeeded (we are the winner), proceed to decrypt
        // If cmpxchg failed (someone else is decrypting or already done), go to spin_wait
        builder.build_conditional_branch(cmpxchg_success, entry, spin_wait)?;

        // Spin-wait loop: wait until flag becomes 2 (decryption complete)
        builder.position_at_end(spin_wait);
        let flag_val = builder.build_load2(i32_ty, flag_ptr, "flag_val")?.into_int_value();
        // Make the load atomic to ensure we see the latest value
        if let Some(load_inst) = flag_val.as_instruction_value() {
            load_inst.set_atomic_ordering(AtomicOrdering::Acquire)?;
        }
        let is_complete = builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            flag_val,
            i32_ty.const_int(2, false), // 2 = decrypted (complete)
            "is_complete",
        )?;
        // If complete, exit; otherwise keep spinning
        builder.build_conditional_branch(is_complete, exit, spin_wait)?;

        // Mark decryption as complete (flag = 2)
        // This block is only reached after successful decryption
        builder.position_at_end(do_mark_done);
        let store_inst = builder.build_store(flag_ptr, i32_ty.const_int(2, false))?;
        store_inst.set_atomic_ordering(AtomicOrdering::Release)?;
        builder.build_unconditional_branch(exit)?;
    } else {
        builder.build_unconditional_branch(entry)?;
    }

    builder.position_at_end(entry);
    let idx = builder.build_alloca(i32_ty, "idx")?;
    builder.build_store(idx, ctx.i32_type().const_zero())?;
    builder.build_unconditional_branch(body)?;

    builder.position_at_end(body);
    let index = builder.build_load2(i32_ty, idx, "cur_idx")?.into_int_value();
    let cond = builder.build_int_compare(inkwell::IntPredicate::ULT, index, len, "cond")?;
    // When done decrypting:
    // - If we came from prepare_has_flags path (flag_ptr not NULL), go to do_mark_done
    // - If we came from prepare path (flag_ptr NULL), go directly to exit
    // We detect this by checking if flag_ptr is NULL again
    let done_block = if has_flag && do_mark_done.is_some() {
        // Build a check at the end of decryption loop
        let check_block = ctx.append_basic_block(decrypt_fn, "check_mark_done");
        builder.build_conditional_branch(cond, next, check_block)?;

        builder.position_at_end(check_block);
        let flag_is_null = builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            flag_ptr,
            flag_ptr.get_type().const_null(),
            "flag_is_null_exit",
        )?;
        builder.build_conditional_branch(flag_is_null, exit, do_mark_done.unwrap())?;

        // Return a dummy - we've already built the branch
        None
    } else {
        Some(exit)
    };

    if let Some(done_block) = done_block {
        builder.build_conditional_branch(cond, next, done_block)?;
    }

    builder.position_at_end(next);

    // 从源地址读取
    let src_gep = builder.build_gep2(i8_ty, ptr, &[index], "src_gep")?;
    let ch = builder.build_load2(i8_ty, src_gep, "cur")?.into_int_value();
    if let Some(load_inst) = ch.as_instruction_value() {
        // Keep GlobalOpt from evaluating the decryptor back into plaintext.
        load_inst.set_volatile(true)?;
    }
    // 解密
    let xored = builder.build_xor(ch, xor_mask, "new")?;
    // 写入目标地址（栈上）
    let dst_gep = builder.build_gep2(i8_ty, dst_ptr, &[index], "dst_gep")?;
    let store_inst = builder.build_store(dst_gep, xored)?;
    store_inst.set_volatile(true)?;

    let next_index = builder.build_int_add(index, ctx.i32_type().const_int(1, false), "")?;
    builder.build_store(idx, next_index)?;
    builder.build_unconditional_branch(body)?;

    builder.position_at_end(exit);
    builder.build_return(None)?;

    Ok(decrypt_fn)
}
