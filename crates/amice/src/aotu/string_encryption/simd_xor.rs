use crate::aotu::string_encryption::{EncryptedGlobalValue, StringEncryptionAlgo};
use crate::config::{StringDecryptTiming as DecryptTiming, StringEncryptionConfig};
use amice_llvm::const_array;
use amice_llvm::inkwell2::{BuilderExt, ModuleExt};
use amice_llvm::ptr_type;
use amice_plugin::inkwell;
use amice_plugin::inkwell::AddressSpace;
use amice_plugin::inkwell::AtomicOrdering;
use amice_plugin::inkwell::attributes::{Attribute, AttributeLoc};
use amice_plugin::inkwell::module::{Linkage, Module};
use amice_plugin::inkwell::values::{BasicValue, FunctionValue, GlobalValue};
use anyhow::anyhow;
use log::{Level, debug, log_enabled};
use rand::Rng;

#[derive(Default)]
pub(super) struct SimdXorAlgo {
    pub(super) key: [u8; 32],
}

impl StringEncryptionAlgo for SimdXorAlgo {
    fn initialize(&mut self, _cfg: &StringEncryptionConfig, _module: &mut Module<'_>) -> anyhow::Result<()> {
        rand::rng().fill(&mut self.key);
        Ok(())
    }

    fn do_string_encrypt(&mut self, cfg: &StringEncryptionConfig, module: &mut Module<'_>) -> anyhow::Result<()> {
        do_handle(cfg, module, &self.key)
    }
}

fn do_handle<'a>(cfg: &StringEncryptionConfig, module: &mut Module<'a>, key: &[u8; 32]) -> anyhow::Result<()> {
    let candidates = super::collect_string_candidates(module, cfg);
    let ctx = module.get_context();
    let i32_ty = ctx.i32_type();
    let is_lazy_mode = matches!(cfg.timing, DecryptTiming::Lazy);
    let i8_ty = ctx.i8_type();
    let global_key = module.add_global(i8_ty.array_type(32), None, ".amice_string_key");
    global_key.set_linkage(Linkage::Private);
    global_key.set_alignment(32);
    global_key.set_initializer(&const_array(
        i8_ty,
        &key.map(|byte| i8_ty.const_int(byte as u64, false)),
    ));

    let mut strings = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let global = candidate.global;
        let name = global.get_name().to_string_lossy();
        let encoded: Vec<_> = candidate
            .bytes
            .iter()
            .enumerate()
            .map(|(index, byte)| byte ^ key[index % 32])
            .collect();
        let flag = is_lazy_mode.then(|| {
            let flag = module.add_global(i32_ty, None, &format!("dec_flag_{name}"));
            flag.set_initializer(&i32_ty.const_zero());
            flag.set_linkage(Linkage::Private);
            flag
        });
        global.set_initializer(&ctx.const_string(&encoded, false));
        global.set_constant(false);
        let string = EncryptedGlobalValue::new_array_string(global, encoded.len() as u32, flag, false, candidate.users);
        strings.push(string);
    }
    let decrypt_fn = add_decrypt_function(
        module,
        &format!("simd_xor_cipher_{}", rand::random::<u32>()),
        is_lazy_mode,
        cfg.inline_decrypt,
        false,
    )?;
    match cfg.timing {
        DecryptTiming::Lazy => {
            emit_decrypt_before_inst(module, strings.clone(), decrypt_fn, false, false, global_key)?;
        },
        DecryptTiming::Global => emit_global_string_decryptor_ctor(module, &strings, decrypt_fn, global_key)?,
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
    global_key: GlobalValue<'a>,
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
                    ptr.into(),
                    len.into(),
                    global_key.as_pointer_value().into(),
                    flag.as_pointer_value().into(),
                ],
                "",
            )?;
        }
    }
    Ok(())
}
fn emit_global_string_decryptor_ctor<'a>(
    module: &mut Module<'a>,
    gs: &Vec<EncryptedGlobalValue<'a>>,
    decrypt_fn: FunctionValue<'a>,
    global_key: GlobalValue,
) -> anyhow::Result<()> {
    let ctx = module.get_context();
    let i32_ty = ctx.i32_type();
    let i32_ptr = ptr_type!(ctx, i32_type);

    let decrypt_stub_ty = ctx.void_type().fn_type(&[], false);
    let decrypt_stub = module.add_function("simd_xor_decrypt_stub", decrypt_stub_ty, None);
    decrypt_stub.set_linkage(Linkage::Internal);

    let entry = ctx.append_basic_block(decrypt_stub, "entry");
    let builder = ctx.create_builder();

    builder.position_at_end(entry);
    for string in gs {
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

        let dst = ptr; // In-place decryption: src == dst
        let len_val = i32_ty.const_int(string.str_len as u64, false);
        let key_ptr = global_key.as_pointer_value();
        let flag_ptr = i32_ptr.const_null(); // No flag for global timing
        builder.build_call(
            decrypt_fn,
            &[ptr.into(), dst.into(), len_val.into(), key_ptr.into(), flag_ptr.into()],
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
    _stack_alloc: bool,
) -> anyhow::Result<FunctionValue<'a>> {
    // 密钥总是32字节的！必须是32字节的！
    // void @simd_xor_cipher(i8* src, i8* dst, i32 len, i8* key, i32* flag)
    let ctx = module.get_context();
    let i8_ty = ctx.i8_type();
    let i32_ty = ctx.i32_type();
    let i8_ptr = ptr_type!(ctx, i8_type);
    let i32_ptr = ptr_type!(ctx, i32_type);
    let vector_256 = i8_ty.vec_type(32);

    let fn_ty = i8_ptr.fn_type(
        &[
            i8_ptr.into(),
            i8_ptr.into(),
            i32_ty.into(),
            i8_ptr.into(),
            i32_ptr.into(),
        ],
        false,
    );
    let decrypt_fn = module.add_function(name, fn_ty, None);
    decrypt_fn.set_linkage(Linkage::Internal);

    if inline_fn {
        let inlinehint_attr = ctx.create_enum_attribute(Attribute::get_named_enum_kind_id("alwaysinline"), 0);
        decrypt_fn.add_attribute(AttributeLoc::Function, inlinehint_attr);
    }

    let entry = ctx.append_basic_block(decrypt_fn, "entry");
    let entry_has_flags = ctx.append_basic_block(decrypt_fn, "entry_has_flags");
    let spin_wait = ctx.append_basic_block(decrypt_fn, "spin_wait");
    let do_mark_done = ctx.append_basic_block(decrypt_fn, "do_mark_done");
    let main_loop = ctx.append_basic_block(decrypt_fn, "main_loop");
    let key_prepare = ctx.append_basic_block(decrypt_fn, "key_prepare");
    let next = ctx.append_basic_block(decrypt_fn, "next");
    let check_rest = ctx.append_basic_block(decrypt_fn, "check_rest");
    let rest = ctx.append_basic_block(decrypt_fn, "rest");
    let exit = ctx.append_basic_block(decrypt_fn, "exit");

    let builder = ctx.create_builder();

    let src_ptr = decrypt_fn
        .get_nth_param(0)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow!("Failed to get source pointer parameter"))?;
    let dst_ptr = decrypt_fn
        .get_nth_param(1)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow!("Failed to get destination pointer parameter"))?;
    let len = decrypt_fn
        .get_nth_param(2)
        .map(|param| param.into_int_value())
        .ok_or_else(|| anyhow!("Failed to get length parameter"))?;
    let key_ptr = decrypt_fn
        .get_nth_param(3)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow!("Failed to get key pointer parameter"))?;
    let flag_ptr = decrypt_fn
        .get_nth_param(4)
        .map(|param| param.into_pointer_value())
        .ok_or_else(|| anyhow!("Failed to get flag pointer parameter"))?;

    builder.position_at_end(entry);
    let idx = builder.build_alloca(i32_ty, "idx")?;
    builder.build_store(idx, ctx.i32_type().const_zero())?;

    if has_flag {
        // Check if flag_ptr is NULL (stack allocation case)
        let flag_is_null = builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            flag_ptr,
            flag_ptr.get_type().const_null(),
            "flag_is_null",
        )?;
        // If flag_ptr is NULL, go directly to decrypt (no synchronization needed)
        // If flag_ptr is not NULL, go to entry_has_flags for synchronization
        builder.build_conditional_branch(flag_is_null, key_prepare, entry_has_flags)?;

        builder.position_at_end(entry_has_flags);
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
        builder.build_conditional_branch(cmpxchg_success, key_prepare, spin_wait)?;

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
        // This block is only reached after successful decryption when flag_ptr is not NULL
        builder.position_at_end(do_mark_done);
        let store_inst = builder.build_store(flag_ptr, i32_ty.const_int(2, false))?;
        store_inst.set_atomic_ordering(AtomicOrdering::Release)?;
        builder.build_unconditional_branch(exit)?;
    } else {
        builder.build_unconditional_branch(key_prepare)?;
        // global timing 不使用同步 flag，但这些预建 block 仍属于函数，必须有 terminator。
        builder.position_at_end(entry_has_flags);
        builder.build_unconditional_branch(key_prepare)?;
        builder.position_at_end(spin_wait);
        builder.build_unconditional_branch(exit)?;
        builder.position_at_end(do_mark_done);
        builder.build_unconditional_branch(exit)?;
    }

    builder.position_at_end(key_prepare);
    let key_load_inst = builder.build_load2(vector_256, key_ptr, "key_vec")?;
    let key_vec = key_load_inst.into_vector_value();
    builder.build_unconditional_branch(main_loop)?;

    // 检查是否还有完整的32字节块
    builder.position_at_end(main_loop);
    ();
    let index = builder.build_load2(i32_ty, idx, "cur_idx")?.into_int_value();
    let tmp = builder.build_int_add(index, ctx.i32_type().const_int(31, false), "tmp")?;
    let cond = builder.build_int_compare(inkwell::IntPredicate::ULT, tmp, len, "cond")?;
    builder.build_conditional_branch(cond, next, check_rest)?;

    builder.position_at_end(next);
    let src_gep = builder.build_gep2(i8_ty, src_ptr, &[index], "src_gep")?;
    let src_load_inst = builder.build_load2(vector_256, src_gep, "src_vec")?;
    if let Some(load_inst) = src_load_inst.as_instruction_value() {
        load_inst
            .set_alignment(1)
            .map_err(|e| anyhow!("Failed to set alignment for load instruction: {}", e))?;
    }
    let src_vec = src_load_inst.into_vector_value();
    let xored_vec = builder.build_xor(src_vec, key_vec, "xored_vec")?;
    let dst_gep = builder.build_gep2(i8_ty, dst_ptr, &[index], "dst_gep")?;
    let store_inst = builder.build_store(dst_gep, xored_vec)?;
    store_inst
        .set_alignment(1)
        .map_err(|e| anyhow!("Failed to set alignment for store instruction: {}", e))?;

    let next_index = builder.build_int_add(index, ctx.i32_type().const_int(32, false), "next_index")?;
    builder.build_store(idx, next_index)?;
    builder.build_unconditional_branch(main_loop)?;

    builder.position_at_end(check_rest);
    // 查询是否有剩余的字节没有处理
    let index = builder.build_load2(i32_ty, idx, "")?.into_int_value();
    let cond = builder.build_int_compare(inkwell::IntPredicate::ULT, index, len, "cond2")?;
    // When done decrypting, need to check if we should mark done or exit directly
    if has_flag {
        let check_mark_done = ctx.append_basic_block(decrypt_fn, "check_mark_done");
        builder.build_conditional_branch(cond, rest, check_mark_done)?;

        builder.position_at_end(check_mark_done);
        let flag_is_null = builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            flag_ptr,
            flag_ptr.get_type().const_null(),
            "flag_is_null_exit",
        )?;
        builder.build_conditional_branch(flag_is_null, exit, do_mark_done)?;
    } else {
        builder.build_conditional_branch(cond, rest, exit)?;
    }

    builder.position_at_end(rest);
    // 处理剩余的字节
    // for(;i<len;i++) output[i] = input[i] ^ key[i % 32];

    let src_gep = builder.build_gep2(i8_ty, src_ptr, &[index], "src_rest_gep")?;
    let ch = builder.build_load2(i8_ty, src_gep, "cur")?.into_int_value();
    let key_index = builder.build_int_signed_rem(index, ctx.i32_type().const_int(32, false), "key_index")?;
    let key_gep = builder.build_gep2(i8_ty, key_ptr, &[key_index], "key_rest_gep")?;
    let key_ch = builder.build_load2(i8_ty, key_gep, "key_cur")?.into_int_value();
    let xored = builder.build_xor(ch, key_ch, "xored_rest")?;
    let dst_gep = builder.build_gep2(i8_ty, dst_ptr, &[index], "dst_rest_gep")?;
    builder.build_store(dst_gep, xored)?;

    let next_index = builder.build_int_add(index, ctx.i32_type().const_int(1, false), "next_index_rest")?;
    builder.build_store(idx, next_index)?;
    builder.build_unconditional_branch(check_rest)?;

    builder.position_at_end(exit);
    builder.build_return(Some(&dst_ptr))?;

    Ok(decrypt_fn)
}
