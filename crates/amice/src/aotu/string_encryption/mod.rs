mod simd_xor;
mod xor;

use crate::aotu::string_encryption::simd_xor::SimdXorAlgo;
use crate::aotu::string_encryption::xor::XorAlgo;
use crate::config::{Config, StringAlgorithm, StringDecryptTiming, StringEncryptionConfig};
use crate::pass_registry::{AmicePass, AmicePassFlag};
use amice_llvm::inkwell2::{BasicBlockExt, FunctionExt, LLVMValueRefExt, VerifyResult};
use amice_macro::amice;
use amice_plugin::inkwell::llvm_sys::prelude::LLVMValueRef;
use amice_plugin::inkwell::module::Module;
use amice_plugin::inkwell::values::{
    AnyValueEnum, ArrayValue, AsValueRef, BasicValue, BasicValueEnum, GlobalValue, InstructionOpcode, InstructionValue,
    PointerValue, StructValue,
};
use amice_plugin::{LlvmModulePass, ModuleAnalysisManager, PreservedAnalyses, inkwell};
use inkwell::llvm_sys::core::{
    LLVMGetAsString, LLVMGetFirstUse, LLVMGetNextUse, LLVMGetUser, LLVMIsAConstantArray, LLVMIsAConstantExpr,
    LLVMIsAConstantStruct, LLVMIsAGlobalVariable, LLVMIsAInstruction, LLVMIsGlobalConstant, LLVMIsThreadLocal,
};
use std::ptr::NonNull;

/// Stack allocation threshold: strings larger than this will use global timing
/// even when stack allocation is enabled
const STACK_ALLOC_THRESHOLD: u32 = 4096; // 4KB
const MAX_STRING_BYTES: usize = 65_536;

#[amice(
    priority = 1000,
    name = "StringEncryption",
    flag = AmicePassFlag::PipelineStart | AmicePassFlag::ModuleLevel,
    config = StringEncryptionConfig,
)]
#[derive(Default)]
pub struct StringEncryption {}

impl AmicePass for StringEncryption {
    fn init(&mut self, cfg: &Config, _flag: AmicePassFlag) {
        self.default_config = cfg.string_encryption.clone();

        assert!(
            (self.default_config.timing == StringDecryptTiming::Global && !self.default_config.stack_alloc)
                || self.default_config.timing != StringDecryptTiming::Global,
            "stack alloc is not supported with global decrypt timing: {:?}",
            self.default_config.timing
        );
    }

    fn do_pass(&self, module: &mut Module<'_>) -> anyhow::Result<PreservedAnalyses> {
        if !self.default_config.enable {
            return Ok(PreservedAnalyses::All);
        }
        if self.default_config.stack_alloc {
            anyhow::bail!("string stack allocation cannot preserve Rust static pointer lifetimes");
        }

        let mut algo: Box<dyn StringEncryptionAlgo> = match self.default_config.algorithm {
            StringAlgorithm::Xor => Box::new(XorAlgo::default()),
            StringAlgorithm::SimdXor => Box::new(SimdXorAlgo::default()),
        };

        if let Err(err) = algo.initialize(&self.default_config, module) {
            error!("initialize failed: {}", err);
            return Ok(PreservedAnalyses::All);
        }

        if let Err(err) = algo.do_string_encrypt(&self.default_config, module) {
            error!("do_string_encrypt failed: {}", err);
            return Ok(PreservedAnalyses::All);
        }

        clear_string_effect_attributes(module);

        for x in module.get_functions() {
            if let VerifyResult::Broken(err) = x.verify_function() {
                error!("function {:?} verify failed: {}", x.get_name(), err);
            }
        }

        Ok(PreservedAnalyses::None)
    }
}

pub(super) trait StringEncryptionAlgo {
    fn initialize(&mut self, cfg: &StringEncryptionConfig, module: &mut Module<'_>) -> anyhow::Result<()>;

    fn do_string_encrypt(&mut self, cfg: &StringEncryptionConfig, module: &mut Module<'_>) -> anyhow::Result<()>;
}

#[derive(Copy, Clone)]
struct EncryptedGlobalValue<'a> {
    global: GlobalValue<'a>,
    str_len: u32,
    xor_mask: u8,
    flag: Option<GlobalValue<'a>>,
    #[allow(dead_code)]
    oneshot: bool,
    /// Whether this specific string should use stack allocation for decryption
    /// This can be false even when overall stack_alloc is true, for strings > 4KB
    use_stack_alloc: bool,
    users: NonNull<Vec<(InstructionValue<'a>, u32)>>,

    /// Rust Struct String or C++ NTTP String
    pub struct_value: Option<StructValue<'a>>,
    pub field_index: Option<u32>,
}

impl<'a> EncryptedGlobalValue<'a> {
    pub fn new_array_string(
        global: GlobalValue<'a>,
        len: u32,
        flag: Option<GlobalValue<'a>>,
        use_stack_alloc: bool,
        user: Vec<(LLVMValueRef, u32)>,
    ) -> Self {
        let user = Box::new(
            user.iter()
                .map(|(value_ref, op_num)| (value_ref.into_instruction_value(), *op_num))
                .collect::<Vec<_>>(),
        );
        EncryptedGlobalValue {
            global,
            str_len: len,
            xor_mask: 0,
            flag,
            oneshot: false,
            use_stack_alloc,
            users: NonNull::new(Box::leak(user)).unwrap(),
            struct_value: None,
            field_index: None,
        }
    }

    pub fn new_struct_string(
        global: GlobalValue<'a>,
        len: u32,
        flag: Option<GlobalValue<'a>>,
        use_stack_alloc: bool,
        user: Vec<(LLVMValueRef, u32)>,
        struct_value: Option<StructValue<'a>>,
        field_index: Option<u32>,
    ) -> Self {
        let user = Box::new(
            user.iter()
                .map(|(value_ref, op_num)| (value_ref.into_instruction_value(), *op_num))
                .collect::<Vec<_>>(),
        );
        assert!(struct_value.is_some());
        assert!(field_index.is_some());
        EncryptedGlobalValue {
            global,
            str_len: len,
            xor_mask: 0,
            flag,
            oneshot: false,
            use_stack_alloc,
            users: NonNull::new(Box::leak(user)).unwrap(),
            struct_value,
            field_index,
        }
    }

    #[allow(dead_code)]
    pub fn push(&self, user: InstructionValue<'a>, op_num: u32) {
        unsafe {
            let _ = &(*self.users.as_ptr()).push((user, op_num));
        }
    }

    pub fn is_array_string(&self) -> bool {
        self.struct_value.is_none() && self.field_index.is_none()
    }

    pub fn is_struct_string(&self) -> bool {
        self.struct_value.is_some() && self.field_index.is_some()
    }

    pub fn user_slice(&self) -> &[(InstructionValue<'a>, u32)] {
        unsafe { (*self.users.as_ptr()).as_slice() }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        unsafe { (*self.users.as_ptr()).len() }
    }

    pub fn free(&self) {
        unsafe {
            let _ = Box::from_raw(self.users.as_ptr());
        }
    }
}

#[allow(dead_code)]
pub(crate) fn array_as_rust_string(arr: &ArrayValue) -> Option<String> {
    let str = array_as_const_string(arr)?;
    String::from_utf8(str.to_vec()).ok()
}

#[inline]
pub(crate) fn array_as_const_string<'a>(arr: &'a ArrayValue) -> Option<&'a [u8]> {
    let mut len = 0;
    if arr.is_null() {
        return None;
    }
    let ptr = unsafe { LLVMGetAsString(arr.as_value_ref() as _, &mut len) };
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(std::slice::from_raw_parts(ptr.cast(), len)) }
    }
}

struct StringCandidate<'a> {
    global: GlobalValue<'a>,
    bytes: Vec<u8>,
    users: Vec<(LLVMValueRef, u32)>,
}

fn local_data_global(global: GlobalValue<'_>) -> bool {
    use inkwell::module::Linkage;
    matches!(global.get_linkage(), Linkage::Internal | Linkage::Private)
        && unsafe { LLVMIsThreadLocal(global.as_value_ref()) == 0 }
        && !global.get_name().to_bytes().starts_with(b"llvm.")
}

fn prepare_string_insertion(inst: InstructionValue<'_>) -> anyhow::Result<InstructionValue<'_>> {
    use inkwell::attributes::{Attribute, AttributeLoc};
    let block = inst
        .get_parent()
        .ok_or_else(|| anyhow::anyhow!("detached string user"))?;
    let function = block
        .get_parent()
        .ok_or_else(|| anyhow::anyhow!("string user without function"))?;
    // rustc may have inferred read-only/no-sync behavior before this pass adds
    // atomic initialization and shared-buffer writes.
    for name in ["memory", "readnone", "readonly", "nosync", "speculatable"] {
        let kind = Attribute::get_named_enum_kind_id(name);
        if kind != 0 {
            function.remove_enum_attribute(AttributeLoc::Function, kind);
        }
    }
    Ok(if inst.get_opcode() == InstructionOpcode::Phi {
        block.get_first_insertion_pt()
    } else {
        inst
    })
}

fn clear_string_effect_attributes(module: &Module<'_>) {
    use inkwell::attributes::{Attribute, AttributeLoc};
    use inkwell::llvm_sys::core::LLVMRemoveCallSiteEnumAttribute;
    for function in module
        .get_functions()
        .filter(|function| function.count_basic_blocks() != 0)
    {
        for name in ["memory", "readnone", "readonly", "nosync", "speculatable"] {
            let kind = Attribute::get_named_enum_kind_id(name);
            if kind == 0 {
                continue;
            }
            function.remove_enum_attribute(AttributeLoc::Function, kind);
            for block in function.get_basic_blocks() {
                for inst in block.get_instructions() {
                    if matches!(
                        inst.get_opcode(),
                        InstructionOpcode::Call | InstructionOpcode::Invoke | InstructionOpcode::CallBr
                    ) {
                        unsafe {
                            LLVMRemoveCallSiteEnumAttribute(inst.as_value_ref(), u32::MAX, kind);
                        }
                    }
                }
            }
        }
    }
}

fn has_unsupported_string_control_flow(function: inkwell::values::FunctionValue<'_>) -> bool {
    for block in function.get_basic_blocks() {
        for inst in block.get_instructions() {
            if matches!(
                inst.get_opcode(),
                InstructionOpcode::Invoke
                    | InstructionOpcode::LandingPad
                    | InstructionOpcode::CatchPad
                    | InstructionOpcode::CleanupPad
                    | InstructionOpcode::CatchSwitch
                    | InstructionOpcode::CatchRet
                    | InstructionOpcode::CleanupRet
                    | InstructionOpcode::Resume
                    | InstructionOpcode::CallBr
            ) {
                return true;
            }
            if inst.get_opcode() == InstructionOpcode::Call
                && unsafe { inkwell::llvm_sys::core::LLVMGetTailCallKind(inst.as_value_ref()) }
                    == inkwell::llvm_sys::LLVMTailCallKind::LLVMTailCallKindMustTail
            {
                return true;
            }
        }
    }
    false
}

/// Binary payloads have real backing arrays in the reserved section. A Rust
/// fat-pointer's inline byte fields are never treated as standalone strings.
fn collect_string_candidates<'a>(module: &Module<'a>, cfg: &StringEncryptionConfig) -> Vec<StringCandidate<'a>> {
    let mut candidates = Vec::new();
    for global in module.get_globals() {
        if !local_data_global(global) || unsafe { LLVMIsGlobalConstant(global.as_value_ref()) == 0 } {
            continue;
        }
        if let Some(section) = global.get_section() {
            // Respect explicit storage sections, including .delta.no_obfuscate
            // and read-only/custom sections that cannot support writeback.
            if !section.to_bytes().is_empty() {
                info!("(strenc) skipped {:?}: explicit data section", global.get_name());
                continue;
            }
        }
        if cfg.only_dot_str && !global.get_name().to_str().is_ok_and(|name| name.contains(".str")) {
            continue;
        }
        let Some(BasicValueEnum::ArrayValue(array)) = global.get_initializer() else {
            continue;
        };
        if !array.is_const_string() || array.is_undef() || array.is_null() {
            continue;
        }
        let Some(bytes) = array_as_const_string(&array) else {
            continue;
        };
        let text = bytes.strip_suffix(&[0]).unwrap_or(bytes);
        if bytes.len() <= 1
            || bytes.len() > MAX_STRING_BYTES
            || std::str::from_utf8(text).is_err()
            || text
                .iter()
                .any(|byte| byte.is_ascii_control() && !matches!(byte, b'\n' | b'\r' | b'\t'))
        {
            info!("(strenc) skipped {:?}: not a supported text array", global.get_name());
            continue;
        }
        let mut users = Vec::new();
        if cfg.timing == StringDecryptTiming::Lazy {
            let mut use_value = global.get_first_use();
            let mut complete = true;
            while let Some(current) = use_value {
                use_value = current.get_next_use();
                if collect_insert_points(global, current.get_user(), &mut users).is_err() {
                    complete = false;
                    break;
                }
            }
            if !complete || users.is_empty() {
                info!(
                    "(strenc) skipped {:?}: unsupported or unguarded use chain",
                    global.get_name()
                );
                continue;
            }
            users.sort_unstable_by_key(|(value, operand)| (*value as usize, *operand));
            users.dedup();
        }
        candidates.push(StringCandidate {
            global,
            bytes: bytes.to_vec(),
            users,
        });
    }
    candidates
}

pub(crate) fn collect_insert_points<'a>(
    _string_global: GlobalValue,
    user: AnyValueEnum<'a>,
    output: &mut Vec<(LLVMValueRef, u32)>,
) -> anyhow::Result<()> {
    use inkwell::llvm_sys::LLVMLinkage;
    use inkwell::llvm_sys::core::{LLVMGetLinkage, LLVMGetSection};

    let mut visited = std::collections::HashSet::new();
    let mut worklist = vec![user.as_value_ref()];
    while let Some(value) = worklist.pop() {
        if !visited.insert(value) {
            continue;
        }
        // Guard the first runtime instruction on each constant-use path.
        // Guarding a load of a Rust slice also covers escaped/returned pointers.
        if !unsafe { LLVMIsAInstruction(value) }.is_null() {
            let instruction = value.into_instruction_value();
            if matches!(
                instruction.get_opcode(),
                InstructionOpcode::LandingPad | InstructionOpcode::CatchPad | InstructionOpcode::CleanupPad
            ) {
                anyhow::bail!("exception-pad string use cannot be guarded");
            }
            let block = instruction
                .get_parent()
                .ok_or_else(|| anyhow::anyhow!("detached string user"))?;
            let function = block
                .get_parent()
                .ok_or_else(|| anyhow::anyhow!("string user has no parent function"))?;
            if has_unsupported_string_control_flow(function) {
                anyhow::bail!("string use in EH/funclet/musttail function cannot be guarded");
            }
            output.push((value, 0));
            continue;
        }
        if !unsafe { LLVMIsAGlobalVariable(value) }.is_null() {
            let linkage = unsafe { LLVMGetLinkage(value) };
            let section = unsafe { LLVMGetSection(value) };
            let has_section = !section.is_null() && !unsafe { std::ffi::CStr::from_ptr(section) }.to_bytes().is_empty();
            if !matches!(
                linkage,
                LLVMLinkage::LLVMInternalLinkage | LLVMLinkage::LLVMPrivateLinkage
            ) || unsafe { LLVMIsThreadLocal(value) != 0 }
                || has_section
            {
                anyhow::bail!("string escapes through an external, TLS, or sectioned global");
            }
        } else if unsafe {
            LLVMIsAConstantExpr(value).is_null()
                && LLVMIsAConstantStruct(value).is_null()
                && LLVMIsAConstantArray(value).is_null()
        } {
            anyhow::bail!("unsupported constant string user");
        }
        let mut next_use = unsafe { LLVMGetFirstUse(value) };
        if next_use.is_null() {
            anyhow::bail!("string constant chain has no guarded runtime use");
        }
        while !next_use.is_null() {
            worklist.push(unsafe { LLVMGetUser(next_use) });
            next_use = unsafe { LLVMGetNextUse(next_use) };
        }
    }
    Ok(())
}
fn alloc_stack_string<'a>(
    module: &mut Module<'a>,
    string: EncryptedGlobalValue,
    in_entry_block: bool,
    inst: &InstructionValue,
) -> anyhow::Result<PointerValue<'a>> {
    let ctx = module.get_context();
    let i32_ty = ctx.i32_type();
    let i8_ty = ctx.i8_type();
    let string_len = i32_ty.const_int(string.str_len as u64 + 1, false);

    let builder = ctx.create_builder();
    if !in_entry_block {
        // 在非入口块分配，许多LLVM优化pass假设所有 alloca 都在入口块
        // 可能阻止某些优化的进行
        // 寄存器提升等优化可能受影响

        // 如果指令是PHI节点，需要在基本块的第一个非PHI指令位置分配
        let insert_point = if inst.get_opcode() == InstructionOpcode::Phi {
            if let Some(parent_bb) = inst.get_parent() {
                parent_bb.get_first_insertion_pt()
            } else {
                *inst
            }
        } else {
            *inst
        };

        builder.position_before(&insert_point);
        let container = builder.build_array_alloca(i8_ty, string_len, "string_container")?;
        return Ok(container);
    }

    if in_entry_block
        && let Some(parent_block) = inst.get_parent()
        && let Some(parent_function) = parent_block.get_parent()
        && let Some(entry_block) = parent_function.get_entry_block()
        && let Some(terminator) = entry_block.get_terminator()
    {
        builder.position_before(&terminator);
        let container = builder.build_array_alloca(i8_ty, string_len, "string_container")?;
        Ok(container)
    } else {
        // 尝试栈入口块分配栈空间失败！
        Err(anyhow::anyhow!("Failed to allocate stack string"))
    }
}
