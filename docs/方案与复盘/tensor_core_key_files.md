Tensor Core 支持：关键文件修改详解
=================================

本文把实现 Prajna 调用 NVIDIA Tensor Core（WMMA）过程中，几处关键 C++ 源码的修改点、动机与其和 WMMA 的关系写清楚，便于后续整理为设计文档/论文材料。

涉及文件：
- `prajna/prajna/ir/type.hpp`
- `prajna/prajna/codegen/llvm_codegen.cpp`
- `prajna/prajna/lowering/statement_lowering_visitor.hpp`
- `prajna/prajna/lowering/expression_lowering_visitor.hpp`
- `prajna/prajna/transform/transform.h`

---

1) `prajna/prajna/ir/type.hpp`
-----------------------------

### 改了什么
1. `PointerType` 增加 `address_space` 字段，并扩展工厂函数：
   - `PointerType::Create(value_type, address_space = 0)`
   - 类型缓存/去重时把 `address_space` 也纳入比较键
   - `name/fullname` 在非 0 addrspace 时追加 `*@<n>`（例如 `f16*@3`）

2. `StructType` 增加 `is_literal` 标志：
   - `is_literal == true` 表示该 struct 在 LLVM 中要生成为匿名 literal struct（而非命名 struct）

### 为什么要改
- **addrspace**：GPU 相关 intrinsic/ABI 经常要求指针的地址空间匹配。即使当前 WMMA minimal 用例走 `.p0`，addrspace 仍是后续扩展 shared `.p3`、ldmatrix 等的基础设施。
- **literal struct**：NVVM/LLVM 的 WMMA 片段类型在 IR 里往往表现为“匿名 struct 聚合”。LLVM 的类型系统中“命名 struct”和“匿名 literal struct”不是同一类型，即使字段完全一致也会被判为不匹配；要想让内建匹配，需要能选择性生成 literal struct。

### 和 WMMA/Tensor Core 的关系
- WMMA 内建签名极其严格，片段类型必须和 LLVM/NVVM 期望一致；`is_literal` 是让 fragment 在 LLVM 层长成正确形状的关键。
- addrspace 是后续把 WMMA 从 global 版本扩展到 shared/ldmatrix 的关键基础。

---

2) `prajna/prajna/codegen/llvm_codegen.cpp`
-----------------------------------------

### 改了什么
1. 指针类型 codegen 不再硬编码 addrspace=0：
   - 原：`llvm::PointerType::get(T, 0)`
   - 现：`llvm::PointerType::get(T, ir_pointer_type->address_space)`

2. struct codegen 支持 `is_literal`，并修复递归/自引用类型的生成方式：
   - 命名 struct：先 `llvm::StructType::create(ctx, name)` 建占位，再 Emit 字段类型，最后 `setBody(...)`
   - literal struct：直接 `llvm::StructType::get(ctx, element_types, false)` 生成匿名 struct

### 为什么要改
- **addrspace**：IR 里记录的地址空间必须落实到 LLVM 类型，否则 IR 层信息会丢失。
- **自引用递归崩溃修复**：如果 struct 的字段中间接引用自身（常见是字段为 `ptr<ThisStruct>`），不先创建 LLVM 占位类型会导致 `EmitType` 无限递归并崩溃。该修复保证编译链能跑到 codegen/PTX。
- **literal struct**：NVVM 内建常用匿名聚合描述片段，必须能生成 literal struct 才能匹配签名。

### 和 WMMA/Tensor Core 的关系
- 只有 LLVM 类型正确，NVVM 内建才能匹配，NVPTX 后端才会把 `llvm.nvvm.wmma.*` 降成 PTX 的 `wmma.load/mma/store` 指令（进而使用 Tensor Core）。
- 递归崩溃修复是“能走到 PTX 生成/执行验证”的基础保障。

---

3) `prajna/prajna/lowering/statement_lowering_visitor.hpp`
--------------------------------------------------------

### 改了什么
1. 新增内建模板 `vec`（与 `array/ptr` 一样的 builtin template generator）：
   - 增加 `CreateRawVectorTypeTemplate()`
   - 在 `Stage1()` 注册 `vec`
   - 语义：`vec<Len, Elem>` → `ir::VectorType::Create(Elem, Len)`（Len 为正的常量 int）

2. 标记 WMMA fragment 为 literal struct（按名字匹配）：
   - 在 struct lowering 处对 `WmmaFragA/WmmaFragB/WmmaFragAcc` 设置 `ir_struct_type->is_literal = true`

### 为什么要改
- **vec 走模板机制**：Prajna 的设计里 `array<T,N>`、`ptr<T>` 都是模板 generator 生成 IR 类型。把 `vec` 也做成模板 generator，可保持语言语义一致，减少语法/AST 特例，利于后续扩展（复杂元素类型、嵌套 vec 等）。
- **fragment literal**：否则 fragment 会变成 LLVM 命名 struct，即便字段一致也匹配不上 NVVM 期望的匿名聚合，导致 verifier/内建匹配失败。

### 和 WMMA/Tensor Core 的关系
- WMMA 片段字段必须是 `<2 x half>` 这样的 LLVM 向量类型，`vec` 模板负责构造该向量类型。
- `is_literal` 确保片段类型和 NVVM 内建声明一致，从而成功降到 Tensor Core 指令。

---

4) `prajna/prajna/lowering/expression_lowering_visitor.hpp`
---------------------------------------------------------

### 改了什么
1. 类型解析回到统一路径：
   - `ApplyType(ast::Type)` 直接把 `Type` 当作 `IdentifierPath` 走符号解析/模板实例化

2. 模板实参解析统一：
   - `ApplyTemplateArguments` 解析 `IdentifierPath` 或 `IntLiteral`
   - `vec<2,f16>` 的模板参数 `(2, f16)` 能正确交给 `vec` 的 generator

3. 清理诊断期加入的刷屏调试输出：
   - 移除 `[dbg] ApplyIdentifierPath ...` 大量 stderr 打印

### 为什么要改
- 选择了“vec 走模板 generator”的路线后，必须保证：
  - `IdentifierPath` 能解析到 `vec` 这个模板实体
  - 模板参数能以统一方式传入模板实例化
- 调试输出长期保留会影响所有测试与使用体验。

### 和 WMMA/Tensor Core 的关系
- WMMA 绑定依赖 `vec<2,f16>` 类型，类型解析/模板实例化稳定是 WMMA 片段能被正确构造的前提。

---

5) `prajna/prajna/transform/transform.h`
---------------------------------------

### 改了什么
- 多处遍历 `instruction_with_index_list`（weak_ptr 列表）时增加 `expired()` 检查，避免访问已销毁指令：
  - `if (wp.expired()) continue;`
- 一些场景下改为返回 `nullptr` 而不是 assert/崩溃（例如找不到写入点、OperandSize=0）

### 为什么要改
- transform 阶段会 clone/替换 IR 指令引用，`instruction_with_index_list` 里弱引用可能过期；未保护时会导致编译过程崩溃。
- 这些崩溃会让流程走不到 LLVM codegen/PTX 生成，即使 WMMA 签名已经正确也无法验证。

### 和 WMMA/Tensor Core 的关系
- 这是保证“端到端链路可跑”的稳定性修复：没有它，你可能无法走到 NVPTX codegen，更不可能看到 PTX 里的 `wmma.*` 指令并让测试通过。

