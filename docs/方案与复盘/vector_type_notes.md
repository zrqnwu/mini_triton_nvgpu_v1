## 向量类型支持变更记录（vec<Len, Elem>）

位置：`prajna/`

### 改动内容
- **AST**：在 `ast/ast.hpp` 新增 `VectorTypeLiteral`（包含长度和元素类型），并将 `Type` 扩展为 `IdentifierPath | VectorTypeLiteral`。
- **语法**：在 `parser/grammar/expression_grammar_def.hpp` 的 `type` 规则增加 `vector_type` 分支，语法形态为 `vec<长度, 元素类型>`（沿用尖括号模板风格）。对应 rule 声明添加于 `expression_grammar.h`。
- **Lowering**：`lowering/expression_lowering_visitor.hpp::ApplyType` 增加向量分支，校验长度>0、元素类型非 void，生成 `ir::VectorType(elem_type, len)`。

### 变更原因
- NVVM WMMA/ldmatrix 等官方 intrinsic 签名要求 `<2 x half>/<2 x float>` 这样的向量字段，当前前端缺少向量语法导致只能用 Array 伪装，LLVM verifier 报类型不匹配。
- Prajna 后端已有 `ir::VectorType` 和 LLVM 映射，补前端语法即可直接产出正确的 IR 类型，解锁 WMMA 等特性。
- 采用 `vec<Len, Elem>` 语法复用现有尖括号模板风格，改动面小、可读性一致。

### 下一步
- 加一个最小测试验证 `vec<2, f16>` 等语法/IR 生效。
- 用新语法重写 `builtin_packages/nvgpu_wmma` 绑定为 NVVM 官方签名（`.p3` 版本，fragment 用向量字段），再跑 wmma_minimal 用例。
