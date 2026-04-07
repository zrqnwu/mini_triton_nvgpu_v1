Tensor Core 集成改动梳理
========================

目标：让 Prajna 通过 NVVM 内建使用 NVIDIA Tensor Core（m16n16k16 f16→f32 路径），从前端到 PTX/驱动全链路可跑通。

主要改动
--------
1) 前端/类型层
- 增加 `vec<Len, Element>` 语法与 IR 支持，能生成 LLVM 向量类型作为字段/参数（文件：`prajna/ast/ast.hpp`, `prajna/parser/grammar/*`, `prajna/ir/type.hpp`, `prajna/codegen/llvm_codegen.cpp`）。
- LLVM codegen 改进：
  - 指针类型保留 `address_space`，不再硬编码 0。
  - Struct 先建占位再 setBody，支持 `is_literal` 生成匿名 struct，避免自引用递归崩溃，并匹配 NVVM 的 literal 聚合。

2) WMMA 绑定
- `prajna/builtin_packages/nvgpu_wmma/.prajna`：
  - 片段字段对齐官方签名：A/B 用 8 个 `vec<2,f16>`，累加片段 8 个 `f32`。
  - 内建签名对齐 LLVM19 官方：`load.*.f16.p0`、`mma.row.col.f32.f32`、`store.*.f32.p0`，`WmmaMma` 扁平展开 24+8 参数。
  - `WmmaFill` 去掉未定义的 intrinsic，内联构造零片段，避免 PTX INVALID_PTX。

3) 验证用例
- `tests/prajna_sources/nvgpu/wmma_minimal.prajna`：使用新接口（stride 显式 i32、命名空间前缀），验证 m16n16k16 f16→f32 row/col 路径。

验证结果
--------
- 生成的 PTX（含 `wmma.load/mma/store` 指令，目标 `.target sm_86`，无未定义内建。
- 测试通过：`./build_release/bin/prajna_compiler_tests --gtest_filter=PrajnaTestsInstance/PrajnaTests.TestSourceFile/nvgpu__wmma_minimal`。
- 说明 Tensor Core m16n16k16 f16→f32 路径已从前端到驱动跑通。

当前限制 / 待拓展
-----------------
- vec 尚未提供 `[]` 元素级读写；WMMA 调用不受影响。
- 仅验证了 m16n16k16 row/col 这一条路径，其他布局/尺寸、ldmatrix/fragment 变体未覆盖。
- 如需共享内存 `.p3` 版本或更多 intrinsic，需在绑定中补充对应 addrspace/签名并加用例。
