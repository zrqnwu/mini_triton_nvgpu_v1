# `mini_triton_nvgpu_v1` 当前剩余性能差距的一次性修复方案

## 1. 文档目的

这份文档只讨论当前最新代码里还真实成立的性能问题，以及对应的一次性修复方案。

它建立在下面这份复核文档之上：

- [mini_triton_v1_current_perf_vs_triton_root_cause_reassessment_20260402.md](/home/zhangruiqi/docs/mini_triton_v1_current_perf_vs_triton_root_cause_reassessment_20260402.md)

因此本文不再重复讨论已经被排除的旧根因，只处理当前仍成立的两类问题：

1. `C / epilogue` 的 direct-global-vector 真相没有真正保到最终 target。
2. `large tile` 下后段 `shared / register / global` 组织仍未贴到 Triton。

---

## 2. 当前结论

基于当前代码、当前实测和当前 PTX 证据，现状是：

- `A/B cp.async + wait + ldmatrix + mma.sync` 主线已经回来。
- 当前主要差距不再是 async producer 缺失。
- 当前主要差距集中在 `C / epilogue` 的 target materialization。

最关键的代码证据是：

- 当前 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 中的 `loadDirectGlobalPack(...)` 和 `storeDirectGlobalPack(...)` 已经在使用 `vector::LoadOp / vector::StoreOp`。
- 但最终产物 [mini_tb_current_64.mlir](/tmp/mini_tb_current_64.mlir) 和 [mini_tb_current_128.mlir](/tmp/mini_tb_current_128.mlir) 里仍然落成大量 `ld.global.b32 / st.global.b32`。
- Triton 当前对应产物 [triton_exact_tile_64x64x32_real.ptx](/tmp/triton_exact_tile_64x64x32_real.ptx) 和 [triton_exact_tile_128x128x32_real.ptx](/tmp/triton_exact_tile_128x128x32_real.ptx) 则是：
  - `ld.global.v4.b32`
  - `st.global.v4.b32`
  - `st.shared.v4.b32`
  - `ld.shared.v2.f32`

这说明：

- 当前问题不是“上层没有 direct_global_vector plan”。
- 当前问题是“现有 direct_global_vector plan 没有形成 target 可保持的内存 owner，最后在 lowering sink 里被 scalarize 了”。

所以这次必须做的是结构性修复，不是继续改 vector 宽度、barrier 数量或局部 thread decomposition。

---

## 3. 设计原则

## 3.1 不回头重做已经正确的 A/B async 主线

当前 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 已经明确要求：

- `transport.operandA.kind == "cp_async"`
- `transport.operandB.kind == "cp_async"`
- `asyncPlan.waits` 非空

而最新产物也已经有：

- `cp.async.cg.shared.global`
- `cp.async.commit_group`
- `cp.async.wait_group`

因此，这次方案不再把精力放到 A/B async 主线上。

## 3.2 不再把“logical direct”误当成“最终 target 一定直接 vector.load/store scalar memref”

当前 `DirectGlobalVectorPlan` 的命名容易造成误解。

对当前 mini 来说，“direct_global_vector”真正应该表达的是：

- C init/store 不通过语义层的 `SharedRelayPlan`
- C pack 与 accumulator fragment 的 owner 是直接对齐的
- 最终 global memory 必须保持 vectorized landing

它**不应该再被偷换成**：

- “lowerer 直接对 `memref<?x?xf32>` 发 `vector::LoadOp / vector::StoreOp` 就算完成”

因为这条路径已经被当前产物证明会在 sink 阶段 scalarize。

## 3.3 语义层 shared relay 和 target-local epilogue staging 必须区分

当前 [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h) 里已经有：

- `DirectGlobalVectorPlan`
- `SharedRelayPlan`

但 Triton 当前 PTX 说明，存在第三类东西：

- 它不是语义层 `SharedRelay`
- 但 target materialization 时会使用 shared pack / unpack 来完成更优的 global vector landing

这类东西在当前 mini 里还没有明确 owner。

所以这次必须引入一层新的真相：

- 语义层是否 direct
- target 层如何完成 direct landing

这两层不能再混在一起。

## 3.4 不再使用 reinterpret_cast / type_cast 之类的 lowerer 局部补丁

之前已经验证过：

- 想在 lowerer 末端靠 `memref.reinterpret_cast`
- 或者靠 `vector.type_cast`
- 从标量 `memref<?x?xf32>` 临时拼出 vector memref

这条路在当前主线上不可靠，也不符合 owner 清晰原则。

因此本次方案明确禁止：

- 继续把“保住 v4 global I/O”当成 lowerer 局部 hack

必须改成显式的数据结构和显式的 late target path。

---

## 4. 完整方案概览

这次一次性修复分成四条链，必须全部完成。

### 第一条链：重建 `EpiloguePlan` 的 owner 边界

目标：

- 让 `DirectGlobalVectorPlan` 只表达逻辑 pack / fragment / lane owner
- 让 target-specific landing 另有明确 owner

### 第二条链：增加显式的 target epilogue landing contract

目标：

- 不再依赖 `vector::LoadOp / vector::StoreOp` 能否在 sink 阶段自动保住 `v4`
- 把 target 需要的 `shared pack / global v4 load-store` 变成明确合同

### 第三条链：把 `LowerPipelineToNVGPU` 从“直接尝试 vector memref I/O”改成“发出晚期 target-owned epilogue ops”

目标：

- `tb-lower-pipeline-to-nvgpu` 只负责把 pack/fragment/warp owner 变成显式 epilogue materialization 主线
- 不再在这里赌官方通用 sink 会自动保住最终向量内存访问

### 第四条链：新增晚期 epilogue vector I/O 收口 pass

目标：

- 在真正接近 LLVM/NVVM ptr 语义的阶段，把 epilogue target contract 收口成稳定的 `ld/st v4`
- 从结构上拔掉当前 scalarization 根因

---

## 5. 详细修改方案

## 5.1 重构 `EpiloguePlan`

修改文件：

- [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)
- [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp)
- [BuildCRegisterPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp)

当前问题：

- `DirectGlobalVectorPlan` 同时承担了：
  - 逻辑 fragment/pack owner
  - target memory 落地方式假设
- 这导致 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 只能直接把它翻译成 `vector::LoadOp / vector::StoreOp`
- 一旦 sink 阶段 scalarize，整个 contract 就失真

这次要改成两层：

### 层一：逻辑 direct pack truth

保留当前 `DirectGlobalVectorPlan` 中真正属于逻辑层的字段：

- `laneAccess`
- `packs`
- `vectorWidth`
- `boundaryAware`
- `scalarTail`

这层只表达：

- 哪些 accumulator fragment 组成一个 pack
- 每个 lane 负责哪几行
- 每一行逻辑向量宽度是多少

### 层二：target landing truth

新增一个显式 target landing 结构，建议命名为：

- `TargetLandingPlan`

建议至少包含：

- `kind`
  - `shared_pack_then_global_vector`
  - 后续如有必要可扩展别的 target 落地方式
- `globalVectorWidth`
- `globalAccessBytes`
- `sharedPackVectorWidth`
- `sharedPackRows`
- `sharedPackCols`
- `useSharedPackForInit`
- `useSharedPackForStore`

它负责表达的是：

- direct-global 语义最终在 target 上怎么 materialize
- 是否需要 target-local shared pack staging
- global 侧最终想保住的访问粒度

这里最重要的一条是：

- `TargetLandingPlan` 不是 `SharedRelayPlan`
- 它不改变语义层 owner
- 它只是 direct path 的 target-local materialization contract

### 选择规则

对当前 `sm_86 + exact_tile + stage1` 主线，必须固定选择：

- `kind = shared_pack_then_global_vector`

不再允许：

- `kind = implicit_vector_memref_io`
- 或任何“让 lowerer 直接对 scalar memref 发 vector::Load/Store 然后赌 sink 不 scalarize”的隐式策略

### 验证规则

在 `EpiloguePlan` 验证里增加硬约束：

1. 如果 `initMode/storeMode == DirectGlobalVector`，则必须同时携带 target landing payload。
2. 如果 target landing 缺失，就直接报错。
3. 如果 stage1 exact-tile 仍落到旧的隐式 vector memref I/O，直接报错，不再接受静默退化。

---

## 5.2 让 `BuildCRegisterPlan` 产出完整的 target landing owner

修改文件：

- [BuildCRegisterPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp)
- [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp)

当前问题：

- `deriveEpiloguePlan(...)` 只生成逻辑 direct pack
- target-specific landing 仍留给 lowerer 猜

这次要改成：

- `deriveEpiloguePlan(...)` 直接根据：
  - `KernelConfig`
  - `TargetInfo`
  - `EncodingPlan`
  - `AccumulatorPlan`
- 产出完整的：
  - 逻辑 direct pack plan
  - target landing plan

这样 `tb.epilogue_plan` 才是单一真相。

后续 pass 不再允许：

- 自己额外推测 shared pack 形状
- 自己推测 global v4 落地宽度

---

## 5.3 改写 `LowerPipelineToNVGPU` 的职责边界

修改文件：

- [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

当前问题：

- `loadDirectGlobalPack(...)`
- `storeDirectGlobalPack(...)`

这两段当前直接操作 `cMemref`，使用：

- `vector::LoadOp`
- `vector::MaskedLoadOp`
- `vector::StoreOp`
- `vector::MaskedStoreOp`

这在 IR 上看起来像 vector path，但最终 target 并没有保住。

### 必须删除的错误前提

不再允许继续依赖下面这条前提：

- “只要 `DirectGlobalVectorPlan` 的 `vectorWidth` 对了，`vector::StoreOp` 最后自然会变成 `st.global.v4.b32`”

这条前提已经被当前产物证伪。

### 新职责

`LowerPipelineToNVGPU` 改成只做下面三件事：

1. 读出 `tb.epilogue_plan` 中的逻辑 direct pack owner。
2. 读出 `tb.epilogue_plan` 中的 target landing owner。
3. 生成显式的 late target-owned epilogue materialization op。

也就是说，当前这两个函数的职责要拆开：

- `loadDirectGlobalPack(...)`
- `storeDirectGlobalPack(...)`

应当重构为：

- 构造逻辑 pack value
- 构造 target epilogue I/O op

而不是：

- 直接在这里对最终 memory 形态做 `vector::Load/Store`

---

## 5.4 新增显式的 epilogue target op

修改文件：

- [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td)
- [TBOps.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.h)
- 以及对应生成与注册文件

建议新增两类 op，名字可以按实现调整，但职责必须明确：

- `tb.epilogue_global_vector_load`
- `tb.epilogue_global_vector_store`

可选地，如果需要把 shared pack staging 也单独显式化，可以再加：

- `tb.epilogue_shared_pack_store`
- `tb.epilogue_shared_pack_load`

这些 op 的目标不是成为长期公共前端语义，而是：

- 作为 `tb-lower-pipeline-to-nvgpu` 到 LLVM/NVVM sink 之间的一层显式 target contract

### 为什么必须加 op

因为当前真正缺的不是“再多一点 plan 字段”，而是一个**不会在官方通用 sink 中丢失语义**的晚期 owner。

如果没有这层 op：

- 上层 plan 再精确
- lowerer 再努力拼 `vector::Load/Store`

最终还是可能在 sink 里被统一 scalarize。

新增 op 的意义就是：

- 把“必须保住 global v4 I/O”从一个希望，变成一个必须被后续 pass 消费的显式约束

---

## 5.5 新增晚期 epilogue vector I/O pass

修改文件：

- [Passes.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Transforms/Passes.td)
- 新增 pass 实现文件，建议命名：
  - `LowerEpilogueVectorIO.cpp`
- [Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)

### 这一步为什么必须新增 pass

当前 [Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp) 已经写了注释：

- “direct-global C pack 需要保留 vector load/store 到 LLVM/NVVM”

但现实是：

- 当前 pipeline 仍然没有把这件事真正做成

所以这次必须新增一条专门的晚期 pass，而不是继续把责任堆给：

- `tb-lower-pipeline-to-nvgpu`
- 或官方 `convert-vector-to-llvm`

### 新 pass 的职责

它应当位于：

- `tb-lower-pipeline-to-nvgpu`
之后
- `convert-vector-to-llvm`
之前

它只做一件事：

- 把 `tb.epilogue_*` target op 收口成不会再被错误 scalarize 的 LLVM/NVVM-level vector memory I/O

### 这一步的实现原则

1. 不再依赖 scalar memref 上的通用 `vector::Load/Store`。
2. 必须直接面向最终 target memory access 语义。
3. 只服务当前 stage1 exact-tile 主线，不抢前面 passes 的 owner。

这条 pass 是这次方案能否“去根”的关键。

如果不加它，前面的 plan 重构仍然会在最后一步失真。

---

## 5.6 把 Triton 的 shared pack 组织补进 direct path 的 target materialization

修改文件：

- [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)
- 新增的晚期 epilogue pass

当前从 PTX 对比可见，Triton 的 epilogue 并不是简单的：

- accumulator -> global vector store

它实际上还有：

- `st.shared.v4.b32`
- `ld.shared.v2.f32`
- 再到 `st.global.v4.b32`

这说明 Triton 在大 tile 上依赖 shared pack/unpack 来重排 epilogue 数据形态。

因此本次方案明确规定：

- 对当前 stage1 exact-tile 主线，target landing 必须实现 `shared_pack_then_global_vector`

### 注意

这里的 shared pack：

- 不是语义层 `SharedRelayPlan`
- 不是旧意义上的 C relay fallback
- 它只是 direct path 在 target 层的 pack/unpack 物理组织

这条边界必须在文档、plan 和代码里都保持清楚。

---

## 5.7 `MatmulRewritePlan` 中的 direct flags 只保留逻辑语义，不再暗示最终 materialization

修改文件：

- [MatmulRewritePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulRewritePlan.h)
- [MatmulRewritePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulRewritePlan.cpp)

当前已有：

- `directAccumulatorInit`
- `directAccumulatorStore`

这两个字段可以保留，但语义必须重新收紧：

- 它们只说明当前主线在语义层不经过 `SharedRelayPlan`
- 它们不再被允许暗示“lowerer 最终直接对 scalar memref 发 vector::Load/Store”

否则就会再次出现：

- 名义 direct
- 实际 materialization owner 仍不清晰

---

## 6. 实施顺序

这次必须按下面顺序做，不能打散。

### 第一步：先改 `EpiloguePlan` 数据结构

先把：

- 逻辑 direct pack
- target landing

拆清楚。

如果这一步不做，后面所有修改都会继续变成 lowerer 局部猜测。

### 第二步：让 `deriveEpiloguePlan(...)` 真正产出 target landing contract

只有这一步完成，`tb.epilogue_plan` 才是完整单一真相。

### 第三步：新增 `tb.epilogue_*` target op

把“必须保住 global v4 I/O”的要求变成显式 late owner。

### 第四步：重写 `LowerPipelineToNVGPU`

删掉当前直接对 `cMemref` 发 `vector::Load/Store` 的主线，改成发 target epilogue op。

### 第五步：新增晚期 epilogue vector I/O pass 并接入 `Stage1Pipelines.cpp`

只有这一步完成，前面的新 contract 才能在 LLVM/NVVM 前真正落地。

### 第六步：最后统一验收

只在全部改完后再测。

---

## 7. 验收标准

## 7.1 Plan / contract 层

`tb.epilogue_plan` 必须同时表达：

- 逻辑 direct pack owner
- target landing owner

不再允许：

- direct path 只带逻辑 pack，不带 target landing truth

## 7.2 Lowering 中间层

`tb-lower-pipeline-to-nvgpu` 之后，不再出现当前这种：

- `loadDirectGlobalPack(...)` 直接对 `cMemref` 发通用 `vector::LoadOp`
- `storeDirectGlobalPack(...)` 直接对 `cMemref` 发通用 `vector::StoreOp`

而是要出现：

- 显式 epilogue target op

## 7.3 最终 PTX 层

当前 stage1 exact-tile 主线至少要恢复出下面形态：

- `ld.global.v4.b32`
- `st.global.v4.b32`

并且不再是现在的大量：

- `ld.global.b32`
- `st.global.b32`

如果最终 PTX 仍然是大片标量 global I/O，就说明这次方案没有真正落地。

## 7.4 性能层

新的修复至少应满足：

### `64x64x32`

- 从当前 `85% Triton` 再进一步靠近 Triton

### `128x128x32`

- 必须显著缩小当前 `73% Triton` 的 gap

因为当前真正的问题主要伤害大 tile。

如果改完后：

- `64` 基本不动
- `128` 也基本不动

那就说明仍然没有打到真正根因。

---

## 8. 这次方案为什么是“去根”而不是局部调参

因为当前已经被证明：

- 不是 async producer 缺失
- 不是 barrier 个数本身
- 不是某个 vector width 数字没调对

当前真正的问题是：

- direct C path 的 target owner 不完整
- epilogue 的 late materialization contract 缺失

只要这两件事不改，后面无论怎么调：

- warp 分解
- vector 宽度
- barrier 数量
- cp.async issue 细节

都会继续在错误的后段形态上打转。

所以这次方案不是“把现有 lowering 写得更聪明一点”，而是：

- 重建 epilogue owner
- 新增晚期 target pass
- 把当前 scalarization 根因结构性拔掉

---

## 9. 最终目标

做完这套之后，当前 stage1 主线应当变成：

1. A/B producer:
   - 继续保持 `cp.async + wait + ldmatrix + mma.sync`
2. C logical path:
   - 继续保持 `direct_global_vector`
3. C target path:
   - 明确变成 `shared_pack_then_global_vector`
4. late lowering:
   - 明确保住 `global v4` I/O，不再被 sink 随机 scalarize

达到这个状态后，当前 mini 和 Triton 剩下的差距才会真正收缩到：

- 调度细节
- 更高阶 pipeline overlap
- 更细的 target peephole

而不会继续卡在现在这类一级结构问题上。
