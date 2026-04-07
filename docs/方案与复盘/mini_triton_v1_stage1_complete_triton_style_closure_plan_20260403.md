# `mini_triton_nvgpu_v1` Stage1 完整 Triton 式收口方案

## 1. 文档目的

这份文档只回答一个问题：

- 基于 `2026-04-03` 当前代码和最新实测，`mini_triton_nvgpu_v1` 想把 stage1 exact-tile matmul 做成真正的 Triton 式实现，还差什么，应该怎么一次性收口。

这里的“完整 Triton 式方案”不是指继续局部调参，而是指：

1. 先明确当前哪些思想已经对了。
2. 再明确哪些地方还不是 Triton 的核心做法。
3. 最后给出一套从数据结构、analysis、lowering、resource closure 到验收口径都闭环的方案。

本文只覆盖 stage1 exact-tile matmul 主线，不扩展到 stage2/general-shape。

---

## 2. 当前状态

## 2.1 最新实测结果

当前统一用同一个 `driver_matmul_bench` 串行交错 benchmark，对 mini 和 Triton 进行同口径测试。

最近一轮得到的 steady-state 结果如下：

| shape | mini | Triton | 结论 |
| --- | ---: | ---: | --- |
| `64x64x32` | `3976.14 ns` | `3651.98 ns` | mini 慢 `8.88%` |
| `128x128x32` | `5737.68 ns` | `5962.09 ns` | mini 快 `3.76%` |
| `64x128x32` | `4994.76 ns` | `4857.65 ns` | mini 慢 `2.82%` |
| `128x64x32` | `5123.57 ns` | `4855.50 ns` | mini 慢 `5.52%` |
| `64x64x64` | `6952.24 ns` | `6497.79 ns` | mini 慢 `6.99%` |
| `128x128x64` | `7244.08 ns` | `7389.39 ns` | mini 快 `1.97%` |

结论很明确：

1. 大方块 exact-tile 已经基本追平，甚至略超 Triton。
2. 小方块和矩形 tile 还有尾差。
3. `K` 拉长后，`64x64x64` 的差距会重新放大。

---

## 2.2 当前已经对齐 Triton 的部分

从 PTX 计数和产物形态看，下面这些主线已经基本对了：

1. `A/B` operand 的主线塑形已经基本对齐。
   - `ldmatrix` 计数能和 Triton 对上。
   - `mma.sync` 计数能和 Triton 对上。

2. exact-tile 大方块的主吞吐已经能贴近 Triton。
   - `128x128x32`
   - `128x128x64`

3. `direct_global_vector + target landing` 已经比早期版本收口得多。
   - 不再是最早那种“完全丢 direct path”的状态。
   - [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp#L860) 的 `row-run + static-shared-budget` 逻辑已经比硬编码常数好很多。

换句话说：

- 现在已经不是“matmul 主线思想全错”的阶段。
- 当前剩余问题已经集中到 `epilogue / dependency / resource closure`。

---

## 2.3 当前还没对齐 Triton 的部分

最新实测和 NCU 指向的是同一类问题：

1. 当前慢的 case，主要都表现为 `long_scoreboard` 明显偏高。
2. 但 `barrier stall` 往往并不高，甚至比 Triton 低。
3. 所以当前剩余慢点不是“barrier 太多”，而是“dependency hiding 不够”。

例如：

### `64x64x32`

- mini `long_scoreboard = 4.24`
- Triton `long_scoreboard = 0.57`

### `64x128x32`

- mini `long_scoreboard = 8.28`
- Triton `long_scoreboard = 2.13`

### `128x64x32`

- mini `long_scoreboard = 5.62`
- Triton `long_scoreboard = 1.74`

### `64x64x64`

- mini `long_scoreboard = 4.42`
- Triton `long_scoreboard = 1.03`

这说明当前剩余 gap 的核心，不在 mainloop 核心计算，而在：

1. C/epilogue 后段落地形态
2. 依赖链组织
3. shared/register/global 的资源闭环

---

## 3. 当前不是 Triton 思想的地方

这部分是整份文档最重要的结论。

## 3.1 `row-run` 还是 heuristic，不是 Triton 式 fixed-point

当前 [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp#L860) 里，`TargetLandingPlan` 的核心仍然是：

1. 先数同一 `rowBase` 的 direct pack 连续 run
2. 再结合 static shared 预算算 `sharedPackSlots`
3. 用 `min(rowRunPackCount, maxPackSlotsByBudget)` 决定 landing

这比硬编码常数强很多，但它仍然是 heuristic。

它的问题不是“逻辑错”，而是“不是 Triton 式的 owner 真相”：

1. 它依赖第一行 pack run 的形状。
2. 它天然带有方向偏置。
3. 它不是从 `layout relation + warp decomposition + fragment ownership` 推导出来的 fixed-point。

从结果上看，这个偏置是能测出来的：

- `64x128x32` 只慢 `2.82%`
- `128x64x32` 慢 `5.52%`

这说明当前 landing 方案对宽 `N` 的矩形 tile 更友好，对高 `M` 的矩形 tile 更保守。

这不是 Triton 的做法。

---

## 3.2 当前 direct epilogue 仍然带着 shared-pack relay 痕迹

从 PTX 看，当前 mini 的 direct epilogue 仍然明显有一套自己的 shared relay 风格：

- `bar.warp.sync`
- `ld.shared.v2.b32`
- `ld.shared.v4.b32`
- `st.shared.v2.b32`
- 再 `st.global.v4.b32`

而 Triton 当前对应 shape 更接近：

- `st.shared.v2.b32`
- `ld.shared.v2.f32`
- `st.global.v4.b32`

也就是说：

1. Triton 的 epilogue landing 更像“consumer-shaped fragment -> target-owned final landing”。
2. 你现在更像“shared pack/unpack 中转后，再回 global vector store”。

这会直接造成：

1. 额外的 shared read/write dependency
2. 更多的 warp-local sync
3. 更高的 `long_scoreboard`

这也是为什么当前 barrier 明明不高，但 kernel 还是慢。

---

## 3.3 当前 resource closure 还不够 Triton 式

`64x64x64` 是最能暴露这个问题的 case。

资源对比：

- mini：`regs=254`, `shared=13312`
- Triton：`regs=168`, `shared=22016`

这说明 Triton 在这个 case 里的策略更像：

- 多花 shared
- 少压 register
- 用 shared 来换更短的 dependency chain

而当前 mini 更像：

- static shared 比较保守
- 状态更多留在寄存器和后段依赖里
- 导致 deep-K 小 tile 的 scoreboarding 更重

所以 current gap 不是“少一个指令 pattern”，而是：

- shared/register/global 三者之间还没有被同一个 resource plan 收口。

---

## 3.4 stage1 覆盖边界还不是完整的 Triton 式

当前 stage1 还有两个结构性覆盖问题：

1. [KernelConfig.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/KernelConfig.cpp#L225) 仍把 `num_warps` 限制在 `{1, 4, 8}`。
2. `cubin-format=mlir` 在扩展 shape 上还会触发 sink 路径 bug。

这两点不一定是当前已测 shape 的主性能根因，但它们说明：

- 现在的 stage1 还没形成真正完整的 Triton 式 shape closure。

---

## 4. 目标

## 4.1 性能目标

stage1 common exact-tile matmul 至少覆盖下面六组：

1. `64x64x32`
2. `64x128x32`
3. `128x64x32`
4. `64x64x64`
5. `128x128x32`
6. `128x128x64`

目标不是“某一组刷分”，而是：

1. 这六组都走同一条算法链。
2. 慢的几组都收敛到 Triton `95%+`。
3. 已经好的几组不能回退。

建议验收线：

- `64x64x32`：达到 Triton `95%+`
- `64x128x32`：达到 Triton `97%+`
- `128x64x32`：达到 Triton `97%+`
- `64x64x64`：达到 Triton `95%+`
- `128x128x32`：不允许比当前退化超过 `2%`
- `128x128x64`：不允许比当前退化超过 `2%`

---

## 4.2 思想目标

真正的目标不是数字本身，而是把 stage1 的核心思想改成 Triton 式：

1. `layout-first`
2. `consumer-shaped fixed-point`
3. `lowering-thin`
4. `resource closure`

只有这四条同时成立，性能才会稳定，而不是一组快一组慢。

---

## 5. 完整 Triton 式方案

下面是建议的一次性收口方案。它不是“哪里慢修哪里”，而是按 Triton 核心思想，把当前剩余问题都收在同一条链上。

## 5.1 第一层：补齐 warp decomposition owner truth

### 目标

把 stage1 exact-tile 的 warp 级几何和 fragment owner 显式化，不能再只靠 `num_warps + block_m + block_n` 的隐含关系推断。

### 要做什么

新增一层显式 analysis，建议命名为：

- `WarpDecompositionPlan`

建议至少包含：

1. `ctaTile`
2. `numWarps`
3. `warpTiles`
4. `warpOrder`
5. `warpMmaGroups`
6. `warpAccumulatorCoverage`
7. `warpEpilogueCoverage`

每个 `warpTile` 至少要能表达：

1. `warpId`
2. 负责的 `(m,n)` 子区域
3. 对应的 `mma group` 列表
4. 对应的 `accumulator pack` 范围
5. 对应的 `epilogue direct pack` 范围

### 为什么这是 Triton 式

Triton 的性能关键不是“先写 shared 再说”，而是：

- 每个 consumer 对 fragment/pack 的 owner 是明确的
- target landing 只是把这个 owner 忠实落地

如果 warp 级 owner 还没有显式化，后面的 epilogue landing 一定会继续退化成 heuristic。

### 建议落点

新增/修改：

- 新增 `include/tb/Analysis/WarpDecompositionPlan.h`
- 新增 `lib/Analysis/WarpDecompositionPlan.cpp`
- 修改 [KernelConfig.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h)
- 修改 [KernelConfig.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/KernelConfig.cpp)
- 修改 [ProgramMappingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/ProgramMappingPlan.cpp)

### 这一层完成后的效果

1. 矩形 tile 的 warp owner 真相明确。
2. 后续 epilogue 不再需要从 `rowBase` 猜 batch。
3. stage1 覆盖边界可以从“固定 `{1,4,8}`”逐步走向“按几何合法性决定”。

---

## 5.2 第二层：重构 `AccumulatorPlan / EpiloguePlan` 的 owner 分工

### 目标

把现在混在 `EpiloguePlan` 里的逻辑 owner 和 target landing owner 真正拆开。

### 当前问题

当前 [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h) 里虽然已经有：

- `DirectGlobalVectorPlan`
- `SharedRelayPlan`
- `TargetLandingPlan`

但 `TargetLandingPlan` 的生成仍然带着：

- `row-run`
- `sharedPackSlots`
- static shared budget heuristic

它还不是由 `AccumulatorPlan + WarpDecompositionPlan + layout relation` 共同推导出来的。

### 要做什么

把当前的 `EpiloguePlan` 拆成三层真相：

#### 层一：逻辑 accumulator owner

继续由 [AccumulatorPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h) 负责，但需要补：

1. 每个 pack 的 warp owner
2. 每个 pack 的 fragment owner
3. 每个 pack 与 epilogue direct pack 的映射关系

#### 层二：逻辑 epilogue direct owner

`DirectGlobalVectorPlan` 只表达：

1. 哪些 fragment 组成一个 direct pack
2. 每个 pack 的 lane ownership
3. 每个 pack 的 global vector width
4. boundary/scalar tail 语义

它不再表达：

1. slot 数
2. shared staging 次序
3. static shared budgeting

#### 层三：target landing owner

`TargetLandingPlan` 改成纯 target contract，建议显式表达：

1. `landingKind`
2. `producerFragmentShape`
3. `sharedStoreShape`
4. `sharedLoadShape`
5. `globalStoreShape`
6. `warpBatchingGroup`
7. `requiredSyncKind`
8. `requiredSharedBytes`
9. `expectedRegisterFootprint`

最重要的一条：

- `TargetLandingPlan` 不允许再从第一行 pack run 猜 slot 数。

它必须由：

1. `WarpDecompositionPlan`
2. `AccumulatorPlan`
3. `DirectGlobalVectorPlan`
4. target layout relation

共同推导出来。

### 建议落点

- 修改 [AccumulatorPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h)
- 修改 [AccumulatorPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AccumulatorPlan.cpp)
- 修改 [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)
- 修改 [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp)
- 修改 [BuildCRegisterPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp)

### 这一层完成后的效果

1. `64x128x32` 和 `128x64x32` 的 landing 不再出现明显方向偏置。
2. `TargetLandingPlan` 不再依赖 `rowRunPackCount`。
3. `EpiloguePlan` 真正开始像 Triton 的 fixed-point owner contract。

---

## 5.3 第三层：把 `LowerPipelineToNVGPU` 改成真正的 thin lowering

### 目标

让 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 不再靠 lowering-local 的 shared-pack/unpack 顺序“猜”最终形态，而是严格执行上层 contract。

### 当前问题

从 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 的 direct path 可以看到：

1. lowering 里仍然显式安排了 pack/stage/store/load 组织。
2. 这导致：
   - `bar.warp.sync`
   - `ld.shared.v4.b32`
   - `ld.shared.v2.b32`
   - `st.shared.v2.b32`
   的具体顺序被写死在 lowerer 里。

这不是 Triton 的方式。

### 要做什么

把 lowering 的职责收紧成两件事：

1. mainloop 侧忠实落 A/B contract
2. epilogue 侧忠实落 `TargetLandingPlan`

具体来说：

1. direct epilogue path 不允许再自己决定 shared pack slot 次序。
2. 不允许再自己决定是否插 `bar.warp.sync`。
3. 不允许再在 lowerer 里默默把一个逻辑 direct path扩成“shared unpack relay”。

可以接受的 lowering 行为只有：

1. 根据 `TargetLandingPlan` 明确发：
   - shared fragment store
   - shared fragment load
   - global vector store
2. 根据 `requiredSyncKind` 明确发同步

不接受：

1. 只因为当前 helper 好写，就额外插入 `bar.warp.sync`
2. 只因为当前 vector shape 顺手，就生成 `ld.shared.v4.b32`
3. 只因为现在已有 `row-run` slot，就沿用旧的 batch 顺序

### 建议落点

- 修改 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

### 这一层完成后的效果

1. mini 的 direct epilogue 会更接近 Triton 的 target landing 形态。
2. 慢 case 的 `long_scoreboard` 才有机会实质下降。
3. “barrier 少但还是慢”的问题才会被真正解决。

---

## 5.4 第四层：补一层显式的 `ResourceClosurePlan`

### 目标

把 shared/register/global 之间的取舍变成显式 plan，而不是隐含地落在 lowering 里。

### 当前问题

`64x64x64` 上的资源差距说明：

- Triton 用更多 shared，压更少 register
- mini 用更少 shared，压更多 register

这说明现在缺的不是“一个更好的 PTX pattern”，而是：

- 没有一层统一的 resource closure plan 来做 tradeoff

### 要做什么

新增一层 analysis，建议命名为：

- `ResourceClosurePlan`

至少包含：

1. `estimatedAccumulatorRegs`
2. `estimatedEpilogueRegs`
3. `estimatedABShared`
4. `estimatedEpilogueShared`
5. `staticSharedBudget`
6. `dynamicSharedBudget`
7. `chosenLandingTradeoff`
8. `chosenBufferingTradeoff`
9. `reason`

它的职责不是“做自动调参”，而是：

1. 对当前 exact-tile case 给出一套明确的 shared/register 分配决策
2. 让 deep-K 小 tile 不再因为 shared 太保守而把依赖压进寄存器

### 建议落点

- 新增 `include/tb/Analysis/ResourceClosurePlan.h`
- 新增 `lib/Analysis/ResourceClosurePlan.cpp`
- 修改 [Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)
- 修改 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

### 这一层完成后的效果

1. `64x64x64` 这类 deep-K 小 tile 会明显受益。
2. 已经表现好的 `128x128x32/64` 不会被误伤。
3. stage1 终于具备 Triton 式 resource closure 的雏形。

---

## 5.5 第五层：把 stage1 coverage 和测试口径一起补齐

### 目标

保证后面所有优化都基于稳定、可复现、同口径的测试矩阵。

### 要做什么

#### 覆盖面

1. stage1 analysis 不再只在最早两组 shape 上稳。
2. 至少稳定覆盖：
   - `64x64x32`
   - `64x128x32`
   - `128x64x32`
   - `64x64x64`
   - `128x128x32`
   - `128x128x64`

#### 工具链

1. 修掉 `cubin-format=mlir` 在扩展 shape 上的 crash。
2. bench driver 保持公平支持大 shared launch。
   - 当前已经在 [driver_matmul_bench.cpp](/home/zhangruiqi/tmp/driver_matmul_bench.cpp#L249) 加了 `MAX_DYNAMIC_SHARED_SIZE_BYTES` opt-in。

#### 验证矩阵

每次改动后都固定检查：

1. benchmark
2. NCU
3. PTX 计数
4. resource usage

### 为什么这也是 Triton 式的一部分

Triton 的强点不只是“能出一个快 kernel”，而是：

- 同一种思想在多个 shape 上都能稳定成立

如果 stage1 只有两组三方块跑得好，那还不是完整方案。

---

## 6. 实施顺序

这套方案建议按下面顺序推进，不能打散：

1. 先补 `WarpDecompositionPlan`
2. 再重构 `AccumulatorPlan / EpiloguePlan` owner 分工
3. 再重写 `LowerPipelineToNVGPU` 的 direct epilogue lowering
4. 再补 `ResourceClosurePlan`
5. 最后统一补 stage1 coverage、PTX/NCU/benchmark 验收

这个顺序的原因是：

1. warp owner 不明，epilogue fixed-point 一定还是 heuristic
2. epilogue contract 不明，lowerer 一定还会自己猜
3. lowering contract 不稳，resource closure 无法真正落地

---

## 7. 验收标准

## 7.1 结构验收

必须同时满足：

1. `TargetLandingPlan` 不再用 `rowRunPackCount` 决策 slot 数。
2. direct epilogue owner 不再依赖第一行 pack run。
3. lowering direct path 不再内嵌 heuristic `bar.warp.sync + shared unpack` 序列。
4. `ResourceClosurePlan` 对 shared/register tradeoff 有显式真相。

---

## 7.2 PTX 验收

对慢 shape：

- `64x64x32`
- `64x128x32`
- `128x64x32`
- `64x64x64`

至少要看到下面几个方向成立：

1. direct epilogue 的 shared load/store 形态更接近 Triton
2. 不再出现当前这么重的 `ld.shared.v4.b32 + bar.warp.sync`
3. global vector landing 保持稳定

---

## 7.3 硬件验收

1. `64x64x32`：进入 Triton `95%+`
2. `64x128x32`：进入 Triton `97%+`
3. `128x64x32`：进入 Triton `97%+`
4. `64x64x64`：进入 Triton `95%+`
5. `128x128x32/64` 不明显回退

NCU 方向必须同时改善：

1. `long_scoreboard` 明显下降
2. `barrier stall` 不要求机械压到更低，但不能靠“少 barrier”掩盖依赖问题
3. `regs/thread` 在 deep-K 小 tile 上不能继续明显高于 Triton

---

## 8. 这套方案为什么是“完整 Triton 式方案”

因为它不是在修某一个 shape，而是在把当前还不像 Triton 的三个核心问题一次性收口：

1. `epilogue owner` 还不是 fixed-point
2. `direct landing` 还不是 thin lowering
3. `resource closure` 还没有显式 plan

只要这三条不改完，现象就一定还是：

1. 方块大 shape 很强
2. 矩形和 deep-K 小 shape 还会掉
3. barrier 不高但 scoreboard 很高

这正是当前代码的真实状态。

---

## 9. 最终判断

当前 `mini_triton_nvgpu_v1` 的 stage1 已经做到了：

- mainloop 核心思路基本对
- exact-tile 大方块已经接近或超过 Triton

但如果目标是：

- “做成真正的 Triton 式 stage1 common exact-tile matmul”

那还必须补完：

1. warp owner truth
2. epilogue fixed-point owner
3. thin lowering
4. resource closure

把这四层补齐后，stage1 才算真正从“能跑、部分 shape 很快”走到“思想上像 Triton，而且性能也稳定像 Triton”。
