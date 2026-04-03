# Stage1 数据结构与主线 Pass 讲解

本文档解释 `mini_triton_nvgpu_v1` 里 stage1 exact-tile matmul 的两件核心事情：

1. 数据结构是怎样一层层表达语义、布局、流水和 lowering 合同的
2. 主线 pass 是怎样把 `tb.matmul` 推到 `tb.pipeline_mainline`，再推到 NVGPU/NVVM/PTX 的

这份说明面向当前仓库真实代码，不是抽象设计稿。

## 1. 项目当前想解决什么

这个项目不是做一个“能跑就行”的 matmul demo，而是尝试把 Triton/NVIDIA backend 里对性能真正关键的几层硬真相拆出来，放进：

- 显式的 MLIR attr/type
- 显式的 analysis contract
- 显式的 pass 边界

当前 stage1 的工作范围是：

- 单 CTA matmul
- `fp16 x fp16 -> fp32`
- `mma.sync m16n8k16`
- `sm80/sm86`
- 以 exact-tile 为主线
- 重点做对齐 Triton 思想的主线 contract 和 lowering，而不是先做大而全的 general-shape/fallback

## 2. 总体结构图

从高层到低层，主线可以看成下面这条链：

```text
tb.matmul
  -> KernelConfig / TargetInfo
  -> MatmulSemantics
  -> ProgramMappingPlan
  -> EncodingPlan
  -> TransportPlan
  -> AccumulatorPlan + EpiloguePlan
  -> MatmulRewritePlan
  -> BufferModel
  -> LoopPlan
  -> LatencyPlan
  -> PipelinePlan
  -> AsyncPlan
  -> PipelineExpansion
  -> tb.pipeline_mainline + PipelineReady
  -> TBLowerPipelineToNVGPU
  -> TBLowerEpilogueVectorIO
  -> NVGPU / NVVM / PTX
```

这里最重要的原则是：

- 每一层只拥有一类真相
- 后一层只能消费前一层已经公开的合同
- 不允许在 lowering 末端重新“猜”上层已经应该决定好的东西

## 3. 数据结构分层

### 3.1 入口上下文层

#### `KernelConfig`

文件：

- `include/tb/Analysis/KernelConfig.h`

作用：

- 表达 kernel 的问题规模和 tile 请求
- 表达 `numWarps`、`requestedStages`
- 表达当前 stage1 支持的 `mmaKind` 和标量类型

关键字段：

- `problemM/N/K`
- `blockM/N/K`
- `numWarps`
- `requestedStages`
- `groupM`
- `exactTile`

这一层回答的问题是：

- 用户要算多大的矩阵
- 这次 kernel 想用多大的 CTA tile
- 请求了多少 warp、多少 stage

#### `TargetInfo`

文件：

- `include/tb/Analysis/TargetInfo.h`

作用：

- 表达目标 GPU 的硬件能力和 lowering 约束
- 这些信息挂在 module attr 上，而不是等到后端再临时查

关键字段：

- `gpuArch`
- `threadsPerWarp`
- `maxRegistersPerThread`
- `maxSharedBytesPerCTA`
- `supportsAsyncCopy`
- `supportsLdMatrix`
- `supportsMmaSync`
- `asyncCopyPreferredBytes`
- `mmaInstrShape`
- `preferredAsyncTransport`

这一层回答的问题是：

- 目标硬件能不能做 `cp.async`
- ldmatrix/mma.sync 的能力边界在哪里
- 共享内存、寄存器预算大概是什么

### 3.2 语义与映射层

#### `MatmulSemantics`

文件：

- `include/tb/Analysis/MatmulSemantics.h`

作用：

- 把 `tb.matmul` 语义化成“问题大小 + tile + 边界形态”
- 这是 TTIR 向 TTGIR 风格语义化的边界

关键字段：

- `aDescType/bDescType/cDescType`
- `problemM/N/K`
- `tileM/N/K`
- `exactTile`
- `hasBoundaryM/N/K`

这一层不做物理布局，不做 warp/CTA 内部拆分，只负责讲清楚“这个 matmul 的数学语义和 tile 语义是什么”。

#### `ProgramMappingPlan`

文件：

- `include/tb/Analysis/ProgramMappingPlan.h`

作用：

- 表达 program id 到 tile 的映射
- 表达 grouped launch / split-k / persistent 的占位语义
- 明确区分 CTA 外部映射和 CTA 内部 pipeline

关键字段：

- `mappingKind`
- `problemTilesM/N/K`
- `groupM`
- `groupTileSpanM/N`
- `programsPerLaunchGroup`
- `launchGroupCount`
- `totalPrograms`
- `splitK`
- `launchOrder`
- `swizzleKind`

这一层回答的问题是：

- 一个 CTA 负责哪块 tile
- launch 顺序是什么
- grouped launch 的公式组件是什么

### 3.3 布局与搬运层

#### `EncodingPlan`

文件：

- `include/tb/Analysis/EncodingPlan.h`

作用：

- 这是第一层真正的物理合同
- 决定 blocked/shared/dot/accumulator 编码
- 决定 operand fragment 如何和 lane、ldmatrix 对齐

关键字段：

- `aSharedSpec/bSharedSpec`
- `fragmentA/fragmentB/fragmentAcc`
- `encodings`
- `aGlobal/bGlobal/aShared/bShared/aDot/bDot/acc/cStore`

这里最关键的不是“有几种 encoding”，而是：

- 哪个 value/view 对应哪个 encoding id
- operand fragment 的 lane 访问模式和 `ldmatrix` 参数在这里就是硬真相

#### `TransportPlan`

文件：

- `include/tb/Analysis/TransportPlan.h`

作用：

- 表达 global 到 shared 的运输合同
- 把 async/vector/cache 这些问题从 `EncodingPlan` 中剥离出来

关键字段：

- `operandA/operandB`
- `kind`
- `vectorBytes`
- `asyncVectorBytes`
- `transactionBytes`
- `asyncEligible`
- `bypassL1`
- `cachePolicy`

这一层回答的问题是：

- A/B 走什么 transport
- 一次搬多少字节
- 能不能合法发 `cp.async`

### 3.4 C 路径与 matmul 专项重写层

#### `AccumulatorPlan`

文件：

- `include/tb/Analysis/AccumulatorPlan.h`

作用：

- 表达 C accumulator 在寄存器里的 pack 组织
- 说明每个 warp 拥有哪些 accumulator pack

关键字段：

- `registersPerWarp`
- `ownerScope`
- `laneAccess`
- `packs`
- `liveAcrossStages`
- `multiBufferDepth`

这一层是“寄存器里 C 长什么样”的合同。

#### `EpiloguePlan`

文件：

- `include/tb/Analysis/EpiloguePlan.h`

作用：

- 表达 accumulator 初始化和写回的路径
- 区分 direct global vector 和 shared relay
- 给 fused epilogue 留显式语义位置

关键字段：

- `initMode`
- `storeMode`
- `init/store`
- `targetLanding`
- `exprs`

其中：

- `DirectGlobalVectorPlan` 表达 direct path 的 pack 和向量宽度
- `SharedRelayPlan` 表达 relay path 的 shared scratch 和 pack 组织
- `TargetLandingPlan` 表达最后目标侧真正要落成什么 landing

#### `MatmulRewritePlan`

文件：

- `include/tb/Analysis/MatmulRewritePlan.h`

作用：

- 把前面的语义/布局/C 路径合同收束成“真正的 tensor-core mainloop 形态”
- 这是 matmul 专项重写边界

关键字段：

- `mainloopKind`
- `instructionM/N/K`
- `kGroups`
- `accTilesM/N`
- `accumulatorFragments`
- `directAccumulatorInit`
- `directAccumulatorStore`
- `aPath/bPath`

这一层回答的问题是：

- mainloop 里每个 K group 怎么消费 A/B fragment
- C init/store 是否能走 direct 路径
- operand 是不是走 `ldmatrix`

### 3.5 资源图与循环层

#### `BufferModel`

文件：

- `include/tb/Analysis/BufferModel.h`

作用：

- 这是整个 stage1 执行主线里最关键的中间层之一
- 把 kernel 拆成 backings、views、values、ops 四张表
- 让后续 pipeline/latency/wait 都建立在同一份资源图上

关键字段：

- `backings`
- `views`
- `values`
- `ops`

其中：

- `BufferBacking` 表达真正的底层存储实体
- `BufferView` 表达 backing 的某个 stage/tile/fragment/pack 视图
- `ValueState` 表达某个中间值的定义、使用、owner view
- `PipelineOp` 表达 pipeline 级语义 op，而不是低层 opcode

这是“资源真相”的 owner。

#### `LoopPlan`

文件：

- `include/tb/Analysis/LoopPlan.h`

作用：

- 把 stage1 主线规整成单一 K-loop owner
- 明确每个 `kGroup` 的 producer/consumer/compute 结构

关键字段：

- `loopAxis`
- `iterationCount`
- `singleMainLoop`
- `iterations`
- `carriedValues`

这一层回答的问题是：

- 现在主线到底有几个 K group
- 哪些值跨迭代携带

#### `LatencyPlan`

文件：

- `include/tb/Analysis/LatencyPlan.h`

作用：

- 给资源图上的 op 标显式延迟合同
- 这是 schedule 的输入，不是 lowering 再现算的

关键字段：

- `opId`
- `targetLatency`
- `selfLatency`
- `bufferDistance`
- `pipelineable`
- `accMultiBuffer`

### 3.6 调度、异步与展开层

#### `PipelinePlan`

文件：

- `include/tb/Analysis/PipelinePlan.h`

作用：

- 根据 loop 和 latency 得到 coarse pipeline 布局
- 决定 op 放在哪个 stage/cluster/order

关键字段：

- `scheduledMaxStage`
- `placements`
- `stageOwnedBuffers`

这一层回答的问题是：

- 每个 op 被放到哪个 stage
- 哪个 shared slot 归哪个 stage 使用

#### `AsyncPlan`

文件：

- `include/tb/Analysis/AsyncPlan.h`

作用：

- 表达 async producer、group、wait、reuse fence
- 这是 wait frontier 的 owner

关键字段：

- `producers`
- `groups`
- `waits`
- `reuseFences`

其中最关键的是：

- `AsyncProducer.srcOffsets` 直接拥有 global tile 偏移
- `WaitInfo` 拥有 wait 应该插在谁前面
- `ReuseFence` 拥有 slot 什么时候可以复用

#### `PipelineExpansion`

文件：

- `include/tb/Analysis/PipelineExpansion.h`

作用：

- 把 schedule + async 真正展开成 lowering 可消费的 cluster 序列
- 这里已经不再只是“某个 op 在 stage 2”，而是“stage 2 cluster 1 是什么种类、有哪些 op、等哪些 wait group”

关键字段：

- `clusters`

每个 `ExpandedCluster` 带：

- `ordinal`
- `stage`
- `cluster`
- `kGroup`
- `kind`
- `opIds`
- `waitGroupIds`
- `needsBarrier`

### 3.7 cleanup 收口与 lowering 入口层

#### `tb.pipeline_mainline`

文件：

- `include/tb/IR/TBOps.td`

作用：

- 这是 post-cleanup、pre-lowering 的显式 IR owner
- 不再只靠 attr，而是把真正要 lower 的主线 cluster 变成 IR op

body 里当前只允许三类 cluster：

- `tb.async_issue_cluster`
- `tb.consumer_wait_cluster`
- `tb.mma_compute_cluster`

这层存在的意义是：

- 让 `TBLowerPipelineToNVGPU` 直接消费显式 cluster stream
- 不要在 executor/lowering 末端再去重建 pipeline 骨架

#### `PipelineReady`

文件：

- `include/tb/Analysis/PipelineReady.h`

作用：

- 给 cleanup 收口后留一个最小 summary

关键字段：

- `scheduledMaxStage`
- `asyncGroups`
- `requestedStages`

#### `WarpDecompositionPlan`

文件：

- `include/tb/Analysis/WarpDecompositionPlan.h`

作用：

- 表达 CTA 内 warp 到 tile/mma group/pack 的覆盖关系
- 这不是纯几何壳子，而是把 accumulator/epilogue pack 真正挂到 warp 上

关键字段：

- `ctaTile`
- `numWarps`
- `warpGrid`
- `warpTile`
- `ownerScope`
- `warps`

每个 `WarpTileCoverage` 带：

- `mmaGroupIds`
- `accumulatorPackIds`
- `epiloguePackIds`

#### `ResourceClosurePlan`

文件：

- `include/tb/Analysis/ResourceClosurePlan.h`

作用：

- 这是资源预算收口层
- 估算 accumulator/shared/epilogue 的预算和取舍

#### `KernelContract`

文件：

- `include/tb/Analysis/KernelContract.h`

作用：

- 把 lowering 真正关心的核心合同收拢成一份可解析结构
- 当前是从 `tb.pipeline_mainline` 侧解析

## 4. 数据结构依赖图

可以把主要依赖关系简化成下面这样：

```text
KernelConfig + TargetInfo
  -> MatmulSemantics
  -> ProgramMappingPlan

MatmulSemantics + ProgramMappingPlan + TargetInfo
  -> EncodingPlan

EncodingPlan + TargetInfo
  -> TransportPlan

EncodingPlan + TargetInfo + KernelConfig
  -> AccumulatorPlan
  -> EpiloguePlan

EncodingPlan + AccumulatorPlan + EpiloguePlan + KernelConfig
  -> MatmulRewritePlan

MatmulRewritePlan + EncodingPlan + AccumulatorPlan + EpiloguePlan
  -> BufferModel

BufferModel
  -> LoopPlan

BufferModel + LoopPlan + TargetInfo + KernelConfig
  -> LatencyPlan

BufferModel + LoopPlan + LatencyPlan
  -> PipelinePlan

BufferModel + TransportPlan + LatencyPlan + PipelinePlan
  -> AsyncPlan

BufferModel + PipelinePlan + AsyncPlan
  -> PipelineExpansion

PipelineExpansion + AsyncPlan + KernelConfig
  -> tb.pipeline_mainline + PipelineReady

tb.pipeline_mainline + public contracts
  -> TBLowerPipelineToNVGPU
```

这里的关键不是“层数多”，而是每层都在把一种过去容易在 lowering 里偷偷重建的真相提前公开化。

## 5. 主线 Pass 讲解

当前 stage1 主线 pipeline 在：

- `lib/Transforms/Stage1Pipelines.cpp`

顺序如下：

```text
tb-verify-scope
tb-attach-target-info
tb-semanticize-matmul
tb-build-program-mapping-plan
tb-build-layout-plan
tb-build-transport-plan
tb-build-c-register-plan
tb-rewrite-matmul-mainloop
tb-cleanup-layout-conversions
tb-build-mainloop-graph
tb-regularize-k-loop
tb-assign-latencies
tb-schedule-loops
tb-derive-waits
tb-expand-pipeline
tb-cleanup-pipeline
tb-lower-pipeline-to-nvgpu
tb-lower-epilogue-vector-io
... NVGPU/NVVM/LLVM sink pipeline
```

下面按职责解释。

### `tb-verify-scope`

作用：

- 拦住超出当前 V1/stage1 能力边界的输入

它不是优化 pass，而是合法性门槛。

### `tb-attach-target-info`

作用：

- 把目标硬件信息和 module-level 执行上下文挂进 IR

这是后面所有合法性和 lowering 的基础输入。

### `tb-semanticize-matmul`

作用：

- 生成 `tb.semantic_matmul`

这里开始把高层 `tb.matmul` 从“语法壳子”变成“有显式语义合同的 op”。

### `tb-build-program-mapping-plan`

作用：

- 生成 CTA/program 级映射合同

这一层只处理 CTA 外部映射，不碰 CTA 内部执行主线。

### `tb-build-layout-plan`

作用：

- 生成 `tb.encoding_plan`

这是第一个真正决定物理布局的 pass。

### `tb-build-transport-plan`

作用：

- 生成 `tb.transport_plan`

把 A/B transport 合同独立出来，防止 async/vector/cache 语义散落在别处。

### `tb-build-c-register-plan`

作用：

- 同时生成 `tb.accumulator_plan` 和 `tb.epilogue_plan`

也就是把 C 路径拆成：

- 寄存器中 accumulator 的组织
- epilogue init/store/landing 的组织

### `tb-rewrite-matmul-mainloop`

作用：

- 生成 `tb.matmul_rewrite`

这是 matmul 专项重写边界。到了这里，主线已经开始明确贴近 tensor-core fragment 执行模型。

### `tb-cleanup-layout-conversions`

作用：

- 删除没必要的 `tb.convert_layout` 往返

这个 pass 不是为了“美化 IR”，而是为了避免后面的资源图把伪边界也当成真的布局边界。

### `tb-build-mainloop-graph`

作用：

- 生成 `tb.buffer_model`

这一步把主线变成资源图。后面的 loop/latency/schedule/async 都统一从这里出发。

### `tb-regularize-k-loop`

作用：

- 生成 `tb.loop_plan`

把 stage1 主线规整成单一 K-loop owner。

### `tb-assign-latencies`

作用：

- 生成 `tb.latency_plan`

把 op 延迟真相显式化，为后续 schedule 提供输入。

### `tb-schedule-loops`

作用：

- 生成 `tb.pipeline_plan`

决定 coarse 的 stage/cluster/order 布局。

### `tb-derive-waits`

作用：

- 生成 `tb.async_plan`

明确：

- 哪些 producer 归哪个 async group
- 哪些 wait 插在谁前面
- 哪些 slot 何时能复用

### `tb-expand-pipeline`

作用：

- 生成 `tb.pipeline_expansion`

把 coarse schedule 展开成 lowering 真正能吃的 cluster 序列。

### `tb-cleanup-pipeline`

作用：

- 校验 `tb.pipeline_expansion`
- 生成 `tb.pipeline_mainline`
- 生成 `tb.pipeline_ready`
- 删除只在 rewrite/schedule 阶段临时使用的 attr

这是 stage1 里非常关键的边界：

- `tb.matmul` 到这里结束
- `tb.pipeline_mainline` 成为显式 lowering owner

### `tb-lower-pipeline-to-nvgpu`

作用：

- 把 `tb.pipeline_mainline` 和公开合同 lower 成 `gpu/nvgpu` IR

它消费的是公开合同：

- `tb.program_mapping_plan`
- `tb.encoding_plan`
- `tb.transport_plan`
- `tb.buffer_model`
- `tb.async_plan`
- `tb.accumulator_plan`
- `tb.epilogue_plan`

它不应该重新推断隐藏 pipeline 骨架。

### `tb-lower-epilogue-vector-io`

作用：

- 把 late direct C path 的
  `tb.epilogue_global_vector_load/store`
  收口成稳定的 memref vector 访问

这个 pass 的意义是：

- 避免最后 NVVM sink 把 direct global vector path 静默标量化

### NVGPU/NVVM/LLVM sink pipeline

作用：

- 从 repo-native NVGPU IR 落到 NVVM/LLVM/PTX

这个阶段做的是 target lowering，不应该再改上层执行语义。

## 6. 为什么这里要有 `tb.pipeline_mainline`

这层设计是当前仓库最重要的思想之一。

如果没有 `tb.pipeline_mainline`，那么前面虽然有一堆 attr：

- `tb.pipeline_plan`
- `tb.async_plan`
- `tb.pipeline_expansion`

但真正到 lowering 时，后端还是很容易再偷偷重建：

- stage/cluster 顺序
- wait 插入位置
- barrier 聚合方式
- k-group 归属

一旦 lowering 自己重建这些东西，就会重新回到“上层合同说一套、后端执行另一套”的老问题。

所以这里选择在 cleanup 之后直接生成显式 IR：

- `tb.async_issue_cluster`
- `tb.consumer_wait_cluster`
- `tb.mma_compute_cluster`

这样 `TBLowerPipelineToNVGPU` 做的是翻译，不是再发明一个隐藏执行器。

## 7. 推荐的阅读顺序

如果你是第一次读这个仓库，建议按下面顺序看：

1. `README.md`
2. `include/tb/IR/TBOps.td`
3. `include/tb/Transforms/Passes.td`
4. `include/tb/Analysis/KernelConfig.h`
5. `include/tb/Analysis/TargetInfo.h`
6. `include/tb/Analysis/MatmulSemantics.h`
7. `include/tb/Analysis/ProgramMappingPlan.h`
8. `include/tb/Analysis/EncodingPlan.h`
9. `include/tb/Analysis/TransportPlan.h`
10. `include/tb/Analysis/AccumulatorPlan.h`
11. `include/tb/Analysis/EpiloguePlan.h`
12. `include/tb/Analysis/MatmulRewritePlan.h`
13. `include/tb/Analysis/BufferModel.h`
14. `include/tb/Analysis/LoopPlan.h`
15. `include/tb/Analysis/LatencyPlan.h`
16. `include/tb/Analysis/PipelinePlan.h`
17. `include/tb/Analysis/AsyncPlan.h`
18. `include/tb/Analysis/PipelineExpansion.h`
19. `lib/Transforms/CleanupPipeline.cpp`
20. `lib/Transforms/LowerPipelineToNVGPU.cpp`
21. `lib/Transforms/Stage1Pipelines.cpp`

## 8. 一句话总结

这个仓库当前 stage1 的核心不是“实现了多少优化技巧”，而是先把 Triton 风格 matmul 主线里真正需要公开拥有的真相拆清楚：

- 语义谁拥有
- layout 谁拥有
- transport 谁拥有
- accumulator/epilogue 谁拥有
- buffer graph 谁拥有
- wait frontier 谁拥有
- pipeline mainline 谁拥有

只有这些 owner 真正固定住，后面的性能优化和 Triton 对齐才不会反复回退到“后端临时猜出来的近似实现”。
