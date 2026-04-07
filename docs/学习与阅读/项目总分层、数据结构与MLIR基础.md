# 项目总分层、数据结构与MLIR基础

更新时间：2026-04-06

适用项目：

- `/home/zhangruiqi/mini_triton_nvgpu_v1`

适用对象：

- 目前只有基础 `MLIR` 和基础 `CUDA`
- 想先搞清楚项目总共分多少层、每层用什么数据结构、先要会哪些 MLIR 基础

---

## 一、先说结论

按当前仓库主线来看，这个项目严格说可以分成：

- `7 层主线结构`
- 如果把最后的 target lowering / NVGPU-NVVM-PTX sink 单独算一层，就是 `8 层`

主依据：

- [Stage1 数据结构与主线 Pass 讲解](/home/zhangruiqi/mini_triton_nvgpu_v1/docs/stage1_data_structures_and_passes.md)
- [Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)

这个项目的核心原则是：

- 每一层只拥有一类真相
- 后一层只能消费前一层已经公开的合同
- 不允许在 lowering 末端重新猜上层已经决定好的东西

---

## 二、项目总共分多少层

## 第 1 层：入口上下文层

作用：

- 先回答“问题是什么、硬件是什么”

主要数据结构：

- [KernelConfig.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h)
- [TargetInfo.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h)

这层负责的真相：

- `problemM / problemN / problemK`
- `blockM / blockN / blockK`
- `numWarps`
- `requestedStages`
- `exactTile`
- GPU 架构、warp 大小、shared/register 预算、是否支持 `cp.async / ldmatrix / mma.sync`

为什么先有这一层：

- 后面的 layout、transport、epilogue、lowering 全都依赖它
- 如果问题规模和硬件能力都没先公开，后面每层都会变成“边走边猜”

## 第 2 层：语义与映射层

作用：

- 先把 matmul 语义讲清楚
- 再决定 CTA/program 怎么映射

主要数据结构：

- [MatmulSemantics.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulSemantics.h)
- [ProgramMappingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/ProgramMappingPlan.h)

这层负责的真相：

- 这是一个什么 matmul
- 是否 exact-tile
- 哪些维度有 boundary
- `tile / grouped_tile / split-k / persistent` 的映射语义
- CTA 外部 launch 顺序

为什么在 layout/pipeline 之前：

- 先要知道哪个 CTA 干哪块活
- 后面才能谈 CTA 内部的 layout、fragment 和 pipeline

## 第 3 层：布局与搬运层

作用：

- 决定值怎么摆
- 决定值怎么搬

主要数据结构：

- [EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h)
- [TransportPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TransportPlan.h)

这层负责的真相：

- blocked/shared/dot/accumulator encoding
- operand fragment 和 lane / `ldmatrix` 的对应关系
- shared encoding 形态
- `global -> shared` 的 transport kind
- vector bytes / async legality / cache policy

为什么要拆成两层结构：

- `EncodingPlan` 回答“值怎么摆”
- `TransportPlan` 回答“值怎么搬”

这两件事相关，但不是一个问题。

## 第 4 层：C 路径与 matmul 专项重写层

作用：

- 决定 accumulator 怎么组织
- 决定 C 最后怎么落地
- 把前面合同收成真正的 tensor-core mainloop 形态

主要数据结构：

- [AccumulatorPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h)
- [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)
- [MatmulRewritePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulRewritePlan.h)

这层负责的真相：

- accumulator pack 怎么放在寄存器里
- direct-global-vector 还是 shared-relay / shared-pack
- `TargetLandingPlan` 决定最后目标侧 C landing
- mainloop 的 tensor-core 形态

为什么这一层必须在 lowering 之前：

- C path 不能等到 target lowering 才突然决定
- rewrite 时就已经要知道 direct path 是否成立

## 第 5 层：资源图与循环层

作用：

- 把 kernel 变成统一资源图
- 再规整成一个公开的 K-loop owner
- 再标 latency

主要数据结构：

- [BufferModel.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h)
- [LoopPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LoopPlan.h)
- [LatencyPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LatencyPlan.h)

这层负责的真相：

- backing / view / value / op 四张资源表
- 单一 K-loop 结构
- op 的 target latency 和 pipeline 属性

为什么这层重要：

- 后面的 schedule、async、wait 都建立在这份统一资源图上
- 没有 `BufferModel`，后面每层都会各自重建一份局部事实

## 第 6 层：调度、异步与展开层

作用：

- 决定每个 op 放在哪个 stage/cluster/order
- 决定 async producer/group/wait/reuse
- 把抽象 schedule 展开成真正 cluster 序列

主要数据结构：

- [PipelinePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h)
- [AsyncPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h)
- [PipelineExpansion.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelineExpansion.h)

这层负责的真相：

- stage / cluster / order
- async producer
- wait frontier
- expanded cluster stream

为什么顺序是 `loop -> latency -> schedule -> waits -> expansion`：

- 没有 loop，schedule 没对象
- 没有 latency，schedule 没依据
- 没有 schedule，wait 没锚点
- 没有 expansion，cleanup 没法把主线收口成显式 cluster stream

## 第 7 层：cleanup 收口与 lowering 入口层

作用：

- 把真正要 lower 的执行主线收成显式 IR owner
- 为 lowering 准备最小总结合同

主要数据结构与 IR：

- [tb.pipeline_mainline](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td#L60)
- [PipelineReady.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelineReady.h)
- [WarpDecompositionPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/WarpDecompositionPlan.h)
- [ResourceClosurePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/ResourceClosurePlan.h)
- [KernelContract.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelContract.h)

这层负责的真相：

- post-cleanup、pre-lowering 的显式 cluster stream
- lowering 所需的最小 summary
- warp 覆盖和资源预算收口

为什么这层必须存在：

- 不让 `TBLowerPipelineToNVGPU` 在末端重建 pipeline 骨架
- 不让 lowering 再做太多上层决策

## 第 8 层：target lowering / sink 层

作用：

- 消费前面已经公开好的合同
- 翻译到 `gpu / nvgpu / nvvm / ptx`

主要实现文件：

- [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)
- [LowerEpilogueVectorIO.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerEpilogueVectorIO.cpp)

这层理想职责：

- 做翻译
- 不重新决定 layout
- 不重新决定 mapping
- 不重新决定 C landing kind

---

## 三、每层各自会用什么数据结构

如果只想先记一张最简表，可以记这个：

| 层 | 主要结构 |
|---|---|
| 入口上下文层 | `KernelConfig`、`TargetInfo` |
| 语义与映射层 | `MatmulSemantics`、`ProgramMappingPlan` |
| 布局与搬运层 | `EncodingPlan`、`TransportPlan` |
| C 路径与专项重写层 | `AccumulatorPlan`、`EpiloguePlan`、`TargetLandingPlan`、`MatmulRewritePlan` |
| 资源图与循环层 | `BufferModel`、`LoopPlan`、`LatencyPlan` |
| 调度、异步与展开层 | `PipelinePlan`、`AsyncPlan`、`PipelineExpansion` |
| cleanup 与 lowering 入口层 | `tb.pipeline_mainline`、`PipelineReady`、`WarpDecompositionPlan`、`ResourceClosurePlan`、`KernelContract` |
| target lowering 层 | `LowerPipelineToNVGPU`、`LowerEpilogueVectorIO` 消费前面全部公开合同 |

---

## 四、首先你该会哪些 MLIR 相关基础知识

对这个项目来说，你不需要一开始就把 MLIR 学到很深，但下面这些必须先会。

## 1. `Operation / Value / Type / Attribute`

你至少要能看懂：

- 一个 op 有什么 operands
- 一个 op 有什么 results
- attr 和 type 分别在表达什么

因为这个项目最核心的事就是：

- 某些真相放 `attr/type`
- 某些真相放显式 `op`

## 2. 会读 `ODS / TableGen`

你至少要能读：

- [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td)
- [Passes.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Transforms/Passes.td)

你不需要现在就很熟练地写，但至少要会看：

- op 的 arguments / results / verifier
- pass 的 summary / description

## 3. 理解 pass 的几类职责

你至少要能区分：

- verify
- analysis/build-contract
- rewrite
- cleanup/canonicalize
- lowering

如果这点不清楚，你会看不懂为什么这个项目把 pass 拆这么细。

## 4. 知道 module attr 和 op attr 的区别

这个项目里：

- module attr 会挂 target truth
- 一些语义和合同挂在 op attr 上

如果 module-level owner 和 op-level owner 的区别不清楚，后面会很乱。

## 5. 知道 region / block 的基本概念

因为：

- `tb.pipeline_mainline` 有 body region
- body 里还会出现 cluster op

所以你至少要会看 region 结构，不然 cleanup 和 lowering 入口层会很抽象。

## 6. 理解“verifier 不等于 lowering”

这点非常重要。

你要先建立这个观念：

- verifier 是提前检查合同
- cleanup 是收口和清理
- lowering 是翻译，不该重新发明上层真相

这也是为什么项目现在有：

- [tb-verify-pipeline-mainline-contract](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Transforms/Passes.td#L184)

## 7. 理解“谁拥有真相，谁只消费真相”

这是你看这个项目最重要的 MLIR 观念。

你以后看任何一个结构或 pass，都先问：

1. 这层拥有哪类真相
2. 后一层是在消费它，还是又开始猜它

只要你有这个意识，项目再复杂也不会完全乱掉。

---

## 五、最短 MLIR 学习清单

如果你现在时间不多，只先补下面这 6 个点就够开始读仓库：

1. `op / attr / type / value` 基本概念
2. 如何读 `*.td`
3. pass 的 5 种职责：
   - verify
   - analysis/build-contract
   - rewrite
   - cleanup
   - lowering
4. module attr 和 op attr 的区别
5. region / block 的基本概念
6. “谁拥有真相，谁只消费真相” 这套边界观

---

## 六、你现在最先该读哪些文件

如果你想马上开始，推荐优先读这几份：

1. [Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)
2. [Stage1 数据结构与主线 Pass 讲解](/home/zhangruiqi/mini_triton_nvgpu_v1/docs/stage1_data_structures_and_passes.md)
3. [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td)
4. [Passes.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Transforms/Passes.td)
5. [KernelConfig.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h)
6. [EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h)
7. [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)

不要一上来先啃：

- [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

因为那是结果层，不是最适合入门的入口层。

---

## 七、一句话总结

这个项目按主线可以看成：

- `7 层结构`
- 或者算上 target sink 是 `8 层`

你现在最先要会的 MLIR 基础不是“很多 API”，而是：

- 会读 op
- 会读 attr/type
- 会读 pass
- 会区分 verify / analysis / rewrite / cleanup / lowering
- 会用“owner truth”思维看边界

只要这几件事先立住，你后面再顺着主线去读，就不会只是看见一堆代码，而是真的能看出这个项目为什么这么设计。
