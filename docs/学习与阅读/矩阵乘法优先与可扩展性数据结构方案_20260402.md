# `mini_triton_nvgpu_v1` 矩阵乘法优先、保留扩展性的最终数据结构方案

## 1. 文档目的

这份文档回答的是下面这个问题：

- 当前 `mini_triton_nvgpu_v1` 已经完成 public contract 收口与无桥接重构之后，
- 如果目标是**先把矩阵乘法这个单 kernel 的 Triton 式优化做好**，
- 但又不希望后面为了扩展性再推翻数据结构，
- 那么现在的数据结构应该如何冻结，哪些地方还必须补齐。

本文只讨论：

- 当前结构是否已经够支撑 matmul 主线优化；
- 哪些结构已经可以定为稳定骨架；
- 哪些字段和层次还必须补，才能避免以后重构；
- 后续扩展应该加在哪里，而不是散落到各层。

本文不讨论：

- 具体性能问题怎么调；
- 具体 schedule 算法怎么写；
- 具体 lowering 序列怎么写；
- 当前某个 bug 的局部修补。

---

## 2. 最终结论

结论分成两句：

1. **对当前单 kernel matmul 主线来说，现有数据结构已经够了。**
2. **对“以后不再为 Triton 式 matmul 优化重构结构”来说，现有数据结构还差少量关键扩展。**

也就是说：

- 现在不应该再推翻 `EncodingPlan -> AccumulatorPlan -> BufferModel -> PipelinePlan -> AsyncPlan -> EpiloguePlan` 这条骨架；
- 现在应该做的是：在这条骨架上，补齐少量 matmul 必需但尚未显式建模的 contract。

所以后续路线不是：

- 再来一轮大重构；

而是：

- **冻结主骨架**
- **定点扩容**
- **继续把单 kernel matmul 做强**

---

## 3. 当前结构已经足够支撑什么

从当前真实代码看，下面这些能力已经有稳定结构承载：

### 3.1 Layout / Fragment / Shared 物理真相

`EncodingPlan` 已经可以承载：

- global blocked layout
- shared layout
- dot operand layout
- accumulator layout
- fragment instruction shape / repeat shape / logical shape
- shared async legality

对应文件：

- [EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h)

这意味着：

- exact-tile matmul 的 layout contract 已经不是问题；
- `ldmatrix` / `mma.sync` 主线所需的结构骨架已经成立。

### 3.2 Register / C pack 真相

`AccumulatorPlan` 已经独立表达：

- lane access pattern
- accumulator packs
- registers per warp
- accumulator encoding owner

对应文件：

- [AccumulatorPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h)

这意味着：

- C accumulator register topology 已经独立；
- 不需要再通过旧 `CRegisterPlan` 回推主语义。

### 3.3 Resource graph / ownership 真相

`BufferModel` 已经表达：

- backing
- view
- value
- pipeline op

并且 owner truth 已经统一到 concrete backing/view/value 上。

对应文件：

- [BufferModel.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h)

这意味着：

- shared/register ownership 已经可表达；
- multistage slot reuse 也已经有结构落点；
- 不再需要旧 `MainloopGraph` 做 owner。

### 3.4 Mainloop schedule / overlap 真相

`PipelinePlan` 和 `AsyncPlan` 已经表达：

- op placement
- stage/cluster/order
- stage-owned shared buffers
- async producer
- async group
- first-use wait
- reuse fence

对应文件：

- [PipelinePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h)
- [AsyncPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h)

这意味着：

- `cp.async + wait + barrier + ldmatrix + mma.sync` 这条严格 mainloop 主线已经有稳定 owner；
- 当前单 kernel matmul 的 async overlap 主问题已经有结构承载。

### 3.5 C landing 真相

`EpiloguePlan` 已经把：

- `direct_global_vector`
- `shared_relay`

分开表达。

对应文件：

- [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)

这意味着：

- direct C path 和 relay C path 已经不再混为一谈；
- 当前单 kernel matmul 的 epilogue owner 也是成立的。

---

## 4. 现在还不够的地方

不够的地方不是“主骨架错了”，而是“还有几类 matmul 优化没有专门 contract”。

这些缺口主要有四类。

### 4.1 Program mapping 没有独立 contract

当前 `KernelConfig` 只表达：

- `blockM/N/K`
- `numWarps`
- `requestedStages`
- `mmaKind`
- dtype
- `exactTile`

对应文件：

- [KernelConfig.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h)

这还不足以承载后面 Triton matmul 常见的：

- `GROUP_M`
- CTA swizzle
- persistent matmul
- split-K
- cluster launch

这些东西如果继续硬塞进 lowering，后面一定会重新改结构。

### 4.2 Hardware/resource model 太薄

当前 `TargetInfo` 只有：

- `sm_86`
- warp size
- async copy 基本能力
- `mmaInstrShape`

对应文件：

- [TargetInfo.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h)

它还表达不了：

- register budget
- shared memory budget
- CTA occupancy 上限
- SM 数量
- 不同 async transport 能力
- `mbarrier` / TMA / WGMMA 等后续能力

这意味着：

- 现在够做当前 V1；
- 但不够做“matmul 优化选择器”。

### 4.3 Async contract 还只覆盖 `cp.async`

当前 `AsyncPlan` 只有：

- `CpAsync`
- `SyncCopyFallback`

它还没有一个更一般的“transport contract”层次。

这意味着：

- 现在表达 `cp.async` 足够；
- 以后如果要支持别的 async transport，会面临再次改 `AsyncPlan` 语义的问题。

### 4.4 Epilogue contract 还只覆盖 landing，不覆盖表达式

当前 `EpiloguePlan` 只表达：

- init mode
- store mode
- direct/relay payload

它没有表达：

- bias
- convert
- activation
- clamp
- reduction merge

这意味着：

- 现在做纯 matmul store 足够；
- 但只要做 fused epilogue，就还会再改。

---

## 5. 当前阶段不应该做什么

为了“保留扩展性”，最容易犯的错是过度泛化。

当前明确不建议做下面这些事：

### 5.1 不要把 `BufferModel` 改成通用算子图

`BufferModel` 现在的职责是：

- 承载 matmul mainloop 的 resource graph；

它不应该现在就变成：

- 通用融合图 IR；
- 通用算子执行图；
- 通用 SSA 替代物。

否则你会把 matmul 主线复杂度和未来通用化复杂度重新绑死。

### 5.2 不要单独引入巨型 `ResourcePlan`

当前硬件/resource 信息不足，这是事实。

但正确做法是：

- 扩 `TargetInfo`

而不是马上再加一个巨型 `ResourcePlan`。

原因是 matmul 现在需要的只是：

- 预算信息
- 能力信息

还没有到需要独立资源规划图的阶段。

### 5.3 不要为了未来 TMA/WGMMA 现在就重写主链

现在真正要做好的还是：

- `mma.sync`
- `ldmatrix`
- `cp.async`
- multistage
- direct/relay epilogue

如果为了 future features 提前把主链改成“大一统 transport/compute abstraction”，会明显拖慢当前主目标。

### 5.4 不要为了 fusion 先改 `PipelinePlan`

`PipelinePlan` 现在服务的是：

- single-kernel matmul 的 coarse placement / ownership

这时去做通用 scheduler 是错误时机。

---

## 6. 建议的最终策略

最终策略很简单：

- **保留现有八层骨架**
- **只新增一个真正的新 plan**
- **其余需求通过扩现有结构解决**

也就是说，不建议再引入很多新层。

建议的 public contract 终态是：

1. `KernelConfig`
2. `TargetInfo`
3. `ProgramMappingPlan`
4. `EncodingPlan`
5. `AccumulatorPlan`
6. `BufferModel`
7. `PipelinePlan`
8. `AsyncPlan`
9. `EpiloguePlan`
10. `KernelContract`

其中真正新增的一层只有：

- `ProgramMappingPlan`

其余扩展全部落到原有层里。

---

## 7. 为什么只新增 `ProgramMappingPlan`

因为当前缺的最明显、又不适合塞进现有层的，就是：

- program id 到 tile 的映射；
- CTA 顺序；
- persistent / split-K / grouped launch；

这些东西：

- 不属于 `KernelConfig` 的输入本身；
- 不属于 `EncodingPlan` 的物理 layout；
- 不属于 `PipelinePlan` 的 CTA 内调度；
- 但又是 Triton matmul 性能优化里非常关键的一层。

所以它应该独立。

---

## 8. `ProgramMappingPlan` 应该长什么样

### 8.1 职责

`ProgramMappingPlan` 只表达 CTA / program 级别的映射 contract。

它不表达：

- shared layout
- register layout
- async wait
- CTA 内 op placement

### 8.2 建议字段

建议至少包含：

```text
ProgramMappingPlan
  mappingKind
  tileShapeM/N/K
  groupM
  splitK
  launchOrder
  swizzleKind
  persistent
  clusterShape
  reductionMode
```

### 8.3 依赖关系

建议依赖：

- `KernelConfig`
- `TargetInfo`

然后被：

- lowering
- 未来的 split-K/reduction plan

消费。

### 8.4 作用

这样做之后：

- `GROUP_M`
- CTA swizzle
- persistent matmul
- split-K

都有独立归宿，不会污染 `KernelConfig` 或 `PipelinePlan`。

---

## 9. 现有层各自应该怎样扩，不需要新增新层

### 9.1 `KernelConfig` 只加输入参数，不加推导结果

`KernelConfig` 未来可以增加：

- `groupM`
- `splitK`
- `clusterM/N/K`
- `persistent`

但这些都只能是：

- 用户/前端输入配置

不能变成：

- program mapping 结果
- schedule 结果

也就是说：

- `KernelConfig` 负责“想要什么”
- `ProgramMappingPlan` 负责“最终怎么映射”

### 9.2 `TargetInfo` 直接扩成 matmul resource/capability model

建议后续直接在 [TargetInfo.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h) 增：

- `numSms`
- `maxWarpsPerCTA`
- `maxRegistersPerThread`
- `maxRegistersPerCTA`
- `maxSharedBytesPerCTA`
- `supportsMBarrier`
- `supportsTMA`
- `supportsWGMMA`
- `preferredAsyncTransport`

不要单独再拆新 plan。

### 9.3 `EncodingPlan` 继续做唯一 layout 真相

`EncodingPlan` 当前骨架已经足够。

只建议补“未来保留位”，不建议改层次：

- descriptor/tensor-map 相关 encoding
- 更一般的 shared swizzle metadata
- 更一般的 accumulator fragment family

但不建议把它变成：

- 调度 plan
- launch plan
- async plan

### 9.4 `BufferModel` 只补 matmul 必需的语义，不做通用化

建议后续只加这些真正必需的扩展：

- `BufferBacking` 增 element scalar kind
- `BufferBacking` 增 alignment / vectorBytes
- `BufferRole` 预留：
  - `Predicate`
  - `ReductionPartial`
  - `TensorMap`
- `BufferOpKind` 预留：
  - `Convert`
  - `PredicateLoad`
  - `ReductionMerge`

但不建议现在就把它变成任意算子 DAG。

### 9.5 `PipelinePlan` 扩调度标签，不扩成通用 scheduler

建议只加：

- `warpRole`
- `scheduleClass`
- `issueDomain`
- `overlapClass`

这些标签能帮助以后做：

- warp specialization
- 更激进的 producer/consumer overlap

但不建议现在把它改成任意通用调度器 contract。

### 9.6 `AsyncPlan` 在现有层内 generalize

不建议单独新增 `AsyncTransferPlan`。

更合理的做法是直接扩 [AsyncPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h)：

- `AsyncProducerKind` 从
  - `CpAsync`
  - `SyncCopyFallback`

扩成更一般的 transport kind：

- `CpAsync`
- `TMA`
- `BulkCopy`
- `SyncCopyFallback`

并给 `AsyncProducer` 增：

- transport attributes
- barrier kind
- zero-fill / predicate info
- cache policy

这样当前实现仍然只实现 `cp.async`，
但结构不需要以后再推翻。

### 9.7 `EpiloguePlan` 直接内嵌 expression payload

不建议新增单独 `EpilogueExprPlan` 文件层。

更合理的做法是直接扩 [EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)：

- 保留现有 `direct_global_vector / shared_relay`
- 再增加 epilogue expression payload

比如表达：

- `load_bias`
- `add`
- `convert`
- `activation`
- `clamp`
- `store`

这样：

- 单 kernel matmul 仍然是主线；
- fused epilogue 也有稳定入口；
- 不需要把 epilogue 体系拆成新大层。

---

## 10. 建议的终态依赖图

推荐长期维持下面这个依赖图：

```text
tb.matmul
  -> KernelConfig
  -> TargetInfo

KernelConfig + TargetInfo
  -> ProgramMappingPlan

KernelConfig + TargetInfo
  -> EncodingPlan

KernelConfig + TargetInfo + EncodingPlan
  -> AccumulatorPlan

KernelConfig + EncodingPlan + AccumulatorPlan
  -> BufferModel

BufferModel
  -> PipelinePlan

BufferModel + PipelinePlan
  -> AsyncPlan

EncodingPlan + AccumulatorPlan + TargetInfo
  -> EpiloguePlan

KernelContract
  = KernelConfig
  + TargetInfo
  + ProgramMappingPlan
  + EncodingPlan
  + AccumulatorPlan
  + BufferModel
  + PipelinePlan
  + AsyncPlan
  + EpiloguePlan

Lowering
  consumes KernelContract directly
```

这个图里最重要的三条原则是：

1. `ProgramMappingPlan` 不进 CTA 内 resource graph
2. `AsyncPlan` 继续 owner transport/wait/reuse，而不是 layout
3. `EpiloguePlan` 继续 owner C landing/expr，而不是 buffer allocation

---

## 11. 实施顺序

如果要按“先把 matmul 做好，但以后不留结构漏洞”的方式推进，建议顺序固定为：

### Step 1

冻结下面这些层的 owner 边界，不再重写语义：

- `EncodingPlan`
- `AccumulatorPlan`
- `BufferModel`
- `PipelinePlan`
- `AsyncPlan`
- `EpiloguePlan`

### Step 2

新增 `ProgramMappingPlan`。

这是唯一建议新增的 public plan。

### Step 3

扩 `TargetInfo`，补齐 matmul 选择器真正需要的 resource/capability 字段。

### Step 4

扩 `AsyncPlan` 的 transport 表达，但实现层仍然先只做 `cp.async`。

### Step 5

扩 `EpiloguePlan` 的 expression payload，但实现层仍先只做最小 direct/relay path。

### Step 6

最后再逐步做：

- grouped launch
- persistent
- split-K
- fused epilogue

这样每一步都建立在前一步稳定 contract 上，不会反向推翻。

---

## 12. 一句话总结

当前 `mini_triton_nvgpu_v1` 的数据结构，已经足够把**单 kernel matmul 主线优化**做好。

下一步不该再重构主骨架。

下一步该做的是：

- 把当前骨架定住；
- 增一个 `ProgramMappingPlan`；
- 扩 `TargetInfo / AsyncPlan / EpiloguePlan`；
- 继续沿着 matmul 主线做算法和 lowering。

这样才能同时满足两件事：

1. 先把矩阵乘法单核优化做强
2. 不给以后 Triton 式扩展留下结构性返工

