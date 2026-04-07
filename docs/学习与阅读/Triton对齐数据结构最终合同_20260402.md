# `mini_triton_nvgpu_v1` Triton 对齐的数据结构最终定稿方案

## 1. 文档目的

这份文档不是再讨论“要不要重构”，而是给出最终定稿版结论：

- `mini_triton_nvgpu_v1` 的数据结构应该如何一次定死；
- 哪些结构属于 public contract；
- 哪些旧结构必须彻底退到 internal-only；
- Triton 的主要 matmul 优化分别应该落在哪一层；
- 后续算法优化为什么不应该再反向推翻这些结构。

本文只讨论**最终数据结构边界**。

本文不讨论：

- 具体 schedule 算法怎么写；
- 具体 async/wait 算法怎么写；
- 具体 lowering 指令序列怎么写；
- 当前代码里某个 bug 的局部修补办法。

本文目标只有一个：

- 让后续所有优化都只是在既定 contract 上填算法，而不是再改 contract 本身。

---

## 2. 最终结论

`mini_triton_nvgpu_v1` 的最终 public 数据结构必须固定为下面九层：

1. `KernelConfig`
2. `TargetInfo`
3. `EncodingPlan`
4. `AccumulatorPlan`
5. `BufferModel`
6. `PipelinePlan`
7. `AsyncPlan`
8. `EpiloguePlan`
9. `KernelContract`

其中真实依赖方向必须固定为：

`KernelConfig + TargetInfo -> EncodingPlan -> AccumulatorPlan -> BufferModel -> PipelinePlan -> AsyncPlan`

`EncodingPlan + AccumulatorPlan + TargetInfo -> EpiloguePlan`

`Lowering` 只消费：

- `KernelConfig`
- `TargetInfo`
- `EncodingPlan`
- `AccumulatorPlan`
- `BufferModel`
- `PipelinePlan`
- `AsyncPlan`
- `EpiloguePlan`

不允许再有下面这些反向依赖：

- `BufferModel -> EpiloguePlan`
- `PipelinePlan -> LatencyPlan` 作为 public contract 依赖
- `AsyncPlan -> LatencyPlan` 作为 public contract 依赖
- `EpiloguePlan -> PipelinePlan`
- `Lowering -> legacy graph/schedule/wait` 作为结构真相

这就是最终定稿版依赖图。

---

## 3. 相比上一版文档的关键修正

上一版文档的核心方向是对的，但还需要做三处修正，才能成为终稿。

### 3.1 `AccumulatorPlan` 不能放在 `AsyncPlan` 之后

旧文档把总依赖图写成：

`KernelConfig/TargetInfo -> EncodingPlan -> BufferModel -> PipelinePlan -> AsyncPlan -> AccumulatorPlan/EpiloguePlan -> Lowering`

这条链不够严谨。

原因：

- accumulator register topology 不是 async/schedule 的产物；
- 它本质上是 `EncodingPlan + TargetInfo + KernelConfig` 决定的 register contract；
- `BufferModel` 如果要表达 accumulator 相关 backing/view/value，就至少要晚于 `AccumulatorPlan`。

所以终稿里必须改成：

`EncodingPlan -> AccumulatorPlan -> BufferModel`

而不是：

`BufferModel -> ... -> AccumulatorPlan`

### 3.2 `EpiloguePlan` 必须和 `Pipeline/Async` 解耦

`EpiloguePlan` 的职责是：

- 定义 C init mode
- 定义 C store mode
- 定义 direct / relay 的 landing contract

它不是：

- pipeline contract
- wait contract
- backing allocation contract

所以它应该依赖：

- `EncodingPlan`
- `AccumulatorPlan`
- `TargetInfo`

但不应该依赖：

- `PipelinePlan`
- `AsyncPlan`
- 具体 backing id

### 3.3 `LatencyPlan` 不再属于 public 主合同

`LatencyPlan` 可以继续存在，但它的角色只能是：

- internal analysis result
- schedule/async derive 的内部输入

它不能再是：

- public contract 的一层
- lowering 直接消费的主合同
- async 语义的 owner

如果继续把它留在 public 主链里，后面还是会回到“analysis 标记冒充 physical contract”的老问题。

---

## 4. Public Contract 与 Internal Contract 的最终边界

### 4.1 Public Contract

下面这些头文件是最终对外稳定边界：

```text
include/tb/Analysis/KernelConfig.h
include/tb/Analysis/TargetInfo.h
include/tb/Analysis/EncodingPlan.h
include/tb/Analysis/AccumulatorPlan.h
include/tb/Analysis/BufferModel.h
include/tb/Analysis/PipelinePlan.h
include/tb/Analysis/AsyncPlan.h
include/tb/Analysis/EpiloguePlan.h
include/tb/Analysis/KernelContract.h
```

它们的职责是：

- 定义最终 contract；
- 定义 parse/build/validate；
- 被 passes 和 lowering 直接依赖。

这些头文件一旦定稿，后续只允许：

- 扩 enum；
- 扩 payload 字段；
- 扩合法性检查；

不允许再改 ownership 边界和层次关系。

### 4.2 Internal Contract

下面这些旧结构必须全部降级为 internal-only：

- `KernelSpec`
- `LayoutPlan`
- `CRegisterPlan`
- `MainloopGraph`
- `LatencyPlan`
- `SchedulePlan`
- `WaitPlan`
- `LegacyInterop`

它们可以在迁移阶段继续存在，但只能用于：

- 过渡期桥接；
- 对照验证；
- 局部旧算法复用。

它们不能再：

- 作为 public include 被新 pass 直接依赖；
- 作为 lowering 的外层输入真相；
- 出现在最终结构说明文档的主链里。

---

## 5. 每一层最终应该表达什么

### 5.1 `KernelConfig`

`KernelConfig` 是**输入真相**，不包含任何推导结果。

必须保留的内容：

- problem shape
- tile shape
- dtype
- mma kind
- `numWarps`
- `requestedStages`
- `exactTile`
- 未来的 `numCTAs/splitK/clusterShape` 等输入配置

必须删除的内容：

- `numStages` 和 `requestedStages` 双名共存
- 与 schedule 结果混在一起的任何字段

必须满足的原则：

- `requestedStages` 只表示输入配置
- `scheduledMaxStage` 只出现在 `PipelinePlan`

禁止再出现：

- 一个 struct 同时表达“用户配置 stage 深度”和“调度结果 stage 深度”

### 5.2 `TargetInfo`

`TargetInfo` 是**目标硬件能力真相**。

它必须能稳定表达：

- warp size
- shared bank bytes
- cp.async 能力
- ldmatrix 能力
- mma.sync / wgmma 能力
- descriptor / TMA 能力
- async vector byte 限制
- native mma instruction shapes
- 未来的 bank/swizzle/TMA 约束

它不能是：

- 一个硬编码的 `sm_86` 占位返回值

因为 Triton 的布局、pipeline、epilogue legality 都依赖 target capability。

### 5.3 `EncodingPlan`

`EncodingPlan` 是**唯一布局真相**。

它必须是 encoding entry 表，而不是平面字段大 struct。

必须能表达的 encoding kind：

- `Blocked`
- `DotOperand`
- `SwizzledShared`
- `PaddedShared`
- `NVMMAShared`
- `Accumulator`
- `Linear`
- 未来的 `Descriptor/TensorMap` 相关 encoding

它必须承担的职责：

- A/B/C global layout contract
- shared layout contract
- dot operand fragment layout contract
- accumulator encoding contract
- CTA/warp/value 的线性映射 contract

它不能承担的职责：

- pipeline placement
- wait frontier
- epilogue relay allocation

### 5.4 `AccumulatorPlan`

`AccumulatorPlan` 是**accumulator register topology 真相**。

它必须表达：

- accumulator encoding ref
- registers-per-warp
- lane-to-value mapping
- fragment pack decomposition
- live-across-stages
- accumulator multi-buffer depth

它不能表达：

- direct global vector init/store
- relay scratch
- epilogue mode
- barrier/wait

后续 Triton 优化里，凡是属于：

- register fragment topology
- lane packing
- accumulator persistence

都应该落在这一层。

### 5.5 `BufferModel`

`BufferModel` 是**真实资源图**。

这是整个结构系统里最关键的一层。

它必须只表达：

- `BufferBacking`
- `BufferView`
- `ValueState`
- `PipelineOp`

它必须能稳定承载：

- multibuffer backing
- stage slice view
- tile/fragment/pack view
- alias group
- overwrite owner
- global/shared/register/descriptor/relay 等不同资源

#### `BufferBacking` 最终语义

必须表达：

- resource identity
- resource role
- memory space
- encoding ref
- logical shape
- alloc shape
- depth
- alias group
- stage-indexed 与否

未来允许扩：

- bank/swizzle metadata
- descriptor metadata
- barrier backing metadata

#### `BufferView` 最终语义

必须表达：

- concrete `viewId`
- `backingId`
- `ViewKind`
- concrete stage
- concrete buffer index
- encoding ref
- offsets
- shape

关键原则：

- `ValueState.ownerView` 必须指向真实 `viewId`
- exact-tile multistage shared value 的 owner 必须是 stage slice view
- 不能再让 owner 偷偷回退成旧 slot id

#### `ValueState` 最终语义

必须表达：

- value kind
- defining op
- users
- owner view
- loop-carried distance
- bundle
- use frontier ordinal

如果后续需要：

- top-level first use
- top-level last use
- loop-carried class

都允许继续扩在这里。

#### `PipelineOp` 最终语义

`PipelineOp` 不能再以 exact-tile 特化字段作为核心 schema。

最终必须改成：

- `kind`
- `inputs`
- `outputs`
- `iterationCoords`

其中 `iterationCoords` 统一表达：

- k-group
- m-tile
- n-tile
- warp-group
- cluster-local coords

而不是在核心 struct 里固化：

- `kGroup`
- `mTile`
- `nTile`
- `nPair`
- `half`
- `accIndex`

这些 exact-tile 字段可以作为过渡期派生视图存在，但不能再是核心 public schema。

### 5.6 `PipelinePlan`

`PipelinePlan` 是**唯一调度结果真相**。

它必须表达：

- `scheduledMaxStage`
- op placement
- stage-owned view/buffer slice

它不能再表达：

- 输入 stage 深度
- latency analysis 结果
- wait 语义

`StageBufferUse` 最终必须升级成真正的 stage ownership record，至少包含：

- concrete stage-owned `viewId`
- 对应 `backingId`
- `bufferIndex`
- `producerOp`
- first consumer frontier
- overwrite frontier

关键点：

- 不能再用 `producerOp = -1` 占位
- 不能只写“这个 backing 有几个 stage”
- 必须能回答“哪个 stage 对应哪个具体 view，谁生产，谁复用”

### 5.7 `AsyncPlan`

`AsyncPlan` 是**唯一 async/wait/reuse 真相**。

它必须表达：

- async producer
- async group
- wait frontier
- reuse fence

#### `AsyncProducer`

必须表达：

- producer op
- producer kind
- produced value
- source view
- destination view
- group id
- vec bytes
- legality

关键点：

- `dstView` 必须是 concrete stage slice view
- 不能再沿用 full buffer view 或旧 slot

#### `WaitInfo`

必须表达：

- group id
- before op
- required stage/cluster/order
- barrier needed or not

关键点：

- wait 必须锚到 first-use frontier
- wait 服务的是 async producer contract，不是一般 value lifetime contract

#### `ReuseFence`

必须表达：

- backing/view reuse frontier
- retiring value
- acquiring value
- after op
- required frontier position

关键点：

- reuse fence 的主语必须是 backing/view
- 不能退回“slot reuse”

### 5.8 `EpiloguePlan`

`EpiloguePlan` 是**唯一 C init/store/relay landing 真相**。

它必须保持 direct / relay 类型互斥。

它应该表达：

- init mode
- store mode
- direct-global-vector payload
- shared-relay payload
- future convert/reduction payload

#### `DirectGlobalVectorPlan`

应该只表达：

- packs
- vector width
- 未来可能的 convert policy

不应该携带：

- relay scratch 字段
- shared backing 字段

#### `SharedRelayPlan`

应该表达：

- relay encoding
- logical shape
- alloc shape
- pack decomposition
- 未来的 vector/store policy

不应该表达：

- concrete `relayBacking` id

原因很简单：

- backing id 属于资源图和 lowering 分配层；
- `EpiloguePlan` 只应该决定“要什么 relay contract”，不应该决定“分配到哪个具体 backing id”。

### 5.9 `KernelContract`

`KernelContract` 只负责聚合 public contract。

它最终应该只聚合：

- `KernelConfig`
- `TargetInfo`
- `EncodingPlan`
- `AccumulatorPlan`
- `BufferModel`
- `PipelinePlan`
- `AsyncPlan`
- `EpiloguePlan`

不应该再聚合：

- `LatencyPlan`

---

## 6. Triton 主要优化分别落在哪一层

为了不给以后留漏洞，必须先把“某种优化属于哪一层”写死。

### 6.1 Shared swizzle / padding / mma shared

落在：

- `EncodingPlan`

### 6.2 Dot operand fragment decomposition

落在：

- `EncodingPlan`

### 6.3 Accumulator register layout / lane packing

落在：

- `AccumulatorPlan`

### 6.4 Multistage shared ring / single-buffer stage slice

落在：

- `BufferModel`

### 6.5 Coarse schedule / stage placement / cluster placement

落在：

- `PipelinePlan`

### 6.6 CpAsync / future TMA producer / wait frontier

落在：

- `AsyncPlan`

### 6.7 C direct vector init/store

落在：

- `EpiloguePlan`

### 6.8 C shared relay landing

落在：

- `EpiloguePlan`

### 6.9 硬件能力 gating

落在：

- `TargetInfo`

### 6.10 lowering 中的具体 IR/SASS 翻译

落在：

- `Lowering`

但 lowering 只能消费 contract，不能再反向决定 contract。

---

## 7. 最终必须删掉的漏洞

如果目标是“以后优化不再推翻结构”，下面这些漏洞必须一次删掉。

### 7.1 `KernelConfig = KernelSpec` alias

必须删除。

原因：

- alias 会让旧字段命名和旧语义继续泄漏进新 public contract；
- 这会让“过渡期桥接”永久化。

### 7.2 `numStages/requestedStages` 双名 union

必须删除。

最终 public 字段名只允许：

- `requestedStages`

### 7.3 `BufferModel` 反向依赖 `EpiloguePlan`

必须删除。

原因：

- `BufferModel` 是资源图；
- `EpiloguePlan` 是 landing mode；
- 资源图不能反向依赖 landing mode。

### 7.4 `PipelinePlan` / `AsyncPlan` public 依赖 `LatencyPlan`

必须删除。

原因：

- `LatencyPlan` 是内部分析；
- `PipelinePlan/AsyncPlan` 是 public contract；
- 不能把 analysis 主语义泄漏进 public contract。

### 7.5 `ValueState.ownerView` 指向旧 slot/full-buffer

必须删除。

最终必须保证：

- shared multistage producer value 指向 concrete stage slice view
- reuse frontier 也引用 backing/view，而不是 slot

### 7.6 `StageBufferUse.producerOp = -1`

必须删除。

这类占位字段说明 ownership 还没有真正立住。

### 7.7 `SharedRelayPlan.relayBacking`

必须删除。

这会把具体资源分配泄漏到 epilogue mode contract。

### 7.8 `PipelineOp` exact-tile 私有字段作为核心 schema

必须删除。

否则以后一做 general shape / warp-group / descriptor path，就还得回头改这个 struct。

### 7.9 Lowering 全量转回 legacy contract 才能工作

这是迁移阶段可以接受的，但不是终态。

终态要求：

- legacy lowering 只能作为短期桥接；
- 最终 lowering 必须直接消费 public contract。

---

## 8. 最终验证器要求

每一层都必须有独立 `validate` 逻辑。

这不是可选项。

### 8.1 `KernelConfig` 验证

验证：

- problem shape 合法
- tile shape 合法
- `requestedStages > 0`
- dtype/mma kind 匹配

### 8.2 `TargetInfo` 验证

验证：

- target capability 自洽
- mma shape 与 capability 对齐
- async vector byte 范围自洽

### 8.3 `EncodingPlan` 验证

验证：

- root refs 有效
- encoding kind 与 payload 匹配
- shared/dot/accumulator 的 parent relation 合法

### 8.4 `AccumulatorPlan` 验证

验证：

- accumulator encoding 存在
- lane mapping 与 pack decomposition 自洽
- multi-buffer depth 合法

### 8.5 `BufferModel` 验证

验证：

- backing/view 引用合法
- `ownerView` 指向存在 view
- stage-indexed backing 的 stage slice 完整
- alias group / overwrite ownership 自洽

### 8.6 `PipelinePlan` 验证

验证：

- every op has placement
- no duplicate placement
- `scheduledMaxStage` 与 placement 一致
- stage-owned view 是 concrete ownership，不是占位

### 8.7 `AsyncPlan` 验证

验证：

- producer/group/wait/fence 引用合法
- wait frontier 锚到 scheduled placement
- `dstView` 是 concrete async landing view
- reuse fence 锚到 backing/view reuse frontier

### 8.8 `EpiloguePlan` 验证

验证：

- direct / relay payload 与 mode 一致
- direct path 不带 relay 字段
- relay path 不带 direct-only 字段
- payload 中不出现 concrete backing id

### 8.9 `KernelContract` 验证

验证：

- 所有子 plan 的 cross-reference 一致
- lowering 不需要再补猜字段

---

## 9. 一次性实施顺序

如果要彻底定死结构，实施顺序必须固定：

1. `KernelConfig` 独立化，删 alias 与双名 union
2. `TargetInfo` 真正 target 化
3. `LatencyPlan` 从 public contract 退场
4. 重写 `AccumulatorPlan` 为原生 derive，不再从旧 `CRegisterPlan` 反推
5. 重写 `BufferModel` 为原生 derive，不再从旧 `MainloopGraph` 反推
6. 重写 `PipelinePlan`，使 stage ownership 指向 concrete stage slice view
7. 重写 `AsyncPlan`，使 producer/wait/reuse 全部锚到 backing/view
8. 收紧 `EpiloguePlan`，去掉 concrete relay backing id
9. `KernelContract` 移除 `LatencyPlan`
10. lowering 改为直接消费 public contract
11. 最后才删除 legacy 结构或把它们完全迁入 internal/legacy

这 11 步做完，数据结构才算真正定稿。

---

## 10. 最终验收标准

这套终稿是否完成，只看下面几条。

### 10.1 Public 头文件验收

public include 里只剩：

- `KernelConfig`
- `TargetInfo`
- `EncodingPlan`
- `AccumulatorPlan`
- `BufferModel`
- `PipelinePlan`
- `AsyncPlan`
- `EpiloguePlan`
- `KernelContract`

### 10.2 Public contract 验收

下面这些内容全部不能再出现在 public contract 中：

- `KernelSpec`
- `LayoutPlan`
- `CRegisterPlan`
- `MainloopGraph`
- `LatencyPlan`
- `SchedulePlan`
- `WaitPlan`

### 10.3 Ownership 验收

必须满足：

- shared multistage owner 是 concrete stage slice view
- wait/reuse 的主语是 backing/view
- direct/relay 互斥
- lowering 不再猜 ring depth / wait frontier / relay mode

### 10.4 可扩展性验收

后续如果新增下面任一种优化：

- 新 shared encoding
- 新 async producer kind
- 新 epilogue mode
- 新 target capability
- 新 general-shape path

都只能是：

- 扩 enum
- 扩 payload
- 扩 derive/validate

而不需要再重构 ownership 边界和层次关系。

---

## 11. 一句话总结

最终要定死的，不是“几个 struct 长什么样”，而是下面三件事：

- public contract 里不能再有 legacy 真相泄漏；
- owner 必须彻底落在 `backing/view`；
- lowering 只能翻译 public contract，不能重新发明 contract。

只要这三件事做死，`mini_triton_nvgpu_v1` 后续要继续对齐 Triton 的主要性能优化，就不会再因为数据结构边界错误而反复返工。
