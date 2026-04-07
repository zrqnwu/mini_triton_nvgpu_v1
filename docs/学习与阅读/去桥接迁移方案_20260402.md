# `mini_triton_nvgpu_v1` 无桥接终态与下一步迁移方案

## 1. 文档目的

这份文档不是再讨论 public contract 应该长什么样。

那部分已经在：

- [mini_triton_v1_triton_aligned_data_structure_final_contract_20260402.md](/home/zhangruiqi/docs/mini_triton_v1_triton_aligned_data_structure_final_contract_20260402.md)

里定稿了。

本文只回答四件事：

1. 为什么现在还保留过渡实现
2. 当前代码里的真实结构关系图是什么
3. 哪些地方还在用 internal bridge，是否有冲突
4. 下一步如何把 bridge 一次次删掉，走到最终无桥接状态

---

## 2. 当前结论

当前 `mini_triton_nvgpu_v1` 已经完成的是：

- public contract 收口
- public attr 不再泄漏 `LatencyPlan`
- `KernelConfig` 已独立，不再是 `KernelSpec` alias
- `BufferModel / PipelinePlan / AsyncPlan / EpiloguePlan` 的 owner truth 已经基本站住

当前还没完成的是：

- internal algorithm/lowering 彻底脱离 legacy bridge

所以现在的真实状态是：

- **结构边界已经正确**
- **实现层还没有完全换芯**

这就是为什么现在还存在过渡实现。

---

## 3. 为什么还要有过渡实现

原因很简单：

- 这次完成的是“数据结构去根”
- 不是“所有算法和 lowering 一次重写”

如果在这一步同时把 bridge 全删掉，就等于要把下面几件事一次全重写：

1. `BufferModel` 主图构造
2. `PipelinePlan` 调度算法
3. `AsyncPlan` wait/reuse 算法
4. `Lowering` 到 NVGPU 的主执行器

这四件事虽然都依赖同一套 public contract，但它们是四个不同问题：

- 资源建模问题
- 调度问题
- 同步与复用问题
- 代码生成问题

如果把它们和 public contract 收口绑在同一轮里做，会重新把“结构错”和“算法错”混在一起，结果就是很难定位问题。

所以当前保留过渡实现的意义是：

- 先把 public contract 定死
- 再逐层替换 internal bridge
- 每替换一层，都还能继续编译和验证

这不是结构漏洞。

这是**受控技术债**。

---

## 4. 最终目标图

这是应该长期维持的**终态依赖图**。

```text
tb.matmul
  ├─ getKernelConfig() -> KernelConfig
  └─ getTargetInfo()   -> TargetInfo

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
  + EncodingPlan
  + AccumulatorPlan
  + BufferModel
  + PipelinePlan
  + AsyncPlan
  + EpiloguePlan

Lowering
  consumes KernelContract directly
```

这个图里有几个关键原则：

- `AccumulatorPlan` 在 `BufferModel` 之前
- `EpiloguePlan` 不反向依赖 `BufferModel/Pipeline/Async`
- `LatencyPlan` 不在 public 主链里
- `Lowering` 不再把 public contract 反解成 legacy graph/schedule/wait

---

## 5. 当前代码里的实际关系图

当前代码真实关系不是纯终态，而是“public contract + internal bridge”的双层结构。

### 5.1 Public 层

```text
KernelConfig
TargetInfo
EncodingPlan
AccumulatorPlan
BufferModel
PipelinePlan
AsyncPlan
EpiloguePlan
KernelContract
```

这些是对外稳定边界。

对应文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelContract.h`

### 5.2 Internal 过渡层

```text
EncodingPlan <-> LayoutPlan
AccumulatorPlan/EpiloguePlan <-> CRegisterPlan
BufferModel <-> MainloopGraph
PipelinePlan <-> SchedulePlan
AsyncPlan <-> WaitPlan
KernelConfig -> internal LatencyPlan
KernelContract -> legacy internal forms -> NVGPU lowering
```

对应文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LegacyInterop.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LegacyInterop.cpp`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LayoutPlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/CRegisterPlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MainloopGraph.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/SchedulePlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/WaitPlan.h`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LatencyPlan.h`

---

## 6. 每层会被什么地方使用

### 6.1 `KernelConfig`

用途：

- 代表 kernel 输入配置真相
- 驱动 encoding / accumulator / buffer / internal latency / lowering legality

当前使用点：

- `BuildLayoutPlan`
- `BuildCRegisterPlan`
- `BuildMainloopGraph`
- `ScheduleLoops`
- `DeriveWaits`
- `LowerPipelineToNVGPU`

### 6.2 `TargetInfo`

用途：

- 表示硬件能力真相
- 决定 shared async legality、mma/ldmatrix 能力、vector byte 约束

当前使用点：

- `deriveEncodingPlan`
- `deriveAccumulatorPlan`
- `deriveEpiloguePlan`

### 6.3 `EncodingPlan`

用途：

- 唯一 encoding 真相
- 表达 global/shared/dot/accumulator layout

当前使用点：

- `AccumulatorPlan`
- `BufferModel`
- `EpiloguePlan`
- `Lowering`

### 6.4 `AccumulatorPlan`

用途：

- 唯一 accumulator register topology 真相
- 表达 lane access / packs / register contract

当前使用点：

- `BufferModel`
- `EpiloguePlan`
- `Lowering`

### 6.5 `BufferModel`

用途：

- 唯一资源图真相
- 表达 backing/view/value/op

当前使用点：

- `PipelinePlan`
- `AsyncPlan`
- `Lowering`

### 6.6 `PipelinePlan`

用途：

- 唯一 stage/cluster/order 真相
- 表达 concrete stage ownership

当前使用点：

- `AsyncPlan`
- `Lowering`

### 6.7 `AsyncPlan`

用途：

- 唯一 async producer / wait / reuse 真相

当前使用点：

- `Lowering`

### 6.8 `EpiloguePlan`

用途：

- 唯一 C init/store/relay contract

当前使用点：

- `Lowering`

### 6.9 `KernelContract`

用途：

- public contract 聚合器
- 给 lowering 一次性消费

当前使用点：

- `LowerPipelineToNVGPU`

---

## 7. 当前还存在的 internal bridge 点

这些地方是现在还没彻底去掉的 bridge。

### 7.1 `BufferModel` 仍通过 legacy graph 构造

当前位置：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/BufferModel.cpp`

当前路径：

```text
EncodingPlan + AccumulatorPlan + KernelConfig
  -> legacy LayoutPlan
  -> legacy CRegisterPlan
  -> legacy MainloopGraph
  -> BufferModel
```

问题性质：

- public 结构已经正确
- 但资源图生成算法还没 native 化

### 7.2 `PipelinePlan` 仍复用 legacy schedule

当前位置：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelinePlan.cpp`

当前路径：

```text
BufferModel
  -> legacy MainloopGraph
  -> internal LatencyPlan
  -> legacy SchedulePlan
  -> PipelinePlan
```

问题性质：

- `PipelinePlan` 的 schema 已经正确
- 但 scheduler 还不是直接跑在 `BufferModel` 上

### 7.3 `AsyncPlan` 仍复用 legacy wait plan

当前位置：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AsyncPlan.cpp`

当前路径：

```text
BufferModel + PipelinePlan
  -> legacy MainloopGraph
  -> internal LatencyPlan
  -> legacy WaitPlan
  -> AsyncPlan
```

问题性质：

- `AsyncPlan` 的 owner truth 已经改正
- 但 wait/reuse 推导还没 native 化

### 7.4 `Lowering` 仍会把 public contract 转回 legacy internal form

当前位置：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp`

当前路径：

```text
KernelContract
  -> toLegacyLayoutPlan
  -> toLegacyCRegisterPlan
  -> toLegacyMainloopGraph
  -> toLegacySchedulePlan
  -> toLegacyWaitPlan
  -> emitKernelBody(...)
```

问题性质：

- 代码生成能跑
- 但 lowering 还没有直接消费 public contract

---

## 8. 当前是否有结构冲突

### 8.1 已经解决的冲突

这些已经不是问题了：

1. `KernelConfig = KernelSpec` alias
2. `numStages/requestedStages` 双名
3. `LatencyPlan` 暴露成 public contract
4. `BufferModel -> EpiloguePlan` 反向依赖
5. `SharedRelayPlan.relayBacking`
6. `PipelineOp` exact-tile 私有字段作为核心 schema
7. multistage shared value 的 `ownerView` 回退到 slot/full-buffer
8. `StageBufferUse.producerOp = -1`
9. `cp.async` 的 `src_view = -1`

### 8.2 现在剩下的不是“结构冲突”，而是“实现债”

现在还剩的是：

- internal bridge 还在
- legacy derive 还在被复用
- lowering 还不是 direct-contract lowering

这三类问题不会再反向污染 public contract。

所以它们已经不是“结构有漏洞”，而是“实现还没最终收尾”。

---

## 9. 下一步的真正目标

下一步不是再改 public contract。

下一步只有一个目标：

- **删掉 internal bridge，让新结构自己成为算法和 lowering 的真实 owner**

也就是从：

```text
new public contract -> legacy bridge -> algorithm/lowering
```

变成：

```text
new public contract -> native algorithm/lowering
```

---

## 10. 下一步迁移顺序

这部分必须按顺序做。

原则是：

- 后一步可以依赖前一步
- 前一步不能依赖后一步
- 每一步都能单独出效果

### Step 1. `BufferModel` native 化

目标：

- 不再通过 `MainloopGraph` 构造 `BufferModel`

要做的事：

1. 直接在 `BufferModel.cpp` 里 native 生成：
   - `BufferBacking`
   - `BufferView`
   - `ValueState`
   - `PipelineOp`
2. 让 `PipelineOp.iterationCoords` 成为原生生成物
3. 不再先生成 legacy slot/value/op 再转回来

要改的文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/BufferModel.cpp`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LegacyInterop.cpp`

完成标准：

- `deriveBufferModel(...)` 不再调用 `deriveMainloopGraph(...)`
- `fromLegacyMainloopGraph(...)` 只剩 compatibility/debug 价值

### Step 2. `PipelinePlan` native scheduler 化

目标：

- 调度直接跑在 `BufferModel` 上

要做的事：

1. 基于 `BufferModel.ops` 和 `ValueState.users` 直接建立依赖图
2. 直接生成：
   - `placements`
   - `scheduledMaxStage`
   - `stageOwnedBuffers`
3. 内部 latency 仍可保留，但不再绕到 `SchedulePlan`

要改的文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelinePlan.cpp`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LegacyInterop.cpp`
- 如有必要，新建
  `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelineScheduler.cpp`

完成标准：

- `derivePipelinePlan(...)` 不再调用 `toLegacyMainloopGraph(...)`
- `derivePipelinePlan(...)` 不再调用 `deriveSchedulePlan(...)`

### Step 3. `AsyncPlan` native wait/reuse 化

目标：

- wait/reuse 直接从 `BufferModel + PipelinePlan` 推导

要做的事：

1. 直接从 `ValueState.users` 找 first-use frontier
2. 直接从 `ownerView/backing` 找 overwrite/reuse frontier
3. 直接构造：
   - `AsyncProducer`
   - `AsyncGroup`
   - `WaitInfo`
   - `ReuseFence`

要改的文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AsyncPlan.cpp`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LegacyInterop.cpp`

完成标准：

- `deriveAsyncPlan(...)` 不再调用 `deriveWaitPlan(...)`
- `deriveAsyncPlan(...)` 不再调用 `toLegacyMainloopGraph(...)`

### Step 4. `Lowering` 直接消费 public contract

目标：

- `Lowering` 不再把 public contract 反解成 legacy internal form

要做的事：

1. `emitKernelBody(...)` 改成直接读取：
   - `BufferModel.ops`
   - `PipelinePlan.placements`
   - `AsyncPlan.waits/reuseFences`
   - `AccumulatorPlan`
   - `EncodingPlan`
2. 把现在的这些桥接删除：
   - `toLegacyLayoutPlan(...)`
   - `toLegacyCRegisterPlan(...)`
   - `toLegacyMainloopGraph(...)`
   - `toLegacySchedulePlan(...)`
   - `toLegacyWaitPlan(...)`
3. 如果 lowering 还需要一些 exact-tile helper，就直接做成 public-contract helper，不再借 legacy struct

要改的文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp`
- 如有必要，新建：
  - `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LoweringHelpers.cpp`
  - `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Transforms/LoweringHelpers.h`

完成标准：

- lowering 不再调用任何 `toLegacy*`

### Step 5. 删除 internal legacy 真相

目标：

- 让 bridge 从“存在但不用”变成“彻底消失”

要做的事：

1. 删除或降级这些结构：
   - `MainloopGraph`
   - `SchedulePlan`
   - `WaitPlan`
   - `LegacyInterop`
2. `KernelSpec / LayoutPlan / CRegisterPlan / LatencyPlan`
   只保留仍然确实需要的 internal 算法壳
3. 如果已经没有实际使用，就继续删除

完成标准：

- `rg "toLegacy|fromLegacy|LegacyInterop"` 不再命中主路径

---

## 11. 推荐的实施阶段

如果按风险最小来做，建议分成三轮。

### Phase A. 先去掉分析侧 bridge

包含：

- Step 1
- Step 2
- Step 3

结果：

- public contract 已经不仅是“展示层”
- 它本身就成为 analysis 的真实 owner

### Phase B. 再去掉 lowering bridge

包含：

- Step 4

结果：

- public contract 直接进入 NVGPU lowering

### Phase C. 最后删 dead legacy code

包含：

- Step 5

结果：

- 项目彻底进入无桥接终态

---

## 12. 每一步怎么验证

### Step 1 验证

- `tb.buffer_model` 输出不变或更强
- `iteration_coords` 保持稳定
- `owner_view` 不退化

### Step 2 验证

- `tb.pipeline_plan` 仍正确生成
- `stage_owned_buffers` 仍是 concrete ownership

### Step 3 验证

- `tb.async_plan` 仍有：
  - concrete `src_view`
  - concrete `dst_view`
  - concrete `view_id`
- 不回退成 slot 语义

### Step 4 验证

- `tb-lower-pipeline-to-nvgpu` 仍能产出合法 GPU/NVGPU IR
- 代码里不再依赖 `toLegacy*`

### Step 5 验证

- 主路径 `rg` 不再出现 legacy bridge API
- build 与 smoke 仍通过

---

## 13. 一句话总结

当前 `mini_triton_nvgpu_v1` 的问题已经不再是“数据结构还有没有根本漏洞”。

现在的问题是：

- **public contract 已经收口**
- **但 internal 算法和 lowering 还没完全迁移到这套 contract 上**

所以下一步不该再改 public 数据结构。

下一步应该做的是：

- 先让 `BufferModel / PipelinePlan / AsyncPlan` 自己成为分析真相
- 再让 `Lowering` 直接消费它们
- 最后彻底删除 legacy bridge

这才是 `mini_triton_nvgpu_v1` 真正的终态收尾路线。
