# `mini_triton_nvgpu_v1` 面向矩阵乘法优化的重构后数据结构最终文档

## 1. 文档目的

这份文档回答的是下面这组最终问题：

- 如果目标是把 `mini_triton_nvgpu_v1` 做成一个真正能承载 Triton 式矩阵乘法优化的项目，
- 又希望它作为面试项目时结构清楚、ownership 清楚、便于讲解，
- 那么重构后的数据结构到底应该怎么设计；
- 哪些信息应该融入 MLIR 语义层；
- 哪些信息应该继续保留为自定义 plan；
- Triton 自己是怎么设计这些东西的；
- 为什么这里不应该机械复刻 Triton 外形，而应该提炼 Triton 的设计原则。

本文是终稿性质的结构设计文档。

本文不讨论：

- 具体性能调优参数怎么选；
- 某个 pass 的局部 bug 怎么修；
- 某条 lowering 序列怎么写到指令级；
- 某次实测为什么慢。

本文只讨论一件事：

- 重构后，数据结构和 IR 语义层应该怎样一次定稿。

---

## 2. 最终结论

最终结构不应该走两个极端：

- 不能继续让所有真相都停留在 C++ 外部 struct 里；
- 也不应该把所有 plan 都强行改造成 Triton 那样的 type/attr 外形。

最终方案应该是：

1. 把 `layout` 和 `memory legality` 做成 **MLIR 语义层硬真相**。
2. 把 `program mapping / pipeline / async frontier / epilogue strategy` 保留成 **显式 plan 层**。
3. 让 lowering 只消费这两层合同，不再重新推导任何物理真相。

一句话概括：

**Triton 风格的硬真相层 + 你自己的显式计划层。**

这就是最适合 `mini_triton_nvgpu_v1` 的终态。

---

## 3. 什么叫“MLIR 语义层硬真相”

这里的“硬真相”不是指普通 C++ struct，也不是指某个 pass 里顺手算出来的临时信息。

它必须同时满足下面五条：

1. 信息直接存在于 IR 本身。
2. 有明确的 `Attr` / `Type` / `Op` 结构承载。
3. 有 parser / printer / verifier。
4. 所有 pass 和 lowering 都读这一份。
5. lowering 不允许自己再猜一份。

也就是说：

- 如果一个信息只存在于 `deriveXXXPlan()` 的内存结果里，它不是硬真相；
- 如果一个信息挂在 `tb.matmul` 或某个自定义 type/attr 上，并且 verifier 能检查，它才是硬真相。

对这个项目来说，最应该变成硬真相的不是所有东西，而是两类：

1. `layout`
2. `memory legality`

---

## 4. Triton 是怎么设计的

Triton 的核心不是“pass 很多”，而是 ownership 非常清楚。

### 4.1 Triton 的 layout 真相在 encoding attr 里

Triton 用 encoding attr 直接表达：

- blocked layout
- shared layout
- dot operand layout
- mma result layout

可以直接参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)

其中典型定义有：

- `SwizzledSharedEncodingAttr`
- `BlockedEncodingAttr`
- `NvidiaMmaEncodingAttr`
- `DotOperandEncodingAttr`

这意味着 Triton 不会让 lowering 再“猜 shared 怎么排、mma fragment 怎么排、dot operand 怎么排”。

### 4.2 Triton 的 memory legality 真相在 memdesc/type 里

Triton 用 `MemDescType` 表达：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td)

这意味着 Triton 不会让 lowering 再“猜这个 shared/global backing 的 element type、alloc shape、memory space、是否可变”。

### 4.3 Triton 不是把所有东西都做成 type

Triton 并不会把 pipeline、wait frontier、schedule placement 都做成 type。

它真正做成 IR 语义层的，主要是：

- layout
- memory object
- target / numWarps / numCTAs 等全局 contract

而 schedule / pipelining / async grouping 仍主要属于 transform 分析与变换逻辑。

所以 Triton 的原则不是：

- “所有东西都做成 type/attr”

而是：

- “物理真相进入 IR，调度真相留给分析层”

---

## 5. 这个项目为什么不能直接照抄 Triton 外形

如果完全照抄 Triton 外形，会有两个问题。

### 5.1 对项目目标不划算

当前项目目标是：

- 先把矩阵乘法单 kernel 的优化做好；
- 保持结构稳定；
- 保持面试时可讲清楚。

如果现在整体变成 Triton 式完整 IR 体系：

- 工作量会非常大；
- 迁移成本高；
- 很容易变成“表面像 Triton，但真正的优化链没有讲清楚”。

### 5.2 对面试项目表达反而不一定更强

面试项目最强的信号不是：

- “我把 Triton 复刻了一遍”

而是：

- “我理解 Triton 为什么这么做，并且把这些原则提炼成了一个更适合教学、验证、调试的矩阵乘法编译器结构”

因此最终路线应该是：

- 学 Triton 的 ownership；
- 不机械复刻 Triton 的全部外形。

---

## 6. 重构后的最终三层结构

重构后，整个系统固定为三层。

### 6.1 第一层：MLIR 语义层

职责：

- 承载 layout 真相；
- 承载 memory legality 真相；
- 成为所有 pass 和 lowering 的唯一物理语义来源。

这层由下面几类对象组成：

- `tb.matmul`
- `#tb.*` layout attrs
- `!tb.memdesc`

### 6.2 第二层：TB 自定义 plan 层

职责：

- 承载映射策略；
- 承载 schedule 结果；
- 承载 async frontier；
- 承载 epilogue strategy。

这层由下面这些 public contract 组成：

- `KernelConfig`
- `TargetInfo`
- `ProgramMappingPlan`
- `EncodingPlan`
- `AccumulatorPlan`
- `BufferModel`
- `PipelinePlan`
- `AsyncPlan`
- `EpiloguePlan`
- `KernelContract`

### 6.3 第三层：Lowering 层

职责：

- 只消费第一层和第二层；
- 不再重新推导任何 layout / type / vector / alignment / copy size 真相。

---

## 7. 第一层：MLIR 语义层应该长什么样

### 7.1 `tb.matmul` 保持根 op，不必过度复杂化

当前根 op 仍然可以保持简单：

- 输入是 A/B/C memref
- 属性里携带 kernel 入口配置

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td)

这层不需要一下子变成 Triton 那样完整的 tensor IR。

它当前主要负责：

- 给出 kernel 入口；
- 挂载后续 public contract attr；
- 作为严格 matmul V1 的唯一顶层 op。

### 7.2 layout 用自定义 `Attr`

layout 真相不应该继续只是：

- `EncodingKind + payload variant`

最终应该演进为真正的 dialect attr family。

建议新增：

- `#tb.blocked`
- `#tb.shared`
- `#tb.dot_operand`
- `#tb.nvgpu_mma`
- 可选 `#tb.accumulator`

建议新增文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBAttrs.cpp`

示意结构如下。

```text
#tb.blocked<{
  logical_shape = [64, 32],
  size_per_thread = [1, 1],
  threads_per_warp = [32, 1],
  warps_per_cta = [1, 1],
  order = [1, 0]
}>
```

```text
#tb.shared<{
  role = "a",
  logical_shape = [64, 16],
  alloc_shape = [64, 16],
  order = [1, 0],
  vec_bytes = 16,
  per_phase = 1,
  max_phase = 1,
  padding_intervals = [],
  paddings = [],
  transposed = false,
  swizzling_byte_width = 0,
  async_eligible = true,
  async_vec_bytes = 16
}>
```

```text
#tb.nvgpu_mma<{
  mma = "m16n8k16",
  warps_per_cta = [1, 1],
  instr_shape = [16, 8, 16]
}>
```

```text
#tb.dot_operand<{
  operand_index = 0,
  parent = #tb.nvgpu_mma<...>,
  k_width = 1
}>
```

这样做的含义是：

- layout 本体就是 attr；
- `EncodingPlan` 里只负责引用和组织这些 attr；
- lowering 永远不能再造第二套 layout 语义。

### 7.3 memory legality 用自定义 `Type`

最应该新增的 type 是：

- `!tb.memdesc`

建议新增文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBTypes.cpp`

推荐语义如下：

```text
!tb.memdesc<
  shape = [64, 16],
  element = f16,
  encoding = #tb.shared<...>,
  memory_space = shared,
  mutable = true,
  alloc_shape = [64, 16],
  alignment_bytes = 16,
  vector_bytes = 16
>
```

这个 type 的职责非常明确：

- `shape`
- `element type`
- `encoding`
- `memory space`
- `mutable`
- `alloc shape`
- `alignment`
- `vector bytes`

也就是说，下面这些原来分散在多处的东西应该统一收口进 `!tb.memdesc`：

- backing 的 `memorySpace`
- backing 的 `encoding`
- backing 的 `allocShape`
- backing 的 `logicalShape`
- backing 的 `elementScalar`
- backing 的 `alignmentBytes`
- backing 的 `vectorBytes`
- backing 的 `mutableMemory`

这就是“把 memory legality 做成 MLIR 语义层硬真相”。

### 7.4 语义层的 verifier 负责什么

自定义 attr/type 一旦建立，就必须有 verifier。

至少要检查：

- `#tb.shared.vec_bytes > 0`
- `#tb.shared.async_vec_bytes` 合法
- `#tb.dot_operand.parent` 必须是合法的 MMA attr
- `!tb.memdesc.vector_bytes % element_byte_width == 0`
- `!tb.memdesc.alignment_bytes >= vector_bytes`
- `!tb.memdesc.alloc_shape` 与 `shape` rank 匹配
- `!tb.memdesc.memory_space` 与 `encoding` 类型匹配

只有 verifier 真正管住，才叫“硬真相”。

---

## 8. 第二层：Plan 层最终应该怎么设计

### 8.1 `KernelConfig`

继续保留为输入真相：

- `blockM`
- `blockN`
- `blockK`
- `numWarps`
- `requestedStages`
- `mmaKind`
- `aScalar`
- `bScalar`
- `cScalar`
- `exactTile`

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h)

它不应该承载：

- schedule 结果
- async frontier
- 具体 vector width
- derived mapping 结果

### 8.2 `TargetInfo`

继续保留为目标硬件能力真相：

- warp size
- shared bank bytes
- async copy capability
- ldmatrix capability
- mma capability
- async byte limits
- 后续的 TMA / WGMMA / mbarrier capability

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h)

### 8.3 `ProgramMappingPlan`

保留为 plan，不改成 type。

职责：

- CTA / program 映射
- `tileM/N/K`
- `mappingKind`
- `launchOrder`
- `groupM`
- `splitK`
- `persistent`
- `clusterShape`

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/ProgramMappingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/ProgramMappingPlan.h)

原因：

- 它是“映射策略”，不是“内存/布局本体类型”。

### 8.4 `EncodingPlan`

继续保留，但职责要变化。

当前它既像：

- layout registry

又像：

- payload 聚合器

最终应该变成：

- “typed encoding attr 的 registry 和 handle 表”

建议最终 `EncodingEntry` 长成：

```text
EncodingEntry
  name
  encoding_attr
```

而不是：

```text
EncodingEntry
  name
  kind
  payload
```

也就是说：

- attr 自己就是 kind；
- attr 自己就是 payload owner；
- `EncodingPlan` 只负责引用和组织；
- 不再保留第二套 `kind + payload` 解释真相。

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h)

### 8.5 `AccumulatorPlan`

继续保留为 plan。

职责：

- accumulator register topology
- `registersPerWarp`
- `laneAccess`
- `packs`
- accumulator encoding handle

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h)

原因：

- 它不是简单 type；
- 它是 epilogue 和 register fragment 的桥梁；
- 非常适合作为面试项目里可解释的显式合同层。

### 8.6 `BufferModel`

`BufferModel` 继续保留，但 `BufferBacking` 必须重构。

当前 `BufferBacking` 同时保存：

- role
- memorySpace
- encoding
- logicalShape
- allocShape
- depth
- aliasGroup
- stageIndexed

这会导致：

- memory legality 还是 plan 自己保存；
- 还没真正进入 MLIR type 语义层。

最终 `BufferBacking` 建议变成：

```text
BufferBacking
  id
  role
  descType        // TypeAttr(!tb.memdesc)
  depth
  aliasGroup
  stageIndexed
  debugName
```

也就是说：

- `memorySpace` 迁到 `!tb.memdesc`
- `encoding` 迁到 `!tb.memdesc`
- `logicalShape` 迁到 `!tb.memdesc`
- `allocShape` 迁到 `!tb.memdesc`
- `elementScalar` 迁到 `!tb.memdesc`
- `alignmentBytes` 迁到 `!tb.memdesc`
- `vectorBytes` 迁到 `!tb.memdesc`
- `mutableMemory` 迁到 `!tb.memdesc`

这样 `BufferModel` 继续负责：

- resource graph
- ownership
- backing/view/value/op 关系

而不再负责“再保存一份 memory legality 真相”。

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h)

### 8.7 `PipelinePlan`

继续保留为 plan。

职责：

- `placements`
- `stageOwnedBuffers`
- `scheduledMaxStage`

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h)

原因：

- pipeline placement 是调度结果；
- 它不属于 type 或 layout 本体。

### 8.8 `AsyncPlan`

继续保留为 plan。

职责：

- async producer
- async group
- wait
- reuse fence

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h)

但最终必须满足一条硬规则：

- `AsyncPlan` 不再定义第二套 copy legality；
- copy size、element type、vector width、memory space 一律从 `!tb.memdesc` 读取。

### 8.9 `EpiloguePlan`

继续保留为 plan。

职责：

- `initMode`
- `storeMode`
- `DirectGlobalVectorPlan`
- `SharedRelayPlan`
- 预留 `exprs`

对应现有文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)

它只拥有：

- C landing strategy

它不再拥有：

- backing type
- element type
- alignment
- vector legality

这些都应该从 `!tb.memdesc` 和 `AccumulatorPlan` 读。

---

## 9. 最终 ownership 总表

| 类别 | 唯一 owner | 不允许再由谁拥有 |
|---|---|---|
| global/shared/register layout | `#tb.*` encoding attrs | `Lowering`、`AsyncPlan`、`EpiloguePlan` |
| memory space / element / alloc shape / vector / alignment | `!tb.memdesc` | `Lowering`、`BufferBacking` 平面字段、`AsyncPlan` |
| accumulator register topology | `AccumulatorPlan` | `Lowering`、`EpiloguePlan` |
| CTA/program 映射 | `ProgramMappingPlan` | `KernelConfig`、`PipelinePlan` |
| stage/order placement | `PipelinePlan` | `KernelConfig`、`EncodingPlan` |
| async frontier / wait / reuse | `AsyncPlan` | `Lowering`、`PipelinePlan` |
| C direct/relay strategy | `EpiloguePlan` | `BufferModel`、`Lowering` |

这张表是整个重构的核心。

只要这个 ownership 不再被破坏，后面优化都只是在填算法，不是在改结构。

---

## 10. 重构后的依赖图

最终依赖图固定为：

```text
KernelConfig + TargetInfo
  -> ProgramMappingPlan

KernelConfig + TargetInfo
  -> EncodingPlan
  -> AccumulatorPlan
  -> BufferModel
  -> PipelinePlan
  -> AsyncPlan

KernelConfig + TargetInfo + EncodingPlan + AccumulatorPlan
  -> EpiloguePlan

ProgramMappingPlan
+ EncodingPlan
+ AccumulatorPlan
+ BufferModel
+ PipelinePlan
+ AsyncPlan
+ EpiloguePlan
  -> Lowering
```

其中还有一个更底层的语义关系：

```text
EncodingPlan
  references #tb.* attrs

BufferModel.backings[*]
  references !tb.memdesc

!tb.memdesc
  references #tb.* attrs
```

这意味着：

- `EncodingPlan` 和 `BufferModel` 都建立在 MLIR 语义层之上；
- plan 层不再独立发明物理真相。

---

## 11. 当前代码里具体哪些地方要迁移

### 11.1 现有 IR 层文件

当前已经有：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBDialect.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBDialect.td)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBDialect.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBDialect.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBOps.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBOps.cpp)

这说明 dialect 基础骨架已经有了，新增 attr/type 是顺理成章的下一步，不是推翻重来。

### 11.2 现有最需要迁移的分析层

最需要迁移的是：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h)

原因：

- `EncodingPlan` 当前还保留 `kind + payload` 风格；
- `BufferBacking` 当前还保留平面 memory legality 字段；
- 这两处是现在最不像 Triton ownership 的地方。

### 11.3 现有最需要收口的 lowering

最需要收口的是：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

当前这里还存在典型“软真相”：

- 直接写死 `f16Type`
- 直接写死 `f32Type`
- 直接写死 async copy 的搬运粒度
- 直接从局部 pack 推 `vectorWidth`

这些都说明：

- lowering 目前还在创造物理真相；
- 最终必须改成只读 `#tb.*` 和 `!tb.memdesc`。

---

## 12. 最终建议的数据结构示意

### 12.1 IR 语义层

```text
tb.matmul
  attrs:
    block_m
    block_n
    block_k
    num_warps
    num_stages
    exact_tile
    mma
    tb.program_mapping_plan = {...}
    tb.encoding_plan = {...}      // 引用 typed encoding attrs
    tb.accumulator_plan = {...}
    tb.buffer_model = {...}       // backing 引用 !tb.memdesc
    tb.pipeline_plan = {...}
    tb.async_plan = {...}
    tb.epilogue_plan = {...}
```

```text
#tb.blocked<...>
#tb.shared<...>
#tb.dot_operand<...>
#tb.nvgpu_mma<...>
!tb.memdesc<...>
```

### 12.2 Plan 层 C++ 结构示意

`EncodingEntry`：

```text
struct EncodingEntry {
  std::string name;
  mlir::Attribute encodingAttr;
};
```

`BufferBacking`：

```text
struct BufferBacking {
  int64_t id = -1;
  BufferRole role = BufferRole::Generic;
  mlir::Type descType;  // !tb.memdesc
  int64_t depth = 1;
  int64_t aliasGroup = -1;
  bool stageIndexed = false;
  std::string debugName;
};
```

这两个示意就是本轮重构最核心的收口方向。

---

## 13. 迁移顺序

这套重构必须按顺序做，不能乱。

### 第一步：新增 IR attr/type

新增：

- `TBAttrDefs.td`
- `TBTypeDefs.td`

先把：

- `#tb.blocked`
- `#tb.shared`
- `#tb.dot_operand`
- `#tb.nvgpu_mma`
- `!tb.memdesc`

定义出来。

### 第二步：让 `EncodingPlan` 引用 typed attrs

把 `EncodingEntry` 从：

- `kind + payload`

改成：

- `encodingAttr`

### 第三步：让 `BufferBacking` 引用 `!tb.memdesc`

把 backing 的 memory legality 字段迁移到 `!tb.memdesc`。

### 第四步：重写 build/parse/validate

重写：

- `EncodingPlan` 的 build/parse/validate
- `BufferModel` 的 build/parse/validate

保证 IR 里真正出现 typed attr/type。

### 第五步：收口 lowering

改 `LowerPipelineToNVGPU`：

- 所有 element type 从 `!tb.memdesc` 读；
- 所有 vector width 从 `!tb.memdesc` 读；
- 所有 async copy bytes 从 `!tb.memdesc` 和 encoding attr 读；
- 不再写死 `f16/f32`。

### 第六步：最后再考虑可选扩展

例如：

- `!tb.async.token`
- 更一般的 shared encoding attr
- 更一般的 accumulator encoding attr

这些都属于第二优先级。

---

## 14. 这套设计为什么最适合面试项目

它同时满足三件事。

### 14.1 有 Triton 味道，但不是照抄

你可以明确说：

- 我学习了 Triton 的 ownership 原则；
- 把 layout 和 memdesc legality 提升为 IR 硬真相；
- 但没有机械复刻 Triton 的完整 IR 体系。

这比“我照着 Triton 搭了个小版”更有思考深度。

### 14.2 结构清楚，便于讲解

你可以把项目直接分成三层来讲：

1. MLIR 语义层
2. 显式 plan 层
3. lowering 层

每层的 owner 都很清楚。

### 14.3 真正能承载优化

这套结构不是为了文档好看，而是真的能支撑：

- `cp.async`
- `ldmatrix`
- `mma.sync`
- direct epilogue
- 后续的 TMA / split-K / persistent / fused epilogue

其中最关键的是：

- 后续这些优化不再要求你推翻数据结构。

---

## 15. 最终禁止事项

重构完成后，下面这些事情必须禁止。

1. 不允许 lowering 再自己决定 layout。
2. 不允许 lowering 再自己决定 element type。
3. 不允许 lowering 再自己决定 copy bytes / vector bytes。
4. 不允许 `EncodingPlan` 再保留一套与 typed attr 不一致的 payload 真相。
5. 不允许 `BufferBacking` 再保留一套与 `!tb.memdesc` 不一致的平面字段真相。
6. 不允许 `AsyncPlan` 再拥有第二套 memory legality。
7. 不允许 `EpiloguePlan` 再越权拥有 backing/type 真相。

只要这几条守住，后面所有工作都是“做优化”，而不是“救结构”。

---

## 16. 最终定稿

最终定稿版一句话如下：

`mini_triton_nvgpu_v1` 的重构后数据结构应固定为：

- 用 `#tb.*` attr 表达 layout 真相；
- 用 `!tb.memdesc` 表达 memory legality 真相；
- 用 `ProgramMappingPlan / PipelinePlan / AsyncPlan / EpiloguePlan` 表达策略与分析结果；
- 用 `KernelContract` 把它们收束成 lowering 的唯一输入合同。

这就是：

- 对 Triton 原则的真正对齐；
- 对矩阵乘法优化的真正可承载结构；
- 对面试项目最有表达力的最终设计。
