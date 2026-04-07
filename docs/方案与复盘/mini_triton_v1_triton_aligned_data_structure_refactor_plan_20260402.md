# `mini_triton_nvgpu_v1` Triton 对齐的数据结构重构总方案

## 1. 文档目的

这份文档只讨论一件事：

- `mini_triton_nvgpu_v1` 为了严格对齐 Triton 的结构思想，数据结构层应该怎么重构。

这里**不讨论算法实现细节**，也**不讨论 pass 顺序优化细节**。

本文只回答下面四个问题：

1. Triton 的数据结构到底是怎么分层的。
2. 当前 `mini_triton_nvgpu_v1` 的结构为什么不够。
3. 最终应该引入哪些稳定的一等数据结构。
4. 为什么这样设计之后，后续算法修改只需要“填充这些结构”，而不需要再推翻这些结构本身。

本文目标不是“先凑合设计一版能跑”，而是：

- 一次把数据结构边界定对；
- 后续 stage 可以继续扩展算法；
- 但不会再出现“前面的结构被后面的实现推翻”的情况。

---

## 2. 结论先行

当前 `mini_triton_nvgpu_v1` 最大的问题不是字段太少，而是**缺少 Triton 式的一等 physical contract**。

现在的问题主要有四类：

1. **双真相**
   - `KernelSpec.numStages` 是用户输入真相。
   - `SchedulePlan.numStages` 又被重新推导成另一份真相。
   - 这会导致 pipeline 深度到底是谁说了算不清楚。

2. **假布局**
   - `LayoutPlan` 里已经有 `perPhase/maxPhase/padding/swizzle`。
   - 但这些字段既不是 Triton 式类型级 encoding，也没有形成真正 physical layout contract。
   - 结果就是“看起来像布局，实际上只是描述字符串”。

3. **假 multistage**
   - `MainloopGraph` 只有 slot，没有 backing/view/alias/ring ownership。
   - `LowerPipelineToNVGPU` 只分配一块 `aShared` 和一块 `bShared`。
   - 这从结构上就不是 Triton 的 multistage shared ring。

4. **C 路径语义太薄**
   - `CRegisterPlan` 只有一个 `directGlobalVector` 布尔值加 pack 信息。
   - 它表达不了 Triton 那种“累加器布局”和“epilogue landing 模式”是两回事。

所以，最终必须把当前结构改成下面七层：

1. `KernelConfig`
2. `TargetInfo`
3. `EncodingPlan`
4. `BufferModel`
5. `PipelinePlan`
6. `AsyncPlan`
7. `AccumulatorPlan + EpiloguePlan`

并且固定唯一依赖方向：

`KernelConfig/TargetInfo -> EncodingPlan -> BufferModel -> PipelinePlan -> AsyncPlan -> AccumulatorPlan/EpiloguePlan -> Lowering`

这个方向必须是单向的：

- 后面的层可以依赖前面的层。
- 前面的层绝不能反向依赖后面的算法决定。

---

## 3. Triton 的结构原则到底是什么

这里不复刻 Triton 的类名，而复刻 Triton 的结构真相。

### 3.1 Triton 把布局做成一等 encoding

Triton 不是在 lowering 里临时判断：

- shared 有没有 swizzle
- padding 是多少
- dot operand 怎么分布
- mma shared 应该长什么样

而是把这些都做成独立 encoding：

- `BlockedEncodingAttr`
- `DotOperandEncodingAttr`
- `SwizzledSharedEncodingAttr`
- `PaddedSharedEncodingAttr`
- `NVMMASharedEncodingAttr`
- `LinearEncodingAttr`

并且这些 encoding 都能统一映射到 `LinearLayout`。

这意味着 Triton 的布局真相是：

- **布局先于算法**
- **布局先于 pipeline**
- **布局是类型级 contract，不是 lowering 里的散装 if/else**

### 3.2 Triton 把“用户指定的 stage 深度”和“调度结果”分开

Triton 中：

- `tt.num_stages` 是用户给定或默认的 pipeline 深度输入。
- `CoarseSchedule` 保存的是每个 op 的 `stage + cluster` 放置结果。
- 调度器可以生成 `scheduled_max_stage`，但这不等于重新发明一份用户配置。

所以 Triton 的真相是：

- `requestedStages` 是输入合同。
- `placement` 是调度结果。
- 两者必须分开。

### 3.3 Triton 的 multistage 是“真实 backing + 单 buffer view”

Triton 不把 pipeline stage 只表示成数字。

它真正有：

- multibuffer allocation
- single-buffer view
- stage-owned storage slice

也就是说，stage 不是抽象标签，而是实际 backing 上的一个切片视图。

这件事决定了后续 async copy、wait、reuse frontier 才能有真实 owner。

### 3.4 Triton 的 wait 依附于 producer 和 scheduled use frontier

Triton 的 wait 不是：

- 顺序扫描补几条 wait
- 根据某个 shape 模板补 barrier
- 根据 value slot 猜一个大概的 first use

而是基于：

- 哪个 producer 被 pipeline 了
- 它的 top-level first use 在 schedule 上哪里
- 它的 top-level last use 在 schedule 上哪里

所以 Triton 的 wait 真相是：

- wait 服务的是 async producer contract
- 不是服务于“value lifetime 的通用描述”

### 3.5 Triton 把 accumulator 和 epilogue 从主循环里解耦

Triton 里：

- dot operand 布局优化是一层
- accumulator init 优化是一层
- async/pipeline 是一层
- 最后 lowering 是另一层

所以 C 路径不是：

- 一个布尔值 `directGlobalVector`
- 加上若干 pack 字段

而应该拆成：

- 累加器寄存器布局
- init 模式
- store 模式
- relay 模式

这是为了保证 direct / relay / zero-init 这些语义不会被混在一个平面结构里。

---

## 4. 当前 `mini_triton_nvgpu_v1` 的结构缺陷

### 4.1 `numStages` 双真相

当前结构里：

- `KernelSpec.numStages` 直接来自 `tb.matmul`
- `SchedulePlan.numStages` 又被 longest-path 重新算了一遍

这会造成两个后果：

1. 用户输入的 pipeline 深度不是唯一真相。
2. 后续 lowering 不知道自己消费的是“配置深度”还是“调度推导深度”。

这类冲突在后续扩展 stage、ring depth、acc multibuffering 时一定继续出问题。

### 4.2 `LayoutPlan` 不是 Triton 式 encoding plan

当前 `LayoutPlan` 的问题不是完全没字段，而是字段没有真正组成稳定 contract：

- `perPhase/maxPhase/padding/swizzle` 是平铺字段
- 它们没有形成 tagged layout kind
- lowering 也几乎不真正消费这些字段

结果就是：

- 结构上“像 encoding”
- 语义上“不是 encoding”

这会直接导致后续如果补 shared swizzle、padded shared、mma shared 时，必须继续改这个 struct。

### 4.3 `MainloopGraph` 只有逻辑 slot，没有物理 owner

当前 `MainloopGraph` 里：

- `SlotInfo` 只有 `id + role`
- `shared_a` 和 `shared_b` 各只有一个 slot

这个模型表达不了：

- multistage depth
- ring buffer index
- backing / view 的区别
- alias group
- overwrite owner

所以当前 graph 只能描述“逻辑依赖图”，表达不了“真实资源图”。

### 4.4 `LatencyPlan` 还是 analysis 标记，不是 execution contract

现在的：

- `bufferDistance`
- `accMultiBuffer`
- `pipelineable`

更多是分析标签。

真正被 lowering 消费的只有非常薄的一部分语义。

这会导致后面如果补真正的 async group、wait count、barrier scope，就不得不继续改这个 plan 的职责。

### 4.5 `WaitPlan` 把 lifetime、overwrite、async frontier 混在一起

当前 `WaitPlan` 既放：

- lifetime
- first-use wait
- overwrite frontier

但这些其实不是同一层语义。

问题在于：

- lifetime 是通用 dataflow 信息
- overwrite 是 storage reuse 信息
- async wait 是 pipeline producer-consumer 同步信息

把这三类东西塞在同一个平面结构里，短期能跑，长期一定继续分裂。

### 4.6 `CRegisterPlan` 太薄，无法稳定扩展

当前 `CRegisterPlan` 里真正的主语只有：

- `directGlobalVector`
- `laneAccess`
- `initPacks`
- `storePacks`

这表达不了下面这些本来应该分开的事实：

- 累加器寄存器分布
- 初始值从哪里来
- 最终写回怎么落地
- 是否需要 relay
- relay 用什么 shared encoding

所以只要后面再支持一种新的 C path，这个结构本身就还要改。

---

## 5. 最终稳定的数据结构总图

最终建议以一个顶层只读合同聚合所有分析结果：

```c++
struct KernelContract {
  KernelConfig kernel;
  TargetInfo target;
  EncodingPlan encodings;
  BufferModel buffers;
  PipelinePlan pipeline;
  AsyncPlan async;
  AccumulatorPlan accumulator;
  EpiloguePlan epilogue;
};
```

设计原则：

- 每一层只回答一个问题。
- 每一层只引用更前面的层。
- lowering 只消费 `KernelContract`，不再重新推导合同。

下面分别展开。

---

## 6. `KernelConfig`：唯一的输入真相

### 6.1 结构定义

```c++
enum class ScalarKind {
  F16,
  F32,
};

enum class MmaKind {
  M16N8K16,
};

struct KernelConfig {
  int64_t blockM = 0;
  int64_t blockN = 0;
  int64_t blockK = 0;
  int64_t numWarps = 0;
  int64_t requestedStages = 0;
  bool exactTile = false;
  MmaKind mmaKind = MmaKind::M16N8K16;
  ScalarKind aScalar = ScalarKind::F16;
  ScalarKind bScalar = ScalarKind::F16;
  ScalarKind cScalar = ScalarKind::F32;
};
```

### 6.2 为什么必须这样设计

它只保存：

- 用户写在 `tb.matmul` 上的输入配置
- legality 需要的 kernel 级真相

它**绝不保存**：

- schedule 推导结果
- ring depth
- async producer kind
- C relay 选择

原因很简单：

- 这些不是输入事实，而是后续分析结果。

### 6.3 和 Triton 的对齐关系

它对应 Triton 中：

- loop 或 op 上的用户配置
- `tt.num_stages`
- kernel shape / warps / mma 配置

### 6.4 稳定性保证

后续如果支持更多形状、更多 mma、更多 dtypes：

- 只扩展 enum 或 legality 表
- 不需要修改这个 struct 的职责边界

---

## 7. `TargetInfo`：唯一的硬件真相

### 7.1 结构定义

```c++
struct TargetInfo {
  std::string gpuArch;              // e.g. "sm_86"
  int64_t threadsPerWarp = 32;
  int64_t sharedBankBytes = 4;
  bool supportsAsyncCopy = true;
  bool supportsLdMatrix = true;
  bool supportsMmaSync = true;
  int64_t asyncCopyMinBytes = 4;
  int64_t asyncCopyMaxBytes = 16;
  int64_t asyncCopyPreferredBytes = 16;
  llvm::SmallVector<int64_t, 4> mmaInstrShape; // {16, 8, 16}
};
```

### 7.2 为什么必须单独拆出来

当前项目把很多硬件常量散落在：

- `LayoutPlan`
- `CRegisterPlan`
- `LowerPipelineToNVGPU`

这样的问题是：

- kernel 配置
- 目标硬件限制
- lowering 实现细节

三者混在了一起。

一旦后面支持别的 target 或别的 mma 版本，这些字段就会继续互相污染。

### 7.3 和 Triton 的对齐关系

它对应 Triton 里分散在：

- encoding builder
- async legality
- copy vec bytes
- mma shape
- dialect / target lowering

这些地方的硬件固定事实。

### 7.4 稳定性保证

后续算法可以读取 `TargetInfo` 做 legality 和派生；

但不会再把硬件常量硬编码回其他 plan 里。

---

## 8. `EncodingPlan`：真正的一等布局合同

### 8.1 设计原则

这里必须严格学 Triton：

- 不再用一个平面 struct 放所有布局字段
- 改成 tagged union / variant 风格
- 每一种 encoding 只携带自己合法的字段

这样才能杜绝：

- shared path 带上不属于它的字段
- direct path 混进 relay 字段
- blocked layout 和 shared layout 公用一堆无意义成员

### 8.2 结构定义

```c++
enum class EncodingKind {
  Blocked,
  DotOperand,
  SwizzledShared,
  PaddedShared,
  NVMmaShared,
  Linear,
  Accumulator,
};

struct LinearBasis {
  std::string inputDim;                  // register/lane/warp/block/offset/iter
  llvm::SmallVector<int64_t, 8> basis;   // mapped basis on output dims
};

struct LinearLayoutSpec {
  llvm::SmallVector<std::string, 4> inputDims;
  llvm::SmallVector<std::string, 4> outputDims;
  llvm::SmallVector<int64_t, 4> outputShape;
  llvm::SmallVector<LinearBasis, 8> bases;
  bool surjective = true;
};

struct BlockedEncodingSpec {
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<unsigned, 4> sizePerThread;
  llvm::SmallVector<unsigned, 4> threadsPerWarp;
  llvm::SmallVector<unsigned, 4> warpsPerCTA;
  llvm::SmallVector<unsigned, 4> order;
  LinearLayoutSpec ctaLayout;
};

struct SharedEncodingSpec {
  EncodingKind kind = EncodingKind::SwizzledShared;
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> allocShape;
  llvm::SmallVector<unsigned, 4> order;
  int64_t vecBytes = 0;
  int64_t perPhase = 0;
  int64_t maxPhase = 0;
  llvm::SmallVector<unsigned, 4> paddingIntervals;
  llvm::SmallVector<unsigned, 4> paddings;
  bool transposed = false;
  int64_t swizzlingByteWidth = 0;
  LinearLayoutSpec linearMap;
  bool asyncEligible = false;
  int64_t asyncVecBytes = 0;
};

struct DotOperandEncodingSpec {
  int operandIndex = -1;                 // A=0, B=1
  int64_t kWidth = 0;
  int encodingParent = -1;               // index into EncodingPlan::encodings
};

struct AccumulatorEncodingSpec {
  MmaKind mmaKind = MmaKind::M16N8K16;
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> instructionShape;
  llvm::SmallVector<int64_t, 4> repShape;
  llvm::SmallVector<unsigned, 4> repOrder;
  LinearLayoutSpec warpValueLayout;
};

struct EncodingEntry {
  std::string name;                      // a_global, a_shared, a_dot, acc, ...
  EncodingKind kind;
  std::variant<BlockedEncodingSpec,
               SharedEncodingSpec,
               DotOperandEncodingSpec,
               AccumulatorEncodingSpec,
               LinearLayoutSpec> payload;
};

struct EncodingPlan {
  llvm::SmallVector<EncodingEntry, 16> encodings;
  int aGlobal = -1;
  int bGlobal = -1;
  int aShared = -1;
  int bShared = -1;
  int aDot = -1;
  int bDot = -1;
  int acc = -1;
  int cStore = -1;
};
```

### 8.3 为什么必须这样设计

#### 原因一：布局必须成为“可引用对象”

后续所有层都应该引用 encoding id：

- backing 引用 encoding
- view 引用 encoding
- async producer 引用 shared encoding
- epilogue 引用 accumulator/store encoding

而不是把 `padding/swizzle/order` 再复制一遍。

#### 原因二：必须能表达 Triton 的多类 shared encoding

当前 `LayoutPlan` 最大的问题是它只能表达一种想象中的 shared layout。

但 Triton 真正需要的是：

- swizzled shared
- padded shared
- nvmma shared
- linear shared

如果今天不把 `kind` 和 `payload` 分开，明天支持第二种 shared layout 时，这个结构还要继续推翻。

#### 原因三：必须显式区分 `logicalShape` 和 `allocShape`

这是当前 mini 最缺的东西之一。

Triton 中共享布局通常有：

- 逻辑 tensor shape
- 实际 shared alloc shape

如果不把这两个字段单独作为一等概念，后续任何 padding、swizzle、mma shared、relay shared 都还会回到“lowering 里临时猜 physical layout”。

### 8.4 和 Triton 的对齐关系

它直接对齐 Triton 中：

- `BlockedEncodingAttr`
- `DotOperandEncodingAttr`
- `SwizzledSharedEncodingAttr`
- `PaddedSharedEncodingAttr`
- `NVMMASharedEncodingAttr`
- `LinearEncodingAttr`
- `LinearLayout`

### 8.5 稳定性保证

后续即便要支持：

- 新 shared encoding
- 新 mma encoding
- 新的 dot operand 变体

也只是：

- 新增 `EncodingKind`
- 新增一个 payload 类型

而不会改变 `EncodingPlan` 作为“布局一等引用表”的角色。

---

## 9. `BufferModel`：从逻辑 slot 图改成真实资源图

### 9.1 设计原则

必须放弃当前 `SlotInfo` 的主语。

正确主语应该是：

1. `BufferBacking`
2. `BufferView`
3. `ValueState`
4. `PipelineOp`

也就是说：

- backing 是存储体
- view 是这个存储体在某个 stage / 某个 slice 上的视图
- value 是 dataflow 中流动的值，拥有某个 view
- op 消费和产生 value

### 9.2 结构定义

```c++
enum class MemorySpace {
  Global,
  Shared,
  Registers,
};

enum class BufferRole {
  OperandA,
  OperandB,
  Accumulator,
  EpilogueRelay,
  Barrier,
  Descriptor,
};

struct BufferBacking {
  int64_t id = -1;
  BufferRole role;
  MemorySpace memorySpace;
  int encoding = -1;                     // EncodingPlan reference
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> allocShape;
  int64_t depth = 1;                     // ring depth
  int64_t aliasGroup = -1;               // -1 means no alias
  bool stageIndexed = false;
};

enum class ViewKind {
  FullBuffer,
  StageSlice,
  TileSlice,
  DotOperandSlice,
  AccumulatorPack,
  EpiloguePack,
};

struct BufferView {
  int64_t id = -1;
  int64_t backing = -1;                  // BufferBacking reference
  ViewKind kind = ViewKind::FullBuffer;
  int64_t stage = -1;
  int64_t bufferIndex = -1;
  int encoding = -1;                     // view-level encoding
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> shape;
};

enum class ValueKind {
  GlobalTile,
  SharedTile,
  DotOperandFragment,
  AccumulatorFragment,
  EpilogueFragment,
};

struct ValueState {
  int64_t id = -1;
  ValueKind kind;
  int64_t definingOp = -1;
  llvm::SmallVector<int64_t, 8> users;
  int64_t ownerView = -1;                // BufferView reference
  int64_t loopDistance = 0;              // 0 for same-iter, 1 for carried
};

enum class PipelineOpKind {
  LoadGlobalToShared,
  LoadSharedToDot,
  Mma,
  AccumulatorInit,
  AccumulatorStore,
};

struct PipelineOp {
  int64_t id = -1;
  PipelineOpKind kind;
  llvm::SmallVector<int64_t, 8> inputs;
  llvm::SmallVector<int64_t, 8> outputs;
  llvm::SmallVector<int64_t, 4> iterationCoords;  // k-group, m-tile, n-tile...
};

struct BufferModel {
  llvm::SmallVector<BufferBacking, 8> backings;
  llvm::SmallVector<BufferView, 32> views;
  llvm::SmallVector<ValueState, 64> values;
  llvm::SmallVector<PipelineOp, 64> ops;
};
```

### 9.3 为什么必须这样设计

#### 原因一：multistage 的 owner 必须是 backing/view，不是 slot

只有这样，后续算法才能稳定表达：

- ring depth = 2 或更多
- 第 `i` 个 stage 对应哪个 buffer slice
- 哪个 view 可以被 overwrite

如果继续用 slot，后面必然还要再加：

- `stage`
- `ringIndex`
- `aliasOwner`
- `backingId`

最后 slot 自己就会膨胀成一个拙劣的 backing/view 混合体。

#### 原因二：overwrite frontier 必须挂在 backing/view 上

overwrite 的本质不是“两个值占了同一个逻辑 slot”。

overwrite 的本质是：

- 两个值最终指向同一个 backing 的不同时间片
- 下一个 producer 会复用这个 backing 的某个 buffer slice

这必须由 `BufferBacking` 和 `BufferView` 来描述。

#### 原因三：C relay、barrier、descriptor 也能自然纳入同一模型

如果未来引入：

- shared relay
- barrier backing
- descriptor backing

只要增加 `BufferRole` 和对应 `ViewKind` 即可。

不需要重新发明第二套“特殊资源图”。

### 9.4 和 Triton 的对齐关系

它对应 Triton 中的：

- multibuffer alloc
- single-buffer view
- memdesc view
- loop-carried value
- pipelined op graph

这是对 Triton “真实资源图”思想的简化复刻。

### 9.5 稳定性保证

这一步做完后，后续算法只会：

- 往 `BufferModel` 里填更多 backing / view / op / value

而不会再改变：

- backing/view/value/op 四层分工本身。

---

## 10. `PipelinePlan`：唯一的 stage ownership 和调度合同

### 10.1 设计原则

它只负责回答两个问题：

1. pipeline 有多少真实 buffer depth
2. 每个 op 在 schedule 上位于哪个 `stage/cluster/order`

它不负责：

- 描述布局
- 描述 async issue 细节
- 描述 epilogue 模式

### 10.2 结构定义

```c++
struct StageBufferUse {
  int64_t backing = -1;
  int64_t stage = -1;
  int64_t bufferIndex = -1;
  int64_t producerOp = -1;
};

struct Placement {
  int64_t opId = -1;
  int64_t stage = 0;
  int64_t cluster = 0;
  int64_t order = 0;
  std::string reason;
};

struct PipelinePlan {
  int64_t scheduledMaxStage = 0;
  llvm::SmallVector<Placement, 64> placements;
  llvm::SmallVector<StageBufferUse, 16> stageOwnedBuffers;
};
```

### 10.3 为什么必须这样设计

#### 原因一：彻底消灭 `numStages` 双真相

这里必须固定规则：

- `KernelConfig.requestedStages` 是输入真相
- `PipelinePlan.scheduledMaxStage` 是调度结果

禁止再出现：

- `SchedulePlan.numStages = 重新推导出的另一份用户深度`

#### 原因二：stage 必须和 backing 使用关系显式绑定

当前项目把 stage 只用来排序 op。

这还不够。

要表达 Triton 式 multibuffer，必须再有：

- 哪个 stage 使用 backing 的哪个 `bufferIndex`

否则 lowering 根本没有真实 owner 可以消费。

### 10.4 和 Triton 的对齐关系

它对齐 Triton 中：

- `tt.num_stages`
- `CoarseSchedule`
- `loop.stage`
- `loop.cluster`
- multibuffer alloc depth

### 10.5 稳定性保证

后续不管 schedule 算法怎么变：

- longest-path
- latency-driven
- future generalized schedule

都只是填 `placements`。

不会再改 `PipelinePlan` 这层的结构边界。

---

## 11. `AsyncPlan`：唯一的 async producer / wait / barrier 合同

### 11.1 设计原则

必须把 async 语义从 `LatencyPlan` 和 `WaitPlan` 中剥离出来。

因为：

- latency 是分析
- wait 是同步
- barrier 是存储可见性

这三者相关，但不是同一层。

### 11.2 结构定义

```c++
enum class AsyncProducerKind {
  CpAsync,
  SyncCopyFallback,
};

struct AsyncProducer {
  int64_t opId = -1;                     // producer op
  AsyncProducerKind kind = AsyncProducerKind::CpAsync;
  int64_t srcView = -1;
  int64_t dstView = -1;
  int64_t groupId = -1;
  int64_t vecBytes = 0;
  bool legal = false;
  std::string reason;
};

struct AsyncGroup {
  int64_t id = -1;
  llvm::SmallVector<int64_t, 8> producers;
};

struct WaitInfo {
  int64_t groupId = -1;
  int64_t beforeOpId = -1;
  int64_t requiredStage = -1;
  int64_t requiredCluster = -1;
  int64_t requiredOrder = -1;
  bool needsBarrier = true;
  std::string reason;
};

struct ReuseFence {
  int64_t backing = -1;
  int64_t retiringView = -1;
  int64_t acquiringView = -1;
  int64_t afterOpId = -1;
  std::string reason;
};

struct AsyncPlan {
  llvm::SmallVector<AsyncProducer, 16> producers;
  llvm::SmallVector<AsyncGroup, 16> groups;
  llvm::SmallVector<WaitInfo, 16> waits;
  llvm::SmallVector<ReuseFence, 16> reuseFences;
};
```

### 11.3 为什么必须这样设计

#### 原因一：wait 应该服务于 async group，而不是 value lifetime

这一步是当前结构最容易继续出错的地方。

如果 wait 继续绑在：

- `valueId`
- `slot`
- 通用 lifetime

那后面一旦出现：

- 多 producer 合组
- wait count 变化
- barrier scope 变化

还得继续改结构。

#### 原因二：sync fallback 和 async mainline 必须由同一层显式区分

Triton 里 async 是有明确 legality 的。

你这里未来也必须是：

- `CpAsync`
- `SyncCopyFallback`

作为明确 kind，而不是靠某个布尔字段推测。

### 11.4 和 Triton 的对齐关系

它对齐 Triton 中：

- async load legality
- pipelined producer
- first use / last use frontier
- wait group
- combined waits

### 11.5 稳定性保证

以后如果要支持：

- TMA
- 更复杂 wait 策略
- 更复杂 barrier 作用域

也只是扩展：

- producer kind
- group / wait 字段

不会再动“async 是单独一层合同”这个边界。

---

## 12. `AccumulatorPlan`：累加器寄存器布局合同

### 12.1 设计原则

必须把“累加器长什么样”从“C 如何 init/store”里拆开。

因为：

- 累加器寄存器拓扑是 mainloop/MMA 语义
- init/store 是 epilogue 语义

这两件事不应该继续混在同一个 struct 里。

### 12.2 结构定义

```c++
struct LaneAccessPattern {
  int64_t laneRowGroupSize = 0;
  int64_t laneColGroupSize = 0;
  int64_t laneColStride = 0;
  llvm::SmallVector<int64_t, 4> rowOffsets;
};

struct AccumulatorPack {
  int64_t packId = 0;
  int64_t rowBase = 0;
  int64_t colBase = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t elemCount = 0;
  int64_t vectorWidth = 0;
};

struct AccumulatorPlan {
  int encoding = -1;                     // AccumulatorEncodingSpec
  int64_t registersPerWarp = 0;
  LaneAccessPattern laneAccess;
  llvm::SmallVector<AccumulatorPack, 16> packs;
  bool liveAcrossStages = false;
  int64_t multiBufferDepth = 1;
};
```

### 12.3 为什么必须这样设计

当前 `CRegisterPlan` 里：

- `initPacks`
- `storePacks`

表面上是 C 路径信息，实际上里面混着：

- 累加器寄存器拓扑
- store vector 宽度
- lane 到 pack 的映射

这些应该先抽成累加器自身的结构。

只有这样，后续：

- direct init
- zero init
- relay init
- direct store
- relay store

才能共享同一份 accumulator register topology。

### 12.4 和 Triton 的对齐关系

它对齐 Triton 中：

- MMA result layout
- accumulator register distribution
- accumulator multibuffering 可能性

### 12.5 稳定性保证

以后不管 C 路径怎么改：

- accumulator topology 这部分不再改职责
- 只改消费它的 epilogue mode

---

## 13. `EpiloguePlan`：唯一的 C init / store / relay 合同

### 13.1 设计原则

这是当前最需要用 tagged union 的地方。

必须彻底禁止这种状态：

- 名义 direct
- 实际还带 relay/shared 字段

正确设计必须让：

- direct path
- relay path

在类型层面互斥。

### 13.2 结构定义

```c++
enum class AccumulatorInitMode {
  Zero,
  DirectGlobalVector,
  SharedRelay,
};

enum class AccumulatorStoreMode {
  DirectGlobalVector,
  SharedRelay,
};

struct DirectGlobalVectorPlan {
  llvm::SmallVector<AccumulatorPack, 16> packs;
  int64_t vectorWidth = 0;
};

struct SharedRelayPlan {
  int64_t relayBacking = -1;             // BufferBacking
  int relayEncoding = -1;                // SharedEncodingSpec
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> allocShape;
  llvm::SmallVector<AccumulatorPack, 16> packs;
};

struct EpiloguePlan {
  AccumulatorInitMode initMode = AccumulatorInitMode::Zero;
  AccumulatorStoreMode storeMode =
      AccumulatorStoreMode::DirectGlobalVector;
  std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan> init;
  std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan> store;
};
```

### 13.3 为什么必须这样设计

#### 原因一：direct 和 relay 必须在类型上互斥

这是为了彻底杜绝以后再出现：

- 一个 bool 叫 `directGlobalVector`
- 但结构里还挂着 relay 相关字段

tagged union 的意义就在这里：

- 直接路径只有 direct payload
- relay 路径只有 relay payload

#### 原因二：init 和 store 必须分开

因为未来完全可能出现：

- init 是 zero
- store 是 direct

或者：

- init 是 direct vector
- store 是 relay

如果把 init/store 继续绑死到一个结构里，后面功能一扩就还得重构。

### 13.4 和 Triton 的对齐关系

它对齐 Triton 中：

- accumulator init 优化
- store landing 模式
- direct / relay 语义分离

### 13.5 稳定性保证

以后如果支持：

- generic epilogue
- fused epilogue
- 更多 relay 变体

也只是：

- 扩展 mode
- 增加 payload variant

而不会再改 `EpiloguePlan` 的根结构。

---

## 14. 顶层依赖关系为什么必须是这样

必须固定为：

### 14.1 `KernelConfig + TargetInfo -> EncodingPlan`

因为布局依赖：

- block shape
- mma kind
- warps
- target 硬件约束

但不依赖：

- schedule
- wait
- epilogue 决策

### 14.2 `EncodingPlan -> BufferModel`

因为 backing/view 必须引用真实 encoding。

如果没有 encoding，buffer 只能退化回 slot。

### 14.3 `BufferModel -> PipelinePlan`

因为 pipeline 必须知道：

- 哪些 backing 可以 multibuffer
- 哪些 value 是 loop-carried
- 哪些 op 消费哪些 view

### 14.4 `PipelinePlan -> AsyncPlan`

因为 async wait/frontier 必须锚到：

- scheduled placement
- stage-owned buffers

### 14.5 `EncodingPlan + AccumulatorPlan -> EpiloguePlan`

因为 epilogue 需要：

- accumulator topology
- 目标 landing encoding

但不应该反过来影响 accumulator 自身布局。

### 14.6 最后才是 lowering

lowering 只消费：

- backing/view
- placements
- async groups/waits
- accumulator/epilogue mode

它绝不能再反向决定：

- shared layout
- ring depth
- direct/relay 模式

---

## 15. 旧结构到新结构的映射

### 15.1 `KernelSpec`

保留的内容：

- block shape
- dtypes
- mma kind
- `exactTile`

修改：

- `numStages` 重命名为 `requestedStages`

### 15.2 `LayoutPlan`

退场。

替换为：

- `EncodingPlan`

原因：

- `LayoutPlan` 是平面字段，不是 encoding 表。

### 15.3 `MainloopGraph`

退场。

替换为：

- `BufferModel`

原因：

- 旧 graph 只有逻辑 slot，没有 backing/view。

### 15.4 `LatencyPlan`

不再作为 lowering 主合同。

后续可以保留为内部分析结果，但：

- 不再承载 async/wait 主语义。

### 15.5 `SchedulePlan`

退场。

其 placement 部分并入：

- `PipelinePlan`

并且：

- 删除 `numStages` 作为第二真相。

### 15.6 `WaitPlan`

拆分：

- lifetime / reuse 进入 `BufferModel` 与 `AsyncPlan`
- async wait 进入 `AsyncPlan`

### 15.7 `CRegisterPlan`

拆分：

- `AccumulatorPlan`
- `EpiloguePlan`

---

## 16. 为什么这套结构后续不需要再改

这里的“不需要再改”不是指以后一个字段都不能加。

这里的意思是：

- **结构分工不需要再改**
- **ownership 边界不需要再改**
- **前后依赖方向不需要再改**

### 16.1 以后允许扩展的只有三类

#### 第一类：扩 enum / kind

例如：

- 新 shared encoding
- 新 async producer kind
- 新 epilogue mode

#### 第二类：扩 legality

例如：

- 支持更多 shape
- 支持更多 mma kind

#### 第三类：扩某个 payload 的字段

例如：

- 新 target 需要额外 swizzle 参数
- relay path 需要更多 packing 描述

### 16.2 以后不应该再发生的事

#### 不应该再发生一：把输入真相和推导真相混在一起

禁止再出现：

- `requestedStages` 和 `scheduledNumStages` 混成一个字段

#### 不应该再发生二：把 layout 做成平面大 struct

禁止再出现：

- 一个 shared/layout struct 里平铺所有 kind 的字段

#### 不应该再发生三：把 slot 当成资源主语

禁止再出现：

- 用 `sharedASlot = 0` 这种结构试图表达 multistage backing

#### 不应该再发生四：direct/relay 共存于同一 payload

禁止再出现：

- 一个 direct path 同时携带 relay scratch 字段

#### 不应该再发生五：lowering 重新发明合同

禁止再出现：

- lowering 根据当前形状再次猜 ring depth / wait frontier / relay 语义

---

## 17. 推荐的头文件重组方式

建议直接重组为下面这些头文件：

```text
include/tb/Analysis/KernelConfig.h
include/tb/Analysis/TargetInfo.h
include/tb/Analysis/EncodingPlan.h
include/tb/Analysis/BufferModel.h
include/tb/Analysis/PipelinePlan.h
include/tb/Analysis/AsyncPlan.h
include/tb/Analysis/AccumulatorPlan.h
include/tb/Analysis/EpiloguePlan.h
include/tb/Analysis/KernelContract.h
```

其中：

- `KernelContract.h` 只负责聚合
- 其他头文件各自只定义一层合同

这样后续 pass 也会自然变清楚：

- build encodings
- build buffer model
- build pipeline plan
- build async plan
- build accumulator plan
- build epilogue plan
- lower

---

## 18. 最终原则清单

### 原则 1：布局一定先于流水线

没有 encoding contract，就不要谈 pipeline contract。

### 原则 2：真实 owner 一定先于 wait

没有 backing/view，就不要谈 overwrite 和 wait frontier。

### 原则 3：输入真相和推导真相必须分离

`requestedStages` 绝不能和 schedule 结果混在一起。

### 原则 4：direct / relay 必须类型互斥

不能再用布尔值加一堆潜在无效字段。

### 原则 5：lowering 只能消费，不能重新决策

它是翻译层，不是策略层。

### 原则 6：所有 plan 都必须是可验证合同

每一层都应该能单独验证“字段是否自洽”，而不是依赖 lowering 最后碰撞出错。

---

## 19. 一句话总结

这次重构的根本目标，不是把当前几个 struct 改得更大，而是把 `mini_triton_nvgpu_v1` 从“分析属性的集合”改成“和 Triton 一样按物理合同分层的结构系统”。

只要这个分层立住：

- 后续算法可以继续演进；
- 但不会再出现前面的结构被后面的实现推翻；
- 也不会再出现“名义 direct、实际 relay”或“名义 multistage、实际单 shared”这类双真相问题。
