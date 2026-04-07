# `mini_triton` V1 严格复刻 Triton 思想并在窄形状上追平 Triton 的完整方案

## 1. 文档目的

这份文档只回答一个问题：

- 如果第一版就要求在特定形状上达到 Triton 级性能；
- 同时禁止特判、禁止模板式调度、禁止执行层反向发明语义；
- 并且要求整体思路严格复刻 Triton；

那么项目应该怎么设计、怎么实现、怎么验收。

这里的第一版不是“最小能跑版”，而是：

- 最小可讲清楚；
- 最小可测清楚；
- 最小可在窄形状上和 Triton 正面对比；
- 没达到性能目标就不算完成。

---

## 2. 2026-04-01 当前事实

当前 `triton_backend_nvgpu` 的正式 exact case 还没有贴近 Triton。

来自 `/tmp/stage1_triton_compare_20260401/summary.json` 的 `exact_128x128x32` 数据：

- backend `nsys_kernel_ns = 21889`
- Triton `nsys_kernel_ns = 8577`
- backend `gpu_time_ns = 18688`
- Triton `gpu_time_ns = 13504`
- backend `eligible_warps = 0.03`
- Triton `eligible_warps = 0.07`
- backend `stall_long_scoreboard = 14.39`
- Triton `stall_long_scoreboard = 2.79`
- backend `smem_allocated = 34944`
- Triton `smem_allocated = 17920`
- backend 仍然有 C relay 痕迹：
  - `has_c_relay_16x68 = true`
  - `has_c_relay_vec2 = true`
  - `has_c_relay_vec4 = true`

`tiny_exact_64x64x32` 也同样显示相同问题：

- backend `nsys_kernel_ns = 11168`
- Triton `nsys_kernel_ns = 5089`
- backend `stall_long_scoreboard = 12.95`
- Triton `stall_long_scoreboard = 0.72`
- backend `smem_allocated = 13696`
- Triton `smem_allocated = 9216`

这说明当前主要差距不是：

- 没有 tensor core；
- A/B async 完全没回来；
- 只有 bank conflict 问题；

而是：

- 主链外面还包着额外 glue；
- C 路径仍带 shared relay；
- 中间调度真相没有彻底前移；
- 执行层仍然过厚。

结论只有一个：

- 如果目标是第一版就在窄形状上达到 Triton 级性能，就不应该继续在当前 Stage1 主链上堆局部补丁。
- 应该新建一条严格按 Triton 思想组织的 V1 主线。

---

## 3. 总体决策

### 3.1 架构决策

第一版采用：

- `MLIR 外壳`
- `自定义领域数据结构`
- `薄 lowering`

不采用：

- 当前 `triton_backend_nvgpu` 的中间层主链继续修补
- 复制 Triton 内部类体系
- 把所有调度真相都塞进 executor

推荐落地方式：

- 新建独立项目，或在当前仓库下新建完全隔离的子目录
- 旧项目只保留为 correctness / benchmark / profiling oracle

### 3.2 第一版的唯一完成标准

第一版不以“功能跑通”为完成标准。

第一版完成的定义是：

1. 在官方窄形状集上正确；
2. 不靠特判和模板调度；
3. IR 与 pass 链严格遵守 Triton 的依赖方向；
4. 官方测量指标贴近 Triton，未贴近则 V1 视为未完成。

---

## 4. 第一版严格范围

### 4.1 支持范围

第一版只支持：

- NVIDIA
- `mma.sync`
- `fp16 x fp16 -> fp32`
- single-CTA
- exact-tile
- `BLOCK_M x BLOCK_N x BLOCK_K = 64x64x32` 或 `128x128x32`
- `num_warps = 1` 或 `4`
- `num_stages = 2`
- 无 fusion
- 无 autotune
- 无 multi-CTA
- 无 split-K
- 无 generic epilogue

### 4.2 这不叫“特判”

上面的范围收窄是产品范围，不是算法特判。

允许：

- 只支持少数合法 config
- 用一张 legality 表声明哪些 config 合法
- 使用硬件 ISA 固有常量，例如 `mma.sync.m16n8k16`
- 使用 cp.async / ldmatrix 的对齐和 vector 宽度约束

不允许：

- `if (blockM == 64 && blockN == 64) 用一套固定 stage/cluster 模板`
- `if (shape == 128) 直接切到另一套 wait 方案`
- `static const schedule_for_64[]`
- `static const wait_plan_for_128[]`
- `if (numWarps == 4) 就启用额外 C relay`
- executor 按运行时形状重新猜 latency / stage / wait

也就是说：

- 可以有“支持哪些 config”的表；
- 不可以有“每个 config 预烘焙一套调度答案”的表。

---

## 5. 第一版必须严格复刻的 Triton 思想

第一版要复刻的不是 Triton 的类名，而是下面这些不可动摇的结构真相。

### 5.1 布局真相先于流水线真相

先固定：

- operand/shared/layout/descriptor
- C register-set 与 direct landing

再固定：

- mainloop graph
- latency
- schedule
- wait/overwrite

最后才：

- 降成 async copy / wait / local load / mma / store

对应本机 Triton 参考：

- `/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp`
- `/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp`
- `/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp`
- `/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp`
- `/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp`

### 5.2 `AssignLatencies` 必须从真实 use-distance 出发

必须学的不是“load latency = 常数”，而是：

- 先求 load 到 first-use 的 indirection level
- 冲突 distance 直接视为不合法，而不是随手回退
- 再结合 `numStages` 和 pipelineability 分配 target latency
- 再给 MMA 分配 self latency
- 再决定 accumulator multibuffering

### 5.3 `ScheduleLoops` 必须是 latency-driven

必须学的不是“按 kind 分 cluster”，而是：

1. 先排 key ops
2. 再补 distance-one 依赖
3. 再把剩余 op 放到最后 stage / epilogue
4. cluster 只是结果，不是模板

### 5.4 wait 必须锚到 use frontier

wait 不能是：

- 顺序扫描后补丁
- 根据 stage 常数表补几拍

wait 必须来自：

- value 的 def
- value 的 first-use
- value 的 last-use
- slot 的 reuse frontier

### 5.5 lowering 只能消费，不能重新决策

lowering 层只负责把已确定的计划翻译成：

- async copy
- async wait
- barrier
- local load / ldmatrix
- mma.sync
- direct global load/store

它不能：

- 再决定 shared 布局
- 再决定 direct/relay
- 再决定 latency
- 再决定 wait 点

---

## 6. 为什么窄形状 parity 是现实可达的

在窄形状集下追平 Triton 是现实目标，不是空想。

原因是：

1. 目标范围很窄：
   - single-CTA
   - exact-tile
   - 两个 block shape
   - 两个 warps 配置

2. Triton 在这些形状上的主导成本很明确：
   - A/B async
   - ldmatrix/local load
   - mma.sync
   - direct C load/store

3. 当前 backend 与 Triton 的 HMMA 主体已经不是完全不同的路线。
   差距主要在 HMMA 外围 glue，而不是完全没有走到同一类核心指令。

4. 在同一硬件、同一数据类型、同一 tile 形状下，只要：
   - 主循环 pipeline 结构接近；
   - C path 不再 relay；
   - shared footprint 接近；
   - wait frontier 不保守；

   那么窄形状 parity 是可争取的。

但要注意：

- 这不意味着“第一天就一定跑赢”；
- 这意味着“V1 的结构必须从一开始就是为 parity 设计的，而不是先做一套教学版再重来”。

---

## 7. V1 架构总览

### 7.1 核心原则

项目结构必须满足：

- MLIR 是外壳和承载层
- 自定义结构是领域分析层
- lowering 是消费层
- benchmark/profiler 是验收层

### 7.2 推荐目录结构

```text
mini_triton_nvgpu_v1/
├── include/tb/IR/
│   ├── TBDialect.h
│   ├── TBDialect.td
│   ├── TBOps.h
│   └── TBOps.td
├── include/tb/Analysis/
│   ├── KernelSpec.h
│   ├── LayoutPlan.h
│   ├── CRegisterPlan.h
│   ├── MainloopGraph.h
│   ├── LatencyPlan.h
│   ├── SchedulePlan.h
│   └── WaitPlan.h
├── include/tb/Transforms/
│   ├── Passes.h
│   └── Passes.td
├── include/tb/Conversion/
│   └── TBToNVGPU.h
├── lib/IR/
├── lib/Analysis/
├── lib/Transforms/
├── lib/Conversion/
├── tools/
│   └── tb-opt.cpp
├── test/
│   ├── ir/
│   ├── transforms/
│   └── conversion/
└── bench/
    ├── bench_matmul.py
    └── profile_ncu.sh
```

### 7.3 第一版 MLIR 外壳

第一版只需要一个高层核心 op：

```mlir
tb.matmul %A, %B, %C
  {block_m = 128, block_n = 128, block_k = 32,
   num_warps = 4, num_stages = 2,
   exact_tile = true,
   mma = "m16n8k16"}
  : memref<128x32xf16>, memref<32x128xf16>, memref<128x128xf32>
```

不需要第一天就把中间所有 op 都物化出来。

第一版完全可以：

- 前半段 pass 先产 attr
- 后半段 lowering 直接消费这些 attr

如果为了 debug 需要更强的可视化，再加一个可选的中间 materialize pass。

---

## 8. 哪些地方用 MLIR，哪些地方不用

### 8.1 必须用 MLIR 的地方

这些地方必须保留 MLIR：

- dialect / op / attr 定义
- verifier
- pass 注册与 pass manager
- IR dump
- pass 间合同挂载
- `nvgpu` / `nvvm` lowering
- lit 测试

### 8.2 不把 MLIR 当唯一数据模型的地方

这些分析不应该直接拿 `Operation*` 当唯一主模型：

- mainloop graph
- latency
- schedule
- wait/overwrite
- direct C register mapping

原因不是“不要 MLIR”，而是：

- 这些是 matmul 专用算法；
- 直接吃 `Operation*` 会把算法和 IR plumbing 缠死；
- 对窄域项目来说，领域结构更清楚、更易测、更易 debug。

### 8.3 正确边界

正确边界是：

1. pass 进入 `tb.matmul`
2. 从 attr 解析 `KernelSpec`
3. 构造自定义 `LayoutPlan / MainloopGraph / LatencyPlan / SchedulePlan / WaitPlan`
4. 再把结果写回 attr
5. lowering 继续消费 attr，生成 `nvgpu / nvvm`

所以：

- pass 仍然是 MLIR pass
- 只是 pass 的核心算法不是直接围着裸 `Operation*` 拼装

---

## 9. Triton 结构与 V1 简化结构对照

| Triton 里的结构 | V1 对应结构 | 必须保留的字段/语义 | 可以删掉或简化的部分 |
|---|---|---|---|
| `scf::ForOp` + loop body `Operation*` | `KernelSpec + MainloopGraph` | loop-carried 关系、def-use、op kind、slot owner、frontier | 通用 region/block/control-flow 世界 |
| `Value` / `BlockArgument` | `ValueId + ValueInfo` | def、users、slot、bundle、first-use、last-use | 通用 SSA 细节和所有 block/yield 特性 |
| `loadOpsToIndirectionLevel` 的输出 | `IndirectionInfo` | load 到 consumer 的 distance、冲突检查 | 与 MLIR `yield` 的绑定写法 |
| `DenseMap<Operation*, int> opLatency` | `LatencyPlan` | target latency、self latency、buffer distance、pipelineable、reason | 直接以 `Operation*` 为 key 的实现方式 |
| `CoarseSchedule` | `SchedulePlan` | stage、cluster、order、placement reason | 复杂 iterator、cluster list 技巧、通用线性化 API |
| `MMAv5PipelineableOperandsHelper` | `MmaPipelineInfo` | operand pipelineability、acc multibuffer、RMW 检查 | Triton 内部 helper 类的组织方式 |
| `EncodingAttr / MemDescType / LinearLayout` | `LayoutPlan + SharedSpec + FragmentMap` | tile shape、warp decomposition、shared swizzle/padding、register mapping | 完整 encoding lattice、通用 reshape/transpose 系统 |
| `PlanCTA` | `CTAPlan` | threads-per-CTA、warp binding、block/grid shape | multi-CTA 与复杂 CGA 布局 |
| `LowerLoops` | `LowerPipelineToNVGPU` | async issue/wait、barrier、local load、mma、store 的消费顺序 | 全套通用 MLIR rewriter 框架 |
| `MemoryOpToLLVM` patterns | `TBToNVGPU / TBToNVVM` | 正确地把既定计划翻译成 target op | 第一版就复刻全部 conversion pattern 类 |

这个表的关键结论是：

- 必须复刻 Triton 的语义；
- 不必复制 Triton 的类层次和工程形态。

---

## 10. 第一版核心数据结构

### 10.1 `KernelSpec`

```cpp
struct KernelSpec {
  int blockM;
  int blockN;
  int blockK;
  int numWarps;
  int numStages;
  bool exactTile;
  MmaKind mmaKind;    // e.g. m16n8k16
  ElementType aType;  // fp16
  ElementType bType;  // fp16
  ElementType cType;  // fp32
};
```

职责：

- 只描述合法问题和合法 config
- 不带任何 schedule 结果

### 10.2 `LayoutPlan`

```cpp
struct OperandSharedSpec {
  OperandRole role;            // A / B
  SmallVector<int64_t> shape;  // shared tile shape
  int vecBytes;
  int perPhase;
  int maxPhase;
  SwizzleKind swizzle;
  bool asyncEligible;
};

struct FragmentSpec {
  OperandRole role;            // A / B / Acc
  int valuesPerLane;
  SmallVector<int64_t> logicalShape;
  SmallVector<int64_t> registerOrder;
};

struct LayoutPlan {
  OperandSharedSpec sharedA;
  OperandSharedSpec sharedB;
  FragmentSpec fragA;
  FragmentSpec fragB;
  FragmentSpec fragAcc;
};
```

职责：

- 固定 A/B shared backing
- 固定 fragment/register 映射
- 固定 direct C path 需要的 register-set 形状

### 10.3 `CRegisterPlan`

```cpp
struct CPackInfo {
  int groupId;
  int rowBase;
  int colBase;
  int elemCount;
  int vectorWidth;
};

struct CRegisterPlan {
  SmallVector<CPackInfo> initLoads;
  SmallVector<CPackInfo> stores;
  int registersPerWarp;
  bool directGlobalVector;
};
```

职责：

- 描述 `global <-> register` 的 direct C path
- 不允许引入 shared relay

### 10.4 `MainloopGraph`

```cpp
enum class OpKind { LoadA, LoadB, LocalLoadA, LocalLoadB, Mma };

struct ValueInfo {
  int id;
  ValueKind kind;
  int defOp;
  SmallVector<int> users;
  int slot;
  int bundle;
  int firstUseOrdinal;
  int lastUseOrdinal;
};

struct OpInfo {
  int id;
  OpKind kind;
  SmallVector<int> inputs;
  SmallVector<int> outputs;
  int kGroup;
};

struct MainloopGraph {
  SmallVector<ValueInfo> values;
  SmallVector<OpInfo> ops;
  SmallVector<SlotInfo> slots;
};
```

职责：

- 只描述 def-use / bundle / slot / frontier
- 不描述 latency、stage、wait

### 10.5 `LatencyPlan`

```cpp
struct OpLatencyInfo {
  int opId;
  int targetLatency;
  int selfLatency;
  int bufferDistance;
  bool pipelineable;
  bool accMultiBuffer;
  std::string reason;
};

struct LatencyPlan {
  SmallVector<OpLatencyInfo> ops;
};
```

职责：

- 与 Triton 的 `AssignLatencies` 对齐
- 从 indirection 与 pipelineability 出发
- 不直接做 schedule

### 10.6 `SchedulePlan`

```cpp
struct Placement {
  int opId;
  int stage;
  int cluster;
  int order;
  std::string reason;
};

struct SchedulePlan {
  int numStages;
  SmallVector<Placement> placements;
};
```

职责：

- 与 Triton 的 `CoarseSchedule` 对齐
- 是调度结果，不是模板

### 10.7 `WaitPlan`

```cpp
struct Lifetime {
  int valueId;
  int defStage;
  int defCluster;
  int defOrder;
  int firstUseStage;
  int firstUseCluster;
  int firstUseOrder;
  int lastUseStage;
  int lastUseCluster;
  int lastUseOrder;
};

struct WaitEvent {
  int valueId;
  int beforeOpId;
  int requiredStage;
  int requiredCluster;
  int requiredOrder;
  std::string reason;
};

struct OverwriteEvent {
  int slotId;
  int prevValueId;
  int nextValueId;
  int atOpId;
  int requiredAfterStage;
  int requiredAfterCluster;
  int requiredAfterOrder;
  std::string reason;
};
```

职责：

- 明确 first-use frontier
- 明确 slot reuse frontier
- 明确 overwrite legality

---

## 11. 严格 Triton 的 pass 链

### 11.1 总顺序

V1 必须固定为：

1. `tb-verify-scope`
2. `tb-build-layout-plan`
3. `tb-build-c-register-plan`
4. `tb-build-mainloop-graph`
5. `tb-assign-latencies`
6. `tb-schedule-loops`
7. `tb-derive-waits`
8. `tb-lower-pipeline-to-nvgpu`
9. `tb-lower-nvgpu-to-nvvm`

### 11.2 每一步的输入输出

| Pass | 输入 | 输出 | 必须满足的 invariant |
|---|---|---|---|
| `tb-verify-scope` | `tb.matmul` | 同 op | 只接受 V1 支持范围 |
| `tb-build-layout-plan` | `KernelSpec` | `tb.layout_plan` | A/B shared 与 fragment 先固定 |
| `tb-build-c-register-plan` | `KernelSpec + LayoutPlan` | `tb.c_register_plan` | C path 只允许 direct global vector |
| `tb-build-mainloop-graph` | `KernelSpec + LayoutPlan + CRegisterPlan` | `tb.mainloop_graph` | 图里绝不能提前出现 stage/wait |
| `tb-assign-latencies` | `MainloopGraph` | `tb.latency_plan` | latency 来自 indirection 与 pipelineability |
| `tb-schedule-loops` | `MainloopGraph + LatencyPlan` | `tb.schedule_plan` | key-op first，distance-one backfill |
| `tb-derive-waits` | `MainloopGraph + LatencyPlan + SchedulePlan` | `tb.wait_plan` | wait 来自 first-use frontier |
| `tb-lower-pipeline-to-nvgpu` | 全部计划 attr | `nvgpu/gpu/vector/...` | lowering 只消费，不反推 |

### 11.3 这条 pass 链故意不保留的层

V1 不保留当前项目里这些厚中间层：

- `ScheduledProgram` 作为第二 owner
- `AsyncCopyPlan` 作为第二 producer truth
- `CDirectConvertPlan` 作为 C relay owner

原因：

- Triton 思想是 layout + graph + latency + schedule + lowering
- 不是 graph 之后再造多个镜像层共享 owner

如果需要 debug，可做：

- `tb-dump-mainloop-graph`
- `tb-dump-latency-plan`
- `tb-dump-schedule-plan`
- `tb-dump-wait-plan`

但 debug dump 不是新的 owner。

---

## 12. “不能特判、不能模板化”在实现上到底意味着什么

### 12.1 允许的东西

允许：

- 合法 config 表
- 硬件指令形状表
- cp.async / vector 宽度对齐表
- 由 config 推导出的 tile 维度和 warp decomposition

例如下面这种是允许的：

```cpp
struct SupportedConfig {
  int blockM, blockN, blockK;
  int numWarps, numStages;
};

static constexpr SupportedConfig kConfigs[] = {
  {64, 64, 32, 1, 2},
  {128, 128, 32, 4, 2},
};
```

因为这只是“支持范围声明”，不是“调度答案”。

### 12.2 禁止的东西

下面这些都禁止：

```cpp
if (blockM == 64 && blockN == 64)
  return preBakedSchedule64();

if (blockM == 128 && blockN == 128)
  return preBakedSchedule128();
```

```cpp
static constexpr int kWaitBeforeMma64[] = { ... };
static constexpr int kWaitBeforeMma128[] = { ... };
```

```cpp
switch (shapeCase) {
case kTiny64:
  useDirectNoBarrier();
case kExact128:
  useSharedRelay();
}
```

禁止的根本原因是：

- 这些不是 Triton 思想；
- 这些只是把结论写死。

### 12.3 第一版该怎么做到“既不特判又只支持窄范围”

正确方式是：

- 用同一套 graph builder
- 用同一套 latency 规则
- 用同一套 schedule 规则
- 用同一套 wait frontier 规则

让 `64x64x32` 和 `128x128x32` 都通过同一条算法链得到结果。

不同 shape 的差异，只允许来自：

- `KernelSpec`
- `LayoutPlan`
- `MainloopGraph` 的规模
- 真实 longest path / first-use frontier

不允许来自：

- 预写好的答案分支

---

## 13. V1 的 direct C path 必须怎么做

### 13.1 总原则

对 V1 官方 exact path，C 只允许：

- `global -> register` 初始化
- `register -> global` 写回

不允许：

- `global -> shared -> register`
- `register -> shared -> global`
- 任何 f32 C shared scratch

### 13.2 这和 Triton 的对齐点

当前本机 Triton exact case 的 TTGIR 没有 f32 C shared memdesc。

V1 必须达到同类结构特征：

- direct C path 没有 shared relay
- 只允许 lane-local register regroup / shuffle
- 不能通过 shared scratch 完成 pack / transpose / landing

### 13.3 允许的局部变换

允许：

- `vector.extract`
- `vector.insert`
- `vector.shuffle`
- lane-local reorder

不允许：

- 以 shared scratch 作为中转站
- 为 direct path 插入额外 `barrier`
- executor 运行时创建 C relay view

---

## 14. V1 lowering 设计

### 14.1 lowering 的 owner 边界

`tb-lower-pipeline-to-nvgpu` 只负责消费：

- `tb.layout_plan`
- `tb.c_register_plan`
- `tb.mainloop_graph`
- `tb.latency_plan`
- `tb.schedule_plan`
- `tb.wait_plan`

它不负责重新决定：

- 哪个 load 要不要 async
- 哪个 wait 放哪里
- C 是 direct 还是 relay
- stage 怎么摆

### 14.2 lowering 需要产出的主要指令

V1 的目标 IR 主体应接近：

- async copy for A/B
- async wait + CTA barrier
- shared/local load or `ldmatrix`
- `mma.sync`
- direct global C load/store

### 14.3 V1 不做的 lowering 花活

V1 不做：

- 通用 conversion pattern 大全
- 跨多 CTA 的 shared/fence 策略
- 复杂 epilogue fusion
- 通用 layout conversion 清理器

如果 V1 性能没到位，优先查主链，而不是继续加 lowering 花活。

### 14.4 V1 之后如何引入 `scf`

`scf` 不是 V1 前半段主链的必须 owner。

原因不是 `scf` 没价值，而是：

- V1 当前最重要的是保证 owner 单一；
- `graph / latency / schedule / wait` 先由 `tb` plan 固定；
- 避免一边有 plan，一边又从 loop body 里重新推一遍调度真相。

因此，`scf` 最合适的引入时机不是 M3/M4/M5，
而是：

- 两个官方 case 已经 correctness 通过；
- `tb-build-mainloop-graph`
- `tb-assign-latencies`
- `tb-schedule-loops`
- `tb-derive-waits`
- `tb-lower-pipeline-to-nvgpu`

这条 V1 主链已经闭环并接近 Triton 性能之后。

也就是说：

- 先做 parity；
- 先把 owner 钉死；
- 再考虑把执行骨架显式物化成 `scf`。

### 14.5 `scf` 的正确放置位置

如果后续要引入 `scf`，推荐新增：

1. `tb-materialize-scf-pipeline`
2. `tb-lower-scf-pipeline-to-nvgpu`

推荐 pass 顺序变为：

1. `tb-verify-scope`
2. `tb-build-layout-plan`
3. `tb-build-c-register-plan`
4. `tb-build-mainloop-graph`
5. `tb-assign-latencies`
6. `tb-schedule-loops`
7. `tb-derive-waits`
8. `tb-materialize-scf-pipeline`
9. `tb-lower-scf-pipeline-to-nvgpu`
10. `tb-lower-nvgpu-to-nvvm`

这里 `tb-materialize-scf-pipeline` 的职责必须很克制：

- 只把已经确定的 plan 物化成 `scf.for` / prologue / kernel / epilogue
- 不重新决定 latency
- 不重新决定 stage
- 不重新决定 wait
- 不重新决定 C landing

换句话说：

- `scf` 在你项目里将来更适合当执行容器；
- 不适合重新当调度 owner。

### 14.6 `tb-materialize-scf-pipeline` 应该产什么 IR

这一步的目标不是重新发明 pipeline，
而是把已经定好的 plan 显式物化成一个可读、可检查、可继续 lowering 的执行骨架。

它的输入必须只有：

- `tb.layout_plan`
- `tb.c_register_plan`
- `tb.mainloop_graph`
- `tb.latency_plan`
- `tb.schedule_plan`
- `tb.wait_plan`

它的输出建议是：

- `scf`
- `arith`
- `memref`
- 必要时少量 `vector`

在这一层，仍然不应该直接生成：

- `nvgpu`
- `nvvm`

因为这一步的职责只是把执行骨架显式化，
不是直接下沉到最终硬件指令。

推荐的 IR 骨架形态是：

1. loop 外：
   - materialize C 初始化需要的 direct-global register load
   - materialize A/B 的 prologue issue
2. loop 内：
   - 一个 `scf.for`
   - `iter_args` 只携带真正跨迭代存活的值
   - 包括 buffer slot index / async token / accumulator state
3. loop 后：
   - epilogue wait
   - direct global C store

可以接受的伪 IR 轮廓大致是：

```mlir
%c0 = arith.constant 0 : index
%c32 = arith.constant 32 : index
%acc0 = ...            // direct C init, not shared relay
%slotA0 = ...
%slotB0 = ...
%tok0 = ...

scf.for %k = %c0 to %kEnd step %c32
    iter_args(%slotA = %slotA0, %slotB = %slotB0,
              %tok = %tok0, %acc = %acc0) -> (...) {
  // issue async copies according to schedule_plan
  // waits inserted exactly from wait_plan
  // local fragment load according to layout_plan
  // mma/update acc according to mainloop_graph + schedule_plan
  scf.yield %slotA_next, %slotB_next, %tok_next, %acc_next : ...
}

// epilogue wait
// direct global C store
```

这里最关键的 invariant 是：

- `scf.for` 只是执行容器；
- `iter_args` 只是显式承载 loop-carried state；
- stage / cluster / wait / landing 的真相仍然来自 plan；
- 不能在 materialize 阶段再通过 IR 局部形状重新猜一遍答案。

因此这一步允许做的是：

- 把 prologue / kernel / epilogue 从 plan 中展开出来
- 把 loop-carried token / slot / acc 显式化
- 把 C path 的 direct-register 生命周期显式化

这一步不允许做的是：

- 因为 `scf.for` 看起来更自然，就现场重排 stage
- 因为当前 case 是 `64` 或 `128`，就选另一套 skeleton
- 在这里引入新的 shared relay
- 在这里改变 wait frontier

一句话说：

- `tb-materialize-scf-pipeline` 是 plan 的显式打印版和执行骨架版；
- 不是第二个调度器。

### 14.7 哪些数据结构以后继续保留

即使未来引入 `scf`，下面这些结构也建议继续保留：

- `KernelSpec`
- `LayoutPlan`
- `CRegisterPlan`
- `MainloopGraph`

原因：

- `KernelSpec` 是合法 config 的唯一描述；
- `LayoutPlan` 是 shared/fragment/register 真相；
- `CRegisterPlan` 是 direct C path 的 owner；
- `MainloopGraph` 是 first-use / last-use / slot frontier 的分析载体。

后续可以逐步弱化、更多写回 IR attr 的，是：

- `LatencyPlan`
- `SchedulePlan`
- `WaitPlan`

但第一版不要急着把这些完全并回 `scf` 或通用 IR，
否则很容易重新回到“语义散落在 IR 细节里”的状态。

### 14.8 后续官方方言引入优先级

V1 当前明确依赖的前半段主链仍然是：

- `builtin`
- `func`
- `memref`
- `tb`

V1 当前明确依赖的后半段 lowering 方言是：

- `arith`
- `vector`
- `gpu`
- `nvgpu`
- `nvvm`
- `llvm`

在此基础上，后续官方方言的引入优先级建议如下。

第一优先级：

- `scf`

适用时机：

- 开始支持真正的 `K` 主循环
- 需要显式 prologue / kernel / epilogue
- 要从 exact-tile 扩到更一般的 loop materialization

第二优先级：

- `cf`
- `math`

适用时机：

- kernel 控制流开始复杂化
- 开始做更一般的 epilogue / activation / scale 等小型算子

低优先级：

- `tensor`
- `linalg`
- `affine`
- `transform`

原因不是这些方言没用，而是：

- `tensor / linalg` 更适合做高层前端入口或互操作层；
- `affine` 更适合更通用的 loop/address 变换；
- `transform` 更适合实验调度、自动化变换和 autotune 驱动；
- 它们都不是当前“窄形状 parity 主线”的第一优先级。

### 14.9 哪些点必须尽量 1:1 对齐 Triton，哪些点只要语义对齐即可

这里必须明确区分两类东西：

1. 直接决定性能语义和 owner 边界的真相
2. 工程包装、类名和具体组织方式

第一类必须尽量贴住 Triton；
第二类不需要机械照抄，只要语义不偏即可。

必须尽量 1:1 对齐 Triton 的，是：

- pass 依赖顺序
- C path 的 direct register landing
- A/B producer 的 async copy / wait / local load / mma 主线
- latency 的来源必须是 indirection + pipelineability
- schedule 的来源必须是 latency-driven，而不是 per-shape 模板
- wait 的来源必须是 use frontier / slot reuse frontier
- lowering 只能消费 plan，不能重新猜 stage / wait / landing

这些地方如果不贴，通常会直接导致：

- C relay 回来
- wait 过保守
- shared footprint 变大
- scoreboard stall 上升
- HMMA 外围 glue 变厚

只要语义对齐即可，不必工程形态 1:1 复制的，是：

- 内部数据结构名字
- pass 内部 helper 的类层次
- attr 的编码细节
- 文档结构、目录结构、文件拆分方式
- 是否立刻把 `scf` 纳入主链容器
- 是否直接复刻 Triton 的 `Operation*` / `Value` 驱动写法

原因不是这些东西不重要，而是：

- 它们主要影响工程可维护性和实现成本；
- 不直接决定窄范围官方 case 的性能上限；
- 对你这个 V1 来说，更重要的是保证 owner 单一、依赖方向干净。

判断标准可以很简单：

- 如果一个点决定的是“最终硬件上看见什么执行结构”，那它应尽量 1:1 对齐 Triton；
- 如果一个点决定的是“代码在仓库里长什么样、类怎么命名、helper 怎么拆”，那它通常只需要语义对齐。

因此：

- 必须严格学 Triton 的性能语义；
- 不必机械抄 Triton 的工程外形。

---

## 15. 官方验收标准

### 15.1 官方形状集

V1 只认下面两组官方 case：

1. `64x64x32`, `num_warps=1`, `num_stages=2`
2. `128x128x32`, `num_warps=4`, `num_stages=2`

### 15.2 公平测量协议

必须保证：

- 同一台机器
- 同一 CUDA / driver / 时钟环境
- 同一 dtype
- 同一 exact-tile 语义
- 同一 warmup 次数
- 同一轮数的计时
- backend 和 Triton 都测：
  - `nsys kernel time`
  - `ncu gpu__time_duration.sum`
  - `eligible warps`
  - `stall_long_scoreboard`
  - `tensor pipe active`
  - `smem_allocated`
  - `regs_per_thread`

### 15.3 V1 结构验收

以下全部满足才允许谈性能：

1. direct C path 不出现 f32 shared scratch
2. runtime IR 不出现 C relay memref
3. lowering 层不持有第二套 wait / stage / landing 真相
4. A/B producer 通过显式 async copy / wait 主线执行
5. `MainloopGraph` 不带 stage 预设

### 15.4 V1 性能验收

V1 的完成标准定义为：

1. 两个官方 case 都 correctness 通过
2. 两个官方 case 都满足结构验收
3. 两个官方 case 的 median kernel time 不超过 Triton 的 `1.05x`
4. 至少一个官方 case 达到 Triton `1.00x` 以内

如果只做到：

- 正确
- 能跑
- 某个 case 只有 `1.3x ~ 1.5x`

那不算 V1 完成。

### 15.5 诊断红线

如果出现下面任意一条，必须先停下来查根因：

- `eligible_warps` 明显低于 Triton，而 `active_warps` 接近 Triton
- `stall_long_scoreboard` 显著高于 Triton
- `smem_allocated` 明显大于 Triton
- 出现额外的 f32 shared relay
- HMMA 数量和 Triton 明显不一致

这些现象通常说明：

- wait frontier 太保守
- C 路径仍在 relay
- shared footprint 过大
- lowering 插入了多余 glue

---

## 16. 实施路线

### 16.1 M0：冻结 Triton baseline

产物：

- 官方两组 case 的 Triton benchmark 脚本
- `nsys`
- `ncu`
- TTGIR / lowered IR dump

目标：

- 冻结官方对照组
- 后续每一步只和这一版对比

### 16.2 M1：搭出 MLIR 外壳

产物：

- `tb.matmul`
- verifier
- `tb-opt`
- `tb-verify-scope`

停止条件：

- 非法 case 被 verifier 拒绝
- 两个官方 case 能稳定进入 pass pipeline

### 16.3 M2：完成 `LayoutPlan + CRegisterPlan`

产物：

- A/B shared spec
- fragment/register map
- C direct register map

停止条件：

- direct C path 设计层明确不需要 shared relay
- 两个官方 case 的 layout/register map 都可打印、可检查

### 16.4 M3：完成 `MainloopGraph`

产物：

- values / bundles / slots / ops
- frontier dump

停止条件：

- graph 不携带 stage/wait
- 两个官方 case 都能生成纯 dataflow graph

### 16.5 M4：完成 `AssignLatencies`

产物：

- load indirection level
- target latency
- MMA self latency
- acc multibuffering decision

停止条件：

- 不存在 per-shape schedule 分支
- 规则完全由 graph + config 推导

### 16.6 M5：完成 `ScheduleLoops`

产物：

- stage/cluster/order
- placement reasons

停止条件：

- 先 key ops，再 distance-one deps，再 remaining ops
- 没有静态模板 schedule

### 16.7 M6：完成 `WaitPlan`

产物：

- lifetimes
- waits
- overwrites

停止条件：

- wait 只由 first-use frontier 导出
- overwrite 只由 slot retirement legality 导出

### 16.8 M7：完成 `LowerPipelineToNVGPU`

产物：

- async copy / wait / barrier
- ldmatrix/local load
- mma.sync
- direct global C load/store

停止条件：

- IR 中无 C relay
- lowering 不再重新猜 schedule / wait / landing

### 16.9 M8：做 parity closure

产物：

- backend vs Triton benchmark report
- profiler report
- 差距根因表

停止条件：

- 达到第 15 节的官方验收标准

---

## 17. 和当前 `triton_backend_nvgpu` 的关系

当前项目仍然有价值，但它不应该再作为 V1 主架构模板。

它更适合保留为：

- correctness oracle
- performance oracle
- profiling oracle
- 反例库

不适合继续承担：

- V1 的中间层主线模板
- V1 的 executor 结构模板
- V1 的 owner 边界模板

如果要从旧项目复用，最多只复用：

- 数学帮助函数
- benchmark 工具
- profiling 脚本
- 局部 layout / register 计算逻辑

不复用：

- 当前厚 executor 主线
- 当前中间镜像层叠加结构
- 当前 direct C relay 相关语义

---

## 18. 这份方案最终要求什么

第一版的目标不是“像 Triton 一点点”。

第一版的要求是：

- 只做很窄的范围；
- 但在这个窄范围里，从布局、graph、latency、schedule、wait 到 lowering，全部按 Triton 的依赖顺序组织；
- 绝不靠特判、模板调度、执行层补丁去凑结果；
- 最终直接用真实硬件指标和 Triton 对齐。

一句话总结：

- 第一版必须是“窄而真”的 Triton 式编译器闭环；
- 不是“功能最少”的教学版；
- 也不是“当前项目再修一点”的折中版。

只要这个闭环做出来，并且官方窄形状集实测贴到 Triton，项目就已经足够强，既能讲编译器结构，也能讲硬件性能闭环。
