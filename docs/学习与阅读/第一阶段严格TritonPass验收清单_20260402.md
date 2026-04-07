# `mini_triton_nvgpu_v1` Stage1 严格 Triton 思想逐 Pass 验收清单

## 1. 文档目的

这份文档不是再讲一遍原理，而是给后续执行直接使用的 checklist。

使用方式只有一种：

1. 做完一个 pass。
2. 按本文对应章节逐条检查。
3. 只有当前 pass 验收通过，才允许进入下一个 pass。

本文配套的原理文档是：

- [/home/zhangruiqi/docs/mini_triton_v1_stage1_strict_triton_optimization_writing_validation_and_effect_proof_guide_20260402.md](/home/zhangruiqi/docs/mini_triton_v1_stage1_strict_triton_optimization_writing_validation_and_effect_proof_guide_20260402.md)

当前 Stage1 主线顺序以这里为准：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)

---

## 2. 统一验收规则

### 2.1 统一通过条件

每个 pass 要算通过，至少要同时满足下面三条：

- 当前层 owner truth 已经进入 IR attr/type 或 public plan。
- 下一层将只消费这份 truth，不需要重新猜。
- 当前层没有把本应属于别层的责任偷偷带进来。

### 2.2 统一失败信号

下面任意一条出现，都算当前 pass 未通过：

- 需要 lowering 再补猜当前层真相。
- 新旧两套 owner 同时存在。
- 看起来生成了正确指令，但合同层仍然不闭合。
- 只能通过某个临时 fallback 路线跑通。
- 必须依赖硬编码常量而不是 plan/attr 才能成立。

### 2.3 统一命令模板

单步检查：

```bash
/home/zhangruiqi/mini_triton_nvgpu_v1/build/bin/tb-opt \
  --tb-verify-scope \
  --tb-attach-target-info='gpu-arch=sm_86 ptx-features=+ptx60' \
  --tb-semanticize-matmul \
  --tb-build-program-mapping-plan \
  --tb-build-layout-plan \
  --tb-build-transport-plan \
  --tb-build-c-register-plan \
  --tb-rewrite-matmul-mainloop \
  --tb-cleanup-layout-conversions \
  --tb-build-mainloop-graph \
  --tb-regularize-k-loop \
  --tb-assign-latencies \
  --tb-schedule-loops \
  --tb-derive-waits \
  --tb-expand-pipeline \
  --tb-cleanup-pipeline \
  /home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir
```

完整 lowering：

```bash
/home/zhangruiqi/mini_triton_nvgpu_v1/build/bin/tb-opt \
  --pass-pipeline='builtin.module(tb-stage1-full-to-nvvm-pipeline{cubin-chip=sm_86 cubin-features=+ptx60 cubin-format=isa})' \
  /home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir
```

---

## 3. 总体打勾表

| Pass | 当前层 owner | 通过后最核心证据 |
| --- | --- | --- |
| `TBVerifyScope` | 支持边界 | 非法 case 立刻失败 |
| `TBAttachTargetInfo` | target truth | module `tb.target` 存在 |
| `TBSemanticizeMatmul` | semantic truth | `tb.semantic_matmul` 存在 |
| `TBBuildProgramMappingPlan` | program/CTA mapping | `tb.program_mapping_plan` 和 module `tb.num-ctas` |
| `TBBuildLayoutPlan` | layout/fragment truth | `tb.encoding_plan` 完整 |
| `TBBuildTransportPlan` | transport truth | `tb.transport_plan` 完整 |
| `TBBuildCRegisterPlan` | accumulator/epilogue truth | `tb.accumulator_plan`、`tb.epilogue_plan` |
| `TBRewriteMatmulMainloop` | tensor-core mainloop truth | `tb.matmul_rewrite` 完整 |
| `TBCleanupLayoutConversions` | layout cleanup truth | 冗余 `tb.convert_layout` 删除 |
| `TBBuildMainloopGraph` | resource graph truth | `tb.buffer_model` 完整 |
| `TBRegularizeKLoop` | loop owner truth | `tb.loop_plan` 完整 |
| `TBAssignLatencies` | latency truth | `tb.latency_plan` 完整 |
| `TBScheduleLoops` | coarse schedule truth | `tb.pipeline_plan` 完整 |
| `TBDeriveWaits` | async frontier truth | `tb.async_plan` 完整 |
| `TBExpandPipeline` | expanded cluster truth | `tb.pipeline_expansion` 完整 |
| `TBCleanupPipeline` | strict mainline truth | `tb.pipeline_mainline`、`tb.pipeline_ready` |
| `TBLowerPipelineToNVGPU` | target materialization | `device_async_copy` / `ldmatrix` / `mma.sync` / direct C path |

---

## 4. 逐 Pass 验收清单

## 4.1 `TBVerifyScope`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/VerifyScope.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/VerifyScope.cpp)

输入前提：

- `tb.matmul` 已存在。
- `KernelConfig` 可被解析。

必须出现：

- 对不支持的 kernel 直接失败。
- 对支持范围内 kernel 原样通过。

必须不出现：

- 自动补默认值。
- 在 verifier 中做 rewrite。
- 放行非法 case，期待后续 fallback。

检查点：

- `block_m/block_n/block_k`、`mma`、`num_warps`、`requested_stages` 超出范围时必须报错。
- 合法输入时不新增任何“补救 attr”。

通过标志：

- 这一层只做过滤，不制造新真相。

## 4.2 `TBAttachTargetInfo`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp)

输入前提：

- module 上已有 `tb.num-warps`、`tb.requested-stages`。
- pipeline 参数中显式提供 `gpu-arch`。

必须出现：

- module `tb.target`
- module `tb.threads-per-warp`

必须不出现：

- hidden default chip
- per-op target attr 继续作为主 owner

检查点：

- 缺失 `tb.num-warps` 或 `tb.requested-stages` 必须失败。
- 缺失 `gpu-arch` 必须失败。
- 下游只读 module target truth。

通过标志：

- 后续 pass 不再需要自己推 target 能力。

## 4.3 `TBSemanticizeMatmul`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/SemanticizeMatmul.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/SemanticizeMatmul.cpp)

必须出现：

- `tb.semantic_matmul`
- `problemM/N/K`
- `tileM/N/K`
- `exactTile`
- `hasBoundaryM/N/K`
- A/B/C descriptor type truth

必须不出现：

- warpGrid
- shared physical layout
- transport legality

检查点：

- exact-tile case 的边界字段必须自洽。
- descriptor type 必须直接可被后续 layout pass 消费。

通过标志：

- 后续不再从原始 `tb.matmul` 重新推 semantic truth。

## 4.4 `TBBuildProgramMappingPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildProgramMappingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildProgramMappingPlan.cpp)

必须出现：

- `tb.program_mapping_plan`
- module `tb.num-ctas`
- `mappingKind`
- `launchOrder`
- `groupM`
- `totalPrograms`
- `numCTAs`

必须不出现：

- `AttachTargetInfo` 或 lowerer 中硬写 `num_ctas`
- CTA 内部 pipeline 信息混进 program mapping

检查点：

- grouped launch 公式组件必须都在 plan 中。
- lowerer 只能消费这些组件。

通过标志：

- program id 到 tile 坐标的逻辑只来自这份 plan。

## 4.5 `TBBuildLayoutPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildLayoutPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildLayoutPlan.cpp)

必须出现：

- `tb.encoding_plan`
- A global encoding
- B global encoding
- A shared encoding
- B shared encoding
- A dot encoding
- B dot encoding
- accumulator encoding
- C store encoding
- fragment lane mapping truth

必须不出现：

- transport legality 混入 `SharedEncodingSpec`
- lowerer 重新发明 fragment lane mapping
- 旧 `tb.layout_plan` 继续作为主 owner

检查点：

- `FragmentEncodingSpec` 字段完整。
- `getSharedEncodingSpec` 只提供 layout/alloc truth。
- 所有后续 pass 都消费 `tb.encoding_plan`。

通过标志：

- 这是 `ldmatrix`、`mma.sync`、C direct path 的唯一物理 layout owner。

## 4.6 `TBBuildTransportPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildTransportPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildTransportPlan.cpp)

必须出现：

- `tb.transport_plan`
- operandA transport spec
- operandB transport spec
- `vectorBytes`
- `asyncVectorBytes`
- `transactionBytes`
- `asyncEligible`

必须不出现：

- `EncodingPlan` 继续偷偷拥有 transport legality
- 因为 exact-tile 就直接默认 async
- lowerer 中硬编码 async 宽度

检查点：

- transport legality 由 `TargetInfo + EncodingPlan` 推导。
- `asyncEligible` 是合同字段，不是推测结果。

通过标志：

- 后续 async plan 和 lowering 只消费 `tb.transport_plan`。

## 4.7 `TBBuildCRegisterPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp)

必须出现：

- `tb.accumulator_plan`
- `tb.epilogue_plan`
- register packs
- `initMode`
- `storeMode`

必须不出现：

- direct path 仍带 relay scratch truth
- shared physical backing 混进 direct path
- lowerer 中再决定 direct/relay

检查点：

- direct-global case 下，init/store 都是 direct。
- relay case 才允许 shared relay truth。

通过标志：

- C path 的 direct 与 relay owner 在这里一次定死。

## 4.8 `TBRewriteMatmulMainloop`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RewriteMatmulMainloop.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RewriteMatmulMainloop.cpp)

必须出现：

- `tb.matmul_rewrite`
- `mainloopKind`
- `instructionM/N/K`
- `kGroups`
- `aPath`
- `bPath`
- direct accumulator init/store truth

必须不出现：

- generic tile matmul 直接留给 lowerer 再凑 tensor-core 主线
- fragment path 在 loop/schedule 阶段再补

检查点：

- rewrite 只依赖 `EncodingPlan + AccumulatorPlan + EpiloguePlan`。
- operand fragment path 是显式 owner。

通过标志：

- 从这一层开始，主线已经是 tensor-core mainloop，而不是 generic matmul。

## 4.9 `TBCleanupLayoutConversions`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupLayoutConversions.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupLayoutConversions.cpp)

必须出现：

- identity conversion 删除
- immediate round-trip 删除

必须不出现：

- 删除仍承载真实 encoding 区别的转换
- 通过 cleanup 改写 layout truth

检查点：

- 清理前后，逻辑 encoding 边界仍然正确。
- 冗余 `tb.convert_layout` 数量下降。

通过标志：

- IR 更干净，但 layout owner 不变。

## 4.10 `TBBuildMainloopGraph`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildMainloopGraph.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildMainloopGraph.cpp)

必须出现：

- `tb.buffer_model`
- backings
- views
- values
- ops

必须不出现：

- target-specific opcode 决策
- 无 owner 的 value/view/backing

检查点：

- 每个 pipeline op 都有 inputs 和 outputs。
- operand、accumulator、epilogue 资源都能在图中找到。

通过标志：

- 后续 latency/schedule/wait 不再依赖猜测资源关系。

## 4.11 `TBRegularizeKLoop`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RegularizeKLoop.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RegularizeKLoop.cpp)

必须出现：

- `tb.loop_plan`
- `singleMainLoop = true`
- `iterationCount`
- loop-carried values

必须不出现：

- 多个等价 main loop owner
- scheduler 仍需要自己识别主 K-loop

检查点：

- `iterationCount` 与 mainloop `kGroups` 可对齐。
- carried values 明确记录 owner。

通过标志：

- pipeline 将围绕这条唯一主 K-loop 展开。

## 4.12 `TBAssignLatencies`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AssignLatencies.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AssignLatencies.cpp)

必须出现：

- `tb.latency_plan`
- 每个 op 的 `targetLatency`
- `selfLatency`
- `bufferDistance`
- `pipelineable`

必须不出现：

- scheduler 内部私有 latency 真相
- 固定硬编码 latency 不经过 `TargetInfo`

检查点：

- latency plan 依赖 `TargetInfo + BufferModel + LoopPlan`。
- `reason` 字段可解释每个 latency 决策。

通过标志：

- 后续 schedule 只消费公开 latency truth。

## 4.13 `TBScheduleLoops`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ScheduleLoops.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ScheduleLoops.cpp)

必须出现：

- `tb.pipeline_plan`
- `scheduledMaxStage`
- 全部 op 的 `stage`
- 全部 op 的 `cluster`
- 全部 op 的 `order`
- `stageOwnedBuffers`

必须不出现：

- lowerer 再给 op 重排顺序
- 通过 barrier 修 schedule
- scheduler 偷偷修改 transport 或 epilogue truth

检查点：

- 所有主线 op 被完整调度。
- producer、consumer、compute 的相对时序可读。

通过标志：

- 当前层已经给出 coarse schedule owner。

## 4.14 `TBDeriveWaits`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/DeriveWaits.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/DeriveWaits.cpp)

必须出现：

- `tb.async_plan`
- async producers
- async groups
- waits
- reuse fences

必须不出现：

- exact-tile async 主线只有 barrier 没有 wait
- wait owner 留给 lowerer 再猜
- 复用边界靠“经验上应该没问题”

检查点：

- producer/group/wait/reuse 关系闭合。
- `beforeOpId`、`groupId`、`srcOffsets` 明确。

通过标志：

- async frontier 在这一层已经完全显式化。

## 4.15 `TBExpandPipeline`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ExpandPipeline.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ExpandPipeline.cpp)

必须出现：

- `tb.pipeline_expansion`
- `AsyncIssue` clusters
- `ConsumerWait` clusters
- `MmaCompute` clusters
- `kGroup`
- `waitGroupIds`
- `needsBarrier`

必须不出现：

- lowerer 再按 op 顺序自己分 cluster
- `k_group` 仍然是隐式信息

检查点：

- 所有 op 覆盖一次且只覆盖一次。
- 所有 wait group 覆盖一次且只覆盖一次。
- `kGroup` 连续完整。

通过标志：

- expanded cluster stream 成为 lowering 前最后的显式 pipeline owner。

## 4.16 `TBCleanupPipeline`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupPipeline.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupPipeline.cpp)

必须出现：

- `tb.pipeline_mainline`
- `tb.pipeline_ready`
- 按 stage/cluster 有序的 cluster stream

必须不出现：

- 非 direct-global epilogue 混入 strict mainline
- 非 legal `cp.async` producer 混入 strict mainline
- expansion 未完全覆盖仍然往下放行

检查点：

- direct-global epilogue 是强前提。
- async producer 是 legal `cp.async` 是强前提。
- `tb.matmul` 被 `tb.pipeline_mainline` 取代。

通过标志：

- lowering 不再需要隐藏 skeleton 和补丁逻辑。

## 4.17 `TBLowerPipelineToNVGPU`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

必须出现：

- `nvgpu.device_async_copy`
- `nvgpu.device_async_wait`
- `nvgpu.ldmatrix`
- `nvgpu.mma.sync`
- direct-global C vector path

必须不出现：

- direct C path 的 shared relay backing
- lowerer 重建 transport/layout/epilogue truth
- 主线 cluster 顺序与 `tb.pipeline_mainline` 脱钩

检查点：

- A/B producer 只来自 `tb.transport_plan + tb.async_plan`。
- shared->register 只来自 `tb.encoding_plan`。
- C path 只来自 `tb.epilogue_plan`。
- program mapping 只来自 `tb.program_mapping_plan`。

通过标志：

- 这是“按合同 materialize”，不是“重新设计策略”。

---

## 5. Lowering 之后的统一检查表

## 5.1 NVGPU 层

必须看到：

- `nvgpu.device_async_copy`
- `nvgpu.device_async_wait`
- `nvgpu.ldmatrix`
- `nvgpu.mma.sync`

必须明显改善：

- barrier 数量
- C path 的 shared relay 膨胀

## 5.2 PTX 层

必须看到：

- `cp.async`
- `cp.async.wait_group`
- `ldmatrix`
- `mma.sync`

不应看到：

- A/B 主线退化成纯同步 `ld/st/bar`

## 5.3 SASS 层

必须看到：

- `LDGSTS`
- `LDGDEPBAR` 或 `DEPBAR`
- `LDSM`
- `HMMA`

不应看到：

- 因 C relay 人为增加的大量 `BAR.SYNC`

---

## 6. 性能验收清单

只有前面合同层、IR 层、lowering 层都通过，才允许做性能验收。

性能验收必须满足：

- 形状一致
- `num_warps` 一致
- `requested_stages` 一致
- `num_ctas` 一致
- target `sm` 一致
- 同类 exact-tile case
- 同类 direct epilogue path

性能结论至少要同时看三件事：

- runtime
- 真实硬件指令链是否对齐
- NCU 指标是否支持这个 runtime 结论

下面这些不算通过：

- 只在一个规模上偶然快
- 指令链还是错的，但测量偶然快
- path 不同还硬拿来和 Triton 比

---

## 7. 使用顺序

后续每做一个 pass，都按下面顺序走：

1. 看本章对应 checklist。
2. 先过“必须出现 / 必须不出现”。
3. 再过“检查点”。
4. 再看是否满足“通过标志”。
5. 只有通过，才继续下一个 pass。
6. 全部通过后，再做 lowering 和性能验收。

---

## 8. 最后结论

这份 checklist 的目标只有一个：

- 防止 Stage1 再退化成“指令像 Triton，但 owner 不是 Triton”；
- 防止后续继续把本应在前层表达的真相塞回 lowering；
- 防止用偶然跑快一次来替代严格验收。

只要严格按这份清单走，后面每个 pass 是否真正站在 Triton 思想上，就能被逐层检查出来。
