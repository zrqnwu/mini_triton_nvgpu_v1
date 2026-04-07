# `mini_triton_nvgpu_v1` Stage1 严格 Triton 思想优化编写、验证与效果证明文档

## 1. 文档目的

这份文档回答四类实际执行问题：

- Stage1 这条 pass 主线里，每一步到底在优化什么；
- 每一步在当前代码里对应什么结构、什么 attr、什么 lowering 结果；
- 每一步应该怎么写，才算严格参考 Triton 思想，而不是“能跑的近似实现”；
- 每一步做完之后，应该怎么验证，怎么证明它真的有效果。

本文是执行文档，不是抽象介绍。

本文只讨论当前 `mini_triton_nvgpu_v1` 的 Stage1 矩阵乘法主线，默认边界是：

- 单核 matmul；
- 先聚焦 `exact-tile`；
- 单 CTA 主线；
- `f16xf16 -> f32`；
- `mma.sync m16n8k16`；
- 先闭合性能主链，再谈扩展覆盖面。

---

## 2. 参考基线

当前项目的 Stage1 主线顺序，以这里为准：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)

Triton NVIDIA backend 的真实主线思想，以这里为准：

- [/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py](/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py)

当前项目对数据结构和 owner truth 的最终设计，以这两份文档为前置：

- [/home/zhangruiqi/docs/mini_triton_v1_mlir_semantic_hard_truth_data_structure_final_design_20260402.md](/home/zhangruiqi/docs/mini_triton_v1_mlir_semantic_hard_truth_data_structure_final_design_20260402.md)
- [/home/zhangruiqi/docs/mini_triton_v1_triton_optimization_execution_and_strict_pass_acceptance_plan_20260402.md](/home/zhangruiqi/docs/mini_triton_v1_triton_optimization_execution_and_strict_pass_acceptance_plan_20260402.md)

一句话概括当前 Stage1 的真实顺序：

`verify -> target -> semanticize -> program mapping -> layout -> transport -> accumulator/epilogue -> matmul rewrite -> cleanup conversions -> buffer graph -> loop regularization -> latency -> schedule -> waits -> pipeline expansion -> pipeline cleanup -> NVGPU lowering -> NVVM/PTX`

---

## 3. 什么叫“严格参考 Triton 思想”

这里的“严格参考 Triton 思想”，不是指外形一模一样，而是指下面五条必须同时成立。

### 3.1 顺序必须等价

必须先定语义真相和 layout/memory legality，再做 matmul rewrite、loop regularization、pipeline、lowering。

不允许倒过来做。

尤其不允许：

- 先在 lowering 里写 `cp.async`，再回头补 transport owner；
- 先在 lowering 里写 `mma.sync`，再回头补 fragment contract；
- 先在 executor 里凑 C relay，再回头补 epilogue truth。

### 3.2 ownership 必须单一

每一种真相都只能有一个 owner。

例如：

- layout 真相由 `tb.encoding_plan` 和相关 attr/type owner 持有；
- transport legality 真相由 `tb.transport_plan` 持有；
- async producer/group/wait 真相由 `tb.async_plan` 持有；
- C 路径是 direct 还是 relay，由 `tb.epilogue_plan` 持有；
- pipeline cluster 顺序和 `k_group` owner，由 `tb.pipeline_expansion` 和 `tb.pipeline_mainline` 持有。

不允许“双真相”。

例如：

- `EncodingPlan::SharedEncodingSpec` 同时偷偷携带 transport legality；
- `EpiloguePlan` 说 direct，但 lowering 里还是分配 shared relay；
- `pipeline_expansion` 已经有 wait owner，lowering 还按 stage/cluster 再重组一遍 wait。

### 3.3 lowering 只能消费，不能再发明

严格 Triton 思想下，lowering 的职责是“翻译合同”，不是“重做分析”。

不允许在 lowering 里重新决定：

- async copy 粒度；
- shared alloc 形状；
- fragment lane mapping；
- direct epilogue 是否改走 relay；
- 某个 wait 应该挂在哪个 consumer 前面；
- 某个 cluster 属于哪个 `k_group`。

### 3.4 每一步只负责自己的层

每个 pass 只能做本层该做的事。

例如：

- `TBSemanticizeMatmul` 只收语义真相，不负责物理 warp 布局；
- `TBBuildLayoutPlan` 负责 encoding 和 fragment truth，不负责时序；
- `TBScheduleLoops` 负责 stage/cluster/order，不负责重建 transport legality；
- `TBLowerPipelineToNVGPU` 负责 materialization，不负责重新设计 pipeline。

### 3.5 验证口径必须分层

不是所有 pass 都应该直接跑出速度提升。

必须把验收分成五层：

1. 合同层：attr/type/plan 是否存在且自洽；
2. 结构 IR 层：IR 形态是否符合该层职责；
3. target IR 层：NVGPU/PTX 是否出现预期指令链；
4. 机器层：SASS 是否出现预期主链；
5. 性能层：在公平测量下，runtime/NCU 是否改善。

如果一开始就只盯 runtime，很容易把错误的实现误判为成功。

---

## 4. 哪些写法算“简化”或“退化”

下面这些做法，在本文口径里都算违规，不允许作为 Stage1 正解。

### 4.1 把本应在 plan 层表达的真相，偷塞回 lowering

典型违规：

- 在 lowering 中根据 `exactTile` 直接决定是否发 `cp.async`；
- 在 lowering 中根据 tile shape 临时猜 shared alloc shape；
- 在 lowering 中根据某种固定公式猜 wait frontier；
- 在 lowering 中为了图省事给 direct C path 也分配 relay scratch。

### 4.2 用一个 plan 同时承载两个层次的责任

典型违规：

- layout plan 同时负责 transport legality；
- epilogue plan 同时偷偷决定 pipeline wait；
- buffer model 里塞进 target-specific opcode 选择。

### 4.3 用“能跑的 fallback”替代主线 owner

典型违规：

- exact-tile 主线本该是 explicit async，结果直接回退到同步 copy；
- direct C path 本该是纯 global vector，结果走 shared relay 只是因为实现更简单；
- pipeline expansion 本该显式拥有 wait，结果留给 lowerer 再聚合。

### 4.4 用静态硬编码顶替合同

典型违规：

- 在 lowerer 中硬编码 `sm_86` 的 async vector bytes；
- 硬编码 `m16n8k16` 的某种 lane mapping，而不从 `tb.encoding_plan` 读取；
- 直接在 executor 里写死 `num_ctas = 1`，而不是消费 `tb.program_mapping_plan`。

### 4.5 用“外形像 Triton”代替“思想对齐”

典型违规：

- 只是生成了 `mma.sync`，但前面的 layout/fragment owner 仍然是错的；
- 只是生成了 `cp.async`，但 wait frontier 仍由 lowering 猜；
- 只是 PTX 里有 `ldmatrix`，但 C path 还走着 shared relay。

---

## 5. 总体验证与效果证明方法

每一步做完后，验证必须按固定顺序进行。

### 5.1 合同层验证

目标是证明 owner truth 已经进入 IR 或 plan attr，而不是停留在 C++ 临时变量里。

检查内容包括：

- 对应 attr 是否已经写回 op 或 module；
- attr 字段是否完整；
- 相关旧 attr 是否已经移除；
- verifier 是否会在合同不自洽时失败。

### 5.2 结构 IR 层验证

目标是证明这一层的 IR 形态已经对齐该层职责。

检查内容包括：

- 不该存在的过渡 op 是否被删掉；
- 主线 op 是否已经显式出现；
- 依赖覆盖是否完整；
- 是否还存在 dual truth 的痕迹。

### 5.3 Target IR 层验证

目标是证明 lowering 后的 NVGPU/PTX 已经出现该优化应有的主链。

重点看：

- `nvgpu.device_async_copy`
- `nvgpu.device_async_wait`
- `nvgpu.ldmatrix`
- `nvgpu.mma.sync`
- vectorized global load/store
- barrier 的数量和位置

### 5.4 机器层验证

目标是证明真正落到了相应硬件指令。

重点看：

- `LDGSTS`
- `LDGDEPBAR` / `DEPBAR`
- `LDSM`
- `HMMA`
- `LDG.128` / `STG.128`
- `BAR.SYNC` 是否异常偏多

### 5.5 性能层验证

目标是证明效果不是偶然。

必须遵守公平性：

- 与 Triton 或历史版本对比时，形状一致；
- `num_warps / requested_stages / num_ctas / arch` 一致；
- 都是同一类 `exact-tile` case；
- 使用同样的 warmup 和统计口径；
- 不允许一边测 direct path，一边拿 relay path 对比。

### 5.6 建议的基础命令模板

构建：

```bash
cmake -S /home/zhangruiqi/mini_triton_nvgpu_v1 \
  -B /home/zhangruiqi/mini_triton_nvgpu_v1/build \
  -G Ninja \
  -DMLIR_DIR=/home/zhangruiqi/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/zhangruiqi/llvm-project/build/lib/cmake/llvm

cmake --build /home/zhangruiqi/mini_triton_nvgpu_v1/build -j8
```

跑到某个 pass 停下，检查 attr/IR：

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

完整 lowering 到 NVVM/PTX-ready ISA 输出：

```bash
/home/zhangruiqi/mini_triton_nvgpu_v1/build/bin/tb-opt \
  --pass-pipeline='builtin.module(tb-stage1-full-to-nvvm-pipeline{cubin-chip=sm_86 cubin-features=+ptx60 cubin-format=isa})' \
  /home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir
```

如果需要看机器层证据，原则上应对最终 cubin 使用 `cuobjdump --dump-sass`，并在相同 launch/config 下对比。

---

## 6. Stage1 主线总表

| Pass | 优化对象 | 效果类型 | 主要产物 | 主要硬件指纹 |
| --- | --- | --- | --- | --- |
| `TBVerifyScope` | 合法范围 | 前置闭合 | 无新 attr | 无，负责提前失败 |
| `TBAttachTargetInfo` | 目标硬件真相 | 前置闭合 | module `tb.target` | 决定后续是否可发 async/ldmatrix/mma |
| `TBSemanticizeMatmul` | 语义真相 | 前置闭合 | `tb.semantic_matmul` | 边界与 exact-tile 逻辑收口 |
| `TBBuildProgramMappingPlan` | program/CTA 映射 | 轻度性能相关 | `tb.program_mapping_plan` | launch/index 计算模式 |
| `TBBuildLayoutPlan` | layout/fragment | 直接性能主因 | `tb.encoding_plan` | `ldmatrix`、`mma.sync`、C direct |
| `TBBuildTransportPlan` | global/shared transport | 直接性能主因 | `tb.transport_plan` | `cp.async`、`LDGSTS` |
| `TBBuildCRegisterPlan` | accumulator/epilogue | 直接性能主因 | `tb.accumulator_plan`、`tb.epilogue_plan` | direct C vector path |
| `TBRewriteMatmulMainloop` | tensor-core 主线重写 | 直接性能主因 | `tb.matmul_rewrite` | `mma.sync` 主链 |
| `TBCleanupLayoutConversions` | 删除伪转换 | 结构清理 | 更干净 IR | 更少多余 move/convert |
| `TBBuildMainloopGraph` | 资源/依赖图 | 中间支撑 | `tb.buffer_model` | 为后续 schedule/wait 提供真相 |
| `TBRegularizeKLoop` | 单一 K 主循环 | 中间支撑 | `tb.loop_plan` | 稳定 steady-state pipeline |
| `TBAssignLatencies` | 延迟模型 | 直接性能主因 | `tb.latency_plan` | overlap 提前量 |
| `TBScheduleLoops` | stage/cluster/order | 直接性能主因 | `tb.pipeline_plan` | producer/consumer/compute 重叠 |
| `TBDeriveWaits` | async frontier | 直接性能主因 | `tb.async_plan` | `device_async_wait` / `DEPBAR` |
| `TBExpandPipeline` | 显式 cluster 主线 | 后收口支撑 | `tb.pipeline_expansion` | wait 和 `k_group` owner 显式化 |
| `TBCleanupPipeline` | strict mainline 收口 | 后收口 | `tb.pipeline_mainline`、`tb.pipeline_ready` | lowering 不再重建主线 |
| `TBLowerPipelineToNVGPU` | target lowering | 最终落地 | NVGPU/GPU IR | `cp.async` / `ldmatrix` / `mma.sync` / direct vector store |

---

## 7. 逐 Pass 严格编写、验证与效果证明

## 7.1 `TBVerifyScope`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/VerifyScope.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/VerifyScope.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelConfig.h)

它在优化什么：

- 不是直接提速；
- 它优化的是“后续所有优化成立的边界条件”；
- 没有这一步，后面很多性能问题只是非法配置造成的假象。

必须怎么写：

- 只验证 Stage1 支持边界；
- 读取 `KernelConfig` 和 module-level 执行请求；
- 对不支持的 case 直接失败；
- 不在这里做 layout、pipeline、transport 的任何决定。

严禁写法：

- 在这里偷偷根据配置改写 kernel；
- 在 verifier 中补默认值；
- 用“放过非法 case，后面 fallback”代替失败。

怎么验证：

- 非法输入必须在这一层失败；
- 合法输入必须原样通过；
- 不能引入任何新的 attr 作为功能替代。

怎么证明有效果：

- 这一步的效果证明不是 runtime；
- 它的效果是让后续 pass 的输入域收敛，减少错误结论；
- 证明口径是“非法 case 被及时挡住，合法 case 不被污染”。

## 7.2 `TBAttachTargetInfo`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TargetInfo.h)

它在优化什么：

- 优化的是“目标硬件真相的显式 owner”；
- 让后续 pass 不再依赖隐藏默认 chip 或硬编码能力。

必须怎么写：

- 必须从 pipeline target 选项显式构造 `TargetInfo`；
- 必须写回 module `tb.target`、`tb.threads-per-warp`；
- 必须验证 `tb.num-warps`、`tb.requested-stages` 已经存在；
- 不允许回退到隐式默认硬件。

严禁写法：

- 在 lowerer 中硬编码 `sm_86`；
- 在后续 pass 中再次解析 target 选项字符串；
- 用 per-op target attr 作为主 owner。

怎么验证：

- module 上存在 `tb.target`、`tb.threads-per-warp`；
- 缺少执行请求时直接失败；
- 下游 pass 只读取这里写回的 target contract。

怎么证明有效果：

- 这一步不直接提升 GFLOPS；
- 它的效果是消除 target dual truth；
- 后续出现 `cp.async`、`ldmatrix`、`mma.sync` 的合法性都必须能追溯到这里。

## 7.3 `TBSemanticizeMatmul`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/SemanticizeMatmul.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/SemanticizeMatmul.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulSemantics.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulSemantics.h)

它在优化什么：

- 优化的是 TTIR 到 TTGIR 风格的语义真相；
- 把 problem/tile/exact-tile/boundary 和 A/B/C descriptor truth 固定下来。

必须怎么写：

- 只收 semantic truth；
- 语义描述中不混入 warpGrid、physical blocked owner；
- `exactTile`、边界、descriptor type 全部显式进入 `tb.semantic_matmul`；
- 后续 layout pass 必须消费这份真相。

严禁写法：

- 在 semantics 里直接决定 shared physical layout；
- 在这里写 lane mapping；
- 在这里把 transport legality 也带进去。

怎么验证：

- `tb.semantic_matmul` 必须存在；
- exact-tile 和 boundary 字段必须自洽；
- 不应再需要 lowerer 根据原始 `tb.matmul` 重新推 semantic truth。

怎么证明有效果：

- 效果体现在“后续不再重复猜边界”；
- exact-tile case 的后续 path 选择必须稳定；
- 不是直接看速度，而是看是否消除了 semantic dual truth。

## 7.4 `TBBuildProgramMappingPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildProgramMappingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildProgramMappingPlan.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/ProgramMappingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/ProgramMappingPlan.h)

它在优化什么：

- 优化的是 program/CTA 对 tile 的覆盖方式和 launch 顺序；
- 这一层不是 CTA 内 pipeline，而是 CTA 级 mapping truth。

必须怎么写：

- `mappingKind`、`launchOrder`、`groupM`、`totalPrograms`、`numCTAs` 必须由 plan 显式拥有；
- module `tb.num-ctas` 必须由这一步写回；
- lowering 只能消费这些公式组件，不再重拼 launch 公式。

严禁写法：

- 在 lowering 里根据 `group_m` 再手工重写 grouped launch 算法；
- 把 CTA 内部 warp mapping 和 program mapping 混在一起；
- 在 `AttachTargetInfo` 中硬写 `num_ctas`。

怎么验证：

- `tb.program_mapping_plan` 存在；
- module `tb.num-ctas` 来自这里；
- `buildProgramTileCoordinates` 一类逻辑只能消费 plan，不应该重新推导 launch truth。

怎么证明有效果：

- 结构层：program id 到 tile 坐标的公式稳定；
- 性能层：它对 locality 和 grouped launch 有影响，但不是当前最大主因；
- 若这一步错误，后续速度波动会表现为地址计算乱、跨 tile 访问顺序差。

## 7.5 `TBBuildLayoutPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildLayoutPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildLayoutPlan.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h)

它在优化什么：

- 这是 Stage1 第一大性能 owner；
- 优化的是 A/B global、A/B shared、dot operand、accumulator、C store 的 layout 和 fragment truth；
- 它直接决定 `ldmatrix`、`mma.sync`、bank conflict、C direct vector 路径能否成立。

必须怎么写：

- `tb.encoding_plan` 必须完整拥有 physical encoding truth；
- `FragmentEncodingSpec` 必须显式拥有 lane 到 fragment 的访问规律；
- shared spec 只表达 layout/alloc truth，不携带 transport legality；
- 后续 accumulator、rewrite、buffer model、lowering 全部消费这一份 encoding truth。

严禁写法：

- 让 `SharedEncodingSpec` 携带 async/vector bytes；
- 在 lowerer 中根据指令形状临时重建 fragment lane mapping；
- 在 epilogue path 再重新决定 C store layout。

怎么验证：

- `tb.encoding_plan` 存在且角色完整；
- A/B shared、A/B dot、acc、C store encoding 都齐全；
- lane/fragment 相关字段非空且自洽；
- 旧 `tb.layout_plan` 不再作为 owner。

怎么证明有效果：

- NVGPU/PTX 层出现 `nvgpu.ldmatrix` / `ldmatrix`；
- NVGPU/PTX 层出现 `nvgpu.mma.sync` / `mma.sync`；
- C path 能形成 direct vector store；
- SASS 层出现 `LDSM`、`HMMA`；
- 如果做错，哪怕后续也生成了 `mma.sync`，性能仍会因布局不对而掉下去。

## 7.6 `TBBuildTransportPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildTransportPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildTransportPlan.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TransportPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/TransportPlan.h)

它在优化什么：

- 这是 A/B producer 主线 owner；
- 优化的是 global 到 shared 的 transport legality、vector bytes、transaction bytes、cache policy、async eligibility。

必须怎么写：

- transport truth 必须从 layout truth 和 target truth 推导；
- `tb.transport_plan` 必须独立存在；
- `asyncEligible` 必须由 transport legality 决定，不由 `exactTile` 布尔值直接替代；
- 下游 async plan 和 lowering 必须只消费 transport contract。

严禁写法：

- 继续让 `EncodingPlan` 兼管 transport；
- 看到 exact-tile 就默认 async；
- 看到某个 shape 就默认 vector bytes 是 16。

怎么验证：

- `tb.transport_plan` 存在；
- operandA、operandB 都有 `kind`、`vectorBytes`、`asyncEligible`；
- `BuildTransportPlan` 不读取 epilogue、pipeline、wait 信息。

怎么证明有效果：

- NVGPU 层出现 `nvgpu.device_async_copy`；
- NVGPU 层 wait 可对应 async group；
- PTX 层出现 `cp.async`；
- SASS 层出现 `LDGSTS`；
- 若这一步错了，会退化成 `LDG + STS + BAR` 同步主链。

## 7.7 `TBBuildCRegisterPlan`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildCRegisterPlan.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AccumulatorPlan.h)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EpiloguePlan.h)

它在优化什么：

- 这是 C path 的 owner；
- 优化的是 accumulator register organization 和 epilogue init/store 路径；
- 它决定 C 是否真 direct-global vector，而不是 shared relay。

必须怎么写：

- `AccumulatorPlan` 只拥有 register 和 pack truth；
- `EpiloguePlan` 明确区分 `DirectGlobalVector` 和 `SharedRelay`；
- direct path 必须是零 relay 语义；
- fused epilogue expression 必须挂在 epilogue plan，不塞进 lowering。

严禁写法：

- direct path 仍分配 shared relay backing；
- init/store 名义 direct、实际 shared staging；
- 用 lowerer 中的局部捷径决定 direct/relay。

怎么验证：

- `tb.accumulator_plan`、`tb.epilogue_plan` 同时存在；
- exact-tile direct case 下，`initMode` 和 `storeMode` 都是 direct；
- direct case 不携带 relay alloc shape 或 shared physical model；
- cleanup pass 能强校验 direct-global epilogue。

怎么证明有效果：

- lowering 后 C 路径无多余 shared relay memref；
- global load/store 以向量形式出现；
- `BAR.SYNC` 明显减少；
- SASS 中 C 路径更接近 `LDG.128/STG.128`；
- 若这一步错了，哪怕 A/B async 和 mma 都对，整体性能仍会被 C path 拖住。

## 7.8 `TBRewriteMatmulMainloop`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RewriteMatmulMainloop.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RewriteMatmulMainloop.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulRewritePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulRewritePlan.h)

它在优化什么：

- 这是把 generic tile matmul 变成 tensor-core 主线的核心边界；
- 优化对象是 instruction geometry、operand fragment path、`kGroups`、direct epilogue routing。

必须怎么写：

- rewrite 必须基于 `EncodingPlan + AccumulatorPlan + EpiloguePlan`；
- `aPath`、`bPath` 必须显式描述 fragment owner；
- `mainloopKind`、instruction 形状、fragments per k-group 必须显式化；
- rewrite 只定主链结构，不做 scheduling。

严禁写法：

- 在 loop regularization 或 lowerer 中再补 fragment path；
- 用原始 tile matmul 外形直接在 lowerer 里凑 `mma.sync`；
- direct epilogue routing 不在这里定，而是后面看情况选择。

怎么验证：

- `tb.matmul_rewrite` 存在；
- `instructionM/N/K`、`kGroups`、`aPath`、`bPath` 完整；
- directAccumulatorInit/store 与 epilogue plan 一致。

怎么证明有效果：

- NVGPU/PTX 层形成清晰 `ldmatrix -> mma.sync` 主线；
- buffer model 中有明确 compute op 链；
- SASS 层出现 `HMMA` 主链，而不是普通 FMA 主链。

## 7.9 `TBCleanupLayoutConversions`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupLayoutConversions.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupLayoutConversions.cpp)

它在优化什么：

- 优化的是“删除已经失去语义价值的 layout 边界”；
- 它是结构清理，不是主策略生成。

必须怎么写：

- 只删除 identity conversion 和紧邻 round-trip；
- 不能把仍承载不同 encoding truth 的转换误删；
- cleanup 必须发生在 rewrite 后、pipeline 前。

严禁写法：

- 依靠这个 pass 偷偷改变 layout truth；
- 为了让 IR 好看而删除真实边界；
- 把 layout propagation 的责任简化成“多删几个 convert”。

怎么验证：

- 重复 `tb.convert_layout` 明显减少；
- 真正不同 encoding 的边界仍保留；
- 后续 buffer model 不依赖已删除的中间转换。

怎么证明有效果：

- IR 更干净；
- lowerer 中多余 move/convert 减少；
- 通常会带来寄存器压力和指令数下降，但它的主证明仍是结构正确，而不是单独追 GFLOPS。

## 7.10 `TBBuildMainloopGraph`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildMainloopGraph.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildMainloopGraph.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/BufferModel.h)

它在优化什么：

- 它优化的不是单条指令，而是“后续 schedule/wait 依赖的资源真相”；
- 负责把 backings、views、values、ops 显式化。

必须怎么写：

- `BufferModel` 只表达语义资源图；
- `PipelineOpSemanticClass` 必须停在语义级，不写 target opcode；
- 所有 operand、accumulator、epilogue 相关资源都必须能在图中找到 owner。

严禁写法：

- 在 buffer model 里塞入 NVGPU 特定 opcode 决策；
- 直接用 lowering 临时 SSA 值替代资源图；
- 让某些关键 value 没有 owner view。

怎么验证：

- `tb.buffer_model` 存在；
- 每个 pipeline op 都有 inputs/outputs；
- shared buffer、register fragment、epilogue fragment 的 owner 都可追溯。

怎么证明有效果：

- 这一步本身不直接出速度；
- 它的效果是让后续 latency/schedule/wait 不再靠猜；
- 如果这一步错，后面 schedule 和 wait 全部会漂。

## 7.11 `TBRegularizeKLoop`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RegularizeKLoop.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/RegularizeKLoop.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LoopPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LoopPlan.h)

它在优化什么：

- 优化的是主 K 循环骨架；
- 让主线变成可 pipeline 的单一 steady-state loop owner。

必须怎么写：

- `tb.loop_plan` 必须显式拥有 iteration 和 carried values；
- 必须形成单一主 K-loop；
- 只做 loop 规整，不做 latency/schedule。

严禁写法：

- 在 schedule pass 中再临时识别主 K-loop；
- 保留多个等价循环骨架，等后面再挑一个；
- 用 raw buffer graph 直接替代 loop owner。

怎么验证：

- `tb.loop_plan` 存在；
- `singleMainLoop = true`；
- `iterationCount` 与 rewrite 的 `kGroups` 可对齐；
- carried values 记录完整。

怎么证明有效果：

- 后续 `LatencyPlan` 和 `PipelinePlan` 的输入稳定；
- steady-state pipeline 可以围绕这条 loop 展开；
- 若这一步错，后面 async/schedule 会变成局部 patch。

## 7.12 `TBAssignLatencies`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AssignLatencies.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AssignLatencies.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LatencyPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/LatencyPlan.h)

它在优化什么：

- 这是 pipeline 时序主链的第一步；
- 优化的是每个 op 的 target latency、自身 latency、buffer distance、pipelineability。

必须怎么写：

- 必须消费 `TargetInfo + BufferModel + LoopPlan`；
- latency 是公开合同，不允许只存在 scheduler 私有内部；
- latency 只负责时间代价建模，不决定最终 stage/cluster。

严禁写法：

- 在 scheduler 中写死 producer 应提前几拍；
- 不通过 `TargetInfo`，直接写固定 latency 常数；
- 把 reuse/buffer distance 留给 wait 推导时再猜。

怎么验证：

- `tb.latency_plan` 存在；
- pipelineable op 被显式标出；
- `bufferDistance` 与 loop-carried 关系合理；
- `reason` 能说明为什么这么赋值。

怎么证明有效果：

- 这一步本身未必改变指令集合；
- 它的效果要在 `ScheduleLoops` 和 `DeriveWaits` 之后体现为更好的 overlap；
- 如果 latency 建模错误，后面会出现 producer 发得过晚或 wait 过早。

## 7.13 `TBScheduleLoops`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ScheduleLoops.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ScheduleLoops.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelinePlan.h)

它在优化什么：

- 这是 Stage1 真实性能主因之一；
- 优化的是 async producer、shared consumer、mma compute 的 stage/cluster/order 排布。

必须怎么写：

- 必须只消费 `BufferModel + LoopPlan + LatencyPlan`；
- `PipelinePlacement` 必须公开写入 `tb.pipeline_plan`；
- `stageOwnedBuffers` 必须显式表达 stage-buffer ownership；
- scheduler 不能重新定义 transport 或 epilogue truth。

严禁写法：

- 根据 NVGPU lowering 习惯倒推 schedule；
- 在 lowerer 中再给 op 排顺序；
- 靠 barrier 修补错误 schedule。

怎么验证：

- `tb.pipeline_plan` 存在；
- 每个 pipeline op 都有明确 stage/cluster/order；
- stage-owned buffer 信息齐全；
- schedule 覆盖全部主线 op。

怎么证明有效果：

- 后续 expansion 后，producer 和 compute 在 steady-state 中交错出现；
- NVGPU/PTX/SASS 上虽然指令种类可能不变，但顺序会更接近 Triton 主线；
- runtime/NCU 上会体现在更好的 overlap、更低 stall。

## 7.14 `TBDeriveWaits`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/DeriveWaits.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/DeriveWaits.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/AsyncPlan.h)

它在优化什么：

- 这是 async frontier owner；
- 优化的是 producer/group/wait/reuse-fence 的最小正确集合。

必须怎么写：

- `tb.async_plan` 必须显式拥有 producers、groups、waits、reuseFences；
- `srcOffsets`、`vecBytes`、`groupId` 等 transport 相关字段必须在这里落地；
- wait 必须锚到 first shared consumer 或显式 frontier；
- reuse fence 必须显式表达 slot overwrite 安全边界。

严禁写法：

- 只有 barrier，没有 async wait；
- exact-tile 主线跳过 wait；
- wait 位置由 lowering 再按 cluster 顺序推断。

怎么验证：

- `tb.async_plan` 存在；
- async producer/group/wait 数量关系合理；
- 每个 producer 都有 group owner；
- wait 有明确 `beforeOpId` 和必要 barrier 信息。

怎么证明有效果：

- NVGPU 层出现 `nvgpu.device_async_wait`；
- PTX 层出现 `cp.async.wait_group`；
- SASS 层出现 `LDGDEPBAR/DEPBAR`；
- barrier 数量下降，stall reason 改善。

## 7.15 `TBExpandPipeline`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ExpandPipeline.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/ExpandPipeline.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelineExpansion.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelineExpansion.h)

它在优化什么：

- 它不是新策略生成，而是把 schedule/wait 真相显式展开；
- 它优化的是 lowering 前的 ownership 清晰度。

必须怎么写：

- `ExpandedCluster` 必须显式拥有 `kind`、`stage`、`cluster`、`kGroup`、`waitGroupIds`；
- `k_group` 必须在这一步成为 owner；
- `needsBarrier` 必须显式化，而不是后面按位置推断。

严禁写法：

- 仅保留抽象 schedule，不展开 cluster stream；
- 让 lowerer 依据 op 序号或 stage 自己重组 cluster；
- 不记录 wait owner，只记录一个模糊的“此处需要同步”。

怎么验证：

- `tb.pipeline_expansion` 存在；
- 全部 op 被覆盖且只覆盖一次；
- 全部 wait group 被覆盖且只覆盖一次；
- `k_group` 连续且完整。

怎么证明有效果：

- `TBCleanupPipeline` 能基于这份显式 expansion 通过严格校验；
- lowerer 不再需要隐藏 skeleton；
- 机器层效果体现在主线更可预测、同步点更准确。

## 7.16 `TBCleanupPipeline`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupPipeline.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupPipeline.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelineReady.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/PipelineReady.h)

它在优化什么：

- 这是 post-pipeline 严格收口边界；
- 它优化的是“让 lowering 之前只剩真 mainline”。

必须怎么写：

- 必须强校验 direct-global epilogue；
- 必须强校验每个 async producer 都是 legal `cp.async`；
- 必须强校验 expansion 对 op 和 wait 的覆盖；
- 必须 materialize `tb.pipeline_mainline` 作为显式 cluster stream owner；
- 必须写出 `tb.pipeline_ready`。

严禁写法：

- cleanup 只是简单复制 attr，不做一致性检查；
- 允许 direct epilogue 混入 relay path；
- 允许 async producer 不是 `cp.async` 主线却继续往下走。

怎么验证：

- 原始 `tb.matmul` 被 `tb.pipeline_mainline` 替换；
- `tb.pipeline_ready` 存在；
- 非 direct-global epilogue 或非法 async producer 直接失败；
- mainline body 中 cluster 顺序与 expansion 一致。

怎么证明有效果：

- lowering 不再需要隐藏逻辑补主线；
- 后续 IR 中主线清晰、错误更早暴露；
- 性能证明要在 lowering 后看，但这一步是稳定性能的必要边界。

## 7.17 `TBLowerPipelineToNVGPU`

对应代码：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelContract.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/KernelContract.h)

它在优化什么：

- 这是最终 target lowering；
- 它的优化点不该是“重新设计策略”，而是“高保真 materialize”。

必须怎么写：

- 只消费 `KernelContract` 中公开合同；
- A/B producer 必须来自 `tb.transport_plan + tb.async_plan`；
- shared->register 必须来自 `tb.encoding_plan` 的 fragment truth；
- C path 必须来自 `tb.epilogue_plan` 的 direct/relay truth；
- pipeline cluster 顺序必须来自 `tb.pipeline_mainline`；
- program tile 坐标必须来自 `tb.program_mapping_plan`。

严禁写法：

- 在 lowerer 中重新推 `k_group`；
- 在 lowerer 中重新设计 async wait frontier；
- 在 direct C path 上临时创建 shared relay；
- 在 lowerer 中把 transport、layout、epilogue 几层职责重新混在一起。

怎么验证：

- NVGPU 层出现 `nvgpu.device_async_copy`、`nvgpu.device_async_wait`、`nvgpu.ldmatrix`、`nvgpu.mma.sync`；
- direct C path 不创建多余 shared relay backing；
- barrier 只出现在 contract 允许的位置；
- launch 索引和 tile 坐标逻辑只消费 program mapping truth。

怎么证明有效果：

- PTX 层出现 `cp.async`、`ldmatrix`、`mma.sync`；
- SASS 层出现 `LDGSTS`、`DEPBAR`、`LDSM`、`HMMA`；
- C path 不再有 shared relay 膨胀；
- 在公平基准下 runtime/NCU 接近 Triton 主线。

---

## 8. 每一步该看什么证据

为了避免以后重复犯“看起来像对了，实际上 owner 还是错的”这个问题，证据必须按下面这张表看。

| 步骤类型 | 最低验收证据 | 不能只看什么 |
| --- | --- | --- |
| 前置闭合 pass | verifier 和 attr 是否正确 | 不能只看能否编译 |
| layout/transport/epilogue pass | plan attr 是否成为唯一 owner | 不能只看最终 SASS 有某条指令 |
| rewrite/loop/pipeline pass | 结构 IR 是否显式承载主线 | 不能只看 runtime 偶然提升 |
| lowering pass | NVGPU/PTX/SASS 是否按合同落地 | 不能只看 IR 里 op 名字像 Triton |
| 最终性能验收 | 同口径 runtime + NCU + 形状一致 | 不能只看单次快了 |

---

## 9. Stage1 最终必须出现的硬件主链

如果 Stage1 真正对齐 Triton 思想，最终必须能在不同层面看到下面四条主链闭合。

### 9.1 A/B producer 主链

应该出现：

- `tb.transport_plan` 中 async eligible 的 global/shared transport；
- `tb.async_plan` 中显式 producer/group/wait；
- NVGPU `nvgpu.device_async_copy`；
- PTX `cp.async`；
- SASS `LDGSTS`。

### 9.2 A/B consumer 主链

应该出现：

- `tb.encoding_plan` 中显式 fragment/lane truth；
- NVGPU `nvgpu.ldmatrix`；
- PTX `ldmatrix`；
- SASS `LDSM`。

### 9.3 Tensor Core compute 主链

应该出现：

- `tb.matmul_rewrite` 的 instruction/fragment truth；
- NVGPU `nvgpu.mma.sync`；
- PTX `mma.sync`；
- SASS `HMMA`。

### 9.4 C epilogue 主链

应该出现：

- `tb.epilogue_plan` 的 direct-global truth；
- 无 relay scratch 的 direct path；
- vectorized global load/store；
- 更少 barrier；
- 不再有 shared relay 膨胀。

---

## 10. 什么叫“证明这一优化有效”

一个优化要被认定为有效，至少要满足下面三条，而不是只满足一条。

### 10.1 它必须把该层 owner truth 固定下来

如果只是生成了几条指令，但 owner 仍是错的，这不算有效。

### 10.2 它必须在下一层被消费，而不是被绕开

如果 plan 写出来了，但 lowering 没按它消费，这不算有效。

### 10.3 它必须在对应层面留下证据

不同 pass 的证据不同：

- 有的是 attr；
- 有的是 IR 结构；
- 有的是 PTX/SASS；
- 有的是 runtime/NCU。

不能统一用“速度有没有快”这一种口径。

---

## 11. 最后的执行原则

后续任何 Stage1 优化，都应该按下面这个节奏做。

1. 先明确这一层的 owner 应该是谁。
2. 先把 owner truth 做成 public contract。
3. 再写只消费前一层 truth 的 pass。
4. 先做合同层和结构层验收。
5. 再做 lowering 层和硬件层验收。
6. 最后才做性能对比。

如果顺序反了，就很容易再次回到：

- owner 混乱；
- lowering 越权；
- direct path 名义 direct、实际 relay；
- async path 名义 async、实际同步；
- 某次测量偶然快了，但主线并不稳定。

---

## 12. 结论

Stage1 的核心，不是“多写几条 `cp.async` / `mma.sync`”，而是把下面这条严格链路真正闭合：

`semantic truth -> program mapping -> layout truth -> transport truth -> accumulator/epilogue truth -> tensor-core mainloop rewrite -> regular loop owner -> latency/schedule/wait -> explicit pipeline mainline -> target lowering`

只要这条链里任何一层重新退化成“lowering 临时猜”，性能就不会稳定。

严格参考 Triton 思想，真正指的就是：

- 顺序对；
- owner 对；
- 责任边界对；
- 验证口径对。

这四件都对了，Stage1 才算真正走在 Triton 主线上。
