# `mini_triton_nvgpu_v1` Stage1 做强且后续可扩展的 Triton 对齐总方案

## 1. 文档目的

这份文档回答的是当前这个更实际的问题：

- 现在的 `mini_triton_nvgpu_v1` 数据结构，是否已经可以正式开始做 `stage1` 矩阵乘法优化；
- 如果目标不是“先凑出一条能跑的主线”，而是：
  - 先把 `stage1 matmul` 性能主链做强；
  - 同时保证以后可以扩到更一般 shape、更大规模、更丰富 target；
- 那么接下来应该怎样参考 Triton，给出一套一次性的完整方案。

本文不是局部修 bug 文档，也不是某次性能测试复盘。

本文只回答三件事：

1. 当前已经做到了什么；
2. 还缺什么，这些缺口分别会影响什么；
3. 要怎样按 Triton 的真实思想，把 `stage1 做强` 和 `后续可扩展` 统一成一条前后依赖清楚的路线。

---

## 2. 当前结论

当前代码已经跨过了“数据结构不够、不能开始优化”的阶段。

已经完成的关键基础有：

1. `layout / mma / dot / accumulator / shared / memdesc` 已经进入 MLIR attr/type 层，参考 [TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td) 和 [TBTypeDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td)。
2. module 上下文 owner 已经建立：`tb.target / tb.num-warps / tb.threads-per-warp / tb.num-ctas`，参考 [AttachTargetInfo.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp#L36)。
3. `tb.convert_layout` 已经存在，说明后续 layout 传播与消解已经有了 Triton 式“中间货币”，参考 [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td#L7)。
4. `EncodingPlan` 已经不再 owning `warp_grid_* / warp_tile_*`，而是退成 summary / registry，参考 [EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h#L48)。
5. `tb.pipeline_mainline` 顶层已经不再重复携带 source config，post-cleanup IR owner 边界已经干净，参考 [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td#L61)。

所以：

**现在已经可以开始做 `stage1` 的真正优化。**

但还不能说“数据结构已经完全封顶”，因为还剩几类会影响性能上限与后续扩展的 owner 漏项。

---

## 3. Triton 的真实思想

当前应该继续对齐的，不是 Triton 的外形，而是 Triton 的三条主原则。

### 3.1 执行上下文属于 IR 上下文 owner

Triton 把：

- `ttg.num-warps`
- `ttg.threads-per-warp`
- `ttg.num-ctas`
- `ttg.target`

直接挂在 IR 上，而不是让某个 analysis struct 长期 owning。

参考：

- [/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py](/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py#L257)

### 3.2 物理 layout 真相属于 encoding attr

Triton 的 `BlockedEncodingAttr / NvidiaMmaEncodingAttr / DotOperandEncodingAttr / CGAEncodingAttr` 负责表达：

- warp 分布；
- CTA/CGA 分布；
- mma 指令族与 instruction shape；
- dot operand fragment 真相。

参考：

- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L85)
- [/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp#L64)
- [/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp#L258)

### 3.3 Triton 先冻结真相，再做 pipeline 和 lowering

Triton 的真实顺序不是先做 `async/mma`，而是：

`layout/CTA freeze -> accelerate matmul -> remove layout conversions -> pipeline -> target lowering`

参考：

- [/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py](/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py#L265)
- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L42)

这意味着：

**如果当前项目里还有“语义层先拍板物理分解”“lowering 再补 transport legality”“program owner 还是写死单 CTA”这些情况，就会同时伤害 `stage1` 性能和后续扩展。**

---

## 4. 当前还没做完的 5 个关键缺陷，以及它们分别影响什么

## 4.1 `MatmulSemantics` 仍在提前决定 `warpGrid`

当前 [MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp#L56) 和 [MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp#L265) 仍在语义化阶段推导 `warpGridM / warpGridN`，再直接做出 `#tb.blocked`。

影响：

- 会压低 `stage1` 性能上限，因为 warp 分解在语义层就冻结了；
- 会破坏后续扩展，因为 layout-freeze pass 失去真正的决定权；
- 会让 `MatmulSemantics` 继续半语义半物理，不利于后续支持 general-shape / 一般 stride。

## 4.2 语义层仍把 row-major contiguous 当永久事实

当前 [MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp#L123) 会拒绝非零 offset、非静态 stride、非 row-major contiguous。

影响：

- 不直接伤害当前 exact-tile 窄主线性能；
- 但会直接卡住后续 transpose / subview / 一般 stride 输入；
- 会让“支持范围不足”表现为语义结构缺陷，而不是“当前 stage1 verifier 的暂时边界”。

## 4.3 shared/async legality 仍停留在 `EncodingPlan::SharedEncodingSpec`

当前 [EncodingPlan.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/EncodingPlan.h#L40) 的 `SharedEncodingSpec` 仍 owning：

- `vectorBytes`
- `asyncVectorBytes`
- `asyncEligible`

并且 lowering 直接消费这些 summary，参考 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp#L568) 和 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp#L691)。

影响：

- 当前 `stage1` 能跑，但 async/shared 主链仍然脆；
- 将来做更复杂 shared swizzle、更多 async transport、TMA、不同 cache policy 时，容易再次出现 dual truth；
- lowering 还没有彻底退成“只消费合同”。

## 4.4 `tb.matmul` 仍带着 target-specific bootstrap 字段

当前 [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td#L32) 里的 `tb.matmul` 还带：

- `num_warps`
- `num_stages`
- `mma`

影响：

- 不阻塞当前 `stage1` 主线；
- 但会卡住以后做“自动选 mma / 自动选 warp 分解 / 自动选 pipeline”的 pass；
- source op 还不够纯，后面做更强 planning 时仍会被 bootstrap 配置回拉。

## 4.5 `ProgramMappingPlan / tb.num-ctas / #tb.cga` 还没有形成真 program owner

当前：

- [AttachTargetInfo.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp#L52) 把 `tb.num-ctas` 写死成 `1`
- [ProgramMappingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/ProgramMappingPlan.cpp#L246) 把 `splitK = 1`
- [ProgramMappingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/ProgramMappingPlan.cpp#L247) 把 `persistent = false`

影响：

- 对当前单 CTA `stage1` 影响有限；
- 但会直接卡住后面的大 shape、高 SM 利用率、grouped launch、split-k、persistent、multi-CTA 扩展；
- `#tb.cga` 虽然存在，但还没有真实 program owner 驱动它。

---

## 5. 总体目标

接下来的完整方案必须同时满足两个目标：

1. **短期目标**：把 `stage1` 的 `single-CTA + mma.sync + cp.async + ldmatrix + direct epilogue` 主线做强。
2. **长期目标**：以后扩展到 general-shape、split-k、persistent、multi-CTA、更多 mma family 时，不需要推翻当前主数据结构。

因此方案不能走两个极端：

- 不能为了赶 `stage1`，继续把物理真相留在 analysis struct；
- 也不能为了“看起来像 Triton”，现在就把所有未来能力都一口气实装。

正确路线是：

**先把 owner 立完整，再只把 `stage1` 主线实现做满，未来能力先保留结构和 verifier。**

---

## 6. 最终结构设计

## 6.1 第一层：纯语义层

`MatmulSemantics` 只允许拥有：

- `problemM/N/K`
- `tileM/N/K`
- `stride/order`
- `exact/boundary`
- A/B/C 逻辑 memdesc

不允许继续拥有：

- `warpGrid`
- 真实 `#tb.blocked`
- 真实 mma family / version / warpsPerCTA

建议新增一个语义 layout attr，例如：

- `#tb.semantic_layout`

它只表达：

- rank
- order
- stride model
- contiguous/boundary 语义

而不是 warp/CTA 物理分发。

## 6.2 第二层：layout-freeze / accelerate 层

这一步才真正决定：

- `#tb.blocked`
- `#tb.nvgpu_mma`
- `#tb.dot_operand`
- `#tb.accumulator`
- `#tb.cga`

它的输入应该是：

- `tb.semantic_matmul`
- `tb.target`
- `tb.num-warps`
- `tb.num-ctas`

它的输出是：

- 一次性冻结好的 encoding owner truth

这一步的职责，等价于 Triton 的：

- `PlanCTA`
- `AccelerateMatmul`
- `OptimizeDotOperands`

但当前项目不必机械复刻 Triton 粒度，只要 owner 责任一致。

## 6.3 第三层：transport contract 层

必须新增一个独立的 `tb.transport_plan` 或 `tb.transport_contract`。

这层 owning：

- global->shared 的 transport kind
- `vectorBytes`
- `asyncVectorBytes`
- `transactionBytes`
- `asyncEligible`
- cache policy / bypass L1

这样 [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 和 [AsyncPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AsyncPlan.cpp) 就不再读取 `EncodingPlan::SharedEncodingSpec` 来推 transport legality。

## 6.4 第四层：program owner 层

`ProgramMappingPlan` 继续存在，但要从“单 CTA 默认值”升级成真正的 program owner。

它应该决定：

- mapping kind
- group launch
- total programs
- split-k
- persistent
- programs per tile

而：

- `tb.num-ctas`
- `#tb.cga`

应该由它和 CTA/layout freeze 一起最终确定，不再由 attach-target pass 直接写死。

---

## 7. 一次性完整改造顺序

必须按这个顺序做，前一步不能依赖后一步。

### 第一步：把 `MatmulSemantics` 去物理化

修改：

- [MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp)
- [MatmulSemantics.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulSemantics.h)
- [TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td)

目标：

- 删掉 `deriveWarpGrid(...)` 对物理 encoding 的 owner 作用
- 让 semanticization 只产出中性语义布局

效果：

- 这一步完成后，后面所有优化 pass 才真正拥有重新规划 warp/CTA 的自由度

### 第二步：新增 `layout-freeze / accelerate matmul` pass

修改：

- [BuildLayoutPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/BuildLayoutPlan.cpp)
- [EncodingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EncodingPlan.cpp)

目标：

- 由这里统一冻结 `warpsPerCTA / mma / dot / acc / cga`
- `EncodingPlan` 退回 registry/summary

效果：

- `stage1` 性能主线会真正从这里开始变强
- 以后扩到不同 warp 分解和 mma family 时，不需要动语义层

### 第三步：新增 `tb.transport_plan`

修改：

- 新增 `TransportPlan.h/.cpp`
- [LowerPipelineToNVGPU.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)
- [AsyncPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AsyncPlan.cpp)

目标：

- transport legality 独立 owner
- `EncodingPlan::SharedEncodingSpec` 只保留共享布局几何，不再保留 async owner

效果：

- async/shared 主线稳定
- future TMA / different transport / different async widths 不会返工

### 第四步：瘦身 `tb.matmul`

修改：

- [TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td)
- [TBOps.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBOps.cpp)
- [KernelConfig.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/KernelConfig.cpp)

目标：

- `num_warps` 完全移到 module/function attr
- `mma` 完全移到 `#tb.nvgpu_mma`
- `num_stages` 变成 pipeline request，而不是 source semantic owner

效果：

- source op 纯净
- acceleration / pipeline pass 才能真正独立决策

### 第五步：把 `ProgramMappingPlan / tb.num-ctas / #tb.cga` 接成真 owner 链

修改：

- [ProgramMappingPlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/ProgramMappingPlan.cpp)
- [AttachTargetInfo.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp)

目标：

- `tb.num-ctas` 不再在 attach-target pass 里固定为 `1`
- `ProgramMappingPlan` 产出真实 `splitK / persistent / totalPrograms`
- `#tb.cga` 由布局/CTA 规划真实派生

效果：

- 当前 `stage1` 仍可由 verifier 限制为单 CTA
- 以后扩展 multi-CTA / grouped / persistent / split-k 时不用返工

### 第六步：最后才放宽 verifier

包括：

- general stride
- transpose/subview
- `num-ctas > 1`
- split-k
- persistent
- 多 mma family

注意：

- 这一轮不要求全部 lowering 完
- 但结构和 owner 必须先立好

---

## 8. Stage1 的严格边界

为了做到“现在就做强，但未来不返工”，必须把 `stage1` 的边界写清楚：

当前只强制支持：

- single CTA
- `mma_sync`
- `m16n8k16`
- `fp16 x fp16 -> fp32`
- `cp.async`
- `ldmatrix`
- direct epilogue
- exact-tile 优先

但这些应该是：

- **verifier 边界**

而不是：

- **数据结构边界**

也就是说：

- 数据结构要允许更一般情况存在；
- 当前不支持的情况，由 verifier 明确拒绝；
- 不能继续把“不支持”编码成“永远不存在这种语义”。

---

## 9. 验收标准

完成这份方案后，要看四层验收。

### 9.1 结构层

必须满足：

- `MatmulSemantics` 不再 owning `warpGrid`
- transport legality 不再停留在 `EncodingPlan::SharedEncodingSpec`
- `tb.matmul` 不再 owning `mma / num_warps`
- `tb.num-ctas` 不再由 attach-target pass 写死

### 9.2 IR owner 层

必须满足：

- module attr 提供 `tb.target / tb.num-warps / tb.threads-per-warp / tb.num-ctas`
- encoding attr 提供 `blocked / nvgpu_mma / dot_operand / accumulator / cga`
- `tb.pipeline_mainline` 顶层只拥有 pipeline region 和附着合同，不重复 source config

### 9.3 Stage1 性能主线层

必须满足：

- warp/CTA/mma/dot 真相在 layout-freeze 后固定
- async/shared legality 由独立 transport contract 提供
- lowering 不再从 summary 层重建 transport/vector legality

### 9.4 后续扩展层

必须满足：

- 即使当前 verifier 仍限制 single CTA / row-major / one mma family
- 结构上也已经能表达 general stride、split-k、persistent、multi-CTA

如果做完后仍然出现：

- 前面 pass 读一部分语义层；
- 中间 pass 读一部分 plan summary；
- lowering 再自己猜一部分物理真相；

那就说明这份方案还没有真正落地。

---

## 10. 最终一句话

`mini_triton_nvgpu_v1` 现在已经具备开始做 `stage1` 真优化的结构基础。

接下来最重要的不是继续补零散字段，而是一次性完成下面这条 owner 收口：

**语义层去物理化，layout/mma/CTA 在 accelerate 阶段冻结，transport legality 独立成合同，program owner 不再写死单 CTA。**

按这条路线做，才能同时得到两件事：

1. `stage1 matmul` 现在就能做强；
2. 后面扩到更一般的 Triton 式优化时，不需要推翻当前结构。
