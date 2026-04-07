# `mini_triton_nvgpu_v1` 参考 Triton 的 owner-truth / attr-type 重构方案

## 1. 文档目的

这份文档回答的是下面这个更具体的问题：

- 已经看过 `mini_triton_nvgpu_v1` 当前代码；
- 已经看过本地 Triton 源码；
- 现在要明确：
  - Triton 到底是怎么放置 `warp / layout / CTA / mma / shared` 这些硬真相的；
  - 当前 `mini_triton_nvgpu_v1` 为什么还没有完全做到这一点；
  - 如果继续往 Triton 思想靠拢，数据结构应该怎样修改；
  - 哪些字段必须迁入 MLIR attr/type；
  - 哪些 analysis struct 必须降级成 derived summary；
  - 哪些 source op 字段应该保留，哪些应该退场。

本文只讨论：

- owner 边界；
- attr/type 设计；
- analysis struct 的重排；
- 后续 pass 能否稳定建立在这些结构之上。

本文不讨论：

- 具体某个 pass 的实现细节；
- 某次 benchmark 为什么快慢；
- 某条 lowering 的局部 bug；
- 某个规模的性能调优参数。

---

## 2. 最终结论

当前 `mini_triton_nvgpu_v1` 已经做对了一半：

1. 已经有 `!tb.memdesc` 和 `#tb.*` encoding attr 这条方向；
2. 已经开始把 layout/memory legality 放进 MLIR 语义层；
3. 已经把 `KernelContract` 做成统一入口。

但还没做对的另一半是：

1. 仍有太多硬真相停留在 `MatmulSemantics / EncodingPlan / ProgramMappingPlan` 里；
2. `warp-grid` 还是 derived truth，不是 IR owner truth；
3. 一部分 attr 里还混着 layout truth 和 transport/debug truth；
4. source op 还带着过多 target-specific 语义；
5. 还没有形成类似 Triton 的 “type encoding 真相 + convert_layout 传播/消解” 结构闭环。

因此最终重构方向必须是：

1. 把 `warp / CTA / layout / mma / dot operand / shared layout` 这些真相继续上提到 attr/type 层；
2. 把 `num_warps / threads_per_warp / num_ctas / target` 放到 module/function 上下文 attr；
3. 把 `MatmulSemantics / EncodingPlan / ProgramMappingPlan` 从 owner 降级为 summary / validator / cache；
4. 给 IR 增加 layout conversion 这个中间货币，不再让 pass 靠外部 struct 手搓布局传递；
5. 让后续优化 pass 只消费 IR 真相，不再各自重新推导。

一句话概括：

**不是继续给 plan struct 加字段，而是把硬真相迁入 IR，把 plan struct 收缩成 derived layer。**

---

## 3. Triton 的真实做法

下面这些结论都来自本地 Triton 源码，而不是二手描述。

### 3.1 执行上下文不是 analysis struct owner，而是 IR 上下文 owner

Triton 把下面这些信息作为上下文属性挂在 IR 上：

- `ttg.num-warps`
- `ttg.threads-per-warp`
- `ttg.num-ctas`
- `ttg.target`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h#L48)
- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp#L4140)
- [/home/zhangruiqi/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td](/home/zhangruiqi/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td#L25)

Triton 的 pass 后面统一用：

- `lookupNumWarps(...)`
- `lookupThreadsPerWarp(...)`
- `lookupNumCTAs(...)`

去读上下文，而不是从某个 `Semantics` struct 一层层往下传。

这说明 Triton 的原则是：

- `num_warps / threads_per_warp / num_ctas` 是执行上下文；
- 不是某个 matmul 局部 analysis plan 的私有字段。

### 3.2 layout 真相在 encoding attr

Triton 的 `BlockedEncodingAttr` 直接拥有：

- `sizePerThread`
- `threadsPerWarp`
- `warpsPerCTA`
- `order`
- `CGALayout`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L703)

Triton 的 shared layout 也在 attr 里，不在 lowering 外层另存一套：

- `SwizzledSharedEncodingAttr`
- `NVMMASharedEncodingAttr`
- `PaddedSharedEncodingAttr`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L10)
- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L426)

### 3.3 mma / dot operand 真相也在 encoding attr

Triton 的 `NvidiaMmaEncodingAttr` 直接拥有：

- `versionMajor`
- `versionMinor`
- `warpsPerCTA`
- `CGALayout`
- `instrShape`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L1232)

Triton 的 `DotOperandEncodingAttr` 直接拥有：

- `opIdx`
- `parent`
- `kWidth`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L1391)

也就是说：

- A/B operand 的 fragment 真相不是 lowering 临时拼出来的；
- 而是直接存在 tensor type encoding 里。

### 3.4 Triton 允许推导，但推导一次后会冻结进 encoding

Triton 的 `AccelerateMatmul` 会根据：

- dot shape
- target
- `numWarps`

去选择合适的 `warpsPerCTA`

参考：

- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L85)
- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L249)

但关键点不是“它也推导”，而是：

- Triton 推导完以后，会把结果固化到 `NvidiaMmaEncodingAttr / BlockedEncodingAttr`；
- 后面 pass 不再各自重新推一遍。

### 3.5 Triton 的 CTA 规划也是改 layout，不是长期依赖 plan struct

`PlanCTA` 做的事情本质上是：

- 改 `BlockedEncodingAttr`
- 改 `CGAEncodingAttr`
- 再让布局传播到相关值

参考：

- [/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp#L64)
- [/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp#L258)

这意味着 Triton 的真实 owner 是：

- CTA 切分真相在 layout attr；
- 不是 `ProgramMappingPlan` 这种 analysis struct。

### 3.6 Triton 后续 layout 优化依赖 `convert_layout`

`RemoveLayoutConversions` 的真实算法是：

1. 找 layout anchor
2. 向前向后传播 encoding
3. 冲突时插 `convert_layout`
4. 最后统一重写 IR

参考：

- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L42)

这说明 Triton 并不是靠外部 plan struct 传布局，而是：

- 布局真相在 type；
- 再用 `convert_layout` 作为中间货币做传播和消解。

---

## 4. 当前 `mini_triton_nvgpu_v1` 的核心结构偏差

### 4.1 `warp-grid` 仍然是 derived owner，不是 IR owner

当前 [MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp#L56) 中的 `deriveWarpGrid(...)` 会从：

- `num_warps`
- block shape
- mma instruction shape

推导 `warpGridM / warpGridN`

问题不是“不能推导”，而是：

- 推导结果没有成为 attr/type 层唯一 owner；
- 后续 `EncodingPlan`、lowering、其它校验逻辑里又各自保存了一份相关真相。

这和 Triton 的差别是：

- Triton 允许推导，但推导完必须写回 encoding attr；
- 当前项目还是“推导完留在 analysis struct”。

### 4.2 当前 attr 里仍有双真相

当前 `TB_BlockedEncodingAttr` 带有：

- `logicalShape`

而 `TB_MemDescType` 又带有：

- `shape`

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td#L10)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td#L11)

这会造成：

- layout attr 和 shaped type 同时拥有 shape；
- 后续 verifier 和 lowering 都可能被迫检查两份是否一致。

这不符合 Triton 的 owner 风格。

### 4.3 `SharedEncodingAttr` 混入了非 layout truth

当前 `TB_SharedEncodingAttr` 同时带：

- `logicalShape`
- `allocShape`
- `role`
- `vecBytes`
- `asyncEligible`
- `asyncVecBytes`

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td#L22)

这里面只有一部分是真 layout truth：

- `order`
- `perPhase`
- `maxPhase`
- `paddingIntervals`
- `paddings`
- `transposed`
- `swizzlingByteWidth`

而下面这些不应该属于 layout attr owner：

- `role`
- `asyncEligible`
- `asyncVecBytes`

因为它们分别属于：

- buffer/backing 角色；
- transport legality；
- async issue 宽度。

### 4.4 source op 仍带着太多 target-specific 真相

当前 `tb.matmul` / `tb.pipeline_mainline` 带：

- `num_warps`
- `num_stages`
- `mma` string

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td#L7)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td#L36)

其中问题最大的是：

- `mma` 还是一个 source-level string

而 Triton 的思路是：

- source op 只表达 dot/matmul 语义；
- tensor core 版本和 instruction 形状由 acceleration/layout pass 决定；
- 最终真相体现在 `NvidiaMmaEncodingAttr`。

### 4.5 `MatmulSemantics / EncodingPlan / ProgramMappingPlan` 还在过度拥有真相

这三个结构当前仍然在做：

- 保存 `warpGrid`
- 保存 layout 几何
- 保存 CTA/program 几何

它们现在更像：

- owner + summary 的混合体

而 Triton 的方向是：

- owner 在 IR；
- analysis struct 负责 parse / derive / validate / cache。

---

## 5. 最终 owner 重排原则

重构后必须固定下面这套 owner 原则。

### 5.1 上下文 owner

下面这些信息属于 module/function context：

- `num_warps`
- `threads_per_warp`
- `num_ctas`
- `target`

不再长期属于：

- `MatmulSemantics`
- `KernelConfig`
- `EncodingPlan`

### 5.2 layout owner

下面这些信息属于 encoding attr：

- blocked 分布
- warp 维度分解
- CTA/CGA 分布
- shared swizzle/padding/order
- mma result 布局
- dot operand 布局

不再长期属于：

- `MatmulSemantics`
- `ProgramMappingPlan`
- `EncodingPlan` 的字段本体

### 5.3 memory owner

下面这些信息属于 `MemDescType`：

- logical `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`

### 5.4 analysis owner

analysis struct 只允许拥有：

- 源问题语义；
- derived summary；
- validation/cache；
- schedule/async/frontier 这类变换结果。

analysis struct 不允许再长期拥有：

- 第二份 layout truth；
- 第二份 warp truth；
- 第二份 CTA truth。

---

## 6. 推荐的 attr/type 最终设计

下面是这次文档最核心的部分。

### 6.1 新增 `TB_CGAEncodingAttr`

新增：

- `TB_CGAEncodingAttr`

职责：

- 表达 CTA group arrangement
- 表达 CTA split / order

建议字段：

- `ctasPerCGA`
- `ctaSplitNum`
- `ctaOrder`

说明：

- 这是当前项目里最缺的一个结构；
- 没有它，`ProgramMappingPlan` 还会继续被迫拥有 CTA 分布真相；
- 有了它以后，`BlockedEncodingAttr` / `SharedEncodingAttr` / `NVGPUMmaEncodingAttr` 都可以统一挂一个 CGA owner。

建议修改文件：

- `include/tb/IR/TBAttrDefs.td`
- `lib/IR/TBAttrs.cpp`

### 6.2 重定义 `TB_BlockedEncodingAttr`

当前：

- `logicalShape`
- `sizePerThread`
- `threadsPerWarp`
- `warpsPerCTA`
- `order`

最终应改成：

- `sizePerThread`
- `threadsPerWarp`
- `warpsPerCTA`
- `order`
- `CGA`

必须删除：

- `logicalShape`

原因：

- shape 应只属于 tensor/memdesc；
- blocked attr 只表达“怎么分发”，不表达“分发谁”。

### 6.3 重定义 `TB_SharedEncodingAttr`

当前字段太杂，必须拆 owner。

最终 `TB_SharedEncodingAttr` 只应该保留 layout/memory 物理排布相关字段：

- `order`
- `perPhase`
- `maxPhase`
- `paddingIntervals`
- `paddings`
- `transposed`
- `swizzlingByteWidth`
- `CGA`

建议删除出 public layout owner 的字段：

- `role`
- `logicalShape`
- `allocShape`
- `vecBytes`
- `asyncEligible`
- `asyncVecBytes`

这些字段的去向应该是：

- `role` -> `BufferBacking.role`
- `logicalShape / allocShape` -> `MemDescType.shape / allocShape`
- `vecBytes / asyncVecBytes` -> `AsyncPlan` 或 transport legality summary
- `asyncEligible` -> target + transport legality 推导结果，不做 layout truth

### 6.4 重定义 `TB_NVGPUMmaEncodingAttr`

当前：

- `mma` string
- `warpsPerCTA`
- `instrShape`

最终建议改成结构化字段：

- `mmaFamily`
- `versionMajor`
- `versionMinor`
- `warpsPerCTA`
- `instrShape`
- `CGA`

说明：

- `mma="m16n8k16"` 这种 string 太弱；
- Triton 真正持有的是 `NvidiaMmaEncodingAttr` 结构化语义，而不是 source op string；
- 你这里至少要做到 `instrShape + version + warpsPerCTA` 结构化。

### 6.5 保留并强化 `TB_DotOperandEncodingAttr`

最终保留：

- `opIdx`
- `parent`
- `kWidth`

要求：

- `parent` 必须是 `TB_NVGPUMmaEncodingAttr`
- `kWidth` 不再让 lowering 现猜

作用：

- A/B operand 的 fragment / ldmatrix / operand legality 真相以后都应从这里读。

### 6.6 `TB_AccumulatorEncodingAttr` 保留，但降为 pure encoding owner

保留：

- `parent`
- `logicalShape`
- `instructionShape`
- `repeatShape`
- `repeatOrder`

要求：

- 它只表达 accumulator distribution，不再带 epilogue/direct-store 策略。

### 6.7 重定义 `TB_MemDescType`

当前：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`
- `alignmentBytes`
- `vectorBytes`

最终建议 public type 只保留：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`

建议从 public type 中移除：

- `alignmentBytes`
- `vectorBytes`

原因：

- Triton 的 `MemDescType` 不把 transport 宽度当作内存对象 owner；
- `alignment/vector width` 更接近 transport/cache legality；
- 如果保留在 type 里，后面很容易又演化成 lowering 侧第二份真相。

如果迁移期暂时不删：

- 也必须明确降级为 derived/non-owner 字段；
- 不能再拿它们当 layout/legality 主真相。

参考 Triton：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td#L23)

---

## 7. source op 和 IR op 的重构原则

### 7.1 `tb.matmul` 应降回 source semantic op

最终 `tb.matmul` 应只表达：

- problem/tile 语义
- group mapping 输入
- exact/boundary 语义
- 输入输出 operand

建议长期移出 `tb.matmul` 的字段：

- `num_warps`
- `mma` string

说明：

- `num_warps` 更像 launch context；
- `mma` 更像 target-specific acceleration result；
- Triton 不是在 source dot 上携带 `mma` 版本字符串。

如果短期迁移期不能一下删掉：

- 允许保留；
- 但必须标记为 bootstrap-only；
- 最终 owner 必须迁到 module attr 与 `TB_NVGPUMmaEncodingAttr`。

### 7.2 `tb.pipeline_mainline` 不能继续充当第二 source op

它应该是：

- post-cleanup mainline owner

但它不应该继续重复携带一套 source config 字段，再和其它 attr 做双真相校验。

长期建议：

- `tb.pipeline_mainline` 只拥有 explicit pipeline region；
- 其它真相从 context attr / type encoding / attached semantic attr 读取。

### 7.3 增加 `tb.convert_layout`

如果要真正走 Triton 风格路线，必须新增：

- `tb.convert_layout`

作用：

- 作为 layout propagation / conflict resolution 的中间货币；
- 让后续 pass 能像 Triton `RemoveLayoutConversions` 一样工作。

没有这一步，会出现的问题是：

- 你虽然有 encoding attr；
- 但布局还是只能靠 `EncodingPlan`、`MatmulRewritePlan`、lowering 私下传递；
- 最终还是回不到 Triton 那种 “IR 自带布局真相” 的模式。

---

## 8. analysis struct 的最终降级方案

### 8.1 `KernelConfig`

最终职责：

- 输入问题规模
- tile 输入
- group_m / exact_tile
- source-level kernel 配置

不应再长期拥有：

- warp-grid
- mma encoding 版本真相
- CTA 分布真相

### 8.2 `MatmulSemantics`

最终职责：

- problem M/N/K
- tile M/N/K
- boundary truth
- source operand stride/order truth
- semantic memdesc 的初始合法性

必须移出的字段：

- `warpGridM`
- `warpGridN`
- `threadsPerWarp`

说明：

- 这些要么属于 context attr；
- 要么属于 encoding attr；
- 不应继续在 `MatmulSemantics` 里当 owner。

另外：

- 目前 row-major-only 限制必须从“语义层永久事实”降级成“stage1 当前支持边界”；
- 语义层应该允许更一般的 stride/order 真相存在；
- 当前阶段不支持的情况由后续 envelope verifier 拒绝，而不是在语义设计层直接掐死。

### 8.3 `EncodingPlan`

最终职责：

- 作为对 IR encoding 的 registry / summary / lookup cache；
- 不是 layout 真相本体。

必须移出的 owner 内容：

- `warpGridM`
- `warpGridN`
- `warpTileM`
- `warpTileN`

这些内容以后可以：

- 由 encoding attr 实时解析；
- 或作为 non-owner summary 缓存在 `EncodingPlan` 里。

但不能再反过来让 attr/type 去服从 `EncodingPlan`。

### 8.4 `ProgramMappingPlan`

最终职责：

- launch order
- grouped/tile mapping
- split-k / persistent / reduction mode
- totalPrograms / programsPerTile 这类 launch 统计

应移出的 owner 内容：

- CTA split
- CGA shape
- warp-grid

说明：

- 这些应进入 `TB_CGAEncodingAttr` 和具体 encoding attr；
- `ProgramMappingPlan` 只保留 program 级 launch 逻辑。

### 8.5 `KernelContract`

最终职责：

- 聚合入口

但要明确：

- 它是 parse/cache/validation 的统一入口；
- 不是新一层 owner；
- 它不应该再成为“把别处不该 owning 的字段重新兜一遍”的地方。

---

## 9. 字段迁移表

下面是推荐的一次性迁移方向。

### 9.1 从 `MatmulSemantics` 迁出

- `warpGridM` -> `#tb.blocked` / `#tb.nvgpu_mma` 的 `warpsPerCTA`
- `warpGridN` -> `#tb.blocked` / `#tb.nvgpu_mma` 的 `warpsPerCTA`
- `threadsPerWarp` -> module attr `tb.threads-per-warp`

### 9.2 从 `ProgramMappingPlan` 迁出

- `clusterShape` 中属于 CTA/CGA 的部分 -> `#tb.cga`

### 9.3 从 `TB_BlockedEncodingAttr` 迁出

- `logicalShape` -> tensor/memdesc `shape`

### 9.4 从 `TB_SharedEncodingAttr` 迁出

- `role` -> `BufferBacking.role`
- `logicalShape` -> `MemDescType.shape`
- `allocShape` -> `MemDescType.allocShape`
- `asyncEligible` -> derived legality
- `asyncVecBytes` -> `AsyncPlan`/transport summary
- `vecBytes` -> transport summary 或删除

### 9.5 从 `tb.matmul` 迁出

- `num_warps` -> module/function attr
- `mma` -> `TB_NVGPUMmaEncodingAttr`

---

## 10. 建议修改的文件范围

如果按本文执行，主要涉及下面这些文件。

IR 层：

- `include/tb/IR/TBAttrDefs.td`
- `include/tb/IR/TBTypeDefs.td`
- `include/tb/IR/TBOps.td`
- `lib/IR/TBAttrs.cpp`
- `lib/IR/TBTypes.cpp`

analysis 层：

- `include/tb/Analysis/KernelConfig.h`
- `include/tb/Analysis/MatmulSemantics.h`
- `include/tb/Analysis/EncodingPlan.h`
- `include/tb/Analysis/ProgramMappingPlan.h`
- `lib/Analysis/KernelConfig.cpp`
- `lib/Analysis/MatmulSemantics.cpp`
- `lib/Analysis/EncodingPlan.cpp`
- `lib/Analysis/ProgramMappingPlan.cpp`

transform / lowering 层：

- `lib/Transforms/LowerPipelineToNVGPU.cpp`
- 后续新增 `convert_layout` 相关 pass

---

## 11. 实施顺序

这套重构必须按依赖顺序做，不能反着来。

### 第一步：补 `TB_CGAEncodingAttr`

先把 CTA/CGA owner 准备好。

### 第二步：重写 `TB_BlockedEncodingAttr / TB_SharedEncodingAttr / TB_NVGPUMmaEncodingAttr`

把 layout/mma 的 owner 收回 attr 层。

### 第三步：瘦身 `TB_MemDescType`

删掉或降级 `alignmentBytes/vectorBytes`，并让 shape/allocShape 成为唯一 memory object 真相。

### 第四步：把上下文 attr 接入 module/function

接入：

- `tb.num-warps`
- `tb.threads-per-warp`
- `tb.num-ctas`
- `tb.target`

### 第五步：瘦身 `MatmulSemantics / EncodingPlan / ProgramMappingPlan`

把 owner 字段迁出，只保留 summary/cache/validation。

### 第六步：加 `tb.convert_layout`

让后续 pass 有真正的 layout propagation 货币。

### 第七步：最后才重写具体优化 pass

因为到这一步之前，pass 还没有稳定的 owner 基座。

---

## 12. 验收标准

完成本文这轮重构后，必须满足下面这些结构验收标准。

### 12.1 不能再出现的现象

- `warpGrid` 只存在于 `MatmulSemantics`
- `logicalShape` 同时存在于 encoding attr 和 memdesc/type
- `mma` string 仍是最终 tensor-core 真相
- `SharedEncodingAttr` 同时拥有 layout truth 和 async transport truth
- lowering 还需要从多个 analysis struct 拼完整布局真相

### 12.2 必须出现的现象

- module/function 上下文 attr 能独立提供 `num_warps / threads_per_warp / num_ctas`
- `BlockedEncodingAttr` 能独立表达 warp/CTA 分发真相
- `NVGPUMmaEncodingAttr` 能独立表达 mma 版本、指令形状、warp 分布真相
- `DotOperandEncodingAttr` 能独立表达 A/B operand 布局真相
- `MemDescType` 成为 memory object 唯一 owner
- `MatmulSemantics / EncodingPlan / ProgramMappingPlan` 退为 derived layer

### 12.3 最终判断标准

如果某个 pass 仍然需要：

- 从 `MatmulSemantics` 取一部分；
- 从 `EncodingPlan` 取一部分；
- 从 `ProgramMappingPlan` 取一部分；
- 再自己拼出 warp/layout/cta 真相；

那就说明这轮重构没有做完。

真正完成的状态应该是：

- pass 主要读 IR attr/type；
- analysis struct 只做解释、缓存和合法性检查。

---

## 13. 最终一句话

参考 Triton 后，`mini_triton_nvgpu_v1` 接下来最该做的不是“继续扩 plan struct”，而是：

**把 `warp / layout / CTA / mma / shared` 真相迁入 attr/type，把 `Semantics / Plan` 退成 derived layer，再补 `convert_layout` 作为布局传播中间货币。**

这才是从“像 Triton”走向“真正按 Triton owner 思想组织”的关键一步。
