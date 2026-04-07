# `mini_triton_nvgpu_v1` 矩阵乘法 Triton 核心思想对齐重构方案

## 1. 文档目的

这份文档回答的是下面这组实际问题：

- 当前 `mini_triton_nvgpu_v1` 的矩阵乘法主线，哪些地方已经接近 Triton；
- 哪些地方还只是 `strict V1 / exact-tile` 切片，不算真正复刻了 Triton 的核心思想；
- 如果目标不是做 Triton 全量，而是做一个“常用 GEMM 范围内真正像 Triton”的实现，后续应该怎么重构；
- 这些修改应该按什么顺序做，才能保证前面的步骤不依赖后面的步骤；
- 每一步改完之后，应该看什么，才算真正收进了 Triton 的思想，而不是又做成一个局部特化实现。

本文是重构执行文档，不是概念综述。

本文只讨论矩阵乘法主线，不讨论：

- split-k
- persistent matmul
- Hopper `TMA/WGMMA`
- batched matmul
- 复杂 epilogue fusion
- autotune 全空间搜索

这些都不是“不重要”，而是不能在矩阵乘法主干思想还没完全对齐之前提前插队。

---

## 2. 当前结论

当前项目已经具备了一条真实的 Triton 风格主性能链：

- 显式 target / semantic / layout / mapping / buffer graph / pipeline contract
- `cp.async + wait + ldmatrix + mma.sync`
- 真 `direct_global_vector` C path

这些并不是假的。

但是当前实现还不算“矩阵乘法意义上的 Triton 核心思想已完全对齐”。

根本原因不是某个微观 bug，而是下面几类核心思想仍未完全收进来：

1. legality 还主要依赖白名单限制，而不是依赖语义和 layout 合法性；
2. general-shape / boundary / mask 还没有成为主线的一部分；
3. program mapping 结构有了，但 derive 逻辑还不是 Triton 式真算法；
4. matmul rewrite / epilogue 还带着当前 exact-tile 主线的写死规则；
5. schedule / pipeline 仍然带有 strict exact-tile 模板化结构；
6. lowering 仍然只接受当前这条窄主线，不能消费更一般的合同。

一句话概括：

**你已经做到了 Triton 的主性能链，但还没有完全做到 Triton 的通用语义驱动思想。**

---

## 3. 本文的最终目标范围

本文不追求 Triton 全量。

本文的目标范围是：

- 硬件：`sm80 / sm86`
- 数据类型：`fp16 x fp16 -> fp32`，随后平移到 `bf16 x bf16 -> fp32`
- 主计算链：`cp.async + ldmatrix + mma.sync + direct global vector epilogue`
- 形状：同时覆盖 `exact-tile` 和 `general-shape`
- 常见配置：
  - `num_warps in {1, 4, 8}`
  - `num_stages in {2, 3, 4}`
- program mapping：
  - `single-CTA tile`
  - `grouped launch`

本文明确暂不覆盖：

- `split-k`
- `persistent`
- `TMA/WGMMA`
- int8 / fp8
- batched matmul

原因很简单：

- 这些东西不是 Triton 的“核心思想入口”；
- 如果当前主线还不能把 general-shape / mapping / schedule / epilogue 做对，
  先去做这些扩展只会继续制造 dual truth。

---

## 4. Triton 的核心思想，在这个项目里到底指什么

这里的“对齐 Triton 核心思想”，不是指：

- pass 名字一样；
- pass 个数一样；
- PTX / SASS 逐字一样；
- 每个细节都和 Triton 源码一模一样。

这里真正要对齐的是下面六条。

### 4.1 legality 来自 IR 语义和 layout 真相

Triton 的思路不是先写一堆 shape 白名单，再让 pass 在白名单里工作。

Triton 的思路是：

- 先让 IR 自己能表达 operand layout、memory legality、tile geometry、program mapping；
- 然后由 pass 检查这些语义是否合法；
- 最后由 lowering 消费这些语义。

也就是说：

- “为什么这个 matmul 合法”这件事，必须主要由 IR 语义回答；
- 不能主要由 [`KernelConfig.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/KernelConfig.cpp) 里的白名单回答。

### 4.2 general-shape 是主线，不是 fallback

Triton 并不是“exact-tile 一条主线，剩下形状统统另外处理”。

Triton 的主思想是：

- exact-tile 是 boundary-free 特例；
- general-shape 是同一条主线加上 boundary / mask / predicate；
- 不能因为有边界，就整条链退回另一个完全不同的实现。

所以：

- boundary/mask 不是附加补丁；
- boundary/mask 必须进入语义层和计划层。

### 4.3 program mapping 必须是算法，不是占位结构

Triton 里 program id 到 tile id 的映射是优化的一部分。

所以：

- `grouped launch`
- tile traversal
- program ordering
- cluster ownership

都不能只是结构预留。

必须有真实 derive 逻辑。

### 4.4 matmul rewrite / epilogue 必须从 layout 真相推导

Triton 的 rewrite 不是靠“当前 shape 正好满足某个规则”。

Triton 的 rewrite 应该由：

- instruction shape
- warp tile
- fragment decomposition
- accumulator layout
- operand path

共同决定。

所以：

- `B pair`
- `fragment pair`
- `direct pack`

都不应该继续由当前 exact-tile 的写死常数决定。

### 4.5 schedule / pipeline 必须由 dataflow 和语义边界推导

Triton 不是先有一个固定模板，再把所有 op 塞进去。

Triton 的思想是：

- schedule 的边界来自 dataflow
- wait frontier 来自 producer-consumer frontier
- cluster kind 来自 op semantic class
- pipeline 结构来自 contract，而不是来自某个 lowering 层的惯例

### 4.6 lowering 只消费合同，不再重新发明合同

这是最关键的一条。

如果某个事实已经在：

- `tb.semantic_matmul`
- `tb.program_mapping_plan`
- `tb.encoding_plan`
- `tb.matmul_rewrite`
- `tb.accumulator_plan`
- `tb.epilogue_plan`
- `tb.async_plan`

里出现了，那么 lowering 不允许再自己重新猜一遍。

---

## 5. 当前代码里，哪些地方还不算 Triton 核心思想

下面这些都是当前代码里已经能明确指认的偏差点。

### 5.1 scope verifier 仍然是白名单主导

当前 [`KernelConfig.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/KernelConfig.cpp) 把矩阵乘法合法性直接收窄成：

- `fp16 x fp16 -> fp32`
- `exact_tile = true`
- `mma = m16n8k16`
- `num_stages = 2`
- `num_warps in {1,4}`
- shape 只允许 `64x64x32 / 128x128x32`

这说明当前 legality 的主要 owner 还是 scope verifier，而不是 IR 语义层。

### 5.2 IR verifier 仍然把 exact-tile 和静态 rank-2 当成默认现实

当前 [`TBOps.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBOps.cpp) 仍然要求：

- `a/b/c` 都是 static rank-2 memref
- exact-tile 时 shape 必须和 block 完全相等

这会把 general-shape 直接挡在语义层外面。

### 5.3 semanticization 仍然带固定模板倾向

当前 [`MatmulSemantics.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp) 里：

- warp grid 由 `sqrt(numWarps)` 导出方阵
- global operand encoding 固定用 blocked
- vector bytes 和 encoding 仍然是窄主线规则

这不是错误，但它仍然更像“当前 stage1 模板”，而不是“可承载常见 matmul 的 semantic truth”。

### 5.4 program mapping 结构有了，但还不是 Triton 式 mapping 算法

当前 [`ProgramMappingPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/ProgramMappingPlan.cpp) 里：

- `groupM = 1`
- `splitK = 1`
- `persistent = false`
- `clusterShape = {1,1,1}`
- `programsPerTile = 1`

这说明 program mapping 现在主要是一个“能挂真相的壳”，还不是 Triton 式真实 derive。

### 5.5 matmul rewrite 仍然带 exact-tile 主线写死规则

当前 [`MatmulRewritePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulRewritePlan.cpp) 里仍然要求：

- `accTilesN` 必须为偶数
- `B` 按 pair 处理
- `directAccumulatorStore` 是 stage1 硬要求

这说明 rewrite 仍然依赖当前主线结构假设。

### 5.6 epilogue direct pack 仍然依赖固定 fragment 配对

当前 [`EpiloguePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp) 里：

- `kFragmentsPerDirectPack = 2`
- 要求水平相邻 fragment 才能组成 direct pack

这仍然不是“由 accumulator layout 自动推出 epilogue pack 规则”。

### 5.7 async path 还没有真正收进 boundary / predicate

当前 [`AsyncPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AsyncPlan.cpp) 里：

- `producer.predicated = false`
- strict async mainline 不允许 sync fallback

这保证了 current exact-tile 主线纯度，但也说明 general-shape 还没有真正进入 async 合同。

### 5.8 loop / schedule / expansion 仍然是 strict exact-tile 模板

当前：

- [`LoopPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LoopPlan.cpp)
- [`PipelinePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelinePlan.cpp)
- [`PipelineExpansion.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelineExpansion.cpp)

仍然把主线固定成：

- 单一 `k_group` 主循环
- `LoadA/B -> LocalLoadA/B -> Mma`
- `async_issue / consumer_wait / mma_compute`

这对当前 stage1 是干净的，但还不是 Triton 更一般的 schedule/pipeline 思想。

### 5.9 lowering 还只接受单一主线合同

当前 [`LowerPipelineToNVGPU.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp) 只接受：

- `strict_async_tensor_core`
- single-CTA row-major tile mapping
- static rank-2 direct-global-vector C memref

这说明 lowering 现在是“严格消费当前窄主线”，还不能消费更一般的 Triton 风格合同。

---

## 6. 重构总原则

后续所有代码修改，都必须遵守下面六条原则。

### 6.1 不能回滚当前 exact-tile 主性能链

当前 exact-tile 主线已经具备：

- async producer
- async wait
- ldmatrix
- mma.sync
- direct global vector epilogue

这些东西不能在 general-shape 改造中被打碎。

### 6.2 不允许新引入 dual truth

任何一个事实只能有一个 owner。

例如：

- boundary 行为如果属于 async producer，就写进 `tb.async_plan`
- C init/store 的 boundary 行为如果属于 epilogue，就写进 `tb.epilogue_plan`
- program id 到 tile id 的规则如果属于 mapping，就写进 `tb.program_mapping_plan`

不能同时在 verifier、plan、lowering 各写一份。

### 6.3 exact-tile 只是 general-shape 的边界特例

设计上必须做到：

- exact-tile = boundary 全为 false 的 general-shape 子集

而不是：

- exact-tile 一条实现
- general-shape 另一条实现

### 6.4 write-up / rewrite / schedule 都必须从已有真相推导

任何 pass 都不能继续依赖：

- 某个神秘常数
- 某个固定 shape
- 某个“如果是当前例子就这么做”的分支

必须只从：

- config
- semantic
- layout
- mapping
- buffer graph
- async frontier

里显式推导。

### 6.5 lowering 只能做 target-specific materialization

lowering 允许做：

- op materialization
- vector/scalar 分流
- predication 细化
- address calculation

lowering 不允许做：

- 重新选择 mapping
- 重新决定 epilogue strategy
- 重新决定 async producer kind
- 重新决定 fragment pack 规则

### 6.6 每一步必须独立见效，前一步不依赖后一步

后续实施顺序必须满足：

- 前一阶段做完后，语义或合同层就已经更接近 Triton；
- 后一阶段只是在消费前一阶段产出的真相；
- 不能靠“先在 lowering 里偷做一版，后面再补合同”。

---

## 7. 终态结构方案

为了让矩阵乘法真正对齐 Triton 核心思想，终态结构必须是下面这样。

### 7.1 `KernelConfig` 只负责 supported envelope，不再负责 shape 白名单

`KernelConfig` 保留，但职责要变成：

- 数据类型
- 指令族
- tile shape
- stage / warp 数
- 是否启用 grouped launch

`KernelConfig` 不再负责：

- exact-tile 强绑定
- shape 白名单
- 某个具体例子是否合法

也就是说：

- `KernelConfig` 是“候选配置”
- legality 最终要由语义层和后续 verifier 共同判断

### 7.2 `tb.semantic_matmul` 变成真正的 problem + operand semantic truth

`tb.semantic_matmul` 必须明确携带：

- problem shape：`M/N/K`
- tile shape：`blockM/blockN/blockK`
- warp grid：显式 `warpGridM/warpGridN`，不再只由 `sqrt(numWarps)` 暗推
- operand order / stride truth
- exact-tile 与 boundary 关系
- A/B/C 的 memdesc type 与 encoding truth

这里的重点不是再造一个大而全 struct，而是把“矩阵乘法为什么合法”这件事变成 IR 可见真相。

### 7.3 `tb.program_mapping_plan` 变成真 mapping 算法 owner

`tb.program_mapping_plan` 必须真正拥有：

- mapping kind：`tile` / `grouped_tile`
- launch order
- grouped 参数
- program id 到 tile id 的公式
- tile coverage 和 boundary ownership

这里不要求一步做到 split-k / persistent。

但必须做到：

- single-CTA tile
- grouped launch

而且 lowering 不再自己推 tile id。

### 7.4 `tb.encoding_plan` 继续做 layout owner，但去掉当前模板化假设

`tb.encoding_plan` 仍然是：

- blocked / shared / dot / accumulator encoding 的 owner

但必须从“当前 exact-tile 模板”升级成“当前目标范围内的合法 encoding derive”。

要点是：

- warp grid 不再固定方阵暗推
- fragment decomposition 不再依赖当前两个 shape
- shared / dot / accumulator 编码仍由 target + config + semantic truth 推导

### 7.5 `tb.matmul_rewrite` 必须改成“由 encoding 推导 rewrite”

`tb.matmul_rewrite` 仍然保留。

但其内容必须只由：

- instruction shape
- warp tile
- fragment decomposition
- operand path
- accumulator plan
- epilogue plan

推导。

这里必须拔掉：

- `B pair` 写死
- `accTilesN` 偶数特判主导 rewrite

这些规则以后可以保留为某些形态的自然结果，但不能再当主设计前提。

### 7.6 `tb.accumulator_plan + tb.epilogue_plan` 必须拥有完整 C path 真相

这层必须清楚回答：

- accumulator register layout 是什么
- direct-global-vector 是否合法
- 如果是 direct path，pack/store 规则是什么
- boundary 下 init/store 应该是 vector、masked vector 还是 scalar tail

重点是：

- C epilogue 的 pack 规则必须从 accumulator layout 派生
- 不能继续固定 `2 fragment -> 1 direct pack`

### 7.7 `tb.async_plan` 必须收进 boundary/predicate 真相

`tb.async_plan` 必须明确：

- producer kind
- source/destination view
- vec bytes / transaction bytes
- 是否 predicated
- 是否 zero-fill
- wait frontier

这样 exact-tile 和 general-shape 才能共用同一个 async owner。

exact-tile 的情况只是：

- `predicated = false`
- 没有 boundary

而不是一条单独实现。

### 7.8 `tb.loop_plan / tb.pipeline_plan / tb.pipeline_expansion` 必须从 exact-tile 模板升级到 semantic op class

这三层仍然保留。

但它们的 owner 要从：

- “当前 exact-tile 三段式模板”

升级为：

- “根据 op semantic class、frontier、dataflow 形成的 pipeline contract”

也就是说：

- cluster kind 不能只由硬编码枚举决定
- wait frontier 必须严格来自 async producer-consumer frontier
- body shape regularization 不能只靠 exact-tile 规则

### 7.9 `LowerPipelineToNVGPU` 只做 materialization，不再做主策略选择

这层终态必须做到：

- exact-tile 继续 materialize 为当前高性能主线
- boundary 形状 materialize 为同一主线下的 predicated / scalar-tail 子路径
- grouped launch 按 mapping 合同 materialize program/tile 映射

它不再负责：

- 决定是否 direct epilogue
- 决定是否 async
- 决定 pack 方式
- 决定 tile mapping

---

## 8. 分阶段实施方案

下面这套顺序满足：

- 前一步不依赖后一步
- 每一步做完都有独立价值
- 每一步都更接近 Triton 的核心思想

## 阶段 1：先把 legality 从白名单迁到语义层

修改重点：

- [`KernelConfig.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/KernelConfig.cpp)
- [`TBOps.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBOps.cpp)
- [`MatmulSemantics.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp)

要做的事：

1. 删掉 shape 白名单和 exact-tile-only 入口。
2. 把 `M/N/K`、tile shape、warp grid、boundary 信息收进语义层。
3. 让 verifier 校验的是“语义是否一致”，而不是“是不是这两个样例 shape”。

这一阶段完成后，哪怕还没支持 general-shape lowering，至少：

- IR 已经能合法表达它；
- legality owner 已经开始从白名单迁移到 semantic truth。

## 阶段 2：把 program mapping 从占位结构变成真 owner

修改重点：

- [`ProgramMappingPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/ProgramMappingPlan.cpp)
- [`LowerPipelineToNVGPU.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)

要做的事：

1. 真正实现 `tile` 和 `grouped_tile`。
2. 让 plan 明确拥有 program id 到 tile id 的公式。
3. 让 lowering 只消费 mapping plan。

这一阶段完成后：

- program mapping 就不再只是结构预留；
- Triton 的 launch/mapping 思想才算真正落地。

## 阶段 3：把 rewrite / epilogue 改成 layout-driven，而不是 stage1-driven

修改重点：

- [`MatmulRewritePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulRewritePlan.cpp)
- [`AccumulatorPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AccumulatorPlan.cpp)
- [`EpiloguePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp)

要做的事：

1. 去掉 `B pair` 和 `kFragmentsPerDirectPack = 2` 这种当前主线常数。
2. 从 accumulator fragment layout 自动推导 direct pack 规则。
3. 明确 boundary 下 C init/store 的合法策略。

这一阶段完成后：

- C path 将真正由 accumulator/layout 驱动；
- direct epilogue 不再只是当前例子的幸运结果。

## 阶段 4：把 boundary/mask 收进 async 和 epilogue 主线

修改重点：

- [`AsyncPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/AsyncPlan.cpp)
- [`LowerPipelineToNVGPU.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)
- [`EpiloguePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp)

要做的事：

1. async producer 支持 boundary 语义。
2. exact-tile 保持 `predicated = false`。
3. general-shape 下引入显式 predication / zero-fill / vector-tail / scalar-tail 规则。
4. C direct path 在边界下仍然保持 direct owner，不再偷偷退成 relay fallback。

这一阶段完成后：

- exact-tile 和 general-shape 才真正进入同一条主线；
- Triton 的“general-shape 是主线，不是另外一套实现”才算落地。

## 阶段 5：把 loop / schedule / pipeline 从 exact-tile 模板升级为语义驱动

修改重点：

- [`LoopPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LoopPlan.cpp)
- [`LatencyPlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/LatencyPlan.cpp)
- [`PipelinePlan.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelinePlan.cpp)
- [`PipelineExpansion.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/PipelineExpansion.cpp)
- [`CleanupPipeline.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/CleanupPipeline.cpp)

要做的事：

1. 把 cluster kind 从 exact-tile 模板化枚举升级为 semantic op class。
2. 让 wait frontier 继续严格锚到 producer-consumer frontier。
3. 让 regularization 仍然要求 owner 清楚，但不再只接受当前 exact-tile 体形。

这一阶段完成后：

- pipeline 才真正从“当前模板”升级到“当前范围内的 Triton 思想”。

## 阶段 6：最后收口 lowering 和 sink pipeline

修改重点：

- [`LowerPipelineToNVGPU.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/LowerPipelineToNVGPU.cpp)
- [`Stage1Pipelines.cpp`](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/Stage1Pipelines.cpp)

要做的事：

1. 让 lowering 只 materialize 前面已经定好的合同。
2. exact-tile 保持当前高性能路径。
3. general-shape 增加边界 materialization 子路径。
4. sink pipeline 继续保护真正的 vector global C path，不要回退到 scalarization。

这一阶段完成后：

- target-specific lowering 才真正退回到它该在的位置。

---

## 9. 每一步的严格验收标准

后续不能只看“能跑”。

必须按下面四层验收。

### 9.1 语义层

- 不再需要 shape 白名单才能表达合法 matmul
- exact-tile 和 general-shape 都能进入同一套 semantic truth
- program mapping 不再是占位字段

### 9.2 plan 层

- exact-tile 和 general-shape 使用同一类 owner
- async / epilogue / mapping / rewrite 不再有双真相
- 任何一个 fallback 都只能是 plan 里的显式子路径

### 9.3 lowering 层

- exact-tile 继续 materialize 为 `cp.async + wait + ldmatrix + mma.sync + direct global vector`
- general-shape 是这条主线的带边界版本，不是另一条完全不同的实现
- grouped launch 的 tile mapping 直接来自 `tb.program_mapping_plan`

### 9.4 性能层

- exact-tile 不得因 generalization 明显退化
- representative general-shape 不得整体塌到同步 fallback 路线
- 常见 GEMM 范围内应稳定逼近 Triton，而不是只在两个样例 shape 上好看

---

## 10. 这份方案做完后，哪些东西仍然可以不做

即使把本文方案完整做完，下面这些仍然可以暂缓：

- split-k
- persistent
- Hopper `TMA/WGMMA`
- batched matmul
- 大规模 fusion
- autotune

因为做完本文后，你已经会得到：

- 一个在常见 GEMM 范围内真正遵守 Triton 核心思想的 matmul 主线；
- 一个可以继续扩展而不需要回炉重构 ownership 的结构；
- 一个可解释、可验证、可持续逼近 Triton 性能的实现。

这已经是“矩阵乘法项目真正站住”的门槛了。

---

## 11. 最终结论

当前项目的问题，不是“没有 async / 没有 tensor core / 没有 direct epilogue”。

当前项目真正还没完全做到的，是：

- legality 仍然白名单化
- general-shape 仍然不在主线
- mapping 仍然不是算法 owner
- rewrite / epilogue 仍然依赖 stage1 专用规则
- schedule / pipeline 仍然带 exact-tile 模板

所以后续重构的总方向不是继续围绕一个 micro shape 调参数，而是按本文顺序，把矩阵乘法主线从：

**“性能链已经通，但仍然是 strict V1 切片”**

推进到：

**“在常见 GEMM 范围内真正按照 Triton 核心思想组织起来的实现”**

这就是当前最重要的总方案。
