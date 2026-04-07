# `mini_triton_nvgpu_v1` 对照 Triton 的 owner truth 迁移最终方案

## 1. 文档目的

这份文档回答的是下面这个更具体的问题：

- 参考 Triton 当前真实代码，
- `mini_triton_nvgpu_v1` 的数据结构到底应该怎么改，
- 才能把 `layout / warp / CTA / MMA / memory legality` 这些硬真相放到正确层级，
- 让后续矩阵乘法优化真正建立在稳定 contract 上，
- 而不是继续依赖 `MatmulSemantics / EncodingPlan / ProgramMappingPlan` 这几层 C++ struct 反复推导和转述。

本文不是再讨论“要不要重构”，而是给出：

1. Triton 真正把 owner 放在哪；
2. 当前项目的 owner 为什么还没完全对齐；
3. 你的 attr/type/plan 应该如何一次迁移到正确层级；
4. 迁完后哪些结构还能保留，哪些必须降级。

本文只讨论**数据结构与 ownership**。

本文不讨论：

- 某条具体 async 指令序列怎么发；
- 某个性能问题的局部参数调优；
- 某次 benchmark 为什么快或慢；
- 某个 pass 里临时 bug 的点状修补。

---

## 2. 最终结论

对照 Triton 真实代码后，结论非常明确：

`mini_triton_nvgpu_v1` 下一步不应该继续给 analysis struct 加字段，而应该做一次 **owner truth 迁移**。

这次迁移的核心不是“把代码改得更像 Triton 外形”，而是把 ownership 改成和 Triton 同一种层级关系：

1. `module / func attr` 负责执行上下文真相。
2. `type encoding attr` 负责 layout / warp / CTA / MMA / dot operand 真相。
3. `memdesc type` 负责 memory legality 与 allocation object 真相。
4. `analysis plan` 只负责解析、汇总、验证和调度派生，不再拥有这些硬真相。

一句话概括：

**硬真相前移到 IR；plan 层退回 derived summary。**

这就是和 Triton 真正对齐的方向。

---

## 3. Triton 的真实做法

这里不讲抽象印象，只看真实代码。

### 3.1 上下文真相放在 module / func attr

Triton 把下面这些上下文真相放在 IR attr 上：

- `ttg.num-warps`
- `ttg.threads-per-warp`
- `ttg.num-ctas`
- `ttg.target`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h)
- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp#L4140)
- [/home/zhangruiqi/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td](/home/zhangruiqi/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td#L25)

后续 pass 统一通过：

- `lookupNumWarps(...)`
- `lookupThreadsPerWarp(...)`
- `lookupNumCTAs(...)`

去读它们，而不是每个 pass 再各自保存一份。

### 3.2 layout / warp / CTA 真相放在 encoding attr

Triton 的 `BlockedEncodingAttr` 直接携带：

- `sizePerThread`
- `threadsPerWarp`
- `warpsPerCTA`
- `order`
- `CGALayout`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L703)

也就是说：

- warp-grid 不只是一个 analysis 数字，
- 而是 layout encoding 的组成部分。

### 3.3 MMA 真相放在 MMA encoding attr

Triton 的 `NvidiaMmaEncodingAttr` 直接携带：

- `versionMajor`
- `versionMinor`
- `warpsPerCTA`
- `instrShape`
- `CGALayout`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L1232)

所以 Triton 不会把：

- “这次 matmul 是什么 MMA 形状”
- “warp 如何覆盖 tile”

继续放在 pass 外部 struct 里反复转述。

### 3.4 Dot operand 真相放在 dot operand encoding attr

Triton 的 `DotOperandEncodingAttr` 直接挂在 dot operand 类型上，包含：

- `opIdx`
- `parent`
- `kWidth`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L1391)

这说明 Triton 的 A/B operand fragment 真相不是靠 lowering 现猜，而是类型就已经说明“我是 dot operand A 还是 B、我的 K packing 是多少”。

### 3.5 memory legality 放在 memdesc type

Triton 的 `MemDescType` 负责：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`

参考：

- [/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td](/home/zhangruiqi/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td#L23)
- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/IR/Types.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/IR/Types.cpp#L90)

也就是说：

- allocation shape
- memory space
- object mutability

这些都属于 memory object 自己，不属于 pipeline plan。

### 3.6 Triton 允许推导，但只推导一次并冻结到 encoding

Triton 不是完全不推导。

例如 `AccelerateMatmul` 会根据：

- target
- dot shape
- `numWarps`

计算合适的 `warpsPerTile` / MMA 组织。

但关键点是：

- 它推导完以后会把结果冻结到 `NvidiaMmaEncodingAttr` 和相关类型上；
- 后面 pass 只消费 encoding，不再各自重推。

参考：

- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L85)

### 3.7 Triton 的 layout 优化是“传播和消除 conversion”，不是“反复发明 layout”

`RemoveLayoutConversions` 的算法是：

1. 找 layout anchor；
2. 传播 layout；
3. 处理冲突；
4. 插入或消除 `convert_layout`；
5. 重写 IR。

参考：

- [/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp](/home/zhangruiqi/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L42)

这说明 Triton 的前提是：

- layout 是 type truth；
- conversion 是显式 IR；
- pass 做 propagation，而不是每层都重新构造 layout 计划。

---

## 4. 当前项目的一级 owner 问题

当前 `mini_triton_nvgpu_v1` 已经比之前好很多，但和 Triton 真正对齐时，还存在四类一级问题。

### 4.1 `warp-grid` 仍然是分析层推导真相

当前：

- `MatmulSemantics.cpp` 里的 `deriveWarpGrid(...)` 会从 `num_warps + block shape + mma instr shape` 推导 `warpGridM / warpGridN`。

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp#L56)

问题在于：

- 这份 warp-grid 不是 encoding attr 自己拥有的；
- 它只是 analysis 阶段的一份推导结果；
- 后续 `EncodingPlan`、`LowerPipelineToNVGPU` 再读它，本质上还是“推导真相被转述”。

这和 Triton 的差别不是“能不能推导”，而是：

- Triton 推导完会冻结到 layout / MMA encoding；
- 你这里推导完还停留在 plan 层。

### 4.2 `layout truth` 虽然有字段，但还没有完全成为 type owner

当前：

- `MatmulSemantics` 持有 `aStrides / bStrides / cStrides`
- `aOrder / bOrder / cOrder`

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulSemantics.h](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/Analysis/MatmulSemantics.h#L17)

这说明你已经开始把 layout 真相显式化了，这是对的。

但问题是：

- 这些信息还是先挂在 analysis struct 上；
- 不是一开始就通过 type encoding 成为主要 owner；
- 同时语义化还把 layout 限死在 row-major contiguous。

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/MatmulSemantics.cpp#L142)

这导致现在更像：

- “带 layout 字段的 row-major 主线”

而不是：

- “真正 layout-driven 的 matmul 底座”

### 4.3 当前 attr/type 里还有双真相

当前 `TB_BlockedEncodingAttr` 带：

- `logicalShape`

而 `MemDescType` 也带：

- `shape`

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td#L10)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td#L11)

这和 Triton 的方向相反。

在 Triton 里：

- shape 是 tensor/memdesc type 的事实；
- layout attr 描述“如何分布和排列”；
- 二者不应该重复持有同一份逻辑 shape。

### 4.4 当前 shared encoding attr 混入了“角色”和“transport”真相

当前 `TB_SharedEncodingAttr` 同时带：

- `role`
- `logicalShape`
- `allocShape`
- `asyncEligible`
- `asyncVecBytes`

参考：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td#L22)

这里混了三类不同 owner：

1. 物理 shared layout 真相；
2. memory object 形状真相；
3. async transport 合法性真相；
4. operand 角色真相。

这会让 shared attr 变成一个“什么都装一点”的混合层。

Triton 不这么做：

- shared layout attr 只表达 shared layout；
- memory object 形状由 memdesc type 负责；
- operand 角色由 use-site / dot operand / op 语义负责；
- async 是否可用由 target + layout + op legality 决定，不直接塞进 shared layout attr。

---

## 5. 迁移后的最终 owner 分层

迁移后，整个系统必须固定为四层。

### 5.1 第一层：module / func context attr

职责：

- 表达执行上下文真相；
- 供所有 pass 统一 lookup。

必须拥有的字段：

- `tb.num-warps`
- `tb.threads-per-warp`
- `tb.num-ctas`
- `tb.target`

可扩展字段：

- `tb.cluster-shape`
- `tb.maxnreg`
- `tb.enable-warp-specialize`

这一层不表达：

- operand layout
- MMA fragment 形状
- shared swizzle

### 5.2 第二层：type encoding attr

职责：

- 表达 distributed layout / CTA / warp / MMA / dot operand / shared layout 真相。

这一层必须成为下面真相的唯一 owner：

- `threadsPerWarp`
- `warpsPerCTA`
- `order`
- `CGA`
- `instrShape`
- `dot operand opIdx`
- `dot operand kWidth`
- shared swizzle / padding / transpose

### 5.3 第三层：memdesc type

职责：

- 表达 memory object legality 和 allocation object 真相。

这一层必须成为下面真相的唯一 owner：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`

### 5.4 第四层：analysis / plan summary

职责：

- 解析 attr/type；
- 汇总为 lowering 和调度易读的 summary；
- 做 cross-check；
- 做 schedule / async / epilogue 派生。

这一层不再拥有：

- warp-grid 主真相；
- CTA/CGA 主真相；
- layout 主真相；
- MMA 形状主真相；
- dot operand 主真相。

---

## 6. attr/type 的最终修改方案

下面是这次迁移真正需要的结构性修改。

## 6.1 新增 `TB_CGAEncodingAttr`

当前项目缺一个对标 Triton `CGAEncodingAttr` 的 owner。

建议新增：

```text
#tb.cga<{
  ctasPerCGA = [...],
  ctaSplitNum = [...],
  ctaOrder = [...]
}>
```

职责：

- 表达 CTA 级分布；
- 被 `BlockedEncodingAttr`、`SharedEncodingAttr`、`NVGPUMmaEncodingAttr` 统一引用。

这样以后：

- CTA split / grouped CTA / future cluster

都不会继续挂在 `ProgramMappingPlan` 当主 owner。

## 6.2 重构 `TB_BlockedEncodingAttr`

当前定义：

- `logicalShape`
- `sizePerThread`
- `threadsPerWarp`
- `warpsPerCTA`
- `order`

建议改成：

```text
#tb.blocked<{
  sizePerThread = [...],
  threadsPerWarp = [...],
  warpsPerCTA = [...],
  order = [...],
  cga = #tb.cga<...>
}>
```

必须删除：

- `logicalShape`

原因：

- shape 属于 tensor/memdesc type；
- blocked encoding 只描述“怎么分布”，不描述“逻辑张量多大”。

这一步做完以后：

- `warpGridM / warpGridN`

就不应该再作为独立 owner 存在，而应该从 `warpsPerCTA` 派生。

## 6.3 重构 `TB_SharedEncodingAttr`

当前定义里混入了太多异质 owner。

建议改成只保留 shared physical layout 真相：

```text
#tb.shared<{
  order = [...],
  vecBytes = ...,
  perPhase = ...,
  maxPhase = ...,
  paddingIntervals = [...],
  paddings = [...],
  transposed = ...,
  swizzlingByteWidth = ...,
  cga = #tb.cga<...>
}>
```

必须删除：

- `role`
- `logicalShape`
- `allocShape`
- `asyncEligible`
- `asyncVecBytes`

删除原因分别是：

- `role` 是语义角色，不是 layout 本身；
- `logicalShape` / `allocShape` 属于 memdesc；
- `asyncEligible` / `asyncVecBytes` 属于 transport legality，不是 shared layout 本身。

做完后：

- async copy 是否合法，应由 `target + shared layout + op kind + element bytes` 推导；
- 不再从 shared attr 直接读一个“假答案”。

## 6.4 重构 `TB_NVGPUMmaEncodingAttr`

当前定义：

- `mma`
- `warpsPerCTA`
- `instrShape`

建议升级为更接近 Triton 的结构化字段：

```text
#tb.nvgpu_mma<{
  versionMajor = ...,
  versionMinor = ...,
  warpsPerCTA = [...],
  instrShape = [...],
  cga = #tb.cga<...>
}>
```

建议删除或降级：

- 纯字符串 `mma`

如果为了项目可读性保留，也只能作为：

- parser / pretty print 的别名

而不能继续作为唯一真相。

真正的 lowering / rewrite / validation 应统一读：

- `instrShape`
- `warpsPerCTA`
- `versionMajor/versionMinor`

## 6.5 新增 `TB_DotOperandEncodingAttr`

当前项目缺少这一层，是和 Triton 对齐时最应该补的一块。

建议新增：

```text
#tb.dot_operand<{
  opIdx = 0 | 1,
  parent = #tb.nvgpu_mma<...>,
  kWidth = ...
}>
```

职责：

- 表达 A/B operand fragment 的身份；
- 表达 dot operand 的 K packing；
- 成为 local load / fragment rewrite / MMA lowering 的统一事实来源。

做完以后：

- A/B operand fragment 的真相不再停留在 `EncodingPlan.fragmentA/fragmentB` 那组 struct 上；
- `EncodingPlan` 只负责解析和摘要，不负责主 ownership。

## 6.6 `TB_AccumulatorEncodingAttr` 继续保留，但只负责 accumulator layout

当前 `TB_AccumulatorEncodingAttr` 的方向基本是对的：

- parent
- logicalShape
- instructionShape
- repeatShape
- repeatOrder

建议保留。

但要求明确：

- accumulator encoding 只表达 accumulator result layout；
- 不负责 epilogue pack strategy；
- 不负责 C store vector 宽度；
- 不负责 direct/relay 选择。

这些都应该由更高层计划派生。

## 6.7 重构 `MemDescType`

当前 `MemDescType` 带：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`
- `alignmentBytes`
- `vectorBytes`

建议最终保留主结构：

- `shape`
- `elementType`
- `encoding`
- `memorySpace`
- `mutableMemory`
- `allocShape`

建议降级或移出：

- `vectorBytes`

原因：

- vector width 是访问/transport 合同，不是 memory object 自身真相；
- Triton 的 `MemDescType` 也不把它当对象主事实。

`alignmentBytes` 可以暂时保留，但需要明确：

- 它是 object property；
- 不是 layout property；
- 更不是 pipeline property。

---

## 7. analysis struct 的最终降级方案

和 Triton 对齐，不是把 analysis 层全删掉，而是要把它们从 owner 降成 summary。

## 7.1 `KernelConfig`

保留职责：

- 输入配置真相；
- problem/block/MMA family/用户给定 `numWarps`/`numStages`。

不再拥有：

- 推导出的 warp-grid；
- layout order；
- operand distribution；
- CTA split。

## 7.2 `MatmulSemantics`

保留职责：

- problem M/N/K
- tile M/N/K
- boundary ownership
- 输入 memref stride/order 的语义化结果
- 初始 source memdesc 构造

必须移出的 owner：

- `warpGridM`
- `warpGridN`

如果保留，也只能作为从 encoding 解析出的 cached summary，而不是 primary truth。

并且：

- `row-major contiguous only`

这种限制不应继续作为“语义层永久事实”，而应该降为当前 `stage1` lowering envelope。

## 7.3 `EncodingPlan`

迁移后它仍然可以存在，但角色必须改变。

它应该变成：

- 对当前 kernel 相关 encoding attr 的解析结果；
- 给后续 analysis/lowering 提供 typed handle 和摘要；
- 提供 `getBlockedEncodingAttr / getSharedEncodingAttr / getDotOperandEncodingAttr / getAccumulatorEncodingAttr` 之类 helper。

它不应该继续成为：

- layout 的主 owner；
- warp-grid 的主 owner；
- fragment lane 公式的唯一事实来源。

更直接地说：

- `EncodingPlan` 以后是 “encoding cache / validation summary”，
- 不是 “替代 encoding attr 的第二套 layout 模型”。

## 7.4 `ProgramMappingPlan`

保留职责：

- 程序 ID 到 tile 坐标的映射公式；
- grouped launch / row-major launch 的运行时 summary；
- totalPrograms / launchGroupCount 这类 launch 计算结果。

必须移出的 owner：

- CTA split
- CGA
- cluster physical distribution

这些应该进入：

- module attr
- `#tb.cga`
- 具体 encoding attr

`ProgramMappingPlan` 只表达 launch mapping，不表达 layout owner。

## 7.5 `BufferModel`

保留职责：

- backing / view / value / op graph；
- pipeline 和 async 的对象图。

但要求明确：

- backing 的 `descType` 才是真正的 memory/layout owner；
- `BufferModel` 只是引用它们，不再重新发明 shared/global/register 物理真相。

## 7.6 `AsyncPlan`

保留职责：

- producer/group/wait/reuse fence。

但要从这次迁移里获得一个新边界：

- async legality 读取 target + memdesc + shared encoding；
- 不再从 shared encoding attr 上读 `asyncEligible / asyncVecBytes` 这种内嵌捷径字段。

## 7.7 `EpiloguePlan`

保留职责：

- direct vs relay
- pack strategy
- direct global vector plan
- shared relay plan

但要求明确：

- direct/relay 是 epilogue owner；
- accumulator layout 不是 epilogue owner；
- shared physical layout 由 encoding/memdesc owner；
- epilogue 只能引用，不再复制。

---

## 8. op schema 的修改原则

这里最关键的一点是：

**不要把所有新 truth 又塞回 `tb.matmul` attr。**

这会走回旧路。

### 8.1 `tb.matmul` 应继续只承载输入配置

建议 `tb.matmul` 保留：

- `block_m`
- `block_n`
- `block_k`
- `num_warps`
- `num_stages`
- `group_m`
- `exact_tile`
- `mma`

如果未来要支持手工 override，可增加：

- `warp_partition`
- `cta_split`

但这只能是输入 override，不是强制所有 truth 永久都靠 op attr 保存。

### 8.2 真正冻结 owner 的地方是早期 layout-assign / matmul-accelerate pass

正确顺序应该是：

1. `tb.matmul` 提供输入配置；
2. `tb-semanticize-matmul` 只语义化源问题和 source memdesc；
3. 一个早期 pass 计算并冻结：
   - blocked encoding
   - MMA encoding
   - dot operand encoding
   - shared encoding
4. 后续 pass 一律消费 encoding attr；
5. 如果需要 layout 变换，显式引入 `tb.convert_layout` 或等价机制。

也就是说：

- `warp-grid` 最终 owner 应落到 encoding attr；
- 不是落到 `tb.matmul`，也不是继续落在 `MatmulSemantics`。

---

## 9. pass 重排原则

迁移完成后，主线应该更接近 Triton 的思想顺序：

1. `tb.matmul` 输入配置
2. attach context attr
3. semanticize source operands
4. freeze distributed encodings
5. matmul accelerate / mainloop rewrite
6. layout propagation / conversion cleanup
7. buffer / pipeline / async derivation
8. target lowering

其中最重要的变化是：

- layout/MMA/dot operand 真相要在 pipeline/async 之前冻结；
- pipeline/async 不能继续反向定义 layout。

---

## 10. 迁移实施顺序

这次迁移必须按下面顺序做，不能打散。

### 第一步：先改 attr/type

修改文件：

- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBAttrDefs.td)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBTypeDefs.td)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBAttrs.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBAttrs.cpp)
- [/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBTypes.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/IR/TBTypes.cpp)

目标：

- 建好新的 owner 语义层；
- verifier 能抓住双真相和非法组合。

### 第二步：再改 context attr 和 lookup API

需要补：

- `tb.num-warps`
- `tb.threads-per-warp`
- `tb.num-ctas`
- `tb.target`

并提供统一 lookup helper。

目标：

- 所有 pass 读同一份执行上下文。

### 第三步：再改 semanticize / encoding derive

修改重点：

- `MatmulSemantics`
- `EncodingPlan`

目标：

- 从 owner 层退回 summary 层；
- 删除 `warpGrid` 的 primary ownership；
- 由 early pass 冻结 encoding truth。

### 第四步：再改 mainloop rewrite 和 lowering

目标：

- rewrite / lowering 不再依赖旧式 heuristic plan；
- 全部改为读 attr/type truth。

### 第五步：最后才补 layout conversion / propagation

如果要继续严格向 Triton 靠拢，必须补：

- `tb.convert_layout` 或等价 IR；
- layout propagation / cleanup pass。

这一步不是当前 `stage1` 能否工作的前置条件，
但它是后续真正做 Triton 式布局优化的必要基础。

---

## 11. 验收标准

这次迁移是否完成，不看“字段变多了没有”，只看下面六条。

### 11.1 没有双真相

以下信息只能有一个 primary owner：

- shape
- allocShape
- warpsPerCTA
- threadsPerWarp
- order
- instrShape
- dot operand identity

### 11.2 `deriveWarpGrid(...)` 不再是主 owner

允许存在辅助推导函数。

但必须满足：

- 推导结果会冻结到 encoding attr；
- 后续 pass 不再把 `MatmulSemantics.warpGrid*` 当主来源。

### 11.3 shared attr 不再混角色和 transport

`#tb.shared` 不应再带：

- role
- logicalShape
- allocShape
- asyncEligible
- asyncVecBytes

### 11.4 lowering 主要从 type encoding 读 layout 真相

而不是继续从：

- `MatmulSemantics`
- `EncodingPlan.fragmentA/fragmentB`
- `ProgramMappingPlan`

里面拼接出第二份 layout 真相。

### 11.5 `ProgramMappingPlan` 不再拥有 CTA/CGA

它只表达 launch mapping，不表达 layout owner。

### 11.6 `EncodingPlan` 退成解析/缓存层

它可以保留，但它的职责必须明确变为：

- parse
- cache
- validate
- summary

而不是第二套 encoding model。

---

## 12. 迁移后对 `stage1` 的意义

这套修改不是为了“更像 Triton 而已”，而是为了解决你当前最容易反复返工的根因：

1. `warp-grid` 没冻结成唯一真相；
2. `layout truth` 还没有完全前移到 type encoding；
3. shared layout / memory object / transport legality 还混在一起；
4. analysis struct 还在代替 IR 做 ownership。

只要这四个点不改，后面无论你继续做：

- matmul rewrite
- async overlap
- layout cleanup
- epilogue 扩展

都容易再次回到“再补一个字段、再写一条 heuristic”的路上。

这份文档的目标，就是一次把这条路堵死。

---

## 13. 最终一句话结论

和 Triton 真正对齐的关键，不是继续扩 `MatmulSemantics / EncodingPlan / ProgramMappingPlan`，而是：

- 把执行上下文放回 module attr；
- 把 warp / CTA / MMA / dot operand / shared layout 放进 encoding attr；
- 把 shape / alloc / memory space 放进 memdesc；
- 让 analysis struct 全部退成 summary 和 validation 层。

这才是 `mini_triton_nvgpu_v1` 下一步应该做的 **根本性数据结构迁移**。
