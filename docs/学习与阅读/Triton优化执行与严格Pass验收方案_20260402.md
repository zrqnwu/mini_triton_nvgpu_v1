# `mini_triton_nvgpu_v1` Triton 优化顺序执行与逐 Pass 严格验收文档

## 1. 文档目的

这份文档回答的是下面这组实际执行问题：

- 在 `mini_triton_nvgpu_v1` 的数据结构重构完成之后，真正应该按什么顺序去做 Triton 式矩阵乘法优化；
- 哪些步骤是性能主链上的必须项；
- 哪些 Triton 里的能力当前可以暂缓，不必现在就做；
- 为什么不能一上来先做 `async` / `mma` / `lowering`，而必须先把结构真相定死；
- 后续每做完一个 pass，应该如何严格检查它是否真正复刻了 Triton 的思想，而不是只做了一个“能跑”的近似物。

本文是执行文档，不是抽象综述。

本文默认前提是：

- 目标先聚焦在单核矩阵乘法；
- 优先做 `exact-tile` 的窄主线；
- 第一优先级是性能主链闭合；
- 第一优先级不是功能覆盖面，不是高级硬件特性，不是 autotune。

---

## 2. 前置条件

在开始本文的优化顺序之前，必须先完成数据结构重构。

这里的“完成”不是指 public struct 名字改完，而是指下面两件事都成立：

1. `layout` 已经成为 MLIR 语义层硬真相。
2. `memory legality` 已经成为 MLIR 语义层硬真相。

也就是：

- `tb.matmul` 以及后续中层 IR 上，必须能直接读到 layout 真相；
- backing / view / memory space / alloc shape / encoding 这些信息，必须不再依赖 lowering 里二次猜测；
- lowering 只能消费合同，不能重新发明合同。

这部分的最终结构设计，以上一份文档为准：

- [mini_triton_v1_mlir_semantic_hard_truth_data_structure_final_design_20260402.md](/home/zhangruiqi/docs/mini_triton_v1_mlir_semantic_hard_truth_data_structure_final_design_20260402.md)

如果这一步没做完，就不能开始本文的优化主线。

原因很简单：

- Triton 的性能不是靠 pass 数量堆出来的；
- Triton 的性能首先来自 ownership 正确；
- ownership 不正确时，后面所有 async / wait / fragment / lowering 都会持续漏真相。

---

## 3. 总原则

## 3.1 不是复刻 Triton 的表面外形，而是复刻 Triton 的优化思想

这里的“复刻 Triton”不是指：

- pass 名称一模一样；
- pass 个数一模一样；
- 每个 pass 的粒度一模一样；
- PTX / SASS 一定逐字相同。

这里真正要复刻的是三件事：

1. 优化顺序
2. 真相归属
3. 每一步对后一步的依赖关系

一句话概括：

**先定结构真相，再做时序优化，最后做 target lowering。**

这和 Triton NVIDIA backend 的真实主线一致：

`TTIR -> TTGIR 语义化 -> layout/CTA 规划 -> matmul 专项重写 -> loop 规整 -> pipeline -> pipeline 后收口 -> target-specific lowering -> LLVM/PTX`

参考：

- [/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py](/home/zhangruiqi/triton/third_party/nvidia/backend/compiler.py)

## 3.2 性能导向下，必须先保证“思想等价”，再谈“指标接近”

后续每个 pass 完成后，不能只看：

- 能不能编译；
- IR 长得像不像；
- 某个 micro benchmark 有没有偶然变快。

必须先看：

- 这个 pass 是否真的把 Triton 在这一层负责的真相收进来了；
- 它的输入输出边界是否清楚；
- 是否把本应属于前一层或后一层的责任偷塞到了当前层；
- 是否引入了新的 dual truth；
- lowering 是否仍然在越权重推导。

如果思想没对齐，性能不会稳定。

---

## 4. 哪些必须做，哪些暂时没必要

## 4.1 必须做

下面这些，是单核 matmul 性能主链上的必须项。

### A. TTIR 到 TTGIR 的语义化

必须把高层 `tb.matmul` 变成真正带 layout / memdesc / program mapping 真相的中层 IR。

没有这一步，后续所有 pass 都会继续依赖：

- C++ plan 临时信息；
- lowering 内部硬编码；
- 某个 pass 的局部猜测。

### B. layout 规划

这是第一优先级。

至少必须定死：

- global 读的 layout；
- A/B shared layout；
- dot operand layout；
- accumulator layout；
- C direct epilogue layout；
- CTA 内 warp / lane 的分工与映射。

注意：

- 这里的 CTA 规划，当前必须的是 CTA 内部线程组织；
- 多 CTA 的分裂策略不是当前 blocking item。

### C. matmul 专项重写

必须把 matmul 改造成明确面向硬件 fragment / `ldmatrix` / `mma.sync` 的表达。

不能继续停留在：

- 只有 tile shape；
- 没有 operand fragment；
- 没有 lane access；
- 没有明确 A/B/C 主链形态。

### D. loop 规整

必须把主 K 循环规整成可 pipeline 的单一主循环。

没有这一层，后面的：

- latency；
- schedule；
- async producer；
- wait frontier；
- multibuffer；

都会退化成 patch。

### E. pipeline 主链

至少必须形成下面这条链：

`assign latencies -> schedule loops -> pipeline expand -> explicit async producer/wait`

这是性能主因，绝不是可选项。

### F. post-pipeline 最小收口

至少必须保证：

- 不再残留多余 layout conversion；
- async copy 粒度与 shared 视图合法；
- direct epilogue 不会被重新绕回 relay/shared staging；
- lowering 前的 IR 已经处于“真 mainline”状态。

### G. target-specific lowering

必须保证 lowering 只是消费前面的真相，不再重建前面的真相。

如果 lowering 里还写死：

- 元素类型；
- fragment 形状；
- async copy 粒度；
- shared alloc 形状；
- target chip；

那么前面数据结构和 pass 的收益会被直接打穿。

---

## 4.2 暂时没必要

下面这些能力是 Triton 有、但当前阶段可以明确后放的。

### A. 多 CTA 的完整 `PlanCTA`

当前目标是单核 matmul 主线闭合，不需要立刻做多 CTA 协同。

但要注意：

- CTA 内 warp/lane mapping 不是可选项；
- 可暂缓的是多 CTA 分裂，不是 CTA 内部映射。

### B. 通用 `OptimizeThreadLocality`

Triton 的这个 pass 还处理 reduction / gather / reshape。

你当前只做 matmul 主线，不需要先做成通用 pass。

只要 matmul 上的 lane/register/locality 真相已被正确表达即可。

### C. `F32DotTC`

如果当前先收敛在 `f16xf16 -> f32` 主线，这项可以暂时不做。

### D. 独立的 `Prefetch` 专项优化

先把：

- async producer；
- multibuffer；
- wait frontier；
- direct epilogue；

做对，再谈 prefetch。

### E. TMA / descriptor encoding / TMEM / WGMMA / warp specialization

这些都属于更高阶段的硬件专项。

它们不是当前“做出一条性能主链”的必要条件。

### F. `OptimizePartitionWarps` / `InterleaveTMEM` / `RemoveTMEMTokens`

这些也都不是当前 stage1 的 blocking item。

### G. autotune / heuristic / general-shape fallback

当前不应该优先做这些。

现在要先做的是：

- 窄主线；
- exact-tile；
- single-CTA；
- strict mainline。

---

## 5. 数据结构重构完成之后的严格优化顺序

下面这条顺序，是后续执行的强制顺序。

不能打散。

不能跳着做。

不能先改后面的 lowering 再回头补前面的结构层。

## Pass 0. 结构真相收口

目标：

- 把 `layout` 固化到 MLIR attr；
- 把 `memory legality` 固化到 MLIR type；
- 让 `KernelContract` 只承载计划层，不再偷带语义层硬真相的备份。

这一层对应的不是 Triton 某一个单独 pass，而是 Triton 的设计前提。

如果这一步没做完，后面的 pass 都会变成“半结构、半猜测”。

通过标准：

1. lowering 不再根据名字或 magic number 推断 shared/global/register 语义。
2. backing / view 的合法性可以通过 type/attr verifier 检查。
3. `EncodingPlan` 不再只是 `kind + payload` 的弱结构，而是走向 typed attr registry。
4. `BufferBacking` 的 memory legality 由 `!tb.memdesc` 一类语义结构承载。

## Pass 1. TTIR 到 TTGIR 语义化

目标：

- 让中层 IR 成为真正的优化入口；
- 把高层 matmul 的结构真相下沉到中层语义，而不是停在 C++ 临时结果里。

要复刻的 Triton 思想：

- 优化在 TTGIR 上进行，而不是在最终 lowering 阶段临时补形状；
- 真 layout 必须在 IR 中可见。

通过标准：

1. TTGIR 级别已经能看出 A/B/C 的 layout 与 memory legality。
2. 后续 pass 只读 TTGIR 语义，不再回头看高层参数猜物理布局。
3. 没有新的 bridge truth。

## Pass 2. layout / CTA 内线程映射规划

目标：

- 固定 global/shared/dot/acc/store 的布局关系；
- 固定 CTA 内 warp 与 lane 的职责。

要复刻的 Triton 思想：

- 先定访问布局和线程映射，再做 matmul 加速；
- `PlanCTA` 的当前必要内核是 CTA 内部映射，不是多 CTA 分裂全量功能。

通过标准：

1. A/B global 访问已经是确定的 coalesced truth。
2. A/B shared 物理布局已确定，且能支持后续 async copy 与 fragment load。
3. dot operand 与 accumulator encoding 已确定。
4. lane 到 fragment 的访问关系已固定，不能在 lowering 里重新发明。

## Pass 3. matmul 专项重写

目标：

- 把 matmul 改造成硬件友好的 fragment 主链。

要复刻的 Triton 思想：

- `accelerate_matmul` 和 `optimize_dot_operands` 的责任，是把 dot 真正改写成 tensor core 友好表达；
- 不是在 lowering 里看见 `dot` 再临时拼 `mma.sync`。

通过标准：

1. A/B operand fragment 的所有权明确。
2. accumulator fragment 的所有权明确。
3. shared 到 fragment、fragment 到 MMA 的路径已在中层被定出来。
4. 这个阶段之后，lowering 不再需要自己判断“应该用什么 fragment 形状”。

## Pass 4. loop 规整

目标：

- 形成单一、可 pipeline 的主 K 循环。

要复刻的 Triton 思想：

- 先把 loop shape 变成 pipelineable，再谈 latency 和 wait。

通过标准：

1. 主循环边界清楚。
2. 循环内部 A/B producer、MMA、C path 的依赖顺序清楚。
3. 不再存在结构性阻碍 pipeline 的嵌套形态。

## Pass 5. latency 标注

目标：

- 给 load / MMA / 关键操作标出 pipeline 所需 latency 真相。

要复刻的 Triton 思想：

- `assign_latencies` 先决定哪些 load 值得 pipeline、哪些 MMA 可以 overlap；
- 不是直接拍脑袋插 wait。

通过标准：

1. latency 来自明确的 producer-consumer 关系，而不是常量补丁。
2. async 资格、MMA overlap 资格有统一判断入口。
3. 不能把 wait policy 直接硬塞到 lowering。

## Pass 6. loop scheduling

目标：

- 根据 latency 给循环体操作分配 stage。

要复刻的 Triton 思想：

- 先 schedule，再 expand pipeline；
- schedule 是结构问题，不是 codegen 细节。

通过标准：

1. 每个关键 op 都有明确 stage 位置。
2. producer 与 first-use 的 frontier 是可解释的。
3. 没有“stage 看起来存在，但 lowering 不使用”的双真相。

## Pass 7. pipeline expand 与 explicit async 主线

目标：

- 真正形成 multibuffer、async producer、wait frontier。

要复刻的 Triton 思想：

- `pipeline` 的本质是先 lower loop 到可 async 的结构，再 expand，再更新 waits；
- exact-tile 主线必须是 explicit async，不允许退回同步主线伪装成 pipeline。

通过标准：

1. A/B producer 是显式 async producer。
2. wait 是显式 async wait，不是只剩 barrier。
3. multibuffer 是主线 truth，不是 fallback。
4. 不能在 exact-tile 主线里混入同步 producer fallback。

## Pass 8. post-pipeline 收口

目标：

- 收掉 pipeline 后制造的结构噪音，保住主线性能。

要复刻的 Triton 思想：

- Triton 在 pipeline 之后还会继续做 operand 收口、async copy coalescing、layout conversion 清理；
- pipeline 不是结束点。

通过标准：

1. 多余 convert 被压掉。
2. async copy 颗粒、shared view、dot operand 之间的一致性被重新验证。
3. direct epilogue 不会被重新绑回 relay/shared staging。

## Pass 9. target-specific lowering

目标：

- 无失真地把前面已确定的真相降到 NVGPU / NVVM。

要复刻的 Triton 思想：

- lowering 只消费上层真相；
- lowering 不负责重新规划 layout / fragment / memory legality。

通过标准：

1. lowering 中没有新的 layout 推导。
2. lowering 中没有新的 memory legality 推导。
3. lowering 中没有新的 async/mainloop owner 改写。
4. lowering 中没有针对当前 case 的 magic number 硬编码。

## Pass 10. LLVM / PTX

目标：

- 把已经正确的目标 IR 转成 PTX / cubin。

要复刻的 Triton 思想：

- PTX / SASS 接近是结果，不是前面的替代品；
- 不能用 PTX 层面的 patch 去弥补 TTGIR 层的结构错误。

---

## 6. 每做完一个 Pass 必须执行的严格验收

后续每完成一个 pass，都必须做下面五类检查。

缺一不可。

## 6.1 Ownership 检查

必须回答：

- 这个 pass 新引入或新固定的真相，owner 是谁；
- 它是 MLIR 语义层 owner，还是 plan 层 owner；
- 是否和已有 owner 冲突。

只要回答不清楚，这个 pass 就不能算完成。

## 6.2 输入输出边界检查

必须回答：

- 这个 pass 依赖前一层哪些真相；
- 这个 pass 产出给后一层哪些真相；
- 哪些事明确不归它管。

如果一个 pass 既在读取上层缺失真相，又在偷偷替后层做事，说明边界已经错了。

## 6.3 IR 证据检查

必须在 IR 层能看到这个 pass 产生的结果。

不能只说：

- C++ 结构里已经有了；
- 调试打印里看起来像对了；
- 跑起来似乎更快了。

IR 看不到，说明真相还没真正落地。

## 6.4 无 dual truth 检查

必须检查：

- 同一件事是否同时在 attr/type/plan/lowering 层各保存了一份；
- 是否存在前层说 direct、后层实际走 relay；
- 是否存在前层说 async、后层实际走 sync。

只要有 dual truth，这一 pass 不算完成。

## 6.5 禁止局部补丁检查

必须检查这个 pass 是否依赖了下面这些错误方式：

- case-specific magic number；
- lowering 内硬编码形状；
- fallback 假装 mainline；
- 先把东西塞进 shared 再声称是 direct path；
- 先同步搬运再声称做了 pipeline。

一旦发现这种情况，就不是 Triton 思想。

---

## 7. 性能导向下的执行纪律

## 7.1 不能边补结构边测性能

性能是整条链条的结果，不是单个 pass 的局部结果。

所以：

- pass 级别主要验思想、边界、IR 证据；
- stage1 主链闭合之后，再做性能测试；
- 不能一边结构半成品，一边反复测性能，然后被偶然结果带偏。

## 7.2 每个 pass 验的是“是否复刻了 Triton 思想”，不是“名字是否一样”

判断标准不是：

- 你有没有做一个叫 `RemoveLayoutConversions` 的 pass；

而是：

- 你有没有在正确时机做 layout 收口；
- 你有没有把本应清掉的 conversion 真正清掉；
- 你有没有防止 layout truth 在 lowering 里再次泄漏。

## 7.3 当前阶段先追求窄主线的真实性

当前阶段必须优先保证：

- exact-tile；
- single-CTA；
- direct epilogue；
- explicit async producer/wait；
- strict matmul mainline。

在这条线没有彻底闭合之前，不要优先做：

- fallback 扩展；
- general-shape 覆盖；
- 高级硬件特性；
- autotune。

---

## 8. 当前项目的实际执行结论

如果 `mini_triton_nvgpu_v1` 的数据结构重构已经完成，那么后续真正必须做的核心主线可以压成下面这 8 步：

1. `TTIR/高层 matmul -> 语义化 TTGIR`
2. `layout + CTA 内线程映射`
3. `matmul fragment / operand / accumulator 专项重写`
4. `单一主 K-loop 规整`
5. `latency 标注`
6. `schedule loops`
7. `pipeline + explicit async producer/wait`
8. `post-pipeline 收口 + target-specific lowering`

这就是当前版本最应该做的“性能骨架”。

只要这条骨架没有完整闭合，后面任何局部加速项都不能说明问题已经解决。

---

## 9. 最终原则

后续执行时，必须一直坚持下面三条。

1. 先做完数据结构重构，再进入优化顺序，不允许反过来。
2. 每做完一个 pass，先检查是否复刻了 Triton 的思想和 ownership，再判断是否完成。
3. 性能最终取决于整条主链是否闭合，而不是某个局部 patch 是否偶然有效。

这份文档就是后续 `mini_triton_nvgpu_v1` 矩阵乘法优化执行的总规范。
