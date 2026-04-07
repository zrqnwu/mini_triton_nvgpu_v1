# `mini_triton_nvgpu_v1` 当前性能与 Triton 对比及根因复核

## 1. 文档目的

这份文档只回答一个问题：

- 截至 `2026-04-02` 当前最新代码，`mini_triton_nvgpu_v1` 和 Triton 的真实性能差距还有多少。
- 之前判断过的根因里，哪些现在仍然成立，哪些已经不再是主根因。

本文不复述旧阶段方案，也不沿用历史结论。只认这次基于当前代码、当前产物、当前硬件的重新测量与代码证据。

---

## 2. 测试对象与口径

测试机器：

- GPU: `NVIDIA GeForce RTX 3050`
- 架构: `sm_86`
- 驱动: `570.211.01`

测试形状：

- `64x64x32`
- `128x128x32`

统一约束：

- 都是 exact-tile 单 CTA。
- `64x64x32` 对齐 `num_warps=1, num_stages=2`。
- `128x128x32` 对齐 `num_warps=4, num_stages=2`。
- mini 与 Triton 都用同一套 host driver 计时逻辑，串行运行，每个 case 跑 `3` 次，取中位数。
- 不接受并行 benchmark 结果，因为同卡并行会互相抢资源，口径不公平。

相关输入文件：

- mini `64`:
  - [smoke_64x64x32.mlir](/home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir)
- mini `128`:
  - [smoke_128x128x32.mlir](/home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_128x128x32.mlir)
- 统一计时工具：
  - [driver_matmul_bench.cpp](/home/zhangruiqi/tmp/driver_matmul_bench.cpp)

这次实际生成的当前产物：

- mini lowering:
  - [/tmp/mini_tb_current_64.mlir](/tmp/mini_tb_current_64.mlir)
  - [/tmp/mini_tb_current_128.mlir](/tmp/mini_tb_current_128.mlir)
- mini 可加载二进制 payload:
  - [/tmp/mini_tb_current_64.cubin](/tmp/mini_tb_current_64.cubin)
  - [/tmp/mini_tb_current_128.cubin](/tmp/mini_tb_current_128.cubin)
- Triton 当前产物：
  - [/tmp/triton_exact_tile_64x64x32_real.ptx](/tmp/triton_exact_tile_64x64x32_real.ptx)
  - [/tmp/triton_exact_tile_128x128x32_real.ptx](/tmp/triton_exact_tile_128x128x32_real.ptx)
  - [/tmp/triton_exact_tile_64x64x32_real.cubin](/tmp/triton_exact_tile_64x64x32_real.cubin)
  - [/tmp/triton_exact_tile_128x128x32_real.cubin](/tmp/triton_exact_tile_128x128x32_real.cubin)

补充说明：

- 当前 mini 的提取文件名虽然叫 `.cubin`，但 `file` 显示它是 `ASCII text`，说明当前提取出来的是可被 `cuModuleLoadData` 接受的文本 payload，不是 Triton 那种 ELF cubin。
- 这不影响本次 kernel 计时，因为计时区间只包 kernel launch，不包 module load。
- 但这意味着如果后续要做严格的 `cuobjdump / ncu / SASS` 对齐，mini 这边最好继续收口到真正的 ELF cubin。

---

## 3. 当前真实性能结果

### 3.1 `64x64x32`

中位数结果：

- mini: `4309.04 ns`, `60.8358 GFLOPS`
- Triton: `3677.80 ns`, `71.2774 GFLOPS`
- mini / Triton: `85.35%`

### 3.2 `128x128x32`

中位数结果：

- mini: `10544.90 ns`, `99.4387 GFLOPS`
- Triton: `7710.11 ns`, `136.0000 GFLOPS`
- mini / Triton: `73.12%`

### 3.3 当前结论

当前 mini 已经明显接近 Triton，但还没有达到 Triton。

而且 gap 呈现出明显特征：

- 小 tile `64x64x32` 已经比较接近。
- 大 tile `128x128x32` 差距更大。

这说明当前问题已经不是“主线完全走错”那种一级崩坏，而是后段数据组织与落地方式在更大 tile 下仍然不够贴近 Triton。

---

## 4. 资源形态对比

### 4.1 `64x64x32`

- mini:
  - `regs=192`
  - `static_shared=8192`
- Triton:
  - `regs=252`
  - `static_shared=0`
  - runtime metadata `shared=12544`

### 4.2 `128x128x32`

- mini:
  - `regs=172`
  - `static_shared=16384`
- Triton:
  - `regs=237`
  - `static_shared=0`
  - runtime metadata `shared=33280`

### 4.3 这组差异说明什么

mini 和 Triton 当前不是同一种 kernel 组织方式。

如果只是“同一路径上还有一些小调优没做”，通常不会同时出现下面这几件事：

- register 压力分布差很多
- static/dynamic shared 使用方式不同
- 大 tile 下性能差距明显放大

所以当前差距不是简单的“还差一点点调参”，而是 kernel 后段组织方式还没有完全贴到 Triton 的真实形态。

---

## 5. 当前代码证据

## 5.1 mini 当前 A/B producer 主线

在当前 mini lowering 中，A/B async producer 主线已经明确存在：

- `cp.async.cg.shared.global`
- `cp.async.commit_group`
- `cp.async.wait_group`

对应文件：

- [/tmp/mini_tb_current_64.mlir](/tmp/mini_tb_current_64.mlir)
- [/tmp/mini_tb_current_128.mlir](/tmp/mini_tb_current_128.mlir)

计数：

### `64x64x32`

- `cp.async.cg.shared.global = 16`
- `cp.async.commit_group = 4`
- `cp.async.wait_group = 4`
- `ldmatrix = 17`
- `mma.sync = 64`
- `bar.sync = 2`

### `128x128x32`

- `cp.async.cg.shared.global = 8`
- `cp.async.commit_group = 4`
- `cp.async.wait_group = 4`
- `ldmatrix = 17`
- `mma.sync = 64`
- `bar.sync = 2`

结论：

- “当前主线没有 async producer / async wait”这条旧结论，已经不再成立。
- 至少对当前 exact-tile 主线，A/B async 这条链已经被修回来了。

## 5.2 mini 当前 C / epilogue 落地形态

当前 mini 的 C 路径仍然是大量标量 global load/store：

### `64x64x32`

- `ld.global.b32 = 128`
- `st.global.b32 = 128`
- `ld.global.v4.b32 = 0`
- `st.global.v4.b32 = 0`

### `128x128x32`

- `ld.global.b32 = 128`
- `st.global.b32 = 128`
- `ld.global.v4.b32 = 0`
- `st.global.v4.b32 = 0`

对应文件：

- [/tmp/mini_tb_current_64.mlir](/tmp/mini_tb_current_64.mlir)
- [/tmp/mini_tb_current_128.mlir](/tmp/mini_tb_current_128.mlir)

## 5.3 Triton 当前 C / epilogue 落地形态

Triton 当前对应 PTX 明确保留了 vectorized global landing：

### `64x64x32`

- `ld.global.v4.b32 = 48`
- `st.global.v4.b32 = 32`
- `ld.global.b32 = 0`
- `st.global.b32 = 0`
- `st.shared.v4.b32 = 48`
- `ld.shared.v2.f32 = 64`

### `128x128x32`

- `ld.global.v4.b32 = 40`
- `st.global.v4.b32 = 32`
- `ld.global.b32 = 0`
- `st.global.b32 = 0`
- `st.shared.v4.b32 = 40`
- `ld.shared.v2.f32 = 64`

对应文件：

- [/tmp/triton_exact_tile_64x64x32_real.ptx](/tmp/triton_exact_tile_64x64x32_real.ptx)
- [/tmp/triton_exact_tile_128x128x32_real.ptx](/tmp/triton_exact_tile_128x128x32_real.ptx)

结论：

- 当前 mini 和 Triton 的最大形态差异，仍然在 C / epilogue 这一段。
- mini 现在还没有把 direct/global vector landing 保到最终 PTX。
- Triton 已经把它保留成 `global v4` 和 `shared pack` 的组合。

---

## 6. 对“之前那些根因”的重新排序

## 6.1 已经不再是主根因的项

### A/B async producer 主线丢失

这条在当前最新代码里不再成立。

理由：

- mini 当前已经有 `cp.async + commit_group + wait_group`。
- 说明“完全退回同步 producer”不是当前主线的真实问题。

因此，后续不应该再把主要精力放在“重新证明 async 已经存在”这件事上。

## 6.2 仍然成立的主根因

### 根因一：C / epilogue direct-global-vector 没有真正保到最终 target

当前 mini 的最终产物仍然把 C 路径落成：

- `ld.global.b32`
- `st.global.b32`

而不是 Triton 的：

- `ld.global.v4.b32`
- `st.global.v4.b32`

这说明问题不是“有没有 direct 名义”，而是 direct/vector 的真实 owner 没有一路保到最终 target 代码。

也就是说，当前 mini 仍然存在：

- 上层合同里已经想表达 direct/vector
- 但 lower 到最后又被标量化

这和之前判断完全一致，而且现在被最新产物再次证实。

### 根因二：后段 shared/register/global 组织仍未贴到 Triton

Triton 当前明显存在：

- `st.shared.v4.b32`
- `ld.shared.v2.f32`
- `st.global.v4.b32`

而 mini 当前没有这套后段组织。

这说明 mini 现在虽然已经把前半段 mainloop 对齐到：

- `cp.async`
- `ldmatrix`
- `mma.sync`

但 epilogue 那段还没有形成 Triton 风格的：

- vector pack
- shared staging
- vector global landing

对于更大的 tile，这会直接拉开吞吐差距。

### 根因三：大 tile 的真实 kernel 形态仍未闭合

`128x128x32` 的 gap 比 `64x64x32` 更大，这一点很关键。

如果当前剩下的只是一些局部小问题，那么大 tile 和小 tile 不会出现这么明显的比例差异。

这说明：

- 小 exact-tile 主线已经较接近 Triton
- 但大 tile 下，Triton 的后段组织优势开始充分体现
- mini 这边还没有把同等级的执行形态闭合出来

---

## 7. 哪些旧判断现在应该明确废弃

## 7.1 “当前慢，是因为 async 没了”

对当前最新代码，这个说法不准确。

更准确的说法是：

- async 主线已经回来了
- 但 C / epilogue 与后段组织还没贴齐 Triton

## 7.2 “只要 barrier 少就会更快”

这也是错误判断。

当前 Triton PTX 中 `bar.sync` 计数反而明显更多，但性能仍然更好。

说明：

- `barrier` 数量本身不是主解释变量
- 真正关键的是 barrier 所服务的数据组织方式是否正确

不能再用“barrier 个数接近 Triton”替代真实的 kernel 组织对齐。

## 7.3 “opcode 数量看起来像了，就说明性能问题快解决了”

这条也必须废弃。

现在 mini 已经有：

- `cp.async`
- `ldmatrix`
- `mma.sync`

但性能仍然落后 Triton，特别是大 tile。

这说明：

- 只看主链 opcode 是否出现，不足以判断是否真正对齐 Triton
- epilogue 与后段数据落地组织才是当前差距的关键部分

---

## 8. 当前最终判断

截至 `2026-04-02`，当前代码的真实状态可以总结为三句话：

1. Stage1 的 exact-tile mainloop 主线已经基本站住了，A/B async producer 不再是当前主要矛盾。
2. 当前与 Triton 的主要差距，已经集中到 C / epilogue 的 vector landing 和后段 shared/register/global 组织。
3. 现在最不该做的事，是继续围绕“async 有没有”“barrier 多不多”这种已经不是主因的问题反复调整。

因此，后续如果还要继续追平 Triton，重点必须转到下面两件事：

- 把 C direct/global vector owner 真正保到最终 target，不再在 lower 末端退化成标量 `ld.global.b32 / st.global.b32`
- 按 Triton 当前真实形态，把 epilogue 的 shared pack / register unpack / vector global landing 主线补齐

---

## 9. 当前建议的执行原则

后续修改时应只遵守下面这三条：

1. 不再把“async producer 是否存在”当作当前第一优先级。
2. 所有优化判断必须同时看：
   - 性能
   - 资源形态
   - 最终 PTX / SASS 的真实落地形态
3. 只要最终仍然是 `st.global.b32` 大片标量写回，就不能宣称已经真正对齐 Triton 的 direct/vector C path。

这份文档就是当前阶段的复核结论基线。后续再做性能判断时，应以本文而不是更早的历史结论为准。
