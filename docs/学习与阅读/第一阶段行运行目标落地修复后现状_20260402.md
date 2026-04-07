# `mini_triton_nvgpu_v1` Stage1 当前状态收口文档

## 1. 本轮真正完成了什么

这轮不是继续局部调参，而是把 `EpiloguePlan` 里 target landing 的一个错误 owner 修掉：

- 不再把 `sharedPackSlots = 4` 当成硬编码常数。
- 改成从 direct pack 的自然 `row-run` 批次推导 slot 数。
- 同时受静态 shared 保守预算约束，避免生成“总 shared 看起来够，但 static attribution 已经越界”的错误形状。

对应代码在：

- [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp#L860)
- [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp#L888)
- [EpiloguePlan.cpp](/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Analysis/EpiloguePlan.cpp#L907)

本轮收口后的关键逻辑是：

1. 先从 direct packs 里找出同一 `rowBase` 的完整连续 pack-run。
2. 把它当成 direct C landing 的自然 batch 单位。
3. 再根据 `sharedABBytes` 和静态 shared 预算，计算最多还能容纳多少个 C pack slot。
4. 最终 `sharedPackSlots = min(rowRunPackCount, maxPackSlotsByBudget)`。

这条改动的意义是：

- 去掉了 magic number。
- 保住了 direct C landing 的自然 pack 组织。
- 避免回到之前那种 full-pack staging 导致 `64` 退化、`128` 直接 `CUDA_ERROR_INVALID_PTX` 的错误方向。

---

## 2. 现在的产物形状

当前稳定产物：

- mini runtime:
  - [/tmp/mini_tb_current_64.mlir](/tmp/mini_tb_current_64.mlir)
  - [/tmp/mini_tb_current_128.mlir](/tmp/mini_tb_current_128.mlir)
- mini payload:
  - [/tmp/mini_tb_current_64.cubin](/tmp/mini_tb_current_64.cubin)
  - [/tmp/mini_tb_current_128.cubin](/tmp/mini_tb_current_128.cubin)

当前 emitted kernel 形状已经回到稳定版本：

### `64x64x32`

- `static_shared = 12288`
- `bar.warp.sync = 8`
- `ld.global.v4.b32 = 32`
- `st.global.v4.b32 = 32`
- `st.shared.v4.b32 = 32`
- `ld.shared.v2.b32 = 64`
- `st.shared.v2.b32 = 64`

### `128x128x32`

- `static_shared = 32768`
- `bar.warp.sync = 8`
- `ld.global.v4.b32 = 32`
- `st.global.v4.b32 = 32`
- `st.shared.v4.b32 = 32`
- `ld.shared.v2.b32 = 64`
- `st.shared.v2.b32 = 64`

这说明当前 direct epilogue target landing 已经稳定回到之前最佳结构，不再是 full-pack staging 的错误版本。

---

## 3. 最终 benchmark 结果

最终串行交错 benchmark 文件：

- [/tmp/bench_final_64.txt](/tmp/bench_final_64.txt)
- [/tmp/bench_final_128.txt](/tmp/bench_final_128.txt)

统一口径：

- 单卡串行
- `warmup=500`
- `iters=20000`
- mini 和 Triton 交错跑，避免同一方长期占热态

### `64x64x32`

warm 后 median：

- mini: `3976.14 ns`
- Triton: `3651.98 ns`
- mini 相对 Triton: `+8.88%`

结论：

- 还没有追平 Triton。
- 但已经明显接近，不再是早期那种主线走错导致的大幅落后。

### `128x128x32`

warm 后 median：

- mini: `5737.68 ns`
- Triton: `5962.09 ns`
- mini 相对 Triton: `-3.76%`

结论：

- 当前 `128x128x32` 已经达到并略快于 Triton。

---

## 4. 最终 NCU 结果

最终 NCU 文件：

- [/tmp/ncu_final_mini_64.csv](/tmp/ncu_final_mini_64.csv)
- [/tmp/ncu_final_triton_64.csv](/tmp/ncu_final_triton_64.csv)
- [/tmp/ncu_final_mini_128.csv](/tmp/ncu_final_mini_128.csv)
- [/tmp/ncu_final_triton_128.csv](/tmp/ncu_final_triton_128.csv)

### `64x64x32`

mini：

- `gpu__time_duration.sum = 8768`
- `registers/thread = 254`
- `shared/block = 13312`
- `long_scoreboard = 4.24`
- `barrier = 0.03`
- `dram throughput = 2.98`

Triton：

- `gpu__time_duration.sum = 7392`
- `registers/thread = 252`
- `shared/block = 13568`
- `long_scoreboard = 0.57`
- `barrier = 0.57`
- `dram throughput = 3.52`

结论：

- `64` 的主差距不是 barrier。
- shared 和 register 也已经非常接近。
- 真正的剩余差距更像是后段 memory dependency / scoreboard hiding 不如 Triton。

### `128x128x32`

mini：

- `gpu__time_duration.sum = 11872`
- `registers/thread = 254`
- `shared/block = 33792`
- `long_scoreboard = 5.88`
- `barrier = 0.35`
- `dram throughput = 6.27`

Triton：

- `gpu__time_duration.sum = 11616`
- `registers/thread = 237`
- `shared/block = 34304`
- `long_scoreboard = 1.13`
- `barrier = 1.48`
- `dram throughput = 5.93`

结论：

- 单次 NCU 采样里 Triton 的 kernel duration 仍略好。
- 但 steady-state benchmark 已经是 mini 更快。
- 这说明当前 `128` 至少在真实多次运行口径下，已经不构成 stage1 的主要性能阻塞点。

---

## 5. 现在还剩下的真实问题

截至本轮，已经可以明确排除的旧根因有：

1. `A/B async producer` 主线缺失。
2. `direct_global_vector` 完全丢失。
3. `sharedPackSlots=4` 只是 magic number，必须继续扩大。

当前还真实成立的问题只有一类：

- `64x64x32` 上，C/epilogue 后段的 dependency 链和 Triton 仍不完全一样，导致 `long_scoreboard` 明显偏高。

换句话说：

- 现在不是“结构完全错了”。
- 现在是“结构已经基本对了，但 `64` 小 tile 的 epilogue 指令/依赖组织还不够像 Triton”。

---

## 6. 为什么不能再回到 full-pack staging

这轮已经实际验证过，full-pack staging 不是正确方向。

它带来的结果是：

- `64` static shared 直接翻倍到 `24576`，benchmark 退化。
- `128` static shared 变成 `81920`，直接 `CUDA_ERROR_INVALID_PTX`。

所以后续不能再把“让 sharedPackSlots 更大”当成默认正确方向。

正确结论是：

- `4-slot` 之前看起来像 magic number。
- 但根因上它对应的是 direct row-run 的自然 batch。
- 现在已经把它从硬编码常数改成了 row-run + static-shared-budget 推导。

这一步已经收口，不应该再反复改回去。

---

## 7. 当前 stage1 结论

如果只看 stage1 exact-tile matmul 主线：

- `128x128x32`：已经达到 Triton 水平，steady-state 甚至略快。
- `64x64x32`：仍差约 `9%`，但差距已经缩到单一剩余根因。

因此当前更准确的判断是：

- stage1 主结构已经基本成立。
- 还没有做到“所有 exact-tile case 都严格贴平 Triton”。
- 剩余最值得继续攻的是 `64x64x32` 的 epilogue dependency / scheduling。

---

## 8. 下一步如果继续做，正确方向是什么

如果后面还要继续追 `64`，正确方向应该是：

1. 对照 Triton 的 `64` epilogue 指令序列，分析 shared unpack 与 global vector load/store 的交错顺序。
2. 看当前 mini 的 C 后段是否把 load-use / store-dependency 链排得过于紧。
3. 优先改 epilogue materialization 的 dependency 组织，而不是再改 slot 数、barrier 数或 shared 大小。

不应该再做的事：

1. 不要把 `sharedPackSlots` 再改成 full-pack。
2. 不要回到 generic `vector.load/store` 直接赌 sink 不 scalarize。
3. 不要把当前 `64` 的问题误判成 async producer 或 shared 容量问题。

---

## 9. 本轮最终结论

这轮已经完成的收口是有效的。

它的结果不是“完全追平 Triton”，而是：

- 把 target landing 的 slot 决策从硬编码常数改成了结构化推导。
- 保住了当前最优稳定产物形状。
- 让 `128x128x32` 达到并略超 Triton。
- 把 `64x64x32` 的剩余差距收敛到单一主根因：`epilogue` 后段 dependency / scoreboard 组织。

所以当前的正确状态不是“还在乱调”，而是：

- 大方向已经对了。
- 剩余问题已经足够具体，可以继续做下一轮针对 `64` 的定向重构。
