# 2026-04-07 结果归档

这批结果对应当前 `stage1` 主线的真实硬件测试与整理。

## 目录

- `fair_benchmark/`
  - `stage1_fair_report.txt`
  - `stage1_fair_report.json`
  - `160x96_after_fix.lowered.mlir`
  - `triton_160x96_inspect.ptx`
  - `triton_160x96_inspect.json`
- `hardware_profile/`
  - `mini_triton_stage1_hw_summary_20260407.txt`
  - `mini_triton_stage1_hw_summary_20260407.json`
  - `ncu_160x96_mini.csv`
  - `ncu_160x96_triton.csv`
  - `ncu_96x80_mini.csv`
  - `ncu_96x80_triton.csv`
- `large_shapes/`
  - `large_shapes_report.json`
- `stage1_closure_20260407/`
  - `stage1_fair_report.txt`
  - `stage1_fair_report.json`
  - `smoke_splitk_64x64x64.lowered.mlir`
  - `smoke_persistent_192x128x32.lowered.mlir`

## 关键结论

- `160x96x32_general`
  - mini `3631.59 ns`
  - Triton `3674.75 ns`
  - mini 快 `1.19%`
- `96x80x32_general`
  - mini `4143.04 ns`
  - Triton `5005.82 ns`
  - mini 快 `20.82%`
- `256x256x64`
  - mini `6291.41 ns`
  - Triton `8035.56 ns`
  - mini 快 `27.72%`
- `256x256x128`
  - mini `12170.40 ns`
  - Triton `13585.50 ns`
  - mini 快 `11.63%`
- `320x224x64_general`
  - mini `5424.57 ns`
  - Triton `6534.27 ns`
  - mini 快 `20.46%`

新增这轮 stage1 收口结果：

- `224x160x64_general`
  - mini `4617.42 ns`
  - Triton `4732.08 ns`
  - mini 快 `2.48%`
- `320x224x64_general`
  - mini `5388.66 ns`
  - Triton `6548.27 ns`
  - mini 快 `21.52%`
- `64x64x64_splitk2`
  - mini-only runtime/correctness 闭环已打通
  - 结果归档在 `stage1_closure_20260407/`
- `192x128x32_persistent`
  - mini-only runtime/correctness 闭环已打通
  - 结果归档在 `stage1_closure_20260407/`

## 备注

- 这里保留的是文本、JSON、CSV、MLIR、PTX 结果，方便复盘与对比。
- 没有把临时 `.bin` / `.cubin` 构建产物整体提交进仓库，避免仓库变成制品仓库。
- `split-k / persistent` 这轮先补的是 mini runtime 闭环与 lowering 归档，不是 Triton 一对一公平对照。
