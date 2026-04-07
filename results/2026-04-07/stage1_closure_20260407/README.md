# stage1_closure_20260407

这批归档对应 2026-04-07 针对以下收口项的验证：

- `bench_stage1_fair.py` 支持 `split-k / persistent / mini-only`
- `split-k` smoke lowering
- `persistent` smoke lowering
- general-shape 固定回归形状扩到更大的 `224x160x64` 与 `320x224x64`

## 文件

- `stage1_fair_report.txt`
  - 文本摘要
- `stage1_fair_report.json`
  - 完整样本、配置、launch、寄存器和 shared 信息
- `smoke_splitk_64x64x64.lowered.mlir`
  - split-k lowering 归档
- `smoke_persistent_192x128x32.lowered.mlir`
  - persistent lowering 归档

## 关键结论

- `96x80x32_general`
  - mini `5539.33 ns`
  - Triton `6656.51 ns`
  - mini 快 `20.17%`
- `224x160x64_general`
  - mini `4617.42 ns`
  - Triton `4732.08 ns`
  - mini 快 `2.48%`
- `320x224x64_general`
  - mini `5388.66 ns`
  - Triton `6548.27 ns`
  - mini 快 `21.52%`
- `64x64x64_splitk2`
  - mini-only correctness/runtime 已打通
- `192x128x32_persistent`
  - mini-only correctness/runtime 已打通

## 备注

- 这批 bench 使用的是快速闭环参数：
  - `rounds=3`
  - `warmup=50`
  - `iters=2000`
- `split-k / persistent` 这轮重点是把运行闭环和归档入口补起来，不是做 Triton 一对一对照。
