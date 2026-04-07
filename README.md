# `mini_triton_nvgpu_v1`

这是一个独立于现有 `triton_backend_nvgpu` 主链的 V1 原型。

当前已经完成：

- `tb` dialect 外壳
- 高层 `tb.matmul` op
- `tb-opt`
- `tb-semanticize-matmul`
- `tb-attach-target-info`
- Triton 对齐的 `KernelConfig / TargetInfo / EncodingPlan / BufferModel / PipelinePlan / AsyncPlan / AccumulatorPlan / EpiloguePlan`
- 旧 public `KernelSpec / LayoutPlan / CRegisterPlan` 已从数据结构主链移除
- `tb-verify-scope`
- `tb-build-program-mapping-plan`
- `tb-build-layout-plan`（兼容旧脚本名，实际构建 `tb.encoding_plan`）
- `tb-build-transport-plan`
- `tb-build-c-register-plan`（兼容旧脚本名，实际构建 `tb.accumulator_plan + tb.epilogue_plan`）
- `tb-rewrite-matmul-mainloop`
- `tb-cleanup-layout-conversions`
- `tb-build-mainloop-graph`
- `tb-regularize-k-loop`
- `tb-assign-latencies`
- `tb-schedule-loops`
- `tb-derive-waits`
- `tb-expand-pipeline`
- `tb-cleanup-pipeline`
- `tb.pipeline_mainline` 显式 IR owner
- `tb-stage1-full-to-nvvm-pipeline`
- `tb-stage1-nvgpu-to-nvvm-pipeline`

## 架构文档

- [Stage1 数据结构与主线 Pass 讲解](docs/stage1_data_structures_and_passes.md)
- [文档总览](docs/README.md)

仓库内还额外整理了两类中文文档：

- `docs/方案与复盘/`
  主要放阶段方案、根因复盘、性能收口与 Triton 对齐设计文档
- `docs/学习与阅读/`
  主要放学习路径、阅读顺序、项目讲解和面试准备材料

## 结果归档

- 最新真实 benchmark / NCU / 大形状测试结果归档在 `results/2026-04-07/`
- 入口说明见 [results/2026-04-07/README.md](results/2026-04-07/README.md)

## 构建

```bash
cmake -S /home/zhangruiqi/mini_triton_nvgpu_v1 \
  -B /home/zhangruiqi/mini_triton_nvgpu_v1/build \
  -G Ninja \
  -DMLIR_DIR=/home/zhangruiqi/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/zhangruiqi/llvm-project/build/lib/cmake/llvm

cmake --build /home/zhangruiqi/mini_triton_nvgpu_v1/build -j8
```

## Smoke Run

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

当前这个阶段只验证：

- V1 合法范围
- module-level `tb.target / tb.num-warps / tb.threads-per-warp / tb.num-ctas` 显式 owner 挂载
- `tb.semantic_matmul` 的 TTGIR 语义化挂载
- `tb.program_mapping_plan` 的 program/CTA owner truth 挂载
- `tb.encoding_plan` 的显式挂载
- `tb.transport_plan` 的显式 global/shared transport owner 挂载
- `tb.accumulator_plan + tb.epilogue_plan` 的 C contract 挂载
- `tb.matmul_rewrite` 的 fragment/mainloop 重写挂载
- `tb.convert_layout` / `tb-cleanup-layout-conversions` 的显式 layout 转换货币与基础收口
- `tb.buffer_model` 的 backing/view/value/op 挂载
- `tb.loop_plan` 的单一 K-loop 规整挂载
- `tb.latency_plan` 的显式挂载
- `tb.pipeline_plan` 的 stage/cluster/order 挂载
- `tb.async_plan` 的 producer/group/wait/reuse 挂载
- `tb.pipeline_expansion` 的 expanded-cluster/wait owner 挂载
- `tb.pipeline_mainline` 的显式 stage/cluster 有序 mainline cluster IR
- `tb.pipeline_ready` 的 strict mainline 收口挂载

## Full Lowering

```bash
/home/zhangruiqi/mini_triton_nvgpu_v1/build/bin/tb-opt \
  --pass-pipeline='builtin.module(tb-stage1-full-to-nvvm-pipeline{cubin-chip=sm_86 cubin-features=+ptx60 cubin-format=isa})' \
  /home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir
```

这条 pipeline 做两件事：

- 先用 `cubin-chip / cubin-features` 显式种下 module-level target context attr，再跑完整 stage1 strict mainline pass 链，直到 `tb-lower-pipeline-to-nvgpu`
- 再跑 repo-native NVGPU/NVVM sink pipeline，显式补上 `convert-vector-to-llvm`
  这一步是为了消除 fragment slice 在 `nvgpu -> nvvm` 之后遗留的 `unrealized_conversion_cast`

## Fair Benchmark

```bash
python3 /home/zhangruiqi/mini_triton_nvgpu_v1/tools/bench_stage1_fair.py
```

这条脚本会：

- fresh lowering 当前 mini stage1 exact-tile 六个 shape
- 交错运行 mini / Triton，避免单边先热机
- 多轮统计 `median/mean/min/max/spread`
- 输出
  - `/tmp/mini_triton_stage1_fair_bench/stage1_fair_report.txt`
  - `/tmp/mini_triton_stage1_fair_bench/stage1_fair_report.json`
