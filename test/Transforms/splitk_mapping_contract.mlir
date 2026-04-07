// RUN: %tb-opt %s --pass-pipeline='builtin.module(tb-verify-scope,tb-attach-target-info{gpu-arch=sm_86 ptx-features=+ptx60},tb-semanticize-matmul,tb-build-program-mapping-plan,tb-build-layout-plan,tb-build-transport-plan,tb-build-c-register-plan,tb-rewrite-matmul-mainloop,tb-cleanup-layout-conversions,tb-build-mainloop-graph,tb-regularize-k-loop,tb-assign-latencies,tb-schedule-loops,tb-derive-waits,tb-expand-pipeline,tb-cleanup-pipeline,tb-verify-pipeline-mainline-contract)' | FileCheck %s

module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64,
                   "tb.split-k" = 2 : i64,
                   "tb.reduction-mode" = "split_k_parallel"} {
  func.func @kernel(%A: memref<64x64xf16>, %B: memref<64x64xf16>,
                    %C: memref<64x64xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = true}
      : memref<64x64xf16>, memref<64x64xf16>, memref<64x64xf32>
    func.return
  }
}

// CHECK: tb.pipeline_mainline
// CHECK-SAME: mapping_kind = "split_k"
// CHECK-SAME: split_k = 2
// CHECK-SAME: tb.reduction_plan = {
// CHECK-SAME: kind = "split_k_atomic"
// CHECK-SAME: selected_mainline_kind = "split_k"
