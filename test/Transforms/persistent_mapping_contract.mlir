// RUN: %tb-opt %s --pass-pipeline='builtin.module(tb-verify-scope,tb-attach-target-info{gpu-arch=sm_86 ptx-features=+ptx60},tb-semanticize-matmul,tb-build-program-mapping-plan,tb-build-layout-plan,tb-build-transport-plan,tb-build-c-register-plan,tb-rewrite-matmul-mainloop,tb-cleanup-layout-conversions,tb-build-mainloop-graph,tb-regularize-k-loop,tb-assign-latencies,tb-schedule-loops,tb-derive-waits,tb-expand-pipeline,tb-cleanup-pipeline,tb-verify-pipeline-mainline-contract)' | FileCheck %s

module attributes {"tb.num-warps" = 4 : i64, "tb.requested-stages" = 2 : i64,
                   "tb.persistent" = true, "tb.num-ctas" = 2 : i64} {
  func.func @kernel(%A: memref<192x32xf16>, %B: memref<32x128xf16>,
                    %C: memref<192x128xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = true}
      : memref<192x32xf16>, memref<32x128xf16>, memref<192x128xf32>
    func.return
  }
}

// CHECK: tb.pipeline_mainline
// CHECK-SAME: tb.persistent_work_plan = {
// CHECK-SAME: kind = "tile_stride_loop"
// CHECK-SAME: resident_programs = 2
// CHECK-SAME: mapping_kind = "persistent_tile"
// CHECK-SAME: selected_mainline_kind = "persistent_tile_loop"
