// RUN: %tb-opt %s --pass-pipeline='builtin.module(tb-verify-scope,tb-attach-target-info{gpu-arch=sm_86 ptx-features=+ptx60},tb-semanticize-matmul,tb-build-program-mapping-plan,tb-build-layout-plan,tb-build-transport-plan,tb-build-c-register-plan,tb-rewrite-matmul-mainloop,tb-cleanup-layout-conversions,tb-build-mainloop-graph,tb-regularize-k-loop,tb-assign-latencies,tb-schedule-loops,tb-derive-waits,tb-expand-pipeline,tb-cleanup-pipeline,tb-verify-pipeline-mainline-contract)' | FileCheck %s

module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64} {
  func.func @kernel(%A: memref<64x32xf16>, %B: memref<32x64xf16>,
                    %C: memref<64x64xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = true}
      : memref<64x32xf16>, memref<32x64xf16>, memref<64x64xf32>
    func.return
  }
}

// CHECK: tb.pipeline_mainline
// CHECK: #tb.shared<order = [1, 0], perPhase = 4, maxPhase = 2, transposed = false, swizzlingByteWidth = 16
// CHECK: kind = "register_pack_global_vector"
// CHECK: required_sync_kind = "none"
