module attributes {"tb.num-warps" = 4 : i64, "tb.requested-stages" = 2 : i64} {
  func.func @kernel(%A: memref<192x32xf16>, %B: memref<32x128xf16>,
                    %C: memref<192x128xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = true,
                          group_m = 2 : i64}
      : memref<192x32xf16>, memref<32x128xf16>, memref<192x128xf32>
    func.return
  }
}
