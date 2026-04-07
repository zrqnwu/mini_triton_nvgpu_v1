module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64} {
  func.func @kernel(%A: memref<96x32xf16>, %B: memref<32x80xf16>,
                    %C: memref<96x80xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = false}
      : memref<96x32xf16>, memref<32x80xf16>, memref<96x80xf32>
    func.return
  }
}
