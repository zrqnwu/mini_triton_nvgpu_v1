module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64} {
  func.func @kernel(%A: memref<64x32xf16>, %B: memref<32x64xf16>,
                    %C: memref<64x64xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = true}
      : memref<64x32xf16>, memref<32x64xf16>, memref<64x64xf32>
    func.return
  }
}
