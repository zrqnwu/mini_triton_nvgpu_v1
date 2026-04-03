#cga = #tb.cga<ctasPerCGA = [1, 1, 1], ctaSplitNum = [1, 1, 1], ctaOrder = [0, 1, 2]>
#blocked_row = #tb.blocked<sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], cga = #tb.cga<ctasPerCGA = [1, 1, 1], ctaSplitNum = [1, 1, 1], ctaOrder = [0, 1, 2]>>
#blocked_col = #tb.blocked<sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1], cga = #tb.cga<ctasPerCGA = [1, 1, 1], ctaSplitNum = [1, 1, 1], ctaOrder = [0, 1, 2]>>

module {
  func.func @layout_convert_roundtrip(
      %src: !tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_row, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]>)
      -> !tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_row, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]> {
    %0 = tb.convert_layout %src : (!tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_row, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]>) -> !tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_col, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]>
    %1 = tb.convert_layout %0 : (!tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_col, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]>) -> !tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_row, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]>
    return %1 : !tb.memdesc<shape = [64, 32], elementType = f16, encoding = #blocked_row, memorySpace = "global", mutableMemory = true, allocShape = [64, 32]>
  }
}
