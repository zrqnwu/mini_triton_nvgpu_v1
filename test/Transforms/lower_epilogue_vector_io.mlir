// RUN: /home/zhangruiqi/mini_triton_nvgpu_v1/build/bin/tb-opt --tb-lower-epilogue-vector-io %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @predicated_vector_io
  // CHECK: arith.extui
  // CHECK: nvvm.inline_ptx
  // CHECK: nvvm.inline_ptx
  // CHECK-NOT: vector.maskedload
  // CHECK-NOT: vector.maskedstore
  func.func @predicated_vector_io(%src: memref<160x96xf32>,
                                  %dst: memref<160x96xf32>,
                                  %row: index, %col: index,
                                  %value: vector<4xf32>) {
    %loaded = tb.epilogue_global_vector_load %src[%row, %col]
      {boundary_aware = true, scalar_tail = false, vector_width = 4 : i64}
      : memref<160x96xf32> -> vector<4xf32>
    tb.epilogue_global_vector_store %value, %dst[%row, %col]
      {boundary_aware = true, scalar_tail = false, vector_width = 4 : i64}
      : vector<4xf32>, memref<160x96xf32>
    func.return
  }

  // CHECK-LABEL: func.func @scalar_tail_vector_io
  // CHECK: vector.maskedload
  // CHECK: vector.maskedstore
  func.func @scalar_tail_vector_io(%src: memref<161x98xf32>,
                                   %dst: memref<161x98xf32>,
                                   %row: index, %col: index,
                                   %value: vector<4xf32>) {
    %loaded = tb.epilogue_global_vector_load %src[%row, %col]
      {boundary_aware = true, scalar_tail = true, vector_width = 4 : i64}
      : memref<161x98xf32> -> vector<4xf32>
    tb.epilogue_global_vector_store %value, %dst[%row, %col]
      {boundary_aware = true, scalar_tail = true, vector_width = 4 : i64}
      : vector<4xf32>, memref<161x98xf32>
    func.return
  }
}
