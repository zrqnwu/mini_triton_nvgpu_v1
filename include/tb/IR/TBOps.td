#ifndef TB_V1_OPS
#define TB_V1_OPS

include "tb/IR/TBDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TB_ConvertLayoutOp : TB_Op<"convert_layout", [Pure]> {
  let summary = "Explicit layout-conversion currency between tb encodings";
  let description = [{
    `tb.convert_layout` is the value-level layout owner boundary introduced by
    the owner-truth rebuild. It keeps layout conversion explicit in IR instead
    of letting later passes rebuild hidden encoding changes from side-car
    analysis structs.
  }];

  let arguments = (ins
    AnyType:$source
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $source attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def TB_MatmulOp : TB_Op<"matmul"> {
  let summary = "High-level stage1 matmul op for V1";
  let description = [{
    `tb.matmul` is the single high-level kernel op in V1. It now owns only
    problem/tile semantics plus grouped launch hints. Execution context
    (`tb.num-warps`, `tb.requested-stages`, target ownership) lives on module
    attrs and the physical mma/layout truth is derived later by the layout and
    transport planning passes.
  }];

  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$b,
    AnyMemRef:$c,
    I64Attr:$block_m,
    I64Attr:$block_n,
    I64Attr:$block_k,
    DefaultValuedAttr<I64Attr, "8">:$group_m,
    BoolAttr:$exact_tile
  );

  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c)
  }];

  let hasVerifier = 1;
}

def TB_PipelineMainlineOp : TB_Op<"pipeline_mainline", [NoTerminator]> {
  let summary = "Explicit stage1 pipeline mainline owner";
  let description = [{
    `tb.pipeline_mainline` is the post-cleanup, pre-lowering owner of the
    strict async tensor-core matmul mainline. It replaces `tb.matmul` at
    the Pass 8 boundary and carries only the explicit stage/cluster-ordered
    async-issue / consumer-wait / mma-compute mainline IR consumed by the
    NVGPU lowering. Kernel configuration owner truth must already live in
    module attrs plus attached semantic/plan attrs by the time this op exists.
  }];

  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$b,
    AnyMemRef:$c
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) $body
  }];

  let hasVerifier = 1;
}

def TB_AsyncIssueClusterOp : TB_Op<"async_issue_cluster"> {
  let summary = "Explicit async-issue pipeline cluster";
  let arguments = (ins
    I64Attr:$ordinal,
    I64Attr:$stage,
    I64Attr:$cluster,
    I64Attr:$k_group,
    DenseI64ArrayAttr:$op_ids,
    StrAttr:$reason
  );
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
}

def TB_ConsumerWaitClusterOp : TB_Op<"consumer_wait_cluster"> {
  let summary = "Explicit consumer-wait pipeline cluster";
  let arguments = (ins
    I64Attr:$ordinal,
    I64Attr:$stage,
    I64Attr:$cluster,
    I64Attr:$k_group,
    DenseI64ArrayAttr:$op_ids,
    DenseI64ArrayAttr:$wait_group_ids,
    BoolAttr:$needs_barrier,
    StrAttr:$reason
  );
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
}

def TB_MmaComputeClusterOp : TB_Op<"mma_compute_cluster"> {
  let summary = "Explicit MMA-compute pipeline cluster";
  let arguments = (ins
    I64Attr:$ordinal,
    I64Attr:$stage,
    I64Attr:$cluster,
    I64Attr:$k_group,
    DenseI64ArrayAttr:$op_ids,
    StrAttr:$reason
  );
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
}

def TB_EpilogueGlobalVectorLoadOp
    : TB_Op<"epilogue_global_vector_load", [Pure]> {
  let summary = "Late target-owned global vector load for direct epilogues";
  let description = [{
    `tb.epilogue_global_vector_load` is the late direct-C target owner between
    `tb-lower-pipeline-to-nvgpu` and the final NVVM sink. It represents one
    row-vector load from global memory that must survive to the backend as a
    vector access instead of being silently scalarized by generic lowering.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Index:$row,
    Index:$col,
    I64Attr:$vector_width,
    BoolAttr:$boundary_aware
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $source `[` $row `,` $col `]` attr-dict `:` type($source) `->` type($result)
  }];

  let hasVerifier = 1;
}

def TB_EpilogueGlobalVectorStoreOp : TB_Op<"epilogue_global_vector_store"> {
  let summary = "Late target-owned global vector store for direct epilogues";
  let description = [{
    `tb.epilogue_global_vector_store` is the write-side counterpart of
    `tb.epilogue_global_vector_load`. It makes the final direct-global vector
    landing an explicit late owner instead of a best-effort `vector.store`.
  }];

  let arguments = (ins
    AnyType:$value,
    AnyMemRef:$dest,
    Index:$row,
    Index:$col,
    I64Attr:$vector_width,
    BoolAttr:$boundary_aware
  );

  let assemblyFormat = [{
    $value `,` $dest `[` $row `,` $col `]` attr-dict `:` type($value) `,` type($dest)
  }];

  let hasVerifier = 1;
}

#endif // TB_V1_OPS
