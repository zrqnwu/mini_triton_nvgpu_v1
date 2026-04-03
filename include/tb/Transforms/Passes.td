#ifndef MINI_TRITON_TB_TRANSFORMS_PASSES
#define MINI_TRITON_TB_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def TBVerifyScope : Pass<"tb-verify-scope", "::mlir::ModuleOp"> {
  let summary = "Verify that each tb.matmul falls inside the strict V1 scope";
  let description = [{
    Checks that every `tb.matmul` satisfies the current stage1 legality
    envelope: NVIDIA `sm80/sm86`, static rank-2 `fp16xfp16->fp32`, 
    `mma.sync m16n8k16`, tile shapes divisible by the instruction shape,
    and module execution requests (`tb.num-warps` / `tb.requested-stages`)
    inside the supported single-CTA matmul range.
  }];
}

def TBAttachTargetInfo : Pass<"tb-attach-target-info", "::mlir::ModuleOp"> {
  let summary = "Attach the explicit target contract consumed by stage1";
  let description = [{
    Materializes the module-level target attrs (`tb.target` /
    `tb.threads-per-warp`) from the outer pipeline target options and validates
    that the user-provided execution requests (`tb.num-warps` /
    `tb.requested-stages`) already exist as explicit module owner truth.
    Legacy per-op `tb.target_info` survives only as a parse fallback.
  }];
  let options = [
    Option<"gpuArch", "gpu-arch", "std::string", "\"\"",
           "GPU chip name such as sm_86">,
    Option<"ptxFeatures", "ptx-features", "std::string", "\"\"",
           "PTX feature string attached to the target contract">
  ];
}

def TBSemanticizeMatmul
    : Pass<"tb-semanticize-matmul", "::mlir::ModuleOp"> {
  let summary = "Attach TTGIR-style semantic memdesc truth to tb.matmul";
  let description = [{
    Builds `tb.semantic_matmul` from `tb.matmul` and `KernelConfig`. This pass
    is the strict TTIR->TTGIR semanticization boundary: global A/B/C semantic
    layout truth becomes visible in IR before later layout / fragment /
    transport planning. Physical warp/CTA decomposition is no longer decided
    here.
  }];
}

def TBBuildProgramMappingPlan
    : Pass<"tb-build-program-mapping-plan", "::mlir::ModuleOp"> {
  let summary = "Derive the CTA/program mapping contract for tb.matmul";
  let description = [{
    Builds `tb.program_mapping_plan` from `tb.matmul`, `KernelConfig` and
    `TargetInfo`. This contract owns program-to-tile mapping, launch order,
    grouped launch / split-k / persistent placeholders, but does not own CTA
    internal scheduling.
  }];
}

def TBBuildLayoutPlan : Pass<"tb-build-layout-plan", "::mlir::ModuleOp"> {
  let summary = "Derive the encoding contract for tb.matmul";
  let description = [{
    Compatibility pass name retained for existing scripts.
    Builds `tb.encoding_plan` from `tb.matmul`, `tb.semantic_matmul` and
    `tb.program_mapping_plan`. This is the first Triton-aligned physical
    contract layer: blocked/shared/dot/accumulator encodings plus the exact
    fragment decomposition consumed by all later plans.
  }];
}

def TBBuildTransportPlan
    : Pass<"tb-build-transport-plan", "::mlir::ModuleOp"> {
  let summary = "Derive the explicit global/shared transport contract";
  let description = [{
    Builds `tb.transport_plan` from `tb.encoding_plan` and `TargetInfo`.
    Transport legality, vector widths and async eligibility live here instead
    of leaking through `EncodingPlan::SharedEncodingSpec`.
  }];
}

def TBCleanupLayoutConversions
    : Pass<"tb-cleanup-layout-conversions", "::mlir::ModuleOp"> {
  let summary = "Prune redundant tb.convert_layout round-trips";
  let description = [{
    Performs the minimal layout-conversion cleanup needed by the owner-truth
    rebuild: identity conversions and immediate round-trips are removed from
    IR, leaving only the layout boundaries that still carry distinct encoding
    truth. This is the first explicit cleanup hook that later Triton-style
    layout propagation can build on.
  }];
}

def TBBuildCRegisterPlan : Pass<"tb-build-c-register-plan", "::mlir::ModuleOp"> {
  let summary = "Derive accumulator and epilogue contracts for C";
  let description = [{
    Compatibility pass name retained for existing scripts.
    Builds `tb.accumulator_plan` and `tb.epilogue_plan` from `tb.matmul` and
    `tb.encoding_plan`. The pass splits the old flat C landing contract into
    an accumulator register-layout contract and an epilogue init/store contract.
  }];
}

def TBRewriteMatmulMainloop
    : Pass<"tb-rewrite-matmul-mainloop", "::mlir::ModuleOp"> {
  let summary = "Rewrite tb.matmul into a tensor-core fragment mainloop contract";
  let description = [{
    Builds `tb.matmul_rewrite` from `tb.matmul`, `tb.encoding_plan`,
    `tb.accumulator_plan` and `tb.epilogue_plan`. This is the strict
    matmul-specific rewrite boundary: fragment ownership, instruction geometry,
    operand paths and direct epilogue routing are fixed here instead of in
    loop regularization or lowering.
  }];
}

def TBBuildMainloopGraph : Pass<"tb-build-mainloop-graph", "::mlir::ModuleOp"> {
  let summary = "Construct the buffer model for tb.matmul";
  let description = [{
    Builds `tb.buffer_model` from `tb.matmul`, `tb.encoding_plan`,
    `tb.accumulator_plan` and `tb.epilogue_plan`. The pass materializes the
    Triton-aligned resource graph boundary: backings, views, values and ops.
  }];
}

def TBRegularizeKLoop : Pass<"tb-regularize-k-loop", "::mlir::ModuleOp"> {
  let summary = "Regularize the strict single K mainloop before pipelining";
  let description = [{
    Builds `tb.loop_plan` from `tb.matmul` and `tb.buffer_model`. This is the
    explicit Pass 4 boundary from the execution doc: the stage1 matmul body is
    regularized into one public K-loop owner before latency assignment,
    scheduling and pipeline expansion.
  }];
}

def TBAssignLatencies : Pass<"tb-assign-latencies", "::mlir::ModuleOp"> {
  let summary = "Assign latency contracts from the pure dataflow graph";
  let description = [{
    Builds `tb.latency_plan` from `tb.matmul`, `tb.buffer_model` and
    `tb.loop_plan`. `tb.loop_plan` is the public loop-regularization owner;
    the pass no longer infers the single main K-loop directly from the raw
    buffer graph.
  }];
}

def TBScheduleLoops : Pass<"tb-schedule-loops", "::mlir::ModuleOp"> {
  let summary = "Build the coarse latency-driven schedule for tb.matmul";
  let description = [{
    Builds `tb.pipeline_plan` from `tb.matmul`, `tb.buffer_model`,
    `tb.loop_plan` and `tb.latency_plan`. The pass consumes the public regular
    K-loop owner instead of reconstructing the loop body shape from the raw
    graph during scheduling.
  }];
}

def TBDeriveWaits : Pass<"tb-derive-waits", "::mlir::ModuleOp"> {
  let summary = "Derive first-use waits and slot overwrite frontiers";
  let description = [{
    Builds `tb.async_plan` from `tb.matmul`, `tb.buffer_model`,
    `tb.latency_plan` and `tb.pipeline_plan`. The pass directly derives the
    producer/group/wait/reuse-fence contract and no longer routes through the
    legacy wait plan.
  }];
}

def TBExpandPipeline : Pass<"tb-expand-pipeline", "::mlir::ModuleOp"> {
  let summary = "Build the explicit pipeline-expanded mainline contract";
  let description = [{
    Builds `tb.pipeline_expansion` from `tb.matmul`, `tb.buffer_model`,
    `tb.pipeline_plan` and `tb.async_plan`. This is the strict Pass 7 boundary:
    stage/cluster order, explicit wait ownership and the expanded mainline
    execution clusters become visible in IR before cleanup and target lowering.
  }];
}

def TBCleanupPipeline
    : Pass<"tb-cleanup-pipeline", "::mlir::ModuleOp"> {
  let summary = "Validate and seal the strict async tensor-core mainline";
  let description = [{
    Performs the post-pipeline cleanup boundary from the execution doc.
    The pass validates that `tb.pipeline_expansion` already describes the true
    exact-tile async mainline, materializes the explicit `tb.pipeline_mainline`
    IR owner as a stage/cluster-ordered async/wait/mma cluster stream, removes
    temporary rewrite-only attrs, and emits a compact `tb.pipeline_ready`
    summary for lowering.
  }];
}

def TBLowerPipelineToNVGPU
    : Pass<"tb-lower-pipeline-to-nvgpu", "::mlir::ModuleOp"> {
  let summary = "Lower the strict V1 pipeline contract to gpu/nvgpu IR";
  let description = [{
    Consumes the explicit `tb.pipeline_mainline` IR plus
    `tb.program_mapping_plan`, `tb.encoding_plan`, `tb.transport_plan`,
    `tb.buffer_model`, `tb.async_plan`, `tb.accumulator_plan` and
    `tb.epilogue_plan`.
    The lowering reads the public contracts plus the ordered explicit pipeline
    cluster stream directly and does not rebuild any hidden pipeline expansion
    or synthetic loop skeleton inside the NVGPU executor.
  }];
}

def TBLowerEpilogueVectorIO
    : Pass<"tb-lower-epilogue-vector-io", "::mlir::ModuleOp"> {
  let summary = "Lower late epilogue global vector IO owners to stable memref vector accesses";
  let description = [{
    Rewrites `tb.epilogue_global_vector_{load,store}` into vector-preserving
    memref views and vector-element loads/stores before the generic NVVM sink.
    This pass is the late target-owned C epilogue boundary that prevents the
    final direct-global vector IO from being silently scalarized.
  }];
}

#endif // MINI_TRITON_TB_TRANSFORMS_PASSES
