#ifndef TB_V1_ATTRDEFS
#define TB_V1_ATTRDEFS

include "tb/IR/TBDialect.td"
include "mlir/IR/AttrTypeBase.td"

class TB_Attr<string name, list<Trait> traits = []>
    : AttrDef<TB_Dialect, name, traits>;

def TB_SemanticLayoutAttr : TB_Attr<"SemanticLayout"> {
  let mnemonic = "semantic_layout";
  let genVerifyDecl = 1;
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$order,
    ArrayRefParameter<"int64_t">:$strides,
    "bool":$zeroOffset,
    "bool":$hasStaticStrides,
    "bool":$unitStrideOnMinor,
    "bool":$contiguous
  );
  let assemblyFormat = [{
    `<`
      `order` `=` `[` $order `]` `,`
      `strides` `=` `[` $strides `]` `,`
      `zeroOffset` `=` $zeroOffset `,`
      `hasStaticStrides` `=` $hasStaticStrides `,`
      `unitStrideOnMinor` `=` $unitStrideOnMinor `,`
      `contiguous` `=` $contiguous
    `>`
  }];
}

def TB_CGAEncodingAttr : TB_Attr<"CGAEncoding"> {
  let mnemonic = "cga";
  let genVerifyDecl = 1;
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$ctasPerCGA,
    ArrayRefParameter<"int64_t">:$ctaSplitNum,
    ArrayRefParameter<"int64_t">:$ctaOrder
  );
  let assemblyFormat = [{
    `<`
      `ctasPerCGA` `=` `[` $ctasPerCGA `]` `,`
      `ctaSplitNum` `=` `[` $ctaSplitNum `]` `,`
      `ctaOrder` `=` `[` $ctaOrder `]`
    `>`
  }];
}

def TB_BlockedEncodingAttr : TB_Attr<"BlockedEncoding"> {
  let mnemonic = "blocked";
  let genVerifyDecl = 1;
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$sizePerThread,
    ArrayRefParameter<"int64_t">:$threadsPerWarp,
    ArrayRefParameter<"int64_t">:$warpsPerCTA,
    ArrayRefParameter<"int64_t">:$order,
    TB_CGAEncodingAttr:$cga
  );
  let assemblyFormat = [{
    `<`
      `sizePerThread` `=` `[` $sizePerThread `]` `,`
      `threadsPerWarp` `=` `[` $threadsPerWarp `]` `,`
      `warpsPerCTA` `=` `[` $warpsPerCTA `]` `,`
      `order` `=` `[` $order `]` `,`
      `cga` `=` qualified($cga)
    `>`
  }];
}

def TB_SharedEncodingAttr : TB_Attr<"SharedEncoding"> {
  let mnemonic = "shared";
  let genVerifyDecl = 1;
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$order,
    "int64_t":$perPhase,
    "int64_t":$maxPhase,
    OptionalArrayRefParameter<"int64_t">:$paddingIntervals,
    OptionalArrayRefParameter<"int64_t">:$paddings,
    "bool":$transposed,
    "int64_t":$swizzlingByteWidth,
    TB_CGAEncodingAttr:$cga
  );
  let assemblyFormat = [{
    `<`
      `order` `=` `[` $order `]` `,`
      `perPhase` `=` $perPhase `,`
      `maxPhase` `=` $maxPhase `,`
      `transposed` `=` $transposed `,`
      `swizzlingByteWidth` `=` $swizzlingByteWidth `,`
      `cga` `=` qualified($cga)
      (`,` struct($paddingIntervals, $paddings)^)?
    `>`
  }];
}

def TB_NVGPUMmaEncodingAttr : TB_Attr<"NVGPUMmaEncoding"> {
  let mnemonic = "nvgpu_mma";
  let genVerifyDecl = 1;
  let parameters = (ins
    StringRefParameter<>:$mmaFamily,
    "int64_t":$versionMajor,
    "int64_t":$versionMinor,
    ArrayRefParameter<"int64_t">:$warpsPerCTA,
    ArrayRefParameter<"int64_t">:$instrShape,
    TB_CGAEncodingAttr:$cga
  );
  let assemblyFormat = [{
    `<`
      `mmaFamily` `=` $mmaFamily `,`
      `versionMajor` `=` $versionMajor `,`
      `versionMinor` `=` $versionMinor `,`
      `warpsPerCTA` `=` `[` $warpsPerCTA `]` `,`
      `instrShape` `=` `[` $instrShape `]` `,`
      `cga` `=` qualified($cga)
    `>`
  }];
}

def TB_DotOperandEncodingAttr : TB_Attr<"DotOperandEncoding"> {
  let mnemonic = "dot_operand";
  let genVerifyDecl = 1;
  let parameters = (ins
    "int64_t":$operandIndex,
    TB_NVGPUMmaEncodingAttr:$parent,
    "int64_t":$kWidth
  );
  let assemblyFormat = [{
    `<`
      `operandIndex` `=` $operandIndex `,`
      `parent` `=` qualified($parent) `,`
      `kWidth` `=` $kWidth
    `>`
  }];
}

def TB_AccumulatorEncodingAttr : TB_Attr<"AccumulatorEncoding"> {
  let mnemonic = "accumulator";
  let parameters = (ins
    TB_NVGPUMmaEncodingAttr:$parent,
    ArrayRefParameter<"int64_t">:$logicalShape,
    ArrayRefParameter<"int64_t">:$instructionShape,
    ArrayRefParameter<"int64_t">:$repeatShape,
    ArrayRefParameter<"int64_t">:$repeatOrder
  );
  let assemblyFormat = [{
    `<`
      `parent` `=` qualified($parent) `,`
      `logicalShape` `=` `[` $logicalShape `]` `,`
      `instructionShape` `=` `[` $instructionShape `]` `,`
      `repeatShape` `=` `[` $repeatShape `]` `,`
      `repeatOrder` `=` `[` $repeatOrder `]`
    `>`
  }];
}

#endif // TB_V1_ATTRDEFS
