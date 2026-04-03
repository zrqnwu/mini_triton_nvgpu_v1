#ifndef TB_V1_TYPEDEFS
#define TB_V1_TYPEDEFS

include "tb/IR/TBDialect.td"
include "tb/IR/TBAttrDefs.td"
include "mlir/IR/AttrTypeBase.td"

class TB_Type<string name, list<Trait> traits = []>
    : TypeDef<TB_Dialect, name, traits>;

def TB_MemDescType : TB_Type<"MemDesc"> {
  let mnemonic = "memdesc";
  let genVerifyDecl = 1;
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType,
    "::mlir::Attribute":$encoding,
    StringRefParameter<>:$memorySpace,
    "bool":$mutableMemory,
    ArrayRefParameter<"int64_t">:$allocShape
  );
  let assemblyFormat = [{
    `<`
      `shape` `=` `[` $shape `]` `,`
      `elementType` `=` $elementType `,`
      `encoding` `=` $encoding `,`
      `memorySpace` `=` $memorySpace `,`
      `mutableMemory` `=` $mutableMemory `,`
      `allocShape` `=` `[` $allocShape `]`
    `>`
  }];
  let extraClassDeclaration = [{
    bool isGlobalMemory() const { return getMemorySpace() == "global"; }
    bool isSharedMemory() const { return getMemorySpace() == "shared"; }
    bool isRegisterMemory() const { return getMemorySpace() == "registers"; }
  }];
}

#endif // TB_V1_TYPEDEFS
