add_mlir_dialect(TBOps tb)

set(LLVM_TARGET_DEFINITIONS TBAttrDefs.td)
mlir_tablegen(TBAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tb)
mlir_tablegen(TBAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tb)
add_public_tablegen_target(MLIRTBAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS TBTypeDefs.td)
mlir_tablegen(TBTypes.h.inc -gen-typedef-decls -typedefs-dialect=tb)
mlir_tablegen(TBTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tb)
add_public_tablegen_target(MLIRTBTypeDefsIncGen)
