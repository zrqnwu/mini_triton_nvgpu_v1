#ifndef TB_V1_DIALECT
#define TB_V1_DIALECT

include "mlir/IR/OpBase.td"

def TB_Dialect : Dialect {
  let name = "tb";
  let summary = "Mini Triton V1 dialect for strict shape-parity matmul";
  let description = [{
    This dialect holds a narrow matmul-oriented IR used to prototype a
    Triton-inspired pipeline on top of MLIR.
  }];
  let cppNamespace = "::mlir::tb";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();

    ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type type) const override;
    void printAttribute(::mlir::Attribute attr,
                        ::mlir::DialectAsmPrinter &printer) const override;

    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

class TB_Op<string mnemonic, list<Trait> traits = []> :
    Op<TB_Dialect, mnemonic, traits>;

#endif // TB_V1_DIALECT
