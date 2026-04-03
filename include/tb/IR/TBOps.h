#ifndef MINI_TRITON_TB_OPS_H
#define MINI_TRITON_TB_OPS_H

#include "tb/IR/TBDialect.h"
#include "tb/IR/TBAttrs.h"
#include "tb/IR/TBTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tb/IR/TBOps.h.inc"

#endif // MINI_TRITON_TB_OPS_H
