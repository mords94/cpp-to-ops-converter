

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"

#ifndef OPS_CONVERTER_HELPERS_H
#define OPS_CONVERTER_HELPERS_H

int parseMlir(mlir::MLIRContext &context,
              mlir::OwningOpRef<mlir::ModuleOp> &module,
              llvm::StringRef inputFilename);

#endif