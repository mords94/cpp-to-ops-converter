
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace cl = llvm::cl;

int parseMlir(mlir::MLIRContext &context,
              mlir::OwningOpRef<mlir::ModuleOp> &module,
              llvm::StringRef inputFilename) {

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

// init a map that stores lines associated with filename

llvm::StringMap<std::vector<std::string>> sourceFiles;

static void readSourceFile(llvm::StringRef filename) {
  std::ifstream file(filename.str());
  std::vector<std::string> lines;

  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  sourceFiles[filename] = lines;
}

std::vector<std::string> *getSource(llvm::StringRef filename) {
  if (sourceFiles.count(filename) == 0) {
    readSourceFile(filename);
  }

  return &sourceFiles[filename];
}
