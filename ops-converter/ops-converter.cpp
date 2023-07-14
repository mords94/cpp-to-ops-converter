//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Lib/ExpressionTree.h"
#include "Lib/helpers.h"

#include <algorithm>
#include <string>
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<kernel mlir file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::opt<std::string> originalSource(cl::Positional,
                                           cl::desc("<original source file>"),
                                           cl::value_desc("path"),
                                           cl::Required);

static cl::opt<std::string> functionName("function", cl::desc("Function name"),
                                         cl::value_desc("string"),
                                         cl::Required);

static cl::opt<bool> enableDebug("opconv-debug", cl::desc("Enable debug"),
                                 cl::init(false));

static cl::opt<bool> enableDebugModule("opconv-debug-module",
                                       cl::desc("Print module after labelings"),
                                       cl::init(false));

static cl::opt<bool>
    enableDetailedPrint("opconv-detailed",
                        cl::desc("Add loop and call orders for dag"),
                        cl::init(false));

static cl::opt<bool> emitMLIR("opconv-emit-mlir", cl::desc("Emit MLIR"),
                              cl::init(false));

// container for the source code
std::vector<std::string> sourceLines;

bool isWhileLoop(mlir::Operation *op) { return isa<mlir::scf::WhileOp>(op); }
bool isForLoop(mlir::Operation *op) { return isa<mlir::scf::ForOp>(op); }
bool isLoop(mlir::Operation *op) { return isWhileLoop(op) || isForLoop(op); }

/**
 * @brief Check if value has a use of type T
 * @param value The value to check
 */
template <typename T> bool has_use(mlir::Value value) {
  return llvm::any_of(value.getUses(), [&](mlir::OpOperand &op) {
    return isa<T>(op.getOwner());
  });
};

/**
 * @brief Returns an iterator for all uses of the value that are of type T
 * @param value The value to check
 */
template <typename T> auto get_use_range(mlir::Value value) {
  return llvm::make_filter_range(value.getUses(), [&](mlir::OpOperand &op) {
    return isa<T>(op.getOwner());
  });
};

/**
 * @brief Returns all uses of the value that are of type T
 * @param value The value to check
 */
template <typename T> llvm::SmallVector<T> get_use(mlir::Value value) {
  llvm::SmallVector<T> uses;

  for (auto &op : value.getUses()) {
    if (isa<T>(op.getOwner())) {
      uses.push_back(cast<T>(op.getOwner()));
    }
  }

  return uses;
};

int toInt(llvm::StringRef str) {
  int value;
  str.getAsInteger(10, value);
  return value;
}

bool isFunctionArgument(mlir::Value value) {
  if (value.isa<mlir::BlockArgument>()) {
    auto blockArg = value.dyn_cast<mlir::BlockArgument>();
    return isa<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
  }
  return false;
}

std::string getInductionVariableFromStatement(std::string forStmt) {

  std::vector<std::string> forStmtParts;
  std::string forStmtPart;
  std::istringstream forStmtStream(forStmt);
  while (std::getline(forStmtStream, forStmtPart, ';')) {
    forStmtParts.push_back(forStmtPart);
  }

  auto condition = forStmtParts[1];

  std::string inductionVariable = "";
  if (condition.find(">") != std::string::npos) {
    inductionVariable = condition.substr(0, condition.find(">"));
  } else if (condition.find("<") != std::string::npos) {
    inductionVariable = condition.substr(0, condition.find("<"));
  }

  inductionVariable.erase(std::remove_if(inductionVariable.begin(),
                                         inductionVariable.end(), isspace),
                          inductionVariable.end());

  return inductionVariable;
}

std::string getInductionVariableFromLoop(mlir::Operation *op) {
  auto loc = op->getLoc().dyn_cast<mlir::FileLineColLoc>();
  auto count = loc.getLine() - 1;
  std::string stmt = sourceLines[count];

  return getInductionVariableFromStatement(stmt);
}

std::string getStrippedFileNameFromFileLineColLoc(mlir::FileLineColLoc loc) {
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    auto fileName = fileLineColLoc.getFilename();
    auto strippedFileName =
        fileName.str().substr(fileName.str().find_last_of("/") + 1);

    return strippedFileName;
  }
  return "";
}

llvm::json::Object getLoopAttrObject(mlir::Location loc) {
  llvm::json::Object attr;
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    llvm::json::Object location;
    std::string fileName =
        getStrippedFileNameFromFileLineColLoc(fileLineColLoc);

    location["file"] = llvm::json::Value(fileName);
    location["line"] = llvm::json::Value(fileLineColLoc.getLine());
    location["col"] = llvm::json::Value(fileLineColLoc.getColumn());
    location["source"] =
        llvm::json::Value(sourceLines[fileLineColLoc.getLine() - 1]);
    attr["induction_var"] = llvm::json::Value(getInductionVariableFromStatement(
        sourceLines[fileLineColLoc.getLine() - 1]));

    attr["location"] = llvm::json::Value(std::move(location));
  }

  return attr;
}

llvm::json::Object getArgLocation(mlir::Location loc) {
  llvm::json::Object attr;
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    llvm::json::Object location;
    std::string fileName =
        getStrippedFileNameFromFileLineColLoc(fileLineColLoc);

    location["file"] = llvm::json::Value(fileName);
    location["line"] = llvm::json::Value(fileLineColLoc.getLine());
    location["col"] = llvm::json::Value(fileLineColLoc.getColumn());
    location["associated_line"] =
        llvm::json::Value(sourceLines[fileLineColLoc.getLine() - 1]);

    attr["location"] = llvm::json::Value(std::move(location));
  }

  return attr;
}

/**
 * @brief Recursively builds an Expression tree for the given value
 * @param node Root node of the expression tree
 * @param value The value to build the tree for
 * @param depth The depth of the value in the tree
 *
 */
void buildTree(Node &node, mlir::Value value, int depth = 0) {
  node.setDepth(depth);

  llvm::errs() << "   [buildTree] Building tree for value " << value << "\n";

  if (auto op = value.getDefiningOp()) {
    llvm::errs() << "   [buildTree] Bulding a new node with value on level "
                 << depth << "\n";

    if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(op)) {
      llvm::errs() << "   [buildTree] Load operation \n";

      // global memref
      node.setValue(loadOp.getMemRef());
    } else if (auto constantIntOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      llvm::errs() << "   [buildTree] Constant offset operation \n";

      // index offset
      node.setValue(constantIntOp.getResult());
    } else if (auto indexCastOp = dyn_cast<mlir::arith::IndexCastOp>(op)) {
      llvm::errs() << "   [buildTree] Index cast operation \n";

      // Node *children[1] = {new Node()};

      // node.setLeft(children[0]);
      // node.setValue(indexCastOp.getIn());

      buildTree(node, indexCastOp.getIn(), depth);

    } else if (op->getOperands().size() == 2) {
      llvm::errs() << "   [buildTree] Binary operation \n";
      // binary (addi or muli) operation
      Node *children[2] = {new Node(), new Node()};

      node.setLeft(children[0]);
      node.setRight(children[1]);
      node.setValue(value);
      for (auto en : llvm::enumerate(value.getDefiningOp()->getOperands())) {
        buildTree(*children[en.index()], en.value(), depth + 1);
      }
    } else {
      op->emitOpError("   [buildTree] Unknown operation while building memref "
                      "indexing expression tree");
    }
  } else if (auto blockArgument = dyn_cast<mlir::BlockArgument>(value)) {
    llvm::errs()
        << "   [buildTree] Bulding a new node with a block argument on level "
        << depth << "\n";

    // FIXME: finish rewrite pattern for this
    auto parent = blockArgument.getOwner()->getParentOp();

    if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(parent)) {
      auto conditionOperand =
          whileOp.getConditionOp().getArgs()[blockArgument.getArgNumber()];

      if (conditionOperand.isa<mlir::BlockArgument>()) {
        node.setOwnerLoop(blockArgument.getOwner()->getParentOp());
        node.setValue(value);
      } else {
        if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(
                conditionOperand.getDefiningOp())) {
          node.setValue(loadOp.getMemRef());
        }
      }
    }
  }
};

/**
 * @brief Counts the maximum depth of nested loops within the given loop
 * operation
 * @param loopOp The loop operation to check
 * @param depth The current depth
 */
auto countLoopNestDepth(mlir::Operation *op, int64_t depth = 1) {
  auto max = depth;

  op->walk([&max](mlir::Operation *childOp) {
    if (!isLoop(childOp)) {
      return mlir::WalkResult::skip();
    }

    auto nextDepth =
        childOp->getAttr("depth").cast<mlir::IntegerAttr>().getInt();

    if (nextDepth > max) {
      max = nextDepth;
    }

    return mlir::WalkResult::advance();
  });

  return max;
}

auto getLoopsOnLevel(mlir::Operation *op, int64_t level) {
  mlir::SmallVector<mlir::Operation *> loops;
  op->walk([&](mlir::Operation *childOp) {
    if (!isLoop(childOp)) {
      return;
    }

    if (level == childOp->getAttr("depth").cast<mlir::IntegerAttr>().getInt()) {
      loops.push_back(childOp);
    }
  });

  return loops;
}

llvm::StringMap<mlir::BlockArgument>
getFuncArgumentsUsedInLoop(mlir::Operation *loop) {
  llvm::StringMap<mlir::BlockArgument> argumentMap;

  loop->getRegions().back().walk([&](mlir::Operation *op) {
    if (isa<mlir::memref::LoadOp>(op) || isa<mlir::memref::StoreOp>(op)) {
      llvm::errs()
          << "   [getFuncArgumentsUsedInLoop] Found load or store op \n";
      mlir::Value memref =
          op->getOperand(isa<mlir::memref::LoadOp>(op) ? 0 : 1);

      if (memref.isa<mlir::BlockArgument>()) {
        llvm::errs() << "   [getFuncArgumentsUsedInLoop] " << op->getName()
                     << " block argument \n";
        mlir::BlockArgument arg = memref.cast<mlir::BlockArgument>();
        argumentMap[Twine(arg.getArgNumber()).str()] = arg;
      } else {
        llvm::errs() << "   [getFuncArgumentsUsedInLoop] " << op->getName()
                     << " memref \n";
        memref.dump();
      }
    } else {

      const bool hasFunctionArgumentAsOperand =
          llvm::any_of(op->getOperands(), [&](mlir::Value value) {
            return isFunctionArgument(value);
          });

      if (hasFunctionArgumentAsOperand) {
        op->emitError("   [getFuncArgumentsUsedInLoop] Argument is used by an "
                      "unsupported operation " +
                      op->getName().getStringRef());
      }
    }
  });

  return argumentMap;
}

mlir::Operation *getParentLoopOperation(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();

  while (parent != nullptr) {
    if (isLoop(parent)) {
      return parent;
    }

    parent = parent->getParentOp();
  }

  return nullptr;
}

void dummy() { mlir::Value a = nullptr; }

template <typename T>
llvm::SmallVector<T> filterStoreUses(llvm::SmallVector<T> &argStoreUses,
                                     int loopIndex) {
  llvm::SmallVector<T> filteredStoreUses;
  llvm::errs()
      << "       [filterStoreUses] Filtering store uses for loop index "
      << loopIndex << "\n";
  for (auto use : argStoreUses) {
    auto parentLoop = getParentLoopOperation(use);

    if (parentLoop == nullptr) {
      continue;
    }

    if (parentLoop->template getAttrOfType<mlir::IntegerAttr>("loop_index")
            .getInt() == loopIndex) {
      filteredStoreUses.push_back(use);
    }
  }

  return filteredStoreUses;
}

template <typename T>
void setReadOrStoreValues(mlir::BlockArgument arg,
                          llvm::json::Object &jsonContainer, int loopIndex,
                          llvm::StringRef key) {

  if (has_use<T>(arg)) {
    llvm::errs() << "     [setReadOrStoreValues] Found use of argument (" << key
                 << ")\n";
    llvm::SmallVector<T> argStoreUses = get_use<T>(arg);

    llvm::errs() << "     [setReadOrStoreValues] Found " << argStoreUses.size()
                 << " uses of argument (" << key << ")\n";

    argStoreUses = filterStoreUses<T>(argStoreUses, loopIndex);

    llvm::errs() << "     [setReadOrStoreValues] Found " << argStoreUses.size()
                 << " uses of argument (" << key << ") on loop level "
                 << loopIndex << "\n";

    // we use flat 1d index for 2D and 3D arrays therefore it will be  only
    // one index
    llvm::json::Array arrOfUseIndices;
    jsonContainer[key.str() + std::string("_count")] =
        llvm::json::Value((int64_t)argStoreUses.size());

    for (auto use : llvm::enumerate(argStoreUses)) {
      llvm::json::Object useDescription;

      auto index = use.value().getIndices()[0];

      llvm::errs() << "     [setReadOrStoreValues] Building expression tree\n";
      Node *tree = new Node();

      auto isIndexOp = index.getDefiningOp() != nullptr &&
                       isa<mlir::arith::IndexCastOp>(index.getDefiningOp());

      buildTree(*tree, isIndexOp ? index.getDefiningOp()->getOperand(0) : index,
                0);

      llvm::errs()
          << "      [setReadOrStoreValues] End of Building expression tree\n";

      if (enableDebug) {
        tree->dump();
        argStoreUses[0].dump();
      }

      if (enableDebug) {
        useDescription["debug"] = tree->toDot();
      }

      useDescription["indexes"] = llvm::json::Value(tree->getAllIndexes());
      useDescription["dims"] = llvm::json::Value(tree->getMemrefSize());

      arrOfUseIndices.push_back(llvm::json::Value(std::move(useDescription)));
    }
    jsonContainer[(key.str() + std::string("_offsets"))] =
        llvm::json::Value(std::move(arrOfUseIndices));
    jsonContainer[key] = llvm::json::Value(argStoreUses.size() > 0);
  } else {
    llvm::errs() << "     [setReadOrStoreValues] No use of argument " << key
                 << "\n";
    jsonContainer[key] = llvm::json::Value(false);
  }
}

/*
2nd FIXME: Is this comment written by me? If so, I should rewrite it.
TODO: This is a hacky way to get the name of the argument. We should
      find a better way to do this.
      For example, we can use the name of the function argument but we need to
      merge the rows of the function argument declarations.
*/
std::string getArgNameFromFuncLocation(mlir::BlockArgument arg) {

  std::string result = "";
  llvm::raw_string_ostream os(result);

  auto firstUseOfArg = arg.getUses().begin().getUser();
  auto fileLineCol = firstUseOfArg->getLoc().cast<mlir::FileLineColLoc>();
  std::string line = sourceLines[fileLineCol.getLine() - 1];

  auto nextChar = line[fileLineCol.getColumn() - 1];

  auto plusEqual = line[fileLineCol.getColumn() - 1] == '+' &&
                   line[fileLineCol.getColumn()] == '=';

  auto minusEqual = line[fileLineCol.getColumn() - 1] == '-' &&
                    line[fileLineCol.getColumn()] == '=';

  auto equal = line[fileLineCol.getColumn() - 1] == '=';

  auto initialColumn =
      plusEqual || minusEqual || equal ? 0 : fileLineCol.getColumn() - 1;

  bool hit = false;
  for (int i = initialColumn; i < line.size(); i++) {
    if (hit && line[i] == '*') {
      break;
    }

    if (line[i] == ' ' || line[i] == '*' || line[i] == '(') {
      continue;
    }
    if (line[i] == '[' || line[i] == '=' || line[i] == ';' || line[i] == ',' ||
        line[i] == ')' || line[i] == '!' || line[i] == '<' || line[i] == '>') {
      break;
    }
    os << line[i];
    hit = true;
  }

  result.erase(std::remove_if(result.begin(), result.end(), isspace),
               result.end());

  return result;
}

llvm::json::Object getArgumentMetadata(mlir::BlockArgument arg, int argIndex,
                                       int loopIndex) {

  llvm::json::Object argObject;

  std::string backupName = "arg" + Twine(arg.getArgNumber()).str();

  std::string argName = getArgNameFromFuncLocation(arg);

  if (argName == "") {
    argName = backupName;
  }

  argObject["arg_index"] = llvm::json::Value(argIndex);

  argObject["name"] = llvm::json::Value(argName);
  argObject["meta"] = llvm::json::Value(
      getArgLocation(arg.getUses().begin().getUser()->getLoc()));

  llvm::errs() << "   [getArgumentMetadata] " << argObject["name"]
               << " write\n";
  setReadOrStoreValues<mlir::memref::StoreOp>(arg, argObject, loopIndex,
                                              "write");

  llvm::errs() << "   [getArgumentMetadata] " << argObject["name"] << " read\n";
  setReadOrStoreValues<mlir::memref::LoadOp>(arg, argObject, loopIndex, "read");

  return argObject;
}

// TODO: Consider rewrite this completely:
// Since we have the debug information and it is already parsed to obtain the
// inductuion variable we can use it to get the loop bounds also.
mlir::WalkResult setLoopDepthAndBoundsAttributes(mlir::Operation *op,
                                                 uint64_t depth) {

  // assert is loopp
  if (!isLoop(op)) {
    llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                 << "ERRO! Operation is not a loop\n";
    return mlir::WalkResult::interrupt();
  }

  // if has depth attribute, it means that it was already visited

  auto inductionVariable = getInductionVariableFromLoop(op);

  op->setAttr("depth", mlir::IntegerAttr::get(
                           mlir::IntegerType::get(op->getContext(), 8), depth));

  op->setAttr("induction_variable",
              mlir::StringAttr::get(op->getContext(), inductionVariable));

  mlir::Value lbValue;
  mlir::Value ubValue;
  mlir::Region *loopBodyRegion;

  bool increment;

  auto loop = op;

  if (isWhileLoop(op)) {
    auto whileLoop = cast<mlir::scf::WhileOp>(op);
    lbValue = whileLoop.getInits()[0];
    loopBodyRegion = &whileLoop.getAfter();

    ubValue = cast<mlir::arith::CmpIOp>(
                  whileLoop.getConditionOp().getOperand(0).getDefiningOp())
                  ->getOperand(1);

    llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                 << "condition: " << whileLoop.getConditionOp().getOperand(0)
                 << "\n";

    // if condition compare is sgt
    if (auto cmp = dyn_cast<mlir::arith::CmpIOp>(
            whileLoop.getConditionOp().getOperand(0).getDefiningOp())) {
      if (cmp.getPredicate() == mlir::arith::CmpIPredicate::sgt) {
        increment = false;
      } else {
        increment = true;
      }
    }
  }

  if (isForLoop(op)) {
    auto forLoop = cast<mlir::scf::ForOp>(op);
    lbValue = forLoop.getLowerBound();
    ubValue = forLoop.getUpperBound();
    loopBodyRegion = &forLoop.getLoopBody();
    increment = true;
  }

  loop->setAttr("step", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(op->getContext(), 8),
                            increment ? 1 : -1));

  // llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
  //              << "lbValue: " << lbValue << "\n";
  // llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
  //              << "ubValue: " << ubValue << "\n";

  // llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
  //              << "loopBodyRegion: " << loopBodyRegion << "\n";

  // // lb def op
  // llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
  //              << "lbValue def op: " << lbValue.getDefiningOp()->getName()
  //              << "\n";

  // // ub def op
  // llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
  //              << "ubValue def op: " << ubValue.getDefiningOp()->getName()
  //              << "\n";

  llvm::TypeSwitch<mlir::Operation *>(lbValue.getDefiningOp())
      .Case([&](mlir::arith::ConstantOp constantOp) {
        loop->setAttr("lb", constantOp.getValueAttr());
      })
      .Case([&](mlir::arith::IndexCastOp indexCastOp) {
        auto lbValueOperand =
            indexCastOp.getIn().getDefiningOp()->getOperand(0);

        llvm::TypeSwitch<mlir::Operation *>(lbValueOperand.getDefiningOp())
            .Case([&](mlir::memref::GetGlobalOp lbGlobal) {
              loop->setAttr("lb", lbGlobal.getNameAttr());
            })
            // memref load
            .Case([&](mlir::memref::LoadOp lbLoad) {
              auto lbValue = lbLoad.getMemRef().getDefiningOp();

              llvm::TypeSwitch<mlir::Operation *>(lbValue)
                  .Case([&](mlir::memref::GetGlobalOp lbGlobal) {
                    loop->setAttr("lb", lbGlobal.getNameAttr());
                  })
                  .Case([&](mlir::arith::ConstantOp constantOp) {
                    loop->setAttr("lb", constantOp.getValueAttr());
                  })
                  .Default([&](mlir::Operation *op) {
                    llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                                 << "For loop has non-constant lower bound\n";
                    return mlir::WalkResult::interrupt();
                  });
            })
            .Default([&](mlir::Operation *op) {
              llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                           << "For loop has non-constant lower bound\n";
              return mlir::WalkResult::interrupt();
            });
      })
      .Case([&](mlir::memref::LoadOp lbLoad) {
        auto lbValue = lbLoad.getMemRef().getDefiningOp();

        llvm::TypeSwitch<mlir::Operation *>(lbValue)
            .Case([&](mlir::memref::GetGlobalOp lbGlobal) {
              loop->setAttr("lb", lbGlobal.getNameAttr());
            })
            .Case([&](mlir::arith::ConstantOp constantOp) {
              loop->setAttr("lb", constantOp.getValueAttr());
            })
            .Default([&](mlir::Operation *op) {
              llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                           << "For loop has non-constant lower bound\n";
              return mlir::WalkResult::interrupt();
            });
      })
      .Default([&](mlir::Operation *op) {
        llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                     << "For loop has non-constant lower bound\n";

        auto lbValueAsString =
            lbValue.getDefiningOp()->getName().getStringRef();

        loop->setAttr("lb",
                      mlir::StringAttr::get(op->getContext(), lbValueAsString));
        return mlir::WalkResult::interrupt();
      });

  llvm::TypeSwitch<mlir::Operation *>(ubValue.getDefiningOp())
      .Case([&](mlir::arith::ConstantOp constantOp) {
        loop->setAttr("ub", constantOp.getValueAttr());
      })
      .Case([&](mlir::arith::IndexCastOp indexCastOp) {
        auto ubValueOperand =
            indexCastOp.getIn().getDefiningOp()->getOperand(0);

        llvm::TypeSwitch<mlir::Operation *>(ubValueOperand.getDefiningOp())
            .Case([&](mlir::memref::GetGlobalOp ubGlobal) {
              loop->setAttr("ub", ubGlobal.getNameAttr());
            })
            // memref load
            .Case([&](mlir::memref::LoadOp ubLoad) {
              auto ubValue = ubLoad.getMemRef().getDefiningOp();

              llvm::TypeSwitch<mlir::Operation *>(ubValue)
                  .Case([&](mlir::memref::GetGlobalOp ubGlobal) {
                    loop->setAttr("ub", ubGlobal.getNameAttr());
                  })
                  .Default([&](mlir::Operation *op) {
                    loop->setAttr("ub", mlir::StringAttr::get(
                                            loop->getContext(),
                                            op->getName().getStringRef()));
                  });
            })
            .Default([&](mlir::Operation *op) {
              loop->setAttr(
                  "ub", mlir::StringAttr::get(loop->getContext(),
                                              op->getName().getStringRef()));
            });
      })

      // get global
      .Case([&](mlir::memref::GetGlobalOp ubGlobal) {
        loop->setAttr("ub", ubGlobal.getNameAttr());
      })
      .Case([&](mlir::memref::LoadOp ubLoad) {
        auto ubValue = ubLoad.getMemRef().getDefiningOp();

        llvm::TypeSwitch<mlir::Operation *>(ubValue)
            .Case([&](mlir::memref::GetGlobalOp ubGlobal) {
              loop->setAttr("ub", ubGlobal.getNameAttr());
            })
            .Default([&](mlir::Operation *op) {
              loop->setAttr(
                  "ub", mlir::StringAttr::get(loop->getContext(),
                                              op->getName().getStringRef()));
            });
      })
      .Default([&](mlir::Operation *attr) {
        llvm::errs() << "Unknown attribute UB type: " << attr << "\n";

        loop->setAttr("ub", mlir::StringAttr::get(op->getContext(), "unknown"));
      });

  loopBodyRegion->walk([&](mlir::Operation *inner) {
    if (!isLoop(inner)) {
      return mlir::WalkResult::skip();
    }

    return setLoopDepthAndBoundsAttributes(inner, depth + 1);
  });

  return mlir::WalkResult::advance();
}

llvm::json::Value attributeToJson(mlir::Attribute attribute) {
  llvm::json::Value jsonValue = llvm::json::Value("<null>");

  assert(attribute && "Attribute is null");

  llvm::TypeSwitch<mlir::Attribute>(attribute)
      .Case([&](mlir::IntegerAttr attr) {
        jsonValue = llvm::json::Value(attr.getInt());
      })
      .Case([&](mlir::FlatSymbolRefAttr attr) {
        jsonValue = llvm::json::Value(attr.getValue());
      })
      .Case([&](mlir::StringAttr attr) {
        jsonValue = llvm::json::Value(attr.getValue());
      })
      .Default([&](mlir::Attribute attr) {
        llvm::errs() << "Unknown attribute type: " << attr << "\n";
      });

  return jsonValue;
}

std::string printOpAttributes(mlir::Operation *op) {
  std::string result = "";
  llvm::raw_string_ostream stream(result);
  op->getAttrDictionary().print(stream);

  return stream.str();
}

void collectLoopInfo(mlir::Operation *loop, llvm::json::Array &bounds,
                     llvm::json::Array &loopAttrs, llvm::json::Array &steps) {
  llvm::errs() << "    [collectLoopBounds] Attributes"
               << printOpAttributes(loop) << "\n";
  llvm::errs() << "   [collectLoopBounds] Collecting loop bounds lb\n";

  auto lb = loop->getAttr("lb");

  bounds.insert(bounds.end(), attributeToJson(lb));
  llvm::errs() << "   [collectLoopBounds] Collecting loop bounds ub\n";

  auto ub = loop->getAttr("ub");

  bounds.insert(bounds.end(), attributeToJson(ub));

  llvm::errs() << "   [collectLoopBounds] Collecting step\n";
  auto step = loop->getAttr("step");

  steps.insert(steps.end(), attributeToJson(step));

  llvm::errs()
      << "   [collectLoopBounds] Collecting loop bounds induction_variable\n";

  loopAttrs.insert(loopAttrs.end(),
                   llvm::json::Value(getLoopAttrObject(loop->getLoc())));
  if (auto parentWhileOp = loop->getParentOfType<mlir::scf::WhileOp>()) {
    collectLoopInfo(parentWhileOp, bounds, loopAttrs, steps);
  } else if (auto parentForOp = loop->getParentOfType<mlir::scf::ForOp>()) {
    collectLoopInfo(parentForOp, bounds, loopAttrs, steps);
  }
}

int main(int argc, char **argv) {

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ops compiler\n");

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  mlir::MLIRContext context;

  context.getOrLoadDialect<mlir::AffineDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();

  context.allowUnregisteredDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  parseMlir(context, module, inputFilename.getValue());

  readSourceFile(originalSource.getValue(), sourceLines);

  mlir::func::FuncOp funcOp =
      module->lookupSymbol<mlir::func::FuncOp>(functionName.getValue());

  if (!funcOp) {
    llvm::errs() << "Function " << functionName.getValue() << " not found\n";

    return -1;
  }

  /**
   * Pass for remove all functions except the one we are interested in
   */
  module->walk([&](mlir::func::FuncOp f) {
    if (!f.getName().contains(functionName.getValue())) {
      f.erase();
    }

    return mlir::WalkResult::advance();
  });

  llvm::json::Object function;
  function["name"] = llvm::json::Value(functionName.getValue());
  u_int16_t loopNestCounter = 0;
  llvm::json::Array loops;

  int loopIndex = 0;

  // // replace while loops with for loops
  // funcOp.walk([&](mlir::Operation *op) {
  //   // if not while loop, skip
  //   if (!isa<mlir::scf::WhileOp>(op)) {
  //     return mlir::WalkResult::skip();
  //   }

  //   auto whileOp = cast<mlir::scf::WhileOp>(op);
  //   llvm::errs() << "    [whileToFor] WhileOp: " << whileOp << "\n";
  //   auto conditionOp = whileOp.getConditionOp();
  //   llvm::errs() << "    [whileToFor] ConditionOp: " << conditionOp << "\n";

  //   auto predicate = conditionOp.getCondition();

  //   llvm::errs() << "    [whileToFor] Predicate: " << predicate << "\n";

  //   auto ubValue = cast<mlir::arith::CmpIOp>(
  //                      whileOp.getConditionOp().getOperand(0).getDefiningOp())
  //                      ->getOperand(1)
  //                      .getDefiningOp()
  //                      ->getOperand(0);

  //   auto lbValue = whileOp.getOperand(0);

  //   llvm::errs() << "    [whileToFor] lb: " << lbValue << "\n";
  //   llvm::errs() << "    [whileToFor] ub: " << ubValue << "\n";

  //   // if lte condition
  //   if (predicate.getDefiningOp<mlir::arith::CmpIOp>().getPredicate() ==
  //       mlir::arith::CmpIPredicate::sge) {
  //     // swap ubValue and lbValue
  //     auto temp = ubValue;
  //     ubValue = lbValue;
  //     lbValue = temp;

  //     llvm::errs() << "    [whileToFor] Swapping lb and ub\n";
  //   }

  //   auto builder = mlir::OpBuilder(op);

  //   auto contantOneIndex =
  //       builder.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 1);
  //   auto forOp = builder.create<mlir::scf::ForOp>(op->getLoc(), lbValue,
  //                                                 ubValue, contantOneIndex);

  //   // move all operations from whileOp to forOp
  //   forOp.getBody()->getOperations().splice(
  //       forOp.getBody()->getOperations().begin(),
  //       whileOp.getAfter().begin()->getOperations());

  //   // if forOp has terminator, remove it
  //   if (forOp.getBody()->getTerminator()) {
  //     forOp.getBody()->getTerminator()->erase();
  //   }

  //   // add "replaced" attribute
  //   forOp->setAttr("replaced", mlir::UnitAttr::get(&context));

  //   llvm::errs() << "    [whileToFor] ForOp: " << forOp << "\n";
  //   // replace whileOp with forOp
  //   // whileOp.replaceAllUsesWith(forOp);
  //   module->dump();

  //   if (whileOp->use_empty()) {

  //     whileOp.erase();
  //   }

  //   return mlir::WalkResult::advance();
  // });

  // // dump module and exit
  // module->dump();
  // return 0;

  /**
   * Pass for annotating loops with index and LB and UB
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not a loop, skip
    if (!isLoop(op)) {
      return mlir::WalkResult::skip();
    }

    op->setAttr("loop_index",
                mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32),
                                       loopIndex++));

    return setLoopDepthAndBoundsAttributes(op, 1);
  });

  /**
   * Pass for swapping bounds of adds where the operands are an
   * induction variable and a constant to ensure that the right child is the
   * constant
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not an add operation, skip
    if (!isa<mlir::arith::AddIOp>(op)) {
      return mlir::WalkResult::skip();
    }

    auto addOp = cast<mlir::arith::AddIOp>(op);

    // if any of the operands is a block argument, skip
    if (addOp.getLhs().isa<mlir::BlockArgument>() ||
        addOp.getRhs().isa<mlir::BlockArgument>()) {
      return mlir::WalkResult::skip();
    }

    // if rhs is not a constant, skip
    if (!isa<mlir::arith::ConstantOp>(addOp.getRhs().getDefiningOp())) {
      return mlir::WalkResult::skip();
    }

    // if lhs is not a muli operation, skip
    if (!isa<mlir::arith::MulIOp>(addOp.getLhs().getDefiningOp())) {
      return mlir::WalkResult::skip();
    }

    auto builder = mlir::OpBuilder::atBlockBegin(addOp->getBlock());
    builder.setInsertionPoint(addOp);

    auto newOp = builder.create<mlir::arith::AddIOp>(
        addOp.getLoc(), addOp.getRhs(), addOp.getLhs());

    addOp.replaceAllUsesWith(newOp.getResult());

    addOp.erase();
  });

  /**
   * Pass for swapping bounds of multiplications where the operands are an
   * induction variable and a constant to ensure that the right child is the
   * constant
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not an muli operation, skip
    if (!isa<mlir::arith::MulIOp>(op)) {
      return mlir::WalkResult::skip();
    }

    auto mulOp = cast<mlir::arith::MulIOp>(op);

    // if rhs is not a constant, skip
    if (mulOp.getRhs().isa<mlir::BlockArgument>() ||
        !isa<mlir::arith::ConstantOp>(mulOp.getRhs().getDefiningOp())) {
      return mlir::WalkResult::skip();
    }

    auto builder = mlir::OpBuilder::atBlockBegin(mulOp->getBlock());
    builder.setInsertionPoint(mulOp);

    auto newOp = builder.create<mlir::arith::MulIOp>(
        mulOp.getLoc(), mulOp.getRhs(), mulOp.getLhs());

    mulOp.replaceAllUsesWith(newOp.getResult());

    mulOp.erase();
  });

  if (emitMLIR) {
    module->print(llvm::outs());

    return 0;
  }

  /**
   * Main pass for collecting metadata about loops
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not a loop, skip
    if (!isLoop(op)) {
      return mlir::WalkResult::skip();
    }

    llvm::errs() << "[main] Checking whether it is an outermost loop... \n";

    // if op is a loop but it is not an outermost loop of a loop nest, skip
    if (op->getParentOfType<mlir::scf::WhileOp>() != nullptr ||
        op->getParentOfType<mlir::scf::ForOp>() != nullptr) {
      return mlir::WalkResult::skip();
    }

    llvm::errs() << "[main] Counting loopnest depth... \n";
    llvm::json::Object loopObject;

    auto loopNestSize = countLoopNestDepth(op);

    llvm::errs() << "[main] Collecting inner loops for nest with size: "
                 << Twine(loopNestSize).str() << "... \n";
    auto innerLoops = getLoopsOnLevel(op, loopNestSize);

    uint64_t loopCounter = 0;

    llvm::errs() << "[main] Collected inner loops: " << innerLoops.size()
                 << "\n";

    // 2nd FIXME: this is not  needed; we should bail earlier...

    // this for loop is needed to handle edge cases where there are not
    // perfectly nested loops TODO: there should not be any inperfectly nested
    // loops
    for (auto en : llvm::enumerate(innerLoops)) {

      auto innerLoop = en.value();
      auto index = en.index();
      llvm::errs() << "[main] Iterating loop: " << index << "\n";

      llvm::errs() << "[main] Collecting args... \n";

      auto arguments = getFuncArgumentsUsedInLoop(innerLoop);

      llvm::errs() << "[main] Collected args: " << arguments.size() << "... \n";

      int currentLoopIndex =
          innerLoop->getAttrOfType<mlir::IntegerAttr>("loop_index").getInt();
      llvm::json::Array args;
      // for every function arguments used in the innermost loop
      for (auto arg : llvm::enumerate(arguments)) {
        llvm::errs() << "[main] Get argument metadata for arg: " << en.index()
                     << "... \n";

        llvm::json::Object argObject =
            getArgumentMetadata(arg.value().getValue(),
                                toInt(arg.value().getKey()), currentLoopIndex);
        args.push_back(std::move(argObject));
      }

      llvm::json::Object indexRange;
      indexRange["begin"] = llvm::json::Value(loopNestCounter);
      indexRange["end"] =
          llvm::json::Value(loopNestCounter + loopNestSize - 1 + loopCounter);
      loopObject["loop_position_index_range"] =
          llvm::json::Value(std::move(indexRange));

      loopObject["size"] = llvm::json::Value(loopNestSize);
      loopObject["args"] = llvm::json::Value(std::move(args));

      llvm::errs() << "[main] Collecting loop bounds\n";
      llvm::json::Array bounds;
      llvm::json::Array loopAttrs;
      llvm::json::Array steps;
      collectLoopInfo(innerLoop, bounds, loopAttrs, steps);
      loopObject["bounds"] = llvm::json::Value(std::move(bounds));
      loopObject["steps"] = llvm::json::Value(std::move(steps));

      loopObject["loop_attributes"] = llvm::json::Value(std::move(loopAttrs));

      loops.push_back(llvm::json::Value(std::move(loopObject)));
      loopCounter++;
    }

    loopNestCounter++;

    return mlir::WalkResult::advance();
  });

  if (enableDetailedPrint) {

    llvm::json::Array loopsOrCalls;

    /**
     * Main pass for collecting metadata about calls
     */
    int indexM = 0;
    for (mlir::Operation &op : funcOp.getBody().getOps()) {
      llvm::json::Object loopOrCallObject;

      if (isWhileLoop(&op)) {
        mlir::scf::WhileOp whileOp = cast<mlir::scf::WhileOp>(op);
        loopOrCallObject["type"] = llvm::json::Value("loop_nest");
        loopOrCallObject["outermost_loop_index"] = llvm::json::Value(
            whileOp->getAttrOfType<mlir::IntegerAttr>("loop_index").getInt());
        loopsOrCalls.push_back(llvm::json::Value(std::move(loopOrCallObject)));
      }

      // if callop
      if (isa<mlir::func::CallOp>(op)) {
        mlir::func::CallOp callOp = cast<mlir::func::CallOp>(op);
        llvm::errs() << "[main] Found call: " << indexM << "\n";
        loopOrCallObject["type"] = llvm::json::Value("call");
        loopOrCallObject["callee"] = llvm::json::Value(callOp.getCallee());
        mlir::ValueRange args = callOp.getOperands();
        llvm::json::Array argsArray;
        for (auto arg : llvm::enumerate(args)) {
          llvm::json::Object argObject;

          argObject["name"] = llvm::json::Value(getArgNameFromFuncLocation(
              arg.value().cast<mlir::BlockArgument>()));
          argObject["index"] = llvm::json::Value(
              arg.value().cast<mlir::BlockArgument>().getArgNumber());

          argsArray.push_back(llvm::json::Value(std::move(argObject)));
        }
        loopOrCallObject["args"] = llvm::json::Value(std::move(argsArray));
        loopsOrCalls.push_back(llvm::json::Value(std::move(loopOrCallObject)));
      }

      indexM++;
    }
    function["loops_or_calls"] = llvm::json::Value(std::move(loopsOrCalls));
  }

  llvm::errs() << "[main] Saving result \n";

  function["loops"] = llvm::json::Value(std::move(loops));
  llvm::outs() << formatv("{0:2}", llvm::json::Value(std::move(function)))
               << "\n";

  return 0;
}