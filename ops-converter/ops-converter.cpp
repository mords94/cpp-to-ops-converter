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

  std::string inductionVariable = condition.substr(0, condition.find("<"));
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

      Node *children[1] = {new Node()};

      node.setLeft(children[0]);
      node.setValue(indexCastOp.getIn());

      buildTree(*children[0], indexCastOp.getIn(), depth + 1);

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

template <typename T>
llvm::SmallVector<T> filterStoreUses(llvm::SmallVector<T> &argStoreUses,
                                     int loopIndex) {
  llvm::SmallVector<T> filteredStoreUses;

  for (auto use : argStoreUses) {
    auto parent = use->getParentOp();
    if (!isa<mlir::scf::WhileOp>(parent) && !isa<mlir::scf::ForOp>(parent)) {
      continue;
    }

    if (parent->template getAttrOfType<mlir::IntegerAttr>("loop_index")
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
    llvm::errs() << "   [setReadOrStoreValues] Found use of argument (" << key
                 << ")\n";
    llvm::SmallVector<T> argStoreUses = get_use<T>(arg);

    llvm::errs() << "   [setReadOrStoreValues] Found " << argStoreUses.size()
                 << " uses of argument (" << key << ")\n";

    argStoreUses = filterStoreUses<T>(argStoreUses, loopIndex);

    llvm::errs() << "   [setReadOrStoreValues] Found " << argStoreUses.size()
                 << " uses of argument (" << key << ") on loop level "
                 << loopIndex << "\n";

    // we use flat 1d index for 2D and 3D arrays therefore it will be  only
    // one index
    llvm::json::Array arrOfUseIndices;
    jsonContainer[key.str() + std::string("_count")] =
        llvm::json::Value(Twine(argStoreUses.size()).str());

    for (auto use : llvm::enumerate(argStoreUses)) {
      llvm::json::Object useDescription;

      auto index = use.value().getIndices()[0];

      llvm::errs() << "   [setReadOrStoreValues] Building expression tree\n";
      Node *tree = new Node();

      auto isIndexOp = index.getDefiningOp() != nullptr &&
                       isa<mlir::arith::IndexCastOp>(index.getDefiningOp());

      buildTree(*tree, isIndexOp ? index.getDefiningOp()->getOperand(0) : index,
                0);

      llvm::errs()
          << "   [setReadOrStoreValues] End of Building expression tree\n";

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
    jsonContainer[key] = llvm::json::Value(argStoreUses.length() > 0);
  } else {
    llvm::errs() << "   [setReadOrStoreValues] No use of argument " << key
                 << "\n";
    jsonContainer[key] = llvm::json::Value(false);
  }
}

std::string getArgNameFromFuncLocation(mlir::BlockArgument arg) {

  std::string result = "";
  llvm::raw_string_ostream os(result);

  auto firstUseOfArg = arg.getUses().begin().getUser();
  auto fileLineCol = firstUseOfArg->getLoc().cast<mlir::FileLineColLoc>();
  std::string line = sourceLines[fileLineCol.getLine() - 1];

  auto nextChar = line[fileLineCol.getColumn() - 1];

  auto initialColumn = nextChar == '=' ? 0 : fileLineCol.getColumn() - 1;

  bool hit = false;
  for (int i = initialColumn; i < line.size(); i++) {
    if (hit && line[i] == '*') {
      break;
    }

    if (line[i] == ' ' || line[i] == '*' || line[i] == '(') {
      continue;
    }
    if (line[i] == '[' || line[i] == '=' || line[i] == ';' || line[i] == ',' ||
        line[i] == ')') {
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

  llvm::errs() << "   [getArgumentMetadata] " << argObject["name"]
               << " write\n";
  setReadOrStoreValues<mlir::memref::StoreOp>(arg, argObject, loopIndex,
                                              "write");

  llvm::errs() << "   [getArgumentMetadata] " << argObject["name"] << " read\n";
  setReadOrStoreValues<mlir::memref::LoadOp>(arg, argObject, loopIndex, "read");

  return argObject;
}

mlir::WalkResult setLoopDepthAndBoundsAttributes(mlir::Operation *op,
                                                 uint64_t depth) {

  op->setAttr("depth", mlir::IntegerAttr::get(
                           mlir::IntegerType::get(op->getContext(), 8), depth));

  auto inductionVariable = getInductionVariableFromLoop(op);

  op->setAttr("induction_variable",
              mlir::StringAttr::get(op->getContext(), inductionVariable));

  op->dump();

  if (isWhileLoop(op)) {
    auto loop = cast<mlir::scf::WhileOp>(op);
    auto initialValue = loop.getInits()[0];

    if (auto constantIntOp =
            dyn_cast<mlir::arith::ConstantOp>(initialValue.getDefiningOp())) {
      loop->setAttr("lb", constantIntOp.getValueAttr());
    }

    auto ubValue = cast<mlir::arith::CmpIOp>(
                       loop.getConditionOp().getOperand(0).getDefiningOp())
                       ->getOperand(1)
                       .getDefiningOp()
                       ->getOperand(0);

    if (auto ubGlobal =
            dyn_cast<mlir::memref::GetGlobalOp>(ubValue.getDefiningOp())) {

      loop->setAttr("ub", ubGlobal.getNameAttr());
    }

    loop.getAfter().walk([&](mlir::Operation *inner) {
      if (!isLoop(inner)) {
        return mlir::WalkResult::skip();
      }

      return setLoopDepthAndBoundsAttributes(inner, depth + 1);
    });
  }

  if (isForLoop(op)) {
    auto loop = cast<mlir::scf::ForOp>(op);

    if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(
            loop.getLowerBound().getDefiningOp())) {
      loop->setAttr("lb", constantOp.getValueAttr());
    }

    // Consider this for lb and while too
    llvm::TypeSwitch<mlir::Operation *>(loop.getUpperBound().getDefiningOp())
        .Case([&](mlir::arith::ConstantOp constantOp) {
          loop->setAttr("ub", constantOp.getValueAttr());
        })
        .Case([&](mlir::arith::IndexCastOp indexCastOp) {
          auto ubValue = indexCastOp.getIn().getDefiningOp()->getOperand(0);

          if (auto ubGlobal = dyn_cast<mlir::memref::GetGlobalOp>(
                  ubValue.getDefiningOp())) {

            loop->setAttr("ub", ubGlobal.getNameAttr());
          }
        })
        .Default([&](mlir::Operation *attr) {
          // llvm::errs() << "Unknown attribute type: " << attr << "\n";
        });

    loop.getBodyRegion().walk([&](mlir::Operation *inner) {
      if (!isLoop(inner)) {
        return mlir::WalkResult::skip();
      }

      return setLoopDepthAndBoundsAttributes(inner, depth + 1);
    });
  }

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

      });

  return jsonValue;
}

void collectLoopInfo(mlir::Operation *loop, llvm::json::Array &bounds,
                     llvm::json::Array &loopAttrs) {
  llvm::errs() << "   [collectLoopBounds] Collecting loop bounds lb\n";

  bounds.insert(bounds.end(), attributeToJson(loop->getAttr("lb")));
  llvm::errs() << "   [collectLoopBounds] Collecting loop bounds ub\n";

  bounds.insert(bounds.end(), attributeToJson(loop->getAttr("ub")));
  llvm::errs()
      << "   [collectLoopBounds] Collecting loop bounds induction_variable\n";

  loopAttrs.insert(loopAttrs.end(),
                   llvm::json::Value(getLoopAttrObject(loop->getLoc())));
  if (auto parentWhileOp = loop->getParentOfType<mlir::scf::WhileOp>()) {
    collectLoopInfo(parentWhileOp, bounds, loopAttrs);
  } else if (auto parentForOp = loop->getParentOfType<mlir::scf::ForOp>()) {
    collectLoopInfo(parentForOp, bounds, loopAttrs);
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
      collectLoopInfo(innerLoop, bounds, loopAttrs);
      loopObject["bounds"] = llvm::json::Value(std::move(bounds));

      loopObject["loop_attributes"] = llvm::json::Value(std::move(loopAttrs));

      loops.push_back(llvm::json::Value(std::move(loopObject)));
      loopCounter++;
    }

    loopNestCounter++;

    return mlir::WalkResult::advance();
  });

  llvm::errs() << "[main] Saving result \n";

  function["loops"] = llvm::json::Value(std::move(loops));
  llvm::outs() << formatv("{0:2}", llvm::json::Value(std::move(function)))
               << "\n";

  return 0;
}