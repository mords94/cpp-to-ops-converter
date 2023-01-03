#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <iostream>
#include <list>
#include <math.h>
#include <random>
#include <string>

enum IterationOrder {
  PreOrder,
  InOrder,
  PostOrder,
};

std::string gen_random(const int len) {
  std::string generated = "";
  llvm::raw_string_ostream stream(generated);

  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_int_distribution<int> dist(0, 25);
  for (int i = 0; i < len; ++i) {
    stream << (char)('a' + dist(mt));
  }

  return stream.str();
}

class Node {

protected:
  Node *parent_;
  int depth_ = 0;
  Node *left_ = nullptr;
  Node *right_ = nullptr;
  bool isRight_ = false;
  bool isLeft_ = false;
  mlir::Value value_;
  mlir::Operation *ownerLoop = nullptr;

public:
  virtual ~Node() {}
  void setParent(Node *parent) { this->parent_ = parent; }
  Node *getParent() const { return this->parent_; }
  void setDepth(int depth) { this->depth_ = depth; }

  void setValue(mlir::Value value) { this->value_ = value; }
  mlir::Value getValue() { return this->value_; }

  mlir::Operation *getOwnerLoop() { return this->ownerLoop; }
  void setOwnerLoop(mlir::Operation *op) { this->ownerLoop = op; }
  // bool isInductionVar() { this->getOwnerLoop() != nullptr; }

  void setIsRightChild(bool isRight) { this->isRight_ = isRight; }
  bool isRightChild() { return this->isRight_; }
  void setIsLeftChild(bool isLeft) { this->isLeft_ = isLeft; }
  bool isLeftChild() { return this->isLeft_; }

  bool hasRightChild() { return this->right_ != nullptr; }
  bool hasLeftChild() { return this->left_ != nullptr; }

  Node *getRightChild() { return this->right_; }
  Node *getLeftChild() { return this->left_; }

  llvm::SmallVector<Node *, 4> getChildren() {
    llvm::SmallVector<Node *, 4> children;
    if (this->left_ != nullptr) {
      children.push_back(this->left_);
    }

    if (this->right_ != nullptr) {
      children.push_back(this->right_);
    }

    return children;
  }

  Node *getSibling() {
    if (this->isLeftChild()) {
      return this->getParent()->getRightChild();
    }

    if (this->isRightChild()) {
      return this->getParent()->getLeftChild();
    }

    return nullptr;
  }

  mlir::Operation *getSiblingOperation() {
    if (this->getSibling() != nullptr) {
      return this->getSibling()->getValue().getDefiningOp();
    }

    return nullptr;
  }

  mlir::Operation *getOperation() {
    if (this->getValue() != nullptr) {
      return this->getValue().getDefiningOp();
    }

    return nullptr;
  }

  mlir::Operation *getParentOperation() {
    if (this->getParent() != nullptr) {
      return this->getParent()->getValue().getDefiningOp();
    }

    return nullptr;
  }

  int getDepth() { return this->depth_; }

  Node *getRoot() {
    Node *current = this;
    while (current->getParent() != nullptr) {
      current = current->getParent();
    }

    return current;
  }

  int getOwnerLoopDepth() {
    if (this->getOwnerLoop() != nullptr) {
      return this->getOwnerLoop()
          ->getAttr("depth")
          .cast<mlir::IntegerAttr>()
          .getInt();
    }

    return -1;
  }

  int getOwnerLoopDepthReverse() {
    int ownerLoopDepth = this->getOwnerLoopDepth();

    if (ownerLoopDepth != -1) {
      return this->getMaxLoopDepth() - ownerLoopDepth;
    }

    return 0;
  }

  template <typename T> bool hasChildOfType() {
    if (this->left_ != nullptr) {
      if (llvm::isa<T>(this->left_->getValue().getDefiningOp())) {
        return true;
      }
    }

    if (this->right_ != nullptr) {
      if (llvm::isa<T>(this->right_->getValue().getDefiningOp())) {
        return true;
      }
    }

    return false;
  }

  void setLeft(Node *component) {
    this->left_ = component;
    component->setParent(this);
    component->setIsLeftChild(true);
  }

  void setRight(Node *component) {
    this->right_ = component;
    component->setParent(this);
    component->setIsRightChild(true);
  }

  void iterate(std::function<void(Node *)> lambda,
               IterationOrder order = IterationOrder::PreOrder) {
    if (order == IterationOrder::PreOrder) {
      lambda(this);
    }

    if (this->left_) {
      this->left_->iterate(lambda, order);
    }

    if (order == IterationOrder::InOrder) {
      lambda(this);
    }

    if (this->right_) {
      this->right_->iterate(lambda, order);
    }

    if (order == IterationOrder::PostOrder) {
      lambda(this);
    }
  }

  std::string getOwnerLoopAttributes() {
    std::string result = "";
    if (this->getOwnerLoop() != nullptr) {
      llvm::raw_string_ostream ownerStream(result);
      this->getOwnerLoop()->getAttrDictionary().print(ownerStream);
    }

    return result;
  }

  auto getOwnerLoopDepthAttribute() {
    if (this->getOwnerLoop() != nullptr) {
      return this->getOwnerLoop()->getAttr("depth");
    }

    return mlir::Attribute();
  }

  void dump() {
    this->iterate([](Node *node) {
      printWithDepth(node->getValue(), node->getDepth(),
                     node->getOwnerLoopAttributes());
    });
  }

  std::string toString() {
    std::string result = "";
    llvm::raw_string_ostream result_stream(result);

    if (this->getValue().isa<mlir::BlockArgument>()) {
      result_stream << this->getInductionVarName();
    } else {
      llvm::TypeSwitch<mlir::Operation *>(this->getValue().getDefiningOp())
          .Case([&](mlir::arith::ConstantOp constantOp) {
            result_stream << constantOp.getValue();
          })
          .Case(
              [&](mlir::arith::MulIOp indexCastOp) { result_stream << " * "; })
          .Case(
              [&](mlir::arith::AddIOp indexCastOp) { result_stream << " + "; })
          .Case([&](mlir::memref::GetGlobalOp getGlobalOp) {
            result_stream << getGlobalOp.getName();
          })
          .Default([&](mlir::Operation *attr) { result_stream << attr; });
    }

    return result;
  }

  void toDotRecursive(Node *node, llvm::raw_string_ostream *result_stream,
                      std::string iterationCounter) {

    *result_stream << iterationCounter << " [label=\"" << node->toString()
                   << "\"];";

    for (auto en : llvm::enumerate(node->getChildren())) {
      std::string index = iterationCounter + "_" + std::to_string(en.index());
      *result_stream << iterationCounter << " -- " << index << ";";
      this->toDotRecursive(en.value(), result_stream, index);
    }
  }

  std::string toDot() {
    // std::string result = "digraph G { ";
    std::string result = "";
    llvm::raw_string_ostream result_stream(result);

    std::string nodePrefix = gen_random(6);
    this->toDotRecursive(this, &result_stream, nodePrefix);

    // result_stream << "} ";
    return result_stream.str();
  }

  llvm::json::Value toJSON() {

    llvm::json::Array root;
    llvm::json::Array children;

    this->iterate(
        [&](Node *node) {
          llvm::json::Object obj;
          std::string valueStr;
          llvm::raw_string_ostream o(valueStr);

          obj["_treeDepth"] = llvm::json::Value(node->getDepth());
          node->getValue().print(o);
          obj["text"] = llvm::json::Value(o.str());

          if (node->getOwnerLoop() != nullptr) {
            llvm::json::Object loopDescriptor;

            for (auto attr : node->getOwnerLoop()->getAttrs()) {
              std::string attrValueStr;
              llvm::raw_string_ostream av(attrValueStr);

              attr.getValue().print(av);
              loopDescriptor[attr.getName().str()] =
                  llvm::json::Value(av.str());
            }
            obj["_ownerLoopDescriptor"] =
                llvm::json::Value(std::move(loopDescriptor));
            obj["_parentOp"] = llvm::json::Value(o.str());
          }

          if (node->hasLeftChild() || node->hasRightChild()) {
            obj["children"] = llvm::json::Value(std::move(children));
            children = llvm::json::Array();
          }

          if (node->getParent() != nullptr) {
            children.push_back(llvm::json::Value(std::move(obj)));
          }

          if (node->getParent() == nullptr) {
            root.push_back(llvm::json::Value(std::move(obj)));
          }
        },
        IterationOrder::PostOrder);

    return llvm::json::Value(std::move(root));
  }

  int getTreeDepth() {
    int depth = 0;
    this->iterate([&](Node *node) {
      if (node->getDepth() > depth) {
        depth = node->getDepth();
      }
    });

    return depth;
  }

  int getMultiplyCount() {
    int count = 0;
    this->iterate([&](Node *node) {
      if (!node->getValue().getDefiningOp()) {
        return;
      }

      if (isa<mlir::arith::MulIOp>(node->getValue().getDefiningOp())) {
        count++;
      }
    });

    return count;
  }

  int getMemrefSize() {
    return getMemrefSizeFromMultiplicationCount(this->getMultiplyCount());
  }

  std::string getInductionVarName() {

    assert(this->getOwnerLoop() != nullptr &&
           "Node does not represent an induction variable");

    return this->getOwnerLoop()
        ->getAttrOfType<mlir::StringAttr>("induction_variable")
        .getValue()
        .str();
  };

  auto getAllLeafs() {
    std::vector<Node *> leafs;
    this->iterate([&](Node *node) {
      if (!node->hasLeftChild() && !node->hasRightChild()) {
        leafs.push_back(node);
      }
    });

    return leafs;
  }

  void getFarLeftAllChildren(std::vector<Node *> &children) {
    if (this->hasLeftChild()) {
      this->getLeftChild()->getFarLeftAllChildren(children);
    } else {
      children.push_back(this);
    }
  }

  // void getFarLeftAddIWithMulIRightChild() {

  //   if (!node->getValue().isa<mlir::BlockArgument>() &&
  //       isa<mlir::arith::AddIOp>(node->getValue().getDefiningOp())) {

  //     if (node->hasRightChild() &&
  //         isa<mlir::arith::MulIOp>(
  //             node->getRightChild()->getValue().getDefiningOp())) {
  //       llvm::errs() << "Found addi node: ";
  //       node->getValue().dump();
  //       llvm::errs() << "\n";
  //     }
  //   }
  // }

  void getFarLeftNodeTravel(Node **node) {
    if (this->hasLeftChild()) {
      this->getLeftChild()->getFarLeftNodeTravel(node);
    } else {
      *node = this;
    }
  }

  Node *getFarLeftNode() {
    Node *node = nullptr;
    this->getFarLeftNodeTravel(&node);
    return node;
  }

  mlir::Value getFarLeftValue() {
    Node *node = this->getFarLeftNode();
    return node->getValue();
  }

  bool isOnlyTerm() {

    if (this->hasLeftChild() && this->hasRightChild()) {
      return true;
    }

    return false;
  }

  bool isFirstOnlyTerm() {

    if (this->getSibling() == nullptr) {
      return false;
    }

    if (this->getSibling()->contains<mlir::arith::MulIOp>()) {
      return true;
    }

    return false;
  }

  template <typename T> bool contains() {
    if (this->getValue() == nullptr) {
      return false;
    }

    if (this->getValue().isa<mlir::BlockArgument>()) {
      return false;
    }

    return mlir::isa<T>(this->getValue().getDefiningOp());
  }

  int getMaxDepth() {
    int maxDepth = 0;
    this->iterate([&](Node *node) {
      if (node->getDepth() > maxDepth) {
        maxDepth = node->getDepth();
      }
    });

    return maxDepth;
  }

  int getMaxLoopDepth() {
    int maxDepth = 0;
    this->getRoot()->iterate([&](Node *node) {
      auto currentDepth = node->getOwnerLoopDepth();

      if (currentDepth > maxDepth) {
        maxDepth = currentDepth;
      }
    });

    return maxDepth;
  }

  /* Retuns all nodes that have MulI value and parent with MulI value */
  llvm::json::Array getAllIndexes() {
    llvm::json::Array arr;

    llvm::errs() << "Getting all indexes\n";

    this->dump();
    // get first index term
    auto first = getTermOrSubExprJson(this->getFarLeftNode(), [](Node *node) {
      return node->isFirstOnlyTerm() || node->getMaxDepth() < 2;
    });

    arr.push_back(std::move(first));

    llvm::errs() << "First OK\n";
    this->iterate([&](Node *node) {
      if (!node->contains<mlir::arith::MulIOp>()) {
        return;
      }

      if (!node->getLeftChild()->contains<mlir::arith::MulIOp>()) {

        /*
          Left child of a mulI operation represents the indexing expression
          e,g.,
            * induction var:          i
            * induction with offset:  i + 1
            * global const:           N
            * constant nubmer:        1
        */
        auto leftChild = node->getLeftChild();

        /*
          If the indexing expression is a binary operation (add), we use the
          first operand of the add
          else we use the indexing expression itself which is a term only
        */
        auto term =
            leftChild->hasLeftChild() ? leftChild->getLeftChild() : leftChild;

        auto subTermOrSubExpr = getTermOrSubExprJson(
            term, [&](Node *_innerNode) { return !leftChild->hasLeftChild(); });

        arr.push_back(std::move(subTermOrSubExpr));
      }
    });

    return arr;
  }

  llvm::json::Object
  getTermOrSubExprJson(Node *node, std::function<bool(Node *)> testIfTerm) {
    llvm::json::Object obj;

    assert(node != nullptr && "Node is null");
    llvm::errs() << "Getting term or sub expr" << node->getValue() << "\n";
    if (testIfTerm(node)) {
      if (isa<mlir::BlockArgument>(node->getValue())) {
        // loop induction with offset 0, e.g.: i

        llvm::errs() << "Found loop induction: " << node->getValue() << "\n";
        obj["type"] = "induction_var";
        obj["offset"] = 0;
        obj["stencil"] = node->getInductionVarName();
      } else if (auto globalOp = dyn_cast<mlir::memref::GetGlobalOp>(
                     node->getValue().getDefiningOp())) {

        // global memref with offset 0 e.g.: N
        llvm::errs() << "Found global memref: " << node->getValue() << "\n";
        obj["type"] = "global_memref";
        obj["name"] = globalOp.getName();
      } else if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(
                     node->getValue().getDefiningOp())) {
        auto constValue =
            cast<mlir::IntegerAttr>(constOp.getValueAttr()).getInt();
        // global constant e.g.: 1
        llvm::errs() << "Found global constant: " << node->getValue() << "\n";
        obj["type"] = "global_constant";
        obj["value"] = constValue;
      } else {
        llvm::errs() << "Test: true, type: unknown " << node->getValue()
                     << "\n";
        // unhandled case
        obj["type"] = "unknown";
      }
    } else {
      if (node->getValue().isa<mlir::BlockArgument>()) {
        // loop induction with offset e.g: i+1
        auto constOp =
            dyn_cast<mlir::arith::ConstantOp>(node->getSiblingOperation());

        auto offset = cast<mlir::IntegerAttr>(constOp.getValueAttr()).getInt();

        llvm::errs() << "Found loop induction with offset: " << node->getValue()
                     << " offset: " << offset << "\n";
        obj["type"] = "induction_var";
        obj["offset"] = offset;
        obj["stencil"] = node->getInductionVarName();
      } else {

        if (auto getGlobalOp = dyn_cast<mlir::memref::GetGlobalOp>(
                node->getValue().getDefiningOp())) {
          // global memref with offset e.g.: N+1
          auto constOp =
              dyn_cast<mlir::arith::ConstantOp>(node->getSiblingOperation());

          auto offset =
              cast<mlir::IntegerAttr>(constOp.getValueAttr()).getInt();
          llvm::errs() << "Found global memref with offset: "
                       << node->getValue() << " offset: " << offset << "\n";
          obj["type"] = "global_memref";
          obj["name"] = getGlobalOp.getName();
          obj["offset"] = offset;
        } else {
          // unhandled case
          llvm::errs() << "Unknown with offset: " << node->getValue() << "\n";
          obj["type"] = "unknown_with_offset";
        }
      }
    }

    return obj;
  }

  /**
   * @brief Prints an MLIR value with indentation
   * @param value The value to print
   * @param indent The indentation level
   */
  static void printWithDepth(mlir::Value value, int indent,
                             std::string additional = "") {

    llvm::errs() << std::string(indent, '  ') << value << " " << additional
                 << "  (" << indent << ")"
                 << "\n";
  }

  /**
   * @brief Calculates the rank of a memref from the number of multiplications
   * @param multiplyCount The number of multiplications
   * @return The rank of the memref
   * @details f(m) = (1 + floor(sqrt(1+8m))) / 2
   */
  static int getMemrefSizeFromMultiplicationCount(int multiplyCount) {
    float sqrt = std::sqrt(1 + 8 * multiplyCount);
    float result = 1 + sqrt;
    result = result / 2;
    return (int)result;
  }
};