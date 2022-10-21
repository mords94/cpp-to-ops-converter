PREFIX=/Users/mac/Projects/mlir/cpp-to-ops-converter/llvm-project/llvm/build-release
BUILD_DIR=/Users/mac/Projects/mlir/cpp-to-ops-converter/llvm-project/llvm/build-release
cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
