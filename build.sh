PREFIX=/Users/mac/Projects/mlir/cpp-to-ops-converter/llvm-project/llvm/build-release
BUILD_DIR=/Users/mac/Projects/mlir/cpp-to-ops-converter/llvm-project/llvm/build-release
cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone

# Polygiest
# cmake -G Ninja .. \
#     -DMLIR_DIR=$PWD/../../llvm-project/llvm/build-release/lib/cmake/mlir \
#     -DCLANG_DIR=$PWD/../../llvm-project/llvm/build-release/lib/cmake/clang \
#     -DLLVM_TARGETS_TO_BUILD="host" \
#     -DLLVM_ENABLE_ASSERTIONS=ON \
#     -DCMAKE_BUILD_TYPE=DEBUG
# ninja
