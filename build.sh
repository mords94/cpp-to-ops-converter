scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
WORKPATH=$scriptDir

BUILD_DIR=$scriptDir/Polygeist/build
OPS_BUILD_DIR=$scriptDir/build

# Build mlir and clang
mkdir $BUILD_DIR
mkdir $OPS_BUILD_DIR

echo "Build MLIR and Clang and Polygeist"
cmake -G Ninja ../llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_EXTERNAL_PROJECTS="polygeist" \
    -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist

echo "Build Project"

cd $OPS_BUILD_DIR
cmake -G Ninja .. -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir
ninja
