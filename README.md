# CPP to OPS converter

## Building
```sh
mkdir build && cd build

PREFIX=../llvm-project/build
BUILD_DIR=../llvm-project/build/bin
```

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/llvm-lit
ninja
```

