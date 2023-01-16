#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

WORKPATH=$scriptDir/../
CGEIST=$WORKPATH/Polygeist/build/bin
MLIR=$WORKPATH/llvm-project/llvm/build-release/bin

$CGEIST/cgeist $WORKPATH/ITK_POM2K/pom2k_fun.c -S --function=* -print-debug-info | $MLIR/mlir-opt -allow-unregistered-dialect --mlir-print-debuginfo --lower-affine >$WORKPATH/ITK_POM2K/pom2k_fun.memref.mlir

echo Generated $WORKPATH/ITK_POM2K/pom2k_fun.memref.mlir
