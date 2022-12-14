#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

WORKPATH=$scriptDir/../
CGEIST=$WORKPATH/Polygeist/build/bin
MLIR=$WORKPATH/llvm-project/llvm/build-release/bin

echo $1

$CGEIST/cgeist $WORKPATH/ITK_POM2K/pom2k_fun.c -S --function="$1" -print-debug-info | $MLIR/mlir-opt --mlir-print-debuginfo --lower-affine >$WORKPATH/out/temp.mlir
$WORKPATH/build/bin/ops-converter $WORKPATH/out/temp.mlir $WORKPATH/ITK_POM2K/pom2k_fun.c --function=$1 $2 >$WORKPATH/out/$1.json
# $WORKPATH/build/bin/ops-converter $WORKPATH/ITK_POM2K/pom2k_fun.memref.mlir --function=$1 $2 >$WORKPATH/out/$1.json
echo Generated $WORKPATH/out/$1.json
cat $WORKPATH/out/$1.json | ./script/merge_debugs.py
