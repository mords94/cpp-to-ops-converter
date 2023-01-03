#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

WORKPATH=$scriptDir/../
CGEIST=$WORKPATH/Polygeist/build/bin
MLIR=$WORKPATH/llvm-project/llvm/build-release/bin

echo $1

$CGEIST/cgeist $WORKPATH/ITK_POM2K/pom2k_reduced.c -S --function="$1" -print-debug-info | $MLIR/mlir-opt --lower-affine --mlir-print-debuginfo >$WORKPATH/out/test_$1.mlir
$WORKPATH/build/bin/ops-converter $WORKPATH/out/test_$1.mlir $WORKPATH/ITK_POM2K/pom2k_reduced.c --function=$1 $2 >$WORKPATH/out/test_$1.json &&
    code $WORKPATH/out/test_$1.json

# cat $WORKPATH/out/test_$1.json | ./script/merge_debugs.py
