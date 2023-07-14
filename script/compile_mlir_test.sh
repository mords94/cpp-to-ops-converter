#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

WORKPATH=$scriptDir/../
CGEIST=$WORKPATH/Polygeist/build/bin
MLIR=$WORKPATH/Polygeist/build/bin

echo $1

$CGEIST/cgeist $WORKPATH/in/test.c -S --function="$1" -print-debug-info | $MLIR/mlir-opt --mlir-print-debuginfo --lower-affine >$WORKPATH/out/temp.mlir
$WORKPATH/build/bin/ops-converter $WORKPATH/out/temp.mlir $WORKPATH/in/test.c --function=$1 $2 --opconv-emit-mlir >$WORKPATH/out/test_$1_annotated.mlir

echo Generated $WORKPATH/out/test_$1_annotated.mlir
