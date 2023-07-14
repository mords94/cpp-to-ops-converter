#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

WORKPATH=$scriptDir/../
CGEIST=$WORKPATH/Polygeist/build/bin
MLIR=$WORKPATH/Polygeist/build/bin

echo $1

# $CGEIST/cgeist $WORKPATH/in/test.c -S --function="$1" -print-debug-info >$WORKPATH/out/test_$1.affine.mlir

$CGEIST/cgeist $WORKPATH/in/test.c -S --function="$1" -print-debug-info | $MLIR/mlir-opt --lower-affine --mlir-print-debuginfo >$WORKPATH/out/test_$1.mlir
$WORKPATH/build/bin/ops-converter $WORKPATH/out/test_$1.mlir $WORKPATH/in/test.c --function=$1 $2 >$WORKPATH/out/test_$1.json
# # code $WORKPATH/out/test_$1.json
echo Generated TEST: $WORKPATH/out/test_$1.json

cat $WORKPATH/out/test_$1.json | ./script/merge_debugs.py
