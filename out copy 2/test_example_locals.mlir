#loc1 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":54:6)
#loc11 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:3)
#loc12 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":55:3)
#loc15 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":59:5)
#loc16 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":56:3)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx13.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @im : memref<1xi32> loc(#loc0)
  memref.global @jm : memref<1xi32> loc(#loc0)
  func.func @example_locals(%arg0: memref<?xf32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":54:6), %arg1: memref<?xf32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":54:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %cst = arith.constant 2.000000e+00 : f64 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %cst_0 = arith.constant 1.000000e-01 : f32 loc(#loc5)
    %cst_1 = arith.constant 0.000000e+00 : f32 loc(#loc6)
    %c0 = arith.constant 0 : index loc(#loc7)
    %0 = memref.get_global @jm : memref<1xi32> loc(#loc8)
    %c0_2 = arith.constant 0 : index loc(#loc8)
    %1 = memref.load %0[%c0_2] : memref<1xi32> loc(#loc8)
    %2 = memref.get_global @im : memref<1xi32> loc(#loc9)
    %3:3 = scf.while (%arg2 = %c0_i32, %arg3 = %cst_0, %arg4 = %cst_1) : (i32, f32, f32) -> (f32, i32, f32) {
      %4 = arith.cmpi slt, %arg2, %1 : i32 loc(#loc10)
      scf.condition(%4) %arg4, %arg2, %arg3 : f32, i32, f32 loc(#loc11)
    } do {
    ^bb0(%arg2: f32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":55:3), %arg3: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:3), %arg4: f32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:3)):
      %c0_4 = arith.constant 0 : index loc(#loc9)
      %4 = memref.load %2[%c0_4] : memref<1xi32> loc(#loc9)
      %5 = arith.muli %arg3, %4 : i32 loc(#loc13)
      %6:3 = scf.while (%arg5 = %c0_i32, %arg6 = %arg4, %arg7 = %arg2) : (i32, f32, f32) -> (f32, f32, i32) {
        %8 = arith.cmpi slt, %arg5, %4 : i32 loc(#loc14)
        scf.condition(%8) %arg6, %arg7, %arg5 : f32, f32, i32 loc(#loc15)
      } do {
      ^bb0(%arg5: f32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":56:3), %arg6: f32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":55:3), %arg7: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":59:5)):
        %8 = arith.addi %arg7, %5 : i32 loc(#loc17)
        %9 = arith.index_cast %8 : i32 to index loc(#loc18)
        %10 = arith.addi %9, %c0 : index loc(#loc19)
        %11 = memref.load %arg1[%10] : memref<?xf32> loc(#loc19)
        %12 = arith.extf %11 : f32 to f64 loc(#loc19)
        %13 = math.absf %12 : f64 loc(#loc20)
        %14 = arith.extf %arg5 : f32 to f64 loc(#loc21)
        %15 = arith.cmpf olt, %13, %14 : f64 loc(#loc22)
        %16:2 = scf.if %15 -> (f32, f32) {
          %18 = memref.load %arg1[%10] : memref<?xf32> loc(#loc24)
          %19 = arith.extf %18 : f32 to f64 loc(#loc24)
          %20 = math.absf %19 : f64 loc(#loc25)
          %21 = arith.truncf %20 : f64 to f32 loc(#loc25)
          %22 = arith.mulf %20, %cst : f64 loc(#loc26)
          %23 = arith.truncf %22 : f64 to f32 loc(#loc27)
          scf.yield %23, %21 : f32, f32 loc(#loc23)
        } else {
          scf.yield %arg5, %arg6 : f32, f32 loc(#loc23)
        } loc(#loc23)
        %17 = arith.addi %arg7, %c1_i32 : i32 loc(#loc2)
        scf.yield %17, %16#0, %16#1 : i32, f32, f32 loc(#loc15)
      } loc(#loc9)
      %7 = arith.addi %arg3, %c1_i32 : i32 loc(#loc28)
      scf.yield %7, %6#0, %6#1 : i32, f32, f32 loc(#loc11)
    } loc(#loc8)
    %c0_3 = arith.constant 0 : index loc(#loc29)
    memref.store %3#0, %arg0[%c0_3] : memref<?xf32> loc(#loc29)
    return loc(#loc30)
  } loc(#loc1)
} loc(#loc0)
#loc0 = loc(unknown)
#loc2 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":59:30)
#loc3 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":62:41)
#loc4 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:16)
#loc5 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":56:18)
#loc6 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":55:17)
#loc7 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":59:10)
#loc8 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:23)
#loc9 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":59:25)
#loc10 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:21)
#loc13 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":5:43)
#loc14 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":59:23)
#loc17 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":5:38)
#loc18 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":60:29)
#loc19 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":60:16)
#loc20 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":60:11)
#loc21 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":60:34)
#loc22 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":60:32)
#loc23 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":60:7)
#loc24 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":61:22)
#loc25 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":61:17)
#loc26 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":62:39)
#loc27 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":62:18)
#loc28 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":58:28)
#loc29 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":67:8)
#loc30 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":68:1)

