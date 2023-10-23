#loc1 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":95:6)
#loc12 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":103:3)
#loc14 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":104:5)
#loc15 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":97:3)
#loc17 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:7)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx14.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @im : memref<1xi32> loc(#loc0)
  memref.global @jm : memref<1xi32> loc(#loc0)
  memref.global @kb : memref<1xi32> loc(#loc0)
  func.func @red1(%arg0: memref<?xf32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":95:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %cst = arith.constant 5.000000e-01 : f32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %cst_0 = arith.constant 1.100000e+01 : f32 loc(#loc5)
    %c0 = arith.constant 0 : index loc(#loc6)
    %0 = memref.alloca() : memref<memref<?xf32>> loc(#loc7)
    %1 = memref.get_global @kb : memref<1xi32> loc(#loc8)
    %2 = memref.get_global @jm : memref<1xi32> loc(#loc9)
    %3 = memref.get_global @im : memref<1xi32> loc(#loc10)
    %4:2 = scf.while (%arg1 = %c0_i32, %arg2 = %c0_i32) : (i32, i32) -> (i32, i32) {
      %c0_1 = arith.constant 0 : index loc(#loc8)
      %5 = memref.load %1[%c0_1] : memref<1xi32> loc(#loc8)
      %6 = arith.cmpi slt, %arg1, %5 : i32 loc(#loc11)
      scf.condition(%6) %arg1, %arg2 : i32, i32 loc(#loc12)
    } do {
    ^bb0(%arg1: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":103:3), %arg2: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":103:3)):
      %5:2 = scf.while (%arg3 = %c0_i32, %arg4 = %arg2) : (i32, i32) -> (i32, i32) {
        %c0_1 = arith.constant 0 : index loc(#loc9)
        %7 = memref.load %2[%c0_1] : memref<1xi32> loc(#loc9)
        %8 = arith.cmpi slt, %arg3, %7 : i32 loc(#loc13)
        scf.condition(%8) %arg4, %arg3 : i32, i32 loc(#loc14)
      } do {
      ^bb0(%arg3: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":97:3), %arg4: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":104:5)):
        %7:3 = scf.while (%arg5 = %c0_i32, %arg6 = %arg3) : (i32, i32) -> (i32, i32, i32) {
          %c0_1 = arith.constant 0 : index loc(#loc10)
          %9 = memref.load %3[%c0_1] : memref<1xi32> loc(#loc10)
          %10 = arith.cmpi slt, %arg5, %9 : i32 loc(#loc16)
          scf.condition(%10) %arg6, %9, %arg5 : i32, i32, i32 loc(#loc17)
        } do {
        ^bb0(%arg5: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":97:3), %arg6: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:7), %arg7: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:7)):
          %9 = memref.load %0[] : memref<memref<?xf32>> loc(#loc18)
          %10 = arith.muli %arg4, %arg6 : i32 loc(#loc19)
          %11 = arith.addi %arg7, %10 : i32 loc(#loc20)
          %12 = arith.muli %arg1, %arg6 : i32 loc(#loc21)
          %c0_1 = arith.constant 0 : index loc(#loc22)
          %13 = memref.load %2[%c0_1] : memref<1xi32> loc(#loc22)
          %14 = arith.muli %12, %13 : i32 loc(#loc23)
          %15 = arith.addi %11, %14 : i32 loc(#loc24)
          %16 = arith.index_cast %15 : i32 to index loc(#loc25)
          %17 = arith.sitofp %arg5 : i32 to f32 loc(#loc26)
          %18 = arith.addi %16, %c0 : index loc(#loc27)
          %19 = memref.load %arg0[%18] : memref<?xf32> loc(#loc27)
          %20 = arith.addf %17, %19 : f32 loc(#loc28)
          memref.store %20, %9[%18] : memref<?xf32> loc(#loc29)
          %c0_2 = arith.constant 0 : index loc(#loc30)
          %21 = memref.load %3[%c0_2] : memref<1xi32> loc(#loc30)
          %22 = arith.muli %arg4, %21 : i32 loc(#loc19)
          %23 = arith.addi %arg7, %22 : i32 loc(#loc20)
          %24 = arith.muli %arg1, %21 : i32 loc(#loc21)
          %c0_3 = arith.constant 0 : index loc(#loc22)
          %25 = memref.load %2[%c0_3] : memref<1xi32> loc(#loc22)
          %26 = arith.muli %24, %25 : i32 loc(#loc23)
          %27 = arith.addi %23, %26 : i32 loc(#loc24)
          %28 = arith.index_cast %27 : i32 to index loc(#loc31)
          %29 = arith.addi %28, %c0 : index loc(#loc32)
          %30 = memref.load %9[%29] : memref<?xf32> loc(#loc32)
          %31 = arith.subf %cst_0, %30 : f32 loc(#loc33)
          %32 = arith.fptosi %31 : f32 to i32 loc(#loc34)
          %33 = func.call @abs(%32) : (i32) -> i32 loc(#loc35)
          %34 = arith.sitofp %33 : i32 to f32 loc(#loc35)
          %35 = arith.addf %34, %cst : f32 loc(#loc36)
          %36 = arith.addf %35, %17 : f32 loc(#loc37)
          %37 = arith.fptosi %36 : f32 to i32 loc(#loc38)
          %38 = arith.addi %arg7, %c1_i32 : i32 loc(#loc2)
          scf.yield %38, %37 : i32, i32 loc(#loc17)
        } loc(#loc10)
        %8 = arith.addi %arg4, %c1_i32 : i32 loc(#loc39)
        scf.yield %8, %7#0 : i32, i32 loc(#loc14)
      } loc(#loc9)
      %6 = arith.addi %arg1, %c1_i32 : i32 loc(#loc40)
      scf.yield %6, %5#0 : i32, i32 loc(#loc12)
    } loc(#loc8)
    return loc(#loc41)
  } loc(#loc1)
  func.func private @abs(i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} loc(#loc35)
} loc(#loc0)
#loc0 = loc(unknown)
#loc2 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:32)
#loc3 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":108:17)
#loc4 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":97:11)
#loc5 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":96:16)
#loc6 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:12)
#loc7 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":4:16)
#loc8 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":103:23)
#loc9 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":104:25)
#loc10 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:27)
#loc11 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":103:21)
#loc13 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":104:23)
#loc16 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":105:25)
#loc18 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":106:9)
#loc19 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":7:50)
#loc20 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":7:45)
#loc21 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":7:59)
#loc22 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":8:46)
#loc23 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":7:63)
#loc24 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":7:54)
#loc25 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":106:27)
#loc26 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":106:31)
#loc27 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":106:35)
#loc28 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":106:33)
#loc29 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":106:29)
#loc30 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/pom2k_c_header.h":8:42)
#loc31 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":107:41)
#loc32 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":107:23)
#loc33 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":107:21)
#loc34 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":107:17)
#loc35 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":107:13)
#loc36 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":108:15)
#loc37 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":108:22)
#loc38 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":108:13)
#loc39 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":104:30)
#loc40 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":103:28)
#loc41 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//in/test.c":112:1)

