#loc1 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1617:6)
#loc9 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1620:3)
#loc10 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1619:3)
#loc11 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_c_header.h":4:16)
#loc14 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1621:5)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx13.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @im : memref<1xi32> loc(#loc0)
  memref.global @jm : memref<1xi32> loc(#loc0)
  func.func @ext_comp_vamax_(%arg0: memref<?xf32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1617:6), %arg1: memref<?xf32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1617:6), %arg2: memref<?xi32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1617:6), %arg3: memref<?xi32> loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1617:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc3)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc4)
    %c0 = arith.constant 0 : index loc(#loc5)
    %0 = llvm.mlir.undef : i32 loc(#loc5)
    %1 = memref.get_global @jm : memref<1xi32> loc(#loc6)
    %c0_0 = arith.constant 0 : index loc(#loc6)
    %2 = memref.load %1[%c0_0] : memref<1xi32> loc(#loc6)
    %3 = memref.get_global @im : memref<1xi32> loc(#loc7)
    %4:4 = scf.while (%arg4 = %c0_i32, %arg5 = %0, %arg6 = %0, %arg7 = %cst) : (i32, i32, i32, f32) -> (i32, i32, f32, i32) {
      %5 = arith.cmpi slt, %arg4, %2 : i32 loc(#loc8)
      scf.condition(%5) %arg5, %arg6, %arg7, %arg4 : i32, i32, f32, i32 loc(#loc9)
    } do {
    ^bb0(%arg4: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1619:3), %arg5: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1619:3), %arg6: f32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_c_header.h":4:16), %arg7: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1620:3)):
      %c0_4 = arith.constant 0 : index loc(#loc7)
      %5 = memref.load %3[%c0_4] : memref<1xi32> loc(#loc7)
      %6 = arith.muli %arg7, %5 : i32 loc(#loc12)
      %7:4 = scf.while (%arg8 = %c0_i32, %arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6) : (i32, i32, i32, f32) -> (i32, i32, f32, i32) {
        %9 = arith.cmpi slt, %arg8, %5 : i32 loc(#loc13)
        scf.condition(%9) %arg9, %arg10, %arg11, %arg8 : i32, i32, f32, i32 loc(#loc14)
      } do {
      ^bb0(%arg8: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1619:3), %arg9: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1619:3), %arg10: f32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_c_header.h":4:16), %arg11: i32 loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1621:5)):
        %9 = arith.addi %arg11, %6 : i32 loc(#loc15)
        %10 = arith.index_cast %9 : i32 to index loc(#loc16)
        %11 = arith.addi %10, %c0 : index loc(#loc17)
        %12 = memref.load %arg1[%11] : memref<?xf32> loc(#loc17)
        %13 = arith.extf %12 : f32 to f64 loc(#loc17)
        %14 = math.absf %13 : f64 loc(#loc18)
        %15 = arith.extf %arg10 : f32 to f64 loc(#loc19)
        %16 = arith.cmpf ogt, %14, %15 : f64 loc(#loc20)
        %17 = arith.select %16, %arg7, %arg8 : i32 loc(#loc21)
        %18 = arith.select %16, %arg11, %arg9 : i32 loc(#loc21)
        %19 = scf.if %16 -> (f32) {
          %21 = memref.load %arg1[%11] : memref<?xf32> loc(#loc22)
          %22 = arith.extf %21 : f32 to f64 loc(#loc22)
          %23 = math.absf %22 : f64 loc(#loc23)
          %24 = arith.truncf %23 : f64 to f32 loc(#loc23)
          scf.yield %24 : f32 loc(#loc21)
        } else {
          scf.yield %arg10 : f32 loc(#loc21)
        } loc(#loc21)
        %20 = arith.addi %arg11, %c1_i32 : i32 loc(#loc2)
        scf.yield %20, %17, %18, %19 : i32, i32, i32, f32 loc(#loc14)
      } loc(#loc7)
      %8 = arith.addi %arg7, %c1_i32 : i32 loc(#loc24)
      scf.yield %8, %7#0, %7#1, %7#2 : i32, i32, i32, f32 loc(#loc9)
    } loc(#loc6)
    %c0_1 = arith.constant 0 : index loc(#loc25)
    memref.store %4#2, %arg0[%c0_1] : memref<?xf32> loc(#loc25)
    %c0_2 = arith.constant 0 : index loc(#loc26)
    memref.store %4#1, %arg2[%c0_2] : memref<?xi32> loc(#loc26)
    %c0_3 = arith.constant 0 : index loc(#loc27)
    memref.store %4#0, %arg3[%c0_3] : memref<?xi32> loc(#loc27)
    return loc(#loc28)
  } loc(#loc1)
} loc(#loc0)
#loc0 = loc(unknown)
#loc2 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1621:30)
#loc3 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1620:16)
#loc4 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1618:18)
#loc5 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1621:10)
#loc6 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1620:23)
#loc7 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1621:25)
#loc8 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1620:21)
#loc12 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_c_header.h":5:43)
#loc13 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1621:23)
#loc15 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_c_header.h":5:38)
#loc16 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1622:30)
#loc17 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1622:16)
#loc18 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1622:11)
#loc19 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1622:35)
#loc20 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1622:33)
#loc21 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1622:7)
#loc22 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1623:22)
#loc23 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1623:17)
#loc24 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1620:28)
#loc25 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1629:11)
#loc26 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1630:10)
#loc27 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1631:10)
#loc28 = loc("/Users/mac/Projects/mlir/cpp-to-ops-converter/script/..//ITK_POM2K/pom2k_fun.c":1632:1)
