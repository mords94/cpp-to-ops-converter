module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx13.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @small : memref<1xf32>
  memref.global @kb : memref<1xi32>
  memref.global @rhoref : memref<1xf32>
  memref.global @grav : memref<1xf32>
  memref.global @sbias : memref<1xf32>
  memref.global @tbias : memref<1xf32>
  memref.global @im : memref<1xi32>
  memref.global @jm : memref<1xi32>
  memref.global @kbm1 : memref<1xi32>
  func.func @ext_profq_original_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xf32>, %arg10: memref<?xf32>, %arg11: memref<?xf32>, %arg12: memref<?xf32>, %arg13: memref<?xf32>, %arg14: memref<?xf32>, %arg15: memref<?xf32>, %arg16: memref<?xf32>, %arg17: memref<?xf32>, %arg18: memref<?xf32>, %arg19: memref<?xf32>, %arg20: memref<?xf32>, %arg21: memref<?xf32>, %arg22: memref<?xf32>, %arg23: memref<?xf32>, %arg24: memref<?xf32>, %arg25: memref<?xf32>, %arg26: memref<?xf32>, %arg27: memref<?xf32>, %arg28: memref<?xf32>, %arg29: memref<?xf32>, %arg30: memref<?xf32>, %arg31: memref<?xf32>, %arg32: memref<?xf32>, %arg33: memref<?xf32>, %arg34: memref<?xf32>, %arg35: memref<?xf32>, %arg36: memref<?xf32>, %arg37: memref<?xf32>, %arg38: memref<?xf32>, %arg39: memref<?xf32>, %arg40: memref<?xf32>, %arg41: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 12.2543993 : f32
    %cst_0 = arith.constant 0.332530111 : f32
    %cst_1 = arith.constant 6.12719965 : f32
    %cst_2 = arith.constant 2.136240e+01 : f32
    %cst_3 = arith.constant 2.242200e+01 : f32
    %cst_4 = arith.constant 7.600000e-01 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant 9.99999974E-5 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 1.660000e+01 : f32
    %cst_8 = arith.constant 7.400000e-01 : f32
    %cst_9 = arith.constant 9.200000e-01 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @kbm1 : memref<1xi32>
    %1 = memref.get_global @jm : memref<1xi32>
    %2 = memref.get_global @im : memref<1xi32>
    %3 = memref.get_global @tbias : memref<1xf32>
    %4 = memref.get_global @sbias : memref<1xf32>
    %5 = memref.get_global @grav : memref<1xf32>
    %6 = memref.get_global @rhoref : memref<1xf32>
    %7 = scf.while (%arg42 = %c0_i32) : (i32) -> i32 {
      %c0_10 = arith.constant 0 : index
      %12 = memref.load %0[%c0_10] : memref<1xi32>
      %13 = arith.cmpi slt, %arg42, %12 : i32
      scf.condition(%13) %arg42 : i32
    } do {
    ^bb0(%arg42: i32):
      %12 = arith.index_cast %arg42 : i32 to index
      %13 = arith.addi %12, %c0 : index
      %14 = scf.while (%arg43 = %c0_i32) : (i32) -> i32 {
        %c0_10 = arith.constant 0 : index
        %16 = memref.load %1[%c0_10] : memref<1xi32>
        %17 = arith.cmpi slt, %arg43, %16 : i32
        scf.condition(%17) %arg43 : i32
      } do {
      ^bb0(%arg43: i32):
        %16:2 = scf.while (%arg44 = %c0_i32) : (i32) -> (i32, i32) {
          %c0_10 = arith.constant 0 : index
          %18 = memref.load %2[%c0_10] : memref<1xi32>
          %19 = arith.cmpi slt, %arg44, %18 : i32
          scf.condition(%19) %18, %arg44 : i32, i32
        } do {
        ^bb0(%arg44: i32, %arg45: i32):
          %18 = arith.muli %arg43, %arg44 : i32
          %19 = arith.addi %arg45, %18 : i32
          %20 = arith.muli %arg42, %arg44 : i32
          %c0_10 = arith.constant 0 : index
          %21 = memref.load %1[%c0_10] : memref<1xi32>
          %22 = arith.muli %20, %21 : i32
          %23 = arith.addi %19, %22 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = arith.addi %24, %c0 : index
          %26 = memref.load %arg18[%25] : memref<?xf32>
          %c0_11 = arith.constant 0 : index
          %27 = memref.load %3[%c0_11] : memref<1xf32>
          %28 = arith.addf %26, %27 : f32
          %29 = memref.load %arg19[%25] : memref<?xf32>
          %c0_12 = arith.constant 0 : index
          %30 = memref.load %4[%c0_12] : memref<1xf32>
          %31 = arith.addf %29, %30 : f32
          %c0_13 = arith.constant 0 : index
          %32 = memref.load %5[%c0_13] : memref<1xf32>
          %c0_14 = arith.constant 0 : index
          %33 = memref.load %6[%c0_14] : memref<1xf32>
          %34 = arith.mulf %32, %33 : f32
          %35 = memref.load %arg20[%13] : memref<?xf32>
          %36 = arith.negf %35 : f32
          %37 = arith.index_cast %19 : i32 to index
          %38 = arith.addi %37, %c0 : index
          %39 = memref.load %arg4[%38] : memref<?xf32>
          %40 = arith.mulf %36, %39 : f32
          %41 = arith.mulf %34, %40 : f32
          %42 = arith.mulf %41, %cst_5 : f32
          %43 = arith.mulf %31, %28 : f32
          %44 = arith.mulf %43, %42 : f32
          memref.store %44, %arg3[%25] : memref<?xf32>
          %45 = arith.addi %arg45, %c1_i32 : i32
          scf.yield %45 : i32
        } attributes {arg1 = "for (int i = 0; i < im; i++) {", arg3 = "for (int i = 0; i < im; i++) {", depth = 3 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 0 : i32, step = 1 : i8, ub = @im}
        %17 = arith.addi %arg43, %c1_i32 : i32
        scf.yield %17 : i32
      } attributes {arg1 = "for (int j = 0; j < jm; j++) {", depth = 2 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 1 : i32, step = 1 : i8, ub = @jm}
      %15 = arith.addi %arg42, %c1_i32 : i32
      scf.yield %15 : i32
    } attributes {arg1 = "for (int k = 0; k < kbm1; k++) {", depth = 1 : i8, induction_variable = "k", lb = 0 : i32, loop_index = 2 : i32, step = 1 : i8, ub = @kbm1}
    %8 = memref.get_global @kb : memref<1xi32>
    %9 = memref.get_global @small : memref<1xf32>
    %10 = scf.while (%arg42 = %c0_i32) : (i32) -> i32 {
      %c0_10 = arith.constant 0 : index
      %12 = memref.load %8[%c0_10] : memref<1xi32>
      %13 = arith.cmpi slt, %arg42, %12 : i32
      scf.condition(%13) %arg42 : i32
    } do {
    ^bb0(%arg42: i32):
      %12 = scf.while (%arg43 = %c0_i32) : (i32) -> i32 {
        %c0_10 = arith.constant 0 : index
        %14 = memref.load %1[%c0_10] : memref<1xi32>
        %15 = arith.cmpi slt, %arg43, %14 : i32
        scf.condition(%15) %arg43 : i32
      } do {
      ^bb0(%arg43: i32):
        %14:2 = scf.while (%arg44 = %c0_i32) : (i32) -> (i32, i32) {
          %c0_10 = arith.constant 0 : index
          %16 = memref.load %2[%c0_10] : memref<1xi32>
          %17 = arith.cmpi slt, %arg44, %16 : i32
          scf.condition(%17) %16, %arg44 : i32, i32
        } do {
        ^bb0(%arg44: i32, %arg45: i32):
          %16 = arith.muli %arg43, %arg44 : i32
          %17 = arith.addi %arg45, %16 : i32
          %18 = arith.muli %arg42, %arg44 : i32
          %c0_10 = arith.constant 0 : index
          %19 = memref.load %1[%c0_10] : memref<1xi32>
          %20 = arith.muli %18, %19 : i32
          %21 = arith.addi %17, %20 : i32
          %22 = arith.index_cast %21 : i32 to index
          %23 = arith.addi %22, %c0 : index
          memref.store %cst_6, %arg38[%23] : memref<?xf32>
          %c0_11 = arith.constant 0 : index
          %24 = memref.load %2[%c0_11] : memref<1xi32>
          %25 = arith.muli %arg43, %24 : i32
          %26 = arith.addi %arg45, %25 : i32
          %27 = arith.muli %arg42, %24 : i32
          %c0_12 = arith.constant 0 : index
          %28 = memref.load %1[%c0_12] : memref<1xi32>
          %29 = arith.muli %27, %28 : i32
          %30 = arith.addi %26, %29 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.addi %31, %c0 : index
          %33 = memref.load %arg21[%32] : memref<?xf32>
          %34 = math.absf %33 : f32
          %35 = math.sqrt %34 : f32
          %36 = memref.load %arg38[%32] : memref<?xf32>
          %37 = arith.mulf %35, %36 : f32
          %38 = memref.load %arg23[%32] : memref<?xf32>
          %39 = arith.mulf %38, %cst_7 : f32
          %c0_13 = arith.constant 0 : index
          %40 = memref.load %9[%c0_13] : memref<1xf32>
          %41 = arith.addf %39, %40 : f32
          %42 = arith.divf %37, %41 : f32
          memref.store %42, %arg34[%32] : memref<?xf32>
          %43 = arith.addi %arg45, %c1_i32 : i32
          scf.yield %43 : i32
        } attributes {arg1 = "for (int i = 0; i < im; i++) {", arg3 = "for (int i = 0; i < im; i++) {", depth = 3 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 3 : i32, step = 1 : i8, ub = @im}
        %15 = arith.addi %arg43, %c1_i32 : i32
        scf.yield %15 : i32
      } attributes {arg1 = "for (int j = 0; j < jm; j++) {", depth = 2 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 4 : i32, step = 1 : i8, ub = @jm}
      %13 = arith.addi %arg42, %c1_i32 : i32
      scf.yield %13 : i32
    } attributes {arg1 = "for (int k = 0; k < kb; k++) {", depth = 1 : i8, induction_variable = "k", lb = 0 : i32, loop_index = 5 : i32, step = 1 : i8, ub = @kb}
    %11 = scf.while (%arg42 = %c0_i32) : (i32) -> i32 {
      %c0_10 = arith.constant 0 : index
      %12 = memref.load %8[%c0_10] : memref<1xi32>
      %13 = arith.cmpi slt, %arg42, %12 : i32
      scf.condition(%13) %arg42 : i32
    } do {
    ^bb0(%arg42: i32):
      %12 = scf.while (%arg43 = %c0_i32) : (i32) -> i32 {
        %c0_10 = arith.constant 0 : index
        %14 = memref.load %1[%c0_10] : memref<1xi32>
        %15 = arith.cmpi slt, %arg43, %14 : i32
        scf.condition(%15) %arg43 : i32
      } do {
      ^bb0(%arg43: i32):
        %14:2 = scf.while (%arg44 = %c0_i32) : (i32) -> (i32, i32) {
          %c0_10 = arith.constant 0 : index
          %16 = memref.load %2[%c0_10] : memref<1xi32>
          %17 = arith.cmpi slt, %arg44, %16 : i32
          scf.condition(%17) %16, %arg44 : i32, i32
        } do {
        ^bb0(%arg44: i32, %arg45: i32):
          %16 = arith.muli %arg43, %arg44 : i32
          %17 = arith.addi %arg45, %16 : i32
          %18 = arith.muli %arg42, %arg44 : i32
          %c0_10 = arith.constant 0 : index
          %19 = memref.load %1[%c0_10] : memref<1xi32>
          %20 = arith.muli %18, %19 : i32
          %21 = arith.addi %17, %20 : i32
          %22 = arith.index_cast %21 : i32 to index
          %23 = arith.addi %22, %c0 : index
          %24 = memref.load %arg38[%23] : memref<?xf32>
          %25 = arith.mulf %24, %cst_0 : f32
          %26 = arith.subf %cst_6, %25 : f32
          %27 = arith.mulf %26, %cst_8 : f32
          %28 = arith.divf %cst_3, %24 : f32
          %29 = arith.addf %28, %cst : f32
          %30 = arith.subf %cst_4, %25 : f32
          %31 = arith.mulf %30, %cst_9 : f32
          %32 = memref.load %arg36[%23] : memref<?xf32>
          %33 = arith.mulf %29, %32 : f32
          %34 = arith.subf %cst_6, %33 : f32
          %35 = arith.divf %27, %34 : f32
          memref.store %35, %arg1[%23] : memref<?xf32>
          %c0_11 = arith.constant 0 : index
          %36 = memref.load %2[%c0_11] : memref<1xi32>
          %37 = arith.muli %arg43, %36 : i32
          %38 = arith.addi %arg45, %37 : i32
          %39 = arith.muli %arg42, %36 : i32
          %c0_12 = arith.constant 0 : index
          %40 = memref.load %1[%c0_12] : memref<1xi32>
          %41 = arith.muli %39, %40 : i32
          %42 = arith.addi %38, %41 : i32
          %43 = arith.index_cast %42 : i32 to index
          %44 = arith.addi %43, %c0 : index
          %45 = memref.load %arg1[%44] : memref<?xf32>
          %46 = arith.mulf %45, %cst_2 : f32
          %47 = memref.load %arg36[%44] : memref<?xf32>
          %48 = arith.mulf %46, %47 : f32
          %49 = arith.addf %31, %48 : f32
          memref.store %49, %arg0[%44] : memref<?xf32>
          %c0_13 = arith.constant 0 : index
          %50 = memref.load %2[%c0_13] : memref<1xi32>
          %51 = arith.muli %arg43, %50 : i32
          %52 = arith.addi %arg45, %51 : i32
          %53 = arith.muli %arg42, %50 : i32
          %c0_14 = arith.constant 0 : index
          %54 = memref.load %1[%c0_14] : memref<1xi32>
          %55 = arith.muli %53, %54 : i32
          %56 = arith.addi %52, %55 : i32
          %57 = arith.index_cast %56 : i32 to index
          %58 = arith.addi %57, %c0 : index
          %59 = memref.load %arg0[%58] : memref<?xf32>
          %60 = memref.load %arg36[%58] : memref<?xf32>
          %61 = arith.mulf %60, %cst_1 : f32
          %62 = arith.subf %cst_6, %61 : f32
          %63 = arith.divf %59, %62 : f32
          memref.store %63, %arg0[%58] : memref<?xf32>
          %64 = arith.addi %arg45, %c1_i32 : i32
          scf.yield %64 : i32
        } attributes {arg1 = "for (int i = 0; i < im; i++) {", arg3 = "for (int i = 0; i < im; i++) {", depth = 3 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 6 : i32, step = 1 : i8, ub = @im}
        %15 = arith.addi %arg43, %c1_i32 : i32
        scf.yield %15 : i32
      } attributes {arg1 = "for (int j = 0; j < jm; j++) {", depth = 2 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 7 : i32, step = 1 : i8, ub = @jm}
      %13 = arith.addi %arg42, %c1_i32 : i32
      scf.yield %13 : i32
    } attributes {arg1 = "for (int k = 0; k < kb; k++) {", depth = 1 : i8, induction_variable = "k", lb = 0 : i32, loop_index = 8 : i32, step = 1 : i8, ub = @kb}
    return
  }
}
