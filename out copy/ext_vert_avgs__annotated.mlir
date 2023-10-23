module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx13.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @kbm1 : memref<1xi32>
  memref.global @im : memref<1xi32>
  memref.global @jm : memref<1xi32>
  func.func @ext_vert_avgs_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xf32>, %arg10: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @jm : memref<1xi32>
    %1 = memref.get_global @im : memref<1xi32>
    %2 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
      %c0_0 = arith.constant 0 : index
      %5 = memref.load %0[%c0_0] : memref<1xi32>
      %6 = arith.cmpi slt, %arg11, %5 : i32
      scf.condition(%6) %arg11 : i32
    } do {
    ^bb0(%arg11: i32):
      %5:2 = scf.while (%arg12 = %c0_i32) : (i32) -> (i32, i32) {
        %c0_0 = arith.constant 0 : index
        %7 = memref.load %1[%c0_0] : memref<1xi32>
        %8 = arith.cmpi slt, %arg12, %7 : i32
        scf.condition(%8) %7, %arg12 : i32, i32
      } do {
      ^bb0(%arg12: i32, %arg13: i32):
        %7 = arith.muli %arg11, %arg12 : i32
        %8 = arith.addi %arg13, %7 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = arith.addi %9, %c0 : index
        memref.store %cst, %arg0[%10] : memref<?xf32>
        %c0_0 = arith.constant 0 : index
        %11 = memref.load %1[%c0_0] : memref<1xi32>
        %12 = arith.muli %arg11, %11 : i32
        %13 = arith.addi %arg13, %12 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = arith.addi %14, %c0 : index
        memref.store %cst, %arg1[%15] : memref<?xf32>
        %c0_1 = arith.constant 0 : index
        %16 = memref.load %1[%c0_1] : memref<1xi32>
        %17 = arith.muli %arg11, %16 : i32
        %18 = arith.addi %arg13, %17 : i32
        %19 = arith.index_cast %18 : i32 to index
        %20 = arith.addi %19, %c0 : index
        memref.store %cst, %arg2[%20] : memref<?xf32>
        %c0_2 = arith.constant 0 : index
        %21 = memref.load %1[%c0_2] : memref<1xi32>
        %22 = arith.muli %arg11, %21 : i32
        %23 = arith.addi %arg13, %22 : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = arith.addi %24, %c0 : index
        memref.store %cst, %arg3[%25] : memref<?xf32>
        %c0_3 = arith.constant 0 : index
        %26 = memref.load %1[%c0_3] : memref<1xi32>
        %27 = arith.muli %arg11, %26 : i32
        %28 = arith.addi %arg13, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.addi %29, %c0 : index
        memref.store %cst, %arg4[%30] : memref<?xf32>
        %31 = arith.addi %arg13, %c1_i32 : i32
        scf.yield %31 : i32
      } attributes {depth = 2 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 0 : i32, ub = @im}
      %6 = arith.addi %arg11, %c1_i32 : i32
      scf.yield %6 : i32
    } attributes {depth = 1 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 1 : i32, ub = @jm}
    %3 = memref.get_global @kbm1 : memref<1xi32>
    %4 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
      %c0_0 = arith.constant 0 : index
      %5 = memref.load %3[%c0_0] : memref<1xi32>
      %6 = arith.cmpi slt, %arg11, %5 : i32
      scf.condition(%6) %arg11 : i32
    } do {
    ^bb0(%arg11: i32):
      %5 = arith.index_cast %arg11 : i32 to index
      %6 = arith.addi %5, %c0 : index
      %7 = scf.while (%arg12 = %c0_i32) : (i32) -> i32 {
        %c0_0 = arith.constant 0 : index
        %9 = memref.load %0[%c0_0] : memref<1xi32>
        %10 = arith.cmpi slt, %arg12, %9 : i32
        scf.condition(%10) %arg12 : i32
      } do {
      ^bb0(%arg12: i32):
        %9:2 = scf.while (%arg13 = %c0_i32) : (i32) -> (i32, i32) {
          %c0_0 = arith.constant 0 : index
          %11 = memref.load %1[%c0_0] : memref<1xi32>
          %12 = arith.cmpi slt, %arg13, %11 : i32
          scf.condition(%12) %11, %arg13 : i32, i32
        } do {
        ^bb0(%arg13: i32, %arg14: i32):
          %11 = arith.muli %arg12, %arg13 : i32
          %12 = arith.addi %arg14, %11 : i32
          %13 = arith.index_cast %12 : i32 to index
          %14 = arith.muli %arg11, %arg13 : i32
          %c0_0 = arith.constant 0 : index
          %15 = memref.load %0[%c0_0] : memref<1xi32>
          %16 = arith.muli %14, %15 : i32
          %17 = arith.addi %12, %16 : i32
          %18 = arith.index_cast %17 : i32 to index
          %19 = arith.addi %18, %c0 : index
          %20 = memref.load %arg5[%19] : memref<?xf32>
          %21 = memref.load %arg10[%6] : memref<?xf32>
          %22 = arith.mulf %20, %21 : f32
          %23 = arith.addi %13, %c0 : index
          %24 = memref.load %arg0[%23] : memref<?xf32>
          %25 = arith.addf %24, %22 : f32
          memref.store %25, %arg0[%23] : memref<?xf32>
          %c0_1 = arith.constant 0 : index
          %26 = memref.load %1[%c0_1] : memref<1xi32>
          %27 = arith.muli %arg12, %26 : i32
          %28 = arith.addi %arg14, %27 : i32
          %29 = arith.index_cast %28 : i32 to index
          %30 = arith.muli %arg11, %26 : i32
          %c0_2 = arith.constant 0 : index
          %31 = memref.load %0[%c0_2] : memref<1xi32>
          %32 = arith.muli %30, %31 : i32
          %33 = arith.addi %28, %32 : i32
          %34 = arith.index_cast %33 : i32 to index
          %35 = arith.addi %34, %c0 : index
          %36 = memref.load %arg6[%35] : memref<?xf32>
          %37 = memref.load %arg10[%6] : memref<?xf32>
          %38 = arith.mulf %36, %37 : f32
          %39 = arith.addi %29, %c0 : index
          %40 = memref.load %arg1[%39] : memref<?xf32>
          %41 = arith.addf %40, %38 : f32
          memref.store %41, %arg1[%39] : memref<?xf32>
          %c0_3 = arith.constant 0 : index
          %42 = memref.load %1[%c0_3] : memref<1xi32>
          %43 = arith.muli %arg12, %42 : i32
          %44 = arith.addi %arg14, %43 : i32
          %45 = arith.index_cast %44 : i32 to index
          %46 = arith.muli %arg11, %42 : i32
          %c0_4 = arith.constant 0 : index
          %47 = memref.load %0[%c0_4] : memref<1xi32>
          %48 = arith.muli %46, %47 : i32
          %49 = arith.addi %44, %48 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.addi %50, %c0 : index
          %52 = memref.load %arg7[%51] : memref<?xf32>
          %53 = memref.load %arg10[%6] : memref<?xf32>
          %54 = arith.mulf %52, %53 : f32
          %55 = arith.addi %45, %c0 : index
          %56 = memref.load %arg2[%55] : memref<?xf32>
          %57 = arith.addf %56, %54 : f32
          memref.store %57, %arg2[%55] : memref<?xf32>
          %c0_5 = arith.constant 0 : index
          %58 = memref.load %1[%c0_5] : memref<1xi32>
          %59 = arith.muli %arg12, %58 : i32
          %60 = arith.addi %arg14, %59 : i32
          %61 = arith.index_cast %60 : i32 to index
          %62 = arith.muli %arg11, %58 : i32
          %c0_6 = arith.constant 0 : index
          %63 = memref.load %0[%c0_6] : memref<1xi32>
          %64 = arith.muli %62, %63 : i32
          %65 = arith.addi %60, %64 : i32
          %66 = arith.index_cast %65 : i32 to index
          %67 = arith.addi %66, %c0 : index
          %68 = memref.load %arg8[%67] : memref<?xf32>
          %69 = memref.load %arg10[%6] : memref<?xf32>
          %70 = arith.mulf %68, %69 : f32
          %71 = arith.addi %61, %c0 : index
          %72 = memref.load %arg3[%71] : memref<?xf32>
          %73 = arith.addf %72, %70 : f32
          memref.store %73, %arg3[%71] : memref<?xf32>
          %c0_7 = arith.constant 0 : index
          %74 = memref.load %1[%c0_7] : memref<1xi32>
          %75 = arith.muli %arg12, %74 : i32
          %76 = arith.addi %arg14, %75 : i32
          %77 = arith.index_cast %76 : i32 to index
          %78 = arith.muli %arg11, %74 : i32
          %c0_8 = arith.constant 0 : index
          %79 = memref.load %0[%c0_8] : memref<1xi32>
          %80 = arith.muli %78, %79 : i32
          %81 = arith.addi %76, %80 : i32
          %82 = arith.index_cast %81 : i32 to index
          %83 = arith.addi %82, %c0 : index
          %84 = memref.load %arg9[%83] : memref<?xf32>
          %85 = memref.load %arg10[%6] : memref<?xf32>
          %86 = arith.mulf %84, %85 : f32
          %87 = arith.addi %77, %c0 : index
          %88 = memref.load %arg4[%87] : memref<?xf32>
          %89 = arith.addf %88, %86 : f32
          memref.store %89, %arg4[%87] : memref<?xf32>
          %90 = arith.addi %arg14, %c1_i32 : i32
          scf.yield %90 : i32
        } attributes {depth = 3 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 2 : i32, ub = @im}
        %10 = arith.addi %arg12, %c1_i32 : i32
        scf.yield %10 : i32
      } attributes {depth = 2 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 3 : i32, ub = @jm}
      %8 = arith.addi %arg11, %c1_i32 : i32
      scf.yield %8 : i32
    } attributes {depth = 1 : i8, induction_variable = "k", lb = 0 : i32, loop_index = 4 : i32, ub = @kbm1}
    return
  }
}
