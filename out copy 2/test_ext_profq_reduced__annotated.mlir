module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx13.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @kb : memref<1xi32>
  memref.global @grav : memref<1xf32>
  memref.global @jm : memref<1xi32>
  memref.global @im : memref<1xi32>
  memref.global @imm1 : memref<1xi32>
  memref.global @jmm1 : memref<1xi32>
  func.func @ext_profq_reduced_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xf32>, %arg10: memref<?xf32>, %arg11: memref<?xf32>, %arg12: memref<?xf32>, %arg13: memref<?xf32>, %arg14: memref<?xf32>, %arg15: memref<?xf32>, %arg16: memref<?xf32>, %arg17: memref<?xf32>, %arg18: memref<?xf32>, %arg19: memref<?xf32>, %arg20: memref<?xf32>, %arg21: memref<?xf32>, %arg22: memref<?xf32>, %arg23: memref<?xf32>, %arg24: memref<?xf32>, %arg25: memref<?xf32>, %arg26: memref<?xf32>, %arg27: memref<?xf32>, %arg28: memref<?xf32>, %arg29: memref<?xf32>, %arg30: memref<?xf32>, %arg31: memref<?xf32>, %arg32: memref<?xf32>, %arg33: memref<?xf32>, %arg34: memref<?xf32>, %arg35: memref<?xf32>, %arg36: memref<?xf32>, %arg37: memref<?xf32>, %arg38: memref<?xf32>, %arg39: memref<?xf32>, %arg40: memref<?xf32>, %arg41: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 11.5635948 : f32
    %cst_0 = arith.constant 135.655716 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+05 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @jmm1 : memref<1xi32>
    %1 = memref.get_global @imm1 : memref<1xi32>
    %2 = memref.get_global @im : memref<1xi32>
    %3 = memref.get_global @jm : memref<1xi32>
    %4 = memref.get_global @grav : memref<1xf32>
    %5 = memref.get_global @kb : memref<1xi32>
    %6 = scf.while (%arg42 = %c0_i32) : (i32) -> i32 {
      %c0_4 = arith.constant 0 : index
      %7 = memref.load %0[%c0_4] : memref<1xi32>
      %8 = arith.cmpi slt, %arg42, %7 : i32
      scf.condition(%8) %arg42 : i32
    } do {
    ^bb0(%arg42: i32):
      %7 = arith.addi %arg42, %c1_i32 : i32
      %8 = scf.while (%arg43 = %c0_i32) : (i32) -> i32 {
        %c0_4 = arith.constant 0 : index
        %9 = memref.load %1[%c0_4] : memref<1xi32>
        %10 = arith.cmpi slt, %arg43, %9 : i32
        scf.condition(%10) %arg43 : i32
      } do {
      ^bb0(%arg43: i32):
        %c0_4 = arith.constant 0 : index
        %9 = memref.load %2[%c0_4] : memref<1xi32>
        %10 = arith.muli %arg42, %9 : i32
        %11 = arith.addi %arg43, %10 : i32
        %12 = arith.index_cast %11 : i32 to index
        %13 = arith.addi %12, %c0 : index
        %14 = memref.load %arg13[%13] : memref<?xf32>
        %15 = arith.addi %arg43, %c1_i32 : i32
        %16 = arith.addi %15, %10 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = arith.addi %17, %c0 : index
        %19 = memref.load %arg13[%18] : memref<?xf32>
        %20 = arith.addf %14, %19 : f32
        %21 = arith.mulf %20, %cst_1 : f32
        %22 = arith.mulf %21, %21 : f32
        %23 = memref.load %arg14[%13] : memref<?xf32>
        %24 = arith.muli %7, %9 : i32
        %25 = arith.addi %arg43, %24 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.addi %26, %c0 : index
        %28 = memref.load %arg14[%27] : memref<?xf32>
        %29 = arith.addf %23, %28 : f32
        %30 = arith.mulf %29, %cst_1 : f32
        %31 = arith.mulf %30, %30 : f32
        %32 = arith.addf %22, %31 : f32
        %33 = math.sqrt %32 : f32
        %34 = arith.muli %c0_i32, %9 : i32
        %c0_5 = arith.constant 0 : index
        %35 = memref.load %3[%c0_5] : memref<1xi32>
        %36 = arith.muli %34, %35 : i32
        %37 = arith.addi %11, %36 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.addi %38, %c0 : index
        memref.store %cst_2, %arg11[%39] : memref<?xf32>
        %c0_6 = arith.constant 0 : index
        %40 = memref.load %2[%c0_6] : memref<1xi32>
        %41 = arith.muli %arg42, %40 : i32
        %42 = arith.addi %arg43, %41 : i32
        %43 = arith.muli %c0_i32, %40 : i32
        %c0_7 = arith.constant 0 : index
        %44 = memref.load %3[%c0_7] : memref<1xi32>
        %45 = arith.muli %43, %44 : i32
        %46 = arith.addi %42, %45 : i32
        %47 = arith.index_cast %46 : i32 to index
        %48 = arith.mulf %33, %cst_0 : f32
        %49 = arith.addi %47, %c0 : index
        memref.store %48, %arg12[%49] : memref<?xf32>
        %c0_8 = arith.constant 0 : index
        %50 = memref.load %2[%c0_8] : memref<1xi32>
        %51 = arith.muli %arg42, %50 : i32
        %52 = arith.addi %arg43, %51 : i32
        %53 = arith.index_cast %52 : i32 to index
        %54 = arith.mulf %33, %cst_3 : f32
        %c0_9 = arith.constant 0 : index
        %55 = memref.load %4[%c0_9] : memref<1xf32>
        %56 = arith.divf %54, %55 : f32
        %57 = arith.addi %53, %c0 : index
        memref.store %56, %arg35[%57] : memref<?xf32>
        %c0_10 = arith.constant 0 : index
        %58 = memref.load %2[%c0_10] : memref<1xi32>
        %59 = arith.muli %arg42, %58 : i32
        %60 = arith.addi %arg43, %59 : i32
        %c0_11 = arith.constant 0 : index
        %61 = memref.load %5[%c0_11] : memref<1xi32>
        %62 = arith.addi %61, %c-1_i32 : i32
        %63 = arith.muli %62, %58 : i32
        %c0_12 = arith.constant 0 : index
        %64 = memref.load %3[%c0_12] : memref<1xi32>
        %65 = arith.muli %63, %64 : i32
        %66 = arith.addi %60, %65 : i32
        %67 = arith.index_cast %66 : i32 to index
        %68 = arith.index_cast %60 : i32 to index
        %69 = arith.addi %68, %c0 : index
        %70 = memref.load %arg16[%69] : memref<?xf32>
        %71 = arith.addi %15, %59 : i32
        %72 = arith.index_cast %71 : i32 to index
        %73 = arith.addi %72, %c0 : index
        %74 = memref.load %arg16[%73] : memref<?xf32>
        %75 = arith.addf %70, %74 : f32
        %76 = arith.mulf %75, %cst_1 : f32
        %77 = arith.mulf %76, %76 : f32
        %78 = memref.load %arg17[%69] : memref<?xf32>
        %79 = arith.muli %7, %58 : i32
        %80 = arith.addi %arg43, %79 : i32
        %81 = arith.index_cast %80 : i32 to index
        %82 = arith.addi %81, %c0 : index
        %83 = memref.load %arg17[%82] : memref<?xf32>
        %84 = arith.addf %78, %83 : f32
        %85 = arith.mulf %84, %cst_1 : f32
        %86 = arith.mulf %85, %85 : f32
        %87 = arith.addf %77, %86 : f32
        %88 = math.sqrt %87 : f32
        %89 = arith.mulf %88, %cst : f32
        %90 = arith.addi %67, %c0 : index
        memref.store %89, %arg15[%90] : memref<?xf32>
        scf.yield %15 : i32
      } attributes {arg1 = "for (int i = 0; i < imm1; i++) {", depth = 2 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 0 : i32, step = 1 : i8, ub = @imm1}
      scf.yield %7 : i32
    } attributes {arg1 = "for (int j = 0; j < jmm1; j++) {", depth = 1 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 1 : i32, step = 1 : i8, ub = @jmm1}
    return
  }
}
