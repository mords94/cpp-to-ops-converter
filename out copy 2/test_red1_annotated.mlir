"builtin.module"() ({
  "memref.global"() {initial_value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>, sym_name = "const_test", type = memref<2xf32>} : () -> ()
  "memref.global"() {sym_name = "im", type = memref<1xi32>} : () -> ()
  "memref.global"() {sym_name = "jm", type = memref<1xi32>} : () -> ()
  "memref.global"() {sym_name = "kb", type = memref<1xi32>} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<?xf32>, %arg1: memref<?xf32>):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    %2 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %3 = "arith.constant"() {value = 1.100000e+01 : f32} : () -> f32
    %4 = "arith.constant"() {value = 0 : index} : () -> index
    %5 = "memref.get_global"() {name = @kb} : () -> memref<1xi32>
    %6 = "memref.get_global"() {name = @jm} : () -> memref<1xi32>
    %7 = "memref.get_global"() {name = @im} : () -> memref<1xi32>
    %8 = "memref.get_global"() {name = @const_test} : () -> memref<2xf32>
    %9:2 = "scf.while"(%1, %2) ({
    ^bb0(%arg2: i32, %arg3: f32):
      %10 = "arith.constant"() {value = 0 : index} : () -> index
      %11 = "memref.load"(%5, %10) : (memref<1xi32>, index) -> i32
      %12 = "arith.cmpi"(%arg2, %11) {predicate = 2 : i64} : (i32, i32) -> i1
      "scf.condition"(%12, %arg2, %arg3) : (i1, i32, f32) -> ()
    }, {
    ^bb0(%arg2: i32, %arg3: f32):
      %10:2 = "scf.while"(%1, %arg3) ({
      ^bb0(%arg4: i32, %arg5: f32):
        %12 = "arith.constant"() {value = 0 : index} : () -> index
        %13 = "memref.load"(%6, %12) : (memref<1xi32>, index) -> i32
        %14 = "arith.cmpi"(%arg4, %13) {predicate = 2 : i64} : (i32, i32) -> i1
        "scf.condition"(%14, %arg5, %arg4) : (i1, f32, i32) -> ()
      }, {
      ^bb0(%arg4: f32, %arg5: i32):
        %12:3 = "scf.while"(%1, %arg4) ({
        ^bb0(%arg6: i32, %arg7: f32):
          %14 = "arith.constant"() {value = 0 : index} : () -> index
          %15 = "memref.load"(%7, %14) : (memref<1xi32>, index) -> i32
          %16 = "arith.cmpi"(%arg6, %15) {predicate = 2 : i64} : (i32, i32) -> i1
          "scf.condition"(%16, %arg7, %15, %arg6) : (i1, f32, i32, i32) -> ()
        }, {
        ^bb0(%arg6: f32, %arg7: i32, %arg8: i32):
          %14 = "arith.muli"(%arg5, %arg7) : (i32, i32) -> i32
          %15 = "arith.addi"(%arg8, %14) : (i32, i32) -> i32
          %16 = "arith.muli"(%arg2, %arg7) : (i32, i32) -> i32
          %17 = "arith.constant"() {value = 0 : index} : () -> index
          %18 = "memref.load"(%6, %17) : (memref<1xi32>, index) -> i32
          %19 = "arith.muli"(%16, %18) : (i32, i32) -> i32
          %20 = "arith.addi"(%15, %19) : (i32, i32) -> i32
          %21 = "arith.index_cast"(%20) : (i32) -> index
          %22 = "arith.addi"(%21, %4) : (index, index) -> index
          %23 = "memref.load"(%arg0, %22) : (memref<?xf32>, index) -> f32
          %24 = "arith.addf"(%arg6, %23) : (f32, f32) -> f32
          "memref.store"(%24, %arg1, %22) : (f32, memref<?xf32>, index) -> ()
          %25 = "arith.constant"() {value = 0 : index} : () -> index
          %26 = "memref.load"(%7, %25) : (memref<1xi32>, index) -> i32
          %27 = "arith.muli"(%arg5, %26) : (i32, i32) -> i32
          %28 = "arith.addi"(%arg8, %27) : (i32, i32) -> i32
          %29 = "arith.muli"(%arg2, %26) : (i32, i32) -> i32
          %30 = "arith.constant"() {value = 0 : index} : () -> index
          %31 = "memref.load"(%6, %30) : (memref<1xi32>, index) -> i32
          %32 = "arith.muli"(%29, %31) : (i32, i32) -> i32
          %33 = "arith.addi"(%28, %32) : (i32, i32) -> i32
          %34 = "arith.index_cast"(%33) : (i32) -> index
          %35 = "arith.addi"(%34, %4) : (index, index) -> index
          %36 = "memref.load"(%arg1, %35) : (memref<?xf32>, index) -> f32
          %37 = "arith.subf"(%3, %36) : (f32, f32) -> f32
          %38 = "arith.fptosi"(%37) : (f32) -> i32
          %39 = "func.call"(%38) {callee = @abs} : (i32) -> i32
          %40 = "arith.sitofp"(%39) : (i32) -> f32
          %41 = "arith.constant"() {value = 1 : index} : () -> index
          %42 = "memref.load"(%8, %41) : (memref<2xf32>, index) -> f32
          %43 = "arith.addf"(%40, %42) : (f32, f32) -> f32
          %44 = "arith.constant"() {value = 0 : index} : () -> index
          %45 = "memref.load"(%8, %44) : (memref<2xf32>, index) -> f32
          %46 = "arith.addf"(%43, %45) : (f32, f32) -> f32
          %47 = "arith.addf"(%46, %arg6) : (f32, f32) -> f32
          %48 = "arith.addi"(%arg8, %0) : (i32, i32) -> i32
          "scf.yield"(%48, %47) : (i32, f32) -> ()
        }) {arg1 = "float e = 0.0f;", arg3 = "for (int i = 0; i < im; i++) {", arg5 = "for (int i = 0; i < im; i++) {", depth = 3 : i8, induction_variable = "i", lb = 0 : i32, loop_index = 0 : i32, step = 1 : i8, ub = @im} : (i32, f32) -> (f32, i32, i32)
        %13 = "arith.addi"(%arg5, %0) : (i32, i32) -> i32
        "scf.yield"(%13, %12#0) : (i32, f32) -> ()
      }) {arg1 = "float e = 0.0f;", arg3 = "for (int j = 0; j < jm; j++) {", depth = 2 : i8, induction_variable = "j", lb = 0 : i32, loop_index = 1 : i32, step = 1 : i8, ub = @jm} : (i32, f32) -> (f32, i32)
      %11 = "arith.addi"(%arg2, %0) : (i32, i32) -> i32
      "scf.yield"(%11, %10#0) : (i32, f32) -> ()
    }) {arg1 = "for (int k = 0; k < kb; k++) {", arg3 = "for (int k = 0; k < kb; k++) {", depth = 1 : i8, induction_variable = "k", lb = 0 : i32, loop_index = 2 : i32, step = 1 : i8, ub = @kb} : (i32, f32) -> (i32, f32)
    "func.return"() : () -> ()
  }) {function_type = (memref<?xf32>, memref<?xf32>) -> (), llvm.linkage = #llvm.linkage<external>, sym_name = "red1"} : () -> ()
}) {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx13.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} : () -> ()
