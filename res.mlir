"builtin.module"() ({
  "memref.global"() {initial_value, sym_name = "kbm1", type = memref<1xi32>} : () -> ()
  "memref.global"() {initial_value, sym_name = "jm", type = memref<1xi32>} : () -> ()
  "memref.global"() {initial_value, sym_name = "im", type = memref<1xi32>} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<?xf32>, %arg1: memref<?xf32>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : i32} : () -> i32
    %2 = "arith.constant"() {value = 0 : i32} : () -> i32
    %3 = "arith.constant"() {value = 2 : i32} : () -> i32
    %4 = "arith.constant"() {value = 4 : i32} : () -> i32
    %5 = "arith.constant"() {value = 3 : i32} : () -> i32
    %6 = "memref.get_global"() {name = @kbm1} : () -> memref<1xi32>
    %7 = "memref.get_global"() {name = @jm} : () -> memref<1xi32>
    %8 = "memref.get_global"() {name = @im} : () -> memref<1xi32>
    %9 = "arith.constant"() {value = 0 : index} : () -> index
    %10 = "memref.load"(%6, %9) : (memref<1xi32>, index) -> i32
    %11 = "arith.index_cast"(%3) : (i32) -> index
    %12 = "arith.index_cast"(%10) : (i32) -> index
    %13 = "arith.constant"() {value = 1 : index} : () -> index
    %14 = "scf.for"(%11, %12, %13, %14) ({
    ^bb0(%arg2: i32):
      %15 = "arith.addi"(%arg2, %1) : (i32, i32) -> i32
      %16 = "arith.constant"() {value = 0 : index} : () -> index
      %17 = "memref.load"(%7, %16) : (memref<1xi32>, index) -> i32
      %18 = "arith.index_cast"(%5) : (i32) -> index
      %19 = "arith.index_cast"(%17) : (i32) -> index
      %20 = "arith.constant"() {value = 1 : index} : () -> index
      %21 = "scf.for"(%18, %19, %20, %21) ({
      ^bb0(%arg3: index):
        %22 = "arith.addi"(%arg3, %1) : (i32, i32) -> i32
        %23 = "arith.constant"() {value = 0 : index} : () -> index
        %24 = "memref.load"(%8, %23) : (memref<1xi32>, index) -> i32
        %25 = "arith.index_cast"(%4) : (i32) -> index
        %26 = "arith.index_cast"(%24) : (i32) -> index
        %27 = "arith.constant"() {value = 1 : index} : () -> index
        %28 = "scf.for"(%25, %26, %27) ({
        ^bb0(%arg4: index):
          %29 = "memref.load"(%8, %0) : (memref<1xi32>, index) -> i32
          %30 = "arith.addi"(%arg4, %1) : (i32, i32) -> i32
          %31 = "arith.muli"(%arg3, %29) : (i32, i32) -> i32
          %32 = "arith.addi"(%30, %31) : (i32, i32) -> i32
          %33 = "arith.muli"(%arg2, %29) : (i32, i32) -> i32
          %34 = "memref.load"(%7, %0) : (memref<1xi32>, index) -> i32
          %35 = "arith.muli"(%33, %34) : (i32, i32) -> i32
          %36 = "arith.addi"(%32, %35) : (i32, i32) -> i32
          %37 = "arith.index_cast"(%36) : (i32) -> index
          %38 = "arith.addi"(%arg4, %31) : (i32, i32) -> i32
          %39 = "arith.addi"(%38, %35) : (i32, i32) -> i32
          %40 = "arith.index_cast"(%39) : (i32) -> index
          %41 = "memref.load"(%arg0, %40) : (memref<?xf32>, index) -> f32
          %42 = "arith.index_cast"(%38) : (i32) -> index
          %43 = "memref.load"(%arg1, %42) : (memref<?xf32>, index) -> f32
          %44 = "arith.mulf"(%41, %43) : (f32, f32) -> f32
          "memref.store"(%44, %arg0, %37) : (f32, memref<?xf32>, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> i32
        "scf.yield"() : () -> ()
      }) : (index, index, index, i32) -> i32
      "scf.yield"() : () -> ()
    }) : (index, index, index, i32) -> i32
    "func.return"() : () -> ()
  }) {function_type = (memref<?xf32>, memref<?xf32>) -> (), llvm.linkage = #llvm.linkage<external>, sym_name = "test2"} : () -> ()
}) : () -> ()