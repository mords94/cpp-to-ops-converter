#loc1 = loc("./in/test.c":268:6)
#loc24 = loc("./in/test.c":297:3)
#loc27 = loc("./in/test.c":298:5)
#loc30 = loc("./in/test.c":299:7)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-macosx14.0.0", "polygeist.target-cpu" = "penryn", "polygeist.target-features" = "+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @im : memref<1xi32> loc(#loc0)
  memref.global @jm : memref<1xi32> loc(#loc0)
  memref.global @kb : memref<1xi32> loc(#loc0)
  func.func @cbc(%arg0: memref<?xf32> loc("./in/test.c":268:6), %arg1: memref<?xf32> loc("./in/test.c":268:6), %arg2: memref<?xf32> loc("./in/test.c":268:6), %arg3: memref<?xf32> loc("./in/test.c":268:6), %arg4: memref<?xf32> loc("./in/test.c":268:6), %arg5: memref<?xf32> loc("./in/test.c":268:6), %arg6: memref<?xf32> loc("./in/test.c":268:6), %arg7: memref<?xf32> loc("./in/test.c":268:6), %arg8: memref<?xf32> loc("./in/test.c":268:6), %arg9: memref<?xf32> loc("./in/test.c":268:6), %arg10: memref<?xf32> loc("./in/test.c":268:6), %arg11: memref<?xf32> loc("./in/test.c":268:6), %arg12: memref<?xf32> loc("./in/test.c":268:6), %arg13: memref<?xf32> loc("./in/test.c":268:6), %arg14: memref<?xf32> loc("./in/test.c":268:6), %arg15: memref<?xf32> loc("./in/test.c":268:6), %arg16: memref<?xf32> loc("./in/test.c":268:6), %arg17: memref<?xf32> loc("./in/test.c":268:6), %arg18: memref<?xf32> loc("./in/test.c":268:6), %arg19: memref<?xf32> loc("./in/test.c":268:6), %arg20: memref<?xf32> loc("./in/test.c":268:6), %arg21: memref<?xf32> loc("./in/test.c":268:6), %arg22: memref<?xf32> loc("./in/test.c":268:6), %arg23: memref<?xf32> loc("./in/test.c":268:6), %arg24: memref<?xf32> loc("./in/test.c":268:6), %arg25: memref<?xf32> loc("./in/test.c":268:6), %arg26: memref<?xf32> loc("./in/test.c":268:6), %arg27: memref<?xf32> loc("./in/test.c":268:6), %arg28: memref<?xf32> loc("./in/test.c":268:6), %arg29: memref<?xf32> loc("./in/test.c":268:6), %arg30: memref<?xf32> loc("./in/test.c":268:6), %arg31: memref<?xf32> loc("./in/test.c":268:6), %arg32: memref<?xf32> loc("./in/test.c":268:6), %arg33: memref<?xf32> loc("./in/test.c":268:6), %arg34: memref<?xf32> loc("./in/test.c":268:6), %arg35: memref<?xf32> loc("./in/test.c":268:6), %arg36: memref<?xf32> loc("./in/test.c":268:6), %arg37: memref<?xf32> loc("./in/test.c":268:6), %arg38: memref<?xf32> loc("./in/test.c":268:6), %arg39: memref<?xf32> loc("./in/test.c":268:6), %arg40: memref<?xf32> loc("./in/test.c":268:6), %arg41: memref<?xf32> loc("./in/test.c":268:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %cst = arith.constant 3.000000e+00 : f32 loc(#loc3)
    %cst_0 = arith.constant 6.000000e+00 : f32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %cst_1 = arith.constant 9.000000e+00 : f32 loc(#loc6)
    %cst_2 = arith.constant 1.800000e+01 : f32 loc(#loc7)
    %cst_3 = arith.constant 1.000000e+00 : f32 loc(#loc8)
    %cst_4 = arith.constant 8.000000e-02 : f32 loc(#loc9)
    %cst_5 = arith.constant 1.010000e+01 : f32 loc(#loc10)
    %cst_6 = arith.constant 1.660000e+01 : f32 loc(#loc11)
    %cst_7 = arith.constant 7.400000e-01 : f32 loc(#loc12)
    %cst_8 = arith.constant 9.200000e-01 : f32 loc(#loc13)
    %c0 = arith.constant 0 : index loc(#loc14)
    %0 = arith.mulf %cst_2, %cst_8 : f32 loc(#loc15)
    %1 = arith.mulf %0, %cst_8 : f32 loc(#loc16)
    %2 = arith.mulf %cst_1, %cst_8 : f32 loc(#loc17)
    %3 = arith.mulf %2, %cst_7 : f32 loc(#loc18)
    %4 = arith.addf %1, %3 : f32 loc(#loc19)
    %5 = arith.mulf %cst_1, %cst_8 : f32 loc(#loc20)
    %6 = arith.mulf %5, %cst_7 : f32 loc(#loc21)
    %7 = scf.while (%arg42 = %c0_i32) : (i32) -> i32 {
      %8 = memref.get_global @kb : memref<1xi32> loc(#loc22)
      %9 = affine.load %8[0] : memref<1xi32> loc(#loc22)
      %10 = arith.cmpi slt, %arg42, %9 : i32 loc(#loc23)
      scf.condition(%10) %arg42 : i32 loc(#loc24)
    } do {
    ^bb0(%arg42: i32 loc("./in/test.c":297:3)):
      %8 = scf.while (%arg43 = %c0_i32) : (i32) -> i32 {
        %10 = memref.get_global @jm : memref<1xi32> loc(#loc25)
        %11 = affine.load %10[0] : memref<1xi32> loc(#loc25)
        %12 = arith.cmpi slt, %arg43, %11 : i32 loc(#loc26)
        scf.condition(%12) %arg43 : i32 loc(#loc27)
      } do {
      ^bb0(%arg43: i32 loc("./in/test.c":298:5)):
        %10 = memref.get_global @jm : memref<1xi32> loc(#loc25)
        %11:2 = scf.while (%arg44 = %c0_i32) : (i32) -> (i32, i32) {
          %13 = memref.get_global @im : memref<1xi32> loc(#loc28)
          %14 = affine.load %13[0] : memref<1xi32> loc(#loc28)
          %15 = arith.cmpi slt, %arg44, %14 : i32 loc(#loc29)
          scf.condition(%15) %14, %arg44 : i32, i32 loc(#loc30)
        } do {
        ^bb0(%arg44: i32 loc("./in/test.c":299:7), %arg45: i32 loc("./in/test.c":299:7)):
          %13 = memref.get_global @im : memref<1xi32> loc(#loc28)
          %14 = arith.mulf %cst_0, %cst_8 : f32 loc(#loc31)
          %15 = arith.divf %14, %cst_6 : f32 loc(#loc32)
          %16 = arith.muli %arg43, %arg44 : i32 loc(#loc33)
          %17 = arith.addi %arg45, %16 : i32 loc(#loc34)
          %18 = arith.muli %arg42, %arg44 : i32 loc(#loc35)
          %19 = affine.load %10[0] : memref<1xi32> loc(#loc36)
          %20 = arith.muli %18, %19 : i32 loc(#loc37)
          %21 = arith.addi %17, %20 : i32 loc(#loc38)
          %22 = arith.index_cast %21 : i32 to index loc(#loc39)
          %23 = arith.addi %22, %c0 : index loc(#loc40)
          %24 = memref.load %arg38[%23] : memref<?xf32> loc(#loc40)
          %25 = arith.mulf %15, %24 : f32 loc(#loc41)
          %26 = arith.subf %cst_3, %25 : f32 loc(#loc42)
          %27 = arith.mulf %cst_7, %26 : f32 loc(#loc43)
          %28 = arith.mulf %cst, %cst_7 : f32 loc(#loc44)
          %29 = arith.mulf %28, %cst_5 : f32 loc(#loc45)
          %30 = arith.muli %arg43, %arg44 : i32 loc(#loc33)
          %31 = arith.addi %arg45, %30 : i32 loc(#loc34)
          %32 = arith.muli %arg42, %arg44 : i32 loc(#loc35)
          %33 = arith.muli %32, %19 : i32 loc(#loc37)
          %34 = arith.addi %31, %33 : i32 loc(#loc38)
          %35 = arith.index_cast %34 : i32 to index loc(#loc46)
          %36 = arith.addi %35, %c0 : index loc(#loc47)
          %37 = memref.load %arg38[%36] : memref<?xf32> loc(#loc47)
          %38 = arith.divf %29, %37 : f32 loc(#loc48)
          %39 = arith.mulf %cst_2, %cst_8 : f32 loc(#loc49)
          %40 = arith.mulf %39, %cst_7 : f32 loc(#loc50)
          %41 = arith.addf %38, %40 : f32 loc(#loc51)
          %42 = arith.mulf %cst, %cst_4 : f32 loc(#loc52)
          %43 = arith.subf %cst_3, %42 : f32 loc(#loc53)
          %44 = arith.mulf %cst_0, %cst_8 : f32 loc(#loc54)
          %45 = arith.divf %44, %cst_6 : f32 loc(#loc55)
          %46 = arith.muli %arg43, %arg44 : i32 loc(#loc33)
          %47 = arith.addi %arg45, %46 : i32 loc(#loc34)
          %48 = arith.muli %arg42, %arg44 : i32 loc(#loc35)
          %49 = arith.muli %48, %19 : i32 loc(#loc37)
          %50 = arith.addi %47, %49 : i32 loc(#loc38)
          %51 = arith.index_cast %50 : i32 to index loc(#loc56)
          %52 = arith.addi %51, %c0 : index loc(#loc57)
          %53 = memref.load %arg38[%52] : memref<?xf32> loc(#loc57)
          %54 = arith.mulf %45, %53 : f32 loc(#loc58)
          %55 = arith.subf %43, %54 : f32 loc(#loc59)
          %56 = arith.mulf %cst_8, %55 : f32 loc(#loc60)
          %57 = arith.muli %arg43, %arg44 : i32 loc(#loc33)
          %58 = arith.addi %arg45, %57 : i32 loc(#loc34)
          %59 = arith.muli %arg42, %arg44 : i32 loc(#loc35)
          %60 = arith.muli %59, %19 : i32 loc(#loc37)
          %61 = arith.addi %58, %60 : i32 loc(#loc38)
          %62 = arith.index_cast %61 : i32 to index loc(#loc61)
          %63 = arith.addi %62, %c0 : index loc(#loc62)
          %64 = memref.load %arg36[%63] : memref<?xf32> loc(#loc62)
          %65 = arith.mulf %41, %64 : f32 loc(#loc63)
          %66 = arith.subf %cst_3, %65 : f32 loc(#loc64)
          %67 = arith.divf %27, %66 : f32 loc(#loc65)
          %68 = arith.addi %62, %c0 : index loc(#loc66)
          memref.store %67, %arg1[%68] : memref<?xf32> loc(#loc67)
          %69 = affine.load %13[0] : memref<1xi32> loc(#loc68)
          %70 = arith.muli %arg43, %69 : i32 loc(#loc33)
          %71 = arith.addi %arg45, %70 : i32 loc(#loc34)
          %72 = arith.muli %arg42, %69 : i32 loc(#loc35)
          %73 = affine.load %10[0] : memref<1xi32> loc(#loc36)
          %74 = arith.muli %72, %73 : i32 loc(#loc37)
          %75 = arith.addi %71, %74 : i32 loc(#loc38)
          %76 = arith.index_cast %75 : i32 to index loc(#loc69)
          %77 = arith.addi %76, %c0 : index loc(#loc70)
          %78 = memref.load %arg1[%77] : memref<?xf32> loc(#loc70)
          %79 = arith.mulf %78, %4 : f32 loc(#loc71)
          %80 = arith.addi %76, %c0 : index loc(#loc72)
          %81 = memref.load %arg36[%80] : memref<?xf32> loc(#loc72)
          %82 = arith.mulf %79, %81 : f32 loc(#loc73)
          %83 = arith.addf %56, %82 : f32 loc(#loc74)
          %84 = arith.addi %76, %c0 : index loc(#loc75)
          memref.store %83, %arg0[%84] : memref<?xf32> loc(#loc76)
          %85 = affine.load %13[0] : memref<1xi32> loc(#loc68)
          %86 = arith.muli %arg43, %85 : i32 loc(#loc33)
          %87 = arith.addi %arg45, %86 : i32 loc(#loc34)
          %88 = arith.muli %arg42, %85 : i32 loc(#loc35)
          %89 = affine.load %10[0] : memref<1xi32> loc(#loc36)
          %90 = arith.muli %88, %89 : i32 loc(#loc37)
          %91 = arith.addi %87, %90 : i32 loc(#loc38)
          %92 = arith.index_cast %91 : i32 to index loc(#loc77)
          %93 = arith.addi %92, %c0 : index loc(#loc78)
          %94 = memref.load %arg0[%93] : memref<?xf32> loc(#loc79)
          %95 = arith.addi %92, %c0 : index loc(#loc80)
          %96 = memref.load %arg36[%95] : memref<?xf32> loc(#loc80)
          %97 = arith.mulf %6, %96 : f32 loc(#loc81)
          %98 = arith.subf %cst_3, %97 : f32 loc(#loc82)
          %99 = arith.divf %94, %98 : f32 loc(#loc83)
          %100 = arith.addi %92, %c0 : index loc(#loc78)
          memref.store %99, %arg0[%100] : memref<?xf32> loc(#loc84)
          %101 = arith.addi %arg45, %c1_i32 : i32 loc(#loc2)
          scf.yield %101 : i32 loc(#loc30)
        } loc(#loc28)
        %12 = arith.addi %arg43, %c1_i32 : i32 loc(#loc85)
        scf.yield %12 : i32 loc(#loc27)
      } loc(#loc25)
      %9 = arith.addi %arg42, %c1_i32 : i32 loc(#loc86)
      scf.yield %9 : i32 loc(#loc24)
    } loc(#loc22)
    return loc(#loc87)
  } loc(#loc1)
} loc(#loc0)
#loc0 = loc(unknown)
#loc2 = loc("./in/test.c":299:32)
#loc3 = loc("./in/test.c":301:24)
#loc4 = loc("./in/test.c":300:37)
#loc5 = loc("./in/test.c":297:16)
#loc6 = loc("./in/test.c":294:36)
#loc7 = loc("./in/test.c":294:18)
#loc8 = loc("./in/test.c":284:16)
#loc9 = loc("./in/test.c":281:15)
#loc10 = loc("./in/test.c":280:15)
#loc11 = loc("./in/test.c":279:15)
#loc12 = loc("./in/test.c":278:15)
#loc13 = loc("./in/test.c":277:15)
#loc14 = loc("./in/pom2k_c_header.h":4:16)
#loc15 = loc("./in/test.c":294:24)
#loc16 = loc("./in/test.c":294:29)
#loc17 = loc("./in/test.c":294:41)
#loc18 = loc("./in/test.c":294:46)
#loc19 = loc("./in/test.c":294:34)
#loc20 = loc("./in/test.c":295:23)
#loc21 = loc("./in/test.c":295:28)
#loc22 = loc("./in/test.c":297:23)
#loc23 = loc("./in/test.c":297:21)
#loc25 = loc("./in/test.c":298:25)
#loc26 = loc("./in/test.c":298:23)
#loc28 = loc("./in/test.c":299:27)
#loc29 = loc("./in/test.c":299:25)
#loc31 = loc("./in/test.c":300:42)
#loc32 = loc("./in/test.c":300:47)
#loc33 = loc("./in/pom2k_c_header.h":7:50)
#loc34 = loc("./in/pom2k_c_header.h":7:45)
#loc35 = loc("./in/pom2k_c_header.h":7:59)
#loc36 = loc("./in/pom2k_c_header.h":8:46)
#loc37 = loc("./in/pom2k_c_header.h":7:63)
#loc38 = loc("./in/pom2k_c_header.h":7:54)
#loc39 = loc("./in/test.c":300:71)
#loc40 = loc("./in/test.c":300:54)
#loc41 = loc("./in/test.c":300:52)
#loc42 = loc("./in/test.c":300:35)
#loc43 = loc("./in/test.c":300:27)
#loc44 = loc("./in/test.c":301:29)
#loc45 = loc("./in/test.c":301:34)
#loc46 = loc("./in/test.c":301:58)
#loc47 = loc("./in/test.c":301:41)
#loc48 = loc("./in/test.c":301:39)
#loc49 = loc("./in/test.c":301:68)
#loc50 = loc("./in/test.c":301:73)
#loc51 = loc("./in/test.c":301:60)
#loc52 = loc("./in/test.c":303:31)
#loc53 = loc("./in/test.c":303:24)
#loc54 = loc("./in/test.c":303:43)
#loc55 = loc("./in/test.c":303:48)
#loc56 = loc("./in/test.c":303:72)
#loc57 = loc("./in/test.c":303:55)
#loc58 = loc("./in/test.c":303:53)
#loc59 = loc("./in/test.c":303:36)
#loc60 = loc("./in/test.c":303:16)
#loc61 = loc("./in/test.c":304:25)
#loc62 = loc("./in/test.c":304:53)
#loc63 = loc("./in/test.c":304:51)
#loc64 = loc("./in/test.c":304:43)
#loc65 = loc("./in/test.c":304:35)
#loc66 = loc("./in/test.c":304:9)
#loc67 = loc("./in/test.c":304:27)
#loc68 = loc("./in/pom2k_c_header.h":8:42)
#loc69 = loc("./in/test.c":305:25)
#loc70 = loc("./in/test.c":306:21)
#loc71 = loc("./in/test.c":306:39)
#loc72 = loc("./in/test.c":306:49)
#loc73 = loc("./in/test.c":306:47)
#loc74 = loc("./in/test.c":306:19)
#loc75 = loc("./in/test.c":305:9)
#loc76 = loc("./in/test.c":305:27)
#loc77 = loc("./in/test.c":307:25)
#loc78 = loc("./in/test.c":307:9)
#loc79 = loc("./in/test.c":308:13)
#loc80 = loc("./in/test.c":308:49)
#loc81 = loc("./in/test.c":308:47)
#loc82 = loc("./in/test.c":308:39)
#loc83 = loc("./in/test.c":308:31)
#loc84 = loc("./in/test.c":307:27)
#loc85 = loc("./in/test.c":298:30)
#loc86 = loc("./in/test.c":297:28)
#loc87 = loc("./in/test.c":312:1)