
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )


set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIRMemRefDialect
        MLIROptLib
        MLIRStandalone
        )

add_llvm_executable(ops-converter 
                    ops-converter.cpp
)

llvm_update_compile_flags(ops-converter)

target_sources(ops-converter PRIVATE Lib/helpers.cpp)
target_link_libraries(ops-converter PRIVATE ${LIBS})




mlir_check_all_link_libraries(ops-converter)


