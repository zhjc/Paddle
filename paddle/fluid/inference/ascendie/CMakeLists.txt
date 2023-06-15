if(WITH_ASCEND)
  cc_library(
    ascendinfer_engine
    SRCS engine.cc
    DEPS ${GLOB_OPERATOR_DEPS} framework_proto device_context ascend_ge
         ascend_graph
  )# TODO  GLOB OPERATOR_DEPS 变量可能需要包含ascend_wrapper.cc文件，该文件在 fleet文件夹下
  add_subdirectory(convert)
  cc_library(
    ascend_op_teller
    SRCS op_teller.cc
    DEPS ${GLOB_OPERATOR_DEPS} framework_proto device_context)
endif()
