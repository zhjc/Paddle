list(APPEND CONVERT_FILES cast_op.cc)

if(WITH_ASCEND)
  cc_library(
    ascendie_converter
    SRCS ${CONVERT_FILES}
    DEPS ascendinfer_engine ascend_op_teller operator scope framework_proto op_registry)
endif()
