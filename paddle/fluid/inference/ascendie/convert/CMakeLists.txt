list(APPEND 
  CONVERT_FILES 
  cast_op.cc 
  elementwise_op.cc 
  slice_op.cc 
  concat_op.cc 
  range_op.cc 
  softmax_op.cc
  where_op.cc
  transpose_op.cc)

if(WITH_ASCEND)
  cc_library(
    ascendie_converter
    SRCS ${CONVERT_FILES}
    DEPS ascendinfer_engine ascend_op_teller operator scope framework_proto op_registry)
endif()