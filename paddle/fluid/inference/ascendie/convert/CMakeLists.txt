list(APPEND 
  CONVERT_FILES 
  cast_op.cc 
  elementwise_op.cc 
  slice_op.cc 
  concat_op.cc 
  range_op.cc
  layer_norm_op.cc
  softmax_op.cc
  where_op.cc
  transpose_op.cc
  pad3d_op.cc
  set_value_op.cc
  scale_op.cc
  silu_op.cc
  top_k_op.cc
  lookup_table.cc
  fill_any_like_op.cc
  fill_constant_op.cc
  split_op.cc
  gelu_op.cc
  logsigmoid_op.cc
  unsqueeze2_op.cc
  c_allreduce_op.cc
  c_concat.cc)

if(WITH_ASCEND)
  cc_library(
    ascendie_converter
    SRCS ${CONVERT_FILES}
    DEPS ascendinfer_engine ascend_op_teller operator scope framework_proto op_registry)
endif()