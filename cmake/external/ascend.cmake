# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#NOTE: Logic is from
# https://github.com/mindspore-ai/graphengine/blob/master/CMakeLists.txt
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()

if(EXISTS
   ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include/graph/ascend_string.h)
  # It means CANN 20.2 +
  add_definitions(-DPADDLE_WITH_ASCEND_STRING)
endif()

if(WITH_ASCEND)
  set(ASCEND_DRIVER_DIR ${ASCEND_DIR}/driver/lib64)
  set(ASCEND_DRIVER_COMMON_DIR ${ASCEND_DIR}/driver/lib64/common)
  set(ASCEND_DRIVER_SHARE_DIR ${ASCEND_DIR}/driver/lib64/share)
  set(ASCEND_RUNTIME_DIR ${ASCEND_DIR}/fwkacllib/lib64)
  set(ASCEND_ATC_DIR ${ASCEND_DIR}/atc/lib64)
  set(ASCEND_ACL_DIR ${ASCEND_DIR}/acllib/lib64)
  set(STATIC_ACL_LIB ${ASCEND_ACL_DIR})

  set(ASCEND_MS_RUNTIME_PATH ${ASCEND_RUNTIME_DIR} ${ASCEND_ACL_DIR}
                             ${ASCEND_ATC_DIR})
  set(ASCEND_MS_DRIVER_PATH ${ASCEND_DRIVER_DIR} ${ASCEND_DRIVER_COMMON_DIR})
  set(ATLAS_RUNTIME_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)
  set(ATLAS_RUNTIME_INC_DIR
      ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include)
  set(ATLAS_ACL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/acllib/lib64)
  set(ATLAS_ATC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/atc/lib64)
  set(ATLAS_MS_RUNTIME_PATH ${ATLAS_RUNTIME_DIR} ${ATLAS_ACL_DIR}
                            ${ATLAS_ATC_DIR})

  set(atlas_graph_lib ${ATLAS_RUNTIME_DIR}/libgraph.so)
  set(atlas_ge_runner_lib ${ATLAS_RUNTIME_DIR}/libge_runner.so)
  set(atlas_acl_lib ${ATLAS_RUNTIME_DIR}/libascendcl.so)
  include_directories(${ATLAS_RUNTIME_INC_DIR})

  add_library(ascend_ge SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_ge PROPERTY IMPORTED_LOCATION
                                         ${atlas_ge_runner_lib})

  add_library(ascend_graph SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_graph PROPERTY IMPORTED_LOCATION
                                            ${atlas_graph_lib})

  add_library(atlas_acl SHARED IMPORTED GLOBAL)
  set_property(TARGET atlas_acl PROPERTY IMPORTED_LOCATION ${atlas_acl_lib})

  add_custom_target(extern_ascend DEPENDS ascend_ge ascend_graph atlas_acl)
endif()
