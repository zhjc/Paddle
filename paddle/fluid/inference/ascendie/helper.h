/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/dynload/tensorrt.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/utils/data_type.h"

namespace paddle {
namespace inference {
namespace ascendie {

template <typename T>
struct Destroyer {
  void operator()(T* x) {
    if (x) {
      x->destroy();
    }
  }
};
template <typename T>
using infer_ptr = std::unique_ptr<T, Destroyer<T>>;

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle
