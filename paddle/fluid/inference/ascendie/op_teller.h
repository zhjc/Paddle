// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/ascendie/engine.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace ascendie {

/*
 * Single Op teller definition.
 * One can override this and define a more complex tell logic, considerring more
 * issues such as op_desc.
 */
struct Teller {
  virtual bool operator()(const framework::OpDesc& desc,
                          bool use_no_calib_int8 = false,
                          bool with_dynamic_shape = false) = 0;

  virtual ~Teller() = default;
};

/*
 * class OpTeller helps to tell whether a fluid
 * operator can be transformed to a TensorRT layer
 * and use which kind of OpConverter
 */
class OpTeller {
 public:
  static OpTeller& Global() {
    static std::unique_ptr<OpTeller> x(new OpTeller);
    return *x;
  }

  bool Tell(const framework::ir::Node* node,
            bool use_no_calib_int8 = false,
            bool with_dynamic_shape = false);

  std::unique_ptr<Teller>& GetDefaultTeller() { return tellers_.at(0); }

 private:
  OpTeller();

 private:
  std::vector<std::unique_ptr<Teller>> tellers_;
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle
