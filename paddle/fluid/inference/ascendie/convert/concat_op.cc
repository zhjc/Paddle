/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/ascendie/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace ascendie {

/*
 * ConcatOp
 */
class ConcatOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a concat op to ascendie concat layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<AscendIE::Tensor*> itensors;
    for (auto& input_name : op_desc.Input("X")) {
      itensors.push_back(engine_->GetITensor(input_name));
    }
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    if (axis < 0) {
      axis = engine_->GetITensor(op_desc.Input("X").front())->GetDimensions().Size() + axis;
    } else {
      if (!engine_->with_dynamic_shape()) {
        axis = axis - 1;  // Remove batch dim
      }
    }
    AscendIE::ConcatenationLayer* layer = engine_->network()->AddConcatenation(itensors.data(), itensors.size());
    layer->SetAxis(axis);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "concat", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(concat, ConcatOpConverter);
