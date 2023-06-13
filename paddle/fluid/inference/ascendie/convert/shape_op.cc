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

#include "paddle/fluid/inference/ascendie/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace ascendie {

class ShapeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a shape op to ascendie shape layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    AscendIE::ShapeLayer* shapeLayer = engine_->network()->AddShape(input);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(shapeLayer, "shape", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(shape, ShapeOpConverter);
