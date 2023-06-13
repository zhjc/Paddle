/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class RangeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a range op to ascendie layer";
    framework::OpDesc op_desc(op, nullptr);
    AscendIE::BaseLayer* layer = nullptr;
    AscendIE::Tensor* quotient_tensor;

    // Declare inputs
    auto* start = engine_->GetITensor(op_desc.Input("Start")[0]);
    auto* end = engine_->GetITensor(op_desc.Input("End")[0]);
    auto* step = engine_->GetITensor(op_desc.Input("Step")[0]);
    auto output_name = op_desc.Output("Out")[0];

    auto zero_tensor = Add1DConstantLayer(0, output_name + "_zero_tensor_");
    auto fquotient_tensor = FloorDiv(Sub(start, end), step);
    if (start->getType() == AscendIE::DataType::FLOAT) {
      auto* cast_int32_layer = engine_->network()->AddCast(fquotient_tensor, AscendIE::DataType::INT32);
      cast_int32_layer->SetToType(AscendIE::DataType:::INT32);
      cast_int32_layer->GetOutput(0)->SetType(AscendIE::DataType::INT32);
      quotient_tensor = cast_int32_layer->getOutput(0);
    } else {
      quotient_tensor = fquotient_tensor;
    }
    auto number_tensor = Max(Sub(zero_tensor, quotient_tensor), zero_tensor);
    auto* start1 = engine_->GetITensor(op_desc.Input("Start")[0], true);

    layer = engine_->network()->AddFill(AscendIE::Dims{}, AscendIE::FillOperation::LINSPACE);

    layer->SetInput(0, *number_tensor);
    layer->SetInput(1, *start1);
    layer->SetInput(2, *step);

    RreplenishLayerAndOutput(layer, "range", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(range, RangeOpConverter);
