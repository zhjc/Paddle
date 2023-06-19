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

class LogSigmoidOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert LogSigmoid op to ascendie layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(
        input_num,
        1,
        platform::errors::InvalidArgument(
            "The input X's size must equal to 1 in TRT LogSigmoid op."
            " But received X's size %d.",
            input_num));
			
	auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
	
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        platform::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT LogSigmoid op. "
            "But received Out's size %u.",
            output_num));
    
	AscendIE::BaseLayer* layer = nullptr;
	AscendIE::ActivationLayer* sigmoid = engine_->network()->AddActivationLayer(input, AscendIE::ActivationKind::SIGMOID);
    layer = engine_->network()->AddUnary(sigmoid->GetOutput(0), AscendIE::UnaryOperation::EXP); // TODO: need use AscendIE::UnaryOperation::LOG.
	
	auto output_name = op_desc.Output("Out")[0];
    
    RreplenishLayerAndOutput(layer, "logsigmoid", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(logsigmoid, LogSigmoidOpConverter);
