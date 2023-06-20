// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/ascendie/convert/op_converter.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace ascendie {
using ReduceType = paddle::inference::ascendie::ReduceType;
std::map<std::string, ReduceType> op_to_reduce_type = {
    {"c_allreduce_sum", paddle::inference::ascendie::kRedSum}};

class CAllReduceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert callreduce op to ascendie layer";
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(
          platform::errors::Fatal("Unsupported static graph mode. Please set "
                                  "dynamic shape of inputs."));
    }
    ReduceType red_type = op_to_reduce_type[op.type()];
    std::string name = op.type();

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(
        input_num,
        1,
        platform::errors::InvalidArgument(
            "The input X's size must equal to 1 in TRT c_allreduce op."
            " But received X's size %d.",
            input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        platform::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT c_allreduce op. "
            "But received Out's size %u.",
            output_num));
    // Get attrs
    int ring_id = PADDLE_GET_CONST(int, op_desc.GetAttr("ring_id"));
    bool use_calc_stream =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("use_calc_stream"));
    AscendIE::BaseLayer* layer = nullptr;
    /* TODO:AddReduce待完善 */
    // AscendIE::AllReduceLayer* layer =  engine_->network()->AddReduce(input, red_type);

    auto output_name = op_desc.Output("Out")[0];

    RreplenishLayerAndOutput(layer, name, {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(c_allreduce_sum, CAllReduceOpConverter);
