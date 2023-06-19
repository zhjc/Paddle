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

class Unsqueeze2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a unsqueeze2 op to ascendie shuffle layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    AscendIE::Tensor* input = engine_->GetITensor(op_desc.Input("X")[0]);
    AscendIE::Dims input_dims = input->GetDimensions();
    auto output_name = op_desc.Output("Out")[0];

    // Get Attrs
    std::vector<int> axes =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    PADDLE_ENFORCE_GT(
        axes.size(),
        0,
        platform::errors::InvalidArgument(
            "Attr(axes).size should be > 0 in unsqueeze2 op in ascendie,"
            "but received axes.size() = %d.",
            axes.size()));

    std::vector<bool> should_unsqueeze(input_dims.Size() + axes.size(), false);
    int cur_out_rank = input_dims.Size();
    for (size_t i = 0; i < axes.size(); i++) {
      cur_out_rank++;
      if (engine_->with_dynamic_shape()) {
        axes[i] += (axes[i] < 0) ? cur_out_rank : 0;
      } else {
        axes[i] += (axes[i] < 0) ? cur_out_rank : -1;
      }
      // axes[i] is relative to cur_out_rank
      // we make [axes[i], cur_out_rank - 2] shift right
      // and make (axes[i]) to true!
      for (int j = cur_out_rank - 1; j > axes[i]; j--) {
        should_unsqueeze[j] = should_unsqueeze[j - 1];
      }
      if (axes[i] >= cur_out_rank)
        should_unsqueeze[cur_out_rank - 1] = true;
      else
        should_unsqueeze[axes[i]] = true;
    }


    std::vector<int64_t> temp(should_unsqueeze.size(), 0);
    std::vector<int32_t> gather_indices;
    int in_rank_i = 0;
    for (size_t i = 0; i < should_unsqueeze.size(); i++) {
      if (should_unsqueeze[i]) {
        temp[i] = 1;
        gather_indices.push_back(input_dims.Size());
        continue;
      }
      temp[i] = input_dims[in_rank_i];
      gather_indices.push_back(in_rank_i);
      in_rank_i++;
    }
    AscendIE::Dims trt_out_dims(temp);

    AscendIE::ShuffleLayer* layer = engine_->network()->AddShuffle(input);
    if (engine_->with_dynamic_shape()) {
      AscendIE::Tensor* shape_tensor = Shape(input);
      std::vector<int32_t> all_one(axes.size(), 1);
      AscendIE::Tensor* all_one_tensor = Add1DConstantLayer(all_one);
      std::vector<AscendIE::Tensor*> concat_inputs = {shape_tensor,
                                                       all_one_tensor};
      AscendIE::Tensor* real_shape_tensor = Gather(Concat(concat_inputs), gather_indices);
      layer->SetInput(1, real_shape_tensor);
    } else {
      layer->SetReshapeDimensions(trt_out_dims);
    }
    RreplenishLayerAndOutput(layer, "unsqueeze2", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(unsqueeze2, Unsqueeze2OpConverter);
