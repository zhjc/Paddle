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

class LayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a layer_norm op to ascendie layer_norm plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Bias = engine_->GetITensor(op_desc.Input("Bias").front());
    auto* Scale = engine_->GetITensor(op_desc.Input("Scale").front());

    const int begin_norm_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
            : 1;
    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    int nbDims = X->GetDimensions().Size();
    AscendIE::NormalizationLayer* layernorm_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      uint32_t axesMask{0};
      for (int32_t i = begin_norm_axis; i < nbDims; i++) {
        axesMask |= 1 << i;
      }
      /* LayerNorm使用Normalization代替，TODO：动态shape待适配 */
      layernorm_layer = engine_->network()->AddNormalization(X, Scale, Bias, axesMask);
      layernorm_layer->SetEpsilon(eps);
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "static shape mode not supported in layer norm yet"));
    }
    auto output_name = op_desc.Output("Y").front();
    RreplenishLayerAndOutput(
        layernorm_layer, "layer_norm", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(layer_norm, LayerNormOpConverter);
