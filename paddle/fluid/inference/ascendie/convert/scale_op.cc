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

/*
 * Scale Op
 */
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a scale op to ascendie mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<AscendIE::Tensor*> itensors;
    std::string input_name = op_desc.Input("X").front();
    std::string out_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    bool bias_after_scale =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("bias_after_scale"));
    float bias = PADDLE_GET_CONST(float, op_desc.GetAttr("bias"));
    float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("scale"));
    bool is_int = input->GetType() == AscendIE::DataType::INT32;
    AscendIE::BaseLayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      AscendIE::Tensor* bias_tensor =
          is_int ? Add1DConstantLayer(
                       static_cast<int>(bias > 0 ? bias + 0.5 : bias - 0.5))
                 : Add1DConstantLayer(bias);
      bool is_bias_0 = (bias < 1e-06 && bias > -1e-06);

      std::vector<int32_t> bias_shapes(input->GetDimensions().Size(), 1);
      auto* bias_shapes_tensor = Add1DConstantLayer(bias_shapes);
      auto* reshape_layer_bias = engine_->network()->AddShuffle(bias_tensor);
 
      reshape_layer_bias->SetInput(1, bias_shapes_tensor);

      bool has_scale_tensor;
      AscendIE::Tensor* scale_tensor;
      bool is_scale_1;

      auto scale_inputs = op_desc.Inputs();
      if (scale_inputs.find("ScaleTensor") != scale_inputs.end() &&
          op_desc.Input("ScaleTensor").size()) {  // has EndsTensor input
        has_scale_tensor = true;
        scale_tensor = engine_->GetITensor(op_desc.Input("ScaleTensor")[0]);
        is_scale_1 = false;
      } else {
        has_scale_tensor = false;
        scale_tensor = is_int ? Add1DConstantLayer(static_cast<int>(
                                    scale > 0 ? scale + 0.5 : scale - 0.5))
                              : Add1DConstantLayer(scale);
        is_scale_1 = ((scale - 1.0) < 1e-06 && (scale - 1.0) > -1e-06);
      }

      std::vector<int32_t> scale_shapes(input->GetDimensions().Size(), 1);
      auto* scale_shapes_tensor = Add1DConstantLayer(scale_shapes);
      auto* reshape_layer_scale = engine_->network()->AddShuffle(scale_tensor);

      reshape_layer_scale->SetInput(1, scale_shapes_tensor);

      if (!has_scale_tensor && is_scale_1 && is_bias_0) {
        /* TODO:identity layer当前没有，用cast代替 */
        layer = engine_->network()->AddCast(input, input->GetType());
      } else {
        if (bias_after_scale) {
          if (!is_scale_1) {
            layer = engine_->network()->AddElementWise(input,
              reshape_layer_scale->GetOutput(0), AscendIE::ElementWiseOperation::MUL);
            input = layer->GetOutput(0);
          }
          if (!is_bias_0) {
            layer = engine_->network()->AddElementWise(input,
              reshape_layer_scale->GetOutput(0), AscendIE::ElementWiseOperation::ADD);
          }
        } else {
          if (!is_bias_0) {
            layer = engine_->network()->AddElementWise(input,
              reshape_layer_scale->GetOutput(0), AscendIE::ElementWiseOperation::ADD);
            input = layer->GetOutput(0);
          }
          if (!is_scale_1) {
            layer = engine_->network()->AddElementWise(input,
              reshape_layer_scale->GetOutput(0), AscendIE::ElementWiseOperation::MUL);
          }
        }
      }
    } else {
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_scale_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };

      float* bias_ptr = create_weights(bias, "bias");
      float* scale_ptr = create_weights(scale, "scale");

      AscendIEEngine::Weight scale_weights{
          AscendIE::DataType::FLOAT, static_cast<void*>(scale_ptr), 1};
      AscendIEEngine::Weight shift_weights{
          AscendIE::DataType::FLOAT, static_cast<void*>(bias_ptr), 1};
      AscendIEEngine::Weight power_weights{
          AscendIE::DataType::FLOAT, nullptr, 0};

      auto input_dim = input->GetDimensions();

      AscendIE::ShuffleLayer* expand_layer = nullptr;
      AscendIE::ShuffleLayer* squeeze_layer = nullptr;

      if (input_dim.Size() < 3) {
        int64_t expand_shape_arr[3];
        for (int i = 0; i < 3; i++) {
          if (i < input_dim.Size()) {
            expand_shape_arr[i] = input_dim[i] < 0 ? 0 : input_dim[i];
          } else {
            expand_shape_arr[i] = 1;
          }
        }
        AscendIE::Dims expand_shape(3, expand_shape_arr);
        expand_layer = engine_->network()->AddShuffle(input);
        expand_layer->SetReshapeDimensions(expand_shape);
        input = expand_layer->GetOutput(0);
        expand_layer->GetOutput(0)->SetName(
            ("before_reshape_out: " + out_name).c_str());
        expand_layer->SetName(
            ("Scale: before_reshape (Output: " + out_name + ")").c_str());
      }
      auto data_layout = phi::StringToDataLayout(
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_layout")));
      int32_t channelAxis = data_layout == phi::DataLayout::kNCHW? 1 : -1;

      if (bias_after_scale) {
        /* TODO：待AIE支持UNIFORM mode */
        layer = engine_->network()->AddScaleNd(input, AscendIE::ScaleMode::KCHANNEL,
          shift_weights.get(),
          scale_weights.get(),
          channelAxis);

        layer->GetOutput(0)->SetName(
            ("bias_after_scale_out: " + out_name).c_str());
        layer->SetName(("Scale: scale (Output: " + out_name + ")").c_str());
      } else {
        // add bias
        /* TODO：待AIE支持UNIFORM mode */
        layer = engine_->network()->AddScaleNd(input, AscendIE::ScaleMode::KCHANNEL,
          shift_weights.get(),
          power_weights.get(),
          channelAxis);
        layer->GetOutput(0)->SetName(
            ("bias_before_scale：bias_out: " + out_name).c_str());
        layer->SetName(
            ("Scale: scale_bias (Output: " + out_name + ")").c_str());
        // mul scale
        /* TODO：待AIE支持UNIFORM mode */
        layer = engine_->network()->AddScaleNd(input, AscendIE::ScaleMode::KCHANNEL,
          power_weights.get(),
          scale_weights.get(),
          channelAxis);
        layer->GetOutput(0)->SetName(
            ("bias_before_scale：scale_out: " + out_name).c_str());
        layer->SetName(
            ("Scale: scale_scale (Output: " + out_name + ")").c_str());
      }

      PADDLE_ENFORCE_EQ(layer != nullptr, true,
                        platform::errors::Fatal("Create scale layer failed."));

      if (input_dim.Size() < 3) {
        std::vector<int64_t> squeeze_shape_vector(input_dim.Size());
        for (int i = 0; i < input_dim.Size(); i++) {
          squeeze_shape_vector[i] = input_dim[i] < 0 ? 0 : input_dim[i];
        }
        AscendIE::Dims squeeze_shape(squeeze_shape_vector);

        squeeze_layer = engine_->network()->AddShuffle(layer->GetOutput(0));
        squeeze_layer->SetReshapeDimensions(squeeze_shape);
        layer = static_cast<AscendIE::BaseLayer*>(squeeze_layer);
        layer->GetOutput(0)->SetName(
            ("after_reshape_out: " + out_name).c_str());
        layer->SetName(
            ("Scale: Shuffle_reshape (Output: " + out_name + ")").c_str());
      }
    }
    RreplenishLayerAndOutput(layer, "scale", {out_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(scale, ScaleOpConverter);
