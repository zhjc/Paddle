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
// #include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace ascendie {

/*
 * Gelu converter from fluid to tensorRT.
 */
/*
 * Gelu converter from fluid to tensorRT.
 */
class GeluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert gelu op to tensorrt gelu layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    AscendIE::BaseLayer* layer = nullptr;
    if (op_desc.HasAttr("approximate") &&
        PADDLE_GET_CONST(bool, op_desc.GetAttr("approximate"))) {
      AscendIE::Dims input_shape = input->GetDimensions();
      for (int i = 0; i < input_shape.Size(); ++i) {
        input_shape[i] = 1;
      }

      std::string out_name = op_desc.Output("Out").front();
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_gelu_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };

      float* constant_pow = create_weights(3.0f, "constant_pow");
      float* constant_multiply = create_weights(0.044715f, "constant_multiply");
      float* constant_sqrt =
          create_weights(0.79788456080286535587989211986876f, "constant_sqrt");
      float* constant_one = create_weights(1.0f, "constant_one");
      float* constant_half = create_weights(0.5f, "constant_half");
	  
	  AscendIE::ConstantLayer* constant_layer_pow = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_pow), 1});
	  AscendIE::ConstantLayer* constant_layer_multiply = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_multiply), 1});  
	  AscendIE::ConstantLayer* constant_layer_sqrt = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_sqrt), 1});
	  AscendIE::ConstantLayer* constant_layer_one = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_one), 1});
      AscendIE::ConstantLayer* constant_layer_half = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_half), 1});
      
      AscendIE::ElementWiseLayer* layer_pow = engine_->network()->AddElementWise(
          input,
          constant_layer_pow->GetOutput(0),
          AscendIE::ElementWiseOperation::POW);

      AscendIE::ElementWiseLayer* layer_mul = engine_->network()->AddElementWise(
          layer_pow->GetOutput(0),
          constant_layer_multiply->GetOutput(0),
          AscendIE::ElementWiseOperation::MUL); // need PROD==MUL.
      
      AscendIE::ElementWiseLayer* layer_add = engine_->network()->AddElementWise(
          layer_mul->GetOutput(0),
          input,
          AscendIE::ElementWiseOperation::ADD); // need SUM==ADD.

      AscendIE::ElementWiseLayer* layer_sqrt = engine_->network()->AddElementWise(
          layer_add->GetOutput(0),
          constant_layer_sqrt->GetOutput(0),
          AscendIE::ElementWiseOperation::MUL);
      
      AscendIE::ActivationLayer* layer_tanh = engine_->network()->AddActivationLayer(
          layer_sqrt->GetOutput(0),
          AscendIE::ActivationKind::TANH);

      AscendIE::ElementWiseLayer* layer_one = engine_->network()->AddElementWise(
          layer_tanh->GetOutput(0),
          constant_layer_one->GetOutput(0),
          AscendIE::ElementWiseOperation::ADD);

      AscendIE::ElementWiseLayer* layer_CDF = engine_->network()->AddElementWise(
          layer_one->GetOutput(0),
          constant_layer_half->GetOutput(0),
          AscendIE::ElementWiseOperation::MUL);
      
      AscendIE::ElementWiseLayer* y = engine_->network()->AddElementWise(
          layer_CDF->GetOutput(0),
          input,
          AscendIE::ElementWiseOperation::MUL);
      
      layer = y;

    } else {

      AscendIE::Dims input_shape = input->GetDimensions();
      for (int i = 0; i < input_shape.Size(); ++i) {
        input_shape[i] = 1;
      }

      std::string out_name = op_desc.Output("Out").front();
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_gelu_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };

      float* constant_one = create_weights(1.0f, "constant_one");
      float* constant_half = create_weights(0.5f, "constant_half");
      float* constant_rsqrt2 =
          create_weights(0.70710678118f, "constant_rsqrt2");

      AscendIE::ConstantLayer* constant_layer_one = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_one), 1});
      AscendIE::ConstantLayer* constant_layer_half = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_half), 1});
      AscendIE::ConstantLayer* constant_layer_rsqrt2 = engine_->network()->AddConstantLayer(
	      input_shape, 
		  AscendIE::WeightsBuf{
              AscendIE::DataType::FLOAT, static_cast<void*>(constant_rsqrt2), 1});
      
      AscendIE::ElementWiseLayer* layer_mul = engine_->network()->AddElementWise(
          input,
          constant_layer_rsqrt2->GetOutput(0),
          AscendIE::ElementWiseOperation::MUL);

      AscendIE::UnaryLayer* layer_erf = engine_->network()->AddUnary(
          layer_mul->GetOutput(0),
          AscendIE::UnaryOperation::EXP); // TODO: need use AscendIE::UnaryOperation::ERF.
      
      AscendIE::ElementWiseLayer* layer_add = engine_->network()->AddElementWise(
          layer_erf->GetOutput(0),
          constant_layer_one->GetOutput(0),
          AscendIE::ElementWiseOperation::ADD);
      AscendIE::ElementWiseLayer* layer_CDF = engine_->network()->AddElementWise(
          layer_add->GetOutput(0),
          constant_layer_half->GetOutput(0),
          AscendIE::ElementWiseOperation::MUL);
      AscendIE::ElementWiseLayer* y = engine_->network()->AddElementWise(
          layer_CDF->GetOutput(0),
          input,
          AscendIE::ElementWiseOperation::MUL);
      
      layer = y;
    }

    auto output_name = op_desc.Output("Out")[0];

    RreplenishLayerAndOutput(layer, "gelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(gelu, GeluOpConverter);
