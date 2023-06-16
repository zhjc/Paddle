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
 * Pad3dOp.
 */
class Pad3dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a pad3d op to tensorrt pad3d layer";

    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    AscendIE::Tensor* paddings;
    if (op_desc.HasInput("Paddings") && op_desc.Input("Paddings").size() > 0) {
      paddings = engine_->GetITensor(op_desc.Input("Paddings")[0]);
    } else {
      std::vector<int> paddings_v =
          PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
      paddings = Add1DConstantLayer(paddings_v);
    }

    float value{0.F};
    if (op_desc.HasAttr("value")) {
      value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
    }

    std::string padding_mode = "constant";
    if (op_desc.HasAttr("mode")) {
      padding_mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    }

    const int input_dim = input->GetDimensions().Size();
    const int pad_size = paddings->GetDimensions()[0];
    PADDLE_ENFORCE_EQ(input_dim * 2 - 4,
                      pad_size,
                      phi::errors::InvalidArgument(
                          "Expected paddings size is %d, but received %d.",
                          input_dim * 2 - 4,
                          pad_size));
    // convert paddle pad to tensorrt pad
    std::vector<int> shuffle_index{4, 2, 0, 5, 3, 1};
    std::vector<AscendIE::Tensor*> shuffle_inputs;
    for (int i = 0; i < pad_size; i++) {
      shuffle_inputs.push_back(GetEleTensorOfShape(paddings, shuffle_index[i]));
    }
    paddings = Concat(shuffle_inputs);
    auto* pre_zeros = Add1DConstantLayer(std::vector<int>(2, 0));
    auto start_slice1 = AscendIE::Dims{1, { 0 }};
    auto start_slice2 = AscendIE::Dims{1, { 3 }};
    auto size_slice = AscendIE::Dims{1, { 3 }};
    auto stride_slice = AscendIE::Dims{1, { 1 }};

    auto* pre_pad = engine_->network()->AddSlice(paddings, start_slice1, size_slice, stride_slice)
            ->GetOutput(0);
    pre_pad = Concat(std::vector<AscendIE::Tensor*>{pre_zeros, pre_pad});
    auto* post_pad = engine_->network()->AddSlice(paddings, start_slice2, size_slice, stride_slice)
            ->GetOutput(0);
    post_pad = Concat(std::vector<AscendIE::Tensor*>{pre_zeros, post_pad});

    std::vector<int> zeros_v(input_dim, 0);
    auto const zeros = Add1DConstantLayer(zeros_v);

    AscendIE::Tensor* start{};
    AscendIE::Tensor* size{};
    // elementwise add zeros and pre_pad
    start = engine_->network()->AddElementWise(zeros,
                                               pre_pad,
                                               AscendIE::ElementWiseOperation::SUB)
                ->GetOutput(0);

    auto const total_padding = engine_->network()
                                   ->AddElementWise(pre_pad,
                                                    post_pad,
                                                    AscendIE::ElementWiseOperation::ADD)
                                   ->GetOutput(0);

    auto* input_shape = Shape(input);
    size = engine_->network()->AddElementWise(input_shape,
                                              total_padding,
                                              AscendIE::ElementWiseOperation::ADD)
               ->GetOutput(0);
    // add slice layer
    std::vector<int64_t> dimVec;
    for (size_t i = 0; i < input_dim; i++) {
      dimVec.push_back(1);
    }
    AscendIE::Dims stride(dimVec);

    auto const& dummy = stride;
    auto* slice_layer = engine_->network()->AddSlice(
                            const_cast<AscendIE::Tensor*>(input),
                            dummy,
                            dummy,
                            stride);
    slice_layer->SetInput(1, start);
    slice_layer->SetInput(2, size);
    if (padding_mode == "constant") {
// #if IS_TRT_VERSION_GE(8500)
//       slice_layer->setMode(AscendIE::SampleMode::kFILL);
// #else
//       slice_layer->setMode(AscendIE::SliceMode::kFILL);
// #endif
      if (value != 0.F) {
        AscendIE::Tensor* fill_value = nullptr;
        switch (input->GetType()) {
          case AscendIE::DataType::FLOAT:
          case AscendIE::DataType::FLOAT16:
          case AscendIE::DataType::INT8: {
            fill_value = Add1DConstantLayer(value);
            break;
          }
          default: {
            int value_int = static_cast<int>(value);
            fill_value = Add1DConstantLayer(value_int);
            break;
          }
        }
        slice_layer->SetInput(4, fill_value);
      }
    } else if (padding_mode == "reflect") {
// #if IS_TRT_VERSION_GE(8500)
//       slice_layer->setMode(AscendIE::SampleMode::kREFLECT);
// #else
//       slice_layer->setMode(AscendIE::SliceMode::kREFLECT);
// #endif
    } else if (padding_mode == "replicate") {
// #if IS_TRT_VERSION_GE(8500)
//       slice_layer->setMode(AscendIE::SampleMode::kCLAMP);
// #else
//       slice_layer->setMode(AscendIE::SliceMode::kCLAMP);
// #endif
    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unsupported mode: %s",
                                                   padding_mode));
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(slice_layer, "pad3d", {output_name}, test_mode);


  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(pad3d, Pad3dOpConverter);
