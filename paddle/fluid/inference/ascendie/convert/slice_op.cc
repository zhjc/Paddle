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

class SliceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {

    VLOG(4) << "convert slice op to ascendie layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto output_name = op_desc.Output("Out")[0];

    float out_scale = 1;
    // if (op_desc.HasAttr("out_threshold")) {
    //   out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
    //   engine_->SetTensorDynamicRange(input, out_scale);
    // }
    // std::vector<int> axes =
    //     PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    // std::vector<int> starts =
    //     PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    // std::vector<int> ends =
    //     PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));
    // std::vector<int> decrease_axises =
    //     PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("decrease_axis"));
    // auto input_dims = input->GetDimensions();
    // AscendIE::BaseLayer* layer = nullptr;

    // if (engine_->with_dynamic_shape()) {
    //   auto* shape_tensor = Shape(input);
    //   AscendIE::Dims aie_start_dims(input_dims);

    //   AscendIE::Dims aie_size_dims = aie_start_dims;
    //   AscendIE::Dims aie_step_dims = aie_start_dims;
    //   for (int i = 0; i < aie_step_dims.Size(); i++) aie_step_dims[i] = 1;
    //   AscendIE::Tensor* start_tensor = nullptr;
    //   AscendIE::Tensor* end_tensor = nullptr;

    //   std::vector<AscendIE::Tensor*> starts_tensor;
    //   std::vector<AscendIE::Tensor*> ends_tensor;
    //   for (int32_t i = 0; i < input_dims.Size(); ++i) {
    //     starts_tensor.push_back(Add1DConstantLayer(0));
    //     ends_tensor.push_back(GetEleTensorOfShape(shape_tensor, i));
    //   }
    //   auto slice_inputs = op_desc.Inputs();
    //   if (slice_inputs.find("StartsTensor") != slice_inputs.end() &&
    //       op_desc.Input("StartsTensor").size()) {  // has StartsTensor input
    //     for (size_t i = 0; i < axes.size(); ++i) {
    //       starts_tensor[axes[i]] = GetEleTensorOfShape(
    //           engine_->GetITensor(op_desc.Input("StartsTensor")[0]), i);
    //     }
    //   } else {
    //     PADDLE_ENFORCE_EQ(starts.size(),
    //                       axes.size(),
    //                       platform::errors::InvalidArgument(
    //                           "The size of this starts: %d must be "
    //                           "equal to the axes: %d.",
    //                           starts.size(),
    //                           axes.size()));
    //     for (size_t i = 0; i < axes.size(); i++) {  // same as starts.size()
    //       if (starts[i] < 0) {
    //         starts_tensor[axes[i]] =
    //             Max(Sum(Add1DConstantLayer(starts[i]),
    //                     GetEleTensorOfShape(shape_tensor, axes[i])),
    //                 Add1DConstantLayer(0));
    //       } else {
    //         starts_tensor[axes[i]] =
    //             Min(Add1DConstantLayer(starts[i]),
    //                 GetEleTensorOfShape(shape_tensor, axes[i]));
    //       }
    //     }
    //   }
    //   start_tensor = Concat(starts_tensor);

    //   if (slice_inputs.find("EndsTensor") != slice_inputs.end() &&
    //       op_desc.Input("EndsTensor").size()) {  // has EndsTensor input
    //     for (size_t i = 0; i < axes.size(); ++i) {
    //       ends_tensor[axes[i]] = GetEleTensorOfShape(
    //           engine_->GetITensor(op_desc.Input("EndsTensor")[0]), i);
    //     }
    //   } else {
    //     PADDLE_ENFORCE_EQ(ends.size(),
    //                       axes.size(),
    //                       platform::errors::InvalidArgument(
    //                           "The size of this ends: %d must be "
    //                           "equal to the axes: %d.",
    //                           ends.size(),
    //                           axes.size()));
    //     for (size_t i = 0; i < axes.size(); i++) {  // same as ends.size()
    //       if (ends[i] < 0) {
    //         ends_tensor[axes[i]] =
    //             Max(Sum(Add1DConstantLayer(ends[i]),
    //                     GetEleTensorOfShape(shape_tensor, axes[i])),
    //                 Add1DConstantLayer(0));
    //       } else {
    //         ends_tensor[axes[i]] =
    //             Min(Add1DConstantLayer(ends[i]),
    //                 GetEleTensorOfShape(shape_tensor, axes[i]));
    //       }
    //     }
    //   }
    //   end_tensor = Concat(ends_tensor);
    //   auto* size_tensor = Sub(end_tensor, start_tensor);
    //   layer = engine_->network()->AddSlice(*input, aie_start_dims, aie_size_dims, aie_step_dims);
    //   layer->SetInput(1, start_tensor);
    //   layer->SetInput(2, size_tensor);

    //   if (decrease_axises.size() > 0) {
    //     std::vector<int32_t> gather_indices;
    //     for (int i = 0; i < aie_size_dims.Size(); i++) {
    //       if (decrease_axises.end() !=
    //           std::find(decrease_axises.begin(), decrease_axises.end(), i))
    //         continue;
    //       gather_indices.push_back(i);
    //     }
    //     if (gather_indices.empty())
    //       gather_indices.push_back(decrease_axises[0]);
    //     auto real_size_tensor = Gather(size_tensor, gather_indices);
    //     layer = engine_->network()->AddShuffle(*layer->GetInput(0))
    //     layer->SetInput(1, real_size_tensor);
    //   }
    // } else {
    //   // notice that input shape is [CHW] without batch axis when input has
    //   // static shape
    //   for (size_t i = input_dims.Size(); i > 0; i--) {
    //     input_dims[i] = input_dims[i - 1];
    //   }
    //   input_dims[0] = 1;  // fake batchsize, not useful here
    //   for (size_t i = 0; i < axes.size(); i++) {
    //     if (starts[i] < 0) {
    //       starts[i] = std::max(starts[i] + input_dims[axes[i]], static_cast<int64_t>(0));
    //     }
    //     if (ends[i] < 0) {
    //       ends[i] = std::max(ends[i] + input_dims[axes[i]], static_cast<int64_t>(0));
    //     }
    //     ends[i] = std::min(static_cast<int64_t>(ends[i]), input_dims[axes[i]]);
    //     PADDLE_ENFORCE_GT(
    //         ends[i],
    //         starts[i],
    //         platform::errors::InvalidArgument(
    //             "Attr(ends) should be greater than attr(starts) in "
    //             "slice op. But received ends = %d, starts = %d.",
    //             ends[i],
    //             starts[i]));
    //   }
    //   auto chw_input_dims = input->GetDimensions();
    //   AscendIE::Dims aie_start_dims(chw_input_dims);

    //   AscendIE::Dims aie_size_dims = chw_input_dims;
    //   AscendIE::Dims aie_step_dims(chw_input_dims);

    //   for (int i = 0; i < aie_step_dims.Size(); i++) aie_step_dims[i] = 1;

    //   // input : [C,H,W]
    //   for (size_t i = 0; i < axes.size(); i++) {
    //     int aie_axis = axes[i] - 1;
    //     aie_start_dims[aie_axis] = starts[i];
    //     aie_size_dims[aie_axis] = ends[i] - starts[i];
    //   }
    //   layer = engine_->network()->AddSlice(*input, aie_start_dims, aie_size_dims, aie_step_dims);
    //   AscendIE::Dims real_aie_size_dims;

    //   AscendIE::Dims fill_from_aie_size_dims(aie_size_dims);
    //   int filled_cnt = 0;

    //   if (decrease_axises.size() > 0) {
    //     for (size_t i = 0; i < decrease_axises.size(); i++) {
    //       decrease_axises[i]--;
    //     }
    //     for (int i = 0; i < aie_size_dims.Size(); i++) {
    //       if (decrease_axises.end() !=
    //           std::find(decrease_axises.begin(), decrease_axises.end(), i))
    //         continue;
    //       fill_from_aie_size_dims[filled_cnt] = aie_size_dims[i];
    //       filled_cnt += 1;
    //     }
    //     if (filled_cnt == 0) {
    //       AscendIE::Dims tmp_dims({1});
    //       real_aie_size_dims = tmp_dims;
    //     } else {
    //       AscendIE::Dims tmp_dims(filled_cnt, fill_from_aie_size_dims.Data());
    //       real_aie_size_dims = tmp_dims;
    //     }
    //     auto reshape_layer = engine_->network()->AddShuffle(*layer->GetInput(0))
    //     reshape_layer->setReshapeDimensions(real_aie_size_dims);
    //     layer = static_cast<AscendIE::BaseLayer*>(reshape_layer);
    //   }
    // }
    // RreplenishLayerAndOutput(layer, "slice", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(slice, SliceOpConverter);
