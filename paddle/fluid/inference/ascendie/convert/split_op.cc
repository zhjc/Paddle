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

class SplitOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a split op to tensorrt split layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    AscendIE::Tensor* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto inputs = op_desc.Inputs();
    AscendIE::Dims input_dims = input->GetDimensions();
    int output_num = op_desc.Output("Out").size();

    // Get Attrs
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    int num = 0;
    std::vector<int> output_lengths =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("sections"));
    if (op_desc.HasAttr("num")) {
      num = PADDLE_GET_CONST(int, op_desc.GetAttr("num"));
    }
    AscendIE::Tensor* shape_tensor = nullptr;
    if (engine_->with_dynamic_shape()) {
      axis += (axis < 0) ? input_dims.Size() : 0;
      // only be called in dynamic_shape mode
      shape_tensor = Shape(input);
    } else {
      axis += (axis < 0) ? input_dims.Size() : -1;
    }
    bool in_axis_dim_dynamic = false;
    bool sections_tensor_list = false;
    AscendIE::Tensor* sections_tensor = nullptr;

    // need infer output_lengths
    if (inputs.find("SectionsTensorList") != inputs.end() &&
        op_desc.Input("SectionsTensorList").size() >= 1) {
      int32_t sections_size = op_desc.Input("SectionsTensorList").size();
      std::vector<AscendIE::Tensor*> sections_tensors;
      for (int32_t i = 0; i < sections_size; ++i) {
        sections_tensors.push_back(
            engine_->GetITensor(op_desc.Input("SectionsTensorList")[i]));
      }
      sections_tensor = Concat(sections_tensors);
      sections_tensor_list = true;
    } else if (!output_lengths.empty()) {
      sections_tensor = Add1DConstantLayer(output_lengths);
    } else if (num > 0 && output_lengths.empty()) {
      if (input_dims[axis] > 0) {
        int64_t in_axis_dim = input_dims[axis];
        size_t out_axis_dim = in_axis_dim / num;
        for (int i = 0; i < num; ++i) {
          output_lengths.push_back(out_axis_dim);
        }
        sections_tensor = Add1DConstantLayer(output_lengths);
      } else {
        in_axis_dim_dynamic = true;
        auto* num_tensor = Add1DConstantLayer(num);
        sections_tensor =
            Div(GetEleTensorOfShape(shape_tensor, axis), num_tensor);
      }
    }

    AscendIE::BaseLayer* layer = nullptr;
    // if (engine_->with_dynamic_shape()) {
    int64_t shape[1] = {1};
    AscendIE::Dims trt_step_dims(input->GetDimensions().Size(), shape);

    std::vector<int32_t> gather_indices;
    gather_indices.resize(trt_step_dims.Size());
    std::iota(gather_indices.begin(), gather_indices.end(), 0);
    gather_indices[axis] = gather_indices.size();
    std::vector<int32_t> zeros(trt_step_dims.Size(), 0);
    std::vector<int32_t> stride(trt_step_dims.Size(), 1);
    AscendIE::Tensor* zeros_tensor = Add1DConstantLayer(zeros);
    AscendIE::Tensor* stride_tensor = Add1DConstantLayer(stride);
    // input : [N,C,H,W]
    AscendIE::Tensor* start_point_tensor = zeros_tensor;
    AscendIE::Tensor* this_len_tensor = zeros_tensor;
    for (int i = 0; i < output_num; i++) {
    if (sections_tensor_list || !in_axis_dim_dynamic) {
        start_point_tensor = Sum(start_point_tensor, this_len_tensor);
        this_len_tensor = Gather(sections_tensor, std::vector<int32_t>{i});
    } else {
        this_len_tensor = sections_tensor;
        AscendIE::Tensor* i_tensor = Add1DConstantLayer(static_cast<int>(i));
        start_point_tensor = Prod(i_tensor, sections_tensor);
    }

    std::vector<AscendIE::Tensor*> concat_inputs1 = {zeros_tensor,
                                                        start_point_tensor};
    std::vector<AscendIE::Tensor*> concat_inputs2 = {shape_tensor,
                                                        this_len_tensor};
    AscendIE::Tensor* start_tensor = Gather(Concat(concat_inputs1), gather_indices);
    AscendIE::Tensor* size_tensor = Gather(Concat(concat_inputs2), gather_indices);
    layer = engine_->network()->AddSlice(input, start_tensor->GetDimensions(), size_tensor->GetDimensions(), stride_tensor->GetDimensions());

    auto output_name = op_desc.Output("Out")[i];
    RreplenishLayerAndOutput(layer, "split", {output_name}, test_mode);
    }
    // } else {
    //   auto chw_input_dims = input->getDimensions();
    //   AscendIE::Dims trt_start_dims(chw_input_dims.Size(), );
    //   memset(trt_start_dims.d, 0, sizeof(int32_t) * chw_input_dims.Size());
    //   AscendIE::Dims trt_size_dims = chw_input_dims;
    //   AscendIE::Dims trt_step_dims;
    //   trt_step_dims.nbDims = chw_input_dims.nbDims;
    //   for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

    //   // input : [C,H,W]
    //   for (int i = 0; i < output_num; i++) {
    //     trt_start_dims.d[axis] = std::accumulate(
    //         output_lengths.begin(), output_lengths.begin() + i, 0);
    //     trt_size_dims.d[axis] = output_lengths[i];
    //     layer = TRT_ENGINE_ADD_LAYER(engine_,
    //                                  Slice,
    //                                  *input,
    //                                  trt_start_dims,
    //                                  trt_size_dims,
    //                                  trt_step_dims);
    //     auto output_name = op_desc.Output("Out")[i];
    //     RreplenishLayerAndOutput(layer, "split", {output_name}, test_mode);
    //   }
    // }
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(split, SplitOpConverter);
