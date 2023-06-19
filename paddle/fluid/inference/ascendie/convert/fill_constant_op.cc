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

class FillConstantOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fill_constant op to ascendie fill_constant layer";

    framework::OpDesc op_desc(op, nullptr);
    phi::ProtoDataType dtype = static_cast<phi::ProtoDataType>(PADDLE_GET_CONST(int, op_desc.GetAttr("out_dtype")));
    std::string str_value =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("str_value"));
    std::vector<int64_t> shape =
        PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("shape"));
    if (str_value == "") {
      float value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
      str_value = std::to_string(value);
    }
    std::unique_ptr<phi::DenseTensor> out_tensor(new phi::DenseTensor());
    out_tensor->Resize(phi::make_ddim(shape));
    AscendIE::DataType trt_dtype = AscendIE::DataType::FLOAT;
    void* trt_data = nullptr;
    size_t trt_num;
    if (dtype == phi::ProtoDataType::INT32 || dtype == phi::ProtoDataType::INT64) {
      int* tmp_ptr = out_tensor->mutable_data<int>(platform::CPUPlace());
      for (int64_t i = 0; i < out_tensor->numel(); i++)
        tmp_ptr[i] = std::stoi(str_value);
      trt_dtype = AscendIE::DataType::INT32;
      trt_data = static_cast<void*>(tmp_ptr);
    } else if (dtype == phi::ProtoDataType::FP32) {
      float* tmp_ptr = out_tensor->mutable_data<float>(platform::CPUPlace());
      for (int64_t i = 0; i < out_tensor->numel(); i++)
        tmp_ptr[i] = std::stof(str_value);
      trt_data = static_cast<void*>(tmp_ptr);
    }

    trt_num = static_cast<size_t>(out_tensor->numel());
    engine_->SetWeights("fill_constant_value", std::move(out_tensor));
    AscendIEEngine::Weight weight{trt_dtype, trt_data, trt_num};

    AscendIE::Dims trt_in_shape(shape.size(), &shape[0]);
    AscendIE::BaseLayer* layer = engine_->network()->AddConstantLayer(trt_in_shape, weight.get());
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "fill_constant", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(fill_constant, FillConstantOpConverter);
