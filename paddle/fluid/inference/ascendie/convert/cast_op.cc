/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class CastOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {

    VLOG(3) << "convert a cast op to ascendie";
    framework::OpDesc op_desc(op, nullptr);

    AscendIE::Tensor* input = engine_->GetITensor(op_desc.Input("X")[0]);

    phi::ProtoDataType out_dtype = static_cast<phi::ProtoDataType>(PADDLE_GET_CONST(int, op_desc.GetAttr("out_dtype")));

    AscendIE::CastLayer* cast = nullptr;

    switch (out_dtype) {
      case phi::ProtoDataType::BOOL:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::BOOL);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::BOOL);
        break;
      case phi::ProtoDataType::INT16:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::INT16);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::INT16);
      case phi::ProtoDataType::INT32:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::INT32);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::INT32);
      case phi::ProtoDataType::INT64:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::INT64);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::INT64);
        break;
      case phi::ProtoDataType::UINT8:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::UINT8);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::UINT8);
        break;
      case phi::ProtoDataType::INT8:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::INT8);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::INT8);
        break;
      case phi::ProtoDataType::FP16:
        cast = engine_->network()->AddCast(input, AscendIE::DataType::FLOAT16);
        if(cast != nullptr) cast->GetOutput(0)->SetType(AscendIE::DataType::FLOAT16);
        break;
      // case phi::ProtoDataType::FP32:
      // case phi::ProtoDataType::FP64:
      // case phi::ProtoDataType::RAW:
      // case phi::ProtoDataType::BF16:
      // case phi::ProtoDataType::COMPLEX64:
      // case phi::ProtoDataType::COMPLEX128:
      //   LOG(ERROR) << "Unsupport type(" << out_dtype
      //              << ") to a Ascend DataType";
        break;
      default:
        LOG(ERROR) << "Unable to convert a fluid data type(" << out_dtype
                   << ") to a Ascend DataType";
        break;
    }

    if(cast == nullptr) {
      LOG(ERROR) << "cast is nullptr";
      return;
    }
    engine_->network()->SetAsOutput(cast->GetOutput(0));

    auto output_name = op_desc.Output("Out")[0];

    RreplenishLayerAndOutput(cast, "cast", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(cast, CastOpConverter);
