/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/engine.h"

#include <NvInfer.h>
#include <glog/logging.h>

#include <string>

#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"  // NOLINT
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

int AscendEngine::runtime_batch_ = 1;
thread_local int AscendEngine::predictor_id_per_thread = -1;

void AscendEngine::Weight::SetDataType(phi::DataType type) {
  AscendIE::DataType nv_type = AscendIE::DataType::FLOAT;
  switch (type) {
    case phi::DataType::FLOAT32:
      nv_type = AscendIE::DataType::FLOAT;
      break;
    case phi::DataType::FLOAT16:
      nv_type = AscendIE::DataType::FLOAT16;
      break;
    case phi::DataType::INT32:
      nv_type = AscendIE::DataType::INT32;
      break;
    case phi::DataType::INT8:
      nv_type = AscendIE::DataType::INT8;
      break;
    case phi::DataType::BOOL:
      nv_type = AscendIE::DataType::BOOL;
      break;
    default:
      paddle::platform::errors::InvalidArgument(
          "Paddle-TRT loads weighths failed, found not supported data type %s.",
          type);
      break;
  }
  w_.type = nv_type;
}

AscendIE::Tensor *AscendEngine::DeclareInput(const std::string &name,
                                                AscendIE::DataType dtype,
                                                const AscendIE::Dims &dims) {
  PADDLE_ENFORCE_EQ(network() != nullptr,
                    true,
                    platform::errors::InvalidArgument(
                        "The TRT network should be initialized first."));
  AscendIE::Tensor *input = network()->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE_NOT_NULL(
      input,
      platform::errors::InvalidArgument("Adding input %s failed in "
                                        "TensorRT inference network. "
                                        "Please recheck your input.",
                                        name));
  PADDLE_ENFORCE_EQ(input->isNetworkInput(),
                    true,
                    platform::errors::InvalidArgument(
                        "Input %s is not the input of TRT inference network. "
                        "Please recheck your input.",
                        name));
  AscendEngine::SetITensor(name, input);
  return input;
}

void AscendEngine::DeclareOutput(const AscendIE::BaseLayer *layer,
                                   int offset,
                                   const std::string &name) {
  auto *output = layer->getOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(),
                    false,
                    platform::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
  PADDLE_ENFORCE_EQ(
      output->isNetworkOutput(),
      true,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should be the output of the network.",
          name));
}

void AscendEngine::DeclareOutput(const std::string &name) {
  auto *output = AscendEngine::GetITensor(name);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(),
                    false,
                    platform::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
}

void AscendEngine::DeclareOutput(const std::string &name,
                                   AscendIE::DataType dtype) {
  auto *output = AscendEngine::GetITensor(name);
  DeclareOutput(name);
  output->setType(dtype);
}

void AscendEngine::DeleteITensor(const std::string &name,
                                   AscendIE::Tensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      true,
      itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null", name));
  itensor_map_.erase(name);
}

void AscendEngine::SetITensor(const std::string &name,
                                AscendIE::Tensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      0,
      itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be duplicated", name));
  itensor_map_[name] = tensor;
}

AscendIE::Tensor *AscendEngine::GetITensor(const std::string &name,
                                              bool scalar) {
  if (scalar) {
    return ConvertWeight2ITensor(name, true);
  }
  if (itensor_map_.count(name)) {
    return itensor_map_[name];
  } else {
    ConvertWeight2ITensor(name);
    return itensor_map_[name];
  }
}

// For cases when input is not middle-tensor , but persistable tensor
// you should call this.
AscendIE::Tensor *AscendEngine::ConvertWeight2ITensor(
    const std::string &name, bool scalar) {
  auto *var_v = scope_->FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var_v,
      platform::errors::NotFound("You are converting a persistable weight to a "
                                 "tensor, but there is no "
                                 "persistable variable called %s in scope.",
                                 name));
  auto *var_t = var_v->GetMutable<phi::DenseTensor>();
  auto weight = this->GetTrtWeight(name, *var_t);

  // Now we have create weights, then we need create a itensor
  auto var_dims = var_t->dims();

  AscendIE::Dims trt_in_shape(var_t->dims().size());
  for (int64_t i = 0; i < trt_in_shape.size(); i++) {
    trt_in_shape[i] = var_dims[i];
  }
  if (scalar) {
    trt_in_shape.nbDims = 0;
    trt_in_shape.d[0] = var_dims[0];
  }
  AscendIE::ConstantLayer *constant =
      this->network().AddConstantLayer(trt_in_shape, weight.get());
  if (!scalar) {
    this->SetITensor(name, constant->GetOutput(0));
  }
  return constant->getOutput(0);
}

std::unordered_map<std::string, AscendIE::ITensor *>
    *AscendEngine::GetITensorMap() {
  return &itensor_map_;
}


AscendEngine::Weight AscendEngine::GetTrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    platform::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));

  if (weight_tensor.place() == PlaceType::kGPU ||
      weight_tensor.dtype() != phi::DataType::FLOAT32) {
    weight_map[name_with_suffix].reset(new phi::DenseTensor());
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
  }

  AscendEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());

  // if trt not support dtype, we need to cast to fp32.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::FLOAT32);
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(platform::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(bf16_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT32);
    weight.SetValues(fp32_data);
  } else if (weight_tensor.dtype() == phi::DataType::INT64) {
    phi::DenseTensor int64_tensor;
    int64_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &int64_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::INT32);
    auto *int32_data = weight_map[name_with_suffix]->mutable_data<int32_t>(
        platform::CPUPlace());
    auto *int64_data = int64_tensor.mutable_data<int64_t>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      int32_data[i] = int64_data[i];
    }
    weight.SetDataType(phi::DataType::INT32);
    weight.SetValues(int32_data);
  } else {
    if (weight_tensor.place() == PlaceType::kGPU) {
      paddle::framework::TensorCopySync(
          weight_tensor, cpu_place, weight_map[name_with_suffix].get());
      weight.SetDataType(weight_tensor.dtype());
      weight.SetValues(weight_map[name_with_suffix]->data());
    } else {
      weight.SetDataType(weight_tensor.dtype());
      weight.SetValues(weight_tensor.data());
    }
  }

  name_suffix_counter += 1;
  return weight;
}


}  // namespace ascendie
}  // namespace inference
}  // namespace paddle
