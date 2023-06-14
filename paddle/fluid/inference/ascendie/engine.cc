/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/ascendie/engine.h"
#include <glog/logging.h>
#include <string>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace inference {
namespace ascendie {

int AscendIEEngine::runtime_batch_ = 1;
thread_local int AscendIEEngine::predictor_id_per_thread = -1;

void AscendIEEngine::Weight::SetDataType(phi::DataType type) {
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
          "Paddle-AIE loads weighths failed, found not supported data type %s.",
          type);
      break;
  }
  w_.type = nv_type;
}

void AscendIEEngine::InitNetwork() {
  freshDeviceId();
  infer_builder_.reset(AscendIE::Builder::CreateInferBuilder("Ascend910A"));

  infer_network_.reset(infer_builder_->CreateNetwork());

  // infer_builder_config_.reset(infer_builder_->createBuilderConfig());
  // optim_profiles_.resize(max_profile_num_);
  // for (int i = 0; i < max_profile_num_; i++)
  //   optim_profiles_[i] = infer_builder_->createOptimizationProfile();
}
AscendIE::Context *AscendIEEngine::context() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (infer_context_.find(predictor_id_per_thread) == infer_context_.end()) {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::InvalidArgument(
            "You should build engine first and then set the context."));
    // We may see aie warning: Profile 0 has been chosen by another
    // IExecutionContext...
    // It's ok. We will set it later.
    AscendIE::Context *infer_context = infer_engine_->CreateContext();

    PADDLE_ENFORCE_NOT_NULL(
        infer_context,
        platform::errors::InvalidArgument(
            "Ascend engine can not build execution context."));
    // if (with_dynamic_shape_) {
    //   // need new profile if it's not the first
    //   if (cur_profile_num_ > 0) {
    //     infer_context->setOptimizationProfile(cur_profile_num_);
    //   }
    //   profile_index_[predictor_id_per_thread] = cur_profile_num_;
    //   ++cur_profile_num_;
    // }
    infer_context_[predictor_id_per_thread].reset(infer_context);
  }
  return infer_context_[predictor_id_per_thread].get();
}

void AscendIEEngine::Execute(int batch_size,
                             std::vector<void *> *buffers,
                             AscendIE::aieStream stream) {
  freshDeviceId();
  auto infer_context = context();
  Enqueue(infer_context, buffers, batch_size, stream);
}

bool AscendIEEngine::Enqueue(AscendIE::Context *context,
                             std::vector<void *> *buffers,
                             int batch_size,
                             AscendIE::aieStream stream) {
  bool ret = context->Enqueue(stream);

  SetRuntimeBatch(batch_size);
  return ret;
}


// void AscendIEEngine::FreezeNetwork() {

// }

AscendIE::Tensor *AscendIEEngine::DeclareInput(const std::string &name,
                                                AscendIE::DataType dtype,
                                                const AscendIE::Dims &dims) {
  PADDLE_ENFORCE_EQ(network() != nullptr,
                    true,
                    platform::errors::InvalidArgument(
                        "The AIE network should be initialized first."));
  AscendIE::Tensor *input = network()->AddInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE_NOT_NULL(
      input,
      platform::errors::InvalidArgument("Adding input %s failed in "
                                        "ascend inference network. "
                                        "Please recheck your input.",
                                        name));
  PADDLE_ENFORCE_EQ(input->IsInput(),
                    true,
                    platform::errors::InvalidArgument(
                        "Input %s is not the input of AIE inference network. "
                        "Please recheck your input.",
                        name));
  AscendIEEngine::SetITensor(name, input);
  return input;
}

void AscendIEEngine::DeclareOutput(const AscendIE::BaseLayer *layer,
                                   int offset,
                                   const std::string &name) {
  auto *output = layer->GetOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      platform::errors::InvalidArgument(
          "The output %s of AIE engine should not be null.", name));
  output->SetName(name.c_str());
  PADDLE_ENFORCE_EQ(output->IsInput(),
                    false,
                    platform::errors::InvalidArgument(
                        "The output %s of AIE engine should not be the input "
                        "of the network at the same time.",
                        name));
  this->network()->SetAsOutput(output);
  PADDLE_ENFORCE_EQ(
      output->IsOutput(),
      true,
      platform::errors::InvalidArgument(
          "The output %s of AIE engine should be the output of the network.",
          name));
}

void AscendIEEngine::DeclareOutput(const std::string &name) {
  auto *output = AscendIEEngine::GetITensor(name);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      platform::errors::InvalidArgument(
          "The output %s of AIE engine should not be null.", name));
  output->SetName(name.c_str());
  PADDLE_ENFORCE_EQ(output->IsInput(),
                    false,
                    platform::errors::InvalidArgument(
                        "The output %s of AIE engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->SetAsOutput(output);
}

void AscendIEEngine::DeclareOutput(const std::string &name,
                                   AscendIE::DataType dtype) {
  auto *output = AscendIEEngine::GetITensor(name);
  DeclareOutput(name);
  output->SetType(dtype);
}

void AscendIEEngine::DeleteITensor(const std::string &name,
                                   AscendIE::Tensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      platform::errors::InvalidArgument(
          "Tensor named %s of AIE engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      true,
      itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of AIE engine should not be null", name));
  itensor_map_.erase(name);
}

void AscendIEEngine::SetITensor(const std::string &name,
                                AscendIE::Tensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      platform::errors::InvalidArgument(
          "Tensor named %s of AIE engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      0,
      itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of AIE engine should not be duplicated", name));
  itensor_map_[name] = tensor;
}

AscendIE::Tensor *AscendIEEngine::GetITensor(const std::string &name,
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
AscendIE::Tensor *AscendIEEngine::ConvertWeight2ITensor(
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

  AscendIE::Dims aie_in_shape(var_dims.size(), var_dims.Get());
  if (scalar) {
    aie_in_shape = AscendIE::Dims(0, &var_dims[0]);
  }
  AscendIE::ConstantLayer *constant =
      this->network()->AddConstantLayer(aie_in_shape, weight.get());
  if (!scalar) {
    this->SetITensor(name, constant->GetOutput(0));
  }
  return constant->GetOutput(0);
}

std::unordered_map<std::string, AscendIE::Tensor *>
    *AscendIEEngine::GetITensorMap() {
  return &itensor_map_;
}

void AscendIEEngine::Deserialize(const std::string &engine_serialized_data) {
  freshDeviceId();
  infer_runtime_.reset(AscendIE::Runtime::GetInstance());
  infer_engine_.reset(infer_runtime_->DeserializeEngineFromMem(
      const_cast<char*>(engine_serialized_data.c_str()), engine_serialized_data.size()));

  PADDLE_ENFORCE_NOT_NULL(
      infer_engine_,
      platform::errors::Fatal(
          "Building AIE engine failed when deserializing engine info. "
          "Please check:\n1. Your AIE serialization is generated and loaded "
          "on the same GPU architecture;\n2. The Paddle Inference version of "
          "generating serialization file and doing inference are "
          "consistent."));
}

void AscendIEEngine::SetRuntimeBatch(size_t batch_size) {
  runtime_batch_ = batch_size;
}

AscendIEEngine::Weight AscendIEEngine::GetTrtWeight(
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
                        "twice in AIE OP converter.",
                        name_with_suffix));

  if (weight_tensor.place() == PlaceType::kGPU ||
      weight_tensor.dtype() != phi::DataType::FLOAT32) {
    weight_map[name_with_suffix].reset(new phi::DenseTensor());
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
  }

  AscendIEEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());

  // if aie not support dtype, we need to cast to fp32.
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

int AscendIEEngine::GetRuntimeBatch() { return runtime_batch_; }

void AscendIEEngine::freshDeviceId() {
  // platform::SetDeviceId(device_id_);
}

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle
