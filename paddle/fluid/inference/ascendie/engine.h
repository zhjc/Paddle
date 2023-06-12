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

#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "NvInferRuntimeCommon.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/stream.h"
#include "paddle/utils/any.h"

PHI_DECLARE_bool(trt_ibuilder_cache);

namespace paddle {
namespace inference {
namespace ascendie {

using FluidDT = framework::proto::VarType_Type;

namespace {  // NOLINT

AscendIE::DataType FluidDataType2Ascend(FluidDT type) {
  switch (type) {
    case FluidDT::VarType_Type_FP32:
      return AscendIE::DataType::FLOAT;
    case FluidDT::VarType_Type_INT32:
      return AscendIE::DataType::INT32;
    case FluidDT::VarType_Type_INT64:
      return AscendIE::DataType::INT64;
    case FluidDT::VarType_Type_FP16:
      return AscendIE::DataType::FLOAT16;
    case FluidDT::VarType_Type_BOOL:
      return AscendIE::DataType::BOOL;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "unsupported datatype in TRT op converter, type: %s. "
          "Boolean type is supported as TRT input/output "
          "using TensorRT v8.4+.",
          VarType_Type_Name(type)));
  }
  return AscendIE::DataType::INT32;
}

// The T can be int32 or int64 type.
template <typename T>
AscendIE::Dims Vec2ASCEND_Dims(const std::vector<T>& shape,
                            std::string input,
                            bool with_dynamic_shape = false) {
  PADDLE_ENFORCE_GE(shape.size(),
                    0UL,
                    platform::errors::InvalidArgument(
                        "Ascend's tensor input requires at least 0 "
                        "dimensions, but input %s has %d dims.",
                        input,
                        shape.size()));

  auto ShapeStr = [](const std::vector<T>& shape) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == shape.size() - 1) {
        os << shape[i];
      } else {
        os << shape[i] << ",";
      }
    }
    os << "]";
    return os.str();
  };
  if (!with_dynamic_shape) {
    if (shape.size() == 4UL) {
      if (shape[2] == -1 || shape[3] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      return AscendIE::Dims(3, {shape[1], shape[2], shape[3]});
    } else if (shape.size() == 5UL) {
      if (shape[2] == -1 || shape[3] == -1 || shape[4] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      return AscendIE::Dims(4, {shape[1], shape[2], shape[3], shape[4]});
    } else if (shape.size() == 3UL) {
      if (shape[1] == -1 || shape[2] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      return AscendIE::Dims(2, {shape[1], shape[2]});
    } else if (shape.size() == 2UL) {
      if (shape[1] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      AscendIE::Dims dims(1, {shape[1]})
      return dims;
    }
    // static shape doesn't support 1D op so far.
    PADDLE_ENFORCE_NE(shape.size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The input [%s] shape of trt subgraph is %s."
                          "it's not supported by trt so far",
                          input,
                          ShapeStr(shape)));

    AscendIE::Dims dims(shape.size() - 1);
    for (size_t i = 1; i < shape.size(); i++) {
      dims[i - 1] = shape[i];
    }
    return dims;
  } else {
    if (shape.size() == 4UL) {
      return nvinfer1::Dims4(shape[0], shape[1], shape[2], shape[3]);
    } else if (shape.size() == 3UL) {
      return nvinfer1::Dims3(shape[0], shape[1], shape[2]);
    }
    AscendIE::Dims dims(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
      dims[i] = shape[i];
    }
    return dims;
  }
}
}  // namespace

// class TRTInt8Calibrator;s

/*
 * Ascend Engine.
 *
 * There are two alternative ways to use it, one is to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */
class AscendEngine {
  using DescType = ::paddle::framework::proto::BlockDesc;
  using ShapeMapType = std::map<std::string, std::vector<int>>;
  using PredictorID = int;

 public:
  // Weight is model parameter.
  class Weight {
   public:
    Weight() = default;
    Weight(AscendIE::DataType dtype, void* value, size_t num_elem) {
      w_.type = dtype;
      w_.values = value;
      w_.count = num_elem;
    }
    const AscendIE::WeightsBuf& get() { return w_; }

    void SetDataType(AscendIE::DataType type) { w_.type = type; }

    void SetDataType(phi::DataType type);

    void SetValues(const void* values) { w_.values = values; }

    void SetCount(int64_t num) { w_.count = num; }

    std::vector<int64_t> dims;

   private:
    AscendIE::WeightsBuf w_;
  };

  AscendEngine(int max_batch,
                 int64_t max_workspace,
                 phi::DataType precision = phi::DataType::FLOAT32,
                //  TRTInt8Calibrator* calibrator = nullptr,
                 int device_id = 0,
                 bool with_dynamic_shape = false,
                 const ShapeMapType& min_input_shape = {},
                 const ShapeMapType& max_input_shape = {},
                 const ShapeMapType& optim_input_shape = {},
                 const ShapeMapType& min_shape_tensor = {},
                 const ShapeMapType& max_shape_tensor = {},
                 const ShapeMapType& optim_shape_tensor = {},
                 bool disable_trt_plugin_fp16 = false,
                 phi::DataType model_precision = phi::DataType::FLOAT32)
                // nvinfer1::ILogger& logger = NaiveLogger::Global())
      : max_batch_(max_batch),
        max_workspace_(max_workspace),
        precision_(precision),
        calibrator_(calibrator),
        device_id_(device_id),
        with_dynamic_shape_(with_dynamic_shape),
        min_input_shape_(min_input_shape),
        max_input_shape_(max_input_shape),
        optim_input_shape_(optim_input_shape),
        min_shape_tensor_(min_shape_tensor),
        max_shape_tensor_(max_shape_tensor),
        optim_shape_tensor_(optim_shape_tensor),
        disable_trt_plugin_fp16_(disable_trt_plugin_fp16),
        model_precision_(model_precision) {
        // logger_(logger) {
    // dy::initLibNvInferPlugins(&logger, "");
  }

  ~AscendEngine() {
    for (auto& attr : attrs_) {
      if (attr_dels_.find(attr.first) != attr_dels_.end()) {
        attr_dels_[attr.first]();
      }
    }
    attrs_.clear();
    attr_dels_.clear();
  }

  // Add an input and set its name, data type and dimension.
  AscendIE::ITensor* DeclareInput(const std::string& name,
                                  AscendIE::DataType dtype,
                                  const AscendIE::Dims& dim);
  // Set the offset-th output from a layer as the network's output, and set its
  // name.
  void DeclareOutput(const AscendIE::BaseLayer* layer,
                     int offset,
                     const std::string& name);
  // Set the itensor_map_[name] as the network's output, and set its name.
  void DeclareOutput(const std::string& name);
  // Set the itensor_map_[name] as the network's output, and set its name and
  // data type.
  void DeclareOutput(const std::string& name, AscendIE::DataType dtype);
  void ClearTensorMap() { itensor_map_.clear(); }

  void DeleteITensor(const std::string& name, AscendIE::Tensor* tensor);
  void SetITensor(const std::string& name, AscendIE::Tensor* tensor);
  // Get an ITensor called name.
  AscendIE::Tensor* GetITensor(const std::string& name, bool scalar = false);
  AscendIE::Tensor* ConvertWeight2ITensor(const std::string& name,
                                           bool scalar = false);
  std::unordered_map<std::string, AscendIE::Tensor*>* GetITensorMap();

  AscendIE::CudaEngine* engine() { return infer_engine_.get(); }
  AscendIE::ExecutionContext* context();

  int GetProfileIndex() {
    if (max_profile_num_ > 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      return profile_index_[predictor_id_per_thread];
    } else {
      return 0;
    }
  }

  int GetBindingsOffset() {
    return (binding_num_ / max_profile_num_) * GetProfileIndex();
  }

  int GetNbBindings() { return binding_num_; }

  void ResetContext() {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::InvalidArgument(
            "You should build engine first and then set the context."));
    std::unique_lock<std::mutex> lock(mutex_);
    infer_context_[predictor_id_per_thread].reset(nullptr);
    infer_context_.erase(predictor_id_per_thread);
    cur_profile_num_ = 0;
  }

  void Deserialize(const std::string& engine_serialized_data);

  void SetRuntimeBatch(size_t batch_size);
  int GetRuntimeBatch();

  bool WithFp16() {
    bool enable_fp16 = (precision_ == phi::DataType::FLOAT16);
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    // below is consistent with setFlag in engine.cc
    bool fall_back_fp16 = WithInt8() && !use_dla_;
    return (enable_fp16 || fall_back_fp16) && support_fp16;
  }

  bool WithInt8() {
    bool enable_int8 = (precision_ == phi::DataType::INT8);
    bool support_int8 = infer_builder_->platformHasFastInt8();
    return enable_int8 && support_int8;
  }

  int GetDeviceId() { return device_id_; }

  void SetTensorDynamicRange(AscendIE::Tensor* tensor, float range) {
    quant_dynamic_range_[tensor] = range;
  }

  // Get fp16 trt weight. If src weight is not fp16, we will cast.
  Weight GetFp16TrtWeight(const std::string& name,
                          const phi::DenseTensor& weight_tensor);

  // Get fp32 trt weight. If src weight is not fp32, we will cast.
  Weight GetFp32TrtWeight(const std::string& name,
                          const phi::DenseTensor& weight_tensor);

  // if the src weight type is fp16, then return fp16 trt weight, etc.
  Weight GetTrtWeight(const std::string& name,
                      const phi::DenseTensor& weight_tensor);

  float GetTensorDynamicRange(AscendIE::Tensor* tensor) {
    return quant_dynamic_range_[tensor];
  }

  bool DynamicRangeIsSet(AscendIE::Tensor* tensor) {
    return quant_dynamic_range_.count(tensor);
  }

  // A pointer to CPU memory is needed of the TRT weight.
  // Before TRT runs, fluid loads weight into GPU storage.
  // so we need to copy the weights from GPU to CPU in our op converter.
  // We use a map to store these weights for the weight memory is not released
  // in advance, which affecting the construction of TRT Op.
  std::unordered_map<std::string /*name*/, std::unique_ptr<phi::DenseTensor>>
      weight_map;

  // When setting weight_map, a self-increasing suffix is needed for the names
  // so as to avoid repeatedly setting weights with the same name.
  void SetWeights(std::string w_name,
                  std::unique_ptr<phi::DenseTensor> w_tensor) {
    static int suffix_counter = 0;
    std::string suffix = std::to_string(suffix_counter);
    std::string splitter = "__";
    std::string name_with_suffix = w_name + splitter + suffix;
    PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                      0,
                      platform::errors::AlreadyExists(
                          "The weight named %s is set into the weight map "
                          "twice in TRT OP converter.",
                          name_with_suffix));
    weight_map[name_with_suffix] = std::move(w_tensor);
    suffix_counter += 1;
  }

  void SetUseOSS(bool use_varseqlen) { use_varseqlen_ = use_varseqlen; }
  void SetUseDLA(bool use_dla) { use_dla_ = use_dla; }
  void SetDLACore(int dla_core) { dla_core_ = dla_core; }
  void SetWithErnie(bool with_ernie) { with_ernie_ = with_ernie; }
  void SetWithInterleaved(bool with_interleaved) {
    with_interleaved_ = with_interleaved;
  }
  void SetTransformerPosid(std::string tensorrt_transformer_posid) {
    tensorrt_transformer_posid_ = tensorrt_transformer_posid;
  }
  void SetTransformerMaskid(std::string tensorrt_transformer_maskid) {
    tensorrt_transformer_maskid_ = tensorrt_transformer_maskid;
  }
  void ClearWeights() {
    for (auto& weight_pair : weight_map) {
      weight_pair.second.reset(nullptr);
    }
  }

  // NOTE: The func bellow was modified to adapt the dynamic shape.
  // Initialize the inference network, so that TensorRT layers can add to this
  // network.
  void InitNetwork();
  // After finishing adding ops, freeze this network and creates the execution
  // environment.
  void FreezeNetwork();
  void Execute(int batch_size,
               std::vector<void*>* buffers,
               cudaStream_t stream = nullptr);

  AscendIE::Network* network() { return infer_network_.get(); }

  ShapeMapType& min_input_shape() { return min_input_shape_; }
  ShapeMapType& max_input_shape() { return max_input_shape_; }
  ShapeMapType& optim_input_shape() { return optim_input_shape_; }
  ShapeMapType& min_shape_tensor() { return min_shape_tensor_; }
  ShapeMapType& max_shape_tensor() { return max_shape_tensor_; }
  ShapeMapType& optim_shape_tensor() { return optim_shape_tensor_; }

  bool AdjustDynamicShapeRange(const ShapeMapType& runtime_input_shape,
                               const ShapeMapType& runtime_shape_tensor,
                               std::vector<std::string>* changed,
                               std::vector<std::string>* tensor_changed) {
    bool ret = false;
    changed->clear();
    tensor_changed->clear();
    for (const auto& it : runtime_input_shape) {
      auto name = it.first;
      auto input_shape = it.second;
      bool min_change = false;
      bool max_change = false;
      std::vector<int> bak_min_shape;
      std::vector<int> bak_max_shape;
      if (!min_input_shape_.count(name)) {
        min_input_shape_[name] = input_shape;
        max_input_shape_[name] = input_shape;
        optim_input_shape_[name] = input_shape;
        min_change = true;
        max_change = true;
        ret = true;
      } else {
        PADDLE_ENFORCE_EQ(min_input_shape_[name].size(),
                          input_shape.size(),
                          platform::errors::InvalidArgument(
                              "TRT dynamic_shape min_input_shape %s size not "
                              "equal, the min_input_shape[%s].size()=%d"
                              ", but the runtime_input_shape[%s].size()=%d.",
                              name,
                              name,
                              min_input_shape_[name].size(),
                              name,
                              input_shape.size()));

        bak_min_shape = min_input_shape_[name];
        bak_max_shape = max_input_shape_[name];
        for (size_t d = 0; d < input_shape.size(); ++d) {
          if (input_shape[d] < min_input_shape_[name][d]) {
            ret = true;
            min_change = true;
            min_input_shape_[name][d] = input_shape[d];
          }
          if (input_shape[d] > max_input_shape_[name][d]) {
            ret = true;
            max_change = true;
            max_input_shape_[name][d] = input_shape[d];
          }
        }
      }
      if (min_change)
        LOG(INFO) << "refactor tensor shape range: " << name
                  << ", min_shape from " << Vec2Str(bak_min_shape) << " to "
                  << Vec2Str(min_input_shape_[name]);
      if (max_change)
        LOG(INFO) << "refactor tensor shape range: " << name
                  << ", max_shape from " << Vec2Str(bak_max_shape) << " to "
                  << Vec2Str(max_input_shape_[name]);
      if (min_change || max_change) changed->push_back(name);
    }
    for (const auto& it : runtime_shape_tensor) {
      auto name = it.first;
      auto shape_tensor = it.second;
      bool min_change = false;
      bool max_change = false;
      std::vector<int> bak_min_shape;
      std::vector<int> bak_max_shape;
      if (!min_shape_tensor_.count(name)) {
        min_shape_tensor_[name] = shape_tensor;
        max_shape_tensor_[name] = shape_tensor;
        optim_shape_tensor_[name] = shape_tensor;
        min_change = true;
        max_change = true;
        ret = true;
      } else {
        PADDLE_ENFORCE_EQ(min_shape_tensor_[name].size(),
                          shape_tensor.size(),
                          platform::errors::InvalidArgument(
                              "TRT dynamic_shape min_shape_tensor %s size not "
                              "equal, the min_shape_tensor[%s].size()=%d"
                              ", but the runtime_shape_tensor[%s].size()=%d.",
                              name,
                              name,
                              min_shape_tensor_[name].size(),
                              name,
                              shape_tensor.size()));

        bak_min_shape = min_shape_tensor_[name];
        bak_max_shape = max_shape_tensor_[name];
        for (size_t d = 0; d < shape_tensor.size(); ++d) {
          if (shape_tensor[d] < min_shape_tensor_[name][d]) {
            ret = true;
            min_change = true;
            min_shape_tensor_[name][d] = shape_tensor[d];
          }
          if (shape_tensor[d] > max_shape_tensor_[name][d]) {
            ret = true;
            max_change = true;
            max_shape_tensor_[name][d] = shape_tensor[d];
          }
        }
      }
      if (min_change)
        LOG(INFO) << "refactor shape tensor range: " << name
                  << ", min_shape from " << Vec2Str(bak_min_shape) << " to "
                  << Vec2Str(min_shape_tensor_[name]);
      if (max_change)
        LOG(INFO) << "refactor shape tensor range: " << name
                  << ", max_shape from " << Vec2Str(bak_max_shape) << " to "
                  << Vec2Str(max_shape_tensor_[name]);
      if (min_change || max_change) tensor_changed->push_back(name);
    }
    return ret;
  }

  bool use_varseqlen() { return use_varseqlen_; }
  bool with_ernie() { return with_ernie_; }
  bool with_interleaved() { return with_interleaved_; }
  std::string tensorrt_transformer_posid() {
    return tensorrt_transformer_posid_;
  }
  std::string tensorrt_transformer_maskid() {
    return tensorrt_transformer_maskid_;
  }
  bool disable_trt_plugin_fp16() { return disable_trt_plugin_fp16_; }
  bool with_dynamic_shape() { return with_dynamic_shape_; }
  phi::DataType precision() { return precision_; }

  bool Has(const std::string& attr_name) const {
    return attrs_.count(attr_name) > 0;
  }

  void Erase(const std::string& attr_name) {
    if (!Has(attr_name)) {
      return;
    }
    if (attr_dels_.find(attr_name) != attr_dels_.end()) {
      attr_dels_[attr_name]();
      attr_dels_.erase(attr_name);
    }
    attrs_.erase(attr_name);
  }

  // Set a pointer to the attribute. Engine takes ownership of the attribute.
  template <typename AttrType>
  void Set(const std::string& attr_name, AttrType* attr) {
    if (attrs_.count(attr_name) == 0) {
      PADDLE_ENFORCE_EQ(
          attrs_.count(attr_name),
          0,
          platform::errors::AlreadyExists(
              "Attribute %s already set in trt engine.", attr_name));
    } else {
      VLOG(3) << "Setting the attribute " << attr_name << " for trt engine "
              << this;
    }
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  // Set a pointer to the attribute. Engine doesn't take ownership. Caller
  // should delete the attribute.
  template <typename AttrType>
  void SetNotOwned(const std::string& attr_name, AttrType* attr) {
    PADDLE_ENFORCE_EQ(
        attrs_.count(attr_name),
        0,
        platform::errors::AlreadyExists(
            "Attribute %s already set in trt engine.", attr_name));
    attrs_[attr_name] = attr;
  }

  // Get a reference to the attributed previously set.
  template <typename AttrType>
  AttrType& Get(const std::string& attr_name) const {
    PADDLE_ENFORCE_NE(attrs_.find(attr_name),
                      attrs_.end(),
                      platform::errors::InvalidArgument(
                          "Attribute %s not found in trt engine.", attr_name));
    try {
      return *paddle::any_cast<AttrType*>(attrs_.at(attr_name));
    } catch (paddle::bad_any_cast&) {
      auto TypeToString = [](const std::type_info& info) -> std::string {
        if (std::type_index(info) == std::type_index(typeid(bool*))) {
          return "bool";
        } else if (std::type_index(info) == std::type_index(typeid(int*))) {
          return "int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(const int*))) {
          return "const int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(std::string*))) {
          return "std::string";
        }
        return info.name();
      };

      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid type for attritube %s, expected: %s, actual: %s.",
          attr_name,
          TypeToString(typeid(AttrType*)),
          TypeToString(attrs_.at(attr_name).type())));
    }
  }

  void SetProfileNum(int num) { max_profile_num_ = num; }

  void GetEngineInfo();

  void SetUseInspector(bool use_inspector) { use_inspector_ = use_inspector; }
  void SetScope(const framework::Scope& scope) { scope_ = &scope; }

  void SetContextMemorySharing(bool context_memory_sharing) {
    context_memory_sharing_ = context_memory_sharing;
  }

  void SetLowPrecisionIO(bool low_precision_io) {
    low_precision_io_ = low_precision_io;
  }

  bool EnableLowPrecisionIO() const { return low_precision_io_; }

  void SetAllNodesLowerToTrt(bool all_nodes_offload_to_trt) {
    // all nodes are in trt, so we can use cudaGraph to optimize runtime.
    startup_with_cudagraph_ = all_nodes_offload_to_trt;
  }

 private:
  // Each ICudaEngine object is bound to a specific GPU when it is instantiated,
  // ensure that the thread is associated with the correct device by calling
  // freshDeviceId().
  void freshDeviceId();
  // Used for convert weight into Itensor
  const framework::Scope* scope_;

  // the max batch size
  int max_batch_;
  // the runtime batch size
  static int runtime_batch_;
  // the max memory size the engine uses
  int64_t max_workspace_;

  phi::DataType precision_;
  TRTInt8Calibrator* calibrator_;
  // batch size of the current data, will be updated each Executation.
  int batch_size_{-1};

  // use for engine context memory sharing
  bool context_memory_sharing_{false};

  bool low_precision_io_{false};

  int device_id_;
  int max_profile_num_{1};
  int cur_profile_num_{0};
  std::unordered_map<PredictorID, int> profile_index_;
  bool with_dynamic_shape_{false};
  ShapeMapType min_input_shape_;
  ShapeMapType max_input_shape_;
  ShapeMapType optim_input_shape_;
  ShapeMapType min_shape_tensor_;
  ShapeMapType max_shape_tensor_;
  ShapeMapType optim_shape_tensor_;
  bool disable_trt_plugin_fp16_{false};
  phi::DataType model_precision_{phi::DataType::FLOAT32};
  bool use_varseqlen_{false};
  bool use_dla_{false};
  int dla_core_{0};
  bool with_ernie_{false};
  bool with_interleaved_{false};
  std::string tensorrt_transformer_posid_;
  std::string tensorrt_transformer_maskid_;
  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, AscendIE::Tensor* /*ITensor*/>
      itensor_map_;

  // TensorRT related internal members
  // infer_ptr<nvinfer1::IBuilder> infer_builder_;
  infer_ptr<AscendIE::Network> infer_network_;
  // infer_ptr<nvinfer1::IRuntime> infer_runtime_;
  infer_ptr<AscendIE::Engine> infer_engine_;

  std::unordered_map<nvinfer1::ITensor*, float> quant_dynamic_range_;

  std::unordered_map<std::string, paddle::any> attrs_;
  std::unordered_map<std::string, std::function<void(void)>> attr_dels_;
  std::mutex mutex_;
  bool use_inspector_;

 public:
  thread_local static int predictor_id_per_thread;
};  // class AscendEngine

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle
