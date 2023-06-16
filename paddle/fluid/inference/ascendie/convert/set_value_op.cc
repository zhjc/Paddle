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

#define GET_ATTR_FROM_VECTOR(attr_name__)                                   \
  do {                                                                      \
    std::vector<int64_t> vec_##attr_name__;                                 \
    if (op_desc.HasAttr(#attr_name__)) {                                    \
      vec_##attr_name__ = PADDLE_GET_CONST(std::vector<int64_t>,            \
                                           op_desc.GetAttr(#attr_name__));  \
      if (vec_##attr_name__.size() > 0) attr_name__ = vec_##attr_name__[0]; \
    }                                                                       \
  } while (0)

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace ascendie {
// we use tensorrt ScatterElement to generate set value
// For example, if indices has dimensions [N,C,H,W] and axis is 2, then the
// updates happen as: for n in [0,n)
//     for c in [0,n)
//         for h in [0,n)
//             for w in [0,n)
//                 output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]]

class SetValueConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a set value op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);

    auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* updates = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);

    int64_t axes = 0;
    int64_t starts = 0;
    int64_t steps = 1;
    int64_t ends = 0;

    GET_ATTR_FROM_VECTOR(axes);
    GET_ATTR_FROM_VECTOR(starts);
    GET_ATTR_FROM_VECTOR(steps);
    GET_ATTR_FROM_VECTOR(ends);

    // calculate dims
    auto input_dims = inputs->GetDimensions();
    auto update_dims = updates->GetDimensions();

    // check params and refill
    if (axes == -1) {
      axes = input_dims.Size(); - 1;
    }

    if (ends == -1 || ends > input_dims[axes]) {
      ends = input_dims[axes];
    }

    if (axes >= input_dims.Size()) {
      platform::errors::InvalidArgument(
          "The axes %d is larger than total axes %d", axes, input_dims.Size());
    }
    if (starts >= input_dims[axes]) {
      platform::errors::InvalidArgument(
          "The start %d of dim %d is larger than origin shape %d",
          starts,
          axes,
          input_dims[axes]);
    }
    if (update_dims[axes] != (input_dims[axes] - starts) / steps) {
      platform::errors::InvalidArgument("The update dim error, should be %d",
                                        (input_dims[axes] - starts) / steps);
    }
    if (engine_->with_dynamic_shape()) {
      // generate indice
      int post_size = 1;
      for (int j = axes + 1; j < update_dims.Size(); ++j) {
        post_size = post_size * update_dims[j];
      }
      std::vector<int> axes_index;
      for (int i = starts; i < ends; i += steps) {
        for (int j = 0; j < post_size; ++j) {
          axes_index.emplace_back(i);
        }
      }
      int pre_size = 1;
      for (int i = 0; i < axes; ++i) {
        pre_size *= update_dims[i];
      }
      std::vector<int> indices;
      for (int i = 0; i < pre_size; ++i) {
        indices.insert(indices.end(), axes_index.begin(), axes_index.end());
      }

      AscendIE::Dims indice_dims = update_dims;

      // create a tensor to store data
      std::vector<int> indice_dim_vec;
      for (int i = 0; i < update_dims.Size(); i++) {
        indice_dim_vec.emplace_back(update_dims[i]);
      }
      auto indice_tensor_dims = phi::make_ddim(indice_dim_vec);
      std::unique_ptr<phi::DenseTensor> indice_tensor(
          std::make_unique<phi::DenseTensor>());
      indice_tensor->Resize(indice_tensor_dims);

      auto* dev_ctx = static_cast<phi::CPUContext*>(
          platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));
      auto* weight_data = dev_ctx->template HostAlloc<int>(indice_tensor.get());

      memcpy(weight_data, indices.data(), sizeof(int) * indice_tensor->numel());

      AscendIEEngine::Weight weight{
          AscendIE::DataType::INT32,
          static_cast<void*>(weight_data),
          static_cast<size_t>(indice_tensor->numel())};
      auto output_name = op_desc.Output("Out")[0];
      engine_->SetWeights("set_value_index_" + output_name,
                          std::move(indice_tensor));

      auto const_layer = engine_->network()->AddConstantLayer(indice_dims, weight.get());

      // auto* layer = engine_->network()->AddScatter(input, 
      //                                    const_layer->GetOutput(0),
      //                                    updates,
      //                                    AscendIE::ScatterMode::kELEMENT);

      // layer->SetAxis(axes);

      // RreplenishLayerAndOutput(layer, "set_value", {output_name}, test_mode);
      RreplenishLayerAndOutput(const_layer, "set_value", {output_name}, test_mode);
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "static shape mode not supported in set value yet"));
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(set_value, SetValueConverter);