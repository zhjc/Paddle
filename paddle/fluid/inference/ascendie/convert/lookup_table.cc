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

class LookupTableOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3)
        << "convert lookup_table(lookup_table_v2) op to TensorRT IGatherLayer";
    
	  framework::OpDesc op_desc(op, nullptr);
	
    auto ids_name = op_desc.Input("Ids").front();
    auto w_name = op_desc.Input("W").front();
    auto out_name = op_desc.Output("Out").front();

    AscendIE::Tensor* ids_tensor = engine_->GetITensor(ids_name);
    AscendIE::Tensor* w_tensor = engine_->GetITensor(w_name);

    AscendIE::GatherLayer* layer = engine_->network()->AddGather(w_tensor, ids_tensor, 0, AscendIE::GatherOperation::DEFAULT);
	
    RreplenishLayerAndOutput(layer, "gather", {out_name}, test_mode);
  }
};
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(lookup_table, LookupTableOpConverter);
