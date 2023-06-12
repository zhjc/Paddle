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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/ascendie/engine.h"
// #include "paddle/fluid/inference/tensorrt/helper.h"
// #include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/phi/common/data_type.h"

// namespace paddle {
// namespace inference {
// namespace ascendie {

// /*
//  * Convert Op from Fluid to Ascend Engine.
//  */
// class OpConverter {
//  public:
//   OpConverter() {}

//   // Converter logic for an op.
//   virtual void operator()(const framework::proto::OpDesc& op,
//                           const framework::Scope& scope,
//                           bool test_mode = false) {}

//   // Convert a single fluid operator and add the corresponding layer to ascend.
//   // test_mode: whether the instance executes in an unit test.
//   void ConvertOp(const framework::proto::OpDesc& op,
//                  const std::unordered_set<std::string>& parameters,
//                  const framework::Scope& scope,
//                  AscendEngine* engine,
//                  bool test_mode = false,
//                  const framework::proto::BlockDesc* block = nullptr) {
//     framework::OpDesc op_desc(op, nullptr);

//     OpConverter* it{nullptr};

//     auto converter_type = static_cast<OpConverterType>(
//         PADDLE_GET_CONST(int, op_desc.GetAttr("converter_type")));
//     switch (converter_type) {
//       case OpConverterType::Default:
//         if (op_desc.Type().find("elementwise") != std::string::npos) {
//           static std::unordered_set<std::string> add_tensor_op_set{
//               "add", "mul", "sub", "div", "max", "min", "pow", "mod"};
//           static std::unordered_set<std::string> add_weight_op_set{
//               "add", "mul", "sub", "div", "max", "min", "pow", "mod"};
//           PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
//                             1UL,
//                             platform::errors::InvalidArgument(
//                                 "The input op's Input(\"Y\")."
//                                 "size() should equal to 1, but reveceid "
//                                 "Input(\"Y\").size() = %u.",
//                                 op_desc.Input("Y").size()));
//           int op_type_len = op_desc.Type().size();
//           std::string op_type =
//               op_desc.Type().substr(op_type_len - 3, op_type_len);
//           std::string Y = op_desc.Input("Y")[0];
//           if (parameters.count(Y)) {
//             PADDLE_ENFORCE_GT(
//                 add_weight_op_set.count(op_type),
//                 0,
//                 platform::errors::Unimplemented(
//                     "Unsupported elementwise type %s", op_type.c_str()));
//             it = Registry<OpConverter>::Global().Lookup("elementwise_" +
//                                                         op_type + "_weight");
//             PADDLE_ENFORCE_NOT_NULL(
//                 it,
//                 platform::errors::Unimplemented(
//                     "no OpConverter for optype [%s]", op_desc.Type()));
//           } else {
//             PADDLE_ENFORCE_GT(
//                 add_tensor_op_set.count(op_type),
//                 0,
//                 platform::errors::Unimplemented(
//                     "Unsupported elementwise type %s", op_type.c_str()));
//             it = Registry<OpConverter>::Global().Lookup("elementwise_" +
//                                                         op_type + "_tensor");
//           }
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }

//         if (op_desc.Type() == "depthwise_conv2d") {
//           it = Registry<OpConverter>::Global().Lookup("conv2d");
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }
//         if (op_desc.Type() == "depthwise_conv2d_transpose") {
//           it = Registry<OpConverter>::Global().Lookup("conv2d_transpose");
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }
//         if (op_desc.Type() == "transpose2") {
//           it = Registry<OpConverter>::Global().Lookup("transpose");
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }
//         if (op_desc.Type() == "flatten2") {
//           it = Registry<OpConverter>::Global().Lookup("flatten");
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }
//         // reshape2 == reshape
//         if (op_desc.Type() == "reshape2") {
//           it = Registry<OpConverter>::Global().Lookup("reshape");
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }
//         // lookup_table_v2 == lookup_table
//         if (op_desc.Type() == "lookup_table_v2") {
//           it = Registry<OpConverter>::Global().Lookup("lookup_table");
//           PADDLE_ENFORCE_NOT_NULL(
//               it,
//               platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                               op_desc.Type()));
//         }
//         if (!it) {
//           it = Registry<OpConverter>::Global().Lookup(op_desc.Type());
//         }
//         break;

//       case OpConverterType::GenericPluginCreater:
//         LOG(INFO) << "There is no OpConverter for type " << op_desc.Type()
//                   << ", now use generic_plugin_creater!";
//         it = Registry<OpConverter>::Global().Lookup("generic_plugin_creater");
//         break;

//       case OpConverterType::CustomPluginCreater:
//         LOG(INFO) << "There is no OpConverter for type " << op_desc.Type()
//                   << ", now use custom_plugin_creater!";
//         it = Registry<OpConverter>::Global().Lookup("custom_plugin_creater");
//         break;

//       default:
//         CHECK(false) << "no OpConverter for optype " << op_desc.Type();
//     }

//     PADDLE_ENFORCE_NOT_NULL(
//         it,
//         platform::errors::Unimplemented("no OpConverter for optype [%s]",
//                                         op_desc.Type()));

//     it->SetEngine(engine);
//     engine->SetScope(scope);
//     it->SetBlockDesc(block);
//     (*it)(op, scope, test_mode);

//     size_t output_num = op_desc.OutputNames().size();
//     // only one out settensordynamicRange
//     if (op_desc.HasAttr("out_threshold")) {
//       float out_scale =
//           PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
//       std::string output_name = "";
//       if (op_desc.HasOutput("Output")) {
//         output_name = op_desc.Output("Output").front();
//       } else if (op_desc.HasOutput("Out")) {
//         output_name = op_desc.Output("Out").front();
//       } else if (op_desc.HasOutput("Y")) {
//         output_name = op_desc.Output("Y").front();
//       } else {
//         PADDLE_THROW(
//             platform::errors::NotFound("Op %s has out threshold but doesn't "
//                                        "have an output named \"Output\", "
//                                        "\"Out\" or \"Y\".",
//                                        op_desc.Type()));
//       }
//       auto* output_itensor = engine->GetITensor(output_name);
//       engine->SetTensorDynamicRange(output_itensor, out_scale);
//       VLOG(1) << "Set out scale = " << out_scale << " for tensor "
//               << output_name << ".";
//     }
//     // outs settensordynamicRange
//     for (size_t i = 0; i < output_num; ++i) {
//       if (op_desc.HasAttr("out_" + std::to_string(i) + "_threshold")) {
//         float out_scale = PADDLE_GET_CONST(
//             float, op_desc.GetAttr("out_" + std::to_string(i) + "_threshold"));
//         std::string output_name =
//             op_desc.Output(op_desc.OutputNames()[i]).front();
//         auto* output_itensor = engine->GetITensor(output_name);
//         engine->SetTensorDynamicRange(output_itensor, out_scale);
//         VLOG(1) << "Set out scale = " << out_scale << " for tensor "
//                 << output_name << ".";
//       }
//     }

//     // quant_dequant_linear support for paddle ascend

//     std::vector<std::string> inputs_name = op_desc.InputNames();
//     std::vector<std::string> outputs_name = op_desc.OutputNames();

//     for (size_t i = 0; i < inputs_name.size(); i++) {
//       if (op_desc.HasAttr(inputs_name[i])) {
//         std::string input_tensor_name = op_desc.Input(inputs_name[i])[0];
//         auto* input_itensor = engine->GetITensor(input_tensor_name);
//         float input_scale =
//             PADDLE_GET_CONST(float, op_desc.GetAttr(inputs_name[i]));
//         engine->SetTensorDynamicRange(input_itensor, input_scale);
//         VLOG(1) << "Set input tensor scale = " << input_scale
//                 << " for tensor: " << input_tensor_name << ".";
//       }
//     }
//     for (size_t i = 0; i < outputs_name.size(); i++) {
//       if (op_desc.HasAttr(outputs_name[i])) {
//         std::string output_tensor_name = op_desc.Output(outputs_name[i])[0];
//         auto* output_itensor = engine->GetITensor(output_tensor_name);
//         float output_scale =
//             PADDLE_GET_CONST(float, op_desc.GetAttr(outputs_name[i]));
//         engine->SetTensorDynamicRange(output_itensor, output_scale);
//         VLOG(1) << "Set output tensor scale = " << output_scale
//                 << " for tensor: " << output_tensor_name << ".";
//       }
//     }
//   }

//   // Convert a fluid block to ascend network, NOTE it just convert operators,
//   // the INetwork's inputs and outputs should specified in some other modules.
//   void ConvertBlock(const framework::proto::BlockDesc& block,
//                     const std::unordered_set<std::string>& parameters,
//                     const framework::Scope& scope,
//                     AscendEngine* engine) {
//     std::unique_lock<std::mutex> lk(mut_);
//     for (int i = 0; i < block.ops_size(); i++) {
//       const auto& op = block.ops(i);
//       ConvertOp(op, parameters, scope, engine, false, &block);
//     }
//     // DOPO没有遍历network中layer的接口
//     for (int i = 0; i < engine->network()->getNbLayers(); i++) {
//       auto layer = engine->network()->getLayer(i);
//       if (layer->GetLayerKind() == AscendIE::LayerKind::SHUFFLE) {
//         auto* input_tensor = layer->getInput(0);
//         auto* output_tensor = layer->getOutput(0);
//         auto output_tensor_name = output_tensor->getName();
//         auto input_tensor_name = input_tensor->getName();
//         if (engine->DynamicRangeIsSet(input_tensor) &&
//             !engine->DynamicRangeIsSet(output_tensor)) {
//           float output_scale = engine->GetTensorDynamicRange(input_tensor);
//           VLOG(1) << "Set output tensor scale = " << output_scale
//                   << " for tensor in Ascend: " << output_tensor_name << ".";
//           engine->SetTensorDynamicRange(output_tensor, output_scale);
//         } else {
//           VLOG(1) << "Failed to get input tensor scale for tensor in Ascend: "
//                   << input_tensor_name << ".";
//         }
//       }
//     }
//   }

//   // The scope here should be inited with the parameter vars.
//   void ConvertBlockToAscendEngine(
//       framework::BlockDesc* block_desc,
//       const framework::Scope& scope,
//       const std::vector<std::string>& inputs,
//       const std::unordered_set<std::string>& parameters,
//       const std::vector<std::string>& outputs,
//       AscendEngine* engine) {
//     engine->InitNetwork();
//     for (auto input : inputs) {
//       if (parameters.count(input)) continue;
//       // NOTE(liuyuanle): It is a trick. If you need a name [input], then you
//       // need to use [input.substr(0, idx)].
//       // Maybe we insert suffix of "_cast_auto_mixed.tmp_" in
//       // auto_mixed_precision_pass.
//       auto idx = input.find("_cast_auto_mixed.tmp_");
//       input = input.substr(0, idx);

//       auto* var = block_desc->FindVar(input);
//       PADDLE_ENFORCE_NOT_NULL(
//           var,
//           platform::errors::NotFound("no variable called %s in block.",
//                                      input.c_str()));
//       PADDLE_ENFORCE_EQ(
//           var->GetType(),
//           FluidDT::VarType_Type_LOD_TENSOR,
//           platform::errors::InvalidArgument("Ascend engine only takes "
//                                             "LoDTensor as input"));
//       AscendIE::DataType in_dtype = FluidDataType2Ascend(var->GetDataType());
//       if (engine->precision() == phi::DataType::FLOAT16 &&
//           in_dtype == AscendIE::DataType::FLOAT &&
//           engine->EnableLowPrecisionIO()) {
//         in_dtype = AscendIE::DataType::FLOAT16;
//       }

//       auto var_shape = var->GetShape();
//       if (engine->with_dynamic_shape()) {
//         auto min_input_shape = engine->min_input_shape()[input];
//         auto max_input_shape = engine->max_input_shape()[input];
//         auto optim_input_shape = engine->optim_input_shape()[input];
//         size_t ranks = min_input_shape.size();

//         std::vector<int64_t> input_shape;
//         // input_shape.push_back(-1);
//         for (size_t i = 0; i < ranks; i++) {
//           if (min_input_shape[i] != max_input_shape[i]) {
//             input_shape.push_back(-1);
//           } else {
//             input_shape.push_back(min_input_shape[i]);
//             // the i dimension should be same.
//             PADDLE_ENFORCE_EQ(min_input_shape[i],
//                               optim_input_shape[i],
//                               platform::errors::InvalidArgument(
//                                   "The dim (%d) of the min_input_shape and "
//                                   "optim_input_shape should be same."));
//           }
//         }
//         engine->DeclareInput(
//             input, in_dtype, Vec2ASCEND_Dims(input_shape, input, true));
//       } else {
//         engine->DeclareInput(input, in_dtype, Vec2ASCEND_Dims(var_shape, input));
//       }
//       VLOG(1) << "set ascend engine input dtype " << static_cast<int>(in_dtype);
//     }

//     framework::proto::BlockDesc* block_proto = block_desc->Proto();
//     ConvertBlock(*block_proto, parameters, scope, engine);

//     for (auto& output : outputs) {
//       auto* var = block_desc->FindVar(output);
//       PADDLE_ENFORCE_NOT_NULL(
//           var,
//           platform::errors::NotFound("no variable called %s in block.",
//                                      output.c_str()));
//       PADDLE_ENFORCE_EQ(
//           var->GetType(),
//           FluidDT::VarType_Type_LOD_TENSOR,
//           platform::errors::InvalidArgument(
//               "The output tensor in Ascend subgraph should be LoDTensor"));
//       AscendIE::DataType out_dtype = FluidDataType2Ascend(var->GetDataType());
//       if (engine->precision() == phi::DataType::FLOAT16 &&
//           out_dtype == AscendIE::DataType::FLOAT &&
//           engine->EnableLowPrecisionIO()) {
//         out_dtype = AscendIE::DataType::FLOAT16;
//       }
//       engine->DeclareOutput(output, out_dtype);
//       VLOG(1) << "set ascend engine output dtype " << static_cast<int>(out_dtype);
//     }

//     engine->FreezeNetwork();
//     engine->ClearWeights();
//   }

//   // rank(result) = rank(input)
//   AscendIE::Tensor* Gather(AscendIE::Tensor* input,
//                             const std::vector<int32_t> indices,
//                             int axis = 0) {
//     auto* indices_tensor = Add1DConstantLayer(indices, " ");
//     GatherLayer* gather = engine_->network()->AddGather(input, indices_tensor, axis, AscendIE::GatherOperation::DEFAULT)->getOutput(0);
//     return result;
//   }

//   // paddle allows negative index
//   // for axis length = 5, paddle allows [-5, 4]
//   AscendIE::Tensor* FixNegIndices(AscendIE::Tensor* input_shape,
//                                    AscendIE::Tensor* indices) {
//     int rank = input_shape->getDimensions().nbDims;
//     std::vector<int32_t> zero = std::vector<int32_t>(rank, 0);
//     std::vector<int32_t> minus_one = std::vector<int32_t>(rank, -1);
//     AscendIE::Tensor* zero_tensor = Add1DConstantLayer(zero);
//     AscendIE::Tensor* minus_one_tensor = Add1DConstantLayer(minus_one);
//     // -1, 0
//     auto* sign = Max(Min(indices, zero_tensor), minus_one_tensor);
//     return Sub(indices, Prod(sign, input_shape));
//   }

//   AscendIE::Tensor* Shape(AscendIE::Tensor* input) {
//     return engine_->network()->AddShape(*input)->getOutput(0);
//   }

//   AscendIE::Tensor* Reshape(AscendIE::Tensor* input,
//                              AscendIE::Tensor* newShape,
//                              const std::string& name = "") {
//     ShuffleLayer* shuffle = engine_->AddShuffle(*input);
//     // DOPO没有setInput接口
//     Dims dim = newShape->GetDimensions()
//     shuffle->setReshapeDimensions(dim);
//     if (name != "") {
//       shuffle->setName(name.c_str());
//     }
//     return shuffle->getOutput(0);
//   }

//   AscendIE::Tensor* BroadcastTensor(AscendIE::Tensor* input,
//                                      const int nbDims,
//                                      const std::string& name = "") {
//     auto oldShape = Shape(input);
//     auto oldShapeDims = oldShape->getDimensions();
//     const int rank = oldShapeDims.nbDims;
//     if (rank > nbDims) {
//       PADDLE_THROW(platform::errors::InvalidArgument(
//           "Cannot broadcast a higher rank tensor to a lower rank tensor."));
//     }
//     if (rank < nbDims) {
//       AscendIE::Tensor* concat_shape_tensor;
//       auto* one_rank_tensor =
//           Add1DConstantLayer(std::vector<int32_t>(nbDims - rank, 1));
//       std::vector<nvinfer1::ITensor*> itensors;
//       itensors.push_back(one_rank_tensor);
//       itensors.push_back(oldShape);
//       concat_shape_tensor = Concat(itensors);
//       input = Reshape(input, concat_shape_tensor, name);
//     }
//     return input;
//   }

//   AscendIE::Tensor* BroadcastTensors(AscendIE::Tensor* a,
//                                       AscendIE::Tensor* b,
//                                       const std::string& name = "") {
//     const int aDims = a->getDimensions().size();
//     const int bDims = b->getDimensions().size();
//     if (aDims == bDims) {
//       VLOG(3) << "Broadcast two equal rank tensors";
//     }
//     if (aDims > bDims) {
//       return BroadcastTensor(b, aDims, name);
//     }
//     return BroadcastTensor(a, bDims, name);
//   }

//   // Concat not make rank changed
//   AscendIE::Tensor* Concat(const std::vector<AscendIE::Tensor*>& inputs,
//                             int axis = 0) {
//     ConcatenationLayer* layer = engine_->network()->AddConcatenation(inputs.data(), inputs.size())
//     if (axis != 0) layer->SetAxis(axis);
//     AscendIE::Tensor* c = layer->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Sum(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     // DOPO ADD == SUM?
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::SUM)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Prod(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::PROD)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Min(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::MIN)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Max(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::MAX)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Sub(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::SUB)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Div(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::DIV)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* FloorDiv(AscendIE::Tensor* a, AscendIE::Tensor* b) {
//     AscendIE::Tensor* c = engine_->network()->AddElementWise(a, b, AscendIE::ElementWiseOperation::FLOOR_DIV)->getOutput(0);
//     return c;
//   }

//   AscendIE::Tensor* Act(AscendIE::Tensor* a,
//                          AscendIE::ActivationType act_type) {
//     AscendIE::Tensor* c = engine_->network()->AddActivationLayer(*a, act_type)->getOutput(0);
//     return c;
//   }

//   // Get element tensor of 1D shape tensor
//   AscendIE::Tensor* GetEleTensorOfShape(AscendIE::Tensor* shape_tensor,
//                                          int index,
//                                          bool is_scalar = false) {
//     PADDLE_ENFORCE_GE(
//         index,
//         0,
//         platform::errors::PreconditionNotMet(
//             "The index should be greater or equal than 0, but got %d", index));
//     AscendIE::Tensor* tensor = engine_->network()->AddGather(*shape_tensor,
//                              *Add1DConstantLayer(index, " ", is_scalar),
//                              0,
//                              AscendIE::GatherOperation::DEFAULT)->getOutput(0);
//     return tensor;
//   }
//   template <typename T>
//   // Create and add Multi-D constant float/int32 layer
//   AscendIE::Tensor* AddConstantLayer(const T* data,
//                                       AscendIE::Dims shape,
//                                       const std::string& weight_name = "") {
//     if (!(std::is_same<T, float>::value ||
//           std::is_same<T, platform::float16>::value ||
//           std::is_same<T, int32_t>::value)) {
//       PADDLE_THROW(platform::errors::InvalidArgument(
//           "Unsupported data type (%s) for Ascend AddConstantLayer, only "
//           "supports float, half or int32_t."));
//     }

//     int data_size = std::accumulate(
//         shape.d, shape.d + shape.nbDims, 1, std::multiplies<int>());
//     std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
//     tmp_tensor->Resize({data_size});
//     auto* tmp_data = tmp_tensor->mutable_data<T>(platform::CPUPlace());
//     for (int i = 0; i < data_size; i++) {
//       tmp_data[i] = data[i];
//     }
//     engine_->SetWeights(weight_name, std::move(tmp_tensor));

//     AscendIE::DataType asc_dtype = AscendIE::DataType::FLOAT;
//     if (std::is_integral<T>::value) {
//       asc_dtype = AscendIE::DataType::INT32;
//     }

//     AscendEngine::Weight weight{asc_dtype,
//                                   static_cast<void*>(tmp_data),
//                                   static_cast<size_t>(data_size)};

//     auto const_layer = engine_->network()->AddConstantLayer(shape, weight.get());
//     return const_layer->getOutput(0);
//   }

//   // Create and add 1D constant float/int32 layer
//   template <typename T>
//   AscendIE::Tensor* Add1DConstantLayer(const std::vector<T>& data,
//                                         const std::string& weight_name = "",
//                                         bool scalar = false) {
//     if (!(std::is_same<T, float>::value ||
//           std::is_same<T, platform::float16>::value ||
//           std::is_same<T, int32_t>::value)) {
//       PADDLE_THROW(platform::errors::InvalidArgument(
//           "Unsupported data type (%s) for Ascend AddConstantLayer, only "
//           "supports float, half or int32_t."));
//     }

//     std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
//     int data_size = data.size();
//     tmp_tensor->Resize({data_size});
//     auto* tmp_data = tmp_tensor->mutable_data<T>(platform::CPUPlace());
//     for (int i = 0; i < data_size; i++) {
//       tmp_data[i] = data[i];
//     }
//     engine_->SetWeights(weight_name, std::move(tmp_tensor));

//     nvinfer1::DataType asc_dtype = AscendIE::DataType::FLOAT;
//     if (std::is_integral<T>::value) {
//       asc_dtype = nvinfer1::DataType::kINT32;
//     }

//     AscendEngine::Weight weight{asc_dtype,
//                                   static_cast<void*>(tmp_data),
//                                   static_cast<size_t>(data_size)};
//     AscendIE::Dims input_shape(scalar ? 0 : 1, {data_size});
//     AscendIE::ConstantLayer *constant =
//       network.AddConstantLayer(input_shape, weight.get());
//     return const_layer->getOutput(0);
//   }

//   AscendIE::Tensor* Add1DConstantLayer(AscendIE::Dims data,
//                                         const std::string& weight_name = "",
//                                         bool scalar = false) {
//     std::vector<int> tmp_data;
//     for (int i = 0; i < data.Size(); i++) tmp_data.push_back(data.d[i]);
//     return Add1DConstantLayer(tmp_data, weight_name, scalar);
//   }

//   template <typename T>
//   AscendIE::Tensor* Add1DConstantLayer(T data,
//                                         const std::string& weight_name = "",
//                                         bool scalar = false) {
//     std::vector<T> input_data;
//     input_data.push_back(data);
//     return Add1DConstantLayer(input_data, weight_name, scalar);
//   }

//   void RreplenishLayerAndOutput(
//       AscendIE::BaseLayer* layer,
//       const std::string& layer_type,
//       const std::vector<std::string>& output_tensor_names,
//       bool test_mode = false) {
//     if (layer == nullptr) {
//       return;
//     }
//     size_t num_out = output_tensor_names.size();
//     std::string layer_name = layer_type + " (Output: ";
//     for (size_t i = 0; i < num_out; i++) {
//       layer->getOutput(i)->setName(output_tensor_names[i].c_str());
//       // test_mode怎么处理？
//       engine_->SetITensor(output_tensor_names[i], layer->getOutput(i));
//       if (test_mode) {
//         engine_->DeclareOutput(output_tensor_names[i]);
//       }
//       layer_name += output_tensor_names[i];
//       if (i != num_out - 1) layer_name += ", ";
//     }
//     layer->setName((layer_name + ")").c_str());
//   }
//   void SetEngine(AscendEngine* engine) { engine_ = engine; }

//   void SetBlockDesc(const framework::proto::BlockDesc* block) {
//     block_ = block;
//   }

//   virtual ~OpConverter() {}

//   // Ascend engine
//   AscendEngine* engine_{nullptr};
//   // BlockDesc
//   const framework::proto::BlockDesc* block_{nullptr};

//  protected:
//   bool test_mode_;

//  private:
//   // registered op converter map, whose key is the fluid op type, and value is
//   // the pointer position of corresponding OpConverter class.
//   std::unordered_map<std::string, OpConverter*> converters_;
//   // fluid inference scope
//   framework::Scope* scope_{nullptr};
//   std::mutex mut_;
// };

// }  // namespace ascendie
// }  // namespace inference
// }  // namespace paddle

// #define REGISTER_ASCEND_OP_CONVERTER(op_type__, Converter__)                      \
//   struct asc_##op_type__##_converter : public ::paddle::framework::Registrar { \
//     asc_##op_type__##_converter() {                                            \
//       ::paddle::inference::Registry<                                           \
//           paddle::inference::ascendie::OpConverter>::Global()                  \
//           .Register<::paddle::inference::ascendie::Converter__>(#op_type__);   \
//     }                                                                          \
//   };                                                                           \
//   asc_##op_type__##_converter asc_##op_type__##_converter__;                   \
//   int TouchConverterRegister_##op_type__() {                                   \
//     asc_##op_type__##_converter__.Touch();                                     \
//     return 0;                                                                  \
//   }

// #define USE_ASCEND_CONVERTER(op_type__)                   \
//   extern int TouchConverterRegister_##op_type__();     \
//   static int use_op_converter_asc_##op_type__ UNUSED = \
//       TouchConverterRegister_##op_type__();
