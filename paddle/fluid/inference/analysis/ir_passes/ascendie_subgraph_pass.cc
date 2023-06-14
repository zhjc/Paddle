
// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/analysis/ir_passes/ascendie_subgraph_pass.h"
#include <fcntl.h>
#include <cstddef>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"
#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/ascendie/convert/op_converter.h"
#include "paddle/fluid/inference/ascendie/engine.h"
#include "paddle/fluid/inference/ascendie/op_teller.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace {

// if in mixed model precision, we should make all ascendie_engine's output
// floats dtype to float32 dtype.
void OutputProcess(framework::ir::Graph *graph,
                   const std::unordered_set<framework::ir::Node *> &aie_outputs,
                   phi::Backend backend,
                   phi::DataType precision,
                   const std::unordered_set<std::string> &blacklist) {
  framework::BlockDesc *block_desc{nullptr};
  int suffix = 0;
  std::unordered_map<framework::ir::Node *, framework::ir::Node *>
      var_to_cast_op_map;

  framework::proto::VarType::Type to_type;
  if (precision == phi::DataType::FLOAT16) {
    to_type = framework::proto::VarType::FP16;
  } else if (precision == phi::DataType::BFLOAT16) {
    to_type = framework::proto::VarType::BF16;
  } else if (precision == phi::DataType::FLOAT32) {
    return;
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only support "
        "fp16 and bf16.",
        static_cast<int>(precision)));
  }

  for (auto *op_node : framework::ir::TopologySortOperations(*graph)) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed") block_desc = op_node->Op()->Block();
    if (op_type != "ascendie_engine") continue;
    for (auto *var_node : op_node->outputs) {
      if (!aie_outputs.count(var_node)) continue;
      if (!var_node->Var()->Persistable() &&
          IsFloatVar(var_node->Var()->GetDataType()) &&
          var_node->Var()->GetDataType() != framework::proto::VarType::FP32) {
        for (auto *next_op : var_node->outputs) {
          // if next_op support mixed_precision, we need to add cast op.
          if (OpSupportPrecision(
                  phi::TransToPhiKernelName(next_op->Op()->Type()),
                  backend,
                  precision,
                  blacklist)) {
            InsertCastOp(graph,
                         var_node,
                         next_op,
                         framework::proto::VarType::FP32,
                         to_type,
                         block_desc,
                         &suffix,
                         &var_to_cast_op_map);
            var_node->Var()->SetDataType(framework::proto::VarType::FP32);
          }
        }
      }
    }
  }
}

// Determine whether the whole graph offload to ascendie. If so we can try to
// enable optimization such as npuGraph.
bool AllNodesLowerToAiePostProcess(framework::ir::Graph *graph) {
  std::unordered_set<std::string> aie_nodes_set{
      "feed", "fetch", "ascendie_engine"};
  bool all_nodes_offload_to_aie = true;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp()) {
      if (!aie_nodes_set.count(node->Op()->Type())) {
        all_nodes_offload_to_aie = false;
        break;
      }
    }
  }
  return all_nodes_offload_to_aie;
}
}  // namespace

using framework::ir::Node;

void analysis::AscendIESubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("ascendie_subgraph_pass", graph);

  static std::once_flag aie_plugin_registered;
  std::call_once(aie_plugin_registered, []() {
    ascendie::plugin::AiePluginRegistry::Global()->RegistToAie();
  });

  auto model_precision =
      static_cast<phi::DataType>(Get<int>("model_precision"));
  if (model_precision == phi::DataType::BFLOAT16) {
    LOG(WARNING)
        << "Paddle-AIE not support bf16 mixed precison, just fallback.";
    return;
  }

  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  bool use_npu_graph = Get<bool>("use_npu_graph");
  bool no_calib_int8 = enable_int8 && !(use_calib_mode);
  auto aie_disabled_ops = Get<std::vector<std::string>>("aie_disabled_ops");
  auto with_dynamic_shape = Get<bool>("with_dynamic_shape");
  auto teller = [&](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    if (find(aie_disabled_ops.begin(),
             aie_disabled_ops.end(),
             node->Op()->Type()) != aie_disabled_ops.end()) {
      VLOG(3) << node->Op()->Type().c_str()
              << " is diabled by config in AscendIE";
      return false;
    }
    for (const auto &out_var : node->Op()->OutputNames()) {
      for (const auto &var_name : node->Op()->Output(out_var)) {
        if (find(aie_disabled_ops.begin(), aie_disabled_ops.end(), var_name) !=
            aie_disabled_ops.end()) {
          VLOG(3) << node->Op()->Type().c_str()
                  << " is diabled by config in AscendIE";
          return false;
        }
      }
    }
    bool is_ok = ascendie::OpTeller::Global().Tell(
        node, no_calib_int8, with_dynamic_shape);
    if (!is_ok)
      VLOG(3) << node->Op()->Type().c_str() << " op is not in AscendIE";
    return is_ok;
  };

  framework::ir::SubGraphFuser fuser(
      graph,
      teller,
      Get<int>("min_subgraph_size") /*min subgraph size*/,
      "ascendie_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in aie, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;
  std::vector<std::string> engine_names;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !framework::ir::Agent(node).subgraph()->empty()) {
      engine_names.push_back(CreateAscendIEOp(
          node, graph, graph_param_names, &repetitive_params, use_npu_graph));
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && framework::ir::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));

  bool all_nodes_offload_to_aie = AllNodesLowerToAiePostProcess(graph);
  if (all_nodes_offload_to_aie) {
    LOG(INFO) << "The entire graph is offloaded to AscendIE.";
  }
  if (use_npu_graph && !all_nodes_offload_to_aie) {
    LOG_FIRST_N(WARNING, 1)
        << "You have enabled NpuGraph, but not the entire graph offload to "
           "aie, now return to normal mode.";
    use_npu_graph = false;
  }
  if (use_npu_graph && all_nodes_offload_to_aie) {
    for (auto &name : engine_names) {
      PADDLE_ENFORCE_EQ(
          paddle::inference::Singleton<
              inference::ascendie::AIEEngineManager>::Global()
              .Has(name),
          true,
          platform::errors::PreconditionNotMet(
              "AIEEnegineManager shoud has engine %s, but not found.", name));
      paddle::inference::Singleton<
          inference::ascendie::AIEEngineManager>::Global()
          .Get(name)
          ->SetAllNodesLowerToAie(use_npu_graph);
    }
  }

  // some ops are only implemented in paddle-aie,
  // but not in paddle ,we should revert it.
  for (auto *op_node : framework::ir::TopologyVarientSort(
           *graph, static_cast<framework::ir::SortKind>(0))) {
    if (op_node->Op()->Type() == "matrix_multiply") {
      auto origin_type =
          op_node->Op()->GetAttrIfExists<std::string>("original_type");
      LOG(WARNING) << "matrix_multiply can't enter into paddle-aie,"
                   << "we will revert to " << origin_type;
      op_node->Op()->SetType(origin_type);
      op_node->RenameOp(origin_type);
    }
  }
}

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &predictor_id,
                              const std::string &max_batch_size,
                              const std::string &precision,
                              bool use_npu_graph,
                              const bool for_calibration) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
    engine_hash_key += "#";
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
    engine_hash_key += "#";
  }
  engine_hash_key += predictor_id;
  if (!for_calibration) {
    engine_hash_key += "#";
    engine_hash_key += max_batch_size;
  }
  engine_hash_key += "#";
  engine_hash_key += precision;

  engine_hash_key += "#";
  engine_hash_key += use_npu_graph;

  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  VLOG(2) << "AIE engine hash key: " << engine_hash_key;
  VLOG(2) << "AIE engine key: " << engine_key;
  return engine_key;
}

std::string AscendIESubgraphPass::CreateAscendIEOp(
    framework::ir::Node *node,
    framework::ir::Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params,
    bool use_npu_graph) const {
  auto *op_desc = node->Op();
  auto &subgraph = *framework::ir::Agent(node).subgraph();
  PADDLE_ENFORCE_EQ(subgraph.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "The subgraph should not be empty."));

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  // Add new block for AscendIEEngineOP
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  // const framework::BlockDesc& main_block = program_desc->Block(0);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  // A fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  LOG(INFO) << "---  detect a sub-graph with " << subgraph.size() << " nodes";
  for (auto node : subgraph) {
    if (node->NodeType() == Node::Type::kOperation) {
      VLOG(5) << "aie subgraph has op: " << (node->Op()->Type());
    }
  }

  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the engine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;
  // if we delete fluid copy of params shared by more than 1 ops, there will be
  // problem, so we filter them out.
  std::vector<std::string> params_not_shared;

  auto *scope = param_scope();
  // The node->inputs contains input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0 &&
        x->outputs.size() <= 1) {
      params_not_shared.push_back(x->Name());
    }
    // When AIE Engine's input is INT64, we need do some extra work.
    // So we reserved a name for later use when casting INT64 -> INT32.
    // We must check whether scope has had the same name var!
    if (x->Var()->GetDataType() == framework::proto::VarType::INT64) {
      std::string tmp_name = x->Name() + "_cast_to_INT32";
      LOG(WARNING)
          << "ascendie_subgraph's input named " << x->Name()
          << " having int64 dtype in pdmodel description, we will cast them to "
             "int32 dtype to feed them into paddle-aie.";
      /*
            PADDLE_ENFORCE_EQ(scope->FindVar(tmp_name),
                              nullptr,
                              platform::errors::InvalidArgument(
                                  "The  var name %s has exists in scope.",
         tmp_name));
      */
      scope->Var(tmp_name);
    }
  }

  // var may have the same name but not have the same id.
  // e.g., var(batch_norm2d_0.w_1) may have id: 10, 13, 25.... in a graph.
  // so we must find all the var_name+id.
  // https://github.com/PaddlePaddle/Paddle/pull/53184
  for (auto *n : graph->Nodes()) {
    if (n->IsVar() && input_names.count(n->Name())) {
      input_names_with_id.insert(n->Name() + std::to_string(n->id()));
    }
  }

  auto model_precision =
      static_cast<phi::DataType>(Get<int>("model_precision"));
  auto mixed_black_list =
      Get<std::unordered_set<std::string>>("mixed_black_list");

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  std::map<std::string, int> origin_name_output_rank;
  std::unordered_set<Node *> aie_outputs;
  // record the origin output data type
  std::vector<int> origin_outputs_dtype;
  std::map<std::string, int> map_origin_outputs_dtype;
  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
    origin_name_output_rank[x->Name()] = x->Var()->GetShape().size();
    aie_outputs.insert(x);
    map_origin_outputs_dtype[x->Name()] =
        static_cast<int>(x->Var()->GetDataType());
  }

  OutputProcess(
      graph, aie_outputs, phi::Backend::NPU, model_precision, mixed_black_list);

  std::unordered_map<std::string, std::string> output_name_map;
  std::unordered_map<std::string, framework::ir::Node *> graph_var_map;

  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      graph_var_map[node->Name()] = node;
    }
  }
  auto precision_mode = Get<int>("aie_precision_mode");
  bool enable_fp16 = false;
  if (precision_mode == static_cast<int>(phi::DataType::FLOAT16))
    enable_fp16 = true;
  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  auto &subgraph_nodes = *framework::ir::Agent(node).subgraph();
  auto min_input_shape =
      Get<std::map<std::string, std::vector<int>>>("min_input_shape");
  auto max_input_shape =
      Get<std::map<std::string, std::vector<int>>>("max_input_shape");
  auto opt_input_shape =
      Get<std::map<std::string, std::vector<int>>>("optim_input_shape");

  auto min_shape_tensor =
      Get<std::map<std::string, std::vector<int>>>("min_shape_tensor");
  auto max_shape_tensor =
      Get<std::map<std::string, std::vector<int>>>("max_shape_tensor");
  auto opt_shape_tensor =
      Get<std::map<std::string, std::vector<int>>>("optim_shape_tensor");

  auto allow_build_at_runtime = Get<bool>("aie_allow_build_at_runtime");
  auto with_dynamic_shape = Get<bool>("with_dynamic_shape");
  auto shape_range_info_path = Get<std::string>("aie_shape_range_info_path");
  auto aie_tuned_dynamic_shape = Get<bool>("aie_tuned_dynamic_shape");
  int max_batch_size = Get<int>("max_batch_size");
  if (aie_tuned_dynamic_shape) {
    if (!shape_range_info_path.empty()) {
      VLOG(1) << "aie dynamic_shape deserialize from " << shape_range_info_path;
      inference::DeserializeShapeRangeInfo(shape_range_info_path,
                                           &min_input_shape,
                                           &max_input_shape,
                                           &opt_input_shape,
                                           &min_shape_tensor,
                                           &max_shape_tensor,
                                           &opt_shape_tensor);
    } else {
      shape_range_info_path =
          Get<std::string>("model_opt_cache_dir") + "shape_range_info.pbtxt";
      if (open(shape_range_info_path.c_str(), O_RDONLY) != -1) {
        VLOG(1) << "aie dynamic_shape deserialize from "
                << shape_range_info_path;
        inference::DeserializeShapeRangeInfo(shape_range_info_path,
                                             &min_input_shape,
                                             &max_input_shape,
                                             &opt_input_shape,
                                             &min_shape_tensor,
                                             &max_shape_tensor,
                                             &opt_shape_tensor);
      } else {
        int fd = open(shape_range_info_path.c_str(), O_WRONLY | O_CREAT, 0644);
        close(fd);
      }
    }
  }

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to ascendie OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the corresponding ITensor (aie data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.
  RenameAndGetOutputs(subgraph_nodes,
                      &block_desc,
                      input_names_with_id,
                      &output_names_with_id,
                      &output_names,
                      &output_name_map,
                      graph_var_map,
                      !enable_int8);

  // When ascendie engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  std::vector<int> renamed_output_rank;
  for (auto name : output_names) {
    PADDLE_ENFORCE_NE(output_name_map.count(name),
                      0,
                      platform::errors::PreconditionNotMet(
                          "The output_name_map should have %s", name));
    output_mapping.push_back(output_name_map[name]);
    renamed_output_rank.push_back(origin_name_output_rank[name]);
    origin_outputs_dtype.push_back(map_origin_outputs_dtype[name]);

    // When AIE Engine's output is INT64, we need do some extra work.
    // So we reserved a name for later use when casting INT32 -> INT64.
    // We must check whether scope has had the same name var!
    if (static_cast<framework::proto::VarType_Type>(
            map_origin_outputs_dtype[name]) ==
        framework::proto::VarType::INT64) {
      std::string tmp_name = name + "_cast_to_INT64";
      LOG(WARNING) << "ascendie_subgraph's output named " << name
                   << " having int64 dtype in pdmodel description, but in fact "
                      "it is int32 "
                      "dtype after executing this ascendie_subgraph, so we "
                      "need cast them into int64.";
      PADDLE_ENFORCE_EQ(scope->FindVar(tmp_name),
                        nullptr,
                        platform::errors::InvalidArgument(
                            "The  var name %s has exists in scope.", tmp_name));
      scope->Var(tmp_name);
    }
  }
  PADDLE_ENFORCE_EQ(output_mapping.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "The output_mapping should not be empty."));
  PADDLE_ENFORCE_EQ(
      !block_desc.Proto()->vars().empty(),
      true,
      platform::errors::PreconditionNotMet("the block has no var-desc"));

  // Set attrs
  op_desc->SetType("ascendie_engine");
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));

  op_desc->SetBlockAttr("sub_block", new_block);
  op_desc->SetAttr("subgraph", block_desc.Proto()->SerializeAsString());
  op_desc->SetAttr("origin_outputs_dtype", origin_outputs_dtype);
  op_desc->SetAttr("max_batch_size", max_batch_size);
  op_desc->SetAttr("workspace_size", Get<int64_t>("workspace_size"));
  op_desc->SetAttr("npu_id", Get<int>("npu_device_id"));
  op_desc->SetAttr("output_name_mapping", output_mapping);
  op_desc->SetAttr("origin_output_rank", renamed_output_rank);
  op_desc->SetAttr("parameters", params);
  op_desc->SetAttr("allow_build_at_runtime", allow_build_at_runtime);
  op_desc->SetAttr("shape_range_info_path", shape_range_info_path);
  op_desc->SetAttr("use_inspector", Get<bool>("use_inspector"));
  op_desc->SetAttr("model_precision", Get<int>("model_precision"));
  op_desc->SetAttr("with_dynamic_shape", with_dynamic_shape);
  op_desc->SetAttr("enable_low_precision_io",
                   Get<bool>("enable_low_precision_io"));

  // we record all inputs' shapes in attr to check if they are consistent
  // with the real inputs' shapes retrieved from scope when aie runs.
  for (auto *x : node->inputs) {
    if (x->IsVar() && x->Var()) {
      framework::VarDesc *var = x->Var();
      op_desc->SetAttr(var->Name() + "_shape", var->GetShape());
    }
  }

  auto use_static_engine = Get<bool>("use_static_engine");
  op_desc->SetAttr("use_static_engine", use_static_engine);
  if (use_static_engine)
    op_desc->SetAttr("model_opt_cache_dir",
                     Get<std::string>("model_opt_cache_dir"));

  // TODO(NHZlX)
  // There are models with the same structure but the different parameters,
  // when running in the 'use_serialize' mode, there is a bug.
  // serialization is affected by max_batch_size, but calibration is not.
  // So we use separate engine keys in serialization and calibration.
  auto engine_key =
      GenerateEngineKey(input_names_with_id,
                        output_names_with_id,
                        std::to_string(0),
                        std::to_string(max_batch_size),
                        std::to_string(static_cast<int>(precision_mode)),
                        use_npu_graph,
                        false);
  auto calibration_engine_key =
      GenerateEngineKey(input_names_with_id,
                        output_names_with_id,
                        std::to_string(0),
                        std::to_string(max_batch_size),
                        std::to_string(static_cast<int>(precision_mode)),
                        use_npu_graph,
                        true);
  auto predictor_id = Get<int>("predictor_id");

  // Get "" when there is no cached calibration table data.
  std::string calibration_data = "";
  // if (enable_int8 && use_calib_mode) {
  //   calibration_data =
  //       GetAieCalibTableData(Get<std::string>("model_opt_cache_dir"),
  //                            calibration_engine_key,
  //                            enable_int8);
  // }
  op_desc->SetAttr("calibration_data", calibration_data);
  op_desc->SetAttr("enable_int8", enable_int8);
  op_desc->SetAttr("enable_fp16", enable_fp16);
  op_desc->SetAttr("use_calib_mode", use_calib_mode);
  op_desc->SetAttr("engine_key", engine_key);
  op_desc->SetAttr("calibration_engine_key", calibration_engine_key);
  op_desc->SetAttr("predictor_id", predictor_id);

  std::string aie_engine_serialized_data = "";
  op_desc->SetAttr("engine_serialized_data", aie_engine_serialized_data);
  op_desc->Flush();

  // std::unique_ptr<ascendie::AIEInt8Calibrator> calibrator;
  // if (enable_int8 && calibration_data.size() != 0) {
  //   calibrator.reset(new ascendie::AIEInt8Calibrator(calibration_data));
  //   LOG(INFO) << "RUN Paddle AIE int8 calibration mode...";
  // }
  // When in int8 mode and calibration_mode, the program just produce the
  // calibration table data.
  bool calibration_mode =
      (enable_int8 && calibration_data.size() == 0 && use_calib_mode);
  if (calibration_mode) {
    // calibraion mode means generate int8 calibration table data process.
    return calibration_engine_key;
  }

  std::copy(params_not_shared.begin(),
            params_not_shared.end(),
            std::back_inserter(*repetitive_params));

  // Check aie version for dynamic shape input.

  if (min_input_shape.size() > 0 && AIE_VERSION < 6000) {
    LOG_FIRST_N(WARNING, 1) << "You are using the dynamic size input mode of "
                               "Paddle-AIE, but we found that the version of "
                               "the AscendIE is less than 6.0, so we use the "
                               "static shape mode instead.";
    min_input_shape = {};
    max_input_shape = {};
    opt_input_shape = {};
  }

  const float aie_compile_version = ascendie::AieMajorVersion(AIE_VERSION);
  const float aie_runtime_version =
      ascendie::AieMajorVersion(ascendie::GetInferLibVersion());
  if (aie_compile_version != aie_runtime_version) {
    LOG_FIRST_N(WARNING, 1)
        << "The Paddle Inference library is compiled with "
        << aie_compile_version << " version AscendIE, "
        << "but the runtime AscendIE you are using is " << aie_runtime_version
        << " version. "
           "This might cause serious compatibility issues. We strongly "
           "recommend using the same AIE version at runtime.";
  }

  std::unordered_set<const Node *> nodes2remove(
      framework::ir::Agent(node).subgraph()->begin(),
      framework::ir::Agent(node).subgraph()->end());
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);

  // Setting the disable_aie_plugin_fp16 to true means that AIE plugin will not
  // run fp16.
  // When running fp16, the output accuracy of the model will be affected,
  // closing the plugin fp16 may bring some improvement on accuracy.
  bool disable_aie_plugin_fp16 = Get<bool>("disable_aie_plugin_fp16");
  ascendie::AscendIEEngine *aie_engine =
      inference::Singleton<inference::ascendie::AIEEngineManager>::Global()
          .Create(engine_key + std::to_string(predictor_id),
                  max_batch_size,
                  Get<int64_t>("workspace_size"),
                  static_cast<phi::DataType>(precision_mode),
                  // calibrator.get(),
                  Get<int>("npu_device_id"),
                  with_dynamic_shape,
                  min_input_shape,
                  max_input_shape,
                  opt_input_shape,
                  min_shape_tensor,
                  max_shape_tensor,
                  opt_shape_tensor,
                  disable_aie_plugin_fp16,
                  static_cast<phi::DataType>(Get<int>("model_precision")));
  aie_engine->SetUseOSS(Get<bool>("use_varseqlen"));
  aie_engine->SetWithInterleaved(Get<bool>("with_interleaved"));
  aie_engine->SetTransformerPosid(
      Get<std::string>("ascendie_transformer_posid"));
  aie_engine->SetTransformerMaskid(
      Get<std::string>("ascendie_transformer_maskid"));
  aie_engine->SetUseDLA(Get<bool>("aie_use_dla"));
  aie_engine->SetDLACore(Get<int>("aie_dla_core"));
  aie_engine->SetUseInspector(Get<bool>("use_inspector"));
  aie_engine->SetWithErnie(
      graph->Has(framework::ir::kEmbEltwiseLayernormPass) &&
      graph->Has(framework::ir::kMultiheadMatmulPass));
  aie_engine->SetContextMemorySharing(Get<bool>("context_memory_sharing"));
  aie_engine->SetLowPrecisionIO(Get<bool>("enable_low_precision_io"));

  if (use_static_engine) {
    aie_engine_serialized_data = GetAieEngineSerializedData(
        Get<std::string>("model_opt_cache_dir"), engine_key);
    // we can load the engine info serialized before from the disk.
    if (!aie_engine_serialized_data.empty()) {
      try {
        aie_engine->Deserialize(aie_engine_serialized_data);
        LOG(INFO) << "Load AIE Optimized Info from "
                  << GetAieEngineSerializedPath(
                         Get<std::string>("model_opt_cache_dir"), engine_key);
        return engine_key + std::to_string(predictor_id);
      } catch (const std::exception &exp) {
        LOG(WARNING)
            << "Fail to load AIE Optimized Info from "
            << GetAieEngineSerializedPath(
                   Get<std::string>("model_opt_cache_dir"), engine_key)
            << ". Engine deserialization failed: Serialized Engine Version "
               "does not match Current Version, AIE engine will be rebuilded";
      }
    }
  }

  // If with_dynamic_shape is configured, but min_input_shape is empty,
  // create aie engine in runtime instead of in pass.
  if (with_dynamic_shape && min_input_shape.empty()) {
    return engine_key + std::to_string(predictor_id);
  }

  // the following code will NOT run in following situation:
  // 1. calibraion mode (generate aie int8 calibraiton table data)
  // 2. already load serialized aie engine info.
  LOG(INFO) << "Prepare AIE engine (Optimize model structure, Select OP "
               "kernel etc). This process may cost a lot of time.";

  framework::BlockDesc block_desc_temp(nullptr, block_desc.Proto());
  std::unordered_set<std::string> param_set(params.begin(), params.end());
  inference::Singleton<inference::ascendie::OpConverter>::Global()
      .ConvertBlockToAIEEngine(
          &block_desc_temp,
          *scope,
          std::vector<std::string>(input_names.begin(), input_names.end()),
          param_set,
          output_mapping,
          aie_engine);

  if (use_static_engine) {
    nvinfer1::IHostMemory *serialized_engine_data = aie_engine->Serialize();
    aie_engine_serialized_data =
        std::string((const char *)serialized_engine_data->data(),
                    serialized_engine_data->size());
    SaveAieEngineSerializedDataToFile(
        GetAieEngineSerializedPath(Get<std::string>("model_opt_cache_dir"),
                                   engine_key),
        aie_engine_serialized_data);
    LOG(INFO) << "Save AIE Optimized Info to "
              << GetAieEngineSerializedPath(
                     Get<std::string>("model_opt_cache_dir"), engine_key);
  }

  return engine_key + std::to_string(predictor_id);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(ascendie_subgraph_pass,
              paddle::inference::analysis::AscendIESubgraphPass)
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");

REGISTER_PASS_CAPABILITY(ascendie_subgraph_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("pool2d", 0)
            .EQ("relu", 0)
            .EQ("softmax", 0)
            .EQ("sigmoid", 0)
            .EQ("hard_swish", 0)
            .LE("depthwise_conv2d", 1)
            .EQ("batch_norm", 0)
            .EQ("concat", 0)
            .EQ("tanh", 0)
            .EQ("pad", 0)
            .LE("elementwise_add", 1)
            .LE("elementwise_mul", 1)
            .EQ("prelu", 0)
            .LE("conv2d_transpose", 2)
            .LE("leaky_relu", 1)
            .EQ("fc", 0)
            .EQ("shuffle_channel", 0)
            .EQ("swish", 0)
            .EQ("silu", 0)
            .EQ("split", 0)
            .LE("instance_norm", 1)
            .EQ("gelu", 0)
            .EQ("layer_norm", 0)
            .EQ("scale", 0)
            .LE("matmul", 1));
