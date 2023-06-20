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
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace ascendie {

/*
 * After aie_map_ops_to_matrix_multiply_pass(mul, matmul, matmul_v2 ->
 * matrix_multiply), use MatrixMultiply layer, ElementWiseOperation::MUL
 * layer.
 */
class MatrixMultiplyOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3)
        << "convert a matrix_multiply op to AscendIE MatrixMultiply layer +  "
           "ElementWiseOperation::MUL layer(if alpha != 1).";

    // Input: X, Y
    // Output: Out
    // Attributes: transpose_x, transpose_y, x_num_col_dims, y_num_col_dims,
    // alpha. extra Attributes(for quant dequant): X, Y, Out, Input_scale,
    // out_threshold.
    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);

    bool enable_int8 = (engine_->precision() == phi::DataType::INT8);
    float x_scale = 0;
    float y_scale = 0;
    float out_scale = 0;

    if (enable_int8) {
      if (op_desc.HasAttr("Input_scale")) {
        x_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Input_scale"));
        engine_->SetTensorDynamicRange(input1, x_scale);
      }
      if (op_desc.HasAttr("X")) {
        x_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("X"));
        engine_->SetTensorDynamicRange(input1, x_scale);
      }

      if (op_desc.HasAttr("Y")) {
        y_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Y"));
        engine_->SetTensorDynamicRange(input2, y_scale);
      }

      if (op_desc.HasAttr("out_threshold")) {
        out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      }
      if (op_desc.HasAttr("Out")) {
        out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
      }
    }

    auto output_name = op_desc.Output("Out")[0];

    AscendIE::Dims dims_x = input1->GetDimensions();
    int32_t x_rank = dims_x.Size();
    AscendIE::Dims dims_y = input2->GetDimensions();
    int32_t y_rank = dims_y.Size();

    int32_t x_num_col_dims =
        PADDLE_GET_CONST(int32_t, op_desc.GetAttr("x_num_col_dims"));
    if (x_num_col_dims < 0) {
      x_num_col_dims += x_rank;
    }

    // Temporarily solve the reformat problem of matrix multiplication, make
    // input.rank == 4. 
    if (x_rank == 2 && x_num_col_dims == 1 && engine_->use_varseqlen()) {
      VLOG(3) << "Temporarily solve the reformat problem of matrix "
                 "multiplication, make input.rank == 4. ";
      AscendIE::ShuffleLayer* reshape_before_matrix = engine_->network()->AddShuffle(input1);
      std::vector<AscendIE::Tensor*> reshape_before_tensor;
      reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input1), 0));
      reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input1), 1));
      reshape_before_tensor.push_back(Add1DConstantLayer(1));
      reshape_before_tensor.push_back(Add1DConstantLayer(1));

      reshape_before_matrix->SetInput(1, Concat(reshape_before_tensor));
      reshape_before_matrix->SetName(
          ("reshape_before_matrix(Output: " + output_name + ")").c_str());
      input1 = reshape_before_matrix->GetOutput(0);
      dims_x = input1->GetDimensions();
      x_rank = dims_x.Size();

      if (enable_int8) {
        if (op_desc.HasAttr("Input_scale") || op_desc.HasAttr("X")) {
          engine_->SetTensorDynamicRange(input1, x_scale);
        }
      }
    }

    if (x_num_col_dims != x_rank - 1) {
      std::vector<AscendIE::Tensor*> before_shape_tensors;
      AscendIE::Tensor* input_shape_tensor = Shape(input1);
      for (int i = 0; i < x_num_col_dims; ++i) {
        before_shape_tensors.push_back(
            GetEleTensorOfShape(input_shape_tensor, i));
      }
      AscendIE::Tensor* producted = Add1DConstantLayer(1);
      for (int i = x_num_col_dims; i < x_rank; ++i) {
        producted = Prod(producted, GetEleTensorOfShape(input_shape_tensor, i));
      }
      before_shape_tensors.push_back(producted);
      AscendIE::Tensor* before_shape_tensor = Concat(before_shape_tensors);
      auto* reshape_before_layer = engine_->network()->AddShuffle(input1);
      reshape_before_layer->SetInput(1, before_shape_tensor);
      reshape_before_layer->SetName(
          ("reshape_x_before_matrix_multiply: Shuffle (Output: " + output_name +
           ")")
              .c_str());
      input1 = reshape_before_layer->GetOutput(0);

      if (enable_int8) {
        if (op_desc.HasAttr("Input_scale") || op_desc.HasAttr("X")) {
          engine_->SetTensorDynamicRange(input1, x_scale);
        }
      }

      x_rank = x_num_col_dims + 1;
    }

    int32_t y_num_col_dims =
        PADDLE_GET_CONST(int32_t, op_desc.GetAttr("y_num_col_dims"));
    if (y_num_col_dims < 0) {
      y_num_col_dims += y_rank;
    }
    PADDLE_ENFORCE_EQ(
        y_num_col_dims,
        y_rank - 1,
        platform::errors::InvalidArgument(
            "The matrix_multiply op'y_num_col_dims should be equal "
            "to y'rank - 1, but got y_num_col_dims = %d, and y_rank = %d",
            y_num_col_dims,
            y_rank - 1));

    if (x_rank != 1 && y_rank != 1 && x_rank != y_rank) {
      if (x_rank < y_rank) {
        std::vector<AscendIE::Tensor*> before_shape_tensors;
        AscendIE::Tensor* input_shape_tensor = Shape(input1);
        for (int i = 0; i < y_rank - x_rank; ++i) {
          before_shape_tensors.push_back(Add1DConstantLayer(1));
        }
        for (int i = 0; i < x_rank; ++i) {
          before_shape_tensors.push_back(
              GetEleTensorOfShape(input_shape_tensor, i));
        }
        AscendIE::Tensor* before_shape_tensor = Concat(before_shape_tensors);
        auto* reshape_before_layer = engine_->network()->AddShuffle(input1);
        reshape_before_layer->SetInput(1, before_shape_tensor);
        reshape_before_layer->SetName(
            ("full_x_before_matrix_multiply: Shuffle (Output: " + output_name +
             ")")
                .c_str());
        input1 = reshape_before_layer->GetOutput(0);

        if (enable_int8) {
          if (op_desc.HasAttr("Input_scale") || op_desc.HasAttr("X")) {
            engine_->SetTensorDynamicRange(input1, x_scale);
          }
        }
        x_rank = y_rank;
      } else {
        std::vector<AscendIE::Tensor*> before_shape_tensors;
        AscendIE::Tensor* input_shape_tensor = Shape(input2);

        for (int i = 0; i < x_rank - y_rank; ++i) {
          before_shape_tensors.push_back(Add1DConstantLayer(1));
        }
        for (int i = 0; i < y_rank; ++i) {
          before_shape_tensors.push_back(
              GetEleTensorOfShape(input_shape_tensor, i));
        }
        AscendIE::Tensor* before_shape_tensor = Concat(before_shape_tensors);
        auto* reshape_before_layer = engine_->network()->AddShuffle(input2);
        reshape_before_layer->SetInput(1, before_shape_tensor);
        reshape_before_layer->SetName(
            ("full_y_before_matrix_multiply: Shuffle (Output: " + output_name +
             ")")
                .c_str());
        input2 = reshape_before_layer->GetOutput(0);

        if (enable_int8) {
          if (op_desc.HasAttr("Y")) {
            engine_->SetTensorDynamicRange(input2, y_scale);
          }
        }
      }
      y_rank = x_rank;
    }

    AscendIE::MatrixOperation matrix_operation_x;
    AscendIE::MatrixOperation matrix_operation_y;

    if (x_rank == 1) {
      matrix_operation_x = AscendIE::MatrixOperation::VECTOR;
    } else {
      bool transpose_x = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_x"));
      matrix_operation_x = transpose_x ? AscendIE::MatrixOperation::TRANSPOSE
                                       : AscendIE::MatrixOperation::NONE;
    }

    if (y_rank == 1) {
      matrix_operation_y = AscendIE::MatrixOperation::VECTOR;
    } else {
      bool transpose_y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_y"));
      matrix_operation_y = transpose_y ? AscendIE::MatrixOperation::TRANSPOSE
                                       : AscendIE::MatrixOperation::NONE;
    }

    AscendIE::BaseLayer* layer = nullptr;
    MatrixMultiplyLayer *layer =
        engine_->network()->AddMatrixMultiply(input1, matrix_operation_x, input2, matrix_operation_y);

    if (enable_int8) {
      if (op_desc.HasAttr("out_threshold") || op_desc.HasAttr("Out")) {
        engine_->SetTensorDynamicRange(layer->GetOutput(0), out_scale);
      }
    }

    float alpha = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));
    if (alpha < 0.999 || alpha > 1.001) {
      auto* alpha_tensor = Add1DConstantLayer(alpha);
      std::vector<AscendIE::Tensor*> alpha_shape_tensors;
      for (int i = 0; i < layer->GetOutput(0)->GetDimensions().Size(); i++) {
        alpha_shape_tensors.push_back(Add1DConstantLayer(1));
      }
      auto* reshape_alpha = engine_->network()->AddShuffle(alpha_tensor);
      reshape_alpha->SetInput(1, Concat(alpha_shape_tensors));
      layer = engine_->network()->AddElementWise(layer->GetOutput(0), reshape_alpha->GetOutput(0), AscendIE::ElementWiseOperation::MUL);
    }
    RreplenishLayerAndOutput(
        layer, "matrix_multiply_op", {output_name}, test_mode);
  }
};

}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(matrix_multiply, MatrixMultiplyOpConverter);
