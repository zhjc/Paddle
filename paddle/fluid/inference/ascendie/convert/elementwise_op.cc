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

#include "paddle/fluid/inference/ascendie/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace ascendie { // "ascendie" is namespace in paddle paddle code; "AscendIE" is namespace of Ascend Inference Engine

class ElementwiseTensorOpConverter : public OpConverter {
 public:
  ElementwiseTensorOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "Convert a elementwise op to AscendIE IElementWiseLayer";
    framework::OpDesc op_desc(op, nullptr);
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    AscendIE::Tensor* Y = nullptr;
    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    if (Y_v) {
      // Y is weight
      auto* Y_t = Y_v->GetMutable<phi::DenseTensor>();
      std::vector<int> dims_y = phi::vectorize<int>(Y_t->dims());
      auto y_weight = engine_->GetTrtWeight(op_desc.Input("Y").front(), *Y_t);

      std::vector<int64_t> tmp_dims_y(dims_y.begin(), dims_y.end());
      AscendIE::Dims aie_dims_y(tmp_dims_y);

      // this is the special case when dims_y includes batch dimension!
      // we need remove batch dimension!
      if (!engine_->with_dynamic_shape() &&
          aie_dims_y.Size() == (X->GetDimensions().Size() + 1)) {
        PADDLE_ENFORCE_EQ(aie_dims_y[0],
                          1,
                          platform::errors::InvalidArgument(
                              "Elementwise type(%s) op's Y is a weight "
                              "including batch dimension. Please "
                              "check if the 0th dimension equals 1.",
                              op_type_));
        AscendIE::Dims remove_batch_dim_aie_dims_y(aie_dims_y.Size() - 1, aie_dims_y.Data() + 1);
        aie_dims_y = remove_batch_dim_aie_dims_y;

      }
      Y = engine_->network()->AddConstantLayer(aie_dims_y, y_weight.get())->GetOutput(0);

    } else {
      Y = engine_->GetITensor(op_desc.Input("Y").front());
    }
    bool swap_xy = false;
    // Swap X and Y
    if (X->GetDimensions().Size() < Y->GetDimensions().Size()) {
      auto* tmp = X;
      X = Y;
      Y = tmp;
      swap_xy = true;
    }
    AscendIE::Dims dims_x = X->GetDimensions();
    AscendIE::Dims dims_y = Y->GetDimensions();
    auto output_name = op_desc.Output("Out")[0];

    int axis = -1;
    // axis here is relative to explicit batch
    if (op_type_ != "logical_or" && op_type_ != "logical_xor" &&
        op_type_ != "logical_and") {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    }
    int real_x_rank = dims_x.Size();
    int real_y_rank = dims_y.Size();
    if (!engine_->with_dynamic_shape()) {
      real_x_rank++;
      real_y_rank++;
      if (Y_v) real_y_rank--;
    }
    if (axis == -1) {
      axis = real_x_rank - real_y_rank;
    }
    if (!engine_->with_dynamic_shape() && axis > 0) {
      axis--;
    }

    // X: - -  -    - - - -
    //        axis
    // Y:      -    - -
    // we need expand Y's rank = X's rank
    int left_one_num = axis;
    int right_one_num = dims_x.Size() - axis - dims_y.Size();
    AscendIE::ShuffleLayer* reshape_layer;
    AscendIE::Tensor* reshape_y_tensor;
    if (left_one_num > 0 || right_one_num > 0) {
      if (engine_->with_dynamic_shape()) {
        auto* y_shape_tensor = Shape(Y);
        auto* new_y_shape_tensor = y_shape_tensor;
        if (axis > 0) {
          std::vector<int32_t> left_one(left_one_num, 1);
          auto* left_one_tensor = Add1DConstantLayer(left_one); // TODO  ADD1DConstantLayer是nv 的接口，需要换成ascend的
          new_y_shape_tensor = Concat(std::vector<AscendIE::Tensor*>{
              left_one_tensor, new_y_shape_tensor});
        }
        if (right_one_num > 0) {
          std::vector<int32_t> right_one(right_one_num, 1);
          auto* right_one_tensor = Add1DConstantLayer(right_one); // TODO  ADD1DConstantLayer是nv 的接口，需要换成ascend的
          new_y_shape_tensor = Concat(std::vector<AscendIE::Tensor*>{  // TODO 需要改Concat
              new_y_shape_tensor, right_one_tensor});
        }
        reshape_layer = engine_->network()->AddShuffle(Y);
        reshape_layer->SetInput(1, new_y_shape_tensor);
      } else {
        std::vector<int64_t> new_y_vector(left_one_num + dims_y.Size() + right_one_num);
        for (int i = 0; i < new_y_vector.size(); i++) new_y_vector[i] = 1;
        AscendIE::Dims new_y_dims(new_y_vector);
      
        for (int i = 0; i < dims_y.Size(); i++)
          new_y_dims[left_one_num + i] = dims_y[i];
        reshape_layer = engine_->network()->AddShuffle(Y);
        reshape_layer->SetReshapeDimensions(new_y_dims);
      }
      reshape_y_tensor = reshape_layer->GetOutput(0);
    } else {
      // In fact , we can remove this `else`, but -> rt_resnet50_test CI in ascend
      // 6015 faling, how ridiculous！
      reshape_y_tensor = Y;
    }

    // We should swap X and Y back, because some operators do not have symmetry
    if (swap_xy) {
      auto* tmp = reshape_y_tensor;
      reshape_y_tensor = X;
      X = tmp;
    }

    auto op_pair = ops.find(op_type_);
    PADDLE_ENFORCE_NE(
        op_pair,
        ops.end(),
        platform::errors::InvalidArgument(
            "Elementwise op's type(%s) is not supported. Please "
            "check if the op_type is correct.",
            op_type_));

    auto* layer = engine_->network()->AddElementWise(X, reshape_y_tensor, op_pair->second);

    RreplenishLayerAndOutput(layer, "elementwise", {output_name}, test_mode);

  }

 protected:
  static const std::unordered_map<std::string, AscendIE::ElementWiseOperation>
      ops;
  std::string op_type_;
};

const std::unordered_map<std::string, AscendIE::ElementWiseOperation>
    ElementwiseTensorOpConverter::ops = {
        {"add", AscendIE::ElementWiseOperation::ADD},
        {"mul", AscendIE::ElementWiseOperation::MUL},
        {"sub", AscendIE::ElementWiseOperation::SUB},
        {"div", AscendIE::ElementWiseOperation::DIV},
        {"less_than", AscendIE::ElementWiseOperation::LESS},
        // {"logical_and", AscendIE::ElementWiseOperation::AND},
};

class ElementwiseTensorAddOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorAddOpConverter() { op_type_ = "add"; }
};

class ElementwiseTensorMulOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMulOpConverter() { op_type_ = "mul"; }
};

class ElementwiseTensorSubOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorSubOpConverter() { op_type_ = "sub"; }
};

class ElementwiseTensorDivOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorDivOpConverter() { op_type_ = "div"; }
};

class ElementwiseTensorLessThanOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorLessThanOpConverter() { op_type_ = "less_than"; }
};

// class ElementwiseTensorLogicalAndOpConverter
//     : public ElementwiseTensorOpConverter {
//  public:
//   ElementwiseTensorLogicalAndOpConverter() { op_type_ = "logical_and"; }
// };


}  // namespace ascendie
}  // namespace inference
}  // namespace paddle

REGISTER_ASCEND_OP_CONVERTER(elementwise_add_weight,
                          ElementwiseTensorAddOpConverter);
REGISTER_ASCEND_OP_CONVERTER(elementwise_mul_weight,
                          ElementwiseTensorMulOpConverter);
REGISTER_ASCEND_OP_CONVERTER(elementwise_sub_weight,
                          ElementwiseTensorSubOpConverter);
REGISTER_ASCEND_OP_CONVERTER(elementwise_div_weight,
                          ElementwiseTensorDivOpConverter);


REGISTER_ASCEND_OP_CONVERTER(elementwise_add_tensor,
                          ElementwiseTensorAddOpConverter);
REGISTER_ASCEND_OP_CONVERTER(elementwise_sub_tensor,
                          ElementwiseTensorSubOpConverter);
REGISTER_ASCEND_OP_CONVERTER(elementwise_div_tensor,
                          ElementwiseTensorDivOpConverter);
REGISTER_ASCEND_OP_CONVERTER(elementwise_mul_tensor,
                          ElementwiseTensorMulOpConverter);

REGISTER_ASCEND_OP_CONVERTER(less_than, ElementwiseTensorLessThanOpConverter);

// REGISTER_ASCEND_OP_CONVERTER(logical_and, ElementwiseTensorLogicalAndOpConverter);
