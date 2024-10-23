// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/core/node.hpp"
#include "openvino/frontend/node_context.hpp"
#include "default_opset.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace paddle {

std::tuple<Output<Node>, Output<Node>> get_shape_rank(const Output<Node>& x,
                                                      bool as_scalar) {
    auto shape = std::make_shared<ov::opset6::ShapeOf>(x);
    Output<Node> rank = std::make_shared<ov::opset6::ShapeOf>(shape);
    if (as_scalar) {
        auto axis_0 = ov::opset6::Constant::create(ov::element::i64, Shape{}, {0});
        rank = std::make_shared<ov::opset6::Squeeze>(rank, axis_0);
    }
    return std::make_tuple(shape, rank);
}
namespace op {
NamedOutputs scatter(const NodeContext& node) {
    // auto data = node.get_input("X");
    // auto alpha = node.get_attribute<float>("alpha");
    // aten::index_copy_(self, dim, index, tensor) → Tensor
    // num_inputs_check(context, 4, 4);
    auto x = node.get_input("X");
    auto ids = node.get_input("Ids");
    auto updates = node.get_input("Updates");
    auto dim_node = default_opset::Constant::create(element::i32, Shape{}, {0});
    auto overwrite = node.get_attribute<bool>("overwrite");
    std::shared_ptr<Node> input1_vec;
    
    auto const_1_vec = ov::opset6::Constant::create(ov::element::i32, ov::Shape{1}, {1});

    Output<Node> tensor_rank = std::get<1>(get_shape_rank(updates, true));
    auto tensor_rank_correct_type = std::make_shared<ov::opset1::ConvertLike>(tensor_rank, dim_node);
    // auto positive_dim = normalize_axis(dim, tensor_rank_correct_type);

    // begin the computation
    auto tensor_shape = std::make_shared<ov::opset6::ShapeOf>(updates, element::i32);
    auto dim_vec = std::make_shared<ov::opset6::Reshape>(dim_node, const_1_vec, false);
    auto broadcasted_index = std::make_shared<ov::opset6::Broadcast>(ids, tensor_shape, dim_vec);
    return node.default_single_output_mapping({std::make_shared<ov::opset6::ScatterElementsUpdate>(x, broadcasted_index, updates, dim_node)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov