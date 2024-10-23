// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "default_opset.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs eye(const NodeContext& node) {
    auto num_columns = node.get_attribute<int64_t>("num_columns");
    auto num_rows = node.get_attribute<int64_t>("num_rows");
    auto out_dtype = node.get_attribute<ov::element::Type>("dtype");

    auto x=default_opset::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(num_columns)});
    auto y=default_opset::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(num_rows)});
    // eye support only main diagonal
    auto diagonal = default_opset::Constant::create(element::i32, Shape{}, {0});
    auto eye=std::make_shared<default_opset::Eye>(x,y,diagonal,ov::element::i32);
    return node.default_single_output_mapping({std::make_shared<default_opset::Convert>(eye, out_dtype)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov