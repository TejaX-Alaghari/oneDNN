/*******************************************************************************
 * Copyright 2021 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "interface/shape_infer.hpp"
#include "interface/value.hpp"
#include "utils/debug.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/passes/utils.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper_t;

// this function fuse a op to its successor.
// you should guarantee that the op has only one successor
//
//   in_val
//     |
//   next_op         in_val
//     |      --->     |
//   base_op         base_op
//     |               |
//   out_val         out_val
void fuse_op_to_successor(op_t *op, std::vector<op_ptr> &subgraph) {
    assertm(op->num_inputs() == 1, "this op should have only one input value.");
    value_ptr in_val = op->get_input_value(0);
    in_val->remove_consumer(*op, 0);

    assertm(op->num_outputs() == 1,
            "this op should have only one output value.");
    value_ptr out_val = op->get_output_value(0);
    auto consumers = out_val->get_consumers();
    assertm(!consumers.empty() && consumers.size() == 1,
            "this op has zero consumer or more than one consumers.");

    op_t &successor = consumers[0].get_op();
    size_t offset = consumers[0].get_offset();
    in_val->add_consumer(successor, offset);
    successor.connect_input(offset, in_val);

    auto pos = std::find_if(subgraph.begin(), subgraph.end(),
            [op](const op_ptr &tmp) { return op == tmp.get(); });
    if (pos != subgraph.end()) subgraph.erase(pos);
}

//   in_val                  in_val     in_val2
//     |                         \       /
//   base_op  in_val2             base_op
//      \       /       --->         |
//       next_op                  out_val
//          |
//       out_val
void fuse_op_to_predecessor(
        op_t *op, std::vector<op_ptr> &subgraph, size_t in_offset) {
    value_ptr in_val = op->get_input_value(in_offset);
    assertm(op->num_outputs() == 1,
            "this op should have only one output value.");
    value_ptr out_val = op->get_output_value(0);

    op_t &predecessor = in_val->get_producer();
    size_t offset = in_val->get_offset();
    predecessor.connect_output(offset, out_val);

    for (size_t i = 0; i < op->num_inputs(); i++) {
        value_ptr tmp = op->get_input_value(i);
        if (tmp == in_val) { continue; }

        tmp->remove_consumer(*op, i);
        tmp->add_consumer(predecessor, predecessor.num_inputs());
        predecessor.add_input(tmp);
    }

    auto pos = std::find_if(subgraph.begin(), subgraph.end(),
            [op](const op_ptr &tmp) { return op == tmp.get(); });
    if (pos != subgraph.end()) subgraph.erase(pos);
}

//   in_val          in_val
//     |               |
//     |     ->    inserted_op
//     |               |
//     |             new_val
//     |               |
//  base_op         base_op
void insert_op_before(op_ptr &inserted_op, op_ptr &base_op, size_t offset) {
    return insert_op_before(inserted_op.get(), base_op.get(), offset);
}

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t offset) {
    value_ptr in_val = base_op->get_input_value(offset);
    in_val->remove_consumer(*base_op, offset);
    in_val->add_consumer(*inserted_op, inserted_op->num_inputs());
    inserted_op->add_input(in_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*inserted_op, 0, new_lt, true);
    inserted_op->add_output(new_val);

    new_val->add_consumer(*base_op, offset);
    base_op->connect_input(offset, new_val);
}

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t base_offset,
        size_t inserted_offset) {
    value_ptr in_val = base_op->get_input_value(base_offset);
    in_val->remove_consumer(*base_op, base_offset);
    inserted_op->connect_input(inserted_offset, in_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*inserted_op, 0, new_lt, true);
    inserted_op->add_output(new_val);

    new_val->add_consumer(*base_op, base_offset);
    base_op->connect_input(base_offset, new_val);
}

//   base_op         base_op
//     |               |
//     |             new_val
//     |               |
//     |     ->    inserted_op
//     |               |
//  out_val         out_value
void insert_op_after(op_ptr &inserted_op, op_ptr &base_op, size_t offset) {
    insert_op_after(inserted_op.get(), base_op.get(), offset);
}

void insert_op_after(op_t *inserted_op, op_t *base_op, size_t offset) {
    value_ptr out_val = base_op->get_output_value(offset);
    inserted_op->add_output(out_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*base_op, 0, new_lt, true);
    base_op->connect_output(offset, new_val);

    new_val->add_consumer(*inserted_op, inserted_op->num_inputs());
    inserted_op->add_input(new_val);
}

void insert_op_after(op_t *inserted_op, op_t *base_op, size_t output_offset,
        size_t input_offset) {
    value_ptr out_val = base_op->get_output_value(output_offset);
    inserted_op->add_output(out_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*base_op, 0, new_lt, true);
    base_op->connect_output(output_offset, new_val);

    new_val->add_consumer(*inserted_op, input_offset);
    inserted_op->connect_input(input_offset, new_val);
}

status_t set_given_inputs_outputs(std::shared_ptr<subgraph_t> &sg,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs) {
    sg->ins_ = inputs;
    sg->outs_ = outputs;

    // set the inputs's layout to subgraph's inputs value
    auto graph_in_vals = sg->get_input_values();
    auto graph_out_vals = sg->get_output_values();

    auto func = [](std::vector<value_t *> &edges,
                        const std::vector<impl::logical_tensor_t> &givens,
                        bool check_given, bool must_have_shape) {
        for (auto &edge : edges) {
            size_t edge_id = edge->get_logical_tensor().id;

            // partition in/outs should not have default id. There must be some
            // errors in previous graph transformation stage
            if (edge_id == std::numeric_limits<size_t>::max())
                return status::invalid_graph;

            bool found = false;
            for (const auto &given : givens) {
                if (edge_id == given.id) {
                    if (check_given) {
                        // check given lts
                        bool valid = given.data_type != impl::data_type::undef;
                        if (must_have_shape) {
                            valid = valid && given.ndims > 0;
                            for (size_t i = 0; i < given.ndims; i++) {
                                valid = valid && given.dims[i] != -1;
                            }
                        }
                        if (!valid) return status::invalid_argument;
                    }

                    edge->set_logical_tensor(given);
                    found = true;
                    break;
                }
            }

            if (!found) return status::miss_ins_outs;
        }
        return status::success;
    };

    status_t ret;
    ret = func(graph_in_vals, inputs, true, true);
    if (ret != status::success) return ret;

    ret = func(graph_out_vals, outputs, true, false);
    return ret;
}

status_t set_given_inputs_outputs(std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs) {
    auto sg = std::make_shared<subgraph_t>(subgraph);
    return set_given_inputs_outputs(sg, inputs, outputs);
}

void set_all_layout_to_any(std::vector<op_ptr> &subgraph) {
    for (auto &cur_op : subgraph) {
        for (const auto &val : cur_op->get_input_values()) {
            val->set_layout_type(impl::layout_type::any);
        }

        for (const auto &val : cur_op->get_output_values()) {
            val->set_layout_type(impl::layout_type::any);
        }
    }
}

// Constant property should be set by users from API level, this function is
// just a workaround at this moment.
void set_weight_bias_constant(std::vector<op_ptr> &subgraph) {
    for (auto &op : subgraph) {
        if (!(op->get_kind() == impl::op_kind::MatMul
                    || op->get_kind() == impl::op_kind::Convolution
                    || op->get_kind() == op_kind::dnnl_convolution))
            continue;

        // set weight to be constant
        op->get_input_value(1)->set_property(property_type::constant);

        // set bias to be constant
        if (op->get_attr<bool>("with_bias")) {
            op->get_input_value(2)->set_property(property_type::constant);
        }
    }
}

#ifdef DNNL_GRAPH_ENABLE_DUMP
namespace {
std::string layout2str(const dnnl::memory::desc &md) {
    std::string str;

    if (md.dims().empty()) return "";

    // format tag
    if (md.data.format_kind == dnnl_blocked) {
        std::string blk_tag;

        int ndims = md.data.ndims;
        auto &blk = md.data.format_desc.blocking;

        dnnl_dims_t blocks = {0};
        std::fill(blocks, blocks + ndims, 1);
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk)
            blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];

        char dim_chars[DNNL_MAX_NDIMS + 1] = {'\0'};

        dims_t ou_blocks = {0};
        std::copy(md.data.padded_dims, md.data.padded_dims + ndims, ou_blocks);

        bool plain = true;
        for (int d = 0; d < ndims; ++d) {
            dim_chars[d] = static_cast<char>((blocks[d] == 1 ? 'a' : 'A') + d);
            if (blocks[d] != 1) plain = false;
            ou_blocks[d] /= blocks[d];
        }

        dnnl_dims_t strides = {0};
        std::copy(blk.strides, blk.strides + ndims, strides);

        utils::simultaneous_sort(strides, ou_blocks, dim_chars, ndims,
                [](dim_t a, dim_t b) { return b - a; });

        blk_tag = std::string(dim_chars);

        if (!plain) {
            for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
                blk_tag += std::to_string(blk.inner_blks[iblk])
                        + static_cast<char>('a' + blk.inner_idxs[iblk]);
            }
        }

        str += blk_tag;
    } else if (md.data.format_kind == dnnl_format_kind_any) {
        str += "any";
    } else if (md.data.format_kind == dnnl_format_kind_undef) {
        str += "undef";
    }

    return str;
}

const std::string &kind2str(op_kind_t kind) {
    // 0: Abs, ..., N: LastSymbol, 0x1234: any, ...
    const size_t k = static_cast<size_t>(kind);
    const size_t l
            = static_cast<size_t>(dnnl::graph::impl::op_kind::LastSymbol);

    if (k <= l) {
        return impl::op_kind::op_kind_strings.at(k);
    } else {
        return impl::dnnl_impl::op_kind::internal_op_strings.at(k
                - static_cast<size_t>(op_kind::kDNNL_INTERNAL_OP_STARTER) - 1);
    }
}

std::string property2str(impl::property_type_t ptype) {
    std::string str;
    switch (ptype) {
        case impl::property_type::undef: str = "undef"; break;
        case impl::property_type::variable: str = "variable"; break;
        case impl::property_type::constant: str = "constant"; break;
        default: break;
    }
    return str;
}
} // namespace
#endif

status_t subgraph_visualizer_t::run(const std::shared_ptr<subgraph_t> &sg,
        const std::string &name_suffix, bool is_layout_sensitive,
        bool is_memory_sensitive) {
#ifdef DNNL_GRAPH_ENABLE_DUMP
    if (!enabled_) return status::success;

    std::ofstream out;

    std::string backend_name = dnnl_backend::get_singleton().get_name();
    std::string partition_name = "partition_" + std::to_string(partition_id_);
    std::string index_str = std::to_string(index_++);
    const std::string &pass_name = name_suffix;

    // file_name: (backend_name)_partition_(id)_(index)_(pass_name).dot
    std::string file_name = backend_name + "_" + partition_name + "_"
            + index_str + "_" + pass_name + ".dot";
    std::cout << "visualize partition subgraph to a dot file: " << file_name
              << std::endl;

    // ID or address when ID is not available
    auto get_op_identifier = [](op_t *op) {
        if (op->get_id() != op_t::DEFAULT_ID) return op->get_id();
        return reinterpret_cast<size_t>(op);
    };

    out.open(file_name);
    out << "digraph G {\n";
    topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        const auto &cur_op_name = kind2str(op->get_kind());
        const size_t cur_op_id = get_op_identifier(op);
        if (op->num_inputs() > 0) {
            for (size_t i = 0; i < op->num_inputs(); ++i) {
                auto input_value = op->get_input_value(i);
                if (input_value->has_producer()) {
                    op_t *input_op = &(input_value->get_producer());
                    const auto &input_op_name = kind2str(input_op->get_kind());
                    const size_t input_op_id = get_op_identifier(input_op);
                    out << "\"" << input_op_name << "_" << input_op_id
                        << "\" -> \"" << cur_op_name << "_" << cur_op_id
                        << "\";\n";
                }
            }
        } else {
            out << "\"" << cur_op_name << "_" << cur_op_id << "\"[label=\""
                << cur_op_name << "_" << cur_op_id << "\"];\n";
        }
        return status::success;
    });

    // value str: (data_type):(logical tensor id):(layout type):(dims):(layout
    // desc):(property):(mem_info)
    auto val2str = [this, is_layout_sensitive, is_memory_sensitive](
                           const value_t *val) {
        auto dims2str = [](const impl::dims &dims) {
            if (dims.empty()) return std::string("");

            std::string str;
            str += std::to_string(dims[0]);
            for (int d = 1; d < dims.size(); ++d)
                str += ("x" + std::to_string(dims[d]));
            return str;
        };

        auto lt = val->get_logical_tensor();
        auto ltw = impl::logical_tensor_wrapper_t(lt);
        std::string str
                = std::string(impl::utils::data_type2str(ltw.data_type())) + ":"
                + ((ltw.id() < std::numeric_limits<size_t>::max())
                                ? std::to_string(ltw.id())
                                : "def")
                + ":"
                + std::string(impl::utils::layout_type2str(ltw.layout_type()))
                + ":"
                + dims2str(ltw.ndims() < 0 ? std::vector<impl::dim_t>()
                                           : ltw.vdims())
                + ":"
                + (is_layout_sensitive ? layout2str(make_dnnl_memory_desc(lt))
                                       : "")
                + ":" + property2str(ltw.property_type()) + ":"
                + (is_memory_sensitive ? this->mem_info_func_(val) : "");
        return str;
    };

    // dump inputs/outputs info
    // in(no)_(lt str) or out(no)_(lt str)
    topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        const auto &op_name = kind2str(op->get_kind());
        const size_t op_id = get_op_identifier(op);
        out << "\"" << op_name << "_" << op_id << "\"[label=\"" << op_name
            << "_" << op_id;

        for (size_t i = 0; i < op->num_inputs(); i++) {
            out << "\\n"
                << "in" << std::to_string(i) << "_"
                << val2str(op->get_input_value(i).get());
        }

        for (size_t i = 0; i < op->num_outputs(); i++) {
            out << "\\n"
                << "out" << std::to_string(i) << "_"
                << val2str(op->get_output_value(i).get());
        }

        out << "\"];\n";
        return status::success;
    });

    out << "}\n";
    out.close();
#else
    UNUSED(sg);
    UNUSED(name_suffix);
    UNUSED(is_layout_sensitive);
    UNUSED(is_memory_sensitive);
#endif

    return status::success;
}

void replace_op(op_ptr &org_op, op_ptr &new_op) {
    new_op->merge_attributes(org_op->get_attributes());

    for (size_t i = 0; i < org_op->num_inputs(); i++) {
        auto in_val = org_op->get_input_value(i);
        in_val->remove_consumer(*org_op, i);
        in_val->add_consumer(*new_op, new_op->num_inputs());
        new_op->add_input(in_val);
    }
    for (size_t i = 0; i < org_op->num_outputs(); i++) {
        auto out_val = org_op->get_output_value(i);
        new_op->add_output(out_val);
    }
}

std::vector<value_t *> get_constant_block_output_values(
        const std::vector<op_ptr> &subgraph) {
    using ltw = impl::logical_tensor_wrapper_t;
    std::vector<value_t *> ret;
    for (auto &cur_op : subgraph) {
        auto out_vals = cur_op->get_output_values();
        for (auto &val : out_vals) {
            if (!ltw(val->get_logical_tensor()).is_constant()) continue;
            // if a constant value feed into a consumer whose output is not
            // constant, then the value is the final output of a constant block
            auto consumers = val->get_consumers();
            for (auto &csm : consumers) {
                // A consumer is not constant
                if (!csm.get_op().get_attr<bool>("is_constant")) {
                    ret.emplace_back(val.get());
                    break;
                }
            }
        }
    }
    return ret;
}

impl::status_t infer_shape(std::shared_ptr<subgraph_t> &sg) {
    auto ret = sg->infer_shape();
    if (ret != impl::status::success) return ret;

    // Fill the inferred shape and strides to subgraph's outputs
    for (size_t i = 0; i < sg->outs_.size(); i++) {
        for (auto val : sg->get_output_values()) {
            auto lt = val->get_logical_tensor();
            if (lt.id == sg->outs_[i].id) {
                auto inferred_shape = ltw(lt).vdims();
                set_shape_and_strides(sg->outs_[i], inferred_shape);
            }
        }
    }

    return ret;
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
        bool reset_layout)
    : impl::graph_t(ops), p_engine_(&eng) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, bool reset_layout)
    : impl::graph_t(ops), p_engine_(nullptr) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

const std::map<op_kind_t, dnnl::algorithm> &get_binary_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &binary_alg_map = {
            {impl::op_kind::Add, dnnl::algorithm::binary_add},
            {impl::op_kind::Multiply, dnnl::algorithm::binary_mul},
            {impl::op_kind::Divide, dnnl::algorithm::binary_div},
            {impl::op_kind::Minimum, dnnl::algorithm::binary_min},
            {impl::op_kind::Maximum, dnnl::algorithm::binary_max},
    };
    return binary_alg_map;
}

bool binary_doable(
        const std::vector<dim_t> &shape_0, const std::vector<dim_t> &shape_1) {
    const int ndims_0 = static_cast<int>(shape_0.size());
    const int ndims_1 = static_cast<int>(shape_1.size());
    const int small = ndims_0 < ndims_1 ? ndims_0 : ndims_1;
    for (int i = 1; i <= small; ++i) {
        bool match = shape_0[ndims_0 - i] == shape_1[ndims_1 - i]
                || shape_0[ndims_0 - i] == 1 || shape_1[ndims_1 - i] == 1;
        if (!match) return false;
    }
    return true;
}

static bool post_binary_fusible_impl(const impl::op_t *base_op,
        const std::vector<dim_t> &fused_shape,
        const std::vector<dim_t> &other_shape, const std::string &data_fmt) {
    assertm(fused_shape.size() == other_shape.size(),
            "must have same ndims, pls run binary_canonicalization pass first");
    // full tensor and per tensor broadcasted
    if (fused_shape == other_shape
            || std::all_of(other_shape.begin(), other_shape.end(),
                    [](dim_t i) { return i == 1; }))
        return true;

    // per mb_w broadcasted for 4d tensor MatMul
    int32_t output_ndims = static_cast<int32_t>(fused_shape.size());
    if (base_op->get_kind() == impl::op_kind::MatMul && output_ndims == 4) {
        int32_t w_axis = data_fmt == "NXC" ? 2 : 3;
        for (int32_t i = output_ndims - 1; i >= 0; i--) {
            if (other_shape[i] == 1) continue;
            if ((i != 0 && i != w_axis) || fused_shape[i] != other_shape[i]) {
                return false;
            }
        }
        return true;
    }

    // per channel broadcasted
    int32_t c_axis = data_fmt == "NXC" ? output_ndims - 1 : 1;
    for (int32_t i = output_ndims - 1; i >= 0; i--) {
        if (other_shape[i] == 1) continue;
        if (i != c_axis || fused_shape[i] != other_shape[i]) { return false; }
    }
    return true;
}

std::pair<bool, std::pair<size_t, int64_t>> shuffle_fusible(
        const impl::op_t *reshape0, impl::op_t *reshape1,
        impl::op_t *transpose) {
    using result_t = std::pair<bool, std::pair<size_t, int64_t>>;
    const result_t dflt_res {false, {0, 0}};

    const logical_tensor_t src_port
            = reshape0->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t dst_port
            = reshape1->get_output_value(0)->get_logical_tensor();
    const auto src_lt_shape = ltw(src_port).vdims();
    const auto dst_lt_shape = ltw(dst_port).vdims();
    const auto attr_shape = reshape0->get_attr<impl::dims>("shape");
    const auto tp_order = transpose->get_attr<impl::dims>("order");

    if (src_lt_shape != dst_lt_shape) return dflt_res;
    if (src_lt_shape.size() + 1 != attr_shape.size()) return dflt_res;

    size_t last_unmatched_pos = tp_order.size();
    size_t matched_pos = 0;
    for (size_t i = 0; i < tp_order.size(); ++i) {
        if (tp_order[i] == i)
            ++matched_pos;
        else
            last_unmatched_pos = i;
    }

    // more or less than two positions were swapped
    if (matched_pos != tp_order.size() - 2) return dflt_res;
    // all positions were matched
    if (last_unmatched_pos == tp_order.size()) return dflt_res;
    // transposition not on consecutive positions
    if (last_unmatched_pos != tp_order[last_unmatched_pos - 1]) return dflt_res;

    const size_t g_pos = last_unmatched_pos;
    const size_t c_over_g_pos = g_pos - 1;
    const int64_t groups = attr_shape[g_pos];
    auto mod_attr_shape = attr_shape;
    mod_attr_shape[c_over_g_pos] *= groups;
    mod_attr_shape.erase(mod_attr_shape.begin() + g_pos);

    if (src_lt_shape != mod_attr_shape) return dflt_res;

    return {true, {c_over_g_pos, groups}};
}

bool post_binary_fusible(const impl::op_t *base_op, const impl::op_t *bin_op) {
    std::string data_fmt = base_op->has_attr("data_format")
            ? base_op->get_attr<std::string>("data_format")
            : "NCX";
    auto fused_out = base_op->get_output_values()[0];
    auto consumers = fused_out->get_consumers();
    if (consumers.size() != 1) return false;

    size_t fused_in_off = consumers[0].get_offset();
    auto fused_in = bin_op->get_input_value(fused_in_off)->get_logical_tensor();
    auto other_in
            = bin_op->get_input_value(1 - fused_in_off)->get_logical_tensor();
    return post_binary_fusible_impl(
            base_op, ltw(fused_in).vdims(), ltw(other_in).vdims(), data_fmt);
}

bool post_depthwise_conv_fusible(const impl::op_t *conv_op) {
    if (!conv_op->has_attr("groups")) return false;
    if (conv_op->has_attr("auto_pad")
            && conv_op->get_attr<std::string>("auto_pad") != "None")
        return false;
    const auto strides = conv_op->get_attr<dims>("strides");
    const auto pads_begin = conv_op->get_attr<dims>("pads_begin");
    const auto pads_end = conv_op->get_attr<dims>("pads_end");
    const int32_t attrs_size = 2;
    for (int32_t i = 0; i < attrs_size; ++i) {
        if ((strides[i] != 1 && strides[i] != 2) || pads_begin[i] != 1
                || pads_end[i] != 1)
            return false;
    }
    const size_t wei_offset = 1;
    const logical_tensor_t wei_port
            = conv_op->get_input_value(wei_offset)->get_logical_tensor();
    if (wei_port.ndims != 4) return false;
    const auto groups = conv_op->get_attr<int64_t>("groups");
    const std::string wei_format = (conv_op->has_attr("filter_format"))
            ? conv_op->get_attr<std::string>("filter_format")
            : "XIO";
    const size_t oc_offset = (wei_format == "OIX") ? 0 : wei_port.ndims - 1;
    const size_t ic_offset = (wei_format == "OIX") ? 1 : wei_port.ndims - 2;
    const auto oc = wei_port.dims[oc_offset];
    const auto ic_over_g = wei_port.dims[ic_offset];
    if (groups == oc && oc == groups * ic_over_g) return true;
    return false;
}

const std::unordered_map<impl::op_kind_t, std::unordered_set<impl::op_kind_t>> &
get_post_ops_fusible_map() {
    using namespace impl::op_kind;
    using namespace dnnl_impl::op_kind;
    static const std::unordered_map<impl::op_kind_t,
            std::unordered_set<impl::op_kind_t>>
            fusible_map = {// conv
                    {Convolution,
                            {dnnl_eltwise, dnnl_binary, Convolution,
                                    dnnl_convolution}},
                    {dnnl_convolution,
                            {dnnl_eltwise, dnnl_binary, Convolution,
                                    dnnl_convolution}},
                    // deconv
                    {ConvTranspose, {dnnl_eltwise, dnnl_binary}},
                    {dnnl_convtranspose, {dnnl_eltwise, dnnl_binary}},
                    // matmul
                    {MatMul, {dnnl_eltwise, dnnl_binary}},
                    // pool
                    {AvgPool, {dnnl_binary}}, {MaxPool, {dnnl_binary}},
                    {dnnl_pool, {dnnl_binary}},
                    // eltwise
                    {dnnl_eltwise, {dnnl_binary}},
                    // binary
                    {dnnl_binary, {dnnl_eltwise, dnnl_binary}},
                    // bn
                    {dnnl_batchnorm, {dnnl_eltwise}},
                    {BatchNormInference, {dnnl_eltwise}},
                    // reorder
                    {Reorder, {dnnl_binary}}, {int8_reorder, {dnnl_binary}},
                    // reduction
                    {dnnl_reduction, {dnnl_eltwise, dnnl_binary}},
                    // resample
                    {Interpolate, {dnnl_eltwise, dnnl_binary}}};
    return fusible_map;
}

// data_format = NXC:
// (1, 2, 3, 4); (4) is doable
// data_format = NCX, channel broadcast = false:
// (1, 2, 3, 4); (4) is doable
// data_format = NCX, channel broadcast = true:
// (1, 2, 3, 4); (2) is doable

// src      wei
// (3, 4); (3, 4) is doable
// (1, 4); (3, 4) is not doable
// (3, 4); (1, 4) is doable
// (3, 4, 5); (4, 5) is doable
// (3, 4, 5); (1, 5) is doable
// (3, 4, 5); (2, 4, 5) is NOT doable
bool prelu_doable(const std::vector<dim_t> &src_dims,
        const std::vector<dim_t> &wei_dims, const std::string &data_format,
        const bool per_channel_broadcast) {
    const int src_ndims = static_cast<int>(src_dims.size());
    const int wei_ndims = static_cast<int>(wei_dims.size());
    // src ndims should be equal or greater than wei ndims
    if (src_ndims < wei_ndims) return false;

    bool doable = false;
    if (wei_ndims == 1) {
        if (per_channel_broadcast) {
            // if broadcast to channel,
            // then src channel dim should be equal to wei dim
            const int channel_dim_num
                    = data_format == "NCX" ? 1 : src_dims[src_ndims - 1];
            doable = src_dims[channel_dim_num] == wei_dims[0];
        } else {
            // if no broadcast to channel,
            // then wei dim should be equal to last src dim,
            // or equal to 1.
            doable = src_dims[src_ndims - 1] == wei_dims[0] || wei_dims[0] == 1;
        }
    } else {
        for (int i = 1; i <= wei_ndims; ++i) {
            // Weights are broadcastable to src when:
            // 1) they are equal on the same ndims,
            // 2) one of them is 1,
            // 3) In the case when weights have fewer dimensions,
            //    1s are added to the front and then 1) and 2) must be met.
            doable = src_dims[src_ndims - i] == wei_dims[wei_ndims - i]
                    || wei_dims[wei_ndims - i] == 1;
            if (!doable) break;
        }
    }
    return doable;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl