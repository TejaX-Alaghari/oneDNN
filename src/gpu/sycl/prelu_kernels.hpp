/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_SYCL_PRELU_KERNELS_HPP
#define GPU_SYCL_PRELU_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct prelu_fwd_kernel_vec_t {
    static constexpr int vec_len = 8;

    prelu_fwd_kernel_vec_t(const sycl_prelu_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_in_memory_arg_t &weights,
            sycl_out_memory_arg_t &dst)
        : conf_(conf), data_(data), weights_(weights), dst_(dst) {}

    [[sycl::reqd_sub_group_size(32)]] void operator()(
            ::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();

        size_t base = ((item.get_group(0) * conf_.wg_size
                               + sg.get_group_id()[0] * sg.get_local_range()[0])
                                      * conf_.block_size
                              + sg.get_local_id() * conf_.block_size)
                / vec_len;

        size_t base_idx = base * vec_len;

        if (base_idx + conf_.block_size < conf_.nelems) {
            for (int i = 0; i < conf_.block_size / vec_len; i++) {
                auto data_vec = load_float_vec<vec_len>(
                        data_md().data_type(), data_ptr(), base + i);
                auto weights_vec = load_float_vec<vec_len>(
                        weights_md().data_type(), weights_ptr(), base + i);
                auto dst_vec = load_float_vec<vec_len>(
                        dst_md().data_type(), dst_ptr(), base + i);

                auto acc_vec = compute_prop(
                        data_vec, weights_vec, dst_vec, conf_.prop_kind);

                // TODO: Adding post-ops seems to be interfering with compiler's
                // optimizations. Figure out how to make the compiler to generate
                // the right code.
                //acc_vec = conf_.post_ops.apply(acc_vec, dst_vec);
                store_float_vec(
                        dst_md().data_type(), acc_vec, dst_ptr(), base + i);
            }
        } else {
            if (prop == prop_kind::forward_training
                    || prop == prop_kind::forward_inference) {
                for (int i = base_idx; i < conf_.nelems; i++)
                    (dst_ptr())[i] = (data_ptr())[i] > 0
                            ? (data_ptr())[i]
                            : ((data_ptr())[i] * (weights_ptr())[i]);
            }
        }
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &weights_md() const { return conf_.weights_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *weights_ptr() const { return weights_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    template <int width>
    ::sycl::vec<float, width> compute_prop(::sycl::vec<float, width> data,
            ::sycl::vec<float, width> weights, ::sycl::vec<float, width> dst,
            prop_kind_t prop) const {
        if (prop == prop_kind::forward_training
                || prop == prop_kind::forward_inference) {
            for (int i = 0; i < data.size(); i++) {
                dst[i] = data[i] > 0 ? data[i] : (data[i] * weights[i]);
            }
        }

        return dst.template convert<float>();
    }

    sycl_prelu_conf_t conf_;

    sycl_in_memory_arg_t data_;
    sycl_in_memory_arg_t weights_;
    sycl_out_memory_arg_t dst_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif