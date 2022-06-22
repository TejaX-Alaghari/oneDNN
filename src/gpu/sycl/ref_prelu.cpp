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

#include "gpu/sycl/ref_prelu.hpp"
#include "gpu/sycl/prelu_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_prelu_fwd_t::pd_t::init_conf() {
    conf_ = sycl_prelu_conf_t();

    conf_.data_md = sycl_md_t(src_md(0));
    conf_.weights_md = sycl_md_t(weights_md(0));
    conf_.dst_md = sycl_md_t(dst_md(0));
    conf_.ndims = ndims();

    // XXX: should probably be tuned.
    conf_.block_size = 32;
    conf_.wg_size = 32;

    // TODO: uniform work groups are not supported for CUDA backend.
    // Need to find a way to circumvent it.
    if (memory_desc_wrapper(src_md(0)).nelems() % conf_.block_size != 0)
        return status::unimplemented;

    // TODO: Support for postops & scales

    return status::success;
}

status_t ref_prelu_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<prelu_fwd_kernel_vec_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_prelu_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto *data = CTX_IN_SYCL_STORAGE(DNNL_ARG_SRC);
    const auto *weights = CTX_IN_SYCL_STORAGE(DNNL_ARG_WEIGHTS);
    auto *dst = CTX_OUT_SYCL_STORAGE(DNNL_ARG_DST);

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto data_mem_arg = data->get_in_memory_arg(ctx.stream(), cgh);
        auto weights_mem_arg = weights->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_mem_arg = dst->get_out_memory_arg(ctx.stream(), cgh);
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        pd()->conf_.nelems = nelems_A;

        prelu_fwd_kernel_vec_t prelu_fwd_kernel(
                pd()->conf_, data_mem_arg, weights_mem_arg, dst_mem_arg);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        cgh.parallel_for(::sycl::nd_range<1>(nelems_A / block_size, wg_size),
                prelu_fwd_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl