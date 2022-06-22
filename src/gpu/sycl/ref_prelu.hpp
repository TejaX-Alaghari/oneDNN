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

#ifndef GPU_SYCL_GENERIC_PRELU_HPP
#define GPU_SYCL_GENERIC_PRELU_HPP

#include "gpu/gpu_prelu_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_prelu_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_prelu_fwd_pd_t {
        using gpu_prelu_fwd_pd_t::gpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_prelu_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));

            const bool ok = check_data_types(data_d, weights_d)
                    && check_formats(data_d, weights_d);

            if (!ok) return status::unimplemented;
            // TODO: Support for postops & scales

            // TODO: extend sycl device info to check supported sub-group sizes.
            auto *sycl_engine
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine);
            const auto supported_sub_group_sizes
                    = sycl_engine->device()
                              .template get_info<
                                      ::sycl::info::device::sub_group_sizes>();
            if (!std::any_of(supported_sub_group_sizes.cbegin(),
                        supported_sub_group_sizes.cend(),
                        [](size_t size) { return size == 32; })) {
                return status::unimplemented;
            }

            return init_conf();
        }
        status_t init_conf();

        sycl_prelu_conf_t conf_;

    private:
        static bool check_data_types(const memory_desc_wrapper &data,
                const memory_desc_wrapper &weights) {
            using namespace data_type;

            const auto data_dt = data.data_type();
            const auto weights_dt = weights.data_type();

            for (auto t : {data_dt, weights_dt}) {
                if (!utils::one_of(t, f32, s32, bf16, s8, u8)) return false;
            }

            return IMPLICATION(utils::one_of(bf16, data_dt, weights_dt),
                    data_dt == weights_dt);
        }

        static bool check_formats(const memory_desc_wrapper &data,
                const memory_desc_wrapper &weights) {
            using namespace format_tag;

            for (const auto &mdw : {data, weights}) {
                if (mdw.matches_one_of_tag(ab, abc, abcd, abcde) == undef) {
                    return false;
                }
            }
            return true;
        }
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif