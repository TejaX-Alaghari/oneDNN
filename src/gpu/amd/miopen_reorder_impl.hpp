/*******************************************************************************
* Copyright 2020 Intel Corporation
* 
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

#ifndef GPU_AMD_MIOPEN_REORDER_IMPL_HPP
#define GPU_AMD_MIOPEN_REORDER_IMPL_HPP

#include "common/type_helpers.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
// #include <cstdint>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

template <data_type_t dt>
struct dt_traits;

#define DECLARE_TRAIT(dt, t) \
    template <> \
    struct dt_traits<data_type_t::dt> { \
        using type = t; \
    };

DECLARE_TRAIT(dnnl_f16, ::sycl::half)
DECLARE_TRAIT(dnnl_f32, float)
DECLARE_TRAIT(dnnl_s32, int32_t)
DECLARE_TRAIT(dnnl_s8, int8_t)
DECLARE_TRAIT(dnnl_u8, uint8_t)
// DECLARE_TRAIT(dnnl_bf16, t)
#undef DECLARE_TRAIT

::sycl::nd_range<1> get_nd_range(const ::sycl::device &dev, int nelems);

void *allocate_buffer(data_type_t dt, int nelems, ::sycl::queue q);

template <data_type_t sdt, data_type_t ddt>
struct transform_kernel_t {
    using src_type = typename dt_traits<sdt>::type;
    using dst_type = typename dt_traits<ddt>::type;

    transform_kernel_t(const void *src, void *dst, int nelems)
        : src(src), dst(dst), nelems(nelems) {}

    void operator()(::sycl::nd_item<1> id) const {
        const auto local_id = id.get_global_id(0);
        if (local_id < nelems) {
            const auto *src_typed = reinterpret_cast<const src_type *>(src);
            auto *dst_typed = reinterpret_cast<dst_type *>(dst);
            dst_typed[local_id] = static_cast<dst_type>(src_typed[local_id]);
        }
    }

    const void *src;
    void *dst;
    int nelems;
};

struct miopen_reorder_generic_t {
public:
    virtual status_t init(const reorder_pd_t *pd) = 0;

    virtual void execute(miopenHandle_t handle, void *src, void *dst) const = 0;

    virtual ~miopen_reorder_generic_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, src_desc_);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, dst_desc_);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
    miopenDataType_t src_data_type_;
    miopenDataType_t dst_data_type_;
    int ndims_;
    int dims_[DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t src_desc_;
    miopenTensorDescriptor_t dst_desc_;
    float alpha_, beta_;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
    int nelems;

    data_type_t src_dt;
    data_type_t dst_dt;
};

// This structure is used when the memory format does not include blocking
struct miopen_reorder_stride_t : public miopen_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        memory_desc_wrapper wrap(pd->src_md());
        if (wrap.size() == 0) { return status::success; }

        nelems = wrap.nelems();

        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);
        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);
        alpha_ = pd->with_alpha() ? ((float)pd->attr()->output_scales_.mask_)
                                  : 1;
        beta_ = pd->beta();

        convert_dims(pd->dst_md()->dims, dims_, pd->dst_md()->ndims);
        convert_dims(pd->src_md()->format_desc.blocking.strides, src_strides_,
                pd->src_md()->ndims);
        convert_dims(pd->dst_md()->format_desc.blocking.strides, dst_strides_,
                pd->dst_md()->ndims);
        adjust_dim_for_dnn(dims_, pd->dst_md()->ndims, pd->src_md());
        adjust_stride_for_dnn(src_strides_, pd->dst_md()->ndims, pd->src_md());
        adjust_stride_for_dnn(dst_strides_, pd->dst_md()->ndims, pd->dst_md());
        ndims_ = pd->dst_md()->ndims >= 4 ? pd->dst_md()->ndims
                        + pd->dst_md()->format_desc.blocking.inner_nblks
                                          : 4;
        bool vectorized = has_different_block_size(pd->src_md(), pd->dst_md());
        convert_data_type(pd->src_md(), &src_data_type_, vectorized);
        convert_data_type(pd->dst_md(), &dst_data_type_, vectorized);

        src_dt = pd->src_md()->data_type;
        dst_dt = pd->dst_md()->data_type;

        // Create and set source tensor descriptor
        MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, &src_desc_);
        MIOPEN_EXECUTE_FUNC_S(miopenSetTensorDescriptor, src_desc_,
                dst_data_type_, ndims_, dims_, src_strides_); // src_data_type_

        // Create and set destination tensor descriptor
        MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, &dst_desc_);
        MIOPEN_EXECUTE_FUNC_S(miopenSetTensorDescriptor, dst_desc_,
                dst_data_type_, ndims_, dims_, dst_strides_);

        return status::success;
    }

    void execute(miopenHandle_t handle, void *src, void *dst) const override {
        // We don't need to specify the format (deducible using the strides)
        // in case of cudnnTransformTensor().
        // For example, this is useful when converting from abcd to bacd

        auto dev = ::sycl::device(::sycl::gpu_selector {});
        auto q = ::sycl::queue(dev);
        void *src_cpy = src;

        std::cout << "Starting execute" << std::endl;

#define SUBMIT_CASE(sdt, ddt) \
    if (src_dt == data_type_t::sdt && dst_dt == data_type_t::ddt) { \
        src_cpy = allocate_buffer(data_type_t::sdt, nelems, q); \
        const auto nd_range = get_nd_range(dev, nelems); \
        using src_type = typename dt_traits<sdt>::type; \
        using dst_type = typename dt_traits<ddt>::type; \
        int nelems_cpy = nelems; \
        std::cout << "nelems_cpy: " << nelems_cpy << std::endl; \
        std::cout << "nd_range: " << nd_range.get_global_range()[0] << " " \
                  << nd_range.get_local_range()[0] << std::endl; \
        q.submit([&](::sycl::handler &cgh) { \
            transform_kernel_t<data_type_t::sdt, data_type_t::ddt> tk( \
                    src, src_cpy, nelems); \
            cgh.parallel_for(nd_range, [=](::sycl::nd_item<1> id) { \
                const auto local_id = id.get_global_id(0); \
                if (local_id < nelems_cpy) { \
                    const auto *src_typed \
                            = reinterpret_cast<const src_type *>(src); \
                    auto *dst_typed = reinterpret_cast<dst_type *>(src_cpy); \
                    dst_typed[local_id] \
                            = static_cast<dst_type>(src_typed[local_id]); \
                } \
            }); \
        }); \
    }

        SUBMIT_CASE(dnnl_f16, dnnl_f32)
        SUBMIT_CASE(dnnl_f32, dnnl_f16)
        /*
        SUBMIT_CASE(miopenHalf, miopenInt32)
        SUBMIT_CASE(miopenHalf, miopenInt8)
        // SUBMIT_CASE(miopenHalf, u8)
        SUBMIT_CASE(miopenFloat, miopenHalf)
        SUBMIT_CASE(miopenFloat, miopenInt32)
        SUBMIT_CASE(miopenFloat, miopenInt8)
        // SUBMIT_CASE(miopenFloat, u8)
        SUBMIT_CASE(miopenInt32, miopenFloat)
        SUBMIT_CASE(miopenInt32, miopenInt8)
        // SUBMIT_CASE(miopenInt32, u8)
        SUBMIT_CASE(miopenInt8, miopenHalf)
        SUBMIT_CASE(miopenInt8, miopenFloat)
        SUBMIT_CASE(miopenInt8, miopenInt32)
        // SUBMIT_CASE(miopenInt8, u8)
        // SUBMIT_CASE(u8, miopenHalf)
        // SUBMIT_CASE(u8, miopenFloat)
        // SUBMIT_CASE(u8, miopenInt32)
        // SUBMIT_CASE(u8, miopenInt8)
        */
#undef SUBMIT_CASE

        q.wait_and_throw();

        std::cout << "Starting Transform Tensor" << std::endl;

        // ((float *)dst)[0] = 0; ((float *)dst)[1] = 0; ((float *)dst)[2] = 0; ((float *)dst)[3] = 0;

        std::cout << "src: " << src << " " << ((float *)src)[0] << " "
                  << ((float *)src)[1] << " " << ((float *)src)[2] << " "
                  << ((float *)src)[3] << std::endl;

        std::cout << "src_cpy: " << src_cpy << " " << ((::sycl::half *)src)[0]
                  << " " << ((::sycl::half *)src)[1] << " "
                  << ((::sycl::half *)src)[2] << " " << ((::sycl::half *)src)[3]
                  << std::endl;

        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &alpha_, src_desc_,
                src_cpy, &beta_, dst_desc_, dst);

        hipDeviceSynchronize();

        std::cout << "dst: " << dst << " " << ((::sycl::half *)dst)[0] << " "
                  << ((::sycl::half *)dst)[1] << " " << ((::sycl::half *)dst)[2]
                  << " " << ((::sycl::half *)dst)[3] << std::endl;

        /*
        float *dst_cpy = (float *)malloc(nelems * sizeof(float), 32);
        hipMemcpy(dst_cpy, dst, sizeof(float) * nelems, hipMemcpyDeviceToHost);

        std::cout << "dst: " << dst_cpy
                  << " " << ((float *)dst_cpy)[0]
                  << " " << ((float *)dst_cpy)[1]
                  << " " << ((float *)dst_cpy)[2]
                  << " " << ((float *)dst_cpy)[3] << std::endl;
        */

        /*
        std::cout << "dst: " << dst
                 << " " << ((float *)dst)[0]
                 << " " << ((float *)dst)[1]
                 << " " << ((float *)dst)[2]
                 << " " << ((float *)dst)[3] << std::endl;
        
        std::cout << "src: " << src
                  << " " << ((float *)src)[0]
                  << " " << ((float *)src)[1]
                  << " " << ((float *)src)[2]
                  << " " << ((float *)src)[3] << std::endl;
        */
    }

private:
    int src_strides_[DNNL_MAX_NDIMS];
    int dst_strides_[DNNL_MAX_NDIMS];

    using miopen_reorder_generic_t::miopen_reorder_generic_t;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_AMD_MIOPEN_REORDER_IMPL_HPP

/*
if (src_data_type_ == miopenDataType_t::sdt && dst_data_type_ == miopenDataType_t::ddt) { \
            const auto nd_range = get_nd_range(dev, nelems); \
            q.submit([&](::sycl::handler &cgh) { \
                transform_kernel_t<miopenDataType_t::sdt, miopenDataType_t::ddt> tk(src, dst, nelems); \
                cgh.parallel_for(nd_range, [=](::sycl::nd_item<1> id) { \
                    int i = id.get_global_id(0); \
                }); \
            }); \
        }
*/

/*
if (src_data_type_ == miopenDataType_t::sdt && dst_data_type_ == miopenDataType_t::ddt) { \
            src_cpy = allocate_buffer(miopenDataType_t::sdt, nelems, q); \
            const auto nd_range = get_nd_range(dev, nelems); \
            using src_type = typename dt_traits<sdt>::type; \
            using dst_type = typename dt_traits<ddt>::type; \
            int nelems_cpy = nelems; \
            std::cout << "nelems_cpy: " << nelems_cpy << std::endl; \
            std::cout << "nd_range: " << nd_range.get_global_range()[0] << " " << nd_range.get_local_range()[0] << std::endl; \
            q.submit([&](::sycl::handler &cgh) { \
                transform_kernel_t<miopenDataType_t::sdt, miopenDataType_t::ddt> tk(src, src_cpy, nelems); \
                cgh.parallel_for(nd_range, [=](::sycl::nd_item<1> id) { \
                    const auto local_id = id.get_global_id(0); \
                    if (local_id < nelems_cpy) { \
                        int a; \
                    } \
                }); \
            }); \
        }
*/

/*
if (src_data_type_ == data_type_t::sdt && dst_data_type_ == data_type_t::ddt) { \
            src_cpy = allocate_buffer(data_type_t::sdt, nelems, q); \
            const auto nd_range = get_nd_range(dev, nelems); \
            std::cout << "nd_range: " << nd_range.get_global_range()[0] << " " << nd_range.get_local_range()[0] << std::endl; \
            q.submit([&](::sycl::handler &cgh) { \
                transform_kernel_t<data_type_t::sdt, data_type_t::ddt> tk(src, src_cpy, nelems); \
                cgh.parallel_for(nd_range, tk); \
            }); \
        }
*/

/*
if (src_data_type_ == data_type_t::sdt && dst_data_type_ == data_type_t::ddt) { \
            src_cpy = allocate_buffer(data_type_t::sdt, nelems, q); \
            const auto nd_range = get_nd_range(dev, nelems); \
            using src_type = typename dt_traits<sdt>::type; \
            using dst_type = typename dt_traits<ddt>::type; \
            int nelems_cpy = nelems; \
            std::cout << "nelems_cpy: " << nelems_cpy << std::endl; \
            std::cout << "nd_range: " << nd_range.get_global_range()[0] << " " << nd_range.get_local_range()[0] << std::endl; \
            q.submit([&](::sycl::handler &cgh) { \
                transform_kernel_t<data_type_t::sdt, data_type_t::ddt> tk(src, src_cpy, nelems); \
                cgh.parallel_for(nd_range, [=](::sycl::nd_item<1> id) {                         \
                    const auto local_id = id.get_global_id(0);                                  \
                    if (local_id < nelems_cpy) {                                                \
                        const auto *src_typed   = reinterpret_cast<const src_type *>(src);      \
                        auto *dst_typed         = reinterpret_cast<dst_type *>(src_cpy);        \
                        dst_typed[local_id]     = static_cast<dst_type>(src_typed[local_id]);   \
                    }                                                                           \
                });                                                                             \
            }); \
        }
*/