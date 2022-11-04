/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_AMD_MIOPEN_CONV_FILTER_ADJUSTMENT_BASE_HPP
#define GPU_AMD_MIOPEN_CONV_FILTER_ADJUSTMENT_BASE_HPP

#include "miopen/miopen.h"

#include "common/type_helpers.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_conv_filter_adjustment_base_t {
public:
    float filter_alpha_ = 1, filter_beta_ = 0;
    miopenTensorDescriptor_t current_filter_desc_, transform_filter_desc_;
    virtual bool supported_filter_format(const memory_desc_t *md) const {
        const memory_desc_wrapper mem_wrapper(md);
        return (!(mem_wrapper.matches_one_of_tag(/*format_tag::oidhw,*/
                /*format_tag::odhwi,*/ format_tag::dhwio, format_tag::hwio)));
    }

    virtual ~miopen_conv_filter_adjustment_base_t() {
        if (current_filter_desc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, current_filter_desc_);
        }
        if (transform_filter_desc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, transform_filter_desc_);
        }
    }

    void propagate_strides(int *strides, const int *dims,
            std::initializer_list<int> perm) const {
        int prev_p = -1;
        for (auto p : perm) {
            strides[p] = prev_p == -1 ? 1 : strides[prev_p] * dims[prev_p];
            prev_p = p;
        }
    }

    virtual status_t init_filter_transformation(
            miopenDataType_t filter_data_types, int filter_ndims,
            int *filter_dims, int *current_filter_strides,
            int *transform_filter_strides) {
        // Set a descriptor for the current filter.
        CHECK(create_and_set_tensor_descriptor(&current_filter_desc_,
                filter_data_types, filter_ndims, filter_dims,
                current_filter_strides));

        // Set a descriptor for the transform filter.
        CHECK(create_and_set_tensor_descriptor(&transform_filter_desc_,
                filter_data_types, filter_ndims, filter_dims,
                transform_filter_strides));
        return status::success;
    }

    virtual void set_filter_nchw(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4: // Convert to KCRS
                return propagate_strides(
                        transform_filter_strides, filter_dims, {3, 2, 1, 0});
            case 5:
                return propagate_strides(
                        transform_filter_strides, filter_dims, {4, 3, 2, 1, 0});
            case 6:
                return propagate_strides(transform_filter_strides, filter_dims,
                        {5, 4, 3, 2, 1, 0});
        }
    }
    virtual void set_filter_nhwc(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4: // Convert to krsc
                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 3, 2, 0});
            case 5:
                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 4, 3, 2, 0});
            case 6:
                return propagate_strides(transform_filter_strides, filter_dims,
                        {1, 5, 4, 3, 2, 0});
        }
    }

    void set_filter_format(int filter_ndims, int *filter_dims,
            int *transform_filter_strides, miopenTensorLayout_t format) {
        if (format == miopenTensorNCHW) {
            set_filter_nchw(
                    filter_ndims, transform_filter_strides, filter_dims);
        } else {
            set_filter_nhwc(
                    filter_ndims, transform_filter_strides, filter_dims);
        }
    }
    void transform_filter(miopenHandle_t handle, void *current_filter,
            void *transform_filter) const {
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &filter_alpha_,
                current_filter_desc_, current_filter, &filter_beta_,
                transform_filter_desc_, transform_filter);
    }
    void undo_transform_filter(miopenHandle_t handle, void *transform_filter,
            void *current_filter) const {
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &filter_alpha_,
                transform_filter_desc_, transform_filter, &filter_beta_,
                current_filter_desc_, current_filter);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
