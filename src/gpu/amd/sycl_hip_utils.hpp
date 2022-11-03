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

#ifndef GPU_AMD_SYCL_HIP_UTILS_HPP
#define GPU_AMD_SYCL_HIP_UTILS_HPP

#include <stdexcept>
#include "miopen/miopen.h"
#include <hip/hip_runtime.h>

#include "dnnl_sycl.hpp"

#include "common/engine.hpp"
#include "common/z_magic.hpp"

#include "sycl/sycl_utils.hpp"

#include "gpu/amd/sycl_hip_compat.hpp"

#define MIOPEN_DIM_MAX 5

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

bool compare_hip_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);

// Check if the device type matches the passed engine kind
inline status_t check_device(dnnl::impl::engine_kind_t eng_kind) {
    return (eng_kind == dnnl::impl::engine_kind::gpu
                    ? status::success
                    : status::invalid_arguments);
}

inline void convert_dnnl_dims_array(
        const dnnl_dim_t *dims, int *new_dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        new_dims[i] = static_cast<int>(dims[i]);
    }
}

inline void convert_dims(const dnnl_dim_t *dims, int *new_dims, int n_dims,
        int adjustment_size = 4, int adjustment_value = 1) {
    convert_dnnl_dims_array(dims, new_dims, n_dims);
    for (size_t i = n_dims; i < adjustment_size; i++) {
        new_dims[i] = adjustment_value;
    }
}

// Check if the dimensions contain any zeros, returns true if they do.
inline bool has_zero_dims(const dnnl_dim_t *dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        if (dims[i] == 0) { return true; }
    }
    return false;
}

static bool has_different_block_size(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    return ((src_md->format_desc.blocking.inner_nblks > 0
                    && dst_md->format_desc.blocking.inner_nblks == 0)
            || (src_md->format_desc.blocking.inner_nblks == 0
                    && dst_md->format_desc.blocking.inner_nblks > 0));
}

inline status_t convert_data_type(const memory_desc_t *mem_desc,
        miopenDataType_t *miopen_data_type, bool vectorized = true) {
    switch (mem_desc->data_type) {
        case data_type_t::dnnl_f16:
            *miopen_data_type = miopenDataType_t::miopenHalf;
            break;
        case data_type_t::dnnl_f32:
            *miopen_data_type = miopenDataType_t::miopenFloat;
            break;

        case data_type_t::dnnl_s8:
            *miopen_data_type
                    = ((vectorized
                               && mem_desc->format_desc.blocking.inner_blks[0]
                                       == 4)
                                    ? miopenDataType_t::miopenInt8x4
                                    : miopenDataType_t::miopenInt8);
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

class hip_error : virtual public std::runtime_error {

protected:
    inline const char *hip_error_map(hipError_t result) {
        switch (result) {
            case hipSuccess: return "hipSuccess";
            case hipErrorNotSupported: return "hipErrorNotSupported";
            case hipErrorInvalidContext: return "hipErrorInvalidContext";
            case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
            case hipErrorInvalidValue: return "hipErrorInvalidValue";
            case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
            case hipErrorLaunchOutOfResources:
                return "hipErrorLaunchOutOfResources";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit hip_error(const std::string &message, hipError_t result)
        : std::runtime_error((message + std::string(hip_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~hip_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

static status_t miopen_to_dnnl_status(miopenStatus_t miopen_status) {
    switch (miopen_status) {
        case miopenStatusSuccess: return status::success;
        case miopenStatusBadParm: return status::invalid_arguments;
        case miopenStatusNotImplemented: return status::unimplemented;
        default: return status::runtime_error;
    }
}

#define HIP_ERROR_LOCATION __FILE__ " : " STRINGIFY(__LINE__)

#define HIP_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != hipSuccess) { \
            throw hip_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define MIOPEN_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != miopenStatusSuccess) { \
            throw miopen_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define HIP_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != HIP_SUCCESS) { \
            std::cout << hip_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define MIOPEN_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != miopenStatusSuccess) { \
            std::cout << miopen_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define MIOPEN_CHECK_V(e) \
    { \
        auto status = (e); \
        if (status != miopenStatusSuccess) { \
            std::cout << miopen_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(" : "), \
                    status) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define MIOPEN_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        if (err != miopenStatusSuccess) { return miopen_to_dnnl_status(err); } \
        return status::success; \
    }()

inline status_t create_and_set_tensor_descriptor(
        miopenTensorDescriptor_t *tensor_desc, miopenDataType_t data_type,
        int ndims, int *dims, int *strides) {

    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, tensor_desc));

    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetTensorDescriptor, *tensor_desc,
            data_type, ndims, dims, strides));

    return status::success;
}

class miopen_error : virtual public std::runtime_error {

protected:
    inline const char *miopen_get_error_string(miopenStatus_t status) {
        switch (status) {
            case miopenStatusSuccess: return "miopenStatusSuccess";
            case miopenStatusNotInitialized:
                return "miopenStatusNotInitialized";
            case miopenStatusAllocFailed: return "miopenStatusAllocFailed";
            case miopenStatusBadParm: return "miopenStatusBadParm";
            case miopenStatusInternalError: return "miopenStatusInternalError";
            case miopenStatusInvalidValue: return "miopenStatusInvalidValue";
            case miopenStatusUnknownError: return "miopenStatusUnknownError";
            case miopenStatusNotImplemented:
                return "miopenStatusNotImplemented";

            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit miopen_error(const std::string &message, miopenStatus_t result)
        : std::runtime_error(
                (message + std::string(miopen_get_error_string(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~miopen_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

static bool memory_desc_matches_nchw_vect_c(const memory_desc_t *mem_desc) {
    // Only one block is supported for second (C) dimension and the block size
    // must be 4 and the dimension has to be a multiple of block size.
    auto is_int_8 = utils::one_of(mem_desc->data_type, data_type::s8);
    auto &strides = mem_desc->format_desc.blocking.strides;
    if (is_int_8 && mem_desc->format_desc.blocking.inner_nblks == 1
            && mem_desc->format_desc.blocking.inner_idxs[0] == 1
            && mem_desc->format_desc.blocking.inner_blks[0] == 4
            && mem_desc->dims[1] % 4 == 0) {
        for (int d = 0; d < mem_desc->ndims - 1; ++d)
            if (strides[d] < strides[d + 1]) return false;
        return true;
    }
    return false;
}

static bool memory_format_ok(const memory_desc_t *mem_desc) {
    return (memory_desc_matches_nchw_vect_c(mem_desc)
            || mem_desc->format_desc.blocking.inner_nblks == 0);
}

static bool adjust_dim_for_dnn(
        int *dims, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        dims[n_dims] = mem_desc->format_desc.blocking.inner_blks[0];
        dims[mem_desc->format_desc.blocking.inner_idxs[0]]
                /= mem_desc->format_desc.blocking.inner_blks[0];
        return true;
    }

    return false;
}

static bool adjust_stride_for_dnn(
        int *stride, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        stride[n_dims] = mem_desc->format_desc.blocking.inner_nblks;
        return true;
    }

    return false;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
