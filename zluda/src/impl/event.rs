use crate::r#impl::virtual_gpu;
use hip_runtime_sys::*;

pub(crate) unsafe fn create(event: *mut hipEvent_t, flags: ::core::ffi::c_uint) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::event_create(event, flags);
    }
    // Flag values are compatible between CUDA and HIP for 0,1,2,4
    hipEventCreateWithFlags(event, flags)
}

pub(crate) unsafe fn elapsed_time(
    milliseconds: *mut ::core::ffi::c_float,
    start: hipEvent_t,
    end: hipEvent_t,
) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::event_elapsed_time(milliseconds, start, end);
    }
    // Flag values are compatible between CUDA and HIP for 0,1,2,4
    hipEventElapsedTime(milliseconds, start, end)
}

pub(crate) unsafe fn elapsed_time_v2(
    milliseconds: *mut ::core::ffi::c_float,
    start: hipEvent_t,
    end: hipEvent_t,
) -> hipError_t {
    elapsed_time(milliseconds, start, end)
}

pub(crate) unsafe fn query(event: hipEvent_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::event_query(event);
    }
    hipEventQuery(event)
}

pub(crate) unsafe fn destroy_v2(event: hipEvent_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::event_destroy(event);
    }
    hipEventDestroy(event)
}

pub(crate) unsafe fn record(event: hipEvent_t, stream: hipStream_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::event_record(event, stream);
    }
    hipEventRecord(event, stream)
}

pub(crate) unsafe fn record_with_flags(
    event: hipEvent_t,
    stream: hipStream_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    if virtual_gpu::enabled() {
        if flags != hipEventRecordDefault {
            return hipError_t::ErrorInvalidValue;
        }
        return virtual_gpu::event_record(event, stream);
    }

    // Flag values are compatible between CUDA and HIP for 0,1

    // The ROCm 6.4.0 headers have a declaration for hipEventRecordWithFlags, but the library has
    // no implementation. The implementation was added in ROCm 6.4.2. We only support the default flag for now.
    if flags != hipEventRecordDefault {
        return hipError_t::ErrorInvalidValue;
    }

    hipEventRecord(event, stream)
}

pub(crate) unsafe fn synchronize(event: hipEvent_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::event_synchronize(event);
    }
    hipEventSynchronize(event)
}
