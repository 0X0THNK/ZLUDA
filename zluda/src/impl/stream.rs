use crate::r#impl::virtual_gpu;
use hip_runtime_sys::*;

pub(crate) fn synchronize(stream: hipStream_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_synchronize(stream);
    }
    unsafe { hipStreamSynchronize(stream) }
}

pub(crate) fn create(stream: *mut hipStream_t, flags: ::core::ffi::c_uint) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_create(stream, flags);
    }
    unsafe { hipStreamCreateWithFlags(stream, flags) }
}

pub(crate) fn create_with_priority(
    stream: *mut hipStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_create_with_priority(stream, flags, priority);
    }
    unsafe { hipStreamCreateWithPriority(stream, flags, priority) }
}

pub(crate) fn destroy_v2(stream: hipStream_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_destroy(stream);
    }
    unsafe { hipStreamDestroy(stream) }
}

pub(crate) fn begin_capture_v2(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_begin_capture(stream, mode);
    }
    unsafe { hipStreamBeginCapture(stream, mode) }
}

pub(crate) fn end_capture(stream: hipStream_t, graph: *mut hipGraph_t) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_end_capture(stream, graph);
    }
    unsafe { hipStreamEndCapture(stream, graph) }
}

pub(crate) fn is_capturing(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_is_capturing(stream, capture_status);
    }
    unsafe { hipStreamIsCapturing(stream, capture_status) }
}

pub(crate) fn get_capture_info_v2(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_get_capture_info(
            stream,
            capture_status,
            id,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        );
    }
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status,
            id,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

pub(crate) fn get_capture_info_v3(
    stream: hipStream_t,
    capture_status_out: *mut hipStreamCaptureStatus,
    id_out: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    edge_data_out: *mut *const cuda_types::cuda::CUgraphEdgeData,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    if !edge_data_out.is_null() {
        return hipError_t::ErrorNotSupported;
    }
    if virtual_gpu::enabled() {
        return get_capture_info_v2(
            stream,
            capture_status_out,
            id_out,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        );
    }
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status_out,
            id_out,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

pub(crate) fn wait_event(
    stream: hipStream_t,
    event: hipEvent_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    if virtual_gpu::enabled() {
        return virtual_gpu::stream_wait_event(stream, event, flags);
    }
    unsafe { hipStreamWaitEvent(stream, event, flags) }
}
