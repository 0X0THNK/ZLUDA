use cuda_types::cuda::{CUdevice_attribute, CUerror, CUresult};
use hip_runtime_sys::{
    hipDevice_t, hipDeviceptr_t, hipErrorCode_t, hipError_t, hipEvent_t, hipFunction_t,
    hipMemoryType, hipStreamCaptureMode, hipStreamCaptureStatus, hipStream_t, hip_Memcpy2D,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

const DEFAULT_RAM_BUDGET: usize = 64 * 1024 * 1024 * 1024;
const BASE_VPTR: usize = 0x1000_0000;
const VPTR_ALIGN: usize = 256;
const STREAM_ALLOWED_FLAGS: u32 = 0x1;
const EVENT_ALLOWED_FLAGS: u32 = 0x1 | 0x2 | 0x4;
const EVENT_DISABLE_TIMING_FLAG: u32 = 0x2;
const EVENT_INTERPROCESS_FLAG: u32 = 0x4;
const STREAM_WAIT_EVENT_ALLOWED_FLAGS: u32 = 0;

#[derive(Clone, Copy)]
struct VirtualEvent {
    flags: u32,
    recorded: bool,
}

#[derive(Clone)]
struct Allocation {
    base: usize,
    size: usize,
    data: Vec<u8>,
}

struct VirtualGpuState {
    allocations: HashMap<usize, Allocation>,
    total_allocated: usize,
    ram_budget: usize,
    function_names: HashMap<usize, String>,
    next_vptr: usize,
    streams: HashSet<usize>,
    events: HashMap<usize, VirtualEvent>,
    next_stream: usize,
    next_event: usize,
    stream_capture_ids: HashMap<usize, u64>,
    next_capture_id: u64,
}

impl VirtualGpuState {
    fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            ram_budget: std::env::var("ZLUDA_VIRTUAL_GPU_RAM")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(DEFAULT_RAM_BUDGET),
            function_names: HashMap::new(),
            next_vptr: BASE_VPTR,
            streams: HashSet::new(),
            events: HashMap::new(),
            next_stream: 0x5000_0000,
            next_event: 0x6000_0000,
            stream_capture_ids: HashMap::new(),
            next_capture_id: 1,
        }
    }

    fn alloc_vptr(&mut self, size: usize) -> Option<usize> {
        let aligned = (self.next_vptr + (VPTR_ALIGN - 1)) & !(VPTR_ALIGN - 1);
        let end = aligned.checked_add(size.max(VPTR_ALIGN))?;
        self.next_vptr = end;
        Some(aligned)
    }

    fn resolve_mut(&mut self, ptr: usize) -> Option<(&mut Allocation, usize)> {
        for alloc in self.allocations.values_mut() {
            if ptr >= alloc.base && ptr < alloc.base + alloc.size {
                return Some((alloc, ptr - alloc.base));
            }
        }
        None
    }

    fn resolve(&self, ptr: usize) -> Option<(&Allocation, usize)> {
        for alloc in self.allocations.values() {
            if ptr >= alloc.base && ptr < alloc.base + alloc.size {
                return Some((alloc, ptr - alloc.base));
            }
        }
        None
    }

    fn validate_range(alloc_size: usize, off: usize, byte_count: usize) -> bool {
        off.checked_add(byte_count)
            .map(|x| x <= alloc_size)
            .unwrap_or(false)
    }
}

fn state() -> &'static Mutex<VirtualGpuState> {
    static STATE: OnceLock<Mutex<VirtualGpuState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(VirtualGpuState::new()))
}

fn is_valid_capture_mode(mode: hipStreamCaptureMode) -> bool {
    mode == hipStreamCaptureMode::hipStreamCaptureModeGlobal
        || mode == hipStreamCaptureMode::hipStreamCaptureModeThreadLocal
        || mode == hipStreamCaptureMode::hipStreamCaptureModeRelaxed
}

fn stream_is_default(stream: hipStream_t) -> bool {
    stream.0.is_null()
}

fn stream_exists(s: &VirtualGpuState, stream: hipStream_t) -> bool {
    stream_is_default(stream) || s.streams.contains(&(stream.0 as usize))
}

fn is_user_stream(s: &VirtualGpuState, stream: hipStream_t) -> bool {
    !stream_is_default(stream) && s.streams.contains(&(stream.0 as usize))
}

fn event_exists(s: &VirtualGpuState, event: hipEvent_t) -> bool {
    s.events.contains_key(&(event.0 as usize))
}

pub(crate) fn enabled() -> bool {
    matches!(
        std::env::var("ZLUDA_VIRTUAL_GPU").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

pub(crate) fn alloc(dptr: &mut hipDeviceptr_t, bytesize: usize) -> CUresult {
    let mut s = state().lock().map_err(|_| CUerror::UNKNOWN)?;
    if s.total_allocated + bytesize > s.ram_budget {
        return Err(CUerror::OUT_OF_MEMORY);
    }
    let base = s.alloc_vptr(bytesize).ok_or(CUerror::OUT_OF_MEMORY)?;
    s.allocations.insert(
        base,
        Allocation {
            base,
            size: bytesize,
            data: vec![0u8; bytesize],
        },
    );
    s.total_allocated += bytesize;
    dptr.0 = base as *mut c_void;
    Ok(())
}

pub(crate) fn free(dptr: hipDeviceptr_t) -> CUresult {
    let mut s = state().lock().map_err(|_| CUerror::UNKNOWN)?;
    let base = dptr.0 as usize;
    if let Some(alloc) = s.allocations.remove(&base) {
        s.total_allocated = s.total_allocated.saturating_sub(alloc.size);
        Ok(())
    } else {
        Err(CUerror::INVALID_VALUE)
    }
}

pub(crate) fn memcpy_hto_d(
    dst_device: hipDeviceptr_t,
    src_host: *const c_void,
    byte_count: usize,
) -> hipError_t {
    if src_host.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (dst, off) = s
        .resolve_mut(dst_device.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    if !VirtualGpuState::validate_range(dst.size, off, byte_count) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_host as *const u8,
            dst.data.as_mut_ptr().add(off),
            byte_count,
        );
    }
    Ok(())
}

pub(crate) fn memcpy_dto_h(
    dst_host: *mut c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    if dst_host.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (src, off) = s
        .resolve(src_device.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    if !VirtualGpuState::validate_range(src.size, off, byte_count) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src.data.as_ptr().add(off), dst_host as *mut u8, byte_count);
    }
    Ok(())
}

pub(crate) fn memcpy_dto_d(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (src_data, src_off) = {
        let (src, off) = s
            .resolve(src_device.0 as usize)
            .ok_or(hipErrorCode_t::InvalidValue)?;
        if !VirtualGpuState::validate_range(src.size, off, byte_count) {
            return Err(hipErrorCode_t::InvalidValue);
        }
        (src.data.clone(), off)
    };
    let (dst, dst_off) = s
        .resolve_mut(dst_device.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    if !VirtualGpuState::validate_range(dst.size, dst_off, byte_count) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    dst.data[dst_off..dst_off + byte_count]
        .copy_from_slice(&src_data[src_off..src_off + byte_count]);
    Ok(())
}

pub(crate) fn get_address_range(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    if pbase.is_null() || psize.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (alloc, _) = s
        .resolve(dptr.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    unsafe {
        *pbase = hipDeviceptr_t(alloc.base as *mut c_void);
        *psize = alloc.size;
    }
    Ok(())
}

pub(crate) fn memset_d8(dst: hipDeviceptr_t, value: u8, n: usize) -> hipError_t {
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (alloc, off) = s
        .resolve_mut(dst.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    if !VirtualGpuState::validate_range(alloc.size, off, n) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    alloc.data[off..off + n].fill(value);
    Ok(())
}

pub(crate) fn memset_d16(dst: hipDeviceptr_t, value: u16, n: usize) -> hipError_t {
    let byte_count = n.checked_mul(2).ok_or(hipErrorCode_t::InvalidValue)?;
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (alloc, off) = s
        .resolve_mut(dst.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    if !VirtualGpuState::validate_range(alloc.size, off, byte_count) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let v = value.to_ne_bytes();
    for i in 0..n {
        let idx = off + i * 2;
        alloc.data[idx..idx + 2].copy_from_slice(&v);
    }
    Ok(())
}

pub(crate) fn memset_d32(dst: hipDeviceptr_t, value: u32, n: usize) -> hipError_t {
    let byte_count = n.checked_mul(4).ok_or(hipErrorCode_t::InvalidValue)?;
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (alloc, off) = s
        .resolve_mut(dst.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;
    if !VirtualGpuState::validate_range(alloc.size, off, byte_count) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let v = value.to_ne_bytes();
    for i in 0..n {
        let idx = off + i * 4;
        alloc.data[idx..idx + 4].copy_from_slice(&v);
    }
    Ok(())
}

pub(crate) fn memset2d_d32(
    dst_device: hipDeviceptr_t,
    dst_pitch: usize,
    value: u32,
    width: usize,
    height: usize,
) -> hipError_t {
    let bytes_per_row = width.checked_mul(4).ok_or(hipErrorCode_t::InvalidValue)?;
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let (alloc, off) = s
        .resolve_mut(dst_device.0 as usize)
        .ok_or(hipErrorCode_t::InvalidValue)?;

    let total_span = if height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(dst_pitch)
            .and_then(|x| x.checked_add(bytes_per_row))
            .ok_or(hipErrorCode_t::InvalidValue)?
    };
    if !VirtualGpuState::validate_range(alloc.size, off, total_span) {
        return Err(hipErrorCode_t::InvalidValue);
    }

    let v = value.to_ne_bytes();
    for row in 0..height {
        let row_start = off + row * dst_pitch;
        for col in 0..width {
            let idx = row_start + col * 4;
            alloc.data[idx..idx + 4].copy_from_slice(&v);
        }
    }
    Ok(())
}
pub(crate) fn alloc_pitch(
    dptr: *mut hipDeviceptr_t,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: u32,
) -> CUresult {
    if dptr.is_null() || p_pitch.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let align = usize::max(element_size_bytes as usize, 16);
    let pitch = (width_in_bytes + (align - 1)) & !(align - 1);
    let size = pitch.checked_mul(height).ok_or(CUerror::OUT_OF_MEMORY)?;
    let mut out = hipDeviceptr_t(std::ptr::null_mut());
    alloc(&mut out, size)?;
    unsafe {
        *dptr = out;
        *p_pitch = pitch;
    }
    Ok(())
}

pub(crate) fn copy_2d(memcpy: hip_Memcpy2D) -> CUresult {
    let width = memcpy.WidthInBytes;
    let height = memcpy.Height;

    for row in 0..height {
        let src_off = memcpy
            .srcY
            .checked_add(row)
            .and_then(|y| y.checked_mul(memcpy.srcPitch))
            .and_then(|base| base.checked_add(memcpy.srcXInBytes))
            .ok_or(CUerror::INVALID_VALUE)?;
        let dst_off = memcpy
            .dstY
            .checked_add(row)
            .and_then(|y| y.checked_mul(memcpy.dstPitch))
            .and_then(|base| base.checked_add(memcpy.dstXInBytes))
            .ok_or(CUerror::INVALID_VALUE)?;

        let src_ptr = match memcpy.srcMemoryType {
            t if t == hipMemoryType::hipMemoryTypeHost => unsafe {
                (memcpy.srcHost as *const u8).add(src_off) as *const c_void
            },
            t if t == hipMemoryType::hipMemoryTypeDevice => {
                (memcpy.srcDevice.0 as usize + src_off) as *const c_void
            }
            _ => return Err(CUerror::INVALID_VALUE),
        };

        let dst_ptr = match memcpy.dstMemoryType {
            t if t == hipMemoryType::hipMemoryTypeHost => unsafe {
                (memcpy.dstHost as *mut u8).add(dst_off) as *mut c_void
            },
            t if t == hipMemoryType::hipMemoryTypeDevice => {
                (memcpy.dstDevice.0 as usize + dst_off) as *mut c_void
            }
            _ => return Err(CUerror::INVALID_VALUE),
        };

        match (memcpy.srcMemoryType, memcpy.dstMemoryType) {
            (s, d)
                if s == hipMemoryType::hipMemoryTypeHost
                    && d == hipMemoryType::hipMemoryTypeDevice =>
            {
                memcpy_hto_d(hipDeviceptr_t(dst_ptr), src_ptr, width)?;
            }
            (s, d)
                if s == hipMemoryType::hipMemoryTypeDevice
                    && d == hipMemoryType::hipMemoryTypeHost =>
            {
                memcpy_dto_h(dst_ptr, hipDeviceptr_t(src_ptr as *mut c_void), width)?;
            }
            (s, d)
                if s == hipMemoryType::hipMemoryTypeDevice
                    && d == hipMemoryType::hipMemoryTypeDevice =>
            {
                memcpy_dto_d(
                    hipDeviceptr_t(dst_ptr),
                    hipDeviceptr_t(src_ptr as *mut c_void),
                    width,
                )?;
            }
            _ => {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr as *const u8, dst_ptr as *mut u8, width)
                };
            }
        }
    }

    Ok(())
}
pub(crate) fn mem_get_info(free: *mut usize, total: *mut usize) -> hipError_t {
    if free.is_null() || total.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    unsafe {
        *total = s.ram_budget;
        *free = s.ram_budget.saturating_sub(s.total_allocated);
    }
    Ok(())
}

pub(crate) fn total_mem(bytes: *mut usize) -> hipError_t {
    if bytes.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    unsafe { *bytes = s.ram_budget };
    Ok(())
}

pub(crate) fn stream_create(stream: *mut hipStream_t, flags: u32) -> hipError_t {
    if stream.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    if flags & !STREAM_ALLOWED_FLAGS != 0 {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let token = s.next_stream;
    s.next_stream = s.next_stream.saturating_add(VPTR_ALIGN);
    s.streams.insert(token);
    unsafe {
        *stream = hipStream_t(token as *mut c_void);
    }
    Ok(())
}

pub(crate) fn stream_create_with_priority(
    stream: *mut hipStream_t,
    flags: u32,
    _priority: i32,
) -> hipError_t {
    stream_create(stream, flags)
}

pub(crate) fn stream_destroy(stream: hipStream_t) -> hipError_t {
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if stream_is_default(stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    if s.streams.remove(&(stream.0 as usize)) {
        s.stream_capture_ids.remove(&(stream.0 as usize));
        Ok(())
    } else {
        Err(hipErrorCode_t::InvalidResourceHandle)
    }
}

pub(crate) fn stream_synchronize(stream: hipStream_t) -> hipError_t {
    if stream_is_default(stream) {
        return Ok(());
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if is_user_stream(&s, stream) {
        Ok(())
    } else {
        Err(hipErrorCode_t::InvalidResourceHandle)
    }
}

pub(crate) fn stream_begin_capture(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t {
    if !is_valid_capture_mode(mode) {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if !is_user_stream(&s, stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    if s.stream_capture_ids.contains_key(&(stream.0 as usize)) {
        return Err(hipErrorCode_t::IllegalState);
    }
    let id = s.next_capture_id;
    s.next_capture_id = s.next_capture_id.saturating_add(1);
    s.stream_capture_ids.insert(stream.0 as usize, id);
    Ok(())
}

pub(crate) fn stream_end_capture(
    stream: hipStream_t,
    graph: *mut hip_runtime_sys::hipGraph_t,
) -> hipError_t {
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if !is_user_stream(&s, stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    if s.stream_capture_ids.remove(&(stream.0 as usize)).is_none() {
        return Err(hipErrorCode_t::IllegalState);
    }
    if !graph.is_null() {
        unsafe { *graph = hip_runtime_sys::hipGraph_t(std::ptr::null_mut()) };
    }
    Ok(())
}

pub(crate) fn stream_is_capturing(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
) -> hipError_t {
    if capture_status.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if !is_user_stream(&s, stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    unsafe {
        *capture_status = if s.stream_capture_ids.contains_key(&(stream.0 as usize)) {
            hipStreamCaptureStatus::hipStreamCaptureStatusActive
        } else {
            hipStreamCaptureStatus::hipStreamCaptureStatusNone
        };
    }
    Ok(())
}

pub(crate) fn stream_get_capture_info(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hip_runtime_sys::hipGraph_t,
    dependencies_out: *mut *const hip_runtime_sys::hipGraphNode_t,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    if capture_status.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if !is_user_stream(&s, stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    let capture_id = s.stream_capture_ids.get(&(stream.0 as usize)).copied();
    unsafe {
        if !capture_status.is_null() {
            *capture_status = if capture_id.is_some() {
                hipStreamCaptureStatus::hipStreamCaptureStatusActive
            } else {
                hipStreamCaptureStatus::hipStreamCaptureStatusNone
            };
        }
        if !id.is_null() {
            *id = capture_id.unwrap_or(0);
        }
        if !graph_out.is_null() {
            *graph_out = hip_runtime_sys::hipGraph_t(std::ptr::null_mut());
        }
        if !dependencies_out.is_null() {
            *dependencies_out = std::ptr::null();
        }
        if !num_dependencies_out.is_null() {
            *num_dependencies_out = 0;
        }
    }
    Ok(())
}

pub(crate) fn stream_wait_event(stream: hipStream_t, event: hipEvent_t, flags: u32) -> hipError_t {
    if flags != STREAM_WAIT_EVENT_ALLOWED_FLAGS {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if !stream_exists(&s, stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    let Some(wait_event) = s.events.get(&(event.0 as usize)) else {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    };
    if !wait_event.recorded {
        return Err(hipErrorCode_t::NotReady);
    }
    Ok(())
}

pub(crate) fn event_create(event: *mut hipEvent_t, flags: u32) -> hipError_t {
    if event.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    if flags & !EVENT_ALLOWED_FLAGS != 0 {
        return Err(hipErrorCode_t::InvalidValue);
    }
    if (flags & EVENT_INTERPROCESS_FLAG) != 0 && (flags & EVENT_DISABLE_TIMING_FLAG) == 0 {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let token = s.next_event;
    s.next_event = s.next_event.saturating_add(VPTR_ALIGN);
    s.events.insert(
        token,
        VirtualEvent {
            flags,
            recorded: false,
        },
    );
    unsafe {
        *event = hipEvent_t(token as *mut c_void);
    }
    Ok(())
}

pub(crate) fn event_destroy(event: hipEvent_t) -> hipError_t {
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if s.events.remove(&(event.0 as usize)) {
        Ok(())
    } else {
        Err(hipErrorCode_t::InvalidResourceHandle)
    }
}

pub(crate) fn event_record(event: hipEvent_t, stream: hipStream_t) -> hipError_t {
    let mut s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if !stream_exists(&s, stream) {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    let Some(virtual_event) = s.events.get_mut(&(event.0 as usize)) else {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    };
    virtual_event.recorded = true;
    Ok(())
}

pub(crate) fn event_synchronize(event: hipEvent_t) -> hipError_t {
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    if event_exists(&s, event) {
        Ok(())
    } else {
        Err(hipErrorCode_t::InvalidResourceHandle)
    }
}

pub(crate) fn event_query(event: hipEvent_t) -> hipError_t {
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let Some(virtual_event) = s.events.get(&(event.0 as usize)) else {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    };
    if virtual_event.recorded {
        Ok(())
    } else {
        Err(hipErrorCode_t::NotReady)
    }
}

pub(crate) fn event_elapsed_time(
    milliseconds: *mut f32,
    start: hipEvent_t,
    end: hipEvent_t,
) -> hipError_t {
    if milliseconds.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let s = state().lock().map_err(|_| hipErrorCode_t::InvalidValue)?;
    let Some(start_event) = s.events.get(&(start.0 as usize)) else {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    };
    let Some(end_event) = s.events.get(&(end.0 as usize)) else {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    };
    if (start_event.flags & EVENT_DISABLE_TIMING_FLAG) != 0
        || (end_event.flags & EVENT_DISABLE_TIMING_FLAG) != 0
    {
        return Err(hipErrorCode_t::InvalidResourceHandle);
    }
    if !start_event.recorded || !end_event.recorded {
        return Err(hipErrorCode_t::NotReady);
    }
    unsafe { *milliseconds = 0.0 };
    Ok(())
}
pub(crate) fn get_count(count: &mut i32) {
    *count = 1;
}

pub(crate) fn get_attribute(pi: &mut i32, attrib: CUdevice_attribute) -> bool {
    let s = match state().lock() {
        Ok(v) => v,
        Err(_) => return false,
    };
    match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => {
            *pi = std::thread::available_parallelism()
                .map(|x| x.get() as i32)
                .unwrap_or(8)
                .max(8);
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => {
            *pi = 32;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => {
            *pi = 99 * 1024;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => {
            *pi = 65536;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => {
            *pi = 1024;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => {
            *pi = 2048;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
            *pi = 8;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
            *pi = 0;
            true
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => {
            *pi = (s.ram_budget.min(i32::MAX as usize)) as i32;
            true
        }
        _ => false,
    }
}

pub(crate) fn register_function_name(function: hipFunction_t, name: &str) {
    if let Ok(mut s) = state().lock() {
        s.function_names
            .insert(function.0 as usize, name.to_owned());
    }
}

fn bytes_to_f32(data: &[u8], idx: usize) -> Option<f32> {
    let start = idx.checked_mul(4)?;
    let end = start.checked_add(4)?;
    let bytes: [u8; 4] = data.get(start..end)?.try_into().ok()?;
    Some(f32::from_ne_bytes(bytes))
}

fn write_f32(data: &mut [u8], idx: usize, value: f32) -> bool {
    let start = match idx.checked_mul(4) {
        Some(v) => v,
        None => return false,
    };
    let end = match start.checked_add(4) {
        Some(v) => v,
        None => return false,
    };
    if let Some(dst) = data.get_mut(start..end) {
        dst.copy_from_slice(&value.to_ne_bytes());
        true
    } else {
        false
    }
}

pub(crate) fn launch_known_kernel(
    f: hipFunction_t,
    kernel_params: *mut *mut c_void,
) -> Option<hipError_t> {
    let mut s = state().lock().ok()?;
    let name = s.function_names.get(&(f.0 as usize))?.clone();
    if kernel_params.is_null() {
        return Some(Err(hipErrorCode_t::InvalidValue));
    }

    unsafe {
        if name == "vector_add" {
            let a_ptr = *(*(kernel_params.add(0)) as *mut hipDeviceptr_t);
            let b_ptr = *(*(kernel_params.add(1)) as *mut hipDeviceptr_t);
            let c_ptr = *(*(kernel_params.add(2)) as *mut hipDeviceptr_t);
            let n = *(*(kernel_params.add(3)) as *mut i32);
            let n_usize = n.max(0) as usize;

            let a_data = s.resolve(a_ptr.0 as usize)?.0.data.clone();
            let b_data = s.resolve(b_ptr.0 as usize)?.0.data.clone();
            let (c_alloc, _) = s.resolve_mut(c_ptr.0 as usize)?;

            for i in 0..n_usize {
                let a = match bytes_to_f32(&a_data, i) {
                    Some(v) => v,
                    None => return Some(Err(hipErrorCode_t::InvalidValue)),
                };
                let b = match bytes_to_f32(&b_data, i) {
                    Some(v) => v,
                    None => return Some(Err(hipErrorCode_t::InvalidValue)),
                };
                if !write_f32(&mut c_alloc.data, i, a + b) {
                    return Some(Err(hipErrorCode_t::InvalidValue));
                }
            }
            return Some(Ok(()));
        }

        if name == "matrix_mul" {
            let a_ptr = *(*(kernel_params.add(0)) as *mut hipDeviceptr_t);
            let b_ptr = *(*(kernel_params.add(1)) as *mut hipDeviceptr_t);
            let c_ptr = *(*(kernel_params.add(2)) as *mut hipDeviceptr_t);
            let n = *(*(kernel_params.add(3)) as *mut i32);
            let n_usize = n.max(0) as usize;

            let a_data = s.resolve(a_ptr.0 as usize)?.0.data.clone();
            let b_data = s.resolve(b_ptr.0 as usize)?.0.data.clone();
            let (c_alloc, _) = s.resolve_mut(c_ptr.0 as usize)?;

            for row in 0..n_usize {
                for col in 0..n_usize {
                    let mut sum = 0.0f32;
                    for k in 0..n_usize {
                        let a = bytes_to_f32(&a_data, row * n_usize + k)
                            .ok_or(hipErrorCode_t::InvalidValue)
                            .ok()?;
                        let b = bytes_to_f32(&b_data, k * n_usize + col)
                            .ok_or(hipErrorCode_t::InvalidValue)
                            .ok()?;
                        sum += a * b;
                    }
                    if !write_f32(&mut c_alloc.data, row * n_usize + col, sum) {
                        return Some(Err(hipErrorCode_t::InvalidValue));
                    }
                }
            }
            return Some(Ok(()));
        }
    }

    None
}

pub(crate) fn virtual_device_for_ordinal(ordinal: i32) -> Option<hipDevice_t> {
    if ordinal == 0 {
        Some(hipDevice_t(0))
    } else {
        None
    }
}
