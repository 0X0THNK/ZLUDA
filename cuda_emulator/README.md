# CUDA Emulator (ZLUDA-oriented execution layer)

This directory provides a CUDA interception and virtualization layer oriented for integration with ZLUDA workflows.

## Implemented modules

- API Interceptor: runtime and driver entrypoints for memory, memcpy sync/async, module load/unload, function lookup, kernel launch, streams, events, and device attributes.
- Virtual VRAM Manager: host RAM-backed virtual allocations with pointer+offset mapping, bound checks, and virtual memory accounting.
- GPU Compute Emulator: CPU software execution for `vector_add` and chunked execution fallback for `matrix_mul`.
- Scheduler: async chunk dispatcher with round-robin simulated multi-GPU assignment.
- Device Properties Emulation: virtualized memory and CUDA properties.
- PTX Parser: `.entry` extraction used by module/function validation.
- Utils: logging and scoped timing.

## Build

```bash
cmake -S cuda_emulator -B build/cuda_emulator
cmake --build build/cuda_emulator
ctest --test-dir build/cuda_emulator --output-on-failure
```

## Notes

- Module loading validates PTX presence and resolves functions only if declared via `.entry`.
- Stream/event orchestration is host-scheduled with explicit synchronization primitives.
- Additional backend integration can route scheduler chunks to concrete ZLUDA execution backends.
