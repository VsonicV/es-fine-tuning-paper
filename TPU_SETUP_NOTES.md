# TPU vLLM Acceleration Setup Notes

## Overview

This document details the setup process, compatibility issues, and fixes applied while attempting to run the TPU acceleration script (`es_fine-tuning_countdown_accl_tpu_vllm_patched.py`) with vLLM 0.11.0 on TPU v6 lite.

## Environment

- Hardware: TPU v6 lite
- Python: 3.12
- Primary Libraries:
  - vLLM: 0.11.0
  - PyTorch: 2.8.0
  - torch_xla: 2.8.1
  - JAX: 0.7.2
  - jaxlib: 0.7.2
  - libtpu: 0.0.17

## Issues Encountered and Fixes Applied

### 1. PJRT API Version Mismatch

**Error:**
```
Unexpected PJRT_Plugin_Attributes_Args size: expected 32, got 24.
The plugin is likely built with a later version than the framework.
```

**Root Cause:**
- torch_xla 2.8.1 expects PJRT structure size 24
- libtpu 0.0.23 (initially installed) provides structure size 32
- Version mismatch between the PJRT API implementations

**Fix Applied:**
- Downgraded libtpu from 0.0.23 to 0.0.17
- Command: `pip install libtpu==0.0.17 --force-reinstall`
- Updated `requirement.txt` to pin `libtpu==0.0.17`

**Status:** ✅ Resolved

---

### 2. JAX Pallas TPUMemorySpace API Change

**Error:**
```
AttributeError: module 'jax.experimental.pallas.tpu' has no attribute 'TPUMemorySpace'.
Did you mean: 'MemorySpace'?
```

**Root Cause:**
- JAX renamed `pltpu.TPUMemorySpace` to `pltpu.MemorySpace` in newer versions
- vLLM 0.11.0's pallas backend code still uses the old API name
- The compatibility shim in the main script only patches the script itself, not vLLM's internal code

**Fix Applied:**
- Manually edited `/usr/local/lib/python3.12/dist-packages/vllm/attention/ops/pallas_kv_cache_update.py`
- Changed 3 occurrences in lines 92-97:
  ```python
  # BEFORE:
  in_specs = [
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
  ]
  out_specs = [pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)]

  # AFTER:
  in_specs = [
      pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
      pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
  ]
  out_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)]
  ```

**File Modified:**
- `/usr/local/lib/python3.12/dist-packages/vllm/attention/ops/pallas_kv_cache_update.py:92-97`

**Status:** ✅ Resolved

---

### 3. libtpu Version Age Requirement (UNRESOLVED)

**Error:**
```
RuntimeError: Pallas TPU requires a libtpu version that's at most a month old.
Found version string: Built on Jun 12 2025 16:39:57 (1749771597) cl/769337304
```

**Root Cause:**
- JAX Pallas requires libtpu versions built within the last month
- libtpu 0.0.17 (built June 12, 2025) is too old for JAX Pallas
- This creates a fundamental dependency conflict:
  - torch_xla 2.8.1 requires libtpu 0.0.17 for PJRT API compatibility
  - JAX Pallas requires libtpu < 1 month old (would need 0.0.23 or newer)

**Dependency Conflict Matrix:**

| Component | libtpu 0.0.17 | libtpu 0.0.23 |
|-----------|---------------|---------------|
| torch_xla 2.8.1 | ✅ PJRT compatible | ❌ PJRT size mismatch |
| JAX Pallas | ❌ Too old | ✅ Age requirement met |

**Status:** ❌ UNRESOLVED - Fundamental incompatibility

---

## Attempted Solutions

### Approach 1: Upgrade torch_xla
- **Attempted:** Upgrading to torch_xla > 2.8.1 to match newer libtpu
- **Result:** No compatible version found for torch 2.8.0

### Approach 2: Use Older JAX
- **Attempted:** Using older JAX version that doesn't check libtpu age
- **Result:** vLLM 0.11.0 requires jax==0.7.2 specifically

### Approach 3: Manual Patches
- **Applied:** Successfully patched MemorySpace API naming issue
- **Limitation:** Cannot resolve the libtpu age/PJRT compatibility conflict

---

## Current Status

### Working Components
- ✅ All base dependencies installed
- ✅ vLLM 0.11.0 installed
- ✅ JAX with TPU support installed
- ✅ MemorySpace API compatibility patched
- ✅ PJRT API version mismatch resolved (with libtpu 0.0.17)

### Blocking Issues
- ❌ Cannot run TPU inference due to libtpu version conflict
- ❌ No version combination satisfies both torch_xla and JAX Pallas requirements

---

## Recommendations

### Short-term Solutions

1. **Wait for vLLM Update**
   - Monitor vLLM releases for versions compatible with newer torch_xla
   - Check: https://github.com/vllm-project/vllm/releases

2. **Try Different vLLM Versions**
   - Test vLLM 0.10.x with the same dependency stack
   - Test vLLM 0.12.x (if available) with updated dependencies

3. **Contact vLLM Maintainers**
   - Report the torch_xla + JAX + libtpu incompatibility
   - Request guidance on compatible version combinations
   - Issue tracker: https://github.com/vllm-project/vllm/issues

### Long-term Solutions

1. **Use CPU/GPU for Development**
   - Continue ES fine-tuning development on CPU/GPU
   - Switch to TPU when compatibility is resolved

2. **Alternative TPU Frameworks**
   - Consider using JAX directly without vLLM
   - Investigate other TPU-compatible inference frameworks

3. **Monitor Dependency Updates**
   - Watch for torch_xla updates that support newer libtpu
   - Track JAX releases for PJRT API compatibility improvements

---

## Patches Applied

### 1. vLLM Pallas Backend Patch

**File:** `/usr/local/lib/python3.12/dist-packages/vllm/attention/ops/pallas_kv_cache_update.py`

**Changes:** Lines 92-97
```python
# Replace all instances of:
pltpu.TPUMemorySpace.ANY
# With:
pltpu.MemorySpace.ANY
```

**Reason:** JAX Pallas API naming change

### 2. Dependency Version Pins

**File:** `requirement.txt`

**Added/Updated:**
```
libtpu==0.0.17  # Downgraded from 0.0.23 for PJRT compatibility
```

---

## Testing Log

### Test 1: Initial Run
- **Date:** 2025-10-26
- **Result:** PJRT API version mismatch
- **Fix:** Downgraded libtpu to 0.0.17

### Test 2: After libtpu Downgrade
- **Date:** 2025-10-26
- **Result:** TPUMemorySpace AttributeError
- **Fix:** Patched vLLM pallas backend

### Test 3: After MemorySpace Patch
- **Date:** 2025-10-26
- **Result:** libtpu version too old for Pallas
- **Fix:** None available - fundamental incompatibility

---

## Version Compatibility Notes

### Known Working Combinations (Partial)
- torch 2.8.0 + torch_xla 2.8.1 + libtpu 0.0.17 → ✅ PJRT compatible
- jax 0.7.2 + jaxlib 0.7.2 + libtpu 0.0.23 → ✅ Pallas compatible

### Known Incompatible Combinations
- torch_xla 2.8.1 + libtpu 0.0.23 → ❌ PJRT size mismatch
- jax 0.7.2 + libtpu 0.0.17 → ❌ libtpu too old

---

## Additional Notes

### Warnings Observed
- "Transparent hugepages are not enabled" - Performance warning, not blocking
- "Failed to deserialize executable" - Expected on first compile, not an error
- "VLLM_ENABLE_V1_MULTIPROCESSING is set to False" - Expected for single-engine setup

### Successful Initialization Steps
1. TPU platform detection ✅
2. Ray distributed framework initialization ✅
3. Model loading (Qwen2.5-3B-Instruct) ✅
4. TPU worker initialization ✅
5. XLA compilation started ✅
6. **Failed at:** Pallas kernel compilation ❌

---

## Files Modified

1. `/root/es-fine-tuning-paper-jhfork1/requirement.txt`
   - Updated libtpu version to 0.0.17

2. `/usr/local/lib/python3.12/dist-packages/vllm/attention/ops/pallas_kv_cache_update.py`
   - Patched MemorySpace API calls

---

## Next Steps

1. Monitor vLLM repository for compatibility updates
2. Test with alternative vLLM versions
3. Consider fallback to CPU/GPU inference for development
4. Report issue to vLLM maintainers with detailed version info

---

## Contact & Support

- vLLM GitHub: https://github.com/vllm-project/vllm
- JAX GitHub: https://github.com/google/jax
- PyTorch XLA GitHub: https://github.com/pytorch/xla

---

**Last Updated:** 2025-10-26
**Environment:** TPU v6 lite on Cloud TPU
**Status:** Blocked by dependency incompatibility
