/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// gpu_native_resolves: writes a resolve rectangle from a host render target
// straight into a texture cache texture when the resolve is not a plain copy of
// the render target bits.
// Each thread handles one host pixel (already scaled with resolution scaling).
//
// Modes:
// 0 - out = in * xe_native_resolve_scale (exponent bias of float formats).
// 1 - out = in.bgra (red and blue swapped).
// 2 - out.r = depth (the source SRV returns the depth in the red component).

cbuffer XeNativeResolveConstants : register(b0) {
  uint xe_native_resolve_mode;
  float xe_native_resolve_scale;
  int2 xe_native_resolve_source_offset;
  int2 xe_native_resolve_dest_offset;
  uint2 xe_native_resolve_size;
};

Texture2D<float4> xe_native_resolve_source : register(t0);
RWTexture2D<float4> xe_native_resolve_dest : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  [branch] if (any(xe_thread_id.xy >= xe_native_resolve_size)) {
    return;
  }
  int2 pixel = int2(xe_thread_id.xy);
  float4 value = xe_native_resolve_source.Load(
      int3(xe_native_resolve_source_offset + pixel, 0));
  [branch] if (xe_native_resolve_mode == 1u) {
    value = value.bgra;
  } else if (xe_native_resolve_mode == 2u) {
    value = float4(value.r, 0.0, 0.0, 0.0);
  } else {
    value *= xe_native_resolve_scale;
  }
  xe_native_resolve_dest[uint2(xe_native_resolve_dest_offset + pixel)] = value;
}
