/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// gpu_native_resolves, multisampled source (native_resolve.cs.hlsl for a
// single-sampled one): writes a resolve rectangle from a 2x MSAA host render
// target straight into a texture cache texture. Each thread handles one host
// pixel (already scaled with resolution scaling). The pixel is one host sample
// of the render target, or the mean of two (the guest's k01 / k0123 average of
// a 2x MSAA source), then written like native_resolve.cs.hlsl's modes:
// 0 - out = in * xe_native_resolve_scale (exponent bias of float formats).
// 1 - out = in.bgra (red and blue swapped).
//
// xe_native_resolve_samples: bits 0-1 the number of samples (1 or 2), bits 4-5
// the first host sample index, bits 6-7 the second.

cbuffer XeNativeResolveConstants : register(b0) {
  uint xe_native_resolve_mode;
  float xe_native_resolve_scale;
  int2 xe_native_resolve_source_offset;
  int2 xe_native_resolve_dest_offset;
  uint2 xe_native_resolve_size;
  uint2 xe_native_resolve_fill;
  uint xe_native_resolve_samples;
};

Texture2DMS<float4> xe_native_resolve_source : register(t0);
RWTexture2D<float4> xe_native_resolve_dest : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  [branch] if (any(xe_thread_id.xy >= xe_native_resolve_size)) {
    return;
  }
  int2 pixel = int2(xe_thread_id.xy);
  int2 source_pixel = xe_native_resolve_source_offset +
                      int2(max(xe_thread_id.xy, xe_native_resolve_fill));
  float4 value = xe_native_resolve_source.Load(
      source_pixel, int((xe_native_resolve_samples >> 4u) & 3u));
  [branch] if ((xe_native_resolve_samples & 3u) >= 2u) {
    value = (value + xe_native_resolve_source.Load(
                         source_pixel, int((xe_native_resolve_samples >> 6u) & 3u))) *
            0.5;
  }
  [branch] if (xe_native_resolve_mode == 1u) {
    value = value.bgra;
  } else {
    value *= xe_native_resolve_scale;
  }
  xe_native_resolve_dest[uint2(xe_native_resolve_dest_offset + pixel)] = value;
}
