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
// the first host sample index, bits 6-7 the second, bits 8-9 how two samples
// are averaged: 0 - in float as the view returns them (float formats); 1 -
// k_8_8_8_8, 2 - k_2_10_10_10: as the resolve shaders do (each sample back to
// the integers the guest holds, XeUnpackR8G8B8A8UNorm / XeUnpackR10G10B10A2UNorm,
// the sum halved, XePackR8G8B8A8UNorm / XePackR10G10B10A2UNorm's rounding), so
// the texture gets the copy's values bit for bit (the plain float mean of the
// UNORM view rounds some halves the other way).

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
  uint unorm_average = 0u;
  float4 unorm_max = float4(255.0f, 255.0f, 255.0f, 255.0f);
  float4 unorm_max_rcp = float4(1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f);
  [branch] if ((xe_native_resolve_samples & 3u) >= 2u) {
    float4 value_1 = xe_native_resolve_source.Load(
        source_pixel, int((xe_native_resolve_samples >> 6u) & 3u));
    unorm_average = (xe_native_resolve_samples >> 8u) & 3u;
    [branch] if (unorm_average != 0u) {
      [flatten] if (unorm_average == 2u) {
        unorm_max = float4(1023.0f, 1023.0f, 1023.0f, 3.0f);
        unorm_max_rcp = float4(1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 3.0f);
      }
      value = round(value * unorm_max) * unorm_max_rcp;
      value_1 = round(value_1 * unorm_max) * unorm_max_rcp;
    }
    value = (value + value_1) * 0.5;
  }
  [branch] if (xe_native_resolve_mode == 1u) {
    value = value.bgra;
  } else {
    value *= xe_native_resolve_scale;
  }
  [branch] if (unorm_average != 0u) {
    // The integers XePack*UNorm writes; the UNORM store gives them back exactly.
    value = float4(uint4(saturate(value) * unorm_max + 0.5f)) * unorm_max_rcp;
  }
  xe_native_resolve_dest[uint2(xe_native_resolve_dest_offset + pixel)] = value;
}
