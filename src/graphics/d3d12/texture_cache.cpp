/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <xxhash.h>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/d3d12/shared_memory.h>
#include <rex/graphics/d3d12/texture_cache.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/texture_patch.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/perf/counter.h>
#include <rex/math.h>
#include <rex/ui/d3d12/d3d12_upload_buffer_pool.h>
#include <rex/ui/d3d12/d3d12_util.h>

REXCVAR_DEFINE_STRING(texture_dump_dir, "", "GPU/D3D12",
                      "Write each new 2D texture in DXT1/DXT3/DXT5 (BC1-BC3) the game loads, its "
                      "top level as a DDS named by the hash of its content, into this folder (once "
                      "each), to make replacements from; empty = off")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_BOOL(texture_replace_async, true, "GPU/D3D12",
                    "Load replacement textures on worker threads: the game's own texture shows until "
                    "the file is ready, instead of the game waiting for every file");

REXCVAR_DEFINE_STRING(texture_replace_dir, "", "GPU/D3D12",
                      "Load <hash>.dds from this folder in place of the game's texture of that "
                      "content (the names texture_dump_dir gives): the same block format, any size "
                      "(e.g. 4x), best with a full mip chain; empty = off")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

namespace rex::graphics::d3d12 {

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/native_resolve_cs.h"
#include "../shaders/bytecode/d3d12_5_1/native_resolve_ms_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_128bpb_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_128bpb_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_16bpb_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_16bpb_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_32bpb_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_32bpb_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_64bpb_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_64bpb_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_8bpb_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_8bpb_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_bgrg8_rgb8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_bgrg8_rgbg8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_ctx1_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_depth_float_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_depth_float_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_depth_unorm_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_depth_unorm_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxn_rg8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxt1_rgba8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxt3_rgba8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxt3a_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxt3aas1111_bgra4_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxt5_rgba8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_dxt5a_r8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_gbgr8_grgb8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_gbgr8_rgb8_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r10g11b11_rgba16_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r10g11b11_rgba16_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r10g11b11_rgba16_snorm_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r10g11b11_rgba16_snorm_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r11g11b10_rgba16_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r11g11b10_rgba16_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r11g11b10_rgba16_snorm_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r11g11b10_rgba16_snorm_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r4g4b4a4_b4g4r4a4_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r4g4b4a4_b4g4r4a4_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r5g5b5a1_b5g5r5a1_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r5g5b5a1_b5g5r5a1_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r5g6b5_b5g6r5_cs.h"
#include "../shaders/bytecode/d3d12_5_1/texture_load_r5g6b5_b5g6r5_scaled_cs.h"
}  // namespace shaders

const D3D12TextureCache::HostFormat D3D12TextureCache::host_formats_[64] = {
    // k_1_REVERSE
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_1
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8
    {DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_UNORM, kLoadShaderIndex8bpb, DXGI_FORMAT_R8_SNORM,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_1_5_5_5
    // Red and blue swapped in the load shader for simplicity.
    {DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_B5G5R5A1_UNORM, kLoadShaderIndexR5G5B5A1ToB5G5R5A1,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_5_6_5
    // Red and blue swapped in the load shader for simplicity.
    {DXGI_FORMAT_B5G6R5_UNORM, DXGI_FORMAT_B5G6R5_UNORM, kLoadShaderIndexR5G6B5ToB5G6R5,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_6_5_5
    // On the host, green bits in blue, blue bits in green.
    {DXGI_FORMAT_B5G6R5_UNORM, DXGI_FORMAT_B5G6R5_UNORM,
     kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, XE_GPU_MAKE_TEXTURE_SWIZZLE(R, B, G, G)},
    // k_8_8_8_8
    {DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndex32bpb,
     DXGI_FORMAT_R8G8B8A8_SNORM, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10
    {DXGI_FORMAT_R10G10B10A2_TYPELESS, DXGI_FORMAT_R10G10B10A2_UNORM, kLoadShaderIndex32bpb,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_8_A
    {DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_UNORM, kLoadShaderIndex8bpb, DXGI_FORMAT_R8_SNORM,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8_B
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8_8
    {DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UNORM, kLoadShaderIndex16bpb,
     DXGI_FORMAT_R8G8_SNORM, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_Cr_Y1_Cb_Y0_REP
    // Red and blue swapped in the load shader for simplicity.
    // TODO(Triang3l): The DXGI_FORMAT_R8G8B8A8_U/SNORM conversion is usable for
    // the signed version, separate unsigned and signed load shaders completely
    // (as one doesn't need decompression for this format, while another does).
    {DXGI_FORMAT_G8R8_G8B8_UNORM, DXGI_FORMAT_G8R8_G8B8_UNORM, kLoadShaderIndexGBGR8ToGRGB8,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM,
     kLoadShaderIndexGBGR8ToRGB8, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_Y1_Cr_Y0_Cb_REP
    // Red and blue swapped in the load shader for simplicity.
    // TODO(Triang3l): The DXGI_FORMAT_R8G8B8A8_U/SNORM conversion is usable for
    // the signed version, separate unsigned and signed load shaders completely
    // (as one doesn't need decompression for this format, while another does).
    {DXGI_FORMAT_R8G8_B8G8_UNORM, DXGI_FORMAT_R8G8_B8G8_UNORM, kLoadShaderIndexBGRG8ToRGBG8,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM,
     kLoadShaderIndexBGRG8ToRGB8, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_16_16_EDRAM
    // Not usable as a texture, also has -32...32 range.
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_8_8_8_A
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_4_4_4_4
    // Red and blue swapped in the load shader for simplicity.
    {DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_B4G4R4A4_UNORM, kLoadShaderIndexRGBA4ToBGRA4,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_10_11_11
    {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM,
     kLoadShaderIndexR11G11B10ToRGBA16, DXGI_FORMAT_R16G16B16A16_SNORM,
     kLoadShaderIndexR11G11B10ToRGBA16SNorm, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_11_11_10
    {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM,
     kLoadShaderIndexR10G11B11ToRGBA16, DXGI_FORMAT_R16G16B16A16_SNORM,
     kLoadShaderIndexR10G11B11ToRGBA16SNorm, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_DXT1
    {DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM, kLoadShaderIndex64bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndexDXT1ToRGBA8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT2_3
    {DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM, kLoadShaderIndex128bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndexDXT3ToRGBA8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT4_5
    {DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM, kLoadShaderIndex128bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndexDXT5ToRGBA8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_16_16_16_16_EDRAM
    // Not usable as a texture, also has -32...32 range.
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // R32_FLOAT for depth because shaders would require an additional SRV to
    // sample stencil, which we don't provide.
    // k_24_8
    {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT, kLoadShaderIndexDepthUnorm,
     DXGI_FORMAT_R32_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_24_8_FLOAT
    {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT, kLoadShaderIndexDepthFloat,
     DXGI_FORMAT_R32_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16
    {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, kLoadShaderIndex16bpb, DXGI_FORMAT_R16_SNORM,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16
    {DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_UNORM, kLoadShaderIndex32bpb,
     DXGI_FORMAT_R16G16_SNORM, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_16_16_16
    {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM, kLoadShaderIndex64bpb,
     DXGI_FORMAT_R16G16B16A16_SNORM, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_16_EXPAND
    {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_FLOAT, kLoadShaderIndex16bpb, DXGI_FORMAT_R16_FLOAT,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_EXPAND
    {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, kLoadShaderIndex32bpb,
     DXGI_FORMAT_R16G16_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_16_16_16_EXPAND
    {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, kLoadShaderIndex64bpb,
     DXGI_FORMAT_R16G16B16A16_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_16_FLOAT
    {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_FLOAT, kLoadShaderIndex16bpb, DXGI_FORMAT_R16_FLOAT,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_FLOAT
    {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, kLoadShaderIndex32bpb,
     DXGI_FORMAT_R16G16_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_16_16_16_FLOAT
    {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, kLoadShaderIndex64bpb,
     DXGI_FORMAT_R16G16B16A16_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_32
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_32
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_32_32_32_32
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_32_FLOAT
    // Typeless for gpu_lazy_resolve_copies (the rebuild reads the bits through an R32_UINT view).
    {DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT, kLoadShaderIndex32bpb, DXGI_FORMAT_R32_FLOAT,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_32_FLOAT
    {DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, kLoadShaderIndex64bpb,
     DXGI_FORMAT_R32G32_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_32_32_32_32_FLOAT
    {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, kLoadShaderIndex128bpb,
     DXGI_FORMAT_R32G32B32A32_FLOAT, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_32_AS_8
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_8
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_MPEG
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_MPEG
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_INTERLACED
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_INTERLACED
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_8_INTERLACED
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_INTERLACED
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_MPEG_INTERLACED
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_MPEG_INTERLACED
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_DXN
    {DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_UNORM, kLoadShaderIndex128bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8_UNORM, kLoadShaderIndexDXNToRG8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_8_8_8_AS_16_16_16_16
    {DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndex32bpb,
     DXGI_FORMAT_R8G8B8A8_SNORM, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT1_AS_16_16_16_16
    {DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM, kLoadShaderIndex64bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndexDXT1ToRGBA8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT2_3_AS_16_16_16_16
    {DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM, kLoadShaderIndex128bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndexDXT3ToRGBA8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT4_5_AS_16_16_16_16
    {DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM, kLoadShaderIndex128bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8G8B8A8_UNORM, kLoadShaderIndexDXT5ToRGBA8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10_AS_16_16_16_16
    {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, kLoadShaderIndex32bpb,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_10_11_11_AS_16_16_16_16
    {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM,
     kLoadShaderIndexR11G11B10ToRGBA16, DXGI_FORMAT_R16G16B16A16_SNORM,
     kLoadShaderIndexR11G11B10ToRGBA16SNorm, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_11_11_10_AS_16_16_16_16
    {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM,
     kLoadShaderIndexR10G11B11ToRGBA16, DXGI_FORMAT_R16G16B16A16_SNORM,
     kLoadShaderIndexR10G11B11ToRGBA16SNorm, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_32_32_32_FLOAT
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_DXT3A
    // R8_UNORM has the same size as BC2, but doesn't have the 4x4 size
    // alignment requirement.
    {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UNORM, kLoadShaderIndexDXT3A, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_DXT5A
    {DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_UNORM, kLoadShaderIndex64bpb, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, true, DXGI_FORMAT_R8_UNORM, kLoadShaderIndexDXT5AToR8,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_CTX1
    {DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UNORM, kLoadShaderIndexCTX1, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_DXT3A_AS_1_1_1_1
    {DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_B4G4R4A4_UNORM, kLoadShaderIndexDXT3AAs1111ToBGRA4,
     DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_8_8_8_8_GAMMA_EDRAM
    // Not usable as a texture.
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10_FLOAT_EDRAM
    // Not usable as a texture.
    {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown, DXGI_FORMAT_UNKNOWN,
     kLoadShaderIndexUnknown, false, DXGI_FORMAT_UNKNOWN, kLoadShaderIndexUnknown,
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
};

D3D12TextureCache::D3D12TextureCache(const RegisterFile& register_file,
                                     D3D12SharedMemory& shared_memory,
                                     uint32_t draw_resolution_scale_x,
                                     uint32_t draw_resolution_scale_y,
                                     D3D12CommandProcessor& command_processor,
                                     bool bindless_resources_used)
    : TextureCache(register_file, shared_memory, draw_resolution_scale_x, draw_resolution_scale_y),
      command_processor_(command_processor),
      bindless_resources_used_(bindless_resources_used) {}

D3D12TextureCache::~D3D12TextureCache() {
  shared_memory().SetBeforeRequestHook(nullptr, nullptr);
  StopReplacementWorkers();
  // While the texture descriptor cache still exists (referenced by
  // ~D3D12Texture), destroy all textures.
  DestroyAllTextures(true);

  // First release the buffers to detach them from the heaps.
  for (std::unique_ptr<ScaledResolveVirtualBuffer>& scaled_resolve_buffer_ptr :
       scaled_resolve_2gb_buffers_) {
    scaled_resolve_buffer_ptr.reset();
  }
  scaled_resolve_heaps_.clear();
  COUNT_profile_set("gpu/texture_cache/scaled_resolve_buffer_used_mb", 0);
}

bool D3D12TextureCache::Initialize() {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  if (IsDrawResolutionScaled()) {
    // Buffers not used yet - no need aliasing barriers to change ownership of
    // gigabytes between even and odd buffers.
    std::memset(scaled_resolve_1gb_buffer_indices_, UINT8_MAX,
                sizeof(scaled_resolve_1gb_buffer_indices_));
    assert_true(scaled_resolve_heaps_.empty());
    uint64_t scaled_resolve_address_space_size =
        uint64_t(SharedMemory::kBufferSize) *
        (draw_resolution_scale_x() * draw_resolution_scale_y());
    scaled_resolve_heaps_.resize(
        size_t(scaled_resolve_address_space_size >> kScaledResolveHeapSizeLog2));
  }
  scaled_resolve_heap_count_ = 0;

  // Create the loading root signature.
  D3D12_ROOT_PARAMETER root_parameters[3];
  // Parameter 0 is constants (changed multiple times when untiling).
  root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  root_parameters[0].Constants.ShaderRegister = 0;
  root_parameters[0].Constants.RegisterSpace = 0;
  root_parameters[0].Constants.Num32BitValues = sizeof(LoadConstants) / sizeof(uint32_t);
  root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  // Parameter 1 is the source (may be changed multiple times for the same
  // destination).
  D3D12_DESCRIPTOR_RANGE root_dest_range;
  root_dest_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  root_dest_range.NumDescriptors = 1;
  root_dest_range.BaseShaderRegister = 0;
  root_dest_range.RegisterSpace = 0;
  root_dest_range.OffsetInDescriptorsFromTableStart = 0;
  root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
  root_parameters[1].DescriptorTable.pDescriptorRanges = &root_dest_range;
  root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  // Parameter 2 is the destination.
  D3D12_DESCRIPTOR_RANGE root_source_range;
  root_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  root_source_range.NumDescriptors = 1;
  root_source_range.BaseShaderRegister = 0;
  root_source_range.RegisterSpace = 0;
  root_source_range.OffsetInDescriptorsFromTableStart = 0;
  root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
  root_parameters[2].DescriptorTable.pDescriptorRanges = &root_source_range;
  root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC root_signature_desc;
  root_signature_desc.NumParameters = UINT(rex::countof(root_parameters));
  root_signature_desc.pParameters = root_parameters;
  root_signature_desc.NumStaticSamplers = 0;
  root_signature_desc.pStaticSamplers = nullptr;
  root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(load_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, root_signature_desc);
  if (!load_root_signature_) {
    REXGPU_ERROR(
        "D3D12TextureCache: Failed to create the texture loading root "
        "signature");
    return false;
  }

  // Specify the load shader code.
  D3D12_SHADER_BYTECODE load_shader_code[kLoadShaderCount] = {};
  load_shader_code[kLoadShaderIndex8bpb] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_8bpb_cs, sizeof(shaders::texture_load_8bpb_cs)};
  load_shader_code[kLoadShaderIndex16bpb] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_16bpb_cs, sizeof(shaders::texture_load_16bpb_cs)};
  load_shader_code[kLoadShaderIndex32bpb] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_32bpb_cs, sizeof(shaders::texture_load_32bpb_cs)};
  load_shader_code[kLoadShaderIndex64bpb] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_64bpb_cs, sizeof(shaders::texture_load_64bpb_cs)};
  load_shader_code[kLoadShaderIndex128bpb] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_128bpb_cs, sizeof(shaders::texture_load_128bpb_cs)};
  load_shader_code[kLoadShaderIndexR5G5B5A1ToB5G5R5A1] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_r5g5b5a1_b5g5r5a1_cs,
                            sizeof(shaders::texture_load_r5g5b5a1_b5g5r5a1_cs)};
  load_shader_code[kLoadShaderIndexR5G6B5ToB5G6R5] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_r5g6b5_b5g6r5_cs, sizeof(shaders::texture_load_r5g6b5_b5g6r5_cs)};
  load_shader_code[kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs,
                            sizeof(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs)};
  load_shader_code[kLoadShaderIndexRGBA4ToBGRA4] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_r4g4b4a4_b4g4r4a4_cs,
                            sizeof(shaders::texture_load_r4g4b4a4_b4g4r4a4_cs)};
  load_shader_code[kLoadShaderIndexGBGR8ToGRGB8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_gbgr8_grgb8_cs, sizeof(shaders::texture_load_gbgr8_grgb8_cs)};
  load_shader_code[kLoadShaderIndexGBGR8ToRGB8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_gbgr8_rgb8_cs, sizeof(shaders::texture_load_gbgr8_rgb8_cs)};
  load_shader_code[kLoadShaderIndexBGRG8ToRGBG8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_bgrg8_rgbg8_cs, sizeof(shaders::texture_load_bgrg8_rgbg8_cs)};
  load_shader_code[kLoadShaderIndexBGRG8ToRGB8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_bgrg8_rgb8_cs, sizeof(shaders::texture_load_bgrg8_rgb8_cs)};
  load_shader_code[kLoadShaderIndexR10G11B11ToRGBA16] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_r10g11b11_rgba16_cs, sizeof(shaders::texture_load_r10g11b11_rgba16_cs)};
  load_shader_code[kLoadShaderIndexR10G11B11ToRGBA16SNorm] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_r10g11b11_rgba16_snorm_cs,
                            sizeof(shaders::texture_load_r10g11b11_rgba16_snorm_cs)};
  load_shader_code[kLoadShaderIndexR11G11B10ToRGBA16] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_r11g11b10_rgba16_cs, sizeof(shaders::texture_load_r11g11b10_rgba16_cs)};
  load_shader_code[kLoadShaderIndexR11G11B10ToRGBA16SNorm] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_r11g11b10_rgba16_snorm_cs,
                            sizeof(shaders::texture_load_r11g11b10_rgba16_snorm_cs)};
  load_shader_code[kLoadShaderIndexDXT1ToRGBA8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_dxt1_rgba8_cs, sizeof(shaders::texture_load_dxt1_rgba8_cs)};
  load_shader_code[kLoadShaderIndexDXT3ToRGBA8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_dxt3_rgba8_cs, sizeof(shaders::texture_load_dxt3_rgba8_cs)};
  load_shader_code[kLoadShaderIndexDXT5ToRGBA8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_dxt5_rgba8_cs, sizeof(shaders::texture_load_dxt5_rgba8_cs)};
  load_shader_code[kLoadShaderIndexDXNToRG8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_dxn_rg8_cs, sizeof(shaders::texture_load_dxn_rg8_cs)};
  load_shader_code[kLoadShaderIndexDXT3A] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_dxt3a_cs, sizeof(shaders::texture_load_dxt3a_cs)};
  load_shader_code[kLoadShaderIndexDXT3AAs1111ToBGRA4] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_dxt3aas1111_bgra4_cs,
                            sizeof(shaders::texture_load_dxt3aas1111_bgra4_cs)};
  load_shader_code[kLoadShaderIndexDXT5AToR8] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_dxt5a_r8_cs, sizeof(shaders::texture_load_dxt5a_r8_cs)};
  load_shader_code[kLoadShaderIndexCTX1] =
      D3D12_SHADER_BYTECODE{shaders::texture_load_ctx1_cs, sizeof(shaders::texture_load_ctx1_cs)};
  load_shader_code[kLoadShaderIndexDepthUnorm] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_depth_unorm_cs, sizeof(shaders::texture_load_depth_unorm_cs)};
  load_shader_code[kLoadShaderIndexDepthFloat] = D3D12_SHADER_BYTECODE{
      shaders::texture_load_depth_float_cs, sizeof(shaders::texture_load_depth_float_cs)};
  D3D12_SHADER_BYTECODE load_shader_code_scaled[kLoadShaderCount] = {};
  if (IsDrawResolutionScaled()) {
    load_shader_code_scaled[kLoadShaderIndex8bpb] = D3D12_SHADER_BYTECODE{
        shaders::texture_load_8bpb_scaled_cs, sizeof(shaders::texture_load_8bpb_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndex16bpb] = D3D12_SHADER_BYTECODE{
        shaders::texture_load_16bpb_scaled_cs, sizeof(shaders::texture_load_16bpb_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndex32bpb] = D3D12_SHADER_BYTECODE{
        shaders::texture_load_32bpb_scaled_cs, sizeof(shaders::texture_load_32bpb_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndex64bpb] = D3D12_SHADER_BYTECODE{
        shaders::texture_load_64bpb_scaled_cs, sizeof(shaders::texture_load_64bpb_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndex128bpb] = D3D12_SHADER_BYTECODE{
        shaders::texture_load_128bpb_scaled_cs, sizeof(shaders::texture_load_128bpb_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR5G5B5A1ToB5G5R5A1] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r5g5b5a1_b5g5r5a1_scaled_cs,
                              sizeof(shaders::texture_load_r5g5b5a1_b5g5r5a1_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR5G6B5ToB5G6R5] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r5g6b5_b5g6r5_scaled_cs,
                              sizeof(shaders::texture_load_r5g6b5_b5g6r5_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs,
                              sizeof(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexRGBA4ToBGRA4] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r4g4b4a4_b4g4r4a4_scaled_cs,
                              sizeof(shaders::texture_load_r4g4b4a4_b4g4r4a4_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR10G11B11ToRGBA16] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r10g11b11_rgba16_scaled_cs,
                              sizeof(shaders::texture_load_r10g11b11_rgba16_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR10G11B11ToRGBA16SNorm] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r10g11b11_rgba16_snorm_scaled_cs,
                              sizeof(shaders::texture_load_r10g11b11_rgba16_snorm_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR11G11B10ToRGBA16] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r11g11b10_rgba16_scaled_cs,
                              sizeof(shaders::texture_load_r11g11b10_rgba16_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexR11G11B10ToRGBA16SNorm] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_r11g11b10_rgba16_snorm_scaled_cs,
                              sizeof(shaders::texture_load_r11g11b10_rgba16_snorm_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexDepthUnorm] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_depth_unorm_scaled_cs,
                              sizeof(shaders::texture_load_depth_unorm_scaled_cs)};
    load_shader_code_scaled[kLoadShaderIndexDepthFloat] =
        D3D12_SHADER_BYTECODE{shaders::texture_load_depth_float_scaled_cs,
                              sizeof(shaders::texture_load_depth_float_scaled_cs)};
  }

  // Create the loading pipelines.
  for (size_t i = 0; i < kLoadShaderCount; ++i) {
    const D3D12_SHADER_BYTECODE& current_load_shader_code = load_shader_code[i];
    if (!current_load_shader_code.pShaderBytecode) {
      continue;
    }
    *(load_pipelines_[i].ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
        device, current_load_shader_code.pShaderBytecode, current_load_shader_code.BytecodeLength,
        load_root_signature_.Get());
    if (!load_pipelines_[i]) {
      REXGPU_ERROR(
          "D3D12TextureCache: Failed to create the texture loading pipeline "
          "for shader {}",
          i);
      return false;
    }
    if (IsDrawResolutionScaled()) {
      const D3D12_SHADER_BYTECODE& current_load_shader_code_scaled = load_shader_code_scaled[i];
      if (current_load_shader_code_scaled.pShaderBytecode) {
        *(load_pipelines_scaled_[i].ReleaseAndGetAddressOf()) =
            ui::d3d12::util::CreateComputePipeline(
                device, current_load_shader_code_scaled.pShaderBytecode,
                current_load_shader_code_scaled.BytecodeLength, load_root_signature_.Get());
        if (!load_pipelines_scaled_[i]) {
          REXGPU_ERROR(
              "D3D12TextureCache: Failed to create the resolution-scaled "
              "texture loading pipeline for shader {}",
              i);
          return false;
        }
      }
    }
  }

  // gpu_native_resolves: the resolve writes that are not plain copies. Native resolves of those
  // fall back to reloading from memory if this fails.
  if (REXCVAR_GET(gpu_native_resolves)) {
    D3D12_ROOT_PARAMETER native_resolve_root_parameters[3];
    native_resolve_root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    native_resolve_root_parameters[0].Constants.ShaderRegister = 0;
    native_resolve_root_parameters[0].Constants.RegisterSpace = 0;
    native_resolve_root_parameters[0].Constants.Num32BitValues =
        sizeof(NativeResolveConstants) / sizeof(uint32_t);
    native_resolve_root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_DESCRIPTOR_RANGE native_resolve_source_range;
    native_resolve_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    native_resolve_source_range.NumDescriptors = 1;
    native_resolve_source_range.BaseShaderRegister = 0;
    native_resolve_source_range.RegisterSpace = 0;
    native_resolve_source_range.OffsetInDescriptorsFromTableStart = 0;
    native_resolve_root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    native_resolve_root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    native_resolve_root_parameters[1].DescriptorTable.pDescriptorRanges =
        &native_resolve_source_range;
    native_resolve_root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_DESCRIPTOR_RANGE native_resolve_dest_range;
    native_resolve_dest_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    native_resolve_dest_range.NumDescriptors = 1;
    native_resolve_dest_range.BaseShaderRegister = 0;
    native_resolve_dest_range.RegisterSpace = 0;
    native_resolve_dest_range.OffsetInDescriptorsFromTableStart = 0;
    native_resolve_root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    native_resolve_root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
    native_resolve_root_parameters[2].DescriptorTable.pDescriptorRanges = &native_resolve_dest_range;
    native_resolve_root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC native_resolve_root_signature_desc;
    native_resolve_root_signature_desc.NumParameters =
        UINT(rex::countof(native_resolve_root_parameters));
    native_resolve_root_signature_desc.pParameters = native_resolve_root_parameters;
    native_resolve_root_signature_desc.NumStaticSamplers = 0;
    native_resolve_root_signature_desc.pStaticSamplers = nullptr;
    native_resolve_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    *(native_resolve_root_signature_.ReleaseAndGetAddressOf()) =
        ui::d3d12::util::CreateRootSignature(provider, native_resolve_root_signature_desc);
    if (native_resolve_root_signature_) {
      *(native_resolve_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
          device, shaders::native_resolve_cs, sizeof(shaders::native_resolve_cs),
          native_resolve_root_signature_.Get());
      *(native_resolve_ms_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
          device, shaders::native_resolve_ms_cs, sizeof(shaders::native_resolve_ms_cs),
          native_resolve_root_signature_.Get());
    }
    if (!native_resolve_root_signature_ || !native_resolve_pipeline_) {
      native_resolve_pipeline_.Reset();
      native_resolve_ms_pipeline_.Reset();
      native_resolve_root_signature_.Reset();
      REXGPU_WARN("D3D12TextureCache: Failed to create the native resolve pipeline");
    }
  }

  srv_descriptor_cache_allocated_ = 0;

  // Create a heap with null SRV descriptors, since it's faster to copy a
  // descriptor than to create an SRV, and null descriptors are used a lot (for
  // the signed version when only unsigned is used, for instance).
  D3D12_DESCRIPTOR_HEAP_DESC null_srv_descriptor_heap_desc;
  null_srv_descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  null_srv_descriptor_heap_desc.NumDescriptors = uint32_t(NullSRVDescriptorIndex::kCount);
  null_srv_descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  null_srv_descriptor_heap_desc.NodeMask = 0;
  if (FAILED(device->CreateDescriptorHeap(&null_srv_descriptor_heap_desc,
                                          IID_PPV_ARGS(&null_srv_descriptor_heap_)))) {
    REXGPU_ERROR(
        "D3D12TextureCache: Failed to create the descriptor heap for null "
        "SRVs");
    return false;
  }
  null_srv_descriptor_heap_start_ = null_srv_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
  D3D12_SHADER_RESOURCE_VIEW_DESC null_srv_desc;
  null_srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  null_srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
      D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
      D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0);
  null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
  null_srv_desc.Texture2DArray.MostDetailedMip = 0;
  null_srv_desc.Texture2DArray.MipLevels = 1;
  null_srv_desc.Texture2DArray.FirstArraySlice = 0;
  null_srv_desc.Texture2DArray.ArraySize = 1;
  null_srv_desc.Texture2DArray.PlaneSlice = 0;
  null_srv_desc.Texture2DArray.ResourceMinLODClamp = 0.0f;
  device->CreateShaderResourceView(
      nullptr, &null_srv_desc,
      provider.OffsetViewDescriptor(null_srv_descriptor_heap_start_,
                                    uint32_t(NullSRVDescriptorIndex::k2DArray)));
  null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
  null_srv_desc.Texture3D.MostDetailedMip = 0;
  null_srv_desc.Texture3D.MipLevels = 1;
  null_srv_desc.Texture3D.ResourceMinLODClamp = 0.0f;
  device->CreateShaderResourceView(
      nullptr, &null_srv_desc,
      provider.OffsetViewDescriptor(null_srv_descriptor_heap_start_,
                                    uint32_t(NullSRVDescriptorIndex::k3D)));
  null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
  null_srv_desc.TextureCube.MostDetailedMip = 0;
  null_srv_desc.TextureCube.MipLevels = 1;
  null_srv_desc.TextureCube.ResourceMinLODClamp = 0.0f;
  device->CreateShaderResourceView(
      nullptr, &null_srv_desc,
      provider.OffsetViewDescriptor(null_srv_descriptor_heap_start_,
                                    uint32_t(NullSRVDescriptorIndex::kCube)));

  // gpu_lazy_resolve_copies_unscaled: every reader of shared memory requests its ranges first.
  shared_memory().SetBeforeRequestHook(BeforeSharedMemoryRequest, this);

  return true;
}

void D3D12TextureCache::ClearCache() {
  // The mirrors are destroyed with the rest: their pending copies can't be rebuilt any more (the
  // textures reload from memory, as after any cache clear).
  if (!lazy_resolve_copies_.empty()) {
    REXGPU_WARN("LAZYRESOLVE: {} pending copies dropped with the texture cache",
                lazy_resolve_copies_.size());
    lazy_resolve_copies_.clear();
    lazy_unscaled_pending_ = 0;
    lazy_unscaled_start_ = UINT32_MAX;
    lazy_unscaled_end_ = 0;
  }
  last_native_resolve_mirror_ = nullptr;
  TextureCache::ClearCache();

  // Clear texture descriptor cache.
  srv_descriptor_cache_free_.clear();
  srv_descriptor_cache_allocated_ = 0;
  srv_descriptor_cache_.clear();
}

void D3D12TextureCache::BeginSubmission(uint64_t new_submission_index) {
  TextureCache::BeginSubmission(new_submission_index);

  // gpu_lazy_resolve_copies: the mirrors of pending copies stay in the cache (the memory limit only
  // destroys textures unused since a completed submission).
  for (PendingLazyResolveCopy& pending : lazy_resolve_copies_) {
    pending.mirror->MarkAsUsed();
  }

  // ExecuteCommandLists is a full UAV and aliasing barrier.
  if (IsDrawResolutionScaled()) {
    size_t scaled_resolve_buffer_count = GetScaledResolveBufferCount();
    for (size_t i = 0; i < scaled_resolve_buffer_count; ++i) {
      ScaledResolveVirtualBuffer* scaled_resolve_buffer = scaled_resolve_2gb_buffers_[i].get();
      if (scaled_resolve_buffer) {
        scaled_resolve_buffer->ClearUAVBarrierPending();
      }
    }
    std::memset(scaled_resolve_1gb_buffer_indices_, UINT8_MAX,
                sizeof(scaled_resolve_1gb_buffer_indices_));
  }
}

void D3D12TextureCache::BeginFrame() {
  TextureCache::BeginFrame();
  ApplyReadyReplacements();

  std::memset(unsupported_format_features_used_, 0, sizeof(unsupported_format_features_used_));
}

void D3D12TextureCache::EndFrame() {
  // Report used unsupported texture formats.
  bool unsupported_header_written = false;
  for (uint32_t i = 0; i < 64; ++i) {
    uint32_t unsupported_features = unsupported_format_features_used_[i];
    if (unsupported_features == 0) {
      continue;
    }
    if (!unsupported_header_written) {
      REXGPU_ERROR("Unsupported texture formats used in the frame:");
      unsupported_header_written = true;
    }
    REXGPU_ERROR("* {}{}{}{}", FormatInfo::Get(xenos::TextureFormat(i))->name,
                 unsupported_features & kUnsupportedResourceBit ? " resource" : "",
                 unsupported_features & kUnsupportedUnormBit ? " unsigned" : "",
                 unsupported_features & kUnsupportedSnormBit ? " signed" : "");
    unsupported_format_features_used_[i] = 0;
  }
}

void D3D12TextureCache::RequestTextures(uint32_t used_texture_mask) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  TextureCache::RequestTextures(used_texture_mask);

  // Pre-create 3D-as-2D wrappers before draw setup. Wrapper loading may bind
  // compute pipelines and must happen in the texture request phase.
  if (REXCVAR_GET(gpu_3d_to_2d_texture)) {
    uint32_t textures_3d = used_texture_mask;
    uint32_t index_3d;
    while (rex::bit_scan_forward(textures_3d, &index_3d)) {
      textures_3d &= ~(uint32_t(1) << index_3d);
      const TextureBinding* binding = GetValidTextureBinding(index_3d);
      if (!binding || binding->key.dimension != xenos::DataDimension::k3D) {
        continue;
      }
      D3D12Texture* texture = static_cast<D3D12Texture*>(binding->texture);
      if (texture) {
        texture->GetOrCreate3DAs2DResource(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
      D3D12Texture* texture_signed = static_cast<D3D12Texture*>(binding->texture_signed);
      if (texture_signed) {
        texture_signed->GetOrCreate3DAs2DResource(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                                  D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
    }
  }

  // Transition the textures to the needed usage - always in
  // NON_PIXEL_SHADER_RESOURCE | PIXEL_SHADER_RESOURCE states because barriers
  // between read-only stages, if needed, are discouraged (also if these were
  // tracked separately, checks would be needed to make sure, if the same
  // texture is bound through different fetch constants to both VS and PS, it
  // would be in both states).
  uint32_t textures_remaining = used_texture_mask;
  uint32_t index;
  while (rex::bit_scan_forward(textures_remaining, &index)) {
    textures_remaining &= ~(uint32_t(1) << index);
    const TextureBinding* binding = GetValidTextureBinding(index);
    if (!binding) {
      continue;
    }
    D3D12Texture* binding_texture = static_cast<D3D12Texture*>(binding->texture);
    if (binding_texture != nullptr) {
      // Will be referenced by the command list, so mark as used.
      binding_texture->MarkAsUsed();
      command_processor_.PushTransitionBarrier(
          binding_texture->resource(),
          binding_texture->SetResourceState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
              D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    D3D12Texture* binding_texture_signed = static_cast<D3D12Texture*>(binding->texture_signed);
    if (binding_texture_signed != nullptr) {
      binding_texture_signed->MarkAsUsed();
      command_processor_.PushTransitionBarrier(
          binding_texture_signed->resource(),
          binding_texture_signed->SetResourceState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
              D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
  }
}

bool D3D12TextureCache::AreActiveTextureSRVKeysUpToDate(
    const TextureSRVKey* keys, const D3D12Shader::TextureBinding* host_shader_bindings,
    size_t host_shader_binding_count) const {
  for (size_t i = 0; i < host_shader_binding_count; ++i) {
    const TextureSRVKey& key = keys[i];
    const TextureBinding* binding = GetValidTextureBinding(host_shader_bindings[i].fetch_constant);
    if (!binding) {
      if (key.key.is_valid) {
        return false;
      }
      continue;
    }
    if (key.key != binding->key || key.host_swizzle != binding->host_swizzle ||
        key.swizzled_signs != binding->swizzled_signs) {
      return false;
    }
  }
  return true;
}

void D3D12TextureCache::WriteActiveTextureSRVKeys(
    TextureSRVKey* keys, const D3D12Shader::TextureBinding* host_shader_bindings,
    size_t host_shader_binding_count) const {
  for (size_t i = 0; i < host_shader_binding_count; ++i) {
    TextureSRVKey& key = keys[i];
    const TextureBinding* binding = GetValidTextureBinding(host_shader_bindings[i].fetch_constant);
    if (!binding) {
      key.key.MakeInvalid();
      key.host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_0000;
      key.swizzled_signs = kSwizzledSignsUnsigned;
      continue;
    }
    key.key = binding->key;
    key.host_swizzle = binding->host_swizzle;
    key.swizzled_signs = binding->swizzled_signs;
  }
}

void D3D12TextureCache::WriteActiveTextureBindfulSRV(
    const D3D12Shader::TextureBinding& host_shader_binding, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
  assert_false(bindless_resources_used_);
  uint32_t descriptor_index = UINT32_MAX;
  Texture* texture = nullptr;
  uint32_t fetch_constant_index = host_shader_binding.fetch_constant;
  const TextureBinding* binding = GetValidTextureBinding(fetch_constant_index);
  if (binding && AreDimensionsCompatible(host_shader_binding.dimension, binding->key.dimension)) {
    bool force_special_view = binding->key.dimension == xenos::DataDimension::k3D &&
                              (host_shader_binding.dimension == xenos::FetchOpDimension::k1D ||
                               host_shader_binding.dimension == xenos::FetchOpDimension::k2D);
    const D3D12TextureBinding& d3d12_binding = d3d12_texture_bindings_[fetch_constant_index];
    if (host_shader_binding.is_signed) {
      // Not supporting signed compressed textures - hopefully DXN and DXT5A are
      // not used as signed.
      if (texture_util::IsAnySignSigned(binding->swizzled_signs)) {
        texture = IsSignedVersionSeparateForFormat(binding->key) ? binding->texture_signed
                                                                 : binding->texture;
        if (force_special_view && texture) {
          descriptor_index = FindOrCreateTextureDescriptor(*static_cast<D3D12Texture*>(texture),
                                                           xenos::DataDimension::k2DOrStacked, true,
                                                           binding->host_swizzle);
        } else {
          descriptor_index = d3d12_binding.descriptor_index_signed;
        }
      }
    } else {
      if (texture_util::IsAnySignNotSigned(binding->swizzled_signs)) {
        texture = binding->texture;
        if (force_special_view && texture) {
          descriptor_index = FindOrCreateTextureDescriptor(*static_cast<D3D12Texture*>(texture),
                                                           xenos::DataDimension::k2DOrStacked,
                                                           false, binding->host_swizzle);
        } else {
          descriptor_index = d3d12_binding.descriptor_index;
        }
      }
    }
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  D3D12_CPU_DESCRIPTOR_HANDLE source_handle;
  if (descriptor_index != UINT32_MAX) {
    assert_not_null(texture);
    texture->MarkAsUsed();
    source_handle = GetTextureDescriptorCPUHandle(descriptor_index);
  } else {
    NullSRVDescriptorIndex null_descriptor_index;
    switch (host_shader_binding.dimension) {
      case xenos::FetchOpDimension::k3DOrStacked:
        null_descriptor_index = NullSRVDescriptorIndex::k3D;
        break;
      case xenos::FetchOpDimension::kCube:
        null_descriptor_index = NullSRVDescriptorIndex::kCube;
        break;
      default:
        assert_true(host_shader_binding.dimension == xenos::FetchOpDimension::k1D ||
                    host_shader_binding.dimension == xenos::FetchOpDimension::k2D);
        null_descriptor_index = NullSRVDescriptorIndex::k2DArray;
    }
    source_handle = provider.OffsetViewDescriptor(null_srv_descriptor_heap_start_,
                                                  uint32_t(null_descriptor_index));
  }
  auto device = provider.GetDevice();
  {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
    SCOPE_profile_cpu_i("gpu",
                        "rex::graphics::d3d12::D3D12TextureCache::WriteActiveTextureBindfulSRV->"
                        "CopyDescriptorsSimple");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES
    device->CopyDescriptorsSimple(1, handle, source_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  }
}

uint32_t D3D12TextureCache::GetActiveTextureBindlessSRVIndex(
    const D3D12Shader::TextureBinding& host_shader_binding) {
  assert_true(bindless_resources_used_);
  uint32_t descriptor_index = UINT32_MAX;
  uint32_t fetch_constant_index = host_shader_binding.fetch_constant;
  const TextureBinding* binding = GetValidTextureBinding(fetch_constant_index);
  if (binding && AreDimensionsCompatible(host_shader_binding.dimension, binding->key.dimension)) {
    bool force_special_view = binding->key.dimension == xenos::DataDimension::k3D &&
                              (host_shader_binding.dimension == xenos::FetchOpDimension::k1D ||
                               host_shader_binding.dimension == xenos::FetchOpDimension::k2D);
    const D3D12TextureBinding& d3d12_binding = d3d12_texture_bindings_[fetch_constant_index];
    if (force_special_view) {
      Texture* texture = nullptr;
      bool use_signed =
          host_shader_binding.is_signed && texture_util::IsAnySignSigned(binding->swizzled_signs);
      if (use_signed) {
        texture = IsSignedVersionSeparateForFormat(binding->key) ? binding->texture_signed
                                                                 : binding->texture;
      } else {
        texture = binding->texture;
      }
      if (texture) {
        descriptor_index = FindOrCreateTextureDescriptor(*static_cast<D3D12Texture*>(texture),
                                                         xenos::DataDimension::k2DOrStacked,
                                                         use_signed, binding->host_swizzle);
      }
    } else {
      descriptor_index = host_shader_binding.is_signed ? d3d12_binding.descriptor_index_signed
                                                       : d3d12_binding.descriptor_index;
    }
  }
  if (descriptor_index == UINT32_MAX) {
    switch (host_shader_binding.dimension) {
      case xenos::FetchOpDimension::k3DOrStacked:
        descriptor_index = uint32_t(D3D12CommandProcessor::SystemBindlessView::kNullTexture3D);
        break;
      case xenos::FetchOpDimension::kCube:
        descriptor_index = uint32_t(D3D12CommandProcessor::SystemBindlessView::kNullTextureCube);
        break;
      default:
        assert_true(host_shader_binding.dimension == xenos::FetchOpDimension::k1D ||
                    host_shader_binding.dimension == xenos::FetchOpDimension::k2D);
        descriptor_index = uint32_t(D3D12CommandProcessor::SystemBindlessView::kNullTexture2DArray);
    }
  }
  return descriptor_index;
}

D3D12TextureCache::SamplerParameters D3D12TextureCache::GetSamplerParameters(
    const D3D12Shader::SamplerBinding& binding) const {
  const auto& regs = register_file();
  xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(binding.fetch_constant);

  SamplerParameters parameters;

  xenos::ClampMode fetch_clamp_x, fetch_clamp_y, fetch_clamp_z;
  texture_util::GetClampModesForDimension(fetch, fetch_clamp_x, fetch_clamp_y, fetch_clamp_z);
  parameters.clamp_x = NormalizeClampMode(fetch_clamp_x);
  parameters.clamp_y = NormalizeClampMode(fetch_clamp_y);
  parameters.clamp_z = NormalizeClampMode(fetch_clamp_z);
  if (xenos::ClampModeUsesBorder(parameters.clamp_x) ||
      xenos::ClampModeUsesBorder(parameters.clamp_y) ||
      xenos::ClampModeUsesBorder(parameters.clamp_z)) {
    parameters.border_color = fetch.border_color;
  } else {
    parameters.border_color = xenos::BorderColor::k_ABGR_Black;
  }

  uint32_t mip_min_level, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 &mip_min_level, &mip_max_level);
  parameters.mip_min_level = mip_min_level;
  bool has_mips = mip_max_level > mip_min_level;

  xenos::TextureFilter mag_filter = binding.mag_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.mag_filter
                                        : binding.mag_filter;
  xenos::TextureFilter min_filter = binding.min_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.min_filter
                                        : binding.min_filter;
  xenos::TextureFilter mip_filter = binding.mip_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.mip_filter
                                        : binding.mip_filter;
  bool min_mag_linear = (mag_filter == xenos::TextureFilter::kLinear) &&
                        (min_filter == xenos::TextureFilter::kLinear);
  bool mip_filter_bilinear_or_trilinear =
      mip_filter == xenos::TextureFilter::kPoint || mip_filter == xenos::TextureFilter::kLinear;
  bool mip_base_map = mip_filter == xenos::TextureFilter::kBaseMap;

  // TODO(Triang3l): Disable filtering for texture formats not supporting it.
  xenos::AnisoFilter aniso_filter = binding.aniso_filter == xenos::AnisoFilter::kUseFetchConst
                                        ? fetch.aniso_filter
                                        : binding.aniso_filter;
  int32_t anisotropic_override = REXCVAR_GET(anisotropic_override);
  if (anisotropic_override > -1 && anisotropic_override < 6 && has_mips && !mip_base_map &&
      min_mag_linear && mip_filter_bilinear_or_trilinear) {
    aniso_filter = xenos::AnisoFilter(anisotropic_override);
  }
  aniso_filter = std::min(aniso_filter, xenos::AnisoFilter::kMax_16_1);
  parameters.aniso_filter = aniso_filter;
  if (aniso_filter != xenos::AnisoFilter::kDisabled) {
    parameters.mag_linear = 1;
    parameters.min_linear = 1;
    parameters.mip_linear = 1;
  } else {
    parameters.mag_linear = mag_filter == xenos::TextureFilter::kLinear;
    parameters.min_linear = min_filter == xenos::TextureFilter::kLinear;
    parameters.mip_linear = mip_filter == xenos::TextureFilter::kLinear;
  }
  parameters.mip_base_map = mip_base_map;

  return parameters;
}

void D3D12TextureCache::WriteSampler(SamplerParameters parameters,
                                     D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
  D3D12_SAMPLER_DESC desc;
  if (parameters.aniso_filter != xenos::AnisoFilter::kDisabled) {
    desc.Filter = D3D12_FILTER_ANISOTROPIC;
    desc.MaxAnisotropy = 1u << (uint32_t(parameters.aniso_filter) - 1);
  } else {
    D3D12_FILTER_TYPE d3d_filter_min =
        parameters.min_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    D3D12_FILTER_TYPE d3d_filter_mag =
        parameters.mag_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    D3D12_FILTER_TYPE d3d_filter_mip =
        parameters.mip_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    desc.Filter = D3D12_ENCODE_BASIC_FILTER(d3d_filter_min, d3d_filter_mag, d3d_filter_mip,
                                            D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    desc.MaxAnisotropy = 1;
  }
  static const D3D12_TEXTURE_ADDRESS_MODE kAddressModeMap[] = {
      /* kRepeat               */ D3D12_TEXTURE_ADDRESS_MODE_WRAP,
      /* kMirroredRepeat       */ D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
      /* kClampToEdge          */ D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
      /* kMirrorClampToEdge    */ D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
      // No GL_CLAMP (clamp to half edge, half border) equivalent in Direct3D
      // 12, but there's no Direct3D 9 equivalent anyway, and too weird to be
      // suitable for intentional real usage.
      /* kClampToHalfway       */ D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
      // No mirror and clamp to border equivalents in Direct3D 12, but they
      // aren't there in Direct3D 9 either.
      /* kMirrorClampToHalfway */ D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
      /* kClampToBorder        */ D3D12_TEXTURE_ADDRESS_MODE_BORDER,
      /* kMirrorClampToBorder  */ D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
  };
  desc.AddressU = kAddressModeMap[uint32_t(parameters.clamp_x)];
  desc.AddressV = kAddressModeMap[uint32_t(parameters.clamp_y)];
  desc.AddressW = kAddressModeMap[uint32_t(parameters.clamp_z)];
  // LOD biasing is performed in shaders.
  desc.MipLODBias = 0.0f;
  desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
  switch (parameters.border_color) {
    case xenos::BorderColor::k_ABGR_White:
      desc.BorderColor[0] = 1.0f;
      desc.BorderColor[1] = 1.0f;
      desc.BorderColor[2] = 1.0f;
      desc.BorderColor[3] = 1.0f;
      break;
    case xenos::BorderColor::k_ACBYCR_Black:
      desc.BorderColor[0] = 0.5f;
      desc.BorderColor[1] = 0.0f;
      desc.BorderColor[2] = 0.5f;
      desc.BorderColor[3] = 0.0f;
      break;
    case xenos::BorderColor::k_ACBCRY_Black:
      desc.BorderColor[0] = 0.0f;
      desc.BorderColor[1] = 0.5f;
      desc.BorderColor[2] = 0.5f;
      desc.BorderColor[3] = 0.0f;
      break;
    default:
      assert_true(parameters.border_color == xenos::BorderColor::k_ABGR_Black);
      desc.BorderColor[0] = 0.0f;
      desc.BorderColor[1] = 0.0f;
      desc.BorderColor[2] = 0.0f;
      desc.BorderColor[3] = 0.0f;
      break;
  }
  desc.MinLOD = float(parameters.mip_min_level);
  if (parameters.mip_base_map) {
    // "It is undefined whether LOD clamping based on MinLOD and MaxLOD Sampler
    // states should happen before or after deciding if magnification is
    // occuring" - Direct3D 11.3 Functional Specification.
    // Using the GL_NEAREST / GL_LINEAR minification filter emulation logic
    // described in the Vulkan VkSamplerCreateInfo specification, preserving
    // magnification vs. minification - point mip sampling (usable only without
    // anisotropic filtering on Direct3D 12) and MaxLOD 0.25. With anisotropic
    // filtering, magnification vs. minification doesn't matter as the filter is
    // always linear for both on Direct3D 12 - but linear filtering specifically
    // is what must not be done for kBaseMap, so setting MaxLOD to MinLOD.
    desc.MaxLOD = desc.MinLOD;
    if (parameters.aniso_filter == xenos::AnisoFilter::kDisabled) {
      assert_false(parameters.mip_linear);
      desc.MaxLOD += 0.25f;
    }
  } else {
    // Maximum mip level is in the texture resource itself.
    desc.MaxLOD = FLT_MAX;
  }
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  device->CreateSampler(&desc, handle);
}

bool D3D12TextureCache::ClampDrawResolutionScaleToMaxSupported(
    uint32_t& scale_x, uint32_t& scale_y, const ui::d3d12::D3D12Provider& provider) {
  bool was_clamped;
  if (provider.GetTiledResourcesTier() < D3D12_TILED_RESOURCES_TIER_1) {
    was_clamped = scale_x > 1 || scale_y > 1;
    scale_x = 1;
    scale_y = 1;
    return !was_clamped;
  }
  // Limit to the virtual address space available for a resource.
  was_clamped = false;
  uint32_t virtual_address_bits_per_resource = provider.GetVirtualAddressBitsPerResource();
  while (scale_x > 1 || scale_y > 1) {
    uint64_t highest_scaled_address = uint64_t(SharedMemory::kBufferSize) * (scale_x * scale_y) - 1;
    if (uint32_t(64) - rex::lzcnt(highest_scaled_address) <= virtual_address_bits_per_resource) {
      break;
    }
    // When reducing from a square size, prefer decreasing the horizontal
    // resolution as vertical resolution difference is visible more clearly in
    // perspective.
    was_clamped = true;
    if (scale_x >= scale_y) {
      --scale_x;
    } else {
      --scale_y;
    }
  }
  return !was_clamped;
}

bool D3D12TextureCache::EnsureScaledResolveMemoryCommitted(uint32_t start_unscaled,
                                                           uint32_t length_unscaled,
                                                           uint32_t length_scaled_alignment_log2) {
  assert_true(IsDrawResolutionScaled());

  if (length_unscaled == 0) {
    return true;
  }
  if (start_unscaled > SharedMemory::kBufferSize ||
      (SharedMemory::kBufferSize - start_unscaled) < length_unscaled) {
    // Exceeds the physical address space.
    return false;
  }

  uint32_t draw_resolution_scale_area = draw_resolution_scale_x() * draw_resolution_scale_y();
  uint64_t first_scaled = uint64_t(start_unscaled) * draw_resolution_scale_area;
  uint64_t length_scaled_alignment_bits = (UINT64_C(1) << length_scaled_alignment_log2) - 1;
  uint64_t last_scaled =
      (uint64_t(start_unscaled + (length_unscaled - 1)) * draw_resolution_scale_area +
       length_scaled_alignment_bits) &
      ~length_scaled_alignment_bits;

  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  // Ensure GPU virtual memory for buffers that may be used to access the range
  // is allocated - buffers are created. Always creating both buffers for all
  // addresses before creating the heaps so when creating a new buffer, it can
  // be safely assumed that no existing heaps should be mapped to it.
  std::array<size_t, 2> possible_buffers_first =
      GetPossibleScaledResolveBufferIndices(first_scaled);
  std::array<size_t, 2> possible_buffers_last = GetPossibleScaledResolveBufferIndices(last_scaled);
  size_t possible_buffer_first = std::min(possible_buffers_first[0], possible_buffers_first[1]);
  size_t possible_buffer_last = std::max(possible_buffers_last[0], possible_buffers_last[1]);
  for (size_t i = possible_buffer_first; i <= possible_buffer_last; ++i) {
    if (scaled_resolve_2gb_buffers_[i]) {
      continue;
    }
    D3D12_RESOURCE_DESC scaled_resolve_buffer_desc;
    // Buffer indices are gigabytes.
    ui::d3d12::util::FillBufferResourceDesc(
        scaled_resolve_buffer_desc,
        std::min(
            uint64_t(1) << 31,
            uint64_t(SharedMemory::kBufferSize) * draw_resolution_scale_area - (uint64_t(i) << 30)),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    // The first access will be a resolve.
    constexpr D3D12_RESOURCE_STATES kScaledResolveVirtualBufferInitialState =
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ID3D12Resource* scaled_resolve_buffer_resource;
    if (FAILED(device->CreateReservedResource(&scaled_resolve_buffer_desc,
                                              kScaledResolveVirtualBufferInitialState, nullptr,
                                              IID_PPV_ARGS(&scaled_resolve_buffer_resource)))) {
      REXGPU_ERROR(
          "D3D12TextureCache: Failed to create a 2 GB tiled buffer for draw "
          "resolution scaling");
      return false;
    }
    scaled_resolve_2gb_buffers_[i] =
        std::unique_ptr<ScaledResolveVirtualBuffer>(new ScaledResolveVirtualBuffer(
            scaled_resolve_buffer_resource, kScaledResolveVirtualBufferInitialState));
    scaled_resolve_buffer_resource->Release();
  }

  uint32_t heap_first = uint32_t(first_scaled >> kScaledResolveHeapSizeLog2);
  uint32_t heap_last = uint32_t(last_scaled >> kScaledResolveHeapSizeLog2);
  for (uint32_t i = heap_first; i <= heap_last; ++i) {
    if (scaled_resolve_heaps_[i]) {
      continue;
    }
    auto direct_queue = provider.GetDirectQueue();
    D3D12_HEAP_DESC heap_desc = {};
    heap_desc.SizeInBytes = kScaledResolveHeapSize;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | provider.GetHeapFlagCreateNotZeroed();
    Microsoft::WRL::ComPtr<ID3D12Heap> scaled_resolve_heap;
    if (FAILED(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&scaled_resolve_heap)))) {
      REXGPU_ERROR("D3D12TextureCache: Failed to create a scaled resolve tile heap");
      return false;
    }
    scaled_resolve_heaps_[i] = scaled_resolve_heap;
    ++scaled_resolve_heap_count_;
    COUNT_profile_set("gpu/texture_cache/scaled_resolve_buffer_used_mb",
                      scaled_resolve_heap_count_ << (kScaledResolveHeapSizeLog2 - 20));
    D3D12_TILED_RESOURCE_COORDINATE region_start_coordinates;
    region_start_coordinates.Y = 0;
    region_start_coordinates.Z = 0;
    region_start_coordinates.Subresource = 0;
    D3D12_TILE_REGION_SIZE region_size;
    region_size.NumTiles = kScaledResolveHeapSize / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    region_size.UseBox = FALSE;
    D3D12_TILE_RANGE_FLAGS range_flags = D3D12_TILE_RANGE_FLAG_NONE;
    UINT heap_range_start_offset = 0;
    UINT range_tile_count = kScaledResolveHeapSize / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    std::array<size_t, 2> buffer_indices =
        GetPossibleScaledResolveBufferIndices(uint64_t(i) << kScaledResolveHeapSizeLog2);
    for (size_t j = 0; j < 2; ++j) {
      size_t buffer_index = buffer_indices[j];
      if (j && buffer_index == buffer_indices[0]) {
        break;
      }
      region_start_coordinates.X =
          UINT(((uint64_t(i) << kScaledResolveHeapSizeLog2) - (uint64_t(buffer_index) << 30)) /
               D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);
      direct_queue->UpdateTileMappings(
          scaled_resolve_2gb_buffers_[buffer_index]->resource(), 1, &region_start_coordinates,
          &region_size, scaled_resolve_heap.Get(), 1, &range_flags, &heap_range_start_offset,
          &range_tile_count, D3D12_TILE_MAPPING_FLAG_NONE);
    }
    command_processor_.NotifyQueueOperationsDoneDirectly();
  }
  return true;
}

bool D3D12TextureCache::MakeScaledResolveRangeCurrent(uint32_t start_unscaled,
                                                      uint32_t length_unscaled,
                                                      uint32_t length_scaled_alignment_log2) {
  assert_true(IsDrawResolutionScaled());

  if (!length_unscaled || start_unscaled >= SharedMemory::kBufferSize ||
      (SharedMemory::kBufferSize - start_unscaled) < length_unscaled) {
    // If length is 0, the needed buffer can't be chosen because no buffer is
    // needed.
    return false;
  }

  uint32_t draw_resolution_scale_area = draw_resolution_scale_x() * draw_resolution_scale_y();
  uint64_t start_scaled = uint64_t(start_unscaled) * draw_resolution_scale_area;
  uint64_t length_scaled_alignment_bits = (UINT64_C(1) << length_scaled_alignment_log2) - 1;
  uint64_t length_scaled =
      (uint64_t(length_unscaled) * draw_resolution_scale_area + length_scaled_alignment_bits) &
      ~length_scaled_alignment_bits;
  uint64_t last_scaled = start_scaled + (length_scaled - 1);

  // Get one or two buffers that can hold the whole range.
  std::array<size_t, 2> possible_buffer_indices_first =
      GetPossibleScaledResolveBufferIndices(start_scaled);
  std::array<size_t, 2> possible_buffer_indices_last =
      GetPossibleScaledResolveBufferIndices(last_scaled);
  size_t possible_buffer_indices_common[2];
  size_t possible_buffer_indices_common_count = 0;
  for (size_t i = 0;
       i <= size_t(possible_buffer_indices_first[0] != possible_buffer_indices_first[1]); ++i) {
    size_t possible_buffer_index_first = possible_buffer_indices_first[i];
    for (size_t j = 0;
         j <= size_t(possible_buffer_indices_last[0] != possible_buffer_indices_last[1]); ++j) {
      if (possible_buffer_indices_last[j] == possible_buffer_index_first) {
        bool possible_buffer_index_already_added = false;
        for (size_t k = 0; k < possible_buffer_indices_common_count; ++k) {
          if (possible_buffer_indices_common[k] == possible_buffer_index_first) {
            possible_buffer_index_already_added = true;
            break;
          }
        }
        if (!possible_buffer_index_already_added) {
          assert_true(possible_buffer_indices_common_count < 2);
          possible_buffer_indices_common[possible_buffer_indices_common_count++] =
              possible_buffer_index_first;
        }
      }
    }
  }
  if (!possible_buffer_indices_common_count) {
    // Too wide range requested - no buffer that contains both the start and the
    // end.
    return false;
  }

  size_t gigabyte_first = size_t(start_scaled >> 30);
  size_t gigabyte_last = size_t(last_scaled >> 30);

  // Choose the buffer that the range will be accessed through.
  size_t new_buffer_index;
  if (possible_buffer_indices_common_count >= 2) {
    // Prefer the buffer that is already used to make less aliasing barriers.
    assert_true(gigabyte_first + 1 >= gigabyte_last);
    size_t possible_buffer_indices_already_used[2] = {};
    for (size_t i = gigabyte_first; i <= gigabyte_last; ++i) {
      size_t gigabyte_current_buffer_index = scaled_resolve_1gb_buffer_indices_[i];
      for (size_t j = 0; j < possible_buffer_indices_common_count; ++j) {
        if (possible_buffer_indices_common[j] == gigabyte_current_buffer_index) {
          ++possible_buffer_indices_already_used[j];
        }
      }
    }
    new_buffer_index = possible_buffer_indices_common[size_t(
        possible_buffer_indices_already_used[1] > possible_buffer_indices_already_used[0])];
  } else {
    // The range can be accessed only by one buffer.
    new_buffer_index = possible_buffer_indices_common[0];
  }

  // Switch the current buffer for the range.
  const ScaledResolveVirtualBuffer* new_buffer =
      scaled_resolve_2gb_buffers_[new_buffer_index].get();
  assert_not_null(new_buffer);
  ID3D12Resource* new_buffer_resource = new_buffer->resource();
  for (size_t i = gigabyte_first; i <= gigabyte_last; ++i) {
    size_t gigabyte_current_buffer_index = scaled_resolve_1gb_buffer_indices_[i];
    if (gigabyte_current_buffer_index == new_buffer_index) {
      continue;
    }
    if (gigabyte_current_buffer_index != SIZE_MAX) {
      ScaledResolveVirtualBuffer* gigabyte_current_buffer =
          scaled_resolve_2gb_buffers_[gigabyte_current_buffer_index].get();
      assert_not_null(gigabyte_current_buffer);
      command_processor_.PushAliasingBarrier(gigabyte_current_buffer->resource(),
                                             new_buffer_resource);
      // An aliasing barrier synchronizes and flushes everything.
      gigabyte_current_buffer->ClearUAVBarrierPending();
    }
    scaled_resolve_1gb_buffer_indices_[i] = new_buffer_index;
  }

  scaled_resolve_current_range_start_scaled_ = start_scaled;
  scaled_resolve_current_range_length_scaled_ = length_scaled;
  return true;
}

void D3D12TextureCache::TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATES new_state) {
  assert_true(IsDrawResolutionScaled());
  ScaledResolveVirtualBuffer& buffer = GetCurrentScaledResolveBuffer();
  command_processor_.PushTransitionBarrier(buffer.resource(), buffer.SetResourceState(new_state),
                                           new_state);
}

void D3D12TextureCache::WriteNativeResolveToTextures(const NativeResolveWrite& write,
                                                     D3D12_RESOURCE_STATES& source_state,
                                                     NativeResolveWriteCounts& counts) {
  last_native_resolve_mirror_ = nullptr;
  if (!write.source || !write.width || !write.height) {
    return;
  }
  const bool scaled = IsDrawResolutionScaled() && !write.unscaled;
  const uint32_t scale_x = scaled ? draw_resolution_scale_x() : 1;
  const uint32_t scale_y = scaled ? draw_resolution_scale_y() : 1;
  // gpu_lazy_resolve_copies: a written texture containing the whole rectangle mirrors the
  // destination memory there. The rectangle is rounded up to 8 pixels (1040x620 resolves as
  // 1040x624): up to 7 more columns and rows than the texture are its padding, which the rebuild
  // leaves out.
  auto note_mirror = [&](D3D12Texture& texture) {
    const TextureKey& key = texture.key();
    if (!counts.mirror && write.dest_x < key.GetWidth() && write.dest_y < key.GetHeight() &&
        write.dest_x + write.width < key.GetWidth() + 8 &&
        write.dest_y + write.height < key.GetHeight() + 8) {
      counts.mirror = true;
      last_native_resolve_mirror_ = &texture;
    }
  };
  ForEachTextureAtBasePage(write.base_page, [&](Texture& texture) {
    const TextureKey& key = texture.key();
    if (!key.tiled || key.dimension != xenos::DataDimension::k2DOrStacked ||
        key.depth_or_array_size_minus_1 != 0 || key.mip_max_level != 0 || IsBaseInPackedMipTail(key) ||
        key.signed_separate || key.format != write.format || key.pitch != write.pitch ||
        key.endianness != write.endianness || bool(key.scaled_resolve) != scaled ||
        texture.GetGuestMipsSize() != 0) {
      ++counts.incompatible;
      if (REXCVAR_GET(gpu_texture_load_report)) {
        // Each kind once (a debug report): what the texture is against what the resolve wrote.
        static std::set<uint64_t> logged;
        const uint64_t kind = uint64_t(key.format) | (uint64_t(write.format) << 6) |
                              (uint64_t(key.pitch) << 12) | (uint64_t(write.pitch) << 21) |
                              (uint64_t(key.mip_max_level) << 30) | (uint64_t(key.dimension) << 34) |
                              (uint64_t(key.tiled) << 36) | (uint64_t(key.packed_mips) << 37) |
                              (uint64_t(key.endianness) << 38) | (uint64_t(write.endianness) << 40) |
                              (uint64_t(key.scaled_resolve) << 42) |
                              (uint64_t(key.depth_or_array_size_minus_1 != 0) << 43) |
                              (uint64_t(key.signed_separate) << 44) |
                              (uint64_t(texture.GetGuestMipsSize() != 0) << 45);
        if (logged.size() < 64 && logged.insert(kind).second) {
          REXGPU_INFO(
              "NATIVERESOLVE incompatible: texture format {} {}x{} pitch {} mips {} dim {} tiled {} "
              "packed {} endian {} scaled {} array {} signed {} guest mips {}; resolve format {} pitch {} "
              "endian {} {}x{}",
              uint32_t(key.format), key.GetWidth(), key.GetHeight(), uint32_t(key.pitch),
              uint32_t(key.mip_max_level), uint32_t(key.dimension), uint32_t(key.tiled),
              uint32_t(key.packed_mips), uint32_t(key.endianness), uint32_t(key.scaled_resolve),
              key.depth_or_array_size_minus_1, uint32_t(key.signed_separate),
              texture.GetGuestMipsSize(), uint32_t(write.format), write.pitch,
              uint32_t(write.endianness), write.width, write.height);
        }
      }
      return;
    }
    uint32_t texture_width = key.GetWidth();
    uint32_t texture_height = key.GetHeight();
    // A texture the resolve covers completely gets all its texels from it: what it held before
    // doesn't matter, so it may be written even if it wasn't up to date (memory resolved into with
    // two formats in a frame outdates each format's texture every time).
    const bool covered = !write.dest_x && !write.dest_y && write.width >= texture_width &&
                         write.height >= texture_height;
    if (!texture.base_up_to_date_before_resolve() && !covered) {
      ++counts.not_up_to_date;
      return;
    }
    auto& d3d12_texture = static_cast<D3D12Texture&>(texture);
    // Sampling goes to the replacement resource, and the reload decides whether to keep it.
    if (d3d12_texture.replaced_mip_levels() || !d3d12_texture.resource()) {
      ++counts.replacement;
      return;
    }
    if (write.dest_x >= texture_width || write.dest_y >= texture_height) {
      ++counts.outside;
      return;
    }
    if (!texture.base_up_to_date_before_resolve()) {
      ++counts.covered;
    }
    uint32_t width = std::min(write.width, texture_width - write.dest_x);
    uint32_t height = std::min(write.height, texture_height - write.dest_y);
    ID3D12Resource* texture_resource = d3d12_texture.resource();

    if (write.mode != NativeResolveMode::kCopy) {
      ID3D12PipelineState* pipeline =
          write.source_samples ? native_resolve_ms_pipeline_.Get() : native_resolve_pipeline_.Get();
      if (!pipeline) {
        ++counts.no_pipeline;
        return;
      }
      // Created before gpu_native_resolves was enabled, or not matching the CreateTexture
      // conditions.
      if (!(texture_resource->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        ++counts.no_uav;
        return;
      }
      ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[2];
      if (!command_processor_.RequestOneUseSingleViewDescriptors(
              uint32_t(rex::countof(descriptors)), descriptors)) {
        ++counts.descriptors;
        return;
      }
      ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
      if (write.source_srv_format == DXGI_FORMAT_UNKNOWN) {
        device->CopyDescriptorsSimple(1, descriptors[0].first, write.source_srv,
                                      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
        srv_desc.Format = write.source_srv_format;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (write.source_samples) {
          srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
          srv_desc.Texture2DMS.UnusedField_NothingToDefine = 0;
        } else {
          srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
          srv_desc.Texture2D.MostDetailedMip = 0;
          srv_desc.Texture2D.MipLevels = 1;
          srv_desc.Texture2D.PlaneSlice = 0;
          srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
        }
        device->CreateShaderResourceView(write.source, &srv_desc, descriptors[0].first);
      }
      D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;
      uav_desc.Format = write.dest_uav_format;
      uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      uav_desc.Texture2D.MipSlice = 0;
      uav_desc.Texture2D.PlaneSlice = 0;
      device->CreateUnorderedAccessView(texture_resource, nullptr, &uav_desc,
                                        descriptors[1].first);

      // Referenced by the command list now.
      texture.MarkAsUsed();
      command_processor_.PushTransitionBarrier(write.source, source_state,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      source_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      D3D12_RESOURCE_STATES texture_state =
          d3d12_texture.SetResourceState(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      command_processor_.PushTransitionBarrier(texture_resource, texture_state,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      command_processor_.SubmitBarriers();

      DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
      command_processor_.SetExternalPipeline(pipeline);
      command_list.D3DSetComputeRootSignature(native_resolve_root_signature_.Get());
      command_list.D3DSetComputeRootDescriptorTable(1, descriptors[0].second);
      command_list.D3DSetComputeRootDescriptorTable(2, descriptors[1].second);
      NativeResolveConstants constants;
      constants.mode = uint32_t(write.mode);
      constants.scale = write.scale;
      constants.source_offset[0] = int32_t(write.source_x * scale_x);
      constants.source_offset[1] = int32_t(write.source_y * scale_y);
      constants.dest_offset[0] = int32_t(write.dest_x * scale_x);
      constants.dest_offset[1] = int32_t(write.dest_y * scale_y);
      constants.size[0] = width * scale_x;
      constants.size[1] = height * scale_y;
      constants.fill[0] = write.fill_x;
      constants.fill[1] = write.fill_y;
      constants.samples = write.source_samples;
      command_list.D3DSetComputeRoot32BitConstants(0, sizeof(constants) / sizeof(uint32_t),
                                                   &constants, 0);
      command_list.D3DDispatch((constants.size[0] + 7) >> 3, (constants.size[1] + 7) >> 3, 1);

      // Leave the texture in the state its users expect it to be in.
      command_processor_.PushUAVBarrier(texture_resource);
      command_processor_.PushTransitionBarrier(
          texture_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, texture_state);
      d3d12_texture.SetResourceState(texture_state);

      MakeTextureUpToDateAndWatch(texture);
      ++counts.written;
      note_mirror(d3d12_texture);
      return;
    }

    // The half-pixel offset gap fill and a multisampled source need a shader mode.
    assert_true(!write.fill_x && !write.fill_y && !write.source_samples);

    // Referenced by the command list now.
    texture.MarkAsUsed();
    command_processor_.PushTransitionBarrier(write.source, source_state,
                                             D3D12_RESOURCE_STATE_COPY_SOURCE);
    source_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    D3D12_RESOURCE_STATES texture_state =
        d3d12_texture.SetResourceState(D3D12_RESOURCE_STATE_COPY_DEST);
    command_processor_.PushTransitionBarrier(texture_resource, texture_state,
                                             D3D12_RESOURCE_STATE_COPY_DEST);
    command_processor_.SubmitBarriers();

    D3D12_TEXTURE_COPY_LOCATION location_dest;
    location_dest.pResource = texture_resource;
    location_dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location_dest.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION location_source;
    location_source.pResource = write.source;
    location_source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location_source.SubresourceIndex = 0;
    D3D12_BOX source_box;
    source_box.left = write.source_x * scale_x;
    source_box.top = write.source_y * scale_y;
    source_box.front = 0;
    source_box.right = (write.source_x + width) * scale_x;
    source_box.bottom = (write.source_y + height) * scale_y;
    source_box.back = 1;
    command_processor_.GetDeferredCommandList().D3DCopyTextureRegion(
        &location_dest, write.dest_x * scale_x, write.dest_y * scale_y, 0, &location_source,
        &source_box);

    // Leave the texture in the state its users expect it to be in.
    command_processor_.PushTransitionBarrier(texture_resource, D3D12_RESOURCE_STATE_COPY_DEST,
                                             texture_state);
    d3d12_texture.SetResourceState(texture_state);

    MakeTextureUpToDateAndWatch(texture);
    ++counts.written;
    note_mirror(d3d12_texture);
  });
  // gpu_lazy_resolve_copies: textures written, none holding the whole rectangle - which sizes they
  // have against the rectangle, each kind once (16 at most).
  if (counts.written && !counts.mirror && REXCVAR_GET(gpu_texture_load_report)) {
    static std::set<uint64_t> logged;
    const uint64_t kind = uint64_t(write.base_page) ^ (uint64_t(write.dest_x) << 20) ^
                          (uint64_t(write.dest_y) << 32) ^ (uint64_t(write.width) << 44) ^
                          (uint64_t(write.height) << 54);
    if (logged.size() < 16 && logged.insert(kind).second) {
      std::string sizes;
      ForEachTextureAtBasePage(write.base_page, [&](Texture& texture) {
        const TextureKey& key = texture.key();
        sizes += fmt::format(" {}x{} f{} p{}", key.GetWidth(), key.GetHeight(), uint32_t(key.format),
                             uint32_t(key.pitch));
      });
      REXGPU_INFO("LAZYRESOLVE no mirror: base {:05X} rect {},{} {}x{} format {} pitch {}; textures:{}",
                  write.base_page, write.dest_x, write.dest_y, write.width, write.height,
                  uint32_t(write.format), write.pitch, sizes);
    }
  }
}

bool D3D12TextureCache::RebuildLazyResolveCopy(const LazyResolveCopy& copy, D3D12Texture& mirror) {
  // A replacement (texture_replace_dir) would be read instead of the guest's data.
  if (!IsLazyResolveRebuildAvailable(copy.unscaled) || mirror.replaced_mip_levels() ||
      !mirror.resource() || !copy.group_count_x || !copy.group_count_y) {
    return false;
  }
  D3D12SharedMemory& d3d12_shared_memory = static_cast<D3D12SharedMemory&>(shared_memory());
  if (!copy.unscaled &&
      (!EnsureScaledResolveMemoryCommitted(copy.extent_start, copy.extent_length) ||
       !MakeScaledResolveRangeCurrent(copy.dest_base, copy.current_length))) {
    return false;
  }
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[2];
  if (!command_processor_.RequestOneUseSingleViewDescriptors(uint32_t(rex::countof(descriptors)),
                                                             descriptors)) {
    return false;
  }
  // The destination as the copy writes it (uint4 elements, resolve_copy_shader_info's
  // dest_bpe_log2 of the fast 32bpp copy): the current scaled resolve range, or all of shared
  // memory (the unscaled copy's constants address it from dest_base, as the copy does). The
  // range was made resident when the copy was skipped (RequestRange there).
  if (copy.unscaled) {
    d3d12_shared_memory.WriteUintPow2UAVDescriptor(descriptors[0].first, 4);
    d3d12_shared_memory.UseForWriting();
  } else {
    CreateCurrentScaledResolveRangeUintPow2UAV(descriptors[0].first, 4);
    TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
  srv_desc.Format = copy.srv_format;
  srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc.Texture2D.MostDetailedMip = 0;
  srv_desc.Texture2D.MipLevels = 1;
  srv_desc.Texture2D.PlaneSlice = 0;
  srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
  ID3D12Resource* mirror_resource = mirror.resource();
  command_processor_.GetD3D12Provider().GetDevice()->CreateShaderResourceView(
      mirror_resource, &srv_desc, descriptors[1].first);
  // Referenced by the command list now.
  mirror.MarkAsUsed();
  const D3D12_RESOURCE_STATES mirror_state =
      mirror.SetResourceState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  command_processor_.PushTransitionBarrier(mirror_resource, mirror_state,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  command_processor_.SubmitBarriers();
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  command_processor_.SetExternalPipeline(copy.unscaled ? lazy_rebuild_unscaled_pipeline_
                                                       : lazy_rebuild_pipeline_);
  command_list.D3DSetComputeRootSignature(lazy_rebuild_root_signature_);
  // As the direct resolve colour root signature lays them out: constants, destination, source.
  command_list.D3DSetComputeRootDescriptorTable(1, descriptors[0].second);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptors[1].second);
  // The first of the two direct resolve constants: the mirror's size in host pixels (the rebuild
  // writes nothing past it); an unscaled mirror is an unscaled texture.
  std::array<uint32_t, 8> root_constants = copy.root_constants;
  const TextureKey& mirror_key = mirror.key();
  const uint32_t mirror_scale_x = copy.unscaled ? 1 : draw_resolution_scale_x();
  const uint32_t mirror_scale_y = copy.unscaled ? 1 : draw_resolution_scale_y();
  root_constants[copy.root_constant_count - 2] =
      std::min(mirror_key.GetWidth() * mirror_scale_x, uint32_t(0xFFFF)) |
      (std::min(mirror_key.GetHeight() * mirror_scale_y, uint32_t(0xFFFF)) << 16);
  command_list.D3DSetComputeRoot32BitConstants(0, copy.root_constant_count, root_constants.data(),
                                               0);
  command_list.D3DDispatch(copy.group_count_x, copy.group_count_y, 1);
  if (copy.unscaled) {
    d3d12_shared_memory.MarkUAVWritesCommitNeeded();
  } else {
    MarkCurrentScaledResolveRangeUAVWritesCommitNeeded();
  }
  // Leave the mirror in the state its users expect it to be in.
  command_processor_.PushTransitionBarrier(mirror_resource,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           mirror_state);
  mirror.SetResourceState(mirror_state);
  return true;
}

void D3D12TextureCache::PrepareLazyResolveCopiesForResolve(const LazyResolveCopy& resolve) {
  const uint64_t end = uint64_t(resolve.extent_start) + resolve.extent_length;
  for (size_t i = 0; i < lazy_resolve_copies_.size();) {
    PendingLazyResolveCopy& pending = lazy_resolve_copies_[i];
    const uint64_t pending_end = uint64_t(pending.copy.extent_start) + pending.copy.extent_length;
    if (pending.copy.extent_start >= end || resolve.extent_start >= pending_end) {
      ++i;
      continue;
    }
    // A resolve of the other kind (scaled or not) at the same place writes the other buffer, so
    // it never makes a pending copy moot (Covers compares the kinds), but it's rebuilt first.
    if (resolve.Covers(pending.copy)) {
      // Overwritten completely by the resolve, by its copy or by its own pending one.
      ++lazy_dropped_;
    } else if (RebuildLazyResolveCopy(pending.copy, *pending.mirror)) {
      ++lazy_rebuilt_;
    } else {
      ++lazy_failed_;
    }
    EraseLazyResolveCopy(i);
  }
  ReportLazyResolveCopies();
}

bool D3D12TextureCache::AddLazyResolveCopy(const LazyResolveCopy& copy) {
  if (!last_native_resolve_mirror_ || !IsLazyResolveRebuildAvailable(copy.unscaled)) {
    return false;
  }
  lazy_resolve_copies_.push_back({copy, last_native_resolve_mirror_});
  ++lazy_skipped_;
  if (copy.unscaled) {
    ++lazy_unscaled_pending_;
    ++lazy_skipped_unscaled_;
    lazy_unscaled_start_ = std::min(lazy_unscaled_start_, copy.extent_start);
    lazy_unscaled_end_ = std::max(lazy_unscaled_end_, copy.extent_start + copy.extent_length);
  }
  ReportLazyResolveCopies();
  return true;
}

void D3D12TextureCache::EraseLazyResolveCopy(size_t i) {
  const bool unscaled = lazy_resolve_copies_[i].copy.unscaled;
  lazy_resolve_copies_.erase(lazy_resolve_copies_.begin() + i);
  if (!unscaled) {
    return;
  }
  assert_not_zero(lazy_unscaled_pending_);
  --lazy_unscaled_pending_;
  // The span of the ones left (a few at a time, so recounted).
  lazy_unscaled_start_ = UINT32_MAX;
  lazy_unscaled_end_ = 0;
  for (const PendingLazyResolveCopy& pending : lazy_resolve_copies_) {
    if (pending.copy.unscaled) {
      lazy_unscaled_start_ = std::min(lazy_unscaled_start_, pending.copy.extent_start);
      lazy_unscaled_end_ =
          std::max(lazy_unscaled_end_, pending.copy.extent_start + pending.copy.extent_length);
    }
  }
}

void D3D12TextureCache::MaterializeLazyResolveCopies(uint32_t start, uint32_t length,
                                                     uint32_t kinds) {
  if (lazy_resolve_copies_.empty() || !length) {
    return;
  }
  const uint64_t end = uint64_t(start) + length;
  // Unscaled ones only (shared memory's readers, every draw's vertex buffers): outside their span
  // nothing is pending.
  if (kinds == kLazyResolveUnscaled &&
      (!lazy_unscaled_pending_ || start >= lazy_unscaled_end_ || end <= lazy_unscaled_start_)) {
    return;
  }
  for (size_t i = 0; i < lazy_resolve_copies_.size();) {
    PendingLazyResolveCopy& pending = lazy_resolve_copies_[i];
    const uint64_t pending_end = uint64_t(pending.copy.extent_start) + pending.copy.extent_length;
    const uint32_t kind = pending.copy.unscaled ? kLazyResolveUnscaled : kLazyResolveScaled;
    if (!(kinds & kind) || pending.copy.extent_start >= end || start >= pending_end) {
      ++i;
      continue;
    }
    if (RebuildLazyResolveCopy(pending.copy, *pending.mirror)) {
      ++lazy_rebuilt_;
    } else {
      ++lazy_failed_;
    }
    EraseLazyResolveCopy(i);
  }
  ReportLazyResolveCopies();
}

void D3D12TextureCache::BeforeSharedMemoryRequest(void* context,
                                                  const std::pair<uint32_t, uint32_t>* ranges,
                                                  size_t count, bool suppressed) {
  auto& cache = *static_cast<D3D12TextureCache*>(context);
  if (!cache.lazy_unscaled_pending_) {
    return;
  }
  for (size_t r = 0; r < count; ++r) {
    if (suppressed) {
      // A draw's bindings are set up: nothing may be dispatched (the draw wrote its vertex and
      // memexport ranges' copies before). Only count what is still pending there.
      const uint64_t end = uint64_t(ranges[r].first) + ranges[r].second;
      if (ranges[r].first >= cache.lazy_unscaled_end_ || end <= cache.lazy_unscaled_start_) {
        continue;
      }
      for (const PendingLazyResolveCopy& pending : cache.lazy_resolve_copies_) {
        if (pending.copy.unscaled && pending.copy.extent_start < end &&
            ranges[r].first < uint64_t(pending.copy.extent_start) + pending.copy.extent_length) {
          ++cache.lazy_late_;
        }
      }
      continue;
    }
    cache.MaterializeLazyResolveCopies(ranges[r].first, ranges[r].second, kLazyResolveUnscaled);
    if (!cache.lazy_unscaled_pending_) {
      break;
    }
  }
}

bool D3D12TextureCache::RebuildFromLastMirror(const LazyResolveCopy& copy) {
  return last_native_resolve_mirror_ && RebuildLazyResolveCopy(copy, *last_native_resolve_mirror_);
}

void D3D12TextureCache::ReportLazyResolveCopies() {
  const uint64_t now_ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
  if (!lazy_last_report_ms_) {
    lazy_last_report_ms_ = now_ms;
    return;
  }
  if (now_ms - lazy_last_report_ms_ < 5000) {
    return;
  }
  REXGPU_INFO(
      "LAZYRESOLVE skipped {} rebuilt {} dropped {} failed {} pending {} in {} ms; unscaled skipped "
      "{} pending {} late {}",
      lazy_skipped_, lazy_rebuilt_, lazy_dropped_, lazy_failed_, lazy_resolve_copies_.size(),
      now_ms - lazy_last_report_ms_, lazy_skipped_unscaled_, lazy_unscaled_pending_, lazy_late_);
  lazy_skipped_ = 0;
  lazy_skipped_unscaled_ = 0;
  lazy_rebuilt_ = 0;
  lazy_dropped_ = 0;
  lazy_failed_ = 0;
  lazy_late_ = 0;
  lazy_last_report_ms_ = now_ms;
}

void D3D12TextureCache::CreateCurrentScaledResolveRangeUintPow2SRV(
    D3D12_CPU_DESCRIPTOR_HANDLE handle, uint32_t element_size_bytes_pow2) {
  assert_true(IsDrawResolutionScaled());
  size_t buffer_index = GetCurrentScaledResolveBufferIndex();
  const ScaledResolveVirtualBuffer* buffer = scaled_resolve_2gb_buffers_[buffer_index].get();
  assert_not_null(buffer);
  ui::d3d12::util::CreateBufferTypedSRV(
      command_processor_.GetD3D12Provider().GetDevice(), handle, buffer->resource(),
      ui::d3d12::util::GetUintPow2DXGIFormat(element_size_bytes_pow2),
      uint32_t(scaled_resolve_current_range_length_scaled_ >> element_size_bytes_pow2),
      (scaled_resolve_current_range_start_scaled_ - (uint64_t(buffer_index) << 30)) >>
          element_size_bytes_pow2);
}

void D3D12TextureCache::CreateCurrentScaledResolveRangeUintPow2UAV(
    D3D12_CPU_DESCRIPTOR_HANDLE handle, uint32_t element_size_bytes_pow2) {
  assert_true(IsDrawResolutionScaled());
  size_t buffer_index = GetCurrentScaledResolveBufferIndex();
  const ScaledResolveVirtualBuffer* buffer = scaled_resolve_2gb_buffers_[buffer_index].get();
  assert_not_null(buffer);
  ui::d3d12::util::CreateBufferTypedUAV(
      command_processor_.GetD3D12Provider().GetDevice(), handle, buffer->resource(),
      ui::d3d12::util::GetUintPow2DXGIFormat(element_size_bytes_pow2),
      uint32_t(scaled_resolve_current_range_length_scaled_ >> element_size_bytes_pow2),
      (scaled_resolve_current_range_start_scaled_ - (uint64_t(buffer_index) << 30)) >>
          element_size_bytes_pow2);
}

ID3D12Resource* D3D12TextureCache::RequestSwapTexture(D3D12_SHADER_RESOURCE_VIEW_DESC& srv_desc_out,
                                                      xenos::TextureFormat& format_out,
                                                      uint32_t* width_unscaled_out,
                                                      uint32_t* height_unscaled_out) {
  const auto& regs = register_file();
  xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(0);
  TextureKey key;
  BindingInfoFromFetchConstant(fetch, key, nullptr);
  if (!key.is_valid || key.base_page == 0 || key.dimension != xenos::DataDimension::k2DOrStacked) {
    return nullptr;
  }
  D3D12Texture* texture = static_cast<D3D12Texture*>(FindOrCreateTexture(key));
  if (texture == nullptr || !LoadTextureData(*texture)) {
    return nullptr;
  }
  texture->MarkAsUsed();
  // The swap texture is likely to be used only for the presentation compute
  // shader, and not during emulation, where it'd be NON_PIXEL_SHADER_RESOURCE |
  // PIXEL_SHADER_RESOURCE.
  ID3D12Resource* texture_resource = texture->resource();
  command_processor_.PushTransitionBarrier(
      texture_resource, texture->SetResourceState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  srv_desc_out.Format = GetDXGIUnormFormat(key);
  srv_desc_out.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc_out.Shader4ComponentMapping =
      GuestToHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(key)) |
      D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES;
  srv_desc_out.Texture2D.MostDetailedMip = 0;
  srv_desc_out.Texture2D.MipLevels = 1;
  srv_desc_out.Texture2D.PlaneSlice = 0;
  srv_desc_out.Texture2D.ResourceMinLODClamp = 0.0f;
  // Only texture->key, not the result of BindingInfoFromFetchConstant, contains
  // whether the texture is scaled.
  key = texture->key();
  if (width_unscaled_out) {
    *width_unscaled_out = key.GetWidth();
  }
  if (height_unscaled_out) {
    *height_unscaled_out = key.GetHeight();
  }
  format_out = key.format;
  return texture_resource;
}

D3D12TextureCache::D3D12Texture::D3D12Texture(D3D12TextureCache& texture_cache,
                                              const TextureKey& key, ID3D12Resource* resource,
                                              D3D12_RESOURCE_STATES resource_state,
                                              bool track_usage)
    : Texture(texture_cache, key, track_usage),
      resource_(resource),
      resource_state_(resource_state) {
  ID3D12Device* device = texture_cache.command_processor_.GetD3D12Provider().GetDevice();
  D3D12_RESOURCE_DESC resource_desc = resource_->GetDesc();
  SetHostMemoryUsage(device->GetResourceAllocationInfo(0, 1, &resource_desc).SizeInBytes);
}

void D3D12TextureCache::D3D12Texture::SetReplacement(Microsoft::WRL::ComPtr<ID3D12Resource> resource,
                                                     D3D12_RESOURCE_STATES state,
                                                     Microsoft::WRL::ComPtr<ID3D12Resource> upload,
                                                     uint32_t mip_levels, bool in_use) {
  auto& d3d12_texture_cache = static_cast<D3D12TextureCache&>(texture_cache());
  for (const auto& descriptor_pair : srv_descriptors_) {
    // Frames under way may still read through them.
    if (in_use) {
      retired_srv_descriptors_.push_back(descriptor_pair.second);
    } else {
      d3d12_texture_cache.ReleaseTextureDescriptor(descriptor_pair.second);
    }
  }
  srv_descriptors_.clear();
  if (in_use) replaced_original_ = std::move(resource_);
  resource_ = std::move(resource);
  resource_state_ = state;
  replacement_upload_ = std::move(upload);
  replaced_mip_levels_ = mip_levels;
  ID3D12Device* device = d3d12_texture_cache.command_processor_.GetD3D12Provider().GetDevice();
  D3D12_RESOURCE_DESC resource_desc = resource_->GetDesc();
  SetHostMemoryUsage(device->GetResourceAllocationInfo(0, 1, &resource_desc).SizeInBytes);
}

D3D12TextureCache::D3D12Texture::~D3D12Texture() {
  auto& d3d12_texture_cache = static_cast<D3D12TextureCache&>(texture_cache());
  for (uint32_t descriptor_index : retired_srv_descriptors_) {
    d3d12_texture_cache.ReleaseTextureDescriptor(descriptor_index);
  }
  for (const auto& descriptor_pair : srv_descriptors_) {
    d3d12_texture_cache.ReleaseTextureDescriptor(descriptor_pair.second);
  }
}

bool D3D12TextureCache::IsDecompressionNeeded(xenos::TextureFormat format, uint32_t width,
                                              uint32_t height) const {
  DXGI_FORMAT dxgi_format_uncompressed = host_formats_[uint32_t(format)].dxgi_format_uncompressed;
  if (dxgi_format_uncompressed == DXGI_FORMAT_UNKNOWN) {
    return false;
  }
  const FormatInfo* format_info = FormatInfo::Get(format);
  if (!(width & (format_info->block_width - 1)) && !(height & (format_info->block_height - 1))) {
    return false;
  }
  // UnalignedBlockTexturesSupported is for block-compressed textures with the
  // block size of 4x4, but not for 2x1 (4:2:2) subsampled formats.
  if (format_info->block_width == 4 && format_info->block_height == 4 &&
      command_processor_.GetD3D12Provider().AreUnalignedBlockTexturesSupported()) {
    return false;
  }
  return true;
}

TextureCache::LoadShaderIndex D3D12TextureCache::GetLoadShaderIndex(TextureKey key) const {
  const HostFormat& host_format = host_formats_[uint32_t(key.format)];
  if (key.signed_separate) {
    return host_format.load_shader_signed;
  }
  if (IsDecompressionNeeded(key.format, key.GetWidth(), key.GetHeight())) {
    return host_format.load_shader_decompress;
  }
  return host_format.load_shader;
}

bool D3D12TextureCache::IsSignedVersionSeparateForFormat(TextureKey key) const {
  const HostFormat& host_format = host_formats_[uint32_t(key.format)];
  return host_format.load_shader_signed != kLoadShaderIndexUnknown &&
         host_format.load_shader_signed != host_format.load_shader;
}

bool D3D12TextureCache::IsScaledResolveSupportedForFormat(TextureKey key) const {
  LoadShaderIndex load_shader = GetLoadShaderIndex(key);
  return load_shader != kLoadShaderIndexUnknown && load_pipelines_scaled_[load_shader] != nullptr;
}

uint32_t D3D12TextureCache::GetHostFormatSwizzle(TextureKey key) const {
  // Dense cache-line-aligned swizzle array avoids cache misses from accessing
  // the full HostFormat struct on every texture fetch.
  alignas(64) static const auto swizzle_cache = []() {
    std::array<uint16_t, 64> arr{};
    for (int i = 0; i < 64; ++i) {
      arr[i] = static_cast<uint16_t>(host_formats_[i].swizzle);
    }
    return arr;
  }();
  return swizzle_cache[uint32_t(key.format)];
}

uint32_t D3D12TextureCache::GetMaxHostTextureWidthHeight(xenos::DataDimension dimension) const {
  switch (dimension) {
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      // 1D and 2D are emulated as 2D arrays.
      return D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    case xenos::DataDimension::k3D:
      return D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    case xenos::DataDimension::kCube:
      return D3D12_REQ_TEXTURECUBE_DIMENSION;
    default:
      assert_unhandled_case(dimension);
      return 0;
  }
}

uint32_t D3D12TextureCache::GetMaxHostTextureDepthOrArraySize(
    xenos::DataDimension dimension) const {
  switch (dimension) {
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      // 1D and 2D are emulated as 2D arrays.
      return D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
    case xenos::DataDimension::k3D:
      return D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    case xenos::DataDimension::kCube:
      return D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION / 6 * 6;
    default:
      assert_unhandled_case(dimension);
      return 0;
  }
}

std::unique_ptr<TextureCache::Texture> D3D12TextureCache::CreateTexture(TextureKey key) {
  D3D12_RESOURCE_DESC desc;
  desc.Format = GetDXGIResourceFormat(key);
  if (desc.Format == DXGI_FORMAT_UNKNOWN) {
    unsupported_format_features_used_[uint32_t(key.format)] |= kUnsupportedResourceBit;
    return nullptr;
  }
  if (key.dimension == xenos::DataDimension::k3D) {
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  } else {
    // 1D textures are treated as 2D for simplicity.
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  }
  desc.Alignment = 0;
  desc.Width = key.GetWidth();
  desc.Height = key.GetHeight();
  if (key.scaled_resolve) {
    desc.Width *= draw_resolution_scale_x();
    desc.Height *= draw_resolution_scale_y();
  }
  desc.DepthOrArraySize = key.GetDepthOrArraySize();
  desc.MipLevels = key.mip_max_level + 1;
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  // Untiling through a buffer instead of using unordered access because copying
  // is not done that often.
  desc.Flags = D3D12_RESOURCE_FLAG_NONE;
  if (REXCVAR_GET(gpu_native_resolves) && key.tiled &&
      key.dimension == xenos::DataDimension::k2DOrStacked && key.depth_or_array_size_minus_1 == 0 &&
      key.mip_max_level == 0 &&
      (key.format == xenos::TextureFormat::k_16_16_FLOAT ||
       key.format == xenos::TextureFormat::k_16_16_16_16_FLOAT ||
       key.format == xenos::TextureFormat::k_8_8_8_8 ||
       key.format == xenos::TextureFormat::k_2_10_10_10 ||
       key.format == xenos::TextureFormat::k_32_FLOAT ||
       key.format == xenos::TextureFormat::k_24_8 ||
       key.format == xenos::TextureFormat::k_24_8_FLOAT)) {
    // Native resolve destinations written by native_resolve_cs (WriteNativeResolveToTextures).
    desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  // Assuming untiling will be the next operation.
  D3D12_RESOURCE_STATES resource_state = D3D12_RESOURCE_STATE_COPY_DEST;
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             provider.GetHeapFlagCreateNotZeroed(), &desc,
                                             resource_state, nullptr, IID_PPV_ARGS(&resource)))) {
    return nullptr;
  }
  return std::unique_ptr<Texture>(new D3D12Texture(*this, key, resource.Get(), resource_state));
}

bool D3D12TextureCache::LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                                              bool load_mips) {
  D3D12Texture& d3d12_texture = static_cast<D3D12Texture&>(texture);
  // A replacement (texture_replace_dir) stays: the guest's data doesn't go into it.
  if (d3d12_texture.replaced_mip_levels()) {
    return true;
  }
  if (load_base && !d3d12_texture.files_checked) {
    d3d12_texture.files_checked = true;
    if (DumpOrReplaceTexture(d3d12_texture)) {
      return true;
    }
  }
  TextureKey texture_key = d3d12_texture.key();

  // gpu_lazy_resolve_copies: the scaled resolve memory this load reads holds the skipped copies
  // first (from their mirrors, which may be this texture: it's only overwritten below).
  // (Unscaled ones: shared memory's RequestRanges hook, before the load requested its ranges.)
  if (texture_key.scaled_resolve && !lazy_resolve_copies_.empty()) {
    if (load_base) {
      MaterializeLazyResolveCopies(texture_key.base_page << 12, d3d12_texture.GetGuestBaseSize(),
                                   kLazyResolveScaled);
    }
    if (load_mips) {
      MaterializeLazyResolveCopies(texture_key.mip_page << 12, d3d12_texture.GetGuestMipsSize(),
                                   kLazyResolveScaled);
    }
  }

  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();

  // Get the pipeline.
  LoadShaderIndex load_shader = GetLoadShaderIndex(texture_key);
  if (load_shader == kLoadShaderIndexUnknown) {
    return false;
  }
  bool texture_resolution_scaled = texture_key.scaled_resolve;
  ID3D12PipelineState* pipeline = texture_resolution_scaled
                                      ? load_pipelines_scaled_[load_shader].Get()
                                      : load_pipelines_[load_shader].Get();
  if (pipeline == nullptr) {
    return false;
  }
  const LoadShaderInfo& load_shader_info = GetLoadShaderInfo(load_shader);

  // Get the guest layout.
  const texture_util::TextureGuestLayout& guest_layout = d3d12_texture.guest_layout();
  xenos::DataDimension dimension = texture_key.dimension;
  bool is_3d = dimension == xenos::DataDimension::k3D;
  bool is_3d_tiling = is_3d || d3d12_texture.force_load_3d_tiling();
  uint32_t width = texture_key.GetWidth();
  uint32_t height = texture_key.GetHeight();
  uint32_t depth_or_array_size = texture_key.GetDepthOrArraySize();
  uint32_t depth = is_3d ? depth_or_array_size : 1;
  uint32_t array_size = is_3d ? 1 : depth_or_array_size;
  xenos::TextureFormat guest_format = texture_key.format;
  const FormatInfo* guest_format_info = FormatInfo::Get(guest_format);
  uint32_t block_width = guest_format_info->block_width;
  uint32_t block_height = guest_format_info->block_height;
  uint32_t bytes_per_block = guest_format_info->bytes_per_block();
  uint32_t level_first = load_base ? 0 : 1;
  uint32_t level_last = load_mips ? texture_key.mip_max_level : 0;
  assert_true(level_first <= level_last);
  uint32_t level_packed = guest_layout.packed_level;
  uint32_t level_stored_first = std::min(level_first, level_packed);
  uint32_t level_stored_last = std::min(level_last, level_packed);
  uint32_t texture_resolution_scale_x = texture_resolution_scaled ? draw_resolution_scale_x() : 1;
  uint32_t texture_resolution_scale_y = texture_resolution_scaled ? draw_resolution_scale_y() : 1;

  // The loop counter can mean two things depending on whether the packed mip
  // tail is stored as mip 0, because in this case, it would be ambiguous since
  // both the base and the mips would be on "level 0", but stored in separate
  // places.
  uint32_t loop_level_first, loop_level_last;
  if (level_packed == 0) {
    // Packed mip tail is the level 0 - may need to load mip tails for the base,
    // the mips, or both.
    // Loop iteration 0 - base packed mip tail.
    // Loop iteration 1 - mips packed mip tail.
    loop_level_first = uint32_t(level_first != 0);
    loop_level_last = uint32_t(level_last != 0);
  } else {
    // Packed mip tail is not the level 0.
    // Loop iteration is the actual level being loaded.
    loop_level_first = level_stored_first;
    loop_level_last = level_stored_last;
  }

  // Get the host layout and the buffer.
  bool host_block_compressed = host_formats_[uint32_t(guest_format)].is_block_compressed &&
                               !IsDecompressionNeeded(guest_format, width, height);
  uint32_t host_block_width = host_block_compressed ? block_width : 1;
  uint32_t host_block_height = host_block_compressed ? block_height : 1;
  uint32_t host_x_blocks_per_thread = UINT32_C(1)
                                      << load_shader_info.guest_x_blocks_per_thread_log2;
  if (!host_block_compressed) {
    // Decompressing guest blocks.
    host_x_blocks_per_thread *= block_width;
  }
  UINT64 copy_buffer_size = 0;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT host_slice_layout_base;
  UINT64 host_slice_size_base;
  // Indexing is the same as for guest stored mips:
  // 1...min(level_last, level_packed) if level_packed is not 0, or only 0 if
  // level_packed == 0.
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT
  host_slice_layouts_mips[xenos::kTextureMaxMips];
  UINT64 host_slice_sizes_mips[xenos::kTextureMaxMips];
  // Using custom calculations instead of GetCopyableFootprints because
  // shaders may unconditionally copy multiple blocks along X per thread for
  // simplicity, to make sure all rows (also including the last one -
  // GetCopyableFootprints aligns row offsets, but not the total size) are
  // properly padded to the number of blocks copied in an invocation without
  // implicit assumptions about D3D12_TEXTURE_DATA_PITCH_ALIGNMENT.
  DXGI_FORMAT host_copy_format = GetDXGIResourceFormat(guest_format, width, height);
  for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
    bool is_base = loop_level == 0;
    uint32_t level = (level_packed == 0) ? 0 : loop_level;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT& level_host_slice_layout =
        is_base ? host_slice_layout_base : host_slice_layouts_mips[level];
    level_host_slice_layout.Offset = copy_buffer_size;
    level_host_slice_layout.Footprint.Format = host_copy_format;
    if (level == level_packed) {
      // Loading the packed tail for the base or the mips - load the whole tail
      // to copy regions out of it.
      const texture_util::TextureGuestLayout::Level& guest_layout_packed =
          is_base ? guest_layout.base : guest_layout.mips[level];
      level_host_slice_layout.Footprint.Width = guest_layout_packed.x_extent_blocks * block_width;
      level_host_slice_layout.Footprint.Height = guest_layout_packed.y_extent_blocks * block_height;
      level_host_slice_layout.Footprint.Depth = guest_layout_packed.z_extent;
    } else {
      level_host_slice_layout.Footprint.Width = std::max(width >> level, uint32_t(1));
      level_host_slice_layout.Footprint.Height = std::max(height >> level, uint32_t(1));
      level_host_slice_layout.Footprint.Depth = std::max(depth >> level, uint32_t(1));
    }
    level_host_slice_layout.Footprint.Width =
        rex::round_up(level_host_slice_layout.Footprint.Width * texture_resolution_scale_x,
                      UINT(host_block_width));
    level_host_slice_layout.Footprint.Height =
        rex::round_up(level_host_slice_layout.Footprint.Height * texture_resolution_scale_y,
                      UINT(host_block_height));
    level_host_slice_layout.Footprint.RowPitch =
        rex::align(rex::round_up(level_host_slice_layout.Footprint.Width / host_block_width,
                                 host_x_blocks_per_thread) *
                       load_shader_info.bytes_per_host_block,
                   uint32_t(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
    UINT64 level_host_slice_size =
        rex::align(UINT64(level_host_slice_layout.Footprint.RowPitch) *
                       (level_host_slice_layout.Footprint.Height / host_block_height) *
                       level_host_slice_layout.Footprint.Depth,
                   UINT64(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
    (is_base ? host_slice_size_base : host_slice_sizes_mips[level]) = level_host_slice_size;
    copy_buffer_size += level_host_slice_size * array_size;
  }
  D3D12_RESOURCE_STATES copy_buffer_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  ID3D12Resource* copy_buffer =
      command_processor_.RequestScratchGPUBuffer(uint32_t(copy_buffer_size), copy_buffer_state);
  if (copy_buffer == nullptr) {
    return false;
  }

  // Begin loading.
  // May use different buffers for scaled base and mips, and also addressability
  // of more than 128 * 2^20 (2^D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP)
  // texels is not mandatory - need two separate UAV descriptors for base and
  // mips.
  // Destination.
  uint32_t descriptor_count = 1;
  if (texture_resolution_scaled) {
    // Source - base and mips, one or both.
    descriptor_count += (level_first == 0 && level_last != 0) ? 2 : 1;
  } else {
    // Source - shared memory.
    if (!bindless_resources_used_) {
      ++descriptor_count;
    }
  }
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors_allocated[3];
  if (!command_processor_.RequestOneUseSingleViewDescriptors(descriptor_count,
                                                             descriptors_allocated)) {
    command_processor_.ReleaseScratchGPUBuffer(copy_buffer, copy_buffer_state);
    return false;
  }
  uint32_t descriptor_write_index = 0;
  command_processor_.SetExternalPipeline(pipeline);
  command_list.D3DSetComputeRootSignature(load_root_signature_.Get());
  // Set up the destination descriptor.
  assert_true(descriptor_write_index < descriptor_count);
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptor_dest =
      descriptors_allocated[descriptor_write_index++];
  ui::d3d12::util::CreateBufferTypedUAV(
      device, descriptor_dest.first, copy_buffer,
      ui::d3d12::util::GetUintPow2DXGIFormat(load_shader_info.dest_bpe_log2),
      uint32_t(copy_buffer_size) >> load_shader_info.dest_bpe_log2);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptor_dest.second);
  // Set up the unscaled source descriptor (scaled needs two descriptors that
  // depend on the buffer being current, so they will be set later - for mips,
  // after loading the base is done).
  if (!texture_resolution_scaled) {
    D3D12SharedMemory& d3d12_shared_memory = static_cast<D3D12SharedMemory&>(shared_memory());
    d3d12_shared_memory.UseForReading();
    ui::d3d12::util::DescriptorCpuGpuHandlePair descriptor_unscaled_source;
    if (bindless_resources_used_) {
      descriptor_unscaled_source = command_processor_.GetSharedMemoryUintPow2BindlessSRVHandlePair(
          load_shader_info.source_bpe_log2);
    } else {
      assert_true(descriptor_write_index < descriptor_count);
      descriptor_unscaled_source = descriptors_allocated[descriptor_write_index++];
      d3d12_shared_memory.WriteUintPow2SRVDescriptor(descriptor_unscaled_source.first,
                                                     load_shader_info.source_bpe_log2);
    }
    command_list.D3DSetComputeRootDescriptorTable(1, descriptor_unscaled_source.second);
  }

  // Submit the copy buffer population commands.

  auto& cbuffer_pool = command_processor_.GetConstantBufferPool();
  LoadConstants load_constants;
  // 3 bits for each.
  assert_true(texture_resolution_scale_x <= 7);
  assert_true(texture_resolution_scale_y <= 7);
  load_constants.is_tiled_3d_endian_scale =
      uint32_t(texture_key.tiled) | (uint32_t(is_3d_tiling) << 1) |
      (uint32_t(texture_key.endianness) << 2) | (texture_resolution_scale_x << 4) |
      (texture_resolution_scale_y << 7);

  // The loop is slices within levels because the base and the levels may need
  // different portions of the scaled resolve virtual address space to be
  // available through buffers, and to create a descriptor, the buffer start
  // address is required - which may be different for base and mips.
  bool scaled_mips_source_set_up = false;
  uint32_t guest_x_blocks_per_group_log2 = load_shader_info.GetGuestXBlocksPerGroupLog2();
  for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
    bool is_base = loop_level == 0;
    uint32_t level = (level_packed == 0) ? 0 : loop_level;

    uint32_t guest_address = (is_base ? texture_key.base_page : texture_key.mip_page) << 12;

    // Set up the base or mips source, also making it accessible if loading from
    // scaled resolve memory.
    if (texture_resolution_scaled && (is_base || !scaled_mips_source_set_up)) {
      uint32_t guest_size_unscaled =
          is_base ? d3d12_texture.GetGuestBaseSize() : d3d12_texture.GetGuestMipsSize();
      if (!MakeScaledResolveRangeCurrent(guest_address, guest_size_unscaled,
                                         load_shader_info.source_bpe_log2)) {
        command_processor_.ReleaseScratchGPUBuffer(copy_buffer, copy_buffer_state);
        return false;
      }
      TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      assert_true(descriptor_write_index < descriptor_count);
      ui::d3d12::util::DescriptorCpuGpuHandlePair descriptor_scaled_source =
          descriptors_allocated[descriptor_write_index++];
      CreateCurrentScaledResolveRangeUintPow2SRV(descriptor_scaled_source.first,
                                                 load_shader_info.source_bpe_log2);
      command_list.D3DSetComputeRootDescriptorTable(1, descriptor_scaled_source.second);
      if (!is_base) {
        scaled_mips_source_set_up = true;
      }
    }

    if (texture_resolution_scaled) {
      // Offset already applied in the buffer because more than 512 MB can't be
      // directly addresses as R32 on some hardware (above
      // 2^D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP).
      load_constants.guest_offset = 0;
    } else {
      load_constants.guest_offset = guest_address;
    }
    if (!is_base) {
      load_constants.guest_offset += guest_layout.mip_offsets_bytes[level] *
                                     (texture_resolution_scale_x * texture_resolution_scale_y);
    }
    const texture_util::TextureGuestLayout::Level& level_guest_layout =
        is_base ? guest_layout.base : guest_layout.mips[level];
    uint32_t level_guest_pitch = level_guest_layout.row_pitch_bytes;
    if (texture_key.tiled) {
      // Shaders expect pitch in blocks for tiled textures.
      level_guest_pitch /= bytes_per_block;
      assert_zero(level_guest_pitch & (xenos::kTextureTileWidthHeight - 1));
    }
    load_constants.guest_pitch_aligned = level_guest_pitch;
    load_constants.guest_z_stride_block_rows_aligned = level_guest_layout.z_slice_stride_block_rows;
    assert_true(!is_3d_tiling || !(load_constants.guest_z_stride_block_rows_aligned &
                                   (xenos::kTextureTileWidthHeight - 1)));

    uint32_t level_width, level_height, level_depth;
    if (level == level_packed) {
      // This is the packed mip tail, containing not only the specified level,
      // but also other levels at different offsets - load the entire needed
      // extents.
      level_width = level_guest_layout.x_extent_blocks * block_width;
      level_height = level_guest_layout.y_extent_blocks * block_height;
      level_depth = level_guest_layout.z_extent;
    } else {
      level_width = std::max(width >> level, uint32_t(1));
      level_height = std::max(height >> level, uint32_t(1));
      level_depth = std::max(depth >> level, uint32_t(1));
    }
    load_constants.size_blocks[0] =
        (level_width + (block_width - 1)) / block_width * texture_resolution_scale_x;
    load_constants.size_blocks[1] =
        (level_height + (block_height - 1)) / block_height * texture_resolution_scale_y;
    load_constants.size_blocks[2] = level_depth;
    load_constants.height_texels = level_height;

    uint32_t group_count_x =
        (load_constants.size_blocks[0] + ((UINT32_C(1) << guest_x_blocks_per_group_log2) - 1)) >>
        guest_x_blocks_per_group_log2;
    uint32_t group_count_y =
        (load_constants.size_blocks[1] + ((UINT32_C(1) << kLoadGuestYBlocksPerGroupLog2) - 1)) >>
        kLoadGuestYBlocksPerGroupLog2;

    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& level_host_slice_layout =
        is_base ? host_slice_layout_base : host_slice_layouts_mips[level];
    uint32_t host_slice_size =
        uint32_t(is_base ? host_slice_size_base : host_slice_sizes_mips[level]);
    load_constants.host_offset = uint32_t(level_host_slice_layout.Offset);
    load_constants.host_pitch = level_host_slice_layout.Footprint.RowPitch;

    command_list.D3DSetComputeRoot32BitConstants(0, sizeof(load_constants) / sizeof(uint32_t),
                                                 &load_constants, 0);

    uint32_t level_array_slice_stride_bytes_scaled =
        level_guest_layout.array_slice_stride_bytes *
        (texture_resolution_scale_x * texture_resolution_scale_y);
    for (uint32_t slice = 0; slice < array_size; ++slice) {
      if (slice != 0) {
        command_list.D3DSetComputeRoot32BitConstants(
            0, sizeof(load_constants.guest_offset) / sizeof(uint32_t), &load_constants.guest_offset,
            offsetof(LoadConstants, guest_offset) / sizeof(uint32_t));
        command_list.D3DSetComputeRoot32BitConstants(
            0, sizeof(load_constants.host_offset) / sizeof(uint32_t), &load_constants.host_offset,
            offsetof(LoadConstants, host_offset) / sizeof(uint32_t));
      }
      assert_true(copy_buffer_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      command_processor_.SubmitBarriers();
      command_list.D3DDispatch(group_count_x, group_count_y, load_constants.size_blocks[2]);
      load_constants.guest_offset += level_array_slice_stride_bytes_scaled;
      load_constants.host_offset += host_slice_size;
    }
  }

  // Update LRU caching because the texture will be used by the command list.
  d3d12_texture.MarkAsUsed();

  // Submit copying from the copy buffer to the host texture.
  ID3D12Resource* texture_resource = d3d12_texture.resource();
  command_processor_.PushTransitionBarrier(
      texture_resource, d3d12_texture.SetResourceState(D3D12_RESOURCE_STATE_COPY_DEST),
      D3D12_RESOURCE_STATE_COPY_DEST);
  command_processor_.PushTransitionBarrier(copy_buffer, copy_buffer_state,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
  copy_buffer_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  command_processor_.SubmitBarriers();
  uint32_t texture_level_count = texture_key.mip_max_level + 1;
  D3D12_TEXTURE_COPY_LOCATION location_source, location_dest;
  location_source.pResource = copy_buffer;
  location_source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  location_dest.pResource = texture_resource;
  location_dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  for (uint32_t level = level_first; level <= level_last; ++level) {
    uint32_t guest_level = std::min(level, level_packed);
    location_source.PlacedFootprint =
        level ? host_slice_layouts_mips[guest_level] : host_slice_layout_base;
    location_dest.SubresourceIndex = level;
    UINT64 host_slice_size = level ? host_slice_sizes_mips[guest_level] : host_slice_size_base;
    D3D12_BOX source_box;
    const D3D12_BOX* source_box_ptr;
    if (level >= level_packed) {
      uint32_t level_offset_blocks_x, level_offset_blocks_y, level_offset_z;
      texture_util::GetPackedMipOffset(width, height, depth, guest_format, level,
                                       level_offset_blocks_x, level_offset_blocks_y,
                                       level_offset_z);
      source_box.left = level_offset_blocks_x * block_width * texture_resolution_scale_x;
      source_box.top = level_offset_blocks_y * block_height * texture_resolution_scale_y;
      source_box.front = level_offset_z;
      source_box.right =
          source_box.left +
          rex::align(std::max((width * texture_resolution_scale_x) >> level, uint32_t(1)),
                     host_block_width);
      source_box.bottom =
          source_box.top +
          rex::align(std::max((height * texture_resolution_scale_y) >> level, uint32_t(1)),
                     host_block_height);
      source_box.back = source_box.front + std::max(depth >> level, uint32_t(1));
      source_box_ptr = &source_box;
    } else {
      source_box_ptr = nullptr;
    }
    for (uint32_t slice = 0; slice < array_size; ++slice) {
      command_list.D3DCopyTextureRegion(&location_dest, 0, 0, 0, &location_source, source_box_ptr);
      location_dest.SubresourceIndex += texture_level_count;
      location_source.PlacedFootprint.Offset += host_slice_size;
    }
  }

  command_processor_.ReleaseScratchGPUBuffer(copy_buffer, copy_buffer_state);

  return true;
}

void D3D12TextureCache::UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) {
  uint32_t bindings_remaining = fetch_constant_mask;
  uint32_t binding_index;
  while (rex::bit_scan_forward(bindings_remaining, &binding_index)) {
    bindings_remaining &= ~(UINT32_C(1) << binding_index);
    D3D12TextureBinding& d3d12_binding = d3d12_texture_bindings_[binding_index];
    d3d12_binding.Reset();
    const TextureBinding* binding = GetValidTextureBinding(binding_index);
    if (!binding) {
      continue;
    }
    if (IsSignedVersionSeparateForFormat(binding->key)) {
      if (binding->texture && texture_util::IsAnySignNotSigned(binding->swizzled_signs)) {
        d3d12_binding.descriptor_index =
            FindOrCreateTextureDescriptor(*static_cast<D3D12Texture*>(binding->texture),
                                          binding->key.dimension, false, binding->host_swizzle);
      }
      if (binding->texture_signed && texture_util::IsAnySignSigned(binding->swizzled_signs)) {
        d3d12_binding.descriptor_index_signed =
            FindOrCreateTextureDescriptor(*static_cast<D3D12Texture*>(binding->texture_signed),
                                          binding->key.dimension, true, binding->host_swizzle);
      }
    } else {
      D3D12Texture* texture = static_cast<D3D12Texture*>(binding->texture);
      if (texture) {
        if (texture_util::IsAnySignNotSigned(binding->swizzled_signs)) {
          d3d12_binding.descriptor_index = FindOrCreateTextureDescriptor(
              *texture, binding->key.dimension, false, binding->host_swizzle);
        }
        if (texture_util::IsAnySignSigned(binding->swizzled_signs)) {
          d3d12_binding.descriptor_index_signed = FindOrCreateTextureDescriptor(
              *texture, binding->key.dimension, true, binding->host_swizzle);
        }
      }
    }
  }
}

ID3D12Resource* D3D12TextureCache::D3D12Texture::GetOrCreate3DAs2DResource(
    D3D12_RESOURCE_STATES end_state) {
  if (!REXCVAR_GET(gpu_3d_to_2d_texture)) {
    return nullptr;
  }

  auto& d3d12_cache = static_cast<D3D12TextureCache&>(texture_cache());

  if (texture_3d_as_2d_) {
    d3d12_cache.command_processor_.PushTransitionBarrier(
        texture_3d_as_2d_->resource(), texture_3d_as_2d_->SetResourceState(end_state), end_state);
    return texture_3d_as_2d_->resource();
  }

  const ui::d3d12::D3D12Provider& provider = d3d12_cache.command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Alignment = 0;
  desc.Width = key().GetWidth();
  desc.Height = key().GetHeight();
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = d3d12_cache.GetDXGIResourceFormat(key());
  if (desc.Format == DXGI_FORMAT_UNKNOWN) {
    return nullptr;
  }
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  desc.Flags = D3D12_RESOURCE_FLAG_NONE;

  D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COPY_DEST;
  Microsoft::WRL::ComPtr<ID3D12Resource> resource_2d;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             provider.GetHeapFlagCreateNotZeroed(), &desc,
                                             initial_state, nullptr, IID_PPV_ARGS(&resource_2d)))) {
    REXGPU_ERROR("D3D12TextureCache: Failed to create 3D-as-2D wrapper resource");
    return nullptr;
  }

  TextureKey key_2d = key();
  key_2d.depth_or_array_size_minus_1 = 0;
  key_2d.mip_max_level = 0;
  texture_3d_as_2d_.reset(
      new D3D12Texture(d3d12_cache, key_2d, resource_2d.Get(), initial_state, false));
  texture_3d_as_2d_->SetForceLoad3DTiling(true);

  if (!d3d12_cache.LoadTextureData(*texture_3d_as_2d_)) {
    REXGPU_ERROR("D3D12TextureCache: Failed to load 3D-as-2D wrapper data");
    texture_3d_as_2d_.reset();
    return nullptr;
  }

  d3d12_cache.command_processor_.PushTransitionBarrier(
      texture_3d_as_2d_->resource(), texture_3d_as_2d_->SetResourceState(end_state), end_state);
  return texture_3d_as_2d_->resource();
}

uint32_t D3D12TextureCache::FindOrCreateTextureDescriptor(D3D12Texture& texture,
                                                          xenos::DataDimension dimension,
                                                          bool is_signed, uint32_t host_swizzle) {
  D3D12Texture::SRVDescriptorKey descriptor_key;
  descriptor_key.key = 0;
  descriptor_key.is_signed = uint32_t(is_signed);
  descriptor_key.host_swizzle = host_swizzle;
  descriptor_key.dimension = uint32_t(dimension);

  // Try to find an existing descriptor.
  uint32_t existing_descriptor_index = texture.GetSRVDescriptorIndex(descriptor_key);
  if (existing_descriptor_index != UINT32_MAX) {
    PROFILE_TEXTURE_CACHE_HIT();
    return existing_descriptor_index;
  }
  PROFILE_TEXTURE_CACHE_MISS();

  TextureKey texture_key = texture.key();

  // Create a new bindless or cached descriptor if supported.
  D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};

  if (IsSignedVersionSeparateForFormat(texture_key) &&
      texture_key.signed_separate != uint32_t(is_signed)) {
    // Not the version with the needed signedness.
    return UINT32_MAX;
  }
  xenos::TextureFormat format = texture_key.format;
  if (is_signed) {
    // Not supporting signed compressed textures - hopefully DXN and DXT5A are
    // not used as signed.
    desc.Format = host_formats_[uint32_t(format)].dxgi_format_signed;
  } else {
    desc.Format = GetDXGIUnormFormat(texture_key);
  }
  if (desc.Format == DXGI_FORMAT_UNKNOWN) {
    unsupported_format_features_used_[uint32_t(format)] |=
        is_signed ? kUnsupportedSnormBit : kUnsupportedUnormBit;
    return UINT32_MAX;
  }

  // A replacement has its own mip count (texture_replace_dir).
  uint32_t mip_levels = texture.replaced_mip_levels() ? texture.replaced_mip_levels()
                                                      : texture_key.mip_max_level + 1;
  ID3D12Resource* resource_for_view = texture.resource();
  switch (dimension) {
    case xenos::DataDimension::k3D:
      desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
      desc.Texture3D.MostDetailedMip = 0;
      desc.Texture3D.MipLevels = mip_levels;
      desc.Texture3D.ResourceMinLODClamp = 0.0f;
      break;
    case xenos::DataDimension::kCube:
      desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
      desc.TextureCube.MostDetailedMip = 0;
      desc.TextureCube.MipLevels = mip_levels;
      desc.TextureCube.ResourceMinLODClamp = 0.0f;
      break;
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
      if (texture_key.dimension == xenos::DataDimension::k3D) {
        resource_for_view =
            texture.GetOrCreate3DAs2DResource(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                              D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        if (!resource_for_view) {
          return UINT32_MAX;
        }
        desc.Texture2DArray.MostDetailedMip = 0;
        desc.Texture2DArray.MipLevels = 1;
        desc.Texture2DArray.FirstArraySlice = 0;
        desc.Texture2DArray.ArraySize = 1;
        desc.Texture2DArray.PlaneSlice = 0;
        desc.Texture2DArray.ResourceMinLODClamp = 0.0f;
      } else {
        desc.Texture2DArray.MostDetailedMip = 0;
        desc.Texture2DArray.MipLevels = mip_levels;
        desc.Texture2DArray.FirstArraySlice = 0;
        desc.Texture2DArray.ArraySize = texture_key.GetDepthOrArraySize();
        desc.Texture2DArray.PlaneSlice = 0;
        desc.Texture2DArray.ResourceMinLODClamp = 0.0f;
      }
      break;
    default:
      assert_unhandled_case(dimension);
      return UINT32_MAX;
  }

  desc.Shader4ComponentMapping =
      host_swizzle | D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES;

  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  uint32_t descriptor_index;
  if (bindless_resources_used_) {
    descriptor_index = command_processor_.RequestPersistentViewBindlessDescriptor();
    if (descriptor_index == UINT32_MAX) {
      REXGPU_ERROR(
          "Failed to create a texture descriptor - no free bindless view "
          "descriptors");
      return UINT32_MAX;
    }
  } else {
    if (!srv_descriptor_cache_free_.empty()) {
      descriptor_index = srv_descriptor_cache_free_.back();
      srv_descriptor_cache_free_.pop_back();
    } else {
      // Allocated + 1 (including the descriptor that is being added), rounded
      // up to kSRVDescriptorCachePageSize, (allocated + 1 + size - 1).
      uint32_t cache_pages_needed =
          (srv_descriptor_cache_allocated_ + kSRVDescriptorCachePageSize) /
          kSRVDescriptorCachePageSize;
      if (srv_descriptor_cache_.size() < cache_pages_needed) {
        D3D12_DESCRIPTOR_HEAP_DESC cache_heap_desc;
        cache_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        cache_heap_desc.NumDescriptors = kSRVDescriptorCachePageSize;
        cache_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        cache_heap_desc.NodeMask = 0;
        while (srv_descriptor_cache_.size() < cache_pages_needed) {
          Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> cache_heap;
          if (FAILED(device->CreateDescriptorHeap(&cache_heap_desc, IID_PPV_ARGS(&cache_heap)))) {
            REXGPU_ERROR(
                "D3D12TextureCache: Failed to create a texture descriptor - "
                "couldn't create a descriptor cache heap");
            return UINT32_MAX;
          }
          srv_descriptor_cache_.emplace_back(cache_heap.Get());
        }
      }
      descriptor_index = srv_descriptor_cache_allocated_++;
    }
  }
  device->CreateShaderResourceView(resource_for_view, &desc,
                                   GetTextureDescriptorCPUHandle(descriptor_index));
  texture.AddSRVDescriptorIndex(descriptor_key, descriptor_index);
  return descriptor_index;
}

void D3D12TextureCache::ReleaseTextureDescriptor(uint32_t descriptor_index) {
  if (bindless_resources_used_) {
    command_processor_.ReleaseViewBindlessDescriptorImmediately(descriptor_index);
  } else {
    srv_descriptor_cache_free_.push_back(descriptor_index);
  }
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12TextureCache::GetTextureDescriptorCPUHandle(
    uint32_t descriptor_index) const {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  if (bindless_resources_used_) {
    return provider.OffsetViewDescriptor(command_processor_.GetViewBindlessHeapCPUStart(),
                                         descriptor_index);
  }
  D3D12_CPU_DESCRIPTOR_HANDLE heap_start =
      srv_descriptor_cache_[descriptor_index / kSRVDescriptorCachePageSize].heap_start();
  uint32_t heap_offset = descriptor_index % kSRVDescriptorCachePageSize;
  return provider.OffsetViewDescriptor(heap_start, heap_offset);
}

xenos::ClampMode D3D12TextureCache::NormalizeClampMode(xenos::ClampMode clamp_mode) const {
  if (clamp_mode == xenos::ClampMode::kClampToHalfway) {
    // No GL_CLAMP (clamp to half edge, half border) equivalent in Direct3D 12,
    // but there's no Direct3D 9 equivalent anyway, and too weird to be suitable
    // for intentional real usage.
    return xenos::ClampMode::kClampToEdge;
  }
  if (clamp_mode == xenos::ClampMode::kMirrorClampToHalfway ||
      clamp_mode == xenos::ClampMode::kMirrorClampToBorder) {
    // No Direct3D 12 equivalents.
    return xenos::ClampMode::kMirrorClampToEdge;
  }
  return clamp_mode;
}

namespace {

// DDS's legacy header (DDS_HEADER, after the "DDS " magic) for the BC1-BC3 textures dumped, and
// what the replacement reader takes from one (also with the DX10 header).
struct DdsHeader {
  uint32_t size, flags, height, width, pitch_or_linear_size, depth, mip_map_count;
  uint32_t reserved1[11];
  struct {
    uint32_t size, flags, four_cc, rgb_bit_count, r_mask, g_mask, b_mask, a_mask;
  } pixel_format;
  uint32_t caps, caps2, caps3, caps4, reserved2;
};
static_assert(sizeof(DdsHeader) == 124);
constexpr uint32_t FourCC(char a, char b, char c, char d) {
  return uint32_t(uint8_t(a)) | uint32_t(uint8_t(b)) << 8 | uint32_t(uint8_t(c)) << 16 |
         uint32_t(uint8_t(d)) << 24;
}
constexpr uint32_t kFourCC[3] = {FourCC('D', 'X', 'T', '1'), FourCC('D', 'X', 'T', '3'),
                                 FourCC('D', 'X', 'T', '5')};
// 8888 texels, kept as the guest has them after the endian swap (B, G, R, A in memory for the
// games' UI): a DDS with A8R8G8B8 masks shows them in their colours.
constexpr int kFamily8888 = 3;

// 0 BC1, 1 BC2, 2 BC3, or -1 for the formats this doesn't handle.
int BlockFamily(xenos::TextureFormat format) {
  switch (format) {
    case xenos::TextureFormat::k_DXT1:
      return 0;
    case xenos::TextureFormat::k_DXT2_3:
      return 1;
    case xenos::TextureFormat::k_DXT4_5:
      return 2;
    case xenos::TextureFormat::k_8_8_8_8:
      return kFamily8888;
    default:
      return -1;
  }
}
int BlockFamily(DXGI_FORMAT format) {
  switch (format) {
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
      return 0;
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
      return 1;
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
      return 2;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
      return kFamily8888;
    default:
      return -1;
  }
}

std::filesystem::path Utf8Dir(const std::string& utf8) {
  return std::filesystem::u8path(utf8);  // this file builds as C++17
}

// The DDS files of the replacement folder by their hash name, read once.
const std::unordered_map<uint64_t, std::filesystem::path>& ReplacementIndex() {
  static std::unordered_map<uint64_t, std::filesystem::path> index;
  static std::once_flag once;
  std::call_once(once, [] {
    std::error_code ec;
    for (const auto& entry :
         std::filesystem::directory_iterator(Utf8Dir(REXCVAR_GET(texture_replace_dir)), ec)) {
      const std::filesystem::path& path = entry.path();
      const std::string stem = path.stem().string();
      if (path.extension() != ".dds" || stem.size() != 16) continue;
      char* end = nullptr;
      const uint64_t hash = std::strtoull(stem.c_str(), &end, 16);
      if (end && *end == '\0') index[hash] = path;
    }
    REXGPU_INFO("Texture replacement: {} DDS files in {}", index.size(),
                REXCVAR_GET(texture_replace_dir));
  });
  return index;
}

}  // namespace

bool D3D12TextureCache::DumpOrReplaceTexture(D3D12Texture& texture) {
  const std::string& dump_dir = REXCVAR_GET(texture_dump_dir);
  const std::string& replace_dir = REXCVAR_GET(texture_replace_dir);
  if (dump_dir.empty() && replace_dir.empty() && !HasTexturePatches()) return false;
  const TextureKey key = texture.key();
  const int family = BlockFamily(key.format);
  const uint32_t width = key.GetWidth(), height = key.GetHeight();
  // 2D textures in BC1-BC3 the host keeps compressed and 8888 ones up to 1024x1024 (UI atlases;
  // bigger ones are mostly render targets), not render-to-texture results, not tiny.
  if (family < 0 || (family == kFamily8888 && (width > 1024 || height > 1024)) ||
      key.scaled_resolve || key.dimension != xenos::DataDimension::k2DOrStacked ||
      key.GetDepthOrArraySize() != 1 || width < 32 || height < 32 ||
      IsDecompressionNeeded(key.format, width, height) ||
      BlockFamily(GetDXGIResourceFormat(key)) != family) {
    return false;
  }
  // The top level in the guest's memory, untiled and in little-endian order, as a DDS has it.
  const FormatInfo* info = FormatInfo::Get(key.format);
  const uint32_t bytes_per_block = info->bytes_per_block();
  const uint32_t bytes_per_block_log2 = bytes_per_block == 16 ? 4 : bytes_per_block == 8 ? 3 : 2;
  const uint32_t blocks_x = (width + info->block_width - 1) / info->block_width,
                 blocks_y = (height + info->block_height - 1) / info->block_height;
  const auto& base = texture.guest_layout().base;
  const uint32_t guest_size = texture.GetGuestBaseSize();
  const uint8_t* guest =
      static_cast<const uint8_t*>(shared_memory().guest_memory().TranslatePhysical(key.base_page << 12));
  if (!guest || !base.row_pitch_bytes) return false;
  std::vector<uint8_t> linear(size_t(blocks_x) * blocks_y * bytes_per_block);
  for (uint32_t y = 0; y < blocks_y; ++y) {
    for (uint32_t x = 0; x < blocks_x; ++x) {
      const uint32_t offset =
          key.tiled ? uint32_t(texture_util::GetTiledOffset2D(int32_t(x), int32_t(y),
                                                              base.row_pitch_bytes / bytes_per_block,
                                                              bytes_per_block_log2))
                    : y * base.row_pitch_bytes + x * bytes_per_block;
      if (offset + bytes_per_block > guest_size) return false;
      std::memcpy(&linear[(size_t(y) * blocks_x + x) * bytes_per_block], guest + offset,
                  bytes_per_block);
    }
  }
  switch (key.endianness) {
    case xenos::Endian::k8in16:
      for (size_t i = 0; i + 1 < linear.size(); i += 2) std::swap(linear[i], linear[i + 1]);
      break;
    case xenos::Endian::k8in32:
      for (size_t i = 0; i + 3 < linear.size(); i += 4) {
        std::swap(linear[i], linear[i + 3]);
        std::swap(linear[i + 1], linear[i + 2]);
      }
      break;
    case xenos::Endian::k16in32:
      for (size_t i = 0; i + 3 < linear.size(); i += 4) {
        std::swap(linear[i], linear[i + 2]);
        std::swap(linear[i + 1], linear[i + 3]);
      }
      break;
    default:
      break;
  }
  const uint64_t hash =
      XXH3_64bits_withSeed(linear.data(), linear.size(),
                           uint64_t(family) | uint64_t(width) << 8 | uint64_t(height) << 24);
  const std::string name = fmt::format("{:016X}.dds", hash);

  // The dump: once per content, written on a thread of its own.
  if (!dump_dir.empty()) {
    static std::mutex dumped_mutex;
    static std::unordered_set<uint64_t> dumped;
    bool first;
    {
      std::lock_guard lock(dumped_mutex);
      first = dumped.insert(hash).second;
    }
    if (first) {
      std::thread([path = Utf8Dir(dump_dir) / name, data = linear, width, height, family] {
        DdsHeader h = {};
        h.size = sizeof(DdsHeader);
        h.height = height;
        h.width = width;
        h.mip_map_count = 1;
        h.pixel_format.size = 32;
        if (family == kFamily8888) {
          h.flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000;  // caps, height, width, pitch, pixel format
          h.pitch_or_linear_size = width * 4;
          h.pixel_format.flags = 0x40 | 0x1;  // DDPF_RGB | DDPF_ALPHAPIXELS
          h.pixel_format.rgb_bit_count = 32;
          h.pixel_format.r_mask = 0x00FF0000;
          h.pixel_format.g_mask = 0x0000FF00;
          h.pixel_format.b_mask = 0x000000FF;
          h.pixel_format.a_mask = 0xFF000000;
        } else {
          h.flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000;  // caps, height, width, pixel format, linear size
          h.pitch_or_linear_size = uint32_t(data.size());
          h.pixel_format.flags = 0x4;  // DDPF_FOURCC
          h.pixel_format.four_cc = kFourCC[family];
        }
        h.caps = 0x1000;  // DDSCAPS_TEXTURE
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write("DDS ", 4);
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
        out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
      }).detach();
    }
  }

  // A texture patch (texture_patch.h) first: game code redraws it from the guest's data. Else
  // the replacement: <hash>.dds of the same format.
  ReplacementJob job;
  if (HasTexturePatch(hash)) {
    job.patch_source = std::move(linear);
    job.hash = hash;
  } else {
    if (replace_dir.empty()) return false;
    const auto& index = ReplacementIndex();
    const auto it = index.find(hash);
    if (it == index.end()) return false;
    job.path = it->second;
  }
  job.key = key;
  job.desc = texture.resource()->GetDesc();
  job.family = family;
  job.width = width;
  job.height = height;
  job.name = name;
  if (REXCVAR_GET(texture_replace_async)) {
    QueueReplacement(std::move(job));
    // The guest's own texture until the replacement is ready (ApplyReadyReplacements).
    return false;
  }
  ReplacementResult result;
  if (!MakeReplacement(job, result)) return false;
  ApplyReplacement(texture, result, false);
  return true;
}

// Reads the file and makes the resources, without touching anything of the GPU thread: D3D12's
// device methods may be called from any thread.
bool D3D12TextureCache::MakeReplacement(const ReplacementJob& job, ReplacementResult& out) const {
  const auto load_start = std::chrono::steady_clock::now();
  // One read of the whole file: reading it through stream iterators byte by byte made a figure
  // change stutter with an HD pack (a 1024x1024 DXT5 with mips is about 1.4 MB, and a new character
  // brings dozens).
  std::vector<uint8_t> file;
  if (!job.patch_source.empty()) {
    if (!ApplyTexturePatch(job.hash, job.width, job.height, job.patch_source.data(),
                           job.patch_source.size(), file)) {
      return false;
    }
  } else {
    std::ifstream in(job.path, std::ios::binary | std::ios::ate);
    const std::streamoff file_size = in ? std::streamoff(in.tellg()) : std::streamoff(0);
    file.resize(file_size > 0 ? size_t(file_size) : 0);
    if (!file.empty()) {
      in.seekg(0);
      if (!in.read(reinterpret_cast<char*>(file.data()), std::streamsize(file.size()))) file.clear();
    }
  }
  const std::string source = job.patch_source.empty() ? job.path.string() : job.name + " (patch)";
  if (file.size() < 4 + sizeof(DdsHeader) || std::memcmp(file.data(), "DDS ", 4) != 0) return false;
  DdsHeader header;
  std::memcpy(&header, file.data() + 4, sizeof(header));
  size_t data_offset = 4 + sizeof(DdsHeader);
  int dds_family = -1;
  for (int f = 0; f < 3; ++f) {
    if (header.pixel_format.four_cc == kFourCC[f]) dds_family = f;
  }
  // 8888 as the dump writes it: 32 bits a texel, A8R8G8B8 masks (B, G, R, A in memory).
  if (!(header.pixel_format.flags & 0x4) && header.pixel_format.rgb_bit_count == 32 &&
      header.pixel_format.r_mask == 0x00FF0000 && header.pixel_format.g_mask == 0x0000FF00 &&
      header.pixel_format.b_mask == 0x000000FF) {
    dds_family = kFamily8888;
  }
  if (header.pixel_format.four_cc == FourCC('D', 'X', '1', '0') && file.size() >= data_offset + 20) {
    uint32_t dxgi_format;
    std::memcpy(&dxgi_format, file.data() + data_offset, 4);
    dds_family = BlockFamily(DXGI_FORMAT(dxgi_format));
    data_offset += 20;
  }
  const uint32_t mips = std::max(uint32_t(1), header.mip_map_count);
  const bool blocks = dds_family >= 0 && dds_family < kFamily8888;
  if (dds_family != job.family || !header.width || !header.height ||
      (blocks && ((header.width & 3) || (header.height & 3))) || mips > 16 ||
      header.width > 16384 || header.height > 16384) {
    REXGPU_WARN("Texture replacement: {} is not a {}x{}-compatible {} DDS", source, job.width,
                job.height,
                job.family == 0   ? "DXT1"
                : job.family == 1 ? "DXT3"
                : job.family == 2 ? "DXT5"
                                  : "8888");
    return false;
  }
  D3D12_RESOURCE_DESC desc = job.desc;
  desc.Width = header.width;
  desc.Height = header.height;
  desc.MipLevels = UINT16(mips);
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  std::array<UINT, 16> rows;
  std::array<UINT64, 16> row_bytes;
  UINT64 upload_size = 0;
  device->GetCopyableFootprints(&desc, 0, mips, 0, out.layouts.data(), rows.data(), row_bytes.data(),
                                &upload_size);
  // The file's levels one after the other, each its blocks row by row.
  size_t needed = data_offset;
  for (uint32_t m = 0; m < mips; ++m) needed += size_t(rows[m]) * row_bytes[m];
  if (file.size() < needed) {
    REXGPU_WARN("Texture replacement: {} is too short for {} mips", source, mips);
    return false;
  }
  D3D12_RESOURCE_DESC upload_desc;
  ui::d3d12::util::FillBufferResourceDesc(upload_desc, upload_size, D3D12_RESOURCE_FLAG_NONE);
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             provider.GetHeapFlagCreateNotZeroed(), &desc,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&out.resource))) ||
      FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesUpload,
                                             provider.GetHeapFlagCreateNotZeroed(), &upload_desc,
                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                             IID_PPV_ARGS(&out.upload)))) {
    return false;
  }
  uint8_t* mapped = nullptr;
  D3D12_RANGE no_read = {0, 0};
  if (FAILED(out.upload->Map(0, &no_read, reinterpret_cast<void**>(&mapped)))) return false;
  const uint8_t* source = file.data() + data_offset;
  for (uint32_t m = 0; m < mips; ++m) {
    for (UINT row = 0; row < rows[m]; ++row) {
      std::memcpy(mapped + out.layouts[m].Offset + UINT64(row) * out.layouts[m].Footprint.RowPitch,
                  source, size_t(row_bytes[m]));
      source += row_bytes[m];
    }
  }
  out.upload->Unmap(0, nullptr);
  out.key = job.key;
  out.name = job.name;
  out.guest_width = job.width;
  out.guest_height = job.height;
  out.width = header.width;
  out.height = header.height;
  out.mips = mips;
  out.make_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - load_start)
                             .count());
  return true;
}

// On the GPU thread, with its command list open. in_use: frames under way may already draw with
// the guest's texture.
void D3D12TextureCache::ApplyReplacement(D3D12Texture& texture, ReplacementResult& result,
                                         bool in_use) {
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  for (uint32_t m = 0; m < result.mips; ++m) {
    D3D12_TEXTURE_COPY_LOCATION dst, src;
    dst.pResource = result.resource.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = m;
    src.pResource = result.upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = result.layouts[m];
    command_list.D3DCopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  }
  texture.SetReplacement(std::move(result.resource), D3D12_RESOURCE_STATE_COPY_DEST,
                         std::move(result.upload), result.mips, in_use);
  static uint32_t replaced = 0;
  // The time reading and making it took (on a worker with texture_replace_async), for the log.
  static uint64_t spent_us = 0;
  spent_us += result.make_us;
  ++replaced;
  if (replaced <= 20 || replaced % 100 == 0) {
    REXGPU_INFO("Texture replacement: {} ({}x{} -> {}x{}, {} mips, {:.1f} ms), {} so far, {} ms in all", result.name,
                result.guest_width, result.guest_height, result.width, result.height, result.mips,
                double(result.make_us) / 1000.0, replaced, spent_us / 1000);
  }
}

void D3D12TextureCache::QueueReplacement(ReplacementJob job) {
  std::lock_guard lock(replacement_mutex_);
  if (replacement_workers_.empty()) {
    for (int i = 0; i < 2; ++i) replacement_workers_.emplace_back([this] { ReplacementWorker(); });
  }
  replacement_jobs_.push_back(std::move(job));
  replacement_wake_.notify_one();
}

void D3D12TextureCache::ReplacementWorker() {
  std::unique_lock lock(replacement_mutex_);
  while (true) {
    replacement_wake_.wait(lock, [this] { return replacement_stop_ || !replacement_jobs_.empty(); });
    if (replacement_stop_) return;
    ReplacementJob job = std::move(replacement_jobs_.front());
    replacement_jobs_.pop_front();
    lock.unlock();
    ReplacementResult result;
    const bool made = MakeReplacement(job, result);
    lock.lock();
    if (made) {
      replacement_results_.push_back(std::move(result));
      replacement_results_ready_.store(true, std::memory_order_relaxed);
    }
  }
}

void D3D12TextureCache::ApplyReadyReplacements() {
  if (!replacement_results_ready_.load(std::memory_order_relaxed)) return;
  std::vector<ReplacementResult> ready;
  {
    std::lock_guard lock(replacement_mutex_);
    ready.swap(replacement_results_);
    replacement_results_ready_.store(false, std::memory_order_relaxed);
  }
  bool applied = false;
  for (ReplacementResult& result : ready) {
    // Looked up again: the texture may have gone since the file was queued.
    Texture* found = FindExistingTexture(result.key);
    if (!found) continue;
    D3D12Texture& texture = static_cast<D3D12Texture&>(*found);
    if (texture.replaced_mip_levels()) continue;
    ApplyReplacement(texture, result, true);
    applied = true;
  }
  // The bindings hold descriptors of the old resources.
  if (applied) ResetTextureBindings();
}

void D3D12TextureCache::StopReplacementWorkers() {
  {
    std::lock_guard lock(replacement_mutex_);
    replacement_stop_ = true;
  }
  replacement_wake_.notify_all();
  for (std::thread& worker : replacement_workers_) worker.join();
  replacement_workers_.clear();
}

}  // namespace rex::graphics::d3d12
