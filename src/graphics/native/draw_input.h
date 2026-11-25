#pragma once

#include <array>
#include <bit>
#include <compare>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <xxhash.h>

#include <rex/graphics/edram_layout.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>

namespace rex::graphics::native {

constexpr uint32_t kPhysicalMemorySize = 512 * 1024 * 1024;

// Host sampler state derived from a guest sampler binding. Also the device-lifetime
// cache key, so it must order identically to how it compares.
struct DrawSampler {
  VkFilter min_filter = VK_FILTER_NEAREST;
  VkFilter mag_filter = VK_FILTER_NEAREST;
  VkSamplerMipmapMode mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  float max_lod = 0.0f;
  VkSamplerAddressMode address_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSamplerAddressMode address_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  auto operator<=>(const DrawSampler&) const = default;
};

// Hashes only the initialized fields so equal sampler states always map to the
// same bucket regardless of any padding in the struct.
struct DrawSamplerHash {
  size_t operator()(const DrawSampler& key) const noexcept {
    const std::array<uint32_t, 6> fields{
        uint32_t(key.min_filter), uint32_t(key.mag_filter), uint32_t(key.mip_mode),
        std::bit_cast<uint32_t>(key.max_lod), uint32_t(key.address_u), uint32_t(key.address_v)};
    return size_t(XXH3_64bits(fields.data(), sizeof(fields)));
  }
};

// Everything that determines the uploaded host texel data of a guest texture.
// The view swizzle is deliberately excluded because a single image is sampled
// with several component mappings.
struct GuestTextureKey {
  uint32_t base_page = 0;
  uint32_t mip_page = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t array_layers = 1;
  xenos::TextureFormat format = xenos::TextureFormat::k_8_8_8_8;
  xenos::DataDimension dimension = xenos::DataDimension::k2DOrStacked;
  bool tiled = false;
  bool packed_mips = false;
  xenos::Endian endianness = xenos::Endian::kNone;
  uint32_t pitch = 0;
  uint32_t mip_max = 0;
  auto operator<=>(const GuestTextureKey&) const = default;
};

// Hashes only the initialized fields, so the padding between the bool flags and
// the following enum can never make two identical keys hash differently.
struct GuestTextureKeyHash {
  size_t operator()(const GuestTextureKey& key) const noexcept {
    const std::array<uint32_t, 12> fields{
        key.base_page,      key.mip_page,       key.width,          key.height,
        key.array_layers,   uint32_t(key.format), uint32_t(key.dimension), key.tiled,
        key.packed_mips,    uint32_t(key.endianness), key.pitch,     key.mip_max};
    return size_t(XXH3_64bits(fields.data(), sizeof(fields)));
  }
};

// Whether two [start, start + length) physical memory ranges share any byte.
// Empty ranges never overlap. Arithmetic is done in 64 bits so a range ending at
// the top of the 512 MB guest address space does not wrap.
constexpr bool RangesOverlap(uint32_t a_start, uint32_t a_length, uint32_t b_start,
                             uint32_t b_length) {
  if (!a_length || !b_length) {
    return false;
  }
  const uint64_t a_end = uint64_t(a_start) + a_length;
  const uint64_t b_end = uint64_t(b_start) + b_length;
  return a_start < b_end && b_start < a_end;
}

constexpr bool IsPhysicalRange(uint32_t address, size_t size) {
  return address < kPhysicalMemorySize && size && size <= kPhysicalMemorySize - address;
}

// Number of components a guest texture format actually samples. Single-channel
// formats replicate R into all channels, two-channel ones replicate G into B
// and A, matching the host format swizzle of the Xenos Vulkan backend.
constexpr uint32_t TextureChannelCount(xenos::TextureFormat format) {
  switch (format) {
    case xenos::TextureFormat::k_8:
    case xenos::TextureFormat::k_8_A:
    case xenos::TextureFormat::k_8_B:
    case xenos::TextureFormat::k_32_FLOAT:
    case xenos::TextureFormat::k_24_8:
    case xenos::TextureFormat::k_24_8_FLOAT:
      return 1;
    case xenos::TextureFormat::k_8_8:
    case xenos::TextureFormat::k_16_16_FLOAT:
    case xenos::TextureFormat::k_DXN:
      return 2;
    default:
      return 4;
  }
}

// Component mapping for sampling a guest texture. One- and two-channel formats
// replicate the stored components into the unused ones, matching the Xenos
// Vulkan host format swizzle. 32bpp 3/4-channel resolves copied with
// copy_dest_swap store red and blue swapped, which the guest fetch swizzle
// (0x60A BGRA or 0xA0A BGR1) normally undoes - swap_red_blue applies that swap
// here for cached resolves that keep the EDRAM RGBA order.
constexpr VkComponentMapping MakeTextureComponentMapping(uint32_t fetch_swizzle,
                                                        uint32_t channel_count,
                                                        bool swap_red_blue) {
  const VkComponentSwizzle r = VK_COMPONENT_SWIZZLE_R;
  const VkComponentSwizzle g = VK_COMPONENT_SWIZZLE_G;
  const VkComponentSwizzle b = VK_COMPONENT_SWIZZLE_B;
  const VkComponentSwizzle a = VK_COMPONENT_SWIZZLE_A;
  const VkComponentSwizzle first = channel_count >= 3 && swap_red_blue ? b : r;
  const VkComponentSwizzle third = channel_count >= 3 && swap_red_blue ? r : b;
  const std::array<VkComponentSwizzle, 8> swizzles{
      first,
      channel_count == 1 ? r : g,
      channel_count == 1 ? r : channel_count == 2 ? g : third,
      channel_count == 1 ? r : channel_count == 2 ? g : a,
      VK_COMPONENT_SWIZZLE_ZERO,
      VK_COMPONENT_SWIZZLE_ONE,
      VK_COMPONENT_SWIZZLE_ZERO,
      VK_COMPONENT_SWIZZLE_ZERO};
  return {swizzles[fetch_swizzle & 7], swizzles[(fetch_swizzle >> 3) & 7],
          swizzles[(fetch_swizzle >> 6) & 7], swizzles[(fetch_swizzle >> 9) & 7]};
}

// Packs a component mapping into one 32-bit cache key. Every swizzle value the
// translator produces fits in a byte, so distinct mappings never collide.
constexpr uint32_t PackTextureComponentMapping(VkComponentMapping mapping) {
  return (uint32_t(mapping.r) & 0xFF) | ((uint32_t(mapping.g) & 0xFF) << 8) |
         ((uint32_t(mapping.b) & 0xFF) << 16) | ((uint32_t(mapping.a) & 0xFF) << 24);
}

// IEEE floating-point host formats are natively signed, so a signed guest fetch
// maps directly onto them without a separate SNORM view.
constexpr bool IsFloatTextureFormat(xenos::TextureFormat format) {
  return format == xenos::TextureFormat::k_32_FLOAT ||
         format == xenos::TextureFormat::k_16_16_FLOAT;
}

// A fetch constant that cannot produce texels - not a texture fetch, no base
// level, stacked, or a resource dimension the shader doesn't fetch as - reads as
// zero on the guest, so bind a zero-cleared fallback view rather than rejecting
// the draw. Valid textures that merely use unsupported formats or unreadable
// memory are still rejected so real errors stay visible.
constexpr bool ShouldUseFallbackTexture(bool fetch_is_texture, bool has_base_level, bool stacked,
                                        bool dimension_matches) {
  return !fetch_is_texture || !has_base_level || stacked || !dimension_matches;
}

// Resolve offsets and sizes are 8x8-aligned, so a cached resolve can be up to
// one alignment unit wider and taller than the texture that samples it. Such a
// resolve can be cropped from its top-left corner instead of being rejected.
constexpr bool CanCropAlignedResolve(uint32_t cached_width, uint32_t cached_height,
                                     uint32_t requested_width, uint32_t requested_height) {
  return cached_width >= requested_width && cached_height >= requested_height &&
         cached_width - requested_width < xenos::kResolveAlignmentPixels &&
         cached_height - requested_height < xenos::kResolveAlignmentPixels;
}

// Cube textures are hosted as a cube-compatible image with all six faces as
// array layers of a single level; everything else is a single layer.
constexpr uint32_t kTextureCubeArrayLayers = 6;

// Cache key for a cube texture assembled from six GPU-resolved face images.
// The view swizzle is excluded because one cached cube image can be sampled
// with different component mappings.
struct ResolvedCubeKey {
  uint32_t address = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  auto operator<=>(const ResolvedCubeKey&) const = default;
};

struct ResolvedCubeKeyHash {
  size_t operator()(const ResolvedCubeKey& key) const noexcept {
    const std::array<uint32_t, 4> fields{key.address, key.width, key.height,
                                         uint32_t(key.format)};
    return size_t(XXH3_64bits(fields.data(), sizeof(fields)));
  }
};

// Each cube face is stored like its own 2D texture, `array_slice_stride_bytes`
// apart within the level. A face rendered on the GPU keeps its resolve at that
// same address, so a cube fetch has to look for the face images there.
constexpr uint32_t GetResolvedCubeFaceAddress(uint32_t base, uint32_t array_slice_stride_bytes,
                                              uint32_t face) {
  return base + array_slice_stride_bytes * face;
}

// A resolved cube face can stand in for guest memory only when its host image
// matches the texture format and covers the face within the resolve alignment,
// exactly like a 2D resolve crop.
constexpr bool CanUseResolvedCubeFace(bool format_matches, uint32_t face_width,
                                      uint32_t face_height, uint32_t texture_width,
                                      uint32_t texture_height) {
  return format_matches &&
         CanCropAlignedResolve(face_width, face_height, texture_width, texture_height);
}

// A cached GPU-resolved cube only needs to re-copy its six faces when it has
// never been uploaded or at least one face resolve version changed.
constexpr bool ShouldUploadResolvedCube(
    bool uploaded, const std::array<uint64_t, kTextureCubeArrayLayers>& cached_versions,
    const std::array<uint64_t, kTextureCubeArrayLayers>& current_versions) {
  return !uploaded || cached_versions != current_versions;
}

constexpr uint32_t GetTextureArrayLayers(xenos::DataDimension dimension) {
  return dimension == xenos::DataDimension::kCube ? kTextureCubeArrayLayers : 1;
}

// Array slices of a level are stored in guest memory `array_slice_stride_bytes`
// apart within the level, so the last slice's data has to be readable as well.
constexpr size_t GetTextureLevelReadSize(uint32_t array_slice_stride_bytes,
                                         uint32_t array_slice_data_extent_bytes,
                                         uint32_t array_layers) {
  return size_t(array_slice_stride_bytes) * (array_layers - 1) + array_slice_data_extent_bytes;
}

inline bool DecodeDmaIndices(std::span<const uint8_t> source, xenos::IndexFormat format,
                             xenos::Endian endian, std::span<uint32_t> indices) {
  const bool wide = format == xenos::IndexFormat::kInt32;
  const size_t stride = wide ? sizeof(uint32_t) : sizeof(uint16_t);
  if (indices.size() > source.size() / stride) {
    return false;
  }
  for (uint32_t& index : indices) {
    if (wide) {
      uint32_t value;
      std::memcpy(&value, source.data(), sizeof(value));
      index = xenos::GpuSwap(value, endian) & xenos::kVertexIndexMask;
    } else {
      uint16_t value;
      std::memcpy(&value, source.data(), sizeof(value));
      // Match the guest primitive processor's normalization for 16-bit DMA.
      index = endian == xenos::Endian::k8in16 || endian == xenos::Endian::k8in32
                  ? xenos::GpuSwap(value, xenos::Endian::k8in16)
                  : value;
    }
    source = source.subspan(stride);
  }
  return true;
}

// The translator expands each guest rectangle or point to four host vertices.
inline std::vector<uint32_t> QuadStripIndices(uint32_t primitive_count) {
  std::vector<uint32_t> indices;
  indices.reserve(5 * primitive_count);
  for (uint32_t primitive = 0; primitive < primitive_count; ++primitive) {
    if (primitive) {
      indices.push_back(UINT32_MAX);
    }
    for (uint32_t corner = 0; corner < 4; ++corner) {
      indices.push_back(primitive * 4 + corner);
    }
  }
  return indices;
}

// Converts the sanitized guest copy sample selection (the copy_sample_select of
// GetResolveInfo's destination coordinate info) to the sample index of the host
// multisampled image. 4x guest samples match the standard Vulkan sample
// locations; 2x guest samples are stored top/bottom, which the reference Vulkan
// resolve path converts through GetD3D10SampleIndexForGuest2xMSAA.
inline uint32_t GetResolveSampleIndex(xenos::MsaaSamples samples,
                                      xenos::CopySampleSelect selection) {
  if (samples == xenos::MsaaSamples::k2X) {
    return draw_util::GetD3D10SampleIndexForGuest2xMSAA(
        selection == xenos::CopySampleSelect::k1 ? 1 : 0, true);
  }
  return uint32_t(selection);
}

struct ResolveRectangle {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
};

// shortcut: blit unbiased, unswapped 1x UNORM copies until native packing supports other modes.
inline bool IsResolveFormatConvertSupported(const draw_util::ResolveInfo& copy,
                                            xenos::MsaaSamples samples,
                                            xenos::ColorRenderTargetFormat source_format,
                                            xenos::ColorFormat destination_format) {
  return samples == xenos::MsaaSamples::k1X &&
         source_format == xenos::ColorRenderTargetFormat::k_8_8_8_8 &&
         (destination_format == xenos::ColorFormat::k_8 ||
          destination_format == xenos::ColorFormat::k_8_8) &&
         copy.copy_dest_info.copy_dest_number ==
             xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction &&
         !copy.copy_dest_info.copy_dest_swap && !copy.copy_dest_info.copy_dest_exp_bias &&
         !copy.copy_dest_info.copy_dest_array;
}

// shortcut: cap attachments to full tile rows until native EDRAM supports split images.
constexpr uint32_t GetTargetHeight(uint32_t pitch, xenos::MsaaSamples samples,
                                   uint32_t tile_count, uint32_t max_height) {
  const uint32_t tiles_per_row = xenos::GetSurfacePitchTiles(pitch, samples, false);
  if (!tiles_per_row) {
    return 0;
  }
  const uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> (samples >= xenos::MsaaSamples::k2X);
  const uint32_t height = tile_count / tiles_per_row * tile_height;
  const uint32_t aligned_max = max_height / tile_height * tile_height;
  return height < aligned_max ? height : aligned_max;
}

// Xbox 360 D3D9 clears 1x- and 2x-MSAA EDRAM targets by issuing a 4x-MSAA
// depth-only draw at the same tile with half the horizontal pitch, so one host
// pixel of the alias covers scale_x by scale_y guest pixels.
struct FastClearScale {
  uint32_t x;
  uint32_t y;
};

constexpr FastClearScale GetMsaa4xAliasScale(xenos::MsaaSamples target_samples) {
  switch (target_samples) {
    case xenos::MsaaSamples::k1X:
      return {2, 2};
    case xenos::MsaaSamples::k2X:
      return {2, 1};
    default:
      return {1, 1};
  }
}

// Rescales a guest-space viewport, scissor and NDC scale/offset to cover the
// aliased target. When the scaled viewport would exceed the host limits, the
// extent is clamped and the NDC scale/offset are adjusted so the covered
// framebuffer coordinates stay identical.
inline void ScaleViewportAndScissor(draw_util::ViewportInfo& viewport, draw_util::Scissor& scissor,
                                    std::array<float, 3>& ndc_scale,
                                    std::array<float, 3>& ndc_offset, uint32_t scale_x,
                                    uint32_t scale_y, uint32_t max_viewport_w,
                                    uint32_t max_viewport_h) {
  scissor.offset[0] *= scale_x;
  scissor.offset[1] *= scale_y;
  scissor.extent[0] *= scale_x;
  scissor.extent[1] *= scale_y;
  viewport.xy_offset[0] *= scale_x;
  viewport.xy_offset[1] *= scale_y;
  for (uint32_t axis = 0; axis < 2; ++axis) {
    const uint32_t scale = axis ? scale_y : scale_x;
    const uint32_t max_viewport = axis ? max_viewport_h : max_viewport_w;
    const uint32_t orig_extent = viewport.xy_extent[axis];
    if (uint64_t(orig_extent) * scale <= max_viewport) {
      viewport.xy_extent[axis] = orig_extent * scale;
      continue;
    }
    viewport.xy_extent[axis] = max_viewport;
    const double k = (double(orig_extent) * double(scale)) / double(max_viewport);
    ndc_scale[axis] *= float(k);
    ndc_offset[axis] = float((double(ndc_offset[axis]) + 1.0) * k - 1.0);
  }
}

// Whether a 4x-MSAA depth-only draw is actually a D3D9 fast clear of an aliased
// 32bpp color target: it writes the whole depth range and stencil without a
// depth test and replaces or zeros the stencil, which is how the "color" is
// packed into the depth-stencil words.
constexpr bool IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat depth_format,
                                        reg::RB_DEPTHCONTROL depth_control,
                                        uint8_t stencil_write_mask) {
  return depth_format == xenos::DepthRenderTargetFormat::kD24S8 && depth_control.z_enable &&
         depth_control.z_write_enable &&
         depth_control.zfunc == xenos::CompareFunction::kAlways && depth_control.stencil_enable &&
         depth_control.stencilfunc == xenos::CompareFunction::kAlways &&
         (depth_control.stencilzpass == xenos::StencilOp::kReplace ||
          depth_control.stencilzpass == xenos::StencilOp::kZero) &&
         stencil_write_mask == 0xFF;
}

// Index of the shader packing mode that unpacks a 32-bit EDRAM clear color into
// an aliased color target of this host format, matching the format indices used
// by ColorResolve.
constexpr std::optional<uint32_t> GetEdramClearColorFormatMode(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
      return 0;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return 1;
    case VK_FORMAT_R16G16_SFLOAT:
      return 2;
    case VK_FORMAT_R32_SFLOAT:
      return 3;
    default:
      return std::nullopt;
  }
}

// Picks the host depth-stencil format for a guest depth render target. D24S8
// maps to D24_UNORM_S8_UINT only when the device supports it as both a sampled
// image and a depth/stencil attachment for optimal-tiling images; some drivers
// (notably AMD RADV) expose no such support, so it falls back to
// D32_SFLOAT_S8_UINT. D24FS8 always uses D32_SFLOAT_S8_UINT.
constexpr VkFormat SelectDepthFormat(xenos::DepthRenderTargetFormat format,
                                     bool depth_unorm24_supported) {
  return format == xenos::DepthRenderTargetFormat::kD24S8 && depth_unorm24_supported
             ? VK_FORMAT_D24_UNORM_S8_UINT
             : VK_FORMAT_D32_SFLOAT_S8_UINT;
}

// Rasterizer face culling for a host pipeline. Guest rectangle, point and line
// primitives are non-polygonal, so they have no front or back face and must not
// inherit the cull state left behind by the last polygonal draw.
struct CullFaces {
  bool front;
  bool back;
};
constexpr CullFaces GetCullFaces(bool primitive_polygonal, bool cull_front, bool cull_back) {
  return primitive_polygonal ? CullFaces{cull_front, cull_back} : CullFaces{false, false};
}

// The guest turns off clipping with PA_CL_CLIP_CNTL.CLIP_DISABLE and expects
// vertices outside the depth range to be clamped rather than discarded. Vulkan
// has no "no clipping" mode, so it is mapped to depth clamping where the device
// supports it. Without this, geometry such as the 4x-MSAA fast-clear quads,
// which clear with a near-zero Z, is clipped away entirely.
constexpr bool GetDepthClampEnable(bool device_depth_clamp, bool clip_disable) {
  return device_depth_clamp && clip_disable;
}

// Returns the least recently used cache entry whose GPU work has retired, so it
// can be dropped when the cache reaches its limit. Entries last used in the
// protected submissions (the one being recorded and the one still in flight) may
// still be referenced by queued or submitted commands, so they are skipped -
// pass first_protected_submission = last_retired_submission + 1 to keep only
// fences that have already been waited for. The caller's may_evict predicate
// can preserve the presented image. Nullopt if nothing can be evicted safely.
template <typename EntryMap, typename Predicate>
std::optional<typename EntryMap::key_type> SelectRetiredEviction(
    const EntryMap& entries, uint64_t first_protected_submission, Predicate may_evict) {
  std::optional<typename EntryMap::key_type> victim;
  uint64_t victim_use = UINT64_MAX;
  for (const auto& [key, entry] : entries) {
    if (entry.last_use_submission >= first_protected_submission || !may_evict(key, entry)) {
      continue;
    }
    if (!victim || entry.last_use_submission < victim_use) {
      victim = key;
      victim_use = entry.last_use_submission;
    }
  }
  return victim;
}

// Place a resolve in the cached image at its absolute guest texture coordinates.
inline std::optional<ResolveRectangle> GetCachedResolveRectangle(const draw_util::ResolveInfo& copy,
                                                                 uint32_t image_width,
                                                                 uint32_t image_height) {
  const uint32_t width = uint32_t(copy.coordinate_info.width_div_8)
                         << xenos::kResolveAlignmentPixelsLog2;
  const uint32_t height = copy.height_div_8 << xenos::kResolveAlignmentPixelsLog2;
  const uint32_t x = copy.copy_dest_x;
  const uint32_t y = copy.copy_dest_y;
  if (!width || !height || x > image_width || y > image_height || width > image_width - x ||
      height > image_height - y) {
    return std::nullopt;
  }
  return ResolveRectangle{x, y, width, height};
}

inline std::optional<ResolveRectangle> GetResolveRectangle(const draw_util::ResolveInfo& copy,
                                                           EdramLayout layout, uint32_t width,
                                                           uint32_t height, bool depth = false) {
  const auto& target = depth ? copy.depth_edram_info : copy.color_edram_info;
  const uint32_t pitch = target.pitch_tiles;
  if (!pitch) {
    return std::nullopt;
  }
  const uint32_t base = depth ? copy.depth_original_base : copy.color_original_base;
  const uint32_t tiles = layout.Wrap(target.base_tiles - base);
  const uint32_t tile_width =
      xenos::kEdramTileWidthSamples >> (target.msaa_samples >= xenos::MsaaSamples::k4X);
  const uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> (target.msaa_samples >= xenos::MsaaSamples::k2X);
  const ResolveRectangle rect{
      tiles % pitch * tile_width + copy.coordinate_info.edram_offset_x_div_8 * 8,
      tiles / pitch * tile_height + copy.coordinate_info.edram_offset_y_div_8 * 8,
      uint32_t(copy.coordinate_info.width_div_8) * 8, copy.height_div_8 * 8};
  if (rect.x > width || rect.y > height || !rect.width || !rect.height ||
      rect.width > width - rect.x || rect.height > height - rect.y) {
    return std::nullopt;
  }
  return rect;
}

}  // namespace rex::graphics::native
