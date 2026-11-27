#include "texture_cache.h"

#include <algorithm>
#include <bit>
#include <cstring>

#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {
namespace {

VkFilter ToFilter(xenos::TextureFilter filter) {
  return filter == xenos::TextureFilter::kLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

// Converts one endian-swapped guest texel (or block) to host bytes.
void UnpackTexel(const TextureFormatInfo& info, bool is_signed, const uint8_t* guest,
                 uint8_t* host) {
  uint32_t texel = 0;
  std::memcpy(&texel, guest, std::min(info.block_size, uint32_t(sizeof(texel))));
  switch (info.unpack) {
    case TextureUnpack::kDepthUnorm24:
    case TextureUnpack::kDepthFloat24: {
      const float depth = info.unpack == TextureUnpack::kDepthFloat24
                              ? xenos::Float20e4To32(texel >> 8)
                              : float(texel >> 8) / 16777215.0f;
      std::memcpy(host, &depth, sizeof(depth));
      return;
    }
    case TextureUnpack::kRgba4ToRgba8:
    case TextureUnpack::kRgb5a1ToRgba8: {
      const uint32_t rgba8 = info.unpack == TextureUnpack::kRgba4ToRgba8
                                 ? UnpackRgba4ToRgba8(uint16_t(texel))
                                 : UnpackRgb5a1ToRgba8(uint16_t(texel));
      std::memcpy(host, &rgba8, sizeof(rgba8));
      return;
    }
    case TextureUnpack::kR11G11B10ToRgba16:
    case TextureUnpack::kR10G11B11ToRgba16: {
      const auto rgba16 = UnpackR11G11B10ToRgba16(
          texel, info.unpack == TextureUnpack::kR10G11B11ToRgba16, is_signed);
      std::memcpy(host, rgba16.data(), sizeof(rgba16));
      return;
    }
    default:
      std::memcpy(host, guest, info.block_size);
      return;
  }
}

}  // namespace

NativeTextureCache::~NativeTextureCache() {
  // Draws still referencing cached textures hold their own shared pointers.
  cached_textures_.clear();
  resolved_cubes_.clear();
  resolved_crops_.clear();
  resolved_images_.clear();
  ClearSamplers();
  DestroyNullTextures();
}

bool NativeTextureCache::Initialize(bool watch_pages) {
  if (watch_pages) {
    const size_t words = (kSharedVertexPageCount + 63) / 64;
    cached_only_pages_.assign(words, 0);
    resolved_pages_.assign(words, 0);
    scratch_resolved_pages_.assign(words, 0);
  }
  return InitializeNullTextures();
}

void NativeTextureCache::InvalidateAll() {
  cached_textures_.clear();
  for (auto& bucket : texture_buckets_) {
    bucket.clear();
  }
  resolved_images_.clear();
  resolved_cubes_.clear();
  resolved_crops_.clear();
  presented_resolve_address_.reset();
  for (auto& w : cached_only_pages_) {
    std::atomic_ref<uint64_t>(w).store(0, std::memory_order_relaxed);
  }
  for (auto& w : resolved_pages_) {
    std::atomic_ref<uint64_t>(w).store(0, std::memory_order_relaxed);
  }
}

void NativeTextureCache::ClearSamplers() {
  for (const auto& entry : samplers_) {
    device_.functions().vkDestroySampler(device_.device(), entry.second.sampler, nullptr);
  }
  samplers_.clear();
  custom_border_color_samplers_ = 0;
}

void NativeTextureCache::MarkCachedPages(uint32_t start, uint32_t length) {
  if (resolved_pages_.empty()) {
    return;
  }
  const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
  AtomicSetPageRangeBits(resolved_pages_, first_page, page_count);
}

std::pair<uint32_t, uint32_t> NativeTextureCache::WidenInvalidation(uint32_t start,
                                                                    uint32_t length) {
  if (cached_only_pages_.empty() && resolved_pages_.empty()) {
    return {start, 0};
  }
  const uint32_t max_blocks =
      uint32_t(std::max(cached_only_pages_.size(), resolved_pages_.size()));
  return WidenInvalidationPageRangeWithMask(
      [this](uint32_t b) { return ProtectedPagesWord(b); }, max_blocks, start, length);
}

bool NativeTextureCache::InitializeNullTextures() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  image.imageType = VK_IMAGE_TYPE_2D;
  image.format = ColorImage::kFormat;
  image.extent = {1, 1, 1};
  image.mipLevels = 1;
  image.arrayLayers = 6;
  image.samples = VK_SAMPLE_COUNT_1_BIT;
  image.tiling = VK_IMAGE_TILING_OPTIMAL;
  image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          &device_, image, ui::vulkan::util::MemoryPurpose::kDeviceLocal, null_texture_image_,
          null_texture_memory_)) {
    return false;
  }
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = null_texture_image_;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view.format = ColorImage::kFormat;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vk.vkCreateImageView(device, &view, nullptr, &null_texture_2d_view_) != VK_SUCCESS) {
    return false;
  }
  view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  view.subresourceRange.layerCount = 6;
  if (vk.vkCreateImageView(device, &view, nullptr, &null_texture_cube_view_) != VK_SUCCESS) {
    return false;
  }
  image.flags = 0;
  image.imageType = VK_IMAGE_TYPE_3D;
  image.arrayLayers = 1;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          &device_, image, ui::vulkan::util::MemoryPurpose::kDeviceLocal, null_texture_3d_image_,
          null_texture_3d_memory_)) {
    return false;
  }
  view.image = null_texture_3d_image_;
  view.viewType = VK_IMAGE_VIEW_TYPE_3D;
  view.subresourceRange.layerCount = 1;
  return vk.vkCreateImageView(device, &view, nullptr, &null_texture_3d_view_) == VK_SUCCESS;
}

void NativeTextureCache::DestroyNullTextures() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  vk.vkDestroyImageView(device, null_texture_cube_view_, nullptr);
  vk.vkDestroyImageView(device, null_texture_2d_view_, nullptr);
  vk.vkDestroyImage(device, null_texture_image_, nullptr);
  vk.vkFreeMemory(device, null_texture_memory_, nullptr);
  vk.vkDestroyImageView(device, null_texture_3d_view_, nullptr);
  vk.vkDestroyImage(device, null_texture_3d_image_, nullptr);
  vk.vkFreeMemory(device, null_texture_3d_memory_, nullptr);
  null_texture_3d_view_ = VK_NULL_HANDLE;
  null_texture_3d_image_ = VK_NULL_HANDLE;
  null_texture_3d_memory_ = VK_NULL_HANDLE;
  null_texture_cube_view_ = null_texture_2d_view_ = VK_NULL_HANDLE;
  null_texture_image_ = VK_NULL_HANDLE;
  null_texture_memory_ = VK_NULL_HANDLE;
  null_textures_initialized_ = false;
}

void NativeTextureCache::RecordNullTextureInitialization(VkCommandBuffer commands) {
  if (null_textures_initialized_ || null_texture_image_ == VK_NULL_HANDLE ||
      null_texture_3d_image_ == VK_NULL_HANDLE) {
    return;
  }
  const auto& vk = device_.functions();
  const std::array<VkImage, 2> images{null_texture_image_, null_texture_3d_image_};
  const std::array<uint32_t, 2> layers{6, 1};
  std::array<VkImageMemoryBarrier, 2> barriers{};
  for (size_t i = 0; i < images.size(); ++i) {
    auto& barrier = barriers[i];
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = images[i];
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers[i]};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  }
  vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                          uint32_t(barriers.size()), barriers.data());
  const VkClearColorValue zero{};
  for (auto& barrier : barriers) {
    vk.vkCmdClearColorImage(commands, barrier.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero,
                            1, &barrier.subresourceRange);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, uint32_t(barriers.size()), barriers.data());
  null_textures_initialized_ = true;
}

bool NativeTextureCache::ReadTextures(const SpirvShader& shader, bool vertex, DrawInput& input,
                                      ui::vulkan::VulkanUploadBufferPool* uploads,
                                      uint64_t submission) {
  const auto& textures = shader.GetTextureBindingsAfterTranslation();
  const auto& samplers = shader.GetSamplerBindingsAfterTranslation();
  const auto& properties = device_.properties();
  if (textures.size() > properties.maxPerStageDescriptorSampledImages ||
      samplers.size() > properties.maxPerStageDescriptorSamplers) {
    return false;
  }
  auto& texture_bindings = vertex ? input.vertex_texture_bindings : input.texture_bindings;
  auto& sampler_handles = vertex ? input.vertex_samplers : input.samplers;
  const auto read = [&](uint32_t slot, const xenos::xe_gpu_texture_fetch_t& fetch,
                        bool signed_variant) {
    auto& texture = input.textures.emplace_back();
    if (!ReadTexture(fetch, texture, signed_variant, uploads, submission)) {
      REXGPU_DEBUG(
          "Native Vulkan texture {} unsupported: format={} dimension={} base={:05X} "
          "flags={:08X}/{:08X}",
          slot, uint32_t(fetch.format), uint32_t(fetch.dimension), uint32_t(fetch.base_address),
          fetch.dword_4, fetch.dword_5);
      return false;
    }
    input.system.texture_swizzled_signs[slot / 4] |= uint32_t(texture_util::SwizzleSigns(fetch))
                                                     << ((slot % 4) * 8);
    if (texture.resolution_scaled) {
      input.system.textures_resolution_scaled |= uint32_t(1) << slot;
    }
    if (!properties.imageViewFormatSwizzle) {
      PackSystemTextureSwizzle(input.system.texture_swizzles, slot, texture.host_swizzle);
    }
    return true;
  };

  for (const auto& binding : textures) {
    const uint32_t slot = binding.fetch_constant;
    const auto fetch = regs_.GetTextureFetch(slot);
    uint32_t base_page = 0, mip_page = 0;
    texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, &base_page,
                                                   &mip_page, nullptr, nullptr);
    // A fetch constant that isn't a usable texture of the binding's dimension
    // reads as zero on the guest - bind a zero-cleared fallback of the same
    // dimension rather than dropping the draw. Its signs stay unsigned.
    if (ShouldUseFallbackTexture(fetch.type == xenos::FetchConstantType::kTexture,
                                 base_page || mip_page,
                                 TextureBindingMatchesDimension(binding.dimension,
                                                                fetch.dimension))) {
      input.textures.emplace_back().fallback_view =
          binding.dimension == xenos::FetchOpDimension::kCube           ? null_texture_cube_view_
          : binding.dimension == xenos::FetchOpDimension::k3DOrStacked ? null_texture_3d_view_
                                                                        : null_texture_2d_view_;
      texture_bindings.push_back(uint32_t(input.textures.size() - 1));
      continue;
    }
    // A signed binding is only sampled for the signed components, and float
    // formats are signed in their unsigned view already.
    const TextureFormatInfo info = GetTextureFormatInfo(fetch.format);
    const bool sampled_signed =
        binding.is_signed && texture_util::IsAnySignSigned(texture_util::SwizzleSigns(fetch));
    if (sampled_signed && info.signed_format == VK_FORMAT_UNDEFINED) {
      REXGPU_DEBUG("Native Vulkan texture {} format {} has no signed host format", slot,
                   uint32_t(fetch.format));
      return false;
    }
    const bool signed_variant = sampled_signed && info.signed_format != info.format;
    int& index = input.texture_indices[slot * 2 + (signed_variant ? 1 : 0)];
    if (index < 0) {
      if (signed_variant && !info.unpack_depends_on_sign()) {
        // The unsigned texels, viewed through the SNORM format.
        int& unsigned_index = input.texture_indices[slot * 2];
        if (unsigned_index < 0) {
          if (!read(slot, fetch, false)) {
            return false;
          }
          unsigned_index = int(input.textures.size() - 1);
        }
        DrawTexture texture = input.textures[unsigned_index];
        if (!MakeSignedView(texture, info.signed_format)) {
          return false;
        }
        input.textures.push_back(std::move(texture));
      } else if (!read(slot, fetch, signed_variant)) {
        return false;
      }
      index = int(input.textures.size() - 1);
    }
    texture_bindings.push_back(uint32_t(index));
  }

  for (const auto& sampler : samplers) {
    const auto fetch = regs_.GetTextureFetch(sampler.fetch_constant);
    DrawSampler out;
    out.min_filter =
        ToFilter(sampler.min_filter == xenos::TextureFilter::kUseFetchConst ? fetch.min_filter
                                                                            : sampler.min_filter);
    out.mag_filter =
        ToFilter(sampler.mag_filter == xenos::TextureFilter::kUseFetchConst ? fetch.mag_filter
                                                                            : sampler.mag_filter);
    const auto mip_filter = sampler.mip_filter == xenos::TextureFilter::kUseFetchConst
                                ? fetch.mip_filter
                                : sampler.mip_filter;
    out.mip_mode = mip_filter == xenos::TextureFilter::kLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                               : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    xenos::ClampMode clamp_x, clamp_y, clamp_z;
    texture_util::GetClampModesForDimension(fetch, clamp_x, clamp_y, clamp_z);
    out.address_u = GetSamplerAddressMode(clamp_x, properties.samplerMirrorClampToEdge);
    out.address_v = GetSamplerAddressMode(clamp_y, properties.samplerMirrorClampToEdge);
    out.address_w = GetSamplerAddressMode(clamp_z, properties.samplerMirrorClampToEdge);
    out.border_color = fetch.border_color;
    uint32_t mip_min = 0, mip_max = 0;
    texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, nullptr,
                                                   nullptr, &mip_min, &mip_max);
    // LOD bias is applied by the translated shader, so mipLodBias stays 0.
    const bool base_map = mip_filter == xenos::TextureFilter::kBaseMap;
    out.min_lod = float(mip_min);
    out.max_lod = base_map ? out.min_lod + 0.25f : float(mip_max);
    const auto aniso = sampler.aniso_filter == xenos::AnisoFilter::kUseFetchConst
                           ? fetch.aniso_filter
                           : sampler.aniso_filter;
    const bool override_eligible =
        mip_max > mip_min && !base_map && out.min_filter == VK_FILTER_LINEAR &&
        out.mag_filter == VK_FILTER_LINEAR &&
        (mip_filter == xenos::TextureFilter::kPoint || mip_filter == xenos::TextureFilter::kLinear);
    out.max_anisotropy = GetSamplerAnisotropy(
        aniso, REXCVAR_GET(anisotropic_override), override_eligible,
        properties.samplerAnisotropy ? properties.maxSamplerAnisotropy : 0.0f);
    const TextureFormatInfo info = GetTextureFormatInfo(fetch.format);
    const bool any_signed = texture_util::IsAnySignSigned(texture_util::SwizzleSigns(fetch));
    const bool linear_filterable =
        info.format != VK_FORMAT_UNDEFINED && IsLinearFilterable(fetch.format, info.format) &&
        (!any_signed || info.signed_format == VK_FORMAT_UNDEFINED ||
         IsLinearFilterable(fetch.format, info.signed_format));
    const VkSampler handle = GetSampler(NormalizeDrawSampler(out, linear_filterable));
    if (handle == VK_NULL_HANDLE) {
      return false;
    }
    sampler_handles.push_back(handle);
  }
  return true;
}

bool NativeTextureCache::MakeSignedView(DrawTexture& texture, VkFormat signed_format) {
  // GPU resolves are kept in their unsigned render target format only.
  if (texture.resolved_image || texture.crop_source || texture.resolved_cube_faces[0]) {
    return false;
  }
  texture.format = signed_format;
  if (!texture.cached_texture) {
    // Uploads its own copy of the texels in the signed format.
    return true;
  }
  texture.cached_view = texture.cached_texture->GetOrCreateView(
      texture.view_type, signed_format, texture.swizzle,
      texture.view_type == VK_IMAGE_VIEW_TYPE_3D ? 1 : texture.array_layers);
  // The unsigned texture bound earlier in the draw uploads the shared image.
  texture.upload_cached = false;
  return texture.cached_view != VK_NULL_HANDLE;
}

bool NativeTextureCache::IsLinearFilterable(xenos::TextureFormat guest_format, VkFormat format) {
  const size_t idx = size_t(guest_format);
  if (idx < linear_filterable_.size() && linear_filterable_[idx] >= 0) {
    return linear_filterable_[idx] != 0;
  }
  VkFormatProperties format_properties{};
  device_.vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
      device_.physical_device(), format, &format_properties);
  const bool supported = (format_properties.optimalTilingFeatures &
                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
  if (idx < linear_filterable_.size()) {
    linear_filterable_[idx] = supported ? 1 : 0;
  }
  return supported;
}

VkSampler NativeTextureCache::GetSampler(const DrawSampler& key) {
  auto existing = samplers_.find(key);
  if (existing != samplers_.end()) {
    existing->second.last_use_submission = clock_.next;
    return existing->second.sampler;
  }
  const auto& properties = device_.properties();
  if (samplers_.size() >= kMaxSamplers) {
    const auto victim =
        SelectRetiredEviction(samplers_, clock_.retired + 1,
                              [](const DrawSampler&, const CachedSampler&) { return true; });
    if (!victim) {
      return VK_NULL_HANDLE;
    }
    auto victim_it = samplers_.find(*victim);
    custom_border_color_samplers_ -= victim_it->second.custom_border_color ? 1 : 0;
    device_.functions().vkDestroySampler(device_.device(), victim_it->second.sampler, nullptr);
    samplers_.erase(victim_it);
  }
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  if (properties.nonSeamlessCubeMap && REXCVAR_GET(non_seamless_cube_map)) {
    info.flags |= VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT;
  }
  info.minFilter = key.min_filter;
  info.magFilter = key.mag_filter;
  info.mipmapMode = key.mip_mode;
  info.minLod = key.min_lod;
  info.maxLod = key.max_lod;
  info.addressModeU = key.address_u;
  info.addressModeV = key.address_v;
  info.addressModeW = key.address_w;
  if (key.max_anisotropy != 0.0f) {
    info.anisotropyEnable = VK_TRUE;
    info.maxAnisotropy = key.max_anisotropy;
  }
  // YCbCr black borders are 0.5 in the chroma components, which only a custom
  // border color can express; out of custom samplers, they're transparent black.
  VkSamplerCustomBorderColorCreateInfoEXT custom{
      VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
  bool custom_border_color = false;
  switch (key.border_color) {
    case xenos::BorderColor::k_ABGR_White:
      info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      break;
    case xenos::BorderColor::k_ACBYCR_Black:
    case xenos::BorderColor::k_ACBCRY_Black:
      if (properties.customBorderColors &&
          custom_border_color_samplers_ < properties.maxCustomBorderColorSamplers) {
        const bool cbycr = key.border_color == xenos::BorderColor::k_ACBYCR_Black;
        custom.customBorderColor.float32[0] = cbycr ? 0.5f : 0.0f;
        custom.customBorderColor.float32[1] = cbycr ? 0.0f : 0.5f;
        custom.customBorderColor.float32[2] = 0.5f;
        custom.customBorderColor.float32[3] = 0.0f;
        custom.format = properties.customBorderColorWithoutFormat
                            ? VK_FORMAT_UNDEFINED
                            : VK_FORMAT_R32G32B32A32_SFLOAT;
        info.pNext = &custom;
        info.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        custom_border_color = true;
        break;
      }
      [[fallthrough]];
    default:
      info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      break;
  }
  VkSampler sampler = VK_NULL_HANDLE;
  if (device_.functions().vkCreateSampler(device_.device(), &info, nullptr, &sampler) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  custom_border_color_samplers_ += custom_border_color ? 1 : 0;
  samplers_.emplace(key, CachedSampler{sampler, clock_.next, custom_border_color});
  return sampler;
}

void NativeTextureCache::InvalidateTextures(uint32_t start, uint32_t length) {
  const PhysicalRange range{start, length};
  InvalidateRanges({&range, 1}, InvalidationTarget::kTextures);
}

void NativeTextureCache::InvalidateResolvedTextures(uint32_t start, uint32_t length) {
  const PhysicalRange range{start, length};
  InvalidateRanges({&range, 1}, InvalidationTarget::kResolves);
}

void NativeTextureCache::InvalidateWrittenRanges(std::span<const PhysicalRange> ranges) {
  InvalidateRanges(ranges, InvalidationTarget::kBoth);
}

void NativeTextureCache::AddTextureToBuckets(const GuestTextureKey& key,
                                             const CachedTexture& texture) {
  auto add_range = [&](uint32_t start, uint32_t length, int32_t skip_first, int32_t skip_last) {
    if (!length || start >= kPhysicalMemorySize) {
      return std::pair<int32_t, int32_t>{-1, -2};
    }
    const uint32_t clamped = std::min(length, kPhysicalMemorySize - start);
    const uint32_t b_first = start >> kBucketShift;
    const uint32_t b_last = (start + clamped - 1) >> kBucketShift;
    for (uint32_t b = b_first; b <= b_last; ++b) {
      if (int32_t(b) >= skip_first && int32_t(b) <= skip_last) {
        continue;
      }
      texture_buckets_[b].push_back(&key);
    }
    return std::pair<int32_t, int32_t>{int32_t(b_first), int32_t(b_last)};
  };
  const auto [bf, bl] = add_range(texture.base_start, texture.base_length, -1, -2);
  add_range(texture.mip_start, texture.mip_length, bf, bl);
}

void NativeTextureCache::RemoveTextureFromBuckets(const GuestTextureKey& key,
                                                  const CachedTexture& texture,
                                                  std::vector<uint32_t>* dirty_buckets) {
  auto remove_range = [&](uint32_t start, uint32_t length) {
    if (!length || start >= kPhysicalMemorySize) {
      return;
    }
    const uint32_t clamped = std::min(length, kPhysicalMemorySize - start);
    const uint32_t b_first = start >> kBucketShift;
    const uint32_t b_last = (start + clamped - 1) >> kBucketShift;
    for (uint32_t b = b_first; b <= b_last; ++b) {
      auto& bucket = texture_buckets_[b];
      for (size_t i = 0; i < bucket.size(); ++i) {
        if (bucket[i] == &key) {
          bucket[i] = bucket.back();
          bucket.pop_back();
          if (dirty_buckets) {
            dirty_buckets->push_back(b);
          }
          break;
        }
      }
    }
  };
  remove_range(texture.base_start, texture.base_length);
  remove_range(texture.mip_start, texture.mip_length);
}

void NativeTextureCache::RecomputeTexturePageWord(uint32_t block) {
  if (block >= cached_only_pages_.size()) {
    return;
  }
  const uint32_t block_page_base = block << 6;
  uint64_t mask = 0;
  auto accum_range = [&](uint32_t start, uint32_t length) {
    const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
    if (!page_count) {
      return;
    }
    const uint32_t end_page = first_page + page_count;
    const uint32_t lo = std::max(first_page, block_page_base);
    const uint32_t hi = std::min(end_page, block_page_base + 64u);
    if (lo >= hi) {
      return;
    }
    const uint32_t bit = lo - block_page_base;
    const uint32_t count = hi - lo;
    mask |= (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
  };
  for (const GuestTextureKey* key : texture_buckets_[block]) {
    const auto it = cached_textures_.find(*key);
    if (it == cached_textures_.end()) {
      continue;
    }
    const CachedTexture& texture = *it->second.texture;
    accum_range(texture.base_start, texture.base_length);
    accum_range(texture.mip_start, texture.mip_length);
  }
  std::atomic_ref<uint64_t>(cached_only_pages_[block]).store(mask, std::memory_order_relaxed);
}

void NativeTextureCache::RecomputeResolvedPages() {
  if (resolved_pages_.empty()) {
    return;
  }
  std::fill(scratch_resolved_pages_.begin(), scratch_resolved_pages_.end(), 0);
  for (const auto& [addr, entry] : resolved_images_) {
    const auto [first, count] = GetPhysicalPageRange(entry.guest_start, entry.guest_length);
    SetPageRangeBits(scratch_resolved_pages_, first, count);
  }
  for (size_t i = 0; i < resolved_pages_.size(); ++i) {
    std::atomic_ref<uint64_t>(resolved_pages_[i])
        .store(scratch_resolved_pages_[i], std::memory_order_relaxed);
  }
}

void NativeTextureCache::InvalidateRanges(std::span<const PhysicalRange> ranges,
                                          InvalidationTarget target) {
  if (ranges.empty()) return;
  if (target != InvalidationTarget::kResolves) {
    const bool check_textures =
        cached_only_pages_.empty() ||
        std::any_of(ranges.begin(), ranges.end(), [&](const PhysicalRange& range) {
          const auto [first, count] = GetPhysicalPageRange(range.first, range.second);
          return AtomicHasAnyPageBitSet(cached_only_pages_, first, count);
        });
    if (check_textures) {
      std::vector<const GuestTextureKey*> victims;
      std::vector<uint32_t> dirty_buckets;
      for (const auto& [r_start, r_len] : ranges) {
        if (!r_len || r_start >= kPhysicalMemorySize) {
          continue;
        }
        const uint32_t clamped = std::min(r_len, kPhysicalMemorySize - r_start);
        const uint32_t b_first = r_start >> kBucketShift;
        const uint32_t b_last = (r_start + clamped - 1) >> kBucketShift;
        for (uint32_t b = b_first; b <= b_last; ++b) {
          if (!cached_only_pages_.empty() &&
              std::atomic_ref<const uint64_t>(cached_only_pages_[b])
                      .load(std::memory_order_relaxed) == 0) {
            continue;
          }
          for (const GuestTextureKey* key : texture_buckets_[b]) {
            const auto it = cached_textures_.find(*key);
            if (it == cached_textures_.end()) {
              continue;
            }
            const CachedTexture& texture = *it->second.texture;
            if (OverlapsPhysicalRanges(ranges, texture.base_start, texture.base_length) ||
                OverlapsPhysicalRanges(ranges, texture.mip_start, texture.mip_length)) {
              victims.push_back(key);
            }
          }
        }
      }
      if (!victims.empty()) {
        std::sort(victims.begin(), victims.end());
        victims.erase(std::unique(victims.begin(), victims.end()), victims.end());
        for (const GuestTextureKey* key : victims) {
          const auto it = cached_textures_.find(*key);
          if (it == cached_textures_.end()) {
            continue;
          }
          RemoveTextureFromBuckets(*key, *it->second.texture, &dirty_buckets);
          cached_textures_.erase(it);
        }
        if (!cached_only_pages_.empty()) {
          std::sort(dirty_buckets.begin(), dirty_buckets.end());
          dirty_buckets.erase(std::unique(dirty_buckets.begin(), dirty_buckets.end()),
                              dirty_buckets.end());
          for (uint32_t b : dirty_buckets) {
            RecomputeTexturePageWord(b);
          }
        }
      }
    }
  }
  if (target != InvalidationTarget::kTextures) {
    const bool check_resolves =
        resolved_pages_.empty() ||
        std::any_of(ranges.begin(), ranges.end(), [&](const PhysicalRange& range) {
          const auto [first, count] = GetPhysicalPageRange(range.first, range.second);
          return AtomicHasAnyPageBitSet(resolved_pages_, first, count);
        });
    if (check_resolves) {
      const bool resolves_erased =
          std::erase_if(resolved_images_, [&](const auto& entry) {
            if (!OverlapsPhysicalRanges(ranges, entry.second.guest_start,
                                        entry.second.guest_length)) {
              return false;
            }
            if (presented_resolve_address_ == entry.first) {
              presented_resolve_address_.reset();
            }
            return true;
          }) != 0;
      if (resolves_erased) {
        PruneResolvedDependents();
        RecomputeResolvedPages();
      }
    }
  }
}

void NativeTextureCache::PruneResolvedDependents() {
  for (auto it = resolved_crops_.begin(); it != resolved_crops_.end();) {
    if (!resolved_images_.contains(it->first.address)) {
      it = resolved_crops_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = resolved_cubes_.begin(); it != resolved_cubes_.end();) {
    bool alive = true;
    for (uint32_t face_addr : it->second.texture->resolved_face_addresses) {
      if (!face_addr || !resolved_images_.contains(face_addr)) {
        alive = false;
        break;
      }
    }
    if (!alive) {
      it = resolved_cubes_.erase(it);
    } else {
      ++it;
    }
  }
}

void NativeTextureCache::RebuildCachedTexturePages() { RecomputeResolvedPages(); }

template <typename EntryMap, typename Key, typename OnCreated>
std::shared_ptr<CachedTexture> NativeTextureCache::GetOrCreateCachedTexture(
    EntryMap& cache, size_t max_entries, const Key& key, const VkImageCreateInfo& image_info,
    OnCreated&& on_created, bool& failed) {
  failed = false;
  if (auto found = cache.find(key); found != cache.end()) {
    found->second.last_use_submission = clock_.next;
    found->second.texture->last_use_submission = clock_.next;
    return found->second.texture;
  }
  if (cache.size() >= max_entries) {
    const auto victim = SelectRetiredEviction(
        cache, clock_.retired + 1, [](const Key&, const CachedTextureEntry&) { return true; });
    if (victim) {
      if constexpr (std::is_same_v<Key, GuestTextureKey>) {
        if (auto vit = cache.find(*victim); vit != cache.end()) {
          std::vector<uint32_t> dirty_buckets;
          RemoveTextureFromBuckets(vit->first, *vit->second.texture, &dirty_buckets);
          cache.erase(vit);
          if (!cached_only_pages_.empty()) {
            std::sort(dirty_buckets.begin(), dirty_buckets.end());
            dirty_buckets.erase(std::unique(dirty_buckets.begin(), dirty_buckets.end()),
                                dirty_buckets.end());
            for (uint32_t b : dirty_buckets) {
              RecomputeTexturePageWord(b);
            }
          }
        }
      } else {
        cache.erase(*victim);
      }
    }
  }
  if (cache.size() >= max_entries) {
    return {};
  }
  auto entry = std::make_shared<CachedTexture>();
  entry->device = &device_;
  entry->allocator = allocator_;
  entry->level_count = image_info.mipLevels;
  entry->last_use_submission = clock_.next;
  if (!CreateDeviceImage(device_, allocator_, image_info, entry->image, entry->memory,
                         entry->allocation)) {
    failed = true;
    return {};
  }
  on_created(*entry);
  auto [inserted, _] = cache.emplace(key, CachedTextureEntry{entry, clock_.next});
  if constexpr (std::is_same_v<Key, GuestTextureKey>) {
    AddTextureToBuckets(inserted->first, *entry);
  }
  return entry;
}

bool NativeTextureCache::ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch,
                                     DrawTexture& texture, bool signed_variant,
                                     ui::vulkan::VulkanUploadBufferPool* uploads,
                                     uint64_t submission) {
  const TextureFormatInfo info = GetTextureFormatInfo(fetch.format);
  if (fetch.type != xenos::FetchConstantType::kTexture || info.format == VK_FORMAT_UNDEFINED) {
    return false;
  }
  const bool cube = fetch.dimension == xenos::DataDimension::kCube;
  const bool is_3d = fetch.dimension == xenos::DataDimension::k3D;
  uint32_t width_minus_1, height_minus_1, depth_minus_1;
  uint32_t base_page, mip_page, mip_min, mip_max;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &width_minus_1, &height_minus_1,
                                                 &depth_minus_1, &base_page, &mip_page, &mip_min,
                                                 &mip_max);
  texture.width = width_minus_1 + 1;
  texture.height = height_minus_1 + 1;
  texture.depth = is_3d ? depth_minus_1 + 1 : 1;
  // Cube faces and stacked slices are array layers; 3D slices are depth.
  texture.array_layers = is_3d ? 1 : depth_minus_1 + 1;
  if ((cube && texture.width != texture.height) || (!base_page && !mip_page)) {
    return false;
  }
  texture.view_type = cube    ? VK_IMAGE_VIEW_TYPE_CUBE
                      : is_3d ? VK_IMAGE_VIEW_TYPE_3D
                              : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  const bool sign_unpacked = signed_variant && info.unpack_depends_on_sign();
  texture.format = sign_unpacked ? info.signed_format : info.format;
  texture.mutable_format = info.needs_signed_view() &&
                           texture_util::IsAnySignSigned(texture_util::SwizzleSigns(fetch));
  const uint32_t block_width = info.block_width;
  const uint32_t block_size = info.block_size;
  const bool depth_texture = info.unpack == TextureUnpack::kDepthUnorm24 ||
                             info.unpack == TextureUnpack::kDepthFloat24;
  // Single 2D or cube level resolves only.
  const bool resolvable =
      base_page && !mip_max && !is_3d && (cube || texture.array_layers == 1);

  // Replicate the guest components that the format actually stores, matching
  // the Xenos Vulkan host format swizzle: RRRR for one channel, RGGG for two,
  // RGBA otherwise. Block sizes don't map to channels, so use the count.
  const uint32_t channel_count = TextureChannelCount(fetch.format);
  const uint32_t address = base_page << 12;
  const bool use_view_swizzle = device_.properties().imageViewFormatSwizzle;
  auto resolved = resolvable ? resolved_images_.find(address) : resolved_images_.end();
  // ponytail: array resolves stay in their cached layers; 3D textures decode guest memory.
  if (resolved != resolved_images_.end() && resolved->second.image->array_layers() != 1) {
    resolved = resolved_images_.end();
  }
  if (cube && resolved != resolved_images_.end()) {
    const uint32_t res_sx = resolved->second.resolution_scaled ? draw_resolution_scale_x_ : 1;
    const uint32_t res_sy = resolved->second.resolution_scaled ? draw_resolution_scale_y_ : 1;
    const uint32_t scaled_width = texture.width * res_sx;
    const uint32_t scaled_height = texture.height * res_sy;
    const bool is_scaled = res_sx > 1 || res_sy > 1;
    // Some titles render each cube face as a separate small color target and
    // resolve it to its own slice of the guest layout, leaving guest memory
    // zero. Assemble and cache a cube image from those six resolved faces,
    // re-copying only when a face resolve version advances.
    const auto cube_layout = texture_util::GetGuestTextureLayout(
        fetch.dimension, fetch.pitch, texture.width, texture.height, 1, fetch.tiled, fetch.format,
        fetch.packed_mips, true, mip_max);
    const uint32_t face_stride = cube_layout.base.array_slice_stride_bytes;
    std::array<std::shared_ptr<ColorImage>, kTextureCubeArrayLayers> faces{};
    std::array<uint32_t, kTextureCubeArrayLayers> face_addresses{};
    std::array<uint64_t, kTextureCubeArrayLayers> face_versions{};
    std::optional<bool> face_swap;
    bool faces_usable = face_stride != 0;
    for (uint32_t face = 0; faces_usable && face < kTextureCubeArrayLayers; ++face) {
      const uint32_t face_addr = GetResolvedCubeFaceAddress(address, face_stride, face);
      auto face_resolved = resolved_images_.find(face_addr);
      if (face_resolved == resolved_images_.end() ||
          face_resolved->second.image->array_layers() != 1) {
        faces_usable = false;
        break;
      }
      auto& entry = face_resolved->second;
      const auto face_extent = entry.image->extent();
      if (!CanUseResolvedCubeFace(texture.format == entry.image->format(), face_extent.width,
                                  face_extent.height, scaled_width, scaled_height, res_sx,
                                  res_sy)) {
        faces_usable = false;
        break;
      }
      const bool swap_red_blue =
          !depth_texture && reg::RB_COPY_DEST_INFO{entry.dest_info}.copy_dest_swap != 0;
      if (face_swap && *face_swap != swap_red_blue) {
        faces_usable = false;
        break;
      }
      face_swap = swap_red_blue;
      faces[face] = entry.image;
      face_addresses[face] = face_addr;
      face_versions[face] = entry.version;
      entry.last_use_submission = clock_.next;
    }
    if (faces_usable) {
      texture.width = scaled_width;
      texture.height = scaled_height;
      texture.resolution_scaled = is_scaled;
      texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count, *face_swap);
      texture.swizzle =
          MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);
      const ResolvedCubeKey cube_key{address, scaled_width, scaled_height, texture.format};
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = texture.format;
      image.extent = {scaled_width, scaled_height, 1};
      image.mipLevels = 1;
      image.arrayLayers = kTextureCubeArrayLayers;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      bool failed = false;
      auto entry = GetOrCreateCachedTexture(
          resolved_cubes_, kMaxResolvedCubes, cube_key, image,
          [&](CachedTexture& created) { created.resolved_face_addresses = face_addresses; },
          failed);
      if (failed) {
        return false;
      }
      if (entry) {
        entry->resolved_face_addresses = face_addresses;
        const VkImageView image_view = entry->GetOrCreateView(
            VK_IMAGE_VIEW_TYPE_CUBE, texture.format, texture.swizzle, kTextureCubeArrayLayers);
        if (image_view == VK_NULL_HANDLE) {
          return false;
        }
        const bool needs_upload =
            ShouldUploadResolvedCube(entry->uploaded, entry->resolved_face_versions, face_versions);
        texture.cached_texture = entry;
        texture.cached_view = image_view;
        texture.upload_cached = needs_upload;
        if (needs_upload) {
          texture.resolved_cube_faces = std::move(faces);
          texture.resolved_cube_face_versions = face_versions;
        }
        return true;
      }
      texture.resolved_cube_faces = std::move(faces);
      return true;
    }
  }
  if (!cube && resolved != resolved_images_.end() &&
      texture.format == resolved->second.image->format()) {
    const uint32_t res_sx = resolved->second.resolution_scaled ? draw_resolution_scale_x_ : 1;
    const uint32_t res_sy = resolved->second.resolution_scaled ? draw_resolution_scale_y_ : 1;
    const uint32_t scaled_width = texture.width * res_sx;
    const uint32_t scaled_height = texture.height * res_sy;
    const bool is_scaled = res_sx > 1 || res_sy > 1;
    // A 32bpp 3/4-channel resolve copied with copy_dest_swap keeps the EDRAM
    // red/blue order, while the guest fetch swizzle expects them swapped in
    // memory, so apply that swap to the host view instead.
    const bool swap_red_blue =
        !depth_texture && reg::RB_COPY_DEST_INFO{resolved->second.dest_info}.copy_dest_swap != 0;
    const auto extent = resolved->second.image->extent();
    if (extent.width == scaled_width && extent.height == scaled_height) {
      resolved->second.last_use_submission = clock_.next;
      texture.width = scaled_width;
      texture.height = scaled_height;
      texture.resolution_scaled = is_scaled;
      texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count, swap_red_blue);
      texture.swizzle =
          MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);
      texture.resolved_image = resolved->second.image;
      return true;
    }
    // Resolves are 8x8-aligned, so the cached image can be up to one alignment
    // unit larger than the texture. Crop its top-left corner during the upload;
    // otherwise fall through to guest memory so a newly written CPU texture at
    // the same address with a different size is not dropped.
    const uint32_t alignment = resolved->second.tiled_destination ? xenos::kTextureTileWidthHeight
                                                                  : xenos::kResolveAlignmentPixels;
    if (CanCropAlignedResolve(extent.width, extent.height, scaled_width, scaled_height, res_sx,
                              res_sy, alignment)) {
      resolved->second.last_use_submission = clock_.next;
      texture.width = scaled_width;
      texture.height = scaled_height;
      texture.resolution_scaled = is_scaled;
      texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count, swap_red_blue);
      texture.swizzle =
          MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);
      const ResolvedCropKey crop_key{address, scaled_width, scaled_height, texture.format};
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = texture.format;
      image.extent = {scaled_width, scaled_height, 1};
      image.mipLevels = 1;
      image.arrayLayers = 1;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      bool failed = false;
      auto entry = GetOrCreateCachedTexture(
          resolved_crops_, kMaxResolvedCrops, crop_key, image, [](CachedTexture&) {}, failed);
      if (failed) {
        return false;
      }
      if (entry) {
        const VkImageView image_view =
            entry->GetOrCreateView(VK_IMAGE_VIEW_TYPE_2D_ARRAY, texture.format, texture.swizzle, 1);
        if (image_view == VK_NULL_HANDLE) {
          return false;
        }
        const bool needs_upload = ShouldUploadResolvedCrop(
            entry->uploaded, entry->resolved_crop_version, resolved->second.version);
        texture.cached_texture = entry;
        texture.cached_view = image_view;
        texture.upload_cached = needs_upload;
        if (needs_upload) {
          texture.crop_source = resolved->second.image;
          texture.resolved_crop_version = resolved->second.version;
        }
        return true;
      }
      texture.crop_source = resolved->second.image;
      return true;
    }
  }
  texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count,
                                                fetch.format == xenos::TextureFormat::k_5_6_5);
  texture.swizzle =
      MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);

  GuestTextureKey key;
  key.base_page = base_page;
  key.mip_page = mip_page;
  key.width = texture.width;
  key.height = texture.height;
  key.array_layers = texture.array_layers;
  key.format = fetch.format;
  key.dimension = fetch.dimension;
  key.tiled = fetch.tiled != 0;
  key.packed_mips = fetch.packed_mips != 0;
  key.endianness = fetch.endianness;
  key.pitch = fetch.pitch;
  key.mip_max = mip_max;
  key.depth = texture.depth;
  key.is_signed = sign_unpacked;
  key.signed_view = texture.mutable_format;

  // A cached texture whose texels are already uploaded can be sampled straight
  // away: the guest layout and the physical page checks only matter when the
  // image is created or decoded for the first time.
  const uint32_t view_layers = is_3d ? 1 : texture.array_layers;
  std::shared_ptr<CachedTexture> entry;
  auto cached = cached_textures_.find(key);
  if (cached != cached_textures_.end()) {
    entry = cached->second.texture;
    cached->second.last_use_submission = clock_.next;
    entry->last_use_submission = clock_.next;
    if (entry->uploaded) {
      const VkImageView image_view = entry->GetOrCreateView(texture.view_type, texture.format,
                                                            texture.swizzle, view_layers);
      if (image_view == VK_NULL_HANDLE) {
        return false;
      }
      texture.level_count = entry->level_count;
      texture.cached_texture = entry;
      texture.cached_view = image_view;
      texture.upload_cached = false;
      return true;
    }
  }

  const bool has_base = base_page != 0;
  const auto layout = texture_util::GetGuestTextureLayout(
      fetch.dimension, fetch.pitch, texture.width, texture.height,
      is_3d ? texture.depth : texture.array_layers, fetch.tiled, fetch.format, fetch.packed_mips,
      has_base, mip_max);

  // Without a base level (mip_min_level > 0) level 0 exists in the image but is
  // never sampled: the sampler's minLod starts at the first stored mip.
  const uint32_t level_count = layout.max_level + 1;
  const uint32_t first_level = has_base ? 0 : 1;
  if (first_level > layout.max_level) {
    return false;
  }
  texture.level_count = level_count;
  texture.first_level = first_level;
  const uint32_t base_length =
      has_base ? uint32_t(rex::align(layout.base.level_data_extent_bytes, uint32_t(4))) : 0;
  if (has_base && !IsPhysicalRange(address, base_length)) {
    return false;
  }
  uint32_t mip_start = 0;
  uint32_t mip_length = 0;
  if (layout.max_level > 0) {
    mip_start = mip_page << 12;
    mip_length = uint32_t(rex::align(layout.mips_total_extent_bytes, uint32_t(4)));
    if (!IsPhysicalRange(mip_start, mip_length)) {
      return false;
    }
  }

  if (!entry) {
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    image.format = texture.format;
    image.extent = {texture.width, texture.height, texture.depth};
    image.mipLevels = level_count;
    image.arrayLayers = view_layers;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (cube) {
      image.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (texture.mutable_format) {
      // Signed components sample the same texels through an SNORM view.
      image.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }
    bool failed = false;
    entry = GetOrCreateCachedTexture(
        cached_textures_, kMaxCachedTextures, key, image,
        [&](CachedTexture& created) {
          created.base_start = address;
          created.base_length = base_length;
          created.mip_start = mip_start;
          created.mip_length = mip_length;
          if (!cached_only_pages_.empty()) {
            if (base_length) {
              const auto [base_first, base_count] = GetPhysicalPageRange(address, base_length);
              AtomicSetPageRangeBits(cached_only_pages_, base_first, base_count);
            }
            if (mip_length) {
              const auto [mip_first, mip_count] = GetPhysicalPageRange(mip_start, mip_length);
              AtomicSetPageRangeBits(cached_only_pages_, mip_first, mip_count);
            }
          }
          // Watch the guest pages before reading them, so a write between the read and
          // the next frame invalidates this entry.
          if (watch_range_) {
            if (base_length) {
              watch_range_(address, base_length);
            }
            if (mip_length) {
              watch_range_(mip_start, mip_length);
            }
          } else {
            if (base_length) {
              memory_.EnablePhysicalMemoryAccessCallbacks(address, base_length, true, false);
            }
            if (mip_length) {
              memory_.EnablePhysicalMemoryAccessCallbacks(mip_start, mip_length, true, false);
            }
          }
        },
        failed);
    if (failed) {
      return false;
    }
  }

  const bool upload = !entry || !entry->uploaded;
  if (upload && !DecodeTextureLevels(fetch, info, layout, address, mip_page, signed_variant,
                                     texture, uploads, submission)) {
    return false;
  }

  if (!entry) {
    return true;
  }
  const VkImageView image_view =
      entry->GetOrCreateView(texture.view_type, texture.format, texture.swizzle, view_layers);
  if (image_view == VK_NULL_HANDLE) {
    return false;
  }
  texture.cached_texture = entry;
  texture.cached_view = image_view;
  texture.upload_cached = upload;
  return true;
}

namespace {

template <uint32_t BlockSize, size_t Swap>
void DecodeTiled2DCopyLevel(const uint8_t* __restrict source, size_t source_size,
                            uint8_t* __restrict dst, uint32_t slices, uint32_t blocks_x,
                            uint32_t blocks_y, uint32_t start_x, uint32_t start_y,
                            uint32_t pitch_aligned, uint32_t bpb_log2,
                            size_t slice_stride_bytes, size_t dst_slice_bytes) {
  for (uint32_t slice = 0; slice < slices; ++slice) {
    const size_t slice_base = size_t(slice) * slice_stride_bytes;
    uint8_t* slice_dst = dst + size_t(slice) * dst_slice_bytes;
    for (uint32_t y = 0; y < blocks_y; ++y) {
      const auto row = MakeTiled2DRowState(int32_t(y + start_y), pitch_aligned, bpb_log2);
      uint8_t* row_dst = slice_dst + size_t(y) * blocks_x * BlockSize;
      for (uint32_t x = 0; x < blocks_x; ++x) {
        const size_t offset =
            slice_base +
            size_t(GetTiledOffset2DFromRow(int32_t(x + start_x), row, bpb_log2));
        if (offset <= source_size && BlockSize <= source_size - offset) {
          CopySwapBlock<BlockSize, Swap>(source, offset, row_dst + size_t(x) * BlockSize);
        }
      }
    }
  }
}

template <uint32_t BlockSize, size_t Swap>
void DecodeLinear2DCopyLevel(const uint8_t* __restrict source, uint8_t* __restrict dst,
                             uint32_t slices, uint32_t blocks_x, uint32_t blocks_y,
                             uint32_t start_x, uint32_t start_y, uint32_t row_pitch_bytes,
                             size_t slice_stride_bytes, size_t dst_slice_bytes) {
  const size_t row_copy_bytes = size_t(blocks_x) * BlockSize;
  for (uint32_t slice = 0; slice < slices; ++slice) {
    const size_t slice_base = size_t(slice) * slice_stride_bytes;
    uint8_t* slice_dst = dst + size_t(slice) * dst_slice_bytes;
    for (uint32_t y = 0; y < blocks_y; ++y) {
      const size_t row_src_offset =
          slice_base + size_t(y + start_y) * row_pitch_bytes + size_t(start_x) * BlockSize;
      uint8_t* row_dst = slice_dst + size_t(y) * row_copy_bytes;
      if constexpr (Swap == 0) {
        std::memcpy(row_dst, source + row_src_offset, row_copy_bytes);
      } else {
        for (uint32_t x = 0; x < blocks_x; ++x) {
          CopySwapBlock<BlockSize, Swap>(source, row_src_offset + size_t(x) * BlockSize,
                                         row_dst + size_t(x) * BlockSize);
        }
      }
    }
  }
}

template <uint32_t BlockSize>
void Dispatch2DCopyBySwap(size_t swap, bool tiled, const uint8_t* source, size_t source_size,
                          uint8_t* dst, uint32_t slices, uint32_t blocks_x, uint32_t blocks_y,
                          uint32_t start_x, uint32_t start_y, uint32_t row_pitch_bytes,
                          uint32_t pitch_aligned, uint32_t bpb_log2, size_t slice_stride_bytes,
                          size_t dst_slice_bytes) {
  if (tiled) {
    switch (swap) {
      case 0:
        DecodeTiled2DCopyLevel<BlockSize, 0>(source, source_size, dst, slices, blocks_x, blocks_y,
                                             start_x, start_y, pitch_aligned, bpb_log2,
                                             slice_stride_bytes, dst_slice_bytes);
        return;
      case 1:
        DecodeTiled2DCopyLevel<BlockSize, 1>(source, source_size, dst, slices, blocks_x, blocks_y,
                                             start_x, start_y, pitch_aligned, bpb_log2,
                                             slice_stride_bytes, dst_slice_bytes);
        return;
      case 2:
        DecodeTiled2DCopyLevel<BlockSize, 2>(source, source_size, dst, slices, blocks_x, blocks_y,
                                             start_x, start_y, pitch_aligned, bpb_log2,
                                             slice_stride_bytes, dst_slice_bytes);
        return;
      default:
        DecodeTiled2DCopyLevel<BlockSize, 3>(source, source_size, dst, slices, blocks_x, blocks_y,
                                             start_x, start_y, pitch_aligned, bpb_log2,
                                             slice_stride_bytes, dst_slice_bytes);
        return;
    }
  } else {
    switch (swap) {
      case 0:
        DecodeLinear2DCopyLevel<BlockSize, 0>(source, dst, slices, blocks_x, blocks_y, start_x,
                                              start_y, row_pitch_bytes, slice_stride_bytes,
                                              dst_slice_bytes);
        return;
      case 1:
        DecodeLinear2DCopyLevel<BlockSize, 1>(source, dst, slices, blocks_x, blocks_y, start_x,
                                              start_y, row_pitch_bytes, slice_stride_bytes,
                                              dst_slice_bytes);
        return;
      case 2:
        DecodeLinear2DCopyLevel<BlockSize, 2>(source, dst, slices, blocks_x, blocks_y, start_x,
                                              start_y, row_pitch_bytes, slice_stride_bytes,
                                              dst_slice_bytes);
        return;
      default:
        DecodeLinear2DCopyLevel<BlockSize, 3>(source, dst, slices, blocks_x, blocks_y, start_x,
                                              start_y, row_pitch_bytes, slice_stride_bytes,
                                              dst_slice_bytes);
        return;
    }
  }
}

}  // namespace

bool NativeTextureCache::DecodeTextureLevels(const xenos::xe_gpu_texture_fetch_t& fetch,
                                             const TextureFormatInfo& info,
                                             const texture_util::TextureGuestLayout& layout,
                                             uint32_t address, uint32_t mip_page,
                                             bool signed_variant, DrawTexture& texture,
                                             ui::vulkan::VulkanUploadBufferPool* uploads,
                                             uint64_t submission) {
  constexpr std::array<size_t, 4> byte_xor{0, 1, 3, 2};
  const size_t swap = byte_xor[uint32_t(fetch.endianness)];
  const bool is_3d = fetch.dimension == xenos::DataDimension::k3D;
  const uint32_t block_width = info.block_width;
  const uint32_t block_size = info.block_size;
  const uint32_t host_size = info.host_block_size();
  const uint32_t bpb_log2 = uint32_t(std::countr_zero(block_size));

  // Precompute level geometry and total upload size so we can decode directly
  // into a single frame-upload-pool slice when it fits in one page.
  struct LevelPlan {
    uint32_t level;
    uint32_t level_storage;
    std::span<const uint8_t> source;
    uint32_t start_x = 0;
    uint32_t start_y = 0;
    uint32_t start_z = 0;
    uint32_t level_width = 0;
    uint32_t level_height = 0;
    uint32_t blocks_x = 0;
    uint32_t blocks_y = 0;
    uint32_t slices = 0;
    uint32_t host_x = 0;
    uint32_t host_y = 0;
    size_t slice_bytes = 0;
    size_t total_bytes = 0;
    VkDeviceSize pool_offset = 0;
  };
  std::array<LevelPlan, xenos::kTextureMaxMips + 1> plans;
  uint32_t plan_count = 0;
  VkDeviceSize total_upload_bytes = 0;
  for (uint32_t level = texture.first_level; level <= layout.max_level; ++level) {
    auto& plan = plans[plan_count++];
    plan.level = level;
    plan.level_storage = std::min(level, layout.packed_level);
    const auto& level_layout = level ? layout.mips[plan.level_storage] : layout.base;
    const uint64_t level_address =
        level ? (uint64_t(mip_page) << 12) + layout.mip_offsets_bytes[plan.level_storage] : address;
    if (level_address >= kPhysicalMemorySize) {
      return false;
    }
    const size_t level_read_size =
        GetTextureLevelReadSize(level_layout.array_slice_stride_bytes,
                                level_layout.array_slice_data_extent_bytes, texture.array_layers);
    plan.source =
        ReadGuestPhysical(memory_, uint32_t(level_address), rex::align(level_read_size, size_t(4)));
    if (plan.source.empty()) {
      return false;
    }
    if (level >= layout.packed_level) {
      texture_util::GetPackedMipOffset(texture.width, texture.height, texture.depth, fetch.format,
                                       level, plan.start_x, plan.start_y, plan.start_z);
    }
    plan.level_width = std::max(1u, texture.width >> level);
    plan.level_height = std::max(1u, texture.height >> level);
    const uint32_t level_depth = std::max(1u, texture.depth >> level);
    plan.blocks_x = (plan.level_width + block_width - 1) / block_width;
    plan.blocks_y = (plan.level_height + block_width - 1) / block_width;
    plan.slices = is_3d ? level_depth : texture.array_layers;
    plan.host_x = info.decompresses_blocks() ? plan.level_width : plan.blocks_x;
    plan.host_y = info.decompresses_blocks() ? plan.level_height : plan.blocks_y;
    plan.slice_bytes = size_t(plan.host_x) * plan.host_y * host_size;
    plan.total_bytes = plan.slice_bytes * plan.slices;
    total_upload_bytes = rex::align(total_upload_bytes, VkDeviceSize(16));
    plan.pool_offset = total_upload_bytes;
    total_upload_bytes += plan.total_bytes;
  }

  uint8_t* direct_base = nullptr;
  VkBuffer direct_buffer = VK_NULL_HANDLE;
  VkDeviceSize direct_offset = 0;
  if (uploads && total_upload_bytes > 0 && total_upload_bytes <= kDrawUploadPageSize) {
    direct_base = uploads->Request(submission, total_upload_bytes, 16, direct_buffer, direct_offset);
    if (direct_base) {
      texture.direct_upload_buffer = direct_buffer;
      texture.direct_uploads.resize(plan_count);
    }
  }

  for (uint32_t p = 0; p < plan_count; ++p) {
    const auto& plan = plans[p];
    const auto& level_layout =
        plan.level ? layout.mips[plan.level_storage] : layout.base;
    uint8_t* dst = nullptr;
    if (direct_base) {
      dst = direct_base + plan.pool_offset;
      auto& copy = texture.direct_uploads[p];
      copy = {};
      copy.bufferOffset = direct_offset + plan.pool_offset;
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, plan.level, 0,
                               is_3d ? 1 : texture.array_layers};
      copy.imageExtent = {plan.level_width, plan.level_height,
                          is_3d ? std::max(1u, texture.depth >> plan.level) : 1};
    } else {
      dst = texture.levels.emplace_back(plan.total_bytes).data();
    }

    if (!is_3d && info.unpack == TextureUnpack::kCopy &&
        (block_size == 1 || block_size == 2 || block_size == 4 || block_size == 8 ||
         block_size == 16)) {
      const uint32_t pitch_blocks = level_layout.row_pitch_bytes / block_size;
      const uint32_t pitch_aligned =
          (pitch_blocks + (xenos::kTextureTileWidthHeight - 1)) &
          ~(xenos::kTextureTileWidthHeight - 1);
      // Verify bounds once for untiled; tiled checks per-block inside helper.
      if (!fetch.tiled) {
        const size_t needed =
            size_t(plan.slices - 1) * level_layout.array_slice_stride_bytes +
            size_t(plan.start_y + plan.blocks_y - 1) * level_layout.row_pitch_bytes +
            size_t(plan.start_x + plan.blocks_x) * block_size;
        if (needed > plan.source.size()) {
          return false;
        }
      }
      switch (block_size) {
        case 1:
          Dispatch2DCopyBySwap<1>(swap, fetch.tiled != 0, plan.source.data(), plan.source.size(),
                                  dst, plan.slices, plan.blocks_x, plan.blocks_y, plan.start_x,
                                  plan.start_y, level_layout.row_pitch_bytes, pitch_aligned,
                                  bpb_log2, level_layout.array_slice_stride_bytes,
                                  plan.slice_bytes);
          break;
        case 2:
          Dispatch2DCopyBySwap<2>(swap, fetch.tiled != 0, plan.source.data(), plan.source.size(),
                                  dst, plan.slices, plan.blocks_x, plan.blocks_y, plan.start_x,
                                  plan.start_y, level_layout.row_pitch_bytes, pitch_aligned,
                                  bpb_log2, level_layout.array_slice_stride_bytes,
                                  plan.slice_bytes);
          break;
        case 4:
          Dispatch2DCopyBySwap<4>(swap, fetch.tiled != 0, plan.source.data(), plan.source.size(),
                                  dst, plan.slices, plan.blocks_x, plan.blocks_y, plan.start_x,
                                  plan.start_y, level_layout.row_pitch_bytes, pitch_aligned,
                                  bpb_log2, level_layout.array_slice_stride_bytes,
                                  plan.slice_bytes);
          break;
        case 8:
          Dispatch2DCopyBySwap<8>(swap, fetch.tiled != 0, plan.source.data(), plan.source.size(),
                                  dst, plan.slices, plan.blocks_x, plan.blocks_y, plan.start_x,
                                  plan.start_y, level_layout.row_pitch_bytes, pitch_aligned,
                                  bpb_log2, level_layout.array_slice_stride_bytes,
                                  plan.slice_bytes);
          break;
        case 16:
          Dispatch2DCopyBySwap<16>(swap, fetch.tiled != 0, plan.source.data(), plan.source.size(),
                                   dst, plan.slices, plan.blocks_x, plan.blocks_y, plan.start_x,
                                   plan.start_y, level_layout.row_pitch_bytes, pitch_aligned,
                                   bpb_log2, level_layout.array_slice_stride_bytes,
                                   plan.slice_bytes);
          break;
      }
      continue;
    }

    const uint32_t z_stride_bytes =
        level_layout.row_pitch_bytes * level_layout.z_slice_stride_block_rows;
    for (uint32_t slice = 0; slice < plan.slices; ++slice) {
      const size_t slice_base =
          is_3d ? 0 : size_t(slice) * level_layout.array_slice_stride_bytes;
      const uint32_t z = is_3d ? slice + plan.start_z : 0;
      uint8_t* const slice_data = dst + size_t(slice) * plan.slice_bytes;
      for (uint32_t y = 0; y < plan.blocks_y; ++y) {
        for (uint32_t x = 0; x < plan.blocks_x; ++x) {
          const int32_t gx = int32_t(x + plan.start_x);
          const int32_t gy = int32_t(y + plan.start_y);
          size_t offset = slice_base;
          if (fetch.tiled) {
            const uint32_t pitch_blocks = level_layout.row_pitch_bytes / block_size;
            offset += is_3d ? size_t(texture_util::GetTiledOffset3D(
                                  gx, gy, int32_t(z), pitch_blocks,
                                  level_layout.z_slice_stride_block_rows, bpb_log2))
                            : size_t(GetTiledOffset2DInline(gx, gy, pitch_blocks, bpb_log2));
          } else {
            offset += size_t(z) * z_stride_bytes + size_t(gy) * level_layout.row_pitch_bytes +
                      size_t(gx) * block_size;
          }
          if (offset > plan.source.size() || block_size > plan.source.size() - offset) {
            return false;
          }
          std::array<uint8_t, 16> block;
          for (uint32_t i = 0; i < block_size; ++i) {
            block[i] = plan.source[(offset + i) ^ swap];
          }
          if (info.unpack == TextureUnpack::kDxt3aToR8) {
            uint64_t bits;
            std::memcpy(&bits, block.data(), sizeof(bits));
            const auto alpha = DecodeDxt3aBlock(bits);
            for (uint32_t texel = 0; texel < 16; ++texel) {
              const uint32_t tx = x * 4 + texel % 4;
              const uint32_t ty = y * 4 + texel / 4;
              if (tx < plan.level_width && ty < plan.level_height) {
                slice_data[size_t(ty) * plan.host_x + tx] = alpha[texel];
              }
            }
            continue;
          }
          if (info.unpack == TextureUnpack::kCtx1ToRg8) {
            uint64_t bits;
            std::memcpy(&bits, block.data(), sizeof(bits));
            const auto rg = DecodeCtx1Block(bits);
            for (uint32_t texel = 0; texel < 16; ++texel) {
              const uint32_t tx = x * 4 + texel % 4;
              const uint32_t ty = y * 4 + texel / 4;
              if (tx < plan.level_width && ty < plan.level_height) {
                std::memcpy(slice_data + (size_t(ty) * plan.host_x + tx) * 2, &rg[texel], 2);
              }
            }
            continue;
          }
          UnpackTexel(info, signed_variant, block.data(),
                      slice_data + (size_t(y) * plan.host_x + x) * host_size);
        }
      }
    }
  }
  return true;
}

}  // namespace rex::graphics::native
