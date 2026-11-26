#include "texture_cache.h"

#include <algorithm>
#include <bit>
#include <cstring>

#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {
namespace {

VkFilter ToFilter(xenos::TextureFilter filter) {
  return filter == xenos::TextureFilter::kLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerAddressMode ToAddressMode(xenos::ClampMode mode) {
  switch (mode) {
    case xenos::ClampMode::kRepeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case xenos::ClampMode::kMirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
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
    cached_texture_pages_.assign((kSharedVertexPageCount + 63) / 64, 0);
  }
  return InitializeNullTextures();
}

void NativeTextureCache::InvalidateAll() {
  cached_textures_.clear();
  resolved_images_.clear();
  resolved_cubes_.clear();
  resolved_crops_.clear();
  presented_resolve_address_.reset();
  if (!cached_texture_pages_.empty()) {
    std::lock_guard<std::mutex> lock(pages_mutex_);
    std::fill(cached_texture_pages_.begin(), cached_texture_pages_.end(), 0);
  }
}

void NativeTextureCache::ClearSamplers() {
  for (const auto& entry : samplers_) {
    device_.functions().vkDestroySampler(device_.device(), entry.second.sampler, nullptr);
  }
  samplers_.clear();
}

void NativeTextureCache::MarkCachedPages(uint32_t start, uint32_t length) {
  if (cached_texture_pages_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(pages_mutex_);
  const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
  SetPageRangeBits(cached_texture_pages_, first_page, page_count);
}

std::pair<uint32_t, uint32_t> NativeTextureCache::WidenInvalidation(uint32_t start,
                                                                    uint32_t length) {
  std::lock_guard<std::mutex> lock(pages_mutex_);
  if (cached_texture_pages_.empty()) {
    return {start, 0};
  }
  return WidenInvalidationPageRange(cached_texture_pages_, start, length);
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
  return vk.vkCreateImageView(device, &view, nullptr, &null_texture_cube_view_) == VK_SUCCESS;
}

void NativeTextureCache::DestroyNullTextures() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  vk.vkDestroyImageView(device, null_texture_cube_view_, nullptr);
  vk.vkDestroyImageView(device, null_texture_2d_view_, nullptr);
  vk.vkDestroyImage(device, null_texture_image_, nullptr);
  vk.vkFreeMemory(device, null_texture_memory_, nullptr);
  null_texture_cube_view_ = null_texture_2d_view_ = VK_NULL_HANDLE;
  null_texture_image_ = VK_NULL_HANDLE;
  null_texture_memory_ = VK_NULL_HANDLE;
  null_textures_initialized_ = false;
}

void NativeTextureCache::RecordNullTextureInitialization(VkCommandBuffer commands) {
  if (null_textures_initialized_ || null_texture_image_ == VK_NULL_HANDLE) {
    return;
  }
  const auto& vk = device_.functions();
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = null_texture_image_;
  barrier.subresourceRange = range;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  const VkClearColorValue zero{};
  vk.vkCmdClearColorImage(commands, null_texture_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          &zero, 1, &range);
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vk.vkCmdPipelineBarrier(
      commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
      0, nullptr, 1, &barrier);
  null_textures_initialized_ = true;
}

bool NativeTextureCache::ReadTextures(const SpirvShader& shader, DrawInput& input) {
  const auto& textures = shader.GetTextureBindingsAfterTranslation();
  const auto& samplers = shader.GetSamplerBindingsAfterTranslation();
  const auto& properties = device_.properties();
  if (textures.size() > properties.maxPerStageDescriptorSampledImages ||
      samplers.size() > properties.maxPerStageDescriptorSamplers) {
    return false;
  }

  std::array<int, 32> indices;
  indices.fill(-1);
  for (const auto& binding : textures) {
    const bool cube = binding.dimension == xenos::FetchOpDimension::kCube;
    if (!cube && binding.dimension != xenos::FetchOpDimension::k2D) {
      return false;
    }
    const uint32_t slot = binding.fetch_constant;
    const auto fetch = regs_.GetTextureFetch(slot);
    // A fetch constant that isn't a usable texture of the fetched dimension
    // reads as zero on the guest - bind a zero-cleared fallback rather than
    // dropping the draw. Its signs stay unsigned.
    const bool use_fallback =
        ShouldUseFallbackTexture(fetch.type == xenos::FetchConstantType::kTexture,
                                 (fetch.base_address & 0x1FFFF) != 0, fetch.stacked,
                                 fetch.dimension == (cube ? xenos::DataDimension::kCube
                                                          : xenos::DataDimension::k2DOrStacked));
    if (indices[slot] < 0) {
      auto& texture = input.textures.emplace_back();
      if (use_fallback) {
        texture.fallback_view = cube ? null_texture_cube_view_ : null_texture_2d_view_;
      } else if (!ReadTexture(fetch, texture)) {
        REXGPU_DEBUG(
            "Native Vulkan texture {} unsupported: format={} base={:05X} flags={:08X}/{:08X}", slot,
            uint32_t(fetch.format), uint32_t(fetch.base_address), fetch.dword_4, fetch.dword_5);
        return false;
      }
      if (!use_fallback) {
        input.system.texture_swizzled_signs[slot / 4] |= uint32_t(texture_util::SwizzleSigns(fetch))
                                                         << ((slot % 4) * 8);
        if (texture.resolution_scaled) {
          input.system.textures_resolution_scaled |= uint32_t(1) << slot;
        }
        if (!properties.imageViewFormatSwizzle) {
          PackSystemTextureSwizzle(input.system.texture_swizzles, slot, texture.host_swizzle);
        }
      }
      indices[slot] = int(input.textures.size() - 1);
    }
    input.texture_bindings.push_back(uint32_t(indices[slot]));
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
    out.max_lod = mip_filter == xenos::TextureFilter::kBaseMap ? 0.0f : float(fetch.mip_max_level);
    out.address_u = ToAddressMode(fetch.clamp_x);
    out.address_v = ToAddressMode(fetch.clamp_y);
    const VkSampler handle = GetSampler(out);
    if (handle == VK_NULL_HANDLE) {
      return false;
    }
    input.samplers.push_back(handle);
  }
  return true;
}

VkSampler NativeTextureCache::GetSampler(const DrawSampler& key) {
  auto existing = samplers_.find(key);
  if (existing != samplers_.end()) {
    existing->second.last_use_submission = clock_.next;
    return existing->second.sampler;
  }
  if (samplers_.size() >= kMaxSamplers) {
    const auto victim =
        SelectRetiredEviction(samplers_, clock_.retired + 1,
                              [](const DrawSampler&, const CachedSampler&) { return true; });
    if (!victim) {
      return VK_NULL_HANDLE;
    }
    auto victim_it = samplers_.find(*victim);
    device_.functions().vkDestroySampler(device_.device(), victim_it->second.sampler, nullptr);
    samplers_.erase(victim_it);
  }
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  info.minFilter = key.min_filter;
  info.magFilter = key.mag_filter;
  info.mipmapMode = key.mip_mode;
  info.maxLod = key.max_lod;
  info.addressModeU = key.address_u;
  info.addressModeV = key.address_v;
  info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSampler sampler = VK_NULL_HANDLE;
  if (device_.functions().vkCreateSampler(device_.device(), &info, nullptr, &sampler) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  samplers_.emplace(key, CachedSampler{sampler, clock_.next});
  return sampler;
}

void NativeTextureCache::InvalidateTextures(uint32_t start, uint32_t length) {
  bool erased = false;
  for (auto it = cached_textures_.begin(); it != cached_textures_.end();) {
    const CachedTexture& texture = *it->second.texture;
    if (RangesOverlap(start, length, texture.base_start, texture.base_length) ||
        RangesOverlap(start, length, texture.mip_start, texture.mip_length)) {
      it = cached_textures_.erase(it);
      erased = true;
    } else {
      ++it;
    }
  }
  if (erased) {
    RebuildCachedTexturePages();
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

void NativeTextureCache::InvalidateResolvedTextures(uint32_t start, uint32_t length) {
  bool erased = false;
  for (auto it = resolved_images_.begin(); it != resolved_images_.end();) {
    if (RangesOverlap(start, length, it->second.guest_start, it->second.guest_length)) {
      if (presented_resolve_address_ && *presented_resolve_address_ == it->first) {
        presented_resolve_address_.reset();
      }
      it = resolved_images_.erase(it);
      erased = true;
    } else {
      ++it;
    }
  }
  if (erased) {
    PruneResolvedDependents();
    RebuildCachedTexturePages();
  }
}

void NativeTextureCache::RebuildCachedTexturePages() {
  if (cached_texture_pages_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(pages_mutex_);
  std::fill(cached_texture_pages_.begin(), cached_texture_pages_.end(), 0);
  for (const auto& [key, entry] : cached_textures_) {
    const auto& texture = *entry.texture;
    const auto [base_first, base_count] =
        GetPhysicalPageRange(texture.base_start, texture.base_length);
    SetPageRangeBits(cached_texture_pages_, base_first, base_count);
    if (texture.mip_length) {
      const auto [mip_first, mip_count] =
          GetPhysicalPageRange(texture.mip_start, texture.mip_length);
      SetPageRangeBits(cached_texture_pages_, mip_first, mip_count);
    }
  }
  for (const auto& [addr, entry] : resolved_images_) {
    const auto [first, count] = GetPhysicalPageRange(entry.guest_start, entry.guest_length);
    SetPageRangeBits(cached_texture_pages_, first, count);
  }
}

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
      cache.erase(*victim);
    }
  }
  if (cache.size() >= max_entries) {
    return {};
  }
  auto entry = std::make_shared<CachedTexture>();
  entry->device = &device_;
  entry->level_count = image_info.mipLevels;
  entry->last_use_submission = clock_.next;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          &device_, image_info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, entry->image,
          entry->memory)) {
    failed = true;
    return {};
  }
  on_created(*entry);
  cache.emplace(key, CachedTextureEntry{entry, clock_.next});
  return entry;
}

bool NativeTextureCache::ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch,
                                     DrawTexture& texture) {
  const bool cube = fetch.dimension == xenos::DataDimension::kCube;
  if (fetch.type != xenos::FetchConstantType::kTexture ||
      (!cube && fetch.dimension != xenos::DataDimension::k2DOrStacked) || fetch.stacked) {
    return false;
  }
  // IEEE float host formats are natively signed, so a signed guest fetch maps
  // straight onto them. Integer, UNORM and BC formats would need separate SNORM
  // views, so those signed fetches are still rejected until they exist.
  if (!IsFloatTextureFormat(fetch.format) &&
      (fetch.sign_x == xenos::TextureSign::kSigned || fetch.sign_y == xenos::TextureSign::kSigned ||
       fetch.sign_z == xenos::TextureSign::kSigned ||
       fetch.sign_w == xenos::TextureSign::kSigned)) {
    return false;
  }

  texture.width = fetch.size_2d.width + 1;
  texture.height = fetch.size_2d.height + 1;
  if (cube && texture.width != texture.height) {
    return false;
  }
  texture.array_layers = GetTextureArrayLayers(fetch.dimension);
  texture.view_type = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  uint32_t block_size;
  uint32_t block_width = 1;
  bool depth_texture = false;
  switch (fetch.format) {
    case xenos::TextureFormat::k_8:
    case xenos::TextureFormat::k_8_A:
    case xenos::TextureFormat::k_8_B:
      texture.format = VK_FORMAT_R8_UNORM;
      block_size = 1;
      break;
    case xenos::TextureFormat::k_8_8:
      texture.format = VK_FORMAT_R8G8_UNORM;
      block_size = 2;
      break;
    case xenos::TextureFormat::k_5_6_5:
      texture.format = VK_FORMAT_R5G6B5_UNORM_PACK16;
      block_size = 2;
      break;
    case xenos::TextureFormat::k_32_FLOAT:
      texture.format = VK_FORMAT_R32_SFLOAT;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_16_16_FLOAT:
      texture.format = VK_FORMAT_R16G16_SFLOAT;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_16_16_16_16_FLOAT:
      texture.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      block_size = 8;
      break;
    case xenos::TextureFormat::k_24_8:
    case xenos::TextureFormat::k_24_8_FLOAT:
      texture.format = VK_FORMAT_R32_SFLOAT;
      block_size = 4;
      depth_texture = true;
      break;
    case xenos::TextureFormat::k_2_10_10_10:
    case xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16:
      texture.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16:
    case xenos::TextureFormat::k_8_8_8_8_A:
    case xenos::TextureFormat::k_8_8_8_8:
      texture.format = VK_FORMAT_R8G8B8A8_UNORM;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT1:
      texture.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
      block_width = 4;
      block_size = 8;
      break;
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT2_3:
      texture.format = VK_FORMAT_BC2_UNORM_BLOCK;
      block_width = 4;
      block_size = 16;
      break;
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT4_5:
      texture.format = VK_FORMAT_BC3_UNORM_BLOCK;
      block_width = 4;
      block_size = 16;
      break;
    case xenos::TextureFormat::k_DXN:
      texture.format = VK_FORMAT_BC5_UNORM_BLOCK;
      block_width = 4;
      block_size = 16;
      break;
    default:
      return false;
  }

  // Replicate the guest components that the format actually stores, matching
  // the Xenos Vulkan host format swizzle: RRRR for one channel, RGGG for two,
  // RGBA otherwise. Block sizes don't map to channels, so use the count.
  const uint32_t channel_count = TextureChannelCount(fetch.format);
  uint32_t base_page, mip_page, mip_min, mip_max;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, &base_page,
                                                 &mip_page, &mip_min, &mip_max);
  if (!base_page || mip_min != 0) {
    return false;
  }
  const uint32_t address = base_page << 12;
  const uint32_t scaled_width = texture.width * draw_resolution_scale_x_;
  const uint32_t scaled_height = texture.height * draw_resolution_scale_y_;
  const bool is_scaled = draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1;
  const bool use_view_swizzle = device_.properties().imageViewFormatSwizzle;
  auto resolved = resolved_images_.find(address);
  if (cube && !mip_max && resolved != resolved_images_.end()) {
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
      if (face_resolved == resolved_images_.end()) {
        faces_usable = false;
        break;
      }
      auto& entry = face_resolved->second;
      const auto face_extent = entry.image->extent();
      if (!CanUseResolvedCubeFace(texture.format == entry.image->format(), face_extent.width,
                                  face_extent.height, scaled_width, scaled_height,
                                  draw_resolution_scale_x_, draw_resolution_scale_y_)) {
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
  if (!cube && !mip_max && resolved != resolved_images_.end() &&
      texture.format == resolved->second.image->format()) {
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
    if (CanCropAlignedResolve(extent.width, extent.height, scaled_width, scaled_height,
                              draw_resolution_scale_x_, draw_resolution_scale_y_, alignment)) {
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

  // A cached texture whose texels are already uploaded can be sampled straight
  // away: the guest layout and the physical page checks only matter when the
  // image is created or decoded for the first time.
  std::shared_ptr<CachedTexture> entry;
  auto cached = cached_textures_.find(key);
  if (cached != cached_textures_.end()) {
    entry = cached->second.texture;
    cached->second.last_use_submission = clock_.next;
    entry->last_use_submission = clock_.next;
    if (entry->uploaded) {
      const VkImageView image_view = entry->GetOrCreateView(texture.view_type, texture.format,
                                                            texture.swizzle, texture.array_layers);
      if (image_view == VK_NULL_HANDLE) {
        return false;
      }
      texture.cached_texture = entry;
      texture.cached_view = image_view;
      texture.upload_cached = false;
      return true;
    }
  }

  const auto layout = texture_util::GetGuestTextureLayout(
      fetch.dimension, fetch.pitch, texture.width, texture.height, 1, fetch.tiled, fetch.format,
      fetch.packed_mips, true, mip_max);

  const uint32_t level_count = layout.max_level + 1;
  const uint32_t base_length =
      uint32_t(rex::align(layout.base.level_data_extent_bytes, uint32_t(4)));
  if (!IsPhysicalRange(address, base_length)) {
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
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = texture.format;
    image.extent = {texture.width, texture.height, 1};
    image.mipLevels = level_count;
    image.arrayLayers = texture.array_layers;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (texture.view_type == VK_IMAGE_VIEW_TYPE_CUBE) {
      image.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    bool failed = false;
    const size_t size_before = cached_textures_.size();
    entry = GetOrCreateCachedTexture(
        cached_textures_, kMaxCachedTextures, key, image,
        [&](CachedTexture& created) {
          created.base_start = address;
          created.base_length = base_length;
          created.mip_start = mip_start;
          created.mip_length = mip_length;
          if (!cached_texture_pages_.empty()) {
            std::lock_guard<std::mutex> lock(pages_mutex_);
            const auto [base_first, base_count] = GetPhysicalPageRange(address, base_length);
            SetPageRangeBits(cached_texture_pages_, base_first, base_count);
            if (mip_length) {
              const auto [mip_first, mip_count] = GetPhysicalPageRange(mip_start, mip_length);
              SetPageRangeBits(cached_texture_pages_, mip_first, mip_count);
            }
          }
          // Watch the guest pages before reading them, so a write between the read and
          // the next frame invalidates this entry.
          memory_.EnablePhysicalMemoryAccessCallbacks(address, base_length, true, false);
          if (mip_length) {
            memory_.EnablePhysicalMemoryAccessCallbacks(mip_start, mip_length, true, false);
          }
        },
        failed);
    if (failed) {
      return false;
    }
    if (cached_textures_.size() <= size_before && entry) {
      RebuildCachedTexturePages();
    }
  }

  const bool upload = !entry || !entry->uploaded;
  if (upload) {
    constexpr std::array<size_t, 4> byte_xor{0, 1, 3, 2};
    const size_t swap = byte_xor[uint32_t(fetch.endianness)];
    for (uint32_t level = 0; level <= layout.max_level; ++level) {
      const uint32_t level_storage = std::min(level, layout.packed_level);
      const auto& level_layout = level ? layout.mips[level_storage] : layout.base;
      const uint64_t level_address =
          level ? (uint64_t(mip_page) << 12) + layout.mip_offsets_bytes[level_storage] : address;
      if (level_address >= kPhysicalMemorySize) {
        return false;
      }
      // Array slices are stored within the level, each `array_slice_stride_bytes`
      // apart, so the last slice's data must be readable as well.
      const size_t level_read_size =
          GetTextureLevelReadSize(level_layout.array_slice_stride_bytes,
                                  level_layout.array_slice_data_extent_bytes, texture.array_layers);
      const auto source = ReadGuestPhysical(memory_, uint32_t(level_address),
                                            rex::align(level_read_size, size_t(4)));
      if (source.empty()) {
        return false;
      }
      uint32_t start_x = 0, start_y = 0, start_z = 0;
      if (level >= layout.packed_level) {
        texture_util::GetPackedMipOffset(texture.width, texture.height, 1, fetch.format, level,
                                         start_x, start_y, start_z);
      }
      const uint32_t level_width = std::max(1u, texture.width >> level);
      const uint32_t level_height = std::max(1u, texture.height >> level);
      const uint32_t blocks_x = (level_width + block_width - 1) / block_width;
      const uint32_t blocks_y = (level_height + block_width - 1) / block_width;
      auto& data = texture.levels.emplace_back(size_t(blocks_x) * blocks_y * block_size *
                                               texture.array_layers);
      for (uint32_t slice = 0; slice < texture.array_layers; ++slice) {
        const size_t slice_base = size_t(slice) * level_layout.array_slice_stride_bytes;
        // Each face is tiled independently from the origin of its own data.
        for (uint32_t y = 0; y < blocks_y; ++y) {
          for (uint32_t x = 0; x < blocks_x; ++x) {
            const size_t offset =
                slice_base + (fetch.tiled ? size_t(texture_util::GetTiledOffset2D(
                                                int32_t(x + start_x), int32_t(y + start_y),
                                                level_layout.row_pitch_bytes / block_size,
                                                std::countr_zero(block_size)))
                                          : size_t(y + start_y) * level_layout.row_pitch_bytes +
                                                size_t(x + start_x) * block_size);
            if (offset > source.size() || block_size > source.size() - offset) {
              return false;
            }
            uint8_t* dst =
                data.data() + ((size_t(slice) * blocks_y + y) * blocks_x + x) * block_size;
            for (uint32_t i = 0; i < block_size; ++i) {
              dst[i] = source[(offset + i) ^ swap];
            }
            if (depth_texture) {
              uint32_t packed;
              std::memcpy(&packed, dst, sizeof(packed));
              const float depth = fetch.format == xenos::TextureFormat::k_24_8_FLOAT
                                      ? xenos::Float20e4To32(packed >> 8)
                                      : float(packed >> 8) / 16777215.0f;
              std::memcpy(dst, &depth, sizeof(depth));
            }
          }
        }
      }
    }
  }

  if (!entry) {
    return true;
  }
  const VkImageView image_view = entry->GetOrCreateView(texture.view_type, texture.format,
                                                        texture.swizzle, texture.array_layers);
  if (image_view == VK_NULL_HANDLE) {
    return false;
  }
  texture.cached_texture = entry;
  texture.cached_view = image_view;
  texture.upload_cached = upload;
  return true;
}

}  // namespace rex::graphics::native
