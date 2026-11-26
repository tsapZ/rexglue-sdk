#include "target_cache.h"

#include <algorithm>
#include <optional>

namespace rex::graphics::native {

using Translator = SpirvShaderTranslator;

namespace {

template <typename A, typename B>
bool SameOwner(const A& a, const B& b) {
  return !a.owner_before(b) && !b.owner_before(a);
}

}  // namespace

VkFormat NativeTargetCache::GetDepthFormat(xenos::DepthRenderTargetFormat format) const {
  return SelectDepthFormat(format, depth_unorm24_supported_);
}

DrawInput NativeTargetCache::RescaleFastClearInput(const DrawInput& base,
                                                   xenos::MsaaSamples samples) const {
  // The 4x-MSAA clear covers the aliased target with half the horizontal pitch;
  // scale the guest-space viewport and scissor to the host pixels of the alias.
  const auto scale = GetMsaa4xAliasScale(samples);
  const auto& properties = device_.properties();
  std::array<float, 3> ndc_scale;
  std::array<float, 3> ndc_offset;
  std::copy(std::begin(base.viewport.ndc_scale), std::end(base.viewport.ndc_scale),
            ndc_scale.begin());
  std::copy(std::begin(base.viewport.ndc_offset), std::end(base.viewport.ndc_offset),
            ndc_offset.begin());

  DrawInput input = base;
  input.samples = VkSampleCountFlagBits(1u << uint32_t(samples));
  constexpr uint32_t kMsaaSamplesFlagMask = ((1u << xenos::kMsaaSamplesBits) - 1u)
                                            << Translator::kSysFlag_MsaaSamples_Shift;
  input.system.flags = (base.system.flags & ~kMsaaSamplesFlagMask) |
                       (uint32_t(samples) << Translator::kSysFlag_MsaaSamples_Shift);
  ScaleViewportAndScissor(input.viewport, input.scissor, ndc_scale, ndc_offset, scale.x, scale.y,
                          properties.maxViewportDimensions[0], properties.maxViewportDimensions[1]);
  std::copy(ndc_scale.begin(), ndc_scale.end(), input.viewport.ndc_scale);
  std::copy(ndc_offset.begin(), ndc_offset.end(), input.viewport.ndc_offset);
  std::copy(ndc_scale.begin(), ndc_scale.end(), input.system.ndc_scale);
  std::copy(ndc_offset.begin(), ndc_offset.end(), input.system.ndc_offset);
  return input;
}

void NativeTargetCache::ReplayAliasedFastClear(const PendingFastClear& clear,
                                               const std::shared_ptr<ColorImage>& image,
                                               xenos::MsaaSamples samples) {
  const bool depth_target = (image->aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
  std::optional<uint32_t> clear_format;
  if (!depth_target) {
    clear_format = GetEdramClearColorFormatMode(image->format());
    if (!clear_format) {
      return;
    }
  }
  DrawInput input = RescaleFastClearInput(clear.input, samples);
  if (!depth_target) {
    // The clear writes a 32-bit EDRAM depth-stencil word into an aliased color
    // target; the fragment shader unpacks it for the target format.
    input.color_count = 1;
    input.color_formats[0] = image->format();
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_FALSE;
    blend.colorWriteMask = 0xF;
    input.blend[0] = blend;
    input.depth_format = VK_FORMAT_UNDEFINED;
    input.depth_stencil = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    input.edram_depth_clear_color = true;
    input.edram_clear_stencil = clear.clear_stencil;
    input.edram_clear_format = *clear_format;
  }
  queue_draw_(input, clear.indices, *clear.vertex, nullptr, 0, image);
}

void NativeTargetCache::ApplyPendingFastClear(uint32_t tile, uint32_t pitch_tiles,
                                              xenos::MsaaSamples samples,
                                              const std::shared_ptr<ColorImage>& image) {
  if (samples >= xenos::MsaaSamples::k4X) {
    return;
  }
  const auto pending = pending_fast_clears_.find({tile, pitch_tiles});
  if (pending == pending_fast_clears_.end()) {
    return;
  }
  const PendingFastClear& clear = pending->second;
  const bool depth_target = (image->aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
                            image->format() == GetDepthFormat(clear.depth_format);
  const bool color_target = (image->aspects() & VK_IMAGE_ASPECT_COLOR_BIT) != 0 &&
                            clear.color_clear_capable &&
                            GetEdramClearColorFormatMode(image->format()).has_value();
  if (depth_target || color_target) {
    ReplayAliasedFastClear(clear, image, samples);
  }
}

void NativeTargetCache::ReplayAliasedColorFastClear(const PendingColorFastClear& clear,
                                                    const std::shared_ptr<ColorImage>& image,
                                                    xenos::MsaaSamples samples) {
  queue_draw_(RescaleFastClearInput(clear.input, samples), clear.indices, *clear.vertex,
              clear.pixel, clear.rt_index, image);
}

void NativeTargetCache::ApplyPendingColorFastClear(uint32_t tile, uint32_t pitch_tiles,
                                                   xenos::MsaaSamples samples, VkFormat format,
                                                   const std::shared_ptr<ColorImage>& image) {
  if (samples >= xenos::MsaaSamples::k4X || !pitch_tiles) {
    return;
  }
  const auto pending = pending_color_fast_clears_.find({tile, pitch_tiles, format});
  if (pending == pending_color_fast_clears_.end()) {
    return;
  }
  ReplayAliasedColorFastClear(pending->second, image, samples);
}

void NativeTargetCache::ErasePendingColorFastClears(uint32_t tile, uint32_t pitch_tiles) {
  for (auto it = pending_color_fast_clears_.lower_bound({tile, pitch_tiles, VK_FORMAT_UNDEFINED});
       it != pending_color_fast_clears_.end() && std::get<0>(it->first) == tile &&
       std::get<1>(it->first) == pitch_tiles;) {
    it = pending_color_fast_clears_.erase(it);
  }
}

std::shared_ptr<ColorImage> NativeTargetCache::FindTarget(uint32_t tile, uint32_t pitch,
                                                          xenos::MsaaSamples samples,
                                                          VkFormat format,
                                                          VkImageAspectFlags aspects, bool is_64bpp,
                                                          uint32_t edram_tile_count,
                                                          const std::shared_ptr<ColorImage>& keep,
                                                          uint32_t used_height) {
  const auto key = std::tuple(tile, pitch, samples, format);
  if (auto existing = targets_.find(key); existing != targets_.end()) {
    existing->second.last_use_submission = clock_.next;
    ClaimTiles(existing->second, tile, pitch, samples, format, is_64bpp, used_height,
               edram_tile_count);
    return existing->second.image;
  }
  if (!pitch) {
    return {};
  }
  if (targets_.size() >= kMaxTargets) {
    // Evict the least recently used target whose GPU work has completed. Only
    // submissions up to the last retired fence are eligible, so queued and
    // in-flight draws keep their images, and the presented image is kept too.
    const auto victim = SelectRetiredEviction(
        targets_, clock_.retired + 1,
        [&keep](const auto&, const CachedImage& entry) { return entry.image != keep; });
    if (!victim) {
      return {};
    }
    targets_.erase(*victim);
  }
  const auto& properties = device_.properties();
  const uint32_t max_width =
      std::min(properties.maxImageDimension2D, properties.maxFramebufferWidth);
  const uint32_t max_height =
      std::min(properties.maxImageDimension2D, properties.maxFramebufferHeight);
  if (uint64_t(pitch) * draw_resolution_scale_x_ > max_width) {
    return {};
  }
  const uint32_t height = GetTargetHeight(pitch, samples, edram_tile_count,
                                          max_height / draw_resolution_scale_y_, is_64bpp);
  auto image = std::make_shared<ColorImage>(device_, allocator_);
  if (!image->Initialize({pitch * draw_resolution_scale_x_, height * draw_resolution_scale_y_},
                         format, VkSampleCountFlagBits(1u << uint32_t(samples)), aspects)) {
    return {};
  }
  // ponytail: only same-format, same-sample, same-pitch aliases transfer their
  // tiles; reinterpreting other formats needs transfer shaders.
  auto& created = targets_.emplace(key, CachedTarget{{image, clock_.next}}).first->second;
  // Inherit the aliased tiles first: a fast clear still pending is newer.
  ClaimTiles(created, tile, pitch, samples, format, is_64bpp, used_height, edram_tile_count);
  if (!is_64bpp) {
    ApplyPendingFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, false), samples, image);
  }
  if ((aspects & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
    ApplyPendingColorFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp), samples,
                               format, image);
  }
  return image;
}

void NativeTargetCache::ClaimTiles(CachedTarget& target, uint32_t tile, uint32_t pitch,
                                   xenos::MsaaSamples samples, VkFormat format, bool is_64bpp,
                                   uint32_t used_height, uint32_t edram_tile_count) {
  const uint32_t pitch_tiles = xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp);
  const VkExtent2D tile_extent =
      GetEdramTileHostExtent(samples, is_64bpp, draw_resolution_scale_x_, draw_resolution_scale_y_);
  const uint32_t guest_tile_height = tile_extent.height / draw_resolution_scale_y_;
  if (!used_height || !pitch_tiles || !guest_tile_height || tile >= edram_tile_count) {
    return;
  }
  const uint64_t used_rows = (uint64_t(used_height) + guest_tile_height - 1) / guest_tile_height;
  const uint32_t end =
      uint32_t(std::min<uint64_t>(tile + used_rows * pitch_tiles, edram_tile_count));
  if (target.claimed_epoch == ownership_epoch_ && target.claimed_end >= end) {
    return;
  }
  if (tile_owners_.size() < edram_tile_count) {
    tile_owners_.resize(edram_tile_count);
  }
  const VkExtent2D extent = target.image->extent();
  bool changed = false;
  for (uint32_t run_start = tile; run_start < end;) {
    const TileOwner owner = tile_owners_[run_start];
    uint32_t run_end = run_start + 1;
    while (run_end < end && SameOwner(tile_owners_[run_end].image, owner.image)) {
      ++run_end;
    }
    if (SameOwner(owner.image, target.image)) {
      run_start = run_end;
      continue;
    }
    const auto previous = owner.image.lock();
    // Whether the source was ever written is only known when the copy is recorded.
    if (previous && owner.pitch_tiles == pitch_tiles &&
        owner.samples == samples && owner.format == format && owner.is_64bpp == is_64bpp) {
      const VkExtent2D source_extent = previous->extent();
      std::vector<VkImageCopy> copies;
      for (const auto& rect : ComputeOverlappingEdramTileCopies(owner.base, tile, pitch_tiles,
                                                                run_start, run_end)) {
        const uint32_t src_x = rect.src_x * tile_extent.width;
        const uint32_t src_y = rect.src_y * tile_extent.height;
        const uint32_t dst_x = rect.dst_x * tile_extent.width;
        const uint32_t dst_y = rect.dst_y * tile_extent.height;
        if (src_x >= source_extent.width || src_y >= source_extent.height ||
            dst_x >= extent.width || dst_y >= extent.height) {
          continue;
        }
        VkImageCopy copy{};
        copy.srcSubresource = copy.dstSubresource = {target.image->aspects(), 0, 0, 1};
        copy.srcOffset = {int32_t(src_x), int32_t(src_y), 0};
        copy.dstOffset = {int32_t(dst_x), int32_t(dst_y), 0};
        copy.extent = {std::min({rect.width * tile_extent.width, source_extent.width - src_x,
                                 extent.width - dst_x}),
                       std::min({rect.height * tile_extent.height, source_extent.height - src_y,
                                 extent.height - dst_y}),
                       1};
        copies.push_back(copy);
      }
      if (!copies.empty()) {
        queue_transfer_(previous, target.image, std::move(copies));
      }
    }
    for (uint32_t i = run_start; i < run_end; ++i) {
      tile_owners_[i] = {target.image, tile, pitch_tiles, samples, format, is_64bpp};
    }
    changed = true;
    run_start = run_end;
  }
  if (changed) {
    ++ownership_epoch_;
  }
  target.claimed_epoch = ownership_epoch_;
  target.claimed_end = end;
}

}  // namespace rex::graphics::native
