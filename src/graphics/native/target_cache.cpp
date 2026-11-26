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
                                                   xenos::MsaaSamples samples, bool source_scaled,
                                                   bool target_scaled) const {
  // The 4x-MSAA clear covers the aliased target with half the horizontal pitch;
  // scale the guest-space viewport and scissor to the host pixels of the alias.
  const auto scale = GetMsaa4xAliasScale(samples);
  uint32_t sx = scale.x;
  uint32_t sy = scale.y;
  if (!source_scaled && target_scaled) {
    sx *= draw_resolution_scale_x_;
    sy *= draw_resolution_scale_y_;
  }
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
  ScaleViewportAndScissor(input.viewport, input.scissor, ndc_scale, ndc_offset, sx, sy,
                          properties.maxViewportDimensions[0], properties.maxViewportDimensions[1]);
  std::copy(ndc_scale.begin(), ndc_scale.end(), input.viewport.ndc_scale);
  std::copy(ndc_offset.begin(), ndc_offset.end(), input.viewport.ndc_offset);
  std::copy(ndc_scale.begin(), ndc_scale.end(), input.system.ndc_scale);
  std::copy(ndc_offset.begin(), ndc_offset.end(), input.system.ndc_offset);
  return input;
}

void NativeTargetCache::ReplayAliasedFastClear(const PendingFastClear& clear,
                                               const std::shared_ptr<ColorImage>& image,
                                               xenos::MsaaSamples samples, bool target_scaled) {
  const bool depth_target = (image->aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
  std::optional<uint32_t> clear_format;
  if (!depth_target) {
    clear_format = GetEdramClearColorFormatMode(image->format());
    if (!clear_format) {
      return;
    }
  }
  DrawInput input = RescaleFastClearInput(clear.input, samples, clear.resolution_scaled,
                                          target_scaled);
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
                                              const std::shared_ptr<ColorImage>& image,
                                              bool target_scaled) {
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
    ReplayAliasedFastClear(clear, image, samples, target_scaled);
  }
}

void NativeTargetCache::ReplayAliasedColorFastClear(const PendingColorFastClear& clear,
                                                    const std::shared_ptr<ColorImage>& image,
                                                    xenos::MsaaSamples samples,
                                                    bool target_scaled) {
  const Shader::Translation* ps =
      (target_scaled && !clear.resolution_scaled && clear.pixel_scaled) ? clear.pixel_scaled
                                                                        : clear.pixel;
  queue_draw_(RescaleFastClearInput(clear.input, samples, clear.resolution_scaled, target_scaled),
              clear.indices, *clear.vertex, ps, clear.rt_index, image);
}

void NativeTargetCache::ApplyPendingColorFastClear(uint32_t tile, uint32_t pitch_tiles,
                                                   xenos::MsaaSamples samples, VkFormat format,
                                                   const std::shared_ptr<ColorImage>& image,
                                                   bool target_scaled) {
  if (samples >= xenos::MsaaSamples::k4X || !pitch_tiles) {
    return;
  }
  const auto pending = pending_color_fast_clears_.find({tile, pitch_tiles, format});
  if (pending == pending_color_fast_clears_.end()) {
    return;
  }
  ReplayAliasedColorFastClear(pending->second, image, samples, target_scaled);
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
                                                          uint32_t used_height,
                                                          bool resolution_scaled) {
  const bool scaled =
      resolution_scaled && (draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1);
  const uint32_t sx = scaled ? draw_resolution_scale_x_ : 1;
  const uint32_t sy = scaled ? draw_resolution_scale_y_ : 1;
  const TargetKey key{tile, pitch, samples, format, scaled};
  if (last_target_valid_ && last_target_key_ == key) {
    last_target_->last_use_submission = clock_.next;
    ClaimTiles(*last_target_, tile, pitch, samples, format, is_64bpp, used_height, edram_tile_count,
               scaled);
    return last_target_->image;
  }
  if (auto existing = targets_.find(key); existing != targets_.end()) {
    last_target_key_ = key;
    last_target_ = &existing->second;
    last_target_valid_ = true;
    existing->second.last_use_submission = clock_.next;
    ClaimTiles(existing->second, tile, pitch, samples, format, is_64bpp, used_height,
               edram_tile_count, scaled);
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
    if (auto it = targets_.find(*victim); it != targets_.end()) {
      owners_.erase(it->second.owner_id);
      if (last_target_valid_ && last_target_key_ == *victim) {
        last_target_valid_ = false;
        last_target_ = nullptr;
      }
      targets_.erase(it);
    }
  }
  const auto& properties = device_.properties();
  const uint32_t max_width =
      std::min(properties.maxImageDimension2D, properties.maxFramebufferWidth);
  const uint32_t max_height =
      std::min(properties.maxImageDimension2D, properties.maxFramebufferHeight);
  if (uint64_t(pitch) * sx > max_width) {
    return {};
  }
  const uint32_t height =
      GetTargetHeight(pitch, samples, edram_tile_count, max_height / sy, is_64bpp);
  auto image = std::make_shared<ColorImage>(device_, allocator_);
  if (!image->Initialize({pitch * sx, height * sy}, format,
                         VkSampleCountFlagBits(1u << uint32_t(samples)), aspects)) {
    return {};
  }
  // ponytail: only same-format, same-sample, same-pitch aliases transfer their
  // tiles; reinterpreting other formats needs transfer shaders.
  const uint64_t owner_id = next_owner_id_++;
  auto& created =
      targets_.emplace(key, CachedTarget{{image, clock_.next}, owner_id}).first->second;
  owners_.emplace(owner_id, image);
  last_target_key_ = key;
  last_target_ = &created;
  last_target_valid_ = true;
  // Inherit the aliased tiles first: a fast clear still pending is newer.
  ClaimTiles(created, tile, pitch, samples, format, is_64bpp, used_height, edram_tile_count,
             scaled);
  if (!is_64bpp) {
    ApplyPendingFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, false), samples, image,
                          scaled);
  }
  if ((aspects & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
    ApplyPendingColorFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp), samples,
                               format, image, scaled);
  }
  return image;
}

void NativeTargetCache::ClaimTiles(CachedTarget& target, uint32_t tile, uint32_t pitch,
                                   xenos::MsaaSamples samples, VkFormat format, bool is_64bpp,
                                   uint32_t used_height, uint32_t edram_tile_count,
                                   bool resolution_scaled) {
  const uint32_t pitch_tiles = xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp);
  const uint32_t sx = resolution_scaled ? draw_resolution_scale_x_ : 1;
  const uint32_t sy = resolution_scaled ? draw_resolution_scale_y_ : 1;
  const VkExtent2D tile_extent = GetEdramTileHostExtent(samples, is_64bpp, sx, sy);
  const uint32_t guest_tile_height = tile_extent.height / sy;
  if (!used_height || !pitch_tiles || !guest_tile_height || tile >= edram_tile_count) {
    return;
  }
  const uint64_t used_rows = (uint64_t(used_height) + guest_tile_height - 1) / guest_tile_height;
  const uint32_t end =
      uint32_t(std::min<uint64_t>(tile + used_rows * pitch_tiles, edram_tile_count));
  if (target.claimed_epoch == ownership_epoch_ && target.claimed_end >= end) {
    return;
  }
  const VkExtent2D extent = target.image->extent();
  const TileRun claim{tile,    end,     target.owner_id, tile,
                      pitch_tiles, samples, format,          is_64bpp,
                      resolution_scaled};
  const bool changed = SpliceTileRun(
      tile_runs_, claim, [&](const TileRun& owner, uint32_t run_start, uint32_t run_end) {
        if (owner.pitch_tiles != pitch_tiles || owner.samples != samples ||
            owner.format != format || owner.is_64bpp != is_64bpp ||
            owner.resolution_scaled != resolution_scaled) {
          return;
        }
        auto prev_it = owners_.find(owner.owner_id);
        if (prev_it == owners_.end()) {
          return;
        }
        const auto& previous = prev_it->second;
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
      });
  if (changed) {
    ++ownership_epoch_;
  }
  target.claimed_epoch = ownership_epoch_;
  target.claimed_end = end;
}

}  // namespace rex::graphics::native
