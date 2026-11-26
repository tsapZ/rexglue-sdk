#include "target_cache.h"

#include <algorithm>
#include <optional>

namespace rex::graphics::native {

using Translator = SpirvShaderTranslator;

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
                                                          const std::shared_ptr<ColorImage>& keep) {
  const auto key = std::tuple(tile, pitch, samples, format);
  if (auto existing = targets_.find(key); existing != targets_.end()) {
    existing->second.last_use_submission = clock_.next;
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
  auto image = std::make_shared<ColorImage>(device_);
  if (!image->Initialize({pitch * draw_resolution_scale_x_, height * draw_resolution_scale_y_},
                         format, VkSampleCountFlagBits(1u << uint32_t(samples)), aspects)) {
    return {};
  }
  // shortcut: separate host images, share overlapping tiles when ownership tracking is added.
  targets_.emplace(key, CachedImage{image, clock_.next});
  if (!is_64bpp) {
    ApplyPendingFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, false), samples, image);
  }
  if ((aspects & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
    ApplyPendingColorFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp), samples,
                               format, image);
  }
  return image;
}

}  // namespace rex::graphics::native
