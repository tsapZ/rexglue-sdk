#pragma once

#include <functional>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <rex/ui/vulkan/device.h>

#include "color_image.h"
#include "draw_pass.h"

namespace rex::graphics::native {

// EDRAM render targets as host images, keyed by tile, pitch, sample count and
// format, plus the aliased 4x-MSAA fast clears D3D9 issues for 1x/2x tiles.
class NativeTargetCache {
 public:
  // A 4x-MSAA depth-only draw that D3D9 issued to clear an aliased 1x- or
  // 2x-MSAA EDRAM tile. Kept so a target created later for the same tile is
  // initialized with the clear before it is drawn to.
  struct PendingFastClear {
    DrawInput input;
    std::vector<uint32_t> indices;
    const Shader::Translation* vertex = nullptr;
    xenos::DepthRenderTargetFormat depth_format = xenos::DepthRenderTargetFormat::kD24S8;
    bool color_clear_capable = false;
    uint32_t clear_stencil = 0;
  };
  struct PendingColorFastClear {
    DrawInput input;
    std::vector<uint32_t> indices;
    const Shader::Translation* vertex = nullptr;
    const Shader::Translation* pixel = nullptr;
    uint32_t rt_index = 0;
  };
  // Queues a replayed clear draw on `image`: color attachment `rt_index` of a
  // color image, or the depth attachment of a depth image.
  using QueueDraw =
      std::function<void(const DrawInput& input, std::span<const uint32_t> indices,
                         const Shader::Translation& vertex, const Shader::Translation* pixel,
                         uint32_t rt_index, const std::shared_ptr<ColorImage>& image)>;
  // Queues copies of EDRAM tiles from the target that used them last into a
  // target aliasing them.
  using QueueTransfer = std::function<void(const std::shared_ptr<ColorImage>& source,
                                           const std::shared_ptr<ColorImage>& destination,
                                           std::vector<VkImageCopy> copies)>;
  using TargetKey = std::tuple<uint32_t, uint32_t, xenos::MsaaSamples, VkFormat>;

  NativeTargetCache(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                    const SubmissionClock& clock, uint32_t draw_resolution_scale_x,
                    uint32_t draw_resolution_scale_y, bool depth_unorm24_supported,
                    QueueDraw queue_draw, QueueTransfer queue_transfer)
      : device_(device),
        allocator_(allocator),
        clock_(clock),
        draw_resolution_scale_x_(draw_resolution_scale_x),
        draw_resolution_scale_y_(draw_resolution_scale_y),
        depth_unorm24_supported_(depth_unorm24_supported),
        queue_draw_(std::move(queue_draw)),
        queue_transfer_(std::move(queue_transfer)) {}

  // Never evicts `keep` (the presented image). used_height: guest pixel rows
  // from the top of the target about to be drawn or resolved; their EDRAM
  // tiles become owned by this target, taking over the contents left there by
  // an aliasing target of the same pitch, sample count and format.
  std::shared_ptr<ColorImage> FindTarget(uint32_t tile, uint32_t pitch, xenos::MsaaSamples samples,
                                         VkFormat format, VkImageAspectFlags aspects, bool is_64bpp,
                                         uint32_t edram_tile_count,
                                         const std::shared_ptr<ColorImage>& keep,
                                         uint32_t used_height = 0);
  VkFormat GetDepthFormat(xenos::DepthRenderTargetFormat format) const;

  // Replays an aliased 4x-MSAA fast clear onto a 1x/2x target that already
  // exists (depth or color).
  void ReplayAliasedFastClear(const PendingFastClear& clear,
                              const std::shared_ptr<ColorImage>& image, xenos::MsaaSamples samples);
  void ReplayAliasedColorFastClear(const PendingColorFastClear& clear,
                                   const std::shared_ptr<ColorImage>& image,
                                   xenos::MsaaSamples samples);
  void ErasePendingColorFastClears(uint32_t tile, uint32_t pitch_tiles);
  // The 1x/2x targets matching predicate(tile, pitch, samples, format, image),
  // marked used; collected first so replaying never mutates targets_ while
  // iterating it.
  template <typename Predicate>
  std::vector<std::pair<std::shared_ptr<ColorImage>, xenos::MsaaSamples>> CollectAliases(
      Predicate&& predicate) {
    std::vector<std::pair<std::shared_ptr<ColorImage>, xenos::MsaaSamples>> aliases;
    for (auto& [target_key, entry] : targets_) {
      const auto& [tile, pitch, samples, format] = target_key;
      if (samples >= xenos::MsaaSamples::k4X ||
          !predicate(tile, pitch, samples, format, *entry.image)) {
        continue;
      }
      entry.last_use_submission = clock_.next;
      aliases.emplace_back(entry.image, samples);
    }
    return aliases;
  }

  std::map<std::pair<uint32_t, uint32_t>, PendingFastClear>& pending_fast_clears() {
    return pending_fast_clears_;
  }
  std::map<std::tuple<uint32_t, uint32_t, VkFormat>, PendingColorFastClear>&
  pending_color_fast_clears() {
    return pending_color_fast_clears_;
  }
  void Clear() {
    pending_fast_clears_.clear();
    pending_color_fast_clears_.clear();
    targets_.clear();
    tile_owners_.clear();
    ++ownership_epoch_;
  }

 private:
  static constexpr size_t kMaxTargets = 64;

  struct CachedTarget : CachedImage {
    // The tiles up to claimed_end were owned by this target at claimed_epoch.
    uint64_t claimed_epoch = 0;
    uint32_t claimed_end = 0;
  };
  // The target that last used an EDRAM tile.
  struct TileOwner {
    // Compared by control block, so a later image at the same address never
    // looks like the owner.
    std::weak_ptr<ColorImage> image;
    uint32_t base = 0;
    uint32_t pitch_tiles = 0;
    xenos::MsaaSamples samples = xenos::MsaaSamples::k1X;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool is_64bpp = false;
  };
  void ClaimTiles(CachedTarget& target, uint32_t tile, uint32_t pitch, xenos::MsaaSamples samples,
                  VkFormat format, bool is_64bpp, uint32_t used_height,
                  uint32_t edram_tile_count);

  DrawInput RescaleFastClearInput(const DrawInput& base, xenos::MsaaSamples samples) const;
  // Finds a pending fast clear for an aliased target tile and replays it if it
  // is compatible with the newly created image, then drops it.
  void ApplyPendingFastClear(uint32_t tile, uint32_t pitch_tiles, xenos::MsaaSamples samples,
                             const std::shared_ptr<ColorImage>& image);
  void ApplyPendingColorFastClear(uint32_t tile, uint32_t pitch_tiles, xenos::MsaaSamples samples,
                                  VkFormat format, const std::shared_ptr<ColorImage>& image);

  const ui::vulkan::VulkanDevice& device_;
  VmaAllocator allocator_;
  const SubmissionClock& clock_;
  uint32_t draw_resolution_scale_x_;
  uint32_t draw_resolution_scale_y_;
  // Whether VK_FORMAT_D24_UNORM_S8_UINT supports optimal-tiling sampled image
  // and depth/stencil attachment usage on this device.
  bool depth_unorm24_supported_;
  QueueDraw queue_draw_;
  QueueTransfer queue_transfer_;
  std::map<TargetKey, CachedTarget> targets_;
  std::vector<TileOwner> tile_owners_;
  // Advances whenever a tile changes owner, so targets whose tiles are all
  // still their own skip the ownership scan.
  uint64_t ownership_epoch_ = 1;
  std::map<std::pair<uint32_t, uint32_t>, PendingFastClear> pending_fast_clears_;
  std::map<std::tuple<uint32_t, uint32_t, VkFormat>, PendingColorFastClear>
      pending_color_fast_clears_;
};

}  // namespace rex::graphics::native
