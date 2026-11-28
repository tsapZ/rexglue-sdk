#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/device.h>

#include "color_image.h"
#include "draw_pass.h"

namespace rex::graphics::native {

// Guest physical bytes, or empty when the range is outside physical memory or
// not accessible.
inline std::span<const uint8_t> ReadGuestPhysical(memory::Memory& memory, uint32_t address,
                                                  size_t size) {
  if (!IsPhysicalRange(address, size) ||
      memory.GetPhysicalHeap()->QueryRangeAccess(address, address + uint32_t(size) - 1) ==
          memory::PageAccess::kNoAccess) {
    return {};
  }
  return {memory.TranslatePhysical(address), size};
}

// Decoded guest textures, GPU resolves and the samplers draws bind, plus the
// physical pages they cover so guest writes to them invalidate the entries.
class NativeTextureCache {
 public:
  struct CachedResolve : CachedImage {
    uint32_t dest_info;
    uint32_t dest_pitch;
    uint64_t version = 0;
    bool tiled_destination = false;
    uint32_t guest_start = 0;
    uint32_t guest_length = 0;
    ResolveRectangle last_rect{};
    uint32_t copy_dest_pitch = 0;
    xenos::Endian128 endian = xenos::Endian128::kNone;
    bool rgba8_readback_valid = false;
    bool resolution_scaled = true;
    bool is_depth = false;
    xenos::DepthRenderTargetFormat depth_format = xenos::DepthRenderTargetFormat::kD24S8;
  };
  static constexpr size_t kMaxResolvedImages = 256;

  using WatchRangeFn = std::function<void(uint32_t start, uint32_t length)>;
  using KeepAliveTextureFn = std::function<void(const std::shared_ptr<CachedTexture>&)>;
  using KeepAliveImageFn = std::function<void(const std::shared_ptr<ColorImage>&)>;

  NativeTextureCache(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                     memory::Memory& memory, const RegisterFile& regs,
                     const SubmissionClock& clock, uint32_t draw_resolution_scale_x,
                     uint32_t draw_resolution_scale_y, WatchRangeFn watch_range = {})
      : device_(device),
        allocator_(allocator),
        memory_(memory),
        regs_(regs),
        clock_(clock),
        draw_resolution_scale_x_(draw_resolution_scale_x),
        draw_resolution_scale_y_(draw_resolution_scale_y),
        watch_range_(std::move(watch_range)) {}
  ~NativeTextureCache();
  NativeTextureCache(const NativeTextureCache&) = delete;
  NativeTextureCache& operator=(const NativeTextureCache&) = delete;

  // watch_pages: track the cached pages, so invalidations can be widened to them.
  bool Initialize(bool watch_pages);
  // Clears the fallback textures and moves them to the shader-read layout once,
  // before any draw that may sample them.
  void RecordNullTextureInitialization(VkCommandBuffer commands);
  // Reads the textures and samplers a vertex or pixel shader binds.
  bool ReadTextures(const SpirvShader& shader, bool vertex, DrawInput& input,
                    ui::vulkan::VulkanUploadBufferPool* uploads = nullptr,
                    uint64_t submission = 0);

  void InvalidateTextures(uint32_t start, uint32_t length);
  void InvalidateResolvedTextures(uint32_t start, uint32_t length);
  // Invalidates both caches in a single walk for sorted, coalesced write ranges.
  void InvalidateWrittenRanges(std::span<const PhysicalRange> ranges);
  // Drops every texture and resolve, keeping samplers.
  void InvalidateAll();
  void ClearSamplers();
  void PruneResolvedDependents();
  void RebuildCachedTexturePages();
  // Marks guest pages as holding cached resolve data.
  void MarkCachedPages(uint32_t start, uint32_t length);
  void set_view_purger(const DrawBindingsViewPurger* purger) { view_purger_ = purger; }
  void set_keep_alive_texture(KeepAliveTextureFn fn) { keep_alive_texture_ = std::move(fn); }
  void set_keep_alive_image(KeepAliveImageFn fn) { keep_alive_image_ = std::move(fn); }
  void BumpCacheEpoch() { ++cache_epoch_; }
  // Lock-free atomic read of a 256 KiB block's GPU-resolved-page mask.
  uint64_t ResolvedPagesWord(uint32_t block) const {
    return block < resolved_pages_.size()
               ? std::atomic_ref<const uint64_t>(resolved_pages_[block])
                     .load(std::memory_order_relaxed)
               : 0;
  }
  // Lock-free atomic read of a 256 KiB block's protected-page mask (textures | resolves).
  uint64_t ProtectedPagesWord(uint32_t block) const {
    uint64_t word = 0;
    if (block < cached_only_pages_.size()) {
      word |= std::atomic_ref<const uint64_t>(cached_only_pages_[block])
                  .load(std::memory_order_relaxed);
    }
    if (block < resolved_pages_.size()) {
      word |=
          std::atomic_ref<const uint64_t>(resolved_pages_[block]).load(std::memory_order_relaxed);
    }
    return word;
  }
  // Called from guest threads: widens [start, start + length) to the cached
  // pages it touches, or returns a zero length when it touches none.
  std::pair<uint32_t, uint32_t> WidenInvalidation(uint32_t start, uint32_t length);

  std::unordered_map<uint32_t, CachedResolve>& resolved_images() { return resolved_images_; }
  // Physical front buffer address of the last presented resolve. Kept across
  // submissions so eviction cannot drop the image the presenter is showing.
  std::optional<uint32_t>& presented_resolve_address() { return presented_resolve_address_; }

 private:
  enum class InvalidationTarget { kTextures, kResolves, kBoth };
  void InvalidateRanges(std::span<const PhysicalRange> ranges, InvalidationTarget target);

  struct CachedSampler {
    VkSampler sampler = VK_NULL_HANDLE;
    uint64_t last_use_submission = 0;
    bool custom_border_color = false;
  };
  struct CachedTextureEntry {
    std::shared_ptr<CachedTexture> texture;
    uint64_t last_use_submission = 0;
  };
  static constexpr size_t kMaxSamplers = 256;
  static constexpr size_t kMaxCachedTextures = 8192;
  static constexpr size_t kMaxResolvedCubes = 64;
  static constexpr size_t kMaxResolvedCrops = 64;

  // signed_variant: the texels for signed components, for formats whose
  // unpacking depends on the sign.
  bool ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch, DrawTexture& texture,
                   bool signed_variant, ui::vulkan::VulkanUploadBufferPool* uploads = nullptr,
                   uint64_t submission = 0, CachedTextureEntry** cached_entry_out = nullptr);
  CachedTexture* KeepTextureAlive(const std::shared_ptr<CachedTexture>& texture) const {
    if (texture && keep_alive_texture_ && texture->last_keepalive_submission != clock_.next) {
      keep_alive_texture_(texture);
    }
    return texture.get();
  }
  ColorImage* KeepImageAlive(const std::shared_ptr<ColorImage>& image) const {
    if (image && keep_alive_image_ && image->last_keepalive_submission() != clock_.next) {
      keep_alive_image_(image);
    }
    return image.get();
  }
  bool DecodeTextureLevels(const xenos::xe_gpu_texture_fetch_t& fetch,
                           const TextureFormatInfo& info,
                           const texture_util::TextureGuestLayout& layout, uint32_t address,
                           uint32_t mip_page, bool signed_variant, DrawTexture& texture,
                           ui::vulkan::VulkanUploadBufferPool* uploads = nullptr,
                           uint64_t submission = 0);
  // Turns a copy of an unsigned texture into its view for signed components.
  bool MakeSignedView(DrawTexture& texture, VkFormat signed_format);
  bool IsLinearFilterable(xenos::TextureFormat guest_format, VkFormat format);
  VkSampler GetSampler(const DrawSampler& key);
  bool InitializeNullTextures();
  void DestroyNullTextures();
  template <typename EntryMap, typename Key, typename OnCreated>
  std::shared_ptr<CachedTexture> GetOrCreateCachedTexture(EntryMap& cache, size_t max_entries,
                                                          const Key& key,
                                                          const VkImageCreateInfo& image_info,
                                                          OnCreated&& on_created, bool& failed);

  static constexpr uint32_t kBucketShift = 18;  // 256 KiB = 64 pages = 1 bitmap word
  static constexpr size_t kBucketCount = kPhysicalMemorySize >> kBucketShift;

  void AddTextureToBuckets(const GuestTextureKey& key, const CachedTexture& texture);
  void RemoveTextureFromBuckets(const GuestTextureKey& key, const CachedTexture& texture,
                                std::vector<uint32_t>* dirty_buckets = nullptr);
  void RecomputeTexturePageWord(uint32_t block);
  void RecomputeResolvedPages();

  const ui::vulkan::VulkanDevice& device_;
  VmaAllocator allocator_;
  memory::Memory& memory_;
  const RegisterFile& regs_;
  const SubmissionClock& clock_;
  uint32_t draw_resolution_scale_x_;
  uint32_t draw_resolution_scale_y_;
  WatchRangeFn watch_range_;
  // 1x1 zero-cleared cube-compatible image shared by the 2D array and cube
  // fallback views bound for fetch constants that cannot produce texels.
  VkImage null_texture_image_ = VK_NULL_HANDLE;
  VkDeviceMemory null_texture_memory_ = VK_NULL_HANDLE;
  VkImageView null_texture_2d_view_ = VK_NULL_HANDLE;
  VkImageView null_texture_cube_view_ = VK_NULL_HANDLE;
  VkImage null_texture_3d_image_ = VK_NULL_HANDLE;
  VkDeviceMemory null_texture_3d_memory_ = VK_NULL_HANDLE;
  VkImageView null_texture_3d_view_ = VK_NULL_HANDLE;
  bool null_textures_initialized_ = false;
  std::array<int8_t, 64> linear_filterable_ = [] {
    std::array<int8_t, 64> a{};
    a.fill(-1);
    return a;
  }();
  uint32_t custom_border_color_samplers_ = 0;
  std::unordered_map<uint32_t, CachedResolve> resolved_images_;
  std::optional<uint32_t> presented_resolve_address_;
  std::unordered_map<DrawSampler, CachedSampler, DrawSamplerHash> samplers_;
  std::unordered_map<GuestTextureKey, CachedTextureEntry, GuestTextureKeyHash> cached_textures_;
  std::unordered_map<ResolvedCubeKey, CachedTextureEntry, ResolvedCubeKeyHash> resolved_cubes_;
  std::unordered_map<ResolvedCropKey, CachedTextureEntry, ResolvedCropKeyHash> resolved_crops_;
  std::array<std::vector<const GuestTextureKey*>, kBucketCount> texture_buckets_;
  // Fixed-size bitsets read atomically from guest threads by ProtectedPagesWord.
  std::vector<uint64_t> cached_only_pages_;
  std::vector<uint64_t> resolved_pages_;
  std::vector<uint64_t> scratch_resolved_pages_;
  std::vector<uint8_t> decode_row_scratch_;
  struct SamplerMruSlot {
    DrawSampler key{};
    VkSampler sampler = VK_NULL_HANDLE;
    uint64_t* last_use = nullptr;
    bool valid = false;
  };
  struct RawFetchCacheEntry {
    uint64_t epoch = 0;
    std::array<uint32_t, 6> dwords{};
    uint8_t slot = 0;
    bool signed_variant = false;
    bool valid = false;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t array_layers = 1;
    uint32_t level_count = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool mutable_format = false;
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t host_swizzle = 0;
    VkComponentMapping swizzle{};
    CachedTextureEntry* cached_entry = nullptr;
    VkImageView cached_view = VK_NULL_HANDLE;
    bool resolution_scaled = false;
  };
  uint64_t cache_epoch_ = 1;
  std::array<SamplerMruSlot, 4> sampler_mru_{};
  std::array<RawFetchCacheEntry, 32> raw_fetch_cache_{};
  KeepAliveTextureFn keep_alive_texture_;
  KeepAliveImageFn keep_alive_image_;
  const DrawBindingsViewPurger* view_purger_ = nullptr;
};

}  // namespace rex::graphics::native
