#pragma once

#include <array>
#include <memory>
#include <mutex>
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
  };
  static constexpr size_t kMaxResolvedImages = 256;

  NativeTextureCache(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                     memory::Memory& memory, const RegisterFile& regs,
                     const SubmissionClock& clock, uint32_t draw_resolution_scale_x,
                     uint32_t draw_resolution_scale_y)
      : device_(device),
        allocator_(allocator),
        memory_(memory),
        regs_(regs),
        clock_(clock),
        draw_resolution_scale_x_(draw_resolution_scale_x),
        draw_resolution_scale_y_(draw_resolution_scale_y) {}
  ~NativeTextureCache();
  NativeTextureCache(const NativeTextureCache&) = delete;
  NativeTextureCache& operator=(const NativeTextureCache&) = delete;

  // watch_pages: track the cached pages, so invalidations can be widened to them.
  bool Initialize(bool watch_pages);
  // Clears the fallback textures and moves them to the shader-read layout once,
  // before any draw that may sample them.
  void RecordNullTextureInitialization(VkCommandBuffer commands);
  // Reads the textures and samplers a vertex or pixel shader binds.
  bool ReadTextures(const SpirvShader& shader, bool vertex, DrawInput& input);

  void InvalidateTextures(uint32_t start, uint32_t length);
  void InvalidateResolvedTextures(uint32_t start, uint32_t length);
  // Drops every texture and resolve, keeping samplers.
  void InvalidateAll();
  void ClearSamplers();
  void PruneResolvedDependents();
  void RebuildCachedTexturePages();
  // Marks guest pages as holding cached data.
  void MarkCachedPages(uint32_t start, uint32_t length);
  // Called from guest threads: widens [start, start + length) to the cached
  // pages it touches, or returns a zero length when it touches none.
  std::pair<uint32_t, uint32_t> WidenInvalidation(uint32_t start, uint32_t length);

  std::unordered_map<uint32_t, CachedResolve>& resolved_images() { return resolved_images_; }
  // Physical front buffer address of the last presented resolve. Kept across
  // submissions so eviction cannot drop the image the presenter is showing.
  std::optional<uint32_t>& presented_resolve_address() { return presented_resolve_address_; }

 private:
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
  static constexpr size_t kMaxCachedTextures = 1024;
  static constexpr size_t kMaxResolvedCubes = 64;
  static constexpr size_t kMaxResolvedCrops = 64;

  // signed_variant: the texels for signed components, for formats whose
  // unpacking depends on the sign.
  bool ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch, DrawTexture& texture,
                   bool signed_variant);
  bool DecodeTextureLevels(const xenos::xe_gpu_texture_fetch_t& fetch,
                           const TextureFormatInfo& info,
                           const texture_util::TextureGuestLayout& layout, uint32_t address,
                           uint32_t mip_page, bool signed_variant, DrawTexture& texture);
  // Turns a copy of an unsigned texture into its view for signed components.
  bool MakeSignedView(DrawTexture& texture, VkFormat signed_format);
  bool IsLinearFilterable(VkFormat format);
  VkSampler GetSampler(const DrawSampler& key);
  bool InitializeNullTextures();
  void DestroyNullTextures();
  template <typename EntryMap, typename Key, typename OnCreated>
  std::shared_ptr<CachedTexture> GetOrCreateCachedTexture(EntryMap& cache, size_t max_entries,
                                                          const Key& key,
                                                          const VkImageCreateInfo& image_info,
                                                          OnCreated&& on_created, bool& failed);

  const ui::vulkan::VulkanDevice& device_;
  VmaAllocator allocator_;
  memory::Memory& memory_;
  const RegisterFile& regs_;
  const SubmissionClock& clock_;
  uint32_t draw_resolution_scale_x_;
  uint32_t draw_resolution_scale_y_;
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
  std::unordered_map<uint32_t, bool> linear_filterable_;
  uint32_t custom_border_color_samplers_ = 0;
  std::unordered_map<uint32_t, CachedResolve> resolved_images_;
  std::optional<uint32_t> presented_resolve_address_;
  std::unordered_map<DrawSampler, CachedSampler, DrawSamplerHash> samplers_;
  std::unordered_map<GuestTextureKey, CachedTextureEntry, GuestTextureKeyHash> cached_textures_;
  std::unordered_map<ResolvedCubeKey, CachedTextureEntry, ResolvedCubeKeyHash> resolved_cubes_;
  std::unordered_map<ResolvedCropKey, CachedTextureEntry, ResolvedCropKeyHash> resolved_crops_;
  // Read from guest threads by WidenInvalidation, so it only guards the bitset.
  std::mutex pages_mutex_;
  std::vector<uint64_t> cached_texture_pages_;
};

}  // namespace rex::graphics::native
