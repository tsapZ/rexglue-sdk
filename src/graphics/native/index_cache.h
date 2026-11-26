#pragma once

#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/graphics/xenos.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/mem_alloc.h>
#include <rex/ui/vulkan/upload_buffer_pool.h>

#include "color_image.h"
#include "draw_input.h"

namespace rex::graphics::native {

struct IndexCacheKey {
  uint32_t guest_base = 0;
  uint32_t index_count = 0;
  xenos::IndexFormat format = xenos::IndexFormat::kInt16;
  xenos::Endian endianness = xenos::Endian::kNone;
  bool primitive_restart = false;
  uint32_t reset_index = 0;
  bool use_uint16 = false;
  auto operator<=>(const IndexCacheKey&) const = default;
};

struct IndexCacheKeyHash {
  size_t operator()(const IndexCacheKey& key) const noexcept {
    uint64_t h = uint64_t(key.guest_base) * 0x9E3779B97F4A7C15ull;
    h ^= uint64_t(key.index_count) * 0xBF58476D1CE4E5B9ull;
    h ^= (uint64_t(key.format) | (uint64_t(key.endianness) << 4) |
          (uint64_t(key.primitive_restart) << 8) | (uint64_t(key.use_uint16) << 9)) *
         0x94D049BB133111EBull;
    h ^= uint64_t(key.reset_index) * 0x517CC1B727220A95ull;
    return size_t(h ^ (h >> 32));
  }
};

struct CachedIndexUpload {
  VkBuffer src_buffer = VK_NULL_HANDLE;
  VkDeviceSize src_offset = 0;
  VkBuffer dst_buffer = VK_NULL_HANDLE;
  VkDeviceSize dst_offset = 0;
  VkDeviceSize size = 0;
};

// Device-local cache of decoded static DMA index buffers, page-invalidated via
// guest memory write watches.
class NativeIndexCache {
 public:
  NativeIndexCache(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                   memory::Memory& memory, const SubmissionClock& clock)
      : device_(device), allocator_(allocator), memory_(memory), clock_(clock) {}
  ~NativeIndexCache();
  NativeIndexCache(const NativeIndexCache&) = delete;
  NativeIndexCache& operator=(const NativeIndexCache&) = delete;

  bool Initialize(bool watch_pages);
  void Invalidate(uint32_t start, uint32_t length);
  void InvalidateAll();
  void ClearRecentInvalidations();
  std::pair<uint32_t, uint32_t> WidenInvalidation(uint32_t start, uint32_t length);

  // Looks up or decodes+caches a static index buffer. Returns true with `binding`
  // populated on hit or queued upload (with `upload_out` filled on miss); returns
  // false when the range should be streamed transiently instead (too small,
  // recently invalidated, or cache full).
  bool GetOrCreate(const IndexCacheKey& key, std::span<const uint8_t> guest_bytes,
                   ui::vulkan::VulkanUploadBufferPool& uploads, IndexBinding& binding_out,
                   CachedIndexUpload& upload_out, bool& uploaded_out);

 private:
  struct Chunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize used = 0;
    uint64_t last_use_submission = 0;
  };
  struct Entry {
    size_t chunk_index = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    uint32_t guest_start = 0;
    uint32_t guest_length = 0;
    uint64_t last_use_submission = 0;
  };

  static constexpr VkDeviceSize kChunkSize = 4 * 1024 * 1024;
  static constexpr size_t kMaxChunks = 8;
  static constexpr uint32_t kMinCachedIndices = 64;

  bool AllocateInChunks(VkDeviceSize byte_size, size_t& chunk_index_out, VkBuffer& buffer_out,
                        VkDeviceSize& offset_out);
  void RebuildCachedPages();

  const ui::vulkan::VulkanDevice& device_;
  VmaAllocator allocator_;
  memory::Memory& memory_;
  const SubmissionClock& clock_;
  std::vector<Chunk> chunks_;
  std::unordered_map<IndexCacheKey, Entry, IndexCacheKeyHash> entries_;
  std::mutex pages_mutex_;
  std::vector<uint64_t> cached_pages_;
  std::vector<uint64_t> recent_invalidated_pages_;
};

}  // namespace rex::graphics::native
