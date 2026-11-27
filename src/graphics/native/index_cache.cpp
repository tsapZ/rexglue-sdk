#include "index_cache.h"

#include <algorithm>
#include <cstring>

#include <rex/math.h>
#include <rex/ui/vulkan/util.h>

#include "draw_pass.h"

namespace rex::graphics::native {

NativeIndexCache::~NativeIndexCache() {
  entries_.clear();
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  for (const auto& chunk : chunks_) {
    if (chunk.allocation != VK_NULL_HANDLE) {
      vmaDestroyBuffer(allocator_, chunk.buffer, chunk.allocation);
    } else {
      vk.vkDestroyBuffer(device, chunk.buffer, nullptr);
      vk.vkFreeMemory(device, chunk.memory, nullptr);
    }
  }
  chunks_.clear();
}

bool NativeIndexCache::Initialize(bool watch_pages) {
  if (watch_pages) {
    const size_t words = (kSharedVertexPageCount + 63) / 64;
    cached_pages_.assign(words, 0);
    scratch_pages_.assign(words, 0);
    recent_invalidated_pages_.assign(words, 0);
  }
  return true;
}

void NativeIndexCache::InvalidateAll() {
  entries_.clear();
  for (auto& chunk : chunks_) {
    chunk.used = 0;
  }
  for (auto& w : cached_pages_) {
    std::atomic_ref<uint64_t>(w).store(0, std::memory_order_relaxed);
  }
  std::fill(recent_invalidated_pages_.begin(), recent_invalidated_pages_.end(), 0);
}

void NativeIndexCache::ClearRecentInvalidations() {
  std::fill(recent_invalidated_pages_.begin(), recent_invalidated_pages_.end(), 0);
}

std::pair<uint32_t, uint32_t> NativeIndexCache::WidenInvalidation(uint32_t start, uint32_t length) {
  if (cached_pages_.empty()) {
    return {start, 0};
  }
  return WidenInvalidationPageRangeWithMask(
      [this](uint32_t b) { return ProtectedPagesWord(b); }, uint32_t(cached_pages_.size()), start,
      length);
}

void NativeIndexCache::RebuildCachedPages() {
  if (cached_pages_.empty()) {
    return;
  }
  std::fill(scratch_pages_.begin(), scratch_pages_.end(), 0);
  for (const auto& [key, entry] : entries_) {
    const auto [first, count] = GetPhysicalPageRange(entry.guest_start, entry.guest_length);
    SetPageRangeBits(scratch_pages_, first, count);
  }
  for (size_t i = 0; i < cached_pages_.size(); ++i) {
    std::atomic_ref<uint64_t>(cached_pages_[i]).store(scratch_pages_[i], std::memory_order_relaxed);
  }
}

void NativeIndexCache::Invalidate(uint32_t start, uint32_t length) {
  const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
  if (!recent_invalidated_pages_.empty()) {
    SetPageRangeBits(recent_invalidated_pages_, first_page, page_count);
  }
  if (!cached_pages_.empty() && !AtomicHasAnyPageBitSet(cached_pages_, first_page, page_count)) {
    return;
  }
  bool erased = false;
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (RangesOverlap(start, length, it->second.guest_start, it->second.guest_length)) {
      it = entries_.erase(it);
      erased = true;
    } else {
      ++it;
    }
  }
  if (erased) {
    RebuildCachedPages();
  }
}

bool NativeIndexCache::AllocateInChunks(VkDeviceSize byte_size, size_t& chunk_index_out,
                                        VkBuffer& buffer_out, VkDeviceSize& offset_out) {
  const VkDeviceSize aligned_size = rex::align(byte_size, VkDeviceSize(4));
  for (size_t i = 0; i < chunks_.size(); ++i) {
    auto& chunk = chunks_[i];
    if (chunk.size - chunk.used >= aligned_size) {
      chunk_index_out = i;
      buffer_out = chunk.buffer;
      offset_out = chunk.used;
      chunk.used += aligned_size;
      chunk.last_use_submission = clock_.next;
      return true;
    }
  }
  if (chunks_.size() < kMaxChunks) {
    Chunk chunk{};
    chunk.size = std::max(kChunkSize, aligned_size);
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = chunk.size;
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bool created = false;
    if (allocator_ != VK_NULL_HANDLE) {
      VmaAllocationCreateInfo alloc_info{};
      alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      created = vmaCreateBuffer(allocator_, &info, &alloc_info, &chunk.buffer, &chunk.allocation,
                                nullptr) == VK_SUCCESS;
    } else {
      created = ui::vulkan::util::CreateDedicatedAllocationBuffer(
          &device_, chunk.size, info.usage, ui::vulkan::util::MemoryPurpose::kDeviceLocal,
          chunk.buffer, chunk.memory);
    }
    if (!created) {
      return false;
    }
    chunk.used = aligned_size;
    chunk.last_use_submission = clock_.next;
    chunk_index_out = chunks_.size();
    buffer_out = chunk.buffer;
    offset_out = 0;
    chunks_.push_back(chunk);
    return true;
  }
  // Recycle a retired chunk if all chunks are full and one is no longer in flight.
  for (size_t i = 0; i < chunks_.size(); ++i) {
    auto& chunk = chunks_[i];
    if (chunk.last_use_submission <= clock_.retired && chunk.size >= aligned_size) {
      bool erased = false;
      for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.chunk_index == i) {
          it = entries_.erase(it);
          erased = true;
        } else {
          ++it;
        }
      }
      if (erased) {
        RebuildCachedPages();
      }
      chunk.used = aligned_size;
      chunk.last_use_submission = clock_.next;
      chunk_index_out = i;
      buffer_out = chunk.buffer;
      offset_out = 0;
      return true;
    }
  }
  return false;
}

bool NativeIndexCache::GetOrCreate(const IndexCacheKey& key, std::span<const uint8_t> guest_bytes,
                                   ui::vulkan::VulkanUploadBufferPool& uploads,
                                   IndexBinding& binding_out, CachedIndexUpload& upload_out,
                                   bool& uploaded_out) {
  uploaded_out = false;
  if (auto found = entries_.find(key); found != entries_.end()) {
    found->second.last_use_submission = clock_.next;
    chunks_[found->second.chunk_index].last_use_submission = clock_.next;
    binding_out = {true, found->second.buffer, found->second.offset,
                   key.use_uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32, key.index_count};
    return true;
  }
  if (cached_pages_.empty() || key.index_count < kMinCachedIndices || guest_bytes.empty()) {
    return false;
  }
  const uint32_t guest_length = uint32_t(guest_bytes.size());
  const auto [first_page, page_count] = GetPhysicalPageRange(key.guest_base, guest_length);
  if (HasAnyPageBitSet(recent_invalidated_pages_, first_page, page_count)) {
    return false;
  }
  const VkDeviceSize host_stride = key.use_uint16 ? sizeof(uint16_t) : sizeof(uint32_t);
  const VkDeviceSize host_bytes = VkDeviceSize(key.index_count) * host_stride;
  if (host_bytes > kDrawUploadPageSize) {
    return false;
  }
  VkBuffer src_buffer = VK_NULL_HANDLE;
  VkDeviceSize src_offset = 0;
  uint8_t* staging = uploads.Request(clock_.next, host_bytes, 4, src_buffer, src_offset);
  if (!staging) {
    return false;
  }
  size_t chunk_index = 0;
  VkBuffer dst_buffer = VK_NULL_HANDLE;
  VkDeviceSize dst_offset = 0;
  if (!AllocateInChunks(host_bytes, chunk_index, dst_buffer, dst_offset)) {
    return false;
  }
  if (watch_range_) {
    watch_range_(key.guest_base, guest_length);
  } else {
    memory_.EnablePhysicalMemoryAccessCallbacks(key.guest_base, guest_length, true, false);
  }
  if (key.use_uint16) {
    DecodeDmaIndices16(guest_bytes, key.endianness, reinterpret_cast<uint16_t*>(staging),
                       key.index_count);
  } else {
    DecodeDmaIndices32(guest_bytes, key.format, key.endianness,
                       reinterpret_cast<uint32_t*>(staging), key.index_count, key.primitive_restart,
                       key.reset_index);
  }
  AtomicSetPageRangeBits(cached_pages_, first_page, page_count);
  entries_.emplace(key, Entry{chunk_index, dst_buffer, dst_offset, key.guest_base, guest_length,
                              clock_.next});
  binding_out = {true, dst_buffer, dst_offset,
                 key.use_uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32, key.index_count};
  upload_out = {src_buffer, src_offset, dst_buffer, dst_offset, host_bytes};
  uploaded_out = true;
  return true;
}

}  // namespace rex::graphics::native
