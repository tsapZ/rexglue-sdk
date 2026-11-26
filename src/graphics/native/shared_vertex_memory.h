#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <rex/ui/vulkan/device.h>

#include "draw_input.h"

namespace rex::graphics::native {

// The guest physical memory mirrored in one device-local storage buffer that
// vertex fetches read, filled page by page on demand and invalidated when the
// guest writes the pages.
class SharedVertexMemory {
 public:
  explicit SharedVertexMemory(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~SharedVertexMemory();
  SharedVertexMemory(const SharedVertexMemory&) = delete;
  SharedVertexMemory& operator=(const SharedVertexMemory&) = delete;

  bool Initialize();
  VkBuffer buffer() const { return buffer_; }
  void Invalidate(uint32_t start, uint32_t length);
  void InvalidateAll() { std::fill(valid_pages_.begin(), valid_pages_.end(), 0); }
  // Calls upload(address, size) for every run of [address, address + size)
  // not valid on the GPU yet, at most max_chunk_size bytes at a time, and marks
  // each chunk valid once upload returns true.
  template <typename Upload>
  bool EnsureRange(uint32_t address, uint32_t size, uint32_t max_chunk_size, Upload&& upload) {
    const auto [first_page, page_count] = GetPhysicalPageRange(address, size);
    const uint32_t end_page = first_page + page_count;
    const uint32_t max_chunk_pages = max_chunk_size >> kSharedVertexPageShift;
    uint32_t cursor = first_page;
    uint32_t run_first_page = 0;
    uint32_t run_page_count = 0;
    while (FindNextInvalidPageRun(valid_pages_, cursor, end_page, run_first_page, run_page_count)) {
      uint32_t chunk_page = run_first_page;
      uint32_t remaining_pages = run_page_count;
      while (remaining_pages) {
        const uint32_t chunk_pages = std::min(remaining_pages, max_chunk_pages);
        const uint32_t chunk_addr = chunk_page << kSharedVertexPageShift;
        const uint32_t chunk_size =
            std::min(chunk_pages << kSharedVertexPageShift, kPhysicalMemorySize - chunk_addr);
        if (!upload(chunk_addr, chunk_size)) {
          return false;
        }
        SetPageRangeBits(valid_pages_, chunk_page, chunk_pages);
        chunk_page += chunk_pages;
        remaining_pages -= chunk_pages;
      }
    }
    return true;
  }

 private:
  const ui::vulkan::VulkanDevice& device_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  std::vector<uint64_t> valid_pages_;
};

}  // namespace rex::graphics::native
