#include "shared_vertex_memory.h"

#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {

SharedVertexMemory::~SharedVertexMemory() {
  const auto& vk = device_.functions();
  vk.vkDestroyBuffer(device_.device(), buffer_, nullptr);
  vk.vkFreeMemory(device_.device(), memory_, nullptr);
}

bool SharedVertexMemory::Initialize() {
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          &device_, kPhysicalMemorySize,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, buffer_, memory_)) {
    return false;
  }
  valid_pages_.assign((kSharedVertexPageCount + 63) / 64, 0);
  return true;
}

void SharedVertexMemory::Invalidate(uint32_t start, uint32_t length) {
  if (valid_pages_.empty()) {
    return;
  }
  const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
  ClearPageRangeBits(valid_pages_, first_page, page_count);
}

}  // namespace rex::graphics::native
