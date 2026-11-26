#include "shared_vertex_memory.h"

#include <bit>

#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {

SharedVertexMemory::~SharedVertexMemory() {
  const auto& vk = device_.functions();
  vk.vkDestroyBuffer(device_.device(), buffer_, nullptr);
  vk.vkFreeMemory(device_.device(), memory_, nullptr);
  for (VkDeviceMemory granule : granules_) {
    vk.vkFreeMemory(device_.device(), granule, nullptr);
  }
}

bool SharedVertexMemory::Initialize(bool try_sparse) {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  // Transfer source for memexport readback.
  constexpr VkBufferUsageFlags kUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  valid_pages_.assign((kSharedVertexPageCount + 63) / 64, 0);
  if (try_sparse && device_.properties().sparseResidencyBuffer &&
      device_.queue_family_sparse_binding() != UINT32_MAX) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    info.size = kPhysicalMemorySize;
    info.usage = kUsage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk.vkCreateBuffer(device, &info, nullptr, &buffer_) == VK_SUCCESS) {
      VkMemoryRequirements requirements{};
      vk.vkGetBufferMemoryRequirements(device, buffer_, &requirements);
      const uint32_t types = requirements.memoryTypeBits & device_.memory_types().device_local;
      if (types) {
        memory_type_ = uint32_t(std::countr_zero(types));
        granule_size_ = ComputeSparseGranuleSize(requirements.alignment);
        resource_size_ = requirements.size;
        granules_.assign(size_t((resource_size_ + granule_size_ - 1) / granule_size_),
                         VK_NULL_HANDLE);
        REXLOG_INFO("Native Vulkan: sparse shared vertex memory, {} KiB granules",
                    granule_size_ >> 10);
        return true;
      }
      vk.vkDestroyBuffer(device, buffer_, nullptr);
      buffer_ = VK_NULL_HANDLE;
    }
    REXLOG_WARN("Native Vulkan: sparse shared vertex memory unavailable, using a dedicated buffer");
  }
  return ui::vulkan::util::CreateDedicatedAllocationBuffer(
      &device_, kPhysicalMemorySize, kUsage, ui::vulkan::util::MemoryPurpose::kDeviceLocal, buffer_,
      memory_);
}

void SharedVertexMemory::Invalidate(uint32_t start, uint32_t length) {
  if (valid_pages_.empty()) {
    return;
  }
  const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
  ClearPageRangeBits(valid_pages_, first_page, page_count);
}

bool SharedVertexMemory::CommitRange(uint32_t address, uint32_t size) {
  if (granules_.empty()) {
    return true;
  }
  const auto [first, count] = GetSparseGranuleRange(address, size, granule_size_);
  for (uint32_t granule = first; granule < first + count && granule < granules_.size(); ++granule) {
    if (granules_[granule] != VK_NULL_HANDLE) {
      continue;
    }
    const uint64_t offset = uint64_t(granule) * granule_size_;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = std::min(granule_size_, resource_size_ - offset);
    allocation.memoryTypeIndex = memory_type_;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (device_.functions().vkAllocateMemory(device_.device(), &allocation, nullptr, &memory) !=
        VK_SUCCESS) {
      REXLOG_ERROR("Native Vulkan: could not commit {} KiB of shared vertex memory",
                   allocation.allocationSize >> 10);
      return false;
    }
    granules_[granule] = memory;
    pending_binds_.push_back({offset, allocation.allocationSize, memory, 0, 0});
  }
  return true;
}

VkResult SharedVertexMemory::FlushBinds(VkSemaphore semaphore, bool& signaled) {
  signaled = false;
  if (pending_binds_.empty()) {
    return VK_SUCCESS;
  }
  const VkSparseBufferMemoryBindInfo buffer_binds{buffer_, uint32_t(pending_binds_.size()),
                                                  pending_binds_.data()};
  VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
  info.bufferBindCount = 1;
  info.pBufferBinds = &buffer_binds;
  info.signalSemaphoreCount = 1;
  info.pSignalSemaphores = &semaphore;
  VkResult result;
  {
    auto queue = device_.AcquireQueue(device_.queue_family_sparse_binding(), 0);
    result = device_.functions().vkQueueBindSparse(queue.queue(), 1, &info, VK_NULL_HANDLE);
  }
  if (result == VK_SUCCESS) {
    pending_binds_.clear();
    signaled = true;
  }
  return result;
}

}  // namespace rex::graphics::native
