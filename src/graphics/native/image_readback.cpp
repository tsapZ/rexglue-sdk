#include "image_readback.h"

#include <algorithm>

#include <rex/logging.h>
#include <rex/math.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {

ImageReadback::~ImageReadback() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  if (mapped_) {
    vk.vkUnmapMemory(device, memory_);
  }
  vk.vkDestroyBuffer(device, buffer_, nullptr);
  vk.vkFreeMemory(device, memory_, nullptr);
}

bool ImageReadback::Initialize(VkRect2D rectangle) {
  if (buffer_) {
    REXLOG_ERROR("Native Vulkan readback: already initialized");
    return false;
  }
  const auto& properties = device_.properties();
  if (!rectangle.extent.width || !rectangle.extent.height || rectangle.offset.x < 0 ||
      rectangle.offset.y < 0 || rectangle.extent.width > properties.maxImageDimension2D ||
      rectangle.extent.height > properties.maxImageDimension2D) {
    REXLOG_ERROR("Native Vulkan readback: invalid rectangle {}x{}+{},{}", rectangle.extent.width,
                 rectangle.extent.height, rectangle.offset.x, rectangle.offset.y);
    return false;
  }
  const VkDeviceSize row_pitch = VkDeviceSize(rectangle.extent.width) * 4;
  const VkDeviceSize size = row_pitch * rectangle.extent.height;
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          &device_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kReadback, buffer_, memory_, &memory_type_,
          &memory_size_)) {
    REXLOG_ERROR("Native Vulkan readback: could not allocate {} bytes for {}x{}", size,
                 rectangle.extent.width, rectangle.extent.height);
    return false;
  }
  if (device_.functions().vkMapMemory(device_.device(), memory_, 0, VK_WHOLE_SIZE, 0, &mapped_) !=
      VK_SUCCESS) {
    REXLOG_ERROR("Native Vulkan readback: could not map {} bytes", size);
    device_.functions().vkDestroyBuffer(device_.device(), buffer_, nullptr);
    device_.functions().vkFreeMemory(device_.device(), memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    return false;
  }
  rectangle_ = rectangle;
  size_ = size;
  return true;
}

bool ImageReadback::Record(VkCommandBuffer commands, VkImage image, VkExtent2D image_extent) {
  if (!buffer_ || !mapped_) {
    REXLOG_ERROR("Native Vulkan readback: copy recorded before allocation");
    return false;
  }
  if (!commands || !image) {
    REXLOG_ERROR("Native Vulkan readback: missing command buffer or image");
    return false;
  }
  if (uint64_t(rectangle_.offset.x) + rectangle_.extent.width > image_extent.width ||
      uint64_t(rectangle_.offset.y) + rectangle_.extent.height > image_extent.height) {
    REXLOG_ERROR("Native Vulkan readback: rectangle {}x{}+{},{} does not fit in image {}x{}",
                 rectangle_.extent.width, rectangle_.extent.height, rectangle_.offset.x,
                 rectangle_.offset.y, image_extent.width, image_extent.height);
    return false;
  }

  const auto& vk = device_.functions();
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageOffset = {rectangle_.offset.x, rectangle_.offset.y, 0};
  copy.imageExtent = {rectangle_.extent.width, rectangle_.extent.height, 1};
  vk.vkCmdCopyImageToBuffer(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer_, 1,
                            &copy);

  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer_;
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;
  vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                          0, nullptr, 1, &barrier, 0, nullptr);
  recorded_ = true;
  return true;
}

std::span<const uint8_t> ImageReadback::Map(VkFence fence) {
  if (!recorded_ || !mapped_ || !fence) {
    return {};
  }
  if (device_.functions().vkGetFenceStatus(device_.device(), fence) != VK_SUCCESS) {
    return {};
  }
  if (!Invalidate()) {
    return {};
  }
  return {static_cast<const uint8_t*>(mapped_), size_t(size_)};
}

bool ImageReadback::Invalidate() {
  if (device_.memory_types().host_coherent & (uint32_t(1) << memory_type_)) {
    return true;
  }
  const VkDeviceSize atom_size = device_.properties().nonCoherentAtomSize;
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = memory_;
  range.offset = 0;
  range.size = std::min(rex::round_up(size_, atom_size), memory_size_);
  if (device_.functions().vkInvalidateMappedMemoryRanges(device_.device(), 1, &range) !=
      VK_SUCCESS) {
    REXLOG_ERROR("Native Vulkan readback: could not invalidate {} bytes", size_);
    return false;
  }
  return true;
}

}  // namespace rex::graphics::native
