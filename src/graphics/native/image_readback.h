#pragma once

#include <cstdint>
#include <span>

#include <rex/ui/vulkan/device.h>

namespace rex::graphics::native {

// Host-visible buffer for a tightly packed RGBA8 rectangle. Keep it alive
// until the submission completes.
class ImageReadback {
 public:
  explicit ImageReadback(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~ImageReadback();

  ImageReadback(const ImageReadback&) = delete;
  ImageReadback& operator=(const ImageReadback&) = delete;

  bool Initialize(VkRect2D rectangle);
  // The source image must be in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
  bool Record(VkCommandBuffer commands, VkImage image, VkExtent2D image_extent);
  // Access is available only after the copy's fence signals.
  std::span<const uint8_t> Map(VkFence fence);

 private:
  bool Invalidate();

  const ui::vulkan::VulkanDevice& device_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  uint32_t memory_type_ = UINT32_MAX;
  VkDeviceSize memory_size_ = 0;
  void* mapped_ = nullptr;
  VkRect2D rectangle_{};
  VkDeviceSize size_ = 0;
  bool recorded_ = false;
};

}  // namespace rex::graphics::native
