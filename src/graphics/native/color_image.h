#pragma once

#include <utility>

#include <rex/ui/vulkan/device.h>

namespace rex::graphics::native {

// A host image representing an EDRAM attachment or resolved texture.
class ColorImage {
 public:
  explicit ColorImage(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~ColorImage();

  ColorImage(const ColorImage&) = delete;
  ColorImage& operator=(const ColorImage&) = delete;

  bool Initialize(VkExtent2D extent, VkFormat format = kFormat,
                  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                  VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT);
  void Transition(VkCommandBuffer commands, VkImageLayout layout, VkPipelineStageFlags stage,
                  VkAccessFlags access);
  void Clear(VkCommandBuffer commands, const VkClearColorValue& color);

  VkImageAspectFlags aspects() const { return aspects_; }
  VkSampleCountFlagBits samples() const { return samples_; }
  VkFormat format() const { return format_; }
  VkImage image() const { return image_; }
  VkImageView view() const { return view_; }
  VkImageView depth_view() const { return depth_view_; }
  VkExtent2D extent() const { return extent_; }
  void MarkResolved() { unpublished_resolve_ = true; }
  bool TakeResolvedOutput() { return std::exchange(unpublished_resolve_, false); }
  bool initialized() const { return layout_ != VK_IMAGE_LAYOUT_UNDEFINED; }

  static constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

 private:
  const ui::vulkan::VulkanDevice& device_;
  VkExtent2D extent_{};
  VkFormat format_ = kFormat;
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkImageAspectFlags aspects_ = VK_IMAGE_ASPECT_COLOR_BIT;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkImageView depth_view_ = VK_NULL_HANDLE;
  VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  VkPipelineStageFlags stage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkAccessFlags access_ = 0;
  bool unpublished_resolve_ = false;
};

}  // namespace rex::graphics::native
