#include "color_image.h"

#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {

ColorImage::~ColorImage() {
  const auto& vk = device_.functions();
  vk.vkDestroyImageView(device_.device(), view_, nullptr);
  vk.vkDestroyImageView(device_.device(), depth_view_, nullptr);
  vk.vkDestroyImage(device_.device(), image_, nullptr);
  vk.vkFreeMemory(device_.device(), memory_, nullptr);
}

bool ColorImage::Initialize(VkExtent2D extent, VkFormat format, VkSampleCountFlagBits samples,
                            VkImageAspectFlags aspects) {
  const auto& properties = device_.properties();
  if (!extent.width || !extent.height || extent.width > properties.maxImageDimension2D ||
      extent.height > properties.maxImageDimension2D) {
    return false;
  }
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {extent.width, extent.height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = samples;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
    info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_SAMPLED_BIT;
  } else {
    info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (format == VK_FORMAT_R32_SFLOAT) {
    info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          &device_, info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, image_, memory_)) {
    return false;
  }
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view_info.format = format;
  view_info.subresourceRange = {aspects, 0, 1, 0, 1};
  if (device_.functions().vkCreateImageView(device_.device(), &view_info, nullptr, &view_) !=
      VK_SUCCESS) {
    return false;
  }
  if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (device_.functions().vkCreateImageView(device_.device(), &view_info, nullptr,
                                              &depth_view_) != VK_SUCCESS) {
      return false;
    }
  } else if (format == VK_FORMAT_R32_SFLOAT) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    if (device_.functions().vkCreateImageView(device_.device(), &view_info, nullptr,
                                              &depth_view_) != VK_SUCCESS) {
      return false;
    }
  }
  extent_ = extent;
  format_ = format;
  samples_ = samples;
  aspects_ = aspects;
  return true;
}

void ColorImage::Transition(VkCommandBuffer commands, VkImageLayout layout,
                            VkPipelineStageFlags stage, VkAccessFlags access) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = access_;
  barrier.dstAccessMask = access;
  barrier.oldLayout = layout_;
  barrier.newLayout = layout;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image_;
  barrier.subresourceRange = {aspects_, 0, 1, 0, 1};
  device_.functions().vkCmdPipelineBarrier(commands, stage_, stage, 0, 0, nullptr, 0, nullptr, 1,
                                           &barrier);
  access_ = access;
  layout_ = layout;
  stage_ = stage;
}

void ColorImage::Clear(VkCommandBuffer commands, const VkClearColorValue& color) {
  Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_ACCESS_TRANSFER_WRITE_BIT);
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  device_.functions().vkCmdClearColorImage(commands, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           &color, 1, &range);
}

}  // namespace rex::graphics::native
