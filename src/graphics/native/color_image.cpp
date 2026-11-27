#include "color_image.h"

#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {

bool CreateDeviceImage(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                       const VkImageCreateInfo& info, VkImage& image, VkDeviceMemory& memory,
                       VmaAllocation& allocation) {
  image = VK_NULL_HANDLE;
  memory = VK_NULL_HANDLE;
  allocation = VK_NULL_HANDLE;
  if (allocator == VK_NULL_HANDLE) {
    return ui::vulkan::util::CreateDedicatedAllocationImage(
        &device, info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, image, memory);
  }
  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (vmaCreateImage(allocator, &info, &allocation_info, &image, &allocation, nullptr) !=
      VK_SUCCESS) {
    image = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

void DestroyDeviceImage(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                        VkImage image, VkDeviceMemory memory, VmaAllocation allocation) {
  if (allocation != VK_NULL_HANDLE) {
    vmaDestroyImage(allocator, image, allocation);
    return;
  }
  device.functions().vkDestroyImage(device.device(), image, nullptr);
  device.functions().vkFreeMemory(device.device(), memory, nullptr);
}

ColorImage::~ColorImage() {
  if (view_purger_ && view_purger_->purge_views) {
    std::vector<VkImageView> dead_views;
    dead_views.reserve(sampled_views_.size() + 2);
    if (view_ != VK_NULL_HANDLE) {
      dead_views.push_back(view_);
    }
    if (depth_view_ != VK_NULL_HANDLE) {
      dead_views.push_back(depth_view_);
    }
    for (const auto& entry : sampled_views_) {
      if (entry.second != VK_NULL_HANDLE) {
        dead_views.push_back(entry.second);
      }
    }
    view_purger_->PurgeImageViews(dead_views);
  }
  const auto& vk = device_.functions();
  for (const auto& entry : sampled_views_) {
    vk.vkDestroyImageView(device_.device(), entry.second, nullptr);
  }
  vk.vkDestroyImageView(device_.device(), view_, nullptr);
  vk.vkDestroyImageView(device_.device(), depth_view_, nullptr);
  DestroyDeviceImage(device_, allocator_, image_, memory_, allocation_);
}

VkImageView ColorImage::GetSampledView(VkComponentMapping swizzle) {
  const uint32_t key = PackTextureComponentMapping(swizzle);
  if (has_last_sampled_view_ && last_sampled_key_ == key) {
    return last_sampled_view_;
  }
  auto found = sampled_views_.find(key);
  if (found != sampled_views_.end()) {
    last_sampled_key_ = key;
    last_sampled_view_ = found->second;
    has_last_sampled_view_ = true;
    return found->second;
  }
  VkImageViewCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.image = image_;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  info.format = format_;
  info.components = swizzle;
  info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, array_layers_};
  VkImageView view = VK_NULL_HANDLE;
  if (device_.functions().vkCreateImageView(device_.device(), &info, nullptr, &view) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  sampled_views_.emplace(key, view);
  last_sampled_key_ = key;
  last_sampled_view_ = view;
  has_last_sampled_view_ = true;
  return view;
}

bool ColorImage::Initialize(VkExtent2D extent, VkFormat format, VkSampleCountFlagBits samples,
                            VkImageAspectFlags aspects, uint32_t array_layers,
                            bool allow_storage) {
  const auto& properties = device_.properties();
  if (!extent.width || !extent.height || extent.width > properties.maxImageDimension2D ||
      extent.height > properties.maxImageDimension2D || !array_layers ||
      array_layers > properties.maxImageArrayLayers) {
    return false;
  }
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {extent.width, extent.height, 1};
  info.mipLevels = 1;
  info.arrayLayers = array_layers;
  info.samples = samples;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  // Compute resolves write single-sample color images as storage images.
  // Disabling STORAGE_BIT on render targets and non-storage resolves preserves DCC compression.
  bool storage_supported = false;
  if (allow_storage && (aspects & VK_IMAGE_ASPECT_COLOR_BIT) &&
      samples == VK_SAMPLE_COUNT_1_BIT) {
    VkFormatProperties format_properties{};
    device_.vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
        device_.physical_device(), format, &format_properties);
    storage_supported =
        (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
  }
  storage_enabled_ = storage_supported;
  info.usage = GetColorImageUsage(aspects, storage_supported);
  if (!CreateDeviceImage(device_, allocator_, info, image_, memory_, allocation_)) {
    return false;
  }
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view_info.format = format;
  view_info.subresourceRange = {aspects, 0, 1, 0, array_layers};
  if (device_.functions().vkCreateImageView(device_.device(), &view_info, nullptr, &view_) !=
      VK_SUCCESS) {
    return false;
  }
  view_info.subresourceRange.layerCount = 1;
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
  array_layers_ = array_layers;
  format_ = format;
  samples_ = samples;
  aspects_ = aspects;
  return true;
}

bool ColorImage::PrepareTransition(VkImageLayout layout, VkPipelineStageFlags stage,
                                   VkAccessFlags access, VkImageMemoryBarrier& barrier,
                                   VkPipelineStageFlags& src_stage_out) {
  if (layout_ == layout &&
      ((access_ == VK_ACCESS_SHADER_READ_BIT && access == VK_ACCESS_SHADER_READ_BIT) ||
       (layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
        access_ == VK_ACCESS_TRANSFER_READ_BIT && access == VK_ACCESS_TRANSFER_READ_BIT)) &&
      (stage_ & stage) == stage) {
    return false;
  }
  barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = access_;
  barrier.dstAccessMask = access;
  barrier.oldLayout = layout_;
  barrier.newLayout = layout;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image_;
  barrier.subresourceRange = {aspects_, 0, 1, 0, array_layers_};
  src_stage_out = stage_;
  access_ = access;
  layout_ = layout;
  stage_ = stage;
  return true;
}

void ColorImage::Transition(VkCommandBuffer commands, VkImageLayout layout,
                            VkPipelineStageFlags stage, VkAccessFlags access) {
  VkImageMemoryBarrier barrier{};
  VkPipelineStageFlags src_stage = 0;
  if (!PrepareTransition(layout, stage, access, barrier, src_stage)) {
    return;
  }
  device_.functions().vkCmdPipelineBarrier(commands, src_stage, stage, 0, 0, nullptr, 0, nullptr, 1,
                                           &barrier);
}

void ColorImage::Clear(VkCommandBuffer commands, const VkClearColorValue& color) {
  Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_ACCESS_TRANSFER_WRITE_BIT);
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, array_layers_};
  device_.functions().vkCmdClearColorImage(commands, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           &color, 1, &range);
}

}  // namespace rex::graphics::native
