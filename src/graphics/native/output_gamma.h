#pragma once

#include <span>

#include <rex/ui/vulkan/device.h>

namespace rex::graphics::native {

class OutputGamma {
 public:
  explicit OutputGamma(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~OutputGamma();

  bool Initialize();
  bool Record(VkCommandBuffer commands, VkImageView source, VkImageView output, VkExtent2D extent,
              std::span<const std::byte> register_values, bool pwl);

 private:
  const ui::vulkan::VulkanDevice& device_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkBuffer lookup_ = VK_NULL_HANDLE;
  VkDeviceMemory lookup_memory_ = VK_NULL_HANDLE;
  void* lookup_data_ = nullptr;
};

}  // namespace rex::graphics::native
