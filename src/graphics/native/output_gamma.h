#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <rex/ui/vulkan/device.h>

namespace rex::graphics::native {

class OutputGamma {
 public:
  explicit OutputGamma(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~OutputGamma();

  // One descriptor set and lookup slice per frame in flight, so recording a frame
  // never rewrites what the GPU may still read for the previous one.
  static constexpr uint32_t kSlotCount = 2;

  bool Initialize();
  bool Record(VkCommandBuffer commands, uint32_t slot, VkImageView source, VkImageView output,
              VkExtent2D extent, std::span<const std::byte> register_values, bool pwl);

 private:
  const ui::vulkan::VulkanDevice& device_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, kSlotCount> sets_{};
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkBuffer lookup_ = VK_NULL_HANDLE;
  VkDeviceMemory lookup_memory_ = VK_NULL_HANDLE;
  void* lookup_data_ = nullptr;
  VkDeviceSize lookup_stride_ = 0;
};

}  // namespace rex::graphics::native
