#pragma once

#include <array>
#include <memory>

#include <rex/ui/vulkan/single_layout_descriptor_set_pool.h>

namespace rex::graphics::native {

class DepthResolve {
 public:
  explicit DepthResolve(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~DepthResolve();
  bool Initialize();

  size_t Allocate() { return descriptors_->Allocate(); }
  void Free(size_t descriptor) { descriptors_->Free(descriptor); }
  void Record(VkCommandBuffer commands, size_t descriptor, VkImageView source,
              VkImageView destination, VkRect2D rectangle, VkOffset2D destination_offset,
              bool multisample, uint32_t sample, float scale);

 private:
  const ui::vulkan::VulkanDevice& device_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  std::unique_ptr<ui::vulkan::SingleLayoutDescriptorSetPool> descriptors_;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  std::array<VkPipeline, 2> pipelines_{};
  VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace rex::graphics::native
