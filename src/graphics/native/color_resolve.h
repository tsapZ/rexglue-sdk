#pragma once

#include <array>
#include <memory>

#include <rex/ui/vulkan/single_layout_descriptor_set_pool.h>

namespace rex::graphics::native {

// Copies one sample (or the average of all samples) of a color render target
// to a layer of a single-sample destination, like the guest's RB_COPY_CONTROL
// sample selection, applying copy_dest_exp_bias.
class ColorResolve {
 public:
  explicit ColorResolve(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~ColorResolve();
  bool Initialize();

  size_t Allocate() { return descriptors_->Allocate(); }
  void Free(size_t descriptor) { descriptors_->Free(descriptor); }
  // sample: negative to average all samples. multisampled: the source has
  // more than one sample.
  void Record(VkCommandBuffer commands, size_t descriptor, VkImageView source,
              VkImageView destination, VkRect2D rectangle, VkOffset2D destination_offset,
              int32_t sample, VkFormat format, bool multisampled, float exp_bias_scale,
              uint32_t destination_layer);

 private:
  const ui::vulkan::VulkanDevice& device_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  std::unique_ptr<ui::vulkan::SingleLayoutDescriptorSetPool> descriptors_;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  // Per destination format: multisampled, then single-sample source.
  std::array<VkPipeline, 10> pipelines_{};
  VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace rex::graphics::native
