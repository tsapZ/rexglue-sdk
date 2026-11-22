#include "depth_resolve.h"

#include <rex/ui/vulkan/util.h>

#include "depth_resolve_spirv.h"

namespace rex::graphics::native {
namespace {

struct ResolveConstants {
  VkOffset2D source_offset;
  VkExtent2D extent;
  uint32_t sample;
  float scale;
};

}  // namespace

DepthResolve::~DepthResolve() {
  const auto& vk = device_.functions();
  const auto device = device_.device();
  descriptors_.reset();
  vk.vkDestroySampler(device, sampler_, nullptr);
  for (auto pipeline : pipelines_) {
    vk.vkDestroyPipeline(device, pipeline, nullptr);
  }
  vk.vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
  vk.vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
}

bool DepthResolve::Initialize() {
  const auto& vk = device_.functions();
  const auto device = device_.device();
  constexpr VkDescriptorType sampled = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  constexpr VkDescriptorType storage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  constexpr auto compute = VK_SHADER_STAGE_COMPUTE_BIT;
  constexpr std::array<VkDescriptorSetLayoutBinding, 2> bindings{
      {{0, sampled, 1, compute, nullptr}, {1, storage, 1, compute, nullptr}}};
  VkDescriptorSetLayoutCreateInfo layout{};
  layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout.bindingCount = uint32_t(bindings.size());
  layout.pBindings = bindings.data();
  if (vk.vkCreateDescriptorSetLayout(device, &layout, nullptr, &set_layout_) != VK_SUCCESS) {
    return false;
  }
  constexpr VkPushConstantRange push{compute, 0, sizeof(ResolveConstants)};
  VkPipelineLayoutCreateInfo pipeline_layout{};
  pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout.setLayoutCount = 1;
  pipeline_layout.pSetLayouts = &set_layout_;
  pipeline_layout.pushConstantRangeCount = 1;
  pipeline_layout.pPushConstantRanges = &push;
  if (vk.vkCreatePipelineLayout(device, &pipeline_layout, nullptr, &pipeline_layout_) !=
      VK_SUCCESS) {
    return false;
  }
  pipelines_[0] = ui::vulkan::util::CreateComputePipeline(&device_, pipeline_layout_, kDepthResolve,
                                                          sizeof(kDepthResolve));
  pipelines_[1] = ui::vulkan::util::CreateComputePipeline(
      &device_, pipeline_layout_, kDepthResolveMsaa, sizeof(kDepthResolveMsaa));
  VkSamplerCreateInfo sampler{};
  sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  if (!pipelines_[0] || !pipelines_[1] ||
      vk.vkCreateSampler(device, &sampler, nullptr, &sampler_) != VK_SUCCESS) {
    return false;
  }
  constexpr std::array<VkDescriptorPoolSize, 2> sizes{{{sampled, 1}, {storage, 1}}};
  descriptors_ = std::make_unique<ui::vulkan::SingleLayoutDescriptorSetPool>(
      &device_, 32, uint32_t(sizes.size()), sizes.data(), set_layout_);
  return true;
}

void DepthResolve::Record(VkCommandBuffer commands, size_t descriptor, VkImageView source,
                          VkImageView destination, VkRect2D rectangle, bool multisample,
                          uint32_t sample, float scale) {
  const auto& vk = device_.functions();
  const auto set = descriptors_->Get(descriptor);
  const VkDescriptorImageInfo src{sampler_, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  const VkDescriptorImageInfo dst{VK_NULL_HANDLE, destination, VK_IMAGE_LAYOUT_GENERAL};
  const std::array<VkWriteDescriptorSet, 2> writes{{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &src, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dst, nullptr, nullptr},
  }};
  vk.vkUpdateDescriptorSets(device_.device(), uint32_t(writes.size()), writes.data(), 0, nullptr);
  vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[multisample]);
  vk.vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set,
                             0, nullptr);
  const ResolveConstants constants{rectangle.offset, rectangle.extent, sample, scale};
  vk.vkCmdPushConstants(commands, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(constants), &constants);
  vk.vkCmdDispatch(commands, (rectangle.extent.width + 7) / 8, (rectangle.extent.height + 7) / 8,
                   1);
}

}  // namespace rex::graphics::native
