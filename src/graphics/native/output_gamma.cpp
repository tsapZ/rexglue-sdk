#include "output_gamma.h"

#include <array>
#include <cstring>

#include <rex/ui/vulkan/util.h>

#include "output_gamma_spirv.h"

namespace rex::graphics::native {
namespace {

constexpr size_t kTableBytes = 128 * 3 * sizeof(uint32_t);
constexpr uint32_t kWorkgroupWidth = 16;
constexpr uint32_t kWorkgroupHeight = 8;

}  // namespace

OutputGamma::~OutputGamma() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  vk.vkDestroyPipeline(device, pipeline_, nullptr);
  vk.vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
  vk.vkDestroyDescriptorPool(device, descriptors_, nullptr);
  vk.vkDestroySampler(device, sampler_, nullptr);
  vk.vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
  if (lookup_data_) {
    vk.vkUnmapMemory(device, lookup_memory_);
  }
  vk.vkDestroyBuffer(device, lookup_, nullptr);
  vk.vkFreeMemory(device, lookup_memory_, nullptr);
}

bool OutputGamma::Initialize() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  constexpr std::array<VkDescriptorSetLayoutBinding, 3> bindings{{
      {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
      {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
  }};
  VkDescriptorSetLayoutCreateInfo set_layout{};
  set_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_layout.bindingCount = uint32_t(bindings.size());
  set_layout.pBindings = bindings.data();
  if (vk.vkCreateDescriptorSetLayout(device, &set_layout, nullptr, &set_layout_) != VK_SUCCESS) {
    return false;
  }

  constexpr VkPushConstantRange mode{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo pipeline_layout{};
  pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout.setLayoutCount = 1;
  pipeline_layout.pSetLayouts = &set_layout_;
  pipeline_layout.pushConstantRangeCount = 1;
  pipeline_layout.pPushConstantRanges = &mode;
  if (vk.vkCreatePipelineLayout(device, &pipeline_layout, nullptr, &pipeline_layout_) !=
      VK_SUCCESS) {
    return false;
  }
  pipeline_ = ui::vulkan::util::CreateComputePipeline(&device_, pipeline_layout_, kOutputGamma,
                                                      sizeof(kOutputGamma));
  if (!pipeline_ ||
      !ui::vulkan::util::CreateDedicatedAllocationBuffer(
          &device_, kTableBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          ui::vulkan::util::MemoryPurpose::kUpload, lookup_, lookup_memory_) ||
      vk.vkMapMemory(device, lookup_memory_, 0, VK_WHOLE_SIZE, 0, &lookup_data_) != VK_SUCCESS) {
    return false;
  }

  constexpr std::array<VkDescriptorPoolSize, 3> sizes{{
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
  }};
  VkDescriptorPoolCreateInfo pool{};
  pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool.maxSets = 1;
  pool.poolSizeCount = uint32_t(sizes.size());
  pool.pPoolSizes = sizes.data();
  if (vk.vkCreateDescriptorPool(device, &pool, nullptr, &descriptors_) != VK_SUCCESS) {
    return false;
  }
  VkDescriptorSetAllocateInfo set{};
  set.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set.descriptorPool = descriptors_;
  set.descriptorSetCount = 1;
  set.pSetLayouts = &set_layout_;
  if (vk.vkAllocateDescriptorSets(device, &set, &set_) != VK_SUCCESS) {
    return false;
  }
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.addressModeU = sampler_info.addressModeV = sampler_info.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  if (vk.vkCreateSampler(device, &sampler_info, nullptr, &sampler_) != VK_SUCCESS) {
    return false;
  }
  const VkDescriptorBufferInfo buffer{lookup_, 0, kTableBytes};
  const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_,   1, 0, 1,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,      nullptr, &buffer};
  vk.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  return true;
}

bool OutputGamma::Record(VkCommandBuffer commands, VkImageView source, VkImageView output,
                         VkExtent2D extent, std::span<const std::byte> register_values, bool pwl) {
  if (register_values.size() != (pwl ? kTableBytes : 256 * sizeof(uint32_t))) {
    return false;
  }

  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  std::memcpy(lookup_data_, register_values.data(), register_values.size());
  VkMappedMemoryRange memory{};
  memory.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memory.memory = lookup_memory_;
  memory.size = VK_WHOLE_SIZE;
  if (vk.vkFlushMappedMemoryRanges(device, 1, &memory) != VK_SUCCESS) {
    return false;
  }

  const VkDescriptorImageInfo image{VK_NULL_HANDLE, output, VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo source_info{sampler_, source,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  const std::array<VkWriteDescriptorSet, 2> writes{{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 0, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &image},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 2, 0, 1,
       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &source_info},
  }};
  vk.vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);
  vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vk.vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                             &set_, 0, nullptr);
  const uint32_t mode = pwl ? 1 : 0;
  vk.vkCmdPushConstants(commands, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mode),
                        &mode);
  vk.vkCmdDispatch(commands, (extent.width + kWorkgroupWidth - 1) / kWorkgroupWidth,
                   (extent.height + kWorkgroupHeight - 1) / kWorkgroupHeight, 1);
  return true;
}

}  // namespace rex::graphics::native
