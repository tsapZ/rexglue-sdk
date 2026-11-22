#include "draw_pass.h"

#include <algorithm>
#include <cstring>

#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::native {
namespace {

using Translator = SpirvShaderTranslator;

VkShaderModule CreateShaderModule(const ui::vulkan::VulkanDevice& device,
                                  const Shader::Translation& translation) {
  const auto& binary = translation.translated_binary();
  if (binary.empty() || binary.size() % sizeof(uint32_t)) {
    return VK_NULL_HANDLE;
  }
  std::vector<uint32_t> code(binary.size() / sizeof(uint32_t));
  std::memcpy(code.data(), binary.data(), binary.size());
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = binary.size();
  info.pCode = code.data();
  VkShaderModule module = VK_NULL_HANDLE;
  return device.functions().vkCreateShaderModule(device.device(), &info, nullptr, &module) ==
                 VK_SUCCESS
             ? module
             : VK_NULL_HANDLE;
}

}  // namespace

DrawBindings::~DrawBindings() {
  const auto& vk = device.functions();
  for (size_t i = 0; i < layouts.size(); ++i) {
    pools[i].reset();
    vk.vkDestroyDescriptorSetLayout(device.device(), layouts[i], nullptr);
  }
  vk.vkDestroyPipelineLayout(device.device(), pipeline_layout, nullptr);
}

bool DrawBindings::Initialize(uint32_t shared_count, uint32_t image_count, uint32_t sampler_count) {
  const auto& vk = device.functions();
  const auto handle = device.device();
  std::array<VkDescriptorSetLayoutBinding, Translator::kConstantBufferCount> constants{};
  for (uint32_t i = 0; i < constants.size(); ++i) {
    constants[i] = {i, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  }
  const VkDescriptorSetLayoutBinding shared{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shared_count,
                                            VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  std::vector<VkDescriptorSetLayoutBinding> textures(image_count + sampler_count);
  for (uint32_t i = 0; i < textures.size(); ++i) {
    textures[i] = {i,
                   i < image_count ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
                   1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  }
  const std::array<std::span<const VkDescriptorSetLayoutBinding>, 4> bindings{
      std::span(&shared, 1), constants, {}, textures};
  for (size_t i = 0; i < layouts.size(); ++i) {
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = uint32_t(bindings[i].size());
    info.pBindings = bindings[i].data();
    if (vk.vkCreateDescriptorSetLayout(handle, &info, nullptr, &layouts[i]) != VK_SUCCESS) {
      return false;
    }
    std::vector<VkDescriptorPoolSize> sizes;
    for (const auto& binding : bindings[i]) {
      if (!sizes.empty() && sizes.back().type == binding.descriptorType) {
        sizes.back().descriptorCount += binding.descriptorCount;
      } else {
        sizes.push_back({binding.descriptorType, binding.descriptorCount});
      }
    }
    pools[i] = std::make_unique<ui::vulkan::SingleLayoutDescriptorSetPool>(
        &device, 64, uint32_t(sizes.size()), sizes.data(), layouts[i]);
  }
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.setLayoutCount = uint32_t(layouts.size());
  info.pSetLayouts = layouts.data();
  return vk.vkCreatePipelineLayout(handle, &info, nullptr, &pipeline_layout) == VK_SUCCESS;
}

DrawPass::~DrawPass() {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  for (size_t i = 0; i < set_indices_.size(); ++i) {
    if (set_indices_[i] != SIZE_MAX) {
      bindings_->pools[i]->Free(set_indices_[i]);
    }
  }
  for (auto sampler : samplers_) {
    vk.vkDestroySampler(device, sampler, nullptr);
  }
  for (const auto& texture : textures_) {
    vk.vkDestroyImageView(device, texture.view, nullptr);
    vk.vkDestroyImage(device, texture.image, nullptr);
    vk.vkFreeMemory(device, texture.memory, nullptr);
  }
  if (buffer_memory_) {
    vk.vkDestroyBuffer(device, buffer_, nullptr);
    vk.vkFreeMemory(device, buffer_memory_, nullptr);
  }
}

bool DrawPass::Initialize(const DrawInput& input, std::span<const uint32_t> indices,
                          const Shader::Translation& vertex, const Shader::Translation* pixel,
                          uint32_t shared_buffer_count, VkPipelineCache pipeline_cache,
                          DrawPipelines& pipelines, DrawBindingCache& binding_cache,
                          ui::vulkan::VulkanUploadBufferPool& uploads, uint64_t submission) {
  const auto& vk = device_.functions();
  const VkDevice device = device_.device();
  const bool has_textures = !input.texture_bindings.empty() || !input.samplers.empty();
  textures_.resize(input.textures.size());
  samplers_.resize(input.samplers.size(), VK_NULL_HANDLE);
  blend_constants_ = input.blend_constants;
  viewport_ = input.viewport;
  scissor_ = input.scissor;
  index_count_ = uint32_t(indices.size());

  const uint32_t image_count = uint32_t(input.texture_bindings.size());
  const uint32_t sampler_count = uint32_t(input.samplers.size());
  const std::array binding_key{shared_buffer_count, image_count, sampler_count};
  auto found = binding_cache.find(binding_key);
  if (found == binding_cache.end()) {
    auto layout = std::make_unique<DrawBindings>(device_);
    if (!layout->Initialize(shared_buffer_count, image_count, sampler_count)) {
      return false;
    }
    found = binding_cache.emplace(binding_key, std::move(layout)).first;
  }
  bindings_ = found->second.get();
  for (size_t i = 0; i < sets_.size(); ++i) {
    auto& pool = *bindings_->pools[i];
    set_indices_[i] = pool.Allocate();
    if (set_indices_[i] == SIZE_MAX) {
      return false;
    }
    sets_[i] = pool.Get(set_indices_[i]);
  }

  const auto props = device_.properties();
  const VkDeviceSize uniform_alignment = props.minUniformBufferOffsetAlignment;
  const std::array<std::span<const uint8_t>, Translator::kConstantBufferCount> constants{{
      {reinterpret_cast<const uint8_t*>(&input.system), sizeof(input.system)},
      {reinterpret_cast<const uint8_t*>(input.float_constants[0].data()),
       input.float_constants[0].size() * sizeof(uint32_t)},
      {reinterpret_cast<const uint8_t*>(input.float_constants[1].data()),
       input.float_constants[1].size() * sizeof(uint32_t)},
      {reinterpret_cast<const uint8_t*>(input.bool_loop.data()), sizeof(input.bool_loop)},
      {reinterpret_cast<const uint8_t*>(input.fetch.data()), sizeof(input.fetch)},
  }};
  const VkDeviceSize vertex_size = input.vertices.size();
  VkDeviceSize size = vertex_size;
  std::array<VkDescriptorBufferInfo, Translator::kConstantBufferCount> constant_infos{};
  for (size_t i = 0; i < constants.size(); ++i) {
    size = rex::align(size, uniform_alignment);
    constant_infos[i].offset = size;
    constant_infos[i].range = std::max<size_t>(constants[i].size(), 16);
    size += constant_infos[i].range;
  }
  index_offset_ = rex::align(size, VkDeviceSize(4));
  size = index_offset_ + indices.size_bytes();
  for (size_t i = 0; i < textures_.size(); ++i) {
    const auto& input_texture = input.textures[i];
    for (uint32_t level = 0; level < input_texture.levels.size(); ++level) {
      auto& copy = textures_[i].uploads.emplace_back();
      copy.bufferOffset = rex::align(size, VkDeviceSize(16));
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
      copy.imageExtent = {std::max(1u, input_texture.width >> level),
                          std::max(1u, input_texture.height >> level), 1};
      size = copy.bufferOffset + input_texture.levels[level].size();
    }
  }
  const VkDeviceSize alignment =
      std::max({VkDeviceSize(16), uniform_alignment, props.minStorageBufferOffsetAlignment});
  VkDeviceSize buffer_offset = 0;
  uint8_t* bytes = nullptr;
  if (rex::align(size, alignment) <= kDrawUploadPageSize) {
    bytes = uploads.Request(submission, size, alignment, buffer_, buffer_offset);
  } else {
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(&device_, size, kDrawUploadUsage,
                                                           ui::vulkan::util::MemoryPurpose::kUpload,
                                                           buffer_, buffer_memory_)) {
      return false;
    }
    void* mapped = nullptr;
    if (vk.vkMapMemory(device, buffer_memory_, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
      bytes = static_cast<uint8_t*>(mapped);
    }
  }
  if (!bytes) {
    return false;
  }
  std::memcpy(bytes, input.vertices.data(), vertex_size);
  for (size_t i = 0; i < constants.size(); ++i) {
    constant_infos[i].buffer = buffer_;
    uint8_t* target = bytes + constant_infos[i].offset;
    if (constants[i].empty()) {
      std::memset(target, 0, constant_infos[i].range);
    } else {
      std::memcpy(target, constants[i].data(), constants[i].size());
    }
    constant_infos[i].offset += buffer_offset;
  }
  std::memcpy(bytes + index_offset_, indices.data(), indices.size_bytes());
  index_offset_ += buffer_offset;
  for (size_t i = 0; i < textures_.size(); ++i) {
    for (size_t level = 0; level < textures_[i].uploads.size(); ++level) {
      const auto& pixels = input.textures[i].levels[level];
      auto& copy = textures_[i].uploads[level];
      std::memcpy(bytes + copy.bufferOffset, pixels.data(), pixels.size());
      copy.bufferOffset += buffer_offset;
    }
  }
  if (buffer_memory_) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer_memory_;
    range.size = VK_WHOLE_SIZE;
    const VkResult flushed = vk.vkFlushMappedMemoryRanges(device, 1, &range);
    vk.vkUnmapMemory(device, buffer_memory_);
    if (flushed != VK_SUCCESS) {
      return false;
    }
  }

  // The captured vertex ranges are packed at the beginning of the first segment.
  std::vector<VkDescriptorBufferInfo> shared_infos(shared_buffer_count,
                                                   {buffer_, buffer_offset, vertex_size});
  std::array<VkWriteDescriptorSet, 1 + Translator::kConstantBufferCount> writes{};
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               nullptr,
               sets_[0],
               0,
               0,
               shared_buffer_count,
               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
               nullptr,
               shared_infos.data()};
  for (uint32_t i = 0; i < constant_infos.size(); ++i) {
    writes[i + 1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets_[1],          i, 0, 1,
                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,      nullptr, &constant_infos[i]};
  }
  vk.vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);

  for (size_t i = 0; i < textures_.size(); ++i) {
    const auto& input_texture = input.textures[i];
    auto& texture = textures_[i];
    texture.resolved = input_texture.resolved_image;
    const uint32_t level_count = texture.resolved ? 1 : uint32_t(texture.uploads.size());
    if (!texture.resolved) {
      VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = input_texture.format;
      info.extent = {input_texture.width, input_texture.height, 1};
      info.mipLevels = level_count;
      info.arrayLayers = 1;
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      if (!ui::vulkan::util::CreateDedicatedAllocationImage(
              &device_, info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, texture.image,
              texture.memory)) {
        return false;
      }
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = texture.resolved ? texture.resolved->image() : texture.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view.format = input_texture.format;
    view.components = input_texture.swizzle;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, level_count, 0, 1};
    if (vk.vkCreateImageView(device, &view, nullptr, &texture.view) != VK_SUCCESS) {
      return false;
    }
  }
  for (size_t i = 0; i < samplers_.size(); ++i) {
    const auto& input_sampler = input.samplers[i];
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.minFilter = input_sampler.min_filter;
    info.magFilter = input_sampler.mag_filter;
    info.mipmapMode = input_sampler.mip_mode;
    info.maxLod = input_sampler.max_lod;
    info.addressModeU = input_sampler.address_u;
    info.addressModeV = input_sampler.address_v;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vk.vkCreateSampler(device, &info, nullptr, &samplers_[i]) != VK_SUCCESS) {
      return false;
    }
  }
  if (has_textures) {
    std::vector<VkDescriptorImageInfo> image_infos;
    std::vector<VkWriteDescriptorSet> image_writes;
    image_infos.reserve(image_count + sampler_count);
    image_writes.reserve(image_count + sampler_count);
    for (uint32_t texture_index : input.texture_bindings) {
      image_infos.push_back({VK_NULL_HANDLE, textures_[texture_index].view,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    }
    for (VkSampler sampler : samplers_) {
      image_infos.push_back({sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    }
    for (uint32_t i = 0; i < image_infos.size(); ++i) {
      image_writes.push_back(
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets_[3], i, 0, 1,
           i < image_count ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
           &image_infos[i]});
    }
    vk.vkUpdateDescriptorSets(device, uint32_t(image_writes.size()), image_writes.data(), 0,
                              nullptr);
  }

  const auto& depth = input.depth_stencil;
  const auto& front = depth.front;
  const auto& back = depth.back;
  DrawPipelineKey key{&vertex,
                      pixel,
                      {uint32_t(input.topology),
                       input.primitive_restart,
                       input.cull,
                       uint32_t(input.front_face),
                       input.color_count,
                       uint32_t(input.depth_format),
                       uint32_t(input.samples),
                       depth.depthTestEnable,
                       depth.depthWriteEnable,
                       uint32_t(depth.depthCompareOp),
                       depth.stencilTestEnable,
                       uint32_t(front.failOp),
                       uint32_t(front.passOp),
                       uint32_t(front.depthFailOp),
                       uint32_t(front.compareOp),
                       front.compareMask,
                       front.writeMask,
                       front.reference,
                       uint32_t(back.failOp),
                       uint32_t(back.passOp),
                       uint32_t(back.depthFailOp),
                       uint32_t(back.compareOp),
                       back.compareMask,
                       back.writeMask,
                       back.reference,
                       shared_buffer_count},
                      {}};
  for (uint32_t i = 0; i < input.color_count; ++i) {
    const auto& color = input.blend[i];
    key.colors[i] = {uint32_t(input.color_formats[i]),
                     color.blendEnable,
                     uint32_t(color.srcColorBlendFactor),
                     uint32_t(color.dstColorBlendFactor),
                     uint32_t(color.colorBlendOp),
                     uint32_t(color.srcAlphaBlendFactor),
                     uint32_t(color.dstAlphaBlendFactor),
                     uint32_t(color.alphaBlendOp),
                     color.colorWriteMask};
  }
  if (auto cached = pipelines.find(key); cached != pipelines.end()) {
    pipeline_ = cached->second;
    return true;
  }

  const VkShaderModule vs = CreateShaderModule(device_, vertex);
  const VkShaderModule ps = pixel ? CreateShaderModule(device_, *pixel) : VK_NULL_HANDLE;
  if (!vs || (pixel && !ps)) {
    vk.vkDestroyShaderModule(device, vs, nullptr);
    vk.vkDestroyShaderModule(device, ps, nullptr);
    return false;
  }
  const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
       vs, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, ps, "main"},
  }};
  const VkPipelineVertexInputStateCreateInfo vertex_info{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  assembly.topology = input.topology;
  assembly.primitiveRestartEnable = input.primitive_restart;
  VkPipelineViewportStateCreateInfo viewport_info{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport_info.viewportCount = viewport_info.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.frontFace = input.front_face;
  raster.cullMode = input.cull;
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo samples{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  samples.rasterizationSamples = input.samples;
  VkPipelineColorBlendStateCreateInfo blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  blend.attachmentCount = input.color_count;
  blend.pAttachments = input.blend.data();
  const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                  VK_DYNAMIC_STATE_BLEND_CONSTANTS};
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = uint32_t(dynamic_states.size());
  dynamic.pDynamicStates = dynamic_states.data();
  VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = blend.attachmentCount;
  rendering.pColorAttachmentFormats = input.color_formats.data();
  rendering.depthAttachmentFormat = input.depth_format;
  rendering.stencilAttachmentFormat = input.depth_format;
  VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline.pNext = &rendering;
  pipeline.stageCount = pixel ? uint32_t(stages.size()) : 1;
  pipeline.pStages = stages.data();
  pipeline.pVertexInputState = &vertex_info;
  pipeline.pInputAssemblyState = &assembly;
  pipeline.pViewportState = &viewport_info;
  pipeline.pRasterizationState = &raster;
  pipeline.pMultisampleState = &samples;
  pipeline.pDepthStencilState = &input.depth_stencil;
  pipeline.pColorBlendState = &blend;
  pipeline.pDynamicState = &dynamic;
  pipeline.layout = bindings_->pipeline_layout;
  const VkResult result =
      vk.vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline, nullptr, &pipeline_);
  vk.vkDestroyShaderModule(device, vs, nullptr);
  vk.vkDestroyShaderModule(device, ps, nullptr);
  if (result != VK_SUCCESS) {
    return false;
  }
  pipelines.emplace(key, pipeline_);
  return true;
}

void DrawPass::RecordTextureUpload(VkCommandBuffer commands) {
  const auto& vk = device_.functions();
  for (const auto& texture : textures_) {
    if (texture.resolved) {
      texture.resolved->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_SHADER_READ_BIT);
      continue;
    }
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = texture.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, uint32_t(texture.uploads.size()), 0,
                                1};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    vk.vkCmdCopyBufferToImage(commands, buffer_, texture.image, barrier.newLayout,
                              uint32_t(texture.uploads.size()), texture.uploads.data());
    barrier.oldLayout = barrier.newLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);
  }
}

void DrawPass::Record(VkCommandBuffer commands, VkExtent2D extent) {
  if (!viewport_.xy_extent[0] || !viewport_.xy_extent[1] || scissor_.offset[0] >= extent.width ||
      scissor_.offset[1] >= extent.height) {
    return;
  }
  const auto& vk = device_.functions();
  const VkViewport viewport{float(viewport_.xy_offset[0]),
                            float(viewport_.xy_offset[1]),
                            float(viewport_.xy_extent[0]),
                            float(viewport_.xy_extent[1]),
                            viewport_.z_min,
                            viewport_.z_max};
  const VkRect2D scissor{{int32_t(scissor_.offset[0]), int32_t(scissor_.offset[1])},
                         {std::min(scissor_.extent[0], extent.width - scissor_.offset[0]),
                          std::min(scissor_.extent[1], extent.height - scissor_.offset[1])}};
  vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vk.vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, bindings_->pipeline_layout,
                             0, uint32_t(sets_.size()), sets_.data(), 0, nullptr);
  vk.vkCmdSetViewport(commands, 0, 1, &viewport);
  vk.vkCmdSetScissor(commands, 0, 1, &scissor);
  vk.vkCmdSetBlendConstants(commands, blend_constants_.data());
  vk.vkCmdBindIndexBuffer(commands, buffer_, index_offset_, VK_INDEX_TYPE_UINT32);
  vk.vkCmdDrawIndexed(commands, index_count_, 1, 0, 0, 0);
}

}  // namespace rex::graphics::native
