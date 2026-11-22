#pragma once

#include <array>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/util/draw.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/single_layout_descriptor_set_pool.h>
#include <rex/ui/vulkan/upload_buffer_pool.h>

#include "color_image.h"

namespace rex::graphics::native {

struct DrawTexture {
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkComponentMapping swizzle{};
  std::vector<std::vector<uint8_t>> levels;
  std::shared_ptr<ColorImage> resolved_image;
};

struct DrawSampler {
  VkFilter min_filter = VK_FILTER_NEAREST;
  VkFilter mag_filter = VK_FILTER_NEAREST;
  VkSamplerMipmapMode mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  float max_lod = 0.0f;
  VkSamplerAddressMode address_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSamplerAddressMode address_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
};

struct DrawInput {
  SpirvShaderTranslator::SystemConstants system{};
  std::array<std::vector<uint32_t>, 2> float_constants;
  std::array<uint32_t, 40> bool_loop{};
  std::array<uint32_t, 192> fetch{};
  draw_util::ViewportInfo viewport{};
  draw_util::Scissor scissor{};
  bool primitive_restart = false;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  std::array<VkPipelineColorBlendAttachmentState, xenos::kMaxColorRenderTargets> blend{};
  std::array<VkFormat, xenos::kMaxColorRenderTargets> color_formats{};
  uint32_t color_count = 0;
  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkFormat depth_format = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  std::array<float, 4> blend_constants{};
  VkCullModeFlags cull = VK_CULL_MODE_NONE;
  VkFrontFace front_face = VK_FRONT_FACE_CLOCKWISE;
  std::vector<uint8_t> vertices;
  std::vector<DrawTexture> textures;
  std::vector<uint32_t> texture_bindings;
  std::vector<DrawSampler> samplers;
};

struct DrawPipelineKey {
  const Shader::Translation* vertex;
  const Shader::Translation* pixel;
  std::array<uint32_t, 27> state;
  std::array<std::array<uint32_t, 9>, xenos::kMaxColorRenderTargets> colors;
  auto operator<=>(const DrawPipelineKey&) const = default;
};

using DrawPipelines = std::map<DrawPipelineKey, VkPipeline>;
constexpr size_t kDrawUploadPageSize = 16 * 1024 * 1024;
constexpr VkBufferUsageFlags kDrawUploadUsage =
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

struct DrawBindings {
  explicit DrawBindings(const ui::vulkan::VulkanDevice& device) : device(device) {}
  ~DrawBindings();
  bool Initialize(uint32_t shared_count, uint32_t image_count, uint32_t sampler_count);

  const ui::vulkan::VulkanDevice& device;
  std::array<VkDescriptorSetLayout, 4> layouts{};
  std::array<std::unique_ptr<ui::vulkan::SingleLayoutDescriptorSetPool>, 4> pools;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
};

using DrawBindingCache = std::map<std::array<uint32_t, 3>, std::unique_ptr<DrawBindings>>;

class DrawPass {
 public:
  explicit DrawPass(const ui::vulkan::VulkanDevice& device) : device_(device) {}
  ~DrawPass();

  bool Initialize(const DrawInput& input, std::span<const uint32_t> indices,
                  const Shader::Translation& vertex, const Shader::Translation* pixel,
                  uint32_t shared_buffer_count, VkPipelineCache pipeline_cache,
                  DrawPipelines& pipelines, DrawBindingCache& binding_cache,
                  ui::vulkan::VulkanUploadBufferPool& uploads, uint64_t submission);
  void Record(VkCommandBuffer commands, VkExtent2D extent);
  void RecordTextureUpload(VkCommandBuffer commands);
  const draw_util::Scissor& scissor() const { return scissor_; }

 private:
  const ui::vulkan::VulkanDevice& device_;
  DrawBindings* bindings_ = nullptr;
  std::array<size_t, 4> set_indices_{SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  std::array<VkDescriptorSet, 4> sets_{};
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  std::array<float, 4> blend_constants_{};
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory buffer_memory_ = VK_NULL_HANDLE;
  struct Texture {
    std::shared_ptr<ColorImage> resolved;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    std::vector<VkBufferImageCopy> uploads;
  };
  std::vector<Texture> textures_;
  std::vector<VkSampler> samplers_;
  VkDeviceSize index_offset_ = 0;
  uint32_t index_count_ = 0;
  draw_util::ViewportInfo viewport_{};
  draw_util::Scissor scissor_{};
};

}  // namespace rex::graphics::native
