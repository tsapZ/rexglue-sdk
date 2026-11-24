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
#include "draw_input.h"

namespace rex::graphics::native {

// Decoded guest texture uploaded to a device-local image that survives across
// draws and frames. Owned through shared_ptr so draws keep it alive while their
// GPU work is still queued after the cache evicts or invalidates the entry.
struct CachedTexture {
  const ui::vulkan::VulkanDevice* device = nullptr;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  // Guest physical byte ranges covered by the decoded texels.
  uint32_t base_start = 0;
  uint32_t base_length = 0;
  uint32_t mip_start = 0;
  uint32_t mip_length = 0;
  // Set once a draw carrying this entry is accepted, so later draws in the same
  // batch can sample it without uploading again.
  bool uploaded = false;
  uint64_t last_use_submission = 0;
  std::map<uint32_t, VkImageView> views;
  CachedTexture() = default;
  ~CachedTexture();
  CachedTexture(const CachedTexture&) = delete;
  CachedTexture& operator=(const CachedTexture&) = delete;
};

struct DrawTexture {
  uint32_t width = 0;
  uint32_t height = 0;
  // Array layers per level (6 for cube textures, 1 otherwise). Each level in
  // `levels` stores all layers tightly packed.
  uint32_t array_layers = 1;
  VkFormat format = VK_FORMAT_UNDEFINED;
  // VK_IMAGE_VIEW_TYPE_2D_ARRAY, or VK_IMAGE_VIEW_TYPE_CUBE for cubes.
  VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  VkComponentMapping swizzle{};
  std::vector<std::vector<uint8_t>> levels;
  std::shared_ptr<ColorImage> resolved_image;
  // Cached resolve of the same format but up to 8x8 larger than this texture;
  // its top-left corner is copied into a host image during the upload.
  std::shared_ptr<ColorImage> crop_source;
  // Device-lifetime zero-cleared view bound when the fetch constant cannot
  // produce texels. Non-owning; the host image is allocated once.
  VkImageView fallback_view = VK_NULL_HANDLE;
  // Device-lifetime texture from the command processor cache. Non-owning when
  // null; the shared pointer keeps the image alive for this draw's GPU work.
  std::shared_ptr<CachedTexture> cached_texture;
  VkImageView cached_view = VK_NULL_HANDLE;
  // Whether this draw must copy `levels` into the cached image. False when the
  // image was already uploaded by an earlier draw in the same batch.
  bool upload_cached = false;
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
  // Device-lifetime samplers resolved by the command processor's sampler cache.
  std::vector<VkSampler> samplers;
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
    std::shared_ptr<ColorImage> crop_source;
    std::shared_ptr<CachedTexture> cached;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView fallback_view = VK_NULL_HANDLE;
    VkExtent2D extent{};
    uint32_t array_layers = 1;
    bool upload_cached = false;
    std::vector<VkBufferImageCopy> uploads;
  };
  std::vector<Texture> textures_;
  VkDeviceSize index_offset_ = 0;
  uint32_t index_count_ = 0;
  draw_util::ViewportInfo viewport_{};
  draw_util::Scissor scissor_{};
};

}  // namespace rex::graphics::native
