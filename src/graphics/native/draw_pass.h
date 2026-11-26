#pragma once

#include <array>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
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
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  // Guest physical byte ranges covered by the decoded texels.
  uint32_t base_start = 0;
  uint32_t base_length = 0;
  uint32_t mip_start = 0;
  uint32_t mip_length = 0;
  // Number of mip levels stored in the image, needed to create a view for a new
  // component mapping without recomputing the guest texture layout.
  uint32_t level_count = 1;
  // Set once a draw carrying this entry is accepted, so later draws in the same
  // batch can sample it without uploading again.
  bool uploaded = false;
  // Per-face resolve addresses and versions last copied into a cached GPU-resolved cube.
  std::array<uint32_t, kTextureCubeArrayLayers> resolved_face_addresses{};
  std::array<uint64_t, kTextureCubeArrayLayers> resolved_face_versions{};
  // Resolve version last copied into a cached cropped 2D GPU resolve.
  uint64_t resolved_crop_version = 0;
  uint64_t last_use_submission = 0;
  // Keyed by format (UNORM and SNORM views share MUTABLE_FORMAT images) and
  // PackTextureComponentMapping.
  std::map<uint64_t, VkImageView> views;
  CachedTexture() = default;
  ~CachedTexture();
  CachedTexture(const CachedTexture&) = delete;
  CachedTexture& operator=(const CachedTexture&) = delete;

  VkImageView GetOrCreateView(VkImageViewType view_type, VkFormat format,
                              VkComponentMapping swizzle, uint32_t array_layers);
};

struct DrawTexture {
  uint32_t width = 0;
  uint32_t height = 0;
  // Slices of a 3D texture (1 otherwise).
  uint32_t depth = 1;
  // Array layers per level (6 for cubes, the stack depth for stacked textures,
  // 1 otherwise). Each level in `levels` stores all layers (or 3D slices)
  // tightly packed.
  uint32_t array_layers = 1;
  // Mip levels of the image; `levels` starts at first_level, which is 1 when
  // the fetch has no base level.
  uint32_t level_count = 1;
  uint32_t first_level = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  // The image is also sampled through an SNORM view.
  bool mutable_format = false;
  // VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE or VK_IMAGE_VIEW_TYPE_3D.
  VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  uint32_t host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_0000;
  VkComponentMapping swizzle{};
  std::vector<std::vector<uint8_t>> levels;
  std::shared_ptr<ColorImage> resolved_image;
  // Cached resolve of the same format but up to 8x8 larger than this texture;
  // its top-left corner is copied into a host image during the upload.
  std::shared_ptr<ColorImage> crop_source;
  uint64_t resolved_crop_version = 0;
  // Resolved host image of each cube face, in face order, when the whole cube
  // was rendered on the GPU instead of being stored in guest memory. Each face
  // is copied into its array layer of the uploaded cube during the upload.
  std::array<std::shared_ptr<ColorImage>, kTextureCubeArrayLayers> resolved_cube_faces;
  std::array<uint64_t, kTextureCubeArrayLayers> resolved_cube_face_versions{};
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
  bool resolution_scaled = false;
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
  // PA_CL_CLIP_CNTL.CLIP_DISABLE, mapped to depth clamping since Vulkan cannot
  // disable clipping entirely.
  bool depth_clamp_enable = false;
  DepthBias depth_bias{};
  std::vector<uint8_t> vertices;
  std::vector<DrawTexture> textures;
  // Indices into `textures` for the pixel (set 3) and vertex (set 2) shader
  // texture bindings.
  std::vector<uint32_t> texture_bindings;
  std::vector<uint32_t> vertex_texture_bindings;
  // Device-lifetime samplers resolved by the command processor's sampler cache.
  std::vector<VkSampler> samplers;
  std::vector<VkSampler> vertex_samplers;
  // `textures` index of each fetch constant's unsigned and signed texture, so
  // both stages share one decode.
  std::array<int, 64> texture_indices = MakeTextureIndices();
  // The draw writes guest memory through the shared vertex buffer.
  bool memexport = false;
  // No attachments: the draw only memexports from the vertex shader.
  bool rasterizer_discard = false;
  // When set, the draw has no pixel shader and instead a fragment shader that
  // packs the EDRAM depth-stencil word of a fast clear into an aliased color
  // target; the stencil is written and the format selects the unpacking.
  bool edram_depth_clear_color = false;
  uint32_t edram_clear_stencil = 0;
  uint32_t edram_clear_format = 0;

  // Resets all fields for reuse across draws while keeping vector capacities.
  void Clear() {
    system = {};
    float_constants[0].clear();
    float_constants[1].clear();
    bool_loop.fill(0);
    fetch.fill(0);
    viewport = {};
    scissor = {};
    primitive_restart = false;
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    blend.fill({});
    color_formats.fill(VK_FORMAT_UNDEFINED);
    color_count = 0;
    depth_stencil = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth_format = VK_FORMAT_UNDEFINED;
    samples = VK_SAMPLE_COUNT_1_BIT;
    blend_constants.fill(0.0f);
    cull = VK_CULL_MODE_NONE;
    front_face = VK_FRONT_FACE_CLOCKWISE;
    depth_clamp_enable = false;
    depth_bias = {};
    vertices.clear();
    textures.clear();
    texture_bindings.clear();
    vertex_texture_bindings.clear();
    samplers.clear();
    vertex_samplers.clear();
    texture_indices = MakeTextureIndices();
    memexport = false;
    rasterizer_discard = false;
    edram_depth_clear_color = false;
    edram_clear_stencil = 0;
    edram_clear_format = 0;
  }

 private:
  static constexpr std::array<int, 64> MakeTextureIndices() {
    std::array<int, 64> indices{};
    indices.fill(-1);
    return indices;
  }
};

struct DrawPipelineKey {
  const Shader::Translation* vertex;
  const Shader::Translation* pixel;
  std::array<uint32_t, 23> state;
  std::array<std::array<uint32_t, 9>, xenos::kMaxColorRenderTargets> colors;
  auto operator<=>(const DrawPipelineKey&) const = default;
};

// Hashes the explicitly initialized members rather than the raw struct bytes,
// so the hash never depends on padding between the pointers and the arrays.
struct DrawPipelineKeyHash {
  size_t operator()(const DrawPipelineKey& key) const noexcept {
    uint64_t hash = XXH3_64bits(&key.vertex, sizeof(key.vertex));
    hash = XXH3_64bits_withSeed(&key.pixel, sizeof(key.pixel), hash);
    hash = XXH3_64bits_withSeed(key.state.data(), sizeof(key.state), hash);
    hash = XXH3_64bits_withSeed(key.colors.data(), sizeof(key.colors), hash);
    return size_t(hash);
  }
};

using DrawPipelines = std::unordered_map<DrawPipelineKey, VkPipeline, DrawPipelineKeyHash>;
constexpr size_t kDrawUploadPageSize = 16 * 1024 * 1024;
constexpr VkBufferUsageFlags kDrawUploadUsage =
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

// Descriptor range of each constant buffer, bound as
// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: the largest data a draw can have.
constexpr std::array<uint64_t, SpirvShaderTranslator::kConstantBufferCount>
    kDynamicConstantBufferRanges{sizeof(SpirvShaderTranslator::SystemConstants),
                                 256 * 4 * sizeof(uint32_t), 256 * 4 * sizeof(uint32_t),
                                 40 * sizeof(uint32_t), 192 * sizeof(uint32_t)};

struct DrawBindings {
  explicit DrawBindings(const ui::vulkan::VulkanDevice& device) : device(device) {}
  ~DrawBindings();
  bool Initialize(uint32_t shared_count, uint32_t vertex_image_count,
                  uint32_t vertex_sampler_count, uint32_t pixel_image_count,
                  uint32_t pixel_sampler_count);
  VkDescriptorSet GetOrCreateSharedVertexSet(VkBuffer buffer, uint32_t shared_count);
  // Dynamic constant buffer set of an upload pool page. Pool pages live until
  // the pool is destroyed, so the set is kept as long as these bindings.
  VkDescriptorSet GetOrCreateConstantSet(VkBuffer buffer);
  void WriteConstantSet(VkDescriptorSet set, VkBuffer buffer) const;
  // The single set shared by every draw for a layout without bindings, or
  // VK_NULL_HANDLE when the layout has bindings (or allocation failed).
  VkDescriptorSet GetEmptySet(size_t index);

  const ui::vulkan::VulkanDevice& device;
  std::array<VkDescriptorSetLayout, 4> layouts{};
  std::array<std::unique_ptr<ui::vulkan::SingleLayoutDescriptorSetPool>, 4> pools;
  std::array<bool, 4> empty_layouts{};
  std::array<size_t, 4> empty_set_indices{SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSet shared_vertex_set = VK_NULL_HANDLE;
  size_t shared_vertex_set_index = SIZE_MAX;
  VkBuffer shared_vertex_buffer = VK_NULL_HANDLE;
  std::unordered_map<VkBuffer, size_t> constant_sets;
};

// Keyed by {shared buffer count, vertex images, vertex samplers, pixel images,
// pixel samplers}.
using DrawBindingCache = std::map<std::array<uint32_t, 5>, std::unique_ptr<DrawBindings>>;

class DrawPass {
 public:
  DrawPass(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator)
      : device_(&device), allocator_(allocator) {}
  ~DrawPass();
  // Movable so frames store passes flat; a moved-from pass owns nothing.
  DrawPass(DrawPass&& other) noexcept;
  DrawPass& operator=(DrawPass&& other) noexcept;
  DrawPass(const DrawPass&) = delete;
  DrawPass& operator=(const DrawPass&) = delete;

  bool Initialize(const DrawInput& input, std::span<const uint32_t> indices,
                  const Shader::Translation& vertex, const Shader::Translation* pixel,
                  uint32_t shared_buffer_count, VkPipelineCache pipeline_cache,
                  DrawPipelines& pipelines, DrawBindingCache& binding_cache,
                  ui::vulkan::VulkanUploadBufferPool& uploads, uint64_t submission,
                  VkBuffer shared_vertex_buffer = VK_NULL_HANDLE);
  // Records only the binds and dynamic state that differ from `state`.
  void Record(VkCommandBuffer commands, VkExtent2D extent, DrawRenderState& state);
  void RecordTextureUpload(VkCommandBuffer commands);
  const draw_util::Scissor& scissor() const { return scissor_; }
  bool rasterizer_discard() const { return rasterizer_discard_; }

 private:
  void Release();

  const ui::vulkan::VulkanDevice* device_;
  VmaAllocator allocator_;
  DrawBindings* bindings_ = nullptr;
  // Sets this pass allocated and frees; shared and cached sets stay SIZE_MAX.
  std::array<size_t, 4> set_indices_{SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  std::array<VkDescriptorSet, 4> sets_{};
  std::array<uint32_t, SpirvShaderTranslator::kConstantBufferCount> dynamic_offsets_{};
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  std::array<float, 4> blend_constants_{};
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory buffer_memory_ = VK_NULL_HANDLE;
  struct Texture {
    std::shared_ptr<ColorImage> resolved;
    std::shared_ptr<ColorImage> crop_source;
    std::array<std::shared_ptr<ColorImage>, kTextureCubeArrayLayers> resolved_cube_faces;
    std::shared_ptr<CachedTexture> cached;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView fallback_view = VK_NULL_HANDLE;
    VkExtent2D extent{};
    uint32_t array_layers = 1;
    uint32_t level_count = 1;
    bool upload_cached = false;
    std::vector<VkBufferImageCopy> uploads;
  };
  std::vector<Texture> textures_;
  VkDeviceSize index_offset_ = 0;
  uint32_t index_count_ = 0;
  draw_util::ViewportInfo viewport_{};
  draw_util::Scissor scissor_{};
  bool depth_bias_enable_ = false;
  DepthBias depth_bias_{};
  bool stencil_enable_ = false;
  std::array<uint32_t, 2> stencil_compare_mask_{};
  std::array<uint32_t, 2> stencil_write_mask_{};
  std::array<uint32_t, 2> stencil_reference_{};
  bool edram_depth_clear_color_ = false;
  std::array<uint32_t, 2> edram_clear_constants_{};
  bool rasterizer_discard_ = false;
};

}  // namespace rex::graphics::native
