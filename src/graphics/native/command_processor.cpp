#include "command_processor.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <optional>

#include <xxhash.h>

#include <rex/cvar.h>
#include <rex/graphics/edram_layout.h>
#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/video_mode_util.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/instance.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/provider.h>
#include <rex/ui/vulkan/util.h>

#include "draw_input.h"
#include "graphics_system.h"
#include "resolve_guest_memory.h"

REXCVAR_DEFINE_STRING(native_frame_timing_path, "", "GPU/Native",
                      "Optional CSV for native Vulkan guest-output frame timing");
REXCVAR_DECLARE(bool, present_use_actual_aspect);
REXCVAR_DECLARE(std::string, resolution);

namespace rex::graphics::native {
namespace {

using Translator = SpirvShaderTranslator;

VkFilter ToFilter(xenos::TextureFilter filter) {
  return filter == xenos::TextureFilter::kLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerAddressMode ToAddressMode(xenos::ClampMode mode) {
  switch (mode) {
    case xenos::ClampMode::kRepeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case xenos::ClampMode::kMirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

std::optional<VkPrimitiveTopology> GetTopology(xenos::PrimitiveType type) {
  switch (type) {
    case xenos::PrimitiveType::kTriangleList:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case xenos::PrimitiveType::kPointList:
    case xenos::PrimitiveType::kRectangleList:
    case xenos::PrimitiveType::kTriangleStrip:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case xenos::PrimitiveType::kTriangleFan:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case xenos::PrimitiveType::kLineList:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case xenos::PrimitiveType::kLineStrip:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    default:
      return std::nullopt;
  }
}

std::optional<VkPipelineColorBlendAttachmentState> GetBlend(const RegisterFile& regs,
                                                            uint32_t target, uint32_t mask) {
  const auto guest =
      regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[target]);
  constexpr VkBlendFactor invalid = VK_BLEND_FACTOR_MAX_ENUM;
  constexpr std::array factors{VK_BLEND_FACTOR_ZERO,
                               VK_BLEND_FACTOR_ONE,
                               invalid,
                               invalid,
                               VK_BLEND_FACTOR_SRC_COLOR,
                               VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                               VK_BLEND_FACTOR_SRC_ALPHA,
                               VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                               VK_BLEND_FACTOR_DST_COLOR,
                               VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                               VK_BLEND_FACTOR_DST_ALPHA,
                               VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                               VK_BLEND_FACTOR_CONSTANT_COLOR,
                               VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                               VK_BLEND_FACTOR_CONSTANT_ALPHA,
                               VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                               VK_BLEND_FACTOR_SRC_ALPHA_SATURATE};
  constexpr std::array operations{VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_MIN,
                                  VK_BLEND_OP_MAX, VK_BLEND_OP_REVERSE_SUBTRACT};
  const auto factor = [&](xenos::BlendFactor f) {
    return uint32_t(f) < factors.size() ? factors[uint32_t(f)] : invalid;
  };
  const auto op = [&](xenos::BlendOp value) {
    return uint32_t(value) < operations.size() ? operations[uint32_t(value)] : VK_BLEND_OP_MAX_ENUM;
  };
  VkPipelineColorBlendAttachmentState blend{};
  blend.srcColorBlendFactor = factor(guest.color_srcblend);
  blend.dstColorBlendFactor = factor(guest.color_destblend);
  blend.colorBlendOp = op(guest.color_comb_fcn);
  blend.srcAlphaBlendFactor = factor(guest.alpha_srcblend);
  blend.dstAlphaBlendFactor = factor(guest.alpha_destblend);
  blend.alphaBlendOp = op(guest.alpha_comb_fcn);
  if (blend.srcColorBlendFactor == invalid || blend.dstColorBlendFactor == invalid ||
      blend.srcAlphaBlendFactor == invalid || blend.dstAlphaBlendFactor == invalid ||
      blend.colorBlendOp == VK_BLEND_OP_MAX_ENUM || blend.alphaBlendOp == VK_BLEND_OP_MAX_ENUM) {
    return std::nullopt;
  }
  blend.blendEnable = (guest.value & 0x1FFF1FFF) != 0x00010001;
  blend.colorWriteMask = mask;
  return blend;
}

VkFormat ColorTargetFormat(xenos::ColorRenderTargetFormat format) {
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      return ColorImage::kFormat;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

VkFormat CopyColorFormat(xenos::ColorFormat format) {
  switch (format) {
    case xenos::ColorFormat::k_8_8_8_8:
    case xenos::ColorFormat::k_8_8_8_8_A:
      return ColorImage::kFormat;
    case xenos::ColorFormat::k_8:
      return VK_FORMAT_R8_UNORM;
    case xenos::ColorFormat::k_8_8:
      return VK_FORMAT_R8G8_UNORM;
    case xenos::ColorFormat::k_2_10_10_10:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case xenos::ColorFormat::k_16_16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
    case xenos::ColorFormat::k_16_16_16_16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case xenos::ColorFormat::k_32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

bool SupportsResolveFormatBlit(const ui::vulkan::VulkanDevice& device, VkFormat source,
                               VkFormat destination) {
  const auto& ifn = device.vulkan_instance()->functions();
  VkFormatProperties source_properties{};
  VkFormatProperties destination_properties{};
  ifn.vkGetPhysicalDeviceFormatProperties(device.physical_device(), source, &source_properties);
  ifn.vkGetPhysicalDeviceFormatProperties(device.physical_device(), destination,
                                          &destination_properties);
  return (source_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0 &&
         (destination_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
}

VkClearColorValue UnpackColor(VkFormat format, uint32_t packed, uint32_t packed_hi = 0) {
  switch (format) {
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return {{float(packed & 1023) / 1023.0f, float((packed >> 10) & 1023) / 1023.0f,
               float((packed >> 20) & 1023) / 1023.0f, float(packed >> 30) / 3.0f}};
    case VK_FORMAT_R16G16_SFLOAT:
      return {{rex::xenos_half_to_float(uint16_t(packed)),
               rex::xenos_half_to_float(uint16_t(packed >> 16)), 0, 0}};
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return {{rex::xenos_half_to_float(uint16_t(packed)),
               rex::xenos_half_to_float(uint16_t(packed >> 16)),
               rex::xenos_half_to_float(uint16_t(packed_hi)),
               rex::xenos_half_to_float(uint16_t(packed_hi >> 16))}};
    case VK_FORMAT_R32_SFLOAT:
      return {{std::bit_cast<float>(packed), 0, 0, 0}};
    default:
      return {{float(packed & 255) / 255.0f, float((packed >> 8) & 255) / 255.0f,
               float((packed >> 16) & 255) / 255.0f, float(packed >> 24) / 255.0f}};
  }
}

}  // namespace

NativeCommandProcessor::NativeCommandProcessor(NativeGraphicsSystem* graphics_system,
                                               system::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state) {}

NativeCommandProcessor::~NativeCommandProcessor() = default;

bool NativeCommandProcessor::SetupContext() {
  auto* provider = static_cast<ui::vulkan::VulkanProvider*>(graphics_system_->provider());
  if (!provider || !(vulkan_device_ = provider->vulkan_device()) ||
      !vulkan_device_->properties().dynamicRendering ||
      !vulkan_device_->properties().shaderStorageImageExtendedFormats) {
    REXLOG_ERROR("Native Vulkan requires dynamic rendering and extended storage image formats");
    return false;
  }

  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  const auto& instance = vulkan_device_->vulkan_instance()->functions();
  cmd_resolve_image_ = reinterpret_cast<PFN_vkCmdResolveImage>(
      instance.vkGetDeviceProcAddr(device, "vkCmdResolveImage"));
  cmd_blit_image_ =
      reinterpret_cast<PFN_vkCmdBlitImage>(instance.vkGetDeviceProcAddr(device, "vkCmdBlitImage"));
  if (cmd_blit_image_) {
    r8_resolve_blit_ =
        SupportsResolveFormatBlit(*vulkan_device_, ColorImage::kFormat, VK_FORMAT_R8_UNORM);
    rg8_resolve_blit_ =
        SupportsResolveFormatBlit(*vulkan_device_, ColorImage::kFormat, VK_FORMAT_R8G8_UNORM);
  }
  constexpr VkFormatFeatureFlags kUsedDepthFormatFeatures =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
  VkFormatProperties depth_unorm24_properties{};
  instance.vkGetPhysicalDeviceFormatProperties(vulkan_device_->physical_device(),
                                               VK_FORMAT_D24_UNORM_S8_UINT,
                                               &depth_unorm24_properties);
  depth_unorm24_supported_ = (depth_unorm24_properties.optimalTilingFeatures &
                              kUsedDepthFormatFeatures) == kUsedDepthFormatFeatures;
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool.queueFamilyIndex = vulkan_device_->queue_family_graphics_compute();
  if (pool.queueFamilyIndex == UINT32_MAX ||
      vk.vkCreateCommandPool(device, &pool, nullptr, &command_pool_) != VK_SUCCESS) {
    return false;
  }
  VkCommandBufferAllocateInfo buffer{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  buffer.commandPool = command_pool_;
  buffer.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  buffer.commandBufferCount = 1;
  const VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vk.vkAllocateCommandBuffers(device, &buffer, &command_buffer_) != VK_SUCCESS ||
      vk.vkCreateFence(device, &fence, nullptr, &fence_) != VK_SUCCESS) {
    ShutdownContext();
    return false;
  }

  TextureCache::GetConfigDrawResolutionScale(draw_resolution_scale_x_, draw_resolution_scale_y_);
  const uint32_t max_w = std::min(vulkan_device_->properties().maxImageDimension2D,
                                  vulkan_device_->properties().maxFramebufferWidth);
  const uint32_t max_h = std::min(vulkan_device_->properties().maxImageDimension2D,
                                  vulkan_device_->properties().maxFramebufferHeight);
  uint32_t guest_w = 1280;
  uint32_t guest_h = 720;
  int32_t configured_w = 0;
  int32_t configured_h = 0;
  if (video_mode_util::TryParseResolutionPreset(REXCVAR_GET(resolution), configured_w,
                                                configured_h) &&
      configured_w > 0 && configured_h > 0) {
    guest_w = std::clamp(uint32_t(configured_w), 1280u, 1920u);
    guest_h = std::clamp(uint32_t(configured_h), 720u, 1080u);
  }
  draw_resolution_scale_x_ = ClampResolutionScaleAxis(draw_resolution_scale_x_, guest_w, max_w);
  draw_resolution_scale_y_ = ClampResolutionScaleAxis(draw_resolution_scale_y_, guest_h, max_h);
  translator_ =
      std::make_unique<Translator>(Translator::Features(vulkan_device_), false, false, false,
                                   draw_resolution_scale_x_, draw_resolution_scale_y_);
  VkPipelineCacheCreateInfo pipeline_cache{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  draw_uploads_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
      vulkan_device_, kDrawUploadUsage, kDrawUploadPageSize);
  output_gamma_ = std::make_unique<OutputGamma>(*vulkan_device_);
  depth_resolve_ = std::make_unique<DepthResolve>(*vulkan_device_);
  color_resolve_ = std::make_unique<ColorResolve>(*vulkan_device_);
  if (!output_gamma_->Initialize() || !depth_resolve_->Initialize() ||
      !color_resolve_->Initialize() ||
      vk.vkCreatePipelineCache(device, &pipeline_cache, nullptr, &pipeline_cache_) != VK_SUCCESS) {
    ShutdownContext();
    return false;
  }
  const auto& timing_path = REXCVAR_GET(native_frame_timing_path);
  if (!timing_path.empty()) {
    frame_timing_.open(timing_path);
    if (!frame_timing_) {
      REXLOG_ERROR("Native Vulkan: could not open frame timing file {}", timing_path);
      ShutdownContext();
      return false;
    }
    frame_timing_ << "swap,address,width,height,ready_ns,result\n";
  }
  if (!InitializeNullTextures()) {
    ShutdownContext();
    return false;
  }
  const uint32_t shared_segment_size =
      kPhysicalMemorySize >> translator_->GetSharedMemoryStorageBufferCountLog2();
  if (vulkan_device_->properties().maxStorageBufferRange >= shared_segment_size &&
      ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device_, kPhysicalMemorySize,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, shared_vertex_buffer_,
          shared_vertex_memory_)) {
    shared_vertex_valid_pages_.assign((kSharedVertexPageCount + 63) / 64, 0);
    cached_texture_pages_.assign((kSharedVertexPageCount + 63) / 64, 0);
  }
  memory_invalidation_handle_ =
      memory_->RegisterPhysicalMemoryInvalidationCallback(MemoryInvalidationCallbackThunk, this);
  if (!memory_invalidation_handle_) {
    ShutdownContext();
    return false;
  }
  SetActiveEdramLayout(RequestedEdramLayout());
  REXLOG_INFO("Native Vulkan: command processor initialized ({}x{} scale)", draw_resolution_scale_x_,
              draw_resolution_scale_y_);
  return true;
}

bool NativeCommandProcessor::InitializeNullTextures() {
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  image.imageType = VK_IMAGE_TYPE_2D;
  image.format = ColorImage::kFormat;
  image.extent = {1, 1, 1};
  image.mipLevels = 1;
  image.arrayLayers = 6;
  image.samples = VK_SAMPLE_COUNT_1_BIT;
  image.tiling = VK_IMAGE_TILING_OPTIMAL;
  image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          vulkan_device_, image, ui::vulkan::util::MemoryPurpose::kDeviceLocal, null_texture_image_,
          null_texture_memory_)) {
    return false;
  }
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = null_texture_image_;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view.format = ColorImage::kFormat;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vk.vkCreateImageView(device, &view, nullptr, &null_texture_2d_view_) != VK_SUCCESS) {
    return false;
  }
  view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  view.subresourceRange.layerCount = 6;
  return vk.vkCreateImageView(device, &view, nullptr, &null_texture_cube_view_) == VK_SUCCESS;
}

void NativeCommandProcessor::DestroyNullTextures() {
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  vk.vkDestroyImageView(device, null_texture_cube_view_, nullptr);
  vk.vkDestroyImageView(device, null_texture_2d_view_, nullptr);
  vk.vkDestroyImage(device, null_texture_image_, nullptr);
  vk.vkFreeMemory(device, null_texture_memory_, nullptr);
  null_texture_cube_view_ = null_texture_2d_view_ = VK_NULL_HANDLE;
  null_texture_image_ = VK_NULL_HANDLE;
  null_texture_memory_ = VK_NULL_HANDLE;
  null_textures_initialized_ = false;
}

void NativeCommandProcessor::RecordNullTextureInitialization() {
  if (null_textures_initialized_ || null_texture_image_ == VK_NULL_HANDLE) {
    return;
  }
  const auto& vk = vulkan_device_->functions();
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = null_texture_image_;
  barrier.subresourceRange = range;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  const VkClearColorValue zero{};
  vk.vkCmdClearColorImage(command_buffer_, null_texture_image_,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &barrier);
  null_textures_initialized_ = true;
}

void NativeCommandProcessor::ShutdownContext() {
  depth_unorm24_supported_ = false;
  if (memory_invalidation_handle_) {
    memory_->UnregisterPhysicalMemoryInvalidationCallback(memory_invalidation_handle_);
    memory_invalidation_handle_ = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    pending_invalidations_.clear();
    has_pending_invalidations_.store(false, std::memory_order_relaxed);
  }
  if (!vulkan_device_) {
    return;
  }
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  if (!RetireSubmission() && submitted_) {
    auto queue = vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
    vk.vkQueueWaitIdle(queue.queue());
  }
  if (frame_timing_.is_open()) {
    frame_timing_.close();
  }
  pending_.clear();
  in_flight_.clear();
  scratch_input_.Clear();
  in_flight_readback_.reset();
  // Draws in pending_ / in_flight_ have released their shared textures by now.
  cached_textures_.clear();
  resolved_cubes_.clear();
  resolved_crops_.clear();
  for (const auto& entry : samplers_) {
    vk.vkDestroySampler(device, entry.second.sampler, nullptr);
  }
  samplers_.clear();
  draw_uploads_.reset();
  if (shared_vertex_buffer_ != VK_NULL_HANDLE) {
    vk.vkDestroyBuffer(device, shared_vertex_buffer_, nullptr);
    shared_vertex_buffer_ = VK_NULL_HANDLE;
  }
  if (shared_vertex_memory_ != VK_NULL_HANDLE) {
    vk.vkFreeMemory(device, shared_vertex_memory_, nullptr);
    shared_vertex_memory_ = VK_NULL_HANDLE;
  }
  shared_vertex_valid_pages_.clear();
  cached_texture_pages_.clear();
  presented_.reset();
  presented_resolve_address_.reset();
  resolved_images_.clear();
  pending_fast_clears_.clear();
  pending_color_fast_clears_.clear();
  targets_.clear();
  DestroyNullTextures();
  for (const auto& entry : pipelines_) {
    vk.vkDestroyPipeline(device, entry.second, nullptr);
  }
  pipelines_.clear();
  draw_bindings_.clear();
  vk.vkDestroyPipelineCache(device, pipeline_cache_, nullptr);
  pipeline_cache_ = VK_NULL_HANDLE;
  color_resolve_.reset();
  depth_resolve_.reset();
  output_gamma_.reset();
  translator_.reset();
  active_vertex_shader_ = active_pixel_shader_ = nullptr;
  shaders_.clear();
  vk.vkDestroyFence(device, fence_, nullptr);
  vk.vkDestroyCommandPool(device, command_pool_, nullptr);
  command_buffer_ = VK_NULL_HANDLE;
  command_pool_ = VK_NULL_HANDLE;
  fence_ = VK_NULL_HANDLE;
  vulkan_device_ = nullptr;
}

Shader* NativeCommandProcessor::LoadShader(xenos::ShaderType shader_type,
                                           uint32_t /*guest_address*/, const uint32_t* host_address,
                                           uint32_t dword_count) {
  if (!host_address || !dword_count) {
    return nullptr;
  }
  const size_t byte_count = size_t(dword_count) * sizeof(uint32_t);
  const uint64_t hash = XXH3_64bits(host_address, byte_count);
  const auto [first, last] = shaders_.equal_range(hash);
  for (auto it = first; it != last; ++it) {
    auto& shader = *it->second;
    if (shader.type() == shader_type && shader.ucode_dword_count() == dword_count &&
        std::equal(shader.ucode_data().begin(), shader.ucode_data().end(), host_address,
                   [](uint32_t cached, uint32_t guest) {
                     if constexpr (std::endian::native == std::endian::little) {
                       return cached == rex::byte_swap(guest);
                     }
                     return cached == guest;
                   })) {
      return &shader;
    }
  }
  auto shader = std::make_unique<SpirvShader>(shader_type, hash, host_address, dword_count);
  return shaders_.emplace(hash, std::move(shader))->second.get();
}

std::span<const uint8_t> NativeCommandProcessor::ReadPhysical(uint32_t address, size_t size) {
  if (!IsPhysicalRange(address, size) ||
      memory_->GetPhysicalHeap()->QueryRangeAccess(address, address + uint32_t(size) - 1) ==
          memory::PageAccess::kNoAccess) {
    return {};
  }
  return {memory_->TranslatePhysical(address), size};
}

bool NativeCommandProcessor::ReadTextures(const SpirvShader& shader, DrawInput& input) {
  DrainMemoryInvalidations();
  const auto& textures = shader.GetTextureBindingsAfterTranslation();
  const auto& samplers = shader.GetSamplerBindingsAfterTranslation();
  const auto& properties = vulkan_device_->properties();
  if (textures.size() > properties.maxPerStageDescriptorSampledImages ||
      samplers.size() > properties.maxPerStageDescriptorSamplers) {
    return false;
  }

  std::array<int, 32> indices;
  indices.fill(-1);
  for (const auto& binding : textures) {
    const bool cube = binding.dimension == xenos::FetchOpDimension::kCube;
    if (!cube && binding.dimension != xenos::FetchOpDimension::k2D) {
      return false;
    }
    const uint32_t slot = binding.fetch_constant;
    const auto fetch = register_file_->GetTextureFetch(slot);
    // A fetch constant that isn't a usable texture of the fetched dimension
    // reads as zero on the guest - bind a zero-cleared fallback rather than
    // dropping the draw. Its signs stay unsigned.
    const bool use_fallback = ShouldUseFallbackTexture(
        fetch.type == xenos::FetchConstantType::kTexture,
        (fetch.base_address & 0x1FFFF) != 0, fetch.stacked,
        fetch.dimension == (cube ? xenos::DataDimension::kCube
                                 : xenos::DataDimension::k2DOrStacked));
    if (indices[slot] < 0) {
      auto& texture = input.textures.emplace_back();
      if (use_fallback) {
        texture.fallback_view =
            cube ? null_texture_cube_view_ : null_texture_2d_view_;
      } else if (!ReadTexture(fetch, texture)) {
        REXGPU_DEBUG(
            "Native Vulkan texture {} unsupported: format={} base={:05X} flags={:08X}/{:08X}", slot,
            uint32_t(fetch.format), uint32_t(fetch.base_address), fetch.dword_4, fetch.dword_5);
        return false;
      }
      if (!use_fallback) {
        input.system.texture_swizzled_signs[slot / 4] |= uint32_t(texture_util::SwizzleSigns(fetch))
                                                         << ((slot % 4) * 8);
        if (texture.resolution_scaled) {
          input.system.textures_resolution_scaled |= uint32_t(1) << slot;
        }
        if (!properties.imageViewFormatSwizzle) {
          PackSystemTextureSwizzle(input.system.texture_swizzles, slot, texture.host_swizzle);
        }
      }
      indices[slot] = int(input.textures.size() - 1);
    }
    input.texture_bindings.push_back(uint32_t(indices[slot]));
  }
  for (const auto& sampler : samplers) {
    const auto fetch = register_file_->GetTextureFetch(sampler.fetch_constant);
    DrawSampler out;
    out.min_filter =
        ToFilter(sampler.min_filter == xenos::TextureFilter::kUseFetchConst ? fetch.min_filter
                                                                            : sampler.min_filter);
    out.mag_filter =
        ToFilter(sampler.mag_filter == xenos::TextureFilter::kUseFetchConst ? fetch.mag_filter
                                                                            : sampler.mag_filter);
    const auto mip_filter = sampler.mip_filter == xenos::TextureFilter::kUseFetchConst
                                ? fetch.mip_filter
                                : sampler.mip_filter;
    out.mip_mode = mip_filter == xenos::TextureFilter::kLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                               : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    out.max_lod = mip_filter == xenos::TextureFilter::kBaseMap ? 0.0f : float(fetch.mip_max_level);
    out.address_u = ToAddressMode(fetch.clamp_x);
    out.address_v = ToAddressMode(fetch.clamp_y);
    const VkSampler handle = GetSampler(out);
    if (handle == VK_NULL_HANDLE) {
      return false;
    }
    input.samplers.push_back(handle);
  }
  return true;
}

VkSampler NativeCommandProcessor::GetSampler(const DrawSampler& key) {
  auto existing = samplers_.find(key);
  if (existing != samplers_.end()) {
    existing->second.last_use_submission = next_submission_;
    return existing->second.sampler;
  }
  if (samplers_.size() >= kMaxSamplers) {
    const auto victim = SelectRetiredEviction(
        samplers_, retired_submission_ + 1,
        [](const DrawSampler&, const CachedSampler&) { return true; });
    if (!victim) {
      return VK_NULL_HANDLE;
    }
    auto victim_it = samplers_.find(*victim);
    vulkan_device_->functions().vkDestroySampler(vulkan_device_->device(),
                                                 victim_it->second.sampler, nullptr);
    samplers_.erase(victim_it);
  }
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  info.minFilter = key.min_filter;
  info.magFilter = key.mag_filter;
  info.mipmapMode = key.mip_mode;
  info.maxLod = key.max_lod;
  info.addressModeU = key.address_u;
  info.addressModeV = key.address_v;
  info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSampler sampler = VK_NULL_HANDLE;
  if (vulkan_device_->functions().vkCreateSampler(vulkan_device_->device(), &info, nullptr,
                                                  &sampler) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  samplers_.emplace(key, CachedSampler{sampler, next_submission_});
  return sampler;
}

void NativeCommandProcessor::InvalidateTextures(uint32_t start, uint32_t length) {
  bool erased = false;
  for (auto it = cached_textures_.begin(); it != cached_textures_.end();) {
    const CachedTexture& texture = *it->second.texture;
    if (RangesOverlap(start, length, texture.base_start, texture.base_length) ||
        RangesOverlap(start, length, texture.mip_start, texture.mip_length)) {
      it = cached_textures_.erase(it);
      erased = true;
    } else {
      ++it;
    }
  }
  if (erased) {
    RebuildCachedTexturePages();
  }
}

void NativeCommandProcessor::PruneResolvedDependents() {
  for (auto it = resolved_crops_.begin(); it != resolved_crops_.end();) {
    if (!resolved_images_.contains(it->first.address)) {
      it = resolved_crops_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = resolved_cubes_.begin(); it != resolved_cubes_.end();) {
    bool alive = true;
    for (uint32_t face_addr : it->second.texture->resolved_face_addresses) {
      if (!face_addr || !resolved_images_.contains(face_addr)) {
        alive = false;
        break;
      }
    }
    if (!alive) {
      it = resolved_cubes_.erase(it);
    } else {
      ++it;
    }
  }
}

void NativeCommandProcessor::InvalidateResolvedTextures(uint32_t start, uint32_t length) {
  bool erased = false;
  for (auto it = resolved_images_.begin(); it != resolved_images_.end();) {
    if (RangesOverlap(start, length, it->second.guest_start, it->second.guest_length)) {
      if (presented_resolve_address_ && *presented_resolve_address_ == it->first) {
        presented_resolve_address_.reset();
      }
      it = resolved_images_.erase(it);
      erased = true;
    } else {
      ++it;
    }
  }
  if (erased) {
    PruneResolvedDependents();
    RebuildCachedTexturePages();
  }
}

void NativeCommandProcessor::RebuildCachedTexturePages() {
  if (cached_texture_pages_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(invalidation_mutex_);
  std::fill(cached_texture_pages_.begin(), cached_texture_pages_.end(), 0);
  for (const auto& [key, entry] : cached_textures_) {
    const auto& texture = *entry.texture;
    const auto [base_first, base_count] =
        GetPhysicalPageRange(texture.base_start, texture.base_length);
    SetPageRangeBits(cached_texture_pages_, base_first, base_count);
    if (texture.mip_length) {
      const auto [mip_first, mip_count] =
          GetPhysicalPageRange(texture.mip_start, texture.mip_length);
      SetPageRangeBits(cached_texture_pages_, mip_first, mip_count);
    }
  }
  for (const auto& [addr, entry] : resolved_images_) {
    const auto [first, count] = GetPhysicalPageRange(entry.guest_start, entry.guest_length);
    SetPageRangeBits(cached_texture_pages_, first, count);
  }
}

void NativeCommandProcessor::DrainMemoryInvalidations() {
  if (!has_pending_invalidations_.load(std::memory_order_acquire)) {
    return;
  }
  std::vector<std::pair<uint32_t, uint32_t>> invalidations;
  {
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    invalidations.swap(pending_invalidations_);
    has_pending_invalidations_.store(false, std::memory_order_release);
  }
  for (const auto& [start, length] : invalidations) {
    InvalidateTextures(start, length);
    InvalidateResolvedTextures(start, length);
    InvalidateSharedVertices(start, length);
  }
}

void NativeCommandProcessor::InvalidateGpuMemory() {
  cached_textures_.clear();
  resolved_images_.clear();
  resolved_cubes_.clear();
  resolved_crops_.clear();
  presented_resolve_address_.reset();
  std::fill(shared_vertex_valid_pages_.begin(), shared_vertex_valid_pages_.end(), 0);
  if (!cached_texture_pages_.empty()) {
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    std::fill(cached_texture_pages_.begin(), cached_texture_pages_.end(), 0);
  }
}

void NativeCommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  if (!vulkan_device_) {
    return;
  }
  if (!pending_.empty() && BeginCommands()) {
    RecordPending();
    SubmitCommands();
  }
  if (!RetireSubmission() && submitted_) {
    auto queue = vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
    vulkan_device_->functions().vkQueueWaitIdle(queue.queue());
    submitted_ = false;
  }
  pending_.clear();
  in_flight_.clear();
  in_flight_readback_.reset();
  presented_.reset();
  InvalidateGpuMemory();
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  for (const auto& entry : samplers_) {
    vk.vkDestroySampler(device, entry.second.sampler, nullptr);
  }
  samplers_.clear();
  for (const auto& entry : pipelines_) {
    vk.vkDestroyPipeline(device, entry.second, nullptr);
  }
  pipelines_.clear();
  draw_bindings_.clear();
  pending_fast_clears_.clear();
  pending_color_fast_clears_.clear();
  targets_.clear();
}

void NativeCommandProcessor::InvalidateSharedVertices(uint32_t start, uint32_t length) {
  if (shared_vertex_valid_pages_.empty()) {
    return;
  }
  const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
  ClearPageRangeBits(shared_vertex_valid_pages_, first_page, page_count);
}

bool NativeCommandProcessor::EnsureSharedVertexRange(uint32_t address, uint32_t size) {
  if (!size || ReadPhysical(address, size).empty()) {
    return false;
  }
  const auto [first_page, page_count] = GetPhysicalPageRange(address, size);
  const uint32_t end_page = first_page + page_count;
  uint32_t cursor = first_page;
  uint32_t run_first_page = 0;
  uint32_t run_page_count = 0;
  while (FindNextInvalidPageRun(shared_vertex_valid_pages_, cursor, end_page, run_first_page,
                                run_page_count)) {
    const uint32_t max_chunk_pages = uint32_t(kDrawUploadPageSize >> kSharedVertexPageShift);
    uint32_t chunk_page = run_first_page;
    uint32_t remaining_pages = run_page_count;
    while (remaining_pages) {
      const uint32_t chunk_pages = std::min(remaining_pages, max_chunk_pages);
      const uint32_t chunk_addr = chunk_page << kSharedVertexPageShift;
      const uint32_t chunk_size =
          std::min(chunk_pages << kSharedVertexPageShift, kPhysicalMemorySize - chunk_addr);
      const auto bytes = ReadPhysical(chunk_addr, chunk_size);
      if (bytes.empty()) {
        return false;
      }
      VkBuffer src_buffer = VK_NULL_HANDLE;
      VkDeviceSize src_offset = 0;
      uint8_t* dst =
          draw_uploads_->Request(next_submission_, chunk_size, 16, src_buffer, src_offset);
      if (!dst) {
        return false;
      }
      // Watch the guest pages before reading them so any write after the copy
      // invalidates them before a later draw reads stale vertices.
      memory_->EnablePhysicalMemoryAccessCallbacks(chunk_addr, chunk_size, true, false);
      std::memcpy(dst, bytes.data(), chunk_size);
      pending_.emplace_back(QueuedBufferUpload{src_buffer, src_offset, chunk_addr, chunk_size});
      SetPageRangeBits(shared_vertex_valid_pages_, chunk_page, chunk_pages);
      chunk_page += chunk_pages;
      remaining_pages -= chunk_pages;
    }
  }
  return true;
}

std::pair<uint32_t, uint32_t> NativeCommandProcessor::MemoryInvalidationCallbackThunk(
    void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range) {
  return reinterpret_cast<NativeCommandProcessor*>(context_ptr)
      ->MemoryInvalidationCallback(physical_address_start, length, exact_range);
}

std::pair<uint32_t, uint32_t> NativeCommandProcessor::MemoryInvalidationCallback(
    uint32_t physical_address_start, uint32_t length, bool exact_range) {
  if (!length || physical_address_start >= kPhysicalMemorySize) {
    return {0, UINT32_MAX};
  }
  length = std::min(length, kPhysicalMemorySize - physical_address_start);
  {
    // Only the tiny pending list is touched here; this runs on a guest thread
    // while the global critical region is held.
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    if (!exact_range && !cached_texture_pages_.empty()) {
      const auto [widened_start, widened_length] =
          WidenInvalidationPageRange(cached_texture_pages_, physical_address_start, length);
      if (widened_length) {
        physical_address_start = widened_start;
        length = widened_length;
      }
    }
    pending_invalidations_.emplace_back(physical_address_start, length);
    has_pending_invalidations_.store(true, std::memory_order_release);
  }
  return {physical_address_start, length};
}

template <typename EntryMap, typename Key, typename OnCreated>
std::shared_ptr<CachedTexture> NativeCommandProcessor::GetOrCreateCachedTexture(
    EntryMap& cache, size_t max_entries, const Key& key, const VkImageCreateInfo& image_info,
    OnCreated&& on_created, bool& failed) {
  failed = false;
  if (auto found = cache.find(key); found != cache.end()) {
    found->second.last_use_submission = next_submission_;
    found->second.texture->last_use_submission = next_submission_;
    return found->second.texture;
  }
  if (cache.size() >= max_entries) {
    const auto victim = SelectRetiredEviction(
        cache, retired_submission_ + 1,
        [](const Key&, const CachedTextureEntry&) { return true; });
    if (victim) {
      cache.erase(*victim);
    }
  }
  if (cache.size() >= max_entries) {
    return {};
  }
  auto entry = std::make_shared<CachedTexture>();
  entry->device = vulkan_device_;
  entry->level_count = image_info.mipLevels;
  entry->last_use_submission = next_submission_;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          vulkan_device_, image_info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, entry->image,
          entry->memory)) {
    failed = true;
    return {};
  }
  on_created(*entry);
  cache.emplace(key, CachedTextureEntry{entry, next_submission_});
  return entry;
}

bool NativeCommandProcessor::ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch,
                                         DrawTexture& texture) {
  const bool cube = fetch.dimension == xenos::DataDimension::kCube;
  if (fetch.type != xenos::FetchConstantType::kTexture ||
      (!cube && fetch.dimension != xenos::DataDimension::k2DOrStacked) || fetch.stacked) {
    return false;
  }
  // IEEE float host formats are natively signed, so a signed guest fetch maps
  // straight onto them. Integer, UNORM and BC formats would need separate SNORM
  // views, so those signed fetches are still rejected until they exist.
  if (!IsFloatTextureFormat(fetch.format) &&
      (fetch.sign_x == xenos::TextureSign::kSigned ||
       fetch.sign_y == xenos::TextureSign::kSigned ||
       fetch.sign_z == xenos::TextureSign::kSigned ||
       fetch.sign_w == xenos::TextureSign::kSigned)) {
    return false;
  }

  texture.width = fetch.size_2d.width + 1;
  texture.height = fetch.size_2d.height + 1;
  if (cube && texture.width != texture.height) {
    return false;
  }
  texture.array_layers = GetTextureArrayLayers(fetch.dimension);
  texture.view_type = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  uint32_t block_size;
  uint32_t block_width = 1;
  bool depth_texture = false;
  switch (fetch.format) {
    case xenos::TextureFormat::k_8:
    case xenos::TextureFormat::k_8_A:
    case xenos::TextureFormat::k_8_B:
      texture.format = VK_FORMAT_R8_UNORM;
      block_size = 1;
      break;
    case xenos::TextureFormat::k_8_8:
      texture.format = VK_FORMAT_R8G8_UNORM;
      block_size = 2;
      break;
    case xenos::TextureFormat::k_5_6_5:
      texture.format = VK_FORMAT_R5G6B5_UNORM_PACK16;
      block_size = 2;
      break;
    case xenos::TextureFormat::k_32_FLOAT:
      texture.format = VK_FORMAT_R32_SFLOAT;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_16_16_FLOAT:
      texture.format = VK_FORMAT_R16G16_SFLOAT;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_16_16_16_16_FLOAT:
      texture.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      block_size = 8;
      break;
    case xenos::TextureFormat::k_24_8:
    case xenos::TextureFormat::k_24_8_FLOAT:
      texture.format = VK_FORMAT_R32_SFLOAT;
      block_size = 4;
      depth_texture = true;
      break;
    case xenos::TextureFormat::k_2_10_10_10:
    case xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16:
      texture.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16:
    case xenos::TextureFormat::k_8_8_8_8_A:
    case xenos::TextureFormat::k_8_8_8_8:
      texture.format = VK_FORMAT_R8G8B8A8_UNORM;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT1:
      texture.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
      block_width = 4;
      block_size = 8;
      break;
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT2_3:
      texture.format = VK_FORMAT_BC2_UNORM_BLOCK;
      block_width = 4;
      block_size = 16;
      break;
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT4_5:
      texture.format = VK_FORMAT_BC3_UNORM_BLOCK;
      block_width = 4;
      block_size = 16;
      break;
    case xenos::TextureFormat::k_DXN:
      texture.format = VK_FORMAT_BC5_UNORM_BLOCK;
      block_width = 4;
      block_size = 16;
      break;
    default:
      return false;
  }

  // Replicate the guest components that the format actually stores, matching
  // the Xenos Vulkan host format swizzle: RRRR for one channel, RGGG for two,
  // RGBA otherwise. Block sizes don't map to channels, so use the count.
  const uint32_t channel_count = TextureChannelCount(fetch.format);
  uint32_t base_page, mip_page, mip_min, mip_max;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, &base_page,
                                                 &mip_page, &mip_min, &mip_max);
  if (!base_page || mip_min != 0) {
    return false;
  }
  const uint32_t address = base_page << 12;
  const uint32_t scaled_width = texture.width * draw_resolution_scale_x_;
  const uint32_t scaled_height = texture.height * draw_resolution_scale_y_;
  const bool is_scaled = draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1;
  const bool use_view_swizzle = vulkan_device_->properties().imageViewFormatSwizzle;
  auto resolved = resolved_images_.find(address);
  if (cube && !mip_max && resolved != resolved_images_.end()) {
    // Some titles render each cube face as a separate small color target and
    // resolve it to its own slice of the guest layout, leaving guest memory
    // zero. Assemble and cache a cube image from those six resolved faces,
    // re-copying only when a face resolve version advances.
    const auto cube_layout = texture_util::GetGuestTextureLayout(
        fetch.dimension, fetch.pitch, texture.width, texture.height, 1, fetch.tiled,
        fetch.format, fetch.packed_mips, true, mip_max);
    const uint32_t face_stride = cube_layout.base.array_slice_stride_bytes;
    std::array<std::shared_ptr<ColorImage>, kTextureCubeArrayLayers> faces{};
    std::array<uint32_t, kTextureCubeArrayLayers> face_addresses{};
    std::array<uint64_t, kTextureCubeArrayLayers> face_versions{};
    std::optional<bool> face_swap;
    bool faces_usable = face_stride != 0;
    for (uint32_t face = 0; faces_usable && face < kTextureCubeArrayLayers; ++face) {
      const uint32_t face_addr = GetResolvedCubeFaceAddress(address, face_stride, face);
      auto face_resolved = resolved_images_.find(face_addr);
      if (face_resolved == resolved_images_.end()) {
        faces_usable = false;
        break;
      }
      auto& entry = face_resolved->second;
      const auto face_extent = entry.image->extent();
      if (!CanUseResolvedCubeFace(texture.format == entry.image->format(), face_extent.width,
                                  face_extent.height, scaled_width, scaled_height,
                                  draw_resolution_scale_x_, draw_resolution_scale_y_)) {
        faces_usable = false;
        break;
      }
      const bool swap_red_blue =
          !depth_texture && reg::RB_COPY_DEST_INFO{entry.dest_info}.copy_dest_swap != 0;
      if (face_swap && *face_swap != swap_red_blue) {
        faces_usable = false;
        break;
      }
      face_swap = swap_red_blue;
      faces[face] = entry.image;
      face_addresses[face] = face_addr;
      face_versions[face] = entry.version;
      entry.last_use_submission = next_submission_;
    }
    if (faces_usable) {
      texture.width = scaled_width;
      texture.height = scaled_height;
      texture.resolution_scaled = is_scaled;
      texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count, *face_swap);
      texture.swizzle =
          MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);
      const ResolvedCubeKey cube_key{address, scaled_width, scaled_height, texture.format};
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = texture.format;
      image.extent = {scaled_width, scaled_height, 1};
      image.mipLevels = 1;
      image.arrayLayers = kTextureCubeArrayLayers;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      bool failed = false;
      auto entry = GetOrCreateCachedTexture(
          resolved_cubes_, kMaxResolvedCubes, cube_key, image,
          [&](CachedTexture& created) { created.resolved_face_addresses = face_addresses; },
          failed);
      if (failed) {
        return false;
      }
      if (entry) {
        entry->resolved_face_addresses = face_addresses;
        const VkImageView image_view = entry->GetOrCreateView(
            VK_IMAGE_VIEW_TYPE_CUBE, texture.format, texture.swizzle, kTextureCubeArrayLayers);
        if (image_view == VK_NULL_HANDLE) {
          return false;
        }
        const bool needs_upload =
            ShouldUploadResolvedCube(entry->uploaded, entry->resolved_face_versions, face_versions);
        texture.cached_texture = entry;
        texture.cached_view = image_view;
        texture.upload_cached = needs_upload;
        if (needs_upload) {
          texture.resolved_cube_faces = std::move(faces);
          texture.resolved_cube_face_versions = face_versions;
        }
        return true;
      }
      texture.resolved_cube_faces = std::move(faces);
      return true;
    }
  }
  if (!cube && !mip_max && resolved != resolved_images_.end() &&
      texture.format == resolved->second.image->format()) {
    // A 32bpp 3/4-channel resolve copied with copy_dest_swap keeps the EDRAM
    // red/blue order, while the guest fetch swizzle expects them swapped in
    // memory, so apply that swap to the host view instead.
    const bool swap_red_blue =
        !depth_texture && reg::RB_COPY_DEST_INFO{resolved->second.dest_info}.copy_dest_swap != 0;
    const auto extent = resolved->second.image->extent();
    if (extent.width == scaled_width && extent.height == scaled_height) {
      resolved->second.last_use_submission = next_submission_;
      texture.width = scaled_width;
      texture.height = scaled_height;
      texture.resolution_scaled = is_scaled;
      texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count, swap_red_blue);
      texture.swizzle =
          MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);
      texture.resolved_image = resolved->second.image;
      return true;
    }
    // Resolves are 8x8-aligned, so the cached image can be up to one alignment
    // unit larger than the texture. Crop its top-left corner during the upload;
    // otherwise fall through to guest memory so a newly written CPU texture at
    // the same address with a different size is not dropped.
    const uint32_t alignment = resolved->second.tiled_destination ? xenos::kTextureTileWidthHeight
                                                                  : xenos::kResolveAlignmentPixels;
    if (CanCropAlignedResolve(extent.width, extent.height, scaled_width, scaled_height,
                              draw_resolution_scale_x_, draw_resolution_scale_y_, alignment)) {
      resolved->second.last_use_submission = next_submission_;
      texture.width = scaled_width;
      texture.height = scaled_height;
      texture.resolution_scaled = is_scaled;
      texture.host_swizzle = MakeHostTextureSwizzle(fetch.swizzle, channel_count, swap_red_blue);
      texture.swizzle =
          MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);
      const ResolvedCropKey crop_key{address, scaled_width, scaled_height, texture.format};
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = texture.format;
      image.extent = {scaled_width, scaled_height, 1};
      image.mipLevels = 1;
      image.arrayLayers = 1;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      bool failed = false;
      auto entry = GetOrCreateCachedTexture(resolved_crops_, kMaxResolvedCrops, crop_key, image,
                                            [](CachedTexture&) {}, failed);
      if (failed) {
        return false;
      }
      if (entry) {
        const VkImageView image_view = entry->GetOrCreateView(
            VK_IMAGE_VIEW_TYPE_2D_ARRAY, texture.format, texture.swizzle, 1);
        if (image_view == VK_NULL_HANDLE) {
          return false;
        }
        const bool needs_upload = ShouldUploadResolvedCrop(
            entry->uploaded, entry->resolved_crop_version, resolved->second.version);
        texture.cached_texture = entry;
        texture.cached_view = image_view;
        texture.upload_cached = needs_upload;
        if (needs_upload) {
          texture.crop_source = resolved->second.image;
          texture.resolved_crop_version = resolved->second.version;
        }
        return true;
      }
      texture.crop_source = resolved->second.image;
      return true;
    }
  }
  texture.host_swizzle = MakeHostTextureSwizzle(
      fetch.swizzle, channel_count, fetch.format == xenos::TextureFormat::k_5_6_5);
  texture.swizzle =
      MakeTextureComponentMappingFromHostSwizzle(texture.host_swizzle, use_view_swizzle);

  GuestTextureKey key;
  key.base_page = base_page;
  key.mip_page = mip_page;
  key.width = texture.width;
  key.height = texture.height;
  key.array_layers = texture.array_layers;
  key.format = fetch.format;
  key.dimension = fetch.dimension;
  key.tiled = fetch.tiled != 0;
  key.packed_mips = fetch.packed_mips != 0;
  key.endianness = fetch.endianness;
  key.pitch = fetch.pitch;
  key.mip_max = mip_max;

  // A cached texture whose texels are already uploaded can be sampled straight
  // away: the guest layout and the physical page checks only matter when the
  // image is created or decoded for the first time.
  std::shared_ptr<CachedTexture> entry;
  auto cached = cached_textures_.find(key);
  if (cached != cached_textures_.end()) {
    entry = cached->second.texture;
    cached->second.last_use_submission = next_submission_;
    entry->last_use_submission = next_submission_;
    if (entry->uploaded) {
      const VkImageView image_view = entry->GetOrCreateView(
          texture.view_type, texture.format, texture.swizzle, texture.array_layers);
      if (image_view == VK_NULL_HANDLE) {
        return false;
      }
      texture.cached_texture = entry;
      texture.cached_view = image_view;
      texture.upload_cached = false;
      return true;
    }
  }

  const auto layout = texture_util::GetGuestTextureLayout(
      fetch.dimension, fetch.pitch, texture.width, texture.height, 1, fetch.tiled, fetch.format,
      fetch.packed_mips, true, mip_max);

  const uint32_t level_count = layout.max_level + 1;
  const uint32_t base_length =
      uint32_t(rex::align(layout.base.level_data_extent_bytes, uint32_t(4)));
  if (!IsPhysicalRange(address, base_length)) {
    return false;
  }
  uint32_t mip_start = 0;
  uint32_t mip_length = 0;
  if (layout.max_level > 0) {
    mip_start = mip_page << 12;
    mip_length = uint32_t(rex::align(layout.mips_total_extent_bytes, uint32_t(4)));
    if (!IsPhysicalRange(mip_start, mip_length)) {
      return false;
    }
  }

  if (!entry) {
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = texture.format;
    image.extent = {texture.width, texture.height, 1};
    image.mipLevels = level_count;
    image.arrayLayers = texture.array_layers;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (texture.view_type == VK_IMAGE_VIEW_TYPE_CUBE) {
      image.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    bool failed = false;
    const size_t size_before = cached_textures_.size();
    entry = GetOrCreateCachedTexture(
        cached_textures_, kMaxCachedTextures, key, image,
        [&](CachedTexture& created) {
          created.base_start = address;
          created.base_length = base_length;
          created.mip_start = mip_start;
          created.mip_length = mip_length;
          if (!cached_texture_pages_.empty()) {
            std::lock_guard<std::mutex> lock(invalidation_mutex_);
            const auto [base_first, base_count] = GetPhysicalPageRange(address, base_length);
            SetPageRangeBits(cached_texture_pages_, base_first, base_count);
            if (mip_length) {
              const auto [mip_first, mip_count] = GetPhysicalPageRange(mip_start, mip_length);
              SetPageRangeBits(cached_texture_pages_, mip_first, mip_count);
            }
          }
          // Watch the guest pages before reading them, so a write between the read and
          // the next frame invalidates this entry.
          memory_->EnablePhysicalMemoryAccessCallbacks(address, base_length, true, false);
          if (mip_length) {
            memory_->EnablePhysicalMemoryAccessCallbacks(mip_start, mip_length, true, false);
          }
        },
        failed);
    if (failed) {
      return false;
    }
    if (cached_textures_.size() <= size_before && entry) {
      RebuildCachedTexturePages();
    }
  }

  const bool upload = !entry || !entry->uploaded;
  if (upload) {
    constexpr std::array<size_t, 4> byte_xor{0, 1, 3, 2};
    const size_t swap = byte_xor[uint32_t(fetch.endianness)];
    for (uint32_t level = 0; level <= layout.max_level; ++level) {
      const uint32_t level_storage = std::min(level, layout.packed_level);
      const auto& level_layout = level ? layout.mips[level_storage] : layout.base;
      const uint64_t level_address =
          level ? (uint64_t(mip_page) << 12) + layout.mip_offsets_bytes[level_storage] : address;
      if (level_address >= kPhysicalMemorySize) {
        return false;
      }
      // Array slices are stored within the level, each `array_slice_stride_bytes`
      // apart, so the last slice's data must be readable as well.
      const size_t level_read_size =
          GetTextureLevelReadSize(level_layout.array_slice_stride_bytes,
                                  level_layout.array_slice_data_extent_bytes, texture.array_layers);
      const auto source =
          ReadPhysical(uint32_t(level_address), rex::align(level_read_size, size_t(4)));
      if (source.empty()) {
        return false;
      }
      uint32_t start_x = 0, start_y = 0, start_z = 0;
      if (level >= layout.packed_level) {
        texture_util::GetPackedMipOffset(texture.width, texture.height, 1, fetch.format, level,
                                         start_x, start_y, start_z);
      }
      const uint32_t level_width = std::max(1u, texture.width >> level);
      const uint32_t level_height = std::max(1u, texture.height >> level);
      const uint32_t blocks_x = (level_width + block_width - 1) / block_width;
      const uint32_t blocks_y = (level_height + block_width - 1) / block_width;
      auto& data = texture.levels.emplace_back(size_t(blocks_x) * blocks_y * block_size *
                                               texture.array_layers);
      for (uint32_t slice = 0; slice < texture.array_layers; ++slice) {
        const size_t slice_base = size_t(slice) * level_layout.array_slice_stride_bytes;
        // Each face is tiled independently from the origin of its own data.
        for (uint32_t y = 0; y < blocks_y; ++y) {
          for (uint32_t x = 0; x < blocks_x; ++x) {
            const size_t offset =
                slice_base +
                (fetch.tiled ? size_t(texture_util::GetTiledOffset2D(
                                   int32_t(x + start_x), int32_t(y + start_y),
                                   level_layout.row_pitch_bytes / block_size,
                                   std::countr_zero(block_size)))
                             : size_t(y + start_y) * level_layout.row_pitch_bytes +
                                   size_t(x + start_x) * block_size);
            if (offset > source.size() || block_size > source.size() - offset) {
              return false;
            }
            uint8_t* dst = data.data() +
                           ((size_t(slice) * blocks_y + y) * blocks_x + x) * block_size;
            for (uint32_t i = 0; i < block_size; ++i) {
              dst[i] = source[(offset + i) ^ swap];
            }
            if (depth_texture) {
              uint32_t packed;
              std::memcpy(&packed, dst, sizeof(packed));
              const float depth = fetch.format == xenos::TextureFormat::k_24_8_FLOAT
                                      ? xenos::Float20e4To32(packed >> 8)
                                      : float(packed >> 8) / 16777215.0f;
              std::memcpy(dst, &depth, sizeof(depth));
            }
          }
        }
      }
    }
  }

  if (!entry) {
    return true;
  }
  const VkImageView image_view = entry->GetOrCreateView(texture.view_type, texture.format,
                                                        texture.swizzle, texture.array_layers);
  if (image_view == VK_NULL_HANDLE) {
    return false;
  }
  texture.cached_texture = entry;
  texture.cached_view = image_view;
  texture.upload_cached = upload;
  return true;
}

bool NativeCommandProcessor::IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                                       IndexBufferInfo* index_buffer_info,
                                       bool /*major_mode_explicit*/) {
  const RegisterFile& regs = *register_file_;
  const auto mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (mode == xenos::EdramMode::kCopy) {
    return IssueCopy();
  }
  if (!active_vertex_shader_) {
    return false;
  }
  auto& candidate_vs = *static_cast<SpirvShader*>(active_vertex_shader_);
  if (!candidate_vs.is_ucode_analyzed()) {
    string::StringBuffer disassembly;
    candidate_vs.AnalyzeUcode(disassembly);
  }
  const auto dimensions = regs.Get<reg::RB_SURFACE_INFO>();
  const uint32_t color_mask = regs.values[XE_GPU_REG_RB_COLOR_MASK] & 0xFFFF;
  if ((!dimensions.surface_pitch || !color_mask) && !candidate_vs.memexport_eM_written() &&
      !regs.Get<reg::RB_DEPTHCONTROL>().z_write_enable &&
      !regs.Get<reg::RB_DEPTHCONTROL>().stencil_enable) {
    return true;
  }
  const bool rectangle = prim_type == xenos::PrimitiveType::kRectangleList;
  const bool point = prim_type == xenos::PrimitiveType::kPointList;
  const bool expanded = rectangle || point;
  if (rectangle && index_count < 3) {
    return true;
  }
  const auto topology = GetTopology(prim_type);
  const auto depth_control = draw_util::GetNormalizedDepthControl(regs);
  const bool use_depth = depth_control.z_enable || depth_control.stencil_enable;
  const bool use_color = mode == xenos::EdramMode::kColorDepth && color_mask;
  const auto depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  const bool float_depth = depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8;
  if (!use_color && !use_depth && !candidate_vs.memexport_eM_written()) {
    return true;
  }
  if (!topology || !index_count || !dimensions.surface_pitch ||
      dimensions.msaa_samples > xenos::MsaaSamples::k4X || (use_color && !active_pixel_shader_)) {
    return false;
  }

  if (!FlushPendingIfFull()) {
    return false;
  }
  std::vector<uint32_t>& indices = scratch_indices_;
  indices.resize(index_count);
  if (index_buffer_info) {
    const auto& index = *index_buffer_info;
    const size_t index_bytes =
        size_t(index_count) *
        (index.format == xenos::IndexFormat::kInt32 ? sizeof(uint32_t) : sizeof(uint16_t));
    if (index_bytes > index.length || !DecodeDmaIndices(ReadPhysical(index.guest_base, index_bytes),
                                                        index.format, index.endianness, indices)) {
      return false;
    }
  } else {
    std::iota(indices.begin(), indices.end(), uint32_t(0));
  }

  auto& vs = *static_cast<SpirvShader*>(active_vertex_shader_);
  auto* ps = mode == xenos::EdramMode::kColorDepth ? static_cast<SpirvShader*>(active_pixel_shader_)
                                                   : nullptr;
  if (ps && !ps->is_ucode_analyzed()) {
    string::StringBuffer disassembly;
    ps->AnalyzeUcode(disassembly);
  }
  if (vs.memexport_eM_written() || (ps && ps->memexport_eM_written())) {
    return false;
  }

  const auto program = regs.Get<reg::SQ_PROGRAM_CNTL>();
  uint32_t param_gen = UINT32_MAX;
  const auto interpolators =
      ps ? vs.writes_interpolators() &
               ps->GetInterpolatorInputMask(program, regs.Get<reg::SQ_CONTEXT_MISC>(), param_gen)
         : 0;
  auto host_vertex_type = Shader::HostVertexShaderType::kVertex;
  if (rectangle) {
    host_vertex_type = Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
  } else if (point) {
    host_vertex_type = Shader::HostVertexShaderType::kPointListAsTriangleStrip;
  }
  const auto& properties = vulkan_device_->properties();
  const auto clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  const auto ucp = GetUserClipPlaneConfig(clip_cntl, properties.shaderClipDistance,
                                          properties.shaderCullDistance);
  Translator::Modification vertex_modification(translator_->GetDefaultVertexShaderModification(
      vs.GetDynamicAddressableRegisterCount(program.vs_num_reg), host_vertex_type));
  vertex_modification.vertex.interpolator_mask = interpolators;
  vertex_modification.vertex.user_clip_plane_count = ucp.count;
  vertex_modification.vertex.user_clip_plane_cull = ucp.cull_only ? 1u : 0u;
  vertex_modification.vertex.point_ps_ucp_mode = clip_cntl.ps_ucp_mode;
  vertex_modification.vertex.output_point_parameters =
      point && param_gen < xenos::kMaxInterpolators;
  Shader::Translation* pixel = nullptr;
  if (ps) {
    Translator::Modification pixel_modification(translator_->GetDefaultPixelShaderModification(
        ps->GetDynamicAddressableRegisterCount(program.ps_num_reg)));
    pixel_modification.pixel.interpolator_mask = interpolators;
    pixel_modification.pixel.interpolators_centroid =
        interpolators & ~xenos::GetInterpolatorSamplingPattern(
                            dimensions.msaa_samples,
                            regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
                            regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);
    if (param_gen < xenos::kMaxInterpolators) {
      pixel_modification.pixel.param_gen_enable = 1;
      pixel_modification.pixel.param_gen_interpolator = param_gen;
      pixel_modification.pixel.param_gen_point = point;
    }
    pixel = ps->GetOrCreateTranslation(pixel_modification.value);
  }
  auto& vertex = *vs.GetOrCreateTranslation(vertex_modification.value);
  if ((!vertex.is_translated() && !translator_->TranslateAnalyzedShader(vertex)) ||
      (pixel && ((!pixel->is_translated() && !translator_->TranslateAnalyzedShader(*pixel)) ||
                 !pixel->is_valid())) ||
      !vertex.is_valid() || !vs.GetTextureBindingsAfterTranslation().empty() ||
      !vs.GetSamplerBindingsAfterTranslation().empty()) {
    return false;
  }

  scratch_input_.Clear();
  DrawInput& input = scratch_input_;
  struct ScratchInputGuard {
    DrawInput& input;
    ~ScratchInputGuard() { input.textures.clear(); }
  } scratch_guard{scratch_input_};
  input.topology = *topology;
  input.primitive_restart = index_buffer_info &&
                            regs.Get<reg::PA_SU_SC_MODE_CNTL>().multi_prim_ib_ena &&
                            (prim_type == xenos::PrimitiveType::kTriangleStrip ||
                             prim_type == xenos::PrimitiveType::kTriangleFan ||
                             prim_type == xenos::PrimitiveType::kLineStrip);
  if (input.primitive_restart) {
    const uint32_t reset = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
    const uint32_t mask = index_buffer_info->format == xenos::IndexFormat::kInt16
                              ? uint32_t(UINT16_MAX)
                              : xenos::kVertexIndexMask;
    std::ranges::replace(indices, reset & mask, UINT32_MAX);
  }
  input.primitive_restart |= expanded;
  input.samples = VkSampleCountFlagBits(1u << uint32_t(dimensions.msaa_samples));
  input.depth_format = use_depth ? GetDepthFormat(depth_info.depth_format) : VK_FORMAT_UNDEFINED;
  auto& depth = input.depth_stencil;
  depth.depthTestEnable = depth_control.z_enable;
  depth.depthWriteEnable = depth_control.z_enable && depth_control.z_write_enable;
  depth.depthCompareOp = VkCompareOp(depth_control.zfunc);
  depth.stencilTestEnable = depth_control.stencil_enable;
  const auto front = regs.Get<reg::RB_STENCILREFMASK>();
  const auto back = regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  depth.front = {VkStencilOp(depth_control.stencilfail),
                 VkStencilOp(depth_control.stencilzpass),
                 VkStencilOp(depth_control.stencilzfail),
                 VkCompareOp(depth_control.stencilfunc),
                 front.stencilmask,
                 front.stencilwritemask,
                 front.stencilref};
  depth.back = depth_control.backface_enable
                   ? VkStencilOpState{VkStencilOp(depth_control.stencilfail_bf),
                                      VkStencilOp(depth_control.stencilzpass_bf),
                                      VkStencilOp(depth_control.stencilzfail_bf),
                                      VkCompareOp(depth_control.stencilfunc_bf),
                                      back.stencilmask,
                                      back.stencilwritemask,
                                      back.stencilref}
                   : depth.front;
  const auto raster = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  const bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  const auto cull = GetCullFaces(primitive_polygonal, raster.cull_front, raster.cull_back);
  input.cull = (cull.front ? VK_CULL_MODE_FRONT_BIT : 0) | (cull.back ? VK_CULL_MODE_BACK_BIT : 0);
  if (point) {
    input.cull = VK_CULL_MODE_NONE;
  }
  if (!properties.separateStencilMaskRef && primitive_polygonal && depth_control.backface_enable) {
    if (cull.front && !cull.back) {
      depth.front = depth.back;
    } else {
      depth.back = depth.front;
    }
  }
  input.front_face = raster.face ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  input.depth_clamp_enable = GetDepthClampEnable(properties.depthClamp, clip_cntl.clip_disable);
  input.depth_bias =
      use_depth ? ComputeHostDepthBias(regs, primitive_polygonal, depth_info.depth_format,
                                       draw_resolution_scale_x_, draw_resolution_scale_y_)
                : DepthBias{};
  if (ucp.enabled_mask) {
    uint32_t ucp_remaining = ucp.enabled_mask;
    uint32_t out_index = 0;
    uint32_t plane_index = 0;
    while (rex::bit_scan_forward(ucp_remaining, &plane_index)) {
      ucp_remaining &= ~(UINT32_C(1) << plane_index);
      std::memcpy(input.system.user_clip_planes[out_index++],
                  &regs[XE_GPU_REG_PA_CL_UCP_0_X + plane_index * 4], 4 * sizeof(float));
    }
  }
  for (size_t channel = 0; channel < input.blend_constants.size(); ++channel) {
    input.blend_constants[channel] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED + uint32_t(channel));
  }
  if (ps && !ReadTextures(*ps, input)) {
    return false;
  }
  draw_util::GetHostViewportInfo(regs, draw_resolution_scale_x_, draw_resolution_scale_y_, false,
                                 properties.maxViewportDimensions[0],
                                 properties.maxViewportDimensions[1], true, depth_control,
                                 float_depth, true, ps && ps->writes_depth(), input.viewport);
  draw_util::GetScissor(regs, input.scissor);
  input.scissor.offset[0] *= draw_resolution_scale_x_;
  input.scissor.offset[1] *= draw_resolution_scale_y_;
  input.scissor.extent[0] *= draw_resolution_scale_x_;
  input.scissor.extent[1] *= draw_resolution_scale_y_;
  if (!input.viewport.xy_extent[0] || !input.viewport.xy_extent[1] || !input.scissor.extent[0] ||
      !input.scissor.extent[1]) {
    return true;
  }
  const auto transform = regs.Get<reg::PA_CL_VTE_CNTL>();
  const auto color = regs.Get<reg::RB_COLORCONTROL>();
  input.system.flags =
      (primitive_polygonal ? Translator::kSysFlag_PrimitivePolygonal : 0) |
      (draw_util::IsPrimitiveLine(regs) ? Translator::kSysFlag_PrimitiveLine : 0) |
      (uint32_t(color.alpha_test_enable ? color.alpha_func : xenos::CompareFunction::kAlways)
       << Translator::kSysFlag_AlphaPassIfLess_Shift);
  input.system.flags |= uint32_t(dimensions.msaa_samples) << Translator::kSysFlag_MsaaSamples_Shift;
  if (float_depth) {
    input.system.flags |= Translator::kSysFlag_DepthFloat24;
  }
  if (transform.vtx_xy_fmt)
    input.system.flags |= Translator::kSysFlag_XYDividedByW;
  if (transform.vtx_z_fmt)
    input.system.flags |= Translator::kSysFlag_ZDividedByW;
  if (transform.vtx_w0_fmt)
    input.system.flags |= Translator::kSysFlag_WNotReciprocal;
  if (expanded && index_buffer_info) {
    input.system.flags |= Translator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad |
                          Translator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit;
    if (raster.multi_prim_ib_ena) {
      const uint32_t mask = index_buffer_info->format == xenos::IndexFormat::kInt16
                                ? uint32_t(UINT16_MAX)
                                : xenos::kVertexIndexMask;
      input.system.flags |= Translator::kSysFlag_ComputeOrPrimitiveVertexIndexReset;
      input.system.vertex_index_reset =
          regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx & mask;
    }
    const auto index_bytes = std::as_bytes(std::span(indices));
    input.vertices.resize(index_bytes.size());
    std::memcpy(input.vertices.data(), index_bytes.data(), index_bytes.size());
  }
  if (expanded) {
    FillQuadStripIndices(rectangle ? index_count / 3 : index_count, indices);
  }
  if (point) {
    const auto minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    const auto size = regs.Get<reg::PA_SU_POINT_SIZE>();
    input.system.point_vertex_diameter_min = float(minmax.min_size) / 8.0f;
    input.system.point_vertex_diameter_max = float(minmax.max_size) / 8.0f;
    input.system.point_constant_diameter[0] = float(size.width) / 8.0f;
    input.system.point_constant_diameter[1] = float(size.height) / 8.0f;
    input.system.point_screen_diameter_to_ndc_radius[0] =
        float(draw_resolution_scale_x_) / float(input.viewport.xy_extent[0]);
    input.system.point_screen_diameter_to_ndc_radius[1] =
        float(draw_resolution_scale_y_) / float(input.viewport.xy_extent[1]);
  }
  input.system.alpha_test_reference = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  input.system.alpha_to_mask = color.alpha_to_mask_enable ? (color.value >> 24) | (1u << 8) : 0;
  input.system.vertex_base_index = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  input.system.vertex_index_min = regs.Get<uint32_t>(XE_GPU_REG_VGT_MIN_VTX_INDX);
  input.system.vertex_index_max = regs.Get<uint32_t>(XE_GPU_REG_VGT_MAX_VTX_INDX);
  std::ranges::copy(input.viewport.ndc_scale, input.system.ndc_scale);
  std::ranges::copy(input.viewport.ndc_offset, input.system.ndc_offset);
  std::ranges::fill(input.system.color_exp_bias, 1.0f);
  std::memcpy(input.bool_loop.data(), &regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
              sizeof(input.bool_loop));
  std::memcpy(input.fetch.data(), &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
              sizeof(input.fetch));
  const std::array<const Shader*, 2> shaders{&vs, ps};
  for (size_t stage = 0; stage < shaders.size(); ++stage) {
    if (!shaders[stage]) {
      continue;
    }
    const auto& map = shaders[stage]->constant_register_map();
    for (uint32_t i = 0; i < 256; ++i) {
      if (map.float_bitmap[i / 64] & (uint64_t(1) << (i % 64))) {
        const auto* value = &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + (stage * 256 + i) * 4];
        input.float_constants[stage].insert(input.float_constants[stage].end(), value, value + 4);
      }
    }
  }

  DrainMemoryInvalidations();
  const bool may_snapshot_fast_clear =
      dimensions.msaa_samples == xenos::MsaaSamples::k4X &&
      ((!use_color && use_depth && !ps) || rectangle);
  const bool use_shared_vertices = shared_vertex_buffer_ != VK_NULL_HANDLE &&
                                   !(expanded && index_buffer_info) && !may_snapshot_fast_clear;
  std::array<bool, 96> uploaded{};
  for (const auto& binding : vs.vertex_bindings()) {
    if (uploaded[binding.fetch_constant]) {
      continue;
    }
    auto fetch = regs.GetVertexFetch(binding.fetch_constant);
    if (fetch.type != xenos::FetchConstantType::kVertex) {
      return false;
    }
    if (use_shared_vertices) {
      if (!EnsureSharedVertexRange(fetch.address << 2, fetch.size << 2)) {
        return false;
      }
    } else {
      const auto bytes = ReadPhysical(fetch.address << 2, fetch.size << 2);
      if (bytes.empty() || bytes.size() > properties.maxStorageBufferRange - input.vertices.size()) {
        return false;
      }
      fetch.address = uint32_t(input.vertices.size() / sizeof(uint32_t));
      std::memcpy(&input.fetch[binding.fetch_constant * 2], &fetch, sizeof(fetch));
      input.vertices.insert(input.vertices.end(), bytes.begin(), bytes.end());
    }
    uploaded[binding.fetch_constant] = true;
  }

  if (!use_shared_vertices && input.vertices.empty()) {
    input.vertices.resize(sizeof(uint32_t));
  }

  QueuedDraw draw{};
  const auto layout = ActiveEdramLayout();
  const uint32_t writes =
      use_color ? draw_util::GetNormalizedColorMask(regs, ps->writes_color_targets()) : 0;
  std::array<bool, xenos::kMaxColorRenderTargets> bound_rt{};
  std::array<uint32_t, xenos::kMaxColorRenderTargets> bound_rt_tile{};
  std::array<bool, xenos::kMaxColorRenderTargets> bound_rt_64bpp{};
  bool any_blend_enabled = false;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    const uint32_t mask = (writes >> (i * 4)) & 15;
    if (!mask) {
      continue;
    }
    const auto target = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    const auto blend = GetBlend(regs, i, mask);
    const VkFormat format = ColorTargetFormat(target.color_format);
    if (!blend || format == VK_FORMAT_UNDEFINED ||
        (!vulkan_device_->properties().independentBlend && i != 0)) {
      return false;
    }
    const uint32_t rt_tile = layout.DecodeBase(target.value);
    const bool rt_is_64bpp = xenos::IsColorRenderTargetFormat64bpp(target.color_format);
    draw.colors[i] = FindTarget(rt_tile, dimensions.surface_pitch, dimensions.msaa_samples,
                                format, VK_IMAGE_ASPECT_COLOR_BIT, rt_is_64bpp);
    if (!draw.colors[i]) {
      return false;
    }
    if (std::find(draw.colors.begin(), draw.colors.begin() + i, draw.colors[i]) !=
        draw.colors.begin() + i) {
      draw.colors[i].reset();
      continue;
    }
    bound_rt[i] = true;
    bound_rt_tile[i] = rt_tile;
    bound_rt_64bpp[i] = rt_is_64bpp;
    any_blend_enabled |= blend->blendEnable != VK_FALSE;
    input.blend[i] = *blend;
    input.color_formats[i] = format;
    input.system.color_exp_bias[i] = std::ldexp(1.0f, target.color_exp_bias);
    input.color_count = i + 1;
    if (target.color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
      input.system.flags |= Translator::kSysFlag_ConvertColor0ToGamma << i;
    }
  }
  if (!input.color_count && !use_depth) {
    return true;
  }
  draw.color_count = input.color_count;
  if (use_depth) {
    draw.depth = FindTarget(layout.DecodeBase(depth_info.value), dimensions.surface_pitch,
                            dimensions.msaa_samples, input.depth_format,
                            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (!draw.depth) {
      return false;
    }
  }
  draw.pass = std::make_unique<DrawPass>(*vulkan_device_);
  const uint32_t shared_count = uint32_t(1) << translator_->GetSharedMemoryStorageBufferCountLog2();
  if (!draw.pass->Initialize(input, indices, vertex, pixel, shared_count, pipeline_cache_,
                             pipelines_, draw_bindings_, *draw_uploads_, next_submission_,
                             use_shared_vertices ? shared_vertex_buffer_ : VK_NULL_HANDLE)) {
    return false;
  }
  pending_.emplace_back(std::move(draw));
  // The draw, including its texture uploads, will be recorded, so later draws in
  // the same batch can sample these cached images without copying them again.
  for (const auto& texture : input.textures) {
    if (texture.upload_cached && texture.cached_texture) {
      texture.cached_texture->uploaded = true;
      if (texture.resolved_cube_face_versions[0] != 0) {
        texture.cached_texture->resolved_face_versions = texture.resolved_cube_face_versions;
      }
      if (texture.resolved_crop_version != 0) {
        texture.cached_texture->resolved_crop_version = texture.resolved_crop_version;
      }
    }
  }

  // D3D9 clears whole EDRAM tiles of 1x/2x depth and color targets by issuing a
  // 4x-MSAA depth-only draw at the same tile with half the horizontal pitch.
  // Replay it on every aliased target that already exists, and remember it so
  // targets created later start out cleared instead of uninitialized.
  const bool aliased_4x_depth_clear =
      !use_color && use_depth && dimensions.msaa_samples == xenos::MsaaSamples::k4X && !ps;
  const bool aliased_4x_color_depth_clear = IsAliased4xColorDepthClear(
      use_color, pixel != nullptr, !input.textures.empty(), prim_type, dimensions.msaa_samples,
      regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable != 0, any_blend_enabled, use_depth,
      depth_control);
  const uint32_t depth_tile = layout.DecodeBase(depth_info.value);
  const uint32_t pitch_tiles_4x =
      xenos::GetSurfacePitchTiles(dimensions.surface_pitch, xenos::MsaaSamples::k4X, false);
  // Collect matching aliases first so replaying never mutates targets_ during iteration.
  const auto collect_aliases = [&](auto&& predicate) {
    std::vector<std::pair<std::shared_ptr<ColorImage>, xenos::MsaaSamples>> aliases;
    for (auto& [target_key, entry] : targets_) {
      const uint32_t tile = std::get<0>(target_key);
      const uint32_t pitch = std::get<1>(target_key);
      const xenos::MsaaSamples samples = std::get<2>(target_key);
      const VkFormat fmt = std::get<3>(target_key);
      if (samples >= xenos::MsaaSamples::k4X || !predicate(tile, pitch, samples, fmt, *entry.image)) {
        continue;
      }
      entry.last_use_submission = next_submission_;
      aliases.emplace_back(entry.image, samples);
    }
    return aliases;
  };
  if (aliased_4x_depth_clear && pitch_tiles_4x) {
    PendingFastClear clear;
    clear.input = input;
    clear.indices = indices;
    clear.vertex = &vertex;
    clear.depth_format = depth_info.depth_format;
    clear.color_clear_capable =
        IsEdramDepthClearToColor(depth_info.depth_format, depth_control, front.stencilwritemask);
    clear.clear_stencil =
        depth_control.stencilzpass == xenos::StencilOp::kReplace ? front.stencilref : 0u;
    const auto aliases = collect_aliases([&](uint32_t tile, uint32_t pitch,
                                             xenos::MsaaSamples samples, VkFormat,
                                             const ColorImage& img) {
      if (tile != depth_tile ||
          xenos::GetSurfacePitchTiles(pitch, samples, false) != pitch_tiles_4x) {
        return false;
      }
      const bool depth_target =
          (img.aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 && img.format() == input.depth_format;
      const bool color_target = (img.aspects() & VK_IMAGE_ASPECT_COLOR_BIT) != 0 &&
                                clear.color_clear_capable &&
                                GetEdramClearColorFormatMode(img.format()).has_value();
      return depth_target || color_target;
    });
    for (const auto& [image, samples] : aliases) {
      ReplayAliasedFastClear(clear, image, samples);
    }
    pending_fast_clears_.insert_or_assign({depth_tile, pitch_tiles_4x}, std::move(clear));
    if (clear.color_clear_capable) {
      ErasePendingColorFastClears(depth_tile, pitch_tiles_4x);
    }
  } else if (aliased_4x_color_depth_clear) {
    if (use_depth && pitch_tiles_4x) {
      const Shader::Translation* depth_vs = &vertex;
      if (interpolators != 0) {
        auto depth_vs_mod = vertex_modification;
        depth_vs_mod.vertex.interpolator_mask = 0;
        auto* candidate_depth_vs = vs.GetOrCreateTranslation(depth_vs_mod.value);
        if (candidate_depth_vs &&
            (candidate_depth_vs->is_translated() ||
             translator_->TranslateAnalyzedShader(*candidate_depth_vs)) &&
            candidate_depth_vs->is_valid()) {
          depth_vs = candidate_depth_vs;
        }
      }
      PendingFastClear clear;
      clear.input = input;
      clear.input.color_count = 0;
      clear.input.color_formats.fill(VK_FORMAT_UNDEFINED);
      clear.input.blend = {};
      clear.input.float_constants[1].clear();
      clear.input.system.flags &= ~(uint32_t(0xF) << Translator::kSysFlag_ConvertColor0ToGamma_Shift);
      clear.indices = indices;
      clear.vertex = depth_vs;
      clear.depth_format = depth_info.depth_format;
      clear.color_clear_capable = false;
      clear.clear_stencil = 0;
      const auto depth_aliases = collect_aliases([&](uint32_t tile, uint32_t pitch,
                                                     xenos::MsaaSamples samples, VkFormat,
                                                     const ColorImage& img) {
        return tile == depth_tile &&
               xenos::GetSurfacePitchTiles(pitch, samples, false) == pitch_tiles_4x &&
               (img.aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
               img.format() == input.depth_format;
      });
      for (const auto& [image, samples] : depth_aliases) {
        ReplayAliasedFastClear(clear, image, samples);
      }
      pending_fast_clears_.insert_or_assign({depth_tile, pitch_tiles_4x}, std::move(clear));
    }
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (!bound_rt[i]) {
        continue;
      }
      const uint32_t rt_tile = bound_rt_tile[i];
      const bool rt_is_64bpp = bound_rt_64bpp[i];
      const VkFormat rt_format = input.color_formats[i];
      const uint32_t rt_pitch_tiles_4x =
          xenos::GetSurfacePitchTiles(dimensions.surface_pitch, xenos::MsaaSamples::k4X, rt_is_64bpp);
      if (!rt_pitch_tiles_4x) {
        continue;
      }
      PendingColorFastClear color_clear;
      color_clear.input = input;
      color_clear.input.depth_format = VK_FORMAT_UNDEFINED;
      color_clear.input.depth_stencil = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      color_clear.input.color_count = i + 1;
      for (uint32_t j = 0; j < xenos::kMaxColorRenderTargets; ++j) {
        if (j != i) {
          color_clear.input.color_formats[j] = VK_FORMAT_UNDEFINED;
          color_clear.input.blend[j] = {};
        }
      }
      color_clear.indices = indices;
      color_clear.vertex = &vertex;
      color_clear.pixel = pixel;
      color_clear.rt_index = i;
      const auto color_aliases = collect_aliases([&](uint32_t tile, uint32_t pitch,
                                                     xenos::MsaaSamples samples, VkFormat fmt,
                                                     const ColorImage& img) {
        return tile == rt_tile && fmt == rt_format &&
               (img.aspects() & VK_IMAGE_ASPECT_COLOR_BIT) != 0 &&
               xenos::GetSurfacePitchTiles(pitch, samples, rt_is_64bpp) == rt_pitch_tiles_4x;
      });
      for (const auto& [image, samples] : color_aliases) {
        ReplayAliasedColorFastClear(color_clear, image, samples);
      }
      pending_color_fast_clears_.insert_or_assign({rt_tile, rt_pitch_tiles_4x, rt_format},
                                                  std::move(color_clear));
      if (!rt_is_64bpp) {
        if (auto fc = pending_fast_clears_.find({rt_tile, rt_pitch_tiles_4x});
            fc != pending_fast_clears_.end()) {
          fc->second.color_clear_capable = false;
        }
      }
    }
  } else if (!aliased_4x_depth_clear && !aliased_4x_color_depth_clear) {
    // Any ordinary draw at a tile consumes its pending fast clear.
    const uint32_t pitch_tiles =
        xenos::GetSurfacePitchTiles(dimensions.surface_pitch, dimensions.msaa_samples, false);
    if (pitch_tiles && use_depth) {
      pending_fast_clears_.erase({depth_tile, pitch_tiles});
    }
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if ((writes >> (i * 4)) & 15) {
        const auto target =
            regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
        const uint32_t rt_tile = layout.DecodeBase(target.value);
        if (pitch_tiles) {
          pending_fast_clears_.erase({rt_tile, pitch_tiles});
        }
        const uint32_t rt_pitch_tiles = xenos::GetSurfacePitchTiles(
            dimensions.surface_pitch, dimensions.msaa_samples,
            xenos::IsColorRenderTargetFormat64bpp(target.color_format));
        if (rt_pitch_tiles) {
          ErasePendingColorFastClears(rt_tile, rt_pitch_tiles);
        }
      }
    }
  }
  return true;
}

VkFormat NativeCommandProcessor::GetDepthFormat(xenos::DepthRenderTargetFormat format) const {
  return SelectDepthFormat(format, depth_unorm24_supported_);
}

DrawInput NativeCommandProcessor::RescaleFastClearInput(const DrawInput& base,
                                                        xenos::MsaaSamples samples) const {
  // The 4x-MSAA clear covers the aliased target with half the horizontal pitch;
  // scale the guest-space viewport and scissor to the host pixels of the alias.
  const auto scale = GetMsaa4xAliasScale(samples);
  const auto& properties = vulkan_device_->properties();
  std::array<float, 3> ndc_scale;
  std::array<float, 3> ndc_offset;
  std::copy(std::begin(base.viewport.ndc_scale), std::end(base.viewport.ndc_scale),
            ndc_scale.begin());
  std::copy(std::begin(base.viewport.ndc_offset), std::end(base.viewport.ndc_offset),
            ndc_offset.begin());

  DrawInput input = base;
  input.samples = VkSampleCountFlagBits(1u << uint32_t(samples));
  constexpr uint32_t kMsaaSamplesFlagMask =
      ((1u << xenos::kMsaaSamplesBits) - 1u) << Translator::kSysFlag_MsaaSamples_Shift;
  input.system.flags = (base.system.flags & ~kMsaaSamplesFlagMask) |
                       (uint32_t(samples) << Translator::kSysFlag_MsaaSamples_Shift);
  ScaleViewportAndScissor(input.viewport, input.scissor, ndc_scale, ndc_offset, scale.x, scale.y,
                          properties.maxViewportDimensions[0], properties.maxViewportDimensions[1]);
  std::copy(ndc_scale.begin(), ndc_scale.end(), input.viewport.ndc_scale);
  std::copy(ndc_offset.begin(), ndc_offset.end(), input.viewport.ndc_offset);
  std::copy(ndc_scale.begin(), ndc_scale.end(), input.system.ndc_scale);
  std::copy(ndc_offset.begin(), ndc_offset.end(), input.system.ndc_offset);
  return input;
}

void NativeCommandProcessor::ReplayAliasedFastClear(const PendingFastClear& clear,
                                                    const std::shared_ptr<ColorImage>& image,
                                                    xenos::MsaaSamples samples) {
  const bool depth_target = (image->aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
  std::optional<uint32_t> clear_format;
  if (!depth_target) {
    clear_format = GetEdramClearColorFormatMode(image->format());
    if (!clear_format) {
      return;
    }
  }
  DrawInput input = RescaleFastClearInput(clear.input, samples);
  QueuedDraw draw{};
  if (!depth_target) {
    // The clear writes a 32-bit EDRAM depth-stencil word into an aliased color
    // target; the fragment shader unpacks it for the target format.
    input.color_count = 1;
    input.color_formats[0] = image->format();
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_FALSE;
    blend.colorWriteMask = 0xF;
    input.blend[0] = blend;
    input.depth_format = VK_FORMAT_UNDEFINED;
    input.depth_stencil = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    input.edram_depth_clear_color = true;
    input.edram_clear_stencil = clear.clear_stencil;
    input.edram_clear_format = *clear_format;
    draw.colors[0] = image;
    draw.color_count = 1;
  } else {
    draw.depth = image;
  }
  draw.pass = std::make_unique<DrawPass>(*vulkan_device_);
  const uint32_t shared_count = uint32_t(1) << translator_->GetSharedMemoryStorageBufferCountLog2();
  if (!draw.pass->Initialize(input, clear.indices, *clear.vertex, nullptr, shared_count,
                             pipeline_cache_, pipelines_, draw_bindings_, *draw_uploads_,
                             next_submission_)) {
    return;
  }
  pending_.emplace_back(std::move(draw));
}

void NativeCommandProcessor::ApplyPendingFastClear(uint32_t tile, uint32_t pitch_tiles,
                                                   xenos::MsaaSamples samples,
                                                   const std::shared_ptr<ColorImage>& image) {
  if (samples >= xenos::MsaaSamples::k4X) {
    return;
  }
  const auto pending = pending_fast_clears_.find({tile, pitch_tiles});
  if (pending == pending_fast_clears_.end()) {
    return;
  }
  const PendingFastClear& clear = pending->second;
  const bool depth_target = (image->aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
                            image->format() == GetDepthFormat(clear.depth_format);
  const bool color_target =
      (image->aspects() & VK_IMAGE_ASPECT_COLOR_BIT) != 0 && clear.color_clear_capable &&
      GetEdramClearColorFormatMode(image->format()).has_value();
  if (depth_target || color_target) {
    ReplayAliasedFastClear(clear, image, samples);
  }
}

void NativeCommandProcessor::ReplayAliasedColorFastClear(const PendingColorFastClear& clear,
                                                         const std::shared_ptr<ColorImage>& image,
                                                         xenos::MsaaSamples samples) {
  const DrawInput input = RescaleFastClearInput(clear.input, samples);
  QueuedDraw draw{};
  draw.colors[clear.rt_index] = image;
  draw.color_count = clear.rt_index + 1;
  draw.pass = std::make_unique<DrawPass>(*vulkan_device_);
  const uint32_t shared_count = uint32_t(1) << translator_->GetSharedMemoryStorageBufferCountLog2();
  if (!draw.pass->Initialize(input, clear.indices, *clear.vertex, clear.pixel, shared_count,
                             pipeline_cache_, pipelines_, draw_bindings_, *draw_uploads_,
                             next_submission_)) {
    return;
  }
  pending_.emplace_back(std::move(draw));
}

void NativeCommandProcessor::ApplyPendingColorFastClear(uint32_t tile, uint32_t pitch_tiles,
                                                         xenos::MsaaSamples samples,
                                                         VkFormat format,
                                                         const std::shared_ptr<ColorImage>& image) {
  if (samples >= xenos::MsaaSamples::k4X || !pitch_tiles) {
    return;
  }
  const auto pending = pending_color_fast_clears_.find({tile, pitch_tiles, format});
  if (pending == pending_color_fast_clears_.end()) {
    return;
  }
  ReplayAliasedColorFastClear(pending->second, image, samples);
}

void NativeCommandProcessor::ErasePendingColorFastClears(uint32_t tile, uint32_t pitch_tiles) {
  for (auto it = pending_color_fast_clears_.lower_bound({tile, pitch_tiles, VK_FORMAT_UNDEFINED});
       it != pending_color_fast_clears_.end() && std::get<0>(it->first) == tile &&
       std::get<1>(it->first) == pitch_tiles;) {
    it = pending_color_fast_clears_.erase(it);
  }
}

std::shared_ptr<ColorImage> NativeCommandProcessor::FindTarget(uint32_t tile, uint32_t pitch,
                                                               xenos::MsaaSamples samples,
                                                               VkFormat format,
                                                               VkImageAspectFlags aspects,
                                                               bool is_64bpp) {
  const auto key = std::tuple(tile, pitch, samples, format);
  if (auto existing = targets_.find(key); existing != targets_.end()) {
    existing->second.last_use_submission = next_submission_;
    return existing->second.image;
  }
  if (!pitch) {
    return {};
  }
  if (targets_.size() >= kMaxTargets) {
    // Evict the least recently used target whose GPU work has completed. Only
    // submissions up to the last retired fence are eligible, so queued and
    // in-flight draws keep their images, and the presented image is kept too.
    const auto victim = SelectRetiredEviction(
        targets_, retired_submission_ + 1,
        [this](const auto&, const CachedImage& entry) { return entry.image != presented_; });
    if (!victim) {
      return {};
    }
    targets_.erase(*victim);
  }
  const auto& properties = vulkan_device_->properties();
  const uint32_t max_width = std::min(properties.maxImageDimension2D,
                                      properties.maxFramebufferWidth);
  const uint32_t max_height = std::min(properties.maxImageDimension2D,
                                       properties.maxFramebufferHeight);
  if (uint64_t(pitch) * draw_resolution_scale_x_ > max_width) {
    return {};
  }
  const uint32_t height =
      GetTargetHeight(pitch, samples, ActiveEdramLayout().tile_count,
                      max_height / draw_resolution_scale_y_, is_64bpp);
  auto image = std::make_shared<ColorImage>(*vulkan_device_);
  if (!image->Initialize({pitch * draw_resolution_scale_x_, height * draw_resolution_scale_y_},
                         format, VkSampleCountFlagBits(1u << uint32_t(samples)), aspects)) {
    return {};
  }
  // shortcut: separate host images, share overlapping tiles when ownership tracking is added.
  targets_.emplace(key, CachedImage{image, next_submission_});
  if (!is_64bpp) {
    ApplyPendingFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, false), samples, image);
  }
  if ((aspects & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
    ApplyPendingColorFastClear(tile, xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp), samples,
                               format, image);
  }
  return image;
}

bool NativeCommandProcessor::IssueCopy() {
  if (!FlushPendingIfFull()) {
    return false;
  }
  const auto& regs = *register_file_;
  const EdramLayout layout = ActiveEdramLayout();
  draw_util::ResolveInfo copy;
  if (!draw_util::GetResolveInfo(regs, *memory_, draw_resolution_scale_x_,
                                 draw_resolution_scale_y_, false, false, layout, copy)) {
    return false;
  }
  if (!copy.coordinate_info.width_div_8 || !copy.height_div_8) {
    return true;
  }
  const auto surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  std::optional<QueuedDepthClear> depth_clear;
  if (copy.IsClearingDepth()) {
    const auto depth_format = xenos::DepthRenderTargetFormat(copy.depth_edram_info.format);
    auto depth = FindTarget(copy.depth_original_base, surface_info.surface_pitch,
                            surface_info.msaa_samples, GetDepthFormat(depth_format),
                            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (!depth) {
      return false;
    }
    const auto extent = depth->extent();
    const auto rect = GetResolveRectangle(copy, layout, extent.width, extent.height, true,
                                          draw_resolution_scale_x_, draw_resolution_scale_y_);
    if (!rect) {
      return false;
    }
    const uint32_t packed_depth = copy.rb_depth_clear >> 8;
    const float depth_value = depth_format == xenos::DepthRenderTargetFormat::kD24FS8
                                  ? xenos::Float20e4To32(packed_depth) * 0.5f
                                  : float(packed_depth) / 16777215.0f;
    depth_clear =
        QueuedDepthClear{std::move(depth),
                         {{int32_t(rect->x), int32_t(rect->y)}, {rect->width, rect->height}},
                         {depth_value, copy.rb_depth_clear & 255}};
  }
  const auto samples = surface_info.msaa_samples;
  const auto selection = copy.copy_dest_coordinate_info.copy_sample_select;
  const bool average_all =
      cmd_resolve_image_ &&
      ((samples == xenos::MsaaSamples::k2X && selection == xenos::CopySampleSelect::k01) ||
       (samples == xenos::MsaaSamples::k4X && selection == xenos::CopySampleSelect::k0123));
  const bool depth = copy.IsCopyingDepth();
  const VkFormat color_format =
      ColorTargetFormat(xenos::ColorRenderTargetFormat(copy.color_edram_info.format));
  const bool single_sample = !depth && samples != xenos::MsaaSamples::k1X &&
                             color_format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                             xenos::IsSingleCopySampleSelected(selection);
  const VkFormat output_format =
      depth ? VK_FORMAT_R32_SFLOAT : CopyColorFormat(copy.copy_dest_info.copy_dest_format);
  const uint32_t resolve_address = regs[XE_GPU_REG_RB_COPY_DEST_BASE];
  const bool format_convert =
      !depth &&
      IsResolveFormatConvertSupported(copy, samples,
                                      xenos::ColorRenderTargetFormat(copy.color_edram_info.format),
                                      copy.copy_dest_info.copy_dest_format) &&
      (output_format == VK_FORMAT_R8_UNORM ? r8_resolve_blit_ : rg8_resolve_blit_);
  if ((!depth &&
       ((samples != xenos::MsaaSamples::k1X && !average_all && !single_sample) ||
        color_format == VK_FORMAT_UNDEFINED || (color_format != output_format && !format_convert) ||
        copy.copy_dest_info.copy_dest_exp_bias)) ||
      copy.copy_dest_info.copy_dest_array || !IsPhysicalRange(resolve_address, 1) ||
      copy.copy_dest_extent_start < resolve_address ||
      !IsPhysicalRange(copy.copy_dest_extent_start, copy.copy_dest_extent_length)) {
    if (depth_clear) {
      pending_.emplace_back(std::move(*depth_clear));
    }
    return false;
  }
  const auto depth_format = xenos::DepthRenderTargetFormat(copy.depth_edram_info.format);
  const bool color_is_64bpp = !depth && xenos::IsColorRenderTargetFormat64bpp(
                                            xenos::ColorRenderTargetFormat(copy.color_edram_info.format));
  auto source = depth ? FindTarget(copy.depth_original_base, surface_info.surface_pitch, samples,
                                   GetDepthFormat(depth_format),
                                   VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                      : FindTarget(copy.color_original_base, surface_info.surface_pitch, samples,
                                   color_format, VK_IMAGE_ASPECT_COLOR_BIT, color_is_64bpp);
  if (!source) {
    return false;
  }
  const uint32_t copy_pitch_tiles =
      xenos::GetSurfacePitchTiles(surface_info.surface_pitch, samples, false);
  if (copy_pitch_tiles) {
    if (copy.IsClearingDepth() || depth) {
      pending_fast_clears_.erase({copy.depth_original_base, copy_pitch_tiles});
    }
    if (copy.IsClearingColor() || !depth) {
      pending_fast_clears_.erase({copy.color_original_base, copy_pitch_tiles});
      const uint32_t color_copy_pitch_tiles =
          xenos::GetSurfacePitchTiles(surface_info.surface_pitch, samples, color_is_64bpp);
      if (color_copy_pitch_tiles) {
        ErasePendingColorFastClears(copy.color_original_base, color_copy_pitch_tiles);
      }
    }
  }
  const auto source_extent = source->extent();
  const auto rect =
      GetResolveRectangle(copy, layout, source_extent.width, source_extent.height, depth,
                          draw_resolution_scale_x_, draw_resolution_scale_y_);
  if (!rect) {
    return false;
  }
  const VkExtent2D extent{rect->width, rect->height};

  const uint32_t dest_info = copy.copy_dest_info.value;
  const auto dest_pitch_reg = regs.Get<reg::RB_COPY_DEST_PITCH>();
  const uint32_t dest_pitch = dest_pitch_reg.value;
  const uint32_t guest_width =
      uint32_t(copy.coordinate_info.width_div_8) << xenos::kResolveAlignmentPixelsLog2;
  const uint32_t guest_height = copy.height_div_8 << xenos::kResolveAlignmentPixelsLog2;
  const VkExtent2D target_extent = GetResolveDestinationExtent(
      guest_width, guest_height, dest_pitch_reg.copy_dest_pitch,
      dest_pitch_reg.copy_dest_height, draw_resolution_scale_x_, draw_resolution_scale_y_);
  bool tiled_destination =
      target_extent.width != extent.width || target_extent.height != extent.height;
  const bool has_offset = copy.copy_dest_x || copy.copy_dest_y;
  auto existing = resolved_images_.find(resolve_address);
  std::shared_ptr<ColorImage> destination;
  VkOffset2D destination_offset{};
  uint32_t dest_resolve_address = resolve_address;
  uint32_t dest_pitch_to_store = dest_pitch;
  if (existing != resolved_images_.end() && existing->second.image->format() == output_format) {
    const auto existing_extent = existing->second.image->extent();
    if (!has_offset && existing_extent.width == target_extent.width &&
        existing_extent.height == target_extent.height) {
      destination = existing->second.image;
    } else if (has_offset && existing->second.dest_info == dest_info &&
               reg::RB_COPY_DEST_PITCH{existing->second.dest_pitch}.copy_dest_pitch ==
                   dest_pitch_reg.copy_dest_pitch) {
      const auto placed =
          GetCachedResolveRectangle(copy, existing_extent.width, existing_extent.height,
                                    draw_resolution_scale_x_, draw_resolution_scale_y_);
      if (placed) {
        destination = existing->second.image;
        destination_offset = {int32_t(placed->x), int32_t(placed->y)};
        dest_pitch_to_store = existing->second.dest_pitch;
        tiled_destination |= existing->second.tiled_destination;
      }
    }
  }
  if (!destination && dest_pitch_reg.copy_dest_pitch != 0) {
    const uint32_t bpp_log2 = GetResolveBytesPerBlockLog2(output_format);
    uint64_t best_version = 0;
    for (const auto& [base_addr, entry] : resolved_images_) {
      if (!entry.tiled_destination || base_addr >= resolve_address ||
          entry.last_use_submission + 4 < next_submission_ ||
          entry.image->format() != output_format || entry.dest_info != dest_info ||
          reg::RB_COPY_DEST_PITCH{entry.dest_pitch}.copy_dest_pitch !=
              dest_pitch_reg.copy_dest_pitch) {
        continue;
      }
      uint32_t tile_x = 0;
      uint32_t tile_y = 0;
      const uint32_t delta = resolve_address - base_addr;
      if (!TryGetTiledResolveOffset2D(delta, dest_pitch_reg.copy_dest_pitch, bpp_log2,
                                      tile_x, tile_y)) {
        continue;
      }
      if (texture_util::GetTiledOffset2D(int32_t(tile_x), int32_t(tile_y),
                                         dest_pitch_reg.copy_dest_pitch, bpp_log2) !=
          int32_t(delta)) {
        continue;
      }
      const uint64_t dest_x = uint64_t(tile_x + copy.copy_dest_x) * draw_resolution_scale_x_;
      const uint64_t dest_y = uint64_t(tile_y + copy.copy_dest_y) * draw_resolution_scale_y_;
      const auto entry_extent = entry.image->extent();
      if (dest_x > entry_extent.width || dest_y > entry_extent.height ||
          extent.width > entry_extent.width - dest_x ||
          extent.height > entry_extent.height - dest_y) {
        continue;
      }
      if (!destination || entry.version > best_version) {
        destination = entry.image;
        destination_offset = {int32_t(dest_x), int32_t(dest_y)};
        dest_resolve_address = base_addr;
        dest_pitch_to_store = entry.dest_pitch;
        tiled_destination = true;
        best_version = entry.version;
      }
    }
  }
  if (!destination) {
    if (has_offset) {
      // A sub-rectangle that cannot be placed in an existing resolve would need
      // guest writeback, which this path does not provide.
      if (depth_clear) {
        pending_.emplace_back(std::move(*depth_clear));
      }
      return false;
    }
    if (existing == resolved_images_.end() && resolved_images_.size() >= kMaxResolvedImages) {
      // Drop the least recently used resolve whose GPU work has completed. Only
      // retired submissions are eligible, and the presented image is kept; the
      // shared owners in pending_ / in_flight_ keep surviving images alive.
      const auto victim = SelectRetiredEviction(resolved_images_, retired_submission_ + 1,
                                                [this](uint32_t address, const CachedImage& entry) {
                                                  return entry.image != presented_ &&
                                                         address != presented_resolve_address_;
                                                });
      if (!victim) {
        return false;
      }
      resolved_images_.erase(*victim);
      PruneResolvedDependents();
      RebuildCachedTexturePages();
    }
    destination = std::make_shared<ColorImage>(*vulkan_device_);
    if (!destination->Initialize(target_extent, output_format)) {
      return false;
    }
  }
  const size_t depth_descriptor = depth ? depth_resolve_->Allocate() : SIZE_MAX;
  const size_t color_descriptor = single_sample ? color_resolve_->Allocate() : SIZE_MAX;
  if ((depth && depth_descriptor == SIZE_MAX) || (single_sample && color_descriptor == SIZE_MAX)) {
    return false;
  }
  InvalidateTextures(copy.copy_dest_extent_start, copy.copy_dest_extent_length);
  bool erased_other_resolved = false;
  for (auto it = resolved_images_.begin(); it != resolved_images_.end();) {
    if (it->first != dest_resolve_address &&
        RangesOverlap(copy.copy_dest_extent_start, copy.copy_dest_extent_length,
                      it->second.guest_start, it->second.guest_length)) {
      if (presented_resolve_address_ && *presented_resolve_address_ == it->first) {
        presented_resolve_address_.reset();
      }
      it = resolved_images_.erase(it);
      erased_other_resolved = true;
    } else {
      ++it;
    }
  }
  uint32_t guest_start = copy.copy_dest_extent_start;
  uint32_t guest_end = copy.copy_dest_extent_start + copy.copy_dest_extent_length;
  if (auto prev = resolved_images_.find(dest_resolve_address);
      prev != resolved_images_.end() && prev->second.image == destination &&
      prev->second.guest_length) {
    guest_start = std::min(guest_start, prev->second.guest_start);
    guest_end = std::max(guest_end, prev->second.guest_start + prev->second.guest_length);
  }
  const uint32_t guest_length = guest_end - guest_start;
  const bool rgba8_readback_eligible =
      !depth && output_format == ColorImage::kFormat &&
      copy.copy_dest_info.copy_dest_format == xenos::ColorFormat::k_8_8_8_8 &&
      copy.copy_dest_info.copy_dest_number ==
          xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction &&
      !copy.copy_dest_info.copy_dest_swap && uint32_t(copy.copy_dest_info.copy_dest_endian) <= 3;
  const uint32_t copy_dest_pitch = regs.Get<reg::RB_COPY_DEST_PITCH>().copy_dest_pitch;
  const ResolveRectangle guest_rect{
      copy.copy_dest_x, copy.copy_dest_y,
      uint32_t(copy.coordinate_info.width_div_8) << xenos::kResolveAlignmentPixelsLog2,
      copy.height_div_8 << xenos::kResolveAlignmentPixelsLog2};
  CachedResolve resolve_entry{{destination, next_submission_},
                              dest_info,
                              dest_pitch_to_store,
                              next_resolve_version_++,
                              tiled_destination,
                              guest_start,
                              guest_length,
                              guest_rect,
                              copy_dest_pitch,
                              copy.copy_dest_info.copy_dest_endian,
                              rgba8_readback_eligible};
  resolved_images_.insert_or_assign(dest_resolve_address, std::move(resolve_entry));
  if (erased_other_resolved) {
    PruneResolvedDependents();
    RebuildCachedTexturePages();
  } else if (!cached_texture_pages_.empty()) {
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    const auto [first_page, page_count] = GetPhysicalPageRange(guest_start, guest_length);
    SetPageRangeBits(cached_texture_pages_, first_page, page_count);
  }
  memory_->EnablePhysicalMemoryAccessCallbacks(copy.copy_dest_extent_start,
                                               copy.copy_dest_extent_length, true, false);
  const size_t resolve_index = pending_.size();
  const uint32_t depth_sample = depth ? GetResolveSampleIndex(samples, selection) : 0;
  pending_.emplace_back(
      QueuedResolve{std::move(source),
                    std::move(destination),
                    {int32_t(rect->x), int32_t(rect->y)},
                    destination_offset,
                    extent,
                    copy.IsClearingColor(),
                    UnpackColor(color_format, copy.rb_color_clear, copy.rb_color_clear_lo),
                    depth_descriptor,
                    depth_sample,
                    depth_format == xenos::DepthRenderTargetFormat::kD24FS8 ? 2.0f : 1.0f,
                    color_descriptor,
                    single_sample ? GetResolveSampleIndex(samples, selection) : 0,
                    format_convert});
  if (depth_clear) {
    pending_.emplace_back(std::move(*depth_clear));
  }
  // shortcut: immediate RGBA8 copies only, add delayed modes and other layouts with native packing.
  if (rgba8_readback_eligible) {
    if (GetReadbackResolveMode(false) == ReadbackResolveMode::kFull &&
        IsResolveReadbackWanted(copy.copy_dest_extent_start, copy.copy_dest_extent_length)) {
      return ReadBackResolve(copy, resolve_address, copy_dest_pitch,
                             std::get<QueuedResolve>(pending_[resolve_index]));
    }
    NoteResolvedRange(copy.copy_dest_extent_start, copy.copy_dest_extent_length, 32,
                      draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1);
  }
  return true;
}

bool NativeCommandProcessor::CaptureResolveImage(ColorImage& image, VkOffset2D src_offset,
                                                 VkExtent2D src_extent,
                                                 std::vector<uint8_t>& pixels_out) {
  const bool scaled = draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1;
  const VkExtent2D guest_extent{src_extent.width / draw_resolution_scale_x_,
                                src_extent.height / draw_resolution_scale_y_};
  if (scaled && (!cmd_blit_image_ || !guest_extent.width || !guest_extent.height)) {
    return false;
  }
  std::optional<ColorImage> downsampled;
  if (scaled) {
    downsampled.emplace(*vulkan_device_);
    if (!downsampled->Initialize(guest_extent, ColorImage::kFormat)) {
      return false;
    }
  }
  auto readback = std::make_unique<ImageReadback>(*vulkan_device_);
  if (!readback->Initialize(scaled ? VkRect2D{{0, 0}, guest_extent}
                                   : VkRect2D{src_offset, src_extent}) ||
      !BeginCommands()) {
    return false;
  }
  RecordPending();
  image.Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  bool recorded = false;
  if (scaled) {
    downsampled->Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0] = {src_offset.x, src_offset.y, 0};
    blit.srcOffsets[1] = {src_offset.x + int32_t(src_extent.width),
                          src_offset.y + int32_t(src_extent.height), 1};
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {int32_t(guest_extent.width), int32_t(guest_extent.height), 1};
    cmd_blit_image_(command_buffer_, image.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    downsampled->image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                    VK_FILTER_LINEAR);
    downsampled->Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    recorded = readback->Record(command_buffer_, downsampled->image(), downsampled->extent());
  } else {
    recorded = readback->Record(command_buffer_, image.image(), image.extent());
  }
  image.Transition(command_buffer_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT);
  if (!SubmitCommands()) {
    return false;
  }
  in_flight_readback_ = std::move(readback);
  if (!RetireSubmission()) {
    return false;
  }
  const auto mapped = in_flight_readback_->Map(fence_);
  if (!recorded || mapped.empty()) {
    in_flight_readback_.reset();
    return false;
  }
  pixels_out.assign(mapped.begin(), mapped.end());
  in_flight_readback_.reset();
  return true;
}

bool NativeCommandProcessor::ReadBackResolve(const draw_util::ResolveInfo& copy, uint32_t address,
                                             uint32_t pitch, const QueuedResolve& resolve) {
  const uint32_t end = copy.copy_dest_extent_start + copy.copy_dest_extent_length;
  const auto access = memory_->GetPhysicalHeap()->QueryRangeAccess(address, end - 1);
  if (access != memory::PageAccess::kReadWrite && access != memory::PageAccess::kExecuteReadWrite) {
    return true;
  }
  std::vector<uint8_t> pixels;
  if (!CaptureResolveImage(*resolve.destination, resolve.destination_offset, resolve.extent,
                           pixels)) {
    return false;
  }
  const ResolveRectangle rect{copy.copy_dest_x, copy.copy_dest_y,
                              resolve.extent.width / draw_resolution_scale_x_,
                              resolve.extent.height / draw_resolution_scale_y_};
  const bool stored = StoreTiledResolveRgba8({memory_->TranslatePhysical(address), end - address},
                                             pitch, rect, copy.copy_dest_info.copy_dest_endian,
                                             pixels);
  if (stored) {
    // The CPU wrote guest memory directly, bypassing the access-violation watch,
    // so anything cached from the written range is stale now.
    InvalidateTextures(address, end - address);
    InvalidateSharedVertices(address, end - address);
  }
  return stored;
}

bool NativeCommandProcessor::ReadBackResolvedRange(uint32_t address, uint32_t length,
                                                   uint32_t bits_per_pixel, bool /*scaled*/,
                                                   uint32_t clip_start, uint32_t clip_end) {
  if (!vulkan_device_ || !length || bits_per_pixel != 32) {
    return false;
  }
  DrainMemoryInvalidations();
  const CachedResolve* best = nullptr;
  uint32_t best_base = 0;
  for (const auto& [base_addr, entry] : resolved_images_) {
    if (!entry.rgba8_readback_valid ||
        !RangesOverlap(address, length, entry.guest_start, entry.guest_length)) {
      continue;
    }
    if (!best || base_addr == address || entry.version > best->version) {
      best = &entry;
      best_base = base_addr;
      if (base_addr == address) {
        break;
      }
    }
  }
  if (!best || !best->guest_length || !best->image->initialized()) {
    return false;
  }
  const uint32_t entry_end = best->guest_start + best->guest_length;
  const uint32_t clip_first = std::max(best_base, clip_start);
  const uint32_t clip_last = std::min(entry_end, clip_end);
  if (clip_first >= clip_last || !IsPhysicalRange(best_base, entry_end - best_base)) {
    return false;
  }
  const auto access = memory_->GetPhysicalHeap()->QueryRangeAccess(clip_first, clip_last - 1);
  if (access != memory::PageAccess::kReadWrite && access != memory::PageAccess::kExecuteReadWrite) {
    return false;
  }
  const VkOffset2D src_offset{int32_t(best->last_rect.x * draw_resolution_scale_x_),
                              int32_t(best->last_rect.y * draw_resolution_scale_y_)};
  const VkExtent2D src_extent{best->last_rect.width * draw_resolution_scale_x_,
                              best->last_rect.height * draw_resolution_scale_y_};
  std::vector<uint8_t> pixels;
  if (!CaptureResolveImage(*best->image, src_offset, src_extent, pixels)) {
    return false;
  }
  std::vector<uint8_t> tiled(entry_end - best_base);
  if (!StoreTiledResolveRgba8(tiled, best->copy_dest_pitch, best->last_rect, best->endian,
                              pixels)) {
    return false;
  }
  std::memcpy(memory_->TranslatePhysical(clip_first), tiled.data() + (clip_first - best_base),
              clip_last - clip_first);
  InvalidateTextures(clip_first, clip_last - clip_first);
  InvalidateSharedVertices(clip_first, clip_last - clip_first);
  return true;
}

void NativeCommandProcessor::RecordDraws(std::span<const FrameCommand> commands) {
  const auto& vk = vulkan_device_->functions();
  const auto& draw = std::get<QueuedDraw>(commands.front());
  for (const auto& command : commands) {
    std::get<QueuedDraw>(command).pass->RecordTextureUpload(command_buffer_);
  }
  std::array<VkRenderingAttachmentInfo, xenos::kMaxColorRenderTargets> colors{};
  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering.layerCount = 1;
  rendering.colorAttachmentCount = draw.color_count;
  rendering.pColorAttachments = colors.data();
  bool needs_full_clear = false;
  VkExtent2D attachment_extent{UINT32_MAX, UINT32_MAX};
  for (uint32_t i = 0; i < draw.color_count; ++i) {
    auto& color = colors[i];
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    const auto& image = draw.colors[i];
    if (!image) {
      continue;
    }
    const auto ext = image->extent();
    attachment_extent.width = std::min(attachment_extent.width, ext.width);
    attachment_extent.height = std::min(attachment_extent.height, ext.height);
    color.loadOp = image->initialized() ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    needs_full_clear |= color.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.imageView = image->view();
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    image->Transition(command_buffer_, color.imageLayout,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  }
  if (draw.depth) {
    const auto ext = draw.depth->extent();
    attachment_extent.width = std::min(attachment_extent.width, ext.width);
    attachment_extent.height = std::min(attachment_extent.height, ext.height);
    depth.loadOp =
        draw.depth->initialized() ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    needs_full_clear |= depth.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // Unowned EDRAM starts with zero bits, including depth used by reverse-Z draws.
    depth.clearValue.depthStencil = {0.0f, 0};
    depth.imageView = draw.depth->view();
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    draw.depth->Transition(
        command_buffer_, depth.imageLayout,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    rendering.pDepthAttachment = rendering.pStencilAttachment = &depth;
  }
  if (attachment_extent.width == UINT32_MAX) {
    attachment_extent = {};
  }
  rendering.renderArea.extent = attachment_extent;
  if (!needs_full_clear) {
    VkExtent2D covered{};
    for (const auto& command : commands) {
      const auto& scissor = std::get<QueuedDraw>(command).pass->scissor();
      covered.width = std::max(covered.width, scissor.offset[0] + scissor.extent[0]);
      covered.height = std::max(covered.height, scissor.offset[1] + scissor.extent[1]);
    }
    rendering.renderArea.extent = {std::min(covered.width, attachment_extent.width),
                                   std::min(covered.height, attachment_extent.height)};
  }
  vk.vkCmdBeginRendering(command_buffer_, &rendering);
  for (const auto& command : commands) {
    std::get<QueuedDraw>(command).pass->Record(command_buffer_, attachment_extent);
  }
  vk.vkCmdEndRendering(command_buffer_);
}

void NativeCommandProcessor::RecordDepthClear(const QueuedDepthClear& clear) {
  const auto& vk = vulkan_device_->functions();
  const auto full_extent = clear.target->extent();
  if (!clear.target->initialized() && !CoversFullExtent(clear.area, full_extent)) {
    RecordDepthClear({clear.target, {{0, 0}, full_extent}, {0.0f, 0}});
  }
  clear.target->Transition(
      command_buffer_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depth.imageView = clear.target->view();
  depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depth.clearValue.depthStencil = clear.value;
  VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering.renderArea = clear.area;
  rendering.layerCount = 1;
  rendering.pDepthAttachment = rendering.pStencilAttachment = &depth;
  vk.vkCmdBeginRendering(command_buffer_, &rendering);
  vk.vkCmdEndRendering(command_buffer_);
}

void NativeCommandProcessor::RecordResolve(const QueuedResolve& resolve) {
  if (resolve.depth_descriptor != SIZE_MAX) {
    if (!resolve.source->initialized()) {
      RecordDepthClear({resolve.source, {{0, 0}, resolve.source->extent()}, {0.0f, 0}});
    }
    constexpr auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    resolve.source->Transition(command_buffer_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, compute,
                               VK_ACCESS_SHADER_READ_BIT);
    resolve.destination->Transition(command_buffer_, VK_IMAGE_LAYOUT_GENERAL, compute,
                                    VK_ACCESS_SHADER_WRITE_BIT);
    depth_resolve_->Record(command_buffer_, resolve.depth_descriptor, resolve.source->depth_view(),
                           resolve.destination->depth_view(),
                           {resolve.source_offset, resolve.extent}, resolve.destination_offset,
                           resolve.source->samples() != VK_SAMPLE_COUNT_1_BIT, resolve.depth_sample,
                           resolve.depth_scale);
    return;
  }
  const auto& vk = vulkan_device_->functions();
  if (!resolve.source->initialized()) {
    resolve.source->Clear(command_buffer_, {});
  }
  if (resolve.color_descriptor != SIZE_MAX) {
    constexpr auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    resolve.source->Transition(command_buffer_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, compute,
                               VK_ACCESS_SHADER_READ_BIT);
    resolve.destination->Transition(command_buffer_, VK_IMAGE_LAYOUT_GENERAL, compute,
                                    VK_ACCESS_SHADER_WRITE_BIT);
    color_resolve_->Record(command_buffer_, resolve.color_descriptor, resolve.source->view(),
                           resolve.destination->view(), {resolve.source_offset, resolve.extent},
                           resolve.destination_offset, resolve.color_sample,
                           resolve.destination->format());
  } else {
    constexpr auto transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
    resolve.source->Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, transfer,
                               VK_ACCESS_TRANSFER_READ_BIT);
    resolve.destination->Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, transfer,
                                    VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.srcOffset = {resolve.source_offset.x, resolve.source_offset.y, 0};
    copy.dstOffset = {resolve.destination_offset.x, resolve.destination_offset.y, 0};
    copy.extent = {resolve.extent.width, resolve.extent.height, 1};
    if (resolve.format_convert) {
      VkImageBlit blit{};
      blit.srcSubresource = copy.srcSubresource;
      blit.dstSubresource = copy.dstSubresource;
      blit.srcOffsets[0] = copy.srcOffset;
      blit.srcOffsets[1] = {copy.srcOffset.x + int32_t(copy.extent.width),
                            copy.srcOffset.y + int32_t(copy.extent.height), 1};
      blit.dstOffsets[0] = copy.dstOffset;
      blit.dstOffsets[1] = {copy.dstOffset.x + int32_t(copy.extent.width),
                            copy.dstOffset.y + int32_t(copy.extent.height), 1};
      cmd_blit_image_(command_buffer_, resolve.source->image(),
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, resolve.destination->image(),
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    } else if (resolve.source->samples() == VK_SAMPLE_COUNT_1_BIT) {
      vk.vkCmdCopyImage(command_buffer_, resolve.source->image(),
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, resolve.destination->image(),
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    } else {
      const VkImageResolve region{copy.srcSubresource, copy.srcOffset, copy.dstSubresource,
                                  copy.dstOffset, copy.extent};
      cmd_resolve_image_(command_buffer_, resolve.source->image(),
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, resolve.destination->image(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
  }
  resolve.destination->Transition(
      command_buffer_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
  if (resolve.clear_color) {
    resolve.source->Transition(command_buffer_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = resolve.source->view();
    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.clearValue.color = resolve.clear;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = {resolve.source_offset, resolve.extent};
    rendering.layerCount = rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &attachment;
    vk.vkCmdBeginRendering(command_buffer_, &rendering);
    vk.vkCmdEndRendering(command_buffer_);
  }
}

void NativeCommandProcessor::LogFrameTiming(const FrameTiming& frame, const char* result) {
  frame_timing_ << frame.swap << ',' << frame.address << ',' << frame.extent.width << ','
                << frame.extent.height << ',' << frame.ready_ns << ',' << result << '\n';
  if (frame.swap % 60 == 0) {
    frame_timing_.flush();
  }
}

bool NativeCommandProcessor::RetireSubmission() {
  if (!submitted_) {
    return true;
  }
  const auto& vk = vulkan_device_->functions();
  const auto result = vk.vkWaitForFences(vulkan_device_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
    return false;
  }
  if (submitted_frame_) {
    LogFrameTiming(*submitted_frame_, result == VK_SUCCESS ? "rendered" : "gpu_failed");
    submitted_frame_.reset();
  }
  for (const auto& command : in_flight_) {
    const auto* resolve = std::get_if<QueuedResolve>(&command);
    if (!resolve) {
      continue;
    }
    if (resolve->depth_descriptor != SIZE_MAX) {
      depth_resolve_->Free(resolve->depth_descriptor);
    }
    if (resolve->color_descriptor != SIZE_MAX) {
      color_resolve_->Free(resolve->color_descriptor);
    }
  }
  if (result == VK_SUCCESS) {
    draw_uploads_->Reclaim(next_submission_ - 1);
    retired_submission_ = next_submission_ - 1;
  }
  submitted_ = false;
  if (result == VK_ERROR_DEVICE_LOST) {
    graphics_system_->OnHostGpuLossFromAnyThread(true);
  }
  return result == VK_SUCCESS;
}

void NativeCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t width, uint32_t height) {
  auto* presenter = graphics_system_->presenter();
  const uint32_t scaled_width = width * draw_resolution_scale_x_;
  const uint32_t scaled_height = height * draw_resolution_scale_y_;
  std::optional<VkExtent2D> resolved_extent;
  if (const auto resolved = resolved_images_.find(frontbuffer_ptr & (kPhysicalMemorySize - 1));
      resolved != resolved_images_.end()) {
    resolved_extent = resolved->second.image->extent();
  }
  const VkExtent2D output_extent =
      GetSwapOutputExtent({scaled_width, scaled_height}, resolved_extent);
  uint32_t display_width = width;
  uint32_t display_height = height;
  if (REXCVAR_GET(present_use_actual_aspect)) {
    int32_t configured_width = 0;
    int32_t configured_height = 0;
    if (video_mode_util::TryParseResolutionPreset(REXCVAR_GET(resolution), configured_width,
                                                  configured_height) &&
        configured_width > 0 && configured_height > 0) {
      display_width = uint32_t(configured_width);
      display_height = uint32_t(configured_height);
    } else {
      display_width = output_extent.width;
      display_height = output_extent.height;
    }
  }
  const bool ready = presenter && output_extent.width && output_extent.height &&
                     presenter->RefreshGuestOutput(
                         output_extent.width, output_extent.height, display_width, display_height,
                         [this, output_extent,
                          frontbuffer_ptr](ui::Presenter::GuestOutputRefreshContext& context) {
                           return RenderGuestOutput(context, output_extent, frontbuffer_ptr);
                         });
  const bool fresh = ready && presented_ && presented_->TakeResolvedOutput();
  if (!frame_timing_.is_open()) {
    return;
  }
  using namespace std::chrono;
  const FrameTiming timing{
      guest_frame_count(),
      frontbuffer_ptr,
      output_extent,
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()};
  if (!ready) {
    LogFrameTiming(timing, "refresh_failed");
  } else if (!presented_) {
    LogFrameTiming(timing, "missing");
  } else if (!fresh) {
    LogFrameTiming(timing, "stale");
  } else {
    submitted_frame_ = timing;
  }
}

bool NativeCommandProcessor::BeginCommands() {
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  if (!RetireSubmission()) {
    return false;
  }
  if (vk.vkResetFences(device, 1, &fence_) != VK_SUCCESS ||
      vk.vkResetCommandPool(device, command_pool_, 0) != VK_SUCCESS) {
    return false;
  }
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vk.vkBeginCommandBuffer(command_buffer_, &begin) != VK_SUCCESS) {
    return false;
  }
  in_flight_.clear();
  in_flight_readback_.reset();
  presented_.reset();
  return true;
}

void NativeCommandProcessor::RecordBufferUploads(std::span<const FrameCommand> commands) {
  const auto& vk = vulkan_device_->functions();
  VkBufferMemoryBarrier pre{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  pre.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.srcQueueFamilyIndex = pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  pre.buffer = shared_vertex_buffer_;
  pre.offset = 0;
  pre.size = VK_WHOLE_SIZE;
  vk.vkCmdPipelineBarrier(command_buffer_,
                          VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &pre, 0, nullptr);
  for (const auto& command : commands) {
    const auto& upload = std::get<QueuedBufferUpload>(command);
    const VkBufferCopy copy{upload.src_offset, upload.dst_offset, upload.size};
    vk.vkCmdCopyBuffer(command_buffer_, upload.src_buffer, shared_vertex_buffer_, 1, &copy);
  }
  VkBufferMemoryBarrier post{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  post.srcQueueFamilyIndex = post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  post.buffer = shared_vertex_buffer_;
  post.offset = 0;
  post.size = VK_WHOLE_SIZE;
  vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &post, 0, nullptr);
}

size_t NativeCommandProcessor::RecordPending() {
  // Clear the fallback textures and move them to the shader-read layout once,
  // before any draw that may sample them.
  RecordNullTextureInitialization();
  size_t drawn = 0;
  for (size_t first = 0; first < pending_.size();) {
    const auto& command = pending_[first];
    if (const auto* draw = std::get_if<QueuedDraw>(&command)) {
      size_t end = first + 1;
      for (; end < pending_.size(); ++end) {
        const auto* next = std::get_if<QueuedDraw>(&pending_[end]);
        if (!next || next->colors != draw->colors || next->depth != draw->depth) {
          break;
        }
      }
      RecordDraws(std::span(pending_).subspan(first, end - first));
      drawn += end - first;
      first = end;
    } else if (std::holds_alternative<QueuedBufferUpload>(command)) {
      size_t end = first + 1;
      for (; end < pending_.size(); ++end) {
        if (!std::holds_alternative<QueuedBufferUpload>(pending_[end])) {
          break;
        }
      }
      RecordBufferUploads(std::span(pending_).subspan(first, end - first));
      first = end;
    } else {
      if (const auto* resolve = std::get_if<QueuedResolve>(&command)) {
        RecordResolve(*resolve);
      } else {
        RecordDepthClear(std::get<QueuedDepthClear>(command));
      }
      ++first;
    }
  }

  return drawn;
}

bool NativeCommandProcessor::SubmitCommands() {
  const auto& vk = vulkan_device_->functions();
  if (vk.vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
    return false;
  }
  draw_uploads_->FlushWrites();
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer_;
  VkResult result;
  {
    auto queue = vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
    result = vk.vkQueueSubmit(queue.queue(), 1, &submit, fence_);
  }
  if (result != VK_SUCCESS) {
    if (result == VK_ERROR_DEVICE_LOST)
      graphics_system_->OnHostGpuLossFromAnyThread(true);
    return false;
  }
  ++next_submission_;
  submitted_ = true;
  for (const auto& command : pending_) {
    if (const auto* resolve = std::get_if<QueuedResolve>(&command)) {
      resolve->destination->MarkResolved();
    }
  }
  pending_.swap(in_flight_);
  return true;
}

bool NativeCommandProcessor::FlushPendingIfFull() {
  // Grow the batch enough that a whole frame records into a single command
  // buffer, so no mid-frame vkWaitForFences stall is needed. The largest
  // observed scene queues about 5.6K draws and resolves, so 8K leaves headroom
  // while still bounding memory.
  constexpr size_t kMaxPendingCommands = 8192;
  if (pending_.size() < kMaxPendingCommands) {
    return true;
  }
  if (!BeginCommands()) {
    return false;
  }
  RecordPending();
  return SubmitCommands();
}

bool NativeCommandProcessor::RenderGuestOutput(ui::Presenter::GuestOutputRefreshContext& context,
                                               VkExtent2D extent, uint32_t frontbuffer_address) {
  auto& output =
      static_cast<ui::vulkan::VulkanPresenter::VulkanGuestOutputRefreshContext&>(context);
  const auto& vk = vulkan_device_->functions();
  if (!BeginCommands()) {
    return false;
  }
  const size_t drawn = RecordPending();

  const auto resolved = resolved_images_.find(frontbuffer_address & (kPhysicalMemorySize - 1));
  const bool use_pwl = resolved != resolved_images_.end() &&
                       resolved->second.image->format() == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  if (resolved != resolved_images_.end() && resolved->second.image->initialized() &&
      (resolved->second.image->format() == ColorImage::kFormat || use_pwl) &&
      resolved->second.image->extent().width >= extent.width &&
      resolved->second.image->extent().height >= extent.height) {
    presented_ = resolved->second.image;
    resolved->second.last_use_submission = next_submission_;
    presented_resolve_address_ = resolved->first;
  } else {
    presented_resolve_address_.reset();
  }

  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = output.image();
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  const bool written = output.image_ever_written_previously();
  barrier.oldLayout =
      written ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout =
      presented_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrier.srcAccessMask = written ? VK_ACCESS_SHADER_READ_BIT : 0;
  const VkPipelineStageFlags output_stage = presented_
                                                ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                                : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  const VkAccessFlags output_access =
      presented_ ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  barrier.dstAccessMask = output_access;
  vk.vkCmdPipelineBarrier(
      command_buffer_,
      written ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      output_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  if (presented_) {
    presented_->Transition(command_buffer_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    const auto ramp = use_pwl ? std::as_bytes(std::span(gamma_ramp_pwl_rgb(), 128 * 3))
                              : std::as_bytes(std::span(gamma_ramp_256_entry_table(), 256));
    if (!output_gamma_->Record(command_buffer_, presented_->view(), output.image_view(), extent,
                               ramp, use_pwl)) {
      return false;
    }
  } else {
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = output.image_view();
    attachment.imageLayout = barrier.newLayout;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea.extent = extent;
    rendering.layerCount = rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &attachment;
    vk.vkCmdBeginRendering(command_buffer_, &rendering);
    vk.vkCmdEndRendering(command_buffer_);
  }
  barrier.oldLayout = barrier.newLayout;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = output_access;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vk.vkCmdPipelineBarrier(command_buffer_, output_stage, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &barrier);
  if (!SubmitCommands()) {
    return false;
  }
  if (presented_ && drawn && !reported_frame_) {
    reported_frame_ = true;
    REXLOG_INFO(
        "Native Vulkan: presented guest framebuffer {:08X} at {}x{} with {} submitted draws",
        frontbuffer_address, extent.width, extent.height, drawn);
  }
  output.SetIs8bpc(!presented_ || !use_pwl);
  return true;
}

}  // namespace rex::graphics::native
