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
#include <rex/graphics/flags.h>
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

// Vertex fetches read the shared vertex buffer; vertex and pixel shader memexports write it.
constexpr VkPipelineStageFlags kSharedMemoryShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

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
  const bool depth_unorm24_supported = (depth_unorm24_properties.optimalTilingFeatures &
                                         kUsedDepthFormatFeatures) == kUsedDepthFormatFeatures;
  // Images are sub-allocated; without VMA they fall back to dedicated allocations.
  vma_allocator_ = ui::vulkan::CreateVmaAllocator(vulkan_device_, true);
  get_pipeline_cache_data_ = reinterpret_cast<PFN_vkGetPipelineCacheData>(
      instance.vkGetDeviceProcAddr(device, "vkGetPipelineCacheData"));
  merge_pipeline_caches_ = reinterpret_cast<PFN_vkMergePipelineCaches>(
      instance.vkGetDeviceProcAddr(device, "vkMergePipelineCaches"));
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool.queueFamilyIndex = vulkan_device_->queue_family_graphics_compute();
  if (pool.queueFamilyIndex == UINT32_MAX) {
    ShutdownContext();
    return false;
  }
  for (auto& frame : frames_) {
    VkCommandBufferAllocateInfo buffer{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    buffer.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    buffer.commandBufferCount = 1;
    const VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (vk.vkCreateCommandPool(device, &pool, nullptr, &frame.command_pool) != VK_SUCCESS ||
        vk.vkCreateSemaphore(device, &semaphore, nullptr, &frame.sparse_bind_semaphore) !=
            VK_SUCCESS ||
        (buffer.commandPool = frame.command_pool,
         vk.vkAllocateCommandBuffers(device, &buffer, &frame.command_buffer) != VK_SUCCESS) ||
        vk.vkCreateFence(device, &fence, nullptr, &frame.fence) != VK_SUCCESS) {
      ShutdownContext();
      return false;
    }
  }
  current_frame_ = last_submitted_frame_ = 0;

  graphics::GetConfigDrawResolutionScale(draw_resolution_scale_x_, draw_resolution_scale_y_);
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
  const uint32_t shared_segment_size =
      kPhysicalMemorySize >> translator_->GetSharedMemoryStorageBufferCountLog2();
  if (vulkan_device_->properties().maxStorageBufferRange >= shared_segment_size) {
    shared_vertex_ = std::make_unique<SharedVertexMemory>(*vulkan_device_);
    if (!shared_vertex_->Initialize(REXCVAR_GET(vulkan_sparse_shared_memory))) {
      shared_vertex_.reset();
    }
  }
  index_cache_ =
      std::make_unique<NativeIndexCache>(*vulkan_device_, vma_allocator_, *memory_, clock_);
  if (!index_cache_->Initialize(shared_vertex_ != nullptr)) {
    ShutdownContext();
    return false;
  }
  {
    const VkDeviceSize quad_bytes =
        VkDeviceSize(kMaxBuiltinQuadCount * 5 - 1) * sizeof(uint32_t);
    VkBufferCreateInfo quad_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    quad_info.size = quad_bytes;
    quad_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    quad_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bool quad_ok = false;
    if (vma_allocator_ != VK_NULL_HANDLE) {
      VmaAllocationCreateInfo alloc_info{};
      alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      quad_ok = vmaCreateBuffer(vma_allocator_, &quad_info, &alloc_info, &builtin_quad_indices_,
                                &builtin_quad_indices_allocation_, nullptr) == VK_SUCCESS;
    } else {
      quad_ok = ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device_, quad_bytes, quad_info.usage,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, builtin_quad_indices_,
          builtin_quad_indices_memory_);
    }
    if (!quad_ok) {
      builtin_quad_indices_ = VK_NULL_HANDLE;
      builtin_quad_indices_memory_ = VK_NULL_HANDLE;
      builtin_quad_indices_allocation_ = VK_NULL_HANDLE;
    }
    builtin_quad_indices_uploaded_ = false;
  }
  texture_cache_ = std::make_unique<NativeTextureCache>(*vulkan_device_, vma_allocator_, *memory_,
                                                        *register_file_,
                                                        clock_, draw_resolution_scale_x_,
                                                        draw_resolution_scale_y_);
  target_cache_ = std::make_unique<NativeTargetCache>(
      *vulkan_device_, vma_allocator_, clock_, draw_resolution_scale_x_, draw_resolution_scale_y_,
      depth_unorm24_supported,
      [this](const DrawInput& input, std::span<const uint32_t> indices,
             const Shader::Translation& vertex, const Shader::Translation* pixel,
             uint32_t rt_index, const std::shared_ptr<ColorImage>& image) {
        QueueReplayDraw(input, indices, vertex, pixel, rt_index, image);
      },
      [this](const std::shared_ptr<ColorImage>& source,
             const std::shared_ptr<ColorImage>& destination, std::vector<VkImageCopy> copies) {
        pending_.emplace_back(QueuedTileTransfer{source, destination, std::move(copies)});
      });
  shader_pages_.assign((kSharedVertexPageCount + 63) / 64, 0);
  if (!texture_cache_->Initialize(shared_vertex_ != nullptr)) {
    ShutdownContext();
    return false;
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

void NativeCommandProcessor::ShutdownContext() {
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
  WaitForAllSubmissions();
  if (frame_timing_.is_open()) {
    frame_timing_.close();
  }
  pending_.clear();
  pending_passes_.clear();
  scratch_input_.Clear();
  // Pending and in-flight draws have released their shared textures by now.
  draw_uploads_.reset();
  if (builtin_quad_indices_ != VK_NULL_HANDLE) {
    if (builtin_quad_indices_allocation_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(vma_allocator_, builtin_quad_indices_, builtin_quad_indices_allocation_);
    } else {
      vk.vkDestroyBuffer(device, builtin_quad_indices_, nullptr);
      vk.vkFreeMemory(device, builtin_quad_indices_memory_, nullptr);
    }
    builtin_quad_indices_ = VK_NULL_HANDLE;
    builtin_quad_indices_memory_ = VK_NULL_HANDLE;
    builtin_quad_indices_allocation_ = VK_NULL_HANDLE;
    builtin_quad_indices_uploaded_ = false;
  }
  index_cache_.reset();
  shared_vertex_.reset();
  presented_.reset();
  target_cache_.reset();
  texture_cache_.reset();
  for (const auto& entry : pipelines_) {
    vk.vkDestroyPipeline(device, entry.second, nullptr);
  }
  pipelines_.clear();
  draw_bindings_.clear();
  SavePipelineCache();
  pipeline_cache_path_.clear();
  vk.vkDestroyPipelineCache(device, pipeline_cache_, nullptr);
  pipeline_cache_ = VK_NULL_HANDLE;
  color_resolve_.reset();
  depth_resolve_.reset();
  output_gamma_.reset();
  translator_.reset();
  active_vertex_shader_ = active_pixel_shader_ = nullptr;
  shader_addr_cache_.clear();
  shader_pages_.clear();
  shaders_.clear();
  InvalidateConstantState();
  if (vma_allocator_ != VK_NULL_HANDLE) {
    // Every image is gone: frames, caches, targets and resolves are released.
    vmaDestroyAllocator(vma_allocator_);
    vma_allocator_ = VK_NULL_HANDLE;
  }
  for (auto& frame : frames_) {
    vk.vkDestroyFence(device, frame.fence, nullptr);
    vk.vkDestroySemaphore(device, frame.sparse_bind_semaphore, nullptr);
    vk.vkDestroyCommandPool(device, frame.command_pool, nullptr);
    frame = {};
  }
  command_buffer_ = VK_NULL_HANDLE;
  vulkan_device_ = nullptr;
}

namespace {

std::array<uint8_t, VK_UUID_SIZE> GetPipelineCacheUuid(const ui::vulkan::VulkanDevice& device) {
  VkPhysicalDeviceProperties properties{};
  device.vulkan_instance()->functions().vkGetPhysicalDeviceProperties(device.physical_device(),
                                                                      &properties);
  std::array<uint8_t, VK_UUID_SIZE> uuid;
  std::memcpy(uuid.data(), properties.pipelineCacheUUID, uuid.size());
  return uuid;
}

}  // namespace

void NativeCommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                                     uint32_t title_id, bool blocking) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking);
  if (!vulkan_device_ || pipeline_cache_ == VK_NULL_HANDLE || !merge_pipeline_caches_ ||
      !get_pipeline_cache_data_) {
    return;
  }
  // A new title: keep what the previous one built before switching files.
  SavePipelineCache();
  const auto directory = cache_root / "shaders" / "local";
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    REXLOG_WARN("Native Vulkan: pipeline cache disabled, could not create {}: {}",
                directory.string(), error.message());
    pipeline_cache_path_.clear();
    return;
  }
  pipeline_cache_path_ = directory / fmt::format("{:08X}.native.vk.pcache", title_id);
  std::ifstream file(pipeline_cache_path_, std::ios::binary);
  if (!file) {
    return;
  }
  const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(file),
                                   std::istreambuf_iterator<char>()};
  const auto& properties = vulkan_device_->properties();
  const auto data = GetPipelineCacheFileData(bytes, properties.vendorID, properties.deviceID,
                                             GetPipelineCacheUuid(*vulkan_device_));
  if (data.empty()) {
    REXLOG_WARN("Native Vulkan: ignoring stale or corrupt pipeline cache {}",
                pipeline_cache_path_.string());
    return;
  }
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  info.initialDataSize = data.size();
  info.pInitialData = data.data();
  VkPipelineCache stored = VK_NULL_HANDLE;
  if (vk.vkCreatePipelineCache(device, &info, nullptr, &stored) != VK_SUCCESS) {
    return;
  }
  if (merge_pipeline_caches_(device, pipeline_cache_, 1, &stored) == VK_SUCCESS) {
    REXLOG_INFO("Native Vulkan: loaded {} bytes of pipeline cache from {}", data.size(),
                pipeline_cache_path_.string());
  }
  vk.vkDestroyPipelineCache(device, stored, nullptr);
}

void NativeCommandProcessor::SavePipelineCache() {
  std::lock_guard<std::timed_mutex> lock(pipeline_cache_mutex_);
  if (!vulkan_device_ || pipeline_cache_path_.empty() || pipeline_cache_ == VK_NULL_HANDLE ||
      !get_pipeline_cache_data_) {
    return;
  }
  const VkDevice device = vulkan_device_->device();
  size_t size = 0;
  if (get_pipeline_cache_data_(device, pipeline_cache_, &size, nullptr) != VK_SUCCESS || !size) {
    return;
  }
  std::vector<uint8_t> data(size);
  if (get_pipeline_cache_data_(device, pipeline_cache_, &size, data.data()) != VK_SUCCESS) {
    return;
  }
  data.resize(size);
  const auto& properties = vulkan_device_->properties();
  const auto header = MakePipelineCacheFileHeader(properties.vendorID, properties.deviceID,
                                                  GetPipelineCacheUuid(*vulkan_device_), data);
  // Written next to the cache and renamed over it, so a crash never leaves a
  // half-written file behind.
  auto temporary = pipeline_cache_path_;
  temporary += ".tmp";
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    if (!file) {
      REXLOG_WARN("Native Vulkan: could not write pipeline cache {}", temporary.string());
      return;
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, pipeline_cache_path_, error);
  if (error) {
    REXLOG_WARN("Native Vulkan: could not replace pipeline cache {}: {}",
                pipeline_cache_path_.string(), error.message());
    std::filesystem::remove(temporary, error);
    return;
  }
  saved_pipeline_count_ = pipelines_.size();
}

void NativeCommandProcessor::FlushShaderStorage() {
  std::unique_lock<std::timed_mutex> lock(pipeline_cache_mutex_, std::chrono::milliseconds(250));
  if (!lock.owns_lock() || !vulkan_device_ || pipeline_cache_path_.empty() ||
      pipeline_cache_ == VK_NULL_HANDLE || !get_pipeline_cache_data_) {
    return;
  }
  if (pipelines_.size() <= saved_pipeline_count_) {
    return;
  }
  lock.unlock();
  SavePipelineCache();
}

void NativeCommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  CommandProcessor::WriteRegister(index, value);
  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_255_W) {
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFloatVertex;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_256_X &&
             index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFloatPixel;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferBoolLoop;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFetch;
  }
}

void NativeCommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                                   uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  const uint32_t end_index = start_index + num_registers - 1;
  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    if (start_index <= XE_GPU_REG_SHADER_CONSTANT_255_W) {
      dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFloatVertex;
    }
    if (end_index >= XE_GPU_REG_SHADER_CONSTANT_256_X) {
      dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFloatPixel;
    }
    return;
  }
  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferBoolLoop;
    return;
  }
  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFetch;
    return;
  }
  CommandProcessor::WriteRegistersFromMem(start_index, base, num_registers);
}

Shader* NativeCommandProcessor::LoadShader(xenos::ShaderType shader_type,
                                           uint32_t guest_address, const uint32_t* host_address,
                                           uint32_t dword_count) {
  if (!host_address || !dword_count) {
    return nullptr;
  }
  const uint32_t byte_count = dword_count * uint32_t(sizeof(uint32_t));
  const bool cacheable_addr =
      guest_address != 0 && IsPhysicalRange(guest_address, byte_count) &&
      memory_->TranslatePhysical(guest_address) == reinterpret_cast<const uint8_t*>(host_address);
  const uint64_t addr_key = (uint64_t(guest_address) << 32) | dword_count;
  if (cacheable_addr) {
    if (auto found = shader_addr_cache_.find(addr_key);
        found != shader_addr_cache_.end() && found->second->type() == shader_type) {
      return found->second;
    }
  }
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
      if (cacheable_addr) {
        const auto [first_page, page_count] = GetPhysicalPageRange(guest_address, byte_count);
        SetPageRangeBits(shader_pages_, first_page, page_count);
        memory_->EnablePhysicalMemoryAccessCallbacks(guest_address, byte_count, true, false);
        shader_addr_cache_.emplace(addr_key, &shader);
      }
      return &shader;
    }
  }
  auto shader = std::make_unique<SpirvShader>(shader_type, hash, host_address, dword_count);
  auto* ptr = shaders_.emplace(hash, std::move(shader))->second.get();
  if (cacheable_addr) {
    const auto [first_page, page_count] = GetPhysicalPageRange(guest_address, byte_count);
    SetPageRangeBits(shader_pages_, first_page, page_count);
    memory_->EnablePhysicalMemoryAccessCallbacks(guest_address, byte_count, true, false);
    shader_addr_cache_.emplace(addr_key, ptr);
  }
  return ptr;
}

std::span<const uint8_t> NativeCommandProcessor::ReadPhysical(uint32_t address, size_t size) {
  return ReadGuestPhysical(*memory_, address, size);
}

void NativeCommandProcessor::DrainMemoryInvalidations() {
  if (!has_pending_invalidations_.load(std::memory_order_acquire)) {
    return;
  }
  std::vector<PhysicalRange> invalidations;
  {
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    invalidations.swap(pending_invalidations_);
    has_pending_invalidations_.store(false, std::memory_order_release);
  }
  if (invalidations.empty()) return;
  invalidated_since_last_draw_ = true;
  for (const auto& [start, length] : invalidations) {
    if (index_cache_) {
      index_cache_->Invalidate(start, length);
    }
    if (!shader_pages_.empty()) {
      const auto [first_page, page_count] = GetPhysicalPageRange(start, length);
      if (HasAnyPageBitSet(shader_pages_, first_page, page_count)) {
        for (auto it = shader_addr_cache_.begin(); it != shader_addr_cache_.end();) {
          const uint32_t s_addr = uint32_t(it->first >> 32);
          const uint32_t s_len = uint32_t(it->first) * uint32_t(sizeof(uint32_t));
          if (RangesOverlap(start, length, s_addr, s_len)) {
            it = shader_addr_cache_.erase(it);
          } else {
            ++it;
          }
        }
        ClearPageRangeBits(shader_pages_, first_page, page_count);
      }
    }
    InvalidateSharedVertices(start, length);
  }
  // Keep the other caches' notification order. The texture cache only needs
  // the union of written bytes and can process it once before the next draw.
  CoalescePhysicalRanges(invalidations);
  texture_cache_->InvalidateWrittenRanges(invalidations);
}

void NativeCommandProcessor::InvalidateGpuMemory() {
  if (texture_cache_) {
    texture_cache_->InvalidateAll();
  }
  if (index_cache_) {
    index_cache_->InvalidateAll();
  }
  shader_addr_cache_.clear();
  std::fill(shader_pages_.begin(), shader_pages_.end(), 0);
  if (shared_vertex_) {
    shared_vertex_->InvalidateAll();
  }
  InvalidateConstantState();
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
  WaitForAllSubmissions();
  pending_.clear();
  pending_passes_.clear();
  presented_.reset();
  InvalidateGpuMemory();
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  texture_cache_->ClearSamplers();
  for (const auto& entry : pipelines_) {
    vk.vkDestroyPipeline(device, entry.second, nullptr);
  }
  pipelines_.clear();
  draw_bindings_.clear();
  target_cache_->Clear();
  InvalidateConstantState();
}

bool NativeCommandProcessor::EnsureSharedVertexRange(uint32_t address, uint32_t size) {
  if (!size) {
    return false;
  }
  if (shared_vertex_->ArePagesValid(address, size)) {
    return true;
  }
  if (ReadPhysical(address, size).empty()) {
    return false;
  }
  return shared_vertex_->EnsureRange(
      address, size, uint32_t(kDrawUploadPageSize), [&](uint32_t chunk_addr, uint32_t chunk_size) {
        const auto bytes = ReadPhysical(chunk_addr, chunk_size);
        if (bytes.empty()) {
          return false;
        }
        VkBuffer src_buffer = VK_NULL_HANDLE;
        VkDeviceSize src_offset = 0;
        uint8_t* dst = draw_uploads_->Request(clock_.next, chunk_size, 16, src_buffer, src_offset);
        if (!dst) {
          return false;
        }
        // Watch the guest pages before reading them so any write after the copy
        // invalidates them before a later draw reads stale vertices.
        memory_->EnablePhysicalMemoryAccessCallbacks(chunk_addr, chunk_size, true, false);
        std::memcpy(dst, bytes.data(), chunk_size);
        pending_.emplace_back(QueuedBufferUpload{src_buffer, src_offset, chunk_addr, chunk_size,
                                                 !invalidated_since_last_draw_});
        return true;
      });
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
    if (!exact_range) {
      const auto [tex_start, tex_len] =
          texture_cache_->WidenInvalidation(physical_address_start, length);
      if (tex_len) {
        physical_address_start = tex_start;
        length = tex_len;
      } else if (index_cache_) {
        const auto [idx_start, idx_len] =
            index_cache_->WidenInvalidation(physical_address_start, length);
        if (idx_len) {
          physical_address_start = idx_start;
          length = idx_len;
        }
      }
      if (!shader_pages_.empty()) {
        const auto [s_start, s_len] =
            WidenInvalidationPageRange(shader_pages_, physical_address_start, length);
        if (s_len) {
          physical_address_start = s_start;
          length = s_len;
        }
      }
    }
    std::lock_guard<std::mutex> lock(invalidation_mutex_);
    pending_invalidations_.emplace_back(physical_address_start, length);
    has_pending_invalidations_.store(true, std::memory_order_release);
  }
  return {physical_address_start, length};
}

bool NativeCommandProcessor::PreparePreuploadedConstants(const DrawInput& input,
                                                         const SpirvShader& vs,
                                                         const SpirvShader* ps,
                                                         bool use_shared_vertices,
                                                         PreuploadedConstants& out) {
  const RegisterFile& regs = *register_file_;
  const auto& vs_map = vs.constant_register_map();
  for (uint32_t i = 0; i < 4; ++i) {
    if (last_vs_float_bitmap_[i] != vs_map.float_bitmap[i]) {
      last_vs_float_bitmap_[i] = vs_map.float_bitmap[i];
      if (vs_map.float_count) {
        dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFloatVertex;
      }
    }
  }
  if (ps) {
    const auto& ps_map = ps->constant_register_map();
    for (uint32_t i = 0; i < 4; ++i) {
      if (last_ps_float_bitmap_[i] != ps_map.float_bitmap[i]) {
        last_ps_float_bitmap_[i] = ps_map.float_bitmap[i];
        if (ps_map.float_count) {
          dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFloatPixel;
        }
      }
    }
  }
  if (!has_last_system_constants_ ||
      std::memcmp(&last_system_constants_, &input.system, sizeof(input.system)) != 0) {
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferSystem;
  }
  if (!use_shared_vertices) {
    dirty_constants_ |= UINT32_C(1) << Translator::kConstantBufferFetch;
  }

  uint32_t upload_mask = dirty_constants_ & kAllConstantBuffersMask;
  if (last_constant_buffer_ == VK_NULL_HANDLE) {
    upload_mask = kAllConstantBuffersMask;
  } else {
    for (uint32_t i = 0; i < Translator::kConstantBufferCount; ++i) {
      if (!(upload_mask & (UINT32_C(1) << i)) &&
          uint64_t(last_constant_offsets_[i]) + kDynamicConstantBufferRanges[i] >
              kDrawUploadPageSize) {
        upload_mask = kAllConstantBuffersMask;
        break;
      }
    }
  }

  const VkDeviceSize align = vulkan_device_->properties().minUniformBufferOffsetAlignment;
  auto compute_dirty_layout = [&](uint32_t mask,
                                  std::array<VkDeviceSize, Translator::kConstantBufferCount>& rel) {
    VkDeviceSize sz = 0;
    for (uint32_t i = 0; i < Translator::kConstantBufferCount; ++i) {
      if (!(mask & (UINT32_C(1) << i))) {
        continue;
      }
      sz = rex::align(sz, align);
      rel[i] = sz;
      VkDeviceSize bytes = 16;
      switch (i) {
        case Translator::kConstantBufferSystem:
          bytes = sizeof(input.system);
          break;
        case Translator::kConstantBufferFloatVertex:
          bytes = std::max<VkDeviceSize>(16, VkDeviceSize(vs_map.float_count) * 16);
          break;
        case Translator::kConstantBufferFloatPixel:
          bytes = std::max<VkDeviceSize>(
              16, VkDeviceSize(ps ? ps->constant_register_map().float_count : 0) * 16);
          break;
        case Translator::kConstantBufferBoolLoop:
          bytes = sizeof(input.bool_loop);
          break;
        case Translator::kConstantBufferFetch:
          bytes = sizeof(input.fetch);
          break;
      }
      sz += bytes;
    }
    for (uint32_t i = 0; i < Translator::kConstantBufferCount; ++i) {
      if (mask & (UINT32_C(1) << i)) {
        sz = std::max<VkDeviceSize>(sz, rel[i] + kDynamicConstantBufferRanges[i]);
      }
    }
    return sz;
  };

  if (upload_mask == 0) {
    out.buffer = last_constant_buffer_;
    out.dynamic_offsets = last_constant_offsets_;
    return true;
  }

  std::array<VkDeviceSize, Translator::kConstantBufferCount> rel_offsets{};
  VkDeviceSize req_size = compute_dirty_layout(upload_mask, rel_offsets);
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceSize buf_off = 0;
  uint8_t* mapped = draw_uploads_->Request(clock_.next, req_size, align, buf, buf_off);
  if (!mapped) {
    return false;
  }
  if (buf != last_constant_buffer_ && upload_mask != kAllConstantBuffersMask) {
    upload_mask = kAllConstantBuffersMask;
    req_size = compute_dirty_layout(upload_mask, rel_offsets);
    mapped = draw_uploads_->Request(clock_.next, req_size, align, buf, buf_off);
    if (!mapped) {
      return false;
    }
  }

  for (uint32_t i = 0; i < Translator::kConstantBufferCount; ++i) {
    if (!(upload_mask & (UINT32_C(1) << i))) {
      continue;
    }
    uint8_t* dst = mapped + rel_offsets[i];
    switch (i) {
      case Translator::kConstantBufferSystem:
        std::memcpy(dst, &input.system, sizeof(input.system));
        last_system_constants_ = input.system;
        has_last_system_constants_ = true;
        break;
      case Translator::kConstantBufferFloatVertex:
        if (vs_map.float_count == 0) {
          std::memset(dst, 0, 16);
        } else {
          for (uint32_t w = 0; w < 4; ++w) {
            uint64_t bits = vs_map.float_bitmap[w];
            while (bits) {
              const uint32_t b = uint32_t(std::countr_zero(bits));
              bits &= bits - 1;
              std::memcpy(dst, &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + ((w * 64 + b) << 2)],
                          16);
              dst += 16;
            }
          }
        }
        break;
      case Translator::kConstantBufferFloatPixel:
        if (!ps || ps->constant_register_map().float_count == 0) {
          std::memset(dst, 0, 16);
        } else {
          const auto& pmap = ps->constant_register_map();
          for (uint32_t w = 0; w < 4; ++w) {
            uint64_t bits = pmap.float_bitmap[w];
            while (bits) {
              const uint32_t b = uint32_t(std::countr_zero(bits));
              bits &= bits - 1;
              std::memcpy(dst, &regs.values[XE_GPU_REG_SHADER_CONSTANT_256_X + ((w * 64 + b) << 2)],
                          16);
              dst += 16;
            }
          }
        }
        break;
      case Translator::kConstantBufferBoolLoop:
        std::memcpy(dst, &regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                    sizeof(input.bool_loop));
        break;
      case Translator::kConstantBufferFetch:
        std::memcpy(dst,
                    use_shared_vertices
                        ? reinterpret_cast<const void*>(
                              &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0])
                        : reinterpret_cast<const void*>(input.fetch.data()),
                    sizeof(input.fetch));
        break;
    }
    last_constant_offsets_[i] = uint32_t(buf_off + rel_offsets[i]);
  }
  last_constant_buffer_ = buf;
  dirty_constants_ = use_shared_vertices ? 0u : (UINT32_C(1) << Translator::kConstantBufferFetch);
  out.buffer = last_constant_buffer_;
  out.dynamic_offsets = last_constant_offsets_;
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
  // Without a surface pitch nothing is rasterized; a vertex shader may still memexport.
  const bool has_pitch = dimensions.surface_pitch != 0;
  const bool use_depth = has_pitch && (depth_control.z_enable || depth_control.stencil_enable);
  const bool use_color = has_pitch && mode == xenos::EdramMode::kColorDepth && color_mask;
  const auto depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  const bool float_depth = depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8;
  if (!use_color && !use_depth && !candidate_vs.memexport_eM_written()) {
    return true;
  }
  if (!topology || !index_count || dimensions.msaa_samples > xenos::MsaaSamples::k4X ||
      (use_color && !active_pixel_shader_)) {
    return false;
  }

  if (!FlushPendingIfFull()) {
    return false;
  }

  auto& vs = *static_cast<SpirvShader*>(active_vertex_shader_);
  auto* ps = mode == xenos::EdramMode::kColorDepth ? static_cast<SpirvShader*>(active_pixel_shader_)
                                                   : nullptr;
  if (ps && !ps->is_ucode_analyzed()) {
    string::StringBuffer disassembly;
    ps->AnalyzeUcode(disassembly);
  }
  const auto& properties = vulkan_device_->properties();
  const bool vs_memexport = vs.memexport_eM_written() != 0;
  const bool ps_memexport = ps && ps->memexport_eM_written() != 0;
  const bool memexport = vs_memexport || ps_memexport;
  if ((vs_memexport && !properties.vertexPipelineStoresAndAtomics) ||
      (ps_memexport && !properties.fragmentStoresAndAtomics) || (memexport && !shared_vertex_)) {
    REXGPU_DEBUG("Native Vulkan: memexport draw unsupported on this device");
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
  const bool draw_native_1x = IsDrawScaleNativeForPitch(
      dimensions.surface_pitch, dimensions.msaa_samples, REXCVAR_GET(draw_resolution_scale_threshold),
      draw_resolution_scale_x_, draw_resolution_scale_y_);
  const uint32_t sx = draw_native_1x ? 1u : draw_resolution_scale_x_;
  const uint32_t sy = draw_native_1x ? 1u : draw_resolution_scale_y_;
  const bool draw_scaled =
      !draw_native_1x && (draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1);
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
    pixel_modification.pixel.resolution_scale_native = draw_native_1x ? 1u : 0u;
    pixel = ps->GetOrCreateTranslation(pixel_modification.value);
  }
  auto& vertex = *vs.GetOrCreateTranslation(vertex_modification.value);
  if ((!vertex.is_translated() && !translator_->TranslateAnalyzedShader(vertex)) ||
      (pixel && ((!pixel->is_translated() && !translator_->TranslateAnalyzedShader(*pixel)) ||
                 !pixel->is_valid())) ||
      !vertex.is_valid()) {
    return false;
  }

  scratch_input_.Clear();
  DrawInput& input = scratch_input_;
  struct ScratchInputGuard {
    DrawInput& input;
    ~ScratchInputGuard() { input.textures.clear(); }
  } scratch_guard{scratch_input_};
  input.topology = *topology;
  const bool multi_prim_restart = index_buffer_info &&
                                  regs.Get<reg::PA_SU_SC_MODE_CNTL>().multi_prim_ib_ena &&
                                  (prim_type == xenos::PrimitiveType::kTriangleStrip ||
                                   prim_type == xenos::PrimitiveType::kTriangleFan ||
                                   prim_type == xenos::PrimitiveType::kLineStrip);
  input.primitive_restart = multi_prim_restart || expanded;
  const uint32_t restart_reset =
      multi_prim_restart
          ? (regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx &
             (index_buffer_info->format == xenos::IndexFormat::kInt16 ? uint32_t(UINT16_MAX)
                                                                      : xenos::kVertexIndexMask))
          : 0u;
  input.samples = VkSampleCountFlagBits(1u << uint32_t(dimensions.msaa_samples));
  input.depth_format = use_depth ? target_cache_->GetDepthFormat(depth_info.depth_format) : VK_FORMAT_UNDEFINED;
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
      use_depth ? ComputeHostDepthBias(regs, primitive_polygonal, depth_info.depth_format, sx, sy)
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
  DrainMemoryInvalidations();
  if (!texture_cache_->ReadTextures(vs, true, input, draw_uploads_.get(), clock_.next) ||
      (ps && !texture_cache_->ReadTextures(*ps, false, input, draw_uploads_.get(), clock_.next))) {
    return false;
  }
  input.memexport = memexport;
  input.rasterizer_discard = !use_color && !use_depth;
  draw_util::GetHostViewportInfo(regs, sx, sy, false, properties.maxViewportDimensions[0],
                                 properties.maxViewportDimensions[1], true, depth_control,
                                 float_depth, true, ps && ps->writes_depth(), input.viewport);
  draw_util::GetScissor(regs, input.scissor);
  input.scissor.offset[0] *= sx;
  input.scissor.offset[1] *= sy;
  input.scissor.extent[0] *= sx;
  input.scissor.extent[1] *= sy;
  if (input.rasterizer_discard) {
    // Only the vertex shader's memexport runs; keep valid dynamic state.
    input.viewport.xy_offset[0] = input.viewport.xy_offset[1] = 0;
    input.viewport.xy_extent[0] = input.viewport.xy_extent[1] = 1;
    input.scissor = {};
    input.scissor.extent[0] = input.scissor.extent[1] = 1;
  } else if (!memexport &&
             (!input.viewport.xy_extent[0] || !input.viewport.xy_extent[1] ||
              !input.scissor.extent[0] || !input.scissor.extent[1])) {
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
    const auto& index = *index_buffer_info;
    const size_t index_bytes =
        size_t(index_count) *
        (index.format == xenos::IndexFormat::kInt32 ? sizeof(uint32_t) : sizeof(uint16_t));
    const auto src_bytes =
        index_bytes <= index.length ? ReadPhysical(index.guest_base, index_bytes)
                                    : std::span<const uint8_t>{};
    if (src_bytes.empty()) {
      return false;
    }
    input.vertices.resize(size_t(index_count) * sizeof(uint32_t));
    DecodeDmaIndices32(src_bytes, index.format, index.endianness,
                       reinterpret_cast<uint32_t*>(input.vertices.data()), index_count);
  }
  const bool may_snapshot_fast_clear =
      dimensions.msaa_samples == xenos::MsaaSamples::k4X &&
      ((!use_color && use_depth && !ps) || rectangle);
  const bool use_shared_vertices =
      shared_vertex_ && !(expanded && index_buffer_info) && !may_snapshot_fast_clear;
  IndexBinding index_binding{};
  std::span<const uint32_t> inline_indices{};
  if (expanded) {
    const uint32_t quad_count = rectangle ? index_count / 3 : index_count;
    if (!quad_count) {
      return true;
    }
    if (quad_count <= kMaxBuiltinQuadCount && builtin_quad_indices_ != VK_NULL_HANDLE) {
      index_binding = {true, builtin_quad_indices_, 0, VK_INDEX_TYPE_UINT32, quad_count * 5 - 1};
    } else {
      FillQuadStripIndices(quad_count, scratch_indices_);
      index_binding.indexed = true;
      inline_indices = scratch_indices_;
    }
  } else if (index_buffer_info) {
    const auto& index = *index_buffer_info;
    const size_t index_bytes =
        size_t(index_count) *
        (index.format == xenos::IndexFormat::kInt32 ? sizeof(uint32_t) : sizeof(uint16_t));
    const auto src_bytes =
        index_bytes <= index.length ? ReadPhysical(index.guest_base, index_bytes)
                                    : std::span<const uint8_t>{};
    if (src_bytes.empty()) {
      return false;
    }
    const bool use_uint16 = (index.format == xenos::IndexFormat::kInt16) &&
                            (!multi_prim_restart || restart_reset == UINT16_MAX);
    const IndexCacheKey cache_key{index.guest_base,    index_count,   index.format,
                                  index.endianness,    multi_prim_restart, restart_reset,
                                  use_uint16};
    CachedIndexUpload queued_upload{};
    bool uploaded = false;
    if (!may_snapshot_fast_clear && index_cache_ &&
        index_cache_->GetOrCreate(cache_key, src_bytes, *draw_uploads_, index_binding,
                                  queued_upload, uploaded)) {
      if (uploaded) {
        pending_.emplace_back(QueuedIndexUpload{queued_upload.src_buffer, queued_upload.src_offset,
                                                queued_upload.dst_buffer, queued_upload.dst_offset,
                                                queued_upload.size});
      }
    } else {
      const VkDeviceSize host_stride = use_uint16 ? sizeof(uint16_t) : sizeof(uint32_t);
      const VkDeviceSize host_bytes = VkDeviceSize(index_count) * host_stride;
      VkBuffer up_buf = VK_NULL_HANDLE;
      VkDeviceSize up_off = 0;
      uint8_t* dst =
          (!may_snapshot_fast_clear && host_bytes <= kDrawUploadPageSize)
              ? draw_uploads_->Request(clock_.next, host_bytes, 4, up_buf, up_off)
              : nullptr;
      if (dst) {
        if (use_uint16) {
          DecodeDmaIndices16(src_bytes, index.endianness, reinterpret_cast<uint16_t*>(dst),
                             index_count);
        } else {
          DecodeDmaIndices32(src_bytes, index.format, index.endianness,
                             reinterpret_cast<uint32_t*>(dst), index_count, multi_prim_restart,
                             restart_reset);
        }
        index_binding = {true, up_buf, up_off,
                         use_uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32, index_count};
      } else {
        scratch_indices_.resize(index_count);
        DecodeDmaIndices32(src_bytes, index.format, index.endianness, scratch_indices_.data(),
                           index_count, multi_prim_restart, restart_reset);
        index_binding.indexed = true;
        inline_indices = scratch_indices_;
      }
    }
  } else {
    index_binding = {false, VK_NULL_HANDLE, 0, VK_INDEX_TYPE_UINT32, index_count};
  }
  if (point) {
    const auto minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    const auto size = regs.Get<reg::PA_SU_POINT_SIZE>();
    input.system.point_vertex_diameter_min = float(minmax.min_size) / 8.0f;
    input.system.point_vertex_diameter_max = float(minmax.max_size) / 8.0f;
    input.system.point_constant_diameter[0] = float(size.width) / 8.0f;
    input.system.point_constant_diameter[1] = float(size.height) / 8.0f;
    input.system.point_screen_diameter_to_ndc_radius[0] =
        float(sx) / float(input.viewport.xy_extent[0]);
    input.system.point_screen_diameter_to_ndc_radius[1] =
        float(sy) / float(input.viewport.xy_extent[1]);
  }
  input.system.alpha_test_reference = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  input.system.alpha_to_mask = color.alpha_to_mask_enable ? (color.value >> 24) | (1u << 8) : 0;
  input.system.vertex_base_index = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  input.system.vertex_index_min = regs.Get<uint32_t>(XE_GPU_REG_VGT_MIN_VTX_INDX);
  input.system.vertex_index_max = regs.Get<uint32_t>(XE_GPU_REG_VGT_MAX_VTX_INDX);
  std::ranges::copy(input.viewport.ndc_scale, input.system.ndc_scale);
  std::ranges::copy(input.viewport.ndc_offset, input.system.ndc_offset);
  std::ranges::fill(input.system.color_exp_bias, 1.0f);
  if (!use_shared_vertices) {
    std::memcpy(input.fetch.data(), &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                sizeof(input.fetch));
  }
  const bool need_fast_clear_constants =
      (!use_color && use_depth && dimensions.msaa_samples == xenos::MsaaSamples::k4X && !ps) ||
      (use_color && pixel != nullptr && input.textures.empty() &&
       prim_type == xenos::PrimitiveType::kRectangleList &&
       dimensions.msaa_samples == xenos::MsaaSamples::k4X);
  if (need_fast_clear_constants) {
    std::memcpy(input.bool_loop.data(), &regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                sizeof(input.bool_loop));
    if (use_shared_vertices) {
      std::memcpy(input.fetch.data(), &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                  sizeof(input.fetch));
    }
    const std::array<const Shader*, 2> shaders{&vs, ps};
    for (size_t stage = 0; stage < shaders.size(); ++stage) {
      if (!shaders[stage]) {
        continue;
      }
      const auto& map = shaders[stage]->constant_register_map();
      for (uint32_t w = 0; w < 4; ++w) {
        uint64_t bits = map.float_bitmap[w];
        while (bits) {
          const uint32_t b = uint32_t(std::countr_zero(bits));
          bits &= bits - 1;
          const auto* value =
              &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + (stage * 256 + w * 64 + b) * 4];
          input.float_constants[stage].insert(input.float_constants[stage].end(), value, value + 4);
        }
      }
    }
  }

  DrainMemoryInvalidations();
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
  // Memexport writes the shared vertex buffer, so the guest data it partially
  // overwrites must be there first.
  std::vector<draw_util::MemExportRange> memexport_ranges;
  if (memexport) {
    if (!use_shared_vertices) {
      return false;
    }
    if (vs_memexport) {
      draw_util::AddMemExportRanges(regs, vs, memexport_ranges);
    }
    if (ps_memexport) {
      draw_util::AddMemExportRanges(regs, *ps, memexport_ranges);
    }
    for (const auto& range : memexport_ranges) {
      if (!EnsureSharedVertexRange(range.base_address_dwords << 2, range.size_bytes)) {
        return false;
      }
    }
  }

  QueuedDraw draw{};
  const auto layout = ActiveEdramLayout();
  // Guest rows the draw may touch, for EDRAM tile ownership.
  const uint32_t used_height = (input.scissor.offset[1] + input.scissor.extent[1] + sy - 1) / sy;
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
                                format, VK_IMAGE_ASPECT_COLOR_BIT, rt_is_64bpp, used_height,
                                draw_scaled);
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
  if (!input.color_count && !use_depth && !memexport) {
    return true;
  }
  draw.color_count = input.color_count;
  draw.memexport = memexport;
  if (use_depth) {
    draw.depth = FindTarget(layout.DecodeBase(depth_info.value), dimensions.surface_pitch,
                            dimensions.msaa_samples, input.depth_format,
                            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false,
                            used_height, draw_scaled);
    if (!draw.depth) {
      return false;
    }
  }
  PreuploadedConstants preuploaded_constants{};
  if (!PreparePreuploadedConstants(input, vs, ps, use_shared_vertices, preuploaded_constants)) {
    return false;
  }
  draw.pass_index =
      AddPendingPass(input, index_binding, inline_indices, vertex, pixel,
                     use_shared_vertices ? shared_vertex_->buffer() : VK_NULL_HANDLE,
                     &preuploaded_constants);
  if (draw.pass_index == SIZE_MAX) {
    return false;
  }
  pending_.emplace_back(std::move(draw));
  invalidated_since_last_draw_ = false;
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
  if (memexport) {
    // The shared vertex buffer now holds the newest data of these ranges, and
    // anything decoded from the guest copies is stale. Unknown destinations
    // (dynamic stream constants) conservatively invalidate every texture.
    if (memexport_ranges.empty()) {
      texture_cache_->InvalidateTextures(0, kPhysicalMemorySize);
      texture_cache_->InvalidateResolvedTextures(0, kPhysicalMemorySize);
    }
    for (const auto& range : memexport_ranges) {
      texture_cache_->InvalidateTextures(range.base_address_dwords << 2, range.size_bytes);
      texture_cache_->InvalidateResolvedTextures(range.base_address_dwords << 2, range.size_bytes);
    }
    if (IsReadbackMemexportEnabled(false) && !memexport_ranges.empty() &&
        !ReadBackMemexport(memexport_ranges)) {
      return false;
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
  auto fill_clear_indices = [&](std::vector<uint32_t>& out) {
    if (!inline_indices.empty()) {
      out.assign(inline_indices.begin(), inline_indices.end());
    } else if (expanded) {
      FillQuadStripIndices(rectangle ? index_count / 3 : index_count, out);
    } else {
      out.resize(index_count);
      std::iota(out.begin(), out.end(), 0u);
    }
  };
  if (aliased_4x_depth_clear && pitch_tiles_4x) {
    PendingFastClear clear;
    clear.input = input;
    fill_clear_indices(clear.indices);
    clear.vertex = &vertex;
    clear.depth_format = depth_info.depth_format;
    clear.color_clear_capable =
        IsEdramDepthClearToColor(depth_info.depth_format, depth_control, front.stencilwritemask);
    clear.clear_stencil =
        depth_control.stencilzpass == xenos::StencilOp::kReplace ? front.stencilref : 0u;
    clear.resolution_scaled = draw_scaled;
    const auto aliases = target_cache_->CollectAliases([&](uint32_t tile, uint32_t pitch,
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
    for (const auto& [image, samples, alias_scaled] : aliases) {
      target_cache_->ReplayAliasedFastClear(clear, image, samples, alias_scaled);
    }
    target_cache_->pending_fast_clears().insert_or_assign({depth_tile, pitch_tiles_4x}, std::move(clear));
    if (clear.color_clear_capable) {
      target_cache_->ErasePendingColorFastClears(depth_tile, pitch_tiles_4x);
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
      fill_clear_indices(clear.indices);
      clear.vertex = depth_vs;
      clear.depth_format = depth_info.depth_format;
      clear.color_clear_capable = false;
      clear.clear_stencil = 0;
      clear.resolution_scaled = draw_scaled;
      const auto depth_aliases = target_cache_->CollectAliases([&](uint32_t tile, uint32_t pitch,
                                                     xenos::MsaaSamples samples, VkFormat,
                                                     const ColorImage& img) {
        return tile == depth_tile &&
               xenos::GetSurfacePitchTiles(pitch, samples, false) == pitch_tiles_4x &&
               (img.aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
               img.format() == input.depth_format;
      });
      for (const auto& [image, samples, alias_scaled] : depth_aliases) {
        target_cache_->ReplayAliasedFastClear(clear, image, samples, alias_scaled);
      }
      target_cache_->pending_fast_clears().insert_or_assign({depth_tile, pitch_tiles_4x}, std::move(clear));
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
      fill_clear_indices(color_clear.indices);
      color_clear.vertex = &vertex;
      color_clear.pixel = pixel;
      color_clear.pixel_scaled = pixel;
      if (!draw_scaled && (draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1) && ps) {
        Translator::Modification scaled_ps_mod(translator_->GetDefaultPixelShaderModification(
            ps->GetDynamicAddressableRegisterCount(program.ps_num_reg)));
        scaled_ps_mod.pixel.interpolator_mask = interpolators;
        scaled_ps_mod.pixel.interpolators_centroid =
            interpolators & ~xenos::GetInterpolatorSamplingPattern(
                                dimensions.msaa_samples,
                                regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
                                regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);
        if (param_gen < xenos::kMaxInterpolators) {
          scaled_ps_mod.pixel.param_gen_enable = 1;
          scaled_ps_mod.pixel.param_gen_interpolator = param_gen;
          scaled_ps_mod.pixel.param_gen_point = point;
        }
        scaled_ps_mod.pixel.resolution_scale_native = 0;
        if (auto* candidate_ps = ps->GetOrCreateTranslation(scaled_ps_mod.value);
            candidate_ps &&
            (candidate_ps->is_translated() ||
             translator_->TranslateAnalyzedShader(*candidate_ps)) &&
            candidate_ps->is_valid()) {
          color_clear.pixel_scaled = candidate_ps;
        }
      }
      color_clear.rt_index = i;
      color_clear.resolution_scaled = draw_scaled;
      const auto color_aliases = target_cache_->CollectAliases([&](uint32_t tile, uint32_t pitch,
                                                     xenos::MsaaSamples samples, VkFormat fmt,
                                                     const ColorImage& img) {
        return tile == rt_tile && fmt == rt_format &&
               (img.aspects() & VK_IMAGE_ASPECT_COLOR_BIT) != 0 &&
               xenos::GetSurfacePitchTiles(pitch, samples, rt_is_64bpp) == rt_pitch_tiles_4x;
      });
      for (const auto& [image, samples, alias_scaled] : color_aliases) {
        target_cache_->ReplayAliasedColorFastClear(color_clear, image, samples, alias_scaled);
      }
      target_cache_->pending_color_fast_clears().insert_or_assign({rt_tile, rt_pitch_tiles_4x, rt_format},
                                                  std::move(color_clear));
      if (!rt_is_64bpp) {
        if (auto fc = target_cache_->pending_fast_clears().find({rt_tile, rt_pitch_tiles_4x});
            fc != target_cache_->pending_fast_clears().end()) {
          fc->second.color_clear_capable = false;
        }
      }
    }
  } else if (!aliased_4x_depth_clear && !aliased_4x_color_depth_clear) {
    // Any ordinary draw at a tile consumes its pending fast clear.
    const uint32_t pitch_tiles =
        xenos::GetSurfacePitchTiles(dimensions.surface_pitch, dimensions.msaa_samples, false);
    if (pitch_tiles && use_depth) {
      target_cache_->pending_fast_clears().erase({depth_tile, pitch_tiles});
    }
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if ((writes >> (i * 4)) & 15) {
        const auto target =
            regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
        const uint32_t rt_tile = layout.DecodeBase(target.value);
        if (pitch_tiles) {
          target_cache_->pending_fast_clears().erase({rt_tile, pitch_tiles});
        }
        const uint32_t rt_pitch_tiles = xenos::GetSurfacePitchTiles(
            dimensions.surface_pitch, dimensions.msaa_samples,
            xenos::IsColorRenderTargetFormat64bpp(target.color_format));
        if (rt_pitch_tiles) {
          target_cache_->ErasePendingColorFastClears(rt_tile, rt_pitch_tiles);
        }
      }
    }
  }
  return true;
}

std::shared_ptr<ColorImage> NativeCommandProcessor::FindTarget(uint32_t tile, uint32_t pitch,
                                                               xenos::MsaaSamples samples,
                                                               VkFormat format,
                                                               VkImageAspectFlags aspects,
                                                               bool is_64bpp,
                                                               uint32_t used_height,
                                                               bool resolution_scaled) {
  return target_cache_->FindTarget(tile, pitch, samples, format, aspects, is_64bpp,
                                   ActiveEdramLayout().tile_count, presented_, used_height,
                                   resolution_scaled);
}

void NativeCommandProcessor::QueueReplayDraw(const DrawInput& input,
                                             std::span<const uint32_t> indices,
                                             const Shader::Translation& vertex,
                                             const Shader::Translation* pixel, uint32_t rt_index,
                                             const std::shared_ptr<ColorImage>& image) {
  QueuedDraw draw{};
  if (image->aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) {
    draw.depth = image;
  } else {
    draw.colors[rt_index] = image;
    draw.color_count = rt_index + 1;
  }
  const IndexBinding index_binding{true, VK_NULL_HANDLE, 0, VK_INDEX_TYPE_UINT32,
                                   uint32_t(indices.size())};
  draw.pass_index = AddPendingPass(input, index_binding, indices, vertex, pixel);
  if (draw.pass_index != SIZE_MAX) {
    pending_.emplace_back(std::move(draw));
  }
}

size_t NativeCommandProcessor::AddPendingPass(const DrawInput& input,
                                              const IndexBinding& index_binding,
                                              std::span<const uint32_t> inline_indices,
                                              const Shader::Translation& vertex,
                                              const Shader::Translation* pixel,
                                              VkBuffer shared_vertex_buffer,
                                              const PreuploadedConstants* preuploaded_constants) {
  auto& pass = pending_passes_.emplace_back(*vulkan_device_, vma_allocator_);
  const uint32_t shared_count = uint32_t(1) << translator_->GetSharedMemoryStorageBufferCountLog2();
  const size_t pipelines_before = pipelines_.size();
  if (!pass.Initialize(input, index_binding, inline_indices, vertex, pixel, shared_count,
                       pipeline_cache_, pipelines_, draw_bindings_, *draw_uploads_, clock_.next,
                       DrawBindings::kPendingTextureSetSlot, shared_vertex_buffer,
                       &pipeline_cache_mutex_, preuploaded_constants)) {
    pending_passes_.pop_back();
    return SIZE_MAX;
  }
  if (pipelines_.size() > pipelines_before) {
    last_pipeline_created_swap_ = guest_frame_count();
  }
  return pending_passes_.size() - 1;
}

bool NativeCommandProcessor::IssueCopy() {
  if (!FlushPendingIfFull()) {
    return false;
  }
  const auto& regs = *register_file_;
  const auto surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  const bool copy_native_1x = IsDrawScaleNativeForPitch(
      surface_info.surface_pitch, surface_info.msaa_samples,
      REXCVAR_GET(draw_resolution_scale_threshold), draw_resolution_scale_x_,
      draw_resolution_scale_y_);
  const uint32_t sx = copy_native_1x ? 1u : draw_resolution_scale_x_;
  const uint32_t sy = copy_native_1x ? 1u : draw_resolution_scale_y_;
  const bool copy_scaled =
      !copy_native_1x && (draw_resolution_scale_x_ > 1 || draw_resolution_scale_y_ > 1);
  const EdramLayout layout = ActiveEdramLayout();
  draw_util::ResolveInfo copy;
  if (!draw_util::GetResolveInfo(regs, *memory_, sx, sy, false, false, layout, copy)) {
    return false;
  }
  if (!copy.coordinate_info.width_div_8 || !copy.height_div_8) {
    return true;
  }
  std::optional<QueuedDepthClear> depth_clear;
  if (copy.IsClearingDepth()) {
    const auto depth_format = xenos::DepthRenderTargetFormat(copy.depth_edram_info.format);
    auto depth = FindTarget(copy.depth_original_base, surface_info.surface_pitch,
                            surface_info.msaa_samples, target_cache_->GetDepthFormat(depth_format),
                            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false, 0,
                            copy_scaled);
    if (!depth) {
      return false;
    }
    const auto extent = depth->extent();
    const auto rect = GetResolveRectangle(copy, layout, extent.width, extent.height, true, sx, sy);
    if (!rect) {
      return false;
    }
    // Claims the cleared rows (a hit now, so only the ownership is updated).
    FindTarget(copy.depth_original_base, surface_info.surface_pitch, surface_info.msaa_samples,
               target_cache_->GetDepthFormat(depth_format),
               VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false,
               (rect->y + rect->height + sy - 1) / sy, copy_scaled);
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
  const bool average_selection =
      (samples == xenos::MsaaSamples::k2X && selection == xenos::CopySampleSelect::k01) ||
      (samples == xenos::MsaaSamples::k4X && selection == xenos::CopySampleSelect::k0123);
  const bool average_all = cmd_resolve_image_ && average_selection;
  const bool depth = copy.IsCopyingDepth();
  const VkFormat color_format =
      ColorTargetFormat(xenos::ColorRenderTargetFormat(copy.color_edram_info.format));
  const bool single_sample = !depth && samples != xenos::MsaaSamples::k1X &&
                             xenos::IsSingleCopySampleSelected(selection);
  // Slices of an array destination are layers of one cached resolve image.
  constexpr uint32_t kResolveArrayLayers = 8;
  const bool dest_array = copy.copy_dest_info.copy_dest_array != 0;
  const int32_t exp_bias = depth ? 0 : copy.copy_dest_info.copy_dest_exp_bias;
  // The compute resolve picks one sample or averages them, and applies the exp
  // bias and the destination layer.
  const bool compute_resolve =
      !depth && color_format != VK_FORMAT_UNDEFINED &&
      (single_sample || exp_bias != 0 || dest_array) &&
      (samples == xenos::MsaaSamples::k1X || single_sample || average_selection);
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
       ((samples != xenos::MsaaSamples::k1X && !average_all && !compute_resolve) ||
        color_format == VK_FORMAT_UNDEFINED ||
        (color_format != output_format && (!format_convert || compute_resolve)) ||
        ((exp_bias != 0 || dest_array) && !compute_resolve))) ||
      (depth && dest_array) || !IsPhysicalRange(resolve_address, 1) ||
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
                                   target_cache_->GetDepthFormat(depth_format),
                                   VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false,
                                   0, copy_scaled)
                      : FindTarget(copy.color_original_base, surface_info.surface_pitch, samples,
                                   color_format, VK_IMAGE_ASPECT_COLOR_BIT, color_is_64bpp, 0,
                                   copy_scaled);
  if (!source) {
    return false;
  }
  const uint32_t copy_pitch_tiles =
      xenos::GetSurfacePitchTiles(surface_info.surface_pitch, samples, false);
  if (copy_pitch_tiles) {
    if (copy.IsClearingDepth() || depth) {
      target_cache_->pending_fast_clears().erase({copy.depth_original_base, copy_pitch_tiles});
    }
    if (copy.IsClearingColor() || !depth) {
      target_cache_->pending_fast_clears().erase({copy.color_original_base, copy_pitch_tiles});
      const uint32_t color_copy_pitch_tiles =
          xenos::GetSurfacePitchTiles(surface_info.surface_pitch, samples, color_is_64bpp);
      if (color_copy_pitch_tiles) {
        target_cache_->ErasePendingColorFastClears(copy.color_original_base, color_copy_pitch_tiles);
      }
    }
  }
  const auto source_extent = source->extent();
  const auto rect =
      GetResolveRectangle(copy, layout, source_extent.width, source_extent.height, depth, sx, sy);
  if (!rect) {
    return false;
  }
  // Claims the resolved rows (a hit now, so only the ownership is updated).
  const uint32_t resolve_height = (rect->y + rect->height + sy - 1) / sy;
  if (depth) {
    FindTarget(copy.depth_original_base, surface_info.surface_pitch, samples,
               target_cache_->GetDepthFormat(depth_format),
               VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false, resolve_height,
               copy_scaled);
  } else {
    FindTarget(copy.color_original_base, surface_info.surface_pitch, samples, color_format,
               VK_IMAGE_ASPECT_COLOR_BIT, color_is_64bpp, resolve_height, copy_scaled);
  }
  const VkExtent2D extent{rect->width, rect->height};

  const uint32_t dest_info = copy.copy_dest_info.value;
  const auto dest_pitch_reg = regs.Get<reg::RB_COPY_DEST_PITCH>();
  const uint32_t dest_pitch = dest_pitch_reg.value;
  const uint32_t guest_width =
      uint32_t(copy.coordinate_info.width_div_8) << xenos::kResolveAlignmentPixelsLog2;
  const uint32_t guest_height = copy.height_div_8 << xenos::kResolveAlignmentPixelsLog2;
  const VkExtent2D target_extent = GetResolveDestinationExtent(
      guest_width, guest_height, dest_pitch_reg.copy_dest_pitch, dest_pitch_reg.copy_dest_height,
      sx, sy);
  bool tiled_destination =
      target_extent.width != extent.width || target_extent.height != extent.height;
  const bool has_offset = copy.copy_dest_x || copy.copy_dest_y;
  auto existing = resolved_images().find(resolve_address);
  std::shared_ptr<ColorImage> destination;
  VkOffset2D destination_offset{};
  uint32_t dest_resolve_address = resolve_address;
  uint32_t dest_pitch_to_store = dest_pitch;
  const uint32_t dest_layers = dest_array ? kResolveArrayLayers : 1;
  if (existing != resolved_images().end() && existing->second.image->format() == output_format &&
      existing->second.image->array_layers() == dest_layers &&
      existing->second.resolution_scaled == copy_scaled) {
    const auto existing_extent = existing->second.image->extent();
    if (!has_offset && existing_extent.width == target_extent.width &&
        existing_extent.height == target_extent.height) {
      destination = existing->second.image;
    } else if (has_offset && existing->second.dest_info == dest_info &&
               reg::RB_COPY_DEST_PITCH{existing->second.dest_pitch}.copy_dest_pitch ==
                   dest_pitch_reg.copy_dest_pitch) {
      const auto placed =
          GetCachedResolveRectangle(copy, existing_extent.width, existing_extent.height, sx, sy);
      if (placed) {
        destination = existing->second.image;
        destination_offset = {int32_t(placed->x), int32_t(placed->y)};
        dest_pitch_to_store = existing->second.dest_pitch;
        tiled_destination |= existing->second.tiled_destination;
      }
    }
  }
  if (!destination && dest_pitch_reg.copy_dest_pitch != 0 && !dest_array) {
    const uint32_t bpp_log2 = GetResolveBytesPerBlockLog2(output_format);
    uint64_t best_version = 0;
    for (const auto& [base_addr, entry] : resolved_images()) {
      if (!entry.tiled_destination || base_addr >= resolve_address ||
          entry.last_use_submission + 4 < clock_.next ||
          entry.image->format() != output_format || entry.dest_info != dest_info ||
          entry.resolution_scaled != copy_scaled ||
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
      const uint64_t dest_x = uint64_t(tile_x + copy.copy_dest_x) * sx;
      const uint64_t dest_y = uint64_t(tile_y + copy.copy_dest_y) * sy;
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
    if (existing == resolved_images().end() && resolved_images().size() >= NativeTextureCache::kMaxResolvedImages) {
      // Drop the least recently used resolve whose GPU work has completed. Only
      // retired submissions are eligible, and the presented image is kept; the
      // shared owners in pending and in-flight commands keep surviving images alive.
      const auto victim = SelectRetiredEviction(resolved_images(), clock_.retired + 1,
                                                [this](uint32_t address, const CachedImage& entry) {
                                                  return entry.image != presented_ &&
                                                         address != presented_resolve_address();
                                                });
      if (!victim) {
        return false;
      }
      resolved_images().erase(*victim);
      texture_cache_->PruneResolvedDependents();
      texture_cache_->RebuildCachedTexturePages();
    }
    destination = std::make_shared<ColorImage>(*vulkan_device_, vma_allocator_);
    if (!destination->Initialize(target_extent, output_format, VK_SAMPLE_COUNT_1_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT, dest_layers)) {
      return false;
    }
  }
  const size_t depth_descriptor = depth ? depth_resolve_->Allocate() : SIZE_MAX;
  const size_t color_descriptor = compute_resolve ? color_resolve_->Allocate() : SIZE_MAX;
  if ((depth && depth_descriptor == SIZE_MAX) || (compute_resolve && color_descriptor == SIZE_MAX)) {
    return false;
  }
  texture_cache_->InvalidateTextures(copy.copy_dest_extent_start, copy.copy_dest_extent_length);
  bool erased_other_resolved = false;
  for (auto it = resolved_images().begin(); it != resolved_images().end();) {
    if (it->first != dest_resolve_address &&
        RangesOverlap(copy.copy_dest_extent_start, copy.copy_dest_extent_length,
                      it->second.guest_start, it->second.guest_length)) {
      if (presented_resolve_address() && *presented_resolve_address() == it->first) {
        presented_resolve_address().reset();
      }
      it = resolved_images().erase(it);
      erased_other_resolved = true;
    } else {
      ++it;
    }
  }
  uint32_t guest_start = copy.copy_dest_extent_start;
  uint32_t guest_end = copy.copy_dest_extent_start + copy.copy_dest_extent_length;
  if (auto prev = resolved_images().find(dest_resolve_address);
      prev != resolved_images().end() && prev->second.image == destination &&
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
      !copy.copy_dest_info.copy_dest_swap && !dest_array &&
      uint32_t(copy.copy_dest_info.copy_dest_endian) <= 3;
  const uint32_t copy_dest_pitch = regs.Get<reg::RB_COPY_DEST_PITCH>().copy_dest_pitch;
  const ResolveRectangle guest_rect{
      copy.copy_dest_x, copy.copy_dest_y,
      uint32_t(copy.coordinate_info.width_div_8) << xenos::kResolveAlignmentPixelsLog2,
      copy.height_div_8 << xenos::kResolveAlignmentPixelsLog2};
  CachedResolve resolve_entry{{destination, clock_.next},
                              dest_info,
                              dest_pitch_to_store,
                              next_resolve_version_++,
                              tiled_destination,
                              guest_start,
                              guest_length,
                              guest_rect,
                              copy_dest_pitch,
                              copy.copy_dest_info.copy_dest_endian,
                              rgba8_readback_eligible,
                              copy_scaled};
  resolved_images().insert_or_assign(dest_resolve_address, std::move(resolve_entry));
  if (erased_other_resolved) {
    texture_cache_->PruneResolvedDependents();
    texture_cache_->RebuildCachedTexturePages();
  } else {
    texture_cache_->MarkCachedPages(guest_start, guest_length);
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
                    single_sample ? int32_t(GetResolveSampleIndex(samples, selection))
                    : average_selection ? -1
                                        : 0,
                    format_convert,
                    std::ldexp(1.0f, exp_bias),
                    dest_array ? copy.copy_dest_info.copy_dest_slice : 0u});
  if (depth_clear) {
    pending_.emplace_back(std::move(*depth_clear));
  }
  // shortcut: immediate RGBA8 copies only, add delayed modes and other layouts with native packing.
  if (rgba8_readback_eligible) {
    if (GetReadbackResolveMode(false) == ReadbackResolveMode::kFull &&
        IsResolveReadbackWanted(copy.copy_dest_extent_start, copy.copy_dest_extent_length)) {
      return ReadBackResolve(copy, resolve_address, copy_dest_pitch,
                             std::get<QueuedResolve>(pending_[resolve_index]), sx, sy);
    }
    NoteResolvedRange(copy.copy_dest_extent_start, copy.copy_dest_extent_length, 32, copy_scaled);
  }
  return true;
}

bool NativeCommandProcessor::CaptureResolveImage(ColorImage& image, VkOffset2D src_offset,
                                                 VkExtent2D src_extent, uint32_t scale_x,
                                                 uint32_t scale_y,
                                                 std::vector<uint8_t>& pixels_out) {
  const bool scaled = scale_x > 1 || scale_y > 1;
  const VkExtent2D guest_extent{src_extent.width / scale_x, src_extent.height / scale_y};
  if (scaled && (!cmd_blit_image_ || !guest_extent.width || !guest_extent.height)) {
    return false;
  }
  std::optional<ColorImage> downsampled;
  if (scaled) {
    downsampled.emplace(*vulkan_device_, vma_allocator_);
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
  const uint32_t frame = current_frame_;
  if (!SubmitCommands() || !RetireAllSubmissions()) {
    return false;
  }
  const auto mapped = readback->Map(frames_[frame].fence);
  if (!recorded || mapped.empty()) {
    return false;
  }
  pixels_out.assign(mapped.begin(), mapped.end());
  return true;
}

bool NativeCommandProcessor::ReadBackMemexport(
    std::span<const draw_util::MemExportRange> ranges) {
  uint64_t total_size = 0;
  for (const auto& range : ranges) {
    total_size += range.size_bytes;
  }
  if (!total_size) {
    return true;
  }
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device_, total_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kReadback, buffer, memory)) {
    REXLOG_ERROR("Native Vulkan: could not create a {} byte memexport readback buffer",
                 total_size);
    return false;
  }
  const auto release = [&] {
    vk.vkDestroyBuffer(device, buffer, nullptr);
    vk.vkFreeMemory(device, memory, nullptr);
  };
  if (!BeginCommands()) {
    release();
    return false;
  }
  RecordPending();
  std::vector<VkBufferCopy> copies;
  VkDeviceSize offset = 0;
  for (const auto& range : ranges) {
    copies.push_back({VkDeviceSize(range.base_address_dwords) << 2, offset, range.size_bytes});
    offset += range.size_bytes;
  }
  vk.vkCmdCopyBuffer(command_buffer_, shared_vertex_->buffer(), buffer, uint32_t(copies.size()),
                     copies.data());
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer;
  barrier.size = VK_WHOLE_SIZE;
  vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
  void* mapped = nullptr;
  const VkMappedMemoryRange whole{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory, 0,
                                  VK_WHOLE_SIZE};
  if (!SubmitCommands() || !RetireAllSubmissions() ||
      vk.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
    release();
    return false;
  }
  vk.vkInvalidateMappedMemoryRanges(device, 1, &whole);
  // The CPU writes guest memory directly, bypassing the access watches; the
  // shared vertex buffer already holds the same bytes.
  const auto* bytes = static_cast<const uint8_t*>(mapped);
  for (const auto& range : ranges) {
    const uint32_t address = range.base_address_dwords << 2;
    const auto access = IsPhysicalRange(address, range.size_bytes)
                            ? memory_->GetPhysicalHeap()->QueryRangeAccess(
                                  address, address + range.size_bytes - 1)
                            : memory::PageAccess::kNoAccess;
    if (access == memory::PageAccess::kReadWrite ||
        access == memory::PageAccess::kExecuteReadWrite) {
      std::memcpy(memory_->TranslatePhysical(address), bytes, range.size_bytes);
    }
    bytes += range.size_bytes;
  }
  vk.vkUnmapMemory(device, memory);
  release();
  return true;
}

bool NativeCommandProcessor::ReadBackResolve(const draw_util::ResolveInfo& copy, uint32_t address,
                                             uint32_t pitch, const QueuedResolve& resolve,
                                             uint32_t scale_x, uint32_t scale_y) {
  const uint32_t end = copy.copy_dest_extent_start + copy.copy_dest_extent_length;
  const auto access = memory_->GetPhysicalHeap()->QueryRangeAccess(address, end - 1);
  if (access != memory::PageAccess::kReadWrite && access != memory::PageAccess::kExecuteReadWrite) {
    return true;
  }
  std::vector<uint8_t> pixels;
  if (!CaptureResolveImage(*resolve.destination, resolve.destination_offset, resolve.extent,
                           scale_x, scale_y, pixels)) {
    return false;
  }
  const ResolveRectangle rect{copy.copy_dest_x, copy.copy_dest_y, resolve.extent.width / scale_x,
                              resolve.extent.height / scale_y};
  const bool stored = StoreTiledResolveRgba8({memory_->TranslatePhysical(address), end - address},
                                             pitch, rect, copy.copy_dest_info.copy_dest_endian,
                                             pixels);
  if (stored) {
    // The CPU wrote guest memory directly, bypassing the access-violation watch,
    // so anything cached from the written range is stale now.
    texture_cache_->InvalidateTextures(address, end - address);
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
  for (const auto& [base_addr, entry] : resolved_images()) {
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
  const uint32_t sx = best->resolution_scaled ? draw_resolution_scale_x_ : 1;
  const uint32_t sy = best->resolution_scaled ? draw_resolution_scale_y_ : 1;
  const VkOffset2D src_offset{int32_t(best->last_rect.x * sx), int32_t(best->last_rect.y * sy)};
  const VkExtent2D src_extent{best->last_rect.width * sx, best->last_rect.height * sy};
  std::vector<uint8_t> pixels;
  if (!CaptureResolveImage(*best->image, src_offset, src_extent, sx, sy, pixels)) {
    return false;
  }
  std::vector<uint8_t> tiled(entry_end - best_base);
  if (!StoreTiledResolveRgba8(tiled, best->copy_dest_pitch, best->last_rect, best->endian,
                              pixels)) {
    return false;
  }
  std::memcpy(memory_->TranslatePhysical(clip_first), tiled.data() + (clip_first - best_base),
              clip_last - clip_first);
  texture_cache_->InvalidateTextures(clip_first, clip_last - clip_first);
  InvalidateSharedVertices(clip_first, clip_last - clip_first);
  return true;
}

void NativeCommandProcessor::RecordDraws(std::span<const FrameCommand> commands) {
  const auto& vk = vulkan_device_->functions();
  const auto& draw = std::get<QueuedDraw>(commands.front());
  for (const auto& command : commands) {
    pending_passes_[std::get<QueuedDraw>(command).pass_index].RecordTextureUpload(command_buffer_);
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
    // Memexport-only draws have no attachments.
    attachment_extent = draw.memexport ? VkExtent2D{1, 1} : VkExtent2D{};
  }
  if (draw.memexport) {
    // Earlier draws reading the memory this one writes must be done.
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = shared_vertex_->buffer();
    barrier.size = VK_WHOLE_SIZE;
    vk.vkCmdPipelineBarrier(command_buffer_, kSharedMemoryShaderStages, kSharedMemoryShaderStages,
                            0, 0, nullptr, 1, &barrier, 0, nullptr);
  }
  rendering.renderArea.extent = attachment_extent;
  if (!needs_full_clear) {
    VkExtent2D covered{};
    for (const auto& command : commands) {
      const auto& scissor = pending_passes_[std::get<QueuedDraw>(command).pass_index].scissor();
      covered.width = std::max(covered.width, scissor.offset[0] + scissor.extent[0]);
      covered.height = std::max(covered.height, scissor.offset[1] + scissor.extent[1]);
    }
    rendering.renderArea.extent = {std::min(covered.width, attachment_extent.width),
                                   std::min(covered.height, attachment_extent.height)};
  }
  vk.vkCmdBeginRendering(command_buffer_, &rendering);
  DrawRenderState state;
  for (const auto& command : commands) {
    pending_passes_[std::get<QueuedDraw>(command).pass_index].Record(command_buffer_,
                                                                     attachment_extent, state);
  }
  vk.vkCmdEndRendering(command_buffer_);
  if (draw.memexport) {
    // Make the exported data visible to later draws, copies and readback.
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = shared_vertex_->buffer();
    barrier.size = VK_WHOLE_SIZE;
    vk.vkCmdPipelineBarrier(command_buffer_, kSharedMemoryShaderStages,
                            kSharedMemoryShaderStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                            nullptr, 1, &barrier, 0, nullptr);
  }
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

void NativeCommandProcessor::RecordTileTransfer(const QueuedTileTransfer& transfer) {
  auto& destination = *transfer.destination;
  if (!transfer.source->initialized()) {
    // Nothing was ever drawn to the aliased tiles.
    return;
  }
  if (!destination.initialized()) {
    // Unowned EDRAM outside the copied tiles reads as zero bits.
    if (destination.aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) {
      RecordDepthClear({transfer.destination, {{0, 0}, destination.extent()}, {0.0f, 0}});
    } else {
      destination.Clear(command_buffer_, {});
    }
  }
  constexpr auto transfer_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  transfer.source->Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              transfer_stage, VK_ACCESS_TRANSFER_READ_BIT);
  destination.Transition(command_buffer_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, transfer_stage,
                         VK_ACCESS_TRANSFER_WRITE_BIT);
  vulkan_device_->functions().vkCmdCopyImage(
      command_buffer_, transfer.source->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      destination.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(transfer.copies.size()),
      transfer.copies.data());
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
                           resolve.destination->format(),
                           resolve.source->samples() != VK_SAMPLE_COUNT_1_BIT,
                           resolve.exp_bias_scale, resolve.destination_layer);
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

bool NativeCommandProcessor::RetireFrame(FrameSlot& frame) {
  if (!frame.submitted) {
    return true;
  }
  const auto& vk = vulkan_device_->functions();
  const auto result =
      vk.vkWaitForFences(vulkan_device_->device(), 1, &frame.fence, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
    return false;
  }
  if (frame.frame) {
    LogFrameTiming(*frame.frame, result == VK_SUCCESS ? "rendered" : "gpu_failed");
    frame.frame.reset();
  }
  for (const auto& command : frame.commands) {
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
  frame.commands.clear();
  frame.passes.clear();
  const uint32_t frame_index = uint32_t(&frame - frames_.data());
  for (auto& [key, bindings] : draw_bindings_) {
    bindings->ReleaseFrameTextureSets(frame_index);
  }
  // A fence also waits for every earlier submission to the queue.
  if (result == VK_SUCCESS) {
    clock_.retired = std::max(clock_.retired, frame.submission);
    draw_uploads_->Reclaim(clock_.retired);
  }
  frame.submitted = false;
  if (result == VK_ERROR_DEVICE_LOST) {
    graphics_system_->OnHostGpuLossFromAnyThread(true);
  }
  return result == VK_SUCCESS;
}

bool NativeCommandProcessor::RetireAllSubmissions() {
  bool retired = true;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    retired &= RetireFrame(frames_[(current_frame_ + i) % kMaxFramesInFlight]);
  }
  return retired;
}

void NativeCommandProcessor::WaitForAllSubmissions() {
  const bool any_submitted =
      std::ranges::any_of(frames_, [](const FrameSlot& frame) { return frame.submitted; });
  if (!RetireAllSubmissions() && any_submitted) {
    auto queue = vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
    vulkan_device_->functions().vkQueueWaitIdle(queue.queue());
  }
  for (auto& frame : frames_) {
    frame.commands.clear();
    frame.passes.clear();
    frame.frame.reset();
    frame.submitted = false;
  }
  for (auto& [key, bindings] : draw_bindings_) {
    for (uint32_t f = 0; f <= DrawBindings::kPendingTextureSetSlot; ++f) {
      bindings->ReleaseFrameTextureSets(f);
    }
  }
  InvalidateConstantState();
}

void NativeCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t width, uint32_t height) {
  auto* presenter = graphics_system_->presenter();
  bool swap_scaled = true;
  std::optional<VkExtent2D> resolved_extent;
  if (const auto resolved = resolved_images().find(frontbuffer_ptr & (kPhysicalMemorySize - 1));
      resolved != resolved_images().end()) {
    resolved_extent = resolved->second.image->extent();
    swap_scaled = resolved->second.resolution_scaled;
  }
  const uint32_t sx = swap_scaled ? draw_resolution_scale_x_ : 1;
  const uint32_t sy = swap_scaled ? draw_resolution_scale_y_ : 1;
  const uint32_t scaled_width = width * sx;
  const uint32_t scaled_height = height * sy;
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
  if (index_cache_ && (guest_frame_count() % 60) == 0) {
    index_cache_->ClearRecentInvalidations();
  }
  if (pipelines_.size() > saved_pipeline_count_ &&
      (guest_frame_count() >= last_pipeline_created_swap_ + 120 ||
       pipelines_.size() - saved_pipeline_count_ >= 128)) {
    SavePipelineCache();
  }
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
    frames_[last_submitted_frame_].frame = timing;
  }
}

bool NativeCommandProcessor::BeginCommands() {
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  // Only the slot submitted kMaxFramesInFlight submissions ago is waited for.
  auto& frame = frames_[current_frame_];
  if (!RetireFrame(frame)) {
    return false;
  }
  if (vk.vkResetFences(device, 1, &frame.fence) != VK_SUCCESS ||
      vk.vkResetCommandPool(device, frame.command_pool, 0) != VK_SUCCESS) {
    return false;
  }
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vk.vkBeginCommandBuffer(frame.command_buffer, &begin) != VK_SUCCESS) {
    return false;
  }
  command_buffer_ = frame.command_buffer;
  presented_.reset();
  return true;
}

void NativeCommandProcessor::RecordBufferUploads(std::span<const FrameCommand> commands) {
  const auto& vk = vulkan_device_->functions();
  VkBufferMemoryBarrier pre{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  pre.srcAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.srcQueueFamilyIndex = pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  pre.buffer = shared_vertex_->buffer();
  pre.offset = 0;
  pre.size = VK_WHOLE_SIZE;
  vk.vkCmdPipelineBarrier(command_buffer_,
                          kSharedMemoryShaderStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &pre, 0, nullptr);
  for (const auto& command : commands) {
    const auto& upload = std::get<QueuedBufferUpload>(command);
    const VkBufferCopy copy{upload.src_offset, upload.dst_offset, upload.size};
    vk.vkCmdCopyBuffer(command_buffer_, upload.src_buffer, shared_vertex_->buffer(), 1, &copy);
  }
  VkBufferMemoryBarrier post{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  post.srcQueueFamilyIndex = post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  post.buffer = shared_vertex_->buffer();
  post.offset = 0;
  post.size = VK_WHOLE_SIZE;
  vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          kSharedMemoryShaderStages, 0, 0, nullptr, 1, &post, 0, nullptr);
}

void NativeCommandProcessor::RecordIndexUploads(std::span<const FrameCommand> commands) {
  const auto& vk = vulkan_device_->functions();
  for (const auto& command : commands) {
    const auto& upload = std::get<QueuedIndexUpload>(command);
    const VkBufferCopy copy{upload.src_offset, upload.dst_offset, upload.size};
    vk.vkCmdCopyBuffer(command_buffer_, upload.src_buffer, upload.dst_buffer, 1, &copy);
  }
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
  vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0,
                          nullptr);
}

size_t NativeCommandProcessor::RecordPending() {
  // Clear the fallback textures and move them to the shader-read layout once,
  // before any draw that may sample them.
  texture_cache_->RecordNullTextureInitialization(command_buffer_);
  if (!builtin_quad_indices_uploaded_ && builtin_quad_indices_ != VK_NULL_HANDLE) {
    const VkDeviceSize quad_bytes =
        VkDeviceSize(kMaxBuiltinQuadCount * 5 - 1) * sizeof(uint32_t);
    VkBuffer src_buf = VK_NULL_HANDLE;
    VkDeviceSize src_off = 0;
    if (uint8_t* dst = draw_uploads_->Request(clock_.next, quad_bytes, 4, src_buf, src_off)) {
      auto* out = reinterpret_cast<uint32_t*>(dst);
      for (uint32_t p = 0; p < kMaxBuiltinQuadCount; ++p) {
        if (p) {
          *(out++) = UINT32_MAX;
        }
        const uint32_t base = p * 4;
        out[0] = base;
        out[1] = base + 1;
        out[2] = base + 2;
        out[3] = base + 3;
        out += 4;
      }
      const auto& vk = vulkan_device_->functions();
      const VkBufferCopy copy{src_off, 0, quad_bytes};
      vk.vkCmdCopyBuffer(command_buffer_, src_buf, builtin_quad_indices_, 1, &copy);
      VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = builtin_quad_indices_;
      barrier.size = VK_WHOLE_SIZE;
      vk.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &barrier, 0,
                              nullptr);
      builtin_quad_indices_uploaded_ = true;
    }
  }
  size_t drawn = 0;
  for (size_t first = 0; first < pending_.size();) {
    const auto& command = pending_[first];
    if (const auto* draw = std::get_if<QueuedDraw>(&command)) {
      size_t end = first + 1;
      size_t last_draw_end = first + 1;
      bool has_interleaved_upload = false;
      // A memexport draw is recorded alone, so the barriers around it can be
      // outside the render pass. Hoist hoist-safe buffer/index uploads inside a
      // compatible draw run ahead of the render pass so it stays unbroken.
      for (; end < pending_.size() && !draw->memexport; ++end) {
        const auto& candidate = pending_[end];
        if (const auto* next = std::get_if<QueuedDraw>(&candidate)) {
          if (next->memexport || next->colors != draw->colors || next->depth != draw->depth) {
            break;
          }
          last_draw_end = end + 1;
        } else if (std::holds_alternative<QueuedIndexUpload>(candidate)) {
          has_interleaved_upload = true;
        } else if (const auto* buf = std::get_if<QueuedBufferUpload>(&candidate)) {
          if (!buf->hoist_safe) {
            break;
          }
          has_interleaved_upload = true;
        } else {
          break;
        }
      }
      if (has_interleaved_upload && last_draw_end > first + 1) {
        auto run_begin = pending_.begin() + ptrdiff_t(first);
        auto run_end = pending_.begin() + ptrdiff_t(last_draw_end);
        auto mid1 = std::stable_partition(run_begin, run_end, [](const FrameCommand& c) {
          return std::holds_alternative<QueuedBufferUpload>(c);
        });
        auto mid2 = std::stable_partition(mid1, run_end, [](const FrameCommand& c) {
          return std::holds_alternative<QueuedIndexUpload>(c);
        });
        if (mid1 != run_begin) {
          RecordBufferUploads(std::span(&*run_begin, size_t(mid1 - run_begin)));
        }
        if (mid2 != mid1) {
          RecordIndexUploads(std::span(&*mid1, size_t(mid2 - mid1)));
        }
        RecordDraws(std::span(&*mid2, size_t(run_end - mid2)));
        drawn += size_t(run_end - mid2);
      } else {
        RecordDraws(std::span(pending_).subspan(first, last_draw_end - first));
        drawn += last_draw_end - first;
      }
      first = last_draw_end;
    } else if (std::holds_alternative<QueuedBufferUpload>(command)) {
      size_t end = first + 1;
      for (; end < pending_.size(); ++end) {
        if (!std::holds_alternative<QueuedBufferUpload>(pending_[end])) {
          break;
        }
      }
      RecordBufferUploads(std::span(pending_).subspan(first, end - first));
      first = end;
    } else if (std::holds_alternative<QueuedIndexUpload>(command)) {
      size_t end = first + 1;
      for (; end < pending_.size(); ++end) {
        if (!std::holds_alternative<QueuedIndexUpload>(pending_[end])) {
          break;
        }
      }
      RecordIndexUploads(std::span(pending_).subspan(first, end - first));
      first = end;
    } else {
      if (const auto* resolve = std::get_if<QueuedResolve>(&command)) {
        RecordResolve(*resolve);
      } else if (const auto* transfer = std::get_if<QueuedTileTransfer>(&command)) {
        RecordTileTransfer(*transfer);
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
  auto& frame = frames_[current_frame_];
  // Granules committed while recording are bound before the commands using them run.
  bool wait_for_binds = false;
  VkResult result =
      shared_vertex_ ? shared_vertex_->FlushBinds(frame.sparse_bind_semaphore, wait_for_binds)
                     : VK_SUCCESS;
  if (result == VK_SUCCESS) {
    constexpr VkPipelineStageFlags kBindWaitStages = VK_PIPELINE_STAGE_TRANSFER_BIT |
                                                     VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = wait_for_binds ? 1 : 0;
    submit.pWaitSemaphores = &frame.sparse_bind_semaphore;
    submit.pWaitDstStageMask = &kBindWaitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    auto queue = vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
    result = vk.vkQueueSubmit(queue.queue(), 1, &submit, frame.fence);
  }
  if (result != VK_SUCCESS) {
    if (result == VK_ERROR_DEVICE_LOST)
      graphics_system_->OnHostGpuLossFromAnyThread(true);
    return false;
  }
  frame.submission = clock_.next++;
  frame.submitted = true;
  for (const auto& command : pending_) {
    if (const auto* resolve = std::get_if<QueuedResolve>(&command)) {
      resolve->destination->MarkResolved();
    }
  }
  frame.commands.swap(pending_);
  frame.passes.swap(pending_passes_);
  for (auto& [key, bindings] : draw_bindings_) {
    bindings->CommitPendingTextureSets(current_frame_);
  }
  InvalidateConstantState();
  last_submitted_frame_ = current_frame_;
  current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
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

  const auto resolved = resolved_images().find(frontbuffer_address & (kPhysicalMemorySize - 1));
  const bool use_pwl = resolved != resolved_images().end() &&
                       resolved->second.image->format() == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  if (resolved != resolved_images().end() && resolved->second.image->initialized() &&
      (resolved->second.image->format() == ColorImage::kFormat || use_pwl) &&
      resolved->second.image->extent().width >= extent.width &&
      resolved->second.image->extent().height >= extent.height) {
    presented_ = resolved->second.image;
    resolved->second.last_use_submission = clock_.next;
    presented_resolve_address() = resolved->first;
  } else {
    presented_resolve_address().reset();
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
    if (!output_gamma_->Record(command_buffer_, current_frame_, presented_->view(),
                               output.image_view(), extent, ramp, use_pwl)) {
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
