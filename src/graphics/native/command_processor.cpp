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
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/instance.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/provider.h>

#include "draw_input.h"
#include "graphics_system.h"

REXCVAR_DEFINE_STRING(native_frame_timing_path, "", "GPU/Native",
                      "Optional CSV for native Vulkan guest-output frame timing");

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
    case xenos::ColorFormat::k_2_10_10_10:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case xenos::ColorFormat::k_16_16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
    case xenos::ColorFormat::k_32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

VkClearColorValue UnpackColor(VkFormat format, uint32_t packed) {
  switch (format) {
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return {{float(packed & 1023) / 1023.0f, float((packed >> 10) & 1023) / 1023.0f,
               float((packed >> 20) & 1023) / 1023.0f, float(packed >> 30) / 3.0f}};
    case VK_FORMAT_R16G16_SFLOAT:
      return {{rex::xenos_half_to_float(uint16_t(packed)),
               rex::xenos_half_to_float(uint16_t(packed >> 16)), 0, 0}};
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

  translator_ =
      std::make_unique<Translator>(Translator::Features(vulkan_device_), false, false, false);
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
  SetActiveEdramLayout(RequestedEdramLayout());
  REXLOG_INFO("Native Vulkan: command processor initialized");
  return true;
}

void NativeCommandProcessor::ShutdownContext() {
  if (!vulkan_device_) {
    return;
  }
  const auto& vk = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  RetireSubmission();
  if (frame_timing_.is_open()) {
    frame_timing_.close();
  }
  pending_.clear();
  in_flight_.clear();
  draw_uploads_.reset();
  presented_.reset();
  presented_resolve_address_.reset();
  resolved_images_.clear();
  targets_.clear();
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
    if (binding.dimension != xenos::FetchOpDimension::k2D) {
      return false;
    }
    const uint32_t slot = binding.fetch_constant;
    if (indices[slot] < 0) {
      const auto fetch = register_file_->GetTextureFetch(slot);
      auto& texture = input.textures.emplace_back();
      if (!ReadTexture(fetch, texture)) {
        REXGPU_DEBUG(
            "Native Vulkan texture {} unsupported: format={} base={:05X} flags={:08X}/{:08X}", slot,
            uint32_t(fetch.format), uint32_t(fetch.base_address), fetch.dword_4, fetch.dword_5);
        return false;
      }
      input.system.texture_swizzled_signs[slot / 4] |= uint32_t(texture_util::SwizzleSigns(fetch))
                                                       << ((slot % 4) * 8);
      indices[slot] = int(input.textures.size() - 1);
    }
    input.texture_bindings.push_back(uint32_t(indices[slot]));
  }
  for (const auto& sampler : samplers) {
    const auto fetch = register_file_->GetTextureFetch(sampler.fetch_constant);
    auto& out = input.samplers.emplace_back();
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
  }
  return true;
}

bool NativeCommandProcessor::ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch,
                                         DrawTexture& texture) {
  if (fetch.type != xenos::FetchConstantType::kTexture ||
      fetch.dimension != xenos::DataDimension::k2DOrStacked || fetch.stacked ||

      fetch.sign_x == xenos::TextureSign::kSigned || fetch.sign_y == xenos::TextureSign::kSigned ||
      fetch.sign_z == xenos::TextureSign::kSigned || fetch.sign_w == xenos::TextureSign::kSigned) {
    return false;
  }

  texture.width = fetch.size_2d.width + 1;
  texture.height = fetch.size_2d.height + 1;
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
    case xenos::TextureFormat::k_32_FLOAT:
      texture.format = VK_FORMAT_R32_SFLOAT;
      block_size = 4;
      break;
    case xenos::TextureFormat::k_16_16_FLOAT:
      texture.format = VK_FORMAT_R16G16_SFLOAT;
      block_size = 4;
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
    default:
      return false;
  }

  const VkComponentSwizzle r = VK_COMPONENT_SWIZZLE_R;
  const std::array<VkComponentSwizzle, 8> swizzles{
      r,
      (block_size == 1 || depth_texture) ? r : VK_COMPONENT_SWIZZLE_G,
      block_size == 1   ? r
      : block_size == 2 ? VK_COMPONENT_SWIZZLE_G
                        : VK_COMPONENT_SWIZZLE_B,
      block_size == 1   ? r
      : block_size == 2 ? VK_COMPONENT_SWIZZLE_G
                        : VK_COMPONENT_SWIZZLE_A,
      VK_COMPONENT_SWIZZLE_ZERO,
      VK_COMPONENT_SWIZZLE_ONE,
      VK_COMPONENT_SWIZZLE_ZERO,
      VK_COMPONENT_SWIZZLE_ZERO};
  texture.swizzle = {swizzles[fetch.swizzle & 7], swizzles[(fetch.swizzle >> 3) & 7],
                     swizzles[(fetch.swizzle >> 6) & 7], swizzles[(fetch.swizzle >> 9) & 7]};
  uint32_t base_page, mip_page, mip_min, mip_max;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, &base_page,
                                                 &mip_page, &mip_min, &mip_max);
  if (!base_page || mip_min != 0) {
    return false;
  }
  const uint32_t address = base_page << 12;
  auto resolved = resolved_images_.find(address);
  if (!mip_max && resolved != resolved_images_.end() &&
      texture.format == resolved->second.image->format()) {
    const auto extent = resolved->second.image->extent();
    if (extent.width != texture.width || extent.height != texture.height) {
      return false;
    }
    resolved->second.last_use_submission = next_submission_;
    texture.resolved_image = resolved->second.image;
    return true;
  }

  const auto layout = texture_util::GetGuestTextureLayout(
      fetch.dimension, fetch.pitch, texture.width, texture.height, 1, fetch.tiled, fetch.format,
      fetch.packed_mips, true, mip_max);
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
    const auto source =
        ReadPhysical(uint32_t(level_address),
                     rex::align(size_t(level_layout.array_slice_data_extent_bytes), size_t(4)));
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
    auto& data = texture.levels.emplace_back(size_t(blocks_x) * blocks_y * block_size);
    for (uint32_t y = 0; y < blocks_y; ++y) {
      for (uint32_t x = 0; x < blocks_x; ++x) {
        const size_t offset =
            fetch.tiled
                ? size_t(texture_util::GetTiledOffset2D(int32_t(x + start_x), int32_t(y + start_y),
                                                        level_layout.row_pitch_bytes / block_size,
                                                        std::countr_zero(block_size)))
                : size_t(y + start_y) * level_layout.row_pitch_bytes + (x + start_x) * block_size;
        if (offset > source.size() || block_size > source.size() - offset) {
          return false;
        }
        uint8_t* dst = data.data() + (size_t(y) * blocks_x + x) * block_size;
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
  string::StringBuffer disassembly;
  candidate_vs.AnalyzeUcode(disassembly);
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
  std::vector<uint32_t> indices(index_count);
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
  if (ps) {
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
  Translator::Modification vertex_modification(translator_->GetDefaultVertexShaderModification(
      vs.GetDynamicAddressableRegisterCount(program.vs_num_reg), host_vertex_type));
  vertex_modification.vertex.interpolator_mask = interpolators;
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

  DrawInput input;
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
  input.front_face = raster.face ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  for (size_t channel = 0; channel < input.blend_constants.size(); ++channel) {
    input.blend_constants[channel] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED + uint32_t(channel));
  }
  if (ps && !ReadTextures(*ps, input)) {
    return false;
  }
  const auto& properties = vulkan_device_->properties();
  draw_util::GetHostViewportInfo(regs, 1, 1, false, properties.maxViewportDimensions[0],
                                 properties.maxViewportDimensions[1], true, depth_control,
                                 float_depth, true, ps && ps->writes_depth(), input.viewport);
  draw_util::GetScissor(regs, input.scissor);
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
    indices = QuadStripIndices(rectangle ? index_count / 3 : index_count);
  }
  if (point) {
    const auto minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    const auto size = regs.Get<reg::PA_SU_POINT_SIZE>();
    input.system.point_vertex_diameter_min = float(minmax.min_size) / 8.0f;
    input.system.point_vertex_diameter_max = float(minmax.max_size) / 8.0f;
    input.system.point_constant_diameter[0] = float(size.width) / 8.0f;
    input.system.point_constant_diameter[1] = float(size.height) / 8.0f;
    input.system.point_screen_diameter_to_ndc_radius[0] = 1.0f / float(input.viewport.xy_extent[0]);
    input.system.point_screen_diameter_to_ndc_radius[1] = 1.0f / float(input.viewport.xy_extent[1]);
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

  std::array<bool, 96> uploaded{};
  for (const auto& binding : vs.vertex_bindings()) {
    if (uploaded[binding.fetch_constant]) {
      continue;
    }
    auto fetch = regs.GetVertexFetch(binding.fetch_constant);
    if (fetch.type != xenos::FetchConstantType::kVertex) {
      return false;
    }
    const auto bytes = ReadPhysical(fetch.address << 2, fetch.size << 2);
    if (bytes.empty() || bytes.size() > properties.maxStorageBufferRange - input.vertices.size()) {
      return false;
    }
    fetch.address = uint32_t(input.vertices.size() / sizeof(uint32_t));
    std::memcpy(&input.fetch[binding.fetch_constant * 2], &fetch, sizeof(fetch));
    input.vertices.insert(input.vertices.end(), bytes.begin(), bytes.end());
    uploaded[binding.fetch_constant] = true;
  }

  if (input.vertices.empty()) {
    input.vertices.resize(sizeof(uint32_t));
  }

  QueuedDraw draw{};
  const auto layout = ActiveEdramLayout();
  const uint32_t writes =
      use_color ? draw_util::GetNormalizedColorMask(regs, ps->writes_color_targets()) : 0;
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
    draw.colors[i] = FindTarget(layout.DecodeBase(target.value), dimensions.surface_pitch,
                                dimensions.msaa_samples, format, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!draw.colors[i]) {
      return false;
    }
    if (std::find(draw.colors.begin(), draw.colors.begin() + i, draw.colors[i]) !=
        draw.colors.begin() + i) {
      draw.colors[i].reset();
      continue;
    }
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
                             pipelines_, draw_bindings_, *draw_uploads_, next_submission_)) {
    return false;
  }
  pending_.emplace_back(std::move(draw));
  return true;
}

VkFormat NativeCommandProcessor::GetDepthFormat(xenos::DepthRenderTargetFormat format) const {
  if (format == xenos::DepthRenderTargetFormat::kD24S8) {
    return VK_FORMAT_D24_UNORM_S8_UINT;
  }
  return VK_FORMAT_D32_SFLOAT_S8_UINT;
}

std::shared_ptr<ColorImage> NativeCommandProcessor::FindTarget(uint32_t tile, uint32_t pitch,
                                                               xenos::MsaaSamples samples,
                                                               VkFormat format,
                                                               VkImageAspectFlags aspects) {
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
  const uint32_t tiles_per_row = xenos::GetSurfacePitchTiles(pitch, samples, false);
  const uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> (samples >= xenos::MsaaSamples::k2X);
  const uint32_t height = ActiveEdramLayout().tile_count / tiles_per_row * tile_height;
  auto image = std::make_shared<ColorImage>(*vulkan_device_);
  if (!image->Initialize({pitch, height}, format, VkSampleCountFlagBits(1u << uint32_t(samples)),
                         aspects)) {
    return {};
  }
  // shortcut: separate host images, share overlapping tiles when ownership tracking is added.
  targets_.emplace(key, CachedImage{image, next_submission_});
  return image;
}

bool NativeCommandProcessor::IssueCopy() {
  if (!FlushPendingIfFull()) {
    return false;
  }
  const auto& regs = *register_file_;
  const EdramLayout layout = ActiveEdramLayout();
  draw_util::ResolveInfo copy;
  if (!draw_util::GetResolveInfo(regs, *memory_, 1, 1, false, false, layout, copy)) {
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
    const auto rect = GetResolveRectangle(copy, layout, extent.width, extent.height, true);
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
  const bool single_sample =
      !depth && samples != xenos::MsaaSamples::k1X && xenos::IsSingleCopySampleSelected(selection);
  const VkFormat color_format =
      ColorTargetFormat(xenos::ColorRenderTargetFormat(copy.color_edram_info.format));
  const VkFormat output_format =
      depth ? VK_FORMAT_R32_SFLOAT : CopyColorFormat(copy.copy_dest_info.copy_dest_format);
  const uint32_t resolve_address = regs[XE_GPU_REG_RB_COPY_DEST_BASE];
  if ((!depth && ((samples != xenos::MsaaSamples::k1X && !average_all && !single_sample) ||
                  color_format == VK_FORMAT_UNDEFINED || color_format != output_format ||
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
  auto source = depth ? FindTarget(copy.depth_original_base, surface_info.surface_pitch, samples,
                                   GetDepthFormat(depth_format),
                                   VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                      : FindTarget(copy.color_original_base, surface_info.surface_pitch, samples,
                                   color_format, VK_IMAGE_ASPECT_COLOR_BIT);
  if (!source) {
    return false;
  }
  const auto source_extent = source->extent();
  const auto rect =
      GetResolveRectangle(copy, layout, source_extent.width, source_extent.height, depth);
  if (!rect) {
    return false;
  }
  const VkExtent2D extent{rect->width, rect->height};

  const uint32_t dest_info = copy.copy_dest_info.value;
  const uint32_t dest_pitch = regs.Get<reg::RB_COPY_DEST_PITCH>().value;
  const bool has_offset = copy.copy_dest_x || copy.copy_dest_y;
  auto existing = resolved_images_.find(resolve_address);
  std::shared_ptr<ColorImage> destination;
  VkOffset2D destination_offset{};
  if (existing != resolved_images_.end() && existing->second.image->format() == output_format) {
    const auto existing_extent = existing->second.image->extent();
    if (!has_offset && existing_extent.width == extent.width &&
        existing_extent.height == extent.height) {
      destination = existing->second.image;
    } else if (has_offset && !depth && existing->second.image->initialized() &&
               existing->second.dest_info == dest_info && existing->second.dest_pitch == dest_pitch) {
      const auto placed =
          GetCachedResolveRectangle(copy, existing_extent.width, existing_extent.height);
      if (placed) {
        destination = existing->second.image;
        destination_offset = {int32_t(placed->x), int32_t(placed->y)};
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
    }
    destination = std::make_shared<ColorImage>(*vulkan_device_);
    if (!destination->Initialize(extent, output_format)) {
      return false;
    }
  }
  const size_t depth_descriptor = depth ? depth_resolve_->Allocate() : SIZE_MAX;
  const size_t color_descriptor = single_sample ? color_resolve_->Allocate() : SIZE_MAX;
  if ((depth && depth_descriptor == SIZE_MAX) || (single_sample && color_descriptor == SIZE_MAX)) {
    return false;
  }
  resolved_images_.insert_or_assign(
      resolve_address, CachedResolve{{destination, next_submission_}, dest_info, dest_pitch});
  pending_.emplace_back(
      QueuedResolve{std::move(source),
                    std::move(destination),
                    {int32_t(rect->x), int32_t(rect->y)},
                    destination_offset,
                    extent,
                    copy.IsClearingColor(),
                    UnpackColor(color_format, copy.rb_color_clear),
                    depth_descriptor,
                    uint32_t(selection),
                    depth_format == xenos::DepthRenderTargetFormat::kD24FS8 ? 2.0f : 1.0f,
                    color_descriptor,
                    single_sample ? GetResolveSampleIndex(samples, selection) : 0});
  if (depth_clear) {
    pending_.emplace_back(std::move(*depth_clear));
  }
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
  for (uint32_t i = 0; i < draw.color_count; ++i) {
    auto& color = colors[i];
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    const auto& image = draw.colors[i];
    if (!image) {
      continue;
    }
    rendering.renderArea.extent = image->extent();
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
    rendering.renderArea.extent = draw.depth->extent();
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
  const VkExtent2D attachment_extent = rendering.renderArea.extent;
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
                           {resolve.source_offset, resolve.extent},
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
    if (resolve.source->samples() == VK_SAMPLE_COUNT_1_BIT) {
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
  const bool ready = presenter && width && height &&
                     presenter->RefreshGuestOutput(
                         width, height, width, height,
                         [this, width, height,
                          frontbuffer_ptr](ui::Presenter::GuestOutputRefreshContext& context) {
                           return RenderGuestOutput(context, {width, height}, frontbuffer_ptr);
                         });
  const bool fresh = ready && presented_ && presented_->TakeResolvedOutput();
  if (!frame_timing_.is_open()) {
    return;
  }
  using namespace std::chrono;
  const FrameTiming timing{
      guest_frame_count(),
      frontbuffer_ptr,
      {width, height},
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
  presented_.reset();
  return true;
}

size_t NativeCommandProcessor::RecordPending() {
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
  if (pending_.size() < 512) {
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
      resolved->second.image->extent().width == extent.width &&
      resolved->second.image->extent().height == extent.height) {
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
    const auto ramp = use_pwl
                          ? std::as_bytes(std::span(gamma_ramp_pwl_rgb(), 128 * 3))
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
