#pragma once

#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <rex/graphics/command_processor.h>
#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/ui/presenter.h>

#include "color_resolve.h"
#include "depth_resolve.h"
#include "draw_input.h"
#include "draw_pass.h"
#include "image_readback.h"
#include "output_gamma.h"

namespace rex::graphics::native {

class NativeGraphicsSystem;

class NativeCommandProcessor : public CommandProcessor {
 public:
  NativeCommandProcessor(NativeGraphicsSystem* graphics_system, system::KernelState* kernel_state);
  ~NativeCommandProcessor() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t width, uint32_t height) override;
  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address, uint32_t dword_count) override;
  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info, bool major_mode_explicit) override;
  bool IssueCopy() override;

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;

 private:
  std::span<const uint8_t> ReadPhysical(uint32_t address, size_t size);
  bool ReadTextures(const SpirvShader& shader, DrawInput& input);
  bool ReadTexture(const xenos::xe_gpu_texture_fetch_t& fetch, DrawTexture& texture);
  bool InitializeNullTextures();
  void DestroyNullTextures();
  void RecordNullTextureInitialization();
  struct QueuedDraw {
    std::array<std::shared_ptr<ColorImage>, xenos::kMaxColorRenderTargets> colors;
    std::shared_ptr<ColorImage> depth;
    uint32_t color_count;
    std::unique_ptr<DrawPass> pass;
  };
  struct QueuedResolve {
    std::shared_ptr<ColorImage> source;
    std::shared_ptr<ColorImage> destination;
    VkOffset2D source_offset;
    VkOffset2D destination_offset;
    VkExtent2D extent;
    bool clear_color;
    VkClearColorValue clear;
    size_t depth_descriptor = SIZE_MAX;
    uint32_t depth_sample = 0;
    float depth_scale = 1.0f;
    size_t color_descriptor = SIZE_MAX;
    uint32_t color_sample = 0;
    bool format_convert = false;
  };
  struct QueuedDepthClear {
    std::shared_ptr<ColorImage> target;
    VkRect2D area;
    VkClearDepthStencilValue value;
  };
  using FrameCommand = std::variant<QueuedDraw, QueuedResolve, QueuedDepthClear>;

  // A 4x-MSAA depth-only draw that D3D9 issued to clear an aliased 1x- or
  // 2x-MSAA EDRAM tile. Kept so a target created later for the same tile is
  // initialized with the clear before it is drawn to.
  struct PendingFastClear {
    DrawInput input;
    std::vector<uint32_t> indices;
    const Shader::Translation* vertex = nullptr;
    xenos::DepthRenderTargetFormat depth_format = xenos::DepthRenderTargetFormat::kD24S8;
    bool color_clear_capable = false;
    uint32_t clear_stencil = 0;
  };
  struct PendingColorFastClear {
    DrawInput input;
    std::vector<uint32_t> indices;
    const Shader::Translation* vertex = nullptr;
    const Shader::Translation* pixel = nullptr;
    uint32_t rt_index = 0;
    bool is_64bpp = false;
  };

  struct CachedImage {
    std::shared_ptr<ColorImage> image;
    uint64_t last_use_submission = 0;
  };
  struct CachedResolve : CachedImage {
    uint32_t dest_info;
    uint32_t dest_pitch;
    uint64_t version = 0;
  };
  struct CachedSampler {
    VkSampler sampler = VK_NULL_HANDLE;
    uint64_t last_use_submission = 0;
  };
  struct CachedTextureEntry {
    std::shared_ptr<CachedTexture> texture;
    uint64_t last_use_submission = 0;
  };
  static constexpr size_t kMaxTargets = 64;
  static constexpr size_t kMaxResolvedImages = 256;
  static constexpr size_t kMaxSamplers = 256;
  static constexpr size_t kMaxCachedTextures = 1024;
  static constexpr size_t kMaxResolvedCubes = 64;

  VkSampler GetSampler(const DrawSampler& key);
  void InvalidateTextures(uint32_t start, uint32_t length);
  static std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range);
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallback(uint32_t physical_address_start,
                                                           uint32_t length, bool exact_range);

  std::shared_ptr<ColorImage> FindTarget(uint32_t tile, uint32_t pitch, xenos::MsaaSamples samples,
                                         VkFormat format, VkImageAspectFlags aspects,
                                         bool is_64bpp = false);
  // Replays an aliased 4x-MSAA fast clear onto a 1x/2x target that already
  // exists (depth or color), appending the queued draw to pending_.
  void ReplayAliasedFastClear(const PendingFastClear& clear, const std::shared_ptr<ColorImage>& image,
                              xenos::MsaaSamples samples);
  // Finds a pending fast clear for an aliased target tile and replays it if it
  // is compatible with the newly created image, then drops it.
  void ApplyPendingFastClear(uint32_t tile, uint32_t pitch_tiles, xenos::MsaaSamples samples,
                             const std::shared_ptr<ColorImage>& image);
  void ReplayAliasedColorFastClear(const PendingColorFastClear& clear,
                                   const std::shared_ptr<ColorImage>& image,
                                   xenos::MsaaSamples samples);
  void ApplyPendingColorFastClear(uint32_t tile, uint32_t pitch_tiles, xenos::MsaaSamples samples,
                                  VkFormat format, const std::shared_ptr<ColorImage>& image);
  void ErasePendingColorFastClears(uint32_t tile, uint32_t pitch_tiles);
  VkFormat GetDepthFormat(xenos::DepthRenderTargetFormat format) const;
  bool BeginCommands();
  size_t RecordPending();
  bool SubmitCommands();
  bool FlushPendingIfFull();
  void RecordDraws(std::span<const FrameCommand> commands);
  void RecordResolve(const QueuedResolve& resolve);
  void RecordDepthClear(const QueuedDepthClear& clear);
  bool ReadBackResolve(const draw_util::ResolveInfo& copy, uint32_t address, uint32_t pitch,
                       const QueuedResolve& resolve);
  bool RenderGuestOutput(ui::Presenter::GuestOutputRefreshContext& context, VkExtent2D extent,
                         uint32_t frontbuffer_address);

  ui::vulkan::VulkanDevice* vulkan_device_ = nullptr;
  PFN_vkCmdResolveImage cmd_resolve_image_ = nullptr;
  PFN_vkCmdBlitImage cmd_blit_image_ = nullptr;
  bool r8_resolve_blit_ = false;
  bool rg8_resolve_blit_ = false;
  // Whether VK_FORMAT_D24_UNORM_S8_UINT supports optimal-tiling sampled image
  // and depth/stencil attachment usage on this device.
  bool depth_unorm24_supported_ = false;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  // 1x1 zero-cleared cube-compatible image shared by the 2D array and cube
  // fallback views bound for fetch constants that cannot produce texels.
  VkImage null_texture_image_ = VK_NULL_HANDLE;
  VkDeviceMemory null_texture_memory_ = VK_NULL_HANDLE;
  VkImageView null_texture_2d_view_ = VK_NULL_HANDLE;
  VkImageView null_texture_cube_view_ = VK_NULL_HANDLE;
  bool null_textures_initialized_ = false;
  std::multimap<uint64_t, std::unique_ptr<SpirvShader>> shaders_;
  std::unique_ptr<SpirvShaderTranslator> translator_;
  DrawPipelines pipelines_;
  DrawBindingCache draw_bindings_;
  std::unique_ptr<ui::vulkan::VulkanUploadBufferPool> draw_uploads_;
  uint64_t next_submission_ = 1;
  uint64_t next_resolve_version_ = 1;
  VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
  std::map<std::tuple<uint32_t, uint32_t, xenos::MsaaSamples, VkFormat>, CachedImage> targets_;
  std::map<std::pair<uint32_t, uint32_t>, PendingFastClear> pending_fast_clears_;
  std::map<std::tuple<uint32_t, uint32_t, VkFormat>, PendingColorFastClear>
      pending_color_fast_clears_;
  std::unordered_map<uint32_t, CachedResolve> resolved_images_;
  std::unordered_map<DrawSampler, CachedSampler, DrawSamplerHash> samplers_;
  std::unordered_map<GuestTextureKey, CachedTextureEntry, GuestTextureKeyHash> cached_textures_;
  std::unordered_map<ResolvedCubeKey, CachedTextureEntry, ResolvedCubeKeyHash> resolved_cubes_;
  // Written from guest threads while the global critical region is held, so it
  // only guards the pending list - never call into memory_ or Vulkan here.
  std::mutex invalidation_mutex_;
  std::vector<std::pair<uint32_t, uint32_t>> pending_invalidations_;
  std::atomic<bool> has_pending_invalidations_{false};
  void* memory_invalidation_handle_ = nullptr;
  std::vector<FrameCommand> pending_;
  std::vector<FrameCommand> in_flight_;
  // Reused by IssueDraw so decoding guest indices does not allocate per draw.
  std::vector<uint32_t> scratch_indices_;
  std::unique_ptr<ImageReadback> in_flight_readback_;
  std::shared_ptr<ColorImage> presented_;
  // Physical front buffer address of the last presented resolve. Kept across
  // submissions so eviction cannot drop the image the presenter is showing.
  std::optional<uint32_t> presented_resolve_address_;
  uint64_t retired_submission_ = 0;
  std::unique_ptr<OutputGamma> output_gamma_;
  std::unique_ptr<DepthResolve> depth_resolve_;
  std::unique_ptr<ColorResolve> color_resolve_;
  struct FrameTiming {
    uint64_t swap;
    uint32_t address;
    VkExtent2D extent;
    int64_t ready_ns;
  };
  std::ofstream frame_timing_;
  std::optional<FrameTiming> submitted_frame_;
  void LogFrameTiming(const FrameTiming& frame, const char* result);
  bool RetireSubmission();
  bool submitted_ = false;
  bool reported_frame_ = false;
};

}  // namespace rex::graphics::native
