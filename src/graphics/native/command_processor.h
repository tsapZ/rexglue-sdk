#pragma once

#include <atomic>
#include <filesystem>
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
#include "index_cache.h"
#include "output_gamma.h"
#include "shared_vertex_memory.h"
#include "target_cache.h"
#include "texture_cache.h"

namespace rex::graphics::native {

class NativeGraphicsSystem;

class NativeCommandProcessor : public CommandProcessor {
 public:
  NativeCommandProcessor(NativeGraphicsSystem* graphics_system, system::KernelState* kernel_state);
  ~NativeCommandProcessor() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t width, uint32_t height) override;
  void ClearCaches() override;
  void InvalidateGpuMemory() override;
  void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                               bool blocking) override;
  void FlushShaderStorage() override;
  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address, uint32_t dword_count) override;
  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info, bool major_mode_explicit) override;
  bool IssueCopy() override;

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;
  void WriteRegister(uint32_t index, uint32_t value) override;
  void WriteRegistersFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers) override;
  bool ReadBackResolvedRange(uint32_t address, uint32_t length, uint32_t bits_per_pixel,
                             bool scaled, uint32_t clip_start, uint32_t clip_end) override;

 private:
  std::span<const uint8_t> ReadPhysical(uint32_t address, size_t size);
  struct QueuedDraw {
    std::array<std::shared_ptr<ColorImage>, xenos::kMaxColorRenderTargets> colors;
    std::shared_ptr<ColorImage> depth;
    uint32_t color_count;
    // Index into the passes recorded with the same commands.
    size_t pass_index;
    bool memexport = false;
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
    // Negative to average all samples.
    int32_t color_sample = 0;
    bool format_convert = false;
    float exp_bias_scale = 1.0f;
    uint32_t destination_layer = 0;
  };
  struct QueuedDepthClear {
    std::shared_ptr<ColorImage> target;
    VkRect2D area;
    VkClearDepthStencilValue value;
  };
  struct QueuedBufferUpload {
    VkBuffer src_buffer = VK_NULL_HANDLE;
    VkDeviceSize src_offset = 0;
    VkDeviceSize dst_offset = 0;
    VkDeviceSize size = 0;
    bool hoist_safe = true;
  };
  struct QueuedIndexUpload {
    VkBuffer src_buffer = VK_NULL_HANDLE;
    VkDeviceSize src_offset = 0;
    VkBuffer dst_buffer = VK_NULL_HANDLE;
    VkDeviceSize dst_offset = 0;
    VkDeviceSize size = 0;
  };
  // EDRAM tiles an aliasing target left behind, copied into the target now using them.
  struct QueuedTileTransfer {
    std::shared_ptr<ColorImage> source;
    std::shared_ptr<ColorImage> destination;
    std::vector<VkImageCopy> copies;
  };
  using FrameCommand = std::variant<QueuedDraw, QueuedResolve, QueuedDepthClear,
                                    QueuedBufferUpload, QueuedIndexUpload, QueuedTileTransfer>;
  using PendingFastClear = NativeTargetCache::PendingFastClear;
  using PendingColorFastClear = NativeTargetCache::PendingColorFastClear;
  using CachedResolve = NativeTextureCache::CachedResolve;

  std::unordered_map<uint32_t, CachedResolve>& resolved_images() {
    return texture_cache_->resolved_images();
  }
  std::optional<uint32_t>& presented_resolve_address() {
    return texture_cache_->presented_resolve_address();
  }
  void InvalidateSharedVertices(uint32_t start, uint32_t length) {
    if (shared_vertex_) {
      shared_vertex_->Invalidate(start, length);
    }
  }
  void DrainMemoryInvalidations();
  bool EnsureSharedVertexRange(uint32_t address, uint32_t size);
  static std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range);
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallback(uint32_t physical_address_start,
                                                           uint32_t length, bool exact_range);

  // used_height: guest rows about to be drawn or resolved (see NativeTargetCache).
  std::shared_ptr<ColorImage> FindTarget(uint32_t tile, uint32_t pitch, xenos::MsaaSamples samples,
                                         VkFormat format, VkImageAspectFlags aspects,
                                         bool is_64bpp = false, uint32_t used_height = 0,
                                         bool resolution_scaled = true);
  void QueueReplayDraw(const DrawInput& input, std::span<const uint32_t> indices,
                       const Shader::Translation& vertex, const Shader::Translation* pixel,
                       uint32_t rt_index, const std::shared_ptr<ColorImage>& image);
  bool BeginCommands();
  size_t RecordPending();
  bool SubmitCommands();
  bool FlushPendingIfFull();
  void RecordDraws(std::span<const FrameCommand> commands);
  void RecordBufferUploads(std::span<const FrameCommand> commands);
  void RecordIndexUploads(std::span<const FrameCommand> commands);
  void RecordResolve(const QueuedResolve& resolve);
  void RecordDepthClear(const QueuedDepthClear& clear);
  void RecordTileTransfer(const QueuedTileTransfer& transfer);
  bool CaptureResolveImage(ColorImage& image, VkOffset2D src_offset, VkExtent2D src_extent,
                           uint32_t scale_x, uint32_t scale_y, std::vector<uint8_t>& pixels_out);
  bool ReadBackResolve(const draw_util::ResolveInfo& copy, uint32_t address, uint32_t pitch,
                       const QueuedResolve& resolve, uint32_t scale_x, uint32_t scale_y);
  // Copies memexported ranges from the shared vertex buffer back to guest
  // memory, waiting for the GPU.
  bool ReadBackMemexport(std::span<const draw_util::MemExportRange> ranges);
  bool RenderGuestOutput(ui::Presenter::GuestOutputRefreshContext& context, VkExtent2D extent,
                         uint32_t frontbuffer_address);

  struct FrameTiming {
    uint64_t swap;
    uint32_t address;
    VkExtent2D extent;
    int64_t ready_ns;
  };
  // Two submissions in flight: the CPU records one while the GPU runs the other.
  static constexpr uint32_t kMaxFramesInFlight = 2;
  struct FrameSlot {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // Signaled by the sparse binds this slot's submission waits for.
    VkSemaphore sparse_bind_semaphore = VK_NULL_HANDLE;
    bool submitted = false;
    uint64_t submission = 0;
    std::vector<FrameCommand> commands;
    std::vector<DrawPass> passes;
    std::optional<FrameTiming> frame;
  };
  // Waits for the slot's submission and releases what it kept alive.
  bool RetireFrame(FrameSlot& frame);
  // Waits for every submission, oldest first.
  bool RetireAllSubmissions();
  // RetireAllSubmissions, falling back to waiting for the queue to go idle, and
  // dropping everything the frames kept alive either way.
  void WaitForAllSubmissions();
  // Adds a pass for the next pending draw, or returns SIZE_MAX if it fails to initialize.
  size_t AddPendingPass(const DrawInput& input, const IndexBinding& index_binding,
                        std::span<const uint32_t> inline_indices,
                        const Shader::Translation& vertex, const Shader::Translation* pixel,
                        VkBuffer shared_vertex_buffer = VK_NULL_HANDLE,
                        const PreuploadedConstants* preuploaded_constants = nullptr);
  void SavePipelineCache();

  ui::vulkan::VulkanDevice* vulkan_device_ = nullptr;
  VmaAllocator vma_allocator_ = VK_NULL_HANDLE;
  PFN_vkGetPipelineCacheData get_pipeline_cache_data_ = nullptr;
  PFN_vkMergePipelineCaches merge_pipeline_caches_ = nullptr;
  std::filesystem::path pipeline_cache_path_;
  std::timed_mutex pipeline_cache_mutex_;
  size_t saved_pipeline_count_ = 0;
  uint64_t last_pipeline_created_swap_ = 0;
  PFN_vkCmdResolveImage cmd_resolve_image_ = nullptr;
  PFN_vkCmdBlitImage cmd_blit_image_ = nullptr;
  bool r8_resolve_blit_ = false;
  bool rg8_resolve_blit_ = false;
  uint32_t draw_resolution_scale_x_ = 1;
  uint32_t draw_resolution_scale_y_ = 1;
  std::array<FrameSlot, kMaxFramesInFlight> frames_;
  // The slot being recorded, and the one submitted last.
  uint32_t current_frame_ = 0;
  uint32_t last_submitted_frame_ = 0;
  // Command buffer of the slot being recorded.
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  std::multimap<uint64_t, std::unique_ptr<SpirvShader>> shaders_;
  std::unordered_map<uint64_t, SpirvShader*> shader_addr_cache_;
  std::vector<uint64_t> shader_pages_;
  std::unique_ptr<SpirvShaderTranslator> translator_;
  DrawPipelines pipelines_;
  DrawBindingCache draw_bindings_;
  std::unique_ptr<ui::vulkan::VulkanUploadBufferPool> draw_uploads_;
  uint64_t next_resolve_version_ = 1;
  SubmissionClock clock_;
  std::unique_ptr<SharedVertexMemory> shared_vertex_;
  std::unique_ptr<NativeIndexCache> index_cache_;
  std::unique_ptr<NativeTextureCache> texture_cache_;
  std::unique_ptr<NativeTargetCache> target_cache_;
  static constexpr uint32_t kMaxBuiltinQuadCount = 65536;
  VkBuffer builtin_quad_indices_ = VK_NULL_HANDLE;
  VkDeviceMemory builtin_quad_indices_memory_ = VK_NULL_HANDLE;
  VmaAllocation builtin_quad_indices_allocation_ = VK_NULL_HANDLE;
  bool builtin_quad_indices_uploaded_ = false;
  static constexpr uint32_t kAllConstantBuffersMask =
      (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferCount) - 1;
  uint32_t dirty_constants_ = kAllConstantBuffersMask;
  VkBuffer last_constant_buffer_ = VK_NULL_HANDLE;
  std::array<uint32_t, SpirvShaderTranslator::kConstantBufferCount> last_constant_offsets_{};
  std::array<uint64_t, 4> last_vs_float_bitmap_{};
  std::array<uint64_t, 4> last_ps_float_bitmap_{};
  SpirvShaderTranslator::SystemConstants last_system_constants_{};
  bool has_last_system_constants_ = false;
  void InvalidateConstantState() {
    dirty_constants_ = kAllConstantBuffersMask;
    last_constant_buffer_ = VK_NULL_HANDLE;
    has_last_system_constants_ = false;
  }
  bool PreparePreuploadedConstants(const DrawInput& input, const SpirvShader& vs,
                                   const SpirvShader* ps, bool use_shared_vertices,
                                   PreuploadedConstants& out);
  VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
  // Written from guest threads while the global critical region is held, so it
  // only guards the pending list - never call into memory_ or Vulkan here.
  std::mutex invalidation_mutex_;
  std::vector<PhysicalRange> pending_invalidations_;
  std::atomic<bool> has_pending_invalidations_{false};
  bool invalidated_since_last_draw_ = false;
  void* memory_invalidation_handle_ = nullptr;
  std::vector<FrameCommand> pending_;
  std::vector<DrawPass> pending_passes_;
  // Reused by IssueDraw so decoding guest indices and building DrawInput do not allocate per draw.
  std::vector<uint32_t> scratch_indices_;
  DrawInput scratch_input_;
  std::shared_ptr<ColorImage> presented_;
  std::unique_ptr<OutputGamma> output_gamma_;
  std::unique_ptr<DepthResolve> depth_resolve_;
  std::unique_ptr<ColorResolve> color_resolve_;
  std::ofstream frame_timing_;
  void LogFrameTiming(const FrameTiming& frame, const char* result);
  bool reported_frame_ = false;
};

}  // namespace rex::graphics::native
