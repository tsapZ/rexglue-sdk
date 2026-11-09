/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>

#include <rex/assert.h>
#include <rex/graphics/d3d12/shared_memory.h>
#include <rex/graphics/d3d12/texture_cache.h>
#include <rex/graphics/edram_layout.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/render_target/cache.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>
#include <rex/ui/d3d12/d3d12_cpu_descriptor_pool.h>
#include <rex/ui/d3d12/d3d12_provider.h>
#include <rex/ui/d3d12/d3d12_upload_buffer_pool.h>
#include <rex/ui/d3d12/d3d12_util.h>
namespace rex::graphics::d3d12 {

class D3D12CommandProcessor;

class D3D12RenderTargetCache final : public RenderTargetCache {
 public:
  D3D12RenderTargetCache(const RegisterFile& register_file, const memory::Memory& memory,
                         uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
                         D3D12CommandProcessor& command_processor, bool bindless_resources_used)
      : RenderTargetCache(register_file, memory, draw_resolution_scale_x, draw_resolution_scale_y,
                          FitEdramLayoutToResolutionScale(RequestedEdramLayout(),
                                                          draw_resolution_scale_x,
                                                          draw_resolution_scale_y)),
        command_processor_(command_processor),
        bindless_resources_used_(bindless_resources_used) {}
  ~D3D12RenderTargetCache() override;

  bool Initialize();
  void Shutdown(bool from_destructor = false);

  void CompletedSubmissionUpdated();
  void BeginSubmission();

  Path GetPath() const override { return path_; }

  bool Update(bool is_rasterization_done, reg::RB_DEPTHCONTROL normalized_depth_control,
              uint32_t normalized_color_mask, const Shader& vertex_shader) override;

  void InvalidateCommandListRenderTargets() {
    are_current_command_list_render_targets_valid_ = false;
  }

  bool msaa_2x_supported() const { return msaa_2x_supported_; }

  void WriteEdramRawSRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle);
  void WriteEdramRawUAVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle);
  void WriteEdramUintPow2SRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                       uint32_t element_size_bytes_pow2);
  void WriteEdramUintPow2UAVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                       uint32_t element_size_bytes_pow2);

  // Performs the resolve to a shared memory area according to the current
  // register values, and also clears the render targets if needed. Must be in a
  // frame for calling.
  // written_scaled_out: whether the data went to the scaled resolve memory rather than shared
  // memory (a resolve at 1x under draw_resolution_scale_threshold doesn't).
  bool Resolve(const memory::Memory& memory, D3D12SharedMemory& shared_memory,
               D3D12TextureCache& texture_cache, uint32_t& written_address_out,
               uint32_t& written_length_out, bool* written_scaled_out = nullptr);

  // For host render targets.

  bool gamma_render_target_as_unorm16() const { return gamma_render_target_as_unorm16_; }

  // Using R16G16[B16A16]_SNORM, which are -1...1, not the needed -32...32.
  // Persistent data doesn't depend on this, so can be overriden by per-game
  // configuration.
  bool IsFixed16TruncatedToMinus1To1() const {
    return GetPath() == Path::kHostRenderTargets && !REXCVAR_GET(snorm16_render_target_full_range);
  }

  bool depth_float24_round() const { return depth_float24_round_; }
  bool depth_float24_convert_in_pixel_shader() const {
    return depth_float24_convert_in_pixel_shader_;
  }

  DXGI_FORMAT GetColorResourceDXGIFormat(xenos::ColorRenderTargetFormat format) const;
  DXGI_FORMAT GetColorDrawDXGIFormat(xenos::ColorRenderTargetFormat format) const;
  DXGI_FORMAT GetColorOwnershipTransferDXGIFormat(xenos::ColorRenderTargetFormat format,
                                                  bool* is_integer_out = nullptr) const;
  static DXGI_FORMAT GetDepthResourceDXGIFormat(xenos::DepthRenderTargetFormat format);
  static DXGI_FORMAT GetDepthDSVDXGIFormat(xenos::DepthRenderTargetFormat format);
  static DXGI_FORMAT GetDepthSRVDepthDXGIFormat(xenos::DepthRenderTargetFormat format);
  static DXGI_FORMAT GetDepthSRVStencilDXGIFormat(xenos::DepthRenderTargetFormat format);

 protected:
  uint32_t GetMaxRenderTargetWidth() const override { return D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION; }
  uint32_t GetMaxRenderTargetHeight() const override {
    return D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
  }

  RenderTarget* CreateRenderTarget(RenderTargetKey key) override;

  bool IsHostDepthEncodingDifferent(xenos::DepthRenderTargetFormat format) const override;

  bool IsGammaFormatHostStorageSeparate() const override;

  void RequestPixelShaderInterlockBarrier() override;

 private:
  enum class EdramBufferModificationStatus {
    // The values are ordered by how strong the barrier conditions are.
    // No uncommitted ROV/UAV writes.
    kUnmodified,
    // Need to commit before the next ROV usage with overlap.
    kAsROV,
    // Need to commit before any next ROV usage.
    kAsUAV,
  };
  void TransitionEdramBuffer(D3D12_RESOURCE_STATES new_state);
  void MarkEdramBufferModified(
      EdramBufferModificationStatus modification_status = EdramBufferModificationStatus::kAsUAV);
  void CommitEdramBufferUAVWrites(
      EdramBufferModificationStatus commit_status = EdramBufferModificationStatus::kAsROV);

  D3D12CommandProcessor& command_processor_;
  bool bindless_resources_used_;

  Path path_ = Path::kHostRenderTargets;

  // For host render targets, an EDRAM-sized scratch buffer for:
  // - Guest render target data copied from host render targets during copying
  //   in resolves.
  // - Host float32 depth in ownership transfers when the host depth texture and
  //   the destination are the same.
  // For rasterizer-ordered view, the buffer containing the EDRAM data.
  // (Note that if a hybrid RTV / DSV + ROV approach to color render targets is
  //  added, which is, however, unlikely as it would have very complicated
  //  interaction with depth / stencil testing, host depth will need to be
  //  copied to a different buffer - the same range may have ROV-owned color and
  //  host float32 depth at the same time).
  ID3D12Resource* edram_buffer_ = nullptr;
  D3D12_RESOURCE_STATES edram_buffer_state_;
  EdramBufferModificationStatus edram_buffer_modification_status_ =
      EdramBufferModificationStatus::kUnmodified;

  // Non-shader-visible descriptor heap containing pre-created SRV and UAV
  // descriptors of the EDRAM buffer, for faster binding (by copying rather
  // than creation).
  enum class EdramBufferDescriptorIndex : uint32_t {
    kRawSRV,
    kR32UintSRV,
    kR32G32UintSRV,
    kR32G32B32A32UintSRV,
    kRawUAV,
    kR32UintUAV,
    kR32G32UintUAV,
    kR32G32B32A32UintUAV,

    kCount,
  };
  ID3D12DescriptorHeap* edram_buffer_descriptor_heap_ = nullptr;
  D3D12_CPU_DESCRIPTOR_HANDLE edram_buffer_descriptor_heap_start_;

  // Resolve copying root signature and pipelines.
  // Parameter 0 - draw_util::ResolveCopyShaderConstants or its ::DestRelative.
  // Parameter 1 - destination (shared memory or a part of it).
  // Parameter 2 - source (EDRAM).
  ID3D12RootSignature* resolve_copy_root_signature_ = nullptr;
  struct ResolveCopyShaderCode {
    const void* unscaled;
    size_t unscaled_size;
    const void* scaled;
    size_t scaled_size;
  };
  static const ResolveCopyShaderCode
      kResolveCopyShaders[size_t(draw_util::ResolveCopyShaderIndex::kCount)];
  static const ResolveCopyShaderCode
      kResolveCopyShaders4096[size_t(draw_util::ResolveCopyShaderIndex::kCount)];
  ID3D12PipelineState* resolve_copy_pipelines_[size_t(draw_util::ResolveCopyShaderIndex::kCount)] =
      {};
  // Unscaled variants for resolves whose sources are all 1x (draw_resolution_scale_threshold),
  // made only with resolution scaling (else the main set is unscaled already). Their own root
  // signature: the scaled shaders' constants and destination assume the scaled resolve memory.
  ID3D12RootSignature* resolve_copy_native_root_signature_ = nullptr;
  ID3D12PipelineState*
      resolve_copy_native_pipelines_[size_t(draw_util::ResolveCopyShaderIndex::kCount)] = {};

  // For host render targets.

  class D3D12RenderTarget final : public RenderTarget {
   public:
    // descriptor_load_separate is present when the DXGI formats are different
    // for drawing and bit-exact loading (for NaN pattern preservation across
    // EDRAM tile ownership transfers in floating-point formats, and to
    // distinguish between two -1 representations in snorm formats).
    D3D12RenderTarget(RenderTargetKey key, ID3D12Resource* resource,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_draw,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_load_separate,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_srv,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_srv_stencil,
                      D3D12_RESOURCE_STATES resource_state)
        : RenderTarget(key),
          resource_(resource),
          descriptor_draw_(std::move(descriptor_draw)),
          descriptor_load_separate_(std::move(descriptor_load_separate)),
          descriptor_srv_(std::move(descriptor_srv)),
          descriptor_srv_stencil_(std::move(descriptor_srv_stencil)),
          resource_state_(resource_state) {}

    ID3D12Resource* resource() const { return resource_.Get(); }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_draw() const {
      return descriptor_draw_;
    }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_srv() const {
      return descriptor_srv_;
    }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_srv_stencil() const {
      return descriptor_srv_stencil_;
    }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_load_separate() const {
      return descriptor_load_separate_;
    }

    D3D12_RESOURCE_STATES SetResourceState(D3D12_RESOURCE_STATES new_state) {
      D3D12_RESOURCE_STATES old_state = resource_state_;
      resource_state_ = new_state;
      return old_state;
    }

    uint32_t temporary_srv_descriptor_index() const { return temporary_srv_descriptor_index_; }
    void SetTemporarySRVDescriptorIndex(uint32_t index) { temporary_srv_descriptor_index_ = index; }
    uint32_t temporary_srv_descriptor_index_stencil() const {
      return temporary_srv_descriptor_index_stencil_;
    }
    void SetTemporarySRVDescriptorIndexStencil(uint32_t index) {
      temporary_srv_descriptor_index_stencil_ = index;
    }
    uint32_t temporary_sort_index() const { return temporary_sort_index_; }
    void SetTemporarySortIndex(uint32_t index) { temporary_sort_index_ = index; }

   private:
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_draw_;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_load_separate_;
    // Texture SRV non-shader-visible descriptors, to prepare shader-visible
    // descriptors faster, by copying rather than by creating every time.
    // TODO(Triang3l): With bindless resources, persistently store them in the
    // heap.
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv_;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv_stencil_;
    D3D12_RESOURCE_STATES resource_state_;
    // Temporary storage for indices in operations like transfers and dumps.
    uint32_t temporary_srv_descriptor_index_ = UINT32_MAX;
    uint32_t temporary_srv_descriptor_index_stencil_ = UINT32_MAX;
    uint32_t temporary_sort_index_ = 0;
  };

  enum TransferCBVRegister : uint32_t {
    kTransferCBVRegisterStencilMask,
    kTransferCBVRegisterAddress,
    kTransferCBVRegisterHostDepthAddress,
  };
  enum TransferSRVRegister : uint32_t {
    kTransferSRVRegisterColor,
    kTransferSRVRegisterDepth,
    kTransferSRVRegisterStencil,
    kTransferSRVRegisterHostDepth,
    kTransferSRVRegisterCount,
  };
  enum TransferUsedRootParameter : uint32_t {
    // Changed 8 times per transfer.
    kTransferUsedRootParameterStencilMaskConstant,
    kTransferUsedRootParameterColorSRV,
    // Mutually exclusive with ColorSRV.
    kTransferUsedRootParameterDepthSRV,
    // Mutually exclusive with ColorSRV.
    kTransferUsedRootParameterStencilSRV,
    // May happen to be the same for different sources.
    kTransferUsedRootParameterAddressConstant,
    kTransferUsedRootParameterHostDepthSRV,
    kTransferUsedRootParameterHostDepthAddressConstant,
    kTransferUsedRootParameterCount,

    kTransferUsedRootParameterStencilMaskConstantBit =
        uint32_t(1) << kTransferUsedRootParameterStencilMaskConstant,
    kTransferUsedRootParameterColorSRVBit = uint32_t(1) << kTransferUsedRootParameterColorSRV,
    kTransferUsedRootParameterDepthSRVBit = uint32_t(1) << kTransferUsedRootParameterDepthSRV,
    kTransferUsedRootParameterStencilSRVBit = uint32_t(1) << kTransferUsedRootParameterStencilSRV,
    kTransferUsedRootParameterAddressConstantBit = uint32_t(1)
                                                   << kTransferUsedRootParameterAddressConstant,
    kTransferUsedRootParameterHostDepthSRVBit = uint32_t(1)
                                                << kTransferUsedRootParameterHostDepthSRV,
    kTransferUsedRootParameterHostDepthAddressConstantBit =
        uint32_t(1) << kTransferUsedRootParameterHostDepthAddressConstant,

    kTransferUsedRootParametersDescriptorMask =
        kTransferUsedRootParameterColorSRVBit | kTransferUsedRootParameterDepthSRVBit |
        kTransferUsedRootParameterStencilSRVBit | kTransferUsedRootParameterHostDepthSRVBit,
  };
  enum class TransferRootSignatureIndex {
    kColor,
    kDepth,
    kDepthStencil,
    kColorToStencilBit,
    kStencilToStencilBit,
    kColorAndHostDepth,
    kDepthAndHostDepth,
    kDepthStencilAndHostDepth,
    kCount,
  };
  static const uint32_t kTransferUsedRootParameters[size_t(TransferRootSignatureIndex::kCount)];
  enum class TransferMode : uint32_t {
    // 1 SRV (color texture), source constant.
    kColorToDepth,
    // 1 SRV (color texture), source constant.
    kColorToColor,

    // 1 or 2 SRVs (depth texture, stencil texture if SV_StencilRef is
    // supported), source constant.
    kDepthToDepth,
    // 2 SRVs (depth texture, stencil texture), source constant.
    kDepthToColor,

    // 1 SRV (color texture), mask constant (most frequently changed, 8 times
    // per transfer), source constant.
    kColorToStencilBit,
    // 1 SRV (stencil texture), mask constant, source constant.
    kDepthToStencilBit,

    // Two-source modes, using the host depth if it, when converted to the guest
    // format, matches what's in the owner source (not modified, keep host
    // precision), or the guest data otherwise (significantly modified, possibly
    // cleared). Stencil for SV_StencilRef is always taken from the guest
    // source.

    // 2 SRVs (color texture, host depth texture or buffer), source constant,
    // host depth source constant.
    kColorAndHostDepthToDepth,
    // When using different source and destination depth formats. 2 or 3 SRVs
    // (depth texture, stencil texture if SV_StencilRef is supported, host depth
    // texture or buffer), source constant, host depth source constant.
    kDepthAndHostDepthToDepth,

    kCount,
  };
  enum class TransferOutput {
    kColor,
    kDepth,
    // With this output, kTransferCBVRegisterStencilMask is used.
    kStencilBit,
  };
  struct TransferModeInfo {
    TransferOutput output;
    TransferRootSignatureIndex root_signature_no_stencil_ref;
    TransferRootSignatureIndex root_signature_with_stencil_ref;
  };
  static const TransferModeInfo kTransferModes[size_t(TransferMode::kCount)];

  union TransferShaderKey {
    uint32_t key;
    struct {
      xenos::MsaaSamples dest_msaa_samples : xenos::kMsaaSamplesBits;
      uint32_t dest_resource_format : xenos::kRenderTargetFormatBits;
      xenos::MsaaSamples source_msaa_samples : xenos::kMsaaSamplesBits;
      // Always 1x when host_depth_source_is_copy is true not to create the same
      // pipeline for different MSAA sample counts as it doesn't matter in this
      // case.
      xenos::MsaaSamples host_depth_source_msaa_samples : xenos::kMsaaSamplesBits;
      uint32_t source_resource_format : xenos::kRenderTargetFormatBits;
      // If host depth is also fetched, whether it's pre-copied to the EDRAM
      // buffer (but since it's just a scratch buffer, with tiles laid out
      // linearly with the same pitch as in the original render target; also no
      // swapping of 40-sample columns as opposed to the host render target -
      // this is done only for the color source).
      uint32_t host_depth_source_is_copy : 1;
      // The scale classes of the two sides (draw_resolution_scale_threshold): the shader bakes
      // each side's tile size and the conversion between the scale spaces.
      uint32_t dest_scale_native : 1;
      uint32_t source_scale_native : 1;

      // Last bits because this affects the root signature - after sorting, only
      // change it as fewer times as possible. Depth buffers have an additional
      // stencil SRV.
      static_assert(size_t(TransferMode::kCount) <= (size_t(1) << 3));
      TransferMode mode : 3;
    };

    TransferShaderKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const TransferShaderKey& key) const {
        return std::hash<uint32_t>{}(key.key);
      }
    };
    bool operator==(const TransferShaderKey& other_key) const { return key == other_key.key; }
    bool operator!=(const TransferShaderKey& other_key) const { return !(*this == other_key); }
    bool operator<(const TransferShaderKey& other_key) const { return key < other_key.key; }
  };

  union TransferAddressConstant {
    uint32_t constant;
    struct {
      // All in tiles.
      uint32_t dest_pitch : xenos::kEdramPitchTilesBits;
      uint32_t source_pitch : xenos::kEdramPitchTilesBits;
      // Destination base in tiles minus source base in tiles (not vice versa
      // because this is a transform of the coordinate system, not addresses
      // themselves).
      // + 1 bit because this is a signed difference between two EDRAM bases.
      // 0 for host_depth_source_is_copy (ignored in this case anyway as
      // destination == source anyway).
      // With a 4096-tile EDRAM the difference is stored modulo 4096: the shader
      // wraps the sum at the EDRAM tile count, so the result is the same.
      int32_t source_to_dest : xenos::kEdramBaseTilesBitsMax;
    };
    TransferAddressConstant() : constant(0) { static_assert_size(*this, sizeof(constant)); }
    bool operator==(const TransferAddressConstant& other_constant) const {
      return constant == other_constant.constant;
    }
    bool operator!=(const TransferAddressConstant& other_constant) const {
      return !(*this == other_constant);
    }
  };

  struct TransferInvocation {
    Transfer transfer;
    TransferShaderKey shader_key;
    TransferInvocation(const Transfer& transfer, const TransferShaderKey& shader_key)
        : transfer(transfer), shader_key(shader_key) {}
    bool operator<(const TransferInvocation& other_invocation) const {
      // TODO(Triang3l): See if it may be better to sort by the source in the
      // first place, especially when reading the same data multiple times (like
      // to write the stencil bits after depth) for better read locality.
      // Sort by the shader key primarily to reduce pipeline state (context)
      // switches.
      if (shader_key != other_invocation.shader_key) {
        return shader_key < other_invocation.shader_key;
      }
      // Host depth render targets are changed rarely if they exist, won't save
      // many binding changes, ignore them for simplicity (their existence is
      // caught by the shader key change).
      assert_not_null(transfer.source);
      assert_not_null(other_invocation.transfer.source);
      uint32_t source_index =
          static_cast<const D3D12RenderTarget*>(transfer.source)->temporary_sort_index();
      uint32_t other_source_index =
          static_cast<const D3D12RenderTarget*>(other_invocation.transfer.source)
              ->temporary_sort_index();
      if (source_index != other_source_index) {
        return source_index < other_source_index;
      }
      return transfer.start_tiles < other_invocation.transfer.start_tiles;
    }
    bool CanBeMergedIntoOneDraw(const TransferInvocation& other_invocation) const {
      return shader_key == other_invocation.shader_key &&
             transfer.AreSourcesSame(other_invocation.transfer);
    }
  };

  enum {
    kHostDepthStoreRootParameterConstants,
    kHostDepthStoreRootParameterSource,
    kHostDepthStoreRootParameterDest,
    kHostDepthStoreRootParameterCount,
  };

  union DumpPipelineKey {
    uint32_t key;
    struct {
      xenos::MsaaSamples msaa_samples : 2;
      uint32_t resource_format : 4;
      // Last bit because this affects the root signature - after sorting, only
      // change it at most once. Depth buffers have an additional stencil SRV.
      uint32_t is_depth : 1;
      // A 1x render target (draw_resolution_scale_threshold): dumping it to the scaled EDRAM
      // layout duplicates its guest pixels.
      uint32_t source_scale_native : 1;
      // source_scale_native only: address the EDRAM buffer with the plain 1x tile layout (a
      // resolve whose sources are all 1x).
      uint32_t native_layout : 1;
    };

    DumpPipelineKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const DumpPipelineKey& key) const { return std::hash<uint32_t>{}(key.key); }
    };
    bool operator==(const DumpPipelineKey& other_key) const { return key == other_key.key; }
    bool operator!=(const DumpPipelineKey& other_key) const { return !(*this == other_key); }
    bool operator<(const DumpPipelineKey& other_key) const { return key < other_key.key; }

    xenos::ColorRenderTargetFormat GetColorFormat() const {
      assert_false(is_depth);
      return xenos::ColorRenderTargetFormat(resource_format);
    }
    xenos::DepthRenderTargetFormat GetDepthFormat() const {
      assert_true(is_depth);
      return xenos::DepthRenderTargetFormat(resource_format);
    }
  };

  union DumpOffsets {
    uint32_t offsets;
    struct {
      // May be beyond the EDRAM tile count in case of EDRAM addressing
      // wrapping, thus + 1 bit.
      uint32_t dispatch_first_tile : xenos::kEdramBaseTilesBitsMax + 1;
      uint32_t source_base_tiles : xenos::kEdramBaseTilesBitsMax;
    };
    DumpOffsets() : offsets(0) { static_assert_size(*this, sizeof(offsets)); }
    bool operator==(const DumpOffsets& other_offsets) const {
      return offsets == other_offsets.offsets;
    }
    bool operator!=(const DumpOffsets& other_offsets) const { return !(*this == other_offsets); }
  };

  union DumpPitches {
    uint32_t pitches;
    struct {
      // Both in tiles.
      uint32_t dest_pitch : xenos::kEdramPitchTilesBits;
      uint32_t source_pitch : xenos::kEdramPitchTilesBits;
    };
    DumpPitches() : pitches(0) { static_assert_size(*this, sizeof(pitches)); }
    bool operator==(const DumpPitches& other_pitches) const {
      return pitches == other_pitches.pitches;
    }
    bool operator!=(const DumpPitches& other_pitches) const { return !(*this == other_pitches); }
  };

  enum DumpCbuffer : uint32_t {
    kDumpCbufferOffsets,
    kDumpCbufferPitches,
    kDumpCbufferCount,
  };

  enum DumpRootParameter : uint32_t {
    // May be changed multiple times for the same source.
    kDumpRootParameterOffsets,
    // One resolve may need multiple sources.
    kDumpRootParameterSource,

    // May be different for different sources.
    kDumpRootParameterColorPitches = kDumpRootParameterSource + 1,
    // Only changed between 32bpp and 64bpp.
    kDumpRootParameterColorEdram,

    kDumpRootParameterColorCount,

    // Same change frequency than the source (though currently the command
    // processor can't contiguously allocate multiple descriptors with bindless,
    // when such functionality is added, switch to one root signature).
    kDumpRootParameterDepthStencil = kDumpRootParameterSource + 1,
    kDumpRootParameterDepthPitches,
    kDumpRootParameterDepthEdram,

    kDumpRootParameterDepthCount,
  };

  struct DumpInvocation {
    ResolveCopyDumpRectangle rectangle;
    DumpPipelineKey pipeline_key;
    DumpInvocation(const ResolveCopyDumpRectangle& rectangle, const DumpPipelineKey& pipeline_key)
        : rectangle(rectangle), pipeline_key(pipeline_key) {}
    bool operator<(const DumpInvocation& other_invocation) const {
      // Sort by the pipeline key primarily to reduce pipeline state (context)
      // switches.
      if (pipeline_key != other_invocation.pipeline_key) {
        return pipeline_key < other_invocation.pipeline_key;
      }
      assert_not_null(rectangle.render_target);
      uint32_t render_target_index =
          static_cast<const D3D12RenderTarget*>(rectangle.render_target)->temporary_sort_index();
      const ResolveCopyDumpRectangle& other_rectangle = other_invocation.rectangle;
      uint32_t other_render_target_index =
          static_cast<const D3D12RenderTarget*>(other_rectangle.render_target)
              ->temporary_sort_index();
      if (render_target_index != other_render_target_index) {
        return render_target_index < other_render_target_index;
      }
      if (rectangle.row_first != other_rectangle.row_first) {
        return rectangle.row_first < other_rectangle.row_first;
      }
      return rectangle.row_first_start < other_rectangle.row_first_start;
    }
  };

  struct DirectResolvePushConstants {
    draw_util::ResolveCopyShaderConstants resolve;
    uint32_t source_base_tiles;
    uint32_t source_pitch_tiles;
    uint32_t dispatch_first_tile;
  };

  struct DirectResolvePipelineKey {
    DumpPipelineKey dump_pipeline_key;
    draw_util::ResolveCopyShaderIndex copy_shader;
    bool draw_resolution_scaled;

    uint64_t packed() const {
      return uint64_t(dump_pipeline_key.key) | (uint64_t(size_t(copy_shader)) << 32) |
             (uint64_t(draw_resolution_scaled ? 1 : 0) << 40);
    }
    struct Hasher {
      size_t operator()(const DirectResolvePipelineKey& key) const {
        return std::hash<uint64_t>{}(key.packed());
      }
    };
    bool operator==(const DirectResolvePipelineKey& other_key) const {
      return packed() == other_key.packed();
    }
  };

  // Returns:
  // - A pointer to 1 pipeline for writing color or depth (or stencil via
  //   SV_StencilRef).
  // - A pointer to 8 pipelines for writing stencil by discarding samples
  //   depending on whether they have one bit set, from 1 << 0 to 1 << 7.
  // - Null if failed to create.
  ID3D12PipelineState* const* GetOrCreateTransferPipelines(TransferShaderKey key);

  static TransferMode GetTransferMode(bool dest_is_stencil_bit, bool dest_is_depth,
                                      bool source_is_depth, bool source_has_host_depth) {
    assert_true(dest_is_depth || (!dest_is_stencil_bit && !source_has_host_depth));
    if (dest_is_stencil_bit) {
      return source_is_depth ? TransferMode::kDepthToStencilBit : TransferMode::kColorToStencilBit;
    }
    if (dest_is_depth) {
      if (source_is_depth) {
        return source_has_host_depth ? TransferMode::kDepthAndHostDepthToDepth
                                     : TransferMode::kDepthToDepth;
      }
      return source_has_host_depth ? TransferMode::kColorAndHostDepthToDepth
                                   : TransferMode::kColorToDepth;
    }
    return source_is_depth ? TransferMode::kDepthToColor : TransferMode::kColorToColor;
  }

  // Do ownership transfers for render targets - each render target / vector may
  // be null / empty in case there's nothing to do for them.
  // resolve_clear_rectangle is expected to be provided by
  // PrepareHostRenderTargetsResolveClear which should do all the needed size
  // bound checks.
  void PerformTransfersAndResolveClears(
      uint32_t render_target_count, RenderTarget* const* render_targets,
      const std::vector<Transfer>* render_target_transfers,
      const uint64_t* render_target_resolve_clear_values = nullptr,
      const Transfer::Rectangle* resolve_clear_rectangle = nullptr);

  // Accepts an array of (1 + xenos::kMaxColorRenderTargets) render targets,
  // first depth, then color.
  void SetCommandListRenderTargets(RenderTarget* const* depth_and_color_render_targets);

  ID3D12PipelineState* GetOrCreateDumpPipeline(DumpPipelineKey key);
  ID3D12PipelineState* GetOrCreateDirectResolvePipeline(DirectResolvePipelineKey key);
  // Direct resolve: the render target to copy from instead of dumping to the EDRAM buffer, when
  // one render target owns the resolve's whole source span and the copy is one the direct shaders
  // do (fast 32bpp, 8_8_8_8 or 2_10_10_10 at 1x or 2x MSAA); else nullptr. rt_tiles_out and
  // rt_flags_out are the shader's xe_resolve_direct_rt_tiles / _flags.
  D3D12RenderTarget* GetDirectResolveSource(const draw_util::ResolveInfo& resolve_info,
                                            draw_util::ResolveCopyShaderIndex copy_shader,
                                            uint32_t& rt_tiles_out, uint32_t& rt_flags_out);

  // Writes contents of host render targets within rectangles from
  // ResolveInfo::GetCopyEdramTileSpan to edram_buffer_.
  // native_layout: the plain 1x tile layout (a resolve whose sources are all 1x).
  bool DumpRenderTargets(uint32_t dump_base, uint32_t dump_row_length_used, uint32_t dump_rows,
                         uint32_t dump_pitch, bool native_layout);

  // gpu_native_resolves: why a resolve or a texture was not written natively.
  enum class NativeResolveFallback : uint32_t {
    kDepth,
    kMsaa,
    kExpBias,
    kSwap,
    kNoSource,
    kMultipleSources,
    kPartialSource,
    kSourceFormat,
    kDestFormat,
    kDestArray,
    kDestBaseAdjusted,
    kEndian,
    kHalfPixelFill,
    kSourcePlacement,
    kTextureIncompatible,
    kTextureNotUpToDate,
    kTextureReplacement,
    kTextureOutside,
    kTextureNoUAV,
    kNoPipeline,
    kDescriptors,
    kCount,
  };
  static const char* const kNativeResolveFallbackNames[size_t(NativeResolveFallback::kCount)];
  // gpu_native_resolves: the source render target and destination texture formats written.
  enum class NativeResolvePair : uint32_t {
    // Color k_16_16_16_16_FLOAT to k_16_16_16_16_FLOAT, copied or scaled by the exponent bias.
    k16161616Float,
    // Color k_16_16_FLOAT to k_16_16_FLOAT, copied.
    k1616Float,
    // Color k_2_10_10_10_FLOAT to k_16_16_16_16_FLOAT, scaled by the exponent bias.
    k2101010FloatTo16161616Float,
    // Color k_8_8_8_8 to k_8_8_8_8, copied or with red and blue swapped.
    k8888,
    // Depth kD24S8 to k_24_8.
    kD24S8,
    // Depth kD24FS8 to k_24_8_FLOAT.
    kD24FS8,
    kCount,
  };
  static const char* const kNativeResolvePairNames[size_t(NativeResolvePair::kCount)];
  // Whether a host render target resolve copy may be written to the textures natively, counting
  // the reason if not.
  bool IsNativeResolveEligible(const draw_util::ResolveInfo& resolve_info);
  // After DumpRenderTargets and MarkRangeAsResolved for the resolve, copies the resolved region
  // from the source host render target into the texture cache textures viewing the destination
  // that were up to date before the resolve, and marks them up to date.
  void WriteResolveToTextures(const draw_util::ResolveInfo& resolve_info,
                              D3D12TextureCache& texture_cache);
  void CountNativeResolveFallback(NativeResolveFallback reason, uint64_t count = 1) {
    native_resolve_fallbacks_[size_t(reason)] += count;
  }
  // Logs the native resolve counts every few seconds.
  void ReportNativeResolves();

  bool use_stencil_reference_output_ = false;

  bool gamma_render_target_as_unorm16_ = false;

  bool depth_float24_round_ = false;
  bool depth_float24_convert_in_pixel_shader_ = false;

  bool msaa_2x_supported_ = false;

  std::shared_ptr<ui::d3d12::D3D12CpuDescriptorPool> descriptor_pool_color_;
  std::shared_ptr<ui::d3d12::D3D12CpuDescriptorPool> descriptor_pool_depth_;
  std::shared_ptr<ui::d3d12::D3D12CpuDescriptorPool> descriptor_pool_srv_;
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor null_rtv_descriptor_ss_;
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor null_rtv_descriptor_ms_;

  // Possible tile ownership transfer paths:
  // - To color:
  //   - From color: 1 SRV (color).
  //   - From depth: 2 SRVs (depth, stencil).
  // - To depth / stencil (with SV_StencilRef):
  //   - From color: 1 SRV (color).
  //   - From depth: 2 SRVs (depth, stencil).
  //   - From color and float32 depth: 2 SRVs (color with stencil, depth).
  //     - Different depth buffer: depth SRV is a texture.
  //     - Same depth buffer: depth SRV is a buffer (pre-copied).
  // - To depth (no SV_StencilRef):
  //   - From color: 1 SRV (color).
  //   - From depth: 1 SRV (depth).
  //   - From color and float32 depth: 2 SRVs (color, depth).
  //     - Different depth buffer: depth SRV is a texture.
  //     - Same depth buffer: depth SRV is a buffer (pre-copied).
  // - To stencil (no SV_StencilRef):
  //   - From color: 1 SRV (color).
  //   - From depth: 1 SRV (stencil).

  const RenderTarget* const*
      current_command_list_render_targets_[1 + xenos::kMaxColorRenderTargets];
  bool are_current_command_list_render_targets_valid_ = false;

  // Temporary storage for descriptors used in PerformTransfersAndResolveClears
  // and DumpRenderTargets.
  std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> current_temporary_descriptors_cpu_;
  std::vector<ui::d3d12::util::DescriptorCpuGpuHandlePair> current_temporary_descriptors_gpu_;

  ID3D12RootSignature* host_depth_store_root_signature_ = nullptr;
  ID3D12PipelineState* host_depth_store_pipelines_[size_t(xenos::MsaaSamples::k4X) + 1] = {};

  std::unique_ptr<ui::d3d12::D3D12UploadBufferPool> transfer_vertex_buffer_pool_;

  ID3D12RootSignature* transfer_root_signatures_[size_t(TransferRootSignatureIndex::kCount)] = {};
  std::unordered_map<TransferShaderKey, ID3D12PipelineState*, TransferShaderKey::Hasher>
      transfer_pipelines_;
  std::unordered_map<TransferShaderKey, std::array<ID3D12PipelineState*, 8>,
                     TransferShaderKey::Hasher>
      transfer_stencil_bit_pipelines_;

  // Temporary storage for PerformTransfersAndResolveClears.
  std::vector<TransferInvocation> current_transfer_invocations_;

  // Temporary storage for DumpRenderTargets.
  std::vector<ResolveCopyDumpRectangle> dump_rectangles_;
  std::vector<DumpInvocation> dump_invocations_;
  std::vector<ResolveCopyDispatch> direct_resolve_dispatches_;

  ID3D12RootSignature* dump_root_signature_color_ = nullptr;
  ID3D12RootSignature* dump_root_signature_depth_ = nullptr;
  // Compute pipelines for copying host render target contents to the EDRAM
  // buffer. May be null if failed to create.
  std::unordered_map<DumpPipelineKey, ID3D12PipelineState*, DumpPipelineKey::Hasher>
      dump_pipelines_;
  ID3D12RootSignature* direct_resolve_root_signature_color_ = nullptr;
  ID3D12RootSignature* direct_resolve_root_signature_depth_ = nullptr;
  std::unordered_map<DirectResolvePipelineKey, ID3D12PipelineState*,
                     DirectResolvePipelineKey::Hasher>
      direct_resolve_pipelines_;
  // resolve_direct_fast_32bpp_{1x,2x}msaa, at the draw resolution scale and EDRAM size in use.
  ID3D12PipelineState* direct_resolve_fast_32bpp_pipelines_[2] = {};
  uint64_t direct_resolve_attempt_count_ = 0;
  uint64_t direct_resolve_success_count_ = 0;
  uint64_t direct_resolve_fallback_count_ = 0;

  // gpu_native_resolves: textures written and fallbacks since the last report.
  uint64_t native_resolve_written_ = 0;
  // Of them, written although not up to date before (the resolve covered the texture).
  uint64_t native_resolve_covered_ = 0;
  std::array<uint64_t, size_t(NativeResolvePair::kCount)> native_resolve_written_pairs_ = {};
  std::array<uint64_t, size_t(NativeResolveFallback::kCount)> native_resolve_fallbacks_ = {};
  std::chrono::steady_clock::time_point native_resolve_last_report_ =
      std::chrono::steady_clock::now();
  // direct_host_resolve_verify (a debug setting): a direct resolve copied both ways, the destination read
  // back after each, compared once the submission is done (the scene doesn't matter then).
  struct DirectResolveVerify {
    ID3D12Resource* readback[2];  // after the copy from the dump, after the direct copy
    uint32_t length;
    uint64_t submission;
    std::string what;
  };
  std::vector<DirectResolveVerify> direct_resolve_verify_pending_;
  std::vector<uint64_t> direct_resolve_verify_keys_;
  uint64_t direct_resolve_verify_exact_ = 0;
  uint64_t direct_resolve_verify_differ_ = 0;
  // Compares the finished ones (all: releases the rest unchecked).
  void CheckDirectResolveVerifies(bool release_all);

  // Parameter 0 - 2 root constants (red, green).
  ID3D12RootSignature* uint32_rtv_clear_root_signature_ = nullptr;
  // [32 or 32_32][MSAA samples].
  ID3D12PipelineState* uint32_rtv_clear_pipelines_[2][size_t(xenos::MsaaSamples::k4X) + 1] = {};

  std::vector<Transfer> clear_transfers_[2];

  // Temporary storage for DXBC building.
  std::vector<uint32_t> built_shader_;

  // For rasterizer-ordered view (pixel shader interlock).

  ID3D12RootSignature* resolve_rov_clear_root_signature_ = nullptr;
  // Clearing 32bpp color or depth.
  ID3D12PipelineState* resolve_rov_clear_32bpp_pipeline_ = nullptr;
  // Clearing 64bpp color.
  ID3D12PipelineState* resolve_rov_clear_64bpp_pipeline_ = nullptr;
};

}  // namespace rex::graphics::d3d12
