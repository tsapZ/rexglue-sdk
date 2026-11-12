/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/graphics/d3d12/shader.h>
#include <rex/graphics/d3d12/shared_memory.h>
#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/xenos.h>
#include <rex/ui/d3d12/d3d12_api.h>
#include <rex/ui/d3d12/d3d12_provider.h>

namespace rex::graphics::d3d12 {

class D3D12CommandProcessor;

class D3D12TextureCache final : public TextureCache {
 public:
  // Keys that can be stored for checking validity whether descriptors for host
  // shader bindings are up to date.
  struct TextureSRVKey {
    TextureKey key;
    uint32_t host_swizzle;
    uint8_t swizzled_signs;
  };

  // Sampler parameters that can be directly converted to a host sampler or used
  // for binding checking validity whether samplers are up to date.
  union SamplerParameters {
    uint32_t value;
    struct {
      xenos::ClampMode clamp_x : 3;         // 3
      xenos::ClampMode clamp_y : 3;         // 6
      xenos::ClampMode clamp_z : 3;         // 9
      xenos::BorderColor border_color : 2;  // 11
      // For anisotropic, these are true.
      uint32_t mag_linear : 1;              // 12
      uint32_t min_linear : 1;              // 13
      uint32_t mip_linear : 1;              // 14
      xenos::AnisoFilter aniso_filter : 3;  // 17
      uint32_t mip_min_level : 4;           // 21
      uint32_t mip_base_map : 1;            // 22
      // Maximum mip level is in the texture resource itself, but mip_base_map
      // can be used to limit fetching to mip_min_level.
    };

    SamplerParameters() : value(0) { static_assert_size(*this, sizeof(value)); }
    bool operator==(const SamplerParameters& parameters) const { return value == parameters.value; }
    bool operator!=(const SamplerParameters& parameters) const { return value != parameters.value; }
  };

  static std::unique_ptr<D3D12TextureCache> Create(const RegisterFile& register_file,
                                                   D3D12SharedMemory& shared_memory,
                                                   uint32_t draw_resolution_scale_x,
                                                   uint32_t draw_resolution_scale_y,
                                                   D3D12CommandProcessor& command_processor,
                                                   bool bindless_resources_used) {
    std::unique_ptr<D3D12TextureCache> texture_cache(
        new D3D12TextureCache(register_file, shared_memory, draw_resolution_scale_x,
                              draw_resolution_scale_y, command_processor, bindless_resources_used));
    if (!texture_cache->Initialize()) {
      return nullptr;
    }
    return std::move(texture_cache);
  }

  ~D3D12TextureCache();

  void ClearCache() override;

  void BeginSubmission(uint64_t new_submission_index) override;
  void BeginFrame() override;
  void EndFrame();

  // Must be called within a submission - creates and untiles textures needed by
  // shaders and puts them in the SRV state. This may bind compute pipelines
  // (notifying the command processor about that), so this must be called before
  // binding the actual drawing pipeline.
  void RequestTextures(uint32_t used_texture_mask) override;

  // Returns whether texture SRV keys stored externally are still valid for the
  // current bindings and host shader binding layout. Both keys and
  // host_shader_bindings must have host_shader_binding_count elements
  // (otherwise they are incompatible - like if this function returned false).
  bool AreActiveTextureSRVKeysUpToDate(const TextureSRVKey* keys,
                                       const D3D12Shader::TextureBinding* host_shader_bindings,
                                       size_t host_shader_binding_count) const;
  // Exports the current binding data to texture SRV keys so they can be stored
  // for checking whether subsequent draw calls can keep using the same
  // bindings. Write host_shader_binding_count keys.
  void WriteActiveTextureSRVKeys(TextureSRVKey* keys,
                                 const D3D12Shader::TextureBinding* host_shader_bindings,
                                 size_t host_shader_binding_count) const;
  void WriteActiveTextureBindfulSRV(const D3D12Shader::TextureBinding& host_shader_binding,
                                    D3D12_CPU_DESCRIPTOR_HANDLE handle);
  uint32_t GetActiveTextureBindlessSRVIndex(const D3D12Shader::TextureBinding& host_shader_binding);

  SamplerParameters GetSamplerParameters(const D3D12Shader::SamplerBinding& binding) const;
  void WriteSampler(SamplerParameters parameters, D3D12_CPU_DESCRIPTOR_HANDLE handle) const;

  // Returns whether the actual scale is not smaller than the requested one.
  static bool ClampDrawResolutionScaleToMaxSupported(uint32_t& scale_x, uint32_t& scale_y,
                                                     const ui::d3d12::D3D12Provider& provider);
  // Ensures the tiles backing the range in the buffers are allocated.
  bool EnsureScaledResolveMemoryCommitted(uint32_t start_unscaled, uint32_t length_unscaled,
                                          uint32_t length_scaled_alignment_log2 = 0) override;
  // Makes the specified range of up to 1-2 GB currently accessible on the GPU.
  // One draw call can access only at most one range - the same memory is
  // accessible through different buffers based on the range needed, so aliasing
  // barriers are required.
  bool MakeScaledResolveRangeCurrent(uint32_t start_unscaled, uint32_t length_unscaled,
                                     uint32_t length_scaled_alignment_log2 = 0);
  // These functions create a view of the range specified in the last successful
  // MakeScaledResolveRangeCurrent call because that function must be called
  // before this.
  void CreateCurrentScaledResolveRangeUintPow2SRV(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                                  uint32_t element_size_bytes_pow2);
  void CreateCurrentScaledResolveRangeUintPow2UAV(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                                  uint32_t element_size_bytes_pow2);
  void TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATES new_state);
  uint64_t GetCurrentScaledResolveRangeStartScaled() const {
    return scaled_resolve_current_range_start_scaled_;
  }
  uint64_t GetCurrentScaledResolveRangeLengthScaled() const {
    return scaled_resolve_current_range_length_scaled_;
  }
  ID3D12Resource* GetCurrentScaledResolveBufferResource() {
    return GetCurrentScaledResolveBuffer().resource();
  }
  size_t GetCurrentScaledResolveBufferIndexPublic() const {
    return GetCurrentScaledResolveBufferIndex();
  }
  void MarkCurrentScaledResolveRangeUAVWritesCommitNeeded() {
    assert_true(IsDrawResolutionScaled());
    GetCurrentScaledResolveBuffer().SetUAVBarrierPending();
  }

  // gpu_native_resolves: how the source is written into the textures.
  enum class NativeResolveMode : uint32_t {
    // The values of the native_resolve.cs.hlsl modes.
    // out = in * scale.
    kScale,
    // out = in.bgra.
    kSwapRedBlue,
    // out.r = depth.
    kDepthToRed,
    // CopyTextureRegion.
    kCopy,
  };
  // gpu_native_resolves: a resolve rectangle to copy from a host render target straight into the
  // textures viewing the resolve destination. Coordinates and sizes are in guest pixels, unscaled.
  struct NativeResolveWrite {
    // Physical 4 KB page of the texture base the destination coordinates are relative to.
    uint32_t base_page;
    xenos::TextureFormat format;
    // In TextureKey::pitch units (32 texels).
    uint32_t pitch;
    xenos::Endian endianness;
    // Subresource 0 is read. With kCopy, the format must be in the same typeless family as the
    // textures'.
    ID3D12Resource* source;
    uint32_t source_x;
    uint32_t source_y;
    uint32_t dest_x;
    uint32_t dest_y;
    uint32_t width;
    uint32_t height;
    NativeResolveMode mode = NativeResolveMode::kCopy;
    // A multisampled source (shader modes only, native_resolve_ms_cs): bits 0-1 the number of host
    // samples read per pixel (1, or 2 averaged), bits 4-5 and 6-7 their indices, bits 8-9 how two
    // are averaged (0 float, 1 k_8_8_8_8 and 2 k_2_10_10_10 as the resolve shaders do it).
    // 0: single-sampled.
    uint32_t source_samples = 0;
    // kScale multiplier.
    float scale = 1.0f;
    // Shader modes: resolution-scaled host pixel columns / rows at the left / top of the rectangle
    // taken from the first column / row after them - the half-pixel offset gap fill of the resolve
    // shaders (ResolveEdramInfo::fill_half_pixel_offset).
    uint32_t fill_x = 0;
    uint32_t fill_y = 0;
    // Shader modes: the non-shader-visible SRV of the source to copy if source_srv_format is
    // DXGI_FORMAT_UNKNOWN, otherwise a 2D source SRV with source_srv_format is created.
    D3D12_CPU_DESCRIPTOR_HANDLE source_srv = {};
    DXGI_FORMAT source_srv_format = DXGI_FORMAT_UNKNOWN;
    // Shader modes: the format of the UAV of the textures.
    DXGI_FORMAT dest_uav_format = DXGI_FORMAT_UNKNOWN;
    // With resolution scaling, a 1x copy (draw_resolution_scale_threshold) from a 1x render target:
    // the source pixels and the textures (unscaled ones) are guest pixels.
    bool unscaled = false;
  };
  struct NativeResolveWriteCounts {
    uint32_t written = 0;
    uint32_t incompatible = 0;
    uint32_t not_up_to_date = 0;
    uint32_t replacement = 0;
    uint32_t outside = 0;
    // Written although not up to date before, as the resolve covered the whole texture.
    uint32_t covered = 0;
    // Shader modes: the texture was created without D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    uint32_t no_uav = 0;
    // Shader modes: native_resolve_cs pipeline not created.
    uint32_t no_pipeline = 0;
    // Shader modes: failed to get the one-use descriptors.
    uint32_t descriptors = 0;
    // gpu_lazy_resolve_copies: a written texture contains the whole rectangle, so it mirrors the
    // destination memory there (kept as the mirror for AddLazyResolveCopy).
    bool mirror = false;
  };
  // Writes the rectangle into every texture at write.base_page that is a single-level unsigned 2D
  // tiled texture with the format, pitch and endianness of the resolve, matching resolution
  // scaling, and up to date before the resolve (CaptureUpToDateBeforeResolve), then marks those
  // up to date. source_state is the current state of the source, updated if it's transitioned.
  // kCopy uses CopyTextureRegion, the other modes native_resolve_cs.
  void WriteNativeResolveToTextures(const NativeResolveWrite& write,
                                    D3D12_RESOURCE_STATES& source_state,
                                    NativeResolveWriteCounts& counts);

  // gpu_lazy_resolve_copies: a direct colour resolve whose rectangle a natively written texture
  // contains completely (the mirror: same base, pitch, format and endianness as the destination,
  // so it holds what the copy would write) skips its copy into the scaled resolve memory. The
  // memory is rebuilt from the mirror (resolve_lazy_rebuild_32bpp, the copy's dispatch with the
  // mirror as the source) before anything reads it: a scaled texture load overlapping it, a resolve
  // readback, another resolve partly overlapping it. A later resolve covering the same rectangle of
  // the same surface makes it moot. The mirror stays in the cache while the copy is pending.
  struct LazyResolveCopy {
    // The surface and the rectangle (guest pixels, unscaled, relative to dest_base).
    uint32_t dest_base = 0;
    uint32_t pitch = 0;
    xenos::TextureFormat format = xenos::TextureFormat::k_8_8_8_8;
    xenos::Endian endianness = xenos::Endian::kNone;
    uint32_t rect_x = 0, rect_y = 0, rect_width = 0, rect_height = 0;
    // The range the copy would have written, and the length to make current from dest_base (as
    // the copy does).
    uint32_t extent_start = 0, extent_length = 0, current_length = 0;
    // The rebuild's root constants: the copy's dest_relative constants, then 0 and the packing.
    std::array<uint32_t, 8> root_constants = {};
    uint32_t root_constant_count = 0;
    uint32_t group_count_x = 0, group_count_y = 0;
    // The integer view of the mirror the rebuild reads (R8G8B8A8_UINT for k_8_8_8_8).
    DXGI_FORMAT srv_format = DXGI_FORMAT_UNKNOWN;
    // gpu_lazy_resolve_copies_unscaled: a copy into shared memory (scale 1, or a 1x copy under
    // draw_resolution_scale_threshold): the constants are the copy's full ones (dest_base too),
    // the rebuild writes shared memory, and it waits for SharedMemory::RequestRanges (every
    // reader of shared memory asks for its ranges first) instead of scaled texture loads.
    bool unscaled = false;
    // Same kind, same surface and this rectangle containing the other's.
    bool Covers(const LazyResolveCopy& other) const {
      return unscaled == other.unscaled && dest_base == other.dest_base && pitch == other.pitch &&
             format == other.format && endianness == other.endianness && rect_x <= other.rect_x &&
             rect_y <= other.rect_y && rect_x + rect_width >= other.rect_x + other.rect_width &&
             rect_y + rect_height >= other.rect_y + other.rect_height;
    }
  };
  // The rebuild pipelines (into the scaled resolve memory, into shared memory) and their root
  // signature (the direct resolve colour one), owned by the render target cache.
  void SetLazyResolveRebuild(ID3D12RootSignature* root_signature, ID3D12PipelineState* pipeline,
                             ID3D12PipelineState* pipeline_unscaled) {
    lazy_rebuild_root_signature_ = root_signature;
    lazy_rebuild_pipeline_ = pipeline;
    lazy_rebuild_unscaled_pipeline_ = pipeline_unscaled;
  }
  bool IsLazyResolveRebuildAvailable(bool unscaled = false) const {
    return lazy_rebuild_root_signature_ &&
           (unscaled ? lazy_rebuild_unscaled_pipeline_ != nullptr
                     : lazy_rebuild_pipeline_ != nullptr && IsDrawResolutionScaled());
  }
  bool HasLazyResolveCopies() const { return !lazy_resolve_copies_.empty(); }
  bool HasUnscaledLazyResolveCopies() const { return lazy_unscaled_pending_ != 0; }
  // Before a resolve writes its destination (resolve: its surface, rectangle and extent): rebuilds
  // the pending copies its extent overlaps, except those it covers completely, which are dropped.
  void PrepareLazyResolveCopiesForResolve(const LazyResolveCopy& resolve);
  // After WriteNativeResolveToTextures reported a mirror: the copy stays pending instead of being
  // made. False if the mirror is gone (the copy must be made then).
  bool AddLazyResolveCopy(const LazyResolveCopy& copy);
  // Which pending copies MaterializeLazyResolveCopies looks at.
  enum LazyResolveKinds : uint32_t {
    kLazyResolveScaled = 1,
    kLazyResolveUnscaled = 2,
    kLazyResolveAll = 3,
  };
  // Rebuilds the pending copies of the kinds overlapping the range (before something reads it).
  void MaterializeLazyResolveCopies(uint32_t start, uint32_t length,
                                    uint32_t kinds = kLazyResolveAll);
  // gpu_lazy_resolve_copies_verify: the rebuild of a copy that was made anyway, from the mirror
  // WriteNativeResolveToTextures reported last, over the copy's result (the caller reads the
  // destination back before and after).
  bool RebuildFromLastMirror(const LazyResolveCopy& copy);

  // Returns the ID3D12Resource of the front buffer texture (in
  // NON_PIXEL_SHADER_RESOURCE state), or nullptr in case of failure, and writes
  // the description of its SRV. May call LoadTextureData, so the same
  // restrictions (such as about descriptor heap change possibility) apply.
  ID3D12Resource* RequestSwapTexture(D3D12_SHADER_RESOURCE_VIEW_DESC& srv_desc_out,
                                     xenos::TextureFormat& format_out,
                                     uint32_t* width_unscaled_out = nullptr,
                                     uint32_t* height_unscaled_out = nullptr);

 protected:
  bool IsSignedVersionSeparateForFormat(TextureKey key) const override;
  bool IsScaledResolveSupportedForFormat(TextureKey key) const override;
  uint32_t GetHostFormatSwizzle(TextureKey key) const override;

  uint32_t GetMaxHostTextureWidthHeight(xenos::DataDimension dimension) const override;
  uint32_t GetMaxHostTextureDepthOrArraySize(xenos::DataDimension dimension) const override;

  std::unique_ptr<Texture> CreateTexture(TextureKey key) override;

  // This binds pipelines, allocates descriptors, and copies!
  bool LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                             bool load_mips) override;

  void UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) override;

 private:
  // texture_dump_dir / texture_replace_dir on a texture's first load: writes its top level as a
  // DDS named by its content hash, and/or puts <hash>.dds from the replacement folder in its
  // place. True when it was replaced (nothing else to load then).
  class D3D12Texture;
  bool DumpOrReplaceTexture(D3D12Texture& texture);
  // A replacement file to load (or a texture patch to run on the guest's data), and what came
  // of it.
  struct ReplacementJob {
    TextureKey key;
    std::filesystem::path path;
    // texture_patch.h: the top level as the dump has it, for the patch of `hash`.
    std::vector<uint8_t> patch_source;
    uint64_t hash = 0;
    D3D12_RESOURCE_DESC desc;
    int family = -1;
    uint32_t width = 0, height = 0;
    std::string name;
  };
  struct ReplacementResult {
    TextureKey key;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource, upload;
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 16> layouts;
    std::string name;
    uint32_t guest_width = 0, guest_height = 0, width = 0, height = 0, mips = 0;
    uint64_t make_us = 0;
  };
  bool MakeReplacement(const ReplacementJob& job, ReplacementResult& out) const;
  void ApplyReplacement(D3D12Texture& texture, ReplacementResult& result, bool in_use);
  // texture_replace_async: workers make the resources, BeginFrame puts them in.
  void QueueReplacement(ReplacementJob job);
  void ReplacementWorker();
  void ApplyReadyReplacements();
  void StopReplacementWorkers();
  std::mutex replacement_mutex_;
  std::condition_variable replacement_wake_;
  std::deque<ReplacementJob> replacement_jobs_;
  std::vector<ReplacementResult> replacement_results_;
  std::atomic<bool> replacement_results_ready_{false};
  bool replacement_stop_ = false;
  std::vector<std::thread> replacement_workers_;
  static constexpr uint32_t kLoadGuestXThreadsPerGroupLog2 = 2;
  static constexpr uint32_t kLoadGuestYBlocksPerGroupLog2 = 5;

  struct HostFormat {
    // Format info for the regular case.
    // DXGI format (typeless when different signedness or number representation
    // is used) for the texture resource.
    DXGI_FORMAT dxgi_format_resource;
    // DXGI format for unsigned normalized or unsigned/signed float SRV.
    DXGI_FORMAT dxgi_format_unsigned;
    // The regular load shader, used when special load shaders (like
    // signed-specific or decompressing) aren't needed.
    LoadShaderIndex load_shader;
    // DXGI format for signed normalized or unsigned/signed float SRV.
    DXGI_FORMAT dxgi_format_signed;
    // If the signed version needs a different bit representation on the host,
    // this is the load shader for the signed version. Otherwise the regular
    // load_shader will be used for the signed version, and a single copy will
    // be created if both unsigned and signed are used.
    LoadShaderIndex load_shader_signed;

    // Do NOT add integer DXGI formats to this - they are not filterable, can
    // only be read with Load, not Sample! If any game is seen using num_format
    // 1 for fixed-point formats (for floating-point, it's normally set to 1
    // though), add a constant buffer containing multipliers for the
    // textures and multiplication to the tfetch implementation.

    // Whether the DXGI format, if not uncompressing the texture, consists of
    // blocks, thus copy regions must be aligned to block size (assuming it's
    // the same as the guest block size).
    bool is_block_compressed;
    // Uncompression info for when the regular host format for this texture is
    // block-compressed, but the size is not block-aligned, and thus such
    // texture cannot be created in Direct3D on PC and needs decompression,
    // however, such textures are common, for instance, in 4D5307E6. This only
    // supports unsigned normalized formats - let's hope GPUSIGN_SIGNED was not
    // used for DXN and DXT5A.
    DXGI_FORMAT dxgi_format_uncompressed;
    LoadShaderIndex load_shader_decompress;

    // Mapping of Xenos swizzle components to DXGI format components.
    uint32_t swizzle;
  };

  class D3D12Texture final : public Texture {
   public:
    union SRVDescriptorKey {
      uint32_t key;
      struct {
        uint32_t is_signed : 1;
        uint32_t host_swizzle : 12;
        uint32_t dimension : 2;
      };

      SRVDescriptorKey() : key(0) { static_assert_size(*this, sizeof(key)); }

      struct Hasher {
        size_t operator()(const SRVDescriptorKey& key) const {
          return std::hash<decltype(key.key)>{}(key.key);
        }
      };
      bool operator==(const SRVDescriptorKey& other_key) const { return key == other_key.key; }
      bool operator!=(const SRVDescriptorKey& other_key) const { return !(*this == other_key); }
    };

    ID3D12Resource* GetOrCreate3DAs2DResource(D3D12_RESOURCE_STATES end_state);

    explicit D3D12Texture(D3D12TextureCache& texture_cache, const TextureKey& key,
                          ID3D12Resource* resource, D3D12_RESOURCE_STATES resource_state,
                          bool track_usage = true);
    ~D3D12Texture();

    ID3D12Resource* resource() const { return resource_.Get(); }

    D3D12_RESOURCE_STATES SetResourceState(D3D12_RESOURCE_STATES new_state) {
      D3D12_RESOURCE_STATES old_state = resource_state_;
      resource_state_ = new_state;
      return old_state;
    }

    uint32_t GetSRVDescriptorIndex(SRVDescriptorKey descriptor_key) const {
      auto it = srv_descriptors_.find(descriptor_key);
      return it != srv_descriptors_.cend() ? it->second : UINT32_MAX;
    }

    void AddSRVDescriptorIndex(SRVDescriptorKey descriptor_key, uint32_t descriptor_index) {
      srv_descriptors_.emplace(descriptor_key, descriptor_index);
    }

    // A replacement from texture_replace_dir (its own size and mip count) in place of the
    // resource made for the guest texture; the guest's data isn't loaded into it any more.
    // `upload` holds the replacement's data until the texture goes. in_use: frames under way may
    // draw with the old resource and descriptors, so they stay until the texture goes too.
    void SetReplacement(Microsoft::WRL::ComPtr<ID3D12Resource> resource,
                        D3D12_RESOURCE_STATES state, Microsoft::WRL::ComPtr<ID3D12Resource> upload,
                        uint32_t mip_levels, bool in_use);
    uint32_t replaced_mip_levels() const { return replaced_mip_levels_; }
    // Looked at once for texture_dump_dir / texture_replace_dir (on its first load).
    bool files_checked = false;

   private:
    uint32_t replaced_mip_levels_ = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> replacement_upload_;
    Microsoft::WRL::ComPtr<ID3D12Resource> replaced_original_;
    std::vector<uint32_t> retired_srv_descriptors_;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
    D3D12_RESOURCE_STATES resource_state_;
    std::unique_ptr<D3D12Texture> texture_3d_as_2d_;

    // For bindful - indices in the non-shader-visible descriptor cache for
    // copying to the shader-visible heap (much faster than recreating, which,
    // according to profiling, was often a bottleneck in many games).
    // For bindless - indices in the global shader-visible descriptor heap.
    std::unordered_map<SRVDescriptorKey, uint32_t, SRVDescriptorKey::Hasher> srv_descriptors_;
  };

  static constexpr uint32_t kSRVDescriptorCachePageSize = 65536;

  struct SRVDescriptorCachePage {
   public:
    explicit SRVDescriptorCachePage(ID3D12DescriptorHeap* heap)
        : heap_(heap), heap_start_(heap->GetCPUDescriptorHandleForHeapStart()) {}
    SRVDescriptorCachePage(const SRVDescriptorCachePage& page) = delete;
    SRVDescriptorCachePage& operator=(const SRVDescriptorCachePage& page) = delete;
    SRVDescriptorCachePage(SRVDescriptorCachePage&& page) {
      std::swap(heap_, page.heap_);
      std::swap(heap_start_, page.heap_start_);
    }
    SRVDescriptorCachePage& operator=(SRVDescriptorCachePage&& page) {
      std::swap(heap_, page.heap_);
      std::swap(heap_start_, page.heap_start_);
      return *this;
    }

    ID3D12DescriptorHeap* heap() const { return heap_.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE heap_start() const { return heap_start_; }

   private:
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
    D3D12_CPU_DESCRIPTOR_HANDLE heap_start_;
  };

  struct D3D12TextureBinding {
    // Descriptor indices of texture and texture_signed of the respective
    // TextureBinding returned from FindOrCreateTextureDescriptor.
    uint32_t descriptor_index;
    uint32_t descriptor_index_signed;

    D3D12TextureBinding() { Reset(); }

    void Reset() {
      descriptor_index = UINT32_MAX;
      descriptor_index_signed = UINT32_MAX;
    }
  };

  class ScaledResolveVirtualBuffer {
   public:
    explicit ScaledResolveVirtualBuffer(ID3D12Resource* resource,
                                        D3D12_RESOURCE_STATES resource_state)
        : resource_(resource), resource_state_(resource_state) {}
    ID3D12Resource* resource() const { return resource_.Get(); }
    D3D12_RESOURCE_STATES SetResourceState(D3D12_RESOURCE_STATES new_state) {
      D3D12_RESOURCE_STATES old_state = resource_state_;
      if (old_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        uav_barrier_pending_ = false;
      }
      resource_state_ = new_state;
      return old_state;
    }
    // After writing through a UAV.
    void SetUAVBarrierPending() {
      if (resource_state_ == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        uav_barrier_pending_ = true;
      }
    }
    // After an aliasing barrier (which is even stronger than an UAV barrier).
    void ClearUAVBarrierPending() { uav_barrier_pending_ = false; }

   private:
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
    D3D12_RESOURCE_STATES resource_state_;
    bool uav_barrier_pending_ = false;
  };

  explicit D3D12TextureCache(const RegisterFile& register_file, D3D12SharedMemory& shared_memory,
                             uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
                             D3D12CommandProcessor& command_processor,
                             bool bindless_resources_used);

  bool Initialize();

  // Whether decompression is needed on the host (Direct3D only allows creation
  // of block-compressed textures with 4x4-aligned dimensions on PC).
  bool IsDecompressionNeeded(xenos::TextureFormat format, uint32_t width, uint32_t height) const;
  DXGI_FORMAT GetDXGIResourceFormat(xenos::TextureFormat format, uint32_t width,
                                    uint32_t height) const {
    const HostFormat& host_format = host_formats_[uint32_t(format)];
    return IsDecompressionNeeded(format, width, height) ? host_format.dxgi_format_uncompressed
                                                        : host_format.dxgi_format_resource;
  }
  DXGI_FORMAT GetDXGIResourceFormat(TextureKey key) const {
    return GetDXGIResourceFormat(key.format, key.GetWidth(), key.GetHeight());
  }
  DXGI_FORMAT GetDXGIUnormFormat(xenos::TextureFormat format, uint32_t width,
                                 uint32_t height) const {
    const HostFormat& host_format = host_formats_[uint32_t(format)];
    return IsDecompressionNeeded(format, width, height) ? host_format.dxgi_format_uncompressed
                                                        : host_format.dxgi_format_unsigned;
  }
  DXGI_FORMAT GetDXGIUnormFormat(TextureKey key) const {
    return GetDXGIUnormFormat(key.format, key.GetWidth(), key.GetHeight());
  }

  LoadShaderIndex GetLoadShaderIndex(TextureKey key) const;

  static constexpr bool AreDimensionsCompatible(xenos::FetchOpDimension binding_dimension,
                                                xenos::DataDimension resource_dimension) {
    switch (binding_dimension) {
      case xenos::FetchOpDimension::k1D:
      case xenos::FetchOpDimension::k2D:
        return resource_dimension == xenos::DataDimension::k1D ||
               resource_dimension == xenos::DataDimension::k2DOrStacked ||
               resource_dimension == xenos::DataDimension::k3D;
      case xenos::FetchOpDimension::k3DOrStacked:
        return resource_dimension == xenos::DataDimension::k3D;
      case xenos::FetchOpDimension::kCube:
        return resource_dimension == xenos::DataDimension::kCube;
      default:
        return false;
    }
  }

  // Returns the index of an existing of a newly created non-shader-visible
  // cached (for bindful) or a shader-visible global (for bindless) descriptor,
  // or UINT32_MAX if failed to create.
  uint32_t FindOrCreateTextureDescriptor(D3D12Texture& texture, xenos::DataDimension dimension,
                                         bool is_signed, uint32_t host_swizzle);
  void ReleaseTextureDescriptor(uint32_t descriptor_index);
  D3D12_CPU_DESCRIPTOR_HANDLE GetTextureDescriptorCPUHandle(uint32_t descriptor_index) const;

  size_t GetScaledResolveBufferCount() const {
    assert_true(IsDrawResolutionScaled());
    // Make sure any range up to 1 GB is accessible through 1 or 2 buffers.
    // 2x2 scale buffers - just one 2 GB buffer for all 2 GB.
    // 3x3 scale buffers - 4 buffers:
    //  +0.0 +0.5 +1.0 +1.5 +2.0 +2.5 +3.0 +3.5 +4.0 +4.5
    // |___________________|___________________|
    //           |___________________|______________|
    // Buffer N has an offset of N * 1 GB in the scaled resolve address space.
    // The logic is:
    // - 2 GB can be accessed through a [0 GB ... 2 GB) buffer - only need one.
    // - 2.1 GB needs [0 GB ... 2 GB) and [1 GB ... 2.1 GB) - two buffers.
    // - 3 GB needs [0 GB ... 2 GB) and [1 GB ... 3 GB) - two buffers.
    // - 3.1 GB needs [0 GB ... 2 GB), [1 GB ... 3 GB) and [2 GB ... 3.1 GB) -
    //   three buffers.
    uint64_t address_space_size = uint64_t(SharedMemory::kBufferSize) *
                                  (draw_resolution_scale_x() * draw_resolution_scale_y());
    return size_t((address_space_size - 1) >> 30);
  }
  // Returns indices of two scaled resolve virtual buffers that the location in
  // memory may be accessible through. May be the same if it's a location near
  // the beginning or the end of the address represented only by one buffer.
  std::array<size_t, 2> GetPossibleScaledResolveBufferIndices(uint64_t address_scaled) const {
    assert_true(IsDrawResolutionScaled());
    size_t address_gb = size_t(address_scaled >> 30);
    size_t max_index = GetScaledResolveBufferCount() - 1;
    // In different cases for 3x3:
    //  +0.0 +0.5 +1.0 +1.5 +2.0 +2.5 +3.0 +3.5 +4.0 +4.5
    // |12________2________|1_________2________|
    //           |1_________2________|1_________12__|
    return std::array<size_t, 2>{std::min(address_gb, max_index),
                                 std::min(std::max(address_gb, size_t(1)) - size_t(1), max_index)};
  }
  // The index is also the gigabyte offset of the buffer from the start of the
  // scaled physical memory address space.
  size_t GetCurrentScaledResolveBufferIndex() const {
    return scaled_resolve_1gb_buffer_indices_[scaled_resolve_current_range_start_scaled_ >> 30];
  }
  ScaledResolveVirtualBuffer& GetCurrentScaledResolveBuffer() {
    ScaledResolveVirtualBuffer* scaled_resolve_buffer =
        scaled_resolve_2gb_buffers_[GetCurrentScaledResolveBufferIndex()].get();
    assert_not_null(scaled_resolve_buffer);
    return *scaled_resolve_buffer;
  }

  xenos::ClampMode NormalizeClampMode(xenos::ClampMode clamp_mode) const;

  static const HostFormat host_formats_[64];

  D3D12CommandProcessor& command_processor_;
  bool bindless_resources_used_;

  Microsoft::WRL::ComPtr<ID3D12RootSignature> load_root_signature_;
  std::array<Microsoft::WRL::ComPtr<ID3D12PipelineState>, kLoadShaderCount> load_pipelines_;
  // Load pipelines for resolution-scaled resolve targets.
  std::array<Microsoft::WRL::ComPtr<ID3D12PipelineState>, kLoadShaderCount> load_pipelines_scaled_;

  // gpu_native_resolves: native_resolve_cs, created only with the cvar enabled.
  // Root parameter 0 - NativeResolveConstants, 1 - source SRV table, 2 - destination UAV table.
  struct NativeResolveConstants {
    uint32_t mode;
    float scale;
    int32_t source_offset[2];
    int32_t dest_offset[2];
    uint32_t size[2];
    uint32_t fill[2];
    uint32_t samples;  // NativeResolveWrite::source_samples
  };
  Microsoft::WRL::ComPtr<ID3D12RootSignature> native_resolve_root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> native_resolve_pipeline_;
  // The same with a multisampled source (native_resolve_ms.cs.hlsl), same root signature.
  Microsoft::WRL::ComPtr<ID3D12PipelineState> native_resolve_ms_pipeline_;

  std::vector<SRVDescriptorCachePage> srv_descriptor_cache_;
  uint32_t srv_descriptor_cache_allocated_;
  // Indices of cached descriptors used by deleted textures, for reuse.
  std::vector<uint32_t> srv_descriptor_cache_free_;

  enum class NullSRVDescriptorIndex {
    k2DArray,
    k3D,
    kCube,

    kCount,
  };
  // Contains null SRV descriptors of dimensions from NullSRVDescriptorIndex.
  // For copying, not shader-visible.
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> null_srv_descriptor_heap_;
  D3D12_CPU_DESCRIPTOR_HANDLE null_srv_descriptor_heap_start_;

  std::array<D3D12TextureBinding, xenos::kTextureFetchConstantCount> d3d12_texture_bindings_;

  // Unsupported texture formats used during this frame (for research and
  // testing).
  enum : uint8_t {
    kUnsupportedResourceBit = 1,
    kUnsupportedUnormBit = kUnsupportedResourceBit << 1,
    kUnsupportedSnormBit = kUnsupportedUnormBit << 1,
  };
  uint8_t unsupported_format_features_used_[64];

  // The tiled buffer for resolved data with resolution scaling.
  // Because on Direct3D 12 (at least on Windows 10 2004) typed SRV or UAV
  // creation fails for offsets above 4 GB, a single tiled 4.5 GB buffer can't
  // be used for 3x3 resolution scaling.
  // Instead, "sliding window" buffers allowing to access a single range of up
  // to 1 GB (or up to 2 GB, depending on the low bits) at any moment are used.
  // Parts of 4.5 GB address space can be accessed through 2 GB buffers as:
  //  +0.0 +0.5 +1.0 +1.5 +2.0 +2.5 +3.0 +3.5 +4.0 +4.5
  // |___________________|___________________|      or
  //           |___________________|______________|
  // (2 GB is also the amount of scaled physical memory with 2x resolution
  // scale, and older Intel GPUs, while support tiled resources, only support 31
  // virtual address bits per resource).
  // Index is first gigabyte. Only including buffers containing over 1 GB
  // (because otherwise the data will be fully contained in another).
  // Size is calculated the same as in GetScaledResolveBufferCount.
  std::array<std::unique_ptr<ScaledResolveVirtualBuffer>,
             (uint64_t(SharedMemory::kBufferSize) *
                  (kMaxDrawResolutionScaleAlongAxis * kMaxDrawResolutionScaleAlongAxis) -
              1) /
                 (UINT32_C(1) << 30)>
      scaled_resolve_2gb_buffers_;
  // Not very big heaps (16 MB) because they are needed pretty sparsely. One
  // 2x-scaled 1280x720x32bpp texture is slighly bigger than 14 MB.
  static constexpr uint32_t kScaledResolveHeapSizeLog2 = 24;
  static constexpr uint32_t kScaledResolveHeapSize = uint32_t(1) << kScaledResolveHeapSizeLog2;
  static_assert((kScaledResolveHeapSize % D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) == 0,
                "Scaled resolve heap size must be a multiple of Direct3D tile size");
  static_assert(kScaledResolveHeapSizeLog2 <= SharedMemory::kBufferSizeLog2,
                "Scaled resolve heaps are assumed to be wholly mappable irrespective of "
                "resolution scale, never truncated, for example, if the scaled resolve "
                "address space is 4.5 GB, but the heap size is 1 GB");
  static_assert(kScaledResolveHeapSizeLog2 <= 30,
                "Scaled resolve heaps are assumed to only be wholly mappable to up to "
                "two 2 GB buffers");
  // Resident portions of the tiled buffer.
  std::vector<Microsoft::WRL::ComPtr<ID3D12Heap>> scaled_resolve_heaps_;
  // Number of currently resident portions of the tiled buffer, for profiling.
  uint32_t scaled_resolve_heap_count_ = 0;
  // Current scaled resolve state.
  // For aliasing barrier placement, last owning buffer index for each of 1 GB.
  size_t scaled_resolve_1gb_buffer_indices_[(uint64_t(SharedMemory::kBufferSize) *
                                                 kMaxDrawResolutionScaleAlongAxis *
                                                 kMaxDrawResolutionScaleAlongAxis +
                                             ((uint32_t(1) << 30) - 1)) >>
                                            30];
  // Range used in the last successful MakeScaledResolveRangeCurrent call.
  uint64_t scaled_resolve_current_range_start_scaled_;
  uint64_t scaled_resolve_current_range_length_scaled_;

  // gpu_lazy_resolve_copies (see LazyResolveCopy): the pending copies with their mirrors, kept in
  // the cache by marking them used every submission.
  struct PendingLazyResolveCopy {
    LazyResolveCopy copy;
    D3D12Texture* mirror;
  };
  std::vector<PendingLazyResolveCopy> lazy_resolve_copies_;
  // The mirror of the last WriteNativeResolveToTextures call, if it found one.
  D3D12Texture* last_native_resolve_mirror_ = nullptr;
  // Owned by the render target cache.
  ID3D12RootSignature* lazy_rebuild_root_signature_ = nullptr;
  ID3D12PipelineState* lazy_rebuild_pipeline_ = nullptr;
  ID3D12PipelineState* lazy_rebuild_unscaled_pipeline_ = nullptr;
  // How many of lazy_resolve_copies_ are unscaled (the RequestRanges hook returns at once if 0),
  // and the range their extents span (the hook and the draws' vertex buffers check it first).
  uint32_t lazy_unscaled_pending_ = 0;
  uint32_t lazy_unscaled_start_ = UINT32_MAX;
  uint32_t lazy_unscaled_end_ = 0;
  // Counts since the last report (every few seconds).
  uint64_t lazy_skipped_ = 0;
  uint64_t lazy_skipped_unscaled_ = 0;
  uint64_t lazy_rebuilt_ = 0;
  uint64_t lazy_dropped_ = 0;
  uint64_t lazy_failed_ = 0;
  // Unscaled copies a reader asked for while the hook was suppressed (should stay 0: the draw
  // writes them before; one would mean a reader the draw doesn't know of).
  uint64_t lazy_late_ = 0;
  uint64_t lazy_last_report_ms_ = 0;
  // Dispatches the copy's rebuild from the mirror into the scaled resolve memory, or into shared
  // memory for an unscaled one.
  bool RebuildLazyResolveCopy(const LazyResolveCopy& copy, D3D12Texture& mirror);
  // Removes pending copy i (keeping lazy_unscaled_pending_).
  void EraseLazyResolveCopy(size_t i);
  // SharedMemory's before-request hook: the unscaled pending copies overlapping the ranges.
  static void BeforeSharedMemoryRequest(void* context, const std::pair<uint32_t, uint32_t>* ranges,
                                        size_t count, bool suppressed);
  void ReportLazyResolveCopies();
};

}  // namespace rex::graphics::d3d12
