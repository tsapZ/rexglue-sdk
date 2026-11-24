#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include <rex/graphics/edram_layout.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>

namespace rex::graphics::native {

constexpr uint32_t kPhysicalMemorySize = 512 * 1024 * 1024;

constexpr bool IsPhysicalRange(uint32_t address, size_t size) {
  return address < kPhysicalMemorySize && size && size <= kPhysicalMemorySize - address;
}

inline bool DecodeDmaIndices(std::span<const uint8_t> source, xenos::IndexFormat format,
                             xenos::Endian endian, std::span<uint32_t> indices) {
  const bool wide = format == xenos::IndexFormat::kInt32;
  const size_t stride = wide ? sizeof(uint32_t) : sizeof(uint16_t);
  if (indices.size() > source.size() / stride) {
    return false;
  }
  for (uint32_t& index : indices) {
    if (wide) {
      uint32_t value;
      std::memcpy(&value, source.data(), sizeof(value));
      index = xenos::GpuSwap(value, endian) & xenos::kVertexIndexMask;
    } else {
      uint16_t value;
      std::memcpy(&value, source.data(), sizeof(value));
      // Match the guest primitive processor's normalization for 16-bit DMA.
      index = endian == xenos::Endian::k8in16 || endian == xenos::Endian::k8in32
                  ? xenos::GpuSwap(value, xenos::Endian::k8in16)
                  : value;
    }
    source = source.subspan(stride);
  }
  return true;
}

// The translator expands each guest rectangle or point to four host vertices.
inline std::vector<uint32_t> QuadStripIndices(uint32_t primitive_count) {
  std::vector<uint32_t> indices;
  indices.reserve(5 * primitive_count);
  for (uint32_t primitive = 0; primitive < primitive_count; ++primitive) {
    if (primitive) {
      indices.push_back(UINT32_MAX);
    }
    for (uint32_t corner = 0; corner < 4; ++corner) {
      indices.push_back(primitive * 4 + corner);
    }
  }
  return indices;
}

// Converts the sanitized guest copy sample selection (the copy_sample_select of
// GetResolveInfo's destination coordinate info) to the sample index of the host
// multisampled image. 4x guest samples match the standard Vulkan sample
// locations; 2x guest samples are stored top/bottom, which the reference Vulkan
// resolve path converts through GetD3D10SampleIndexForGuest2xMSAA.
inline uint32_t GetResolveSampleIndex(xenos::MsaaSamples samples,
                                      xenos::CopySampleSelect selection) {
  if (samples == xenos::MsaaSamples::k2X) {
    return draw_util::GetD3D10SampleIndexForGuest2xMSAA(
        selection == xenos::CopySampleSelect::k1 ? 1 : 0, true);
  }
  return uint32_t(selection);
}

struct ResolveRectangle {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
};

// Rasterizer face culling for a host pipeline. Guest rectangle, point and line
// primitives are non-polygonal, so they have no front or back face and must not
// inherit the cull state left behind by the last polygonal draw.
struct CullFaces {
  bool front;
  bool back;
};

constexpr CullFaces GetCullFaces(bool primitive_polygonal, bool cull_front, bool cull_back) {
  return primitive_polygonal ? CullFaces{cull_front, cull_back} : CullFaces{false, false};
}

// Returns the least recently used cache entry whose GPU work has retired, so it
// can be dropped when the cache reaches its limit. Entries last used in the
// protected submissions (the one being recorded and the one still in flight) may
// still be referenced by queued or submitted commands, so they are skipped -
// pass first_protected_submission = last_retired_submission + 1 to keep only
// fences that have already been waited for. The caller's may_evict predicate
// can preserve the presented image. Nullopt if nothing can be evicted safely.
template <typename EntryMap, typename Predicate>
std::optional<typename EntryMap::key_type> SelectRetiredEviction(
    const EntryMap& entries, uint64_t first_protected_submission, Predicate may_evict) {
  std::optional<typename EntryMap::key_type> victim;
  uint64_t victim_use = UINT64_MAX;
  for (const auto& [key, entry] : entries) {
    if (entry.last_use_submission >= first_protected_submission || !may_evict(key, entry)) {
      continue;
    }
    if (!victim || entry.last_use_submission < victim_use) {
      victim = key;
      victim_use = entry.last_use_submission;
    }
  }
  return victim;
}

inline std::optional<ResolveRectangle> GetResolveRectangle(const draw_util::ResolveInfo& copy,
                                                           EdramLayout layout, uint32_t width,
                                                           uint32_t height, bool depth = false) {
  const auto& target = depth ? copy.depth_edram_info : copy.color_edram_info;
  const uint32_t pitch = target.pitch_tiles;
  if (!pitch) {
    return std::nullopt;
  }
  const uint32_t base = depth ? copy.depth_original_base : copy.color_original_base;
  const uint32_t tiles = layout.Wrap(target.base_tiles - base);
  const uint32_t tile_width =
      xenos::kEdramTileWidthSamples >> (target.msaa_samples >= xenos::MsaaSamples::k4X);
  const uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> (target.msaa_samples >= xenos::MsaaSamples::k2X);
  const ResolveRectangle rect{
      tiles % pitch * tile_width + copy.coordinate_info.edram_offset_x_div_8 * 8,
      tiles / pitch * tile_height + copy.coordinate_info.edram_offset_y_div_8 * 8,
      uint32_t(copy.coordinate_info.width_div_8) * 8, copy.height_div_8 * 8};
  if (rect.x > width || rect.y > height || !rect.width || !rect.height ||
      rect.width > width - rect.x || rect.height > height - rect.y) {
    return std::nullopt;
  }
  return rect;
}

}  // namespace rex::graphics::native
