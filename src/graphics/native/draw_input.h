#pragma once

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

struct ResolveRectangle {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
};

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
