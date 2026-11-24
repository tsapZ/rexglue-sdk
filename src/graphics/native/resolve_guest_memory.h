#pragma once

#include <cstdint>
#include <cstring>
#include <span>

#include <rex/graphics/pipeline/texture/util.h>

#include "draw_input.h"

namespace rex::graphics::native {

inline bool StoreTiledResolveRgba8(std::span<uint8_t> guest, uint32_t pitch,
                                   const ResolveRectangle& rect, xenos::Endian128 endian,
                                   std::span<const uint8_t> pixels) {
  constexpr uint32_t kTexelBytes = 4;
  if (!rect.width || !rect.height || !pitch || rect.x >= pitch || rect.width > pitch - rect.x ||
      rect.y >= xenos::kTexture2DCubeMaxWidthHeight ||
      rect.height > xenos::kTexture2DCubeMaxWidthHeight - rect.y ||
      pitch > xenos::kTexture2DCubeMaxWidthHeight || uint32_t(endian) > 3 ||
      pixels.size() != uint64_t(rect.width) * rect.height * kTexelBytes) {
    return false;
  }

  // The caller maps the entire tiled extent so none of the writes can fail partway.
  const uint32_t end = texture_util::GetTiledAddressUpperBound2D(rect.x + rect.width,
                                                                 rect.y + rect.height, pitch, 2);
  if (end > guest.size()) {
    return false;
  }
  const auto dword_endian = xenos::Endian(endian);
  for (uint32_t y = 0; y < rect.height; ++y) {
    for (uint32_t x = 0; x < rect.width; ++x) {
      const size_t offset = size_t(
          texture_util::GetTiledOffset2D(int32_t(rect.x + x), int32_t(rect.y + y), pitch, 2));
      uint32_t texel;
      std::memcpy(&texel, pixels.data(), sizeof(texel));
      texel = xenos::GpuSwap(texel, dword_endian);
      std::memcpy(guest.data() + offset, &texel, sizeof(texel));
      pixels = pixels.subspan(kTexelBytes);
    }
  }
  return true;
}

}  // namespace rex::graphics::native
