#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <compare>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <xxhash.h>

#include <rex/graphics/edram_layout.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>

namespace rex::graphics::native {

constexpr uint32_t kPhysicalMemorySize = 512 * 1024 * 1024;
constexpr uint32_t kSharedVertexPageShift = 12;
constexpr uint32_t kSharedVertexPageSize = 1u << kSharedVertexPageShift;
constexpr uint32_t kSharedVertexPageCount = kPhysicalMemorySize >> kSharedVertexPageShift;

constexpr std::pair<uint32_t, uint32_t> GetPhysicalPageRange(uint32_t address, uint32_t size) {
  if (!size || address >= kPhysicalMemorySize) {
    return {0, 0};
  }
  const uint32_t clamped_size = std::min(size, kPhysicalMemorySize - address);
  const uint32_t first_page = address >> kSharedVertexPageShift;
  const uint32_t last_page = (address + clamped_size - 1) >> kSharedVertexPageShift;
  return {first_page, last_page - first_page + 1};
}

inline void ClearPageRangeBits(std::vector<uint64_t>& valid_words, uint32_t first_page,
                               uint32_t page_count) {
  if (!page_count) {
    return;
  }
  const uint32_t end_page = std::min(first_page + page_count, uint32_t(valid_words.size() * 64));
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    valid_words[word] &= ~mask;
    page += count;
  }
}

inline void SetPageRangeBits(std::vector<uint64_t>& valid_words, uint32_t first_page,
                             uint32_t page_count) {
  if (!page_count) {
    return;
  }
  const uint32_t end_page = std::min(first_page + page_count, uint32_t(valid_words.size() * 64));
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    valid_words[word] |= mask;
    page += count;
  }
}

inline bool HasAnyPageBitSet(const std::vector<uint64_t>& words, uint32_t first_page,
                             uint32_t page_count) {
  if (!page_count || words.empty()) {
    return false;
  }
  const uint32_t end_page = std::min(first_page + page_count, uint32_t(words.size() * 64));
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    if ((words[word] & mask) != 0) {
      return true;
    }
    page += count;
  }
  return false;
}

inline bool AreAllPageBitsSet(const std::vector<uint64_t>& words, uint32_t first_page,
                              uint32_t page_count) {
  if (!page_count) {
    return true;
  }
  const uint32_t limit = uint32_t(words.size() * 64);
  if (first_page >= limit || page_count > limit - first_page) {
    return false;
  }
  const uint32_t end_page = first_page + page_count;
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    if ((words[word] & mask) != mask) {
      return false;
    }
    page += count;
  }
  return true;
}

inline void AtomicSetPageRangeBits(std::vector<uint64_t>& words, uint32_t first_page,
                                     uint32_t page_count) {
  if (!page_count || words.empty()) {
    return;
  }
  const uint32_t end_page = std::min(first_page + page_count, uint32_t(words.size() * 64));
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    std::atomic_ref<uint64_t>(words[word]).fetch_or(mask, std::memory_order_relaxed);
    page += count;
  }
}

inline void AtomicClearPageRangeBits(std::vector<uint64_t>& words, uint32_t first_page,
                                       uint32_t page_count) {
  if (!page_count || words.empty()) {
    return;
  }
  const uint32_t end_page = std::min(first_page + page_count, uint32_t(words.size() * 64));
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    std::atomic_ref<uint64_t>(words[word]).fetch_and(~mask, std::memory_order_release);
    page += count;
  }
}

inline bool AtomicHasAnyPageBitSet(const std::vector<uint64_t>& words, uint32_t first_page,
                                   uint32_t page_count) {
  if (!page_count || words.empty()) {
    return false;
  }
  const uint32_t end_page = std::min(first_page + page_count, uint32_t(words.size() * 64));
  for (uint32_t page = first_page; page < end_page;) {
    const uint32_t word = page >> 6;
    const uint32_t bit = page & 63;
    const uint32_t count = std::min(64 - bit, end_page - page);
    const uint64_t mask = (count == 64 ? ~uint64_t(0) : ((uint64_t(1) << count) - 1)) << bit;
    if ((std::atomic_ref<const uint64_t>(words[word]).load(std::memory_order_relaxed) & mask) !=
        0) {
      return true;
    }
    page += count;
  }
  return false;
}

inline bool AtomicFindNextUnsetPageRun(const std::vector<uint64_t>& words, uint32_t& cursor,
                                       uint32_t end_page, uint32_t& run_first_page,
                                       uint32_t& run_page_count) {
  const uint32_t limit = std::min(end_page, uint32_t(words.size() * 64));
  while (cursor < limit) {
    const uint32_t word = cursor >> 6;
    const uint32_t bit = cursor & 63;
    const uint64_t inv =
        ~std::atomic_ref<const uint64_t>(words[word]).load(std::memory_order_acquire) &
        (~uint64_t(0) << bit);
    if (!inv) {
      cursor = (word + 1) << 6;
      continue;
    }
    const uint32_t first = (word << 6) + uint32_t(std::countr_zero(inv));
    if (first >= limit) {
      cursor = limit;
      return false;
    }
    run_first_page = first;
    uint32_t end = first;
    while (end < limit) {
      const uint32_t w = end >> 6;
      const uint32_t b = end & 63;
      const uint64_t valid_mask =
          std::atomic_ref<const uint64_t>(words[w]).load(std::memory_order_acquire) >> b;
      if (!valid_mask) {
        end = std::min(limit, (w + 1) << 6);
      } else {
        end = std::min(limit, end + uint32_t(std::countr_zero(valid_mask)));
        break;
      }
    }
    run_page_count = end - first;
    cursor = end;
    return true;
  }
  return false;
}

// Inline 2D Xenos texture tiling offset (matches texture_util::GetTiledOffset2D).
struct Tiled2DRowState {
  int32_t y_macro;
  int32_t y_micro;
  int32_t y_bit0;
  int32_t y_bit4;
  int32_t y_bit3;
};

constexpr Tiled2DRowState MakeTiled2DRowState(int32_t gy, uint32_t pitch_aligned,
                                              uint32_t bpb_log2 = 0) {
  (void)bpb_log2;
  return {(gy >> 5) * int32_t(pitch_aligned >> 5), (gy & 0xE) << 2, (gy & 1) << 4,
          (gy & 16) << 7, (gy & 8) >> 2};
}

template <uint32_t BpbLog2>
constexpr int32_t GetTiledOffset2DFromRowT(int32_t gx, const Tiled2DRowState& row) {
  const int32_t macro = ((gx >> 5) + row.y_macro) << (BpbLog2 + 7);
  const int32_t micro = ((gx & 7) + row.y_micro) << BpbLog2;
  const int32_t offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + row.y_bit0;
  return ((offset & ~0x1FF) << 3) + row.y_bit4 + ((offset & 0x1C0) << 2) +
         (((row.y_bit3 + (gx >> 3)) & 3) << 6) + (offset & 0x3F);
}

constexpr int32_t GetTiledOffset2DFromRow(int32_t gx, const Tiled2DRowState& row,
                                          uint32_t bpb_log2) {
  const int32_t macro = ((gx >> 5) + row.y_macro) << (bpb_log2 + 7);
  const int32_t micro = ((gx & 7) + row.y_micro) << bpb_log2;
  const int32_t offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + row.y_bit0;
  return ((offset & ~0x1FF) << 3) + row.y_bit4 + ((offset & 0x1C0) << 2) +
         (((row.y_bit3 + (gx >> 3)) & 3) << 6) + (offset & 0x3F);
}

constexpr int32_t GetTiledOffset2DInline(int32_t gx, int32_t gy, uint32_t pitch,
                                         uint32_t bpb_log2) {
  const uint32_t pitch_aligned =
      (pitch + (xenos::kTextureTileWidthHeight - 1)) & ~(xenos::kTextureTileWidthHeight - 1);
  return GetTiledOffset2DFromRow(gx, MakeTiled2DRowState(gy, pitch_aligned, bpb_log2), bpb_log2);
}

template <size_t Swap>
inline void CopySwapDwords(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t dwords) {
  if constexpr (Swap == 0) {
    std::memcpy(dst, src, dwords * 4);
  } else {
    for (size_t i = 0; i < dwords; ++i) {
      uint32_t v;
      std::memcpy(&v, src + i * 4, 4);
      if constexpr (Swap == 1) {
        v = ((v & 0x00FF00FFu) << 8) | ((v & 0xFF00FF00u) >> 8);
      } else if constexpr (Swap == 2) {
        v = std::rotl(v, 16);
      } else {
        v = rex::byte_swap(v);
      }
      std::memcpy(dst + i * 4, &v, 4);
    }
  }
}

template <uint32_t BlockSize, size_t Swap>
inline void CopySwapBlock(const uint8_t* __restrict src, size_t offset,
                          uint8_t* __restrict dst) {
  if constexpr (BlockSize == 1) {
    *dst = src[offset ^ Swap];
  } else if constexpr (BlockSize == 2) {
    uint16_t v;
    if constexpr (Swap == 0) {
      std::memcpy(&v, src + offset, 2);
    } else if constexpr (Swap == 1) {
      std::memcpy(&v, src + offset, 2);
      v = uint16_t((v >> 8) | (v << 8));
    } else if constexpr (Swap == 2) {
      std::memcpy(&v, src + (offset ^ 2), 2);
    } else {
      std::memcpy(&v, src + (offset ^ 2), 2);
      v = uint16_t((v >> 8) | (v << 8));
    }
    std::memcpy(dst, &v, 2);
  } else {
    static_assert(BlockSize % 4 == 0);
    for (uint32_t w = 0; w < BlockSize; w += 4) {
      uint32_t v;
      std::memcpy(&v, src + offset + w, 4);
      if constexpr (Swap == 1) {
        v = ((v & 0x00FF00FFu) << 8) | ((v & 0xFF00FF00u) >> 8);
      } else if constexpr (Swap == 2) {
        v = std::rotl(v, 16);
      } else if constexpr (Swap == 3) {
        v = ((v >> 24) & 0xFFu) | ((v >> 8) & 0xFF00u) | ((v & 0xFF00u) << 8) | (v << 24);
      }
      std::memcpy(dst + w, &v, 4);
    }
  }
}

template <uint32_t BlockSize, size_t Swap>
inline void DecodeTiled2DCopyLevel(const uint8_t* __restrict source, size_t source_size,
                                   uint8_t* __restrict dst, uint32_t slices, uint32_t blocks_x,
                                   uint32_t blocks_y, uint32_t start_x, uint32_t start_y,
                                   uint32_t pitch_aligned, uint32_t bpb_log2,
                                   size_t slice_stride_bytes, size_t dst_slice_bytes,
                                   uint8_t* row_scratch = nullptr) {
  (void)bpb_log2;
  constexpr uint32_t kBpbLog2 = uint32_t(std::countr_zero(BlockSize));
  const size_t row_bytes = size_t(blocks_x) * BlockSize;
  const size_t max_slice_end =
      (blocks_x && blocks_y)
          ? size_t(texture_util::GetTiledAddressUpperBound2D(
                start_x + blocks_x, start_y + blocks_y, pitch_aligned, kBpbLog2))
          : 0;
  const bool in_bounds =
      slices > 0 && blocks_x > 0 && blocks_y > 0 && max_slice_end <= source_size &&
      size_t(slices - 1) * slice_stride_bytes <= source_size - max_slice_end;

  for (uint32_t slice = 0; slice < slices; ++slice) {
    const size_t slice_base = size_t(slice) * slice_stride_bytes;
    uint8_t* slice_dst = dst + size_t(slice) * dst_slice_bytes;
    for (uint32_t y = 0; y < blocks_y; ++y) {
      const auto row = MakeTiled2DRowState(int32_t(y + start_y), pitch_aligned, kBpbLog2);
      uint8_t* row_dst = slice_dst + size_t(y) * row_bytes;
      uint8_t* out = row_scratch ? row_scratch : row_dst;
      if (in_bounds) {
        uint32_t x = 0;
        if constexpr (BlockSize == 4) {
          while (x < blocks_x && ((x + start_x) & 3) != 0) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<2>(int32_t(x + start_x), row));
            CopySwapBlock<4, Swap>(source, off, out + size_t(x) * 4);
            ++x;
          }
          if (((x + start_x) & 7) == 4 && x + 4 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<2>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 4, 4);
            x += 4;
          }
          while (x + 8 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<2>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 4, 4);
            CopySwapDwords<Swap>(source + off + 32, out + size_t(x + 4) * 4, 4);
            x += 8;
          }
          if (x + 4 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<2>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 4, 4);
            x += 4;
          }
        } else if constexpr (BlockSize == 2) {
          while (x < blocks_x && ((x + start_x) & 7) != 0) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<1>(int32_t(x + start_x), row));
            CopySwapBlock<2, Swap>(source, off, out + size_t(x) * 2);
            ++x;
          }
          while (x + 8 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<1>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 2, 4);
            x += 8;
          }
        } else if constexpr (BlockSize == 8) {
          if (x < blocks_x && ((x + start_x) & 1) != 0) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<3>(int32_t(x + start_x), row));
            CopySwapBlock<8, Swap>(source, off, out);
            ++x;
          }
          if (((x + start_x) & 3) == 2 && x + 2 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<3>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 8, 4);
            x += 2;
          }
          while (x + 4 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<3>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 8, 4);
            CopySwapDwords<Swap>(source + off + 32, out + size_t(x + 2) * 8, 4);
            x += 4;
          }
          if (x + 2 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<3>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 8, 4);
            x += 2;
          }
        } else if constexpr (BlockSize == 1) {
          while (x < blocks_x && ((x + start_x) & 7) != 0) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<0>(int32_t(x + start_x), row));
            CopySwapBlock<1, Swap>(source, off, out + x);
            ++x;
          }
          while (x + 8 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<0>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + x, 2);
            x += 8;
          }
        } else if constexpr (BlockSize == 16) {
          if (x < blocks_x && ((x + start_x) & 1) != 0) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<4>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out, 4);
            ++x;
          }
          while (x + 2 <= blocks_x) {
            const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<4>(int32_t(x + start_x), row));
            CopySwapDwords<Swap>(source + off, out + size_t(x) * 16, 4);
            CopySwapDwords<Swap>(source + off + 32, out + size_t(x + 1) * 16, 4);
            x += 2;
          }
        }
        while (x < blocks_x) {
          const size_t off = slice_base + size_t(GetTiledOffset2DFromRowT<kBpbLog2>(int32_t(x + start_x), row));
          CopySwapBlock<BlockSize, Swap>(source, off, out + size_t(x) * BlockSize);
          ++x;
        }
      } else {
        for (uint32_t x = 0; x < blocks_x; ++x) {
          const size_t offset =
              slice_base +
              size_t(GetTiledOffset2DFromRowT<kBpbLog2>(int32_t(x + start_x), row));
          if (offset <= source_size && BlockSize <= source_size - offset) {
            CopySwapBlock<BlockSize, Swap>(source, offset, out + size_t(x) * BlockSize);
          }
        }
      }
      if (row_scratch) {
        std::memcpy(row_dst, row_scratch, row_bytes);
      }
    }
  }
}

template <uint32_t BlockSize, size_t Swap>
inline void DecodeLinear2DCopyLevel(const uint8_t* __restrict source, uint8_t* __restrict dst,
                                    uint32_t slices, uint32_t blocks_x, uint32_t blocks_y,
                                    uint32_t start_x, uint32_t start_y, uint32_t row_pitch_bytes,
                                    size_t slice_stride_bytes, size_t dst_slice_bytes,
                                    uint8_t* row_scratch = nullptr) {
  const size_t row_copy_bytes = size_t(blocks_x) * BlockSize;
  for (uint32_t slice = 0; slice < slices; ++slice) {
    const size_t slice_base = size_t(slice) * slice_stride_bytes;
    uint8_t* slice_dst = dst + size_t(slice) * dst_slice_bytes;
    for (uint32_t y = 0; y < blocks_y; ++y) {
      const size_t row_src_offset =
          slice_base + size_t(y + start_y) * row_pitch_bytes + size_t(start_x) * BlockSize;
      uint8_t* row_dst = slice_dst + size_t(y) * row_copy_bytes;
      if constexpr (Swap == 0) {
        std::memcpy(row_dst, source + row_src_offset, row_copy_bytes);
      } else {
        uint8_t* out = row_scratch ? row_scratch : row_dst;
        if constexpr (BlockSize >= 4) {
          CopySwapDwords<Swap>(source + row_src_offset, out, row_copy_bytes / 4);
        } else if ((row_src_offset & 3) == 0) {
          const size_t dwords = row_copy_bytes / 4;
          CopySwapDwords<Swap>(source + row_src_offset, out, dwords);
          const size_t tail_byte = dwords * 4;
          for (size_t b = tail_byte; b < row_copy_bytes; b += BlockSize) {
            CopySwapBlock<BlockSize, Swap>(source, row_src_offset + b, out + b);
          }
        } else {
          for (uint32_t x = 0; x < blocks_x; ++x) {
            CopySwapBlock<BlockSize, Swap>(source, row_src_offset + size_t(x) * BlockSize,
                                           out + size_t(x) * BlockSize);
          }
        }
        if (row_scratch) {
          std::memcpy(row_dst, row_scratch, row_copy_bytes);
        }
      }
    }
  }
}

inline bool FindNextInvalidPageRun(const std::vector<uint64_t>& valid_words, uint32_t& cursor,
                                   uint32_t end_page, uint32_t& run_first_page,
                                   uint32_t& run_page_count) {
  const uint32_t limit = std::min(end_page, uint32_t(valid_words.size() * 64));
  while (cursor < limit) {
    const uint32_t word = cursor >> 6;
    const uint32_t bit = cursor & 63;
    const uint64_t inv = ~valid_words[word] & (~uint64_t(0) << bit);
    if (!inv) {
      cursor = (word + 1) << 6;
      continue;
    }
    const uint32_t first = (word << 6) + uint32_t(std::countr_zero(inv));
    if (first >= limit) {
      cursor = limit;
      return false;
    }
    run_first_page = first;
    uint32_t end = first;
    while (end < limit) {
      const uint32_t w = end >> 6;
      const uint32_t b = end & 63;
      const uint64_t valid_mask = valid_words[w] >> b;
      if (!valid_mask) {
        end = std::min(limit, (w + 1) << 6);
      } else {
        end = std::min(limit, end + uint32_t(std::countr_zero(valid_mask)));
        break;
      }
    }
    run_page_count = end - first;
    cursor = end;
    return true;
  }
  return false;
}

template <typename BlockMaskFn>
inline std::pair<uint32_t, uint32_t> WidenInvalidationPageRangeWithMask(
    BlockMaskFn&& block_mask, uint32_t max_blocks, uint32_t address, uint32_t length) {
  if (!length || address >= kPhysicalMemorySize) {
    return {0, 0};
  }
  const auto [initial_first, initial_count] = GetPhysicalPageRange(address, length);
  if (!initial_count) {
    return {0, 0};
  }
  uint32_t page_first = initial_first;
  uint32_t page_last = initial_first + initial_count - 1;
  const uint32_t block_first = page_first >> 6;
  const uint32_t block_last = page_last >> 6;
  if (block_first < max_blocks && (page_first & 63) != 0) {
    const uint64_t before =
        block_mask(block_first) & ((uint64_t(1) << (page_first & 63)) - 1);
    page_first = (page_first & ~uint32_t(63)) + (64u - uint32_t(std::countl_zero(before)));
  }
  if (block_last < max_blocks && (page_last & 63) != 63) {
    const uint64_t after =
        block_mask(block_last) & ~((uint64_t(1) << ((page_last & 63) + 1)) - 1);
    page_last =
        (page_last & ~uint32_t(63)) + (std::max(uint32_t(std::countr_zero(after)), 1u) - 1u);
  }
  const uint32_t widened_start = page_first << kSharedVertexPageShift;
  const uint32_t widened_length = std::min((page_last - page_first + 1) << kSharedVertexPageShift,
                                           kPhysicalMemorySize - widened_start);
  return {widened_start, widened_length};
}

inline std::pair<uint32_t, uint32_t> WidenInvalidationPageRange(
    const std::vector<uint64_t>& protected_pages, uint32_t address, uint32_t length) {
  return WidenInvalidationPageRangeWithMask(
      [&protected_pages](uint32_t block) { return protected_pages[block]; },
      uint32_t(protected_pages.size()), address, length);
}

// Host sampler state derived from a guest sampler binding. Also the device-lifetime
// cache key, so it must order identically to how it compares.
struct DrawSampler {
  VkFilter min_filter = VK_FILTER_NEAREST;
  VkFilter mag_filter = VK_FILTER_NEAREST;
  VkSamplerMipmapMode mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  float max_lod = 0.0f;
  VkSamplerAddressMode address_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSamplerAddressMode address_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSamplerAddressMode address_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  // Guest border color, only meaningful with a border address mode.
  xenos::BorderColor border_color = xenos::BorderColor::k_ABGR_Black;
  float min_lod = 0.0f;
  // 0 when anisotropic filtering is disabled.
  float max_anisotropy = 0.0f;
  auto operator<=>(const DrawSampler&) const = default;
};

// Hashes only the initialized fields so equal sampler states always map to the
// same bucket regardless of any padding in the struct.
struct DrawSamplerHash {
  size_t operator()(const DrawSampler& key) const noexcept {
    uint64_t h = uint64_t(key.min_filter) | (uint64_t(key.mag_filter) << 4) |
                 (uint64_t(key.mip_mode) << 8) | (uint64_t(key.address_u) << 12) |
                 (uint64_t(key.address_v) << 16) | (uint64_t(key.address_w) << 20) |
                 (uint64_t(key.border_color) << 24);
    h = (h ^ uint64_t(std::bit_cast<uint32_t>(key.max_lod)) * 0x9E3779B97F4A7C15ull) *
        0xBF58476D1CE4E5B9ull;
    h ^= (uint64_t(std::bit_cast<uint32_t>(key.min_lod)) |
          (uint64_t(std::bit_cast<uint32_t>(key.max_anisotropy)) << 32)) *
         0x94D049BB133111EBull;
    return size_t(h ^ (h >> 32));
  }
};

// Everything that determines the uploaded host texel data of a guest texture.
// The view swizzle is deliberately excluded because a single image is sampled
// with several component mappings.
struct GuestTextureKey {
  uint32_t base_page = 0;
  uint32_t mip_page = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t array_layers = 1;
  xenos::TextureFormat format = xenos::TextureFormat::k_8_8_8_8;
  xenos::DataDimension dimension = xenos::DataDimension::k2DOrStacked;
  bool tiled = false;
  bool packed_mips = false;
  xenos::Endian endianness = xenos::Endian::kNone;
  uint32_t pitch = 0;
  uint32_t mip_max = 0;
  // 3D texture depth (1 otherwise).
  uint32_t depth = 1;
  // Only set for formats whose CPU unpacking depends on the sign.
  bool is_signed = false;
  // Created with VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT for an SNORM view, only
  // when signed components are fetched, so other images keep compression.
  bool signed_view = false;
  auto operator<=>(const GuestTextureKey&) const = default;
};

// Hashes only the initialized fields, so the padding between the bool flags and
// the following enum can never make two identical keys hash differently.
struct GuestTextureKeyHash {
  size_t operator()(const GuestTextureKey& key) const noexcept {
    uint64_t h = (uint64_t(key.base_page) | (uint64_t(key.mip_page) << 32)) * 0x9E3779B97F4A7C15ull;
    h ^= (uint64_t(key.width) | (uint64_t(key.height) << 16) |
          (uint64_t(key.array_layers) << 32) | (uint64_t(key.depth) << 48)) *
         0xBF58476D1CE4E5B9ull;
    const uint64_t flags = uint64_t(key.format) | (uint64_t(key.dimension) << 8) |
                           (uint64_t(key.tiled) << 12) | (uint64_t(key.packed_mips) << 13) |
                           (uint64_t(key.endianness) << 14) | (uint64_t(key.mip_max) << 18) |
                           (uint64_t(key.is_signed) << 24) | (uint64_t(key.signed_view) << 25) |
                           (uint64_t(key.pitch) << 32);
    h ^= flags * 0x94D049BB133111EBull;
    return size_t(h ^ (h >> 32));
  }
};

// Whether two [start, start + length) physical memory ranges share any byte.
// Empty ranges never overlap. Arithmetic is done in 64 bits so a range ending at
// the top of the 512 MB guest address space does not wrap.
constexpr bool RangesOverlap(uint32_t a_start, uint32_t a_length, uint32_t b_start,
                             uint32_t b_length) {
  if (!a_length || !b_length) {
    return false;
  }
  const uint64_t a_end = uint64_t(a_start) + a_length;
  const uint64_t b_end = uint64_t(b_start) + b_length;
  return a_start < b_end && b_start < a_end;
}

using PhysicalRange = std::pair<uint32_t, uint32_t>;  // start, length

// Input ranges are bounded to physical memory by the invalidation callback.
inline void CoalescePhysicalRanges(std::vector<PhysicalRange>& ranges) {
  std::sort(ranges.begin(), ranges.end());
  size_t count = 0;
  for (const auto [start, length] : ranges) {
    if (!length) continue;
    if (count && start <= ranges[count - 1].first + ranges[count - 1].second) {
      auto& [previous_start, previous_length] = ranges[count - 1];
      previous_length = std::max(previous_length, start + length - previous_start);
    } else {
      ranges[count++] = {start, length};
    }
  }
  ranges.resize(count);
}

// The ranges must be sorted and coalesced so their endpoints are ordered too.
inline bool OverlapsPhysicalRanges(std::span<const PhysicalRange> ranges, uint32_t start,
                                    uint32_t length) {
  const auto entry = std::lower_bound(
      ranges.begin(), ranges.end(), start, [](const PhysicalRange& range, uint32_t address) {
        return uint64_t(range.first) + range.second <= address;
      });
  return entry != ranges.end() && RangesOverlap(entry->first, entry->second, start, length);
}

constexpr bool IsPhysicalRange(uint32_t address, size_t size) {
  return address < kPhysicalMemorySize && size && size <= kPhysicalMemorySize - address;
}

// Number of components a guest texture format actually samples. Single-channel
// formats replicate R into all channels, two-channel ones replicate G into B
// and A, matching the host format swizzle of the Xenos Vulkan backend.
constexpr uint32_t TextureChannelCount(xenos::TextureFormat format) {
  switch (format) {
    case xenos::TextureFormat::k_8:
    case xenos::TextureFormat::k_8_A:
    case xenos::TextureFormat::k_8_B:
    case xenos::TextureFormat::k_32_FLOAT:
    case xenos::TextureFormat::k_24_8:
    case xenos::TextureFormat::k_24_8_FLOAT:
    case xenos::TextureFormat::k_16:
    case xenos::TextureFormat::k_16_EXPAND:
    case xenos::TextureFormat::k_16_FLOAT:
    case xenos::TextureFormat::k_DXT3A:
    case xenos::TextureFormat::k_DXT5A:
      return 1;
    case xenos::TextureFormat::k_8_8:
    case xenos::TextureFormat::k_16_16:
    case xenos::TextureFormat::k_16_16_EXPAND:
    case xenos::TextureFormat::k_16_16_FLOAT:
    case xenos::TextureFormat::k_32_32_FLOAT:
    case xenos::TextureFormat::k_DXN:
    case xenos::TextureFormat::k_CTX1:
      return 2;
    case xenos::TextureFormat::k_5_6_5:
    case xenos::TextureFormat::k_10_11_11:
    case xenos::TextureFormat::k_10_11_11_AS_16_16_16_16:
    case xenos::TextureFormat::k_11_11_10:
    case xenos::TextureFormat::k_11_11_10_AS_16_16_16_16:
      return 3;
    default:
      return 4;
  }
}

// Host 12-bit texture swizzle word matching SpirvShaderTranslator's
// SystemConstants::texture_swizzles encoding. One- and two-channel formats
// replicate the stored components into the unused ones, matching the Xenos
// Vulkan host format swizzle. 32bpp 3/4-channel resolves copied with
// copy_dest_swap store red and blue swapped, which the guest fetch swizzle
// (0x60A BGRA or 0xA0A BGR1) normally undoes - swap_red_blue applies that swap
// here for cached resolves that keep the EDRAM RGBA order.
constexpr uint32_t MakeHostTextureSwizzle(uint32_t fetch_swizzle, uint32_t channel_count,
                                          bool swap_red_blue) {
  const uint32_t first = channel_count >= 3 && swap_red_blue ? 2u : 0u;
  const uint32_t third = channel_count >= 3 && swap_red_blue ? 0u : 2u;
  const std::array<uint32_t, 8> swizzles{
      first,
      channel_count == 1 ? 0u : 1u,
      channel_count == 1 ? 0u : channel_count == 2 ? 1u : third,
      channel_count == 1 ? 0u : channel_count == 2 ? 1u : channel_count == 3 ? third : 3u,
      uint32_t(xenos::XE_GPU_TEXTURE_SWIZZLE_0),
      uint32_t(xenos::XE_GPU_TEXTURE_SWIZZLE_1),
      uint32_t(xenos::XE_GPU_TEXTURE_SWIZZLE_0),
      uint32_t(xenos::XE_GPU_TEXTURE_SWIZZLE_0)};
  return swizzles[fetch_swizzle & 7] | (swizzles[(fetch_swizzle >> 3) & 7] << 3) |
         (swizzles[(fetch_swizzle >> 6) & 7] << 6) |
         (swizzles[(fetch_swizzle >> 9) & 7] << 9);
}

// Component mapping for sampling a guest texture. When imageViewFormatSwizzle
// is unavailable, returns all-IDENTITY because SpirvShaderTranslator performs
// the swizzle in the shader from SystemConstants::texture_swizzles.
constexpr VkComponentMapping MakeTextureComponentMappingFromHostSwizzle(
    uint32_t host_swizzle, bool image_view_format_swizzle = true) {
  if (!image_view_format_swizzle) {
    return {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
  }
  constexpr std::array<VkComponentSwizzle, 8> map{
      VK_COMPONENT_SWIZZLE_R,    VK_COMPONENT_SWIZZLE_G,    VK_COMPONENT_SWIZZLE_B,
      VK_COMPONENT_SWIZZLE_A,    VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE,
      VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO};
  return {map[host_swizzle & 7], map[(host_swizzle >> 3) & 7], map[(host_swizzle >> 6) & 7],
          map[(host_swizzle >> 9) & 7]};
}

constexpr VkComponentMapping MakeTextureComponentMapping(uint32_t fetch_swizzle,
                                                        uint32_t channel_count,
                                                        bool swap_red_blue,
                                                        bool image_view_format_swizzle = true) {
  return MakeTextureComponentMappingFromHostSwizzle(
      MakeHostTextureSwizzle(fetch_swizzle, channel_count, swap_red_blue),
      image_view_format_swizzle);
}

// Packs a 12-bit host swizzle into SpirvShaderTranslator::SystemConstants::texture_swizzles.
constexpr void PackSystemTextureSwizzle(uint32_t (&texture_swizzles)[16], uint32_t slot,
                                        uint32_t host_swizzle) {
  if (slot >= 32) {
    return;
  }
  uint32_t& word = texture_swizzles[slot >> 1];
  const uint32_t shift = 12 * (slot & 1);
  const uint32_t mask = uint32_t(0xFFF) << shift;
  word = (word & ~mask) | ((host_swizzle & 0xFFF) << shift);
}

// Packs a component mapping into one 32-bit cache key. Every swizzle value the
// translator produces fits in a byte, so distinct mappings never collide.
constexpr uint32_t PackTextureComponentMapping(VkComponentMapping mapping) {
  return (uint32_t(mapping.r) & 0xFF) | ((uint32_t(mapping.g) & 0xFF) << 8) |
         ((uint32_t(mapping.b) & 0xFF) << 16) | ((uint32_t(mapping.a) & 0xFF) << 24);
}

// How guest texels become host texels on the CPU before the upload.
enum class TextureUnpack : uint8_t {
  // The host format stores the guest blocks as they are.
  kCopy,
  // 24-bit depth in the high bits of each texel, to R32_SFLOAT.
  kDepthUnorm24,
  kDepthFloat24,
  kRgba4ToRgba8,
  kRgb5a1ToRgba8,
  // Unsigned or signed (by the sign variant) normalized 16-bit RGBA, alpha 1.
  kR11G11B10ToRgba16,
  kR10G11B11ToRgba16,
  // 4x4 blocks decoded to texels.
  kDxt3aToR8,
  kCtx1ToRg8,
};

struct TextureFormatInfo {
  // Host format sampled for unsigned components; VK_FORMAT_UNDEFINED when the
  // guest format is unsupported.
  VkFormat format = VK_FORMAT_UNDEFINED;
  // Host format sampled for signed components: the same image viewed with a
  // compatible format (VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT), or a separately
  // unpacked image when the unpacking depends on the sign. VK_FORMAT_UNDEFINED
  // when signed fetches can't be represented.
  VkFormat signed_format = VK_FORMAT_UNDEFINED;
  // Texels along each side of a guest block, and its size in bytes.
  uint32_t block_width = 1;
  uint32_t block_size = 0;
  TextureUnpack unpack = TextureUnpack::kCopy;

  // Whether signed components need a differently unpacked image.
  constexpr bool unpack_depends_on_sign() const {
    return unpack == TextureUnpack::kR11G11B10ToRgba16 ||
           unpack == TextureUnpack::kR10G11B11ToRgba16;
  }
  // Whether signed components need an image view with another format.
  constexpr bool needs_signed_view() const {
    return signed_format != VK_FORMAT_UNDEFINED && signed_format != format &&
           !unpack_depends_on_sign();
  }
  // Host bytes per texel, or per block for block-compressed host formats.
  constexpr uint32_t host_block_size() const {
    switch (unpack) {
      case TextureUnpack::kDepthUnorm24:
      case TextureUnpack::kDepthFloat24:
      case TextureUnpack::kRgba4ToRgba8:
      case TextureUnpack::kRgb5a1ToRgba8:
        return 4;
      case TextureUnpack::kR11G11B10ToRgba16:
      case TextureUnpack::kR10G11B11ToRgba16:
        return 8;
      case TextureUnpack::kDxt3aToR8:
        return 1;
      case TextureUnpack::kCtx1ToRg8:
        return 2;
      default:
        return block_size;
    }
  }
  // Whether the host image is uncompressed although the guest blocks are 4x4.
  constexpr bool decompresses_blocks() const {
    return unpack == TextureUnpack::kDxt3aToR8 || unpack == TextureUnpack::kCtx1ToRg8;
  }
};

// Host format and CPU conversion of a guest texture format, matching the host
// formats of the Xenos Vulkan backend.
constexpr TextureFormatInfo GetTextureFormatInfo(xenos::TextureFormat format) {
  using F = xenos::TextureFormat;
  using U = TextureUnpack;
  switch (format) {
    case F::k_8:
    case F::k_8_A:
    case F::k_8_B:
      return {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, 1, 1};
    case F::k_1_5_5_5:
      return {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, 1, 2, U::kRgb5a1ToRgba8};
    case F::k_5_6_5:
      return {VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_UNDEFINED, 1, 2};
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_AS_16_16_16_16:
      return {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, 1, 4};
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_16_16_16_16:
      return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_UNDEFINED, 1, 4};
    case F::k_8_8:
      return {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, 1, 2};
    case F::k_4_4_4_4:
      return {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, 1, 2, U::kRgba4ToRgba8};
    case F::k_10_11_11:
    case F::k_10_11_11_AS_16_16_16_16:
      return {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, 1, 4,
              U::kR11G11B10ToRgba16};
    case F::k_11_11_10:
    case F::k_11_11_10_AS_16_16_16_16:
      return {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, 1, 4,
              U::kR10G11B11ToRgba16};
    case F::k_DXT1:
    case F::k_DXT1_AS_16_16_16_16:
      return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_UNDEFINED, 4, 8};
    case F::k_DXT2_3:
    case F::k_DXT2_3_AS_16_16_16_16:
      return {VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_UNDEFINED, 4, 16};
    case F::k_DXT4_5:
    case F::k_DXT4_5_AS_16_16_16_16:
      return {VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_UNDEFINED, 4, 16};
    case F::k_24_8:
      return {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT, 1, 4, U::kDepthUnorm24};
    case F::k_24_8_FLOAT:
      return {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT, 1, 4, U::kDepthFloat24};
    case F::k_16:
      return {VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, 1, 2};
    case F::k_16_16:
      return {VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, 1, 4};
    case F::k_16_16_16_16:
      return {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, 1, 8};
    case F::k_16_EXPAND:
    case F::k_16_FLOAT:
      return {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_SFLOAT, 1, 2};
    case F::k_16_16_EXPAND:
    case F::k_16_16_FLOAT:
      return {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, 1, 4};
    case F::k_16_16_16_16_EXPAND:
    case F::k_16_16_16_16_FLOAT:
      return {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 8};
    case F::k_32_FLOAT:
      return {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT, 1, 4};
    case F::k_32_32_FLOAT:
      return {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, 1, 8};
    case F::k_32_32_32_32_FLOAT:
      return {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 1, 16};
    case F::k_DXN:
      return {VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_SNORM_BLOCK, 4, 16};
    case F::k_DXT5A:
      return {VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_SNORM_BLOCK, 4, 8};
    case F::k_DXT3A:
      return {VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, 4, 8, U::kDxt3aToR8};
    case F::k_CTX1:
      return {VK_FORMAT_R8G8_UNORM, VK_FORMAT_UNDEFINED, 4, 8, U::kCtx1ToRg8};
    default:
      return {};
  }
}

// Replicates the high bits of an n-bit unsigned normalized value into 16 bits.
constexpr uint16_t ExpandUnorm16(uint32_t value, uint32_t bits) {
  value &= (1u << bits) - 1;
  uint32_t result = value << (16 - bits);
  for (uint32_t shift = bits; shift < 16; shift += bits) {
    result |= value << (16 - bits) >> shift;
  }
  return uint16_t(result);
}

// An n-bit two's complement normalized value as 16-bit SNORM, with both of
// the most negative values -1.0 (Xenos XeSNorm10To16/XeSNorm11To16).
constexpr uint16_t ExpandSnorm16(uint32_t value, uint32_t bits) {
  value &= (1u << bits) - 1;
  const uint32_t sign = value >> (bits - 1);
  if (value == 1u << (bits - 1)) {
    ++value;
  }
  uint32_t magnitude = ((value ^ (sign ? (1u << bits) - 1 : 0u)) + sign) & ((1u << bits) - 1);
  // The (bits - 1)-bit magnitude replicated into 15 bits like UNORM.
  magnitude = ExpandUnorm16(magnitude, bits - 1) >> 1;
  return uint16_t(((magnitude ^ (sign ? 0xFFFFu : 0u)) + sign) & 0xFFFF);
}

// k_4_4_4_4 (R in the low bits) to R8G8B8A8.
constexpr uint32_t UnpackRgba4ToRgba8(uint16_t texel) {
  uint32_t result = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    result |= ((uint32_t(texel) >> (i * 4)) & 0xF) * 0x11 << (i * 8);
  }
  return result;
}

// k_1_5_5_5 (R in the low bits, A in the top bit) to R8G8B8A8.
constexpr uint32_t UnpackRgb5a1ToRgba8(uint16_t texel) {
  uint32_t result = (texel >> 15) ? 0xFF000000u : 0u;
  for (uint32_t i = 0; i < 3; ++i) {
    result |= uint32_t(ExpandUnorm16(texel >> (i * 5), 5) >> 8) << (i * 8);
  }
  return result;
}

// k_10_11_11 (R11 G11 B10 from the low bits) or k_11_11_10 (R10 G11 B11) to
// four 16-bit normalized components with alpha 1, as the Xenos load shaders.
constexpr std::array<uint16_t, 4> UnpackR11G11B10ToRgba16(uint32_t texel, bool r10g11b11,
                                                         bool is_signed) {
  const std::array<uint32_t, 3> bits =
      r10g11b11 ? std::array<uint32_t, 3>{10, 11, 11} : std::array<uint32_t, 3>{11, 11, 10};
  std::array<uint16_t, 4> result{0, 0, 0, uint16_t(is_signed ? 0x7FFF : 0xFFFF)};
  uint32_t shift = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    const uint32_t value = texel >> shift;
    result[i] = is_signed ? ExpandSnorm16(value, bits[i]) : ExpandUnorm16(value, bits[i]);
    shift += bits[i];
  }
  return result;
}

// Decodes a k_DXT3A block (4 bits per texel, row by row) to 16 R8 texels.
constexpr std::array<uint8_t, 16> DecodeDxt3aBlock(uint64_t block) {
  std::array<uint8_t, 16> texels{};
  for (uint32_t i = 0; i < 16; ++i) {
    texels[i] = uint8_t(((block >> (i * 4)) & 0xF) * 0x11);
  }
  return texels;
}

// Decodes a k_CTX1 block (bytes X0 Y0 X1 Y1, then 2 bits per texel selecting
// endpoint 0, endpoint 1 or the 2:1 and 1:2 blends) to 16 R8G8 texels.
constexpr std::array<uint16_t, 16> DecodeCtx1Block(uint64_t block) {
  const std::array<uint32_t, 2> end0{uint32_t(block & 0xFF), uint32_t((block >> 8) & 0xFF)};
  const std::array<uint32_t, 2> end1{uint32_t((block >> 16) & 0xFF),
                                     uint32_t((block >> 24) & 0xFF)};
  // Weight of endpoint 1 (in thirds) for each index.
  constexpr std::array<uint32_t, 4> weights{0, 3, 1, 2};
  std::array<uint16_t, 16> texels{};
  for (uint32_t i = 0; i < 16; ++i) {
    const uint32_t weight = weights[(block >> (32 + i * 2)) & 3];
    uint32_t texel = 0;
    for (uint32_t c = 0; c < 2; ++c) {
      texel |= ((end0[c] * (3 - weight) + end1[c] * weight) / 3) << (c * 8);
    }
    texels[i] = uint16_t(texel);
  }
  return texels;
}

// A fetch constant that cannot produce texels - not a texture fetch, no base
// or mip level, or a resource dimension the binding doesn't sample - reads as
// zero on the guest, so bind a zero-cleared fallback view rather than rejecting
// the draw. Valid textures that merely use unsupported formats or unreadable
// memory are still rejected so real errors stay visible.
constexpr bool ShouldUseFallbackTexture(bool fetch_is_texture, bool has_levels,
                                        bool dimension_matches) {
  return !fetch_is_texture || !has_levels || !dimension_matches;
}

// Whether a binding of the shader's fetch dimension samples a texture of the
// guest data dimension: 1D and stacked 2D textures are bound as 2D arrays,
// and a k3DOrStacked fetch has separate 2D array and 3D bindings.
constexpr bool TextureBindingMatchesDimension(xenos::FetchOpDimension binding,
                                              xenos::DataDimension data) {
  switch (binding) {
    case xenos::FetchOpDimension::kCube:
      return data == xenos::DataDimension::kCube;
    case xenos::FetchOpDimension::k3DOrStacked:
      return data == xenos::DataDimension::k3D;
    default:
      return data == xenos::DataDimension::k1D || data == xenos::DataDimension::k2DOrStacked;
  }
}

// Host address mode for a guest clamp mode: no Vulkan equivalent of the
// halfway modes, and mirrored clamps only to the edge (if supported).
constexpr VkSamplerAddressMode GetSamplerAddressMode(xenos::ClampMode mode,
                                                     bool mirror_clamp_to_edge) {
  switch (mode) {
    case xenos::ClampMode::kRepeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case xenos::ClampMode::kMirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case xenos::ClampMode::kMirrorClampToEdge:
    case xenos::ClampMode::kMirrorClampToHalfway:
    case xenos::ClampMode::kMirrorClampToBorder:
      return mirror_clamp_to_edge ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
                                  : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case xenos::ClampMode::kClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

// Host anisotropy for a guest filter (0 for disabled): anisotropic_override
// forces it on trilinear/bilinear mipmapped textures, as the Xenos backend
// does, and the device limit caps it.
constexpr float GetSamplerAnisotropy(xenos::AnisoFilter filter, int32_t anisotropic_override,
                                     bool override_eligible, float device_max_anisotropy) {
  if (anisotropic_override > -1 && anisotropic_override < 6 && override_eligible) {
    filter = xenos::AnisoFilter(anisotropic_override);
  }
  if (filter == xenos::AnisoFilter::kDisabled || filter > xenos::AnisoFilter::kMax_16_1 ||
      device_max_anisotropy < 1.0f) {
    return 0.0f;
  }
  return std::min(float(1u << (uint32_t(filter) - 1)), std::min(device_max_anisotropy, 16.0f));
}

// Linear filtering and anisotropy forced off for a host format that can't be
// linearly filtered, and anisotropy forcing linear filtering otherwise.
constexpr DrawSampler NormalizeDrawSampler(DrawSampler sampler, bool linear_filterable) {
  if (!linear_filterable) {
    sampler.min_filter = sampler.mag_filter = VK_FILTER_NEAREST;
    sampler.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.max_anisotropy = 0.0f;
  } else if (sampler.max_anisotropy != 0.0f) {
    sampler.min_filter = sampler.mag_filter = VK_FILTER_LINEAR;
    sampler.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
  const bool border = sampler.address_u == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                      sampler.address_v == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                      sampler.address_w == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  if (!border) {
    // The color is unused, so samplers differing only in it are the same.
    sampler.border_color = xenos::BorderColor::k_ABGR_Black;
  }
  return sampler;
}

// Resolve offsets and sizes are 8x8-aligned, so a cached resolve can be up to
// one alignment unit wider and taller than the texture that samples it. Such a
// resolve can be cropped from its top-left corner instead of being rejected.
constexpr bool CanCropAlignedResolve(uint32_t cached_width, uint32_t cached_height,
                                     uint32_t requested_width, uint32_t requested_height,
                                     uint32_t scale_x = 1, uint32_t scale_y = 1,
                                     uint32_t alignment = xenos::kResolveAlignmentPixels) {
  return cached_width >= requested_width && cached_height >= requested_height &&
         cached_width - requested_width < alignment * scale_x &&
         cached_height - requested_height < alignment * scale_y;
}

// Cube textures are hosted as a cube-compatible image with all six faces as
// array layers of a single level; everything else is a single layer.
constexpr uint32_t kTextureCubeArrayLayers = 6;

// Cache key for a cube texture assembled from six GPU-resolved face images.
// The view swizzle is excluded because one cached cube image can be sampled
// with different component mappings.
struct ResolvedCubeKey {
  uint32_t address = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  auto operator<=>(const ResolvedCubeKey&) const = default;
};

struct ResolvedCubeKeyHash {
  size_t operator()(const ResolvedCubeKey& key) const noexcept {
    const std::array<uint32_t, 4> fields{key.address, key.width, key.height,
                                         uint32_t(key.format)};
    return size_t(XXH3_64bits(fields.data(), sizeof(fields)));
  }
};

// Each cube face is stored like its own 2D texture, `array_slice_stride_bytes`
// apart within the level. A face rendered on the GPU keeps its resolve at that
// same address, so a cube fetch has to look for the face images there.
constexpr uint32_t GetResolvedCubeFaceAddress(uint32_t base, uint32_t array_slice_stride_bytes,
                                              uint32_t face) {
  return base + array_slice_stride_bytes * face;
}

// A resolved cube face can stand in for guest memory only when its host image
// matches the texture format and covers the face within the resolve alignment,
// exactly like a 2D resolve crop.
constexpr bool CanUseResolvedCubeFace(bool format_matches, uint32_t face_width,
                                      uint32_t face_height, uint32_t texture_width,
                                      uint32_t texture_height, uint32_t scale_x = 1,
                                      uint32_t scale_y = 1) {
  return format_matches && CanCropAlignedResolve(face_width, face_height, texture_width,
                                                 texture_height, scale_x, scale_y);
}

// A cached GPU-resolved cube only needs to re-copy its six faces when it has
// never been uploaded or at least one face resolve version changed.
constexpr bool ShouldUploadResolvedCube(
    bool uploaded, const std::array<uint64_t, kTextureCubeArrayLayers>& cached_versions,
    const std::array<uint64_t, kTextureCubeArrayLayers>& current_versions) {
  return !uploaded || cached_versions != current_versions;
}

// Cache key for a cropped GPU-resolved 2D texture. The view swizzle is excluded
// because one cached crop image can be sampled with different component mappings.
struct ResolvedCropKey {
  uint32_t address = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t pack_flags = 0;
  auto operator<=>(const ResolvedCropKey&) const = default;
};

struct ResolvedCropKeyHash {
  size_t operator()(const ResolvedCropKey& key) const noexcept {
    const std::array<uint32_t, 5> fields{key.address, key.width, key.height,
                                         uint32_t(key.format), key.pack_flags};
    return size_t(XXH3_64bits(fields.data(), sizeof(fields)));
  }
};

// A cached cropped GPU resolve only needs to re-copy when it has never been
// uploaded or the source resolve version changed.
constexpr bool ShouldUploadResolvedCrop(bool uploaded, uint64_t cached_version,
                                        uint64_t current_version) {
  return !uploaded || cached_version != current_version;
}

constexpr uint32_t GetTextureArrayLayers(xenos::DataDimension dimension) {
  return dimension == xenos::DataDimension::kCube ? kTextureCubeArrayLayers : 1;
}

// Array slices of a level are stored in guest memory `array_slice_stride_bytes`
// apart within the level, so the last slice's data has to be readable as well.
constexpr size_t GetTextureLevelReadSize(uint32_t array_slice_stride_bytes,
                                         uint32_t array_slice_data_extent_bytes,
                                         uint32_t array_layers) {
  return size_t(array_slice_stride_bytes) * (array_layers - 1) + array_slice_data_extent_bytes;
}

struct IndexBinding {
  bool indexed = false;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkIndexType type = VK_INDEX_TYPE_UINT32;
  uint32_t count = 0;
};

inline void DecodeDmaIndices16(std::span<const uint8_t> source, xenos::Endian endian,
                               uint16_t* __restrict dst, size_t count) {
  const bool swap = endian == xenos::Endian::k8in16 || endian == xenos::Endian::k8in32;
  if (!swap) {
    std::memcpy(dst, source.data(), count * sizeof(uint16_t));
    return;
  }
  const uint8_t* __restrict src = source.data();
  for (size_t i = 0; i < count; ++i) {
    uint16_t v;
    std::memcpy(&v, src + i * 2, 2);
    v = uint16_t((v >> 8) | (v << 8));
    dst[i] = v;
  }
}

template <bool ReplaceRestart>
inline void DecodeDmaIndices32From16(std::span<const uint8_t> source, xenos::Endian endian,
                                     uint32_t* __restrict dst, size_t count, uint32_t reset_value) {
  const bool swap = endian == xenos::Endian::k8in16 || endian == xenos::Endian::k8in32;
  const uint8_t* __restrict src = source.data();
  for (size_t i = 0; i < count; ++i) {
    uint16_t v;
    std::memcpy(&v, src + i * 2, 2);
    if (swap) {
      v = uint16_t((v >> 8) | (v << 8));
    }
    const uint32_t u = uint32_t(v);
    if constexpr (ReplaceRestart) {
      dst[i] = (u == reset_value) ? UINT32_MAX : u;
    } else {
      dst[i] = u;
    }
  }
}

template <xenos::Endian EndianMode, bool ReplaceRestart>
inline void DecodeDmaIndices32From32Loop(const uint8_t* __restrict src, uint32_t* __restrict dst,
                                         size_t count, uint32_t reset_value) {
  for (size_t i = 0; i < count; ++i) {
    uint32_t v;
    std::memcpy(&v, src + i * 4, 4);
    v = xenos::GpuSwap(v, EndianMode) & xenos::kVertexIndexMask;
    if constexpr (ReplaceRestart) {
      dst[i] = (v == reset_value) ? UINT32_MAX : v;
    } else {
      dst[i] = v;
    }
  }
}

inline void DecodeDmaIndices32(std::span<const uint8_t> source, xenos::IndexFormat format,
                               xenos::Endian endian, uint32_t* __restrict dst, size_t count,
                               bool replace_restart = false, uint32_t reset_value = 0) {
  if (format == xenos::IndexFormat::kInt16) {
    if (replace_restart) {
      DecodeDmaIndices32From16<true>(source, endian, dst, count, reset_value);
    } else {
      DecodeDmaIndices32From16<false>(source, endian, dst, count, reset_value);
    }
    return;
  }
  const uint8_t* src = source.data();
  switch (endian) {
    case xenos::Endian::k8in16:
      replace_restart ? DecodeDmaIndices32From32Loop<xenos::Endian::k8in16, true>(
                            src, dst, count, reset_value)
                      : DecodeDmaIndices32From32Loop<xenos::Endian::k8in16, false>(
                            src, dst, count, reset_value);
      break;
    case xenos::Endian::k8in32:
      replace_restart ? DecodeDmaIndices32From32Loop<xenos::Endian::k8in32, true>(
                            src, dst, count, reset_value)
                      : DecodeDmaIndices32From32Loop<xenos::Endian::k8in32, false>(
                            src, dst, count, reset_value);
      break;
    case xenos::Endian::k16in32:
      replace_restart ? DecodeDmaIndices32From32Loop<xenos::Endian::k16in32, true>(
                            src, dst, count, reset_value)
                      : DecodeDmaIndices32From32Loop<xenos::Endian::k16in32, false>(
                            src, dst, count, reset_value);
      break;
    default:
      replace_restart ? DecodeDmaIndices32From32Loop<xenos::Endian::kNone, true>(
                            src, dst, count, reset_value)
                      : DecodeDmaIndices32From32Loop<xenos::Endian::kNone, false>(
                            src, dst, count, reset_value);
      break;
  }
}

inline bool DecodeDmaIndices(std::span<const uint8_t> source, xenos::IndexFormat format,
                             xenos::Endian endian, std::span<uint32_t> indices) {
  const bool wide = format == xenos::IndexFormat::kInt32;
  const size_t stride = wide ? sizeof(uint32_t) : sizeof(uint16_t);
  if (indices.size() > source.size() / stride) {
    return false;
  }
  DecodeDmaIndices32(source, format, endian, indices.data(), indices.size());
  return true;
}

// The translator expands each guest rectangle or point to four host vertices.
inline void FillQuadStripIndices(uint32_t primitive_count, std::vector<uint32_t>& indices) {
  indices.clear();
  indices.reserve(5 * primitive_count);
  for (uint32_t primitive = 0; primitive < primitive_count; ++primitive) {
    if (primitive) {
      indices.push_back(UINT32_MAX);
    }
    for (uint32_t corner = 0; corner < 4; ++corner) {
      indices.push_back(primitive * 4 + corner);
    }
  }
}

inline std::vector<uint32_t> QuadStripIndices(uint32_t primitive_count) {
  std::vector<uint32_t> indices;
  FillQuadStripIndices(primitive_count, indices);
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

// shortcut: blit unbiased, unswapped 1x UNORM copies until native packing supports other modes.
inline bool IsResolveFormatConvertSupported(const draw_util::ResolveInfo& copy,
                                            xenos::MsaaSamples samples,
                                            xenos::ColorRenderTargetFormat source_format,
                                            xenos::ColorFormat destination_format) {
  return samples == xenos::MsaaSamples::k1X &&
         source_format == xenos::ColorRenderTargetFormat::k_8_8_8_8 &&
         (destination_format == xenos::ColorFormat::k_8 ||
          destination_format == xenos::ColorFormat::k_8_8) &&
         copy.copy_dest_info.copy_dest_number ==
             xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction &&
         !copy.copy_dest_info.copy_dest_swap && !copy.copy_dest_info.copy_dest_exp_bias &&
         !copy.copy_dest_info.copy_dest_array;
}

// shortcut: cap attachments to full tile rows until native EDRAM supports split images.
constexpr uint32_t GetTargetHeight(uint32_t pitch, xenos::MsaaSamples samples,
                                   uint32_t tile_count, uint32_t max_height,
                                   bool is_64bpp = false) {
  const uint32_t tiles_per_row = xenos::GetSurfacePitchTiles(pitch, samples, is_64bpp);
  if (!tiles_per_row) {
    return 0;
  }
  const uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> (samples >= xenos::MsaaSamples::k2X);
  const uint32_t height = tile_count / tiles_per_row * tile_height;
  const uint32_t aligned_max = max_height / tile_height * tile_height;
  return height < aligned_max ? height : aligned_max;
}

// Xbox 360 D3D9 clears 1x- and 2x-MSAA EDRAM targets by issuing a 4x-MSAA
// depth-only draw at the same tile with half the horizontal pitch, so one host
// pixel of the alias covers scale_x by scale_y guest pixels.
struct FastClearScale {
  uint32_t x;
  uint32_t y;
};

constexpr FastClearScale GetMsaa4xAliasScale(xenos::MsaaSamples target_samples) {
  switch (target_samples) {
    case xenos::MsaaSamples::k1X:
      return {2, 2};
    case xenos::MsaaSamples::k2X:
      return {2, 1};
    default:
      return {1, 1};
  }
}

// Rescales a guest-space viewport, scissor and NDC scale/offset to cover the
// aliased target. When the scaled viewport would exceed the host limits, the
// extent is clamped and the NDC scale/offset are adjusted so the covered
// framebuffer coordinates stay identical.
inline void ScaleViewportAndScissor(draw_util::ViewportInfo& viewport, draw_util::Scissor& scissor,
                                    std::array<float, 3>& ndc_scale,
                                    std::array<float, 3>& ndc_offset, uint32_t scale_x,
                                    uint32_t scale_y, uint32_t max_viewport_w,
                                    uint32_t max_viewport_h) {
  scissor.offset[0] *= scale_x;
  scissor.offset[1] *= scale_y;
  scissor.extent[0] *= scale_x;
  scissor.extent[1] *= scale_y;
  viewport.xy_offset[0] *= scale_x;
  viewport.xy_offset[1] *= scale_y;
  for (uint32_t axis = 0; axis < 2; ++axis) {
    const uint32_t scale = axis ? scale_y : scale_x;
    const uint32_t max_viewport = axis ? max_viewport_h : max_viewport_w;
    const uint32_t orig_extent = viewport.xy_extent[axis];
    if (uint64_t(orig_extent) * scale <= max_viewport) {
      viewport.xy_extent[axis] = orig_extent * scale;
      continue;
    }
    viewport.xy_extent[axis] = max_viewport;
    const double k = (double(orig_extent) * double(scale)) / double(max_viewport);
    ndc_scale[axis] *= float(k);
    ndc_offset[axis] = float((double(ndc_offset[axis]) + 1.0) * k - 1.0);
  }
}

// Whether a 4x-MSAA depth-only draw is actually a D3D9 fast clear of an aliased
// 32bpp color target: it writes the whole depth range and stencil without a
// depth test and replaces or zeros the stencil, which is how the "color" is
// packed into the depth-stencil words.
constexpr bool IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat depth_format,
                                        reg::RB_DEPTHCONTROL depth_control,
                                        uint8_t stencil_write_mask) {
  return depth_format == xenos::DepthRenderTargetFormat::kD24S8 && depth_control.z_enable &&
         depth_control.z_write_enable &&
         depth_control.zfunc == xenos::CompareFunction::kAlways && depth_control.stencil_enable &&
         depth_control.stencilfunc == xenos::CompareFunction::kAlways &&
         (depth_control.stencilzpass == xenos::StencilOp::kReplace ||
          depth_control.stencilzpass == xenos::StencilOp::kZero) &&
         stencil_write_mask == 0xFF;
}

// D3D9 also clears 1x/2x color (including 64bpp) and depth/stencil targets
// with a 4x-MSAA kColorDepth rectangle-list draw using a constant-color
// pixel shader, clip disabled, blending off, and ALWAYS depth/stencil tests.
constexpr bool IsAliased4xColorDepthClear(bool use_color, bool has_pixel_shader,
                                          bool has_textures, xenos::PrimitiveType prim_type,
                                          xenos::MsaaSamples samples, bool clip_disable,
                                          bool any_blend_enabled, bool use_depth,
                                          reg::RB_DEPTHCONTROL depth_control) {
  return use_color && has_pixel_shader && !has_textures &&
         prim_type == xenos::PrimitiveType::kRectangleList &&
         samples == xenos::MsaaSamples::k4X && clip_disable && !any_blend_enabled &&
         (!use_depth ||
          ((!depth_control.z_enable || depth_control.zfunc == xenos::CompareFunction::kAlways) &&
           (!depth_control.stencil_enable ||
            depth_control.stencilfunc == xenos::CompareFunction::kAlways)));
}

// Index of the shader packing mode that unpacks a 32-bit EDRAM clear color into
// an aliased color target of this host format, matching the format indices used
// by ColorResolve.
constexpr std::optional<uint32_t> GetEdramClearColorFormatMode(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
      return 0;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return 1;
    case VK_FORMAT_R16G16_SFLOAT:
      return 2;
    case VK_FORMAT_R32_SFLOAT:
      return 3;
    default:
      return std::nullopt;
  }
}

// Picks the host depth-stencil format for a guest depth render target. D24S8
// maps to D24_UNORM_S8_UINT only when the device supports it as both a sampled
// image and a depth/stencil attachment for optimal-tiling images; some drivers
// (notably AMD RADV) expose no such support, so it falls back to
// D32_SFLOAT_S8_UINT. D24FS8 always uses D32_SFLOAT_S8_UINT.
constexpr VkFormat SelectDepthFormat(xenos::DepthRenderTargetFormat format,
                                     bool depth_unorm24_supported) {
  return format == xenos::DepthRenderTargetFormat::kD24S8 && depth_unorm24_supported
             ? VK_FORMAT_D24_UNORM_S8_UINT
             : VK_FORMAT_D32_SFLOAT_S8_UINT;
}

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

// The guest turns off clipping with PA_CL_CLIP_CNTL.CLIP_DISABLE and expects
// vertices outside the depth range to be clamped rather than discarded. Vulkan
// has no "no clipping" mode, so it is mapped to depth clamping where the device
// supports it. Without this, geometry such as the 4x-MSAA fast-clear quads,
// which clear with a near-zero Z, is clipped away entirely.
constexpr bool GetDepthClampEnable(bool device_depth_clamp, bool clip_disable) {
  return device_depth_clamp && clip_disable;
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

// Place a resolve in the cached image at its absolute guest texture coordinates.
inline std::optional<ResolveRectangle> GetCachedResolveRectangle(const draw_util::ResolveInfo& copy,
                                                                 uint32_t image_width,
                                                                 uint32_t image_height,
                                                                 uint32_t scale_x = 1,
                                                                 uint32_t scale_y = 1) {
  const uint32_t width = (uint32_t(copy.coordinate_info.width_div_8)
                          << xenos::kResolveAlignmentPixelsLog2) *
                         scale_x;
  const uint32_t height = (copy.height_div_8 << xenos::kResolveAlignmentPixelsLog2) * scale_y;
  const uint32_t x = copy.copy_dest_x * scale_x;
  const uint32_t y = copy.copy_dest_y * scale_y;
  if (!width || !height || x > image_width || y > image_height || width > image_width - x ||
      height > image_height - y) {
    return std::nullopt;
  }
  return ResolveRectangle{x, y, width, height};
}

inline std::optional<ResolveRectangle> GetResolveRectangle(const draw_util::ResolveInfo& copy,
                                                           EdramLayout layout, uint32_t width,
                                                           uint32_t height, bool depth = false,
                                                           uint32_t scale_x = 1,
                                                           uint32_t scale_y = 1) {
  const auto& target = depth ? copy.depth_edram_info : copy.color_edram_info;
  const uint32_t pitch = target.pitch_tiles;
  if (!pitch) {
    return std::nullopt;
  }
  const uint32_t base = depth ? copy.depth_original_base : copy.color_original_base;
  const uint32_t tiles = layout.Wrap(target.base_tiles - base);
  const bool is_64bpp =
      !depth && xenos::IsColorRenderTargetFormat64bpp(
                    xenos::ColorRenderTargetFormat(copy.color_edram_info.format));
  const uint32_t tile_width =
      xenos::kEdramTileWidthSamples >> ((target.msaa_samples >= xenos::MsaaSamples::k4X) + is_64bpp);
  const uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> (target.msaa_samples >= xenos::MsaaSamples::k2X);
  const ResolveRectangle rect{
      (tiles % pitch * tile_width + copy.coordinate_info.edram_offset_x_div_8 * 8) * scale_x,
      (tiles / pitch * tile_height + copy.coordinate_info.edram_offset_y_div_8 * 8) * scale_y,
      uint32_t(copy.coordinate_info.width_div_8) * 8 * scale_x,
      copy.height_div_8 * 8 * scale_y};
  if (rect.x > width || rect.y > height || !rect.width || !rect.height ||
      rect.width > width - rect.x || rect.height > height - rect.y) {
    return std::nullopt;
  }
  return rect;
}

// Destination image extent for a resolve: a single-pass resolve uses the
// resolve rectangle directly (preserving 8px alignment for crop checks),
// while a predicated-tiling pass whose destination surface is at least one
// 32px texture tile wider or taller than the first strip allocates the full
// destination surface so subsequent strips land inside the same image.
constexpr VkExtent2D GetResolveDestinationExtent(uint32_t guest_width, uint32_t guest_height,
                                                 uint32_t copy_dest_pitch,
                                                 uint32_t copy_dest_height, uint32_t scale_x = 1,
                                                 uint32_t scale_y = 1) {
  const uint32_t width = copy_dest_pitch >= guest_width + xenos::kTextureTileWidthHeight
                             ? copy_dest_pitch
                             : guest_width;
  const uint32_t height = copy_dest_height >= guest_height + xenos::kTextureTileWidthHeight
                              ? copy_dest_height
                              : guest_height;
  return {width * scale_x, height * scale_y};
}

constexpr uint32_t GetResolveBytesPerBlockLog2(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UNORM:
      return 0;
    case VK_FORMAT_R8G8_UNORM:
      return 1;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 3;
    default:
      return 2;
  }
}

// Exact inverse of texture_util::GetTiledOffset2D for 32x32 macro-tile origins
// (x and y multiples of 32). Returns false when delta_bytes is not a valid
// 32x32 macro-tile byte offset for pitch and bytes_per_block_log2.
constexpr bool TryGetTiledResolveOffset2D(uint32_t delta_bytes, uint32_t pitch,
                                          uint32_t bytes_per_block_log2, uint32_t& x_out,
                                          uint32_t& y_out) {
  if (!pitch || bytes_per_block_log2 > 4 || (delta_bytes & 0x8FFu) != 0) {
    return false;
  }
  const uint32_t aligned_pitch =
      (pitch + (xenos::kTextureTileWidthHeight - 1)) & ~(xenos::kTextureTileWidthHeight - 1);
  const uint32_t tiles_per_row = aligned_pitch >> xenos::kTextureTileWidthHeightLog2;
  if (!tiles_per_row) {
    return false;
  }
  const uint32_t macro = ((delta_bytes >> 3) & ~0x1FFu) | ((delta_bytes >> 2) & 0x1C0u);
  const uint32_t shift = bytes_per_block_log2 + 7;
  if ((macro & ((1u << shift) - 1)) != 0) {
    return false;
  }
  const uint32_t tile_index = macro >> shift;
  x_out = (tile_index % tiles_per_row) << xenos::kTextureTileWidthHeightLog2;
  y_out = (tile_index / tiles_per_row) << xenos::kTextureTileWidthHeightLog2;
  return true;
}

// Active swap output extent: clamp the scaled swap packet extent to the
// resolved frontbuffer image so 8px-padded resolves (e.g. 900p -> 904,
// 1080p -> 1088) crop cleanly and smaller guest frontbuffers still present.
constexpr VkExtent2D GetSwapOutputExtent(VkExtent2D requested,
                                         std::optional<VkExtent2D> resolved) {
  if (!requested.width || !requested.height) {
    return {};
  }
  if (!resolved || !resolved->width || !resolved->height) {
    return requested;
  }
  return {std::min(requested.width, resolved->width),
          std::min(requested.height, resolved->height)};
}

// Host Vulkan depth bias factors converted from Xenos absolute polygon offset
// registers, matching the Xenos Vulkan backend.
struct DepthBias {
  float constant_factor = 0.0f;
  float slope_factor = 0.0f;
};

constexpr DepthBias ScaleHostDepthBias(float slope, float offset,
                                       xenos::DepthRenderTargetFormat depth_format,
                                       uint32_t scale_x = 1, uint32_t scale_y = 1) {
  return {offset * (depth_format == xenos::DepthRenderTargetFormat::kD24S8
                        ? draw_util::kD3D10PolygonOffsetFactorUnorm24
                        : draw_util::kD3D10PolygonOffsetFactorFloat24),
          slope * xenos::kPolygonOffsetScaleSubpixelUnit *
              float(scale_x > scale_y ? scale_x : scale_y)};
}

inline DepthBias ComputeHostDepthBias(const RegisterFile& regs, bool primitive_polygonal,
                                      xenos::DepthRenderTargetFormat depth_format,
                                      uint32_t scale_x = 1, uint32_t scale_y = 1) {
  float slope = 0.0f;
  float offset = 0.0f;
  draw_util::GetPreferredFacePolygonOffset(regs, primitive_polygonal, slope, offset);
  return ScaleHostDepthBias(slope, offset, depth_format, scale_x, scale_y);
}

// User clip plane configuration for vertex shader translation, matching
// VulkanPipelineCache.
struct UserClipPlaneConfig {
  uint32_t enabled_mask = 0;
  uint32_t count = 0;
  bool cull_only = false;
};

constexpr UserClipPlaneConfig GetUserClipPlaneConfig(reg::PA_CL_CLIP_CNTL clip_cntl,
                                                     bool shader_clip_distance,
                                                     bool shader_cull_distance) {
  UserClipPlaneConfig config{};
  if (clip_cntl.clip_disable || !clip_cntl.ucp_ena) {
    return config;
  }
  if ((clip_cntl.ucp_cull_only_ena && shader_cull_distance) ||
      (!shader_clip_distance && shader_cull_distance)) {
    config.cull_only = true;
  } else if (!shader_clip_distance) {
    return config;
  }
  config.enabled_mask = clip_cntl.ucp_ena;
  config.count = uint32_t(std::popcount(config.enabled_mask));
  return config;
}

// Whether a render area covers the entire image extent from (0, 0).
constexpr bool CoversFullExtent(VkRect2D area, VkExtent2D extent) {
  return area.offset.x == 0 && area.offset.y == 0 && area.extent.width == extent.width &&
         area.extent.height == extent.height;
}

// Clamps one axis of draw resolution scaling so guest_extent * scale fits
// within the device's host attachment/viewport limits.
constexpr uint32_t ClampResolutionScaleAxis(uint32_t scale, uint32_t guest_extent,
                                            uint32_t max_host_extent) {
  if (!guest_extent) {
    return scale;
  }
  while (scale > 1 && uint64_t(guest_extent) * scale > max_host_extent) {
    --scale;
  }
  return scale;
}

// Size a draw's upload allocation needs so every uniform buffer bound with
// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC stays in bounds: the descriptor
// range is fixed, and Vulkan requires dynamic offset + range <= buffer size.
constexpr uint64_t ComputeDynamicUboRequiredSize(std::span<const uint64_t> offsets,
                                                 std::span<const uint64_t> ranges,
                                                 uint64_t size) {
  for (size_t i = 0; i < offsets.size() && i < ranges.size(); ++i) {
    size = std::max(size, offsets[i] + ranges[i]);
  }
  return size;
}

// Graphics state bound in the current render pass, so a draw only records the
// binds and dynamic state that differ from the previous draw.
struct DrawRenderState {
  static constexpr uint32_t kSetCount = 4;
  static constexpr uint32_t kDynamicOffsetCount = 5;
  // Set that holds the dynamic uniform buffers.
  static constexpr uint32_t kDynamicSet = 1;

  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, kSetCount> sets{};
  std::array<uint32_t, kDynamicOffsetCount> dynamic_offsets{};
  bool viewport_valid = false;
  VkViewport viewport{};
  bool scissor_valid = false;
  VkRect2D scissor{};
  bool depth_bias_valid = false;
  std::array<float, 2> depth_bias{};
  bool blend_constants_valid = false;
  std::array<float, 4> blend_constants{};
  // Front and back face values, with validity bits 1 (front) and 2 (back).
  std::array<uint32_t, 2> stencil_compare_mask{};
  std::array<uint32_t, 2> stencil_write_mask{};
  std::array<uint32_t, 2> stencil_reference{};
  uint32_t stencil_compare_mask_valid = 0;
  uint32_t stencil_write_mask_valid = 0;
  uint32_t stencil_reference_valid = 0;
  VkBuffer index_buffer = VK_NULL_HANDLE;
  uint64_t index_offset = 0;
  VkIndexType index_type = VK_INDEX_TYPE_UINT32;

  // Returns whether `current` must be rebound to `wanted`, updating it.
  template <typename T>
  static bool Update(bool& valid, T& current, const T& wanted) {
    if (valid && std::memcmp(&current, &wanted, sizeof(T)) == 0) {
      return false;
    }
    valid = true;
    current = wanted;
    return true;
  }
  bool SetPipeline(VkPipeline wanted) {
    if (pipeline == wanted) {
      return false;
    }
    pipeline = wanted;
    return true;
  }
  // First descriptor set to rebind (kSetCount when none). A different pipeline
  // layout may disturb every bound set, so it rebinds all of them; the dynamic
  // set also rebinds when its offsets change.
  uint32_t FirstChangedSet(VkPipelineLayout wanted_layout,
                           const std::array<VkDescriptorSet, kSetCount>& wanted_sets,
                           const std::array<uint32_t, kDynamicOffsetCount>& wanted_offsets) {
    uint32_t first = kSetCount;
    if (layout != wanted_layout) {
      first = 0;
    } else {
      for (uint32_t i = 0; i < kSetCount; ++i) {
        if (sets[i] != wanted_sets[i] || (i == kDynamicSet && dynamic_offsets != wanted_offsets)) {
          first = i;
          break;
        }
      }
    }
    layout = wanted_layout;
    sets = wanted_sets;
    dynamic_offsets = wanted_offsets;
    return first;
  }
  // Faces (bit 1 front, bit 2 back) whose value must be set to `wanted`.
  static uint32_t UpdateStencil(std::array<uint32_t, 2>& current, uint32_t& valid,
                                const std::array<uint32_t, 2>& wanted) {
    uint32_t faces = 0;
    for (uint32_t face = 0; face < 2; ++face) {
      if (!(valid & (1u << face)) || current[face] != wanted[face]) {
        faces |= 1u << face;
        current[face] = wanted[face];
      }
    }
    valid = 3;
    return faces;
  }
  bool SetIndexBuffer(VkBuffer buffer, uint64_t offset, VkIndexType type) {
    if (index_buffer == buffer && index_offset == offset && index_type == type) {
      return false;
    }
    index_buffer = buffer;
    index_offset = offset;
    index_type = type;
    return true;
  }
};

// Header of a VkPipelineCache blob stored on disk: the data is merged only on
// the device that wrote it and only when it is intact.
struct PipelineCacheFileHeader {
  static constexpr uint32_t kMagic = 0x4350564E;  // "NVPC"
  static constexpr uint32_t kVersion = 1;
  uint32_t magic = kMagic;
  uint32_t version = kVersion;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  std::array<uint8_t, VK_UUID_SIZE> cache_uuid{};
  uint64_t data_size = 0;
  uint64_t data_hash = 0;
};
static_assert(sizeof(PipelineCacheFileHeader) == 48);

inline PipelineCacheFileHeader MakePipelineCacheFileHeader(
    uint32_t vendor_id, uint32_t device_id, const std::array<uint8_t, VK_UUID_SIZE>& cache_uuid,
    std::span<const uint8_t> data) {
  PipelineCacheFileHeader header;
  header.vendor_id = vendor_id;
  header.device_id = device_id;
  header.cache_uuid = cache_uuid;
  header.data_size = data.size();
  header.data_hash = XXH3_64bits(data.data(), data.size());
  return header;
}

// The pipeline cache data in a stored file, or empty when the file is
// truncated, corrupt, from another version or from another device.
inline std::span<const uint8_t> GetPipelineCacheFileData(
    std::span<const uint8_t> file, uint32_t vendor_id, uint32_t device_id,
    const std::array<uint8_t, VK_UUID_SIZE>& cache_uuid) {
  PipelineCacheFileHeader header;
  if (file.size() < sizeof(header)) {
    return {};
  }
  std::memcpy(&header, file.data(), sizeof(header));
  const auto data = file.subspan(sizeof(header));
  if (header.magic != PipelineCacheFileHeader::kMagic ||
      header.version != PipelineCacheFileHeader::kVersion || header.vendor_id != vendor_id ||
      header.device_id != device_id || header.cache_uuid != cache_uuid ||
      header.data_size != data.size() ||
      header.data_hash != XXH3_64bits(data.data(), data.size())) {
    return {};
  }
  return data;
}

// Sparse binding granule of the shared vertex buffer: 4 MiB commits at most
// 128 allocations for the 512 MiB buffer, rounded up to the sparse alignment.
constexpr uint64_t kSparseVertexGranuleSize = 4 * 1024 * 1024;
constexpr uint64_t ComputeSparseGranuleSize(uint64_t alignment,
                                            uint64_t preferred = kSparseVertexGranuleSize) {
  alignment = std::max<uint64_t>(alignment, 1);
  return (std::max(preferred, alignment) + alignment - 1) / alignment * alignment;
}

// First granule and granule count covering [address, address + size).
constexpr std::pair<uint32_t, uint32_t> GetSparseGranuleRange(uint64_t address, uint64_t size,
                                                              uint64_t granule_size) {
  if (!size || !granule_size) {
    return {0, 0};
  }
  const uint64_t first = address / granule_size;
  const uint64_t last = (address + size - 1) / granule_size;
  return {uint32_t(first), uint32_t(last - first + 1)};
}

// A rectangle of EDRAM tiles (in tile units) to copy between two host images
// of targets with the same pitch, sample count and format that alias the same
// guest tiles.
struct EdramTileCopy {
  uint32_t src_x;
  uint32_t src_y;
  uint32_t dst_x;
  uint32_t dst_y;
  uint32_t width;
  uint32_t height;
  auto operator<=>(const EdramTileCopy&) const = default;
};

// Rectangles covering the guest tiles [overlap_start, overlap_end) shared by a
// target at src_base and one at dst_base, both pitch_tiles wide. Runs of tiles
// map to rows that may wrap at different columns in the two targets, and
// consecutive rows with the same columns merge.
inline std::vector<EdramTileCopy> ComputeOverlappingEdramTileCopies(uint32_t src_base,
                                                                   uint32_t dst_base,
                                                                   uint32_t pitch_tiles,
                                                                   uint32_t overlap_start,
                                                                   uint32_t overlap_end) {
  std::vector<EdramTileCopy> copies;
  if (!pitch_tiles || overlap_start >= overlap_end || overlap_start < src_base ||
      overlap_start < dst_base) {
    return copies;
  }
  for (uint32_t tile = overlap_start; tile < overlap_end;) {
    const uint32_t src = tile - src_base;
    const uint32_t dst = tile - dst_base;
    const uint32_t run = std::min({overlap_end - tile, pitch_tiles - src % pitch_tiles,
                                   pitch_tiles - dst % pitch_tiles});
    const EdramTileCopy row{src % pitch_tiles, src / pitch_tiles, dst % pitch_tiles,
                            dst / pitch_tiles, run, 1};
    auto* last = copies.empty() ? nullptr : &copies.back();
    if (last && last->src_x == row.src_x && last->dst_x == row.dst_x && last->width == row.width &&
        last->src_y + last->height == row.src_y && last->dst_y + last->height == row.dst_y) {
      ++last->height;
    } else {
      copies.push_back(row);
    }
    tile += run;
  }
  return copies;
}

// Host image usage flags for an EDRAM attachment or resolved texture. Both
// color and depth/stencil images need TRANSFER_SRC_BIT for aliased tile copies.
constexpr VkImageUsageFlags GetColorImageUsage(VkImageAspectFlags aspects, bool storage_supported) {
  VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
  if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (storage_supported) {
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
  } else {
    usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  }
  return usage;
}

// Non-overlapping EDRAM tile ownership interval used by NativeTargetCache.
struct TileRun {
  uint32_t start = 0;
  uint32_t end = 0;
  uint64_t owner_id = 0;
  uint32_t base = 0;
  uint32_t pitch_tiles = 0;
  xenos::MsaaSamples samples = xenos::MsaaSamples::k1X;
  VkFormat format = VK_FORMAT_UNDEFINED;
  bool is_64bpp = false;
  bool resolution_scaled = false;

  bool SameAttributes(const TileRun& o) const {
    return owner_id == o.owner_id && base == o.base && pitch_tiles == o.pitch_tiles &&
           samples == o.samples && format == o.format && is_64bpp == o.is_64bpp &&
           resolution_scaled == o.resolution_scaled;
  }
};

// Splices `claim` ([claim.start, claim.end)) into sorted non-overlapping `runs`,
// invoking `on_overwritten(previous_run, overlap_start, overlap_end)` for every
// covered sub-interval owned by a different `owner_id`. Returns true if
// `runs` changed.
template <typename OnOverwritten>
bool SpliceTileRun(std::vector<TileRun>& runs, const TileRun& claim,
                   OnOverwritten&& on_overwritten) {
  if (claim.start >= claim.end) {
    return false;
  }
  auto it = std::lower_bound(runs.begin(), runs.end(), claim.start,
                             [](const TileRun& r, uint32_t s) { return r.end <= s; });
  if (it != runs.end() && it->start <= claim.start && it->end >= claim.end &&
      it->SameAttributes(claim)) {
    return false;
  }
  bool changed = false;
  std::vector<TileRun> replacements;
  auto erase_begin = it;
  auto erase_end = it;
  if (it != runs.end() && it->start < claim.start) {
    TileRun left = *it;
    left.end = claim.start;
    replacements.push_back(left);
  }
  for (; erase_end != runs.end() && erase_end->start < claim.end; ++erase_end) {
    const uint32_t ov_start = std::max(erase_end->start, claim.start);
    const uint32_t ov_end = std::min(erase_end->end, claim.end);
    if (erase_end->owner_id != claim.owner_id) {
      on_overwritten(*erase_end, ov_start, ov_end);
      changed = true;
    } else if (!erase_end->SameAttributes(claim)) {
      changed = true;
    }
  }
  TileRun right{};
  bool has_right = false;
  if (erase_end != erase_begin) {
    const TileRun& last = *(erase_end - 1);
    if (last.end > claim.end) {
      right = last;
      right.start = claim.end;
      has_right = true;
    }
  } else {
    changed = true;
  }
  replacements.push_back(claim);
  if (has_right) {
    replacements.push_back(right);
  }
  // Merge adjacent runs with identical attributes inside replacements and neighbors.
  const size_t insert_pos = size_t(erase_begin - runs.begin());
  runs.erase(erase_begin, erase_end);
  runs.insert(runs.begin() + ptrdiff_t(insert_pos), replacements.begin(), replacements.end());
  size_t check_start = insert_pos > 0 ? insert_pos - 1 : 0;
  size_t check_end = std::min(runs.size(), insert_pos + replacements.size() + 1);
  for (size_t i = check_start; i + 1 < check_end && i + 1 < runs.size();) {
    if (runs[i].end == runs[i + 1].start && runs[i].SameAttributes(runs[i + 1])) {
      runs[i].end = runs[i + 1].end;
      runs.erase(runs.begin() + ptrdiff_t(i + 1));
      if (check_end > 0) --check_end;
    } else {
      ++i;
    }
  }
  return changed;
}

// Whether a surface pitch qualifies for 1x rendering under draw_resolution_scale_threshold.
constexpr bool IsDrawScaleNativeForPitch(uint32_t surface_pitch, xenos::MsaaSamples samples,
                                         int32_t threshold, uint32_t scale_x, uint32_t scale_y) {
  if (threshold <= 0 || (scale_x <= 1 && scale_y <= 1) || !surface_pitch) {
    return false;
  }
  const uint32_t msaa_x_log2 = uint32_t(samples >= xenos::MsaaSamples::k4X);
  const uint32_t pitch_tiles_32bpp =
      ((surface_pitch << msaa_x_log2) + (xenos::kEdramTileWidthSamples - 1)) /
      xenos::kEdramTileWidthSamples;
  const uint32_t pitch_pixels_aligned =
      pitch_tiles_32bpp * (xenos::kEdramTileWidthSamples >> msaa_x_log2);
  return pitch_pixels_aligned <= uint32_t(threshold);
}

// Host pixels of one EDRAM tile of a target: 80x16 samples, half as wide for
// 64bpp formats and 4x MSAA, half as tall for 2x and 4x MSAA.
constexpr VkExtent2D GetEdramTileHostExtent(xenos::MsaaSamples samples, bool is_64bpp,
                                            uint32_t scale_x, uint32_t scale_y) {
  const uint32_t width = xenos::kEdramTileWidthSamples >>
                         ((samples >= xenos::MsaaSamples::k4X ? 1 : 0) + (is_64bpp ? 1 : 0));
  const uint32_t height =
      xenos::kEdramTileHeightSamples >> (samples >= xenos::MsaaSamples::k2X ? 1 : 0);
  return {width * scale_x, height * scale_y};
}

}  // namespace rex::graphics::native
