#include <array>
#include <cstring>
#include <map>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <rex/graphics/pipeline/texture/util.h>

#include "../../../src/graphics/native/draw_input.h"
#include "../../../src/graphics/native/resolve_guest_memory.h"

using namespace rex::graphics;

TEST_CASE("Native cube textures upload all six faces", "[gpu][native]") {
  CHECK(native::GetTextureArrayLayers(xenos::DataDimension::kCube) == 6);
  CHECK(native::GetTextureArrayLayers(xenos::DataDimension::k2DOrStacked) == 1);

  // A 32x32 tiled k_8_8_8_8 cube: the base level holds six 4 KB faces, each
  // addressed relative to its own origin, so the upload has to reach into the
  // last face rather than stopping after the first.
  const auto layout = texture_util::GetGuestTextureLayout(
      xenos::DataDimension::kCube, 1, 32, 32, 1, true, xenos::TextureFormat::k_8_8_8_8, false,
      true, 0);
  REQUIRE(layout.array_size == 6);
  CHECK(layout.base.array_slice_stride_bytes >= xenos::kTextureSubresourceAlignmentBytes);
  CHECK(native::GetTextureLevelReadSize(layout.base.array_slice_stride_bytes,
                                        layout.base.array_slice_data_extent_bytes,
                                        layout.array_size) >=
        layout.base.array_slice_stride_bytes * 5);
}

TEST_CASE("Native DMA draws decode guest indices", "[gpu][native]") {
  const std::array<uint8_t, 4> index_bytes{0, 6, 0, 7};
  std::array<uint32_t, 2> indices{};
  REQUIRE(native::DecodeDmaIndices(index_bytes, xenos::IndexFormat::kInt16, xenos::Endian::k8in16,
                                   indices));
  CHECK(indices == std::array<uint32_t, 2>{6, 7});
  REQUIRE(native::DecodeDmaIndices(index_bytes, xenos::IndexFormat::kInt32, xenos::Endian::k8in32,
                                   std::span(indices).first(1)));
  CHECK(indices[0] == 0x60007);
  CHECK_FALSE(native::DecodeDmaIndices(index_bytes, xenos::IndexFormat::kInt32,
                                       xenos::Endian::k8in32, indices));
}

TEST_CASE("Native draw uploads stay inside guest physical memory", "[gpu][native]") {
  CHECK(native::IsPhysicalRange(0, 2));
  CHECK(native::IsPhysicalRange(0x1FFFFFFC, 4));
  CHECK_FALSE(native::IsPhysicalRange(0x1FFFFFFC, 8));
  CHECK_FALSE(native::IsPhysicalRange(0x20000000, 1));
  CHECK_FALSE(native::IsPhysicalRange(0, 0));
  CHECK_FALSE(native::IsPhysicalRange(8, SIZE_MAX));
}

TEST_CASE("Native resolve rectangles account for tile origin and wrap", "[gpu][native]") {
  draw_util::ResolveInfo copy{};
  copy.color_original_base = 2047;
  copy.color_edram_info.base_tiles = 1;
  copy.color_edram_info.pitch_tiles = 2;
  copy.coordinate_info.edram_offset_x_div_8 = 1;
  copy.coordinate_info.edram_offset_y_div_8 = 1;
  copy.coordinate_info.width_div_8 = 19;
  copy.height_div_8 = 3;

  const auto rect = native::GetResolveRectangle(copy, EdramLayout{}, 160, 48);
  REQUIRE(rect);
  CHECK(rect->x == 8);
  CHECK(rect->y == 24);
  CHECK(rect->width == 152);
  CHECK(rect->height == 24);
  CHECK_FALSE(native::GetResolveRectangle(copy, EdramLayout{}, 159, 48));
  CHECK_FALSE(native::GetResolveRectangle(copy, EdramLayout{}, 160, 47));

  copy.color_edram_info.pitch_tiles = 0;
  CHECK_FALSE(native::GetResolveRectangle(copy, EdramLayout{}, 160, 48));
}

TEST_CASE("Native resolve format conversion is bounded", "[gpu][native]") {
  draw_util::ResolveInfo copy{};
  copy.copy_dest_info.copy_dest_number = xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction;

  CHECK(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                xenos::ColorFormat::k_8));
  CHECK(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                xenos::ColorFormat::k_8_8));
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                      xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                      xenos::ColorFormat::k_8_8_8_8));
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k2X,
                                                      xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                      xenos::ColorFormat::k_8));
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                      xenos::ColorRenderTargetFormat::k_2_10_10_10,
                                                      xenos::ColorFormat::k_8));

  copy.copy_dest_info.copy_dest_swap = 1;
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                      xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                      xenos::ColorFormat::k_8));
  copy.copy_dest_info.copy_dest_swap = 0;

  copy.copy_dest_info.copy_dest_number = xenos::SurfaceNumberFormat::kUnsignedInteger;
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                      xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                      xenos::ColorFormat::k_8));
  copy.copy_dest_info.copy_dest_number = xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction;

  copy.copy_dest_info.copy_dest_exp_bias = 1;
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                      xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                      xenos::ColorFormat::k_8));
  copy.copy_dest_info.copy_dest_exp_bias = 0;

  copy.copy_dest_info.copy_dest_array = 1;
  CHECK_FALSE(native::IsResolveFormatConvertSupported(copy, xenos::MsaaSamples::k1X,
                                                      xenos::ColorRenderTargetFormat::k_8_8_8_8,
                                                      xenos::ColorFormat::k_8));
}

TEST_CASE("Native subrect resolves place inside a cached surface", "[gpu][native]") {
  draw_util::ResolveInfo copy{};
  copy.coordinate_info.width_div_8 = 4;
  copy.height_div_8 = 2;
  copy.copy_dest_x = 8;
  copy.copy_dest_y = 16;

  const auto placed = native::GetCachedResolveRectangle(copy, 256, 256);
  REQUIRE(placed);
  CHECK(placed->x == 8);
  CHECK(placed->y == 16);
  CHECK(placed->width == 32);
  CHECK(placed->height == 16);

  copy.copy_dest_x = 40;
  copy.copy_dest_y = 64;
  const auto later_tile = native::GetCachedResolveRectangle(copy, 72, 80);
  REQUIRE(later_tile);
  CHECK(later_tile->x == 40);
  CHECK(later_tile->y == 64);
  CHECK(later_tile->width == 32);
  CHECK(later_tile->height == 16);
  CHECK_FALSE(native::GetCachedResolveRectangle(copy, 71, 80));
  CHECK_FALSE(native::GetCachedResolveRectangle(copy, 72, 79));
  CHECK_FALSE(native::GetCachedResolveRectangle(copy, 39, 80));
}

TEST_CASE("Native target height fits Vulkan limits without moving guest tiles", "[gpu][native]") {
  // SSA pitch80 at 1x is one tile wide: the whole 2048-tile EDRAM spans 32768
  // rows, which Steam Frame Turnip caps at 16384. Capping keeps every tile row
  // at its guest coordinate, so the image is exactly 16384 rows tall.
  CHECK(native::GetTargetHeight(80, xenos::MsaaSamples::k1X, xenos::kEdramTileCount, 16384) ==
        16384);
  // A device whose limit already fits the whole guest span keeps it unchanged,
  // so desktop behaviour is identical to before.
  CHECK(native::GetTargetHeight(80, xenos::MsaaSamples::k1X, xenos::kEdramTileCount, 32768) ==
        32768);
  CHECK(native::GetTargetHeight(80, xenos::MsaaSamples::k1X, xenos::kEdramTileCount, 65536) ==
        32768);
  // Extended 4096-tile EDRAM would need 65536 rows, still capped at the limit.
  CHECK(native::GetTargetHeight(80, xenos::MsaaSamples::k1X, xenos::kEdramTileCountMax, 16384) ==
        16384);
  // The cap is rounded down to whole tile rows, never past the limit.
  CHECK(native::GetTargetHeight(80, xenos::MsaaSamples::k1X, xenos::kEdramTileCount, 16380) ==
        16368);
  // Wider surfaces are shorter and stay below the cap.
  CHECK(native::GetTargetHeight(1280, xenos::MsaaSamples::k1X, xenos::kEdramTileCount, 16384) ==
        2048);
  CHECK(native::GetTargetHeight(0, xenos::MsaaSamples::k1X, xenos::kEdramTileCount, 16384) == 0);
}

TEST_CASE("Native resolve sample selection maps to host samples", "[gpu][native]") {
  using xenos::CopySampleSelect;
  using xenos::MsaaSamples;
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k1X, CopySampleSelect::k0) == 0);
  // 2x guest samples are top/bottom, which the standard Vulkan locations swap.
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k2X, CopySampleSelect::k0) == 1);
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k2X, CopySampleSelect::k1) == 0);
  // 4x guest samples match the standard Vulkan sample locations directly.
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k4X, CopySampleSelect::k0) == 0);
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k4X, CopySampleSelect::k1) == 1);
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k4X, CopySampleSelect::k2) == 2);
  CHECK(native::GetResolveSampleIndex(MsaaSamples::k4X, CopySampleSelect::k3) == 3);
}

TEST_CASE("Native draw expanded strips preserve guest primitive boundaries", "[gpu][native]") {
  CHECK(native::QuadStripIndices(0).empty());
  CHECK(native::QuadStripIndices(2) == std::vector<uint32_t>{0, 1, 2, 3, UINT32_MAX, 4, 5, 6, 7});
}

TEST_CASE("Native non-polygonal primitives ignore inherited cull state", "[gpu][native]") {
  // Rectangles are non-polygonal, so leftover front/back culling must not hide them.
  const auto rectangle = native::GetCullFaces(false, true, true);
  CHECK_FALSE(rectangle.front);
  CHECK_FALSE(rectangle.back);
  const auto rectangle_front = native::GetCullFaces(false, true, false);
  CHECK_FALSE(rectangle_front.front);
  CHECK_FALSE(rectangle_front.back);
  // Polygonal primitives keep the guest's per-face culling.
  const auto triangle = native::GetCullFaces(true, true, false);
  CHECK(triangle.front);
  CHECK_FALSE(triangle.back);
  const auto triangle_back = native::GetCullFaces(true, false, true);
  CHECK_FALSE(triangle_back.front);
  CHECK(triangle_back.back);
}

TEST_CASE("Native clip disable maps to depth clamping", "[gpu][native]") {
  // CLIP_DISABLE makes the guest expect out-of-range depth to clamp instead of
  // being clipped, which the aliased 4x-MSAA fast clears depend on: their quads
  // use a near-zero Z that would otherwise be discarded.
  CHECK(native::GetDepthClampEnable(true, true));
  CHECK_FALSE(native::GetDepthClampEnable(true, false));
  // Never request the feature when the device does not support it.
  CHECK_FALSE(native::GetDepthClampEnable(false, true));
  CHECK_FALSE(native::GetDepthClampEnable(false, false));
}

TEST_CASE("Native cache eviction skips in-flight and protected entries", "[gpu][native]") {
  struct Entry {
    uint64_t last_use_submission;
    int id;
  };
  const std::map<int, Entry> entries{{10, {1, 10}}, {20, {2, 20}}, {30, {5, 30}}};
  const auto evict_any = [](int, const Entry&) { return true; };
  const auto protect_oldest = [](int, const Entry& entry) { return entry.id != 10; };

  // The second argument is the first submission still protected as in flight or
  // being recorded. Entries used in an earlier submission are retired.
  CHECK(native::SelectRetiredEviction(entries, 2, evict_any) == 10);
  CHECK(native::SelectRetiredEviction(entries, 6, evict_any) == 10);
  // Submission 1 is protected here, so the last use at 1 is not yet evictable.
  CHECK_FALSE(native::SelectRetiredEviction(entries, 1, evict_any).has_value());
  // At boundary 3 the entry used in submission 2 is retired; key 10 is protected.
  CHECK(native::SelectRetiredEviction(entries, 3, protect_oldest) == 20);
  // Protecting the presented or still-referenced entries can leave no victim.
  CHECK_FALSE(native::SelectRetiredEviction(entries, 6, [](int, const Entry&) {
                return false;
              }).has_value());
}

TEST_CASE("Native texture channel counts drive swizzle replication", "[gpu][native]") {
  using xenos::TextureFormat;
  // Single-channel formats replicate R into every component.
  CHECK(native::TextureChannelCount(TextureFormat::k_8) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_8_A) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_8_B) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_32_FLOAT) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_24_8) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_24_8_FLOAT) == 1);
  // Two-channel formats replicate RGGG.
  CHECK(native::TextureChannelCount(TextureFormat::k_8_8) == 2);
  CHECK(native::TextureChannelCount(TextureFormat::k_16_16_FLOAT) == 2);
  // DXN is BC5: two channels in a 16-byte block, so a block size of 16 must not
  // be mistaken for a channel count.
  CHECK(native::TextureChannelCount(TextureFormat::k_DXN) == 2);
  // Four-channel formats pass RGBA through.
  CHECK(native::TextureChannelCount(TextureFormat::k_8_8_8_8) == 4);
  CHECK(native::TextureChannelCount(TextureFormat::k_2_10_10_10) == 4);
  CHECK(native::TextureChannelCount(TextureFormat::k_DXT1) == 4);
  // Three-channel k_5_6_5 swaps R and B and replicates the 3rd channel into A.
  CHECK(native::TextureChannelCount(TextureFormat::k_5_6_5) == 3);
  const auto k565_swizzle = native::MakeTextureComponentMapping(
      xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 3, true);
  CHECK(k565_swizzle.r == VK_COMPONENT_SWIZZLE_B);
  CHECK(k565_swizzle.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(k565_swizzle.b == VK_COMPONENT_SWIZZLE_R);
  CHECK(k565_swizzle.a == VK_COMPONENT_SWIZZLE_R);
}

TEST_CASE("Native signed fetches use the same view only for float formats", "[gpu][native]") {
  using xenos::TextureFormat;
  for (const auto format : {TextureFormat::k_32_FLOAT, TextureFormat::k_16_16_FLOAT,
                            TextureFormat::k_16_16_16_16_FLOAT}) {
    const auto info = native::GetTextureFormatInfo(format);
    CHECK(info.signed_format == info.format);
    CHECK_FALSE(info.needs_signed_view());
  }
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_8_8_8_8).needs_signed_view());
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_8_8).needs_signed_view());
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_DXN).needs_signed_view());
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_2_10_10_10).signed_format ==
        VK_FORMAT_UNDEFINED);
}

TEST_CASE("Native invalid fetches bind a zero-cleared fallback", "[gpu][native]") {
  // Only a real texture of the binding's dimension with a level to sample
  // avoids the fallback; stacked textures are real textures now.
  CHECK_FALSE(native::ShouldUseFallbackTexture(true, true, true));
  // Anything the guest would read as zero takes the fallback instead.
  CHECK(native::ShouldUseFallbackTexture(false, true, true));
  CHECK(native::ShouldUseFallbackTexture(true, false, true));
  CHECK(native::ShouldUseFallbackTexture(true, true, false));
  // Unsupported formats or unreadable memory still reject, and those never
  // reach this decision because ReadTexture fails first - the checks above only
  // cover the pre-ReadTexture fallback conditions.
}

TEST_CASE("Native aligned resolve crop stays within one alignment unit", "[gpu][native]") {
  // Giants resolves a 500x300 texture from an 8x8-aligned 504x304 image.
  CHECK(native::CanCropAlignedResolve(504, 304, 500, 300));
  // 504 - 500 = 4 < 8, 304 - 300 = 4 < 8: both fit.
  CHECK(native::CanCropAlignedResolve(500, 300, 500, 300));
  // Too much slack cannot be cropped from the top-left corner.
  CHECK_FALSE(native::CanCropAlignedResolve(512, 304, 500, 300));
  CHECK_FALSE(native::CanCropAlignedResolve(504, 312, 500, 300));
  // A cached image smaller than the texture cannot be cropped from.
  CHECK_FALSE(native::CanCropAlignedResolve(496, 304, 500, 300));
  CHECK_FALSE(native::CanCropAlignedResolve(504, 296, 500, 300));
  // The maximum accepted slack is one alignment unit less.
  CHECK(native::CanCropAlignedResolve(507, 307, 500, 300));
  CHECK_FALSE(native::CanCropAlignedResolve(508, 300, 500, 300));
  CHECK_FALSE(native::CanCropAlignedResolve(500, 308, 500, 300));
}

TEST_CASE("Native resolved cube faces map to per-face resolve addresses", "[gpu][native]") {
  // Skylanders SSA renders each ambient cube face as a 32x32 color target and
  // resolves it to its own slice of a tiled k_8_8_8_8 cube, leaving guest
  // memory zero, so the fetch has to find the face images at these addresses.
  const auto layout = texture_util::GetGuestTextureLayout(
      xenos::DataDimension::kCube, 1, 32, 32, 1, true, xenos::TextureFormat::k_8_8_8_8, false,
      true, 0);
  const uint32_t base = 0x1C2B8000u;
  CHECK(layout.base.array_slice_stride_bytes == 0x1000u);
  for (uint32_t face = 0; face < native::kTextureCubeArrayLayers; ++face) {
    CHECK(native::GetResolvedCubeFaceAddress(base, layout.base.array_slice_stride_bytes, face) ==
          base + face * 0x1000u);
  }
  // A resolved face is usable when its format matches and it covers the face
  // within one 8x8 resolve alignment unit.
  CHECK(native::CanUseResolvedCubeFace(true, 32, 32, 32, 32));
  CHECK(native::CanUseResolvedCubeFace(true, 39, 39, 32, 32));
  CHECK_FALSE(native::CanUseResolvedCubeFace(true, 40, 32, 32, 32));
  CHECK_FALSE(native::CanUseResolvedCubeFace(false, 32, 32, 32, 32));
  CHECK_FALSE(native::CanUseResolvedCubeFace(true, 48, 32, 32, 32));
  CHECK_FALSE(native::CanUseResolvedCubeFace(true, 24, 32, 32, 32));
}

TEST_CASE("Native resolved cube cache keys and face versions control re-uploads",
          "[gpu][native]") {
  native::ResolvedCubeKey key{0x1C2B8000u, 32, 32, VK_FORMAT_R8G8B8A8_UNORM};
  native::ResolvedCubeKey same = key;
  CHECK(same == key);
  CHECK_FALSE(same < key);
  CHECK_FALSE(key < same);

  native::ResolvedCubeKeyHash hasher;
  CHECK(hasher(key) == hasher(same));

  native::ResolvedCubeKey diff_addr = key;
  diff_addr.address = 0x1C2C0000u;
  CHECK_FALSE(diff_addr == key);
  CHECK(hasher(diff_addr) != hasher(key));

  native::ResolvedCubeKey diff_size = key;
  diff_size.width = 64;
  CHECK_FALSE(diff_size == key);
  CHECK(hasher(diff_size) != hasher(key));

  native::ResolvedCubeKey diff_fmt = key;
  diff_fmt.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  CHECK_FALSE(diff_fmt == key);
  CHECK(hasher(diff_fmt) != hasher(key));

  const std::array<uint64_t, 6> v1{1, 2, 3, 4, 5, 6};
  // First draw before any upload must copy the six faces.
  CHECK(native::ShouldUploadResolvedCube(false, {}, v1));
  // Subsequent draws with unchanged face versions reuse the cached cube.
  CHECK_FALSE(native::ShouldUploadResolvedCube(true, v1, v1));
  // Updating any single face version invalidates the cached cube.
  std::array<uint64_t, 6> v2 = v1;
  v2[4] = 7;
  CHECK(native::ShouldUploadResolvedCube(true, v1, v2));
}

TEST_CASE("Native resolve rectangles use multisample tile dimensions", "[gpu][native]") {
  draw_util::ResolveInfo copy{};
  copy.depth_edram_info.pitch_tiles = 4;
  copy.depth_edram_info.base_tiles = 6;
  copy.depth_edram_info.msaa_samples = xenos::MsaaSamples::k4X;
  copy.depth_original_base = 1;
  copy.coordinate_info.edram_offset_y_div_8 = 0;
  copy.coordinate_info.width_div_8 = 3;
  copy.height_div_8 = 1;
  const auto rect = native::GetResolveRectangle(copy, EdramLayout{}, 160, 48, true);
  REQUIRE(rect);
  CHECK(rect->x == 40);
  CHECK(rect->y == 8);
  CHECK(rect->width == 24);
}

TEST_CASE("Native 64bpp targets halve EDRAM tile width and target height", "[gpu][native]") {
  // At 1280px pitch, 32bpp uses 16 tiles/row (2048/16*16 = 2048px) while 64bpp
  // uses 32 tiles/row (2048/32*16 = 1024px).
  CHECK(native::GetTargetHeight(1280, xenos::MsaaSamples::k1X, 2048, 8192, false) == 2048);
  CHECK(native::GetTargetHeight(1280, xenos::MsaaSamples::k1X, 2048, 8192, true) == 1024);

  draw_util::ResolveInfo copy{};
  copy.color_edram_info.format =
      uint32_t(xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT);
  copy.color_edram_info.pitch_tiles = 32;
  copy.color_edram_info.base_tiles = 33;
  copy.color_edram_info.msaa_samples = xenos::MsaaSamples::k1X;
  copy.color_original_base = 0;
  copy.coordinate_info.width_div_8 = 2;
  copy.height_div_8 = 1;
  const auto rect = native::GetResolveRectangle(copy, EdramLayout{}, 1280, 1024, false);
  REQUIRE(rect);
  CHECK(rect->x == 40);
  CHECK(rect->y == 16);
}

TEST_CASE("Native guest texture cache keys cover the uploaded texels", "[gpu][native]") {
  native::GuestTextureKey key{};
  key.base_page = 0x12;
  key.mip_page = 0x34;
  key.width = 128;
  key.height = 64;
  key.array_layers = 1;
  key.format = xenos::TextureFormat::k_8_8_8_8;
  key.dimension = xenos::DataDimension::k2DOrStacked;
  key.tiled = true;
  key.packed_mips = false;
  key.endianness = xenos::Endian::k8in32;
  key.pitch = 4;
  key.mip_max = 3;

  native::GuestTextureKey same = key;
  CHECK(same == key);
  CHECK_FALSE(same < key);
  CHECK_FALSE(key < same);

  // Each field that changes the decoded bytes must split the cache.
  const auto differs = [&key](auto mutate) {
    native::GuestTextureKey other = key;
    mutate(other);
    return other != key;
  };
  CHECK(differs([](auto& k) { ++k.base_page; }));
  CHECK(differs([](auto& k) { ++k.mip_page; }));
  CHECK(differs([](auto& k) { ++k.width; }));
  CHECK(differs([](auto& k) { ++k.height; }));
  CHECK(differs([](auto& k) { ++k.array_layers; }));
  CHECK(differs([](auto& k) { k.format = xenos::TextureFormat::k_8_8; }));
  CHECK(differs([](auto& k) { k.dimension = xenos::DataDimension::kCube; }));
  CHECK(differs([](auto& k) { k.tiled = false; }));
  CHECK(differs([](auto& k) { k.packed_mips = true; }));
  CHECK(differs([](auto& k) { k.endianness = xenos::Endian::k16in32; }));
  CHECK(differs([](auto& k) { ++k.pitch; }));
  CHECK(differs([](auto& k) { ++k.mip_max; }));

  // Ordering is total over the key, which a std::map requires.
  native::GuestTextureKey larger = key;
  ++larger.base_page;
  CHECK(key < larger);
  CHECK_FALSE(larger < key);
}

TEST_CASE("Native dirty ranges coalesce without covering clean gaps", "[gpu][native]") {
  using native::PhysicalRange;
  std::vector<PhysicalRange> ranges{
      {0x4000, 0x100}, {0x1100, 0x100}, {0x1080, 0x80}, {0x1000, 0x100},
      {0x1100, 0x100}, {0x2000, 0}, {native::kPhysicalMemorySize - 16, 16}};
  native::CoalescePhysicalRanges(ranges);
  REQUIRE(ranges == std::vector<PhysicalRange>{
                        {0x1000, 0x200}, {0x4000, 0x100},
                        {native::kPhysicalMemorySize - 16, 16}});
  CHECK(native::OverlapsPhysicalRanges(ranges, 0x10FF, 2));
  CHECK_FALSE(native::OverlapsPhysicalRanges(ranges, 0x1200, 0x2E00));
  CHECK(native::OverlapsPhysicalRanges(ranges, 0x1200, 0x2E01));
  CHECK_FALSE(native::OverlapsPhysicalRanges(ranges, 0x1008, 0));
  CHECK_FALSE(native::OverlapsPhysicalRanges(ranges, native::kPhysicalMemorySize, 1));
}

TEST_CASE("Native physical range overlap is inclusive of shared bytes only", "[gpu][native]") {
  // Disjoint ranges never overlap.
  CHECK_FALSE(native::RangesOverlap(0x1000, 0x100, 0x2000, 0x100));
  // Touching at a boundary is not an overlap.
  CHECK_FALSE(native::RangesOverlap(0x1000, 0x100, 0x1100, 0x100));
  CHECK_FALSE(native::RangesOverlap(0x1100, 0x100, 0x1000, 0x100));
  // Partial overlaps in either direction.
  CHECK(native::RangesOverlap(0x1000, 0x200, 0x1100, 0x200));
  CHECK(native::RangesOverlap(0x1100, 0x200, 0x1000, 0x200));
  // Containment, including an identical range.
  CHECK(native::RangesOverlap(0x1000, 0x1000, 0x1800, 0x10));
  CHECK(native::RangesOverlap(0x1800, 0x10, 0x1000, 0x1000));
  CHECK(native::RangesOverlap(0x1000, 0x100, 0x1000, 0x100));
  // Empty ranges never overlap anything.
  CHECK_FALSE(native::RangesOverlap(0x1000, 0, 0x1000, 0x100));
  CHECK_FALSE(native::RangesOverlap(0x1000, 0x100, 0x1000, 0));
  CHECK_FALSE(native::RangesOverlap(0, 0, 0, 0));
  // A range ending at the top of the 512 MB space must not wrap around.
  CHECK_FALSE(native::RangesOverlap(0xFFFFFFF0u, 0x20u, 0, 0x10u));
  CHECK(native::RangesOverlap(0x1FFFFFF0u, 0x10u, 0x1FFFFFF8u, 0x8u));
  CHECK_FALSE(native::RangesOverlap(0x1FFFFFF8u, 0x8u, 0x20000000u, 0x10u));
}

TEST_CASE("Native sampler cache keys order by sampler state", "[gpu][native]") {
  native::DrawSampler sampler{};
  sampler.min_filter = VK_FILTER_LINEAR;
  sampler.mag_filter = VK_FILTER_NEAREST;
  sampler.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler.max_lod = 4.0f;
  sampler.address_u = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler.address_v = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;

  native::DrawSampler same = sampler;
  CHECK(same == sampler);
  CHECK_FALSE(same < sampler);
  CHECK_FALSE(sampler < same);

  const auto differs = [&sampler](auto mutate) {
    native::DrawSampler other = sampler;
    mutate(other);
    return other != sampler;
  };
  CHECK(differs([](auto& s) { s.min_filter = VK_FILTER_NEAREST; }));
  CHECK(differs([](auto& s) { s.mag_filter = VK_FILTER_LINEAR; }));
  CHECK(differs([](auto& s) { s.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST; }));
  CHECK(differs([](auto& s) { s.max_lod = 5.0f; }));
  CHECK(differs([](auto& s) { s.address_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }));
  CHECK(differs([](auto& s) { s.address_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }));

  native::DrawSampler later = sampler;
  later.max_lod = 8.0f;
  CHECK(sampler < later);
  CHECK_FALSE(later < sampler);
}

TEST_CASE("Native texture swizzle handles channel replication and copy_dest_swap", "[gpu][native]") {
  // 4-channel guest fetch 0x60A = (B, G, R, A), 0xA0A = (B, G, R, 1), 0x688 = (R, G, B, A).
  const auto bgra_unswapped = native::MakeTextureComponentMapping(0x60A, 4, false);
  CHECK(bgra_unswapped.r == VK_COMPONENT_SWIZZLE_B);
  CHECK(bgra_unswapped.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(bgra_unswapped.b == VK_COMPONENT_SWIZZLE_R);
  CHECK(bgra_unswapped.a == VK_COMPONENT_SWIZZLE_A);

  // When a resolved image skipped copy_dest_swap, swapping R/B cancels 0x60A -> RGBA.
  const auto bgra_swapped = native::MakeTextureComponentMapping(0x60A, 4, true);
  CHECK(bgra_swapped.r == VK_COMPONENT_SWIZZLE_R);
  CHECK(bgra_swapped.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(bgra_swapped.b == VK_COMPONENT_SWIZZLE_B);
  CHECK(bgra_swapped.a == VK_COMPONENT_SWIZZLE_A);

  const auto bgr1_swapped = native::MakeTextureComponentMapping(0xA0A, 4, true);
  CHECK(bgr1_swapped.r == VK_COMPONENT_SWIZZLE_R);
  CHECK(bgr1_swapped.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(bgr1_swapped.b == VK_COMPONENT_SWIZZLE_B);
  CHECK(bgr1_swapped.a == VK_COMPONENT_SWIZZLE_ONE);

  const auto rgba_swapped = native::MakeTextureComponentMapping(0x688, 4, true);
  CHECK(rgba_swapped.r == VK_COMPONENT_SWIZZLE_B);
  CHECK(rgba_swapped.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(rgba_swapped.b == VK_COMPONENT_SWIZZLE_R);
  CHECK(rgba_swapped.a == VK_COMPONENT_SWIZZLE_A);

  // 1- and 2-channel formats replicate stored components and ignore copy_dest_swap.
  const auto single = native::MakeTextureComponentMapping(0x688, 1, true);
  CHECK(single.r == VK_COMPONENT_SWIZZLE_R);
  CHECK(single.g == VK_COMPONENT_SWIZZLE_R);
  CHECK(single.b == VK_COMPONENT_SWIZZLE_R);
  CHECK(single.a == VK_COMPONENT_SWIZZLE_R);

  const auto two = native::MakeTextureComponentMapping(0x688, 2, true);
  CHECK(two.r == VK_COMPONENT_SWIZZLE_R);
  CHECK(two.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(two.b == VK_COMPONENT_SWIZZLE_G);
  CHECK(two.a == VK_COMPONENT_SWIZZLE_G);
}

TEST_CASE("Native 4x-MSAA fast-clear scaling preserves framebuffer coordinates", "[gpu][native]") {
  CHECK(native::GetMsaa4xAliasScale(xenos::MsaaSamples::k1X).x == 2);
  CHECK(native::GetMsaa4xAliasScale(xenos::MsaaSamples::k1X).y == 2);
  CHECK(native::GetMsaa4xAliasScale(xenos::MsaaSamples::k2X).x == 2);
  CHECK(native::GetMsaa4xAliasScale(xenos::MsaaSamples::k2X).y == 1);
  CHECK(native::GetMsaa4xAliasScale(xenos::MsaaSamples::k4X).x == 1);
  CHECK(native::GetMsaa4xAliasScale(xenos::MsaaSamples::k4X).y == 1);

  // Unclamped scaling: 632x352 -> 1264x704 and 8192 -> 16384 within maxViewportDimensions.
  {
    draw_util::ViewportInfo viewport{};
    viewport.xy_offset[0] = 0;
    viewport.xy_offset[1] = 0;
    viewport.xy_extent[0] = 8192;
    viewport.xy_extent[1] = 8192;
    draw_util::Scissor scissor{};
    scissor.offset[0] = 0;
    scissor.offset[1] = 0;
    scissor.extent[0] = 632;
    scissor.extent[1] = 352;
    std::array<float, 3> ndc_scale{2.0f / 8192.0f, 2.0f / 8192.0f, 1.0f};
    std::array<float, 3> ndc_offset{-1.0f, -1.0f, 0.0f};
    native::ScaleViewportAndScissor(viewport, scissor, ndc_scale, ndc_offset, 2, 2, 16384, 16384);
    CHECK(viewport.xy_extent[0] == 16384);
    CHECK(viewport.xy_extent[1] == 16384);
    CHECK(scissor.extent[0] == 1264);
    CHECK(scissor.extent[1] == 704);
    CHECK(ndc_scale[0] == Catch::Approx(2.0f / 8192.0f));
    CHECK(ndc_offset[0] == Catch::Approx(-1.0f));
  }

  // Clamped scaling: when extent * scale exceeds max_viewport, NDC scaling keeps fb coords exact.
  {
    draw_util::ViewportInfo viewport{};
    viewport.xy_offset[0] = 0;
    viewport.xy_offset[1] = 0;
    viewport.xy_extent[0] = 8192;
    viewport.xy_extent[1] = 8192;
    draw_util::Scissor scissor{};
    scissor.extent[0] = 640;
    scissor.extent[1] = 352;
    std::array<float, 3> ndc_scale{2.0f / 8192.0f, 2.0f / 8192.0f, 1.0f};
    std::array<float, 3> ndc_offset{-1.0f, -1.0f, 0.0f};
    native::ScaleViewportAndScissor(viewport, scissor, ndc_scale, ndc_offset, 2, 2, 8192, 8192);
    CHECK(viewport.xy_extent[0] == 8192);
    CHECK(viewport.xy_extent[1] == 8192);
    // Guest vertex x = 600 (4x) must map to fb_x = 1200 (1x).
    const float ndc_x = 600.0f * ndc_scale[0] + ndc_offset[0];
    const float fb_x = (ndc_x + 1.0f) * 0.5f * float(viewport.xy_extent[0]) +
                       float(viewport.xy_offset[0]);
    CHECK(fb_x == Catch::Approx(1200.0f));
  }
}

TEST_CASE("Native component mappings pack into unique cache keys", "[gpu][native]") {
  // The packing is byte-stable, so identical mappings share a cached view.
  CHECK(native::PackTextureComponentMapping(
            {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
             VK_COMPONENT_SWIZZLE_A}) == 0x06050403u);
  CHECK(native::PackTextureComponentMapping(native::MakeTextureComponentMapping(0x688, 4, false)) ==
        native::PackTextureComponentMapping(native::MakeTextureComponentMapping(0x688, 4, false)));
  // Different swizzles must not share a view.
  CHECK(native::PackTextureComponentMapping(native::MakeTextureComponentMapping(0x688, 4, false)) !=
        native::PackTextureComponentMapping(native::MakeTextureComponentMapping(0x60A, 4, false)));
  // copy_dest_swap changes the mapping even for the same fetch swizzle.
  CHECK(native::PackTextureComponentMapping(native::MakeTextureComponentMapping(0x60A, 4, false)) !=
        native::PackTextureComponentMapping(native::MakeTextureComponentMapping(0x60A, 4, true)));
}

TEST_CASE("Native cache hashes ignore padding and track key fields", "[gpu][native]") {
  native::GuestTextureKey key{};
  key.base_page = 0x12;
  key.mip_page = 0x34;
  key.width = 128;
  key.height = 64;
  key.array_layers = 1;
  key.format = xenos::TextureFormat::k_8_8_8_8;
  key.dimension = xenos::DataDimension::k2DOrStacked;
  key.tiled = true;
  key.endianness = xenos::Endian::k8in32;
  key.pitch = 4;
  key.mip_max = 3;
  native::GuestTextureKey same = key;
  // An identical key built independently must hash into the same bucket.
  native::GuestTextureKey rebuilt{};
  rebuilt.base_page = key.base_page;
  rebuilt.mip_page = key.mip_page;
  rebuilt.width = key.width;
  rebuilt.height = key.height;
  rebuilt.array_layers = key.array_layers;
  rebuilt.format = key.format;
  rebuilt.dimension = key.dimension;
  rebuilt.tiled = key.tiled;
  rebuilt.packed_mips = key.packed_mips;
  rebuilt.endianness = key.endianness;
  rebuilt.pitch = key.pitch;
  rebuilt.mip_max = key.mip_max;
  const native::GuestTextureKeyHash texture_hash;
  CHECK(texture_hash(key) == texture_hash(same));
  CHECK(texture_hash(key) == texture_hash(rebuilt));
  rebuilt.pitch = key.pitch + 1;
  CHECK(texture_hash(key) != texture_hash(rebuilt));

  native::DrawSampler sampler{};
  sampler.min_filter = VK_FILTER_LINEAR;
  sampler.mag_filter = VK_FILTER_NEAREST;
  sampler.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler.max_lod = 4.0f;
  sampler.address_u = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler.address_v = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  native::DrawSampler same_sampler = sampler;
  const native::DrawSamplerHash sampler_hash;
  CHECK(sampler_hash(sampler) == sampler_hash(same_sampler));
  same_sampler.max_lod = 5.0f;
  CHECK(sampler_hash(sampler) != sampler_hash(same_sampler));
}

TEST_CASE("Native EDRAM depth-to-color clear detection and format modes", "[gpu][native]") {
  reg::RB_DEPTHCONTROL depth{};
  depth.z_enable = 1;
  depth.z_write_enable = 1;
  depth.zfunc = xenos::CompareFunction::kAlways;
  depth.stencil_enable = 1;
  depth.stencilfunc = xenos::CompareFunction::kAlways;
  depth.stencilzpass = xenos::StencilOp::kReplace;
  CHECK(native::IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat::kD24S8, depth, 0xFF));

  depth.stencilzpass = xenos::StencilOp::kZero;
  CHECK(native::IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat::kD24S8, depth, 0xFF));
  CHECK_FALSE(
      native::IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat::kD24FS8, depth, 0xFF));
  CHECK_FALSE(native::IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat::kD24S8, depth, 0x0F));

  depth.z_write_enable = 0;
  CHECK_FALSE(native::IsEdramDepthClearToColor(xenos::DepthRenderTargetFormat::kD24S8, depth, 0xFF));

  CHECK(native::GetEdramClearColorFormatMode(VK_FORMAT_R8G8B8A8_UNORM) == 0u);
  CHECK(native::GetEdramClearColorFormatMode(VK_FORMAT_A2B10G10R10_UNORM_PACK32) == 1u);
  CHECK(native::GetEdramClearColorFormatMode(VK_FORMAT_R16G16_SFLOAT) == 2u);
  CHECK(native::GetEdramClearColorFormatMode(VK_FORMAT_R32_SFLOAT) == 3u);
  CHECK_FALSE(native::GetEdramClearColorFormatMode(VK_FORMAT_D24_UNORM_S8_UINT).has_value());
}

TEST_CASE("Native depth format falls back when D24_UNORM_S8_UINT is unsupported",
          "[gpu][native]") {
  CHECK(native::SelectDepthFormat(xenos::DepthRenderTargetFormat::kD24S8, true) ==
        VK_FORMAT_D24_UNORM_S8_UINT);
  CHECK(native::SelectDepthFormat(xenos::DepthRenderTargetFormat::kD24S8, false) ==
        VK_FORMAT_D32_SFLOAT_S8_UINT);
  CHECK(native::SelectDepthFormat(xenos::DepthRenderTargetFormat::kD24FS8, true) ==
        VK_FORMAT_D32_SFLOAT_S8_UINT);
  CHECK(native::SelectDepthFormat(xenos::DepthRenderTargetFormat::kD24FS8, false) ==
        VK_FORMAT_D32_SFLOAT_S8_UINT);
}

TEST_CASE("Native 4x-MSAA kColorDepth fast-clear detection and 64bpp pitch alias parity",
          "[gpu][native]") {
  reg::RB_DEPTHCONTROL depth{};
  depth.z_enable = 1;
  depth.z_write_enable = 1;
  depth.zfunc = xenos::CompareFunction::kAlways;
  depth.stencil_enable = 1;
  depth.stencilfunc = xenos::CompareFunction::kAlways;

  // Color + depth, color-only, and stencil-only 4x-MSAA rectangle clears.
  CHECK(native::IsAliased4xColorDepthClear(true, true, false,
                                            xenos::PrimitiveType::kRectangleList,
                                            xenos::MsaaSamples::k4X, true, false, true, depth));
  CHECK(native::IsAliased4xColorDepthClear(true, true, false,
                                            xenos::PrimitiveType::kRectangleList,
                                            xenos::MsaaSamples::k4X, true, false, false, depth));
  reg::RB_DEPTHCONTROL stencil_only{};
  stencil_only.stencil_enable = 1;
  stencil_only.stencilfunc = xenos::CompareFunction::kAlways;
  CHECK(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k4X, true,
      false, true, stencil_only));

  // Rejects when any non-clear condition is present.
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k4X, true,
      true, true, depth));
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, true, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k4X, true,
      false, true, depth));
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k4X, false,
      false, true, depth));
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k1X, true,
      false, true, depth));
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kTriangleList, xenos::MsaaSamples::k4X, true,
      false, true, depth));
  depth.zfunc = xenos::CompareFunction::kGreaterEqual;
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k4X, true,
      false, true, depth));
  depth.zfunc = xenos::CompareFunction::kAlways;
  depth.stencilfunc = xenos::CompareFunction::kEqual;
  CHECK_FALSE(native::IsAliased4xColorDepthClear(
      true, true, false, xenos::PrimitiveType::kRectangleList, xenos::MsaaSamples::k4X, true,
      false, true, depth));

  // 64bpp 4x->1x pitch tile parity and alias scale.
  CHECK(xenos::GetSurfacePitchTiles(640, xenos::MsaaSamples::k4X, true) ==
        xenos::GetSurfacePitchTiles(1280, xenos::MsaaSamples::k1X, true));
  const auto scale_1x = native::GetMsaa4xAliasScale(xenos::MsaaSamples::k1X);
  CHECK(scale_1x.x == 2);
  CHECK(scale_1x.y == 2);
}

TEST_CASE("Native cached cropped resolve key and version gating", "[gpu][native]") {
  const native::ResolvedCropKey key{0x141EC000u, 64, 2, VK_FORMAT_R8G8B8A8_UNORM};
  native::ResolvedCropKey same = key;
  const native::ResolvedCropKeyHash hasher;
  CHECK(hasher(key) == hasher(same));
  same.height = 4;
  CHECK(hasher(key) != hasher(same));

  CHECK(native::ShouldUploadResolvedCrop(false, 0, 1));
  CHECK_FALSE(native::ShouldUploadResolvedCrop(true, 42, 42));
  CHECK(native::ShouldUploadResolvedCrop(true, 42, 43));
}

TEST_CASE("Native shared vertex page range tracking and invalidation", "[gpu][native]") {
  const auto empty = native::GetPhysicalPageRange(0x1000u, 0);
  CHECK(empty.first == 0u);
  CHECK(empty.second == 0u);

  const auto oob = native::GetPhysicalPageRange(native::kPhysicalMemorySize, 64);
  CHECK(oob.first == 0u);
  CHECK(oob.second == 0u);

  const auto span = native::GetPhysicalPageRange(0x1800u, 0x2000u);
  CHECK(span.first == 1u);
  CHECK(span.second == 3u);

  std::vector<uint64_t> valid(4, 0);
  native::SetPageRangeBits(valid, 1, 3);
  native::SetPageRangeBits(valid, 62, 5);
  native::ClearPageRangeBits(valid, 2, 1);

  uint32_t cursor = 0;
  uint32_t run_first = 0;
  uint32_t run_count = 0;
  REQUIRE(native::FindNextInvalidPageRun(valid, cursor, 70, run_first, run_count));
  CHECK(run_first == 0u);
  CHECK(run_count == 1u);
  REQUIRE(native::FindNextInvalidPageRun(valid, cursor, 70, run_first, run_count));
  CHECK(run_first == 2u);
  CHECK(run_count == 1u);
  REQUIRE(native::FindNextInvalidPageRun(valid, cursor, 70, run_first, run_count));
  CHECK(run_first == 4u);
  CHECK(run_count == 58u);
  REQUIRE(native::FindNextInvalidPageRun(valid, cursor, 70, run_first, run_count));
  CHECK(run_first == 67u);
  CHECK(run_count == 3u);
  CHECK_FALSE(native::FindNextInvalidPageRun(valid, cursor, 70, run_first, run_count));
}

TEST_CASE("Native invalidation widening respects protected texture pages", "[gpu][native]") {
  std::vector<uint64_t> protected_pages(4, 0);

  // Unprotected 256 KiB block widens a single 4 KiB page to the full 64-page block.
  const auto full_block = native::WidenInvalidationPageRange(protected_pages, 0x10000u, 4096u);
  CHECK(full_block.first == 0u);
  CHECK(full_block.second == 64u * 4096u);

  // Protected pages at 5 and 40 constrain page 16 to [6, 39].
  native::SetPageRangeBits(protected_pages, 5, 1);
  native::SetPageRangeBits(protected_pages, 40, 1);
  const auto bounded = native::WidenInvalidationPageRange(protected_pages, 0x10000u, 4096u);
  CHECK(bounded.first == 6u * 4096u);
  CHECK(bounded.second == 34u * 4096u);

  // Adjacent protected pages at 15 and 17 constrain page 16 to [16, 16].
  native::SetPageRangeBits(protected_pages, 15, 1);
  native::SetPageRangeBits(protected_pages, 17, 1);
  const auto tight = native::WidenInvalidationPageRange(protected_pages, 0x10000u, 4096u);
  CHECK(tight.first == 16u * 4096u);
  CHECK(tight.second == 4096u);

  // Protected page on the invalidated page itself (16) with neighbor at 17 still includes 16.
  std::fill(protected_pages.begin(), protected_pages.end(), 0);
  native::SetPageRangeBits(protected_pages, 16, 2);
  const auto self_protected = native::WidenInvalidationPageRange(protected_pages, 0x10000u, 4096u);
  CHECK(self_protected.first == 0u);
  CHECK(self_protected.second == 17u * 4096u);

  // Empty protected_pages leaves the range page-aligned without widening.
  const std::vector<uint64_t> empty_protected;
  const auto no_widen = native::WidenInvalidationPageRange(empty_protected, 0x10000u, 4096u);
  CHECK(no_widen.first == 0x10000u);
  CHECK(no_widen.second == 4096u);

  // End of 512 MiB physical memory clamps safely without overflow.
  const auto end_clamped =
      native::WidenInvalidationPageRange(empty_protected, native::kPhysicalMemorySize - 4096u, 4096u);
  CHECK(end_clamped.first == native::kPhysicalMemorySize - 4096u);
  CHECK(end_clamped.second == 4096u);

  // Union mask across multiple bitmaps stops at the tightest protected boundary
  // even when another bitmap (e.g., shader pages) is all zeros in that block.
  std::vector<uint64_t> tex_pages(4, 0);
  std::vector<uint64_t> idx_pages(4, 0);
  std::vector<uint64_t> shader_pages(4, 0);
  native::AtomicSetPageRangeBits(tex_pages, 10, 1);
  native::AtomicSetPageRangeBits(idx_pages, 20, 1);
  const auto union_widened = native::WidenInvalidationPageRangeWithMask(
      [&](uint32_t b) { return tex_pages[b] | idx_pages[b] | shader_pages[b]; }, 4u, 16u * 4096u,
      4096u);
  CHECK(union_widened.first == 11u * 4096u);
  CHECK(union_widened.second == 9u * 4096u);
}

TEST_CASE("Native atomic watched page run finder skips already-watched pages", "[gpu][native]") {
  std::vector<uint64_t> watched(4, 0);
  native::AtomicSetPageRangeBits(watched, 62, 4);  // spans word 0->1 boundary
  native::AtomicClearPageRangeBits(watched, 63, 1);
  uint32_t cursor = 60;
  uint32_t run_first = 0;
  uint32_t run_count = 0;
  REQUIRE(native::AtomicFindNextUnsetPageRun(watched, cursor, 70, run_first, run_count));
  CHECK(run_first == 60u);
  CHECK(run_count == 2u);
  REQUIRE(native::AtomicFindNextUnsetPageRun(watched, cursor, 70, run_first, run_count));
  CHECK(run_first == 63u);
  CHECK(run_count == 1u);
  REQUIRE(native::AtomicFindNextUnsetPageRun(watched, cursor, 70, run_first, run_count));
  CHECK(run_first == 66u);
  CHECK(run_count == 4u);
  CHECK_FALSE(native::AtomicFindNextUnsetPageRun(watched, cursor, 70, run_first, run_count));
}

TEST_CASE("Native draw resolution scale rect and crop helpers", "[gpu][native]") {
  CHECK(native::CanCropAlignedResolve(3840, 2160, 3840 - 3, 2160 - 3, 3, 3));
  CHECK_FALSE(native::CanCropAlignedResolve(3840, 2160, 3840 - 24, 2160, 3, 3));
  CHECK(native::CanUseResolvedCubeFace(true, 384, 384, 384 - 9, 384 - 9, 3, 3));
  CHECK_FALSE(native::CanUseResolvedCubeFace(true, 384, 384, 384 - 24, 384, 3, 3));

  draw_util::ResolveInfo copy{};
  copy.color_original_base = 0;
  copy.color_edram_info.base_tiles = 0;
  copy.color_edram_info.pitch_tiles = 16;
  copy.color_edram_info.msaa_samples = xenos::MsaaSamples::k1X;
  copy.color_edram_info.format = uint32_t(xenos::ColorRenderTargetFormat::k_8_8_8_8);
  copy.coordinate_info.edram_offset_x_div_8 = 1;
  copy.coordinate_info.edram_offset_y_div_8 = 1;
  copy.coordinate_info.width_div_8 = 160;
  copy.height_div_8 = 90;
  copy.copy_dest_x = 16;
  copy.copy_dest_y = 24;

  const EdramLayout layout{};
  const auto rect = native::GetResolveRectangle(copy, layout, 3864, 2184, false, 3, 3);
  REQUIRE(rect.has_value());
  CHECK(rect->x == 24u);
  CHECK(rect->y == 24u);
  CHECK(rect->width == 3840u);
  CHECK(rect->height == 2160u);
  CHECK_FALSE(native::GetResolveRectangle(copy, layout, 3840, 2160, false, 3, 3).has_value());
  CHECK_FALSE(native::GetResolveRectangle(copy, layout, 3864, 2184, false, 4, 4).has_value());

  const auto cached = native::GetCachedResolveRectangle(copy, 3888, 2232, 3, 3);
  REQUIRE(cached.has_value());
  CHECK(cached->x == 48u);
  CHECK(cached->y == 72u);
  CHECK(cached->width == 3840u);
  CHECK(cached->height == 2160u);
  CHECK_FALSE(native::GetCachedResolveRectangle(copy, 3840, 2160, 3, 3).has_value());

  const auto swap_720p =
      native::GetSwapOutputExtent({1280u, 720u}, VkExtent2D{1280u, 720u});
  CHECK(swap_720p.width == 1280u);
  CHECK(swap_720p.height == 720u);

  const auto swap_900p =
      native::GetSwapOutputExtent({1600u, 900u}, VkExtent2D{1600u, 904u});
  CHECK(swap_900p.width == 1600u);
  CHECK(swap_900p.height == 900u);

  const auto swap_1080p_padded =
      native::GetSwapOutputExtent({1920u, 1080u}, VkExtent2D{1920u, 1088u});
  CHECK(swap_1080p_padded.width == 1920u);
  CHECK(swap_1080p_padded.height == 1080u);

  const auto swap_1080p_small_fb =
      native::GetSwapOutputExtent({1920u, 1080u}, VkExtent2D{1280u, 720u});
  CHECK(swap_1080p_small_fb.width == 1280u);
  CHECK(swap_1080p_small_fb.height == 720u);

  const auto swap_1440p =
      native::GetSwapOutputExtent({2560u, 1440u}, VkExtent2D{2560u, 1440u});
  CHECK(swap_1440p.width == 2560u);
  CHECK(swap_1440p.height == 1440u);

  const auto swap_1800p =
      native::GetSwapOutputExtent({3200u, 1800u}, VkExtent2D{3200u, 1808u});
  CHECK(swap_1800p.width == 3200u);
  CHECK(swap_1800p.height == 1800u);

  const auto swap_4k =
      native::GetSwapOutputExtent({3840u, 2160u}, VkExtent2D{3840u, 2160u});
  CHECK(swap_4k.width == 3840u);
  CHECK(swap_4k.height == 2160u);

  const auto single_720p = native::GetResolveDestinationExtent(1280, 720, 1280, 736, 1, 1);
  CHECK(single_720p.width == 1280u);
  CHECK(single_720p.height == 720u);

  const auto single_crop = native::GetResolveDestinationExtent(504, 304, 512, 320, 1, 1);
  CHECK(single_crop.width == 504u);
  CHECK(single_crop.height == 304u);

  const auto tiled_1080p_2x = native::GetResolveDestinationExtent(544, 1080, 1920, 1088, 2, 2);
  CHECK(tiled_1080p_2x.width == 3840u);
  CHECK(tiled_1080p_2x.height == 2160u);

  const auto tiled_1080p_3x = native::GetResolveDestinationExtent(544, 1080, 1920, 1088, 3, 3);
  CHECK(tiled_1080p_3x.width == 5760u);
  CHECK(tiled_1080p_3x.height == 3240u);

  const auto tiled_horiz = native::GetResolveDestinationExtent(1920, 544, 1920, 1088, 1, 1);
  CHECK(tiled_horiz.width == 1920u);
  CHECK(tiled_horiz.height == 1088u);
  CHECK(native::CanCropAlignedResolve(1920, 1088, 1920, 1080, 1, 1,
                                      xenos::kTextureTileWidthHeight));
}

TEST_CASE("Native tiled resolve offset inverse matches GetTiledOffset2D", "[gpu][native]") {
  CHECK(native::GetResolveBytesPerBlockLog2(VK_FORMAT_R8_UNORM) == 0u);
  CHECK(native::GetResolveBytesPerBlockLog2(VK_FORMAT_R8G8_UNORM) == 1u);
  CHECK(native::GetResolveBytesPerBlockLog2(VK_FORMAT_R8G8B8A8_UNORM) == 2u);
  CHECK(native::GetResolveBytesPerBlockLog2(VK_FORMAT_A2B10G10R10_UNORM_PACK32) == 2u);
  CHECK(native::GetResolveBytesPerBlockLog2(VK_FORMAT_R32_SFLOAT) == 2u);
  CHECK(native::GetResolveBytesPerBlockLog2(VK_FORMAT_R16G16B16A16_SFLOAT) == 3u);

  const std::array<std::pair<uint32_t, uint32_t>, 7> coords{
      {{0, 0}, {32, 0}, {544, 0}, {1088, 0}, {1632, 0}, {0, 544}, {640, 320}}};
  for (uint32_t bpp_log2 = 0; bpp_log2 <= 3; ++bpp_log2) {
    for (uint32_t pitch : {1920u, 1904u}) {
      for (const auto& [expected_x, expected_y] : coords) {
        const uint32_t delta = uint32_t(texture_util::GetTiledOffset2D(
            int32_t(expected_x), int32_t(expected_y), pitch, bpp_log2));
        uint32_t x = 0;
        uint32_t y = 0;
        REQUIRE(native::TryGetTiledResolveOffset2D(delta, pitch, bpp_log2, x, y));
        CHECK(x == expected_x);
        CHECK(y == expected_y);
      }
    }
  }

  uint32_t x = 0;
  uint32_t y = 0;
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0u, 0u, 2u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0u, 1920u, 5u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0x1u, 1920u, 2u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0x100u, 1920u, 2u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0x800u, 1920u, 2u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0x200u, 1920u, 1u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0x400u, 1920u, 2u, x, y));
  CHECK_FALSE(native::TryGetTiledResolveOffset2D(0x1000u, 1920u, 3u, x, y));
}

TEST_CASE("Native tiled RGBA8 resolve readback scatters and bounds-checks guest memory",
          "[gpu][native]") {
  constexpr uint32_t kPitch = 64;
  const native::ResolveRectangle rect{8, 16, 4, 2};
  const uint32_t upper_bound =
      texture_util::GetTiledAddressUpperBound2D(rect.x + rect.width, rect.y + rect.height, kPitch, 2);
  REQUIRE(upper_bound > 0);

  std::vector<uint8_t> pixels(size_t(rect.width) * rect.height * 4);
  for (uint32_t y = 0; y < rect.height; ++y) {
    for (uint32_t x = 0; x < rect.width; ++x) {
      const size_t idx = (size_t(y) * rect.width + x) * 4;
      pixels[idx + 0] = uint8_t(0x10 + x);
      pixels[idx + 1] = uint8_t(0x20 + y);
      pixels[idx + 2] = uint8_t(0x30 + x + y);
      pixels[idx + 3] = 0xAA;
    }
  }

  std::vector<uint8_t> guest(upper_bound + 16, 0xCC);
  REQUIRE(native::StoreTiledResolveRgba8(guest, kPitch, rect, xenos::Endian128::k8in32, pixels));
  for (uint32_t y = 0; y < rect.height; ++y) {
    for (uint32_t x = 0; x < rect.width; ++x) {
      const size_t tiled_off = size_t(
          texture_util::GetTiledOffset2D(int32_t(rect.x + x), int32_t(rect.y + y), kPitch, 2));
      uint32_t stored = 0;
      std::memcpy(&stored, guest.data() + tiled_off, sizeof(stored));
      const size_t idx = (size_t(y) * rect.width + x) * 4;
      uint32_t expected = 0;
      std::memcpy(&expected, pixels.data() + idx, sizeof(expected));
      expected = xenos::GpuSwap(expected, xenos::Endian::k8in32);
      CHECK(stored == expected);
    }
  }
  for (size_t i = upper_bound; i < guest.size(); ++i) {
    CHECK(guest[i] == 0xCC);
  }

  // A 1-byte-short buffer must be rejected without modifying any guest byte.
  std::vector<uint8_t> short_guest(upper_bound - 1, 0x55);
  CHECK_FALSE(
      native::StoreTiledResolveRgba8(short_guest, kPitch, rect, xenos::Endian128::k8in32, pixels));
  for (uint8_t byte : short_guest) {
    CHECK(byte == 0x55);
  }

  // Out-of-range pitch, rect, endian, and pixel span sizes must be rejected.
  CHECK_FALSE(
      native::StoreTiledResolveRgba8(guest, 0, rect, xenos::Endian128::k8in32, pixels));
  CHECK_FALSE(native::StoreTiledResolveRgba8(guest, kPitch, {kPitch, 0, 4, 2},
                                             xenos::Endian128::k8in32, pixels));
  CHECK_FALSE(native::StoreTiledResolveRgba8(guest, kPitch, rect, xenos::Endian128(4), pixels));
  CHECK_FALSE(native::StoreTiledResolveRgba8(guest, kPitch, rect, xenos::Endian128::k8in32,
                                             std::span(pixels).first(pixels.size() - 4)));
}

TEST_CASE("Native portability texture swizzles pack for shader fallback", "[gpu][native]") {
  const uint32_t host_565 =
      native::MakeHostTextureSwizzle(xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 3, true);
  // k_5_6_5 swaps R/B and replicates the 3rd channel into A: B(2), G(1), R(0), R(0).
  CHECK(host_565 == (2u | (1u << 3) | (0u << 6) | (0u << 9)));

  const auto view_swizzle = native::MakeTextureComponentMappingFromHostSwizzle(host_565, true);
  CHECK(view_swizzle.r == VK_COMPONENT_SWIZZLE_B);
  CHECK(view_swizzle.g == VK_COMPONENT_SWIZZLE_G);
  CHECK(view_swizzle.b == VK_COMPONENT_SWIZZLE_R);
  CHECK(view_swizzle.a == VK_COMPONENT_SWIZZLE_R);

  const auto identity_swizzle = native::MakeTextureComponentMappingFromHostSwizzle(host_565, false);
  CHECK(identity_swizzle.r == VK_COMPONENT_SWIZZLE_IDENTITY);
  CHECK(identity_swizzle.g == VK_COMPONENT_SWIZZLE_IDENTITY);
  CHECK(identity_swizzle.b == VK_COMPONENT_SWIZZLE_IDENTITY);
  CHECK(identity_swizzle.a == VK_COMPONENT_SWIZZLE_IDENTITY);

  uint32_t words[16]{};
  native::PackSystemTextureSwizzle(words, 0, 0x688u);
  native::PackSystemTextureSwizzle(words, 1, host_565);
  CHECK((words[0] & 0xFFFu) == 0x688u);
  CHECK(((words[0] >> 12) & 0xFFFu) == host_565);
}

TEST_CASE("Native depth bias, user clip planes, and clear helpers", "[gpu][native]") {
  const float slope = 2.0f;
  const float offset = 0.0005f;
  const auto bias_unorm = native::ScaleHostDepthBias(
      slope, offset, xenos::DepthRenderTargetFormat::kD24S8, 2, 3);
  CHECK(bias_unorm.constant_factor ==
        Catch::Approx(offset * draw_util::kD3D10PolygonOffsetFactorUnorm24));
  CHECK(bias_unorm.slope_factor ==
        Catch::Approx(slope * xenos::kPolygonOffsetScaleSubpixelUnit * 3.0f));

  const auto bias_float = native::ScaleHostDepthBias(
      slope, offset, xenos::DepthRenderTargetFormat::kD24FS8, 1, 1);
  CHECK(bias_float.constant_factor ==
        Catch::Approx(offset * draw_util::kD3D10PolygonOffsetFactorFloat24));

  reg::PA_CL_CLIP_CNTL clip{};
  clip.ucp_ena = 0b001101;
  const auto ucp_clip = native::GetUserClipPlaneConfig(clip, true, true);
  CHECK(ucp_clip.enabled_mask == 0b001101u);
  CHECK(ucp_clip.count == 3u);
  CHECK_FALSE(ucp_clip.cull_only);

  const auto ucp_fallback_cull = native::GetUserClipPlaneConfig(clip, false, true);
  CHECK(ucp_fallback_cull.count == 3u);
  CHECK(ucp_fallback_cull.cull_only);

  clip.clip_disable = 1;
  CHECK(native::GetUserClipPlaneConfig(clip, true, true).count == 0u);

  CHECK(native::CoversFullExtent({{0, 0}, {1280, 720}}, {1280, 720}));
  CHECK_FALSE(native::CoversFullExtent({{0, 0}, {1280, 720}}, {1280, 2048}));
  CHECK_FALSE(native::CoversFullExtent({{8, 0}, {1272, 720}}, {1280, 720}));

  std::vector<uint32_t> scratch{99, 100};
  native::FillQuadStripIndices(2, scratch);
  CHECK(scratch == std::vector<uint32_t>{0, 1, 2, 3, UINT32_MAX, 4, 5, 6, 7});

  CHECK(native::ClampResolutionScaleAxis(3, 1920, 4096) == 2u);
  CHECK(native::ClampResolutionScaleAxis(3, 1280, 4096) == 3u);
}

TEST_CASE("Native dynamic constant buffers stay in bounds", "[gpu][native]") {
  const std::array<uint64_t, 3> offsets{0, 256, 512};
  const std::array<uint64_t, 3> ranges{100, 4096, 16};
  // The float constants at 256 bind 4096 bytes, past the 600 bytes of data.
  CHECK(native::ComputeDynamicUboRequiredSize(offsets, ranges, 600) == 256 + 4096);
  // Already large enough: unchanged.
  CHECK(native::ComputeDynamicUboRequiredSize(offsets, ranges, 8192) == 8192);
  CHECK(native::ComputeDynamicUboRequiredSize({}, {}, 64) == 64);
}

TEST_CASE("Native sparse vertex granules cover committed ranges", "[gpu][native]") {
  constexpr uint64_t kMiB = 1024 * 1024;
  CHECK(native::ComputeSparseGranuleSize(64 * 1024) == 4 * kMiB);
  CHECK(native::ComputeSparseGranuleSize(0) == 4 * kMiB);
  // Rounded up to a multiple of an unusual alignment.
  CHECK(native::ComputeSparseGranuleSize(3 * kMiB) == 6 * kMiB);
  CHECK(native::ComputeSparseGranuleSize(8 * kMiB) == 8 * kMiB);
  // At most 128 granules for the 512 MiB buffer.
  CHECK(native::kPhysicalMemorySize / native::ComputeSparseGranuleSize(64 * 1024) == 128);

  using Range = std::pair<uint32_t, uint32_t>;
  CHECK(native::GetSparseGranuleRange(0, 1, 4 * kMiB) == Range{0, 1});
  CHECK(native::GetSparseGranuleRange(4 * kMiB - 1, 2, 4 * kMiB) == Range{0, 2});
  CHECK(native::GetSparseGranuleRange(8 * kMiB, 4 * kMiB, 4 * kMiB) == Range{2, 1});
  CHECK(native::GetSparseGranuleRange(8 * kMiB, 4 * kMiB + 1, 4 * kMiB) == Range{2, 2});
  CHECK(native::GetSparseGranuleRange(0, 0, 4 * kMiB) == Range{0, 0});
}

TEST_CASE("Native draw state tracking elides redundant binds", "[gpu][native]") {
  const auto handle = [](auto type, uintptr_t value) {
    return reinterpret_cast<decltype(type)>(value);
  };
  native::DrawRenderState state;
  const VkPipeline pipeline_a = handle(VkPipeline{}, 0x10);
  const VkPipeline pipeline_b = handle(VkPipeline{}, 0x20);
  CHECK(state.SetPipeline(pipeline_a));
  CHECK_FALSE(state.SetPipeline(pipeline_a));
  CHECK(state.SetPipeline(pipeline_b));

  const VkPipelineLayout layout_a = handle(VkPipelineLayout{}, 0x30);
  const VkPipelineLayout layout_b = handle(VkPipelineLayout{}, 0x40);
  std::array<VkDescriptorSet, 4> sets{handle(VkDescriptorSet{}, 1), handle(VkDescriptorSet{}, 2),
                                      handle(VkDescriptorSet{}, 3), handle(VkDescriptorSet{}, 4)};
  std::array<uint32_t, 5> offsets{0, 256, 512, 768, 1024};
  // The first bind, and any pipeline layout change, rebind every set.
  CHECK(state.FirstChangedSet(layout_a, sets, offsets) == 0);
  CHECK(state.FirstChangedSet(layout_a, sets, offsets) == 4);
  sets[3] = handle(VkDescriptorSet{}, 5);
  CHECK(state.FirstChangedSet(layout_a, sets, offsets) == 3);
  // New dynamic offsets rebind the constant set (1) and everything after it.
  offsets[2] = 1280;
  CHECK(state.FirstChangedSet(layout_a, sets, offsets) == 1);
  sets[0] = handle(VkDescriptorSet{}, 6);
  CHECK(state.FirstChangedSet(layout_a, sets, offsets) == 0);
  CHECK(state.FirstChangedSet(layout_b, sets, offsets) == 0);
  CHECK(state.FirstChangedSet(layout_b, sets, offsets) == 4);

  const VkViewport viewport{0.0f, 0.0f, 1280.0f, 720.0f, 0.0f, 1.0f};
  CHECK(native::DrawRenderState::Update(state.viewport_valid, state.viewport, viewport));
  CHECK_FALSE(native::DrawRenderState::Update(state.viewport_valid, state.viewport, viewport));
  const VkViewport smaller{0.0f, 0.0f, 640.0f, 360.0f, 0.0f, 1.0f};
  CHECK(native::DrawRenderState::Update(state.viewport_valid, state.viewport, smaller));
  const std::array<float, 4> blend{0.5f, 0.5f, 0.5f, 1.0f};
  CHECK(native::DrawRenderState::Update(state.blend_constants_valid, state.blend_constants, blend));
  CHECK_FALSE(
      native::DrawRenderState::Update(state.blend_constants_valid, state.blend_constants, blend));

  CHECK(native::DrawRenderState::UpdateStencil(state.stencil_reference,
                                               state.stencil_reference_valid, {1, 1}) == 3);
  CHECK(native::DrawRenderState::UpdateStencil(state.stencil_reference,
                                               state.stencil_reference_valid, {1, 1}) == 0);
  CHECK(native::DrawRenderState::UpdateStencil(state.stencil_reference,
                                               state.stencil_reference_valid, {1, 2}) == 2);
  CHECK(native::DrawRenderState::UpdateStencil(state.stencil_reference,
                                               state.stencil_reference_valid, {3, 2}) == 1);

  const VkBuffer buffer = handle(VkBuffer{}, 0x50);
  CHECK(state.SetIndexBuffer(buffer, 64, VK_INDEX_TYPE_UINT32));
  CHECK_FALSE(state.SetIndexBuffer(buffer, 64, VK_INDEX_TYPE_UINT32));
  CHECK(state.SetIndexBuffer(buffer, 128, VK_INDEX_TYPE_UINT32));
  CHECK(state.SetIndexBuffer(buffer, 128, VK_INDEX_TYPE_UINT16));
}

TEST_CASE("Native samplers normalize guest address, anisotropy and filtering", "[gpu][native]") {
  using xenos::ClampMode;
  CHECK(native::GetSamplerAddressMode(ClampMode::kRepeat, false) == VK_SAMPLER_ADDRESS_MODE_REPEAT);
  CHECK(native::GetSamplerAddressMode(ClampMode::kClampToHalfway, true) ==
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  CHECK(native::GetSamplerAddressMode(ClampMode::kClampToBorder, false) ==
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
  CHECK(native::GetSamplerAddressMode(ClampMode::kMirrorClampToBorder, true) ==
        VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE);
  // Without samplerMirrorClampToEdge, mirrored clamps fall back to mirrored repeat.
  CHECK(native::GetSamplerAddressMode(ClampMode::kMirrorClampToEdge, false) ==
        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);

  using xenos::AnisoFilter;
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kMax_4_1, -1, false, 16.0f) == 4.0f);
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kDisabled, -1, true, 16.0f) == 0.0f);
  // anisotropic_override applies only to eligible (mipmapped linear) samplers.
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kDisabled, 5, true, 16.0f) == 16.0f);
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kDisabled, 5, false, 16.0f) == 0.0f);
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kMax_16_1, 0, true, 16.0f) == 0.0f);
  // The device limit caps it, and no anisotropy support disables it.
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kMax_16_1, -1, false, 8.0f) == 8.0f);
  CHECK(native::GetSamplerAnisotropy(AnisoFilter::kMax_16_1, -1, false, 0.0f) == 0.0f);

  native::DrawSampler sampler;
  sampler.min_filter = sampler.mag_filter = VK_FILTER_LINEAR;
  sampler.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler.max_anisotropy = 4.0f;
  sampler.border_color = xenos::BorderColor::k_ABGR_White;
  const auto unfilterable = native::NormalizeDrawSampler(sampler, false);
  CHECK(unfilterable.min_filter == VK_FILTER_NEAREST);
  CHECK(unfilterable.mag_filter == VK_FILTER_NEAREST);
  CHECK(unfilterable.mip_mode == VK_SAMPLER_MIPMAP_MODE_NEAREST);
  CHECK(unfilterable.max_anisotropy == 0.0f);
  // Without a border address mode the color is irrelevant and dropped from the key.
  CHECK(unfilterable.border_color == xenos::BorderColor::k_ABGR_Black);

  native::DrawSampler anisotropic;
  anisotropic.max_anisotropy = 2.0f;
  anisotropic.address_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  anisotropic.border_color = xenos::BorderColor::k_ACBYCR_Black;
  const auto forced = native::NormalizeDrawSampler(anisotropic, true);
  CHECK(forced.min_filter == VK_FILTER_LINEAR);
  CHECK(forced.mag_filter == VK_FILTER_LINEAR);
  CHECK(forced.mip_mode == VK_SAMPLER_MIPMAP_MODE_LINEAR);
  CHECK(forced.border_color == xenos::BorderColor::k_ACBYCR_Black);

  // The new fields are part of the cache key and its hash.
  native::DrawSampler other = forced;
  other.min_lod = 1.0f;
  CHECK(other != forced);
  CHECK(native::DrawSamplerHash{}(other) != native::DrawSamplerHash{}(forced));
}

TEST_CASE("Native texture unpackers match the Xenos load shaders", "[gpu][native]") {
  CHECK(native::ExpandUnorm16(0x1F, 5) == 0xFFFF);
  CHECK(native::ExpandUnorm16(0x10, 5) == 0x8421);
  CHECK(native::ExpandUnorm16(0x7FF, 11) == 0xFFFF);
  CHECK(native::ExpandUnorm16(0x200, 10) == 0x8020);
  // Both most negative values are -1.0, and the extremes map to +-32767.
  CHECK(native::ExpandSnorm16(0x3FF, 11) == 0x7FFF);
  CHECK(native::ExpandSnorm16(0x401, 11) == 0x8001);
  CHECK(native::ExpandSnorm16(0x400, 11) == 0x8001);
  CHECK(native::ExpandSnorm16(0x1FF, 10) == 0x7FFF);
  CHECK(native::ExpandSnorm16(0x200, 10) == 0x8001);
  CHECK(native::ExpandSnorm16(0, 11) == 0);

  CHECK(native::UnpackRgba4ToRgba8(0xF0A5) == 0xFF00AA55u);
  CHECK(native::UnpackRgb5a1ToRgba8(0x801F) == 0xFF0000FFu);
  CHECK(native::UnpackRgb5a1ToRgba8(0x7C00) == 0x00FF0000u);

  using Rgba16 = std::array<uint16_t, 4>;
  // k_10_11_11: R11 G11 B10 from the low bits.
  CHECK(native::UnpackR11G11B10ToRgba16(0x7FFu | (0x3FFu << 22), false, false) ==
        Rgba16{0xFFFF, 0, 0xFFFF, 0xFFFF});
  CHECK(native::UnpackR11G11B10ToRgba16(0x3FFu, false, true) == Rgba16{0x7FFF, 0, 0, 0x7FFF});
  // k_11_11_10: R10 G11 B11.
  CHECK(native::UnpackR11G11B10ToRgba16(0x3FFu | (0x7FFu << 21), true, false) ==
        Rgba16{0xFFFF, 0, 0xFFFF, 0xFFFF});
  CHECK(native::UnpackR11G11B10ToRgba16(0x200u, true, true) == Rgba16{0x8001, 0, 0, 0x7FFF});

  const auto alpha = native::DecodeDxt3aBlock(0x100000000000000Full);
  CHECK(alpha[0] == 0xFF);
  CHECK(alpha[1] == 0);
  CHECK(alpha[15] == 0x11);

  // Endpoints (0, 30) and (90, 60); texels 0-3 select 0, 1, 2:1 and 1:2.
  const uint64_t ctx1 = uint64_t(30u << 8 | 90u << 16 | 60u << 24) | (uint64_t(0xE4) << 32);
  const auto rg = native::DecodeCtx1Block(ctx1);
  CHECK(rg[0] == 0x1E00);
  CHECK(rg[1] == 0x3C5A);
  CHECK(rg[2] == 0x281E);
  CHECK(rg[3] == 0x323C);
  CHECK(rg[4] == 0x1E00);
}

TEST_CASE("Native texture formats select host formats and signed views", "[gpu][native]") {
  using xenos::TextureFormat;
  CHECK(native::TextureChannelCount(TextureFormat::k_16) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_DXT3A) == 1);
  CHECK(native::TextureChannelCount(TextureFormat::k_16_16) == 2);
  CHECK(native::TextureChannelCount(TextureFormat::k_CTX1) == 2);
  CHECK(native::TextureChannelCount(TextureFormat::k_10_11_11) == 3);
  CHECK(native::TextureChannelCount(TextureFormat::k_11_11_10_AS_16_16_16_16) == 3);
  CHECK(native::TextureChannelCount(TextureFormat::k_32_32_32_32_FLOAT) == 4);

  const auto rgba8 = native::GetTextureFormatInfo(TextureFormat::k_8_8_8_8);
  CHECK(rgba8.format == VK_FORMAT_R8G8B8A8_UNORM);
  CHECK(rgba8.signed_format == VK_FORMAT_R8G8B8A8_SNORM);
  CHECK(rgba8.needs_signed_view());
  CHECK(rgba8.host_block_size() == 4);

  const auto dxn = native::GetTextureFormatInfo(TextureFormat::k_DXN);
  CHECK(dxn.signed_format == VK_FORMAT_BC5_SNORM_BLOCK);
  CHECK(dxn.block_width == 4);
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_DXT5A).signed_format ==
        VK_FORMAT_BC4_SNORM_BLOCK);

  // Signed 10_11_11 is a differently unpacked image, not a view.
  const auto packed = native::GetTextureFormatInfo(TextureFormat::k_10_11_11);
  CHECK(packed.unpack_depends_on_sign());
  CHECK_FALSE(packed.needs_signed_view());
  CHECK(packed.host_block_size() == 8);
  CHECK(packed.signed_format == VK_FORMAT_R16G16B16A16_SNORM);

  const auto dxt3a = native::GetTextureFormatInfo(TextureFormat::k_DXT3A);
  CHECK(dxt3a.decompresses_blocks());
  CHECK(dxt3a.host_block_size() == 1);
  CHECK(dxt3a.signed_format == VK_FORMAT_UNDEFINED);

  CHECK(native::GetTextureFormatInfo(TextureFormat::k_4_4_4_4).host_block_size() == 4);
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_24_8).unpack ==
        native::TextureUnpack::kDepthUnorm24);
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_1).format == VK_FORMAT_UNDEFINED);
  CHECK(native::GetTextureFormatInfo(TextureFormat::k_32_32_32_FLOAT).format ==
        VK_FORMAT_UNDEFINED);

  using xenos::DataDimension;
  using xenos::FetchOpDimension;
  CHECK(native::TextureBindingMatchesDimension(FetchOpDimension::k2D, DataDimension::k1D));
  CHECK(native::TextureBindingMatchesDimension(FetchOpDimension::k2D, DataDimension::k2DOrStacked));
  CHECK_FALSE(native::TextureBindingMatchesDimension(FetchOpDimension::k2D, DataDimension::k3D));
  CHECK(native::TextureBindingMatchesDimension(FetchOpDimension::k3DOrStacked, DataDimension::k3D));
  CHECK_FALSE(native::TextureBindingMatchesDimension(FetchOpDimension::k3DOrStacked,
                                                     DataDimension::k2DOrStacked));
  CHECK(native::TextureBindingMatchesDimension(FetchOpDimension::kCube, DataDimension::kCube));
  CHECK_FALSE(native::TextureBindingMatchesDimension(FetchOpDimension::kCube,
                                                     DataDimension::k2DOrStacked));

  // Keys for the signed unpack and the mutable image differ from the plain one.
  native::GuestTextureKey key;
  native::GuestTextureKey signed_key = key;
  signed_key.is_signed = true;
  native::GuestTextureKey view_key = key;
  view_key.signed_view = true;
  CHECK(key != signed_key);
  CHECK(key != view_key);
  CHECK(native::GuestTextureKeyHash{}(key) != native::GuestTextureKeyHash{}(view_key));
}

TEST_CASE("Native EDRAM tile overlaps become image copies", "[gpu][native]") {
  using Copy = native::EdramTileCopy;
  // Bases a whole number of rows apart: one rectangle.
  CHECK(native::ComputeOverlappingEdramTileCopies(0, 16, 16, 16, 48) ==
        std::vector<Copy>{{0, 1, 0, 0, 16, 2}});
  // Half a row apart: the runs wrap at different columns.
  CHECK(native::ComputeOverlappingEdramTileCopies(0, 8, 16, 8, 40) ==
        std::vector<Copy>{{8, 0, 0, 0, 8, 1},
                          {0, 1, 8, 0, 8, 1},
                          {8, 1, 0, 1, 8, 1},
                          {0, 2, 8, 1, 8, 1}});
  // A partial last row after full rows.
  CHECK(native::ComputeOverlappingEdramTileCopies(32, 0, 16, 32, 52) ==
        std::vector<Copy>{{0, 0, 0, 2, 16, 1}, {0, 1, 0, 3, 4, 1}});
  CHECK(native::ComputeOverlappingEdramTileCopies(0, 16, 16, 8, 32).empty());
  CHECK(native::ComputeOverlappingEdramTileCopies(0, 0, 0, 0, 16).empty());
  CHECK(native::ComputeOverlappingEdramTileCopies(0, 0, 16, 16, 16).empty());

  using xenos::MsaaSamples;
  CHECK(native::GetEdramTileHostExtent(MsaaSamples::k1X, false, 1, 1).width == 80);
  CHECK(native::GetEdramTileHostExtent(MsaaSamples::k1X, false, 1, 1).height == 16);
  CHECK(native::GetEdramTileHostExtent(MsaaSamples::k2X, false, 1, 1).height == 8);
  CHECK(native::GetEdramTileHostExtent(MsaaSamples::k4X, false, 1, 1).width == 40);
  CHECK(native::GetEdramTileHostExtent(MsaaSamples::k1X, true, 1, 1).width == 40);
  const auto scaled = native::GetEdramTileHostExtent(MsaaSamples::k4X, true, 2, 3);
  CHECK(scaled.width == 40);
  CHECK(scaled.height == 24);
}

TEST_CASE("Native pipeline cache files are validated before merging", "[gpu][native]") {
  const std::array<uint8_t, VK_UUID_SIZE> uuid{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const std::vector<uint8_t> data{0xDE, 0xAD, 0xBE, 0xEF, 0x01};
  const auto header = native::MakePipelineCacheFileHeader(0x10DE, 0x2684, uuid, data);
  std::vector<uint8_t> file(sizeof(header));
  std::memcpy(file.data(), &header, sizeof(header));
  file.insert(file.end(), data.begin(), data.end());

  const auto stored = native::GetPipelineCacheFileData(file, 0x10DE, 0x2684, uuid);
  CHECK(std::vector<uint8_t>(stored.begin(), stored.end()) == data);
  // Another device or driver cache version.
  CHECK(native::GetPipelineCacheFileData(file, 0x1002, 0x2684, uuid).empty());
  CHECK(native::GetPipelineCacheFileData(file, 0x10DE, 0x2685, uuid).empty());
  auto other_uuid = uuid;
  other_uuid[0] ^= 1;
  CHECK(native::GetPipelineCacheFileData(file, 0x10DE, 0x2684, other_uuid).empty());
  // Truncated or corrupted.
  CHECK(native::GetPipelineCacheFileData(std::span(file).first(file.size() - 1), 0x10DE, 0x2684,
                                         uuid)
            .empty());
  CHECK(native::GetPipelineCacheFileData(std::span(file).first(10), 0x10DE, 0x2684, uuid).empty());
  auto corrupt = file;
  corrupt.back() ^= 0xFF;
  CHECK(native::GetPipelineCacheFileData(corrupt, 0x10DE, 0x2684, uuid).empty());
}

TEST_CASE("Native color and depth images include transfer-source usage", "[gpu][native]") {
  const VkImageUsageFlags color_usage =
      native::GetColorImageUsage(VK_IMAGE_ASPECT_COLOR_BIT, true);
  CHECK((color_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0);
  CHECK((color_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0);
  CHECK((color_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0);
  CHECK((color_usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0);

  const VkImageUsageFlags depth_usage =
      native::GetColorImageUsage(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false);
  CHECK((depth_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0);
  CHECK((depth_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0);
  CHECK((depth_usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0);
  CHECK((depth_usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0);
}

TEST_CASE("Native inline 2D tiling and block swap match reference", "[gpu][native]") {
  for (uint32_t bpb_log2 = 0; bpb_log2 <= 4; ++bpb_log2) {
    for (uint32_t pitch : {32u, 64u, 96u, 160u}) {
      for (int32_t y : {0, 1, 7, 15, 16, 31, 32, 47}) {
        for (int32_t x : {0, 1, 3, 7, 8, 15, 31, 32, 63}) {
          CHECK(native::GetTiledOffset2DInline(x, y, pitch, bpb_log2) ==
                texture_util::GetTiledOffset2D(x, y, pitch, bpb_log2));
        }
      }
    }
  }

  std::array<uint8_t, 32> src{};
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = uint8_t(i + 1);
  }
  auto check_swap = [&](auto tag_bs, auto tag_sw, uint32_t offset) {
    constexpr uint32_t bs = decltype(tag_bs)::value;
    constexpr size_t sw = decltype(tag_sw)::value;
    std::array<uint8_t, bs> got{};
    std::array<uint8_t, bs> ref{};
    native::CopySwapBlock<bs, sw>(src.data(), offset, got.data());
    for (uint32_t i = 0; i < bs; ++i) {
      ref[i] = src[(offset + i) ^ sw];
    }
    CHECK(got == ref);
  };
  for (size_t off1 = 0; off1 < 8; ++off1) {
    check_swap(std::integral_constant<uint32_t, 1>{}, std::integral_constant<size_t, 0>{}, off1);
    check_swap(std::integral_constant<uint32_t, 1>{}, std::integral_constant<size_t, 1>{}, off1);
    check_swap(std::integral_constant<uint32_t, 1>{}, std::integral_constant<size_t, 2>{}, off1);
    check_swap(std::integral_constant<uint32_t, 1>{}, std::integral_constant<size_t, 3>{}, off1);
  }
  for (uint32_t off2 : {0u, 2u, 4u, 6u}) {
    check_swap(std::integral_constant<uint32_t, 2>{}, std::integral_constant<size_t, 0>{}, off2);
    check_swap(std::integral_constant<uint32_t, 2>{}, std::integral_constant<size_t, 1>{}, off2);
    check_swap(std::integral_constant<uint32_t, 2>{}, std::integral_constant<size_t, 2>{}, off2);
    check_swap(std::integral_constant<uint32_t, 2>{}, std::integral_constant<size_t, 3>{}, off2);
  }
  for (uint32_t off4 : {0u, 4u}) {
    check_swap(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 0>{}, off4);
    check_swap(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 1>{}, off4);
    check_swap(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 2>{}, off4);
    check_swap(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 3>{}, off4);
    check_swap(std::integral_constant<uint32_t, 8>{}, std::integral_constant<size_t, 3>{}, off4);
    check_swap(std::integral_constant<uint32_t, 16>{}, std::integral_constant<size_t, 3>{}, off4);
  }
}

TEST_CASE("Native EDRAM tile interval splicing tracks ownership and splits", "[gpu][native]") {
  std::vector<native::TileRun> runs;
  struct Overwrite {
    uint64_t prev_owner;
    uint32_t start;
    uint32_t end;
  };
  std::vector<Overwrite> overwrites;
  auto claim = [&](uint32_t start, uint32_t end, uint64_t owner, uint32_t base,
                   bool scaled = true) {
    overwrites.clear();
    return native::SpliceTileRun(
        runs,
        native::TileRun{start, end, owner, base, 16, xenos::MsaaSamples::k1X,
                        VK_FORMAT_R8G8B8A8_UNORM, false, scaled},
        [&](const native::TileRun& prev, uint32_t s, uint32_t e) {
          overwrites.push_back({prev.owner_id, s, e});
        });
  };
  CHECK(claim(0, 100, 1, 0));
  CHECK(runs.size() == 1);
  // Same owner/attributes is a no-op.
  CHECK_FALSE(claim(10, 50, 1, 0));
  // Split middle with owner 2.
  CHECK(claim(20, 60, 2, 20));
  REQUIRE(runs.size() == 3);
  CHECK(runs[0].start == 0);
  CHECK(runs[0].end == 20);
  CHECK(runs[0].owner_id == 1);
  CHECK(runs[1].start == 20);
  CHECK(runs[1].end == 60);
  CHECK(runs[1].owner_id == 2);
  CHECK(runs[2].start == 60);
  CHECK(runs[2].end == 100);
  CHECK(runs[2].owner_id == 1);
  REQUIRE(overwrites.size() == 1);
  CHECK(overwrites[0].prev_owner == 1);
  CHECK(overwrites[0].start == 20);
  CHECK(overwrites[0].end == 60);

  // Reclaim across the gap with owner 1, base 0 -> merges all 3 back into 1!
  CHECK(claim(20, 60, 1, 0));
  CHECK(runs.size() == 1);
  CHECK(runs[0].start == 0);
  CHECK(runs[0].end == 100);
  CHECK(runs[0].owner_id == 1);
}

TEST_CASE("Native draw_resolution_scale_threshold classifies pitches", "[gpu][native]") {
  using xenos::MsaaSamples;
  CHECK_FALSE(native::IsDrawScaleNativeForPitch(640, MsaaSamples::k1X, 0, 2, 2));
  CHECK_FALSE(native::IsDrawScaleNativeForPitch(640, MsaaSamples::k1X, 640, 1, 1));
  CHECK(native::IsDrawScaleNativeForPitch(640, MsaaSamples::k1X, 640, 2, 2));
  CHECK(native::IsDrawScaleNativeForPitch(640, MsaaSamples::k2X, 640, 3, 3));
  CHECK_FALSE(native::IsDrawScaleNativeForPitch(1280, MsaaSamples::k1X, 640, 2, 2));
  CHECK_FALSE(native::IsDrawScaleNativeForPitch(641, MsaaSamples::k1X, 640, 2, 2));
}

TEST_CASE("Native chunked tiled texture decode matches byte-by-byte reference", "[gpu][native]") {
  auto run_case = [&](auto tag_bs, auto tag_sw, uint32_t pitch, uint32_t sx, uint32_t sy,
                      uint32_t w, uint32_t h) {
    constexpr uint32_t bs = decltype(tag_bs)::value;
    constexpr size_t sw = decltype(tag_sw)::value;
    constexpr uint32_t bpb_log2 = uint32_t(std::countr_zero(bs));
    const uint32_t pitch_aligned = (pitch + 31u) & ~31u;
    const size_t upper =
        size_t(texture_util::GetTiledAddressUpperBound2D(sx + w, sy + h, pitch_aligned, bpb_log2));
    std::vector<uint8_t> src(upper + 64);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i] = uint8_t((i * 131 + 17) & 0xFF);
    }
    const size_t dst_bytes = size_t(w) * h * bs;
    std::vector<uint8_t> got(dst_bytes, 0);
    std::vector<uint8_t> scratch(size_t(w) * bs, 0);
    native::DecodeTiled2DCopyLevel<bs, sw>(src.data(), src.size(), got.data(), 1, w, h, sx, sy,
                                           pitch_aligned, bpb_log2, 0, dst_bytes, scratch.data());
    std::vector<uint8_t> ref(dst_bytes, 0);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        const size_t off = size_t(texture_util::GetTiledOffset2D(
            int32_t(x + sx), int32_t(y + sy), pitch_aligned, bpb_log2));
        native::CopySwapBlock<bs, sw>(src.data(), off, ref.data() + (size_t(y) * w + x) * bs);
      }
    }
    CHECK(got == ref);
  };
  for (uint32_t pitch : {32u, 64u, 96u}) {
    for (uint32_t sx : {0u, 1u, 4u, 7u}) {
      for (uint32_t w : {1u, 3u, 7u, 13u, 19u, 37u}) {
        run_case(std::integral_constant<uint32_t, 1>{}, std::integral_constant<size_t, 0>{}, pitch, sx, 3, w, 5);
        run_case(std::integral_constant<uint32_t, 1>{}, std::integral_constant<size_t, 3>{}, pitch, sx, 3, w, 5);
        run_case(std::integral_constant<uint32_t, 2>{}, std::integral_constant<size_t, 1>{}, pitch, sx, 5, w, 5);
        run_case(std::integral_constant<uint32_t, 2>{}, std::integral_constant<size_t, 2>{}, pitch, sx, 5, w, 5);
        run_case(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 0>{}, pitch, sx, 4, w, 5);
        run_case(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 1>{}, pitch, sx, 4, w, 5);
        run_case(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 2>{}, pitch, sx, 4, w, 5);
        run_case(std::integral_constant<uint32_t, 4>{}, std::integral_constant<size_t, 3>{}, pitch, sx, 4, w, 5);
        run_case(std::integral_constant<uint32_t, 8>{}, std::integral_constant<size_t, 3>{}, pitch, sx, 1, w, 5);
        run_case(std::integral_constant<uint32_t, 16>{}, std::integral_constant<size_t, 3>{}, pitch, sx, 2, w, 5);
      }
    }
  }
  CHECK(0.0f * (1.0f / 16777215.0f) == 0.0f);
  CHECK(float(0xFFFFFFu) * (1.0f / 16777215.0f) == 1.0f);
  CHECK(float(0x7FFFFFu) * (1.0f / 16777215.0f) == Catch::Approx(0x7FFFFFu / 16777215.0f));
}
