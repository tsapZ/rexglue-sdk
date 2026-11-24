#include <array>
#include <map>

#include <catch2/catch_test_macros.hpp>

#include "../../../src/graphics/native/draw_input.h"

using namespace rex::graphics;

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
