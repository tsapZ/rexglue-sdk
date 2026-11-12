// HostPathDevice::AddOverlay (vfs_overlay_roots, mods): files of an overlay take the place of the
// device's files at the same path (names without case), new ones are added, folders merge, a later
// overlay wins, and the device's own folder is never written.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <rex/filesystem.h>
#include <rex/filesystem/devices/host_path_device.h>
#include <rex/filesystem/entry.h>

using rex::filesystem::Entry;
using rex::filesystem::HostPathDevice;

namespace {

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  FILE* file = rex::filesystem::OpenFile(path, "wb");
  REQUIRE(file != nullptr);
  fwrite(text.data(), 1, text.size(), file);
  fclose(file);
}

class ScopedFolders {
 public:
  ScopedFolders() {
    root_ = std::filesystem::temp_directory_path() / ("rexglue_vfs_overlay_" + std::to_string(counter_++));
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
    WriteText(rom() / "text" / "eng" / "global.lxb", "disc");
    WriteText(rom() / "default.xex", "xex");
    WriteText(mod_a() / "files" / "TEXT" / "Eng" / "GLOBAL.lxb", "mod a, longer");
    WriteText(mod_a() / "files" / "text" / "dut" / "global.lxb", "new folder");
    WriteText(mod_a() / "files" / "extra.txt", "added");
    WriteText(mod_b() / "files" / "text" / "eng" / "global.lxb", "mod b");
    WriteText(mod_b() / "files" / "default.xex" / "inside.txt", "a folder where the disc has a file");
  }
  ~ScopedFolders() {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }
  std::filesystem::path rom() const { return root_ / "rom"; }
  std::filesystem::path mod_a() const { return root_ / "mods" / "a"; }
  std::filesystem::path mod_b() const { return root_ / "mods" / "b"; }

 private:
  static int counter_;
  std::filesystem::path root_;
};
int ScopedFolders::counter_ = 0;

}  // namespace

TEST_CASE("Overlay replaces and adds files, later overlays win", "[vfs][overlay]") {
  ScopedFolders folders;
  HostPathDevice device("\\Overlay", folders.rom(), true);
  REQUIRE(device.Initialize());

  const auto [replaced_a, added_a] = device.AddOverlay(folders.mod_a() / "files");
  CHECK(replaced_a == 1);
  CHECK(added_a == 2);
  Entry* global = device.ResolvePath("text\\eng\\global.lxb");
  REQUIRE(global != nullptr);
  CHECK(global->size() == std::string("mod a, longer").size());
  REQUIRE(device.ResolvePath("text\\dut\\global.lxb") != nullptr);
  REQUIRE(device.ResolvePath("extra.txt") != nullptr);

  const auto [replaced_b, added_b] = device.AddOverlay(folders.mod_b() / "files");
  CHECK(replaced_b == 1);
  CHECK(added_b == 0);  // default.xex\inside.txt left out: the disc has a file there
  CHECK(device.ResolvePath("text\\eng\\global.lxb")->size() == std::string("mod b").size());
  REQUIRE(device.ResolvePath("default.xex") != nullptr);
  CHECK(device.ResolvePath("default.xex")->size() == std::string("xex").size());

  // The disc's folder is unchanged.
  CHECK(std::filesystem::file_size(folders.rom() / "text" / "eng" / "global.lxb") == 4);
  CHECK(!std::filesystem::exists(folders.rom() / "extra.txt"));
}

TEST_CASE("Overlay of a missing folder does nothing", "[vfs][overlay]") {
  ScopedFolders folders;
  HostPathDevice device("\\Overlay", folders.rom(), true);
  REQUIRE(device.Initialize());
  const auto [replaced, added] = device.AddOverlay(folders.mod_a() / "no such folder");
  CHECK(replaced == 0);
  CHECK(added == 0);
}
