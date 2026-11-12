#include <rex/graphics/texture_patch.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace rex::graphics {
namespace {

struct Entry {
  TexturePatch fn = nullptr;
  void* user = nullptr;
};

std::shared_mutex g_mutex;
std::unordered_map<uint64_t, Entry> g_patches;
// Whether any patch is there: the texture cache asks on every first load, without a lock.
std::atomic<bool> g_any{false};

}  // namespace

void RegisterTexturePatch(uint64_t hash, TexturePatch fn, void* user) {
  std::unique_lock lock(g_mutex);
  if (fn) {
    g_patches[hash] = Entry{fn, user};
  } else {
    g_patches.erase(hash);
  }
  g_any.store(!g_patches.empty());
}

void ClearTexturePatches() {
  std::unique_lock lock(g_mutex);
  g_patches.clear();
  g_any.store(false);
}

bool HasTexturePatches() { return g_any.load(std::memory_order_relaxed); }

bool HasTexturePatch(uint64_t hash) {
  if (!HasTexturePatches()) return false;
  std::shared_lock lock(g_mutex);
  return g_patches.count(hash) != 0;
}

bool ApplyTexturePatch(uint64_t hash, uint32_t width, uint32_t height, const uint8_t* data,
                       size_t size, std::vector<uint8_t>& dds) {
  Entry entry;
  {
    std::shared_lock lock(g_mutex);
    const auto it = g_patches.find(hash);
    if (it == g_patches.end()) return false;
    entry = it->second;
  }
  dds.clear();
  return entry.fn(hash, width, height, data, size, dds, entry.user) && !dds.empty();
}

}  // namespace rex::graphics
