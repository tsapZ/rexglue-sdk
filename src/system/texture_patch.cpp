#include <rex/graphics/texture_patch.h>

#include <atomic>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace rex::graphics {
namespace {

struct Entry {
  TexturePatch fn = nullptr;
  void* user = nullptr;
  uint32_t width = 0, height = 0;
};

std::shared_mutex g_mutex;
std::unordered_map<uint64_t, Entry> g_patches;
// How many patches there are of each size: textures of other sizes need no hash.
std::map<std::pair<uint32_t, uint32_t>, int> g_sizes;
// Whether any patch is there: the texture cache asks on every first load, without a lock.
std::atomic<bool> g_any{false};

void Forget(std::unordered_map<uint64_t, Entry>::iterator it) {
  const auto size = std::make_pair(it->second.width, it->second.height);
  if (--g_sizes[size] <= 0) g_sizes.erase(size);
  g_patches.erase(it);
}

}  // namespace

void RegisterTexturePatch(uint64_t hash, uint32_t width, uint32_t height, TexturePatch fn,
                          void* user) {
  std::unique_lock lock(g_mutex);
  const auto it = g_patches.find(hash);
  if (it != g_patches.end()) Forget(it);
  if (fn) {
    g_patches[hash] = Entry{fn, user, width, height};
    ++g_sizes[std::make_pair(width, height)];
  }
  g_any.store(!g_patches.empty());
}

void ClearTexturePatches() {
  std::unique_lock lock(g_mutex);
  g_patches.clear();
  g_sizes.clear();
  g_any.store(false);
}

bool HasTexturePatches() { return g_any.load(std::memory_order_relaxed); }

bool HasTexturePatchSize(uint32_t width, uint32_t height) {
  if (!HasTexturePatches()) return false;
  std::shared_lock lock(g_mutex);
  return g_sizes.count(std::make_pair(width, height)) != 0;
}

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
