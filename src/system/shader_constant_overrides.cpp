#include <rex/graphics/shader_constant_overrides.h>

#include <atomic>
#include <charconv>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(debug_pixel_constant_override, "", "GPU",
                      "Debug: override pixel shader constants without a rebuild. Entries "
                      "\"<hash hex>:<index>:<component>=<value>\" joined by ';'.");

namespace rex::graphics {
namespace {

struct Entry {
  PixelConstantOverride fn;
  void* user;
};

std::shared_mutex g_mutex;
std::unordered_map<uint64_t, Entry> g_callbacks;
std::atomic<uint64_t> g_generation{1};
// Whether any override is there (callbacks or debug ones): without, the per-draw questions need no
// lock (set under the unique lock).
std::atomic<bool> g_any{false};

// Debug overrides, re-parsed when the setting string changes.
std::string g_debug_text;
std::vector<DebugPixelConstantOverride> g_debug;

void RefreshDebugOverrides() {
  const std::string text = REXCVAR_GET(debug_pixel_constant_override);
  {
    std::shared_lock lock(g_mutex);
    if (text == g_debug_text) return;
  }
  auto parsed = ParseDebugPixelConstantOverrides(text);
  std::unique_lock lock(g_mutex);
  if (text == g_debug_text) return;
  g_debug_text = text;
  g_debug = std::move(parsed);
  g_any.store(!g_callbacks.empty() || !g_debug.empty());
  g_generation.fetch_add(1);
}

bool ParseHex(std::string_view s, uint64_t& out) {
  if (s.empty()) return false;
  auto r = std::from_chars(s.data(), s.data() + s.size(), out, 16);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

bool ParseUint(std::string_view s, uint32_t& out) {
  if (s.empty()) return false;
  auto r = std::from_chars(s.data(), s.data() + s.size(), out, 10);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

bool ParseFloat(std::string_view s, float& out) {
  if (s.empty()) return false;
  auto r = std::from_chars(s.data(), s.data() + s.size(), out);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

}  // namespace

void RegisterPixelConstantOverride(uint64_t ucode_hash, PixelConstantOverride fn, void* user) {
  std::unique_lock lock(g_mutex);
  g_callbacks[ucode_hash] = {fn, user};
  g_any.store(true);
  g_generation.fetch_add(1);
}

void ClearPixelConstantOverrides() {
  std::unique_lock lock(g_mutex);
  g_callbacks.clear();
  g_any.store(!g_debug.empty());
  g_generation.fetch_add(1);
}

void InvalidatePixelConstantOverrides() { g_generation.fetch_add(1); }

uint64_t PixelConstantOverrideGeneration() {
  // Asked for every draw: the setting (a string copy and a lock) is read again only every 1024
  // calls, a few frames (it cost about 3 % of the command processor thread in Spyro's Adventure).
  static std::atomic<uint32_t> calls{0};
  if ((calls.fetch_add(1, std::memory_order_relaxed) & 1023) == 0) {
    RefreshDebugOverrides();
  }
  return g_generation.load();
}

bool HasPixelConstantOverrides(uint64_t ucode_hash) {
  if (!g_any.load()) return false;
  std::shared_lock lock(g_mutex);
  if (g_callbacks.count(ucode_hash)) return true;
  for (const auto& d : g_debug) {
    if (d.hash == ucode_hash) return true;
  }
  return false;
}

void ApplyPixelConstantOverrides(uint64_t ucode_hash, uint32_t index, float xyzw[4]) {
  std::shared_lock lock(g_mutex);
  if (auto it = g_callbacks.find(ucode_hash); it != g_callbacks.end() && it->second.fn) {
    it->second.fn(index, xyzw, it->second.user);
  }
  for (const auto& d : g_debug) {
    if (d.hash == ucode_hash && d.index == index) xyzw[d.component] = d.value;
  }
}

std::vector<DebugPixelConstantOverride> ParseDebugPixelConstantOverrides(std::string_view text) {
  std::vector<DebugPixelConstantOverride> out;
  while (!text.empty()) {
    const size_t end = text.find(';');
    std::string_view item = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view() : text.substr(end + 1);
    if (item.empty()) continue;
    const size_t c1 = item.find(':');
    const size_t c2 = c1 == std::string_view::npos ? c1 : item.find(':', c1 + 1);
    const size_t eq = c2 == std::string_view::npos ? c2 : item.find('=', c2 + 1);
    DebugPixelConstantOverride d{};
    if (eq == std::string_view::npos || !ParseHex(item.substr(0, c1), d.hash) ||
        !ParseUint(item.substr(c1 + 1, c2 - c1 - 1), d.index) ||
        !ParseUint(item.substr(c2 + 1, eq - c2 - 1), d.component) ||
        !ParseFloat(item.substr(eq + 1), d.value) || d.index > 255 || d.component > 3) {
      REXLOG_WARN("debug_pixel_constant_override: ignoring malformed entry '{}'", item);
      continue;
    }
    out.push_back(d);
  }
  return out;
}

}  // namespace rex::graphics
