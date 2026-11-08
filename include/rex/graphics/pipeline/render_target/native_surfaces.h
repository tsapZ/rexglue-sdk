#pragma once

// Adapted from gow2-recomp's patch 0036 (santiquiroz/gears-of-war-2-pc-port), BSD-3-Clause, see
// docs/third_party/gow2-recomp-LICENSE.txt.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace rex::graphics::native_surfaces {

struct SurfaceKind {
  bool is_depth;
  uint32_t format;
  uint32_t msaa_samples_log2;
};

// k_16_16 as a colour render target resource format.
constexpr uint32_t kColorFormat16_16 = 4;

// Which EDRAM ownership copies survive with gpu_native_surfaces. This is Gears of War 2's list as a
// starting point: depth to depth, and k_16_16 colour to depth. Each game's own list comes from its
// counts and pictures.
inline bool KeepTransfer(const SurfaceKind& source, const SurfaceKind& dest) {
  if (!dest.is_depth) {
    return false;
  }
  return source.is_depth || source.format == kColorFormat16_16;
}

// A kind as the report names it: "C2x1" is colour format 2 at 1x, "D0x4" depth format 0 at 4x.
inline std::string KindName(const SurfaceKind& kind) {
  return std::string(kind.is_depth ? "D" : "C") + std::to_string(kind.format) + "x" +
         std::to_string(1u << kind.msaa_samples_log2);
}

// A game's own keep list on top of KeepTransfer (gpu_native_surfaces_keep), in the report's names,
// separated by commas: "C2x1->D0x1" keeps that one pair, "C2x1" every transfer from or to it.
inline bool KeepByList(const SurfaceKind& source, const SurfaceKind& dest, const std::string& list) {
  if (list.empty()) {
    return false;
  }
  const std::string from = KindName(source), to = KindName(dest);
  const std::string pair = from + "->" + to;
  size_t start = 0;
  while (start <= list.size()) {
    size_t end = list.find(',', start);
    if (end == std::string::npos) end = list.size();
    std::string item = list.substr(start, end - start);
    item.erase(0, item.find_first_not_of(' '));
    item.erase(item.find_last_not_of(' ') + 1);
    if (item == pair || (item.find("->") == std::string::npos && (item == from || item == to))) {
      return true;
    }
    // "D<->C": every transfer between a depth and a colour target. Giants, SuperChargers,
    // Imaginators and Spyro's Adventure need theirs (2026-11-08: dark floors, white or blown-out
    // pictures, wrong lighting without them).
    if (item == "D<->C" && source.is_depth != dest.is_depth) {
      return true;
    }
    start = end + 1;
  }
  return false;
}

class TransferCounters {
 public:
  void Record(const SurfaceKind& source, const SurfaceKind& dest, uint32_t tiles) {
    Totals& totals = counters_[Key(source, dest)];
    ++totals.count;
    totals.tiles += tiles;
  }

  std::string ReportAndReset(double seconds) {
    std::vector<std::pair<uint64_t, std::string>> entries;
    for (const auto& [key, totals] : counters_) {
      entries.emplace_back(totals.tiles, FormatEntry(key, totals, seconds));
    }
    counters_.clear();
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::string report;
    for (const auto& entry : entries) {
      if (!report.empty()) {
        report += "; ";
      }
      report += entry.second;
    }
    return report;
  }

  bool empty() const { return counters_.empty(); }

 private:
  using KindKey = std::tuple<bool, uint32_t, uint32_t>;
  using PairKey = std::pair<KindKey, KindKey>;

  struct Totals {
    uint64_t count = 0;
    uint64_t tiles = 0;
  };

  static KindKey ToKey(const SurfaceKind& kind) {
    return {kind.is_depth, kind.format, kind.msaa_samples_log2};
  }

  static PairKey Key(const SurfaceKind& source, const SurfaceKind& dest) {
    return {ToKey(source), ToKey(dest)};
  }

  static std::string FormatKind(const KindKey& kind) {
    const auto& [is_depth, format, msaa_samples_log2] = kind;
    return std::string(is_depth ? "D" : "C") + std::to_string(format) + "x" +
           std::to_string(1u << msaa_samples_log2);
  }

  static std::string FormatEntry(const PairKey& key, const Totals& totals, double seconds) {
    char rates[64];
    std::snprintf(rates, sizeof(rates), " %.1f/s %.0f tiles/s", totals.count / seconds,
                  totals.tiles / seconds);
    return FormatKind(key.first) + "->" + FormatKind(key.second) + rates;
  }

  std::map<PairKey, Totals> counters_;
};

}  // namespace rex::graphics::native_surfaces
