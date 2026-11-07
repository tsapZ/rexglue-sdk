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
