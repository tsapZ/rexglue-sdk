#include <rex/system/frame_gate.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace rex::system {
namespace {

std::atomic<bool> g_held{false};
std::mutex g_mutex;
std::condition_variable g_released;

}  // namespace

void SetGuestFramesHeld(bool held) {
  {
    std::lock_guard lock(g_mutex);
    g_held = held;
  }
  if (!held) g_released.notify_all();
}

bool GuestFramesHeld() { return g_held; }

void WaitWhileGuestFramesHeld() {
  if (!g_held) return;
  std::unique_lock lock(g_mutex);
  // In slices, so a release that somehow came without the notification is still seen.
  while (g_held) g_released.wait_for(lock, std::chrono::milliseconds(100));
}

}  // namespace rex::system
