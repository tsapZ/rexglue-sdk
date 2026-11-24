/**
 * @file        tests/unit/core/wait_multiple_test.cpp
 * @brief       Waiting on several objects must sleep, not spin
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <rex/thread.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

using namespace std::chrono_literals;
using rex::thread::Event;
using rex::thread::WaitHandle;
using rex::thread::WaitResult;

namespace {

// CPU time this thread has used so far.
std::chrono::nanoseconds ThreadCpuTime() {
#if defined(_WIN32)
  FILETIME created, exited, kernel, user;
  GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user);
  auto ticks = [](const FILETIME& t) {
    return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime;
  };
  return std::chrono::nanoseconds((ticks(kernel) + ticks(user)) * 100);
#else
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec);
#endif
}

// Waits on two events nobody sets and returns how much of the wait the thread spent on the CPU.
double BusyShareOfWait(bool alertable, std::chrono::milliseconds timeout) {
  auto a = Event::CreateAutoResetEvent(false);
  auto b = Event::CreateAutoResetEvent(false);
  WaitHandle* handles[] = {a.get(), b.get()};
  auto cpu_before = ThreadCpuTime();
  auto wall_before = std::chrono::steady_clock::now();
  auto result = rex::thread::WaitAny(handles, 2, alertable, timeout);
  auto wall = std::chrono::steady_clock::now() - wall_before;
  auto cpu = ThreadCpuTime() - cpu_before;
  REQUIRE(result.first == WaitResult::kTimeout);
  REQUIRE(wall >= timeout - 5ms);
  return double(cpu.count()) / double(std::chrono::duration_cast<std::chrono::nanoseconds>(wall).count());
}

}  // namespace

// An alertable wait polls in 1 ms slices; a slice's time left used to be rounded down to a zero sleep, so
// the thread never slept (CoL #136: a whole core busy in Spyro's Adventure on Linux).
TEST_CASE("rex::thread::WaitAny - an alertable wait on two objects sleeps", "[thread]") {
  CHECK(BusyShareOfWait(true, 400ms) < 0.5);
}

TEST_CASE("rex::thread::WaitAny - a plain wait on two objects sleeps", "[thread]") {
  CHECK(BusyShareOfWait(false, 400ms) < 0.5);
}

TEST_CASE("rex::thread::WaitAny - wakes for the object that gets set", "[thread]") {
  auto a = Event::CreateAutoResetEvent(false);
  auto b = Event::CreateAutoResetEvent(false);
  WaitHandle* handles[] = {a.get(), b.get()};
  std::thread setter([&] {
    std::this_thread::sleep_for(50ms);
    b->Set();
  });
  auto result = rex::thread::WaitAny(handles, 2, true, 5000ms);
  setter.join();
  CHECK(result.first == WaitResult::kSuccess);
  CHECK(result.second == 1);
}
