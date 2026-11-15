#include <rex/platform.h>
#include <rex/platform/dynlib.h>

static_assert(REX_PLATFORM_WIN32, "This file is Windows-only");

#include "platform_win.h"

namespace rex::platform {

DynamicLibrary::~DynamicLibrary() {
  Close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    Close();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

bool DynamicLibrary::Load(const std::filesystem::path& path, SymbolResolution /*mode*/) {
  // Windows resolves all imports at load time; mode is informational only.
  Close();
  handle_ = static_cast<void*>(LoadLibraryW(path.c_str()));
  return handle_ != nullptr;
}

bool DynamicLibrary::OpenLoaded(const std::filesystem::path& path, SymbolResolution /*mode*/) {
  Close();
  HMODULE module = nullptr;
  // Default flags (0) increment the refcount so Close()'s FreeLibrary stays balanced.
  if (!GetModuleHandleExW(0, path.c_str(), &module)) {
    return false;
  }
  handle_ = static_cast<void*>(module);
  return true;
}

void DynamicLibrary::Close() {
  if (handle_) {
    FreeLibrary(static_cast<HMODULE>(handle_));
    handle_ = nullptr;
  }
}

void* DynamicLibrary::GetRawSymbol(const char* name) const {
  if (!handle_)
    return nullptr;
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
}

}  // namespace rex::platform
