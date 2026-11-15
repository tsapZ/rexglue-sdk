#include <rex/platform.h>
#include <rex/platform/dynlib.h>

static_assert(REX_PLATFORM_LINUX || REX_PLATFORM_MAC, "This file is POSIX-only");

#include <dlfcn.h>

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

static int ResolveDlopenFlags(SymbolResolution mode, bool only_if_loaded) {
  int flags = (mode == SymbolResolution::kImmediate) ? RTLD_NOW : RTLD_LAZY;
  if (only_if_loaded) {
    flags |= RTLD_NOLOAD;
  }
  return flags;
}

bool DynamicLibrary::Load(const std::filesystem::path& path, SymbolResolution mode) {
  Close();
  handle_ = dlopen(path.c_str(), ResolveDlopenFlags(mode, false));
  return handle_ != nullptr;
}

bool DynamicLibrary::OpenLoaded(const std::filesystem::path& path, SymbolResolution mode) {
  Close();
  handle_ = dlopen(path.c_str(), ResolveDlopenFlags(mode, true));
  return handle_ != nullptr;
}

void DynamicLibrary::Close() {
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
  }
}

void* DynamicLibrary::GetRawSymbol(const char* name) const {
  if (!handle_)
    return nullptr;
  return dlsym(handle_, name);
}

}  // namespace rex::platform
