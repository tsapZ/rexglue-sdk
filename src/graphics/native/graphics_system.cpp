/**
 * @file        graphics/native/graphics_system.cpp
 * @brief       From-scratch Vulkan graphics system for Skylanders.
 */

#include "graphics_system.h"

#include <rex/ui/vulkan/provider.h>

#include "command_processor.h"

namespace rex::graphics::native {

NativeGraphicsSystem::NativeGraphicsSystem() = default;
NativeGraphicsSystem::~NativeGraphicsSystem() = default;

std::string NativeGraphicsSystem::name() const { return "Native Vulkan (from scratch)"; }

void NativeGraphicsSystem::CreateProvider(bool with_presentation) {
  provider_ = rex::ui::vulkan::VulkanProvider::Create(true, with_presentation);
}

std::unique_ptr<CommandProcessor> NativeGraphicsSystem::CreateCommandProcessor() {
  return std::make_unique<NativeCommandProcessor>(this, kernel_state_);
}

}  // namespace rex::graphics::native
