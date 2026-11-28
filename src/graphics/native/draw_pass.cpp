#include "draw_pass.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/util.h>

#include "depth_resolve.h"
#include "edram_depth_clear_spirv.h"

namespace rex::graphics::native {
namespace {

using Translator = SpirvShaderTranslator;

// Both vertex and pixel shaders sample textures; compute resolves also read/write views.
constexpr VkPipelineStageFlags kTextureStages = kAllShaderReadStages;

VkShaderModule CreateShaderModule(const ui::vulkan::VulkanDevice& device, const uint32_t* code,
                                  size_t size) {
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = size;
  info.pCode = code;
  VkShaderModule module = VK_NULL_HANDLE;
  return device.functions().vkCreateShaderModule(device.device(), &info, nullptr, &module) ==
                 VK_SUCCESS
             ? module
             : VK_NULL_HANDLE;
}

VkShaderModule CreateShaderModule(const ui::vulkan::VulkanDevice& device,
                                  const Shader::Translation& translation) {
  const auto& binary = translation.translated_binary();
  if (binary.empty() || binary.size() % sizeof(uint32_t)) {
    return VK_NULL_HANDLE;
  }
  return CreateShaderModule(device, reinterpret_cast<const uint32_t*>(binary.data()),
                            binary.size());
}

}  // namespace

CachedTexture::~CachedTexture() {
  if (!device) {
    return;
  }
  if (view_purger && view_purger->purge_views) {
    std::vector<VkImageView> dead_views;
    dead_views.reserve(views.size() + 1);
    for (const auto& entry : views) {
      if (entry.second != VK_NULL_HANDLE) {
        dead_views.push_back(entry.second);
      }
    }
    if (storage_view != VK_NULL_HANDLE) {
      dead_views.push_back(storage_view);
    }
    view_purger->PurgeImageViews(dead_views);
  }
  const auto& vk = device->functions();
  const VkDevice handle = device->device();
  for (const auto& entry : views) {
    vk.vkDestroyImageView(handle, entry.second, nullptr);
  }
  vk.vkDestroyImageView(handle, storage_view, nullptr);
  DestroyDeviceImage(*device, allocator, image, memory, allocation);
}

VkImageView CachedTexture::GetOrCreateStorageView(VkFormat format) {
  if (storage_view != VK_NULL_HANDLE) {
    return storage_view;
  }
  VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  info.image = image;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.format = format;
  info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (device->functions().vkCreateImageView(device->device(), &info, nullptr, &storage_view) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  return storage_view;
}

VkImageView CachedTexture::GetOrCreateView(VkImageViewType view_type, VkFormat format,
                                           VkComponentMapping swizzle, uint32_t array_layers) {
  const uint64_t view_key = (uint64_t(format) << 32) | PackTextureComponentMapping(swizzle);
  if (last_view != VK_NULL_HANDLE && last_view_key == view_key) {
    return last_view;
  }
  if (auto found = views.find(view_key); found != views.end()) {
    last_view_key = view_key;
    last_view = found->second;
    return last_view;
  }
  VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  info.image = image;
  info.viewType = view_type;
  info.format = format;
  info.components = swizzle;
  info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, level_count, 0, array_layers};
  VkImageView image_view = VK_NULL_HANDLE;
  if (device->functions().vkCreateImageView(device->device(), &info, nullptr, &image_view) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  views.emplace(view_key, image_view);
  last_view_key = view_key;
  last_view = image_view;
  return image_view;
}

DrawBindings::~DrawBindings() {
  const auto& vk = device.functions();
  if (shared_vertex_set_index != SIZE_MAX && pools[0]) {
    pools[0]->Free(shared_vertex_set_index);
  }
  if (pools[1]) {
    for (const auto& entry : constant_sets) {
      pools[1]->Free(entry.second);
    }
  }
  for (uint32_t f = 0; f < 3; ++f) {
    ReleaseFrameTextureSets(f);
  }
  for (size_t i = 0; i < empty_set_indices.size(); ++i) {
    if (empty_set_indices[i] != SIZE_MAX && pools[i]) {
      pools[i]->Free(empty_set_indices[i]);
    }
  }
  for (size_t i = 0; i < layouts.size(); ++i) {
    pools[i].reset();
    if (i >= 2 || owns_prefix_layouts[i]) {
      vk.vkDestroyDescriptorSetLayout(device.device(), layouts[i], nullptr);
    }
  }
  vk.vkDestroyPipelineLayout(device.device(), pipeline_layout, nullptr);
}

namespace {

void WriteSharedVertexSetDescriptors(const ui::vulkan::VulkanDevice& device, VkDescriptorSet set,
                                     VkBuffer buffer, uint32_t shared_count) {
  constexpr size_t kInlineSharedBuffers = 8;
  std::array<VkDescriptorBufferInfo, kInlineSharedBuffers> inline_shared_infos{};
  std::vector<VkDescriptorBufferInfo> heap_shared_infos;
  VkDescriptorBufferInfo* shared_infos;
  if (shared_count <= kInlineSharedBuffers) {
    shared_infos = inline_shared_infos.data();
  } else {
    heap_shared_infos.resize(shared_count);
    shared_infos = heap_shared_infos.data();
  }
  const VkDeviceSize range = kPhysicalMemorySize / shared_count;
  for (uint32_t i = 0; i < shared_count; ++i) {
    shared_infos[i] = {buffer, VkDeviceSize(i) * range, range};
  }
  const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   nullptr,
                                   set,
                                   0,
                                   0,
                                   shared_count,
                                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   nullptr,
                                   shared_infos};
  device.functions().vkUpdateDescriptorSets(device.device(), 1, &write, 0, nullptr);
}

void WriteConstantSetDescriptors(const ui::vulkan::VulkanDevice& device, VkDescriptorSet set,
                                 VkBuffer buffer) {
  std::array<VkDescriptorBufferInfo, Translator::kConstantBufferCount> infos{};
  std::array<VkWriteDescriptorSet, Translator::kConstantBufferCount> writes{};
  for (uint32_t i = 0; i < infos.size(); ++i) {
    infos[i] = {buffer, 0, kDynamicConstantBufferRanges[i]};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 set,
                 i,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                 nullptr,
                 &infos[i]};
  }
  device.functions().vkUpdateDescriptorSets(device.device(), uint32_t(writes.size()),
                                            writes.data(), 0, nullptr);
}

}  // namespace

VkDescriptorSet DrawBindingCache::GetOrCreateSharedVertexSet(VkBuffer buffer,
                                                             uint32_t shared_count) {
  if (shared_vertex_set != VK_NULL_HANDLE && shared_vertex_buffer == buffer) {
    return shared_vertex_set;
  }
  if (!layout_device || !shared_set0_pool) {
    return VK_NULL_HANDLE;
  }
  if (shared_vertex_set_index == SIZE_MAX) {
    shared_vertex_set_index = shared_set0_pool->Allocate();
    if (shared_vertex_set_index == SIZE_MAX) {
      return VK_NULL_HANDLE;
    }
    shared_vertex_set = shared_set0_pool->Get(shared_vertex_set_index);
  }
  WriteSharedVertexSetDescriptors(*layout_device, shared_vertex_set, buffer, shared_count);
  shared_vertex_buffer = buffer;
  return shared_vertex_set;
}

VkDescriptorSet DrawBindingCache::GetOrCreateConstantSet(VkBuffer buffer) {
  if (buffer == last_constant_buffer && last_constant_set != VK_NULL_HANDLE) {
    return last_constant_set;
  }
  if (auto found = constant_sets.find(buffer); found != constant_sets.end()) {
    last_constant_buffer = buffer;
    last_constant_set = shared_set1_pool->Get(found->second);
    return last_constant_set;
  }
  if (!layout_device || !shared_set1_pool) {
    return VK_NULL_HANDLE;
  }
  const size_t index = shared_set1_pool->Allocate();
  if (index == SIZE_MAX) {
    return VK_NULL_HANDLE;
  }
  const VkDescriptorSet set = shared_set1_pool->Get(index);
  WriteConstantSetDescriptors(*layout_device, set, buffer);
  constant_sets.emplace(buffer, index);
  last_constant_buffer = buffer;
  last_constant_set = set;
  return set;
}

VkDescriptorSet DrawBindings::GetOrCreateSharedVertexSet(VkBuffer buffer, uint32_t shared_count) {
  if (shared_vertex_set != VK_NULL_HANDLE && shared_vertex_buffer == buffer) {
    return shared_vertex_set;
  }
  if (shared_vertex_set_index == SIZE_MAX) {
    shared_vertex_set_index = pools[0]->Allocate();
    if (shared_vertex_set_index == SIZE_MAX) {
      return VK_NULL_HANDLE;
    }
    shared_vertex_set = pools[0]->Get(shared_vertex_set_index);
  }
  WriteSharedVertexSetDescriptors(device, shared_vertex_set, buffer, shared_count);
  shared_vertex_buffer = buffer;
  return shared_vertex_set;
}

VkDescriptorSet DrawBindings::GetOrCreateConstantSet(VkBuffer buffer) {
  if (buffer == last_constant_buffer && last_constant_set != VK_NULL_HANDLE) {
    return last_constant_set;
  }
  if (auto found = constant_sets.find(buffer); found != constant_sets.end()) {
    last_constant_buffer = buffer;
    last_constant_set = pools[1]->Get(found->second);
    return last_constant_set;
  }
  const size_t index = pools[1]->Allocate();
  if (index == SIZE_MAX) {
    return VK_NULL_HANDLE;
  }
  const VkDescriptorSet set = pools[1]->Get(index);
  WriteConstantSet(set, buffer);
  constant_sets.emplace(buffer, index);
  last_constant_buffer = buffer;
  last_constant_set = set;
  return set;
}

void DrawBindings::CommitPendingTextureSets(uint32_t frame_slot) {
  if (frame_slot >= 2) {
    return;
  }
  ReleaseFrameTextureSets(frame_slot);
  for (size_t s = 0; s < 2; ++s) {
    ephemeral_texture_sets_[frame_slot][s].swap(
        ephemeral_texture_sets_[kPendingTextureSetSlot][s]);
  }
}

void DrawBindings::ReleaseFrameTextureSets(uint32_t frame_slot) {
  if (frame_slot >= ephemeral_texture_sets_.size()) {
    return;
  }
  for (size_t s = 0; s < 2; ++s) {
    if (pools[s + 2]) {
      for (const size_t index : ephemeral_texture_sets_[frame_slot][s]) {
        pools[s + 2]->Free(index);
      }
    }
    ephemeral_texture_sets_[frame_slot][s].clear();
  }
}

void DrawBindings::ReclaimTextureSets(uint64_t retired_submission) {
  for (size_t s = 0; s < 2; ++s) {
    auto& retired = retired_texture_sets_[s];
    if (retired.empty() || !pools[s + 2]) {
      continue;
    }
    size_t keep = 0;
    for (size_t i = 0; i < retired.size(); ++i) {
      if (retired[i].second <= retired_submission) {
        pools[s + 2]->Free(retired[i].first);
      } else {
        retired[keep++] = retired[i];
      }
    }
    retired.resize(keep);
  }
}

void DrawBindings::RemoveTextureSetFromViewIndex(size_t stage, const TextureSetKey* key_ptr,
                                                 VkImageView skip_view) {
  auto& index = view_to_texture_keys_[stage];
  for (VkImageView v : key_ptr->views) {
    if (v == VK_NULL_HANDLE || v == skip_view) {
      continue;
    }
    auto map_it = index.find(v);
    if (map_it == index.end()) {
      continue;
    }
    auto& keys = map_it->second;
    for (size_t i = 0; i < keys.size();) {
      if (keys[i] == key_ptr) {
        keys[i] = keys.back();
        keys.pop_back();
      } else {
        ++i;
      }
    }
    if (keys.empty()) {
      index.erase(map_it);
    }
  }
}

void DrawBindings::PurgeImageViews(std::span<const VkImageView> views) {
  if (views.empty()) {
    return;
  }
  for (size_t s = 0; s < 2; ++s) {
    auto& cache = persistent_texture_sets_[s];
    auto& index = view_to_texture_keys_[s];
    if (index.empty()) {
      continue;
    }
    bool purged_any = false;
    for (VkImageView v : views) {
      auto map_it = index.find(v);
      if (map_it == index.end()) {
        continue;
      }
      auto keys = std::move(map_it->second);
      index.erase(map_it);
      for (const TextureSetKey* key_ptr : keys) {
        auto it = cache.find(*key_ptr);
        if (it != cache.end()) {
          retired_texture_sets_[s].emplace_back(it->second.pool_index,
                                                it->second.last_use_submission);
          RemoveTextureSetFromViewIndex(s, key_ptr, v);
          cache.erase(it);
          purged_any = true;
        }
      }
    }
    if (purged_any) {
      last_texture_set_[s] = VK_NULL_HANDLE;
      last_texture_entry_[s] = nullptr;
    }
  }
}

void DrawBindings::PurgeSampler(VkSampler sampler) {
  if (sampler == VK_NULL_HANDLE) {
    return;
  }
  for (size_t s = 0; s < 2; ++s) {
    auto& cache = persistent_texture_sets_[s];
    bool purged_any = false;
    for (auto it = cache.begin(); it != cache.end();) {
      if (std::ranges::find(it->first.samplers, sampler) != it->first.samplers.end()) {
        retired_texture_sets_[s].emplace_back(it->second.pool_index,
                                              it->second.last_use_submission);
        RemoveTextureSetFromViewIndex(s, &it->first);
        it = cache.erase(it);
        purged_any = true;
      } else {
        ++it;
      }
    }
    if (purged_any) {
      last_texture_set_[s] = VK_NULL_HANDLE;
      last_texture_entry_[s] = nullptr;
    }
  }
}

VkDescriptorSet DrawBindings::GetOrCreateTextureSet(size_t set_slot, uint32_t frame_slot,
                                                    uint64_t submission,
                                                    uint64_t retired_submission,
                                                    std::span<const VkImageView> views,
                                                    std::span<const VkSampler> samplers,
                                                    bool cacheable) {
  if (set_slot < 2 || set_slot > 3 || frame_slot >= 3) {
    return VK_NULL_HANDLE;
  }
  const size_t stage = set_slot - 2;
  auto& cache = persistent_texture_sets_[stage];
  if (cacheable) {
    if (last_texture_set_[stage] != VK_NULL_HANDLE &&
        std::ranges::equal(last_texture_views_[stage], views) &&
        std::ranges::equal(last_texture_samplers_[stage], samplers)) {
      last_texture_entry_[stage]->last_use_submission = submission;
      return last_texture_set_[stage];
    }
    const TextureSetKeyView key_view{views, samplers};
    if (auto found = cache.find(key_view); found != cache.end()) {
      found->second.last_use_submission = submission;
      const VkDescriptorSet hit_set = pools[set_slot]->Get(found->second.pool_index);
      last_texture_views_[stage].assign(views.begin(), views.end());
      last_texture_samplers_[stage].assign(samplers.begin(), samplers.end());
      last_texture_set_[stage] = hit_set;
      last_texture_entry_[stage] = &found->second;
      return hit_set;
    }
    if (cache.size() >= kMaxPersistentTextureSets) {
      ReclaimTextureSets(retired_submission);
      // Evict up to 4 retired LRU entries if full.
      for (size_t pass = 0; pass < 4 && cache.size() >= kMaxPersistentTextureSets; ++pass) {
        auto victim = cache.end();
        uint64_t oldest = UINT64_MAX;
        for (auto it = cache.begin(); it != cache.end(); ++it) {
          if (it->second.last_use_submission <= retired_submission &&
              it->second.last_use_submission < oldest) {
            oldest = it->second.last_use_submission;
            victim = it;
          }
        }
        if (victim == cache.end()) {
          break;
        }
        pools[set_slot]->Free(victim->second.pool_index);
        RemoveTextureSetFromViewIndex(stage, &victim->first);
        if (last_texture_entry_[stage] == &victim->second) {
          last_texture_set_[stage] = VK_NULL_HANDLE;
          last_texture_entry_[stage] = nullptr;
        }
        cache.erase(victim);
      }
    }
  }
  const size_t index = pools[set_slot]->Allocate();
  if (index == SIZE_MAX) {
    return VK_NULL_HANDLE;
  }
  const VkDescriptorSet set = pools[set_slot]->Get(index);
  const size_t total = views.size() + samplers.size();
  constexpr size_t kInlineImageInfos = 64;
  std::array<VkDescriptorImageInfo, kInlineImageInfos> inline_image_infos{};
  std::vector<VkDescriptorImageInfo> heap_image_infos;
  VkDescriptorImageInfo* image_infos = inline_image_infos.data();
  if (total > kInlineImageInfos) {
    heap_image_infos.resize(total);
    image_infos = heap_image_infos.data();
  }
  uint32_t info_count = 0;
  for (VkImageView view : views) {
    image_infos[info_count++] = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }
  for (VkSampler sampler : samplers) {
    image_infos[info_count++] = {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
  }
  std::array<VkWriteDescriptorSet, 2> writes{};
  uint32_t write_count = 0;
  if (!views.empty()) {
    writes[write_count++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             set,
                             0,
                             0,
                             uint32_t(views.size()),
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             image_infos,
                             nullptr,
                             nullptr};
  }
  if (!samplers.empty()) {
    writes[write_count++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             set,
                             uint32_t(views.size()),
                             0,
                             uint32_t(samplers.size()),
                             VK_DESCRIPTOR_TYPE_SAMPLER,
                             image_infos + views.size(),
                             nullptr,
                             nullptr};
  }
  if (write_count) {
    device.functions().vkUpdateDescriptorSets(device.device(), write_count, writes.data(), 0,
                                              nullptr);
  }
  if (cacheable && cache.size() < kMaxPersistentTextureSets) {
    TextureSetKey key;
    key.views.assign(views.begin(), views.end());
    key.samplers.assign(samplers.begin(), samplers.end());
    auto [it, inserted] = cache.emplace(std::move(key), PersistentTextureSet{index, submission});
    if (inserted) {
      const TextureSetKey* key_ptr = &it->first;
      for (size_t vi = 0; vi < key_ptr->views.size(); ++vi) {
        const VkImageView v = key_ptr->views[vi];
        if (v == VK_NULL_HANDLE) {
          continue;
        }
        bool duplicate = false;
        for (size_t pj = 0; pj < vi; ++pj) {
          if (key_ptr->views[pj] == v) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          view_to_texture_keys_[stage][v].push_back(key_ptr);
        }
      }
      last_texture_views_[stage].assign(views.begin(), views.end());
      last_texture_samplers_[stage].assign(samplers.begin(), samplers.end());
      last_texture_set_[stage] = set;
      last_texture_entry_[stage] = &it->second;
    }
  } else {
    ephemeral_texture_sets_[frame_slot][stage].push_back(index);
  }
  return set;
}

void DrawBindings::WriteConstantSet(VkDescriptorSet set, VkBuffer buffer) const {
  std::array<VkDescriptorBufferInfo, Translator::kConstantBufferCount> infos{};
  std::array<VkWriteDescriptorSet, Translator::kConstantBufferCount> writes{};
  for (uint32_t i = 0; i < infos.size(); ++i) {
    infos[i] = {buffer, 0, kDynamicConstantBufferRanges[i]};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 set,
                 i,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                 nullptr,
                 &infos[i],
                 nullptr};
  }
  device.functions().vkUpdateDescriptorSets(device.device(), uint32_t(writes.size()),
                                            writes.data(), 0, nullptr);
}

VkDescriptorSet DrawBindings::GetEmptySet(size_t index) {
  if (!empty_layouts[index]) {
    return VK_NULL_HANDLE;
  }
  if (empty_set_indices[index] == SIZE_MAX) {
    empty_set_indices[index] = pools[index]->Allocate();
    if (empty_set_indices[index] == SIZE_MAX) {
      return VK_NULL_HANDLE;
    }
  }
  return pools[index]->Get(empty_set_indices[index]);
}

bool DrawBindings::Initialize(uint32_t shared_count, uint32_t vertex_image_count,
                              uint32_t vertex_sampler_count, uint32_t pixel_image_count,
                              uint32_t pixel_sampler_count,
                              VkDescriptorSetLayout shared_set0_layout,
                              VkDescriptorSetLayout shared_set1_layout) {
  // Each fetch constant may have an unsigned and a signed binding per dimension.
  constexpr uint32_t kMaxImages = xenos::kTextureFetchConstantCount * 4;
  if (vertex_image_count > kMaxImages || pixel_image_count > kMaxImages ||
      vertex_sampler_count > xenos::kTextureFetchConstantCount ||
      pixel_sampler_count > xenos::kTextureFetchConstantCount) {
    return false;
  }
  const auto& vk = device.functions();
  const auto handle = device.device();
  std::array<VkDescriptorSetLayoutBinding, Translator::kConstantBufferCount> constants{};
  for (uint32_t i = 0; i < constants.size(); ++i) {
    constants[i] = {i, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  }
  // Pixel shaders memexport through the shared memory binding too.
  const VkDescriptorSetLayoutBinding shared{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shared_count,
                                            VK_SHADER_STAGE_VERTEX_BIT |
                                                VK_SHADER_STAGE_FRAGMENT_BIT,
                                            nullptr};
  const auto texture_bindings = [](uint32_t image_count, uint32_t sampler_count,
                                   VkShaderStageFlags stage) {
    std::vector<VkDescriptorSetLayoutBinding> textures(image_count + sampler_count);
    for (uint32_t i = 0; i < textures.size(); ++i) {
      textures[i] = {
          i, i < image_count ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER, 1,
          stage, nullptr};
    }
    return textures;
  };
  const auto vertex_textures =
      texture_bindings(vertex_image_count, vertex_sampler_count, VK_SHADER_STAGE_VERTEX_BIT);
  const auto pixel_textures =
      texture_bindings(pixel_image_count, pixel_sampler_count, VK_SHADER_STAGE_FRAGMENT_BIT);
  const std::array<std::span<const VkDescriptorSetLayoutBinding>, 4> bindings{
      std::span(&shared, 1), constants, vertex_textures, pixel_textures};
  for (size_t i = 0; i < layouts.size(); ++i) {
    empty_layouts[i] = bindings[i].empty();
    if (i == 0 && shared_set0_layout != VK_NULL_HANDLE) {
      layouts[0] = shared_set0_layout;
      owns_prefix_layouts[0] = false;
    } else if (i == 1 && shared_set1_layout != VK_NULL_HANDLE) {
      layouts[1] = shared_set1_layout;
      owns_prefix_layouts[1] = false;
    } else {
      VkDescriptorSetLayoutCreateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      info.bindingCount = uint32_t(bindings[i].size());
      info.pBindings = bindings[i].data();
      if (vk.vkCreateDescriptorSetLayout(handle, &info, nullptr, &layouts[i]) != VK_SUCCESS) {
        return false;
      }
    }
    const bool adopt_shared_pool =
        (i == 0 && shared_set0_layout != VK_NULL_HANDLE) ||
        (i == 1 && shared_set1_layout != VK_NULL_HANDLE);
    if (!adopt_shared_pool) {
      std::vector<VkDescriptorPoolSize> sizes;
      for (const auto& binding : bindings[i]) {
        if (!sizes.empty() && sizes.back().type == binding.descriptorType) {
          sizes.back().descriptorCount += binding.descriptorCount;
        } else {
          sizes.push_back({binding.descriptorType, binding.descriptorCount});
        }
      }
      pools[i] = std::make_unique<ui::vulkan::SingleLayoutDescriptorSetPool>(
          &device, empty_layouts[i] ? 1 : 64, uint32_t(sizes.size()), sizes.data(), layouts[i]);
    }
  }
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.setLayoutCount = uint32_t(layouts.size());
  info.pSetLayouts = layouts.data();
  // Only the EDRAM depth-clear fragment shader reads push constants.
  const VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 2 * sizeof(uint32_t)};
  info.pushConstantRangeCount = 1;
  info.pPushConstantRanges = &push;
  return vk.vkCreatePipelineLayout(handle, &info, nullptr, &pipeline_layout) == VK_SUCCESS;
}

DrawPass::DrawPass(DrawPass&& other) noexcept
    : device_(other.device_), allocator_(other.allocator_) {
  *this = std::move(other);
}

DrawPass& DrawPass::operator=(DrawPass&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  Release();
  device_ = other.device_;
  allocator_ = other.allocator_;
  bindings_ = std::exchange(other.bindings_, nullptr);
  binding_cache_ = std::exchange(other.binding_cache_, nullptr);
  set0_from_cache_ = std::exchange(other.set0_from_cache_, false);
  set1_from_cache_ = std::exchange(other.set1_from_cache_, false);
  set_indices_ = std::exchange(other.set_indices_, {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX});
  sets_ = other.sets_;
  dynamic_offsets_ = other.dynamic_offsets_;
  pipeline_ = other.pipeline_;
  blend_constants_ = other.blend_constants_;
  buffer_ = other.buffer_;
  buffer_memory_ = std::exchange(other.buffer_memory_, VkDeviceMemory(VK_NULL_HANDLE));
  first_texture_op_ = std::exchange(other.first_texture_op_, 0);
  texture_op_count_ = std::exchange(other.texture_op_count_, 0);
  indexed_ = other.indexed_;
  index_buffer_ = other.index_buffer_;
  index_offset_ = other.index_offset_;
  index_type_ = other.index_type_;
  index_count_ = other.index_count_;
  viewport_ = other.viewport_;
  scissor_ = other.scissor_;
  depth_bias_enable_ = other.depth_bias_enable_;
  depth_bias_ = other.depth_bias_;
  stencil_enable_ = other.stencil_enable_;
  stencil_compare_mask_ = other.stencil_compare_mask_;
  stencil_write_mask_ = other.stencil_write_mask_;
  stencil_reference_ = other.stencil_reference_;
  edram_depth_clear_color_ = other.edram_depth_clear_color_;
  edram_clear_constants_ = other.edram_clear_constants_;
  rasterizer_discard_ = other.rasterizer_discard_;
  return *this;
}

DrawPass::~DrawPass() { Release(); }

void DrawPass::Release() {
  const auto& vk = device_->functions();
  const VkDevice device = device_->device();
  for (size_t i = 0; i < set_indices_.size(); ++i) {
    if (set_indices_[i] != SIZE_MAX) {
      if (i == 0 && set0_from_cache_ && binding_cache_ && binding_cache_->shared_set0_pool) {
        binding_cache_->shared_set0_pool->Free(set_indices_[i]);
      } else if (i == 1 && set1_from_cache_ && binding_cache_ && binding_cache_->shared_set1_pool) {
        binding_cache_->shared_set1_pool->Free(set_indices_[i]);
      } else if (bindings_ && bindings_->pools[i]) {
        bindings_->pools[i]->Free(set_indices_[i]);
      }
      set_indices_[i] = SIZE_MAX;
    }
  }
  set0_from_cache_ = false;
  set1_from_cache_ = false;
  if (buffer_memory_) {
    vk.vkDestroyBuffer(device, buffer_, nullptr);
    vk.vkFreeMemory(device, buffer_memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    buffer_memory_ = VK_NULL_HANDLE;
  }
}

bool DrawPass::Initialize(const DrawInput& input, const IndexBinding& index_binding,
                          std::span<const uint32_t> inline_indices,
                          const Shader::Translation& vertex, const Shader::Translation* pixel,
                          uint32_t shared_buffer_count, VkPipelineCache pipeline_cache,
                          DrawPipelines& pipelines, DrawBindingCache& binding_cache,
                          ui::vulkan::VulkanUploadBufferPool& uploads, uint64_t submission,
                          uint64_t retired_submission, uint32_t frame_slot,
                          VkBuffer shared_vertex_buffer, std::timed_mutex* pipeline_cache_mutex,
                          const PreuploadedConstants* preuploaded_constants,
                          DepthResolve* depth_resolve, std::vector<DrawTextureOp>* texture_ops) {
  const auto& vk = device_->functions();
  const VkDevice device = device_->device();

  std::vector<DrawTextureOp> local_ops;
  std::vector<DrawTextureOp>& ops = texture_ops ? *texture_ops : local_ops;
  const size_t ops_start = ops.size();
  first_texture_op_ = uint32_t(ops_start);
  texture_op_count_ = 0;
  constexpr size_t kInlineTextures = 32;
  std::array<uint32_t, kInlineTextures> inline_op_index{};
  std::array<VkImageView, kInlineTextures> inline_texture_views{};
  std::vector<uint32_t> heap_op_index;
  std::vector<VkImageView> heap_texture_views;
  uint32_t* op_index_for_input = inline_op_index.data();
  VkImageView* texture_views = inline_texture_views.data();
  if (input.textures.size() > kInlineTextures) {
    heap_op_index.assign(input.textures.size(), UINT32_MAX);
    heap_texture_views.assign(input.textures.size(), VK_NULL_HANDLE);
    op_index_for_input = heap_op_index.data();
    texture_views = heap_texture_views.data();
  } else {
    std::fill_n(op_index_for_input, input.textures.size(), UINT32_MAX);
    std::fill_n(texture_views, input.textures.size(), VK_NULL_HANDLE);
  }
  struct OpsGuard {
    const ui::vulkan::VulkanDevice* dev;
    VmaAllocator alloc;
    DepthResolve* resolve;
    std::vector<DrawTextureOp>& ops_ref;
    size_t start;
    uint32_t& count_ref;
    bool committed = false;
    ~OpsGuard() {
      if (!committed) {
        for (size_t j = start; j < ops_ref.size(); ++j) {
          ReleaseDrawTextureOp(*dev, alloc, resolve, ops_ref[j]);
        }
        ops_ref.resize(start);
        count_ref = 0;
      }
    }
  } ops_guard{device_, allocator_, depth_resolve, ops, ops_start, texture_op_count_};
  blend_constants_ = input.blend_constants;
  viewport_ = input.viewport;
  scissor_ = input.scissor;
  depth_bias_enable_ = input.depth_format != VK_FORMAT_UNDEFINED;
  depth_bias_ = input.depth_bias;
  stencil_enable_ = input.depth_stencil.stencilTestEnable != VK_FALSE;
  stencil_compare_mask_ = {input.depth_stencil.front.compareMask,
                           input.depth_stencil.back.compareMask};
  stencil_write_mask_ = {input.depth_stencil.front.writeMask, input.depth_stencil.back.writeMask};
  stencil_reference_ = {input.depth_stencil.front.reference, input.depth_stencil.back.reference};
  indexed_ = index_binding.indexed;
  index_buffer_ = index_binding.buffer;
  index_offset_ = index_binding.offset;
  index_type_ = index_binding.type;
  index_count_ = indexed_ && index_buffer_ == VK_NULL_HANDLE ? uint32_t(inline_indices.size())
                                                             : index_binding.count;
  const bool upload_inline_indices = indexed_ && index_buffer_ == VK_NULL_HANDLE;
  edram_depth_clear_color_ = input.edram_depth_clear_color;
  edram_clear_constants_ = {input.edram_clear_stencil, input.edram_clear_format};
  rasterizer_discard_ = input.rasterizer_discard;

  const std::array binding_key{shared_buffer_count, uint32_t(input.vertex_texture_bindings.size()),
                               uint32_t(input.vertex_samplers.size()),
                               uint32_t(input.texture_bindings.size()),
                               uint32_t(input.samplers.size())};
  binding_cache_ = &binding_cache;
  set0_from_cache_ = false;
  set1_from_cache_ = false;
  bindings_ = binding_cache.FindOrNull(binding_key);
  if (!bindings_) {
    auto layout = std::make_unique<DrawBindings>(*device_);
    const VkDescriptorSetLayout set0_shared =
        binding_cache.shared_set0_count == binding_key[0] ? binding_cache.shared_set0_layout
                                                          : VkDescriptorSetLayout(VK_NULL_HANDLE);
    if (!layout->Initialize(binding_key[0], binding_key[1], binding_key[2], binding_key[3],
                            binding_key[4], set0_shared, binding_cache.shared_set1_layout)) {
      return false;
    }
    if (binding_cache.shared_set0_layout == VK_NULL_HANDLE) {
      binding_cache.layout_device = device_;
      binding_cache.shared_set0_layout = layout->layouts[0];
      binding_cache.shared_set0_count = binding_key[0];
      binding_cache.shared_set0_pool = std::move(layout->pools[0]);
      layout->owns_prefix_layouts[0] = false;
    }
    if (binding_cache.shared_set1_layout == VK_NULL_HANDLE) {
      binding_cache.layout_device = device_;
      binding_cache.shared_set1_layout = layout->layouts[1];
      binding_cache.shared_set1_pool = std::move(layout->pools[1]);
      layout->owns_prefix_layouts[1] = false;
    }
    bindings_ = binding_cache.Insert(binding_key, std::move(layout));
  }
  for (size_t i = 0; i < sets_.size(); ++i) {
    if (i == 0 && shared_vertex_buffer != VK_NULL_HANDLE) {
      sets_[0] = (shared_buffer_count == binding_cache.shared_set0_count &&
                  binding_cache.shared_set0_pool)
                     ? binding_cache.GetOrCreateSharedVertexSet(shared_vertex_buffer,
                                                                shared_buffer_count)
                     : bindings_->GetOrCreateSharedVertexSet(shared_vertex_buffer,
                                                             shared_buffer_count);
      if (sets_[0] == VK_NULL_HANDLE) {
        return false;
      }
      continue;
    }
    // The constant set depends on the upload buffer, chosen below.
    if (i == DrawRenderState::kDynamicSet) {
      continue;
    }
    if (i >= 2) {
      // Texture/sampler sets (2 & 3) are cached per frame slot below.
      continue;
    }
    if (bindings_->empty_layouts[i]) {
      sets_[i] = bindings_->GetEmptySet(i);
      if (sets_[i] == VK_NULL_HANDLE) {
        return false;
      }
      continue;
    }
    const bool use_cache_set0 =
        i == 0 && shared_buffer_count == binding_cache.shared_set0_count &&
        binding_cache.shared_set0_pool;
    auto& pool = use_cache_set0 ? *binding_cache.shared_set0_pool : *bindings_->pools[i];
    set_indices_[i] = pool.Allocate();
    if (set_indices_[i] == SIZE_MAX) {
      return false;
    }
    set0_from_cache_ = use_cache_set0;
    sets_[i] = pool.Get(set_indices_[i]);
  }

  const auto props = device_->properties();
  const VkDeviceSize uniform_alignment = props.minUniformBufferOffsetAlignment;
  const bool use_preuploaded_constants = preuploaded_constants != nullptr;
  const std::array<std::span<const uint8_t>, Translator::kConstantBufferCount> constants{{
      {reinterpret_cast<const uint8_t*>(&input.system), sizeof(input.system)},
      {reinterpret_cast<const uint8_t*>(input.float_constants[0].data()),
       input.float_constants[0].size() * sizeof(uint32_t)},
      {reinterpret_cast<const uint8_t*>(input.float_constants[1].data()),
       input.float_constants[1].size() * sizeof(uint32_t)},
      {reinterpret_cast<const uint8_t*>(input.bool_loop.data()), sizeof(input.bool_loop)},
      {reinterpret_cast<const uint8_t*>(input.fetch.data()), sizeof(input.fetch)},
  }};
  const VkDeviceSize vertex_size =
      shared_vertex_buffer != VK_NULL_HANDLE ? 0 : input.vertices.size();
  VkDeviceSize size = vertex_size;
  std::array<VkDescriptorBufferInfo, Translator::kConstantBufferCount> constant_infos{};
  std::array<uint64_t, Translator::kConstantBufferCount> constant_offsets{};
  if (!use_preuploaded_constants) {
    for (size_t i = 0; i < constants.size(); ++i) {
      size = rex::align(size, uniform_alignment);
      constant_infos[i].offset = constant_offsets[i] = size;
      constant_infos[i].range = std::max<size_t>(constants[i].size(), 16);
      size += constant_infos[i].range;
    }
  }
  if (upload_inline_indices) {
    index_offset_ = rex::align(size, VkDeviceSize(4));
    size = index_offset_ + inline_indices.size_bytes();
  }
  for (size_t i = 0; i < input.textures.size(); ++i) {
    const auto& input_texture = input.textures[i];
    if (input_texture.fallback_view != VK_NULL_HANDLE) {
      texture_views[i] = input_texture.fallback_view;
      continue;
    }
    if (input_texture.cached_texture) {
      texture_views[i] = input_texture.cached_view;
      if (!input_texture.upload_cached) {
        continue;
      }
    }
    const uint32_t op_idx = uint32_t(ops.size());
    op_index_for_input[i] = op_idx;
    auto& op = ops.emplace_back();
    op.resolved = input_texture.resolved_image;
    op.crop_source = input_texture.crop_source;
    op.depth_pack_source = input_texture.depth_pack_source;
    op.depth_pack_float = input_texture.depth_pack_float;
    op.depth_pack_swap = input_texture.depth_pack_swap;
    for (uint32_t face = 0; face < kTextureCubeArrayLayers; ++face) {
      op.resolved_cube_faces[face] = input_texture.resolved_cube_faces[face];
    }
    op.cached = input_texture.cached_texture;
    op.upload_cached = input_texture.upload_cached;
    op.array_layers = op.resolved || input_texture.view_type == VK_IMAGE_VIEW_TYPE_3D
                          ? 1
                          : input_texture.array_layers;
    op.level_count = input_texture.level_count;
    op.extent = {input_texture.width, input_texture.height};
    if (input_texture.direct_upload_buffer != VK_NULL_HANDLE) {
      op.direct_upload_buffer = input_texture.direct_upload_buffer;
      op.uploads = input_texture.direct_uploads;
      continue;
    }
    const bool is_3d = input_texture.view_type == VK_IMAGE_VIEW_TYPE_3D;
    for (uint32_t index = 0; index < input_texture.levels.size(); ++index) {
      const uint32_t level = input_texture.first_level + index;
      auto& copy = op.uploads.emplace_back();
      copy.bufferOffset = rex::align(size, VkDeviceSize(16));
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0,
                               is_3d ? 1 : input_texture.array_layers};
      copy.imageExtent = {std::max(1u, input_texture.width >> level),
                          std::max(1u, input_texture.height >> level),
                          is_3d ? std::max(1u, input_texture.depth >> level) : 1};
      size = copy.bufferOffset + input_texture.levels[index].size();
    }
  }
  texture_op_count_ = uint32_t(ops.size() - ops_start);
  if (!use_preuploaded_constants) {
    size = ComputeDynamicUboRequiredSize(constant_offsets, kDynamicConstantBufferRanges, size);
  }
  const VkDeviceSize alignment =
      std::max({VkDeviceSize(16), uniform_alignment, props.minStorageBufferOffsetAlignment});
  VkDeviceSize buffer_offset = 0;
  uint8_t* bytes = nullptr;
  if (size > 0) {
    if (rex::align(size, alignment) <= kDrawUploadPageSize) {
      bytes = uploads.Request(submission, size, alignment, buffer_, buffer_offset);
    } else {
      if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
              device_, size, kDrawUploadUsage, ui::vulkan::util::MemoryPurpose::kUpload, buffer_,
              buffer_memory_)) {
        return false;
      }
      void* mapped = nullptr;
      if (vk.vkMapMemory(device, buffer_memory_, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
        bytes = static_cast<uint8_t*>(mapped);
      }
    }
    if (!bytes) {
      return false;
    }
  }
  if (vertex_size) {
    std::memcpy(bytes, input.vertices.data(), vertex_size);
  }
  if (use_preuploaded_constants) {
    dynamic_offsets_ = preuploaded_constants->dynamic_offsets;
    sets_[1] = binding_cache.shared_set1_pool
                   ? binding_cache.GetOrCreateConstantSet(preuploaded_constants->buffer)
                   : bindings_->GetOrCreateConstantSet(preuploaded_constants->buffer);
    if (sets_[1] == VK_NULL_HANDLE) {
      return false;
    }
  } else {
    for (size_t i = 0; i < constants.size(); ++i) {
      constant_infos[i].buffer = buffer_;
      uint8_t* target = bytes + constant_infos[i].offset;
      if (constants[i].empty()) {
        std::memset(target, 0, constant_infos[i].range);
      } else {
        std::memcpy(target, constants[i].data(), constants[i].size());
      }
      constant_infos[i].offset += buffer_offset;
      dynamic_offsets_[i] = uint32_t(constant_infos[i].offset);
    }
    if (buffer_memory_) {
      // A dedicated buffer outlives no other draw, so it gets a set of its own.
      const bool use_cache_set1 = binding_cache.shared_set1_pool != nullptr;
      auto& set1_pool = use_cache_set1 ? *binding_cache.shared_set1_pool : *bindings_->pools[1];
      set_indices_[1] = set1_pool.Allocate();
      if (set_indices_[1] == SIZE_MAX) {
        return false;
      }
      set1_from_cache_ = use_cache_set1;
      sets_[1] = set1_pool.Get(set_indices_[1]);
      WriteConstantSetDescriptors(*device_, sets_[1], buffer_);
    } else {
      sets_[1] = binding_cache.shared_set1_pool
                     ? binding_cache.GetOrCreateConstantSet(buffer_)
                     : bindings_->GetOrCreateConstantSet(buffer_);
      if (sets_[1] == VK_NULL_HANDLE) {
        return false;
      }
    }
  }
  if (upload_inline_indices) {
    std::memcpy(bytes + index_offset_, inline_indices.data(), inline_indices.size_bytes());
    index_buffer_ = buffer_;
    index_offset_ += buffer_offset;
    index_type_ = VK_INDEX_TYPE_UINT32;
  }
  for (size_t i = 0; i < input.textures.size(); ++i) {
    const uint32_t op_idx = op_index_for_input[i];
    if (op_idx == UINT32_MAX) {
      continue;
    }
    auto& op = ops[op_idx];
    op.copy_buffer = op.direct_upload_buffer != VK_NULL_HANDLE ? op.direct_upload_buffer : buffer_;
    if (op.direct_upload_buffer != VK_NULL_HANDLE) {
      continue;
    }
    for (size_t level = 0; level < op.uploads.size(); ++level) {
      const auto& pixels = input.textures[i].levels[level];
      auto& copy = op.uploads[level];
      std::memcpy(bytes + copy.bufferOffset, pixels.data(), pixels.size());
      copy.bufferOffset += buffer_offset;
    }
  }
  if (buffer_memory_) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer_memory_;
    range.size = VK_WHOLE_SIZE;
    const VkResult flushed = vk.vkFlushMappedMemoryRanges(device, 1, &range);
    vk.vkUnmapMemory(device, buffer_memory_);
    if (flushed != VK_SUCCESS) {
      return false;
    }
  }

  // The captured vertex ranges are packed at the beginning of the first segment.
  // The shared memory buffer count is at most four, so keep the bindings on the
  // stack and only fall back to the heap for an unexpectedly large count.
  constexpr size_t kInlineSharedBuffers = 8;
  std::array<VkDescriptorBufferInfo, kInlineSharedBuffers> inline_shared_infos{};
  std::vector<VkDescriptorBufferInfo> heap_shared_infos;
  std::array<VkWriteDescriptorSet, 1> writes{};
  uint32_t write_count = 0;
  if (shared_vertex_buffer == VK_NULL_HANDLE) {
    VkDescriptorBufferInfo* shared_infos;
    if (shared_buffer_count <= kInlineSharedBuffers) {
      shared_infos = inline_shared_infos.data();
    } else {
      heap_shared_infos.resize(shared_buffer_count);
      shared_infos = heap_shared_infos.data();
    }
    for (uint32_t i = 0; i < shared_buffer_count; ++i) {
      shared_infos[i] = {buffer_, buffer_offset, vertex_size};
    }
    writes[write_count++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             sets_[0],
                             0,
                             0,
                             shared_buffer_count,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             shared_infos};
  }
  if (write_count) {
    vk.vkUpdateDescriptorSets(device, write_count, writes.data(), 0, nullptr);
  }

  for (size_t i = 0; i < input.textures.size(); ++i) {
    const uint32_t op_idx = op_index_for_input[i];
    if (op_idx == UINT32_MAX) {
      continue;
    }
    const auto& input_texture = input.textures[i];
    auto& op = ops[op_idx];
    if (op.cached) {
      if (op.upload_cached && op.depth_pack_source) {
        if (!depth_resolve) {
          return false;
        }
        op.cached_storage_view = op.cached->GetOrCreateStorageView(input_texture.format);
        op.depth_pack_descriptor = depth_resolve->Allocate();
        if (op.cached_storage_view == VK_NULL_HANDLE || op.depth_pack_descriptor == SIZE_MAX) {
          return false;
        }
      }
      continue;
    }
    if (op.resolved) {
      texture_views[i] = op.resolved->GetSampledView(input_texture.swizzle);
      if (texture_views[i] == VK_NULL_HANDLE) {
        return false;
      }
      continue;
    }
    const uint32_t level_count = std::max(1u, input_texture.level_count);
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType =
        input_texture.view_type == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    info.format = input_texture.format;
    info.extent = {input_texture.width, input_texture.height, input_texture.depth};
    info.mipLevels = level_count;
    info.arrayLayers = op.array_layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (input_texture.view_type == VK_IMAGE_VIEW_TYPE_CUBE) {
      info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (!CreateDeviceImage(*device_, allocator_, info, op.image, op.memory, op.allocation)) {
      return false;
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = op.image;
    view.viewType = input_texture.view_type;
    view.format = input_texture.format;
    view.components = input_texture.swizzle;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, level_count, 0, op.array_layers};
    if (vk.vkCreateImageView(device, &view, nullptr, &op.view) != VK_SUCCESS) {
      return false;
    }
    texture_views[i] = op.view;
  }
  // Set 2 holds the vertex shader textures, set 3 the pixel shader ones.
  const auto resolve_texture_set = [&](size_t set_slot, std::span<const uint32_t> bindings,
                                       std::span<const VkSampler> samplers) -> bool {
    if (bindings_->empty_layouts[set_slot]) {
      sets_[set_slot] = bindings_->GetEmptySet(set_slot);
      return sets_[set_slot] != VK_NULL_HANDLE;
    }
    constexpr size_t kInlineViews = 32;
    std::array<VkImageView, kInlineViews> inline_views{};
    std::vector<VkImageView> heap_views;
    VkImageView* view_ptr = inline_views.data();
    if (bindings.size() > kInlineViews) {
      heap_views.resize(bindings.size());
      view_ptr = heap_views.data();
    }
    bool stage_cacheable = true;
    for (size_t i = 0; i < bindings.size(); ++i) {
      const uint32_t tex_idx = bindings[i];
      view_ptr[i] = texture_views[tex_idx];
      if (op_index_for_input[tex_idx] != UINT32_MAX &&
          ops[op_index_for_input[tex_idx]].view != VK_NULL_HANDLE) {
        stage_cacheable = false;
      }
    }
    sets_[set_slot] = bindings_->GetOrCreateTextureSet(
        set_slot, frame_slot, submission, retired_submission,
        std::span<const VkImageView>(view_ptr, bindings.size()), samplers, stage_cacheable);
    return sets_[set_slot] != VK_NULL_HANDLE;
  };
  if (!resolve_texture_set(2, input.vertex_texture_bindings, input.vertex_samplers) ||
      !resolve_texture_set(3, input.texture_bindings, input.samplers)) {
    return false;
  }

  const auto& depth = input.depth_stencil;
  const auto& front = depth.front;
  const auto& back = depth.back;
  DrawPipelineKey key{};
  key.vertex = &vertex;
  key.pixel = pixel;
  key.state0 = (uint64_t(input.topology) & 0xFu) |
               (uint64_t(input.primitive_restart != 0) << 4) |
               ((uint64_t(input.cull) & 0x3u) << 5) |
               ((uint64_t(input.front_face) & 0x1u) << 7) |
               ((uint64_t(input.color_count) & 0x7u) << 8) |
               ((uint64_t(input.samples) & 0xFu) << 11) |
               (uint64_t(depth.depthTestEnable != 0) << 15) |
               (uint64_t(depth.depthWriteEnable != 0) << 16) |
               ((uint64_t(depth.depthCompareOp) & 0x7u) << 17) |
               (uint64_t(depth.stencilTestEnable != 0) << 20) |
               ((uint64_t(front.failOp) & 0x7u) << 21) |
               ((uint64_t(front.passOp) & 0x7u) << 24) |
               ((uint64_t(front.depthFailOp) & 0x7u) << 27) |
               ((uint64_t(front.compareOp) & 0x7u) << 30) |
               ((uint64_t(back.failOp) & 0x7u) << 33) |
               ((uint64_t(back.passOp) & 0x7u) << 36) |
               ((uint64_t(back.depthFailOp) & 0x7u) << 39) |
               ((uint64_t(back.compareOp) & 0x7u) << 42) |
               ((uint64_t(shared_buffer_count) & 0x3Fu) << 45) |
               (uint64_t(input.edram_depth_clear_color != 0) << 51) |
               (uint64_t(input.depth_clamp_enable != 0) << 52) |
               (uint64_t(input.rasterizer_discard != 0) << 53);
  key.state1 = uint64_t(uint32_t(input.depth_format));
  for (uint32_t i = 0; i < input.color_count; ++i) {
    const auto& color = input.blend[i];
    key.colors[i] = (uint64_t(uint32_t(input.color_formats[i])) & 0xFFFFu) |
                    (uint64_t(color.blendEnable != 0) << 16) |
                    ((uint64_t(color.srcColorBlendFactor) & 0x1Fu) << 17) |
                    ((uint64_t(color.dstColorBlendFactor) & 0x1Fu) << 22) |
                    ((uint64_t(color.colorBlendOp) & 0x7u) << 27) |
                    ((uint64_t(color.srcAlphaBlendFactor) & 0x1Fu) << 30) |
                    ((uint64_t(color.dstAlphaBlendFactor) & 0x1Fu) << 35) |
                    ((uint64_t(color.alphaBlendOp) & 0x7u) << 40) |
                    ((uint64_t(color.colorWriteMask) & 0xFu) << 43);
  }
  if (const VkPipeline cached = pipelines.FindOrNull(key)) {
    pipeline_ = cached;
    ops_guard.committed = true;
    return true;
  }

  const VkShaderModule vs = CreateShaderModule(*device_, vertex);
  VkShaderModule ps = VK_NULL_HANDLE;
  if (input.edram_depth_clear_color) {
    ps = CreateShaderModule(*device_, kEdramDepthClear, sizeof(kEdramDepthClear));
  } else if (pixel) {
    ps = CreateShaderModule(*device_, *pixel);
  }
  const bool has_fragment = ps != VK_NULL_HANDLE;
  if (!vs || ((pixel || input.edram_depth_clear_color) && !has_fragment)) {
    vk.vkDestroyShaderModule(device, vs, nullptr);
    vk.vkDestroyShaderModule(device, ps, nullptr);
    return false;
  }
  const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
       vs, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, ps, "main"},
  }};
  const VkPipelineVertexInputStateCreateInfo vertex_info{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  assembly.topology = input.topology;
  assembly.primitiveRestartEnable = input.primitive_restart;
  VkPipelineViewportStateCreateInfo viewport_info{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport_info.viewportCount = viewport_info.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.depthClampEnable = input.depth_clamp_enable ? VK_TRUE : VK_FALSE;
  raster.rasterizerDiscardEnable = input.rasterizer_discard ? VK_TRUE : VK_FALSE;
  raster.depthBiasEnable = depth_bias_enable_ ? VK_TRUE : VK_FALSE;
  raster.frontFace = input.front_face;
  raster.cullMode = input.cull;
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo samples{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  samples.rasterizationSamples = input.samples;
  VkPipelineColorBlendStateCreateInfo blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  blend.attachmentCount = input.color_count;
  blend.pAttachments = input.blend.data();
  const std::array dynamic_states{
      VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_DEPTH_BIAS,         VK_DYNAMIC_STATE_BLEND_CONSTANTS,
      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
      VK_DYNAMIC_STATE_STENCIL_REFERENCE};
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = uint32_t(dynamic_states.size());
  dynamic.pDynamicStates = dynamic_states.data();
  VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = blend.attachmentCount;
  rendering.pColorAttachmentFormats = input.color_formats.data();
  rendering.depthAttachmentFormat = input.depth_format;
  rendering.stencilAttachmentFormat = input.depth_format;
  VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline.pNext = &rendering;
  pipeline.stageCount = has_fragment ? uint32_t(stages.size()) : 1;
  pipeline.pStages = stages.data();
  pipeline.pVertexInputState = &vertex_info;
  pipeline.pInputAssemblyState = &assembly;
  pipeline.pViewportState = &viewport_info;
  pipeline.pRasterizationState = &raster;
  pipeline.pMultisampleState = &samples;
  pipeline.pDepthStencilState = &input.depth_stencil;
  pipeline.pColorBlendState = &blend;
  pipeline.pDynamicState = &dynamic;
  pipeline.layout = bindings_->pipeline_layout;
  VkResult result;
  if (pipeline_cache_mutex) {
    std::lock_guard<std::timed_mutex> lock(*pipeline_cache_mutex);
    result =
        vk.vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline, nullptr, &pipeline_);
  } else {
    result =
        vk.vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline, nullptr, &pipeline_);
  }
  vk.vkDestroyShaderModule(device, vs, nullptr);
  vk.vkDestroyShaderModule(device, ps, nullptr);
  if (result != VK_SUCCESS) {
    return false;
  }
  pipelines.Insert(key, pipeline_);
  ops_guard.committed = true;
  return true;
}

void ReleaseDrawTextureOp(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                          DepthResolve* depth_resolve, DrawTextureOp& op) {
  if (op.depth_pack_descriptor != SIZE_MAX && depth_resolve) {
    depth_resolve->Free(op.depth_pack_descriptor);
    op.depth_pack_descriptor = SIZE_MAX;
  }
  if (op.view != VK_NULL_HANDLE) {
    device.functions().vkDestroyImageView(device.device(), op.view, nullptr);
    op.view = VK_NULL_HANDLE;
  }
  DestroyDeviceImage(device, allocator, op.image, op.memory, op.allocation);
  op.image = VK_NULL_HANDLE;
  op.memory = VK_NULL_HANDLE;
  op.allocation = VK_NULL_HANDLE;
}

void RecordDrawTextureOp(const ui::vulkan::VulkanDevice& device, DepthResolve* depth_resolve,
                         VkCommandBuffer commands, const DrawTextureOp& op) {
  const auto& vk = device.functions();
  if (op.resolved) {
    op.resolved->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kTextureStages,
                            VK_ACCESS_SHADER_READ_BIT);
    return;
  }
  if (op.cached && !op.upload_cached) {
    return;
  }
  bool resolved_faces = false;
  for (const auto* face : op.resolved_cube_faces) {
    resolved_faces |= face != nullptr;
  }
  const VkImage image = op.cached ? op.cached->image : op.image;
  if (op.depth_pack_source) {
    op.depth_pack_source->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_SHADER_READ_BIT);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, op.array_layers};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vk.vkCmdPipelineBarrier(commands, kTextureStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);
    depth_resolve->RecordPackRgba8(commands, op.depth_pack_descriptor,
                                   op.depth_pack_source->depth_view(), op.cached_storage_view,
                                   op.extent, op.depth_pack_float, op.depth_pack_swap);
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kTextureStages, 0, 0,
                            nullptr, 0, nullptr, 1, &barrier);
    return;
  }
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                              op.crop_source || resolved_faces ? 1u : op.level_count, 0,
                              op.array_layers};
  const bool cached_resolve_upload =
      op.cached && (resolved_faces || op.crop_source != nullptr);
  const VkPipelineStageFlags src_stage =
      cached_resolve_upload
          ? (kTextureStages | VK_PIPELINE_STAGE_TRANSFER_BIT)
          : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  barrier.srcAccessMask = cached_resolve_upload ? VK_ACCESS_TRANSFER_WRITE_BIT : 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vk.vkCmdPipelineBarrier(commands, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 1, &barrier);
  if (op.crop_source) {
    op.crop_source->Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {op.extent.width, op.extent.height, 1};
    vk.vkCmdCopyImage(commands, op.crop_source->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    op.crop_source->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kTextureStages,
                               VK_ACCESS_SHADER_READ_BIT);
  } else if (resolved_faces) {
    for (uint32_t face = 0; face < kTextureCubeArrayLayers; ++face) {
      ColorImage* face_image = op.resolved_cube_faces[face];
      if (!face_image) {
        continue;
      }
      face_image->Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
      VkImageCopy copy{};
      copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, face, 1};
      copy.extent = {op.extent.width, op.extent.height, 1};
      vk.vkCmdCopyImage(commands, face_image->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
      face_image->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kTextureStages,
                             VK_ACCESS_SHADER_READ_BIT);
    }
  } else {
    vk.vkCmdCopyBufferToImage(commands, op.copy_buffer, image, barrier.newLayout,
                              uint32_t(op.uploads.size()), op.uploads.data());
  }
  barrier.oldLayout = barrier.newLayout;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, kTextureStages, 0, 0, nullptr,
                          0, nullptr, 1, &barrier);
}

void DrawPass::Record(VkCommandBuffer commands, VkExtent2D extent, DrawRenderState& state) {
  if (!rasterizer_discard_ &&
      (!viewport_.xy_extent[0] || !viewport_.xy_extent[1] || scissor_.offset[0] >= extent.width ||
       scissor_.offset[1] >= extent.height)) {
    return;
  }
  const auto& vk = device_->functions();
  const VkViewport viewport{float(viewport_.xy_offset[0]),
                            float(viewport_.xy_offset[1]),
                            float(viewport_.xy_extent[0]),
                            float(viewport_.xy_extent[1]),
                            viewport_.z_min,
                            viewport_.z_max};
  const VkRect2D scissor{{int32_t(scissor_.offset[0]), int32_t(scissor_.offset[1])},
                         {std::min(scissor_.extent[0], extent.width - scissor_.offset[0]),
                          std::min(scissor_.extent[1], extent.height - scissor_.offset[1])}};
  if (state.SetPipeline(pipeline_)) {
    vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  }
  const auto changed = state.GetChangedSets(bindings_->pipeline_layout, bindings_->layouts, sets_,
                                            dynamic_offsets_);
  if (!changed.empty()) {
    const uint32_t count = changed.last - changed.first + 1;
    // The dynamic offsets belong to the constant set, so only pass them when set 1 is in range.
    const bool with_constants =
        changed.first <= DrawRenderState::kDynamicSet &&
        changed.last >= DrawRenderState::kDynamicSet;
    vk.vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               bindings_->pipeline_layout, changed.first, count,
                               sets_.data() + changed.first,
                               with_constants ? uint32_t(dynamic_offsets_.size()) : 0,
                               with_constants ? dynamic_offsets_.data() : nullptr);
  }
  if (DrawRenderState::Update(state.viewport_valid, state.viewport, viewport)) {
    vk.vkCmdSetViewport(commands, 0, 1, &viewport);
  }
  if (DrawRenderState::Update(state.scissor_valid, state.scissor, scissor)) {
    vk.vkCmdSetScissor(commands, 0, 1, &scissor);
  }
  if (depth_bias_enable_ &&
      DrawRenderState::Update(state.depth_bias_valid, state.depth_bias,
                              {depth_bias_.constant_factor, depth_bias_.slope_factor})) {
    vk.vkCmdSetDepthBias(commands, depth_bias_.constant_factor, 0.0f, depth_bias_.slope_factor);
  }
  if (DrawRenderState::Update(state.blend_constants_valid, state.blend_constants,
                              blend_constants_)) {
    vk.vkCmdSetBlendConstants(commands, blend_constants_.data());
  }
  if (stencil_enable_) {
    const bool split_stencil = device_->properties().separateStencilMaskRef;
    const auto set_stencil = [&](auto set, const std::array<uint32_t, 2>& values,
                                 std::array<uint32_t, 2>& current, uint32_t& valid) {
      const std::array<uint32_t, 2> wanted =
          split_stencil ? values : std::array<uint32_t, 2>{values[0], values[0]};
      const uint32_t faces = DrawRenderState::UpdateStencil(current, valid, wanted);
      if (faces == 3 && wanted[0] == wanted[1]) {
        set(commands, VK_STENCIL_FACE_FRONT_AND_BACK, wanted[0]);
        return;
      }
      if (faces & 1) {
        set(commands, VK_STENCIL_FACE_FRONT_BIT, wanted[0]);
      }
      if (faces & 2) {
        set(commands, VK_STENCIL_FACE_BACK_BIT, wanted[1]);
      }
    };
    set_stencil(vk.vkCmdSetStencilCompareMask, stencil_compare_mask_, state.stencil_compare_mask,
                state.stencil_compare_mask_valid);
    set_stencil(vk.vkCmdSetStencilWriteMask, stencil_write_mask_, state.stencil_write_mask,
                state.stencil_write_mask_valid);
    set_stencil(vk.vkCmdSetStencilReference, stencil_reference_, state.stencil_reference,
                state.stencil_reference_valid);
  }
  if (edram_depth_clear_color_) {
    vk.vkCmdPushConstants(commands, bindings_->pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                          sizeof(edram_clear_constants_), edram_clear_constants_.data());
  }
  if (indexed_) {
    if (state.SetIndexBuffer(index_buffer_, index_offset_, index_type_)) {
      vk.vkCmdBindIndexBuffer(commands, index_buffer_, index_offset_, index_type_);
    }
    vk.vkCmdDrawIndexed(commands, index_count_, 1, 0, 0, 0);
  } else {
    vk.vkCmdDraw(commands, index_count_, 1, 0, 0);
  }
}

}  // namespace rex::graphics::native
