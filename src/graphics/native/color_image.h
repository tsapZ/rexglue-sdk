#pragma once

#include <map>
#include <memory>
#include <utility>

#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/mem_alloc.h>

#include "draw_input.h"

namespace rex::graphics::native {

// A device-local image sub-allocated from `allocator`, or with a dedicated
// allocation when the allocator is VK_NULL_HANDLE (VMA unavailable).
bool CreateDeviceImage(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                       const VkImageCreateInfo& info, VkImage& image, VkDeviceMemory& memory,
                       VmaAllocation& allocation);
void DestroyDeviceImage(const ui::vulkan::VulkanDevice& device, VmaAllocator allocator,
                        VkImage image, VkDeviceMemory memory, VmaAllocation allocation);

// A host image representing an EDRAM attachment or resolved texture.
class ColorImage {
 public:
  explicit ColorImage(const ui::vulkan::VulkanDevice& device,
                      VmaAllocator allocator = VK_NULL_HANDLE)
      : device_(device), allocator_(allocator) {}
  ~ColorImage();

  ColorImage(const ColorImage&) = delete;
  ColorImage& operator=(const ColorImage&) = delete;

  // array_layers > 1 for resolves into slices of an array destination.
  bool Initialize(VkExtent2D extent, VkFormat format = kFormat,
                  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                  VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT,
                  uint32_t array_layers = 1);
  void Transition(VkCommandBuffer commands, VkImageLayout layout, VkPipelineStageFlags stage,
                  VkAccessFlags access);
  void Clear(VkCommandBuffer commands, const VkClearColorValue& color);

  VkImageAspectFlags aspects() const { return aspects_; }
  VkSampleCountFlagBits samples() const { return samples_; }
  VkFormat format() const { return format_; }
  VkImage image() const { return image_; }
  VkImageView view() const { return view_; }
  VkImageView depth_view() const { return depth_view_; }
  // Returns a cached sampler view with the requested component mapping,
  // creating it on first use. The view is owned by this image and stays valid
  // until destruction, so callers must not destroy it.
  VkImageView GetSampledView(VkComponentMapping swizzle);
  VkExtent2D extent() const { return extent_; }
  uint32_t array_layers() const { return array_layers_; }
  void MarkResolved() { unpublished_resolve_ = true; }
  bool TakeResolvedOutput() { return std::exchange(unpublished_resolve_, false); }
  bool initialized() const { return layout_ != VK_IMAGE_LAYOUT_UNDEFINED; }
  void set_view_purger(const DrawBindingsViewPurger* purger) { view_purger_ = purger; }
  uint64_t last_keepalive_submission() const { return last_keepalive_submission_; }
  void set_last_keepalive_submission(uint64_t submission) {
    last_keepalive_submission_ = submission;
  }

  static constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

 private:
  const ui::vulkan::VulkanDevice& device_;
  VmaAllocator allocator_;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VkExtent2D extent_{};
  uint32_t array_layers_ = 1;
  VkFormat format_ = kFormat;
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkImageAspectFlags aspects_ = VK_IMAGE_ASPECT_COLOR_BIT;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkImageView depth_view_ = VK_NULL_HANDLE;
  // Sampler views keyed by PackTextureComponentMapping, created on demand when
  // the image is sampled with different component swizzles.
  std::map<uint32_t, VkImageView> sampled_views_;
  VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  VkPipelineStageFlags stage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkAccessFlags access_ = 0;
  bool unpublished_resolve_ = false;
  const DrawBindingsViewPurger* view_purger_ = nullptr;
  uint64_t last_keepalive_submission_ = 0;
};

struct CachedImage {
  std::shared_ptr<ColorImage> image;
  uint64_t last_use_submission = 0;
};

// Submission indices shared by the caches: `next` is the submission being
// recorded, and everything up to `retired` has completed on the GPU.
struct SubmissionClock {
  uint64_t next = 1;
  uint64_t retired = 0;
};

}  // namespace rex::graphics::native
