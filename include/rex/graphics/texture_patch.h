// Lets game code redraw a texture when the GPU first loads it (Direct3D 12). A texture is named by
// the hash texture_dump_dir gives its file (its top level's content, format and size). The
// callback gets that top level as such a dump has it (untiled, little-endian: DXT1/3/5 blocks, or
// 8888 texels as B, G, R, A bytes) and may give back a whole DDS file in memory of the same
// format, any size, with mips, as texture_replace_dir takes them; a patch wins over a file of the
// same hash. It runs on a replacement worker (on the GPU thread without texture_replace_async), so
// it must only work on its input. Used for button prompts kept in an atlas with other art: only
// the buttons' cells are redrawn, the rest stays the player's own disc's.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rex::graphics {

using TexturePatch = bool (*)(uint64_t hash, uint32_t width, uint32_t height, const uint8_t* data,
                              size_t size, std::vector<uint8_t>& dds, void* user);

// A null `fn` takes the hash's patch away.
void RegisterTexturePatch(uint64_t hash, TexturePatch fn, void* user);
void ClearTexturePatches();
// Without any, the texture cache skips this entirely.
bool HasTexturePatches();
bool HasTexturePatch(uint64_t hash);
// Runs the callback registered for `hash`; false when there is none or it gave nothing.
bool ApplyTexturePatch(uint64_t hash, uint32_t width, uint32_t height, const uint8_t* data,
                       size_t size, std::vector<uint8_t>& dds);

}  // namespace rex::graphics
