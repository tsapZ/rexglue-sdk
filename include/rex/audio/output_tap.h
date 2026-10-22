#pragma once

// A tap on the audio the PC plays: the host can see each block of the game's sound as it goes to
// the output device, e.g. to record a game's title music for a launcher. One tap at a time.

#include <cstdint>

namespace rex::audio {

// Called on the audio thread with each block: `frames` frames of `channels` interleaved float
// samples (2 or 6, as the device plays them) at 48 kHz, with the output gain applied. It also sees
// the sound while audio_mute silences the device. Each of the game's audio clients has its own
// stream (`source`, one per client: a game may play music and effects on two), which the PC mixes;
// a recording sums them. Keep it short; it must not block.
using OutputTap = void (*)(const void* source, const float* samples, uint32_t frames, uint32_t channels,
                           void* user);

// Sets the tap (nullptr removes it).
void SetOutputTap(OutputTap tap, void* user);

}  // namespace rex::audio
