#pragma once

// Holding the game still for a menu: while held, the guest's next VdSwap waits until it is
// released. VdSwap is where the game hands over a finished frame, so its render thread stops at a
// point where it holds no host lock (unlike suspending threads wherever they are); the threads that
// wait for that frame stop with it. The window, the overlays and the sound output go on.

namespace rex::system {

// Holds (true) or releases (false) the game's frames. Releasing wakes a waiting VdSwap at once.
void SetGuestFramesHeld(bool held);
bool GuestFramesHeld();

// For VdSwap: returns once the frames aren't held (at once when they aren't).
void WaitWhileGuestFramesHeld();

}  // namespace rex::system
