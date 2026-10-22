static const AVCodec* const codec_list[] = {
#if CONFIG_MP3FLOAT_DECODER
    &ff_mp3float_decoder,
#endif
#if CONFIG_MP3_DECODER
    &ff_mp3_decoder,
#endif
#if CONFIG_WMAPRO_DECODER
    &ff_wmapro_decoder,
#endif
#if CONFIG_WMAV2_DECODER
    &ff_wmav2_decoder,
#endif
#if CONFIG_XMAFRAMES_DECODER
    &ff_xmaframes_decoder,
#endif
    // Whole XMA2 streams (2048-byte packets), for tools that decode a file from the disc instead
    // of the emulated XMA hardware's frames: wmaprodec.c builds it unconditionally, but the
    // generated config leaves CONFIG_XMA2_DECODER at 0, so it is listed without the guard.
    &ff_xma2_decoder,
    NULL};
