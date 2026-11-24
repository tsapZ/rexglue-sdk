#version 450

layout(location = 0) out vec4 out_color;
layout(push_constant) uniform ClearInfo {
  uint stencil;
  uint format;
} info;

void main() {
  uint depth24 = uint(roundEven(clamp(gl_FragCoord.z, 0.0, 1.0) * 16777215.0));
  uint word = (depth24 << 8u) | (info.stencil & 0xFFu);
  if (info.format == 1u) {
    out_color = vec4(float(word & 1023u) / 1023.0,
                     float((word >> 10u) & 1023u) / 1023.0,
                     float((word >> 20u) & 1023u) / 1023.0,
                     float(word >> 30u) / 3.0);
  } else if (info.format == 2u) {
    out_color = vec4(unpackHalf2x16(word), 0.0, 0.0);
  } else if (info.format == 3u) {
    out_color = vec4(uintBitsToFloat(word), 0.0, 0.0, 0.0);
  } else {
    out_color = vec4(float(word & 255u) / 255.0,
                     float((word >> 8u) & 255u) / 255.0,
                     float((word >> 16u) & 255u) / 255.0,
                     float(word >> 24u) / 255.0);
  }
}
