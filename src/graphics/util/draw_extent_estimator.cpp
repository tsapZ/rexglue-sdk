/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/format/ucode.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw_extent_estimator.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>
#include <rex/ui/graphics_util.h>

REXCVAR_DEFINE_BOOL(execute_unclipped_draw_vs_on_cpu, false, "GPU",
                    "Execute unclipped draw vertex shader on CPU");

REXCVAR_DEFINE_BOOL(execute_unclipped_draw_vs_on_cpu_with_scissor, false, "GPU",
                    "Execute unclipped draw VS on CPU with scissor");

// DEFINE_bool(
//     execute_unclipped_draw_vs_on_cpu, true,
//     "Execute the vertex shader for draws with clipping disabled, primarily "
//     "screen-space draws (such as clears), on the CPU when possible to estimate "
//     "the extent of the EDRAM involved in the draw.\n"
//     "Enabling this may significantly improve GPU performance as otherwise up "
//     "to the entire EDRAM may be considered used in draws without clipping, "
//     "potentially resulting in spurious EDRAM range ownership transfer round "
//     "trips between host render targets.\n"
//     "Also, on hosts where certain render target formats have to be emulated in "
//     "a lossy way (for instance, 16-bit fixed-point via 16-bit floating-point), "
//     "this prevents corruption of other render targets located after the "
//     "current ones in the EDRAM by lossy range ownership transfers done for "
//     "those draws.",
//     "GPU");
// DEFINE_bool(
//     execute_unclipped_draw_vs_on_cpu_with_scissor, false,
//     "Don't restrict the usage of execute_unclipped_draw_vs_on_cpu to only "
//     "non-scissored draws (with the right and the bottom sides of the scissor "
//     "rectangle at 8192 or beyond) even though if the scissor rectangle is "
//     "present, it's usually sufficient for esimating the height of the render "
//     "target.\n"
//     "Enabling this may cause excessive processing of vertices on the CPU, as "
//     "some games draw rectangles (for their UI, for instance) without clipping, "
//     "but with a proper scissor rectangle.",
//     "GPU");

namespace rex::graphics {

void DrawExtentEstimator::PositionYExportSink::Export(ucode::ExportRegister export_register,
                                                      const float* value, uint32_t value_mask) {
  if (export_register == ucode::ExportRegister::kVSPosition) {
    if (value_mask & 0b0010) {
      position_y_ = value[1];
    }
    if (value_mask & 0b1000) {
      position_w_ = value[3];
    }
  } else if (export_register == ucode::ExportRegister::kVSPointSizeEdgeFlagKillVertex) {
    if (value_mask & 0b0001) {
      point_size_ = value[0];
    }
    if (value_mask & 0b0100) {
      vertex_kill_ = rex::memory::Reinterpret<uint32_t>(value[2]);
    }
  }
}

void DrawExtentEstimator::PositionExportSink::Export(ucode::ExportRegister export_register,
                                                     const float* value, uint32_t value_mask) {
  if (export_register == ucode::ExportRegister::kVSPosition) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (value_mask & (uint32_t(1) << i)) {
        position_[i] = value[i];
      }
    }
  } else if (export_register == ucode::ExportRegister::kVSPointSizeEdgeFlagKillVertex) {
    if (value_mask & 0b0100) {
      vertex_kill_ = rex::memory::Reinterpret<uint32_t>(value[2]);
    }
  }
}

bool DrawExtentEstimator::ShapeCoversRectangle(const Shader& vertex_shader, int32_t x0,
                                               int32_t y0, int32_t x1, int32_t y1) {
  const RegisterFile& regs = register_file_;
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  bool rectangle = vgt_draw_initiator.prim_type == xenos::PrimitiveType::kRectangleList;
  if ((vgt_draw_initiator.prim_type != xenos::PrimitiveType::kTriangleList && !rectangle) ||
      vgt_draw_initiator.num_indices != 3 ||
      (vgt_draw_initiator.source_select != xenos::SourceSelect::kDMA &&
       vgt_draw_initiator.source_select != xenos::SourceSelect::kAutoIndex)) {
    return false;
  }
  if (xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type) &&
      regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select ==
          xenos::VGTOutputPath::kTessellationEnable) {
    return false;
  }
  if (!vertex_shader.is_ucode_analyzed() || !ShaderInterpreter::CanInterpretShader(vertex_shader)) {
    return false;
  }
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  if (pa_cl_clip_cntl.ucp_ena ||
      (!pa_cl_clip_cntl.clip_disable && (pa_cl_vte_cntl.vtx_xy_fmt || pa_cl_vte_cntl.vtx_z_fmt))) {
    return false;
  }

  // The indices, as EstimateVertexMaxY reads them.
  auto vgt_dma_size = regs.Get<reg::VGT_DMA_SIZE>();
  const uint16_t* index_buffer_16 = nullptr;
  const uint32_t* index_buffer_32 = nullptr;
  xenos::Endian index_endian = vgt_dma_size.swap_mode;
  if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
    if (vgt_dma_size.num_words < 3) {
      return false;
    }
    uint32_t index_buffer_base = regs[XE_GPU_REG_VGT_DMA_BASE];
    if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16) {
      if (index_endian == xenos::Endian::k8in32) {
        index_endian = xenos::Endian::k8in16;
      } else if (index_endian == xenos::Endian::k16in32) {
        index_endian = xenos::Endian::kNone;
      }
      index_buffer_16 = memory_.TranslatePhysical<const uint16_t*>(
          index_buffer_base & ~uint32_t(sizeof(uint16_t) - 1));
    } else {
      index_buffer_32 = memory_.TranslatePhysical<const uint32_t*>(
          index_buffer_base & ~uint32_t(sizeof(uint32_t) - 1));
    }
  }
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  uint32_t reset_index = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
  uint32_t index_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  uint32_t min_index = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
  uint32_t max_index = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;

  float x_scale = pa_cl_vte_cntl.vport_x_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XSCALE)
                                                   : 1.0f;
  float x_offset = pa_cl_vte_cntl.vport_x_offset_ena
                       ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XOFFSET)
                       : 0.0f;
  float y_scale = pa_cl_vte_cntl.vport_y_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE)
                                                   : 1.0f;
  float y_offset = pa_cl_vte_cntl.vport_y_offset_ena
                       ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET)
                       : 0.0f;
  if (pa_su_sc_mode_cntl.vtx_window_offset_enable) {
    auto pa_sc_window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
    x_offset += float(pa_sc_window_offset.window_x_offset);
    y_offset += float(pa_sc_window_offset.window_y_offset);
  }
  if (!pa_cl_vte_cntl.vtx_xy_fmt) {
    // The viewport clips too: it must hold the centers of the rectangle's pixels (a full-screen pass
    // has it exactly the size of the target; half a pixel either way for the pixel center offset).
    if (x_offset - std::abs(x_scale) > float(x0) + 0.5f ||
        x_offset + std::abs(x_scale) < float(x1) - 0.5f ||
        y_offset - std::abs(y_scale) > float(y0) + 0.5f ||
        y_offset + std::abs(y_scale) < float(y1) - 0.5f) {
      return false;
    }
  }

  // Screen positions, and clip space z and w for a rectangle's fourth vertex.
  float screen_x[3], screen_y[3];
  float clip_z[3], clip_w[3];
  shader_interpreter_.SetShader(vertex_shader);
  PositionExportSink sink;
  shader_interpreter_.SetExportSink(&sink);
  bool vertices_usable = true;
  for (uint32_t i = 0; i < 3 && vertices_usable; ++i) {
    uint32_t vertex_index = i;
    if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
      vertex_index = index_buffer_16 ? uint32_t(index_buffer_16[i]) : index_buffer_32[i];
      vertex_index = xenos::GpuSwap(vertex_index, index_endian) & 0xFFFFFF;
      if (pa_su_sc_mode_cntl.multi_prim_ib_ena && vertex_index == reset_index) {
        vertices_usable = false;
        break;
      }
    }
    vertex_index =
        std::min(max_index, std::max(min_index, (vertex_index + index_offset) & 0xFFFFFF));
    sink.Reset();
    shader_interpreter_.temp_registers()[0] = float(vertex_index);
    shader_interpreter_.Execute();
    if ((sink.vertex_kill().has_value() && (sink.vertex_kill().value() & ~(UINT32_C(1) << 31))) ||
        !sink.position(0).has_value() || !sink.position(1).has_value()) {
      vertices_usable = false;
      break;
    }
    float x = sink.position(0).value();
    float y = sink.position(1).value();
    clip_z[i] = sink.position(2).value_or(0.0f);
    clip_w[i] = sink.position(3).value_or(1.0f);
    if (!pa_cl_vte_cntl.vtx_xy_fmt || !pa_cl_clip_cntl.clip_disable) {
      if (!sink.position(3).has_value() || !(sink.position(3).value() > 0.0f)) {
        vertices_usable = false;
        break;
      }
    }
    if (!pa_cl_clip_cntl.clip_disable) {
      // Depth within the clip range at every vertex: then nowhere in the triangle is it clipped.
      if (!sink.position(2).has_value()) {
        vertices_usable = false;
        break;
      }
      float z = sink.position(2).value();
      float w = sink.position(3).value();
      if (!(z <= w && z >= (pa_cl_clip_cntl.dx_clip_space_def ? 0.0f : -w))) {
        vertices_usable = false;
        break;
      }
    }
    if (!pa_cl_vte_cntl.vtx_xy_fmt) {
      float w = sink.position(3).value();
      x /= w;
      y /= w;
    }
    screen_x[i] = x * x_scale + x_offset;
    screen_y[i] = y * y_scale + y_offset;
    if (!std::isfinite(screen_x[i]) || !std::isfinite(screen_y[i])) {
      vertices_usable = false;
    }
  }
  shader_interpreter_.SetExportSink(nullptr);
  if (!vertices_usable) {
    return false;
  }

  // The shape the rasterizer fills, as a convex polygon: the triangle, or for a rectangle list the
  // strip the host's geometry shader makes of it (pipeline_cache.cpp, PipelineGeometryShader::
  // kRectangleList): the vertex opposite the longest edge mirrored across that edge, v3 = -va + vb
  // + vc. The shader compares the edges in the host's clip space, which is scaled per axis, so only
  // rectangles are taken where no such scaling changes its pick: a right angle at a with both legs
  // along the axes (the hypotenuse b-c is then the longest under any scaling), the same w at every
  // vertex (the mirroring is then the same on the screen). In polygon order that is a, b, v3, c; the
  // strip's second triangle shares the edge b-c, where the top-left rule gives every pixel center to
  // exactly one of the two.
  double polygon_x[4], polygon_y[4];
  uint32_t polygon_size = 3;
  for (uint32_t i = 0; i < 3; ++i) {
    polygon_x[i] = screen_x[i];
    polygon_y[i] = screen_y[i];
  }
  if (rectangle) {
    auto axis_legs = [&](uint32_t a, uint32_t b, uint32_t c) {
      float bx = screen_x[b] - screen_x[a], by = screen_y[b] - screen_y[a];
      float cx = screen_x[c] - screen_x[a], cy = screen_y[c] - screen_y[a];
      bool b_horizontal = std::abs(by) < 1e-3f && std::abs(bx) > 1.0f;
      bool b_vertical = std::abs(bx) < 1e-3f && std::abs(by) > 1.0f;
      bool c_horizontal = std::abs(cy) < 1e-3f && std::abs(cx) > 1.0f;
      bool c_vertical = std::abs(cx) < 1e-3f && std::abs(cy) > 1.0f;
      return (b_horizontal && c_vertical) || (b_vertical && c_horizontal);
    };
    uint32_t a = 3, b = 0, c = 0;
    for (uint32_t i = 0; i < 3 && a == 3; ++i) {
      if (axis_legs(i, (i + 1) % 3, (i + 2) % 3)) {
        a = i;
        b = (i + 1) % 3;
        c = (i + 2) % 3;
      }
    }
    if (a == 3) {
      return false;
    }
    if (!pa_cl_vte_cntl.vtx_xy_fmt &&
        (std::abs(clip_w[b] - clip_w[a]) > 1e-6f * std::abs(clip_w[a]) ||
         std::abs(clip_w[c] - clip_w[a]) > 1e-6f * std::abs(clip_w[a]))) {
      return false;
    }
    if (!pa_cl_clip_cntl.clip_disable) {
      // The fourth vertex's depth must be in the clip range too.
      float z = -clip_z[a] + clip_z[b] + clip_z[c];
      float w = -clip_w[a] + clip_w[b] + clip_w[c];
      if (!(w > 0.0f) || !(z <= w && z >= (pa_cl_clip_cntl.dx_clip_space_def ? 0.0f : -w))) {
        return false;
      }
    }
    polygon_x[0] = screen_x[a];
    polygon_y[0] = screen_y[a];
    polygon_x[1] = screen_x[b];
    polygon_y[1] = screen_y[b];
    polygon_x[2] = -double(screen_x[a]) + double(screen_x[b]) + double(screen_x[c]);
    polygon_y[2] = -double(screen_y[a]) + double(screen_y[b]) + double(screen_y[c]);
    polygon_x[3] = screen_x[c];
    polygon_y[3] = screen_y[c];
    polygon_size = 4;
  }

  // Twice the signed area; positive is clockwise on the render target (y down). Every turn must go
  // the same way.
  double area = 0.0;
  for (uint32_t i = 0; i < polygon_size; ++i) {
    uint32_t j = (i + 1) % polygon_size;
    area += polygon_x[i] * polygon_y[j] - polygon_x[j] * polygon_y[i];
  }
  if (!(std::abs(area) > 1.0)) {
    return false;
  }
  for (uint32_t i = 0; i < polygon_size; ++i) {
    uint32_t j = (i + 1) % polygon_size, k = (i + 2) % polygon_size;
    double turn = (polygon_x[j] - polygon_x[i]) * (polygon_y[k] - polygon_y[j]) -
                  (polygon_y[j] - polygon_y[i]) * (polygon_x[k] - polygon_x[j]);
    if (!(turn * area > 0.0)) {
      return false;
    }
  }
  // With culling on, a triangle must face the side that is drawn. Facing as the host rasterizer
  // gets it (pipeline_cache: FrontCounterClockwise = face == 0), on the render target with y down.
  // Rectangles are never culled (draw_util::IsPrimitivePolygonal: not polygonal).
  if (!rectangle && (pa_su_sc_mode_cntl.cull_front || pa_su_sc_mode_cntl.cull_back)) {
    bool clockwise = area > 0.0;
    bool front = pa_su_sc_mode_cntl.face ? clockwise : !clockwise;
    if (front ? pa_su_sc_mode_cntl.cull_front : pa_su_sc_mode_cntl.cull_back) {
      return false;
    }
  }

  // The centers of the rectangle's corner pixels must be covered by the shape as the rasterizer
  // decides it: inside, or exactly on a top or left edge (the top-left rule; a full-screen triangle
  // has two of its edges right on the viewport's sides). With D3D9 pixel centers (the half-pixel
  // offset) the centers are at whole coordinates, else half a pixel in. The shape is convex, so
  // with its corners the whole rectangle is inside.
  double center = (REXCVAR_GET(half_pixel_offset) &&
                   regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero)
                      ? 0.0
                      : 0.5;
  const double corners_x[4] = {x0 + center, x1 - 1 + center, x0 + center, x1 - 1 + center};
  const double corners_y[4] = {y0 + center, y0 + center, y1 - 1 + center, y1 - 1 + center};
  // The edge function is flipped so that inside is positive.
  double sign = area > 0.0 ? 1.0 : -1.0;
  for (uint32_t e = 0; e < polygon_size; ++e) {
    uint32_t a = e, b = (e + 1) % polygon_size;
    double dx = polygon_x[b] - polygon_x[a];
    double dy = polygon_y[b] - polygon_y[a];
    double length = std::sqrt(dx * dx + dy * dy);
    // The inward normal (y down): a top edge is horizontal with the inside below it, a left edge
    // has the inside to its right.
    double normal_x = -dy * sign, normal_y = dx * sign;
    bool top_left = (std::abs(dy) < 1e-6 * length) ? (normal_y > 0.0) : (normal_x > 0.0);
    for (uint32_t c = 0; c < 4; ++c) {
      // How far inside the corner pixel's center is, in pixels.
      double inside =
          (dx * (corners_y[c] - polygon_y[a]) - dy * (corners_x[c] - polygon_x[a])) * sign / length;
      if (!(inside > 1e-3 || (top_left && inside > -1e-3))) {
        return false;
      }
    }
  }
  return true;
}

uint32_t DrawExtentEstimator::EstimateVertexMaxY(const Shader& vertex_shader) {
  SCOPE_profile_cpu_f("gpu");

  const RegisterFile& regs = register_file_;

  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  if (!vgt_draw_initiator.num_indices) {
    return 0;
  }
  if (vgt_draw_initiator.source_select != xenos::SourceSelect::kDMA &&
      vgt_draw_initiator.source_select != xenos::SourceSelect::kAutoIndex) {
    // TODO(Triang3l): Support immediate indices.
    return xenos::kTexture2DCubeMaxWidthHeight;
  }

  // Not reproducing tessellation.
  if (xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type) &&
      regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select ==
          xenos::VGTOutputPath::kTessellationEnable) {
    return xenos::kTexture2DCubeMaxWidthHeight;
  }

  assert_true(vertex_shader.type() == xenos::ShaderType::kVertex);
  assert_true(vertex_shader.is_ucode_analyzed());
  if (!ShaderInterpreter::CanInterpretShader(vertex_shader)) {
    return xenos::kTexture2DCubeMaxWidthHeight;
  }

  auto vgt_dma_size = regs.Get<reg::VGT_DMA_SIZE>();
  union {
    const void* index_buffer;
    const uint16_t* index_buffer_16;
    const uint32_t* index_buffer_32;
  };
  xenos::Endian index_endian = vgt_dma_size.swap_mode;
  if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
    xenos::IndexFormat index_format = vgt_draw_initiator.index_size;
    uint32_t index_buffer_base = regs[XE_GPU_REG_VGT_DMA_BASE];
    if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16) {
      // Handle the index endianness to same way as the PrimitiveProcessor.
      if (index_endian == xenos::Endian::k8in32) {
        index_endian = xenos::Endian::k8in16;
      } else if (index_endian == xenos::Endian::k16in32) {
        index_endian = xenos::Endian::kNone;
      }
      index_buffer_base &= ~uint32_t(sizeof(uint16_t) - 1);
    } else {
      assert_true(vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32);
      index_buffer_base &= ~uint32_t(sizeof(uint32_t) - 1);
    }
    index_buffer = memory_.TranslatePhysical(index_buffer_base);
  }
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  uint32_t reset_index = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
  uint32_t index_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  uint32_t min_index = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
  uint32_t max_index = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;

  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  float viewport_y_scale =
      pa_cl_vte_cntl.vport_y_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0f;
  float viewport_y_offset =
      pa_cl_vte_cntl.vport_y_offset_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET) : 0.0f;

  int32_t point_vertex_min_diameter_float = 0;
  int32_t point_vertex_max_diameter_float = 0;
  float point_constant_radius_y = 0.0f;
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    point_vertex_min_diameter_float =
        rex::memory::Reinterpret<int32_t>(float(pa_su_point_minmax.min_size) * (2.0f / 16.0f));
    point_vertex_max_diameter_float =
        rex::memory::Reinterpret<int32_t>(float(pa_su_point_minmax.max_size) * (2.0f / 16.0f));
    point_constant_radius_y = float(regs.Get<reg::PA_SU_POINT_SIZE>().height) * (1.0f / 16.0f);
  }

  float max_y = -FLT_MAX;

  shader_interpreter_.SetShader(vertex_shader);

  PositionYExportSink position_y_export_sink;
  shader_interpreter_.SetExportSink(&position_y_export_sink);
  for (uint32_t i = 0; i < vgt_draw_initiator.num_indices; ++i) {
    uint32_t vertex_index;
    if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
      if (i < vgt_dma_size.num_words) {
        if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16) {
          vertex_index = index_buffer_16[i];
        } else {
          vertex_index = index_buffer_32[i];
        }
        // The Xenos only uses 24 bits of the index (reset_indx is 24-bit).
        vertex_index = xenos::GpuSwap(vertex_index, index_endian) & 0xFFFFFF;
      } else {
        vertex_index = 0;
      }
      if (pa_su_sc_mode_cntl.multi_prim_ib_ena && vertex_index == reset_index) {
        continue;
      }
    } else {
      assert_true(vgt_draw_initiator.source_select == xenos::SourceSelect::kAutoIndex);
      vertex_index = i;
    }
    vertex_index =
        std::min(max_index, std::max(min_index, (vertex_index + index_offset) & 0xFFFFFF));

    position_y_export_sink.Reset();

    shader_interpreter_.temp_registers()[0] = float(vertex_index);
    shader_interpreter_.Execute();

    if (position_y_export_sink.vertex_kill().has_value() &&
        (position_y_export_sink.vertex_kill().value() & ~(UINT32_C(1) << 31))) {
      continue;
    }
    if (!position_y_export_sink.position_y().has_value()) {
      continue;
    }
    float vertex_y = position_y_export_sink.position_y().value();
    if (!pa_cl_vte_cntl.vtx_xy_fmt) {
      if (!position_y_export_sink.position_w().has_value()) {
        continue;
      }
      vertex_y /= position_y_export_sink.position_w().value();
    }

    vertex_y = vertex_y * viewport_y_scale + viewport_y_offset;

    if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
      float point_radius_y;
      if (position_y_export_sink.point_size().has_value()) {
        // Vertex-specified diameter. Clamped effectively as a signed integer in
        // the hardware, -NaN, -Infinity ... -0 to the minimum, +Infinity, +NaN
        // to the maximum.
        point_radius_y = 0.5f * rex::memory::Reinterpret<float>(std::min(
                                    point_vertex_max_diameter_float,
                                    std::max(point_vertex_min_diameter_float,
                                             rex::memory::Reinterpret<int32_t>(
                                                 position_y_export_sink.point_size().value()))));
      } else {
        // Constant radius.
        point_radius_y = point_constant_radius_y;
      }
      vertex_y += point_radius_y;
    }

    // std::max is `a < b ? b : a`, thus in case of NaN, the first argument is
    // always returned - max_y, which is initialized to a normalized value.
    max_y = std::max(max_y, vertex_y);
  }
  shader_interpreter_.SetExportSink(nullptr);

  int32_t max_y_24p8 = ui::FloatToD3D11Fixed16p8(max_y);
  // 16p8 range is -32768 to 32767+255/256, but it's stored as uint32_t here,
  // as 24p8, so overflowing up to -8388608 to 8388608+255/256 is safe. The
  // range of the window offset plus the half-pixel offset is -16384 to 16384.5,
  // so it's safe to add both - adding it will neither move the 16p8 clamping
  // bounds -32768 and 32767+255/256 into the 0...8192 screen space range, nor
  // cause 24p8 overflow.
  if (regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero) {
    max_y_24p8 += 128;
  }
  if (pa_su_sc_mode_cntl.vtx_window_offset_enable) {
    max_y_24p8 += regs.Get<reg::PA_SC_WINDOW_OFFSET>().window_y_offset * 256;
  }
  // Top-left rule - .5 exclusive without MSAA, 1. exclusive with MSAA.
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  return (uint32_t(std::max(int32_t(0), max_y_24p8)) +
          ((rb_surface_info.msaa_samples == xenos::MsaaSamples::k1X) ? 127 : 255)) >>
         8;
}

uint32_t DrawExtentEstimator::EstimateMaxY(bool try_to_estimate_vertex_max_y,
                                           const Shader& vertex_shader) {
  SCOPE_profile_cpu_f("gpu");

  const RegisterFile& regs = register_file_;

  auto pa_sc_window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
  int32_t window_y_offset = pa_sc_window_offset.window_y_offset;

  // Scissor.
  auto pa_sc_window_scissor_br = regs.Get<reg::PA_SC_WINDOW_SCISSOR_BR>();
  int32_t scissor_bottom = int32_t(pa_sc_window_scissor_br.br_y);
  bool scissor_window_offset = !regs.Get<reg::PA_SC_WINDOW_SCISSOR_TL>().window_offset_disable;
  if (scissor_window_offset) {
    scissor_bottom += window_y_offset;
  }
  auto pa_sc_screen_scissor_br = regs.Get<reg::PA_SC_SCREEN_SCISSOR_BR>();
  scissor_bottom = std::min(scissor_bottom, int32_t(pa_sc_screen_scissor_br.br_y));
  uint32_t max_y = uint32_t(std::max(scissor_bottom, int32_t(0)));

  if (regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable) {
    // Actual extent from the vertices.
    if (try_to_estimate_vertex_max_y && REXCVAR_GET(execute_unclipped_draw_vs_on_cpu)) {
      bool estimate_vertex_max_y;
      if (REXCVAR_GET(execute_unclipped_draw_vs_on_cpu_with_scissor)) {
        estimate_vertex_max_y = true;
      } else {
        estimate_vertex_max_y = false;
        if (scissor_bottom >= xenos::kTexture2DCubeMaxWidthHeight) {
          // Handle just the usual special 8192x8192 case in Direct3D 9 - 8192
          // may be a normal render target height (80x8192 is well within the
          // EDRAM size, for instance), no need to process the vertices on the
          // CPU in this case.
          int32_t scissor_right = int32_t(pa_sc_window_scissor_br.br_x);
          if (scissor_window_offset) {
            scissor_right += pa_sc_window_offset.window_x_offset;
          }
          scissor_right = std::min(scissor_right, int32_t(pa_sc_screen_scissor_br.br_x));
          if (scissor_right >= xenos::kTexture2DCubeMaxWidthHeight) {
            estimate_vertex_max_y = true;
          }
        }
      }
      if (estimate_vertex_max_y) {
        max_y = std::min(max_y, EstimateVertexMaxY(vertex_shader));
      }
    }
  } else {
    // Viewport. Though the Xenos itself doesn't have an implicit viewport
    // scissor (it's set by Direct3D 9 when a viewport is used), on hosts, it
    // usually exists and can't be disabled.
    auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
    float viewport_bottom = 0.0f;
    // First calculate all the integer.0 or integer.5 offsetting exactly at full
    // precision.
    if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().vtx_window_offset_enable) {
      viewport_bottom += float(window_y_offset);
    }
    if (regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero) {
      viewport_bottom += 0.5f;
    }
    // Then apply the floating-point viewport offset.
    if (pa_cl_vte_cntl.vport_y_offset_ena) {
      viewport_bottom += regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET);
    }
    viewport_bottom += pa_cl_vte_cntl.vport_y_scale_ena
                           ? std::abs(regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE))
                           : 1.0f;
    // Using floor, or, rather, truncation (because maxing with zero anyway)
    // similar to how viewport scissoring behaves on real AMD, Intel and Nvidia
    // GPUs on Direct3D 12 (but not WARP), also like in
    // draw_util::GetHostViewportInfo.
    // max(0.0f, viewport_bottom) to drop NaN and < 0 - max picks the first
    // argument in the !(a < b) case (always for NaN), min as float (max_y is
    // well below 2^24) to safely drop very large values.
    max_y = uint32_t(std::min(float(max_y), std::max(0.0f, viewport_bottom)));
  }

  return max_y;
}

}  // namespace rex::graphics
