"""Check the FidelityFX Vulkan scratch-alignment patch against synthetic sources."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_FILE = REPO_ROOT / "cmake" / "rexglue_fidelityfx.cmake"

_ARRAY_LINE = (
    "    uint32_t array{n}ArraySize = "
    "FFX_ALIGN_UP(maxContexts * {n}, sizeof(uint32_t));\n"
)

_DRIVER = """\
set(REXGLUE_ENABLE_FIDELITYFX OFF)
include("{patch_file}")
file(READ "${{INPUT}}" text)
_rexglue_patch_ffx_vk_scratch_arrays(text)
file(WRITE "${{OUTPUT}}" "${{text}}")
"""


def _source(count):
    return "".join(_ARRAY_LINE.format(n=n) for n in range(count))


def _run_patch(tmp_path, source):
    source_path = tmp_path / "ffx_vk.cpp"
    output_path = tmp_path / "ffx_vk.out.cpp"
    driver_path = tmp_path / "driver.cmake"
    source_path.write_text(source, encoding="utf-8")
    driver_path.write_text(_DRIVER.format(patch_file=PATCH_FILE.as_posix()), encoding="utf-8")
    return subprocess.run(
        [
            "cmake",
            f"-DINPUT={source_path}",
            f"-DOUTPUT={output_path}",
            "-P",
            str(driver_path),
        ],
        capture_output=True,
        text=True,
    )


pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake is not installed")


def test_aligns_every_pinned_array_declaration(tmp_path):
    result = _run_patch(tmp_path, _source(12))

    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / "ffx_vk.out.cpp").read_text(encoding="utf-8")
    assert output.count(", 16);") == 12
    assert ", sizeof(uint32_t));" not in output


def test_leaves_unrelated_alignments_alone(tmp_path):
    unrelated = "    uint32_t size = FFX_ALIGN_UP(width, sizeof(uint32_t));\n"
    result = _run_patch(tmp_path, unrelated + _source(12))

    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / "ffx_vk.out.cpp").read_text(encoding="utf-8")
    assert unrelated in output
    assert output.count(", 16);") == 12


def test_rejects_a_pin_with_fewer_array_declarations(tmp_path):
    result = _run_patch(tmp_path, _source(11))

    assert result.returncode != 0
    assert "expected 12 scratch arrays" in result.stderr
