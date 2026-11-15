"""Test setup.

Two modes:
  * REAL: ComfyUI source is importable (set COMFYUI_SRC). Native nodes are registered on CPU, so the pack is
    tested against the genuine comfy_api and the genuine native H3 nodes.
  * STUB: no ComfyUI available. Only the pure-logic tests (core / prompting) are meaningful.
"""
import asyncio
import importlib.util
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

COMFY_SRC = os.environ.get("COMFYUI_SRC", "")
REAL = False
if COMFY_SRC and pathlib.Path(COMFY_SRC, "comfy_api").exists():
    sys.path.insert(0, COMFY_SRC)
    sys.argv = ["pytest", "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import nodes as comfy_nodes
        asyncio.run(comfy_nodes.init_extra_nodes(init_custom_nodes=False))
        REAL = True
    except Exception as e:  # pragma: no cover
        print("could not boot real ComfyUI:", e)

if not REAL:
    try:
        import comfy_api.latest  # noqa: F401
    except ImportError:
        import stub_comfy_api
        stub_comfy_api.install()


def pytest_report_header(config):
    return f"H3 Smart tests: {'REAL ComfyUI (CPU)' if REAL else 'stub comfy_api'}"


@pytest.fixture(scope="session")
def real():
    return REAL


@pytest.fixture(scope="session")
def pack():
    """Import the pack by path (its folder name contains a hyphen)."""
    spec = importlib.util.spec_from_file_location(
        "h3_smart_under_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["h3_smart_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_SMART_CACHE_DIR", str(tmp_path / "cond_cache"))
