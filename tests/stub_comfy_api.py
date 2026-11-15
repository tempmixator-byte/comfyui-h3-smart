"""Minimal stand-in for comfy_api.latest so a pack's logic can be tested without ComfyUI.

It records schemas and lets execute() run. It is NOT a faithful ComfyUI: it proves your pack's
structure and Python logic, not behaviour inside the real server. Unknown io.* names are
created on demand so new types do not break imports.
"""
import sys
import types


class _Meta(type):
    def __getattr__(cls, name):            # io.Hidden.unique_id, io.NumberDisplay.slider, ...
        if name.startswith("__"):
            raise AttributeError(name)
        return name


def _make_type(name):
    class T:
        class Input:
            def __init__(self, id, **kw):
                self.id, self.kw = id, kw

        class Output:
            def __init__(self, **kw):
                self.kw = kw
    T.__name__ = name
    return T


class Schema:
    def __init__(self, node_id, **kw):
        self.node_id = node_id
        self.__dict__.update(kw)


class NodeOutput:
    def __init__(self, *args, ui=None, **kw):
        self.args, self.ui = args, ui


class ComfyNode:
    @classmethod
    def define_schema(cls):
        raise NotImplementedError

    @classmethod
    def execute(cls, **kw):
        raise NotImplementedError


class ComfyExtension:
    async def on_load(self):
        pass

    async def get_node_list(self):
        raise NotImplementedError


class _IO(types.ModuleType):
    Schema, NodeOutput, ComfyNode = Schema, NodeOutput, ComfyNode
    Hidden = _Meta("Hidden", (), {})
    NumberDisplay = _Meta("NumberDisplay", (), {})
    ControlAfterGenerate = _Meta("ControlAfterGenerate", (), {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        t = _make_type(name)
        setattr(self, name, t)
        return t

    @staticmethod
    def Custom(type_name):
        return _make_type(type_name)


class _UI(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def factory(*a, **kw):
            return {"ui_helper": name, "args": a, "kw": kw}
        return factory


def install():
    if "comfy_api.latest" in sys.modules:
        return
    pkg = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _IO("comfy_api.latest.io")
    latest.ui = _UI("comfy_api.latest.ui")
    latest.ComfyExtension = ComfyExtension
    latest.ComfyAPI = lambda: types.SimpleNamespace()
    pkg.latest = latest
    sys.modules["comfy_api"] = pkg
    sys.modules["comfy_api.latest"] = latest
