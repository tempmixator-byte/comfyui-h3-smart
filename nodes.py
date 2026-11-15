"""H3 Smart: ComfyUI custom nodes for MiniMax H3 (V3 schema).

Node implementations live in nodes_*.py; this module registers them with ComfyUI.
  * every node_id is prefixed with "H3Smart_" so it cannot collide with other packs
  * all node methods are classmethods; nodes hold no state
  * heavy imports (torch, comfy.*) happen inside methods
"""
from comfy_api.latest import ComfyExtension, io

from .nodes_cache import (
    H3Smart_CachedConditioning,
    H3Smart_CachedReferenceConditioning,
    H3Smart_CondCacheManager,
)
from .nodes_plan import (
    H3Smart_BudgetPlanner,
    H3Smart_SamplingSetup,
    H3Smart_ShotPromptBuilder,
)
from .nodes_system import H3Smart_MemoryBarrier, H3Smart_SystemAdvisor

NODES = [
    H3Smart_SystemAdvisor,
    H3Smart_BudgetPlanner,
    H3Smart_ShotPromptBuilder,
    H3Smart_CachedConditioning,
    H3Smart_CachedReferenceConditioning,
    H3Smart_SamplingSetup,
    H3Smart_MemoryBarrier,
    H3Smart_CondCacheManager,
]


class H3SmartExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> H3SmartExtension:
    return H3SmartExtension()
