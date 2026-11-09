"""Rename adapter and base-model tensor names.

Default names keep the layer index, the projection role (q/k/v/o, mlp gate/up/down),
and the A/B factor. They drop the ``base_model.model...`` prefix and any adapter
name, which can carry a label. Module-role tokens such as ``proj_q`` are a
deliberate borderline: the task asked to keep structure that the shapes alone
do not show (which factor, which layer) while hiding the base-model identity.

``neutral=False`` keeps the original names. Callers still strip header metadata.
"""
from __future__ import annotations

import re

from padbench_audit.common import leak_hits

_ADAPTER = re.compile(
    r"^(?:base_model\.model\.)*"
    r"(?:model\.)*"
    r"layers\.(?P<layer>\d+)\."
    r"(?P<block>self_attn|mlp)\."
    r"(?P<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_(?P<factor>A|B)"
    r"(?:\.[A-Za-z0-9_]+)?"
    r"\.weight$"
)

_PROJ = {
    "q_proj": "proj_q",
    "k_proj": "proj_k",
    "v_proj": "proj_v",
    "o_proj": "proj_o",
    "gate_proj": "mlp_gate",
    "up_proj": "mlp_up",
    "down_proj": "mlp_down",
}

# Rotary inv_freq is a derived buffer, not a learned weight. It is dropped
# rather than renamed, so the mounted tree does not carry it.
_DROP_BASE = re.compile(r"^model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq$")

_BASE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:model\.)?embed_tokens\.weight$"), "embed"),
    (re.compile(r"^lm_head\.weight$"), "output"),
    (re.compile(r"^model\.norm\.weight$"), "final_norm"),
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"), "layer_{0}.norm_in"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"), "layer_{0}.norm_post"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$"), "layer_{0}.proj_{1}"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$"), "layer_{0}.mlp_{1}"),
]


def is_dropped_base_buffer(original: str) -> bool:
    return _DROP_BASE.match(original) is not None


def base_layer_index(original: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", original)
    if not match:
        return None
    return int(match.group(1))


class NameError(ValueError):
    pass


def neutral_adapter_name(original: str) -> str:
    match = _ADAPTER.match(original)
    if not match:
        raise NameError(original)
    role = _PROJ[match.group("proj")]
    return f"layer_{match.group('layer')}.{role}.{match.group('factor')}"


def neutral_base_name(original: str) -> str:
    for pattern, template in _BASE_RULES:
        match = pattern.match(original)
        if match:
            return template.format(*match.groups())
    raise NameError(original)


def rename_map(originals: list[str], *, kind: str, neutral: bool) -> dict[str, str]:
    """Map original name -> published name. Raises NameError on the first unknown name."""
    mapping: dict[str, str] = {}
    used: dict[str, str] = {}
    for original in originals:
        if neutral:
            published = neutral_adapter_name(original) if kind == "adapter" else neutral_base_name(original)
        else:
            published = original
        hits = leak_hits(published)
        if hits:
            raise NameError(f"{original} publishes as {published!r}, which contains {hits}")
        if published in used and used[published] != original:
            raise NameError(f"{original} and {used[published]} both publish as {published}")
        used[published] = original
        mapping[original] = published
    return mapping
