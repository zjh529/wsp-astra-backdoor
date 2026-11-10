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

from padbench_audit.common import LEAK_SUBSTRINGS, leak_hits

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


def _as_language_path(original: str) -> str:
    """Drop a language-model prefix so the shared layer rules still match."""
    prefix = "model.language_model."
    if original.startswith(prefix):
        return "model." + original[len(prefix) :]
    return original


_FULL_EXTRA: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$"), "layer_{0}.norm_q"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$"), "layer_{0}.norm_k"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$"), "layer_{0}.linear_qkv"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$"), "layer_{0}.linear_z"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight$"), "layer_{0}.linear_b"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight$"), "layer_{0}.linear_a"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.A_log$"), "layer_{0}.linear_a_log"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.dt_bias$"), "layer_{0}.linear_dt"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$"), "layer_{0}.linear_conv"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.norm\.weight$"), "layer_{0}.linear_norm"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$"), "layer_{0}.linear_out"),
]


def _role_token(rest: str) -> str:
    text = rest.replace(".weight", "").replace(".bias", "_bias").replace(".", "_").lower()
    for token in sorted(LEAK_SUBSTRINGS, key=len, reverse=True):
        text = text.replace(token, "")
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "tensor"


def _visual_name(original: str) -> str:
    rest = original[len("model.visual.") :]
    block = re.match(r"blocks\.(\d+)\.(.+)", rest)
    if block:
        return f"vis_{block.group(1)}.{_role_token(block.group(2))}"
    return "vis_" + _role_token(rest)


def _fallback_structural(original: str) -> str:
    layer = base_layer_index(original)
    tail = re.sub(r"^.*layers\.\d+\.", "", original) if layer is not None else original
    role = _role_token(tail)
    published = f"layer_{layer}.{role}" if layer is not None else role
    hits = leak_hits(published)
    if hits:
        raise NameError(f"{original} still contains {hits} after renaming")
    return published


def neutral_full_name(original: str) -> str:
    """Layer index and role, including hybrid linear-attention tensors.

    The published name is a function of the original name only, so a tensor
    that is identical in two checkpoints keeps the same name and the same bytes.
    """
    path = _as_language_path(original)
    try:
        return neutral_base_name(path)
    except NameError:
        pass
    for pattern, template in _FULL_EXTRA:
        match = pattern.match(path)
        if match:
            return template.format(*match.groups())
    if original.startswith("model.visual."):
        return _visual_name(original)
    return _fallback_structural(original)


def is_dropped_full_buffer(original: str) -> bool:
    path = _as_language_path(original)
    return path.endswith("rotary_emb.inv_freq") or is_dropped_base_buffer(path)


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
            if kind == "adapter":
                published = neutral_adapter_name(original)
            elif kind == "full":
                published = neutral_full_name(original)
            else:
                published = neutral_base_name(original)
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
