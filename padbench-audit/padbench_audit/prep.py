"""Build blind case directories from PADBench adapter folders.

The case directory receives adapter.safetensors and shapes.json only.
Original adapter_config.json, tokenizers, READMEs, and CSVs are not copied.
Labels are read from the folder name (label0 = clean, label1 = backdoor) and
written only under the private directory, which the sandbox never mounts.
"""
from __future__ import annotations

import random
import re
import secrets
import shutil
from pathlib import Path

from padbench_audit.common import (
    assert_mount_path_clean,
    assert_outside,
    leak_hits,
    sha256_file,
    write_json,
)
from padbench_audit.names import NameError, base_layer_index, is_dropped_base_buffer, rename_map
from padbench_audit.safetensors_io import (
    TensorRef,
    header_problems,
    ranges_equal,
    read_header,
    read_safetensors,
    write_refs,
    write_safetensors,
)

_LABEL = re.compile(r"^label([01])(?:_.*)?$")


def label_from_folder(name: str) -> str:
    match = _LABEL.match(name)
    if not match:
        raise SystemExit(
            f"folder {name!r} is not label0* or label1*; ground truth is the folder name and nothing else"
        )
    return "clean" if match.group(1) == "0" else "backdoor"


def find_single_safetensors(folder: Path) -> Path:
    if not folder.is_dir():
        raise SystemExit(f"missing adapter folder {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and not p.is_symlink() and p.suffix == ".safetensors"]
    if len(files) != 1:
        nested = [
            p
            for p in folder.rglob("*.safetensors")
            if p.is_file() and not p.is_symlink() and "original" not in p.name
        ]
        files = nested if len(files) == 0 else files
    if len(files) != 1:
        raise SystemExit(f"{folder} contains {len(files)} safetensors files; expected exactly one")
    return files[0]


def _shapes(published: list[tuple[str, str, list[int]]]) -> dict:
    rows = [{"dtype": dtype, "name": name, "shape": shape} for name, dtype, shape in published]
    rows.sort(key=lambda row: row["name"])
    return {"tensors": rows}


def anonymize_file(
    src: Path,
    dest: Path,
    *,
    kind: str,
    neutral: bool,
) -> dict:
    header, metadata, blobs = read_safetensors(src)
    try:
        mapping = rename_map(list(header), kind=kind, neutral=neutral)
    except NameError as exc:
        raise SystemExit(
            f"{src} has a tensor name this harness will not publish ({exc}). "
            "The original name was not written into the case directory."
        ) from exc
    ordered = sorted(mapping.items(), key=lambda item: item[1])
    tensors = []
    for original, published in ordered:
        info = header[original]
        tensors.append((published, info["dtype"], list(info["shape"]), blobs[original]))
    write_safetensors(dest, tensors)
    problems = header_problems(dest)
    if problems:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"{dest} failed the anonymous-header check: {problems}")
    written, written_meta, written_blobs = read_safetensors(dest)
    if written_meta:
        raise SystemExit(f"{dest} still has header metadata")
    for original, published in mapping.items():
        if written_blobs[published] != blobs[original]:
            raise SystemExit(f"tensor bytes changed for {published}")
        if list(written[published]["shape"]) != list(header[original]["shape"]):
            raise SystemExit(f"shape changed for {published}")
        if written[published]["dtype"] != header[original]["dtype"]:
            raise SystemExit(f"dtype changed for {published}")
    hits = leak_hits(json_names(written))
    if hits:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"{dest} header names contain {hits}")
    return {
        "tensor_map": {published: original for original, published in mapping.items()},
        "metadata_stripped": bool(metadata),
        "source_sha256": sha256_file(src),
        "output_sha256": sha256_file(dest),
        "nbytes": sum(len(raw) for raw in blobs.values()),
    }


def json_names(header: dict) -> str:
    return "\n".join(header)


def prepare_case(
    *,
    source_dir: Path,
    case_dir: Path,
    neutral: bool,
    expose_tokenizer: bool,
) -> dict:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    assert_mount_path_clean(case_dir, what="case")
    try:
        return _fill_case(source_dir, case_dir, neutral=neutral, expose_tokenizer=expose_tokenizer)
    except BaseException:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise


def _fill_case(source_dir: Path, case_dir: Path, *, neutral: bool, expose_tokenizer: bool) -> dict:
    src = find_single_safetensors(source_dir)
    info = anonymize_file(src, case_dir / "adapter.safetensors", kind="adapter", neutral=neutral)
    header, _, _ = read_safetensors(case_dir / "adapter.safetensors")
    published = [(name, item["dtype"], list(item["shape"])) for name, item in header.items()]
    shapes = _shapes(published)
    write_json(case_dir / "shapes.json", shapes)
    if expose_tokenizer:
        _copy_tokenizer(source_dir, case_dir / "tokenizer")
    leftovers = sorted(p.name for p in case_dir.iterdir())
    allowed = {"adapter.safetensors", "shapes.json"}
    if expose_tokenizer:
        allowed.add("tokenizer")
    if set(leftovers) - allowed:
        raise SystemExit(f"case directory has unexpected files: {leftovers}")
    info["shapes_sha256"] = sha256_file(case_dir / "shapes.json")
    info["neutral_names"] = neutral
    info["expose_tokenizer"] = expose_tokenizer
    info["source_dir_name"] = source_dir.name
    info["label"] = label_from_folder(source_dir.name)
    return info


def _copy_tokenizer(source_dir: Path, dest: Path) -> None:
    found = list(source_dir.rglob("tokenizer.json"))
    found = [p for p in found if p.is_file() and not p.is_symlink()]
    if len(found) != 1:
        raise SystemExit(f"expose_tokenizer is set but found {len(found)} tokenizer.json files")
    text = found[0].read_text(encoding="utf-8", errors="replace")
    hits = leak_hits(text[:200_000])
    if hits:
        raise SystemExit(
            f"tokenizer.json contains {hits}. Refusing to mount it. "
            "The default experiment leaves the vocabulary off for this reason."
        )
    dest.mkdir()
    (dest / "tokenizer.json").write_text(text, encoding="utf-8")


def prepare_base(*, source_dir: Path, dest_dir: Path, neutral: bool) -> dict:
    """Repack the base network into per-layer files.

    Each source shard is opened for its header, then each tensor is copied once
    in chunks. The original shard boundaries are not preserved: those byte sizes
    match public checkpoints. Rotary ``inv_freq`` buffers are dropped.
    """
    if not source_dir.is_dir():
        raise SystemExit(f"missing base model directory {source_dir}")
    files = sorted(
        p for p in source_dir.iterdir() if p.is_file() and not p.is_symlink() and p.suffix == ".safetensors"
    )
    if not files:
        raise SystemExit(f"{source_dir} has no *.safetensors files")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    assert_mount_path_clean(dest_dir, what="base")
    located: dict[str, TensorRef] = {}
    metadata_any = False
    for src in files:
        header, metadata, data_start = read_header(src)
        metadata_any = metadata_any or bool(metadata)
        for name, info in header.items():
            if name in located:
                raise SystemExit(f"duplicate tensor name across base shards ({name})")
            start, end = info["data_offsets"]
            located[name] = TensorRef(name, info["dtype"], list(info["shape"]), src, data_start, start, end)
    dropped = sorted(name for name in located if is_dropped_base_buffer(name))
    kept = [name for name in located if name not in set(dropped)]
    try:
        mapping = rename_map(kept, kind="base", neutral=neutral)
    except NameError as exc:
        raise SystemExit(f"base tensor name will not be published ({exc})") from exc
    width = 2
    if any((base_layer_index(name) or 0) >= 100 for name in kept):
        width = 3
    groups: dict[str, list[TensorRef]] = {}
    group_of: dict[str, str] = {}
    for original, published in mapping.items():
        layer = base_layer_index(original)
        filename = f"group_{layer:0{width}d}.safetensors" if layer is not None else "group_other.safetensors"
        ref = located[original]
        groups.setdefault(filename, []).append(
            TensorRef(published, ref.dtype, ref.shape, ref.path, ref.data_start, ref.start, ref.end)
        )
        group_of[published] = filename
    published_rows = []
    for filename, refs in groups.items():
        refs.sort(key=lambda item: item.name)
        dest = dest_dir / filename
        write_refs(dest, refs)
        problems = header_problems(dest)
        if problems:
            raise SystemExit(f"{filename} failed the anonymous-header check: {problems}")
        written, _meta, data_start = read_header(dest)
        for ref in refs:
            info = written[ref.name]
            start, end = info["data_offsets"]
            if not ranges_equal(ref, dest, data_start, start, end):
                raise SystemExit(f"base tensor bytes changed for {ref.name}")
            if list(info["shape"]) != list(ref.shape) or info["dtype"] != ref.dtype:
                raise SystemExit(f"base tensor metadata changed for {ref.name}")
            published_rows.append((ref.name, ref.dtype, list(ref.shape)))
    hits = leak_hits("\n".join(name for name, _dtype, _shape in published_rows))
    if hits:
        raise SystemExit(f"base header names contain {hits}")
    write_json(dest_dir / "shapes.json", _shapes(published_rows))
    return {
        "kind": "base",
        "neutral_names": neutral,
        "metadata_stripped": metadata_any,
        "dropped_buffers": dropped,
        "group_files": group_of,
        "tensor_map": {published: original for original, published in mapping.items()},
        "source_files": {src.name: sha256_file(src) for src in files},
        "shapes_sha256": sha256_file(dest_dir / "shapes.json"),
        "note": (
            "Output files are one per layer plus group_other for embeddings and the output "
            "projection. Original shard boundaries are not kept. Width, depth, and vocabulary "
            "still follow from the tensor shapes."
        ),
    }


_CHECKPOINT_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.model",
}
_CHECKPOINT_DENY = {"edit.npz", "manifest.json", "reconstruct_stats.json"}


def _relative_checkpoint(name: str) -> str:
    path = Path(name)
    if not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"checkpoint path must be a relative directory: {name!r}")
    return name


def pair_kind(config: dict) -> str:
    kind = config.get("kind") or "lora"
    if kind not in {"lora", "full_model"}:
        raise SystemExit("pair kind must be lora or full_model")
    return kind


def assign_full_slots(clean: str, edited: str, rng: random.Random) -> tuple[str, str, str]:
    """Return (dir_for_X, dir_for_Y, backdoored slot). Seed stays in the private map."""
    if not clean or not edited or clean == edited:
        raise SystemExit("a full-model pair needs two different checkpoint directories")
    order = [clean, edited]
    if rng.randrange(2):
        order.reverse()
    backdoored = "X" if order[0] == edited else "Y"
    return order[0], order[1], backdoored


def copy_private_checkpoint(source_dir: Path, dest_dir: Path) -> list[str]:
    """Copy weights plus tokenizer and config. Leave answer files behind.

    edit.npz, manifest.json, and reconstruct_stats.json describe the edit.
    They are not copied. The destination is mounted only in the probe worker.
    """
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    copied = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in _CHECKPOINT_DENY:
            continue
        allowed = path.name in _CHECKPOINT_FILES or path.suffix == ".safetensors"
        if not allowed:
            continue
        target = dest_dir / path.name
        shutil.copyfile(path, target)
        copied.append(path.name)
    if not any(name.endswith(".safetensors") for name in copied):
        raise SystemExit(f"{source_dir} has no checkpoint weights to copy")
    if "config.json" not in copied:
        raise SystemExit(f"{source_dir} has no config.json for the probe worker")
    return copied


def prepare_full_model(*, source_dir: Path, dest_dir: Path, neutral: bool) -> dict:
    """Anonymize one full checkpoint into per-layer files.

    Tensor bytes are copied unchanged, including BF16. The numpy prelude
    decodes BF16 to float32 on load. A tensor that is identical in two
    checkpoints stays byte-identical here, under the same published name.
    Config and tokenizer files are not written into the blind tree.
    """
    from padbench_audit.names import is_dropped_full_buffer

    if not source_dir.is_dir():
        raise SystemExit(f"missing checkpoint directory {source_dir}")
    files = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_file() and not p.is_symlink() and p.suffix == ".safetensors" and p.name not in _CHECKPOINT_DENY
    )
    if not files:
        raise SystemExit(f"{source_dir} has no *.safetensors files")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    assert_mount_path_clean(dest_dir, what="full model")
    located: dict[str, TensorRef] = {}
    for src in files:
        header, _metadata, data_start = read_header(src)
        for name, info in header.items():
            if name in located:
                raise SystemExit(f"duplicate tensor name in {source_dir} ({name})")
            start, end = info["data_offsets"]
            located[name] = TensorRef(name, info["dtype"], list(info["shape"]), src, data_start, start, end)
    dropped = sorted(name for name in located if is_dropped_full_buffer(name))
    kept = [name for name in located if name not in set(dropped)]
    try:
        mapping = rename_map(kept, kind="full", neutral=neutral)
    except NameError as exc:
        raise SystemExit(f"checkpoint tensor name will not be published ({exc})") from exc
    groups: dict[str, list[TensorRef]] = {}
    for original, published in mapping.items():
        layer = base_layer_index(original)
        if original.startswith("model.visual."):
            filename = f"vis_{layer:02d}.safetensors" if layer is not None else "vis_other.safetensors"
        elif layer is not None:
            filename = f"group_{layer:02d}.safetensors"
        else:
            filename = "group_other.safetensors"
        ref = located[original]
        groups.setdefault(filename, []).append(
            TensorRef(published, ref.dtype, ref.shape, ref.path, ref.data_start, ref.start, ref.end)
        )
    published_rows = []
    for filename, refs in groups.items():
        refs.sort(key=lambda item: item.name)
        dest = dest_dir / filename
        write_refs(dest, refs)
        problems = header_problems(dest)
        if problems:
            raise SystemExit(f"{filename} failed the anonymous-header check: {problems}")
        written, _meta, data_start = read_header(dest)
        for ref in refs:
            info = written[ref.name]
            start, end = info["data_offsets"]
            if not ranges_equal(ref, dest, data_start, start, end):
                raise SystemExit(f"full-model tensor bytes changed for {ref.name}")
            if list(info["shape"]) != list(ref.shape) or info["dtype"] != ref.dtype:
                raise SystemExit(f"full-model tensor metadata changed for {ref.name}")
            published_rows.append((ref.name, ref.dtype, list(ref.shape)))
    hits = leak_hits("\n".join(name for name, _dtype, _shape in published_rows))
    if hits:
        raise SystemExit(f"full-model header names contain {hits}")
    write_json(dest_dir / "shapes.json", _shapes(published_rows))
    leftovers = [p.name for p in dest_dir.iterdir() if p.suffix not in {".safetensors", ".json"}]
    if leftovers:
        raise SystemExit(f"full-model directory has unexpected files: {leftovers}")
    return {
        "kind": "full_model",
        "neutral_names": neutral,
        "dropped_buffers": dropped,
        "tensor_map": {published: original for original, published in mapping.items()},
        "source_files": {src.name: sha256_file(src) for src in files},
        "shapes_sha256": sha256_file(dest_dir / "shapes.json"),
        "note": "BF16 tensor bytes are unchanged. The prelude decodes them to float32 on load.",
    }


def assign_pair_slots(folders: list[str], rng: random.Random) -> tuple[str, str, str]:
    """Return (folder_for_X, folder_for_Y, backdoored slot).

    The caller stores the random choice only in the private map. Exactly one
    folder must be clean and one backdoored. The check uses the folder name.
    """
    if len(folders) != 2:
        raise SystemExit("a pair needs two folders")
    labels = [label_from_folder(name) for name in folders]
    if sorted(labels) != ["backdoor", "clean"]:
        raise SystemExit(f"a pair must contain one clean folder and one backdoor folder, got {folders}")
    order = list(folders)
    if rng.randrange(2):
        order.reverse()
    backdoored = "X" if label_from_folder(order[0]) == "backdoor" else "Y"
    return order[0], order[1], backdoored


def case_count(config: dict) -> int:
    if config.get("condition") == "pair-probe":
        return len(config.get("pairs") or [])
    return len(config.get("cases") or [])


def case_id_for(index: int) -> str:
    return f"case_{index:03d}"


def shuffled_case_ids(count: int, seed: int) -> list[str]:
    """Permute case ids. The seed is stored only in the private map."""
    ids = [case_id_for(index) for index in range(1, count + 1)]
    random.Random(seed).shuffle(ids)
    return ids


def prepare_experiment(
    *,
    config: dict,
    source_root: Path,
    out_dir: Path,
    private_dir: Path,
    base_src: Path | None,
    allow_tokenizer: bool = False,
    case_order_seed: int | None = None,
) -> dict:
    """Write blind cases and the private label map. Returns the private record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    assert_mount_path_clean(out_dir, what="blind output")
    assert_outside(private_dir, [out_dir], what="private dir")
    cases_root = out_dir / "cases"
    cases_root.mkdir(exist_ok=True)
    neutral = bool(config.get("neutral_names", True))
    expose = bool(config.get("expose_tokenizer", False))
    if expose and not allow_tokenizer:
        raise SystemExit(
            "expose_tokenizer is true. The default experiment keeps it false because a vocabulary "
            "can name the task. Pass --allow-tokenizer only after that design choice is settled."
        )
    seed = secrets.randbits(63) if case_order_seed is None else int(case_order_seed)
    count = case_count(config)
    if count < 1:
        raise SystemExit("config has no cases")
    ids = shuffled_case_ids(count, seed)
    records = {}
    if config["condition"] == "pair-probe" and pair_kind(config) == "full_model":
        slot_rng = random.Random(seed)
        for item, cid in zip(config["pairs"], ids):
            clean = _relative_checkpoint(item["clean"])
            edited = _relative_checkpoint(item["edited"])
            dir_x, dir_y, backdoored = assign_full_slots(clean, edited, slot_rng)
            case_dir = cases_root / cid
            info_x = prepare_full_model(source_dir=source_root / dir_x, dest_dir=case_dir / "model_X", neutral=neutral)
            info_y = prepare_full_model(source_dir=source_root / dir_y, dest_dir=case_dir / "model_Y", neutral=neutral)
            rel_x = f"checkpoints/{cid}/X"
            rel_y = f"checkpoints/{cid}/Y"
            info_x["copied_files"] = copy_private_checkpoint(source_root / dir_x, private_dir / rel_x)
            info_y["copied_files"] = copy_private_checkpoint(source_root / dir_y, private_dir / rel_y)
            info_x["checkpoint"] = rel_x
            info_y["checkpoint"] = rel_y
            info_x["source_dir_name"] = dir_x
            info_y["source_dir_name"] = dir_y
            info_x["label"] = "backdoor" if dir_x == edited else "clean"
            info_y["label"] = "backdoor" if dir_y == edited else "clean"
            records[cid] = {
                "case_id": cid,
                "kind": "full_model",
                "group": config.get("group"),
                "backdoored": backdoored,
                "X": info_x,
                "Y": info_y,
            }
    elif config["condition"] == "pair-probe":
        slot_rng = random.Random(seed)
        for item, cid in zip(config["pairs"], ids):
            folders = item if isinstance(item, list) else item.get("folders")
            folder_x, folder_y, backdoored = assign_pair_slots(list(folders), slot_rng)
            case_dir = cases_root / cid
            info_x = prepare_case(
                source_dir=source_root / folder_x,
                case_dir=case_dir / "adapter_X",
                neutral=neutral,
                expose_tokenizer=expose,
            )
            info_y = prepare_case(
                source_dir=source_root / folder_y,
                case_dir=case_dir / "adapter_Y",
                neutral=neutral,
                expose_tokenizer=expose,
            )
            records[cid] = {
                "case_id": cid,
                "kind": "lora",
                "group": config.get("group"),
                "backdoored": backdoored,
                "X": info_x,
                "Y": info_y,
            }
    else:
        for item, cid in zip(config["cases"], ids):
            folder = item["folder"] if isinstance(item, dict) else str(item)
            source = source_root / folder
            info = prepare_case(
                source_dir=source,
                case_dir=cases_root / cid,
                neutral=neutral,
                expose_tokenizer=expose,
            )
            info["case_id"] = cid
            info["group"] = config.get("group")
            records[cid] = info
    private = {
        "experiment": config["experiment"],
        "condition": config["condition"],
        "group": config.get("group"),
        "neutral_names": neutral,
        "expose_tokenizer": expose,
        "case_order_seed": seed,
        "cases": records,
    }
    if config["condition"] == "pair-probe" and pair_kind(config) == "full_model":
        if base_src is not None:
            raise SystemExit("a full-model pair does not take --base-src; each checkpoint is a complete model")
    elif config["condition"] in {"base-adapter", "pair-probe"}:
        if base_src is None:
            raise SystemExit(f"condition {config['condition']} needs --base-src (the unanonymized safetensors directory)")
        private["base"] = prepare_base(source_dir=base_src, dest_dir=out_dir / "base", neutral=neutral)
    elif base_src is not None:
        raise SystemExit(
            "this condition does not copy a base tree into the blind directory. "
            "For forward-probe, pass --base-src to run; it is mounted only inside the probe worker."
        )
    write_json(private_dir / "cases.json", private)
    canary = private_dir / "DO_NOT_MOUNT"
    canary.write_text("private ground truth\n", encoding="utf-8")
    return private
