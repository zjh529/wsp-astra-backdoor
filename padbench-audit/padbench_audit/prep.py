"""Build blind case directories from PADBench adapter folders.

The case directory receives adapter.safetensors and shapes.json only.
Original adapter_config.json, tokenizers, READMEs, and CSVs are not copied.
Labels are read from the folder name (label0 = clean, label1 = backdoor) and
written only under the private directory, which the sandbox never mounts.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from padbench_audit.common import (
    assert_mount_path_clean,
    assert_outside,
    leak_hits,
    sha256_file,
    write_json,
)
from padbench_audit.names import NameError, rename_map
from padbench_audit.safetensors_io import header_problems, read_safetensors, write_safetensors

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
    combined_header: dict = {}
    combined_blobs: dict[str, bytes] = {}
    file_map = {}
    metadata_any = False
    per_file = []
    for index, src in enumerate(files):
        header, metadata, blobs = read_safetensors(src)
        metadata_any = metadata_any or bool(metadata)
        overlap = set(header) & set(combined_header)
        if overlap:
            raise SystemExit(f"duplicate tensor names across base shards ({len(overlap)} names)")
        combined_header.update(header)
        combined_blobs.update(blobs)
        file_map[src.name] = f"shard_{index:03d}.safetensors"
        per_file.append((src, header, blobs, file_map[src.name]))
    try:
        mapping = rename_map(list(combined_header), kind="base", neutral=neutral)
    except NameError as exc:
        raise SystemExit(f"base tensor name will not be published ({exc})") from exc
    by_shard: dict[str, list] = {name: [] for name in file_map.values()}
    original_file_of = {}
    for src, header, blobs, shard_name in per_file:
        for original in header:
            original_file_of[original] = shard_name
    for original, published in mapping.items():
        info = combined_header[original]
        by_shard[original_file_of[original]].append(
            (published, info["dtype"], list(info["shape"]), combined_blobs[original])
        )
    for shard_name, tensors in by_shard.items():
        tensors.sort(key=lambda item: item[0])
        write_safetensors(dest_dir / shard_name, tensors)
        problems = header_problems(dest_dir / shard_name)
        if problems:
            raise SystemExit(f"{shard_name} failed the anonymous-header check: {problems}")
    published_rows = []
    for original, published in mapping.items():
        info = combined_header[original]
        if combined_blobs[original] != _reload_blob(dest_dir, published):
            raise SystemExit(f"base tensor bytes changed for {published}")
        published_rows.append((published, info["dtype"], list(info["shape"])))
    write_json(dest_dir / "shapes.json", _shapes(published_rows))
    return {
        "kind": "base",
        "neutral_names": neutral,
        "metadata_stripped": metadata_any,
        "shard_map": file_map,
        "tensor_map": {published: original for original, published in mapping.items()},
        "source_files": {src.name: sha256_file(src) for src in files},
        "shapes_sha256": sha256_file(dest_dir / "shapes.json"),
    }


def _reload_blob(dest_dir: Path, name: str) -> bytes:
    for path in sorted(dest_dir.glob("*.safetensors")):
        header, _, blobs = read_safetensors(path)
        if name in blobs:
            return blobs[name]
    raise SystemExit(f"rewritten base is missing {name}")


def case_id_for(index: int) -> str:
    return f"case_{index:03d}"


def prepare_experiment(
    *,
    config: dict,
    source_root: Path,
    out_dir: Path,
    private_dir: Path,
    base_src: Path | None,
    allow_tokenizer: bool = False,
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
    records = {}
    for index, item in enumerate(config["cases"], start=1):
        folder = item["folder"] if isinstance(item, dict) else str(item)
        source = source_root / folder
        cid = case_id_for(index)
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
        "cases": records,
    }
    if config["condition"] == "base-adapter":
        if base_src is None:
            raise SystemExit("condition base-adapter needs --base-src (the unanonymized safetensors directory)")
        private["base"] = prepare_base(source_dir=base_src, dest_dir=out_dir / "base", neutral=neutral)
    elif base_src is not None:
        raise SystemExit("adapter-only does not take --base-src; use the base-adapter config")
    write_json(private_dir / "cases.json", private)
    canary = private_dir / "DO_NOT_MOUNT"
    canary.write_text("private ground truth\n", encoding="utf-8")
    return private
