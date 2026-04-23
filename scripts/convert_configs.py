"""
One-shot Lightning CLI -> Hydra config converter.

Walks `cfg/data/*.yaml` and `cfg/model/**/*.yaml`, applies the mechanical
transform (`class_path` -> `_target_`, unwraps `init_args`, fixes broken
`nablafx.system.*` / `nablafx.models.*` paths), and emits the results to
`conf/data/` and `conf/model/` preserving the directory tree.

Trainer YAMLs are NOT converted here — they need structural reorganization
(callback list -> dict, split into logger/callbacks subgroups) that is
hand-authored in `conf/trainer/`.

After writing, the script performs a dry-run validation pass: `OmegaConf.load`
every produced file and walk the tree calling `hydra.utils.get_class` on every
`_target_`. Any unresolvable symbol prints a hard error with the offending
file path.

Usage:
    uv run python scripts/convert_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_ROOT = REPO_ROOT / "cfg"
CONF_ROOT = REPO_ROOT / "conf"

# class_path rewrites: left side is what the old YAML says, right side is
# the current valid import path. Top-level re-exports in nablafx/__init__.py
# let us target `nablafx.BlackBoxSystem` etc. without caring about internal
# module moves.
TARGET_REWRITES: dict[str, str] = {
    "nablafx.system.BlackBoxSystem":          "nablafx.BlackBoxSystem",
    "nablafx.system.BlackBoxSystemWithTBPTT": "nablafx.BlackBoxSystemWithTBPTT",
    "nablafx.system.GreyBoxSystem":           "nablafx.GreyBoxSystem",
    "nablafx.system.GreyBoxSystemWithTBPTT":  "nablafx.GreyBoxSystemWithTBPTT",
    "nablafx.models.BlackBoxModel":           "nablafx.BlackBoxModel",
    "nablafx.models.GreyBoxModel":            "nablafx.GreyBoxModel",
}


def _transform(node: Any) -> Any:
    """Recursively convert `class_path`/`init_args` to `_target_`/kwargs."""
    if isinstance(node, dict):
        if "class_path" in node:
            target = TARGET_REWRITES.get(node["class_path"], node["class_path"])
            init_args = node.get("init_args", {}) or {}
            out: dict[str, Any] = {"_target_": target}
            for k, v in init_args.items():
                out[k] = _transform(v)
            return out
        return {k: _transform(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_transform(x) for x in node]
    return node


def _unwrap_top_level(raw: dict[str, Any]) -> dict[str, Any]:
    """Lightning CLI YAMLs always start with one of {data, model, trainer}.
    Hydra expects the config to BE that object, so strip the outer key.
    """
    if len(raw) != 1:
        raise ValueError(f"expected single top-level key, got {list(raw.keys())}")
    return next(iter(raw.values()))


def convert_tree(cfg_subdir: str, conf_subdir: str) -> list[Path]:
    """Convert every YAML under `cfg/<cfg_subdir>/` to `conf/<conf_subdir>/`.

    Returns the list of written files.
    """
    src_root = CFG_ROOT / cfg_subdir
    dst_root = CONF_ROOT / conf_subdir
    written: list[Path] = []

    for src in sorted(src_root.rglob("*.yaml")):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        raw = yaml.safe_load(src.read_text())
        inner = _unwrap_top_level(raw)
        converted = _transform(inner)
        dst.write_text(yaml.safe_dump(converted, sort_keys=False))
        written.append(dst)

    return written


def validate_targets(files: list[Path]) -> list[tuple[Path, str, str]]:
    """Walk each YAML, `hydra.utils.get_class` every `_target_`.

    Returns (file, target, error_msg) tuples for any that failed to resolve.
    """
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    failures: list[tuple[Path, str, str]] = []

    def walk(cfg, file: Path):
        if isinstance(cfg, dict) or hasattr(cfg, "items"):
            items = cfg.items() if hasattr(cfg, "items") else cfg.items()
            for k, v in items:
                if k == "_target_":
                    try:
                        get_class(v)
                    except Exception as e:
                        failures.append((file, v, f"{type(e).__name__}: {e}"))
                else:
                    walk(v, file)
        elif isinstance(cfg, list) or hasattr(cfg, "__iter__") and not isinstance(cfg, (str, bytes)):
            for item in cfg:
                walk(item, file)

    for f in files:
        cfg = OmegaConf.load(f)
        walk(cfg, f)

    return failures


def main() -> int:
    if not CFG_ROOT.exists():
        print(f"error: {CFG_ROOT} does not exist", file=sys.stderr)
        return 2

    CONF_ROOT.mkdir(exist_ok=True)

    print("Converting cfg/data -> conf/data ...")
    data_files = convert_tree("data", "data")
    print(f"  wrote {len(data_files)} file(s)")

    print("Converting cfg/model -> conf/model ...")
    model_files = convert_tree("model", "model")
    print(f"  wrote {len(model_files)} file(s)")

    all_files = data_files + model_files
    print(f"\nValidating {len(all_files)} file(s) ...")
    failures = validate_targets(all_files)
    if failures:
        print(f"\n{len(failures)} unresolvable _target_(s):")
        for f, tgt, err in failures:
            print(f"  {f.relative_to(REPO_ROOT)}: {tgt}  ({err})")
        return 1

    print("  all _target_ symbols resolve OK")
    print("\nNOTE: trainer YAMLs are NOT converted by this script — they are")
    print("hand-authored in conf/trainer/ because they need structural")
    print("reorganization (callback list -> dict, split logger/callbacks into")
    print("their own config groups).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
