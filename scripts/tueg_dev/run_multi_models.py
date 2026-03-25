#!/usr/bin/env python3
import argparse
import ast
import copy
import gc
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.trainer import Trainer
from datasets.builder import build_dataloader
from models.wrapper import CBraModWrapper


@dataclass
class Experiment:
    model: str
    variant: str
    script_path: str
    config_path: str
    opts: Dict[str, object]
    config: Dict


def merge_cfg_from_list(cfg: Dict, opts_list: List[str]) -> Dict:
    for opt in opts_list:
        key, value = opt.split("=", 1)
        keys = key.split(".")
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].split(",")
            value = [v.strip().replace("'", "").replace('"', "") for v in value]
        else:
            if isinstance(value, str) and value.lower() == "true":
                value = True
            elif isinstance(value, str) and value.lower() == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
        sub_cfg = cfg
        for k in keys[:-1]:
            sub_cfg = sub_cfg.setdefault(k, {})
        sub_cfg[keys[-1]] = value
    return cfg


def set_nested(cfg: Dict, key: str, value) -> None:
    keys = key.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def parse_scalar(value: str):
    v = value.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return ast.literal_eval(v)
    except Exception:
        return v.strip("\"'")


def expand_with_vars(text: str, vars_map: Dict[str, str]) -> str:
    s = text
    for _ in range(3):
        s = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*)\}",
            lambda m: str(vars_map.get(m.group(1), os.environ.get(m.group(1), m.group(2)))),
            s,
        )
        s = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda m: str(vars_map.get(m.group(1), os.environ.get(m.group(1), m.group(0)))),
            s,
        )
        s = re.sub(
            r"\$([A-Za-z_][A-Za-z0-9_]*)",
            lambda m: str(vars_map.get(m.group(1), os.environ.get(m.group(1), m.group(0)))),
            s,
        )
    return s


def parse_script(script_path: str) -> Tuple[str, Dict[str, object], Dict[str, str]]:
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    vars_map: Dict[str, str] = {}
    assign_pat = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

    for line in lines:
        s = line.strip()
        m = assign_pat.match(s)
        if not m:
            continue
        key = m.group(1)
        raw_v = m.group(2).strip()
        if (raw_v.startswith("'") and raw_v.endswith("'")) or (raw_v.startswith('"') and raw_v.endswith('"')):
            raw_v = raw_v[1:-1]
        vars_map[key] = expand_with_vars(raw_v, vars_map)

    cmd_lines: List[str] = []
    started = False
    for line in lines:
        s = line.strip()
        if not started and s.startswith("python3 main.py"):
            started = True
        if started:
            cmd_lines.append(s.rstrip("\\").strip())

    cmd = " ".join(cmd_lines)
    tokens = shlex.split(cmd)

    if "--config" not in tokens:
        raise ValueError(f"No --config found in script: {script_path}")
    config_path = tokens[tokens.index("--config") + 1]

    opts: Dict[str, object] = {}
    if "--opts" in tokens:
        idx = tokens.index("--opts")
        for t in tokens[idx + 1 :]:
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            v_expanded = expand_with_vars(v, vars_map)
            opts[k] = parse_scalar(v_expanded)

    return config_path, opts, vars_map


def discover_scripts(script_dir: str) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    pat = re.compile(r"^run_([a-z0-9]+)_(recon|joint|feat_only|feat_scheme_a|feat_scheme_b)\.sh$")
    for p in sorted(Path(script_dir).glob("run_*.sh")):
        m = pat.match(p.name)
        if not m:
            continue
        model = m.group(1)
        variant = m.group(2)
        mapping[(model, variant)] = str(p)
    return mapping


def normalize_variant(v: str) -> str:
    m = {
        "recon": "recon",
        "joint": "joint",
        "feat_only": "feat_only",
        "scheme_a": "feat_scheme_a",
        "scheme_b": "feat_scheme_b",
        "feat_scheme_a": "feat_scheme_a",
        "feat_scheme_b": "feat_scheme_b",
    }
    vv = v.strip().lower()
    if vv not in m:
        raise ValueError(f"Unsupported variant: {v}")
    return m[vv]


def flatten_dataset_cfg(config: Dict) -> Dict:
    cfg = copy.deepcopy(config)
    if "datasets" in cfg:
        selected = cfg.get("selected_dataset")
        if not selected:
            raise ValueError("Config has datasets but no selected_dataset")
        if selected not in cfg["datasets"]:
            raise ValueError(f"selected_dataset {selected} not in config.datasets")
        cfg["dataset"] = cfg["datasets"][selected]
    if "dataset" not in cfg:
        raise ValueError("Config must contain dataset section")
    return cfg


def signature_for_dataset(dataset_cfg: Dict) -> str:
    return json.dumps(dataset_cfg, sort_keys=True, ensure_ascii=False)


def maybe_sync_feature_dim(config: Dict, train_loader) -> None:
    dataset_obj = train_loader.dataset
    if hasattr(dataset_obj, "dataset"):
        dataset_obj = dataset_obj.dataset
    if hasattr(dataset_obj, "feature_dim") and getattr(dataset_obj, "feature_dim", 0) > 0:
        detected_dim = int(dataset_obj.feature_dim)
        config_dim = int(config.get("model", {}).get("feature_dim", 0))
        if detected_dim != config_dim:
            print(
                f"[FeatureDim] model.feature_dim {config_dim} -> {detected_dim} (from dataset)",
                flush=True,
            )
            config.setdefault("model", {})["feature_dim"] = detected_dim


def resolve_resume(output_dir: str, policy: str) -> Optional[str]:
    latest = os.path.join(output_dir, "latest.pth")
    if policy == "never":
        return None
    if policy == "always":
        return latest if os.path.isfile(latest) else None
    if os.path.isfile(latest):
        return latest
    return None


def close_loader_dataset(loader) -> None:
    if loader is None:
        return
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    if hasattr(ds, "close"):
        try:
            ds.close()
        except Exception:
            pass


def _fmt_seconds(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def remap_output_dir(original_output_dir: str, output_root: str) -> str:
    norm = (original_output_dir or "").strip().strip("/")
    if not norm:
        return output_root
    leaf = os.path.basename(norm)
    if not leaf:
        return output_root
    return os.path.join(output_root, leaf)


def build_experiments(script_dir: str, models: List[str], variants: List[str], tiny: bool) -> List[Experiment]:
    scripts = discover_scripts(script_dir)
    experiments: List[Experiment] = []

    for model in models:
        for variant in variants:
            key = (model, variant)
            if key not in scripts:
                raise FileNotFoundError(f"Script not found for model={model}, variant={variant}")
            spath = scripts[key]
            config_path, opts_map, _ = parse_script(spath)

            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            cfg = flatten_dataset_cfg(cfg)
            for k, v in opts_map.items():
                set_nested(cfg, k, v)
            if tiny:
                cfg.setdefault("dataset", {})["tiny"] = True

            experiments.append(
                Experiment(
                    model=model,
                    variant=variant,
                    script_path=spath,
                    config_path=config_path,
                    opts=opts_map,
                    config=cfg,
                )
            )
    return experiments


def main() -> int:
    ap = argparse.ArgumentParser(description="Run selected TUEG models with shared dataloader(s) in one process")
    ap.add_argument("--script-dir", default="scripts/tueg_dev", help="Directory that contains run_*.sh")
    ap.add_argument("--models", default="cbramod,eegmamba,reve,tech", help="Comma-separated model list")
    ap.add_argument(
        "--variants",
        default="joint",
        help="Comma-separated variants: recon,joint,feat_only,scheme_a,scheme_b",
    )
    ap.add_argument("--tiny", action="store_true", help="Enable tiny mode for quick debug")
    ap.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    ap.add_argument(
        "--output-root",
        default="",
        help="If set, remap each run output_dir into this root while keeping per-run leaf folder",
    )
    ap.add_argument("--batch-size", type=int, default=0, help="Override dataset.batch_size for all selected runs")
    ap.add_argument("--num-workers", type=int, default=-1, help="Override dataset.num_workers for all selected runs")
    ap.add_argument("--prefetch-factor", type=int, default=-1, help="Override dataset.prefetch_factor for all selected runs")
    ap.add_argument("--val-num-workers", type=int, default=-1, help="Override dataset.val_num_workers for all selected runs")
    ap.add_argument(
        "--val-persistent-workers",
        choices=["true", "false"],
        default="",
        help="Override dataset.val_persistent_workers for all selected runs",
    )
    ap.add_argument("--val-prefetch-factor", type=int, default=-1, help="Override dataset.val_prefetch_factor for all selected runs")
    ap.add_argument("--loader-timeout", type=int, default=-1, help="Override dataset.loader_timeout for all selected runs")
    ap.add_argument("--val-timeout", type=int, default=-1, help="Override dataset.val_timeout for all selected runs")
    ap.add_argument("--val-freq-split", type=int, default=-1, help="Override val_freq_split for all selected runs")
    ap.add_argument("--epochs", type=int, default=-1, help="Override epochs for all selected runs")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = ap.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    variants = [normalize_variant(v) for v in args.variants.split(",") if v.strip()]

    valid_models = {"cbramod", "eegmamba", "reve", "tech"}
    unknown = [m for m in models if m not in valid_models]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    exps = build_experiments(args.script_dir, models, variants, args.tiny)
    if not exps:
        raise RuntimeError("No experiments selected")

    output_root = args.output_root.strip()
    if output_root:
        output_root = os.path.abspath(output_root)
        os.makedirs(output_root, exist_ok=True)
        for e in exps:
            old_out = str(e.config.get("output_dir", f"output/tueg_dev/{e.model}_{e.variant}"))
            e.config["output_dir"] = remap_output_dir(old_out, output_root)

    for e in exps:
        ds = e.config.setdefault("dataset", {})
        if args.batch_size > 0:
            ds["batch_size"] = int(args.batch_size)
        if args.num_workers >= 0:
            ds["num_workers"] = int(args.num_workers)
        if args.prefetch_factor >= 0:
            ds["prefetch_factor"] = int(args.prefetch_factor)
        if args.val_num_workers >= 0:
            ds["val_num_workers"] = int(args.val_num_workers)
        if args.val_persistent_workers:
            ds["val_persistent_workers"] = args.val_persistent_workers.lower() == "true"
        if args.val_prefetch_factor >= 0:
            ds["val_prefetch_factor"] = int(args.val_prefetch_factor)
        if args.loader_timeout >= 0:
            ds["loader_timeout"] = int(args.loader_timeout)
        if args.val_timeout >= 0:
            ds["val_timeout"] = int(args.val_timeout)
        if args.val_freq_split >= 0:
            e.config["val_freq_split"] = int(args.val_freq_split)
        if args.epochs > 0:
            e.config["epochs"] = int(args.epochs)

        # torch DataLoader with num_workers=0 requires timeout=0.
        if int(ds.get("val_num_workers", -1)) == 0 and int(ds.get("val_timeout", 0)) > 0:
            print(
                f"[ConfigFix] val_num_workers=0 requires val_timeout=0. Auto-fix: {ds.get('val_timeout')} -> 0",
                flush=True,
            )
            ds["val_timeout"] = 0

    groups: Dict[str, List[Experiment]] = {}
    for e in exps:
        sig = signature_for_dataset(e.config["dataset"])
        groups.setdefault(sig, []).append(e)

    print("=== Multi-Model Plan ===", flush=True)
    print(f"Selected models: {models}", flush=True)
    print(f"Selected variants: {variants}", flush=True)
    if output_root:
        print(f"Output root remap: {output_root}", flush=True)
    print(f"Dataset groups: {len(groups)}", flush=True)
    for idx, (_, gexps) in enumerate(groups.items(), 1):
        first = gexps[0]
        bs = first.config["dataset"].get("batch_size")
        nw = first.config["dataset"].get("num_workers")
        print(f"  Group {idx}: batch_size={bs}, num_workers={nw}, runs={len(gexps)}", flush=True)
        for e in gexps:
            out_dir = e.config.get("output_dir", "output")
            print(f"    - {e.model}/{e.variant} -> {out_dir}", flush=True)

    if args.dry_run:
        return 0

    summary_rows: List[Dict[str, str]] = []
    all_start = time.time()
    had_failure = False

    try:
        for group_idx, (_, gexps) in enumerate(groups.items(), 1):
            g0 = gexps[0]
            dataset_cfg = copy.deepcopy(g0.config["dataset"])
            dataset_name = dataset_cfg.get("name")
            print(
                f"\n[Group {group_idx}] Building shared dataloaders: dataset={dataset_name}, batch_size={dataset_cfg.get('batch_size')}",
                flush=True,
            )

            train_loader = build_dataloader(dataset_name, dataset_cfg, mode="train")

            need_val = any(int(e.config.get("val_freq_split", 0)) > 0 for e in gexps)
            val_loader = build_dataloader(dataset_name, dataset_cfg, mode="val") if need_val else None

            for run_idx, exp in enumerate(gexps, 1):
                cfg = copy.deepcopy(exp.config)
                maybe_sync_feature_dim(cfg, train_loader)

                output_dir = cfg.get("output_dir", f"output/tueg_dev/{exp.model}_{exp.variant}")
                os.makedirs(output_dir, exist_ok=True)
                resume_path = resolve_resume(output_dir, args.resume)

                print(
                    f"[Group {group_idx}][{run_idx}/{len(gexps)}] Start {exp.model}/{exp.variant} | output={output_dir}",
                    flush=True,
                )
                if resume_path:
                    print(f"  Resume: {resume_path}", flush=True)

                start_t = time.time()
                status = "ok"
                err_msg = ""
                try:
                    cfg.setdefault("model", {})["task_type"] = cfg.get("task_type")
                    model = CBraModWrapper(cfg)
                    trainer = Trainer(cfg, model, train_loader, val_loader)
                    trainer.train(resume_path=resume_path)
                except Exception as e:
                    status = "failed"
                    err_msg = str(e)
                    had_failure = True
                finally:
                    elapsed = _fmt_seconds(time.time() - start_t)
                    summary_rows.append(
                        {
                            "run": f"{exp.model}/{exp.variant}",
                            "status": status,
                            "elapsed": elapsed,
                            "output": output_dir,
                            "error": err_msg,
                        }
                    )
                    print(
                        f"[RunSummary] {exp.model}/{exp.variant} | status={status} | elapsed={elapsed}",
                        flush=True,
                    )

                    if "trainer" in locals():
                        del trainer
                    if "model" in locals():
                        del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if status != "ok":
                    break

            close_loader_dataset(train_loader)
            close_loader_dataset(val_loader)
            del train_loader
            del val_loader
            gc.collect()

            if had_failure:
                break
    finally:
        print("\n=== Run Summary (short) ===", flush=True)
        for row in summary_rows:
            line = f"- {row['run']}: {row['status']} | {row['elapsed']} | {row['output']}"
            print(line, flush=True)
            if row["status"] != "ok" and row["error"]:
                print(f"  error: {row['error']}", flush=True)
        print(f"Total elapsed: {_fmt_seconds(time.time() - all_start)}", flush=True)

    if had_failure:
        print("Stopped early due to failure. Check summary above.", flush=True)
        return 1

    print("All selected experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
