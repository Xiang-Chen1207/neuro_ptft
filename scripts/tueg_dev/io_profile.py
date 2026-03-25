import argparse
import ast
import cProfile
import io
import json
import os
import pstats
import shlex
import time
import re

import yaml

from datasets.builder import build_dataloader
from datasets.tueg import TUEGDataset


def _parse_scalar(value):
    v = value.strip()
    v = os.path.expandvars(v)
    if v.startswith("${") and v.endswith("}"):
        name = v[2:-1]
        env_v = os.environ.get(name)
        if env_v is not None:
            return _parse_scalar(env_v)
    if v.startswith("$") and len(v) > 1:
        name = v[1:]
        env_v = os.environ.get(name)
        if env_v is not None:
            return _parse_scalar(env_v)
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return ast.literal_eval(v)
    except Exception:
        return v.strip("\"'")


def _set_nested(cfg, key, value):
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _expand_with_vars(text, vars_map):
    s = text
    for _ in range(2):
        s = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*)\}", lambda m: str(vars_map.get(m.group(1), m.group(2))), s)
        s = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(vars_map.get(m.group(1), os.environ.get(m.group(1), m.group(0)))), s)
        s = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: str(vars_map.get(m.group(1), os.environ.get(m.group(1), m.group(0)))), s)
    return s


def _parse_script(script_path):
    with open(script_path, "r") as f:
        lines = f.readlines()
    vars_map = {}
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
        vars_map[key] = _expand_with_vars(raw_v, vars_map)
    cmd_lines = []
    started = False
    for line in lines:
        s = line.strip()
        if not started and s.startswith("python3 main.py"):
            started = True
        if started:
            cmd_lines.append(s.rstrip("\\").strip())
    cmd = " ".join(cmd_lines)
    tokens = shlex.split(cmd)
    config_path = None
    opts = {}
    if "--config" in tokens:
        idx = tokens.index("--config")
        config_path = tokens[idx + 1]
    if "--opts" in tokens:
        idx = tokens.index("--opts")
        for t in tokens[idx + 1 :]:
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            v = _expand_with_vars(v, vars_map)
            opts[k] = _parse_scalar(v)
    if config_path is None:
        raise ValueError(f"未在脚本中找到 --config: {script_path}")
    return config_path, opts


def _collect_batches(loader, warmup, batches):
    it = iter(loader)
    w = 0
    retry_guard = 0
    while w < warmup:
        try:
            b = next(it)
            if b is None:
                retry_guard += 1
                if retry_guard > 1000:
                    raise RuntimeError("dataloader连续返回空batch，无法完成warmup")
                continue
            w += 1
        except StopIteration:
            it = iter(loader)
    n = 0
    samples = 0
    t0 = time.perf_counter()
    while n < batches:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if batch is None:
            retry_guard += 1
            if retry_guard > 5000:
                raise RuntimeError("dataloader连续返回空batch，无法完成采样")
            continue
        x = batch["x"] if isinstance(batch, dict) else batch[0]
        samples += int(x.shape[0])
        n += 1
    elapsed = time.perf_counter() - t0
    return {
        "batches": n,
        "samples": samples,
        "elapsed_s": elapsed,
        "batch_s": elapsed / max(n, 1),
        "samples_per_s": samples / max(elapsed, 1e-9),
    }


def _line_profile_once(cfg, warmup, batches):
    try:
        from line_profiler import LineProfiler
    except Exception as e:
        return {"available": False, "error": str(e)}
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["dataset"]["num_workers"] = 0
    cfg2["dataset"]["persistent_workers"] = False
    loader = build_dataloader(cfg2["dataset"]["name"], cfg2["dataset"], mode="train")
    lp = LineProfiler()
    lp.add_function(TUEGDataset.__getitem__)
    wrapped = lp(_collect_batches)
    _ = wrapped(loader, warmup, batches)
    s = io.StringIO()
    lp.print_stats(stream=s)
    return {"available": True, "stats": s.getvalue()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--prefetch-factor", type=int, default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    config_path, opts = _parse_script(args.script)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in opts.items():
        _set_nested(cfg, k, v)
    if args.tiny:
        _set_nested(cfg, "dataset.tiny", True)
    if args.num_workers is not None:
        _set_nested(cfg, "dataset.num_workers", args.num_workers)
    if args.prefetch_factor is not None:
        _set_nested(cfg, "dataset.prefetch_factor", args.prefetch_factor)
    for item in args.set:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        _set_nested(cfg, k.strip(), _parse_scalar(v.strip()))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cprof_path = args.output.replace(".json", ".cprofile")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    loader = build_dataloader(cfg["dataset"]["name"], cfg["dataset"], mode="train")
    init_elapsed = time.perf_counter() - t0
    pr.enable()
    bench = _collect_batches(loader, args.warmup, args.batches)
    pr.disable()
    pr.dump_stats(cprof_path)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(30)

    line_stats = _line_profile_once(cfg, warmup=2, batches=min(10, args.batches))

    report = {
        "script": args.script,
        "config_path": config_path,
        "dataset_name": cfg["dataset"]["name"],
        "dataset_dir": cfg["dataset"].get("dataset_dir"),
        "feature_path": cfg["dataset"].get("feature_path"),
        "init_elapsed_s": init_elapsed,
        "benchmark": bench,
        "cprofile_top": s.getvalue(),
        "line_profiler": line_stats,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["benchmark"], ensure_ascii=False))
    print(f"saved: {args.output}")
    print(f"saved: {cprof_path}")


if __name__ == "__main__":
    main()
