#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage1Case:
    name: str
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    exact_tile: bool
    group_m: int = 1
    cubin_format: str = "isa"
    triton_kind: str = "triton9"
    triton_binary: str | None = None
    triton_kernel: str = "matmul_kernel"
    triton_shared_fallback: int = 0
    triton_meta: str | None = None

    @property
    def block_x(self) -> int:
        return self.num_warps * 32

    @property
    def grid_x(self) -> int:
        return math.ceil(self.m / self.block_m) * math.ceil(self.n / self.block_n)

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


CASES: tuple[Stage1Case, ...] = (
    Stage1Case(
        name="64x64x32",
        m=64,
        n=64,
        k=32,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=1,
        num_stages=2,
        exact_tile=True,
        triton_kind="triton",
        triton_binary="/tmp/triton_exact_tile_64x64x32_real.cubin",
        triton_kernel="exact_tile_matmul_kernel",
        triton_shared_fallback=12544,
        triton_meta=(
            "/tmp/triton_cache_20260331_current/"
            "Xo37VqbntYhdv4N6n5wnd58PwNPHHtQAf1xXp5G7ru0/"
            "matmul_exact_tile_kernel.json"
        ),
    ),
    Stage1Case(
        name="64x128x32",
        m=64,
        n=128,
        k=32,
        block_m=64,
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=2,
        exact_tile=True,
        triton_kind="triton9",
        triton_binary="/tmp/triton_stage1_64x128x32.cubin",
        triton_kernel="matmul_kernel",
        triton_shared_fallback=16896,
    ),
    Stage1Case(
        name="128x64x32",
        m=128,
        n=64,
        k=32,
        block_m=128,
        block_n=64,
        block_k=32,
        num_warps=4,
        num_stages=2,
        exact_tile=True,
        triton_kind="triton9",
        triton_binary="/tmp/triton_stage1_128x64x32.cubin",
        triton_kernel="matmul_kernel",
        triton_shared_fallback=17408,
    ),
    Stage1Case(
        name="64x64x64",
        m=64,
        n=64,
        k=64,
        block_m=64,
        block_n=64,
        block_k=64,
        num_warps=1,
        num_stages=2,
        exact_tile=True,
        triton_kind="triton9",
        triton_binary="/tmp/triton_stage1_64x64x64.cubin",
        triton_kernel="matmul_kernel",
        triton_shared_fallback=8192,
        triton_meta=(
            "/tmp/triton_stage1_cache_test/"
            "bCFZDj0ru7pAIQM6jSDOhMvV4oNpkyliS7CwAynZ8I8/"
            "matmul_kernel.json"
        ),
    ),
    Stage1Case(
        name="128x128x32",
        m=128,
        n=128,
        k=32,
        block_m=128,
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=2,
        exact_tile=True,
        triton_kind="triton",
        triton_binary="/tmp/triton_exact_tile_128x128x32_real.cubin",
        triton_kernel="exact_tile_matmul_kernel",
        triton_shared_fallback=33280,
    ),
    Stage1Case(
        name="128x128x64",
        m=128,
        n=128,
        k=64,
        block_m=128,
        block_n=128,
        block_k=64,
        num_warps=4,
        num_stages=2,
        exact_tile=True,
        triton_kind="triton9",
        triton_binary="/tmp/triton_stage1_128x128x64.cubin",
        triton_kernel="matmul_kernel",
        triton_shared_fallback=16896,
    ),
    Stage1Case(
        name="96x80x32_general",
        m=96,
        n=80,
        k=32,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=1,
        num_stages=2,
        exact_tile=False,
        triton_kind="triton9",
    ),
    Stage1Case(
        name="192x128x32_grouped",
        m=192,
        n=128,
        k=32,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        num_stages=2,
        exact_tile=True,
        group_m=2,
        triton_kind="triton9",
    ),
    Stage1Case(
        name="160x96x32_general",
        m=160,
        n=96,
        k=32,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        num_stages=2,
        exact_tile=False,
        triton_kind="triton9",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interleaved fair benchmark for mini_triton_nvgpu_v1 stage1 matmul."
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated shape list, e.g. 64x64x32,64x128x32. Default: all",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument(
        "--execution-order",
        choices=("input", "hot_first"),
        default="hot_first",
        help="Measure cases in input order or by descending FLOPs to keep the GPU hot.",
    )
    parser.add_argument(
        "--enable-preheat",
        action="store_true",
        help="Preheat the GPU with a larger stage1 case before timed measurements.",
    )
    parser.add_argument(
        "--preheat-case",
        default="128x128x64",
        help="Shape used for GPU preheat. Default: 128x128x64",
    )
    parser.add_argument("--preheat-warmup", type=int, default=50)
    parser.add_argument("--preheat-iters", type=int, default=20000)
    parser.add_argument("--preheat-max-attempts", type=int, default=3)
    parser.add_argument(
        "--preheat-clock-floor-pct",
        type=float,
        default=90.0,
        help="Repeat preheat until current SM clock reaches this percentage of max clock.",
    )
    parser.add_argument(
        "--unstable-spread-threshold",
        type=float,
        default=10.0,
        help="If mini/triton spread exceeds this percentage, append extra rounds.",
    )
    parser.add_argument(
        "--extra-rounds-for-unstable",
        type=int,
        default=5,
        help="Extra interleaved rounds for unstable cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mini_triton_stage1_fair_bench"),
    )
    parser.add_argument(
        "--tb-opt",
        type=Path,
        default=Path("/home/zhangruiqi/mini_triton_nvgpu_v1/build/bin/tb-opt"),
    )
    parser.add_argument(
        "--extract-script",
        type=Path,
        default=Path("/home/zhangruiqi/triton_backend_nvgpu/tools/extract_gpu_binary.py"),
    )
    parser.add_argument(
        "--triton-builder",
        type=Path,
        default=Path("/home/zhangruiqi/mini_triton_nvgpu_v1/tools/build_triton_stage1_kernel.py"),
    )
    parser.add_argument(
        "--bench-driver",
        type=Path,
        default=Path("/home/zhangruiqi/tmp/driver_matmul_bench"),
    )
    parser.add_argument(
        "--bench-driver-src",
        type=Path,
        default=Path("/home/zhangruiqi/tmp/driver_matmul_bench.cpp"),
    )
    parser.add_argument("--gpu-arch", default="sm_86")
    parser.add_argument("--ptx-features", default="+ptx60")
    parser.add_argument(
        "--skip-mini-build",
        action="store_true",
        help="Reuse already-lowered mini binaries in the output directory.",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def find_cxx() -> str:
    for candidate in ("c++", "g++", "clang++"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("missing C++ compiler: tried c++, g++, clang++")


def parse_int_field(value: str) -> int:
    token = value.strip().split()[0]
    if token in {"N/A", "[N/A]"}:
        return -1
    return int(float(token))


def query_gpu_state() -> dict[str, Any]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    parts = [part.strip() for part in output.split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"unexpected nvidia-smi state line: {output}")
    return {
        "pstate": parts[0],
        "sm_clock_mhz": parse_int_field(parts[1]),
        "sm_clock_max_mhz": parse_int_field(parts[2]),
        "mem_clock_mhz": parse_int_field(parts[3]),
        "temperature_c": parse_int_field(parts[4]),
    }


def clock_floor_mhz(state: dict[str, Any], floor_pct: float) -> int:
    max_clock = state.get("sm_clock_max_mhz", -1)
    if max_clock <= 0:
        return -1
    return math.floor(max_clock * floor_pct / 100.0)


def is_clock_hot_enough(state: dict[str, Any], floor_pct: float) -> bool:
    required = clock_floor_mhz(state, floor_pct)
    current = state.get("sm_clock_mhz", -1)
    return required > 0 and current >= required


def ensure_bench_driver(binary: Path, source: Path) -> None:
    if binary.is_file() and (not source.is_file() or binary.stat().st_mtime >= source.stat().st_mtime):
        return
    ensure_file(source, "bench driver source")
    cxx = find_cxx()
    run(
        [
            cxx,
            "-std=c++20",
            "-O2",
            str(source),
            "-lcuda",
            "-o",
            str(binary),
        ]
    )


def make_case_mlir(case: Stage1Case) -> str:
    group_m_attr = f", group_m = {case.group_m} : i64" if case.group_m != 1 else ""
    return (
        f'module attributes {{"tb.num-warps" = {case.num_warps} : i64, '
        f'"tb.requested-stages" = {case.num_stages} : i64}} {{\n'
        f"  func.func @kernel(%A: memref<{case.m}x{case.k}xf16>, "
        f"%B: memref<{case.k}x{case.n}xf16>, %C: memref<{case.m}x{case.n}xf32>) {{\n"
        f"    tb.matmul %A, %B, %C {{block_m = {case.block_m} : i64, "
        f"block_n = {case.block_n} : i64, block_k = {case.block_k} : i64, "
        f"exact_tile = {'true' if case.exact_tile else 'false'}{group_m_attr}}}\n"
        f"      : memref<{case.m}x{case.k}xf16>, memref<{case.k}x{case.n}xf16>, "
        f"memref<{case.m}x{case.n}xf32>\n"
        f"    func.return\n"
        f"  }}\n"
        f"}}\n"
    )


def lower_mini_case(
    case: Stage1Case,
    out_dir: Path,
    tb_opt: Path,
    extract_script: Path,
    gpu_arch: str,
    ptx_features: str,
) -> Path:
    ensure_file(tb_opt, "tb-opt")
    ensure_file(extract_script, "extract_gpu_binary.py")
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    input_mlir = case_dir / f"{case.name}.mlir"
    lowered_mlir = case_dir / f"{case.name}.lowered.mlir"
    mini_binary = case_dir / f"{case.name}.mini.bin"
    input_mlir.write_text(make_case_mlir(case))
    pass_pipeline = (
        "builtin.module("
        "tb-stage1-full-to-nvvm-pipeline{"
        f"cubin-chip={gpu_arch} "
        f"cubin-features={ptx_features} "
        f"cubin-format={case.cubin_format}" + "})"
    )
    run(
        [
            str(tb_opt),
            f"--pass-pipeline={pass_pipeline}",
            str(input_mlir),
            "-o",
            str(lowered_mlir),
        ]
    )
    run([sys.executable, str(extract_script), str(lowered_mlir), str(mini_binary)])
    return mini_binary


def parse_bench_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw": text}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    attrs = result.get("attrs")
    if isinstance(attrs, str):
        parsed_attrs: dict[str, int] = {}
        for part in attrs.split():
            name, number = part.split(":")
            parsed_attrs[name] = int(number)
        result["attrs_parsed"] = parsed_attrs
    for numeric_key in ("avg_ns", "gflops", "c00", "cmid", "clast", "checksum"):
        if numeric_key in result:
            result[numeric_key] = float(result[numeric_key])
    return result


def read_triton_shared(case: Stage1Case) -> tuple[int, str]:
    if case.triton_meta:
        meta_path = Path(case.triton_meta)
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            shared = int(meta["shared"])
            return shared, str(meta_path)
    return case.triton_shared_fallback, "fallback"


def build_triton_case(
    case: Stage1Case,
    out_dir: Path,
    builder_script: Path,
) -> tuple[Path, str, int, str]:
    ensure_file(builder_script, "build_triton_stage1_kernel.py")
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    output_stem = case_dir / f"{case.name}.triton"
    cubin_path = output_stem.with_suffix(".cubin")
    metadata_path = output_stem.with_suffix(".json")
    if not cubin_path.is_file() or not metadata_path.is_file():
        run(
            [
                sys.executable,
                str(builder_script),
                "--output-stem",
                str(output_stem),
                "--m",
                str(case.m),
                "--n",
                str(case.n),
                "--k",
                str(case.k),
                "--block-m",
                str(case.block_m),
                "--block-n",
                str(case.block_n),
                "--block-k",
                str(case.block_k),
                "--num-warps",
                str(case.num_warps),
                "--num-stages",
                str(case.num_stages),
                "--group-m",
                str(case.group_m),
                "--exact-tile",
                "true" if case.exact_tile else "false",
            ]
        )
    meta = json.loads(metadata_path.read_text())
    return cubin_path, str(meta["kernel_name"]), int(meta["shared"]), str(metadata_path)


def ensure_triton_case(
    case: Stage1Case,
    out_dir: Path,
    builder_script: Path,
) -> tuple[str, Path, str, int, str]:
    if case.triton_binary:
        binary = Path(case.triton_binary)
        ensure_file(binary, f"triton binary for {case.name}")
        shared, source = read_triton_shared(case)
        return case.triton_kind, binary, case.triton_kernel, shared, source
    binary, kernel_name, shared, source = build_triton_case(
        case, out_dir, builder_script
    )
    return case.triton_kind, binary, kernel_name, shared, source


def find_case(name: str) -> Stage1Case:
    for case in CASES:
        if case.name == name:
            return case
    raise ValueError(f"unknown case: {name}")


def run_bench_once(
    bench_driver: Path,
    kind: str,
    binary: Path,
    kernel: str,
    case: Stage1Case,
    warmup: int,
    iters: int,
    shared_bytes: int,
    verify: bool,
) -> dict[str, Any]:
    ensure_file(bench_driver, "driver_matmul_bench")
    ensure_file(binary, f"{kind} binary")
    cmd = [
        str(bench_driver),
        f"--kind={kind}",
        f"--binary={binary}",
        f"--kernel={kernel}",
        f"--m={case.m}",
        f"--n={case.n}",
        f"--k={case.k}",
        f"--block-x={case.block_x}",
        "--block-y=1",
        "--block-z=1",
        f"--grid-x={case.grid_x}",
        "--grid-y=1",
        "--grid-z=1",
        f"--shared={shared_bytes}",
        f"--warmup={warmup}",
        f"--iters={iters}",
    ]
    if not verify:
        cmd.append("--no-verify")
    output = run(cmd)
    return parse_bench_output(output)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    summary = {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }
    summary["spread_pct"] = (summary["max"] - summary["min"]) / summary["median"] * 100.0
    if len(values) > 1:
        summary["stdev"] = statistics.stdev(values)
    else:
        summary["stdev"] = 0.0
    return summary


def spread_exceeds(summary: dict[str, float], threshold_pct: float) -> bool:
    return summary["spread_pct"] > threshold_pct


def format_delta(mini_ns: float, triton_ns: float) -> str:
    if mini_ns < triton_ns:
        return f"mini 快 {(triton_ns / mini_ns - 1.0) * 100.0:.2f}%"
    if mini_ns > triton_ns:
        return f"mini 慢 {(mini_ns / triton_ns - 1.0) * 100.0:.2f}%"
    return "持平"


def select_cases(selector: str) -> list[Stage1Case]:
    if selector == "all":
        return list(CASES)
    wanted = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [case for case in CASES if case.name in wanted]
    missing = sorted(wanted - {case.name for case in selected})
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return selected


def order_cases(cases: list[Stage1Case], mode: str) -> list[Stage1Case]:
    if mode == "input":
        return list(cases)
    if mode == "hot_first":
        return sorted(cases, key=lambda case: case.flops, reverse=True)
    raise ValueError(f"unknown execution order: {mode}")


def collect_interleaved_samples(
    case: Stage1Case,
    mini_binary: Path,
    triton_kind: str,
    triton_binary: Path,
    triton_kernel: str,
    bench_driver: Path,
    rounds: int,
    warmup: int,
    iters: int,
    triton_shared: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mini_samples: list[dict[str, Any]] = []
    triton_samples: list[dict[str, Any]] = []
    for round_index in range(rounds):
        order = ("mini", "triton") if round_index % 2 == 0 else ("triton", "mini")
        for label in order:
            if label == "mini":
                mini_samples.append(
                    run_bench_once(
                        bench_driver,
                        "mini",
                        mini_binary,
                        "kernel_tb_kernel_0",
                        case,
                        warmup,
                        iters,
                        0,
                        False,
                    )
                )
            else:
                triton_samples.append(
                    run_bench_once(
                        bench_driver,
                        triton_kind,
                        triton_binary,
                        triton_kernel,
                        case,
                        warmup,
                        iters,
                        triton_shared,
                        False,
                    )
                )
    return mini_samples, triton_samples


def preheat_gpu(
    heater_case: Stage1Case,
    mini_binary: Path,
    triton_kind: str,
    triton_binary: Path,
    triton_kernel: str,
    bench_driver: Path,
    warmup: int,
    iters: int,
    triton_shared: int,
    max_attempts: int,
    floor_pct: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        before = query_gpu_state()
        mini = run_bench_once(
            bench_driver,
            "mini",
            mini_binary,
            "kernel_tb_kernel_0",
            heater_case,
            warmup,
            iters,
            0,
            False,
        )
        triton = run_bench_once(
            bench_driver,
            triton_kind,
            triton_binary,
            triton_kernel,
            heater_case,
            warmup,
            iters,
            triton_shared,
            False,
        )
        after = query_gpu_state()
        attempt_record = {
            "attempt": attempt + 1,
            "before": before,
            "after": after,
            "mini_avg_ns": mini["avg_ns"],
            "triton_avg_ns": triton["avg_ns"],
        }
        attempts.append(attempt_record)
        if is_clock_hot_enough(after, floor_pct):
            return {
                "enabled": True,
                "heater_case": heater_case.name,
                "attempts": attempts,
                "target_floor_pct": floor_pct,
                "target_floor_mhz": clock_floor_mhz(after, floor_pct),
                "reached_hot_clock": True,
            }
    final_state = attempts[-1]["after"] if attempts else query_gpu_state()
    return {
        "enabled": True,
        "heater_case": heater_case.name,
        "attempts": attempts,
        "target_floor_pct": floor_pct,
        "target_floor_mhz": clock_floor_mhz(final_state, floor_pct),
        "reached_hot_clock": is_clock_hot_enough(final_state, floor_pct),
    }


def main() -> int:
    args = parse_args()
    cases = select_cases(args.cases)
    execution_cases = order_cases(cases, args.execution_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_bench_driver(args.bench_driver, args.bench_driver_src)
    mini_binaries: dict[str, Path] = {}
    triton_cases: dict[str, dict[str, Any]] = {}
    for case in cases:
        if args.skip_mini_build:
            mini_binary = args.output_dir / case.name / f"{case.name}.mini.bin"
            ensure_file(mini_binary, f"cached mini binary for {case.name}")
        else:
            mini_binary = lower_mini_case(
                case,
                args.output_dir,
                args.tb_opt,
                args.extract_script,
                args.gpu_arch,
                args.ptx_features,
            )
        mini_binaries[case.name] = mini_binary
        triton_kind, triton_binary, triton_kernel, triton_shared, triton_shared_source = (
            ensure_triton_case(case, args.output_dir, args.triton_builder)
        )
        triton_cases[case.name] = {
            "kind": triton_kind,
            "binary": triton_binary,
            "kernel": triton_kernel,
            "shared": triton_shared,
            "shared_source": triton_shared_source,
        }

    report: dict[str, Any] = {
        "rounds": args.rounds,
        "warmup": args.warmup,
        "iters": args.iters,
        "gpu_state_before_all": query_gpu_state(),
        "cases": [],
    }
    preheat_report: dict[str, Any] = {"enabled": False}
    if args.enable_preheat:
        heater_case = find_case(args.preheat_case)
        if heater_case.name not in mini_binaries:
            if args.skip_mini_build:
                cached_heater_binary = (
                    args.output_dir / heater_case.name / f"{heater_case.name}.mini.bin"
                )
                ensure_file(cached_heater_binary, f"cached mini binary for {heater_case.name}")
                mini_binaries[heater_case.name] = cached_heater_binary
            else:
                mini_binaries[heater_case.name] = lower_mini_case(
                    heater_case,
                    args.output_dir,
                    args.tb_opt,
                    args.extract_script,
                    args.gpu_arch,
                    args.ptx_features,
                )
        if heater_case.name not in triton_cases:
            triton_kind, triton_binary, triton_kernel, triton_shared, triton_shared_source = (
                ensure_triton_case(heater_case, args.output_dir, args.triton_builder)
            )
            triton_cases[heater_case.name] = {
                "kind": triton_kind,
                "binary": triton_binary,
                "kernel": triton_kernel,
                "shared": triton_shared,
                "shared_source": triton_shared_source,
            }
        heater_triton = triton_cases[heater_case.name]
        preheat_report = preheat_gpu(
            heater_case,
            mini_binaries[heater_case.name],
            heater_triton["kind"],
            heater_triton["binary"],
            heater_triton["kernel"],
            args.bench_driver,
            args.preheat_warmup,
            args.preheat_iters,
            heater_triton["shared"],
            args.preheat_max_attempts,
            args.preheat_clock_floor_pct,
        )
    report["preheat"] = preheat_report
    report["gpu_state_after_preheat"] = query_gpu_state()
    report["execution_order"] = args.execution_order
    report["requested_case_order"] = [case.name for case in cases]
    report["measured_case_order"] = [case.name for case in execution_cases]

    case_summaries_by_name: dict[str, dict[str, Any]] = {}
    for case in execution_cases:
        gpu_state_before_case = query_gpu_state()
        triton_case = triton_cases[case.name]
        triton_shared = triton_case["shared"]
        triton_shared_source = triton_case["shared_source"]
        case_preheat_report: dict[str, Any] | None = None
        if args.enable_preheat and not is_clock_hot_enough(
            gpu_state_before_case, args.preheat_clock_floor_pct
        ):
            heater_case = find_case(args.preheat_case)
            if heater_case.name not in triton_cases:
                triton_kind, triton_binary, triton_kernel, triton_shared, triton_shared_source = (
                    ensure_triton_case(heater_case, args.output_dir, args.triton_builder)
                )
                triton_cases[heater_case.name] = {
                    "kind": triton_kind,
                    "binary": triton_binary,
                    "kernel": triton_kernel,
                    "shared": triton_shared,
                    "shared_source": triton_shared_source,
                }
            heater_triton = triton_cases[heater_case.name]
            case_preheat_report = preheat_gpu(
                heater_case,
                mini_binaries[heater_case.name],
                heater_triton["kind"],
                heater_triton["binary"],
                heater_triton["kernel"],
                args.bench_driver,
                args.preheat_warmup,
                args.preheat_iters,
                heater_triton["shared"],
                args.preheat_max_attempts,
                args.preheat_clock_floor_pct,
            )
            gpu_state_before_case = query_gpu_state()
        run_bench_once(
            args.bench_driver,
            "mini",
            mini_binaries[case.name],
            "kernel_tb_kernel_0",
            case,
            warmup=5,
            iters=10,
            shared_bytes=0,
            verify=True,
        )
        run_bench_once(
            args.bench_driver,
            triton_case["kind"],
            triton_case["binary"],
            triton_case["kernel"],
            case,
            warmup=5,
            iters=10,
            shared_bytes=triton_shared,
            verify=True,
        )
        mini_samples, triton_samples = collect_interleaved_samples(
            case,
            mini_binaries[case.name],
            triton_case["kind"],
            triton_case["binary"],
            triton_case["kernel"],
            args.bench_driver,
            args.rounds,
            args.warmup,
            args.iters,
            triton_shared,
        )

        mini_ns = [sample["avg_ns"] for sample in mini_samples]
        triton_ns = [sample["avg_ns"] for sample in triton_samples]
        mini_gflops = [sample["gflops"] for sample in mini_samples]
        triton_gflops = [sample["gflops"] for sample in triton_samples]
        mini_summary = summarize(mini_ns)
        triton_summary = summarize(triton_ns)
        mini_gflops_summary = summarize(mini_gflops)
        triton_gflops_summary = summarize(triton_gflops)
        extra_rounds_run = 0
        if (
            args.extra_rounds_for_unstable > 0
            and (
                spread_exceeds(mini_summary, args.unstable_spread_threshold)
                or spread_exceeds(triton_summary, args.unstable_spread_threshold)
            )
        ):
            extra_rounds_run = args.extra_rounds_for_unstable
            extra_mini_samples, extra_triton_samples = collect_interleaved_samples(
                case,
                mini_binaries[case.name],
                triton_case["kind"],
                triton_case["binary"],
                triton_case["kernel"],
                args.bench_driver,
                args.extra_rounds_for_unstable,
                args.warmup,
                args.iters,
                triton_shared,
            )
            mini_samples.extend(extra_mini_samples)
            triton_samples.extend(extra_triton_samples)
            mini_ns = [sample["avg_ns"] for sample in mini_samples]
            triton_ns = [sample["avg_ns"] for sample in triton_samples]
            mini_gflops = [sample["gflops"] for sample in mini_samples]
            triton_gflops = [sample["gflops"] for sample in triton_samples]
            mini_summary = summarize(mini_ns)
            triton_summary = summarize(triton_ns)
            mini_gflops_summary = summarize(mini_gflops)
            triton_gflops_summary = summarize(triton_gflops)
        unstable = (
            spread_exceeds(mini_summary, args.unstable_spread_threshold)
            or spread_exceeds(triton_summary, args.unstable_spread_threshold)
        )
        case_summary = {
            "shape": case.name,
            "config": {
                "m": case.m,
                "n": case.n,
                "k": case.k,
                "block_m": case.block_m,
                "block_n": case.block_n,
                "block_k": case.block_k,
                "num_warps": case.num_warps,
                "num_stages": case.num_stages,
                "exact_tile": case.exact_tile,
                "group_m": case.group_m,
                "block_x": case.block_x,
                "grid_x": case.grid_x,
                "cubin_format": case.cubin_format,
            },
            "gpu_state_before_case": gpu_state_before_case,
            "gpu_state_after_case": query_gpu_state(),
            "case_preheat": case_preheat_report,
            "mini": {
                "binary": str(mini_binaries[case.name]),
                "summary_ns": mini_summary,
                "summary_gflops": mini_gflops_summary,
                "attrs": mini_samples[0].get("attrs_parsed", {}),
                "samples": mini_samples,
            },
            "triton": {
                "binary": str(triton_case["binary"]),
                "kernel": triton_case["kernel"],
                "kind": triton_case["kind"],
                "shared_bytes": triton_shared,
                "shared_source": triton_shared_source,
                "summary_ns": triton_summary,
                "summary_gflops": triton_gflops_summary,
                "attrs": triton_samples[0].get("attrs_parsed", {}),
                "samples": triton_samples,
            },
            "comparison": {
                "median_ns_delta": format_delta(
                    mini_summary["median"], triton_summary["median"]
                ),
                "mini_over_triton_median_pct": triton_summary["median"]
                / mini_summary["median"]
                * 100.0,
            },
            "measurement": {
                "unstable": unstable,
                "unstable_spread_threshold": args.unstable_spread_threshold,
                "extra_rounds_run": extra_rounds_run,
                "total_rounds_run": len(mini_samples),
            },
        }
        case_summaries_by_name[case.name] = case_summary

    report["cases"] = [case_summaries_by_name[case.name] for case in cases]

    report_path = args.output_dir / "stage1_fair_report.json"
    report["gpu_state_after_all"] = query_gpu_state()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        f"rounds={args.rounds} warmup={args.warmup} iters={args.iters}",
        f"gpu_before={report['gpu_state_before_all']}",
        f"gpu_after_preheat={report['gpu_state_after_preheat']}",
        "",
    ]
    if preheat_report.get("enabled"):
        status = "hot" if preheat_report.get("reached_hot_clock") else "not_hot_enough"
        lines.append(
            "preheat="
            f"{preheat_report['heater_case']} status={status} "
            f"target_floor={preheat_report['target_floor_mhz']}MHz "
            f"attempts={len(preheat_report['attempts'])}"
        )
        lines.append("")
    for case_summary in report["cases"]:
        mini_summary = case_summary["mini"]["summary_ns"]
        triton_summary = case_summary["triton"]["summary_ns"]
        mini_gflops_summary = case_summary["mini"]["summary_gflops"]
        triton_gflops_summary = case_summary["triton"]["summary_gflops"]
        lines.append(f"===== {case_summary['shape']} =====")
        lines.append(
            "mini median_ns="
            f"{mini_summary['median']:.2f} "
            f"(mean={mini_summary['mean']:.2f}, min={mini_summary['min']:.2f}, "
            f"max={mini_summary['max']:.2f}, spread={mini_summary['spread_pct']:.2f}%)"
        )
        lines.append(
            "triton median_ns="
            f"{triton_summary['median']:.2f} "
            f"(mean={triton_summary['mean']:.2f}, min={triton_summary['min']:.2f}, "
            f"max={triton_summary['max']:.2f}, spread={triton_summary['spread_pct']:.2f}%)"
        )
        lines.append(
            "mini median_gflops="
            f"{mini_gflops_summary['median']:.2f} "
            f"triton median_gflops={triton_gflops_summary['median']:.2f}"
        )
        lines.append(case_summary["comparison"]["median_ns_delta"])
        lines.append(
            "mini_vs_triton="
            f"{case_summary['comparison']['mini_over_triton_median_pct']:.2f}%"
        )
        lines.append(
            "gpu_case="
            f"before={case_summary['gpu_state_before_case']} "
            f"after={case_summary['gpu_state_after_case']}"
        )
        if case_summary["case_preheat"]:
            case_preheat = case_summary["case_preheat"]
            lines.append(
                "case_preheat="
                f"{case_preheat['heater_case']} "
                f"reached_hot_clock={case_preheat['reached_hot_clock']} "
                f"attempts={len(case_preheat['attempts'])}"
            )
        measurement = case_summary["measurement"]
        if measurement["unstable"]:
            lines.append(
                "measurement=unstable "
                f"(threshold={measurement['unstable_spread_threshold']:.2f}%, "
                f"total_rounds={measurement['total_rounds_run']})"
            )
        elif measurement["extra_rounds_run"] > 0:
            lines.append(
                "measurement=restabilized "
                f"(total_rounds={measurement['total_rounds_run']})"
            )
        lines.append("")

    text_path = args.output_dir / "stage1_fair_report.txt"
    text_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"json_report={report_path}")
    print(f"text_report={text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
