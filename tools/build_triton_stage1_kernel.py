#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_bk,
    stride_cm,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EXACT_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M > 1:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k0 = tl.arange(0, BLOCK_K)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if EXACT_TILE:
        acc = tl.load(c_ptrs)
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        acc = tl.load(c_ptrs, mask=c_mask, other=0.0)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k0
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]
        if EXACT_TILE:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = acc + tl.dot(a, b, out_dtype=tl.float32)

    if EXACT_TILE:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--block-m", type=int, required=True)
    parser.add_argument("--block-n", type=int, required=True)
    parser.add_argument("--block-k", type=int, required=True)
    parser.add_argument("--num-warps", type=int, required=True)
    parser.add_argument("--num-stages", type=int, required=True)
    parser.add_argument("--group-m", type=int, default=1)
    parser.add_argument(
        "--exact-tile",
        choices=("true", "false"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_stem = args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    exact_tile = args.exact_tile == "true"

    kernel = matmul_kernel.warmup(
        torch.float16,
        torch.float16,
        torch.float32,
        args.k,
        args.n,
        args.n,
        args.m,
        args.n,
        args.k,
        grid=(triton.cdiv(args.m, args.block_m) * triton.cdiv(args.n, args.block_n),),
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        BLOCK_K=args.block_k,
        GROUP_M=args.group_m,
        EXACT_TILE=exact_tile,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )

    cubin_path = output_stem.with_suffix(".cubin")
    ptx_path = output_stem.with_suffix(".ptx")
    meta_path = output_stem.with_suffix(".json")
    cubin_path.write_bytes(kernel.asm["cubin"])
    ptx_path.write_text(kernel.asm["ptx"])
    meta_path.write_text(
        json.dumps(
            {
                "kernel_name": kernel.name,
                "shared": int(kernel.metadata.shared),
                "num_warps": int(kernel.metadata.num_warps),
                "exact_tile": exact_tile,
                "group_m": int(args.group_m),
            },
            indent=2,
        )
    )
    print(json.dumps({"cubin": str(cubin_path), "meta": str(meta_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
