#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
#
# CPU micro-benchmark comparing Gloo and UCC backends.
#
# Single rank:
#   MASTER_ADDR=localhost MASTER_PORT=0 WORLD_SIZE=1 RANK=0 \
#     python comms/torchcomms/tests/perf/py/gloo_vs_ucc_bench.py
#
# Multi-rank:
#   torchrun --nproc_per_node=8 comms/torchcomms/tests/perf/py/gloo_vs_ucc_bench.py

import os
import time

import torch
import torchcomms


WARMUP = 3
MIN_ITERS = 5
TARGET_SECS = 1.0  # run each (backend, collective, size) for ~1s
DTYPE = torch.float32
ELEMENT_SIZE = 4

SIZES = [4, 64, 1024, 16384, 262144, 1048576, 4194304, 16777216, 67108864, 104857600]
COLLECTIVES = ["all_reduce", "all_gather", "broadcast", "barrier"]


def bench_collective(comm, name, size_bytes):
    n_elem = max(size_bytes // ELEMENT_SIZE, 1)
    rank = comm.get_rank()
    n_ranks = comm.get_size()
    tensor = torch.ones(n_elem, dtype=DTYPE) * float(rank + 1)

    if name == "all_reduce":
        def op():
            comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
    elif name == "all_gather":
        out = [torch.zeros(n_elem, dtype=DTYPE) for _ in range(n_ranks)]
        def op():
            comm.all_gather(out, tensor, False)
    elif name == "broadcast":
        def op():
            comm.broadcast(tensor, 0, False)
    elif name == "barrier":
        def op():
            comm.barrier(False)
    else:
        return None

    for _ in range(WARMUP):
        op()
    comm.barrier(False)

    # Time-based: run until TARGET_SECS elapsed, at least MIN_ITERS
    iters = 0
    start = time.perf_counter()
    while True:
        op()
        iters += 1
        elapsed = time.perf_counter() - start
        if iters >= MIN_ITERS and elapsed >= TARGET_SECS:
            break

    return (elapsed / iters) * 1e6, iters


def human_size(b):
    if b >= 1048576:
        return f"{b / 1048576:.0f}MB"
    if b >= 1024:
        return f"{b / 1024:.0f}KB"
    return f"{b}B"


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    comms = {}
    for backend in ["gloo", "ucc"]:
        try:
            comms[backend] = torchcomms.new_comm(
                backend, torch.device("cpu"), f"bench_{backend}"
            )
        except Exception:
            if rank == 0:
                print(f"WARNING: {backend} backend not available")

    backends = list(comms.keys())
    if len(backends) < 2:
        if rank == 0:
            print("Need both gloo and ucc backends to compare.")
        return

    if rank == 0:
        print(f"Backends: {', '.join(backends)}  Ranks: {world}")
        print(f"Warmup: {WARMUP}  Target: {TARGET_SECS}s/test  Dtype: {DTYPE}")
        print()

    for coll_name in COLLECTIVES:
        sizes = [0] if coll_name == "barrier" else SIZES

        if rank == 0:
            print(f"=== {coll_name} ===")
            print(
                f"{'Size':<10}{'Iters':>8}"
                + "".join(f"{b:>12}(us)" for b in backends)
                + "   Ratio (gloo/ucc)"
            )
            print("-" * (18 + 15 * len(backends) + 20))

        for size in sizes:
            results = {}
            iters_used = 0
            for backend in backends:
                avg_us, iters = bench_collective(comms[backend], coll_name, size)
                results[backend] = avg_us
                iters_used = max(iters_used, iters)

            if rank == 0:
                size_str = "N/A" if coll_name == "barrier" else human_size(size)
                line = f"{size_str:<10}{iters_used:>8}"
                for b in backends:
                    line += f"{results[b]:>14.2f} "
                if results.get("ucc", 0) > 0:
                    ratio = results["gloo"] / results["ucc"]
                    line += f"  {ratio:>6.2f}x"
                print(line)

        if rank == 0:
            print()

    for c in comms.values():
        c.finalize()


if __name__ == "__main__":
    main()
