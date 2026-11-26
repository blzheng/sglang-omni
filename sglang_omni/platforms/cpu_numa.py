# SPDX-License-Identifier: Apache-2.0
"""NUMA placement for CPU stage worker processes.

CPU has no device index to place a stage with, so placement is the thread binding SGLang derives from SGLANG_CPU_OMP_THREADS_BIND. The binding unit is the OS process: stages sharing a process share its node, and a request that would split one process across nodes is rejected.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable, Mapping

import sglang.srt.distributed.bootstrap as sglang_bootstrap
from sglang.srt.utils.numa_utils import get_cpu_ids_by_node

from sglang_omni.utils.cpu import parse_cpu_list

logger = logging.getLogger(__name__)

THREADS_BIND_ENV = "SGLANG_CPU_OMP_THREADS_BIND"
NUMA_MAP_ENV = "SGLANG_OMNI_CPU_NUMA_MAP"
WIDEN_HOOK_MARKER = "omni_widen_hooked"

node_by_process_name: dict[str, int] = {}
node_assignment_lock = threading.Lock()


class NumaPlacementConflict(ValueError):
    """Two stages in one process were asked for different NUMA nodes."""


def parse_numa_map(raw: str | None) -> dict[str, int]:
    """Parse "thinker=0,talker=1" into {"thinker": 0, "talker": 1}."""
    mapping: dict[str, int] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name or not value:
            raise ValueError(
                f"{NUMA_MAP_ENV} entries must look like '<stage>=<node>', got {item!r}"
            )
        try:
            node = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{NUMA_MAP_ENV} node for stage {name!r} must be an integer, got {value!r}"
            ) from exc
        if node < 0:
            raise ValueError(
                f"{NUMA_MAP_ENV} node for stage {name!r} must be >= 0, got {node}"
            )
        mapping[name] = node
    return mapping


def resolve_process_node(
    stage_names: list[str],
    *,
    node_count: int,
    user_map: Mapping[str, int],
    auto_index: int,
) -> int:
    """Pick the node for one process: a user entry decides, else spread round-robin."""
    requested = {name: user_map[name] for name in stage_names if name in user_map}
    for name, node in requested.items():
        if node >= node_count:
            raise ValueError(
                f"{NUMA_MAP_ENV} asks for node {node} for stage {name!r}, but "
                f"this host has only {node_count} NUMA node(s)."
            )
    if not requested:
        return auto_index % node_count
    if len(set(requested.values())) > 1:
        raise NumaPlacementConflict(
            f"{NUMA_MAP_ENV} asks to split one process across NUMA nodes: "
            + ", ".join(
                f"{name}->node {node}" for name, node in sorted(requested.items())
            )
            + ". Stages sharing a process share its binding."
        )
    return next(iter(requested.values()))


def reset_assignment() -> None:
    """Forget recorded placements, for tests and for a changed topology or map."""
    with node_assignment_lock:
        node_by_process_name.clear()


def threads_bind_env(
    *,
    process_name: str,
    stage_names: list[str],
    tp_size: int,
    gpu_ids: Iterable[int | None] = (),
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the SGLANG_CPU_OMP_THREADS_BIND override for one process.

    SGLang indexes the bind list by gpu_id, so the list repeats the node once per index up to the largest gpu_id.
    """
    source_env = env if env is not None else os.environ
    if source_env.get(THREADS_BIND_ENV):
        return {}
    if tp_size > 1:
        logger.warning(
            "Process %s has tp_size=%d; skipping NUMA placement and leaving "
            "SGLang's per-rank default.",
            process_name,
            tp_size,
        )
        return {}

    try:
        cpu_ids_per_node = get_cpu_ids_by_node()
    except Exception as exc:
        logger.warning(
            "Could not read the NUMA topology (%s); leaving %s unset.",
            exc,
            THREADS_BIND_ENV,
        )
        return {}
    if not cpu_ids_per_node:
        return {}

    user_map = parse_numa_map(source_env.get(NUMA_MAP_ENV))
    with node_assignment_lock:
        node = node_by_process_name.get(process_name)
        if node is None:
            node = resolve_process_node(
                stage_names,
                node_count=len(cpu_ids_per_node),
                user_map=user_map,
                auto_index=len(node_by_process_name),
            )
            if node in node_by_process_name.values():
                logger.warning(
                    "Process %s shares NUMA node %d with another worker process; "
                    "their OMP threads are pinned to the same cores. Spread them with %s.",
                    process_name,
                    node,
                    NUMA_MAP_ENV,
                )
            node_by_process_name[process_name] = node
    logger.info(
        "Binding CPU process %s (stages: %s) to NUMA node %d of %d.",
        process_name,
        ", ".join(stage_names) or "<none>",
        node,
        len(cpu_ids_per_node),
    )
    entry_count = (
        max((gpu_id for gpu_id in gpu_ids if gpu_id is not None), default=0) + 1
    )
    return {THREADS_BIND_ENV: "|".join([cpu_ids_per_node[node]] * entry_count)}


def bound_core_count(env: Mapping[str, str]) -> int | None:
    """Cores in the first range of SGLANG_CPU_OMP_THREADS_BIND, if set.

    SGLang resizes the OMP pool to this range and ignores OMP_NUM_THREADS, so it is the real thread ceiling.
    """
    raw = env.get(THREADS_BIND_ENV, "").strip()
    if not raw or raw == "all":
        return None
    try:
        return len(parse_cpu_list(raw.split("|", 1)[0])) or None
    except ValueError:
        return None


def install_binding_hook() -> bool:
    """Widen the main thread's mask right after SGLang pins it to one core.

    init_cpu_threads_env pins the main thread to a single core, and every thread created afterwards inherits that mask. OMP workers keep their own per-core affinity, so widening the main thread to the bound range does not disturb them.
    """
    original = getattr(sglang_bootstrap, "_init_cpu_threads_env", None)
    if original is None or getattr(sglang_bootstrap, WIDEN_HOOK_MARKER, False):
        return False

    def widen_after_binding(
        *, local_omp_cpuid: str | list[int] | None, **kwargs
    ) -> None:
        original(local_omp_cpuid=local_omp_cpuid, **kwargs)
        if not local_omp_cpuid:
            return
        bound_cpus = (
            parse_cpu_list(local_omp_cpuid)
            if isinstance(local_omp_cpuid, str)
            else set(local_omp_cpuid)
        )
        try:
            os.sched_setaffinity(0, bound_cpus)
        except OSError as exc:
            logger.warning("Could not widen the CPU mask to the bound range (%s).", exc)

    sglang_bootstrap._init_cpu_threads_env = widen_after_binding
    setattr(sglang_bootstrap, WIDEN_HOOK_MARKER, True)
    return True
