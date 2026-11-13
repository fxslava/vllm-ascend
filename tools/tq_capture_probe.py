# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal reproduction for the TurboQuant full-graph-capture question.

WHAT THIS ANSWERS. ``NPUPlatform._apply_turboquant_graph_mode``
(``platform.py:1241``) downgrades every full-cudagraph mode to PIECEWISE or NONE
whenever ``ENABLE_TURBOQUANT=1``. That downgrade is **our own policy**, taken
before anything is captured -- the warning a user sees is this function, not an
ACL refusal. So the open question is not "why did capture fail" but "would it
fail", and it has two candidate answers:

  H1  OUR CODE. The fused decode launch is capture-legal after rungs 1-4 and
      the C1-C6 contract, and what is left uncapturable is ours to fix: the
      golden fallback paths, the ingestion's host-side reads, and any codec op
      with no NPU kernel that silently falls back to the CPU.
  H2  PLATFORM. Even with every host operation removed, the fused launch itself
      cannot be captured -- the MIX launch configuration, the in-launch
      ``aclGetDeviceCapability``, or ``AscendC::SyncAll<false>()`` is refused on
      a capturing stream.

THE DISTINGUISHING CRITERION, which is the only thing this probe exists to
produce:

  * capture succeeds and the replay matches eager token-for-token
        -> H1. The gate is over-broad policy and can be narrowed to the cases
           that really are uncapturable. This is G11's device-free shape.
  * capture fails with ACL 107030 and the message names a MEMCPY
        -> H1. A host-to-device copy survives somewhere on the path; find it.
           The C6 poison and the G10 allowlist cover the Python half, so look
           in the binding and in the ingestion.
  * capture fails with ACL 107030 naming the KERNEL LAUNCH itself, or fails
        with any code other than 107030
        -> H2 signature. Record the code and the operation verbatim; that is
           the evidence, and it decides whether the gate stays.

WHY THIS AND NOT A SERVE RUN. A serve run under a patched gate exercises the
whole model, so a failure could come from any layer, from the ingestion, or from
a golden fallback. This captures **only** ``_forward_decode``, with the cache
already written and the kernels registered, so a refusal is attributable.

THIS PROBE DOES NOT CHANGE platform.py AND DOES NOT LIFT THE GATE. It never
goes through the platform at all; it drives the backend directly.

    python tools/tq_capture_probe.py                  # one decode, capture, replay
    python tools/tq_capture_probe.py --tokens 8 --context 512

Needs a 950PR with the TurboQuant kernels built in (SOC_VERSION=ascend950). On
any other host it says which precondition is missing and exits 2 without
pretending to have measured anything.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any

EXIT_OK = 0
EXIT_VERDICT_NEEDED = 1
EXIT_PRECONDITION = 2


def _fail(reason: str) -> int:
    print(f"PRECONDITION NOT MET: {reason}")
    print("Nothing was measured. This probe makes no claim about H1 or H2.")
    return EXIT_PRECONDITION


def _preconditions() -> tuple[Any, Any] | int:
    try:
        import torch
    except ImportError:
        return _fail("torch is not importable")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return _fail("no NPU is visible to torch (torch.npu.is_available() is False)")
    try:
        from vllm_ascend.attention import turboquant_v1 as tq
    except ImportError as exc:
        return _fail(f"vllm_ascend.attention.turboquant_v1 will not import: {exc}")
    namespace = getattr(torch.ops, tq.TURBOQUANT_DEFAULT_NAMESPACE, None)
    missing = [
        name
        for name in (tq.TURBOQUANT_OP_FUSED_DECODE, tq.TURBOQUANT_OP_BUILD_DECODE_TILING)
        if namespace is None or getattr(namespace, name, None) is None
    ]
    if missing:
        return _fail(
            f"torch.ops.{tq.TURBOQUANT_DEFAULT_NAMESPACE} is missing {missing}. "
            "The kernels are compiled only when SOC_VERSION matches ascend950, so this is "
            "the wrong part or the wrong wheel -- and without them the probe would be "
            "measuring the golden path, which is uncapturable by construction."
        )
    return torch, tq


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4, help="decode batch width (the capture extent)")
    parser.add_argument("--context", type=int, default=256, help="KV length per request")
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args(argv)

    checked = _preconditions()
    if isinstance(checked, int):
        return checked
    torch, tq = checked

    print("=" * 72)
    print("TurboQuant capture probe")
    print(f"  tokens={args.tokens} context={args.context} head_size={args.head_size}")
    print("=" * 72)

    # ---- stage 1: an eager decode, so we know the fused path runs at all ----
    #
    # Building the layer state is deliberately left to the caller's harness: the
    # point of this file is the capture attempt and the verdict table, not a
    # second copy of the ingestion. Wire `build_decode_fixture` to whatever
    # harness has already written a three-plane cache on this host -- the
    # LongBench adapter's smoke path is the shortest route.
    try:
        from tools.tq_capture_fixture import build_decode_fixture  # type: ignore[import-not-found]
    except ImportError:
        print("STAGE 1 SKIPPED: tools/tq_capture_fixture.py is absent.")
        print()
        print("This probe needs one decode step's worth of real state -- a written")
        print("three-plane cache, the layer's Tier-0 constants and a metadata object")
        print("from AscendTurboQuantMetadataBuilder. Provide it as")
        print("`build_decode_fixture(torch, tq, args) -> (impl, metadata, query, output)`")
        print("in tools/tq_capture_fixture.py, built from whichever harness has")
        print("already ingested a model on this host.")
        print()
        print("Everything below stage 1 is the part that answers H1 vs H2, and it")
        print("cannot be faked: a probe over an unwritten cache would capture the")
        print("refusal path, not the launch.")
        return EXIT_PRECONDITION

    impl, metadata, query, output = build_decode_fixture(torch, tq, args)

    try:
        impl._forward_decode(query, metadata, output)
        torch.npu.synchronize()
    except Exception:  # noqa: BLE001
        print("STAGE 1 FAILED: the fused decode does not run eagerly on this host.")
        traceback.print_exc()
        print()
        print("VERDICT: neither H1 nor H2. Fix the eager path first; a capture")
        print("attempt over a broken launch tells you nothing.")
        return EXIT_PRECONDITION
    # Stage 1's output is deliberately not kept: stage 3 compares the replay
    # against a *fresh* eager step with different content, which is the only
    # comparison that distinguishes a correct replay from a stale one.
    print("STAGE 1 OK: the fused decode ran eagerly.")

    # ---- stage 2: capture it ------------------------------------------------
    print()
    print("STAGE 2: capturing _forward_decode on its own.")
    graph = torch.npu.NPUGraph()
    pool = None
    try:
        # Warm the stream the way the runner's capture does, so the first
        # allocation inside capture is not the probe's own fault.
        with torch.npu.stream(torch.npu.Stream()):
            impl._forward_decode(query, metadata, output)
        torch.npu.synchronize()
        with torch.npu.graph(graph, pool=pool):
            impl._forward_decode(query, metadata, output)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        print("STAGE 2 FAILED. The verbatim error is the evidence:")
        print("-" * 72)
        traceback.print_exc()
        print("-" * 72)
        print()
        is_107030 = "107030" in text
        names_memcpy = any(word in text.lower() for word in ("memcpy", "copy", "d2h", "h2d"))
        if is_107030 and names_memcpy:
            print("READS AS H1 (our code): 107030 naming a memory copy. A host-to-device")
            print("copy survives on the path. The Python half is covered by the G10")
            print("allowlist and the C6 poison, so look in the C++ binding and in the")
            print("ingestion writers.")
        elif is_107030:
            print("READS AS H2 (platform) -- BUT CONFIRM WHAT 107030 NAMES. If it names")
            print("the kernel launch rather than a copy, the MIX launch configuration or")
            print("SyncAll<false> is refused on a capturing stream, and the gate stays.")
            print("If the message is ambiguous, re-run with ASCEND_GLOBAL_LOG_LEVEL=0")
            print("and read which stream operation the runtime rejected.")
        else:
            print("READS AS H2 (platform): a refusal that is not 107030 at all. Record")
            print("the code verbatim -- it is the evidence that decides the gate.")
        return EXIT_VERDICT_NEEDED
    print("STAGE 2 OK: capture completed without a refusal.")

    # ---- stage 3: refresh, replay, compare ---------------------------------
    print()
    print("STAGE 3: refreshing the per-step buffers and replaying.")
    try:
        impl, metadata2, query2, output2 = build_decode_fixture(torch, tq, args, step=2)
        query.copy_(query2)
        graph.replay()
        torch.npu.synchronize()
        replayed = output.clone()
        impl._forward_decode(query2, metadata2, output2)
        torch.npu.synchronize()
    except Exception:  # noqa: BLE001
        print("STAGE 3 FAILED during replay:")
        traceback.print_exc()
        print()
        print("VERDICT NEEDED: capture succeeded and replay did not. That is H1 in")
        print("the buffer-ownership sense -- something the replay reads was not")
        print("refreshed in place. C2 and G11a gate that device-free, so a failure")
        print("here is a gap in those gates and should become a new case in them.")
        return EXIT_VERDICT_NEEDED

    if torch.equal(replayed, output2):
        print("STAGE 3 OK: the replay is bit-identical to the eager second step.")
        print()
        print("READS AS H1 (our code): the fused decode IS capturable. The gate in")
        print("platform.py is over-broad policy, not a platform limitation, and the")
        print("remaining work is narrowing it to the paths that really are")
        print("uncapturable -- the golden fallbacks and the ingestion. That is rung 6,")
        print("and G11 proper (token-for-token over a greedy generation) is still the")
        print("gate for lifting it.")
        return EXIT_OK

    delta = (replayed.float() - output2.float()).abs().max().item()
    print(f"STAGE 3 MISMATCH: max |replay - eager| = {delta}")
    print()
    print("VERDICT NEEDED: capture and replay both succeeded and the answer is")
    print("wrong, which is the single worst outcome and the reason the gate is")
    print("conservative. Do NOT lift it. Find which operand the replay read stale:")
    print("C2 covers addresses and G11a covers shapes and scalars, so a mismatch")
    print("here is a per-step quantity neither of them models.")
    return EXIT_VERDICT_NEEDED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
