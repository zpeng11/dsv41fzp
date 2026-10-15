"""Verify effective compile flags per target from compile_commands.json (P0009, P0010).

Strict numeric flags must reach the targets that link df_strict_fp and must not
leak into unrelated targets; no target may carry fast-math or a global -march.
Profile `default` covers cpu-debug/cpu-release, where no target is sanitized;
profile `quant-sanitize` covers cpu-asan, where only the quant targets are built
and each must carry the D0007 sanitizer flags.
"""

import argparse
import json
import shlex
import sys

STRICT = ["-fno-fast-math", "-ffp-contract=off"]
FORBIDDEN_EVERYWHERE = ["-ffast-math", "-Ofast", "-funsafe-math-optimizations"]

SANITIZE = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]

# profile -> target -> (flags that must be present, flags that must be absent)
PROFILES = {
    "default": {
        "bb_smoke": (["-std=c++20"], STRICT + ["-mfma"]),
        "bb_smoke_test": (["-std=c++20"], STRICT + ["-mfma"]),
        "bb_version": (["-std=c++20"], STRICT + ["-mfma"]),
        "bb_fp_env": (["-std=c++20"] + STRICT, ["-mfma"]),
        "bb_contract_strict": (["-std=c++20", "-O2", "-mfma"] + STRICT, ["-ffp-contract=fast"]),
        "bb_contract_fast": (["-std=c++20", "-O2", "-mfma", "-ffp-contract=fast"], STRICT),
        "df_quant": (["-std=c++20"] + STRICT, ["-mfma", "-fno-exceptions"]),
        "df_quant_exec": (["-std=c++20"], STRICT + ["-mfma", "-fno-exceptions"]),
        "df_linear_consumer": (["-std=c++20"], STRICT + ["-mfma", "-fno-exceptions"]),
        "df_swiglu_consumer": (["-std=c++20"], STRICT + ["-mfma", "-fno-exceptions"]),
    },
    "quant-sanitize": {
        "df_quant": (["-std=c++20"] + STRICT + SANITIZE, ["-mfma", "-fno-exceptions"]),
        "df_quant_exec": (["-std=c++20"] + SANITIZE, STRICT + ["-mfma", "-fno-exceptions"]),
        "df_linear_consumer": (["-std=c++20"] + SANITIZE, STRICT + ["-mfma", "-fno-exceptions"]),
        "df_swiglu_consumer": (["-std=c++20"] + SANITIZE, STRICT + ["-mfma", "-fno-exceptions"]),
    },
}


def target_of(entry):
    marker = "CMakeFiles/"
    out = entry.get("output", "")
    if marker in out:
        return out.split(marker, 1)[1].split(".dir/", 1)[0]
    return None


def last_opt_level(flags):
    levels = [f for f in flags if f.startswith("-O")]
    return levels[-1] if levels else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-commands", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="default")
    args = ap.parse_args()
    expect = PROFILES[args.profile]

    with open(args.compile_commands, encoding="utf-8") as f:
        entries = json.load(f)

    ok = True
    seen = set()
    for entry in entries:
        target = target_of(entry)
        flags = entry.get("arguments") or shlex.split(entry["command"])
        name = f"{target}:{entry['file'].rsplit('/', 1)[-1]}"

        bad = [x for x in flags if x in FORBIDDEN_EVERYWHERE or x.startswith("-march=")]
        if args.profile == "default":
            bad += [x for x in flags if x.startswith("-fsanitize")]
        if bad:
            ok = False
            print(f"FAIL {name}: forbidden flags {bad}")

        if target not in expect:
            continue
        seen.add(target)
        present, absent = expect[target]
        missing = [x for x in present if x not in flags]
        leaked = [x for x in absent if x in flags]
        if target.startswith("bb_contract_") and last_opt_level(flags) != "-O2":
            missing.append("-O2 as the effective optimization level")
        if missing or leaked:
            ok = False
            print(f"FAIL {name}: missing={missing} unexpected={leaked}")
        else:
            print(f"PASS {name}")

    for target in sorted(set(expect) - seen):
        ok = False
        print(f"FAIL {target}: no compile command found")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
