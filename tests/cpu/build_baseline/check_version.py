"""Compare the built version metadata with VERSION and the current Git work tree (P0009).

A mismatch means the binary is stale relative to the tree: rebuild before testing.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def git(git_exe, source_dir, *args):
    return subprocess.run([git_exe, "-C", source_dir, *args], check=True,
                          capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--git", required=True)
    args = ap.parse_args()

    out = subprocess.run([args.exe], check=True, capture_output=True, text=True).stdout
    got = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)

    base = Path(args.source_dir, "VERSION").read_text().strip()
    sha = git(args.git, args.source_dir, "rev-parse", "--verify", "HEAD").strip()
    dirty = git(args.git, args.source_dir, "status", "--porcelain", "--untracked-files=normal") != ""
    want = {
        "base": base,
        "sha": sha,
        "dirty": "1" if dirty else "0",
        "display": base + ("+dirty" if dirty else ""),
    }

    ok = True
    for key, value in want.items():
        status = "PASS" if got.get(key) == value else "FAIL"
        ok = ok and status == "PASS"
        print(f"{status} {key}: built={got.get(key)!r} tree={value!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
