"""Commit, push, wait for the build, and download the IPA -- in one command.

    python tools/ship_app.py "what changed"

Collapses the iteration loop into a single step. Without this it is: commit,
push, open the browser, find the run, wait, download the zip, unzip it, then
sideload. Six manual steps between writing a fix and testing it is enough
friction to discourage trying things.

Needs the GitHub CLI:  winget install GitHub.cli   (then: gh auth login)

Note that this pushes, rather than re-running the previous build. Re-running a
job rebuilds the commit it was originally triggered by, so it would produce a
fresh-looking IPA containing none of the new changes -- a confusing way to
conclude that a fix did not work.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = "ios.yml"
OUT_DIR = os.path.join(ROOT, "build")


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    printable = " ".join(cmd)
    print(f"  $ {printable}")
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True,
                          capture_output=capture)


def have_gh() -> bool:
    return shutil.which("gh") is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?", default="update app", help="commit message")
    ap.add_argument("--no-push", action="store_true", help="just fetch the latest build")
    args = ap.parse_args()

    if not have_gh():
        print("The GitHub CLI is not installed, so this cannot watch or download the build.")
        print("\n  winget install GitHub.cli")
        print("  gh auth login\n")
        print("Until then, do it by hand:")
        print("  1. git add -A && git commit -m \"...\" && git push")
        print("  2. GitHub > Actions > newest run (do NOT press 'Re-run all jobs',")
        print("     that rebuilds the old commit)")
        print("  3. download the artifact, unzip, sideload the .ipa")
        return 1

    if not args.no_push:
        status = run(["git", "status", "--porcelain"], capture=True).stdout.strip()
        if status:
            run(["git", "add", "-A"])
            run(["git", "commit", "-m", args.message])
        else:
            print("  nothing to commit")
        run(["git", "push"])

    print("\nwaiting for the build (3-6 minutes)...")
    # Give the run a moment to be created before asking about it.
    run(["gh", "run", "list", "--workflow", WORKFLOW, "--limit", "1"], check=False)
    result = run(["gh", "run", "watch", "--exit-status",
                  "--workflow", WORKFLOW], check=False)

    if result.returncode != 0:
        print("\nThe build FAILED. Show me the error with:")
        print("  gh run view --log-failed")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(OUT_DIR, "*.ipa")):
        os.remove(stale)
    run(["gh", "run", "download", "--name", "HandBridge-unsigned-ipa",
         "--dir", OUT_DIR], check=False)

    # The artifact may arrive as a zip depending on gh's version.
    for z in glob.glob(os.path.join(OUT_DIR, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(OUT_DIR)
        os.remove(z)

    ipas = glob.glob(os.path.join(OUT_DIR, "*.ipa"))
    if not ipas:
        print(f"\nBuild succeeded but no .ipa landed in {OUT_DIR}")
        return 1

    print(f"\n  ready: {ipas[0]}")
    print("\nDrag that into Sideloadly and press Start. It installs over the top --")
    print("no need to delete the app, and no need to re-Trust or re-enable")
    print("Developer Mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
