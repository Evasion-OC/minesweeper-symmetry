"""Which code produced a result: the git commit, and whether the code differed from it."""

import subprocess
from pathlib import Path


def code_version(root: Path) -> dict:
    """The short commit of the repository at root, and whether exact_symmetry/ or scripts/ had
    changes not yet committed when the result was made."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    changed = bool(git("status", "--porcelain", "--", "exact_symmetry", "scripts"))
    return {"commit": commit or None, "code changed since that commit": changed}
