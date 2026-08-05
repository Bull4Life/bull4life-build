"""pex_launcher — the PEX one-click bootstrap (Model A, Cludia 0547 + engine 2024 crux).

WHY a launcher and not a frozen onefile of the app: PEX updates itself by SWAPPING .py source and re-execing a real
Python interpreter (node/pex_app_update_client.check_and_apply + node/pex_join_out.auto_update). A frozen Nuitka onefile
runs baked bytecode and ignores swapped .py, and re-execs ITSELF unchanged -> a permanent update-loop-that-never-updates
= Master's exact "I fell out of the world" bug, baked in forever. So the one-click exe is a STABLE SHELL that carries a
real, file-based CPython + the canonical source tree + the tiny deps, materializes them into a writable per-user app dir,
and launches `pythonw -m node.pex_join_out --gui` against that editable tree. From then on the app's OWN update client
self-heals on every future consensus fingerprint bump WITHOUT rebuilding this exe. The launcher is rebuilt ONLY if the
shell itself changes (rare); consensus bumps never touch it.

This module is compiled per-OS by Nuitka onefile (NO cross-compile: pex.exe on Windows, pex-linux on Ubuntu). Three data
payloads are baked in at build time (see .github/workflows/pex-build.yml):
  * runtime/   -> a relocatable CPython (python-build-standalone) with cryptography + certifi (+ pywebview) preinstalled
  * pexsrc/    -> the PEX node source tree pulled from /node_bundle, fp-gated to CANONICAL_FP at build
  * pexsrc.fp  -> the canonical consensus fingerprint the baked tree hashes to

Durability discipline (mirrors reject_non_finite_state / the point-update load-guards, now at the packaging layer):
  * First run / missing tree -> extract the BAKED pexsrc (always valid, works with NO network).
  * Every launch -> integrity-gate the app-dir tree: recompute consensus_fingerprint() and compare to the fp recorded
    for the currently-materialized tree. On mismatch/corruption/poison -> try a fresh /node_bundle pull (self-heal
    FORWARD to live), and if that's unreachable, re-extract the baked pexsrc (clean canonical). NEVER stranded on a
    corrupt/poisoned local copy; NEVER a hard fail when offline.
The launcher never edits the source and never touches consensus; it only bootstraps + integrity-gates + launches.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

BUNDLE_URL = os.environ.get("PEX_BUNDLE_URL", "https://compute.bull4life.com/node_bundle")
APP_MODULE = "node.pex_join_out"          # THE ONE PROGRAM: boots node, pull-joins, auto-updates, opens the GUI window
APP_ARGS = ["--gui"]
FP_ONELINER = "from node.consensus_fingerprint import consensus_fingerprint; print(consensus_fingerprint())"


# ---------------------------------------------------------------------------- payload + app dir
def _payload_dir() -> Path:
    """Where Nuitka onefile unpacked our baked data (runtime/, pexsrc/, pexsrc.fp). Next to this module at runtime."""
    for cand in (getattr(sys, "_MEIPASS", None), os.path.dirname(os.path.abspath(__file__))):
        if cand and (Path(cand) / "pexsrc.fp").exists():
            return Path(cand)
    # fall back to the module dir even if the marker check failed (best effort)
    return Path(os.path.dirname(os.path.abspath(__file__)))


def _app_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    d = Path(base) / "PEX"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log(app: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    try:
        with open(app / "launcher.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# ---------------------------------------------------------------------------- runtime python
def _runtime_python(app: Path, *, windowless: bool = False) -> Path | None:
    """The relocatable CPython we materialized. python-build-standalone extracts to a `python/` dir; we bake that as
    runtime/. Windows: runtime/python.exe (+ pythonw.exe). Linux: runtime/bin/python3."""
    if os.name == "nt":
        names = (["pythonw.exe", "python.exe"] if windowless else ["python.exe"])
        for n in names:
            p = app / "runtime" / n
            if p.exists():
                return p
    else:
        p = app / "runtime" / "bin" / "python3"
        if p.exists():
            return p
        p = app / "runtime" / "bin" / "python"
        if p.exists():
            return p
    return None


def _ensure_runtime(app: Path, payload: Path) -> None:
    if _runtime_python(app) is None:
        src = payload / "runtime"
        dst = app / "runtime"
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        # restore the exec bit Nuitka's data packing may drop (POSIX)
        if os.name != "nt":
            for sub in ("bin/python3", "bin/python3.12", "bin/python"):
                f = dst / sub
                if f.exists():
                    try:
                        f.chmod(0o755)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------- source tree
def _compute_fp(app: Path, srcdir: Path) -> str | None:
    """Recompute the tree's consensus fingerprint with the runtime python (cwd=srcdir), the same one-liner
    pex_self_update uses. Returns the fp string or None if the tree can't even be imported (treat as corrupt)."""
    py = _runtime_python(app)
    if py is None or not (srcdir / "node" / "consensus_fingerprint.py").exists():
        return None
    try:
        out = subprocess.run(
            [str(py), "-c", FP_ONELINER], cwd=str(srcdir), capture_output=True, text=True, timeout=120
        )
        fp = (out.stdout or "").strip().splitlines()[-1].strip() if out.stdout.strip() else ""
        return fp if len(fp) == 64 and all(c in "0123456789abcdef" for c in fp) else None
    except Exception:
        return None


def _extract_tar(data: bytes, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tgz") as tf:
        tf.write(data)
        tmp = tf.name
    try:
        with tarfile.open(tmp, "r:*") as tar:
            # path-safe extract (no absolute/.. members)
            for m in tar.getmembers():
                p = (dst / m.name).resolve()
                if not str(p).startswith(str(dst.resolve())):
                    raise RuntimeError(f"unsafe tar member {m.name}")
            tar.extractall(dst)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _try_download(url: str, timeout: float = 30.0) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pex-launcher/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _materialize_source(app: Path, payload: Path, *, prefer_fresh: bool) -> Path:
    """Put a valid source tree at app/src. prefer_fresh=True (corruption recovery) tries /node_bundle first to heal
    FORWARD to live; else/offline falls back to the baked canonical pexsrc. First run uses the baked tree (offline-safe).
    Records the resulting tree's fp in app/src.fp. Atomic swap via a temp dir."""
    srcdir = app / "src"
    tmp = app / "src.new"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    materialized_from = None
    if prefer_fresh:
        data = _try_download(BUNDLE_URL)
        if data:
            try:
                _extract_tar(data, tmp)
                if (tmp / "node" / "consensus_fingerprint.py").exists():
                    materialized_from = "node_bundle(live)"
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
    if materialized_from is None:
        shutil.copytree(payload / "pexsrc", tmp)     # baked canonical (always valid, no network)
        materialized_from = "baked(canonical)"

    # atomic-ish swap
    if srcdir.exists():
        old = app / f"src.old.{int(time.time())}"
        try:
            srcdir.rename(old)
        except Exception:
            shutil.rmtree(srcdir, ignore_errors=True)
            old = None
    tmp.rename(srcdir)
    if os.name != "nt":
        # clear the exec bit noise; nothing in src needs it
        pass
    fp = _compute_fp(app, srcdir)
    try:
        (app / "src.fp").write_text((fp or "") + "\n", encoding="utf-8")
    except Exception:
        pass
    # best-effort cleanup of any prior src.old.* (keep one for safety is unnecessary here)
    for stale in app.glob("src.old.*"):
        shutil.rmtree(stale, ignore_errors=True)
    _log(app, f"materialized source from {materialized_from}; fp={(fp or 'UNKNOWN')[:16]}...")
    return srcdir


def _ensure_source(app: Path, payload: Path) -> Path:
    srcdir = app / "src"
    expected = ""
    try:
        expected = (payload / "pexsrc.fp").read_text(encoding="utf-8").strip()
    except Exception:
        pass

    if not (srcdir / "node" / "pex_join_out.py").exists():
        _log(app, "no source tree present -> extracting baked canonical (offline-safe first run)")
        return _materialize_source(app, payload, prefer_fresh=False)

    # integrity gate: does the on-disk tree still hash to the fp we recorded for it?
    recorded = ""
    try:
        recorded = (app / "src.fp").read_text(encoding="utf-8").strip()
    except Exception:
        pass
    cur = _compute_fp(app, srcdir)
    if cur is None:
        _log(app, "local source tree does not import (corrupt/poisoned) -> re-materializing")
        return _materialize_source(app, payload, prefer_fresh=True)
    if recorded and cur != recorded:
        _log(app, f"local tree fp {cur[:16]}... != recorded {recorded[:16]}... (tampered/partial) -> re-materializing")
        return _materialize_source(app, payload, prefer_fresh=True)
    if not recorded:
        # no record yet (e.g. upgraded launcher) -> stamp it
        try:
            (app / "src.fp").write_text(cur + "\n", encoding="utf-8")
        except Exception:
            pass
    _log(app, f"source tree OK; fp={cur[:16]}... (expected baked {expected[:16]}...) — app auto-update keeps it current")
    return srcdir


# ---------------------------------------------------------------------------- launch
def _launch(app: Path, srcdir: Path) -> int:
    py = _runtime_python(app, windowless=True) or _runtime_python(app)
    if py is None:
        _log(app, "FATAL: no runtime python materialized")
        return 3
    env = dict(os.environ)
    env["PEX_APP_DIR"] = str(app)
    env["PYTHONUTF8"] = "1"
    cmd = [str(py), "-m", APP_MODULE, *APP_ARGS]
    _log(app, f"launching: {' '.join(cmd)} (cwd={srcdir})")
    if os.name == "nt":
        # hand off to the windowless python and let this shell exit — the app owns its own native window
        flags = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
        subprocess.Popen(cmd, cwd=str(srcdir), env=env, creationflags=flags, close_fds=True)
        return 0
    # POSIX: replace the launcher process with the app
    os.chdir(str(srcdir))
    os.execve(str(py), cmd, env)
    return 0  # unreachable


# ---------------------------------------------------------------------------- entry
def run(selfcheck: bool = False) -> int:
    app = _app_dir()
    payload = _payload_dir()
    _log(app, f"pex launcher start (payload={payload}, app={app}, selfcheck={selfcheck})")
    _ensure_runtime(app, payload)
    srcdir = _ensure_source(app, payload)
    if selfcheck:
        fp = _compute_fp(app, srcdir)
        expected = (payload / "pexsrc.fp").read_text(encoding="utf-8").strip()
        ok = (fp == expected) and bool(expected)
        verdict = "PASS" if ok else "FAIL"
        _log(app, f"SELFCHECK fp={fp} expected={expected} -> {verdict}")
        print(f"SELFCHECK {verdict} fp={fp}")
        # a windowless (GUI-subsystem) exe can't print to the CI shell -> also drop a file the workflow reads
        try:
            (app / "selfcheck.result").write_text(f"{verdict} fp={fp}\n", encoding="utf-8")
        except Exception:
            pass
        return 0 if ok else 1
    return _launch(app, srcdir)


def _fatal_notify(msg: str) -> None:
    """Never fail silently on Master's #1: a windowless launcher that dies must still say so."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "PEX launcher", 0x10)  # MB_ICONERROR
        except Exception:
            pass


def main() -> int:
    try:
        return run(selfcheck=("--selfcheck" in sys.argv[1:]))
    except Exception as e:
        app = _app_dir()
        _log(app, f"FATAL {type(e).__name__}: {e}")
        _fatal_notify(f"PEX couldn't start: {type(e).__name__}: {e}\n\nLog: {app / 'launcher.log'}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
