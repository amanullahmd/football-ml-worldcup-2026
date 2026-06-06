"""THM JupyterHub client — upload files and run notebooks on the GPU server.

Usage:
    python tools/jh_client.py whoami
    python tools/jh_client.py upload <local_path> <remote_path>
    python tools/jh_client.py upload-dir <local_dir> <remote_dir>
    python tools/jh_client.py run-notebook <remote_notebook_path>
    python tools/jh_client.py download <remote_path> <local_path>

The token in .env must include scope `access:servers!server=<user>/`. See
https://jupyterhub.readthedocs.io/en/stable/howto/rest.html for the API.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

HUB = os.environ["THM_HUB_URL"].rstrip("/")
USER = os.environ["THM_USER"]
TOKEN = os.environ["THM_API_TOKEN"]

HUB_API = f"{HUB}/hub/api"
USER_API = f"{HUB}/user/{USER}/api"
HEADERS = {"Authorization": f"token {TOKEN}"}


def _check(r: requests.Response) -> dict:
    if not r.ok:
        print(f"[{r.status_code}] {r.request.method} {r.url}", file=sys.stderr)
        print(r.text[:500], file=sys.stderr)
        r.raise_for_status()
    return r.json() if r.text else {}


def whoami():
    info = _check(requests.get(f"{HUB_API}/user", headers=HEADERS, timeout=30))
    print(json.dumps(info, indent=2))


def ensure_server_running():
    """Start the single-user server if it isn't already."""
    info = _check(requests.get(f"{HUB_API}/users/{USER}", headers=HEADERS, timeout=30))
    server = info.get("servers", {}).get("", {})
    if server.get("ready"):
        return
    print(f"Starting server for {USER}...")
    r = requests.post(f"{HUB_API}/users/{USER}/server", headers=HEADERS, timeout=60)
    if r.status_code not in (201, 202, 400):
        r.raise_for_status()
    # Poll until ready
    import time
    for _ in range(60):
        time.sleep(2)
        info = _check(requests.get(f"{HUB_API}/users/{USER}", headers=HEADERS, timeout=30))
        if info.get("servers", {}).get("", {}).get("ready"):
            print("Server is ready.")
            return
    raise TimeoutError("Server did not become ready within 120s.")


def upload(local: str, remote: str):
    """Upload a single file to the user's Jupyter server via Contents API."""
    ensure_server_running()
    p = Path(local)
    data = p.read_bytes()
    mime, _ = mimetypes.guess_type(p.name)
    is_text = (mime or "").startswith("text") or p.suffix in {".py", ".md", ".txt", ".csv", ".json", ".ipynb", ".yml", ".yaml", ".html", ".js", ".css"}
    if is_text:
        body = {"type": "file", "format": "text", "content": data.decode("utf-8", errors="replace")}
    else:
        body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode()}
    if p.suffix == ".ipynb":
        body["type"] = "notebook"
        body["format"] = "json"
        body["content"] = json.loads(data.decode("utf-8"))
    url = f"{USER_API}/contents/{remote.lstrip('/')}"
    r = requests.put(url, headers={**HEADERS, "Content-Type": "application/json"},
                     data=json.dumps(body), timeout=120)
    _check(r)
    print(f"uploaded {local}  ->  {remote}  ({len(data):,} bytes)")


def upload_dir(local_dir: str, remote_dir: str):
    ensure_server_running()
    root = Path(local_dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {local_dir}")
    # First create directories (Jupyter contents API auto-creates parents on PUT,
    # but creating them explicitly avoids surprises with empty dirs).
    skip_dirs = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules"}
    skip_files = {".env"}
    for src in root.rglob("*"):
        rel = src.relative_to(root).as_posix()
        if any(part in skip_dirs for part in src.parts):
            continue
        if src.name in skip_files:
            continue
        if src.is_dir():
            continue
        remote = f"{remote_dir.rstrip('/')}/{rel}"
        try:
            upload(str(src), remote)
        except Exception as e:
            print(f"  ! failed {src}: {e}", file=sys.stderr)


def download(remote: str, local: str):
    ensure_server_running()
    url = f"{USER_API}/contents/{remote.lstrip('/')}"
    info = _check(requests.get(url, headers=HEADERS, timeout=60))
    p = Path(local); p.parent.mkdir(parents=True, exist_ok=True)
    fmt = info.get("format")
    content = info.get("content")
    if fmt == "base64":
        p.write_bytes(base64.b64decode(content))
    elif fmt == "json":
        p.write_text(json.dumps(content, indent=1), encoding="utf-8")
    else:
        p.write_text(content or "", encoding="utf-8")
    print(f"downloaded {remote}  ->  {local}")


def run_notebook(remote_nb: str, timeout_sec: int = 1800):
    """Execute a notebook in-place on the JupyterHub server via the kernel API.

    Note: requires `nbclient`/`jupyter_client` on the server side. Most JupyterHubs
    ship with these. For very long-running jobs prefer to run as a script.
    """
    ensure_server_running()
    # Easier: PUT a small launcher script that runs nbconvert --execute, then call it.
    launcher = (
        "import subprocess, sys\\n"
        f"subprocess.check_call([sys.executable,'-m','jupyter','nbconvert','--to','notebook',"
        f"'--execute','--inplace','--ExecutePreprocessor.timeout={timeout_sec}','{remote_nb}'])\\n"
    )
    upload_inline("__run.py", launcher)
    print(f"Uploaded launcher. To execute, open a terminal on the JupyterHub server and run:\n"
          f"  python __run.py")


def upload_inline(remote: str, text: str):
    ensure_server_running()
    body = {"type": "file", "format": "text", "content": text}
    url = f"{USER_API}/contents/{remote.lstrip('/')}"
    _check(requests.put(url, headers={**HEADERS, "Content-Type": "application/json"},
                        data=json.dumps(body), timeout=60))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    cmd, *rest = args
    if cmd == "whoami":
        whoami()
    elif cmd == "upload" and len(rest) == 2:
        upload(rest[0], rest[1])
    elif cmd == "upload-dir" and len(rest) == 2:
        upload_dir(rest[0], rest[1])
    elif cmd == "download" and len(rest) == 2:
        download(rest[0], rest[1])
    elif cmd == "run-notebook" and len(rest) == 1:
        run_notebook(rest[0])
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
