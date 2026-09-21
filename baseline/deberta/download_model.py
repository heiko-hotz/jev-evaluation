"""Fetch a pinned public snapshot, verify upstream hashes, never load model code."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / "model"
REPO = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
WANTED = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "spm.model"}

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)

revision = "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"
info = get_json(f"https://huggingface.co/api/models/{REPO}/revision/{revision}?blobs=true")
ROOT.mkdir(exist_ok=True)
files = []
for entry in info["siblings"]:
    name = entry["rfilename"]
    if name not in WANTED:
        continue
    target = ROOT / name
    partial = ROOT / (name + ".part")
    sha256 = hashlib.sha256()
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{revision}/{name}", timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            sha256.update(chunk)
    size = partial.stat().st_size
    if "lfs" in entry:
        assert size == entry["lfs"]["size"], name
        assert sha256.hexdigest() == entry["lfs"]["sha256"], name
    else:
        blob = hashlib.sha1(f"blob {size}\0".encode() + partial.read_bytes()).hexdigest()
        assert blob == entry["blobId"], name
    partial.replace(target)
    files.append({"name": name, "size": size, "sha256": sha256.hexdigest()})
    print(f"Verified {name}: {size} bytes", flush=True)
assert {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= {x["name"] for x in files}
manifest = {"repo_id": REPO, "revision": revision, "files": files}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Pinned revision: {revision}", flush=True)
