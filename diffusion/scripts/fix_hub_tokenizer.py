#!/usr/bin/env python3
"""Patch tokenizer_config.json on an already-uploaded HF repo (all revisions).

Why: tokenizers saved with transformers>=5 write
``"tokenizer_class": "TokenizersBackend"``. The official BabyLM eval pipeline
pins transformers 4.51.x, which doesn't know that class, so
``AutoProcessor.from_pretrained(...)`` fails with::

    ValueError: Unrecognized processing class in <repo>. Can't instantiate a
    processor, a tokenizer, an image processor or a feature extractor ...

This rewrites ``tokenizer_class`` to the version-portable
``"PreTrainedTokenizerFast"`` (understood by both 4.x and 5.x) and re-uploads
ONLY that tiny file to ``main`` and every ``chck_*`` branch -- the multi-hundred-MB
weights are left untouched, so it runs in seconds.

Usage:
    export HF_TOKEN="hf_..."
    python scripts/fix_hub_tokenizer.py --repo-id amosluna/babylm-2026-strict-small-mdlm-seed42
    # patch several repos at once:
    python scripts/fix_hub_tokenizer.py \
        --repo-id amosluna/babylm-2026-strict-small-mdlm-seed42 \
        --repo-id amosluna/babylm-2026-strict-small-mdlm-seed1337
    # add --dry-run to see what would change.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

LOG = logging.getLogger("fix_hub_tokenizer")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

PORTABLE_CLASS = "PreTrainedTokenizerFast"


def load_and_normalize(cfg_text: str) -> tuple[str, bool]:
    """Return (patched_json_text, changed?)."""
    cfg = json.loads(cfg_text)
    changed = cfg.get("tokenizer_class") != PORTABLE_CLASS
    cfg["tokenizer_class"] = PORTABLE_CLASS
    cfg.pop("backend", None)  # tokenizers>=5-only key
    return json.dumps(cfg, indent=2), changed


def patch_repo(api, repo_id: str, local_cfg: Path | None, dry_run: bool) -> None:
    from huggingface_hub import hf_hub_download

    # 1) Source of truth for the file's contents.
    if local_cfg and local_cfg.is_file():
        src_text = local_cfg.read_text()
        LOG.info("[%s] using local tokenizer_config.json: %s", repo_id, local_cfg)
    else:
        path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", revision="main")
        src_text = Path(path).read_text()
        LOG.info("[%s] downloaded tokenizer_config.json from main", repo_id)

    patched_text, changed = load_and_normalize(src_text)
    LOG.info("[%s] tokenizer_class -> %s (changed=%s)", repo_id, PORTABLE_CLASS, changed)
    if not changed and not local_cfg:
        LOG.info("[%s] already portable on main; nothing to do.", repo_id)
        return

    # 2) Every revision that should carry the tokenizer: main + chck_* branches.
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    branches = [b.name for b in refs.branches]
    revisions = ["main"] + sorted(b for b in branches if b != "main")
    LOG.info("[%s] revisions to patch (%d): %s", repo_id, len(revisions), ", ".join(revisions))

    if dry_run:
        LOG.info("[%s] DRY-RUN: would upload patched tokenizer_config.json to each revision.", repo_id)
        return

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "tokenizer_config.json"
        fp.write_text(patched_text)
        for rev in revisions:
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo="tokenizer_config.json",
                repo_id=repo_id,
                repo_type="model",
                revision=rev,
                commit_message="fix: portable tokenizer_class (PreTrainedTokenizerFast) for eval pipeline",
            )
            LOG.info("[%s]   patched %s", repo_id, rev)
    LOG.info("[%s] done. https://huggingface.co/%s", repo_id, repo_id)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, action="append", help="Repeatable.")
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="Local tokenizer dir to take tokenizer_config.json from "
                        "(otherwise it's downloaded from each repo's main).")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        LOG.error("pip install huggingface_hub"); return 2

    if not args.hf_token and not args.dry_run:
        LOG.error("No HF token (set --hf-token or env HF_TOKEN), or pass --dry-run."); return 2
    api = HfApi(token=args.hf_token) if not args.dry_run else HfApi()

    local_cfg = (args.tokenizer_dir / "tokenizer_config.json") if args.tokenizer_dir else None
    for repo_id in args.repo_id:
        patch_repo(api, repo_id, local_cfg, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
