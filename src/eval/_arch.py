"""Register the pruning repo's custom architectures with transformers.

Materialized pruned checkpoints carry ``model_type: custom-llama3`` (a
per-layer flexible Llama-3 defined in the pruning repo). That arch is not
self-contained in the checkpoint (no ``auto_map``/modeling file), so
``trust_remote_code`` cannot help — it is registered only as a side effect
of importing ``language_utils`` from the pruning repo. Base / CPT models are
plain ``llama`` and don't need this; the call is a no-op for them.
"""

import os
import sys

_DEFAULT_PRUNING_REPO = "/leonardo/home/userexternal/adiecidu/scripts/pruning"


def register_pruning_arch() -> None:
    """Import the pruning repo so ``custom-llama3`` resolves via AutoModel.

    Safe to call unconditionally: no-op if the repo is unavailable or already
    imported. Path is taken from ``$PRUNING_REPO`` (falls back to the known
    location). Importing ``language_utils`` only defines classes and runs the
    module-level ``CONFIG_MAPPING.register`` calls — it does not init CUDA or
    a distributed process group.
    """
    repo = os.environ.get("PRUNING_REPO", _DEFAULT_PRUNING_REPO)
    if not repo or not os.path.isdir(repo):
        return
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import language_utils  # noqa: F401  (registers custom-llama3 on import)
    except Exception as e:  # pragma: no cover - registration is best-effort
        print(f"[eval] custom-arch registration skipped: {e.__class__.__name__}: {e}")
