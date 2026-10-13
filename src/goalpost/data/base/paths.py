import os
from pathlib import Path

ARTIFACTS = Path(os.environ.get("GOALPOST_TRANSITION_DATA", "artifacts/transitions")).expanduser().resolve()
EXTRACTED = ARTIFACTS / "extracted"
IMPUTED = ARTIFACTS / "imputed"
