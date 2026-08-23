"""`cne serve` — the read-only lake dashboard."""

from cnequity.serve.app import create_app
from cnequity.serve.lake import LakeView

__all__ = ["LakeView", "create_app"]
