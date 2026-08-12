"""Vocabulary adaptation: EXTEND a trained BPE with merges learned on a new corpus."""
from .run import extend, extended_path, remap_specials

__all__ = ["extend", "extended_path", "remap_specials"]
