"""Shim: actual implementation moved to prompt/seed_factors.py"""
from alphaagent.factor.mining.prompt.seed_factors import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.prompt.seed_factors as _m
    return getattr(_m, name)
