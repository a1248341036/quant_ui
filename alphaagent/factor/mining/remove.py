"""Shim: actual implementation moved to infra/remove.py"""
from alphaagent.factor.mining.infra.remove import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.infra.remove as _m
    return getattr(_m, name)
