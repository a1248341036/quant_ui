"""Shim: actual implementation moved to infra/config.py"""
from alphaagent.factor.mining.infra.config import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.infra.config as _m
    return getattr(_m, name)
