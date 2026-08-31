"""Shim: actual implementation moved to infra/console.py"""
from alphaagent.factor.mining.infra.console import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.infra.console as _m
    return getattr(_m, name)
