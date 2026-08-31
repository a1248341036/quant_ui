"""Shim: actual implementation moved to infra/audit.py"""
from alphaagent.factor.mining.infra.audit import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.infra.audit as _m
    return getattr(_m, name)
