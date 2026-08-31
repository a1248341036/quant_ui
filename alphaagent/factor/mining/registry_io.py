"""Shim: actual implementation moved to infra/registry_io.py"""
from alphaagent.factor.mining.infra.registry_io import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.infra.registry_io as _m
    return getattr(_m, name)
