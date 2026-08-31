"""Shim: actual implementation moved to infra/cli_stream.py"""
from alphaagent.factor.mining.infra.cli_stream import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.infra.cli_stream as _m
    return getattr(_m, name)
