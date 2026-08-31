"""Shim: actual implementation moved to agent/preflight.py"""
from alphaagent.factor.mining.agent.preflight import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.preflight as _m
    return getattr(_m, name)
