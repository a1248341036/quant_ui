"""Shim: actual implementation moved to agent/loop.py"""
from alphaagent.factor.mining.agent.loop import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.loop as _m
    return getattr(_m, name)
