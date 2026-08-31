"""Shim: actual implementation moved to agent/agentscope_tools.py"""
from alphaagent.factor.mining.agent.agentscope_tools import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.agentscope_tools as _m
    return getattr(_m, name)
