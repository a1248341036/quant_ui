"""Shim: actual implementation moved to agent/agentscope_run.py"""
from alphaagent.factor.mining.agent.agentscope_run import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.agentscope_run as _m
    return getattr(_m, name)
