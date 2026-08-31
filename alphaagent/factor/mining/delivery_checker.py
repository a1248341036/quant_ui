"""Shim: actual implementation moved to delivery/delivery_checker.py"""
from alphaagent.factor.mining.delivery.delivery_checker import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.delivery.delivery_checker as _m
    return getattr(_m, name)
