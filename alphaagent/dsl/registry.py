"""DSL 算子注册与命名空间合并：内置 + 扩展模块。"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

_BUILTIN_MODULE = "alphaagent.dsl.core.operators"
_OPERATOR_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class OperatorConflictError(ValueError):
    """扩展算子与内置或其他扩展同名。"""


@dataclass(frozen=True)
class OperatorMeta:
    name: str
    module: str
    default_column: str = "close"
    test_param: int = 20


_REGISTRY: dict[str, OperatorMeta] = {}


def register_operator(
    name: str | None = None,
    *,
    default_column: str = "close",
    test_param: int = 20,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：登记扩展算子元数据（供测试/文档发现）。"""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        op_name = name or fn.__name__
        if not _OPERATOR_NAME_RE.match(op_name):
            raise ValueError(f"算子名须为大写下划线风格: {op_name!r}")
        _REGISTRY[op_name] = OperatorMeta(
            name=op_name,
            module=fn.__module__,
            default_column=default_column,
            test_param=test_param,
        )
        return fn

    return deco


def iter_registered_operators(*, module: str | None = None) -> Iterator[OperatorMeta]:
    for meta in _REGISTRY.values():
        if module is None or meta.module == module:
            yield meta


def is_operator_name(name: str) -> bool:
    return bool(_OPERATOR_NAME_RE.match(name))


def collect_module_operators(mod: ModuleType) -> dict[str, Callable[..., Any]]:
    """收集模块内公开的大写 callable（DSL 算子命名约定）。"""
    out: dict[str, Callable[..., Any]] = {}
    for name in dir(mod):
        if name.startswith("_") or not is_operator_name(name):
            continue
        obj = getattr(mod, name)
        if callable(obj):
            out[name] = obj
    return out


def load_operator_module(path: str) -> ModuleType:
    """加载 Python 模块：``pkg.mod`` 或 ``/abs/path/to/file.py``。"""
    if path.endswith(".py") or "/" in path or "\\" in path:
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"扩展算子文件不存在: {path}")
        stem = file_path.stem
        if "/workspace/sandbox/extensions/" in file_path.as_posix():
            mod_name = f"workspace.sandbox.extensions.{stem}"
        elif "/workspace/dsl/extensions/" in file_path.as_posix():
            mod_name = f"workspace.dsl.extensions.{stem}"
        elif "/sandbox/extensions/" in file_path.as_posix():
            mod_name = f"sandbox.extensions.{stem}"
        else:
            mod_name = f"alphaagent.dsl.extensions.{stem}"
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载扩展模块: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(path)


def extensions_from_env() -> list[str]:
    raw = os.environ.get("ALPHAAGENT_DSL_EXTENSIONS", "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def discover_extension_modules(
    package: str = "alphaagent.dsl.extensions",
    *,
    skip_private: bool = True,
) -> list[str]:
    """扫描包目录下 ``.py`` 模块（默认跳过 ``_`` 开头文件如 ``_example.py``）。"""
    try:
        mod = importlib.import_module(package)
    except ImportError:
        return []
    pkg_path = getattr(mod, "__path__", None)
    if not pkg_path:
        return []
    root = Path(pkg_path[0])
    if not root.is_dir():
        return []
    prefix = package + "."
    out: list[str] = []
    for path in sorted(root.glob("*.py")):
        stem = path.stem
        if stem == "__init__":
            continue
        if skip_private and stem.startswith("_"):
            continue
        out.append(prefix + stem)
    return out


def discover_sandbox_extensions(
    sandbox_dir: str | Path | None = None,
    *,
    skip_private: bool = True,
) -> list[str]:
    """扫描 ``workspace/sandbox/extensions/*.py``，返回绝对路径供 ``load_operator_module`` 使用。"""
    raw = sandbox_dir if sandbox_dir is not None else os.environ.get(
        "ALPHAAGENT_SANDBOX_EXTENSIONS_DIR", "workspace/sandbox/extensions"
    )
    root = Path(raw)
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.is_dir():
        return []
    out: list[str] = []
    for path in sorted(root.glob("*.py")):
        if path.stem == "__init__":
            continue
        if skip_private and path.stem.startswith("_"):
            continue
        out.append(str(path.resolve()))
    return out


def resolve_extension_modules(
    operator_modules: Sequence[str] | None = None,
    *,
    auto_discover: bool | None = None,
) -> list[str]:
    """合并显式模块、环境变量、包内扩展、sandbox 扩展（后者可覆盖同名包内算子）。"""
    if auto_discover is None:
        auto_discover = os.environ.get("ALPHAAGENT_DSL_AUTO_DISCOVER", "0").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    seen: set[str] = set()
    merged: list[str] = []
    for mod in list(operator_modules or ()) + extensions_from_env():
        if mod and mod not in seen:
            seen.add(mod)
            merged.append(mod)
    if auto_discover:
        for mod in discover_extension_modules():
            if mod not in seen:
                seen.add(mod)
                merged.append(mod)
        for path in discover_sandbox_extensions():
            if path not in seen:
                seen.add(path)
                merged.append(path)
    return merged


def _is_sandbox_module_path(mod_path: str) -> bool:
    p = mod_path.replace("\\", "/")
    return (
        "/workspace/sandbox/extensions/" in p
        or "/workspace/dsl/extensions/" in p
        or "/sandbox/extensions/" in p
    )


def build_operator_namespace(
    *,
    include_builtin: bool = True,
    extension_modules: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
    allow_override: bool = False,
) -> dict[str, Any]:
    """合并内置与扩展算子为 evaluator 命名空间。"""
    ns: dict[str, Any] = {}

    if include_builtin:
        builtin = importlib.import_module(_BUILTIN_MODULE)
        ns.update(collect_module_operators(builtin))

    for mod_path in extension_modules:
        mod = load_operator_module(mod_path)
        sandbox = _is_sandbox_module_path(mod_path)
        for name, fn in collect_module_operators(mod).items():
            if name in ns and not allow_override and not sandbox:
                raise OperatorConflictError(
                    f"算子 {name!r} 已存在（模块 {mod_path!r} 与已有定义冲突）"
                )
            ns[name] = fn

    if extra:
        for name, obj in extra.items():
            if name in ns and not allow_override:
                raise OperatorConflictError(f"算子 {name!r} 已存在，extra_namespace 冲突")
            ns[name] = obj

    return ns
