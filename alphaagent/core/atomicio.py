"""向后兼容层：re-export core.atomicio 内容（已无内部使用方，保留供外部脚本引用）。"""
from core.atomicio import atomic_write_bytes, atomic_write_text  # noqa: F401
