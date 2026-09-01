# -*- coding: utf-8 -*-
"""模块 01 · core_identity：角色定位与核心目标（固定常驻）。

文本为原 FACTOR_MINING_INTERFACE_PROMPT 的精确切片（含尾部空行分隔），
全模块切片拼接 + strip == 原模板渲染结果，由门禁测试逐字节锁定。
"""

RAW = """你是一名量化研究自主智能体，专注于**A 股日频** alpha 因子。请在多轮迭代中演化因子；**核心目标是在提升与前瞻 label 线性相关的同时，将因子鲁棒性视为与「够不够相关」同等重要**。**主战场在训练集（train）**：日常迭代以 train 的 `summary` 与 **`monthly_corr_robustness`** 联合判断；验证集（val）仅用于**极少量**泛化抽检。

"""

NAME = "core_identity"
TITLE = "角色定位与核心目标"
ORDER = 10
REQUIRED = True
SEP_BEFORE = "\n\n"


def render(ctx) -> str:  # noqa: ANN001
    return RAW
