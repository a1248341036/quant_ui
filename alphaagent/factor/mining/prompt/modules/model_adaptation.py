# -*- coding: utf-8 -*-
"""模块 · model_adaptation：按挖掘模型注入行为约束（S4）。

背景：项目此前**没有任何按模型调行为的机制**（`infra/provider_compat.py` 只做
schema 归一化与断连重试）。而同一份 prompt 喂给不同模型，实测失效模式完全不同
（画像见 OpenViking `viking://resources/alphaagent/model_behaviors/`）：

- **DeepSeek 系**（deepseek-*）：吞吐高但同族扎堆（dup_dead_end 11.7% vs Qwen 4.4%），
  30 个 promising 却 0 次 submit——"过线继续变异"而非"过线即提交"；
- **Gemini 系**（gemini-*）：换手率盲区——19/28 提交因换手被拒（1d label + weekly
  调仓错配），高轮次 run 重复评估率高；
- **Qwen / 其它**：无专门画像，注入空。

实现：``render(ctx)`` 在 ``ctx.model_name`` 为空时返回空串——默认装配（黄金基线
测试 4 份 case 均不传 model_name）**逐字节不变**；只有显式传入模型名才追加对应约束。
约束为**事实型行为提示**（不是说教）：点出该模型的历史失效模式与对应动作。
"""

from alphaagent.factor.mining.prompt.prompt_modules import PromptContext

NAME = "model_adaptation"
TITLE = "模型适配行为提示"
ORDER = 138  # 放在 facet_focus(135) 之后、extra_instructions 之前
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"

# model 名前缀 → 行为约束块（小写匹配；未命中 → 空串）
_ADAPTATIONS: dict[str, str] = {
    "deepseek": (
        "### 本模型历史行为画像（DeepSeek 系，来自对比实验）\n"
        "- 探索吞吐高但**同族扎堆**：历史 run 重复死路拦截率 11.7%（对比 Qwen 4.4%）。\n"
        "  连续多轮在同一信号族变体上打转且未见 new verdict 时，应主动换族/换数据面。\n"
        "- **高 promising 低提交**：历史 run 30 个 promising 因子 0 次 submit_factor——\n"
        "  训练评估通过海选线（promising）后应直接走 submit_factor 让系统裁决盲测/正交/精筛，"
        "而不是继续堆同族变体。"
    ),
    "gemini": (
        "### 本模型历史行为画像（Gemini 系，来自基线 run）\n"
        "- **换手率盲区**：历史 19/28 提交因换手率被拒（根因：1d 快信号配 weekly 调仓，"
        "换手必然超硬门 0.50）。提交前先自查 label 持有期与调仓频率匹配：\n"
        "  1d label（快信号）只应配 daily 调仓或构造中慢长窗结构；weekly/monthly 调仓"
        "必须用中慢信号（背离/筹码峰距离/资金积累），使信号兼具灵敏度与延续性。\n"
        "- 高轮次 run 重复评估率高：同表达式/同结构指纹不要重复提交评估。"
    ),
}

# 未命中前缀的模型给一个通用轻提示（避免完全空置；可选）
_GENERIC = (
    "### 模型行为提示\n"
    "- 训练评估通过海选线（promising）后直接 submit_factor 走系统裁决，"
    "不要堆同族变体；提交前自查换手与调仓频率匹配。"
)


def render(ctx: PromptContext) -> str:
    name = str(getattr(ctx, "model_name", "") or "").strip()
    if not name:
        return ""
    key = name.lower()
    for prefix, text in _ADAPTATIONS.items():
        if prefix in key:
            return text
    return _GENERIC


def enabled(ctx: PromptContext) -> bool:
    """仅当显式传入模型名时启用；空串 = 模块关闭（默认装配与黄金基线不变）。"""
    return bool(str(getattr(ctx, "model_name", "") or "").strip())
