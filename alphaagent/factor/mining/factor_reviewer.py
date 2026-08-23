"""Independent pre-submission reviewer for mined factors."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Callable

from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.workspace import LocalWorkspace

from alphaagent.factor.mining.cli_stream import MiningStreamObserver, stream_to_cli
from alphaagent.factor.mining.config import MiningConfig


REVIEW_PROMPT = """你是独立的量化因子评估子 Agent（FactorReviewer），职责是阻止伪创新、经典风险暴露改名和数据挖掘后的无效交付。

只审查用户给出的一个候选因子。重点检查：
1. 表达式是否只是教科书因子或其单调变换，例如规模、原始动量、短期反转、低波、流动性；
2. 它是否仅通过 LOG/NEG/RANK/CS_ZSCORE/CS_WINSORIZE 等保序或尺度变换包装已有信号；
3. train 到 val 是否明显衰减，统计显著性和经济量级是否足够；
4. 是否有明确且可检验的新增经济假设，而非只复述指标；
5. 是否应允许进入候选池。

对 `NEG(LOG($float_cap))` 一类仅由流通市值构成的因子，必须判定为 `reject`：这是经典小市值暴露，CS 标准化和 winsorize 不构成创新。

输出严格 JSON，不能使用 Markdown：
{"verdict":"approve|revise|reject","novelty":"high|medium|low","canonical_form":"一句话标准因子名称","reasons":["最多三条可审计原因"],"required_changes":["具体可执行的重构方向"]}
只有在结构性原创、验证稳定、且有清晰经济假设时才允许 approve。JSON 无法保证时用 reject。"""


def _expr_key(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def _classic_precheck(expr: str) -> dict[str, Any] | None:
    """Cheap hard stop for one-field size transforms before spending an LLM call."""
    refs = set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", expr or ""))
    time_or_cross_signal = re.search(r"\b(?:TS_|DELTA|SLOPE|CORR|COV|CHIP_)\w*\s*\(", expr or "", re.I)
    if refs == {"$float_cap"} and not time_or_cross_signal:
        return {
            "verdict": "reject",
            "novelty": "low",
            "canonical_form": "经典小市值 / 流通市值暴露",
            "reasons": [
                "表达式只使用 $float_cap，未引入独立信息源或时序结构。",
                "LOG、NEG、CS_WINSORIZE 与 CS_ZSCORE 只改变单调尺度或截面标准化，不改变小市值排序。",
            ],
            "required_changes": [
                "以市值只作为中性化或分组变量，并引入独立的量价、波动、筹码或基本面机制。",
            ],
            "source": "deterministic_precheck",
        }
    return None


class FactorReviewer:
    """A separate AgentScope agent with its own prompt and per-expression evidence."""

    def __init__(
        self,
        *,
        config: MiningConfig,
        api_key: str,
        base_url: str | None,
        extra_body: dict[str, Any] | None,
        workspace: LocalWorkspace,
        emit: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.base_url = base_url
        self.extra_body = extra_body
        self.workspace = workspace
        self.emit = emit
        self.evaluations: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.reviews: dict[str, dict[str, Any]] = {}
        self.current_turn = 0

    @property
    def policy(self) -> dict[str, Any]:
        return self.config.research_spec or {}

    @property
    def review_on(self) -> set[str]:
        return set(self.policy.get("review_policy", {}).get("review_on", ["validation", "pre_submit"]))

    def record_evaluation(self, split: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        expr = arguments.get("multi_line_expr")
        if isinstance(expr, str) and expr.strip() and result.get("ok", True):
            self.evaluations[_expr_key(expr)][split].append(result)

    def _model(self) -> OpenAIChatModel:
        params: dict[str, Any] = {"max_tokens": min(self.config.max_tokens, 2048), "parallel_tool_calls": False}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return OpenAIChatModel(
            credential=OpenAICredential(api_key=self.api_key, base_url=self.base_url),
            model=self.config.model,
            parameters=OpenAIChatModel.Parameters(**params),
            stream=True,
            extra_body=self.extra_body,
            client_kwargs={"default_headers": {"User-Agent": "quant-ui/1.0"}},
        )

    async def review(self, arguments: dict[str, Any], *, turn: int) -> dict[str, Any]:
        expr = str(arguments.get("multi_line_expr") or "")
        factor_name = str(arguments.get("factor_name") or "expr")
        key = _expr_key(expr)
        if key in self.reviews:
            return self.reviews[key]
        review_policy = self.policy.get("review_policy", {})
        precheck = _classic_precheck(expr) if review_policy.get("block_classic_transforms", True) else None
        if precheck is not None:
            self.emit("factor_review", {"turn": turn, "factor_name": factor_name, "multi_line_expr": expr, **precheck})
            self.reviews[key] = precheck
            return precheck

        metric_precheck = self._metric_precheck(expr)
        if metric_precheck is not None:
            self.emit("factor_review", {"turn": turn, "factor_name": factor_name, "multi_line_expr": expr, **metric_precheck})
            self.reviews[key] = metric_precheck
            return metric_precheck

        packet = {
            "factor_name": factor_name,
            "multi_line_expr": expr,
            "author_comment": arguments.get("comment", ""),
            "evaluations": self.evaluations.get(_expr_key(expr), {}),
        }
        self.emit("reviewer_start", {"turn": turn, "factor_name": factor_name})

        def review_emit(event: str, payload: dict[str, Any]) -> None:
            mapping = {
                "agent_thinking": "reviewer_thinking",
                "assistant_message": "reviewer_message",
                "usage": "reviewer_usage",
            }
            self.emit(mapping.get(event, event), {"turn": turn, "factor_name": factor_name, **payload})

        observer = MiningStreamObserver(emit=review_emit, turn=turn)
        agent = Agent(
            name="FactorReviewer",
            system_prompt=REVIEW_PROMPT,
            model=self._model(),
            offloader=self.workspace,
            react_config=ReActConfig(max_iters=1),
            state=AgentState(permission_context=PermissionContext(mode=PermissionMode.BYPASS)),
            context_config=ContextConfig(trigger_ratio=0.8, reserve_ratio=0.1),
        )
        await stream_to_cli(
            agent,
            UserMsg(name="researcher", content=json.dumps(packet, ensure_ascii=False, default=str)),
            show_thinking=False,
            auto_confirm=True,
            observer=observer,
            quiet=True,
        )
        raw = "\n".join(observer.completed_texts).strip()
        verdict = self._parse_verdict(raw)
        required_novelty = {"low": 0, "medium": 1, "high": 2}
        if required_novelty.get(str(verdict.get("novelty")), 0) < required_novelty.get(str(review_policy.get("minimum_novelty", "medium")), 1):
            verdict["verdict"] = "revise"
            verdict.setdefault("required_changes", []).append(
                f"当前新颖性低于本次 ResearchSpec 要求的 {review_policy.get('minimum_novelty', 'medium')}。"
            )
        self.emit("factor_review", {"turn": turn, "factor_name": factor_name, "multi_line_expr": expr, **verdict})
        self.reviews[key] = verdict
        return verdict

    def _metric_precheck(self, expr: str) -> dict[str, Any] | None:
        evidence = self.evaluations.get(_expr_key(expr), {})
        train = (evidence.get("train") or [])[-1:]
        val = (evidence.get("val") or [])[-1:]
        if not train or not val:
            return None
        train_summary = train[0].get("summary") if isinstance(train[0].get("summary"), dict) else {}
        val_summary = val[0].get("summary") if isinstance(val[0].get("summary"), dict) else {}
        policy = self.policy.get("evaluation_policy", {})
        train_ic = float(train_summary.get("ic") or 0)
        val_ic = float(val_summary.get("ic") or 0)
        train_icir = float(train_summary.get("icir") or 0)
        coverage = float(train_summary.get("factor_coverage") or 0)
        reasons: list[str] = []
        if abs(train_ic) < float(policy.get("min_train_abs_ic", 0.015)):
            reasons.append("训练集 abs(IC) 未达到 ResearchSpec 门槛。")
        if train_icir < float(policy.get("min_train_icir", 0.2)):
            reasons.append("训练集 ICIR 未达到 ResearchSpec 门槛。")
        if coverage < float(policy.get("min_train_coverage", 0.85)):
            reasons.append("训练集 Coverage 未达到 ResearchSpec 门槛。")
        if abs(val_ic) < float(policy.get("min_val_abs_ic", 0.01)):
            reasons.append("验证集 abs(IC) 未达到 ResearchSpec 门槛。")
        if abs(val_ic) / max(abs(train_ic), 1e-12) < float(policy.get("min_val_ic_retention_ratio", 0.5)):
            reasons.append("验证集相对训练集的 IC 保留比例不足。")
        if policy.get("require_sign_consistency", True) and train_ic * val_ic <= 0:
            reasons.append("训练集与验证集 IC 方向不一致。")
        for rule in val[0].get("rule_results", []) if isinstance(val[0].get("rule_results"), list) else []:
            if isinstance(rule, dict) and rule.get("passed") is False:
                reasons.append(
                    f"EvaluationProfile 规则未通过：{rule.get('metric')} {rule.get('op')} {rule.get('expected')}。"
                )
        if not reasons:
            return None
        return {
            "verdict": "revise",
            "novelty": "medium",
            "canonical_form": "统计验证未满足本次 ResearchSpec",
            "reasons": reasons,
            "required_changes": ["修改经济机制或信号结构后重新完成 train/val 评估。"],
            "source": "research_spec_metric_precheck",
        }

    @staticmethod
    def _parse_verdict(raw: str) -> dict[str, Any]:
        match = re.search(r"\{[\s\S]*\}", raw)
        try:
            value = json.loads(match.group(0) if match else raw)
        except (json.JSONDecodeError, AttributeError):
            value = {}
        if not isinstance(value, dict) or value.get("verdict") not in {"approve", "revise", "reject"}:
            return {
                "verdict": "reject",
                "novelty": "low",
                "canonical_form": "审查输出不可解析",
                "reasons": ["FactorReviewer 未返回可验证的结构性原创结论。"],
                "required_changes": ["修订表达式并重新完成 train/val 评估后再提交。"],
                "source": "reviewer_parse_guard",
                "raw": raw[:2000],
            }
        value.setdefault("novelty", "low")
        value.setdefault("canonical_form", "未分类")
        value.setdefault("reasons", [])
        value.setdefault("required_changes", [])
        value["source"] = "agentscope_factor_reviewer"
        return value
