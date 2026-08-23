"""多行因子求值专用异常：MultiLineFactorEvalError 在解析、符号绑定、执行、求值各阶段携带 phase、生成代码片段、用户源行号与 traceback，供上层统一展示或记录。"""
from __future__ import annotations

from typing import List, Optional


class MultiLineFactorEvalError(Exception):
    """parse / symbol / exec / eval 失败时携带生成代码行号等信息。"""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        problem: Optional[str] = None,
        exception_type: Optional[str] = None,
        generated_code: Optional[str] = None,
        generated_line_no: Optional[int] = None,
        generated_line_text: Optional[str] = None,
        user_source: Optional[str] = None,
        user_line_no: Optional[int] = None,
        user_line_text: Optional[str] = None,
        traceback_text: Optional[str] = None,
    ):
        super().__init__(message)
        self.phase = phase
        self.problem = problem
        self.exception_type = exception_type
        self.generated_code = generated_code
        self.generated_line_no = generated_line_no
        self.generated_line_text = generated_line_text
        self.user_source = user_source
        self.user_line_no = user_line_no
        self.user_line_text = user_line_text
        self.traceback_text = traceback_text

    def __str__(self) -> str:
        head = self.args[0] if self.args else "MultiLineFactorEvalError"
        if not (self.generated_code and str(self.generated_code).strip()):
            return head
        err_line = self.generated_line_no
        body: List[str] = []
        for i, ln in enumerate(self.generated_code.strip().split("\n"), start=1):
            mark = "--> " if err_line is not None and i == err_line else "    "
            body.append(f"{mark}{i:4d} | {ln}")
        return head + "\n" + "\n".join(body)


if __name__ == "__main__":
    e = MultiLineFactorEvalError("demo", phase="parse", problem="示例")
    print(e)
