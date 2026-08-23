"""因子表达式语法解析：使用 pyparsing 将 DSL 文本解析为可 exec/eval 的 Python 代码串；变量、函数与中缀经 parse_symbol 与 evaluator 绑定为 DataFrame 列。比较运算 ``> < >= <= == !=`` 在双非数字操作数时改写为 ``GT``/``LT``/… 以便与 ``ADD`` 一样支持列名不同之单列面板。

启用 packrat 与较高递归深度以减轻深层嵌套调用的解析开销。"""
from pyparsing import Word, alphas, alphanums, infix_notation, opAssoc, one_of, Optional, DelimitedList, Forward, Group
from pyparsing import ParseException
from pyparsing import Regex, Combine, Literal
import sys
import re
import numpy as np
import keyword

# 引入pyparsing自带的cache功能
# 加快function_call = var + '(' + Optional(DelimitedList(expr)) + ')'这种嵌套式的pyparsing解析器
from pyparsing import ParserElement
ParserElement.enable_packrat()

sys.setrecursionlimit(4000)  # 设置更高的递归深度限制

# 定义基本元素
# 变量支持可选的频率后缀：`$close@60m` / `$close@1h` / `$close@1d`；不带 `@` 时默认当前面板（通常为 1m）。
var = (
    Combine(
        Optional(Literal("$"))
        + Word(alphas, alphanums + "_")
        + Optional(Literal("@") + Word(alphanums + "_"))
    )
).set_name("variable")
# var = Word(alphas, alphanums + "_")

# 定义数字的正则表达式
# 正则表达式匹配整数和小数，可以有正负号，以及科学计数法
number_pattern = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
number = Regex(number_pattern)

# 定义字符串字面量
# 支持单引号和双引号，包括转义字符
from pyparsing import QuotedString
string_literal = (
    QuotedString("'", esc_char='\\') | 
    QuotedString('"', esc_char='\\')
).set_parse_action(lambda t: f"'{t[0]}'")  # 统一转换为单引号形式

# 定义操作符
mul_div = one_of("* /", use_regex=True)
add_minus = one_of("+ -")
comparison_op = one_of("> < >= <= == !=")
logical_and = one_of("&& &")
logical_or = one_of("|| |")
conditional_op = ("?", ":")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 展平嵌套的 ParseResults 为字符串
def flatten_nested_tokens(tokens):
    # import pdb; pdb.set_trace()
    flattened = []
    for token in tokens:
        if isinstance(token, str):
            flattened.append(token)
        elif isinstance(token, list):
            flattened.extend(flatten_nested_tokens(token))
        else:  # ParseResults
            flattened.extend(flatten_nested_tokens(token.as_list()))
    return flattened




def parse_arith_op(s, loc, tokens):
    # tokens[0] 包含整个运算表达式的分解
    # 因为操作符定义为左结合，我们可以从左到右递归处理tokens列表
    def recursive_build_expression(tokens):
        if len(tokens) == 3:
            A, op, B = tokens
            # 构建表达式
            return build_expression(A, op, B)
        else:
            left = tokens[:-2]
            op = tokens[-2]
            right = tokens[-1]
            left_expr = recursive_build_expression(left)
            return build_expression(left_expr, op, right)
        
    def build_expression(A, op, B):
        A = ''.join(flatten_nested_tokens([A]))
        B = ''.join(flatten_nested_tokens([B]))
        A_is_number = is_number(A)
        B_is_number = is_number(B)
        
        ## 任意一个操作数都是数字
        if A_is_number or B_is_number:
            return f"{A}{op}{B}"
        
        ## 两个操作数都是pd变量
        else:
            if op == '+':
                return f'ADD({A}, {B})'
                # return f'np.add({A}, {B})'
            elif op == '-':
                return f'SUBTRACT({A}, {B})'
                # return f'np.subtract({A}, {B})'
            elif op == '*':
                return f'MULTIPLY({A}, {B})'
                # return f'np.multiply({A}, {B})'
            elif op == '/':
                return f'DIVIDE({A}, {B})'
                # return f'np.divide({A}, {B})'
            else:
                raise NotImplementedError(f'arith op \'{op}\' is not implemented')
            # 操作数2是BENCHMARKINDEX (pd.Series)，而操作数1不是BENCHMARKINDEX (pd.Series)的情况下，Series必须要放在第二操作数，否则会报错
            # if 'BENCHMARKINDEX' in A and 'BENCHMARKINDEX' not in B:
            #     if op == '+':
            #         return f'({B}).add({A}, axis=0)'
            #     elif op == '-':
            #         return f'(-1*{(B)}).add({A}, axis=0)'
            #     elif op == '*':
            #         return f'({B}).mul({A}, axis=0)'
            #     elif op == '/':
            #         return f'(1/{(B)}).mul({A}, axis=0)'
            #     else:
            #         raise NotImplementedError(f'arith op \'{op}\' is not implemented')
            # else:
            #     if op == '+':
            #         return f'({A}).add({B}, axis=0)'
            #     elif op == '-':
            #         return f'({A}).sub({B}, axis=0)'
            #     elif op == '*':
            #         return f'({A}).mul({B}, axis=0)'
            #     elif op == '/':
            #         return f'({A}).div({B}, axis=0)'
            #     else:
            #         raise NotImplementedError(f'arith op \'{op}\' is not implemented')
    
    return recursive_build_expression(tokens[0])


def parse_cmp_op(s, loc, tokens):
    """将比较重写成 ``LT``/``GT``/…，使双列名单列面板可逐元比较（与 ``ADD`` 一致）。"""

    def recursive_build_expression(tokens):
        if len(tokens) == 3:
            A, op, B = tokens
            return build_expression(A, op, B)
        left = tokens[:-2]
        op = tokens[-2]
        right = tokens[-1]
        left_expr = recursive_build_expression(left)
        return build_expression(left_expr, op, right)

    def build_expression(A, op, B):
        A = "".join(flatten_nested_tokens([A]))
        B = "".join(flatten_nested_tokens([B]))
        A_is_number = is_number(A)
        B_is_number = is_number(B)
        if A_is_number or B_is_number:
            return f"{A}{op}{B}"
        op_map = {
            ">": "GT",
            "<": "LT",
            ">=": "GE",
            "<=": "LE",
            "==": "EQ",
            "!=": "NE",
        }
        if op not in op_map:
            raise NotImplementedError(f"cmp op {op!r} is not implemented")
        return f"{op_map[op]}({A}, {B})"

    return recursive_build_expression(tokens[0])


# def parse_arith_op(s, loc, tokens):
#     A = ''.join(flatten_nested_tokens(tokens[0][0]))
#     op = ''.join(flatten_nested_tokens(tokens[0][1]))
#     B = ''.join(flatten_nested_tokens(tokens[0][2]))

#     # 检查操作数是否存在
#     if A == '' or B == '':
#         raise ParseException(s, loc, f"运算符 '{op}' 缺少操作数")
    
#     # 检查操作数是否为数字
#     A_is_number = is_number(A)
#     B_is_number = is_number(B)
    
#     # 根据操作数类型选择操作
    
#     ## 任意一个操作数都是数字
#     if A_is_number or B_is_number:
#         return f"{A}{op}{B}"
    
#     ## 两个操作数都是pd变量
#     else:
#         # 操作数2是BENCHMARKINDEX (pd.Series)，而操作数1不是BENCHMARKINDEX (pd.Series)的情况下，Series必须要放在第二操作数，否则会报错
#         if 'BENCHMARKINDEX' in A and 'BENCHMARKINDEX' not in B:
#             if op == '+':
#                 return f'({B}).add({A}, axis=0)'
#             elif op == '-':
#                 return f'(-1*{(B)}).add({A}, axis=0)'
#             elif op == '*':
#                 return f'({B}).mul({A}, axis=0)'
#             elif op == '/':
#                 return f'(1/{(B)}).mul({A}, axis=0)'
#             else:
#                 raise NotImplementedError(f'arith op \'{op}\' is not implemented')
#         else:
#             if op == '+':
#                 return f'({A}).add({B}, axis=0)'
#             elif op == '-':
#                 return f'({A}).sub({B}, axis=0)'
#             elif op == '*':
#                 return f'({A}).mul({B}, axis=0)'
#             elif op == '/':
#                 return f'({A}).div({B}, axis=0)'
#             else:
#                 raise NotImplementedError(f'arith op \'{op}\' is not implemented')


# 定义条件表达式的解析函数
def parse_conditional_expression(s, loc, tokens):
    A, B, C = tokens[0][0], tokens[0][2], tokens[0][4]
    # 将 A, B, C 转换为字符串
    A = ''.join(flatten_nested_tokens(A))
    B = ''.join(flatten_nested_tokens(B))
    C = ''.join(flatten_nested_tokens(C))

    # 将结果转换为带有datetime和instrument双重索引的Series
    return f"pd.Series(np.where({A}, {B}, {C}), index=($close).index)"

# 定义逻辑运算符的解析函数
def parse_logical_expression(s, loc, tokens):
    # tokens[0] 包含整个表达式的分解，可能包括嵌套的列表
    # 由于操作符定义为左结合，我们可以递归地展开tokens列表
    def recursive_flatten(tokens):
        if len(tokens) == 1:
            return ''.join(flatten_nested_tokens([tokens[0]]))
        else:
            left = tokens[0]
            operator = tokens[1]
            # right = tokens[2]
            left_str = ''.join(flatten_nested_tokens([left]))
            right_str = recursive_flatten(tokens[2:])
            if operator in ["||", "|"]: 
                return f"OR({left_str}, {right_str})"
                # return f"({left_str}) | ({right_str})"
            elif operator in ["&&", "&"]:
                return f"AND({left_str}, {right_str})"
                # return f"({left_str}) & ({right_str})"
    
    return recursive_flatten(tokens[0])


# 定义函数调用解析函数
def parse_function_call(s, loc, tokens):
    # unary_operator = tokens[0]
    function_name = tokens[0]
    arguments = tokens[2:-1] 
    # import pdb; pdb.set_trace()


    # 处理参数列表中的每个参数
    arguments_flat = []
    # import pdb; pdb.set_trace()
    for arg in arguments:
        if isinstance(arg, str):
            arguments_flat.append(arg)
        else:
            # 如果参数是嵌套的表达式或函数调用，递归处理
            flattened_arg = ''.join(flatten_nested_tokens(arg))
            arguments_flat.append(flattened_arg)
    arguments_str = ','.join(arguments_flat)
    return f"{function_name}({arguments_str})"

# 先定义一个 Forward 对象以便在定义 function_call 时引用
expr = Forward()

# 定义函数调用
## 定义可选的一元操作符，这里使用 one_of 选择器来匹配 "+" 或 "-"
unary_op = Optional(one_of("+ -")).set_parse_action(lambda t: t[0] if t else '')
function_call = var + '(' + Optional(DelimitedList(expr)) + ')'  # 使用 expr
function_call.set_parse_action(parse_function_call)
nested_expr = Group('(' + expr + ')')
# sign_var = unary_op + var

# 更新操作数，以包含函数调用和字符串字面量
operand =  Group(unary_op + (function_call | var | string_literal | number | nested_expr | expr))

# unary_operand = one_of("+ -") + operand
# unary_operand.set_parse_action(lambda tokens: ''.join(tokens))
# operand = (unary_operand | function_call | var | number )

# 使用新的 flatten_nested_tokens 函数
def parse_entire_expression(s, loc, tokens):
    # import pdb; pdb.set_trace()
    return ''.join(flatten_nested_tokens(tokens))


def check_for_invalid_operators(expression):
    valid_operators = {"(", ")", ",", "+", "-", "*", "/", "&&", "||", "&", "|", ">", "<", ">=", "<=", "==", "!=", "?", ":", ".", "\'", "\""}
    # 使用正则表达式查找所有的运算符，但排除字符串内容
    # 先移除字符串字面量，避免误判
    import re
    expr_without_strings = re.sub(r"'[^']*'", '', expression)  # 移除单引号字符串
    expr_without_strings = re.sub(r'"[^"]*"', '', expr_without_strings)  # 移除双引号字符串
    # 先移除形如 `$name@60m` 的频率后缀整体，避免 `@` 被误识别为非法运算符
    expr_without_strings = re.sub(r"\$?[A-Za-z_][A-Za-z0-9_]*@[A-Za-z0-9_]+", "", expr_without_strings)

    pattern = r'([+\-*/,><?:.]{2,})|([><=!&|^`~@#%\\;{}[\]"\'\\]+)' # ([|&=]{3,})|
    found_operators_tuples = re.findall(pattern, expr_without_strings)
    found_operators = [operator for tup in found_operators_tuples for operator in tup if operator]
    invalid_operators = set(found_operators) - valid_operators
    
    if invalid_operators:
        raise Exception(f"无效的运算符: \"{''.join(invalid_operators)}\"")


# 现在更新 expr 的定义
expr <<= infix_notation(operand, 
    [
        (mul_div, 2, opAssoc.LEFT, parse_arith_op),
        (add_minus, 2, opAssoc.LEFT, parse_arith_op),
        (comparison_op, 2, opAssoc.LEFT, parse_cmp_op),
        (logical_and, 2, opAssoc.LEFT, parse_logical_expression),
        (logical_or, 2, opAssoc.LEFT, parse_logical_expression),
        (conditional_op, 3, opAssoc.RIGHT, parse_conditional_expression)
    ])

    
def check_parentheses_balance(expr):
    if expr.count('(') != expr.count(')'):
        raise ParseException(f"表达式括号未闭合")

# 定义整个表达式的解析规则
expr.set_parse_action(parse_entire_expression) # check_parentheses_balance, 
# expr.setDebug()

def parse_expression(factor_expression, verbose=False):
    try:
        check_parentheses_balance(factor_expression)
        check_for_invalid_operators(factor_expression)
        if verbose:
            print("因子表达式: ", factor_expression)

        parsed_data_function = expr.parse_string(factor_expression)[0]
        return parsed_data_function
    except Exception as e:
        raise Exception(f"表达式`{factor_expression}`解析失败: {e}")



def dollar_ref_to_pyname(name: str) -> str:
    """把 DSL 中 `$field[@freq]` 形式的变量引用转成合法的 Python 标识符。

    规则：
    - `$close` -> `close`
    - `$adj_close@60m` -> `adj_close__60m`
    - 不带 `$` 也允许（某些内部再次规范化场景）。

    约定用双下划线作为频率分隔符；真实列名中出现 `__` 的极少见场景由调用方保证不冲突。
    """
    n = name.lstrip("$")
    if "@" in n:
        base, freq = n.split("@", 1)
        freq = re.sub(r"[^A-Za-z0-9_]", "_", freq)
        return f"{base}__{freq}"
    return n


def parse_symbol(expr, columns):
    keyword_map = {
        "TRUE": "True",
        "true": "True",
        "FALSE": "False",
        "false": "False",
        "NAN": "np.nan",
        "NaN": "np.nan",
        "nan": "np.nan",
        "NULL": "np.nan",
        "null": "np.nan",
    }

    # 先替换 `$列名[@freq]` -> 目标 Python 标识符；按长度降序处理，避免 `$x` 先匹配导致 `$x@60m` 破碎。
    # 列名以 `$` 前缀起头，不会与普通标识符冲突，可直接做字符串替换。
    col_items = [(col, dollar_ref_to_pyname(col)) for col in columns]
    col_items.sort(key=lambda kv: -len(kv[0]))
    for col, var_df in col_items:
        expr = expr.replace(col, var_df)

    # 关键字替换必须带词边界：否则形如 `dominant` / `tenant` / `null_bar` 这类普通标识符
    # 里的子串 `nan` / `null` 会被错误改写成 `np.nan`，在后续 exec 阶段导致 invalid syntax。
    # 字符串字面量内的关键字不替换，避免破坏 CAST(..., 'float64') 之类的参数。
    def _replace_keyword_safe(s: str, kw: str, val: str) -> str:
        pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(kw) + r"(?![A-Za-z0-9_])")
        out: list[str] = []
        i = 0
        in_sq = False
        in_dq = False
        escape = False
        while i < len(s):
            c = s[i]
            if escape:
                out.append(c)
                escape = False
                i += 1
                continue
            if in_sq or in_dq:
                if c == "\\":
                    out.append(c)
                    escape = True
                    i += 1
                    continue
                if c == "'" and in_sq:
                    in_sq = False
                elif c == '"' and in_dq:
                    in_dq = False
                out.append(c)
                i += 1
                continue
            if c == "'":
                in_sq = True
                out.append(c)
                i += 1
                continue
            if c == '"':
                in_dq = True
                out.append(c)
                i += 1
                continue
            # 在字符串外逐段替换：取到下一个引号为止的切片做 regex 替换。
            j = i
            while j < len(s) and s[j] not in ("'", '"'):
                j += 1
            chunk = s[i:j]
            out.append(pattern.sub(val, chunk))
            i = j
        return "".join(out)

    for kw, val in keyword_map.items():
        expr = _replace_keyword_safe(expr, kw, val)
    return expr


def _strip_hash_comment_from_line(line: str) -> str:
    """去掉从 `#` 到行尾的注释；字符串字面量（单/双引号，支持 `\\` 转义）内的 `#` 保留。"""
    out: list[str] = []
    i = 0
    in_squote = False
    in_dquote = False
    escape = False
    while i < len(line):
        c = line[i]
        if escape:
            out.append(c)
            escape = False
            i += 1
            continue
        if in_squote or in_dquote:
            if c == "\\":
                out.append(c)
                escape = True
                i += 1
                continue
            if c == "'" and in_squote:
                in_squote = False
            elif c == '"' and in_dquote:
                in_dquote = False
            out.append(c)
            i += 1
            continue
        if c == "'":
            in_squote = True
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_dquote = True
            out.append(c)
            i += 1
            continue
        if c == "#":
            break
        out.append(c)
        i += 1
    return "".join(out).strip()


def parse_multi_line_expression(multi_line_expr, verbose=False):
    """
    解析多行表达式，支持中间变量赋值
    
    例如：
    a=(RANK(SLOPE($amount/$volume, 5)) > 0.104)
    b=(RANK(SLOPE($amount/$volume, 90)) > 0.104)
    c=a&b?RANK(CS_NEUTRALIZE($ret, LOG($float_cap))) : nan
    
    返回一个 Python 代码字符串，使用原生变量存储中间结果，避免重复计算
    
    中间变量命名要求：
    1. 基本规则：
       - 必须以字母(a-z, A-Z)或下划线(_)开头
       - 后面可以跟字母、数字(0-9)或下划线
       - 正则表达式: [a-zA-Z_][a-zA-Z0-9_]*
    
    2. 应该避免的命名：
       - Python关键字: if, for, and, or, True, False, None, def, class 等
       - 函数库函数名: RANK, SLOPE, ADD, SUBTRACT, MULTIPLY, DIVIDE, AND, OR 等
       - 数据列名（带$前缀）: $amount, $volume 等（不会冲突，因为带$前缀）
    
    3. 推荐命名风格：
       - 小写字母: a, b, c, temp, result
       - 下划线分隔: my_var, temp_result
       - 避免使用大写（可能与函数名冲突）
    
    参数:
        multi_line_expr: 多行表达式字符串
        
    返回:
        Python 代码字符串，可以直接 eval() 执行

    支持 ``#`` 行注释：从 ``#`` 到行尾（字符串字面量内除外）；仅整行注释的行会被跳过。
    """
    lines = []
    for raw in multi_line_expr.strip().split("\n"):
        cleaned = _strip_hash_comment_from_line(raw)
        if cleaned:
            lines.append(cleaned)
    
    if not lines:
        raise Exception("表达式为空")
    
    # 识别变量赋值语句和最终表达式
    assignments = []  # 存储 (var_name, expression) 元组
    final_expr = None
    
    for i, line in enumerate(lines):
        # 检查是否是赋值语句 (var=...)
        # 使用正则表达式匹配 var=... 的模式
        # 注意：需要处理括号，因为表达式可能包含括号
        assignment_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        
        if assignment_match:
            var_name = assignment_match.group(1)
            expr_str = assignment_match.group(2).strip()
            # 移除表达式两端的括号（如果存在且括号匹配）
            # 注意：只有当整个表达式被括号包裹时才移除，例如 (A) 而不是 (A) & (B)
            if expr_str.startswith('(') and expr_str.endswith(')'):
                # 检查括号是否匹配，并且整个表达式被括号包裹
                # 从第二个字符开始查找，如果找到匹配的右括号在最后，说明整个表达式被括号包裹
                paren_count = 0
                should_remove = False
                for j, char in enumerate(expr_str):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            # 如果第一个左括号在位置0，匹配的右括号在最后，说明整个表达式被括号包裹
                            if j == len(expr_str) - 1:
                                should_remove = True
                            break
                if should_remove:
                    expr_str = expr_str[1:-1]
            
            # 如果是最后一行，且没有其他非赋值语句，则作为最终表达式
            if i == len(lines) - 1 and final_expr is None:
                # 最后一个赋值语句作为最终表达式
                final_expr = line  # 保留完整的赋值语句，后续会提取右侧表达式
            else:
                assignments.append((var_name, expr_str))
        else:
            # 如果不是赋值语句，则作为最终表达式
            if final_expr is None:
                final_expr = line
            else:
                # 如果已经有最终表达式，则追加（可能是多行表达式）
                final_expr += '\n' + line
    
    # 如果没有找到赋值语句，则整个表达式就是最终表达式
    if not assignments and final_expr:
        return parse_expression(final_expr.strip(), verbose=verbose)
    
    # 如果最终表达式是赋值语句（如 c=...），提取右侧表达式
    if final_expr and '=' in final_expr:
        final_expr_match = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$', final_expr)
        if final_expr_match:
            final_expr = final_expr_match.group(1).strip()
            # 移除表达式两端的括号（如果存在）
            if final_expr.startswith('(') and final_expr.endswith(')'):
                if final_expr.count('(') == final_expr.count(')'):
                    final_expr = final_expr[1:-1]
    
    # 如果没有最终表达式，则最后一个赋值的结果就是最终结果
    if not final_expr and assignments:
        final_expr = assignments[-1][0]
        assignments = assignments[:-1]
    
    if not final_expr:
        raise Exception("未找到最终表达式")
    
    # 收集所有中间变量名
    intermediate_vars = {var_name for var_name, _ in assignments}
    
    # 验证中间变量名
    reserved_names = set(keyword.kwlist)  # Python关键字
    # 添加一些常见的函数名（这些在函数库中定义）
    reserved_names.update(['RANK', 'CS_ZSCORE', 'CS_DEMEAN', 'CS_WINSORIZE', 'CS_BUCKET', 'CS_NEUTRALIZE',
                          'SLOPE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 
                          'AND', 'OR', 'MAX', 'MIN', 'MEAN', 'STD', 'ABS', 'SIGN', 'CAST',
                          'DELTA', 'DELAY', 'EMA', 'SMA', 'TS_MAX', 'TS_MIN', 'TS_MEAN',
                          'np', 'pd', 'df'])
    
    for var_name in intermediate_vars:
        if var_name in reserved_names:
            raise Exception(f"中间变量名 '{var_name}' 是保留名称（Python关键字或函数名），请使用其他名称")
    
    # 解析每个赋值语句的表达式
    parsed_assignments = []
    for var_name, expr_str in assignments:
        try:
            parsed_expr = parse_expression(expr_str, verbose=verbose)
            parsed_assignments.append((var_name, parsed_expr))
        except Exception as e:
            raise Exception(f"解析变量 {var_name} 的表达式失败: {e}")
    
    # 解析最终表达式
    # 对于最终表达式，我们需要特殊处理中间变量名
    # 策略：先尝试解析，如果失败，则手动处理中间变量名
    
    # 检查最终表达式中是否包含中间变量名
    has_intermediate_vars = any(re.search(r'\b' + re.escape(var_name) + r'\b', final_expr) for var_name in intermediate_vars)
    
    if has_intermediate_vars:
        # 如果包含中间变量，我们需要手动处理
        # 将中间变量名替换为临时标识符，解析后再替换回来
        # 使用 $ 前缀的临时变量名，这样解析器会将其识别为变量
        var_temp_map = {}
        temp_final_expr = final_expr
        for var_name in intermediate_vars:
            if re.search(r'\b' + re.escape(var_name) + r'\b', temp_final_expr):
                # 对于下划线开头的变量，需要特殊处理
                if var_name.startswith('_'):
                    temp_var = f"$TEMP{var_name}"
                else:
                    temp_var = f"${var_name}_temp"
                var_temp_map[temp_var] = var_name
                temp_final_expr = re.sub(r'\b' + re.escape(var_name) + r'\b', temp_var, temp_final_expr)
        
        try:
            parsed_final_expr = parse_expression(temp_final_expr)
            # 将临时变量名替换回中间变量名
            for temp_var, var_name in var_temp_map.items():
                parsed_final_expr = parsed_final_expr.replace(temp_var, var_name)
        except Exception as e:
            # 如果解析失败，尝试直接使用中间变量名（不解析）
            # 这种情况下，最终表达式可能包含未解析的部分
            raise Exception(f"解析最终表达式失败: {e}")
    else:
        # 如果不包含中间变量，正常解析
        try:
            parsed_final_expr = parse_expression(final_expr)
        except Exception as e:
            raise Exception(f"解析最终表达式失败: {e}")
    
    # 构建 Python 代码
    # 使用原生变量存储中间结果，避免重复计算
    code_lines = []
    
    # 按顺序执行赋值语句
    for var_name, parsed_expr in parsed_assignments:
        code_lines.append(f"{var_name} = {parsed_expr}")
    
    # 最后返回最终表达式的结果
    code_lines.append(parsed_final_expr)
    
    # 将所有代码行合并，用换行符连接
    return '\n'.join(code_lines)


if __name__ == '__main__':
    # 多行表达式示例（勿用变量名 expr，否则会覆盖模块顶层的 Forward 解析器）
    multi_line_sample = """
    x = DELTA($open, 1)
    y = RANK(x - $close)
    y / (1e-8 + 1)
    """
    print(parse_multi_line_expression(multi_line_sample))