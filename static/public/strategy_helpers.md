---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '19cbbe2b-85d3-407e-b4c9-5a7d75c5231c'
  PropagateID: '19cbbe2b-85d3-407e-b4c9-5a7d75c5231c'
  ReservedCode1: 'c85afc2c-dc8f-4713-970f-ee4a126744de'
  ReservedCode2: 'c85afc2c-dc8f-4713-970f-ee4a126744de'
---

# 策略代码常用函数说明

## 一、引擎给你什么数据

在 `build_factor_frames(close, am20, turn20)` 里，三个参数都是 DataFrame：

- 行 = 交易日（时间升序）
- 列 = 股票代码

| 变量 | 含义 | 说明 |
| --- | --- | --- |
| `close` | 收盘价 | 元 |
| `am20` | 20 日平均成交额 | 元 |
| `turn20` | 20 日平均换手率 | 百分数 |

引擎已处理：因子预热(warmup_days)、T+1 成交、涨跌停/停牌过滤、一手过滤。
你只需要返回 `{"score": 分数矩阵}`，剩下的选股、调仓、费用由引擎完成。

## 二、模板自带辅助函数

### `_zscore(df)` —— 横截面标准化

每天（每行）对全市场股票做 `(x - 均值) / 标准差`。
适合把不同数量级、不同单位的因子放在同一尺度比较；正分 = 高于当天全市场平均水平。

### `_rank_pct(df)` —— 横截面百分位

每天把全市场股票排名压到 0~1：0 = 当天最低，1 = 当天最高。
多因子合成时最常用，因为每个因子都变成同一尺度。

### `_ts_zscore(df, n)` —— 时序 Z-Score

每只股票按自己过去 n 天的均值/标准差做标准化，跟其它股票无关。
适合趋势、背离、超买超卖类判断。

## 三、最常用 pandas 函数

### `pct_change(n)` —— 涨跌幅

```python
close.pct_change(20)  # 相对 20 天前的涨幅：(今价 - 20天前价) / 20天前价
close.pct_change()    # 默认 n=1，日涨跌幅
```

### `shift(n)` —— 取 n 天前的值

```python
close.shift(1)        # 昨天的收盘价
```

**警告：禁止使用 `shift(-n)`、`pct_change(-n)`、`iloc` 未来行，会造成前视偏差。**

### `rolling(n)` —— 滚动窗口

```python
close.rolling(5).mean()                      # 5 日均线
close.pct_change().rolling(20).std()         # 20 日波动率
close.rolling(20).mean()                     # 20 日平均价
```

### `rank(axis=1)` —— 截面排名

```python
df.rank(axis=1)             # 每行内排名 1,2,3...
df.rank(axis=1, pct=True)   # 等价于 _rank_pct(df)
```

### `reindex_like(df)` —— 对齐索引/列

`rolling` 会吃掉前 n 行导致错位，用 `.reindex_like(am20)` 补回完整交易日历：

```python
vol20 = close.pct_change().rolling(20).std().reindex_like(am20)
```

### `clip(lower, upper)` —— 截断极端值

```python
scores = scores.clip(-3, 3)   # 防止个别极端值主导选股
```

## 四、常用因子写法

**动量 20 日（买涨得最猛）**

```python
scores = close.pct_change(20)
scores = _zscore(scores)
```

**低波动 20 日（买波动最小的）**

```python
scores = close.pct_change().rolling(20).std()
scores = _rank_pct(scores)
```

**均线乖离（多头排列 / 超短线强度）**

```python
scores = close.rolling(5).mean() / close.rolling(20).mean() - 1
scores = _zscore(scores)
```

**复合因子（低成交 + 低波动）**

```python
scores = _rank_pct(am20) * (-1) + _rank_pct(close.pct_change().rolling(20).std()) * (-1)
```

## 五、策略注册表 STRATEGIES 怎么用

`STRATEGIES` 里每一项告诉引擎怎么消费你算出的分数：

- `factor`：固定 `"score"`，对应 `build_factor_frames` 返回字典里的键名
- `ascending`：`false` = 分数从大到小买高分；`true` = 分数从小到大买低分
- `group` / `desc`：只用于界面展示，不影响回测

## 六、红线（前视偏差）

- 只允许使用当前行及之前的数据
- 禁止 `pct_change(-n)` / `shift(-n)` / `iloc` 取未来行
- 禁止把回测区间之后才出现的信息写进策略

> AI生成