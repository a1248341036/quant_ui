"""探针：中间件现在对 null div_proc 样本行返回什么。用后即删。"""
import sys

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")

from cnequity.config import load_config
from cnequity.external.tushare_fetch import _fetch_with_retry, _get_pro

cfg = load_config(r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml")
pro = _get_pro(cfg)

for code in ("600011.SH", "000001.SZ"):
    df = _fetch_with_retry(pro, "dividend", interval=0.35, ts_code=code)
    print(f"--- {code}: {df.height} rows, cols={df.columns}")
    sub = df.filter(df["end_date"].is_in(["20181231", "20191231", "20231231", "20241231"]))
    if sub.height:
        cols = [c for c in ("end_date", "ann_date", "div_proc", "cash_div", "stk_div", "ex_date", "pay_date") if c in sub.columns]
        print(sub.select(cols).head(8))
    import time
    time.sleep(0.4)
