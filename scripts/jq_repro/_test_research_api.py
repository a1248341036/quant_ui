# -*- coding: utf-8 -*-
"""研究笔记本(板块轮动统计打分)关键调用链兼容验证。

覆盖: get_all_securities(date)/get_extras('is_st')/get_industry/
get_price(panel=True).close/display/pd.set_option(-1)/time.clock。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")                          # 后端无 GUI

import pandas as pd  # noqa: E402
pd.set_option("display.max_rows", 100)

CODE = r'''
from jqdata import *
import pandas as pd
import numpy as np
from datetime import datetime,date,timedelta
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
import time
def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        ret=fn(*args, **kwargs)
        print("%s cost %s second"%(fn.__name__, time.clock() - start))
        return ret
    return _wrapper


def stock_industry(date,industry_category_list):
    dt=datetime.strptime(date,'%Y-%m-%d')
    df = get_all_securities(types=['stock'], date=date)
    df = df[df['start_date'] < (dt - timedelta(days=180)).date()]
    df_st_stock = get_extras('is_st', list(df.index), end_date=date, count=1).reset_index(drop=True).T
    stock_list = list(df_st_stock[df_st_stock[0] == False].index)
    df_sec = df[df.index.isin(stock_list)]
    ind=get_industry(list(df_sec.index),date=date)
    ind_dict={k:[ind[k][icode]['industry_name'] for icode in industry_category_list if len(set(ind[k].keys()).intersection(set(industry_category_list)))==len(industry_category_list)] for k in ind}
    df_ind=pd.DataFrame.from_dict(ind_dict, orient='index',
                       columns=[code for code in industry_category_list])
    df_ret = pd.merge(df_sec,df_ind,left_index=True,right_index=True,how='left')
    return df_ret


def returns_series(stock_df,date,days,back_days):
    panel = get_price(list(stock_df.index), end_date=date, frequency='daily', fields=['close'], fq='pre', count=days+back_days+1,panel=True)
    df_close = panel.close
    for i in range(0,back_days):
        stock_df['return_'+str(i+1)]= (df_close.iloc[-(back_days+1-i)] / df_close.iloc[-(back_days+1-i+days)])-1
    return stock_df


def group_score_series(df,change_limit,days,back_days):
    scores=[]
    for i in range(0,back_days):
        df_ok=df[df['return_'+str(i+1)]>change_limit/100]
        mean =df_ok['return_'+str(i+1)].mean()*100 if len(df_ok)>0 and len(df)>10 else 0
        score= int(len(df_ok)/len(df)*mean*10)
        scores.append(score)
    return scores


def group_top_list(df,top_count,change_limit,days):
    df_top = df.sort_values(by='return_'+str(days), ascending=False).head(top_count)
    df_top['print']=df_top['display_name']+":"+(df_top['return_'+str(days)]*100).round(2).astype('str')+"%"
    df_ok=df[df['return_'+str(days)]>change_limit/100]
    return "["+str(len(df_ok))+"/"+str(len(df))+"]"+",".join(list(df_top['print']))


def top_industry(date,industry_category='sw_l2',return_days=5,back_days=10,stock_top_count=5,industry_top_count=10,up_limit=10):
    df_ind = stock_industry(date, [industry_category])
    df_return = returns_series(df_ind, date, return_days,back_days)
    s_tops = df_return.groupby(industry_category).apply(group_top_list, stock_top_count,up_limit,return_days)
    s_scores = df_return.groupby(industry_category).apply(group_score_series, up_limit,return_days,back_days)
    df= pd.DataFrame({"scores": s_scores, "top": s_tops}, index=s_tops.index)
    display(df.head(industry_top_count))
    return df


def initialize(context):
    pass

df1 = top_industry('2025-06-17','sw_l2',return_days=5,back_days=10,stock_top_count=3,industry_top_count=5,up_limit=5)
df2 = top_industry('2025-06-17','zjw',return_days=5,back_days=10,stock_top_count=3,industry_top_count=5,up_limit=5)
df3 = top_industry('2025-06-17','jq_l2',return_days=5,back_days=10,stock_top_count=3,industry_top_count=5,up_limit=5)
print("sw_l2 top3:", list(df1.index[:3]))
print("zjw top3:", list(df2.index[:3]))
print("jq_l2 top3:", list(df3.index[:3]))
'''


def main() -> None:
    import io
    from contextlib import redirect_stdout

    from core.event_engine.jq.entry import run_jq_backtest
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = run_jq_backtest(CODE, start="2025-01-06", end="2025-06-30",
                              capital=100_000.0)
    out = buf.getvalue()
    print("ok:", res["ok"])
    # 用户代码 print 走进程 stdout(不进 log buffer), 从捕获文本断言输出
    for key in ("sw_l2 top3:", "zjw top3:", "jq_l2 top3:"):
        assert key in out, f"输出缺失: {key}"
    warn = [x for x in res["logs"] if "[warn]" in x]
    print("warn:", warn[:5])
    assert res["ok"], "回测失败"
    assert not warn, f"意外警告: {warn[:3]}"
    print("研究笔记本兼容链路验证通过")


if __name__ == "__main__":
    main()
