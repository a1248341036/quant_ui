#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refresh OTC fund fee data from AkShare/Eastmoney.

The bulk endpoint supplies the current purchase fee. Management, custody,
sales-service and redemption rules are read from each fund's fee page and
cached in data/fund_fee.csv. Existing successful rows are kept unless
--refresh-existing is supplied.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.run_log import job  # noqa: E402
from core.store import DATA_DIR, FUND_DIR, FUND_FILE  # noqa: E402


FEE_FILE = FUND_DIR / "fund_fee.csv"
OPS_INDICATOR = "\u8fd0\u4f5c\u8d39\u7528"
REDEEM_INDICATOR = "\u8d4e\u56de\u8d39\u7387"
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DAY_RE = re.compile(r"(\d+)")


def _rate(value: object) -> float | None:
    text = str(value or "").strip()
    match = PERCENT_RE.search(text)
    if match:
        return float(match.group(1)) / 100
    return None


def _column(df: pd.DataFrame, fragment: str) -> str | None:
    return next((str(c) for c in df.columns if fragment in str(c)), None)


def _parse_operations(df: pd.DataFrame) -> dict[str, float | None]:
    out = {
        "management_fee_rate": None,
        "custodian_fee_rate": None,
        "sales_service_fee_rate": None,
    }
    if df is None or df.empty:
        return out
    values = [str(v) for v in df.iloc[0].tolist()]
    for i, value in enumerate(values[:-1]):
        for label, key in (
            ("\u7ba1\u7406\u8d39\u7387", "management_fee_rate"),
            ("\u6258\u7ba1\u8d39\u7387", "custodian_fee_rate"),
            ("\u9500\u552e\u670d\u52a1\u8d39\u7387", "sales_service_fee_rate"),
        ):
            if label in value:
                out[key] = _rate(values[i + 1])
    return out


def _parse_redemption(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    period_col = _column(df, "\u9002\u7528\u671f\u9650") or str(df.columns[0])
    rate_col = _column(df, "\u8d4e\u56de\u8d39\u7387") or str(df.columns[-1])
    rules = []
    for _, row in df.iterrows():
        period = str(row.get(period_col, "")).strip()
        rate = _rate(row.get(rate_col, ""))
        if not period or rate is None:
            continue
        numbers = [int(x) for x in DAY_RE.findall(period)]
        rule: dict[str, object] = {"period": period, "rate": rate}
        if "\u5c0f\u4e8e" in period and numbers:
            rule["max_days_exclusive"] = numbers[-1]
        if "\u5927\u4e8e\u7b49\u4e8e" in period and numbers:
            rule["min_days"] = numbers[0]
        rules.append(rule)
    return json.dumps(rules, ensure_ascii=False, separators=(",", ":"))


def _bulk_purchase_rates() -> dict[str, float | None]:
    import akshare as ak

    df = ak.fund_open_fund_daily_em()
    code_col = _column(df, "\u57fa\u91d1\u4ee3\u7801")
    fee_col = _column(df, "\u624b\u7eed\u8d39")
    if not code_col or not fee_col:
        return {}
    out: dict[str, float | None] = {}
    for _, row in df.iterrows():
        code = str(row[code_col]).zfill(6)
        text = str(row[fee_col]).strip()
        match = PERCENT_RE.search(text)
        if match:
            out[code] = float(match.group(1)) / 100
        else:
            try:
                out[code] = float(text) / 100
            except (TypeError, ValueError):
                out[code] = None
    return out


def _fetch_one(code: str) -> dict:
    import akshare as ak

    row = {"code": code, "fee_status": "error", "last_error": ""}
    errors = []
    try:
        operations = ak.fund_fee_em(code, indicator=OPS_INDICATOR)
        row.update(_parse_operations(operations))
    except Exception as exc:
        errors.append(f"operations: {exc}")
    try:
        redemption = ak.fund_fee_em(code, indicator=REDEEM_INDICATOR)
        row["redemption_fee_rule"] = _parse_redemption(redemption)
    except Exception as exc:
        errors.append(f"redemption: {exc}")
    row["fee_status"] = "ok" if any(
        row.get(k) is not None for k in (
            "management_fee_rate", "custodian_fee_rate",
            "sales_service_fee_rate", "redemption_fee_rule",
        )
    ) else "error"
    row["last_error"] = "; ".join(errors)[:300]
    if row["fee_status"] == "error" and not row["last_error"]:
        row["last_error"] = "fee page returned no usable fields"
    return row


def _load_existing() -> pd.DataFrame:
    if not FEE_FILE.exists():
        return pd.DataFrame(columns=["code"])
    df = pd.read_csv(FEE_FILE, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh fund fee data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="only fetch N missing funds")
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()

    with job("quant_ui:refresh_fund_fees", metadata={"workers": args.workers, "limit": args.limit, "refresh_existing": args.refresh_existing}) as run:
        if not FUND_FILE.exists():
            print(f"missing fund universe: {FUND_FILE}", file=sys.stderr)
            return 1
        funds = pd.read_csv(FUND_FILE, dtype={"code": str})
        funds["code"] = funds["code"].astype(str).str.zfill(6)
        funds = funds.drop_duplicates("code")[["code", "name", "type"]]
        existing = _load_existing()

        print("loading bulk purchase fees...", flush=True)
        try:
            purchase = _bulk_purchase_rates()
        except Exception as exc:
            print(f"bulk purchase fee load failed: {exc}", file=sys.stderr)
            purchase = {}

        now = datetime.now().isoformat(timespec="seconds")
        base = funds.copy()
        base["purchase_fee_rate"] = base["code"].map(purchase)
        if len(existing):
            keep_cols = [c for c in existing.columns if c not in {"name", "type", "purchase_fee_rate"}]
            base = base.merge(existing[keep_cols], on="code", how="left")
            if "purchase_fee_rate" in existing:
                old = existing[["code", "purchase_fee_rate"]].drop_duplicates("code")
                base = base.merge(old, on="code", how="left", suffixes=("", "_old"))
                base["purchase_fee_rate"] = base["purchase_fee_rate"].fillna(base["purchase_fee_rate_old"])
                base = base.drop(columns=["purchase_fee_rate_old"])

        required = ["management_fee_rate", "custodian_fee_rate", "sales_service_fee_rate", "redemption_fee_rule"]
        for col in required:
            if col not in base:
                base[col] = pd.Series(pd.NA, index=base.index, dtype="object")
        if "fee_status" not in base:
            base["fee_status"] = pd.Series(pd.NA, index=base.index, dtype="object")
        if "last_error" not in base:
            base["last_error"] = pd.Series("", index=base.index, dtype="object")
        if "fee_updated_at" not in base:
            base["fee_updated_at"] = pd.Series("", index=base.index, dtype="object")

        if args.refresh_existing:
            mask = pd.Series(True, index=base.index)
        else:
            mask = base["fee_status"].fillna("").ne("ok")
        codes = base.loc[mask, "code"].tolist()
        if args.limit:
            codes = codes[:args.limit]
        print(f"funds={len(base)} fetch_fee_pages={len(codes)} workers={args.workers}", flush=True)

        index = {str(code): i for i, code in enumerate(base["code"])}
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_fetch_one, code): code for code in codes}
            for future in as_completed(futures):
                result = future.result()
                i = index[result["code"]]
                for key, value in result.items():
                    if key != "code":
                        base.loc[i, key] = value
                if result["fee_status"] == "ok":
                    base.loc[i, "fee_updated_at"] = now
                done += 1
                if done % 100 == 0 or done == len(codes):
                    print(f"  [{done}/{len(codes)}]", flush=True)

        columns = [
            "code", "name", "type", "purchase_fee_rate", "management_fee_rate",
            "custodian_fee_rate", "sales_service_fee_rate", "redemption_fee_rule",
            "fee_status", "fee_updated_at", "last_error",
        ]
        base = base[[c for c in columns if c in base]]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = FEE_FILE.with_suffix(".csv.tmp")
        base.to_csv(tmp, index=False)
        tmp.replace(FEE_FILE)
        n_ok = (base['fee_status'] == 'ok').sum()
        print(f"saved {FEE_FILE}: rows={len(base)} ok={n_ok}", flush=True)
        run.set_rows(rows_written=len(base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
