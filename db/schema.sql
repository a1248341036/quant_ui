-- quant_ui PostgreSQL/TimescaleDB schema
-- 用法: psql "$PG_DSN" -f db/schema.sql
-- 说明: 财务宽表（fina_indicator/income 等）由 scripts/sync_postgres.py 按接口列动态创建

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS trade_cal (
    exchange      VARCHAR(16) NOT NULL,
    cal_date      DATE NOT NULL,
    is_open       SMALLINT NOT NULL DEFAULT 0,
    pretrade_date DATE,
    PRIMARY KEY (exchange, cal_date)
);

CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code   VARCHAR(12) PRIMARY KEY,
    symbol    VARCHAR(8),
    name      VARCHAR(32),
    area      VARCHAR(32),
    industry  VARCHAR(32),
    market    VARCHAR(16),
    list_date DATE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stock_daily (
    ts_code         VARCHAR(12) NOT NULL,
    trade_date      DATE NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    pre_close       DOUBLE PRECISION,
    change          DOUBLE PRECISION,
    pct_chg         DOUBLE PRECISION,
    vol             DOUBLE PRECISION,
    amount          DOUBLE PRECISION,
    turnover_rate   DOUBLE PRECISION,
    turnover_rate_f DOUBLE PRECISION,
    volume_ratio    DOUBLE PRECISION,
    pe              DOUBLE PRECISION,
    pe_ttm          DOUBLE PRECISION,
    pb              DOUBLE PRECISION,
    ps              DOUBLE PRECISION,
    ps_ttm          DOUBLE PRECISION,
    dv_ratio        DOUBLE PRECISION,
    dv_ttm          DOUBLE PRECISION,
    total_share     DOUBLE PRECISION,
    float_share     DOUBLE PRECISION,
    free_share      DOUBLE PRECISION,
    total_mv        DOUBLE PRECISION,
    circ_mv         DOUBLE PRECISION,
    adj_factor      DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);
SELECT create_hypertable('stock_daily', 'trade_date', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS stock_minute (
    ts      TIMESTAMPTZ NOT NULL,
    ts_code VARCHAR(12) NOT NULL,
    freq    VARCHAR(8) NOT NULL DEFAULT '1min',
    open    DOUBLE PRECISION,
    high    DOUBLE PRECISION,
    low     DOUBLE PRECISION,
    close   DOUBLE PRECISION,
    volume  DOUBLE PRECISION,
    amount  DOUBLE PRECISION,
    PRIMARY KEY (ts_code, ts, freq)
);
SELECT create_hypertable('stock_minute', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS dividend (
    id            BIGSERIAL PRIMARY KEY,
    ts_code       VARCHAR(12) NOT NULL,
    end_date      DATE,
    ann_date      DATE,
    div_proc      VARCHAR(32),
    stk_div       DOUBLE PRECISION,
    stk_bo_rate   DOUBLE PRECISION,
    stk_co_rate   DOUBLE PRECISION,
    cash_div      DOUBLE PRECISION,
    cash_div_tax  DOUBLE PRECISION,
    record_date   DATE,
    ex_date       DATE,
    pay_date      DATE,
    div_listdate  DATE,
    imp_ann_date  DATE
);
CREATE INDEX IF NOT EXISTS idx_dividend_code ON dividend (ts_code);

CREATE TABLE IF NOT EXISTS share_float (
    id          BIGSERIAL PRIMARY KEY,
    ts_code     VARCHAR(12) NOT NULL,
    ann_date    DATE,
    float_date  DATE,
    float_share DOUBLE PRECISION,
    float_ratio DOUBLE PRECISION,
    holder_name TEXT,
    share_type  VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS namechange (
    id            BIGSERIAL PRIMARY KEY,
    ts_code       VARCHAR(12) NOT NULL,
    name          VARCHAR(32),
    start_date    DATE,
    end_date      DATE,
    ann_date      DATE,
    change_reason VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS idx_namechange_code ON namechange (ts_code);

CREATE TABLE IF NOT EXISTS forecast (
    id              BIGSERIAL PRIMARY KEY,
    ts_code         VARCHAR(12) NOT NULL,
    ann_date        DATE,
    end_date        DATE,
    type            VARCHAR(32),
    p_change_min    DOUBLE PRECISION,
    p_change_max    DOUBLE PRECISION,
    net_profit_min  DOUBLE PRECISION,
    net_profit_max  DOUBLE PRECISION,
    last_parent_net DOUBLE PRECISION,
    first_ann_date  DATE,
    summary         TEXT,
    change_reason   TEXT,
    update_flag     VARCHAR(8)
);
CREATE INDEX IF NOT EXISTS idx_forecast_code ON forecast (ts_code);

CREATE TABLE IF NOT EXISTS express (
    id            BIGSERIAL PRIMARY KEY,
    ts_code       VARCHAR(12) NOT NULL,
    ann_date      DATE,
    end_date      DATE,
    revenue       DOUBLE PRECISION,
    operate_profit DOUBLE PRECISION,
    total_profit  DOUBLE PRECISION,
    n_income      DOUBLE PRECISION,
    total_assets  DOUBLE PRECISION,
    total_hldr_eqy_exc_min_int DOUBLE PRECISION,
    diluted_eps   DOUBLE PRECISION,
    diluted_roe   DOUBLE PRECISION,
    yoy_net_profit DOUBLE PRECISION,
    bps           DOUBLE PRECISION,
    open_net_assets DOUBLE PRECISION,
    open_bps      DOUBLE PRECISION,
    perf_summary  TEXT
);
CREATE INDEX IF NOT EXISTS idx_express_code ON express (ts_code);

CREATE TABLE IF NOT EXISTS stk_surv (
    id            BIGSERIAL PRIMARY KEY,
    ts_code       VARCHAR(12) NOT NULL,
    name          VARCHAR(32),
    surv_date     DATE,
    fund_visitors TEXT,
    rece_place    TEXT,
    rece_mode     TEXT,
    rece_org      TEXT,
    org_type      TEXT,
    comp_rece     TEXT
);
CREATE INDEX IF NOT EXISTS idx_stk_surv_code ON stk_surv (ts_code);

CREATE TABLE IF NOT EXISTS report_rc (
    id           BIGSERIAL PRIMARY KEY,
    ts_code      VARCHAR(12) NOT NULL,
    name         VARCHAR(32),
    report_date  DATE,
    report_title TEXT,
    report_type  VARCHAR(32),
    classify     VARCHAR(32),
    org_name     VARCHAR(64),
    author_name  TEXT,
    quarter      VARCHAR(16),
    op_rt        DOUBLE PRECISION,
    op_pr        DOUBLE PRECISION,
    tp           DOUBLE PRECISION,
    np           DOUBLE PRECISION,
    eps          DOUBLE PRECISION,
    pe           DOUBLE PRECISION,
    rd           DOUBLE PRECISION,
    roe          DOUBLE PRECISION,
    ev_ebitda    DOUBLE PRECISION,
    rating       VARCHAR(32),
    max_price    DOUBLE PRECISION,
    min_price    DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_report_rc_date ON report_rc (report_date DESC);
CREATE INDEX IF NOT EXISTS idx_report_rc_code ON report_rc (ts_code);

CREATE TABLE IF NOT EXISTS index_weight (
    index_code  VARCHAR(12) NOT NULL,
    con_code    VARCHAR(12) NOT NULL,
    trade_date  DATE NOT NULL,
    weight      DOUBLE PRECISION,
    PRIMARY KEY (index_code, con_code, trade_date)
);

CREATE TABLE IF NOT EXISTS sync_log (
    table_name VARCHAR(64) NOT NULL,
    kind       VARCHAR(32),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    n_rows     BIGINT,
    detail     TEXT,
    PRIMARY KEY (table_name, started_at)
);

-- 回测结果归档（UI/脚本每次跑都落一条，可追溯参数与净值）
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    kind          VARCHAR(16) NOT NULL DEFAULT 'backtest',  -- backtest | compare | sweep
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    params        JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics       JSONB,
    bench_metrics JSONB,
    nav           JSONB,
    bench         JSONB,
    drawdown      JSONB,
    holdings      JSONB,
    trades        JSONB,
    summary       JSONB,
    data_version  VARCHAR(32),
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created ON backtest_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_kind ON backtest_runs (kind);

-- 策略池：配置池（全量池 = 注册表 STRATEGIES + backtest_runs 归档）
CREATE TABLE IF NOT EXISTS strategy_pool (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR(64) NOT NULL UNIQUE,
    factor      VARCHAR(64) NOT NULL,
    ascending   BOOLEAN NOT NULL,
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    group_name  VARCHAR(32),
    description TEXT,
    source      VARCHAR(16) NOT NULL DEFAULT 'registry',
    sort_order  INT NOT NULL DEFAULT 0,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at  TIMESTAMPTZ
);

-- 策略回收站：从全量池删除的策略
CREATE TABLE IF NOT EXISTS strategy_trash (
    name        VARCHAR(64) PRIMARY KEY,
    factor      VARCHAR(64),
    ascending   BOOLEAN,
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    group_name  VARCHAR(32),
    description TEXT,
    source      VARCHAR(16),
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 账户账本（原 CSV，迁入 PG）
CREATE TABLE IF NOT EXISTS ledger_transactions (
    id         BIGSERIAL PRIMARY KEY,
    date       DATE NOT NULL,
    code       VARCHAR(8) NOT NULL,
    name       VARCHAR(32),
    action     VARCHAR(8) NOT NULL,
    shares     DOUBLE PRECISION NOT NULL,
    price      DOUBLE PRECISION NOT NULL,
    fee        DOUBLE PRECISION NOT NULL DEFAULT 0,
    note       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ledger_tx_date ON ledger_transactions (date);

CREATE TABLE IF NOT EXISTS ledger_deposits (
    id         BIGSERIAL PRIMARY KEY,
    date       DATE NOT NULL,
    amount     DOUBLE PRECISION NOT NULL,
    note       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ledger_dep_date ON ledger_deposits (date);

-- 日级模拟盘
CREATE TABLE IF NOT EXISTS paper_accounts (
    id                  BIGSERIAL PRIMARY KEY,
    name                VARCHAR(64) NOT NULL UNIQUE,
    status              VARCHAR(16) NOT NULL DEFAULT 'active',
    strategy_type       VARCHAR(16) NOT NULL DEFAULT 'factor',
    strategy_name       VARCHAR(64) NOT NULL,
    factor              VARCHAR(32) NOT NULL,
    ascending           BOOLEAN NOT NULL DEFAULT FALSE,
    module              TEXT,
    event_strategy      VARCHAR(64),
    start_date          DATE,
    universe            VARCHAR(64) NOT NULL DEFAULT '科技TMT',
    capital             DOUBLE PRECISION NOT NULL DEFAULT 100000,
    top_n               INT NOT NULL DEFAULT 3,
    freq                VARCHAR(16) NOT NULL DEFAULT 'monthly',
    risk_config         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_processed_date DATE,
    last_rebalance_date DATE
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id           BIGSERIAL PRIMARY KEY,
    account_id   BIGINT NOT NULL,
    code         VARCHAR(8) NOT NULL,
    side         VARCHAR(8) NOT NULL,
    target_pct   DOUBLE PRECISION,
    signal_date  DATE NOT NULL,
    exec_date    DATE NOT NULL,
    status       VARCHAR(16) NOT NULL DEFAULT 'pending',
    shares       DOUBLE PRECISION,
    fill_price   DOUBLE PRECISION,
    fee          DOUBLE PRECISION NOT NULL DEFAULT 0,
    reject_reason TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (account_id, code, signal_date, exec_date, side)
);
CREATE INDEX IF NOT EXISTS idx_paper_orders_acct ON paper_orders (account_id, exec_date);

CREATE TABLE IF NOT EXISTS paper_trades (
    id         BIGSERIAL PRIMARY KEY,
    account_id BIGINT NOT NULL,
    order_id   BIGINT,
    exec_date  DATE NOT NULL,
    code       VARCHAR(8) NOT NULL,
    side       VARCHAR(8) NOT NULL,
    shares     DOUBLE PRECISION NOT NULL,
    price      DOUBLE PRECISION NOT NULL,
    fee        DOUBLE PRECISION NOT NULL DEFAULT 0,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_acct ON paper_trades (account_id, exec_date);

CREATE TABLE IF NOT EXISTS paper_positions (
    account_id   BIGINT NOT NULL,
    code         VARCHAR(8) NOT NULL,
    shares       DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_cost     DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_date DATE NOT NULL,
    PRIMARY KEY (account_id, code)
);

CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
    account_id   BIGINT NOT NULL,
    date         DATE NOT NULL,
    cash         DOUBLE PRECISION NOT NULL,
    market_value DOUBLE PRECISION NOT NULL DEFAULT 0,
    equity       DOUBLE PRECISION NOT NULL,
    pnl          DOUBLE PRECISION NOT NULL DEFAULT 0,
    pnl_pct      DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (account_id, date)
);
CREATE INDEX IF NOT EXISTS idx_paper_snap_acct ON paper_equity_snapshots (account_id, date);

CREATE TABLE IF NOT EXISTS paper_events (
    id         BIGSERIAL PRIMARY KEY,
    account_id BIGINT NOT NULL,
    date       DATE NOT NULL,
    level      VARCHAR(8) NOT NULL DEFAULT 'info',
    msg        TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_paper_events_acct ON paper_events (account_id, date);
