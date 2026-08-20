-- SQLite 业务库 schema（等价 db/schema_duckdb.sql）
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL DEFAULT 'backtest',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    params        TEXT NOT NULL DEFAULT '{}',
    metrics       TEXT,
    bench_metrics TEXT,
    nav           TEXT,
    bench         TEXT,
    drawdown      TEXT,
    holdings      TEXT,
    trades        TEXT,
    summary       TEXT,
    data_version  TEXT,
    data_snapshot_hash TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS strategy_pool (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    factor      TEXT NOT NULL,
    ascending   INTEGER NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',
    group_name  TEXT,
    description TEXT,
    source      TEXT NOT NULL DEFAULT 'registry',
    sort_order  INTEGER NOT NULL DEFAULT 0,
    added_at    TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at  TEXT
);

CREATE TABLE IF NOT EXISTS strategy_trash (
    name        TEXT PRIMARY KEY,
    factor      TEXT,
    ascending   INTEGER,
    params      TEXT NOT NULL DEFAULT '{}',
    group_name  TEXT,
    description TEXT,
    source      TEXT,
    deleted_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_transactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL,
    code       TEXT NOT NULL,
    name       TEXT,
    action     TEXT NOT NULL,
    shares     REAL NOT NULL,
    price      REAL NOT NULL,
    fee        REAL NOT NULL DEFAULT 0,
    note       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ledger_deposits (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL,
    amount     REAL NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_accounts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    status              TEXT NOT NULL DEFAULT 'active',
    strategy_type       TEXT NOT NULL DEFAULT 'factor',
    strategy_name       TEXT NOT NULL,
    factor              TEXT NOT NULL,
    ascending           INTEGER NOT NULL DEFAULT 0,
    module              TEXT,
    event_strategy      TEXT,
    start_date          TEXT,
    universe            TEXT NOT NULL DEFAULT '科技TMT',
    capital             REAL NOT NULL DEFAULT 100000,
    top_n               INTEGER NOT NULL DEFAULT 3,
    freq                TEXT NOT NULL DEFAULT 'monthly',
    risk_config         TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_processed_date TEXT,
    last_rebalance_date TEXT
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   INTEGER NOT NULL,
    code         TEXT NOT NULL,
    side         TEXT NOT NULL,
    target_pct   REAL,
    signal_date  TEXT NOT NULL,
    exec_date    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    shares       REAL,
    fill_price   REAL,
    fee          REAL NOT NULL DEFAULT 0,
    reject_reason TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (account_id, code, signal_date, exec_date, side)
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    order_id   INTEGER,
    exec_date  TEXT NOT NULL,
    code       TEXT NOT NULL,
    side       TEXT NOT NULL,
    shares     REAL NOT NULL,
    price      REAL NOT NULL,
    fee        REAL NOT NULL DEFAULT 0,
    note       TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    account_id   INTEGER NOT NULL,
    code         TEXT NOT NULL,
    shares       REAL NOT NULL DEFAULT 0,
    avg_cost     REAL NOT NULL DEFAULT 0,
    updated_date TEXT NOT NULL,
    PRIMARY KEY (account_id, code)
);

CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
    account_id   INTEGER NOT NULL,
    date         TEXT NOT NULL,
    cash         REAL NOT NULL,
    market_value REAL NOT NULL DEFAULT 0,
    equity       REAL NOT NULL,
    pnl          REAL NOT NULL DEFAULT 0,
    pnl_pct      REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (account_id, date)
);

CREATE TABLE IF NOT EXISTS paper_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    date       TEXT NOT NULL,
    level      TEXT NOT NULL DEFAULT 'info',
    msg        TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
