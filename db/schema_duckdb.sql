
CREATE SEQUENCE IF NOT EXISTS backtest_runs_id_seq START 1;
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id        INTEGER PRIMARY KEY DEFAULT nextval('backtest_runs_id_seq'),
    kind          VARCHAR(16) NOT NULL DEFAULT 'backtest',
    created_at    TIMESTAMP NOT NULL DEFAULT current_timestamp,
    params        JSON NOT NULL DEFAULT '{}'::JSON,
    metrics       JSON,
    bench_metrics JSON,
    nav           JSON,
    bench         JSON,
    drawdown      JSON,
    holdings      JSON,
    trades        JSON,
    summary       JSON,
    data_version  VARCHAR(32),
    error         TEXT
);

CREATE SEQUENCE IF NOT EXISTS strategy_pool_id_seq START 1;
CREATE TABLE IF NOT EXISTS strategy_pool (
    id          INTEGER PRIMARY KEY DEFAULT nextval('strategy_pool_id_seq'),
    name        VARCHAR(64) NOT NULL UNIQUE,
    factor      VARCHAR(64) NOT NULL,
    ascending   BOOLEAN NOT NULL,
    params      JSON NOT NULL DEFAULT '{}'::JSON,
    group_name  VARCHAR(32),
    description TEXT,
    source      VARCHAR(16) NOT NULL DEFAULT 'registry',
    sort_order  INT NOT NULL DEFAULT 0,
    added_at    TIMESTAMP NOT NULL DEFAULT current_timestamp,
    deleted_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_trash (
    name        VARCHAR(64) PRIMARY KEY,
    factor      VARCHAR(64),
    ascending   BOOLEAN,
    params      JSON NOT NULL DEFAULT '{}'::JSON,
    group_name  VARCHAR(32),
    description TEXT,
    source      VARCHAR(16),
    deleted_at  TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE SEQUENCE IF NOT EXISTS ledger_transactions_id_seq START 1;
CREATE TABLE IF NOT EXISTS ledger_transactions (
    id         INTEGER PRIMARY KEY DEFAULT nextval('ledger_transactions_id_seq'),
    date       DATE NOT NULL,
    code       VARCHAR(8) NOT NULL,
    name       VARCHAR(32),
    action     VARCHAR(8) NOT NULL,
    shares     DOUBLE PRECISION NOT NULL,
    price      DOUBLE PRECISION NOT NULL,
    fee        DOUBLE PRECISION NOT NULL DEFAULT 0,
    note       TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE SEQUENCE IF NOT EXISTS ledger_deposits_id_seq START 1;
CREATE TABLE IF NOT EXISTS ledger_deposits (
    id         INTEGER PRIMARY KEY DEFAULT nextval('ledger_deposits_id_seq'),
    date       DATE NOT NULL,
    amount     DOUBLE PRECISION NOT NULL,
    note       TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE SEQUENCE IF NOT EXISTS paper_accounts_id_seq START 1;
CREATE TABLE IF NOT EXISTS paper_accounts (
    id                  INTEGER PRIMARY KEY DEFAULT nextval('paper_accounts_id_seq'),
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
    risk_config         JSON NOT NULL DEFAULT '{}'::JSON,
    created_at          TIMESTAMP NOT NULL DEFAULT current_timestamp,
    last_processed_date DATE,
    last_rebalance_date DATE
);

CREATE SEQUENCE IF NOT EXISTS paper_orders_id_seq START 1;
CREATE TABLE IF NOT EXISTS paper_orders (
    id           INTEGER PRIMARY KEY DEFAULT nextval('paper_orders_id_seq'),
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
    created_at   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    UNIQUE (account_id, code, signal_date, exec_date, side)
);

CREATE SEQUENCE IF NOT EXISTS paper_trades_id_seq START 1;
CREATE TABLE IF NOT EXISTS paper_trades (
    id         INTEGER PRIMARY KEY DEFAULT nextval('paper_trades_id_seq'),
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

CREATE SEQUENCE IF NOT EXISTS paper_events_id_seq START 1;
CREATE TABLE IF NOT EXISTS paper_events (
    id         INTEGER PRIMARY KEY DEFAULT nextval('paper_events_id_seq'),
    account_id BIGINT NOT NULL,
    date       DATE NOT NULL,
    level      VARCHAR(8) NOT NULL DEFAULT 'info',
    msg        TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
