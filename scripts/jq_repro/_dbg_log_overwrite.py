# -*- coding: utf-8 -*-
"""最小复现: demo 顶层语句逐组定位 ns['log'] 被覆盖的原因."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402
sys.path.insert(0, str(ROOT / "strategies" / "event"))
sys.path.insert(0, str(ROOT / "scripts" / "jq_repro"))

from _runtime import JQContext  # noqa: E402
from core.event_engine.jq.runtime import JQRuntime  # noqa: E402

CASES = {
    "base": "def initialize(context):\n    log.set_level('order','error')",
    "star_jq": ("from jqdata import *\nfrom jqfactor import *\n"
                "def initialize(context):\n    log.set_level('order','error')"),
    "numpy": ("import numpy as np\n"
              "def initialize(context):\n    log.set_level('order','error')"),
    "set_options": ("import pandas as pd\n"
                    "pd.set_option('display.max_rows', 100)\n"
                    "pd.set_option('display.max_columns', 10)\n"
                    "pd.set_option('display.width', None)\n"
                    "pd.set_option('display.max_colwidth', -1)\n"
                    "import warnings\nwarnings.filterwarnings('ignore')\n"
                    "import time\n"
                    "def initialize(context):\n    log.set_level('order','error')"),
    "full_top": ("from jqdata import *\nfrom jqfactor import *\n"
                 "import numpy as np\nimport pandas as pd\n"
                 "from datetime import time, timedelta, datetime\n"
                 "import warnings\nwarnings.filterwarnings('ignore')\n"
                 "pd.set_option('display.max_rows', 100)\n"
                 "pd.set_option('display.max_columns', 10)\n"
                 "pd.set_option('display.width', None)\n"
                 "pd.set_option('display.max_colwidth', -1)\n"
                 "import time\n"
                 "def initialize(context):\n    log.set_level('order','error')"),
}

for name, code in CASES.items():
    ctx = JQContext(end="2025-02-28", lookback_days=120)
    rt = JQRuntime(code, ctx, 100_000.0)
    print(f"{name:12s} ns['log'] = {type(rt._ns.get('log'))}")
    try:
        rt._init_fn(rt.context)
        print(f"{'':12s} init OK")
    except Exception as exc:  # noqa: BLE001
        print(f"{'':12s} init FAIL: {exc}")
