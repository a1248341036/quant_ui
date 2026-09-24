# cython: language_level=3
import struct
from collections import OrderedDict

from cnequity.adapters.tdx_protocol._wire.parser.base import BaseParser

# 0x000F 股本变迁 / 权息资料. Same command as GetXdXrInfo, but this parser
# keeps every category (1..15) with the four raw float fields, so the adapter
# can apply eltdx's per-category unit semantics instead of tdxpy's
# category-specific interpretation (which only covers 1..14 and misreads the
# share-count categories).
CAPITAL_CHANGE_CATEGORY_NAMES = {
    1: "除权除息",
    2: "送配股上市",
    3: "非流通股上市",
    4: "未知股本变动",
    5: "股本变化",
    6: "增发新股",
    7: "股份回购",
    8: "增发新股上市",
    9: "转配股上市",
    10: "可转债上市",
    11: "扩缩股",
    12: "非流通股缩股",
    13: "送认购权证",
    14: "送认沽权证",
    15: "重整调整",
}


class GetCapitalChangesCmd(BaseParser):
    def setParams(self, market, code):
        """
        设置参数
        :param market: 市场
        :param code: 股票代码
        """
        if type(code) is str:
            code = code.encode("utf-8")

        # count=1: one code per request, matching the lake's per-symbol sweep.
        data = struct.pack("<HB6s", 1, market, code)
        pkg = bytearray(
            struct.pack("<HIHHH", 0x10C, 0x01016408, len(data) + 2, len(data) + 2, 0x000F)
        )
        pkg.extend(data)

        self.send_pkg = pkg

    def parseResponse(self, body_buf):
        """
        解析返回结果
        :param body_buf:
        :return:
        """
        if len(body_buf) < 11:
            return []

        pos = 2  # reported block count; a single-code request returns one block
        rows = []

        while pos + 9 <= len(body_buf):
            (num,) = struct.unpack("<H", body_buf[pos + 7 : pos + 9])
            pos += 9

            for _ in range(num):
                if pos + 29 > len(body_buf):
                    return rows
                market_id, code, reserved, date_raw, category, c1, c2, c3, c4 = (
                    struct.unpack("<B6sBIBffff", body_buf[pos : pos + 29])
                )
                pos += 29

                rows.append(
                    OrderedDict(
                        [
                            ("market", market_id),
                            ("code", code.decode("ascii", errors="replace")),
                            ("reserved", reserved),
                            ("date", date_raw),
                            ("category", category),
                            ("name", CAPITAL_CHANGE_CATEGORY_NAMES.get(category, str(category))),
                            ("c1", c1),
                            ("c2", c2),
                            ("c3", c3),
                            ("c4", c4),
                        ]
                    )
                )

        return rows