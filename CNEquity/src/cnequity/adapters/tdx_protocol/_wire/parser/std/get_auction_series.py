# cython: language_level=3
import struct
from collections import OrderedDict

from cnequity.adapters.tdx_protocol._wire.parser.base import BaseParser

# 0x056A 集合竞价过程快照. Defaults match eltdx's AuctionSeriesRequest:
# selector=3, start=0, count=500 — the server returns the whole session's
# snapshots (opening 09:15-09:25 plus closing 14:57-15:00) in one response.
AUCTION_DEFAULT_SELECTOR = 3
AUCTION_DEFAULT_START = 0
AUCTION_DEFAULT_COUNT = 500


class GetAuctionSeriesCmd(BaseParser):
    def setParams(self, market, code, date=0, mode=AUCTION_DEFAULT_SELECTOR,
                  start=AUCTION_DEFAULT_START, count=AUCTION_DEFAULT_COUNT):
        """
        设置参数
        :param market: 市场
        :param code: 股票代码
        :param date: 交易日期 YYYYMMDD, 0 表示主站当前交易日
        :param mode: 模式/选择器
        :param start: 起始序号
        :param count: 数量
        """
        if type(code) is str:
            code = code.encode("utf-8")

        data = struct.pack(
            "<BB6sIIIII", market, 0, code, int(date), int(mode), 0, int(start), int(count)
        )
        pkg = bytearray(
            struct.pack("<HIHHH", 0x10C, 0x01016408, len(data) + 2, len(data) + 2, 0x056A)
        )
        pkg.extend(data)

        self.send_pkg = pkg

    def parseResponse(self, body_buf):
        """
        解析返回结果
        :param body_buf:
        :return:
        """
        if len(body_buf) < 2:
            return []

        (num,) = struct.unpack("<H", body_buf[:2])
        pos = 2
        rows = []

        for _ in range(num):
            if pos + 16 > len(body_buf):
                return rows
            minute_of_day, price, matched_volume, unmatched_signed, reserved, second = (
                struct.unpack("<HfIiBB", body_buf[pos : pos + 16])
            )
            pos += 16

            rows.append(
                OrderedDict(
                    [
                        ("minute_of_day", minute_of_day),
                        ("second", second),
                        ("price", price),
                        ("matched_volume", matched_volume),
                        ("unmatched_signed", unmatched_signed),
                        ("reserved", reserved),
                    ]
                )
            )

        return rows