"""pg 后端的类型转换回归用例。

Postgres 的 price 是 NUMERIC，psycopg 读回来是 Decimal；内存后端是 float。
不统一的话 json.dumps(工具结果) 会抛 "Object of type Decimal is not JSON
serializable"——只在切到 pg 后端时才暴露。
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

from shopagent.db import rows_to_dicts


class FakeCursor:
    def __init__(self, cols: list[str], rows: list[tuple]):
        self.description = [SimpleNamespace(name=c) for c in cols]
        self._rows = rows

    def fetchall(self):
        return self._rows


def test_decimal_is_converted_to_float():
    cur = FakeCursor(
        ["chunk_id", "price", "title"],
        [(1, Decimal("15.50"), "x"), (2, None, "y")],
    )

    rows = rows_to_dicts(cur)

    assert rows[0]["price"] == 15.5
    assert isinstance(rows[0]["price"], float)
    assert rows[1]["price"] is None


def test_rows_are_json_serializable():
    """这条是 bug 的复现：旧实现直接 json.dumps 会 TypeError。"""
    cur = FakeCursor(["price"], [(Decimal("6.99"),)])

    assert json.dumps(rows_to_dicts(cur)) == '[{"price": 6.99}]'
