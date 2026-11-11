"""价格过滤的回归用例。

背景：目录里约 46.5% 的商品没有价格。早先的实现把 price=None 和超预算的商品
一起丢掉，导致「预算 $20 的瑜伽垫」只剩背带、瑜伽包、护膝这类配件——语义最
匹配的瑜伽垫因为没标价被过滤掉了。现在改为：预算只作用于已知价格，未标价的
保留并打 price_unknown，排在预算内商品之后。
"""

from __future__ import annotations

import pytest

import shopagent.tools.shop as shop


def _hit(asin: str, price, score: float = 0.9) -> dict:
    return {
        "asin": asin,
        "title": f"product {asin}",
        "price": price,
        "content": f"content of {asin}",
        "score": score,
    }


class FakeRetriever:
    def __init__(self, hits: list[dict]):
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]


@pytest.fixture
def fake(monkeypatch):
    def install(hits: list[dict]) -> FakeRetriever:
        retriever = FakeRetriever(hits)
        monkeypatch.setattr(shop, "_retriever", lambda: retriever)
        return retriever

    return install


def test_budget_drops_only_over_budget_priced_items(fake):
    fake([
        _hit("A", 15.0),
        _hit("B", None),      # 未标价：保留
        _hit("C", 25.0),      # 超预算：丢弃
        _hit("D", 10.0),
    ])

    hits = shop.search_products("yoga mat", max_price=20, top_k=5)

    assert [h["asin"] for h in hits] == ["A", "D", "B"]
    by_asin = {h["asin"]: h for h in hits}
    assert by_asin["A"]["price_unknown"] is False
    assert by_asin["B"]["price_unknown"] is True
    assert by_asin["B"]["price"] is None


def test_unpriced_items_survive_budget_filter(fake):
    """回归：这条在旧实现里会返回空列表。"""
    fake([_hit("X", None), _hit("Y", None), _hit("Z", 99.0)])

    hits = shop.search_products("camping tent", max_price=20, top_k=5)

    assert [h["asin"] for h in hits] == ["X", "Y"]


def test_priced_items_come_before_unpriced_under_budget(fake):
    fake([_hit("unknown", None), _hit("cheap", 12.0)])

    hits = shop.search_products("dumbbell", max_price=20, top_k=5)

    assert [h["asin"] for h in hits] == ["cheap", "unknown"]


def test_no_budget_keeps_original_order_and_does_not_filter(fake):
    fake([_hit("unknown", None), _hit("pricey", 99.0), _hit("cheap", 12.0)])

    hits = shop.search_products("dumbbell", top_k=5)

    assert [h["asin"] for h in hits] == ["unknown", "pricey", "cheap"]
    assert [h["price_unknown"] for h in hits] == [True, False, False]


def test_top_k_is_respected(fake):
    fake([_hit(str(i), 10.0) for i in range(10)])

    assert len(shop.search_products("dumbbell", max_price=20, top_k=3)) == 3
    assert len(shop.search_products("dumbbell", top_k=3)) == 3


def test_candidate_k_expands_only_when_budget_given(fake):
    retriever = fake([_hit("A", 10.0)])

    shop.search_products("dumbbell", max_price=20, top_k=5)
    shop.search_products("dumbbell", top_k=5)

    assert [call[1] for call in retriever.calls] == [20, 5]


def test_top_k_is_clamped_to_max(fake):
    retriever = fake([_hit("A", 10.0)])

    shop.search_products("dumbbell", top_k=999)

    assert retriever.calls[0][1] == shop.MAX_TOP_K
