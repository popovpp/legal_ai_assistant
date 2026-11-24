# tests/unit/test_cache.py
from app.cache import InMemoryLRU


def test_lru_put_get():
    c = InMemoryLRU(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    assert c.get("b") == 2
    assert c.get("missing") is None


def test_lru_evicts_oldest():
    c = InMemoryLRU(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # должно вытеснить "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_lru_get_promotes():
    c = InMemoryLRU(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")  # a становится "свежим"
    c.put("c", 3)  # должен вытеснить b
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_make_key_is_stable_and_order_sensitive():
    k1 = InMemoryLRU.make_key("doc1", "q", "risk", 10)
    k2 = InMemoryLRU.make_key("doc1", "q", "risk", 10)
    k3 = InMemoryLRU.make_key("q", "doc1", "risk", 10)
    assert k1 == k2
    assert k1 != k3
