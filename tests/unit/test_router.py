# tests/unit/test_router.py
import pytest
from app.graph.nodes.router import route_node


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Summarize this contract", "summary"),
        ("Give me an overview of the agreement", "summary"),
        ("What are the risks in this contract?", "risk"),
        ("Is there a liability cap?", "risk"),
        ("What is the termination notice period?", "risk"),
        ("Who are the parties?", "qa"),
        ("What is the effective date?", "qa"),
    ],
)
def test_router_heuristic(query, expected):
    out = route_node({"query": query, "mode": "auto"})
    assert out["mode"] == expected


@pytest.mark.parametrize("mode", ["qa", "summary", "risk"])
def test_router_respects_explicit_mode(mode):
    out = route_node({"query": "irrelevant", "mode": mode})
    assert out["mode"] == mode
