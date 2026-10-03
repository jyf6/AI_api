from core.account_pool import BaseAccountPool


def test_scheduler_prefers_least_recently_dispatched_then_lower_inflight():
    pool = BaseAccountPool.__new__(BaseAccountPool)
    accounts = [
        {"name": "recent", "last_dispatched_at": 20.0, "inflight": 0},
        {"name": "older_busy", "last_dispatched_at": 10.0, "inflight": 2},
        {"name": "older_idle", "last_dispatched_at": 10.0, "inflight": 0},
    ]

    assert pool._select_strategy(accounts)["name"] == "older_idle"
