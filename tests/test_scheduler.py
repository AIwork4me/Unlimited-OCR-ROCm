"""Load-balanced multi-GPU sharding (replaces round-robin)."""

from rocm_ocr.scheduler import balance_shards, estimate_cost, write_shard_files


def _make(tmp_path, name, size):
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return str(p)


def test_estimate_cost_monotonic_in_size(tmp_path):
    small = _make(tmp_path, "s.png", 1000)
    big = _make(tmp_path, "b.png", 50000)
    assert estimate_cost(big) > estimate_cost(small)


def test_balance_shards_minimizes_max_load(tmp_path):
    """Largest-first greedy balances total cost across shards."""
    paths = [_make(tmp_path, f"p{i}.png", (i + 1) * 1000) for i in range(10)]
    shards = balance_shards(paths, num_shards=3)
    assert len(shards) == 3
    assert sum(len(s) for s in shards) == 10
    # No shard gets all the big pages: max load <= ~50% of total.
    loads = [sum(estimate_cost(p) for p in s) for s in shards]
    assert max(loads) < sum(loads) * 0.5


def test_write_shard_files_round_trip(tmp_path):
    paths = [_make(tmp_path, f"p{i}.png", 1000) for i in range(4)]
    shards = balance_shards(paths, num_shards=2)
    out = write_shard_files(shards, str(tmp_path / "shards"))
    assert len(out) == 2
    all_again = []
    for f in out:
        with open(f) as file:
            all_again.extend(line.strip() for line in file if line.strip())
    assert sorted(all_again) == sorted(paths)
