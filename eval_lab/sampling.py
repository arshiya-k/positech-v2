"""Stratified random sampling shared by the Q&A and sentiment benchmarks."""
import numpy as np
import pandas as pd


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Split `n` draws across strata in proportion to their size (largest-remainder method).

    Every stratum gets at least one draw when `n` allows it, and no stratum is
    asked for more rows than it has.
    """
    sizes = sizes[sizes > 0].astype(int)
    n = min(int(n), int(sizes.sum()))
    alloc = pd.Series(0, index=sizes.index, dtype=int)
    if n >= len(sizes):
        alloc[:] = 1
    while (remaining := n - int(alloc.sum())) > 0:
        capacity = sizes - alloc
        share = capacity / capacity.sum() * remaining
        add = np.minimum(np.floor(share).astype(int), capacity)
        leftover = remaining - int(add.sum())
        if leftover > 0:
            remainders = (share - add)[capacity > add].sort_values(ascending=False)
            add[remainders.index[:leftover]] += 1
        alloc += add
    return alloc


def _by(strata: list[str]):
    return strata if len(strata) > 1 else strata[0]


def stratified_sample(df: pd.DataFrame, n: int, strata: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = df.groupby(_by(strata), dropna=False)
    alloc = allocate(groups.size(), n)
    parts = [
        groups.get_group(key).sample(n=int(k), random_state=int(rng.integers(2**31)))
        for key, k in alloc.items()
        if k > 0
    ]
    return pd.concat(parts) if parts else df.head(0)


def representativeness(population: pd.DataFrame, sample: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    """Population vs sample share per stratum, plus the design weight that turns
    sample averages back into population estimates."""
    pop = population.groupby(strata, dropna=False).size().rename("population_n")
    smp = sample.groupby(strata, dropna=False).size().rename("sample_n")
    out = pd.concat([pop, smp], axis=1).fillna(0).astype(int).reset_index()
    out["population_share"] = out["population_n"] / out["population_n"].sum()
    out["sample_share"] = out["sample_n"] / max(out["sample_n"].sum(), 1)
    out["design_weight"] = np.where(out["sample_n"] > 0, out["population_n"] / out["sample_n"].replace(0, 1), np.nan)
    return out
