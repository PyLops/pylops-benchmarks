import numpy as np
import pandas as pd
import pytest

from pylops_bench.results import (
    estimate_key,
    key_columns,
    tidy,
    update_results,
    upsert,
)


def _raw(niter_values, psnr=30.0, run_date="2026-01-01T10:00:00", nrep=1):
    rows = []
    for niter in niter_values:
        for rep in range(nrep):
            rows.append(
                {
                    "objective_name": "Reconstruction",
                    "dataset_name": "Toy[seed=0]",
                    "solver_name": f"LSQR[niter={niter}]",
                    "p_dataset_seed": 0,
                    "p_solver_niter": niter,
                    "objective_MAE": 0.1 + rep,
                    "objective_PSNR": psnr,
                    "time": 1.0 + rep,
                    "idx_rep": rep,
                    "run_date": run_date,
                }
            )
    rows.append(
        {
            "objective_name": "Reconstruction",
            "dataset_name": "Toy[seed=0]",
            "solver_name": "FISTA[eps=0.1]",
            "p_dataset_seed": 0,
            "p_solver_eps": 0.1,
            "objective_MAE": 0.2,
            "objective_PSNR": psnr,
            "time": 2.0,
            "idx_rep": 0,
            "run_date": run_date,
        }
    )
    return pd.DataFrame(rows)


def test_tidy_aggregates_repetitions():
    out = tidy(_raw([10], nrep=3))
    lsqr = out[out.solver == "LSQR"].iloc[0]
    assert len(out) == 2
    assert lsqr.n_repetitions == 3
    assert lsqr.MAE == pytest.approx(1.1)
    assert lsqr.time_s == pytest.approx(2.0)
    assert {"pylops_version", "benchopt_version", "run_date"} <= set(out.columns)


def test_tidy_requires_metrics():
    with pytest.raises(KeyError):
        tidy(_raw([10]).drop(columns="objective_PSNR"))


def test_upsert_replaces_and_appends(tmp_path):
    raw1 = tmp_path / "raw1.csv"
    _raw([10]).to_csv(raw1, index=False)
    first = update_results(raw1, "toy", tmp_path)
    assert len(first) == 2

    # rerun the same configuration: rows are replaced, not duplicated
    raw2 = tmp_path / "raw2.csv"
    _raw([10], psnr=40.0, run_date="2026-02-01T10:00:00").to_csv(raw2, index=False)
    second = update_results(raw2, "toy", tmp_path)
    assert len(second) == 2
    assert (second.PSNR == 40.0).all()
    assert (second.run_date == "2026-02-01 10:00:00").all()

    # a new solver parameter value adds a row
    raw3 = tmp_path / "raw3.csv"
    _raw([20]).to_csv(raw3, index=False)
    third = update_results(raw3, "toy", tmp_path)
    assert len(third) == 3
    assert len(pd.read_csv(tmp_path / "toy.csv")) == 3


def test_dry_run_does_not_write(tmp_path):
    raw = tmp_path / "raw.csv"
    _raw([10]).to_csv(raw, index=False)
    update_results(raw, "toy", tmp_path, dry_run=True)
    assert not (tmp_path / "toy.csv").exists()


def test_upsert_without_existing():
    new = tidy(_raw([10]))
    assert upsert(None, new).equals(upsert(pd.DataFrame(), new))


def _raw_with_estimates(raw: pd.DataFrame, value=1.0) -> pd.DataFrame:
    raw = raw.copy()
    raw["final_results"] = [np.full(4, value + i) for i in range(len(raw))]
    return raw


def _write_parquet(raw: pd.DataFrame, path):
    from benchopt.results import to_parquet
    from benchopt.results.parquet import pack

    raw = raw.copy()
    raw["final_results"] = raw["final_results"].map(pack)
    to_parquet(raw, path)
    return path


def test_estimates_are_saved(tmp_path):
    raw = _write_parquet(
        _raw_with_estimates(_raw([10], nrep=2)), tmp_path / "r.parquet"
    )
    table = update_results(raw, "toy", tmp_path, estimates_dir=tmp_path / "est")
    assert table["estimate"].notna().all()
    files = sorted(p.name for p in (tmp_path / "est" / "toy").glob("*.npz"))
    assert files == sorted(table["estimate"])
    lsqr = table[table.solver == "LSQR"].iloc[0]
    x = np.load(tmp_path / "est" / "toy" / lsqr["estimate"])["x"]
    # last repetition, in single precision
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x, np.full(4, 2.0))


def test_estimate_key_is_stable():
    raw = _raw([10])
    new = tidy(raw)
    keys = key_columns(new)
    row = new[new.solver == "LSQR"].iloc[0]
    # the key does not depend on parameter columns added by other solvers,
    # nor on integers read back as floats from the CSV
    wider = row.copy()
    wider["solver_param_tau"] = np.nan
    wider["dataset_param_seed"] = 0.0
    assert estimate_key(row, keys) == estimate_key(wider, keys + ["solver_param_tau"])
    other = new[new.solver == "FISTA"].iloc[0]
    assert estimate_key(row, keys) != estimate_key(other, keys)


def test_estimates_are_replaced_and_pruned(tmp_path):
    est = tmp_path / "est"
    raw1 = _write_parquet(_raw_with_estimates(_raw([10])), tmp_path / "r1.parquet")
    first = update_results(raw1, "toy", tmp_path, estimates_dir=est)

    # rerunning a configuration overwrites its file
    raw2 = _write_parquet(
        _raw_with_estimates(_raw([10]), value=5.0), tmp_path / "r2.parquet"
    )
    second = update_results(raw2, "toy", tmp_path, estimates_dir=est)
    assert sorted(first["estimate"]) == sorted(second["estimate"])
    lsqr = second[second.solver == "LSQR"].iloc[0]
    assert np.load(est / "toy" / lsqr["estimate"])["x"][0] == 5.0

    # a stale file, not referenced by the table, is removed
    (est / "toy" / "stale.npz").write_bytes(b"")
    update_results(raw2, "toy", tmp_path, estimates_dir=est)
    assert not (est / "toy" / "stale.npz").exists()
    assert len(list((est / "toy").glob("*.npz"))) == 2


def test_estimates_dry_run_and_csv(tmp_path):
    est = tmp_path / "est"
    raw = _write_parquet(_raw_with_estimates(_raw([10])), tmp_path / "r.parquet")
    update_results(raw, "toy", tmp_path, estimates_dir=est, dry_run=True)
    assert not est.exists()
    assert not (tmp_path / "toy.csv").exists()

    # CSV raw results have no estimates
    raw_csv = tmp_path / "raw.csv"
    _raw([10]).to_csv(raw_csv, index=False)
    table = update_results(raw_csv, "toy", tmp_path, estimates_dir=est)
    assert "estimate" not in table
    assert "_estimate" not in pd.read_csv(tmp_path / "toy.csv")
