import pandas as pd
import pytest

from pylops_bench.results import tidy, update_results, upsert


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
