# PyLops Benchmarks

A zoo of inverse problems built with [PyLops](https://pylops.readthedocs.io),
each solved with different combinations of solvers and
regularizers/preconditioners, and compared on the same metrics.

Every benchmark is defined by:

- a **true model** $x$;
- a PyLops **operator** $Op$;
- the **data** $y = Op\,x\,(+\,n)$, possibly with noise $n$ drawn with a fixed
  seed.

Each solver estimates $\hat{x}$ from $y$ and $Op$, and is scored against the
true model with PyLops' Mean Absolute Error
([`pylops.utils.metrics.mae`](https://pylops.readthedocs.io/en/stable/api/generated/pylops.utils.metrics.mae.html))
and Peak Signal-to-Noise Ratio
([`pylops.utils.metrics.psnr`](https://pylops.readthedocs.io/en/stable/api/generated/pylops.utils.metrics.psnr.html),
computed over the full dynamic range of the true model). Times are the
median wall-clock time of the inversion only (operator construction is
excluded) over the repetitions of a run.

Benchmarks are run with [BenchOpt](https://benchopt.github.io) on GitHub
Actions and their results are stored as CSV files in the
[`results/`](https://github.com/PyLops/pylops-benchmarks/tree/main/results)
folder of the repository.

```{toctree}
:maxdepth: 2

auto_benchmarks/index
contributing
```
