# Template

Copy this folder to `benchmarks/<your_problem>/` to start a new benchmark.
This file is shown at the top of the benchmark page on the website: describe
the true model, the operator, the data (and noise), and list the solvers.
Keep it generic: details of each dataset (sizes, sampling, content) belong in
the dataset docstring, shown in the results section of that dataset.

The template inverts a blocky 1D signal blurred by a Gaussian kernel with
damped CGLS.
