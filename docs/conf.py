"""Sphinx configuration for the PyLops Benchmarks website."""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

project = "PyLops Benchmarks"
author = "PyLops Development Team"
copyright = f"{datetime.date.today().year}, {author}"

extensions = [
    "myst_parser",
    "generate_benchmarks",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# -- generate_benchmarks ----------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
benchmarks_dir = str(repo_root / "benchmarks")
results_dir = str(repo_root / "results")
estimates_dir = str(repo_root / "estimates")
github_url = "https://github.com/PyLops/pylops-benchmarks"
github_branch = "main"

# -- HTML -------------------------------------------------------------------
html_theme = "shibuya"
html_title = "PyLops Benchmarks"
html_short_title = "PyLops Benchmarks"
html_static_path = ["_static"]
html_css_files = ["sortable.css"]
html_js_files = ["sortable.js"]
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%b %d, %Y"
html_theme_options = {
    "accent_color": "teal",
    "github_url": github_url,
    "light_logo": "_static/pylops_b.png",
    "dark_logo": "_static/pylops.png",
}
