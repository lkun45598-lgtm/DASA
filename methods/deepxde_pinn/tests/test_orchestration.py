import csv
import json
from pathlib import Path

from methods.deepxde_pinn.scripts.aggregate_results import aggregate_results


ROOT = Path(__file__).resolve().parents[1]


def test_aggregator_uses_exact_columns_and_scientific_notation(tmp_path):
    run = tmp_path / "seed_0" / "burgers"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "case": "burgers",
                "seed": 0,
                "l2_relative_error": 0.012345,
                "maximum_pointwise_error": 0.25,
            }
        ),
        encoding="utf-8",
    )

    rows, failures = aggregate_results(
        tmp_path, tmp_path / "results.csv", tmp_path / "results.md"
    )

    assert len(rows) == 1
    assert failures == []
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        values = list(csv.reader(handle))
    assert values[0] == [
        "case",
        "seed",
        "l2_relative_error",
        "maximum_pointwise_error",
    ]
    assert values[1] == ["burgers", "0", "1.234500e-02", "2.500000e-01"]
    markdown = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "1.234500e-02" in markdown
    assert "2.500000e-01" in markdown


def test_tmux_launcher_defines_130_atomic_resumable_tasks():
    script = (ROOT / "scripts" / "launch_seed_matrix_tmux.sh").read_text(
        encoding="utf-8"
    )
    cases = [
        line.strip()
        for line in script.splitlines()
        if line.startswith("    ") and line.strip() in {
            "advection_diffusion",
            "allen_cahn",
            "burgers",
            "cavity",
            "diffusion_reaction",
            "discontinuous_advection",
            "helmholtz_1d",
            "helmholtz_2d",
            "lshape_poisson",
            "poisson_1d_boundary",
            "poisson_1d_steep",
            "poisson_2d_peak",
            "poisson_2d_steep",
        }
    ]
    assert len(cases) == 13
    assert "seeds=(0 1 2 3 4 5 6 7 8 9)" in script
    assert 'mv "$task" "$running"' in script
    assert '"${claim%.running}.done"' in script
    assert '"${claim%.running}.failed"' in script
    assert "--resume" in script
    assert "CUDA_VISIBLE_DEVICES" in script
