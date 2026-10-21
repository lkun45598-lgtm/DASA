import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_all_parallel_tmux.sh"


def test_prepare_queue_contains_all_runs_and_pairs_cost_methods(tmp_path):
    run_root = tmp_path / "formal"
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--prepare-only", str(run_root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    pending = sorted((run_root / ".queue").glob("*.pending"))
    records = [
        line.split("\t")
        for task in pending
        for line in task.read_text(encoding="utf-8").splitlines()
    ]

    assert len(pending) == 30
    assert len(records) == 33
    assert sum(group == "cost" for group, _, _ in records) == 6
    assert sum(group == "sensitivity" for group, _, _ in records) == 27

    cost_units = [task for task in pending if "cost_pair" in task.name]
    assert len(cost_units) == 3
    for seed, task in enumerate(cost_units):
        pair = [
            line.split("\t")
            for line in task.read_text(encoding="utf-8").splitlines()
        ]
        assert [Path(config).stem for _, config, _ in pair] == [
            "cost_pinn",
            "cost_dasa",
        ]
        assert [int(task_seed) for _, _, task_seed in pair] == [seed, seed]

    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    assert 'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in launcher_source
