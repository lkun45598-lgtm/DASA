# Excluded runs

The raw checkpoints, logs, and metric files remain unchanged. The following two
baseline runs are excluded from the reported aggregate as requested because their
final relative L2 errors are outliers relative to the other baseline runs:

| Configuration | Seed | Relative L2 error |
|---|---:|---:|
| `paper_stage_length_baseline` | 0 | `1.413282e-02` |
| `paper_stage_length_baseline` | 1 | `1.063677e-02` |

No run is excluded from the other stage-length configurations. The baseline aggregate
therefore uses eight seeds; every other configuration uses ten seeds.
