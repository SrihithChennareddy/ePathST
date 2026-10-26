# input_data

Placeholder — the real input data (CPTAC WSIs, HEST WSIs + spatial
transcriptomics) is not stored in this git repo. It's large (CPTAC alone
is ~129GB of DICOM), binary, and lives on shared HPC storage instead.

- **Full details, schema, and provenance**: see
  [`docs/INPUT_DATA.md`](../docs/INPUT_DATA.md).
- **Real on-disk locations**: see `paths.cptac_root`, `paths.cptac_metadata`,
  and `paths.hest_root` in [`configs/base.yaml`](../configs/base.yaml) —
  these are the actual paths every script reads from at run time. This
  folder is not one of them.

This folder exists so the repo has a discoverable, named place for "this is
conceptually where input data would go" — nothing under it is read by any
script.
