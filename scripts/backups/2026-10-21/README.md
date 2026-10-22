# Original script backups from 21 October 2026

These files are byte-for-byte copies made before the full Office 26 and Kicker 31 experiment workflow was changed. They are retained for rollback and audit purposes. The active scripts remain in `scripts/`.

| Backup | Original active path | SHA-256 |
| --- | --- | --- |
| `run_compare.py.original` | `scripts/run_compare.py` | `104433421D66DFE0FEC08A991AA56C8BA958798F1A3F32703C75DBDC412AC95F` |
| `package_models_for_review.py.original` | `scripts/package_models_for_review.py` | `99A20E016297D74930183DD63E987149374E8928CED790105F51257B72CFF9AC` |
| `inspect_models.py.original` | `scripts/inspect_models.py` | `3F8D33E80138A1FC469C369474C70905AACF91487546243A3093F75E3520B37D` |
| `plot_colmap_dense.original_2026-10-21.py` | `scripts/plot_colmap_dense.py` | `3BD1BC1ADCCB7578DB52A5BF7E47157C1A542BA4315314E59B898D3A61AA8600` |
| `plot_colmap_sfm.original_2026-10-21.py` | `scripts/plot_colmap_sfm.py` | `8D6A6ED425FAEE2A63FA29DAACD5AF06112951E54AB579286BBF5D6B50A4A9F1` |

To verify a backup in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 scripts\backups\2026-10-21\run_compare.py.original
```

Do not copy a backup over an active script unless you intentionally want to roll back all changes made after this snapshot.
