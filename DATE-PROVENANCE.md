# Date provenance

Source-import commits use preserved RAR last-modified dates for exact content matches. Original relative paths distinguish identical backup copies. If no unambiguous archive match exists, the latest explicit source-comment date is used. Dates are recorded snapshot/work dates; original editing commits and completion dates cannot be recovered.

Current filesystem dates came from the backup transfer and were rejected as project dates. Archive timestamps may record later saves than code headers. Those later snapshot dates are used when the content matches. Archive dates do not prove completion. Code comments remain unchanged.

Original timezone and exact coding times cannot be established. Author timestamps use 12:00 UTC as a documented placeholder for the supported calendar date. Committer timestamps record the actual import. Undated source and the newly written portfolio documentation use their actual import date. Git backups preserve the prior imported history.

## File evidence

- `src/train.py`: 2021-08-28; hash-matched RAR last-saved date. Archive: `Coding/Q learning/Reinfrocment Learning (Q learning) Basic Model mk01 - Copy.py`.
