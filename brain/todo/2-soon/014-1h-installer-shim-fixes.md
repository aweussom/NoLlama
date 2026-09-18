# T-014 — Two shim fixes from the fresh-Ryzen first-contact test

**Effort:** 1 h · **Produces:** `install-windows.bat` + `install.ps1` changes ·
**Area:** installers

Live test of `install-windows.bat` on a fresh Win11 box (Ryzen 5950X / RX580),
2026-08-11: the pwsh prompt, winget install, Store-stub detection and both
re-run paths all worked. Two things did not.

- [ ] **pwsh probe misses a per-user install.** The direct-continue probe only
      checks `%ProgramFiles%\PowerShell\7\pwsh.exe`; winget did a
      per-user/MSIX install, so the `.bat` fell back to "close and re-run".
      Also probe `%LocalAppData%\Microsoft\WindowsApps\pwsh.exe`.
- [ ] **Do not make Python the user's problem.** NoLlama needs Python; the user
      should not need to know. Cheap version: install via winget automatically
      (a notice, not a Y/N prompt), then skip the second re-run by probing
      `%LocalAppData%\Programs\Python\Python3xx\python.exe` and passing it
      to `install.ps1` through a new `-PythonExe` parameter. `install.ps1`
      currently only does `Get-Command python/python3`.

The thorough answer to the second one is `uv` — see T-023, which is a
minor-version project rather than a patch.

## Done when

A fresh Windows box goes from double-click to a running server with at most one
re-run, and no question about Python.
