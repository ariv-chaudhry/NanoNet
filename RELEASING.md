# Releasing NanoNet

This document describes how to publish NanoNet to PyPI.

**Import name:** `nanonet` (`import nanonet as nn`)  
**Distribution name:** `nanonet` (`pip install nanonet`)  
**First release:** `0.1.0` (tag `v0.1.0`)

NanoNet is currently pre-1.0. Public APIs may evolve.

---

## Prerequisites

- [ ] `main` is clean and up to date
- [ ] All CI checks are green on the release commit
- [ ] Canonical version in `nanonet/_version.py` matches the intended release
- [ ] `CHANGELOG.md` has a dated `0.1.0` (or later) section — not `Unreleased`
- [ ] README installation section is ready for `pip install nanonet` (commit with or right after publish)
- [ ] PyPI **Trusted Publisher** configured (see below)
- [ ] GitHub Environment `pypi` exists (optional protection rules recommended)
- [ ] Workflow `.github/workflows/publish.yml` is on `main`

---

## PyPI Trusted Publisher setup

Publication uses GitHub Actions OIDC. **Do not create or store a PyPI API token in this repository.**

1. Sign in to [PyPI](https://pypi.org/) (create an account if needed).
2. Open **Publishing** / **Trusted publishers**.
3. Because the project may not exist yet, create a **Pending Trusted Publisher** with:

| Field | Value |
| --- | --- |
| PyPI project name | `nanonet` |
| Owner | `ariv-chaudhry` |
| Repository | `NanoNet` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

4. In GitHub: **Settings → Environments → New environment** named exactly `pypi`.
5. Optionally require reviewers before the publish job runs.

The first successful workflow run after the GitHub Release is published should create/publish the project when the pending publisher matches.

If `nanonet` becomes unavailable before release, stop and choose a new distribution name (import can stay `nanonet`). Alternatives: `nanonet-ml`, `nanonet-framework`, `nanonet-autograd`.

---

## Pre-release checks (local)

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
python -m twine check dist/*
```

Clean-wheel smoke test (use a fresh venv when possible):

```bash
python -m pip install dist/nanonet-0.1.0-py3-none-any.whl
# from a directory outside the repository checkout:
python /path/to/NanoNet/scripts/release_smoke.py
```

---

## Version procedure

1. Edit **only** `nanonet/_version.py` (single source of truth).
2. Confirm `import nanonet; print(nanonet.__version__)`.
3. Update `CHANGELOG.md`:
   - move items under `## [X.Y.Z] - YYYY-MM-DD`
   - add a fresh `## [Unreleased]` section above for future work

Do not duplicate the version in `pyproject.toml` (it uses dynamic versioning).

---

## Release flow (v0.1.0 and later)

Preferred path: **GitHub Release published → Actions builds → OIDC → PyPI**.

1. Ensure pre-release checks pass.
2. Commit release preparation (version already `0.1.0`, changelog dated, docs ready):

   ```text
   chore: prepare v0.1.0 release
   ```

3. Push to `main`.
4. Create and push the tag (or create the tag when drafting the GitHub Release):

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. Create a **GitHub Release** for `v0.1.0` and publish it.
   - Release notes can be copied from `CHANGELOG.md`.
6. Confirm the **Publish** workflow succeeds (build + smoke + PyPI upload).
7. Verify from a **brand-new** environment:

   ```bash
   python -m pip install nanonet==0.1.0
   python -c "import nanonet as nn; print(nn.__version__, nn.__file__)"
   python scripts/release_smoke.py
   ```

8. Update README Installation to:

   ```bash
   pip install nanonet
   ```

   if that change was not already in the release commit.
9. Optionally add a PyPI version badge after the project page exists.

---

## Manual fallback (TestPyPI / emergency)

Trusted Publishing is the primary production path. Manual Twine is only a fallback.

```bash
python -m build
python -m twine check dist/*
# TestPyPI (optional):
python -m twine upload --repository testpypi dist/*
```

Never put tokens in the repository. Use interactive login or environment variables provided by your shell, not committed files.

TestPyPI may lack some dependencies; you may need `--extra-index-url https://pypi.org/simple` when installing from TestPyPI.

---

## Immutability and failures

- PyPI versions are **immutable**. Never republish `0.1.0` with different files.
- If a bug ships after publish, release **`0.1.1`** (or later).
- If the workflow fails **before** upload, fix and re-run / recreate the release as appropriate.
- If a release is fundamentally broken, prefer **yanking** on PyPI over pretending the version never existed.

Tag / version consistency is enforced in CI:

```text
RELEASE_TAG=v0.1.0  →  nanonet.__version__ must be 0.1.0
```

---

## Release checklist

- [ ] Tests pass
- [ ] Lint passes (`ruff check .`)
- [ ] Version correct in `nanonet/_version.py`
- [ ] Changelog dated for this version
- [ ] `python -m build` succeeds
- [ ] `python -m twine check dist/*` passes
- [ ] Clean-wheel install + `scripts/release_smoke.py` succeeds
- [ ] README reviewed for PyPI strangers
- [ ] Distribution name `nanonet` confirmed available / reserved via Trusted Publisher
- [ ] Trusted Publisher pending/active on PyPI
- [ ] GitHub Environment `pypi` configured
- [ ] Tag `vX.Y.Z` created
- [ ] GitHub Release published
- [ ] Publish workflow succeeded
- [ ] `pip install nanonet==X.Y.Z` works in a fresh environment
- [ ] Observability smoke tests pass on the PyPI install
- [ ] README install command updated / verified
- [ ] No attempt to overwrite an existing version
