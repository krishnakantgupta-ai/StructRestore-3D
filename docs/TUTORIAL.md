# StructRestore-3D: Installation, Usage, Reproducibility, and PyPI Publishing

> **Project status:** Experimental research prototype (`0.1.0a1`). The software is an *auditable feasibility benchmark*, **not a validated high-fidelity FIB-SEM denoiser**. The current models have not outperformed the added-noise baseline in the reported reference-fidelity evaluation.
>
> **Repository:** https://github.com/krishnakantgupta-ai/StructRestore-3D  
> **Dataset:** EMPIAR-10311, https://doi.org/10.6019/EMPIAR-10311  
> **Maintainer:** Krishna Kant Gupta, RGITBT, Bharati Vidyapeeth (Deemed to be University), Pune, India.

## Contents

1. [What the package does](#1-what-the-package-does)
2. [Install it on a computer](#2-install-it-on-a-computer)
3. [Prepare FIB-SEM data](#3-prepare-fib-sem-data)
4. [Quick start: audit the masking procedure](#4-quick-start-audit-the-masking-procedure)
5. [Train a Version 1 model](#5-train-a-version-1-model)
6. [Benchmark and diagnose existing checkpoints](#6-benchmark-and-diagnose-existing-checkpoints)
7. [Registration probe](#7-registration-probe)
8. [How to make and release a pip package](#8-how-to-make-and-release-a-pip-package)
9. [Troubleshooting](#9-troubleshooting)
10. [Research limitations, reproducibility, and citation](#10-research-limitations-reproducibility-and-citation)

## 1. What the package does

StructRestore-3D provides Python command-line tools for investigating self-supervised restoration of serial FIB-SEM sections. The initial prototype implements:

- **Single-slice CNN:** reconstructs masked target pixels using context from one section.
- **Plain 2.5D CNN:** uses three adjacent sections as input channels.
- **Heuristic reliability-gated 2.5D CNN:** weights adjacent-section information using fixed local intensity agreement; the gate is **not** learned attention.
- **Audited Version 1 masking:** excludes the original target value at selected spatial coordinates across all three input channels, with independent random filler.
- **Benchmarking:** reports PSNR, SSIM, and edge RMSE against a *natively noisy* reference, with added synthetic Gaussian noise.
- **Inference and alignment diagnostics:** compare masked/unmasked inference and small integer translations of adjacent sections.

The package provides *code only*. FIB-SEM images and trained checkpoints must be supplied separately.

### Available commands

| Command | Purpose |
| --- | --- |
| `structrestore3d --help` | List commands |
| `structrestore3d audit-v1` | Check direct target-value exclusion in masked inputs |
| `structrestore3d train-v1` | Train single, plain, or gated Version 1 model |
| `structrestore3d compare` | Compare six existing V0/V1 checkpoints and classical filters |
| `structrestore3d diagnose` | Compare masked and unmasked V1 inference |
| `structrestore3d register-probe` | Inference-only adjacent-section translation experiment |
| `structrestore3d mask-audit` | Audit the initial masking implementation |
| `structrestore3d train-single-v0` | Original full-resolution single-slice training |
| `structrestore3d train-25d-v0` | Original plain/gated 2.5D training |
| `structrestore3d train-baseline` | Initial baseline training |
| `structrestore3d infer-baseline` | Initial baseline inference |
| `structrestore3d validate-25d` | Initial quantitative validation |

Use `structrestore3d COMMAND --help` for command-specific arguments.

## 2. Install it on a computer

### 2.1 Choose Python

The package metadata supports **Python >=3.10,<3.13**. The development experiments used **Python 3.12**, **PyTorch 2.2.2**, and Apple M3 / MPS. Other OS/GPU combinations have **not** been fully validated.

On macOS or Linux, create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, if Python 3.12 is installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.2 Install from GitHub (works before PyPI publication)

```bash
python -m pip install 'git+https://github.com/krishnakantgupta-ai/StructRestore-3D.git'
structrestore3d --help
```

This installs the GitHub revision visible at installation time and resolves the declared dependencies. To install a reproducible revision, pin a release tag or full Git commit hash instead of the moving `main` branch.

### 2.3 Install from PyPI (only **after** a successful PyPI release)

```bash
python -m pip install structrestore-3d
structrestore3d --help
```

**GitHub publication does not automatically create a PyPI release.** Verify the actual release at https://pypi.org/project/structrestore-3d/ before recommending the PyPI command.

### 2.4 Developers: install from the cloned repository

```bash
git clone https://github.com/krishnakantgupta-ai/StructRestore-3D.git
cd StructRestore-3D
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

**For the maintainer's existing Apple M3 environment only:** to preserve a previously working PyTorch 2.2.2/MPS installation, first activate that environment and use:

```bash
python -m pip install -e . --no-deps
```

`--no-deps` skips dependency installation and must **not** be recommended to first-time users unless all required packages are already installed.

## 3. Prepare FIB-SEM data

The pilot used 16 consecutive 8-bit grayscale TIFF sections of a HeLa FIB-SEM volume (EMPIAR-10311). The source frames used in the experiments were **3000 x 7000 pixels**. Obtain data in accordance with the archive's instructions; this repository does not redistribute the microscopy images.

For the **paper-reproduction workflow**, arrange the files as follows:

```text
StructRestore-3D/
├── data/
│   └── raw/
│       ├── slice_0000.tif
│       ├── slice_0001.tif
│       ├── ...
│       └── slice_0015.tif
├── scripts/                      # Original research scripts
├── src/structrestore3d/         # Packaged source code
├── results/                     # Outputs and checkpoints (local only)
└── pyproject.toml
```

The source study used:

- **Training:** sections `0000`–`0011`.
- **Held out:** sections `0012`–`0015`.
- **Central held-out sections for 2.5D evaluation:** `0013` and `0014` (so both neighboring sections remain in the held-out subset).

These are consecutive sections from **one specimen volume**, not independent biological replicates. The experiment scripts expect specific filenames and section counts, so their defaults are not a generic dataset loader. Run commands from the project root when using relative data/checkpoint paths.

**Do not commit raw data or trained checkpoints.** Ensure `.gitignore` contains `data/`, `results/`, `models/`, `.venv/`, `*.tif`, `*.tiff`, and `*.pt`.

## 4. Quick start: audit the masking procedure

Run this command from any directory after installing the package and its dependencies:

```bash
structrestore3d audit-v1
```

Expected excerpts:

```text
single: direct target gradient=0; PASS
plain: direct target gradient=0; PASS
gated: direct target gradient=0; PASS
PASS: masked selected coordinates excluded from all input channels; direct gradients zero.
```

**Interpretation:** this only verifies *direct* dependence at tested masked coordinates. Spatially neighboring pixels and adjacent sections remain correlated, so this is **not** a proof of noise independence or denoising effectiveness.

## 5. Train a Version 1 model

First change to the project root, with the raw images in `data/raw/` and the virtual environment active:

```bash
cd ~/Downloads/StructRestore-3D    # Update for your own clone location
source .venv/bin/activate
```

Train the three model variants using different output paths:

```bash
structrestore3d train-v1 --input data/raw --mode single --out results/v1_single --steps 500 --patch 128 --batch 2
structrestore3d train-v1 --input data/raw --mode plain  --out results/v1_plain  --steps 500 --patch 128 --batch 2
structrestore3d train-v1 --input data/raw --mode gated  --out results/v1_gated  --steps 500 --patch 128 --batch 2
```

**Output safety:** the research code rejects nonempty output directories. If you already have results at those paths, specify *new* ones (e.g., `results/replication_v1_single`). Never delete previously published experiment results just to rerun a command.

Each successful run writes:

```text
results/v1_gated/
├── model.pt     # Research checkpoint; not included in pip package
├── history.csv  # Training loss and patch-selection diagnostics
└── run.json     # Experimental configuration and limitations
```

To check the run on macOS/Linux:

```bash
cat results/v1_gated/run.json
head -n 10 results/v1_gated/history.csv
```

The experimental study used a minimum normalized patch standard deviation of `0.04` and shared-coordinate masking (nominal mask rate `0.05`). Even when training loss decreases, it is **not** evidence that fine cellular structures have been recovered correctly.

## 6. Benchmark and diagnose existing checkpoints

These workflows require trained checkpoint files. In particular, `compare` needs **all six** V0 and V1 files in their expected locations:

```text
results/baseline_fullres/baseline.pt
results/25d_plain/model.pt
results/25d_gated/model.pt
results/v1_single/model.pt
results/v1_plain/model.pt
results/v1_gated/model.pt
```

Check the paths before running the comparison:

```bash
structrestore3d compare --help
```

Run the small synthetic-noise benchmark (if the six files are present):

```bash
structrestore3d compare \
  --input data/raw \
  --out results/tutorial_comparison \
  --crop 256 \
  --trials 1 \
  --sigmas 0.05 \
  --groups 20
```

The results include `summary.csv`, `per_trial.csv`, `comparison.png`, and `metadata.json`.

To diagnose **the same** V1 checkpoints under masked versus unmasked inference:

```bash
structrestore3d diagnose \
  --input data/raw \
  --out results/tutorial_diagnostic \
  --slice-index 13 \
  --region 0 \
  --crop 256 \
  --groups 20 \
  --repeats 2 \
  --sigma 0.05
```

Results include a nine-panel comparison image, numerical summaries, and mask-variability measurements. **PSNR/SSIM are computed relative to the original, naturally noisy image**; they test fidelity after *additional* synthetic corruption and cannot establish removal of the original acquisition noise.

### Benchmark numbers reported in the proof of concept

For added Gaussian noise `sigma=0.05` (means from six correlated held-out regions):

| Method | PSNR (dB), higher is better | SSIM, higher is better | Edge RMSE, lower is better |
|---|---:|---:|---:|
| Added-noise input | **27.13** | **0.6747** | **0.0429** |
| Gaussian filtering | 23.66 | 0.4906 | 0.0782 |
| Median filtering | 23.77 | 0.5115 | 0.0931 |
| V0 single | 23.31 | 0.3523 | 0.0969 |
| V0 plain | 23.56 | 0.3781 | 0.0952 |
| V0 gated | 22.69 | 0.3588 | 0.0948 |
| V1 single | 22.16 | 0.2624 | 0.1039 |
| V1 plain | 22.20 | 0.2778 | 0.1048 |
| V1 gated | 22.90 | 0.3025 | 0.1038 |

**Do not interpret these scores as a clean-ground-truth denoising benchmark.** The reference contains unknown native noise. The current models also visibly softened fine ultrastructure in example images.

## 7. Registration probe

The translation-only registration experiment uses four existing plain/gated 2.5D checkpoints; it **does not retrain** any model:

```bash
structrestore3d register-probe --help
```

Run a lightweight exploratory comparison:

```bash
structrestore3d register-probe \
  --input data/raw \
  --out results/tutorial_registration \
  --crop 256 \
  --shift-radius 6 \
  --groups 20 \
  --regions 1 \
  --sigmas 0.05
```

Registration is estimated using synthetic-noise-affected inputs, and shifting adjacent images at inference without registration-aware training is a methodological limitation. Increased correlation does **not** by itself prove better anatomical alignment or restoration quality.

## 8. How to make and release a pip package

This part is primarily for the maintainer preparing a new version; ordinary users can stop after installation and running the commands above.

### 8.1 Synchronize the source tree

After editing any original file in `scripts/`, copy it into the package directory:

```bash
cd ~/Downloads/StructRestore-3D
python tools/sync_package.py
```

The authoritative script copies are kept under `scripts/`; the pip-distributed duplicates are under `src/structrestore3d/`. **Always synchronize before building or committing a release**, or pip users may receive older code.

### 8.2 Check versions before each new release

The current experimental version is `0.1.0a1`. Both fields must agree:

- `version = "0.1.0a1"` in `pyproject.toml`.
- `__version__ = "0.1.0a1"` in `src/structrestore3d/__init__.py`.

For a **new** published release, update both values together (for example, a subsequent prerelease version). **PyPI does not allow overwriting an already published version.**

### 8.3 Build distributions and validate metadata

Run:

```bash
python -m pip install build twine
rm -rf build dist
python -m build
python -m twine check dist/*
```

Expected artifacts:

```text
dist/structrestore_3d-0.1.0a1.tar.gz
dist/structrestore_3d-0.1.0a1-py3-none-any.whl
```

Replace the version in those example names after updating the package version. Validate that the registration module is in the wheel:

```bash
python -m zipfile -l dist/*.whl | grep step29_registration_probe
```

### 8.4 Smoke-test the built wheel outside the source tree

This test uses a separate target directory while retaining the maintainer's existing Python environment and already installed dependencies:

```bash
python -m pip install --no-deps --target /tmp/structrestore-wheel-test dist/*.whl
cd /tmp
PYTHONPATH=/tmp/structrestore-wheel-test python -m structrestore3d register-probe --help
PYTHONPATH=/tmp/structrestore-wheel-test python -m structrestore3d audit-v1
```

For comprehensive release validation, also test in a **clean virtual environment with dependencies installed normally**. The target-directory smoke test alone does not prove fresh dependency resolution or cross-platform compatibility.

### 8.5 Commit and push source code to GitHub

Return to the repository and review changes:

```bash
cd ~/Downloads/StructRestore-3D
git status --short
git add pyproject.toml src/ scripts/ tools/ docs/ README.md .gitignore
git diff --cached --stat
git commit -m "Document and package StructRestore-3D"
git push origin main
```

Only commit files you have inspected. The `.gitignore` must continue to exclude `.venv/`, `data/`, `results/`, `models/`, `dist/`, `build/`, `*.egg-info/`, and secrets such as `.env`. **Do not commit PyPI tokens or image/checkpoint datasets.** If Git says there is nothing to commit, no new commit is needed.

### 8.6 Create a PyPI account and publish

1. Register at https://pypi.org/account/register/ and verify the email address.
2. Configure two-factor authentication.
3. Configure trusted publishing or create an API token. Keep credentials private.
4. Confirm package metadata and release version, and review the files in `dist/`.
5. Upload the tested artifacts:

```bash
cd ~/Downloads/StructRestore-3D
python -m twine upload dist/*
```

For token-based authentication, enter `__token__` as the username and paste the API token at the password prompt. **Never paste a secret into a GitHub file, tutorial, or chat.**

After a successful upload, verify the release at:

https://pypi.org/project/structrestore-3d/

Then, and only then, advertise:

```bash
python -m pip install structrestore-3d
```

**Optional:** trial an upload with https://test.pypi.org/ first. TestPyPI requires its own account/token and may not host all package dependencies; installation experiments against TestPyPI may require choosing PyPI as the dependency index.

## 9. Troubleshooting

| Issue | What to check |
|---|---|
| `command not found: structrestore3d` | Activate the virtual environment where pip installed the package; run `python -m pip show structrestore-3d`. |
| `No module named torch` | Install dependencies in the active environment. Do not use `--no-deps` in a fresh environment unless you separately install the dependencies. |
| `The packaged script step29_registration_probe is missing` | Confirm `scripts/step29_registration_probe.py` exists; run `python tools/sync_package.py`, rebuild and reinstall. |
| `FileNotFoundError` for TIFF or checkpoints | Run from project root and supply valid `--input` and `--models-root` paths; datasets and checkpoints are never bundled with pip. |
| Output directory is not empty | Choose a new unique `--out` directory instead of overwriting an existing experiment. |
| `twine upload` authentication failure | Check PyPI versus TestPyPI credentials and the correct token-scope/project permissions. |
| PyPI rejects a version | Published versions are immutable; increment package version, rebuild, check, and upload a new release. |
| PyPI reports name unavailable | The distribution name may be taken; choose an unused package name and update the project metadata. |
| MPS/PyTorch error on Apple Silicon | Use a tested Python/PyTorch combination; avoid upgrading a functioning research environment without validation. |

## 10. Research limitations, reproducibility, and citation

This repository is intentionally transparent about its current boundaries:

- The pilot uses 16 successive frames from **one** EMPIAR volume.
- Direct masked target-gradient checks pass; **indirect statistical dependence remains possible**.
- Noise from serial acquisition may violate assumptions of simple blind-spot learning.
- Version 0 and Version 1 evaluations have different reconstruction protocols.
- The reported baseline scores measure closeness to a **native-noise reference**, not a verified clean image.
- The code presently provides **feasibility and failure-mode evidence, not demonstrated high-fidelity restoration**.
- The alignment probe does not establish a clinical, biological, or statistically generalized result.

For reproducibility, cite the EMPIAR dataset and the exact software release/commit used. Before citing a publication, verify that the manuscript is actually accepted/published. If available, use an archived version DOI (e.g., via an authorized Zenodo release) rather than only a moving GitHub branch.

**Issues and questions:** https://github.com/krishnakantgupta-ai/StructRestore-3D/issues

---

*Tutorial updated: October 2026. Commands are based on experimental package version 0.1.0a1 and the current public repository layout.*
