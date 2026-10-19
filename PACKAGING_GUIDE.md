# Publish StructRestore-3D as a pip-installable research package

This is **research preview software**, not a validated denoising solution. The package contains code only, not EMPIAR microscopy data or trained models.

## 1. Set up the package (macOS Terminal)

```bash
cd ~/Downloads/StructRestore-3D
unzip -o ~/Downloads/StructRestore3D_PyPI_Kit.zip -d .
python tools/sync_package.py
```

This copies the **current** Python files from `scripts/` into the new installable module in `src/structrestore3d/`. If you change a research script later, repeat the sync command and commit the updated package files.

## 2. Install locally (does not change existing torch)

```bash
source .venv/bin/activate
python -m pip install -e . --no-deps
structrestore3d --help
structrestore3d audit-v1
```

`--no-deps` is intentional for the Apple M3 research environment because the already-working PyTorch 2.2.2 MPS configuration should be preserved. Make sure dependencies are installed in the virtual environment already.

A local install is not a PyPI release. The local project directory remains the place where images and checkpoint paths are resolved. Run train/evaluation commands **from your project root**.

## 3. Test commands

```bash
structrestore3d train-v1 --help
structrestore3d train-v1 --input data/raw --mode single --out results/pkg_v1_single --steps 100 --patch 128 --batch 2
```

Don't overwrite `results/v1_single` or other existing experiments. Model outputs are not bundled with pip.

## 4. Publish the packaging files to GitHub

Ensure `.gitignore` includes build outputs:

```text
dist/
build/
*.egg-info/
```

Then run:

```bash
git add pyproject.toml src/ tools/ PACKAGING_GUIDE.md .gitignore
git commit -m "Package StructRestore-3D for pip installation"
git push origin main
```

Now someone can install directly from your GitHub source:

```bash
python -m pip install 'git+https://github.com/krishnakantgupta-ai/StructRestore-3D.git'
```

Run in a **fresh** Python 3.12 virtual environment; this install will install needed dependencies, including PyTorch.

## 5. Optional: build and publish to PyPI

Create and verify an account at https://pypi.org/ ; verify that the package name `structrestore-3d` is available. The name is not reserved merely because the GitHub repository exists.

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Use https://test.pypi.org/ first if you want to trial the upload (it uses a different account and API token):

```bash
python -m twine upload --repository testpypi dist/*
```

For the public PyPI release:

```bash
python -m twine upload dist/*
```

Do not put your API token in Git or share it in chat. For Twine's token authentication, use username `__token__` and the token as password when requested. Once PyPI confirms publication, users can install with:

```bash
python -m pip install structrestore-3d
```

To release updates, change `version` in `pyproject.toml` and `__version__` in `src/structrestore3d/__init__.py`, rebuild and publish a new version. PyPI does not allow overwriting an existing version.

## Scientific/reproducibility note

- The code depends on the user's own FIB-SEM data and checkpoints for inference; pip installation alone does not perform restoration.
- The original pilot did not outperform the added-noise input baseline; present this as a feasibility and audit package rather than a validated high-quality denoising tool.
- The project has been tested on Apple M3 with Python 3.12/PyTorch 2.2.2, but no cross-platform compatibility claim is made.
