# Repository Guidelines

## Project Structure & Module Organization

`kingofgpu/` is the application package. `cli.py` defines subcommands, `monitor.py`
coordinates GPU leases, `gpu.py` reads GPU/process state, `occupier.py` reserves
VRAM, and `config.py` loads configuration. `feishu.py` sends notifications.
Keep behavior-specific changes close to the responsible module. Tests live in
`tests/`; add `test_<feature>.py` files there. `config.example.json` is the only
configuration file that belongs in version control.

## Build, Test, and Development Commands

The monitor uses the standard library; its occupier needs a CUDA-enabled PyTorch
runtime.

```bash
python3 -m kingofgpu --help        # inspect available CLI commands
python3 -m kingofgpu status        # display GPU and reservation state
python3 -m pytest                  # run the full test suite
python3 -m unittest tests.test_monitor  # run the unittest-based monitor test
```

Run tests before committing. Tests should mock GPU, process, and network
interactions; do not require a physical GPU or send Feishu notifications.

## Coding Style & Naming Conventions

Use Python 3 with four-space indentation, type hints for public and non-trivial
internal APIs, and `from __future__ import annotations` in new modules. Follow
PEP 8 naming: `snake_case` for functions, variables, and modules; `PascalCase`
for classes; `UPPER_CASE` for constants. Prefer small, side-effect-aware helpers
and preserve the project’s standard-library-first approach. There is no mandated
formatter or linter; match surrounding code and keep imports grouped as standard
library, then local imports.

## Testing Guidelines

Name tests `test_<behavior>` and make assertions about observable state or CLI
output. Use `TemporaryDirectory` for filesystem state and `unittest.mock.patch`
for `nvidia-smi`, process lookup, subprocesses, and HTTP. Cover both normal and
safety-sensitive paths: only KingOfGpu occupiers may be terminated, and only the
configured user can establish task bindings.

## Commit & Pull Request Guidelines

This repository has no existing commit history, so use focused Conventional
Commit-style subjects, for example `fix: preserve released GPU bindings` or
`test: cover JSON status accounting`. Keep commits small and independently
testable. Pull requests should state the user-visible change, tests run, any
configuration impact, and include sample CLI output when it changes.

## Security & Configuration

Never commit `config.json`, Feishu webhook URLs/secrets, personal filesystem
paths, GPU/job snapshots, logs, or `run/state.json`. Start from
`config.example.json`, replace placeholders locally, and review `git diff --cached`
before every commit.
