# Copilot Instructions for llama.cpp

## Project Scope

This workspace is a custom Covecube branch of llama.cpp focused on CPU inferencing fixes and optimizations. Preserve `--cpu-range` and `--cpu-mask` behavior through upstream's centralized common threadpool lifecycle, and preserve Cascade Lake or x86 CPU-path changes unless the task explicitly requires altering them.

Fork-only commit history shows this branch exists primarily to preserve three things while merging upstream: correct centralized threadpool ownership for CPU affinity controls, Intel-focused Docker and compiler support in `.devops/intel.Dockerfile`, and AVX-512 or VNNI quant-kernel optimizations for Cascade Lake class CPUs. Treat changes in those areas as intentional branch behavior, not incidental drift.

Prefer minimal, performance-aware changes. Avoid new dependencies unless they are clearly necessary.

## Architecture

- `src/` and `include/llama.h` are the core libllama implementation and public C API.
- `ggml/` is the vendored tensor backend and performance-critical dependency.
- `common/` contains shared helpers used by examples and tools.
- `tools/server/` is the OpenAI-compatible HTTP server.
- `examples/` and `tools/` contain the CLI frontends, benchmarks, conversion tools, and test utilities.

## Build And Test

- Use CMake. Do not use the deprecated Makefile.
- Primary Linux build: `cmake -B build` then `cmake --build build --config Release -j $(nproc)`.
- Primary test run: `ctest --test-dir build --output-on-failure -j $(nproc)`.
- For server changes, build the relevant target and follow `tools/server/tests/README.md` for the Python test workflow.
- For native Windows builds, use presets from `CMakePresets.json` such as `x64-windows-llvm-release`. Do not assume a plain `cmake -B build` flow is correct on Windows.
- If a task needs broader validation, use `ci/run.sh` and `.github/workflows/build-cpu.yml` as the authoritative references.

## Docker And Environment

- This fork is normally built and run on Linux, often through Docker, even when the current editing host is Windows.
- The branch-specific container reference is `.devops/intel.Dockerfile`, typically with the `server` target.
- When a task is about deployment or runtime behavior, prefer the Linux and Docker workflow described in `README.md` and `docs/docker.md` over Windows-native assumptions.
- On this machine, `git` and `docker` are available on Windows. Use native Windows commands only when the task is explicitly Windows-specific.

## Conventions

- Keep changes cross-platform unless the task is explicitly Linux-only or Windows-only.
- Follow `.clang-format`; run `git clang-format` after C++ edits when practical.
- Use the repository `.venv` for Python tooling when available.
- Treat changes to `include/llama.h`, backend code, and low-level CPU optimizations as high impact.
- Fork-only history makes these files especially sensitive: `.devops/intel.Dockerfile`, `ggml/src/ggml-cpu/CMakeLists.txt`, `ggml/src/ggml-cpu/arch/x86/quants.c`, and the centralized threadpool lifecycle in `common/common.cpp` and `common/common.h`.
- Never commit build artifacts, caches, or model files.
- Do not use destructive git commands such as hard reset or clean unless the user explicitly asks for them.

## References

- Build and backend guidance: `docs/build.md`
- Docker workflow: `docs/docker.md`
- Contribution and validation expectations: `CONTRIBUTING.md`
- Server test setup: `tools/server/tests/README.md`
- Build presets and toolchains: `CMakePresets.json`
