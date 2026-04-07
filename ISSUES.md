# TurboMemory GitHub Issues

Copy/paste each issue into GitHub https://github.com/Kubenew/TurboMemory/issues/new

---

## 1) Add GitHub Actions CI (pytest)

**Labels:** good first issue, help wanted

**Description:**
Add a GitHub Actions workflow that runs unit tests on every push/PR.

**Acceptance criteria:**
- workflow file exists under `.github/workflows/tests.yml`
- runs pytest
- runs on Python 3.10+ (or matrix 3.10/3.11/3.12)
- PR shows pass/fail checks

---

## 2) Add Ruff linting

**Labels:** good first issue

**Description:**
Add Ruff config and ensure `ruff check .` passes.

**Acceptance criteria:**
- ruff included in requirements/dev requirements
- `pyproject.toml` contains ruff config
- CI runs ruff

---

## 3) Add Black formatting

**Labels:** good first issue

**Description:**
Add Black config and format the project.

**Acceptance criteria:**
- black config exists in pyproject.toml
- `black .` runs clean
- CI includes formatting check

---

## 4) Add pre-commit hooks

**Labels:** good first issue

**Description:**
Add `.pre-commit-config.yaml` for black + ruff.

**Acceptance criteria:**
- running `pre-commit install` works
- README mentions pre-commit setup

---

## 5) Add examples/basic_demo.py

**Labels:** good first issue

**Description:**
Create a minimal runnable example script.

**Acceptance criteria:**
- demonstrates init/add/query
- prints results
- documented in README

---

## 6) Add examples/benchmark_small.py

**Labels:** good first issue

**Description:**
Small benchmark script to insert/query 1k items.

**Acceptance criteria:**
- prints insert time and query time
- easy to run

---

## 7) Add benchmarks/README.md

**Labels:** good first issue

**Description:**
Explain how to run benchmark scripts and interpret results.

**Acceptance criteria:**
- clear command examples
- describes dataset generation

---

## 8) Add benchmark: compression ratio table

**Labels:** help wanted

**Description:**
Measure float32 vs 8-bit vs 6-bit vs 4-bit disk usage.

**Acceptance criteria:**
- script outputs markdown table
- README includes summary

---

## 9) Add benchmark: query latency scaling

**Labels:** help wanted

**Description:**
Measure query latency at 10k / 100 k / 1M chunks.

**Acceptance criteria:**
- results stored in benchmarks/results.md
- includes CPU/RAM info in output

---

## 10) Add unit tests for packed vector encode/decode

**Labels:** good first issue, help wanted

**Description:**
Test that packing/unpacking works correctly and is deterministic.

**Acceptance criteria:**
- tests cover 4-bit, 6-bit, 8-bit modes
- checks max error bounds
- edge cases (min/max values)

---

## 11) Add unit test for cosine similarity correctness

**Labels:** good first issue

**Description:**
Ensure similarity function returns correct values.

**Acceptance criteria:**
- tests compare with known expected cosine values

---

## 12) Add unit tests for TTL expiration logic

**Labels:** good first issue

**Description:**
Test that expired memories are excluded or downweighted correctly.

**Acceptance criteria:**
- deterministic timestamps
- covers boundary conditions

---

## 13) Add unit tests for confidence decay

**Labels:** good first issue

**Description:**
Verify confidence decreases as expected over time.

**Acceptance criteria:**
- deterministic expected output
- covers 0%, mid, and full decay

---

## 14) Add tests for contradiction detection (basic)

**Labels:** help wanted

**Description:**
Add test cases for contradiction scoring logic.

**Acceptance criteria:**
- simple example contradictions detected reliably
- no false positives in trivial cases

---

## 15) Replace prints with structured logging

**Labels:** help wanted

**Description:**
Replace print statements with Python logging module.

**Acceptance criteria:**
- `logging.getLogger(__name__)` used
- CLI has `--verbose` flag

---

## 16) Add TurboMemory.stats() API

**Labels:** help wanted

**Description:**
Implement a method returning storage and performance stats.

**Acceptance criteria:**
- returns dict with chunk count, topics, disk usage
- CLI command prints it

---

## 17) Add TurboMemory.delete(memory_id) API

**Labels:** help wanted

**Description:**
Support deleting stored chunks by ID.

**Acceptance criteria:**
- deleted chunk no longer appears in query
- deletion tracked in log

---

## 18) Add metadata filtering to query

**Labels:** help wanted

**Description:**
Allow query filtering by topic, time range, tags.

**Acceptance criteria:**
- query supports `topic=`, `since=`, `until=`
- tests included

---

## 19) Add tag support to memory chunks

**Labels:** help wanted

**Description:**
Allow storing tags list and filtering by tag.

**Acceptance criteria:**
- schema updated
- CLI supports `--tags a,b,c`
- query can filter tags

---

## 20) Improve SQLite schema + indexing

**Labels:** help wanted

**Description:**
Optimize schema and add indexes for query speed.

**Acceptance criteria:**
- explain query plan documented
- query performance improves

---

## 21) Add schema versioning + migrations

**Labels:** help wanted

**Description:**
Add a schema version number and migration system.

**Acceptance criteria:**
- old index upgraded automatically
- migration tests exist

---

## 22) Add export / import CLI commands

**Labels:** help wanted

**Description:**
Export a TurboMemory dataset to a single archive file and restore it.

**Acceptance criteria:**
- export creates tar/zip
- import restores identical query behavior

---

## 23) Add integrity check command (doctor)

**Labels:** help wanted

**Description:**
CLI tool to scan index + vector files for corruption.

**Acceptance criteria:**
- reports missing records
- reports broken topic files
- can optionally repair index

---

## 24) Add tmmeta.json file (metadata spec)

**Labels:** help wanted

**Description:**
Create a stable metadata file describing model + quant format.

**Acceptance criteria:**
- includes embedding dim, quant bits, created time
- loaded on init

---

## 25) Create TMF v1 file format spec doc

**Labels:** help wanted

**Description:**
Write docs specifying TurboMemory storage format.

**Acceptance criteria:**
- stored in docs/TMF_SPEC.md
- includes layout of .tmvec, .tmlog, .tmindex

---

## 26) Add FastAPI server prototype

**Labels:** help wanted, advanced

**Description:**
Add server mode exposing REST endpoints.

**Acceptance criteria:**
- `/add`, `/query`, `/stats`
- request/response schema defined
- `docker run` works

---

## 27) Add Dockerfile improvements for server mode

**Labels:** help wanted

**Description:**
Ensure Dockerfile builds minimal image and runs server.

**Acceptance criteria:**
- `docker build` works
- `docker run` launches API

---

## 28) Add replication sync prototype (log-based)

**Labels:** help wanted, advanced

**Description:**
Implement node-to-node sync using append-only log offsets.

**Acceptance criteria:**
- node A pulls missing log entries from node B
- sync is idempotent
- basic tests or demo script

---

## 29) Add hybrid search (BM25 + vector fusion)

**Labels:** help wanted, advanced

**Description:**
Implement keyword scoring and combine with vector similarity.

**Acceptance criteria:**
- query returns fused ranking
- configurable weights
- benchmark shows improved recall on mixed queries

---

## 30) Publish PyPI package + CLI entry point

**Labels:** help wanted

**Description:**
Make TurboMemory pip-installable and expose CLI command turbomemory.

**Acceptance criteria:**
- `pip install turbomemory` works
- `turbomemory --help` works
- version tagging documented
