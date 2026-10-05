# ColiOllama

Ollama's developer experience (CLI, model registry, Hugging Face pulls, HTTP API)
on top of the [Colibrí](https://github.com/JustVugg/colibri) engine, a pure-C,
disk-streamed Mixture-of-Experts runtime.

## 1. Overview & Architecture

Colibrí binds one model directory per engine process and cannot hot-swap it.
ColiOllama owns that process and hides the constraint behind Ollama/OpenAI APIs:

```text
 clients (ollama CLI libs, curl, OpenAI SDK)          coliollama CLI
              │  HTTP :11434                                │
              ▼                                             ▼
   api/routes  (ollama_compat, openai_compat) ──── api/fastapi (app, dependencies)
              │  acquire lease for model X
              ▼
   core/scheduler  queue_manager + policy  ── FIFO queue, drain barrier, swap decision
              │  start(X) / stop()
              ▼
   core/engine     lifecycle (health, shutdown) + process (`coli serve`) + client
              │  OpenAI-style HTTP on a private localhost port
              ▼
   Colibrí C engine  ◄── model directory from registry/local_store (+ huggingface_resolver)
```

| Package | Role |
|---|---|
| `core/engine/process.py` | Launches `coli serve --model <dir> --model-id <name>` in its own process group |
| `core/engine/lifecycle.py` | Health-checks `/health`, graceful SIGTERM→SIGKILL shutdown, state file |
| `core/scheduler/` | `QueueManager` (queues, leases) and `SchedulerPolicy` (admit / wait / swap) |
| `registry/` | JSON registry (`registry.json`) and Hugging Face resolver |
| `api/` | FastAPI app, Ollama-native and OpenAI-compatible routes |
| `cli/` | Typer commands |

Every endpoint that generates tokens goes through the scheduler; the Ollama
routes translate to the engine's OpenAI gateway and back (NDJSON streaming).

## 2. Installation

**Fast path (new machine):** `./scripts/bootstrap.sh` creates `.venv`, installs CPU-only torch
and the package, creates `.env`, downloads the latest Colibrí (building CUDA engines when an
NVIDIA GPU and `nvcc` exist) and prints the hardware scan. Options: `--dev`, `--hf-token TOKEN`,
`--no-engine`, `--no-gpu-build`, `--skip-torch`, `--venv DIR`. It is safe to re-run. Copy the
project to the new machine without `.venv`, `dependencies/` or `.env`.

Prerequisites

- Python ≥ 3.10.
- A working Colibrí checkout/build (`coli` launcher; see the Colibrí README for
  its C toolchain, e.g. a C compiler and `make`, and check with `coli doctor`).
  ColiOllama looks for it, in order, at `COLIOLLAMA_COLI` (a full command, may
  include arguments), `$COLIBRI_HOME/coli`, the install managed by `coliollama update`
  (`~/.coliollama/engines`), `./dependencies/colibri/coli`, the
  project's own `dependencies/colibri/coli`, then `coli` on `PATH`. A found script is
  run with the current Python interpreter.
- Python packages: `pip install -e .` installs everything, including `numpy`, `safetensors`
  and `torch`, which `coli convert` needs. For a smaller CPU-only torch, install it first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Models must be in a format Colibrí can load. ColiOllama checks this and runs
  `coli convert` when needed (see section 3); converting needs Colibrí's Python
  conversion dependencies and plenty of disk and time.

```bash
git clone <this repo> && cd ColiOllama
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add '.[dev]' for the test suite
export COLIBRI_HOME=/path/to/colibri
coliollama serve
```

Environment

| Variable | Meaning | Default |
|---|---|---|
| `COLIOLLAMA_HOME` | Registry, models, logs, state | `~/.coliollama` |
| `COLIOLLAMA_COLI` | Command used to launch Colibrí | auto-detected |
| `COLIOLLAMA_ENGINE_ARGS` | Extra args appended to `coli serve` (e.g. `--ram 32 --cap 2`) | none |
| `COLIOLLAMA_CONVERT_ARGS` | Extra args appended to `coli convert` | none |
| `COLIOLLAMA_GPU` | `auto`, `none`, or a device list such as `0,1` | `auto` |
| `COLIOLLAMA_AUTO_UPDATE` | `0` disables the background Colibrí update check in `serve` | `1` |
| `COLIOLLAMA_UPDATE_INTERVAL_HOURS` | Hours between background checks | `24` |
| `GITHUB_TOKEN` | Optional, raises the GitHub API rate limit for update checks | none |
| `COLIOLLAMA_STARTUP_TIMEOUT` | Seconds to wait for engine health | `900` |
| `COLIOLLAMA_HOST` | Server URL used by `run`/`ps`/`stop` | `http://127.0.0.1:11434` |
| `HF_TOKEN` | Hugging Face token for gated/private repos | none |

Files: `~/.coliollama/models/<org>--<repo>` (downloads), `registry.json`,
`engine.json` (running engine PID), `logs/engine.log`, `logs/server.log`.

## 3. Hugging Face Model Management

```bash
coliollama pull organization/repository-name     # download + register
coliollama run  organization/repository-name     # pulls first if missing
coliollama pull /nvme/glm52_i4                   # register an existing local directory
```

Resolution order for `<model>`: registry name → existing local directory →
Hugging Face repo ID (`org/repo`). Only the last triggers a download, using
`huggingface_hub.snapshot_download` into `~/.coliollama/models/org--repo`, with
its progress bars. Names are matched with an optional `:latest` suffix. The API
never downloads implicitly: an unknown model returns 404 and asks you to `pull`.

### Readiness check and automatic conversion

Unless `--no-convert` is given, `pull` and `run` verify that Colibrí can load the
model before registering it:

1. `coli doctor --model <dir> --deep --json` is run. Failing model checks
   (config, shards, tokenizer, required tensors, index) mean the weights are not
   directly usable; failures in `engine.*`, `accelerator.*`, `memory.*`, `storage.*`
   and `placement.*` (engine not built, GPU, RAM, disk) are about your machine,
   are shown as warnings, and do not trigger a conversion.
   `doctor` only validates tensor layout, and the engine reports healthy before it
   reads any weights, so a raw Hugging Face checkpoint passes both. Therefore, if
   `doctor` passes, a **smoke test** follows: a throwaway engine is started on the
   directory and asked for one token (this loads the model, so it can take a while
   for big models). A crash or error means "not usable". Because the smoke test
   cannot tell a bad model from, say, too little RAM, a smoke-test failure on a
   machine that is genuinely short of memory can trigger an unnecessary conversion.
2. If the model is not usable and came from a Hugging Face repo ID, it is converted
   with `coli convert --repo <org/repo> --model ~/.coliollama/models/org--repo-coli`
   (Colibrí's own download-and-convert, with its progress output; extra flags such
   as `--ebits 4` can be given through `COLIOLLAMA_CONVERT_ARGS`). The result is
   re-checked, registered under the original name, and the raw download is deleted
   to save disk unless `--keep-source` is given.
3. Models that pass both checks are marked verified in the registry and are not re-checked.
   Registry entries from before this feature are checked once.

Limits: `coli convert` always fetches the repo's default branch (`--revision` only
affects the raw download), and a failed conversion is restarted from scratch on the
next attempt. Local directories that fail the check cannot be converted
automatically; convert them with `coli convert`/Colibrí's tools, or register them
as is with `--no-convert`.

### GPU

Release binaries are CPU-only. With `--gpu auto` (default) ColiOllama asks `coli doctor`
whether the engine build and the machine can use a GPU for the model, and if so starts the
engine with `--gpu auto --auto-tier`; otherwise it runs on CPU. `ps` shows the processor.
When an NVIDIA GPU, `nvidia-smi` and the CUDA toolkit (`nvcc`, found on `PATH`, `$CUDA_HOME` or
`/usr/local/cuda`) are present, `coliollama update` (and the background check) also builds the
CUDA engines from the release source (`make CUDA=1`) and installs them next to the release
binaries. Families without a CUDA code path (e.g. OLMoE) stay on CPU. Only CUDA on Linux is
automated; AMD/HIP and DeepSeek-V4 need a manual build. `--gpu none` forces CPU.

## 4. Concurrency Behavior

Colibrí serves one model at a time, so the scheduler is a strict FIFO with a
drain barrier:

1. **Same model.** Requests for the running model are queued and admitted in
   order, at most `--max-concurrency` at a time (default `1`, i.e. sequential).
   The engine is not restarted.
2. **Different model.** A request for model B while A runs joins the queue and
   blocks everything behind it. The scheduler waits until all admitted
   requests for A have finished, then shuts down A's engine, boots B and waits
   for its health check, then admits B's queue.
3. **Fairness.** Requests that arrive *after* the first B request queue behind
   it even if they target A (they run after the swap), so neither model starves.
   The cost is that alternating A/B traffic causes a restart per switch, which
   is expensive for large models: batch work per model where you can.
4. **Failures.** If an engine fails to start, only requests for that model get
   `503`; others proceed. Clients that disconnect while queued are dropped from the queue.
   If the engine crashes, the next request restarts it.

`coliollama ps` shows queue depth (in flight + waiting).

## 5. CLI Reference

```text
coliollama serve [--host 127.0.0.1] [--port 11434] [--max-concurrency 1]
                 [--log-level info] [--gpu auto|none|0,1]
                 [--auto-update/--no-auto-update] [--detach/-d]
coliollama scan   [--online] [--all] [--json]
coliollama update [--check] [--force] [--gpu-build/--no-gpu-build]
coliollama run   <model> [PROMPT] [--host URL] [--revision REV]
coliollama pull  <model> [--revision REV]
coliollama list
coliollama ps    [--host URL]
coliollama stop  [--host URL]
```

- `serve` runs the API server (`--detach` backgrounds it, logs in `logs/server.log`).
- `run` resolves/downloads the model, starts a detached server on port 11434 if
  none is reachable, and opens an interactive chat (`/clear`, `/bye`, Ctrl-D).
  With `PROMPT` it answers once and exits. With `--host` or `COLIOLLAMA_HOST`
  set it never starts a server itself.
- `scan` reports CPU threads, RAM, free disk (where models are stored), GPUs/VRAM and the engine
  build, then rates each known downloadable model: download size, converted size, peak disk
  (raw download plus converted copy), whether it runs on GPU or CPU, and `fits` (converted model
  fits in RAM) or `streams` (experts are read from disk, slower). Models that do not fit the disk
  are hidden unless `--all`. `--online` adds popular Hugging Face repos with a model type Colibrí
  supports (marked unverified), `--json` is machine-readable. Sizes are estimates.
- `update` checks the [Colibrí releases](https://github.com/JustVugg/colibri/releases) and, if a
  newer one exists, downloads it, verifies its SHA256 against `SHA256SUMS.txt`, unpacks it to
  `~/.coliollama/engines/versions/<ver>` and makes it the main engine (the previous version is
  kept for rollback, older ones are pruned). `--check` only reports. The running engine keeps
  its version until it next starts. `serve` does the same check on startup and every
  `COLIOLLAMA_UPDATE_INTERVAL_HOURS`; it is skipped when `COLIOLLAMA_COLI` or `COLIBRI_HOME`
  pins the engine.
- `--no-convert` skips the readiness check/conversion; `--keep-source` keeps the raw download after a conversion.
- `list` prints name, size and path. `ps` prints active model, engine PID and queue depth.
- `stop` force-terminates the engine (via the server, or by the recorded PID if
  the server is gone). In-flight requests fail.

## 6. API Integration Guide

Default port is `11434` (Ollama's); use `--port 8000` if you prefer.

Ollama native: `GET /api/tags`, `GET /api/ps`, `POST /api/chat`, `POST /api/generate`
(streaming NDJSON by default; `"stream": false` for one JSON object; `options`
`temperature`, `top_p`, `num_predict`, `stop`, `seed`, penalties are mapped).
An empty `messages`/`prompt` just loads the model. Extra: `POST /api/stop`.

Model management (full Ollama API coverage): `GET /api/version`, `POST /api/show`,
`POST /api/copy` (an alias, no data duplicated), `DELETE /api/delete` (409 while the model is
loaded; removes the files only if ColiOllama downloaded them and no alias uses them),
`POST /api/pull` (streams `{"status": ...}` lines, runs the readiness check/conversion).
Answered with HTTP 501 because Colibrí cannot back them: `POST /api/create`, `/api/push`,
`/api/embed`, `/api/embeddings`, `POST /api/blobs/:digest` (and `HEAD` always 404), `POST /v1/embeddings`.

OpenAI compatible: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /v1/models/{id}`.

```bash
curl http://localhost:11434/api/tags

curl http://localhost:11434/api/chat -d '{
  "model": "organization/repository-name",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'

curl http://localhost:11434/api/generate -d '{
  "model": "organization/repository-name", "prompt": "Hello", "stream": false
}'

curl http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "organization/repository-name",
  "messages": [{"role": "user", "content": "Hello"}]
}'

curl http://localhost:11434/api/ps
```

The request body is parsed as JSON whatever the `Content-Type`, so plain `curl -d` works.

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

resp = client.chat.completions.create(
    model="organization/repository-name",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)

for chunk in client.chat.completions.create(
    model="organization/repository-name",
    messages=[{"role": "user", "content": "Count to five"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

Use `http://localhost:8000/v1` instead if you started the server with `--port 8000`.
There is no authentication; keep the default `127.0.0.1` bind unless the network is trusted.

## Development

```bash
pip install -e '.[dev]'
pytest        # uses tests/fake_coli.py as a stand-in engine; no Colibrí build needed
```
