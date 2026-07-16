---
title: LLM Load
---

A launcher and load-testing utility for current llama.cpp builds.

## Build and inspect devices

```bash
chmod +x ./runllama.sh ./llmload.py

./runllama.sh UPDATE
./runllama.sh BUILD VULKAN
./runllama.sh LS VULKAN

LLAMA_API_KEY='your_api_key' ./runllama.sh 9000 SYCL 1 35b 512 4 auto
LLAMA_API_KEY='your_api_key' ./llmload.py 192.168.0.123:9000 --concurrency 4 --requests 4 --warmup 0 --max-tokens 1024
```

`UPDATE` uses a fast-forward-only pull, so it will not overwrite divergent or
locally modified llama.cpp work. Build and benchmark with the revision printed
by the launcher; upstream master changes frequently and can regress.

For a build that can expose the Arc cards through Vulkan and the NVIDIA card
through CUDA:

```bash
./runllama.sh BUILD HYBRID
./runllama.sh LS HYBRID
```

Device numbers come from `LS`; do not assume PCI ordering.

## Launch

One Arc GPU, one slot, 262,144-token context:

```bash
./runllama.sh 9000 VULKAN 0 35b 512 1 262144
```

Both Arc GPUs using the default layer split:

```bash
./runllama.sh 9000 VULKAN 0,1 35b 512 1 262144 8 layer 1,1
```

MTP variant with automatic context fitting and a 1 GiB margin per GPU:

```bash
LLAMA_FIT_TARGET=1024 ./runllama.sh 9000 VULKAN 0,1 35bmtp 512 1 auto 8 layer 1,1
```

Mixed Vulkan and CUDA devices require the hybrid build and full device names:

```bash
./runllama.sh 9000 HYBRID Vulkan0,Vulkan1,CUDA0 35b 512 1 262144 8 layer 2,2,1
```

Mixed-vendor splitting can be slower because transfers may cross host memory.
Establish one-card and two-identical-card baselines before adding the RTX card.
`tensor` split mode is experimental; compare `none`, `layer`, `row`, and
`tensor` rather than assuming more GPUs will be faster.

The server binds to localhost by default. For LAN access, use an API key:

```bash
LLAMA_HOST=0.0.0.0 LLAMA_API_KEY='replace-me' \
  ./runllama.sh 9000 VULKAN 0,1 35b 512 1 262144 8 layer 1,1
```

`LLAMA_API_KEY` is consumed by llama-server through its supported environment
variable, so the secret is not copied onto the process command line.

## Diagnose and stop

```bash
./runllama.sh INSPECT 9000
./runllama.sh KILL 9000
```

`KILL ALL` is available but must be requested explicitly.

## Fair performance comparison

Start with one device, one slot, the same GGUF, context size, KV-cache types,
and MTP setting in both applications. The launcher disables DRY and presence
penalties by default so sampling does not dominate generation at long context.

Use llama.cpp's `llama-bench` for backend throughput measurements; server-side
tokens/second also includes scheduling and sampling:

```bash
~/llama.cpp/build-vulkan/bin/llama-bench \
  -m /mnt/llm/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -dev Vulkan0 -sm none -ngl all -fa 1
```

Load test a running server:

```bash
sudo apt install python3-aiohttp
./llmload.py 127.0.0.1:9000 --concurrency 2 --requests 10 --max-tokens 256
```

Alternatively, install `requirements.txt` in a virtual environment rather than
using Ubuntu's `python3-aiohttp` package.

The historical separate host/port form is also accepted:

```bash
./llmload.py 127.0.0.1 9000 2 256
```

For a sustained 60-second run with authentication and JSON output:

```bash
LLAMA_API_KEY='replace-me' ./llmload.py 192.168.0.24:9000 \
  --concurrency 8 --duration 60 --max-tokens 512 --json
```

The benchmark performs an unmeasured warm-up, discovers the loaded model from
`/v1/models`, streams by default to report time to first token, and exits with a
nonzero status if any measured request fails. Use `--help` for prompt, model,
timeout, EOS, streaming, and other controls.

## License

Copyright [AStarship™](https://astarship.net).

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
