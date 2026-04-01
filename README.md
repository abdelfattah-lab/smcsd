# SGLang — SMC Speculative Decoding Branch (`smc_v0`)

This branch implements **Sequential Monte Carlo (SMC) speculative decoding** in SGLang. SMC runs N particles (parallel generation paths) per request: a lightweight draft model proposes tokens, the target model scores them, and particles are resampled by importance weight so compute focuses on the most promising paths.
<img width="772" height="424" alt="image" src="https://github.com/user-attachments/assets/3cda3320-e257-4079-99b3-93e3a7bec627" />

## Installation

```bash
uv venv --python 3.12
uv pip install -e "python"
uv pip install -e "ssd" --no-deps
```


