# Brain Decoding Cookbook

Codebase for brain decoding analysis.

## Environment Setup

You can setup environment with [uv](https://docs.astral.sh/uv/):

```shellsession
$ uv sync
```

To run visualization code, install the optional dependencies:
```shellsession
$ uv sync --extra viz
```

> **Note**
> If you use CUDA, specify the PyTorch index according to your CUDA version.
> ```toml
> # example
> torch = [
>     { index = "pytorch-cu124", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
> ]
> ...
> [[tool.uv.index]]
> name = "pytorch-cu124"
> url = "https://download.pytorch.org/whl/cu124"
> explicit = true
> ```
> The candidate custom index URLs are listed [here](https://pytorch.org/get-started/previous-versions/).