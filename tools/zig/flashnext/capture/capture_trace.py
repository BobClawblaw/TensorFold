"""Full forward trace: Triton launches, extension calls, CUDA ATen ops and comm calls, with named storages."""
import dataclasses, json, struct
from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tdesc(t):
    return {"t": 1, "ptr": t.data_ptr(), "base": t.untyped_storage().data_ptr(), "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape), "stride": list(t.stride()), "dev": t.device.type}


def desc(v):
    if isinstance(v, torch.Tensor):
        return tdesc(v)
    if isinstance(v, (list, tuple)):
        return [desc(x) for x in v]
    if isinstance(v, float):
        return {"f": v, "bits": struct.unpack("<I", struct.pack("<f", v))[0]}
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    return {"repr": repr(v)[:120]}


class Tracer:
    def __init__(self):
        self.events, self.phase, self.on, self.kernels = [], None, False, {}

    def add(self, kind, name, **kw):
        if self.on:
            self.events.append({"i": len(self.events), "phase": self.phase, "kind": kind, "name": name, **kw})

    @contextmanager
    def scope(self, phase):
        self.phase, self.on = phase, True
        mode = Dispatch(self)
        with mode:
            yield
        torch.cuda.synchronize()
        self.on = False


class Dispatch(TorchDispatchMode):
    def __init__(self, tr):
        super().__init__()
        self.tr = tr

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        cuda = any(isinstance(a, torch.Tensor) and a.is_cuda for a in list(args) + list(kwargs.values())) or \
            (isinstance(out, torch.Tensor) and out.is_cuda)
        if cuda:
            self.tr.add("aten", str(func), args=desc(list(args)), kwargs={k: desc(v) for k, v in kwargs.items()},
                        out=desc(out) if isinstance(out, (torch.Tensor, list, tuple)) else None)
        return out


def install(tr, comm, ext_modules):
    from triton.runtime.jit import JITFunction
    original = JITFunction.run

    def run(fn, *args, grid, warmup, **kwargs):
        k = original(fn, *args, grid=grid, warmup=warmup, **kwargs)
        if tr.on and not warmup and k is not None:
            names = fn.arg_names
            bound = dict(zip(names, args))
            bound.update({n: v for n, v in kwargs.items() if n in names})
            meta = dict(bound)
            meta.update(kwargs)
            g = grid(meta) if callable(grid) else grid
            g = [int(x) for x in (tuple(g) + (1, 1, 1))[:3]]
            sig = dict(k.src.signature)
            if k.hash not in tr.kernels:
                attrs = {}
                for key, specs in k.src.attrs.items():
                    idx = key[0] if isinstance(key, tuple) else key
                    attrs[names[idx] if isinstance(idx, int) else idx] = [list(x) for x in specs]
                tr.kernels[k.hash] = {"name": k.name, "files": dict(k.metadata_group), "attrs": attrs,
                                      "function": f"{fn.fn.__module__}.{fn.fn.__qualname__}"}
            tr.add("triton", k.name, hash=k.hash, grid=g, num_warps=k.metadata.num_warps,
                   args=[{"name": n, "type": sig.get(n), "v": desc(bound.get(n))} for n in names
                         if sig.get(n) not in (None, "constexpr")],
                   consts={n: desc(bound.get(n)) for n in names if sig.get(n) == "constexpr"})
        return k

    JITFunction.run = run
    for label, mod in ext_modules:
        for n in dir(mod):
            f = getattr(mod, n)
            if n.startswith("_") or not callable(f):
                continue

            def wrap(*a, _f=f, _n=f"{label}.{n}", **kw):
                at = len(tr.events)
                tr.add("ext", _n, args=desc(list(a)), kwargs={k: desc(v) for k, v in kw.items()})
                res = _f(*a, **kw)
                if tr.on and at < len(tr.events) and isinstance(res, torch.Tensor) and res.is_cuda:
                    tr.events[at]["ret"] = desc(res)
                return res
            setattr(mod, n, wrap)
    if comm is not None:
        for n in ("all_gather", "all_reduce", "exchange", "broadcast"):
            f = getattr(comm, n, None)
            if f is None:
                continue

            def cwrap(*a, _f=f, _n=n, **kw):
                tr.add("comm", _n, args=desc(list(a)), kwargs={k: desc(v) for k, v in kw.items()})
                return _f(*a, **kw)
            setattr(comm, n, cwrap)


def named_map(roots):
    """Every CUDA tensor reachable from the roots: name -> (storage base, storage bytes, data ptr, shape, dtype)."""
    out, seen = [], set()

    def walk(name, v, depth=0):
        if depth > 14:
            return
        if isinstance(v, torch.Tensor):
            if v.is_cuda and id(v) not in seen:
                seen.add(id(v))
                st = v.untyped_storage()
                out.append({"name": name, "base": st.data_ptr(), "bytes": st.nbytes(), "ptr": v.data_ptr(),
                            "dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape), "stride": list(v.stride())})
            return
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(f"{name}.{i}", x, depth + 1)
            return
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{name}.{k}", x, depth + 1)
            return
        if isinstance(v, (int, float, str, bool, bytes)) or v is None or id(v) in seen:
            return
        if type(v).__module__.startswith(("torch", "numpy", "threading", "concurrent", "ctypes")):
            return
        seen.add(id(v))
        fields = ([f.name for f in dataclasses.fields(v)] if dataclasses.is_dataclass(v)
                  else list(vars(v)) if hasattr(v, "__dict__") else [])
        for f in fields:
            if f in ("comm",):
                continue
            walk(f"{name}.{f}" if name else f, getattr(v, f, None), depth + 1)

    for name, v in roots:
        walk(name, v)
    return out


def dump_storages(path, roots):
    """Raw bytes of every CUDA storage reachable from the roots (state after reset): index.json + one file."""
    import os
    entries, seen = [], set()
    with open(os.path.join(path, "storages.bin"), "wb") as f:
        off = 0
        for n in named_map(roots):
            if n["base"] in seen:
                continue
            seen.add(n["base"])
            v = torch.empty(0, dtype=torch.uint8, device="cuda")
            # rebuild a byte view of the whole storage from a tensor of it
            entries.append({"name": n["name"], "base": n["base"], "bytes": n["bytes"], "offset": off})
        torch.cuda.synchronize()
    return entries
