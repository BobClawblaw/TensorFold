"""Dev-only: TF_FIXTURES=<dir> saves one launch per Triton specialization as a fixture once <dir>/GO exists."""
import ctypes, json, os, shutil, struct
from pathlib import Path

LIMIT = int(os.environ.get("TF_FIXTURE_LIMIT", str(1 << 30)))   # skip launches reaching more


def install(out_dir: str) -> None:
    import torch
    from triton.runtime.jit import JITFunction

    out = Path(out_dir)
    go = out / "GO"
    done: set[str] = set()
    skipped: dict[str, int] = {}
    original = JITFunction.run

    def extent(t):
        if t.numel() == 0:
            return 0
        return (sum((s - 1) * st for s, st in zip(t.shape, t.stride())) + 1) * t.element_size()

    def regions(tensors):
        by_storage = {}
        for name, t in tensors:
            st = t.untyped_storage()
            base = st.data_ptr()
            lo = t.data_ptr() - base
            hi = lo + extent(t)
            g = by_storage.setdefault(base, {"storage": st, "lo": lo, "hi": hi, "members": []})
            g["lo"], g["hi"] = min(g["lo"], lo), max(g["hi"], hi)
            g["members"].append((name, t))
        return list(by_storage.values())

    def view(g):
        v = torch.empty(0, dtype=torch.uint8, device="cuda")
        v.set_(g["storage"], g["lo"], (g["hi"] - g["lo"],))
        return v

    def run(fn, *args, grid, warmup, **kwargs):
        if warmup or not go.exists() or torch.cuda.is_current_stream_capturing():
            return original(fn, *args, grid=grid, warmup=warmup, **kwargs)
        names = fn.arg_names
        bound = dict(zip(names, args))
        bound.update({k: v for k, v in kwargs.items() if k in names})
        tensors = [(n, v) for n, v in bound.items() if isinstance(v, torch.Tensor)]
        groups = regions(tensors)
        size = sum(g["hi"] - g["lo"] for g in groups)
        if size > LIMIT:
            k = original(fn, *args, grid=grid, warmup=warmup, **kwargs)
            skipped[k.hash] = size
            return k
        torch.cuda.synchronize()
        before = [view(g).clone() for g in groups]
        kernel = original(fn, *args, grid=grid, warmup=warmup, **kwargs)
        if kernel is None or kernel.hash in done:
            return kernel
        torch.cuda.synchronize()
        after = [view(g).clone() for g in groups]
        done.add(kernel.hash)
        try:
            save(fn, kernel, bound, kwargs, grid, groups, before, after)
        except Exception as exc:                              # noqa: BLE001
            (out / "errors.txt").open("a").write(f"{kernel.name} {kernel.hash}: {exc!r}\n")
        return kernel

    def save(fn, kernel, bound, kwargs, grid, groups, before, after):
        d = out / f"{kernel.name}-{kernel.hash[:10]}"
        d.mkdir(parents=True, exist_ok=True)
        for leaf, where in kernel.metadata_group.items():
            ext = leaf.rsplit(".", 1)[-1]
            if ext in ("cubin", "json", "ptx"):
                shutil.copy(where, d / f"kernel.{ext}")
        arrays, where = {}, {}
        for i, (g, b, a) in enumerate(zip(groups, before, after)):
            name = f"r{i}"
            b.cpu().numpy().tofile(d / f"{name}.bin")
            a.cpu().numpy().tofile(d / f"expected_{name}.bin")
            start = g["storage"].data_ptr() + g["lo"]
            arrays[name] = {"bytes": g["hi"] - g["lo"], "align256": start % 256, "changed": not torch.equal(a, b)}
            for n, t in g["members"]:
                where[n] = (name, t.data_ptr() - start)
        sig = dict(kernel.src.signature)
        args = []
        for n in fn.arg_names:
            ty = sig.get(n)
            if ty is None or ty == "constexpr":
                continue
            v = bound.get(n)
            if ty.startswith("*"):
                if v is None:
                    raise ValueError(f"pointer {n} is None")
                r, off = where[n]
                args.append({"name": n, "kind": "ptr", "array": r, "offset": off})
            elif ty in ("i32", "i1", "i8", "i16", "u32"):
                args.append({"name": n, "kind": "i32", "value": int(v)})
            elif ty in ("i64", "u64"):
                args.append({"name": n, "kind": "i64", "value": int(v)})
            elif ty in ("fp32", "f32"):
                args.append({"name": n, "kind": "f32", "bits": struct.unpack("<I", struct.pack("<f", float(v)))[0]})
            else:
                raise ValueError(f"argument {n} of type {ty}")
        meta = dict(bound)
        meta.update(kwargs)
        g = grid(meta) if callable(grid) else grid
        g = [int(x) for x in (tuple(g) + (1, 1, 1))[:3]]
        runtime = [n for n in fn.arg_names if sig.get(n) not in (None, "constexpr")]
        attrs = {}
        for k, specs in kernel.src.attrs.items():
            idx = k[0] if isinstance(k, tuple) else k
            pname = fn.arg_names[idx] if isinstance(idx, int) else idx
            attrs[pname] = [list(x) for x in specs]
        div = sorted(runtime.index(p) for p, specs in attrs.items() if p in runtime and ["tt.divisibility", 16] in specs)
        (d / "manifest.json").write_text(json.dumps({
            "kernel": kernel.name, "hash": kernel.hash, "function": f"{fn.fn.__module__}.{fn.fn.__qualname__}",
            "arrays": arrays, "args": args, "grid": g, "divisible16": div,
            "num_warps": kernel.metadata.num_warps, "shared": kernel.metadata.shared,
            "global_scratch_size": getattr(kernel.metadata, "global_scratch_size", 0),
            "profile_scratch_size": getattr(kernel.metadata, "profile_scratch_size", 0)}, indent=1))

    JITFunction.run = run
    out.mkdir(parents=True, exist_ok=True)

    # extension calls (our .cu launchers): up to 3 distinct tensor-shape signatures per function
    from tensorfold.cuda import build
    import glob as _glob
    _lib = sorted(_glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")) +
                  _glob.glob("/usr/local/cuda/lib64/libcudart.so*"))
    cudart = ctypes.CDLL(_lib[0])
    load_original = build.load
    seen_ext: dict[str, set] = {}

    def describe_ext(args):
        sig = []
        for a in args:
            if isinstance(a, torch.Tensor):
                sig.append(("T", str(a.dtype), tuple(a.shape)))
            elif isinstance(a, (bool, int, float)):
                sig.append((type(a).__name__, a if isinstance(a, bool) else None))
            else:
                sig.append((type(a).__name__,))
        return tuple(sig)

    def wrap_ext(label, name, fn):
        key = f"{label}.{name}"

        def wrapper(*args, **kwargs):
            if not go.exists() or kwargs or torch.cuda.is_current_stream_capturing():
                return fn(*args, **kwargs)
            sig = describe_ext(args)
            have = seen_ext.setdefault(key, set())
            if sig in have or len(have) >= 3:
                return fn(*args, **kwargs)
            tensors = [(f"a{i}", a) for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_cuda]
            groups = regions(tensors)
            if sum(g["hi"] - g["lo"] for g in groups) > LIMIT:
                return fn(*args, **kwargs)
            torch.cuda.synchronize()
            before = [view(g).clone() for g in groups]
            pointed = []
            if key.endswith("gdn_io.front"):
                P, conv_ptrs, sid, a_log = args[0], args[1], args[2], args[5]
                nv = a_log.numel(); nk = nv // 3; C = 2 * nk * 128 + nv * 128
                table = conv_ptrs.cpu().tolist()
                used = sorted(set(int(x) for x in sid.cpu().tolist()))
                for i in used:
                    n = 3 * C * 2
                    buf = torch.empty(n, dtype=torch.uint8, device="cuda")
                    cudart.cudaMemcpy(ctypes.c_void_p(buf.data_ptr()), ctypes.c_void_p(table[i]), ctypes.c_size_t(n), 3)
                    pointed.append((i, buf))
            res = fn(*args, **kwargs)
            torch.cuda.synchronize()
            after = [view(g).clone() for g in groups]
            have.add(sig)
            try:
                d = out / f"ext-{key}-{len(have)}"
                d.mkdir(parents=True, exist_ok=True)
                arrays, where = {}, {}
                for i, (g, b, a) in enumerate(zip(groups, before, after)):
                    rn = f"r{i}"
                    b.cpu().numpy().tofile(d / f"{rn}.bin")
                    a.cpu().numpy().tofile(d / f"expected_{rn}.bin")
                    start = g["storage"].data_ptr() + g["lo"]
                    arrays[rn] = {"bytes": g["hi"] - g["lo"], "align256": start % 256, "changed": not torch.equal(a, b)}
                    for n, t in g["members"]:
                        where[n] = (rn, t.data_ptr() - start)
                desc = []
                for i, a in enumerate(args):
                    if isinstance(a, torch.Tensor):
                        rn, off = where.get(f"a{i}", (None, 0))
                        desc.append({"kind": "tensor", "array": rn, "offset": off, "dtype": str(a.dtype).replace("torch.", ""),
                                     "shape": list(a.shape), "stride": list(a.stride())})
                    elif isinstance(a, bool):
                        desc.append({"kind": "bool", "value": a})
                    elif isinstance(a, int):
                        desc.append({"kind": "int", "value": a})
                    elif isinstance(a, float):
                        desc.append({"kind": "float", "value": a, "f32_bits": struct.unpack("<I", struct.pack("<f", a))[0]})
                    else:
                        desc.append({"kind": type(a).__name__, "repr": repr(a)[:200]})
                relocate = []
                for i, buf in pointed:
                    pn = f"p{i}"
                    buf.cpu().numpy().tofile(d / f"{pn}.bin")
                    buf.cpu().numpy().tofile(d / f"expected_{pn}.bin")
                    arrays[pn] = {"bytes": buf.numel(), "align256": 0, "changed": False}
                    rn, off = where["a1"]
                    relocate.append({"table": rn, "offset": off + 8 * i, "array": pn})
                returned = None
                if isinstance(res, torch.Tensor) and res.is_cuda and res.numel():
                    res.contiguous().view(torch.uint8).cpu().numpy().tofile(d / "returned.bin")
                    returned = {"dtype": str(res.dtype).replace("torch.", ""), "shape": list(res.shape),
                                "bytes": res.numel() * res.element_size()}
                (d / "manifest.json").write_text(json.dumps({"function": key, "arrays": arrays, "args": desc,
                                                              "returned": returned, "relocate": relocate}, indent=1))
            except Exception as exc:                          # noqa: BLE001
                (out / "errors.txt").open("a").write(f"{key}: {exc!r}\n")
            return res
        wrapper._fixture = True
        return wrapper

    def load(*a, **kw):
        mod = load_original(*a, **kw)
        label = kw.get("name") or (a[0] if a else "ext")
        for n in dir(mod):
            f = getattr(mod, n)
            if not n.startswith("_") and callable(f) and not getattr(f, "_fixture", False):
                setattr(mod, n, wrap_ext(label, n, f))
        return mod

    build.load = load

    def report():
        (out / "skipped.json").write_text(json.dumps(skipped, indent=1))
    import atexit
    atexit.register(report)
    import threading, time
    def loop():
        while True:
            time.sleep(20)
            report()
    threading.Thread(target=loop, daemon=True).start()
