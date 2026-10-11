"""Dev-only: TF_RECORD=<dir> records every Triton and extension launch to <dir>/launches-<host>-<pid>.json."""
import atexit, os, socket, threading, time

def install(out_dir: str) -> None:
    from ._launch_recorder import Recorder
    rec = Recorder().install()
    rec.detail = os.environ.get("TF_RECORD_DETAIL", "1") == "1"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"launches-{socket.gethostname()}-{os.getpid()}.json")
    from tensorfold.cuda import build
    original = build.load

    def load(*a, **kw):
        mod = original(*a, **kw)
        label = kw.get("name") or (a[0] if a else "ext")
        names = tuple(n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n)))
        rec.wrap(mod, names, label)
        return mod

    build.load = load
    state = {"n": -1}

    def flush() -> None:
        if len(rec.log) != state["n"] or not rec.log:
            state["n"] = len(rec.log)
            tmp = path + ".tmp"
            rec.dump(tmp)
            os.replace(tmp, path)

    def loop() -> None:
        while True:
            time.sleep(20)
            try:
                flush()
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()
    atexit.register(flush)
    import tensorfold
    tensorfold._recorder = rec
