"""Trace helpers: resolve addresses to named tensors or step temporaries, keep the events that do GPU work."""
import bisect, json

META = {"aten.slice.Tensor", "aten.select.int", "aten.detach.default", "aten.view.default", "aten.view.dtype",
        "aten.unsqueeze.default", "aten.alias.default", "aten.as_strided.default", "aten.expand.default",
        "aten.t.default", "aten.reshape.default", "aten._unsafe_view.default", "aten.permute.default",
        "aten.squeeze.dim", "aten.transpose.int", "aten.split.Tensor", "aten.unbind.int", "aten.chunk.default",
        "aten.flatten.using_ints", "aten.narrow.default", "aten.split_with_sizes.default", "aten.lift_fresh.default",
        "aten.empty.memory_format", "aten.empty_like.default", "aten.empty_strided.default"}


class Resolver:
    def __init__(self, named):
        self.r = sorted((n["base"], n["base"] + max(n["bytes"], 1), n["name"]) for n in named)
        self.s = [x[0] for x in self.r]
        self.temps = {}          # storage base -> temp id (in order of first allocation in the step)

    def name(self, t):
        i = bisect.bisect_right(self.s, t["base"]) - 1
        if i >= 0 and self.r[i][0] <= t["base"] < self.r[i][1]:
            return self.r[i][2], t["ptr"] - self.r[i][0]
        k = self.temps.setdefault(t["base"], f"tmp{len(self.temps)}")
        return k, t["ptr"] - t["base"]


def load(d, phase):
    ev = json.load(open(d + "/trace.json"))["events"]
    return [e for e in ev if e["phase"] == phase]


def work(events, dumps=True):
    out = []
    for e in events:
        if e["kind"] == "aten" and e["name"] in META:
            continue
        if dumps and e["kind"] == "aten" and e["name"] == "aten._to_copy.default" and \
                e.get("kwargs", {}).get("device") == "cpu":
            continue                     # the capture's own dumps
        out.append(e)
    return out


def symbolic(v, res):
    if isinstance(v, dict) and v.get("t") == 1:
        if v["dev"] != "cuda":
            return {"host": v["dtype"], "shape": v["shape"]}
        n, off = res.name(v)
        return {"ref": n, "off": off, "dtype": v["dtype"], "shape": v["shape"], "stride": v["stride"]}
    if isinstance(v, list):
        return [symbolic(x, res) for x in v]
    if isinstance(v, dict):
        return {k: symbolic(x, res) for k, x in v.items()}
    return v
