"""The KL study's corpus: a three-turn chat (thinking off) mixing prose, code, math, German, Chinese and JSON."""

import json
import sys

from transformers import AutoTokenizer

U1 = """I'm writing a short essay about why lighthouses still matter in the age of satellite navigation. Could you draft a few paragraphs for me? Keep it thoughtful, not sentimental."""

A1 = """Lighthouses are easy to dismiss as relics. A modern cargo ship carries two independent satellite receivers, an electronic chart display, radar with automatic target tracking, and an identification system that broadcasts its position to every vessel within range. On paper, a tower with a lamp on top has nothing left to contribute. Yet coast guards on every continent continue to maintain thousands of them, and the reasons say something interesting about how we build systems we can trust.

The first reason is redundancy of a particular kind. Satellite signals are weak by the time they reach the surface of the Earth, roughly comparable to a car headlight seen from several thousand kilometres away. They can be jammed by cheap transmitters, spoofed by more sophisticated ones, and degraded by solar storms. When that happens, the failure is often silent: the receiver keeps showing a confident position that is simply wrong. A lighthouse fails differently. If the light is out, the navigator notices its absence, and the absence itself is information. A system whose failures are visible is, in an important sense, safer than one whose failures are hidden, even if the hidden one is more accurate most of the time.

The second reason is that a lighthouse requires almost nothing from the person using it. There is no subscription, no software update, no battery to charge, and no menu to navigate. A fisherman in a small open boat with a hand compass and a printed chart can identify a light by its characteristic, for example two white flashes every ten seconds, and know exactly which headland he is looking at. That accessibility matters because the sea is shared by vessels of every size and budget, and the cheapest boats are often the ones most exposed to bad weather.

Finally, lighthouses carry a kind of institutional memory. Each one marks a place where, at some point, enough ships were lost that a community decided to spend a great deal of money and effort on a warning. The towers stand on reefs, sandbars and cliffs that have not moved. Electronic charts are revised and corrected, but the light remains where it was placed, a fixed reference that predates every technology we now layer on top of it. To keep it burning is not nostalgia; it is a modest, inexpensive bet that the newest tools will occasionally fail, and that when they do, something simple should still be there."""

U2 = """Thanks, that's great. Now something completely different. I need a Python function that merges overlapping time intervals, plus a quick explanation. Then please solve this word problem step by step: A tank is filled by two pipes. Pipe A alone fills it in 6 hours and pipe B alone in 9 hours. A drain empties the full tank in 12 hours. If all three are open, how long does it take to fill the empty tank? Finally, give me the result of the interval example as JSON."""

A2 = """Here is a function that merges overlapping or touching intervals. It sorts the intervals by their start time and then walks through them once, extending the current interval whenever the next one begins before the current one ends.

```python
from typing import Iterable


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    \"\"\"Merge overlapping or touching closed intervals.

    >>> merge_intervals([(1, 3), (2, 6), (8, 10), (10, 12), (15, 18)])
    [(1, 6), (8, 12), (15, 18)]
    \"\"\"
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if start > end:
            raise ValueError(f"interval ({start}, {end}) ends before it starts")
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


if __name__ == "__main__":
    meetings = [(9, 10), (13, 15), (9, 11), (14, 16), (17, 18)]
    for start, end in merge_intervals(meetings):
        print(f"busy from {start}:00 to {end}:00")
```

Sorting costs O(n log n), and the single pass afterwards is linear, so the whole function runs in O(n log n) time and uses O(n) extra space for the result. The comparison `start <= merged[-1][1]` treats intervals that merely touch, such as (8, 10) and (10, 12), as overlapping; if you want them kept apart, change it to a strict `<`. Note that `max(last_end, end)` is necessary because an interval can be completely contained in the previous one, for example (1, 10) followed by (2, 5).

**The tank problem, step by step.**

1. Express each pipe as a rate in tanks per hour. Pipe A fills 1/6 of the tank per hour, pipe B fills 1/9, and the drain removes 1/12.
2. With all three open, the net rate is 1/6 + 1/9 - 1/12.
3. Use the least common denominator, 36: 1/6 = 6/36, 1/9 = 4/36, and 1/12 = 3/36.
4. The net rate is (6 + 4 - 3)/36 = 7/36 of the tank per hour.
5. The time to fill one tank is the reciprocal: 36/7 hours, which is about 5.142857 hours.
6. Convert the fraction of an hour: 1/7 of an hour is 60/7 ≈ 8.57 minutes, so the answer is 5 hours and roughly 8.6 minutes, or about 5 hours 9 minutes.

A quick sanity check: pipe A by itself needs 6 hours, and adding pipe B should make things faster while the drain slows them down. Since B's contribution (4/36) is larger than the drain's loss (3/36), the combined time must be a little under 6 hours, and 36/7 ≈ 5.14 fits that expectation.

**The interval example as JSON.** Running the script on the meeting list produces three busy blocks:

```json
{
  "input": [[9, 10], [13, 15], [9, 11], [14, 16], [17, 18]],
  "merged": [
    {"start": 9, "end": 11, "duration_hours": 2},
    {"start": 13, "end": 16, "duration_hours": 3},
    {"start": 17, "end": 18, "duration_hours": 1}
  ],
  "total_busy_hours": 6,
  "free_gaps": [{"start": 11, "end": 13}, {"start": 16, "end": 17}],
  "touching_counts_as_overlap": true
}
```"""

U3 = """Perfect. Last request: could you write a short paragraph in German about autumn in a small town, then a short paragraph in Chinese about the same thing, and end with one English sentence comparing the two?"""

A3 = """Im Herbst verändert sich die kleine Stadt fast unmerklich. Morgens liegt Nebel über dem Fluss, und die Bäcker öffnen ihre Läden, während die Straßenlaternen noch brennen. Auf dem Marktplatz verkaufen die Bauern Kürbisse, Äpfel und die letzten Pflaumen des Jahres, und die Kinder sammeln Kastanien auf dem Weg zur Schule. Am Nachmittag riecht es nach Holzfeuer und frisch gebackenem Zwiebelkuchen. Die Touristen sind längst abgereist, und die Bewohner haben ihre Stadt wieder für sich. Wenn der Wind die bunten Blätter durch die engen Gassen treibt, sitzen die älteren Leute in der Weinstube, trinken neuen Wein und erzählen dieselben Geschichten wie jedes Jahr, als wäre der Herbst nur ein Vorwand, um sich wieder zu treffen.

秋天来到小镇的时候，一切都变得安静而缓慢。清晨，河面上笼罩着一层薄雾，早点铺的蒸笼冒着热气，街上的行人裹紧了外套。桂花的香味从老街的院子里飘出来，银杏叶把石板路铺成了一片金黄。到了傍晚，家家户户的厨房里传出炒栗子和炖汤的香气，孩子们放学后在桥头追逐打闹。夏天的游客已经离开，小镇重新属于住在这里的人。老人们坐在茶馆里喝着热茶，聊着今年的收成，仿佛秋天的意义就在于让大家停下脚步，重新看看身边熟悉的一切。

Both paragraphs describe autumn as the season when a small town returns to its own residents, but the German one lingers on wine, markets and woodsmoke while the Chinese one finds the same quiet in osmanthus, ginkgo leaves and tea houses."""

U4 = """One more thing: summarize, in a few sentences, how the merge function would need to change if the intervals arrived as a stream that is too large to sort in memory."""

A4 = """If the intervals arrive as an unbounded or very large stream, the function can no longer sort everything up front. There are two common approaches. If the stream is already ordered by start time, which is often true for log files and event feeds, the same single pass works unchanged: keep only the current open interval in memory, extend it while new intervals overlap, and emit it as soon as an interval starts after its end. Memory use drops to a constant. If the stream is not ordered, you can perform an external sort, writing sorted chunks of a few million intervals to disk and then merging those chunks with a heap keyed on the start time, which feeds the ordered single pass. A third option, when approximate answers are acceptable, is to bucket the timeline into fixed windows and merge within each window, accepting that intervals crossing a boundary need a small second pass. In every case, the key idea stays the same: overlap can only be decided locally once the intervals are visited in order of their starting point."""


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir)
    msgs = [{"role": "user", "content": U1}, {"role": "assistant", "content": A1},
            {"role": "user", "content": U2}, {"role": "assistant", "content": A2},
            {"role": "user", "content": U3}, {"role": "assistant", "content": A3},
            {"role": "user", "content": U4}, {"role": "assistant", "content": A4}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == text, "round trip"
    parts = {}
    for name, s in (("english_prose", U1 + A1 + U4 + A4), ("python+math+json", U2 + A2), ("german+chinese", U3 + A3)):
        parts[name] = len(tok(s, add_special_tokens=False)["input_ids"])
    json.dump({"ids": ids, "n": len(ids), "parts_tokens": parts, "text": text}, open(out, "w"), ensure_ascii=False, indent=0)
    print(len(ids), parts)


if __name__ == "__main__":
    main()
