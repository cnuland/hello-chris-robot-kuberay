#!/usr/bin/env python
"""DiffusionGemma part-identification labels (v006).

Default mode "facts" (the recipe that worked in v004/v005): the inspection station's 48x48 height scan is turned into
plain-language measurements (inspection_facts.py) and DiffusionGemma names the part from them over the chat API -
text only, answers cached per distinct fact sheet. Mode "image" sends the RGB frame instead; on the 4x T4 TP=4
deployment the image path returned garbage tokens and desynchronised the tensor-parallel ranks, so it is kept only
as an option.

For each sample a random part rides the conveyor under the inspection gantry at a random yaw / lateral offset; at the
scan moment we keep (a) the RGB frame from the `inspect` camera and (b) the 16x16 height scan. DiffusionGemma (vLLM,
OpenAI chat API with an image) names the part from the RGB frame only; the true identity is kept for the audit.
The student's part-ID head is later trained on (height scan -> DiffusionGemma's label): a 26 B multimodal model
teaching a sub-millisecond classifier on a cheaper sensor.

Usage: app/venv/bin/python app/vision_labeler.py --n 300 --url http://127.0.0.1:18000 --out app/ignored/vision/v1
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DESCRIPTIONS = {
    "bracket": "a flat rectangular bar with a single red round fastener head in its centre",
    "cap": "an eight-sided (octagonal) cap with a flat round ring on top",
    "lbracket": "an L-shaped bracket (two arms at a right angle) with a row of small weld marks along one edge",
    "plate": "a square plate with two red round fastener heads on a diagonal",
    "rail": "a long, narrow straight bar with two small weld marks near its ends",
}


def prompt(parts) -> str:
    opts = "\n".join(f"- {p}: {DESCRIPTIONS[p]}" for p in parts)
    return ("This image is from an inspection camera looking straight down at a conveyor belt. Exactly one grey "
            f"metal part is on the belt. Which part is it?\n{opts}\n\nLook at the part's outline and features, then "
            "answer on a final line 'ANSWER: <part>'.")


def ask(url: str, img: np.ndarray, parts, timeout: float = 120.0):
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": "dgemma", "max_tokens": 96, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": prompt(parts)}]}]}
    t0 = time.time()
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout); r.raise_for_status()
        text = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
    except Exception as exc:
        return None, f"error: {exc}"[:200], time.time() - t0
    tail = text.rsplit("answer:", 1)[-1] if "answer:" in text else text
    hits = sorted((tail.find(p), p) for p in parts if p in tail)
    if "lbracket" in tail or "l-bracket" in tail or "l-shaped" in tail:
        hits = [(0, "lbracket")]
    return (hits[0][1] if hits else None), text, time.time() - t0


def ask_facts(url: str, facts, parts, timeout: float = 180.0):
    from inspection_facts import PART_SPECS
    opts = "\n".join(f"- {p}: {PART_SPECS[p]}" for p in parts)
    user = ("Measurements from the inspection scanner over the conveyor:\n" + "\n".join("- " + f for f in facts) +
            f"\n\nParts this cell builds:\n{opts}\n\nIn at most three short sentences compare the measurements with the "
            "parts, then name the part on a final line 'ANSWER: <part>'.")
    body = {"model": "dgemma", "max_tokens": 400, "messages": [
        {"role": "system", "content": "You are the inspection station of a robot assembly cell. Identify incoming parts from scanner measurements."},
        {"role": "user", "content": user}]}
    t0 = time.time()
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout); r.raise_for_status()
        text = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
    except Exception as exc:
        return None, f"error: {exc}"[:200], time.time() - t0
    if "answer:" not in text:
        return None, text, time.time() - t0                                    # no verdict: never guess from prose
    tail = text.rsplit("answer:", 1)[-1].replace("l-bracket", "lbracket").replace("l bracket", "lbracket")
    hits = sorted((tail.find(p), p) for p in parts if p in tail)
    if hits and hits[0][1] == "bracket" and "lbracket" in tail:
        hits = [(0, "lbracket")]
    return (hits[0][1] if hits else None), text, time.time() - t0


def make_samples(n: int, parts, seed: int, size: int = 336):
    """Ride random parts under the gantry; return images, scans, labels."""
    from multi_env import MultiAssemblyEnv
    env = MultiAssemblyEnv({"parts": tuple(parts), "order": "random"}); rng = np.random.default_rng(seed)
    imgs, scans, labels, his = [], [], [], []
    for k in range(n):
        env.reset(seed=int(rng.integers(1 << 30)))
        p = env.fed[0]
        for _ in range(60):
            env.step(np.zeros(5, np.float32))                                  # arm parked at home; belt carries the part
            if env.scan_part == p:
                break
        imgs.append(env.render_inspect(size)); scans.append(env.scans[p].copy()); labels.append(p); his.append(env.scans_hi[p].copy())
    make_samples.hi = np.stack(his)
    return np.stack(imgs), np.stack(scans), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--parts", default="bracket,cap,lbracket")
    ap.add_argument("--url", default=os.environ.get("DGEMMA_URL", "http://127.0.0.1:18000"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ignored", "vision", "v1"))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", default="facts", choices=["facts", "image"])
    a = ap.parse_args(); parts = a.parts.split(","); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); imgs, scans, labels = make_samples(a.n, parts, a.seed); his = make_samples.hi
    print(f"rendered {a.n} inspection samples in {time.time() - t0:.0f}s: {dict(zip(*np.unique(labels, return_counts=True)))}", flush=True)
    if a.mode == "image":
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda im: ask(a.url, im, parts), imgs))
    else:
        from inspection_facts import facts as scan_facts
        sheets = [tuple(scan_facts(h)) for h in his]; uniq = sorted(set(sheets))
        print(f"{len(uniq)} distinct fact sheets -> asking DiffusionGemma once each", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            ans = dict(zip(uniq, ex.map(lambda f: ask_facts(a.url, list(f), parts), uniq)))
        for f in uniq:
            print(f"  {ans[f][0]!s:9s} <- {' '.join(f)}", flush=True)
        res = [ans[f] for f in sheets]
        json.dump({" ".join(f): ans[f][0] for f in uniq}, open(os.path.join(a.out, "sheet_answers.json"), "w"), indent=1)
    dg = np.array([r[0] or "none" for r in res]); lat = [r[2] for r in res]
    ok = dg == labels; acc = float(ok.mean())
    conf = {t: {p: int(((labels == t) & (dg == p)).sum()) for p in parts + ["none"]} for t in parts}
    rep = {"n": a.n, "parts": parts, "accuracy": acc, "confusion_true_to_dgemma": conf,
           "latency_mean_s": float(np.mean(lat)), "examples": [{"true": str(l), "dgemma": str(d), "text": r[1][:300]}
                                                              for l, d, r in list(zip(labels, dg, res))[:12]]}
    np.savez_compressed(os.path.join(a.out, "vision_labels.npz"), scans=scans, true=labels, dgemma=dg)
    from PIL import Image
    for i in range(min(12, a.n)):
        Image.fromarray(imgs[i]).save(os.path.join(a.out, f"sample{i:02d}_{labels[i]}_dg-{dg[i]}.png"))
    json.dump(rep, open(os.path.join(a.out, "vision_report.json"), "w"), indent=2)
    print(f"DiffusionGemma vision: accuracy {acc:.1%} over {a.n} parts, mean latency {np.mean(lat):.1f}s")
    print(json.dumps(conf, indent=1))


if __name__ == "__main__":
    main()
