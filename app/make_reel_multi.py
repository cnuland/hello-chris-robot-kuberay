#!/usr/bin/env python
"""Labelled progression reel for the multi-part cell (~2 min): decision model/teacher -> DAgger Round N ... -> Finished.

Each clip is sped up and capped so the whole reel stays near two minutes; captions are burned in with drawtext
(textfile + expansion=none: no escaping issues). Usage: app/venv/bin/python app/make_reel_multi.py reel.json out.mp4
reel.json: [{"path": ..., "title": ..., "sub": ..., "speed": 4, "max_s": 8}, ...]
"""
import json, os, subprocess, sys, tempfile

FONT = next((f for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if os.path.exists(f)), None)


def clip(c, dst, tmp):
    t1 = os.path.join(tmp, os.path.basename(dst) + ".t1.txt"); open(t1, "w").write(c["title"])
    t2 = os.path.join(tmp, os.path.basename(dst) + ".t2.txt"); open(t2, "w").write(c.get("sub", ""))
    sp = float(c.get("speed", 1)); mx = float(c.get("max_s", 1e9)); start = float(c.get("start_s", 0))
    vf = (f"setpts=PTS/{sp},fps=20,scale=960:720,"
          f"drawtext=fontfile={FONT}:textfile={t1}:expansion=none:x=20:y=18:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=10,"
          f"drawtext=fontfile={FONT}:textfile={t2}:expansion=none:x=20:y=70:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", c["path"], "-vf", vf, "-t", str(mx), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dst], check=True)


def main():
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]; tmp = tempfile.mkdtemp(); parts = []
    for i, c in enumerate(spec):
        d = os.path.join(tmp, f"{i:02d}.mp4"); clip(c, d, tmp); parts.append(d)
    lst = os.path.join(tmp, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                         capture_output=True, text=True).stdout.strip()
    print(f"wrote {out}: {len(parts)} clips, {float(dur):.0f} s")


if __name__ == "__main__":
    main()
