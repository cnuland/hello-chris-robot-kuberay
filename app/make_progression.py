#!/usr/bin/env python
"""Build the progression reel: caption each sampled episode video and concatenate them in training order.

Usage: app/venv/bin/python app/make_progression.py OUT.mp4 "caption one=path1.mp4" "caption two=path2.mp4" ...
Each clip is scaled to 640x480 at 10 fps (worker clips are 10 fps / every 2nd control step) with its caption burned in.
"""
import os, subprocess, sys, tempfile

FONT = next((f for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if os.path.exists(f)), None)


def caption_clip(src: str, text: str, dst: str) -> None:
    tf = dst + ".txt"; open(tf, "w").write(text)            # textfile + expansion=none: no escaping of % , : ( )
    draw = (f"drawtext=fontfile={FONT}:textfile={tf}:expansion=none:x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"
            if FONT else "null")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"scale=640:480,fps=10,{draw}", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-qp", "20", dst], check=True)


def main():
    out = sys.argv[1]; items = [a.split("=", 1) for a in sys.argv[2:]]
    tmp = tempfile.mkdtemp(); parts = []
    for i, (cap, path) in enumerate(items):
        dst = os.path.join(tmp, f"{i:03d}.mp4"); caption_clip(path, cap, dst); parts.append(dst)
    lst = os.path.join(tmp, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    print("wrote", out, f"({len(parts)} clips)")


if __name__ == "__main__":
    main()
