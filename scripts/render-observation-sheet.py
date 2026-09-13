#!/usr/bin/env python3
"""Show each captured habitat frame with its real timestamp and physics state."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--start', type=int, required=True)
p.add_argument('--count', type=int, default=10)
args = p.parse_args()
frames = [json.loads(line) for line in (args.directory / 'frames.jsonl').read_text().splitlines()]
selected = [f for f in frames if args.start <= f['index'] < args.start + args.count]
assert selected, 'No captured frames in range'
font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 12)
tile_w, tile_h = 664, 337
sheet = Image.new('RGB', (2 * tile_w, ((len(selected) + 1) // 2) * tile_h), '#101613')
draw = ImageDraw.Draw(sheet)
for i, frame in enumerate(selected):
    x, y = (i % 2) * tile_w, (i // 2) * tile_h
    im = Image.open(args.directory / frame['file']).convert('RGB')
    im = im.crop((24, 171, 948, 567)).resize((650, 279))
    sheet.paste(im, (x + 7, y + 52))
    state = frame['state']
    title = f"{frame['index']:03d} | wall {frame['elapsedSeconds']:.1f}s | body {state.get('bodyTime', 0):.3f}s"
    up = state.get('upZ')
    details = f"{state.get('motion')} | up {up:.2f} | in={state.get('insideHabitat')}" if up is not None else str(state.get('motion'))
    draw.text((x + 8, y + 3), title, font=font, fill='#d1e7d8')
    draw.text((x + 8, y + 18), frame['version'][:74], font=font, fill='#ff9b53')
    draw.text((x + 8, y + 33), details, font=font, fill='#d1e7d8')
out = args.directory / 'contact-sheets'
out.mkdir(exist_ok=True)
target = out / f"{selected[0]['index']:03d}-{selected[-1]['index']:03d}.png"
sheet.save(target)
print(target)
