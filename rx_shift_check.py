"""Is the RxHandBD label file shifted by one image?  Uses the model's own readings (errors_test.tsv) to find out.
Usage: python rx_shift_check.py <test.tsv> <errors_test.tsv>"""
import sys, re, csv, collections
from rapidfuzz import fuzz
def rd(p):
    with open(p, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'); next(r, None); return [x for x in r if len(x) >= 3]
num = lambda f: int(re.search(r'(\d+)', f).group(1))
rows = sorted([r for r in rd(sys.argv[1]) if r[2] == 'rxhandbd'], key=lambda r: num(r[0]))
pred = {r[0]: r[2].strip() for r in rd(sys.argv[2])}
g = [r[1].strip().lower() for r in rows]; p = [pred.get(r[0], r[1]).strip().lower() for r in rows]
ok = lambda a, b: fuzz.ratio(a, b) >= 85
cnt = collections.Counter(); blocks = collections.defaultdict(collections.Counter)
for k in range(len(rows)):
    own = ok(p[k], g[k]); nxt = k + 1 < len(g) and ok(p[k], g[k + 1]); prv = k > 0 and ok(p[k], g[k - 1])
    lab = 'matches its own label' if own else ('matches the NEXT image label (shifted)' if nxt else ('matches the PREVIOUS image label (shifted)' if prv else 'matches no neighbour'))
    cnt[lab] += 1; blocks[k // 100][lab] += 1
n = len(rows); print(n, 'RxHandBD test words')
for k, v in cnt.most_common(): print(f'  {v:5d} ({100*v/n:4.1f}%)  {k}')
own = cnt['matches its own label']; fixed = own + cnt['matches the NEXT image label (shifted)'] + cnt['matches the PREVIOUS image label (shifted)']
print(f'\nexact-ish match with its own label: {100*own/n:.1f}%   |  if shifted labels are realigned: {100*fixed/n:.1f}%   (upper estimate: a neighbour can match by chance)')
print('\nper block of 100 images (own / shifted / none):')
for b in sorted(blocks):
    c = blocks[b]; sh = sum(v for k, v in c.items() if 'shifted' in k); t = sum(c.values())
    print(f'  images {b*100:4d}-{b*100+t-1:4d}: own {c["matches its own label"]:3d} | shifted {sh:3d} | none {c["matches no neighbour"]:3d}')
