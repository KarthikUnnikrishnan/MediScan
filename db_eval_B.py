"""Model B + medicine database: how often is the right medicine name found?  (offline, run on your laptop)
Usage:  python db_eval_B.py <test.tsv> <errors_test.tsv> <medicines.sqlite>
test.tsv = D:\\MediScan_data\\ModelB\\final\\test.tsv   errors_test.tsv = from the Kaggle Output (the wrong predictions only)
Words the model read correctly are not in errors_test.tsv, so they count as read exactly right."""
import sys, re, csv, sqlite3, collections
from rapidfuzz import process, fuzz
test_p, err_p, db_p = sys.argv[1:4]
def base(n): return re.sub(r'\s+(tablet|tablets|capsule|capsules|syrup|injection|cream|gel|drops?|ointment|lotion|suspension|\d.*)$', '', str(n), flags=re.I).strip()
con = sqlite3.connect(f'file:{db_p}?mode=ro', uri=True)
rows = con.execute('select name, salt from medicines where name is not null').fetchall()
words = {base(n).lower() for n, s in rows if 3 <= len(base(n)) <= 30}
words |= {re.split(r'[\s(+]', str(s).strip())[0].lower() for n, s in rows if s}
words = sorted(w for w in words if w); wset = set(words); print(len(words), 'database words')
def rd(p):
    with open(p, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'); next(r, None); return [x for x in r if len(x) >= 3]
gold = {r[0]: (r[1].strip(), r[2]) for r in rd(test_p)}
pred = {r[0]: r[2].strip() for r in rd(err_p)}            # errors file columns: file, gold, prediction, source
stats = collections.defaultdict(lambda: collections.Counter())
for f, (g, src) in gold.items():
    gl = g.lower()
    if gl not in wset: stats[src]['gold_not_in_db'] += 1; continue     # a name the database does not know cannot be matched by any method
    p = pred.get(f, g).strip(); pl = p.lower()
    st = stats[src]; st['in_db'] += 1; st['ocr_exact'] += (pl == gl)
    top = [m[0] for m in process.extract(pl, words, scorer=fuzz.ratio, limit=5, score_cutoff=60)]
    st['top1'] += (top[:1] == [gl]); st['top3'] += (gl in top[:3]); st['top5'] += (gl in top[:5])
for src, st in list(stats.items()) + [('ALL', sum(stats.values(), collections.Counter()))]:
    n = st['in_db']
    if n: print(f"{src:10s} test words found in database: {n} (+{st['gold_not_in_db']} not in db) | model exact {100*st['ocr_exact']/n:.1f}% | "
                f"after database matching: top-1 {100*st['top1']/n:.1f}%  top-3 {100*st['top3']/n:.1f}%  top-5 {100*st['top5']/n:.1f}%")
