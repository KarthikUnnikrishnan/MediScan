"""
Phase 4 — Build drugs.sqlite (Model 4: Side Effects + DDI)
Sources: SIDER 4.1 (FDA), DrugBank interactions, Mohneesh side effects
TWOSIDES loaded in chunks (704MB compressed)
"""

import os, sqlite3, gzip, pandas as pd

M4_DIR  = r"D:\Coding Section\Mediscan\Datasets\Model4_Side Effects + Interactions"
M3_DIR  = r"D:\Coding Section\Mediscan\Datasets\Model3_Generic Finder"
DB_PATH = r"D:\Coding Section\Mediscan\db_cache\drugs.sqlite"

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

print("=" * 55)
print("Building drugs.sqlite")
print("=" * 55)

# ── TABLE 1: side_effects (from SIDER) ────────────────────────────────
print("\n[1/4] Loading SIDER side effects ...")
cur.execute("DROP TABLE IF EXISTS side_effects")
cur.execute("""
    CREATE TABLE side_effects (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        drug_name     TEXT,
        stitch_id     TEXT,
        se_name       TEXT,
        meddra_type   TEXT
    )
""")

se_path = os.path.join(M4_DIR, "meddra_all_se.tsv.gz")
if os.path.exists(se_path):
    with gzip.open(se_path, 'rt', encoding='utf-8') as f:
        se_df = pd.read_csv(f, sep='\t', header=None,
                            names=['stitch_flat','stitch_stereo',
                                   'umls_label','meddra_type',
                                   'umls_se','se_name'])
    print(f"  SIDER SE rows: {len(se_df):,}")
    se_df[['stitch_flat','se_name','meddra_type']].rename(
        columns={'stitch_flat':'stitch_id'}
    ).assign(drug_name='').to_sql(
        'side_effects', conn, if_exists='append', index=False
    )
    print(f"  Inserted {len(se_df):,} side effect records")
else:
    print("  WARNING: meddra_all_se.tsv.gz not found")

# ── TABLE 2: se_frequency (from SIDER freq file) ──────────────────────
print("\n[2/4] Loading SIDER frequency labels ...")
cur.execute("DROP TABLE IF EXISTS se_frequency")
cur.execute("""
    CREATE TABLE se_frequency (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        stitch_id       TEXT,
        se_name         TEXT,
        freq_lower      REAL,
        freq_upper      REAL,
        freq_label      TEXT
    )
""")

freq_path = os.path.join(M4_DIR, "meddra_freq.tsv.gz")
if os.path.exists(freq_path):
    with gzip.open(freq_path, 'rt', encoding='utf-8') as f:
        freq_df = pd.read_csv(f, sep='\t', header=None,
                              names=['stitch_flat','stitch_stereo',
                                     'umls_label','placebo',
                                     'freq_lower','freq_upper',
                                     'meddra_type','umls_se','se_name'])
    # Assign human-readable frequency label
    def freq_label(row):
        lo = row.get('freq_lower', 0) or 0
        if lo >= 0.10:  return 'Common'
        if lo >= 0.01:  return 'Uncommon'
        return 'Rare'

    freq_df['freq_label'] = freq_df.apply(freq_label, axis=1)
    freq_df[['stitch_flat','se_name','freq_lower','freq_upper','freq_label']].rename(
        columns={'stitch_flat':'stitch_id'}
    ).to_sql('se_frequency', conn, if_exists='append', index=False)
    print(f"  Inserted {len(freq_df):,} frequency records")
else:
    print("  WARNING: meddra_freq.tsv.gz not found")

# ── TABLE 3: drug_stitch_map (SIDER drug name → stitch ID) ────────────
print("\n[3/4] Loading SIDER drug name map ...")
cur.execute("DROP TABLE IF EXISTS drug_stitch_map")
cur.execute("""
    CREATE TABLE drug_stitch_map (
        stitch_id  TEXT,
        drug_name  TEXT
    )
""")

names_path = os.path.join(M4_DIR, "drug_names.tsv")
if os.path.exists(names_path):
    names_df = pd.read_csv(names_path, sep='\t', header=None,
                           names=['stitch_id','drug_name'])
    names_df.to_sql('drug_stitch_map', conn, if_exists='append', index=False)
    print(f"  Inserted {len(names_df):,} drug name mappings")
else:
    print("  WARNING: drug_names.tsv not found")

# ── TABLE 4: drug_interactions (DrugBank) ─────────────────────────────
print("\n[4/5] Loading DrugBank interactions ...")
cur.execute("DROP TABLE IF EXISTS drug_interactions")
cur.execute("""
    CREATE TABLE drug_interactions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        drug1       TEXT,
        drug2       TEXT,
        description TEXT
    )
""")

ddi_path = os.path.join(M4_DIR, "db_drug_interactions.csv")
if os.path.exists(ddi_path):
    ddi_df = pd.read_csv(ddi_path, low_memory=False)
    print(f"  DrugBank columns: {ddi_df.columns.tolist()}")

    # Auto-detect column names
    cols = ddi_df.columns.str.lower().tolist()
    d1   = ddi_df.columns[cols.index(next(c for c in cols if 'drug' in c and '1' in c))]
    d2   = ddi_df.columns[cols.index(next(c for c in cols if 'drug' in c and '2' in c))]
    desc = ddi_df.columns[cols.index(next(
        (c for c in cols if 'desc' in c or 'interaction' in c or 'effect' in c),
        cols[-1]
    ))]

    ddi_df[[d1, d2, desc]].rename(
        columns={d1:'drug1', d2:'drug2', desc:'description'}
    ).to_sql('drug_interactions', conn, if_exists='append', index=False)
    print(f"  Inserted {len(ddi_df):,} interaction pairs")
else:
    print("  WARNING: db_drug_interactions.csv not found")

# ── TABLE 5: TWOSIDES in chunks ────────────────────────────────────────
print("\n[5/5] Loading TWOSIDES (704MB — processing in chunks) ...")
cur.execute("DROP TABLE IF EXISTS twosides")
cur.execute("""
    CREATE TABLE twosides (
        drug1     TEXT,
        drug2     TEXT,
        se_name   TEXT,
        freq      REAL
    )
""")

two_path = os.path.join(M4_DIR, "TWOSIDES.csv.gz")
if os.path.exists(two_path):
    chunk_size = 50_000
    total_rows = 0
    try:
        for chunk in pd.read_csv(two_path, chunksize=chunk_size,
                                 low_memory=False, compression='gzip'):
            if total_rows == 0:
                print(f"  TWOSIDES columns: {chunk.columns.tolist()}")
            cols = chunk.columns.str.lower().tolist()

            # Find drug1, drug2, side_effect columns
            def find(keywords):
                for kw in keywords:
                    matches = [c for c in cols if kw in c]
                    if matches: return chunk.columns[cols.index(matches[0])]
                return None

            d1   = find(['drug1','drug_1','item_id_1','drug_a'])
            d2   = find(['drug2','drug_2','item_id_2','drug_b'])
            se   = find(['side_effect','effect_name','event','se_name'])
            freq = find(['freq','proportion','count','prr'])

            sub = pd.DataFrame()
            sub['drug1']   = chunk[d1].astype(str) if d1 else ''
            sub['drug2']   = chunk[d2].astype(str) if d2 else ''
            sub['se_name'] = chunk[se].astype(str) if se else ''
            sub['freq']    = pd.to_numeric(chunk[freq], errors='coerce') \
                             if freq else None

            sub.to_sql('twosides', conn, if_exists='append', index=False)
            total_rows += len(chunk)
            print(f"  TWOSIDES: {total_rows:,} rows loaded ...", end='\r')

        print(f"\n  TWOSIDES complete: {total_rows:,} rows")
    except Exception as e:
        print(f"  TWOSIDES error: {e} — skipping")
else:
    print("  WARNING: TWOSIDES.csv.gz not found")

# ── Indexes ────────────────────────────────────────────────────────────
print("\nCreating indexes ...")
cur.execute("CREATE INDEX IF NOT EXISTS idx_se_stitch  ON side_effects(stitch_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_freq_stitch ON se_frequency(stitch_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_map_name   ON drug_stitch_map(drug_name COLLATE NOCASE)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_ddi_d1     ON drug_interactions(drug1 COLLATE NOCASE)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_ddi_d2     ON drug_interactions(drug2 COLLATE NOCASE)")
conn.commit()
conn.close()

db_size = os.path.getsize(DB_PATH) // 1024 // 1024
print(f"\n{'='*55}")
print(f"DRUGS DB COMPLETE")
print(f"{'='*55}")
print(f"Database size : {db_size} MB")
print(f"Saved to      : {DB_PATH}")