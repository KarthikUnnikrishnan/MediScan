"""
Phase 4 — Build medicines.sqlite (Model 3: Generic Finder)
Merges 3 CSVs → ~800K medicines → indexed SQLite database
"""

import os, sqlite3, pandas as pd
from pathlib import Path

M3_DIR  = r"D:\Coding Section\Mediscan\Datasets\Model3_Generic Finder"
DB_DIR  = r"D:\Coding Section\Mediscan\db_cache"
DB_PATH = os.path.join(DB_DIR, "medicines.sqlite")
os.makedirs(DB_DIR, exist_ok=True)

FILES = {
    'az'      : 'A-Z Medicine Dataset of India (250K medicines).csv',
    '1mg'     : 'India Medicines and Drug Info Dataset (1mg, 2025).csv',
    'mohneesh': 'Indian Medicine Data (Kaggle — Mohneesh).csv',
}

# ── Column name mapping: different CSVs use different names ───────────
# We map everything to: name, salt, manufacturer, price
NAME_COLS  = ['name','medicine_name','drug_name','product_name',
              'Medicine Name','Name','Drug Name']
SALT_COLS  = ['salt','composition','salt_composition','ingredient',
              'short_composition1','Composition','Salt Composition',
              'sub_category']
MFR_COLS   = ['manufacturer','company','manufacturer_name',
              'Manufacturer','Manufacturer Name','mfr_name']
PRICE_COLS = ['price','mrp','Price (INR)','price_per_unit',
              'Price','MRP','actual_price']

def find_col(df, candidates):
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_and_normalise(filepath, source_tag):
    print(f"\n  Loading {source_tag} ...")
    try:
        df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')

    print(f"  Raw columns: {df.columns.tolist()}")
    print(f"  Raw shape  : {df.shape}")

    name_col  = find_col(df, NAME_COLS)
    salt_col  = find_col(df, SALT_COLS)
    mfr_col   = find_col(df, MFR_COLS)
    price_col = find_col(df, PRICE_COLS)

    print(f"  Mapped → name={name_col}, salt={salt_col}, "
          f"mfr={mfr_col}, price={price_col}")

    out = pd.DataFrame()
    out['name']         = df[name_col].astype(str).str.strip() if name_col else ''
    out['salt']         = df[salt_col].astype(str).str.strip().str.lower() \
                          if salt_col else ''
    out['manufacturer'] = df[mfr_col].astype(str).str.strip() if mfr_col else ''
    out['price']        = pd.to_numeric(
                              df[price_col].astype(str).str.replace(
                                  r'[^\d.]', '', regex=True),
                              errors='coerce'
                          ) if price_col else None
    out['source'] = source_tag

    # Drop rows with no name
    out = out[out['name'].str.len() > 1]
    out = out[out['name'] != 'nan']
    print(f"  Clean rows : {len(out)}")
    return out


# ── Load all three CSVs ───────────────────────────────────────────────
print("=" * 55)
print("Building medicines.sqlite")
print("=" * 55)

frames = []
for tag, fname in FILES.items():
    path = os.path.join(M3_DIR, fname)
    if os.path.exists(path):
        frames.append(load_and_normalise(path, tag))
    else:
        print(f"  WARNING: not found — {fname}")

# ── Merge & deduplicate ───────────────────────────────────────────────
print("\nMerging all sources ...")
df_all = pd.concat(frames, ignore_index=True)
print(f"Before dedup : {len(df_all):,} rows")

# Deduplicate on (lowercase name + salt)
df_all['_key'] = (df_all['name'].str.lower().str.strip() + '|' +
                  df_all['salt'].str.strip())
df_all = df_all.drop_duplicates(subset='_key').drop(columns='_key')
df_all = df_all.reset_index(drop=True)
print(f"After dedup  : {len(df_all):,} rows")

# ── Write to SQLite ───────────────────────────────────────────────────
print(f"\nWriting to {DB_PATH} ...")
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.execute("DROP TABLE IF EXISTS medicines")
cur.execute("""
    CREATE TABLE medicines (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL,
        salt         TEXT,
        manufacturer TEXT,
        price        REAL,
        source       TEXT
    )
""")

# Insert in batches of 10,000
BATCH = 10_000
for i in range(0, len(df_all), BATCH):
    batch = df_all.iloc[i:i+BATCH]
    batch[['name','salt','manufacturer','price','source']].to_sql(
        'medicines', conn, if_exists='append', index=False
    )
    print(f"  Inserted {min(i+BATCH, len(df_all)):,} / {len(df_all):,}", end='\r')

# ── Create indexes for fast lookup ────────────────────────────────────
print("\n\nCreating indexes ...")
cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON medicines(name COLLATE NOCASE)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_salt ON medicines(salt)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON medicines(source)")
conn.commit()

# ── Verify ────────────────────────────────────────────────────────────
total = cur.execute("SELECT COUNT(*) FROM medicines").fetchone()[0]
sample = cur.execute(
    "SELECT name, salt, price FROM medicines WHERE price IS NOT NULL LIMIT 5"
).fetchall()
conn.close()

print(f"\n{'='*55}")
print(f"MEDICINES DB COMPLETE")
print(f"{'='*55}")
print(f"Total records : {total:,}")
print(f"Database size : {os.path.getsize(DB_PATH)//1024//1024} MB")
print(f"Saved to      : {DB_PATH}")
print(f"\nSample records:")
for row in sample:
    print(f"  {row[0][:40]:40s}  salt={str(row[1])[:30]:30s}  price=₹{row[2]}")