-- MediScan SQLite Database Schemas
-- Reference for AI analysis of scanner/ml.py queries

-- ========================================================
-- Database: medicines.sqlite
-- Used for: Medicine brand name lookup, pricing, and generic alternatives
-- ========================================================

CREATE TABLE medicines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    salt         TEXT,
    manufacturer TEXT,
    price        REAL,
    source       TEXT
);

-- ========================================================
-- Database: drugs.sqlite
-- Used for: Side effects, frequency of adverse reactions, drug interactions
-- ========================================================

CREATE TABLE drug_stitch_map (
    stitch_id TEXT,
    drug_name TEXT,
    norm_id   TEXT
);

CREATE TABLE side_effects (
    stitch_id   TEXT,
    se_name     TEXT,
    meddra_type TEXT,
    norm_id     TEXT
);

CREATE TABLE se_frequency (
    stitch_id  TEXT,
    se_name    TEXT,
    freq_lower REAL,
    freq_upper REAL,
    freq_label TEXT,
    norm_id    TEXT
);

CREATE TABLE drug_interactions (
    drug1       TEXT,
    drug2       TEXT,
    description TEXT
);
