"""Keep the postings you applied to or passed on, in a SQLite file next to the code (gitignored).

The database is required: `init` creates it, and everything else fails with a StoreError if it is missing or broken."""
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from assay import paths

TABLES = {
    "postings": """
        CREATE TABLE postings (
            id            INTEGER PRIMARY KEY,
            created_at    TEXT NOT NULL,
            source        TEXT NOT NULL,   -- 'clipboard' or the file path given
            raw_sha256    TEXT NOT NULL,   -- the same posting has the same hash, so it can be found again
            raw_text      TEXT NOT NULL,
            minifier      TEXT NOT NULL,   -- always 'claude' for now
            minified_text TEXT NOT NULL,
            decision      TEXT NOT NULL CHECK (decision IN ('applied', 'passed'))
        )""",
    "sentences": """
        CREATE TABLE sentences (
            posting_id INTEGER NOT NULL REFERENCES postings(id) ON DELETE CASCADE,
            n          INTEGER NOT NULL,   -- position in the posting's sentence split, from 1
            text       TEXT NOT NULL,
            label      TEXT NOT NULL CHECK (label IN ('H', 'R', 'F', 'W', 'X')),  -- H R F W keep, X cut
            PRIMARY KEY (posting_id, n)
        )""",
    "resume": """
        CREATE TABLE resume (
            id         INTEGER PRIMARY KEY CHECK (id = 1),   -- one row: the current résumé
            text       TEXT NOT NULL,
            source     TEXT NOT NULL,   -- the file it was loaded from
            updated_at TEXT NOT NULL
        )""",
}


class StoreError(Exception):
    """The database is missing or a database operation failed."""


def init(path=None):
    """Create the database, or add any tables it is missing without touching its data.

    Returns (existed, created): whether the file was already there, and the names of the tables added."""
    path = Path(path or paths.db_path())
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        if not existed:
            os.chmod(path, 0o600)  # your résumé and postings: readable by you only (ignored on Windows)
        have = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        created = [name for name in TABLES if name not in have]
        for name in created:
            db.execute(TABLES[name])
        db.commit()
        db.close()
    except (sqlite3.Error, OSError) as e:
        raise StoreError(f"Could not set up the database at {path}: {e}")
    return existed, created


def connect(path=None):
    """Open the existing database. Raises StoreError if it is missing, out of date, or not an assay database."""
    path = Path(path or paths.db_path())
    if not path.exists():
        raise StoreError(f"No database at {path}. Run `assay init` first.")
    try:
        db = sqlite3.connect(path)
        db.execute("PRAGMA foreign_keys = ON")  # so removing a posting removes its sentence labels
        have = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = [name for name in TABLES if name not in have]
        if missing:
            raise StoreError(f"The database is out of date (missing: {', '.join(missing)}). Run `assay init`.")
        db.execute("SELECT decision FROM postings LIMIT 1")  # fails if the table has the wrong columns
    except sqlite3.Error as e:
        raise StoreError(f"Could not open the database at {path}: {e}")
    return db


def _sha256(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_posting(db, source, raw, minifier, minified, decision):
    """Save a decided posting. Returns the row id."""
    try:
        cur = db.execute(
            "INSERT INTO postings (created_at, source, raw_sha256, raw_text, minifier, minified_text, decision)"
            " VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), source,
             _sha256(raw), raw, minifier, minified, decision))
        db.commit()
    except sqlite3.Error as e:
        raise StoreError(f"Could not save the posting: {e}")
    return cur.lastrowid


def find_posting(db, raw):
    """The most recently saved copy of this exact posting, as a dict, or None."""
    try:
        row = db.execute(
            "SELECT id, minifier, minified_text, decision FROM postings WHERE raw_sha256=? ORDER BY id DESC LIMIT 1",
            (_sha256(raw),)).fetchone()
    except sqlite3.Error as e:
        raise StoreError(f"Could not look up the posting: {e}")
    return None if row is None else dict(zip(("id", "minifier", "minified_text", "decision"), row))


def set_decision(db, posting_id, decision, minifier=None, minified=None):
    """Mark a saved posting applied or passed. If new minified text is given it replaces the old."""
    try:
        if minified is None:
            db.execute("UPDATE postings SET decision=? WHERE id=?", (decision, posting_id))
        else:
            db.execute(
                "UPDATE postings SET decision=?, minifier=?, minified_text=? WHERE id=?",
                (decision, minifier, minified, posting_id))
        db.commit()
    except sqlite3.Error as e:
        raise StoreError(f"Could not record the decision: {e}")


def list_postings(db):
    """Every saved posting, oldest first, as dicts."""
    try:
        rows = db.execute("SELECT id, created_at, decision, minified_text FROM postings ORDER BY id").fetchall()
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the postings: {e}")
    return [dict(zip(("id", "created_at", "decision", "minified_text"), row)) for row in rows]


def get_posting(db, posting_id):
    """One saved posting as a dict, or None if there is no such id."""
    try:
        row = db.execute(
            "SELECT id, created_at, source, decision, minified_text FROM postings WHERE id=?",
            (posting_id,)).fetchone()
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the posting: {e}")
    return None if row is None else dict(zip(("id", "created_at", "source", "decision", "minified_text"), row))


def delete_posting(db, posting_id):
    try:
        db.execute("DELETE FROM postings WHERE id=?", (posting_id,))
        db.commit()
    except sqlite3.Error as e:
        raise StoreError(f"Could not remove the posting: {e}")


def examples(db, decision):
    """The minified text of every posting with this decision, oldest first."""
    try:
        rows = db.execute("SELECT minified_text FROM postings WHERE decision=? ORDER BY id", (decision,)).fetchall()
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the examples: {e}")
    return [row[0] for row in rows]


def all_postings(db):
    """Every saved posting with every column, oldest first, as dicts."""
    try:
        cur = db.execute("SELECT * FROM postings ORDER BY id")
        names = [col[0] for col in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the postings: {e}")


def unlabeled_postings(db):
    """(id, raw_text) of every posting that has no sentence labels yet, oldest first."""
    try:
        return db.execute(
            "SELECT id, raw_text FROM postings WHERE id NOT IN (SELECT posting_id FROM sentences) ORDER BY id"
        ).fetchall()
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the postings: {e}")


def save_labels(db, posting_id, sentences, labels):
    """Store one label per sentence for a posting, all or nothing."""
    try:
        with db:
            db.executemany(
                "INSERT INTO sentences (posting_id, n, text, label) VALUES (?,?,?,?)",
                [(posting_id, n, text, label) for n, (text, label) in enumerate(zip(sentences, labels), 1)])
    except sqlite3.Error as e:
        raise StoreError(f"Could not save the labels: {e}")


def labeled_postings(db):
    """Every labeled posting as a list of (sentence, label) pairs, oldest posting first."""
    try:
        rows = db.execute("SELECT posting_id, text, label FROM sentences ORDER BY posting_id, n").fetchall()
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the labels: {e}")
    postings = {}
    for posting_id, text, label in rows:
        postings.setdefault(posting_id, []).append((text, label))
    return list(postings.values())


def get_resume(db):
    """The saved résumé as a dict (text, source, updated_at), or None if none has been saved."""
    try:
        row = db.execute("SELECT text, source, updated_at FROM resume WHERE id=1").fetchone()
    except sqlite3.Error as e:
        raise StoreError(f"Could not read the résumé: {e}")
    return None if row is None else dict(zip(("text", "source", "updated_at"), row))


def set_resume(db, text, source):
    """Save the résumé, replacing any earlier one (which is not kept). Returns the replaced text, or None."""
    previous = get_resume(db)
    try:
        db.execute(
            "INSERT OR REPLACE INTO resume (id, text, source, updated_at) VALUES (1, ?, ?, ?)",
            (text, source, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        db.commit()
    except sqlite3.Error as e:
        raise StoreError(f"Could not save the résumé: {e}")
    return None if previous is None else previous["text"]
