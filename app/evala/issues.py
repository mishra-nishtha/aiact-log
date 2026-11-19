"""User-defined RQ3 failure-taxonomy records (categories + issues). Annotation data only: kept in results/issues.db,
separate from the audit log, and NOT read by the LLM pipeline."""
import os
import sqlite3
import threading
import time

from .data import UNITS

CONDS = ("B2", "C3", "C4")
UNIT_IDS = {u["id"] for u in UNITS}
BUILTIN = [  # paper §5.3 (RQ3): cross-references, exceptions, implicit actors, temporal constraints, scope conditions
 ("Cross-references", "RQ3 text characteristic: the provision refers to other provisions."),
 ("Exceptions", "RQ3 text characteristic: the provision contains an exception or reservation."),
 ("Implicit actors", "RQ3 text characteristic: the provision does not name who must act."),
 ("Temporal constraints", "RQ3 text characteristic: the provision constrains time or duration."),
 ("Scope conditions", "RQ3 text characteristic: the provision applies only under a stated condition."),
]
SCHEMA = """
CREATE TABLE IF NOT EXISTS issue_categories(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
 description TEXT NOT NULL, builtin INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS issues(id INTEGER PRIMARY KEY AUTOINCREMENT, category_id INTEGER NOT NULL REFERENCES issue_categories(id),
 unit_id TEXT NOT NULL, cond TEXT, title TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
"""

class IssueStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.lock = path, threading.Lock()
        def init(c):
            c.executescript(SCHEMA)
            for n, d in BUILTIN:
                c.execute("INSERT OR IGNORE INTO issue_categories(name,description,builtin,created_at) VALUES(?,?,1,?)", (n, d, time.time()))
        self._run(init)

    def _run(self, fn):
        with self.lock:
            c = sqlite3.connect(self.path)
            try:
                r = fn(c); c.commit(); return r
            finally:
                c.close()

    def categories(self):
        def f(c):
            rows = c.execute("SELECT k.id,k.name,k.description,k.builtin,k.created_at,COUNT(i.id) FROM issue_categories k "
                             "LEFT JOIN issues i ON i.category_id=k.id GROUP BY k.id ORDER BY k.id").fetchall()
            return [{"id": r[0], "name": r[1], "description": r[2], "builtin": bool(r[3]), "created_at": r[4], "issues": r[5]} for r in rows]
        return self._run(f)

    def add_category(self, name, description):
        name, description = (name or "").strip(), (description or "").strip()
        if not 2 <= len(name) <= 60: raise ValueError("name must be 2-60 characters")
        if not 1 <= len(description) <= 500: raise ValueError("description must be 1-500 characters")
        def f(c):
            try:
                cur = c.execute("INSERT INTO issue_categories(name,description,builtin,created_at) VALUES(?,?,0,?)", (name, description, time.time()))
            except sqlite3.IntegrityError:
                raise ValueError(f"category {name!r} already exists")
            return {"id": cur.lastrowid, "name": name}
        return self._run(f)

    def add_issue(self, category_id, unit_id, cond, title, detail=""):
        title, detail, cond = (title or "").strip(), (detail or "").strip(), (cond or "").strip() or None
        if unit_id not in UNIT_IDS: raise ValueError(f"unknown unit {unit_id!r}")
        if cond is not None and cond not in CONDS: raise ValueError(f"cond must be one of {CONDS} or empty")
        if not 3 <= len(title) <= 120: raise ValueError("title must be 3-120 characters")
        if len(detail) > 2000: raise ValueError("detail must be at most 2000 characters")
        def f(c):
            if c.execute("SELECT 1 FROM issue_categories WHERE id=?", (category_id,)).fetchone() is None:
                raise ValueError("unknown category")
            cur = c.execute("INSERT INTO issues(category_id,unit_id,cond,title,detail,created_at) VALUES(?,?,?,?,?,?)",
                            (category_id, unit_id, cond, title, detail, time.time()))
            return {"id": cur.lastrowid}
        return self._run(f)

    def issues(self, category_id=None):
        def f(c):
            q = ("SELECT i.id,i.category_id,k.name,i.unit_id,i.cond,i.title,i.detail,i.created_at FROM issues i "
                 "JOIN issue_categories k ON k.id=i.category_id")
            rows = c.execute(q + (" WHERE i.category_id=?" if category_id is not None else "") + " ORDER BY i.id DESC",
                             (category_id,) if category_id is not None else ()).fetchall()
            return [{"id": r[0], "category_id": r[1], "category": r[2], "unit_id": r[3], "cond": r[4], "title": r[5],
                     "detail": r[6], "created_at": r[7]} for r in rows]
        return self._run(f)