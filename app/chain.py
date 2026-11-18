"""Tamper-evident, privacy-aware audit log: case-study implementation for EU AI Act Art. 12 / 19(1).
Hash chain + Ed25519 signatures + Merkle checkpoints + external-anchor file + per-subject
encryption (crypto-shredding) + retention floor + async writer."""
import hashlib
import hmac
import json
import math
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import serialization as ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

GENESIS = "0" * 64
MIN_RETENTION_DAYS = 183  # ~6 months, Art. 19(1) floor
EVENT_TYPES = {
    "inference": "Art. 12(1)",
    "risk_flag": "Art. 12(2)(a)",
    "model_update": "Art. 12(2)(a) substantial modification",
    "pmm_signal": "Art. 12(2)(b) / Art. 72",
    "operator_action": "Art. 12(2)(c) / Art. 26(5)",
    "human_override": "Art. 12(2)(c) / Art. 14",
    "biometric_match": "Art. 12(3) (Annex III 1(a))",
    "logger_gap": "Art. 12(1) (writer-recorded loss notice)",
}
BIOMETRIC_FIELDS = ("session_start", "session_end", "reference_db", "matched_input_ref", "verifier_ids")
TAMPER_KINDS = ("modify", "delete", "truncate", "backdate", "reorder", "insert", "replay")
COLS = ("seq", "ts", "event_type", "system_id", "model_version", "subject", "input_hash",
        "output_hash", "meta", "prev_hash", "entry_hash", "sig")
RECK = ("seq", "ts", "event_type", "system_id", "model_version", "subject", "input_hash", "output_hash")
SCHEMA = """
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, ts REAL, event_type TEXT, system_id TEXT,
 model_version TEXT, subject TEXT, input_hash TEXT, output_hash TEXT, meta TEXT, prev_hash TEXT,
 entry_hash TEXT, sig TEXT);
CREATE TABLE IF NOT EXISTS checkpoints(id INTEGER PRIMARY KEY AUTOINCREMENT, from_seq INT, to_seq INT,
 root TEXT, head_hash TEXT, ts REAL, sig TEXT);
CREATE TABLE IF NOT EXISTS pii(seq INTEGER PRIMARY KEY, subject TEXT, blob BLOB);
CREATE TABLE IF NOT EXISTS subject_keys(subject TEXT PRIMARY KEY, key BLOB);
CREATE TABLE IF NOT EXISTS event_ids(event_id TEXT PRIMARY KEY, seq INTEGER);
"""

def h(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canon(o) -> bytes: return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()
def _node(a: bytes, b: bytes) -> bytes: return hashlib.sha256(b"\x01" + a + b).digest()

def merkle_root(hashes):
    if not hashes: return GENESIS
    lvl = [bytes.fromhex(x) for x in hashes]
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        lvl = [_node(lvl[i], lvl[i + 1]) for i in range(0, len(lvl), 2)]
    return lvl[0].hex()

def merkle_path(hashes, idx):
    """Inclusion proof for leaf `idx`: list of [sibling_hex, 'left'|'right']."""
    lvl = [bytes.fromhex(x) for x in hashes]; path = []
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        sib = idx ^ 1
        path.append([lvl[sib].hex(), "left" if sib < idx else "right"])
        lvl = [_node(lvl[i], lvl[i + 1]) for i in range(0, len(lvl), 2)]; idx //= 2
    return path

def verify_path(leaf, path, root):
    cur = bytes.fromhex(leaf)
    for sib, side in path:
        s = bytes.fromhex(sib); cur = _node(s, cur) if side == "left" else _node(cur, s)
    return cur.hex() == root

def _secret(p):
    if not os.path.exists(p):
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f: f.write(os.urandom(32))
        except FileExistsError:
            pass
    with open(p, "rb") as f: b = f.read()
    if len(b) != 32: raise RuntimeError(f"{p}: expected 32 bytes")
    return b

def _wipe(path):
    try:
        n = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.write(b"\0" * n); f.flush(); os.fsync(f.fileno())
    except OSError: pass
    try: os.remove(path)
    except FileNotFoundError: pass

def _epoch(v):
    if isinstance(v, bool): raise ValueError("invalid time value")
    if isinstance(v, (int, float)) and math.isfinite(v) and v > 0: return float(v)
    if isinstance(v, str) and v.strip():
        try: return datetime.fromisoformat(v.strip().replace("Z", "+00:00")).timestamp()
        except ValueError: pass
    raise ValueError("invalid time value")

def validate_event(e):
    """Raises ValueError for any record that must not enter the log."""
    if not isinstance(e, dict): raise ValueError("event must be an object")
    t = e.get("event_type")
    if not isinstance(t, str) or t not in EVENT_TYPES: raise ValueError(f"unknown event_type {t!r}")
    for k in ("system_id", "model_version", "subject"):
        v = e.get(k)
        if v is not None and (not isinstance(v, str) or not v.strip() or len(v) > 128):
            raise ValueError(f"{k} must be a non-empty string (max 128 chars)")
    for k in ("input", "output"):
        v = e.get(k, "")
        if not isinstance(v, str) or len(v) > 100_000: raise ValueError(f"{k} must be a string (max 100000 chars)")
    meta = e.get("meta") or {}
    if not isinstance(meta, dict): raise ValueError("meta must be an object")
    try: json.dumps(meta, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError): raise ValueError("meta must be JSON-serialisable without NaN/Infinity")
    if e.get("ts") is not None:
        if _epoch(e["ts"]) > time.time() + 300: raise ValueError("ts is in the future")
    eid = e.get("event_id")
    if eid is not None and (not isinstance(eid, str) or not 0 < len(eid) <= 64
                            or not all(c.isalnum() or c in "_.:-" for c in eid)):
        raise ValueError("event_id must be 1-64 chars of [A-Za-z0-9_.:-]")
    if t == "biometric_match":
        miss = [f for f in BIOMETRIC_FIELDS if f not in meta]
        if miss: raise ValueError(f"Art.12(3) fields missing: {miss}")
        try: s, en = _epoch(meta["session_start"]), _epoch(meta["session_end"])
        except ValueError: raise ValueError("Art.12(3)(a): session_start/session_end must be epoch seconds or ISO-8601")
        if en < s: raise ValueError("Art.12(3)(a): session_end precedes session_start")
        for f in ("reference_db", "matched_input_ref"):
            if not isinstance(meta[f], str) or not meta[f].strip(): raise ValueError(f"Art.12(3): {f} must be a non-empty string")
        v = meta["verifier_ids"]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            raise ValueError("Art.12(3)(d): verifier_ids must be a non-empty list of identifiers")

class LogStore:
    def __init__(self, d):
        os.makedirs(d, exist_ok=True)
        self.d = d
        self.lock = threading.RLock()
        self.db = sqlite3.connect(os.path.join(d, "log.db"), check_same_thread=False)
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript(SCHEMA)
        self.sk = Ed25519PrivateKey.from_private_bytes(_secret(os.path.join(d, "signing.key")))
        self.pk = self.sk.public_key()
        self.salt = _secret(os.path.join(d, "salt.bin"))
        self.anchor_path = os.path.join(d, "anchors.jsonl")  # simulated external witness
        self.flag_path = os.path.join(d, "tampered.flag")
        self.bak_path = os.path.join(d, "log.bak")
        self.tampered = os.path.exists(self.flag_path) or os.path.exists(self.bak_path)
        self._load_head()

    @property
    def pubkey_hex(self):
        return self.pk.public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw).hex()

    def close(self):
        self.db.close()

    def _load_head(self):
        r = self.db.execute("SELECT seq, entry_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        self.head = (r[0], r[1]) if r else (0, GENESIS)

    def _hm(self, s): return hmac.new(self.salt, s.encode(), hashlib.sha256).hexdigest()
    def _pseudo(self, subject): return "p_" + self._hm("subject:" + subject)[:24]

    def _resolve(self, subject):
        """Accept a pseudonym (as shown in the log) or a raw identifier."""
        r = self.db.execute("SELECT 1 FROM subject_keys WHERE subject=? UNION SELECT 1 FROM events WHERE subject=? LIMIT 1",
                            (subject, subject)).fetchone()
        return subject if r else self._pseudo(subject)

    def _require_clean(self):
        if self.tampered: raise RuntimeError("log is in the demo-tampered state; restore it first")

    # ---------- append ----------
    def append_batch(self, evs):
        evs = list(evs)
        for e in evs: validate_event(e)
        with self.lock:
            self._require_clean()
            seq, prev = self.head
            out = []
            try:
                cur = self.db.cursor()
                for e in evs:
                    t = e["event_type"]
                    meta = dict(e.get("meta") or {})
                    eid = e.get("event_id")
                    if eid is not None:
                        meta["event_id"] = eid
                        try: cur.execute("INSERT INTO event_ids VALUES(?,?)", (eid, seq + 1))
                        except sqlite3.IntegrityError: raise ValueError(f"duplicate event_id {eid!r}")
                    seq += 1
                    subj = self._pseudo(e.get("subject") or "anon")
                    ts = _epoch(e["ts"]) if e.get("ts") is not None else time.time()
                    rec = {"seq": seq, "ts": ts, "event_type": t,
                           "system_id": e.get("system_id") or "sys-1", "model_version": e.get("model_version") or "m-1.0",
                           "subject": subj, "input_hash": self._hm(e.get("input", "")),
                           "output_hash": self._hm(e.get("output", "")), "meta": meta}
                    eh = h(canon(rec) + prev.encode())
                    sig = self.sk.sign(bytes.fromhex(eh)).hex()
                    cur.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                (seq, rec["ts"], t, rec["system_id"], rec["model_version"], subj, rec["input_hash"],
                                 rec["output_hash"], json.dumps(meta, sort_keys=True), prev, eh, sig))
                    if e.get("input") or e.get("output"):
                        self._put_pii(cur, seq, subj, {"input": e.get("input", ""), "output": e.get("output", "")})
                    prev = eh
                    out.append(seq)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            self.head = (seq, prev)
            return out

    # ---------- privacy ----------
    def _put_pii(self, cur, seq, subj, payload):
        r = cur.execute("SELECT key FROM subject_keys WHERE subject=?", (subj,)).fetchone()
        if r is None:
            k = Fernet.generate_key()
            cur.execute("INSERT INTO subject_keys VALUES(?,?)", (subj, k))
        elif r[0] is None:
            return  # subject shredded: no PII retained
        else:
            k = r[0]
        cur.execute("INSERT INTO pii VALUES(?,?,?)", (seq, subj, Fernet(k).encrypt(json.dumps(payload).encode())))

    def read_pii(self, seq):
        with self.lock:
            r = self.db.execute("SELECT p.subject, p.blob, k.key FROM pii p LEFT JOIN subject_keys k "
                                "ON k.subject=p.subject WHERE p.seq=?", (seq,)).fetchone()
        if not r or r[2] is None: return None
        try: return json.loads(Fernet(r[2]).decrypt(r[1]))
        except (InvalidToken, ValueError): return None

    def pii_status(self, seq):
        with self.lock:
            r = self.db.execute("SELECT k.key IS NOT NULL FROM pii p LEFT JOIN subject_keys k "
                                "ON k.subject=p.subject WHERE p.seq=?", (seq,)).fetchone()
        if r is None: return "none"
        return "encrypted" if r[0] else "key destroyed"

    def shred(self, subject):
        with self.lock:
            self._require_clean()
            s = self._resolve(subject)
            n = self.db.execute("SELECT COUNT(*) FROM pii WHERE subject=?", (s,)).fetchone()[0]
            self.db.execute("INSERT INTO subject_keys VALUES(?,NULL) ON CONFLICT(subject) DO UPDATE SET key=NULL", (s,))
            self.db.commit()
            return n

    def purge_pii(self, older_than_days):
        if older_than_days < MIN_RETENTION_DAYS:
            raise ValueError(f"Retention floor: Art.19(1) requires >= {MIN_RETENTION_DAYS} days (~6 months)")
        cutoff = time.time() - older_than_days * 86400
        with self.lock:
            self._require_clean()
            c = self.db.execute("DELETE FROM pii WHERE seq IN (SELECT seq FROM events WHERE ts<?)", (cutoff,))
            self.db.commit()
            return c.rowcount

    # ---------- checkpoints / anchoring ----------
    def checkpoint(self):
        with self.lock:
            self._require_clean()
            last = self.db.execute("SELECT COALESCE(MAX(to_seq),0) FROM checkpoints").fetchone()[0]
            rows = self.db.execute("SELECT seq, entry_hash FROM events WHERE seq>? ORDER BY seq", (last,)).fetchall()
            if not rows: return None
            frm, to, hh = rows[0][0], rows[-1][0], rows[-1][1]
            root = merkle_root([r[1] for r in rows])
            sig = self.sk.sign(canon({"from": frm, "to": to, "root": root, "head": hh})).hex()
            cur = self.db.execute("INSERT INTO checkpoints(from_seq,to_seq,root,head_hash,ts,sig) VALUES(?,?,?,?,?,?)",
                                  (frm, to, root, hh, time.time(), sig))
            self.db.commit()
            self._anchor(cur.lastrowid, to, hh, root)
            return {"id": cur.lastrowid, "from": frm, "to": to, "root": root}

    def _read_anchors(self):
        if not os.path.exists(self.anchor_path): return [], 0
        out, bad = [], 0
        need = {"cp", "to_seq", "head_hash", "root", "ts", "prev", "hash"}
        with open(self.anchor_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    a = json.loads(line)
                    if not isinstance(a, dict) or not need <= a.keys(): raise ValueError
                    out.append(a)
                except ValueError:
                    bad += 1
        return out, bad

    def _anchors(self): return self._read_anchors()[0]

    def _anchor(self, cp, to, hh, root):
        a = self._anchors()
        rec = {"cp": cp, "to_seq": to, "head_hash": hh, "root": root, "ts": time.time(),
               "prev": a[-1]["hash"] if a else GENESIS}
        rec["hash"] = h(canon(rec))
        with open(self.anchor_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

    def checkpoints(self):
        with self.lock:
            anchors = {a["cp"]: a for a in self._anchors()}
            rows = self.db.execute("SELECT id,from_seq,to_seq,root,head_hash,ts FROM checkpoints ORDER BY id").fetchall()
        return [{"id": r[0], "from_seq": r[1], "to_seq": r[2], "root": r[3], "head_hash": r[4], "ts": r[5],
                 "anchored": r[0] in anchors, "anchor_hash": anchors.get(r[0], {}).get("hash")} for r in rows]

    def proof(self, seq):
        """Merkle inclusion proof of event `seq` against the signed checkpoint root."""
        with self.lock:
            r = self.db.execute("SELECT id,from_seq,to_seq,root FROM checkpoints WHERE from_seq<=? AND to_seq>=? "
                                "ORDER BY id LIMIT 1", (seq, seq)).fetchone()
            if not r: return None
            cid, f, t, root = r
            hs = [x[0] for x in self.db.execute("SELECT entry_hash FROM events WHERE seq BETWEEN ? AND ? ORDER BY seq", (f, t))]
        if len(hs) != t - f + 1: raise ValueError("events missing inside checkpoint range")
        idx = seq - f; path = merkle_path(hs, idx)
        return {"seq": seq, "checkpoint": cid, "leaf": hs[idx], "path": path, "root": root,
                "valid": verify_path(hs[idx], path, root)}

    # ---------- verification ----------
    def verify(self):
        with self.lock:
            t0 = time.perf_counter()
            issues = []
            def bad(seq, typ, det): issues.append({"seq": seq, "type": typ, "detail": det})
            rows = self.db.execute("SELECT * FROM events ORDER BY seq").fetchall()
            by_seq = {}
            exp_prev, exp_seq = GENESIS, 1
            for r in rows:
                d = dict(zip(COLS, r))
                by_seq[d["seq"]] = d
                if d["seq"] != exp_seq:
                    bad(d["seq"], "sequence_gap", f"expected seq {exp_seq}, found {d['seq']}")
                exp_seq = d["seq"] + 1
                if d["prev_hash"] != exp_prev:
                    bad(d["seq"], "chain_break", "prev_hash does not match previous entry")
                try:
                    rec = {k: d[k] for k in RECK}
                    rec["meta"] = json.loads(d["meta"])
                    ok = h(canon(rec) + d["prev_hash"].encode()) == d["entry_hash"]
                except Exception:
                    ok = False
                if not ok: bad(d["seq"], "hash_mismatch", "entry content does not match its hash")
                try:
                    self.pk.verify(bytes.fromhex(d["sig"]), bytes.fromhex(d["entry_hash"]))
                except Exception:
                    bad(d["seq"], "bad_signature", "signature invalid (attacker lacks signing key)")
                exp_prev = d["entry_hash"]
            max_seq = max(by_seq) if by_seq else 0

            cps = self.db.execute("SELECT id,from_seq,to_seq,root,head_hash,sig FROM checkpoints ORDER BY id").fetchall()
            anchors, bad_lines = self._read_anchors()
            if bad_lines: bad(None, "anchor_log_corrupt", f"{bad_lines} unreadable anchor line(s)")
            a_by_cp = {}
            for a in anchors: a_by_cp.setdefault(a["cp"], []).append(a)
            prev_to = 0
            for cid, f, t, root, hh, sig in cps:
                try:
                    self.pk.verify(bytes.fromhex(sig), canon({"from": f, "to": t, "root": root, "head": hh}))
                except Exception:
                    bad(None, "checkpoint_bad_signature", f"checkpoint {cid}")
                if f != prev_to + 1:
                    bad(None, "checkpoint_gap", f"checkpoint {cid} starts at seq {f}, expected {prev_to + 1}")
                prev_to = max(prev_to, t)
                hs = [by_seq[s]["entry_hash"] for s in range(f, t + 1) if s in by_seq]
                if len(hs) != t - f + 1 or merkle_root(hs) != root:
                    bad(None, "checkpoint_root_mismatch", f"checkpoint {cid} covers seq {f}-{t}")
                elif by_seq[t]["entry_hash"] != hh:
                    bad(t, "checkpoint_head_mismatch", f"checkpoint {cid} head differs from entry {t}")
                al = a_by_cp.get(cid, [])
                if not al: bad(None, "anchor_missing", f"no anchor for checkpoint {cid}")
                else:
                    if len(al) > 1: bad(None, "anchor_duplicate", f"{len(al)} anchors for checkpoint {cid}")
                    if (al[0]["to_seq"], al[0]["head_hash"], al[0]["root"]) != (t, hh, root):
                        bad(None, "anchor_checkpoint_mismatch", f"anchor differs from checkpoint {cid}")
            known = {c[0] for c in cps}
            prev = GENESIS
            for a in anchors:
                if a["cp"] not in known: bad(a["to_seq"], "anchor_orphan", f"anchor for unknown checkpoint {a['cp']}")
                body = {k: v for k, v in a.items() if k != "hash"}
                if a["prev"] != prev or h(canon(body)) != a["hash"]:
                    bad(None, "anchor_log_corrupt", f"anchor for checkpoint {a['cp']}")
                prev = a["hash"]
                if not isinstance(a["to_seq"], int):
                    bad(None, "anchor_log_corrupt", f"anchor for checkpoint {a['cp']}: to_seq not an integer"); continue
                if max_seq < a["to_seq"]:
                    bad(a["to_seq"], "tail_truncated", f"anchor records seq {a['to_seq']}, log ends earlier")
                elif by_seq.get(a["to_seq"], {}).get("entry_hash") != a["head_hash"]:
                    bad(a["to_seq"], "anchor_mismatch", "entry at anchored seq differs from witnessed head")
            last_anchored = max((a["to_seq"] for a in anchors if isinstance(a["to_seq"], int)), default=0)
            return {"ok": not issues, "checked": len(rows), "issues": issues,
                    "checkpoints": len(cps), "anchors": len(anchors),
                    "unanchored_events": max(0, max_seq - last_anchored), "demo_tampered": self.tampered,
                    "verify_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # ---------- attack simulation (demo / experiments only) ----------
    def backup(self):
        dst = sqlite3.connect(self.bak_path); self.db.backup(dst); dst.close()

    def restore(self):
        with self.lock:
            if not os.path.exists(self.bak_path): raise ValueError("no backup available")
            src = sqlite3.connect(self.bak_path); src.backup(self.db); src.close()
            self.db.execute("PRAGMA secure_delete=ON")
            self.db.executescript(SCHEMA)
            _wipe(self.bak_path)  # the backup holds subject keys: do not leave it behind
            try: os.remove(self.flag_path)
            except FileNotFoundError: pass
            self.tampered = False
            self._load_head()

    def tamper(self, kind, seq=1, n=1, rehash=False):
        if kind not in TAMPER_KINDS: raise ValueError("unknown tamper kind")
        if not isinstance(seq, int) or seq < 1: raise ValueError("seq must be a positive integer")
        if not isinstance(n, int) or not 1 <= n <= 1000: raise ValueError("n must be 1..1000")
        with self.lock:
            if kind == "replay" and self.db.execute("SELECT 1 FROM events WHERE seq=?", (seq,)).fetchone() is None:
                raise ValueError("no such seq")
            if not self.tampered:
                self.backup(); open(self.flag_path, "w").close(); self.tampered = True
            c = self.db
            last = lambda: c.execute("SELECT seq, entry_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone() or (0, GENESIS)
            if kind == "modify":
                c.execute("UPDATE events SET output_hash=? WHERE seq=?", ("f" * 64, seq))
                if rehash: self._rehash_from(seq)
            elif kind == "delete":
                c.execute("DELETE FROM events WHERE seq=?", (seq,))
            elif kind == "truncate":
                m = c.execute("SELECT MAX(seq) FROM events").fetchone()[0] or 0
                c.execute("DELETE FROM events WHERE seq>?", (m - n,))
            elif kind == "backdate":
                c.execute("UPDATE events SET ts=ts-? WHERE seq=?", (30 * 86400, seq))
            elif kind == "reorder":
                c.execute("UPDATE events SET seq=-1 WHERE seq=?", (seq,))
                c.execute("UPDATE events SET seq=? WHERE seq=?", (seq, seq + 1))
                c.execute("UPDATE events SET seq=? WHERE seq=-1", (seq + 1,))
            elif kind == "insert":  # forged entry with a correct hash link but no valid signature
                ls, lh = last()
                rec = {"seq": ls + 1, "ts": time.time(), "event_type": "inference", "system_id": "forged",
                       "model_version": "forged", "subject": "forged", "input_hash": "0" * 64,
                       "output_hash": "0" * 64, "meta": {}}
                eh = h(canon(rec) + lh.encode())
                c.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                          (rec["seq"], rec["ts"], rec["event_type"], rec["system_id"], rec["model_version"],
                           rec["subject"], rec["input_hash"], rec["output_hash"], "{}", lh, eh, "00" * 64))
            elif kind == "replay":  # copy an existing signed entry to the end of the log
                row = list(c.execute("SELECT * FROM events WHERE seq=?", (seq,)).fetchone())
                ls, lh = last()
                row[0], row[9] = ls + 1, lh
                c.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", row)
            c.commit()
            self._load_head()

    def _rehash_from(self, seq):
        """Attacker recomputes hashes but cannot re-sign."""
        rows = self.db.execute("SELECT * FROM events WHERE seq>=? ORDER BY seq", (seq,)).fetchall()
        p = self.db.execute("SELECT entry_hash FROM events WHERE seq=?", (seq - 1,)).fetchone()
        prev = p[0] if p else GENESIS
        for r in rows:
            d = dict(zip(COLS, r))
            rec = {k: d[k] for k in RECK}
            rec["meta"] = json.loads(d["meta"])
            eh = h(canon(rec) + prev.encode())
            self.db.execute("UPDATE events SET prev_hash=?, entry_hash=? WHERE seq=?", (prev, eh, d["seq"]))
            prev = eh

    # ---------- reads ----------
    def list(self, limit=50):
        limit = max(1, min(int(limit), 1000))
        with self.lock:
            rows = self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(zip(COLS, r))
                try: d["meta"] = json.loads(d["meta"])
                except (TypeError, ValueError): d["meta"] = {"_corrupt": True}
                d["art12"] = EVENT_TYPES.get(d["event_type"], "?")
                d["pii_status"] = self.pii_status(d["seq"])
                out.append(d)
            return out

    def stats(self):
        with self.lock:
            n = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            c = self.db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            return {"events": n, "checkpoints": c, "head_hash": self.head[1], "pubkey": self.pubkey_hex,
                    "tampered": self.tampered, "anchors": len(self._anchors())}

def _gap_event(n): return {"event_type": "logger_gap", "subject": "system", "meta": {"dropped": n}}

class AsyncWriter:
    """Bounded buffer + background batch writer. policy: fail_open (drop on overflow, record a gap) | fail_closed (block)."""
    def __init__(self, store, batch=128, policy="fail_open", maxq=5000):
        if policy not in ("fail_open", "fail_closed"): raise ValueError("policy must be fail_open|fail_closed")
        self.store, self.batch, self.policy = store, batch, policy
        self.q = queue.Queue(maxq); self._lk = threading.Lock()
        self.dropped = 0; self.pending_gap = 0; self.failed = 0; self.errors = 0; self.rejected = []
        self._closed = False; self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def submit(self, ev):
        if self._closed: raise RuntimeError("writer is closed")
        validate_event(ev)
        if self.policy == "fail_open":
            try: self.q.put_nowait(ev)
            except queue.Full:
                with self._lk: self.dropped += 1; self.pending_gap += 1
                return False
        else:
            while True:
                if self._closed: raise RuntimeError("writer is closed")
                try: self.q.put(ev, timeout=0.1); break
                except queue.Full: continue
        return True

    def _run(self):
        while not (self._stop.is_set() and self.q.empty()):
            try: b = [self.q.get(timeout=0.05)]
            except queue.Empty: continue
            while len(b) < self.batch:
                try: b.append(self.q.get_nowait())
                except queue.Empty: break
            try: self._write(list(b))
            finally:
                for _ in b: self.q.task_done()

    def _write(self, items):
        with self._lk: gap, self.pending_gap = self.pending_gap, 0
        batch = ([_gap_event(gap)] if gap else []) + items
        try: self.store.append_batch(batch); return
        except Exception: self.errors += 1
        for ev in batch:  # batch is atomic, so retrying one by one cannot duplicate
            try: self.store.append_batch([ev])
            except Exception as ex:
                with self._lk:
                    self.failed += 1; self.rejected = (self.rejected + [str(ex)[:200]])[-100:]
                    if ev.get("event_type") == "logger_gap": self.pending_gap += ev["meta"]["dropped"]

    def _flush_gap(self):
        with self._lk: gap, self.pending_gap = self.pending_gap, 0
        if gap:
            try: self.store.append_batch([_gap_event(gap)])
            except Exception:
                with self._lk: self.failed += 1

    def stats(self):
        return {"dropped": self.dropped, "failed": self.failed, "errors": self.errors, "queued": self.q.qsize(),
                "policy": self.policy, "closed": self._closed}

    def flush(self): self.q.join()

    def close(self):
        self._closed = True
        self.q.join(); self._flush_gap()
        self._stop.set(); self.t.join(timeout=5)