"""Tamper-evident, privacy-aware audit log: case-study implementation for EU AI Act Art. 12 / 19(1).
Hash chain + Ed25519 signatures + Merkle checkpoints + external-anchor file + per-subject
encryption (crypto-shredding) + retention floor + async writer."""
import json, hashlib, hmac, sqlite3, time, os, threading, queue
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization as ser
from cryptography.fernet import Fernet

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
}
BIOMETRIC_FIELDS = ("session_start", "session_end", "reference_db", "matched_input_ref", "verifier_ids")
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
"""

def h(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canon(o) -> bytes: return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()

def merkle_root(hashes):
    if not hashes: return GENESIS
    lvl = [bytes.fromhex(x) for x in hashes]
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        lvl = [hashlib.sha256(b"\x01" + lvl[i] + lvl[i + 1]).digest() for i in range(0, len(lvl), 2)]
    return lvl[0].hex()

class LogStore:
    def __init__(self, d):
        os.makedirs(d, exist_ok=True)
        self.d = d
        self.lock = threading.RLock()
        self.db = sqlite3.connect(os.path.join(d, "log.db"), check_same_thread=False)
        self.db.executescript(SCHEMA)
        kp = os.path.join(d, "signing.key")
        if not os.path.exists(kp): open(kp, "wb").write(os.urandom(32))
        self.sk = Ed25519PrivateKey.from_private_bytes(open(kp, "rb").read())
        self.pk = self.sk.public_key()
        sp = os.path.join(d, "salt.bin")
        if not os.path.exists(sp): open(sp, "wb").write(os.urandom(32))
        self.salt = open(sp, "rb").read()
        self.anchor_path = os.path.join(d, "anchors.jsonl")  # simulated external witness
        self.tampered = False
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

    # ---------- append ----------
    def append_batch(self, evs):
        with self.lock:
            seq, prev = self.head
            out = []
            try:
                cur = self.db.cursor()
                for e in evs:
                    t = e.get("event_type")
                    if t not in EVENT_TYPES: raise ValueError(f"unknown event_type {t!r}")
                    meta = e.get("meta") or {}
                    if t == "biometric_match":
                        miss = [f for f in BIOMETRIC_FIELDS if f not in meta]
                        if miss: raise ValueError(f"Art.12(3) fields missing: {miss}")
                    seq += 1
                    subj = e.get("subject") or "anon"
                    rec = {"seq": seq, "ts": e.get("ts") or time.time(), "event_type": t,
                           "system_id": e.get("system_id", "sys-1"), "model_version": e.get("model_version", "m-1.0"),
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
        r = self.db.execute("SELECT p.subject, p.blob, k.key FROM pii p LEFT JOIN subject_keys k "
                            "ON k.subject=p.subject WHERE p.seq=?", (seq,)).fetchone()
        if not r or r[2] is None: return None
        return json.loads(Fernet(r[2]).decrypt(r[1]))

    def shred(self, subject):
        with self.lock:
            n = self.db.execute("SELECT COUNT(*) FROM pii WHERE subject=?", (subject,)).fetchone()[0]
            self.db.execute("INSERT INTO subject_keys VALUES(?,NULL) ON CONFLICT(subject) DO UPDATE SET key=NULL", (subject,))
            self.db.commit()
            return n

    def purge_pii(self, older_than_days):
        if older_than_days < MIN_RETENTION_DAYS:
            raise ValueError(f"Retention floor: Art.19(1) requires >= {MIN_RETENTION_DAYS} days (~6 months)")
        cutoff = time.time() - older_than_days * 86400
        with self.lock:
            c = self.db.execute("DELETE FROM pii WHERE seq IN (SELECT seq FROM events WHERE ts<?)", (cutoff,))
            self.db.commit()
            return c.rowcount

    # ---------- checkpoints / anchoring ----------
    def checkpoint(self):
        with self.lock:
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

    def _anchors(self):
        if not os.path.exists(self.anchor_path): return []
        return [json.loads(l) for l in open(self.anchor_path) if l.strip()]

    def _anchor(self, cp, to, hh, root):
        a = self._anchors()
        rec = {"cp": cp, "to_seq": to, "head_hash": hh, "root": root, "ts": time.time(),
               "prev": a[-1]["hash"] if a else GENESIS}
        rec["hash"] = h(canon(rec))
        with open(self.anchor_path, "a") as f: f.write(json.dumps(rec) + "\n")

    # ---------- verification ----------
    def verify(self):
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
            rec = {k: d[k] for k in RECK}
            rec["meta"] = json.loads(d["meta"])
            if h(canon(rec) + d["prev_hash"].encode()) != d["entry_hash"]:
                bad(d["seq"], "hash_mismatch", "entry content does not match its hash")
            try:
                self.pk.verify(bytes.fromhex(d["sig"]), bytes.fromhex(d["entry_hash"]))
            except Exception:
                bad(d["seq"], "bad_signature", "signature invalid (attacker lacks signing key)")
            exp_prev = d["entry_hash"]
        for cp in self.db.execute("SELECT id,from_seq,to_seq,root,head_hash,sig FROM checkpoints").fetchall():
            cid, f, t, root, hh, sig = cp
            try:
                self.pk.verify(bytes.fromhex(sig), canon({"from": f, "to": t, "root": root, "head": hh}))
            except Exception:
                bad(None, "checkpoint_bad_signature", f"checkpoint {cid}")
            hs = [by_seq[s]["entry_hash"] for s in range(f, t + 1) if s in by_seq]
            if len(hs) != t - f + 1 or merkle_root(hs) != root:
                bad(None, "checkpoint_root_mismatch", f"checkpoint {cid} covers seq {f}-{t}")
        prev = GENESIS
        for a in self._anchors():
            body = {k: v for k, v in a.items() if k != "hash"}
            if a["prev"] != prev or h(canon(body)) != a["hash"]:
                bad(None, "anchor_log_corrupt", f"anchor for checkpoint {a['cp']}")
            prev = a["hash"]
            if (max(by_seq) if by_seq else 0) < a["to_seq"]:
                bad(a["to_seq"], "tail_truncated", f"anchor records seq {a['to_seq']}, log ends earlier")
            elif by_seq.get(a["to_seq"], {}).get("entry_hash") != a["head_hash"]:
                bad(a["to_seq"], "anchor_mismatch", "entry at anchored seq differs from witnessed head")
        return {"ok": not issues, "checked": len(rows), "issues": issues,
                "verify_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # ---------- attack simulation (demo / experiments only) ----------
    def backup(self):
        dst = sqlite3.connect(os.path.join(self.d, "log.bak")); self.db.backup(dst); dst.close()

    def restore(self):
        with self.lock:
            src = sqlite3.connect(os.path.join(self.d, "log.bak")); src.backup(self.db); src.close()
            self.tampered = False
            self._load_head()

    def tamper(self, kind, seq=1, n=1, rehash=False):
        with self.lock:
            if not self.tampered: self.backup(); self.tampered = True
            c = self.db
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
            else:
                raise ValueError("unknown tamper kind")
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
        rows = self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(zip(COLS, r)); d["meta"] = json.loads(d["meta"]); d["art12"] = EVENT_TYPES.get(d["event_type"], "?")
            out.append(d)
        return out

    def stats(self):
        n = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        c = self.db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        return {"events": n, "checkpoints": c, "head_hash": self.head[1], "pubkey": self.pubkey_hex,
                "tampered": self.tampered, "anchors": len(self._anchors())}

class AsyncWriter:
    """Bounded buffer + background batch writer. policy: fail_open (drop on overflow) | fail_closed (block)."""
    def __init__(self, store, batch=128, policy="fail_open", maxq=5000):
        self.store, self.batch, self.policy = store, batch, policy
        self.q = queue.Queue(maxq); self.dropped = 0; self.errors = 0; self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, ev):
        if self.policy == "fail_open":
            try: self.q.put_nowait(ev)
            except queue.Full: self.dropped += 1; return False
        else:
            self.q.put(ev)
        return True

    def _run(self):
        while not self._stop:
            try: b = [self.q.get(timeout=0.05)]
            except queue.Empty: continue
            while len(b) < self.batch:
                try: b.append(self.q.get_nowait())
                except queue.Empty: break
            try: self.store.append_batch(b)
            except Exception: self.errors += 1
            finally:
                for _ in b: self.q.task_done()

    def flush(self): self.q.join()
    def close(self): self.flush(); self._stop = True
