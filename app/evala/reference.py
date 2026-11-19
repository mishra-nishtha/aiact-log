"""Evaluation-A reference material not in data.py: the paper's 10 core units, the mapping to the pilot's 14,
and AUTHOR-DRAFTED TG/ID/US labels for the gold requirements (paper §9)."""
from .data import CORPUS, GOLD, UNITS

STATUS = ("author-drafted by applying the paper's §9.4 decision procedure; NOT independently annotated. "
          "Not reference labels until two annotators and the legal reviewer confirm them (paper §12.1).")

PAPER_UNITS = [  # paper Table 8 (provisional segmentation)
 ("U1", "Art. 12(1)", "Automatic event recording throughout system lifetime"),
 ("U2", "Art. 12(2)(a)", "Events relevant to identifying risks or substantial modifications"),
 ("U3", "Art. 12(2)(b)", "Events relevant to post-market monitoring"),
 ("U4", "Art. 12(2)(c)", "Events relevant to monitoring operation"),
 ("U5", "Art. 12(3)(a)", "Period of each use, with start and end date/time"),
 ("U6", "Art. 12(3)(b)", "Reference database used for checking input data"),
 ("U7", "Art. 12(3)(c)", "Input data that produced a match"),
 ("U8", "Art. 12(3)(d)", "Identification of natural persons verifying results"),
 ("U9", "Art. 19(1), first sentence", "Provider retention of automatically generated logs under its control"),
 ("U10", "Art. 19(1), second sentence", "Appropriate retention period of at least six months, subject to legal qualifications"),
]
# pilot unit -> (paper unit or None, relation). Derived by comparing provision spans; not stated in either source.
PILOT_TO_PAPER = {
 "U01": ("U1", "part: the recording action"),
 "U02": ("U1", "part: the lifetime temporal constraint"),
 "U03": (None, "not in paper Table 8: the Art. 12(2) chapeau (traceability appropriate to the intended purpose)"),
 "U04": ("U2", "same span"), "U05": ("U3", "same span"), "U06": ("U4", "same span"),
 "U07": ("U5", "same span"), "U08": ("U6", "same span"), "U09": ("U7", "same span"), "U10": ("U8", "same span"),
 "U11": ("U9", "same span"),
 "U12": ("U10", "part: period appropriate to the intended purpose"),
 "U13": ("U10", "part: at least six months"),
 "U14": ("U10", "part: the qualification 'unless provided otherwise in Union or national law'"),
}

def units_report():
    by = {}
    for p, (pid, _r) in PILOT_TO_PAPER.items():
        if pid: by.setdefault(pid, []).append(p)
    return {
        "paper_units": [{"id": i, "provision": p, "obligation": o} for i, p, o in PAPER_UNITS],
        "pilot_units": [{"id": u["id"], "provision": CORPUS[u["parent"]][0], "text": u["text"]} for u in UNITS],
        "mapping": [{"pilot": p, "paper": v[0], "relation": v[1]} for p, v in PILOT_TO_PAPER.items()],
        "split": {k: v for k, v in by.items() if len(v) > 1},
        "pilot_only": [p for p, v in PILOT_TO_PAPER.items() if v[0] is None],
        "counts": {"paper": len(PAPER_UNITS), "pilot": len(UNITS), "clusters_paper": 4,
                   "clusters_pilot": len({u["parent"] for u in UNITS})},
        "paper_expected_range": "10-16 units in 4-8 provision clusters (paper §5.5); 10 is the core scope, extension provisions (13(3)(f), 16(e), 21, 26(6)) are conditional (§7.3)",
        "reason_documented": False,
        "note": ("Mapping derived by comparing the provision spans of paper Table 8 with data.py. Arithmetic: 10 paper units, +1 (U1 split in two), "
                 "+2 (U10 split in three), +1 (chapeau unit with no paper counterpart) = 14. All 14 stay inside the paper's four core clusters. "
                 "Neither the paper nor the code states why the pilot segments more finely, and separating the exception (U14) from the obligation "
                 "it qualifies may conflict with paper rule R2 (§7.1). Both observations are flagged, not resolved."),
    }

def _c(cls, prov, basis, alt=None):
    return {"class": cls, "provision": prov, "basis": basis, "contested": alt is not None, "alternative": alt}

GOLD_CLASS = {
 "G01": _c("TG", "Art. 12(1)", "Restates 'automatic recording of events'; adds no mechanism, threshold or actor."),
 "G02": _c("TG", "Art. 12(1)", "Restates 'over the lifetime of the system'."),
 "G03": _c("ID", "Art. 12(2)", "Adds a documented event taxonomy (a mechanism) to 'appropriate to the intended purpose'; other ways to scope event coverage exist."),
 "G04": _c("TG", "Art. 12(2)(a)", "Restates events relevant for identifying risk situations and substantial modifications."),
 "G05": _c("ID", "Art. 12(2)(b)", "Adds mandatory fields (system id, model version, timestamp) that 12(2)(b) does not name; serves post-market monitoring (Art. 72)."),
 "G06a": _c("ID", "Art. 12(2)(c)", "Adds a retrieval capability for deployers; 12(2)(c) refers to monitoring operation (Art. 26(5)) without naming one."),
 "G06b": _c("ID", "Art. 12(2)(c)", "Integrity verification is an engineering control; Art. 12 does not mention integrity protection (paper §3.1, §9.2 example)."),
 "G07": _c("TG", "Art. 12(3)(a)", "Restates start and end date and time of each use."),
 "G08": _c("TG", "Art. 12(3)(b)", "Restates the reference database against which input data was checked."),
 "G09": _c("ID", "Art. 12(3)(c)", "12(3)(c) requires the input data; '(or a reference to it)' adds an alternative representation.", "TG"),
 "G10": _c("TG", "Art. 12(3)(d)", "Restates identification of the natural persons verifying the results."),
 "G11": _c("TG", "Art. 19(1), first sentence", "Restates retention of automatically generated logs under the provider's control."),
 "G12": _c("TG", "Art. 19(1), second sentence", "Restates a period appropriate to the intended purpose."),
 "G13": _c("TG", "Art. 19(1), second sentence", "Restates 'at least six months' as a floor on deletion."),
 "G14": _c("ID", "Art. 19(1), second sentence", "19(1) yields to Union/national law, 'in particular' data-protection law; 'including erasure' adds a specific reading (paper §10.7 treats the GDPR interaction as unresolved).", "TG"),
}

def reference_labels():
    out = []
    for uid, reqs in GOLD.items():
        for gid, text, _k in reqs:
            out.append({"gold_id": gid, "unit": uid, "requirement": text, **GOLD_CLASS[gid]})
    return {"status": STATUS, "labels": out}

def validate_reference():
    p, gids = [], {g[0] for v in GOLD.values() for g in v}
    if set(GOLD_CLASS) != gids: p.append("GOLD_CLASS keys differ from gold requirement ids")
    for g, v in GOLD_CLASS.items():
        if v["class"] not in ("TG", "ID", "US"): p.append(f"{g}: bad class")
    if set(PILOT_TO_PAPER) != {u["id"] for u in UNITS}: p.append("PILOT_TO_PAPER keys differ from pilot unit ids")
    pids = {x[0] for x in PAPER_UNITS}
    for k, (pid, _r) in PILOT_TO_PAPER.items():
        if pid is not None and pid not in pids: p.append(f"{k}: unknown paper unit {pid}")
    return p