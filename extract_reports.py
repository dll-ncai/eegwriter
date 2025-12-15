#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_reports.py

Usage:
  python extract_reports.py \
    --reports-root "/path/to/reports_root" \
    --out-dir "reports_extracted_txt" \
    --excluded-csv "reports_excluded.csv"
"""

import os
import re
import sys
import csv
import shutil
import argparse
import tempfile
import subprocess
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List

# ---------- Optional imports ----------
HAVE_DOCX = False
HAVE_TEXTRACT = False
try:
    from docx import Document  # python-docx
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

try:
    import textract  # robust .doc support if installed
    HAVE_TEXTRACT = True
except Exception:
    HAVE_TEXTRACT = False

# ---------- Constants ----------
SECTION_TITLES = [
    # "INDICATIONS",
    # "TECHNIQUE",
    "FACTUAL REPORT",
    "IMPRESSION",
]

ADMIN_KEYS = [
    "name", "rank", "age", "gender", "eeg no", "eeg#", "date", "ref.physician",
    "ref physician", "referred by", "bed", "ward", "unit", "department",
    "hospital", "mrn", "cnic", "service no", "d/o", "s/o", "w/o"
]

MONTH_WORDS = [
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december"
]

FOOTER_HINTS = [
    r"\bFCPS\b", r"\bFRCP\b", r"\bMRCP\b", r"\bMBBS\b", r"\bMD\b",
    r"\bDNB\b", r"\bNeurolog(y|ist|y & Neurophysiology)\b",
    r"\bConsultant\b", r"\bProfessor\b", r"\bAssociate Professor\b",
    r"\bAssistant Professor\b", r"\bCol(\.|onel)?\b", r"\bLt\.?\s*Col\b",
    r"\bBrig(\.|adier)?\b", r"\bUHB\b", r"\bFellow(ship)?\b"
]

# corruption heuristics
NONASCII_RUN = 30
ASCII_RATIO_MIN = 0.45

# ---------- Utilities ----------
def safe_check_output(cmd: List[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return None

def read_docx_text(path: Path) -> str:
    if not HAVE_DOCX:
        raise RuntimeError("python-docx not installed (pip install python-docx).")
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_doc_via_external(path: Path) -> str:
    """Try antiword → catdoc → soffice."""
    if shutil.which("antiword"):
        txt = safe_check_output(["antiword", str(path)])
        if txt:
            return txt
    if shutil.which("catdoc"):
        txt = safe_check_output(["catdoc", "-s", "utf-8", str(path)])
        if txt:
            return txt
    if shutil.which("soffice"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            try:
                subprocess.check_call(
                    ["soffice", "--headless", "--convert-to", "txt:Text",
                     "--outdir", str(td), str(path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                out_txt = td / (path.stem + ".txt")
                if out_txt.exists():
                    return out_txt.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    raise RuntimeError(
        f"No working .doc converter found (antiword/catdoc/soffice) for {path.name}. "
        "Try installing one of them or textract."
    )

def read_doc_text(path: Path) -> str:
    if HAVE_TEXTRACT:
        try:
            return textract.process(str(path)).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return read_doc_via_external(path)

def read_report_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return read_docx_text(path)
    if ext == ".doc":
        return read_doc_text(path)
    raise ValueError(f"Unsupported extension: {ext}")

# ---------- Normalization & Cleaning ----------
CTRL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]+")

def normalize_newlines(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = CTRL_RE.sub(" ", s)
    return s

def collapse_blank_lines(s: str) -> str:
    s = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip() + "\n"

def tidy_spaces(s: str) -> str:
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r":\s*(\S)", r": \1", s)
    return s

def fix_stage_roman(s: str) -> str:
    return re.sub(r"\b(Stage)\s*[lI1]{1,2}\b", r"\1 II", s, flags=re.IGNORECASE)

def strip_admin_header(full_text: str) -> str:
    """Drop everything before the first heading (supports inline headings)."""
    hdr_re = build_heading_regex()
    m = hdr_re.search(full_text)
    if not m:
        return full_text
    return full_text[m.start():]

def admin_line_or_table(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if l.count("|") >= 3:
        return True
    if ":" in l:
        left = l.split(":", 1)[0].strip().lower()
        if any(left.startswith(k) for k in ADMIN_KEYS):
            return True
    if len(l) <= 10 and l.lower() in MONTH_WORDS:
        return True
    return False

def looks_like_upper_badge(line: str) -> bool:
    l = line.strip()
    if not l or " " in l:
        return False
    if 2 <= len(l) <= 5 and l.isupper():
        if l in {"EEG", "EKG", "EMG", "AFIC", "HV"}:
            return False
        return True
    return False

def drop_consultant_footer(full_text: str) -> str:
    lines = full_text.rstrip().split("\n")
    if not lines:
        return full_text
    footer_re = re.compile("|".join(FOOTER_HINTS), re.IGNORECASE)
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    cut = len(lines)
    seen_footer_hint = False
    while i >= 0:
        l = lines[i].strip()
        if not l:
            cut = i
            i -= 1
            continue
        footerish = (
            footer_re.search(l) is not None or
            looks_like_upper_badge(l) or
            (len(l) >= 3 and sum(c.isupper() for c in l if c.isalpha()) >= 0.8 * max(1, sum(c.isalpha() for c in l)))
        )
        if footerish:
            seen_footer_hint = True
            cut = i
            i -= 1
            continue
        if seen_footer_hint:
            break
        break
    if cut < len(lines):
        return "\n".join(lines[:cut]).rstrip() + "\n"
    return full_text

def is_corrupted_text(s: str) -> bool:
    if not s:
        return True
    if re.search(rf"[^\x00-\x7F]{{{NONASCII_RUN},}}", s):
        ascii_chars = sum(1 for ch in s if ord(ch) < 128 and (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("P")))
        total_considered = sum(1 for ch in s if ch.strip() != "")
        ratio = (ascii_chars / total_considered) if total_considered else 0.0
        if ratio < ASCII_RATIO_MIN:
            return True
    if len(CTRL_RE.findall(s)) > 0:
        return True
    return False

# ---------- Headings (supports inline) ----------
def build_heading_regex():
    parts = [re.escape(t) for t in SECTION_TITLES]
    # ^ HEADING : [optional inline content] $
    pattern = r"(?m)^\s*(?P<hdr>(" + "|".join(parts) + r"))\s*:?\s*(?P<inline>[^\n]*)\s*$"
    return re.compile(pattern, re.IGNORECASE)

def extract_sections(full_text: str) -> Dict[str, str]:
    text = normalize_newlines(full_text)
    hdr_re = build_heading_regex()
    matches = list(hdr_re.finditer(text))
    sections = {k: "" for k in SECTION_TITLES}

    if not matches:
        sections["FACTUAL REPORT"] = text.strip()
        return sections

    spans: List[Tuple[str, str, int, int]] = []
    for i, m in enumerate(matches):
        hdr = m.group("hdr").upper()
        inline = (m.group("inline") or "").strip()
        start_block = m.end()  # content after the header line
        end_block = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((hdr, inline, start_block, end_block))

    for hdr, inline, start, end in spans:
        body = text[start:end].strip()
        if inline:
            if body:
                body = (inline + "\n" + body).strip()
            else:
                body = inline
        body = re.sub(
            r"^\s*(?:INDICATIONS|TECHNIQUE|FACTUAL REPORT|IMPRESSION)\s*:\s*",
            "", body, flags=re.IGNORECASE
        )
        for k in SECTION_TITLES:
            if hdr == k:
                sections[k] = body
                break
    return sections

# ---------- Unwrap & final passes ----------
def unwrap_soft_lines(s: str) -> str:
    # De-hyphenate
    s = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", s)
    # Generic unwrap: single newlines become spaces
    s = re.sub(r"(?<![.!?:])\n(?!\n)", " ", s)
    return s

def remove_admin_noise_from_section(s: str) -> str:
    s = re.sub(
        r"^\s*(?:INDICATIONS|TECHNIQUE|FACTUAL REPORT|IMPRESSION)\s*:\s*",
        "", s, flags=re.IGNORECASE
    )
    out_lines = []
    seen_sentence = False
    for line in s.splitlines():
        if admin_line_or_table(line):
            continue
        if looks_like_upper_badge(line):
            if not seen_sentence:
                continue
        if re.search(r"[a-z]", line) or re.search(r"[.!?]\s*$", line):
            seen_sentence = True
        out_lines.append(line)
    cleaned = "\n".join(out_lines)
    cleaned = unwrap_soft_lines(cleaned)
    return cleaned

def final_whitespace_pass(s: str) -> str:
    s = normalize_newlines(s)
    s = collapse_blank_lines(s)
    s = tidy_spaces(s)
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
    return s

# ---------- Redaction ----------
def redact_text(s: str) -> str:
    out = s
    out = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "", out)
    out = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "", out)
    title = r"(?:Dr|Mr|Ms|Mrs|Prof|Professor|Col|Lt\.?\s*Col|Brig)\.?"
    name_word = r"[A-Z][a-z]+"
    initials = r"(?:[A-Z]\.)"
    last_name = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
    pat1 = rf"\b{title}\s+(?:{initials}\s+|{name_word}\s+)+{last_name}\b"
    pat2 = rf"\b{name_word}(?:\s+{name_word}){{1,2}}\b"
    out = re.sub(pat1, "", out)
    out = re.sub(pat2, "", out)
    return out

def redact_sections(secs: Dict[str, str]) -> Dict[str, str]:
    return {k: redact_text(v) for k, v in secs.items()}

# ---------- Writing ----------
def write_txt(base_root: Path, out_root: Path, src_path: Path, sections: Dict[str, str]) -> Path:
    rel = src_path.relative_to(base_root)
    out_path = (out_root / rel).with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    
    for k in SECTION_TITLES:
        body = sections.get(k, "").strip()
        
        # MODIFIED: Aggressively flatten text to a SINGLE LINE
        # 1. Replace all newlines with spaces
        body = body.replace("\n", " ")
        # 2. Collapse multiple spaces into one
        body = re.sub(r"\s+", " ", body).strip()
        
        # Format: "HEADING: Content" (on one line)
        parts.append(f"{k}: {body}")

    # MODIFIED: Join with double newline to create a blank line between sections
    final_text = "\n\n".join(parts).rstrip() + "\n"
    out_path.write_text(final_text, encoding="utf-8")
    return out_path

# ---------- Pipeline per file ----------
def process_text_pipeline(raw: str) -> str:
    s = normalize_newlines(raw)
    s = strip_admin_header(s)
    s = drop_consultant_footer(s)
    s = collapse_blank_lines(s)
    s = tidy_spaces(s)
    s = fix_stage_roman(s)
    return s

def postprocess_sections(secs: Dict[str, str]) -> Dict[str, str]:
    cleaned = {}
    for k, v in secs.items():
        v = remove_admin_noise_from_section(v)
        v = final_whitespace_pass(v)
        cleaned[k] = v
    if cleaned.get("IMPRESSION"):
        imp = cleaned["IMPRESSION"]
        imp = drop_consultant_footer(imp)
        imp = final_whitespace_pass(imp)
        cleaned["IMPRESSION"] = imp
    return cleaned

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract EEG report sections; clean; redact; exclude corrupted; mirror subfolders.")
    ap.add_argument("--reports-root", required=True, help="Root containing 'normal/' and 'abnormal/'")
    ap.add_argument("--out-dir", default="reports_extracted_txt", help="Output root")
    ap.add_argument("--excluded-csv", default="reports_excluded.csv", help="CSV for excluded files")
    ap.add_argument("--no-recursive", action="store_true", default=False, help="Do not recurse")
    args = ap.parse_args()

    base_root = Path(args.reports_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not base_root.exists():
        print(f"ERROR: reports_root does not exist: {base_root}", file=sys.stderr)
        sys.exit(2)

    pattern = "*" if args.no_recursive else "**/*"
    exts = {".docx", ".doc"}

    found = 0
    written = 0
    excluded_rows = []

    for p in base_root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        try:
            top = p.relative_to(base_root).parts[0]
        except Exception:
            continue
        if top.lower() not in {"normal", "abnormal"}:
            continue

        found += 1

        try:
            raw = read_report_text(p)
        except RuntimeError as e:
            excluded_rows.append([str(p), str(e)])
            print(f"✖ Excluded {p}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            excluded_rows.append([str(p), f"read_error: {e}"])
            print(f"✖ Excluded {p}: read_error {e}", file=sys.stderr)
            continue

        raw_norm = normalize_newlines(raw)
        if is_corrupted_text(raw_norm):
            excluded_rows.append([str(p), "corrupted_text"])
            print(f"✖ Excluded {p}: corrupted_text", file=sys.stderr)
            continue

        try:
            prepped = process_text_pipeline(raw_norm)
            secs = extract_sections(prepped)
            secs = postprocess_sections(secs)
            secs_red = redact_sections(secs)
            out_path = write_txt(base_root, out_root, p, secs_red)
            written += 1
            print(f"✓ {p} -> {out_path}")
        except Exception as e:
            excluded_rows.append([str(p), f"process_error: {e}"])
            print(f"✖ Excluded {p}: process_error {e}", file=sys.stderr)
            continue

    if excluded_rows:
        try:
            with open(args.excluded_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file_path", "reason"])
                w.writerows(excluded_rows)
            print(f"\nExcluded: {len(excluded_rows)} file(s). Wrote CSV -> {args.excluded_csv}")
        except Exception as e:
            print(f"⚠ Could not write excluded CSV {args.excluded_csv}: {e}", file=sys.stderr)

    print(f"\nDone. Found: {found} file(s). Wrote: {written} txt file(s) to {out_root}.")

if __name__ == "__main__":
    main()