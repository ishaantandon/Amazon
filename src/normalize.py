"""Name and address normalization.

Everything is rule-based and derived from inspecting the training data; no external
lookups. Country only selects which abbreviation table to apply; unknown countries fall
back to the generic rules.
"""
import re
from multiprocessing import Pool

import polars as pl
from anyascii import anyascii

from config import N_JOBS

# ---------------------------------------------------------------- names

LEGAL = {
    # US / generic
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "plc", "llp", "lp", "pc", "pllc", "pa", "lc", "ltda", "gmbh", "ag", "bv", "nv",
    # India
    "pvt", "private", "pvtltd", "opc", "public",
    # France
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "selarl", "scp", "scm", "sca",
    # honorifics / fillers
    "the", "m/s", "ms", "messrs", "sri", "shri", "shree", "dr", "mr", "mrs", "and", "of", "et",
    "le", "la", "les", "de", "du", "des",
}
# Canonical legal-form classes for the "legal form agrees" feature.
LEGAL_CLASS = {
    "llc": "llc", "lc": "llc", "inc": "inc", "incorporated": "inc", "corp": "inc",
    "corporation": "inc", "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt", "llp": "llp", "lp": "lp", "pc": "pc",
    "pllc": "pc", "pa": "pc", "plc": "plc", "public": "plc", "opc": "opc", "sarl": "sarl",
    "sas": "sas", "sasu": "sas", "sa": "sa", "sci": "sci", "eurl": "eurl", "snc": "snc",
    "selarl": "selarl", "gmbh": "gmbh",
}
OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b", "@": "a", "$": "s"})

RE_URL = re.compile(r"https?://|www\.")
RE_TLD = re.compile(r"\.(?:com|c0m|co\.in|net|org|in|fr|co|biz|info|io|us)\b")
RE_ALIAS = re.compile(r"\b(?:f/k/a|a/k/a|d/b/a|fka|aka|dba|formerly)\b")
RE_DOTS = re.compile(r"[.'`’]")
RE_NONALNUM = re.compile(r"[^a-z0-9]+")
RE_DIGIT = re.compile(r"\d")
RE_ALPHA = re.compile(r"[a-z]")


def to_ascii(s: str) -> str:
    s = s.lower()
    return s if s.isascii() else anyascii(s).lower()


def normalize_name(raw: str) -> tuple[str, str, str]:
    """Returns (clean tokens joined by space, legal-form class list, skeleton)."""
    s = to_ascii(raw)
    s = RE_URL.sub(" ", s)
    s = RE_TLD.sub(" ", s)
    s = RE_ALIAS.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " plus ")
    s = RE_DOTS.sub("", s)
    toks = []
    legal = set()
    for t in RE_NONALNUM.sub(" ", s).split():
        # OCR-style digit substitutions inside words ("6eneral", "w0rldwide").
        if RE_DIGIT.search(t) and RE_ALPHA.search(t):
            t = t.translate(OCR)
        if t in LEGAL:
            if t in LEGAL_CLASS:
                legal.add(LEGAL_CLASS[t])
            continue
        if len(t) >= 5 and _skel_token(t) in LEGAL_SKEL:  # transliterated "piraivet", "limitet"
            legal.add(LEGAL_SKEL[_skel_token(t)])
            continue
        if toks and toks[-1] == t:  # "ely ely pediatric"
            continue
        toks.append(t)
    clean = " ".join(toks)
    return clean, " ".join(sorted(legal)), skeleton(clean)


# Consonant skeleton, tuned for names that went English -> Indic script -> Latin:
# "vijy teknoloji" ~ "vijay technology", "praim kmsltemsi" ~ "prime consultancy",
# "siva phainans" ~ "shiva finance".
PHON_RULES = [(re.compile(p), r) for p, r in [
    (r"x", "ks"), (r"ph", "f"), (r"qu", "k"), (r"q", "k"), (r"ck", "k"), (r"sc?h", "s"),
    (r"c(?=[eiy])", "s"), (r"ch", "k"), (r"c", "k"), (r"([tkgbdjw])h", r"\1"),
]]
SKEL = str.maketrans({"j": "k", "g": "k", "z": "s", "w": "b", "v": "b", "d": "t", "m": "n",
                      "y": "", "h": "", "a": "", "e": "", "i": "", "o": "", "u": ""})
RE_DOUBLE = re.compile(r"(.)\1+")


def _skel_token(t: str) -> str:
    for p, r in PHON_RULES:
        t = p.sub(r, t)
    return RE_DOUBLE.sub(r"\1", t.translate(SKEL))


# Skeletons of legal words, so transliterated forms ("piraivet", "limitet") are recognised.
LEGAL_SKEL = {_skel_token(w): c for w, c in [
    ("private", "pvt"), ("limited", "ltd"), ("corporation", "inc"), ("company", "co"),
    ("incorporated", "inc"), ("limitad", "ltd")]}


def skeleton(clean: str) -> str:
    return " ".join(_skel_token(t) or t[:1] for t in clean.split())


# ---------------------------------------------------------------- addresses

COMMON_ABBR = {
    "st": "street", "str": "street", "rd": "road", "roda": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "blvd": "boulevard",
    "bd": "boulevard", "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square",
    "trl": "trail", "ter": "terrace", "cres": "crescent", "mkt": "market", "ext": "extension",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "no": "", "nos": "",
    "apt": "", "apartment": "", "ste": "", "suite": "", "unit": "", "pmb": "", "fl": "floor",
    "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite", "stn": "station",
    "hno": "", "h": "", "plot": "", "flat": "", "shop": "", "door": "", "block": "blk", "blk": "blk",
    "po": "", "box": "", "ward": "ward",
}
FR_ABBR = {"r": "rue", "st": "saint", "ste": "sainte", "imp": "impasse", "che": "chemin", "ch": "chemin",
           "rte": "route", "all": "allee", "fbg": "faubourg", "pl": "place", "av": "avenue",
           "bd": "boulevard", "sq": "square", "qu": "quai", "crs": "cours"}
DROP = {"null", "na", "n/a", "city", "of", "cdp", "the", "de", "du", "des", "la", "le", "les", "d", "l", "et", "and"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
US_CODES = set(US_STATES.values())
RE_US_STATE = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\b")

RE_ADDR_SPLIT = re.compile(r"[^a-z0-9]+")
RE_NUM = re.compile(r"^\d+[a-z]?$|^[a-z]?\d+$")


def normalize_address(raw: str, country: str) -> tuple[str, str, str]:
    """Returns (street/locality tokens, number tokens, state code) — each space-joined."""
    s = to_ascii(raw).replace("<null>", " ")
    state = ""
    if country == "US":
        # Full state name → code; the last code-like token is the state.
        s = RE_US_STATE.sub(lambda m: US_STATES[m.group(1)], s)
    abbr = FR_ABBR if country == "France" else None
    words, nums = [], []
    for t in RE_ADDR_SPLIT.sub(" ", s).split():
        if any(c.isdigit() for c in t):
            if RE_NUM.match(t):
                t = t.lstrip("0") or "0"
                nums.append(t)
            else:
                nums.extend(x.lstrip("0") or "0" for x in re.findall(r"\d+", t))
            continue
        if country == "US" and t in US_CODES and len(t) == 2:
            state = t
            continue
        if abbr and t in abbr:
            t = abbr[t]
        elif t in COMMON_ABBR:
            t = COMMON_ABBR[t]
        if not t or t in DROP or (len(t) == 1 and t not in "nsew"):
            continue
        words.append(t)
    return " ".join(words), " ".join(dict.fromkeys(nums)), state


# ---------------------------------------------------------------- dataframe level

def _norm_row(args):
    name, addr, country = args
    n, legal, skel = normalize_name(name)
    a, nums, state = normalize_address(addr, country)
    return n, legal, skel, a, nums, state


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    with Pool(N_JOBS) as pool:
        out = pool.map(_norm_row, rows, chunksize=20000)
    cols = list(zip(*out)) if out else [[]] * 6
    names = ["name", "legal", "skel", "addr", "nums", "state"]
    return df.select("entity_id", "country").with_columns(
        [pl.Series(n, c, dtype=pl.String) for n, c in zip(names, cols)]
    ).with_columns(
        pl.col("name").str.replace_all(" ", "").alias("compact"),
        (pl.col("addr") == "").and_(pl.col("nums") == "").alias("addr_empty"),
    )
