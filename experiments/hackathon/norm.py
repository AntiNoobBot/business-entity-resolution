"""Normalization v1 (Polars expressions). Imported by 03_norm_eval.py and later by the pipeline."""
import polars as pl

_ACC = [("[àáâãäåāăą]", "a"), ("[çćč]", "c"), ("[èéêëēėęě]", "e"), ("[ìíîïī]", "i"), ("[ñńň]", "n"),
        ("[òóôõöøō]", "o"), ("[ùúûüū]", "u"), ("[ýÿ]", "y"), ("[šś]", "s"), ("[žźż]", "z"),
        ("œ", "oe"), ("æ", "ae"), ("ß", "ss")]

def base(c):
    """lowercase + strip accents (Latin only; other scripts are kept as-is)."""
    c = c.str.to_lowercase()
    for pat, rep in _ACC:
        c = c.str.replace_all(pat, rep)
    return c
 
def plain(c):
    """Base text with punctuation folded to spaces.

    Keep Unicode marks (``\\p{M}``) as well as letters and numbers.  Indic vowel
    signs and viramas are marks, not letters; dropping them corrupts words and
    creates avoidable collisions between distinct business names/locations.
    """
    return (base(c).str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ").str.strip_chars())

# words dropped from the *core* name (legal forms, connectors, junk)
_DROP_RE = r"\b(" + "|".join([
    "private", "pvt", "ltd", "limited", "inc", "incorporated", "llc", "llp", "lp", "corp", "corporation",
    "company", "plc", "pllc", "sarl", "sas", "sasu", "eurl", "ets", "dba"]) + r")\b"

def name_core(c):
    c = base(c)
    c = (c.str.replace_all(r"\bm\s*/\s*s\b", " ")
          .str.replace_all(r"\bl\.?l\.?c\b\.?", "llc").str.replace_all(r"\bl\.?l\.?p\b\.?", "llp")
          .str.replace_all(r"\bp\.?l\.?l\.?c\b\.?", "pllc").str.replace_all(r"\bp\.c\b\.?", "pc")
          .str.replace_all(r"\bwww\.", " ")
          .str.replace_all(r"\.(com|net|org|in|fr|co)\b", " ")
          .str.replace_all("&", " and ")
          .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " "))
    # digit/letter homoglyphs inside words (0->o, 1->l, 5->s, 8->b); two passes for overlaps
    for _ in range(2):
        c = c.str.replace_all(r"([a-z])0([a-z])", "${1}o${2}").str.replace_all(r"([a-z])1([a-z])", "${1}l${2}")
    c = (c.str.replace_all(r"\b5([a-z]{3,})\b", "s${1}").str.replace_all(r"\b8([a-z]{3,})\b", "b${1}")
          .str.replace_all(r"\b0([a-z]{3,})\b", "o${1}").str.replace_all(r"\b1([a-z]{3,})\b", "l${1}"))
    c = c.str.replace_all(_DROP_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    return c.str.split(" ")

def add_name_cols(df, col, p):
    """adds {p}core (ordered), {p}sorted (unique tokens, sorted), {p}compact (no spaces)."""
    t = "_t"
    df = df.with_columns(name_core(pl.col(col)).alias(t))
    return df.with_columns(
        pl.col(t).list.join(" ").alias(f"{p}core"),
        pl.col(t).list.unique().list.sort().list.join(" ").alias(f"{p}sorted"),
        pl.col(t).list.join("").alias(f"{p}compact"),
    ).drop(t)

_STREET = {"str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
           "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pkwy": "parkway",
           "hwy": "highway", "pl": "place", "cir": "circle", "ter": "terrace", "sq": "square",
           "ste": "suite", "apt": "apartment", "flr": "floor", "fl": "floor", "bldg": "building",
           "opp": "opposite", "nr": "near"}
_STREET_US_IN = {**_STREET, "st": "street"}
_STREET_FR = {**_STREET, "st": "saint", "ste": "sainte", "r": "rue", "ch": "chemin", "imp": "impasse",
              "all": "allee", "rte": "route", "bvd": "boulevard", "cs": "cours"}

def norm_addr(c, country):
    """Conservative full-address normalization.

    One-letter compass tokens are deliberately left untouched: in real addresses
    they can be house/building codes or initials (for example ``E-26`` and
    ``C.H.S.``).  Geographic agreement is a later scoring feature, not a rewrite.
    """
    c = base(c)
    c = (c.str.replace_all(r"\bnull\b", " ")
          .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
          .str.replace_all(r"\b0+(\d)", "${1}")                 # leading zeros: 01085 -> 1085
          .str.replace_all(r"\b(\d+)(st|nd|rd|th)\b", "${1}")   # 2nd -> 2
          .str.replace_all(r"\s+", " ").str.strip_chars())
    toks = c.str.split(" ")
    gen = toks.list.eval(pl.element().replace(_STREET_US_IN)).list.join(" ")
    fr = toks.list.eval(pl.element().replace(_STREET_FR)).list.join(" ")
    return pl.when(country == "France").then(fr).otherwise(gen)

def token_key(c):
    """Order-insensitive address key for candidate generation only, never proof."""
    return (c.str.split(" ").list.unique().list.sort().list.join(" "))

def split_last(c):
    """raw address -> (everything before the last comma-part, the last comma-part)."""
    parts = c.str.split(",")
    return parts.list.head(parts.list.len() - 1).list.join(","), parts.list.last()
