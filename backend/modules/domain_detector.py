"""
domain_detector.py — Automatic dataset domain inference engine.

Analyzes column names, sample values, and statistical patterns to infer
the most likely domain of a dataset. Returns confidence scores for each
candidate domain and fires a mismatch warning when the user-selected
domain contradicts the detected one.

Supported domains:
  Healthcare, Finance, Social Media, NLP/Text, Computer Vision,
  Music/Audio, E-Commerce, General

Precision: all confidence scores reported to 6 decimal places.
"""

from __future__ import annotations
import re
from typing import Optional


# ── Domain keyword signatures ─────────────────────────────────────────────────
# Each entry: (column-name keywords, value keywords, weight)
DOMAIN_SIGNATURES: dict[str, list[tuple[set[str], set[str], float]]] = {

    "Healthcare": [
        ({"patient","diagnosis","symptom","disease","condition","treatment",
          "medication","drug","dose","dosage","prescription","icd","cpt",
          "blood","glucose","cholesterol","bmi","weight","height","age",
          "gender","sex","vital","pulse","temperature","oxygen","systolic",
          "diastolic","hemoglobin","creatinine","lab","test","result",
          "hospital","clinic","ward","surgery","procedure"},
         {"cancer","diabetes","hypertension","asthma","stroke","healthy",
          "positive","negative","mg","mmhg","bpm","cm","kg","male","female"},
         1.4),
    ],

    "Finance": [
        ({"transaction","amount","balance","credit","debit","account","payment",
          "merchant","fraud","currency","bank","loan","interest","rate","invest",
          "portfolio","stock","equity","bond","revenue","profit","loss","tax",
          "invoice","fee","charge","transfer","deposit","withdrawal","card",
          "iban","swift","ticker","price","close","open","volume","market"},
         {"usd","eur","gbp","jpy","inr","ngn","fraud","legitimate","buy","sell",
          "debit","credit","approved","declined","visa","mastercard","amex"},
         1.4),
    ],

    "Social Media": [
        ({"tweet","post","like","share","retweet","follower","following",
          "hashtag","mention","comment","reply","engagement","impression",
          "reach","sentiment","user","username","handle","bio","profile",
          "platform","content","viral","trending","influence","subscriber"},
         {"positive","negative","neutral","twitter","instagram","facebook",
          "tiktok","reddit","linkedin","youtube","love","hate","awesome"},
         1.3),
    ],

    "Computer Vision": [
        ({"image","img","pixel","width","height","channel","class","label",
          "bbox","annotation","bounding","split","train","val","test","augment",
          "filename","path","format","jpeg","png","resolution","brightness",
          "contrast","blur","saturation","hue","mask","segment","detection"},
         {"jpg","jpeg","png","bmp","webp","svg","train","val","test",
          "cat","dog","car","person","bird","object","background"},
         1.3),
    ],

    "Music/Audio": [
        ({"song","track","artist","album","genre","tempo","bpm","key","chord",
          "note","frequency","pitch","melody","rhythm","beat","measure","bar",
          "time_signature","lyric","vocal","instrument","midi","audio","waveform",
          "sample","danceability","energy","valence","acousticness","loudness",
          "mode","duration","liveness","speechiness","instrumentalness"},
         {"major","minor","jazz","rock","pop","classical","blues","electronic",
          "hip-hop","reggae","folk","country","metal","bpm","hz","db"},
         1.3),
    ],

    "E-Commerce": [
        ({"product","order","cart","customer","purchase","price","quantity",
          "category","sku","inventory","stock","shipping","delivery","review",
          "rating","seller","buyer","discount","coupon","return","refund",
          "checkout","wishlist","recommendation","sale","item","description"},
         {"in_stock","out_of_stock","pending","shipped","delivered","cancelled",
          "5_star","4_star","electronics","clothing","books","furniture"},
         1.2),
    ],

    "NLP/Text": [
        ({"text","sentence","token","word","corpus","document","abstract",
          "content","review","paragraph","sequence","input","output","label",
          "class","language","translation","summary","source","target","vocab",
          "embedding","tfidf","ngram","pos","ner","entity","relation"},
         {"positive","negative","neutral","english","french","spanish","german",
          "entailment","contradiction","neutral","true","false"},
         1.1),
    ],

    "General": [
        (set(), set(), 0.1),   # fallback — always contributes tiny baseline
    ],
}

# Canonical domain aliases (user-facing → internal key)
DOMAIN_ALIASES: dict[str, str] = {
    "Healthcare":       "Healthcare",
    "Finance":          "Finance",
    "Social Media":     "Social Media",
    "General":          "General",
    "Computer Vision":  "Computer Vision",
    "Music/Audio":      "Music/Audio",
    "E-Commerce":       "E-Commerce",
    "NLP/Text":         "NLP/Text",
}

# Domains that map to each other for mismatch tolerance
RELATED_DOMAINS: dict[str, set[str]] = {
    "Healthcare":      {"Healthcare"},
    "Finance":         {"Finance", "E-Commerce"},
    "Social Media":    {"Social Media", "NLP/Text"},
    "E-Commerce":      {"E-Commerce", "Finance"},
    "NLP/Text":        {"NLP/Text", "Social Media"},
    "Computer Vision": {"Computer Vision"},
    "Music/Audio":     {"Music/Audio"},
    "General":         set(DOMAIN_SIGNATURES.keys()),   # General is compatible with everything
}


def detect_domain(profile: dict, summary: dict) -> dict:
    """
    Infer dataset domain from column names and sample values.

    Returns
    -------
    {
        "detected_domain":   str,
        "confidence":        float (0.0–1.0, 6 dp),
        "scores":            {domain: float},          # raw scores, 6 dp each
        "ranked":            [(domain, confidence)],   # sorted descending
        "evidence":          [str],                    # human-readable signals
        "column_hits":       {domain: [col]},          # which columns matched
    }
    """
    columns: list[str] = [c.lower() for c in (
        profile.get("numeric_columns", []) +
        profile.get("categorical_columns", [])
    )]
    # Also pull column names from summary if profile is sparse
    if not columns:
        columns = [c.lower() for c in summary.get("columns", [])]

    sample_values: list[str] = []
    for row in summary.get("sample_rows", []):
        for v in row.values():
            if isinstance(v, str):
                sample_values.append(v.lower())

    # Balance info from class distribution
    balance_labels: list[str] = []
    for col_data in profile.get("class_balance", {}).values():
        balance_labels.extend(str(l).lower() for l in col_data.get("labels", []))

    all_text = set(columns) | set(balance_labels) | set(sample_values[:200])

    raw_scores: dict[str, float] = {}
    column_hits: dict[str, list[str]] = {}
    evidence: list[str] = []

    for domain, sigs in DOMAIN_SIGNATURES.items():
        domain_score = 0.0
        hits: list[str] = []
        for col_kws, val_kws, weight in sigs:
            # Column name matches
            for col in columns:
                col_tokens = set(re.split(r"[_\-\s\.]+", col))
                matched = col_tokens & col_kws
                if matched:
                    domain_score += weight * len(matched) * 2.0
                    hits.append(col)
                elif any(kw in col for kw in col_kws):
                    domain_score += weight * 1.0
                    hits.append(col)

            # Value matches
            for token in all_text:
                if token in val_kws:
                    domain_score += weight * 0.5

        raw_scores[domain] = round(domain_score, 6)
        column_hits[domain] = list(set(hits))

    # Normalise to [0, 1] sum
    total = sum(raw_scores.values()) or 1.0
    normalised = {d: round(s / total, 6) for d, s in raw_scores.items()}

    ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
    detected = ranked[0][0]
    confidence = ranked[0][1]

    # Build evidence strings
    top_hits = column_hits.get(detected, [])[:5]
    if top_hits:
        evidence.append(f"Column name signals for '{detected}': {top_hits}")
    if confidence < 0.35:
        evidence.append("Low confidence — dataset may be general-purpose or unlabelled")
    if ranked[1][1] > 0.25:
        evidence.append(
            f"Secondary domain '{ranked[1][0]}' also likely "
            f"(confidence {ranked[1][1]:.4f})"
        )

    return {
        "detected_domain": detected,
        "confidence":      confidence,
        "scores":          normalised,
        "ranked":          ranked,
        "evidence":        evidence,
        "column_hits":     column_hits,
    }


def check_domain_mismatch(
    detected: dict,
    user_selected_domain: str,
) -> dict:
    """
    Compare auto-detected domain against user's selection.

    Returns a warning dict if mismatch is found, or a clear dict if aligned.
    """
    detected_domain = detected.get("detected_domain", "General")
    confidence = detected.get("confidence", 0.0)
    related = RELATED_DOMAINS.get(detected_domain, {detected_domain})

    # No mismatch if confidence is very low (can't be sure of domain)
    if confidence < 0.20:
        return {
            "mismatch": False,
            "severity": "NONE",
            "message": (
                f"Domain detection confidence is low ({confidence:.4f}). "
                "Could not reliably determine dataset domain — proceeding with user selection."
            ),
            "detected_domain":       detected_domain,
            "user_selected_domain":  user_selected_domain,
            "confidence":            round(confidence, 6),
        }

    if user_selected_domain in related or user_selected_domain == detected_domain:
        return {
            "mismatch": False,
            "severity": "NONE",
            "message": (
                f"Selected domain '{user_selected_domain}' is consistent with "
                f"detected domain '{detected_domain}' (confidence {confidence:.4f})."
            ),
            "detected_domain":      detected_domain,
            "user_selected_domain": user_selected_domain,
            "confidence":           round(confidence, 6),
        }

    # Mismatch — determine severity
    score_gap = confidence - detected.get("scores", {}).get(user_selected_domain, 0.0)
    severity = "HIGH" if score_gap > 0.40 else "MEDIUM" if score_gap > 0.20 else "LOW"

    return {
        "mismatch": True,
        "severity": severity,
        "message": (
            f"DOMAIN MISMATCH: Dataset appears to be '{detected_domain}' "
            f"(confidence {confidence:.4f}) but you selected '{user_selected_domain}'. "
            f"Quality thresholds and scoring rules for '{user_selected_domain}' may not "
            f"apply correctly to this data. "
            f"Recommended: switch domain to '{detected_domain}' for accurate results."
        ),
        "detected_domain":          detected_domain,
        "user_selected_domain":     user_selected_domain,
        "confidence":               round(confidence, 6),
        "score_gap":                round(score_gap, 6),
        "recommendation":           f"Use domain='{detected_domain}' for this dataset",
        "detected_domain_scores":   detected.get("scores", {}),
    }
