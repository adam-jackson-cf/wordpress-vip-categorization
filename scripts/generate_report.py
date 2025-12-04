#!/usr/bin/env python3
"""Generate comprehensive analysis report from content analysis."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchingResult, TaxonomyPage, WordPressContent
from src.services.detection import (
    AUDIENCE_TERMS,
    SPECIES_TERMS,
    detect_audiences,
    detect_species,
)


AUDIENCE_WILDCARDS = {"", "all", "general", "todos", "todas", "poblacion", "publico"}
SPECIES_WILDCARDS = {"", "n/a", "none", "all"}


def _match_terms(token: str, terms: dict[str, tuple[str, ...]]) -> str:
    candidate = token.strip().lower()
    for canonical, aliases in terms.items():
        if candidate == canonical:
            return canonical
        alias_set = {alias.lower() for alias in aliases}
        if candidate in alias_set:
            return canonical
    return candidate


def canonicalize_audience(value: str | None) -> str | None:
    if not value:
        return None
    token = value.strip().lower()
    if token in AUDIENCE_WILDCARDS:
        return None
    return _match_terms(token, AUDIENCE_TERMS)


def canonicalize_species_list(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    normalized: set[str] = set()
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in SPECIES_WILDCARDS:
            continue
        normalized.add(_match_terms(lowered, SPECIES_TERMS))
    return {value for value in normalized if value}


def _normalize_token(value: str | None) -> str:
    """Normalize token for comparison by stripping and lowercasing.

    Args:
        value: String value to normalize.

    Returns:
        Normalized lowercase string, empty string if None.
    """
    if not value:
        return ""
    return value.strip().lower()


def _is_audience_compatible(taxonomy: TaxonomyPage, content: WordPressContent) -> bool:
    """Check if detected audiences in content overlap with required audiences in taxonomy.

    Mirrors logic from src/services/matching.py (_audience_compatible method).

    Args:
        taxonomy: Taxonomy page with audience requirements.
        content: WordPress content with detected audiences.

    Returns:
        True if audiences are compatible (overlap exists or no constraints), False otherwise.
    """
    primary = _normalize_token(taxonomy.primary_audiance)
    secondary = _normalize_token(taxonomy.secondary_audiance)
    detected = {_normalize_token(aud) for aud in content.detected_audiences if aud}

    # No audience constraints
    if not primary and not secondary:
        return True

    # Missing detections
    if not detected:
        return False

    # Primary only
    if primary and not secondary:
        return primary in detected

    # At least one required audience must be present
    valid = {token for token in (primary, secondary) if token}
    return bool(valid & detected)


def _is_species_compatible(taxonomy: TaxonomyPage, content: WordPressContent) -> bool:
    """Check if detected species in content overlap with required species in taxonomy.

    Mirrors logic from src/services/matching.py (_species_compatible method).

    Args:
        taxonomy: Taxonomy page with species requirements.
        content: WordPress content with detected species.

    Returns:
        True if species are compatible (all required present or no constraints), False otherwise.
    """
    # No species constraints
    if not taxonomy.species:
        return True

    # Filter out wildcards
    required = {
        _normalize_token(sp)
        for sp in taxonomy.species
        if sp and sp.lower() not in {"n/a", "none"}
    }

    # No valid requirements after filtering
    if not required:
        return True

    detected = {_normalize_token(sp) for sp in content.detected_species if sp}

    # Must have detections and all required species must be present
    return bool(detected) and required.issubset(detected)


def load_analysis() -> list[dict[str, Any]]:
    """Load content analysis JSON."""
    with open("data/examples/content_analysis.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_taxonomy() -> dict[str, dict[str, str]]:
    """Load taxonomy data keyed by destination URL."""

    taxonomy: dict[str, dict[str, str]] = {}
    with open("data/Spain_New.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("Destination_URL") or "").strip()
            if not url:
                continue
            normalized = url.rstrip("/") or url
            taxonomy[url] = row
            taxonomy[normalized] = row
    return taxonomy


def load_matching_results() -> dict[str, Any]:
    """Load matching results from Supabase with content and taxonomy lookups.

    Returns:
        Dictionary with keys:
            - matchings: List of MatchingResult objects
            - content_lookup: Dict mapping content_id to WordPressContent
            - taxonomy_lookup: Dict mapping taxonomy_id to TaxonomyPage
    """
    settings = Settings()
    db_client = SupabaseClient(settings)

    # Load all matching results
    matchings = db_client.get_all_matchings()

    # Build content lookup
    content_ids = [m.content_id for m in matchings]
    content_lookup = db_client.get_content_by_ids(content_ids)

    # Build taxonomy lookup
    taxonomy_ids = [m.taxonomy_id for m in matchings if m.taxonomy_id is not None]
    taxonomy_pages = db_client.get_taxonomy_by_ids(taxonomy_ids)
    taxonomy_lookup = {tax.id: tax for tax in taxonomy_pages}

    return {
        "matchings": matchings,
        "content_lookup": content_lookup,
        "taxonomy_lookup": taxonomy_lookup,
    }


def compute_compliant_score_distribution(
    matchings: list[MatchingResult],
    content_lookup: dict[Any, WordPressContent],
    taxonomy_lookup: dict[Any, TaxonomyPage],
) -> dict[str, Any]:
    """Compute score distribution for compliant matches only.

    Filters matches where both audience AND species align between content
    and taxonomy, then buckets by cosine similarity score ranges.

    Args:
        matchings: List of matching results from Supabase.
        content_lookup: Dict mapping content_id to WordPressContent.
        taxonomy_lookup: Dict mapping taxonomy_id to TaxonomyPage.

    Returns:
        Dictionary with keys:
            - buckets: Dict mapping score range to count
            - total_compliant: Total number of compliant matches
            - total_evaluated: Total number of matches evaluated
            - empty_detection_count: Matches missing audience or species detections
            - above_threshold: Count of compliant matches with score >= 0.70
    """
    # Define bucket boundaries
    bucket_ranges = [
        ("0.50-0.60", 0.50, 0.60),
        ("0.60-0.70", 0.60, 0.70),
        ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90),
        ("0.90-1.00", 0.90, 1.00),
    ]

    buckets: dict[str, int] = {label: 0 for label, _, _ in bucket_ranges}
    total_compliant = 0
    empty_detection_count = 0
    total_evaluated = 0

    for match in matchings:
        # Skip matches without taxonomy assignment
        if match.taxonomy_id is None:
            continue

        content = content_lookup.get(match.content_id)
        taxonomy = taxonomy_lookup.get(match.taxonomy_id)

        # Skip if we can't resolve both sides
        if content is None or taxonomy is None:
            continue

        total_evaluated += 1

        # Check compliance (both audience AND species compatible)
        audience_ok = _is_audience_compatible(taxonomy, content)
        species_ok = _is_species_compatible(taxonomy, content)

        if not (audience_ok and species_ok):
            continue

        # Track empty detections
        if not content.detected_audiences or not content.detected_species:
            empty_detection_count += 1

        total_compliant += 1
        score = match.semantic_similarity_score

        # Assign to bucket
        for label, min_score, max_score in bucket_ranges:
            if min_score <= score < max_score:
                buckets[label] += 1
                break
        else:
            # Handle edge case: score == 1.00
            if score == 1.00:
                buckets["0.90-1.00"] += 1

    # Count above threshold
    above_threshold = sum(
        buckets[label]
        for label, min_score, _ in bucket_ranges
        if min_score >= 0.70
    )

    return {
        "buckets": buckets,
        "total_compliant": total_compliant,
        "total_evaluated": total_evaluated,
        "empty_detection_count": empty_detection_count,
        "above_threshold": above_threshold,
    }


def analyze_match_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze matching quality patterns."""
    score_ranges = {"0.5-0.55": [], "0.55-0.60": [], "0.60-0.65": [], "0.65-0.75": []}

    for r in results:
        score = float(r["metadata"]["similarity_score"])
        if 0.5 <= score < 0.55:
            score_ranges["0.5-0.55"].append(r)
        elif 0.55 <= score < 0.60:
            score_ranges["0.55-0.60"].append(r)
        elif 0.60 <= score < 0.65:
            score_ranges["0.60-0.65"].append(r)
        elif 0.65 <= score <= 0.75:
            score_ranges["0.65-0.75"].append(r)

    return score_ranges


def compute_alignment_metrics(
    results: list[dict[str, Any]],
    taxonomy_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    metrics = {
        "evaluated": 0,
        "skipped_no_taxonomy": 0,
        "skipped_no_text": 0,
        "primary_only": {"total": 0, "aligned": 0, "empty_detection": 0},
        "dual_audience": {"total": 0, "aligned": 0, "empty_detection": 0},
        "species": {"total": 0, "aligned": 0, "empty_detection": 0},
    }

    for entry in results:
        metadata = entry.get("metadata") or {}
        extracted = entry.get("extracted") or {}
        target_url = (metadata.get("target_url") or "").strip()
        if not target_url:
            metrics["skipped_no_taxonomy"] += 1
            continue
        taxonomy_row = taxonomy_lookup.get(target_url) or taxonomy_lookup.get(target_url.rstrip("/"))
        if taxonomy_row is None:
            metrics["skipped_no_taxonomy"] += 1
            continue
        body_text = (extracted.get("body_text") or "").strip()
        if not body_text:
            metrics["skipped_no_text"] += 1
            continue

        metrics["evaluated"] += 1
        detected_aud = detect_audiences(body_text)
        detected_species = detect_species(body_text)

        primary = canonicalize_audience(taxonomy_row.get("Primary_Audiance"))
        secondary = canonicalize_audience(taxonomy_row.get("Secondary_Audiance"))
        taxonomy_species = canonicalize_species_list(taxonomy_row.get("Species"))

        if primary and not secondary:
            metrics["primary_only"]["total"] += 1
            if not detected_aud:
                metrics["primary_only"]["empty_detection"] += 1
            if primary in detected_aud:
                metrics["primary_only"]["aligned"] += 1
        elif primary or secondary:
            metrics["dual_audience"]["total"] += 1
            if not detected_aud:
                metrics["dual_audience"]["empty_detection"] += 1
            allowed = {value for value in (primary, secondary) if value}
            if detected_aud & allowed:
                metrics["dual_audience"]["aligned"] += 1

        if taxonomy_species:
            metrics["species"]["total"] += 1
            if not detected_species:
                metrics["species"]["empty_detection"] += 1
            if taxonomy_species.issubset(detected_species):
                metrics["species"]["aligned"] += 1

    return metrics


def _format_ratio(aligned: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{aligned}/{total} ({(aligned / total) * 100:.1f}%)"


def format_compliance_score_table(distribution: dict[str, Any]) -> str:
    """Format compliance score distribution as markdown table.

    Args:
        distribution: Result from compute_compliant_score_distribution().

    Returns:
        Formatted markdown table with distribution statistics.
    """
    buckets = distribution["buckets"]
    total_compliant = distribution["total_compliant"]
    total_evaluated = distribution["total_evaluated"]
    above_threshold = distribution["above_threshold"]
    empty_detection_count = distribution["empty_detection_count"]

    lines = ["## Compliant Score Distribution", ""]

    # Summary stats
    compliance_rate = (
        f"{(total_compliant / total_evaluated * 100):.1f}%"
        if total_evaluated > 0
        else "n/a"
    )
    above_threshold_pct = (
        f"{(above_threshold / total_compliant * 100):.1f}%"
        if total_compliant > 0
        else "n/a"
    )
    below_threshold_pct = (
        f"{((total_compliant - above_threshold) / total_compliant * 100):.1f}%"
        if total_compliant > 0
        else "n/a"
    )
    empty_detection_pct = (
        f"{(empty_detection_count / total_compliant * 100):.1f}%"
        if total_compliant > 0
        else "n/a"
    )

    lines.extend([
        f"**Total Matches Evaluated**: {total_evaluated}",
        f"**Total Compliant Matches**: {total_compliant} ({compliance_rate})",
        f"**Above 0.70 Threshold**: {above_threshold} ({above_threshold_pct})",
        f"**Below 0.70 Threshold**: {total_compliant - above_threshold} ({below_threshold_pct})",
        f"**Empty Detection Gaps**: {empty_detection_count} ({empty_detection_pct})",
        "",
    ])

    # Distribution table
    lines.extend([
        "### Score Distribution",
        "",
        "| Score Range | Count | Percentage |",
        "|-------------|-------|------------|",
    ])

    for bucket_label in ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]:
        count = buckets[bucket_label]
        percentage = f"{(count / total_compliant * 100):.1f}%" if total_compliant > 0 else "0.0%"
        lines.append(f"| {bucket_label} | {count} | {percentage} |")

    lines.append("")
    return "\n".join(lines)


def format_metrics(metrics: dict[str, Any], score_ranges: dict[str, Any]) -> str:
    primary = metrics["primary_only"]
    dual = metrics["dual_audience"]
    species = metrics["species"]
    score_summary = ", ".join(f"{bucket}: {len(entries)}" for bucket, entries in score_ranges.items())

    lines = [
        "### Compliance Snapshot",
        (
            f"- Pairs evaluated: {metrics['evaluated']} "
            f"(skipped taxonomy: {metrics['skipped_no_taxonomy']}, skipped empty body: {metrics['skipped_no_text']})"
        ),
        (
            "- Primary-only alignment: "
            f"{_format_ratio(primary['aligned'], primary['total'])} | "
            f"detection gaps: {primary['empty_detection']}"
        ),
        (
            "- Dual-audience alignment: "
            f"{_format_ratio(dual['aligned'], dual['total'])} | "
            f"detection gaps: {dual['empty_detection']}"
        ),
        (
            "- Species-gated alignment: "
            f"{_format_ratio(species['aligned'], species['total'])} | "
            f"detection gaps: {species['empty_detection']}"
        ),
        f"- Semantic score distribution (0.50-0.75 sample): {score_summary}",
    ]
    return "\n".join(lines)


def generate_report(
    results: list[dict[str, Any]],
    taxonomy: dict[str, dict[str, str]],
    include_supabase_analysis: bool = True,
) -> str:
    """Generate markdown report.

    Args:
        results: Legacy content analysis results from JSON.
        taxonomy: Legacy taxonomy lookup from CSV.
        include_supabase_analysis: Whether to include Supabase compliance distribution.

    Returns:
        Generated markdown report.
    """
    score_ranges = analyze_match_quality(results)
    metrics = compute_alignment_metrics(results, taxonomy)
    metrics_section = format_metrics(metrics, score_ranges)

    # Load and compute compliance distribution from Supabase
    compliance_section = ""
    if include_supabase_analysis:
        try:
            matching_data = load_matching_results()
            distribution = compute_compliant_score_distribution(
                matching_data["matchings"],
                matching_data["content_lookup"],
                matching_data["taxonomy_lookup"],
            )
            compliance_section = "\n\n" + format_compliance_score_table(distribution)
        except Exception as exc:
            compliance_section = f"\n\n## Compliant Score Distribution\n\n*Error loading data from Supabase: {exc}*\n"

    report = f"""# Semantic Match Analysis Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

This analysis examines 50 randomly sampled semantic matches with similarity scores between 0.5-0.75 to identify opportunities for improving the matching system's accuracy. The study focuses on:
- Content field selection for embedding
- Barriers to successful matching
- Recommendations for system improvements

**Key Finding**: The current semantic matching system struggles with **topic vs. content-type confusion** and **temporal/specificity mismatch** between source and target pages.

---

## 1. Fields to Target for Text Embedding with NLP

### 1.1 High-Value Content Fields

Based on analysis of successful matches (scores 0.60-0.75), these fields should be prioritized:

#### **Primary Fields (Highest Priority)**

1. **Article Title + Meta Description** (Example: Page 1, line 1-2)
   - Page 1: "Porcilis® Lawsonia ID" + "MSD Animal Health vuelve a realizar reuniones presenciales para presentar su nueva vacuna intradérmica"
   - **Why**: Highly concentrated semantic signal with product names, event types, and key concepts

2. **Main Body Text (First 500-1000 characters)** (Example: Page 4, lines 76-82)
   - Page 4: Contains "uso responsable de antimicrobianos", "ovino", specific disease names, and technical details
   - **Why**: Core content without navigation/footer noise

3. **Heading Hierarchy (H1-H3)** (Example: Page 8, lines 157-167)
   - Page 8: "¿Qué opinan los ganaderos...?", "Mejoramos mucho en detección de celos..."
   - **Why**: Structured semantic outline of page purpose

#### **Secondary Fields (Medium Priority)**

4. **Product/Technology Names** (Examples throughout)
   - "Porcilis® Lawsonia ID", "SenseHub®", "LeeO®", "IDAL", "Bravecto"
   - **Why**: Strong domain-specific entities for matching to product pages

5. **Species/Animal Type Mentions** (Example: Page 4, line 76)
   - "ovino", "porcino", "bovino", "equino", "avicultura"
   - **Why**: Critical for species landing page matching

6. **Event/Publication Date + Context** (Example: Page 1, line 18)
   - "febrero 17, 2022" + "jornadas técnicas presenciales"
   - **Why**: Helps distinguish news articles from evergreen content

### 1.2 Fields to EXCLUDE or DOWNWEIGHT

❌ **Navigation menus** - Create false semantic similarity
❌ **Footer content** - Company boilerplate repeated across all pages
❌ **Cookie/legal notices** - No semantic value
❌ **Social media sharing buttons** - Structural noise
❌ **Advertisement blocks** - Misleading content

### 1.3 Recommended Preprocessing

```python
# Weighted embedding strategy
embedding_text = (
    f"{{title}} {{title}}  # Double weight
    f"{{meta_description}}
    f"{{headings_concatenated}}
    f"{{body_first_1000_chars}}
    f"{{product_names}}  # Extracted entities
    f"{{species_types}}"  # Extracted entities
)
```

---

## 2. Issues Preventing Successful Matching

### 2.1 Content Type Mismatch (Critical Issue)

**Problem**: Source pages (specific news articles) semantically match target URLs (different specific news articles) instead of category/landing pages.

**Examples**:

- **Page 1** (score 0.5858):
  - Source: "2022 news about Porcilis Lawsonia ID intradermal vaccine"
  - Target: "2025 news about IDAL Leeo tech winning FIGAN award"
  - **Issue**: Both discuss IDAL intradermal technology but are DIFFERENT specific news articles
  - **Expected Target**: Product hub for Porcilis or general swine vaccination page

- **Page 4** (score 0.5399):
  - Source: "2017 article about antimicrobial use in sheep"
  - Target: "2025 article about Campus Porcino (swine education program)"
  - **Issue**: Both are educational/training events but DIFFERENT species and topics
  - **Why Low Score**: Weak semantic overlap - only shares "MSD education event" concept

### 2.2 Temporal Specificity Mismatch

**Problem**: Old news articles (2017-2022) matching to very recent news (2025), creating false matches.

**Example**:
- **Page 5** (score 0.5595): 2018 ESPHM conference → 2024 Bravecto award
- Both are "corporate achievement announcements" but unrelated events

**Impact**: Similarity driven by formulaic press release language, not actual topic relevance

### 2.3 Thank You / Form Submission Pages (Major Data Quality Issue)

**Problem**: Post-form-submission pages with minimal content

**Example**:
- **Page 3** (score 0.5388):
  - Source: "ES_RUBU_Gracias_contacto" (thank you page after contact form)
  - Content: Only 114 characters: "¡Muchas gracias por tu interés!"
  - Target: Contact page
  - **Issue**: No semantic content to match; this is a functional, not informational page

**Recommendation**: **Filter out pages with <200 characters or containing "gracias", "thank you", "submission confirmed"**

### 2.4 Cross-Domain Content

**Problem**: Some sources are from different domains (lawsonia.net) than target (msd-animal-health.es)

**Example**:
- **Page 10** (score 0.6553): lawsonia.net → msd-animal-health.es
- Even with good score, domain mismatch may indicate external reference content

### 2.5 Over-Generic Language in Corporate Content

**Problem**: Corporate boilerplate creates false similarity

**Repeated phrases across multiple pages**:
- "MSD Animal Health, líder en salud animal"
- "compromiso con la formación"
- "Science of Healthier Animals"

**Impact**: Pages match because of shared company messaging, not content topic

### 2.6 Homepage Matching

**Problem**: Homepage contains ALL topics, creating noisy embeddings

**Example**:
- **Page 20** (score 0.6446): Homepage → specific article
- Homepage has navigation to all sections → weak semantic signal

**Recommendation**: Treat homepage differently or exclude from matching

---

## 3. Recommendations for Improving Matching Success

### 3.1 Immediate Improvements (High Impact)

#### A. **Content Filtering Rules**

```python
# Pre-matching filters
EXCLUDE_PATTERNS = [
    r'gracias.*contacto',  # Thank you pages
    r'thank.*you',
    r'form.*submit',
    # URLs with these patterns
    r'/(contacto|contact)-confirmation',
]

MINIMUM_CONTENT_LENGTH = 200  # characters
```

#### B. **Entity-Based Boosting**

Extract and boost these entity types in embeddings:
- **Product names**: Porcilis, Bravecto, SenseHub, etc.
- **Species**: porcino, bovino, ovino, avicultura, etc.
- **Conditions**: ileítis, leishmaniosis, criptosporidiosis
- **Technologies**: IDAL, intradermal, monitorización

#### C. **Content Type Classification**

Add content type as a matching constraint:

| Source Type | Allowed Target Types |
|-------------|---------------------|
| News Article| Product Hub, Condition Page, Species Landing, Section Landing |
| Product Article | Product Page, Product Hub, Product Feature Page |
| Thank You Page | (EXCLUDE FROM MATCHING) |
| Homepage | Section Landing Page only |

### 3.2 Medium-Term Improvements

#### A. **Hierarchical Matching**

1. First match to **category** (Products, Species, Education)
2. Then match to **specific page** within that category

This prevents "Article → Article" false matches.

#### B. **Temporal Decay for News**

Reduce similarity for source/target pairs where:
- Both are news articles
- Dates are >1 year apart
- No product/topic overlap

#### C. **Negative Examples in Training**

If using supervised learning (DSPy), add negative examples:
- ❌ "2022 IDAL vaccine news" should NOT match "2025 IDAL tech award news"
- ✅ "2022 IDAL vaccine news" SHOULD match "IDAL Product Hub"

### 3.3 Long-Term Strategic Improvements

#### A. **Separate Embeddings for Different Content Strata**

1. **Entity Embedding**: Product names, species, conditions
2. **Topic Embedding**: General subject area
3. **Content-Type Embedding**: News, product page, landing page

Combine with weighted scoring:
```
final_score = 0.4 * topic_similarity
            + 0.3 * entity_similarity
            + 0.3 * content_type_compatibility
```

#### B. **URL Pattern Analysis**

Use URL structure as a feature:
- `/YYYY/MM/DD/*` → News article (match to hubs, not articles)
- `/enfermedades/*` → Condition page
- `/profesionales-de-la-salud-animal/*` → Professional landing page

#### C. **Multi-Language Handling** (If applicable)

Some content mixes Spanish with English product names. Consider:
- Translation-aware embeddings
- Multilingual models (e.g., multilingual-e5-large)

---

## 4. Specific Examples with Line Citations

### Example 1: Good Content, Wrong Target Type
**Page 1** (data/examples/content_analysis.json, lines 1-21)
- **Score**: 0.5858
- **Source Content**: Detailed technical article about Porcilis Lawsonia ID intradermal vaccine, mentions "ileítis porcina", "vacunación intradérmica", "IDAL®"
- **Actual Target**: Different news article about IDAL Leeo winning FIGAN award
- **Should Match**: Product page for Porcilis Lawsonia or IDAL technology hub
- **Why Failed**: Both articles share IDAL technology mentions → false positive

### Example 2: Minimal Content Page
**Page 3** (data/examples/content_analysis.json, lines 42-63)
- **Score**: 0.5388
- **Content Length**: 114 characters
- **Issue**: This is a form confirmation page, not informational content
- **Recommendation**: EXCLUDE from source pages before matching

### Example 3: Species Mismatch
**Page 4** (data/examples/content_analysis.json, lines 64-83)
- **Score**: 0.5399
- **Source**: Article about sheep (ovino) antimicrobial use
- **Target**: Article about swine (porcino) education program
- **Why Matched**: Both discuss veterinary education/training events
- **Should Match**: Sheep species landing page or antimicrobial stewardship page

### Example 4: Cross-Domain Reference Content
**Page 10** (data/examples/content_analysis.json, lines 192-200)
- **Score**: 0.6553 (relatively high!)
- **Source**: lawsonia.net/the-disease/clinical-signs-and-forms-of-ileitis/
- **Target**: msd-animal-health.es/enfermedades/ileitis
- **Analysis**: This is actually a GOOD match conceptually (both about ileitis disease), but the domain difference suggests this might be external reference content that shouldn't be in the source set

### Example 5: Homepage Noise
**Page 20** (data/examples/content_analysis.json, line 3)
- **Score**: 0.6446
- **Source**: Homepage (www.msd-animal-health.es/)
- **Target**: Specific news article
- **Issue**: Homepage contains links/text about everything → matches everything weakly

---

## 5. Proposed Evaluation Metrics

To measure improvement, track:

1. **Category Accuracy**: % of matches where source and target are in compatible categories
2. **Temporal Coherence**: % of matches where date difference <1 year (if both dated)
3. **Entity Overlap**: % of matches sharing at least one product/species/condition entity
4. **Content Length Gate**: % of sources with >200 chars meaningful content

**Target Goals**:
- Category Accuracy: >80% (currently estimated ~40-50%)
- Entity Overlap: >70%
- Content Length Gate: 100% (filter before matching)

---

## Appendix: Category Distribution in Sample

| Category | Count | % of Sample |
|----------|-------|-------------|
| Article | 22 | 44% |
| Condition Page | 7 | 14% |
| Generic Listing Page | 4 | 8% |
| Species Landing Pages | 4 | 8% |
| Generic Content Page | 3 | 6% |
| Product Hub | 3 | 6% |
| Product Feature Page | 2 | 4% |
| Others (7 types) | 5 | 10% |

**Observation**: 44% of targets are Articles, suggesting the system over-indexes on matching to news articles rather than evergreen content pages.

---

## Conclusion

The semantic matching system demonstrates moderate performance but suffers from **content type confusion** and **lack of structural awareness**. The primary issue is matching specific news articles to other specific news articles instead of to the evergreen taxonomy pages they should reference.

**Highest Impact Changes**:
1. ✅ Filter out form/thank-you pages pre-matching
2. ✅ Add content-type constraints (article → hub/landing, not article → article)
3. ✅ Boost entity extraction (products, species, conditions) in embeddings
4. ✅ Implement hierarchical matching (category → page)

Implementing these changes should raise the similarity score threshold from ~0.75 to ~0.85+ for confident matches.

---

**Report compiled from analysis of 50 sample pages**
**Source data**: `data/examples/content_analysis.json`
**Taxonomy reference**: `data/Spain_New.csv`
**Match snapshot**: `results/match_snapshot_20251203_151502.csv`
"""
    if "## Executive Summary" in report:
        report = report.replace("## Executive Summary\n\n", f"## Executive Summary\n\n{metrics_section}\n\n", 1)

    # Insert compliance distribution section before Appendix
    if compliance_section and "## Appendix:" in report:
        report = report.replace("## Appendix:", f"{compliance_section}\n\n---\n\n## Appendix:", 1)
    elif compliance_section:
        # Fallback: insert before conclusion if no appendix found
        report = report.replace("## Conclusion", f"{compliance_section}\n\n---\n\n## Conclusion", 1)

    return report


def main() -> None:
    """Generate and save report."""
    results = load_analysis()
    taxonomy = load_taxonomy()

    report = generate_report(results, taxonomy)

    output_file = Path(f"results/semantic_match_analysis_{datetime.now().strftime('%Y%m%d')}.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Report generated: {output_file}")
    print(f"   Total length: {len(report):,} characters")


if __name__ == "__main__":
    main()
