# Taxonomy-Content Matching Analysis Report
**Project:** WordPress VIP Categorization - Spain Pages
**Date:** December 3, 2025
**Analyst:** Claude Code

---

## Executive Summary

This report analyzes 35 downloaded WordPress pages from 12 Spanish domains against a taxonomy of 151 destination pages to identify optimal semantic matching strategies. The analysis reveals significant opportunities for NLP-based matching while highlighting several challenges that will require careful consideration.

**Key Findings:**
- Downloaded content represents diverse veterinary/animal health topics across 12 domains
- Taxonomy structure is well-organized with 100% field completeness
- Semantic Summary and Key Topics fields are prime candidates for embedding
- Content quality varies significantly across domains
- Language consistency (Spanish) is strong but technical depth varies
- Audience segmentation is clear but may create matching barriers

---

## 1. Taxonomy Structure Analysis

### 1.1 Overview Statistics
- **Total Taxonomy Entries:** 151 destination pages
- **Primary Domain:** msd-animal-health.es (consolidated target)
- **Field Completeness:**
  - Semantic_Summary: 100% (151/151)
  - Key_Topics: 100% (151/151)
  - ES_Page_Name: 99.3% (150/151)

### 1.2 Content Type Distribution
The taxonomy is organized into 15 distinct content types:

| Content Type | Count | Percentage |
|--------------|-------|------------|
| Condition Page | 53 | 35.1% |
| Generic Content Page | 18 | 11.9% |
| Generic Listing Page | 18 | 11.9% |
| Species Landing Pages | 16 | 10.6% |
| Product Hub | 11 | 7.3% |
| Article | 10 | 6.6% |
| Product Feature Page | 7 | 4.6% |
| Legal Page | 4 | 2.6% |
| Others | 14 | 9.3% |

**Insight:** The taxonomy heavily favors "Condition Page" content (35%), suggesting medical/disease information is a primary focus.

### 1.3 Audience Segmentation
Primary audience distribution:

| Audience | Count | Percentage |
|----------|-------|------------|
| Veterinarians | 78 | 51.7% |
| Farmers | 31 | 20.5% |
| Pet Owners | 24 | 15.9% |
| General Public | 6 | 4.0% |
| Media | 5 | 3.3% |
| Others | 7 | 4.6% |

**Insight:** The taxonomy is highly specialized toward veterinary professionals, which may create challenges when matching consumer-focused content.

### 1.4 Semantic Summary Characteristics
- **Average Length:** 67 characters
- **Range:** 32-130 characters
- **Quality:** Concise, descriptive summaries following a consistent pattern

**Example Citations:**
```
Line 2 (Spain_New.csv): "Central gateway to the MSD Animal Health Spain ecosystem.
Navigation to corporate info, veterinarian portals, and farmer tools."

Line 25 (Spain_New.csv): "Clinical resources and product info specifically for feline health."

Line 62 (Spain_New.csv): "Consumer hub for Scalibor Antiparasitic Collar.
Focus on Leishmaniasis prevention."
```

---

## 2. Downloaded Content Analysis

### 2.1 Content Distribution
Successfully downloaded 35 pages from 12 domains:

| Domain | Pages | Content Focus |
|--------|-------|---------------|
| msd-animal-health.es | 11 | Corporate news, product launches, events |
| es.mypet.com | 7 | Pet owner education, parasites, behavior |
| lawsonia.net | 4 | Swine ileitis disease information |
| vacunalavaca.com | 4 | Cattle vaccination, technical articles |
| es.numelvi.com | 2 | Canine dermatology product |
| nobivac.es | 2 | Vaccine information |
| repropig-spain.com | 2 | Swine reproduction |
| scalibor.es | 1 | Leishmaniasis prevention |
| expertosenreposicion.com | 1 | Swine health |
| es.sensehub.com | 1 | Farm monitoring technology |

### 2.2 Content Quality Assessment

#### High-Quality Educational Content
**Example: es.mypet.com_salud_bienestar_pulgas_garrapatas_pulgas_en_gatos.txt**
- **Word Count:** ~3,200 words
- **Structure:** Well-organized with 9 numbered sections
- **Content Depth:** Detailed symptoms, prevention, treatment advice
- **Title:** "9 SEÑALES DE QUE TU GATO TIENE PULGAS" (Line 5)
- **Target Audience:** Pet owners (consumer-level language)

**Key Quote (Lines 23-26):**
> "La leishmaniosis es una enfermedad parasitaria grave que puede ser mortal para los perros.
> Puede afectar también a las personas y se transmite a través de la picadura de un insecto llamado flebotomo."

**Matching Challenge:** This consumer-focused content needs to map to professional veterinary resources in the taxonomy.

#### Corporate Communications Content
**Example: msd-animal-health.es_2022_03_24_enfoque_one_health_la_via_para_prevenir_futuros_brotes_de_enfermedades_y_construir_un_mun.txt**
- **Word Count:** ~850 words
- **Content Type:** Press release / corporate news
- **Topics:** One Health initiative, CSR, Salamanca plant
- **Title:** "Enfoque One Health" (Line 5)

**Key Quote (Lines 23-24):**
> "Nuestras decisiones empresariales se entienden desde la interconexión de esos tres ejes,
> basado en la evidencia científica y la innovación en las vacunas y medicamentos"

**Matching Challenge:** Corporate content lacks specific technical detail needed for condition/disease page matching.

#### Technical Disease Content
**Example: lawsonia.net_lawsonia_new.txt**
- **Word Count:** ~350 words
- **Content Type:** Disease overview page
- **Topics:** Porcine ileitis, Lawsonia Intracellularis
- **Scientific Detail:** Mentions pathogen characteristics, prevalence data

**Key Quote (Lines 27-28):**
> "Infectious disease caused by Lawsonia Intracellularis affecting the intestine of pigs
> and causing significant production losses mainly during growth and fattening."

**Matching Potential:** HIGH - Technical disease content matches well with "Condition Page" taxonomy entries.

#### Blog/Article Content
**Example: vacunalavaca.com_home.txt**
- **Word Count:** ~600 words
- **Content Type:** Blog homepage with article summaries
- **Topics:** Multiple cattle health topics (criptosporidiosis, coronavirus, coccidiosis)
- **Structure:** Article listing with categories and tags

**Key Content (Lines 33-49):**
> "Criptosporidiosis en terneras: el secreto está en el cicloII"
> "Los factores de riesgo más importantes giran en torno al ciclo del parásito"
> "CÓMO IDENTIFICAR LOS SIGNOS DEL CORONAVIRUS RESPIRATORIO BOVINO (BOCV)"
> "LA REVISIÓN DEL VIRUS DE LA LENGUA AZUL"

**Matching Challenge:** Blog aggregation pages contain multiple topics, making single-destination matching difficult.

### 2.3 Content Characteristics Summary

| Characteristic | Observation | Impact on Matching |
|----------------|-------------|-------------------|
| Language | Consistent Spanish | ✓ Positive |
| Technical Depth | Highly variable (consumer to clinical) | ⚠ Challenge |
| Content Length | 350-4,000 words | ⚠ Variable signal strength |
| Topic Focus | Single vs. multi-topic pages | ⚠ Diluted semantic matching |
| Audience Level | Consumer, farmer, veterinarian | ⚠ Segmentation mismatch |
| Structure | Headers, lists, numbered sections | ✓ Positive for extraction |

---

## 3. Recommended Fields for NLP Text Embedding

Based on the analysis, the following fields should be targeted for semantic embedding:

### 3.1 Primary Embedding Fields (Taxonomy)

#### 1. Semantic_Summary (Priority: HIGH)
**Rationale:**
- 100% field completeness
- Consistent length (~67 characters)
- Captures core page purpose and value proposition
- Includes target audience context

**Example (Line 65, Spain_New.csv):**
```
"Educational content on the dangers and symptoms of Leishmaniasis in dogs."
```

**Recommendation:** Embed this field as the primary semantic anchor for matching.

#### 2. Key_Topics (Priority: HIGH)
**Rationale:**
- 100% field completeness
- Comma-separated keywords provide semantic breadth
- Covers technical terms, species, conditions, and concepts
- 3-8 topics per entry

**Example (Line 68, Spain_New.csv):**
```
"Fleas, Itch, Infestation, Parasites"
```

**Recommendation:** Embed as secondary signal, potentially with keyword extraction preprocessing.

#### 3. ES_Page_Name + English_Page Name (Priority: MEDIUM)
**Rationale:**
- Page titles often contain primary topic
- Bilingual coverage improves matching robustness
- Short, focused semantic signal

**Example (Line 25, Spain_New.csv):**
```
English: "Feline"
Spanish: "Gatos"
```

**Recommendation:** Combine both language versions for embedding to capture translation variations.

### 3.2 Source Content Embedding Strategy

For downloaded source pages, extract and embed:

#### 1. Page Title (Priority: HIGH)
**Example:** "9 SEÑALES DE QUE TU GATO TIENE PULGAS"
- Strong semantic signal
- Often contains primary topic/condition

#### 2. Main Headings (H1-H3) (Priority: HIGH)
**Example (es.mypet.com_salud_bienestar_pulgas_garrapatas_pulgas_en_gatos.txt, Lines 8-17):**
```
"9 SEÑALES DE QUE TU GATO TIENE PULGAS"
"1. Exceso de acicalamiento"
"2. Desarrollo de rojeces en la piel o protuberancias parecidas a la sarna"
```

**Recommendation:** Extract first 5-10 headings as structured topic indicators.

#### 3. First 500 Words of Content (Priority: MEDIUM)
**Rationale:**
- Opening paragraphs typically establish topic context
- Balances semantic richness with processing efficiency
- Reduces noise from footer/boilerplate content

#### 4. URL Path Segments (Priority: LOW-MEDIUM)
**Example:** `salud_bienestar/pulgas_garrapatas/pulgas_en_gatos`
- Provides hierarchical topic context
- Language-agnostic semantic hints

### 3.3 Combined Embedding Architecture

**Recommended Approach:**
1. **Primary Match:** Embed `Semantic_Summary` vs. source page `Title + First 5 Headings`
2. **Secondary Match:** Embed `Key_Topics` vs. source page `First 500 words`
3. **Tertiary Filter:** Use `Content_Type` and `Primary_Audience` as post-processing filters
4. **Confidence Boost:** URL path keyword overlap analysis

**Embedding Model Recommendation:**
- **Model:** `text-embedding-3-large` or multilingual BERT (e.g., `distilbert-base-multilingual-cased`)
- **Dimensionality:** 768-1024 dimensions
- **Language:** Spanish language support essential

---

## 4. Matching Challenges & Issues

### 4.1 Audience Segmentation Mismatch

**Issue:** Source pages targeting "Pet Owners" (consumer-level) need to match taxonomy pages targeting "Veterinarians" (professional-level).

**Example:**
- **Source:** es.mypet.com_salud_bienestar_pulgas_garrapatas_pulgas_en_gatos.txt
  - Consumer language: "¿Cómo saber si tu gato tiene pulgas?" (Line 22)
  - Practical advice: "acude a tu veterinario" (repeated throughout)
- **Potential Taxonomy Match:** Line 73, Spain_New.csv
  - `Content_Type: "Generic Listing Page"`
  - `Primary_Audiance: "Pet Owners"`
  - `Semantic_Summary: "Common questions about Scalibor efficacy, water resistance, and safety."`

**Impact:** Semantic similarity may be high, but audience level creates conceptual distance.

**Recommendation:**
- Use `Primary_Audience` as a soft filter (not hard constraint)
- Consider audience-level translation in matching algorithm
- Accept cross-audience matches when topic alignment is strong

### 4.2 Multi-Topic vs. Single-Topic Pages

**Issue:** Blog homepage and listing pages contain multiple topics, while taxonomy entries are single-topic focused.

**Example:**
- **Source:** vacunalavaca.com_home.txt (Lines 33-77)
  - Topics: Criptosporidiosis, Coronavirus Respiratorio, Lengua Azul, Coccidiosis
  - Structure: Article listing with brief summaries
- **Taxonomy:** Individual condition pages (Lines 27-30, Spain_New.csv)
  - Separate entries for Dairy Cattle, Beef Cattle, Sheep, Swine

**Impact:**
- Diluted semantic signal (multiple topics reduce embedding specificity)
- One-to-many mapping challenge (should map to parent section page)

**Recommendation:**
- Detect multi-topic pages using heading count/diversity analysis
- Map blog homepages to "Generic Listing Page" entries (e.g., Line 35, Spain_New.csv)
- Use hierarchical matching (homepage → section landing → specific content)

### 4.3 Content Depth Variability

**Issue:** Some source pages are thin on content, reducing semantic matching confidence.

**Examples:**
- **Thin Content:** es.numelvi.com_2025_03_13_hello_world.txt (~941 bytes, Line 8 from file listing)
  - Minimal content for embedding
  - "Hello World" default post
- **Rich Content:** es.mypet.com_salud_bienestar_pulgas_garrapatas_pulgas_en_gatos.txt (~29KB, Line 7 from file listing)
  - Comprehensive information
  - Strong semantic signals

**Impact:**
- Low confidence scores for thin content
- Risk of false negative matches

**Recommendation:**
- Set minimum content threshold (e.g., 200 words)
- Flag thin content pages for manual review
- Consider using URL/title only for pages below threshold

### 4.4 Product-Specific vs. General Information

**Issue:** Some source pages are highly product-specific (e.g., Scalibor collar), while taxonomy may group by condition or species.

**Example:**
- **Source:** scalibor.es_leishmaniosis_canina.txt (Lines 23-32)
  - Product: Scalibor collar
  - Condition: Leishmaniasis
  - Quote: "protegido durante los 12 meses frente al flebotomo, insecto transmisor de la leishmaniosis"
- **Taxonomy:** Multiple potential matches
  - Line 62: Product Hub for Scalibor (product-focused)
  - Line 65: Condition page for Leishmaniasis (disease-focused)
  - Line 67: Condition page for Sandflies (vector-focused)

**Impact:** One source page could semantically match multiple taxonomy entries.

**Recommendation:**
- Prioritize product hub pages when product name is prominent in source content
- Use brand name detection as matching signal
- Consider content hierarchy: Product Hub > Condition Page > Vector/Symptom Page

### 4.5 Language Mixing (Spanish/English)

**Issue:** Some content contains English sections, particularly technical/scientific terms.

**Examples:**
- lawsonia.net pages contain mixed Spanish/English (predominantly English)
- Technical terms: "Lawsonia Intracellularis", "IPMA", "ELISA" (Line 33, lawsonia.net_lawsonia_new.txt)

**Impact:**
- Embedding models may handle language mixing differently
- Potential semantic distance from pure Spanish taxonomy

**Recommendation:**
- Use multilingual embedding models
- Test language detection and potentially apply translation preprocessing
- Consider language similarity as matching feature

### 4.6 Temporal Content (News/Events)

**Issue:** Press releases and event announcements are time-specific and may not have clear taxonomy equivalents.

**Example:**
- **Source:** msd-animal-health.es_2025_06_24_finaliza_la_iii_edicion_del_campus_porcino (filename)
  - Event-specific: "III edición del Campus Porcino"
  - Temporal: June 24, 2025
- **Taxonomy:** Line 19, Spain_New.csv
  - Generic: "Report on the conclusion of the 3rd 'Campus Porcino'"
  - Content_Type: Article

**Impact:**
- Temporal references create noise in semantic matching
- Event-specific content may lack general taxonomy equivalent

**Recommendation:**
- Detect temporal/event content using date patterns and event keywords
- Map to "News & Events" listing pages (e.g., Lines 15, 35, 47, Spain_New.csv)
- Extract core topic (e.g., "Campus Porcino" → swine education) for secondary matching

### 4.7 Legal/Compliance Pages

**Issue:** Legal pages (Terms & Conditions, Privacy Policy, Cookie Policy) have minimal semantic variation but are essential.

**Example:**
- **Source:** repropig-spain.com_politica_de_cookies.txt
  - Standard compliance content
  - Low semantic differentiation from other legal pages
- **Taxonomy:** Lines 3-6, Spain_New.csv (4 legal page entries)

**Impact:**
- Legal content is formulaic and may generate false positive matches across legal pages

**Recommendation:**
- Use URL pattern detection for legal pages (/terms, /privacy, /cookies, /whistleblower)
- Apply rule-based matching for legal content
- Exclude from semantic embedding pipeline (use metadata matching instead)

---

## 5. Suggestions for Improving Matching Success

### 5.1 Pre-Processing Enhancements

1. **Content Cleaning**
   - Remove boilerplate navigation text (detected in multiple files)
   - Extract main content section only
   - Remove social sharing links ("Twitter, Facebook, YouTube, LinkedIn, Instagram")

2. **Keyword Extraction**
   - Apply TF-IDF or KeyBERT to extract top 10 keywords from source content
   - Compare against `Key_Topics` field in taxonomy
   - Use keyword overlap as confidence modifier

3. **Entity Recognition**
   - Extract species mentions (perros, gatos, vacas, cerdos, etc.)
   - Extract product names (Scalibor, Bravecto, Nobivac, etc.)
   - Extract conditions/diseases (leishmaniosis, pulgas, ileitis, etc.)
   - Use entity overlap for match validation

### 5.2 Multi-Stage Matching Pipeline

**Stage 1: Semantic Embedding Match**
- Embed taxonomy `Semantic_Summary + Key_Topics`
- Embed source `Title + Headings + First 500 words`
- Compute cosine similarity
- Threshold: ≥0.75 for auto-accept

**Stage 2: Metadata Filtering**
- Filter by `Content_Type` appropriateness
  - Homepage → Homepage or Landing Page
  - Article → Article or Generic Content Page
  - Product → Product Hub
- Filter by `Primary_Audience` (soft constraint, ±1 level)

**Stage 3: Confidence Scoring**
- URL path keyword overlap (+10%)
- Entity overlap (+5-15% based on count)
- Content length similarity (+5%)
- Language consistency (+5%)

**Stage 4: Manual Review**
- Scores 0.60-0.75: Flag for manual review
- Scores <0.60: No match found (orphan page)

### 5.3 Taxonomy Enhancements

**Add Field: `Related_Conditions`**
- Cross-reference related disease/condition pages
- Example: Leishmaniasis → Sandflies, Ticks, Fleas
- Enables graph-based matching

**Add Field: `Product_Mentions`**
- List products related to this page
- Improves product page → condition page matching

**Add Field: `Target_Keywords`**
- Explicit keywords for SEO and matching purposes
- Supplements `Key_Topics` with matching-specific terms

### 5.4 Hybrid Matching Approach

Combine multiple signals:
1. **Semantic Similarity** (50% weight)
2. **Keyword/Entity Overlap** (25% weight)
3. **Metadata Alignment** (15% weight)
4. **URL Pattern Similarity** (10% weight)

### 5.5 Iterative Refinement

1. **Generate Initial Matches**
   - Run matching algorithm on full dataset
   - Identify high-confidence matches (≥0.85)

2. **Analyze Failures**
   - Review low-confidence matches
   - Identify systematic patterns in failures

3. **Retrain/Refine**
   - Adjust field weights
   - Add new matching features
   - Update content type rules

4. **Validate Sample**
   - Manual validation of 100 random matches
   - Calculate precision, recall, F1 scores
   - Iterate until metrics meet targets (e.g., precision ≥90%)

---

## 6. Predicted Matching Success by Content Type

Based on analysis, predicted matching success rates:

| Source Content Type | Predicted Success Rate | Reasoning |
|---------------------|------------------------|-----------|
| Disease/Condition Pages | 85-95% | Strong semantic alignment with "Condition Page" taxonomy entries |
| Product Pages (branded) | 80-90% | Clear product hub matches in taxonomy |
| Species-Specific Content | 75-85% | "Species Landing Pages" provide good targets |
| Educational Articles | 70-80% | Variable depth; may match multiple taxonomy entries |
| Blog Homepages | 50-60% | Multi-topic issue; should map to listing pages |
| Corporate News/PR | 40-50% | Temporal content; limited taxonomy coverage |
| Legal/Compliance | 95-100% | Rule-based matching; small, well-defined set |

**Overall Predicted Success Rate: 70-80%** with manual review for remaining 20-30%.

---

## 7. Items Likely to Prevent Matching Success

### 7.1 Orphan Pages (No Clear Taxonomy Equivalent)

**Example 1: Podcast Pages**
- **Source:** repropig-spain.com_podcast (mentioned in domain analysis)
- **Issue:** Taxonomy lacks "Podcast" content type
- **Impact:** Will require creation of new taxonomy entry or mapping to parent section

**Example 2: Video Gallery Pages**
- **Source:** es.sensehub.com_video_section_sample.txt (Line 11, file listing)
- **Content:** Video listing page with minimal text
- **Issue:** Insufficient semantic content; video-specific format
- **Recommendation:** Map to "Generic Listing Page" or create multimedia taxonomy category

**Example 3: Interactive Tools/Calculators**
- **Source:** Likely in "Tools & Resources" pages (not in sample)
- **Issue:** Functional pages with minimal descriptive content
- **Recommendation:** Rule-based matching using URL patterns (/tools, /calculator, /resources)

### 7.2 Highly Technical Scientific Content

**Example:** lawsonia.net pages with deep scientific detail
- **Content (Lines 63-75, lawsonia.net_lawsonia_new.txt):**
  > "L. intracellularis is a gram-negative rod with a sigmoid or curved shape
  > and with a single long flagellum. Among all the serological tests developed so far,
  > only blocking ELISA is commercially available globally."
- **Issue:** Scientific depth may exceed taxonomy `Semantic_Summary` scope
- **Impact:** Semantic distance due to abstraction level mismatch

**Recommendation:**
- Use broader topic matching (ileitis → swine health)
- Consider scientific terminology as high-value matching signal

### 7.3 Multi-Language Content

**Example:** lawsonia.net (primarily English content)
- **Issue:** Taxonomy is Spanish-focused
- **Impact:** Language barrier in semantic matching
- **Recommendation:**
  - Apply translation preprocessing
  - Use multilingual embedding models
  - Consider separate matching pipeline for non-Spanish content

### 7.4 User-Generated Content

**Example:** Product review pages (mentioned in taxonomy)
- **Source:** Potentially "reviews" sections like Line 63, Spain_New.csv
- **Issue:** Review content is variable, subjective, and may not exist in source
- **Recommendation:** Map to parent product page if reviews don't exist in source

---

## 8. Conclusions & Next Steps

### 8.1 Key Takeaways

1. **Strong Foundation**: Taxonomy structure is well-designed with 100% field completeness
2. **Primary Embedding Fields**: `Semantic_Summary` and `Key_Topics` are optimal for NLP matching
3. **Content Variability**: Source content ranges from thin (900 bytes) to rich (29KB), requiring adaptive strategies
4. **Audience Mismatch**: Consumer vs. professional content creates semantic distance
5. **Multi-Topic Challenge**: Blog/listing pages require hierarchical matching approach
6. **High Success Potential**: 70-80% automated matching achievable with hybrid approach

### 8.2 Implementation Recommendations

**Phase 1: Foundation (Week 1-2)**
1. Implement embedding pipeline for taxonomy fields
2. Extract and clean source content
3. Generate baseline semantic similarity scores

**Phase 2: Enhancement (Week 3-4)**
4. Add keyword/entity extraction
5. Implement metadata filtering
6. Build hybrid scoring system

**Phase 3: Validation (Week 5-6)**
7. Manual validation of 100 sample matches
8. Refine matching thresholds
9. Document edge cases and exceptions

**Phase 4: Production (Week 7-8)**
10. Run full matching pipeline
11. Generate match reports with confidence scores
12. Manual review queue for low-confidence matches

### 8.3 Success Metrics

- **Precision Target:** ≥90% (manual validation of auto-matches)
- **Recall Target:** ≥75% (percentage of source pages matched)
- **Manual Review Rate:** ≤20% (pages requiring human review)
- **Processing Time:** <5 minutes for full 1,416 page dataset

---

## Appendix A: Sample File References

### Downloaded Content Files Analyzed
1. `es.mypet.com_salud_bienestar_pulgas_garrapatas_pulgas_en_gatos.txt` (29KB)
2. `msd-animal-health.es_2022_03_24_enfoque_one_health_la_via_para_prevenir_futuros_brotes_de_enfermedades_y_construir_un_mun.txt` (20KB)
3. `scalibor.es_leishmaniosis_canina.txt` (8.0KB)
4. `vacunalavaca.com_home.txt` (14KB)
5. `lawsonia.net_lawsonia_new.txt` (7.4KB)

### Taxonomy Reference
- **File:** `Spain_New.csv`
- **Total Entries:** 151
- **Key Fields:** UID, Destination_URL, English_Page Name, ES_Page_Name, Content_Type, Primary_Audiance, Secondary_Audiance, Semantic_Summary, Key_Topics

---

## Appendix B: Technical Specifications

### Recommended Embedding Configuration

```python
# Primary embedding field
taxonomy_text = f"{row['Semantic_Summary']} {row['Key_Topics']}"

# Source content embedding
source_text = f"{page_title} {' '.join(headings[:5])} {content[:500]}"

# Embedding model
model = "text-embedding-3-large"  # or multilingual-bert
dimensions = 1024

# Similarity threshold
auto_match_threshold = 0.75
manual_review_threshold = 0.60
```

### Match Scoring Formula

```python
final_score = (
    semantic_similarity * 0.50 +
    keyword_overlap * 0.25 +
    metadata_alignment * 0.15 +
    url_similarity * 0.10
)
```

---

**Report Generated:** December 3, 2025
**Total Source Pages Analyzed:** 35/50 (70% download success rate)
**Taxonomy Version:** Spain_New.csv (151 entries)
**Analysis Depth:** Detailed content review with manual citation extraction
