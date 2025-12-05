# WordPress HTML Findings (Missing Audience/Species Signals)

Generated: 2025-12-05 via `scripts/analyze_missing_signals.py` (see `missing_signal_analysis.json`).

Each entry summarizes why audience/species detection failed after inspecting the live HTML returned by WordPress.

## Summary

- 8/10 pages simply lack any explicit audience or species vocabulary; detectors behaved as expected.
- 2/10 pages (**ANEMBE** and **Vet Talks**) contain audience cues beyond our 1,500-character preview window. Extending the preview (or using metadata categories) would capture them.
- Species gaps predominantly occur on corporate/announcement posts that never mention animals; additional heuristics may be required if compliance demands a species value regardless of content.

## Detailed Notes

1. **III Jornada Grupo Piensos Costa 2025** (`46b8d54d-8a3a-4e50-ada3-a168e2cdef29`)
   - Short event splash + form (<1.2k chars). No stakeholder or species terms anywhere in HTML.

2. **ANE MBE case hub** (`f1dd0703-00d2-4aca-8bec-5a7f34c7e7d4`)
   - 5.9k-char case study; producer/veterinarian quotes only appear after ~2k chars. Preview truncation hides them, so we miss both audiences.

3. **WeForest anniversary release** (`4555bc5a-2962-43a7-966b-276058df4cb0`)
   - Sustainability article. Zero audience cues; detector misses because none exist.

4. **IoT/AI article** (`50d6319c-0038-4c23-8569-274fd678d979`)
   - Technology-focused press note. Contains audience terms early (already stored) but never references any species, so missing species is correct.

5. **Leadership appointment (Sandra Castillo)** (`e65b9693-f828-4e7b-a80d-2a4e4d2aa665`)
   - Corporate personnel update; no animal references → species absent by design.

6. **Borja Castelar promo** (`100cae57-94ff-4baa-b5ba-cc5b34a31203`)
   - Short registration teaser (~1.1k chars). Lacks any audience/species vocabulary.

7. **Gracias por tu interés** (`04a653b4-2175-4527-810b-7a8479dece8c`)
   - Thank-you/confirmation page. No meaningful copy, so detectors have no signals.

8. **Vet Talks landing** (`1265c063-f345-4fef-8eb4-864e78adf59e`)
   - 3.4k-char speaker bio; "investors" only appears deep in the page and is missed by the preview window. Species never mentioned.

9. **Reunión Vallcompanys 2025** (`eb5cfc74-355a-472e-94d1-b7972f0abf93`)
   - Event blurb with minimal text; no audience/species cues present.

10. **Allflex product page** (`c51a9792-08e4-49cd-a885-f598d70ce2f7`)
    - Species terms survive (bovine/swine) but no stakeholder is ever named, so audience stays blank.

## Next Steps

- Increase detection preview to >=4,000 chars (already discussed) so long-form resources expose late signals.
- Consider metadata-derived audiences for pages with known categories/tags when HTML lacks explicit cues.
- For corporate/news items that truly have no species, rely on compliance overrides rather than forcing noisy guesses.
