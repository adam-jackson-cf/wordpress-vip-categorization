"""Unit tests for the Evaluator helper."""

from uuid import uuid4

from src.models import CategorizationResult, MatchingResult, MatchStage
from src.optimization.evaluator import Evaluator


class DummySupabase:
    """Minimal stub that exposes just the methods Evaluator calls."""

    def __init__(self, matchings, categorizations_by_content):
        self._matchings = matchings
        self._categorizations = categorizations_by_content

    def get_all_matchings(self):
        return self._matchings

    def get_categorizations_by_content(self, content_id):
        return self._categorizations.get(content_id, [])


def _make_matching(
    *,
    match_stage: MatchStage,
    content_id=None,
    similarity: float,
    rubric: dict | None = None,
):
    return MatchingResult(
        taxonomy_id=uuid4(),
        content_id=content_id,
        similarity_score=similarity,
        match_stage=match_stage,
        rubric=rubric,
    )


def _make_cat(content_id):
    return CategorizationResult(
        content_id=content_id,
        category="Tech",
    )


def test_evaluator_full_flow():
    content_good = uuid4()
    content_review = uuid4()
    accepted_rubric = {
        "topic_alignment": 0.9,
        "intent_fit": 0.9,
        "entity_overlap": 0.8,
        "temporal_relevance": 0.7,
        "decision": "accept",
    }
    review_rubric = {
        "topic_alignment": 0.3,
        "intent_fit": 0.4,
        "entity_overlap": 0.2,
        "temporal_relevance": 0.5,
        "decision": "reject",
    }

    matchings = [
        _make_matching(
            match_stage=MatchStage.LLM_CATEGORIZED,
            content_id=content_good,
            similarity=0.92,
            rubric=accepted_rubric,
        ),
        _make_matching(
            match_stage=MatchStage.NEEDS_HUMAN_REVIEW,
            content_id=None,
            similarity=0.10,
            rubric=review_rubric,
        ),
        _make_matching(
            match_stage=MatchStage.NEEDS_HUMAN_REVIEW,
            content_id=content_review,
            similarity=0.55,
            rubric=review_rubric,
        ),
    ]
    categorizations = {
        content_good: [_make_cat(content_good)],
        content_review: [_make_cat(content_review)],
    }

    db = DummySupabase(matchings, categorizations)
    evaluator = Evaluator(db)

    matching_metrics = evaluator.evaluate_matching(matchings)
    assert matching_metrics["matched_count"] == 2
    assert matching_metrics["unmatched_count"] == 1
    assert matching_metrics["match_rate"] == 2 / 3

    categorization_metrics = evaluator.evaluate_categorization(categorizations[content_good])
    assert categorization_metrics["count"] == 1
    assert categorization_metrics["unique_categories"] == 1

    summary = evaluator.evaluate_all()
    assert summary["rubric"]["accepted_count"] == 1
    assert summary["rubric"]["review_count"] == 2
    assert summary["rubric"]["decisions"]["accept"] == 1
    assert summary["rubric"]["decisions"]["reject"] == 2

    low_quality = evaluator.get_low_quality_matches(min_similarity=0.6)
    assert len(low_quality) == 1
    assert low_quality[0].content_id == content_review

    unmatched = evaluator.get_unmatched_taxonomy()
    assert len(unmatched) == 1
    assert unmatched[0].content_id is None
