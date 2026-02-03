# src/evaluation/metrics.py
"""
Evaluation metrics for Egyptian RAG translation system
"""
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
import nltk

# Download required NLTK data
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except:
    pass


# ============================================================================
# TRANSLATION QUALITY METRICS
# ============================================================================


def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score (0-100)

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        float: BLEU score (0-100)
    """
    try:
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        smoothing = SmoothingFunction()
        bleu_score = sentence_bleu(
            [reference_tokens], hypothesis_tokens, smoothing_function=smoothing.method1
        )
        return bleu_score * 100
    except Exception as e:
        print(f"Warning: BLEU calculation failed: {e}")
        return 0.0


def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        dict: ROUGE-1, ROUGE-2, ROUGE-L scores (0-100)
    """
    try:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure * 100,
            "rouge2": scores["rouge2"].fmeasure * 100,
            "rougeL": scores["rougeL"].fmeasure * 100,
        }
    except Exception as e:
        print(f"Warning: ROUGE calculation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculate_meteor(reference, hypothesis):
    """
    Calculate METEOR score (0-100)

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        float: METEOR score (0-100)
    """
    try:
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        meteor = meteor_score([reference_tokens], hypothesis_tokens)
        return meteor * 100
    except Exception as e:
        print(f"Warning: METEOR calculation failed: {e}")
        return 0.0


def calculate_chrf(reference, hypothesis):
    """
    Calculate chrF score (0-100)

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        float: chrF score (0-100)
    """
    try:
        chrf = CHRF()
        score = chrf.sentence_score(hypothesis, [reference])
        return score.score  # Already 0-100
    except Exception as e:
        print(f"Warning: chrF calculation failed: {e}")
        return 0.0


def calculate_exact_match(reference, hypothesis):
    """
    Calculate exact match score

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        float: 100.0 if exact match, 0.0 otherwise
    """
    return 100.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0


def calculate_word_overlap(reference, hypothesis):
    """
    Calculate word-level overlap percentage

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        float: Overlap percentage (0-100)
    """
    try:
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if len(ref_words) == 0:
            return 0.0
        overlap = len(ref_words.intersection(hyp_words))
        return (overlap / len(ref_words)) * 100
    except Exception as e:
        print(f"Warning: Word overlap calculation failed: {e}")
        return 0.0


# ============================================================================
# RETRIEVAL QUALITY METRICS
# ============================================================================


def calculate_recall_at_k(
    reference_german, retrieved_examples, k_values=[1, 3, 5, 10, 20]
):
    """
    Calculate Recall@K - checks if reference appears in top K results

    Args:
        reference_german: The ground truth German translation
        retrieved_examples: List of retrieved examples from RAG
        k_values: List of K values to calculate recall for

    Returns:
        dict: Recall@K for each K value
    """
    recalls = {}

    for k in k_values:
        found = False
        for i, example in enumerate(retrieved_examples[:k]):
            if i >= len(retrieved_examples):
                break
            retrieved_german = example["payload"]["translation_de"]
            # Check if the reference matches (exact or high similarity)
            if reference_german.strip().lower() == retrieved_german.strip().lower():
                found = True
                break

        recalls[f"recall@{k}"] = 100.0 if found else 0.0

    return recalls


def calculate_mrr(reference_german, retrieved_examples):
    """
    Calculate Mean Reciprocal Rank (MRR)

    MRR = 1 / rank of first relevant result
    If no relevant result found, MRR = 0

    Args:
        reference_german: The ground truth German translation
        retrieved_examples: List of retrieved examples from RAG

    Returns:
        float: MRR score (0-100)
    """
    for i, example in enumerate(retrieved_examples):
        retrieved_german = example["payload"]["translation_de"]
        # Check if this is a relevant result
        if reference_german.strip().lower() == retrieved_german.strip().lower():
            # Rank starts at 1, not 0
            mrr = 1.0 / (i + 1)
            return mrr * 100  # Convert to percentage

    # No relevant result found
    return 0.0


def calculate_average_retrieval_score(retrieved_examples, top_k=10):
    """
    Calculate average retrieval score from top K results

    Args:
        retrieved_examples: List of retrieved examples with scores
        top_k: Number of top results to consider

    Returns:
        float: Average RRF score (0-100)
    """
    if not retrieved_examples:
        return 0.0

    scores = [example["rrf_score"] for example in retrieved_examples[:top_k]]
    avg_score = np.mean(scores) if scores else 0.0
    return avg_score * 100  # Convert to percentage


# ============================================================================
# COMBINED METRICS CALCULATION
# ============================================================================


def calculate_all_translation_metrics(reference, hypothesis):
    """
    Calculate all translation quality metrics

    Args:
        reference: Reference translation
        hypothesis: Generated translation

    Returns:
        dict: All translation metrics
    """
    rouge_scores = calculate_rouge(reference, hypothesis)

    metrics = {
        "bleu": calculate_bleu(reference, hypothesis),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "meteor": calculate_meteor(reference, hypothesis),
        "chrf": calculate_chrf(reference, hypothesis),
        "exact_match": calculate_exact_match(reference, hypothesis),
        "word_overlap": calculate_word_overlap(reference, hypothesis),
    }

    return metrics


def calculate_all_retrieval_metrics(reference_german, retrieved_examples):
    """
    Calculate all retrieval quality metrics

    Args:
        reference_german: Ground truth German translation
        retrieved_examples: List of retrieved examples from RAG

    Returns:
        dict: All retrieval metrics
    """
    recall_scores = calculate_recall_at_k(
        reference_german, retrieved_examples, k_values=[1, 3, 5, 10, 20]
    )

    metrics = {
        **recall_scores,
        "mrr": calculate_mrr(reference_german, retrieved_examples),
        "avg_retrieval_score": calculate_average_retrieval_score(
            retrieved_examples, top_k=10
        ),
    }

    return metrics


def calculate_all_metrics(
    reference_english, hypothesis_english, reference_german, retrieved_examples
):
    """
    Calculate all metrics (translation + retrieval)

    Args:
        reference_english: Reference English translation
        hypothesis_english: Generated English translation
        reference_german: Reference German translation
        retrieved_examples: Retrieved examples from RAG

    Returns:
        dict: All metrics combined
    """
    translation_metrics = calculate_all_translation_metrics(
        reference_english, hypothesis_english
    )
    retrieval_metrics = calculate_all_retrieval_metrics(
        reference_german, retrieved_examples
    )

    return {**translation_metrics, **retrieval_metrics}


if __name__ == "__main__":
    # Test metrics
    reference = "A sacrifice given by the King"
    hypothesis = "An offering that the King gives"

    print("Testing translation metrics...")
    metrics = calculate_all_translation_metrics(reference, hypothesis)

    print("\n" + "=" * 70)
    print("TRANSLATION METRICS TEST")
    print("=" * 70)
    for metric, score in metrics.items():
        print(f"{metric:20s}: {score:6.2f}%")
