# src/evaluation/evaluate_rag.py
"""
Complete evaluation script for RAG vs LLM-only comparison
"""
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from src.config import settings
from src.pipeline.rag_pipeline import RAGPipeline
from src.translation.german_to_english import GermanToEnglishTranslator
from src.llm.ollama_client import OllamaClient
from src.preprocessing.normalization import normalize_transliteration
from src.evaluation.metrics import calculate_all_metrics
import re


class RAGEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self, pipeline=None):
        """
        Initialize evaluator

        Args:
            pipeline: RAGPipeline instance (creates new if None)
        """
        print("\n" + "=" * 70)
        print("🔬 Initializing RAG Evaluator")
        print("=" * 70)

        self.pipeline = pipeline or RAGPipeline(in_memory=False)
        self.de_to_en_translator = GermanToEnglishTranslator()

        print("✅ RAG Evaluator ready!")

    def evaluate_test_set(self, test_df, max_samples=None):
        """
        Evaluate RAG system on test set

        Args:
            test_df: Test DataFrame
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            tuple: (results_df, metrics_df, failed_list)
        """
        if max_samples:
            test_df = test_df.head(max_samples)

        print("\n" + "=" * 70)
        print(f"🔄 Evaluating RAG System on {len(test_df)} samples")
        print("=" * 70)

        results = []
        failed = []

        for idx in tqdm(range(len(test_df)), desc="RAG Translation"):
            try:
                query_original = test_df.iloc[idx]["transliteration"]
                reference_german = test_df.iloc[idx]["translation"]

                # Translate using RAG
                result = self.pipeline.translate(
                    query_original=query_original, show_details=False
                )

                if result["success"]:
                    # Translate reference German to English
                    reference_english = self.de_to_en_translator.translate(
                        reference_german
                    )

                    if reference_english:
                        results.append(
                            {
                                "sample_id": idx,
                                "transliteration": query_original,
                                "transliteration_normalized": result[
                                    "query_normalized"
                                ],
                                "reference_german": reference_german,
                                "reference_english": reference_english,
                                "predicted_german": result["german"],
                                "predicted_english": result["english"],
                                "top_matches": result["top_matches"],
                            }
                        )
                    else:
                        failed.append(
                            {
                                "sample_id": idx,
                                "reason": "Reference translation to English failed",
                            }
                        )
                else:
                    failed.append(
                        {
                            "sample_id": idx,
                            "reason": result.get("error", "RAG translation failed"),
                        }
                    )

            except Exception as e:
                failed.append({"sample_id": idx, "reason": f"Exception: {str(e)}"})
                continue

        results_df = pd.DataFrame(results)

        print(f"\n✅ RAG evaluation complete!")
        print(f"   Successful: {len(results)}")
        print(f"   Failed: {len(failed)}")

        # Calculate metrics
        metrics_df = self._calculate_metrics(results_df)

        return results_df, metrics_df, failed

    def _calculate_metrics(self, results_df):
        """Calculate metrics for all results"""
        print("\n" + "=" * 70)
        print("📊 Calculating metrics...")
        print("=" * 70)

        metrics_list = []

        for idx, row in tqdm(
            results_df.iterrows(), total=len(results_df), desc="Computing metrics"
        ):
            reference_english = row["reference_english"]
            hypothesis_english = row["predicted_english"]
            reference_german = row["reference_german"]
            retrieved_examples = row["top_matches"]

            # Calculate all metrics
            metrics = calculate_all_metrics(
                reference_english,
                hypothesis_english,
                reference_german,
                retrieved_examples,
            )
            metrics["sample_id"] = row["sample_id"]

            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)

        print("✅ Metrics calculation complete!")

        return metrics_df


class LLMOnlyEvaluator:
    """Evaluate LLM-only (no RAG) performance"""

    def __init__(self, ollama_client=None):
        """
        Initialize evaluator

        Args:
            ollama_client: OllamaClient instance (creates new if None)
        """
        print("\n" + "=" * 70)
        print("🔬 Initializing LLM-Only Evaluator")
        print("=" * 70)

        self.ollama_client = ollama_client or OllamaClient()
        self.de_to_en_translator = GermanToEnglishTranslator()

        print("✅ LLM-Only Evaluator ready!")

    def translate_without_rag(self, query_original, query_normalized):
        """
        Direct LLM translation WITHOUT RAG retrieval

        Args:
            query_original: Original Egyptian transliteration
            query_normalized: Normalized transliteration

        Returns:
            str: German translation or None
        """
        prompt = f"""You are an expert linguist specialized in Earlier Egyptian grammar and historical translation.

Translate the following Earlier Egyptian transliteration into German.
Use a conservative, grammar-based interpretation.
Do not modernize meanings or add implied words.

Egyptian Transliteration:
{query_original}

Output ONLY the German translation.
Do not add explanations, comments, or alternative readings.

German Translation:
"""

        try:
            llm_output, _ = self.ollama_client.generate(
                system_prompt="You are an expert Ancient Egyptian linguist. Translate Earlier Egyptian to German.",
                user_prompt=prompt,
            )

            if llm_output:
                # Clean the response
                german_translation = llm_output.strip()
                # Remove "German Translation:" prefix if present
                german_translation = re.sub(
                    r"^German Translation:\s*",
                    "",
                    german_translation,
                    flags=re.IGNORECASE,
                )
                return german_translation.strip()

            return None

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None

    def evaluate_test_set(self, test_df, max_samples=None):
        """
        Evaluate LLM-only system on test set

        Args:
            test_df: Test DataFrame
            max_samples: Maximum number of samples to evaluate

        Returns:
            tuple: (results_df, metrics_df, failed_list)
        """
        if max_samples:
            test_df = test_df.head(max_samples)

        print("\n" + "=" * 70)
        print(f"🔄 Evaluating LLM-Only on {len(test_df)} samples")
        print("=" * 70)

        results = []
        failed = []

        for idx in tqdm(range(len(test_df)), desc="LLM-only translation"):
            try:
                query_original = test_df.iloc[idx]["transliteration"]
                query_normalized = normalize_transliteration(query_original)
                reference_german = test_df.iloc[idx]["translation"]

                # Translate using LLM only
                german_translation = self.translate_without_rag(
                    query_original, query_normalized
                )

                if german_translation:
                    english_translation = self.de_to_en_translator.translate(
                        german_translation
                    )

                    if english_translation:
                        # Get reference English (reuse from RAG if available)
                        reference_english = self.de_to_en_translator.translate(
                            reference_german
                        )

                        results.append(
                            {
                                "sample_id": idx,
                                "transliteration": query_original,
                                "reference_german": reference_german,
                                "reference_english": reference_english,
                                "predicted_german_llm": german_translation,
                                "predicted_english_llm": english_translation,
                            }
                        )
                    else:
                        failed.append(
                            {"sample_id": idx, "reason": "English translation failed"}
                        )
                else:
                    failed.append(
                        {"sample_id": idx, "reason": "LLM translation failed"}
                    )

            except Exception as e:
                failed.append({"sample_id": idx, "reason": f"Exception: {str(e)}"})
                continue

        results_df = pd.DataFrame(results)

        print(f"\n✅ LLM-only evaluation complete!")
        print(f"   Successful: {len(results)}")
        print(f"   Failed: {len(failed)}")

        # Calculate metrics (no retrieval metrics for LLM-only)
        metrics_df = self._calculate_metrics(results_df)

        return results_df, metrics_df, failed

    def _calculate_metrics(self, results_df):
        """Calculate metrics for LLM-only results"""
        from src.evaluation.metrics import calculate_all_translation_metrics

        print("\n" + "=" * 70)
        print("📊 Calculating LLM-only metrics...")
        print("=" * 70)

        metrics_list = []

        for idx, row in tqdm(
            results_df.iterrows(), total=len(results_df), desc="Computing metrics"
        ):
            reference = row["reference_english"]
            hypothesis = row["predicted_english_llm"]

            # Calculate translation metrics only
            metrics = calculate_all_translation_metrics(reference, hypothesis)

            # Add zero retrieval metrics (no retrieval for LLM-only)
            metrics.update(
                {
                    "recall@1": 0.0,
                    "recall@3": 0.0,
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "recall@20": 0.0,
                    "mrr": 0.0,
                    "avg_retrieval_score": 0.0,
                }
            )

            metrics["sample_id"] = row["sample_id"]

            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)

        print("✅ Metrics calculation complete!")

        return metrics_df


def compare_rag_vs_llm(rag_metrics_df, llm_metrics_df):
    """
    Compare RAG vs LLM-only performance

    Args:
        rag_metrics_df: RAG metrics DataFrame
        llm_metrics_df: LLM-only metrics DataFrame

    Returns:
        dict: Comparison summary
    """
    print("\n" + "=" * 70)
    print("🆚 RAG vs LLM-Only Comparison")
    print("=" * 70)

    # Calculate averages
    rag_avg = {}
    llm_avg = {}

    metric_columns = [col for col in rag_metrics_df.columns if col != "sample_id"]

    for metric in metric_columns:
        rag_avg[metric] = rag_metrics_df[metric].mean()
        llm_avg[metric] = llm_metrics_df[metric].mean()

    # Print comparison
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'RAG':<15} {'LLM-Only':<15} {'Diff':<15} {'Winner':<15}")
    print("=" * 70)

    for metric in metric_columns:
        rag_score = rag_avg[metric]
        llm_score = llm_avg[metric]
        diff = rag_score - llm_score

        if "recall" in metric.lower() or metric.lower() in [
            "mrr",
            "avg_retrieval_score",
        ]:
            winner = "🏆 RAG (only)" if diff > 0 else "N/A"
        else:
            winner = "🏆 RAG" if diff > 0 else "🏆 LLM" if diff < 0 else "🤝 Tie"

        print(
            f"{metric:<25} {rag_score:>6.2f}%        {llm_score:>6.2f}%        {diff:>+6.2f}%       {winner:<15}"
        )

    print("=" * 70)

    return {"rag_averages": rag_avg, "llm_averages": llm_avg}


if __name__ == "__main__":
    # This will be run separately, not automatically
    print("This module should be imported and used in a separate evaluation script.")
    print("See: notebooks/03_evaluation.ipynb or scripts/run_evaluation.py")
