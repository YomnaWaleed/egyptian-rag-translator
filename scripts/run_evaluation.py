# scripts/run_evaluation.py
"""
Standalone script to run complete RAG vs LLM-only evaluation
"""
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.evaluation.evaluate_rag import (
    RAGEvaluator,
    LLMOnlyEvaluator,
    compare_rag_vs_llm,
)


def main():
    print("\n" + "=" * 70)
    print("🏛️ Egyptian RAG Translator - Complete Evaluation")
    print("=" * 70)

    # Load test set
    print("\n📥 Loading test set...")
    test_path = f"{settings.DATA_PROCESSED_PATH}/tla_test.csv"

    if not os.path.exists(test_path):
        print(f"❌ Test set not found: {test_path}")
        print("   Please run setup.py first!")
        sys.exit(1)

    df_test = pd.read_csv(test_path)

    # Option to limit samples for quick testing
    MAX_SAMPLES = 25  # Change to None for full evaluation

    if MAX_SAMPLES:
        print(f"⚠️  Running evaluation on first {MAX_SAMPLES} samples (for testing)")
        print(f"   Set MAX_SAMPLES = None in this script for full evaluation")
    else:
        print(f"✅ Running full evaluation on {len(df_test)} samples")

    # ========================================================================
    # STEP 1: Evaluate RAG System
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Evaluating RAG System")
    print("=" * 70)

    rag_evaluator = RAGEvaluator()
    rag_results_df, rag_metrics_df, rag_failed = rag_evaluator.evaluate_test_set(
        df_test, max_samples=MAX_SAMPLES
    )

    # Save RAG results
    rag_results_path = f"{settings.DATA_PROCESSED_PATH}/rag_evaluation_results.csv"
    rag_metrics_path = f"{settings.DATA_PROCESSED_PATH}/rag_evaluation_metrics.csv"

    rag_results_df.to_csv(rag_results_path, index=False)
    rag_metrics_df.to_csv(rag_metrics_path, index=False)

    print(f"\n💾 RAG results saved:")
    print(f"   {rag_results_path}")
    print(f"   {rag_metrics_path}")

    # ========================================================================
    # STEP 2: Evaluate LLM-Only System
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Evaluating LLM-Only System")
    print("=" * 70)

    llm_evaluator = LLMOnlyEvaluator()
    llm_results_df, llm_metrics_df, llm_failed = llm_evaluator.evaluate_test_set(
        df_test, max_samples=MAX_SAMPLES
    )

    # Save LLM-only results
    llm_results_path = f"{settings.DATA_PROCESSED_PATH}/llm_only_evaluation_results.csv"
    llm_metrics_path = f"{settings.DATA_PROCESSED_PATH}/llm_only_evaluation_metrics.csv"

    llm_results_df.to_csv(llm_results_path, index=False)
    llm_metrics_df.to_csv(llm_metrics_path, index=False)

    print(f"\n💾 LLM-only results saved:")
    print(f"   {llm_results_path}")
    print(f"   {llm_metrics_path}")

    # ========================================================================
    # STEP 3: Compare RAG vs LLM-Only
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Comparing RAG vs LLM-Only")
    print("=" * 70)

    comparison = compare_rag_vs_llm(rag_metrics_df, llm_metrics_df)

    # Save comparison summary
    comparison_df = pd.DataFrame(
        [
            {"System": "RAG", **comparison["rag_averages"]},
            {"System": "LLM-Only", **comparison["llm_averages"]},
            {
                "System": "Difference (RAG - LLM)",
                **{
                    k: comparison["rag_averages"][k] - comparison["llm_averages"][k]
                    for k in comparison["rag_averages"].keys()
                },
            },
        ]
    )

    comparison_path = f"{settings.DATA_PROCESSED_PATH}/comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\n💾 Comparison summary saved:")
    print(f"   {comparison_path}")

    # ========================================================================
    # STEP 4: Print Final Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)

    rag_avg = comparison["rag_averages"]
    llm_avg = comparison["llm_averages"]

    print(
        f"""
📊 Summary ({len(rag_metrics_df)} samples evaluated):

RAG System:
  • BLEU:         {rag_avg['bleu']:.2f}%
  • METEOR:       {rag_avg['meteor']:.2f}%
  • chrF:         {rag_avg['chrf']:.2f}%
  • Recall@10:    {rag_avg['recall@10']:.2f}%
  
LLM-Only System:
  • BLEU:         {llm_avg['bleu']:.2f}%
  • METEOR:       {llm_avg['meteor']:.2f}%
  • chrF:         {llm_avg['chrf']:.2f}%

Improvement (RAG - LLM):
  • BLEU:    {rag_avg['bleu'] - llm_avg['bleu']:+.2f}%
  • METEOR:  {rag_avg['meteor'] - llm_avg['meteor']:+.2f}%
  • chrF:    {rag_avg['chrf'] - llm_avg['chrf']:+.2f}%

📁 All results saved to: {settings.DATA_PROCESSED_PATH}/
"""
    )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
