# src/ preprocessing/ lemmatization.py
"""
Lemma extraction functionality
"""


def extract_lemmas(lemmatization_text):
    """
    Extract lemma words from lemmatization field

    Args:
        lemmatization_text: Lemmatization string from dataset

    Returns:
        list: List of lemma words
    """
    if not isinstance(lemmatization_text, str):
        return []

    lemmas = []
    parts = lemmatization_text.split()

    for part in parts:
        if "|" in part:
            lemma_id, lemma_word = part.split("|", 1)
            # Skip suffixes/particles
            if not lemma_word.startswith("="):
                lemmas.append(lemma_word)

    return lemmas


def extract_lemmas_from_dataset(df):
    """
    Extract lemmas for entire dataset

    Args:
        df: DataFrame with 'lemmatization' column

    Returns:
        pd.DataFrame: DataFrame with 'lemmas' column added
    """
    print("\n" + "=" * 70)
    print("📝 Extracting lemmas")
    print("=" * 70)

    df["lemmas"] = df["lemmatization"].apply(extract_lemmas)

    print(f"✅ Lemma extraction complete!")
    print(f"\n📋 Sample lemmas:")
    for i in range(min(3, len(df))):
        lemmas = df.iloc[i]["lemmas"]
        print(f"   {i+1}. {lemmas[:5]}")

    return df


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings

    # Test lemma extraction
    df = pd.read_csv(f"{settings.DATA_PROCESSED_PATH}/tla_normalized.csv")
    df_with_lemmas = extract_lemmas_from_dataset(df)

    output_path = f"{settings.DATA_PROCESSED_PATH}/tla_with_lemmas.csv"
    df_with_lemmas.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
