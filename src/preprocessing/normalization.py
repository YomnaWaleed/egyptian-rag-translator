# src/ preprocessing/ normalization.py
"""
Transliteration normalization functionality
"""
import re
import unicodedata
from src.config import settings


def normalize_transliteration(text):
    """
    Normalize Egyptian transliteration:
    1. Remove brackets
    2. Lowercase
    3. Map special characters
    4. Remove suffixes
    5. Clean spaces

    Args:
        text: Input transliteration string

    Returns:
        str: Normalized transliteration
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Step 1: Remove brackets (but keep content)
    text = re.sub(r"[()]", "", text)

    # Step 2: Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove combining marks (important for di̯, etc.)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Step 3: Lowercase
    text = text.lower()

    # Step 4: Map Egyptian characters
    for egy_char, normalized in settings.EGYPTIAN_CHAR_MAP.items():
        text = text.replace(egy_char.lower(), normalized)

    # Step 5: Remove suffixes (pronouns/particles)
    for suffix in settings.SUFFIXES_TO_REMOVE:
        # Match suffix at word boundaries or before spaces/dots
        pattern = re.escape(suffix) + r"(?=[\s\.]|$)"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Step 6: Clean up extra spaces and dots
    text = re.sub(r"\.+", ".", text)  # Multiple dots to single
    text = re.sub(r"\s+", " ", text)  # Multiple spaces to single
    text = text.strip(". ")  # Remove leading/trailing dots and spaces

    return text


def normalize_dataset(df):
    """
    Apply normalization to entire dataset

    Args:
        df: Input DataFrame with 'transliteration' column

    Returns:
        pd.DataFrame: DataFrame with 'transliteration_normalized' column
    """
    print("\n" + "=" * 70)
    print("🔤 Normalizing transliterations")
    print("=" * 70)

    # Test normalization on sample
    sample_text = df.iloc[0]["transliteration"]
    normalized_sample = normalize_transliteration(sample_text)

    print(f"\n📝 Sample normalization:")
    print(f"   Original:   {sample_text}")
    print(f"   Normalized: {normalized_sample}")

    # Apply normalization to entire dataset
    print(f"\n🔄 Normalizing {len(df)} transliterations...")

    df["transliteration_normalized"] = df["transliteration"].apply(
        normalize_transliteration
    )

    # Remove empty normalizations
    df = df[df["transliteration_normalized"].str.len() > 0]
    df = df.reset_index(drop=True)

    print(f"✅ Normalization complete!")
    print(f"   Valid records: {len(df)}")

    # Show more examples
    print(f"\n📋 Sample normalizations:")
    for i in range(min(5, len(df))):
        orig = df.iloc[i]["transliteration"]
        norm = df.iloc[i]["transliteration_normalized"]
        print(f"   {i+1}. {orig[:40]:40} → {norm[:40]}")

    return df


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings

    # Test normalization
    df = pd.read_csv(f"{settings.DATA_RAW_PATH}/tla_original.csv")
    df_normalized = normalize_dataset(df)

    output_path = f"{settings.DATA_PROCESSED_PATH}/tla_normalized.csv"
    df_normalized.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
