# src/ preprocessing/ loader.py
"""
Dataset loading functionality
"""
from datasets import load_dataset
import pandas as pd
from src.config import settings


def load_tla_dataset():
    """
    Load TLA dataset from HuggingFace

    Returns:
        pd.DataFrame: Loaded dataset as pandas DataFrame
    """
    print("📥 Loading TLA dataset from HuggingFace...")

    dataset = load_dataset(settings.DATASET_NAME, split="train")

    df = pd.DataFrame(dataset)

    print(f"✅ Loaded {len(df)} records")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample record:")
    print(df.iloc[0][["transliteration", "translation", "UPOS"]].to_dict())

    return df


def clean_dataset(df):
    """
    Clean dataset by removing unwanted columns and rows

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print("\n" + "=" * 70)
    print("🧹 Cleaning dataset")
    print("=" * 70)

    # Remove unwanted columns
    print(f"\n1️⃣ Removing columns: {settings.COLUMNS_TO_DROP}")
    df_clean = df.drop(columns=settings.COLUMNS_TO_DROP, errors="ignore")
    print(f"   Remaining columns: {list(df_clean.columns)}")

    # Remove rows with missing critical data
    print("\n2️⃣ Removing rows with missing data")
    initial_count = len(df_clean)

    df_clean = df_clean.dropna(subset=["transliteration", "translation"])
    df_clean = df_clean[df_clean["transliteration"].str.strip() != ""]
    df_clean = df_clean[df_clean["translation"].str.strip() != ""]

    print(f"   Removed {initial_count - len(df_clean)} rows")
    print(f"   Records remaining: {len(df_clean)}")

    # Remove duplicates
    print("\n3️⃣ Removing duplicates")
    initial_count = len(df_clean)

    df_clean = df_clean.drop_duplicates(subset=["transliteration"], keep="first")

    print(f"   Removed {initial_count - len(df_clean)} duplicate records")
    print(f"   Unique records: {len(df_clean)}")

    df_clean = df_clean.reset_index(drop=True)

    print(f"\n✅ Cleaning complete! Final count: {len(df_clean)} records")

    return df_clean


if __name__ == "__main__":
    # Test loading
    df = load_tla_dataset()
    df_clean = clean_dataset(df)

    # Save to raw data
    output_path = f"{settings.DATA_RAW_PATH}/tla_original.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
