# src/ preprocessing/ split.py
"""
Train/test splitting functionality
"""
from src.config import settings


def split_train_test(df):
    """
    Split dataset into train and test sets

    Args:
        df: Input DataFrame

    Returns:
        tuple: (df_train, df_test)
    """
    print("\n" + "=" * 70)
    print(
        f"📊 Creating train/test split ({settings.TRAIN_SPLIT*100}%/{(1-settings.TRAIN_SPLIT)*100}%)"
    )
    print("=" * 70)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    split_idx = int(len(df) * settings.TRAIN_SPLIT)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"✅ Split complete!")
    print(
        f"   Training set: {len(df_train)} records ({len(df_train)/len(df)*100:.1f}%)"
    )
    print(f"   Test set:     {len(df_test)} records ({len(df_test)/len(df)*100:.1f}%)")

    return df_train, df_test


def save_splits(df_train, df_test):
    """
    Save train and test splits to CSV

    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
    """
    train_path = f"{settings.DATA_PROCESSED_PATH}/tla_train.csv"
    test_path = f"{settings.DATA_PROCESSED_PATH}/tla_test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\n💾 Train set saved to: {train_path}")
    print(f"💾 Test set saved to: {test_path}")


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings

    # Load processed data
    df = pd.read_csv(f"{settings.DATA_PROCESSED_PATH}/tla_with_lemmas.csv")

    # Split
    df_train, df_test = split_train_test(df)

    # Save
    save_splits(df_train, df_test)
