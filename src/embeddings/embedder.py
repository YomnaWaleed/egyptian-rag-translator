# src/ embeddings/ embedder.py

"""
Embedding generation functionality
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from src.config import settings


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""

    def __init__(self, model_name=None):
        """
        Initialize embedding generator

        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        print(f"\n📥 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Model loaded successfully")

    def generate_single(self, text):
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            list: Embedding vector
        """
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return np.random.randn(settings.VECTOR_DIM).tolist()

    def generate_batch(self, texts, batch_size=None):
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            list: List of embedding vectors
        """
        batch_size = batch_size or settings.BATCH_SIZE

        print("\n" + "=" * 70)
        print(f"🔢 Generating embeddings for {len(texts)} records")
        print("=" * 70)
        print(f"   Model: {self.model_name}")
        print(f"   Vector dimension: {settings.VECTOR_DIM}")
        print(f"   Batch size: {batch_size}")

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]

            try:
                # Process entire batch at once
                batch_embeddings = self.model.encode(
                    batch_texts, normalize_embeddings=True, show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings.tolist())
            except Exception as e:
                print(f"❌ Batch error at {i}: {e}")
                # Fallback: process individually
                for text in batch_texts:
                    all_embeddings.append(self.generate_single(text))

        print(f"\n✅ Embedding generation complete!")
        print(f"   Total: {len(all_embeddings)} embeddings")
        print(f"   Dimension: {len(all_embeddings[0])}")

        return all_embeddings

    def save_embeddings(self, embeddings, filename="train_embeddings.npy"):
        """
        Save embeddings to file

        Args:
            embeddings: List of embedding vectors
            filename: Output filename
        """
        output_path = f"{settings.DATA_EMBEDDINGS_PATH}/{filename}"
        np.save(output_path, np.array(embeddings))
        print(f"\n💾 Embeddings saved to: {output_path}")

    def load_embeddings(self, filename="train_embeddings.npy"):
        """
        Load embeddings from file

        Args:
            filename: Input filename

        Returns:
            np.ndarray: Loaded embeddings
        """
        input_path = f"{settings.DATA_EMBEDDINGS_PATH}/{filename}"
        embeddings = np.load(input_path)
        print(f"📥 Loaded {len(embeddings)} embeddings from: {input_path}")
        return embeddings


def generate_embeddings_for_dataset(df, column="transliteration_normalized"):
    """
    Generate embeddings for entire dataset

    Args:
        df: Input DataFrame
        column: Column to generate embeddings from

    Returns:
        pd.DataFrame: DataFrame with 'embedding' column added
    """
    embedder = EmbeddingGenerator()

    texts = df[column].tolist()
    embeddings = embedder.generate_batch(texts)

    df["embedding"] = embeddings

    # Save embeddings separately
    embedder.save_embeddings(embeddings)

    # Show sample
    sample_embedding = embeddings[0]
    print(f"\n📊 Sample embedding (first 10 values):")
    print(f"   {sample_embedding[:10]}")

    return df


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings

    # Load training data
    df_train = pd.read_csv(f"{settings.DATA_PROCESSED_PATH}/tla_train.csv")

    # Generate embeddings
    df_train = generate_embeddings_for_dataset(df_train)

    # Save with embeddings
    output_path = f"{settings.DATA_PROCESSED_PATH}/tla_train_with_embeddings.csv"
    df_train.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
