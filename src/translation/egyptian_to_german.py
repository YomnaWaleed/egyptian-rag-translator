# src/ translation/egyptian_to_german.py

"""
Egyptian to German translation using LLM and RAG
"""
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_templates import (
    build_translation_prompt,
    extract_german_translation,
)


class EgyptianToGermanTranslator:
    """Translate Egyptian to German using LLM with RAG"""

    def __init__(self, ollama_client=None):
        """
        Initialize translator

        Args:
            ollama_client: OllamaClient instance (creates new if None)
        """
        self.ollama_client = ollama_client or OllamaClient()
        print("✅ Egyptian→German translator initialized")

    def translate(self, query_original, query_normalized, retrieved_examples):
        """
        Translate Egyptian to German

        Args:
            query_original: Original Egyptian transliteration
            query_normalized: Normalized transliteration
            retrieved_examples: Retrieved examples from RAG

        Returns:
            tuple: (german_translation, full_llm_output)
        """
        # Build prompt
        system_prompt, user_prompt = build_translation_prompt(
            query_original, query_normalized, retrieved_examples
        )

        # Call LLM
        llm_output, full_response = self.ollama_client.generate(
            system_prompt, user_prompt
        )

        if not llm_output:
            return None, None

        # Extract German translation
        german_translation = extract_german_translation(llm_output)

        return german_translation, llm_output


def translate_egyptian_to_german(
    query_original, query_normalized, retrieved_examples, ollama_client=None
):
    """
    Convenience function for translation

    Args:
        query_original: Original Egyptian transliteration
        query_normalized: Normalized transliteration
        retrieved_examples: Retrieved examples from RAG
        ollama_client: Optional OllamaClient instance

    Returns:
        tuple: (german_translation, full_llm_output)
    """
    translator = EgyptianToGermanTranslator(ollama_client)
    return translator.translate(query_original, query_normalized, retrieved_examples)


if __name__ == "__main__":
    # Test translation
    from src.config import settings

    query_original = "ḥtp dj njswt"
    query_normalized = "htp dj njswt"

    retrieved_examples = [
        {
            "payload": {
                "transliteration_original": "ḥtp dj njswt",
                "transliteration_normalized": "htp dj njswt",
                "lemmas": ["ḥtp", "dj", "njswt"],
                "UPOS": "NOUN VERB NOUN",
                "glossing": "offering give king",
                "translation_de": "Ein Opfer, das der König gibt",
            }
        }
    ]

    german, llm_output = translate_egyptian_to_german(
        query_original, query_normalized, retrieved_examples
    )

    if german:
        print(f"\n✅ German Translation: {german}")
    else:
        print(f"\n❌ Translation failed")
