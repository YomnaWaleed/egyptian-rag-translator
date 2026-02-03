# src/ translation/german_to_english.py

"""
German to English translation using MarianMT
"""
from transformers import MarianMTModel, MarianTokenizer


class GermanToEnglishTranslator:
    """Translate German to English using MarianMT"""

    def __init__(self, model_name="Helsinki-NLP/opus-mt-de-en"):
        """
        Initialize translator

        Args:
            model_name: MarianMT model name
        """
        print("\n" + "=" * 70)
        print("📥 Loading German→English translation model")
        print("=" * 70)
        print(f"   Model: {model_name}")

        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        print(f"✅ Model loaded successfully")

    def translate(self, german_text):
        """
        Translate German to English

        Args:
            german_text: German text to translate

        Returns:
            str: English translation
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                german_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Generate translation
            outputs = self.model.generate(**inputs)

            # Decode
            english_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return english_text

        except Exception as e:
            print(f"❌ Translation error: {e}")
            return None


def translate_german_to_english(german_text, translator=None):
    """
    Convenience function for translation

    Args:
        german_text: German text to translate
        translator: Optional GermanToEnglishTranslator instance

    Returns:
        str: English translation
    """
    if translator is None:
        translator = GermanToEnglishTranslator()

    return translator.translate(german_text)


if __name__ == "__main__":
    # Test translation
    translator = GermanToEnglishTranslator()

    german_text = "Ein Opfer, das der König gibt"
    english_text = translator.translate(german_text)

    print(f"\n🇩🇪 German:  {german_text}")
    print(f"🇬🇧 English: {english_text}")
