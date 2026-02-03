# src/ llm/prompt_templates.py
"""
Prompt templates for Egyptian-to-German translation
"""


def build_translation_prompt(query_original, query_normalized, retrieved_examples):
    """
    Build prompt for LLM translation

    Args:
        query_original: Original Egyptian transliteration
        query_normalized: Normalized transliteration
        retrieved_examples: List of retrieved examples from database

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Build examples context
    examples_text = ""
    for i, example in enumerate(retrieved_examples, 1):
        payload = example["payload"]

        # Handle lemmas (might be string or list)
        lemmas = payload.get("lemmas", [])
        if isinstance(lemmas, str):
            import ast

            try:
                lemmas = ast.literal_eval(lemmas)
            except:
                lemmas = []

        lemmas_str = ", ".join(lemmas[:5]) if lemmas else "N/A"

        examples_text += f"""
Example {i}:
- Original: {payload['transliteration_original']}
- Normalized: {payload['transliteration_normalized']}
- Lemmas: {lemmas_str}
- POS Tags: {payload.get('UPOS', 'N/A')}
- Glossing: {payload.get('glossing', 'N/A')}
- German: {payload['translation_de']}
---
"""

    # System prompt
    system_prompt = """You are a senior linguist specializing in Earlier Egyptian (Old Egyptian & Early Middle Egyptian),
with strong expertise in morphology, syntax, and historical semantics.

Your task is to translate an Earlier Egyptian transliteration into German
using retrieved linguistic examples ONLY as structural and semantic guidance."""

    # User prompt
    user_prompt = f"""
=====================================
QUERY TO TRANSLATE
=====================================

Normalized Transliteration:
{query_normalized}

=====================================
RETRIEVED DATABASE EXAMPLES
=====================================
{examples_text}

=====================================
INSTRUCTIONS
=====================================
Follow these steps carefully:

1. Linguistic Analysis
   - Identify the grammatical category of each word (verb, noun, particle, suffix, etc.)
   - Detect verb tense/aspect, suffix pronouns, and syntactic order (VSO, SVO, nominal clause).

2. Morphological Alignment
   - Compare suffixes, verb forms, and particles with the retrieved examples.
   - Use lemma meanings as semantic hints, not literal translations.

3. Translation Construction
   - Produce a fluent and historically plausible German translation.
   - Adapt word order to correct German syntax.
   - Prefer linguistically conservative interpretations over speculative ones.

4. Uncertainty Handling
   - If multiple readings are possible, choose the most likely one.
   - Briefly mention ambiguity only if it materially affects meaning.

=====================================
STRICT RULES
=====================================
- DO NOT copy any German translation from the examples.
- DO NOT mention example numbers or quote them.
- DO NOT add explanations unless uncertainty exists.
- DO NOT hallucinate missing words.
- Base your output strictly on Earlier Egyptian grammar.

=====================================
OUTPUT FORMAT (STRICT)
=====================================
German Translation: <one clear German sentence>
Confidence: High | Medium | Low
Notes: <only if confidence is Medium or Low>
"""

    return system_prompt, user_prompt


def extract_german_translation(llm_output):
    """
    Extract German translation from LLM output

    Args:
        llm_output: Full output from LLM

    Returns:
        str: Extracted German translation
    """
    import re

    # Try to extract using regex
    match = re.search(r"German Translation:\s*(.+?)(?:\n|$)", llm_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: return first line
    return llm_output.split("\n")[0].strip()


if __name__ == "__main__":
    # Test prompt building
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

    system_prompt, user_prompt = build_translation_prompt(
        query_original, query_normalized, retrieved_examples
    )

    print("System Prompt:")
    print(system_prompt)
    print("\n" + "=" * 70 + "\n")
    print("User Prompt:")
    print(user_prompt)
