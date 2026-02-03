# ui/app_gradio.py
"""
Gradio UI for Egyptian RAG Translator
Features:
- Egyptian transliteration keyboard
- Real-time translation
- German and English outputs
- Setup system integration
"""
import gradio as gr
import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.rag_pipeline import RAGPipeline
from src.config import settings


# Egyptian transliteration characters organized by type
EGYPTIAN_CHARS = {
    "Consonants": [
        "ꜣ",
        "ꜥ",
        "ʾ",
        "ʿ",
        "j",
        "y",
        "w",
        "b",
        "p",
        "f",
        "m",
        "n",
        "r",
        "h",
        "ḥ",
        "ḫ",
        "ẖ",
        "s",
        "š",
        "z",
        "k",
        "g",
        "t",
        "ṯ",
        "d",
        "ḏ",
        "i̯",
    ],
    "Diacritics": ["ḥ", "ḫ", "ẖ", "ṯ", "ḏ", "š", "ꜣ", "ꜥ"],
    "Symbols": [".", "-", "=", "(", ")", "[", "]", "<", ">"],
    "Common Words": ["ḥtp", "dj", "njswt", "ꜥnḫ", "ḏt", "nb", "tꜣwy"],
}


# Global pipeline
pipeline = None


def check_system_ready():
    """Check if system is set up"""
    required_files = [
        f"{settings.DATA_PROCESSED_PATH}/tla_train.csv",
        f"{settings.DATA_EMBEDDINGS_PATH}/train_embeddings.npy",
        f"{settings.DATA_PROCESSED_PATH}/bm25_corpus.pkl",
    ]

    for filepath in required_files:
        if not os.path.exists(filepath):
            return False, f"Missing: {filepath}"

    if not os.path.exists(settings.QDRANT_PATH):
        return False, f"Missing: {settings.QDRANT_PATH}"

    return True, "System ready!"


def run_setup():
    """Run setup.py"""
    try:
        result = subprocess.run(
            [sys.executable, "setup.py"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )

        if result.returncode == 0:
            return "✅ Setup completed successfully!"
        else:
            return f"❌ Setup failed:\n{result.stderr}"
    except Exception as e:
        return f"❌ Error running setup: {str(e)}"


def init_pipeline():
    """Initialize translation pipeline"""
    global pipeline

    if pipeline is None:
        try:
            ready, msg = check_system_ready()
            if not ready:
                return f"❌ System not ready: {msg}"

            pipeline = RAGPipeline(in_memory=False)
            return "✅ Translator initialized successfully!"
        except Exception as e:
            return f"❌ Failed to initialize: {str(e)}"

    return "✅ Translator already initialized!"


def add_char(current_text, char):
    """Add character to current text"""
    return current_text + char


def translate(text):
    """Translate Egyptian text"""
    global pipeline

    if not text or text.strip() == "":
        return ("", "⚠️ Please enter Egyptian transliteration", "", "")

    # Initialize pipeline if needed
    if pipeline is None:
        try:
            pipeline = RAGPipeline(in_memory=False)
        except Exception as e:
            return ("", f"❌ Failed to initialize translator: {str(e)}", "", "")

    # Translate
    try:
        result = pipeline.translate(text, show_details=False)

        if result["success"]:
            # Format top matches
            examples = ""
            for i, match in enumerate(result.get("top_matches", [])[:3], 1):
                examples += (
                    f"\n**Example {i}** (Score: {match.get('rrf_score', 0):.4f})\n"
                )
                examples += (
                    f"- Egyptian: {match['payload']['transliteration_normalized']}\n"
                )
                examples += f"- German: {match['payload']['translation_de']}\n"

            return (
                result["query_normalized"],
                result["german"],
                result["english"],
                examples,
            )
        else:
            return (
                "",
                f"❌ Translation failed: {result.get('error', 'Unknown error')}",
                "",
                "",
            )

    except Exception as e:
        return ("", f"❌ Translation error: {str(e)}", "", "")


def create_ui():
    """Create Gradio interface"""

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="amber"),
        title="Egyptian RAG Translator",
        css="""
        .main-header {
            text-align: center;
            color: #D4AF37;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #8B7355;
            margin-bottom: 2rem;
        }
        .keyboard-row {
            margin: 0.5rem 0;
        }
        """,
    ) as app:

        # Header
        gr.HTML(
            """
            <h1 class="main-header">🏛️ Egyptian RAG Translator</h1>
            <p class="sub-header">Translate Earlier Egyptian Transliterations to English via German</p>
        """
        )

        with gr.Tabs():
            # Translation Tab
            with gr.Tab("🔤 Translation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Input
                        input_text = gr.Textbox(
                            label="Egyptian Transliteration",
                            placeholder="Enter Egyptian transliteration or use keyboard below...",
                            lines=3,
                            max_lines=5,
                        )

                        # Control buttons
                        with gr.Row():
                            translate_btn = gr.Button(
                                "🔄 Translate", variant="primary", scale=2
                            )
                            clear_btn = gr.ClearButton(
                                [input_text], value="🗑️ Clear", scale=1
                            )
                            space_btn = gr.Button("␣ Space", scale=1)

                        # Egyptian Keyboard
                        gr.Markdown("### ⌨️ Egyptian Transliteration Keyboard")

                        # Consonants
                        with gr.Accordion("Consonants (Basic)", open=True):
                            chars = EGYPTIAN_CHARS["Consonants"]
                            for i in range(0, len(chars), 7):
                                with gr.Row():
                                    for char in chars[i : i + 7]:
                                        btn = gr.Button(char, size="sm")
                                        btn.click(
                                            fn=add_char,
                                            inputs=[input_text, gr.State(char)],
                                            outputs=input_text,
                                        )

                        # Diacritics
                        with gr.Accordion("Diacritics (Important)"):
                            chars = EGYPTIAN_CHARS["Diacritics"]
                            with gr.Row():
                                for char in chars:
                                    btn = gr.Button(char, size="sm")
                                    btn.click(
                                        fn=add_char,
                                        inputs=[input_text, gr.State(char)],
                                        outputs=input_text,
                                    )

                        # Symbols
                        with gr.Accordion("Symbols"):
                            chars = EGYPTIAN_CHARS["Symbols"]
                            with gr.Row():
                                for char in chars:
                                    btn = gr.Button(char, size="sm")
                                    btn.click(
                                        fn=add_char,
                                        inputs=[input_text, gr.State(char)],
                                        outputs=input_text,
                                    )

                        # Common Words
                        with gr.Accordion("Common Words"):
                            chars = EGYPTIAN_CHARS["Common Words"]
                            with gr.Row():
                                for char in chars:
                                    btn = gr.Button(char, size="sm")
                                    btn.click(
                                        fn=add_char,
                                        inputs=[input_text, gr.State(char)],
                                        outputs=input_text,
                                    )

                    with gr.Column(scale=2):
                        # Outputs
                        gr.Markdown("### 📋 Translation Results")

                        normalized_output = gr.Textbox(
                            label="🏛️ Egyptian (Normalized)", lines=2, interactive=False
                        )

                        german_output = gr.Textbox(
                            label="🇩🇪 German Translation", lines=3, interactive=False
                        )

                        english_output = gr.Textbox(
                            label="🇬🇧 English Translation", lines=3, interactive=False
                        )

                        # Top matches
                        with gr.Accordion("🔍 Retrieved Examples (Top 3)", open=False):
                            examples_output = gr.Markdown()

                # Space button action
                space_btn.click(
                    fn=add_char, inputs=[input_text, gr.State(" ")], outputs=input_text
                )

                # Translation button action
                translate_btn.click(
                    fn=translate,
                    inputs=input_text,
                    outputs=[
                        normalized_output,
                        german_output,
                        english_output,
                        examples_output,
                    ],
                )

                # Examples
                gr.Markdown("### 📖 Try These Examples")
                with gr.Row():
                    ex1 = gr.Button("ḥtp dj njswt", size="sm")
                    ex2 = gr.Button("ꜥnḫ ḏt", size="sm")
                    ex3 = gr.Button("nb tꜣwy", size="sm")

                ex1.click(lambda: "ḥtp dj njswt", outputs=input_text)
                ex2.click(lambda: "ꜥnḫ ḏt", outputs=input_text)
                ex3.click(lambda: "nb tꜣwy", outputs=input_text)

            # Setup Tab
            with gr.Tab("⚙️ Setup"):
                gr.Markdown(
                    """
                ## System Setup
                
                Before using the translator, you must set up the system.
                This process will:
                1. Download the TLA dataset (~9,000 texts)
                2. Process and clean the data
                3. Generate AI embeddings (~30 minutes)
                4. Build search databases
                """
                )

                with gr.Row():
                    with gr.Column():
                        system_status = gr.Textbox(
                            label="System Status",
                            value="Click 'Check Status' to verify",
                            lines=2,
                            interactive=False,
                        )

                        with gr.Row():
                            check_btn = gr.Button(
                                "🔍 Check Status", variant="secondary"
                            )
                            init_btn = gr.Button(
                                "🚀 Initialize Translator", variant="secondary"
                            )

                        setup_btn = gr.Button(
                            "🔧 Run Setup", variant="primary", size="lg"
                        )

                        setup_output = gr.Textbox(
                            label="Setup Log", lines=10, interactive=False
                        )

                # Button actions
                check_btn.click(
                    fn=lambda: check_system_ready()[1], outputs=system_status
                )

                init_btn.click(fn=init_pipeline, outputs=setup_output)

                setup_btn.click(fn=run_setup, outputs=setup_output)

            # Help Tab
            with gr.Tab("❓ Help"):
                gr.Markdown(
                    """
                ## How to Use
                
                ### 1. Setup (First Time Only)
                - Go to the **Setup** tab
                - Click **Run Setup** button
                - Wait for completion (~30-40 minutes)
                
                ### 2. Translation
                - Go to the **Translation** tab
                - Enter Egyptian text using:
                  - Type directly in the text box
                  - Use the on-screen keyboard
                  - Click example words
                - Click **Translate** button
                - View results in German and English
                
                ### 3. Keyboard Usage
                - **Consonants**: Basic Egyptian letters
                - **Diacritics**: Special marked letters (ḥ, ḫ, ẖ, etc.)
                - **Symbols**: Punctuation and brackets
                - **Common Words**: Frequently used Egyptian words
                
                ### 4. Examples
                Try these common Egyptian phrases:
                - **ḥtp dj njswt**: "An offering which the king gives"
                - **ꜥnḫ ḏt**: "Living forever"
                - **nb tꜣwy**: "Lord of the Two Lands"
                
                ## System Information
                - **LLM Model**: qwen3-vl:235b-instruct-cloud
                - **Embedding**: BAAI/bge-m3
                - **Database**: 9,000+ Egyptian texts
                - **Languages**: Egyptian → German → English
                
                ## Technical Details
                This system uses:
                - **RAG (Retrieval-Augmented Generation)**: Finds similar texts in database
                - **Hybrid Search**: Combines semantic and keyword matching
                - **LLM Translation**: AI-powered contextual translation
                - **Neural Translation**: German to English conversion
                
                ## Troubleshooting
                
                **"System not ready"**
                - Run setup from the Setup tab
                
                **"Translation failed"**
                - Check your internet connection (for LLM API)
                - Verify API key is set in .env file
                - Re-initialize the translator
                
                **"Setup takes too long"**
                - Normal! Embedding generation takes 30+ minutes
                - You can close and re-run setup - it skips completed steps
                
                ## Support
                - GitHub: [Your Repository]
                - Email: [Your Email]
                """
                )

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; margin-top: 2rem; color: #888;">
                <p>Egyptian RAG Translator v1.0 | Built with Gradio</p>
                <p>Using TLA Dataset & Qwen 3 VL LLM</p>
            </div>
        """
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
