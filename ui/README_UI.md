# 🖥️ Egyptian RAG Translator - UI Guide

Two beautiful user interfaces for the Egyptian RAG Translator!

## 🎨 Available UIs

### 1. Streamlit UI (Recommended)
- Modern, clean interface
- Tabbed Egyptian keyboard
- Real-time updates
- Setup integration

### 2. Gradio UI
- Alternative interface
- Accordion-style keyboard
- Built-in examples
- Easy sharing

## 🚀 Quick Start

### Running Gradio UI

```bash
# From project root
python ui/app_gradio.py

# Or from ui folder
cd ui
python app_gradio.py
```

The app will open in your browser at `http://localhost:7860`

## 📖 How to Use

### First Time Setup

1. **Open the UI** (Gradio)
2. **Go to Setup tab** (Gradio Setup tab)
3. **Click "Run Setup"** button
4. **Wait ~30-40 minutes** for:
   - Dataset download
   - Data processing
   - Embedding generation
   - Database building

The setup script is smart - it won't re-do completed steps if you run it again!

### Translation Workflow

1. **Enter Egyptian Text:**
   - Type directly in the text box
   - Use the on-screen keyboard
   - Click example words

2. **Click Translate:**
   - System processes your text
   - Retrieves similar examples
   - Generates translations

3. **View Results:**
   - 🏛️ Normalized Egyptian
   - 🇩🇪 German translation
   - 🇬🇧 English translation
   - 🔍 Retrieved examples (optional)

## ⌨️ Egyptian Keyboard

The keyboard has 4 sections:

### Consonants (Basic)
All fundamental Egyptian consonants:
```
ꜣ  ꜥ  ʾ  ʿ  j  y  w
b  p  f  m  n
r  h  ḥ  ḫ  ẖ
s  š  z
k  g  t  ṯ
d  ḏ  i̯
```

### Diacritics (Important)
Special marked letters:
```
ḥ  ḫ  ẖ
ṯ  ḏ
š
ꜣ  ꜥ
```

### Symbols
Punctuation and brackets:
```
.  -  =
(  )  [  ]
<  >
```

### Common Words
Frequently used Egyptian words:
```
ḥtp    (offering)
dj     (give)
njswt  (king)
ꜥnḫ   (life)
ḏt     (eternity)
nb     (lord)
tꜣwy   (two lands)
```

## 📝 Examples to Try

Click these in the UI or type them:

1. **ḥtp dj njswt**
   - Translation: "An offering which the king gives"
   - Common offering formula

2. **ꜥnḫ ḏt**
   - Translation: "Living forever"
   - Eternity phrase

3. **nb tꜣwy**
   - Translation: "Lord of the Two Lands"
   - Royal title

## 🔧 Configuration

Both UIs automatically use settings from `src/config/settings.py`:

```python
LLM_MODEL = "qwen3-vl:235b-instruct-cloud"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K_RESULTS = 30
```

Make sure your `.env` file has:
```bash
OLLAMA_API_KEY=your_api_key_here
```

## 🐛 Troubleshooting

### "System not ready"
**Solution:** Run setup from the Setup tab/sidebar

### "Failed to initialize translator"
**Solutions:**
- Check if setup completed successfully
- Verify all data files exist
- Check Qdrant database is built

### "Translation failed"
**Solutions:**
- Verify internet connection (for LLM API)
- Check OLLAMA_API_KEY in .env file
- Try re-initializing the translator

### "Setup takes too long"
**This is normal!** 
- Embedding generation: ~30 minutes
- Total setup: ~30-40 minutes
- You can close and restart - it skips completed steps

### Keyboard characters not working
**Solutions:**
- Try typing directly in the text box
- Copy-paste from examples
- Check browser compatibility (use Chrome/Firefox)

## 💡 Tips

1. **Use Common Words section** for quick input of frequent phrases
2. **Save time** by clicking example sentences first
3. **View Retrieved Examples** to understand how RAG works
4. **Setup once** - the system remembers everything
5. **Try both UIs** - use whichever you prefer!

## 🎨 Customization

### Gradio

Edit theme in `create_ui()`:
```python
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="YOUR_COLOR")
) as app:
```

## 📊 System Requirements

- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 5GB free space
- **Internet:** Required for LLM API calls
- **Browser:** Chrome, Firefox, Safari, Edge

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/egyptian-rag-translator/issues)
- **Email:** yomnawaleed2023@gmail.com
- **Documentation:** See main README.md and DEVELOPER.md

## 📄 License

Same as main project (MIT License)

---

**Enjoy translating Ancient Egyptian! 🏛️**