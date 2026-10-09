import os
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

    class _NoOpSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            # Support both @spaces.GPU and @spaces.GPU(duration=...) usage
            if fn is not None:
                return fn
            return lambda f: f

    spaces = _NoOpSpaces()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# --- Configuration ---
MODEL_DIR = os.environ.get("MODEL_DIR", "tedlai/amharic-news-classifier")
MAX_LENGTH = 256

EXAMPLE_ARTICLES = [
    "የኢትዮጵያ ብሔራዊ ቡድን በአፍሪካ ዋንጫ ማጣሪያ ጨዋታ ድል አስመዝግቧል።",
    "የገንዘብ ሚኒስቴር አዲስ የበጀት ፖሊሲ ለፓርላማ አቅርቧል።",
    "በብዙዎች ዘንድ በተለይ የሚታወቀው ቦክስ ጊታሩን ይዞ ለየት ባለው ቅላፄው በሚጫወታቸው ዘፈኖቹ ነው ድምፃዊው "
    "መስፍን አበበ የሙዚቃ መሳሪያ እየተጫወቱ ከሚያቀነቅኑ ግንባር ቀደምት መካከል ቀድሞ ለታዳሚ የሚመጣው መስፍን "
    "ነውያዢልኝ ቀጠሮ ቦታ ምረጪናወይ ሀረር ወይ ሸገር አጫውቺኝ ነይና እያለ ከምድር ባቡር ጋር አስተሳስሮ በሚጫወተው "
    "ዜማው እንዲሁም አይሽ አይሽና ይመጣል እንባዬ እንዲያው ጉም ጉም መልካም ልደት ተጠቃሾች ናቸው በደሴ ወይዘሮ ስሂን "
    "ትምህርት ቤት የሙዚቃ ባንድ ጋር በመሆንና ሙዚቃ በተለያዩ የአገሪቱ አካባቢዎች በመጫወት ወደ ጥበቡ የዘለቀው ድምፃዊና "
    "ጊታሪስት መስፍን በተለያዩ ሙዚቃ ቤቶች 22 ካሴቶች ማሳተሙ ይነገርለታልመስፍን ህይወቱን የመንፈስ አረቄ የነፍስ እህል "
    "ውሀ እየተባለች ከሚነገርላት ሙዚቃ ጋር ብቻ አላቆራኘም የአድማጮችን ሀሊና የሚኮረኩሩ ብሂልን ከባህል ያዛመዱ ነውን "
    "ከነበር የሚያጣቅሱ ስራዎቹን በሙዚቃው ብቻ ማስተላለፍ አልተወሰነም የሁለተኛ ደረጃ ትምህርቱን ካጠናቀቀ በኋላ ትውልድን "
    "ለማነፅ እንደ አባቶቹ አርአያነት ያለውን የአስተማሪነት ሙያን በመፍቀድና ከመምህራን ትምህርት ኮሌጅ በመዝለቅ ሙያውን "
    "ጨብጧል ሲያስተምር ሲያዜም ጊታር ሲጫወት ኖሯልአስተማሪነቱ የቀለም ትምህርት ብቻ አልነበረም በዘመኑ ወጣት ከነበሩ "
    "አሁን አንጋፋ ድምፃውያን እውቀቱን አልነፈጋቸውም አንዱ ማሳያ የታዋቂው ድምፃዊ ንዋይ ደበበ ምስክርነት ነው በኢትዮጵያ "
    "ሙዚቀኞች ማሀበር ድረ ገፅ ላይ እንደተመለከተው ድምፃዊ ንዋይ ደበበ ዜማዎችን ሲፈጥር በፉጨትና በእንጉርጉሮ ሲሆን ከአንድ "
    "ወቅት በኋላ በቦክስ ጊታር እየተጫወተ የሚያቀነቅነውን ዝነኛ ድምፃዊ መስፍን አበበ ለስራ ወላይታ በመጣ ጊዜ የማግኘት "
    "እድሉ አጋጥሞት ለዜማ መለማመጃ የቦክስ ጊታር አጨዋወት እንዳስተማረው ይታወቃልከአባቱ አቶ አበበ ወልደ ማርያምና "
    "ከእናቱ ከወሮ ወይንሸት ተሾመ በ1944 አም የተወለደው መስፍን አበበ በአስራዎቹ አመታት በተማረው ሙያ በድምፃዊነትና "
    "በሙዚቃ መሳሪያ ተጫዋችነቱ እንዲሁም በመምህርነቱ ይታወሳልባደረበት ድንገተኛ ህመም ግንቦት 25 ቀን 2008 አም በ64 "
    "አመቱ ያረፈው ድምፃዊና መምህር መስፍን ስርአተ ቀብሩ በማግስቱ በብርሀናተ አለም ቅዱስ ጴጥሮስ ወጳውሎስ ቤተ ክርስቲያን "
    "እኩለ ቀን ላይ ተፈፅሟልድምፃዊ ሙዚቀኛና መምህር መስፍን ባለትዳርና የአንዲት ሴት ልጅ አባት ነበር",
]


def load_model():
    print(f"Loading model from: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except Exception as e:
        raise RuntimeError(
            f"Could not load a model from '{MODEL_DIR}'. This should be "
            f"either a local checkpoint directory (see "
            f"src/finetune_transformer.py to train one) or a valid Hugging "
            f"Face Hub repo id. Original error: {e}"
        ) from e
    model.eval()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    print(f"Model loaded on device: {device}")
    print(f"Label set: {model.config.id2label}")
    return tokenizer, model, device


tokenizer, model, device = load_model()


@spaces.GPU
def classify(text: str):
    if not text or not text.strip():
        return {}, "Enter some Amharic text above to classify."

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_LENGTH, padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    id2label = model.config.id2label
    label_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    # Sort descending for a cleaner display
    label_probs = dict(sorted(label_probs.items(), key=lambda kv: -kv[1]))

    top_label = max(label_probs, key=label_probs.get)
    top_conf = label_probs[top_label]
    status = f"Predicted category: **{top_label}** (confidence: {top_conf:.1%})"

    return label_probs, status


with gr.Blocks(title="Amharic News Classifier") as demo:
    gr.Markdown(
        "# Amharic News Topic Classifier\n"
        "Paste an Amharic news article or headline below to predict its "
        "topic category. Fine-tuned from "
        "[rasyosef/bert-medium-amharic](https://huggingface.co/rasyosef/bert-medium-amharic) "
        "on the "
        "[Amharic News Text Classification Dataset](https://huggingface.co/datasets/rasyosef/amharic-news-category-classification) "
        "(Azime & Mohammed, 2021)."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Amharic text",
                placeholder="የአማርኛ ዜና ጽሑፍ እዚህ ያስገቡ...",
                lines=8,
            )
            submit_btn = gr.Button("Classify", variant="primary")
            gr.Examples(
                examples=EXAMPLE_ARTICLES,
                inputs=text_input,
                label="Try an example",
            )

        with gr.Column():
            label_output = gr.Label(label="Predicted category probabilities", num_top_classes=6)
            status_output = gr.Markdown()

    submit_btn.click(fn=classify, inputs=text_input, outputs=[label_output, status_output])
    text_input.submit(fn=classify, inputs=text_input, outputs=[label_output, status_output])

    gr.Markdown(
        "---\n"
        "**Model performance (held-out test set, 10,205 articles):** "
        "87.3% accuracy, 0.833 macro-F1 across 6 categories. "
        "Note: the model performs best on Sport and Entertainment "
        "(near-perfect separability) and is weakest distinguishing "
        "Business from the broader National News category, which overlap "
        "substantially in the source data — see the project report for "
        "full error analysis."
    )

if __name__ == "__main__":
    demo.launch()
