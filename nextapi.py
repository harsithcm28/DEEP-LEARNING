import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

# ======================= PAGE CONFIG =======================
st.set_page_config(page_title="SpeechGuard", page_icon="🕵️‍♂️", layout="centered")

# ======================= SIDEBAR NAVIGATION =======================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "About", "Future Plans"]
)

# ======================= COMMON INFO (appears on sidebar bottom) =======================
st.sidebar.markdown("---")
st.sidebar.info(
    """
    🧠 **Toxic Comment Detector**  
    Detects multiple toxic behaviors in comments using a fine-tuned BERT model.
    - toxic  
    - severe toxic  
    - obscene  
    - threat  
    - insult  
    - identity hate  
    """
)

# ======================= MODEL LOADING =======================
@st.cache_resource
def load_model():
    MODEL_NAME = "unitary/toxic-bert"  # public model for deployment
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ======================= PREDICTION FUNCTION =======================
def predict_toxicity(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
    results = []
    for i, text in enumerate(text_list):
        label_probs = [(LABELS[j], probs[i][j]) for j in range(len(LABELS)) if probs[i][j] > 0.5]
        results.append(label_probs)
    return results

# ======================= PAGE 1: HOME =======================
if page == "Home":
    st.title("🕵️‍♂️ SpeechGuard ✅❌")

    st.markdown(
        """
        Welcome to **SpeechGuard** — a deep learning–powered tool that detects toxic or harmful language in text comments.  
        Type your comment below or choose an example to see real-time predictions.
        """
    )

    example_comments = [
        "You are so stupid!",
        "I hope you get well soon, friend.",
        "I will hurt you!",
        "That was a nice effort.",
        "You smell terrible!",
    ]

    st.markdown("### 💬 Try example comments:")
    cols = st.columns(len(example_comments))
    for i, example in enumerate(example_comments):
        if cols[i].button(example[:20] + "..."):
            st.session_state.comment = example

    comment = st.text_area(
        "🗣️ Enter a comment to analyze:",
        value=st.session_state.get('comment', ''),
        max_chars=500,
        height=100
    )

    if st.button("🔍 Classify"):
        if comment.strip():
            with st.spinner("Analyzing comment..."):
                result = predict_toxicity([comment])[0]
                if result:
                    st.success("**Predicted Labels:**")
                    for label, confidence in result:
                        confidence_out_of_10 = confidence * 10
                        st.markdown(
                            f"<span style='background-color:#f87171;padding:4px 8px; border-radius:4px;"
                            f"color:white; font-weight:bold;'>{label} ({confidence_out_of_10:.1f}/10)</span>",
                            unsafe_allow_html=True)
                else:
                    st.info("✅ No toxic behavior detected.")
        else:
            st.warning("⚠️ Please enter a comment before classifying.")

    if st.button("🧹 Clear"):
        st.session_state.comment = ""

# ======================= PAGE 2: ABOUT =======================
elif page == "About":
    st.title("📘 About SpeechGuard")
    st.markdown(
        """
        ### 🔍 What is SpeechGuard?
        **SpeechGuard** is a deep learning–based web app that identifies and classifies toxic or harmful text 
        using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model.  
        It detects multiple categories of online toxicity, including:
        - 💢 Toxic  
        - 😠 Severe Toxic  
        - 🚫 Obscene  
        - ⚠️ Threat  
        - 🤬 Insult  
        - 🧑‍🤝‍🧑 Identity Hate  

        ### 💡 Objective
        To ensure a **safe digital communication environment** by filtering harmful comments 
        while maintaining freedom of expression.

        ### 🧠 How It Works
        - Preprocesses input text  
        - Uses a Transformer model to predict toxicity levels  
        - Outputs category-specific confidence scores  

        ---
        **Tech Stack Used:**
        - Python 🐍  
        - PyTorch ⚡  
        - Hugging Face Transformers 🤗  
        - Streamlit 🖥️  
        """
    )

# ======================= PAGE 3: FUTURE PLANS =======================
elif page == "Future Plans":
    st.title("🚀 Future Plans for SpeechGuard")
    st.markdown(
        """
        ### 🔮 Planned Enhancements
        - **Multilingual Toxicity Detection:**  
          Extend SpeechGuard to detect toxicity in multiple languages using multilingual BERT.
        
        - **Explainable AI (XAI):**  
          Add interpretability with tools like LIME or SHAP to show key words influencing toxicity.
        
        - **Speech & Voice Input:**  
          Integrate speech-to-text for real-time moderation of voice chat platforms.
        
        - **API Integration:**  
          Deploy SpeechGuard as a REST API for integration into social media and chat systems.
        
        - **Continuous Learning:**  
          Use user feedback loops to refine model predictions and adapt to evolving language.

        ### 🌍 Long-Term Vision
        To evolve into **SpeechGuard AI Suite**, a robust NLP platform 
        capable of detecting **toxicity, hate speech, bias, and sentiment** 
        — enabling responsible and safe communication globally.
        """
    )

# ======================= END OF APP =======================
