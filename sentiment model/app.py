# app.py - Sentiment Analysis Web Application
# Place this file in the SAME folder as your model files:
# model.safetensors, config.json, tokenizer.json, tokenizer_config.json,
# vocab.txt, label_mappings.json, model_info.json

import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.html("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .neutral {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""")

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== MODEL LOADING ==========
def load_model():
    """Load the fine-tuned model and tokenizer from the same folder as app.py"""
    try:
        # All files are in the same folder as app.py
        model_path = BASE_DIR
        
        # === 🚀 ADDED CODE: AUTOMATICALLY FETCH MASSIVE MODEL ON STARTUP ===
        # This downloads model.safetensors straight from your free Hugging Face repo
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="NajafAli01/my-sentiment-model",  # Your HF repository ID
            filename="model.safetensors",              # The specific 255MB file name
            local_dir=model_path                       # Saves it directly into your app directory
        )
        # Load label mappings
        label_mappings_path = os.path.join(model_path, "label_mappings.json")
        with open(label_mappings_path, "r") as f:
            mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings["id2label"].items()}
        label2id = mappings["label2id"]
        
        # Load tokenizer and model from the current directory
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load model info if available
        model_info_path = os.path.join(model_path, "model_info.json")
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
        except FileNotFoundError:
            model_info = {"model_info": {"base_model": "DistilBERT fine-tuned"}}
        
        return model, tokenizer, id2label, label2id, model_info
    
    except Exception as e:
        import traceback
        st.error(f"❌ Error loading model: {str(e)}")
        st.code(traceback.format_exc()) # <-- This prints the exact line causing the crash
        return None, None, None, None, None

# ========== PREDICTION FUNCTION ==========
def predict_sentiment(texts, model, tokenizer, id2label):
    """Predict sentiment for a list of texts"""
    if not texts:
        return [], [], []
    
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        confidence_scores = torch.max(probabilities, dim=-1).values
    
    # Convert to numpy
    predicted_classes = predicted_classes.cpu().numpy()
    confidence_scores = confidence_scores.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Get label names and confidence
    predicted_labels = [id2label[cls] for cls in predicted_classes]
    
    return predicted_labels, confidence_scores, probabilities

# ========== MAIN APP ==========
def main():
    # Load model
    model, tokenizer, id2label, label2id, model_info = load_model()
    
    if model is None:
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/179/179318.png", width=100)
        st.title("⚙️ Settings")
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Information")
        try:
            accuracy = model_info.get('test', {}).get('accuracy', 'N/A')
            f1 = model_info.get('test', {}).get('f1', 'N/A')
            accuracy_str = f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else accuracy
            f1_str = f"{f1:.2%}" if isinstance(f1, (int, float)) else f1
        except Exception:
            accuracy_str = "N/A"
            f1_str = "N/A"
        
        st.info(f"""
        **Base Model:** {model_info.get('model_info', {}).get('base_model', 'DistilBERT')}
        **Training Samples:** {model_info.get('model_info', {}).get('train_samples', 'N/A')}
        **Accuracy:** {accuracy_str}
        **F1 Score:** {f1_str}
        """)
        
        st.markdown("---")
        
        # Batch processing options
        st.subheader("📋 Batch Options")
        batch_mode = st.checkbox("Enable Batch Processing", help="Upload a CSV file for bulk analysis")
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        This app uses a fine-tuned DistilBERT model for sentiment analysis.
        It classifies text into three categories:
        - 😞 Negative
        - 😐 Neutral
        - 😊 Positive
        """)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🧠 Sentiment Analysis with AI")
        st.html("""
        <p style='font-size: 1.2rem; color: #666;'>
        Enter text below to analyze its sentiment. The model will predict if it's 
        <span style='color: #28a745;'>Positive</span>, 
        <span style='color: #ffc107;'>Neutral</span>, or 
        <span style='color: #dc3545;'>Negative</span>.
        </p>
        """)
        
        st.markdown("---")
    
    if not batch_mode:
        # Single text input mode
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_area(
                "✍️ Enter your text here:",
                placeholder="Example: I absolutely love this product! It's amazing!",
                height=150,
                key="single_input"
            )
        with col2:
            st.write("")
            st.write("")
            analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
        
        # Process single text
        if analyze_button and user_input:
            with st.spinner("Analyzing sentiment..."):
                labels, confidences, probs = predict_sentiment(
                    [user_input], model, tokenizer, id2label
                )
                
                label = labels[0]
                confidence = confidences[0]
                
                # Display result
                emoji = "😞" if label == "NEGATIVE" else "😐" if label == "NEUTRAL" else "😊"
                color_class = "negative" if label == "NEGATIVE" else "neutral" if label == "NEUTRAL" else "positive"
                
                st.markdown("---")
                st.subheader("📊 Analysis Result")
                
                # Result card
                st.html(f"""
                <div class='sentiment-box {color_class}'>
                    <h2 style='margin: 0;'>{emoji} {label}</h2>
                    <p style='font-size: 1.1rem; margin: 10px 0;'>
                        Confidence: <strong>{confidence:.2%}</strong>
                    </p>
                </div>
                """)
                
                # Probability distribution
                st.subheader("📈 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': probs[0] * 100
                })
                st.bar_chart(prob_df.set_index('Sentiment'))
                
                # Show raw probabilities
                with st.expander("📊 View Raw Probabilities"):
                    for sentiment, prob in zip(['Negative', 'Neutral', 'Positive'], probs[0]):
                        st.progress(float(prob), text=f"{sentiment}: {prob:.2%}")
        
        elif analyze_button and not user_input:
            st.warning("⚠️ Please enter some text to analyze.")
    
    else:
        # Batch processing mode
        st.subheader("📁 Batch Processing")
        st.markdown("Upload a CSV file with a 'text' column containing the texts to analyze.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("❌ CSV must contain a 'text' column")
                    st.stop()
                
                st.success(f"✅ Successfully loaded {len(df)} rows")
                st.dataframe(df.head(5))
                
                if st.button("🚀 Analyze All Texts", use_container_width=True):
                    with st.spinner(f"Analyzing {len(df)} texts..."):
                        # Process all texts
                        texts = df['text'].tolist()
                        labels, confidences, probs = predict_sentiment(
                            texts, model, tokenizer, id2label
                        )
                        
                        # Add results to dataframe
                        df['sentiment'] = labels
                        df['confidence'] = confidences
                        df['emoji'] = df['sentiment'].map({
                            'NEGATIVE': '😞', 'NEUTRAL': '😐', 'POSITIVE': '😊'
                        })
                        
                        # Create summary
                        summary = df['sentiment'].value_counts().to_dict()
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.html(f"""
                            <div class='metric-card'>
                                <div>😞 Negative</div>
                                <div class='big-number' style='color: #dc3545;'>{summary.get('NEGATIVE', 0)}</div>
                            </div>
                            """)
                        with col2:
                            st.html(f"""
                            <div class='metric-card'>
                                <div>😐 Neutral</div>
                                <div class='big-number' style='color: #ffc107;'>{summary.get('NEUTRAL', 0)}</div>
                            </div>
                            """)
                        with col3:
                            st.html(f"""
                            <div class='metric-card'>
                                <div>😊 Positive</div>
                                <div class='big-number' style='color: #28a745;'>{summary.get('POSITIVE', 0)}</div>
                            </div>
                            """)
                        
                        # Show results table
                        st.subheader("📋 Results")
                        st.dataframe(df[['text', 'emoji', 'sentiment', 'confidence']])
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

if __name__ == "__main__":
    main()
