# LinkedIn / GitHub Portfolio Report — CitizenSense

## Project Title
**CitizenSense — NLP-Based Public Feedback Classification System**

## 1. Project Overview

I built CitizenSense, an NLP-based feedback analysis system that classifies public
complaints into categories such as roads, water, electricity, sanitation, health,
education, traffic, and general issues. The system also identifies sentiment,
priority, keywords, and a short summary for each message, then suggests which
department it should be routed to. The goal was to understand how NLP can help
organize large amounts of unstructured citizen feedback into consistent, useful
insights that would otherwise take a human hours to triage manually.

## 2. Tools and Technologies Used

Tools used: Python, Pandas, NumPy, Scikit-learn, TF-IDF, Logistic Regression,
Hugging Face Transformers (DistilBERT sentiment-analysis pipeline), PyTorch,
Matplotlib, and Streamlit.

## 3. Dataset Preparation

I prepared a synthetic dataset of 739 feedback samples across 8 categories, written
from scratch as sentence templates and deliberately including a slice of genuinely
ambiguous, hard-to-classify wording so the dataset isn't artificially "too easy."
I cleaned the text by removing duplicates, URLs, phone-number patterns, and extra
spaces, and split the data into stratified training (70%), validation (15%), and
test (15%) sets.

## 4. Methodology

The workflow started with text generation and labeling (category, sentiment,
priority). I then cleaned the dataset and converted the text into TF-IDF features.
A Logistic Regression model was trained for category classification. I also used a
Hugging Face sentiment-analysis pipeline (pretrained DistilBERT) for sentiment, and
added TF-IDF-based keyword extraction plus a structured summary generator to produce
a complete "insight card" for each message.

## 5. Results

The category model was evaluated using accuracy, precision, recall, and F1-score,
reaching **92.8% test accuracy** (weighted F1 0.929) across 8 classes. The confusion
matrix showed that Electricity, Roads/Infrastructure, and Sanitation were easiest to
classify, while General and Water Supply were most often confused with each other
and with other categories — mainly on short, vague messages with no distinctive
keyword. The Hugging Face sentiment pipeline reached 74.8% zero-shot accuracy against
the dataset's sentiment labels, performing well on clearly positive/negative text
but weaker on neutral/question-style messages.

## 6. Demo and Outputs

I created a Streamlit demo (plus a CLI and FastAPI version) where users can enter
feedback text and receive the predicted category, sentiment, priority, keywords,
summary, and suggested department in real time.

## 7. Error Analysis

I manually reviewed 15 incorrect or low-confidence predictions. Most errors came
from short, generic complaints that lacked a clear category-specific keyword, or
from the deliberately ambiguous examples where multiple categories were equally
plausible from the wording alone. More labeled data, better label guidelines for
vague complaints, and fine-tuning a transformer could all improve the system further.

## 8. What I Learned

- Learned how to design and clean a labeled NLP dataset from scratch.
- Learned how to intentionally build realistic classification difficulty instead of
  an artificially perfect dataset.
- Learned how TF-IDF represents text, and how corpus-level vs document-level
  TF-IDF changes keyword quality.
- Learned how to train and evaluate a multi-class text classification model.
- Learned how to use Hugging Face pretrained pipelines for tasks beyond their
  default output shape (adapting a binary sentiment model into a 3-way label).
- Learned how to perform structured error analysis on text predictions and turn
  it into concrete next steps.

## 9. Future Improvements

- Add Urdu and Roman Urdu support.
- Add multi-label classification.
- Fine-tune a transformer model on the CitizenSense categories directly.
- Add a dedicated trained priority classifier.
- Add semantic search and duplicate complaint detection.
- Add dashboard charts for category distribution.
- Deploy the model as a public API.

## 10. LinkedIn Post Format

Today, I completed **AIRI Team PITB - AI Internship Task 4**, where I built an
end-to-end **Natural Language Processing project**.

For this task, I developed **CitizenSense**, an NLP-based system that analyzes text
feedback and predicts useful outputs such as category, sentiment, priority,
keywords, and summary.

The project covered the complete NLP workflow:
- Text data collection
- Data cleaning and preprocessing
- Label creation
- TF-IDF feature extraction
- Baseline model training
- Hugging Face transformer usage
- Evaluation metrics
- Error analysis
- Demo interface
- Final documentation

This project helped me understand how NLP systems can convert unstructured text
into useful insights for decision-making.

**Tools used:** Python, Pandas, Scikit-learn, TF-IDF, Hugging Face Transformers, and
Streamlit.

**Key learning:** NLP is not only about generating text. It is about understanding,
organizing, classifying, and extracting meaning from real-world language data.

#NLP #NaturalLanguageProcessing #MachineLearning #Transformers #HuggingFace #Python
#AIRI #PITB #AIInternship #TextClassification
