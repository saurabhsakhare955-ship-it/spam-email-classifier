# 📧 Spam Email Classifier

A machine learning model that classifies emails/SMS as spam or legitimate using TF-IDF + SVM (~98% accuracy).

## 🛠️ Tech Stack
- **Python 3.8+**
- **Scikit-learn** – Naive Bayes & SVM models
- **TF-IDF Vectorizer** – Text feature extraction
- **Streamlit** – Web application
- **Pandas, NumPy, Seaborn** – Data processing & visualization

## 📁 Project Structure
```
spam_classifier/
├── train_model.py        # Train NB + SVM models
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
└── model/                # Auto-created after training
    ├── spam_model.pkl        # Saved SVM model
    ├── vectorizer.pkl        # Saved TF-IDF vectorizer
    └── results.png           # Confusion matrix & plots
```

## 🚀 Setup & Run

### Step 1: Install Python
Download from https://www.python.org/downloads/ (Python 3.8 or higher)

### Step 2: Install dependencies
Open terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Train the model
```bash
python train_model.py
```
Downloads SMS Spam dataset and trains both models.
Expected accuracy: **~98%**

### Step 4: Run the web app
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

## 🎯 Features
- Single message classification with keyword analysis
- Text statistics (caps, exclamations, URLs, phone numbers)
- Bulk CSV classification with downloadable results
- Quick example buttons for demo

## 🧠 How It Works
1. **TF-IDF Vectorization** – Converts text to numerical features
   - Weights rare but important words higher
   - Uses bigrams (1-2 word combinations)
   - Removes common English stop words

2. **Model Training**
   - Naive Bayes: Fast probabilistic classifier
   - SVM (LinearSVC): Better for high-dimensional text data
   - SVM wins with ~98% accuracy

3. **Prediction**
   - New text → TF-IDF transform → SVM classify → Ham/Spam

## 📊 Results
- Accuracy: ~98%
- Precision (Spam): ~99%
- Recall (Spam): ~95%
- Dataset: SMS Spam Collection (5,572 messages)

## 💡 Jury Talking Points
- TF-IDF handles the "term importance" problem better than raw word counts
- SVM is great for high-dimensional sparse data (text)
- Bigrams capture phrases like "click here" and "free offer"
- The model correctly identifies spam indicators: ALL CAPS, $$$, urgency words
