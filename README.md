# 📧 Spam Email Classifier

A machine learning-based Spam Email Classifier built using Python, Scikit-learn, and Streamlit. The application predicts whether a given message is **Spam** or **Not Spam (Ham)** with high accuracy.

## 🚀 Features

* Detects spam and legitimate messages
* User-friendly Streamlit web interface
* TF-IDF text vectorization
* Support Vector Machine (SVM) classifier
* Fast and accurate predictions

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Pickle

## 📂 Project Structure

```text
spam_classifier/
├── model/
│   ├── spam_model.pkl
│   ├── vectorizer.pkl
│   └── results.png
├── app.py
├── train_model.py
├── spam_data.tsv
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

* **Dataset:** SMS Spam Collection Dataset
* **Total Messages:** 5,572
* **Classes:** Spam and Ham

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saurabhsakhare955-ship-it/spam-email-classifier.git
cd spam-email-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📈 Model Performance

* Accuracy: ~98%
* Precision (Spam): ~99%
* Recall (Spam): ~95%

## 💡 Future Improvements

* Deploy using Streamlit Cloud
* Add multiple language support
* Improve UI/UX
* Support batch email prediction

## 👨‍💻 Author

**Saurabh Sakhare**

GitHub: https://github.com/saurabhsakhare955-ship-it
