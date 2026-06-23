import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, urllib.request

print("=" * 55)
print("   Spam Email Classifier - Training")
print("=" * 55)

# ── Download Dataset ───────────────────────────────────────────────────────────
print("\n[1/5] Downloading SMS Spam Collection dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
try:
    urllib.request.urlretrieve(url, "spam_data.tsv")
    df = pd.read_csv("spam_data.tsv", sep="\t", header=None, names=["label", "message"])
    print(f"  ✅ Dataset downloaded: {len(df)} samples")
except Exception as e:
    print(f"  ⚠️  Download failed ({e}), using built-in sample data...")
    # Fallback built-in dataset
    data = {
        "label": (["ham"] * 50 + ["spam"] * 50),
        "message": [
            "Hey, are you free this weekend?", "Let's meet for lunch tomorrow.",
            "I'll call you later tonight.", "Can you send me the report?",
            "Happy birthday! Hope you have a great day.", "Are we still on for Friday?",
            "I just got home, what's up?", "Thanks for your help yesterday.",
            "See you at the office tomorrow.", "Don't forget the meeting at 3pm.",
            "How was your trip?", "Can you pick me up from the station?",
            "I'll be there in 10 minutes.", "Did you get my email?",
            "The project deadline is next week.", "Call me when you get a chance.",
            "I'm running a bit late.", "What time does the event start?",
            "Let me know if you need anything.", "Great work on the presentation!",
            "I'll be working from home today.", "Can we reschedule our meeting?",
            "The wifi password is 12345678.", "I forgot my umbrella at your place.",
            "The package arrived this morning.", "Are you watching the game tonight?",
            "I need the files by end of day.", "Let's grab coffee sometime.",
            "The car is ready for pickup.", "I'll forward the email to you.",
            "Can you review this document?", "We need to talk about the budget.",
            "I'm in a meeting right now.", "What's the address for the event?",
            "Did you see the news today?", "I'll send you the link shortly.",
            "The restaurant opens at 7pm.", "I'm almost done with the report.",
            "Can you cover my shift on Monday?", "The flight lands at 6am.",
            "I'll be out of office next week.", "Did you finish the assignment?",
            "The store closes at 9pm.", "I left my keys at your place.",
            "What do you want for dinner?", "I'm feeling much better today.",
            "The meeting was rescheduled to Tuesday.", "Can you lend me your notes?",
            "I'll pick you up at 8.", "The event was cancelled.",
            "WINNER! You've won a FREE iPhone! Click here NOW!",
            "Congratulations! You've been selected for a cash prize of $1000!",
            "FREE OFFER: Get 1000 free texts now! Call 09061 000 321",
            "You have WON a guaranteed $1,000,000 prize! Reply YES to claim!",
            "Urgent! Your account has been suspended. Click to verify NOW!",
            "CLAIM your FREE gift card worth $500! Limited time offer!",
            "You're a winner! Call 0800 FREE to claim your prize TODAY!",
            "Get out of DEBT now! We can eliminate all your credit card debt!",
            "Hot singles in your area want to meet you tonight! Click here!",
            "WARNING: Your computer is infected! Download our FREE antivirus!",
            "Make $5000 per week working from home! No experience needed!",
            "FREE ringtones! Text TONE to 87070. Ends 30/06. £1.50/msg!",
            "You have been pre-approved for a $10,000 loan! Apply now!",
            "FINAL NOTICE: Your subscription expires today. Renew for FREE!",
            "Congratulations! You are our lucky customer. Click to win!",
            "BUY NOW: Lose 30lbs in 30 days with our miracle diet pill!",
            "Your mobile number has won £1,500,000 in our annual lottery!",
            "Act now! Limited time offer expires midnight. 50% off everything!",
            "We have been trying to reach you about your car's warranty!",
            "FREE trial! No credit card needed! Cancel anytime! Click here!",
            "Your PayPal account has been limited. Verify your identity now!",
            "EARN MONEY: £20 per survey completed. Sign up free today!",
            "You qualify for 0% APR credit card! Apply in 60 seconds!",
            "Double your income! Our system works 100% guaranteed!",
            "Special offer for you only! Buy 1 get 2 FREE! Today only!",
            "Your prize is waiting! Call 0800 PRIZE to claim immediately!",
            "ALERT: Suspicious login detected. Click to secure your account!",
            "FREE! FREE! FREE! 500 SMS messages daily! Txt YES to 80488",
            "You have been selected for an exclusive VIP membership offer!",
            "CASH ADVANCE: Get $1000 deposited to your account today!",
            "Last chance! Your reward expires in 24 hours. Claim now!",
            "We're offering you an exclusive deal. Reply to find out more!",
            "Congratulations on your weight loss! Our pill will help you more!",
            "Your IQ is in top 1%! Take our FREE quiz and win prizes!",
            "SIX FIGURE INCOME possible working from home. Learn how!",
            "FREE entry to our monthly prize draw. Text WIN to 83738!",
            "Reminder: You have an unclaimed prize. Call NOW: 0800-PRIZE",
            "Your account statement is ready. Login at fake-bank.net NOW!",
            "CHEAP MEDS: Viagra, Cialis delivered to your door! No Rx needed!",
            "URGENT: Your account will be closed. Verify at secure-bank-fake.com",
            "You won a Samsung Galaxy! To claim call 09050000460 now!",
            "FREE membership! Access 1000s of lonely wives tonight!",
            "IMPORTANT: Your credit score has changed. View your free report!",
            "Increase your sales by 300%! Buy our leads database now!",
            "FREE! Call from PC to Mobile now! Download our FREE software!",
        ]
    }
    df = pd.DataFrame(data)
    print(f"  ✅ Using built-in dataset: {len(df)} samples")

# ── Explore Data ───────────────────────────────────────────────────────────────
print("\n[2/5] Exploring dataset...")
print(f"  Class distribution:\n{df['label'].value_counts()}")
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df['message_length'] = df['message'].apply(len)

# ── Split Data ─────────────────────────────────────────────────────────────────
print("\n[3/5] Splitting data (80% train / 20% test)...")
X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples:     {len(X_test)}")

# ── Train Models ───────────────────────────────────────────────────────────────
print("\n[4/5] Training models...")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Naive Bayes
print("  → Training Naive Bayes...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)
nb_acc   = accuracy_score(y_test, nb_preds)
print(f"    Naive Bayes Accuracy: {nb_acc*100:.2f}%")

# SVM
print("  → Training SVM (LinearSVC)...")
svm_model = LinearSVC(C=1.0, max_iter=2000)
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)
svm_acc   = accuracy_score(y_test, svm_preds)
print(f"    SVM Accuracy:         {svm_acc*100:.2f}%")

best_model = svm_model
best_preds = svm_preds

print(f"\n  ✅ Best model: SVM ({svm_acc*100:.2f}%)")
print("\n  Classification Report (SVM):")
print(classification_report(y_test, best_preds, target_names=['Ham', 'Spam']))

# ── Visualizations ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating visualizations...")
os.makedirs("model", exist_ok=True)

# 1. Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Spam Email Classifier - Model Results', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
            annot_kws={"size": 16})
axes[0].set_title('Confusion Matrix (SVM)')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# 2. Model Comparison
models_acc = [nb_acc*100, svm_acc*100]
bars = axes[1].bar(['Naive Bayes', 'SVM'], models_acc, color=['#4CAF50','#2196F3'], width=0.5)
axes[1].set_ylim(85, 100)
axes[1].set_title('Model Accuracy Comparison')
axes[1].set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, models_acc):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                 f'{acc:.1f}%', ha='center', va='top', color='white', fontweight='bold', fontsize=13)

# 3. Message Length Distribution
ham_len  = df[df['label'] == 'ham']['message_length']
spam_len = df[df['label'] == 'spam']['message_length']
axes[2].hist(ham_len, bins=30, alpha=0.7, label='Ham', color='#4CAF50')
axes[2].hist(spam_len, bins=30, alpha=0.7, label='Spam', color='#f44336')
axes[2].set_title('Message Length Distribution')
axes[2].set_xlabel('Message Length (chars)')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig("model/results.png", dpi=150, bbox_inches='tight')
print("  ✅ Results plot saved to model/results.png")

# ── Save Model & Vectorizer ────────────────────────────────────────────────────
pickle.dump(best_model,  open("model/spam_model.pkl", "wb"))
pickle.dump(vectorizer,  open("model/vectorizer.pkl", "wb"))
print("  ✅ Model saved to model/spam_model.pkl")
print("  ✅ Vectorizer saved to model/vectorizer.pkl")

print(f"\n{'='*55}")
print(f"  🎉 Training complete!")
print(f"  Naive Bayes Accuracy: {nb_acc*100:.2f}%")
print(f"  SVM Accuracy:         {svm_acc*100:.2f}%")
print(f"  Best Model: SVM")
print(f"{'='*55}")
print(f"\n  ▶ Now run: streamlit run app.py")
