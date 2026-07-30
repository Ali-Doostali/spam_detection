import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("1. Loading Enron Spam Dataset...")
# بارگذاری دیتاست آماده اسپم (enron_spam)
dataset = load_dataset("SetFit/enron_spam")
df = pd.DataFrame(dataset['train'])

# پاک‌سازی اولیه داده‌های خالی
df = df.dropna(subset=['text', 'label'])

print("2. Vectorizing text data (TF-IDF)...")
X = df['text']
y = df['label'] # 1: Spam, 0: Ham (Normal)

# تقسیم داده‌ها به داده آموزش و تست (۸۰ به ۲۰)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تبدیل متن به وکتور
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("3. Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_vec, y_train)

print("4. Evaluating Model...")
y_pred = model.predict(X_test_vec)

# محاسبه ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n--- Performance Metrics ---")
print(f"True Positives (TP) - Spam correctly identified: {tp}")
print(f"True Negatives (TN) - Normal emails correctly identified: {tn}")
print(f"False Positives (FP) - Normal emails marked as Spam: {fp}")
print(f"False Negatives (FN) - Spam emails missed: {fn}")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# رسم نمودار و ذخیره به عنوان عکس برای مقاله
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Spam Detection')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion Matrix saved as 'confusion_matrix.png'.")