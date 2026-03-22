print("🚀 AI Student Learning Model Training Started")

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# =========================
# 1️⃣ Load Dataset
# =========================
print("📊 Loading dataset...")

data = pd.read_csv("../dataset/student_data.csv")

print("✅ Dataset Loaded Successfully")
print("Total Records:", len(data))


# =========================
# 2️⃣ Encode Target
# =========================
le = LabelEncoder()
data["final_performance"] = le.fit_transform(data["final_performance"])


# =========================
# 3️⃣ Split Features & Target
# =========================
X = data.drop(["student_id", "final_performance"], axis=1)
y = data["final_performance"]


# =========================
# 4️⃣ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🧠 Training Random Forest Model...")


# =========================
# 5️⃣ Train Model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model Training Completed")


# =========================
# 6️⃣ Evaluation
# =========================
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")

print("\n📄 Classification Report:\n")
print(classification_report(y_test, pred))


# =========================
# 7️⃣ Confusion Matrix Graph
# =========================
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()


# =========================
# 8️⃣ Feature Importance Graph
# =========================
importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, importance)
plt.title("Feature Importance Analysis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 9️⃣ Save Model
# =========================
joblib.dump(model, "../backend/model.pkl")
joblib.dump(le, "../backend/encoder.pkl")

print("\n💾 Model Saved Successfully in backend folder")

print("🏁 Training Process Finished")