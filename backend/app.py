from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ================= DATABASE CONFIG =================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "students.db")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# ================= DATABASE MODELS =================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50))

from datetime import datetime

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))
    performance = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# create tables safely
with app.app_context():
    db.create_all()

# ================= LOAD MODEL =================
print("Loading ML Model...")

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

print("Model Loaded Successfully")

# ================= RECOMMENDATION ENGINE =================

def recommend(level):

    if level == "Low":
        return [
            "Revise basic concepts",
            "Watch beginner videos",
            "Increase study hours",
            "Practice assignments"
        ]

    elif level == "Medium":
        return [
            "Solve practice quizzes",
            "Focus on weak topics",
            "Follow structured timetable"
        ]

    else:
        return [
            "Solve advanced problems",
            "Join competitions",
            "Explore new courses"
        ]


# ================= REGISTER =================

@app.route("/register", methods=["POST"])
def register():

    data = request.json

    existing = User.query.filter_by(username=data["username"]).first()

    if existing:
        return jsonify({"status":"exists"})

    new_user = User(
        username=data["username"],
        password=data["password"]
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({"status":"registered"})


# ================= LOGIN =================

@app.route("/login", methods=["POST"])
def login():

    data = request.json

    user = User.query.filter_by(
        username=data["username"],
        password=data["password"]
    ).first()

    if user:
        return jsonify({"status":"success"})
    else:
        return jsonify({"status":"fail"})


# ================= PREDICT =================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        data["attendance"],
        data["assignment_score"],
        data["quiz_score"],
        data["study_hours"],
        data["video_watch_time"],
        data["previous_result"],
        data["attempts"],
        data["participation"]
    ]])

    pred = model.predict(features)
    level = encoder.inverse_transform(pred)[0]

    rec = recommend(level)

    # save history
    record = Prediction(
        username=data["username"],
        performance=level
    )

    db.session.add(record)
    db.session.commit()

    return jsonify({
        "performance": level,
        "recommendations": rec
    })


# ================= HISTORY =================

@app.route("/history/<username>", methods=["GET"])
def history(username):

    records = Prediction.query.filter_by(username=username).all()

    data = []

    for r in records:
        data.append({
            "performance": r.performance,
            "time": r.timestamp.strftime("%d-%m-%Y %H:%M")
        })

    return jsonify(data)


# ================= HOME =================

@app.route("/")
def home():
    return "✅ AI Learning Backend Running"


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
