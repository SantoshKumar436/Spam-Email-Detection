import os
import sys
import joblib
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spam_detector import SpamDetector, parse_raw_email, preprocess_text, extract_meta_features

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_spam_detector.pkl")

def load_model():
    print(f"Loading model from {model_path}...")
    obj = joblib.load(model_path)
    # Support dict format (from newer training code) or direct SpamDetector object
    if isinstance(obj, dict):
        model = SpamDetector(
            feature_method=obj["feature_method"],
            classifier=obj["classifier"],
            vectorizer=obj["vectorizer"],
            scaler=obj.get("scaler")
        )
    else:
        model = obj  # Already a SpamDetector instance
    print(f"Model loaded: {type(model).__name__}, method={model.feature_method}")
    return model

model = load_model()

@app.route("/")
def home():
    """Serves the main frontend page."""
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_email():
    """
    API endpoint that accepts email text, processes it,
    predicts spam/ham, and returns confidence scores and feature details.
    """
    try:
        data = request.get_json()
        if not data or "email_text" not in data:
            return jsonify({"error": "No email text provided"}), 400

        email_text = data["email_text"]

        # 1. Parse raw email
        parsed = parse_raw_email(email_text)

        # 2. Get prediction & probabilities (SpamDetector takes a single string)
        prediction = model.predict(email_text)
        ham_prob, spam_prob = model.predict_proba(email_text)

        # 3. Preprocess text to show in UI
        full_text = parsed['body'] + " " + parsed['subject']
        clean_text = preprocess_text(full_text)

        # 4. Extract individual features for visualization
        meta = extract_meta_features([parsed])[0]

        features_breakdown = {
            "caps_ratio": round(float(meta[0]) * 100, 1),
            "url_count": int(meta[1]),
            "exclamation_count": int(meta[2]),
            "suspicion_score": float(meta[3]),
            "has_shortener": bool(meta[4]),
            "is_free_domain": bool(meta[5]),
            "numeric_username_ratio": round(float(meta[6]) * 100, 1)
        }

        confidence = round(spam_prob * 100, 2) if prediction == "spam" else round(ham_prob * 100, 2)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "spam_probability": round(spam_prob * 100, 2),
            "parsed": {
                "sender": parsed["sender"] or "Unknown Sender",
                "subject": parsed["subject"] or "(No Subject)",
                "body_preview": parsed["body"][:500] + ("..." if len(parsed["body"]) > 500 else "")
            },
            "cleaned_text": clean_text,
            "features": features_breakdown
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
