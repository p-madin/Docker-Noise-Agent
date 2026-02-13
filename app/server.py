import nltk
import pickle
import zlib
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from audio_engine import classifier

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database Configuration
# Using the service name 'db' from compose.yaml as the hostname
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://docker_user_lemp:docker_user_lemp@db/stackDB'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Detection Model
class Detection(db.Model):
    __tablename__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    audio_data = db.Column(db.LargeBinary(length=(2**32)-1)) # LONGBLOB
    is_flagged = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "has_audio": self.audio_data is not None,
            "is_flagged": self.is_flagged,
            "timestamp": self.created_at.strftime("%H:%M:%S")
        }

# Create tables if they don't exist
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"Warning: Could not create database tables: {e}", flush=True)

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def get_tweet_features(tokens):
    return {word: True for word in tokens}

# Load the trained classifier for text sentiment
MODEL_PATH = 'my_sentiment_classifier.pickle'
CONFIDENCE_THRESHOLD = 0.6

def load_classifier():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.route('/classify_audio', methods=['POST'])
def classify_audio():
    if not request.data:
        return jsonify({"error": "No audio data received"}), 400
    
    label, confidence = classifier.classify(request.data)
    
    # Save high-confidence detections to the database
    # Non-silence and non-background (if filtered)
    if label != "Silence" and label != "Error":
        try:
            # Compress the raw bytes (request.data contains the 1s WAV chunk)
            # zlib typically gives 20-30% reduction for WAV at level 6
            compressed_data = zlib.compress(request.data)
            new_detection = Detection(label=label, confidence=confidence, audio_data=compressed_data)
            db.session.add(new_detection)
            db.session.commit()
        except Exception as e:
            print(f"Database error: {e}", flush=True)

    return jsonify({
        "detected": label,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

@app.route('/history', methods=['GET'])
def history():
    """Retrieve the last 50 high-confidence detections."""
    try:
        detections = Detection.query.order_by(Detection.created_at.desc()).limit(50).all()
        return jsonify([d.to_dict() for d in detections])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<int:detection_id>')
def get_audio(detection_id):
    """Serve the WAV bytes for a specific detection (handles zlib decompression)."""
    detection = Detection.query.get_or_404(detection_id)
    if not detection.audio_data:
        return "No audio recording found", 404
    
    # Check if data is compressed (zlib header usually starts with 0x78)
    # This maintains backward compatibility with uncompressed records
    data = detection.audio_data
    try:
        # attempt decompression
        data = zlib.decompress(detection.audio_data)
    except zlib.error:
        # if it fails, it's likely already raw wav bytes
        pass
    
    return Response(data, mimetype="audio/wav")

@app.route('/flag/<int:detection_id>', methods=['POST'])
def flag_detection(detection_id):
    """Flag a detection as a 'Hard Example' for future retraining."""
    try:
        detection = Detection.query.get_or_404(detection_id)
        detection.is_flagged = True
        db.session.commit()
        return jsonify({"status": "flagged", "id": detection_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    text = ""
    classifier_text = load_classifier()
    
    if request.method == 'POST':
        if not classifier_text:
            return jsonify({"error": "Classifier model not found"}), 500
            
        text = request.form.get('text', '')
        if text:
            tokens = word_tokenize(text)
            features = get_tweet_features(tokens)
            
            prob_dist = classifier_text.prob_classify(features)
            max_prob = prob_dist.prob(prob_dist.max())
            
            if max_prob < CONFIDENCE_THRESHOLD:
                sentiment = "Neutral"
            else:
                sentiment = prob_dist.max()
            
            probabilities = {
                "Positive": prob_dist.prob("Positive"),
                "Negative": prob_dist.prob("Negative")
            }

            informative_features = []
            cpdist = classifier_text._feature_probdist
            
            for (fname, fval) in features.items():
                p_pos = cpdist['Positive', fname].prob(fval) if ('Positive', fname) in cpdist else 0
                p_neg = cpdist['Negative', fname].prob(fval) if ('Negative', fname) in cpdist else 0
                
                if p_pos > 0 and p_neg > 0:
                    ratio = p_pos / p_neg
                    if ratio > 1:
                        informative_features.append({"feature": fname, "label": "Positive", "ratio": ratio})
                    else:
                        informative_features.append({"feature": fname, "label": "Negative", "ratio": 1/ratio})
            
            informative_features.sort(key=lambda x: x['ratio'], reverse=True)
            top_features = informative_features[:5]
            
            return jsonify({
                "text": text,
                "sentiment": sentiment,
                "probabilities": probabilities,
                "informative_features": top_features
            })
            
    return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
