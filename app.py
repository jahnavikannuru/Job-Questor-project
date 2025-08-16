import re
import spacy
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from PyPDF2 import PdfReader
from docx import Document
from datetime import datetime
from bson.objectid import ObjectId

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

nlp = spacy.load("en_core_web_sm")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "job_questor_db"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]
jobs_collection = db["jobs"]
preferences_collection = db["preferences"]
notifications_collection = db["notifications"]

users_collection.create_index("username", unique=True)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "Username and password required"}), 400

    username = data["username"]
    hashed_password = generate_password_hash(data["password"])

    try:
        users_collection.insert_one({
            "username": username,
            "password_hash": hashed_password
        })
        return jsonify({"message": "Signup successful"}), 200
    except DuplicateKeyError:
        return jsonify({"error": "User already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = users_collection.find_one({"username": username})
    if user and check_password_hash(user["password_hash"], password):
        return jsonify({"username": username}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/set_preferences', methods=['POST'])
def set_preferences():
    data = request.json
    username = data.get("username")
    preferences = data.get("preferences")

    if not username or not preferences:
        return jsonify({"error": "Username and preferences required"}), 400

    result = preferences_collection.update_one(
        {"username": username},
        {"$set": {"preferences": preferences}},
        upsert=True
    )
    return jsonify({"message": "Preferences saved"}), 200

def notify_matching_users(new_job):
    all_preferences = preferences_collection.find()
    for pref in all_preferences:
        user = pref["username"]
        prefs = pref["preferences"]

        matches_role = any(role.lower() in new_job.get("title", "").lower() for role in prefs.get("roles", []))
        matches_location = any(loc.lower() in new_job.get("location", "").lower() for loc in prefs.get("locations", []))
        matches_skills = any(skill.lower() in new_job.get("description", "").lower() for skill in prefs.get("skills", []))

        if matches_role or matches_location or matches_skills:
            notifications_collection.insert_one({
                "username": user,
                "job_id": str(new_job.get("_id")),
                "message": f"New job posted: {new_job.get('title')} at {new_job.get('company')}",
                "created_at": datetime.utcnow(),
                "read": False
            })

@app.route('/notifications/<username>', methods=['GET'])
def get_notifications(username):
    notifications = list(notifications_collection.find(
        {"username": username, "read": False}
    ))

    enriched = []
    for notif in notifications:
        job_id = notif.get("job_id")
        job = jobs_collection.find_one({"_id": ObjectId(job_id)}) if job_id else None

        enriched.append({
            "message": notif.get("message"),
            "created_at": notif.get("created_at"),
            "location": job.get("location") if job else "Unknown",
            "company": job.get("company") if job else "Unknown",
            "title": job.get("title") if job else "Unknown",
            "apply_url": job.get("apply_url") if job else None
        })

    return jsonify(enriched), 200

@app.route('/mark_read', methods=['POST'])
def mark_notification_read():
    data = request.json
    username = data.get("username")

    result = notifications_collection.update_many(
        {"username": username, "read": False},
        {"$set": {"read": True}}
    )
    return jsonify({"message": f"{result.modified_count} notifications marked as read."}), 200

# ALLOWED EXTENSIONS INCLUDING DOC & DOCX
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

SKILL_KEYWORDS = [
    'javascript', 'python', 'java', 'c++', 'react', 'angular', 'node.js', 'node',
    'sql', 'mongodb', 'aws', 'docker', 'kubernetes', 'html', 'css', 'typescript',
    'linux', 'git', 'machine learning', 'deep learning', 'data analysis', 'nlp',
    'communication', 'project management', 'c#', 'php', 'ruby', 'swift', 'go',
    'tensorflow', 'django', 'flask', 'rest api', 'graphql', 'agile', 'devops',
    'ui/ux', 'testing', 'automation', 'sales', 'marketing', 'customer service',
    'accounting', 'finance', 'business', 'analysis'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_txt(file_stream):
    content = file_stream.read()
    return content.decode('utf-8', errors='ignore').lower() if isinstance(content, bytes) else content.lower()

def extract_text_from_pdf(file_stream):
    try:
        reader = PdfReader(file_stream)
        return " ".join([page.extract_text() or "" for page in reader.pages]).lower()
    except Exception:
        return ""

def extract_text_from_docx(file_stream):
    try:
        document = Document(file_stream)
        return " ".join([para.text for para in document.paragraphs]).lower()
    except Exception:
        return ""

def extract_skills(text):
    doc = nlp(text)
    found_skills = set()
    tokens = {token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct}
    noun_chunks = {chunk.lemma_.lower() for chunk in doc.noun_chunks}
    all_phrases = tokens.union(noun_chunks)

    for skill in SKILL_KEYWORDS:
        if skill.lower() in all_phrases:
            found_skills.add(skill.lower())

    return sorted(found_skills)

def find_matching_jobs(skills):
    results = []
    skills_set = set(skills)
    for job in jobs_collection.find({}):
        job_skills = set([s.lower() for s in job.get('skills', [])])
        matched = skills_set.intersection(job_skills)
        if matched:
            results.append({
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "description": job.get("description"),
                "logo": job.get("logo"),
                "apply_url": job.get("apply_url"),
                "matched_skills": sorted(list(matched)),
                "matched_skills_count": len(matched)
            })
    results.sort(key=lambda x: x['matched_skills_count'], reverse=True)
    return results

@app.route('/recommend-jobs', methods=['POST'])
def recommend_jobs():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    resume_file = request.files['resume']
    if resume_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(resume_file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    ext = resume_file.filename.rsplit('.', 1)[1].lower()

    if ext == 'txt':
        text = extract_text_from_txt(resume_file.stream)
    elif ext == 'pdf':
        text = extract_text_from_pdf(resume_file.stream)
    elif ext in ['doc', 'docx']:
        text = extract_text_from_docx(resume_file.stream)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    if not text.strip():
        return jsonify({"error": "Could not extract text from resume"}), 400

    skills = extract_skills(text)
    matches = find_matching_jobs(skills)

    return jsonify({
        "detected_skills": skills,
        "recommended_jobs": [
            {
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "description": job['description'],
                "logo": job['logo'],
                "apply_url": job['apply_url'],
                "matched_skills": job['matched_skills']
            } for job in matches
        ]
    }), 200

@app.route('/')
def serve_root():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    if '..' in filename or filename.startswith('/'):
        return "Forbidden", 403
    return send_from_directory('.', filename)

def seed_jobs_data():
    if jobs_collection.count_documents({}) > 0:
        return
    job_list = [...]  # Replace with your job entries
    jobs_collection.insert_many(job_list)
    print("Seeded job listings into database.")

@app.route('/add_job', methods=['POST'])
def add_job():
    data = request.json
    result = jobs_collection.insert_one(data)
    data['_id'] = result.inserted_id
    notify_matching_users(data)
    return jsonify({"message": "Job added and notifications sent."}), 200

if __name__ == '__main__':
    seed_jobs_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
