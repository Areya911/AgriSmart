# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)

    # user profile fields
    password = db.Column(db.String(200), nullable=True)  # store hashed password
    display_name = db.Column(db.String(200), nullable=True)
    email = db.Column(db.String(200), nullable=True)
    phone = db.Column(db.String(40), nullable=True)
    age = db.Column(db.String(20), nullable=True)
    district = db.Column(db.String(120), nullable=True)
    state = db.Column(db.String(120), nullable=True)
    language = db.Column(db.String(30), nullable=True)
    land_size = db.Column(db.String(80), nullable=True)
    farming_type = db.Column(db.String(120), nullable=True)
    profile_pic = db.Column(db.String(300), nullable=True)  # store path like 'uploads/image.jpg'

    # optional last-run metadata
    last_soil_img = db.Column(db.String(300), nullable=True)
    last_soil_result = db.Column(db.String(200), nullable=True)
    last_soil_date = db.Column(db.String(50), nullable=True)
    last_leaf_img = db.Column(db.String(300), nullable=True)
    last_leaf_result = db.Column(db.String(200), nullable=True)
    last_leaf_date = db.Column(db.String(50), nullable=True)

    # store JSON array of suggestions (text) or fallback plain text
    recent_suggestions = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ----- convenience methods -----
    def set_password(self, raw_password: str):
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        if not self.password:
            return False
        return check_password_hash(self.password, raw_password)

    def recent_suggestions_list(self):
        try:
            return json.loads(self.recent_suggestions) if self.recent_suggestions else []
        except Exception:
            return (self.recent_suggestions or "").splitlines()

    def set_recent_suggestions(self, suggestions: list):
        try:
            self.recent_suggestions = json.dumps(suggestions)
        except Exception:
            self.recent_suggestions = "\n".join(suggestions)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "email": self.email,
            "phone": self.phone,
            "district": self.district,
            "state": self.state,
            "language": self.language,
            "land_size": self.land_size,
            "farming_type": self.farming_type,
            "profile_pic": self.profile_pic,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), db.ForeignKey('users.username'), index=True)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
