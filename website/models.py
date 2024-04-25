from app import db
from flask_login import UserMixin
from sqlalchemy.sql import func
from flask import send_from_directory
from app import UPLOAD_FOLDER


# example model, will need to make something like this to save users images
class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    guess = db.Column(db.String(150))  # maybe turn this into options
    score = db.Column(db.Integer)
    name = db.Column(db.String(150))
    file_path = db.Column(db.String(150), unique=True)

    def __str__(self):
        return f"Image {self.name}"

    def __repr__(self):
        return f"Image {self.name}"

    def get(self, id):
        img = Image.query.filter(Image.id == id).first()
        path = img.file_path
        return send_from_directory(UPLOAD_FOLDER, path).response.file


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    # another example, some flask magic that works from giving the name of the class
    notes = db.relationship("Note")
