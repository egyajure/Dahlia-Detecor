from flask import Blueprint, render_template, request, flash, redirect, url_for
from models import User
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from flask_login import login_user, login_required, logout_user, current_user


auth = Blueprint("auth", __name__)


@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user:
            if check_password_hash(user.password, password):
                flash("Logged in successfully!", category="success")
                login_user(user, remember=True)
                return redirect(url_for("views.home"))
            else:
                flash("Incorrect password, try again.", category="error")
        else:
            flash("User does not exist.", category="error")
    return render_template("login.html", user=current_user)


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@auth.route("/sign-up", methods=["GET", "POST"])
def sign_up():
    if request.method == "POST":
        username = request.form.get("username")
        first_name = request.form.get("firstName")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        user = User.query.filter_by(username=username).first()
        if user:
            flash("Username not available.", category="error")
        elif len(username) < 4:
            flash("Username must be greater than 3 characters.", category="error")
        elif len(first_name) == 0:
            flash("First Name is required.", category="error")
        elif password1 != password2:
            flash("Passwords do not match.", category="error")
        elif len(password1) < 7:
            flash("Password must be at least 7 characters.", category="error")
        else:
            # add user to database
            new_user = User(
                username=username,
                first_name=first_name,
                password=generate_password_hash(password1),
            )
            db.session.add(new_user)
            db.session.commit()
            flash("Account created!", category="success")
            login_user(new_user, remember=True)
            return redirect(url_for("views.home"))

    return render_template("sign_up.html", user=current_user)
