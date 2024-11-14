from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from . import mongo, client, db, users_collection, sessions_collection, mail, socketio
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Message
from flask_login import login_user, login_required, logout_user, current_user
import pickle
from bson.objectid import ObjectId
import datetime 
import jwt
from functools import wraps
from os import getenv
from dotenv import load_dotenv

load_dotenv()

auth = Blueprint('auth', __name__)

SECRET_KEY = getenv("SECRET_KEY")

s = URLSafeTimedSerializer(SECRET_KEY) 

def generate_token(user_id):
    token = jwt.encode({
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm='HS256')
    return token

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user_id = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user_id, *args, **kwargs)
    return decorated

@auth.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        firstName = data.get('firstName')
        password1 = data.get('password')
        password2 = data.get('confirm_password')

        user = users_collection.find_one({"email":email})
        if user:
            return jsonify({"error": "Email already exists"}), 400
        elif password1 != password2:
            return jsonify({"error": "Passwords don't match"}), 400
        else:
            # confirm email stuff                            #  HACK: temp 
            # token = s.dumps(email, salt='email-confirm')

            # msg = Message("Verify email", recipients=[email])
            # link = url_for("auth.confirm_email", token=token, _external = True)
            # msg.body = f"Please click on the link to verify your email: {link}"
            # mail.send(msg)

            new_user = {
                "email": email,
                "firstName": firstName,
                "password": generate_password_hash(password1, method='pbkdf2:sha256'),
                "verified": True                                #  HACK: temp 
            }
            user_id = users_collection.insert_one(new_user).inserted_id
            user_id = str(user_id)
            jwt_token = generate_token(user_id)                 #  HACK: temp 

            # login_user(new_user, remember=True)
            # log_activity("signup", f"User {new_user.firstName} signed up")
            # log_activity("verification_email_sent", f"User {new_user.firstName} was sent the verification email")
            
            return jsonify({"message": "Registered successfully", "jwt_token": jwt_token}), 201    #  HACK: temp                 

    return jsonify({"message": "Signup endpoint ready"}), 200


@auth.route("/confirm_email/<token>")
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=3600)
    except:
        return jsonify({"error": "Invalid or Expired token"}), 400
    
    user = users_collection.find_one({"email": email})
    
    if user:
        if user.get("verified"):
            return jsonify({"message": "Account already confirmed. Please login!"}), 200
        else:
            users_collection.update_one({"email": email}, {"$set": {"verified": True}})    
            user_id = str(user["_id"])
            user_data = {
                "user_id": str(user_id),
                "email": email,
                "firstName": user["firstName"],                                    
            }
            jwt_token = generate_token(user_id)                
            # log_activity("user_verified", f"User {user.firstName} verified their email")
            return jsonify({"message": "Email confirmed! You can now login!", "jwt_token": jwt_token, "user_data": user_data}), 200
    else:
        return jsonify({"error": "Invalid email! Please signup"}), 400


@auth.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        user = users_collection.find_one({"email": email})

        if user:
            if check_password_hash(user.get("password"), password):
                if user.get("verified"):
                    # login_user(user, remember=True)
                    # log_activity("login", f"User {user.firstName} logged in")
                    user_json = pickle.dumps(user)
                    session['user'] = user_json
                    user_id = str(user["_id"])  
                    user_data = {
                        "user_id": str(user_id),
                        "email": email,
                        "firstName": user["firstName"],                                    
                    }
                    jwt_token = generate_token(user_id)
                    return jsonify({"message": "Logged in successfully", "jwt_token": jwt_token, "user_data": user_data}), 200
                else:
                    return jsonify({"error": "Email not verified! Please verify your email"}), 400
            else:
                return jsonify({"error": "Incorrect password"}),400
                # log_activity("failed_login_attempt", f"User {user.firstName} entered incorrect password")
                
        else:
            return jsonify({"error": "User does not exist"}), 400            
            # log_activity("failed_login_attempt", f"User does not exist")

    return jsonify({"message": "Login endpoint ready"}), 200


@auth.route("/logout")
@token_required
def logout(current_user_id):
    token = request.headers.get('x-access-token')
    if not token:
        return jsonify({"message": "Token is missing!"}), 400
    logout_user()
    session.pop('user', None)
    return jsonify({"message": "Logged out successfully"}), 200


@auth.route("/forgot_password", methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        user = users_collection.find_one({"email": email})
        if user:
            token = s.dumps(email, salt='forgot-password')
            msg = Message("Reset password", recipients=[email])
            link = url_for("auth.reset_password", token=token, _external = True)
            msg.body = f"Please click on the link to reset your password: {link}"
            mail.send(msg)
            user_data = {
                "email": email,
                "firstName": user["firstName"]
            }
            return jsonify({"message": "Password reset link sent", "user_data": user_data}), 200
        else:
            return jsonify({"error": "User does not exist"}), 400
    return jsonify({"message": "Forgot password endpoint ready"}), 200


@auth.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        try:
            email = s.loads(token, salt='forgot-password', max_age=3600)
        except:
            return jsonify({"error": "Invalid or Expired token"}), 400
        user = users_collection.find_one({"email": email})
        if user:
            data = request.get_json()
            password1 = data.get('password')
            password2 = data.get('confirm_password')
            if password1 == password2:
                users_collection.update_one({"email": email}, {"$set": {"password": generate_password_hash(password1, method='pbkdf2:sha256')}})
                return jsonify({"message": "Password reset successfully"}), 200
            else:
                return jsonify({"error": "Passwords don't match"}), 400
        else:
            return jsonify({"error": "User does not exist"}), 400
    return jsonify({"message": "Reset password endpoint ready"}), 200


@auth.route("/settings", methods=["PUT"])
@token_required
def settings(current_user_id):
    data = request.get_json()
    user = users_collection.find_one({"_id": ObjectId(current_user_id)})

    email = data.get("email")
    firstName = data.get("firstName")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    if not user:
        return jsonify({"error": "User not found"}), 404

    if email:
        existing_user = users_collection.find_one({"email": email})    
        if existing_user and existing_user["_id"] != ObjectId(current_user_id):
            return jsonify({"error": "Email already exists"}), 400
        user["email"] = email

    if firstName:
        user["firstName"] = firstName

    if password and confirm_password:
        if password == confirm_password:
            user["password"] = generate_password_hash(password, method='pbkdf2:sha256')
        else:
            return jsonify({"error": "Passwords don't match"}), 400
        
    users_collection.update_one({"_id": ObjectId(current_user_id)}, {"$set": user})
    return jsonify({"message": "User updated successfully"}), 200