from flask import Flask, request, jsonify, Blueprint, url_for
from . import sessions_collection, users_collection
from .auth import token_required
from .recc import get_recommendations
from .ocr import processReceipt
import numpy as np
import requests

views = Blueprint('views', __name__)


@views.route('/')
@token_required
def index(current_user_id):
    user_id = str(current_user_id)

    previous_sessions = list(sessions_collection
                             .find({"participants": user_id})
                             .sort("created_at", -1)
                             .limit(4)
                        )
    new_user = np.array([[3, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0, 0, 0, 2.0, 0.6666666666666666, 0.6666666666666666]])
    recommendations = get_recommendations(new_user)

    return jsonify({"recommendations": recommendations, "previous_sessions": previous_sessions}), 200


@views.route('/profile', methods=['GET'])
@token_required
def get_user(current_user_id):
    user_id = str(current_user_id)
    user = users_collection.find_one({"_id": user_id})
    return jsonify({"user": user}), 200

@views.route("/scan_image")
@token_required
def scan_image():
    return jsonify({"message": "Scan image endpoint ready"}), 200

@views.route("/upload_image", methods=["POST"])
@token_required
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image = request.files['image']
    receipt_text = processReceipt(image)

    session_data = {
        "restaurantName": "sample restaurant",
        "receipt": receipt_text
    }
    session_url = url_for("sessions.create_session", _external=True)
    headers = {'x-access-token': request.headers.get("x-access-token")}
    response = requests.post(session_url, json=session_data, headers=headers)

    if response.status_code == 201:
        session_id = response.json().get('session_id')
        return jsonify({"message": "Image uploaded successfully", "session_id": session_id}), 200
    else:
        return jsonify({"error": "Failed to upload image"}), 400
