from flask import Flask, jsonify, request, Blueprint
from . import mongo, client, db, users_collection, sessions_collection, mail, socketio
import datetime
from bson.objectid import ObjectId
from flask_login import login_user, login_required, logout_user, current_user
from urllib.parse import urlencode, urljoin
from .auth import token_required
from flask_socketio import emit 

sessions = Blueprint('sessions', __name__)


@sessions.route('/create_session', methods=['POST'])
@token_required
def create_session(current_user_id):
    data = request.get_json()
    restaurant_details = data.get("restaurantDetails")
    session_name = f"{datetime.datetime.now()} - {data.get('restaurantName')}"
    session_positions = []
    total = 0
    admin_id = str(current_user_id)
    created_at = str(datetime.datetime.now())
    receipt = data.get("receipt")
    status = "active"
    participants = [str(current_user_id)]
    session = {
        "session_name": session_name,
        "session_positions": session_positions,
        "total": float(total),
        "admin_id": admin_id,
        "created_at": created_at,
        "receipt": receipt,
        "status": status,
        "participants": participants
    }
    session_id = str(sessions_collection.insert_one(session).inserted_id)      
    socketio.emit('session_created', {"session": dict(session), "session_id": session_id})

    socketio.emit("user_joined", {"session_id": session_id, "user_id": current_user_id})
    session["_id"] = str(session_id)    

    return jsonify({"message": "Session created", "session_id": session_id, "session": dict(session), "restaurantDetails": restaurant_details}), 201


@sessions.route('/get_old_sessions', methods=['GET'])
@token_required
def get_old_sessions(current_user_id):
    sessions = list(sessions_collection.find({"participants": current_user_id}))

    response_list = []
    for session in sessions:
        session_data = {
            "session_name": session["session_name"],
            "positions": session["session_positions"],
            "total_for_person": float(calculate_total_for_user(current_user_id, session)),
            "total": float(session["total"]),
            "created_at": str(session["created_at"]),
        }
        response_list.append(session_data)
    return jsonify({"sessions_list": response_list}), 200


def calculate_total_for_user(user_id, session):
    total = 0
    for position in session['session_positions']:
        if position['buyer'] == user_id:
            total += position['price']
    return total


@sessions.route('get_session/<session_id>', methods=['GET'])
@token_required
def get_session(current_user_id, session_id):    
    session = sessions_collection.find_one({'_id': ObjectId(session_id)})

    if session:
        session_data = {
            "session_name": session["session_name"],
            "isClosed": session.get("status") == "closed",
            "positions": session["session_positions"],
            "total_for_person": float(calculate_total_for_user(current_user_id, session)),
            "total": float(session["total"]),
            "participants": session["participants"],
            "admin_id": session["admin_id"],
            "created_at": session["created_at"],
            "receipt": session["receipt"]
        }
        return jsonify({"session_data": session_data}), 200
    else:
        return jsonify({"error": "Session not found"}), 404
    

@sessions.route('/join_session', methods=['POST'])
def join_session():
    data = request.get_json()
    session_id = data.get('session_id')
    current_user_id = data.get('user_id')
        
    session = sessions_collection.find_one({'_id': ObjectId(session_id)})
    
    if session:
        if current_user_id in session['participants']:
            return jsonify({'message': 'User is already part of the session'}), 400
                
        sessions_collection.update_one(
            {'_id': ObjectId(session_id)},
            {'$push': {'participants': current_user_id}}
        )
        socketio.emit("user_joined", {"session_id": session_id, "user_id": current_user_id})
        
        return jsonify({'message': 'Joined session successfully'}), 200
    else:
        return jsonify({'message': 'Session not found'}), 404
    

@sessions.route('/update_session', methods=['PUT'])
@token_required
def update_session(current_user_id):
    data = request.get_json()
    session_id = data.get("sessionId")
    position_index = int(data.get("positionIndex"))
    item_name = data.get("itemName")
    amount = int(data.get("amount"))
    session = sessions_collection.find_one({'_id': ObjectId(session_id)})
    print(session_id)
    if session:        
        positions = session.get("session_positions")
        if len(positions) == 0:
            # initialise positions
            positions = [{"buyer": current_user_id, "item_name": item_name, "price": amount}]            
        elif position_index < 0 or position_index >= len(positions):
            return jsonify({"error": "Position not found"}), 404
        
        positions[position_index]["buyer"] = current_user_id
        positions[position_index]["item_name"] = item_name
        positions[position_index]["price"] = amount
        total = sum([position["price"] for position in positions])
        sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"session_positions": positions, "total": total}}
        )
        socketio.emit("session_updated", {"session_id": session_id, "positions": positions, "total": total})
        return jsonify({"message": "Session updated successfully"}), 200
    else:
        return jsonify({"error": "Session not found"}), 404


@sessions.route('/delete_session/<session_id>', methods=['DELETE'])
@token_required
def delete_session(current_user_id, session_id):    
    session = sessions_collection.find_one({"_id": ObjectId(session_id)})

    if session:
        # Check if the current user is the session admin
        if session["admin_id"] == current_user_id:
            # Delete the session
            sessions_collection.delete_one({"_id": ObjectId(session_id)})            
            socketio.emit("session_deleted", {"session_id": session_id})
            return jsonify({"message": "Session deleted successfully"}), 200
        else:
            return jsonify({"error": "Only the session admin can delete the session"}), 403
    else:
        return jsonify({"error": "Session not found"}), 404
    
@sessions.route('create_link/<session_id>', methods=['GET'])
@token_required
def create_link(current_user_id, session_id):
    session = sessions_collection.find_one({'_id': ObjectId(session_id)})

    if session:
        if current_user_id == session['admin_id']:
            base_url = request.host_url
            join_path = f"join_link/{session_id}"
            link = urljoin(base_url, join_path)
            return jsonify({"message": "Link created successfully", "link": link}), 200
        else:
            return jsonify({"error": "Only the session admin can create the link"}), 403
    else:
        return jsonify({"error": "Session not found"}), 404
    

@sessions.route('join_link/<session_id>', methods=['GET'])
def join_link(session_id):
    data = request.get_json()
    current_user_id = data.get('user_id')
    session = sessions_collection.find_one({'_id': ObjectId(session_id)})

    if session:
        if current_user_id in session['participants']:
            return jsonify({'message': 'User is already part of the session'}), 400
                
        sessions_collection.update_one(
            {'_id': ObjectId(session_id)},
            {'$push': {'participants': current_user_id}}
        )
        socketio.emit("user_joined", {"session_id": session_id, "user_id": current_user_id})
        return jsonify({'message': 'Joined session successfully'}), 200
    else:
        return jsonify({'message': 'Session not found'}), 404