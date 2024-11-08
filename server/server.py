from website import create_app
from flask import Flask
from flask_socketio import SocketIO, send, emit, join_room, leave_room

app = create_app()
socketio = SocketIO(app)


@socketio.on('session_created')
def handleSessionCreated(data):
    app.logger.info('Session Created: ' + data.get("session"))
    session_id = data.get("session_id")
    if session_id:
        join_room(session_id)
        emit('session_created_announcement', data, room=session_id)
    else:
        print("Error: session_id is missing in the data")

@socketio.on('user_joined')
def handleUserJoined(data):
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    if session_id and user_id:
        join_room(session_id)
        emit('user_joined_announcement', data, room=session_id)
    else:
        print("Error: session_id or user_id is missing in the data")

@socketio.on('session_updated')
def handleSessionUpdated(data):
    session_id = data.get("session_id")
    if session_id:
        emit('session_updated_announcement', data, room=session_id)
    else:
        print("Error: session_id is missing in the data")

@socketio.on('session_deleted')
def handleSessionDeleted(data):
    session_id = data.get("session_id")
    if session_id:
        emit('session_deleted_announcement', data, room=session_id)
        leave_room(session_id)
    else:
        print("Error: session_id is missing in the data")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000) 