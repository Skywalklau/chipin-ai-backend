from website import create_app
from flask import Flask
from flask_socketio import SocketIO, send, emit, join_room

app = create_app()
socketio = SocketIO(app)

@socketio.on('message')
def handleMessage(msg):
    print('Message: ' + msg)
    send(msg, broadcast=True)

@socketio.on('json')
def handleJson(json):
    print('json: ' + str(json))
    emit('json', json, broadcast=True)

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
    print('User Joined: ' + str(data))
    emit('user_joined', data, broadcast=True)

@socketio.on('session_updated')
def handleSessionUpdated(data):
    print('Session Updated: ' + str(data))
    emit('session_updated', data, broadcast=True)

@socketio.on('session_deleted')
def handleSessionDeleted(data):
    print('Session Deleted: ' + str(data))
    emit('session_deleted', data, broadcast=True)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000) 