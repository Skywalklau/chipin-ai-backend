from flask import Flask, request, jsonify, Blueprint

views = Blueprint('views', __name__)


@views.route('/')
def index():
    return jsonify({"message": "Hello world"})