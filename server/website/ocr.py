from flask import Flask, request, Blueprint, jsonify

ocr = Blueprint('ocr', __name__)

@ocr.route('/get-ocr', methods=['GET'])
def get_ocr():
    user_id = request.args.get('userId')
    # Logic for OCR
    # Placeholder response for now
    return jsonify({'ocr': 'Text extracted from image'}), 200