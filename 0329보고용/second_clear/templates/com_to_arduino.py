from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello from Flask server'})

if __name__ == '__main__':
    app.run(host='192.168.0.6', port=5000)