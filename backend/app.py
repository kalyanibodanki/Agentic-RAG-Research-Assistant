
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from routes.upload import upload_bp
from routes.ask import ask_bp

load_dotenv()

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Register both blueprints
app.register_blueprint(upload_bp)
app.register_blueprint(ask_bp)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to Agentic RAG Scientific Research Paper Assistant!",
        "status": "Backend is running"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Flask backend is working correctly"
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        port=5000
    )
