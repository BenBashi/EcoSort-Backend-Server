# main.py
from flask import Flask, send_from_directory
from flask_cors import CORS
from api.dashboard_api import dashboard_bp
from api.home_api import home_bp


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(dashboard_bp, url_prefix="/dashboard")
    app.register_blueprint(home_bp, url_prefix="/home")

    @app.route("/")
    def home():
        return "Ecosort Flask Backend Server"

    # Serve files from the images folder
    @app.route('/images/<path:filename>')
    def serve_image(filename):
        return send_from_directory('images', filename)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5050)
