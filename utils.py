import os
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv, find_dotenv
from flask import Flask, redirect, jsonify, request, session, url_for

# Load environment variables from .env file
load_dotenv(find_dotenv())

app = Flask(__name__)
app.secret_key = os.urandom(24)  # In production, use a fixed secret key

PORT = 65010

# Zoom OAuth endpoints and configuration
ZOOM_OAUTH_AUTHORIZE_API = "https://zoom.us/oauth/authorize"
ZOOM_TOKEN_API = "https://zoom.us/oauth/token"

CLIENT_ID = "FiyFvBUSSeeXwjDv0tqg"
CLIENT_SECRET = "tygAN91Xd7Wo1YAH056wtbrXQ8I6UieA"
# Use a consistent environment variable for your redirect URI; fallback to localhost if not set
REDIRECT_URI = "https://clear-muskox-grand.ngrok-free.app/zoom_callback"

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing Zoom OAuth credentials. Please set ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET.")

@app.route("/")
def index():
    """Homepage that redirects to the login route."""
    return redirect(url_for("login"))

@app.route("/login")
def login():
    """Initiate the Zoom OAuth flow by redirecting the user to Zoom's authorization page."""
    # Build the authorization URL with URL-encoded redirect URI
    auth_url = (
        f"{ZOOM_OAUTH_AUTHORIZE_API}"
        f"?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={quote_plus('https://clear-muskox-grand.ngrok-free.app/zoom_callback')}"
    )
    return redirect(auth_url)

@app.route("/zoom_callback")
def zoom_callback():
    """Handles the OAuth callback by exchanging the authorization code for an access token."""
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "No authorization code received"}), 400

    # Prepare token request parameters
    params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI
    }

    try:
        # Exchange the authorization code for an access token
        response = requests.post(ZOOM_TOKEN_API, params=params, auth=(CLIENT_ID, CLIENT_SECRET))
    except Exception as e:
        return jsonify({"error": f"Token request failed: {str(e)}"}), 500

    if response.status_code == 200:
        token_data = response.json()
        # Optionally store tokens in session for later use
        session["access_token"] = token_data.get("access_token")
        session["refresh_token"] = token_data.get("refresh_token")
        # Return the token details as a JSON response
        return jsonify(token_data)
    else:
        return jsonify({"error": "Failed to retrieve token", "details": response.text}), response.status_code

if __name__ == '__main__':
    app.run(port=PORT)