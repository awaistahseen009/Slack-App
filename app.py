import os
import secrets
import time
import json
import base64
from flask_talisman import Talisman
from flask import Flask, request, jsonify,render_template
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.oauth.oauth_settings import OAuthSettings
from dotenv import find_dotenv, load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from flask_session import Session
from msal import ConfidentialClientApplication
import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import re
import logging
from threading import Lock

from urllib.parse import quote_plus
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from agents.all_agents import (
    create_schedule_agent, create_update_agent, create_delete_agent, llm,
    create_schedule_group_agent, create_update_group_agent, create_schedule_channel_agent
)
from all_tools import tools, calendar_prompt_tools
from db import init_db

# Load environment variables
load_dotenv(find_dotenv())
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
os.environ['OAUTHLIB_IGNORE_SCOPE_CHANGE'] = '1'


user_cache = {}
user_cache_lock = Lock()  # Example threading lock for cache
preferences_cache = {}
preferences_cache_lock = Lock()
owner_id_cache = {}
owner_id_lock = Lock()
# Configuration
SLACK_CLIENT_ID = os.getenv('SLACK_CLIENT_ID','')
SLACK_CLIENT_SECRET = os.getenv('SLACK_CLIENT_SECRET','')
SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET','')
SLACK_SCOPES = [
    "app_mentions:read",
    "channels:history",
    "chat:write",
    "users:read",
    "im:write",
    "groups:write",
    "mpim:write",
    "commands",
    "team:read",
    "channels:read",
    "groups:read",
    "im:read",
    "mpim:read",
    "groups:history",
    "im:history",
    "mpim:history"
]
import requests
SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")
ZOOM_REDIRECT_URI = "https://clear-muskox-grand.ngrok-free.app/zoom_callback"
CLIENT_ID = "FiyFvBUSSeeXwjDv0tqg"  # Zoom Client ID
CLIENT_SECRET = "tygAN91Xd7Wo1YAH056wtbrXQ8I6UieA"  # Zoom Client Secret
ZOOM_TOKEN_API = "https://zoom.us/oauth/token"
ZOOM_OAUTH_AUTHORIZE_API = os.getenv("ZOOM_OAUTH_AUTHORIZE_API", "https://zoom.us/oauth/authorize")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "https://clear-muskox-grand.ngrok-free.app/oauth2callback")
MICROSOFT_CLIENT_ID = "855e4571-d92a-4d51-802e-e712a879c00b"
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
MICROSOFT_AUTHORITY = "https://login.microsoftonline.com/common"
MICROSOFT_SCOPES = ["User.Read", "Calendars.ReadWrite"]
MICROSOFT_REDIRECT_URI = os.getenv("MICROSOFT_REDIRECT_URI", "https://clear-muskox-grand.ngrok-free.app/microsoft_callback")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
talisman = Talisman(
    app,
    content_security_policy={
        'default-src': "'self'",
        'script-src': "'self'",
        'object-src': "'none'"
    },
    force_https=True,
    strict_transport_security=True,
    strict_transport_security_max_age=31536000,
    x_content_type_options=True,
    referrer_policy='no-referrer-when-downgrade'
)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Installation Store for OAuth
import json
import os
import psycopg2
from datetime import datetime

import json
import os
import psycopg2
from datetime import datetime
from slack_sdk.oauth import InstallationStore
# Custom JSON encoder to handle datetime objects
import json
import os
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import logging
from slack_sdk import WebClient
from slack_sdk.oauth import InstallationStore
from slack_sdk.oauth.installation_store.models.installation import Installation
from slack_bolt.authorization import AuthorizeResult

# Custom JSON encoder for datetime objects (used only if needed)
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class DatabaseInstallationStore(InstallationStore):
    """A database-backed installation store for Slack Bolt using PostgreSQL.
    
    Assumes 'installation_data' is a jsonb column storing JSON data.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def save(self, installation):
        try:
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            cur = conn.cursor()
            
            workspace_id = installation.team_id
            installed_at = datetime.fromtimestamp(installation.installed_at) if installation.installed_at else None
            
            installation_data = {
                "team_id": installation.team_id,
                "enterprise_id": installation.enterprise_id,
                "user_id": installation.user_id,
                "bot_token": installation.bot_token,
                "bot_id": installation.bot_id,
                "bot_user_id": installation.bot_user_id,
                "bot_scopes": installation.bot_scopes,
                "user_token": installation.user_token,
                "user_scopes": installation.user_scopes,
                "incoming_webhook_url": installation.incoming_webhook_url,
                "incoming_webhook_channel": installation.incoming_webhook_channel,
                "incoming_webhook_channel_id": installation.incoming_webhook_channel_id,
                "incoming_webhook_configuration_url": installation.incoming_webhook_configuration_url,
                "app_id": installation.app_id,
                "token_type": installation.token_type,
                "installed_at": installed_at.isoformat() if installed_at else None
            }
            
            current_time = datetime.now()
            
            cur.execute('''
                INSERT INTO Installations (workspace_id, installation_data, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (workspace_id) DO UPDATE SET
                    installation_data = %s, updated_at = %s
            ''', (workspace_id, Json(installation_data), current_time, Json(installation_data), current_time))
            
            conn.commit()
            self._logger.info(f"Saved installation for workspace {workspace_id}")
        
        except Exception as e:
            self._logger.error(f"Failed to save installation for workspace {workspace_id}: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def find_installation(self, enterprise_id=None, team_id=None, user_id=None, is_enterprise_install=False):
        if not team_id:
            self._logger.warning("No team_id provided for find_installation")
            return None
        
        try:
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            cur = conn.cursor()
            cur.execute('SELECT installation_data FROM Installations WHERE workspace_id = %s', (team_id,))
            row = cur.fetchone()
            
            if row:
                # For jsonb, row[0] is already a dict
                installation_data = row[0]
                installed_at = (datetime.fromisoformat(installation_data["installed_at"])
                                if installation_data.get("installed_at") else None)
                
                return Installation(
                    app_id=installation_data["app_id"],
                    enterprise_id=installation_data.get("enterprise_id"),
                    team_id=installation_data["team_id"],
                    bot_token=installation_data["bot_token"],
                    bot_id=installation_data["bot_id"],
                    bot_user_id=installation_data["bot_user_id"],
                    bot_scopes=installation_data["bot_scopes"],
                    user_id=installation_data["user_id"],
                    user_token=installation_data.get("user_token"),
                    user_scopes=installation_data.get("user_scopes"),
                    incoming_webhook_url=installation_data.get("incoming_webhook_url"),
                    incoming_webhook_channel=installation_data.get("incoming_webhook_channel"),
                    incoming_webhook_channel_id=installation_data.get("incoming_webhook_channel_id"),
                    incoming_webhook_configuration_url=installation_data.get("incoming_webhook_configuration_url"),
                    token_type=installation_data["token_type"],
                    installed_at=installed_at
                )
            else:
                self._logger.info(f"No installation found for team_id {team_id}")
                return None
        
        except Exception as e:
            self._logger.error(f"Error retrieving installation for team_id {team_id}: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def find_bot(self, enterprise_id=None, team_id=None, is_enterprise_install=False):
        if not team_id:
            self._logger.warning("No team_id provided for find_bot")
            return None
        
        try:
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            cur = conn.cursor()
            cur.execute('SELECT installation_data FROM Installations WHERE workspace_id = %s', (team_id,))
            row = cur.fetchone()
            
            if row:
                installation_data = row[0]
                return AuthorizeResult(
                    enterprise_id=installation_data.get("enterprise_id"),
                    team_id=installation_data["team_id"],
                    bot_token=installation_data["bot_token"],
                    bot_id=installation_data["bot_id"],
                    bot_user_id=installation_data["bot_user_id"]
                )
            else:
                self._logger.info(f"No bot installation found for team_id {team_id}")
                return None
        
        except Exception as e:
            self._logger.error(f"Error retrieving bot for team_id {team_id}: {e}")
            return None
        finally:
            cur.close()
            conn.close()

# Instantiate the store
installation_store = DatabaseInstallationStore()

def get_client_for_team(team_id):
    """
    Get a Slack WebClient for a given team ID using the stored bot token.

    Args:
        team_id (str): The team ID (workspace ID) to look up.

    Returns:
        WebClient: Slack client instance or None if not found.
    """
    installation = installation_store.find_installation(None, team_id)
    if installation:
        token = installation.bot_token  # Use dot notation instead of subscripting
        return WebClient(token=token)
    return None



# Initialize Slack Bolt app with OAuth settings
oauth_settings = OAuthSettings(
    client_id=SLACK_CLIENT_ID,
    client_secret=SLACK_CLIENT_SECRET,
    scopes=SLACK_SCOPES,
    redirect_uri="https://clear-muskox-grand.ngrok-free.app/slack/oauth_redirect",
    installation_store=installation_store
)
bolt_app = App(signing_secret=SLACK_SIGNING_SECRET, oauth_settings=oauth_settings)
slack_handler = SlackRequestHandler(bolt_app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Calendar API Scopes
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# Initialize Neon Postgres database
init_db()

# State Management Classes
class StateManager:
    def __init__(self):
        self._states = {}
        self._lock = Lock()

    def create_state(self, user_id):
        with self._lock:
            state_token = secrets.token_urlsafe(32)
            self._states[state_token] = {"user_id": user_id, "timestamp": datetime.now(), "used": False}
            return state_token

    def validate_and_consume_state(self, state_token):
        with self._lock:
            if state_token not in self._states:
                return None
            state_data = self._states[state_token]
            if state_data["used"] or (datetime.now() - state_data["timestamp"]).total_seconds() > 600:
                del self._states[state_token]
                return None
            state_data["used"] = True
            return state_data["user_id"]

    def cleanup_expired_states(self):
        with self._lock:
            current_time = datetime.now()
            expired = [s for s, d in self._states.items() if (current_time - d["timestamp"]).total_seconds() > 600]
            for state in expired:
                del self._states[state]

state_manager = StateManager()

class EventDeduplicator:
    def __init__(self, expiration_minutes=5):
        self.processed_events = defaultdict(list)
        self.expiration_minutes = expiration_minutes

    def clean_expired_events(self):
        current_time = datetime.now()
        for event_id in list(self.processed_events.keys()):
            events = [(t, h) for t, h in self.processed_events[event_id]
                      if current_time - t < timedelta(minutes=self.expiration_minutes)]
            if events:
                self.processed_events[event_id] = events
            else:
                del self.processed_events[event_id]

    def is_duplicate_event(self, event_payload):
        self.clean_expired_events()
        event_id = event_payload.get('event_id', '')
        payload_hash = hashlib.md5(str(event_payload).encode('utf-8')).hexdigest()
        if 'challenge' in event_payload:
            return False
        if event_id in self.processed_events and payload_hash in [h for _, h in self.processed_events[event_id]]:
            return True
        self.processed_events[event_id].append((datetime.now(), payload_hash))
        return False

event_deduplicator = EventDeduplicator()

class SessionStore:
    def __init__(self):
        self._store = {}
        self._lock = Lock()

    def set(self, user_id, key, value):
        with self._lock:
            if user_id not in self._store:
                self._store[user_id] = {}
            self._store[user_id][key] = {"value": value, "expires_at": datetime.now() + timedelta(hours=1)}

    def get(self, user_id, key, default=None):
        with self._lock:
            if user_id not in self._store or key not in self._store[user_id]:
                return default
            session_data = self._store[user_id][key]
            if datetime.now() > session_data["expires_at"]:
                del self._store[user_id][key]
                return default
            return session_data["value"]

    def clear(self, user_id, key):
        with self._lock:
            if user_id in self._store and key in self._store[user_id]:
                del self._store[user_id][key]

session_store = SessionStore()

def store_in_session(user_id, key_type, data):
    session_store.set(user_id, key_type, data)

def get_from_session(user_id, key_type, default=None):
    return session_store.get(user_id, key_type, default)

# Global Caches (per workspace)
user_cache = {}  # {team_id: {user_id: user_data}}
user_cache_lock = Lock()

owner_id_cache = {}  # {team_id: owner_id}
owner_id_lock = Lock()

preferences_cache = {}
preferences_cache_lock = Lock()

# Database Helper Functions
def save_preference(team_id, user_id, zoom_config=None, calendar_tool=None):
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute('SELECT zoom_config, calendar_tool FROM Preferences WHERE team_id = %s AND user_id = %s', (team_id, user_id))
    existing = cur.fetchone()
    if existing:
        current_zoom_config, current_calendar_tool = existing
        new_zoom_config = zoom_config if zoom_config is not None else current_zoom_config
        new_calendar_tool = calendar_tool if calendar_tool is not None else current_calendar_tool
        cur.execute('''
            UPDATE Preferences 
            SET zoom_config = %s, calendar_tool = %s, updated_at = %s
            WHERE team_id = %s AND user_id = %s
        ''', (json.dumps(new_zoom_config) if new_zoom_config else None, 
              new_calendar_tool, datetime.now(), team_id, user_id))
    else:
        new_zoom_config = zoom_config or {"mode": "manual", "link": None}
        new_calendar_tool = calendar_tool or "google"
        cur.execute('''
            INSERT INTO Preferences (team_id, user_id, zoom_config, calendar_tool, updated_at)
            VALUES (%s, %s, %s, %s, %s)
        ''', (team_id, user_id, json.dumps(new_zoom_config), new_calendar_tool, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()
    with preferences_cache_lock:
        preferences_cache[(team_id, user_id)] = {"zoom_config": new_zoom_config, "calendar_tool": new_calendar_tool}

def load_preferences(team_id, user_id):
    with preferences_cache_lock:
        if (team_id, user_id) in preferences_cache:
            return preferences_cache[(team_id, user_id)]
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()
        cur.execute('SELECT zoom_config, calendar_tool FROM Preferences WHERE team_id = %s AND user_id = %s', (team_id, user_id))
        row = cur.fetchone()
        if row:
            zoom_config, calendar_tool = row
            # For jsonb, zoom_config is already a dict; no json.loads needed
            preferences = {
                "zoom_config": zoom_config if zoom_config else {"mode": "manual", "link": None},
                "calendar_tool": calendar_tool or "none"
            }
        else:
            preferences = {"zoom_config": {"mode": "manual", "link": None}, "calendar_tool": "none"}
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load preferences for team {team_id}, user {user_id}: {e}")
        preferences = {"zoom_config": {"mode": "manual", "link": None}, "calendar_tool": "none"}
    with preferences_cache_lock:
        preferences_cache[(team_id, user_id)] = preferences
    return preferences

def save_token(team_id, user_id, service, token_data):
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO Tokens (team_id, user_id, service, token_data, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (team_id, user_id, service) DO UPDATE SET token_data = %s, updated_at = %s
    ''', (team_id, user_id, service, json.dumps(token_data), datetime.now(), json.dumps(token_data), datetime.now()))
    conn.commit()
    cur.close()
    conn.close()

def load_token(team_id, user_id, service):
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute('SELECT token_data FROM Tokens WHERE team_id = %s AND user_id = %s AND service = %s', (team_id, user_id, service))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None

# Utility Functions
def initialize_workspace_cache(client, team_id):
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute('SELECT MAX(last_updated) FROM Users WHERE team_id = %s', (team_id,))
    last_updated_row = cur.fetchone()
    last_updated = last_updated_row[0] if last_updated_row and last_updated_row[0] else None
    
    # Check if cache is fresh (e.g., less than 24 hours old)
    if last_updated and (datetime.now() - last_updated).total_seconds() < 86400:
        cur.execute('SELECT user_id, real_name, email, name, is_owner, workspace_name FROM Users WHERE team_id = %s', (team_id,))
        rows = cur.fetchall()
        new_cache = {row[0]: {"real_name": row[1], "email": row[2], "name": row[3], "is_owner": row[4], "workspace_name": row[5]} for row in rows}
        with user_cache_lock:
            user_cache[team_id] = new_cache
        with owner_id_lock:
            owner_id_cache[team_id] = next((user_id for user_id, data in new_cache.items() if data['is_owner']), None)
    else:
        # Fetch user data from Slack and update database
        response = client.users_list()
        users = response["members"]
        workspace_name = client.team_info()["team"]["name"]  # Get workspace name from Slack API
        new_cache = {}
        for user in users:
            user_id = user['id']
            profile = user.get('profile', {})
            real_name = profile.get('real_name', 'Unknown')
            name = user.get('name', '')
            email = f"{name}@gmail.com"  # Placeholder; adjust as needed
            is_owner = user.get('is_owner', False)
            new_cache[user_id] = {"real_name": real_name, "email": email, "name": name, "is_owner": is_owner, "workspace_name": workspace_name}
            cur.execute('''
                INSERT INTO Users (team_id, user_id, workspace_name, real_name, email, name, is_owner, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_id, user_id) DO UPDATE SET
                    workspace_name = %s, real_name = %s, email = %s, name = %s, is_owner = %s, last_updated = %s
            ''', (team_id, user_id, workspace_name, real_name, email, name, is_owner, datetime.now(),
                  workspace_name, real_name, email, name, is_owner, datetime.now()))
        conn.commit()
        with user_cache_lock:
            user_cache[team_id] = new_cache
        with owner_id_lock:
            owner_id_cache[team_id] = next((user_id for user_id, data in new_cache.items() if data['is_owner']), None)
    cur.close()
    conn.close()

def get_all_users(team_id):
    with user_cache_lock:
        if team_id in user_cache:
            return {k: {"Slack Id": k, "real_name": v["real_name"], "email": v["email"], "name": v["name"]} 
                    for k, v in user_cache[team_id].items()}
    return {}

def get_workspace_owner_id(client, team_id):
    with owner_id_lock:
        if team_id in owner_id_cache and owner_id_cache[team_id]:
            return owner_id_cache[team_id]
    initialize_workspace_cache(client, team_id)
    with owner_id_lock:
        return owner_id_cache.get(team_id)

def get_channel_owner_id(client, channel_id):
    try:
        response = client.conversations_info(channel=channel_id)
        return response["channel"].get("creator")
    except SlackApiError as e:
        logger.error(f"Error fetching channel info: {e.response['error']}")
        return None

def get_user_timezone(client, user_id):
    try:
        response = client.users_info(user=user_id)
        return response["user"].get("tz", "UTC")
    except SlackApiError as e:
        logger.error(f"Timezone error: {e.response['error']}")
        return "UTC"

def get_team_id_from_owner_id(owner_id):
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute("SELECT workspace_id FROM Installations WHERE installation_data->>'user_id' = %s", (owner_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None

# def get_client_for_team(team_id):
#     installation = installation_store.find_installation(None, team_id)
#     if installation:
#         print(installation)
#         token = installation['bot_token']
#         return WebClient(token=token)
#     return None

def get_owner_selected_calendar(client, team_id):
    owner_id = get_workspace_owner_id(client, team_id)
    if not owner_id:
        return None
    # Fixed: Pass both team_id and owner_id to load_preferences
    prefs = load_preferences(team_id, owner_id)
    return prefs.get("calendar_tool", "none")

def get_zoom_link(client, team_id):
    owner_id = get_workspace_owner_id(client, team_id)
    if not owner_id:
        return None
    prefs = load_preferences(team_id,owner_id)
    return prefs.get('zoom_config', {}).get('link')

def create_home_tab(client, team_id, user_id):
    logger.info(f"Creating home tab for user {user_id}, team {team_id}")
    
    # Get workspace owner ID
    workspace_owner_id = get_workspace_owner_id(client, team_id)
    if not workspace_owner_id:
        logger.warning(f"No workspace owner for team {team_id}")
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "🤖 Welcome to AI Assistant!", "emoji": True}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Unable to determine workspace owner. Please contact support."}},
        ]
        return {"type": "home", "blocks": blocks}

    # Determine if the user is the workspace owner
    is_owner = user_id == workspace_owner_id

    # Base blocks for all users
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "🤖 Welcome to AI Assistant!", "emoji": True}}
    ]

    # Non-owner view
    if not is_owner:
        blocks.extend([
            {"type": "section", "text": {"type": "mrkdwn", "text": "I help manage schedules and meetings! Please wait for the workspace owner to configure the settings."}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Only the workspace owner can configure the calendar and Zoom settings."}}
        ])
        return {"type": "home", "blocks": blocks}

    # Owner view: Add configuration options
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "I help manage schedules and meetings! Your settings are below."}})
    blocks.append({"type": "divider"})

    # Load preferences and tokens
    prefs = load_preferences(team_id, workspace_owner_id)
    selected_provider = prefs.get("calendar_tool", "none")
    zoom_config = prefs.get("zoom_config", {"mode": "manual", "link": None})
    mode = zoom_config["mode"]
    calendar_token = load_token(team_id, workspace_owner_id, selected_provider) if selected_provider != "none" else None
    zoom_token = load_token(team_id, workspace_owner_id, "zoom") if mode == "automatic" else None
    logger.info(f"Preferences loaded: {prefs}, Calendar token: {calendar_token}, Zoom token: {zoom_token}")

    # Check Zoom token expiration
    zoom_token_expired = False
    if zoom_token and mode == "automatic":
        expires_at = zoom_token.get("expires_at", 0)
        current_time = time.time()
        zoom_token_expired = current_time >= expires_at

    # Configuration status
    calendar_provider_set = selected_provider != "none"
    calendar_configured = calendar_token is not None if calendar_provider_set else False
    zoom_configured = (zoom_token is not None and not zoom_token_expired) if mode == "automatic" else True

    # Setup prompt if configurations are incomplete
    if not calendar_provider_set or not calendar_configured or not zoom_configured:
        prompt_text = "To start using the app, please complete the following setups:"
        if not calendar_provider_set:
            prompt_text += "\n- Select a calendar provider."
        if calendar_provider_set and not calendar_configured:
            prompt_text += f"\n- Configure your {selected_provider.capitalize()} calendar."
        if mode == "automatic" and not zoom_configured:
            if zoom_token_expired:
                prompt_text += "\n- Your Zoom token has expired. Please refresh it."
            else:
                prompt_text += "\n- Authenticate with Zoom for automatic mode."
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": prompt_text}})

    # Calendar Configuration Section
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*🗓️ Calendar Configuration*"}})
    blocks.append({
        "type": "section",
        "block_id": "calendar_provider_block",
        "text": {"type": "mrkdwn", "text": "Select your calendar provider:"},
        "accessory": {
            "type": "static_select",
            "action_id": "calendar_provider_dropdown",
            "placeholder": {"type": "plain_text", "text": "Select provider"},
            "options": [
                {"text": {"type": "plain_text", "text": "Select calendar"}, "value": "none"},
                {"text": {"type": "plain_text", "text": "Google Calendar"}, "value": "google"},
                {"text": {"type": "plain_text", "text": "Microsoft Calendar"}, "value": "microsoft"}
            ],
            "initial_option": {
                "text": {"type": "plain_text", "text": "Select calendar" if selected_provider == "none" else
                        "Google Calendar" if selected_provider == "google" else "Microsoft Calendar"},
                "value": selected_provider
            }
        }
    })

    # Calendar configuration prompts
    if selected_provider == "none":
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Please select a calendar provider to begin configuration."}]})
    elif not calendar_configured:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"Please configure your {selected_provider.capitalize()} calendar."}]})

    # Calendar configure button and status
    if selected_provider != "none":
        status = "⚠️ Not Configured" if not calendar_configured else (
            f":white_check_mark: Connected ({calendar_token.get('google_email', 'unknown')})" if selected_provider == "google" else (
                f":white_check_mark: Connected (expires: {datetime.fromtimestamp(int(calendar_token.get('expires_at', 0))).strftime('%Y-%m-%d %H:%M')})" if calendar_token and calendar_token.get('expires_at') else ":white_check_mark: Connected"
            )
        )
        blocks.extend([
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": f"✨ Configure {selected_provider.capitalize()}" if not calendar_configured else f"✅ Reconfigure {selected_provider.capitalize()}",
                            "emoji": True
                        },
                        "action_id": "configure_gcal" if selected_provider == "google" else "configure_mscal"
                    }
                ]
            },
            {"type": "context", "elements": [{"type": "mrkdwn", "text": status}]}
        ])

    # Zoom Configuration Section
    status = ("⌛ Token Expired" if zoom_token_expired else
              "⚠️ Not Configured" if mode == "automatic" and not zoom_configured else
              "✅ Configured")
    blocks.extend([
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*🔗 Zoom Configuration*\nCurrent mode: {mode}\n{status}"}},
        {
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Configure Zoom Settings", "emoji": True}, "action_id": "open_zoom_config_modal"}
            ]
        }
    ])

    # Zoom authentication/refresh button
    if mode == "automatic":
        if not zoom_configured and not zoom_token_expired:
            blocks[-1]["elements"].append({
                "type": "button",
                "text": {"type": "plain_text", "text": "Authenticate with Zoom", "emoji": True},
                "action_id": "configure_zoom"
            })
        elif zoom_token_expired:
            blocks[-1]["elements"].append({
                "type": "button",
                "text": {"type": "plain_text", "text": "Refresh Zoom Token", "emoji": True},
                "action_id": "configure_zoom"  # Same action_id for refresh
            })

    return {"type": "home", "blocks": blocks}

# Intent Classification
intent_prompt = ChatPromptTemplate.from_template("""
You are an intent classification assistant. Based on the user's message and the conversation history, determine the intent of the user's request. The possible intents are: "schedule meeting", "update event", "delete event", or "other". Provide only the intent as your response.
- By looking at the history if someone is confirming or denying the schedule , also categorize it as a "schedule meeting"
- If someone is asking about update the schedule then its "update event"
- If someone is asking about delete the schedule then its "delete event"                                                 
Conversation History:
{history}

User's Message:
{input}
""")
from prompt import calender_prompt, general_prompt
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

mentioned_users_prompt = ChatPromptTemplate.from_template("""
Given the following chat history, identify the Slack user IDs, Names and emails of the users who are mentioned. Mentions can be in the form of <@user_id> (e.g., <@U12345>) or by their names (e.g., "Alice" or "Alice Smith").
- Do not give 'Bob'<@{bob_id}> in mentions
- Exclude the {bob_id}.                                        
# See the history if there is a request for new meeting or request for new schedule just ignore the mentions in the old messages and consider the new mentions in the new request.
All users in the channel:
{user_information}
Format: Slack Id: U3234234 , Name: Alice , Email: alice@gmail.com (map slack ids to the names)
Chat history:
{chat_history}
# Only output the users which are mentioned not all the users from the user-information.
# Only see the latest message for mention information ignore previous ones.                                                        
Please output the user slack IDs of the mentioned users , their names and emails . If no users are mentioned, output "None".
CURRENT_INPUT: {current_input}                                                          
Example: [[SlackId1 , Name1 , Email@gmal.com], [SlackId2, Name2, Email@gmail.com]...]
""")
mentioned_users_chain = LLMChain(llm=llm, prompt=mentioned_users_prompt)

# Slack Event Handlers
@bolt_app.event("app_home_opened")
def handle_app_home_opened(event, client, context):
    user_id = event.get("user")
    team_id = context['team_id']
    if not user_id:
        return
    try:
        client.views_publish(user_id=user_id, view=create_home_tab(client, team_id, user_id))
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")

@bolt_app.action("calendar_provider_dropdown")
def handle_calendar_provider(ack, body, client, logger):
    ack()
    selected_provider = body["actions"][0]["selected_option"]["value"]
    user_id = body["user"]["id"]
    team_id = body["team"]["id"]
    owner_id = get_workspace_owner_id(client, team_id)
    
    if user_id != owner_id:
        client.chat_postMessage(channel=user_id, text="Only the workspace owner can configure the calendar.")
        return
    
    # Corrected line: pass both team_id and owner_id (user_id) parameters
    save_preference(team_id, owner_id, calendar_tool=selected_provider)
    
    client.views_publish(user_id=owner_id, view=create_home_tab(client, team_id, owner_id))
    if selected_provider != "none":
        client.chat_postMessage(channel=owner_id, text=f"Calendar provider updated to {selected_provider.capitalize()}.")
    else:
        client.chat_postMessage(channel=owner_id, text="Calendar provider reset.")
    logger.info(f"Calendar provider updated to {selected_provider} for owner {owner_id}")

@bolt_app.action("configure_gcal")
def handle_gcal_config(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    team_id = body["team"]["id"]
    owner_id = get_workspace_owner_id(client, team_id)
    if user_id != owner_id:
        client.chat_postMessage(channel=user_id, text="Only the workspace owner can configure the calendar.")
        return
    
    # Generate and store the state using StateManager
    state = state_manager.create_state(owner_id)
    print(f"state stored: {state}")
    store_in_session(owner_id, "gcal_state", state)  # Optional: for additional validation
    
    # Set up the OAuth flow and pass the state
    flow = Flow.from_client_secrets_file('credentials.json', scopes=SCOPES, redirect_uri=OAUTH_REDIRECT_URI)
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
        include_granted_scopes='true',
        state=state  # Use the state from StateManager
    )
    
    # Open the modal with the auth URL
    try:
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "title": {"type": "plain_text", "text": "Google Calendar Auth"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Click below to connect Google Calendar:"}},
                    {"type": "actions", "elements": [{"type": "button", "text": {"type": "plain_text", "text": "Connect Google Calendar"}, "url": auth_url, "action_id": "launch_auth"}]}
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening modal: {e}")

@bolt_app.action("configure_mscal")
def handle_mscal_config(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    team_id = body["team"]["id"]
    owner_id = get_workspace_owner_id(client, team_id)
    if user_id != owner_id:
        client.chat_postMessage(channel=user_id, text="Only the workspace owner can configure the calendar.")
        return
    msal_app = ConfidentialClientApplication(MICROSOFT_CLIENT_ID, authority=MICROSOFT_AUTHORITY, client_credential=MICROSOFT_CLIENT_SECRET)
    state = state_manager.create_state(owner_id)
    auth_url = msal_app.get_authorization_request_url(scopes=MICROSOFT_SCOPES, redirect_uri=MICROSOFT_REDIRECT_URI, state=state)
    try:
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "title": {"type": "plain_text", "text": "Microsoft Calendar Auth"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Click below to authenticate with Microsoft:"}},
                    {"type": "actions", "elements": [{"type": "button", "text": {"type": "plain_text", "text": "Connect Microsoft Calendar"}, "url": auth_url, "action_id": "ms_auth_button"}]}
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening Microsoft auth modal: {e}")
@bolt_app.event("app_mention")
def handle_mentions(event, say, client, context):
    if event_deduplicator.is_duplicate_event(event):
        logger.info("Duplicate event detected, skipping processing")
        return

    # Ignore messages from bots
    if event.get("bot_id"):
        logger.info("Ignoring message from bot")
        return

    user_id = event.get("user")
    channel_id = event.get("channel")
    text = event.get("text", "").strip()
    thread_ts = event.get("thread_ts")
    team_id = context['team_id']
    calendar_tool = get_owner_selected_calendar(client, team_id)
    if not calendar_tool or calendar_tool == "none":
        say("The workspace owner has not configured a calendar yet.", thread_ts=thread_ts)
        return

    # Fetch bot_user_id dynamically from installation
    installation = installation_store.find_installation(team_id=team_id)
    if not installation or not installation.bot_user_id:
        logger.error(f"No bot_user_id found for team {team_id}")
        say("Error: Could not determine bot user ID.", thread_ts=thread_ts)
        return
    print(f"App mention events")
    
    bot_user_id = installation.bot_user_id
    print(f"Bot user id: {bot_user_id}")
    mention = f"<@{bot_user_id}>"
    mentions = list(set(re.findall(r'<@(\w+)>', text)))
    if bot_user_id in mentions:
        mentions.remove(bot_user_id)
    text = text.replace(mention, "").strip()

    workspace_owner_id = get_workspace_owner_id(client, team_id)
    timezone = get_user_timezone(client, user_id)
    zoom_link = get_zoom_link(client, team_id)
    zoom_mode = load_preferences(team_id, workspace_owner_id).get("zoom_config", {}).get("mode", "manual")

    channel_history = client.conversations_history(channel=channel_id, limit=2).get("messages", [])
    channel_history = format_channel_history(channel_history)
    intent = intent_chain.run({"history": channel_history, "input": text})

    relevant_user_ids = get_relevant_user_ids(client, channel_id)
    all_users = get_all_users(team_id)
    relevant_users = {uid: all_users.get(uid, {"real_name": "Unknown", "email": "unknown@example.com", "name": "Unknown"})
                      for uid in relevant_user_ids}
    user_information = "\n".join([f"{uid}: Name={info['real_name']}, Email={info['email']}, Slack Name={info['name']}"
                                  for uid, info in relevant_users.items() if uid != bot_user_id])

    print(f"User Information: {user_information}\n\nRelevant Users: {relevant_user_ids}\n\n All users: {all_users}")
    mentioned_users_output = mentioned_users_chain.run({"user_information": user_information, "chat_history": channel_history, "current_input": text, 'bob_id': bot_user_id})
    
    import pytz
    pst = pytz.timezone('America/Los_Angeles')
    current_time_pst = datetime.now(pst)
    formatted_time = current_time_pst.strftime("%Y-%m-%d | %A | %I:%M %p | %Z")

    from all_tools import GoogleCalendarEvents, MicrosoftListCalendarEvents
    if calendar_tool == "google":
        calendar_events = GoogleCalendarEvents()._run(team_id, workspace_owner_id)
        schedule_tools = [tools[i] for i in [0, 1, 4, 6, 12]]
        update_tools = [tools[i] for i in [0, 7, 12]]
        delete_tools = [tools[i] for i in [0, 8, 12]]
    elif calendar_tool == "microsoft":
        calendar_events = MicrosoftListCalendarEvents()._run(team_id, workspace_owner_id)
        schedule_tools = [tools[i] for i in [0, 1, 9, 12]]
        update_tools = [tools[i] for i in [0, 10, 12]]
        delete_tools = [tools[i] for i in [0, 11, 12]]
    else:
        say("Invalid calendar tool configured.", thread_ts=thread_ts)
        return

    calendar_formatting_chain = LLMChain(llm=llm, prompt=calender_prompt)
    output = calendar_formatting_chain.run({'input': calendar_events, 'admin_id': workspace_owner_id, 'date_time': formatted_time})
    print(f"MENTIONED USERS:{mentioned_users_output}")

    agent_input = {
        'input': f"Here is the input by user: {text} and do not mention <@{bot_user_id}> even tho mentioned in history",
        'event_details': str(event),
        'target_user_id': user_id,
        'timezone': timezone,
        'user_id': user_id,
        'admin': workspace_owner_id,
        'zoom_link': zoom_link,
        'zoom_mode': zoom_mode,
        'channel_history': channel_history,
        'user_information': user_information,
        'calendar_tool': calendar_tool,
        'date_time': formatted_time,
        'formatted_calendar': output,
        'team_id': team_id
    }

    mentions = list(set(re.findall(r'<@(\w+)>', text)))
    if bot_user_id in mentions:
        mentions.remove(bot_user_id)

    schedule_group_exec = create_schedule_channel_agent(schedule_tools)
    update_group_exec = create_update_group_agent(update_tools)
    delete_exec = create_delete_agent(delete_tools)

    if intent == "schedule meeting":
        group_agent_input = agent_input.copy()
        group_agent_input['mentioned_users'] = mentioned_users_output
        response = schedule_group_exec.invoke(group_agent_input)
        say(response['output'])
        return
    elif intent == "update event":
        group_agent_input = agent_input.copy()
        group_agent_input['mentioned_users'] = mentioned_users_output
        response = update_group_exec.invoke(group_agent_input)
        say(response['output'])
        return
    elif intent == "delete event":
        response = delete_exec.invoke(agent_input)
        say(response['output'])
        return
    elif intent == "other":
        response = llm.predict(general_prompt.format(input=text, channel_history=channel_history))
        say(response)
        return
    else:
        say("I'm not sure how to handle that request.")
# @bolt_app.event("app_mention")
# def handle_mentions(event, say, client, context):
#     if event_deduplicator.is_duplicate_event(event):
#         logger.info("Duplicate event detected, skipping processing")
#         return

#     # Ignore messages from bots
#     if event.get("bot_id"):
#         logger.info("Ignoring message from bot")
#         return


#     user_id = event.get("user")
#     channel_id = event.get("channel")
#     text = event.get("text", "").strip()
#     thread_ts = event.get("thread_ts")
#     team_id = context['team_id']
#     calendar_tool = get_owner_selected_calendar(client, team_id)
#     if not calendar_tool or calendar_tool == "none":
#         say("The workspace owner has not configured a calendar yet.", thread_ts=thread_ts)
#         return

#     # Fetch bot_user_id dynamically from installation
#     installation = installation_store.find_installation(team_id=team_id)
#     if not installation or not installation.bot_user_id:
#         logger.error(f"No bot_user_id found for team {team_id}")
#         say("Error: Could not determine bot user ID.", thread_ts=thread_ts)
#         return
#     print(f"App mention events")
    
#     bot_user_id = installation.bot_user_id
#     print(f"Bot user id: {bot_user_id}")
#     mention = f"<@{bot_user_id}>"
#     mentions = list(set(re.findall(r'<@(\w+)>', text)))
#     # Use dynamic bot_user_id instead of SLACK_BOT_USER_ID
#     if bot_user_id in mentions:
#         mentions.remove(bot_user_id)
#     text = text.replace(mention, "").strip()

#     workspace_owner_id = get_workspace_owner_id(client, team_id)
#     timezone = get_user_timezone(client, user_id)
#     zoom_link = get_zoom_link(client, team_id)
#     zoom_mode = load_preferences(team_id, workspace_owner_id).get("zoom_config", {}).get("mode", "manual")

#     channel_history = client.conversations_history(channel=channel_id, limit=2).get("messages", [])
#     channel_history = format_channel_history(channel_history)
#     intent = intent_chain.run({"history": channel_history, "input": text})

#     relevant_user_ids = get_relevant_user_ids(client, channel_id)
#     all_users = get_all_users(team_id)
#     relevant_users = {uid: all_users.get(uid, {"real_name": "Unknown", "email": "unknown@example.com", "name": "Unknown"})
#                       for uid in relevant_user_ids}
#     user_information = "\n".join([f"{uid}: Name={info['real_name']}, Email={info['email']}, Slack Name={info['name']}"
#                               for uid, info in relevant_users.items() if uid != bot_user_id])

#     print(f"User Information: {user_information}\n\nRelevant Users: {relevant_user_ids}\n\n All users: {all_users}")
#     mentioned_users_output = mentioned_users_chain.run({"user_information": user_information, "chat_history": channel_history,"current_input":text, 'bob_id':bot_user_id})

#     import pytz
#     pst = pytz.timezone('America/Los_Angeles')
#     current_time_pst = datetime.now(pst)
#     formatted_time = current_time_pst.strftime("%Y-%m-%d | %A | %I:%M %p | %Z")

#     from all_tools import GoogleCalendarEvents, MicrosoftListCalendarEvents
#     if calendar_tool == "google":
#         calendar_events = GoogleCalendarEvents()._run(team_id, workspace_owner_id)
#         schedule_tools = [tools[i] for i in [0, 1, 4, 6, 12]]
#         update_tools = [tools[i] for i in [0, 7, 12]]
#         delete_tools = [tools[i] for i in [0, 8, 12]]
#     elif calendar_tool == "microsoft":
#         calendar_events = MicrosoftListCalendarEvents()._run(team_id, workspace_owner_id)
#         schedule_tools = [tools[i] for i in [0, 1, 9, 12]]
#         update_tools = [tools[i] for i in [0, 10, 12]]
#         delete_tools = [tools[i] for i in [0, 11, 12]]
#     else:
#         say("Invalid calendar tool configured.", thread_ts=thread_ts)
#         return

#     calendar_formatting_chain = LLMChain(llm=llm, prompt=calender_prompt)
#     output = calendar_formatting_chain.run({'input': calendar_events, 'admin_id': workspace_owner_id, 'date_time': formatted_time})
#     print(f"MENTIONED USERS:{mentioned_users_output}")

#     agent_input = {
#     'input': text,
#     'event_details': str(event),
#     'target_user_id': user_id,
#     'timezone': timezone,
#     'user_id': user_id,
#     'admin': workspace_owner_id,
#     'zoom_link': zoom_link,
#     'zoom_mode': zoom_mode,
#     'channel_history': channel_history,
#     'user_information': mentioned_users_output,
#     'calendar_tool': calendar_tool,
#     'date_time': formatted_time,
#     'formatted_calendar': output,
#     'team_id': team_id  # Added
# }

#     mentions = list(set(re.findall(r'<@(\w+)>', text)))
#     if bot_user_id in mentions:
#         mentions.remove(bot_user_id)

#     schedule_group_exec = create_schedule_channel_agent(schedule_tools)
#     update_group_exec = create_update_group_agent(update_tools)
#     delete_exec = create_delete_agent(delete_tools)

#     if intent == "schedule meeting":
#         group_agent_input = agent_input.copy()
#         group_agent_input['mentioned_users'] = "See from the history except 'Bob'"
#         response = schedule_group_exec.invoke(group_agent_input)
#         say(response['output'])
#         return
#     elif intent == "update event":
#         group_agent_input = agent_input.copy()
#         group_agent_input['mentioned_users'] = "See from the history except 'Bob'"
#         response = update_group_exec.invoke(group_agent_input)
#         say(response['output'])
#         return
#     elif intent == "delete event":
#         response = delete_exec.invoke(agent_input)
#         say(response['output'])
#         return
#     elif intent == "other":
#         response = llm.predict(general_prompt.format(input=text, channel_history=channel_history))
#         say(response)
#         return
#     else:
#         say("I'm not sure how to handle that request.")

def format_channel_history(raw_history):
    cleaned_history = []
    for msg in raw_history:
        if 'bot_id' in msg and 'Calendar provider updated' in msg.get('text', ''):
            continue
        sender = msg.get('user', 'Unknown') if 'bot_id' not in msg else msg.get('bot_profile', {}).get('name', 'Bot')
        message_text = msg.get('text', '').strip()
        timestamp = float(msg.get('ts', 0))
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %I:%M %p')
        user_id = msg.get('user', 'N/A')
        team_id = msg.get('team', 'N/A')
        cleaned_history.append({
            'message': message_text,
            'from': sender,
            'timestamp': readable_time,
            'user_team': f"{user_id}/{team_id}"
        })
    formatted_output = ""
    for i, entry in enumerate(cleaned_history, 1):
        formatted_output += f"Message {i}: {entry['message']}\nFrom: {entry['from']}\nTimestamp: {entry['timestamp']}\nUserId/TeamId: {entry['user_team']}\n\n"
    return formatted_output.strip()

def get_relevant_user_ids(client, channel_id):
    try:
        members = []
        cursor = None
        while True:
            response = client.conversations_members(channel=channel_id, limit=10, cursor=cursor)
            if not response["ok"]:
                logger.error(f"Failed to get members for channel {channel_id}: {response['error']}")
                break
            members.extend(response["members"])
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        return members
    except SlackApiError as e:
        logger.error(f"Error getting conversation members: {e}")
        return []

calendar_formatting_chain = LLMChain(llm=llm, prompt=calender_prompt)

@bolt_app.event("message")
def handle_messages(body, say, client, context):
    if event_deduplicator.is_duplicate_event(body):
        logger.info("Duplicate event detected, skipping processing")
        return
    event = body.get("event", {})
    if event.get("bot_id"):
        logger.info("Ignoring message from bot")
        return

    user_id = event.get("user")
    text = event.get("text", "").strip()
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts")
    team_id = context['team_id']
    calendar_tool = get_owner_selected_calendar(client, team_id)
    channel_info = client.conversations_info(channel=channel_id)
    channel = channel_info["channel"]

    # Fetch bot_user_id dynamically from installation
    installation = installation_store.find_installation(team_id=team_id)
    if not installation or not installation.bot_user_id:
        logger.error(f"No bot_user_id found for team {team_id}")
        say("Error: Could not determine bot user ID.", thread_ts=thread_ts)
        return
    bot_user_id = installation.bot_user_id

    if not channel.get("is_im") and f"<@{bot_user_id}>" in text:
        return
    if not channel.get("is_im") and "thread_ts" not in event:
        return
    if not calendar_tool or calendar_tool == "none":
        say("The workspace owner has not configured a calendar yet.", thread_ts=thread_ts)
        return
    print(f"Message events")
    workspace_owner_id = get_workspace_owner_id(client, team_id)
    is_owner = user_id == workspace_owner_id
    timezone = get_user_timezone(client, user_id)
    zoom_link = get_zoom_link(client, team_id)
    zoom_mode = load_preferences(team_id, workspace_owner_id).get("zoom_config", {}).get("mode", "manual")
    channel_history = client.conversations_history(channel=channel_id, limit=2).get("messages", [])
    channel_history = format_channel_history(channel_history)
    intent = intent_chain.run({"history": channel_history, "input": text})

    if intent == "schedule meeting" and not is_owner and not channel.get("is_group") and not channel.get("is_mpim") and 'thread_ts' not in event:
        admin_dm = client.conversations_open(users=workspace_owner_id)
        prompt = ChatPromptTemplate.from_template("""
        You have this text: {text} and your job is to mention @{workspace_owner_id} and say following in 2 scenarios:
        if message history confirms about scheduling meeting then format below text and return only that response with no other explanation
        "Hi {workspace_owner_id} you wanted to schedule a meeting with {user_id}, {user_id} has proposed these slots [Time slots from the text] , Are you comfortable with these slots ? Confirm so I can fix the meeting."
        else:
            Format the text : {text}
        MESSAGE HISTORY: {channel_history}
        """)
        response = LLMChain(llm=llm, prompt=prompt)
        client.chat_postMessage(channel=admin_dm["channel"]["id"],
                                text=response.run({'text': text, 'workspace_owner_id': workspace_owner_id, 'user_id': user_id, 'channel_history': channel_history}))
        say(f"<@{user_id}> I've notified the workspace owner about your meeting request.", thread_ts=thread_ts)
        return

    mentions = list(set(re.findall(r'<@(\w+)>', text)))
    if bot_user_id in mentions:
        mentions.remove(bot_user_id)

    import pytz
    pst = pytz.timezone('America/Los_Angeles')
    current_time_pst = datetime.now(pst)
    formatted_time = current_time_pst.strftime("%Y-%m-%d | %A | %I:%M %p | %Z")

    from all_tools import MicrosoftListCalendarEvents, GoogleCalendarEvents
    if calendar_tool == "google":
        schedule_tools = [tools[i] for i in [0, 1, 4, 6, 12]]
        update_tools = [tools[i] for i in [0, 7, 12]]
        delete_tools = [tools[i] for i in [0, 8, 12]]
        output = calendar_formatting_chain.run({'input': GoogleCalendarEvents()._run(team_id, workspace_owner_id), 'admin_id': workspace_owner_id, 'date_time': formatted_time})
    elif calendar_tool == "microsoft":
        schedule_tools = [tools[i] for i in [0, 1, 9, 12]]
        update_tools = [tools[i] for i in [0, 10, 12]]
        delete_tools = [tools[i] for i in [0, 11, 12]]
        output = calendar_formatting_chain.run({'input': MicrosoftListCalendarEvents()._run(team_id, workspace_owner_id), 'admin_id': workspace_owner_id, 'date_time': formatted_time})
    else:
        say("Invalid calendar tool configured.", thread_ts=thread_ts)
        return

    relevant_user_ids = get_relevant_user_ids(client, channel_id)
    all_users = get_all_users(team_id)
    relevant_users = {uid: all_users.get(uid, {"real_name": "Unknown", "email": "unknown@example.com", "name": "Unknown"})
                      for uid in relevant_user_ids}
    user_information = "\n".join([f"{uid}: Name={info['real_name']}, Email={info['email']}, Slack Name={info['name']}"
                                  for uid, info in relevant_users.items() if uid != bot_user_id])
    print(f"All users: {all_users}\n\n Relevant users: {relevant_user_ids}")
    mentioned_users_output = mentioned_users_chain.run({"user_information": user_information, "chat_history": channel_history, "current_input": text, 'bob_id': bot_user_id})
    schedule_exec = create_schedule_agent(schedule_tools)
    update_exec = create_update_agent(update_tools)
    delete_exec = create_delete_agent(delete_tools)
    schedule_group_exec = create_schedule_group_agent(schedule_tools)
    update_group_exec = create_update_group_agent(update_tools)
    print(f"MENTIONED USERS:{mentioned_users_output}")
    channel_type = channel.get("is_group", False) or channel.get("is_mpim", False)
    agent_input = {
        'input': f"Here is the input by user: {text} and do not mention <@{bot_user_id}> even tho mentioned in history",
        'event_details': str(event),
        'target_user_id': user_id,
        'timezone': timezone,
        'user_id': user_id,
        'admin': workspace_owner_id,
        'zoom_link': zoom_link,
        'zoom_mode': zoom_mode,
        'channel_history': channel_history,
        'user_information': user_information,
        'calendar_tool': calendar_tool,
        'date_time': formatted_time,
        'formatted_calendar': output,
        'team_id': team_id
    }

    if intent == "schedule meeting":
        if not channel_type and len(mentions) > 1:
            mentions.append(user_id)
            dm_channel_id, error = open_group_dm(client, mentions)
            if dm_channel_id:
                group_agent_input = agent_input.copy()
                group_agent_input['mentioned_users'] = mentioned_users_output
                group_agent_input['channel_history'] = channel_history
                group_agent_input['formatted_calendar'] = output
                response = schedule_group_exec.invoke(group_agent_input)
                client.chat_postMessage(channel=dm_channel_id, text=f"Group conversation started by <@{user_id}>\n\n{response['output']}")
            elif error:
                say(f"Sorry, I couldn't create the group conversation: {error}", thread_ts=thread_ts)
        else:
            if channel_type or 'thread_ts' in event:
                group_agent_input = agent_input.copy()
                if 'thread_ts' in event:
                    schedule_group_exec = create_schedule_channel_agent(schedule_tools)
                    history_response = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=2)
                    channel_history = format_channel_history(history_response.get("messages", []))
                else:
                    channel_history = format_channel_history(client.conversations_history(channel=channel_id, limit=3).get("messages", []))
                group_agent_input['mentioned_users'] = mentioned_users_output
                group_agent_input['channel_history'] = channel_history
                group_agent_input['formatted_calendar'] = output
                response = schedule_group_exec.invoke(group_agent_input)
                say(response['output'], thread_ts=thread_ts)
                return
            response = schedule_exec.invoke(agent_input)
            say(response['output'], thread_ts=thread_ts)
    elif intent == "update event":
        agent_input['current_date'] = formatted_time
        agent_input['calendar_events'] = MicrosoftListCalendarEvents()._run(team_id, workspace_owner_id) if calendar_tool == "microsoft" else GoogleCalendarEvents()._run(team_id, workspace_owner_id)
        if channel_type or 'thread_ts' in event:
            group_agent_input = agent_input.copy()
            channel_history = format_channel_history(client.conversations_history(channel=channel_id, limit=2).get("messages", []))
            group_agent_input['mentioned_users'] = mentioned_users_output
            group_agent_input['channel_history'] = channel_history
            group_agent_input['formatted_calendar'] = output
            response = update_group_exec.invoke(group_agent_input)
            say(response['output'], thread_ts=thread_ts)
            return
        response = update_exec.invoke(agent_input)
        say(response['output'], thread_ts=thread_ts)
    elif intent == "delete event":
        agent_input['current_date'] = formatted_time
        agent_input['calendar_events'] = MicrosoftListCalendarEvents()._run(team_id, workspace_owner_id) if calendar_tool == "microsoft" else GoogleCalendarEvents()._run(team_id, workspace_owner_id)
        response = delete_exec.invoke(agent_input)
        say(response['output'], thread_ts=thread_ts)
    elif intent == "other":
        response = llm.predict(general_prompt.format(input=text, channel_history=channel_history))
        say(response, thread_ts=thread_ts)
    else:
        say("I'm not sure how to handle that request.", thread_ts=thread_ts)
@bolt_app.event("team_join")
def handle_team_join(event, client, context, logger):
    try:
        user_info = event['user']
        team_id = context.team_id
        
        # Fetch workspace name from Slack API
        try:
            team_info = client.team_info()
            workspace_name = team_info['team']['name']
        except SlackApiError as e:
            logger.error(f"Error fetching team info: {e.response['error']}")
            workspace_name = "Unknown Workspace"

        # Extract user details
        user_id = user_info['id']
        real_name = user_info.get('real_name', 'Unknown')
        profile = user_info.get('profile', {})
        email = profile.get('email', f"{user_info.get('name', 'user')}@example.com")  # Fallback email
        name = user_info.get('name', '')
        is_owner = user_info.get('is_owner', False)

        # Connect to database
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()
        
        # Insert/update user in Users table
        cur.execute('''
            INSERT INTO Users (team_id, user_id, workspace_name, real_name, email, name, is_owner, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, user_id) DO UPDATE SET
                workspace_name = EXCLUDED.workspace_name,
                real_name = EXCLUDED.real_name,
                email = EXCLUDED.email,
                name = EXCLUDED.name,
                is_owner = EXCLUDED.is_owner,
                last_updated = EXCLUDED.last_updated
        ''', (team_id, user_id, workspace_name, real_name, email, name, is_owner, datetime.now()))
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Update user cache
        with user_cache_lock:
            if team_id not in user_cache:
                user_cache[team_id] = {}
            user_cache[team_id][user_id] = {
                "real_name": real_name,
                "email": f"{name}@gmail.com",
                "name": name,
                "is_owner": is_owner,
                "workspace_name": workspace_name
            }
        
        # Update owner_id_cache if user is owner
        if is_owner:
            with owner_id_lock:
                owner_id_cache[team_id] = user_id
        
        logger.info(f"Processed team_join event for user {user_id} in team {team_id}")
    
    except KeyError as e:
        logger.error(f"Missing key in event data: {e}")
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error handling team_join: {e}")
def open_group_dm(client, users):
    try:
        response = client.conversations_open(users=",".join(users))
        return response["channel"]["id"] if response["ok"] else (None, "Failed to create group DM")
    except SlackApiError as e:
        return None, f"Error creating group DM: {e.response['error']}"

# Flask Routes
@app.route("/slack/events", methods=["POST"])
def slack_events():
    return slack_handler.handle(request)

@app.route("/slack/install", methods=["GET"])
def slack_install():
    return slack_handler.handle(request)

@app.route("/slack/oauth_redirect", methods=["GET"])
def slack_oauth_redirect():
    return slack_handler.handle(request)

@app.route("/oauth2callback")
def oauth2callback():
    state = request.args.get('state', '')
    print(f"STATE: {state}")
    print(f"STATs: {state_manager._states}")
    user_id = state_manager.validate_and_consume_state(state)
    stored_state = get_from_session(user_id, "gcal_state") if user_id else None
    
    if not user_id or stored_state != state:
        return "Invalid state", 400
    
    team_id = get_team_id_from_owner_id(user_id)
    if not team_id:
        return "Workspace not found", 404
    
    client = get_client_for_team(team_id)
    if not client:
        return "Client not found", 500
    
    flow = Flow.from_client_secrets_file('credentials.json', scopes=SCOPES, redirect_uri=OAUTH_REDIRECT_URI)
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    service = build('oauth2', 'v2', credentials=credentials)
    user_info = service.userinfo().get().execute()
    google_email = user_info.get('email', 'unknown@example.com')
    token_data = json.loads(credentials.to_json())
    token_data['google_email'] = google_email
    
    save_token(team_id, user_id, 'google', token_data)  # Adjusted to use team_id and user_id
    client.views_publish(user_id=user_id, view=create_home_tab(client, team_id, user_id))
    
    return "Google Calendar connected successfully! You can close this window."
@bolt_app.action("launch_auth")
def handle_launch_auth(ack, body, logger):
    ack()  # Acknowledge the action
    logger.info(f"Launch auth triggered by user {body['user']['id']}")
    # No further action needed since the URL redirect handles the OAuth flow
@app.route("/microsoft_callback")
def microsoft_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    if not code or not state:
        return "Missing parameters", 400
    user_id = state_manager.validate_and_consume_state(state)
    if not user_id:
        return "Invalid or expired state parameter", 403
    team_id = get_team_id_from_owner_id(user_id)
    if not team_id:
        return "Workspace not found", 404
    client = get_client_for_team(team_id)
    if not client:
        return "Client not found", 500
    if user_id != get_workspace_owner_id(client, team_id):
        return "Unauthorized", 403
    msal_app = ConfidentialClientApplication(MICROSOFT_CLIENT_ID, authority=MICROSOFT_AUTHORITY, client_credential=MICROSOFT_CLIENT_SECRET)
    result = msal_app.acquire_token_by_authorization_code(code, scopes=MICROSOFT_SCOPES, redirect_uri=MICROSOFT_REDIRECT_URI)
    if "access_token" not in result:
        return "Authentication failed", 400
    token_data = {"access_token": result["access_token"], "refresh_token": result.get("refresh_token", ""), "expires_at": result["expires_in"] + time.time()}
    save_token(user_id, 'microsoft', token_data)
    client.views_publish(user_id=user_id, view=create_home_tab(client, team_id, user_id))
    return "Microsoft Calendar connected successfully! You can close this window."

@app.route("/zoom_callback")
def zoom_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    user_id = state_manager.validate_and_consume_state(state)
    if not user_id:
        return "Invalid or expired state", 403
    team_id = get_team_id_from_owner_id(user_id)
    if not team_id:
        return "Workspace not found", 404
    client = get_client_for_team(team_id)
    if not client:
        return "Client not found", 500
    params = {"grant_type": "authorization_code", "code": code, "redirect_uri": ZOOM_REDIRECT_URI}
    try:
        response = requests.post(ZOOM_TOKEN_API, params=params, auth=(CLIENT_ID, CLIENT_SECRET))
    except Exception as e:
        return jsonify({"error": f"Token request failed: {str(e)}"}), 500
    if response.status_code == 200:
        token_data = response.json()
        token_data["expires_at"] = time.time() + token_data["expires_in"]
        # Fixed: Pass all required arguments in correct order
        save_token(team_id, user_id, 'zoom', token_data)
        client.views_publish(user_id=user_id, view=create_home_tab(client, team_id, user_id))
        return "Zoom connected successfully! You can close this window."
    return "Failed to retrieve token", 400

@bolt_app.action("open_zoom_config_modal")
def handle_open_zoom_config_modal(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    team_id = body["team"]["id"]
    owner_id = get_workspace_owner_id(client, team_id)
    if user_id != owner_id:
        client.chat_postMessage(channel=user_id, text="Only the workspace owner can configure Zoom.")
        return
    
    # Fixed: Pass both team_id and user_id to load_preferences
    prefs = load_preferences(team_id, user_id)
    zoom_config = prefs.get("zoom_config", {"mode": "manual", "link": None})
    mode = zoom_config["mode"]
    link = zoom_config.get("link", "")
    
    try:
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "zoom_config_submit",
                "title": {"type": "plain_text", "text": "Configure Zoom"},
                "submit": {"type": "plain_text", "text": "Save"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "zoom_mode",
                        "label": {"type": "plain_text", "text": "Zoom Mode"},
                        "element": {
                            "type": "static_select",
                            "action_id": "mode_select",
                            "placeholder": {"type": "plain_text", "text": "Select mode"},
                            "options": [
                                {"text": {"type": "plain_text", "text": "Automatic"}, "value": "automatic"},
                                {"text": {"type": "plain_text", "text": "Manual"}, "value": "manual"}
                            ],
                            "initial_option": {"text": {"type": "plain_text", "text": "Automatic" if mode == "automatic" else "Manual"}, "value": mode} if mode else None
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "zoom_link",
                        "label": {"type": "plain_text", "text": "Manual Zoom Link"},
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "link_input",
                            "placeholder": {"type": "plain_text", "text": "Enter Zoom link"},
                            "initial_value": link if isinstance(link, str) else ""
                        },
                        "optional": True
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening Zoom config modal: {e}")

@bolt_app.action("configure_zoom")
def handle_zoom_config(ack, body, client, logger):
    ack()  # Acknowledge the action
    user_id = body["user"]["id"]
    team_id = body["team"]["id"]

    # Ensure only the workspace owner can configure Zoom
    owner_id = get_workspace_owner_id(client, team_id)
    if user_id != owner_id:
        client.chat_postMessage(channel=user_id, text="Only the workspace owner can configure Zoom.")
        return

    # Check if this is a refresh or initial authentication
    zoom_token = load_token(team_id, owner_id, "zoom")
    is_refresh = zoom_token is not None

    # Generate the Zoom OAuth URL
    state = state_manager.create_state(owner_id)  # Assume this generates a unique state
    auth_url = f"{ZOOM_OAUTH_AUTHORIZE_API}?response_type=code&client_id={CLIENT_ID}&redirect_uri={quote_plus(ZOOM_REDIRECT_URI)}&state={state}"

    # Set modal text based on the scenario
    modal_title = "Refresh Zoom Token" if is_refresh else "Authenticate with Zoom"
    button_text = "Refresh Zoom Token" if is_refresh else "Authenticate with Zoom"

    # Open a modal with the appropriate text
    try:
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "title": {"type": "plain_text", "text": modal_title},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Click below to {button_text.lower()}:"}
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": button_text},
                                "url": auth_url,
                                "action_id": "launch_zoom_auth"
                            }
                        ]
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening Zoom auth modal: {e}")

@bolt_app.view("zoom_config_submit")
def handle_zoom_config_submit(ack, body, client, logger):
    ack()  # Ensure ack is called before any processing to avoid warnings
    user_id = body["user"]["id"]
    team_id = body["team"]["id"]
    owner_id = get_workspace_owner_id(client, team_id)
    if user_id != owner_id:
        return  # Early return if not owner; no need to proceed
    
    values = body["view"]["state"]["values"]
    mode = values["zoom_mode"]["mode_select"]["selected_option"]["value"]
    link = values["zoom_link"]["link_input"]["value"] if "zoom_link" in values and "link_input" in values["zoom_link"] else None
    zoom_config = {"mode": mode, "link": link if mode == "manual" else None}
    
    
    save_preference(team_id, user_id, zoom_config=zoom_config)
    
    client.views_publish(user_id=user_id, view=create_home_tab(client, team_id, user_id))
@bolt_app.action("launch_zoom_auth")
def handle_some_action(ack, body, logger):
    ack()

scheduler = BackgroundScheduler()
scheduler.add_job(state_manager.cleanup_expired_states, 'interval', minutes=5)
scheduler.start()

@app.route('/')
def home():
    return render_template('index.html')
# @app.route('/ZOOM_verify_a12f2ccf48a647aa8ebc987a249133f8.html')
# def home():
#     return render_template('ZOOM_verify_a12f2ccf48a647aa8ebc987a249133f8.html')
if __name__ == "__main__":
    app.run(port=3000)