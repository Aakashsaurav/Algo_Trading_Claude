"""
broker/auth.py
--------------
Handles Upstox OAuth2 Authentication and daily token management.

HOW UPSTOX OAuth2 WORKS (Plain English):
-----------------------------------------
Upstox uses a two-step login process called OAuth2:

Step 1 — Get Authorization Code:
    You open a special Upstox URL in your browser.
    You log in with your Upstox credentials.
    Upstox redirects your browser to your redirect_uri with a 'code' in the URL.
    Example: http://127.0.0.1:8000/callback?code=abc123

Step 2 — Exchange Code for Access Token:
    You take that 'code' and POST it to Upstox along with your API key + secret.
    Upstox gives you an access_token (valid for the rest of the trading day).
    You store this token and use it in every subsequent API call.

IMPORTANT:
    - The token expires every day at midnight.
    - You must repeat this flow every morning before trading starts.
    - Our scheduler (Phase 7) will automate this at 9:00 AM daily.

USAGE:
    from broker.auth import AuthManager
    auth = AuthManager()

    # Step 1: Get the login URL and open it in browser
    url = auth.get_login_url()

    # Step 2: After redirect, paste the full redirect URL or just the code
    token = auth.generate_token(auth_code="abc123")

    # All subsequent uses — just get a valid token:
    token = auth.get_valid_token()
"""

import json
import logging
import webbrowser
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests

from config import config

# Module-level logger — follows the pattern established in config.py
logger = logging.getLogger(__name__)


class AuthManager:
    """
    Manages the full Upstox OAuth2 authentication lifecycle.

    Responsibilities:
    - Generate the Upstox login URL for the browser
    - Exchange the auth code for an access token
    - Save the token to disk (so you don't re-login on every restart)
    - Load and validate the saved token on startup
    - Detect if the token is expired (it's a new day) and prompt re-login
    """

    def __init__(self):
        self.api_key = config.API_KEY
        self.api_secret = config.API_SECRET
        self.redirect_uri = config.REDIRECT_URI
        self.auth_url = config.AUTH_URL
        self.token_file = config.TOKEN_FILE_PATH

        # In-memory token cache: loaded from file or freshly generated
        self._token_data: Optional[dict] = None

    # ── Step 1: Generate Login URL ────────────────────────────────────────────

    def get_login_url(self) -> str:
        """
        Build and return the Upstox authorization URL.

        The user must open this URL in a browser, log in to Upstox,
        and then copy the redirected URL (or just the 'code' parameter).

        Returns:
            str: Full authorization URL to open in browser.
        """
        params = {
            "response_type": "code",
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri,
        }
        login_url = f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(params)}"
        logger.info("Login URL generated. Opening in browser...")
        logger.info(f"If browser doesn't open, manually visit:\n{login_url}")
        return login_url

    def open_login_page(self):
        """
        Convenience method: generate the login URL and open it in the default browser.
        After logging in, Upstox will redirect to your redirect_uri with ?code=...
        """
        url = self.get_login_url()
        webbrowser.open(url)
        print("\n" + "=" * 60)
        print("  UPSTOX LOGIN")
        print("=" * 60)
        print("1. Your browser should have opened the Upstox login page.")
        print("2. Log in with your Upstox credentials.")
        print("3. After login, you'll be redirected to a URL like:")
        print("   http://127.0.0.1:8000/callback?code=XXXXX")
        print("4. Copy that full URL (or just the code value) and")
        print("   call: auth.generate_token_from_url('<paste here>')")
        print("=" * 60 + "\n")

    # ── Step 2: Exchange Code for Token ───────────────────────────────────────

    def generate_token(self, auth_code: str) -> dict:
        """
        Exchange the authorization code for an access token.

        Upstox gives us an access_token after we send:
        - The auth code received from the browser redirect
        - Our API key and API secret

        Args:
            auth_code (str): The 'code' value from the redirect URL.

        Returns:
            dict: Token data including 'access_token', 'token_type', etc.

        Raises:
            ValueError: If the API returns an error.
            requests.RequestException: If the network call fails.
        """
        logger.info("Exchanging authorization code for access token...")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        payload = {
            "code": auth_code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        try:
            response = requests.post(
                self.auth_url,
                data=payload,
                headers=headers,
                timeout=15  # 15-second timeout — never block forever
            )
            response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses

        except requests.exceptions.Timeout:
            logger.error("Token request timed out after 15 seconds.")
            raise

        except requests.exceptions.ConnectionError:
            logger.error("Network error: Could not reach Upstox API. Check your internet.")
            raise

        except requests.exceptions.HTTPError as e:
            # Parse the error body from Upstox for a helpful message
            error_body = response.json() if response.content else {}
            error_msg = error_body.get("message", str(e))
            logger.error(f"Upstox API error during token exchange: {error_msg}")
            raise ValueError(f"Token generation failed: {error_msg}") from e

        token_data = response.json()

        # Add the date this token was generated so we can detect expiry tomorrow
        token_data["generated_date"] = date.today().isoformat()

        # Save to disk and cache in memory
        self._save_token(token_data)
        self._token_data = token_data

        logger.info(
            f"✅ Access token generated successfully. "
            f"Token type: {token_data.get('token_type', 'bearer')}"
        )

        return token_data

    def generate_token_from_url(self, redirect_url: str) -> dict:
        """
        Convenience method: extracts the auth code from the full redirect URL
        and then generates the token.

        Args:
            redirect_url (str): The full URL you were redirected to after login.
                                 e.g. http://127.0.0.1:8000/callback?code=abc123

        Returns:
            dict: Token data.
        """
        try:
            parsed = urlparse(redirect_url)
            params = parse_qs(parsed.query)
            auth_code = params.get("code", [None])[0]

            if not auth_code:
                raise ValueError(
                    "Could not find 'code' in the redirect URL. "
                    "Make sure you copied the full URL after login."
                )

            logger.info(f"Extracted auth code from redirect URL.")
            return self.generate_token(auth_code)

        except Exception as e:
            logger.error(f"Failed to extract auth code from URL: {e}")
            raise

    # ── Token Storage ─────────────────────────────────────────────────────────

    def _save_token(self, token_data: dict):
        """
        Save token data to disk as JSON.

        Why we save to disk: If the app restarts during the trading day,
        we don't want to force the user to log in again. As long as the
        saved token was generated today, it's still valid.

        Args:
            token_data (dict): Token data from Upstox API.
        """
        try:
            # Ensure the parent directory exists
            self.token_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.token_file, "w") as f:
                json.dump(token_data, f, indent=2)

            logger.info(f"Token saved to: {self.token_file}")

        except OSError as e:
            # This is non-fatal — we still have the token in memory
            logger.warning(f"Could not save token to disk: {e}")

    def _load_token(self) -> Optional[dict]:
        """
        Load token from disk if it exists.

        Returns:
            dict if a valid (today's) token exists on disk, else None.
        """
        if not self.token_file.exists():
            logger.info("No saved token file found.")
            return None

        try:
            with open(self.token_file, "r") as f:
                token_data = json.load(f)

            generated_date = token_data.get("generated_date")

            # Check if the token was generated today
            if generated_date == date.today().isoformat():
                logger.info(f"Valid token loaded from disk (generated today: {generated_date})")
                return token_data
            else:
                logger.warning(
                    f"Saved token is from {generated_date}, not today. "
                    f"A new login is required."
                )
                return None

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Token file is corrupted or unreadable: {e}. Ignoring.")
            return None

    # ── Main Public Interface ─────────────────────────────────────────────────

    def get_valid_token(self) -> str:
        """
        The main method all other modules should call to get a usable token.

        Logic:
        1. If we have a token in memory that was generated today → return it.
        2. If there's a valid token on disk (generated today) → load and return it.
        3. Otherwise → raise an error telling the user to log in.

        Returns:
            str: The access token string.

        Raises:
            RuntimeError: If no valid token exists and a new login is required.
        """
        # 1. Check in-memory cache first (fastest)
        if self._token_data:
            if self._token_data.get("generated_date") == date.today().isoformat():
                return self._token_data["access_token"]
            else:
                logger.info("In-memory token is from a previous day. Clearing cache.")
                self._token_data = None

        # 2. Try loading from disk
        token_data = self._load_token()
        if token_data:
            self._token_data = token_data
            return token_data["access_token"]

        # 3. No valid token — user must log in
        raise RuntimeError(
            "\n" + "=" * 60 +
            "\n  ⚠️  NO VALID TOKEN FOUND" +
            "\n  You need to log in to Upstox for today's session." +
            "\n  Run the following in your terminal:" +
            "\n\n  from broker.auth import AuthManager" +
            "\n  auth = AuthManager()" +
            "\n  auth.open_login_page()   # opens browser" +
            "\n  auth.generate_token_from_url('<paste redirect URL here>')" +
            "\n" + "=" * 60
        )

    def is_authenticated(self) -> bool:
        """
        Quick check: do we have a valid token for today?

        Returns:
            bool: True if authenticated, False if login is needed.
        """
        try:
            self.get_valid_token()
            return True
        except RuntimeError:
            return False

    def get_token_info(self) -> dict:
        """
        Return non-sensitive info about the current token for display/logging.

        Returns:
            dict with token status info (no actual token value exposed in logs).
        """
        if self._token_data:
            return {
                "status": "valid",
                "generated_date": self._token_data.get("generated_date"),
                "token_type": self._token_data.get("token_type", "bearer"),
                "user_id": self._token_data.get("user_id", "unknown"),
            }

        # Try loading from disk without raising error
        token_data = self._load_token()
        if token_data:
            return {
                "status": "valid (from disk)",
                "generated_date": token_data.get("generated_date"),
                "token_type": token_data.get("token_type", "bearer"),
                "user_id": token_data.get("user_id", "unknown"),
            }

        return {"status": "not authenticated — login required"}

    def logout(self):
        """
        Clear token from memory and disk.
        Also calls the Upstox logout endpoint to invalidate the token server-side.
        """
        logger.info("Logging out and clearing token...")

        # Attempt server-side logout (non-critical — proceed even if it fails)
        try:
            token = self._token_data.get("access_token") if self._token_data else None
            if token:
                requests.delete(
                    f"{config.BASE_URL}/v2/logout",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.warning(f"Server-side logout failed (non-critical): {e}")

        # Clear from memory
        self._token_data = None

        # Delete token file from disk
        if self.token_file.exists():
            self.token_file.unlink()
            logger.info("Token file deleted from disk.")

        logger.info("✅ Logged out successfully.")


# ── Module-level singleton ────────────────────────────────────────────────────
# All other modules use this single instance:
#   from broker.auth import auth_manager
#   token = auth_manager.get_valid_token()
auth_manager = AuthManager()
