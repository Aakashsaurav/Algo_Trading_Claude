# login_helper.py  ← Run this every morning before trading
"""
Minimal HTTPS server that:
1. Opens the Upstox login page in your browser
2. Listens on https://127.0.0.1:5000/ for the redirect
3. Automatically captures the auth code
4. Exchanges it for an access token
5. Saves the token to disk

Run with: python login_helper.py
Requires:  pip install flask pyopenssl
"""

import ssl
import threading
import webbrowser
from flask import Flask, request

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from broker.upstox.auth import auth_manager

app = Flask(__name__)
captured_code = None
server_thread = None


@app.route("/")
def callback():
    """Upstox redirects here after login with ?code=XXXXX"""
    global captured_code

    code = request.args.get("code")
    error = request.args.get("error")

    if error:
        return f"""
        <h2 style='color:red'>Login Failed</h2>
        <p>Error: {error}</p>
        <p>Description: {request.args.get('error_description', 'Unknown')}</p>
        """, 400

    if not code:
        return "<h2 style='color:red'>No code received from Upstox</h2>", 400

    captured_code = code

    # Exchange the code for a token
    try:
        token_data = auth_manager.generate_token(code)
        user_id = token_data.get("user_id", "Unknown")
        return f"""
        <html><body style='font-family:Arial; text-align:center; padding:50px'>
        <h2 style='color:green'>✅ Login Successful!</h2>
        <p>Welcome, <strong>{user_id}</strong></p>
        <p>Access token has been saved. You can close this window.</p>
        <p style='color:gray'>Token is valid until 3:30 AM tomorrow.</p>
        </body></html>
        """
    except Exception as e:
        return f"""
        <h2 style='color:red'>Token Exchange Failed</h2>
        <p>{str(e)}</p>
        """, 500


def run_server():
    """Run Flask with self-signed SSL on port 5000."""
    # Generate a self-signed SSL context for https://127.0.0.1:5000
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_verify_locations  # not needed for self-signed

    # Use adhoc SSL (requires pyopenssl: pip install pyopenssl)
    app.run(
        host="127.0.0.1",
        port=5000,
        ssl_context="adhoc",   # generates a temporary self-signed cert
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  UPSTOX DAILY LOGIN")
    print("=" * 55)
    print("Starting local HTTPS server on https://127.0.0.1:5000/")
    print("Opening Upstox login in your browser...")
    print()
    print("NOTE: Your browser may warn about the self-signed")
    print("certificate. Click 'Advanced' → 'Proceed' to continue.")
    print("=" * 55)

    # Start the Flask server in a background thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    import time
    time.sleep(1)  # Give server a moment to start

    # Open the Upstox login page
    login_url = auth_manager.get_login_url()
    webbrowser.open(login_url)

    print("\nWaiting for login... (check your browser)")
    print("After logging in, this script will auto-capture the token.")
    print("Press Ctrl+C to exit if needed.\n")

    # Keep running until token is captured
    try:
        while True:
            time.sleep(1)
            if captured_code:
                print("\n✅ Token captured and saved successfully!")
                print("You can now run your trading system.")
                time.sleep(2)
                break
    except KeyboardInterrupt:
        print("\nLogin cancelled.")