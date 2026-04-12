#!/usr/bin/env python3
"""
OAuth 2.0 Auth Flow Helper for DVC Viewer
Generates DVC_GDRIVE_CREDENTIALS and DVC_GDRIVE_TOKEN for Google Drive.
"""
import os
import sys
import json
import webbrowser

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("❌ Error: google-auth-oauthlib is not installed.")
    print("Please run: pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def main():
    print("=" * 80)
    print("🔍 DVC Viewer - Google Drive OAuth 2.0 Setup".center(80))
    print("=" * 80)
    print("\nThis script will help you generate the necessary tokens to allow")
    print("dvc-viewer to automatically push your DVC pipeline outputs to Google Drive.\n")
    print("WARNING: Before proceeding, ensure your Google Cloud OAuth app is set to")
    print("'Production' status. If it stays in 'Testing', your token will expire")
    print("in 7 days and auto-push will suddenly fail.\n")
    print("-" * 80)
    print("Step 1: Client Secret JSON")
    print("Please paste the ENTIRE contents of the client_secret.json file you")
    print("downloaded from the Google Cloud Console (OAuth 2.0 Client IDs).")
    print("Press Enter on an empty line when finished pasting.")
    print("-" * 80)

    lines = []
    while True:
        try:
            line = input()
            if not line.strip() and len(lines) > 0:
                # Basic check to see if it looks like a complete JSON
                try:
                    json.loads(''.join(lines))
                    break
                except json.JSONDecodeError:
                    pass
            lines.append(line)
        except EOFError:
            break

    raw_json = ''.join(lines)
    try:
        creds_dict = json.loads(raw_json)
        if "installed" not in creds_dict and "web" not in creds_dict:
             print("\n❌ Error: The provided JSON does not seem to be a valid OAuth 2.0 Client Secret.")
             sys.exit(1)
    except json.JSONDecodeError:
        print("\n❌ Error: Invalid JSON provided.")
        sys.exit(1)

    print("\n" + "-" * 80)
    print("Step 2: Google Drive Authorization")
    print("A browser window will now open to ask for permission to access your Drive.")
    print("Please select your Google account and click 'Allow'.")
    print("-" * 80)

    try:
        # Create flow directly from client config dict
        flow = InstalledAppFlow.from_client_config(creds_dict, SCOPES)
        creds = flow.run_local_server(port=0)
    except Exception as e:
        print(f"\n❌ Error during authentication flow: {e}")
        sys.exit(1)

    token_json = creds.to_json()

    print("\n" + "=" * 80)
    print("✅ Authentication Successful!".center(80))
    print("=" * 80)
    print("\nYour tokens have been generated. Please export the following two environment")
    print("variables on your VM or deployment environment:\n")

    print(f"export DVC_GDRIVE_CREDENTIALS='{raw_json}'")
    print(f"export DVC_GDRIVE_TOKEN='{token_json}'")
    print("\n" + "-" * 80)

if __name__ == "__main__":
    main()
