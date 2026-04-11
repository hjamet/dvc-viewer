with open("dvc_viewer/cli.py", "r") as f:
    content = f.read()

# We need to conditionally use `gdrive_use_service_account` "true" only if it's a real service account.
# If it's `installed` or `web` (OAuth app), DVC behaves differently (gdrive_use_service_account false).
# However, DVC cannot automatically use an OAuth client JSON without interactive browser auth unless DVC itself is configured with a refresh token.
# If DVC has a refresh token configured, DVC remote configuration requires `gdrive_client_id`, `gdrive_client_secret` etc.

# We will patch cli.py to recognize "installed" client creds from the user and set up DVC remote properly.
# The user's provided test_credentials.json is an OAuth2 "installed" Client JSON, not a service account JSON.

content = content.replace(
    'subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "true"],\n                   cwd=str(project_dir), capture_output=True)',
    'if json.loads(creds_data).get("type") == "service_account":\n        subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "true"], cwd=str(project_dir), capture_output=True)\n        subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_service_account_json_file_path", str(creds_file)], cwd=str(project_dir), capture_output=True)\n    else:\n        # It is OAuth client id/secret\n        creds_json = json.loads(creds_data)\n        app_type = "installed" if "installed" in creds_json else "web"\n        client_id = creds_json[app_type]["client_id"]\n        client_secret = creds_json[app_type]["client_secret"]\n        subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "false"], cwd=str(project_dir), capture_output=True)\n        subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_client_id", client_id], cwd=str(project_dir), capture_output=True)\n        subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_client_secret", client_secret], cwd=str(project_dir), capture_output=True)'
)

content = content.replace(
    'subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_service_account_json_file_path", str(creds_file)],\n                   cwd=str(project_dir), capture_output=True)',
    ''
)

with open("dvc_viewer/cli.py", "w") as f:
    f.write(content)
