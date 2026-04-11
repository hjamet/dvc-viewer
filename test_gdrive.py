import sys
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

creds_data = open('/home/jules/test_credentials.json').read()

creds_dict = json.loads(creds_data)
# Notice: test_credentials.json contains "installed" type (OAuth2 client ID), not a Service Account !
print(creds_dict.get('type') or ('installed' in creds_dict and 'installed'))
