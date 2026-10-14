import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.gemini_vision import get_gemini_client
client = get_gemini_client()
print("Available models:")
for m in client.models.list():
    if 'gemini' in m.name.lower():
        print(m.name)
