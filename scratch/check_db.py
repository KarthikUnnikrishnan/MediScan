import sqlite3
from django.conf import settings
import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

conn = sqlite3.connect(str(settings.MEDICINES_DB_PATH))
cur = conn.cursor()
cur.execute('PRAGMA table_info(medicines)')
print('Columns in medicines:')
for col in cur.fetchall():
    print(col)

print('\nCheck Lorinol 10:')
cur.execute("SELECT id, name, salt, manufacturer, price FROM medicines WHERE name LIKE '%Lorinol 10%' LIMIT 3")
for r in cur.fetchall():
    print(r)

print('\nCheck Lorsyn:')
cur.execute("SELECT id, name, salt, manufacturer, price FROM medicines WHERE name LIKE '%Lorsyn%' LIMIT 3")
for r in cur.fetchall():
    print(r)

print('\nCheck StayHappi Loratadine:')
cur.execute("SELECT id, name, salt, manufacturer, price FROM medicines WHERE name LIKE '%StayHappi Loratadine%' LIMIT 3")
for r in cur.fetchall():
    print(r)
