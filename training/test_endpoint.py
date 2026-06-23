import requests

# Make sure server is running in another terminal first
URL       = "http://127.0.0.1:8000/scan/"
IMG_PATH  = r"C:\Users\Karthik\Desktop\strip.jpg"

print("Testing /scan/ endpoint...\n")

with open(IMG_PATH, 'rb') as img:
    response = requests.post(URL, data={'scan_type': 'strip'},
                             files={'image': img})

print(f"Status code : {response.status_code}")
print(f"Response    :\n")

import json
result = response.json()
print(json.dumps(result, indent=2))

print("\n--- Summary ---")
if result.get('success'):
    for med in result.get('medicines', []):
        print(f"Medicine    : {med['name']}")
        print(f"Salt        : {med['salt']}")
        print(f"Alternatives: {len(med['alternatives'])} found")
        print(f"Side effects: {len(med['side_effects'])} found")
else:
    print(f"Failed: {result.get('error')}")