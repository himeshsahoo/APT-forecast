import json

INPUT_JSON = r"C:\Users\Himes\Desktop\apt-forecast-project\AIA-1-25.ecar-last.json\AIA-1-25.ecar.json"

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    # Read just the very first line and print it nicely formatted
    first_line = json.loads(f.readline())
    print(json.dumps(first_line, indent=4))