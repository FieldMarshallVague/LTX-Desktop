import json

with open("pyright.json", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
    
try:
    d = json.loads(text)
except Exception:
    start = text.find('{')
    if start != -1:
        d = json.loads(text[start:])
    else:
        print("COULD NOT PARSE")
        exit(1)

with open("all_errors.txt", "w", encoding="utf-8") as f:
    for x in d.get("generalDiagnostics", []):
        file_path = x.get("file", "")
        f.write(f"{file_path}:{x['range']['start']['line']+1} {x['message']}\n")
