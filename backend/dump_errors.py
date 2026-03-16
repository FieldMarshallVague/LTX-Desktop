import subprocess
import json
import io

print("Running Pyright...")
res = subprocess.run(["uv", "run", "pyright", "--outputjson"], capture_output=True, text=True, encoding='utf-8')
try:
    data = json.loads(res.stdout)
    errors = data.get('generalDiagnostics', [])
    print(f"Found {len(errors)} errors")
    with io.open('last_pyright_errors.txt', 'w', encoding='utf-8') as f:
        for err in errors:
            f.write(f"{err.get('file', 'Unknown')}:{err.get('range', {}).get('start', {}).get('line', 0) + 1} {err.get('message', 'No message')}\n")
    print("Wrote to last_pyright_errors.txt")
except Exception as e:
    print(f"Failed to parse JSON: {e}")
