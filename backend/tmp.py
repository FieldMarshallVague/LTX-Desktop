import json
d = json.load(open("pyright.json", encoding="utf-8", errors="ignore"))
for x in d.get("generalDiagnostics", []):
    file_path = x.get("file", "")
    if "fast_video_pipeline.py" in file_path or "services.py" in file_path or "pipelines_handler.py" in file_path:
        print(f"{file_path}:{x['range']['start']['line']+1} {x['message']}")
