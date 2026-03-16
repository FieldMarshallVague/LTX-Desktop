import urllib.request, json
res = urllib.request.urlopen('https://huggingface.co/api/models/unsloth/LTX-2.3-GGUF/tree/main/distilled')
data = json.loads(res.read())
for m in data:
    if m['path'].endswith(('Q4_K_M.gguf', 'Q5_K_S.gguf', 'Q6_K.gguf', 'Q8_0.gguf')) and 'UD' not in m['path']:
        print(f"{m['path']} : {m['size']}")
