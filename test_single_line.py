import subprocess
import json
import platform

cmd_name = "gemini.cmd" if platform.system() == "Windows" else "gemini"
prompt = "Say hello\nworld"
# Using single line
result = subprocess.run([cmd_name, prompt.replace('\n', ' '), "--output-format", "json"], capture_output=True, text=True)
print(result.stdout)
