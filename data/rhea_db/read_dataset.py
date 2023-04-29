import json
from pathlib import Path
import subprocess as sp

compressed_filename = Path("parsed_rhea.json.zip")
filename = Path("parsed_rhea.json")

# unzip file
if not filename.exists() and compressed_filename.exists():
    sp.call("unzip {filename}", shell=True)

data = json.load(open(filename, "r"))
print(f"length of data: {len(data)}")