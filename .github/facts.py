"""TEMPORARY: describe candidate gesture model files. Removed before merging."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.task")):
    data = path.read_bytes()
    print(
        f"{path.name}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()} "
        f"head={data[:24].hex()} first_local_header_at={data.find(b'PK' + bytes([3, 4]))} "
        f"is_zip={zipfile.is_zipfile(path)}"
    )
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as bundle:
            print("  entries:", [(info.filename, info.file_size) for info in bundle.infolist()][:6])
