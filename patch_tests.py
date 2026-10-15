import os
import re

for filepath in ['backend/tests/test_persistence.py', 'backend/tests/test_recovery.py']:
    if not os.path.exists(filepath): continue
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace('get_config_key("ulb",', 'get_config_key("ulb", "dummy_hash",')
    content = content.replace('get_config_key("synthetic_benchmark",', 'get_config_key("synthetic_benchmark", "dummy_hash",')
    content = content.replace('get_config_key("synthetic",', 'get_config_key("synthetic", "dummy_hash",')
    with open(filepath, 'w') as f:
        f.write(content)
