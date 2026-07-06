# -*- coding: utf-8 -*-
"""快速验证 styles.py 导入"""
import sys
import io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from webui.styles import GLOBAL_CSS, COLORS

print(f"CSS length: {len(GLOBAL_CSS)}")
print(f"Has primary color: {'#2D6A4F' in GLOBAL_CSS}")
print(f"Has unresolved placeholder: {'placeholder' in GLOBAL_CSS}")
print(f"COLORS keys: {len(COLORS)}")

# Check no $ signs remain (would indicate unresolved placeholders)
dollar_count = GLOBAL_CSS.count('$')
print(f"Remaining dollar signs: {dollar_count}")

if dollar_count == 0:
    print("\nPASS: All placeholders resolved correctly!")
else:
    print("\nFAIL: Some placeholders were not resolved!")
