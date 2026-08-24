"""pytest 的全局配置：把仓库根目录加进 import 路径。

不这么做，`from src.xxx import ...` 只在从仓库根目录启动 pytest 时才成立。
放在 conftest 里，从任何目录跑 `pytest` 都能工作。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
