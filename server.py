"""便捷入口:python server.py [--transport stdio|sse|streamable-http]

真正的实现位于 src/finquery_agent/mcp_server/,此处只做路径准备与转发,
方便未安装本包时由 MCP 客户端直接拉起。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from finquery_agent.mcp_server.__main__ import main  # noqa: E402
from finquery_agent.mcp_server.server import mcp  # noqa: E402  MCP Inspector(mcp dev)按模块级变量名查找服务对象

if __name__ == "__main__":
    main()
