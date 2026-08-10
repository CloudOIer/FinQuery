"""MCP 服务入口。

stdio 传输把标准输出当作协议信道,任何库往 stdout 打印一行都会让客户端
解析失败,因此启动前先把日志统一钉到 stderr。
"""

from __future__ import annotations

import argparse
import logging
import sys

from finquery_agent.mcp_server.server import mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FinQuery MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="仅 sse/streamable-http 生效")
    parser.add_argument("--port", type=int, default=8000, help="仅 sse/streamable-http 生效")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
