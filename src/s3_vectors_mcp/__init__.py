"""S3 Vectors MCP Server

A Model Context Protocol server providing tools to interact with AWS S3 Vectors service
for embedding and querying vector data using Amazon Bedrock models.
"""

from .server import serve

__version__ = "0.1.0"
__all__ = ["serve", "main"]


def main():
    """MCP S3 Vectors Server - Vector embedding and querying functionality for MCP"""
    import argparse

    parser = argparse.ArgumentParser(
        description="S3 Vectors MCP Server - Vector embedding and querying functionality"
    )
    parser.add_argument(
        "transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport type to use (default: stdio)",
    )

    args = parser.parse_args()
    serve(args.transport)


if __name__ == "__main__":
    main()
