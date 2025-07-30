# S3 Vectors MCP Server

> [!NOTE]
> This README was generated with Amazon Q Developer.

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that provides tools for interacting with AWS S3 Vectors service. This server enables AI assistants to embed text using Amazon Bedrock models and store/query vector embeddings in S3 Vectors indexes.

## Features

- **Vector Embedding**: Convert text to vector embeddings using Amazon Bedrock models
- **Vector Storage**: Store embeddings in AWS S3 Vectors indexes with optional metadata
- **Similarity Search**: Query for similar vectors with advanced filtering capabilities
- **Multiple Transports**: Support for stdio, Server-Sent Events (SSE), and StreamableHTTP
- **Flexible Configuration**: Environment variable and parameter-based configuration

## Prerequisites

- Python 3.11 or higher
- AWS credentials configured (via AWS CLI, environment variables, or IAM roles)
- Access to Amazon Bedrock embedding models
- An existing S3 Vectors bucket and index

### AWS Permissions

Your AWS credentials need permissions for:
- Amazon Bedrock: `bedrock:InvokeModel` for embedding models
- S3 Vectors: `s3:GetObject`, `s3:PutObject` for vector operations

## Installation

### Quick Install

```bash
pip install git+https://github.com/moritalous/s3-vectors-mcp.git
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/moritalous/s3-vectors-mcp.git
cd s3-vectors-mcp

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Configuration

The server uses environment variables for configuration:

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `S3VECTORS_BUCKET_NAME` | Yes | S3 bucket name for vector storage | `my-vectors-bucket` |
| `S3VECTORS_INDEX_NAME` | Yes | Vector index name | `my-index` |
| `S3VECTORS_MODEL_ID` | Yes | Bedrock embedding model ID | `amazon.titan-embed-text-v2:0` |
| `S3VECTORS_DIMENSIONS` | No | Embedding dimensions | `1024` |
| `S3VECTORS_REGION` | No | AWS region | `us-east-1` |
| `S3VECTORS_PROFILE` | No | AWS profile name | `default` |

### Example Configuration

```bash
export S3VECTORS_BUCKET_NAME="my-vectors-bucket"
export S3VECTORS_INDEX_NAME="my-index"
export S3VECTORS_MODEL_ID="amazon.titan-embed-text-v2:0"
export S3VECTORS_DIMENSIONS="1024"
export S3VECTORS_REGION="us-east-1"
```

## Usage

### Running the Server

The server supports three transport modes:

```bash
# Using stdio transport (default, for MCP clients)
uv run s3-vectors-mcp stdio

# Using Server-Sent Events
uv run s3-vectors-mcp sse

# Using StreamableHTTP
uv run s3-vectors-mcp streamable-http
```

### Development Mode

```bash
# With uv (using Python module)
uv run python -m s3_vectors_mcp stdio

# With uv (using script entry point)
uv run s3-vectors-mcp stdio

# With pip (if installed globally)
s3-vectors-mcp stdio
```

### MCP Client Integration

#### Amazon Q Developer

For Amazon Q Developer, create or update your MCP configuration file (`.amazonq/mcp.json`):

```json
{
  "mcpServers": {
    "s3vector": {
      "command": "uv",
      "args": [
        "run",
        "s3-vectors-mcp"
      ],
      "env": {
        "S3VECTORS_BUCKET_NAME": "your-vectors-bucket",
        "S3VECTORS_INDEX_NAME": "your-index",
        "S3VECTORS_MODEL_ID": "amazon.titan-embed-text-v2:0",
        "S3VECTORS_DIMENSIONS": "1024",
        "S3VECTORS_REGION": "us-east-1"
      },
      "timeout": 120000,
      "disabled": false
    }
  }
}
```

#### Claude Desktop

For Claude Desktop, add to your MCP client configuration:

```json
{
  "mcpServers": {
    "s3-vectors": {
      "command": "uv",
      "args": [
        "run",
        "s3-vectors-mcp"
      ],
      "env": {
        "S3VECTORS_BUCKET_NAME": "your-vectors-bucket",
        "S3VECTORS_INDEX_NAME": "your-index",
        "S3VECTORS_MODEL_ID": "amazon.titan-embed-text-v2:0",
        "S3VECTORS_DIMENSIONS": "1024",
        "S3VECTORS_REGION": "us-east-1"
      }
    }
  }
}
```

#### Configuration Notes

- Replace `your-vectors-bucket` and `your-index` with your actual S3 bucket and index names
- The `timeout` setting (120000ms = 2 minutes) allows for longer embedding operations
- Set `disabled: false` to enable the server
- The `uv run` command assumes you have the project installed in development mode

## Available Tools

### s3vectors_put

Embed text and store as vector in S3 Vectors.

**Parameters:**
- `text` (string, required): Text to embed and store
- `vector_id` (string, optional): Custom vector ID (auto-generated if not provided)
- `metadata` (object, optional): Additional metadata to store with the vector

**Example:**
```json
{
  "text": "This is a sample document about machine learning.",
  "vector_id": "doc-001",
  "metadata": {
    "category": "documentation",
    "version": "1.0",
    "author": "John Doe"
  }
}
```

### s3vectors_query

Query for similar vectors in S3 Vectors.

**Parameters:**
- `query_text` (string, required): Text to query for similar vectors
- `top_k` (integer, optional): Number of similar vectors to return (default: 10)
- `filter_expr` (object, optional): Metadata filter expression
- `return_metadata` (boolean, optional): Include metadata in results (default: true)
- `return_distance` (boolean, optional): Include similarity distance (default: false)

**Example:**
```json
{
  "query_text": "machine learning algorithms",
  "top_k": 5,
  "filter_expr": {
    "$and": [
      {"category": {"$eq": "documentation"}},
      {"version": {"$gte": "1.0"}}
    ]
  },
  "return_metadata": true,
  "return_distance": true
}
```

## Filter Expressions

The `s3vectors_query` tool supports advanced filtering using AWS S3 Vectors API operators:

### Comparison Operators
- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal to
- `$lt`: Less than
- `$lte`: Less than or equal to
- `$in`: Value in array
- `$nin`: Value not in array

### Logical Operators
- `$and`: Logical AND (all conditions must be true)
- `$or`: Logical OR (at least one condition must be true)
- `$not`: Logical NOT (condition must be false)

### Filter Examples

**Simple equality:**
```json
{"category": {"$eq": "documentation"}}
```

**Multiple conditions:**
```json
{
  "$and": [
    {"category": "tech"},
    {"version": {"$gte": "2.0"}},
    {"status": {"$ne": "archived"}}
  ]
}
```

**Complex nested conditions:**
```json
{
  "$or": [
    {
      "$and": [
        {"category": "docs"},
        {"version": "1.0"}
      ]
    },
    {
      "$and": [
        {"category": "guides"},
        {"version": "2.0"}
      ]
    }
  ]
}
```

## Supported Embedding Models

This server works with Amazon Bedrock embedding models, including:

- `amazon.titan-embed-text-v1`
- `amazon.titan-embed-text-v2:0`
- `amazon.titan-embed-image-v1`
- `cohere.embed-english-v3`
- `cohere.embed-multilingual-v3`

Refer to the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) for the complete list of available models.

## Dependencies

This project is built on top of:

- **[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)**: Provides the Model Context Protocol implementation
- **[S3 Vectors Embed CLI](https://github.com/awslabs/s3vectors-embed-cli)**: Core library for S3 Vectors operations and Bedrock integration

## Development

### Project Structure

```
s3-vectors-mcp/
├── src/
│   └── s3_vectors_mcp/
│       ├── __init__.py
│       ├── __main__.py
│       └── server.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### Running Tests

```bash
# Install development dependencies
uv sync --dev

# Run linting
uv run ruff check
uv run ruff format --check
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run linting and tests
6. Submit a pull request

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Ensure the package is properly installed and you're using the correct Python environment.

**AWS Credentials**: Verify your AWS credentials are configured correctly:
```bash
aws sts get-caller-identity
```

**Bedrock Access**: Ensure you have access to the specified Bedrock model:
```bash
aws bedrock list-foundation-models --region us-east-1
```

**S3 Vectors Setup**: Verify your S3 bucket and vector index exist and are properly configured.

### Debug Mode

Enable debug logging by setting the log level:
```bash
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
s3-vectors-mcp stdio
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Related Projects

- [Model Context Protocol](https://modelcontextprotocol.io) - The protocol specification
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Python implementation of MCP
- [S3 Vectors Embed CLI](https://github.com/awslabs/s3vectors-embed-cli) - Command-line tool for S3 Vectors
- [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) - AWS documentation

## Support

For issues and questions:
- Check the [troubleshooting section](#troubleshooting)
- Review the [AWS S3 Vectors documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- Open an issue in this repository
