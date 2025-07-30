"""
S3 Vectors MCP Server using FastMCP

This MCP server provides tools for interacting with AWS S3 Vectors service,
allowing users to embed and store vectors, as well as query for similar vectors.

Environment Variables:
    S3VECTORS_BUCKET_NAME: Default S3 bucket name for vector storage
    S3VECTORS_INDEX_NAME: Default vector index name
    S3VECTORS_MODEL_ID: Default Bedrock embedding model ID
    S3VECTORS_DIMENSIONS: Default embedding dimensions for supported models
    S3VECTORS_REGION: Default AWS region
    S3VECTORS_PROFILE: Default AWS profile name

Usage:
    # Using the installed script
    s3-vectors-mcp stdio
    s3-vectors-mcp sse
    s3-vectors-mcp streamable-http

    # Using Python module directly
    python -m mcp_s3vectors stdio
    python -m mcp_s3vectors sse
    python -m mcp_s3vectors streamable-http
"""

import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, Optional

import boto3
from mcp.server.fastmcp import FastMCP
from s3vectors.core.services import BedrockService, S3VectorService
from s3vectors.utils.config import get_region

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("S3 Vectors")


def get_aws_session(profile_name: Optional[str] = None) -> boto3.Session:
    """Create AWS session with optional profile."""
    if profile_name:
        return boto3.Session(profile_name=profile_name)
    else:
        return boto3.Session()


def get_env_or_param(param_value: Optional[str], env_var: str, param_name: str) -> str:
    """Get value from parameter or environment variable, with validation."""
    if param_value:
        return param_value

    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    raise ValueError(
        f"{param_name} must be provided either as parameter or via {env_var} environment variable"
    )


def get_optional_env_or_param(
    param_value: Optional[str], env_var: str
) -> Optional[str]:
    """Get optional value from parameter or environment variable."""
    if param_value:
        return param_value
    return os.getenv(env_var)


@mcp.tool()
def s3vectors_put(
    text: str,
    vector_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Embed text and store as vector in S3 Vectors.

    Uses environment variables for configuration:
    - S3VECTORS_BUCKET_NAME: S3 bucket name for vector storage
    - S3VECTORS_INDEX_NAME: Vector index name
    - S3VECTORS_MODEL_ID: Bedrock embedding model ID
    - S3VECTORS_REGION: AWS region (optional)
    - S3VECTORS_PROFILE: AWS profile name (optional)
    - S3VECTORS_DIMENSIONS: Embedding dimensions (optional)

    Args:
        text: Text to embed and store
        vector_id: Optional vector ID (auto-generated if not provided)
        metadata: Optional metadata to store with the vector

    Returns:
        JSON string with operation result
    """
    logger.info(
        f"s3vectors_put called with text_length={len(text)}, vector_id={vector_id}, has_metadata={metadata is not None}"
    )

    try:
        # Get required parameters from env vars
        bucket_name = get_env_or_param(
            None, "S3VECTORS_BUCKET_NAME", "S3VECTORS_BUCKET_NAME"
        )
        idx_name = get_env_or_param(
            None, "S3VECTORS_INDEX_NAME", "S3VECTORS_INDEX_NAME"
        )
        mdl_id = get_env_or_param(None, "S3VECTORS_MODEL_ID", "S3VECTORS_MODEL_ID")

        logger.info(
            f"Configuration: bucket={bucket_name}, index={idx_name}, model={mdl_id}"
        )

        # Get optional parameters
        aws_region = get_optional_env_or_param(None, "S3VECTORS_REGION") or get_region()
        aws_profile = get_optional_env_or_param(None, "S3VECTORS_PROFILE")
        dimensions_str = get_optional_env_or_param(None, "S3VECTORS_DIMENSIONS")
        dimensions = int(dimensions_str) if dimensions_str else None

        logger.info(
            f"Optional config: region={aws_region}, profile={aws_profile}, dimensions={dimensions}"
        )

        # Set defaults
        if vector_id is None:
            vector_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}

        logger.info(f"Using vector_id={vector_id}")

        # Initialize AWS session and services
        logger.info("Initializing AWS session and services...")
        session = get_aws_session(aws_profile)
        bedrock_service = BedrockService(session, aws_region, debug=False)
        s3vector_service = S3VectorService(session, aws_region, debug=False)

        # Generate embedding
        logger.info(f"Generating embedding for text with model {mdl_id}...")
        embedding = bedrock_service.embed_text(mdl_id, text, dimensions)
        logger.info(f"Generated embedding with {len(embedding)} dimensions")

        # Store vector
        logger.info("Storing vector in S3 Vectors...")
        result_vector_id = s3vector_service.put_vector(
            bucket_name=bucket_name,
            index_name=idx_name,
            vector_id=vector_id,
            embedding=embedding,
            metadata=metadata,
        )
        logger.info(f"Successfully stored vector with ID: {result_vector_id}")

        # Prepare result
        result = {
            "success": True,
            "vector_id": result_vector_id,
            "bucket": bucket_name,
            "index": idx_name,
            "model_id": mdl_id,
            "region": aws_region,
            "profile": aws_profile,
            "text_length": len(text),
            "embedding_dimensions": len(embedding),
            "metadata": metadata,
        }

        logger.info("s3vectors_put completed successfully")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in s3vectors_put: {str(e)}", exc_info=True)
        error_result = {"success": False, "error": str(e), "operation": "s3vectors_put"}
        return json.dumps(error_result, indent=2)


@mcp.tool()
def s3vectors_query(
    query_text: str,
    top_k: int = 10,
    filter_expr: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    return_distance: bool = False,
) -> str:
    """
    Query for similar vectors in S3 Vectors.

    Uses environment variables for configuration:
    - S3VECTORS_BUCKET_NAME: S3 bucket name for vector storage
    - S3VECTORS_INDEX_NAME: Vector index name
    - S3VECTORS_MODEL_ID: Bedrock embedding model ID
    - S3VECTORS_DIMENSIONS: Embedding dimensions (optional)
    - S3VECTORS_REGION: AWS region (optional)
    - S3VECTORS_PROFILE: AWS profile name (optional)

    Args:
        query_text: Text to query for similar vectors
        top_k: Number of similar vectors to return (default: 10)
        filter_expr: Optional metadata filter expression (JSON format)
        return_metadata: Include metadata in results (default: True)
        return_distance: Include similarity distance in results (default: False)

    Filter Expression (filter_expr):
        Supports AWS S3 Vectors API operators for metadata-based filtering.

        Comparison Operators:
        - $eq: Equal to
        - $ne: Not equal to
        - $gt: Greater than
        - $gte: Greater than or equal to
        - $lt: Less than
        - $lte: Less than or equal to
        - $in: Value in array
        - $nin: Value not in array

        Logical Operators:
        - $and: Logical AND (all conditions must be true)
        - $or: Logical OR (at least one condition must be true)
        - $not: Logical NOT (condition must be false)

        Examples:

        Single condition filters:
        {"category": {"$eq": "documentation"}}
        {"status": {"$ne": "archived"}}
        {"version": {"$gte": "2.0"}}
        {"category": {"$in": ["docs", "guides", "tutorials"]}}

        Multiple condition filters:
        {"$and": [{"category": "tech"}, {"version": "1.0"}]}
        {"$or": [{"category": "docs"}, {"category": "guides"}]}
        {"$not": {"category": {"$eq": "archived"}}}

        Complex nested conditions:
        {"$and": [{"category": "tech"}, {"$or": [{"version": "1.0"}, {"version": "2.0"}]}]}
        {"$and": [{"category": "documentation"}, {"version": {"$gte": "1.0"}}, {"status": {"$ne": "draft"}}]}
        {"$or": [{"$and": [{"category": "docs"}, {"version": "1.0"}]}, {"$and": [{"category": "guides"}, {"version": "2.0"}]}]}

        Notes:
        - String comparisons are case-sensitive
        - Ensure filter values match the data types in your metadata
        - Use proper JSON format with double quotes for keys and string values

    Returns:
        JSON string with query results
    """
    logger.info(
        f"s3vectors_query called with query_text_length={len(query_text)}, top_k={top_k}, has_filter={filter_expr is not None}, return_metadata={return_metadata}, return_distance={return_distance}"
    )

    try:
        # Get required parameters from env vars
        bucket_name = get_env_or_param(
            None, "S3VECTORS_BUCKET_NAME", "S3VECTORS_BUCKET_NAME"
        )
        idx_name = get_env_or_param(
            None, "S3VECTORS_INDEX_NAME", "S3VECTORS_INDEX_NAME"
        )
        mdl_id = get_env_or_param(None, "S3VECTORS_MODEL_ID", "S3VECTORS_MODEL_ID")

        logger.info(
            f"Configuration: bucket={bucket_name}, index={idx_name}, model={mdl_id}"
        )

        # Get optional parameters
        aws_region = get_optional_env_or_param(None, "S3VECTORS_REGION") or get_region()
        aws_profile = get_optional_env_or_param(None, "S3VECTORS_PROFILE")
        dimensions_str = get_optional_env_or_param(None, "S3VECTORS_DIMENSIONS")
        dimensions = int(dimensions_str) if dimensions_str else None

        logger.info(
            f"Optional config: region={aws_region}, profile={aws_profile}, dimensions={dimensions}"
        )

        # Initialize AWS session and services
        logger.info("Initializing AWS session and services...")
        session = get_aws_session(aws_profile)
        bedrock_service = BedrockService(session, aws_region, debug=False)
        s3vector_service = S3VectorService(session, aws_region, debug=False)

        # Generate query embedding
        logger.info(f"Generating query embedding for text with model {mdl_id}...")
        query_embedding = bedrock_service.embed_text(mdl_id, query_text, dimensions)
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")

        # Prepare query parameters
        query_params = {
            "bucket_name": bucket_name,
            "index_name": idx_name,
            "query_embedding": query_embedding,
            "k": top_k,
            "return_metadata": return_metadata,
            "return_distance": return_distance,
        }

        # Add optional parameters if provided
        if filter_expr:
            # Convert dict to JSON string as S3VectorService expects string
            filter_expr_str = json.dumps(filter_expr)
            query_params["filter_expr"] = filter_expr_str

        logger.info(f"Query parameters: {query_params}")
        logger.info("Calling s3vector_service.query_vectors...")

        # Perform vector search
        search_results = s3vector_service.query_vectors(**query_params)
        logger.info(f"Query completed successfully. Raw results: {search_results}")

        # Format results
        formatted_results = []
        for result in search_results:
            formatted_result = {
                "vector_id": result.get("vectorId"),
                "similarity": result.get("similarity"),
            }

            if return_metadata and "metadata" in result:
                formatted_result["metadata"] = result.get("metadata", {})

            if return_distance and "similarity" in result:
                # S3VectorService returns 'similarity' which is the distance/score
                formatted_result["distance"] = result.get("similarity")

            formatted_results.append(formatted_result)

        logger.info(f"Formatted {len(formatted_results)} results")

        # Prepare response
        response = {
            "success": True,
            "query_text": query_text,
            "bucket": bucket_name,
            "index": idx_name,
            "model_id": mdl_id,
            "region": aws_region,
            "profile": aws_profile,
            "top_k": top_k,
            "filter": filter_expr,
            "return_metadata": return_metadata,
            "return_distance": return_distance,
            "query_embedding_dimensions": len(query_embedding),
            "results_count": len(formatted_results),
            "results": formatted_results,
        }

        logger.info("s3vectors_query completed successfully")
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error in s3vectors_query: {str(e)}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e),
            "operation": "s3vectors_query",
        }
        return json.dumps(error_result, indent=2)


def serve(transport: str = "stdio") -> None:
    """Start the S3 Vectors MCP server with the specified transport."""
    logger.info("Starting S3 Vectors MCP Server...")
    logger.info(f"Using transport: {transport}")

    # Log environment variables for debugging
    env_vars = [
        "S3VECTORS_BUCKET_NAME",
        "S3VECTORS_INDEX_NAME",
        "S3VECTORS_MODEL_ID",
        "S3VECTORS_DIMENSIONS",
        "S3VECTORS_REGION",
        "S3VECTORS_PROFILE",
    ]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"Environment variable {var}={value}")
        else:
            logger.info(f"Environment variable {var} not set")

    # Run the FastMCP server with the selected transport
    logger.info("Starting FastMCP server...")
    mcp.run(transport=transport)


def main() -> None:
    """Main entry point with transport selection."""
    parser = argparse.ArgumentParser(
        description="S3 Vectors MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  s3-vectors-mcp stdio                    # Use stdio transport (default)
  s3-vectors-mcp sse                      # Use Server-Sent Events transport
  s3-vectors-mcp streamable-http          # Use StreamableHTTP transport
        """,
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
