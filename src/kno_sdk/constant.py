from typing import Dict, Tuple
from enum import Enum


class EmbeddingMethod(str, Enum):
    OPENAI = "OpenAIEmbedding"
    SBERT = "SBERTEmbedding"
    JINAAI = "JinaAIEmbedding"


MAX_FALLBACK_LINES = 150
TOKEN_LIMIT = 16_000  # per-chunk token cap

LANG_NODE_TARGETS: Dict[str, Tuple[str, ...]] = {
    # Python
    "python": (
        "function_definition",
        "class_definition",
    ),
    # JavaScript
    "javascript": (
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "class_declaration",
    ),
    # TypeScript
    "typescript": (
        "function_declaration",
        "function_signature",
        "arrow_function",
        "method_definition",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
    ),
    # Java
    "java": (
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
    ),
    # Go
    "go": (
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ),
    # C
    "c": (
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    ),
    # C++
    "cpp": (
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
        "template_declaration",
    ),
    # Rust
    "rust": (
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "mod_item",
        "trait_item",
    ),
    # PHP
    "php": (
        "function_definition",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
    ),
    # Ruby
    "ruby": (
        "method",
        "singleton_method",
        "class",
        "module",
    ),
    # Kotlin
    "kotlin": (
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "interface_declaration",
    ),
    "html": (
        "element", 
        "script_element", 
        "style_element"
    ),
}

EXT_TO_LANG = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".json": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    # Java
    ".java": "java",
    # Go
    ".go": "go",
    # C
    ".c": "c",
    ".h": "c",
    # C++
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    # Rust
    ".rs": "rust",
    # PHP
    ".php": "php",
    # Ruby
    ".rb": "ruby",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
}

# extensions that are almost always binary blobs
BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    ".ico",
    ".tiff",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
    ".pdf",
    ".psd",
    ".ai",
    ".eps",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".rar",
    ".exe",
    ".msi",
    ".dll",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
}
