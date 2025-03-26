"""
Specialized code embedding module with support for different models and optimizations.
"""
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import ast
from pathlib import Path
import tempfile
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for code embeddings."""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    use_pooling: bool = True
    pooling_strategy: str = "mean"
    use_ast: bool = False
    hybrid_embedding: bool = False
    cache_embeddings: bool = True
    cache_dir: str = ".embedding_cache"

class CodeEmbedder:
    """Specialized code embedding model with optimizations."""
    
    # Code-specific embedding models
    CODE_MODELS = {
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
        "codet5": "Salesforce/codet5-base",
        "unixcoder": "microsoft/unixcoder-base",
        "code-t5": "Salesforce/codet5-base",
        "code-gpt": "Codium/codium-gpt-1.0",
        "default": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Language parsers for AST extraction
    LANGUAGE_PARSERS = {
        "python": "ast",
        "javascript": "esprima",
        "typescript": "typescript",
        "java": "javalang",
        "cpp": "clang",
        "go": "go",
        "kotlin": "kotlin-compiler",
        "rust": "rust-analyzer"
    }
    
    # File extension to language mapping
    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "cpp",
        ".hpp": "cpp",
        ".go": "go",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".rs": "rust"
    }
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the code embedder."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embedding_cache = {}
        
        # Create cache directory if needed
        if self.config.cache_embeddings:
            Path(self.config.cache_dir).mkdir(exist_ok=True, parents=True)
        
        try:
            # Resolve model name if it's a shorthand
            if config.model_name in self.CODE_MODELS:
                model_name = self.CODE_MODELS[config.model_name]
            else:
                model_name = config.model_name
            
            # Initialize model based on type
            if "sentence-transformers" in model_name:
                self.model = SentenceTransformer(
                    model_name,
                    device=config.device
                )
                self.model_type = "sentence_transformer"
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    device_map=config.device
                )
                self.model_type = "transformer"
            
            self.logger.info(f"Initialized embedding model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def embed_code(
        self,
        code_snippets: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """Generate embeddings for code snippets with optimizations."""
        try:
            # Check cache first if enabled
            if self.config.cache_embeddings:
                cached_embeddings, snippets_to_embed, indices = self._check_cache(code_snippets)
                if len(snippets_to_embed) == 0:
                    self.logger.info("Using cached embeddings for all snippets")
                    return cached_embeddings
            else:
                snippets_to_embed = code_snippets
                indices = list(range(len(code_snippets)))
            
            # Process code snippets
            processed_snippets = []
            ast_features = []
            
            for i, snippet in enumerate(snippets_to_embed):
                # Determine language from metadata
                language = None
                if metadata and i < len(metadata) and "language" in metadata[i]:
                    language = metadata[i]["language"]
                elif metadata and i < len(metadata) and "file_path" in metadata[i]:
                    file_ext = Path(metadata[i]["file_path"]).suffix
                    language = self.EXTENSION_TO_LANGUAGE.get(file_ext.lower())
                
                # Preprocess code
                processed_snippet = self._preprocess_code(snippet, language)
                processed_snippets.append(processed_snippet)
                
                # Extract AST features if enabled
                if self.config.use_ast and language:
                    ast_feature = self._extract_ast_features(snippet, language)
                    ast_features.append(ast_feature)
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(processed_snippets), self.config.batch_size):
                batch = processed_snippets[i:i + self.config.batch_size]
                
                if self.model_type == "sentence_transformer":
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        normalize_embeddings=self.config.normalize
                    )
                else:
                    # Tokenize and generate embeddings
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.config.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Apply pooling if configured
                    if self.config.use_pooling:
                        batch_embeddings = self._apply_pooling(outputs.last_hidden_state)
                    else:
                        batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    # Convert to numpy and normalize if needed
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    if self.config.normalize:
                        batch_embeddings = self._normalize_embeddings(batch_embeddings)
                
                embeddings.append(batch_embeddings)
            
            # Combine all batches
            if embeddings:
                new_embeddings = np.vstack(embeddings)
                
                # Combine with AST features if available
                if self.config.use_ast and ast_features:
                    ast_features_array = np.array(ast_features)
                    new_embeddings = np.hstack([new_embeddings, ast_features_array])
                
                # Add metadata features if provided
                if metadata:
                    new_embeddings = self._add_metadata_features(new_embeddings, metadata)
                
                # Update cache for new embeddings
                if self.config.cache_embeddings:
                    self._update_cache(snippets_to_embed, new_embeddings)
                    
                    # Combine cached and new embeddings
                    all_embeddings = np.zeros((len(code_snippets), new_embeddings.shape[1]))
                    cached_indices = [i for i in range(len(code_snippets)) if i not in indices]
                    
                    for i, idx in enumerate(cached_indices):
                        all_embeddings[idx] = cached_embeddings[i]
                    for i, idx in enumerate(indices):
                        all_embeddings[idx] = new_embeddings[i]
                    
                    return all_embeddings
                
                return new_embeddings
            
            return cached_embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _check_cache(self, code_snippets: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
        """Check if embeddings are in cache and return uncached snippets."""
        snippets_to_embed = []
        indices = []
        cached_embeddings = []
        
        for i, snippet in enumerate(code_snippets):
            snippet_hash = self._hash_snippet(snippet)
            cache_file = Path(self.config.cache_dir) / f"{snippet_hash}.npy"
            
            if cache_file.exists():
                try:
                    embedding = np.load(str(cache_file))
                    cached_embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Error loading cached embedding: {str(e)}")
                    snippets_to_embed.append(snippet)
                    indices.append(i)
            else:
                snippets_to_embed.append(snippet)
                indices.append(i)
        
        if cached_embeddings:
            cached_embeddings = np.stack(cached_embeddings)
        else:
            cached_embeddings = np.array([])
            
        return cached_embeddings, snippets_to_embed, indices
    
    def _update_cache(self, code_snippets: List[str], embeddings: np.ndarray):
        """Update cache with new embeddings."""
        for i, snippet in enumerate(code_snippets):
            snippet_hash = self._hash_snippet(snippet)
            cache_file = Path(self.config.cache_dir) / f"{snippet_hash}.npy"
            np.save(str(cache_file), embeddings[i])
    
    def _hash_snippet(self, snippet: str) -> str:
        """Generate hash for a code snippet."""
        import hashlib
        return hashlib.md5(snippet.encode()).hexdigest()
    
    def _preprocess_code(self, code: str, language: Optional[str] = None) -> str:
        """Preprocess code snippet for better embedding quality."""
        # Remove comments and format based on language
        if not code or not code.strip():
            return ""
            
        if language == "python":
            return self._preprocess_python(code)
        elif language in ["javascript", "typescript"]:
            return self._preprocess_js_ts(code)
        elif language == "java":
            return self._preprocess_java(code)
        elif language in ["cpp", "c"]:
            return self._preprocess_cpp(code)
        elif language == "go":
            return self._preprocess_go(code)
        elif language == "kotlin":
            return self._preprocess_kotlin(code)
        elif language == "rust":
            return self._preprocess_rust(code)
        
        # Generic preprocessing for unknown languages
        return self._preprocess_generic(code)
    
    def _preprocess_python(self, code: str) -> str:
        """Preprocess Python code."""
        lines = code.split("\n")
        cleaned_lines = []
        in_multiline_comment = False
        
        for line in lines:
            # Handle multiline strings/docstrings
            if '"""' in line or "'''" in line:
                if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                    in_multiline_comment = not in_multiline_comment
                    # Extract class/function name from docstring
                    if not in_multiline_comment and len(cleaned_lines) > 0:
                        # Check for class/def definition in previous lines
                        for i in range(len(cleaned_lines)-1, max(0, len(cleaned_lines)-3), -1):
                            if cleaned_lines[i].strip().startswith(("def ", "class ")):
                                # Keep the line but remove the actual comment
                                line = cleaned_lines[i]
                                break
                continue
                
            if in_multiline_comment:
                continue
                
            # Remove single-line comments
            if "#" in line:
                line = line.split("#")[0]
                
            # Remove empty lines
            if line.strip():
                cleaned_lines.append(line)
                
        # Add imports at the top
        code_with_imports = []
        regular_lines = []
        
        for line in cleaned_lines:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                code_with_imports.append(line)
            else:
                regular_lines.append(line)
                
        return "\n".join(code_with_imports + regular_lines)
    
    def _preprocess_js_ts(self, code: str) -> str:
        """Preprocess JavaScript/TypeScript code."""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        
        # Process remaining code
        lines = code.split("\n")
        cleaned_lines = []
        
        imports = []
        regular_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(("import ", "export ")):
                imports.append(line)
            else:
                regular_lines.append(line)
        
        return "\n".join(imports + regular_lines)
    
    def _preprocess_java(self, code: str) -> str:
        """Preprocess Java code."""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        
        # Process remaining code
        lines = code.split("\n")
        cleaned_lines = []
        
        imports = []
        package_lines = []
        regular_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("import "):
                imports.append(line)
            elif line.startswith("package "):
                package_lines.append(line)
            else:
                regular_lines.append(line)
        
        return "\n".join(package_lines + imports + regular_lines)
    
    def _preprocess_cpp(self, code: str) -> str:
        """Preprocess C++ code."""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        
        # Process remaining code
        lines = code.split("\n")
        cleaned_lines = []
        
        includes = []
        regular_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("#include"):
                includes.append(line)
            else:
                regular_lines.append(line)
        
        return "\n".join(includes + regular_lines)
    
    def _preprocess_go(self, code: str) -> str:
        """Preprocess Go code."""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        
        # Process remaining code
        lines = code.split("\n")
        
        package_lines = []
        imports = []
        regular_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("package "):
                package_lines.append(line)
            elif line.startswith("import "):
                imports.append(line)
            else:
                regular_lines.append(line)
        
        return "\n".join(package_lines + imports + regular_lines)
    
    def _preprocess_kotlin(self, code: str) -> str:
        """Preprocess Kotlin code."""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        
        # Process remaining code
        lines = code.split("\n")
        
        package_lines = []
        imports = []
        regular_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("package "):
                package_lines.append(line)
            elif line.startswith("import "):
                imports.append(line)
            else:
                regular_lines.append(line)
        
        return "\n".join(package_lines + imports + regular_lines)
    
    def _preprocess_rust(self, code: str) -> str:
        """Preprocess Rust code."""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        
        # Process remaining code
        lines = code.split("\n")
        
        use_lines = []
        mod_lines = []
        regular_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("use "):
                use_lines.append(line)
            elif line.startswith("mod "):
                mod_lines.append(line)
            else:
                regular_lines.append(line)
        
        return "\n".join(mod_lines + use_lines + regular_lines)
    
    def _preprocess_generic(self, code: str) -> str:
        """Generic preprocessing for unknown languages."""
        # Remove common comment patterns
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)  # C-style multi-line
        code = re.sub(r'//.*', '', code)            # C-style single-line
        code = re.sub(r'#.*', '', code)             # Shell/Python style
        
        # Remove empty lines
        lines = [line for line in code.split("\n") if line.strip()]
        
        return "\n".join(lines)
    
    def _extract_ast_features(self, code: str, language: str) -> List[float]:
        """Extract Abstract Syntax Tree features."""
        features = [0.0] * 20  # Initialize with zeros
        
        try:
            if language == "python":
                tree = ast.parse(code)
                
                # Count different node types
                features[0] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                features[1] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                features[2] = len([n for n in ast.walk(tree) if isinstance(n, ast.Import)])
                features[3] = len([n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)])
                features[4] = len([n for n in ast.walk(tree) if isinstance(n, ast.Call)])
                features[5] = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
                features[6] = len([n for n in ast.walk(tree) if isinstance(n, ast.For)])
                features[7] = len([n for n in ast.walk(tree) if isinstance(n, ast.While)])
                features[8] = len([n for n in ast.walk(tree) if isinstance(n, ast.Try)])
                features[9] = len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)])
            
            # For Kotlin, use a simpler text-based approach since we don't have a Python AST parser for it
            elif language == "kotlin":
                features[0] = code.count("fun ")  # functions
                features[1] = code.count("class ")  # classes
                features[2] = code.count("import ")  # imports
                features[3] = code.count("package ")  # packages
                features[4] = code.count("(")  # potential function calls
                features[5] = code.count("if ")  # if statements
                features[6] = code.count("for ")  # for loops
                features[7] = code.count("while ")  # while loops
                features[8] = code.count("try ")  # try blocks
                features[9] = code.count(" = ")  # assignments
                features[10] = code.count("val ")  # val declarations
                features[11] = code.count("var ")  # var declarations
                features[12] = code.count("companion object")  # companion objects
                features[13] = code.count("object ")  # objects
                features[14] = code.count("interface ")  # interfaces
            
            # For other languages, use similar text-based approach
            else:
                # Generic feature extraction based on text patterns
                features[0] = len(re.findall(r'\b(function|def|fun|fn)\s+\w+', code))  # functions
                features[1] = len(re.findall(r'\b(class|struct|interface)\s+\w+', code))  # classes/structs
                features[2] = len(re.findall(r'\b(import|include|using|use)\s+', code))  # imports
                features[3] = len(re.findall(r'\b(if|else if|elif)\s+', code))  # conditionals
                features[4] = len(re.findall(r'\b(for|while|foreach)\s+', code))  # loops
                features[5] = len(re.findall(r'\b(try|catch|except)\s+', code))  # exception handling
                
        except Exception as e:
            self.logger.warning(f"Error extracting AST features: {str(e)}")
        
        # Normalize features
        total = sum(features)
        if total > 0:
            features = [f / total for f in features]
        
        return features
    
    def _apply_pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to hidden states."""
        if self.config.pooling_strategy == "mean":
            return torch.mean(hidden_states, dim=1)
        elif self.config.pooling_strategy == "max":
            return torch.max(hidden_states, dim=1)[0]
        elif self.config.pooling_strategy == "cls":
            return hidden_states[:, 0, :]
        elif self.config.pooling_strategy == "weighted_mean":
            # Apply attention-weighted mean
            weights = torch.softmax(torch.sum(hidden_states, dim=2), dim=1).unsqueeze(2)
            return torch.sum(hidden_states * weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        return embeddings / norms
    
    def _add_metadata_features(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Add metadata information to embeddings."""
        # Convert metadata to numerical features
        metadata_features = []
        for meta in metadata:
            features = []
            # Add language encoding
            if "language" in meta:
                features.extend(self._encode_language(meta["language"]))
            # Add file type encoding
            if "file_path" in meta:
                features.extend(self._encode_file_type(meta["file_path"]))
            # Add code structure info
            if "size" in meta:
                features.append(min(1.0, meta["size"] / 10000.0))  # Normalize size
            if "complexity" in meta:
                features.append(min(1.0, meta["complexity"] / 100.0))  # Normalize complexity
            
            metadata_features.append(features)
        
        # Combine embeddings with metadata features
        metadata_array = np.array(metadata_features)
        return np.hstack([embeddings, metadata_array])
    
    def _encode_language(self, language: str) -> List[float]:
        """Encode programming language as numerical features."""
        # One-hot encoding for common languages
        languages = ["python", "javascript", "typescript", "java", "cpp", "c", "go", "kotlin", "rust"]
        encoding = [0.0] * len(languages)
        try:
            idx = languages.index(language.lower())
            encoding[idx] = 1.0
        except ValueError:
            pass
        return encoding
    
    def _encode_file_type(self, file_path: str) -> List[float]:
        """Encode file type as numerical features."""
        # One-hot encoding for common file types
        file_types = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".go", ".kt", ".kts", ".rs"]
        encoding = [0.0] * len(file_types)
        try:
            suffix = Path(file_path).suffix.lower()
            idx = file_types.index(suffix)
            encoding[idx] = 1.0
        except (ValueError, IndexError):
            pass
        return encoding 