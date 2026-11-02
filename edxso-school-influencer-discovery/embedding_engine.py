"""
embedding_engine.py
===================
AI Embedding Engine for the EDXSO School Influencer Discovery Tool.
Generates dense vector embeddings using Sentence Transformers (all-MiniLM-L6-v2)
and computes semantic cosine similarity across candidate biographies.
Includes in-memory embedding caching and a graceful TF-IDF fallback.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = None
_CACHE: Dict[str, np.ndarray] = {}
_FALLBACK_MODE = False


def get_model():
    """Lazy-load and cache the SentenceTransformer model instance."""
    global _MODEL, _FALLBACK_MODE
    if _MODEL is None and not _FALLBACK_MODE:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(MODEL_NAME)
            logger.info("Loaded SentenceTransformer model: %s", MODEL_NAME)
        except Exception as exc:
            logger.warning(
                "Could not initialize SentenceTransformer (%s). Activating TF-IDF semantic fallback: %s",
                MODEL_NAME,
                exc,
            )
            _FALLBACK_MODE = True
    return _MODEL


def create_profile_text(profile: Union[Dict[str, Any], Any]) -> str:
    """
    Combine candidate professional attributes into a rich descriptive text
    representation for semantic embedding generation.
    """
    if hasattr(profile, "to_dict"):
        p = profile.to_dict()
    elif isinstance(profile, dict):
        p = profile
    else:
        p = dict(profile)

    name = p.get("name", "")
    current_role = p.get("current_role", "")
    org = p.get("organisation", "")
    location = p.get("location", "")
    bio = p.get("biography", "")
    past = p.get("past_roles", "")
    school_exp = p.get("school_experience", "")

    tags = p.get("professional_tags", "")
    if isinstance(tags, list):
        tags_str = ", ".join(tags)
    else:
        tags_str = str(tags)

    return (
        f"Professional: {name}. "
        f"Current Role: {current_role} at {org}. "
        f"Location: {location}. "
        f"Professional Background: {bio} "
        f"Previous Positions: {past}. "
        f"School Education Experience: {school_exp}. "
        f"Expertise Tags: {tags_str}."
    ).strip()


def _hash_text(text: str) -> str:
    """Compute sha256 hash of text for caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate normalized dense vector embeddings for a list of text strings.
    Utilizes in-memory caching for previously encoded strings.
    """
    model = get_model()

    # Check which texts need encoding
    needed_indices = []
    needed_texts = []
    embeddings_list: List[Optional[np.ndarray]] = [None] * len(texts)

    for i, t in enumerate(texts):
        h = _hash_text(t)
        if h in _CACHE:
            embeddings_list[i] = _CACHE[h]
        else:
            needed_indices.append(i)
            needed_texts.append(t)

    if needed_texts:
        if model is not None and not _FALLBACK_MODE:
            try:
                new_vecs = model.encode(needed_texts, normalize_embeddings=True)
                if not isinstance(new_vecs, np.ndarray):
                    new_vecs = np.array(new_vecs)
                for idx, text, vec in zip(needed_indices, needed_texts, new_vecs):
                    _CACHE[_hash_text(text)] = vec
                    embeddings_list[idx] = vec
            except Exception as e:
                logger.warning("Encoding failed with transformer: %s. Using TF-IDF fallback.", e)
                fallback_vecs = _tfidf_fallback_encode(needed_texts, texts)
                for idx, text, vec in zip(needed_indices, needed_texts, fallback_vecs):
                    _CACHE[_hash_text(text)] = vec
                    embeddings_list[idx] = vec
        else:
            fallback_vecs = _tfidf_fallback_encode(needed_texts, texts)
            for idx, text, vec in zip(needed_indices, needed_texts, fallback_vecs):
                _CACHE[_hash_text(text)] = vec
                embeddings_list[idx] = vec

    return np.array(embeddings_list)


def _tfidf_fallback_encode(needed_texts: List[str], all_corpus: List[str]) -> np.ndarray:
    """Graceful fallback vectorizer using TF-IDF when transformer is unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=512)
    vectorizer.fit(all_corpus)
    vecs = vectorizer.transform(needed_texts).toarray()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def calculate_similarity(reference_text: str, candidate_texts: List[str]) -> np.ndarray:
    """
    Compute cosine similarity between a reference profile description
    and a list of candidate profile descriptions.

    Returns:
        1D numpy array of similarity scores in [0.0, 1.0].
    """
    if not candidate_texts:
        return np.array([], dtype=float)

    ref_vec = generate_embeddings([reference_text])[0]
    cand_vecs = generate_embeddings(candidate_texts)

    # Dot product of normalized vectors equals cosine similarity
    similarities = np.dot(cand_vecs, ref_vec)
    # Clip between 0.0 and 1.0 for stability
    return np.clip(similarities, 0.0, 1.0)
