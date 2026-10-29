import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_defaults():
    s = Settings(_env_file=None)

    assert s.CHUNK_SIZE == 800
    assert s.CHUNK_OVERLAP == 120
    assert s.TOP_K == 5
    assert s.MIN_SCORE == 0.55
    assert s.MAX_UPLOAD_MB == 10
    assert s.HISTORY_LIMIT == 10
    assert s.EMBEDDING_BATCH_SIZE == 100
    assert s.CHAT_LLM_BASE_URL == "https://api.groq.com/openai/v1"
    assert s.CHAT_ANSWER_MODEL == "openai/gpt-oss-120b"
    assert s.CHAT_REWRITE_MODEL == "openai/gpt-oss-20b"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP must be smaller"):
        Settings(_env_file=None, CHUNK_SIZE=200, CHUNK_OVERLAP=200)


def test_embedding_batch_size_capped_at_gemini_limit():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, EMBEDDING_BATCH_SIZE=101)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_score_must_be_between_0_and_1(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MIN_SCORE=value)
