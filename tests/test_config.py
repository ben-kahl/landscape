import pytest

from landscape.config import LLM_PROFILES, Settings

pytestmark = pytest.mark.unit


def test_default_profile_is_local_qwen35():
    s = Settings()
    assert s.llm_profile == "local_qwen35"
    p = LLM_PROFILES[s.llm_profile]
    assert p.base_url == "http://llama-server:8080/v1"
    assert p.model == "Qwen3.5-9B-Q4_K_M"
    assert p.api_key_env is None
    assert p.no_think is True
    assert p.ctx_size == 32768


def test_cloud_profile_references_key_by_env_name_only():
    p = LLM_PROFILES["openai_gpt5"]
    assert p.api_key_env == "OPENAI_API_KEY"
    # The registry must never carry a literal secret.
    assert "sk-" not in p.api_key_env


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown LLM profile"):
        Settings(llm_profile="does_not_exist")


def test_llama31_profile_kept_for_ab():
    assert "local_llama31" in LLM_PROFILES
