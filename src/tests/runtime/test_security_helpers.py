"""Tests for llama_manager.common.security helpers."""

from llama_manager.common.security import (
    REDACTED_VALUE,
    is_sensitive_key,
    redact_env_value,
    redact_log_line,
    redact_sensitive,
    safe_log,
)

# ---------------------------------------------------------------------------
# TestIsSensitiveKey
# ---------------------------------------------------------------------------


class TestIsSensitiveKey:
    """Tests for is_sensitive_key()."""

    def test_key_keyword_detected(self) -> None:
        """Should detect 'KEY' in key name."""
        assert is_sensitive_key("API_KEY") is True

    def test_token_keyword_detected(self) -> None:
        """Should detect 'TOKEN' in key name."""
        assert is_sensitive_key("AUTH_TOKEN") is True

    def test_secret_keyword_detected(self) -> None:
        """Should detect 'SECRET' in key name."""
        assert is_sensitive_key("AWS_SECRET_ACCESS_KEY") is True

    def test_password_keyword_detected(self) -> None:
        """Should detect 'PASSWORD' in key name."""
        assert is_sensitive_key("DB_PASSWORD") is True

    def test_auth_keyword_detected(self) -> None:
        """Should detect 'AUTH' in key name."""
        assert is_sensitive_key("OAUTH_TOKEN") is True

    def test_auth_header_keyword_detected(self) -> None:
        """Should detect 'AUTH_HEADER' in key name."""
        assert is_sensitive_key("PROXY_AUTH_HEADER") is True

    def test_case_insensitive(self) -> None:
        """Should be case-insensitive."""
        assert is_sensitive_key("api_key") is True
        assert is_sensitive_key("Api_Key") is True
        assert is_sensitive_key("API_KEY") is True

    def test_trailing_key_not_matched(self) -> None:
        """'MONKEY' should not match — KEY is trailing with no boundary."""
        assert is_sensitive_key("MONKEY") is False

    def test_leading_key_not_matched(self) -> None:
        """'KEYCHAIN' should not match — KEY is leading with no boundary."""
        assert is_sensitive_key("KEYCHAIN") is False

    def test_suffix_pattern_with_digits(self) -> None:
        """KEY_123 should match (suffix pattern allows _DIGITS)."""
        assert is_sensitive_key("API_KEY_123") is True

    def test_non_sensitive_key(self) -> None:
        """Should return False for non-sensitive key names."""
        assert is_sensitive_key("MODEL_PATH") is False
        assert is_sensitive_key("CONTEXT_SIZE") is False
        assert is_sensitive_key("THREADS") is False

    def test_empty_string(self) -> None:
        """Empty string should not match."""
        assert is_sensitive_key("") is False

    def test_random_string(self) -> None:
        """Random string should not match."""
        assert is_sensitive_key("foobar") is False


# ---------------------------------------------------------------------------
# TestRedactEnvValue
# ---------------------------------------------------------------------------


class TestRedactEnvValue:
    """Tests for redact_env_value()."""

    def test_sensitive_key_redacted(self) -> None:
        """Should redact value when key is sensitive."""
        result = redact_env_value("my_secret", "API_KEY")
        assert result == REDACTED_VALUE

    def test_non_sensitive_key_preserved(self) -> None:
        """Should preserve value when key is not sensitive."""
        result = redact_env_value("/path/to/model", "MODEL_PATH")
        assert result == "/path/to/model"

    def test_empty_value_redacted_for_sensitive(self) -> None:
        """Should redact empty value for sensitive keys too."""
        result = redact_env_value("", "SECRET_TOKEN")
        assert result == REDACTED_VALUE

    def test_password_key_redacted(self) -> None:
        """PASSWORD keys should be redacted."""
        result = redact_env_value("p@ssw0rd", "DB_PASSWORD")
        assert result == REDACTED_VALUE

    def test_token_key_redacted(self) -> None:
        """TOKEN keys should be redacted."""
        result = redact_env_value("abc123", "ACCESS_TOKEN")
        assert result == REDACTED_VALUE


# ---------------------------------------------------------------------------
# TestRedactLogLine
# ---------------------------------------------------------------------------


class TestRedactLogLine:
    """Tests for redact_log_line()."""

    def test_redacts_unquoted_value(self) -> None:
        """Should redact unquoted KEY=value pairs."""
        result = redact_log_line("API_KEY=abc123secret")
        assert result == "API_KEY=[REDACTED]"

    def test_redacts_quoted_value(self) -> None:
        """Should redact quoted KEY=value pairs."""
        result = redact_log_line('API_KEY="my secret value"')
        assert result == "API_KEY=[REDACTED]"

    def test_preserves_non_sensitive_pairs(self) -> None:
        """Should not redact non-sensitive KEY=value pairs."""
        result = redact_log_line("MODEL_PATH=/path/to/model.gguf")
        assert result == "MODEL_PATH=/path/to/model.gguf"

    def test_redacts_multiple_pairs(self) -> None:
        """Should redact all sensitive pairs in a line."""
        result = redact_log_line("API_KEY=abc123 DB_PASSWORD=secret MODEL=/path")
        assert "API_KEY=[REDACTED]" in result
        assert "DB_PASSWORD=[REDACTED]" in result
        assert "MODEL=/path" in result

    def test_preserves_safe_lines(self) -> None:
        """Lines without sensitive pairs should be unchanged."""
        line = "INFO: Starting server on port 8080"
        assert redact_log_line(line) == line

    def test_redacts_token_in_middle_of_line(self) -> None:
        """Sensitive pairs in the middle of a line should be redacted."""
        result = redact_log_line("Connecting with SECRET_KEY=xyz to server")
        assert "SECRET_KEY=[REDACTED]" in result
        assert "to server" in result

    def test_empty_line(self) -> None:
        """Empty line should remain empty."""
        assert redact_log_line("") == ""

    def test_no_equals_sign(self) -> None:
        """Lines without = should not be affected."""
        assert redact_log_line("just a plain log line") == "just a plain log line"


# ---------------------------------------------------------------------------
# TestSafeLog
# ---------------------------------------------------------------------------


class TestSafeLog:
    """Tests for safe_log() — t-string redaction."""

    def test_redacts_sensitive_interpolation(self) -> None:
        """Should redact values bound to sensitive variable names."""
        api_key = "sk-abc123"
        result = safe_log(t"Connecting with api_key={api_key}")
        assert result == "Connecting with api_key=[REDACTED]"

    def test_preserves_safe_interpolation(self) -> None:
        """Should preserve values bound to non-sensitive variable names."""
        model_path = "/path/to/model.gguf"
        result = safe_log(t"Loading model from model_path={model_path}")
        assert result == "Loading model from model_path=/path/to/model.gguf"

    def test_redacts_password_interpolation(self) -> None:
        """PASSWORD-named variables should be redacted."""
        db_password = "super_secret"  # noqa: S105
        result = safe_log(t"Auth with db_password={db_password}")
        assert result == "Auth with db_password=[REDACTED]"

    def test_mixed_sensitive_and_safe(self) -> None:
        """Should redact sensitive and preserve safe in the same template."""
        api_key = "sk-abc"
        server = "localhost"
        result = safe_log(t"key={api_key} on {server}")
        assert result == "key=[REDACTED] on localhost"

    def test_no_interpolations(self) -> None:
        """A template with no interpolations should pass through unchanged."""
        result = safe_log(t"Hello, world!")
        assert result == "Hello, world!"

    def test_multiple_sensitive_interpolations(self) -> None:
        """All sensitive interpolations should be redacted."""
        token = "tok-123"  # noqa: S105
        secret = "sec-456"  # noqa: S105
        result = safe_log(t"token={token} secret={secret}")
        assert result == "token=[REDACTED] secret=[REDACTED]"

    def test_sensitive_via_log_pattern(self) -> None:
        """Variable names matching the log pattern should also be redacted."""
        proxy_auth_header = "bearer_xyz"
        result = safe_log(t"header={proxy_auth_header}")
        assert result == "header=[REDACTED]"


# ---------------------------------------------------------------------------
# TestRedactSensitive
# ---------------------------------------------------------------------------


class TestRedactSensitive:
    """T014: Tests for redact_sensitive() function."""

    def test_redact_sensitive_api_key(self) -> None:
        """redact_sensitive should redact API_KEY values."""
        assert redact_sensitive("API_KEY=secret123") == "API_KEY: [REDACTED]"
        assert redact_sensitive("api_key=lowercase") == "api_key: [REDACTED]"
        assert redact_sensitive("Api_Key=MixedCase") == "Api_Key: [REDACTED]"

    def test_redact_sensitive_token(self) -> None:
        """redact_sensitive should redact TOKEN values."""
        assert redact_sensitive("TOKEN=abc123") == "TOKEN: [REDACTED]"
        assert redact_sensitive("auth_token=xyz789") == "auth_token: [REDACTED]"

    def test_redact_sensitive_secret(self) -> None:
        """redact_sensitive should redact SECRET values."""
        assert redact_sensitive("SECRET=mysecret") == "SECRET: [REDACTED]"
        assert redact_sensitive("api_secret: secret123") == "api_secret: [REDACTED]"

    def test_redact_sensitive_password(self) -> None:
        """redact_sensitive should redact PASSWORD values."""
        assert redact_sensitive("PASSWORD=supersecret") == "PASSWORD: [REDACTED]"
        assert redact_sensitive("db_password: mypass") == "db_password: [REDACTED]"

    def test_redact_sensitive_auth(self) -> None:
        """redact_sensitive should redact AUTH values."""
        assert redact_sensitive("AUTH_HEADER=bearer_xyz") == "AUTH_HEADER: [REDACTED]"
        assert redact_sensitive("auth_token=token123") == "auth_token: [REDACTED]"

    def test_redact_sensitive_case_insensitive(self) -> None:
        """redact_sensitive should work case-insensitively."""
        assert redact_sensitive("API_KEY=secret") == "API_KEY: [REDACTED]"
        assert redact_sensitive("api_key=secret") == "api_key: [REDACTED]"
        assert redact_sensitive("Api_Key=secret") == "Api_Key: [REDACTED]"

    def test_redact_sensitive_standalone_key(self) -> None:
        """redact_sensitive should redact standalone sensitive words."""
        assert redact_sensitive("KEY") == "[REDACTED]"
        assert redact_sensitive("TOKEN") == "[REDACTED]"
        assert redact_sensitive("SECRET") == "[REDACTED]"
        assert redact_sensitive("PASSWORD") == "[REDACTED]"
        assert redact_sensitive("AUTH") == "[REDACTED]"

    def test_redact_sensitive_no_redaction_needed(self) -> None:
        """redact_sensitive should return unchanged text when no sensitive data."""
        assert redact_sensitive("normal log message") == "normal log message"
        assert redact_sensitive("Building model from /path/to/model.gguf") == (
            "Building model from /path/to/model.gguf"
        )
        assert redact_sensitive("Server started on port 8080") == "Server started on port 8080"

    def test_redact_sensitive_multiple_in_one_line(self) -> None:
        """redact_sensitive should redact multiple sensitive values in one line."""
        text = "API_KEY=one TOKEN=two PASSWORD=three"
        result = redact_sensitive(text)
        # All three should be redacted
        assert result.count("[REDACTED]") == 3
        assert "one" not in result
        assert "two" not in result
        assert "three" not in result

    def test_redact_sensitive_key_value_with_colon(self) -> None:
        """redact_sensitive should handle key:value format."""
        assert redact_sensitive("api-key: secret123") == "api-key: [REDACTED]"
        assert redact_sensitive("DB_PASSWORD: mypass") == "DB_PASSWORD: [REDACTED]"

    def test_redact_sensitive_key_value_with_equals(self) -> None:
        """redact_sensitive should handle key=value format."""
        assert redact_sensitive("api-key=secret123") == "api-key: [REDACTED]"
        assert redact_sensitive("MY_TOKEN=abc123") == "MY_TOKEN: [REDACTED]"

    def test_redact_sensitive_key_value_with_space(self) -> None:
        """redact_sensitive should handle key value format (space separated)."""
        result = redact_sensitive("api-key secret123")
        # Should redact the value part
        assert "[REDACTED]" in result

    def test_redact_sensitive_preserves_other_content(self) -> None:
        """redact_sensitive should preserve non-sensitive content."""
        text = "2024-01-01 12:00:00 API_KEY=secret Server started"
        result = redact_sensitive(text)
        assert "2024-01-01 12:00:00" in result
        assert "Server started" in result
        assert "[REDACTED]" in result

    def test_redact_sensitive_empty_string(self) -> None:
        """redact_sensitive should handle empty string."""
        assert redact_sensitive("") == ""

    def test_redact_sensitive_only_sensitive_word(self) -> None:
        """redact_sensitive should redact lines that are only sensitive words."""
        # Standalone sensitive words should be fully redacted
        assert redact_sensitive("API_KEY") == "[REDACTED]"
        assert redact_sensitive("TOKEN") == "[REDACTED]"
        assert redact_sensitive("SECRET") == "[REDACTED]"
        assert redact_sensitive("PASSWORD") == "[REDACTED]"
        assert redact_sensitive("AUTH") == "[REDACTED]"
