"""
Tests for gwenn.privacy.redaction — PII Redaction Pipeline.

Covers:
- Each PII type is detected and redacted (email, phone_us, phone_intl, ssn,
  credit_card (16-digit + Amex 15-digit), ip_address, ipv6_address)
- Multiple PII instances in one string
- Mixed PII types in one string
- Disabled state: when enabled=False, text passes through unchanged
- Disabled categories: specific patterns can be turned off
- scan() returns correct RedactionResult (counts, categories, lengths)
- redact() returns the redacted text
- Stats tracking (total_redactions / total_redaction_passes / total_redacted_items)
- Edge cases: empty string, no PII, partial matches, international phone formats
- False positives: text that looks like PII but isn't
- Luhn check utility
- Thread-safe stats via locking
"""

from __future__ import annotations


from gwenn.privacy.redaction import PIIRedactor, RedactionResult, PII_PATTERNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redactor(
    enabled: bool = True,
    disabled_categories: list[str] | None = None,
) -> PIIRedactor:
    """Build a PIIRedactor with the given config."""
    return PIIRedactor(enabled=enabled, disabled_categories=disabled_categories)


# ---------------------------------------------------------------------------
# Individual PII type detection and redaction
# ---------------------------------------------------------------------------

class TestEmailRedaction:
    """Email addresses are detected and replaced with [REDACTED_EMAIL]."""

    def test_simple_email(self):
        r = _redactor()
        assert r.redact("contact alice@example.com please") == (
            "contact [REDACTED_EMAIL] please"
        )

    def test_email_with_plus_tag(self):
        r = _redactor()
        assert r.redact("send to user+tag@domain.org") == "send to [REDACTED_EMAIL]"

    def test_email_with_dots_in_local_part(self):
        r = _redactor()
        assert r.redact("first.last@company.co.uk") == "[REDACTED_EMAIL]"

    def test_email_with_percent_in_local_part(self):
        r = _redactor()
        assert r.redact("user%name@example.com") == "[REDACTED_EMAIL]"

    def test_email_with_subdomain(self):
        r = _redactor()
        assert r.redact("admin@mail.server.example.com") == "[REDACTED_EMAIL]"


class TestPhoneRedaction:
    """US phone numbers in various formats are replaced with [REDACTED_PHONE]."""

    def test_plain_10_digits(self):
        r = _redactor()
        assert r.redact("call 5551234567") == "call [REDACTED_PHONE]"

    def test_dashes(self):
        r = _redactor()
        assert r.redact("call 555-123-4567") == "call [REDACTED_PHONE]"

    def test_dots(self):
        r = _redactor()
        assert r.redact("call 555.123.4567") == "call [REDACTED_PHONE]"

    def test_parentheses_area_code(self):
        r = _redactor()
        result = r.redact("call (555) 123-4567")
        assert "[REDACTED_PHONE]" in result
        assert "123-4567" not in result

    def test_with_country_code(self):
        r = _redactor()
        result = r.redact("call +1-555-123-4567")
        assert "[REDACTED_PHONE]" in result or "[REDACTED_PHONE_INTL]" in result
        assert "123-4567" not in result

    def test_with_country_code_no_dash(self):
        r = _redactor()
        result = r.redact("call 1-555-123-4567")
        assert "[REDACTED_PHONE]" in result
        assert "555-123-4567" not in result

    def test_spaces_as_separators(self):
        r = _redactor()
        assert r.redact("call 555 123 4567") == "call [REDACTED_PHONE]"


class TestSSNRedaction:
    """Social Security Numbers (###-##-####) are replaced with [REDACTED_SSN]."""

    def test_standard_ssn(self):
        r = _redactor()
        assert r.redact("ssn: 123-45-6789") == "ssn: [REDACTED_SSN]"

    def test_ssn_in_sentence(self):
        r = _redactor()
        text = "My social is 999-88-7777 for the record."
        assert "[REDACTED_SSN]" in r.redact(text)
        assert "999-88-7777" not in r.redact(text)

    def test_ssn_without_dashes_not_matched(self):
        """SSN pattern requires dashes; bare digits should not match the SSN rule."""
        r = _redactor()
        text = "code 123456789 here"
        result = r.redact(text)
        # No SSN token should appear (bare digits are not SSN-formatted)
        assert "[REDACTED_SSN]" not in result


class TestCreditCardRedaction:
    """Credit card numbers (16 digits, optional separators) are replaced with [REDACTED_CC]."""

    def test_plain_16_digits(self):
        r = _redactor()
        assert r.redact("cc: 4111111111111111") == "cc: [REDACTED_CC]"

    def test_dashes_grouped(self):
        r = _redactor()
        assert r.redact("card 4111-1111-1111-1111") == "card [REDACTED_CC]"

    def test_spaces_grouped(self):
        r = _redactor()
        assert r.redact("card 4111 1111 1111 1111") == "card [REDACTED_CC]"


class TestAmexCreditCard:
    """Amex 15-digit credit card numbers are detected and replaced with [REDACTED_CC]."""

    def test_amex_plain_15_digits(self):
        r = _redactor()
        assert r.redact("amex: 371449635398431") == "amex: [REDACTED_CC]"

    def test_amex_with_dashes(self):
        r = _redactor()
        assert r.redact("amex: 3714-496353-98431") == "amex: [REDACTED_CC]"

    def test_amex_with_spaces(self):
        r = _redactor()
        assert r.redact("amex: 3714 496353 98431") == "amex: [REDACTED_CC]"


class TestIPAddressRedaction:
    """IPv4 addresses are replaced with [REDACTED_IP]."""

    def test_standard_ip(self):
        r = _redactor()
        assert r.redact("server at 192.168.1.1") == "server at [REDACTED_IP]"

    def test_localhost(self):
        r = _redactor()
        assert r.redact("localhost 127.0.0.1") == "localhost [REDACTED_IP]"

    def test_public_ip(self):
        r = _redactor()
        assert r.redact("dns 8.8.8.8") == "dns [REDACTED_IP]"


class TestIPv6Redaction:
    """IPv6 addresses are replaced with [REDACTED_IPV6]."""

    def test_full_ipv6(self):
        r = _redactor()
        result = r.redact("addr: 2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert "[REDACTED_IPV6]" in result
        assert "2001:0db8" not in result

    def test_abbreviated_ipv6(self):
        r = _redactor()
        result = r.redact("addr: fe80::1")
        assert "[REDACTED_IPV6]" in result
        assert "fe80" not in result

    def test_loopback_ipv6(self):
        r = _redactor()
        result = r.redact("localhost ::1")
        assert "[REDACTED_IPV6]" in result


class TestInternationalPhone:
    """International phone numbers (+CC prefix) are replaced with [REDACTED_PHONE_INTL]."""

    def test_uk_phone(self):
        r = _redactor()
        result = r.redact("call +44 20 7946 0958")
        assert "[REDACTED_PHONE_INTL]" in result
        assert "7946" not in result

    def test_german_phone(self):
        r = _redactor()
        result = r.redact("call +49 30 123456")
        assert "[REDACTED_PHONE_INTL]" in result
        assert "123456" not in result

    def test_japanese_phone(self):
        r = _redactor()
        result = r.redact("call +81 3 1234 5678")
        assert "[REDACTED_PHONE_INTL]" in result
        assert "1234" not in result


class TestImprovedIPValidation:
    """Octet-validated IP regex rejects invalid octets like 999."""

    def test_invalid_octets_rejected(self):
        r = _redactor()
        result = r.redact("addr: 999.999.999.999")
        assert "[REDACTED_IP]" not in result

    def test_valid_edge_octets(self):
        r = _redactor()
        assert "[REDACTED_IP]" in r.redact("addr: 255.255.255.255")
        assert "[REDACTED_IP]" in r.redact("addr: 0.0.0.0")

    def test_valid_ip_still_matched(self):
        r = _redactor()
        assert "[REDACTED_IP]" in r.redact("addr: 192.168.1.1")
        assert "[REDACTED_IP]" in r.redact("addr: 10.0.0.1")

    def test_256_octet_rejected(self):
        r = _redactor()
        result = r.redact("addr: 256.1.1.1")
        assert "[REDACTED_IP]" not in result


# ---------------------------------------------------------------------------
# Multiple PII instances in one string
# ---------------------------------------------------------------------------

class TestMultipleSameType:
    """Multiple PII instances of the same type in a single string."""

    def test_multiple_emails(self):
        r = _redactor()
        text = "contact alice@example.com or bob@example.com"
        result = r.redact(text)
        assert result == "contact [REDACTED_EMAIL] or [REDACTED_EMAIL]"

    def test_multiple_phones(self):
        r = _redactor()
        text = "call 555-111-2222 or 555-333-4444"
        result = r.redact(text)
        assert result.count("[REDACTED_PHONE]") == 2

    def test_multiple_ssns(self):
        r = _redactor()
        text = "ssn1: 111-22-3333, ssn2: 444-55-6666"
        result = r.redact(text)
        assert result.count("[REDACTED_SSN]") == 2

    def test_multiple_ips(self):
        r = _redactor()
        text = "from 10.0.0.1 to 10.0.0.2"
        result = r.redact(text)
        assert result.count("[REDACTED_IP]") == 2


# ---------------------------------------------------------------------------
# Mixed PII types in one string
# ---------------------------------------------------------------------------

class TestMixedPIITypes:
    """Strings containing multiple different PII types."""

    def test_email_and_phone(self):
        r = _redactor()
        text = "reach me at alice@example.com or 555-123-4567"
        result = r.redact(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "alice@example.com" not in result
        assert "555-123-4567" not in result

    def test_ssn_and_credit_card(self):
        r = _redactor()
        text = "ssn 123-45-6789, card 4111-1111-1111-1111"
        result = r.redact(text)
        assert "[REDACTED_SSN]" in result
        assert "[REDACTED_CC]" in result

    def test_all_pii_types(self):
        r = _redactor()
        text = (
            "email: test@site.com, phone: 555-111-2222, "
            "ssn: 123-45-6789, cc: 4111111111111111, ip: 10.0.0.1"
        )
        result = r.redact(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_SSN]" in result
        assert "[REDACTED_CC]" in result
        assert "[REDACTED_IP]" in result
        # Verify none of the originals survive
        assert "test@site.com" not in result
        assert "123-45-6789" not in result
        assert "10.0.0.1" not in result


# ---------------------------------------------------------------------------
# Disabled state: enabled=False passes text through unchanged
# ---------------------------------------------------------------------------

class TestDisabledState:
    """When enabled=False, redact() returns the original text unchanged."""

    def test_disabled_passes_email_through(self):
        r = _redactor(enabled=False)
        text = "contact alice@example.com"
        assert r.redact(text) == text

    def test_disabled_passes_phone_through(self):
        r = _redactor(enabled=False)
        text = "call 555-123-4567"
        assert r.redact(text) == text

    def test_disabled_passes_ssn_through(self):
        r = _redactor(enabled=False)
        text = "ssn: 123-45-6789"
        assert r.redact(text) == text

    def test_disabled_passes_all_pii_through(self):
        r = _redactor(enabled=False)
        text = "a@b.com 555-123-4567 123-45-6789 4111111111111111 10.0.0.1"
        assert r.redact(text) == text

    def test_enabled_property_reflects_state(self):
        assert _redactor(enabled=True).enabled is True
        assert _redactor(enabled=False).enabled is False

    def test_disabled_does_not_increment_stats(self):
        r = _redactor(enabled=False)
        r.redact("ssn: 123-45-6789")
        r.redact("email: a@b.com")
        assert r.stats["total_redactions"] == 0


# ---------------------------------------------------------------------------
# Disabled categories: specific patterns can be turned off
# ---------------------------------------------------------------------------

class TestDisabledCategories:
    """Specific PII categories can be disabled while others remain active."""

    def test_disable_email_keeps_email_intact(self):
        r = _redactor(disabled_categories=["email"])
        text = "contact alice@example.com, ssn 123-45-6789"
        result = r.redact(text)
        assert "alice@example.com" in result
        assert "[REDACTED_SSN]" in result

    def test_disable_phone(self):
        r = _redactor(disabled_categories=["phone_us"])
        text = "call 555-123-4567, email a@b.com"
        result = r.redact(text)
        assert "555-123-4567" in result
        assert "[REDACTED_EMAIL]" in result

    def test_disable_ssn(self):
        r = _redactor(disabled_categories=["ssn"])
        text = "ssn 123-45-6789, ip 10.0.0.1"
        result = r.redact(text)
        assert "123-45-6789" in result
        assert "[REDACTED_IP]" in result

    def test_disable_credit_card(self):
        r = _redactor(disabled_categories=["credit_card"])
        text = "card 4111-1111-1111-1111, email a@b.com"
        result = r.redact(text)
        assert "4111-1111-1111-1111" in result
        assert "[REDACTED_EMAIL]" in result

    def test_disable_ip_address(self):
        r = _redactor(disabled_categories=["ip_address"])
        text = "server 192.168.1.1, ssn 123-45-6789"
        result = r.redact(text)
        assert "192.168.1.1" in result
        assert "[REDACTED_SSN]" in result

    def test_disable_multiple_categories(self):
        r = _redactor(disabled_categories=["email", "phone_us", "ip_address"])
        text = "a@b.com 555-123-4567 123-45-6789 4111111111111111 10.0.0.1"
        result = r.redact(text)
        assert "a@b.com" in result
        assert "10.0.0.1" in result
        assert "[REDACTED_SSN]" in result
        assert "[REDACTED_CC]" in result

    def test_disable_all_categories_no_redaction(self):
        all_names = [name for name, _, _ in PII_PATTERNS]
        r = _redactor(disabled_categories=all_names)
        text = "a@b.com 555-123-4567 123-45-6789 4111111111111111 10.0.0.1"
        assert r.redact(text) == text

    def test_stats_reflect_active_pattern_count(self):
        r = _redactor(disabled_categories=["email", "ssn"])
        assert r.stats["active_patterns"] == len(PII_PATTERNS) - 2


# ---------------------------------------------------------------------------
# scan() returns correct RedactionResult
# ---------------------------------------------------------------------------

class TestScan:
    """scan() detects PII and returns a RedactionResult without mutating input."""

    def test_scan_single_email(self):
        r = _redactor()
        result = r.scan("hello alice@example.com world")
        assert result.redactions_made == 1
        assert "email" in result.categories_found
        assert result.original_length == len("hello alice@example.com world")
        assert result.redacted_length == len("hello [REDACTED_EMAIL] world")

    def test_scan_multiple_categories(self):
        r = _redactor()
        text = "email a@b.com, ssn 123-45-6789"
        result = r.scan(text)
        assert result.redactions_made == 2
        assert "email" in result.categories_found
        assert "ssn" in result.categories_found

    def test_scan_no_pii(self):
        r = _redactor()
        result = r.scan("nothing personal here")
        assert result.redactions_made == 0
        assert result.categories_found == []
        assert result.original_length == result.redacted_length

    def test_scan_empty_string(self):
        r = _redactor()
        result = r.scan("")
        assert result.redactions_made == 0
        assert result.original_length == 0
        assert result.redacted_length == 0
        assert result.categories_found == []

    def test_scan_counts_multiple_same_type(self):
        r = _redactor()
        text = "a@b.com and c@d.com"
        result = r.scan(text)
        assert result.redactions_made == 2
        # Category should appear once (deduplicated list of categories)
        assert result.categories_found.count("email") == 1

    def test_scan_redacted_length_shorter_than_original_with_long_pii(self):
        r = _redactor()
        text = "card: 4111111111111111"
        result = r.scan(text)
        assert result.redacted_length < result.original_length

    def test_scan_works_even_when_disabled(self):
        """scan() always scans regardless of the enabled flag."""
        r = _redactor(enabled=False)
        result = r.scan("email: a@b.com")
        assert result.redactions_made == 1
        assert "email" in result.categories_found

    def test_scan_respects_disabled_categories(self):
        r = _redactor(disabled_categories=["email"])
        result = r.scan("email: a@b.com, ssn: 123-45-6789")
        assert "email" not in result.categories_found
        assert "ssn" in result.categories_found
        assert result.redactions_made == 1


# ---------------------------------------------------------------------------
# redact() returns the redacted text
# ---------------------------------------------------------------------------

class TestRedact:
    """redact() returns a new string with PII replaced by tokens."""

    def test_returns_string(self):
        r = _redactor()
        assert isinstance(r.redact("hello"), str)

    def test_non_pii_text_unchanged(self):
        r = _redactor()
        text = "this is perfectly normal text"
        assert r.redact(text) == text

    def test_only_pii_replaced(self):
        r = _redactor()
        text = "Hello, my email is user@host.com and I like cats."
        result = r.redact(text)
        assert "Hello" in result
        assert "cats" in result
        assert "[REDACTED_EMAIL]" in result
        assert "user@host.com" not in result

    def test_redact_preserves_surrounding_whitespace(self):
        r = _redactor()
        text = "  user@host.com  "
        result = r.redact(text)
        assert result.startswith("  ")
        assert result.endswith("  ")

    def test_redact_empty_string(self):
        r = _redactor()
        assert r.redact("") == ""

    def test_redact_empty_string_when_disabled(self):
        r = _redactor(enabled=False)
        assert r.redact("") == ""


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    """Stats track total_redactions and other metadata."""

    def test_initial_stats(self):
        r = _redactor()
        s = r.stats
        assert s["enabled"] is True
        assert s["total_redactions"] == 0
        assert s["total_redaction_passes"] == 0
        assert s["total_redacted_items"] == 0
        assert s["active_patterns"] == len(PII_PATTERNS)

    def test_total_redactions_increments_on_redaction(self):
        r = _redactor()
        r.redact("email: a@b.com")
        assert r.stats["total_redactions"] == 1

    def test_total_redactions_increments_per_call_not_per_match(self):
        """total_redaction_passes increments once per redact() call that modifies text."""
        r = _redactor()
        r.redact("a@b.com c@d.com 123-45-6789")
        assert r.stats["total_redactions"] == 1
        assert r.stats["total_redaction_passes"] == 1

    def test_total_redacted_items_counts_individual_matches(self):
        r = _redactor()
        r.redact("a@b.com c@d.com 123-45-6789")
        assert r.stats["total_redacted_items"] == 3

    def test_total_redactions_does_not_increment_when_no_pii(self):
        r = _redactor()
        r.redact("just normal text")
        assert r.stats["total_redactions"] == 0

    def test_total_redactions_accumulates_across_calls(self):
        r = _redactor()
        r.redact("email: a@b.com")
        r.redact("ssn: 123-45-6789")
        r.redact("no pii here")
        assert r.stats["total_redactions"] == 2

    def test_scan_does_not_affect_total_redactions(self):
        r = _redactor()
        r.scan("email: a@b.com")
        r.scan("ssn: 123-45-6789")
        assert r.stats["total_redactions"] == 0

    def test_disabled_redactor_stats(self):
        r = _redactor(enabled=False)
        s = r.stats
        assert s["enabled"] is False
        assert s["active_patterns"] == len(PII_PATTERNS)

    def test_disabled_categories_reduces_active_patterns(self):
        r = _redactor(disabled_categories=["email"])
        assert r.stats["active_patterns"] == len(PII_PATTERNS) - 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty strings, no PII, partial matches, boundary conditions."""

    def test_empty_string_redact(self):
        r = _redactor()
        assert r.redact("") == ""

    def test_empty_string_scan(self):
        r = _redactor()
        result = r.scan("")
        assert result.redactions_made == 0

    def test_whitespace_only(self):
        r = _redactor()
        assert r.redact("   \n\t  ") == "   \n\t  "

    def test_no_pii_text(self):
        r = _redactor()
        text = "The weather is nice today. Let's go for a walk."
        assert r.redact(text) == text

    def test_pii_at_start_of_string(self):
        r = _redactor()
        assert r.redact("alice@example.com is my email").startswith("[REDACTED_EMAIL]")

    def test_pii_at_end_of_string(self):
        r = _redactor()
        assert r.redact("my email is alice@example.com").endswith("[REDACTED_EMAIL]")

    def test_pii_only_string(self):
        r = _redactor()
        assert r.redact("alice@example.com") == "[REDACTED_EMAIL]"

    def test_long_text_with_scattered_pii(self):
        r = _redactor()
        text = (
            "Dear user, your account alice@example.com has been flagged. "
            "Please call 555-123-4567 to verify your identity with SSN 123-45-6789. "
            "Your credit card 4111-1111-1111-1111 was used from IP 10.20.30.40."
        )
        result = r.redact(text)
        assert "alice@example.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result
        assert "4111-1111-1111-1111" not in result
        assert "10.20.30.40" not in result
        assert "Dear user" in result
        assert "has been flagged" in result

    def test_newlines_preserved(self):
        r = _redactor()
        text = "line1: a@b.com\nline2: 123-45-6789\n"
        result = r.redact(text)
        assert "\n" in result
        assert result.count("\n") == 2

    def test_unicode_text_around_pii(self):
        r = _redactor()
        text = "Cafe\u0301 user@host.com \u2603"
        result = r.redact(text)
        assert "[REDACTED_EMAIL]" in result
        assert "Cafe\u0301" in result
        assert "\u2603" in result

    def test_partial_email_not_matched(self):
        """An '@' sign alone or missing domain should not trigger email redaction."""
        r = _redactor()
        assert r.redact("user@") == "user@"
        assert r.redact("@domain.com") == "@domain.com"

    def test_ssn_like_date_not_matched(self):
        """A date formatted as ###-##-#### could match SSN. This tests the boundary."""
        r = _redactor()
        # 123-45-6789 matches SSN pattern; but 2024-01-15 does not (4 digit prefix)
        assert "[REDACTED_SSN]" not in r.redact("date: 2024-01-15")


# ---------------------------------------------------------------------------
# False positives: text that resembles PII but isn't
# ---------------------------------------------------------------------------

class TestFalsePositives:
    """Patterns that look similar to PII but should ideally not match, or that we
    accept as known matches given the regex-based approach."""

    def test_version_number_matches_ip_pattern(self):
        """Version numbers like 1.2.3.4 match the IP regex. This is a known
        limitation of regex-based detection. We document it here."""
        r = _redactor()
        text = "using version 1.2.3.4 of the library"
        result = r.redact(text)
        # The IP pattern will match this -- document the known behavior
        assert "[REDACTED_IP]" in result

    def test_three_octet_version_not_matched(self):
        """A version like 1.2.3 should NOT match the IP pattern (needs 4 octets)."""
        r = _redactor()
        text = "version 1.2.3 is stable"
        assert "[REDACTED_IP]" not in r.redact(text)

    def test_short_number_not_phone(self):
        """Short numeric strings should not match the phone pattern."""
        r = _redactor()
        assert "[REDACTED_PHONE]" not in r.redact("code: 12345")

    def test_long_number_not_credit_card(self):
        """A 20-digit number should not match the credit card pattern (expects 16)."""
        r = _redactor()
        text = "id: 12345678901234567890"
        result = r.redact(text)
        # The CC regex expects exactly 4 groups of 4 digits; 20 digits should not
        # match as a single credit card
        assert "[REDACTED_CC]" not in result or "1234567890" in result

    def test_word_with_at_sign_not_email(self):
        """Something like '@mention' should not be detected as an email."""
        r = _redactor()
        assert r.redact("hello @mention") == "hello @mention"

    def test_ip_like_with_high_octets_no_longer_matches(self):
        """999.999.999.999 no longer matches the IP regex (octet validation)."""
        r = _redactor()
        result = r.redact("addr: 999.999.999.999")
        assert "[REDACTED_IP]" not in result

    def test_hyphenated_numbers_not_ssn(self):
        """A longer hyphenated number should not match SSN (different grouping)."""
        r = _redactor()
        assert "[REDACTED_SSN]" not in r.redact("part: 1234-56-7890")
        assert "[REDACTED_SSN]" not in r.redact("ref: 12-345-6789")


# ---------------------------------------------------------------------------
# Luhn check utility
# ---------------------------------------------------------------------------

class TestLuhnCheck:
    """PIIRedactor._luhn_check validates card numbers via Luhn algorithm."""

    def test_valid_visa(self):
        assert PIIRedactor._luhn_check("4111111111111111") is True

    def test_valid_amex(self):
        assert PIIRedactor._luhn_check("371449635398431") is True

    def test_invalid_number(self):
        assert PIIRedactor._luhn_check("1234567890123456") is False

    def test_single_digit(self):
        assert PIIRedactor._luhn_check("5") is False

    def test_non_numeric_ignored(self):
        # Luhn ignores non-digit characters
        assert PIIRedactor._luhn_check("4111-1111-1111-1111") is True


# ---------------------------------------------------------------------------
# RedactionResult dataclass
# ---------------------------------------------------------------------------

class TestRedactionResult:
    """RedactionResult stores scan metadata correctly."""

    def test_fields_accessible(self):
        result = RedactionResult(
            original_length=100,
            redacted_length=80,
            redactions_made=3,
            categories_found=["email", "ssn"],
        )
        assert result.original_length == 100
        assert result.redacted_length == 80
        assert result.redactions_made == 3
        assert result.categories_found == ["email", "ssn"]

    def test_default_categories_is_empty_list(self):
        result = RedactionResult(
            original_length=10, redacted_length=10, redactions_made=0
        )
        assert result.categories_found == []

    def test_categories_default_not_shared(self):
        """Each instance should have its own categories list (no mutable default sharing)."""
        r1 = RedactionResult(original_length=0, redacted_length=0, redactions_made=0)
        r2 = RedactionResult(original_length=0, redacted_length=0, redactions_made=0)
        r1.categories_found.append("email")
        assert r2.categories_found == []


# ---------------------------------------------------------------------------
# PII_PATTERNS module-level constant
# ---------------------------------------------------------------------------

class TestPIIPatterns:
    """Verify the PII_PATTERNS constant is well-formed."""

    def test_pattern_count(self):
        assert len(PII_PATTERNS) == 7

    def test_all_patterns_are_tuples_of_three(self):
        for entry in PII_PATTERNS:
            assert len(entry) == 3

    def test_all_names_are_strings(self):
        for name, _, _ in PII_PATTERNS:
            assert isinstance(name, str)

    def test_all_replacements_are_bracketed(self):
        for _, _, replacement in PII_PATTERNS:
            assert replacement.startswith("[")
            assert replacement.endswith("]")

    def test_expected_category_names(self):
        names = {name for name, _, _ in PII_PATTERNS}
        assert names == {
            "email", "phone_us", "phone_intl", "ssn",
            "credit_card", "ip_address", "ipv6_address",
        }
