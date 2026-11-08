"""The SSRF re-test: the URL guard against the bypasses people try, with the
system's own resolver for the numeric spellings it accepts. URL import (``extract/url``) and
webhooks both go through this guard."""

import pytest

from stratafi_components.fetch import FetchRefused, UrlGuard

REFUSED = [
    # Spellings of loopback that resolvers accept.
    ("http://2130706433/", "loopback_address"),  # decimal
    ("http://0177.0.0.1/", "loopback_address"),  # octal
    ("http://0x7f.0.0.1/", "loopback_address"),  # hexadecimal
    ("http://0x7f000001/", "loopback_address"),
    ("http://127.1/", "loopback_address"),  # short form
    ("http://127.000.000.001/", "loopback_address"),
    ("http://localhost./", "loopback_address"),  # trailing dot
    ("HTTP://LOCALHOST/", "loopback_address"),  # case
    ("http://[::1]/", "loopback_address"),
    ("http://[0:0:0:0:0:0:0:1]/", "loopback_address"),
    # Unspecified addresses connect to this machine.
    ("http://0.0.0.0/", "reserved_address"),
    ("http://0/", "reserved_address"),
    ("http://[::]/", "reserved_address"),
    # IPv4 inside IPv6.
    ("http://[::ffff:127.0.0.1]/", "loopback_address"),
    ("http://[::ffff:7f00:1]/", "loopback_address"),
    ("http://[::ffff:169.254.169.254]/", "metadata_address"),
    ("http://[64:ff9b::a9fe:a9fe]/", "metadata_address"),  # NAT64
    ("http://[2002:a9fe:a9fe::]/", "metadata_address"),  # 6to4
    ("http://[::127.0.0.1]/", "reserved_address"),  # IPv4-compatible, deprecated
    # The metadata services, every way they're written.
    ("http://169.254.169.254/latest/meta-data/", "metadata_address"),
    ("http://2852039166/", "metadata_address"),
    ("http://0xa9fea9fe/", "metadata_address"),
    ("http://169.254.170.2/v2/credentials", "metadata_address"),
    ("http://[fd00:ec2::254]/", "metadata_address"),
    ("http://100.100.100.200/latest/meta-data/", "private_address"),  # shared space (100.64/10)
    ("http://[fe80::1%25eth0]/", "link_local_address"),
    # Private networks.
    ("http://10.0.0.1/", "private_address"),
    ("http://172.16.0.1/", "private_address"),
    ("http://192.168.1.1/", "private_address"),
    ("http://[fc00::1]/", "private_address"),
    # Parser differentials and credentials.
    ("http://127.0.0.1:80@example.com/", "credentials_in_url"),
    ("http://example.com\\@127.0.0.1/", "credentials_in_url"),
    ("http://user@169.254.169.254/", "credentials_in_url"),
    # Schemes and ports.
    ("file:///etc/passwd", "scheme_not_allowed"),
    ("gopher://127.0.0.1:6379/_INFO", "scheme_not_allowed"),
    ("dict://127.0.0.1:11211/stat", "scheme_not_allowed"),
    ("ftp://example.com/x.pdf", "scheme_not_allowed"),
    ("http://93.184.215.14:6379/", "port_not_allowed"),
    ("http://93.184.215.14:0/", "port_not_allowed"),
    ("http://[::1]:99999/", "url_invalid"),
]


@pytest.mark.parametrize(("url", "code"), REFUSED)
def test_the_guard_refuses_each_bypass(url: str, code: str) -> None:
    with pytest.raises(FetchRefused) as refused:
        UrlGuard().check(url)
    assert refused.value.code == code


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/",
        "http://[::ffff:169.254.169.254]/",
        "http://[fe80::1%25eth0]/",
        "http://224.0.0.1/",
        "http://0.0.0.0/",
    ],
)
def test_allowing_private_addresses_never_allows_the_rest(url: str) -> None:
    # Webhooks in development allow loopback and private receivers; never these.
    with pytest.raises(FetchRefused):
        UrlGuard(allowed_ports=None, allow_private=True).check(url)


def test_a_public_literal_passes() -> None:
    target = UrlGuard().check("https://93.184.215.14/doc.pdf")
    assert str(target.address) == "93.184.215.14"
    assert target.port == 443
