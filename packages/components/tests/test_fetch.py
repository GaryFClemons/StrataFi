"""The URL fetcher's SSRF guard: private, link-local and metadata addresses are
refused, checked again after DNS resolution and on every redirect, with a size cap."""

import ipaddress

import httpx2
import pytest

from stratafi_components.fetch import (
    FetchRefused,
    UrlFetcher,
    UrlGuard,
    address_refusal,
)

PUBLIC = "93.184.215.14"
PUBLIC_V6 = "2606:2800:21f:cb07:6820:80da:af6b:8b2c"


def resolver(table: dict[str, list[str]]):
    def resolve(host: str, port: int) -> list[str]:
        if host not in table:
            raise FetchRefused("host_not_found")
        return table[host]

    return resolve


@pytest.mark.parametrize(
    ("address", "code"),
    [
        ("169.254.169.254", "metadata_address"),
        ("169.254.170.2", "metadata_address"),
        ("fd00:ec2::254", "metadata_address"),
        ("169.254.1.1", "link_local_address"),
        ("fe80::1", "link_local_address"),
        ("10.0.0.1", "private_address"),
        ("10.255.255.255", "private_address"),
        ("172.16.5.4", "private_address"),
        ("192.168.1.1", "private_address"),
        ("100.64.0.1", "private_address"),
        ("fd12:3456:789a::1", "private_address"),
        ("fc00::1", "private_address"),
        ("127.0.0.1", "loopback_address"),
        ("::1", "loopback_address"),
        ("0.0.0.0", "reserved_address"),  # noqa: S104 - an address under test, not a bind
        ("::", "reserved_address"),
        ("224.0.0.1", "multicast_address"),
        ("ff02::1", "multicast_address"),
        ("240.0.0.1", "reserved_address"),
        # IPv4 carried inside IPv6 is judged as the IPv4 address.
        ("::ffff:10.0.0.1", "private_address"),
        ("::ffff:169.254.169.254", "metadata_address"),
        ("2002:a00:1::1", "private_address"),  # 6to4 of 10.0.0.1
        ("64:ff9b::a9fe:a9fe", "metadata_address"),  # NAT64 of 169.254.169.254
    ],
)
def test_non_public_addresses_are_refused(address: str, code: str) -> None:
    assert address_refusal(ipaddress.ip_address(address)) == code


@pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", PUBLIC_V6, "::ffff:93.184.215.14"])
def test_public_addresses_pass(address: str) -> None:
    assert address_refusal(ipaddress.ip_address(address)) is None


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://169.254.169.254/latest/meta-data/", "metadata_address"),
        ("http://10.0.0.8/report.pdf", "private_address"),
        ("http://10.1.2.3:80/", "private_address"),
        ("https://[fd00::1]/a.pdf", "private_address"),
        ("http://[::1]/", "loopback_address"),
        ("http://127.0.0.1/", "loopback_address"),
        ("ftp://example.com/a.pdf", "scheme_not_allowed"),
        ("file:///etc/passwd", "scheme_not_allowed"),
        ("gopher://example.com/", "scheme_not_allowed"),
        ("http://user:secret@example.com/a.pdf", "credentials_in_url"),
        ("http://example.com:22/", "port_not_allowed"),
        ("http://example.com:6379/", "port_not_allowed"),
        ("http:///nohost", "url_invalid"),
        ("http://example.com:99999/", "url_invalid"),
    ],
)
def test_the_guard_refuses_urls(url: str, code: str) -> None:
    guard = UrlGuard(resolver=resolver({"example.com": [PUBLIC]}))
    with pytest.raises(FetchRefused) as refused:
        guard.check(url)
    assert refused.value.code == code


def test_allowing_private_addresses_still_refuses_the_metadata_service() -> None:
    # Webhooks in development and tests deliver to this machine; nothing reaches the
    # metadata service, link-local or multicast addresses that way.
    guard = UrlGuard(allowed_ports=None, allow_private=True)
    assert str(guard.check("http://127.0.0.1:8123/hook").address) == "127.0.0.1"
    assert str(guard.check("http://10.0.0.8/hook").address) == "10.0.0.8"
    for url, code in [
        ("http://169.254.169.254/latest/meta-data/", "metadata_address"),
        ("http://169.254.1.1/", "link_local_address"),
        ("http://224.0.0.1/", "multicast_address"),
    ]:
        with pytest.raises(FetchRefused) as refused:
            guard.check(url)
        assert refused.value.code == code


def test_names_are_checked_after_dns_resolution() -> None:
    guard = UrlGuard(
        resolver=resolver(
            {
                "metadata.example": ["169.254.169.254"],
                "intranet.example": ["10.0.0.7"],
                "ula.example": ["fd00:1234::5"],
                # One private answer among public ones refuses the name.
                "mixed.example": [PUBLIC, "10.0.0.7"],
                "public.example": [PUBLIC, PUBLIC_V6],
            }
        )
    )
    for host, code in [
        ("metadata.example", "metadata_address"),
        ("intranet.example", "private_address"),
        ("ula.example", "private_address"),
        ("mixed.example", "private_address"),
    ]:
        with pytest.raises(FetchRefused) as refused:
            guard.check(f"https://{host}/policy.pdf")
        assert refused.value.code == code, host
    target = guard.check("https://public.example/policy.pdf#section")
    assert str(target.address) == PUBLIC
    assert target.connect_url() == f"https://{PUBLIC}:443/policy.pdf"
    assert target.host_header() == "public.example"
    assert target.url == "https://public.example/policy.pdf"


def _fetcher(handler, table, **kwargs) -> UrlFetcher:
    return UrlFetcher(
        guard=UrlGuard(resolver=resolver(table)),
        transport=httpx2.MockTransport(handler),
        **kwargs,
    )


TABLE = {
    "docs.example": [PUBLIC],
    "cdn.example": ["93.184.215.20"],
    "internal.example": ["10.0.0.9"],
    "metadata.example": ["169.254.169.254"],
}


def test_a_document_is_fetched_from_the_checked_address(tmp_path) -> None:
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((str(request.url), request.headers["host"], request.extensions))
        return httpx2.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7 synthetic"
        )

    fetched = _fetcher(handler, TABLE).fetch("https://docs.example/files/Fee%20Schedule", tmp_path)
    assert fetched.name == "Fee Schedule.pdf"
    assert fetched.size == len(b"%PDF-1.7 synthetic")
    assert fetched.path.read_bytes() == b"%PDF-1.7 synthetic"
    url, host, extensions = seen[0]
    assert url == f"https://{PUBLIC}/files/Fee%20Schedule"
    assert host == "docs.example"
    assert extensions["sni_hostname"] == "docs.example"


@pytest.mark.parametrize(
    "location",
    [
        "http://10.0.0.5/secret",
        "http://169.254.169.254/latest/meta-data/iam/",
        "http://[fd00:ec2::254]/latest/",
        "http://internal.example/report.pdf",
        "http://metadata.example/",
        "http://127.0.0.1:80/admin",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_redirects_to_private_addresses_are_refused(tmp_path, location: str) -> None:
    requested = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requested.append(request.headers["host"])
        if request.headers["host"] == "docs.example":
            return httpx2.Response(302, headers={"location": location})
        raise AssertionError("the private target must never be requested")

    with pytest.raises(FetchRefused) as refused:
        _fetcher(handler, TABLE).fetch("https://docs.example/start", tmp_path)
    assert refused.value.code.endswith("_address")
    assert requested == ["docs.example"]
    assert list(tmp_path.iterdir()) == []


def test_a_redirect_chain_to_a_public_host_is_followed_and_rechecked(tmp_path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.headers["host"] == "docs.example":
            return httpx2.Response(301, headers={"location": "https://cdn.example/a/notes.txt"})
        return httpx2.Response(200, headers={"content-type": "text/plain"}, content=b"notes")

    fetched = _fetcher(handler, TABLE).fetch("http://docs.example/notes", tmp_path)
    assert fetched.redirects == 1
    assert fetched.hops == ["docs.example", "cdn.example"]
    assert fetched.name == "notes.txt"


def test_redirect_loops_stop(tmp_path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(302, headers={"location": "/again"})

    with pytest.raises(FetchRefused) as refused:
        _fetcher(handler, TABLE, max_redirects=3).fetch("https://docs.example/", tmp_path)
    assert refused.value.code == "too_many_redirects"


def test_the_size_cap_holds_with_and_without_a_content_length(tmp_path) -> None:
    def declared(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, headers={"content-type": "text/plain", "content-length": "5000"}, content=b"x"
        )

    with pytest.raises(FetchRefused) as refused:
        _fetcher(declared, TABLE, max_bytes=100).fetch("https://docs.example/a.txt", tmp_path)
    assert refused.value.code == "file_too_large"

    def streamed(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=httpx2.ByteStream(b"y" * 500),
        )

    with pytest.raises(FetchRefused) as refused:
        _fetcher(streamed, TABLE, max_bytes=100).fetch("https://docs.example/b.txt", tmp_path)
    assert refused.value.code == "file_too_large"
    assert list(tmp_path.iterdir()) == []


def test_content_types_stratafi_cannot_process_are_refused(tmp_path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "application/x-msdownload"})

    with pytest.raises(FetchRefused) as refused:
        _fetcher(handler, TABLE).fetch("https://docs.example/setup.exe", tmp_path)
    assert refused.value.code == "content_type_not_allowed"


def test_errors_never_carry_the_url(tmp_path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404)

    with pytest.raises(FetchRefused) as refused:
        _fetcher(handler, TABLE).fetch("https://docs.example/secret-path-name", tmp_path)
    assert refused.value.code == "http_404"
    assert "secret-path-name" not in str(refused.value)
