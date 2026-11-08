"""Fetching a document from a URL behind an SSRF guard (``extract/url``).

A URL comes from a user, so the fetch must never reach what only the server can reach: its own
loopback, the private network, the cloud metadata service. ``UrlGuard`` decides, and the
fetcher holds it to every hop:

- **The URL:** ``http`` or ``https`` only, no user name or password in it, and a port from
  ``allowed_ports`` (80 and 443 by default).
- **The addresses:** the host name is resolved, and *every* address it resolves to must be
  public: not private (10/8, 172.16/12, 192.168/16, IPv6 ULA fc00::/7), loopback, link-local
  (169.254/16, fe80::/10, where the metadata services live: 169.254.169.254 and
  fd00:ec2::254), shared (100.64/10), multicast, reserved or unspecified. IPv4 addresses inside
  IPv6 (mapped, 6to4, NAT64, Teredo) are checked as the IPv4 address they carry. A literal IP in
  the URL is checked the same way.
- **The connection:** goes to the address that was checked, never to a second lookup of the
  name, so a DNS answer that changes between the check and the connection (rebinding) can't
  slip through. HTTPS still verifies the certificate against the host name (SNI).
- **Redirects:** are followed by hand, at most ``max_redirects``, and each target goes through
  every check again.
- **The response:** must be 200, with a content type Stratafi can process (``ALLOWED_TYPES``),
  and at most ``max_bytes``: a ``Content-Length`` over the cap is refused before the body, and
  the body is counted as it streams.

The fetch ignores proxy settings: a proxy would resolve the name itself, and the guard's checks
would no longer hold. Errors are ``FetchRefused`` with a short code; the URL never appears in a
log or an error.
"""

import hashlib
import ipaddress
import socket
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx2

from stratafi_components.detect import EXTENSIONS

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_URL_LENGTH = 2048
# Content types a fetched document may have: everything detection lists, and text.
ALLOWED_TYPES = frozenset(EXTENSIONS.values()) | {"application/octet-stream"}
_TYPE_EXTENSIONS: dict[str, str] = {}
for _extension, _media_type in EXTENSIONS.items():
    _TYPE_EXTENSIONS.setdefault(_media_type, _extension)
# The cloud metadata endpoints, named in errors so a refusal says why.
METADATA_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("169.254.170.2"),
        ipaddress.ip_address("fd00:ec2::254"),
    }
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

Address = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Sequence[str]]


class FetchRefused(Exception):
    """The fetch was refused or failed. ``code`` is short and never holds the URL."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise FetchRefused("host_not_found") from None
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    if address in _NAT64:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


# What ``UrlGuard.allow_private`` lets through: this machine and the private network.
LOCAL_REFUSALS = frozenset({"loopback_address", "private_address"})


def address_refusal(address: Address) -> str | None:
    """Why an address may not be fetched from, or None when it's public."""
    if address in METADATA_ADDRESSES:
        return "metadata_address"
    if isinstance(address, ipaddress.IPv6Address):
        inner = _embedded_ipv4(address)
        if inner is not None:
            return address_refusal(inner)
    if address.is_loopback:
        return "loopback_address"
    if address.is_link_local:
        return "link_local_address"
    if address.is_multicast:
        return "multicast_address"
    if address.is_unspecified or address.is_reserved:
        return "reserved_address"
    if address.is_private or not address.is_global:
        return "private_address"
    return None


@dataclass(frozen=True)
class Target:
    """A URL that passed the guard, with the address to connect to."""

    url: str
    scheme: str
    host: str
    port: int
    address: Address

    def connect_url(self) -> str:
        """The URL with the checked address in place of the host name."""
        parts = urlsplit(self.url)
        literal = (
            f"[{self.address}]"
            if isinstance(self.address, ipaddress.IPv6Address)
            else str(self.address)
        )
        return urlunsplit(
            (parts.scheme, f"{literal}:{self.port}", parts.path or "/", parts.query, "")
        )

    def host_header(self) -> str:
        default = 443 if self.scheme == "https" else 80
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == default else f"{host}:{self.port}"


@dataclass(frozen=True)
class UrlGuard:
    allowed_ports: tuple[int, ...] | None = (80, 443)  # None: any port
    resolver: Resolver = system_resolver
    # Webhooks in development and tests deliver to receivers on the same machine: this
    # allows loopback and private addresses, never the metadata service, link-local, multicast
    # or reserved ones. URL import never sets it.
    allow_private: bool = False

    def check(self, url: str) -> Target:
        """The URL's target, or ``FetchRefused`` saying why it may not be fetched."""
        if len(url) > MAX_URL_LENGTH:
            raise FetchRefused("url_too_long")
        try:
            parts = urlsplit(url.strip())
            port = parts.port
        except ValueError:
            raise FetchRefused("url_invalid") from None
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            raise FetchRefused("scheme_not_allowed")
        if parts.username is not None or parts.password is not None:
            raise FetchRefused("credentials_in_url")
        host = (parts.hostname or "").rstrip(".").lower()
        if not host:
            raise FetchRefused("url_invalid")
        port = port if port is not None else (443 if scheme == "https" else 80)
        if self.allowed_ports is not None and port not in self.allowed_ports:
            raise FetchRefused("port_not_allowed")
        try:
            literal: Address | None = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        resolved = [str(literal)] if literal is not None else self.resolver(host, port)
        addresses: list[Address] = []
        for value in resolved:
            try:
                address = ipaddress.ip_address(value.split("%", 1)[0])
            except ValueError:
                raise FetchRefused("host_not_found") from None
            # Every answer must be public: a name with one private answer is refused whole.
            refusal = address_refusal(address)
            if refusal and not (self.allow_private and refusal in LOCAL_REFUSALS):
                raise FetchRefused(refusal)
            addresses.append(address)
        if not addresses:
            raise FetchRefused("host_not_found")
        return Target(urlunsplit(parts._replace(fragment="")), scheme, host, port, addresses[0])


@dataclass
class Fetched:
    path: Path
    name: str
    sha256: str
    size: int
    content_type: str
    redirects: int = 0
    hops: list[str] = field(default_factory=list)  # each hop's host, for tests; never logged


def _content_type(value: str | None) -> str:
    message = Message()
    message["content-type"] = value or "application/octet-stream"
    return message.get_content_type()


def _file_name(url: str, disposition: str | None, content_type: str) -> str:
    name = ""
    if disposition:
        message = Message()
        message["content-disposition"] = disposition
        name = message.get_filename() or ""
    if not name:
        name = unquote(PurePosixPath(urlsplit(url).path).name)
    name = name.replace("/", "_").replace("\\", "_").strip(". ") or "download"
    if not PurePosixPath(name).suffix and content_type in _TYPE_EXTENSIONS:
        name += _TYPE_EXTENSIONS[content_type]
    return name[:200]


@dataclass
class UrlFetcher:
    guard: UrlGuard = field(default_factory=UrlGuard)
    max_bytes: int = DEFAULT_MAX_BYTES
    max_redirects: int = 5
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    transport: httpx2.BaseTransport | None = None  # tests inject one

    def _client(self) -> httpx2.Client:
        return httpx2.Client(
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            transport=self.transport,
        )

    def fetch(self, url: str, directory: Path) -> Fetched:
        """Fetch ``url`` into a new file in ``directory``. Raises ``FetchRefused``."""
        hops: list[str] = []
        with self._client() as client:
            for redirects in range(self.max_redirects + 1):
                target = self.guard.check(url)
                hops.append(target.host)
                request = client.build_request(
                    "GET",
                    target.connect_url(),
                    headers={"Host": target.host_header(), "Accept": "*/*"},
                    extensions={"sni_hostname": target.host} if target.scheme == "https" else {},
                )
                try:
                    response = client.send(request, stream=True)
                except httpx2.TimeoutException:
                    raise FetchRefused("fetch_timeout") from None
                except httpx2.HTTPError:
                    raise FetchRefused("fetch_failed") from None
                try:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise FetchRefused("redirect_invalid")
                        url = urljoin(target.url, location)
                        continue
                    if response.status_code != 200:
                        raise FetchRefused(f"http_{response.status_code}")
                    return self._save(response, target.url, directory, redirects, hops)
                finally:
                    response.close()
        raise FetchRefused("too_many_redirects")

    def _save(
        self,
        response: httpx2.Response,
        url: str,
        directory: Path,
        redirects: int,
        hops: list[str],
    ) -> Fetched:
        content_type = _content_type(response.headers.get("content-type"))
        if content_type not in ALLOWED_TYPES and not content_type.startswith("text/"):
            raise FetchRefused("content_type_not_allowed")
        length = response.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            raise FetchRefused("file_too_large")
        handle = tempfile.NamedTemporaryFile(dir=directory, prefix="fetch-", delete=False)  # noqa: SIM115
        path = Path(handle.name)
        digest, size = hashlib.sha256(), 0
        try:
            with handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise FetchRefused("file_too_large")
                    digest.update(chunk)
                    handle.write(chunk)
        except FetchRefused:
            path.unlink(missing_ok=True)
            raise
        except httpx2.HTTPError:
            path.unlink(missing_ok=True)
            raise FetchRefused("fetch_failed") from None
        name = _file_name(url, response.headers.get("content-disposition"), content_type)
        return Fetched(path, name, digest.hexdigest(), size, content_type, redirects, hops)
