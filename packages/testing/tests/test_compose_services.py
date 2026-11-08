"""The local runtime: every compose service works, and the fixture's rules hold.

The integration tests prove each store does its job: Postgres creates the ``vector`` extension,
SeaweedFS writes and reads an object for its user and refuses everyone else, and ClamAV
detects the EICAR test file.
"""

import re
import secrets
import uuid
from pathlib import Path
from typing import Any

import boto3
import psycopg
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from psycopg import sql

from stratafi_testing.clamd import FAKE_SIGNATURE, eicar, fake_clamd, instream, ping
from stratafi_testing.compose import LOCAL_ENV, TEST_ENV, ServiceEndpoints, ServiceManager


@pytest.mark.integration
def test_postgres_creates_the_vector_extension(compose_services) -> None:
    services = compose_services("postgres")
    database = f"smoke_{secrets.token_hex(4)}"
    with psycopg.connect(services.pg_admin_dsn(), autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        with psycopg.connect(services.pg_admin_dsn(database), autocommit=True) as conn:
            conn.execute("CREATE EXTENSION vector")
            row = conn.execute("SELECT '[1,2,3]'::vector <-> '[1,2,5]'::vector").fetchone()
            assert row is not None
            assert row[0] == pytest.approx(2.0)
            version = conn.execute("SHOW server_version_num").fetchone()
            assert version is not None
            assert version[0].startswith("16")
    finally:
        with psycopg.connect(services.pg_admin_dsn(), autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


def s3_client(
    services: ServiceEndpoints,
    *,
    secret_key: str | None = None,
    access_key: str | None = None,
    **config: Any,
):
    return boto3.client(
        "s3",
        endpoint_url=services.s3_endpoint(),
        aws_access_key_id=access_key or services.s3_access_key(),
        aws_secret_access_key=secret_key or services.s3_secret_key(),
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 1}, **config),
    )


@pytest.mark.integration
def test_seaweedfs_writes_and_reads_an_object(compose_services) -> None:
    services = compose_services("seaweedfs")
    client = s3_client(services)
    key = f"smoke/{uuid.uuid4()}"
    body = secrets.token_bytes(4096)
    client.put_object(Bucket=services.s3_bucket(), Key=key, Body=body)
    try:
        read = client.get_object(Bucket=services.s3_bucket(), Key=key)["Body"].read()
        assert read == body
    finally:
        client.delete_object(Bucket=services.s3_bucket(), Key=key)


@pytest.mark.integration
def test_seaweedfs_refuses_requests_without_its_users_keys(compose_services) -> None:
    # The identity file is in force: without one, SeaweedFS lets anyone read and write.
    services = compose_services("seaweedfs")
    anonymous = s3_client(services, signature_version=UNSIGNED)
    wrong_secret = s3_client(services, secret_key="not-the-secret")
    for client in (anonymous, wrong_secret):
        with pytest.raises(ClientError) as refused:
            client.put_object(Bucket=services.s3_bucket(), Key=f"smoke/{uuid.uuid4()}", Body=b"x")
        assert refused.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403
        with pytest.raises(ClientError) as refused:
            client.list_objects_v2(Bucket=services.s3_bucket())
        assert refused.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403


@pytest.mark.integration
def test_seaweedfs_never_creates_a_bucket_on_upload(compose_services) -> None:
    # Compose runs SeaweedFS with -s3.autoCreateBucket=false: a wrong bucket name fails, even
    # for the administrator.
    services = compose_services("seaweedfs")
    admin = s3_client(
        services,
        access_key=TEST_ENV["STRATAFI_S3_ADMIN_ACCESS_KEY"],
        secret_key=TEST_ENV["STRATAFI_S3_ADMIN_SECRET_KEY"],
    )
    with pytest.raises(ClientError) as refused:
        admin.put_object(Bucket=f"missing-{secrets.token_hex(4)}", Key="x", Body=b"x")
    assert refused.value.response["Error"]["Code"] == "NoSuchBucket"


@pytest.mark.integration
def test_the_services_user_reaches_only_its_bucket(compose_services) -> None:
    # A scoped user: it can't make, delete or use any other bucket.
    services = compose_services("seaweedfs")
    client = s3_client(services)
    admin = s3_client(
        services,
        access_key=TEST_ENV["STRATAFI_S3_ADMIN_ACCESS_KEY"],
        secret_key=TEST_ENV["STRATAFI_S3_ADMIN_SECRET_KEY"],
    )
    other = f"other-{secrets.token_hex(4)}"
    with pytest.raises(ClientError) as refused:
        client.create_bucket(Bucket=other)
    assert refused.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403
    admin.create_bucket(Bucket=other)
    try:
        admin.put_object(Bucket=other, Key="x", Body=b"x")
        for attempt in (
            lambda: client.get_object(Bucket=other, Key="x"),
            lambda: client.put_object(Bucket=other, Key="y", Body=b"y"),
            lambda: client.list_objects_v2(Bucket=other),
            lambda: client.delete_bucket(Bucket=services.s3_bucket()),
        ):
            with pytest.raises(ClientError) as refused:
                attempt()
            assert refused.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403
    finally:
        admin.delete_object(Bucket=other, Key="x")
        admin.delete_bucket(Bucket=other)


@pytest.mark.integration
def test_clamav_detects_the_eicar_test_file(compose_services) -> None:
    address = compose_services("clamav").clamd_address()
    assert ping(address)
    assert instream(address, b"an ordinary synthetic document\n") == "stream: OK"
    reply = instream(address, eicar())
    assert reply.endswith("FOUND")
    assert "Eicar" in reply


# --- the fixture's own rules, in the unit tier --------------------------------------------


def test_the_fake_clamd_flags_only_the_eicar_file() -> None:
    found = f"stream: {FAKE_SIGNATURE} FOUND"
    with fake_clamd() as address:
        assert ping(address)
        assert instream(address, eicar()) == found
        assert instream(address, eicar() + b"\r\n") == found
        assert instream(address, b"clean") == "stream: OK"
        assert instream(address, b"") == "stream: OK"
        # Like ClamAV's hash signature, EICAR inside other content isn't the EICAR file.
        assert instream(address, b"prefix " + eicar()) == "stream: OK"
        assert instream(address, secrets.token_bytes(100_000)) == "stream: OK"


def test_the_eicar_string_is_assembled_at_runtime(repo_root: Path) -> None:
    source = (repo_root / "packages/testing/src/stratafi_testing/clamd.py").read_bytes()
    assert eicar() not in source
    assert len(eicar()) == 68


def no_docker() -> tuple[bool, str]:
    return False, "no daemon in this test"


def test_without_docker_or_local_services_the_test_skips(tmp_path: Path) -> None:
    manager = ServiceManager(tmp_path, environ={}, docker_check=no_docker)
    with pytest.raises(pytest.skip.Exception, match="STRATAFI_TEST_PG_ADMIN_DSN"):
        manager.start("postgres")


def test_local_services_need_no_docker(tmp_path: Path) -> None:
    environ = {
        "STRATAFI_TEST_PG_ADMIN_DSN": "postgresql://admin:pw@127.0.0.1:5432/postgres",
        "STRATAFI_TEST_S3_ENDPOINT": "http://127.0.0.1:9000",
        "STRATAFI_TEST_S3_ACCESS_KEY": "key",
        "STRATAFI_TEST_S3_SECRET_KEY": "secret",
        "STRATAFI_TEST_S3_BUCKET": "bucket",
        "STRATAFI_TEST_CLAMD": "fake",
    }
    manager = ServiceManager(tmp_path, environ=environ, docker_check=no_docker)
    try:
        services = manager.start("postgres", "seaweedfs", "clamav")
        assert services.pg_admin_dsn("other") == "postgresql://admin:pw@127.0.0.1:5432/other"
        assert services.s3_bucket() == "bucket"
        assert ping(services.clamd_address())
    finally:
        manager.close()


def test_a_partly_configured_service_still_needs_docker(tmp_path: Path) -> None:
    environ = {"STRATAFI_TEST_S3_ENDPOINT": "http://127.0.0.1:9000"}
    manager = ServiceManager(tmp_path, environ=environ, docker_check=no_docker)
    with pytest.raises(pytest.skip.Exception, match="STRATAFI_TEST_S3_BUCKET"):
        manager.start("seaweedfs")


def test_unknown_services_are_an_error(tmp_path: Path) -> None:
    manager = ServiceManager(tmp_path, environ={}, docker_check=no_docker)
    with pytest.raises(ValueError, match="qdrant"):
        manager.start("qdrant")


def test_every_required_compose_variable_has_a_test_value(repo_root: Path) -> None:
    # Compose checks the whole file's required variables, the app profile's too, before it
    # starts any service; without a test value, a checkout with no .env can't start the tier.
    text = (repo_root / "compose.yaml").read_text(encoding="utf-8")
    required = set(re.findall(r"\$\{([A-Z0-9_]+):\?", text))
    assert required
    assert required <= set(TEST_ENV)


def test_every_service_has_local_variables() -> None:
    assert set(LOCAL_ENV) == {"postgres", "seaweedfs", "clamav", "tika"}


def test_compose_services_refuses_unit_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_unit(compose_services):
            pass
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=0, errors=1)
    result.stdout.fnmatch_lines(["*compose_services is for integration tests*"])
