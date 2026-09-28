"""
Make TLS verification use the operating system's trust store.

Why this exists
---------------

The model weights for both retrieval stages are fetched from
huggingface.co on first use. On this machine that connection is
TLS-intercepted by a corporate proxy, so the certificate chain
terminates in a private root. That root is installed in the Windows
certificate store, which is why `urllib` reaches the Hub without
complaint, but `huggingface_hub` verifies through `httpx`, and `httpx`
pins `certifi`'s bundle instead. The private root is not in `certifi`,
so every download failed with

    [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate

which `huggingface_hub` then reports as `LocalEntryNotFoundError:
check your connection`, i.e. as a network outage rather than a trust
problem. That misattribution cost real time, so it is recorded here.

The fix is to verify against the roots the machine already trusts.
`truststore` swaps `ssl.SSLContext` for one backed by the platform
verifier, so `httpx`'s `create_default_context(cafile=certifi...)`
resolves against the OS store and the `cafile` becomes moot. This is
the same mechanism pip uses for `--use-feature=truststore`.

What this deliberately does not do: disable verification. `verify=False`
would also have made the download work and would have made every
subsequent HTTPS call in the process unauthenticated, including the
OpenAI calls the verifier depends on.

The alternative considered was pointing `SSL_CERT_FILE` at the bundle
pip is configured with (`C:\\ProgramData\\Python\\ca-bundle.cer`). It
fails twice over: `httpx` passes `cafile` explicitly so the environment
variable is ignored, and that bundle contains a CA whose Basic
Constraints extension is not marked critical, which Python 3.13+
rejects because it enables `VERIFY_X509_STRICT` by default.

Import side effect, stated plainly: calling `enable_os_trust_store()`
mutates the `ssl` module process-wide. It is called from the two
retrieval modules that load models, because those are the only places
that need it and doing it there keeps it out of unrelated entry points.
"""

_ENABLED = False


def enable_os_trust_store() -> bool:
    """
    Route TLS verification through the platform trust store.

    Idempotent, and a no-op that returns False when `truststore` is not
    installed. A missing `truststore` is not an error: on a machine
    without TLS interception, `certifi` is sufficient and nothing here
    is needed.
    """

    global _ENABLED

    if _ENABLED:
        return True

    try:
        import truststore
    except ImportError:
        return False

    truststore.inject_into_ssl()

    _ENABLED = True

    return True
