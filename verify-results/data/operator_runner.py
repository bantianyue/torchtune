"""Operator helper: run a verify script with TLS verification disabled.

The corporate proxy MITMs TLS with a self-signed chain, so `requests` (used by
huggingface_hub) rejects it while curl -k succeeds. This wrapper disables only
certificate verification for HTTPAdapter.cert_verify and then runs the target
script unchanged via runpy. Nothing else is touched.
"""
import runpy, sys
from requests.adapters import HTTPAdapter

_orig_cert_verify = HTTPAdapter.cert_verify


def _no_verify(self, conn, url, verify, cert):
    return _orig_cert_verify(self, conn, url, False, cert)


HTTPAdapter.cert_verify = _no_verify
target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
