"""Run one custom check inside the sandbox: the container's entry point.

    python -I -B /opt/sandbox/run_check.py /check/<path.py> <function>

It loads the check's file, calls the function with a ``CheckContext`` over ``/bundle`` (mounted
read-only) and ``/scratch`` (the only writable place), and prints its ``CheckResult`` as one
marked JSON line. A check that raises, or returns something else, prints an error code and the
exception's type: never its message, which could hold a document's text. The suite's other
files sit beside the check under ``/check``, so a check may import its helpers.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/sandbox")

from stratafi.checks import RESULT_MARKER, CheckContext, CheckResult


def _emit(payload: dict) -> None:
    sys.stdout.write("\n" + RESULT_MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(
    argv: list[str],
    *,
    bundle: str = "/bundle",
    scratch: str = "/scratch",
    code_root: str = "/check",
) -> int:
    if len(argv) != 2:
        _emit({"error": "usage"})
        return 2
    path, function = Path(argv[0]), argv[1]
    try:
        sys.path.insert(1, code_root)
        spec = importlib.util.spec_from_file_location("custom_check", path)
        if spec is None or spec.loader is None or not path.is_file():
            _emit({"error": "code_unreadable"})
            return 1
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check = getattr(module, function, None)
        if not callable(check):
            _emit({"error": "function_missing"})
            return 1
        result = check(CheckContext(bundle, scratch=scratch))
        if not isinstance(result, CheckResult):
            _emit({"error": "not_a_check_result"})
            return 1
        payload = {"result": result.to_dict()}
    except BaseException as error:  # the check's failure is reported, not raised
        _emit({"error": "check_raised", "exception": type(error).__name__[:64]})
        return 1
    _emit(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
