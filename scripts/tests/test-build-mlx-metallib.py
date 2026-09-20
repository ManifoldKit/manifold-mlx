"""Exercise the real shell script with a deterministic Metal driver fixture."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(sys.argv.pop(1)).resolve()


class MetallibBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metallib-build-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gen = self.root / "sources with spaces"
        self.out = self.root / "output"
        self.bin = self.root / "bin"
        for directory in (self.gen, self.out, self.bin):
            directory.mkdir()
        (self.gen / "kernel.metal").write_text('#include "shared.h"\n')
        (self.gen / "shared.h").write_text("// original header\n")
        self.calls = self.root / "calls"
        self.env = dict(os.environ, PATH=str(self.bin) + ":/usr/bin:/bin",
                        FIXTURE_CALLS=str(self.calls), FIXTURE_MODE="ok",
                        FIXTURE_VERSION="1")
        driver = self.bin / "xcrun"
        driver.write_text('''#!/bin/bash
set -euo pipefail
if [[ "$*" == *--version* ]]; then
  if [ "$FIXTURE_MODE" = unavailable ]; then
    echo 'error: Metal Toolchain component is not installed' >&2
    exit 1
  fi
  echo "fixture compiler $FIXTURE_VERSION"
  exit 0
fi
if [[ "$*" == *--show-sdk-path* ]]; then echo /fixture/SDK; exit 0; fi
if [[ "$*" == *--show-sdk-version* ]]; then echo 27.0; exit 0; fi
echo "$*" >> "$FIXTURE_CALLS"
output=""
previous=""
for argument in "$@"; do
  if [ "$previous" = -o ]; then output="$argument"; fi
  previous="$argument"
done
echo partial > "$output"
if [[ "$*" == *' metal '* ]] && [ "$FIXTURE_MODE" = compile-failure ]; then
  echo 'shader address-space error' >&2
  exit 7
fi
if [[ "$*" == *' metallib '* ]] && [ "$FIXTURE_MODE" = link-failure ]; then
  echo 'metallib link error' >&2
  exit 8
fi
echo complete > "$output"
''')
        driver.chmod(0o755)

    def run_build(self):
        return subprocess.run(["/bin/bash", str(SCRIPT), str(self.gen), str(self.out)],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def assert_failure(self, result, diagnostic):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(diagnostic, result.stderr)
        self.assertFalse((self.out / "mlx.metallib").exists())

    def test_compile_error_is_fatal_and_removes_stale_output(self):
        (self.out / "mlx.metallib").write_text("stale")
        self.env["FIXTURE_MODE"] = "compile-failure"
        self.assert_failure(self.run_build(), "shader address-space error")

    def test_link_error_is_fatal_and_removes_partial_output(self):
        self.env["FIXTURE_MODE"] = "link-failure"
        self.assert_failure(self.run_build(), "metallib link error")

    def test_missing_sources_is_fatal(self):
        for source in self.gen.iterdir():
            source.unlink()
        self.gen.rmdir()
        self.assert_failure(self.run_build(), "sources not found")

    def test_empty_sources_is_fatal(self):
        (self.gen / "kernel.metal").unlink()
        self.assert_failure(self.run_build(), "no .metal kernels")

    def test_missing_compiler_is_reported_and_removes_stale_output(self):
        (self.out / "mlx.metallib").write_text("stale")
        self.env["FIXTURE_MODE"] = "unavailable"
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compile-only", result.stderr)
        self.assertFalse((self.out / "mlx.metallib").exists())

    def test_unchanged_build_reuses_output(self):
        self.assertEqual(self.run_build().returncode, 0)
        calls = self.calls.read_text()
        self.assertEqual(self.run_build().returncode, 0)
        self.assertEqual(self.calls.read_text(), calls)

    def test_header_change_with_preserved_mtime_rebuilds(self):
        self.assertEqual(self.run_build().returncode, 0)
        calls = self.calls.read_text()
        header = self.gen / "shared.h"
        previous = header.stat()
        header.write_text("// changed header\n")
        os.utime(header, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertEqual(self.run_build().returncode, 0)
        self.assertNotEqual(self.calls.read_text(), calls)

    def test_toolchain_change_rebuilds(self):
        self.assertEqual(self.run_build().returncode, 0)
        calls = self.calls.read_text()
        self.env["FIXTURE_VERSION"] = "2"
        self.assertEqual(self.run_build().returncode, 0)
        self.assertNotEqual(self.calls.read_text(), calls)


if __name__ == "__main__":
    unittest.main()
