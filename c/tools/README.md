<!-- Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. -->
# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `make_glm_oracle.py`, `make_glm_bench_model.py`: deterministic fixtures
- `benchmark_cuda_fixture.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `paired_quality_gate.py`: paired B0/B1 noninferiority gate over two
  `eval_glm.py --dump-per-item` JSONL files (exact McNemar, paired
  bootstrap CIs, sample-size planning)
- `gen_unicode.py`: tokenizer table generation
- `provenance_manifest.py`: computation backend for `scripts/provenance.sh`
  (the executable provenance manifest -- see that script's header)
- `provenance_compare.py`: asserts two provenance manifests agree on every
  field except a declared `--vary` set; hard-fails on a binary/generated-source
  mismatch between A/B arms

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/iliria-bench
```
