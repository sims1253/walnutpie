# Warmup trace format

Use `--warmup-trace-dir PATH` to record one row per completed warmup iteration.
PATH must not exist; its parent must already exist. The tracer creates this directory exclusively.
It never deletes or overwrites a pre-existing trace directory. Do not modify its owned directory while running.
A failed run may leave incomplete files there. Use a new path for a retry.

The existing six-file layout is preserved:
- theta.f64, grad.f64, invmass.f64: little-endian IEEE754 binary64, iteration-major [rows,dim].
- step.f64, lp.f64: little-endian binary64 [rows].
- depth.u64: little-endian uint64 [rows].

Theta/gradient/log density are selected endpoints. Inverse mass is the value used for the transition;
step is the updated value after adaptation. Depth is the returned tree depth, not a divergence count.
Zero warmup produces six empty data files and a complete zero-row manifest.
Binary values retain NaN/infinity representations for diagnostics; they are not rejected by the trace writer.

meta.json keeps the original metadata keys and adds format_version1, byte_order, complete,
and explicit phase labels. JSON strings must be valid UTF8; quotes/control characters are escaped.
Nonfinite numeric metadata is null, declared by nonfinite_metadata. Finite values use classic locale
and17 significant digits. Old big-endian-host files were native-endian despite their description;
the new byte writer does not reinterpret or migrate old files.

All data files must close successfully before metadata is written. The temporary metadata file must also
close successfully before an atomic same-directory rename publishes meta.json. Only that manifest marks
success. This is not an fsync/power-loss durability guarantee or protection against concurrent directory tampering.
Shape and output failures throw rather than silently claim a usable trace.

Tracing is disabled by default. Disabled CLI tracing has one null-pointer dispatch with no trace vector copies,
scans, allocations, retained initial vectors, or extra density evaluations. Handlers without the optional
notification compile out the observer call. Enabled tracing buffers O(rows*dim) values until warmup finishes;
this is an audit tool, not a performance improvement. It consumes no sampler or generated-quantity RNG.

The optional on_warmup_trace notification runs synchronously after the ordinary on_warmup callback.
All vector references are borrowed only for that call. Existing handler requirements and callbacks are unchanged.

Composition note: an endpoint-gradient cache must not move the selected gradient before this borrowed
notification runs. When composing with such a cache, notify while the local still owns its values or bind
explicitly to the cache's live selected gradient. This patch is tested on c766697 without that cache.
