# Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE.
"""
Convertitore GLM-5.2-FP8 -> nostro container int4 (STADIO B).

Strategia DISK-SAFE (richiesta dell'utente): scarica UNO shard (~5 GB), lo converte in
int4, lo CANCELLA, passa al prossimo. Il disco non si riempie mai: picco = 1 shard + l'output
int4 che cresce fino a ~372 GB. Controllo di spazio che si ferma se manca margine.

Cosa fa per ogni tensore:
  - pesi FP8 (e4m3) con `*.weight_scale_inv`  -> dequant a blocchi 128x128 -> f32
  - pesi BF16 (norme/embed/lm_head/...)        -> f32
  poi:
  - attn/mlp/shared/expert/embed/lm_head -> QUANTIZZATO int4 (o int8) con la STESSA matematica
    del motore C (np.rint = lrintf, stesse soglie, stesso packing dei nibble) -> token identici
  - norme / router (mlp.gate.weight) / bias / e_score_correction_bias -> tenuti F32
  - indexer DSA / layer MTP (78) / shared_head / eh_proj / *norm dell'indexer -> SALTATI

Output: una dir di safetensors leggibile dal motore C (per ogni peso quantizzato: `nome` U8 =
dati impacchettati, `nome.qs` F32 = scale per riga).

USO:
  # test locale (oracolo tiny, niente download): converte una dir gia' presente
  python3 tools/convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4
  # selftest del dequant fp8 (richiede torch)
  python3 tools/convert_fp8_to_int4.py --selftest
  # reale: scarica+converte+cancella shard per shard
  python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 --outdir ~/models/glm52_i4
"""
import os, sys, glob, json, shutil, struct, argparse
import numpy as np

# ---------- quantizzazione: identica al C (glm.c) ----------
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibble [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int2(w, bits):                        # -> (qbytes U8 [O*ceil(I/4)], scale f32 [O]); 4/byte
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                 # bits=2 -> qmax=1, valori [-2,1]
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -2, qmax).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # impacchetta 4 valori per byte (identico a pack_int2 in C)
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)

# DEFECT (root-caused 2026-07-15, see tests/test_quant_noise_floor.py): quant_int2's
# scale=absmax/qmax=absmax/1 makes code -2 geometrically unreachable in practice --
# it is a ternary quantizer wearing 2-bit clothing (measured 0.868 relative Frobenius
# vs the 4-level Lloyd-Max optimum of 0.343). quant_int2_fitted below is the
# production port of tests/test_quant_noise_floor.py's
# _reference_repaired_quant_int2 (stage 1 of the three-stage repair plan in
# docs/performance-theory.json, expert-quant-error-saliency): SAME on-disk format
# (4 codes/byte, packed value v=code+2, engine dequant (code-2)*scale) and SAME
# codebook {-2,-1,0,1} -- zero engine/kernel change -- but a per-row scale chosen
# by a small grid search that minimizes reconstruction MSE over the FULL codebook
# instead of the shipped absmax/1. Both a positive and a negative candidate scale
# are tried per row (the codebook is asymmetric -- two negative codes, one
# positive -- so a row's own empirical tail picks which side gets the extra code).
# Measured 0.380 relative Frobenius on synthetic Gaussian weights (vs 0.868 for
# quant_int2), see docs/performance-theory.json. quant_int2 (the defective
# quantizer) is kept UNCHANGED and available side by side for comparison -- see
# --int2-quantizer below and tools/eval_activation_error.py for the activation-
# level verdict between the two.
def quant_int2_fitted(w, bits, n_candidates=64):  # -> same shape/packing as quant_int2
    O, I = w.shape
    amax = np.abs(w).max(axis=1)
    best_s = np.zeros(O, np.float32)
    best_mse = np.full(O, np.inf, np.float64)
    for sign in (1.0, -1.0):
        for k in range(1, n_candidates + 1):
            s = sign * (amax * k / n_candidates)
            s = np.where(s == 0, 1e-8, s).astype(np.float32)
            q = np.clip(np.rint(w / s[:, None]), -2, 1)
            recon = q * s[:, None]
            mse = np.mean((w - recon) ** 2, axis=1, dtype=np.float64)
            better = mse < best_mse
            best_mse = np.where(better, mse, best_mse)
            best_s = np.where(better, s, best_s)
    q = np.clip(np.rint(w / best_s[:, None]), -2, 1).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # identical packing to quant_int2
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), best_s.astype(np.float32)


# --int2-quantizer / ILI_INT2_QUANTIZER: selects which int2 path classify()/
# convert_shard() below use for bits<=2 tensors. "defective" (default) preserves
# every existing container byte-for-byte; "fitted" opts into quant_int2_fitted.
INT2_QUANTIZERS = {"defective": quant_int2, "fitted": quant_int2_fitted}

# ---------- classificazione dei tensori ----------
def layer_idx(name):
    p = name.split(".")
    if len(p) > 2 and p[0] == "model" and p[1] == "layers":
        try: return int(p[2])
        except ValueError: return -1
    return -1

def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    if name.endswith("_scale_inv"): return "consumed"   # gestito col suo peso
    li = layer_idx(name)
    if keep_idx:
        # modalita' --indexer: SOLO i pesi del DSA lightning indexer dei layer principali
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"                                       # int8 consigliato (--ebits 8): pesi di scoring
    if keep_mtp:
        if li != n_layers: return "skip"                 # solo il layer MTP
        if "indexer" in name: return "skip"              # il DSA indexer resta un no-op
    else:
        if li >= n_layers: return "skip"                 # layer MTP (78)
        if any(k in name for k in ["indexer", "indexers_proj", "eh_proj",
                                    "enorm", "hnorm", "shared_head"]): return "skip"
    if name.endswith("e_score_correction_bias"): return "f32"
    if name.endswith("mlp.gate.weight"): return "f32"    # router (NON gate_proj)
    if name.endswith("norm.weight") or name == "model.norm.weight": return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"
    if ".mlp.experts." in name and name.endswith(".weight"): return "x"  # expert ROUTED (streaming)
    if name.endswith(".weight"): return "q"              # attn/dense-mlp/shared (residente)
    return "f32"

# ---------- dequant di un tensore (fp8+scale a blocchi / bf16 / f32) ----------
def dequant(f, name):
    import torch
    sl = f.get_slice(name); dt = sl.get_dtype()
    if dt in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)   # [ceil(O/128),ceil(I/128)]
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()

def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits, keep_mtp=False, keep_idx=False,
                   int2_variant="defective"):
    from safetensors import safe_open
    quant2 = INT2_QUANTIZERS[int2_variant]
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            kind = classify(name, n_layers, keep_mtp, keep_idx)
            if kind in ("skip", "consumed"): continue
            w = dequant(f, name)
            if kind == "f32":
                out_dict[name] = w.astype(np.float32)
            else:
                bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                if w.ndim != 2:        # es. bias 1D non previsto come 'q' -> tienilo f32
                    out_dict[name] = w.astype(np.float32); continue
                q, s = (quant2(w, bits) if bits <= 2 else
                        quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
                out_dict[name] = q
                out_dict[name + ".qs"] = s

def free_gb(p): return shutil.disk_usage(p).free / 1e9

def verify_shard_file(path):
    """Structural resume-safety check for an output shard: True only if `path`
    exists, is a well-formed safetensors file, and its header's declared
    tensor byte ranges exactly account for the whole file. `os.path.exists()`
    alone cannot distinguish a complete shard from one truncated by a killed
    process mid-write (safetensors.numpy.save_file() is NOT atomic -- it
    writes directly to the target path with no temp-file+rename, see
    save_file()'s own tmp+os.replace() wrapper below, which is what actually
    makes THIS tool's own resume safe going forward). Cheap: one header-only
    parse + one stat, no torch/safetensors import needed."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) != 8:
                return False
            (hlen,) = struct.unpack("<Q", head)
            header_bytes = f.read(hlen)
            if len(header_bytes) != hlen:
                return False
            header = json.loads(header_bytes)
    except Exception:
        return False
    header.pop("__metadata__", None)
    if not header:
        return False
    try:
        max_end = max(int(meta["data_offsets"][1]) for meta in header.values())
    except (KeyError, ValueError, TypeError, IndexError):
        return False
    expected_size = 8 + hlen + max_end
    try:
        actual_size = os.path.getsize(path)
    except OSError:
        return False
    return actual_size == expected_size


def save_file_atomic(tensors, outp):
    """Atomic replacement for safetensors.numpy.save_file(tensors, outp):
    writes to a sibling `.tmp` file, structurally self-verifies it (catches a
    kill mid-write), then os.replace()s it into place -- so `outp` only ever
    exists as a complete file, matching build_mixed_container.py's /
    encode_mode15_container.py's tmp+rename+self-verify convention. Raises
    SystemExit (not a silent skip) if the just-written file fails
    verification: a shard that cannot even round-trip its own header is a
    bug worth stopping the run for, not quietly retrying forever."""
    from safetensors.numpy import save_file
    tmp_out = outp + ".tmp"
    save_file(tensors, tmp_out)
    if not verify_shard_file(tmp_out):
        try: os.remove(tmp_out)
        except OSError: pass
        raise SystemExit(f"{outp}: just-written shard failed structural self-verification "
                          "-- aborting (this should never happen; investigate before rerunning)")
    os.replace(tmp_out, outp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)   # bit residenti (default 4; 8 per --mtp/--indexer)
    ap.add_argument("--io-bits", type=int, default=8)    # bit di embed/lm_head
    ap.add_argument("--xbits", type=int, default=None)   # bit degli expert ROUTED (streaming); default=ebits
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mtp", action="store_true",
        help="download and convert ONLY the MTP head (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="extract ONLY the DSA lightning-indexer weights -> out-idx-*.safetensors. WARNING: "
             "indexer tensors are spread across nearly every shard, so this re-downloads the whole "
             "repository (~756 GB of traffic) to retain only a few GB. Resumable per shard. "
             "Recommended: --ebits 8.")
    ap.add_argument("--int2-quantizer", choices=sorted(INT2_QUANTIZERS),
        default=os.environ.get("ILI_INT2_QUANTIZER", "defective"),
        help="which int2 path to use for bits<=2 tensors (default 'defective', the "
             "shipped quant_int2; 'fitted' opts into the repaired quant_int2_fitted -- "
             "see docs/performance-theory.json expert-quant-error-saliency). Falls back "
             "to ILI_INT2_QUANTIZER when unset.")
    a = ap.parse_args()
    if a.int2_quantizer not in INT2_QUANTIZERS:
        ap.error(f"--int2-quantizer must be one of {sorted(INT2_QUANTIZERS)}")
    if a.ebits is None:
        # testa MTP a int4 = acceptance ~0-4% (misurato, issue #8): il draft sbaglia sempre
        # e la speculazione non parte mai. A int8: 39-59%, 2.2-2.8 token/forward.
        a.ebits = 8 if (a.mtp or a.indexer) else 4
    if a.xbits is None: a.xbits = a.ebits

    if a.selftest:
        import torch
        w = (torch.randn(256, 256) * 0.3)
        O, I = w.shape; bs = 128
        sc = torch.zeros(O // bs, I // bs)
        for bi in range(O // bs):
            for bj in range(I // bs):
                blk = w[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                sc[bi, bj] = blk.abs().max() / 448.0
        q = (w / sc.repeat_interleave(bs,0).repeat_interleave(bs,1)).to(torch.float8_e4m3fn)
        deq = (q.to(torch.float32) * sc.repeat_interleave(bs,0).repeat_interleave(bs,1))
        rel = (deq - w).abs().mean() / w.abs().mean()
        print(f"[selftest fp8 block-dequant] mean relative error = {rel:.4f}  "
              f"({'OK' if rel < 0.05 else 'HIGH'})")
        return

    os.makedirs(a.outdir, exist_ok=True)
    if a.indir:    # conversione locale (test)
        shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
        for i, sp in enumerate(shards):
            out = {}; convert_shard(sp, out, a.n_layers, a.ebits, a.io_bits, a.xbits,
                                    int2_variant=a.int2_quantizer)
            save_file_atomic(out, os.path.join(a.outdir, f"out-{i:05d}.safetensors"))
        # copia config + tokenizer
        for fn in ["config.json"]:
            src = os.path.join(a.indir, fn)
            if os.path.exists(src): shutil.copy(src, a.outdir)
        print(f"converted {len(shards)} shards -> {a.outdir}")
        return

    # reale: scarica shard per shard, converte, cancella
    # EN: real: download shard by shard, convert, delete
    #
    # ROBUSTEZZA RETE: timeout brevi sulle read cosi' un download appeso FALLISCE invece
    # di restare fermo per sempre. 8s, non 30: "timeout" = ZERO byte ricevuti in quella
    # finestra; su un transfer vivo i chunk arrivano di continuo, quindi 8s e' sicuro e
    # uno stallo costa 8s invece di 30.
    # EN: NETWORK ROBUSTNESS: short read timeouts so a hung download FAILS instead of
    # EN: sitting there forever. 8s, not 30: a "timeout" means ZERO bytes received in that
    # EN: window; a live transfer delivers chunks constantly, so 8s is safe and a stall
    # EN: costs 8s instead of 30.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    # log con timestamp: i messaggi "Trying to resume" di hf_hub diventano databili.
    # EN: timestamped logs: hf_hub's "Trying to resume" messages become datable.
    import logging
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    # hf_xet si blocca quando la rete si riavvia (connessioni zombie senza timeout):
    # forza la via HTTP classica, che curl ha dimostrato funzionare. (misurato 2026-07-02)
    # EN: hf_xet hangs when the network restarts (zombie connections with no timeout):
    # EN: force the classic HTTP path, which curl proved works (measured 2026-07-02).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # =0 per riabilitare xet / to re-enable xet
    from huggingface_hub import HfApi, hf_hub_download

    # lock anti-doppione: DUE convertitori sulla stessa outdir si corrompono a vicenda.
    # EN: anti-duplicate lock: TWO converters on the same outdir corrupt each other.
    import fcntl
    lock = open(os.path.join(a.outdir, ".convert.lock"), "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("ERROR: another converter is already using this output directory. Exiting."); return

    # dimensioni note dei file, riempite dopo repo_info: il downloader multi-stream le usa
    # per calcolare i confini dei segmenti e per sapere quando un file e' completo.
    # EN: known file sizes, filled after repo_info: the multi-stream downloader uses them
    # EN: to compute segment boundaries and to know when a file is complete.
    SIZES = {}

    def download_retry(repo, fn, dest, tries=999):
        """Downloader multi-stream con resume via Range. Apre N segmenti concorrenti
        (default 2, ILI_DL_STREAMS per cambiarli) e salva lo stato per-segmento in un
        sidecar .seg -> NESSUN byte perso comunque muoia la connessione. Un singolo stream
        HF e' limitato a ~2 MB/s (misurato); 2 stream ~ raddoppiano il throughput senza
        saturare una linea domestica. File piccoli, ILI_DL_STREAMS=1 o un vecchio .part
        legacy -> percorso a stream singolo (_download_single).
        EN: multi-stream Range-resume downloader. Opens N concurrent segments (default 2,
        EN: ILI_DL_STREAMS to change) and saves per-segment state in a .seg sidecar -> NO
        EN: byte is lost however the connection dies. A single HF stream is paced at
        EN: ~2 MB/s (measured); 2 streams roughly double throughput without saturating a
        EN: home line. Small files, ILI_DL_STREAMS=1 or a legacy .part -> single-stream
        EN: path (_download_single)."""
        import time as _t, threading, urllib.request, urllib.error
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        out = os.path.join(dest, fn); part = out + ".part"; side = part + ".seg"
        os.makedirs(dest, exist_ok=True)
        expected = SIZES.get(fn)
        if os.path.exists(out) and (expected is None or os.path.getsize(out) == expected):
            return out
        NS = max(1, min(8, int(os.environ.get("ILI_DL_STREAMS",
            os.environ.get("COLI_DL_STREAMS", "2")))))  # legacy fallback
        # un .part senza sidecar l'ha scritto una versione precedente a stream singolo.
        # EN: a .part without a sidecar was written by an older single-stream version.
        legacy = os.path.exists(part) and not os.path.exists(side)
        if expected is None or expected < (256 << 20) or NS == 1 or legacy:
            return _download_single(url, fn, out, part, expected)
        # ---- multi-stream ----
        segs = [(expected * t // NS, expected * (t + 1) // NS) for t in range(NS)]
        done = [0] * NS
        # riprendi lo stato dei segmenti se il sidecar combacia (stesso N, stessa size).
        # EN: resume per-segment progress if the sidecar matches (same N, same size).
        if os.path.exists(side):
            try:
                st = json.loads(open(side).read())
                if st.get("n") == NS and st.get("size") == expected: done = st["done"]
            except Exception: pass
        if not os.path.exists(part):
            with open(part, "wb") as f: f.truncate(expected)   # file sparse / sparse file
        fd = os.open(part, os.O_WRONLY)
        t0 = _t.time(); nres = [0]; log_lock = threading.Lock(); stopfail = []
        def worker(t):
            s0, s1 = segs[t]
            while done[t] < s1 - s0 and not stopfail:
                pos = s0 + done[t]
                req = urllib.request.Request(url, headers={"User-Agent": "iliria-convert",
                                                           "Range": f"bytes={pos}-{s1-1}"})
                try:
                    with urllib.request.urlopen(req, timeout=8) as r:
                        if r.status != 206:               # Range ignorato: multi-stream impossibile
                            stopfail.append(t); return    # EN: Range ignored: multi-stream impossible
                        while done[t] < s1 - s0:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            rem = (s1 - s0) - done[t]     # mai oltre il segmento / never past the segment
                            if len(chunk) > rem: chunk = chunk[:rem]
                            os.pwrite(fd, chunk, s0 + done[t])
                            done[t] += len(chunk)
                except KeyboardInterrupt: raise
                except Exception as ex:
                    with log_lock:
                        nres[0] += 1
                        print(f"    [dl] s{t}: {type(ex).__name__} at {(s0+done[t])/1e9:.2f} GB: "
                              f"resuming (#{nres[0]})", flush=True)
                    _t.sleep(min(15, 1 + nres[0] // NS))
        th = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(NS)]
        for x in th: x.start()
        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected: {NS} streams, "
              f"{sum(done)/1e9:.2f} of {expected/1e9:.2f} GB", flush=True)
        mark = sum(done); tmark = t0
        while any(x.is_alive() for x in th):
            _t.sleep(5)
            have = sum(done)
            tmpside = side + ".tmp"                       # checkpoint atomico / atomic checkpoint
            open(tmpside, "w").write(json.dumps({"n": NS, "size": expected, "done": list(done)}))
            os.replace(tmpside, side)
            now = _t.time()
            if now - tmark >= 30:
                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s, {NS} stream)", flush=True)
                mark = have; tmark = now
        os.close(fd)
        if stopfail:                                      # il server non onora il Range: fallback
            for f2 in (part, side):                       # EN: server won't honor Range: fall back
                if os.path.exists(f2): os.remove(f2)
            return _download_single(url, fn, out, part, expected)
        assert sum(done) == expected
        if os.path.exists(side): os.remove(side)
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9)
        print(f"    [dl] {fn}: {expected/1e9:.2f} GB in {dt/60:.1f} min "
              f"({expected/dt/1e6:.1f} MB/s avg, {NS} streams, {nres[0]} resumes)", flush=True)
        return out

    def _download_single(url, fn, out, part, expected):
        """Percorso a stream singolo con resume via Range (file piccoli / .part legacy /
        ILI_DL_STREAMS=1). Un EOF corto ma pulito conta come ripresa; se non arriva
        NESSUN byte nuovo, backoff invece di girare a vuoto.
        EN: single-stream path with Range resume (small files / legacy .part /
        EN: ILI_DL_STREAMS=1). A clean short EOF counts as a resume; if NO new byte
        EN: arrives, back off instead of spinning."""
        import time as _t, urllib.request, urllib.error
        t0 = _t.time(); nres = 0; mark = 0; tmark = t0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if expected is not None and have >= expected: break
            have0 = have
            req = urllib.request.Request(url, headers={"User-Agent": "iliria-convert"})
            if have: req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    if have and r.status == 200:          # server ha ignorato il Range: riparti pulito
                        have = 0                          # EN: server ignored Range: restart clean
                    if expected is None:
                        cl = r.headers.get("Content-Length")
                        if cl: expected = have + int(cl)
                    if have == 0 or nres:                 # segnale di vita subito / immediate sign of life
                        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected"
                              f"{f' @ {have/1e9:.2f} GB' if have else ''}"
                              f"{f' of {expected/1e9:.2f} GB' if expected else ''}", flush=True)
                    with open(part, "ab" if have else "wb") as f:
                        if not have: f.truncate(0)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            f.write(chunk); have += len(chunk)
                            if have - mark >= 512 * 1024 * 1024 or _t.time() - tmark >= 30:
                                now = _t.time()
                                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s)", flush=True)
                                mark = have; tmark = now
                if expected is None: break                # lunghezza ignota: passata singola / unknown length
                if have < expected:                       # EOF corto ma pulito: conta come ripresa
                    nres += 1                             # EN: clean short EOF: counts as a resume
                    if have == have0: _t.sleep(min(15, 1 + nres))   # zero progresso -> backoff / zero progress -> back off
            except KeyboardInterrupt: raise
            except urllib.error.HTTPError as ex:
                if ex.code == 416: break                  # gia' completo / already complete
                nres += 1
                print(f"    [dl] HTTP {ex.code} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
            except Exception as ex:
                nres += 1
                print(f"    [dl] {type(ex).__name__} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9); sz = os.path.getsize(out)
        print(f"    [dl] {fn}: {sz/1e9:.2f} GB in {dt/60:.1f} min "
              f"({sz/dt/1e6:.1f} MB/s avg, {nres} resumes)", flush=True)
        return out

    import time as _t
    for att in range(999):
        try:
            info = HfApi().repo_info(a.repo, files_metadata=True)
            # dimensioni note dallo store: abilitano il download multi-stream a segmenti.
            # EN: sizes known from the store: enable segmented multi-stream download.
            SIZES.update({s.rfilename: s.size for s in info.siblings if s.size})
            break
        except KeyboardInterrupt: raise
        except Exception as ex:
            w = min(60, 5*(att+1)); print(f"repo_info failed ({type(ex).__name__}); retrying in {w}s", flush=True); _t.sleep(w)
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try: shutil.copy(hf_hub_download(a.repo, fn, local_dir=a.outdir+"/_meta"), a.outdir)
        except Exception: pass
    tmp = os.path.join(a.outdir, "_inflight"); os.makedirs(tmp, exist_ok=True)
    if a.mtp:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        pref = f"model.layers.{a.n_layers}."
        mtp_shards = sorted(set(v for k, v in idx.items() if k.startswith(pref)))
        print(f"[MTP] head at layer {a.n_layers}: {len(mtp_shards)} shards to process: {mtp_shards}")
        for i, sh in enumerate(mtp_shards):
            outp = os.path.join(a.outdir, f"out-mtp-{i:05d}.safetensors")
            if verify_shard_file(outp):
                print(f"[MTP] {outp} already done"); continue
            if os.path.exists(outp):
                print(f"[MTP] {outp} exists but failed structural verification "
                      "(truncated/corrupt from an interrupted run) -- rebuilding", flush=True)
            print(f"[MTP {i+1}/{len(mtp_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=True,
                                    int2_variant=a.int2_quantizer)
            save_file_atomic(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB, {len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[MTP] DONE."); return
    if a.indexer:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        idx_shards = sorted(set(v for k, v in idx.items()
                                if "indexer" in k and 0 <= layer_idx(k) < a.n_layers))
        tot_gb = len(idx_shards) * 5.4
        print(f"[IDX] indexer weights across {len(idx_shards)} shards (~{tot_gb:.0f} GB total download, resumable)")
        for i, sh in enumerate(idx_shards):
            outp = os.path.join(a.outdir, f"out-idx-{i:05d}.safetensors")
            if verify_shard_file(outp): continue           # gia' fatto -> ripartibile
            if os.path.exists(outp):
                print(f"[IDX] {outp} exists but failed structural verification "
                      "(truncated/corrupt from an interrupted run) -- rebuilding", flush=True)
            print(f"[IDX {i+1}/{len(idx_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_idx=True,
                                    int2_variant=a.int2_quantizer)
            if out: save_file_atomic(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[IDX] DONE."); return
    for i, sh in enumerate(shards):
        if free_gb(a.outdir) < a.min_free_gb:
            print(f"STOP: free space is below {a.min_free_gb} GB. Free space and rerun to resume."); break
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if verify_shard_file(outp): continue               # gia' fatto -> ripartibile
        if os.path.exists(outp):
            print(f"[{i+1}/{len(shards)}] {outp} exists but failed structural verification "
                  "(truncated/corrupt from an interrupted run) -- rebuilding", flush=True)
        print(f"[{i+1}/{len(shards)}] downloading {sh} ({free_gb(a.outdir):.0f} GB free)...", flush=True)
        p = download_retry(a.repo, sh, tmp)
        out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits,
                                int2_variant=a.int2_quantizer)
        save_file_atomic(out, outp)
        os.remove(p)                                       # <-- cancella subito lo shard fp8
        for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            if os.path.isfile(blob): os.remove(blob)
        print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("DONE." if i == len(shards)-1 else "INTERRUPTED (rerun to resume).")

if __name__ == "__main__":
    main()
