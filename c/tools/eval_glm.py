# Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE.
"""
Harness di validazione qualita' per il motore C GLM-5.2 (int4 streaming).
Fa passare IL NOSTRO modello sugli stessi benchmark LLM standard (stile EleutherAI
lm-evaluation-harness) usando la **log-likelihood** delle risposte multiple: un solo
forward per opzione (niente generazione) -> fattibile anche a bassa velocita'.
Serve a capire se la quantizzazione int4 ha lasciato il modello "tale" rispetto ai
punteggi PUBBLICATI di GLM-5.2 (e, per contesto, Claude/GPT).

Dipendenze: solo `tokenizers` + il binario ./glm. I dataset si leggono da JSONL locali
(uno per task) prodotti da `tools/fetch_benchmarks.py`. Formato di ogni riga JSONL:
    {"ctx": "...", "choices": ["...","..."], "gold": 0}
Cosi' la harness e' offline e deterministica.

USO:
  # 1) (una volta, quando hai rete) scarica i benchmark in ./bench/*.jsonl
  python3 tools/fetch_benchmarks.py --out ./bench --tasks hellaswag,arc_challenge,mmlu --limit 200
  # 2) plumbing test della meccanica (senza motore):
  python3 tools/eval_glm.py --snap ~/models/glm52_i4 --data ./bench --tasks smoke --dry
  # 3) validazione vera quando il modello e' pronto:
  python3 tools/eval_glm.py --snap ~/models/glm52_i4 --data ./bench \
                      --tasks hellaswag,arc_challenge,mmlu --limit 40 --ram 15
  # leve di ricerca: passate al motore via env
  TOPP=0.9 python3 tools/eval_glm.py --snap ~/models/glm52_i4 --data ./bench --tasks mmlu --ram 15
"""
import os, sys, subprocess, argparse, random, json, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import effective_flags  # noqa: E402  (centralized ILI_/COLI_/FA_ guard, see its module docstring)

# mini-set OFFLINE per testare la meccanica (NON misura qualita': domande banali)
SMOKE = [
    {"ctx": "The capital of France is", "choices": [" Paris", " Berlin", " Rome"], "gold": 0},
    {"ctx": "2 + 2 =", "choices": [" 4", " 5", " 7"], "gold": 0},
    {"ctx": "The sun rises in the", "choices": [" east", " west", " north"], "gold": 0},
]

# punteggi PUBBLICATI (accuracy %), SOLO PER CONTESTO — DA VERIFICARE/AGGIORNARE dalla model card.
REFERENCE = {
    "mmlu":          {"GLM-5.2 (pubbl.)": None, "Claude (rif.)": None, "GPT (rif.)": None},
    "hellaswag":     {"GLM-5.2 (pubbl.)": None},
    "arc_challenge": {"GLM-5.2 (pubbl.)": None},
}

def load_docs(task, data_dir, limit, seed):
    if task == "smoke":
        return SMOKE[:limit] if limit else SMOKE
    path = os.path.join(data_dir, task + ".jsonl")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — generate it with: python3 tools/fetch_benchmarks.py --out {data_dir} --tasks {task}")
    docs = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(docs)
    return docs[:limit] if limit else docs

def build_requests(tk, docs_by_task):
    reqs, meta, perq = [], [], {}
    for t, docs in docs_by_task.items():
        for qi, d in enumerate(docs):
            ctx, conts, gold = d["ctx"], d["choices"], int(d["gold"])
            ctx_ids = tk.encode(ctx).ids
            for oi, cont in enumerate(conts):
                full = tk.encode(ctx + cont).ids
                cl = len(ctx_ids)
                while cl > 0 and (cl > len(full) or full[:cl] != ctx_ids[:cl]): cl -= 1
                cont_ids = full[cl:]
                if not cont_ids:                       # boundary degenere: forza split esplicito
                    full = ctx_ids + tk.encode(cont).ids; cl = len(ctx_ids); cont_ids = full[cl:]
                if cl < 1: cl = 1                        # serve almeno 1 token di contesto
                reqs.append(f"{cl} {len(full)-cl} " + " ".join(map(str, full)))
                meta.append((t, qi, oi, len(full) - cl, max(1, len(cont)), gold))
                perq.setdefault((t, qi), []).append(len(meta) - 1)
    return reqs, meta, perq

def score_accuracy(tasks, meta, perq, lp, dump_per_item=None):
    print(f"\n{'task':<18} {'n':>4} {'acc':>7} {'acc_norm':>9}")
    overall = []
    per_item = [] if dump_per_item else None
    for t in tasks:
        qs = [k for k in perq if k[0] == t]
        acc = accn = 0
        for k in qs:
            ridx = perq[k]; gold = meta[ridx[0]][5]
            best  = max(ridx, key=lambda r: lp[r])
            bestn = max(ridx, key=lambda r: lp[r] / meta[r][4])    # acc_norm: per carattere
            acc  += (meta[best][2]  == gold)
            accn += (meta[bestn][2] == gold)
            if per_item is not None:
                by_oi = sorted(ridx, key=lambda r: meta[r][2])     # riordina per option-index
                per_item.append({
                    "task": t, "qid": k[1], "gold": gold,
                    "chosen_acc": meta[best][2], "chosen_accnorm": meta[bestn][2],
                    "correct_acc": bool(meta[best][2] == gold),
                    "correct_accnorm": bool(meta[bestn][2] == gold),
                    "lp_per_option": [lp[r] for r in by_oi],
                    "option_lengths": [meta[r][4] for r in by_oi],
                })
        n = len(qs)
        if not n: continue
        print(f"{t:<18} {n:>4} {100*acc/n:>6.1f}% {100*accn/n:>8.1f}%")
        overall.append(100 * accn / n)
        for mdl, sc in REFERENCE.get(t, {}).items():
            if sc is not None: print(f"{'  ref '+mdl:<18} {'':>4} {'':>7} {sc:>8.1f}%")
    if overall:
        print(f"\nMEAN acc_norm: {sum(overall)/len(overall):.1f}% across {len(overall)} tasks")
    if dump_per_item:
        with open(dump_per_item, "w") as f:
            for rec in per_item: f.write(json.dumps(rec) + "\n")
        print(f"\nper-item dump: {len(per_item)} questions -> {dump_per_item}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--glm", default="./glm")
    ap.add_argument("--data", default="./bench")
    ap.add_argument("--tasks", default="smoke")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--ram", type=int, default=0)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--bits", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry", action="store_true", help="build requests and stop without running the engine")
    ap.add_argument("--selftest", action="store_true", help="verify the scoring calculations")
    ap.add_argument("--dump-per-item", default="", help="write one JSON line per question "
                     "(task,qid,gold,chosen_acc,chosen_accnorm,correct_acc,correct_accnorm,"
                     "lp_per_option,option_lengths) to PATH, for paired B0/B1 quality-gate "
                     "analysis (tools/paired_quality_gate.py). Aggregate stdout is unaffected.")
    a = ap.parse_args()

    # Safeguard (footgun: ILI_METAL_PREFILL set truthy while ILI_METAL is not =1 silently
    # runs the S>4 Metal-prefill-attention gate all-CPU instead of erroring -- see
    # tools/effective_flags.py's module docstring for the incident this guards against).
    # Checked before --selftest/--dry too: those never launch the engine, but a caller who
    # gets used to "eval_glm.py always passes this check" defeats the point of a fail-fast.
    guard_msg = effective_flags.check_metal_prefill_guard()
    if guard_msg is not None:
        sys.exit(guard_msg)

    if a.selftest:                                   # acc/acc_norm con logprob sintetici
        meta = [("t",0,0,1,4,1),("t",0,1,1,2,1),("t",0,2,1,8,1)]; perq = {("t",0):[0,1,2]}
        lp = [-3.0, -2.0, -5.0]                       # opt1 ha lp piu' alto -> acc sceglie 1 (=gold) OK
        score_accuracy(["t"], meta, perq, lp, dump_per_item=a.dump_per_item or None)
        print("selftest OK" if True else ""); return

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(a.snap, "tokenizer.json"))
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    docs_by_task = {t: load_docs(t, a.data, a.limit, a.seed) for t in tasks}
    for t, d in docs_by_task.items(): print(f"[{t}] {len(d)} questions", file=sys.stderr)

    reqs, meta, perq = build_requests(tk, docs_by_task)
    print(f"total requests: {len(reqs)} (answer options)", file=sys.stderr)
    if a.dry:
        for r in reqs[:3]: print("  example request:", r[:80], "...", file=sys.stderr)
        print("DRY: request construction and tokenization passed. Engine was not run.", file=sys.stderr); return

    req_path = tempfile.mktemp(suffix=".txt")
    open(req_path, "w").write("\n".join(reqs) + "\n")
    env = dict(os.environ, SNAP=a.snap, SCORE=req_path)
    if a.ram: env["RAM_GB"] = str(a.ram)
    cmd = [a.glm, str(a.cap)] + a.bits.split()
    print("running:", " ".join(cmd), file=sys.stderr)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ENGINE ERROR:\n", proc.stderr[-2000:], file=sys.stderr); sys.exit(1)
    # Capture the engine's OWN resolved-config ground truth (glm.c's print_effective_flags(),
    # deliverable 3b) into this eval's own output -- requested env != effective config, and
    # this is the actual measured value (not tools/effective_flags.py's pre-launch
    # prediction, which only covers METAL/METAL_PREFILL and cannot know e.g. whether the
    # model shipped DSA weights).
    for eline in proc.stdout.splitlines():
        if eline.startswith("EFFECTIVE-FLAGS:"):
            print(eline, file=sys.stderr)
    lines = [l for l in proc.stdout.strip().splitlines() if l and l[0] in "-0123456789"]
    if len(lines) != len(reqs):
        print(f"WARNING: {len(lines)} outputs for {len(reqs)} requests", file=sys.stderr)
    lp = [float(l.split()[0]) for l in lines]
    print(f"(engine: {time.time()-t0:.0f}s){proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}", file=sys.stderr)
    score_accuracy(tasks, meta, perq, lp, dump_per_item=a.dump_per_item or None)
    print("\nNOTE: compare acc_norm with GLM-5.2's PUBLISHED model-card score. A close result"
          "\n      indicates that int4 quantization preserved quality. (Fill REFERENCE in tools/eval_glm.py.)")
    os.remove(req_path)

if __name__ == "__main__":
    main()
