#!/usr/bin/env python3
"""Add opt-in two-stage (gate+up / down) miss reads to generated M5 Max C.

n2 falsifier hardware-prep (docs/performance-theory.json,
n2-staged-intra-expert-streaming): today's expert_load reads one coalesced
~18.9MB block per missed expert (gate/up/down contiguous) and waits for the
WHOLE block before any compute starts. This patch adds
expert_load_staged_begin()/expert_load_staged_join(): gate+up (~12.6MB) is
read on the calling thread while down (~6.3MB) streams CONCURRENTLY on a
background thread started FIRST (so it is truly concurrent, not merely
reordered); the caller can start gate/up compute as soon as begin() returns,
and must call join() before touching the down projection ("down-GEMV waits
on both" -- down's own read, joined there, and gate/up's compute, already
done by then via ordinary program order).

Activated at runtime with ILI_STAGED_EXPERT_READ=1 (or COLI_/FA_ legacy
prefixes via ili_env). The production source (glm.c) stays untouched --
same convention as every other tools/patch_m5max_*.py in this chain.

Byte-identical slab/fslab layout to expert_load (same tensor resolution via
st_find, same fmt/O/I setup): a slot loaded via this path is
indistinguishable from one loaded via expert_load -- nothing downstream
(matmul_qt, LRU swap, pin promotion) needs to know which path filled it, and
tests/test_patch_m5max_staged_expert_read.py proves the two produce
byte-identical slabs on the same expert.

Engine wiring is intentionally narrow: only moe()'s BLOCKING (non-PIPE)
miss-load loop is touched, replacing expert_load with
begin()-immediately-followed-by-join() per missed expert -- this buys real
gate-up/down READ-READ concurrency (down's ~6.3MB overlaps gate+up's
~12.6MB within one expert's own load) with zero change to anything
downstream. The FURTHER refinement this entry's hypothesis actually
describes -- gate/up COMPUTE overlapping with down's read, not just two
reads overlapping each other -- needs begin() and join() split across
moe()'s dispatch and compute phases (a bigger, PIPE-Pool-shaped rewrite, out
of scope for this hardware-prep pass); that overlap is measured directly,
in isolation, by tests/staged_expert_read_microbench.c instead, which calls
begin() and join() around its own timed gate/up compute step. Composing
this with PIPE=1 or the grouped-CPU-MoE path is also out of scope (the n2
notes call exactly this kind of composition future work, not required now).

Requires only the SAME on-the-fly tensor-offset lookup expert_load already
does (st_find) -- no precomputed disk-descriptor layout is assumed, per the
2026-07-15 build's explicit note not to depend on a parallel
worktree's descriptor format.

Scope, stated not hidden: no O_DIRECT alignment (always buffered pread,
regardless of ILI_DIRECT) and no ILI_MMAP support (falls back to plain
expert_load when mmap mode is active -- mmap is zero-copy; staging a pread
split does not apply to it). Production hardening of either is future work
once the falsifier justifies it.
"""

from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


STAGED_READ_BLOCK = r'''
/* ---- M5 staged intra-expert reads (opt-in, n2 falsifier hardware-prep) ------
 * See tools/patch_m5max_staged_expert_read.py's module docstring for the full
 * design and scope notes. ILI_STAGED_EXPERT_READ=1 activates it in moe()'s
 * blocking miss-load loop.
 */
typedef struct {
    ESlot *s; int fatal;
    st_tensor *tw_d, *tq_d;      /* down weight + scale tensor descriptors */
    int64_t pos_d, fo_d;         /* byte/float offset of down inside slab/fslab */
    pthread_t th; int started, ok;
} StagedLoad;

static void *m5_staged_down_reader(void *arg){
    StagedLoad *sl=(StagedLoad*)arg;
    st_tensor *tw=sl->tw_d, *tq=sl->tq_d; ESlot *s=sl->s;
    ssize_t r1=pread(tw->fd, s->slab+sl->pos_d, tw->nbytes, tw->off);
    ssize_t r2=pread(tq->fd, (char*)(s->fslab+sl->fo_d), tq->nbytes, tq->off);
    sl->ok = (r1==tw->nbytes && r2==tq->nbytes);
    if(!sl->ok){
        perror("pread staged down");
        if(sl->fatal) exit(1);
    } else if(g_drop){
        posix_fadvise(tw->fd, tw->off, tw->nbytes, POSIX_FADV_DONTNEED);
        posix_fadvise(tq->fd, tq->off, tq->nbytes, POSIX_FADV_DONTNEED);
    }
    return NULL;
}

/* Issues down's read on a background thread FIRST, then reads gate+up
 * synchronously (one coalesced pread when contiguous on disk, else two
 * sequential preads), so gate/up are fully loaded and down's read is still
 * in flight (or already done) by the time this returns. The caller may
 * start gate/up compute immediately; MUST call expert_load_staged_join()
 * before touching s->d. Returns 0 on success (down still in flight -- join
 * it), or -1 on a non-fatal failure (fatal=0 mirrors expert_load's
 * speculative-load contract). Falls back to plain expert_load (no thread,
 * no staging) when g_mmap is active or the model has no .qs sidecars --
 * the SAME fallback conditions expert_load itself has. */
static int expert_load_staged_begin(Model *m, int layer, int eid, ESlot *s, int fatal, StagedLoad *sl){
    memset(sl,0,sizeof(*sl));
    sl->s=s; sl->fatal=fatal;
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(g_mmap || !st_has(&m->S,qn))            /* no staging benefit / no .qs sidecars: plain load */
        return expert_load(m,layer,eid,s,fatal);

    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"missing %s\n",nm[k]); if(fatal) exit(1); return -1; }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    if(!s->slab || wtot+8192 > s->slab_cap){
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=need;
    }
    if(!s->fslab || ftot > s->fslab_cap){
        free(s->fslab);
        if(fatal){ s->fslab=falloc(ftot); }
        else {
            if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
               !(s->fslab=malloc((size_t)ftot*sizeof(float)))){
                fprintf(stderr,"OOM fslab\n");
                compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;
                s->fslab=NULL; s->fslab_cap=0; return -1;
            }
        }
        s->fslab_cap=ftot;
    }
    /* Fixed gate,up,down slab order (staging always reads gate+up as its own
     * pair, so unlike expert_load's contiguous-pread fast path this layout
     * does not need to mirror on-disk order). */
    int64_t pos[3]={0, tw[0]->nbytes, tw[0]->nbytes+tw[1]->nbytes};
    int64_t fo[3]={0, tq[0]->nbytes/4, tq[0]->nbytes/4+tq[1]->nbytes/4};

    sl->tw_d=tw[2]; sl->tq_d=tq[2]; sl->pos_d=pos[2]; sl->fo_d=fo[2];
    if(pthread_create(&sl->th,NULL,m5_staged_down_reader,sl)!=0){
        fprintf(stderr,"staged down-read thread create failed; falling back to single-read\n");
        return expert_load(m,layer,eid,s,fatal);
    }
    sl->started=1;

    int gu_contig = tw[0]->fd==tw[1]->fd && tw[0]->off+tw[0]->nbytes==tw[1]->off;
    int ok=1;
    if(gu_contig){
        if(pread(tw[0]->fd, s->slab+pos[0], tw[0]->nbytes+tw[1]->nbytes, tw[0]->off)
           != tw[0]->nbytes+tw[1]->nbytes){ perror("pread staged gate+up"); ok=0; }
    } else {
        if(pread(tw[0]->fd, s->slab+pos[0], tw[0]->nbytes, tw[0]->off)!=tw[0]->nbytes){ perror("pread staged gate"); ok=0; }
        if(ok && pread(tw[1]->fd, s->slab+pos[1], tw[1]->nbytes, tw[1]->off)!=tw[1]->nbytes){ perror("pread staged up"); ok=0; }
    }
    if(ok){
        if(pread(tq[0]->fd, (char*)(s->fslab+fo[0]), tq[0]->nbytes, tq[0]->off)!=tq[0]->nbytes){ perror("pread staged gate qs"); ok=0; }
        if(ok && pread(tq[1]->fd, (char*)(s->fslab+fo[1]), tq[1]->nbytes, tq[1]->off)!=tq[1]->nbytes){ perror("pread staged up qs"); ok=0; }
    }
    if(ok && g_drop){
        posix_fadvise(tw[0]->fd, tw[0]->off, gu_contig?tw[0]->nbytes+tw[1]->nbytes:tw[0]->nbytes, POSIX_FADV_DONTNEED);
        if(!gu_contig) posix_fadvise(tw[1]->fd, tw[1]->off, tw[1]->nbytes, POSIX_FADV_DONTNEED);
        posix_fadvise(tq[0]->fd, tq[0]->off, tq[0]->nbytes, POSIX_FADV_DONTNEED);
        posix_fadvise(tq[1]->fd, tq[1]->off, tq[1]->nbytes, POSIX_FADV_DONTNEED);
    }
    if(!ok){
        pthread_join(sl->th,NULL);   /* don't leak the down-read thread on the error path */
        if(fatal) exit(1);
        return -1;
    }

    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int fmt = (nb==(int64_t)OO[k]*II[k])?1 : (nb==(int64_t)OO[k]*((II[k]+1)/2))?2 : 3;
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=s->fslab+fo[k];
    }
    s->eid=eid;
    return 0;
}

/* Waits for down's background read (if one was started); a no-op when
 * expert_load_staged_begin() fell back to plain expert_load (no thread was
 * started). Must be called (and must return 0) before touching s->d. */
static int expert_load_staged_join(StagedLoad *sl){
    if(!sl->started) return 0;
    pthread_join(sl->th,NULL);
    sl->started=0;
    if(!sl->ok && sl->fatal) exit(1);   /* belt+braces: m5_staged_down_reader already exited if fatal */
    return sl->ok ? 0 : -1;
}

'''

MARKER = """/* ============================ PIPE: load ‖ matmul ============================
"""

ENV_ANCHOR = '''    g_pipe = getenv("PIPE")?atoi(getenv("PIPE")):0;       /* default OFF: overlap expert load ‖ matmul (byte-identical; reorders I/O). PIPE=1 opts in */
'''
ENV_REPL = '''    g_pipe = getenv("PIPE")?atoi(getenv("PIPE")):0;       /* default OFF: overlap expert load ‖ matmul (byte-identical; reorders I/O). PIPE=1 opts in */
    { const char *ser=ili_env("STAGED_EXPERT_READ"); g_staged_expert_read=ser&&atoi(ser)!=0; }
'''

GLOBAL_ANCHOR = "static int g_pipe=0;      /* PIPE=1: async expert-load pipeline (default OFF) */\n"
GLOBAL_REPL = (
    "static int g_pipe=0;      /* PIPE=1: async expert-load pipeline (default OFF) */\n"
    "static int g_staged_expert_read=0;  /* ILI_STAGED_EXPERT_READ=1: two-stage gate+up/down miss reads (blocking path only) */\n"
)

DISPATCH_ANCHOR = """            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                if(g_a2_on){ g_a2_issue=t0; g_a2_nmiss=(nmiss>64?64:nmiss); }
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++){ expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1);
                    if(g_a2_on && q<64) g_a2_comp[q]=now_s(); }
                double ddt=now_s()-t0; m->t_edisk += ddt;
                if(g_a2_on){ g_a2_load_end=now_s(); g_a2_nb=nb; g_a2_bytes=(int64_t)g_a2_nmiss*18915328; }
                /* No PIPE = no overlap mechanism at all here: this whole span is, by
                 * construction, consumer-blocked-because-data-unavailable, so it is fully
                 * exposed stall too (t_stall_exposed mirrors t_edisk exactly in this mode). */
                m->t_stall_exposed += ddt; }
"""
DISPATCH_REPL = """            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++){
                    if(g_staged_expert_read){
                        StagedLoad m5_sl;
                        expert_load_staged_begin(m,layer,uniq[base+missk[q]],&m->ws[q],1,&m5_sl);
                        expert_load_staged_join(&m5_sl);
                    } else expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1);
                }
                double ddt=now_s()-t0; m->t_edisk += ddt;
                /* staged or not, this whole span remains consumer-blocked-because-data-
                 * unavailable (no PIPE overlap here) -- mirror it into exposed stall exactly
                 * as the unpatched engine does. */
                m->t_stall_exposed += ddt; }
"""


def patch_text(text: str) -> str:
    end_of_expert_load = "    s->eid=eid; return 0;\n}\n"
    text = replace_once(
        text, end_of_expert_load, end_of_expert_load + STAGED_READ_BLOCK,
        "staged read functions insertion (after expert_load)")
    if MARKER not in text:
        raise RuntimeError("PIPE marker comment not found (unexpected glm.c shape)")

    text = replace_once(text, GLOBAL_ANCHOR, GLOBAL_REPL, "g_staged_expert_read global")
    text = replace_once(text, ENV_ANCHOR, ENV_REPL, "g_staged_expert_read env wiring")
    text = replace_once(text, DISPATCH_ANCHOR, DISPATCH_REPL, "blocking miss-load staged wiring")
    return text


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch_text(args.source.read_text()))


if __name__ == "__main__":
    main()
