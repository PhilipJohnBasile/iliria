#!/usr/bin/env python3
"""Generate an M5 Max engine variant with reusable hot-path state.

The stock implementation repeatedly allocates routing/activation workspaces and
recomputes identical CPU RoPE trigonometry across DSA/indexer layers. The
generated variant uses thread-local grow-only scratch and an eight-position RoPE
cache without changing operation order, precision, or public output buffers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_count(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()

    old_cpu_rope = '''/* RoPE interleaved su un vettore di dimensione qk_rope a posizione pos */
static void rope_interleave(float *v, int pos, const Cfg *c){
    int half = c->qk_rope/2; float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
    for(int j=0;j<half;j++){
        float inv = powf(c->theta, -2.0f*j/c->qk_rope);
        float ang = pos*inv, cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}
'''
    new_cpu_rope = '''/* M5 Max: cache CPU RoPE trig tables by position. DSA/indexer layers reuse the
 * same position, theta and qk_rope repeatedly; values are still computed with the
 * original powf/cosf/sinf sequence on the first call for a position. */
typedef struct { int pos, half; float theta, cs[128], sn[128]; } M5RopeSlot;
static _Thread_local M5RopeSlot g_m5_rope[8];
static void rope_interleave(float *v, int pos, const Cfg *c){
    int half=c->qk_rope/2; float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
    M5RopeSlot *r=&g_m5_rope[((unsigned)pos)&7u];
    if(r->pos!=pos || r->half!=half || r->theta!=c->theta){
        r->pos=pos; r->half=half; r->theta=c->theta;
        for(int j=0;j<half;j++){
            float inv=powf(c->theta,-2.0f*j/c->qk_rope), ang=pos*inv;
            r->cs[j]=cosf(ang); r->sn[j]=sinf(ang);
        }
    }
    for(int j=0;j<half;j++){
        float a=in[2*j], b=in[2*j+1], cs=r->cs[j], sn=r->sn[j];
        v[j]=a*cs-b*sn; v[half+j]=b*cs+a*sn;
    }
}
'''
    text = replace_once(text, old_cpu_rope, new_cpu_rope, 'CPU RoPE cache')

    attention_marker = '''static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
'''
    attention_scratch = '''/* M5 Max: retain the dynamically-sized attention-score arena introduced by
 * the >8192-context safety fix. One inference caller owns it; OpenMP workers use
 * disjoint slices inside the arena during each parallel region. */
typedef struct { float *p; size_t cap; } M5AttnScratch;
static _Thread_local M5AttnScratch g_m5_attn_scratch;
static float *m5_attn_scores(size_t elems){
    if(elems>SIZE_MAX/sizeof(float)){ fprintf(stderr,"M5 attention scratch overflow\n"); exit(1); }
    size_t bytes=elems*sizeof(float);
    if(bytes>g_m5_attn_scratch.cap){
        size_t cap=g_m5_attn_scratch.cap?g_m5_attn_scratch.cap:4096;
        while(cap<bytes){
            if(cap>SIZE_MAX/2){ cap=bytes; break; }
            cap*=2;
        }
        void *p=realloc(g_m5_attn_scratch.p,cap);
        if(!p){ fprintf(stderr,"OOM M5 attention scratch (%zu bytes)\n",cap); exit(1); }
        g_m5_attn_scratch.p=(float*)p; g_m5_attn_scratch.cap=cap;
    }
    return g_m5_attn_scratch.p;
}

'''
    text = replace_once(
        text,
        attention_marker,
        attention_scratch + attention_marker,
        'attention scratch insertion',
    )
    text = replace_count(
        text,
        '        float *sc_all=falloc((int64_t)omp_get_max_threads()*sc_cap);\n',
        '        float *sc_all=m5_attn_scores((size_t)omp_get_max_threads()*(size_t)sc_cap);\n',
        1,
        'absorbed attention score allocation',
    )
    text = replace_count(
        text,
        '    float *sc_all=falloc((int64_t)omp_get_max_threads()*sc_cap);\n',
        '    float *sc_all=m5_attn_scores((size_t)omp_get_max_threads()*(size_t)sc_cap);\n',
        1,
        'dense attention score allocation',
    )
    text = replace_once(
        text,
        '        free(ctx); free(Q); free(QR); free(comp); free(sc_all);\n',
        '        free(ctx); free(Q); free(QR); free(comp);\n',
        'absorbed attention score retention',
    )
    text = replace_once(
        text,
        '    free(ctx); free(Q); free(QR); free(comp); free(kvb_all); free(sc_all);\n',
        '    free(ctx); free(Q); free(QR); free(comp); free(kvb_all);\n',
        'dense attention score retention',
    )

    marker = '''/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
'''
    scratch = '''/* M5 Max generated hot path: one grow-only scratch arena per inference thread.
 * Forward execution is sequential on the caller thread; OpenMP is used inside
 * kernels and expert loaders, not to invoke overlapping layer functions.
 * Separate buffers preserve the original aliasing and numerical behavior. */
typedef struct { void *p; size_t cap; } M5ScratchBuf;
typedef struct {
    M5ScratchBuf logit, choice, idxs, ws, keff, uniq, seen;
    M5ScratchBuf xg, gg, uu, hh, rows, rw;
    M5ScratchBuf mxg, mrows, mrw, sg, su;
    M5ScratchBuf router_nrm, router_ch;
    M5ScratchBuf dense_g, dense_u;
} M5MoeScratch;
static _Thread_local M5MoeScratch g_m5_moe_scratch;
static void *m5_scratch_get(M5ScratchBuf *b, size_t bytes){
    if(bytes==0) bytes=1;
    if(bytes>b->cap){
        size_t cap=b->cap?b->cap:4096;
        while(cap<bytes){
            if(cap>SIZE_MAX/2){ cap=bytes; break; }
            cap*=2;
        }
        void *p=realloc(b->p,cap);
        if(!p){ fprintf(stderr,"OOM M5 hot-path scratch (%zu bytes)\n",cap); exit(1); }
        b->p=p; b->cap=cap;
    }
    return b->p;
}

'''
    text = replace_once(text, marker, scratch + marker, 'scratch insertion')

    text = replace_once(
        text,
        '    float *logit=falloc(E), *choice=falloc(E);\n',
        '''    M5MoeScratch *ms=&g_m5_moe_scratch;
    float *logit=(float*)m5_scratch_get(&ms->logit,(size_t)E*sizeof(float));
    float *choice=(float*)m5_scratch_get(&ms->choice,(size_t)E*sizeof(float));
''',
        'routing float buffers',
    )
    text = replace_once(
        text,
        '    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));\n    int *keff=malloc(S*sizeof(int));\n',
        '''    int *idxs=(int*)m5_scratch_get(&ms->idxs,(size_t)S*K*sizeof(int));
    float *ws=(float*)m5_scratch_get(&ms->ws,(size_t)S*K*sizeof(float));
    int *keff=(int*)m5_scratch_get(&ms->keff,(size_t)S*sizeof(int));
''',
        'routing index buffers',
    )
    text = replace_once(
        text,
        '    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;\n    unsigned char seen[E]; memset(seen,0,(size_t)E);\n',
        '''    int *uniq=(int*)m5_scratch_get(&ms->uniq,(size_t)E*sizeof(int)); int nu=0;
    unsigned char *seen=(unsigned char*)m5_scratch_get(&ms->seen,(size_t)E); memset(seen,0,(size_t)E);
''',
        'union buffers',
    )
    text = replace_once(
        text,
        '    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);\n    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));\n',
        '''    float *xg=(float*)m5_scratch_get(&ms->xg,(size_t)S*D*sizeof(float));
    float *gg=(float*)m5_scratch_get(&ms->gg,(size_t)S*I*sizeof(float));
    float *uu=(float*)m5_scratch_get(&ms->uu,(size_t)S*I*sizeof(float));
    float *hh=(float*)m5_scratch_get(&ms->hh,(size_t)S*D*sizeof(float));
    int *rows=(int*)m5_scratch_get(&ms->rows,(size_t)S*sizeof(int));
    float *rw=(float*)m5_scratch_get(&ms->rw,(size_t)S*sizeof(float));
''',
        'expert activation buffers',
    )
    text = replace_once(
        text,
        '            mxg=falloc((int64_t)(nb+1)*S*D);\n            mrows=malloc((size_t)(nb+1)*S*sizeof(int)); mrw=malloc((size_t)(nb+1)*S*sizeof(float));\n',
        '''            mxg=(float*)m5_scratch_get(&ms->mxg,(size_t)(nb+1)*S*D*sizeof(float));
            mrows=(int*)m5_scratch_get(&ms->mrows,(size_t)(nb+1)*S*sizeof(int));
            mrw=(float*)m5_scratch_get(&ms->mrw,(size_t)(nb+1)*S*sizeof(float));
''',
        'Metal staging buffers',
    )
    text = replace_once(
        text,
        '            free(mxg); free(mrows); free(mrw);\n',
        '            /* generated scratch persists for the next layer */\n',
        'Metal staging frees',
    )
    text = replace_once(
        text,
        '    float *sg=falloc((int64_t)S*sI), *su=falloc((int64_t)S*sI);\n',
        '''    float *sg=(float*)m5_scratch_get(&ms->sg,(size_t)S*sI*sizeof(float));
    float *su=(float*)m5_scratch_get(&ms->su,(size_t)S*sI*sizeof(float));
''',
        'shared expert buffers',
    )
    text = replace_once(
        text,
        '    free(logit); free(choice); free(idxs); free(ws); free(keff); free(uniq);\n    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(sg); free(su);\n',
        '    /* generated thread-local scratch is retained across MoE layers and tokens */\n',
        'MoE frees',
    )

    text = replace_count(
        text,
        '    float *nrm=falloc(D), *ch=falloc(E);\n',
        '''    M5MoeScratch *ms=&g_m5_moe_scratch;
    float *nrm=(float*)m5_scratch_get(&ms->router_nrm,(size_t)D*sizeof(float));
    float *ch=(float*)m5_scratch_get(&ms->router_ch,(size_t)E*sizeof(float));
''',
        2,
        'lookahead router workspaces',
    )
    text = replace_count(
        text,
        '    free(nrm); free(ch);\n',
        '    /* generated router scratch persists */\n',
        2,
        'lookahead router frees',
    )

    text = replace_once(
        text,
        '    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);\n',
        '''    M5MoeScratch *ms=&g_m5_moe_scratch;
    float *g=(float*)m5_scratch_get(&ms->dense_g,(size_t)S*I*sizeof(float));
    float *u=(float*)m5_scratch_get(&ms->dense_u,(size_t)S*I*sizeof(float));
''',
        'dense MLP buffers',
    )
    text = replace_once(
        text,
        '    free(g); free(u);\n',
        '    /* generated dense-MLP scratch persists */\n',
        'dense MLP frees',
    )

    args.output.write_text('/* Generated by tools/gen_m5max_engine.py; do not edit directly. */\n' + text)
    patcher = Path(__file__).with_name('patch_m5max_grouped_cpu_moe.py')
    subprocess.run([sys.executable, str(patcher), str(args.output), str(args.output)], check=True)
    fixer = Path(__file__).with_name('fix_m5max_grouped_build.py')
    subprocess.run([sys.executable, str(fixer), str(args.output), str(args.output)], check=True)


if __name__ == '__main__':
    main()
