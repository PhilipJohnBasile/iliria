#!/usr/bin/env python3
"""Add a ragged, pointer-based grouped CPU MoE path to the generated M5 engine.

The generated implementation keeps expert weights in their existing pin/LRU/file-backed
slabs. It never repacks them into a physical 3-D tensor. Instead it:

* gathers the routed rows for all CPU-owned experts in the current block (glm.c's own
  per-expert `handled[]` mask says which experts its per-format Metal sub-blocks did NOT
  cover -- format-inconsistent experts, a format with no GPU sub-block, or a sub-block
  whose GPU submission failed);
* quantizes each gate/up input row once and shares it between both projections;
* runs gate/up, activation, hidden quantization, and down projection under one
  persistent OpenMP team, tiled across every expert;
* keeps expert outputs private and scatters them in deterministic expert/row order.

All changes apply only to glm_m5max.c. The source glm.c remains the reference path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()

    # The grouped path needs enough scratch for every routed row in the batch, not
    # just the rows of one expert. R <= S*K by construction.
    text = replace_once(
        text,
        '''    float *xg=(float*)m5_scratch_get(&ms->xg,(size_t)S*D*sizeof(float));
    float *gg=(float*)m5_scratch_get(&ms->gg,(size_t)S*I*sizeof(float));
    float *uu=(float*)m5_scratch_get(&ms->uu,(size_t)S*I*sizeof(float));
    float *hh=(float*)m5_scratch_get(&ms->hh,(size_t)S*D*sizeof(float));
    int *rows=(int*)m5_scratch_get(&ms->rows,(size_t)S*sizeof(int));
    float *rw=(float*)m5_scratch_get(&ms->rw,(size_t)S*sizeof(float));
''',
        '''    size_t m5_rmax=(size_t)S*(size_t)K;
    float *xg=(float*)m5_scratch_get(&ms->xg,m5_rmax*(size_t)D*sizeof(float));
    float *gg=(float*)m5_scratch_get(&ms->gg,m5_rmax*(size_t)I*sizeof(float));
    float *uu=(float*)m5_scratch_get(&ms->uu,m5_rmax*(size_t)I*sizeof(float));
    float *hh=(float*)m5_scratch_get(&ms->hh,m5_rmax*(size_t)D*sizeof(float));
    int *rows=(int*)m5_scratch_get(&ms->rows,m5_rmax*sizeof(int));
    float *rw=(float*)m5_scratch_get(&ms->rw,m5_rmax*sizeof(float));
''',
        "expanded routed-row scratch",
    )

    marker = '''/* quantizza w[O,I] f32 -> int8 q[O,I] + scala[O] simmetrica per riga */
'''
    grouped = r'''/* ---- M5 MAX RAGGED GROUPED CPU MoE -----------------------------------------
 * Logical shape: weights[expert][output][input]. Physical weights remain in their
 * original ESlot slabs, so this path adds no expert-weight copy or repack.
 */
typedef struct {
    ESlot *slot;
    int off, nr;
} M5CpuExpert;

typedef struct {
    int8_t *xq;
    size_t xq_cap;
    float *sx;
    size_t sx_cap;
} M5GroupedScratch;
static _Thread_local M5GroupedScratch g_m5_grouped_scratch;

static int g_m5_cpu_grouped=-1;
static int g_m5_cpu_all=-1;
static int g_m5_cpu_miss=-2; /* 0=Metal, 1=CPU, -1=auto */
static int g_m5_cpu_miss_max_experts=2;
static int g_m5_cpu_miss_max_rows=4;
static int g_m5_cpu_stats=-1;
static uint64_t g_m5_grouped_calls, g_m5_grouped_experts, g_m5_grouped_rows;
static uint64_t g_m5_hetero_calls, g_m5_pair_rows, g_m5_scatter_rows;
static double g_m5_grouped_sec, g_m5_scatter_sec;

static int m5_env_bool(const char *name,int defv){
    const char *v=ili_env(name); if(!v||!*v) return defv;
    return strcmp(v,"0")!=0 && strcasecmp(v,"false")!=0 && strcasecmp(v,"off")!=0;
}
static int m5_env_int(const char *name,int defv){
    const char *v=ili_env(name); if(!v||!*v) return defv;
    char *e=NULL; long x=strtol(v,&e,10); return e!=v && *e==0 ? (int)x : defv;
}
static void m5_cpu_moe_report(void){
    if(g_m5_cpu_stats!=1) return;
    fprintf(stderr,"[m5-cpu-moe] grouped=%llu experts=%llu rows=%llu pair_rows=%llu hetero=%llu compute=%.6fs scatter=%.6fs\n",
        (unsigned long long)g_m5_grouped_calls,
        (unsigned long long)g_m5_grouped_experts,
        (unsigned long long)g_m5_grouped_rows,
        (unsigned long long)g_m5_pair_rows,
        (unsigned long long)g_m5_hetero_calls,
        g_m5_grouped_sec,g_m5_scatter_sec);
}
static void m5_cpu_moe_config(void){
    if(g_m5_cpu_grouped>=0) return;
    g_m5_cpu_grouped=m5_env_bool("CPU_GROUPED_MOE",1);
    g_m5_cpu_all=m5_env_bool("CPU_MOE_ALL",0);
    const char *mv=ili_env("CPU_MOE_MISSES");
    if(!mv||!*mv||!strcmp(mv,"0")||!strcasecmp(mv,"metal")) g_m5_cpu_miss=0;
    else if(!strcasecmp(mv,"auto")||!strcmp(mv,"-1")) g_m5_cpu_miss=-1;
    else g_m5_cpu_miss=1;
    g_m5_cpu_miss_max_experts=m5_env_int("CPU_MOE_MISS_MAX_EXPERTS",2);
    g_m5_cpu_miss_max_rows=m5_env_int("CPU_MOE_MISS_MAX_ROWS",4);
    g_m5_cpu_stats=m5_env_bool("CPU_MOE_STATS",0);
    atexit(m5_cpu_moe_report);
}
static int m5_cpu_all_experts(void){ m5_cpu_moe_config(); return g_m5_cpu_all; }
static int m5_cpu_misses(int resident_async,int ne,int nr){
    m5_cpu_moe_config();
    if(g_m5_cpu_miss==0) return 0;
    if(g_m5_cpu_miss==1) return 1;
    return resident_async && ne<=g_m5_cpu_miss_max_experts && nr<=g_m5_cpu_miss_max_rows;
}

static void m5_grouped_scratch(size_t xn,size_t sn,int8_t **xq,float **sx){
    if(xn>g_m5_grouped_scratch.xq_cap){
        int8_t *p=(int8_t*)realloc(g_m5_grouped_scratch.xq,xn);
        if(!p){ fprintf(stderr,"OOM grouped CPU MoE quant scratch\n"); exit(1); }
        g_m5_grouped_scratch.xq=p; g_m5_grouped_scratch.xq_cap=xn;
    }
    if(sn>g_m5_grouped_scratch.sx_cap){
        float *p=(float*)realloc(g_m5_grouped_scratch.sx,sn*sizeof(float));
        if(!p){ fprintf(stderr,"OOM grouped CPU MoE quant scales\n"); exit(1); }
        g_m5_grouped_scratch.sx=p; g_m5_grouped_scratch.sx_cap=sn;
    }
    *xq=g_m5_grouped_scratch.xq; *sx=g_m5_grouped_scratch.sx;
}

/* Gate and up share the same quantized activation. Apple SDOT loads x once and
 * advances two independent accumulator chains; other architectures reuse the
 * already exact scalar/SIMD dot kernels. */
static inline void m5_dot_i8_pair(const int8_t *wg,const int8_t *wu,const int8_t *x,int I,
                                  int32_t *og,int32_t *ou){
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int i=0; int32x4_t ag=vdupq_n_s32(0), au=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){ int8x16_t xv=vld1q_s8(x+i);
        ag=vdotq_s32(ag,vld1q_s8(wg+i),xv); au=vdotq_s32(au,vld1q_s8(wu+i),xv); }
    int32_t sg=vaddvq_s32(ag), su=vaddvq_s32(au);
    for(;i<I;i++){ sg+=(int32_t)wg[i]*x[i]; su+=(int32_t)wu[i]*x[i]; }
    *og=sg; *ou=su;
#else
    *og=dot_i8i8(wg,x,I); *ou=dot_i8i8(wu,x,I);
#endif
}
static inline void m5_dot_i4_pair(const uint8_t *wg,const uint8_t *wu,const int8_t *x,int I,
                                  int32_t *og,int32_t *ou){
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int i=0; const uint8x16_t mask=vdupq_n_u8(0x0f); const int8x16_t bias=vdupq_n_s8(8);
    int32x4_t ag=vdupq_n_s32(0), au=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t bg=vld1q_u8(wg+(i>>1)), bu=vld1q_u8(wu+(i>>1));
        uint8x16x2_t zg=vzipq_u8(vandq_u8(bg,mask),vshrq_n_u8(bg,4));
        uint8x16x2_t zu=vzipq_u8(vandq_u8(bu,mask),vshrq_n_u8(bu,4));
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
        ag=vdotq_s32(ag,vsubq_s8(vreinterpretq_s8_u8(zg.val[0]),bias),x0);
        ag=vdotq_s32(ag,vsubq_s8(vreinterpretq_s8_u8(zg.val[1]),bias),x1);
        au=vdotq_s32(au,vsubq_s8(vreinterpretq_s8_u8(zu.val[0]),bias),x0);
        au=vdotq_s32(au,vsubq_s8(vreinterpretq_s8_u8(zu.val[1]),bias),x1);
    }
    int32_t sg=vaddvq_s32(ag), su=vaddvq_s32(au);
    for(;i+1<I;i+=2){ uint8_t bg=wg[i>>1], bu=wu[i>>1];
        sg+=((int)(bg&15)-8)*x[i]+((int)(bg>>4)-8)*x[i+1];
        su+=((int)(bu&15)-8)*x[i]+((int)(bu>>4)-8)*x[i+1]; }
    if(i<I){ sg+=((int)(wg[i>>1]&15)-8)*x[i]; su+=((int)(wu[i>>1]&15)-8)*x[i]; }
    *og=sg; *ou=su;
#else
    *og=dot_i4i8(wg,x,I); *ou=dot_i4i8(wu,x,I);
#endif
}

static int m5_grouped_supported(const M5CpuExpert *ce,int ne){
    m5_cpu_moe_config();
    if(!g_m5_cpu_grouped || !g_idot || omp_in_parallel()) return 0;
#ifdef ILI_CUDA
    if(g_cuda_enabled) return 0;
#endif
    for(int e=0;e<ne;e++){
        ESlot *s=ce[e].slot; int f=s->g.fmt;
        if((f!=1&&f!=2)||s->u.fmt!=f||s->d.fmt!=f) return 0;
        if(s->g.I!=s->u.I||s->g.O!=s->u.O||s->d.I!=s->g.O||s->d.O!=s->g.I) return 0;
        if(f==2 && ce[e].nr<g_i4s) return 0;
    }
    return ne>0;
}

static void m5_grouped_compute(const M5CpuExpert *ce,int ne,int R,int D,int I,
                               const float *xg,float *gg,float *uu,float *hh){
    int maxdim=D>I?D:I; int8_t *xq; float *sx;
    m5_grouped_scratch((size_t)R*(size_t)maxdim,(size_t)R,&xq,&sx);
    const int tile=32, nti=(I+tile-1)/tile, ntd=(D+tile-1)/tile;
    double t0=now_s();
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(int r=0;r<R;r++) sx[r]=qrow_i8(xg+(int64_t)r*D,xq+(int64_t)r*D,D);
#pragma omp for schedule(dynamic,1)
        for(int task=0;task<ne*nti;task++){
            int ei=task/nti, o0=(task%nti)*tile, o1=o0+tile<I?o0+tile:I;
            ESlot *s=ce[ei].slot; int off=ce[ei].off, nr=ce[ei].nr, f=s->g.fmt;
            for(int o=o0;o<o1;o++) for(int r=0;r<nr;r++){
                int rr=off+r; int32_t dg,du;
                if(f==1) m5_dot_i8_pair(s->g.q8+(int64_t)o*D,s->u.q8+(int64_t)o*D,
                                         xq+(int64_t)rr*D,D,&dg,&du);
                else { int rb=(D+1)/2;
                    m5_dot_i4_pair(s->g.q4+(int64_t)o*rb,s->u.q4+(int64_t)o*rb,
                                   xq+(int64_t)rr*D,D,&dg,&du); }
                gg[(int64_t)rr*I+o]=(float)dg*s->g.s[o]*sx[rr];
                uu[(int64_t)rr*I+o]=(float)du*s->u.s[o]*sx[rr];
            }
        }
#pragma omp for schedule(static)
        for(int64_t z=0;z<(int64_t)R*I;z++) gg[z]=siluf(gg[z])*uu[z];
#pragma omp for schedule(static)
        for(int r=0;r<R;r++) sx[r]=qrow_i8(gg+(int64_t)r*I,xq+(int64_t)r*I,I);
#pragma omp for schedule(dynamic,1)
        for(int task=0;task<ne*ntd;task++){
            int ei=task/ntd, o0=(task%ntd)*tile, o1=o0+tile<D?o0+tile:D;
            ESlot *s=ce[ei].slot; int off=ce[ei].off, nr=ce[ei].nr, f=s->d.fmt;
            for(int o=o0;o<o1;o++) for(int r=0;r<nr;r++){
                int rr=off+r; int32_t d;
                if(f==1) d=dot_i8i8(s->d.q8+(int64_t)o*I,xq+(int64_t)rr*I,I);
                else d=dot_i4i8(s->d.q4+(int64_t)o*((I+1)/2),xq+(int64_t)rr*I,I);
                hh[(int64_t)rr*D+o]=(float)d*s->d.s[o]*sx[rr];
            }
        }
    }
    g_m5_grouped_calls++; g_m5_grouped_experts+=(uint64_t)ne;
    g_m5_grouped_rows+=(uint64_t)R; g_m5_pair_rows+=(uint64_t)R;
    g_m5_grouped_sec+=now_s()-t0;
}

static void m5_scatter(const M5CpuExpert *ce,int ne,int D,const int *rows,const float *rw,
                       const float *hh,float *out){
    double t0=now_s();
    for(int e=0;e<ne;e++) for(int r=0;r<ce[e].nr;r++){
        int rr=ce[e].off+r, token=rows[rr]; float w=rw[rr];
        float *dst=out+(int64_t)token*D; const float *src=hh+(int64_t)rr*D; int d=0;
#if defined(__ARM_NEON)
        float32x4_t ww=vdupq_n_f32(w);
        for(;d+16<=D;d+=16){
            vst1q_f32(dst+d,   vfmaq_f32(vld1q_f32(dst+d),   vld1q_f32(src+d),   ww));
            vst1q_f32(dst+d+4, vfmaq_f32(vld1q_f32(dst+d+4), vld1q_f32(src+d+4), ww));
            vst1q_f32(dst+d+8, vfmaq_f32(vld1q_f32(dst+d+8), vld1q_f32(src+d+8), ww));
            vst1q_f32(dst+d+12,vfmaq_f32(vld1q_f32(dst+d+12),vld1q_f32(src+d+12),ww));
        }
#endif
        for(;d<D;d++) dst[d]+=w*src[d];
    }
    g_m5_scatter_rows+=(uint64_t)(ne?ce[ne-1].off+ce[ne-1].nr:0);
    g_m5_scatter_sec+=now_s()-t0;
}

static int m5_cpu_moe_subset(ESlot **use,const int *eids,int nb,const int *handled,
                             const float *x,int S,int D,int I,int K,
                             const int *idxs,const float *route_w,const int *keff,float *out,
                             float *xg,float *gg,float *uu,float *hh,int *rows,float *rw,
                             int do_scatter){
    M5CpuExpert ce[64]; int ne=0,R=0;
    for(int j=0;j<nb;j++){
        if(handled && handled[j]) continue;
        int eid=eids[j], off=R, nr=0;
        for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++) if(idxs[(int64_t)s*K+k]==eid){
            rows[R]=s; rw[R]=route_w[(int64_t)s*K+k];
            memcpy(xg+(int64_t)R*D,x+(int64_t)s*D,(size_t)D*sizeof(float));
            R++; nr++; break;
        }
        if(nr){ ce[ne].slot=use[j]; ce[ne].off=off; ce[ne].nr=nr; ne++; }
    }
    if(!ne) return 0;
    if(m5_grouped_supported(ce,ne)) m5_grouped_compute(ce,ne,R,D,I,xg,gg,uu,hh);
    else for(int e=0;e<ne;e++){
        int off=ce[e].off,nr=ce[e].nr; ESlot *s=ce[e].slot;
        matmul_qt(gg+(int64_t)off*I,xg+(int64_t)off*D,&s->g,nr);
        matmul_qt(uu+(int64_t)off*I,xg+(int64_t)off*D,&s->u,nr);
        for(int64_t z=(int64_t)off*I;z<(int64_t)(off+nr)*I;z++) gg[z]=siluf(gg[z])*uu[z];
        matmul_qt(hh+(int64_t)off*D,gg+(int64_t)off*I,&s->d,nr);
    }
    if(do_scatter) m5_scatter(ce,ne,D,rows,rw,hh,out);
    return R;
}

'''
    text = replace_once(text, marker, grouped + marker, "grouped CPU MoE insertion")

    # Allow an all-CPU expert benchmark without disabling Metal attention/dense work.
    text = replace_once(
        text,
        '''        if(g_metal_enabled){
            for(int q=0;q<nmiss;q++) is_miss[missk[q]]=1;
''',
        '''        if(g_metal_enabled && !m5_cpu_all_experts()){
            for(int q=0;q<nmiss;q++) is_miss[missk[q]]=1;
''',
        "all-CPU expert gate",
    )

    # NOTE: the earlier "compute missed experts on CPU while the resident Metal block is
    # still in flight" heuristic (m5_defer_resident/m5_cpu_misses) does not compose with the
    # mixed-format guard's per-format sub-block partitioning (glm.c can now hold up to 4
    # concurrent resident handles, one per format, instead of one) and is dropped here: every
    # resident handle is ended, and every miss sub-block submitted, directly in glm.c before
    # this file's transform even runs. What remains for the M5 variant is exactly the grouped/
    # batched CPU compute below, driven by glm.c's own per-expert `handled[]` mask.
    old_cpu_loop = '''        if(g_a2_on) g_a2_cs=now_s();
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s();
                m->t_stall_exposed += pipe_wait_timed(qof[j],&m->n_pipe_waits,&m->n_pipe_waits_blocked);
                m->t_edisk += now_s()-tw; }
#ifdef ILI_METAL
            /* skip experts already computed on GPU by one of the per-format sub-blocks above */
            if(g_metal_enabled && handled[j]) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef ILI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
            matmul_qt(gg, xg, &e->g, nr);
            matmul_qt(uu, xg, &e->u, nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
        }
'''
    new_cpu_loop = '''        /* Drain every dispatched miss before any CPU consumer or LRU promotion. */
        if(g_pipe) for(int j=0;j<nb;j++) if(qof[j]>=0){
            double tw=now_s(); pipe_wait(qof[j]); m->t_edisk += now_s()-tw;
        }
        {
            const int *m5_mask=NULL;
#ifdef ILI_METAL
            if(g_metal_enabled && !m5_cpu_all_experts()) m5_mask=handled;
#endif
            double t0=now_s();
            m5_cpu_moe_subset(use,uniq+base,nb,m5_mask,
                              x,S,D,I,K,idxs,ws,keff,out,xg,gg,uu,hh,rows,rw,1);
            m->t_emm += now_s()-t0;
        }
'''
    text = replace_once(text, old_cpu_loop, new_cpu_loop, "grouped CPU fallback replacement")

    args.output.write_text(text)


if __name__ == "__main__":
    main()
