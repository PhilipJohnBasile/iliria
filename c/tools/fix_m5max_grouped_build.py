#!/usr/bin/env python3
"""Final generated-source repairs for the M5 Max engine.

Besides declaration/escape repairs, this pass keeps the original per-token MoE
scratch small and gives grouped CPU execution a separate lazy arena sized from
the actual CPU-owned routed rows. Large Metal prefill therefore does not reserve
`S * topK` CPU expert workspaces that it never uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    a = text.find(start)
    if a < 0:
        raise RuntimeError(f"{label}: start marker not found")
    b = text.find(end, a)
    if b < 0:
        raise RuntimeError(f"{label}: end marker not found")
    if text.find(start, a + 1) >= 0:
        raise RuntimeError(f"{label}: start marker is not unique")
    return text[:a] + replacement + text[b:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()

    repairs = {
        'fprintf(stderr,"M5 attention scratch overflow\n");':
            'fprintf(stderr,"M5 attention scratch overflow\\n");',
        'fprintf(stderr,"OOM M5 attention scratch (%zu bytes)\n",cap);':
            'fprintf(stderr,"OOM M5 attention scratch (%zu bytes)\\n",cap);',
        'fprintf(stderr,"OOM M5 hot-path scratch (%zu bytes)\n",cap);':
            'fprintf(stderr,"OOM M5 hot-path scratch (%zu bytes)\\n",cap);',
    }
    for old, new in repairs.items():
        if old not in text:
            raise RuntimeError(f"generated diagnostic pattern not found: {old!r}")
        text = text.replace(old, new, 1)

    marker = '/* ---- M5 MAX RAGGED GROUPED CPU MoE -----------------------------------------\n'
    text = replace_once(
        text,
        marker,
        'static inline float siluf(float x);\n\n' + marker,
        'siluf forward declaration',
    )

    text = replace_once(
        text,
        '''    size_t m5_rmax=(size_t)S*(size_t)K;
    float *xg=(float*)m5_scratch_get(&ms->xg,m5_rmax*(size_t)D*sizeof(float));
    float *gg=(float*)m5_scratch_get(&ms->gg,m5_rmax*(size_t)I*sizeof(float));
    float *uu=(float*)m5_scratch_get(&ms->uu,m5_rmax*(size_t)I*sizeof(float));
    float *hh=(float*)m5_scratch_get(&ms->hh,m5_rmax*(size_t)D*sizeof(float));
    int *rows=(int*)m5_scratch_get(&ms->rows,m5_rmax*sizeof(int));
    float *rw=(float*)m5_scratch_get(&ms->rw,m5_rmax*sizeof(float));
''',
        '''    float *xg=(float*)m5_scratch_get(&ms->xg,(size_t)S*D*sizeof(float));
    float *gg=(float*)m5_scratch_get(&ms->gg,(size_t)S*I*sizeof(float));
    float *uu=(float*)m5_scratch_get(&ms->uu,(size_t)S*I*sizeof(float));
    float *hh=(float*)m5_scratch_get(&ms->hh,(size_t)S*D*sizeof(float));
    int *rows=(int*)m5_scratch_get(&ms->rows,(size_t)S*sizeof(int));
    float *rw=(float*)m5_scratch_get(&ms->rw,(size_t)S*sizeof(float));
''',
        'restore compact base MoE scratch',
    )

    text = replace_once(
        text,
        '''typedef struct {
    ESlot *slot;
    int off, nr;
} M5CpuExpert;

typedef struct {
    int8_t *xq;
    size_t xq_cap;
    float *sx;
    size_t sx_cap;
} M5GroupedScratch;
''',
        '''typedef struct {
    ESlot *slot;
    int eid, off, nr;
} M5CpuExpert;

typedef struct {
    int8_t *xq; float *sx;
    float *xg, *gg, *uu, *hh, *rw;
    int *rows;
    size_t xq_cap, sx_cap, xg_cap, gg_cap, uu_cap, hh_cap, rows_cap, rw_cap;
} M5GroupedScratch;
''',
        'extend lazy grouped scratch',
    )

    scratch_end = '''    *xq=g_m5_grouped_scratch.xq; *sx=g_m5_grouped_scratch.sx;
}

/* Gate and up share the same quantized activation.'''
    scratch_helpers = '''    *xq=g_m5_grouped_scratch.xq; *sx=g_m5_grouped_scratch.sx;
}
static void *m5_grouped_grow(void *p,size_t *cap,size_t bytes,const char *what){
    if(bytes==0) bytes=1;
    if(bytes>*cap){
        size_t n=*cap?*cap:4096;
        while(n<bytes){ if(n>SIZE_MAX/2){ n=bytes; break; } n*=2; }
        void *q=realloc(p,n);
        if(!q){ fprintf(stderr,"OOM grouped CPU MoE %s (%zu bytes)\\n",what,n); exit(1); }
        p=q; *cap=n;
    }
    return p;
}
static void m5_grouped_work(int R,int D,int I,float **xg,float **gg,float **uu,float **hh,
                            int **rows,float **rw){
    if(R<0||D<0||I<0 || (size_t)R>SIZE_MAX/(size_t)(D?D:1)
       || (size_t)R>SIZE_MAX/(size_t)(I?I:1)){
        fprintf(stderr,"grouped CPU MoE shape overflow\\n"); exit(1);
    }
    size_t rd=(size_t)R*(size_t)D, ri=(size_t)R*(size_t)I;
    if(rd>SIZE_MAX/sizeof(float)||ri>SIZE_MAX/sizeof(float)){
        fprintf(stderr,"grouped CPU MoE byte-size overflow\\n"); exit(1);
    }
    g_m5_grouped_scratch.xg=(float*)m5_grouped_grow(g_m5_grouped_scratch.xg,&g_m5_grouped_scratch.xg_cap,rd*sizeof(float),"inputs");
    g_m5_grouped_scratch.gg=(float*)m5_grouped_grow(g_m5_grouped_scratch.gg,&g_m5_grouped_scratch.gg_cap,ri*sizeof(float),"gate");
    g_m5_grouped_scratch.uu=(float*)m5_grouped_grow(g_m5_grouped_scratch.uu,&g_m5_grouped_scratch.uu_cap,ri*sizeof(float),"up");
    g_m5_grouped_scratch.hh=(float*)m5_grouped_grow(g_m5_grouped_scratch.hh,&g_m5_grouped_scratch.hh_cap,rd*sizeof(float),"output");
    g_m5_grouped_scratch.rows=(int*)m5_grouped_grow(g_m5_grouped_scratch.rows,&g_m5_grouped_scratch.rows_cap,(size_t)R*sizeof(int),"rows");
    g_m5_grouped_scratch.rw=(float*)m5_grouped_grow(g_m5_grouped_scratch.rw,&g_m5_grouped_scratch.rw_cap,(size_t)R*sizeof(float),"weights");
    *xg=g_m5_grouped_scratch.xg; *gg=g_m5_grouped_scratch.gg;
    *uu=g_m5_grouped_scratch.uu; *hh=g_m5_grouped_scratch.hh;
    *rows=g_m5_grouped_scratch.rows; *rw=g_m5_grouped_scratch.rw;
}

/* Gate and up share the same quantized activation.'''
    text = replace_once(text, scratch_end, scratch_helpers, 'lazy grouped workspace helpers')

    subset = '''static int m5_cpu_moe_subset(ESlot **use,const int *eids,int nb,const int *handled,
                             const float *x,int S,int D,int I,int K,
                             const int *idxs,const float *route_w,const int *keff,float *out){
    M5CpuExpert ce[64]; int ne=0,R=0;
    for(int j=0;j<nb;j++){
        if(handled && handled[j]) continue;
        int eid=eids[j], nr=0;
        for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++)
            if(idxs[(int64_t)s*K+k]==eid){ nr++; break; }
        if(nr){ ce[ne].slot=use[j]; ce[ne].eid=eid; ce[ne].off=R; ce[ne].nr=nr; R+=nr; ne++; }
    }
    if(!ne) return 0;

    float *xg,*gg,*uu,*hh,*rw; int *rows;
    m5_grouped_work(R,D,I,&xg,&gg,&uu,&hh,&rows,&rw);
    for(int e=0;e<ne;e++){
        int p=ce[e].off, end=p+ce[e].nr;
        for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++)
            if(idxs[(int64_t)s*K+k]==ce[e].eid){
                rows[p]=s; rw[p]=route_w[(int64_t)s*K+k];
                memcpy(xg+(int64_t)p*D,x+(int64_t)s*D,(size_t)D*sizeof(float));
                p++; break;
            }
        if(p!=end){ fprintf(stderr,"grouped CPU MoE route-count mismatch\\n"); exit(1); }
    }

    if(m5_grouped_supported(ce,ne)) m5_grouped_compute(ce,ne,R,D,I,xg,gg,uu,hh);
    else for(int e=0;e<ne;e++){
        int off=ce[e].off,nr=ce[e].nr; ESlot *s=ce[e].slot;
        matmul_qt(gg+(int64_t)off*I,xg+(int64_t)off*D,&s->g,nr);
        matmul_qt(uu+(int64_t)off*I,xg+(int64_t)off*D,&s->u,nr);
        for(int64_t z=(int64_t)off*I;z<(int64_t)(off+nr)*I;z++) gg[z]=siluf(gg[z])*uu[z];
        matmul_qt(hh+(int64_t)off*D,gg+(int64_t)off*I,&s->d,nr);
    }
    m5_scatter(ce,ne,D,rows,rw,hh,out);
    return R;
}
'''
    text = replace_between(
        text,
        'static int m5_cpu_moe_subset(',
        '\n\n/* quantizza w[O,I] f32',
        subset,
        'lazy grouped subset function',
    )

    text = replace_once(
        text,
        '''m5_cpu_moe_subset(use,uniq+base,nb,m5_mask,
                              x,S,D,I,K,idxs,ws,keff,out,xg,gg,uu,hh,rows,rw,1);''',
        '''m5_cpu_moe_subset(use,uniq+base,nb,m5_mask,
                              x,S,D,I,K,idxs,ws,keff,out);''',
        'main grouped subset call',
    )

    args.output.write_text(text)


if __name__ == '__main__':
    main()
