/* Numerical tests for the generated ragged grouped CPU MoE path.
 *
 * The test includes glm_m5max.c so it exercises the exact generated static
 * microkernels. It compares grouped gate/up/down output against the original
 * per-expert matmul_qt sequence for int8 and packed int4 weights, validates
 * deterministic scatter, and tests the complete routed-subset gather/reduction.
 */
#define main iliria_glm_main_unused
#include "../glm_m5max.c"
#undef main

static uint32_t rng_state=0x9e3779b9u;
static uint32_t xr(void){
    rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5;
    return rng_state;
}
static float frand1(void){ return ((int)(xr()%20001)-10000)/4096.0f; }

static void init_qt(QT *t,int fmt,int O,int I){
    memset(t,0,sizeof(*t)); t->fmt=fmt; t->O=O; t->I=I;
    t->s=(float*)malloc((size_t)O*sizeof(float));
    for(int o=0;o<O;o++) t->s[o]=0.0025f+(float)(1+(o%17))*0.0003f;
    if(fmt==1){
        t->q8=(int8_t*)malloc((size_t)O*I);
        for(int64_t z=0;z<(int64_t)O*I;z++) t->q8[z]=(int8_t)((int)(xr()%255)-127);
    } else {
        int rb=(I+1)/2; t->q4=(uint8_t*)malloc((size_t)O*rb);
        for(int64_t z=0;z<(int64_t)O*rb;z++) t->q4[z]=(uint8_t)xr();
    }
}
static void free_qt_test(QT *t){
    if(t->fmt==1) free(t->q8); else free(t->q4); free(t->s); memset(t,0,sizeof(*t));
}

static int closef_rel(float a,float b,float tol){
    float d=fabsf(a-b), s=fmaxf(1.0f,fmaxf(fabsf(a),fabsf(b)));
    return d<=tol*s;
}

static int test_pair_dots(void){
    enum { N=257 };
    int8_t x[N],a8[N],b8[N]; uint8_t a4[(N+1)/2],b4[(N+1)/2];
    for(int i=0;i<N;i++){ x[i]=(int8_t)((int)(xr()%255)-127); a8[i]=(int8_t)xr(); b8[i]=(int8_t)xr(); }
    for(int i=0;i<(N+1)/2;i++){ a4[i]=(uint8_t)xr(); b4[i]=(uint8_t)xr(); }
    int32_t ga,ua;
    m5_dot_i8_pair(a8,b8,x,N,&ga,&ua);
    if(ga!=dot_i8i8(a8,x,N)||ua!=dot_i8i8(b8,x,N)){ fprintf(stderr,"FAIL int8 pair dot\n"); return 0; }
    m5_dot_i4_pair(a4,b4,x,N,&ga,&ua);
    if(ga!=dot_i4i8(a4,x,N)||ua!=dot_i4i8(b4,x,N)){ fprintf(stderr,"FAIL int4 pair dot\n"); return 0; }
    return 1;
}

static int test_grouped(int fmt){
    enum { NE=3, D=33, I=17, R=6 };
    ESlot slot[NE]; M5CpuExpert ce[NE]; int counts[NE]={2,1,3};
    int off=0;
    for(int e=0;e<NE;e++){
        memset(&slot[e],0,sizeof(slot[e]));
        init_qt(&slot[e].g,fmt,I,D); init_qt(&slot[e].u,fmt,I,D); init_qt(&slot[e].d,fmt,D,I);
        ce[e].slot=&slot[e]; ce[e].eid=e; ce[e].off=off; ce[e].nr=counts[e]; off+=counts[e];
    }
    float *x=(float*)malloc((size_t)R*D*sizeof(float));
    float *gg=(float*)malloc((size_t)R*I*sizeof(float));
    float *uu=(float*)malloc((size_t)R*I*sizeof(float));
    float *got=(float*)malloc((size_t)R*D*sizeof(float));
    float *rg=(float*)malloc((size_t)R*I*sizeof(float));
    float *ru=(float*)malloc((size_t)R*I*sizeof(float));
    float *want=(float*)malloc((size_t)R*D*sizeof(float));
    for(int64_t z=0;z<(int64_t)R*D;z++) x[z]=frand1();

    g_idot=1; g_i4s=1; g_m5_cpu_grouped=1;
    m5_grouped_compute(ce,NE,R,D,I,x,gg,uu,got);
    for(int e=0;e<NE;e++){
        int o=ce[e].off,n=ce[e].nr;
        matmul_qt(rg+(int64_t)o*I,x+(int64_t)o*D,&slot[e].g,n);
        matmul_qt(ru+(int64_t)o*I,x+(int64_t)o*D,&slot[e].u,n);
        for(int64_t z=(int64_t)o*I;z<(int64_t)(o+n)*I;z++) rg[z]=siluf(rg[z])*ru[z];
        matmul_qt(want+(int64_t)o*D,rg+(int64_t)o*I,&slot[e].d,n);
    }
    for(int64_t z=0;z<(int64_t)R*D;z++) if(!closef_rel(got[z],want[z],1e-6f)){
        fprintf(stderr,"FAIL grouped fmt=%d at %lld: %.9g != %.9g\n",fmt,(long long)z,got[z],want[z]);
        return 0;
    }

    int rows[R]={0,1,1,0,1,0}; float rw[R]={0.1f,0.2f,0.3f,0.4f,0.5f,0.6f};
    float out0[2*D],out1[2*D];
    for(int i=0;i<2*D;i++) out0[i]=out1[i]=frand1();
    for(int e=0;e<NE;e++) for(int r=0;r<ce[e].nr;r++){
        int rr=ce[e].off+r; float *dst=out0+(int64_t)rows[rr]*D; const float *src=got+(int64_t)rr*D;
        for(int d=0;d<D;d++) dst[d]+=rw[rr]*src[d];
    }
    m5_scatter(ce,NE,D,rows,rw,got,out1);
    for(int i=0;i<2*D;i++) if(!closef_rel(out0[i],out1[i],2e-6f)){
        fprintf(stderr,"FAIL scatter fmt=%d at %d: %.9g != %.9g\n",fmt,i,out1[i],out0[i]); return 0;
    }

    for(int e=0;e<NE;e++){ free_qt_test(&slot[e].g); free_qt_test(&slot[e].u); free_qt_test(&slot[e].d); }
    free(x); free(gg); free(uu); free(got); free(rg); free(ru); free(want);
    return 1;
}

static int test_routed_subset(int fmt){
    enum { NE=3, S=3, K=2, D=35, I=19 };
    const int eids[NE]={2,5,9};
    const int idxs[S*K]={2,5, 5,9, 2,9};
    const int keff[S]={2,2,2};
    const float route[S*K]={0.15f,0.35f, 0.45f,0.55f, 0.65f,0.75f};
    ESlot slot[NE]; ESlot *use[NE];
    for(int e=0;e<NE;e++){
        memset(&slot[e],0,sizeof(slot[e]));
        init_qt(&slot[e].g,fmt,I,D); init_qt(&slot[e].u,fmt,I,D); init_qt(&slot[e].d,fmt,D,I);
        use[e]=&slot[e];
    }
    float x[S*D],got[S*D],want[S*D];
    for(int i=0;i<S*D;i++){ x[i]=frand1(); got[i]=want[i]=frand1(); }

    int n=m5_cpu_moe_subset(use,eids,NE,NULL,x,S,D,I,K,idxs,route,keff,got);
    if(n!=S*K){ fprintf(stderr,"FAIL subset routed-row count: %d != %d\n",n,S*K); return 0; }

    float gx[S*D],gg[S*I],uu[S*I],hh[S*D];
    for(int e=0;e<NE;e++){
        int nr=0,token[S]; float weight[S];
        for(int s=0;s<S;s++) for(int k=0;k<K;k++) if(idxs[s*K+k]==eids[e]){
            token[nr]=s; weight[nr]=route[s*K+k];
            memcpy(gx+(int64_t)nr*D,x+(int64_t)s*D,D*sizeof(float)); nr++; break;
        }
        matmul_qt(gg,gx,&slot[e].g,nr); matmul_qt(uu,gx,&slot[e].u,nr);
        for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
        matmul_qt(hh,gg,&slot[e].d,nr);
        for(int r=0;r<nr;r++) for(int d=0;d<D;d++)
            want[(int64_t)token[r]*D+d]+=weight[r]*hh[(int64_t)r*D+d];
    }
    for(int i=0;i<S*D;i++) if(!closef_rel(got[i],want[i],2e-6f)){
        fprintf(stderr,"FAIL routed subset fmt=%d at %d: %.9g != %.9g\n",fmt,i,got[i],want[i]); return 0;
    }
    if(!g_m5_grouped_scratch.xg || g_m5_grouped_scratch.xg_cap<(size_t)(S*K)*D*sizeof(float)){
        fprintf(stderr,"FAIL lazy grouped workspace was not sized from actual routed rows\n"); return 0;
    }

    /* handled[] mask: excluding expert index 1 (eid=5) must skip exactly its routed rows,
     * mirroring how glm.c's moe() now marks experts already computed by a per-format Metal
     * sub-block so this CPU path only fills in what Metal did NOT handle. */
    int handled_mask[NE]={0,1,0};
    float got2[S*D],want2[S*D];
    for(int i=0;i<S*D;i++){ got2[i]=want2[i]=frand1(); }
    int n2=m5_cpu_moe_subset(use,eids,NE,handled_mask,x,S,D,I,K,idxs,route,keff,got2);
    int want_rows=0;
    for(int e=0;e<NE;e++) if(!handled_mask[e])
        for(int s=0;s<S;s++) for(int k=0;k<K;k++) if(idxs[s*K+k]==eids[e]) want_rows++;
    if(n2!=want_rows){ fprintf(stderr,"FAIL masked subset row count: %d != %d\n",n2,want_rows); return 0; }
    for(int e=0;e<NE;e++){
        if(handled_mask[e]) continue;
        int nr=0,token[S]; float weight[S];
        for(int s=0;s<S;s++) for(int k=0;k<K;k++) if(idxs[s*K+k]==eids[e]){
            token[nr]=s; weight[nr]=route[s*K+k];
            memcpy(gx+(int64_t)nr*D,x+(int64_t)s*D,D*sizeof(float)); nr++; break;
        }
        matmul_qt(gg,gx,&slot[e].g,nr); matmul_qt(uu,gx,&slot[e].u,nr);
        for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
        matmul_qt(hh,gg,&slot[e].d,nr);
        for(int r=0;r<nr;r++) for(int d=0;d<D;d++)
            want2[(int64_t)token[r]*D+d]+=weight[r]*hh[(int64_t)r*D+d];
    }
    for(int i=0;i<S*D;i++) if(!closef_rel(got2[i],want2[i],2e-6f)){
        fprintf(stderr,"FAIL masked subset fmt=%d at %d: %.9g != %.9g\n",fmt,i,got2[i],want2[i]); return 0;
    }

    for(int e=0;e<NE;e++){ free_qt_test(&slot[e].g); free_qt_test(&slot[e].u); free_qt_test(&slot[e].d); }
    return 1;
}

int main(void){
    if(!test_pair_dots()) return 1;
    if(!test_grouped(1)) return 1;
    if(!test_grouped(2)) return 1;
    if(!test_routed_subset(1)) return 1;
    if(!test_routed_subset(2)) return 1;
    printf("M5 ragged grouped CPU MoE exactness (%s): ok\n",IDOT_KERNEL);
    return 0;
}
