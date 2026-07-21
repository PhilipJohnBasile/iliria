/* sim_expert_io.h -- simulated expert_load(): shard sizing, format inference,
 * byte accounting, and fault injection for the fuzz suite.
 *
 * This models c/st.h (st_read_raw/st_has/st_find) + c/glm.c's expert_load()
 * (~L1165-1325) and qt_from_disk() (~L849-866): each expert is 3 weight
 * tensors (gate_proj, up_proj, down_proj) + 3 matching ".qs" per-row f32
 * scale tensors, six independent on-disk reads that expert_load turns into
 * either one coalesced pread (weights, when contiguous in the file -- the
 * `contig` fast path, glm.c ~L1280-1299) or three separate ones, plus always
 * three separate small preads for the scales.
 *
 * ---- Byte size model (GLM-5.2 int4 shape) ----
 * D=hidden=6144, I=moe_intermediate_size=2048 -- read directly off the exact
 * dimension-guard on the Metal fused decode fast path (glm.c ~L2230:
 * "D==6144 && ... c->n_experts==256 && c->topk==8 ... c->moe_inter==2048").
 * Per-tensor byte counts follow qt_bytes()'s fmt==2 (int4-packed) formula
 * (glm.c ~L109-123): weight = O*ceil(I/2), scale = O*4 (one f32/row):
 *   gate_proj: O=I(moe)=2048, I=D=6144 -> weight 2048*3072=6,291,456  scale 2048*4=8,192
 *   up_proj:   same shape as gate_proj -> weight 6,291,456           scale 8,192
 *   down_proj: O=D=6144, I=I(moe)=2048 -> weight 6144*1024=6,291,456 scale 6144*4=24,576
 *   TOTAL = 18,915,328 bytes/expert
 * CROSS-CHECKED (not just derived): glm.c's own a2-overlap falsifier hardcodes
 * this exact number for this exact shape at ~L1962:
 *   g_a2_bytes=(int64_t)g_a2_nmiss*18915328
 * test_expert_stream_sim.c asserts expert_shard_total_bytes() against this
 * literal, tying our byte model to the engine's own constant.
 *
 * ---- Format inference: a REAL, VERIFIED engine gap, reproduced faithfully ----
 * glm.c infers a tensor's quant packing from its DECLARED byte count alone
 * (never from what pread() actually returns), via the SAME unchecked
 * ternary at three call sites -- qt_from_disk ~L857, the mmap fast path
 * ~L1206, and the main path ~L1319:
 *   int fmt = (nb==(int64_t)O*I) ? 1                       // int8
 *           : (nb==(int64_t)O*((I+1)/2)) ? 2               // int4
 *           : 3;                                           // else: ALWAYS int2,
 *                                                           //   even if nb doesn't
 *                                                           //   match int2's own
 *                                                           //   formula O*ceil(I/4)
 * A declared byte count that matches none of the three formulas is silently
 * treated as int2 packing -- no error, no exit, just a wrong dequant of
 * garbage. sim_format_infer_faithful() below reproduces this exact ternary
 * (including the unchecked fallthrough) so the simulator's "faithful" path
 * has the SAME blind spot as the real engine; sim_format_is_valid_for() is
 * an INDEPENDENT oracle (this simulator's own addition, not present in
 * glm.c) that the fuzz suite uses to DETECT when the faithful ternary landed
 * on a wrong format -- see ExpertLoadResult.format_finding and the
 * "wrong-size shard" fuzz cases in test_expert_stream_sim.c.
 *
 * ---- Corruption: glm.c has NO content checksum on expert payloads ----
 * Confirmed by reading expert_load() end to end: every failure path is a
 * byte-COUNT check (`pread(...) != nbytes`); there is no hash/CRC over the
 * weight or scale bytes anywhere in the load path. So same-size bit
 * corruption is, by construction, undetectable to the real engine. This
 * simulator does not invent a fake detector for the "faithful" path (that
 * would misrepresent the real engine); instead SIM_FAULT_CORRUPT sets
 * ExpertLoadResult.ok=1 (the "faithful" load succeeds, exactly as glm.c's
 * would) while ALSO setting .corrupted=1, a ground-truth flag ONLY visible
 * because this is a simulator that injected the fault itself and therefore
 * knows the answer a real deployment would not. The fuzz suite's assertion
 * is precisely "ok==1 && corrupted==1 was reached" -- i.e., "the faithful
 * path alone did not and could not catch this; only the test harness's
 * independent knowledge does" -- which is the honest way to satisfy this
 * the module's "never silently miscompute" contract for a fault class the real
 * engine, as read, cannot detect at all.
 *
 * ---- Consumer-path fatal contract ----
 * Every expert_load() call reachable from a synchronous decode-path miss
 * (the OMP-parallel-for blocking-load arm, glm.c ~L1958-1960; the PIPE
 * worker, ~L1385; REPIN, ~L3049) passes fatal=1: a short read or missing
 * tensor calls exit(1). Only the cross-layer speculative PILOT_REAL path
 * (~L2151, fatal=0) tolerates a failed load by abandoning it silently. This
 * simulator, to remain one long-running multi-scenario test process, cannot
 * literally call exit() on every expected fuzz failure -- ExpertLoadResult.ok
 * plus the caller's own "this was a critical-path miss" context is the
 * signal a real caller would act on by aborting; the test suite asserts
 * that signal fires correctly instead of ever invoking exit() itself.
 */
#ifndef SIM_EXPERT_IO_H
#define SIM_EXPERT_IO_H
#include <stdint.h>
#include <string.h>
#include "sim_nvme.h"

typedef struct { int64_t gate_w, gate_s, up_w, up_s, down_w, down_s; } ExpertShardSizes;

/* GLM-5.2 int4 shape -- see file header derivation. */
static ExpertShardSizes expert_shard_sizes_glm52_int4(void){
    ExpertShardSizes s;
    s.gate_w=6291456; s.gate_s=8192;
    s.up_w  =6291456; s.up_s  =8192;
    s.down_w=6291456; s.down_s=24576;
    return s;
}
static int64_t expert_shard_total_bytes(const ExpertShardSizes *s){
    return s->gate_w+s->gate_s+s->up_w+s->up_s+s->down_w+s->down_s;
}

/* (O,I) shape per weight tensor, GLM-5.2: hidden D=6144, moe_intermediate_size=2048
 * (glm.c ~L1171: "int I=c->moe_inter, D=c->hidden;" then "int OO[3]={I,I,D}, II[3]={D,D,I};"
 * for {gate,up,down} -- a small helper so call sites never have to re-derive/transpose
 * these six numbers by hand. */
typedef struct { int64_t O_gate,I_gate,O_up,I_up,O_down,I_down; } ExpertShape;
static ExpertShape expert_shape_glm52(void){
    ExpertShape s; int64_t D=6144, I=2048;
    s.O_gate=I; s.I_gate=D; s.O_up=I; s.I_up=D; s.O_down=D; s.I_down=I;
    return s;
}

/* ---- format inference: faithful ternary + independent oracle ---- */
static int sim_format_infer_faithful(int64_t nb, int64_t O, int64_t I){
    if(nb==O*I) return 1;
    if(nb==O*((I+1)/2)) return 2;
    return 3;   /* unchecked fallback -- see file header */
}
static int sim_format_is_valid_for(int fmt, int64_t nb, int64_t O, int64_t I){
    if(fmt==1) return nb==O*I;
    if(fmt==2) return nb==O*((I+1)/2);
    if(fmt==3) return nb==O*((I+3)/4);
    return 0;
}

/* ---- fault injection ---- */
typedef enum {
    SIM_TENSOR_GATE_W=0, SIM_TENSOR_GATE_S, SIM_TENSOR_UP_W, SIM_TENSOR_UP_S,
    SIM_TENSOR_DOWN_W, SIM_TENSOR_DOWN_S, SIM_TENSOR_ANY
} SimTensorWhich;

typedef enum {
    SIM_FAULT_NONE=0,
    SIM_FAULT_SHORT_READ,           /* param = bytes actually delivered (clamped to < declared) */
    SIM_FAULT_WRONG_DECLARED_SIZE,  /* param = bogus declared size a corrupted header would report;
                                      * the "device" faithfully delivers exactly that many bytes --
                                      * models a wrong-size SHARD, not a truncated read */
    SIM_FAULT_CORRUPT,              /* content-level; sizes unaffected; see file header */
    SIM_FAULT_EXTRA_LATENCY_US      /* param = microseconds of extra latency (straggler injection) */
} SimFaultKind;

typedef struct {
    int layer, eid; SimTensorWhich which;
    SimFaultKind kind; int64_t param;
    int hit_count;      /* bumped every time this rule fires; test suite asserts it was >0
                          * ("the fault was actually exercised", not a dead rule) */
} SimFaultRule;

typedef struct { SimFaultRule *rules; int n, cap; } SimFaultProgram;

static void sim_fault_init(SimFaultProgram *fp){ memset(fp,0,sizeof(*fp)); }
static void sim_fault_free(SimFaultProgram *fp){ free(fp->rules); memset(fp,0,sizeof(*fp)); }
static void sim_fault_add(SimFaultProgram *fp, int layer, int eid, SimTensorWhich which,
                           SimFaultKind kind, int64_t param){
    if(fp->n==fp->cap){ fp->cap = fp->cap?fp->cap*2:8; fp->rules=realloc(fp->rules,(size_t)fp->cap*sizeof(SimFaultRule)); }
    SimFaultRule *r=&fp->rules[fp->n++];
    r->layer=layer; r->eid=eid; r->which=which; r->kind=kind; r->param=param; r->hit_count=0;
}
static SimFaultRule* sim_fault_match(SimFaultProgram *fp, int layer, int eid, SimTensorWhich which){
    for(int i=0;i<fp->n;i++){
        SimFaultRule *r=&fp->rules[i];
        if(r->layer==layer && r->eid==eid && (r->which==which || r->which==SIM_TENSOR_ANY)) return r;
    }
    return NULL;
}

/* ---- byte accounting: mirrors Model's io_bytes_requested/io_bytes_read/
 * io_reads_attempted/io_reads_completed (glm.c ~L189-190) exactly, including
 * the asymmetry that `requested` is bumped for every ATTEMPT (glm.c
 * ~L1194-1195, before the read is known to succeed) while `read`/`completed`
 * are bumped ONLY on full success (io_read_done(), glm.c ~L1147-1153,
 * called at the successful-return points of expert_load and nowhere else). */
typedef struct {
    int64_t  bytes_requested;
    int64_t  bytes_read;
    uint64_t reads_attempted;
    uint64_t reads_completed;
} IoByteCounters;

typedef struct {
    int ok;               /* 1 = fully successful load, mirrors expert_load's `return 0` */
    int short_read;       /* 1 if ANY tensor delivered fewer bytes than declared (detected, fatal
                            * on the consumer path -- see file header) */
    int format_finding;   /* 1 if the faithful ternary landed on a format that does NOT actually
                            * match the declared size for at least one weight tensor -- a
                            * silent-would-be-wrong outcome only the ORACLE catches */
    int corrupted;        /* 1 if a SIM_FAULT_CORRUPT fired -- ground truth only the fault
                            * program (not the "faithful" load path) knows about */
    int inferred_fmt[3];  /* gate/up/down, as the faithful ternary would infer: 1/2/3 */
    double t_complete;    /* virtual completion time of the whole (up to 6-issue) load */
    int64_t declared_bytes; /* == req_bytes in glm.c: sum of declared sizes for this attempt */
} ExpertLoadResult;

/* Simulates one expert_load() call for (layer,eid): six tensor reads against
 * `nvme` on worker `w`, issued at virtual time `t_issue`, with `faults`
 * consulted per-tensor. `coalesced` mirrors the `contig` fast path (glm.c
 * ~L1280-1299): when true, the three WEIGHT tensors are issued as ONE
 * combined NVMe request (1x base_latency instead of 3x, same total bytes
 * either way); the three .qs SCALE tensors are always issued separately,
 * exactly as upstream never coalesces those. The O_.../I_... parameters are
 * the (rows,cols) shape used for format inference (see
 * expert_shard_sizes_glm52_int4 for the default GLM-5.2 shape). */
static ExpertLoadResult sim_expert_load(NvmeSim *nvme, int w, double t_issue,
                                         IoByteCounters *io, SimFaultProgram *faults,
                                         int layer, int eid, const ExpertShardSizes *declared,
                                         int coalesced,
                                         int64_t O_gate,int64_t I_gate,
                                         int64_t O_up,  int64_t I_up,
                                         int64_t O_down,int64_t I_down){
    ExpertLoadResult r; memset(&r,0,sizeof(r));
    io->reads_attempted++;

    int64_t decl[6] = { declared->gate_w, declared->gate_s, declared->up_w,
                         declared->up_s,  declared->down_w, declared->down_s };
    int64_t delivered[6];
    double  extra_lat[6] = {0,0,0,0,0,0};
    static const SimTensorWhich which[6] = {
        SIM_TENSOR_GATE_W, SIM_TENSOR_GATE_S, SIM_TENSOR_UP_W,
        SIM_TENSOR_UP_S,   SIM_TENSOR_DOWN_W, SIM_TENSOR_DOWN_S
    };
    int any_short=0;
    for(int k=0;k<6;k++){
        delivered[k]=decl[k];
        SimFaultRule *fr = faults ? sim_fault_match(faults, layer, eid, which[k]) : NULL;
        if(fr){
            fr->hit_count++;
            switch(fr->kind){
                case SIM_FAULT_SHORT_READ:
                    delivered[k] = fr->param < decl[k] ? fr->param : decl[k];
                    if(delivered[k]<0) delivered[k]=0;
                    break;
                case SIM_FAULT_WRONG_DECLARED_SIZE:
                    decl[k]=fr->param; delivered[k]=fr->param;
                    break;
                case SIM_FAULT_EXTRA_LATENCY_US:
                    extra_lat[k]=(double)fr->param/1e6;
                    break;
                case SIM_FAULT_CORRUPT:
                    r.corrupted=1;
                    break;
                case SIM_FAULT_NONE: default: break;
            }
        }
        if(delivered[k] < decl[k]) any_short=1;
    }

    r.declared_bytes = decl[0]+decl[1]+decl[2]+decl[3]+decl[4]+decl[5];
    io->bytes_requested += r.declared_bytes;   /* eager: every attempt, success or not (glm.c ~L1195) */

    double last_complete = t_issue;
    if(coalesced){
        int64_t wdeliv = delivered[0]+delivered[2]+delivered[4];
        double  xl     = extra_lat[0]+extra_lat[2]+extra_lat[4];
        NvmeCompletion c = nvme_issue(nvme, w, t_issue, wdeliv, xl);
        if(c.t_complete>last_complete) last_complete=c.t_complete;
    } else {
        int widx[3]={0,2,4};
        for(int i=0;i<3;i++){
            int k=widx[i];
            NvmeCompletion c = nvme_issue(nvme, w, t_issue, delivered[k], extra_lat[k]);
            if(c.t_complete>last_complete) last_complete=c.t_complete;
        }
    }
    { int sidx[3]={1,3,5};
      for(int i=0;i<3;i++){
          int k=sidx[i];
          NvmeCompletion c = nvme_issue(nvme, w, t_issue, delivered[k], extra_lat[k]);
          if(c.t_complete>last_complete) last_complete=c.t_complete;
      }
    }
    r.t_complete = last_complete;
    r.short_read = any_short;

    /* Format inference runs against the DECLARED size, never the delivered
     * byte count -- glm.c infers dtype from the header's nbytes field, and
     * only separately (via the pread return value) checks for short reads.
     * These are two independent failure classes on purpose (see file header). */
    { int64_t OI[3][2] = { {O_gate,I_gate}, {O_up,I_up}, {O_down,I_down} };
      int64_t wdecl[3] = { decl[0], decl[2], decl[4] };
      for(int i=0;i<3;i++){
          int fmt = sim_format_infer_faithful(wdecl[i], OI[i][0], OI[i][1]);
          r.inferred_fmt[i]=fmt;
          if(!sim_format_is_valid_for(fmt, wdecl[i], OI[i][0], OI[i][1])) r.format_finding=1;
      }
    }

    if(any_short){
        r.ok=0;   /* mirrors `if(pread(...)!=nbytes){ perror(...); return -1 (or exit if fatal); }` */
        return r;
    }
    io->bytes_read += r.declared_bytes;
    io->reads_completed++;
    r.ok=1;
    return r;
}

#endif /* SIM_EXPERT_IO_H */
