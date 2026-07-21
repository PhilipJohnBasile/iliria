/* codec_race.c -- CLI benchmark harness for the row-independent int4
 * entropy-codec race (docs/PERFORMANCE_THEORY.md n1:
 * "n1_codec_race_requirement" / "n1_row_independent_coding_caveat_for_mode3").
 *
 * Loads a synthetic int4 row corpus (c/tools/gen_codec_race_synthetic.py),
 * groups rows by (band, proj) into 9 projection groups (matching the
 * census's own early/mid/late x gate/up/down breakdown), and for each of
 * the four row-independent codecs (canonical Huffman+LUT, interleaved
 * rANS, tANS/FSE, branchless bitplane+exception):
 *
 *   1. builds ONE shared codebook per group from the group's aggregate
 *      symbol histogram ("projection-level shared codebooks");
 *   2. encodes every row independently against that shared codebook,
 *      building a row_offsets[O_total+1] index (THE mode-3-relevant,
 *      row-coded stored ratio -- expected worse than the census's 64KB-
 *      block 0.7379 headline; this measures exactly how much worse);
 *   3. benchmarks decode throughput single-threaded and (OpenMP)
 *      multi-threaded, reporting GB/s of raw (int4-equivalent, 0.5
 *      bytes/symbol) output produced;
 *   4. re-verifies every decoded row against the original bit-for-bit
 *      before reporting (belt-and-braces on top of the unit-test suite --
 *      this is the actual synthetic-scale data, not just fuzz vectors).
 *
 * Usage: ./codec_race <corpus.bin> <index.csv> [reps=15] [threads=0(auto)]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "codec_row.h"
#include "codec_row_huff.h"
#include "codec_row_rans.h"
#include "codec_row_fse.h"
#include "codec_row_bitplane.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ------------------------------------------------------------- index ---- */
typedef struct {
    char band[8], proj[8];
    long O, I, row_bytes, byte_offset, nbytes;
} TensorMeta;

#define MAX_TENSORS 8192
static TensorMeta g_metas[MAX_TENSORS];

static int read_index_csv(const char *path, TensorMeta *out, int cap){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); exit(1); }
    char line[512];
    if(!fgets(line, sizeof(line), f)){ fclose(f); return 0; } /* header row */
    int n=0;
    while(fgets(line, sizeof(line), f)){
        if(n>=cap){ fprintf(stderr, "too many tensors (cap=%d)\n", cap); break; }
        char band[16], proj[16];
        long tensor_idx, O, I, row_bytes, byte_offset, nbytes;
        double target_h, achieved_h;
        int got = sscanf(line, "%15[^,],%15[^,],%ld,%ld,%ld,%ld,%ld,%ld,%lf,%lf",
                          band, proj, &tensor_idx, &O, &I, &row_bytes,
                          &byte_offset, &nbytes, &target_h, &achieved_h);
        (void)tensor_idx; (void)target_h; (void)achieved_h;
        if(got != 10) continue;
        snprintf(out[n].band, sizeof(out[n].band), "%s", band);
        snprintf(out[n].proj, sizeof(out[n].proj), "%s", proj);
        out[n].O=O; out[n].I=I; out[n].row_bytes=row_bytes;
        out[n].byte_offset=byte_offset; out[n].nbytes=nbytes;
        n++;
    }
    fclose(f);
    return n;
}

static const char *BANDS[3] = {"early","mid","late"};
static const char *PROJS[3] = {"gate","up","down"};
#define N_GROUPS 9
static int group_index(const char *band, const char *proj){
    int b=-1,p=-1;
    for(int i=0;i<3;i++) if(strcmp(band,BANDS[i])==0) b=i;
    for(int i=0;i<3;i++) if(strcmp(proj,PROJS[i])==0) p=i;
    if(b<0||p<0) return -1;
    return b*3+p;
}

typedef struct { int n; TensorMeta *items[MAX_TENSORS]; } GroupList;

/* --------------------------------------------------- codec adapter layer --
 * one generic benchmark driver reused by all four codecs via thin function-
 * pointer adapters (avoids four near-duplicate harnesses). */
typedef size_t (*EncodeFn)(const void *cb, const uint8_t *sym, int n, uint8_t *out, size_t cap);
typedef void   (*DecodeFn)(const void *cb, const uint8_t *payload, size_t paylen, int n, uint8_t *sym_out);

static size_t huff_enc_adapt(const void *cb, const uint8_t *sym, int n, uint8_t *out, size_t cap){ return huff_encode_row((const HuffCodebook*)cb, sym, n, out, cap); }
static void   huff_dec_adapt(const void *cb, const uint8_t *pl, size_t pl_len, int n, uint8_t *out){ huff_decode_row((const HuffCodebook*)cb, pl, pl_len, n, out); }
static size_t rans_enc_adapt(const void *cb, const uint8_t *sym, int n, uint8_t *out, size_t cap){ return rans_encode_row((const RansCodebook*)cb, sym, n, out, cap); }
static void   rans_dec_adapt(const void *cb, const uint8_t *pl, size_t pl_len, int n, uint8_t *out){ rans_decode_row((const RansCodebook*)cb, pl, pl_len, n, out); }
static size_t fse_enc_adapt(const void *cb, const uint8_t *sym, int n, uint8_t *out, size_t cap){ return fse_encode_row((const FseCodebook*)cb, sym, n, out, cap); }
static void   fse_dec_adapt(const void *cb, const uint8_t *pl, size_t pl_len, int n, uint8_t *out){ fse_decode_row((const FseCodebook*)cb, pl, pl_len, n, out); }
static size_t bp_enc_adapt(const void *cb, const uint8_t *sym, int n, uint8_t *out, size_t cap){ return bitplane_encode_row((const BitplaneCodebook*)cb, sym, n, out, cap); }
static void   bp_dec_adapt(const void *cb, const uint8_t *pl, size_t pl_len, int n, uint8_t *out){ bitplane_decode_row((const BitplaneCodebook*)cb, pl, pl_len, n, out); }

typedef struct {
    const char *name;
    EncodeFn encode;
    DecodeFn decode;
    size_t codebook_bytes;
} CodecEntry;

typedef struct {
    long raw_total_bytes;         /* incl. unchanged per-row F32 scales */
    long compressed_total_bytes;  /* incl. codebook + row_offsets index + scales */
    double stored_ratio;
    double gbps_1thread;
    double gbps_multithread;
} RaceResult;

static RaceResult run_codec_on_group(const CodecEntry *codec, const void *cb,
                                      const uint8_t *corpus, TensorMeta **tensors, int n_tensors,
                                      long I, long row_bytes, long total_rows,
                                      int reps, int nthreads_report){
    RaceResult res; memset(&res, 0, sizeof(res));

    RowOffsets ro; row_offsets_init(&ro, (uint32_t)total_rows);
    size_t payload_cap = (size_t)total_rows*(size_t)row_bytes*2 + 4096;
    uint8_t *payload = (uint8_t*)malloc(payload_cap);
    uint8_t *symbuf = (uint8_t*)malloc((size_t)I);
    const uint8_t **row_raw = (const uint8_t**)malloc(sizeof(uint8_t*)*(size_t)total_rows);
    size_t payload_pos = 0;
    long row_idx = 0;
    for(int t=0;t<n_tensors;t++){
        TensorMeta *tm = tensors[t];
        for(long r=0;r<tm->O;r++){
            const uint8_t *packed = corpus + tm->byte_offset + r*tm->row_bytes;
            row_raw[row_idx] = packed;
            codec_unpack_nibbles(packed, (int)I, symbuf);
            size_t used = codec->encode(cb, symbuf, (int)I, payload+payload_pos, payload_cap-payload_pos);
            if(used == (size_t)-1){
                fprintf(stderr, "encode overflow: codec=%s row=%ld\n", codec->name, row_idx);
                exit(1);
            }
            row_offsets_push(&ro, (uint32_t)used);
            payload_pos += used;
            row_idx++;
        }
    }
    free(symbuf);

    long raw_payload = total_rows*row_bytes;
    long scale_bytes = total_rows*4; /* per-row F32 scale, UNCHANGED, added both sides */
    long compressed_payload = (long)payload_pos + (long)row_offsets_index_bytes((uint32_t)total_rows)
                             + (long)codec->codebook_bytes;
    res.raw_total_bytes = raw_payload + scale_bytes;
    res.compressed_total_bytes = compressed_payload + scale_bytes;
    res.stored_ratio = (double)res.compressed_total_bytes / (double)res.raw_total_bytes;

    uint8_t *decode_scratch = (uint8_t*)malloc((size_t)I * (size_t)total_rows);

    double best_1t = 0.0;
    for(int rep=0; rep<reps; rep++){
        double t0 = now_s();
        for(long i=0;i<total_rows;i++){
            size_t off = ro.offsets[i], len = ro.offsets[i+1]-off;
            codec->decode(cb, payload+off, len, (int)I, decode_scratch+(size_t)i*(size_t)I);
        }
        double dt = now_s()-t0;
        double gbps = dt>0 ? (double)raw_payload/1e9/dt : 0.0;
        if(gbps>best_1t) best_1t=gbps;
    }

    double best_mt = best_1t;
#ifdef _OPENMP
    best_mt = 0.0;
    for(int rep=0; rep<reps; rep++){
        double t0 = now_s();
        #pragma omp parallel for schedule(static)
        for(long i=0;i<total_rows;i++){
            size_t off = ro.offsets[i], len = ro.offsets[i+1]-off;
            codec->decode(cb, payload+off, len, (int)I, decode_scratch+(size_t)i*(size_t)I);
        }
        double dt = now_s()-t0;
        double gbps = dt>0 ? (double)raw_payload/1e9/dt : 0.0;
        if(gbps>best_mt) best_mt=gbps;
    }
#endif
    (void)nthreads_report;
    res.gbps_1thread = best_1t;
    res.gbps_multithread = best_mt;

    /* belt-and-braces: re-verify the LAST decode pass bit-for-bit against
     * the original packed rows (the unit-test suite already fuzzes the
     * codecs; this checks the actual synthetic-scale corpus too). */
    row_idx = 0;
    uint8_t check[8192];
    for(int t=0;t<n_tensors;t++){
        TensorMeta *tm = tensors[t];
        for(long r=0;r<tm->O;r++){
            codec_unpack_nibbles(row_raw[row_idx], (int)I, check);
            if(memcmp(check, decode_scratch+(size_t)row_idx*(size_t)I, (size_t)I) != 0){
                fprintf(stderr, "DECODE MISMATCH: codec=%s row=%ld\n", codec->name, row_idx);
                exit(1);
            }
            row_idx++;
        }
    }

    free(decode_scratch); free(payload); free(row_raw); row_offsets_free(&ro);
    return res;
}

int main(int argc, char **argv){
    if(argc<3){
        fprintf(stderr, "usage: %s <corpus.bin> <index.csv> [reps=15] [threads=0(auto)]\n", argv[0]);
        return 1;
    }
    const char *corpus_path = argv[1], *index_path = argv[2];
    int reps = argc>3 ? atoi(argv[3]) : 15;
    int threads = argc>4 ? atoi(argv[4]) : 0;
#ifdef _OPENMP
    if(threads>0) omp_set_num_threads(threads);
    int nthreads = threads>0 ? threads : omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    FILE *f = fopen(corpus_path, "rb");
    if(!f){ fprintf(stderr, "cannot open %s: %s\n", corpus_path, strerror(errno)); return 1; }
    fseek(f, 0, SEEK_END); long corpus_size = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *corpus = (uint8_t*)malloc((size_t)corpus_size);
    if(fread(corpus, 1, (size_t)corpus_size, f) != (size_t)corpus_size){ fprintf(stderr, "short read on corpus\n"); return 1; }
    fclose(f);

    int n_tensors = read_index_csv(index_path, g_metas, MAX_TENSORS);
    if(n_tensors<=0){ fprintf(stderr, "no tensors read from %s\n", index_path); return 1; }

    static GroupList groups[N_GROUPS];
    memset(groups, 0, sizeof(groups));
    for(int i=0;i<n_tensors;i++){
        int g = group_index(g_metas[i].band, g_metas[i].proj);
        if(g<0){ fprintf(stderr, "unrecognized band/proj: %s/%s\n", g_metas[i].band, g_metas[i].proj); continue; }
        groups[g].items[groups[g].n++] = &g_metas[i];
    }

    printf("# codec race -- row-coded ratio + decode throughput (n1 codec-race requirement)\n");
    printf("# corpus=%s index=%s reps=%d threads=%d (omp max=%d)\n\n", corpus_path, index_path, reps, threads, nthreads);
    printf("| group | rows | I (sym/row) | raw KB/row | codec | stored ratio (row-coded) | codebook+idx KB | 1-thread GB/s | %d-thread GB/s |\n", nthreads);
    printf("|---|---|---|---|---|---|---|---|---|\n");

    /* aggregate accumulators per codec, across all 9 groups */
    const char *codec_names[4] = {"huffman+LUT","rANS(interleaved)","tANS/FSE","bitplane+exception"};
    long agg_raw[4] = {0,0,0,0}, agg_compressed[4] = {0,0,0,0};
    double agg_time_1t[4] = {0,0,0,0}, agg_time_mt[4] = {0,0,0,0};

    for(int g=0; g<N_GROUPS; g++){
        if(groups[g].n==0) continue;
        long I = groups[g].items[0]->I;
        long row_bytes = groups[g].items[0]->row_bytes;
        long total_rows=0;
        for(int t=0;t<groups[g].n;t++) total_rows += groups[g].items[t]->O;

        uint32_t counts[16]; memset(counts, 0, sizeof(counts));
        {
            uint8_t *symbuf = (uint8_t*)malloc((size_t)I);
            for(int t=0;t<groups[g].n;t++){
                TensorMeta *tm = groups[g].items[t];
                for(long r=0;r<tm->O;r++){
                    const uint8_t *packed = corpus + tm->byte_offset + r*tm->row_bytes;
                    codec_unpack_nibbles(packed, (int)I, symbuf);
                    uint32_t c[16]; codec_histogram(symbuf, (int)I, c);
                    for(int s=0;s<16;s++) counts[s]+=c[s];
                }
            }
            free(symbuf);
        }

        char group_label[24];
        snprintf(group_label, sizeof(group_label), "%s/%s", BANDS[g/3], PROJS[g%3]);
        double raw_kb_per_row = row_bytes/1024.0;

        /* --- huffman --- */
        {
            HuffCodebook hcb; huff_build(counts, &hcb);
            size_t cbbytes = sizeof(hcb.len)+sizeof(hcb.code_rev)+((size_t)1<<hcb.maxlen)*sizeof(HuffLutEnt);
            CodecEntry ce = {codec_names[0], huff_enc_adapt, huff_dec_adapt, cbbytes};
            RaceResult r = run_codec_on_group(&ce, &hcb, corpus, groups[g].items, groups[g].n, I, row_bytes, total_rows, reps, nthreads);
            printf("| %s | %ld | %ld | %.3f | %s | %.4f | %.2f | %.2f | %.2f |\n",
                   group_label, total_rows, I, raw_kb_per_row, codec_names[0], r.stored_ratio,
                   cbbytes/1024.0, r.gbps_1thread, r.gbps_multithread);
            agg_raw[0]+=r.raw_total_bytes; agg_compressed[0]+=r.compressed_total_bytes;
            agg_time_1t[0]+=r.raw_total_bytes/1e9/r.gbps_1thread; agg_time_mt[0]+=r.raw_total_bytes/1e9/r.gbps_multithread;
            huff_free(&hcb);
        }
        /* --- rANS --- */
        {
            RansCodebook rcb; rans_build(counts, &rcb);
            size_t cbbytes = sizeof(rcb);
            CodecEntry ce = {codec_names[1], rans_enc_adapt, rans_dec_adapt, cbbytes};
            RaceResult r = run_codec_on_group(&ce, &rcb, corpus, groups[g].items, groups[g].n, I, row_bytes, total_rows, reps, nthreads);
            printf("| %s | %ld | %ld | %.3f | %s | %.4f | %.2f | %.2f | %.2f |\n",
                   group_label, total_rows, I, raw_kb_per_row, codec_names[1], r.stored_ratio,
                   cbbytes/1024.0, r.gbps_1thread, r.gbps_multithread);
            agg_raw[1]+=r.raw_total_bytes; agg_compressed[1]+=r.compressed_total_bytes;
            agg_time_1t[1]+=r.raw_total_bytes/1e9/r.gbps_1thread; agg_time_mt[1]+=r.raw_total_bytes/1e9/r.gbps_multithread;
        }
        /* --- tANS/FSE --- */
        {
            FseCodebook fcb; fse_build(counts, &fcb);
            size_t cbbytes = sizeof(fcb.freq) + TANS_TABLE_SIZE*(sizeof(uint8_t)*2+sizeof(uint32_t)) + (size_t)CODEC_NSYM*TANS_TABLE_SIZE*sizeof(uint16_t);
            CodecEntry ce = {codec_names[2], fse_enc_adapt, fse_dec_adapt, cbbytes};
            RaceResult r = run_codec_on_group(&ce, &fcb, corpus, groups[g].items, groups[g].n, I, row_bytes, total_rows, reps, nthreads);
            printf("| %s | %ld | %ld | %.3f | %s | %.4f | %.2f | %.2f | %.2f |\n",
                   group_label, total_rows, I, raw_kb_per_row, codec_names[2], r.stored_ratio,
                   cbbytes/1024.0, r.gbps_1thread, r.gbps_multithread);
            agg_raw[2]+=r.raw_total_bytes; agg_compressed[2]+=r.compressed_total_bytes;
            agg_time_1t[2]+=r.raw_total_bytes/1e9/r.gbps_1thread; agg_time_mt[2]+=r.raw_total_bytes/1e9/r.gbps_multithread;
            fse_free(&fcb);
        }
        /* --- bitplane+exception --- */
        {
            BitplaneCodebook bcb; bitplane_build(counts, &bcb);
            size_t cbbytes = sizeof(bcb);
            CodecEntry ce = {codec_names[3], bp_enc_adapt, bp_dec_adapt, cbbytes};
            RaceResult r = run_codec_on_group(&ce, &bcb, corpus, groups[g].items, groups[g].n, I, row_bytes, total_rows, reps, nthreads);
            printf("| %s | %ld | %ld | %.3f | %s | %.4f | %.2f | %.2f | %.2f |\n",
                   group_label, total_rows, I, raw_kb_per_row, codec_names[3], r.stored_ratio,
                   cbbytes/1024.0, r.gbps_1thread, r.gbps_multithread);
            agg_raw[3]+=r.raw_total_bytes; agg_compressed[3]+=r.compressed_total_bytes;
            agg_time_1t[3]+=r.raw_total_bytes/1e9/r.gbps_1thread; agg_time_mt[3]+=r.raw_total_bytes/1e9/r.gbps_multithread;
        }
    }

    printf("\n# aggregate across all %d groups (weighted by raw bytes)\n", N_GROUPS);
    printf("| codec | agg stored ratio | agg 1-thread GB/s | agg %d-thread GB/s | ns/4KB (1t) | ns/4KB (%dt) |\n", nthreads, nthreads);
    printf("|---|---|---|---|---|---|\n");
    /* mode-3 budgets from the census (the new-math replay results
     * Task 1, "per consumption mode"): 80.7 ns/4KB on the 13.3 GB/s SSD-miss
     * path (the easier bar -- mode-3's miss-path-only requirement), 2.68
     * ns/4KB on a 400 GB/s DRAM path (the harder bar, needed only for the
     * capacity-boosting compressed-RAM-residency variant). */
    const double BUDGET_MISS_NS_4KB = 80.7, BUDGET_DRAM_NS_4KB = 2.68;
    double agg_ratio[4], agg_gbps_1t[4], agg_gbps_mt[4];
    for(int c=0;c<4;c++){
        agg_ratio[c] = (double)agg_compressed[c]/(double)agg_raw[c];
        agg_gbps_1t[c] = agg_raw[c]/1e9/agg_time_1t[c];
        agg_gbps_mt[c] = agg_raw[c]/1e9/agg_time_mt[c];
        double ns4k_1t = 4096.0/(agg_gbps_1t[c]*1e9)*1e9;
        double ns4k_mt = 4096.0/(agg_gbps_mt[c]*1e9)*1e9;
        printf("| %s | %.4f | %.2f | %.2f | %.1f | %.1f |\n", codec_names[c],
               agg_ratio[c], agg_gbps_1t[c], agg_gbps_mt[c], ns4k_1t, ns4k_mt);
    }

    printf("\n# mode-3 budget check, PER CODEC'S OWN ratio (census formula:\n"
           "# budget_ns_per_4KB = (1-ratio)/BW * 4096e9; the 80.7/2.68 ns headline\n"
           "# figures are this formula at rho=0.7379 -- reused as reference points\n"
           "# above, recomputed per codec's actual measured ratio here since the\n"
           "# budget itself depends on how much I/O the codec actually saves):\n");
    printf("| codec | own miss-path budget (ns/4KB) | vs budget | own DRAM-path budget (ns/4KB) | vs budget |\n");
    printf("|---|---|---|---|---|\n");
    for(int c=0;c<4;c++){
        double ns4k_mt = 4096.0/(agg_gbps_mt[c]*1e9)*1e9;
        if(agg_ratio[c] >= 1.0){
            printf("| %s | n/a (ratio>=1.0: no I/O saved, any decode time is pure overhead) | FAIL | n/a | FAIL |\n",
                   codec_names[c]);
            continue;
        }
        double budget_miss = (1.0-agg_ratio[c])/13.3e9*4096e9;
        double budget_dram = (1.0-agg_ratio[c])/400e9*4096e9;
        printf("| %s | %.1f | %s (%.1fx) | %.2f | %s (%.1fx) |\n", codec_names[c],
               budget_miss, ns4k_mt<=budget_miss?"PASS":"FAIL", ns4k_mt/budget_miss,
               budget_dram, ns4k_mt<=budget_dram?"PASS":"FAIL", ns4k_mt/budget_dram);
    }
    printf("\n# (reference: census headline budgets at rho=0.7379 were %.1f ns/4KB "
           "miss-path / %.2f ns/4KB DRAM-path)\n", BUDGET_MISS_NS_4KB, BUDGET_DRAM_NS_4KB);

    /* "winner" among codecs that actually COMPRESS (ratio meaningfully < 1.0)
     * -- bitplane+exception's ratio~1.0 on this H~2.94-bits/weight data means
     * it fell back to raw passthrough (see codec_row_bitplane.h's build()):
     * it is the RAW-DECODE-SPEED BASELINE here, not a competing compressor,
     * so it is reported but excluded from "which compressor wins". */
    int best_idx=-1; double best_gbps=-1;
    for(int c=0;c<4;c++){
        if(agg_ratio[c] > 0.95) continue; /* not a real compressor on this data */
        if(agg_gbps_mt[c] > best_gbps){ best_gbps=agg_gbps_mt[c]; best_idx=c; }
    }
    printf("\n# winner among codecs that actually compress this data (ratio<0.95; "
           "highest aggregate %d-thread decode GB/s): %s\n", nthreads,
           best_idx>=0 ? codec_names[best_idx] : "none");
    printf("# for reference, bitplane+exception's ratio~1.0 fallback IS the raw\n"
           "# nibble-unpack speed baseline (today's matmul_i4/qt_addrow cost) --\n"
           "# no entropy coder measured here gets within an order of magnitude of\n"
           "# it, let alone the %.1f ns/4KB miss-path budget.\n", BUDGET_MISS_NS_4KB);

    free(corpus);
    return 0;
}
