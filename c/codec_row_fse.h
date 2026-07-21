/* codec_row_fse.h -- table-driven ANS (tANS / "FSE-style": Finite State
 * Entropy, Duda's ANS theory in its table form) over the 16 int4 symbols.
 *
 * Distinct in character from codec_row_rans.h's rANS: rANS decode does one
 * table lookup (slot->symbol) PLUS one integer multiply per symbol
 * (`freq*(x>>M_BITS)+slot-cum`); tANS decode is PURE TABLE LOOKUP + a bit
 * read + an add, no multiply/divide at all -- the classical "FSE is rANS
 * with the arithmetic precomputed into a table" tradeoff (a bit more setup
 * work per codebook build, in exchange for a simpler decode instruction
 * mix). Same TANS_TABLE_LOG=12 precision as rANS's M_BITS=12 so the
 * achieved ratio is directly comparable in the race.
 *
 * Construction (standard tANS, e.g. Yann Collet's FSE / Jarek Duda's ANS):
 *   1. quantize the aggregate histogram to freq[] summing to TABLE_SIZE;
 *   2. "spread" symbols across table cells with the usual open-addressing
 *      step (5/8 TABLE_SIZE + 3, odd => coprime with the power-of-two table
 *      size => visits every cell exactly once);
 *   3. walk cells in table order with a per-symbol running occurrence
 *      counter (starting at freq[s]) to assign each cell a decode entry
 *      (symbol, nbBits, newStateBase) -- the standard recurrence
 *      nbBits = tableLog - floor(log2(nextState)),
 *      newStateBase = (nextState << nbBits) - tableSize;
 *   4. build the ENCODE table by inverting the decode table directly: for
 *      each decode cell i assigned to symbol s, all "state_post" values in
 *      [newStateBase, newStateBase + 2^nbBits) decode-transition INTO cell
 *      i, so encoding symbol s from state_post finds cell i by a plain
 *      array lookup `enc[s][state_post] = i` built once at codebook time --
 *      this sidesteps re-deriving the closed-form encode shortcuts (delta
 *      tables) from memory, at the cost of a 16*TABLE_SIZE*2B lookup table
 *      (128 KB for TABLE_LOG=12), amortized once per projection.
 *
 * Row layout: [uint16 initial decode state][packed bitstream, LSB-first,
 * ceil(total_bits/8) bytes]. Encode processes the row in REVERSE (n-1..0,
 * same LIFO convention as rANS) computing each step's (bits,width) into a
 * position-indexed buffer, then packs that buffer FORWARD (k=0..n-1) so the
 * decoder -- which naturally runs forward -- reads them in the right order.
 */
#ifndef ILIRIA_CODEC_ROW_FSE_H
#define ILIRIA_CODEC_ROW_FSE_H

#include "codec_row.h"

#ifndef TANS_TABLE_LOG
#define TANS_TABLE_LOG 12
#endif
#define TANS_TABLE_SIZE (1u << TANS_TABLE_LOG)

typedef struct {
    uint32_t freq[CODEC_NSYM];
    uint8_t  *dsym;   /* [TANS_TABLE_SIZE] */
    uint8_t  *dbits;  /* [TANS_TABLE_SIZE] */
    uint32_t *dbase;  /* [TANS_TABLE_SIZE] */
    uint16_t *enc;    /* [CODEC_NSYM * TANS_TABLE_SIZE], row-major per symbol */
} FseCodebook;

static void fse_build(const uint32_t counts[CODEC_NSYM], FseCodebook *cb){
    codec_quantize_freqs(counts, TANS_TABLE_SIZE, cb->freq);
    cb->dsym = (uint8_t*)malloc(TANS_TABLE_SIZE*sizeof(uint8_t));
    cb->dbits = (uint8_t*)malloc(TANS_TABLE_SIZE*sizeof(uint8_t));
    cb->dbase = (uint32_t*)malloc(TANS_TABLE_SIZE*sizeof(uint32_t));
    cb->enc = (uint16_t*)malloc((size_t)CODEC_NSYM*TANS_TABLE_SIZE*sizeof(uint16_t));

    uint8_t *tableSymbol = (uint8_t*)malloc(TANS_TABLE_SIZE*sizeof(uint8_t));
    uint32_t step = (TANS_TABLE_SIZE>>1) + (TANS_TABLE_SIZE>>3) + 3; /* odd: full period vs power-of-2 table */
    uint32_t mask = TANS_TABLE_SIZE - 1;
    uint32_t pos = 0;
    for(int s=0;s<CODEC_NSYM;s++)
        for(uint32_t c=0;c<cb->freq[s];c++){ tableSymbol[pos]=(uint8_t)s; pos=(pos+step)&mask; }

    uint32_t symbolNext[CODEC_NSYM];
    memcpy(symbolNext, cb->freq, sizeof(cb->freq));
    for(uint32_t i=0;i<TANS_TABLE_SIZE;i++){
        uint8_t s = tableSymbol[i];
        uint32_t nextState = symbolNext[s]++;
        int nbBits = TANS_TABLE_LOG - codec_highbit32(nextState);
        uint32_t base = (nextState << nbBits) - TANS_TABLE_SIZE;
        cb->dsym[i]=s; cb->dbits[i]=(uint8_t)nbBits; cb->dbase[i]=base;
        uint32_t blockSize = 1u<<nbBits;
        uint16_t *encrow = cb->enc + (size_t)s*TANS_TABLE_SIZE;
        for(uint32_t v=0; v<blockSize; v++) encrow[base+v] = (uint16_t)i;
    }
    free(tableSymbol);
}
static void fse_free(FseCodebook *cb){
    free(cb->dsym); free(cb->dbits); free(cb->dbase); free(cb->enc);
    cb->dsym=NULL; cb->dbits=NULL; cb->dbase=NULL; cb->enc=NULL;
}

static size_t fse_encode_row(const FseCodebook *cb, const uint8_t *sym, int n,
                              uint8_t *out, size_t cap){
    if(n<=0) return 0;
    uint16_t *bits_buf = (uint16_t*)malloc((size_t)n*sizeof(uint16_t));
    uint8_t  *width_buf = (uint8_t*)malloc((size_t)n*sizeof(uint8_t));
    uint32_t state = 0; /* arbitrary internal seed; never transmitted */
    for(int k=n-1;k>=0;k--){
        uint8_t s = sym[k];
        const uint16_t *encrow = cb->enc + (size_t)s*TANS_TABLE_SIZE;
        uint32_t cell = encrow[state];
        int nbBits = cb->dbits[cell];
        uint32_t base = cb->dbase[cell];
        bits_buf[k] = (uint16_t)(state - base);
        width_buf[k] = (uint8_t)nbBits;
        state = cell;
    }
    uint16_t state0 = (uint16_t)state; /* = state_0: the transmitted decode seed */
    size_t total_bits=0; for(int k=0;k<n;k++) total_bits += width_buf[k];
    size_t paybytes = (total_bits+7)/8;
    size_t total = sizeof(uint16_t) + paybytes;
    if(total > cap){ free(bits_buf); free(width_buf); return (size_t)-1; }
    memcpy(out, &state0, sizeof(uint16_t));
    CodecBitW w; cbw_init(&w, out+sizeof(uint16_t), cap-sizeof(uint16_t));
    for(int k=0;k<n;k++) cbw_put(&w, bits_buf[k], width_buf[k]);
    size_t used = cbw_finish(&w);
    free(bits_buf); free(width_buf);
    return sizeof(uint16_t) + used;
}

/* decode: caller-allocated sym_out[n], no internal allocation */
static void fse_decode_row(const FseCodebook *cb, const uint8_t *payload, size_t paylen,
                            int n, uint8_t *sym_out){
    if(n<=0) return;
    uint16_t state0; memcpy(&state0, payload, sizeof(uint16_t));
    uint32_t state = state0;
    CodecBitR r; cbr_init(&r, payload+sizeof(uint16_t), paylen-sizeof(uint16_t));
    for(int k=0;k<n;k++){
        sym_out[k] = cb->dsym[state];
        int nbBits = cb->dbits[state];
        uint32_t bits = cbr_get(&r, nbBits);
        state = cb->dbase[state] + bits;
    }
}

#endif /* ILIRIA_CODEC_ROW_FSE_H */
