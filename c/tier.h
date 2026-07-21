/* Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. */
#ifndef ILIRIA_TIER_H
#define ILIRIA_TIER_H

#include <stdint.h>
#include <string.h>
#include <ctype.h>

/* Pick one RAM/VRAM hot-store slot to replace from recent routing heat.
 * The fixed margin handles tiny samples; the 25% margin prevents ping-pong. */
static int tier_pick_swap(const uint32_t *heat, int nexpert,
                          const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat || !pinned || npin<1 || nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++) if(heat[pinned[z]]<heat[pinned[cold]]) cold=z;
    int hot=-1; uint32_t fh=0;
    for(int e=0;e<nexpert;e++){
        int resident=0;
        for(int z=0;z<npin;z++) if(pinned[z]==e){ resident=1; break; }
        if(!resident && heat[e]>fh){ fh=heat[e]; hot=e; }
    }
    if(hot<0) return 0;
    uint32_t fc=heat[pinned[cold]];
    if(fh<=fc+(fc>>2)+4) return 0;
    *slot=cold; *eid=hot; *gain=(long)fh-(long)fc;
    return 1;
}

static void tier_decay(uint32_t *heat, int nexpert){
    for(int e=0;e<nexpert;e++) heat[e]>>=1;
}

/* ---- Workload-specific decayed hotsets (Phase 2) ---- */

/* Validate a hotset profile name.
 * Rejects empty strings, names containing path separators, traversal chars,
 * or characters outside [a-z0-9_.-]. Max 63 chars. Returns 1 if valid, 0 if not. */
static int hotset_validate_profile(const char *name){
    if(!name || !*name) return 0;
    size_t len = strlen(name);
    if(len > 63) return 0;
    for(size_t i=0; i<len; i++){
        char c = name[i];
        if(c == '/' || c == '\\' || (c == '.' && i+1<len && name[i+1]=='.')) return 0;
        if(c == ':' || c == ';' || c == '|' || c == '&' || c == '$' ||
           c == '`' || c == '(' || c == ')' || c == '{' || c == '}') return 0;
        if(!isalnum((unsigned char)c) && c != '_' && c != '-' && c != '.') return 0;
    }
    return 1;
}

/* Apply fixed-point multiplicative decay to heat counters.
 * factor is stored as integer percentage (e.g., 98 = 0.98).
 * This avoids floating-point in the hot path. */
static void hotset_decay_fixed(uint32_t *heat, int nexpert, int factor_pct){
    if(!heat || nexpert<1 || factor_pct<=0 || factor_pct>=100) return;
    for(int e=0; e<nexpert; e++){
        uint64_t v = (uint64_t)heat[e] * factor_pct;
        heat[e] = (uint32_t)(v / 100);
    }
}

/* Check if a line is a recognized hotset header comment.
 * Returns 1 if the line should be skipped (header/comment), 0 if data. */
static int hotset_is_header(const char *line){
    if(!line) return 0;
    /* Skip leading whitespace */
    while(*line && isspace((unsigned char)*line)) line++;
    return *line == '#';
}

/* Parse a header field value from a comment line like "# profile: coding".
 * Returns the value string (skipping leading spaces) or NULL if not found. */
static const char *hotset_header_field(const char *line, const char *field){
    if(!line || !field) return NULL;
    /* Skip leading whitespace */
    while(*line && isspace((unsigned char)*line)) line++;
    if(*line != '#') return NULL;
    line++; /* skip '#' */
    while(*line && isspace((unsigned char)*line)) line++;
    size_t flen = strlen(field);
    if(strncasecmp(line, field, flen) != 0) return NULL;
    line += flen;
    while(*line && isspace((unsigned char)*line)) line++;
    if(*line == ':'){ line++; while(*line && isspace((unsigned char)*line)) line++; }
    return *line ? line : NULL;
}

/* Construct the usage file path for a given profile.
 * If profile is NULL or empty, returns the legacy path "<snap>/.fa_usage".
 * Otherwise returns "<snap>/.fa_usage.<profile>".
 * buf must be at least 2100 bytes. Returns buf. */
static char *hotset_usage_path(char *buf, size_t buf_sz, const char *snap,
                                const char *profile){
    if(!profile || !*profile || !hotset_validate_profile(profile)){
        snprintf(buf, buf_sz, "%s/.fa_usage", snap);
    } else {
        snprintf(buf, buf_sz, "%s/.fa_usage.%s", snap, profile);
    }
    return buf;
}

#endif
