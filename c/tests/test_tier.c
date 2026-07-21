/* Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. */
#include <stdio.h>
#include <string.h>
#include "../tier.h"

static int fail(const char *message){
    fprintf(stderr,"tier test failed: %s\n",message);
    return 1;
}

int main(void){
    /* ---- Existing tier_pick_swap / tier_decay tests ---- */
    uint32_t heat[6]={20,2,8,3,30,1};
    int pinned[2]={0,1}, slot=-1, eid=-1; long gain=0;
    if(!tier_pick_swap(heat,6,pinned,2,&slot,&eid,&gain)) return fail("hot expert not promoted");
    if(slot!=1 || eid!=4 || gain!=28) return fail("wrong promotion candidate");

    uint32_t stable[4]={20,18,24,4}; int resident[2]={0,1};
    if(tier_pick_swap(stable,4,resident,2,&slot,&eid,&gain)) return fail("hysteresis did not block churn");

    tier_decay(heat,6);
    if(heat[0]!=10 || heat[1]!=1 || heat[4]!=15) return fail("heat decay");

    /* ---- hotset_validate_profile ---- */
    if(!hotset_validate_profile("coding")) return fail("valid profile 'coding' rejected");
    if(!hotset_validate_profile("reasoning")) return fail("valid profile 'reasoning' rejected");
    if(!hotset_validate_profile("general")) return fail("valid profile 'general' rejected");
    if(!hotset_validate_profile("my-profile_v2.0")) return fail("valid profile with dots/dashes rejected");
    if(!hotset_validate_profile("a")) return fail("single-char profile rejected");

    if(hotset_validate_profile("")) return fail("empty profile accepted");
    if(hotset_validate_profile(NULL)) return fail("NULL profile accepted");
    if(hotset_validate_profile("../escape")) return fail("traversal profile accepted");
    if(hotset_validate_profile("path/separated")) return fail("path-separated profile accepted");
    if(hotset_validate_profile("has;semicolon")) return fail("semicolon profile accepted");
    if(hotset_validate_profile("has$dollar")) return fail("dollar-sign profile accepted");
    { char long_name[66]; memset(long_name, 'a', 65); long_name[65]=0;
      if(hotset_validate_profile(long_name)) return fail("65-char profile accepted (max 63)"); }

    /* ---- hotset_decay_fixed ---- */
    uint32_t d1[3]={1000,500,100};
    hotset_decay_fixed(d1, 3, 98);
    if(d1[0]!=980 || d1[1]!=490 || d1[2]!=98) return fail("decay 98%%");

    uint32_t d2[2]={100,200};
    hotset_decay_fixed(d2, 2, 90);
    if(d2[0]!=90 || d2[1]!=180) return fail("decay 90%%");

    /* boundary: factor=0 or 100 should be no-op */
    uint32_t d3[1]={42};
    hotset_decay_fixed(d3, 1, 0);
    if(d3[0]!=42) return fail("decay factor=0 should be no-op");
    hotset_decay_fixed(d3, 1, 100);
    if(d3[0]!=42) return fail("decay factor=100 should be no-op");
    hotset_decay_fixed(d3, 1, -1);
    if(d3[0]!=42) return fail("decay factor=-1 should be no-op");

    /* ---- hotset_is_header ---- */
    if(!hotset_is_header("# comment")) return fail("plain header not detected");
    if(!hotset_is_header("  # indented comment")) return fail("indented header not detected");
    if(!hotset_is_header("# profile: coding")) return fail("profile header not detected");
    if(hotset_is_header("123 456 789")) return fail("data line detected as header");
    if(hotset_is_header("")) return fail("empty line detected as header");
    if(hotset_is_header(NULL)) return fail("NULL line detected as header");

    /* ---- hotset_header_field ---- */
    const char *v = hotset_header_field("# profile: coding", "profile");
    if(!v || strcmp(v, "coding")!=0) return fail("header field 'profile' not parsed");

    v = hotset_header_field("# decay_factor: 98", "decay_factor");
    if(!v || strcmp(v, "98")!=0) return fail("header field 'decay_factor' not parsed");

    if(hotset_header_field("# other: val", "profile")) return fail("non-matching field returned");
    if(hotset_header_field("123 456", "profile")!=NULL) return fail("non-header line returned field");

    /* ---- hotset_usage_path ---- */
    { char buf[2100];
      char *r = hotset_usage_path(buf, sizeof(buf), "/tmp/snap", NULL);
      if(strcmp(r, "/tmp/snap/.fa_usage")!=0) return fail("usage path with NULL profile");

      r = hotset_usage_path(buf, sizeof(buf), "/tmp/snap", "");
      if(strcmp(r, "/tmp/snap/.fa_usage")!=0) return fail("usage path with empty profile");

      r = hotset_usage_path(buf, sizeof(buf), "/tmp/snap", "coding");
      if(strcmp(r, "/tmp/snap/.fa_usage.coding")!=0) return fail("usage path with profile");

      /* invalid profile falls back to legacy path */
      r = hotset_usage_path(buf, sizeof(buf), "/tmp/snap", "../bad");
      if(strcmp(r, "/tmp/snap/.fa_usage")!=0) return fail("usage path with invalid profile should fallback");
    }

    puts("tier tests: ok");
    return 0;
}
