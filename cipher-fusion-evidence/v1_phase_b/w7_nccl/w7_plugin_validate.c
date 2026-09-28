/* W.7 items 1(symbol)/2(load+getCollInfo+fallback)/4(decision validity)/6(fallback).
 * dlopen the EXISTING libcipher_nccl_tuner.so (no rebuild), drive its ABI directly. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "cipher_nccl_tuner_abi.h"
static void logfn(const char* a,int b,int c,const char* d,...){(void)a;(void)b;(void)c;(void)d;}
static int valid_algo(int a){ return a==NCCL_ALGO_UNDEF || (a>=0 && a<=5); }
static int valid_proto(int p){ return p==NCCL_PROTO_UNDEF || (p>=0 && p<=2); }
int main(int argc, char** argv){
    const char* path = argv[1];
    void* h = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
    if(!h){ fprintf(stderr,"dlopen FAIL: %s\n", dlerror()); return 2; }
    ncclTuner_v2_t* v2 = (ncclTuner_v2_t*)dlsym(h,"ncclTunerPlugin_v2");
    void* v1 = dlsym(h,"ncclTunerPlugin_v1");
    printf("ITEM1 ncclTunerPlugin_v2=%s ncclTunerPlugin_v1=%s name=%s\n",
           v2?"PRESENT":"MISSING", v1?"PRESENT":"MISSING",
           (v2&&v2->name)?v2->name:"?");
    if(!v2) return 2;
    void* br = dlsym(RTLD_DEFAULT,"cipher_nccl_record_decide");
    printf("BRIDGE_RESOLVABLE=%d\n", br!=NULL);
    void* ctx=NULL;
    size_t NR=argc>2?(size_t)atoi(argv[2]):2; size_t NN=argc>3?(size_t)atoi(argv[3]):1; ncclResult_t r = v2->init(NR,NN,logfn,&ctx);
    printf("ITEM2 init rc=%d (0=ncclSuccess)\n", (int)r);
    size_t sizes[]={1024,65536,1048576,16777216,268435456,1073741824};
    ncclFunc_t colls[]={ncclFuncAllReduce,ncclFuncBroadcast,ncclFuncAllGather,ncclFuncReduceScatter};
    const char* cn[]={"AllReduce","Broadcast","AllGather","ReduceScatter"};
    int total=0,bad=0,undef=0,defined=0;
    for(int ci=0;ci<4;ci++) for(int si=0;si<6;si++){
        int a=-99,p=-99,nc=-99;
        ncclResult_t rr=v2->getCollInfo(ctx,colls[ci],sizes[si],0,0,1,&a,&p,&nc);
        total++;
        int ok=(rr==ncclSuccess)&&valid_algo(a)&&valid_proto(p)&&(a==NCCL_ALGO_UNDEF||nc>=0);
        if(!ok){bad++;printf("  BAD %s bytes=%zu rc=%d algo=%d proto=%d nch=%d\n",cn[ci],sizes[si],(int)rr,a,p,nc);}
        if(a==NCCL_ALGO_UNDEF)undef++; else {defined++;
            if(si==0||si==5)printf("  %s bytes=%zu -> algo=%d proto=%d nch=%d\n",cn[ci],sizes[si],a,p,nc);}
    }
    printf("ITEM4/6 SWEEP total=%d bad=%d undef=%d defined=%d\n",total,bad,undef,defined);
    if(v2->destroy) v2->destroy(ctx);
    return bad?1:0;
}
