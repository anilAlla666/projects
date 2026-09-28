/* W.6 sub-C — strict Memory #16 kmod-wide regression: N=128 resolver-coherence soak.
 *
 * Validates the UNCHANGED W7-9 Step 5 multi-tenant resolver hot path
 * (CIPHER_REGISTER_STREAMS NR 29 + the vmalloc'd view-slot table + the
 * lock-free generation-counter mmap lookup) under N=128 concurrent register
 * churn, with kmod 0.6.6 (W.6 sub-C cohort registry) loaded. The hypothesis
 * Memory #16 tests: the new cohort hashtable/spinlock/init-ordering does NOT
 * perturb the existing resolver hot path. Gate: incoherent == 0, no oops/leak,
 * clean ko cycle.
 *
 * Mechanism mirrored byte-exact from cipher_rt_phase4/cipher_stream_resolver.c
 * + cipher_stream_resolver.h. The original cipher_test_commit_n128 /
 * test_observe_publish harnesses were /tmp scratch (not preserved); the
 * resolver path is the clearest single-driver representative of the kmod hot
 * path with a hard coherence metric.
 *
 * Build: gcc -O2 -pthread -o resolver_soak resolver_soak.c
 * Run:   ./resolver_soak [secs=1800] [writers=128] [readers=8]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#define MAGIC 'C'
#define SLOTS 8192
#define PGOFF 0x100000
#define PAGE  4096
#define NONE  0xFFFFFFFFu
#define MAXPT 16

struct slot { uint64_t handle; uint32_t tgid; uint32_t tenant_id; uint32_t generation; uint32_t _pad; } __attribute__((packed));
struct regs { uint32_t tenant_id; uint32_t num_streams; uint64_t handles[MAXPT]; };
#define REGISTER_STREAMS _IOW(MAGIC,29,struct regs)

static int g_fd;
static volatile struct slot *g_view;
static uint32_t g_tgid;
static volatile int g_run = 1;
static int g_nwriters;

static inline uint32_t H(uint32_t tgid, uint64_t h){
    uint64_t x=(uint64_t)tgid*0x100000001b3ULL; x^=h*0x100000001b3ULL; x^=x>>32;
    return (uint32_t)(x&(SLOTS-1));
}
static inline uint64_t handle_of(int i){ return (uint64_t)(i+1)*0x1000ULL + 1; }
static inline uint32_t tenant_of(int i){ return (uint32_t)(i+1); }

static int reg(int i){
    struct regs r; memset(&r,0,sizeof r);
    r.tenant_id=tenant_of(i); r.num_streams=1; r.handles[0]=handle_of(i);
    return ioctl(g_fd,REGISTER_STREAMS,&r)<0 ? -errno : 0;
}

/* lock-free lookup, mirrors cipher_v2_current_tenant_id_from_stream */
static uint32_t lookup(uint64_t handle){
    if(handle==0) return NONE;
    uint32_t idx=H(g_tgid,handle), probes=0;
    while(probes<SLOTS){
        volatile struct slot *s=&g_view[idx];
        uint32_t g1=__atomic_load_n(&s->generation,__ATOMIC_ACQUIRE);
        uint64_t h =__atomic_load_n(&s->handle,__ATOMIC_RELAXED);
        if(h==0) return NONE;
        if(h==handle && __atomic_load_n(&s->tgid,__ATOMIC_RELAXED)==g_tgid){
            uint32_t tid=__atomic_load_n(&s->tenant_id,__ATOMIC_RELAXED);
            uint32_t g2=__atomic_load_n(&s->generation,__ATOMIC_ACQUIRE);
            if(g1==g2) return tid;
            continue;
        }
        idx=(idx+1)&(SLOTS-1); probes++;
    }
    return NONE;
}

static void* writer(void* a){
    int i=(int)(long)a;
    while(g_run){ reg(i); }      /* idempotent re-register => generation churn */
    return NULL;
}

#define SMAX 100000
struct rstat { uint64_t reads, incoherent, misses; uint64_t lat_ns[SMAX]; uint32_t nsamp; };

static uint64_t now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec; }

static void* reader(void* a){
    struct rstat* st=(struct rstat*)a;
    unsigned seed=(unsigned)(uintptr_t)a ^ (unsigned)now_ns();
    uint64_t c=0;
    while(g_run){
        int i=rand_r(&seed)%g_nwriters;
        uint64_t t0=now_ns();
        uint32_t tid=lookup(handle_of(i));
        uint64_t dt=now_ns()-t0;
        st->reads++;
        if(tid==NONE) st->misses++;
        else if(tid!=tenant_of(i)) st->incoherent++;   /* coherence violation */
        if((c++ & 0xFF)==0 && st->nsamp<SMAX) st->lat_ns[st->nsamp++]=dt;
    }
    return NULL;
}

static int cmp_u64(const void*a,const void*b){ uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return x<y?-1:x>y?1:0; }

int main(int argc,char**argv){
    int secs   =argc>1?atoi(argv[1]):1800;
    g_nwriters =argc>2?atoi(argv[2]):128;
    int nread  =argc>3?atoi(argv[3]):8;
    g_tgid=(uint32_t)getpid();

    g_fd=open("/dev/cipher",O_RDWR|O_CLOEXEC);
    if(g_fd<0){ printf("open /dev/cipher errno=%d\n",errno); return 2; }
    size_t sz=((size_t)SLOTS*sizeof(struct slot)+PAGE-1)&~(size_t)(PAGE-1);
    void* p=mmap(NULL,sz,PROT_READ,MAP_SHARED,g_fd,(off_t)PGOFF*PAGE);
    if(p==MAP_FAILED){ printf("mmap view errno=%d\n",errno); return 2; }
    g_view=(volatile struct slot*)p;

    /* --- pre-soak smoke + atomicity gate (advisor pre-check) --- */
    for(int i=0;i<g_nwriters;i++){ int rc=reg(i); if(rc){ printf("SMOKE FAIL: register i=%d rc=%d (errno)\n",i,-rc); return 3; } }
    uint64_t agood=0,abad=0;
    for(int k=0;k<1000;k++) for(int i=0;i<g_nwriters;i++){ uint32_t t=lookup(handle_of(i)); if(t==tenant_of(i))agood++; else abad++; }
    printf("atomicity gate: %llu/%llu coherent (bad=%llu)\n",
           (unsigned long long)agood,(unsigned long long)(agood+abad),(unsigned long long)abad);
    if(abad){ printf("SMOKE FAIL: atomicity gate bad=%llu\n",(unsigned long long)abad); return 3; }
    printf("smoke PASS — %d entries resolve correctly; starting %ds soak (writers=%d readers=%d)\n",
           g_nwriters,secs,g_nwriters,nread);
    fflush(stdout);

    /* --- soak --- */
    pthread_t* wt=calloc(g_nwriters,sizeof(pthread_t));
    pthread_t* rt=calloc(nread,sizeof(pthread_t));
    struct rstat* rs=calloc(nread,sizeof(struct rstat));
    uint64_t reg0=0; /* approximate: count writer iters via a shared counter is racy; report reader-side only */
    (void)reg0;
    for(int i=0;i<g_nwriters;i++) pthread_create(&wt[i],NULL,writer,(void*)(long)i);
    for(int i=0;i<nread;i++) pthread_create(&rt[i],NULL,reader,&rs[i]);

    time_t t0=time(NULL);
    while(time(NULL)-t0<secs){
        sleep(60);
        uint64_t r=0,inc=0,mis=0; for(int i=0;i<nread;i++){r+=rs[i].reads;inc+=rs[i].incoherent;mis+=rs[i].misses;}
        printf("  [t+%lds] reads=%llu incoherent=%llu misses=%llu\n",
               (long)(time(NULL)-t0),(unsigned long long)r,(unsigned long long)inc,(unsigned long long)mis);
        fflush(stdout);
    }
    g_run=0;
    for(int i=0;i<g_nwriters;i++) pthread_join(wt[i],NULL);
    for(int i=0;i<nread;i++) pthread_join(rt[i],NULL);

    /* --- aggregate --- */
    uint64_t reads=0,inc=0,mis=0; uint32_t ns=0;
    static uint64_t lat[SMAX*16];
    for(int i=0;i<nread;i++){ reads+=rs[i].reads; inc+=rs[i].incoherent; mis+=rs[i].misses;
        for(uint32_t j=0;j<rs[i].nsamp && ns<SMAX*16;j++) lat[ns++]=rs[i].lat_ns[j]; }
    qsort(lat,ns,sizeof(uint64_t),cmp_u64);
    uint64_t p50=ns?lat[ns/2]:0, p99=ns?lat[(ns*99)/100]:0, mx=ns?lat[ns-1]:0;
    double rate=reads/(double)secs;
    int ok=(inc==0);
    printf("\n=== resolver_soak %s ===\n", ok?"PASS":"FAIL");
    printf("duration=%ds writers=%d readers=%d\n",secs,g_nwriters,nread);
    printf("total reads=%llu  rate=%.2f M/s\n",(unsigned long long)reads,rate/1e6);
    printf("INCOHERENT=%llu  (gate==0)  misses=%llu\n",(unsigned long long)inc,(unsigned long long)mis);
    printf("reader lookup latency (sampled %u): p50=%lluns p99=%lluns max=%lluns\n",
           ns,(unsigned long long)p50,(unsigned long long)p99,(unsigned long long)mx);
    return ok?0:1;
}
