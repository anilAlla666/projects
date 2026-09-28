/* W.6 sub-C cohort registry churn soak.
 *
 * Rapidly spawns short-lived processes that register a random fingerprint +
 * query + exit, sustained for CHURN_SECS, with a steady-state pool of ~POOL
 * concurrent registrants. Stresses: (1) kzalloc/kfree under rapid insert+prune
 * churn (leak detection), (2) prune correctness under load, (3) open-addressing
 * /hashtable integrity, (4) no kernel oops. A long-lived "anchor" process keeps
 * querying so we can read the live count over time.
 *
 * Build: gcc -O2 -o cohort_churn cohort_churn.c
 * Run:   ./cohort_churn [secs] [pool]      (default 90 s, pool 40)
 * Pair with: dmesg before/after to confirm no oops / no leak warnings;
 *            check final co_resident decays toward the anchor count after churn.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <stdint.h>
#include <linux/ioctl.h>

#define MAGIC 'C'
#define COHORT_MAX 128
struct creg { uint32_t tgid; uint32_t _p; uint64_t fp; };
struct cpeer{ uint32_t tgid; uint32_t _p; uint64_t fp; };
struct cq   { uint64_t cfp; uint32_t maxe; uint32_t n; struct cpeer e[COHORT_MAX]; };
#define COHORT_REGISTER _IOW(MAGIC,30,struct creg)
#define COHORT_QUERY    _IOWR(MAGIC,31,struct cq)

static uint32_t q(int fd, uint64_t fp){
    struct cq c; memset(&c,0,sizeof c); c.cfp=fp; c.maxe=COHORT_MAX;
    if(ioctl(fd,COHORT_QUERY,&c)<0) return 0xffffffff;
    return c.n;
}
static void reg(int fd,uint64_t fp){ struct creg r={0,0,fp}; ioctl(fd,COHORT_REGISTER,&r); }

int main(int argc,char**argv){
    int secs = argc>1?atoi(argv[1]):90;
    int pool = argc>2?atoi(argv[2]):40;
    int afd = open("/dev/cipher",O_RDWR|O_CLOEXEC);
    if(afd<0){ printf("open fail %d\n",errno); return 2; }
    srand(getpid());
    /* anchor registers a stable fp + heartbeats throughout */
    uint64_t anchor_fp = 0xAAAA0000AAAA0000ULL;
    reg(afd,anchor_fp);

    time_t t0=time(NULL); long spawned=0, errs=0; uint32_t peakn=0;
    while(time(NULL)-t0 < secs){
        int live=0;
        for(int i=0;i<pool;i++){
            pid_t p=fork();
            if(p==0){
                int fd=open("/dev/cipher",O_RDWR|O_CLOEXEC);
                if(fd<0)_exit(3);
                uint64_t fp = ((uint64_t)rand()<<32)|rand();
                reg(fd,fp);
                uint32_t n=q(fd,fp);
                _exit(n==0xffffffff?4:0);
            }
            if(p>0) live++; else errs++;
        }
        /* anchor heartbeat + read live count while children alive */
        uint32_t n=q(afd,anchor_fp);
        if(n!=0xffffffff && n>peakn) peakn=n;
        for(int i=0;i<live;i++){ int st; wait(&st); if(WEXITSTATUS(st)) errs++; }
        spawned += live;
    }
    /* let dead children age out: anchor keeps heartbeating, others must prune */
    uint32_t before_prune=q(afd,anchor_fp);
    printf("churn: spawned=%ld errs=%ld peak_co_resident=%u before_prune=%u\n",
           spawned,errs,peakn,before_prune);
    printf("waiting 35s for stale prune...\n"); fflush(stdout);
    for(int i=0;i<20;i++){ sleep(2); }
    uint32_t after=q(afd,anchor_fp);
    /* after churn stops + 40s, only the anchor (still heartbeating) should remain */
    printf("churn: after_prune co_resident=%u (expect ~1: only heartbeating anchor)\n",after);
    int ok = (errs==0) && (after<=2) && (peakn>=2);
    printf("=== cohort_churn: %s (errs=%ld peak=%u after=%u) ===\n", ok?"PASS":"FAIL",errs,peakn,after);
    close(afd);
    return ok?0:1;
}
