/*
 * HyperFlux Spawn Selection v4.0 - IMPLEMENTATION
 * CONFIDENTIAL
 */

#include "HyperFluxSpawn.h"
#include "spawn_weights.inc"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace HyperFlux { namespace Spawn {

inline float gelu(float x) { return 0.5f*x*(1.0f+std::tanh(0.7978845608f*(x+0.044715f*x*x*x))); }
inline float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-x)); }

void layer_norm(float* x, const float* g, const float* b, int d) {
    float m=0,v=0; for(int i=0;i<d;i++)m+=x[i]; m/=d;
    for(int i=0;i<d;i++)v+=(x[i]-m)*(x[i]-m); v/=d;
    float s=1.0f/std::sqrt(v+1e-5f);
    for(int i=0;i<d;i++)x[i]=g[i]*(x[i]-m)*s+b[i];
}

void linear(const float* in, float* out, const float* W, const float* b, int id, int od) {
    for(int o=0;o<od;o++){out[o]=b[o];for(int i=0;i<id;i++)out[o]+=in[i]*W[o*id+i];}
}

void gelu_inplace(float* x, int d) { for(int i=0;i<d;i++)x[i]=gelu(x[i]); }

void resblock(float* x, float* t, const float* W1, const float* b1, const float* g1, const float* be1,
              const float* W2, const float* b2, const float* g2, const float* be2, int d) {
    float r[256]; std::memcpy(r,x,d*sizeof(float));
    linear(x,t,W1,b1,d,d); layer_norm(t,g1,be1,d); gelu_inplace(t,d);
    linear(t,x,W2,b2,d,d); layer_norm(x,g2,be2,d);
    for(int i=0;i<d;i++)x[i]=gelu(x[i]+r[i]);
}

class Engine::Impl {
public:
    bool ready=false;
    Vec3 spawnPos[MAX_SPAWNS];
    mutable uint64_t queries=0;
    
    Impl() {
        const float s=MAP_SIZE; int idx=0;
        float z1[]={-s/3,0,s/3}; for(int i=0;i<3;i++){spawnPos[idx++]=Vec3(-s/2+5,1,z1[i]);spawnPos[idx++]=Vec3(s/2-5,1,z1[i]);}
        float z2[]={-s/4,s/4}; for(int i=0;i<2;i++){spawnPos[idx++]=Vec3(-s/2+8,1,z2[i]);spawnPos[idx++]=Vec3(s/2-8,1,z2[i]);}
        spawnPos[idx++]=Vec3(-s/2+12,1,0); spawnPos[idx++]=Vec3(s/2-12,1,0);
        float in[][2]={{-s/4,-s/4},{-s/4,s/4},{s/4,-s/4},{s/4,s/4},{0,-s/3},{0,s/3},{-s/3,0},{s/3,0},
                       {-s/6,-s/6},{-s/6,s/6},{s/6,-s/6},{s/6,s/6},{0,0},{-s/5,-s/3},{s/5,s/3},{0,-s/6},{-s/4,0},{s/4,0}};
        for(int i=0;i<18&&idx<MAX_SPAWNS;i++)spawnPos[idx++]=Vec3(in[i][0],1,in[i][1]);
    }
    
    void encode(const GameState& st, float* out) const {
        float eH[256]={0},tH[256]={0},thr[8]={0},gF[8];
        for(int i=0;i<st.numEnemies&&i<MAX_ENEMIES;i++)if(st.enemies[i].active){
            int gx=(int)((st.enemies[i].position.x/MAP_SIZE+0.5f)*GRID_RES);
            int gz=(int)((st.enemies[i].position.z/MAP_SIZE+0.5f)*GRID_RES);
            gx=std::max(0,std::min(GRID_RES-1,gx)); gz=std::max(0,std::min(GRID_RES-1,gz));
            eH[gx*GRID_RES+gz]+=1.0f;
        }
        if(st.numEnemies>0)for(int i=0;i<256;i++)eH[i]/=st.numEnemies;
        
        for(int i=0;i<st.numTeammates&&i<MAX_TEAMMATES;i++)if(st.teammates[i].active){
            int gx=(int)((st.teammates[i].position.x/MAP_SIZE+0.5f)*GRID_RES);
            int gz=(int)((st.teammates[i].position.z/MAP_SIZE+0.5f)*GRID_RES);
            gx=std::max(0,std::min(GRID_RES-1,gx)); gz=std::max(0,std::min(GRID_RES-1,gz));
            tH[gx*GRID_RES+gz]+=1.0f;
        }
        if(st.numTeammates>0)for(int i=0;i<256;i++)tH[i]/=st.numTeammates;
        
        float totT=0;
        for(int i=0;i<st.numEnemies&&i<MAX_ENEMIES;i++)if(st.enemies[i].active){
            float ex=st.enemies[i].position.x,ez=st.enemies[i].position.z;
            float d=std::sqrt(ex*ex+ez*ez),a=std::atan2(ez,ex),w=1.0f/(1.0f+d/20.0f);
            int sec=(int)((a+3.14159f)/(2*3.14159f)*8)%8; thr[sec]+=w; totT+=w;
        }
        if(totT>0)for(int i=0;i<8;i++)thr[i]/=totT;
        
        gF[0]=st.gameTime; gF[1]=st.scoreDiff; gF[2]=(float)st.numEnemies/MAX_ENEMIES;
        gF[3]=(float)st.numTeammates/MAX_TEAMMATES; gF[4]=st.objectives.a; gF[5]=st.objectives.b;
        gF[6]=st.objectives.c; gF[7]=(st.objectives.a+st.objectives.b+st.objectives.c>0)?1.0f:0.0f;
        
        std::memcpy(out,eH,256*sizeof(float)); std::memcpy(out+256,tH,256*sizeof(float));
        std::memcpy(out+512,thr,8*sizeof(float)); std::memcpy(out+520,gF,8*sizeof(float));
    }
    
    void forward(const float* gs, float* scores) const {
        const int H=256; float t1[H],t2[H],gE[H],sE[H],comb[512];
        
        linear(gs,t1,g_enc_0_weight,g_enc_0_bias,528,H); layer_norm(t1,g_enc_1_weight,g_enc_1_bias,H); gelu_inplace(t1,H);
        linear(t1,gE,g_enc_3_weight,g_enc_3_bias,H,H); layer_norm(gE,g_enc_4_weight,g_enc_4_bias,H); gelu_inplace(gE,H);
        
        for(int s=0;s<MAX_SPAWNS;s++){
            float sIn[270]; std::memcpy(sIn,&STATIC_FEATURES[s*14],14*sizeof(float));
            std::memcpy(sIn+14,&VISIBILITY_FEATURES[s*256],256*sizeof(float));
            
            linear(sIn,t1,s_enc_0_weight,s_enc_0_bias,270,H); layer_norm(t1,s_enc_1_weight,s_enc_1_bias,H); gelu_inplace(t1,H);
            linear(t1,sE,s_enc_3_weight,s_enc_3_bias,H,H); layer_norm(sE,s_enc_4_weight,s_enc_4_bias,H); gelu_inplace(sE,H);
            
            std::memcpy(comb,gE,H*sizeof(float)); std::memcpy(comb+H,sE,H*sizeof(float));
            
            linear(comb,t1,scorer_0_weight,scorer_0_bias,512,H); layer_norm(t1,scorer_1_weight,scorer_1_bias,H); gelu_inplace(t1,H);
            
            resblock(t1,t2,scorer_3_net_0_weight,scorer_3_net_0_bias,scorer_3_net_1_weight,scorer_3_net_1_bias,
                     scorer_3_net_4_weight,scorer_3_net_4_bias,scorer_3_net_5_weight,scorer_3_net_5_bias,H);
            resblock(t1,t2,scorer_4_net_0_weight,scorer_4_net_0_bias,scorer_4_net_1_weight,scorer_4_net_1_bias,
                     scorer_4_net_4_weight,scorer_4_net_4_bias,scorer_4_net_5_weight,scorer_4_net_5_bias,H);
            resblock(t1,t2,scorer_5_net_0_weight,scorer_5_net_0_bias,scorer_5_net_1_weight,scorer_5_net_1_bias,
                     scorer_5_net_4_weight,scorer_5_net_4_bias,scorer_5_net_5_weight,scorer_5_net_5_bias,H);
            resblock(t1,t2,scorer_6_net_0_weight,scorer_6_net_0_bias,scorer_6_net_1_weight,scorer_6_net_1_bias,
                     scorer_6_net_4_weight,scorer_6_net_4_bias,scorer_6_net_5_weight,scorer_6_net_5_bias,H);
            
            float h128[128]; linear(t1,h128,scorer_7_weight,scorer_7_bias,H,128);
            layer_norm(h128,scorer_8_weight,scorer_8_bias,128); gelu_inplace(h128,128);
            
            float logit=scorer_10_bias[0]; for(int i=0;i<128;i++)logit+=h128[i]*scorer_10_weight[i];
            scores[s]=sigmoid(logit);
        }
    }
};

Engine::Engine():impl_(new Impl()){}
Engine::~Engine(){delete impl_;}
bool Engine::Initialize(const char* k){if(!k||strlen(k)<8)return false;impl_->ready=true;return true;}
bool Engine::IsReady()const{return impl_->ready;}

ErrorCode Engine::SelectSpawn(const GameState& st, SpawnResult& r) const {
    if(!impl_->ready)return ErrorCode::InvalidLicense;
    impl_->queries++;
    float gs[528]; impl_->encode(st,gs); impl_->forward(gs,r.scores);
    float mx=-1; for(int i=0;i<MAX_SPAWNS;i++)if(r.scores[i]>mx){mx=r.scores[i];r.bestSpawnIdx=i;}
    r.confidence=mx;
    uint8_t idx[MAX_SPAWNS]; for(int i=0;i<MAX_SPAWNS;i++)idx[i]=i;
    for(int i=0;i<MAX_SPAWNS-1;i++)for(int j=i+1;j<MAX_SPAWNS;j++)
        if(r.scores[idx[j]]>r.scores[idx[i]]){uint8_t t=idx[i];idx[i]=idx[j];idx[j]=t;}
    for(int rk=0;rk<MAX_SPAWNS;rk++)r.rankings[idx[rk]]=rk;
    r.bestTier=GetTierForRank(r.rankings[r.bestSpawnIdx]);
    return ErrorCode::Success;
}

Vec3 Engine::GetSpawnPosition(uint8_t i)const{return i<MAX_SPAWNS?impl_->spawnPos[i]:Vec3();}
SpawnTier Engine::GetTierForRank(uint8_t r){
    if(r<TIER_EXCELLENT)return SpawnTier::Excellent;
    if(r<TIER_GOOD)return SpawnTier::Good;
    if(r<TIER_ACCEPTABLE)return SpawnTier::Acceptable;
    return SpawnTier::Bad;
}

const char* SpawnTierToString(SpawnTier t){
    switch(t){case SpawnTier::Excellent:return"Excellent";case SpawnTier::Good:return"Good";
              case SpawnTier::Acceptable:return"Acceptable";default:return"Bad";}
}

}} // namespace
