#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>

struct Edge { uint32_t source; float weight; uint32_t receptor, delay; };
static float clamp(float x, float a, float b) { return std::max(a, std::min(b,x)); }

// Float32 conductance model. Units: mV, ms, pF, nS, pA. Layout is shared with WGSL.
extern "C" void br_step(uint32_t n, uint32_t tick, uint32_t slots, uint32_t gaps_on,
  float dt, float hunger, float insulin, float akh, const uint32_t* offsets,
  const Edge* edges, const float* params, const float* old, float* next,
  float* history, float* kinetics, uint32_t* events) {
  const float* receptors = params + n*17;
  // These factors and delayed row addresses are identical for every cell in
  // this tick. Preserve the original expressions and incoming-edge order.
  float rise[9], decay[9];
  for(uint32_t r=0;r<9;++r) {
    rise[r]=1-std::exp(-dt/receptors[r*3]);
    decay[r]=1-std::exp(-dt/receptors[r*3+1]);
  }
  const float* delayed[32];
  for(uint32_t delay=0;delay<slots;++delay)
    delayed[delay]=history+((tick+slots-delay)%slots)*n;
  for(uint32_t i=0; i<n; ++i) {
    const float* p=params+i*16; const float* s=old+i*8; float* out=next+i*8;
    float drive[9]={0};
    for(uint32_t e=offsets[i];e<offsets[i+1];++e) {
      const Edge& edge=edges[e];
      drive[edge.receptor]+=edge.weight*delayed[edge.delay][edge.source];
    }
    float conductance=p[1], reversal=p[1]*p[2], mod[4]={0};
    for(uint32_t r=0;r<9;++r) {
      float* gate=kinetics+(i*9+r)*2;
      gate[0]+=(drive[r]-gate[0])*rise[r];
      gate[1]+=(gate[0]-gate[1])*decay[r];
      if(r<5) { conductance+=gate[1]; reversal+=gate[1]*receptors[r*3+2]; }
      else mod[r-5]=gate[1]/(1+gate[1]);
    }
    if(gaps_on) for(uint32_t e=offsets[n+1+i];e<offsets[n+2+i];++e) {
      conductance+=edges[e].weight; reversal+=edges[e].weight*old[edges[e].source*8];
    }
    float gain=clamp(1+p[12]*mod[0]+p[13]*(mod[1]-.5f*mod[3])+p[14]*mod[2],.25f,4);
    float adaptation=s[1]*std::exp(-dt/p[6]);
    float current=(params[n*16+i]+p[11])*gain+p[15]*(hunger+.3f*akh-.3f*insulin)-adaptation;
    float v=clamp((s[0]+dt*(reversal+current)/p[0])/(1+dt*conductance/p[0]),-100,60);
    float refractory=std::max(0.f,s[2]-dt), spike=0, release=0;
    if(p[8]>.5f) release=.1f/(1+std::exp(-(v-p[9])/p[10]));
    else {
      if(s[2]>0) v=p[4];
      else if(v>=p[3]) { spike=1; v=p[4]; refractory=p[5]; adaptation+=p[7]; }
      release=spike/dt;
    }
    out[0]=v; out[1]=adaptation; out[2]=refractory; out[3]=s[3]+spike;
    out[4]=s[4]+(release*1000-s[4])*(1-std::exp(-dt/50));
    out[5]=release; out[6]=conductance; out[7]=current;
    if(spike>0){
      const float time=(tick+1)*dt;kinetics[n*18+i]=time;
      const uint32_t slot=events[0]++%16384;events[1]=std::min(16384u,events[1]+1);
      std::memcpy(events+2+slot*2,&time,sizeof(float));events[3+slot*2]=i;
    }
    history[(tick%slots)*n+i]=release;
  }
}

// Activation, fatigue and a reduced Hill-type contractile element. SI time.
// Input stride 5: excitation, normalized length, positive shortening velocity,
// Fmax, energy. Shortening velocity is the negative length derivative.
// State stride 3: activation, fatigue, force. No neural writes or behavior program.
extern "C" void muscle_step(uint32_t n, float dt, const float* input, float* state) {
  for(uint32_t i=0;i<n;++i) {
    const float* x=input+5*i; float* s=state+3*i;
    float target=clamp(x[0],0,1), tau=target>s[0]?.015f:.04f;
    s[0]+=(target-s[0])*(1-std::exp(-dt/tau));
    s[1]=clamp(s[1]+dt*(.08f*s[0]-.03f*(1-s[0])),0,.8f);
    float length=std::exp(-std::pow((x[1]-1)/.45f,2));
    float velocity=clamp(1-.25f*x[2],.1f,1.8f);
    s[2]=std::max(0.f,x[3])*s[0]*(1-s[1])*length*velocity*clamp(x[4],0,1);
  }
}

// Passive joint mechanics distilled from the reduced FlyBody teacher.
// Parameter stride 6: inertia, damping, stiffness, neutral, min, max.
extern "C" void joint_step(uint32_t n, float dt, const float* params, const float* torque, float* state) {
  for(uint32_t i=0;i<n;i++) {
    const float* p=params+6*i; float* s=state+2*i;
    float velocity=(s[1]+dt*(torque[i]-p[2]*(s[0]-p[3]))/p[0])/(1+dt*p[1]/p[0]+dt*dt*p[2]/p[0]);
    float q=s[0]+dt*velocity;
    s[0]=clamp(q,p[4],p[5]);s[1]=q==s[0]?velocity:0;
  }
}

#include "dlm-ionic.h"

// Opt-in path; legacy br_step above remains unchanged.
extern "C" int br_step_dlm(uint32_t n, uint32_t tick, uint32_t slots, uint32_t gaps_on,
  float dt, float hunger, float insulin, float akh, const uint32_t* offsets,
  const Edge* edges, const float* params, const float* old, float* next,
  float* history, float* kinetics, uint32_t* events) {
  const float* receptors = params + n*17;
  float rise[9], decay[9];
  for(uint32_t r=0;r<9;++r) {
    rise[r]=1-std::exp(-dt/receptors[r*3]);
    decay[r]=1-std::exp(-dt/receptors[r*3+1]);
  }
  const float* delayed[32];
  for(uint32_t delay=0;delay<slots;++delay)
    delayed[delay]=history+((tick+slots-delay)%slots)*n;
  for(uint32_t i=0; i<n; ++i) {
    const float* p=params+i*16; const float* s=old+i*8; float* out=next+i*8;
    float drive[9]={0};
    for(uint32_t e=offsets[i];e<offsets[i+1];++e) {
      const Edge& edge=edges[e];
      drive[edge.receptor]+=edge.weight*delayed[edge.delay][edge.source];
    }
    float conductance=p[1], reversal=p[1]*p[2], mod[4]={0};
    for(uint32_t r=0;r<9;++r) {
      float* gate=kinetics+(i*9+r)*2;
      gate[0]+=(drive[r]-gate[0])*rise[r];
      gate[1]+=(gate[0]-gate[1])*decay[r];
      if(r<5) { conductance+=gate[1]; reversal+=gate[1]*receptors[r*3+2]; }
      else mod[r-5]=gate[1]/(1+gate[1]);
    }
    if(gaps_on) for(uint32_t e=offsets[n+1+i];e<offsets[n+2+i];++e) {
      conductance+=edges[e].weight; reversal+=edges[e].weight*old[edges[e].source*8];
    }
    float gain=clamp(1+p[12]*mod[0]+p[13]*(mod[1]-.5f*mod[3])+p[14]*mod[2],.25f,4);
    float adaptation=s[1]*std::exp(-dt/p[6]);
    float current=(params[n*16+i]+p[11])*gain+p[15]*(hunger+.3f*akh-.3f*insulin)-adaptation;
    float v=0,refractory=0,spike=0,release=0,time=(tick+1)*dt;
    const uint32_t slot_id=static_cast<uint32_t>(params[n*17+27+i]);
    if(slot_id){
      float* intrinsic=kinetics+n*19+(slot_id-1)*4;
      DlmState x={s[0],intrinsic[0],intrinsic[1]};
      if(intrinsic[3]!=0||!dlm_valid(x)||!std::isfinite(intrinsic[2])||intrinsic[2]<0||intrinsic[2]>100||std::floor(intrinsic[2])!=intrinsic[2]){intrinsic[3]=1;return i+1;}
      uint32_t guard=static_cast<uint32_t>(intrinsic[2]);
      // Remove the LIF adaptation term, preserving real input / modulation.
      current=(params[n*16+i]+p[11])*gain+p[15]*(hunger+.3f*akh-.3f*insulin);
      float g[5];for(uint32_t r=0;r<5;r++)g[r]=kinetics[(i*9+r)*2+1];
      for(uint32_t sub=0;sub<5;sub++){
        x=dlm_rk4(x,current,g,receptors);
        if(!dlm_valid(x)){intrinsic[3]=1;return i+1;}
        if(guard>0)--guard;
        if(x.v>-10.f&&guard==0){spike=1;guard=100;time=static_cast<float>(tick*5+sub+1)/10.f;}
      }
      intrinsic[0]=x.h;intrinsic[1]=x.b;intrinsic[2]=guard;
      v=x.v;adaptation=0;refractory=0;release=spike/dt;conductance=dlm_conductance(x,g);
    }else{
    v=clamp((s[0]+dt*(reversal+current)/p[0])/(1+dt*conductance/p[0]),-100,60);
    refractory=std::max(0.f,s[2]-dt);
    if(p[8]>.5f) release=.1f/(1+std::exp(-(v-p[9])/p[10]));
    else {
      if(s[2]>0) v=p[4];
      else if(v>=p[3]) { spike=1; v=p[4]; refractory=p[5]; adaptation+=p[7]; }
      release=spike/dt;
    }
    }
    out[0]=v; out[1]=adaptation; out[2]=refractory; out[3]=s[3]+spike;
    out[4]=s[4]+(release*1000-s[4])*(1-std::exp(-dt/50));
    out[5]=release; out[6]=conductance; out[7]=current;
    if(spike>0){
      kinetics[n*18+i]=time;
      const uint32_t slot=events[0]++%16384;events[1]=std::min(16384u,events[1]+1);
      std::memcpy(events+2+slot*2,&time,sizeof(float));events[3+slot*2]=i;
    }
    history[(tick%slots)*n+i]=release;
  }
  return 0;
}
