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
  for(uint32_t i=0; i<n; ++i) {
    const float* p=params+i*16; const float* s=old+i*8; float* out=next+i*8;
    float drive[9]={0};
    for(uint32_t e=offsets[i];e<offsets[i+1];++e) {
      const Edge& edge=edges[e];
      drive[edge.receptor]+=edge.weight*history[((tick+slots-edge.delay)%slots)*n+edge.source];
    }
    float conductance=p[1], reversal=p[1]*p[2], mod[4]={0};
    for(uint32_t r=0;r<9;++r) {
      float* gate=kinetics+(i*9+r)*2;
      gate[0]+=(drive[r]-gate[0])*(1-std::exp(-dt/receptors[r*3]));
      gate[1]+=(gate[0]-gate[1])*(1-std::exp(-dt/receptors[r*3+1]));
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
