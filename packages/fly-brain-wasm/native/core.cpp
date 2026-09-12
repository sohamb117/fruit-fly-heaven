// Generic, environment-independent LIF connectome runtime.
// No fruit, body, neuron names, behavioral rules, or dataset-specific sizes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fb {
#ifdef FB_FLOAT32
using State=float;
constexpr double GAIN_MARGIN=1e-5, THRESHOLD_MARGIN=1e-4;
#else
using State=double;
constexpr double GAIN_MARGIN=1e-12, THRESHOLD_MARGIN=1e-10;
#endif
constexpr uint32_t NONE=UINT32_MAX, WHEEL=4096;
thread_local std::string error;
struct Config {
  double dt=.1, tauM=20, tauS=5, rest=-52, threshold=-45, reset=-52;
  double refractory=2.2, delay=1.8, weightScale=.275;
  uint32_t history=8192;
  void validate() const {
    const double vals[]={dt,tauM,tauS,rest,threshold,reset,refractory,delay,weightScale};
    for(auto v:vals) if(!std::isfinite(v)) throw std::runtime_error("Non-finite model parameter");
    if(dt<=0 || dt>10 || tauS<=0 || tauM<=tauS || threshold<=rest || reset>=threshold || refractory<0 || delay<dt || delay>1000 || refractory>1000 || history<1 || history>1048576)
      throw std::runtime_error("Invalid LIF configuration");
    for(double ms:{refractory,delay}) if(std::abs(ms/dt-std::round(ms/dt))>1e-7)
      throw std::runtime_error("Delay and refractory period must be multiples of dtMs");
  }
};
struct Graph {
  uint32_t n;
  std::vector<uint32_t> row, col;
  std::vector<float> weight;
  // Lossless 18-bit target / signed 14-bit weight encoding. Arbitrary graphs
  // retain the ordinary CSR representation when any edge does not fit.
  std::vector<uint32_t> packed;
  bool compact=false;
  Graph(uint32_t n,uint32_t e,const uint32_t* r,const uint32_t* c,const float* w):n(n) {
    if(!n || !r || (!c && e) || (!w && e)) throw std::runtime_error("Invalid graph buffers");
    if(r[0]!=0 || r[n]!=e || !std::is_sorted(r,r+n+1)) throw std::runtime_error("Invalid CSR offsets");
    for(uint32_t i=0;i<e;i++) if(c[i]>=n || !std::isfinite(w[i])) throw std::runtime_error("Invalid edge");
    row.assign(r,r+n+1);
    compact=n<=(1u<<18);
    for(uint32_t i=0;compact && i<e;i++)compact=w[i]>=-8192 && w[i]<=8191 && std::trunc(w[i])==w[i];
    if(compact){
      packed.resize(e);
      for(uint32_t i=0;i<e;i++)packed[i]=c[i]|(static_cast<uint32_t>(static_cast<int32_t>(w[i]))<<18);
    }else{col.assign(c,c+e);weight.assign(w,w+e);}
  }
};
using GraphHandle=std::shared_ptr<const Graph>;
struct Spike {double ms;uint32_t neuron;};
struct Input {uint32_t neuron; double rate, amplitude, logP;uint64_t rng,due;};
class Brain {
public:
  GraphHandle graph; Config config;
  struct alignas(sizeof(State)==8?64:8) Neuron {
    State v=0,g=0,arriving=0;
    uint64_t last=0,refractory=0,when=UINT64_MAX;
    uint32_t previous=NONE,next=NONE,refractoryTicks=0;
    uint8_t marked=0;
  };
  static_assert(sizeof(Neuron)<=64,"Hot neuron state must fit within 64 bytes");
  std::vector<Neuron> neuron;
  std::vector<State> current;
  std::vector<uint32_t> head,touched;
  std::vector<uint64_t> counts;
  std::vector<std::vector<uint32_t>> delayed;
  std::vector<Input> inputs;
  std::vector<Spike> history;
  std::vector<State> em,es;
  uint64_t tick=0,totalSpikes=0,rng;
  uint32_t delayTicks;
  State aFactor,threshold,maxSynapticGain;

  Brain(GraphHandle graph,Config config,uint64_t seed):graph(graph),config(config),rng(seed) {
    config.validate();auto n=graph->n;
    neuron.resize(n);head.resize(WHEEL,NONE);
    for(auto& cell:neuron){cell.v=config.reset-config.rest;cell.refractoryTicks=std::llround(config.refractory/config.dt);}
    counts.resize(n);history.resize(config.history);
    delayTicks=std::llround(config.delay/config.dt);delayed.resize(delayTicks+1);
    aFactor=config.tauS/(config.tauM-config.tauS);threshold=config.threshold-config.rest;
    // Continuous-time upper bound on the positive synaptic impulse response.
    // Inflate it to keep the rejection conservative near floating-point limits.
    double peakMs=std::log(config.tauM/config.tauS)/(1/config.tauS-1/config.tauM);
    maxSynapticGain=aFactor*(std::exp(-peakMs/config.tauM)-std::exp(-peakMs/config.tauS));
    maxSynapticGain+=GAIN_MARGIN*(1+std::abs(aFactor));
    if(!std::isfinite(aFactor)||!std::isfinite(threshold)||!std::isfinite(static_cast<State>(config.reset-config.rest)))throw std::runtime_error("Parameters exceed state precision range");
    em.resize(10001);es.resize(10001);
    for(int i=0;i<=10000;i++){em[i]=std::exp(-i*config.dt/config.tauM);es[i]=std::exp(-i*config.dt/config.tauS);}
  }
  double random(){rng+=0x9e3779b97f4a7c15ULL;uint64_t z=rng;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return (((z^(z>>31))>>11)+.5)*0x1.0p-53;}
  uint64_t wait(Input& s){
    s.rng+=0x9e3779b97f4a7c15ULL;uint64_t z=s.rng;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;
    double u=(((z^(z>>31))>>11)+.5)*0x1.0p-53;
    return static_cast<uint64_t>(std::min(static_cast<double>(UINT64_MAX/4),std::floor(std::log(u)/s.logP)));
  }
  State expM(uint64_t t) const {return t<=10000?em[t]:std::exp(-static_cast<double>(t)*config.dt/config.tauM);}
  State expS(uint64_t t) const {return t<=10000?es[t]:std::exp(-static_cast<double>(t)*config.dt/config.tauS);}
  State drive(uint32_t i) const {return current.empty()?0:current[i];}
  State voltageAt(uint32_t i,uint64_t delta) const {
    State a=expM(delta),b=expS(delta);
    return drive(i)+(neuron[i].v-drive(i))*a+neuron[i].g*aFactor*(a-b);
  }
  void evolve(uint32_t i){
    auto start=std::max(neuron[i].last,neuron[i].refractory);
    if(tick>start){auto delta=tick-start;neuron[i].v=voltageAt(i,delta);neuron[i].g*=expS(delta);}
    neuron[i].last=tick;
  }
  void cancel(uint32_t i){
    if(neuron[i].when==UINT64_MAX)return;
    if(neuron[i].previous==NONE)head[neuron[i].when%WHEEL]=neuron[i].next;else neuron[neuron[i].previous].next=neuron[i].next;
    if(neuron[i].next!=NONE)neuron[neuron[i].next].previous=neuron[i].previous;
    neuron[i].previous=neuron[i].next=NONE;neuron[i].when=UINT64_MAX;
  }
  void enqueue(uint32_t i,uint64_t t){
    auto slot=t%WHEEL;neuron[i].when=t;neuron[i].next=head[slot];neuron[i].previous=NONE;
    if(head[slot]!=NONE)neuron[head[slot]].previous=i;head[slot]=i;
  }
  void schedule(uint32_t i){
    auto prior=neuron[i].when;cancel(i);auto start=std::max(tick,neuron[i].refractory);
    if(neuron[i].v>threshold && start==tick){enqueue(i,tick);return;}
    uint64_t hi=1;
    if(drive(i)>threshold){
      // The constant drive eventually crosses threshold, even after inhibition.
      while(voltageAt(i,hi)<=threshold && hi<(1ULL<<40))hi*=2;
      if(voltageAt(i,hi)<=threshold)return;
    }else{
      if(neuron[i].g<=0 || drive(i)+neuron[i].g<=neuron[i].v || std::max(neuron[i].v,drive(i))+neuron[i].g*aFactor<=threshold)return;
      if(std::max(neuron[i].v,drive(i))+neuron[i].g*maxSynapticGain<threshold-THRESHOLD_MARGIN)return;
      // Reuse a still-valid upper bound before calculating the analytic peak.
      // Both checks preserve the first crossing on the same discrete time grid.
      if(voltageAt(i,1)>threshold){enqueue(i,start+1);return;}
      if(prior!=UINT64_MAX && prior>start && voltageAt(i,prior-start)>threshold)hi=prior-start;
      else {
        double a=neuron[i].v-drive(i)+neuron[i].g*aFactor;
        if(a<=0)return;
        double peak=std::log(neuron[i].g*aFactor*config.tauM/(a*config.tauS))/((1/config.tauS-1/config.tauM)*config.dt);
        if(!std::isfinite(peak) || peak<=0 || peak>1e12)return;
        hi=std::max<uint64_t>(1,std::ceil(peak));
        if(hi>1 && voltageAt(i,hi-1)>voltageAt(i,hi))hi--;
        if(voltageAt(i,hi)<=threshold)return;
      }
    }
    uint64_t lo=1;
    while(lo<hi){auto mid=lo+(hi-lo)/2;if(voltageAt(i,mid)>threshold)hi=mid;else lo=mid+1;}
    enqueue(i,start+lo);
  }
  void mark(uint32_t i){if(!neuron[i].marked){neuron[i].marked=1;touched.push_back(i);}}
  void checkIndex(uint32_t i) const {if(i>=graph->n)throw std::runtime_error("Neuron index outside graph");}
  void setInputs(uint32_t n,const uint32_t* indices,const float* rates,const float* amplitudes){
    std::vector<Input> replacement;replacement.reserve(n);
    for(uint32_t k=0;k<n;k++){
      checkIndex(indices[k]);double rate=rates[k],amp=amplitudes[k];
      if(!std::isfinite(rate)||!std::isfinite(amp)||rate<0||rate*config.dt/1000>=1)throw std::runtime_error("Invalid Poisson input");
      if(rate>0)replacement.push_back({indices[k],rate,amp,std::log1p(-rate*config.dt/1000),rng^((uint64_t(indices[k])+1)*0xd1342543de82ef95ULL),tick});
    }
    std::sort(replacement.begin(),replacement.end(),[](const Input&a,const Input&b){return a.neuron<b.neuron;});
    for(size_t k=0;k<replacement.size();k++){
      auto& s=replacement[k];
      if(k && replacement[k-1].neuron==s.neuron)throw std::runtime_error("Duplicate Poisson input index");
      auto old=std::lower_bound(inputs.begin(),inputs.end(),s.neuron,[](const Input&a,uint32_t i){return a.neuron<i;});
      if(old!=inputs.end() && old->neuron==s.neuron){
        if(old->rate==s.rate && old->amplitude==s.amplitude){s=*old;continue;}
        s.rng=old->rng;
      }
      s.due=tick+wait(s);
    }
    inputs=std::move(replacement);
  }
  void setCurrents(uint32_t n,const uint32_t* ids,const float* values){
    for(uint32_t k=0;k<n;k++){checkIndex(ids[k]);if(!std::isfinite(values[k]))throw std::runtime_error("Non-finite current");}
    if(n && current.empty())current.resize(graph->n);
    for(uint32_t k=0;k<n;k++){auto i=ids[k];evolve(i);current[i]=values[k];schedule(i);}
  }
  void inject(uint32_t n,const uint32_t* ids,const float* values){
    for(uint32_t k=0;k<n;k++){checkIndex(ids[k]);if(!std::isfinite(values[k]))throw std::runtime_error("Non-finite pulse");}
    for(uint32_t k=0;k<n;k++){auto i=ids[k];if(tick<neuron[i].refractory)continue;evolve(i);neuron[i].v+=values[k];schedule(i);}
  }
  void setRefractory(uint32_t n,const uint32_t* ids,double ms){
    if(!std::isfinite(ms)||ms<0||ms>1000||std::abs(ms/config.dt-std::round(ms/config.dt))>1e-7)throw std::runtime_error("Invalid refractory period");
    for(uint32_t k=0;k<n;k++)checkIndex(ids[k]);
    for(uint32_t k=0;k<n;k++)neuron[ids[k]].refractoryTicks=std::llround(ms/config.dt);
  }
  template<bool Compact> void deliver(const std::vector<uint32_t>& pending){
    for(auto i:pending)for(uint32_t e=graph->row[i];e<graph->row[i+1];e++){
      uint32_t edge=Compact?graph->packed[e]:0;
      uint32_t j=Compact?(edge&((1u<<18)-1)):graph->col[e];
      if(tick<neuron[j].refractory)continue;
      State weight=Compact?static_cast<int32_t>(edge)>>18:graph->weight[e];
      neuron[j].arriving+=static_cast<State>(config.weightScale)*weight;mark(j);
    }
  }
  void step(uint32_t steps){
    if(steps>1000000)throw std::runtime_error("Step block exceeds 1,000,000 ticks");
    if(!steps)return;
    std::vector<std::vector<uint32_t>> inputEvents(steps);
    for(uint32_t k=0;k<inputs.size();k++){
      auto& s=inputs[k];
      while(s.due<tick+steps){inputEvents[s.due-tick].push_back(k);s.due+=1+wait(s);}
    }
    for(uint32_t t=0;t<steps;t++,tick++){
      auto& pending=delayed[tick%delayed.size()];
      if(graph->compact)deliver<true>(pending);else deliver<false>(pending);
      pending.clear();
      for(auto i:touched){evolve(i);neuron[i].g+=neuron[i].arriving;neuron[i].arriving=0;}
      for(auto k:inputEvents[t]){const auto& input=inputs[k];auto i=input.neuron;if(tick<neuron[i].refractory)continue;if(!neuron[i].marked)evolve(i);neuron[i].v+=input.amplitude;mark(i);}
      for(auto i:touched){schedule(i);neuron[i].marked=0;}touched.clear();
      auto i=head[tick%WHEEL];
      while(i!=NONE){
        auto nextIndex=neuron[i].next;
        if(neuron[i].when==tick){
          cancel(i);evolve(i);
          if(neuron[i].v>threshold && tick>=neuron[i].refractory){
            neuron[i].v=config.reset-config.rest;neuron[i].g=0;neuron[i].refractory=tick+neuron[i].refractoryTicks;
            delayed[(tick+delayTicks)%delayed.size()].push_back(i);
            counts[i]++;history[totalSpikes%history.size()]={tick*config.dt,i};totalSpikes++;
            // Tonic current can cause another spike without a new synaptic event.
            schedule(i);
          }else schedule(i);
        }
        i=nextIndex;
      }
    }
  }
  double field(uint32_t i,uint32_t kind) const {
    auto start=std::max(neuron[i].last,neuron[i].refractory);auto elapsed=tick>start?tick-start:0;
    if(kind==0)return config.rest+voltageAt(i,elapsed);
    if(kind==1)return neuron[i].g*expS(elapsed);
    if(kind==2)return counts[i];
    throw std::runtime_error("Unknown state field");
  }
};
}

extern "C" {
uint32_t fb_precision_bits(){return sizeof(fb::State)*8;}
const char* fb_error(){return fb::error.c_str();}
void* fb_graph_create(uint32_t n,uint32_t e,const uint32_t* r,const uint32_t* c,const float* w){try{return new fb::GraphHandle(std::make_shared<fb::Graph>(n,e,r,c,w));}catch(const std::exception& e){fb::error=e.what();return nullptr;}}
void fb_graph_destroy(void* p){delete static_cast<fb::GraphHandle*>(p);}
void* fb_brain_create(void* p,uint32_t seedLo,uint32_t seedHi,const double* params,uint32_t history){try{
  fb::Config c; c.dt=params[0];c.tauM=params[1];c.tauS=params[2];c.rest=params[3];c.threshold=params[4];c.reset=params[5];c.refractory=params[6];c.delay=params[7];c.weightScale=params[8];c.history=history;
  return new fb::Brain(*static_cast<fb::GraphHandle*>(p),c,(static_cast<uint64_t>(seedHi)<<32)|seedLo);
}catch(const std::exception& e){fb::error=e.what();return nullptr;}}
void fb_brain_destroy(void* p){delete static_cast<fb::Brain*>(p);}
int fb_inputs(void* p,uint32_t n,const uint32_t* i,const float* r,const float* a){try{static_cast<fb::Brain*>(p)->setInputs(n,i,r,a);return 0;}catch(const std::exception& e){fb::error=e.what();return -1;}}
int fb_currents(void* p,uint32_t n,const uint32_t* i,const float* v){try{static_cast<fb::Brain*>(p)->setCurrents(n,i,v);return 0;}catch(const std::exception& e){fb::error=e.what();return -1;}}
int fb_inject(void* p,uint32_t n,const uint32_t* i,const float* v){try{static_cast<fb::Brain*>(p)->inject(n,i,v);return 0;}catch(const std::exception& e){fb::error=e.what();return -1;}}
int fb_refractory(void* p,uint32_t n,const uint32_t* i,double ms){try{static_cast<fb::Brain*>(p)->setRefractory(n,i,ms);return 0;}catch(const std::exception& e){fb::error=e.what();return -1;}}
int fb_step(void* p,uint32_t ticks){try{static_cast<fb::Brain*>(p)->step(ticks);return 0;}catch(const std::exception& e){fb::error=e.what();return -1;}}
int fb_read(void* p,uint32_t field,uint32_t n,const uint32_t* ids,double* values){try{const auto& b=*static_cast<fb::Brain*>(p);for(uint32_t k=0;k<n;k++){uint32_t i=ids?ids[k]:k;b.checkIndex(i);values[k]=b.field(i,field);}return 0;}catch(const std::exception& e){fb::error=e.what();return -1;}}
double fb_time(void* p){auto& b=*static_cast<fb::Brain*>(p);return b.tick*b.config.dt;}
double fb_spike_total(void* p){return static_cast<fb::Brain*>(p)->totalSpikes;}
uint32_t fb_spikes(void* p,double* times,uint32_t* ids){auto& b=*static_cast<fb::Brain*>(p);auto n=std::min<uint64_t>(b.totalSpikes,b.history.size());for(uint32_t j=0;j<n;j++){auto s=b.history[(b.totalSpikes-n+j)%b.history.size()];times[j]=s.ms;ids[j]=s.neuron;}return n;}
}
