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
  Graph(uint32_t n,uint32_t e,const uint32_t* r,const uint32_t* c,const float* w):n(n) {
    if(!n || !r || (!c && e) || (!w && e)) throw std::runtime_error("Invalid graph buffers");
    if(r[0]!=0 || r[n]!=e || !std::is_sorted(r,r+n+1)) throw std::runtime_error("Invalid CSR offsets");
    for(uint32_t i=0;i<e;i++) if(c[i]>=n || !std::isfinite(w[i])) throw std::runtime_error("Invalid edge");
    row.assign(r,r+n+1);col.assign(c,c+e);weight.assign(w,w+e);
  }
};
using GraphHandle=std::shared_ptr<const Graph>;
struct Spike {double ms;uint32_t neuron;};
struct Input {uint32_t neuron; double rate, amplitude, logP;uint64_t rng,due;};
class Brain {
public:
  GraphHandle graph; Config config;
  std::vector<double> v,g,current,arriving;
  std::vector<uint64_t> last,refractory,when;
  std::vector<uint32_t> refractoryTicks,previous,next,head,touched;
  std::vector<uint64_t> counts;
  std::vector<uint8_t> marked;
  std::vector<std::vector<uint32_t>> delayed;
  std::vector<Input> inputs;
  std::vector<Spike> history;
  std::vector<double> em,es;
  uint64_t tick=0,totalSpikes=0,rng;
  uint32_t delayTicks;
  double aFactor,threshold;

  Brain(GraphHandle graph,Config config,uint64_t seed):graph(graph),config(config),rng(seed) {
    config.validate();auto n=graph->n;
    v.resize(n,config.reset-config.rest);g.resize(n);current.resize(n);arriving.resize(n);
    last.resize(n);refractory.resize(n);when.resize(n,UINT64_MAX);
    refractoryTicks.resize(n,std::llround(config.refractory/config.dt));
    previous.resize(n,NONE);next.resize(n,NONE);head.resize(WHEEL,NONE);
    marked.resize(n);counts.resize(n);history.resize(config.history);
    delayTicks=std::llround(config.delay/config.dt);delayed.resize(delayTicks+1);
    aFactor=config.tauS/(config.tauM-config.tauS);threshold=config.threshold-config.rest;
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
  double expM(uint64_t t) const {return t<=10000?em[t]:std::exp(-t*config.dt/config.tauM);}
  double expS(uint64_t t) const {return t<=10000?es[t]:std::exp(-t*config.dt/config.tauS);}
  double voltageAt(uint32_t i,uint64_t delta) const {
    double a=expM(delta),b=expS(delta);
    return current[i]+(v[i]-current[i])*a+g[i]*aFactor*(a-b);
  }
  void evolve(uint32_t i){
    auto start=std::max(last[i],refractory[i]);
    if(tick>start){auto delta=tick-start;v[i]=voltageAt(i,delta);g[i]*=expS(delta);}
    last[i]=tick;
  }
  void cancel(uint32_t i){
    if(when[i]==UINT64_MAX)return;
    if(previous[i]==NONE)head[when[i]%WHEEL]=next[i];else next[previous[i]]=next[i];
    if(next[i]!=NONE)previous[next[i]]=previous[i];
    previous[i]=next[i]=NONE;when[i]=UINT64_MAX;
  }
  void enqueue(uint32_t i,uint64_t t){
    auto slot=t%WHEEL;when[i]=t;next[i]=head[slot];previous[i]=NONE;
    if(head[slot]!=NONE)previous[head[slot]]=i;head[slot]=i;
  }
  void schedule(uint32_t i){
    cancel(i);auto start=std::max(tick,refractory[i]);
    if(v[i]>threshold && start==tick){enqueue(i,tick);return;}
    uint64_t hi=1;
    if(current[i]>threshold){
      // The constant drive eventually crosses threshold, even after inhibition.
      while(voltageAt(i,hi)<=threshold && hi<(1ULL<<40))hi*=2;
      if(voltageAt(i,hi)<=threshold)return;
    }else{
      if(g[i]<=0 || current[i]+g[i]<=v[i] || std::max(v[i],current[i])+g[i]*aFactor<=threshold)return;
      double a=v[i]-current[i]+g[i]*aFactor;
      if(a<=0)return;
      double peak=std::log(g[i]*aFactor*config.tauM/(a*config.tauS))/((1/config.tauS-1/config.tauM)*config.dt);
      if(!std::isfinite(peak) || peak<=0 || peak>1e12)return;
      hi=std::max<uint64_t>(1,std::ceil(peak));
      if(hi>1 && voltageAt(i,hi-1)>voltageAt(i,hi))hi--;
      if(voltageAt(i,hi)<=threshold)return;
    }
    uint64_t lo=1;
    while(lo<hi){auto mid=lo+(hi-lo)/2;if(voltageAt(i,mid)>threshold)hi=mid;else lo=mid+1;}
    enqueue(i,start+lo);
  }
  void mark(uint32_t i){if(!marked[i]){marked[i]=1;touched.push_back(i);}}
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
    for(uint32_t k=0;k<n;k++){auto i=ids[k];evolve(i);current[i]=values[k];schedule(i);}
  }
  void inject(uint32_t n,const uint32_t* ids,const float* values){
    for(uint32_t k=0;k<n;k++){checkIndex(ids[k]);if(!std::isfinite(values[k]))throw std::runtime_error("Non-finite pulse");}
    for(uint32_t k=0;k<n;k++){auto i=ids[k];if(tick<refractory[i])continue;evolve(i);v[i]+=values[k];schedule(i);}
  }
  void setRefractory(uint32_t n,const uint32_t* ids,double ms){
    if(!std::isfinite(ms)||ms<0||ms>1000||std::abs(ms/config.dt-std::round(ms/config.dt))>1e-7)throw std::runtime_error("Invalid refractory period");
    for(uint32_t k=0;k<n;k++)checkIndex(ids[k]);
    for(uint32_t k=0;k<n;k++)refractoryTicks[ids[k]]=std::llround(ms/config.dt);
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
      for(auto i:pending)for(uint32_t e=graph->row[i];e<graph->row[i+1];e++){
        auto j=graph->col[e];if(tick<refractory[j])continue;
        arriving[j]+=config.weightScale*graph->weight[e];mark(j);
      }
      pending.clear();
      for(auto i:touched){evolve(i);g[i]+=arriving[i];arriving[i]=0;}
      for(auto k:inputEvents[t]){const auto& input=inputs[k];auto i=input.neuron;if(tick<refractory[i])continue;if(!marked[i])evolve(i);v[i]+=input.amplitude;mark(i);}
      for(auto i:touched){schedule(i);marked[i]=0;}touched.clear();
      auto i=head[tick%WHEEL];
      while(i!=NONE){
        auto nextIndex=next[i];
        if(when[i]==tick){
          cancel(i);evolve(i);
          if(v[i]>threshold && tick>=refractory[i]){
            v[i]=config.reset-config.rest;g[i]=0;refractory[i]=tick+refractoryTicks[i];
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
    auto start=std::max(last[i],refractory[i]);auto elapsed=tick>start?tick-start:0;
    if(kind==0)return config.rest+voltageAt(i,elapsed);
    if(kind==1)return g[i]*expS(elapsed);
    if(kind==2)return counts[i];
    throw std::runtime_error("Unknown state field");
  }
};
}

extern "C" {
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
