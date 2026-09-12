// FlyWire sparse LIF engine. Every source connection is retained.
// State is propagated analytically between events; thresholds are checked on a
// fixed 0.1 ms grid. Quiet cells retain their exact decaying state lazily.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double DT=.1, TM=20., TS=5., THRESH=7., WEIGHT=.275;
constexpr uint32_t DELAY=18, REFRACT=22;
thread_local std::string error;

template<class T> std::vector<T> read(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open "+path);
  auto size=f.tellg();
  if (size<0 || size%sizeof(T)) throw std::runtime_error("Invalid array "+path);
  std::vector<T> out(size/sizeof(T)); f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(out.data()),size)) throw std::runtime_error("Truncated "+path);
  return out;
}
struct Graph {
  std::vector<uint32_t> ptr, target;
  std::vector<float> weight;
  std::array<std::vector<uint32_t>,8> groups;
  std::vector<uint16_t> groupMask;
  std::vector<uint8_t> sensory;
  std::array<double,10001> em, es;
  explicit Graph(const std::string& path) {
    ptr=read<uint32_t>(path+"/indptr.bin");
    target=read<uint32_t>(path+"/targets.bin");
    weight=read<float>(path+"/weights.bin");
    if (ptr.size()<2 || ptr.front()!=0 || ptr.back()!=target.size() || target.size()!=weight.size()
        || !std::is_sorted(ptr.begin(),ptr.end())) throw std::runtime_error("Invalid graph CSR");
    for (auto x:target) if (x>=size()) throw std::runtime_error("Invalid target");
    groupMask.resize(size()); sensory.resize(size());
    auto g=read<uint32_t>(path+"/groups.bin"); size_t offset=0;
    for (int k=0;k<8;k++) {
      if (offset>=g.size()) throw std::runtime_error("Missing group");
      auto count=g[offset++];
      if (offset+count>g.size()) throw std::runtime_error("Truncated group");
      for (uint32_t j=0;j<count;j++) {
        auto i=g[offset++]; if(i>=size()) throw std::runtime_error("Invalid group index");
        groups[k].push_back(i); groupMask[i]|=1<<k; if(k<3) sensory[i]=1;
      }
    }
    for (int i=0;i<=10000;i++) { em[i]=std::exp(-i*DT/TM); es[i]=std::exp(-i*DT/TS); }
  }
  uint32_t size() const {return ptr.size()-1;}
  double m(uint32_t t) const {return t<=10000 ? em[t] : std::exp(-t*DT/TM);}
  double s(uint32_t t) const {return t<=10000 ? es[t] : std::exp(-t*DT/TS);}
};
struct Event {
  uint32_t when, neuron, version;
  bool operator<(const Event& b) const {
    if(when!=b.when) return when>b.when;
    return neuron>b.neuron;
  }
};
struct Brain {
  const Graph& graph;
  std::vector<double> v,g,incoming;
  std::vector<uint32_t> last,refractory,version,counts,touched;
  std::vector<uint8_t> marked;
  std::priority_queue<Event> events;
  std::array<std::vector<uint32_t>,DELAY+1> delayed;
  std::array<double,8> rates{};
  std::array<uint64_t,8> blockCounts{};
  std::array<std::array<uint32_t,2>,256> recent{};
  uint32_t tick=0, ever=0;
  uint64_t rng, total=0;
  explicit Brain(const Graph& graph,uint64_t seed):graph(graph),rng(seed) {
    auto n=graph.size(); v.resize(n);g.resize(n);incoming.resize(n);last.resize(n);
    refractory.resize(n);version.resize(n);counts.resize(n);marked.resize(n);
  }
  double random() {
    rng+=0x9e3779b97f4a7c15ULL; uint64_t x=rng;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return (((x^(x>>31))>>11)+.5)*0x1.0p-53;
  }
  void update(uint32_t i) {
    const uint32_t start=std::max(last[i],refractory[i]);
    if(tick>start) {
      auto d=tick-start; double a=graph.m(d),b=graph.s(d);
      v[i]=v[i]*a+g[i]*(a-b)/3.; g[i]*=b;
    }
    last[i]=tick;
  }
  double future(uint32_t i,uint32_t ticks) const {
    double a=graph.m(ticks),b=graph.s(ticks);
    return v[i]*a+g[i]*(a-b)/3.;
  }
  void schedule(uint32_t i) {
    uint32_t ver=++version[i], start=std::max(tick,refractory[i]);
    if(v[i]>THRESH && start==tick) {events.push({tick,i,ver}); return;}
    // A rigorous upper bound for these two exponentials avoids unnecessary roots.
    if(g[i]<=0 || g[i]<=v[i] || std::max(0.,v[i])+g[i]/3.<=THRESH) return;
    double a=v[i]+g[i]/3.;
    if(a<=0) return;
    double peak=std::log((4.*g[i]/3.)/a)/(.15*DT);
    if(!std::isfinite(peak) || peak<=0) return;
    uint32_t hi=std::max(1u,static_cast<uint32_t>(std::ceil(peak)));
    if(hi>1 && future(i,hi-1)>future(i,hi)) hi--;
    if(future(i,hi)<=THRESH) return;
    uint32_t lo=1;
    while(lo<hi) {auto mid=lo+(hi-lo)/2; if(future(i,mid)>THRESH) hi=mid; else lo=mid+1;}
    events.push({start+lo,i,ver});
  }
  void mark(uint32_t i) {if(!marked[i]) {marked[i]=1; touched.push_back(i);}}
  void step(uint32_t steps,double odorL,double odorR,double sweet) {
    if(steps<1 || steps>10000) throw std::runtime_error("Invalid timestep block");
    std::vector<std::vector<uint32_t>> inputs(steps);
    const double hz[3]={odorL,odorR,sweet};
    for(int k=0;k<3;k++) {
      if(!std::isfinite(hz[k]) || hz[k]<0 || hz[k]>1000) throw std::runtime_error("Invalid stimulus rate");
      if(hz[k]==0) continue;
      double logP=std::log1p(-hz[k]*DT/1000.);
      for(auto i:graph.groups[k]) {
        uint64_t offset=static_cast<uint64_t>(std::floor(std::log(random())/logP));
        while(offset<steps) {
          inputs[offset].push_back(i);
          offset+=1+static_cast<uint64_t>(std::floor(std::log(random())/logP));
        }
      }
    }
    blockCounts.fill(0);
    for(uint32_t t=0;t<steps;t++,tick++) {
      auto& pending=delayed[tick%(DELAY+1)];
      for(auto i:pending) {
        for(uint32_t e=graph.ptr[i];e<graph.ptr[i+1];e++) {
          auto j=graph.target[e];
          // Brian2's '(unless refractory)' makes g read-only during this interval.
          if(tick<refractory[j]) continue;
          incoming[j]+=WEIGHT*graph.weight[e]; mark(j);
        }
      }
      pending.clear();
      for(auto i:touched) {update(i);g[i]+=incoming[i];incoming[i]=0;}
      for(auto i:inputs[t]) {if(!marked[i]) update(i);v[i]+=WEIGHT*250.;mark(i);}
      for(auto i:touched) {schedule(i);marked[i]=0;}
      touched.clear();
      while(!events.empty() && events.top().when<=tick) {
        auto e=events.top(); events.pop(); auto i=e.neuron;
        if(e.version!=version[i]) continue;
        update(i);
        if(v[i]<=THRESH || refractory[i]>tick) {schedule(i);continue;}
        v[i]=g[i]=0; refractory[i]=tick+(graph.sensory[i] ? 0 : REFRACT);version[i]++;
        delayed[(tick+DELAY)%(DELAY+1)].push_back(i);
        auto mask=graph.groupMask[i];
        for(int k=0;k<8;k++) if(mask&(1<<k)) blockCounts[k]++;
        if(counts[i]++==0) ever++;
        recent[total%recent.size()]={tick,i}; total++;
      }
    }
    double alpha=1-std::exp(-static_cast<double>(steps)*DT/100.);
    for(int k=0;k<8;k++) {
      double rate=graph.groups[k].empty() ? 0 : blockCounts[k]*1000./(steps*DT*graph.groups[k].size());
      rates[k]+=(rate-rates[k])*alpha;
    }
  }
};
}
extern "C" {
const char* fly_error(){return error.c_str();}
void* fly_graph(const char* path) {try{return new Graph(path);}catch(const std::exception& e){error=e.what();return nullptr;}}
void fly_graph_free(void* p){delete static_cast<Graph*>(p);}
void* fly_create(void* g,uint64_t seed){try{return new Brain(*static_cast<Graph*>(g),seed);}catch(const std::exception& e){error=e.what();return nullptr;}}
void fly_free(void* p){delete static_cast<Brain*>(p);}
int fly_step(void* p,uint32_t steps,double l,double r,double s){try{static_cast<Brain*>(p)->step(steps,l,r,s);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
void fly_stats(void* p,double* out){auto& b=*static_cast<Brain*>(p);for(int k=0;k<8;k++)out[k]=b.rates[k];out[8]=b.total;out[9]=b.ever;out[10]=b.tick*DT;out[11]=b.events.size();}
uint32_t fly_recent(void* p,uint32_t* out){auto& b=*static_cast<Brain*>(p);auto n=std::min<uint64_t>(b.total,b.recent.size());for(uint64_t j=0;j<n;j++){auto e=b.recent[(b.total-n+j)%b.recent.size()];out[j*2]=e[0];out[j*2+1]=e[1];}return n;}
void fly_state(void* p,uint32_t i,double* out){auto& b=*static_cast<Brain*>(p);if(i>=b.graph.size())return;b.update(i);out[0]=b.v[i];out[1]=b.g[i];out[2]=b.counts[i];}
}
