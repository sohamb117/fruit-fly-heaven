#include <algorithm>
#include <cmath>
#include <cstdint>

// Passive point neurons with instantaneous graded release. Units are those of
// the supplied model. Incoming CSR order is preserved; there are no spike gates.
extern "C" int gv_step(uint32_t n, const uint32_t* rows, const uint32_t* sources,
  const float* weights, const float* bias, const float* tau, uint32_t inputs,
  const uint32_t* inputIds, const float* drive, float* state, float* next,
  float* rectified, float dt, uint32_t steps) {
  for(uint32_t t=0;t<steps;++t){
    for(uint32_t i=0;i<n;++i)rectified[i]=std::max(0.f,state[i]);
    for(uint32_t i=0;i<n;++i){
      float current=0;
      for(uint32_t e=rows[i];e<rows[i+1];++e)current+=weights[e]*rectified[sources[e]];
      next[i]=bias[i]+current-state[i];
    }
    for(uint32_t k=0;k<inputs;++k)next[inputIds[k]]+=drive[k];
    for(uint32_t i=0;i<n;++i){
      state[i]+=dt/std::max(tau[i],dt)*next[i];
      if(!std::isfinite(state[i]))return -1;
    }
  }
  return 0;
}
