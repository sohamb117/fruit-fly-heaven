#pragma once
// Independently expressed equations from Hürkey et al., Nature (2023),
// doi:10.1038/s41586-023-06099-0, Methods / supplementary parameter tables.
// Units: mV, ms, nS, pA, pF. Paper Q=39.2/V = .0392/mV.
// This is NOT imported from the authors' CC-BY-NC software. No tonic drive,
// source-network coupling, reset, or generic LIF adaptation is added here.
struct DlmState { float v,h,b; };
static DlmState dlm_derivative(DlmState x,float current,const float* g,const float* receptors){
  const float m=1.f/(1.f+std::exp(-.0392f*3.f*(x.v+33.f)));
  const float eh=std::exp(-.0392f*5.2f*(x.v+39.14f));
  const float eb=std::exp(-.0392f*1.1056f*(x.v+42.14f));
  const float hinf=1.f/(1.f+eh),binf=1.f/(1.f+eb);
  const float tauh=std::exp(-.0392f*5.2f*.38f*(x.v+39.14f))/(.2f*(1.f+eh));
  const float taub=std::exp(-.0392f*1.1056f*.38f*(x.v+42.14f))/(.2f*(1.f+eb));
  float synaptic=0;
  for(uint32_t r=0;r<5;r++)synaptic+=g[r]*(receptors[r*3+2]-x.v);
  const float sodium=431.2f*m*m*m*(1.f-x.h)*(x.v-55.f);
  const float potassium=137.68216f*x.b*x.b*x.b*x.b*(x.v+72.f);
  return {(current+synaptic-8.624f*(x.v+60.f)-sodium-potassium)/130.f,(hinf-x.h)/tauh,(binf-x.b)/taub};
}
static DlmState dlm_add(DlmState x,DlmState d,float t){return {x.v+t*d.v,x.h+t*d.h,x.b+t*d.b};}
static DlmState dlm_rk4(DlmState x,float current,const float* g,const float* receptors){
  const DlmState a=dlm_derivative(x,current,g,receptors);
  const DlmState b=dlm_derivative(dlm_add(x,a,.05f),current,g,receptors);
  const DlmState c=dlm_derivative(dlm_add(x,b,.05f),current,g,receptors);
  const DlmState d=dlm_derivative(dlm_add(x,c,.1f),current,g,receptors);
  return {x.v+(.1f/6.f)*(a.v+2.f*b.v+2.f*c.v+d.v),x.h+(.1f/6.f)*(a.h+2.f*b.h+2.f*c.h+d.h),x.b+(.1f/6.f)*(a.b+2.f*b.b+2.f*c.b+d.b)};
}
static bool dlm_valid(DlmState x){return std::isfinite(x.v)&&std::isfinite(x.h)&&std::isfinite(x.b)&&x.h>=0&&x.h<=1&&x.b>=0&&x.b<=1;}
static float dlm_conductance(DlmState x,const float* g){
  const float m=1.f/(1.f+std::exp(-.0392f*3.f*(x.v+33.f)));
  float result=8.624f+431.2f*m*m*m*(1.f-x.h)+137.68216f*x.b*x.b*x.b*x.b;
  for(uint32_t r=0;r<5;r++)result+=g[r];
  return result;
}
