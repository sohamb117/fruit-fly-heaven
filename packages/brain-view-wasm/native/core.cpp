#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Dataset-independent geometry kernels. Coordinates and signal units are chosen
// by the host; nothing here depends on FlyWire dimensions or neuron identities.
extern "C" {
void nv_colors(uint32_t n, uint32_t neurons, const uint32_t* owners, const double* voltage,
               const double* lastSpike, double now, double rest, double threshold,
               float* out) {
  std::vector<float> palette(neurons*3);
  for (uint32_t id=0; id<neurons; ++id) {
    const double a=std::clamp((voltage[id]-rest)/(threshold-rest),0.0,1.0);
    const double age=now-lastSpike[id];
    const double flash=age>=0 && age<80 ? std::exp(-age/8) : 0;
    const double t=std::max(a,flash);
    const bool inhibited=voltage[id]<rest;
    palette[id*3]  =float((inhibited?.10:.12)+(1-(inhibited?.10:.12))*t);
    palette[id*3+1]=float((inhibited?.27:.32)+.53*t);
    palette[id*3+2]=float((inhibited?.55:.34)+(.28-(inhibited?.55:.34))*t);
  }
  for(uint32_t i=0;i<n;++i)for(int k=0;k<3;++k)out[i*3+k]=palette[owners[i]*3+k];
}

double plane(const float* p,const double* normal,double offset) {
  return p[0]*normal[0]+p[1]*normal[1]+p[2]*normal[2]-offset;
}

uint32_t nv_filter(uint32_t n,const float* positions,const double* normal,
                   double offset,double halfThickness,int mode,uint32_t* out) {
  uint32_t count=0;
  for(uint32_t i=0;i<n;++i){
    double d=plane(positions+i*3,normal,offset);
    if(mode==0 || (mode==1?d<=0:std::abs(d)<=halfThickness)) out[count++]=i;
  }
  return count;
}

int nv_pick(uint32_t n,const float* positions,const float* m,const double* normal,
            double offset,double halfThickness,int mode,double x,double y,
            double width,double height,double radius) {
  int result=-1; double best=radius*radius,depth=2;
  for(uint32_t i=0;i<n;++i){
    const float* p=positions+i*3;double d=plane(p,normal,offset);
    if((mode==1 && d>0)||(mode==2 && std::abs(d)>halfThickness))continue;
    double w=m[3]*p[0]+m[7]*p[1]+m[11]*p[2]+m[15];if(w<=0)continue;
    double px=(m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12])/w;
    double py=(m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13])/w;
    double pz=(m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14])/w;
    if(pz < -1 || pz>1 || px < -1 || px>1 || py < -1 || py>1)continue;
    double dx=(px-x)*width*.5,dy=(py-y)*height*.5,r=dx*dx+dy*dy;
    if(r<best || (std::abs(r-best)<1e-8 && pz<depth)){result=i;best=r;depth=pz;}
  }
  return result;
}

// Intersect triangular surfaces with a plane. Coplanar faces contribute no
// segments; every ordinary crossing contributes exactly two endpoints.
uint32_t nv_contour(const float* positions,uint32_t triangles,const uint32_t* indices,
                    const double* normal,double offset,float* out) {
  uint32_t count=0;
  for(uint32_t t=0;t<triangles;++t){
    double points[3][3];int found=0;
    for(int edge=0;edge<3;++edge){
      const float *a=positions+3*indices[t*3+edge], *b=positions+3*indices[t*3+(edge+1)%3];
      double da=plane(a,normal,offset),db=plane(b,normal,offset);
      if(std::abs(da-db)<1e-12)continue;
      if((da>0 && db>0)||(da<0 && db<0))continue;
      double f=da/(da-db),p[3];for(int k=0;k<3;++k)p[k]=a[k]+f*(b[k]-a[k]);
      bool duplicate=false;
      for(int j=0;j<found;++j){double e=0;for(int k=0;k<3;++k)e+=std::abs(points[j][k]-p[k]);if(e<1e-6)duplicate=true;}
      if(!duplicate && found<3){for(int k=0;k<3;++k)points[found][k]=p[k];++found;}
    }
    if(found==2){for(int j=0;j<2;++j)for(int k=0;k<3;++k)out[count*6+j*3+k]=float(points[j][k]);++count;}
  }
  return count;
}

// Trilinear sampling in voxel-index space. X varies fastest in the volume.
// Origin is the center of output pixel (0,0); u/v are voxel steps per pixel.
void nv_sample(const uint8_t* data,uint32_t sx,uint32_t sy,uint32_t sz,
                const double* origin,const double* u,const double* v,
                uint32_t width,uint32_t height,uint8_t* out) {
  for(uint32_t j=0;j<height;++j)for(uint32_t i=0;i<width;++i){
    double p[3];for(int k=0;k<3;++k)p[k]=origin[k]+i*u[k]+j*v[k];
    if(p[0]<0||p[1]<0||p[2]<0||p[0]>sx-1||p[1]>sy-1||p[2]>sz-1){out[j*width+i]=0;continue;}
    uint32_t lo[3]={uint32_t(p[0]),uint32_t(p[1]),uint32_t(p[2])};
    uint32_t hi[3]={std::min(lo[0]+1,sx-1),std::min(lo[1]+1,sy-1),std::min(lo[2]+1,sz-1)};
    double f[3]={p[0]-lo[0],p[1]-lo[1],p[2]-lo[2]},value=0;
    for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x){
      uint32_t px=x?hi[0]:lo[0],py=y?hi[1]:lo[1],pz=z?hi[2]:lo[2];
      value+=data[px+sx*(py+sy*pz)]*(x?f[0]:1-f[0])*(y?f[1]:1-f[1])*(z?f[2]:1-f[2]);
    }
    out[j*width+i]=uint8_t(std::clamp(std::round(value),0.0,255.0));
  }
}
}
