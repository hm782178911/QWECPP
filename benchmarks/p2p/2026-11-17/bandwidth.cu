#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#define C(x) do{auto e=(x);if(e){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
using clk=std::chrono::steady_clock;
int main(){for(int s=0;s<2;s++){int d=1-s;C(cudaSetDevice(d));C(cudaDeviceEnablePeerAccess(s,0));for(size_t n:{size_t(1)<<20,size_t(16)<<20,size_t(64)<<20,size_t(256)<<20}){void *p,*q,*h,*v;C(cudaMallocHost(&h,n));C(cudaMallocHost(&v,n));memset(h,0x37,n);C(cudaSetDevice(s));C(cudaMalloc(&p,n));C(cudaMemcpy(p,h,n,cudaMemcpyHostToDevice));C(cudaSetDevice(d));C(cudaMalloc(&q,n));for(int mode=0;mode<2;mode++){auto copy=[&](){if(mode==0){C(cudaSetDevice(d));C(cudaMemcpyPeer(q,d,p,s,n));C(cudaDeviceSynchronize());}else{C(cudaSetDevice(s));C(cudaMemcpy(h,p,n,cudaMemcpyDeviceToHost));C(cudaSetDevice(d));C(cudaMemcpy(q,h,n,cudaMemcpyHostToDevice));C(cudaDeviceSynchronize());}};for(int i=0;i<3;i++)copy();int reps=std::max(10,int((size_t(512)<<20)/n));double best=0,sum=0;for(int trial=0;trial<5;trial++){auto t=clk::now();for(int r=0;r<reps;r++)copy();double sec=std::chrono::duration<double>(clk::now()-t).count();double gb=double(n)*reps/sec/1e9;best=std::max(best,gb);sum+=gb;}C(cudaMemcpy(v,q,n,cudaMemcpyDeviceToHost));size_t bad=0;for(size_t i=0;i<n;i++)bad+=((unsigned char*)v)[i]!=0x37;printf("%d->%d MiB=%zu mode=%s best_GBps=%.3f mean_GBps=%.3f mismatches=%zu reps=%d trials=5\n",s,d,n>>20,mode?"host_serial":"p2p",best,sum/5,bad,reps);fflush(stdout);if(bad)return 1;}C(cudaFree(q));C(cudaSetDevice(s));C(cudaFree(p));C(cudaFreeHost(h));C(cudaFreeHost(v));}}}
