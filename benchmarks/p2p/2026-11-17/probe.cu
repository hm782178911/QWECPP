#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#define C(x) do{cudaError_t e=(x);if(e){printf("ERROR %s %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
int main(){size_t total=0;for(int s=0;s<2;s++){int d=1-s;C(cudaSetDevice(d));C(cudaDeviceEnablePeerAccess(s,0));for(size_t n:{size_t(4096),size_t(1048576),size_t(16777216)}){std::vector<unsigned char>a(n),b(n);void *p,*q;C(cudaSetDevice(s));C(cudaMalloc(&p,n));C(cudaSetDevice(d));C(cudaMalloc(&q,n));for(int k=0;k<3;k++){for(size_t i=0;i<n;i++)a[i]=(i*31+k*73+i/251)%256;C(cudaSetDevice(s));C(cudaMemcpy(p,a.data(),n,cudaMemcpyHostToDevice));C(cudaDeviceSynchronize());C(cudaSetDevice(d));C(cudaMemset(q,0xa5,n));C(cudaMemcpyPeer(q,d,p,s,n));C(cudaDeviceSynchronize());C(cudaMemcpy(b.data(),q,n,cudaMemcpyDeviceToHost));size_t bad=0;for(size_t i=0;i<n;i++)bad+=a[i]!=b[i];total+=bad;printf("%d->%d bytes=%zu round=%d peer_bad=%zu ",s,d,n,k,bad);C(cudaMemcpy(q,a.data(),n,cudaMemcpyHostToDevice));C(cudaMemcpy(b.data(),q,n,cudaMemcpyDeviceToHost));bad=0;for(size_t i=0;i<n;i++)bad+=a[i]!=b[i];total+=bad;printf("host_bad=%zu\n",bad);}C(cudaFree(q));C(cudaSetDevice(s));C(cudaFree(p));}}return total?1:0;}
