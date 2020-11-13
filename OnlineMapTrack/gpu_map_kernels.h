#ifndef _SYSAL_GPU_MAP_KERNELS_H_
#define _SYSAL_GPU_MAP_KERNELS_H_

#include "gpu_incremental_map_track.h"
#include "gpu_interruptible_kernels.h"
#include "gpu_defines.h"

using namespace SySal;
using namespace SySal::GPU;

namespace SySal { namespace GPU {


__global__ void curvaturemap_kernel(int *pXYCurv, int *pZCurv, int span, float xy_curvature, float z_curvature);

__global__ void correctcurvature_kernel(IntCluster *pC, short *pZC, int camrotsin, int camrotcos, int *pCurv, int *pCurvY, int *pZCurvX, int *pZCurvY, int dmagdx, int dmagdy, int total, int w2, int h2);

__global__ void correctdemag_kernel(IntCluster *pC, int cblock, int imgclusters, int demag, int width, int height);

__global__ void rescaleshifts_kernel(short *px, short *py, short *pdx, short *pdy, int refimg, int totalimg);

__global__ void setXYZs_kernel(short *pCX, short *pCY, short *pCZ, int total, int img, short *pX, short *pY, short z);

__global__ void resetcounts_kernel(int *pmapcounts, int deltas2, int *pbest);

__global__ void maphash_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny);

__global__ void maphash_minarea_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny, int minarea);

__global__ void makedeltas_kernel(int *pDeltas, int tol, int deltasx, int deltasy, short *pdx, short *pdy, int img);

__global__ void makedeltas_fromshift_kernel(int *pDeltas, int tol, int deltasx, int deltasy, int *pBestDeltas, int *pBest, int bestdeltasx);

__global__ void makefinaldeltas_fromshift_kernel(int *pDeltas, int tol, int *pBestDeltas, int *pBest, int bestdeltasx, short *px, short *py, short *pdx, short *pdy, int img, int totalimg);

struct trymap_kernel_args : public SySal::GPU::InterruptibleKernels::Args
{
	IntCluster *pC;
	int nc;
	int clusterblocksize;
	Cell *pCell;
	IntCluster **pCellContent;	
	int maxcellcontent;
	int minclustersize;
	int *pDeltas;
	int deltasx;
	int deltasy; 
	int cellsize;
	int tol;
	int w;
	int h;
	int demag;
	int nx;
	int ny;
	int *pMapCounts;
	int sampledivider;
	int clustermapmin;
	int *pBest;	
};

struct trymap_kernel_status : public SySal::GPU::InterruptibleKernels::Status
{
	int i;
	int ic;
	int deltas2;
	int w2;
	int h2;		
};

__global__ void trymap_Ikernel(trymap_kernel_args * __restrict__ pargs, trymap_kernel_status * __restrict__ pstatus);

__global__ void trymap2_Ikernel(trymap_kernel_args * __restrict__ pargs, trymap_kernel_status * __restrict__ pstatus);

__global__ void sumcounts_kernel(int *pMapCounts, int deltas2, int total, int step);

__global__ void safesumcounts_kernel(int *pMapCounts, int deltas2, int repeat);

__global__ void safefindbest_kernel(int *pMapCounts, int deltas2, int *pBest);

__global__ void findbest_kernel(int *pMapCounts, int deltas2, int step, int *pBest);

struct refinemap_kernel_args : public SySal::GPU::InterruptibleKernels::Args
{
	IntCluster *pC;
	int nc;
	int clusterblocksize;
	Cell *pCell;
	IntCluster **pCellContent;	
	int maxcellcontent;
	int cellsize;
	int tol;	
	int refinebin;
	int w;
	int h;
	int demag;
	int nx;
	int ny;
	int *pMapCounts;	
	IntCluster **pClusterChain;	
	IntCluster *pBase;
	int *pDeltas;	
	int deltas;
	int *pBest;
};

struct refinemap_kernel_status : public SySal::GPU::InterruptibleKernels::Status
{
	int i;
	int ic;
	int w2;
	int h2;
	int dx, dy;	
	IntCluster *pc;
	int refinedeltas;
	int refinespan;
	int refinespan2;
};

__global__ void refinemap_Ikernel(refinemap_kernel_args * __restrict__ pargs, refinemap_kernel_status * __restrict__ pstatus);

struct finalmap_kernel_args : public SySal::GPU::InterruptibleKernels::Args
{
	IntCluster *pC;
	int nc;
	int clusterblocksize;
	Cell *pCell;
	IntCluster **pCellContent;
	int maxcellcontent;
	int cellsize;
	int tol;
	int w;
	int h;
	int demag;
	int nx;
	int ny;	
	int *pMapCounts;
	IntCluster **pClusterChain;
	int img;
	IntCluster *pBase;
	short *pDX;
	short *pDY;
};

struct finalmap_kernel_status : public SySal::GPU::InterruptibleKernels::Status
{
	int i;
	int ic;
	int w2;
	int h2;
	int dx, dy;	
	IntCluster *pc;
};

__global__ void finalmap_Ikernel(finalmap_kernel_args * __restrict__ pargs, finalmap_kernel_status * __restrict__ pstatus);

__global__ void clearhash_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, int cellsize, int nx, int ny);

struct makechain_kernel_args : public SySal::GPU::InterruptibleKernels::Args
{
	IntCluster *pC;
	IntCluster **pClusterChains;
	short *pClusterXs;
	short *pClusterYs;
	short *pClusterZs;
	IntChain *pChain;
	int *pChainCounts;
	int totalclusters;
	int clusterblocksize;
	int minvol;
	int minclusters;
	float xtomicron;
	float ytomicron;
	short width;
	short height;
	int stagex;
	int stagey;
	int xslant;
	int yslant;
	int viewtag;
};

struct makechain_kernel_status : public SySal::GPU::InterruptibleKernels::Status
{
	IntChain ch;
	int i;
	int count;
	int istart;
	int iend;
	short w2;
	short h2;
};

__global__ void makechain_Ikernel(makechain_kernel_args * __restrict__ pargs, makechain_kernel_status * __restrict__ pstatus);

__global__ void setchainbase_kernel(int *pChainBase, int *pChainCounts, int counts, ChainMapHeader *pCh, ChainView *pChV, int px, int py, int pz, int w, int h);

__global__ void compactchains_kernel(IntChain *pCompact, int *pChainBase, IntChain *pOriginal, int *pChainCounts, int chainblocksize);

__global__ void makechainwindow_kernel(ChainMapWindow *pChMapWnd, short *px, short *py, int imgs, int width, int height, float pxmicron, float pymicron, int maxcells, int mincellsize, ChainView *pChV, Cell *pCells, IntCluster **pCellContent, int maxcellcontent, int stagex, int stagey, ChainView *pChLastV);

__global__ void maphashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider);

__global__ void clearhashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider);

__global__ void makechaindeltas_kernel(int *pDeltas, int xytol, int ztol, int deltasx, int deltasy, int deltasz, ChainView *plastview, int xc, int yc, float xslant, float yslant, float dxdz, float dydz);

__global__ void makechaindeltas_fromshift_kernel(int *pDeltas, int xytol, int ztol, int deltasx, int deltasy, int deltasz, int *pBestDeltas, int *pBest, int bestdeltasx, int bestdeltasy, int bestdeltasz);

struct trymapchain_kernel_args : public SySal::GPU::InterruptibleKernels::Args
{
	IntChain *pChains;
	int *pChainCounts;
	int nc; 
	int chainblocksize;
	ChainMapWindow *pChMapWnd;	
	int *pDeltas;
	int deltasX;
	int deltasY;
	int deltasZ;
	int xytol;
	int ztol; 
	int minchainsize;
	int *pMapCounts;
	int sampledivider;
	int *pBest;
};

struct trymapchain_kernel_status : public SySal::GPU::InterruptibleKernels::Status
{
	int ic;
	int nx, ny;
};

__global__ void trymapchain_Ikernel(trymapchain_kernel_args *pargs, trymapchain_kernel_status *pstatus);

__global__ void trymapchaindxydz_Ikernel(trymapchain_kernel_args *pargs, trymapchain_kernel_status *pstatus);

struct refinemapchain_kernel_args : public SySal::GPU::InterruptibleKernels::Args
{
	IntChain *pChains;
	int *pChainCounts;
	int nc;
	int chainblocksize;
	ChainMapWindow *pChMapWnd;
	int *pDeltas;
	int bestdeltasXY;
	int bestdeltasZ;
	int xyrefinebin;
	int zrefinebin;
	int xytol;
	int ztol;
	int *pMapCounts;
};

struct refinemapchain_kernel_status : public SySal::GPU::InterruptibleKernels::Status
{
	int ic;
	IntChain *pC;
	int nx;
	int ny;
	int dix, diy, diz;
	int xyrefinedeltas;
	int zrefinedeltas;
	int xyrefinespan; 
	int zrefinespan;
	int xyzrefinespan;
};

__global__ void refinemapchain_Ikernel(refinemapchain_kernel_args *pargs, refinemapchain_kernel_status *pstatus);

__global__ void finalmapchain_kernel(IntChain *pChains, int *pChainCounts, int nc, int chainblocksize, ChainMapWindow *pChMapWnd, int *pDeltas, int deltasXY, int deltasZ, int xytol, int ztol, int *pD);

__global__ void negshift_viewchains_kernel(ChainView *pview, int *pDeltas, int deltasXY, int deltasZ, int *pD);

__global__ void setchainviewheader_kernel(ChainMapHeader *pmaph, ChainView *pview, int px, int py, int pz, int *pDeltas, int deltasXY, int deltasZ, int *pD);

__global__ void preparefinalchainviewcorr_kernel(ChainMapHeader *pmaph, int minmap);

__global__ void applyfinalchainviewcorr_kernel(ChainMapHeader *pmaph);

} }

#endif
