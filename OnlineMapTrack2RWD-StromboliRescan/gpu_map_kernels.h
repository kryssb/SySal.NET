#ifndef _SYSAL_GPU_MAP_KERNELS_H_
#define _SYSAL_GPU_MAP_KERNELS_H_

#include "gpu_incremental_map_track.h"
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

__global__ void maphash_kernel(IntCluster *pC, int nc, int clusterblocksize, int i, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny);

__global__ void maphash_minarea_kernel(IntCluster *pC, int nc, int clusterblocksize, int i, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny, int minarea);

__global__ void makedeltas_kernel(int *pDeltas, int tol, int deltasx, int deltasy, short *pdx, short *pdy, int img);

__global__ void makedeltas_fromshift_kernel(int *pDeltas, int tol, int deltasx, int deltasy, int *pBestDeltas, int *pBest, int bestdeltasx);

__global__ void makefinaldeltas_fromshift_kernel(int *pDeltas, int tol, int *pBestDeltas, int *pBest, int bestdeltasx, short *px, short *py, short *pdx, short *pdy, int img, int totalimg);

/****************************/

__global__ void max_check_kernel(int * pInt, int total, int halftotal);

__global__ void max_kernel(int * pInt, int halftotal);

__global__ void compact_kernel(int * pInt, int stride, int count, int * pOut);

__global__ void sum_check_kernel(int * pInt, int total, int halftotal);

__global__ void sum_kernel(int * pInt, int halftotal);

__global__ void sum_check_multiple_kernel(int * pInt, int total, int halftotal);

__global__ void sum_multiple_kernel(int * pInt, int total, int halftotal);

__global__ void shift_postfix_kernel(int *pdest, int *psrc, short postfix);

__global__ void shift_postfixid_kernel(int *pdest, int *psrc, int total);

__global__ void compare_and_setmax(int *pbest, int *pnew);

__global__ void recursive_sum_kernel(int *parrayin, int *parrayout, int insize);

__global__ void split_and_index_kernel(int *paircomputer, int depth, IntPair *pairindices, int totalpairs);

__global__ void trymap2_prepare_clusters_kernel(IntCluster *pc, IntMapCluster *pmc, int totalclusters, int divider, int mingrainsize, int w2, int h2, int demag, int *pValidFlag = 0);

__global__ void trymap2_shift_kernel(IntMapCluster *pmc, int totalmapclusters, int *pDeltaX, int *pDeltaY, int cellsize);

__global__ void trymap2_shiftmatch_kernel(IntMapCluster *pmc, IntPair *pPairs, int totalpairs, int *pDeltas, int cellsize, short nx, short ny, int *pmatchresult, int tol, Cell *pmapcell, IntCluster **pMapCellContent, int maxcellcontent);

__global__ void finalmap_cell_kernel(IntMapCluster *pmc, int totalmapclusters, Cell *pmapcell, int *pClustersInCell, int nx, int ny);

__global__ void finalmap_match_kernel(IntMapCluster *pmc, IntPair *pPairs, int totalpairs, int *pmatchresult, int *pmatchmap, int tol, Cell *pmapcell, IntCluster **pMapCellContent, int maxcellcontent, int nx, int ny);

__global__ void finalmap_optimize_kernel(IntCluster *pc, IntMapCluster *pmc, int clusteroffset, int totalclusters, int *pmatchresult, int *pmatchmap, IntCluster **pMapCellContent, IntCluster **pClusterChain);

__global__ void makechain_kernel(IntCluster *pC, int totalclusters, short w2, short h2, short *pClusterXs, short *pClusterYs, short *pClusterZs, int xslant, int yslant, IntCluster **pClusterChains, short minclusters, short minvol, float xtomicron, float ytomicron, int stagex, int stagey, IntChain *pChain, int viewtag, int *pvalid = 0);

__global__ void maphashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider);

__global__ void clearhashchain2_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider);

__global__ void trymapchain_prepare_chains_kernel(IntChain *pc, IntMapChain *pmc, int totalchains, int minchainsize, int *pValidFlag = 0);

__global__ void trymapchain_shiftmatch_kernel(IntMapChain *pmc, IntPair *pPairs, int totalpairs, int *pMapCount, int *pDeltas, ChainMapWindow *pChMapWnd, int xytol, short zsteps, int ztol);

__global__ void make_finalchainshift_kernel(int *pDeltas, int *pRefineDeltas, int *pBest, int deltasXY);

__global__ void finalmapchain_cell_kernel(IntMapChain *pmc, IntPair *pPairs, int totalpairs, int *pDeltas, ChainMapWindow *pChMapWnd, int *pvalid);

__global__ void finalmapchain_match_kernel(IntChain *pc, IntMapChain *pmc, IntPair *pPairs, int totalpairs, ChainMapWindow *pChMapWnd, int xytol, int ztol);

__global__ void finalmapchain_filter_kernel(IntChain *pc, int totalchains, int *pvalid);

__global__ void compactchains_kernel(IntChain *pcmpct, IntChain *pch, IntPair *pPairs, int totalpairs, ChainView *pChV);

__global__ void negshift_viewchains_kernel(ChainView *pview, int *pDeltas);

/****************************/

__global__ void sumcounts_kernel(int *pMapCounts, int deltas2, int total, int step);

__global__ void safesumcounts_kernel(int *pMapCounts, int deltas2, int repeat);

__global__ void safefindbest_kernel(int *pMapCounts, int deltas2, int *pBest);

__global__ void findbest_kernel(int *pMapCounts, int deltas2, int step, int *pBest);

__global__ void clearhash_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, int cellsize, int nx, int ny);

__global__ void setchainbase_kernel(int *pChainBase, int *pChainCounts, int counts, ChainMapHeader *pCh, ChainView *pChV, int px, int py, int pz, int w, int h);

__global__ void compactchains_kernel(IntChain *pCompact, int *pChainBase, IntChain *pOriginal, int *pChainCounts, int chainblocksize);

__global__ void makechainwindow_kernel(ChainMapWindow *pChMapWnd, short *px, short *py, int imgs, int width, int height, float pxmicron, float pymicron, int maxcells, int mincellsize, ChainView *pChV, Cell *pCells, IntCluster **pCellContent, int maxcellcontent, int stagex, int stagey, ChainView *pChLastV);

__global__ void maphashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int chainblocksize, int i);

__global__ void clearhashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider);

__global__ void makechaindeltas_kernel(int *pDeltas, int xytol, int ztol, int deltasx, int deltasy, int deltasz, ChainView *plastview, int xc, int yc, float xslant, float yslant, float dxdz, float dydz);

__global__ void makechaindeltas_fromshift_kernel(int *pDeltas, int xytol, int ztol, int deltasx, int deltasy, int deltasz, int *pBestDeltas, int *pBest, int bestdeltasx, int bestdeltasy, int bestdeltasz);

__global__ void finalmapchain_kernel(IntChain *pChains, int *pChainCounts, int nc, int chainblocksize, ChainMapWindow *pChMapWnd, int *pDeltas, int deltasXY, int deltasZ, int xytol, int ztol, int *pD);

__global__ void negshift_viewchains_kernel(ChainView *pview, int *pDeltas, int deltasXY, int deltasZ, int *pD);

__global__ void setchainviewheader_kernel(ChainMapHeader *pmaph, ChainView *pview, int px, int py, int pz, int *pDeltas, int deltasXY, int deltasZ, int *pD);

__global__ void preparefinalchainviewcorr_kernel(ChainMapHeader *pmaph, int minmap);

__global__ void applyfinalchainviewcorr_kernel(ChainMapHeader *pmaph);

} }

#endif
