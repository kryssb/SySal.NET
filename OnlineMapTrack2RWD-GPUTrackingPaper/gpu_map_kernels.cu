#include "gpu_map_kernels.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace SySal { namespace GPU {

__global__ void curvaturemap_kernel(int *pXYCurv, int *pZCurv, int span, float xy_curvature, float z_curvature)
{
	int ix = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int span2 = span >> 1;
	if (ix < span)
	{
		int dx = ix - span2;
		int dx2 = dx * dx;
		pXYCurv[ix] = xy_curvature * dx2;
		pZCurv[ix] = z_curvature * dx2;		
	}
}

__global__ void correctcurvature_kernel(IntCluster *pC, short *pZC, int camrotsin, int camrotcos, int *pCurv, int *pCurvY, int *pZCurvX, int *pZCurvY, int dmagdx, int dmagdy, int total, int w2, int h2)
{
	int ix = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (ix < total)
	{
		IntCluster *pc = pC + ix;
		int x = pc->X;
		int y = pc->Y;
		int x2 = x - w2;
		int y2 = y - h2;
		int d = (pCurv[x] + pCurv[y]);
		pc->X += (((x2 * d) >> XY_CURVATURE_SHIFT) + ((dmagdy * y2 * x2) >> XY_MAGNIFICATION_SHIFT)) + ((x2 * camrotcos) >> FRACT_RESCALE_SHIFT) - ((y2 * camrotsin) >> FRACT_RESCALE_SHIFT);
		pc->Y += (((y2 * d) >> XY_CURVATURE_SHIFT) + ((dmagdx * y2 * x2) >> XY_MAGNIFICATION_SHIFT)) + ((x2 * camrotsin) >> FRACT_RESCALE_SHIFT) + ((y2 * camrotcos) >> FRACT_RESCALE_SHIFT);
		pZC[ix] = ((pZCurvX[x] + pZCurvY[y]) >> Z_CURVATURE_SHIFT);
	}
}

__global__ void setXYZs_kernel(short *pCX, short *pCY, short *pCZ, int total, int img, short *pX, short *pY, short z)
{
	int ix = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int x = pX[img];
	int y = pY[img];
	if (ix < total)
	{
		pCX[ix] = x;
		pCY[ix] = y;
		pCZ[ix] += z;
	}
}

__global__ void correctdemag_kernel(IntCluster *pC, int cblock, int imgclusters, int demag, int width, int height)
{
	int w2 = width >> 1;
	int h2 = height >> 1;
	int ic, imin, imax;
	imin = ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x) * cblock;
	imax = imin + cblock;
	if (imax > imgclusters) imax = imgclusters;
	IntCluster *pc = pC + imin;
	for (ic = imin; ic < imax; ic++)
	{
		pc->X = (((pc->X - w2) * demag) >> DEMAG_SHIFT) + pc->X;
		pc->Y = (((pc->Y - h2) * demag) >> DEMAG_SHIFT) + pc->Y;
		pc++;
	}
}

__global__ void resetcounts_kernel(int *pmapcounts, int deltas2, int *pbest)
{
	int ix = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int i;
	if (ix < deltas2) pmapcounts[ix] = 0;
	if (ix == 0) *pbest = 0;//0xffffffff;
}

__global__ void makedeltas_kernel(int *pDeltas, int tol, int deltasx, int deltasy, short *pdx, short *pdy, int img)
{
	int i;
	for (i = 0; i < deltasx; i++)
		pDeltas[i] = tol * (i - deltasx / 2) + pdx[img];
	for (i = 0; i < deltasy; i++)
		pDeltas[i + deltasx] = tol * (i - deltasy / 2) + pdy[img];
}

__global__ void makedeltas_fromshift_kernel(int *pDeltas, int tol, int deltasx, int deltasy, int *pBestDeltas, int *pBest, int bestdeltasx)
{
	int best = *pBest & 0xffff;
	int biy = best / bestdeltasx;
	int bix = best % bestdeltasx;
	int	bestdx = pBestDeltas[bix];
	int bestdy = pBestDeltas[biy + bestdeltasx];	
	int i;
	for (i = 0; i < deltasx; i++)
		pDeltas[i] = tol * (i - deltasx / 2) + bestdx;
	for (i = 0; i < deltasy; i++)
		pDeltas[i + deltasx] = tol * (i - deltasy / 2) + bestdy;
}

__global__ void makefinaldeltas_fromshift_kernel(int *pDeltas, int tol, int *pBestDeltas, int *pBest, int bestdeltasx, short *px, short *py, short *pdx, short *pdy, int img, int totalimg)
{
	int best = *pBest & 0xffff;
	int	bestdx = pBestDeltas[best % bestdeltasx];
	int bestdy = pBestDeltas[best / bestdeltasx + bestdeltasx];	
	pDeltas[0] = bestdx;
	pDeltas[1] = bestdy;
	int corrx = bestdx - pdx[img];
	int corry = bestdy - pdy[img];
	px[img] += corrx;
	py[img] += corry;
	pdx[img] = bestdx;
	pdy[img] = bestdy;
	// THIS WAS A BUG: NO COMPENSATION OF CORRECTIONS! DO NOT RESTORE!
	if (img < totalimg - 1)
	{
		pdx[img + 1] -= corrx;
		pdy[img + 1] -= corry;
	}	
}

__global__ void rescaleshifts_kernel(short *px, short *py, short *pdx, short *pdy, int refimg, int totalimg)
{
	int dx = pdx[refimg];
	int dy = pdy[refimg];
	int img;
	for (img = 0; img < totalimg; img++)
	{
		px[img] -= dx;
		py[img] -= dy;
		pdx[img] -= dx;
		pdy[img] -= dy;
	}
}

__global__ void maphash_kernel(IntCluster *pC, int nc, int clusterblocksize, int i, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	int ic, ix, iy;
	Cell *qCell = 0;
	//for (i = 0; i < clusterblocksize; i++)
	{
		ic = idx * clusterblocksize + i;
		if (ic >= nc) return;
		ix = pC[ic].X / cellsize;
		if (ix < 0 || ix >= nx) return; //continue;
		iy = pC[ic].Y / cellsize;	
		if (iy < 0 || iy >= ny) return; //continue;
		qCell = pCell + iy * nx + ix;
		int a = atomicAdd(&qCell->Count, 1);
		if (a > maxcellcontent)
			atomicExch(&qCell->Count, maxcellcontent);
		else
		{
			IntCluster **qCellContents = pCellContents + maxcellcontent * (iy * nx + ix);
			qCellContents[a] = pC + ic;
		}
	}
}

__global__ void maphash_minarea_kernel(IntCluster *pC, int nc, int clusterblocksize, int i, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny, int minarea)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	int ic, ix, iy;
	Cell *qCell = 0;
	//for (i = 0; i < clusterblocksize; i++)
	{
		ic = idx * clusterblocksize + i;
		if (ic >= nc) return;
		if (pC[ic].Area < minarea) return; //continue;
		ix = pC[ic].X / cellsize;
		if (ix < 0 || ix >= nx) return; //continue;
		iy = pC[ic].Y / cellsize;	
		if (iy < 0 || iy >= ny) return; //continue;
		qCell = pCell + iy * nx + ix;
		int a = atomicAdd(&qCell->Count, 1);
		if (a > maxcellcontent)
			atomicExch(&qCell->Count, maxcellcontent);
		else
		{
			IntCluster **qCellContents = pCellContents + maxcellcontent * (iy * nx + ix);
			qCellContents[a] = pC + ic;
		}
	}
}

__global__ void clearhash_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, int cellsize, int nx, int ny)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	int i, ic, ix, iy;
	for (i = 0; i < clusterblocksize; i++)
	{
		ic = idx * clusterblocksize + i;
		if (ic >= nc) return;
		ix = pC[ic].X / cellsize;
		if (ix < 0 || ix >= nx) continue;
		iy = pC[ic].Y / cellsize;	
		if (iy < 0 || iy >= ny) continue;
		pCell[iy * nx + ix].Count = 0;
	}
}

__global__ void compactchains_kernel(IntChain *pCompact, int *pChainBase, IntChain *pOriginal, int *pChainCounts, int chainblocksize)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int i;
	int base = pChainBase[idx];
	for (i = 0; i < pChainCounts[idx]; i++)
		pCompact[base + i] = pOriginal[idx * chainblocksize + i];
}

__global__ void makechainwindow_kernel(ChainMapWindow *pChMapWnd, short *px, short *py, int imgs, int width, int height, float pxmicron, float pymicron, int maxcells, int mincellsize, ChainView *pChV, Cell *pCells, IntCluster **pCellContent, int maxcellcontent, int stagex, int stagey, ChainView *pChLastV)
{
	int sumx = (px[0] * pxmicron);
	int sumy = (py[0] * pymicron);
	int i;
	int minx = sumx;
	int maxx = sumx;
	int miny = sumy;
	int maxy = sumy;
	int w = abs(width * pxmicron);
	int h = abs(height * pymicron);
	for (i = 1; i < imgs; i++)
	{
		sumx = (px[i] * pxmicron);
		if (sumx < minx) minx = sumx;
		if (sumx > maxx) maxx = sumx;
		sumy = (py[i] * pymicron);
		if (sumy < miny) sumy = miny;
		if (sumy > maxy) sumy = maxy;
	}
	minx -= (w / 2);
	maxx += (w / 2);
	miny -= (h / 2);
	maxy += (h / 2);
	int cells = 2 * pChV->Count;
	if (cells > sqrt((float)maxcells)) cells = sqrt((float)maxcells);
	if (cells < 1) cells = 1;
	width = maxx - minx;
	height = maxy - miny;
	if (cells > width / mincellsize) cells = width / mincellsize;
	if (cells > height / mincellsize) cells = height / mincellsize;
	pChMapWnd->MinX = minx + stagex/* + (pChLastV ? pChLastV->DeltaX : 0)*/;
	pChMapWnd->MaxX = maxx + stagex/* + (pChLastV ? pChLastV->DeltaX : 0)*/;
	pChMapWnd->MinY = miny + stagey/* + (pChLastV ? pChLastV->DeltaY : 0)*/;
	pChMapWnd->MaxY = maxy + stagey/* + (pChLastV ? pChLastV->DeltaY : 0)*/;
	pChMapWnd->Width = width;
	pChMapWnd->Height = height;
	pChMapWnd->CellSize = __max(1, __max(width / cells, height / cells));
	pChMapWnd->MaxCellContent = maxcellcontent;
	pChMapWnd->NXCells = __max(1, width / pChMapWnd->CellSize);
	pChMapWnd->NYCells = __max(1, height / pChMapWnd->CellSize);
	pChMapWnd->pCells = pCells;
	pChMapWnd->pChains = (IntChain **)(void *)pCellContent;
}

__global__ void maphashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int chainblocksize, int i)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	int ic, ix, iy;
	Cell *qCell = 0;
	IntChain *pC = pChV->Chains;
	int nc = pChV->Count;
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;
	int minx = pChMapWnd->MinX;
	int miny = pChMapWnd->MinY;
	int cellsize = pChMapWnd->CellSize;
	int maxcellcontent = pChMapWnd->MaxCellContent;
	Cell *pCell = pChMapWnd->pCells;
	IntChain **pCellContents = pChMapWnd->pChains;
	//for (i = 0; i < chainblocksize; i++)
	{
		ic = idx * chainblocksize + i;
		if (ic >= nc) return;
		ix = (pC[ic].AvgX - minx) / cellsize;
		if (ix < 0 || ix >= nx) return; //continue;
		iy = (pC[ic].AvgY - miny) / cellsize;	
		if (iy < 0 || iy >= ny) return; //continue;
		qCell = pCell + iy * nx + ix;
		int c = atomicAdd(&qCell->Count, 1);
		if (c >= maxcellcontent)
			atomicExch(&qCell->Count, maxcellcontent);
		else
		{
			IntChain **qCellContents = pCellContents + maxcellcontent * (iy * nx + ix);
			qCellContents[c] = pC + ic;
		}		
	}
}

__global__ void makechaindeltas_kernel(int *pDeltas, int xytol, int ztol, int deltasx, int deltasy, int deltasz, ChainView *plastview, int xc, int yc, float xslant, float yslant, float dxdz, float dydz)
{
	plastview = 0;
	int i;
	for (i = 0; i < deltasx; i++)
		pDeltas[i] = xytol * (i - deltasx / 2) + (plastview ? plastview->DeltaX : 0);
	for (i = 0; i < deltasy; i++)
		pDeltas[i + deltasx] = xytol * (i - deltasy / 2) + (plastview ? plastview->DeltaY : 0);
	for (i = 0; i < deltasz; i++)
	{
		pDeltas[i + deltasx + deltasy] = ztol * (i - deltasz / 2) + (plastview ? (plastview->DeltaZ/* + (((int)(xslant * (xc - plastview->PositionX) + yslant * (yc - plastview->PositionY))) >> (XY_SCALE_SHIFT - Z_SCALE_SHIFT))*/) : 0);
		pDeltas[i + deltasx + deltasy + deltasz] = (ztol * (i - deltasz / 2) * dxdz) * (1 << (XY_SCALE_SHIFT - Z_SCALE_SHIFT));//(pDeltas[i + deltasx + deltasy] * dxdz) * (1 << (XY_SCALE_SHIFT - Z_SCALE_SHIFT));
		pDeltas[i + deltasx + deltasy + 2 * deltasz] = (ztol * (i - deltasz / 2) * dydz) * (1 << (XY_SCALE_SHIFT - Z_SCALE_SHIFT));//(pDeltas[i + deltasx + deltasy] * dxdz) * (1 << (XY_SCALE_SHIFT - Z_SCALE_SHIFT));
	}
}

__global__ void makechaindeltas_fromshift_kernel(int *pDeltas, int xytol, int ztol, int deltasx, int deltasy, int deltasz, int *pBestDeltas, int *pBest, int bestdeltasx, int bestdeltasy, int bestdeltasz)
{
	int best = (*pBest) & 0xffff;
	int bdxy2 = bestdeltasx * bestdeltasy;
	int biz = best / bdxy2;
	int biy = (best % bdxy2) / bestdeltasx;
	int bix = best % bestdeltasx;
	int	bestdx = pBestDeltas[bix] + pBestDeltas[biz + bestdeltasx + bestdeltasy + bestdeltasz];
	int bestdy = pBestDeltas[biy + bestdeltasx] + pBestDeltas[biz + bestdeltasx + bestdeltasy + 2 * bestdeltasz];
	int bestdz = pBestDeltas[biz + bestdeltasx + bestdeltasy];
	int i;
	for (i = 0; i < deltasx; i++)
		pDeltas[i] = xytol * (i - deltasx / 2) + bestdx;
	for (i = 0; i < deltasy; i++)
		pDeltas[deltasx + i] = xytol * (i - deltasy / 2) + bestdy;
	for (i = 0; i < deltasz; i++)
		pDeltas[deltasx + deltasy + i] = ztol * (i - deltasz / 2) + bestdz;
}

__global__ void setchainviewheader_kernel(ChainMapHeader *pmaph, ChainView *pview, int px, int py, int pz, int *pDeltas, int deltasXY, int deltasZ, int *pD)
{
	pview->PositionX = px;
	pview->PositionY = py;
	pview->PositionZ = pz;	
	pview->DeltaX = pview->DeltaY = pview->DeltaZ = 0;
	/*
	if (pDeltas == 0)
	{
		pview->DeltaX = pview->DeltaY = pview->DeltaZ = 0;
		pmaph->Views = 1;
	}
	else
	{
		int best = (*pD) & 0xffff;
		pview->DeltaX = pDeltas[best % deltasXY];
		pview->DeltaY = pDeltas[deltasXY + ((best % (deltasXY * deltasXY)) / deltasXY)];
		pview->DeltaZ = pDeltas[2 * deltasXY + (best / (deltasXY * deltasXY))];
		pmaph->Views++;
	}
	*/
}

/*****************************/

__global__ void compact_kernel(int * pInt, int stride, int count, int * pOut)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (i < count) pOut[i] = pInt[i * stride];
}

__global__ void max_check_kernel(int * pInt, int total, int halftotal)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (i + halftotal < total)
		pInt[i] = __max(pInt[i], pInt[i + halftotal]);
}

__global__ void max_kernel(int * pInt, int halftotal)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	pInt[i] = __max(pInt[i], pInt[i + halftotal]);
}

__global__ void sum_check_kernel(int * pInt, int total, int halftotal)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (i + halftotal < total)
		pInt[i] += pInt[i + halftotal];
}

__global__ void sum_kernel(int * pInt, int halftotal)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	pInt[i] += pInt[i + halftotal];
}

__global__ void sum_check_multiple_kernel(int * pInt, int total, int halftotal)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (i + halftotal < total)
	{
		i += blockIdx.z * total;		
		pInt[i] += pInt[i + halftotal];
	}
}

__global__ void sum_multiple_kernel(int * pInt, int total, int halftotal)
{
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x + (blockIdx.z * total);
	pInt[i] += pInt[i + halftotal];
}

__global__ void shift_postfixid_kernel(int *pdest, int *psrc, int total)
{
	unsigned id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (id >= total) return;
	pdest[id] = (psrc[id] << 16) | (id & 0xffff);
}

__global__ void split_and_index_kernel(int *paircomputer, int depth, IntPair *pairindices, int totalpairs)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (id >= totalpairs) return;
	int d;
	int res = id + 1;
	int countatlevel = 2;
	int place = 0;
	for (d = 1; d < depth; d++)
	{
		place <<= 1;
		if (paircomputer[place] < res)
		{
			res -= paircomputer[place];
			place++;
		}
		paircomputer -= countatlevel;			
		countatlevel <<= 1;
	}
	pairindices[id].Index1 = place;
	pairindices[id].Index2 = res - 1;
}

__global__ void trymap2_prepare_clusters_kernel(IntCluster *pc, IntMapCluster *pmc, int totalclusters, int divider, int mingrainsize, int w2, int h2, int demag, int *pValidFlag)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int idd = id * divider;
	if (idd >= totalclusters) return;
	pmc += id;	
	pc += idd;
	if (pc->Area < mingrainsize)
	{
		pmc->idoriginal = -1;
		if (pValidFlag) pValidFlag[id] = 0;
		return;
	}
	pmc->idoriginal = idd;				
	pmc->ibasex = (((pc->X - w2) * demag) >> DEMAG_SHIFT) + pc->X;
	pmc->ibasey = (((pc->Y - h2) * demag) >> DEMAG_SHIFT) + pc->Y;
	if (pValidFlag) pValidFlag[id] = 1;
}

__global__ void trymap2_shift_kernel(IntMapCluster *pmc, int totalmapclusters, int *pDeltaX, int *pDeltaY, int cellsize)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalmapclusters) return;
	pmc += id;	
	pmc->icell = -1;
	if (pmc->idoriginal < 0) return;
	pmc->ishiftedx = pmc->ibasex + *pDeltaX;
	pmc->icellx = pmc->ishiftedx / cellsize;
	pmc->ishiftedy = pmc->ibasey + *pDeltaY;	
	pmc->icelly = pmc->ishiftedy / cellsize;
}

__global__ void trymap2_shiftmatch_kernel(IntMapCluster *pmc, IntPair *pPairs, int totalpairs, int *pDeltas, int cellsize, short nx, short ny, int *pmatchresult, int tol, Cell *pmapcell, IntCluster **pMapCellContent, int maxcellcontent, short deltas)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalpairs) return;	
	pmc += pPairs[id].Index1;
	int iy = blockIdx.z / deltas;
	int ix = blockIdx.z - iy * deltas;
	int ishiftedx = pmc->ibasex + pDeltas[ix];
	int ishiftedy = pmc->ibasey + pDeltas[deltas + iy];
	int icellx = ishiftedx / cellsize;
	int icelly = ishiftedy / cellsize;
	pmatchresult += (iy * deltas + ix) * totalpairs + id;
	int imatchresult = 0;
	if (icellx >= 0 && icellx < nx && icelly >= 0 && icelly < ny)	
	{
		int icell = icelly * (int)nx + icellx;
		pMapCellContent += icell * maxcellcontent;
		short i = pmapcell[icell].Count;
		while (--i >= 0)
		{
			IntCluster *pc2 = pMapCellContent[i];			
			imatchresult |= (abs(ishiftedx - pc2->X) < tol && abs(ishiftedy - pc2->Y) < tol) ? 1 : 0;			
		}
	}
    *pmatchresult = imatchresult;
}

__global__ void finalmap_cell_kernel(IntMapCluster *pmc, int totalmapclusters, Cell *pmapcell, int *pClustersInCell, int nx, int ny)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalmapclusters) return;
	pmc += id;	
	pClustersInCell[id] = 0;	
	pmc->icell = -1;
	if (pmc->idoriginal < 0) return;
	int mapclusters = 0;
#pragma unroll 3
	for (int iiy = -1; iiy <= 1; iiy++)
	{
		int icelly = pmc->icelly + iiy;	
		if (icelly < 0 || icelly >= ny) continue;
#pragma unroll 3
		for (int iix = -1; iix <= 1; iix++)
		{
			int icellx = pmc->icellx + iix;
			if (icellx < 0 || icellx >= nx) continue;
			mapclusters += pmapcell[icelly * nx + icellx].Count;
		}
	}
	pClustersInCell[id] = pmc->ipairblockcount = mapclusters;
}

__global__ void finalmap_match_kernel(IntMapCluster *pmc, IntPair *pPairs, int totalpairs, int *pmatchresult, int *pmatchmap, int tol, Cell *pmapcell, IntCluster **pMapCellContent, int maxcellcontent, int nx, int ny)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalpairs) return;
	pmc += pPairs[id].Index1;	
	if (pmc->idoriginal < 0) return;
	if (pPairs[id].Index2 == 0) pmc->ipairblockstart = id;
	int mapclusters = 0;
#pragma unroll 3
	for (int iiy = -1; iiy <= 1; iiy++)
	{
		int icelly = pmc->icelly + iiy;	
		if (icelly < 0 || icelly >= ny) continue;
#pragma unroll 3
		for (int iix = -1; iix <= 1; iix++)
		{
			int icellx = pmc->icellx + iix;
			if (icellx < 0 || icellx >= nx) continue;	
			int inc = pmapcell[icelly * nx + icellx].Count;
			if (mapclusters + inc <= pPairs[id].Index2) 
			{
				mapclusters += inc;
			}
			else
			{
				int idc2 = (icelly * nx + icellx) * maxcellcontent + pPairs[id].Index2 - mapclusters;
				IntCluster *pc2 = pMapCellContent[idc2];
				int dist = __max(abs(pmc->ishiftedx - pc2->X), abs(pmc->ishiftedy - pc2->Y));
				if (dist < tol)
				{
					pmatchresult[id] = dist;
					pmatchmap[id] = idc2;
				}
				else
				{
					pmatchresult[id] = pmatchmap[id] = -1;
				}
				return;
			}			
		}		
	}
}

__global__ void finalmap_optimize_kernel(IntCluster *pc, IntMapCluster *pmc, int clusteroffset, int totalclusters, int *pmatchresult, int *pmatchmap, IntCluster **pMapCellContent, IntCluster **pClusterChain)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalclusters) return;
	pmc += id;
	if (pmc->idoriginal < 0) return;
	if (pmc->ipairblockcount <= 0) return;
	int i = pmc->ipairblockcount - 1;
	int ibest = i;
	int iblockstart = pmc->ipairblockstart;
	int d;
	int dbest = pmatchresult[iblockstart + i];
	while (--i >= 0)
	{
		d = pmatchresult[iblockstart + i];
		if (d >= 0 && (dbest < 0 || dbest > d))
		{
			ibest = i;
			dbest = d;
		}
	}	
	if (0/*KRYSS DISABLE CHAIN FORMATION 20140728 dbest >= 0*/)
	{
		IntCluster *pBest = pMapCellContent[pmatchmap[iblockstart + ibest]];
		pBest->Area = -abs(pBest->Area);
		pc += pmc->idoriginal;
		pc->X += (pBest->X - pmc->ibasex);
		pc->Y += (pBest->Y - pmc->ibasey);
		pClusterChain[clusteroffset + pmc->idoriginal] = pBest;
	}
	else pClusterChain[clusteroffset + pmc->idoriginal] = 0;
}

__global__ void makechain_kernel(IntCluster *pC, int totalclusters, short w2, short h2, short *pClusterXs, short *pClusterYs, short *pClusterZs, int xslant, int yslant, IntCluster **pClusterChains, short minclusters, short minvol, float xtomicron, float ytomicron, int stagex, int stagey, IntChain *pChain, int viewtag, int *pvalid)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (id >= totalclusters) return;	
	pvalid[id] = pChain[id].Volume = pChain[id].Clusters = 0;		
	int avgx = 0;
	int avgy = 0;
	int avgz = 0;	
	IntCluster *pnc = pC + id;
	if (pnc->Area < 0) return;
	short clusters = 1;
	int area = abs(pnc->Area);
	int volume = area;
	avgx = (pnc->X + pClusterXs[id]) * area;
	avgy = (pnc->Y + pClusterYs[id]) * area;		
	avgz = area * (pClusterZs[id] -  (( (xslant * (pnc->X - w2) + yslant * (pnc->Y - h2)) >> SLOPE_SHIFT)));
	int ip;
	while (pnc = pClusterChains[ip = pnc - pC])
	{	
		ip = pnc - pC;			
		clusters++;			
		area = abs(pnc->Area);			
		volume += area;						
		avgx += area * (pnc->X + pClusterXs[ip]);
		avgy += area * (pnc->Y + pClusterYs[ip]);
		avgz += area * (pClusterZs[ip] -  (( (xslant * (pnc->X - w2) + yslant * (pnc->Y - h2)) >> SLOPE_SHIFT)));			
	}	
	if (clusters >= minclusters && volume >= minvol)
	{
		IntChain *psC = pChain + id;
		psC->Clusters = clusters;
		avgx /= volume;
		avgy /= volume;
		avgz /= volume;
		psC->Volume = volume;
		psC->AvgX = ((avgx - w2) << XY_SCALE_SHIFT) * xtomicron + stagex;
		psC->AvgY = ((avgy - h2) << XY_SCALE_SHIFT) * ytomicron + stagey;
		psC->AvgZ = avgz;
		psC->ViewTag = viewtag;
		psC->Reserved = 0;
		pvalid[id] = 1;
	}
}

__global__ void maphashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx * divider >= pChV->Count) return;		
	IntChain *pC = pChV->Chains + idx * divider;	
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;
	int minx = pChMapWnd->MinX;
	int miny = pChMapWnd->MinY;
	int cellsize = pChMapWnd->CellSize;
	int maxcellcontent = pChMapWnd->MaxCellContent;	
	int ix, iy;
	ix = (pC->AvgX - minx) / cellsize;
	if (ix < 0 || ix >= nx) return;
	iy = (pC->AvgY - miny) / cellsize;	
	if (iy < 0 || iy >= ny) return;
	Cell *qCell = pChMapWnd->pCells + iy * nx + ix;
	int c = atomicAdd(&qCell->Count, 1);
	if (c >= maxcellcontent) atomicExch(&qCell->Count, maxcellcontent);
	else pChMapWnd->pChains[maxcellcontent * (iy * nx + ix) + c] = pC;
}

__global__ void clearhashchain2_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx * divider >= pChV->Count) return;		
	IntChain *pC = pChV->Chains + idx * divider;	
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;
	int minx = pChMapWnd->MinX;
	int miny = pChMapWnd->MinY;
	int cellsize = pChMapWnd->CellSize;
	int maxcellcontent = pChMapWnd->MaxCellContent;	
	int ix, iy;
	ix = (pC->AvgX - minx) / cellsize;
	if (ix < 0 || ix >= nx) return;
	iy = (pC->AvgY - miny) / cellsize;	
	if (iy < 0 || iy >= ny) return;
	pChMapWnd->pCells[iy * nx + ix].Count = 0;
}

__global__ void trymapchain_prepare_chains_kernel(IntChain *pc, IntMapChain *pmc, int totalchains, int minchainsize, int *pValidFlag)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalchains) return;
	pmc += id;	
	pc += id;
	if (pValidFlag) pValidFlag[id] = 0;
	if (pc->Volume < minchainsize)
	{
		pmc->idoriginal = -1;
		return;
	}
	pmc->idoriginal = id;				
	pmc->ibasex = pc->AvgX;
	pmc->ibasey = pc->AvgY;
	pmc->ibasez = pc->AvgZ;
	if (pValidFlag) pValidFlag[id] = 1;
}

__global__ void trymapchain_shiftmatch_kernel(IntMapChain *pmc, IntPair *pPairs, int totalpairs, int *pMapCount, int *pDeltas, ChainMapWindow *pChMapWnd, int xytol, short zsteps, int ztol, short deltas)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalpairs) return;	
	pmc += pPairs[id].Index1;	
	int iy = blockIdx.z / deltas;
	int ix = blockIdx.z - iy * deltas;
	int ishiftedx = pmc->ibasex + pDeltas[ix];
	int ishiftedy = pmc->ibasey + pDeltas[deltas + iy];
	int ibasez = pmc->ibasez;
	int icellx = (ishiftedx - pChMapWnd->MinX) / pChMapWnd->CellSize;
	int icelly = (ishiftedy - pChMapWnd->MinY) / pChMapWnd->CellSize;
	pMapCount += blockIdx.z;
	if (icellx >= 0 && icellx < pChMapWnd->NXCells && icelly >= 0 && icelly < pChMapWnd->NYCells)	
	{
		int icell = icelly * (int)pChMapWnd->NXCells + icellx;
		IntChain **pCellContent = pChMapWnd->pChains + icell * pChMapWnd->MaxCellContent;
		short i = pChMapWnd->pCells[icell].Count;
		for (int iz = zsteps - 1; iz >= 0; iz--)
		{
			bool hasmatch = 0;
			while (--i >= 0)
			{
				IntChain *pc2 = pCellContent[i];
				hasmatch = hasmatch || (abs(ishiftedx - pc2->AvgX) < xytol && abs(ishiftedy - pc2->AvgY) < xytol && abs(ibasez + pDeltas[gridDim.z + iz] - pc2->AvgZ) < ztol);
			}
			if (hasmatch) atomicAdd(pMapCount + (iz * gridDim.z), 1);
		}
	}
}

__global__ void make_finalchainshift_kernel(int *pDeltas, int *pRefineDeltas, int *pBest, int deltasXY)
{
	int best = *pBest & 0xffff;
	int dix = pRefineDeltas[best % deltasXY];
	int diy = pRefineDeltas[deltasXY + ((best % (deltasXY * deltasXY)) / deltasXY)];
	int diz = pRefineDeltas[2 * deltasXY + (best / (deltasXY * deltasXY))];
	pDeltas[0] = dix;
	pDeltas[1] = diy;
	pDeltas[2] = diz;
}

__global__ void finalmapchain_cell_kernel(IntMapChain *pmc, IntPair *pPairs, int totalpairs, int *pDeltas, ChainMapWindow *pChMapWnd, int *pvalid)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalpairs) return;	
	pmc += pPairs[id].Index1;		
	pmc->ishiftedx = pmc->ibasex + pDeltas[0];
	pmc->ishiftedy = pmc->ibasey + pDeltas[1];
	pmc->ishiftedz = pmc->ibasez + pDeltas[2];
	int icellx = (pmc->ishiftedx - pChMapWnd->MinX) / pChMapWnd->CellSize;
	int icelly = (pmc->ishiftedy - pChMapWnd->MinY) / pChMapWnd->CellSize;
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;	
	if (icelly < 0 || icelly >= ny || icellx < 0 || icellx >= nx) 
	{
		pvalid[pPairs[id].Index1] = 0;
		return;
	}	
	pvalid[pPairs[id].Index1] = pChMapWnd->pCells[pmc->icell = icelly * (int)nx + icellx].Count;	
}

__global__ void finalmapchain_match_kernel(IntChain *pc, IntMapChain *pmc, IntPair *pPairs, int totalpairs, ChainMapWindow *pChMapWnd, int xytol, int ztol)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalpairs) return;	
	pmc += pPairs[id].Index1;		
	IntChain *pc2 = pChMapWnd->pChains[pmc->icell * pChMapWnd->MaxCellContent + pPairs[id].Index2];
	if (abs(pmc->ishiftedx - pc2->AvgX) < xytol && abs(pmc->ishiftedy - pc2->AvgY) < xytol && abs(pmc->ishiftedz - pc2->AvgZ) < ztol)
	{
		pc[pPairs[id].Index1].Volume = pc[pPairs[id].Index1].Clusters = 0;
	}
}

__global__ void finalmapchain_filter_kernel(IntChain *pc, int totalchains, int *pvalid)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id >= totalchains) return;
	pvalid[id] = (pc[id].Volume > 0) ? 1 : 0;
}

__global__ void compactchains_kernel(IntChain *pcmpct, IntChain *pch, IntPair *pPairs, int totalpairs, ChainView *pChV)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	if (id == 0) pChV->Count = totalpairs;
	if (id >= totalpairs) return;
	pcmpct[id] = pch[pPairs[id].Index1];
}

__global__ void negshift_viewchains_kernel(ChainView *pview, int *pDeltas)
{
	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= pview->Count) return;
	int dix = pDeltas[0];
	int diy = pDeltas[1];
	int diz = pDeltas[2];
	IntChain *pC = (IntChain *)(void *)((char *)(void *)pview + sizeof(ChainView)) + idx;
	pC->AvgX -= dix;
	pC->AvgY -= diy;
	pC->AvgZ -= diz;
}


/*****************************/

} }