#include "gpu_map_kernels.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace SySal { namespace GPU {

__global__ void curvaturemap_kernel(int *pXYCurv, int *pZCurv, int span, float xy_curvature, float z_curvature)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
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
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
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
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
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
	imin = (threadIdx.x + blockIdx.x * blockDim.x) * cblock;
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
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void findbest_kernel(int *pMapCounts, int deltas2, int step, int *pBest)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2 * step;
	if (step == 1)
	{
		if (idx < deltas2) pBest[idx] = idx;
		if (idx + step < deltas2) pBest[idx + step] = idx + step;
	}
	if (idx < deltas2 && idx + step < deltas2)
	{
		if (pMapCounts[idx] < pMapCounts[idx + step])
		{
			pBest[idx] = pBest[idx + step];
			pMapCounts[idx] = pMapCounts[idx + step];
		}
	}	
}

__global__ void sumcounts_kernel(int *pMapCounts, int deltas2, int total, int step)
{
	int idelta = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ((threadIdx.x + blockIdx.x * blockDim.x) * step) << 1;
	if (idelta < deltas2 && (idx + step) < total)
		pMapCounts[idx * deltas2 + idelta] += pMapCounts[(idx + step) * deltas2 + idelta];	
}

__global__ void trymap_Ikernel(trymap_kernel_args * __restrict__ pargs, trymap_kernel_status * __restrict__ pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	_IKGPU_RESUMEFROM(1, pstatus);

	pstatus->deltas2 = pargs->deltasx * pargs->deltasy;
	pstatus->w2 = pargs->w >> 1;
	pstatus->h2 = pargs->h >> 1;
	for (pstatus->i = 0; pstatus->i < pargs->clusterblocksize; pstatus->i += pargs->sampledivider)
		if ((pstatus->ic = idx * pargs->clusterblocksize + pstatus->i) < pargs->nc)
		{
			_IKGPU_INTERRUPT(1, pstatus, pargs);
			IntCluster *pc = pargs->pC + pstatus->ic;
			if (pc->Area < pargs->minclustersize) continue;			
			int ibasex = (((pc->X - pstatus->w2) * pargs->demag) >> DEMAG_SHIFT) + pc->X;
			int ibasey = (((pc->Y - pstatus->h2) * pargs->demag) >> DEMAG_SHIFT) + pc->Y;
			for (short dix = 0; dix < pargs->deltasx; dix++)
			{
				short ideltax = ibasex + pargs->pDeltas[dix];
				short icellx = ideltax / pargs->cellsize;
				for (short diy = 0; diy < pargs->deltasy; diy++)
				{
					short ideltay = ibasey + pargs->pDeltas[diy + pargs->deltasx];
					short icelly = ideltay / pargs->cellsize;
					int tol = pargs->tol;
					bool increment = false;
#pragma unroll 3
					for (short iix = -1; iix <= 1; iix++)
					{
						short icellix = icellx + iix;
						if (icellix < 0 || icellix >= pargs->nx) continue;
#pragma unroll 3
						for (short iiy = -1; iiy <= 1; iiy++)
						{
							short icelliy = icelly + iiy;
							if (icelliy < 0 || icelliy >= pargs->ny) continue;
							short icellcls;
							Cell * __restrict__ pThisCell = pargs->pCell + icelliy * pargs->nx + icellix;
							IntCluster ** __restrict__ ppcls = pargs->pCellContent + (icelliy * pargs->nx + icellix) * pargs->maxcellcontent;							
							for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
								if (ppcls[icellcls]->Area >= pargs->minclustersize && abs(ideltax - ppcls[icellcls]->X) < tol && abs(ideltay - ppcls[icellcls]->Y) < tol)									
								{				
									increment = true;
								}
						}
					}
					if (increment)
						atomicMax(pargs->pBest, ((atomicAdd(pargs->pMapCounts + diy * pargs->deltasx + dix, 1) + 1) << 16) + diy * pargs->deltasx + dix);
				}
			}
		}
	_IKGPU_END(pstatus);
}

__global__ void trymap2_Ikernel(trymap_kernel_args * __restrict__ pargs, trymap_kernel_status * __restrict__ pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tol = pargs->tol;
	int nx = pargs->nx;
	int ny = pargs->ny;
	int maxcellcontent = pargs->maxcellcontent;
	short w2 = pargs->w >> 1;
	short h2 = pargs->h >> 1;
	int deltasx = pargs->deltasx;
	int deltasy = pargs->deltasy;	
	int cellsize = pargs->cellsize;
	int demag = pargs->demag;
	int *pDeltas = pargs->pDeltas;
	_IKGPU_RESUMEFROM(1, pstatus);

	for (pstatus->i = 0; pstatus->i < pargs->clusterblocksize; pstatus->i += pargs->sampledivider)
		if ((pstatus->ic = idx * pargs->clusterblocksize + pstatus->i) < pargs->nc)
		{
			_IKGPU_INTERRUPT(1, pstatus, pargs);
			IntCluster *pc = pargs->pC + pstatus->ic;	
			if (pc->Area < pargs->minclustersize) continue;
			int ibasex = (((pc->X - w2) * demag) >> DEMAG_SHIFT) + pc->X;
			int ibasey = (((pc->Y - h2) * demag) >> DEMAG_SHIFT) + pc->Y;
			for (short dix = 0; dix < deltasx; dix++)
			{
				short ideltax = ibasex + pDeltas[dix];
				short icellx = ideltax / cellsize;
				if (icellx < 0 || icellx >= nx) continue;
				for (short diy = 0; diy < deltasy; diy++)
				{
					short ideltay = ibasey + pDeltas[diy + deltasx];
					short icelly = ideltay / cellsize;		
					if (icelly < 0 || icelly >= ny) continue;
					bool increment = false;					
					int idcell = icelly * nx + icellx;
					Cell * pThisCell = pargs->pCell + idcell;							
					short icellcls;														
					IntCluster ** ppcls = pargs->pCellContent + idcell * maxcellcontent;							
					for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
						if (abs(ideltax - ppcls[icellcls]->X) < tol && abs(ideltay - ppcls[icellcls]->Y) < tol)									
						{				
							increment = true;
						}
					if (increment)
						atomicMax(pargs->pBest, ((atomicAdd(pargs->pMapCounts + diy * pargs->deltasx + dix, 1) + 1) << 16) + diy * pargs->deltasx + dix);
				}
			}
		}
	_IKGPU_END(pstatus);
}

__global__ void refinemap_Ikernel(refinemap_kernel_args * __restrict__ pargs, refinemap_kernel_status * __restrict__ pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	_IKGPU_RESUMEFROM(1, pstatus);	
	int dix, diy;
	pstatus->dx = pargs->pDeltas[pargs->deltas / 2];
	pstatus->dy = pargs->pDeltas[pargs->deltas / 2 + pargs->deltas];
	pstatus->w2 = pargs->w >> 1;
	pstatus->h2 = pargs->h >> 1;
	pstatus->refinedeltas = pargs->tol / pargs->refinebin;
	pstatus->refinespan = 2 * pstatus->refinedeltas + 1;
	pstatus->refinespan2 = pstatus->refinespan * pstatus->refinespan;
	for (pstatus->i = 0; pstatus->i < pargs->clusterblocksize; pstatus->i++)
		if ((pstatus->ic = idx * pargs->clusterblocksize + pstatus->i) < pargs->nc)
		{
			_IKGPU_INTERRUPT(1, pstatus, pargs);
			int dist = pargs->tol;			
			IntCluster *pBest = 0;
			pstatus->pc = pargs->pC + pstatus->ic;
			int ibasex = (((pstatus->pc->X - pstatus->w2) * pargs->demag) >> DEMAG_SHIFT) + pstatus->pc->X + pstatus->dx;
			int ibasey = (((pstatus->pc->Y - pstatus->h2) * pargs->demag) >> DEMAG_SHIFT) + pstatus->pc->Y + pstatus->dy;
			short icellx = ibasex / pargs->cellsize;			
			short icelly = ibasey / pargs->cellsize;					
#pragma unroll 3
			for (short iix = -1; iix <= 1; iix++)
			{
				short icellix = icellx + iix;
				if (icellix < 0 || icellix >= pargs->nx) continue;
#pragma unroll 3
				for (short iiy = -1; iiy <= 1; iiy++)						
				{								
					short icelliy = icelly + iiy;
					if (icelliy < 0 || icelliy >= pargs->ny) continue;					
					short icellcls;
					Cell * __restrict__ pThisCell = pargs->pCell + icelliy * pargs->nx + icellix;
					IntCluster ** __restrict__ ppcls = pargs->pCellContent + (icelliy * pargs->nx + icellix) * pargs->maxcellcontent;
					for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
					{
						int distc = __max(abs(ibasex - ppcls[icellcls]->X), abs(ibasey - ppcls[icellcls]->Y));
						if (distc < dist)
						{
							pBest = ppcls[icellcls];
							dist = distc;
						}					
					}
				}
			}
			if (pBest) 
			{				
				dix = (pBest->X - ibasex) / pargs->refinebin + pstatus->refinedeltas;
				diy = (pBest->Y - ibasey) / pargs->refinebin + pstatus->refinedeltas;
				if (dix >= 0 && dix < pstatus->refinespan && diy >= 0 && diy < pstatus->refinespan)
					//pargs->pMapCounts[idx * pstatus->refinespan2 + diy * pstatus->refinespan + dix]++;
				{									
					atomicMax(pargs->pBest, ((atomicAdd(pargs->pMapCounts + diy * pstatus->refinespan + dix, 1) + 1) << 16) + diy * pstatus->refinespan + dix);
				}
			}			
		}
	_IKGPU_END(pstatus);
}


__global__ void finalmap_Ikernel(finalmap_kernel_args * __restrict__ pargs, finalmap_kernel_status * __restrict__ pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;	
	_IKGPU_RESUMEFROM(1, pstatus);
	pstatus->w2 = pargs->w >> 1;
	pstatus->h2 = pargs->h >> 1;
	pstatus->dx = pargs->pDX[pargs->img];
	pstatus->dy = pargs->pDY[pargs->img];	
	pargs->pMapCounts[idx] = 0;			
	for (pstatus->i = 0; pstatus->i < pargs->clusterblocksize; pstatus->i++)
		if ((pstatus->ic = idx * pargs->clusterblocksize + pstatus->i) < pargs->nc)
		{
			_IKGPU_INTERRUPT(1, pstatus, pargs);
			int dist = pargs->tol;			
			IntCluster *pBest = 0;
			pstatus->pc = pargs->pC + pstatus->ic;
			int ibasex = (((pstatus->pc->X - pstatus->w2) * pargs->demag) >> DEMAG_SHIFT) + pstatus->pc->X + pstatus->dx;
			int ibasey = (((pstatus->pc->Y - pstatus->h2) * pargs->demag) >> DEMAG_SHIFT) + pstatus->pc->Y + pstatus->dy;
			short icellx = ibasex / pargs->cellsize;			
			short icelly = ibasey / pargs->cellsize;								
#pragma unroll 3
			for (short iix = -1; iix <= 1; iix++)
			{
				short icellix = icellx + iix;
				if (icellix < 0 || icellix >= pargs->nx) continue;
#pragma unroll 3
				for (short iiy = -1; iiy <= 1; iiy++)						
				{								
					short icelliy = icelly + iiy;
					if (icelliy < 0 || icelliy >= pargs->ny) continue;					
					short icellcls;
					Cell * __restrict__ pThisCell = pargs->pCell + icelliy * pargs->nx + icellix;
					IntCluster ** __restrict__ ppcls = pargs->pCellContent + (icelliy * pargs->nx + icellix) * pargs->maxcellcontent;
					for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
					{
						int distc = __max(abs(ibasex - ppcls[icellcls]->X), abs(ibasey - ppcls[icellcls]->Y));
						if (distc < dist)
						{
							pBest = ppcls[icellcls];
							dist = distc;
						}					
					}
				}
			}
			pargs->pClusterChain[pstatus->pc - pargs->pBase] = pBest;
			if (pBest) 
			{				
				pBest->Area = -pBest->Area;
				pstatus->pc->X += (pBest->X - ibasex);
				pstatus->pc->Y += (pBest->Y - ibasey);
				pargs->pMapCounts[idx]++;
			}			
		}
	_IKGPU_END(pstatus);
}

__global__ void maphash_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadlock = idx + 1;
	int i, ic, ix, iy;
	Cell *qCell = 0;
	for (i = 0; i < clusterblocksize; i++)
	{
		ic = idx * clusterblocksize + i;
		if (ic >= nc) return;
		ix = pC[ic].X / cellsize;
		if (ix < 0 || ix >= nx) continue;
		iy = pC[ic].Y / cellsize;	
		if (iy < 0 || iy >= ny) continue;
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

__global__ void maphash_minarea_kernel(IntCluster *pC, int nc, int clusterblocksize, Cell *pCell, IntCluster **pCellContents, int cellsize, int maxcellcontent, int nx, int ny, int minarea)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadlock = idx + 1;
	int i, ic, ix, iy;
	Cell *qCell = 0;
	for (i = 0; i < clusterblocksize; i++)
	{
		ic = idx * clusterblocksize + i;
		if (ic >= nc) return;
		if (pC[ic].Area < minarea) continue;
		ix = pC[ic].X / cellsize;
		if (ix < 0 || ix >= nx) continue;
		iy = pC[ic].Y / cellsize;	
		if (iy < 0 || iy >= ny) continue;
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
	int idx = threadIdx.x + blockIdx.x * blockDim.x;	
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
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i;
	int base = pChainBase[idx];
	for (i = 0; i < pChainCounts[idx]; i++)
		pCompact[base + i] = pOriginal[idx * chainblocksize + i];
}

__global__ void setchainbase_kernel(int *pChainBase, int *pChainCounts, int counts, ChainMapHeader *pCh, ChainView *pChV, int px, int py, int pz, int w, int h)
{
	int i;
	pChainBase[0] = 0;
	for (i = 1; i < counts; i++)
		pChainBase[i] = pChainBase[i - 1] + pChainCounts[i - 1];	
	pChV->Count = pChainBase[i - 1] + pChainCounts[i - 1];
	pChV->PositionX = px;
	pChV->PositionY = py;
	pChV->PositionZ = pz;	
	/*
	if (pCh->Views == 0)
	{
		pCh->Width = w;
		pCh->Height = h;
	}
	*/
}

__global__ void makechain_Ikernel(makechain_kernel_args * __restrict__ pargs, makechain_kernel_status * __restrict__ pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	_IKGPU_RESUMEFROM(1, pstatus);
	pstatus->w2 = pargs->width / 2;
	pstatus->h2 = pargs->height / 2;
	pstatus->istart = ix * pargs->clusterblocksize;
	pstatus->iend = __min(pargs->totalclusters, pstatus->istart + pargs->clusterblocksize);	
	pargs->pChainCounts[ix] = pstatus->count = 0;		
	for (pstatus->i = pstatus->istart; pstatus->i < pstatus->iend; pstatus->i++)
	{
		_IKGPU_INTERRUPT(1, pstatus, pargs);
		int avgx = 0;
		int avgy = 0;
		int avgz = 0;
		int dz = 0;
		IntCluster *pnc = pargs->pC + pstatus->i;
		if (pnc->Area < 0) continue;		
		pstatus->ch.Clusters = 1;
		int area = abs(pnc->Area);
		pstatus->ch.Volume = area;
		avgx = (pnc->X + pargs->pClusterXs[pstatus->i]) * area;
		avgy = (pnc->Y + pargs->pClusterYs[pstatus->i]) * area;		
		//avgz = pargs->pClusterZs[pstatus->i] * area;
		avgz = area * (pargs->pClusterZs[pstatus->i] -  (( (pargs->xslant * (pnc->X - pstatus->w2) + pargs->yslant * (pnc->Y - pstatus->h2)) >> SLOPE_SHIFT)));			
		//pstatus->ch.TopZ = pstatus->ch.BottomZ = pargs->pClusterZs[pstatus->i];
		//pstatus->ch.DeltaX = pstatus->ch.DeltaY = pstatus->ch.DeltaZ = pargs->pClusterZs[pstatus->i];		
		while (pnc = pargs->pClusterChains[pnc - pargs->pC])
		{	
			int ip = pnc - pargs->pC;			
			pstatus->ch.Clusters++;			
			area = abs(pnc->Area);			
			pstatus->ch.Volume += area;						
			avgx += area * (pnc->X + pargs->pClusterXs[ip]);
			avgy += area * (pnc->Y + pargs->pClusterYs[ip]);
			avgz += area * (pargs->pClusterZs[ip] -  (( (pargs->xslant * (pnc->X - pstatus->w2) + pargs->yslant * (pnc->Y - pstatus->h2)) >> SLOPE_SHIFT)));			
			//pstatus->ch.DeltaZ = pargs->pClusterZs[ip];			
			//pstatus->ch.BottomZ = pargs->pClusterZs[ip];
		}	
		if (pnc != 0) break;
		if (pstatus->ch.Clusters >= pargs->minclusters && pstatus->ch.Volume >= pargs->minvol)
		{
			//pstatus->ch.DeltaZ -= pargs->pClusterZs[pstatus->i];						
			pstatus->ch.AvgX = avgx / pstatus->ch.Volume;
			pstatus->ch.AvgY = avgy / pstatus->ch.Volume;
			pstatus->ch.AvgZ = avgz / pstatus->ch.Volume;	
			/*
			if (pstatus->ch.Clusters > 1)
			{
				int dxc = 0;
				int dyc = 0;
				int dzc = 0;
				IntCluster *p1 = pargs->pC + pstatus->i;
				int dza;
				do
				{
					area = abs(p1->Area);
					int ip = p1 - pargs->pC;
					dz = pargs->pClusterZs[ip] - pstatus->ch.AvgZ;	
					dza = dz * area;										
					dxc += (p1->X + pargs->pClusterXs[ip] - pstatus->ch.AvgX) * dza;
					dyc += (p1->Y + pargs->pClusterYs[ip] - pstatus->ch.AvgY) * dza;
					dzc += dz * dza;
					p1 = pargs->pClusterChains[p1 - pargs->pC];
				}
				while (p1);
				pstatus->ch.DeltaX = (((dxc << Z_SCALE_SHIFT) / pstatus->ch.Volume) << SLOPE_SHIFT) / dzc;
				pstatus->ch.DeltaY = (((dyc << Z_SCALE_SHIFT) / pstatus->ch.Volume) << SLOPE_SHIFT) / dzc;				
			}
			else 
			{
				pstatus->ch.DeltaX = 0;
				pstatus->ch.DeltaY = 0;
			}			
			*/
#if 0
			pstatus->ch.AvgX -= pstatus->w2;
			pstatus->ch.AvgY -= pstatus->h2;
			//pstatus->ch.AvgZ -=  ( (pargs->xslant * pstatus->ch.AvgX + pargs->yslant * pstatus->ch.AvgY) >> SLOPE_SHIFT);
			pstatus->ch.AvgX = (pstatus->ch.AvgX << XY_SCALE_SHIFT) * pargs->xtomicron + pargs->stagex;
			pstatus->ch.AvgY = (pstatus->ch.AvgY << XY_SCALE_SHIFT) * pargs->ytomicron + pargs->stagey;
			//pstatus->ch.DeltaX = (pstatus->ch.DeltaX << XY_SCALE_SHIFT) * pargs->xtomicron;
			//pstatus->ch.DeltaY = (pstatus->ch.DeltaY << XY_SCALE_SHIFT) * pargs->ytomicron;	
			pstatus->ch.ViewTag = pargs->viewtag;
			pargs->pChain[ix * pargs->clusterblocksize + pstatus->count] = pstatus->ch;
#else
			IntChain *psC = pargs->pChain + ix * pargs->clusterblocksize + pstatus->count;
			psC->Clusters = pstatus->ch.Clusters;
			psC->Volume = pstatus->ch.Volume;
			psC->AvgX = ((pstatus->ch.AvgX - pstatus->w2) << XY_SCALE_SHIFT) * pargs->xtomicron + pargs->stagex;
			psC->AvgY = ((pstatus->ch.AvgY - pstatus->h2) << XY_SCALE_SHIFT) * pargs->ytomicron + pargs->stagey;
			psC->AvgZ = pstatus->ch.AvgZ;			
			psC->ViewTag = pargs->viewtag;
			psC->Reserved = 0;
#endif
			pstatus->count++;
			if (pstatus->count >= pargs->clusterblocksize)
			{
				pargs->pChainCounts[ix] = pstatus->count;
				_IKGPU_END(pstatus);
			}
		}
	}
	pargs->pChainCounts[ix] = pstatus->count;
	_IKGPU_END(pstatus);
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

__global__ void maphashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadlock = idx + 1;
	int i, ic, ix, iy;
	Cell *qCell = 0;
	int nc = pChV->Count;
	int chainblocksize = nc / divider + 1;
	IntChain *pC = pChV->Chains;
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;
	int minx = pChMapWnd->MinX;
	int miny = pChMapWnd->MinY;
	int cellsize = pChMapWnd->CellSize;
	int maxcellcontent = pChMapWnd->MaxCellContent;
	Cell *pCell = pChMapWnd->pCells;
	IntChain **pCellContents = pChMapWnd->pChains;
	for (i = 0; i < chainblocksize; i++)
	{
		ic = idx * chainblocksize + i;
		if (ic >= nc) return;
		ix = (pC[ic].AvgX - minx) / cellsize;
		if (ix < 0 || ix >= nx) continue;
		iy = (pC[ic].AvgY - miny) / cellsize;	
		if (iy < 0 || iy >= ny) continue;
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

__global__ void clearhashchain_kernel(ChainView *pChV, ChainMapWindow *pChMapWnd, int divider)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;	
	int i, ic, ix, iy;
	int nc = pChV->Count;
	int chainblocksize = nc / divider + 1;
	IntChain *pC = pChV->Chains;
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;
	int cellsize = pChMapWnd->CellSize;
	int maxcellcontent = pChMapWnd->MaxCellContent;
	Cell *pCell = pChMapWnd->pCells;
	IntChain **pCellContents = pChMapWnd->pChains;
	for (i = 0; i < chainblocksize; i++)
	{
		ic = idx * chainblocksize + i;
		if (ic >= nc) return;
		ix = (pC[ic].AvgX - pChMapWnd->MinX) / cellsize;
		if (ix < 0 || ix >= nx) continue;
		iy = (pC[ic].AvgY - pChMapWnd->MinY) / cellsize;	
		if (iy < 0 || iy >= ny) continue;
		pCell[iy * nx + ix].Count = 0;
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

__global__ void trymapchaindxydz_Ikernel(trymapchain_kernel_args *pargs, trymapchain_kernel_status *pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	IntChain *pC = pargs->pChains + pargs->chainblocksize * idx;
	short maxcellcontent = pargs->pChMapWnd->MaxCellContent;
	int nx = pargs->pChMapWnd->NXCells;
	int ny = pargs->pChMapWnd->NYCells;
	int deltasX = pargs->deltasX;
	int deltasY = pargs->deltasY;
	int deltasZ = pargs->deltasZ;
	int *pMapCounts = pargs->pMapCounts;
	int *pDeltas = pargs->pDeltas;
	int MinX = pargs->pChMapWnd->MinX;
	int MinY = pargs->pChMapWnd->MinY;
	int cellsize = pargs->pChMapWnd->CellSize;
	int minchainsize = pargs->minchainsize;
	int xytol = pargs->xytol;
	int ztol = pargs->ztol;
	_IKGPU_RESUMEFROM(1, pstatus);	
	short dix, diy, diz;
	for (pstatus->ic = pargs->pChainCounts[idx] - 1; pstatus->ic >= 0; pstatus->ic -= pargs->sampledivider)
	{		
		_IKGPU_INTERRUPT(1, pstatus, pargs);				
		//if (pC[pstatus->ic].Volume < minchainsize) continue;
		if (pC[pstatus->ic].Volume < pargs->minchainsize)
		{
			pstatus->ic = min(0, pstatus->ic - pargs->sampledivider);
			if (pC[pstatus->ic].Volume < pargs->minchainsize)
			{
				pstatus->ic = min(0, pstatus->ic - pargs->sampledivider);
				if (pstatus->ic >= 0 && pC[pstatus->ic].Volume < pargs->minchainsize)
				{
					pstatus->ic = min(0, pstatus->ic - pargs->sampledivider);
					if (pstatus->ic >= 0 && pC[pstatus->ic].Volume < pargs->minchainsize)
						continue;
				}
			}
		}

		IntChain *pc = pC + pstatus->ic;		
		for (diz = deltasZ - 1; diz >= 0; diz--)								
			for (dix = deltasX - 1; dix >= 0; dix--)
			{	
				int ideltax = pc->AvgX + pDeltas[dix];// + pDeltas[diz + deltasX + deltasY + deltasZ];
				int icellx = (ideltax - MinX) / cellsize;
				if (icellx < 0 || icellx >= nx) continue;
				for (diy = deltasY - 1; diy >= 0; diy--)
				{
					int ideltay = pc->AvgY + pDeltas[diy + deltasX];// + pDeltas[diz + deltasX + deltasY + 2 * deltasZ];
					int icelly = (ideltay - MinY) / cellsize;
					if (icelly < 0 || icelly >= ny) continue;		
					short icellcls;
					bool increment = false;
					Cell *pThisCell = pargs->pChMapWnd->pCells + icelly * nx + icellx;
					IntChain **ppcls = pargs->pChMapWnd->pChains + (icelly * nx + icellx) * maxcellcontent;
					for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
						if (ppcls[icellcls]->Volume >= minchainsize)								
						{
							if (abs(pc->AvgZ + pDeltas[diz + deltasX + deltasY] - ppcls[icellcls]->AvgZ) < ztol)
								if (max(abs(ideltax - ppcls[icellcls]->AvgX), abs(ideltay - ppcls[icellcls]->AvgY)) < xytol)
								{
									increment = true;
									break;
								}
						}
					if (increment)
					{									
						atomicMax(pargs->pBest, ((atomicAdd(pargs->pMapCounts + (diz * deltasY + diy) * deltasX + dix, 1) + 1) << 16) + (diz * deltasY + diy) * deltasX + dix);
					}
				}
			}		
	}	
	_IKGPU_END(pstatus);
}

__global__ void trymapchain_Ikernel(trymapchain_kernel_args *pargs, trymapchain_kernel_status *pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	IntChain *pC = pargs->pChains + pargs->chainblocksize * idx;
	int maxcellcontent = pargs->pChMapWnd->MaxCellContent;
	int nx = pargs->pChMapWnd->NXCells;
	int ny = pargs->pChMapWnd->NYCells;
	int deltasX = pargs->deltasX;
	int deltasY = pargs->deltasY;
	int deltasZ = pargs->deltasZ;
	int *pDeltas = pargs->pDeltas;
	int *pMapCounts = pargs->pMapCounts;
	int xytol = pargs->xytol;
	int ztol = pargs->ztol;
	int MinX = pargs->pChMapWnd->MinX;
	int MinY = pargs->pChMapWnd->MinY;
	int cellsize = pargs->pChMapWnd->CellSize;
	_IKGPU_RESUMEFROM(1, pstatus);	
	short dix, diy, diz;
	for (pstatus->ic = pargs->pChainCounts[idx] - 1; pstatus->ic >= 0; pstatus->ic -= pargs->sampledivider)
	{		
		_IKGPU_INTERRUPT(1, pstatus, pargs);		
		//if (pC[pstatus->ic].Volume < pargs->minchainsize) continue;
		if (pC[pstatus->ic].Volume < pargs->minchainsize)
		{
			pstatus->ic = min(0, pstatus->ic - pargs->sampledivider);
			if (pC[pstatus->ic].Volume < pargs->minchainsize)
			{
				pstatus->ic = min(0, pstatus->ic - pargs->sampledivider);
				if (pstatus->ic >= 0 && pC[pstatus->ic].Volume < pargs->minchainsize)
				{
					pstatus->ic = min(0, pstatus->ic - pargs->sampledivider);
					if (pstatus->ic >= 0 && pC[pstatus->ic].Volume < pargs->minchainsize)
						continue;
				}
			}
		}


		IntChain *pc = pC + pstatus->ic;
		int ibasex = pc->AvgX;
		int ibasey = pc->AvgY;
		for (dix = deltasX - 1; dix >= 0; dix--)
		{
			int ideltax = ibasex + pDeltas[dix];
			int icellx = (ideltax - MinX) / cellsize;
			if (icellx < 0 || icellx >= nx) continue;
			for (diy = deltasY - 1; diy >= 0; diy--)
			{
				int ideltay = ibasey + pDeltas[diy + pargs->deltasX];
				int icelly = (ideltay - MinY) / cellsize;		
				if (icelly < 0 || icelly >= ny) continue;						
				short icellcls;				
				Cell *pThisCell = pargs->pChMapWnd->pCells + icelly * nx + icellx;
				IntChain **ppcls = pargs->pChMapWnd->pChains + (icelly * nx + icellx) * maxcellcontent;
				for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
				if (ppcls[icellcls]->Volume >= pargs->minchainsize && max(abs(ideltax - ppcls[icellcls]->AvgX), abs(ideltay - ppcls[icellcls]->AvgY)) < xytol)
				{
					for (diz = deltasZ - 1; diz >= 0; diz--)
						if  (abs(pc->AvgZ + pargs->pDeltas[diz + deltasX + deltasY] - ppcls[icellcls]->AvgZ) < ztol)
						{
							atomicMax(pargs->pBest, ((atomicAdd(pMapCounts + (diz * deltasY + diy) * deltasX + dix, 1) + 1) << 16) + (diz * deltasY + diy) * deltasX + dix);
						}
				}
			}
		}
	}
	_IKGPU_END(pstatus);
}

#define DELTA_NORM_SHIFT 10

__global__ void refinemapchain_Ikernel(refinemapchain_kernel_args *pargs, refinemapchain_kernel_status *pstatus)
{
	_IKGPU_PROLOG(pargs, pstatus);
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	_IKGPU_RESUMEFROM(1, pstatus);		
	pstatus->pC = pargs->pChains + pargs->chainblocksize * idx;	
	pstatus->nx = pargs->pChMapWnd->NXCells;
	pstatus->ny = pargs->pChMapWnd->NYCells;
	pstatus->xyrefinedeltas = pargs->xytol / pargs->xyrefinebin;
	pstatus->zrefinedeltas = pargs->ztol / pargs->zrefinebin;
	pstatus->xyrefinespan = 2 * pstatus->xyrefinedeltas + 1;
	pstatus->zrefinespan = 2 * pstatus->zrefinedeltas + 1;
	pstatus->xyzrefinespan = pstatus->xyrefinespan * pstatus->xyrefinespan * pstatus->zrefinespan;
	pstatus->dix = pargs->pDeltas[pstatus->xyrefinedeltas];
	pstatus->diy = pargs->pDeltas[pstatus->xyrefinespan + pstatus->xyrefinedeltas];
	pstatus->diz = pargs->pDeltas[2 * pstatus->xyrefinespan + pstatus->zrefinedeltas];
	/*
	for (ic = 0; ic < pChainCounts[idx]; ic++)
	{
		pC[ic].AvgX += dix;
		pC[ic].AvgY += diy;
		pC[ic].AvgZ += diz;
	}
	*/
	for (pstatus->ic = 0; pstatus->ic < pargs->pChainCounts[idx]; pstatus->ic++)
	{
		_IKGPU_INTERRUPT(1, pstatus, pargs);
		int ibasex = pstatus->pC[pstatus->ic].AvgX + pstatus->dix;
		int ibasey = pstatus->pC[pstatus->ic].AvgY + pstatus->diy;	
		int ibasez = pstatus->pC[pstatus->ic].AvgZ + pstatus->diz;	
		int bestnormdelta = 1 << DELTA_NORM_SHIFT;
		IntChain *pBest = 0;
		int icellx = (ibasex - pargs->pChMapWnd->MinX) / pargs->pChMapWnd->CellSize;
		int icelly = (ibasey - pargs->pChMapWnd->MinY) / pargs->pChMapWnd->CellSize;	
#pragma unroll 3
		for (short iix = -1; iix <= 1; iix++)
		{
			int	icellix = icellx + iix;
			if (icellix < 0 || icellix >= pstatus->nx) continue;
#pragma unroll 3
			for (short iiy = -1; iiy <= 1; iiy++)						
			{
				int icelliy = icelly + iiy;
				if (icelliy < 0 || icelliy >= pstatus->ny) continue;
				int icellcls;
				Cell *pThisCell = pargs->pChMapWnd->pCells + icelliy * pstatus->nx + icellix;
				IntChain **ppcls = pargs->pChMapWnd->pChains + (icelliy * pstatus->nx + icellix) * pargs->pChMapWnd->MaxCellContent;
				for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
					if (abs(ibasex - ppcls[icellcls]->AvgX) < pargs->xytol && abs(ibasey - ppcls[icellcls]->AvgY) < pargs->xytol && abs(ibasez - ppcls[icellcls]->AvgZ) < pargs->ztol)
					{
						int normdelta = (abs(ibasex - ppcls[icellcls]->AvgX) << DELTA_NORM_SHIFT) / pargs->xytol;
						int a = (abs(ibasey - ppcls[icellcls]->AvgY) << DELTA_NORM_SHIFT) / pargs->xytol;
						if (normdelta < a) normdelta = a;
						a = (abs(pstatus->pC[pstatus->ic].AvgZ + pstatus->diz - ppcls[icellcls]->AvgZ) << DELTA_NORM_SHIFT) / pargs->ztol;
						if (normdelta < a) normdelta = a;
						if (normdelta < bestnormdelta)
						{
							pBest = ppcls[icellcls];
							bestnormdelta = normdelta;
						}
					}
			}
		}
		if (pBest)
		{
			int ix = (pBest->AvgX - ibasex) / pargs->xyrefinebin + pstatus->xyrefinedeltas;
			int iy = (pBest->AvgY - ibasey) / pargs->xyrefinebin + pstatus->xyrefinedeltas;
			int iz = (pBest->AvgZ - ibasez) / pargs->zrefinebin + pstatus->zrefinedeltas;
			if (ix >= 0 && ix < pstatus->xyrefinespan && iy >= 0 && iy < pstatus->xyrefinespan && iz >= 0 && iz < pstatus->zrefinespan)
			{
				pargs->pMapCounts[((idx * pstatus->zrefinespan + iz) * pstatus->xyrefinespan + iy) * pstatus->xyrefinespan + ix]++;
			}
		}
	}
	_IKGPU_END(pstatus);
}


__global__ void finalmapchain_kernel(IntChain *pChains, int *pChainCounts, int nc, int chainblocksize, ChainMapWindow *pChMapWnd, int *pDeltas, int deltasXY, int deltasZ, int xytol, int ztol, int *pD)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int ic;
	int best = *pD & 0xffff;
	int dix = pDeltas[best % deltasXY];
	int diy = pDeltas[deltasXY + ((best % (deltasXY * deltasXY)) / deltasXY)];
	int diz = pDeltas[2 * deltasXY + (best / (deltasXY * deltasXY))];
	short iix, iiy;
	IntChain *pC = pChains + chainblocksize * idx;	
	int nx = pChMapWnd->NXCells;
	int ny = pChMapWnd->NYCells;	
/*	for (ic = 0; ic < pChainCounts[idx]; ic++)
	{
		pC[ic].AvgX += dix;
		pC[ic].AvgY += diy;
		pC[ic].AvgZ += diz;
	}*/
	for (ic = 0; ic < pChainCounts[idx]; ic++)
	{
		int ibasex = pC[ic].AvgX + dix;
		int ibasey = pC[ic].AvgY + diy;		
		int bestnormdelta = 1 << DELTA_NORM_SHIFT;
		IntChain *pBest = 0;
		int icellx = (ibasex - pChMapWnd->MinX) / pChMapWnd->CellSize;
		int icelly = (ibasey - pChMapWnd->MinY) / pChMapWnd->CellSize;		
		for (iix = -1; iix <= 1; iix++)
		{
			int	icellix = icellx + iix;
			if (icellix < 0 || icellix >= nx) continue;
			for (iiy = -1; iiy <= 1; iiy++)						
			{
				int icelliy = icelly + iiy;
				if (icelliy < 0 || icelliy >= ny) continue;
				int icellcls;
				Cell *pThisCell = pChMapWnd->pCells + icelliy * nx + icellix;
				IntChain **ppcls = pChMapWnd->pChains + (icelliy * nx + icellix) * pChMapWnd->MaxCellContent;
				for (icellcls = pThisCell->Count - 1; icellcls >= 0; icellcls--)
					if (abs(ibasex - ppcls[icellcls]->AvgX) < xytol && abs(ibasey - ppcls[icellcls]->AvgY) < xytol && abs(pC[ic].AvgZ - ppcls[icellcls]->AvgZ) < ztol)
					{
						int normdelta = (abs(ibasex - ppcls[icellcls]->AvgX) << DELTA_NORM_SHIFT) / xytol;
						int a = (abs(ibasey - ppcls[icellcls]->AvgY) << DELTA_NORM_SHIFT) / xytol;
						if (normdelta < a) normdelta = a;
						a = (abs(pC[ic].AvgZ - ppcls[icellcls]->AvgZ) << DELTA_NORM_SHIFT) / ztol;
						if (normdelta < a) normdelta = a;
						if (normdelta < bestnormdelta)
						{
							pBest = ppcls[icellcls];
							bestnormdelta = normdelta;
						}
					}
			}
		}
		if (pBest)
		{
/*			if (pBest->Volume < pC[ic].Volume) 			
				*pBest = pC[ic];*/
			pC[ic] = pC[--pChainCounts[idx]];
			ic--;
		}
	}
}

__global__ void negshift_viewchains_kernel(ChainView *pview, int *pDeltas, int deltasXY, int deltasZ, int *pD)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int inc = gridDim.x * blockDim.x;
	int best = *pD & 0xffff;
	int dix = pDeltas[best % deltasXY];
	int diy = pDeltas[deltasXY + ((best % (deltasXY * deltasXY)) / deltasXY)];
	int diz = pDeltas[2 * deltasXY + (best / (deltasXY * deltasXY))];
	IntChain *pC = (IntChain *)(void *)((char *)(void *)pview + sizeof(ChainView));
	while (idx < pview->Count)
	{
		pC[idx].AvgX -= dix;
		pC[idx].AvgY -= diy;
		pC[idx].AvgZ -= diz;
		idx += inc;
	}
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

__global__ void safesumcounts_kernel(int *pMapCounts, int deltas2, int repeat)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int ir;	
	for (ir = 1; ir < repeat; ir++)
		pMapCounts[id] += pMapCounts[ir * deltas2 + id];
}

__global__ void safefindbest_kernel(int *pMapCounts, int deltas2, int *pBest)
{
	*pBest = 0;
	int i;
	for (i = 1; i < deltas2; i++)
		if (pMapCounts[i] > pMapCounts[*pBest])
			*pBest = i;
	*pMapCounts = pMapCounts[*pBest];
}

__global__ void preparefinalchainviewcorr_kernel(ChainMapHeader *pmaph, int minmap)
{
	int vmaps = 0;
	int dx = 0;
	int dy = 0;
	int dz = 0;
	int basex = 0;
	int basey = 0;
	int rot = 0;
	int dil = 0;
	int i;
	ChainView *pV = (ChainView *)(void *)((char *)(void *)pmaph + sizeof(ChainMapHeader));
	if (pV)
	{
		*(int *)(void *)(&pV->Reserved) = 0;
		pV->DeltaX = pV->DeltaY = pV->DeltaZ = 0;
	}	
	for (i = 0; i < pmaph->Views; i++)
	{
		if (*(int *)(void *)(&pV->Reserved) >= (minmap << 16))
		{
			vmaps++;			
			dx += pV->DeltaX;
			dy += pV->DeltaY;
			dz += pV->DeltaZ;
			basex = pV->PositionX;
			basey = pV->PositionY;
		}
		pV = (ChainView *)(void *)((char *)(void *)pV + sizeof(ChainView) + pV->Count * sizeof(IntChain));
	}	
	((int *)(void *)(&pmaph->Reserved[0]))[0] = 0;
	pV = (ChainView *)(void *)((char *)(void *)pmaph + sizeof(ChainMapHeader));

	((int *)(void *)(&pmaph->Reserved[0]))[7] = dx;
	((int *)(void *)(&pmaph->Reserved[0]))[8] = dy;
	((int *)(void *)(&pmaph->Reserved[0]))[9] = dz;
	((int *)(void *)(&pmaph->Reserved[0]))[10] = vmaps;
	if (vmaps > 0)
	{
		basex = ((basex - pV->PositionX) * vmaps) / (pmaph->Views - 1);
		basey = ((basey - pV->PositionY) * vmaps) / (pmaph->Views - 1);
		dil = -((dx * (long long)basex + dy * (long long)basey) << FRACT_RESCALE_SHIFT) / ((long long)basex * (long long)basex + (long long)basey * (long long)basey);
		rot = -((dy * (long long)basex - dx * (long long)basey) << FRACT_RESCALE_SHIFT) / ((long long)basex * (long long)basex + (long long)basey * (long long)basey);
		dx /= vmaps;
		((int *)(void *)(&pmaph->Reserved[0]))[1] = dx;
		dy /= vmaps;
		((int *)(void *)(&pmaph->Reserved[0]))[2] = dy;
		dz /= vmaps;
		((int *)(void *)(&pmaph->Reserved[0]))[3] = dz;
	}
	else
	{
		basex = basey = 0;
		rot = dil = 0;
		((int *)(void *)(&pmaph->Reserved[0]))[1] = dx = 0;
		((int *)(void *)(&pmaph->Reserved[0]))[2] = dy = 0;
		((int *)(void *)(&pmaph->Reserved[0]))[3] = dz = 0;
	}

	((int *)(void *)(&pmaph->Reserved[0]))[4] = dil;
	((int *)(void *)(&pmaph->Reserved[0]))[5] = rot;

	int cdx = 0;
	int cdy = 0;
	int cdz = 0;
	for (i = 0; i < pmaph->Views; i++)
	{
		if (*(int *)(void *)(&pV->Reserved) >= (minmap << 16))
		{
			cdx += (pV->DeltaX - dx);
			cdy += (pV->DeltaY - dy);
			//cdz += (pV->DeltaZ - dz);			
		}
		pV->DeltaX = cdx;
		pV->DeltaY = cdy;
		pV->DeltaZ = cdz;
		pV = (ChainView *)(void *)((char *)(void *)pV + sizeof(ChainView) + pV->Count * sizeof(IntChain));
	}
}

__global__ void applyfinalchainviewcorr_kernel(ChainMapHeader *pmaph)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int view = blockIdx.y;
	int i;
	ChainView *pV = (ChainView *)(void *)((char *)(void *)pmaph + sizeof(ChainMapHeader));
	for (i = 1; i < pmaph->Views; i++)
		pV = (ChainView *)(void *)((char *)(void *)pV + sizeof(ChainView) + pV->Count * sizeof(IntChain));
	__syncthreads();
	int total = pV->Count;
	int blocksize = total / (gridDim.x * blockDim.x) + 1;
	i = id * blocksize;
	int iend = min(total, i + blocksize);
	int dx = pV->DeltaX;
	int dy = pV->DeltaY;
	int dz = pV->DeltaZ;
	int px = pV->PositionX;
	int py = pV->PositionY;
	long long dil = ((int *)(void *)(&pmaph->Reserved[0]))[4];
	long long rot = ((int *)(void *)(&pmaph->Reserved[0]))[5];
	while (i < iend)
	{
		int x = (pV->Chains[i].AvgX - px);
		int y = (pV->Chains[i].AvgY - py);
		pV->Chains[i].AvgX = px + dx + x + (int)((x * dil) >> FRACT_RESCALE_SHIFT) + (int)((y * rot) >> FRACT_RESCALE_SHIFT);
		pV->Chains[i].AvgY = py + dy + y - (int)((x * rot) >> FRACT_RESCALE_SHIFT) + (int)((y * dil) >> FRACT_RESCALE_SHIFT);
		pV->Chains[i].AvgZ -= dz;
		i++;
	}
}

} }