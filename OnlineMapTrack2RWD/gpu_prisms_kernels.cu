#include "map.h"
#include "Tracker.h"
#include "gpu_track_kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace SySal;
using namespace SySal::GPU;

namespace SySal { namespace GPU {

#define _MIN_CHAIN_VOLUME_ 3

//#define DECODE_CHAIN_INDEX(ppviewentrypoints, index) ((IntChain *)(void *)((char *)(void *)ppviewentrypoints + index))
#define DECODE_CHAIN_INDEX(ppviewentrypoints, index) ((IntChain *)(void *)((char *)(void *)ppviewentrypoints[index & 1] + (index & 0xfffffffe)))

//#define ENCODE_CHAIN_INDEX(pview, ptr, view) ((int)((char *)(void *)ptr - (char *)(void *)pview))
#define ENCODE_CHAIN_INDEX(pview, ptr, view) ((int)((char *)(void *)ptr - (char *)(void *)pview) + view)

__global__ void explore_skewchainmap_kernel(ChainView *pView1, ChainView *pView2, int width, int height, InternalInfo *pI, ChainView **ppViewEntryPoints, TrackMapHeader *pTH)
{
	/* Gets the first view: we can't call FirstView because that is host code */
	
	ChainView *pV = pView2;	
	ppViewEntryPoints[1] = pView2;
	if (pV && pV->Count > 0)
	{
		pI->H.MinX = pI->H.MaxX = (pV->PositionX + pV->DeltaX);
		pI->H.MinY = pI->H.MaxY = (pV->PositionY + pV->DeltaY);
		pI->H.MinZ = pI->H.MaxZ = (pV->PositionZ + pV->DeltaZ);	
		pI->H.DEBUG1 = pV->Count;	
	}
	else if (pView1 != 0 && pView1->Count > 0)
	{
		pI->H.MinX = pI->H.MaxX = (pView1->PositionX + pView1->DeltaX);
		pI->H.MinY = pI->H.MaxY = (pView1->PositionY + pView1->DeltaY);
		pI->H.MinZ = pI->H.MaxZ = (pView1->PositionZ + pView1->DeltaZ);	
		pI->H.DEBUG1 = pView1->Count;	
	}
	else
	{
		pI->H.MinX = pI->H.MinY = pI->H.MinZ = 0;
		pI->H.MaxX = pI->H.MaxY = pI->H.MaxZ = 1;
		pI->H.DEBUG1 = 0;
	}
	
	pV = pView1;	
	ppViewEntryPoints[0] = pView1;
	if (pV && pV->Count > 0)
	{
		if ((pV->PositionX + pV->DeltaX) < pI->H.MinX) pI->H.MinX = (pV->PositionX + pV->DeltaX);
		else if ((pV->PositionX + pV->DeltaX) > pI->H.MaxX) pI->H.MaxX = (pV->PositionX + pV->DeltaX);
		if ((pV->PositionY + pV->DeltaY) < pI->H.MinY) pI->H.MinY = (pV->PositionY + pV->DeltaY);
		else if ((pV->PositionY + pV->DeltaY) > pI->H.MaxY) pI->H.MaxY = (pV->PositionY + pV->DeltaY);
		if ((pV->PositionZ + pV->DeltaZ) < pI->H.MinZ) pI->H.MinZ = (pV->PositionZ + pV->DeltaZ);
		else if ((pV->PositionZ + pV->DeltaZ) > pI->H.MaxZ) pI->H.MaxZ = (pV->PositionZ + pV->DeltaZ);
		pI->H.DEBUG1 += pV->Count;
	}

	int xexcess = ((pI->H.MaxX - pI->H.MinX) >= width) ? (width / 2) : 0;
	int yexcess = ((pI->H.MaxY - pI->H.MinY) >= height) ? (height / 2) : 0;
	pI->H.MinX -= (width / 2 + xexcess + 2 * pI->C.XYHashTableBinSize);
	pI->H.MaxX += (width / 2 + xexcess + 2 * pI->C.XYHashTableBinSize);
	pI->H.MinY -= (height / 2 + yexcess + 2 * pI->C.XYHashTableBinSize);
	pI->H.MaxY += (height / 2 + yexcess + 2 * pI->C.XYHashTableBinSize);
	pI->H.XYBinSize = pI->C.XYHashTableBinSize;
	pI->H.ZBinSize = 0;
	pI->H.XBins = (pI->H.MaxX - pI->H.MinX) / pI->C.XYHashTableBinSize + 1; 
	pI->H.YBins = (pI->H.MaxY - pI->H.MinY) / pI->C.XYHashTableBinSize + 1; 
	pI->H.ZBins = 1;
	pI->H.NBins = pI->H.XBins * pI->H.YBins * pI->H.ZBins;
	pI->H.XTBins = (pI->H.MaxX - pI->H.MinX) / pI->C.MergeTrackCell + 1; 
	pI->H.YTBins = (pI->H.MaxY - pI->H.MinY) / pI->C.MergeTrackCell + 1; 
	pI->H.NTBins = pI->H.XTBins * pI->H.YTBins;
	pI->H.TBinCapacity = max(2, pI->C.MaxTracks / pI->H.NTBins);
	pI->H.BinCapacity = pI->C.HashBinCapacity;
	pI->MinDist2 = (pI->C.MinLength >> SQUARE_RESCALE) * (pI->C.MinLength >> SQUARE_RESCALE);
	pI->MaxDist2 = (pI->C.MaxLength >> SQUARE_RESCALE) * (pI->C.MaxLength >> SQUARE_RESCALE);	
	pTH->MinX = pI->H.MinX;
	pTH->MaxX = pI->H.MaxX;
	pTH->MinY = pI->H.MinY;
	pTH->MaxY = pI->H.MaxY;
	pTH->MinZ = pI->H.MinZ;
	pTH->MaxZ = pI->H.MaxZ;
	pTH->XYScale = (1 << XY_SCALE_SHIFT);
	pTH->ZScale = (1 << Z_SCALE_SHIFT);
	pTH->Reserved[0] = pTH->Reserved[1] = pTH->Reserved[2] = pTH->Reserved[3] = 
		pTH->Reserved[4] = pTH->Reserved[5] = pTH->Reserved[6] = pTH->Reserved[7] = 0;
	pTH->TotalGrains = 0;
	pTH->Count = 0;	
}

__global__ void fill_skewhashtable1view_list_kernel(int *pbinfill, ChainView **ppViewEntryPoints, HashTableBounds *pHashTableBounds, int skewx, int skewy, int minchainvolume, int view)
{
	ChainView *pV = ppViewEntryPoints[view];
	int threadstep = pV->Count / (gridDim.x * blockDim.x) + 1;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int ibase = threadstep * id;
	int iend = min(ibase + threadstep, pV->Count);	
	short bincapacity = pHashTableBounds->BinCapacity;
	short xybinsize = pHashTableBounds->XYBinSize;
	int xbins = pHashTableBounds->XBins;
	int ybins = pHashTableBounds->YBins;
	int xmin =  pHashTableBounds->MinX;
	int ymin =  pHashTableBounds->MinY;
	int nbins = xbins * ybins;
	IntChain *pv = pV->Chains;
	int z;
	int offset = view ? 0x10000 : 1;
	while (ibase < iend)
	{
		IntChain *pCh = &pv[ibase++];
		if (pCh->Volume < minchainvolume) continue;		
		z = pCh->AvgZ;
#ifdef BEFORE_KRYSS_20141029
		int dx = (pCh->AvgX - ((skewx * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT)));
		if (dx < 0) dx -= xybinsize;
		int ix = (dx / xybinsize) % xbins;
		if (ix < 0) ix += xbins;
		int dy = (pCh->AvgY - ((skewy * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT)));
		if (dy < 0) dy -= xybinsize;
		int iy = (dy / xybinsize) % ybins;
		if (iy < 0) iy += ybins;
		int idbin = (iy * xbins + ix);		
#else
		int ix = ((pCh->AvgX - xmin - ((skewx * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT))) / xybinsize) % xbins;
		if (ix < 0) ix += xbins;		
		int iy = ((pCh->AvgY - ymin - ((skewy * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT))) / xybinsize) % ybins;
		if (iy < 0) iy += ybins;
		int idbin = (iy * xbins + ix);
		if (idbin < 0 || idbin >= nbins) continue;
#endif		
		//pCh->Reserved = atomicExch(pbinfill + idbin, (int)((char *)(void *)pCh - (char *)(void *)ppViewEntryPoints));
		pCh->Reserved = atomicExch(pbinfill + idbin, ENCODE_CHAIN_INDEX(pV, pCh, view));
		//pCh->Reserved = atomicExch(pbinfill + idbin, ENCODE_CHAIN_INDEX(ppViewEntryPoints, pCh));
		atomicAdd(pbinfill + (idbin + nbins), offset);		
		//atomicMin(pbinfill + (idbin += nbins), pCh->ViewTag);
	}
}

__global__ void mergetracks_prepare(int *pTBinFill, short xbins, short ybins, int *pcount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = id / xbins;
	int ix = id - iy * xbins;
	if (iy >= ybins) 
	{
		pcount[id] = 0;
		return;	
	}
	int count = (pTBinFill[id] * (pTBinFill[id] - 1)) >> 1;
	if (ix > 0)
	{
		count += pTBinFill[id] * pTBinFill[id - 1];
		if (iy > 0) 
		{
			count += pTBinFill[id] * pTBinFill[id - xbins - 1];
			count += pTBinFill[id] * pTBinFill[id - xbins];
			if (ix < xbins - 1)
				count += pTBinFill[id] * pTBinFill[id - xbins + 1];
		}
	}
	else if (iy > 0)
	{
		count += pTBinFill[id] * pTBinFill[id - xbins];
		if (ix < xbins - 1)
			count += pTBinFill[id] * pTBinFill[id - xbins + 1];
	}
	pcount[id] = count;
}

__global__ void mergetracks_split_and_index_kernel(int *paircomputer, int depth, int *pairindices, int totalpairs)
{
	int id = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (id >= totalpairs)
	{
		__syncthreads();
		return;
	}
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
	id <<= 1;
	pairindices[id] = place;
	pairindices[id + 1] = res - 1;
	__syncthreads();
}

__global__ void mergetracks_mapindex_kernel(int *pIndex, int *pTBinFill, short xbins, short ybins, short binsize, int totalpairs)
{
	int id = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (id >= totalpairs) 
	{
		__syncthreads();
		return;
	}
	pIndex += (id << 1);
	int idbin = pIndex[0];
	int idcheck = pIndex[1];
	int idtrack = 0;
	int idtrack2 = 0;
	int binfill = pTBinFill[idbin];
	int idbin2 = idbin;
	if (idcheck < ((binfill * (binfill - 1)) >> 1))
	{
		idbin2 = idbin;
		idtrack = (int)floor(0.5f * (1.0f + sqrt(1.0f + 8.0f * (float)idcheck)));
		idtrack2 = idcheck - ((idtrack * (idtrack - 1)) >> 1);		
	}
	else
	{
		idcheck -= ((binfill * (binfill - 1)) >> 1);
		int iy = idbin / xbins;
		int ix = idbin - iy * xbins;
		if (ix > 0)
		{
			if (idcheck < binfill * pTBinFill[idbin - 1])
				idbin2--;
			else
			{
				idcheck -= binfill * pTBinFill[idbin - 1];
				if (idcheck < binfill * pTBinFill[idbin - 1 - xbins])
					idbin2 -= (xbins + 1);
				else
				{
					idcheck -= binfill * pTBinFill[idbin - 1 - xbins];
					if (idcheck < binfill * pTBinFill[idbin - xbins])
						idbin2 -= xbins;
					else 
					{
						idcheck -= binfill * pTBinFill[idbin - xbins];
						idbin2 -= (xbins - 1);
					}
				}
			}
		}
		else
		{
			if (idcheck < binfill * pTBinFill[idbin - xbins])
				idbin2 -= xbins;
			else 
			{
				idcheck -= binfill * pTBinFill[idbin - xbins];
				idbin2 -= (xbins - 1);
			}
		}
		idtrack2 = idcheck / binfill;
		idtrack = idcheck - idtrack2 * binfill;
	}	
	pIndex[0] = idbin * binsize + idtrack;
	pIndex[1] = idbin2 * binsize + idtrack2;
	__syncthreads();
}

__global__ void mergetracks_kernel(int *pIndex, TempIntTrack *pTBins, int xytol, int ztol, int totalpairs)
{
	int id = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (id >= totalpairs)
	{
		__syncthreads();
		return;
	}
	id <<= 1;
	TempIntTrack *pTk = &pTBins[pIndex[id]];
	TempIntTrack *pOTk = &pTBins[pIndex[id + 1]];	
/*
	if (
		abs(pTk->X1 - pOTk->X1) < xytol &
		abs(pTk->Y1 - pOTk->Y1) < xytol &
		abs(pTk->Z1 - pOTk->Z1) < ztol &
		abs(pTk->X2 - pOTk->X2) < xytol &
		abs(pTk->Y2 - pOTk->Y2) < xytol &
		abs(pTk->Z2 - pOTk->Z2) < ztol
		)
*/
	bool chk = true;
	chk = (abs(pTk->X1 - pOTk->X1) < xytol) & chk;
	chk = (abs(pTk->Y1 - pOTk->Y1) < xytol) & chk;
	chk = (abs(pTk->Z1 - pOTk->Z1) < ztol) & chk;
	chk = (abs(pTk->X2 - pOTk->X2) < xytol) & chk;
	chk = (abs(pTk->Y2 - pOTk->Y2) < xytol) & chk;
	chk = (abs(pTk->Z2 - pOTk->Z2) < ztol) & chk;
	int dq = ((pOTk->Quality - pTk->Quality) << 10) + pOTk->Chains - pTk->Chains;
	__syncthreads();
	if (chk)
	{
		if (dq >= 0) pTk->MapsTo = pOTk - pTk;
		else pOTk->MapsTo = pTk - pOTk;		
	}
	__syncthreads();
}

__global__ void mergetracks_kernel(int *pCountTempTracks, TempIntTrack **ppTempTracks, int *pTBinFill, TempIntTrack *pTBins, TrackMapHeader *pTrackMapHdr, int xytol, int ztol, short xbins, short ybins, short binsize, int *pOffset, int *pTerminate)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id + pOffset[id] >= *pCountTempTracks) return;
	TempIntTrack *pTk = ppTempTracks[id + pOffset[id]];	
	if (pTk->MapsTo)
	{
		pOffset[id] += gridDim.x * blockDim.x;
		return;
	};	
	*pTerminate = 0;
	int xytol_rescale = ((int)xytol) << FRACT_RESCALE;
	int ztol_rescale = ((int)ztol) << FRACT_RESCALE;		
	if (pTk->IC >= 0) goto jump;	
	for (pTk->IIX = -1; pTk->IIX <= 1; pTk->IIX++)
		for (pTk->IIY = -1; pTk->IIY <= 1; pTk->IIY++)			
			for (pTk->IC = pTBinFill[((pTk->IX + pTk->IIX + xbins) % xbins) + ((pTk->IY + pTk->IIY + ybins) % ybins) * xbins] - 1; pTk->IC >= 0; pTk->IC--)
			{										
				TempIntTrack *pOTk = &pTBins[((pTk->IX + pTk->IIX + xbins) % xbins + ((pTk->IY + pTk->IIY + ybins) % ybins) * xbins) * binsize + pTk->IC];	
				if (
					abs(pTk->X1 - pOTk->X1) < xytol &&
					abs(pTk->Y1 - pOTk->Y1) < xytol &&
					abs(pTk->Z1 - pOTk->Z1) < ztol &&
					abs(pTk->X2 - pOTk->X2) < xytol &&
					abs(pTk->Y2 - pOTk->Y2) < xytol &&
					abs(pTk->Z2 - pOTk->Z2) < ztol
					)
				{
					int dq = (pOTk->Quality - pTk->Quality) * 1000 + pOTk->Chains - pTk->Chains;
					if (dq >= 0)
					{
						pTk->MapsTo = pOTk - pTk;
						pOffset[id] += gridDim.x * blockDim.x;	
						return;
					}
					else if (dq < 0)
					{
						pOTk->MapsTo = pTk - pOTk;
						continue;
					}					
				}
				return;				
jump:
			}
	pOffset[id] += gridDim.x * blockDim.x;	
}

__global__ void mergetracks_kernel_OLD(int *pCountTempTracks, TempIntTrack **ppTempTracks, int *pTBinFill, TempIntTrack *pTBins, TrackMapHeader *pTrackMapHdr, int xytol, int ztol, short xbins, short ybins, short binsize, int *pOffset, int *pTerminate)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id + pOffset[id] >= *pCountTempTracks) return;
	TempIntTrack *pTk = ppTempTracks[id + pOffset[id]];	
	if (pTk->MapsTo)
	{
		pOffset[id] += gridDim.x * blockDim.x;
		return;
	};	
	*pTerminate = 0;
	int xytol_rescale = ((int)xytol) << FRACT_RESCALE;
	int ztol_rescale = ((int)ztol) << FRACT_RESCALE;		
	if (pTk->IC >= 0) goto jump;	
	for (pTk->IIX = -1; pTk->IIX <= 1; pTk->IIX++)
		for (pTk->IIY = -1; pTk->IIY <= 1; pTk->IIY++)			
			for (pTk->IC = pTBinFill[((pTk->IX + pTk->IIX + xbins) % xbins) + ((pTk->IY + pTk->IIY + ybins) % ybins) * xbins] - 1; pTk->IC >= 0; pTk->IC--)
			{						
				TempIntTrack *pOTk = &pTBins[((pTk->IX + pTk->IIX + xbins) % xbins + ((pTk->IY + pTk->IIY + ybins) % ybins) * xbins) * binsize + pTk->IC];	
				if (pOTk->Quality > pTk->Quality || (pOTk->Quality == pTk->Quality && pOTk < pTk) /*pOTk->Chains > pTk->Chains || (pOTk->Chains == pTk->Chains && pOTk < pTk)*/)
				{
					int d2 = pOTk->D2;
					int a = pTk->X1 - pOTk->X1;
					int b = pTk->Y1 - pOTk->Y1;
					int c = (pTk->Z1 - pOTk->Z1) << Z_TO_XY_RESCALE_SHIFT;
					int d = ((a >> SQUARE_RESCALE) * (pOTk->DX >> SQUARE_RESCALE) + (b >> SQUARE_RESCALE) * (pOTk->DY >> SQUARE_RESCALE) + (c >> SQUARE_RESCALE) * (pOTk->DZ >> SQUARE_RESCALE)) / d2;
					a = ((a << FRACT_RESCALE) - pOTk->DX * d);
					if (abs(a) > xytol_rescale) continue;
					b = ((b << FRACT_RESCALE) - pOTk->DY * d);
					if (abs(b) > xytol_rescale) continue;
					c = ((c << FRACT_RESCALE) - pOTk->DZ * d);
					if (abs(c) > ztol_rescale) continue;
					a /= xytol;
					b /= xytol;
					c /= ztol;
					if ((a * a + b * b + c * c) > (1 << (2 * FRACT_RESCALE))) continue;
					a = pTk->X2 - pOTk->X1;
					b = pTk->Y2 - pOTk->Y1;
					c = (pTk->Z2 - pOTk->Z1) << Z_TO_XY_RESCALE_SHIFT;
					d = ((a >> SQUARE_RESCALE) * (pOTk->DX >> SQUARE_RESCALE) + (b >> SQUARE_RESCALE) * (pOTk->DY >> SQUARE_RESCALE) + (c >> SQUARE_RESCALE) * (pOTk->DZ >> SQUARE_RESCALE)) / d2;
					a = ((a << FRACT_RESCALE) - pOTk->DX * d);
					if (abs(a) > xytol_rescale) continue;
					b = ((b << FRACT_RESCALE) - pOTk->DY * d);
					if (abs(b) > xytol_rescale) continue;
					c = ((c << FRACT_RESCALE) - pOTk->DZ * d);
					if (abs(c) > ztol_rescale) continue;
					a /= xytol;
					b /= xytol;
					c /= ztol;
					if ((a * a + b * b + c * c) > (1 << (2 * FRACT_RESCALE))) continue;
					pTk->MapsTo = pOTk - pTk;
					pOffset[id] += gridDim.x * blockDim.x;	
					return;
				}
				return;				
jump:
			}
	pOffset[id] += gridDim.x * blockDim.x;	
}

__global__ void filltracks_kernel(int *pCountTempTracks, TempIntTrack **ppTempTracks, TrackMapHeader *pTrackMapHdr)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *pCountTempTracks) return;
	if (ppTempTracks[id]->MapsTo) return;
	int a = atomicAdd(&pTrackMapHdr->Count, 1);	
	pTrackMapHdr->Tracks[a] = *(IntTrack *)ppTempTracks[id];
}

__global__ void recursive_sum_kernel(int *parrayin, int *parrayout, int insize)
{
	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
	int id2 = id << 1;
	/*
	if (id2 >= insize) parrayout[id] = 0;
	else if (id2 + 1 >= insize) parrayout[id] = parrayin[id2];
	else parrayout[id] = parrayin[id2] + parrayin[id2 + 1];
	*/
	if (id2 >= insize) return;
	parrayout[id] = parrayin[id2] + parrayin[id2 + 1];
}

__global__ void compute_pairs1v_kernel(int *pinchains, int *poutpairs, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) return; 
	else 
	{
		int v0 = pinchains[id] & 0xffff;
		int v1 = pinchains[id] >> 16;
		poutpairs[id] = v1 * v0 + ((v1 * (v1 - 1)) >> 1);
	}
}

__global__ void pair_find1v_kernel(int *parrayind, int depth, int *parraycountchains, int *pbinfill, ChainView **ppViewEntryPoints, int *poutchainindices)
{
#define PF1V_NEXTLEVEL levelbase - (size << 1)
#define PF1V_LEFT(x) PF1V_NEXTLEVEL + ((x - levelbase) << 1)
#define PF1V_RIGHT(x) PF1V_NEXTLEVEL + ((x - levelbase) << 1) + 1
#define PF1V_TOTAL(x) parrayind[x]
#define PF1V_MOVE_LEFT_P(x) //empty, no change
#define PF1V_MOVE_RIGHT_P(x) x += halfplacesize
#define PF1V_MOVETONEXTLEVEL (levelbase -= (size <<= 1)), (halfplacesize >>= 1), (depth--)

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int totalsize = *parrayind;
	int scan = 0;
	int levelbase = 0;
	if (id >= totalsize) return;
	int seek = id;
	int size = 1;
	int place = 0;
	int halfplacesize = (1 << depth) >> 2;
	while (depth > 1)
	{
		if (seek < PF1V_TOTAL(PF1V_LEFT(scan)))
		{
			scan = PF1V_LEFT(scan);
			PF1V_MOVE_LEFT_P(place);
		}
		else
		{
			seek -= PF1V_TOTAL(PF1V_LEFT(scan));
			scan = PF1V_RIGHT(scan);
			PF1V_MOVE_RIGHT_P(place);
		}
		PF1V_MOVETONEXTLEVEL;
	}	
	int nchains0 = parraycountchains[place];
	int nchains1 = nchains0 >> 16;
	nchains0 = nchains0 & 0xffff;
	int ihigher, ilower;
	if (seek < nchains0 * nchains1)
	{
		ihigher = seek / nchains1;
		ilower = (seek - ihigher * nchains1);
		ihigher += nchains1;
	}
	else
	{
		seek -= nchains0 * nchains1;
		ihigher = (int)(0.5f * (1.0f + sqrt(1.0f + 8.0f * (float)seek)));
		ilower = seek - ((ihigher * (ihigher - 1)) >> 1);
	}
	ChainView *ppV[2];
	ppV[0] = ppViewEntryPoints[0];
	ppV[1] = ppViewEntryPoints[1];
	int ind = pbinfill[place];
	int i;
	for (i = 0; i < ilower; i++)
		ind = DECODE_CHAIN_INDEX(ppV, ind)->Reserved;	
	poutchainindices[id * 3] = ind;
	for (; i < ihigher; i++)
		ind = DECODE_CHAIN_INDEX(ppV, ind)->Reserved;	
	poutchainindices[id * 3 + 1] = ind;
	poutchainindices[id * 3 + 2] = pbinfill[place];
}

__global__ void find_track_singlepass_kernel(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag, int maxtracks)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= totalpairs) return;
	int p3id = pindices[3 * id];
	int p3id1 = pindices[3 * id + 1];
	if (p3id == 0 || p3id1 == 0) return;	
	ChainView *ppV[2];
	ppV[0] = ppViewEntryPoints[0];
	ppV[1] = ppViewEntryPoints[1];	
	IntChain *pA = DECODE_CHAIN_INDEX(ppV, p3id);
	//IntChain *pA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id]));
	IntChain *pB = DECODE_CHAIN_INDEX(ppV, p3id1);
	//IntChain *pB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id + 1]));
	int	az = pA->AvgZ;
	int dz = ((pB->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT);
	int adz = abs(dz);
	if (adz < pC->MinLength) return;		
	int	ax = pA->AvgX;
	int dx = pB->AvgX - ax;
	int	ay = pA->AvgY;
	int dy = pB->AvgY - ay;
	if (
			abs((dx << SLOPE_SCALE_SHIFT) - slopex * dz) > (slopeaccx * adz) ||
			abs((dy << SLOPE_SCALE_SHIFT) - slopey * dz) > (slopeaccy * adz)
		)		
		return;		

	float fdx = (float)dx;
	float fdy = (float)dy;
	float fdz = (float)dz;
	float fd2 = fdx * fdx + fdy * fdy;
	float fss2 = fd2 / (fdz * fdz);
	int Chains = 0;
	int Volume = 0;
	int Clusters = 0;
	float fxytol2 = (float)pC->XYTolerance;
	fxytol2 *= fxytol2;
	float fztol2 = (float)(pC->ZTolerance << Z_TO_XY_RESCALE_SHIFT);
	fztol2 = fss2 * fztol2 * fztol2;
	//int xytol_rescale = xytol << FRACT_RESCALE;
	//int ztol_rescale = ztol << FRACT_RESCALE;
	//int MinViewTag = min(pA->ViewTag, pB->ViewTag);
	IntChain *ppTrackGrains[_MAX_GRAINS_PER_TRACK_];

	int iis;
	//for (iis = pindices[3 * id + 2]; iis; iis = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis))->Reserved)
	//for (iis = pindices[3 * id + 2]; iis; iis = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints[iis & 1] + (iis & 0xfffffffe)))->Reserved)
	IntChain *pS;
	for (iis = pindices[3 * id + 2]; iis; iis = pS->Reserved)	
	{
		//IntChain *pS = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints[iis & 1] + (iis & 0xfffffffe));
		pS = DECODE_CHAIN_INDEX(ppV, iis);
		int c = (pS->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT;
		float d = c / fdz;
		float a = (pS->AvgX - ax) - d * fdx;
		float b = (pS->AvgY - ay) - d * fdy;
		float fdd = (a * dy - b * dx);
		if (fdd * fdd > fd2 * fxytol2) continue;
		fdd = (a * dx + b * dy);
		if (fdd * fdd > fd2 * (fxytol2 + fztol2)) continue;
		{
			if (Chains >= _MAX_GRAINS_PER_TRACK_) Chains--;
			ppTrackGrains[Chains++] = pS;
			Volume += pS->Volume;
			Clusters += pS->Clusters;
			//MinViewTag = min(MinViewTag, pS->ViewTag);			
		}
	}

	//if (MinViewTag > minviewtag) return;
	if (Volume < pC->MinVolume) return;
	if (Chains < pC->FilterMinChains) return;
	if (Clusters < pC->FilterChain0) return;

	float AAvgX = 0.0;
	float AAvgY = 0.0;	
	float AAvgZ = 0.0;
	for (iis = Chains - 1; iis >= 0; iis--)
	{
		AAvgX += ppTrackGrains[iis]->AvgX;
		AAvgY += ppTrackGrains[iis]->AvgY;
		AAvgZ += ppTrackGrains[iis]->AvgZ;
	}
	AAvgX /= Chains;
	AAvgY /= Chains;
	AAvgZ /= Chains;
	float Sz = 0.0f;
	float Sz2 = 0.0f;
	float Szx = 0.0f;
	float Sx = 0.0f;
	float Szy = 0.0f;
	float Sy = 0.0f;
	float fz1 = az - AAvgZ;
	float fz2 = fz1;
	for (iis = Chains - 1; iis >= 0; iis--)
	{
		float z = ppTrackGrains[iis]->AvgZ - AAvgZ;
/*
		fz1 = __min(z, fz1);
		fz2 = __max(z, fz2);
*/
		if (z < fz1) fz1 = z;
		if (z > fz2) fz2 = z;
		Sz += z;
		Sz2 += z * z;
		float x = ppTrackGrains[iis]->AvgX - AAvgX;
		Sx += x;
		Szx += x * z;
		float y = ppTrackGrains[iis]->AvgY - AAvgY;
		Sy += y;
		Szy += y * z;
	}
	float det = 1.0f / (Sz * Sz - Chains * Sz2);
	float detN = det / Chains;
	float fitBx = (Sz * Sx - Chains * Szx) * det;
	float fitAx = (Szx * Sz - Sx * Sz2) * detN;
	float fitBy = (Sz * Sy - Chains * Szy) * det;
	float fitAy = (Szy * Sz - Sy * Sz2) * detN;

	az = fz1 + AAvgZ;
	int z2 = fz2 + AAvgZ;
	int x2 = AAvgX + fitAx + fitBx * fz2;
	ax = AAvgX + fitAx + fitBx * fz1;
	int y2 = AAvgY + fitAy + fitBy * fz2;
	ay = AAvgY + fitAy + fitBy * fz1;
	dx = x2 - ax;
	dy = y2 - ay;
	dz = (z2 - az) << Z_TO_XY_RESCALE_SHIFT;

	if (dz <= 0) return;

	int b = (dx << SLOPE_SCALE_SHIFT) / dz;
	int s2 = b * b;	
	b = (dy << SLOPE_SCALE_SHIFT) / dz;
	s2 += b * b;
	b = Clusters - pC->FilterChain0;
	if ((b * b) < (cm * s2)) return;

	long long int c = dx;
	long long int ddd2 = c * c;
	c = dy;
	ddd2 += (c * c);
	c = dz;
	ddd2 += (c * c);

	b = (Volume - pC->FilterVolumeLength0);
	if ((b * b) < (vs * ddd2)) return;

	int bsize = pC->MergeTrackCell;
	int ix = (ax / bsize + pH->XTBins) % pH->XTBins; if (ix < 0) ix += pH->XTBins;
	int iy = (ay / bsize + pH->YTBins) % pH->YTBins; if (iy < 0) iy += pH->YTBins;
	int bin = iy * pH->XTBins + ix;
	int a = atomicAdd(&pTBinFill[bin], 1);			
	if (a < pH->TBinCapacity)
	{
		TempIntTrack *pTk = &pTBins[bin * pH->TBinCapacity + a];
		pTk->Chains = Chains;
		pTk->Volume = Volume;
		pTk->X1 = ax;
		pTk->X2 = x2;
		pTk->Y1 = ay;
		pTk->Y2 = y2;
		pTk->Z1 = az;
		pTk->Z2 = z2;
		pTk->Quality = Clusters;
		pTk->DX = dx;
		pTk->DY = dy;
		pTk->DZ = dz;
		pTk->D2 = (ddd2 >> (2 * SQUARE_RESCALE + FRACT_RESCALE));
		pTk->MapsTo = 0;
		pTk->IX = ix;
		pTk->IY = iy;
		pTk->IIX = -1;
		pTk->IIY = -1;
		pTk->IC = -1;
		a = atomicAdd(pTrackCounter, 1);
		if (a >= maxtracks)
		{
			atomicSub(pTrackCounter, 1);
			atomicSub(&pTBinFill[bin], 1);
		}
		else ppTracks[a] = pTk;
	}
	else
	{
		atomicSub(&pTBinFill[bin], 1);
	}
	pindices[3 * id + 2] = -1;
}

__global__ void find_track_singlepass_kernel_withoutfit(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= totalpairs) return;
	int p3id = pindices[3 * id];
	int p3id1 = pindices[3 * id + 1];
	if (p3id == 0 || p3id1 == 0) return;	
	ChainView *ppV[2];
	ppV[0] = ppViewEntryPoints[0];
	ppV[1] = ppViewEntryPoints[1];	
	IntChain *pA = DECODE_CHAIN_INDEX(ppV, p3id);
	//IntChain *pA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id]));
	IntChain *pB = DECODE_CHAIN_INDEX(ppV, p3id1);
	//IntChain *pB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id + 1]));
	int	az = pA->AvgZ;
	int dz = ((pB->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT);
	int adz = abs(dz);
	if (adz < pC->MinLength) return;		
	int	ax = pA->AvgX;
	int dx = pB->AvgX - ax;
	int	ay = pA->AvgY;
	int dy = pB->AvgY - ay;
	if (
			abs((dx << SLOPE_SCALE_SHIFT) - slopex * dz) > (slopeaccx * adz) ||
			abs((dy << SLOPE_SCALE_SHIFT) - slopey * dz) > (slopeaccy * adz)
		)		
		return;		

	float fdx = (float)dx;
	float fdy = (float)dy;
	float fdz = (float)dz;
	float fd2 = fdx * fdx + fdy * fdy;
	float fss2 = fd2 / (fdz * fdz);
	int Chains = 0;
	int Volume = 0;
	int Clusters = 0;
	float fxytol2 = (float)pC->XYTolerance;
	fxytol2 *= fxytol2;
	float fztol2 = (float)(pC->ZTolerance << Z_TO_XY_RESCALE_SHIFT);
	fztol2 = fss2 * fztol2 * fztol2;
	//int xytol_rescale = xytol << FRACT_RESCALE;
	//int ztol_rescale = ztol << FRACT_RESCALE;
	//int MinViewTag = min(pA->ViewTag, pB->ViewTag);
	IntChain *ppTrackGrains[_MAX_GRAINS_PER_TRACK_];

	int iis;
	IntChain *pS;
	for (iis = pindices[3 * id + 2]; iis; iis = pS->Reserved)	
	{
		//IntChain *pS = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints[iis & 1] + (iis & 0xfffffffe));
		pS = DECODE_CHAIN_INDEX(ppV, iis);
		int c = (pS->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT;
		float d = c / fdz;
		float a = (pS->AvgX - ax) - d * fdx;
		float b = (pS->AvgY - ay) - d * fdy;
		float fdd = (a * dy - b * dx);
		if (fdd * fdd > fd2 * fxytol2) continue;
		fdd = (a * dx + b * dy);
		if (fdd * fdd > fd2 * (fxytol2 + fztol2)) continue;
		{
			if (Chains >= _MAX_GRAINS_PER_TRACK_) Chains--;
			ppTrackGrains[Chains++] = pS;
			Volume += pS->Volume;
			Clusters += pS->Clusters;
			//MinViewTag = min(MinViewTag, pS->ViewTag);			
		}
	}

	//if (MinViewTag > minviewtag) return;
	if (Volume < pC->MinVolume) return;
	if (Chains < pC->FilterMinChains) return;
	if (Clusters < pC->FilterChain0) return;

	int x2 = ax;
	int y2 = ay;
	int z2 = az;
	for (iis = Chains - 1; iis >= 0; iis--)
	{
		if (ppTrackGrains[iis]->AvgZ < az)
		{
			ax = ppTrackGrains[iis]->AvgX;
			ay = ppTrackGrains[iis]->AvgY;
			az = ppTrackGrains[iis]->AvgZ;
		}
		if (ppTrackGrains[iis]->AvgZ > z2)
		{
			x2 = ppTrackGrains[iis]->AvgX;
			y2 = ppTrackGrains[iis]->AvgY;
			z2 = ppTrackGrains[iis]->AvgZ;
		}
	}

	dx = x2 - ax;
	dy = y2 - ay;
	dz = (z2 - az) << Z_TO_XY_RESCALE_SHIFT;

	if (dz <= 0) return;

	int b = (dx << SLOPE_SCALE_SHIFT) / dz;
	int s2 = b * b;	
	b = (dy << SLOPE_SCALE_SHIFT) / dz;
	s2 += b * b;
	b = Clusters - pC->FilterChain0;
	if ((b * b) < (cm * s2)) return;

	long long int c = dx;
	long long int ddd2 = c * c;
	c = dy;
	ddd2 += (c * c);
	c = dz;
	ddd2 += (c * c);

	b = (Volume - pC->FilterVolumeLength0);
	if ((b * b) < (vs * ddd2)) return;

	int bsize = pC->MergeTrackCell;
	int ix = (ax / bsize + pH->XTBins) % pH->XTBins; if (ix < 0) ix += pH->XTBins;
	int iy = (ay / bsize + pH->YTBins) % pH->YTBins; if (iy < 0) iy += pH->YTBins;
	int bin = iy * pH->XTBins + ix;
	int a = atomicAdd(&pTBinFill[bin], 1);			
	if (a < pH->TBinCapacity)
	{
		TempIntTrack *pTk = &pTBins[bin * pH->TBinCapacity + a];
		pTk->Chains = Chains;
		pTk->Volume = Volume;
		pTk->X1 = ax;
		pTk->X2 = x2;
		pTk->Y1 = ay;
		pTk->Y2 = y2;
		pTk->Z1 = az;
		pTk->Z2 = z2;
		pTk->Quality = Clusters;
		pTk->DX = dx;
		pTk->DY = dy;
		pTk->DZ = dz;
		pTk->D2 = (ddd2 >> (2 * SQUARE_RESCALE + FRACT_RESCALE));
		pTk->MapsTo = 0;
		pTk->IX = ix;
		pTk->IY = iy;
		pTk->IIX = -1;
		pTk->IIY = -1;
		pTk->IC = -1;
		bin = atomicAdd(pTrackCounter, 1);
		ppTracks[bin] = pTk;
	}
	else
	{
		atomicExch(&pTBinFill[bin], pH->TBinCapacity);
	}
	pindices[3 * id + 2] = -1;
}


__global__ void _debug_track_pattern_(ChainView *pView, int chainspertrack, int slope_step_scaled, int xystep, int steps, int deltaz)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= pView->Count) return;
	int ix = id % steps;
	int _i = id / steps;
	int iy = _i % steps;
	_i = _i / steps;
	if (_i > chainspertrack)
	{
		pView->Chains[id].Clusters = pView->Chains[id].Volume = 1;
		return;
	}
	IntChain *pCh = &pView->Chains[id];
	pCh->AvgX = pView->PositionX + (ix - steps / 2) * xystep + ((_i * (deltaz << Z_TO_XY_RESCALE_SHIFT) * (ix - steps / 2) * slope_step_scaled) >> SLOPE_SHIFT);
	pCh->AvgY = pView->PositionY + (iy - steps / 2) * xystep + ((_i * (deltaz << Z_TO_XY_RESCALE_SHIFT) * (iy - steps / 2) * slope_step_scaled) >> SLOPE_SHIFT);
	pCh->AvgZ = _i * deltaz;
	pCh->Clusters = 2;
	pCh->Volume = 8;
}

}}