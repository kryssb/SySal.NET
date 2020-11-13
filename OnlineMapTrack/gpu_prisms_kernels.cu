#include "map.h"
#include "Tracker.h"
#include "gpu_track_kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace SySal;
using namespace SySal::GPU;

namespace SySal { namespace GPU {

#define _MIN_CHAIN_VOLUME_ 3

__global__ void reset_scheduler_kernel(int *pScheduler, int value, int setblocks)
{
	pScheduler[0] = value;
	int i;
	for (i = 0; i < setblocks; i++)
		pScheduler[i + 1] = i;
}

__global__ void explore_skewchainmap_kernel(ChainView *pView1, ChainView *pView2, int width, int height, InternalInfo *pI, ChainView **ppViewEntryPoints, TrackMapHeader *pTH)
{
	/* Gets the first view: we can't call FirstView because that is host code */
	
	ChainView *pV = pView1;	
	ppViewEntryPoints[0] = pView1;
	pI->H.MinX = pI->H.MaxX = (pV->PositionX + pV->DeltaX);
	pI->H.MinY = pI->H.MaxY = (pV->PositionY + pV->DeltaY);
	pI->H.MinZ = pI->H.MaxZ = (pV->PositionZ + pV->DeltaZ);	
	pI->H.DEBUG1 = pV->Count;
	
	pV = pView2;
	ppViewEntryPoints[1] = pView2;
	if ((pV->PositionX + pV->DeltaX) < pI->H.MinX) pI->H.MinX = (pV->PositionX + pV->DeltaX);
	else if ((pV->PositionX + pV->DeltaX) > pI->H.MaxX) pI->H.MaxX = (pV->PositionX + pV->DeltaX);
	if ((pV->PositionY + pV->DeltaY) < pI->H.MinY) pI->H.MinY = (pV->PositionY + pV->DeltaY);
	else if ((pV->PositionY + pV->DeltaY) > pI->H.MaxY) pI->H.MaxY = (pV->PositionY + pV->DeltaY);
	if ((pV->PositionZ + pV->DeltaZ) < pI->H.MinZ) pI->H.MinZ = (pV->PositionZ + pV->DeltaZ);
	else if ((pV->PositionZ + pV->DeltaZ) > pI->H.MaxZ) pI->H.MaxZ = (pV->PositionZ + pV->DeltaZ);
	pI->H.DEBUG1 += pV->Count;

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

__global__ void fill_skewhashtable_list_kernel(int *pbinfill, ChainView **ppViewEntryPoints, HashTableBounds *pHashTableBounds, int skewx, int skewy, int minchainvolume)
{
	ChainView *pV = ppViewEntryPoints[blockIdx.y];
	int threadstep = pV->Count / (gridDim.x * blockDim.x) + 1;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int ibase = threadstep * id;
	int iend = min(ibase + threadstep, pV->Count);	
	short bincapacity = pHashTableBounds->BinCapacity;
	short xybinsize = pHashTableBounds->XYBinSize;
	int xbins = pHashTableBounds->XBins;
	int ybins = pHashTableBounds->YBins;
	int nbins = xbins * ybins;
	IntChain *pv = pV->Chains;
	int z;
	while (ibase < iend)
	{
		IntChain *pCh = &pv[ibase++];
		//if (pCh->Volume < minchainvolume) continue;		
		z = pCh->AvgZ;
		int ix = ((pCh->AvgX - ((skewx * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT))) / xybinsize) % xbins;
		if (ix < 0) ix += xbins;		
		int iy = ((pCh->AvgY - ((skewy * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT))) / xybinsize) % ybins;
		if (iy < 0) iy += ybins;
		int idbin = (iy * xbins + ix);		
		pCh->Reserved = atomicExch(pbinfill + idbin, (int)((char *)(void *)pCh - (char *)(void *)ppViewEntryPoints));
		atomicAdd(pbinfill + (idbin += nbins), 1);		
		atomicMin(pbinfill + (idbin += nbins), pCh->ViewTag);		
	}
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
	int nbins = xbins * ybins;
	IntChain *pv = pV->Chains;
	int z;
	int offset = view ? 0x10000 : 1;
	while (ibase < iend)
	{
		IntChain *pCh = &pv[ibase++];
		//if (pCh->Volume < minchainvolume) continue;		
		z = pCh->AvgZ;
		int ix = ((pCh->AvgX - ((skewx * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT))) / xybinsize) % xbins;
		if (ix < 0) ix += xbins;		
		int iy = ((pCh->AvgY - ((skewy * z) >> (SLOPE_SCALE_SHIFT - XY_SCALE_SHIFT + Z_SCALE_SHIFT))) / xybinsize) % ybins;
		if (iy < 0) iy += ybins;
		int idbin = (iy * xbins + ix);		
		pCh->Reserved = atomicExch(pbinfill + idbin, (int)((char *)(void *)pCh - (char *)(void *)ppViewEntryPoints));
		atomicAdd(pbinfill + (idbin += nbins), offset);		
		//atomicMin(pbinfill + (idbin += nbins), pCh->ViewTag);
	}
}

__global__ void find_tracks_skewreset_list_kernel(_segmented_findtrack_kernel_status_ *pstat, int *qBinFill, ChainView **ppViewEntryPoints, int xbins, int ybins, short binsize, int *pScheduler)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	pScheduler[id + 1] = id;
	int a = id / xbins;
	int b = id - a * xbins;
	if (a >= ybins)
	{
		pstat[id].Run = false;
		return;
	}
	pstat += id;
	int binA = a * xbins + b;
	pstat->binA = binA;
	pstat->binX = b;
	pstat->binY = a;
	int iA, iB;
	for (iA = qBinFill[binA]; iA; iA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Reserved )
	{
		//if (((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Volume < _MIN_CHAIN_VOLUME_) continue;
		for (iB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Reserved; iB; iB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->Reserved)
		{
			pstat->iA = iA;
			pstat->iB = iB;
			//if (((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->Volume < _MIN_CHAIN_VOLUME_) continue;
			pstat->Run = true;
			pstat->SkipFirst = false;			
			return;
		}
	}
	pstat->Run = false;
}

__global__ void find_tracks_skewslope_list_kernel_OLD(_segmented_findtrack_kernel_status_ *pstat, ChainView **ppViewEntryPoints, short xbins, short ybins, short binsize, int slopex, int slopey, int slopeaccx, int slopeaccy, int minlength, int *pScheduler)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= pScheduler[0]) return;
	id = pScheduler[1 + id];	
	pstat += id;
	if (pstat->Run == false) return;

	pstat->SearchGrains = false;
	int binA = pstat->binA * binsize;
	IntChain *pA = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pstat->iA);
	IntChain *pB = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pstat->iB);
	int az = pA->AvgZ;
	int c = ((pB->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT);		
	int ac = abs(c);
	if (ac < minlength) return;	
	int ax = pA->AvgX;
	int a = pB->AvgX - ax;
	int ay = pA->AvgY;
	int b = pB->AvgY - ay;									
	if (
			abs((a << SLOPE_SCALE_SHIFT) - slopex * c) > (slopeaccx * ac) ||
			abs((b << SLOPE_SCALE_SHIFT) - slopey * c) > (slopeaccy * ac)
			)
			return;
	pstat->dx = a;
	pstat->dy = b;
	pstat->dz = c;
	pstat->d2 = (a >> SQUARE_RESCALE) * (a >> SQUARE_RESCALE) + (b >> SQUARE_RESCALE) * (b >> SQUARE_RESCALE) + (c >> SQUARE_RESCALE) * (c >> SQUARE_RESCALE);
	pstat->Ax = ax;
	pstat->Ay = ay;
	pstat->Az = az;
	pstat->Chains = 0;
	pstat->Volume = 0;
	pstat->Clusters = 0;
	pstat->MinViewTag = min(pA->ViewTag, pB->ViewTag);
	pstat->SearchGrains = true;
}

__global__ void find_tracks_skewslope_list_kernel(_segmented_findtrack_kernel_status_ *pstat, ChainView **ppViewEntryPoints, short xbins, short ybins, short binsize, int slopex, int slopey, int slopeaccx, int slopeaccy, int minlength, int *pScheduler)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= pScheduler[0]) return;
	id = pScheduler[1 + id];	
	pstat += id;
	if (pstat->Run == false) return;

	pstat->SearchGrains = false;
	int binA = pstat->binA * binsize;
	IntChain *pA = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pstat->iA);
	IntChain *pB;
	int iB = pstat->iB;
	int az;
	int c;
	int ac;
	int ax;
	int a;
	int ay;
	int b;
	do
	{
		pstat->iB = iB;
		pB = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB);		
		az = pA->AvgZ;
		c = ((pB->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT);		
		ac = abs(c);
		if (ac >= minlength) 
		{
			ax = pA->AvgX;
			a = pB->AvgX - ax;
			ay = pA->AvgY;
			b = pB->AvgY - ay;									
			if (
				abs((a << SLOPE_SCALE_SHIFT) - slopex * c) <= (slopeaccx * ac) &&
				abs((b << SLOPE_SCALE_SHIFT) - slopey * c) <= (slopeaccy * ac)
				)		
				break;
		}
		iB = pB->Reserved;		
	}
	while (iB != 0);
	if (iB != 0)
	{
		pstat->dx = a;
		pstat->dx_sqresc = (a >> SQUARE_RESCALE);
		pstat->dy = b;
		pstat->dy_sqresc = (b >> SQUARE_RESCALE);
		pstat->dz = c;
		pstat->dz_sqresc = (c >> SQUARE_RESCALE);
		pstat->d2 = pstat->dx_sqresc * pstat->dx_sqresc + pstat->dy_sqresc * pstat->dy_sqresc + pstat->dz_sqresc * pstat->dz_sqresc;
		pstat->d2_fracresc = (pstat->d2 >> FRACT_RESCALE);
		pstat->Ax = ax;
		pstat->Ay = ay;
		pstat->Az = az;
		pstat->Chains = 0;
		pstat->Volume = 0;
		pstat->Clusters = 0;
		pstat->MinViewTag = min(pA->ViewTag, pB->ViewTag);
		pstat->SearchGrains = true;
	}	
}

__global__ void find_tracks_skewincrement_list_kernel_old(_segmented_findtrack_kernel_status_ *pstat, int *qBinFill, ChainView **ppViewEntryPoints, short xbins, short ybins, int binsize, int *pScheduler, int minlength, int *pNewScheduler)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= pScheduler[0]) return;
	id = pScheduler[1 + id];	
	pstat += id;
	if (pstat->Run == false) return;	
	int iB = pstat->iB;
	int iA = pstat->iA;	
	int binA = pstat->binA;
	goto jump;
	for (iA = qBinFill[binA]; iA; iA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Reserved )
	{
		//if (((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Volume < _MIN_CHAIN_VOLUME_) continue;
		for (iB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Reserved; iB; iB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->Reserved)
		{			
			pstat->iA = iA;
			pstat->iB = iB;
			//if (((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->Volume < _MIN_CHAIN_VOLUME_) continue;
			if (abs(((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->AvgZ - ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->AvgZ) < minlength) continue;
			int a = atomicAdd(pNewScheduler, 1);
			pNewScheduler[1 + a] = id;					
			return;
jump:
		}
	}
	pstat->Run = false;
}

__global__ void find_tracks_skewincrement_list_kernel(_segmented_findtrack_kernel_status_ *pstat, int *qBinFill, ChainView **ppViewEntryPoints, short xbins, short ybins, int binsize, int *pScheduler, int minlength, int *pNewScheduler)
{
	int id0 = blockIdx.x * blockDim.x + threadIdx.x;
	if (id0 >= pScheduler[0]) return;
	pNewScheduler[1 + id0] = -1;
	int id = pScheduler[1 + id0];	
	pstat += id;
	if (pstat->Run == false) return;	
	int iB = pstat->iB;
	int iA = pstat->iA;	
	int binA = pstat->binA;
	goto jump;	
	for (iA = qBinFill[binA]; iA; iA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Reserved )
	{
		//if (((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Volume < _MIN_CHAIN_VOLUME_) continue;
		for (iB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->Reserved; iB; iB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->Reserved)
		{			
			//if (((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->Volume < _MIN_CHAIN_VOLUME_) continue;
			if (abs(((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iA))->AvgZ - ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iB))->AvgZ) < minlength) continue;			
			pNewScheduler[1 + id0] = id;
			pstat->iA = iA;
			pstat->iB = iB;
			return;
jump:
		}
	}
	pstat->iA = iA;
	pstat->iB = iB;
	pstat->Run = false;
}

__global__ void compact_scheduler_kernel(int *pNewScheduler, int *pScheduler)
{
	int i, j, v;
	int maxsched = *pScheduler;
	for (i = j = 0; i < maxsched; i++)
		if ((v = pNewScheduler[1 + i]) >= 0)
		{
			pNewScheduler[1 + j] = v;
			j++;
		}
	pNewScheduler[0] = j;	
}

__global__ void find_tracks_skewgrainseek_list_kernel(_segmented_findtrack_kernel_status_ *pstat, ChainView **ppViewEntryPoints, int *qBinFill, short xbins, short ybins, short binsize, int xytol, int ztol,/* short ix, short iy, */int *pScheduler)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= pScheduler[0]) return;
	id = pScheduler[1 + id];	
	pstat += id;
	if (pstat->Run == false || pstat->SearchGrains == false) return;	
	int ib;
	int ax = pstat->Ax;
	int ay = pstat->Ay;
	int az = pstat->Az;
	int xytol_rescale = xytol << FRACT_RESCALE;
	int ztol_rescale = ztol << FRACT_RESCALE;
	int dx = pstat->dx;
	int dx_sqresc = pstat->dx_sqresc;
	int dy = pstat->dy;
	int dy_sqresc = pstat->dy_sqresc;
	int dz = pstat->dz;
	int dz_sqresc = pstat->dz_sqresc;
	int d2_fracresc = pstat->d2_fracresc;
	int f;
	//for (ib = qBinFill[((pstat->binY + iy + ybins) % ybins) * xbins + (pstat->binX + ix + xbins) % xbins]; ib; ib = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ib))->Reserved)
	for (ib = qBinFill[pstat->binY * xbins + pstat->binX]; ib; ib = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ib))->Reserved)
	{
		IntChain *pS = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ib);		
		int a = pS->AvgX - ax;
		int b = pS->AvgY - ay;
		int c = (pS->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT;
		int d = (a >> SQUARE_RESCALE) * dx_sqresc + (b >> SQUARE_RESCALE) * dy_sqresc + (c >> SQUARE_RESCALE) * dz_sqresc;
		//int d = ((long long int)a * (long long int)dx + (long long int)b * (long long int)dy + (long long int)c * (long long int)dz) >> (2 * SQUARE_RESCALE);
		d = (d / pstat->d2_fracresc);
		a = ((a << FRACT_RESCALE) - dx * d);
		if (abs(a) > xytol_rescale) continue;
		b = ((b << FRACT_RESCALE) - dy * d);
		if (abs(b) > xytol_rescale) continue;
		c = ((c << FRACT_RESCALE) - dz * d);
		if (abs(c) > ztol_rescale) continue;
		a /= xytol;
		b /= xytol;
		c /= ztol;
		//a = ((a << FRACT_RESCALE) - dx * d) / xytol;
		//b = ((b << FRACT_RESCALE) - dy * d) / xytol;
		//c = ((c << FRACT_RESCALE) - dz * d) / ztol;
		//if (abs(a) <= (1 << FRACT_RESCALE) && abs(b) <= (1 << FRACT_RESCALE) && abs(c) <= (1 << FRACT_RESCALE)) 
			if ((a * a + b * b + c * c) <= (1 << (2 * FRACT_RESCALE)))
			{
				d = pstat->Chains;
				pstat->ppTrackGrains[d++] = pS;
				pstat->Volume += pS->Volume;
				pstat->Clusters += pS->Clusters;
				pstat->MinViewTag = min(pstat->MinViewTag, pS->ViewTag);
				if (d < _MAX_GRAINS_PER_TRACK_) pstat->Chains = d;
			}
	}
}

__global__ void find_tracks_skewchecktrack_kernel(_segmented_findtrack_kernel_status_ *pstat, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, short xbins, short ybins, int binsize, int minviewtag, int *pScheduler)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= pScheduler[0]) return;
	id = pScheduler[1 + id];	
	pstat += id;
	if (pstat->Run == false) return;	
	bool check = pstat->SearchGrains;
	int C_0 = pC->ClusterVol0;
	int C_M = pC->ClusterVolM;
	if (check && pstat->MinViewTag < minviewtag && pstat->Volume >= pC->MinVolume && pstat->Chains >= 4 && /*KRYSS pstat->Chains*/ pstat->Clusters >= pC->FilterChain0)
	{
		int b = pstat->Chains - 1;		

		int x1 = pstat->ppTrackGrains[b]->AvgX;
		int x2 = x1;
		int y1 = pstat->ppTrackGrains[b]->AvgY;
		int y2 = y1;
		int z1 = pstat->ppTrackGrains[b]->AvgZ;
		int z2 = z1;
		for (--b; b >= 0; b--)
		{
			if (pstat->ppTrackGrains[b]->AvgZ < z1)
			{
				x1 = pstat->ppTrackGrains[b]->AvgX;
				y1 = pstat->ppTrackGrains[b]->AvgY;
				z1 = pstat->ppTrackGrains[b]->AvgZ;
			}
			if (pstat->ppTrackGrains[b]->AvgZ > z2)
			{
				x2 = pstat->ppTrackGrains[b]->AvgX;
				y2 = pstat->ppTrackGrains[b]->AvgY;
				z2 = pstat->ppTrackGrains[b]->AvgZ;
			}
		}
		pstat->dx = x2 - x1;
		pstat->dy = y2 - y1;
		pstat->dz = (z2 - z1) << Z_TO_XY_RESCALE_SHIFT;

		if (pstat->dz <= 0) return;

		b = (pstat->dx << SLOPE_SCALE_SHIFT) / pstat->dz;
		int s2 = b * b;	
		b = (pstat->dy << SLOPE_SCALE_SHIFT) / pstat->dz;
		s2 += b * b;
		//KRYSS
		//b = pstat->Chains - pC->FilterChain0;
		b = pstat->Clusters - pC->FilterChain0;
		check = check && ((b * b) >= (cm * s2));

		long long int c = pstat->dx;
		long long int d2 = c * c;
		c = pstat->dy;
		d2 += (c * c);
		c = pstat->dz;
		d2 += (c * c);

		b = (pstat->Volume - pC->FilterVolumeLength0);
		check = check && ((b * b) >= (vs * d2))
		// 27/5/2013 KRYSS
			&& (pstat->Clusters * C_M >= C_0 - pstat->Volume)
		// END KRYSS
		;


		
		if (check)
		{
			int bsize = pC->MergeTrackCell;
			c = ((x1 / bsize) + xbins) % xbins; if (c < 0) c += xbins;
			b = ((y1 / bsize) + ybins) % ybins; if (b < 0) b += ybins;
			int bin = b * xbins + c;
			int a = atomicAdd(&pTBinFill[bin], 1);			
			if (a < binsize)
			{
				TempIntTrack *pTk = &pTBins[bin * binsize + a];
				pTk->Chains = pstat->Chains;
				pTk->Volume = pstat->Volume;
				pTk->X1 = x1;
				pTk->X2 = x2;
				pTk->Y1 = y1;
				pTk->Y2 = y2;
				pTk->Z1 = z1;
				pTk->Z2 = z2;
				pTk->Quality = pstat->Clusters;
				pTk->DX = pstat->dx;
				pTk->DY = pstat->dy;
				pTk->DZ = pstat->dz;
				pTk->D2 = (d2 >> (2 * SQUARE_RESCALE + FRACT_RESCALE));
				pTk->MapsTo = 0;
				pTk->IX = c;
				pTk->IY = b;
				pTk->IIX = -1;
				pTk->IIY = -1;
				pTk->IC = -1;
				bin = atomicAdd(pTrackCounter, 1);
				ppTracks[bin] = pTk;
			}
			else
			{
				atomicExch(&pTBinFill[bin], binsize);
			}
		}	

	}	
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

__global__ void reset_compactor_kernel(int *parray, int *parraylen, int *psize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *psize) parraylen[id] = 0;
	else parraylen[id] = (parray[id] >= 0) ? 1 : 0;
}

__global__ void recursive_sum_kernel(int *parrayin, int *parrayout, int insize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int id2 = id << 1;
	if (id2 >= insize) parrayout[id] = 0;
	else if (id2 + 1 >= insize) parrayout[id] = parrayin[id2];
	else parrayout[id] = parrayin[id2] + parrayin[id2 + 1];
}

__global__ void compactor_find_kernel(int *parrayind, int depth, int *parrayin, int *parrayout, int *poutlength)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int totalsize = *parrayind;
	if (id == 0) *poutlength = totalsize;
	if (id >= totalsize) return;
	int size = 1;	
	int seek = id;	
	int place = 0;
	int displace = 0;
	while (depth >= 0)
	{
		if (seek < parrayind[displace])
		{
			displace <<= 1;
		}
		else
		{
			seek -= parrayind[displace];
			place += (1 << depth);
			displace = (displace + 1) << 1;
		}
		parrayind -= size;
		size <<= 1;
		depth--;
	}
	parrayout[id] = parrayin[place];	
}

__global__ void compute_pairs1v_kernel(int *pinchains, int *pmintag, int mintag, int *poutpairs, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) poutpairs[id] = 0;
	//MINTAG CHECK MEANINGLESS else if (pmintag[id] >= mintag) poutpairs[id] = 0;
	else 
	{
		int v0 = pinchains[id] & 0xffff;
		int v1 = pinchains[id] >> 16;
		poutpairs[id] = v1 * v0 + ((v0 * (v0 - 1)) >> 1);
	}
}

__global__ void compute_pairs_kernel(int *pinchains, int *pmintag, int mintag, int *poutpairs, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) poutpairs[id] = 0;
	else if (pmintag[id] >= mintag) poutpairs[id] = 0;
	else poutpairs[id] = (pinchains[id] * (pinchains[id] - 1)) >> 1;
}

__global__ void compute_pairs_kernel_ALT2(int *pinchains, int *pinclusters, int minclusters, int *poutpairs, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) poutpairs[id] = 0;
	else if (pinclusters[id] < minclusters) poutpairs[id] = 0;
	else poutpairs[id] = (pinchains[id] * (pinchains[id] - 1)) >> 1;
}

__global__ void compute_pairs_kernel_ALT3(int *pinchains, int *poutpairs, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) poutpairs[id] = 0;
	else poutpairs[id] = (pinchains[id] * (pinchains[id] - 1)) >> 1;
}

__global__ void pair_find_kernel(int *parrayind, int depth, int *parraycountchains, int *pbinfill, ChainView **ppViewEntryPoints, int *poutchainindices)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int totalsize = *parrayind;
	if (id >= totalsize) return;
	int size = 1;	
	int seek = id;	
	int place = 0;
	int displace = 0;
	while (depth >= 0)
	{
		if (seek < parrayind[displace])
		{
			displace <<= 1;
		}
		else
		{
			seek -= parrayind[displace];
			place += (1 << depth);
			displace = (displace + 1) << 1;
		}
		parrayind -= size;
		size <<= 1;
		depth--;
	}
	int nchains = parraycountchains[place];	
	int ifirst;
	for (ifirst = 1; (ifirst * (ifirst - 1) >> 1) < seek; ifirst++);
	int isecond = seek - ((ifirst - 1) * (ifirst - 2) >> 1);
	int ind = pbinfill[place];
	int i;
	for (i = 1; i < isecond; i++)
		ind = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ind))->Reserved;
	poutchainindices[id * 3] = ind;
	for (; i < ifirst; i++)
		ind = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ind))->Reserved;
	poutchainindices[id * 3 + 1] = ind;
	poutchainindices[id * 3 + 2] = pbinfill[place];
}

__global__ void pair_find1v_kernel(int *parrayind, int depth, int *parraycountchains, int *pbinfill, ChainView **ppViewEntryPoints, int *poutchainindices)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int totalsize = *parrayind;
	if (id >= totalsize) return;
	int size = 1;	
	int seek = id;	
	int place = 0;
	int displace = 0;
	while (depth >= 0)
	{
		if (seek < parrayind[displace])
		{
			displace <<= 1;
		}
		else
		{
			seek -= parrayind[displace];
			place += (1 << depth);
			displace = (displace + 1) << 1;
		}
		parrayind -= size;
		size <<= 1;
		depth--;
	}
	int nchains0 = parraycountchains[place];	
	int nchains1 = nchains0 >> 16;
	nchains0 = nchains0 & 0xffff;
	int ifirst, isecond;
	if (seek < nchains0 * nchains1)
	{
		ifirst = seek / nchains1;
		isecond = (seek - ifirst * nchains1);
		ifirst += nchains1;
	}
	else
	{
		seek = seek - nchains0 * nchains1 + 1;
		for (ifirst = 1; (ifirst * (ifirst - 1) >> 1) < seek; ifirst++);
		isecond = seek - ((ifirst - 1) * (ifirst - 2) >> 1) + nchains1 - 1;
		ifirst += nchains1 - 1;		
	}
	/*
	poutchainindices[id * 3] = ifirst;
	poutchainindices[id * 3 + 1] = isecond;
	poutchainindices[id * 3 + 2] = parraycountchains[place];
	return;
	*/
	int ind = pbinfill[place];
	int i;
	for (i = 0; i < isecond; i++)
		ind = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ind))->Reserved;
	poutchainindices[id * 3] = ind;
	for (; i < ifirst; i++)
		ind = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + ind))->Reserved;
	poutchainindices[id * 3 + 1] = ind;
	poutchainindices[id * 3 + 2] = pbinfill[place];
	//poutchainindices[id * 3 + 2] = (ifirst << 16) + isecond;
}

__global__ void find_track_singlepass_kernel_OLD(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= totalpairs) return;
	if (pindices[3 * id] == 0 || pindices[3 * id + 1] == 0) return;
	IntChain *pA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id]));
	IntChain *pB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id + 1]));
	int	az = pA->AvgZ;
	int dz = ((pB->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT);
	int adz = abs(dz);
	if (adz < pC->MinLength) return;		
	int	ax = pA->AvgX;
	int dx = pB->AvgX - ax;
	int	ay = pA->AvgY;
	int	dy = pB->AvgY - ay;									
	if (
			abs((dx << SLOPE_SCALE_SHIFT) - slopex * dz) > (slopeaccx * adz) ||
			abs((dy << SLOPE_SCALE_SHIFT) - slopey * dz) > (slopeaccy * adz)
		)		
		return;		
	int dx_sqresc = (dx >> SQUARE_RESCALE);
	int dy_sqresc = (dy >> SQUARE_RESCALE);
	int dz_sqresc = (dz >> SQUARE_RESCALE);
	int d2 = dx_sqresc * dx_sqresc + dy_sqresc * dy_sqresc + dz_sqresc * dz_sqresc;
	int d2_fracresc = (d2 >> FRACT_RESCALE);
	int Chains = 0;
	int Volume = 0;
	int Clusters = 0;
	int xytol = pC->XYTolerance;
	int ztol = pC->ZTolerance << Z_TO_XY_RESCALE_SHIFT;
	int xytol_rescale = xytol << FRACT_RESCALE;
	int ztol_rescale = ztol << FRACT_RESCALE;
	//int MinViewTag = min(pA->ViewTag, pB->ViewTag);
	IntChain *ppTrackGrains[_MAX_GRAINS_PER_TRACK_];

	int iis;
	for (iis = pindices[3 * id + 2]; iis; iis = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis))->Reserved)
	{
		IntChain *pS = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis);
		int a = pS->AvgX - ax;
		int b = pS->AvgY - ay;
		int c = (pS->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT;
		int d = (a >> SQUARE_RESCALE) * dx_sqresc + (b >> SQUARE_RESCALE) * dy_sqresc + (c >> SQUARE_RESCALE) * dz_sqresc;		
		d = (d / d2_fracresc);
		a = ((a << FRACT_RESCALE) - dx * d);
		if (abs(a) > xytol_rescale) continue;
		b = ((b << FRACT_RESCALE) - dy * d);
		if (abs(b) > xytol_rescale) continue;
		c = ((c << FRACT_RESCALE) - dz * d);
		if (abs(c) > ztol_rescale) continue;
		a /= xytol;
		b /= xytol;
		c /= ztol;
		if ((a * a + b * b + c * c) <= (1 << (2 * FRACT_RESCALE)))
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
	if (Chains < 4) return;
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

__global__ void find_track_singlepass_kernel(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= totalpairs) return;
	if (pindices[3 * id] == 0 || pindices[3 * id + 1] == 0) return;
	IntChain *pA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id]));
	IntChain *pB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id + 1]));
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
	for (iis = pindices[3 * id + 2]; iis; iis = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis))->Reserved)
	{
		IntChain *pS = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis);
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
	if (Chains < 5) return;
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

__global__ void find_track_singlepass_kernel_TEST(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= totalpairs) return;
	if (pindices[3 * id] == 0 || pindices[3 * id + 1] == 0) return;
	IntChain *pA = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id]));
	IntChain *pB = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + pindices[3 * id + 1]));
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

	int d2 = dx * dx + dy * dy;
	int ss2 = (((long long)d2) << FRACT_RESCALE) / (dz * dz);
	int Chains = 0;
	int Volume = 0;
	int Clusters = 0;
	int xytol = pC->XYTolerance;
	int ztol = (pC->ZTolerance << Z_TO_XY_RESCALE_SHIFT);
	long long xytol2 = xytol * xytol;	
	long long ztol2 = ztol * ztol;	
	//int xytol_rescale = xytol << FRACT_RESCALE;
	//int ztol_rescale = ztol << FRACT_RESCALE;
	//int MinViewTag = min(pA->ViewTag, pB->ViewTag);
	IntChain *ppTrackGrains[_MAX_GRAINS_PER_TRACK_];

	int iis;
	for (iis = pindices[3 * id + 2]; iis; iis = ((IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis))->Reserved)
	{
		IntChain *pS = (IntChain *)(void *)((char *)(void *)ppViewEntryPoints + iis);
		int c = (pS->AvgZ - az) << Z_TO_XY_RESCALE_SHIFT;
		int d = (c << FRACT_RESCALE) / dz;
		int a = (pS->AvgX - ax) - ((d * dx) >> FRACT_RESCALE);
		int b = (pS->AvgY - ay) - ((d * dy) >> FRACT_RESCALE);
		long long dd = (a * dy - b * dx);
		if ((dd * dd) > d2 * xytol2) continue;
		dd = (a * dx + b * dy);
		if ((dd * dd) > d2 * (xytol2 + ((ss2 * ztol2) >> FRACT_RESCALE))) continue;
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
	if (Chains < 5) return;
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

}}