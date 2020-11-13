#include "CUDAMgr.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
 
#define _THROW_ON_CUDAERR_(x) { x; cudaError_t err = cudaDeviceSynchronize(); if (err) { sprintf(LastError, "%s File %s Line %d", cudaGetErrorString(err), __FILE__, __LINE__); return false; } }

#define _DEBUG_THROW_ON_CUDAERR_ { cudaError_t err = cudaDeviceSynchronize(); if (err) { sprintf(LastError, "%s File %s Line %d", cudaGetErrorString(err), __FILE__, __LINE__); return false; } }
#define _DEBUG_THROW_ON_CUDAERR_  { cudaError_t err = cudaDeviceSynchronize(); if (err) { sprintf(LastError, "%s File %s Line %d", cudaGetErrorString(err), __FILE__, __LINE__); return false; } FILE *f = fopen("c:\\temp\\nvidia.txt", "a+t"); fprintf(f, "\nDevice %d Line %d Time %d", DeviceId, __LINE__, clock()); fclose(f); }
#define _DEBUG_THROW_ON_CUDAERR_



inline void DUMP(char * str, int x) 
{ 
//	FILE *f = fopen("c:\\temp\\nvidia.txt", "at"); fprintf(f, "\n%s %d", str, x); fclose(f); 
}

inline void DUMP(char * str, void *px) 
{ 
//	FILE *f = fopen("c:\\temp\\nvidia.txt", "at"); fprintf(f, "\n%s %016X", str, px); fclose(f); 
}

int CUDAManager::GetDeviceCount()
{
	int c = -1;
	cudaError_t err = cudaGetDeviceCount(&c);
	if (err == cudaSuccess) return c;
	if (err == cudaErrorNoDevice) return 0;
	return -1;
}

void *CUDAManager::MallocHost(long size)
{
	/*return malloc(size);
	void *pImage = 0;
	if (cudaMallocHost((void **)&pImage, size)) return 0;
	return pImage;
	*/
	void *pMem = 0;
	if (cudaHostAlloc(&pMem, size, cudaHostAllocPortable)) return 0;
	return pMem;
}

void CUDAManager::FreeHost(void *pbuff)
{
	//free(pbuff); return;
	//cudaFree(pbuff);
	cudaFreeHost(pbuff);
}

#define ZERODEV(p) p = 0;
#define ZEROHOST(p) p = 0;

CUDAManager::CUDAManager(int devid)
{
	pDevImage = 0;
	pHostEqImage = 0;
	pHostBinImage = 0;
	pDevHistoImage = 0;
	pDevLookupTable = 0;
	pDev16Image = 0;
	pDevThresholdImage = 0;
	pDevEmptyImage = 0;
	pDevSegmentCountImage = 0;
	pDevSegmentImage = 0;
	pHostSegmentImage = 0;
	pHostSegmentCountImage = 0;
	pDevClusterWorkImage = 0;
	pDevClusterWorkCountImage = 0;
	pDevClusterBaseImage = 0;
	pDevClusterImage = 0;
	pDevClusterCountImage = 0;
	pHostClusterImage = 0;
	pHostClusterCountImage = 0;
	pDevErrorImage = 0;
	pHostErrorImage = 0;
	
	MemSize_pDevImage = 0;
	MemSize_pHostEqImage = 0;
	MemSize_pHostBinImage = 0;
	MemSize_pDevHistoImage = 0;
	MemSize_pDevLookupTable = 0;
	MemSize_pDev16Image = 0;
	MemSize_pDevThresholdImage = 0;
	MemSize_pDevEmptyImage = 0;
	MemSize_pDevSegmentCountImage = 0;
	MemSize_pDevSegmentImage = 0;
	MemSize_pHostSegmentImage = 0;
	MemSize_pHostSegmentCountImage = 0;
	MemSize_pDevClusterWorkImage = 0;
	MemSize_pDevClusterWorkCountImage = 0;
	MemSize_pDevClusterBaseImage = 0;
	MemSize_pDevClusterImage = 0;
	MemSize_pDevClusterCountImage = 0;
	MemSize_pHostClusterImage = 0;
	MemSize_pHostClusterCountImage = 0;	
	MemSize_pDevErrorImage = 0;
	MemSize_pHostErrorImage = 0;
	
	MaxImages = -1;

	cudaError_t err;
	cudaDeviceProp prop;
	ImageWidth = ImageHeight = -1;

	MaxSegmentsPerScanLine = 0;
	MaxClustersPerImage = 0;
	GreyLevelTargetMedian = 0;

	DeviceId = devid;
	DeviceName[0] = 0;
	strcpy(LastError, "Not Initialized");
	if (err = cudaGetDeviceProperties(&prop, devid)) 
	{
		strcpy(LastError, cudaGetErrorString(err));
		return;
	}
	strcpy(DeviceName, prop.name);
	if (err = cudaSetDevice(DeviceId))
	{
		strcpy(LastError, cudaGetErrorString(err));
		return;	
	}
	if (err = cudaDeviceReset())
	{
		strcpy(LastError, cudaGetErrorString(err));
		return;	
	}		
	if (err = cudaMemGetInfo(&AvailableMemory, &TotalMemory))
	{
		strcpy(LastError, cudaGetErrorString(err));
		return;		
	}
/*	if (AvailableMemory > 1073741824)
	{
		AvailableMemory = TotalMemory = 1073741824;
	}*/
	MaxThreadsPerBlock = prop.maxThreadsPerBlock;
	LastError[0] = 0;	
	GreyLevelTargetMedian = 200;
		
	WkCfgDumpEqImages = false;
	WkCfgDumpBinImages = false;
	WkCfgDumpSegments = false;
	WkCfgDumpClusters = true;
	WkCfgDumpClusters2ndMomenta = true;

	ZERODEV(pDevImage);
	ZEROHOST(pHostEqImage);
	ZEROHOST(pHostBinImage);
	ZERODEV(pDevHistoImage);
	ZERODEV(pDevLookupTable);
	ZERODEV(pDev16Image);
	ZERODEV(pDevThresholdImage);
	ZERODEV(pDevEmptyImage);
	ZERODEV(pDevSegmentCountImage);
	ZERODEV(pDevSegmentImage);
	ZEROHOST(pHostSegmentImage);
	ZEROHOST(pHostSegmentCountImage);
	ZERODEV(pDevClusterWorkImage);
	ZERODEV(pDevClusterWorkCountImage);
	ZERODEV(pDevClusterBaseImage);
	ZERODEV(pDevClusterImage);
	ZERODEV(pDevClusterCountImage);
	ZEROHOST(pHostClusterImage);
	ZEROHOST(pHostClusterCountImage);
	ZERODEV(pDevErrorImage);
	ZEROHOST(pDevErrorImage);
	
	Changed = true;	
}

#define RESETDEV(p) if (p) { cudaFree(p); p = 0; }
#define RESETHOST(p) if (p) { cudaFreeHost(p); p = 0; }
#define COMPUTEMEM(p, _sz_) { FILE *fdump = fopen("c:\\temp\\nvidia.txt", "a+t"); fprintf(fdump, "\ndevid %d compute object %s items %d bytes %lld line %d", DeviceId, # p, _sz_, MemSize_ ## p = _sz_, __LINE__); fclose(fdump);  MemSize_ ## p = _sz_; }
#define ALLOCDEV(p, _count_) { FILE *fdump = fopen("c:\\temp\\nvidia.txt", "a+t"); fprintf(fdump, "\ndevid %d object %s alloc bytes %lld line %d", DeviceId, # p, _count_ * MemSize_ ## p, __LINE__); fclose(fdump); cudaError_t err; if (err = cudaMalloc((void **)&p, _count_ * MemSize_ ## p)) { p = 0; sprintf(LastError, "%s for %d bytes line %d", cudaGetErrorString(err), _count_ * MemSize_ ## p, __LINE__); return false; }  }
#define ALLOCHOST(p, _count_) { cudaError_t err; if (err = cudaMallocHost((void **)&p, _count_ * MemSize_ ## p)) { p = 0; sprintf(LastError, "%s for %lld bytes line %d", cudaGetErrorString(err), _count_ * MemSize_ ## p, __LINE__); return false; } }

CUDAManager::~CUDAManager()
{
	cudaSetDevice(DeviceId);
	RESETDEV(pDevImage);
	RESETHOST(pHostEqImage);
	RESETHOST(pHostBinImage);
	RESETDEV(pDevHistoImage);
	RESETDEV(pDevLookupTable);
	RESETDEV(pDev16Image);
	RESETDEV(pDevThresholdImage);
	RESETDEV(pDevEmptyImage);
	RESETDEV(pDevSegmentCountImage);
	RESETDEV(pDevSegmentImage);
	RESETHOST(pHostSegmentImage);
	RESETHOST(pHostSegmentCountImage);
	RESETDEV(pDevClusterWorkImage);
	RESETDEV(pDevClusterWorkCountImage);
	RESETDEV(pDevClusterBaseImage);
	RESETDEV(pDevClusterImage);
	RESETDEV(pDevClusterCountImage);
	RESETHOST(pHostClusterImage);
	RESETHOST(pHostClusterCountImage);
	RESETDEV(pDevErrorImage);
	RESETHOST(pDevErrorImage);
	
	cudaDeviceReset();
}

bool CUDAManager::ReconfigureMemory()
{	
	cudaSetDevice(DeviceId);
	cudaDeviceReset();

	RESETDEV(pDevImage);
	RESETHOST(pHostEqImage);
	RESETHOST(pHostBinImage);
	RESETDEV(pDevHistoImage);
	RESETDEV(pDevLookupTable);
	RESETDEV(pDev16Image);
	RESETDEV(pDevThresholdImage);
	RESETDEV(pDevEmptyImage);
	RESETDEV(pDevSegmentCountImage);
	RESETDEV(pDevSegmentImage);
	RESETHOST(pHostSegmentImage);
	RESETHOST(pHostSegmentCountImage);
	RESETDEV(pDevClusterWorkImage);
	RESETDEV(pDevClusterWorkCountImage);
	RESETDEV(pDevClusterBaseImage);
	RESETDEV(pDevClusterImage);
	RESETDEV(pDevClusterCountImage);
	RESETHOST(pHostClusterImage);
	RESETHOST(pHostClusterCountImage);	
	RESETDEV(pDevErrorImage);
	RESETHOST(pDevErrorImage);

	COMPUTEMEM(pDevImage, ImageWidth * ImageHeight * sizeof(unsigned char));
	COMPUTEMEM(pHostEqImage, ImageWidth * ImageHeight * sizeof(unsigned char));
	COMPUTEMEM(pHostBinImage, ImageWidth * ImageHeight * sizeof(unsigned char));
	COMPUTEMEM(pDevHistoImage, 256 * MaxThreadsPerBlock * sizeof(unsigned));
	COMPUTEMEM(pDevLookupTable, 256 * sizeof(short));
	COMPUTEMEM(pDev16Image, ImageWidth * ImageHeight * sizeof(short));
	COMPUTEMEM(pDevThresholdImage, ImageWidth * ImageHeight * sizeof(short));
	COMPUTEMEM(pDevEmptyImage, ImageWidth * ImageHeight * sizeof(short));
	COMPUTEMEM(pDevSegmentCountImage, ImageHeight * sizeof(short));
	COMPUTEMEM(pDevSegmentImage, ImageHeight * MaxSegmentsPerScanLine * sizeof(IntSegment));
	COMPUTEMEM(pHostSegmentImage, ImageHeight * MaxSegmentsPerScanLine * sizeof(IntSegment));
	COMPUTEMEM(pHostSegmentCountImage, ImageHeight * sizeof(short));
	COMPUTEMEM(pDevClusterWorkImage, ImageHeight * MaxSegmentsPerScanLine * sizeof(IntClusterW));
	COMPUTEMEM(pDevClusterWorkCountImage, ImageHeight * sizeof(short));
	COMPUTEMEM(pDevClusterBaseImage, ImageHeight * sizeof(int));
	COMPUTEMEM(pDevClusterImage, MaxClustersPerImage * sizeof(IntCluster));
	COMPUTEMEM(pDevClusterCountImage, sizeof(unsigned));
	COMPUTEMEM(pHostClusterImage, MaxClustersPerImage * sizeof(IntCluster));
	COMPUTEMEM(pHostClusterCountImage, sizeof(unsigned));
	COMPUTEMEM(pDevErrorImage, 2 * sizeof(unsigned));
	COMPUTEMEM(pHostErrorImage, 2 * sizeof(unsigned));
	
	{
		cudaError_t err;
		if (err = cudaMemGetInfo(&AvailableMemory, &TotalMemory))
		{
			strcpy(LastError, cudaGetErrorString(err));
			return false;
		}
		{ FILE *fdump = fopen("c:\\temp\\nvidia.txt", "a+t"); fprintf(fdump, "\ndevid %d avail %lld total %lld", DeviceId, AvailableMemory, TotalMemory); fclose(fdump); }
	}
	
	MaxImages = 
		(	
			(long long)(AvailableMemory - (16 * 1048576 /* save 16 MB of GPU memory */)) - 
			(MemSize_pDevThresholdImage + MemSize_pDevEmptyImage + MemSize_pDevHistoImage + MemSize_pDevLookupTable)
		) / 
		(
			MemSize_pDevImage + MemSize_pDev16Image + MemSize_pDevSegmentCountImage + MemSize_pDevSegmentImage + MemSize_pDevClusterWorkImage + MemSize_pDevClusterWorkCountImage + 
			MemSize_pDevClusterBaseImage + MemSize_pDevClusterImage + MemSize_pDevClusterCountImage + 2 * MemSize_pDevErrorImage
		);
	    { FILE *fdump = fopen("c:\\temp\\nvidia.txt", "a+t"); fprintf(fdump, "\ndevid %d num %lld den %d", DeviceId, 
			(	
			(long long)(AvailableMemory - (16 * 1048576 /* save 16 MB of GPU memory */)) - 
			(MemSize_pDevThresholdImage + MemSize_pDevEmptyImage + MemSize_pDevHistoImage + MemSize_pDevLookupTable)
		)
		, 
		(
			MemSize_pDevImage + MemSize_pDev16Image + MemSize_pDevSegmentCountImage + MemSize_pDevSegmentImage + MemSize_pDevClusterWorkImage + MemSize_pDevClusterWorkCountImage + 
			MemSize_pDevClusterBaseImage + MemSize_pDevClusterImage + MemSize_pDevClusterCountImage + 2 * MemSize_pDevErrorImage
		)
		); fclose(fdump); }
		{ FILE *fdump = fopen("c:\\temp\\nvidia.txt", "a+t"); fprintf(fdump, "\ndevid %d maximages %d", DeviceId, MaxImages); fclose(fdump); }

	ALLOCDEV(pDevImage, MaxImages);
	ALLOCHOST(pHostEqImage, MaxImages);
	ALLOCHOST(pHostBinImage, MaxImages);
	ALLOCDEV(pDevHistoImage, 1);
	ALLOCDEV(pDevLookupTable, 1);
	ALLOCDEV(pDev16Image, MaxImages);
	ALLOCDEV(pDevThresholdImage, 1);
	ALLOCDEV(pDevEmptyImage, 1);
	ALLOCDEV(pDevSegmentCountImage, MaxImages);
	ALLOCDEV(pDevSegmentImage, MaxImages);
	ALLOCHOST(pHostSegmentImage, MaxImages);
	ALLOCHOST(pHostSegmentCountImage, MaxImages);
	ALLOCDEV(pDevClusterWorkImage, MaxImages);
	ALLOCDEV(pDevClusterWorkCountImage, MaxImages);
	ALLOCDEV(pDevClusterBaseImage, MaxImages);
	ALLOCDEV(pDevClusterImage, MaxImages);
	ALLOCDEV(pDevClusterCountImage, MaxImages);
	ALLOCHOST(pHostClusterImage, MaxImages);
	ALLOCHOST(pHostClusterCountImage, MaxImages);
	ALLOCDEV(pDevErrorImage, 2 * MaxImages);
	ALLOCHOST(pHostErrorImage, 2 * MaxImages);

	Changed = false;
	return true;
}

bool CUDAManager::SetEmptyImage(unsigned short *img)
{
	cudaSetDevice(DeviceId);
	if (Changed && ReconfigureMemory() == false)
	{	
		strcpy(LastError, "Memory must be configured before loading the empty image.");
		return false;
	}
	cudaError_t err = cudaMemcpy(pDevEmptyImage, img, ImageWidth * ImageHeight * sizeof(short), cudaMemcpyHostToDevice);
	if (err) { strcpy(LastError, cudaGetErrorString(err)); return false; }
	return true;
}

bool CUDAManager::SetThresholdImage(short *img)
{
	cudaSetDevice(DeviceId);
	if (Changed && ReconfigureMemory() == false)
	{
		strcpy(LastError, "Memory must be configured before loading the threshold image.");
		return false;
	}	
	cudaError_t err = cudaMemcpy(pDevThresholdImage, img, ImageWidth * ImageHeight * sizeof(short), cudaMemcpyHostToDevice);
	if (err) { strcpy(LastError, cudaGetErrorString(err)); return false; }
	return true;	
}

void CUDAManager::SetWorkingConfiguration(bool dumpeq, bool dumpbin, bool dumpsegs, bool dumpclusters, bool dumpclusters2ndmom)
{
	cudaSetDevice(DeviceId);
	WkCfgDumpEqImages = dumpeq;
	WkCfgDumpBinImages = dumpbin;
	WkCfgDumpSegments = dumpsegs;
	WkCfgDumpClusters = dumpclusters;
	WkCfgDumpClusters2ndMomenta = dumpclusters2ndmom;
	//printf("\nWkCfg %d %d %d %d %d", WkCfgDumpEqImages, WkCfgDumpBinImages, WkCfgDumpSegments, WkCfgDumpClusters, WkCfgDumpClusters2ndMomenta);
}

/**********************
 *                    *
 * Begin CUDA Kernels *
 *                    *
 **********************/
 
__global__ void resetclusterline_m2_kernel(IntClusterW *pMBlkWorkClusters, short *pMBlkCountWorkClusters, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);	
	if (idx >= totalimages * height) return;
	int yline = idx % height;	
	int i;			
	unsigned xmax, xmin, seglen, SX;
	int sc = pMBlkCountSegments[idx];
	pMBlkCountWorkClusters[idx] = sc;
	int ids = idx * maxsegsperline;		
	IntClusterW *pC = pMBlkWorkClusters + ids;
	pMBlkSegments += ids;
	for (i = 0; i < sc; i++)
	{
		xmax = pMBlkSegments->Right + 1;
		xmin = pMBlkSegments->Left; 
		seglen = xmax - xmin;	
		pMBlkSegments->pCluster = pC;
		pMBlkSegments->Area = seglen;
		pMBlkSegments->YSum = yline * seglen;
		SX = (xmax * (xmax - 1) - xmin * (xmin - 1)) >> 1;
		pMBlkSegments->XSum = SX;
		pMBlkSegments->YYSum = (long long)(yline * yline) * (long long)seglen;
		pMBlkSegments->XYSum = (long long)yline * (long long)SX;
		pMBlkSegments->XXSum = ((1 + (long long)(3 + 2 * xmax) * xmax) * xmax - (1 + (long long)(3 + 2 * xmin) * xmin) * xmin) / 6;	
		pMBlkSegments->Flag = 0;
		pMBlkSegments->pMergeTo = 0;
		pMBlkSegments++;
	}
	/*
	for (; i < maxsegsperline; i++)
	{
		pMBlkWorkClusters[idx * maxsegsperline + i].Area = 0;
	}
	*/	
}

__global__ void resetclusterline_kernel(IntClusterW *pMBlkWorkClusters, short *pMBlkCountWorkClusters, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);	
	if (idx >= totalimages * height) return;
	int yline = idx % height;
	int i;		
	unsigned xmax, xmin, seglen;
	int sc = pMBlkCountSegments[idx];	
	pMBlkCountWorkClusters[idx] = sc;
	int ids = idx * maxsegsperline;		
	IntClusterW *pC = pMBlkWorkClusters + ids;
	pMBlkSegments += ids;
	for (i = 0; i < sc; i++)
	{
		xmax = pMBlkSegments->Right + 1;
		xmin = pMBlkSegments->Left; 
		seglen = xmax - xmin;	
		pMBlkSegments->pCluster = pC;
		pMBlkSegments->Area = seglen;
		pMBlkSegments->YSum = yline * seglen;
		pMBlkSegments->XSum = (xmax * (xmax - 1) - xmin * (xmin - 1)) >> 1;
		pMBlkSegments->Flag = 0;
		pMBlkSegments->pMergeTo = 0;
		pMBlkSegments++;
		
	}
/*	for (; i < maxsegsperline; i++)
	{
		pMBlkWorkClusters[idx * maxsegsperline + i].Area = 0;
	}*/	
}

__global__ void OLD_countclusters_kernel(IntClusterW *pMBlkWorkClusters, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);	
	if (idx >= totalimages * height) return;
	pMBlkCountWorkClusters[idx] = 0;
	int sc = pMBlkCountSegments[idx];
	int i, j;
	for (i = j = 0; i < sc; i++)
		if (pMBlkWorkClusters[idx * maxsegsperline + i].Area > 0) j++;
	pMBlkCountWorkClusters[idx] = j;
}

__global__ void countclusters_kernel(IntSegment *pMBlkSegments, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);	
	if (idx >= totalimages * height) return;
	pMBlkCountWorkClusters[idx] = 0;
	int sc = pMBlkCountSegments[idx];
	int i, j;
	for (i = j = 0; i < sc; i++)
		if (pMBlkSegments[idx * maxsegsperline + i].Area > 0) j++;
	pMBlkCountWorkClusters[idx] = j;
}

__global__ void setclusterbase_kernel(int *pMBlkCountClusters, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int *pMBlkClusterBase, int height, int maxclusters, unsigned *pMBlkErr)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);	
	int base = idx * height;
	int i;	
	pMBlkClusterBase[base] = 0;
	for (i = 1; i < height; i++)			
		pMBlkClusterBase[base + i] = pMBlkClusterBase[base + i - 1] + pMBlkCountWorkClusters[base + i - 1];
	pMBlkCountClusters[idx] = pMBlkClusterBase[base + i - 1] + pMBlkCountWorkClusters[base + i - 1];
	if (pMBlkCountClusters[idx] >= maxclusters) 	
	{
		pMBlkErr[idx] |= NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW;
		pMBlkCountClusters[idx] = maxclusters;
		for (i = 1; i < height; i++)	
			if (pMBlkClusterBase[base + i] >= maxclusters)
				pMBlkClusterBase[base + i] = maxclusters;
		for (i = 1; i < height; i++)
			pMBlkCountWorkClusters[base + i - 1] = pMBlkClusterBase[base + i] - pMBlkClusterBase[base + i - 1];
		pMBlkCountWorkClusters[base + i - 1] = maxclusters - pMBlkClusterBase[base + i - 1];
	}
}

__global__ void OLD_finalizeclusters_m2_kernel(IntCluster *pMBlkCountClusters, IntClusterW *pMBlkClusters, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int *pMBlkClusterBase, int maxsegsperline, int height, int maxclusters, int totalimages)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);		
	if (idx >= (totalimages * height)) return;
	int yline = idx % height;
	if (yline < 4 || yline >= height - 4) return;	
	int base = idx * maxsegsperline;
	int obase = pMBlkClusterBase[idx] + (idx / height) * maxclusters;
	int j, k, sc;
	sc = pMBlkCountSegments[idx];
	for (j = k = 0; j < sc; j++)
	{
		if (pMBlkClusters[base + j].Area > 0)
		{
			IntClusterW *pG = pMBlkClusters + base + j;
			IntCluster *pG1 = pMBlkCountClusters + obase + k++;
			int a = pG->Area;
			int x = (pG->XSum << RESCALING_BITSHIFT) / a;
			int y = (pG->YSum << RESCALING_BITSHIFT) / a;
			int xa = x >> RESCALING_BITSHIFT;
			int ya = y >> RESCALING_BITSHIFT;
			int dx = x - (xa << RESCALING_BITSHIFT);
			int dy = y - (ya << RESCALING_BITSHIFT);
			int xxa = pG->XXSum + xa * xa * a - 2 * xa * pG->XSum;
			int yya = pG->YYSum + ya * ya * a - 2 * ya * pG->YSum;
			int xya = pG->XYSum + xa * ya * a - ya * pG->XSum - xa * pG->YSum;						
			pG1->Area = a;
			pG1->XX = ((xxa << (2 * RESCALING_BITSHIFT)) - dx * dx * a) >> RESCALING_BITSHIFT;			
			pG1->YY = ((yya << (2 * RESCALING_BITSHIFT)) - dy * dy * a) >> RESCALING_BITSHIFT;	
			pG1->XY = (dx * dy * a - (xya << (2 * RESCALING_BITSHIFT))) >> RESCALING_BITSHIFT;
			pG1->X = x;
			pG1->Y = y;
/*
			pG1->pMergeTo = 0;
			pG1->Flag = pG->Flag;
*/
			if (k >= maxclusters) k = maxclusters - 1;
		}	
	}
}

__global__ void OLD_finalizeclusters_kernel(IntCluster *pMBlkCountClusters, IntClusterW *pMBlkClusters, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int *pMBlkClusterBase, int maxsegsperline, int height, int maxclusters, int totalimages)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);		
	if (idx >= (totalimages * height)) return;
	int yline = idx % height;
	if (yline < 4 || yline >= height - 4) return;
	int base = idx * maxsegsperline;
	int obase = pMBlkClusterBase[idx] + (idx / height) * maxclusters;
	int j, k, sc;
	sc = pMBlkCountSegments[idx];
	for (j = k = 0; j < sc; j++)
	{
		if (pMBlkClusters[base + j].Area > 0)
		{
			IntClusterW *pG = pMBlkClusters + base + j;
			IntCluster *pG1 = pMBlkCountClusters + obase + k++;
			pG1->Area = pG->Area;
			pG1->XX = pG1->XY = pG1->YY = 0;
			pG1->X = (pG->XSum << RESCALING_BITSHIFT) / pG->Area;
			pG1->Y = (pG->YSum << RESCALING_BITSHIFT) / pG->Area;
/*
			pG1->pMergeTo = 0;
			pG1->Flag = pG->Flag;
*/
			if (k >= maxclusters) k = maxclusters - 1;			
		}	
	}
}

__global__ void finalizeclusters_m2_kernel(IntCluster *pMBlkCountClusters, IntSegment *pMBlkSegments, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int *pMBlkClusterBase, int maxsegsperline, int height, int maxclusters, int totalimages)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);		
	if (idx >= (totalimages * height)) return;
	int yline = idx % height;
	if (yline < 4 || yline >= height - 4) return;	
	int base = idx * maxsegsperline;
	int obase = pMBlkClusterBase[idx] + (idx / height) * maxclusters;
	int j, k, sc;
	sc = pMBlkCountSegments[idx];
	for (j = k = 0; j < sc; j++)
	{
		if (pMBlkSegments[base + j].Area > 0)
		{
			IntSegment *pG = pMBlkSegments + base + j;
			IntCluster *pG1 = pMBlkCountClusters + obase + k++;
			int a = pG->Area;
			int x = (pG->XSum << RESCALING_BITSHIFT) / a;
			int y = (pG->YSum << RESCALING_BITSHIFT) / a;
			int xa = x >> RESCALING_BITSHIFT;
			int ya = y >> RESCALING_BITSHIFT;
			int dx = x - (xa << RESCALING_BITSHIFT);
			int dy = y - (ya << RESCALING_BITSHIFT);
			int xxa = pG->XXSum + xa * xa * a - 2 * xa * pG->XSum;
			int yya = pG->YYSum + ya * ya * a - 2 * ya * pG->YSum;
			int xya = pG->XYSum + xa * ya * a - ya * pG->XSum - xa * pG->YSum;		
			int xx = ((xxa << (2 * RESCALING_BITSHIFT)) - dx * dx * a) >> RESCALING_BITSHIFT;
			int yy = ((yya << (2 * RESCALING_BITSHIFT)) - dy * dy * a) >> RESCALING_BITSHIFT;
			int xy = (dx * dy * a - (xya << (2 * RESCALING_BITSHIFT))) >> RESCALING_BITSHIFT;
			pG1->Area = a;
			pG1->X = x;
			pG1->Y = y;			
			pG1->XX = xx;			
			pG1->YY = yy;	
			pG1->XY = xy;
/*
			pG1->pMergeTo = 0;
			pG1->Flag = pG->Flag;
*/
			if (k >= maxclusters) k = maxclusters - 1;
		}	
	}
}

__global__ void finalizeclusters_kernel(IntCluster *pMBlkCountClusters, IntSegment *pMBlkSegments, short *pMBlkCountWorkClusters, short *pMBlkCountSegments, int *pMBlkClusterBase, int maxsegsperline, int height, int maxclusters, int totalimages)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);		
	if (idx >= (totalimages * height)) return;
	int yline = idx % height;
	if (yline < 4 || yline >= height - 4) return;
	int base = idx * maxsegsperline;
	int obase = pMBlkClusterBase[idx] + (idx / height) * maxclusters;
	int j, k, sc;
	sc = pMBlkCountSegments[idx];
	for (j = k = 0; j < sc; j++)
	{
		if (pMBlkSegments[base + j].Area > 0)
		{
			IntSegment *pG = pMBlkSegments + base + j;
			IntCluster *pG1 = pMBlkCountClusters + obase + k++;
			pG1->Area = pG->Area;
			pG1->XX = pG1->XY = pG1->YY = 0;
			pG1->X = (pG->XSum << RESCALING_BITSHIFT) / pG->Area;
			pG1->Y = (pG->YSum << RESCALING_BITSHIFT) / pG->Area;
/*
			pG1->pMergeTo = 0;
			pG1->Flag = pG->Flag;
*/
			if (k >= maxclusters) k = maxclusters - 1;			
		}	
	}
}

__global__ void OLD_getclustersline_m2_kernel(int pass, short *pMBlkCountWorkClusters, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	int yline = (2 << pass) * idx + (1 << pass);	
	if (yline / height >= totalimages) return;
	int ylinen = yline % height;
	if (ylinen < 4 || ylinen >= height - 4) return;
	pMBlkCountWorkClusters[yline] = 0;
	if (ylinen >= height - 4) return;	
	int psegs = (ylinen > 4) ? (pMBlkCountSegments[yline - 1]) : 0;
	int tsegs = pMBlkCountSegments[yline];
	int psegbase = (ylinen > 4) ? ((yline - 1) * maxsegsperline) : 0;
	int tsegbase = yline * maxsegsperline;
	int p, t, ps;
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left--;
		pMBlkSegments[psegbase + p].Right++;
	}
	for (t = p = ps = 0; t < tsegs; t++)
	{
		p = ps;
		while (p < psegs && pMBlkSegments[tsegbase + t].Left > pMBlkSegments[psegbase + p].Right) p++;
		ps = p;
		for (p = ps; p < psegs && pMBlkSegments[tsegbase + t].Right >= pMBlkSegments[psegbase + p].Left; p++)
		{
			IntClusterW *pT = pMBlkSegments[tsegbase + t].pCluster;
			while (pT->pMergeTo) pT = pT->pMergeTo;
			IntClusterW *pP = pMBlkSegments[psegbase + p].pCluster;
			while (pP->pMergeTo) pP = pP->pMergeTo;
			if (pT != pP)
			{
				pT->Flag = 1;
				pP->Flag = 2;
				pP->Area += pT->Area;
				pP->XSum += pT->XSum;
				pP->YSum += pT->YSum;
				pP->XXSum += pT->XXSum;
				pP->XYSum += pT->XYSum;
				pP->YYSum += pT->YYSum;				
				IntClusterW *pZ = pT;
				pMBlkSegments[tsegbase + t].pCluster = pP;
				while (pZ->pMergeTo)
				{
					pT = pZ;
					pZ = pZ->pMergeTo;
					pT->pMergeTo = pP;
					pT->Area = 0;
				}
				pZ->pMergeTo = pP;
				pZ->Area = 0;				
			}
		}
	}
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left++;
		pMBlkSegments[psegbase + p].Right--;
	}	
}

__global__ void OLD_getclustersline_kernel(int pass, short *pMBlkCountWorkClusters, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	int yline = (2 << pass) * idx + (1 << pass);	
	if (yline / height >= totalimages) return;
	int ylinen = yline % height;		
	if (ylinen < 4 || ylinen >= height - 4) return;
	pMBlkCountWorkClusters[yline] = 0;
	int psegs = (ylinen > 4) ? (pMBlkCountSegments[yline - 1]) : 0;
	int tsegs = pMBlkCountSegments[yline];
	int psegbase = (ylinen > 4) ? ((yline - 1) * maxsegsperline) : 0;
	int tsegbase = yline * maxsegsperline;
	int p, t, ps;
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left--;
		pMBlkSegments[psegbase + p].Right++;
	}
	for (t = p = ps = 0; t < tsegs; t++)
	{
		p = ps;
		while (p < psegs && pMBlkSegments[tsegbase + t].Left > pMBlkSegments[psegbase + p].Right) p++;
		//__syncthreads();
		ps = p;
		for (p = ps; p < psegs && pMBlkSegments[tsegbase + t].Right >= pMBlkSegments[psegbase + p].Left; p++)
		{
			//__syncthreads();
			IntClusterW *pT = pMBlkSegments[tsegbase + t].pCluster;
			while (pT->pMergeTo) pT = pT->pMergeTo;
			IntClusterW *pP = pMBlkSegments[psegbase + p].pCluster;
			while (pP->pMergeTo) pP = pP->pMergeTo;
			if (pT != pP)
			{
				pT->Flag = 1;
				pP->Flag = 2;
				pP->Area += pT->Area;
				pP->XSum += pT->XSum;
				pP->YSum += pT->YSum;
				IntClusterW *pZ = pT;
				pMBlkSegments[tsegbase + t].pCluster = pP;
				while (pZ->pMergeTo)
				{
					pT = pZ;
					pZ = pZ->pMergeTo;
					pT->pMergeTo = pP;
					pT->Area = 0;
				}
				pZ->pMergeTo = pP;
				pZ->Area = 0;				
			}
		}
	}
	//__syncthreads();
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left++;
		pMBlkSegments[psegbase + p].Right--;
	}	
}

__global__ void getclustersline_m2_kernel(int pass, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	int yline = (2 << pass) * idx + (1 << pass);	
	if (yline / height >= totalimages) return;
	int ylinen = yline % height;
	if (ylinen < 4 || ylinen >= height - 4) return;
	//pMBlkCountWorkClusters[yline] = 0;
	if (ylinen >= height - 4) return;	
	int psegs = (ylinen > 4) ? (pMBlkCountSegments[yline - 1]) : 0;
	int tsegs = pMBlkCountSegments[yline];
	int psegbase = (ylinen > 4) ? ((yline - 1) * maxsegsperline) : 0;
	int tsegbase = yline * maxsegsperline;
	int p, t, ps;
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left--;
		pMBlkSegments[psegbase + p].Right++;
	}
	for (t = p = ps = 0; t < tsegs; t++)
	{
		p = ps;
		while (p < psegs && pMBlkSegments[tsegbase + t].Left > pMBlkSegments[psegbase + p].Right) p++;
		ps = p;
		for (p = ps; p < psegs && pMBlkSegments[tsegbase + t].Right >= pMBlkSegments[psegbase + p].Left; p++)
		{
			IntSegment *pT = pMBlkSegments + tsegbase + t;
			while (pT->pMergeTo) pT = pT->pMergeTo;
			IntSegment *pP = pMBlkSegments + psegbase + p;
			while (pP->pMergeTo) pP = pP->pMergeTo;
			if (pT != pP)
			{
				pT->Flag = 1;
				pP->Flag = 2;
				pP->Area += pT->Area;
				pP->XSum += pT->XSum;
				pP->YSum += pT->YSum;
				pP->XXSum += pT->XXSum;
				pP->XYSum += pT->XYSum;
				pP->YYSum += pT->YYSum;				
				IntSegment *pZ = pT;				
				while (pZ->pMergeTo)
				{
					pT = pZ;
					pZ = pZ->pMergeTo;
					pT->pMergeTo = pP;
					pT->Area = 0;
				}
				pZ->pMergeTo = pP;
				pZ->Area = 0;				
			}
		}
	}
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left++;
		pMBlkSegments[psegbase + p].Right--;
	}	
}

__global__ void getclustersline_kernel(int pass, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int maxsegsperline, int height, int totalimages)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	int yline = (2 << pass) * idx + (1 << pass);	
	if (yline / height >= totalimages) return;
	int ylinen = yline % height;		
	if (ylinen < 4 || ylinen >= height - 4) return;
	//pMBlkCountWorkClusters[yline] = 0;
	int psegs = (ylinen > 4) ? (pMBlkCountSegments[yline - 1]) : 0;
	int tsegs = pMBlkCountSegments[yline];
	int psegbase = (ylinen > 4) ? ((yline - 1) * maxsegsperline) : 0;
	int tsegbase = yline * maxsegsperline;
	int p, t, ps;
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left--;
		pMBlkSegments[psegbase + p].Right++;
	}
	for (t = p = ps = 0; t < tsegs; t++)
	{
		p = ps;
		while (p < psegs && pMBlkSegments[tsegbase + t].Left > pMBlkSegments[psegbase + p].Right) p++;		
		ps = p;
		for (p = ps; p < psegs && pMBlkSegments[tsegbase + t].Right >= pMBlkSegments[psegbase + p].Left; p++)
		{			
			IntSegment *pT = pMBlkSegments + tsegbase + t;
			while (pT->pMergeTo) pT = pT->pMergeTo;
			IntSegment *pP = pMBlkSegments + psegbase + p;
			while (pP->pMergeTo) pP = pP->pMergeTo;
			if (pT != pP)
			{
				pT->Flag = 1;
				pP->Flag = 2;
				pP->Area += pT->Area;
				pP->XSum += pT->XSum;
				pP->YSum += pT->YSum;
				IntSegment *pZ = pT;				
				while (pZ->pMergeTo)
				{
					pT = pZ;
					pZ = pZ->pMergeTo;
					pT->pMergeTo = pP;
					pT->Area = 0;
				}
				pZ->pMergeTo = pP;
				pZ->Area = 0;				
			}
		}
	}	
	for (p = 0; p < psegs; p++)
	{
		pMBlkSegments[psegbase + p].Left++;
		pMBlkSegments[psegbase + p].Right--;
	}	
}

__global__ void segments_kernel(unsigned char *pMBlkImBin, IntSegment *pMBlkSegments, short *pMBlkCountSegments, int width, int maxsegsperline, int totalheight, int height, unsigned *pMBlkErr)
{ 	
	int id = (blockIdx.x * blockDim.x + threadIdx.x);
	if (id >= totalheight) return;
	int ih = id % height;
	if (ih < 4 || ih >= height - 4)
	{
		pMBlkCountSegments[id] = 0;
		return;
	}		
	int idx = id * width;	
	int ids = id * maxsegsperline;	
	int count = 0;
	int segstart;
	int x = idx + 4;
	int e = width + idx;	
	pMBlkImBin[e - 1] = pMBlkImBin[e - 2] = pMBlkImBin[e - 3] = pMBlkImBin[e - 4] = 1;
	while (count < maxsegsperline)
	{	
		while (pMBlkImBin[x] + pMBlkImBin[x + 1] + pMBlkImBin[x + 2] + pMBlkImBin[x + 3] == 0) x += 4;	
		while (pMBlkImBin[x] == 0) x++;
		if (pMBlkImBin[x] == 1)
		{
			pMBlkImBin[x] = 0;
			break;
		}
		segstart = x - idx;		
		while (pMBlkImBin[x] == 255) x++;
		pMBlkSegments[ids + count].Left = segstart;
		pMBlkSegments[ids + count].Right = x - idx - 1;
		//pMBlkSegments[ids + count].pCluster = 0;
		count++;				
	}
	if (count == maxsegsperline) pMBlkErr[id / height] |= NVIDIAIMGPROC_ERR_SEGMENT_OVERFLOW;		
	pMBlkCountSegments[id] = count;
	pMBlkImBin[e - 1] = pMBlkImBin[e - 2] = pMBlkImBin[e - 3] = pMBlkImBin[e - 4] = 0;
}

__global__ void toshort_kernel(unsigned char *pMBlkIm8, short *pMBlkLookupTable, short *pMBlkEmptyImage, short *pMBlkIm16, int imsize)
{ 
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * imsize; 
	pMBlkIm16[idx] = ((short)pMBlkLookupTable[pMBlkIm8[idx]]) + pMBlkEmptyImage[idx % imsize];
}

__global__ void historeset_kernel(unsigned *pMBlkHisto)
{ 
	pMBlkHisto[(blockIdx.x * blockDim.x + threadIdx.x)] = 0;
}

/*
__global__ void histo_kernel(unsigned char *pMBlkIm8, unsigned *pMBlkHisto, int size)
{ 
	int idx = threadIdx.x * size;
	int ide = idx + size;	
	int disp = (threadIdx.x << 8);
	while (idx < ide)
		pMBlkHisto[pMBlkIm8[idx++] + disp]++;
}
*/

#define HISTO_SUBSAMPLE_FACTOR 16

__global__ void histo_kernel(unsigned char *pMBlkIm8, unsigned *pMBlkHisto, int size)
{ 
	int idx = threadIdx.x;		
	int disp = (threadIdx.x << 8);
	int inc = blockDim.x * HISTO_SUBSAMPLE_FACTOR;
	while (idx < size)
	{
		pMBlkHisto[pMBlkIm8[idx] + disp]++;
		idx += inc;
	}
}

__global__ void histosum_kernel(unsigned *pMBlkHisto, int size)
{ 
	while (--size > 0)
		pMBlkHisto[threadIdx.x] += pMBlkHisto[threadIdx.x + (size << 8)];
}

__global__ void hmedian_kernel(unsigned *pMBlkHisto, short *pMBlkLookupTable, int imgsize, int targetmedian, unsigned *pMBlkOut)
{ 
	int med = (imgsize / 2) / HISTO_SUBSAMPLE_FACTOR;
	int i;
	int sum = 0;
	for (i = 0; i < 256 && sum < med; i++) sum += pMBlkHisto[i];
	pMBlkHisto[0] = i;
	pMBlkHisto[1] = sum;
	int lm = (targetmedian << 16) / i;
	pMBlkOut[0] = (unsigned)i;
	if (i >= 64) for (i = 0; i < 256; i++) pMBlkLookupTable[i] = (short)((i * lm) >> 16);
	else for (i = 0; i < 256; i++) pMBlkLookupTable[i] = (short)i;
}

__global__ void convolve_cut_kernel(short *pMBlkIm16, unsigned char *pMBlkImBin, short *pMBlkThresholdImage, int width, int imsize)
{ 
	int idt = (blockIdx.x * blockDim.x + threadIdx.x) + 4 * width;
	int idx = idt + (blockIdx.y * blockDim.y + threadIdx.y) * imsize;
/*	if (idt < (2 * width + 2) || idt >= imsize - 2 * width - 2)
	{
		pMBlkImBin[idx] = 0;
		return;
	}*/
	int idxA = idx - width;
	int idxB = idxA - width;
	int idx1 = idx + width;
	int idx2 = idx1 + width;
	short s = 	
		pMBlkIm16[idxB - 2] + pMBlkIm16[idxB - 1] + pMBlkIm16[idxB    ] + pMBlkIm16[idxB + 1] + pMBlkIm16[idxB + 2] + 
		pMBlkIm16[idxA - 2] - pMBlkIm16[idxA - 1] - 2 * pMBlkIm16[idxA] - pMBlkIm16[idxA + 1] + pMBlkIm16[idxA + 2] +
		pMBlkIm16[idx  - 2] - 2 * pMBlkIm16[idx - 1] - 4 * pMBlkIm16[idx] - 2 * pMBlkIm16[idx + 1] + pMBlkIm16[idx + 2] +
		pMBlkIm16[idx1 - 2] - pMBlkIm16[idx1 - 1] - 2 * pMBlkIm16[idx1] - pMBlkIm16[idx1 + 1] + pMBlkIm16[idx1 + 2] + 
		pMBlkIm16[idx2 - 2] + pMBlkIm16[idx2 - 1] + pMBlkIm16[idx2] + pMBlkIm16[idx2 + 1] + pMBlkIm16[idx2 + 2];	
/*
	int idx1 = idx + width;
	int idx2 = idx1 + width;
	int idx3 = idx2 + width;
	int idx4 = idx3 + width;
	short s = 	
		pMBlkIm16[idx ] + pMBlkIm16[idx  + 1] + pMBlkIm16[idx  + 2] + pMBlkIm16[idx  + 3] + pMBlkIm16[idx  + 4] + 
		pMBlkIm16[idx1] - pMBlkIm16[idx1 + 1] - 2 * pMBlkIm16[idx1 + 2] - pMBlkIm16[idx1 + 3] + pMBlkIm16[idx1 + 4] +
		pMBlkIm16[idx2] - 2 * pMBlkIm16[idx2 + 1] - 4 * pMBlkIm16[idx2 + 2] - 2 * pMBlkIm16[idx2 + 3] + pMBlkIm16[idx2 + 4] +
		pMBlkIm16[idx3] - pMBlkIm16[idx3 + 1] - 2 * pMBlkIm16[idx3 + 2] - pMBlkIm16[idx3 + 3] + pMBlkIm16[idx3 + 4] + 
		pMBlkIm16[idx4] + pMBlkIm16[idx4 + 1] + pMBlkIm16[idx4 + 2] + pMBlkIm16[idx4 + 3] + pMBlkIm16[idx4 + 4];
*/		
	if (s < pMBlkThresholdImage[idt]) pMBlkImBin[idx] = 0;
	else pMBlkImBin[idx] = 255;
}

__global__ void convolve_subtract_kernel(short *pMBlkIm16, unsigned char *pMBlkImSub, short *pMBlkThresholdImage, int width, int imsize)
{ 
	int idt = (blockIdx.x * blockDim.x + threadIdx.x) + 4 * width;
	int idx = idt + (blockIdx.y * blockDim.y + threadIdx.y) * imsize;
	int idxA = idx - width;
	int idxB = idxA - width;
	int idx1 = idx + width;
	int idx2 = idx1 + width;
	short s = 
		pMBlkIm16[idxB - 2] + pMBlkIm16[idxB - 1] + pMBlkIm16[idxB    ] + pMBlkIm16[idxB + 1] + pMBlkIm16[idxB + 2] + 
		pMBlkIm16[idxA - 2] - pMBlkIm16[idxA - 1] - 2 * pMBlkIm16[idxA] - pMBlkIm16[idxA + 1] + pMBlkIm16[idxA + 2] +
		pMBlkIm16[idx  - 2] - 2 * pMBlkIm16[idx - 1] - 4 * pMBlkIm16[idx] - 2 * pMBlkIm16[idx + 1] + pMBlkIm16[idx + 2] +
		pMBlkIm16[idx1 - 2] - pMBlkIm16[idx1 - 1] - 2 * pMBlkIm16[idx1] - pMBlkIm16[idx1 + 1] + pMBlkIm16[idx1 + 2] + 
		pMBlkIm16[idx2 - 2] + pMBlkIm16[idx2 - 1] + pMBlkIm16[idx2] + pMBlkIm16[idx2 + 1] + pMBlkIm16[idx2 + 2]
		-pMBlkThresholdImage[idt];
	if (s <= 0) pMBlkImSub[idx] = 0;
	else pMBlkImSub[idx] = min(255, s);
}

__global__ void binarize_kernel(unsigned char *pMBlkSubIm, int width, int imsize)
{ 
	int idt = (blockIdx.x * blockDim.x + threadIdx.x) + 4 * width;
	int idx = idt + (blockIdx.y * blockDim.y + threadIdx.y) * imsize;
	if (pMBlkSubIm[idx] <= 0) pMBlkSubIm[idx] = 0;
	else pMBlkSubIm[idx] = 255;
}

__global__ void getspotclusters_kernel(unsigned char *pMBlkSubIm, int width, int height, int *pcount, int maxclusters, IntCluster *pClusters, unsigned *pMBlkErr)
{	
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (iy < 4 || iy >= height - 4) return;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if (ix < 4 || ix >= width - 4) return;
	int im = blockIdx.z;
	pMBlkSubIm += ((((im * height) + iy) * width) + ix);
	short s = (short)(unsigned short)*pMBlkSubIm;
	if (s <= 0) return;
	if (s < (short)(unsigned short)pMBlkSubIm[-1]) return;
	short w_l0 = max(0, (short)(unsigned short)pMBlkSubIm[-1]);
	if (s < (short)(unsigned short)pMBlkSubIm[1]) return;
	short w_g0 = max(0, (short)(unsigned short)pMBlkSubIm[1]);
	if (s < (short)(unsigned short)pMBlkSubIm[-width]) return;
	short w_0l = max(0, (short)(unsigned short)pMBlkSubIm[-width]);
	if (s < (short)(unsigned short)pMBlkSubIm[width]) return;
	short w_0g = max(0, (short)(unsigned short)pMBlkSubIm[width]);
	if (s < (short)(unsigned short)pMBlkSubIm[width - 1]) return;
	short w_lg = max(0, (short)(unsigned short)pMBlkSubIm[width - 1]);
	if (s < (short)(unsigned short)pMBlkSubIm[width + 1]) return;
	short w_gg = max(0, (short)(unsigned short)pMBlkSubIm[width + 1]);
	if (s < (short)(unsigned short)pMBlkSubIm[-width - 1]) return;
	short w_ll = max(0, (short)(unsigned short)pMBlkSubIm[-width - 1]);
	if (s < (short)(unsigned short)pMBlkSubIm[-width + 1]) return;
	short w_gl = max(0, pMBlkSubIm[-width + 1]);
	int a = atomicAdd(pcount + im, 1);
	if (a >= maxclusters)
	{
		atomicExch(pcount + im, maxclusters);
		pMBlkErr[im] |= NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW;
		return;
	}
	int sum = 
		w_ll + w_0l + w_gl + 
		w_l0 + s    + w_g0 + 
		w_lg + w_0g + w_gg;
	pClusters += im * maxclusters + a;
	pClusters->X = ((int)((w_gl + w_g0 + w_gg) - (w_ll + w_l0 + w_lg)) << RESCALING_BITSHIFT) / sum + (ix << RESCALING_BITSHIFT);
	pClusters->Y = ((int)((w_lg + w_0g + w_gg) - (w_ll + w_0l + w_gl)) << RESCALING_BITSHIFT) / sum + (iy << RESCALING_BITSHIFT);
	int area = 1;
	if (w_gl > 0) area++;
	if (w_g0 > 0) area++;
	if (w_gg > 0) area++;
	if (w_0l > 0) area++;
	if (w_0g > 0) area++;
	if (w_ll > 0) area++;
	if (w_l0 > 0) area++;
	if (w_lg > 0) area++;
	pClusters->Area = area;
	pClusters->XX = pClusters->YY = 1;
	pClusters->XY = 0;
}

__global__ void reset_err(unsigned *pIm, int maxlen)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < maxlen)
		pIm[id] = 0;
}

/**********************
 *                    *
 *  End CUDA Kernels  *
 *                    *
 **********************/
 
bool CUDAManager::ProcessImagesV1(unsigned char *pim, int totalimages, void *pHostOutClusters)
{
	cudaSetDevice(DeviceId);
	if (Changed && ReconfigureMemory() == false)
		strcpy(LastError, "Memory must be configured before processing an image.");

	_DEBUG_THROW_ON_CUDAERR_
	
	int ClusterPasses = 1;	
	while ((1 << ClusterPasses) < totalimages * ImageHeight) ClusterPasses++;
		
	int totalbytesize = totalimages * (ImageWidth * ImageHeight)/*MemSize_pDevImage*/;			
	
	dim3 hrthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 hrblocks = dim3(256, 1, 1);

	dim3 hthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 hblocks = dim3(1, 1, 1);
	
	dim3 hsthreads = dim3(256, 1, 1);
	dim3 hsblocks = dim3(1, 1, 1);	
	
	dim3 ithreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 iblocks = dim3(MemSize_pDevImage / ithreads.x + 1, totalimages, 1);	
	dim3 cthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 cblocks = dim3((MemSize_pDevImage - 8 * ImageWidth) / ithreads.x + 1, totalimages, 1);	
	int roundedsize = (iblocks.x) * hthreads.x;

/*	int sizepadding = MemSize_pDevImage - roundedsize;
	dim3 ithreads_r = dim3(MemSize_pDevImage - roundedsize, 1, 1);
	dim3 iblocks_r = dim3(1, totalimages, 1);	*/
	
	dim3 segthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 segblocks = dim3((totalimages * ImageHeight) / segthreads.x + 1, 1);	
	
	dim3 imcthreads = dim3(totalimages, 1, 1);
	dim3 imcblocks = dim3(1, 1);	

	DUMP("A0 Images ", totalimages);
	DUMP("A0 Time ", (int)PreciseTimerMilliseconds());

	//_THROW_ON_CUDAERR_(cudaMemset(pDevErrorImage, 0, sizeof(unsigned) * MaxImages * 2))
	{
		dim3 resthreads = dim3(MaxThreadsPerBlock, 1, 1);
		dim3 resblocks = dim3(1 + (MaxImages * 2) / resthreads.x, 1, 1);
		reset_err<<<resblocks, resthreads>>>(pDevErrorImage, MaxImages * 2);	
	}

	DUMP("A1 Time ", (int)PreciseTimerMilliseconds());
	DUMP("A1 pDevErrorImage ", pDevErrorImage);
	DUMP("A1 Size ", sizeof(unsigned) * MaxImages * 2);

	_THROW_ON_CUDAERR_(cudaMemcpy(pDevImage, pim, totalbytesize, cudaMemcpyHostToDevice))	

	DUMP("A2 Time ", (int)PreciseTimerMilliseconds());
	DUMP("A2 pDevImage ",pDevImage);
	DUMP("A2 Size ", totalbytesize);

	historeset_kernel<<<hrblocks, hrthreads>>>(pDevHistoImage);	
	//histo_kernel<<<hblocks, hthreads>>>(pDevImage, pDevHistoImage, MemSize_pDevImage / hthreads.x);
	histo_kernel<<<hblocks, hthreads>>>(pDevImage, pDevHistoImage, MemSize_pDevImage);
	_DEBUG_THROW_ON_CUDAERR_

	DUMP("A3 Time ", (int)PreciseTimerMilliseconds());

	histosum_kernel<<<hsblocks, hsthreads>>>(pDevHistoImage, hthreads.x);	
	hmedian_kernel<<<1, 1>>>(pDevHistoImage, pDevLookupTable, roundedsize, GreyLevelTargetMedian, pDevErrorImage + MaxImages);	
	toshort_kernel<<<iblocks, ithreads>>>(pDevImage, pDevLookupTable, pDevEmptyImage, pDev16Image, MemSize_pDevImage);	
	//KRYSS _THROW_ON_CUDAERR_(cudaMemcpy(pHostErrorImage, pDevErrorImage, sizeof(unsigned) * MaxImages * 2, cudaMemcpyDeviceToHost))
/*	if (sizepadding)
		toshort_kernel<<<iblocks, ithreads>>>(pDevImage + roundedsize, pDevLookupTable, pDevEmptyImage + roundedsize, pDev16Image + roundedsize, MemSize_pDevImage);*/

	_DEBUG_THROW_ON_CUDAERR_

	DUMP("A4 Time ", (int)PreciseTimerMilliseconds());

	convolve_cut_kernel<<<cblocks, cthreads>>>(pDev16Image, pDevImage, pDevThresholdImage, ImageWidth, MemSize_pDevImage);	
	_DEBUG_THROW_ON_CUDAERR_

	DUMP("A5 Time ", (int)PreciseTimerMilliseconds());

/*	if (sizepadding)
		convolve_cut_kernel<<<iblocks, ithreads>>>(pDev16Image + roundedsize, pDevImage + roundedsize, pDevThresholdImage + roundedsize, ImageWidth, MemSize_pDevImage);*/
	if (WkCfgDumpBinImages)
		_THROW_ON_CUDAERR_(cudaMemcpy(pHostBinImage, pDevImage, totalimages * MemSize_pDevImage, cudaMemcpyDeviceToHost)) 

	DUMP("A6 Time ", (int)PreciseTimerMilliseconds());

	if (WkCfgDumpSegments || WkCfgDumpClusters || WkCfgDumpClusters2ndMomenta) 
	{
		segments_kernel<<<segblocks, segthreads>>>(pDevImage, pDevSegmentImage, pDevSegmentCountImage, ImageWidth, MaxSegmentsPerScanLine, totalimages * ImageHeight, ImageHeight, pDevErrorImage);
		_DEBUG_THROW_ON_CUDAERR_

		DUMP("A7 Time ", (int)PreciseTimerMilliseconds());

		if (WkCfgDumpSegments)
		{
			_THROW_ON_CUDAERR_(cudaMemcpy(pHostSegmentImage, pDevSegmentImage, totalimages * MaxSegmentsPerScanLine * ImageHeight * sizeof(IntSegment), cudaMemcpyDeviceToHost))

			DUMP("A8 Time ", (int)PreciseTimerMilliseconds());

			_THROW_ON_CUDAERR_(cudaMemcpy(pHostSegmentCountImage, pDevSegmentCountImage, totalimages * ImageHeight * sizeof(short), cudaMemcpyDeviceToHost))

			DUMP("A9 Time ", (int)PreciseTimerMilliseconds());
		}			
	}
	if (WkCfgDumpClusters || WkCfgDumpClusters2ndMomenta)
	{
		if (WkCfgDumpClusters2ndMomenta) 
		{
			resetclusterline_m2_kernel<<<segblocks, segthreads>>>(pDevClusterWorkImage, pDevClusterWorkCountImage, pDevSegmentImage, pDevSegmentCountImage, MaxSegmentsPerScanLine, ImageHeight, totalimages);
			_DEBUG_THROW_ON_CUDAERR_

			DUMP("A10 Time ", (int)PreciseTimerMilliseconds());
		}
		else 
		{
			resetclusterline_kernel<<<segblocks, segthreads>>>(pDevClusterWorkImage, pDevClusterWorkCountImage, pDevSegmentImage, pDevSegmentCountImage, MaxSegmentsPerScanLine, ImageHeight, totalimages);
			_DEBUG_THROW_ON_CUDAERR_

			DUMP("A11 Time ", (int)PreciseTimerMilliseconds());
		}
		int p = 0;
		for (p = 0; p < ClusterPasses; p++)
		{
			int workers = (int)ceil((totalimages * ImageHeight) / (double)(2 << p));
			dim3 clsthreads = dim3((workers >= MaxThreadsPerBlock) ? MaxThreadsPerBlock : workers, 1, 1);
			dim3 clsblocks = dim3((int)ceil(workers / (double)clsthreads.x), 1);
			if (WkCfgDumpClusters2ndMomenta) 
			{
				getclustersline_m2_kernel<<<clsblocks, clsthreads>>>(p, pDevSegmentImage, pDevSegmentCountImage, MaxSegmentsPerScanLine, ImageHeight, totalimages);
			}
			else 
			{
				getclustersline_kernel<<<clsblocks, clsthreads>>>(p, pDevSegmentImage, pDevSegmentCountImage, MaxSegmentsPerScanLine, ImageHeight, totalimages);
			}
		}
		_DEBUG_THROW_ON_CUDAERR_

		DUMP("A12 Time ", (int)PreciseTimerMilliseconds());

		countclusters_kernel<<<segblocks, segthreads>>>(pDevSegmentImage, pDevClusterWorkCountImage, pDevSegmentCountImage, MaxSegmentsPerScanLine, ImageHeight, totalimages);		
		setclusterbase_kernel<<<imcblocks, imcthreads>>>(pDevClusterCountImage, pDevClusterWorkCountImage, pDevSegmentCountImage, pDevClusterBaseImage, ImageHeight, MaxClustersPerImage, pDevErrorImage);
		if (WkCfgDumpClusters2ndMomenta) 
		{
			finalizeclusters_m2_kernel<<<segblocks, segthreads>>>(pDevClusterImage, pDevSegmentImage, pDevClusterWorkCountImage, pDevSegmentCountImage, pDevClusterBaseImage, MaxSegmentsPerScanLine, ImageHeight, MaxClustersPerImage, totalimages);
			_DEBUG_THROW_ON_CUDAERR_

			DUMP("A13 Time ", (int)PreciseTimerMilliseconds());
		}
		else 
		{
			finalizeclusters_kernel<<<segblocks, segthreads>>>(pDevClusterImage, pDevSegmentImage, pDevClusterWorkCountImage, pDevSegmentCountImage, pDevClusterBaseImage, MaxSegmentsPerScanLine, ImageHeight, MaxClustersPerImage, totalimages);
			_DEBUG_THROW_ON_CUDAERR_

			DUMP("A14 Time ", (int)PreciseTimerMilliseconds());
		}
		int *pHostOutI = (int *)pHostOutClusters;
		pHostOutI[0] = totalimages;
		pHostOutI[1] = RESCALING_FACTOR;
		short *pHostOutS = (short *)(pHostOutI + 2);
		pHostOutS[0] = ImageWidth;
		pHostOutS[1] = ImageHeight;
		pHostOutS[2] = pHostOutS[3] = pHostOutS[4] = pHostOutS[5] = pHostOutS[6] = pHostOutS[7] = pHostOutS[8] = pHostOutS[9] = 0;
		double *pHostOutD = (double *)(void *)(pHostOutS + 10);
		int i;
		for (i = 0; i < 2 + 3 * totalimages; i++) pHostOutD[i] = 0.0;		
		int *pHICC = (int *)(void *)(pHostOutD + 2 + 3 * totalimages);		
		_THROW_ON_CUDAERR_(cudaMemcpy(pHICC, pDevClusterCountImage, totalimages * sizeof(int), cudaMemcpyDeviceToHost))

		DUMP("A15 Time ", (int)PreciseTimerMilliseconds());

		IntCluster *pHC = (IntCluster *)(void *)(pHICC + totalimages);
		for (i = 0; i < totalimages; i++)
		{
			_THROW_ON_CUDAERR_(cudaMemcpy(pHC, pDevClusterImage + i * MaxClustersPerImage, sizeof(IntCluster) * pHICC[i], cudaMemcpyDeviceToHost))
			pHC += pHICC[i];
		}
		DUMP("A16 Time ", (int)PreciseTimerMilliseconds());
	}	
	_THROW_ON_CUDAERR_(cudaMemcpy(pHostErrorImage, pDevErrorImage, sizeof(unsigned) * MaxImages * 2, cudaMemcpyDeviceToHost))
	
	_DEBUG_THROW_ON_CUDAERR_
	
	DUMP("A17 Time ", (int)PreciseTimerMilliseconds());
	
	return true;
}

bool CUDAManager::ProcessImagesV2(unsigned char *pim, int totalimages, void *pHostOutClusters)
{
	cudaSetDevice(DeviceId);
	if (Changed && ReconfigureMemory() == false)
		strcpy(LastError, "Memory must be configured before processing an image.");
	
	int totalbytesize = totalimages * (ImageWidth * ImageHeight)/*MemSize_pDevImage*/;			
	
	dim3 hrthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 hrblocks = dim3(256, 1, 1);

	dim3 hthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 hblocks = dim3(1, 1, 1);
	
	dim3 hsthreads = dim3(256, 1, 1);
	dim3 hsblocks = dim3(1, 1, 1);	
	
	dim3 ithreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 iblocks = dim3((int)ceil(MemSize_pDevImage / (double)ithreads.x), totalimages, 1);	
	dim3 cthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 cblocks = dim3((int)ceil((MemSize_pDevImage - 8 * ImageWidth) / (double)ithreads.x), totalimages, 1);	
	int roundedsize = (iblocks.x) * hthreads.x;

/*	int sizepadding = MemSize_pDevImage - roundedsize;
	dim3 ithreads_r = dim3(MemSize_pDevImage - roundedsize, 1, 1);
	dim3 iblocks_r = dim3(1, totalimages, 1);	*/
	
	dim3 segthreads = dim3(MaxThreadsPerBlock, 1, 1);
	dim3 segblocks = dim3((int)ceil((float)(totalimages * ImageHeight) / (float)segthreads.x), 1);	
	
	dim3 imcthreads = dim3(totalimages, 1, 1);
	dim3 imcblocks = dim3(1, 1);	

	_THROW_ON_CUDAERR_(cudaMemset(pDevErrorImage, 0, sizeof(unsigned) * MaxImages * 2))	
	_THROW_ON_CUDAERR_(cudaMemcpy(pDevImage, pim, totalbytesize, cudaMemcpyHostToDevice))	
	_THROW_ON_CUDAERR_(cudaMemset(pDevClusterCountImage, 0, sizeof(int) * totalimages))

	historeset_kernel<<<hrblocks, hrthreads>>>(pDevHistoImage);
	_DEBUG_THROW_ON_CUDAERR_
	histo_kernel<<<hblocks, hthreads>>>(pDevImage, pDevHistoImage, MemSize_pDevImage / hthreads.x);
	_DEBUG_THROW_ON_CUDAERR_
	histosum_kernel<<<hsblocks, hsthreads>>>(pDevHistoImage, hthreads.x);
	_DEBUG_THROW_ON_CUDAERR_
	hmedian_kernel<<<1, 1>>>(pDevHistoImage, pDevLookupTable, roundedsize, GreyLevelTargetMedian, pDevErrorImage + MaxImages);
	_DEBUG_THROW_ON_CUDAERR_
	toshort_kernel<<<iblocks, ithreads>>>(pDevImage, pDevLookupTable, pDevEmptyImage, pDev16Image, MemSize_pDevImage);
	_DEBUG_THROW_ON_CUDAERR_
	//KRYSS _THROW_ON_CUDAERR_(cudaMemcpy(pHostErrorImage, pDevErrorImage, sizeof(unsigned) * MaxImages * 2, cudaMemcpyDeviceToHost))
/*	if (sizepadding)
		toshort_kernel<<<iblocks, ithreads>>>(pDevImage + roundedsize, pDevLookupTable, pDevEmptyImage + roundedsize, pDev16Image + roundedsize, MemSize_pDevImage);*/

	convolve_subtract_kernel<<<cblocks, cthreads>>>(pDev16Image, pDevImage, pDevThresholdImage, ImageWidth, MemSize_pDevImage);
	_DEBUG_THROW_ON_CUDAERR_

/*	if (sizepadding)
		convolve_cut_kernel<<<iblocks, ithreads>>>(pDev16Image + roundedsize, pDevImage + roundedsize, pDevThresholdImage + roundedsize, ImageWidth, MemSize_pDevImage);*/

	if (WkCfgDumpSegments) 
	{
		memset(pHostSegmentCountImage, 0, sizeof(short) * totalimages * ImageHeight);
	}
	if (WkCfgDumpClusters || WkCfgDumpClusters2ndMomenta)
	{				
		dim3 clsthreads = dim3(32, MaxThreadsPerBlock / 32, 1);
		dim3 clsblocks = dim3(ImageWidth / clsthreads.x, ImageHeight / clsthreads.y, totalimages);		
		getspotclusters_kernel<<<clsblocks, clsthreads>>>(pDevImage, ImageWidth, ImageHeight, pDevClusterCountImage, MaxClustersPerImage, pDevClusterImage, pDevErrorImage);		
		_DEBUG_THROW_ON_CUDAERR_
		int *pHostOutI = (int *)pHostOutClusters;
		pHostOutI[0] = totalimages;
		pHostOutI[1] = RESCALING_FACTOR;
		short *pHostOutS = (short *)(pHostOutI + 2);
		pHostOutS[0] = ImageWidth;
		pHostOutS[1] = ImageHeight;
		pHostOutS[2] = pHostOutS[3] = pHostOutS[4] = pHostOutS[5] = pHostOutS[6] = pHostOutS[7] = pHostOutS[8] = pHostOutS[9] = 0;
		double *pHostOutD = (double *)(void *)(pHostOutS + 10);
		int i;
		for (i = 0; i < 2 + 3 * totalimages; i++) pHostOutD[i] = 0.0;		
		int *pHICC = (int *)(void *)(pHostOutD + 2 + 3 * totalimages);		
		_THROW_ON_CUDAERR_(cudaMemcpy(pHICC, pDevClusterCountImage, totalimages * sizeof(int), cudaMemcpyDeviceToHost))
		IntCluster *pHC = (IntCluster *)(void *)(pHICC + totalimages);
		for (i = 0; i < totalimages; i++)
		{
			_THROW_ON_CUDAERR_(cudaMemcpy(pHC, pDevClusterImage + i * MaxClustersPerImage, sizeof(IntCluster) * pHICC[i], cudaMemcpyDeviceToHost))
			pHC += pHICC[i];
		}
	}	
	_THROW_ON_CUDAERR_(cudaMemcpy(pHostErrorImage, pDevErrorImage, sizeof(unsigned) * MaxImages * 2, cudaMemcpyDeviceToHost))
	if (WkCfgDumpBinImages)
	{
		binarize_kernel<<<cblocks, cthreads>>>(pDevImage, ImageWidth, MemSize_pDevImage);
		_THROW_ON_CUDAERR_(cudaMemcpy(pHostBinImage, pDevImage, totalimages * MemSize_pDevImage, cudaMemcpyDeviceToHost)) 
	}

		
	_THROW_ON_CUDAERR_(cudaDeviceSynchronize())
	return true;
}