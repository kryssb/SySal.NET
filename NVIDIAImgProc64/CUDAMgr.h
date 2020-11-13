#ifndef _CUDA_MANAGER_H_
#define _CUDA_MANAGER_H_

#define CUDAMGR_STRLEN 1024

#define RESCALING_BITSHIFT 3
#define RESCALING_FACTOR (1 << RESCALING_BITSHIFT)

struct IntClusterW
{
	int Area;
	unsigned XSum;
	unsigned YSum;	
	int Flag;
	long long XXSum;
	long long XYSum;
	long long YYSum;	
	IntClusterW *pMergeTo;
};

struct IntCluster
{
	int Area;
	unsigned short X;
	unsigned short Y;	
	unsigned XX;
	unsigned XY;
	unsigned YY;
};

struct IntSegment
{
	int Left;
	int Right;
	IntClusterW *pCluster;

	int Area;
	unsigned XSum;
	unsigned YSum;	
	int Flag;
	long long XXSum;
	long long XYSum;
	long long YYSum;	
	IntSegment *pMergeTo;

};

#define NVIDIAIMGPROC_ERR_SEGMENT_OVERFLOW 0x08
#define NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW 0x10

struct CUDAManager
{
	char LastError[CUDAMGR_STRLEN + 512];
	int DeviceId;
	char DeviceName[CUDAMGR_STRLEN];	
	unsigned long long /*size_t*/ TotalMemory;
	unsigned long long /*size_t*/ AvailableMemory;
	int MaxThreadsPerBlock;
	int ImageWidth;
	int ImageHeight;

	bool Changed;

	unsigned char *pDevImage;
	unsigned char *pHostEqImage;
	unsigned char *pHostBinImage;
	unsigned *pDevHistoImage;
	short *pDevLookupTable;
	short *pDev16Image;
	short *pDevThresholdImage;
	short *pDevEmptyImage;
	short *pDevSegmentCountImage;
	IntSegment *pDevSegmentImage;
	IntSegment *pHostSegmentImage;
	short *pHostSegmentCountImage;
	IntClusterW *pDevClusterWorkImage;
	short *pDevClusterWorkCountImage;
	int *pDevClusterBaseImage;
	IntCluster *pDevClusterImage;
	int *pDevClusterCountImage;
	IntCluster *pHostClusterImage;
	int *pHostClusterCountImage;
	unsigned *pDevErrorImage;
	unsigned *pHostErrorImage;

	unsigned long long MemSize_pDevImage;
	unsigned long long MemSize_pHostEqImage;
	unsigned long long MemSize_pHostBinImage;
	unsigned long long MemSize_pDevHistoImage;
	unsigned long long MemSize_pDevLookupTable;
	unsigned long long MemSize_pDev16Image;
	unsigned long long MemSize_pDevThresholdImage;
	unsigned long long MemSize_pDevEmptyImage;
	unsigned long long MemSize_pDevSegmentCountImage;
	unsigned long long MemSize_pDevSegmentImage;
	unsigned long long MemSize_pHostSegmentImage;
	unsigned long long MemSize_pHostSegmentCountImage;
	unsigned long long MemSize_pDevClusterWorkImage;
	unsigned long long MemSize_pDevClusterWorkCountImage;
	unsigned long long MemSize_pDevClusterBaseImage;
	unsigned long long MemSize_pDevClusterImage;
	unsigned long long MemSize_pDevClusterCountImage;
	unsigned long long MemSize_pHostClusterImage;
	unsigned long long MemSize_pHostClusterCountImage;
	unsigned long long MemSize_pDevErrorImage;
	unsigned long long MemSize_pHostErrorImage;

	int MaxImages;

	bool WkCfgDumpEqImages;
	bool WkCfgDumpBinImages;
	bool WkCfgDumpSegments;
	bool WkCfgDumpClusters;
	bool WkCfgDumpClusters2ndMomenta;

	unsigned MaxSegmentsPerScanLine;
	unsigned MaxClustersPerImage;
	unsigned char GreyLevelTargetMedian;

	CUDAManager(int devid = -1);
	~CUDAManager();
	bool ReconfigureMemory();
	bool SetEmptyImage(unsigned short *);
	bool SetThresholdImage(short *);
	void SetWorkingConfiguration(bool dumpeq, bool dumpbin, bool dumpsegs, bool dumpclusters, bool dumpclusters2ndmom);
	bool ProcessImagesV1(unsigned char *pim, int totalimages, void *pHostOutClusters);
	bool ProcessImagesV2(unsigned char *pim, int totalimages, void *pHostOutClusters);

	static void *MallocHost(long size);
	static void FreeHost(void *pbuff);

	static int GetDeviceCount();

	static long PreciseTimerMilliseconds();
};

#endif