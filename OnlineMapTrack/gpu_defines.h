#ifndef _SYSAL_GPU_DEFINES_H_
#define _SYSAL_GPU_DEFINES_H_

#include "Tracker.h"
#include <string.h>
#include <stdio.h>

#define DEMAG_SHIFT 20

#define XY_CURVATURE_SHIFT 15
#define Z_CURVATURE_SHIFT 15

#define XY_MAGNIFICATION_SHIFT 25

#define Z_SCALE_SHIFT 6
#define XY_SCALE_SHIFT 10

#define SLOPE_SHIFT 10
#define SLOPE_SCALE_SHIFT 10

#define FRACT_RESCALE_SHIFT 20

#define Z_TO_XY_RESCALE_SHIFT (XY_SCALE_SHIFT - Z_SCALE_SHIFT)
#define SQUARE_RESCALE 4
#define FRACT_RESCALE 8

#define Z_POSITIVE_RESCALE 100000

#define _MAX_GRAINS_PER_TRACK_ 32


#define CTOR_INIT(ptr) ptr(0), _MEM_(ptr)(0)

#define DEALLOC(ptr) { if (ptr) { cudaFree(ptr); ptr = 0; _MEM_(ptr) = 0; } }

#define HOST_DEALLOC(ptr) { if (ptr) { free(ptr); ptr = 0; _MEM_(ptr) = 0; } }

#define WISE_MEMSIZE(m) (((int)(m * 1.2 + 8)) & 0xfffffff8 )

#define _CUDA_THROW_ERR_
#define _CUDA_THROW_ERR_ { cudaDeviceSynchronize(); if (err = cudaGetLastError()) { sprintf(pThis->LastError, "Error \"%s\" at file %s, line %d", cudaGetErrorString(err), __FILE__, __LINE__); throw pThis->LastError; } }


#define THROW_ON_CUDA_ERR(x) \
{ \
	if (err = x)\
	{\
		sprintf_s(pThis->LastError, 512, "File %s Line %d Error: %s", __FILE__, __LINE__, cudaGetErrorString(err));\
		throw pThis->LastError;\
	}\
}

#define SHOW_ALLOC(ptr, req) { printf("\nGPU Ptr %s %016X Req %d Mem %d", # ptr, ptr, req, _MEM_(ptr)); }
#define HOST_SHOW_ALLOC(ptr) { printf("\nHost Ptr %s %016X Mem %d", # ptr, ptr, _MEM_(ptr)); }

#define EXACT_ALLOC(ptr, m) \
{\
	if (_MEM_(ptr) < m) \
	{\
		DEALLOC(ptr); \
		cudaError_t err; \
		if (err = cudaMalloc((void **)&ptr, m)) \
		{\
			strcpy(pThis->LastError, cudaGetErrorString(err)); \
			throw pThis->LastError;	\
		}\
		_MEM_(ptr) = m; \
		SHOW_ALLOC(ptr, m) \
	}\
}

#define WISE_ALLOC(ptr, m) \
{\
	if (_MEM_(ptr) < m) \
	{\
		DEALLOC(ptr); \
		cudaError_t err; \
		if (err = cudaMalloc((void **)&ptr, WISE_MEMSIZE(m))) \
		{\
			sprintf(pThis->LastError, "Error allocating %d bytes at file %s line %d: %s", WISE_MEMSIZE(m), __FILE__, __LINE__, cudaGetErrorString(err)); \
			throw pThis->LastError;	\
		}\
		_MEM_(ptr) = WISE_MEMSIZE(m); \
		SHOW_ALLOC(ptr, m) \
	}\
}

#define HOST_WISE_ALLOC(ptr, m) \
{\
	if (_MEM_(ptr) < m) \
	{\
		HOST_DEALLOC(ptr); \
		void *_p = realloc(ptr, WISE_MEMSIZE(m)); \
		if (_p) \
		{\
			*((void **)&ptr) = _p; \
			_MEM_(ptr) = WISE_MEMSIZE(m); \
			HOST_SHOW_ALLOC(ptr) \
		}\
	}\
}

namespace SySal { namespace GPU {

struct Cell
{
	int Count;
};

struct ChainMapWindow
{
	int MinX;
	int MaxX;
	int MinY;
	int MaxY;
	int Width;
	int Height;
	int CellSize;
	int MaxCellContent;
	int NXCells;
	int NYCells;
	Cell *pCells;
	IntChain **pChains;
};

struct HashTableBounds
{
	int MinX;
	int MaxX;
	int MinY;
	int MaxY;
	int MinZ;
	int MaxZ;
	int XYBinSize;
	int ZBinSize;
	int XBins;
	int YBins;
	int ZBins;
	int NBins;
	int XTBins;
	int YTBins;
	int NTBins;
	int BinCapacity;
	int TBinCapacity;
	int DEBUG1, DEBUG2, DEBUG3, DEBUG4;
};

struct InternalInfo
{
	::SySal::Tracker::Configuration C;
	HashTableBounds H;
	int MinDist2;
	int MaxDist2;
};

struct TempIntTrack : public IntTrack
{
	int DX;
	int DY;
	int DZ;
	int D2;
	int MapsTo;
	int IC;
	short IX, IY;
	short IIX, IIY;	
};

struct _segmented_findtrack_kernel_status_
{
	int izA, iA, izB, iB;
	IntChain *pA, *pB;
	int dx, dx_sqresc, dy, dy_sqresc, dz, dz_sqresc, d2, d2_fracresc;
	int Ax, Ay, Az;
	int binX, binY;
	int binA;
	int binB;
	int Run;
	int SkipFirst;
	int SearchGrains;
	int Chains;
	int Volume;
	int x1,y1,z1,x2,y2,z2;
	int Clusters;
	int MinViewTag;	
	IntChain *ppTrackGrains[_MAX_GRAINS_PER_TRACK_];
};

}}

#endif