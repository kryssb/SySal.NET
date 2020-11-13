#ifndef _SYSAL_GPU_TRACK_KERNELS_H_
#define _SYSAL_GPU_TRACK_KERNELS_H_

#include "gpu_defines.h"
#include "Tracker.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace SySal { namespace GPU {

__global__ void reset_scheduler_kernel(int *pScheduler, int value, int setblocks);

__global__ void explore_skewchainmap_kernel(ChainView *pView1, ChainView *pView2, int width, int height, InternalInfo *pI, ChainView **ppViewEntryPoints, TrackMapHeader *pTH);

__global__ void fill_skewhashtable1view_list_kernel(int *pbinfill, ChainView **ppViewEntryPoints, HashTableBounds *pHashTableBounds, int skewx, int skewy, int minchainvolum, int view);

__global__ void compact_scheduler_kernel(int *pNewScheduler, int *pScheduler);

__global__ void recursive_sum_kernel(int *parrayin, int *parrayout, int insize);

__global__ void compute_pairs1v_kernel(int *pinchains, int *pmintag, int mintag, int *poutpairs, int size);

__global__ void pair_find1v_kernel(int *parrayind, int depth, int *parraycountchains, int *pbinfill, ChainView **ppViewEntryPoints, int *poutchainindices);

__global__ void find_track_singlepass_kernel(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag);

__global__ void mergetracks_prepare(int *pTBinFill, short xbins, short ybins, int *pcount);

__global__ void mergetracks_split_and_index_kernel(int *paircomputer, int depth, int *pairindices, int totalpairs/*, int idz*/);

__global__ void mergetracks_mapindex_kernel(int *pIndex, int *pTBinFill, short xbins, short ybins, short binsize, int totalpairs/*, int idz*/);

__global__ void mergetracks_kernel(int *pIndex, TempIntTrack *pTBins, int xytol, int ztol, int totalpairs/*, int idz*/);

__global__ void mergetracks_kernel(int *pIndex, int *pTBinFill, short xbins, short ybins, TempIntTrack *pTBins, int xytol, int ztol, short binsize);

__global__ void mergetracks_kernel(int *pCountTempTracks, TempIntTrack **ppTempTracks, int *pTBinFill, TempIntTrack *pTBins, TrackMapHeader *pTrackMapHdr, int xytol, int ztol, short xbins, short ybins, short binsize, int *pOffset, int *pTerminate);

__global__ void filltracks_kernel(int *pCountTempTracks, TempIntTrack **ppTempTracks, TrackMapHeader *pTrackMapHdr);

__global__ void _debug_track_pattern_(ChainView *pView, int chainspertrack, int slope_step_scaled, int xystep, int steps, int deltaz);

}}

#endif