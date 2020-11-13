#ifndef _SYSAL_GPU_TRACK_KERNELS_H_
#define _SYSAL_GPU_TRACK_KERNELS_H_

#include "gpu_defines.h"
#include "Tracker.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace SySal { namespace GPU {

__global__ void reset_scheduler_kernel(int *pScheduler, int value, int setblocks);

__global__ void explore_skewchainmap_kernel(ChainView *pView1, ChainView *pView2, int width, int height, InternalInfo *pI, ChainView **ppViewEntryPoints, TrackMapHeader *pTH);

__global__ void fill_skewhashtable_list_kernel(int *pbinfill, ChainView **ppViewEntryPoints, HashTableBounds *pHashTableBounds, int skewx, int skewy, int minchainvolume);

__global__ void fill_skewhashtable1view_list_kernel(int *pbinfill, ChainView **ppViewEntryPoints, HashTableBounds *pHashTableBounds, int skewx, int skewy, int minchainvolum, int view);

__global__ void find_tracks_skewreset_list_kernel(_segmented_findtrack_kernel_status_ *pstat, int *qBinFill, ChainView **ppViewEntryPoints, int xbins, int ybins, short binsize, int *pScheduler);

__global__ void find_tracks_skewslope_list_kernel(_segmented_findtrack_kernel_status_ *pstat, ChainView **ppViewEntryPoints, short xbins, short ybins, short binsize, int slopex, int slopey, int slopeaccx, int slopeaccy, int minlength, int *pScheduler);

__global__ void find_tracks_skewincrement_list_kernel(_segmented_findtrack_kernel_status_ *pstat, int *qBinFill, ChainView **ppViewEntryPoints, short xbins, short ybins, int binsize, int *pScheduler, int minlength, int *pNewScheduler);

__global__ void compact_scheduler_kernel(int *pNewScheduler, int *pScheduler);

__global__ void recursive_sum_kernel(int *parrayin, int *parrayout, int insize);

/* recursive compactor */

__global__ void reset_compactor_kernel(int *parray, int *parraylen, int *psize);

__global__ void compactor_find_kernel(int *parrayind, int depth, int *parrayin, int *parrayout, int *poutlength);

/* end recursive compactor */

/* non-iterative pair computer */

//__global__ void compute_pairs_kernel(int *pinchains, int *pinclusters, int minclusters, int *poutpairs, int size);
//__global__ void compute_pairs_kernel(int *pinchains, int *poutpairs, int size);
__global__ void compute_pairs_kernel(int *pinchains, int *pmintag, int mintag, int *poutpairs, int size);

__global__ void compute_pairs1v_kernel(int *pinchains, int *pmintag, int mintag, int *poutpairs, int size);

__global__ void pair_find_kernel(int *parrayind, int depth, int *parraycountchains, int *pbinfill, ChainView **ppViewEntryPoints, int *poutchainindices);

__global__ void pair_find1v_kernel(int *parrayind, int depth, int *parraycountchains, int *pbinfill, ChainView **ppViewEntryPoints, int *poutchainindices);

__global__ void find_track_singlepass_kernel(int *pindices, int totalpairs, ChainView **ppViewEntryPoints, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, int slopex, int slopey, int slopeaccx, int slopeaccy, HashTableBounds *pH, int minviewtag);

/* end non-iterative pair computer */

__global__ void find_tracks_skewgrainseek_list_kernel(_segmented_findtrack_kernel_status_ *pstat, ChainView **ppViewEntryPoints, int *qBinFill, short xbins, short ybins, short binsize, int xytol, int ztol,/* short ix, short iy, */int *pScheduler);

__global__ void find_tracks_skewchecktrack_kernel(_segmented_findtrack_kernel_status_ *pstat, Tracker::Configuration *pC, TempIntTrack *pTBins, int *pTBinFill, TempIntTrack **ppTracks, int *pTrackCounter, float vs, float cm, short xbins, short ybins, int binsize, int minviewtag, int *pScheduler);

__global__ void mergetracks_kernel(int *pCountTempTracks, TempIntTrack **ppTempTracks, int *pTBinFill, TempIntTrack *pTBins, TrackMapHeader *pTrackMapHdr, int xytol, int ztol, short xbins, short ybins, short binsize, int *pOffset, int *pTerminate);

__global__ void filltracks_kernel(int *pCountTempTracks, TempIntTrack **ppTempTracks, TrackMapHeader *pTrackMapHdr);

}}

#endif