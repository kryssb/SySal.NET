#include "gpu_incremental_map_track.h"
#include "gpu_track_kernels.h"
#include "gpu_defines.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <memory.h>
#include <malloc.h>


namespace SySal
{
	namespace GPU
	{

		void PrismMapTracker::Tracker::HardReset()
		{
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))
			
			cudaDeviceReset();

			HOST_DEALLOC(pHostTracks);

			HARD_RESET(ppTempTracks); 
			HARD_RESET(pTBins);
			HARD_RESET(pCountTempTracks);
			HARD_RESET(pTBinFill);
			HARD_RESET(ppBins);
			HARD_RESET(pBinFill);
			HARD_RESET(pPairIndices);
			HARD_RESET(pPairComputer);
			HARD_RESET(pScheduler);
			HARD_RESET(pTrackGrains);
			HARD_RESET(pKernelContinue);
			HARD_RESET(pHashTable);
			HARD_RESET(ppChainMapViewEntryPoints);
			HARD_RESET(pInternalInfo);
			HARD_RESET(pChainMapHeader);
			HARD_RESET(pTracks);
		}

		PrismMapTracker::Tracker::Tracker() :
			CTOR_INIT(pTracks), 
			CTOR_INIT(pHostTracks),
			CTOR_INIT(pChainMapHeader),
			CTOR_INIT(pInternalInfo),
			CTOR_INIT(ppChainMapViewEntryPoints),
			CTOR_INIT(pHashTable),
			CTOR_INIT(pKernelContinue),
			CTOR_INIT(pTrackGrains),
			CTOR_INIT(pScheduler),
			CTOR_INIT(pPairComputer),
			CTOR_INIT(pPairIndices),
			CTOR_INIT(pBinFill),
			CTOR_INIT(ppBins),
			CTOR_INIT(pTBinFill),
			CTOR_INIT(pCountTempTracks),
			CTOR_INIT(pTBins),
			CTOR_INIT(ppTempTracks),
			_MergeTracksKernel_LoopLimiter_(0x7fffffff)
		{
			C.XYTolerance = 0.23 * GetXYScale();
			C.ZTolerance = 1.0 * GetZScale();
			C.ZThickness = 50 * GetZScale();
			C.HashBinCapacity = 6;
			C.XYHashTableBinSize = 20 * GetXYScale();
			C.ZHashTableBinSize = 2 * GetZScale();
			C.ZHashTableBins = 25;
			C.MinLength = 20 * GetXYScale();
			C.MaxLength = 60 * GetXYScale();
			C.MaxTracks = 1000;	
			C.MinVolume = 30;
			C.MinChainVolume = 10;
			C.SlopeCenterX = 0;
			C.SlopeCenterY = 0;
			C.SlopeAcceptanceX = 0;
			C.SlopeAcceptanceY = 0;
			C.FilterVolumeLength0 = 40;
			C.FilterVolumeLength100 = 140;
			C.FilterChain0 = 4;
			C.FilterChainMult = 4.0f;
			C.MergeTrackCell = 150 * GetXYScale();
			C.MergeTrackXYTolerance = 2 * GetXYScale();
			C.MergeTrackZTolerance = 3 * GetZScale();
			C.ClusterVol0 = 220;
			C.ClusterVolM = 10;

		}

		PrismMapTracker::Tracker::~Tracker()
		{
			DEALLOC(ppTempTracks);
			DEALLOC(pTBins);
			DEALLOC(pCountTempTracks);
			DEALLOC(pTBinFill);
			DEALLOC(ppBins);
			DEALLOC(pBinFill);
			DEALLOC(pPairIndices);
			DEALLOC(pPairComputer);
			DEALLOC(pScheduler);
			DEALLOC(pTrackGrains);
			DEALLOC(pKernelContinue);
			DEALLOC(pHashTable);
			DEALLOC(ppChainMapViewEntryPoints);
			DEALLOC(pInternalInfo);
			DEALLOC(pChainMapHeader);
			DEALLOC(pTracks);
			HOST_DEALLOC(pHostTracks);

		}

		void PrismMapTracker::Tracker::Reset(SySal::Tracker::Configuration &c)
		{
			cudaError_t err;
			C = c;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId));
			HOST_WISE_ALLOC(pHostTracks, sizeof(TrackMapHeader));
			pHostTracks->Count = 0;
			pHostTracks->TotalGrains = 0;
			memset(pHostTracks->Reserved, 0, sizeof(short) * 8);
			WISE_ALLOC(pTracks, sizeof(TrackMapHeader) + C.MaxTracks * sizeof(IntTrack));
			//printf("\nDEBUG DAMN %016X %d %016X %d %d", pTracks, _MEM_(pTracks), pHostTracks, sizeof(TrackMapHeader), C.MaxTracks);
			THROW_ON_CUDA_ERR(cudaMemcpy(pTracks, pHostTracks, sizeof(TrackMapHeader), cudaMemcpyHostToDevice));
			EXACT_ALLOC(pKernelContinue, sizeof(int));
			EXACT_ALLOC(pInternalInfo, sizeof(InternalInfo));
			EXACT_ALLOC(pCountTempTracks, sizeof(int));
			EXACT_ALLOC(ppChainMapViewEntryPoints, sizeof(ChainView *) * 2);
			THROW_ON_CUDA_ERR(cudaMemcpy(&pInternalInfo->C, &C, sizeof(C), cudaMemcpyHostToDevice));
		}

		void PrismMapTracker::Tracker::InternalFindTracks(int minviewtag, int width, int height, ChainView *pLastView, ChainView *pThisView)
		{
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId));
			HashTableBounds hostHashTableBounds;						
			WISE_ALLOC(pTracks, sizeof(TrackMapHeader) + C.MaxTracks * sizeof(IntTrack));
			/* We explore the ChainMap here and define the bounds. */
			{
#ifdef DEBUG_GENERATE_PATTERN
				{
					int xy_steps = 11;
					int chains = 12;
					int totalchains = 0;
					cudaMemcpy(&totalchains, &pThisView->Count, sizeof(int), cudaMemcpyDeviceToHost);
					dim3 dthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 dblocks((totalchains / dthreads.x) + 1, 1, 1);
					_debug_track_pattern_<<<dblocks, dthreads>>>(pThisView, chains, 128, 50 << XY_SCALE_SHIFT, xy_steps, 3 << Z_SCALE_SHIFT);
					_CUDA_THROW_ERR_
				}
#endif
				dim3 ithreads(1,1,1);
				dim3 iblocks(1,1,1);
				explore_skewchainmap_kernel<<<iblocks, ithreads>>>((minviewtag > 0) ? pLastView : 0, pThisView, width, height, pInternalInfo, ppChainMapViewEntryPoints, pTracks);
			}

			THROW_ON_CUDA_ERR(cudaMemcpy(&hostHashTableBounds, &pInternalInfo->H, sizeof(HashTableBounds), cudaMemcpyDeviceToHost));
			//printf("\nDEBUG DUMP 1 %d", hostHashTableBounds.DEBUG1);

			if (C.MaxTracks < hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity)
			{
				C.MaxTracks = hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity;
				printf("\nMaxTracks corrected to %d, TBinCapacity corrected to %d, NTBins was %d\nMinX %d MaxX %d\nMinY %d MaxY %d\nMinZ %d MaxZ %d", C.MaxTracks, hostHashTableBounds.TBinCapacity, hostHashTableBounds.NTBins,
					hostHashTableBounds.MinX, hostHashTableBounds.MaxX, hostHashTableBounds.MinY, hostHashTableBounds.MaxY, hostHashTableBounds.MinZ, hostHashTableBounds.MaxZ
				);
			}

			//printf("\nHashTable Grid: %d %d %d (%d)\n%d %d %d", hostHashTableBounds.XBins, hostHashTableBounds.YBins, hostHashTableBounds.ZBins, hostHashTableBounds.NBins, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins, hostHashTableBounds.TBinCapacity);

//			WISE_ALLOC(pTrackGrains, sizeof(IntChain) * C.MaxTracks * _MAX_GRAINS_PER_TRACK_);


			WISE_ALLOC(pBinFill, (sizeof(int) * hostHashTableBounds.NBins) * 2);
			ppBins = 0;
			WISE_ALLOC(pTBinFill, (sizeof(int) * hostHashTableBounds.NTBins));
			cudaMemset(pCountTempTracks, 0, sizeof(int));
			WISE_ALLOC(pTBins, (hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity * sizeof(TempIntTrack)));
			WISE_ALLOC(ppTempTracks, (sizeof(TempIntTrack *) * C.MaxTracks));


			//printf("\nDEBUG %d %d %d %d", hostHashTableBounds.NBins * sizeof(int), hostHashTableBounds.NTBins * sizeof(int), C.HashBinCapacity * hostHashTableBounds.NBins * sizeof(IntChain *), hostHashTableBounds.TBinCapacity * sizeof(TempIntTrack) * hostHashTableBounds.NTBins);
			cudaMemset(pTBinFill, 0, sizeof(int) * (1 + hostHashTableBounds.NTBins));

			_CUDA_THROW_ERR_

			float vs = (((C.FilterVolumeLength100 - C.FilterVolumeLength0) * 0.01) / (1 << XY_SCALE_SHIFT));	
			float cm = C.FilterChainMult  / (1 << SLOPE_SCALE_SHIFT);
			//printf("\nDEBUG VS: %f", vs);
			//printf("\nDEBUG V0: %d", C.FilterVolumeLength0);

			long long seekgraintrials = 0;
			int lasttemptracks = 0;
			/* Fill bins. */
			int isx, isy;
			int isstep = ((int)(((C.XYHashTableBinSize >> XY_SCALE_SHIFT)  << SLOPE_SCALE_SHIFT) / ((float)C.ZThickness / (float)GetZScale() )));	
			for (isx = C.SlopeCenterX - (C.SlopeAcceptanceX/isstep)*isstep; isx <= C.SlopeCenterX + (C.SlopeAcceptanceX/isstep)*isstep; isx += isstep)
				for (isy = C.SlopeCenterY - (C.SlopeAcceptanceY/isstep)*isstep; isy <= C.SlopeCenterY + (C.SlopeAcceptanceY/isstep)*isstep; isy += isstep)
				{				
					cudaMemset(pBinFill, 0, sizeof(int) * hostHashTableBounds.NBins * 2);					
					//cudaMemset(pBinFill + hostHashTableBounds.NBins * 2, 0x7E7E7E7E, sizeof(int) * hostHashTableBounds.NBins);
					_CUDA_THROW_ERR_
					if (minviewtag > 0) /* KRYSS 20140116 allow a valid empty first view */
					{
						dim3 iblocks(pThis->m_Prop.multiProcessorCount /*ChHdr.Views*/, 1/*ChHdr.Views*/, 1);
						dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						//fill_skewhashtable_list_kernel<<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume);
						fill_skewhashtable1view_list_kernel<<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume, 0);
						fill_skewhashtable1view_list_kernel<<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume, 1);
						_CUDA_THROW_ERR_
					}
					{
						int totalpaircomputersize = 1;
						int paircomputersize = 1;
						int depth = 1;
						while (paircomputersize < hostHashTableBounds.NBins)
						{
							paircomputersize <<= 1;
							totalpaircomputersize += paircomputersize;
							depth++;
						}													
						WISE_ALLOC(pPairComputer, sizeof(int) * totalpaircomputersize);
						dim3 pcthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 pcblocks(totalpaircomputersize / pcthreads.x + 1, 1, 1);
						//compute_pairs_kernel<<<pcblocks, pcthreads>>>(pBinFill + hostHashTableBounds.NBins, pBinFill + 2 * hostHashTableBounds.NBins, C.FilterChain0, pPairComputer, hostHashTableBounds.NBins);						
						//compute_pairs_kernel<<<pcblocks, pcthreads>>>(pBinFill + hostHashTableBounds.NBins, pPairComputer, hostHashTableBounds.NBins);
						//compute_pairs_kernel<<<pcblocks, pcthreads>>>(pBinFill + hostHashTableBounds.NBins, pBinFill + 2 * hostHashTableBounds.NBins, minviewtag, pPairComputer, hostHashTableBounds.NBins);
						compute_pairs1v_kernel<<<pcblocks, pcthreads>>>(pBinFill + hostHashTableBounds.NBins, pBinFill + 2 * hostHashTableBounds.NBins, minviewtag, pPairComputer, hostHashTableBounds.NBins);
						/*
						int *pwDEBUG = new int[hostHashTableBounds.NBins];
						int *pzDEBUG = new int[hostHashTableBounds.NBins];
						cudaMemcpy(pwDEBUG, pBinFill + hostHashTableBounds.NBins, sizeof(int) * hostHashTableBounds.NBins, cudaMemcpyDeviceToHost);						
						cudaMemcpy(pzDEBUG, pPairComputer, sizeof(int) * hostHashTableBounds.NBins, cudaMemcpyDeviceToHost);						
						for (int iii = 0; iii < 100; iii++)
							if (pwDEBUG[iii]) 
							{
								printf("\n%d %08X %d", iii, pwDEBUG[iii], pzDEBUG[iii]);
							}
						delete [] pzDEBUG;
						delete [] pwDEBUG;
						*/	
						int d;
						int *pin = pPairComputer;
						int *pout;
						for (d = 0; d < depth; d++)
						{	
							pout = pin + paircomputersize;
							dim3 cthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
							dim3 cblocks((paircomputersize >> 1) / cthreads.x + 1, 1, 1);
							recursive_sum_kernel<<<cblocks, cthreads>>>(pin, pout, paircomputersize);
							paircomputersize >>= 1;									
							pin = pout;
						}
						int totalpairs = 0;
						cudaMemcpy(&totalpairs, pin, sizeof(int), cudaMemcpyDeviceToHost);
						//printf("\nDEBUG-TOTALPAIRS %d", totalpairs);
						WISE_ALLOC(pPairIndices, sizeof(int) * 3 * totalpairs);
						cudaMemset(pPairIndices, 0, sizeof(int) * 3 * totalpairs);

						dim3 pthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 pblocks((totalpairs / pthreads.x) + 1, 1, 1);
						//pair_find_kernel<<<pblocks, pthreads>>>(pin, depth, pBinFill + hostHashTableBounds.NBins, pBinFill, ppChainMapViewEntryPoints, pPairIndices);		
						pair_find1v_kernel<<<pblocks, pthreads>>>(pin, depth, pBinFill + hostHashTableBounds.NBins, pBinFill, ppChainMapViewEntryPoints, pPairIndices);								
						/*
						int *pwDEBUG = new int[totalpairs * 3];
						int *pzDEBUG = new int[hostHashTableBounds.NBins];
						cudaMemcpy(pwDEBUG, pPairIndices, sizeof(int) * totalpairs * 3, cudaMemcpyDeviceToHost);
						cudaMemcpy(pzDEBUG, pBinFill + hostHashTableBounds.NBins, sizeof(int) * hostHashTableBounds.NBins, cudaMemcpyDeviceToHost);
						for (int iii = 0; iii < 50; iii++)							
							{
								printf("\n%d %08X %08X %08X", iii, pwDEBUG[iii * 3], pwDEBUG[iii * 3 + 1], pwDEBUG[iii * 3 + 2]);
							}
						for (int iii = 0; iii < 50; iii++)
							if (pzDEBUG[iii]) 
							{
								printf("\n%d %08X", iii, pzDEBUG[iii]);
							}
						delete [] pzDEBUG;
						delete [] pwDEBUG;
						*/
						find_track_singlepass_kernel<<<pblocks, pthreads>>>(pPairIndices, totalpairs, ppChainMapViewEntryPoints, &pInternalInfo->C, pTBins, pTBinFill, ppTempTracks, pCountTempTracks, vs * vs, cm * cm, C.SlopeCenterX, C.SlopeCenterY, C.SlopeAcceptanceX, C.SlopeAcceptanceY,
							/*isx, isy, isstep, isstep,*/ &pInternalInfo->H, minviewtag);
						_CUDA_THROW_ERR_
					}
				}

			int temptracks = 0;
			THROW_ON_CUDA_ERR(cudaMemcpy(&temptracks, pCountTempTracks, sizeof(int), cudaMemcpyDeviceToHost));
			//printf("\nDEBUG TempTracks %d Seekgraintrials %d", temptracks, seekgraintrials);

			{
				dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks(pThis->m_Prop.multiProcessorCount, 1, 1);
				WISE_ALLOC(pScheduler, (1 + ithreads.x * iblocks.x) * sizeof(int));
				THROW_ON_CUDA_ERR(cudaMemset(pScheduler, 0, (1 + ithreads.x * iblocks.x) * sizeof(int)));
				int terminate;
				int ln = 0;
				do		
				{				
					THROW_ON_CUDA_ERR(cudaMemset(pScheduler, 0xff, sizeof(int)));
					mergetracks_kernel<<<iblocks, ithreads>>>(pCountTempTracks, ppTempTracks, pTBinFill, pTBins, pTracks, C.MergeTrackXYTolerance, C.MergeTrackZTolerance << Z_TO_XY_RESCALE_SHIFT, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins, hostHashTableBounds.TBinCapacity, pScheduler + 1, pScheduler);
					_CUDA_THROW_ERR_
					THROW_ON_CUDA_ERR(cudaMemcpy(&terminate, pScheduler, sizeof(int), cudaMemcpyDeviceToHost));
					ln++;								
				}
				while (terminate == 0);
				//printf("\nLaunches: %d", ln);
				{
					dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks(temptracks / ithreads.x + 1, 1, 1);
					filltracks_kernel<<<iblocks, ithreads>>>(pCountTempTracks, ppTempTracks, pTracks);
				}
			}
			THROW_ON_CUDA_ERR(cudaMemcpy(pHostTracks, pTracks, sizeof(TrackMapHeader), cudaMemcpyDeviceToHost));
			int size1 = pHostTracks->TrackSize();
			int size2 = pHostTracks->TotalSize() - pHostTracks->TrackSize();	
			pThis->m_PerformanceCounters.Tracks = pHostTracks->Count;
			HOST_WISE_ALLOC(pHostTracks, size1 + size2);
			THROW_ON_CUDA_ERR(cudaMemcpy(pHostTracks, pTracks, size1, cudaMemcpyDeviceToHost));
		}
	}
}