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
			CTOR_INIT(pSchedulerCompactor),
			CTOR_INIT(pPairComputer),
			CTOR_INIT(pPairIndices),
			CTOR_INIT(pSegFindTrackStatus),
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
			DEALLOC(pSchedulerCompactor);
			DEALLOC(pScheduler);
			DEALLOC(pSegFindTrackStatus);
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
				dim3 ithreads(1,1,1);
				dim3 iblocks(1,1,1);
				explore_skewchainmap_kernel<<<iblocks, ithreads>>>(pLastView, pThisView, width, height, pInternalInfo, ppChainMapViewEntryPoints, pTracks);
			}

			THROW_ON_CUDA_ERR(cudaMemcpy(&hostHashTableBounds, &pInternalInfo->H, sizeof(HashTableBounds), cudaMemcpyDeviceToHost));
			printf("\nDEBUG DUMP 1 %d", hostHashTableBounds.DEBUG1);

			if (C.MaxTracks < hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity)
			{
				C.MaxTracks = hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity;
				printf("\nMaxTracks corrected to %d", C.MaxTracks);
			}

			printf("\nHashTable Grid: %d %d %d (%d)\n%d %d %d", hostHashTableBounds.XBins, hostHashTableBounds.YBins, hostHashTableBounds.ZBins, hostHashTableBounds.NBins, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins, hostHashTableBounds.TBinCapacity);

//			WISE_ALLOC(pTrackGrains, sizeof(IntChain) * C.MaxTracks * _MAX_GRAINS_PER_TRACK_);


			WISE_ALLOC(pBinFill, (sizeof(int) * hostHashTableBounds.NBins) * 2);
			ppBins = 0;
			WISE_ALLOC(pTBinFill, (sizeof(int) * hostHashTableBounds.NTBins));
			cudaMemset(pCountTempTracks, 0, sizeof(int));
			WISE_ALLOC(pTBins, (hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity * sizeof(TempIntTrack)));
			WISE_ALLOC(ppTempTracks, (sizeof(TempIntTrack *) * C.MaxTracks));


			printf("\nDEBUG %d %d %d %d", hostHashTableBounds.NBins * sizeof(int), hostHashTableBounds.NTBins * sizeof(int), C.HashBinCapacity * hostHashTableBounds.NBins * sizeof(IntChain *), hostHashTableBounds.TBinCapacity * sizeof(TempIntTrack) * hostHashTableBounds.NTBins);
			cudaMemset(pTBinFill, 0, sizeof(int) * (1 + hostHashTableBounds.NTBins));

			_CUDA_THROW_ERR_

			float vs = (((C.FilterVolumeLength100 - C.FilterVolumeLength0) * 0.01) / (1 << XY_SCALE_SHIFT));	
			float cm = C.FilterChainMult  / (1 << SLOPE_SCALE_SHIFT);
			printf("\nDEBUG VS: %f", vs);
			printf("\nDEBUG V0: %d", C.FilterVolumeLength0);

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
					{
						dim3 iblocks(pThis->m_Prop.multiProcessorCount /*ChHdr.Views*/, 1/*ChHdr.Views*/, 1);
						dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						//fill_skewhashtable_list_kernel<<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume);
						fill_skewhashtable1view_list_kernel<<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume, 0);
						fill_skewhashtable1view_list_kernel<<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume, 1);
						_CUDA_THROW_ERR_
					}
#if 1
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
#else
					{
						dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 iblocks((hostHashTableBounds.XBins * hostHashTableBounds.YBins) / ithreads.x + 1, 1, 1);			
						WISE_ALLOC(pSegFindTrackStatus, sizeof(_segmented_findtrack_kernel_status_) * iblocks.x * ithreads.x);			
						WISE_ALLOC(pScheduler, sizeof(int) * (1 + ithreads.x * iblocks.x) * 2);			
						reset_scheduler_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pScheduler, ithreads.x * iblocks.x, 0);			
						find_tracks_skewreset_list_kernel<<<iblocks, ithreads>>>(pSegFindTrackStatus, pBinFill, ppChainMapViewEntryPoints, hostHashTableBounds.XBins, hostHashTableBounds.YBins, C.HashBinCapacity, pScheduler);
						_CUDA_THROW_ERR_				
						int terminate;
						int ln = 0;
						do		
						{			
							find_tracks_skewslope_list_kernel<<<iblocks, ithreads>>>(pSegFindTrackStatus, ppChainMapViewEntryPoints, hostHashTableBounds.XBins, hostHashTableBounds.YBins, C.HashBinCapacity, 
									isx/*C.SlopeCenterX*/, isy/*C.SlopeCenterY*/, isstep/*C.SlopeAcceptanceX*/, isstep/*C.SlopeAcceptanceY*/, 
									C.MinLength, pScheduler);
							_CUDA_THROW_ERR_
							int ix = 0, iy = 0;	
							//for (ix = -1; ix <= 1; ix++)
							//	for (iy = -1; iy <= 1; iy++)
							
							{						
								find_tracks_skewgrainseek_list_kernel<<<iblocks, ithreads>>>(pSegFindTrackStatus, ppChainMapViewEntryPoints, pBinFill, hostHashTableBounds.XBins, hostHashTableBounds.YBins, C.HashBinCapacity, C.XYTolerance, C.ZTolerance << Z_TO_XY_RESCALE_SHIFT, /*ix, iy, */pScheduler);
								_CUDA_THROW_ERR_
							}							
							/*
							{
								_segmented_findtrack_kernel_status_ *pXdebug = new _segmented_findtrack_kernel_status_[iblocks.x * ithreads.x];
								cudaMemcpy(pXdebug, pSegFindTrackStatus, sizeof(_segmented_findtrack_kernel_status_) * iblocks.x * ithreads.x, cudaMemcpyDeviceToHost);
								int ixi;
								for (ixi = 0; ixi < iblocks.x * ithreads.x; ixi++)
									if (pXdebug[ixi].SearchGrains)
										seekgraintrials++;
								delete [] pXdebug;
							}
							*/
							find_tracks_skewchecktrack_kernel<<<iblocks, ithreads>>>(pSegFindTrackStatus, &pInternalInfo->C, pTBins, pTBinFill, ppTempTracks, pCountTempTracks, vs * vs, cm * cm, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins, hostHashTableBounds.TBinCapacity, minviewtag, pScheduler);
							_CUDA_THROW_ERR_					
							reset_scheduler_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pScheduler + (1 + ithreads.x * iblocks.x), 0, 0);
							find_tracks_skewincrement_list_kernel<<<iblocks, ithreads>>>(pSegFindTrackStatus, pBinFill, ppChainMapViewEntryPoints, hostHashTableBounds.XBins, hostHashTableBounds.YBins, C.HashBinCapacity, pScheduler, C.MinLength >> Z_TO_XY_RESCALE_SHIFT, pScheduler + (1 + iblocks.x * ithreads.x));


#if 1
							{
								int totallength = 0;
								cudaMemcpy(&totallength, pScheduler, sizeof(int), cudaMemcpyDeviceToHost);
								int totalcompactorsize = 1;
								int compactorsize = 1;
								int depth = 1;
								while (compactorsize < totallength)
								{
									compactorsize <<= 1;
									totalcompactorsize += compactorsize;
									depth++;
								}								
								WISE_ALLOC(pSchedulerCompactor, sizeof(int) * totalcompactorsize);
								int d;
								dim3 rthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
								dim3 rblocks(compactorsize / rthreads.x + 1, 1, 1);

								reset_compactor_kernel<<<rblocks, rthreads>>>(pScheduler + (2 + ithreads.x * iblocks.x), pSchedulerCompactor, pScheduler);
								int *pin = pSchedulerCompactor;
								int *pout;
								for (d = 0; d < depth; d++)
								{	
									pout = pin + compactorsize;
									dim3 cthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
									dim3 cblocks((compactorsize >> 1) / cthreads.x + 1, 1, 1);
									recursive_sum_kernel<<<cblocks, cthreads>>>(pin, pout, compactorsize);
									compactorsize >>= 1;									
									pin = pout;
								}
								compactor_find_kernel<<<iblocks, ithreads>>>(pin, depth, pScheduler + (2 + ithreads.x * iblocks.x), pScheduler + 1, pScheduler);
							}
#else
							compact_scheduler_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pScheduler + (1 + iblocks.x * ithreads.x), pScheduler);
							THROW_ON_CUDA_ERR(cudaMemcpy(pScheduler, pScheduler + (1 + iblocks.x * ithreads.x), sizeof(int) * (1 + iblocks.x * ithreads.x), cudaMemcpyDeviceToDevice));
#endif
							
							THROW_ON_CUDA_ERR(cudaMemcpy(&terminate, pScheduler, sizeof(int), cudaMemcpyDeviceToHost));							
							//printf("\nTerminate %d", terminate);
							iblocks.x = terminate / ithreads.x + 1;
							ln++;	
						}
						while(terminate > 0);
						int temptracks = 0;
			THROW_ON_CUDA_ERR(cudaMemcpy(&temptracks, pCountTempTracks, sizeof(int), cudaMemcpyDeviceToHost));
			//			printf("\nLaunches: %d Slopes %d %d", ln, isx, isy);
						printf("\nLaunches: %d Slopes %d %d %d", ln, isx, isy, temptracks - lasttemptracks);
						lasttemptracks = temptracks;
					}
#endif
				}

			int temptracks = 0;
			THROW_ON_CUDA_ERR(cudaMemcpy(&temptracks, pCountTempTracks, sizeof(int), cudaMemcpyDeviceToHost));
			printf("\nDEBUG TempTracks %d Seekgraintrials %d", temptracks, seekgraintrials);
#if 0
			{
				int *pwDEBUG = new int[hostHashTableBounds.NTBins];
				cudaMemcpy(pwDEBUG, pTBinFill, sizeof(int) * hostHashTableBounds.NTBins, cudaMemcpyDeviceToHost);
				int totaltemptracks = 0;
				for (int i = 0; i < hostHashTableBounds.NTBins; i++)
				{
				//	printf("\nDEBUG %d %d", i, pwDEBUG[i]);
					totaltemptracks += pwDEBUG[i];
				}
				printf("\nTotal: %d", totaltemptracks);
				delete [] pwDEBUG;
			}
#endif
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
				printf("\nLaunches: %d", ln);
				{
					dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks(temptracks / ithreads.x + 1, 1, 1);
					filltracks_kernel<<<iblocks, ithreads>>>(pCountTempTracks, ppTempTracks, pTracks);
				}
			}
			THROW_ON_CUDA_ERR(cudaMemcpy(pHostTracks, pTracks, sizeof(TrackMapHeader), cudaMemcpyDeviceToHost));
			int size1 = pHostTracks->TrackSize();
			int size2 = pHostTracks->TotalSize() - pHostTracks->TrackSize();	
			HOST_WISE_ALLOC(pHostTracks, size1 + size2);
			THROW_ON_CUDA_ERR(cudaMemcpy(pHostTracks, pTracks, size1, cudaMemcpyDeviceToHost));
		}
	}
}