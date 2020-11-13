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

		bool PrismMapTracker::Tracker::IsFirstViewEmpty()
		{
#ifdef FIRST_VIEW_EMPTY
			return true;
#else
			return false;
#endif
		}

		bool PrismMapTracker::Tracker::IsLastViewEmpty()
		{
#ifdef LAST_VIEW_EMPTY
			return true;
#else
			return false;
#endif
		}

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
			C.FilterMinChains = 5;
			C.MergeTrackCell = 150 * GetXYScale();
			C.MergeTrackXYTolerance = 2 * GetXYScale();
			C.MergeTrackZTolerance = 3 * GetZScale();

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
			/*printf("\nDEBUG DUMP 1 %d %016X %016X", hostHashTableBounds.DEBUG1, pLastView, pThisView);
			{
				ChainView Ch;
				THROW_ON_CUDA_ERR(cudaMemcpy(&Ch, pLastView, sizeof(ChainView), cudaMemcpyDeviceToHost));
				printf("\nDEBUG LAST %d", Ch.Count);
				THROW_ON_CUDA_ERR(cudaMemcpy(&Ch, pThisView, sizeof(ChainView), cudaMemcpyDeviceToHost));
				printf("\nDEBUG THIS %d", Ch.Count);

			}*/
			TRACE_PROLOG3 printf(" %d HHBOUNDS %d %d %d %d %d", minviewtag, hostHashTableBounds.XBins, hostHashTableBounds.YBins, hostHashTableBounds.ZBins, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins);

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
			WISE_ALLOC(pTBins, 2 * (hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity * sizeof(TempIntTrack)));
			WISE_ALLOC(ppTempTracks, (sizeof(TempIntTrack *) * C.MaxTracks));

			TRACE_PROLOG3 printf(" %d WMALLOC %lld %lld %lld %lld %d", minviewtag, _MEM_(pBinFill), _MEM_(pTBinFill), _MEM_(pTBins), _MEM_(ppTempTracks), C.MaxTracks);


			//printf("\nDEBUG %d %d %d %d", hostHashTableBounds.NBins * sizeof(int), hostHashTableBounds.NTBins * sizeof(int), C.HashBinCapacity * hostHashTableBounds.NBins * sizeof(IntChain *), hostHashTableBounds.TBinCapacity * sizeof(TempIntTrack) * hostHashTableBounds.NTBins);
			//printf("\nDEBUG %d %d %d %d", hostHashTableBounds.MinX, hostHashTableBounds.MaxX, hostHashTableBounds.MinY, hostHashTableBounds.MaxY);
			//printf("\nDEBUG %d %d %d %d", hostHashTableBounds.XBins, hostHashTableBounds.YBins, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins);
			cudaMemset(pTBinFill, 0, sizeof(int) * hostHashTableBounds.NTBins);

			_CUDA_THROW_ERR_

			float vs = (((C.FilterVolumeLength100 - C.FilterVolumeLength0) * 0.01) / (1 << XY_SCALE_SHIFT));	
			float cm = C.FilterChainMult  / (1 << SLOPE_SCALE_SHIFT);
			//printf("\nDEBUG VS: %f", vs);
			//printf("\nDEBUG V0: %d", C.FilterVolumeLength0);

			//long long seekgraintrials = 0;
			//int lasttemptracks = 0;
			/* Fill bins. */
			int isx, isy;
			if (pThis->m_EnableDebugDump)
			{				
				pThis->m_DebugDump.MinTagView = minviewtag;
				pThis->m_DebugDump.TempTracks = 0;
				pThis->m_DebugDump.HTBounds = hostHashTableBounds;				
			}
			int isstep = ((int)(((C.XYHashTableBinSize >> XY_SCALE_SHIFT) << SLOPE_SCALE_SHIFT) / ((float)C.ZThickness / (float)GetZScale() )));	
			for (isx = C.SlopeCenterX - (C.SlopeAcceptanceX/isstep)*isstep; isx <= C.SlopeCenterX + (C.SlopeAcceptanceX/isstep)*isstep; isx += isstep)
				for (isy = C.SlopeCenterY - (C.SlopeAcceptanceY/isstep)*isstep; isy <= C.SlopeCenterY + (C.SlopeAcceptanceY/isstep)*isstep; isy += isstep)
				{				
					TRACE_PROLOG1 printf("\nVIEW %d NBins %d pBinfill %d", minviewtag, hostHashTableBounds.NBins, _MEM_(pBinFill));
					cudaMemset(pBinFill, 0, sizeof(int) * hostHashTableBounds.NBins * 2);
					_CUDA_THROW_ERR_
					//cudaMemset(pBinFill + hostHashTableBounds.NBins * 2, 0x7E7E7E7E, sizeof(int) * hostHashTableBounds.NBins);
					//_CUDA_THROW_ERR_					

#ifdef FIRST_VIEW_EMPTY
					if (minviewtag > 0)
#endif
					{
						dim3 iblocks(pThis->m_Prop.multiProcessorCount /*ChHdr.Views*/, 1/*ChHdr.Views*/, 1);
						dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						TRACE_PROLOG1 printf("\nVIEW %d iblocks %d ithreads %d", minviewtag, iblocks.x, ithreads.x);
						fill_skewhashtable1view_list_kernel <<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume, 0);
						_CUDA_ONLOG2_THROW_ERR_
						fill_skewhashtable1view_list_kernel <<<iblocks, ithreads>>>(pBinFill, ppChainMapViewEntryPoints, &pInternalInfo->H, isx, isy, C.MinChainVolume, 1);
						_CUDA_ONLOG2_THROW_ERR_
					}
					if (pThis->m_EnableDebugDump)
					{
#ifdef FIRST_VIEW_EMPTY
						if (minviewtag > 0)
#endif
						{
							int _i;
							for (_i = 0; _i < 2; _i++)
							{
								cudaMemcpy(&pThis->m_DebugDump.Views[_i].pV, ppChainMapViewEntryPoints + _i, sizeof(ChainView *), cudaMemcpyDeviceToHost);
								ChainView tc;
								cudaMemcpy(&tc, pThis->m_DebugDump.Views[_i].pV, sizeof(ChainView), cudaMemcpyDeviceToHost);
								pThis->m_DebugDump.Views[_i].pHostView = (ChainView *)malloc(tc.Size());
								cudaMemcpy(pThis->m_DebugDump.Views[_i].pHostView, pThis->m_DebugDump.Views[_i].pV, tc.Size(), cudaMemcpyDeviceToHost);
							}
						}
						DebugDump::t_TrackIteration &ti = pThis->m_DebugDump.AddIteration();
						ti.ISX = isx;
						ti.ISY = isy;		
						ti.TotalPairs = 0;
						ti.Prepare(hostHashTableBounds.NBins * 2, 0, 0, 0);
						cudaMemcpy(ti.pBinFill[0], pBinFill, sizeof(int) * hostHashTableBounds.NBins * 2, cudaMemcpyDeviceToHost);						
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
						dim3 pcthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 pcblocks(totalpaircomputersize / pcthreads.x + 1, 1, 1);
						WISE_ALLOC(pPairComputer, sizeof(int) * (pcthreads.x * pcblocks.x));
						TRACE_PROLOG1 printf("\nVIEW %d pcthreads %d pcblocks %d NBins %d pBinfill %d", minviewtag, pcthreads.x, pcblocks.x, hostHashTableBounds.NBins, _MEM_(pBinFill));
						cudaMemset(pPairComputer, 0, sizeof(int) * (pcthreads.x * pcblocks.x));						
						compute_pairs1v_kernel<<<pcblocks, pcthreads>>>(pBinFill + hostHashTableBounds.NBins, pPairComputer, hostHashTableBounds.NBins);
						_CUDA_ONLOG2_THROW_ERR_
						if (pThis->m_EnableDebugDump)
						{
							DebugDump::t_TrackIteration &ti = pThis->m_DebugDump.pTrackIterations[pThis->m_DebugDump.TrackIterations - 1];
							cudaMemcpy(ti.pBinFill[1], pBinFill, sizeof(int) * hostHashTableBounds.NBins * 2, cudaMemcpyDeviceToHost);
						}
						int d;
						int *pin = pPairComputer;
						int *pout;
						for (d = 1; d < depth; d++)
						{	
							pout = pin + paircomputersize;
							dim3 cthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
							dim3 cblocks((paircomputersize >> 1) / cthreads.x + 1, 1, 1);
							recursive_sum_kernel<<<cblocks, cthreads>>>(pin, pout, paircomputersize);
							paircomputersize >>= 1;									
							pin = pout;
						}
						if (pThis->m_EnableDebugDump)
						{
							DebugDump::t_TrackIteration &ti = pThis->m_DebugDump.pTrackIterations[pThis->m_DebugDump.TrackIterations - 1];
							cudaMemcpy(ti.pBinFill[2], pBinFill, sizeof(int) * hostHashTableBounds.NBins * 2, cudaMemcpyDeviceToHost);							
							ti.Prepare(0, totalpaircomputersize, 0, 0);
							cudaMemcpy(ti.pPairComputer, pPairComputer, sizeof(int) * totalpaircomputersize, cudaMemcpyDeviceToHost);
						}
						int totalpairs = 0;
						cudaMemcpy(&totalpairs, pin, sizeof(int), cudaMemcpyDeviceToHost);
						TRACE_PROLOG3 printf(" %d TOTALPAIRS %d", minviewtag, totalpairs);
						TRACE_PROLOG2 printf("\nVIEW %d ISX %d ISY %d TOTALPAIRS %d RECURSIONDEPTH %d", minviewtag, isx, isy, totalpairs, depth);
						if (totalpairs < 0)
						{
							totalpairs = 0;
							TRACE_PROLOG2 printf("\nWARNING IN TRACKING: TOTALPAIRS EXCEEDED CAPACITY OF INTEGER DATATYPE AND RESET TO %d.", totalpairs);
						}
						WISE_ALLOC_EXCEPT(pPairIndices, sizeof(int) * 3LL * totalpairs, (totalpairs = 0), "PAIRS RESET TO 0");
						if (totalpairs > 0)
						{
							cudaMemset(pPairIndices, 0, sizeof(int) * 3LL * totalpairs);
							TRACE_PROLOG3 printf(" %d PAIRINDICES %d", minviewtag, _MEM_(pPairIndices));
						}
						dim3 pthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 pblocks((totalpairs / pthreads.x) + 1, 1, 1);
						TRACE_PROLOG2 printf("\nDEBUG totalpairs %d pair_find1v_kernel %d %d", totalpairs, pthreads.x, pblocks.x);
						if (totalpairs > 0)
						{
							pair_find1v_kernel << <pblocks, pthreads >> > (pin, depth, pBinFill + hostHashTableBounds.NBins, pBinFill, ppChainMapViewEntryPoints, pPairIndices);
							_CUDA_THROW_ERRMSG_(::cudaErrorUnknown, "HINT: The error may be due to execution timeout. Too many PAIRS are being formed. Consider reducing the number of combinations.");
						}
						if (pThis->m_EnableDebugDump)
						{
							DebugDump::t_TrackIteration &ti = pThis->m_DebugDump.pTrackIterations[pThis->m_DebugDump.TrackIterations - 1];
							ti.Prepare(0, 0, 3LL * totalpairs, hostHashTableBounds.NTBins);
							ti.TotalPairs = totalpairs;
							cudaMemcpy(ti.pPairIndices, pPairIndices, sizeof(int) * 3LL * totalpairs, cudaMemcpyDeviceToHost);
							cudaMemcpy(ti.pPairComputer, pPairComputer, sizeof(int) * totalpaircomputersize, cudaMemcpyDeviceToHost);
						}
						if (totalpairs > 0)
						{
							find_track_singlepass_kernel << <pblocks, pthreads >> > (pPairIndices, totalpairs, ppChainMapViewEntryPoints, &pInternalInfo->C, pTBins, pTBinFill, ppTempTracks, pCountTempTracks, vs * vs, cm * cm, C.SlopeCenterX, C.SlopeCenterY, C.SlopeAcceptanceX, C.SlopeAcceptanceY,
								&pInternalInfo->H, minviewtag, C.MaxTracks);
							_CUDA_THROW_ERRMSG_(::cudaErrorUnknown, "HINT: The error may be due to execution timeout. Too many TRACKS are being formed. Consider reducing the number of combinations.");
						}
					}
				}

			int temptracks = 0;
			THROW_ON_CUDA_ERR(cudaMemcpy(&temptracks, pCountTempTracks, sizeof(int), cudaMemcpyDeviceToHost));
			//printf("\nDEBUG temptracks %d", temptracks);
			if (pThis->m_EnableDebugDump)
				pThis->m_DebugDump.TempTracks = temptracks;
			TRACE_PROLOG3 printf(" %d TEMPTRACKS %d", minviewtag, temptracks);

			for (int hrow = 0; hrow < hostHashTableBounds.YTBins; hrow++)
			{
				long long totalpaircomputersize = 1;
				long long paircomputersize = 1;
				int depth = 1;
				while (paircomputersize < /*hostHashTableBounds.NTBins*/hostHashTableBounds.XTBins)
				{
					paircomputersize <<= 1;
					totalpaircomputersize += paircomputersize;
					depth++;
				}			
				TRACE_PROLOG3 printf(" %d HROW %d %d", minviewtag, hrow, temptracks);
				//printf("\nDEBUG Totalpaircomputersize %d %d %d %d", totalpaircomputersize, hostHashTableBounds.XTBins, hostHashTableBounds.YTBins, hostHashTableBounds.NTBins);
				WISE_ALLOC(pPairComputer, sizeof(int) * totalpaircomputersize);
				dim3 pcthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 pcblocks(paircomputersize / pcthreads.x + 1, 1, 1);
				cudaMemset(pPairComputer, 0, sizeof(int) * totalpaircomputersize);
				mergetracks_prepare<<<pcblocks, pcthreads>>>(pTBinFill + hostHashTableBounds.XTBins * hrow, hostHashTableBounds.XTBins, 1 /*hostHashTableBounds.YTBins*/, pPairComputer);
				_CUDA_THROW_ERR_
				int d;
				int *pin = pPairComputer;
				int *pout;
				for (d = 1; d < depth; d++)
				{	
					pout = pin + paircomputersize;
					dim3 cthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 cblocks((paircomputersize >> 1) / cthreads.x + 1, 1, 1);
					recursive_sum_kernel<<<cblocks, cthreads>>>(pin, pout, paircomputersize);
					paircomputersize >>= 1;									
					pin = pout;
				}
				long long totalpairs = 0;
				cudaMemcpy(&totalpairs, pin, sizeof(int), cudaMemcpyDeviceToHost);
				TRACE_PROLOG3 printf(" %d MERGETRACKS-TOTALPAIRS %d %d", minviewtag, hrow, totalpairs);
				//printf("\nDEBUG-MERGETRACKS TOTALPAIRS %d", totalpairs);
				if (totalpairs > 0)
				{
					try
					{
						WISE_ALLOC(pPairIndices, sizeof(int) * 2 * totalpairs);
						TRACE_PROLOG3 printf(" %d WMALLOCPAIRINDICES %d %d", minviewtag, hrow, totalpairs);
					}
					catch (...)
					{
						TRACE_PROLOG1 printf("\nERROR: Memory for track merging could not be allocated. Deleting all tracks.");
						cudaGetLastError(); /* reset the error state */
						DEALLOC(pPairIndices);
						totalpairs = 0;
						temptracks = 0;
						cudaMemcpy(pCountTempTracks, &temptracks, sizeof(int), cudaMemcpyHostToDevice);
						break;
					}
					if (totalpairs > 0)
					{
						/*int *pd0 = new int[hostHashTableBounds.NTBins];
						cudaMemcpy(pd0, pTBinFill, sizeof(int) * hostHashTableBounds.NTBins, cudaMemcpyDeviceToHost);
						int _j;
						printf("\nDEBUG pTBinFill DUMP\n");
						for (_j = 0; _j < hostHashTableBounds.NTBins; _j++)
							printf(" %d", pd0[_j]);
						delete [] pd0;
						int *pd1 = new int[totalpairs * 2];
						int *pd2 = new int[totalpairs * 2];
						int *pd3 = new int[hostHashTableBounds.NTBins];
						cudaMemcpy(pd3, pPairComputer, sizeof(int) * hostHashTableBounds.NTBins, cudaMemcpyDeviceToHost);
						*/
						dim3 cthreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						int _sq_blocks = sqrt((float)totalpairs / cthreads.x) + 1;
						dim3 cblocks(1, _sq_blocks, _sq_blocks);						
						mergetracks_split_and_index_kernel<<<cblocks, cthreads>>>(pin, depth + 1, pPairIndices, totalpairs/*, _idz*/);
						//_CUDA_THROW_ERR_
						//cudaMemcpy(pd1, pPairIndices, sizeof(int) * totalpairs * 2, cudaMemcpyDeviceToHost);
						mergetracks_mapindex_kernel<<<cblocks, cthreads>>>(pPairIndices, pTBinFill + hrow * hostHashTableBounds.XTBins, hostHashTableBounds.XTBins, 1 /*hostHashTableBounds.YTBins*/, hostHashTableBounds.TBinCapacity, totalpairs/*, _idz*/);
						//_CUDA_THROW_ERR_
						//cudaMemcpy(pd2, pPairIndices, sizeof(int) * totalpairs * 2, cudaMemcpyDeviceToHost);
						/*
						{
							printf("\nDEBUG-CHECKING PAIRS");
							int _i;
							int _errs = 0;
							for (_i = 0; _i < totalpairs; _i++)
							{
								int a[2];
								a[0] = pd2[_i * 2];
								a[1] = pd2[_i * 2 + 1];
								if (
									a[0] < 0 || 
									a[1] < 0 || 
									a[0] >= hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity ||
									a[1] >= hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity)
								{
									int binfill;
									cudaMemcpy(&binfill, pTBinFill + pd1[_i * 2], sizeof(int), cudaMemcpyDeviceToHost);
									printf("\nDEBUG-ERROR FOUND: %d | %d %d (max %d) | %d %d binfill %d binsize %d pairs %d", _i, a[0], a[1], hostHashTableBounds.NTBins * hostHashTableBounds.TBinCapacity, pd1[_i * 2], pd1[_i * 2 + 1], binfill, hostHashTableBounds.TBinCapacity, pd3[pd1[_i * 2]]);
									if (++_errs >= 10)
									{
										delete [] pd2;
										delete [] pd1;
										throw("DEBUG CHECK FAILED");
									}
								}
							}
							delete [] pd2;
							delete [] pd1;
							printf("\nDEBUG-END PAIR CHECK");
						}*/
						mergetracks_kernel<<<cblocks, cthreads>>>(pPairIndices, pTBins + hrow * hostHashTableBounds.TBinCapacity * hostHashTableBounds.XTBins, C.MergeTrackXYTolerance, C.MergeTrackZTolerance, totalpairs/*, _idz*/);
						//_CUDA_THROW_ERR_
					}
				}
			}
			{
				dim3 ithreads(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks(temptracks / ithreads.x + 1, 1, 1);
				filltracks_kernel<<<iblocks, ithreads>>>(pCountTempTracks, ppTempTracks, pTracks);
				_CUDA_THROW_ERR_
			}

			THROW_ON_CUDA_ERR(cudaMemcpy(pHostTracks, pTracks, sizeof(TrackMapHeader), cudaMemcpyDeviceToHost));
			int size1 = pHostTracks->TrackSize();
			int size2 = pHostTracks->TotalSize() - pHostTracks->TrackSize();				
			pThis->m_PerformanceCounters.Tracks = pHostTracks->Count;
			TRACE_PROLOG3 printf(" %d HOSTTRACKS %d %d", minviewtag, size1, size2);
			//printf("\nDEBUG Tracks %d", pThis->m_PerformanceCounters.Tracks);
			HOST_WISE_ALLOC(pHostTracks, size1 + size2);
			THROW_ON_CUDA_ERR(cudaMemcpy(pHostTracks, pTracks, size1, cudaMemcpyDeviceToHost));
			//DEALLOC(pPairComputer);
			//DEALLOC(pPairIndices);
		}
	}
}