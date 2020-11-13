#include "gpu_incremental_map_track.h"
#include "gpu_map_kernels.h"
#include "gpu_defines.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

namespace SySal
{
	namespace GPU
	{
		int PrismMapTracker::ClusterChainer::FindEdges(EmulsionEdge &t, EmulsionEdge &b, IntClusterFile &cf, int threshold, int &refimg)
		{
			t.Valid = b.Valid = false;
			int retval = 0;
			int i;
			int topout = -1, bottomout = cf.Images();			
			for (i = 0; i < cf.Images() && cf.ImageClusterCounts(i) < threshold; i++)
				topout = i;
			for (i = cf.Images() - 1; i >= 0 && cf.ImageClusterCounts(i) < threshold; i--)
				bottomout = i;	
			if (topout >= bottomout) return 0;
			t.Z = cf.StageZ(0);
			b.Z = cf.StageZ(cf.Images() - 1);
			if (topout >= 0)
			{
				if (topout < cf.Images() - 1)
					t.Z = 0.5f * (cf.StageZ(topout) + cf.StageZ(topout + 1));
				else
					t.Z = cf.StageZ(topout);
				t.Valid = true;
				retval++;
			}
			if (bottomout <= cf.Images() - 1)
			{
				if (bottomout > 0)
					b.Z = 0.5f * (cf.StageZ(bottomout) + cf.StageZ(bottomout - 1));
				else
					b.Z = cf.StageZ(bottomout);
				b.Valid = true;
				retval++;
			}
			refimg = (topout + bottomout) / 2;
			return retval;
		}

		void PrismMapTracker::ClusterChainer::HardReset()
		{
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))
			
			cudaDeviceReset();

			HOST_DEALLOC(pFixedZs)
			HOST_DEALLOC(pHostStagePos)

			HARD_RESET(pChMapWnd);
			HARD_RESET(pChainMapHeader);
			HARD_RESET(pCurv);
			HARD_RESET(pStagePos);
			HARD_RESET(pMapCounts);
			HARD_RESET(pDeltas);
			HARD_RESET(pChainCounts);
			HARD_RESET(pCompactChains);
			HARD_RESET(pChains);
			HARD_RESET(pCells);
			HARD_RESET(pCellContents);
			HARD_RESET(pClusterChains);
			HARD_RESET(pChainCounts);
			HARD_RESET(pClusterData);
			HARD_RESET(pClusterPos);
			HARD_RESET(pMapClusters);
			HARD_RESET(pMapChains);
			HARD_RESET(pPairs);
			HARD_RESET(pClustersInCell);
			HARD_RESET(pPairComputer);
			HARD_RESET(pPairMatchResult);
			HARD_RESET(pMatchMap);

			ThicknessSamples = 0;
			WeakThickness = 0.0;
		}

		PrismMapTracker::ClusterChainer::ClusterChainer() : 
			CTOR_INIT(pClusterPos), 
			CTOR_INIT(pClusterData), 
			CTOR_INIT(pClusterChains), 
			CTOR_INIT(pCellContents), 
			CTOR_INIT(pCells), 
			CTOR_INIT(pChains), 
			CTOR_INIT(pCompactChains), 	
			CTOR_INIT(pChainCounts), 
			CTOR_INIT(pDeltas), 
			CTOR_INIT(pMapCounts),
			CTOR_INIT(pCurv),
			CTOR_INIT(pStagePos),
			CTOR_INIT(pHostStagePos), 
			CTOR_INIT(pChainMapHeader),	
			CTOR_INIT(pFixedZs),
			CTOR_INIT(pChMapWnd),
			CTOR_INIT(pLastView),
			CTOR_INIT(pThisView),
			CTOR_INIT(pMapClusters),
			CTOR_INIT(pMapChains),
			CTOR_INIT(pClustersInCell),
			CTOR_INIT(pPairComputer),
			CTOR_INIT(pPairs),
			CTOR_INIT(pPairMatchResult),
			CTOR_INIT(pMatchMap)
		{
			ThicknessSamples = 0;
			pThicknessSamples = (double *)malloc(sizeof(double) * 32);
			WeakThickness = 0.0;

			C.MaxCellContent = 8;
			C.CellSize = 160;
			C.ClusterMapCoarseTolerance = 16;
			C.ClusterMapFineTolerance = 1;
			C.ClusterMapFineAcceptance = 8;
			C.ClusterMapMaxXOffset = 160;
			C.ClusterMapMaxYOffset = 24;
			C.MinClustersPerChain = 2;
			C.MinVolumePerChain = 8;
			C.ClusterMapMinSize = 6;
			C.ChainMapXYCoarseTolerance = 8;
			C.ChainMapXYFineTolerance = 1;
			C.ChainMapXYFineAcceptance = 8;
			C.ChainMapZCoarseTolerance = 2 << Z_SCALE_SHIFT;
			C.ChainMapZFineTolerance = (1 << Z_SCALE_SHIFT) / 8;
			C.ChainMapZFineAcceptance = (1 << Z_SCALE_SHIFT);
			C.ChainMapMaxXOffset = 160;
			C.ChainMapMaxYOffset = 24;
			C.ChainMapMaxZOffset = 10 << Z_SCALE_SHIFT;
			C.ChainMapMinVolume = 16;
			C.ChainMapSampleDivider = 10;
			C.MaxChains = 2000000;
			C.ClusterMapSampleDivider = 20;
			C.ClusterThreshold = 3000;

			C.MaxCellContent = 0;
			C.CellSize = 0;
			C.ClusterMapCoarseTolerance = 0;
			C.ClusterMapFineTolerance = 0;
			C.ClusterMapMaxXOffset = 0;
			C.ClusterMapMaxYOffset = 0;
			C.MinClustersPerChain = 0;
			C.MinVolumePerChain = 0;
			C.ChainMapXYCoarseTolerance = 0;
			C.ChainMapXYFineTolerance = 0;
			C.ChainMapZCoarseTolerance = 0;
			C.ChainMapZFineTolerance = 0;
			C.ChainMapMaxXOffset = 0;
			C.ChainMapMaxYOffset = 0;
			C.ChainMapMaxZOffset = 0;
			C.MaxChains = 0;

			IC.DMagDX = IC.DMagDY = IC.DMagDZ = 0.0f;
			IC.XYCurvature = IC.ZCurvature = 0.0f;

			IC.DMagDX = IC.DMagDY = IC.DMagDZ = -1.0f;
			IC.XYCurvature = IC.ZCurvature = -1.0f;			
		}



		PrismMapTracker::ClusterChainer::~ClusterChainer() 
		{
			if (pThicknessSamples) free(pThicknessSamples);

			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))
			DEALLOC(pChMapWnd)
			HOST_DEALLOC(pFixedZs)
			DEALLOC(pChainMapHeader)
			DEALLOC(pCurv)
			HOST_DEALLOC(pHostStagePos)
			DEALLOC(pStagePos)
			DEALLOC(pMapCounts)
			DEALLOC(pDeltas)
			DEALLOC(pChainCounts)
			DEALLOC(pCompactChains)
			DEALLOC(pChains)
			DEALLOC(pCells)
			DEALLOC(pCellContents)
			DEALLOC(pClusterChains)
			DEALLOC(pChainCounts)
			DEALLOC(pClusterData)
			DEALLOC(pClusterPos)	
			DEALLOC(pMapClusters)
			DEALLOC(pMapChains)
			DEALLOC(pClustersInCell)
			DEALLOC(pPairComputer)
			DEALLOC(pPairs)
			DEALLOC(pPairMatchResult)
			DEALLOC(pMatchMap)
		}

		int PrismMapTracker::ClusterChainer::DistributeClusterPairsToThreads(int totalmapclusters)
		{
			if (totalmapclusters <= 0) return 0;
			cudaError_t err;

			int totalpaircomputersize = 1;
			int paircomputersize = 1;
			int depth = 1;
			while (paircomputersize < totalmapclusters)
			{
				paircomputersize <<= 1;
				totalpaircomputersize += paircomputersize;
				depth++;
			}
			WISE_ALLOC(pPairComputer, sizeof(int) * totalpaircomputersize)
			THROW_ON_CUDA_ERR(cudaMemcpy(pPairComputer, pClustersInCell, sizeof(int) * totalmapclusters, cudaMemcpyDeviceToDevice))
			if (totalmapclusters < paircomputersize)
				cudaMemset(pPairComputer + totalmapclusters, 0, sizeof(int) * (paircomputersize - totalmapclusters));
			int d;
			int *pin = pPairComputer;
			int *pout;
			dim3 cthreads, cblocks;
			for (d = 1; d < depth; d++)
			{	
				pout = pin + paircomputersize;
				pThis->make_threads_blocks(paircomputersize / 2.0, cthreads, cblocks);
				recursive_sum_kernel<<<cblocks, cthreads>>>(pin, pout, paircomputersize);
				paircomputersize >>= 1;									
				pin = pout;
			}
			int totalpairs = 0;
			THROW_ON_CUDA_ERR(cudaMemcpy(&totalpairs, pin, sizeof(int), cudaMemcpyDeviceToHost))
			if (totalpairs > 0)
			{
				WISE_ALLOC(pPairs, sizeof(IntPair) * totalpairs)
				{
					pThis->make_threads_blocks(totalpairs, cthreads, cblocks);
					split_and_index_kernel<<<cblocks, cthreads>>>(pin, depth + 1, pPairs, totalpairs);					
				}
			}
			_CUDA_THROW_ERR_
			return totalpairs;
		}

		void PrismMapTracker::ClusterChainer::ParallelMax(int *pdata, int total, int *pmax)
		{
			cudaError_t err;

			if (total < 1)
			{
				THROW_ON_CUDA_ERR(cudaMemset(pmax, 0, sizeof(int)))
				return;
			}
		
			int t = 1;
			while (t < total) t <<= 1;
			dim3 ithreads, iblocks;
			if (t > 1)
			{
				t >>= 1;
				pThis->make_threads_blocks(t, ithreads, iblocks);
				max_check_kernel<<<iblocks, ithreads>>>(pdata, total, t);
			}
			while ((t >>= 1) >= 1)
			{
				pThis->make_threads_blocks(t, ithreads, iblocks);
				max_kernel<<<iblocks, ithreads>>>(pdata, t);
			}
			if (pdata != pmax)
			{
				THROW_ON_CUDA_ERR(cudaMemcpy(pmax, pdata, sizeof(int), cudaMemcpyDeviceToDevice))
			}
		}

		void PrismMapTracker::ClusterChainer::ParallelSum(int *pdata, int total, int *psum)
		{
			cudaError_t err;

			if (total < 1)
			{
				THROW_ON_CUDA_ERR(cudaMemset(psum, 0, sizeof(int)))
				return;
			}
		
			int t = 1;
			while (t < total) t <<= 1;
			dim3 ithreads, iblocks;
			if (t > 1)
			{
				t >>= 1;
				pThis->make_threads_blocks(t, ithreads, iblocks);
				sum_check_kernel<<<iblocks, ithreads>>>(pdata, total, t);
			}
			while ((t >>= 1) >= 1)
			{
				pThis->make_threads_blocks(t, ithreads, iblocks);
				sum_kernel<<<iblocks, ithreads>>>(pdata, t);
			}
			if (pdata != psum)
			{
				THROW_ON_CUDA_ERR(cudaMemcpy(psum, pdata, sizeof(int), cudaMemcpyDeviceToDevice))
			}
		}

		void PrismMapTracker::ClusterChainer::MultiParallelSum(int *pdata, int total, int multiplicity, int *psum)
		{
			cudaError_t err;

			if (total < 1)
			{
				THROW_ON_CUDA_ERR(cudaMemset(psum, 0, sizeof(int) * multiplicity))
				return;
			}
		
			int t = 1;
			while (t < total) t <<= 1;
			dim3 ithreads, iblocks;
			if (t > 1)
			{
				t >>= 1;
				pThis->make_threads_blocks(t, ithreads, iblocks);
				iblocks.z = multiplicity;
				sum_check_multiple_kernel<<<iblocks, ithreads>>>(pdata, total, t);
			}
			while ((t >>= 1) >= 1)
			{
				pThis->make_threads_blocks(t, ithreads, iblocks);
				iblocks.z = multiplicity;
				sum_multiple_kernel<<<iblocks, ithreads>>>(pdata, total, t);
			}
			{
				pThis->make_threads_blocks(multiplicity, ithreads, iblocks);
				compact_kernel<<<iblocks, ithreads>>>(pdata, total, multiplicity, psum);
			}
		}

		void PrismMapTracker::ClusterChainer::Reset(SySal::ClusterChainer::Configuration &c, SySal::ImageCorrection &ic, bool istop)
		{
			C = c;
			IC = ic;
			IsTop = istop;
			CurrentView = -1;			
			ThicknessSamples = 0;
			WeakThickness = 0.0;
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))
			WISE_ALLOC(pLastView, sizeof(ChainView) + C.MaxChains * sizeof(IntChain))
			WISE_ALLOC(pThisView, sizeof(ChainView) + C.MaxChains * sizeof(IntChain))
			EXACT_ALLOC(pChMapWnd, sizeof(ChainMapWindow))
		}

		SySal::ClusterChainer::EmulsionFocusInfo PrismMapTracker::ClusterChainer::AddClusters(SySal::IntClusterFile &cf)
		{
			DebugDump::t_Image *pImDebugInfo = 0;
			clock_t time_0 = clock();			
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))

			EmulsionFocusInfo retfi;
			retfi.Valid = false;
			retfi.Top.Z = retfi.Bottom.Z = 0.0;
			retfi.Top.Valid = retfi.Bottom.Valid = false;
			retfi.ZOffset = 0.0;
			int refimg = 0;
			float dz = 0.0f;
			bool isempty = cf.Images() == 0;
			if (isempty == false)
			{
				EmulsionEdge t, b;
				FindEdges(t, b, cf, C.ClusterThreshold, refimg);
				//printf("\nDEBUG ADDCLUSTERS %f %d %f %d %d %d", t.Z, t.Valid, b.Z, b.Valid, C.ClusterThreshold, refimg);
				if (IsTop)
				{
					if (b.Valid) dz = -b.Z;
					else if (t.Valid)
					{
						dz = -(t.Z - pThis->GetThickness());
					}
					else 
					{
						//throw "Cannot work out emulsion reference surface.";
						//isempty = true;
						dz = -cf.StageZ(cf.Images() - 1);
					}
				}
				else
				{
					if (t.Valid) dz = -t.Z;
					else if (b.Valid)
					{
						dz = -(b.Z + pThis->GetThickness());
					}
					else 
					{
						//throw "Cannot work out emulsion reference surface.";
						//isempty = true;
						dz = -cf.StageZ(0);
					}
				}
				retfi.Top = t;
				retfi.Bottom = b;
				retfi.ZOffset = dz;
				retfi.Valid = true;
			}
			CurrentView++;
			TRACE_PROLOG1 printf("\CURRENTVIEW %d %s", CurrentView, IsTop ? "T" : "B");
			if (pThis->m_EnableDebugDump)
			{				
				pThis->m_DebugDump.Free();
				pThis->m_DebugDump.MapValid = false;
				pThis->m_DebugDump.ViewDeltaX = pThis->m_DebugDump.ViewDeltaY = 0;
			}
			int width = cf.Width() * cf.Scale();
			int height = cf.Height() * cf.Scale();
			if (isempty)
			{
				pThis->m_PerformanceCounters.Clusters = 0;
				pThis->m_PerformanceCounters.Chains = 0;
				TRACE_PROLOG2 printf("\nEmpty view found");
				/*printf("\nDEBUG IMAGES %d", cf.Images());
				int img;
				for (img = 0; img < cf.Images(); img++)
					printf("\n\DEBUG IMG %d %f %d", img, cf.StageZ(img), cf.ImageClusterCounts(img));*/
				ChainView *pView = pLastView;
				pLastView = pThisView;
				pThisView = pView;
				ChainView ev;
				ev.PositionX = ev.PositionY = ev.PositionZ = 0;
				ev.DeltaX = ev.DeltaY = ev.DeltaZ = 0;
				if (CurrentView > 1)				
					cudaMemcpy(&ev, pLastView, sizeof(ChainView), cudaMemcpyDeviceToHost);
				ev.Count = 0;
				ev.Reserved[0] = ev.Reserved[1] = ev.Reserved[2] = ev.Reserved[3] = 
				ev.Reserved[4] = ev.Reserved[5] = ev.Reserved[6] = ev.Reserved[7] = 0;
				cudaMemcpy(pThisView, &ev, sizeof(ChainView), cudaMemcpyHostToDevice);
				pThis->SendViewsToTracker(CurrentView, (((int)width) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronX()), (((int)height) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronY()), pLastView, pThisView);
				return retfi;
			}
			float dxdz = (cf.StageX(cf.Images() - 1) - cf.StageX(0)) / (cf.StageZ(cf.Images() - 1) - cf.StageZ(0));
			float dydz = (cf.StageY(cf.Images() - 1) - cf.StageY(0)) / (cf.StageZ(cf.Images() - 1) - cf.StageZ(0));
			//printf("\nZ adjustment: %f\nDXDZ: %f\nDYDZ: %f", dz, dxdz, dydz);


			int img;
			HOST_WISE_ALLOC(pFixedZs, sizeof(double) * cf.Images());
			if (cf.Images() < 5)
			{
				for (img = 0; img < cf.Images(); img++)
					pFixedZs[img] = cf.StageZ(img);	
			}
			else
			{
				for (img = 1; img < cf.Images(); img++)
					pFixedZs[img - 1] = cf.StageZ(img) - cf.StageZ(img - 1);
				double maxd = pFixedZs[0];
				double mind = pFixedZs[0];
				for (img = 1; img < cf.Images() - 1; img++)
					if (pFixedZs[img] < mind) mind = pFixedZs[img];
					else if (pFixedZs[img] > maxd) maxd = pFixedZs[img];
				double sum = 0.0;
				for (img = 0; img < cf.Images() - 1; img++)
					if (pFixedZs[img] > mind && pFixedZs[img] < maxd)
						sum += pFixedZs[img];
				sum /= (cf.Images() - 3);
				pFixedZs[0] = cf.StageZ(0);
				for (img = 1; img < cf.Images(); img++)
					pFixedZs[img] = cf.StageZ(0) + img * sum;
			}
			int cellsize = C.CellSize;
			int ncellsx = (width / cellsize) + 1;
			int ncellsy = (height / cellsize) + 1;

			WISE_ALLOC(pCurv, 2 * sizeof(int) * (width + height));
			WISE_ALLOC(pCells, sizeof(Cell) * ncellsx * ncellsy);
			WISE_ALLOC(pCellContents, sizeof(IntCluster *) * ncellsx * ncellsy * C.MaxCellContent);
			//printf("\nDEBUG-CELLS %d %d %d", ncellsx, ncellsy, ncellsx * ncellsy * sizeof(Cell));
			THROW_ON_CUDA_ERR(cudaMemset(pCells, 0, ncellsx * ncellsy * sizeof(Cell)));
			THROW_ON_CUDA_ERR(cudaMemset(pCellContents, 0, sizeof(IntCluster *) * ncellsx * ncellsy * C.MaxCellContent));
			int totalsize = cf.TotalSize;
			WISE_ALLOC(pClusterData, totalsize);
			THROW_ON_CUDA_ERR(cudaMemcpy(pClusterData, cf.pData, totalsize, cudaMemcpyHostToDevice));
			_CUDA_THROW_ERR_;
			int totalclusters = 0;	
			for (img = 0; img < cf.Images(); img++)
				totalclusters += cf.pImageClusterCounts[img];
			pThis->m_PerformanceCounters.Clusters = totalclusters;
			IntCluster *pImagesBase = (IntCluster *)(void *)(((char *)cf.pClusters - (char *)cf.pData) + (char *)pClusterData);
			IntCluster *pImageNext;
			WISE_ALLOC(pChains, totalclusters * sizeof(IntChain));
			WISE_ALLOC(pClusterPos, totalclusters * 3 * sizeof(short));
			short *pClusterXs = pClusterPos;
			short *pClusterYs = pClusterXs + totalclusters;
			short *pClusterZs = pClusterYs + totalclusters;
			WISE_ALLOC(pClusterChains, sizeof(IntCluster *) * totalclusters);
			THROW_ON_CUDA_ERR(cudaMemset(pClusterChains, 0, sizeof(IntCluster *) * totalclusters));
			WISE_ALLOC(pChainCounts, sizeof(int) * pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount * 2);
			int *pCompactChainCounts = pChainCounts + pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount;
			int *pCurvX, *pCurvY, *pZCurvX, *pZCurvY;

			int deltasZ = 0;
			int deltasXYZ = 0;
			int refinedeltasXY = 0;
			int refinedeltasZ = 0;
			int refinedeltasXYZ = 0;
			int best_matches = 0;

			ChainView *pView = pLastView;
			pLastView = pThisView;
			pThisView = pView;
			IntChain *pCompactChains = (IntChain *)(void *)((char *)(void *)pThisView + sizeof(ChainView));	
			THROW_ON_CUDA_ERR(cudaMemset((int *)(void *)pThisView->Reserved, 0, sizeof(int)));			

			WISE_ALLOC(pCurv, 2 * sizeof(int) * (width + height));
			pCurvX = pCurv;
			pCurvY = pCurv + width;
			pZCurvX = pCurvY + height;
			pZCurvY = pZCurvX + width;

			dim3 ithreads, iblocks;
			{		
				pThis->make_threads_blocks(width, ithreads, iblocks);
				curvaturemap_kernel<<<iblocks, ithreads>>>(pCurvX, pZCurvX, width, (IC.XYCurvature * (1 << XY_CURVATURE_SHIFT) * (cf.PixMicronX() * cf.PixMicronX()) / (cf.Scale() * cf.Scale())), (IC.ZCurvature * (1 << (Z_CURVATURE_SHIFT + Z_SCALE_SHIFT)) * (cf.PixMicronX() * cf.PixMicronX()) / (cf.Scale() * cf.Scale())) );
				_CUDA_THROW_ERR_
			}
			{		
				pThis->make_threads_blocks(height, ithreads, iblocks);
				curvaturemap_kernel<<<iblocks, ithreads>>>(pCurvY, pZCurvY, height, (IC.XYCurvature * (1 << XY_CURVATURE_SHIFT) * (cf.PixMicronY() * cf.PixMicronY()) / (cf.Scale() * cf.Scale())), (IC.ZCurvature * (1 << (Z_CURVATURE_SHIFT + Z_SCALE_SHIFT)) * (cf.PixMicronX() * cf.PixMicronX()) / (cf.Scale() * cf.Scale())) );
				_CUDA_THROW_ERR_
			}

			int deltasX = (C.ClusterMapMaxXOffset / C.ClusterMapCoarseTolerance * 2 + 1);
			int deltasY = (C.ClusterMapMaxYOffset / C.ClusterMapCoarseTolerance * 2 + 1);
			int deltas2 = deltasX * deltasY;
			int refinedeltas = 2 * C.ClusterMapCoarseTolerance / C.ClusterMapFineTolerance + 1;
			int refinedeltas2 = refinedeltas * refinedeltas;
			WISE_ALLOC(pDeltas, sizeof(int) * ((2 * refinedeltas + deltasX + deltasY)));
			WISE_ALLOC(pMapCounts, sizeof(int) * (__max(__max(deltas2, refinedeltas2), __max(deltasXYZ, refinedeltasXYZ)) + 1));
			int *pBest = pMapCounts + __max(__max(deltas2, refinedeltas2), __max(deltasXYZ, refinedeltasXYZ));
			WISE_ALLOC(pStagePos, sizeof(short) * (cf.Images() * 4 + 1));
			short *pStagePosX = pStagePos;
			short *pStagePosY = pStagePosX + cf.Images();
			short *pDeltaStagePosX = pStagePosY + cf.Images();
			short *pDeltaStagePosY = pDeltaStagePosX + cf.Images();	
			short *pDeltaCounter = pDeltaStagePosY + cf.Images();
			HOST_WISE_ALLOC(pHostStagePos, sizeof(short) * (cf.Images() * 4 + 1));
			for (img = 0; img < cf.Images(); img++)
			{
				pHostStagePos[img] =
					(int)(((cf.StageX(img) - cf.StageX(0)) / cf.PixMicronX()) * cf.Scale());			
				pHostStagePos[img + cf.Images()] =
					(int)(((cf.StageY(img) - cf.StageY(0)) / cf.PixMicronY()) * cf.Scale());
				if (img == 0)
				{
					pHostStagePos[img + 2 * cf.Images()] = 0;
					pHostStagePos[img + 3 * cf.Images()] = 0;
				}
				else
				{
					pHostStagePos[img + 2 * cf.Images()] = pHostStagePos[img] - pHostStagePos[img - 1];
					pHostStagePos[img + 3 * cf.Images()] = pHostStagePos[img + cf.Images()] - pHostStagePos[img - 1 + cf.Images()];
				}
			}
			pHostStagePos[4 * cf.Images()] = 0;
			THROW_ON_CUDA_ERR(cudaMemcpy(pStagePos, pHostStagePos, sizeof(short) * (4 * cf.Images() + 1), cudaMemcpyHostToDevice));
			THROW_ON_CUDA_ERR(cudaMemset(pCells, 0, sizeof(Cell) * ncellsx * ncellsy));

			int demagDZ1M;	
			int id = 0;
			IntCluster *pImagesBase1 = pImagesBase;			

			{
				pThis->make_threads_blocks(__max(1, totalclusters), ithreads, iblocks);
				correctcurvature_kernel<<<iblocks, ithreads>>>(pImagesBase, pClusterZs, sin(IC.CameraRotation) * (1 << FRACT_RESCALE_SHIFT), (cos(IC.CameraRotation) - 1) * (1 << FRACT_RESCALE_SHIFT), pCurvX, pCurvY, pZCurvX, pZCurvY, IC.DMagDX * (1 << XY_MAGNIFICATION_SHIFT), IC.DMagDY * (1 << XY_MAGNIFICATION_SHIFT), totalclusters, width / 2, height / 2 );		
				_CUDA_THROW_ERR_
			}	
			{
				pThis->make_threads_blocks(__max(1, cf.ImageClusterCounts(0)), ithreads, iblocks);
				setXYZs_kernel<<<iblocks, ithreads>>>(pClusterXs, pClusterYs, pClusterZs, cf.ImageClusterCounts(0), 0, pStagePosX, pStagePosY, 0);
				_CUDA_THROW_ERR_
			}

			int bestclustermapcount = -1;
			int bestclustermapcount_img = 0;


			for (img = 0; img < cf.Images() - 1; img++)
			{
				if (pThis->m_EnableDebugDump)
				{
					pImDebugInfo = &pThis->m_DebugDump.AddImage();
					pImDebugInfo->Clusters = cf.ImageClusterCounts(img);
					pImDebugInfo->MappingClusters = 0;
					pImDebugInfo->DeltaX = pImDebugInfo->DeltaY = 0;
				}
				else pImDebugInfo = 0;
				pImageNext = pImagesBase + cf.ImageClusterCounts(img);
				demagDZ1M = IC.DMagDZ * (pFixedZs[img + 1] - pFixedZs[img]) * (1 << DEMAG_SHIFT);
				int launches;
				// BEGIN COARSE MAPPING
				{		
					makedeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, C.ClusterMapCoarseTolerance, deltasX, deltasY, pDeltaStagePosX, pDeltaStagePosY, img + 1);
					_CUDA_THROW_ERR_
				}		
				{
					pThis->make_threads_blocks(deltas2, ithreads, iblocks);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, deltas2, pBest);
					_CUDA_THROW_ERR_
				}	
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					int clusterblocksize = (int)ceil((double)cf.ImageClusterCounts(img) / (ithreads.x * iblocks.x * iblocks.y));
					for (int i = 0; i < clusterblocksize; i++)
						maphash_minarea_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), clusterblocksize, i, pCells, pCellContents, cellsize, C.MaxCellContent, ncellsx, ncellsy, C.ClusterMapMinSize);
					_CUDA_THROW_ERR_
				}						

				int totalpairs = 0;
				WISE_ALLOC(pMapClusters, sizeof(IntMapCluster) * cf.ImageClusterCounts(img + 1));
				int itotalmapclusters = cf.ImageClusterCounts(img + 1) / C.ClusterMapSampleDivider;
				if (itotalmapclusters > 0)
				{
					//printf("\nDEBUG Z1 dev %d totalmapclusters %d", pThis->m_DeviceId, itotalmapclusters);
					WISE_ALLOC(pClustersInCell, sizeof(int) * itotalmapclusters);
					{					
						dim3 ithreads, iblocks;
						pThis->make_threads_blocks(__max(1, cf.ImageClusterCounts(img + 1)), ithreads, iblocks);
						trymap2_prepare_clusters_kernel<<<iblocks, ithreads>>>(pImageNext, pMapClusters, cf.ImageClusterCounts(img + 1), C.ClusterMapSampleDivider, C.ClusterMapMinSize, width >> 1, height >> 1, demagDZ1M, pClustersInCell);
						_CUDA_THROW_ERR_
					}
					totalpairs = DistributeClusterPairsToThreads(itotalmapclusters);
				}
				else totalpairs = 0;
				//printf("\nDEBUG Z2 dev %d totalpairs %d", pThis->m_DeviceId, totalpairs);
				WISE_ALLOC(pPairMatchResult, sizeof(int) * __max(1, totalpairs) * deltas2);
				if (totalpairs > 0)
				{						
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);					
					iblocks.z = deltasX * deltasY;
					//printf("\nDEBUG Z3 dev %d  X Y S2 %d %d %d", pThis->m_DeviceId, deltasX, deltasY, deltas2);
					trymap2_shiftmatch_kernel<<<iblocks, ithreads>>>(pMapClusters, pPairs, totalpairs, pDeltas, cellsize, ncellsx, ncellsy, pPairMatchResult, C.ClusterMapCoarseTolerance, pCells, pCellContents, C.MaxCellContent, deltasX);
					_CUDA_THROW_ERRMSG_(cudaErrorUnknown, "HINT: The error may be due to execution timeout. Too many CLUSTERS are being COARSE-mapped. Consider reducing the number of combinations.");
					MultiParallelSum(pPairMatchResult, totalpairs, deltas2, pMapCounts);
					pThis->make_threads_blocks(deltas2, ithreads, iblocks);
					shift_postfixid_kernel<<<iblocks, ithreads>>>(pMapCounts, pMapCounts, deltas2);
					_CUDA_THROW_ERR_
					ParallelMax(pMapCounts, deltas2, pBest);
					_CUDA_THROW_ERR_
				}

				{
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, pBest, sizeof(int), cudaMemcpyDeviceToHost))
					//printf("\nDEBUG D %d %08X", pThis->m_DeviceId, best_matches);
					best_matches = best_matches >> 16;						
				}		
				// END COARSE MAPPING

				// BEGIN FINE MAPPING
				if (best_matches < C.MinClusterMapsValid)
				{
					makedeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, 0, deltasX, deltasY, pDeltaStagePosX, pDeltaStagePosY, img + 1);
					_CUDA_THROW_ERR_
				}
				{		
					makedeltas_fromshift_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas + deltasX + deltasY, (best_matches < C.MinClusterMapsValid) ? 0 : C.ClusterMapFineTolerance, refinedeltas, refinedeltas, pDeltas, pBest, deltasX);
					_CUDA_THROW_ERR_
				}
				{
					pThis->make_threads_blocks(refinedeltas2, ithreads, iblocks);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, refinedeltas2, pBest);
					_CUDA_THROW_ERR_
				}
				{
					if (best_matches > bestclustermapcount)
					{
						bestclustermapcount = best_matches;
						bestclustermapcount_img = img;
					}		

					if (best_matches > 0)
					{					
						WISE_ALLOC(pPairMatchResult, sizeof(int) * __max(1, totalpairs) * refinedeltas2);
						dim3 ithreads, iblocks;
						pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);						
						iblocks.z = refinedeltas2;
						//printf("\nDEBUG Z3 dev %d  X Y S2 %d %d %d", pThis->m_DeviceId, refinedeltas, refinedeltas, refinedeltas2);
						trymap2_shiftmatch_kernel<<<iblocks, ithreads>>>(pMapClusters, pPairs, totalpairs, pDeltas + deltasX + deltasY, cellsize, ncellsx, ncellsy, pPairMatchResult, C.ClusterMapCoarseTolerance, pCells, pCellContents, C.MaxCellContent, refinedeltas);
						_CUDA_THROW_ERRMSG_(cudaErrorUnknown, "HINT: The error may be due to execution timeout. Too many CLUSTERS are being FINE-mapped. Consider reducing the number of combinations.");
						MultiParallelSum(pPairMatchResult, totalpairs, refinedeltas2, pMapCounts);
						pThis->make_threads_blocks(refinedeltas2, ithreads, iblocks);
						shift_postfixid_kernel<<<iblocks, ithreads>>>(pMapCounts, pMapCounts, refinedeltas2);
						_CUDA_THROW_ERR_
						ParallelMax(pMapCounts, refinedeltas2, pBest);
						_CUDA_THROW_ERR_
					}

				}	
				{
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, pBest, sizeof(int), cudaMemcpyDeviceToHost));
					//printf("\nDEBUG E %d %d %08X", pThis->m_DeviceId, img, best_matches);
					best_matches = best_matches >> 16;
					//printf("\n%d %d", img, best_matches);
				}				
				// END FINE MAPPING

				// BEGIN FINAL MAPPING
				if (best_matches > 0)
				{
					{
						dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
						clearhash_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), cf.ImageClusterCounts(img) / (pThis->m_Prop.maxThreadsPerBlock + pThis->m_Prop.multiProcessorCount) + 1, pCells, cellsize, ncellsx, ncellsy);
						_CUDA_THROW_ERR_
					}
					{
						dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
						dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
						int clusterblocksize = (int)ceil((double)cf.ImageClusterCounts(img) / (ithreads.x * iblocks.x * iblocks.y));
						for (int i = 0; i < clusterblocksize; i++)
							maphash_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), clusterblocksize, i, pCells, pCellContents, cellsize, C.MaxCellContent, ncellsx, ncellsy);			
						_CUDA_THROW_ERR_
					}				
					{		
						makefinaldeltas_fromshift_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pDeltas, (best_matches > C.MinClusterMapsValid) ? C.ClusterMapFineTolerance : 0, pDeltas + deltasX + deltasY, pBest, refinedeltas, pStagePosX, pStagePosY, pDeltaStagePosX, pDeltaStagePosY, img + 1, cf.Images(), pDeltaCounter);
						_CUDA_THROW_ERR_			
					}
					if (pImDebugInfo)
					{
						pImDebugInfo->MappingClusters = best_matches;
						cudaMemcpy(&pImDebugInfo->DeltaX, &pDeltas[0], sizeof(int), cudaMemcpyDeviceToHost);
						cudaMemcpy(&pImDebugInfo->DeltaY, &pDeltas[1], sizeof(int), cudaMemcpyDeviceToHost);
					}
					{
						pThis->make_threads_blocks(__max(1, cf.ImageClusterCounts(img + 1)), ithreads, iblocks);
						id = pImageNext - pImagesBase1;
						setXYZs_kernel<<<iblocks, ithreads>>>(pClusterXs + id, pClusterYs + id, pClusterZs + id, cf.ImageClusterCounts(img + 1), img + 1, pStagePosX, pStagePosY, ((pFixedZs[img + 1] /*- pFixedZs[0]*/ + dz) * (1 << Z_SCALE_SHIFT)) );
					}	
					{
						resetcounts_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pMapCounts, 1, pBest);
						_CUDA_THROW_ERR_
					}

					{
						dim3 ithreads, iblocks;
						pThis->make_threads_blocks(__max(1, cf.ImageClusterCounts(img + 1)), ithreads, iblocks);
						trymap2_prepare_clusters_kernel<<<iblocks, ithreads>>>(pImageNext, pMapClusters, cf.ImageClusterCounts(img + 1), 1, 0, width >> 1, height >> 1, demagDZ1M);
						_CUDA_THROW_ERR_
						pThis->make_threads_blocks(__max(1, cf.ImageClusterCounts(img + 1)), ithreads, iblocks);
						trymap2_shift_kernel<<<iblocks, ithreads>>>(pMapClusters, cf.ImageClusterCounts(img + 1), pDeltas, pDeltas + 1, cellsize);
						WISE_ALLOC(pMatchMap, sizeof(int) * cf.ImageClusterCounts(img + 1))
						WISE_ALLOC(pClustersInCell, sizeof(int) * cf.ImageClusterCounts(img + 1));
						finalmap_cell_kernel<<<iblocks, ithreads>>>(pMapClusters, cf.ImageClusterCounts(img + 1), pCells, pClustersInCell, ncellsx, ncellsy);					
						_CUDA_THROW_ERR_
						int totalpairs = DistributeClusterPairsToThreads(cf.ImageClusterCounts(img + 1));
						if (totalpairs > 0)
						{
							pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);
							WISE_ALLOC(pPairMatchResult, sizeof(int) * totalpairs)
							WISE_ALLOC(pMatchMap, sizeof(int) * totalpairs)
							finalmap_match_kernel<<<iblocks, ithreads>>>(pMapClusters, pPairs, totalpairs, pPairMatchResult, pMatchMap, C.ClusterMapFineAcceptance, pCells, pCellContents, C.MaxCellContent, ncellsx, ncellsy);
							_CUDA_THROW_ERR_
							pThis->make_threads_blocks(__max(1, cf.ImageClusterCounts(img + 1)), ithreads, iblocks);
							finalmap_optimize_kernel<<<iblocks, ithreads>>>(pImagesBase, pMapClusters, pImageNext - pImagesBase1, cf.ImageClusterCounts(img + 1), pPairMatchResult, pMatchMap, pCellContents, pClusterChains);
							_CUDA_THROW_ERR_
						}
					}					

				}
				// END FINAL MAPPING		
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					clearhash_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), cf.ImageClusterCounts(img) / (pThis->m_Prop.maxThreadsPerBlock + pThis->m_Prop.multiProcessorCount) + 1, pCells, cellsize, ncellsx, ncellsy);
					_CUDA_THROW_ERR_
				}
				pImagesBase = pImageNext;
			}
			average_deltas<<<dim3(1,1,1), dim3(1,1,1)>>>(pDeltaStagePosX, pDeltaStagePosY, cf.Images(), pDeltaCounter);

			int lastcounts = 0;
			if (CurrentView > 0)
			{				
				makechainwindow_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pChMapWnd, pStagePosX, pStagePosY, cf.Images(), width, height, cf.PixMicronX() / cf.Scale() * (1 << XY_SCALE_SHIFT), cf.PixMicronY() / cf.Scale() * (1 << XY_SCALE_SHIFT), ncellsx * ncellsy, C.ChainMapXYCoarseTolerance, pLastView, pCells, pCellContents, C.MaxCellContent, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), pLastView);
				cudaMemcpy(&lastcounts, &pLastView->Count, sizeof(int), cudaMemcpyDeviceToHost);
				_CUDA_THROW_ERR_
			}
		
			{
				WISE_ALLOC(pClustersInCell, totalclusters * sizeof(int));
				dim3 ithreads, iblocks;				
				pThis->make_threads_blocks(__max(1, totalclusters), ithreads, iblocks);
				makechain_kernel<<<iblocks, ithreads>>>(
					pImagesBase1, totalclusters, width >> 1, height >> 1, pClusterXs, pClusterYs, pClusterZs, 
					IC.XSlant * cf.PixMicronX() / cf.Scale() * (1 << (Z_SCALE_SHIFT + SLOPE_SHIFT)), 
					IC.YSlant * cf.PixMicronY() / cf.Scale() * (1 << (Z_SCALE_SHIFT + SLOPE_SHIFT)), 
					pClusterChains, C.MinClustersPerChain, C.MinVolumePerChain, cf.PixMicronX() / cf.Scale(), cf.PixMicronY() / cf.Scale(), 
					cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), pChains, CurrentView, pClustersInCell, pDeltaStagePosX, pDeltaStagePosY);
				_CUDA_THROW_ERR_
				if (CurrentView == 0 || lastcounts == 0)
				{
					if (totalclusters > C.MaxChains) totalclusters = C.MaxChains;
					int itotalpairs = DistributeClusterPairsToThreads(totalclusters);
					pThis->make_threads_blocks(__max(1, itotalpairs), ithreads, iblocks);
					compactchains_kernel<<<iblocks, ithreads>>>(pCompactChains, pChains, pPairs, itotalpairs, pThisView, C.MaxChains);
					_CUDA_THROW_ERR_
/*
					{
						printf("\nDEBUG chains: %d", itotalpairs);
						FILE *f = fopen("c:\\temp\\t.txt", "at");
						for (int __i = 0; __i < itotalpairs; __i++)
						{
							if (__i > 0) fprintf(f, "\n");
							IntChain a;
							cudaMemcpy(&a, pCompactChains + __i, sizeof(IntChain), cudaMemcpyDeviceToHost);
							fprintf(f, "%d %d %d %d %d %d", CurrentView, a.AvgX, a.AvgY, a.AvgZ, a.Clusters, a.Volume);
						}
						fclose(f);
					}				
*/
				}
/*
				ParallelSum(pClustersInCell, totalclusters, pClustersInCell);
				int a = 0;
				cudaMemcpy(&a, pClustersInCell, sizeof(int), cudaMemcpyDeviceToHost);
				printf("\nDEBUG CHAINS %d", a);
				throw "DEBUG END";
*/
			}

			if (lastcounts > 0)
			{

				{
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, lastcounts), ithreads, iblocks);					
					maphashchain_kernel<<<iblocks, ithreads>>>(pLastView, pChMapWnd, C.ChainMapSampleDivider);
					_CUDA_THROW_ERR_
				}
				// BEGIN COARSE CHAIN MAPPING

				deltasX = (2 * C.ChainMapMaxXOffset / C.ChainMapXYCoarseTolerance + 1);
				deltasY = (2 * C.ChainMapMaxYOffset / C.ChainMapXYCoarseTolerance + 1);
				deltasZ = (2 * C.ChainMapMaxZOffset / C.ChainMapZCoarseTolerance + 1);
				deltasXYZ = deltasX * deltasY * deltasZ;
				refinedeltasXY = (2 * C.ChainMapXYCoarseTolerance / C.ChainMapXYFineTolerance) + 1;
				refinedeltasZ = (2 * C.ChainMapZCoarseTolerance / C.ChainMapZFineTolerance) + 1;
				refinedeltasXYZ = refinedeltasXY * refinedeltasXY * refinedeltasZ;
				WISE_ALLOC(pDeltas, sizeof(int) * ((deltasX + deltasY + 3 * deltasZ + 2 * refinedeltasXY + refinedeltasZ)));
				WISE_ALLOC(pMapCounts, sizeof(int) * __max(deltasXYZ, refinedeltasXYZ) * (pThis->m_Prop.multiProcessorCount * pThis->m_Prop.maxThreadsPerBlock + 1) );
				THROW_ON_CUDA_ERR(cudaMemset(pMapCounts, 0, _MEM_(pMapCounts)));
				pBest = (int *)(void *)(&pThisView->Reserved);

				{		
					makechaindeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, C.ChainMapXYCoarseTolerance, C.ChainMapZCoarseTolerance, deltasX, deltasY, deltasZ, pLastView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), IC.XSlant, IC.YSlant, 0, 0/*dxdz, dydz*/);
					_CUDA_THROW_ERR_
				}
				{
					pThis->make_threads_blocks(deltasXYZ, ithreads, iblocks);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, deltasXYZ, pBest);
					_CUDA_THROW_ERR_
				}

				int totalpairs = 0;
				WISE_ALLOC(pChains, sizeof(IntChain) * totalclusters);
				WISE_ALLOC(pMapChains, sizeof(IntMapChain) * totalclusters);
				WISE_ALLOC(pClustersInCell, sizeof(int) * totalclusters);				
				{					
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, totalclusters), ithreads, iblocks);
					trymapchain_prepare_chains_kernel<<<iblocks, ithreads>>>(pChains, pMapChains, totalclusters, C.ChainMapMinVolume, pClustersInCell);
					_CUDA_THROW_ERR_
				}
				totalpairs = DistributeClusterPairsToThreads(totalclusters);				
				if (totalpairs > 0)
				{						
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);
					iblocks.z = deltasX * deltasY;			
					TRACE_PROLOG2 printf("\nDEBUG CLUSTERMAP totalpairs %d ithreads %d %d %d iblocks %d %d %d", totalpairs, ithreads.x, ithreads.y, ithreads.z, iblocks.x, iblocks.y,	iblocks.z);
					trymapchain_shiftmatch_kernel<<<iblocks, ithreads>>>(pMapChains, pPairs, totalpairs, pMapCounts, 
						pDeltas, pChMapWnd, C.ChainMapXYCoarseTolerance, deltasZ, C.ChainMapZCoarseTolerance, deltasX);
					_CUDA_THROW_ERRMSG_(cudaErrorUnknown, "HINT: The error may be due to execution timeout. Too many CHAINS are being COARSE-mapped. Consider reducing the number of combinations.");
					pThis->make_threads_blocks(deltasXYZ, ithreads, iblocks);
					shift_postfixid_kernel<<<iblocks, ithreads>>>(pMapCounts, pMapCounts, deltasXYZ);
					_CUDA_THROW_ERR_
					ParallelMax(pMapCounts, deltasXYZ, pBest);
					_CUDA_THROW_ERR_
				}

				{
					int mapc, best;
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, (int *)(void *)(&pThisView->Reserved), sizeof(int), cudaMemcpyDeviceToHost));
#ifdef ENABLE_COARSE_CHAIN_MAPPING_LOG
					printf("\nDEBUG %08X", best_matches);					
					{
						best = best_matches & 0xffff;
						int a;
						cudaMemcpy(&a, pDeltas + best % deltasX, sizeof(int), cudaMemcpyDeviceToHost);
						printf("\nDEBUG DX: %d", a);
						cudaMemcpy(&a, pDeltas + deltasX + (best / deltasX) % deltasY, sizeof(int), cudaMemcpyDeviceToHost);
						printf("\nDEBUG DY: %d", a);
						cudaMemcpy(&a, pDeltas + deltasX + deltasY + (best / deltasX / deltasY) % deltasZ, sizeof(int), cudaMemcpyDeviceToHost);
						printf("\nDEBUG DZ: %d", a);
					}					
#endif
					mapc = best_matches >> 16;
					if (mapc < C.MinChainMapsValid) best = deltasX / 2 + (deltasY / 2) * deltasX + (deltasZ / 2) * deltasX * deltasY;
					else best = best_matches & 0xffff;
					best_matches = mapc;


				}					
				// END COARSE CHAIN MAPPING

				// BEGIN FINE CHAIN MAPPING
				if (best_matches < C.MinChainMapsValid)
				{
					TRACE_PROLOG1 printf("\nBad chain mapping (%d/%d), switching to default", best_matches, C.MinChainMapsValid);
					makechaindeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, 0, 0, deltasX, deltasY, deltasZ, pLastView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), IC.XSlant, IC.YSlant, dxdz, dydz);
					_CUDA_THROW_ERR_
				}
				{		
					makechaindeltas_fromshift_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas + deltasX + deltasY + 3 * deltasZ, (best_matches < C.MinChainMapsValid) ? 0 : C.ChainMapXYFineTolerance, 0, refinedeltasXY, refinedeltasXY, refinedeltasZ, pDeltas, pBest, deltasX, deltasY, deltasZ);
					_CUDA_THROW_ERR_
				}
				{
					pThis->make_threads_blocks(refinedeltasXYZ, ithreads, iblocks);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, refinedeltasXYZ, (int *)(void *)(&pThisView->Reserved));			
					_CUDA_THROW_ERR_
				}

				if (totalpairs > 0)
				{						
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);					
					iblocks.z = refinedeltasXY * refinedeltasXY;
					trymapchain_shiftmatch_kernel<<<iblocks, ithreads>>>(pMapChains, pPairs, totalpairs, pMapCounts, 
						pDeltas + deltasX + deltasY + 3 * deltasZ, pChMapWnd, C.ChainMapXYFineAcceptance, refinedeltasZ, C.ChainMapZFineAcceptance, refinedeltasXY);
					_CUDA_THROW_ERRMSG_(cudaErrorUnknown, "HINT: The error may be due to execution timeout. Too many CHAINS are being FINE-mapped. Consider reducing the number of combinations.");					
					pThis->make_threads_blocks(refinedeltasXYZ, ithreads, iblocks);					
					shift_postfixid_kernel<<<iblocks, ithreads>>>(pMapCounts, pMapCounts, refinedeltasXYZ);
					_CUDA_THROW_ERR_
					ParallelMax(pMapCounts, refinedeltasXYZ, pBest);
					_CUDA_THROW_ERR_
				}

				{
					int mapc, best;
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, pBest, sizeof(int), cudaMemcpyDeviceToHost))
#ifdef ENABLE_FINE_CHAIN_MAPPING_LOG
					printf("\nDEBUG %08X", best_matches);
					{
						best = best_matches & 0xffff;
						int a;
						cudaMemcpy(&a, pDeltas + deltasX + deltasY + 3 * deltasZ + (best % refinedeltasXY), sizeof(int), cudaMemcpyDeviceToHost);
						printf("\nDEBUG DX: %d", a);
						cudaMemcpy(&a, pDeltas + deltasX + deltasY + 3 * deltasZ + (refinedeltasXY + (best / refinedeltasXY) % refinedeltasXY), sizeof(int), cudaMemcpyDeviceToHost);
						printf("\nDEBUG DY: %d", a);
						cudaMemcpy(&a, pDeltas + deltasX + deltasY + 3 * deltasZ + (refinedeltasXY + refinedeltasXY + (best / refinedeltasXY / refinedeltasXY) % refinedeltasZ), sizeof(int), cudaMemcpyDeviceToHost);
						printf("\nDEBUG DZ: %d", a);
					}					
#endif
					mapc = best_matches >> 16;
					best = best_matches & 0xffff;
					best_matches = mapc;								
				}				

				// END FINE CHAIN MAPPING

				{					
					make_finalchainshift_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pDeltas, pDeltas + deltasX + deltasY + 3 * deltasZ, pBest, refinedeltasXY);
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, totalclusters), ithreads, iblocks);
					trymapchain_prepare_chains_kernel<<<iblocks, ithreads>>>(pChains, pMapChains, totalclusters, 1, pClustersInCell);
					_CUDA_THROW_ERR_
				}
				totalpairs = DistributeClusterPairsToThreads(totalclusters);
				if (totalpairs > 0)
				{						
					dim3 ithreads, iblocks;					
					pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);					
					finalmapchain_cell_kernel<<<iblocks, ithreads>>>(pMapChains, pPairs, totalpairs, pDeltas, pChMapWnd, pClustersInCell);
					_CUDA_THROW_ERR_
					totalpairs = DistributeClusterPairsToThreads(totalclusters);
					if (totalpairs > 0)
					{
						pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);
						finalmapchain_match_kernel<<<iblocks, ithreads>>>(pChains, pMapChains, pPairs, totalpairs, pChMapWnd, C.ChainMapXYFineAcceptance, C.ChainMapZFineAcceptance);
						_CUDA_THROW_ERR_
						if (pThis->m_EnableDebugDump)
						{
							pThis->m_DebugDump.MapValid = true;
							cudaMemcpy(&pThis->m_DebugDump.ViewDeltaX, &pDeltas[0], sizeof(int), cudaMemcpyDeviceToHost);
							cudaMemcpy(&pThis->m_DebugDump.ViewDeltaY, &pDeltas[1], sizeof(int), cudaMemcpyDeviceToHost);
						}
					}
				}
				{
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, totalclusters), ithreads, iblocks);
					finalmapchain_filter_kernel<<<iblocks, ithreads>>>(pChains, totalclusters, pClustersInCell);
					_CUDA_THROW_ERR_
					totalpairs = DistributeClusterPairsToThreads(totalclusters);
					pThis->m_PerformanceCounters.Chains = totalpairs;
					pThis->make_threads_blocks(__max(1, totalpairs), ithreads, iblocks);
					compactchains_kernel<<<iblocks, ithreads>>>(pCompactChains, pChains, pPairs, totalpairs, pThisView, C.MaxChains);
					_CUDA_THROW_ERRMSG_(cudaErrorUnknown, "HINT: memory corruption may have occurred before chain compacting")
/*
					{
						printf("\nDEBUG chains: %d", totalpairs);
						FILE *f = fopen("c:\\temp\\t.txt", "at");
						for (int __i = 0; __i < totalpairs; __i++)
						{
							// if (__i > 0)
								fprintf(f, "\n");
							IntChain a;
							cudaMemcpy(&a, pCompactChains + __i, sizeof(IntChain), cudaMemcpyDeviceToHost);
							fprintf(f, "%d %d %d %d %d %d", CurrentView, a.AvgX, a.AvgY, a.AvgZ, a.Clusters, a.Volume);
						}
						fclose(f);
					}				
*/
				}


				{
					dim3 ithreads, iblocks;
					pThis->make_threads_blocks(__max(1, lastcounts), ithreads, iblocks);					
					clearhashchain2_kernel<<<iblocks, ithreads>>>(pLastView, pChMapWnd, C.ChainMapSampleDivider);
					_CUDA_THROW_ERR_
					/*{
						int xyz[3];
						cudaMemcpy(xyz, pDeltas, sizeof(int) * 3, cudaMemcpyDeviceToHost);
						printf("\nDEBUG XYZ %d %d %d %d", pThis->m_DeviceId, xyz[0], xyz[1], xyz[2]);
					}*/
					negshift_viewchains_kernel<<<iblocks, ithreads>>>(pLastView, pDeltas);
					_CUDA_THROW_ERR_
				}

			}

			{
				//if (CurrentView == 0)
				setchainviewheader_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pChainMapHeader, pThisView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), cf.StageZ(0) * (1 << Z_SCALE_SHIFT), 0, 0, 0, 0);
				//else
					//setchainviewheader_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pChainMapHeader, pThisView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), cf.StageZ(0) * (1 << Z_SCALE_SHIFT), pDeltas + deltasX + deltasY + 3 * deltasZ, refinedeltasXY, refinedeltasZ, pBest);
				_CUDA_THROW_ERR_
			}			

			/*KRYSS 20140116: remove this line to allow a valid first view
				if (CurrentView > 0) 
				*/
			clock_t time_1 = clock();
			pThis->SendViewsToTracker(CurrentView, (((int)width) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronX()), (((int)height) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronY()), pLastView, pThisView);
			clock_t time_2 = clock();

			pThis->m_PerformanceCounters.MapTimeMS = ((1000.0f) * (time_1 - time_0)) / CLOCKS_PER_SEC;
			pThis->m_PerformanceCounters.TrackTimeMS = ((1000.0f) * (time_2 - time_1)) / CLOCKS_PER_SEC;

			return retfi;
		}


	};
};