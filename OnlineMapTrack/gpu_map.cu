#include "gpu_incremental_map_track.h"
#include "gpu_map_kernels.h"
#include "gpu_defines.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <math.h>

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
			if (topout > 0 && topout < cf.Images() - 1)
			{
				if (cf.StageZ(topout) == cf.StageZ(topout + 1)) t.Z = 0.5f * (cf.StageZ(topout) + cf.StageZ(topout + 1));
				else t.Z = cf.StageZ(topout) + (cf.StageZ(topout + 1) - cf.StageZ(topout)) / (cf.ImageClusterCounts(topout + 1) - cf.ImageClusterCounts(topout)) * (threshold - cf.ImageClusterCounts(topout));
				t.Valid = true;
				retval++;
			}
			if (bottomout > 0 && bottomout < cf.Images() - 1)
			{
				if (cf.StageZ(bottomout) == cf.StageZ(bottomout - 1)) b.Z = 0.5f * (cf.StageZ(bottomout) + cf.StageZ(bottomout - 1));
				else b.Z = cf.StageZ(bottomout) + (cf.StageZ(bottomout - 1) - cf.StageZ(bottomout)) / (cf.ImageClusterCounts(bottomout - 1) - cf.ImageClusterCounts(bottomout)) * (threshold - cf.ImageClusterCounts(bottomout));
				b.Valid = true;
				retval++;
			}
			refimg = (topout + bottomout) / 2;
			return retval;
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
			CTOR_INIT(pThisView)
		{
			ThicknessSamples = 0;
			pThicknessSamples = (double *)malloc(sizeof(double) * 32);

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
		}


		void PrismMapTracker::ClusterChainer::Reset(SySal::ClusterChainer::Configuration &c, SySal::ImageCorrection &ic, bool istop)
		{
			C = c;
			IC = ic;
			IsTop = istop;
			CurrentView = -1;
			ThicknessSamples = 0;
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))
				WISE_ALLOC(pLastView, sizeof(ChainView) + C.MaxChains * sizeof(IntChain))
				WISE_ALLOC(pThisView, sizeof(ChainView) + C.MaxChains * sizeof(IntChain))
				EXACT_ALLOC(pChMapWnd, sizeof(ChainMapWindow))
		}

		int PrismMapTracker::ClusterChainer::AddClusters(SySal::IntClusterFile &cf)
		{
			cudaError_t err;
			THROW_ON_CUDA_ERR(cudaSetDevice(pThis->m_DeviceId))

				int refimg = 0;
			float dz = 0.0f;
			{
				EmulsionEdge t, b;
				FindEdges(t, b, cf, C.ClusterThreshold, refimg);
				if (IsTop)
				{
					if (b.Valid) dz = -b.Z;
					else if (t.Valid)
					{
						dz = -(t.Z - pThis->GetThickness());
					}
					else throw "Cannot work out emulsion reference surface.";
				}
				else
				{
					if (t.Valid) dz = -t.Z;
					else if (b.Valid)
					{
						dz = -(b.Z + pThis->GetThickness());
					}
					else throw "Cannot work out emulsion reference surface.";
				}
			}
			CurrentView++;
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
			int width = cf.Width() * cf.Scale();
			int height = cf.Height() * cf.Scale();
			int cellsize = C.CellSize;
			int ncellsx = (width / cellsize) + 1;
			int ncellsy = (height / cellsize) + 1;

			WISE_ALLOC(pCurv, 2 * sizeof(int) * (width + height));
			WISE_ALLOC(pCells, sizeof(Cell) * ncellsx * ncellsy);
			WISE_ALLOC(pCellContents, sizeof(IntCluster *) * ncellsx * ncellsy * C.MaxCellContent);
			THROW_ON_CUDA_ERR(cudaMemset(pCells, 0, ncellsx * ncellsy * sizeof(Cell)));
			THROW_ON_CUDA_ERR(cudaMemset(pCellContents, 0, sizeof(IntCluster *) * ncellsx * ncellsy * C.MaxCellContent));
			int totalsize = cf.TotalSize;
			WISE_ALLOC(pClusterData, totalsize);
			THROW_ON_CUDA_ERR(cudaMemcpy(pClusterData, cf.pData, totalsize, cudaMemcpyHostToDevice));
			_CUDA_THROW_ERR_;
			int totalclusters = 0;	
			for (img = 0; img < cf.Images(); img++)
				totalclusters += cf.pImageClusterCounts[img];
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
			cudaMemset((int *)(void *)pThisView->Reserved, 0, sizeof(int));
			_CUDA_THROW_ERR_;

			WISE_ALLOC(pCurv, 2 * sizeof(int) * (width + height));
			pCurvX = pCurv;
			pCurvY = pCurv + width;
			pZCurvX = pCurvY + height;
			pZCurvY = pZCurvX + width;
			{		
				dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks = dim3(width / pThis->m_Prop.maxThreadsPerBlock + 1, 1, 1);
				curvaturemap_kernel<<<iblocks, ithreads>>>(pCurvX, pZCurvX, width, (IC.XYCurvature * (1 << XY_CURVATURE_SHIFT) * (cf.PixMicronX() * cf.PixMicronX()) / (cf.Scale() * cf.Scale())), (IC.ZCurvature * (1 << (Z_CURVATURE_SHIFT + Z_SCALE_SHIFT)) * (cf.PixMicronX() * cf.PixMicronX()) / (cf.Scale() * cf.Scale())) );
				_CUDA_THROW_ERR_
			}
			{		
				dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks = dim3(height / pThis->m_Prop.maxThreadsPerBlock + 1, 1, 1);
				curvaturemap_kernel<<<iblocks, ithreads>>>(pCurvY, pZCurvY, height, (IC.XYCurvature * (1 << XY_CURVATURE_SHIFT) * (cf.PixMicronY() * cf.PixMicronY()) / (cf.Scale() * cf.Scale())), (IC.ZCurvature * (1 << (Z_CURVATURE_SHIFT + Z_SCALE_SHIFT)) * (cf.PixMicronX() * cf.PixMicronX()) / (cf.Scale() * cf.Scale())) );
				_CUDA_THROW_ERR_
			}

			int deltasX = (C.ClusterMapMaxXOffset / C.ClusterMapCoarseTolerance * 2 + 1);
			int deltasY = (C.ClusterMapMaxYOffset / C.ClusterMapCoarseTolerance * 2 + 1);
			int deltas2 = deltasX * deltasY;
			int refinedeltas = 2 * C.ClusterMapCoarseTolerance / C.ClusterMapFineTolerance + 1;
			int refinedeltas2 = refinedeltas * refinedeltas;
			WISE_ALLOC(pDeltas, sizeof(int) * ((2 * refinedeltas + deltasX + deltasY)));
			WISE_ALLOC(pMapCounts, sizeof(int) * max(deltas2, refinedeltas2) * (pThis->m_Prop.multiProcessorCount * pThis->m_Prop.maxThreadsPerBlock + 1) );
			int *pBest = pMapCounts + max(deltas2, refinedeltas2) * (pThis->m_Prop.multiProcessorCount * pThis->m_Prop.maxThreadsPerBlock);
			WISE_ALLOC(pStagePos, sizeof(short) * cf.Images() * 4);
			short *pStagePosX = pStagePos;
			short *pStagePosY = pStagePosX + cf.Images();
			short *pDeltaStagePosX = pStagePosY + cf.Images();
			short *pDeltaStagePosY = pDeltaStagePosX + cf.Images();	
			HOST_WISE_ALLOC(pHostStagePos, sizeof(short) * cf.Images() * 4);
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
			THROW_ON_CUDA_ERR(cudaMemcpy(pStagePos, pHostStagePos, sizeof(short) * 4 * cf.Images(), cudaMemcpyHostToDevice));
			THROW_ON_CUDA_ERR(cudaMemset(pCells, 0, sizeof(Cell) * ncellsx * ncellsy));

			int demagDZ1M;	
			int id = 0;
			IntCluster *pImagesBase1 = pImagesBase;			

			{
				dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks = dim3(totalclusters / pThis->m_Prop.maxThreadsPerBlock + 1, 1, 1);
				correctcurvature_kernel<<<iblocks, ithreads>>>(pImagesBase, pClusterZs, sin(IC.CameraRotation) * (1 << FRACT_RESCALE_SHIFT), (cos(IC.CameraRotation) - 1) * (1 << FRACT_RESCALE_SHIFT), pCurvX, pCurvY, pZCurvX, pZCurvY, IC.DMagDX * (1 << XY_MAGNIFICATION_SHIFT), IC.DMagDY * (1 << XY_MAGNIFICATION_SHIFT), totalclusters, width / 2, height / 2 );		
				_CUDA_THROW_ERR_
			}	
			{
				dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks = dim3(cf.ImageClusterCounts(0) / pThis->m_Prop.maxThreadsPerBlock + 1, 1, 1);		
				setXYZs_kernel<<<iblocks, ithreads>>>(pClusterXs, pClusterYs, pClusterZs, cf.ImageClusterCounts(0), 0, pStagePosX, pStagePosY, 0);
				_CUDA_THROW_ERR_
			}

			int bestclustermapcount = -1;
			int bestclustermapcount_img = 0;


			for (img = 0; img < cf.Images() - 1; img++)
			{
				pImageNext = pImagesBase + cf.ImageClusterCounts(img);					
				demagDZ1M = IC.DMagDZ * (pFixedZs[img + 1] - pFixedZs[img]) * (1 << DEMAG_SHIFT);
				int launches;
				// BEGIN COARSE MAPPING
				{		
					makedeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, C.ClusterMapCoarseTolerance, deltasX, deltasY, pDeltaStagePosX, pDeltaStagePosY, img + 1);
					_CUDA_THROW_ERR_
				}		
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(deltas2 / ithreads.x + 1, 1, 1);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, deltas2, pBest);
					_CUDA_THROW_ERR_
				}	
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					maphash_minarea_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), cf.ImageClusterCounts(img) / (pThis->m_Prop.maxThreadsPerBlock + pThis->m_Prop.multiProcessorCount) + 1, pCells, pCellContents, cellsize, C.MaxCellContent, ncellsx, ncellsy, C.ClusterMapMinSize);
					_CUDA_THROW_ERR_
				}						
				{						
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);		
					SySal::GPU::InterruptibleKernels::IntKernel<trymap_kernel_args, trymap_kernel_status, trymap2_Ikernel> Launcher;
					Launcher.Arguments.pC = pImageNext;
					Launcher.Arguments.nc = cf.ImageClusterCounts(img + 1);
					Launcher.Arguments.clusterblocksize = cf.ImageClusterCounts(img + 1) / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1;
					Launcher.Arguments.pCell = pCells;
					Launcher.Arguments.pCellContent = pCellContents;
					Launcher.Arguments.maxcellcontent = C.MaxCellContent;
					Launcher.Arguments.pDeltas = pDeltas;
					Launcher.Arguments.deltasx = deltasX;
					Launcher.Arguments.deltasy = deltasY;
					Launcher.Arguments.cellsize = cellsize;
					Launcher.Arguments.minclustersize = C.ClusterMapMinSize;
					Launcher.Arguments.tol = C.ClusterMapCoarseTolerance;
					Launcher.Arguments.w = width;
					Launcher.Arguments.h = height;
					Launcher.Arguments.demag = demagDZ1M;
					Launcher.Arguments.nx = ncellsx;
					Launcher.Arguments.ny = ncellsy;
					Launcher.Arguments.pMapCounts = pMapCounts;
					Launcher.Arguments.sampledivider = C.ClusterMapSampleDivider;	
					Launcher.Arguments.clustermapmin = C.MinClusterMapsValid;			
					Launcher.Arguments.pBest = pBest;
					launches = Launcher.Launch(iblocks, ithreads, 5);
					_CUDA_THROW_ERR_
				}		
				{
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, pBest, sizeof(int), cudaMemcpyDeviceToHost))
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
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(refinedeltas2 / ithreads.x + 1, 1, 1);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, refinedeltas2, pBest);
					_CUDA_THROW_ERR_
				}
				{
					if (best_matches > bestclustermapcount)
					{
						bestclustermapcount = best_matches;
						bestclustermapcount_img = img;
					}		
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);	
					SySal::GPU::InterruptibleKernels::IntKernel<refinemap_kernel_args, refinemap_kernel_status, refinemap_Ikernel> Launcher;
					Launcher.Arguments.pC = pImageNext;
					Launcher.Arguments.nc = cf.ImageClusterCounts(img + 1);
					Launcher.Arguments.clusterblocksize = cf.ImageClusterCounts(img + 1) / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1;
					Launcher.Arguments.pCell = pCells;
					Launcher.Arguments.pCellContent = pCellContents;
					Launcher.Arguments.maxcellcontent = C.MaxCellContent;
					Launcher.Arguments.cellsize = cellsize;
					Launcher.Arguments.tol = C.ClusterMapCoarseTolerance;			
					Launcher.Arguments.w = width;
					Launcher.Arguments.h = height;	
					Launcher.Arguments.demag = demagDZ1M;
					Launcher.Arguments.nx = ncellsx;
					Launcher.Arguments.ny = ncellsy;	
					Launcher.Arguments.pMapCounts = pMapCounts;
					Launcher.Arguments.pClusterChain = pClusterChains;
					Launcher.Arguments.pBase = pImagesBase1;
					Launcher.Arguments.pDeltas = pDeltas + deltasX + deltasY;
					Launcher.Arguments.deltas = refinedeltas;
					Launcher.Arguments.refinebin = C.ClusterMapFineTolerance;	
					Launcher.Arguments.pBest = pBest;
					launches = Launcher.Launch(iblocks, ithreads, 5);			
					_CUDA_THROW_ERR_
				}	

				{
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, pBest, sizeof(int), cudaMemcpyDeviceToHost));
					best_matches = best_matches >> 16;
					//printf("\n%d %d", img, best_matches);
				}				
				// END FINE MAPPING

				// BEGIN FINAL MAPPING
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					clearhash_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), cf.ImageClusterCounts(img) / (pThis->m_Prop.maxThreadsPerBlock + pThis->m_Prop.multiProcessorCount) + 1, pCells, cellsize, ncellsx, ncellsy);
					_CUDA_THROW_ERR_
				}
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					maphash_kernel<<<iblocks, ithreads>>>(pImagesBase, cf.ImageClusterCounts(img), cf.ImageClusterCounts(img) / (pThis->m_Prop.maxThreadsPerBlock + pThis->m_Prop.multiProcessorCount) + 1, pCells, pCellContents, cellsize, C.MaxCellContent, ncellsx, ncellsy);			
					_CUDA_THROW_ERR_
				}				
				{		
					makefinaldeltas_fromshift_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pDeltas, (best_matches > C.MinClusterMapsValid) ? C.ClusterMapFineTolerance : 0, pDeltas + deltasX + deltasY, pBest, refinedeltas, pStagePosX, pStagePosY, pDeltaStagePosX, pDeltaStagePosY, img + 1, cf.Images());
					_CUDA_THROW_ERR_			
				}		

				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(cf.ImageClusterCounts(img + 1) / pThis->m_Prop.maxThreadsPerBlock + 1, 1, 1);
					id = pImageNext - pImagesBase1;
					setXYZs_kernel<<<iblocks, ithreads>>>(pClusterXs + id, pClusterYs + id, pClusterZs + id, cf.ImageClusterCounts(img + 1), img + 1, pStagePosX, pStagePosY, ((pFixedZs[img + 1] /*- pFixedZs[0]*/ + dz) * (1 << Z_SCALE_SHIFT)) );
				}	
				{
					dim3 ithreads = dim3(1, 1, 1);
					dim3 iblocks = dim3(1, 1, 1);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, 1, pBest);
					_CUDA_THROW_ERR_
				}
#if 1
				{			
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);						
					SySal::GPU::InterruptibleKernels::IntKernel<finalmap_kernel_args, finalmap_kernel_status, finalmap_Ikernel> Launcher;
					Launcher.Arguments.pC = pImageNext;
					Launcher.Arguments.nc = cf.ImageClusterCounts(img + 1);
					Launcher.Arguments.clusterblocksize = cf.ImageClusterCounts(img + 1) / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1;
					Launcher.Arguments.pCell = pCells;
					Launcher.Arguments.pCellContent = pCellContents;
					Launcher.Arguments.maxcellcontent = C.MaxCellContent;
					Launcher.Arguments.cellsize = cellsize;
					Launcher.Arguments.tol = C.ClusterMapFineAcceptance;
					Launcher.Arguments.w = width;
					Launcher.Arguments.h = height;	
					Launcher.Arguments.demag = demagDZ1M;
					Launcher.Arguments.nx = ncellsx;
					Launcher.Arguments.ny = ncellsy;					
					Launcher.Arguments.pMapCounts = pMapCounts;
					Launcher.Arguments.pClusterChain = pClusterChains;
					Launcher.Arguments.img = img + 1;
					Launcher.Arguments.pBase = pImagesBase1;
					Launcher.Arguments.pDX = pDeltaStagePosX;
					Launcher.Arguments.pDY = pDeltaStagePosY;
					int launches = Launcher.Launch(iblocks, ithreads, 5);
					_CUDA_THROW_ERR_
				}		
#endif
				{
					int step;
					for (step = 1; step <= (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount / 2); step <<= 1)
					{
						int thr = (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount / 2) / step;
						dim3 ithreads = dim3(max(1, thr / pThis->m_Prop.multiProcessorCount), 1, 1);
						dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);				
						sumcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, 1, pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount, step);
					}
					_CUDA_THROW_ERR_
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
			if (CurrentView > 0)
			{
				makechainwindow_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pChMapWnd, pStagePosX, pStagePosY, cf.Images(), width, height, cf.PixMicronX() / cf.Scale() * (1 << XY_SCALE_SHIFT), cf.PixMicronY() / cf.Scale() * (1 << XY_SCALE_SHIFT), ncellsx * ncellsy, C.ChainMapXYCoarseTolerance, pLastView, pCells, pCellContents, C.MaxCellContent, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), pLastView);
				_CUDA_THROW_ERR_
			}
			{
				dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
				SySal::GPU::InterruptibleKernels::IntKernel<makechain_kernel_args, makechain_kernel_status, makechain_Ikernel> Launcher;
				Launcher.Arguments.pC = pImagesBase1;
				Launcher.Arguments.pClusterChains = pClusterChains;
				Launcher.Arguments.pClusterXs = pClusterXs;
				Launcher.Arguments.pClusterYs = pClusterYs;
				Launcher.Arguments.pClusterZs = pClusterZs;
				Launcher.Arguments.pChain = pChains;
				Launcher.Arguments.pChainCounts = pChainCounts;
				Launcher.Arguments.totalclusters = totalclusters;
				Launcher.Arguments.clusterblocksize = totalclusters / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1;
				Launcher.Arguments.minvol = C.MinVolumePerChain;
				Launcher.Arguments.minclusters = C.MinClustersPerChain;
				Launcher.Arguments.xtomicron = cf.PixMicronX() / cf.Scale();
				Launcher.Arguments.ytomicron = cf.PixMicronY() / cf.Scale();
				Launcher.Arguments.width = width;
				Launcher.Arguments.height = height;
				Launcher.Arguments.stagex = cf.StageX(0) * (1 << XY_SCALE_SHIFT);
				Launcher.Arguments.stagey = cf.StageY(0) * (1 << XY_SCALE_SHIFT);
				Launcher.Arguments.xslant = IC.XSlant * cf.PixMicronX() / cf.Scale() * (1 << (Z_SCALE_SHIFT + SLOPE_SHIFT));
				Launcher.Arguments.yslant = IC.YSlant * cf.PixMicronY() / cf.Scale() * (1 << (Z_SCALE_SHIFT + SLOPE_SHIFT));
				Launcher.Arguments.viewtag = CurrentView;
				int launches = Launcher.Launch(iblocks, ithreads, 5);
				//printf("\nLaunches %d", launches);
				_CUDA_THROW_ERR_
			}

			if (CurrentView > 0)
			{
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					maphashchain_kernel<<<iblocks, ithreads>>>(pLastView, pChMapWnd, pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount);
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
				WISE_ALLOC(pMapCounts, sizeof(int) * max(deltasXYZ, refinedeltasXYZ) * (pThis->m_Prop.multiProcessorCount * pThis->m_Prop.maxThreadsPerBlock + 1) );
				THROW_ON_CUDA_ERR(cudaMemset(pMapCounts, 0, _MEM_(pMapCounts)));
				pBest = (int *)(void *)(&pThisView->Reserved);

				{		
					makechaindeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, C.ChainMapXYCoarseTolerance, C.ChainMapZCoarseTolerance, deltasX, deltasY, deltasZ, pLastView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), IC.XSlant, IC.YSlant, 0, 0/*dxdz, dydz*/);
					_CUDA_THROW_ERR_
				}		
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(deltasXYZ / ithreads.x + 1, 1, 1);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, deltasXYZ, pBest);
					_CUDA_THROW_ERR_
				}		
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					//printf("\nDeltas %d %d %d %d", deltasX, deltasY, deltasZ, deltasXYZ);
					SySal::GPU::InterruptibleKernels::IntKernel<trymapchain_kernel_args, trymapchain_kernel_status, trymapchaindxydz_Ikernel> Launcher;
					Launcher.Arguments.pChains = pChains;
					Launcher.Arguments.pChainCounts = pChainCounts;
					Launcher.Arguments.nc = totalclusters;
					Launcher.Arguments.chainblocksize = totalclusters / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1;
					Launcher.Arguments.pChMapWnd = pChMapWnd;			
					Launcher.Arguments.pDeltas = pDeltas;
					Launcher.Arguments.deltasX = deltasX;
					Launcher.Arguments.deltasY = deltasY;
					Launcher.Arguments.deltasZ = deltasZ;
					Launcher.Arguments.xytol = C.ChainMapXYCoarseTolerance;
					Launcher.Arguments.ztol = C.ChainMapZCoarseTolerance;
					Launcher.Arguments.minchainsize = C.ChainMapMinVolume;
					Launcher.Arguments.pMapCounts = pMapCounts;
					Launcher.Arguments.sampledivider = C.ChainMapSampleDivider;
					Launcher.Arguments.pBest = pBest;
					int launches = Launcher.Launch(iblocks, ithreads, 2);
					//printf("\nTryMapChain Launches: %d", launches);
					_CUDA_THROW_ERR_
				}
				{
					int mapc, best;
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, (int *)(void *)(&pThisView->Reserved), sizeof(int), cudaMemcpyDeviceToHost));
					//printf("\nDEBUG %08X", best_matches);
					mapc = best_matches >> 16;
					if (mapc < C.MinChainMapsValid) best = deltasX / 2 + (deltasY / 2) * deltasX + (deltasZ / 2) * deltasX * deltasY;
					else best = best_matches & 0xffff;
					best_matches = mapc;


				}					
				// END COARSE CHAIN MAPPING

				// BEGIN FINE CHAIN MAPPING
				if (best_matches < C.MinChainMapsValid)
				{
					printf("\nBad chain mapping (%d/%d), switching to default", best_matches, C.MinChainMapsValid);
					makechaindeltas_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas, 0, 0, deltasX, deltasY, deltasZ, pLastView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), IC.XSlant, IC.YSlant, dxdz, dydz);
					_CUDA_THROW_ERR_
				}
				{		
					makechaindeltas_fromshift_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pDeltas + deltasX + deltasY + 3 * deltasZ, (best_matches < C.MinChainMapsValid) ? 0 : C.ChainMapXYFineTolerance, 0, refinedeltasXY, refinedeltasXY, refinedeltasZ, pDeltas, pBest, deltasX, deltasY, deltasZ);
					_CUDA_THROW_ERR_
				}
				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(refinedeltasXYZ / ithreads.x + 1, 1, 1);
					resetcounts_kernel<<<iblocks, ithreads>>>(pMapCounts, refinedeltasXYZ, (int *)(void *)(&pThisView->Reserved));			
					_CUDA_THROW_ERR_
				}

				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					SySal::GPU::InterruptibleKernels::IntKernel<trymapchain_kernel_args, trymapchain_kernel_status, trymapchain_Ikernel> Launcher;
					Launcher.Arguments.pChains = pChains;
					Launcher.Arguments.pChainCounts = pChainCounts;
					Launcher.Arguments.nc = totalclusters;
					Launcher.Arguments.chainblocksize = totalclusters / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1;
					Launcher.Arguments.pChMapWnd = pChMapWnd;			
					Launcher.Arguments.pDeltas = pDeltas + deltasX + deltasY + 3 * deltasZ;
					Launcher.Arguments.deltasX = refinedeltasXY;
					Launcher.Arguments.deltasY = refinedeltasXY;
					Launcher.Arguments.deltasZ = refinedeltasZ;
					Launcher.Arguments.xytol = C.ChainMapXYFineAcceptance; 
					Launcher.Arguments.ztol = C.ChainMapZFineAcceptance; 			
					Launcher.Arguments.minchainsize = C.ChainMapMinVolume;
					Launcher.Arguments.pMapCounts = pMapCounts;
					Launcher.Arguments.sampledivider = C.ChainMapSampleDivider;
					Launcher.Arguments.pBest = pBest;
					int launches = Launcher.Launch(iblocks, ithreads, 2);
					//printf("\nRefineMapChain Launches: %d", launches);
					_CUDA_THROW_ERR_
				}
				{
					int mapc, best;
					THROW_ON_CUDA_ERR(cudaMemcpy(&best_matches, pBest, sizeof(int), cudaMemcpyDeviceToHost))
					printf("\nDEBUG %08X", best_matches);
					mapc = best_matches >> 16;
					best = best_matches & 0xffff;
					best_matches = mapc;			
				}				

				// END FINE CHAIN MAPPING

				// BEGIN FINAL CHAIN MAPPING

				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					finalmapchain_kernel<<<iblocks, ithreads>>>(pChains, pChainCounts, totalclusters, totalclusters / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1, pChMapWnd, pDeltas + deltasX + deltasY + 3 * deltasZ, refinedeltasXY, refinedeltasZ, C.ChainMapXYFineAcceptance, C.ChainMapZFineAcceptance, pBest);
					_CUDA_THROW_ERR_
				}

				// END FINAL CHAIN MAPPING

				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					clearhashchain_kernel<<<iblocks, ithreads>>>(pLastView, pChMapWnd, pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount);
					_CUDA_THROW_ERR_
				}

				{
					dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
					dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
					negshift_viewchains_kernel<<<iblocks, ithreads>>>(pLastView, pDeltas + deltasX + deltasY + 3 * deltasZ, refinedeltasXY, refinedeltasZ, pBest);
					_CUDA_THROW_ERR_
				}

			}


			{
				setchainbase_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pCompactChainCounts, pChainCounts, pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount, pChainMapHeader, pThisView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), cf.StageZ(0) * (1 << Z_SCALE_SHIFT), (((int)width) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronX()), (((int)height) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronY()));
				_CUDA_THROW_ERR_
			}
			{
				dim3 ithreads = dim3(pThis->m_Prop.maxThreadsPerBlock, 1, 1);
				dim3 iblocks = dim3(pThis->m_Prop.multiProcessorCount, 1, 1);
				compactchains_kernel<<<iblocks, ithreads>>>(pCompactChains, pCompactChainCounts, pChains, pChainCounts, totalclusters / (pThis->m_Prop.maxThreadsPerBlock * pThis->m_Prop.multiProcessorCount) + 1);
				_CUDA_THROW_ERR_
			}

			{
				//if (CurrentView == 0)
				setchainviewheader_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pChainMapHeader, pThisView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), cf.StageZ(0) * (1 << Z_SCALE_SHIFT), 0, 0, 0, 0);
				//else
					//setchainviewheader_kernel<<<dim3(1,1,1), dim3(1,1,1)>>>(pChainMapHeader, pThisView, cf.StageX(0) * (1 << XY_SCALE_SHIFT), cf.StageY(0) * (1 << XY_SCALE_SHIFT), cf.StageZ(0) * (1 << Z_SCALE_SHIFT), pDeltas + deltasX + deltasY + 3 * deltasZ, refinedeltasXY, refinedeltasZ, pBest);
				_CUDA_THROW_ERR_
			}			

			if (CurrentView > 0) 
				pThis->SendViewsToTracker(CurrentView, (((int)width) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronX()), (((int)height) << XY_SCALE_SHIFT) / cf.Scale() * fabs(cf.PixMicronY()), pLastView, pThisView);

			return 0;
		}


	};
};