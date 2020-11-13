#ifndef _SYSAL_GPU_INCREMENTAL_MAP_TRACK_H_
#define _SYSAL_GPU_INCREMENTAL_MAP_TRACK_H_

#include "map.h"
#include "Tracker.h"
#include "cuda_runtime.h"
#include "gpu_defines.h"
#include <vector>

namespace SySal
{
namespace GPU
{

#define FIRST_VIEW_EMPTY

#define _MEM_(ptr) _mem_ ## ptr
#define _MEMPTR_(typ, ptr) \
	typ *ptr; \
	long long _MEM_(ptr); 	

	class PrismMapTracker 
	{		
	public:
		typedef void (*ChainDumper)(void *pContext, ChainView *pLastView, ChainView *pThisView);

		struct PerformanceCounters
		{
			int GPU;
			int GPUCores;
			int GPUClockMHz;
			long MapTimeMS;
			long TrackTimeMS;
			int Clusters;
			int Chains;
			int Tracks;
		};

		struct DebugDump
		{
			struct t_Image
			{
				int DeltaX;
				int DeltaY;
				int Clusters;
				int MappingClusters;
			};
			int Images;
			t_Image *pImages;
			int MinTagView;
			bool MapValid;
			int ViewDeltaX;
			int ViewDeltaY;
			HashTableBounds HTBounds;
			struct t_ViewChains
			{
				ChainView *pV;
				ChainView *pHostView;
			};
			t_ViewChains Views[2];			
			struct t_TrackIteration
			{
				int TotalPairs;
				static const int Replicas = 3;
				int *pBinFill[Replicas];
				long long sz_pBinFill;
				int *pPairIndices;
				long long sz_pPairIndices;
				long long sz_TotalPairs;
				int *pPairComputer;
				int *pTBinFill;
				long long sz_pTBinFill;
				int ISX;
				int ISY;

				inline void Prepare(long long size_pBinFill, long long size_TotalPairs, long long size_pPairIndices, long long size_pTBinFill)
				{					
					if (size_pBinFill)
					{
						sz_pBinFill = size_pBinFill;
						for (int r = 0; r < Replicas; r++)
							pBinFill[r] = (int *)malloc(sz_pBinFill * sizeof(int));
					}
					if (size_TotalPairs) pPairComputer = (int *)malloc((sz_TotalPairs = size_TotalPairs) * sizeof(int));
					if (size_pPairIndices) pPairIndices = (int *)malloc((sz_pPairIndices = size_pPairIndices) * sizeof(int));
					if (size_pTBinFill) pTBinFill = (int *)malloc((sz_pTBinFill = size_pTBinFill) * sizeof(int));
				}
			};
			t_TrackIteration *pTrackIterations;
			int TrackIterations;
			int TempTracks;

			inline void Free()
			{
				int i;
				if (pImages)
				{
					free(pImages);
					pImages = 0;
					Images = 0;
				}
				for (i = 0; i < 2; i++)				
					if (Views[i].pHostView)
					{
						free(Views[i].pHostView);
						Views[i].pHostView = 0;
					}				
				for (i = 0; i < TrackIterations; i++)
				{
					for (int r = 0; r < t_TrackIteration::Replicas; r++)
						if (pTrackIterations[i].pBinFill[r]) { free(pTrackIterations[i].pBinFill[r]); pTrackIterations[i].pBinFill[r] = 0; }
					if (pTrackIterations[i].pPairComputer) { free(pTrackIterations[i].pPairComputer); pTrackIterations[i].pPairComputer	= 0; }
					if (pTrackIterations[i].pPairIndices) { free(pTrackIterations[i].pPairIndices); pTrackIterations[i].pPairIndices = 0; }
					if (pTrackIterations[i].pTBinFill) { free(pTrackIterations[i].pTBinFill); pTrackIterations[i].pTBinFill = 0; }
				}
				if (pTrackIterations)
				{
					free(pTrackIterations);
					pTrackIterations = 0;
					TrackIterations = 0;
				}
			}

			inline t_Image &AddImage()
			{
				pImages = (t_Image *)realloc(pImages, sizeof(t_Image) * (++Images));
				pImages[Images - 1].Clusters = 0;
				pImages[Images - 1].MappingClusters = 0;
				pImages[Images - 1].DeltaX = pImages[Images - 1].DeltaY = 0;
				return pImages[Images - 1];
			}

			inline t_TrackIteration &AddIteration()
			{
				pTrackIterations = (t_TrackIteration *)realloc(pTrackIterations, sizeof(t_TrackIteration) * (++TrackIterations));
				for (int r = 0; r < t_TrackIteration::Replicas; r++) pTrackIterations[TrackIterations - 1].pBinFill[r] = 0;
				pTrackIterations[TrackIterations - 1].pPairIndices = 0;
				pTrackIterations[TrackIterations - 1].pTBinFill = 0;
				pTrackIterations[TrackIterations - 1].pPairComputer = 0;
				pTrackIterations[TrackIterations - 1].sz_pBinFill = 0;
				pTrackIterations[TrackIterations - 1].sz_pPairIndices = 0;
				pTrackIterations[TrackIterations - 1].sz_pTBinFill = 0;
				pTrackIterations[TrackIterations - 1].sz_TotalPairs = 0;
				return pTrackIterations[TrackIterations - 1];
			}

			inline void Init()
			{
				pImages = 0;
				Images = 0;
				MinTagView = -1;
				ViewDeltaX = ViewDeltaY = 0;
				MapValid = false;
				pTrackIterations = 0;
				TrackIterations = 0;
				int i;
				for (i = 0; i < 2; i++)
				{
					Views[i].pV = 0;
					Views[i].pHostView = 0;
				}
			}
		};

	protected:
		int m_DeviceId;

		cudaDeviceProp m_Prop;

		long long m_DebugMarker;

		void make_threads_blocks(int iterations, dim3 &threads, dim3 &blocks);

		char LastError[1024];

		void SendViewsToTracker(int minviewtag, int width, int height, ChainView *pLastView, ChainView *pThisView);

		ChainDumper m_ChainDumper;
		void *m_CDContext;

		PerformanceCounters m_PerformanceCounters;

		bool m_EnableDebugDump;
		DebugDump m_DebugDump;

		int m_Verbosity;

		_MEMPTR_(ChainView, pHLastView)
		_MEMPTR_(ChainView, pHThisView)

	public:

		inline long GetDebugMarker() { return m_DebugMarker; }

		inline void SetDebugMarker(long long mk) { m_DebugMarker = mk; }

		inline void SetEnableDebugDump(bool enable) { m_EnableDebugDump = enable; }

		inline DebugDump &GetDebugDump() { return m_DebugDump; }

		inline void SetVerbosity(int v) { m_Verbosity = v; }

		void HardReset();

		void SetChainDumper(void *pContext, ChainDumper dmp);

		class ClusterChainer : public SySal::ClusterChainer
		{
		protected:
			PrismMapTracker *pThis;			

			SySal::ImageCorrection IC;
			SySal::ClusterChainer::Configuration C;
			bool IsTop;
			int CurrentView;

			void HardReset();

			int FindEdges(EmulsionEdge &t, EmulsionEdge &b, IntClusterFile &cf, int threshold, int &refimg);

			double *pThicknessSamples;
			int ThicknessSamples;
			double WeakThickness;

			_MEMPTR_(short, pClusterPos)
			_MEMPTR_(void, pClusterData)
			_MEMPTR_(IntCluster *, pClusterChains)
			_MEMPTR_(IntCluster *, pCellContents)
			_MEMPTR_(Cell, pCells)
			_MEMPTR_(IntChain, pChains)
			_MEMPTR_(int, pChainCounts)
			_MEMPTR_(IntChain, pCompactChains)
			_MEMPTR_(int, pDeltas)
			_MEMPTR_(int, pMapCounts)
			_MEMPTR_(int, pCurv)
			_MEMPTR_(short, pStagePos)
			_MEMPTR_(short, pHostStagePos)
			_MEMPTR_(ChainMapHeader, pChainMapHeader)
			_MEMPTR_(double, pFixedZs)
			_MEMPTR_(ChainMapWindow, pChMapWnd)
			_MEMPTR_(ChainView, pLastView)
			_MEMPTR_(ChainView, pThisView)
			_MEMPTR_(IntMapCluster, pMapClusters)
			_MEMPTR_(IntMapChain, pMapChains)
			_MEMPTR_(IntPair, pPairs)
			_MEMPTR_(int, pClustersInCell)
			_MEMPTR_(int, pPairComputer)
			_MEMPTR_(int, pPairMatchResult)
			_MEMPTR_(int, pMatchMap)

			int DistributeClusterPairsToThreads(int totalmapclusters);
			void ParallelSum(int *pdata, int total, int *psum);
			void MultiParallelSum(int *pdata, int total, int multiplicity, int *psum);
			void ParallelMax(int *pdata, int total, int *pmax);

		public:
			ClusterChainer();
			virtual ~ClusterChainer();

			virtual SySal::ClusterChainer::Configuration GetConfiguration();
			virtual void Reset(SySal::ClusterChainer::Configuration &c, ImageCorrection &corr, bool istop);
			virtual bool SetReferenceZs(IntClusterFile &cf, bool istop);
			virtual EmulsionFocusInfo AddClusters(IntClusterFile &cf);
			virtual int TotalChains();
			virtual ChainMapHeader *Dump();
			virtual OpaqueChainMap &GetDeviceChainMap();
			virtual int GetXYScale();
			virtual int GetZScale();		
			virtual void SetLogFileName(char *logfile);			

			friend class PrismMapTracker;
		};

		class Tracker : public SySal::Tracker
		{
		protected:
			PrismMapTracker *pThis;			

			SySal::Tracker::Configuration C;

			_MEMPTR_(TrackMapHeader, pTracks)
			_MEMPTR_(TrackMapHeader, pHostTracks)
			_MEMPTR_(ChainMapHeader, pChainMapHeader)
			_MEMPTR_(ChainView *, ppChainMapViewEntryPoints)
			_MEMPTR_(InternalInfo, pInternalInfo)
			_MEMPTR_(char, pHashTable)			
			_MEMPTR_(int, pKernelContinue)
			_MEMPTR_(IntChain, pTrackGrains)
			_MEMPTR_(int, pScheduler)
			_MEMPTR_(int, pPairComputer)
			_MEMPTR_(int, pPairIndices)
			_MEMPTR_(int, pBinFill)
			_MEMPTR_(IntChain *, ppBins)
			_MEMPTR_(int, pTBinFill)
			_MEMPTR_(int, pCountTempTracks)
			_MEMPTR_(TempIntTrack, pTBins)
			_MEMPTR_(TempIntTrack *, ppTempTracks)
			int _MergeTracksKernel_LoopLimiter_;

			void HardReset();

			void InternalFindTracks(int minviewtag, int width, int height, ChainView *pLastView, ChainView *pThisView);

		public:
			Tracker();
			virtual ~Tracker();

			virtual SySal::Tracker::Configuration GetConfiguration();
			virtual void Reset(SySal::Tracker::Configuration &c);
			virtual int FindTracks(ChainMapHeader &cm);
			virtual int FindTracksInDevice(OpaqueChainMap &ocm);
			virtual int TotalTracks();
			virtual TrackMapHeader *Dump();		
			virtual void SetOption(const char *option, const char *value);	
			virtual void SetLogFileName(char *logfile);

			virtual int GetSlopeScale();
			virtual int GetXYScale();
			virtual int GetZScale();

			virtual bool IsFirstViewEmpty();
			virtual bool IsLastViewEmpty();

			friend class PrismMapTracker;
		};

	protected:

		SySal::GPU::PrismMapTracker::ClusterChainer m_Chainer;
		SySal::GPU::PrismMapTracker::Tracker m_Tracker;
	
	public:
		PrismMapTracker(int gpuid);
		virtual ~PrismMapTracker();		

		inline SySal::ClusterChainer &ClusterChainer() { return m_Chainer; }
		inline SySal::Tracker &Tracker() { return m_Tracker; }
		bool IsFirstViewEmpty();
		bool IsLastViewEmpty();


		double GetThickness();
		PerformanceCounters GetPerformanceCounters();

		friend class ClusterChainer;
		friend class Tracker;
	};
}
};

#endif