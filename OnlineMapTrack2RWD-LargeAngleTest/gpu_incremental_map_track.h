#ifndef _SYSAL_GPU_INCREMENTAL_MAP_TRACK_H_
#define _SYSAL_GPU_INCREMENTAL_MAP_TRACK_H_

#include "map.h"
#include "Tracker.h"
#include "cuda_runtime.h"
#include "gpu_defines.h"

namespace SySal
{
namespace GPU
{

#define _MEM_(ptr) _mem_ ## ptr
#define _MEMPTR_(typ, ptr) \
	typ *ptr; \
	int _MEM_(ptr); 	

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

	protected:
		int m_DeviceId;

		cudaDeviceProp m_Prop;

		void make_threads_blocks(int iterations, dim3 &threads, dim3 &blocks);

		char LastError[1024];

		void SendViewsToTracker(int minviewtag, int width, int height, ChainView *pLastView, ChainView *pThisView);

		ChainDumper m_ChainDumper;
		void *m_CDContext;

		PerformanceCounters m_PerformanceCounters;

		_MEMPTR_(ChainView, pHLastView)
		_MEMPTR_(ChainView, pHThisView)

	public:

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

		double GetThickness();
		PerformanceCounters GetPerformanceCounters();

		friend class ClusterChainer;
		friend class Tracker;
	};
}
};

#endif