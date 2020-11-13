#include "gpu_incremental_map_track.h"
#include <malloc.h>
#include <stdio.h>
#include "gpu_defines.h"


namespace SySal
{
namespace GPU
{
	PrismMapTracker::PrismMapTracker(int gpuid) : CTOR_INIT(pHThisView), CTOR_INIT(pHLastView)
	{		
		m_Chainer.pThis = this;
		m_Tracker.pThis = this;
		m_DeviceId = gpuid;
		m_ChainDumper = 0;
		
		if (cudaGetDeviceProperties(&m_Prop, m_DeviceId)) throw "Invalid CUDA device.";

		printf("\n\nCUDA properties for device %d\nMaxThreadsPerMultiProcessor %d\nMaxThreadsPerBlock %d\nMultiprocessors %d\nMaxGridSize: %d %d %d\nMemory %d MB\n\n", 
			gpuid, m_Prop.maxThreadsPerMultiProcessor, m_Prop.maxThreadsPerBlock, m_Prop.multiProcessorCount, m_Prop.maxGridSize[0], m_Prop.maxGridSize[1], m_Prop.maxGridSize[2], m_Prop.totalGlobalMem / 1048576);	
		
	}

	PrismMapTracker::~PrismMapTracker()
	{
		HOST_DEALLOC(pHLastView);
		HOST_DEALLOC(pHThisView);
	}

	void PrismMapTracker::SetChainDumper(void *pContext, ChainDumper dmp)
	{
		m_ChainDumper = dmp;
		m_CDContext = pContext;
	}

	void PrismMapTracker::SendViewsToTracker(int minviewtag, int width, int height, ChainView *pLastView, ChainView *pThisView)
	{
		if (m_ChainDumper)
		{
			int sz;

			HOST_WISE_ALLOC(pHLastView, sizeof(ChainView));
			cudaMemcpy(pHLastView, m_Chainer.pLastView, sizeof(ChainView), cudaMemcpyDeviceToHost);
			sz = pHLastView->Size();
			HOST_WISE_ALLOC(pHLastView, sz);
			cudaMemcpy(pHLastView, m_Chainer.pLastView, sz, cudaMemcpyDeviceToHost);

			HOST_WISE_ALLOC(pHThisView, sizeof(ChainView));
			cudaMemcpy(pHThisView, m_Chainer.pThisView, sizeof(ChainView), cudaMemcpyDeviceToHost);
			sz = pHLastView->Size();
			HOST_WISE_ALLOC(pHThisView, sz);
			cudaMemcpy(pHThisView, m_Chainer.pThisView, sz, cudaMemcpyDeviceToHost);

			m_ChainDumper(m_CDContext, pHLastView, pHThisView);
		}
		m_Tracker.InternalFindTracks(minviewtag, width, height, pLastView, pThisView);
	}



	int PrismMapTracker::ClusterChainer::GetXYScale() { return 1 << XY_SCALE_SHIFT; }

	int PrismMapTracker::ClusterChainer::GetZScale() { return 1 << Z_SCALE_SHIFT; }

	void PrismMapTracker::ClusterChainer::SetLogFileName(char *logfile) {}

	ChainMapHeader *PrismMapTracker::ClusterChainer::Dump()
	{
		return 0;
	}

	SySal::ClusterChainer::Configuration PrismMapTracker::ClusterChainer::GetConfiguration() 
	{
		return C;
	}

	SySal::OpaqueChainMap &PrismMapTracker::ClusterChainer::GetDeviceChainMap()
	{
		throw "Not supported.";
	}
	
	bool PrismMapTracker::ClusterChainer::SetReferenceZs(SySal::IntClusterFile &cf, bool istop) 
	{
		EmulsionEdge t, b;
		int refimg = 0;
		FindEdges(t, b, cf, C.ClusterThreshold, refimg);	
		if (t.Valid && b.Valid)
		{
			int place = 0;
			int i;
			double thk = t.Z - b.Z;
			for (i = 0; i < ThicknessSamples; i++)
				if (pThicknessSamples[i] >= thk)
				{
					place = i;
					break;
				}
			pThicknessSamples = (double *)realloc(pThicknessSamples, (ThicknessSamples + 1) * sizeof(double));
			for (i = ThicknessSamples - 1; i >= place; i--)
				pThicknessSamples[i + 1] = pThicknessSamples[i];
			pThicknessSamples[place] = thk;
			ThicknessSamples++;			
		}
		return (ThicknessSamples >= 1);
	}

	double PrismMapTracker::GetThickness()
	{
		if (m_Chainer.ThicknessSamples <= 0) throw "No thickness info available.";
		if (m_Chainer.ThicknessSamples % 2 == 1) return m_Chainer.pThicknessSamples[m_Chainer.ThicknessSamples / 2];
		return 0.5 * (m_Chainer.pThicknessSamples[m_Chainer.ThicknessSamples / 2] + m_Chainer.pThicknessSamples[m_Chainer.ThicknessSamples / 2 - 1]);
	}

	int PrismMapTracker::ClusterChainer::TotalChains()
	{
		return 0;
	}

	int PrismMapTracker::Tracker::GetXYScale() { return 1 << XY_SCALE_SHIFT; }

	int PrismMapTracker::Tracker::GetZScale() { return 1 << Z_SCALE_SHIFT; }

	int PrismMapTracker::Tracker::GetSlopeScale() { return 1 << SLOPE_SCALE_SHIFT; }

	SySal::Tracker::Configuration PrismMapTracker::Tracker::GetConfiguration() 
	{
		return C;
	}

	void PrismMapTracker::Tracker::SetOption(const char *option, const char *value)
	{
		if (strcmpi(option, "_MergeTracksKernel_LoopLimiter_") == 0)
		{
			if (sscanf(value, "%d", &_MergeTracksKernel_LoopLimiter_) != 1) 
				throw "Bad option value.";
		}
	}

	void PrismMapTracker::Tracker::SetLogFileName(char *logfile) {}

	SySal::TrackMapHeader *PrismMapTracker::Tracker::Dump() { return pHostTracks; }

	int PrismMapTracker::Tracker::TotalTracks() { return pHostTracks->Count; }

	int PrismMapTracker::Tracker::FindTracks(SySal::ChainMapHeader &cm)
	{
		throw "Not supported.";
	}

	int PrismMapTracker::Tracker::FindTracksInDevice(SySal::OpaqueChainMap &cm)
	{
		throw "Superseded by PrismMapTracker::Tracker::Dump.";
	}

};
};