// OnlineMapTrack2RWD.h

#pragma once

using namespace System;
#using "SySalCore.dll"
#using "Imaging.dll"
#using "Tracking.dll"
#using "Scanning.dll"
#using "DAQSystem.dll"
#include "map.h"
#include "Tracker.h"
#include "gpu_incremental_map_track.h"

namespace SySal 
{
namespace GPU 
{
	public ref class Utilities
	{
	public:
		static int GetAvailableGPUs();
	};

	public ref class RawDataViewSide : public SySal::Scanning::Plate::IO::OPERA::RawData::Fragment::View::Side
	{
	public:
		RawDataViewSide(SySal::TrackMapHeader *ptkhdr, SySal::IntClusterFile &cf, SySal::ClusterChainer::EmulsionFocusInfo ef, float sideoffset, SySal::DAQSystem::Scanning::IntercalibrationInfo ^intinfo, bool istop);
	};

	public interface class IRawDataViewSideConsumer
	{
		void ConsumeData(int n, bool istop, RawDataViewSide ^rwdvs);
	};

	public ref class MapTracker
	{
	private:
		int gpu;
		SySal::GPU::PrismMapTracker *pTk;
		SySal::ImageCorrection *pIC;
		SySal::ClusterChainer::Configuration *pCC;
		SySal::Tracker::Configuration *pTC;
		float pix_x_override;
		float pix_y_override;
		IRawDataViewSideConsumer ^rwdvsConsumer;
		void Free();
		char *pStringWorkspace;
		char *c_str(System::String ^s);
		System::String ^Activity;
		System::String ^PerfDumpFile;
		System::Collections::Generic::Queue<IntPtr> ^PreloadQueue;		
		bool TerminatePreloadThread;
		void PreloadFiles(System::Object ^);

	public:
		static int PreloadQueueLength = 3;
		void SetPixelXOverride(float v);
		void SetPixelYOverride(float v);
		void SetGPU(int g);
		void SetImageCorrection(System::String ^ics);
		void SetClusterChainerConfig(System::String ^ccs);
		void SetTrackerConfig(System::String ^tcs);
		void SetRawDataViewSideConsumer(IRawDataViewSideConsumer ^v);
		float GetCurrentThickness();
		System::String ^GetCurrentActivity();
		void SetPerformanceCounterDumpFile(System::String ^perfdumpfile);

		MapTracker();
		virtual ~MapTracker();
		!MapTracker();
		void FindTracks(cli::array<System::String ^> ^inputfiles, bool istop, float zsideoffset, SySal::DAQSystem::Scanning::IntercalibrationInfo ^intinfo);
	};
}
}
