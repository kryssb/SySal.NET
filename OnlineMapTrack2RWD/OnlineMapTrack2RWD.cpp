// This is the main DLL file.

#include "stdafx.h"
#include <vector>
#include "xml-lite.h"
#include "OnlineMapTrack2RWD.h"
#include "gpu_util.h"
#include <stdlib.h>
#include <math.h>

namespace SySal
{
namespace GPU
{
int Utilities::GetAvailableGPUs()
{
	return ::SySal::GPU::GetAvailableGPUs();
}

const int StringWorkspaceSize = 4096;

MapTracker::MapTracker()
{
	gpu = -1;
	debugmark = 0;
	pTk = 0;
	pIC = new SySal::ImageCorrection;
	pCC = new SySal::ClusterChainer::Configuration;
	pTC = new SySal::Tracker::Configuration;
	pStringWorkspace = new char[StringWorkspaceSize];
	pix_x_override = 0.0f;
	pix_y_override = 0.0f;
	rwdvsConsumer = nullptr;
	Activity = gcnew System::String("Idle");
	PerfDumpFile = nullptr;
	DebugDumpTemplate = nullptr;
	EnableDebugDump = false;
	PreloadQueue = gcnew System::Collections::Generic::Queue<IntPtr>();
}

void MapTracker::Free()
{
	if (pTk)
	{
		delete pTk;
		pTk = 0;
	}
	if (pIC)
	{
		delete pIC;
		pIC = 0;
	}
	if (pCC)
	{
		delete pCC;
		pCC = 0;
	}
	if (pTC)
	{
		delete pTC;
		pTC = 0;
	}
	if (pStringWorkspace)
	{
		delete [] pStringWorkspace;
		pStringWorkspace = 0;
	}
	rwdvsConsumer = nullptr;
	gpu = -1;
}

MapTracker::~MapTracker()
{
	Free();
}

MapTracker::!MapTracker()
{
	Free();
}

char *MapTracker::c_str(System::String ^s)
{
	int i;
	for (i = 0; i < StringWorkspaceSize - 1 && i < s->Length; i++)
		pStringWorkspace[i] = s[i];
	pStringWorkspace[i] = 0;
	return pStringWorkspace;
}

void MapTracker::SetDebugMark(long long v)
{
	debugmark = v;
}

void MapTracker::SetDebugDumpTemplate(System::String ^debdumptemplate)
{
	DebugDumpTemplate = debdumptemplate;
	EnableDebugDump = (DebugDumpTemplate != nullptr);
	pTk->SetEnableDebugDump(EnableDebugDump);
}

void MapTracker::SetVerbosity(int v)
{
	pTk->SetVerbosity(v);
}

void MapTracker::SetPerformanceCounterDumpFile(System::String ^perfdumpfile)
{
	PerfDumpFile = perfdumpfile;
}

void MapTracker::SetPixelXOverride(float v)
{
	pix_x_override = v;
}

void MapTracker::SetPixelYOverride(float v)
{
	pix_y_override = v;
}

void MapTracker::SetGPU(int g)
{	
	if (g < 0 || g >= Utilities::GetAvailableGPUs()) throw gcnew System::String("Invalid GPU number supplied.");
	if (gpu >= 0)
	{
		if (pTk) 
		{
			delete pTk;
			pTk = 0;
		}
		gpu = -1;
	}	
	try
	{
		gpu = g;
		pTk = new GPU::PrismMapTracker(gpu);		
	}
	catch (...)
	{
		pTk = 0;
		gpu = -1;
		throw gcnew System::String("Can't create the tracker on the specified GPU: unmanaged C++ exception occurred.");
	}
}

void MapTracker::SetImageCorrection(System::String ^ics)
{
	if (pIC) 
	{
		std::vector<XMLLite::Element> imcorr_elems;
		imcorr_elems = XMLLite::Element::FromString(c_str(ics));
		if (imcorr_elems.size() != 1) throw gcnew System::String("The XML document must contain only one element.");
		XMLLite::Element &el = imcorr_elems[0];
		if (el.Name != "ImageCorrection") throw gcnew System::String("Wrong ImageCorrection config file.");

		SySal::ImageCorrection ic;

#define CH_CONF_XML(x,f) ic.x = f(el[# x].Value.c_str());

		CH_CONF_XML(DMagDX, atof);
		CH_CONF_XML(DMagDY, atof);
		CH_CONF_XML(DMagDZ, atof);
		CH_CONF_XML(XYCurvature, atof);
		CH_CONF_XML(ZCurvature, atof);
		CH_CONF_XML(XSlant, atof);
		CH_CONF_XML(YSlant, atof);
		CH_CONF_XML(CameraRotation, atof);

#undef CH_CONF_XML

		*pIC = ic;

	}
}

void MapTracker::SetClusterChainerConfig(System::String ^ccs)
{
	if (pCC) 
	{
		std::vector<XMLLite::Element> map_elems;
		map_elems = XMLLite::Element::FromString(c_str(ccs));
		if (map_elems.size() != 1) throw "The XML document must contain only one element.";
		XMLLite::Element &el = map_elems[0];
		if (el.Name != "ClusterChainer.Configuration") throw "Wrong ClusterChainer config file.";

		SySal::ClusterChainer::Configuration cc;

#define CH_CONF_XML(x,f) cc.x = f(el[# x].Value.c_str());/* printf("\nDEBUG-CH_CONF_XML %s val = %s cc.x %d %f", # x, el[# x].Value.c_str(), cc.x, *(float *)(void *)&cc.x);*/
#define CH_CONF_F_XML(x,fn) cc.f ## x = fn(el[# x].Value.c_str());/* printf("\nDEBUG-CH_CONF_XML %s val = %s cc.x %d %f", # x, el[# x].Value.c_str(), cc.x, *(float *)(void *)&cc.x);*/

		CH_CONF_XML(MaxCellContent, atoi);
		CH_CONF_F_XML(CellSize, atof);
		CH_CONF_F_XML(ChainMapMaxXOffset, atof);
		CH_CONF_F_XML(ChainMapMaxYOffset, atof);
		CH_CONF_F_XML(ChainMapMaxZOffset, atof);
		CH_CONF_F_XML(ChainMapXYCoarseTolerance, atof);
		CH_CONF_F_XML(ChainMapXYFineTolerance, atof);
		CH_CONF_F_XML(ChainMapXYFineAcceptance, atof);
		CH_CONF_F_XML(ChainMapZCoarseTolerance, atof);
		CH_CONF_F_XML(ChainMapZFineTolerance, atof);
		CH_CONF_F_XML(ChainMapZFineAcceptance, atof);
		CH_CONF_XML(ChainMapSampleDivider, atoi);
		CH_CONF_XML(ChainMapMinVolume, atoi);
		CH_CONF_XML(MinChainMapsValid, atoi);
		CH_CONF_F_XML(ClusterMapCoarseTolerance, atof);
		CH_CONF_F_XML(ClusterMapFineTolerance, atof);
		CH_CONF_F_XML(ClusterMapFineAcceptance, atof);
		CH_CONF_F_XML(ClusterMapMaxXOffset, atof);
		CH_CONF_F_XML(ClusterMapMaxYOffset, atof);
		CH_CONF_XML(ClusterMapSampleDivider, atoi);
		CH_CONF_XML(MaxChains, atoi);
		CH_CONF_XML(MinClustersPerChain, atoi);
		CH_CONF_XML(ClusterMapMinSize, atoi);
		CH_CONF_XML(MinClusterMapsValid, atoi);
		CH_CONF_XML(MinVolumePerChain, atoi);
		CH_CONF_XML(ClusterThreshold, atoi);

#undef CH_CONF_XML

		*pCC = cc;
	}
}

void MapTracker::SetTrackerConfig(System::String ^tcs)
{
	if (pTC) 
	{
		std::vector<XMLLite::Element> track_elems;
		track_elems = XMLLite::Element::FromString(c_str(tcs));
		if (track_elems.size() != 1) throw "The XML document must contain only one element.";
		XMLLite::Element &el = track_elems[0];
		if (el.Name != "Tracker.Configuration") throw "Wrong Tracker config file.";

		SySal::Tracker::Configuration tc;

#define CH_CONF_XML(x,f) tc.x = f(el[# x].Value.c_str());
#define CH_CONF_F_XML(x,fn) tc.f ## x = fn(el[# x].Value.c_str());/* printf("\nDEBUG-CH_CONF_XML %s val = %s tc.x %d %f", # x, el[# x].Value.c_str(), tc.x, *(float *)(void *)&tc.x);*/

		CH_CONF_F_XML(XYTolerance, atof)
		CH_CONF_F_XML(ZTolerance, atof)
		CH_CONF_F_XML(ZThickness, atof)
		CH_CONF_F_XML(XYHashTableBinSize, atof)	
		CH_CONF_F_XML(ZHashTableBinSize, atof)	
		CH_CONF_XML(ZHashTableBins, atoi)
		CH_CONF_XML(HashBinCapacity, atoi)
		CH_CONF_F_XML(MinLength, atof)
		CH_CONF_F_XML(MaxLength, atof)
		CH_CONF_XML(MaxTracks, atoi)
		CH_CONF_XML(MinVolume, atoi)
		CH_CONF_XML(MinChainVolume, atoi)
		CH_CONF_F_XML(SlopeAcceptanceX, atof)
		CH_CONF_F_XML(SlopeAcceptanceY, atof)
		CH_CONF_F_XML(SlopeCenterX, atof)
		CH_CONF_F_XML(SlopeCenterY, atof)
		CH_CONF_XML(FilterVolumeLength0, atoi)
		CH_CONF_XML(FilterVolumeLength100, atoi)
		CH_CONF_XML(FilterChain0, atoi)
		CH_CONF_XML(FilterChainMult, atof)
		CH_CONF_XML(FilterMinChains, atoi)
		CH_CONF_F_XML(MergeTrackCell, atof)
		CH_CONF_F_XML(MergeTrackXYTolerance, atof)
		CH_CONF_F_XML(MergeTrackZTolerance, atof)

#undef CH_CONF_XML

		*pTC = tc;
	}
}

void MapTracker::SetRawDataViewSideConsumer(IRawDataViewSideConsumer ^v)
{
	rwdvsConsumer = v;
}

float MapTracker::GetCurrentThickness()
{
	try
	{
		return pTk->GetThickness();
	}
	catch(...)
	{
		throw gcnew System::String("No valid thickness information found.");
	}
}

void MapTracker::PreloadFiles(System::Object ^obj)
{
	cli::array<System::String ^> ^inputfiles = (cli::array<System::String ^> ^)obj;
	int i;
	for (i = 0; i < inputfiles->Length && TerminatePreloadThread == false; i++)
	{
		System::String ^fname = inputfiles[i];
		while (PreloadQueue->Count >= PreloadQueueLength) 
		{
			if (TerminatePreloadThread) return;
			System::Threading::Thread::Sleep(100);
		}
		int trial;
		SySal::IntClusterFile *pcf = 0;
		for (trial = 3; trial >= 0; trial--)			
			try
			{
				pcf = new SySal::IntClusterFile(c_str(fname), true);
				break;
			}
			catch (...) 
			{
				if (pcf) { delete pcf; pcf = 0; }
			}
		PreloadQueue->Enqueue((IntPtr)(void *)pcf);
	}
}

void MapTracker::FindTracks(cli::array<System::String ^> ^inputfiles, bool istop, float zsideoffset, SySal::DAQSystem::Scanning::IntercalibrationInfo ^intinfo, unsigned firstview, unsigned lastview)
{
	IntClusterFile EmptyClusterFile;

	SySal::IntClusterFile *pcf = 0;	
	System::Threading::Thread ^preloaderThread = nullptr;
	TerminatePreloadThread = false;
	try
	{
		int i;
		float cscale = 0.0f;		
		int XYScale = pTk->Tracker().GetXYScale();
		int ZScale = pTk->Tracker().GetZScale();
		pTk->SetDebugMarker(debugmark);
		SySal::ClusterChainer &oC = pTk->ClusterChainer();
		bool templatefound = false;
		for (i = 0; i < inputfiles->Length; i++)
		{										
			try
			{
				Activity = gcnew System::String("PreloadFile");
				SySal::IntClusterFile cf(c_str(inputfiles[i]), false);
				if (templatefound == false)				
				{
					EmptyClusterFile.CopyEmpty(cf);
					Activity = gcnew System::String("ConfigClusterChainer");
					cscale = cf.Scale();
					if (pix_x_override != 0.0f && pix_y_override != 0.0f)
					{
						*cf.pPixMicronX = pix_x_override;
						*cf.pPixMicronY = pix_y_override;
					}
					SySal::ClusterChainer::Configuration CC = *pCC;				

					CC.CellSize = CC.fCellSize * cscale;
					CC.ChainMapMaxXOffset = CC.fChainMapMaxXOffset * oC.GetXYScale();
					CC.ChainMapMaxYOffset = CC.fChainMapMaxYOffset * oC.GetXYScale();
					CC.ChainMapMaxZOffset = CC.fChainMapMaxZOffset * oC.GetZScale();
					CC.ChainMapXYCoarseTolerance = CC.fChainMapXYCoarseTolerance * oC.GetXYScale();
					CC.ChainMapXYFineTolerance = CC.fChainMapXYFineTolerance * oC.GetXYScale();
					CC.ChainMapXYFineAcceptance = CC.fChainMapXYFineAcceptance * oC.GetXYScale();
					CC.ChainMapZCoarseTolerance = CC.fChainMapZCoarseTolerance * oC.GetZScale();
					CC.ChainMapZFineTolerance = CC.fChainMapZFineTolerance * oC.GetZScale();
					CC.ChainMapZFineAcceptance = CC.fChainMapZFineAcceptance * oC.GetZScale();
					CC.ClusterMapCoarseTolerance = CC.fClusterMapCoarseTolerance * cscale;
					CC.ClusterMapFineTolerance = CC.fClusterMapFineTolerance * cscale;
					CC.ClusterMapFineAcceptance = CC.fClusterMapFineAcceptance * cscale;
					CC.ClusterMapMaxXOffset = CC.fClusterMapMaxXOffset * cscale;
					CC.ClusterMapMaxYOffset = CC.fClusterMapMaxYOffset * cscale;
					try
					{
						Activity = gcnew System::String("Reset");
						oC.Reset(CC, *pIC, istop);
					}
					catch (char *x)
					{
						printf("\nERROR-AT-GPU-LEVEL-EXCEPTION.\nError in \"%s\"\nPerforming hard reset.", c_str(Activity));
						Activity = gcnew System::String("Idle");
						pTk->HardReset();
						printf("\nHard reset done");				
						throw gcnew System::Exception(gcnew System::String("GPU-level exception - Hard reset performed."));				
					}
					templatefound = true;
				}
				Activity = gcnew System::String("SetReferenceZs");
				bool valid = oC.SetReferenceZs(cf, istop);
			}
			catch (char *x)
			{
				printf("\nWarning: Cannot preload file %s", c_str(inputfiles[i]));
				//throw gcnew System::String("Cannot preload file ") + inputfiles[i];
			}
		}
		if (templatefound == false)
		{
			printf("\nERROR: Could not find any valid file to use as template.");
			throw gcnew System::String("Could not find any valid file to use as template.");			
		}
		float thickness = -1.0;
		try
		{
			thickness = GetCurrentThickness();
			//printf("\nThickness %f", thickness);
		}
		catch (char *x)
		{
			printf("\nThickness error %s", x);
			throw gcnew System::String("Thickness error.");
		}
		catch (System::String ^sx)
		{
			printf("\nThickness error %s", c_str(sx));
			throw gcnew System::String("Thickness error.");
		}

		Activity = gcnew System::String("ConfigTracker");
		SySal::Tracker &oT = pTk->Tracker();
		{
			SySal::Tracker::Configuration TC = *pTC;

			TC.XYTolerance = TC.fXYTolerance * oT.GetXYScale();
			TC.ZTolerance = TC.fZTolerance * oT.GetZScale();
			TC.ZThickness = TC.fZThickness * oT.GetZScale();
			TC.XYHashTableBinSize = TC.fXYHashTableBinSize * oT.GetXYScale();
			TC.ZHashTableBinSize = TC.fZHashTableBinSize * oT.GetZScale();
			TC.MinLength = TC.fMinLength * oT.GetXYScale();
			TC.MaxLength = TC.fMaxLength * oT.GetXYScale();
			TC.SlopeAcceptanceX = TC.fSlopeAcceptanceX * oT.GetSlopeScale();
			TC.SlopeAcceptanceY = TC.fSlopeAcceptanceY * oT.GetSlopeScale();
			TC.SlopeCenterX = TC.fSlopeCenterX * oT.GetSlopeScale();
			TC.SlopeCenterY = TC.fSlopeCenterY * oT.GetSlopeScale();
			TC.MergeTrackCell = TC.fMergeTrackCell * oT.GetXYScale();
			TC.MergeTrackXYTolerance = TC.fMergeTrackXYTolerance * oT.GetXYScale();
			TC.MergeTrackZTolerance = TC.fMergeTrackZTolerance * oT.GetZScale();
			try
			{
				oT.Reset(TC);
			}
			catch (char *xstr)
			{
				throw gcnew System::Exception(gcnew System::String(xstr));
			}
		}	

		preloaderThread = gcnew System::Threading::Thread(gcnew System::Threading::ParameterizedThreadStart(this, &MapTracker::PreloadFiles));
		preloaderThread->Start(inputfiles);
		for (i = firstview; i < inputfiles->Length && i <= lastview; i++)
		{
			try
			{
				Activity = gcnew System::String("LoadFile");
				while (PreloadQueue->Count == 0)
				{
					if (preloaderThread->Join(10) && PreloadQueue->Count == 0) throw gcnew System::Exception("Can't load file " + inputfiles[i]);					
				}
				pcf = (SySal::IntClusterFile *)(void *)PreloadQueue->Dequeue();
				if (pcf == 0) pcf = &EmptyClusterFile;
				if (pix_x_override != 0.0f && pix_y_override != 0.0f)
				{
					*pcf->pPixMicronX = pix_x_override;
					*pcf->pPixMicronY = pix_y_override;
				}
				SySal::ClusterChainer::EmulsionFocusInfo ef;
				SySal::TrackMapHeader *pTkHdr = 0;
				try
				{					
					Activity = gcnew System::String("AddClusters");
//					for (int __i =  0; __i < pcf->Images(); __i++)
//						printf("\nDEBUG\t%d %f %d", __i, pcf->StageZ(__i), pcf->ImageClusterCounts(__i));
					ef = oC.AddClusters(*pcf);					
					Activity = gcnew System::String("Tracking");
					pTkHdr = oT.Dump();
				}
				catch (...)
				{
					if (pcf != &EmptyClusterFile)
					{
						delete pcf;
						pcf = 0;
					}
					printf("\nERROR-AT-GPU-LEVEL-EXCEPTION.\nError in \"%s\" side %s view %d\nPerforming hard reset.", c_str(Activity), istop ? "TOP" : "BOTTOM", i);
					Activity = gcnew System::String("Idle");					
					pTk->HardReset();
					printf("\nHard reset done");				
					throw gcnew System::Exception(gcnew System::String("GPU-level exception - Hard reset performed."));
				}
				if (rwdvsConsumer != nullptr && pTkHdr != 0 &&
					((i > firstview) || (oT.IsFirstViewEmpty() == false || i == 0)) ||
					((i == inputfiles->Length - 1) || (oT.IsLastViewEmpty() == false || i == lastview))
					)
				{
					Activity = gcnew System::String("FormattingOutput");
					RawDataViewSide ^rwdds = gcnew RawDataViewSide(pTkHdr, *pcf, ef, zsideoffset, intinfo, istop);
					if (pcf != &EmptyClusterFile)
					{
						delete pcf;
						pcf = 0;
					}
					Activity = gcnew System::String("WritingOutput");
					rwdvsConsumer->ConsumeData(i, istop, rwdds);
				}
				else
				{
					if (pcf != &EmptyClusterFile)
					{
						delete pcf;
						pcf = 0;
					}
				}
				if (PerfDumpFile != nullptr)
				{
					Activity = gcnew System::String("PerformanceCounterDump");
					try
					{
						PrismMapTracker::PerformanceCounters pfc = pTk->GetPerformanceCounters();
						if (System::IO::File::Exists(PerfDumpFile) == false)
							System::IO::File::WriteAllText(PerfDumpFile, "GPU\tGPUClockMHz\tGPUCores\tClusters\tChains\tTracks\tMapTimeMS\tTrackTimeMS");
						System::IO::File::AppendAllText(PerfDumpFile, gcnew System::String("\n") + 
							pfc.GPU.ToString() + gcnew System::String("\t") + 
							pfc.GPUClockMHz.ToString() + gcnew System::String("\t") + 
							pfc.GPUCores.ToString() + gcnew System::String("\t") + 
							pfc.Clusters.ToString() + gcnew System::String("\t") + 
							pfc.Chains.ToString() + gcnew System::String("\t") + 
							pfc.Tracks.ToString() + gcnew System::String("\t") + 
							pfc.MapTimeMS.ToString() + gcnew System::String("\t") + 
							pfc.TrackTimeMS.ToString());
					}
					catch (Exception ^xx) {}
				}
				if (DebugDumpTemplate != nullptr)
					try
					{
						PrismMapTracker::DebugDump dd = pTk->GetDebugDump();
						System::IO::StreamWriter ^swr = gcnew System::IO::StreamWriter(DebugDumpTemplate + gcnew System::String("_") + pTk->GetDebugMarker() + gcnew System::String("_") + i.ToString());
						swr->WriteLine(gcnew System::String("-View ") + i.ToString());
						swr->WriteLine(gcnew System::String("-MapValid ") + dd.MapValid.ToString());
						swr->WriteLine(gcnew System::String("-DeltaX ") + dd.ViewDeltaX.ToString());
						swr->WriteLine(gcnew System::String("-DeltaY ") + dd.ViewDeltaY.ToString());
						swr->WriteLine(gcnew System::String("-HashTableBounds MinX: ") + dd.HTBounds.MinX.ToString() + gcnew System::String(" MaxX: ") + dd.HTBounds.MaxX.ToString());
						swr->WriteLine(gcnew System::String("-HashTableBounds MinY: ") + dd.HTBounds.MinY.ToString() + gcnew System::String(" MaxY: ") + dd.HTBounds.MaxY.ToString());
						swr->WriteLine(gcnew System::String("-HashTableBounds MinZ: ") + dd.HTBounds.MinZ.ToString() + gcnew System::String(" MaxZ: ") + dd.HTBounds.MaxZ.ToString());
						swr->WriteLine(gcnew System::String("-HashTableBounds XBins: ") + dd.HTBounds.XBins.ToString() + gcnew System::String(" YBins: ") + dd.HTBounds.YBins.ToString() + gcnew System::String(" ZBins: ") + dd.HTBounds.ZBins.ToString() + gcnew System::String(" NBins: ") + dd.HTBounds.NBins.ToString());
						swr->WriteLine(gcnew System::String("-HashTableBounds XTBins:") + dd.HTBounds.XTBins.ToString() + gcnew System::String(" YTBins: ") + dd.HTBounds.YTBins.ToString() + gcnew System::String(" NTBins: ") + dd.HTBounds.NTBins.ToString());
						for (int imt = 0; imt < dd.Images; imt++)
						{
							swr->WriteLine(gcnew System::String("--Image ") + imt.ToString() + gcnew System::String("/") + dd.Images.ToString() + gcnew System::String(" Clusters ") + dd.pImages[imt].Clusters.ToString() + gcnew System::String(" MapClusters ") + dd.pImages[imt].MappingClusters.ToString() + gcnew System::String(" DeltaX ") + dd.pImages[imt].DeltaX.ToString() + gcnew System::String(" DeltaY ") + dd.pImages[imt].DeltaY.ToString());
						}
						for (int _i = 0; _i < 2; _i++)
						{
							swr->WriteLine(gcnew System::String("-ChainView ") + _i.ToString() + gcnew System::String(": pV ") + ((long long)dd.Views[_i].pV).ToString("X016"));
							swr->WriteLine(gcnew System::String("--Delta ") + dd.Views[_i].pHostView->DeltaX.ToString() + "/" + dd.Views[_i].pHostView->DeltaY.ToString() + "/" + dd.Views[_i].pHostView->DeltaZ.ToString());
							swr->WriteLine(gcnew System::String("--Position ") + dd.Views[_i].pHostView->PositionX.ToString() + "/" + dd.Views[_i].pHostView->PositionY.ToString() + "/" + dd.Views[_i].pHostView->PositionZ.ToString());
							swr->WriteLine(gcnew System::String("--Count ") + dd.Views[_i].pHostView->Count.ToString());
							swr->WriteLine(gcnew System::String("--Chains BEGIN"));
							for (int _j = 0; _j < dd.Views[_i].pHostView->Count; _j++)
								swr->WriteLine(gcnew System::String("---Chain ") + _j.ToString() + gcnew System::String(" ") + dd.Views[_i].pHostView->Chains[_j].Volume.ToString() + gcnew System::String(" ") + dd.Views[_i].pHostView->Chains[_j].AvgX.ToString() + gcnew System::String(" ") + dd.Views[_i].pHostView->Chains[_j].AvgY.ToString() + gcnew System::String(" ") + dd.Views[_i].pHostView->Chains[_j].AvgZ.ToString());
							swr->WriteLine(gcnew System::String("--Chains END"));
						}
						for (int tit = 0; tit < dd.TrackIterations; tit++)
						{
							int iybase, ix;
							swr->WriteLine(gcnew System::String("--Track Iteration ") + tit.ToString() + gcnew System::String("/") + dd.TrackIterations.ToString() + gcnew System::String(" ISX ") + dd.pTrackIterations[tit].ISX.ToString() + gcnew System::String(" ISY ") + dd.pTrackIterations[tit].ISY.ToString() + gcnew System::String(" TotalPairs ") + dd.pTrackIterations[tit].TotalPairs);
							swr->WriteLine(gcnew System::String("---pBinFill size = ") + dd.pTrackIterations[tit].sz_pBinFill.ToString());
							swr->WriteLine(gcnew System::String("---pPairIndices size = ") + dd.pTrackIterations[tit].sz_pPairIndices.ToString());
							swr->WriteLine(gcnew System::String("---TotalPairs size = ") + dd.pTrackIterations[tit].sz_TotalPairs.ToString());
							swr->WriteLine(gcnew System::String("---pTBinFill size = ") + dd.pTrackIterations[tit].sz_pTBinFill.ToString());
							for (int r = 0; r < PrismMapTracker::DebugDump::t_TrackIteration::Replicas; r++)
							{
								swr->WriteLine(gcnew System::String("----pBinFill BEGIN ") + r.ToString());
								for (iybase = 0; iybase < dd.pTrackIterations[tit].sz_pBinFill; iybase += dd.HTBounds.XBins)
								{
									for (ix = 0; ix < dd.HTBounds.XBins && iybase + ix < dd.pTrackIterations[tit].sz_pBinFill; ix++)
										swr->Write(dd.pTrackIterations[tit].pBinFill[r][iybase + ix].ToString("X08") + gcnew System::String(" "));
									swr->WriteLine();
								}
								swr->WriteLine(gcnew System::String("----pBinFill END ") + r.ToString());
							}
							swr->WriteLine(gcnew System::String("----pPairIndices BEGIN(first 1000)"));
							for (int ip = 0; ip < 1000 && ip < dd.pTrackIterations[tit].sz_pPairIndices; ip++)
								swr->Write(dd.pTrackIterations[tit].pPairIndices[ip].ToString() + gcnew System::String(" "));
							swr->WriteLine();
							swr->WriteLine(gcnew System::String("----pPairIndices END"));
							swr->WriteLine(gcnew System::String("----pPairComputers BEGIN(first 1000 and last 1000)"));
							for (int ic = 0; ic < dd.pTrackIterations[tit].sz_TotalPairs; ic++)
								if (ic < 1000 || (dd.pTrackIterations[tit].sz_TotalPairs - ic) < 1000)
									swr->Write(dd.pTrackIterations[tit].pPairComputer[ic].ToString() + gcnew System::String(" "));
								else if (ic == 1000)
									swr->Write(gcnew System::String(" ... "));
							swr->WriteLine();
							swr->WriteLine(gcnew System::String("----pPairComputers END"));
							swr->WriteLine(gcnew System::String("----pTBinFill BEGIN"));
							for (iybase = 0; iybase < dd.pTrackIterations[tit].sz_pTBinFill; iybase += dd.HTBounds.XTBins)
							{
								for (ix = 0; ix < dd.HTBounds.XTBins && iybase + ix < dd.pTrackIterations[tit].sz_pTBinFill; ix++)
									swr->Write(dd.pTrackIterations[tit].pTBinFill[iybase + ix].ToString() + gcnew System::String(" "));
								swr->WriteLine();
							}
							swr->WriteLine(gcnew System::String("----pTBinFill END"));
						};
						swr->Flush();
						swr->Close();
					}
					catch (Exception ^xx) {}
			}
			catch (char *x)
			{			
				throw gcnew System::Exception(gcnew System::String(x));
			}			
		}
	}
	catch (System::Exception ^x)
	{
		Console::WriteLine("\r\n" + Activity);
		throw x;
	}
	catch (...)
	{
		Console::WriteLine("\r\nGENERIC ERROR in " + Activity);
		throw gcnew System::Exception("GENERIC");
	}
	finally
	{		
		TerminatePreloadThread = true;
		if (preloaderThread != nullptr)
			preloaderThread->Join();
		if (pcf && pcf != &EmptyClusterFile)
		{
			delete pcf;
			pcf = 0;
		}
		while (PreloadQueue->Count > 0)
		{
			pcf = (SySal::IntClusterFile *)(void *)PreloadQueue->Dequeue();
			delete pcf;
		}
		Activity = gcnew System::String("Idle");
	}
}

System::String ^SySal::GPU::MapTracker::GetCurrentActivity() { return Activity; }

RawDataViewSide::RawDataViewSide(SySal::TrackMapHeader *ptkhdr, SySal::IntClusterFile &cf, SySal::ClusterChainer::EmulsionFocusInfo ef, float sideoffset, SySal::DAQSystem::Scanning::IntercalibrationInfo ^intinfo, bool istop)
{	
	m_Pos.X = (ptkhdr->MinX + ptkhdr->MaxX) * 0.5 / ptkhdr->XYScale;
	m_Pos.Y = (ptkhdr->MinY + ptkhdr->MaxY) * 0.5 / ptkhdr->XYScale;
	m_MapPos = intinfo->Transform(m_Pos);
	SetM(this, intinfo->MXX, intinfo->MXY, intinfo->MYX, intinfo->MYY);	
	m_Flags = SySal::Scanning::Plate::IO::OPERA::RawData::Fragment::View::Side::SideFlags::OK;
	cli::array<SySal::Scanning::Plate::IO::OPERA::RawData::Fragment::View::Side::LayerInfo> ^linfos = 
		gcnew cli::array<SySal::Scanning::Plate::IO::OPERA::RawData::Fragment::View::Side::LayerInfo>(cf.Images());
	int i;
	for (i = 0; i < linfos->Length; i++)
	{
		linfos[i].Grains = cf.ImageClusterCounts(i);
		linfos[i].Z = cf.StageZ(i);
	}
	m_Layers = gcnew SySal::Scanning::Plate::IO::OPERA::RawData::Fragment::View::Side::LayerInfoList(linfos);
	if (istop)
	{
		m_TopZ = sideoffset + (ef.Top.Z - ef.Bottom.Z);
		m_BottomZ = sideoffset;
	}
	else
	{
        m_BottomZ = sideoffset - (ef.Top.Z - ef.Bottom.Z);
		m_TopZ = sideoffset;
	}
	m_Tracks = gcnew cli::array<SySal::Scanning::MIPIndexedEmulsionTrack ^>(ptkhdr->Count);
	double ixyscale = 1.0 / ptkhdr->XYScale;
	double izscale = 1.0 / ptkhdr->ZScale;
	for (i = 0; i < m_Tracks->Length; i++)
	{
		SySal::Tracking::MIPEmulsionTrackInfo ^minfo = gcnew SySal::Tracking::MIPEmulsionTrackInfo();
		IntTrack &itk = ptkhdr->Tracks[i];
		minfo->Field = i;
		minfo->Count = itk.Quality;
		minfo->AreaSum = itk.Volume;
		minfo->Slope.X = ((itk.X1 - itk.X2) * ixyscale) / ((itk.Z1 - itk.Z2) * izscale);
		minfo->Slope.Y = ((itk.Y1 - itk.Y2) * ixyscale) / ((itk.Z1 - itk.Z2) * izscale);
		minfo->Slope.Z = 1.0;
		minfo->Intercept.Z = sideoffset;
		bool usez1 = (istop == true && itk.Z1 < itk.Z2) || (istop == false && itk.Z1 > itk.Z2);
		if (usez1)
		{
			minfo->Intercept.X = itk.X1 * ixyscale - itk.Z1 * izscale * minfo->Slope.X - m_Pos.X;
			minfo->Intercept.Y = itk.Y1 * ixyscale - itk.Z1 * izscale * minfo->Slope.Y - m_Pos.Y;
		}
		else
		{
			minfo->Intercept.X = itk.X2 * ixyscale - itk.Z2 * izscale * minfo->Slope.X - m_Pos.X;
			minfo->Intercept.Y = itk.Y2 * ixyscale - itk.Z2 * izscale * minfo->Slope.Y - m_Pos.Y;
		}
		minfo->TopZ = sideoffset + Math::Max(itk.Z1 * izscale, itk.Z2 * izscale);
		minfo->BottomZ = sideoffset + Math::Min(itk.Z1 * izscale, itk.Z2 * izscale);		
		minfo->Sigma = Math::Sqrt(((itk.X1 - itk.X2) * ixyscale) * ((itk.X1 - itk.X2) * ixyscale) + ((itk.Y1 - itk.Y2) * ixyscale) * ((itk.Y1 - itk.Y2) * ixyscale) + ((itk.Z1 - itk.Z2) * izscale) * ((itk.Z1 - itk.Z2) * izscale));
		SySal::Scanning::MIPIndexedEmulsionTrack ^mtk = gcnew SySal::Scanning::MIPIndexedEmulsionTrack(minfo, nullptr, i);
		m_Tracks[i] = mtk;
	}	
}

}
}