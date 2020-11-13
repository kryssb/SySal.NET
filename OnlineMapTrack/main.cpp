#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <memory.h>
#include "xml-lite.h"
#include "gpu_util.h"
#include <vector>
#include <string>
#include "map.h"
#include "Tracker.h"
#include "gpu_incremental_map_track.h"
#include <time.h>
#include <math.h>

bool read_parameters(int argc, char *argv[], std::string &input_file_list, std::string &imcorr_cfg_file, std::string &mapper_cfg_file, std::string &tracker_cfg_file, bool &istop, int &gpu, float &pix_x_override, float &pix_y_override, int &loop_limiter);

bool read_image_correction(string &filename, SySal::ImageCorrection &C);

bool read_mapper_cfg(string &filename, SySal::ClusterChainer::Configuration &C, int cscale, SySal::ClusterChainer &oC);

bool read_tracker_cfg(string &filename, SySal::Tracker::Configuration &C, SySal::Tracker &oT);

int XYScale = 0;
int ZScale = 0;

void dumpchains(void *pctx, SySal::ChainView *pLastView, SySal::ChainView *pThisView);

int main(int argc, char *argv[])
{
	std::string imcorr_cfg_file;
	std::string mapper_cfg_file;
	std::string tracker_cfg_file;
	std::string input_file_list;
	int gpu = 0;
	float pix_x_override = 0.0f;
	float pix_y_override = 0.0f;
	int loop_limiter = 2;
	bool istop = false;

	SySal::ImageCorrection IC;
	SySal::ClusterChainer::Configuration CC;
	SySal::Tracker::Configuration TC;


	int gpuavail = SySal::GPU::GetAvailableGPUs();
	if (argc == 1 || read_parameters(argc, argv, input_file_list, imcorr_cfg_file, mapper_cfg_file, tracker_cfg_file, istop, gpu, pix_x_override, pix_y_override, loop_limiter) == false)
	{
		printf(	"\nusage: OnlineMapTrack <gpuid> <t|b> <input file list> <imcorr_cfg_file> <mapper_cfg_file> <tracker_cfg_file> {parameters}"
				"\nt -> top, b -> bottom"
				"\nCUDA devices available: %d"
				"\nparameters are:"
				"\n  /pix <X> <Y> -> overridden pixel-micron conversion factor (default is value of cls files)"
				"\n  /loopmax <limiter> -> maximum loops (default is 2)", gpuavail);
		return -1;
	}
	if (gpu < 0 || gpu >= gpuavail)
	{
		printf("\nERROR: Invalid GPU Id requested.");
		return -1;
	}
	FILE *finput = fopen(input_file_list.c_str(), "rt");
	if (finput == 0)
	{
		printf("\nERROR: Can't open input file.");
		return -1;
	}


	std::vector<std::string> inputfiles;
	while (!feof(finput))
	{
		static char tempstr[1024];
		if (fscanf_s(finput, "%s", tempstr, sizeof(tempstr) - 1) == 0) break;		
		if (strlen(tempstr) > 0)
		{
			std::string s = tempstr;
			inputfiles.push_back(s);
			printf("\nFILE \"%s\"", s.c_str());
		}
	}
	fclose(finput);
	printf("\n%d files in input", inputfiles.size());
	if (inputfiles.size() == 0)
	{
		printf("\nNothing to do, closing.");
		return 0;
	}


	if (read_image_correction(imcorr_cfg_file, IC) == false)
	{
		printf("\nERROR: Invalid image correction file.");
		return false;
	}	

	int i;
	float cscale = 0.0f;

	SySal::GPU::PrismMapTracker Tk(gpu);
	XYScale = Tk.Tracker().GetXYScale();
	ZScale = Tk.Tracker().GetZScale();

	for (i = 0; i < inputfiles.size(); i++)
	{
		try
		{
			SySal::IntClusterFile cf(inputfiles[i].c_str(), false);
			if (i == 0) 
			{
				cscale = cf.Scale();
				if (pix_x_override != 0.0f && pix_y_override != 0.0f)
				{
					*cf.pPixMicronX = pix_x_override;
					*cf.pPixMicronY = pix_y_override;
				}
				if (read_mapper_cfg(mapper_cfg_file, CC, cscale, Tk.ClusterChainer()) == false)
				{
					printf("\nERROR: Invalid cluster chainer configuration file.");
					return false;
				}
				Tk.ClusterChainer().Reset(CC, IC, istop);
			}
			bool valid = Tk.ClusterChainer().SetReferenceZs(cf, istop);
			printf("\n%s thickness info %s", inputfiles[i].c_str(), valid ? "OK" : "SKIP");
		}
		catch (char *x)
		{
			printf("\nERROR: Cannot preload file %s.\r\nError: %s", inputfiles[i].c_str(), x);
			return -1;
		}
	}

	try
	{
		printf("\nMedian thickness %lf", Tk.GetThickness());
	}
	catch (char *xs)
	{
		printf("\nNo suitable thickness information found, aborting.");
		return -1;
	}

	if (read_tracker_cfg(tracker_cfg_file, TC, Tk.Tracker()) == false)
	{
		printf("\nERROR: Invalid tracker configuration file.");
		return false;
	}

	Tk.Tracker().Reset(TC);

	int nettime = 0;
	clock_t start = clock();
	for (i = 0; i < inputfiles.size(); i++)
	{
		printf("\nDEBUG-FILE %d", i);
		try
		{
			SySal::IntClusterFile cf(inputfiles[i].c_str(), true);
			if (pix_x_override != 0.0f && pix_y_override != 0.0f)
			{
				*cf.pPixMicronX = pix_x_override;
				*cf.pPixMicronY = pix_y_override;
			}
			if (i > 0)
			{
				//Tk.SetChainDumper(0, dumpchains);
			}
			clock_t nets1 = clock();
			Tk.ClusterChainer().AddClusters(cf);			
			clock_t nete1 = clock();
			printf("\nTracks: %d", Tk.Tracker().TotalTracks());

			string outfname = inputfiles[i];
			outfname = outfname.substr(0, outfname.rfind('.')).append(".track");
			FILE *outf = fopen(outfname.c_str(), "wb");
			if (outf == 0) throw "Can't open output file!";
			SySal::TrackMapHeader *pTkHdr = Tk.Tracker().Dump();
			fwrite(pTkHdr, pTkHdr->TotalSize(), 1, outf);
			fclose(outf);
			printf("\nWritten %s", outfname.c_str());

			outfname = inputfiles[i];
			outfname = outfname.substr(0, outfname.rfind('.')).append(".doc.track");
			FILE *track_dump = fopen(outfname.c_str(), "wt");
			fprintf(track_dump, "ID CHAINS VOLUME X1 Y1 Z1 X2 Y2 Z2 Q AID X Y DZ SX SY");

			int j;
			for (j = 0; j < pTkHdr->Count; j++)
				fprintf(track_dump, "\n%d %d %d %d %d %d %d %d %d %d %d %.1f %.1f %.1f %.4f %.4f", 
					j, pTkHdr->Tracks[j].Chains, pTkHdr->Tracks[j].Volume,
					pTkHdr->Tracks[j].X1, pTkHdr->Tracks[j].Y1, pTkHdr->Tracks[j].Z1,
					pTkHdr->Tracks[j].X2, pTkHdr->Tracks[j].Y2, pTkHdr->Tracks[j].Z2,
					pTkHdr->Tracks[j].Quality, 0, 
					pTkHdr->Tracks[j].X1 / (double)pTkHdr->XYScale,
					pTkHdr->Tracks[j].Y1 / (double)pTkHdr->XYScale,
					fabs((pTkHdr->Tracks[j].Z1 - pTkHdr->Tracks[j].Z2) / (double)pTkHdr->ZScale),
					((pTkHdr->Tracks[j].X1 - pTkHdr->Tracks[j].X2) / (double)pTkHdr->XYScale) / ((pTkHdr->Tracks[j].Z1 - pTkHdr->Tracks[j].Z2) / (double)pTkHdr->ZScale),
					((pTkHdr->Tracks[j].Y1 - pTkHdr->Tracks[j].Y2) / (double)pTkHdr->XYScale) / ((pTkHdr->Tracks[j].Z1 - pTkHdr->Tracks[j].Z2) / (double)pTkHdr->ZScale)
					);
			fclose(track_dump);
			printf("\nNET time %d", (nete1 - nets1));
			nettime += (nete1 - nets1);
		}
		catch (char *x)
		{
			printf("\nERROR: %s.", x);
			return -1;
		}
	}
	clock_t end = clock();
	printf("\n%d %d", end - start, nettime);


	return 0;
}


bool read_parameters(int argc, char *argv[], std::string &input_file_list, std::string &imcorr_cfg_file, std::string &mapper_cfg_file, std::string &track_cfg_file, bool &istop, int &gpu, float &pix_x_override, float &pix_y_override, int &loop_limiter)
{
	if (argc < 2) return false;
	gpu = atoi(argv[1]);
	if (argc < 3) return false;
	if (strcmp(argv[2], "t") == 0) istop = true;
	else if (strcmp(argv[2], "b") == 0) istop = false;
	else return false;
	if (argc < 4) return false;
	input_file_list = argv[3];
	if (argc < 5) return false;
	imcorr_cfg_file = argv[4];
	if (argc < 6) return false;
	mapper_cfg_file = argv[5];
	if (argc < 7) return false;
	track_cfg_file = argv[6];
	int argbase = 7;
	while (argbase < argc)
	{
		if (strcmp(argv[argbase], "/pix") == 0)
		{
			if (argc < argbase + 3) return false;
			pix_x_override = atof(argv[argbase + 1]);
			pix_y_override = atof(argv[argbase + 2]);
			argbase += 3;
		} 
		else if (strcmp(argv[argbase], "/loopmax") == 0)
		{
			if (argc < argbase + 2) return false;
			loop_limiter = atoi(argv[argbase + 1]);
			argbase += 2;
		} 
		else return false;
	}
	return true;
}

bool read_image_correction(string &filename, SySal::ImageCorrection &C)
{
	FILE *f = fopen(filename.c_str(), "rt");
	if (f == 0) return false;
	string xmlf = "";
	while (!feof(f)) xmlf += fgetc(f);
	fclose(f);
	try
	{
		std::vector<XMLLite::Element> imcorr_elems;
		imcorr_elems = XMLLite::Element::FromString(xmlf.c_str());
		if (imcorr_elems.size() != 1) throw "The XML document must contain only one element.";
		XMLLite::Element &el = imcorr_elems[0];
		if (el.Name != "ImageCorrection") throw "Wrong ImageCorrection config file.";

#define CH_CONF_XML(x,f) C.x = f(el[# x].Value.c_str());

		CH_CONF_XML(DMagDX, atof);
		CH_CONF_XML(DMagDY, atof);
		CH_CONF_XML(DMagDZ, atof);
		CH_CONF_XML(XYCurvature, atof);
		CH_CONF_XML(ZCurvature, atof);
		CH_CONF_XML(XSlant, atof);
		CH_CONF_XML(YSlant, atof);
		CH_CONF_XML(CameraRotation, atof);

#undef CH_CONF_XML
	}
	catch (const char *xc)
	{
		printf("\nERROR: %s", xc);
		return false;
	}

	return true;
}

bool read_mapper_cfg(string &filename, SySal::ClusterChainer::Configuration &C, int cscale, SySal::ClusterChainer &oC)
{
	FILE *f = fopen(filename.c_str(), "rt");
	if (f == 0) return false;
	string xmlf = "";
	while (!feof(f)) xmlf += fgetc(f);
	fclose(f);
	try
	{
		std::vector<XMLLite::Element> map_elems;
		map_elems = XMLLite::Element::FromString(xmlf.c_str());
		if (map_elems.size() != 1) throw "The XML document must contain only one element.";
		XMLLite::Element &el = map_elems[0];
		if (el.Name != "ClusterChainer.Configuration") throw "Wrong ClusterChainer config file.";

#define CH_CONF_XML(x,f) C.x = f(el[# x].Value.c_str())
		CH_CONF_XML(MaxCellContent, atoi);
		CH_CONF_XML(CellSize, atof) * cscale;
		CH_CONF_XML(ChainMapMaxXOffset, atof) * oC.GetXYScale();
		CH_CONF_XML(ChainMapMaxYOffset, atof) * oC.GetXYScale();
		CH_CONF_XML(ChainMapMaxZOffset, atof) * oC.GetZScale();
		CH_CONF_XML(ChainMapXYCoarseTolerance, atof) * oC.GetXYScale();
		CH_CONF_XML(ChainMapXYFineTolerance, atof) * oC.GetXYScale();
		CH_CONF_XML(ChainMapXYFineAcceptance, atof) * oC.GetXYScale();
		CH_CONF_XML(ChainMapZCoarseTolerance, atof) * oC.GetZScale();
		CH_CONF_XML(ChainMapZFineTolerance, atof) * oC.GetZScale();
		CH_CONF_XML(ChainMapZFineAcceptance, atof) * oC.GetZScale();
		CH_CONF_XML(ChainMapSampleDivider, atoi);
		CH_CONF_XML(ChainMapMinVolume, atoi);
		CH_CONF_XML(MinChainMapsValid, atoi);
		CH_CONF_XML(ClusterMapCoarseTolerance, atof) * cscale;
		CH_CONF_XML(ClusterMapFineTolerance, atof) * cscale;
		CH_CONF_XML(ClusterMapFineAcceptance, atof) * cscale;
		CH_CONF_XML(ClusterMapMaxXOffset, atof) * cscale;
		CH_CONF_XML(ClusterMapMaxYOffset, atof) * cscale;
		CH_CONF_XML(ClusterMapSampleDivider, atoi);
		CH_CONF_XML(MaxChains, atoi);
		CH_CONF_XML(MinClustersPerChain, atoi);
		CH_CONF_XML(ClusterMapMinSize, atoi);
		CH_CONF_XML(MinClusterMapsValid, atoi);
		CH_CONF_XML(MinVolumePerChain, atoi);
		CH_CONF_XML(ClusterThreshold, atoi);

#undef CH_CONF_XML
	}
	catch (const char *xc)
	{
		printf("\nERROR: %s", xc);
		return false;
	}

	return true;
}

bool read_tracker_cfg(string &filename, SySal::Tracker::Configuration &C, SySal::Tracker &oT)
{
	FILE *f = fopen(filename.c_str(), "rt");
	if (f == 0) return false;
	string xmlf = "";
	while (!feof(f)) xmlf += fgetc(f);
	fclose(f);
	try
	{
		std::vector<XMLLite::Element> track_elems;
		track_elems = XMLLite::Element::FromString(xmlf.c_str());
		if (track_elems.size() != 1) throw "The XML document must contain only one element.";
		XMLLite::Element &el = track_elems[0];
		if (el.Name != "Tracker.Configuration") throw "Wrong Tracker config file.";

#define CH_CONF_XML(x,f) C.x = f(el[# x].Value.c_str());
#define CH_CONF_XML_XYSCALE(x,f) C.x = f(el[# x].Value.c_str()) * oT.GetXYScale();
#define CH_CONF_XML_ZSCALE(x,f) C.x = f(el[# x].Value.c_str()) * oT.GetZScale();
#define CH_CONF_XML_SLOPESCALE(x,f) C.x = f(el[# x].Value.c_str()) * oT.GetSlopeScale();

		CH_CONF_XML_XYSCALE(XYTolerance, atof)
		CH_CONF_XML_ZSCALE(ZTolerance, atof)
		CH_CONF_XML_ZSCALE(ZThickness, atof)
		CH_CONF_XML_XYSCALE(XYHashTableBinSize, atof)	
		CH_CONF_XML_ZSCALE(ZHashTableBinSize, atof)	
		CH_CONF_XML(ZHashTableBins, atoi)
		CH_CONF_XML(HashBinCapacity, atoi)
		CH_CONF_XML_XYSCALE(MinLength, atof)
		CH_CONF_XML_XYSCALE(MaxLength, atof)
		CH_CONF_XML(MaxTracks, atoi)
		CH_CONF_XML(MinVolume, atoi)
		CH_CONF_XML(MinChainVolume, atoi)
		CH_CONF_XML_SLOPESCALE(SlopeAcceptanceX, atof)
		CH_CONF_XML_SLOPESCALE(SlopeAcceptanceY, atof)
		CH_CONF_XML_SLOPESCALE(SlopeCenterX, atof)
		CH_CONF_XML_SLOPESCALE(SlopeCenterY, atof)
		CH_CONF_XML(FilterVolumeLength0, atoi)
		CH_CONF_XML(FilterVolumeLength100, atoi)
		CH_CONF_XML(FilterChain0, atoi)
		CH_CONF_XML(FilterChainMult, atof)
		CH_CONF_XML(ClusterVol0, atoi)
		CH_CONF_XML(ClusterVolM, atoi)
		CH_CONF_XML_XYSCALE(MergeTrackCell, atof)
		CH_CONF_XML_XYSCALE(MergeTrackXYTolerance, atof)
		CH_CONF_XML_ZSCALE(MergeTrackZTolerance, atof)

#undef CH_CONF_XML_SLOPESCALE
#undef CH_CONF_XML_ZSCALE
#undef CH_CONF_XML_XYSCALE
#undef CH_CONF_XML
	}
	catch (const char *xc)
	{
		printf("\nERROR: %s", xc);
		return false;
	}

	return true;
}

void dumpchains(void *pctx, SySal::ChainView *pLastView, SySal::ChainView *pThisView)
{
	FILE *f0 = fopen("0.chain", "wb");
	pLastView->Reserved[0] = (short)XYScale;
	pLastView->Reserved[1] = (short)ZScale;
	printf("\nScales %d %d", pLastView->Reserved[0], pLastView->Reserved[1]);
	fwrite(pLastView, pLastView->Size(), 1, f0);	
	fclose(f0);

	FILE *f1 = fopen("1.chain", "wb");
	pThisView->Reserved[0] = (short)XYScale;
	pThisView->Reserved[1] = (short)ZScale;
	printf("\nScales %d %d", pThisView->Reserved[0], pThisView->Reserved[1]);
	fwrite(pThisView, pThisView->Size(), 1, f1);
	fclose(f1);
}