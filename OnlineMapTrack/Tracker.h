#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <stdio.h>
#include <malloc.h>
#include "map.h"

namespace SySal { 

struct IntTrack
{
	int Chains;
	int Volume;
	int X1;
	int Y1;
	int Z1;
	int X2;
	int Y2;
	int Z2;
	int Quality;
};

struct TrackMapHeader
{	
	int Count;
	int XYScale;
	int ZScale;
	int MinX;
	int MaxX;
	int MinY;
	int MaxY;
	int MinZ;
	int MaxZ;
	int TotalGrains;
	short Reserved[8];
	IntTrack Tracks[0];

	TrackMapHeader() {}

	///////////////////////////////////////////////////////
	/// Track grains are stored as a sequential stream. ///
	///////////////////////////////////////////////////////
	inline IntChain *Grains() { if (TotalGrains == 0) return 0; return (IntChain *)(void *)(Tracks + Count); }

	inline int TrackSize() { return sizeof(TrackMapHeader) + Count * sizeof(IntTrack); }

	inline int TotalSize() { return sizeof(TrackMapHeader) + Count * sizeof(IntTrack) + TotalGrains * sizeof(IntChain); }

	static TrackMapHeader *ReadFromFile(FILE *f) 
	{
		TrackMapHeader tm;
		if (fread(&tm, sizeof(tm), 1, f) != 1) throw "Can't read trackmap header.";
		TrackMapHeader *ptm = (TrackMapHeader *)malloc(sizeof(TrackMapHeader) + tm.Count * sizeof(IntTrack));
		*ptm = tm;
		if (fread((char *)(void *)ptm + sizeof(TrackMapHeader), sizeof(IntTrack), tm.Count, f) != tm.Count) 
		{
			free(ptm);
			throw "Can't read all tracks in file.";
		}
		return ptm;
	}

	static void Free(TrackMapHeader *hdr) { free(hdr); }
};

//////////////////////////////////////////////////////////////////////
// Builds a set of tracks from a chainmap.						   ///
// The object stores the tracks built, dumping them on request.    ///
//////////////////////////////////////////////////////////////////////
class Tracker
{	
public:
	//////////////////////////////////////
	/// Configuration for the Tracker. ///
	//////////////////////////////////////
	struct Configuration
	{
		//////////////////////////////////////
		/// XY Tolerance for tracking.     ///
		/// Units are given by GetXYScale. ///
		//////////////////////////////////////
		int XYTolerance;
		/////////////////////////////////////
		/// Z Tolerance for tracking.     ///
		/// Units are given by GetZScale. ///
		/////////////////////////////////////
		int ZTolerance;
		////////////////////////////////////////////
		/// Thickness of the acquisition volume. ///
		/// Units are given by GetZScale.        ///
		////////////////////////////////////////////
		int ZThickness;		
		/////////////////////////////////////
		/// XY size of bins in hash table ///
		/////////////////////////////////////
		int XYHashTableBinSize;
		////////////////////////////////////
		/// Z size of bins in hash table ///
		////////////////////////////////////
		int ZHashTableBinSize;
		/////////////////////////////////////////////////////////////////////////////
		/// Number of Z bins in hash table                                        ///
		/// Notice: wrap-around is allowed, because we don't know the total depth ///
		/////////////////////////////////////////////////////////////////////////////
		int ZHashTableBins;
		///////////////////////////////////////////////////////
		// Maximum number of chains within a hash table bin. //
		///////////////////////////////////////////////////////
		int HashBinCapacity;
		////////////////////////////////////////////////
		/// Minimum length of the track in XY units. ///
		////////////////////////////////////////////////
		int MinLength;
		////////////////////////////////////////////////
		/// Maximum length of the track in XY units. ///
		////////////////////////////////////////////////
		int MaxLength;
		/////////////////////////////////
		/// Maximum number of tracks. ///
		/////////////////////////////////
		int MaxTracks;
		///////////////////////////////////////////////
		/// Minimum total volume to accept a track. ///
		///////////////////////////////////////////////
		int MinVolume;
		/////////////////////////////////////////////////////////////
		/// Chains below this volume are not used to form tracks. ///
		/////////////////////////////////////////////////////////////
		int MinChainVolume;
		//////////////////////////////////////////////////////////////////////////////
		/// Tracks must stay within a slope acceptance window in the X projection. ///
		/// Units are given by GetSlopeScale.                                      ///
		//////////////////////////////////////////////////////////////////////////////
		int SlopeAcceptanceX;
		//////////////////////////////////////////////////////////////////////////////
		/// Tracks must stay within a slope acceptance window in the Y projection. ///
		/// Units are given by GetSlopeScale.                                      ///
		//////////////////////////////////////////////////////////////////////////////
		int SlopeAcceptanceY;
		////////////////////////////////////////////////
		/// X Center of the slope acceptance window. ///
		/// Units are given by GetSlopeScale.        ///
		////////////////////////////////////////////////
		int SlopeCenterX;
		////////////////////////////////////////////////
		/// Y Center of the slope acceptance window. ///
		/// Units are given by GetSlopeScale.        ///
		////////////////////////////////////////////////		
		int SlopeCenterY;
		////////////////////////////////////////////////////////////////////////
		/// A track is required to have a minimum length-volume correlation. ///
		/// Volume > FVL0 + Length FVLS                                      ///
		/// This parameter is FVL0.                                          ///
		////////////////////////////////////////////////////////////////////////
		int FilterVolumeLength0;
		/////////////////////////////////////////////////////////////////////////
		/// A track is required to have a minimum length-volume correlation.  ///
		/// Volume > FVL0 + Length FVLS                                       ///
		/// This parameter defines the volume cut for a length of 100 micron. ///
		/// FVLS is worked out accordingly.                                   ///
		/////////////////////////////////////////////////////////////////////////
		int FilterVolumeLength100;	
		///////////////////////////////////////////////////////////////////////
		/// A track is required to have more chains as its slope increases. ///
		/// Chains >= CH0 + Slope FCM                                       ///
		/// This parameter is CH0                                           ///
		///////////////////////////////////////////////////////////////////////
		int FilterChain0;
		///////////////////////////////////////////////////////////////////////
		/// A track is required to have more chains as its slope increases. ///
		/// Chains >= CH0 + Slope FCM                                       ///
		/// This parameter is FCM                                           ///
		///////////////////////////////////////////////////////////////////////
		float FilterChainMult;
		/////////////////////////////////////////////////////////////////
		/// Background tracks are built with few "oversize" clusters. ///
		/// It is required that    Clusters * C_M >= C_0 - Volume     ///
		/// This parameter is C_0                                     ///
		/////////////////////////////////////////////////////////////////
		int ClusterVol0;
		/////////////////////////////////////////////////////////////////
		/// Background tracks are built with few "oversize" clusters. ///
		/// It is required that    Clusters * C_M >= C_0 - Volume     ///
		/// This parameter is C_M                                     ///
		/////////////////////////////////////////////////////////////////
		int ClusterVolM;
		////////////////////////////////////////////////////////////////////////////////////
		/// Track merging checks two tracks if the have one extent within this distance. ///
		/// Only the first extent is checked, so this distance should not be too small.  ///
		/// Units are given by GetXYScale.                                               ///
		////////////////////////////////////////////////////////////////////////////////////
		int MergeTrackCell;
		//////////////////////////////////////////////////////////////////////////////////////
		/// Two tracks are merged if their extents are aligned better than this tolerance. ///
		/// Units are given by GetXYScale.                                                 ///
		//////////////////////////////////////////////////////////////////////////////////////
		int MergeTrackXYTolerance;
		//////////////////////////////////////////////////////////////////////////////////////
		/// Two tracks are merged if their extents are aligned better than this tolerance. ///
		/// Units are given by GetZScale.                                                  ///
		//////////////////////////////////////////////////////////////////////////////////////
		int MergeTrackZTolerance;
	};

	//////////////////////////
	/// Creates a Tracker. ///
	//////////////////////////
	static Tracker *CreateTracker(int deviceid);

	/////////////////////////////////////////////////////////
	/// Gets a copy for the currently used configuration. ///
	/////////////////////////////////////////////////////////
	virtual Configuration GetConfiguration() = 0;

	//////////////////////////
	/// Resets the content ///
	//////////////////////////
	virtual void Reset(Configuration &c) = 0;
	
	///////////////////////////////////
	/// Finds tracks in a Chain map ///
	///////////////////////////////////
	virtual int FindTracks(ChainMapHeader &cm) = 0;
	
	//////////////////////////////////////////////////////////////////////////
	/// Finds tracks in a Chain map                                        ///
	/// The OpaqueChainMap must be on the same device used by the tracker. ///
	//////////////////////////////////////////////////////////////////////////
	virtual int FindTracksInDevice(OpaqueChainMap &ocm) = 0;
	
	/////////////////////////////////////////////////////////////
	/// Returns the total number of grains currently obtained ///
	/////////////////////////////////////////////////////////////	
	virtual int TotalTracks() = 0;

	/////////////////////////////////////////////////////
	/// Returns a Track map.					      ///
	/// The memory may be reused by the object.		  ///
	/// Calls to methods may alter this memory space. ///
	/////////////////////////////////////////////////////
	virtual TrackMapHeader *Dump() = 0;

	/////////////////////////////////////////////////////////////
	/// Sets the name for the log files.                      ///
	/// A separate log is generated for each processing step. ///
	/////////////////////////////////////////////////////////////
	virtual void SetLogFileName(char *logfile) = 0;

	///////////////////////////////////////////////
	/// Retrieves the XY scale used for tracks. ///
	///////////////////////////////////////////////
	virtual int GetXYScale() = 0;

	//////////////////////////////////////////////
	/// Retrieves the Z scale used for tracks. ///
	//////////////////////////////////////////////
	virtual int GetZScale() = 0;

	//////////////////////////////////////////////////
	/// Retrieves the scale used for track slopes. ///
	//////////////////////////////////////////////////
	virtual int GetSlopeScale() = 0;

	///////////////////////////////////////////////////////////////////////////
	/// Sets additional options.                                            ///
	/// These settings do not affect tracking efficiency, but may be useful ///
	/// to produce logs and computing performance estimators.               ///
	/// The set of options depends on the implementation.                   ///
	///////////////////////////////////////////////////////////////////////////
	virtual void SetOption(const char *option, const char *value) = 0;	
};



}

#endif