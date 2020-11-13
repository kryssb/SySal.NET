#ifndef _MAP_H_
#define _MAP_H_

#include <stdio.h>
#include <malloc.h>

namespace SySal { 

struct IntCluster
{
	int Area;
	short X;
	short Y;
	int IXX;
	int IYY;
	int IXY;
};

struct IntMapCluster
{
	int idoriginal;
	int icell;
	int icellx, icelly;
	int ibasex, ibasey;
	int ishiftedx, ishiftedy;	
	int ipairblockstart, ipairblockcount;
};

struct IntChain
{
	int Clusters;	
	int Volume;
	int AvgX;
	int AvgY;
	int AvgZ;
/*
	int DeltaX;
	int DeltaY;
	int DeltaZ;
	int TopZ;
	int BottomZ;
*/	
	int ViewTag;
	int Reserved;
};


struct IntMapChain
{
	int idoriginal;
	int icell;
	int ibasex, ibasey;
	int ishiftedx, ishiftedy;	
	int ibasez;
	int ishiftedz;
	IntChain **pMapCellContent;
};

struct IntPair
{
	int Index1;
	int Index2;
};

struct ClusterDataHeader
{
	int Images;
	int Scale;
	short Width;
	short Height;
	short Reserved[8];
};

struct ChainView
{	
	int Count;
	int PositionX;
	int PositionY;
	int PositionZ;
	int DeltaX;
	int DeltaY;
	int DeltaZ;
	short Reserved[8];
	IntChain Chains[0];

	inline int Size() { return sizeof(ChainView) + Count * sizeof(IntChain); }

	inline ChainView *Next() { return (ChainView *)(void *)((char *)(void *)this + Size()); }

};

struct ChainMapHeader
{
	int Views;
	int XYScale;
	int ZScale;
	int Width;
	int Height;
	short Reserved[32];

	inline int TotalSize() { int i; int size = sizeof(ChainMapHeader); ChainView *pv = (ChainView *)(void *)((char *)(void *)this + sizeof(ChainMapHeader)); for (i = 0; i < Views; i++) { size += pv->Size(); pv = pv->Next(); } return size; }

	inline int TotalChains() { int i; int ch = 0; ChainView *pv = (ChainView *)((char *)(void *)this + sizeof(ChainMapHeader)); for (i = 0; i < Views; i++) { ch += pv->Count; pv = pv->Next(); } return ch; }	

	inline ChainView *FirstView() { return (ChainView *)(void *)((char *)(void *)this + sizeof(ChainMapHeader)); }

	ChainMapHeader() {}

	static ChainMapHeader *ReadFromFile(FILE *f) 
	{
		ChainMapHeader ch;
		if (fread(&ch, sizeof(ch), 1, f) != 1) throw "Can't read chainmap header.";
		ChainMapHeader *pch = (ChainMapHeader *)malloc(sizeof(ChainMapHeader));
		*pch = ch;
		int v;
		int pos = sizeof(ChainMapHeader);
		for (v = 0; v < ch.Views; v++)
		{
			ChainView chv;
			if (fread(&chv, sizeof(ChainView), 1, f) != 1) throw "Can't read chainview header.";
			pch = (ChainMapHeader *)realloc(pch, pos + sizeof(ChainView) + chv.Count * sizeof(IntChain));
			*(ChainView *)(void *)((char *)(void *)pch + pos) = chv;
			if (fread((char *)(void *)pch + pos + sizeof(ChainView), chv.Count * sizeof(IntChain), 1, f) != 1) throw "Can't read chains";
			pos += sizeof(ChainView) + chv.Count * sizeof(IntChain);
		}
		return pch;
	}

	static void Free(ChainMapHeader *hdr) { free(hdr); }
};

struct OpaqueChainMap
{
protected:
	int DeviceId;
	ChainMapHeader *pData;
	inline OpaqueChainMap(int gpuid, ChainMapHeader *pdata) : pData(pdata), DeviceId(gpuid) {}
	inline OpaqueChainMap(OpaqueChainMap &X) : pData(X.pData), DeviceId(X.DeviceId) {}
};

struct ImageCorrection
{
	float XSlant;
	float YSlant;
	float DMagDX;
	float DMagDY;
	float DMagDZ;		
	float XYCurvature;
	float ZCurvature;	
	float CameraRotation;
};

struct IntClusterFile
{
	void *pData;
	int *pImages;
	int *pScale;
	short *pWidth;
	short *pHeight;
	double *pPixMicronX;
	double *pPixMicronY;
	double *pStagePos;
	int *pImageClusterCounts;
	IntCluster *pClusters;

	inline double StageX(int img) { return pStagePos[3 * img]; }
	inline double StageY(int img) { return pStagePos[3 * img + 1]; }
	inline double StageZ(int img) { return pStagePos[3 * img + 2]; }

	inline int Images() { return *pImages; }

	inline int Scale() { return *pScale; }

	inline short Width() { return *pWidth; }

	inline short Height() { return *pHeight; }

	inline double PixMicronX() { return *pPixMicronX; }
	
	inline double PixMicronY() { return *pPixMicronY; }

	int TotalSize;

	inline IntCluster *pImageClusters(int img)
	{
		int i;
		IntCluster *pStart = pClusters;
		for (i = 0; i < img; i++)
			pStart += pImageClusterCounts[i];
		return pStart;
	}

	inline int ImageClusterCounts(int img) { return pImageClusterCounts[img]; }

	IntClusterFile(const char *filename, bool loadclusters = true);
	IntClusterFile();
	virtual ~IntClusterFile();	

	void CopyEmpty(IntClusterFile &);

	void Dump();
};

//////////////////////////////////////////////////////////////////////////
// Builds a set of cluster chains (grains) from sequences of clusters. ///
// It works in incremental way by adding sequences one by one.         ///
// The object stores the chain maps built, dumping them on request.    ///
//////////////////////////////////////////////////////////////////////////
class ClusterChainer
{	
public:

	struct EmulsionEdge
	{
		bool Valid;
		float Z;
	};

	struct EmulsionFocusInfo
	{
		bool Valid;
		EmulsionEdge Top;
		EmulsionEdge Bottom;
		double ZOffset;
	};

	/////////////////////////////////////////////
	/// Configuration for the ClusterChainer. ///
	/////////////////////////////////////////////
	struct Configuration
	{
		////////////////////////////////////////////
		/// Maximum number of objects in a cell. ///
		////////////////////////////////////////////		
		int MaxCellContent;
		///////////////////////////////////////////////////////
		/// Size of each cell (units of the cluster scale). ///
		///////////////////////////////////////////////////////
		union
		{
			int CellSize;
			float fCellSize;
		};
		/////////////////////////////////////
		/// Tolerance for coarse mapping. ///
		/////////////////////////////////////
		union
		{
			int ClusterMapCoarseTolerance;
			float fClusterMapCoarseTolerance;
		};
		/////////////////////////////////////
		/// Tolerance for fine mapping.   ///
		/////////////////////////////////////
		union
		{
			int ClusterMapFineTolerance;
			float fClusterMapFineTolerance;
		};
		/////////////////////////////////////
		/// Acceptance for fine mapping.  ///
		/////////////////////////////////////
		union
		{
			int ClusterMapFineAcceptance;
			float fClusterMapFineAcceptance;
		};
		/////////////////////////////////////////////
		/// Maximum X offset for cluster mapping. ///
		/////////////////////////////////////////////
		union
		{
			int ClusterMapMaxXOffset;
			float fClusterMapMaxXOffset;
		};
		/////////////////////////////////////////////
		/// Maximum Y offset for cluster mapping. ///
		/////////////////////////////////////////////
		union
		{
			int ClusterMapMaxYOffset;
			float fClusterMapMaxYOffset;
		};
		/////////////////////////////////////////////////////////////////
		/// Minimum size of a cluster to use it for pattern matching. ///
		/////////////////////////////////////////////////////////////////
		int ClusterMapMinSize;
		///////////////////////////////////////////////////////////////////////////////////////////////
		/// Specifies that only 1 out of ClusterMapSampleDivider clusters must be used for mapping. ///
		///////////////////////////////////////////////////////////////////////////////////////////////
		int ClusterMapSampleDivider;
		/////////////////////////////////////////////////////////////////////////
		/// Minimum number of clusters mapping to consider the mapping valid. ///
		/// No translation is applied if this number is not reached.          ///
		/////////////////////////////////////////////////////////////////////////
		int MinClusterMapsValid;
		//////////////////////////////////////////////
		/// Minimum number of clusters in a chain. ///
		//////////////////////////////////////////////
		int MinClustersPerChain;
		//////////////////////////////////
		/// Minimum volume in a chain. ///
		//////////////////////////////////
		int MinVolumePerChain;
		//////////////////////////////////////////////
		/// XY coarse tolerance for chain mapping. ///
		//////////////////////////////////////////////
		union
		{
			int ChainMapXYCoarseTolerance;
			float fChainMapXYCoarseTolerance;
		};
		////////////////////////////////////////////
		/// XY fine tolerance for chain mapping. ///
		////////////////////////////////////////////
		union
		{
			int ChainMapXYFineTolerance;
			float fChainMapXYFineTolerance;
		};
		/////////////////////////////////////////////
		/// XY fine acceptance for chain mapping. ///
		/////////////////////////////////////////////
		union
		{
			int ChainMapXYFineAcceptance;
			float fChainMapXYFineAcceptance;
		};
		/////////////////////////////////////////////
		/// Z coarse tolerance for chain mapping. ///
		/////////////////////////////////////////////
		union
		{
			int ChainMapZCoarseTolerance;
			float fChainMapZCoarseTolerance;
		};
		///////////////////////////////////////////
		/// Z fine tolerance for chain mapping. ///
		///////////////////////////////////////////
		union
		{
			int ChainMapZFineTolerance;
			float fChainMapZFineTolerance;
		};
		////////////////////////////////////////////
		/// Z fine acceptance for chain mapping. ///
		////////////////////////////////////////////
		union
		{
			int ChainMapZFineAcceptance;
			float fChainMapZFineAcceptance;
		};
		///////////////////////////////////////
		/// Max X offset for chain mapping. ///
		///////////////////////////////////////
		union
		{
			int ChainMapMaxXOffset;
			float fChainMapMaxXOffset;
		};
		///////////////////////////////////////
		/// Max Y offset for chain mapping. ///
		///////////////////////////////////////
		union
		{
			int ChainMapMaxYOffset;
			float fChainMapMaxYOffset;
		};
		///////////////////////////////////////
		/// Max Z offset for chain mapping. ///
		///////////////////////////////////////
		union
		{
			int ChainMapMaxZOffset;
			float fChainMapMaxZOffset;
		};
		//////////////////////////////////////////////////////////////////////
		/// Minimum volume required to use the chain for pattern matching. ///
		//////////////////////////////////////////////////////////////////////
		int ChainMapMinVolume;
		///////////////////////////////////////////////////////////////////////////////////////////
		/// Specifies that only 1 out of ChainMapSampleDivider chains must be used for mapping. ///
		///////////////////////////////////////////////////////////////////////////////////////////
		int ChainMapSampleDivider;
		///////////////////////////////////////////////////////////////////////
		/// Minimum number of chains mapping to consider the mapping valid. ///
		/// No translation is applied if this number is not reached.        ///
		///////////////////////////////////////////////////////////////////////
		int MinChainMapsValid;
		//////////////////////////////////////////
		/// Maximum number of chains to store. ///
		//////////////////////////////////////////
		int MaxChains;
		///////////////////////////////////////////////////////
		/// Minimum number of clusters inside the emulsion. ///
		///////////////////////////////////////////////////////
		int ClusterThreshold;
	};

	/////////////////////////////////
	/// Creates a ClusterChainer. ///
	/////////////////////////////////
	static ClusterChainer *CreateClusterChainer(int deviceid);

	/////////////////////////////////////////////////////////
	/// Gets a copy for the currently used configuration. ///
	/////////////////////////////////////////////////////////
	virtual Configuration GetConfiguration() = 0;

	//////////////////////////
	/// Resets the content ///
	//////////////////////////
	virtual void Reset(Configuration &c, ImageCorrection &corr, bool istop) = 0;

	/////////////////////////////////////////////////////
	/// Looks for the first estimate of reference Zs. ///
	/// Returns true if found, false otherwise.       ///
	/////////////////////////////////////////////////////
	virtual bool SetReferenceZs(IntClusterFile &cf, bool istop) = 0;
	
	//////////////////////////////////////////////////////////////////
	/// Adds a set of clusters to the grains of the reconstruction ///
	//////////////////////////////////////////////////////////////////
	virtual EmulsionFocusInfo AddClusters(IntClusterFile &cf) = 0;
	
	/////////////////////////////////////////////////////////////
	/// Returns the total number of grains currently obtained ///
	/////////////////////////////////////////////////////////////	
	virtual int TotalChains() = 0;

	/////////////////////////////////////////////////////
	/// Returns a Chain map.					      ///
	/// The memory may be reused by the object.		  ///
	/// Calls to methods may alter this memory space. ///
	/////////////////////////////////////////////////////
	virtual ChainMapHeader *Dump() = 0;

	/////////////////////////////////////////////////////
	/// Returns a Chain map in device memory.	      ///
	/// The memory may be reused by the object.		  ///
	/// Calls to methods may alter this memory space. ///
	/////////////////////////////////////////////////////
	virtual OpaqueChainMap &GetDeviceChainMap() = 0;

	/////////////////////////////////////////////////////////////
	/// Sets the name for the log files.                      ///
	/// A separate log is generated for each processing step. ///
	/////////////////////////////////////////////////////////////
	virtual void SetLogFileName(char *logfile) = 0;

	///////////////////////////////////////////////
	/// Retrieves the XY scale used for chains. ///
	///////////////////////////////////////////////
	virtual int GetXYScale() = 0;

	//////////////////////////////////////////////
	/// Retrieves the Z scale used for chains. ///
	//////////////////////////////////////////////
	virtual int GetZScale() = 0;
};

}

#endif