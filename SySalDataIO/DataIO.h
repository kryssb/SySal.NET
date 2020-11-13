// DataIO.h : Declaration of the CSySalDataIO

#ifndef __SYSALDATAIO_H_
#define __SYSALDATAIO_H_

#include "resource.h"       // main symbols
#include "..\Common\Connection.h"
#include "..\Common\Config.h"
#include "..\Common\GUIdefs.h"
#include "..\Common\Cluster.h"
#include "..\Common\Track.h"

/////////////////////////////////////////////////////////////////////////////
// CSySalDataIO

struct ConfigData
{
	struct t_InputData
	{
		BOOL Enabled;
		BOOL SkipGrains;
		char Path[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		char Ext[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		char Desc[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		UINT Attempts;
		} InputData;
	struct t_OutputData
	{
		BOOL Enabled;
		char Path[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		char Ext[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		char Desc[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		char RescuePath[SYSAL_MAXCONFIG_ENTRY_VALUE_LEN];
		UINT Attempts;
		BOOL AppendUID;
		struct t_GrainDiffCompressionData
		{
			BOOL Enabled;
			float DepthPrecision;
			float TransversePrecision;
			} GrainDiffCompressionData;
		int AsyncWriteBufferMB;
		} OutputData;
	};

struct QueryDataInfo
{  
	char Name[SYSAL_MAXCONFIG_ENTRY_NAME_LEN];
	char Format[8];
	int Position;
	char Default[16];
	};

typedef struct
{
	int Field;
	int PointsN;				
	TVector Intercept;
	TVector Slope;
	float Sigma;
	float FirstZ;
	float LastZ;
	TVector InterceptErrors;
	TVector SlopeErrors;
	} BULK_TrackInfo;

class ATL_NO_VTABLE CSySalDataIO : 
	public CComObjectRootEx<CComMultiThreadModel>,
	public CComCoClass<CSySalDataIO, &CLSID_SySalDataIO>,
	public ISySalDataIO2,
	public ISySalObject
{
public:
	CSySalDataIO()
	{
		Name[0] = 0;
		HStatusChangeEvent = 0;
		HCurrentFile = INVALID_HANDLE_VALUE;
		OkToRun = true;
		C.InputData.Enabled = true;
		C.InputData.Attempts = 1;
		C.InputData.Path[0] = 0;
		C.InputData.Desc[0] = 0;
		C.InputData.Ext[0] = 0;
		C.InputData.SkipGrains = false;
		C.OutputData.AppendUID = false;
		C.OutputData.Attempts = 1;
		C.OutputData.Path[0] = 0;
		C.OutputData.Desc[0] = 0;
		C.OutputData.Ext[0] = 0;
		C.OutputData.RescuePath[0] = 0;
		C.OutputData.Enabled = true;		
		C.OutputData.GrainDiffCompressionData.Enabled = false;
		C.OutputData.GrainDiffCompressionData.DepthPrecision = 0.1f;
		C.OutputData.GrainDiffCompressionData.TransversePrecision = 0.01f;
	}

	virtual ~CSySalDataIO()
	{
		if (HCurrentFile != INVALID_HANDLE_VALUE) CloseHandle(HCurrentFile);
	}

DECLARE_REGISTRY_RESOURCEID(IDR_SYSALDATAIO)
DECLARE_NOT_AGGREGATABLE(CSySalDataIO)

DECLARE_PROTECT_FINAL_CONSTRUCT()

BEGIN_COM_MAP(CSySalDataIO)
	COM_INTERFACE_ENTRY(ISySalObject)
	COM_INTERFACE_ENTRY(ISySalDataIO)
	COM_INTERFACE_ENTRY(ISySalDataIO2)
END_COM_MAP()

// ISySalDataIO2
public:
	STDMETHOD(Read2)(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName);
	STDMETHOD(Write2)(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName);

// ISySalDataIO
public:
	STDMETHOD(GetStatus)(UINT *pExtErrorInfo);
	STDMETHOD(Read)(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName);
	STDMETHOD(Write)(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName);

// ISySalObject
public:	
	STDMETHOD(IsReady)(long *pWrongConn, long *pRetCode);
	STDMETHOD(AsyncResume)();
	STDMETHOD(AsyncStop)();
	STDMETHOD(AsyncPause)();
	STDMETHOD(AsyncWait)();
	STDMETHOD(AsyncCheckStatus)(long *pStatus);
	STDMETHOD(AsyncExec)(HSySalHANDLE HNotifyStatusChangeEvent);
	STDMETHOD(SyncExec)();
	STDMETHOD(GetClassName)(ConnectionName name);
	STDMETHOD(GetIcon)(int n, HSySalHICON *pHICON);
	STDMETHOD(GetConnection)(int n, SySalConnection *pConnection);
	STDMETHOD(SetConnection)(int n, SySalConnection connection);
	STDMETHOD(GetNumberOfConnections)(int *pNumber);
	STDMETHOD(EnableMonitor)(boolean bEnable);
	STDMETHOD(EditConfig)(SySalConfig *pConfig, HSySalHANDLE hWnd);
	STDMETHOD(GetConfig)(SySalConfig *pConfig);
	STDMETHOD(SetConfig)(SySalConfig *pConfig);
	STDMETHOD(GetName)(ConnectionName name);
	STDMETHOD(SetName)(ConnectionName name);

protected:
	
	class BufferedFile
	{
		protected:

			HANDLE HFile;
			BYTE *pBuffer;
			int FillLevel;
			int BufferSize;
			bool Write;

		public:

			BufferedFile(int buffersize = 65536);
			virtual ~BufferedFile();

			virtual bool Open(char *path, bool write);
			virtual bool WriteBytes(BYTE *pbytes, int length);
			virtual bool ReadBytes(BYTE *pbytes, int length, int *pbytesread);
			virtual bool Close();
			
		};

	class AsyncBufferedFile : public BufferedFile
	{
		protected:

			HANDLE HWriteThread;
			char *pPath;
			char *pAlternatePath;
			int Attempts;

			void WriteThread();
			static void WriteThreadHook(void *powner);

		public:

			AsyncBufferedFile(int attempts, int buffersize = 16 * 1048576);
			virtual ~AsyncBufferedFile();

			virtual bool Open(char *path, bool write);
			virtual bool WriteBytes(BYTE *pbytes, int length);
			virtual bool Close();
		};

	UINT Read(char *FileName, IO_Data **ppData, bool UseISySalDataIO = true);
	UINT ReadCSSSTracks(BufferedFile &F, IO_Header Hdr, IO_CS_SS_Tracks **ppData);
	UINT ReadOPERATracks(BufferedFile &F, IO_Header Hdr, IO_OPERA_Tracks **ppData);
	UINT ReadBulkPhotoSeq(BufferedFile &F, IO_Header Hdr, IO_BULK_PhotoSeq **ppData);
	UINT ReadBulkClustersDump(BufferedFile &F, IO_Header Hdr, IO_BULK_ClustersDump **ppData);
	UINT ReadBulkTracks(BufferedFile &F, IO_Header Hdr, IO_BULK_Tracks **ppData);
	UINT ReadBulkCSSSBatch(BufferedFile &F, IO_Header Hdr, IO_BS_Batch **ppData);
	UINT ReadTSReconstruction(BufferedFile &F, IO_Header Hdr, IO_TS_Reconstruction **ppData);
	UINT ReadMarkMap(BufferedFile &F, IO_Header Hdr, IO_MarkMap **ppData);
	UINT ReadVertigoScanCatalog(BufferedFile &F, IO_Header Hdr, IO_VS_Catalog **ppData);
	UINT ReadVertigoScanFragment(BufferedFile &F, IO_Header Hdr, IO_VS_Fragment **ppData);
	UINT ReadVertigoScanFragment2(BufferedFile &F, IO_Header Hdr, IO_VS_Fragment2 **ppData);
	void GetExtAndDesc(IO_Data *pHdr, char **pExt, char **pDesc);
	UINT Write(char *DestPath, char *RescuePath, char *FileName, IO_Data *pData, bool UseISySalDataIO = true);
	UINT WriteCSSSTracks(char *FileName, IO_CS_SS_Tracks *pData);
	UINT WriteOPERATracks(char *FileName, IO_OPERA_Tracks *pData);
	UINT WriteBulkClustersDump(char *FileName, IO_BULK_ClustersDump *pData);
	UINT WriteBulkPhotoSeq(char *FileName, IO_BULK_PhotoSeq *pData);
	UINT WriteBulkTracks(char *FileName, IO_BULK_Tracks *pData);
	UINT WriteBulkCSSSBatch(char *FileName, IO_BS_Batch *pData);
	UINT WriteTSReconstruction(char *FileName, IO_TS_Reconstruction *pData);
	UINT WriteMarkMap(char *FileName, IO_MarkMap *pData);
	UINT WriteVertigoScanCatalog(char *FileName, IO_VS_Catalog *pData);
	UINT WriteVertigoScanFragment(char *FileName, IO_VS_Fragment *pData);
	UINT WriteVertigoScanFragment2(char *FileName, IO_VS_Fragment2 *pData);
	ConnectionName Name;
	HANDLE HStatusChangeEvent;
	HANDLE HCurrentFile;
	bool OkToRun;
	CString AutoUID;

	ConfigData C;

	static int GetData(QueryDataInfo DataName, void *StartPosition, char *File, int N);
};

#endif //__SYSALDATAIO_H_
