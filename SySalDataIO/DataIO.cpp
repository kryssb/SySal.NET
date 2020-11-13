// DataIO.cpp : Implementation of CSySalDataIO
#include "stdafx.h"
#define USE_MIDL_INCOMPATIBLE_STRUCTS
#include "SySalDataIO.h"
#include "DataIO.h"
#include "EditConfig.h"
#include "FileDlgs.h"
#include "GrainCompressionDiff.h"
#include <stdio.h>
#include <process.h>

#define READ_VERTEXTSTRACK_ALLOC_BLOCK 64

/////////////////////////////////////////////////////////////////////////////
// CSySalDataIO

const int DataION = 17;

#define STRUCT_OFFSET(t, x) (((int)&(((t *)1)->x))-1)

QueryDataInfo DataIOName[DataION] =
{	{"InPath",			"%s",	STRUCT_OFFSET(ConfigData, InputData.Path), ""},
	{"InExt",			"%s",	STRUCT_OFFSET(ConfigData, InputData.Ext), "dat"},
	{"InDesc",			"%s",	STRUCT_OFFSET(ConfigData, InputData.Desc), "Data Files"},
	{"InAttempts",		"%d",	STRUCT_OFFSET(ConfigData, InputData.Attempts), "3"},
	{"InEnable",		"%d",	STRUCT_OFFSET(ConfigData, InputData.Enabled), "1"},
	{"InSkipGrains",	"%d",	STRUCT_OFFSET(ConfigData, InputData.SkipGrains), "0"},
	{"OutPath",			"%s",	STRUCT_OFFSET(ConfigData, OutputData.Path), ""},
	{"OutRescuePath",	"%s",	STRUCT_OFFSET(ConfigData, OutputData.RescuePath), ""},
	{"OutExt",			"%s",	STRUCT_OFFSET(ConfigData, OutputData.Ext), "dat"},
	{"OutDesc",			"%s",	STRUCT_OFFSET(ConfigData, OutputData.Desc), "Data Files"},
	{"OutAttempts",		"%d",	STRUCT_OFFSET(ConfigData, OutputData.Attempts), "3"},
	{"AsyncWriteBuffer","%d",	STRUCT_OFFSET(ConfigData, OutputData.AsyncWriteBufferMB), "0"},
	{"OutAppendUID",	"%d",	STRUCT_OFFSET(ConfigData, OutputData.AppendUID), "1"},
	{"OutEnable",		"%d",	STRUCT_OFFSET(ConfigData, OutputData.Enabled), "1"}, 
	{"GrainCompressEnable",	"%d",	STRUCT_OFFSET(ConfigData, OutputData.GrainDiffCompressionData.Enabled), "0"}, 
	{"TransversePrecision",	"%f",	STRUCT_OFFSET(ConfigData, OutputData.GrainDiffCompressionData.TransversePrecision), "0.01"}, 
	{"DepthPrecision",	"%f",	STRUCT_OFFSET(ConfigData, OutputData.GrainDiffCompressionData.DepthPrecision), "0.1"}
	};

int CSySalDataIO::GetData(QueryDataInfo DataName, void *StartPosition, char *File, int N)
{  
	int Index;
	for (Index = 0; (Index < N) && (strncmp(File + (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) * Index, 
		DataName.Name, SYSAL_MAXCONFIG_ENTRY_NAME_LEN)); Index++);
	if (Index >= N)
	{
   		if (strcmp(DataName.Format, "%s")) sscanf(DataName.Default, DataName.Format, (char *)StartPosition + DataName.Position);
		else strcpy((char *)StartPosition + DataName.Position, DataName.Default);
   		return 0;
		};
	if (strcmp(DataName.Format, "%s")) return sscanf(File + (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) * Index + 
		SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataName.Format, (char *)StartPosition + DataName.Position);
	strcpy((char *)StartPosition + DataName.Position, File + (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) * Index + 
		SYSAL_MAXCONFIG_ENTRY_NAME_LEN);
	return 1;
	};

// ISySalObject
STDMETHODIMP CSySalDataIO::SetName(ConnectionName name)
{
	// TODO: Add your implementation code here
	strncpy(Name, name, SYSAL_MAX_CONN_NAME_LEN);
	return S_OK;
}

STDMETHODIMP CSySalDataIO::GetName(ConnectionName name)
{
	// TODO: Add your implementation code here
	strncpy(name, Name, SYSAL_MAX_CONN_NAME_LEN);
	return S_OK;
}

STDMETHODIMP CSySalDataIO::SetConfig(SySalConfig *pConfig)
{
	// TODO: Add your implementation code here
	OkToRun = false;
	int i;
	for (i = 0; (i < DataION) && (GetData(DataIOName[i], &C, pConfig->pItems, pConfig->CountOfItems)); i++);
	if (i < DataION) return S_FALSE;
	OkToRun = true;
	if (C.OutputData.AppendUID)
	{
		char compname[MAX_COMPUTERNAME_LENGTH + 1];
		DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
		GetComputerName(compname, &size);
		SYSTEMTIME time;
		GetLocalTime(&time);
		AutoUID.Format("_%s_%02X%X%02d%02d%02d%02d%03d_", compname, time.wYear - 1846, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
		}
	else AutoUID = "";
	return S_OK;
}

STDMETHODIMP CSySalDataIO::GetConfig(SySalConfig *pConfig)
{
	// TODO: Add your implementation code here
	if (pConfig->pItems) CoTaskMemFree(pConfig->pItems);
	pConfig->pItems = 0;
	pConfig->CountOfItems = 0;	
	if (!(pConfig->pItems = (char *)CoTaskMemAlloc(DataION * 
		(SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN))))
			return S_FALSE;
	pConfig->CountOfItems = DataION;
	int i;
	for (i = 0; i < DataION; i++)
	{
		strncpy(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN), 
			DataIOName[i].Name, SYSAL_MAXCONFIG_ENTRY_NAME_LEN);
		if (!strcmp(DataIOName[i].Format, "%d"))
			sprintf(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) + 
				SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataIOName[i].Format, *(int *)((char *)&C + DataIOName[i].Position));
		else if (!strcmp(DataIOName[i].Format, "%f"))
			sprintf(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) + 
				SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataIOName[i].Format, *(float *)((char *)&C + DataIOName[i].Position));
		else
			sprintf(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) + 
				SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataIOName[i].Format, ((char *)&C + DataIOName[i].Position));
		};
	return S_OK;	
}

STDMETHODIMP CSySalDataIO::EditConfig(SySalConfig *pConfig, HSySalHANDLE hWnd)
{
	// TODO: Add your implementation code here
	AFX_MANAGE_STATE(AfxGetStaticModuleState())
	ConfigData Temp;
	int i;
	for (i = 0; i < DataION; i++) 
		GetData(DataIOName[i], &Temp, pConfig->pItems, pConfig->CountOfItems);
	CWnd Owner;
	Owner.Attach((HWND)hWnd);
	CEditConfig MyDlg(&Owner);
	MyDlg.m_AutoAppendUID = Temp.OutputData.AppendUID;
	MyDlg.m_DataDest = Temp.OutputData.Path;
	MyDlg.m_RescueDest = Temp.OutputData.RescuePath;
	MyDlg.m_DataSource = Temp.InputData.Path;
	MyDlg.m_InputAttempts = Temp.InputData.Attempts;
	MyDlg.m_OutputAttempts = Temp.OutputData.Attempts;
	MyDlg.m_InputDefDesc = Temp.InputData.Desc;
	MyDlg.m_OutputDefDesc = Temp.OutputData.Desc;
	MyDlg.m_InputDefExt = Temp.InputData.Ext;
	MyDlg.m_OutputDefExt = Temp.OutputData.Ext;
	MyDlg.m_InputEnable = Temp.InputData.Enabled;
	MyDlg.m_OutputEnable = Temp.OutputData.Enabled;
	MyDlg.m_SkipGrains = Temp.InputData.SkipGrains;
	MyDlg.m_GrainCompressEnable = Temp.OutputData.GrainDiffCompressionData.Enabled;
	MyDlg.m_TransversePrecision = Temp.OutputData.GrainDiffCompressionData.TransversePrecision;
	MyDlg.m_DepthPrecision = Temp.OutputData.GrainDiffCompressionData.DepthPrecision;
	MyDlg.m_AsyncWriteBufferMB = Temp.OutputData.AsyncWriteBufferMB;
	if (MyDlg.DoModal() == IDOK)
	{
		Temp.OutputData.AsyncWriteBufferMB = MyDlg.m_AsyncWriteBufferMB;
		Temp.OutputData.GrainDiffCompressionData.Enabled = MyDlg.m_GrainCompressEnable;
		Temp.OutputData.GrainDiffCompressionData.TransversePrecision = MyDlg.m_TransversePrecision;
		Temp.OutputData.GrainDiffCompressionData.DepthPrecision = MyDlg.m_DepthPrecision;
		Temp.OutputData.AppendUID = MyDlg.m_AutoAppendUID;
		strcpy(Temp.OutputData.Path, MyDlg.m_DataDest);
		strcpy(Temp.OutputData.RescuePath, MyDlg.m_RescueDest);
		strcpy(Temp.InputData.Path, MyDlg.m_DataSource);
		Temp.InputData.Attempts = MyDlg.m_InputAttempts;
		Temp.OutputData.Attempts = MyDlg.m_OutputAttempts;
		strcpy(Temp.InputData.Desc, MyDlg.m_InputDefDesc);
		strcpy(Temp.OutputData.Desc, MyDlg.m_OutputDefDesc);
		strcpy(Temp.InputData.Ext, MyDlg.m_InputDefExt);
		strcpy(Temp.OutputData.Ext, MyDlg.m_OutputDefExt);
		Temp.InputData.Enabled = MyDlg.m_InputEnable;
		Temp.OutputData.Enabled = MyDlg.m_OutputEnable;
		Temp.InputData.SkipGrains = MyDlg.m_SkipGrains;
		if (pConfig->pItems) CoTaskMemFree(pConfig->pItems);
		pConfig->pItems = 0;
		pConfig->CountOfItems = 0;	
		if (!(pConfig->pItems = (char *)CoTaskMemAlloc(DataION * 
			(SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN))))
				return S_FALSE;
		pConfig->CountOfItems = DataION;
		int i;
		for (i = 0; i < DataION; i++)
		{
			strncpy(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN), 
				DataIOName[i].Name, SYSAL_MAXCONFIG_ENTRY_NAME_LEN);
			if (!strcmp(DataIOName[i].Format, "%d"))
				sprintf(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) + 
					SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataIOName[i].Format, *(int *)((char *)&Temp + DataIOName[i].Position));
			else if (!strcmp(DataIOName[i].Format, "%f"))
				sprintf(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) + 
					SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataIOName[i].Format, *(float *)((char *)&Temp + DataIOName[i].Position));
			else
				sprintf(pConfig->pItems + i * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) + 
					SYSAL_MAXCONFIG_ENTRY_NAME_LEN, DataIOName[i].Format, ((char *)&Temp + DataIOName[i].Position));
			};
		Owner.Detach();
		return S_OK;
		};
	Owner.Detach();
	return S_FALSE;
}

STDMETHODIMP CSySalDataIO::EnableMonitor(boolean bEnable)
{
	// TODO: Add your implementation code here

	return (bEnable) ? E_NOTIMPL : S_OK;
}

STDMETHODIMP CSySalDataIO::GetNumberOfConnections(int *pNumber)
{
	// TODO: Add your implementation code here

	*pNumber = 0;
	return S_OK;
}

STDMETHODIMP CSySalDataIO::SetConnection(int n, SySalConnection connection)
{
	// TODO: Add your implementation code here

	return E_NOTIMPL;
}

STDMETHODIMP CSySalDataIO::GetConnection(int n, SySalConnection *pConnection)
{
	// TODO: Add your implementation code here

	return E_NOTIMPL;
}

STDMETHODIMP CSySalDataIO::GetIcon(int n, HSySalHICON *pHICON)
{
	// TODO: Add your implementation code here
	AFX_MANAGE_STATE(AfxGetStaticModuleState())

	HANDLE HTemp = 0;
	switch (n)
	{
		case 0:		HTemp = LoadImage(AfxGetResourceHandle(), MAKEINTRESOURCE(IDI_ICON32), IMAGE_ICON, 32, 32, LR_SHARED);
					break;

		case 1:		HTemp = LoadImage(AfxGetResourceHandle(), MAKEINTRESOURCE(IDI_ICON64), IMAGE_ICON, 64, 64, LR_SHARED);
					break;

		default:	return E_INVALIDARG;
		};
	*pHICON = (HSySalHICON)HTemp;
	return S_OK;	
}

STDMETHODIMP CSySalDataIO::GetClassName(ConnectionName name)
{
	// TODO: Add your implementation code here
	
	strncpy(name, "Data I/O Device", SYSAL_MAX_CONN_NAME_LEN);
	return S_OK;
}

STDMETHODIMP CSySalDataIO::SyncExec()
{
	// TODO: Add your implementation code here

	return E_NOTIMPL;
}

STDMETHODIMP CSySalDataIO::AsyncExec(HSySalHANDLE HNotifyStatusChangeEvent)
{
	// TODO: Add your implementation code here

	HStatusChangeEvent = (HANDLE)HNotifyStatusChangeEvent;
	if (HStatusChangeEvent) SetEvent(HStatusChangeEvent);
	return E_NOTIMPL;
}

STDMETHODIMP CSySalDataIO::AsyncCheckStatus(long *pStatus)
{
	// TODO: Add your implementation code here
	
	*pStatus = SYSAL_ASYNC_STATUS_IDLE;
	return S_OK;
}

STDMETHODIMP CSySalDataIO::AsyncWait()
{
	// TODO: Add your implementation code here

	return S_OK;
}

STDMETHODIMP CSySalDataIO::AsyncPause()
{
	// TODO: Add your implementation code here

	if (HStatusChangeEvent) SetEvent(HStatusChangeEvent);
	return S_OK;
}

STDMETHODIMP CSySalDataIO::AsyncStop()
{
	// TODO: Add your implementation code here

	if (HStatusChangeEvent) 
	{
		SetEvent(HStatusChangeEvent);
		HStatusChangeEvent = 0;
		};
	return S_OK;
}

STDMETHODIMP CSySalDataIO::AsyncResume()
{
	// TODO: Add your implementation code here

	if (HStatusChangeEvent) SetEvent(HStatusChangeEvent);
	return S_OK;
}

STDMETHODIMP CSySalDataIO::IsReady(long *pWrongConn, long *pRetCode)
{
	// TODO: Add your implementation code here

	*pWrongConn = -1;
	*pRetCode = SYSAL_READYSTATUS_READY;
	return S_OK;
}

STDMETHODIMP CSySalDataIO::Write(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState())

	// TODO: Add your implementation code here
	if (!OkToRun)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_CANTRUN;
		return S_FALSE;
		};
	if (!C.OutputData.Enabled)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_DISABLED;
		return S_FALSE;
		};
	IO_Data *pMyData = (IO_Data *)(void *)(pData);	
	if ((!FileName || FileName[0] == 0) && (C.OutputData.Path[0] == 0))
	{
		CString TempPath;
		char *Ext, *Desc;
		GetExtAndDesc(pMyData, &Ext, &Desc);
		CWnd Owner;
		Owner.Attach((HWND)hWnd);
		if (!SelectOutputFile(TempPath, "Select Output Path", Ext, Desc, &Owner))
		{
			*pExtErrorInfo = IODATA_EXT_ERROR_ABORTED;
			Owner.Detach();
			return S_FALSE;
			};
		Owner.Detach();
		*pExtErrorInfo = Write(TempPath.GetBuffer(1), C.OutputData.RescuePath, "", pMyData);
		TempPath.ReleaseBuffer();
		}
	else
	{
		if (!FileName || FileName[0] == 0)
		{
			char FileName[512];
			char *Ext, *Desc;
			GetExtAndDesc(pMyData, &Ext, &Desc);
			sprintf(FileName, "%s%08X%08X%08X%08X.%s", AutoUID.GetBuffer(1), pMyData->HdrId.Id.Part[0], pMyData->HdrId.Id.Part[1], 
				pMyData->HdrId.Id.Part[2], pMyData->HdrId.Id.Part[3], Ext);
			AutoUID.ReleaseBuffer();
			*pExtErrorInfo = Write(C.OutputData.Path, C.OutputData.RescuePath, FileName, pMyData);
			}
		else *pExtErrorInfo = Write("", "", (char *)FileName, pMyData);
		};
	return (*pExtErrorInfo == IODATA_EXT_ERROR_OK) ? S_OK : S_FALSE;
}

STDMETHODIMP CSySalDataIO::Write2(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState())

	// TODO: Add your implementation code here
	if (!OkToRun)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_CANTRUN;
		return S_FALSE;
		};
	if (!C.OutputData.Enabled)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_DISABLED;
		return S_FALSE;
		};
	IO_Data *pMyData = (IO_Data *)(void *)(pData);	
	if ((!FileName || FileName[0] == 0) && (C.OutputData.Path[0] == 0))
	{
		CString TempPath;
		char *Ext, *Desc;
		GetExtAndDesc(pMyData, &Ext, &Desc);
		CWnd Owner;
		Owner.Attach((HWND)hWnd);
		if (!SelectOutputFile(TempPath, "Select Output Path", Ext, Desc, &Owner))
		{
			*pExtErrorInfo = IODATA_EXT_ERROR_ABORTED;
			Owner.Detach();
			return S_FALSE;
			};
		Owner.Detach();
		*pExtErrorInfo = Write(TempPath.GetBuffer(1), C.OutputData.RescuePath, "", pMyData, false);
		TempPath.ReleaseBuffer();
		}
	else
	{
		if (!FileName || FileName[0] == 0)
		{
			char FileName[512];
			char *Ext, *Desc;
			GetExtAndDesc(pMyData, &Ext, &Desc);
			sprintf(FileName, "%s%08X%08X%08X%08X.%s", AutoUID.GetBuffer(1), pMyData->HdrId.Id.Part[0], pMyData->HdrId.Id.Part[1], 
				pMyData->HdrId.Id.Part[2], pMyData->HdrId.Id.Part[3], Ext);
			AutoUID.ReleaseBuffer();
			*pExtErrorInfo = Write(C.OutputData.Path, C.OutputData.RescuePath, FileName, pMyData, false);
			}
		else *pExtErrorInfo = Write("", "", (char *)FileName, pMyData, false);
		};
	return (*pExtErrorInfo == IODATA_EXT_ERROR_OK) ? S_OK : S_FALSE;
}

STDMETHODIMP CSySalDataIO::Read(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState())

	// TODO: Add your implementation code here
	if (!OkToRun)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_CANTRUN;
		return S_FALSE;
		};
	if (!C.InputData.Enabled)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_DISABLED;
		return S_FALSE;
		};
	IO_Data **pMyData = (IO_Data **)(void *)(pData);	
	int i;
	if ((!FileName || FileName[0] == 0) && (C.InputData.Path[0] == 0))
	{
		CString TempPath;
		CWnd Owner;
		Owner.Attach((HWND)hWnd);
		if (!SelectInputFile(TempPath, "Select Input File", C.InputData.Ext[0] ? C.InputData.Ext : "", 
			C.InputData.Desc[0] ? C.InputData.Desc : "Track data files|*.tlg|Photo Sequence files|*.phs|Cluster Dump files|*.cls|Data files|*.dat|All files|*.*||", &Owner))
		{
			*pExtErrorInfo = IODATA_EXT_ERROR_ABORTED;
			Owner.Detach();
			return S_FALSE;
			};
		Owner.Detach();
		for (i = 0; i < C.InputData.Attempts && ((*pExtErrorInfo = Read(TempPath.GetBuffer(1), pMyData)) != IODATA_EXT_ERROR_OK); i++)
			TempPath.ReleaseBuffer();
		}
	else
	{
		if (!FileName || FileName[0] == 0)
			for (i = 0; i < C.InputData.Attempts && ((*pExtErrorInfo = Read(C.InputData.Path, pMyData)) != IODATA_EXT_ERROR_OK); i++);
		else
			for (i = 0; i < C.InputData.Attempts && ((*pExtErrorInfo = Read((char *)FileName, pMyData)) != IODATA_EXT_ERROR_OK); i++);
		};
	return (*pExtErrorInfo == IODATA_EXT_ERROR_OK) ? S_OK : S_FALSE;
}

STDMETHODIMP CSySalDataIO::Read2(HSySalHANDLE hWnd, BYTE *pData, UINT *pExtErrorInfo, BYTE *FileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState())

	// TODO: Add your implementation code here
	if (!OkToRun)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_CANTRUN;
		return S_FALSE;
		};
	if (!C.InputData.Enabled)
	{
		*pExtErrorInfo = IODATA_EXT_ERROR_DISABLED;
		return S_FALSE;
		};
	IO_Data **pMyData = (IO_Data **)(void *)(pData);	
	int i;
	if ((!FileName || FileName[0] == 0) && (C.InputData.Path[0] == 0))
	{
		CString TempPath;
		CWnd Owner;
		Owner.Attach((HWND)hWnd);
		if (!SelectInputFile(TempPath, "Select Input File", C.InputData.Ext[0] ? C.InputData.Ext : "", 
			C.InputData.Desc[0] ? C.InputData.Desc : "Track data files|*.tlg|Photo Sequence files|*.phs|Cluster Dump files|*.cls|Data files|*.dat|All files|*.*||", &Owner))
		{
			*pExtErrorInfo = IODATA_EXT_ERROR_ABORTED;
			Owner.Detach();
			return S_FALSE;
			};
		Owner.Detach();
		for (i = 0; i < C.InputData.Attempts && ((*pExtErrorInfo = Read(TempPath.GetBuffer(1), pMyData, false)) != IODATA_EXT_ERROR_OK); i++)
			TempPath.ReleaseBuffer();
		}
	else
	{
		if (!FileName || FileName[0] == 0)
			for (i = 0; i < C.InputData.Attempts && ((*pExtErrorInfo = Read(C.InputData.Path, pMyData, false)) != IODATA_EXT_ERROR_OK); i++);
		else
			for (i = 0; i < C.InputData.Attempts && ((*pExtErrorInfo = Read((char *)FileName, pMyData, false)) != IODATA_EXT_ERROR_OK); i++);
		};
	return (*pExtErrorInfo == IODATA_EXT_ERROR_OK) ? S_OK : S_FALSE;
}

STDMETHODIMP CSySalDataIO::GetStatus(UINT *pExtErrorInfo)
{
	//AFX_MANAGE_STATE(AfxGetStaticModuleState())

	// TODO: Add your implementation code here
	*pExtErrorInfo = (OkToRun) ? IODATA_EXT_ERROR_OK : IODATA_EXT_ERROR_CANTRUN;
	return S_OK;
}

#define TRY_WRITE(h, bw, d, l, err) if (!WriteFile(h, d, l, &bw, NULL) || (bw != l)) throw((unsigned)err);

#define TRY_WRITE_B(h, bw, d, l, err) if (!F.WriteBytes((BYTE *)(void *)d, l)) throw((unsigned)err);

UINT CSySalDataIO::WriteBulkTracks(char *FileName, IO_BULK_Tracks *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	BYTE *pCompressedBlock = 0;			
	try
	{
		BULK_TracksHeader TmpHdr = pData->Hdr;
		TmpHdr.CodingMode = (C.OutputData.GrainDiffCompressionData.Enabled) ? BULK_COMPRESSION_GRAINS_DIFF : BULK_COMPRESSION_NULL;
		DWORD bw;
		TRY_WRITE_B(HFile, bw, &TmpHdr, sizeof(TmpHdr), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, pData->FieldHistory, (2 * pData->Hdr.FCount), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_FIELD_SECTION);
		int i, j;
		for (i = 0; i < 2; i++)
			for (j = 0; j < pData->Hdr.TCount[i]; j++)
			{
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Field, sizeof(pData->pTracks[i][j].Field), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].PointsN, sizeof(pData->pTracks[i][j].PointsN), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Intercept, sizeof(pData->pTracks[i][j].Intercept), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Slope, sizeof(pData->pTracks[i][j].Slope), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Sigma, sizeof(pData->pTracks[i][j].Sigma), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].FirstZ, sizeof(pData->pTracks[i][j].FirstZ), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].LastZ, sizeof(pData->pTracks[i][j].LastZ), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].InterceptErrors, sizeof(pData->pTracks[i][j].InterceptErrors), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].SlopeErrors, sizeof(pData->pTracks[i][j].SlopeErrors), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				if (pData->Hdr.FitCorrectionDataSize)
					TRY_WRITE_B(HFile, bw, pData->pTracks[i][j].pCorrection, pData->Hdr.FitCorrectionDataSize, IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_TRACK_SECTION);
				};
		for (i = 0; i < 2; i++)
			for (j = 0; j < pData->Hdr.BTCount[i]; j++)
			{
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].Field, sizeof(pData->pFlatTracks[i][j].Field), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].StripsN, sizeof(pData->pFlatTracks[i][j].StripsN), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].Intercept, sizeof(pData->pFlatTracks[i][j].Intercept), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].Slope, sizeof(pData->pFlatTracks[i][j].Slope), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].HorizSigma, sizeof(pData->pFlatTracks[i][j].HorizSigma), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].VertSigma, sizeof(pData->pFlatTracks[i][j].VertSigma), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].ZBase, sizeof(pData->pFlatTracks[i][j].ZBase), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].LastL, sizeof(pData->pFlatTracks[i][j].LastL), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].InterceptErrors, sizeof(pData->pFlatTracks[i][j].InterceptErrors), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pFlatTracks[i][j].SlopeErrors, sizeof(pData->pFlatTracks[i][j].SlopeErrors), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				if (pData->Hdr.FitCorrectionDataSize)
					TRY_WRITE_B(HFile, bw, pData->pFlatTracks[i][j].pCorrection, pData->Hdr.FitCorrectionDataSize, IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKTRACK_SECTION);
				};

		if (pData->Hdr.CodingMode == BULK_COMPRESSION_NULL)
		{
			for (i = 0; i < 2; i++)
				for (j = 0; j < pData->Hdr.TCount[i]; j++)
					TRY_WRITE_B(HFile, bw, pData->pTracks[i][j].pPoints, (sizeof(TVector) * pData->pTracks[i][j].PointsN), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_GRAIN_SECTION);
			}
		else
		{
			bw = 0;
			unsigned CompressedBlockSize = 0;
			if (!CompressGrains_DifferentialMethod(pData, C.OutputData.GrainDiffCompressionData.TransversePrecision,
				C.OutputData.GrainDiffCompressionData.DepthPrecision, &pCompressedBlock, &CompressedBlockSize))
				throw MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_COMPRESS, pData->Hdr.Type);

			TRY_WRITE_B(HFile, bw, pCompressedBlock, CompressedBlockSize, IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_BLACKSTRIP_SECTION);
			};
		for (i = 0; i < 2; i++)
			for (j = 0; j < pData->Hdr.BTCount[i]; j++)
				TRY_WRITE_B(HFile, bw, pData->pFlatTracks[i][j].pStrips, (sizeof(FlatStrip) * pData->pFlatTracks[i][j].StripsN), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_GRAIN_SECTION);
		}
	catch (unsigned err)
	{
		if (pCompressedBlock) free(pCompressedBlock);
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	if (pCompressedBlock) free(pCompressedBlock);
	//CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::WriteBulkPhotoSeq(char *FileName, IO_BULK_PhotoSeq *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;

		if (pData->Hdr.Type.HeaderFormat == (BULK_HEADER_BYTES | BULK_IMAGEPRED_SECTION)) 
		{
			TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
			}
		else
		{
			TRY_WRITE_B(HFile, bw, &pData->Hdr, (sizeof(pData->Hdr) - 2 * sizeof(float)), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
			};
		int i;
		int ImageSize = pData->Hdr.Height * pData->Hdr.Width;
		for (i = 0; i < pData->Hdr.Frames; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->pImages[i].Z, sizeof(pData->pImages[i].Z), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_IMAGE_SECTION);
			TRY_WRITE_B(HFile, bw, pData->pImages[i].pImage, ImageSize, IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_IMAGE_SECTION);
			};
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);
	F.Close();
	return IODATA_EXT_ERROR_OK;	
}

UINT CSySalDataIO::WriteBulkClustersDump(char *FileName, IO_BULK_ClustersDump *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;
		if (pData->Hdr.Type.InfoType == (BULK_HEADER_BYTES | BULK_CLUSTERPRED_SECTION)) 
		{
			TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
			}
		else
		{
			TRY_WRITE_B(HFile, bw, &pData->Hdr, (sizeof(pData->Hdr) - 2 * sizeof(float)), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
			};
		int i;
		for (i = 0; i < pData->Hdr.Frames; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->pPlanes[i].Z, sizeof(pData->pPlanes[i].Z), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_CLUSTER_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pPlanes[i].Count, sizeof(pData->pPlanes[i].Count), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_CLUSTER_SECTION);
			TRY_WRITE_B(HFile, bw, pData->pPlanes[i].pClusters, (sizeof(Cluster) * pData->pPlanes[i].Count), IODATA_EXT_ERROR_WRITE + BULK_DATA_BYTES + BULK_CLUSTER_SECTION);
			};
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);
	F.Close();
	return IODATA_EXT_ERROR_OK;	
}

UINT CSySalDataIO::WriteCSSSTracks(char *FileName, IO_CS_SS_Tracks *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, pData->FieldHistory, (2 * pData->Hdr.FCount), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_FIELD_SECTION);
		int i, j;
		for (i = 0; i < 2; i++)
			for (j = 0; j < pData->Hdr.TCount[i]; j++)
			{
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Field, sizeof(pData->pTracks[i][j].Field), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].PointsN, sizeof(pData->pTracks[i][j].PointsN), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Intercept, sizeof(pData->pTracks[i][j].Intercept), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Slope, sizeof(pData->pTracks[i][j].Slope), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Sigma, sizeof(pData->pTracks[i][j].Sigma), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].FirstZ, sizeof(pData->pTracks[i][j].FirstZ), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].LastZ, sizeof(pData->pTracks[i][j].LastZ), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].InterceptErrors, sizeof(pData->pTracks[i][j].InterceptErrors), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].SlopeErrors, sizeof(pData->pTracks[i][j].SlopeErrors), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				};
		for (i = 0; i < pData->Hdr.LCount; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].PointsN, sizeof(pData->pLinked[i].PointsN), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Intercept, sizeof(pData->pLinked[i].Intercept), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Slope, sizeof(pData->pLinked[i].Slope), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Sigma, sizeof(pData->pLinked[i].Sigma), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].InterceptErrors, sizeof(pData->pLinked[i].InterceptErrors), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].SlopeErrors, sizeof(pData->pLinked[i].SlopeErrors), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			j = pData->pLinked[i].pTracks[0] - pData->pTracks[0];
			TRY_WRITE_B(HFile, bw, &j, sizeof(j), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			j = pData->pLinked[i].pTracks[1] - pData->pTracks[1];
			TRY_WRITE_B(HFile, bw, &j, sizeof(j), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			};
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::WriteOPERATracks(char *FileName, IO_OPERA_Tracks *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + BULK_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, pData->FieldHistory, (2 * pData->Hdr.FCount), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_FIELD_SECTION);
		int i, j;
		for (i = 0; i < 2; i++)
			for (j = 0; j < pData->Hdr.TCount[i]; j++)
			{
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Field, sizeof(pData->pTracks[i][j].Field), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].AreaSum, sizeof(pData->pTracks[i][j].AreaSum), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Grains, sizeof(pData->pTracks[i][j].Grains), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Intercept, sizeof(pData->pTracks[i][j].Intercept), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Slope, sizeof(pData->pTracks[i][j].Slope), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].Sigma, sizeof(pData->pTracks[i][j].Sigma), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].FirstZ, sizeof(pData->pTracks[i][j].FirstZ), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pTracks[i][j].LastZ, sizeof(pData->pTracks[i][j].LastZ), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				};
		for (i = 0; i < pData->Hdr.LCount; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].AreaSum, sizeof(pData->pLinked[i].AreaSum), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Grains, sizeof(pData->pLinked[i].Grains), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Intercept, sizeof(pData->pLinked[i].Intercept), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Slope, sizeof(pData->pLinked[i].Slope), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pLinked[i].Sigma, sizeof(pData->pLinked[i].Sigma), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			j = pData->pLinked[i].pTracks[0] - pData->pTracks[0];
			TRY_WRITE_B(HFile, bw, &j, sizeof(j), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			j = pData->pLinked[i].pTracks[1] - pData->pTracks[1];
			TRY_WRITE_B(HFile, bw, &j, sizeof(j), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
			};
		if (pData->Hdr.Type.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX)
		{
			TRY_WRITE_B(HFile, bw, pData->pIndexEntries[0], (sizeof(OPERA_TrackIndexEntry) * pData->Hdr.TCount[0]), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + OPERA_INDEX_SECTION);
			TRY_WRITE_B(HFile, bw, pData->pIndexEntries[1], (sizeof(OPERA_TrackIndexEntry) * pData->Hdr.TCount[1]), IODATA_EXT_ERROR_WRITE + CS_SS_DATA_BYTES + OPERA_INDEX_SECTION);
			}
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::WriteBulkCSSSBatch(char *FileName, IO_BS_Batch *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr);

	try
	{
		DWORD bw;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + BS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->CountOfTracks, sizeof(pData->CountOfTracks), IODATA_EXT_ERROR_WRITE + BS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->CountOfClusterDumps, sizeof(pData->CountOfClusterDumps), IODATA_EXT_ERROR_WRITE + BS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->CountOfPhotoSequences, sizeof(pData->CountOfPhotoSequences), IODATA_EXT_ERROR_WRITE + BS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->Reserved, sizeof(pData->Reserved), IODATA_EXT_ERROR_WRITE + BS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->pReserved, sizeof(pData->pReserved), IODATA_EXT_ERROR_WRITE + BS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, pData->pTracks, (sizeof(BS_Track) * pData->CountOfTracks), IODATA_EXT_ERROR_WRITE + BS_DATA_BYTES + BS_TRACK_SECTION);
		TRY_WRITE_B(HFile, bw, pData->pClusterDumps, (sizeof(BS_ClusterDump) * pData->CountOfClusterDumps), IODATA_EXT_ERROR_WRITE + BS_DATA_BYTES + BS_CLUSTERPRED_SECTION);
		TRY_WRITE_B(HFile, bw, pData->pPhotoSequences, (sizeof(BS_PhotoSequence) * pData->CountOfPhotoSequences), IODATA_EXT_ERROR_WRITE + BS_DATA_BYTES + BS_IMAGEPRED_SECTION);
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
	};

UINT CSySalDataIO::WriteTSReconstruction(char *FileName, IO_TS_Reconstruction *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
	
	try
	{
		DWORD bw;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + TS_HEADER_BYTES);
		int i, s, j, k, h, index, segnum;
		for (i = 0; i < pData->Hdr.SheetsCount; i++)			
		{
			TRY_WRITE_B(HFile, bw, &pData->pLayers[i], (sizeof(pData->pLayers[i]) - sizeof(pData->pLayers[i].pRawTracks) - sizeof(pData->pLayers[i].pSegments) - sizeof(pData->pLayers[i].pReserved)), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_LAYER_SECTION);
			for (s = 0; s < 2; s++)
			{
				TRY_WRITE_B(HFile, bw, pData->pLayers[i].pRawTracks[s], (sizeof(TS_RawTrack) * pData->pLayers[i].RTCount[s]), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
				for (j = 0; j < pData->pLayers[i].SCount[s]; j++)
				{					
					for (k = 0; k < pData->Hdr.MaxTracksInSegment; k++)
					{
						if (pData->pLayers[i].pSegments[s][j].pTracks[k])
							index = pData->pLayers[i].pSegments[s][j].pTracks[k] - pData->pLayers[i].pRawTracks[s];
						else index = -1;
						TRY_WRITE_B(HFile, bw, &index, sizeof(index), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
						};

					if (!(pData->pLayers[i].pSegments[s][j].pBottomLink))
					{
						segnum = -1;
						TRY_WRITE_B(HFile, bw, &segnum, sizeof(segnum), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
						}
					else
					{
						segnum = 0;
						for (h = 0; h < (2 * pData->Hdr.SheetsCount); h++)
						{
							int l = h / 2;
							int t = h % 2;
							k = pData->pLayers[i].pSegments[s][j].pBottomLink - pData->pLayers[l].pSegments[t];
							if (k >= 0 && k < pData->pLayers[l].SCount[t])
							{
								segnum += k;
								TRY_WRITE_B(HFile, bw, &segnum, sizeof(segnum), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
								break;
								}
							else segnum += pData->pLayers[l].SCount[t];
							};
						if (h >= (2 * pData->Hdr.SheetsCount)) throw (IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
						};

					if (!(pData->pLayers[i].pSegments[s][j].pTopLink))
					{
						segnum = -1;
						TRY_WRITE_B(HFile, bw, &segnum, sizeof(segnum), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
						}
					else
					{
						segnum = 0;
						for (h = 0; h < (2 * pData->Hdr.SheetsCount); h++)
						{
							int l = h / 2;
							int t = h % 2;
							k = pData->pLayers[i].pSegments[s][j].pTopLink - pData->pLayers[l].pSegments[t];
							if (k >= 0 && k < pData->pLayers[l].SCount[t])
							{
								segnum += k;
								TRY_WRITE_B(HFile, bw, &segnum, sizeof(segnum), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
								break;
								}
							else segnum += pData->pLayers[l].SCount[t];
							};
						if (h >= (2 * pData->Hdr.SheetsCount)) throw (IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
						};
					
					TRY_WRITE_B(HFile, bw, &pData->pLayers[i].pSegments[s][j].PointsN, sizeof(pData->pLayers[i].pSegments[s][j].PointsN), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_WRITE_B(HFile, bw, &pData->pLayers[i].pSegments[s][j].Intercept, sizeof(pData->pLayers[i].pSegments[s][j].Intercept), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_WRITE_B(HFile, bw, &pData->pLayers[i].pSegments[s][j].Slope, sizeof(pData->pLayers[i].pSegments[s][j].Slope), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_WRITE_B(HFile, bw, &pData->pLayers[i].pSegments[s][j].Sigma, sizeof(pData->pLayers[i].pSegments[s][j].Sigma), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_WRITE_B(HFile, bw, &pData->pLayers[i].pSegments[s][j].FirstZ, sizeof(pData->pLayers[i].pSegments[s][j].FirstZ), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_WRITE_B(HFile, bw, &pData->pLayers[i].pSegments[s][j].LastZ, sizeof(pData->pLayers[i].pSegments[s][j].LastZ), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					};
				};
			};
		for (i = 0; i < pData->Hdr.TSTracksCount; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].Count, sizeof(pData->pTracks[i].Count), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);

			segnum = 0;
			for (j = 0; j < (2 * pData->Hdr.SheetsCount); j++)
			{
				int l = j / 2;
				s = j % 2;
				k = pData->pTracks[i].pBottomEnd - pData->pLayers[l].pSegments[s];
				if (k >= 0 && k < pData->pLayers[l].SCount[s])
				{
					segnum += k;
					TRY_WRITE_B(HFile, bw, &segnum, sizeof(segnum), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
					break;
					}
				else segnum += pData->pLayers[l].SCount[s];
				};
			if (j >= (2 * pData->Hdr.SheetsCount)) throw (IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);

			segnum = 0;
			for (j = 0; j < (2 * pData->Hdr.SheetsCount); j++)
			{
				int l = j / 2;
				s = j % 2;
				k = pData->pTracks[i].pTopEnd - pData->pLayers[l].pSegments[s];
				if (k >= 0 && k < pData->pLayers[l].SCount[s])
				{
					segnum += k;
					TRY_WRITE_B(HFile, bw, &segnum, sizeof(segnum), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
					break;
					}
				else segnum += pData->pLayers[l].SCount[s];
				};
			if (j >= (2 * pData->Hdr.SheetsCount)) throw (IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);

			if (pData->pTracks[i].pBottomVertex)
				index = pData->pTracks[i].pBottomVertex - pData->pVertices;
			else index = -1;
			TRY_WRITE_B(HFile, bw, &index, sizeof(index), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);

			if (pData->pTracks[i].pTopVertex)
				index = pData->pTracks[i].pTopVertex - pData->pVertices;
			else index = -1;
			TRY_WRITE_B(HFile, bw, &index, sizeof(index), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);

			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].Flags, sizeof(pData->pTracks[i].Flags), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].Intercept, sizeof(pData->pTracks[i].Intercept), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].Slope, sizeof(pData->pTracks[i].Slope), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].Sigma, sizeof(pData->pTracks[i].Sigma), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].FirstZ, sizeof(pData->pTracks[i].FirstZ), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].LastZ, sizeof(pData->pTracks[i].LastZ), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pTracks[i].Reserved, sizeof(pData->pTracks[i].Reserved), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_TRACK_SECTION);
			};
		for (i = 0; i < pData->Hdr.VerticesCount; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->pVertices[i].Count, sizeof(pData->pVertices[i].Count), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pVertices[i].Pos, sizeof(pData->pVertices[i].Pos), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->pVertices[i].AvgDistance, sizeof(pData->pVertices[i].AvgDistance), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
			for (j = 0; j < pData->pVertices[i].Count; j++)
			{
				index = pData->pVertices[i].pTSTracks[j].pTSTrack - pData->pTracks;
				TRY_WRITE_B(HFile, bw, &index, sizeof(index), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pVertices[i].pTSTracks[j].IsDownstream, sizeof(pData->pVertices[i].pTSTracks[j].IsDownstream), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
				TRY_WRITE_B(HFile, bw, &pData->pVertices[i].pTSTracks[j].Reserved, sizeof(pData->pVertices[i].pTSTracks[j].Reserved), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
				};
			TRY_WRITE_B(HFile, bw, &pData->pVertices[i].Reserved, sizeof(pData->pVertices[i].Reserved), IODATA_EXT_ERROR_WRITE + TS_DATA_BYTES + TS_VERTEX_SECTION);
			};
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::WriteMarkMap(char *FileName, IO_MarkMap *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Hdr);
*/	
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Hdr);

	try
	{
		DWORD bw;

		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + MAP_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->Map.Count, sizeof(pData->Map.Count), IODATA_EXT_ERROR_WRITE + MAP_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, pData->Map.pMarks, (pData->Map.Count * sizeof(Mark)), IODATA_EXT_ERROR_WRITE + MAP_DATA_BYTES);
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};
	//CloseHandle(HFile);
	F.Close();
	return IODATA_EXT_ERROR_OK;	
}

UINT CSySalDataIO::WriteVertigoScanCatalog(char *FileName, IO_VS_Catalog *pData)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile F;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;
		int i;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + VS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->Area, sizeof(pData->Area), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CATALOG_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Config.CountOfConfigs, sizeof(pData->Config.CountOfConfigs), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CATALOG_SECTION);
		for (i = 0; i < pData->Config.CountOfConfigs; i++)
		{
			TRY_WRITE_B(HFile, bw, &pData->Config.pConfigs[i].ClassName, sizeof(pData->Config.pConfigs[i].ClassName), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CONFIG_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->Config.pConfigs[i].Config.Name, sizeof(pData->Config.pConfigs[i].Config.Name), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CONFIG_SECTION);
			TRY_WRITE_B(HFile, bw, &pData->Config.pConfigs[i].Config.CountOfItems, sizeof(pData->Config.pConfigs[i].Config.CountOfItems), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CONFIG_SECTION);
			TRY_WRITE_B(HFile, bw, pData->Config.pConfigs[i].Config.pItems, ((pData->Config.pConfigs[i].Config.CountOfItems) * (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN)), 
				IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CONFIG_SECTION);
			};
		TRY_WRITE_B(HFile, bw, pData->pFragmentIndices, (sizeof(int) * pData->Area.XViews * pData->Area.YViews), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Reserved, sizeof(pData->Reserved), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_CATALOG_SECTION);
		}
	catch (unsigned err)
	{
		//CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};

	//CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::WriteVertigoScanFragment(char *FileName, IO_VS_Fragment *pData)
{
	switch (pData->Fragment.CodingMode)
	{
		case VS_COMPRESSION_NULL:
		case VS_COMPRESSION_GRAIN_SUPPRESSION:	break;

		default:								return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKCOMPRESS, pData->Hdr.Type);
		};

/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/	
	BufferedFile FB;
	AsyncBufferedFile AB(C.OutputData.Attempts, C.OutputData.AsyncWriteBufferMB * 1048576);
	BufferedFile &F = (C.OutputData.AsyncWriteBufferMB > 0) ? AB : FB;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;
		int i, j, k;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + VS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.Index, sizeof(pData->Fragment.Index), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.StartView, sizeof(pData->Fragment.StartView), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.CountOfViews, sizeof(pData->Fragment.CountOfViews), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.FitCorrectionDataSize, sizeof(pData->Fragment.FitCorrectionDataSize), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.CodingMode, sizeof(pData->Fragment.CodingMode), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Reserved, sizeof(pData->Reserved), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		for (i = 0; i < pData->Fragment.CountOfViews; i++)
		{
			VS_View &v = pData->Fragment.pViews[i];
			TRY_WRITE_B(HFile, bw, &v.TileX, sizeof(v.TileX), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.TileY, sizeof(v.TileY), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.X, sizeof(v.X), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.Y, sizeof(v.Y), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.MapX, sizeof(v.MapX), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.MapY, sizeof(v.MapY), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.ImageMat, sizeof(v.ImageMat), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			for (j = 0; j < 2; j++)	TRY_WRITE_B(HFile, bw, &v.Layers[j].Count, sizeof(v.Layers[j].Count), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.RelevantZs, sizeof(v.RelevantZs), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.Status, sizeof(v.Status), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.TCount, sizeof(v.TCount), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			};
		for (i = 0; i < pData->Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
				TRY_WRITE_B(HFile, bw, pData->Fragment.pViews[i].Layers[j].pZs, (sizeof(float) * pData->Fragment.pViews[i].Layers[j].Count), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
		for (i = 0; i < pData->Fragment.CountOfViews; i++)
		{
			VS_View &v = pData->Fragment.pViews[i];
			for (j = 0; j < 2; j++)
				for (k = 0; k < v.TCount[j]; k++)
				{
					Track &t = v.pTracks[j][k];
					TRY_WRITE_B(HFile, bw, &t.PointsN, sizeof(t.PointsN), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Intercept, sizeof(t.Intercept), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Slope, sizeof(t.Slope), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Sigma, sizeof(t.Sigma), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.FirstZ, sizeof(t.FirstZ), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.LastZ, sizeof(t.LastZ), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.InterceptErrors, sizeof(t.InterceptErrors), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.SlopeErrors, sizeof(t.SlopeErrors), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					if (pData->Fragment.FitCorrectionDataSize)
						TRY_WRITE_B(HFile, bw, t.pCorrection, pData->Fragment.FitCorrectionDataSize, IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					};
			};

		for (i = 0; i < pData->Fragment.CountOfViews; i++)
		{
			switch (pData->Fragment.CodingMode)
			{
				case VS_COMPRESSION_NULL:				for (j = 0; j < 2; j++)
															for (k = 0; k < pData->Fragment.pViews[i].TCount[j]; k++)
																TRY_WRITE_B(HFile, bw, pData->Fragment.pViews[i].pTracks[j][k].pPoints, (sizeof(TVector) * pData->Fragment.pViews[i].pTracks[j][k].PointsN), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_GRAIN_SECTION);
														break;

				case VS_COMPRESSION_GRAIN_SUPPRESSION:	break;
				};
			};
		}
	catch (unsigned err)
	{
//		CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};

//	CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::WriteVertigoScanFragment2(char *FileName, IO_VS_Fragment2 *pData)
{
	if (pData->Fragment.FitCorrectionDataSize != 0) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OBSOLETE, pData->Hdr.Type);
	switch (pData->Fragment.CodingMode)
	{
		case VS_COMPRESSION_NULL:
		case VS_COMPRESSION_GRAIN_SUPPRESSION:	break;

		default:								return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKCOMPRESS, pData->Hdr.Type);
		};

/*
	HANDLE HFile = CreateFile(FileName, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);
*/
	BufferedFile FB;
	AsyncBufferedFile AB(C.OutputData.Attempts, C.OutputData.AsyncWriteBufferMB * 1048576);
	BufferedFile &F = (C.OutputData.AsyncWriteBufferMB > 0) ? AB : FB;
	if (!F.Open(FileName, true)) return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OPEN, pData->Hdr.Type);

	try
	{
		DWORD bw;
		int i, j, k;
		TRY_WRITE_B(HFile, bw, &pData->Hdr, sizeof(pData->Hdr), IODATA_EXT_ERROR_WRITE + VS_HEADER_BYTES);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.Index, sizeof(pData->Fragment.Index), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.StartView, sizeof(pData->Fragment.StartView), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.CountOfViews, sizeof(pData->Fragment.CountOfViews), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.FitCorrectionDataSize, sizeof(pData->Fragment.FitCorrectionDataSize), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Fragment.CodingMode, sizeof(pData->Fragment.CodingMode), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_WRITE_B(HFile, bw, &pData->Reserved, sizeof(pData->Reserved), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		for (i = 0; i < pData->Fragment.CountOfViews; i++)
		{
			VS_View2 &v = pData->Fragment.pViews[i];
			TRY_WRITE_B(HFile, bw, &v.TileX, sizeof(v.TileX), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.TileY, sizeof(v.TileY), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.X, sizeof(v.X), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.Y, sizeof(v.Y), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.MapX, sizeof(v.MapX), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.MapY, sizeof(v.MapY), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.ImageMat, sizeof(v.ImageMat), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			for (j = 0; j < 2; j++)	TRY_WRITE_B(HFile, bw, &v.Layers[j].Count, sizeof(v.Layers[j].Count), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.RelevantZs, sizeof(v.RelevantZs), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.Status, sizeof(v.Status), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_WRITE_B(HFile, bw, &v.TCount, sizeof(v.TCount), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
			};
		for (i = 0; i < pData->Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
				for (k = 0; k < pData->Fragment.pViews[i].Layers[j].Count; k++)
				{
					TRY_WRITE_B(HFile, bw, &pData->Fragment.pViews[i].Layers[j].pLayerInfo[k].Clusters, sizeof(pData->Fragment.pViews[i].Layers[j].pLayerInfo[k].Clusters), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
					TRY_WRITE_B(HFile, bw, &pData->Fragment.pViews[i].Layers[j].pLayerInfo[k].Z, sizeof(pData->Fragment.pViews[i].Layers[j].pLayerInfo[k].Z), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_VIEW_SECTION);
					}
		for (i = 0; i < pData->Fragment.CountOfViews; i++)
		{
			VS_View2 &v = pData->Fragment.pViews[i];
			for (j = 0; j < 2; j++)
				for (k = 0; k < v.TCount[j]; k++)
				{
					Track2 &t = v.pTracks[j][k];
					TRY_WRITE_B(HFile, bw, &t.AreaSum, sizeof(t.AreaSum), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Grains, sizeof(t.Grains), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Intercept, sizeof(t.Intercept), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Slope, sizeof(t.Slope), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.Sigma, sizeof(t.Sigma), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.FirstZ, sizeof(t.FirstZ), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					TRY_WRITE_B(HFile, bw, &t.LastZ, sizeof(t.LastZ), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
					};
			};

		for (i = 0; i < pData->Fragment.CountOfViews; i++)
		{
			switch (pData->Fragment.CodingMode)
			{
				case VS_COMPRESSION_NULL:				for (j = 0; j < 2; j++)
															for (k = 0; k < pData->Fragment.pViews[i].TCount[j]; k++)
																TRY_WRITE_B(HFile, bw, pData->Fragment.pViews[i].pTracks[j][k].pGrains, 
																	(sizeof(Grain) * pData->Fragment.pViews[i].pTracks[j][k].Grains), IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_GRAIN_SECTION);
														break;

				case VS_COMPRESSION_GRAIN_SUPPRESSION:	break;
				};
			};
		}
	catch (unsigned err)
	{
//		CloseHandle(HFile);
		F.Close();
		DeleteFile(FileName);
		return err;
		};

//	CloseHandle(HFile);	
	F.Close();
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::Write(char *DestPath, char *RescuePath, char *FileName, IO_Data *pData, bool UseISySalDataIO)
{	
	if ((pData->Hdr.InfoType != (BULK_HEADER_BYTES | BULK_TRACK_SECTION) || (pData->Hdr.HeaderFormat != BULK_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (BULK_HEADER_BYTES | BULK_IMAGE_SECTION) || (pData->Hdr.HeaderFormat != BULK_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (BULK_HEADER_BYTES | BULK_CLUSTER_SECTION) || (pData->Hdr.HeaderFormat != BULK_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (BULK_HEADER_BYTES | BULK_IMAGEPRED_SECTION) || (pData->Hdr.HeaderFormat != BULK_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (BULK_HEADER_BYTES | BULK_CLUSTERPRED_SECTION) || (pData->Hdr.HeaderFormat != BULK_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) || (pData->Hdr.HeaderFormat != CS_SS_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) || (pData->Hdr.HeaderFormat != OPERA_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) || (pData->Hdr.HeaderFormat != OPERA_HEADER_TYPE_WITH_INDEX)) &&
		(pData->Hdr.InfoType != (BS_HEADER_BYTES | BS_PRED_SECTION) || (pData->Hdr.HeaderFormat != BS_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (TS_HEADER_BYTES | TS_FULL_SECTION) || (pData->Hdr.HeaderFormat != TS_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (MAP_HEADER_BYTES | MAP_FULL_SECTION) || (pData->Hdr.HeaderFormat != MAP_MARK_HEADER_TYPE)) && 
		(pData->Hdr.InfoType != (VS_HEADER_BYTES | VS_CATALOG_SECTION) || (pData->Hdr.HeaderFormat != VS_OLD_HEADER_TYPE)) && 
		(pData->Hdr.InfoType != (VS_HEADER_BYTES | VS_CATALOG_SECTION) || (pData->Hdr.HeaderFormat != VS_HEADER_TYPE) || UseISySalDataIO) && 
		(pData->Hdr.InfoType != (VS_HEADER_BYTES | VS_FRAGMENT_SECTION) || (pData->Hdr.HeaderFormat != VS_OLD_HEADER_TYPE)) &&
		(pData->Hdr.InfoType != (VS_HEADER_BYTES | VS_FRAGMENT_SECTION) || (pData->Hdr.HeaderFormat != VS_HEADER_TYPE) || UseISySalDataIO))
			return MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKFORMAT, pData->Hdr);
	char *dest = new char[1 + strlen(DestPath) + strlen(FileName) + 1 + strlen(RescuePath) + strlen(FileName)];
	char *rescue = dest + 1 + strlen(RescuePath) + strlen(FileName);
	sprintf(dest, "%s%s", DestPath, FileName);
	sprintf(rescue, "%s%s", RescuePath, FileName);	
	char *c = dest;
	UINT Err;
	while (c)
	{
		int i;
		for (i = 0; i < C.OutputData.Attempts; i++)
		{
			if (pData->Hdr.InfoType == (BULK_HEADER_BYTES | BULK_TRACK_SECTION) && pData->Hdr.HeaderFormat == BULK_HEADER_TYPE) Err = WriteBulkTracks(c, &pData->BTks);
			else if (((pData->Hdr.InfoType == (BULK_HEADER_BYTES | BULK_IMAGE_SECTION)) || 
				(pData->Hdr.InfoType == (BULK_HEADER_BYTES | BULK_IMAGEPRED_SECTION))) && pData->Hdr.HeaderFormat == BULK_HEADER_TYPE) Err = WriteBulkPhotoSeq(c, &pData->BPhSeq);			
			else if (((pData->Hdr.InfoType == (BULK_HEADER_BYTES | BULK_CLUSTER_SECTION)) || 
				(pData->Hdr.InfoType == (BULK_HEADER_BYTES | BULK_CLUSTERPRED_SECTION))) && pData->Hdr.HeaderFormat == BULK_HEADER_TYPE) Err = WriteBulkClustersDump(c, &pData->BCDmp);
			else if (pData->Hdr.InfoType == (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) && pData->Hdr.HeaderFormat == CS_SS_HEADER_TYPE) Err = WriteCSSSTracks(c, &pData->STks);
			else if (pData->Hdr.InfoType == (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) && pData->Hdr.HeaderFormat == OPERA_HEADER_TYPE && !UseISySalDataIO) Err = WriteOPERATracks(c, &pData->OTks);
			else if (pData->Hdr.InfoType == (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) && pData->Hdr.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX && !UseISySalDataIO) Err = WriteOPERATracks(c, &pData->OTks);
			else if (pData->Hdr.InfoType == (BS_HEADER_BYTES | BS_PRED_SECTION) && pData->Hdr.HeaderFormat == BS_HEADER_TYPE) Err = WriteBulkCSSSBatch(c, &pData->BSBatch);
			else if (pData->Hdr.InfoType == (TS_HEADER_BYTES | TS_FULL_SECTION) && pData->Hdr.HeaderFormat == TS_HEADER_TYPE) Err = WriteTSReconstruction(c, &pData->TSRec);
			else if (pData->Hdr.InfoType == (MAP_HEADER_BYTES | MAP_FULL_SECTION) && pData->Hdr.HeaderFormat == MAP_MARK_HEADER_TYPE) Err = WriteMarkMap(c, &pData->MkMap);
			else if ((pData->Hdr.InfoType == (VS_HEADER_BYTES | VS_CATALOG_SECTION)) && (pData->Hdr.HeaderFormat == VS_HEADER_TYPE || pData->Hdr.HeaderFormat == VS_OLD_HEADER_TYPE)) Err = WriteVertigoScanCatalog(c, &pData->VSCat);
			else if (pData->Hdr.InfoType == (VS_HEADER_BYTES | VS_FRAGMENT_SECTION) && pData->Hdr.HeaderFormat == VS_OLD_HEADER_TYPE) Err = WriteVertigoScanFragment(c, &pData->VSFrag);
			else if ((pData->Hdr.InfoType == (VS_HEADER_BYTES | VS_FRAGMENT_SECTION)) && (pData->Hdr.HeaderFormat == VS_HEADER_TYPE) && !UseISySalDataIO) Err = WriteVertigoScanFragment2(c, &pData->VSFrag2);
			else 
			{
				/* ??? */
				};
			if (Err == IODATA_EXT_ERROR_OK) break;
			};
		if (i < C.OutputData.Attempts) c = 0;
		else 
		{
			if (c == dest) c = rescue;
			else c = 0;
			};
		};
	delete [] dest;
	return Err;
}

void CSySalDataIO::GetExtAndDesc(IO_Data *pData, char **pExt, char **pDesc)
{
	static char FragExt[16];
	switch (pData->Hdr.InfoType)
	{
//		case (BULK_HEADER_BYTES | BULK_TRACK_SECTION): // this is the same as below!
		case (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION):	*pExt = "tlg";
															*pDesc = "Track data files";
															break;

		case (BULK_HEADER_BYTES | BULK_IMAGE_SECTION):		*pExt = "phs";
															*pDesc = "Photo Sequence files";
															break;

		case (BULK_HEADER_BYTES | BULK_CLUSTER_SECTION):	*pExt = "cls";
															*pDesc = "Cluster Dump files";
															break;

		case (BS_HEADER_BYTES | BS_PRED_SECTION):			*pExt = "pre";
															*pDesc = "Prediction files";
															break;

		case (TS_HEADER_BYTES | TS_FULL_SECTION):			*pExt = "tsr";
															*pDesc = "TotalScan Full Reconstruction files";
															break;	

		case (MAP_HEADER_BYTES | MAP_FULL_SECTION):			*pExt = "map";
															*pDesc = "Map files";
															break;	

		case (VS_HEADER_BYTES | VS_CATALOG_SECTION):		*pExt = "rwc";
															*pDesc = "Raw Data Catalog files";
															break;	

		case (VS_HEADER_BYTES | VS_FRAGMENT_SECTION):		sprintf(FragExt, "rwd.%08X", pData->VSFrag.Fragment.Index);
															*pExt = FragExt;
															*pDesc = "Raw Data Fragment files";
															break;	

		default:											*pExt = "dat";
															*pDesc = "Data files";
		};
}

#define TRY_READ(h, br, d, l, err) if (!ReadFile(h, d, l, &br, NULL) || (br != l)) throw((unsigned)err);

#define TRY_READ_B(h, br, d, l, err) if (!F.ReadBytes((BYTE *)(void *)d, l, (int *)&br)) throw((unsigned)err);
#define QWORD_ALIGN(x) ((x + 7) & 0xfffffff8)
#define QWORD_ALIGNED_COPY_AND_DISPLACE(p, d, s)\
{\
	int e = QWORD_ALIGN(s);\
	memcpy(p, d, s);\
	p += e;\
	}

UINT CSySalDataIO::ReadBulkTracks(BufferedFile &F, IO_Header Hdr, IO_BULK_Tracks **ppData)
{
	IO_BULK_Tracks Tks;
	Tks.Hdr.Type = Hdr;
	DWORD br;
	BYTE *pFields = 0;
	BYTE *pCorrData = 0;
	BYTE *pC;
	Track *pTracks[2] = {0, 0};
	FlatTrack *pFlatTracks[2] = {0, 0};
	TVector *pPoints = 0;
	FlatStrip *pStrips = 0;
	int GSize = 0, FSize = 0;
	int i, j;
	BYTE *pCompressedBlock = 0;
	try
	{
		if (Tks.Hdr.Type.HeaderFormat == BULK_OLD_HEADER_TYPE)
		{
			TRY_READ_B(HFile, br, &Tks.Hdr.ID.Part[3], (sizeof(Tks.Hdr) - sizeof(Tks.Hdr.Type) - 
				3 * sizeof(Tks.Hdr.ID.Part[0])), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
			Tks.Hdr.ID.Part[0] = Tks.Hdr.ID.Part[1] = Tks.Hdr.ID.Part[2] = 0;
			}
		else
		{
			TRY_READ_B(HFile, br, &Tks.Hdr.ID, (sizeof(Tks.Hdr) - sizeof(Tks.Hdr.Type)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
			};
		pFields = (BYTE *)malloc(2 * Tks.Hdr.FCount);
		TRY_READ_B(HFile, br, pFields, (2 * Tks.Hdr.FCount), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_FIELD_SECTION);		
		if (Tks.Hdr.Type.HeaderFormat != BULK_HEADER_TYPE) Tks.Hdr.BTCount[0] = Tks.Hdr.BTCount[1] = 0;
		else if ((Tks.Hdr.CodingMode != BULK_COMPRESSION_NULL) &&
			(Tks.Hdr.CodingMode != BULK_COMPRESSION_GRAINS_DIFF))
			throw MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKCOMPRESS, Tks.Hdr.Type);
		pTracks[0] = (Track *)malloc(Tks.Hdr.TCount[0] * sizeof(Track));
		pTracks[1] = (Track *)malloc(Tks.Hdr.TCount[1] * sizeof(Track));
		pFlatTracks[0] = (FlatTrack *)malloc(Tks.Hdr.BTCount[0] * sizeof(FlatTrack));
		pFlatTracks[1] = (FlatTrack *)malloc(Tks.Hdr.BTCount[1] * sizeof(FlatTrack));
		pC = pCorrData = (BYTE *)malloc(Tks.Hdr.FitCorrectionDataSize * (Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1] + Tks.Hdr.BTCount[0] + Tks.Hdr.BTCount[1]));
		if (Hdr.HeaderFormat == BULK_HEADER_TYPE)
		{
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].PointsN, sizeof(pTracks[i][j].PointsN), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].InterceptErrors, sizeof(pTracks[i][j].InterceptErrors), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].SlopeErrors, sizeof(pTracks[i][j].SlopeErrors), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					pTracks[i][j].Valid = true;
					TRY_READ_B(HFile, br, pC, Tks.Hdr.FitCorrectionDataSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					pC += Tks.Hdr.FitCorrectionDataSize;
					GSize += pTracks[i][j].PointsN;
					};
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.BTCount[i]; j++)
				{
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].Field, sizeof(pFlatTracks[i][j].Field), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].StripsN, sizeof(pFlatTracks[i][j].StripsN), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].Intercept, sizeof(pFlatTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].Slope, sizeof(pFlatTracks[i][j].Slope), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].HorizSigma, sizeof(pFlatTracks[i][j].HorizSigma), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].VertSigma, sizeof(pFlatTracks[i][j].VertSigma), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].ZBase, sizeof(pFlatTracks[i][j].ZBase), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].LastL, sizeof(pFlatTracks[i][j].LastL), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].InterceptErrors, sizeof(pFlatTracks[i][j].InterceptErrors), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pFlatTracks[i][j].SlopeErrors, sizeof(pFlatTracks[i][j].SlopeErrors), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					pFlatTracks[i][j].Valid = true;
					TRY_READ_B(HFile, br, pC, Tks.Hdr.FitCorrectionDataSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					pC += Tks.Hdr.FitCorrectionDataSize;
					FSize += pFlatTracks[i][j].StripsN;
					};
			}
		else
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].PointsN, sizeof(pTracks[i][j].PointsN), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					pTracks[i][j].InterceptErrors.X = pTracks[i][j].InterceptErrors.Y = pTracks[i][j].InterceptErrors.Z = 0.f;
					pTracks[i][j].SlopeErrors.X = pTracks[i][j].SlopeErrors.Y = pTracks[i][j].SlopeErrors.Z = 0.f;
					pTracks[i][j].Valid = true;
					TRY_READ_B(HFile, br, pC, Tks.Hdr.FitCorrectionDataSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_TRACK_SECTION);
					pC += Tks.Hdr.FitCorrectionDataSize;
					GSize += pTracks[i][j].PointsN;
					};
		if (C.InputData.SkipGrains)
		{
			FSize = 0;
			GSize = 0;
			pPoints = 0;
			pStrips = 0;
			}
		else 
		{
			pPoints = (TVector *)malloc(GSize *= sizeof(TVector));
			pStrips = (FlatStrip *)malloc(FSize *= sizeof(FlatStrip));
			};
		if (Tks.Hdr.CodingMode == BULK_COMPRESSION_NULL)
		{
			if (!C.InputData.SkipGrains)
				TRY_READ_B(HFile, br, pPoints, GSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_GRAIN_SECTION);
			}
		else
		{
			IO_BULK_DiffCompression_Header DCHdr;
			TRY_READ_B(HFile, br, &DCHdr, sizeof(DCHdr), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_GRAIN_SECTION);
			if (!(pCompressedBlock = (BYTE *)malloc(DCHdr.Hdr.Size))) throw IODATA_EXT_ERROR_UNKNOWN + BULK_DATA_BYTES + BULK_GRAIN_SECTION;
			memcpy(pCompressedBlock, &DCHdr, sizeof(DCHdr));
			TRY_READ_B(HFile, br, pCompressedBlock + sizeof(DCHdr), (DCHdr.Hdr.Size - sizeof(DCHdr)), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_GRAIN_SECTION);			
			};
		if (!C.InputData.SkipGrains)
			TRY_READ_B(HFile, br, pStrips, FSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_BLACKSTRIP_SECTION);
		}
	catch (unsigned err)
	{
		if (pFields) free(pFields);
		if (pCorrData) free(pCorrData);
		if (pFlatTracks[0]) free(pFlatTracks[0]);
		if (pFlatTracks[1]) free(pFlatTracks[1]);
		if (pTracks[0]) free(pTracks[0]);
		if (pTracks[1]) free(pTracks[1]);
		if (pPoints) free(pPoints);
		if (pStrips) free(pStrips);
		if (pCompressedBlock) free(pCompressedBlock);
		F.Close(); //CloseHandle(HFile);
		return err;		
		}
	catch (...)
	{
		if (pFields) free(pFields);
		if (pCorrData) free(pCorrData);
		if (pFlatTracks[0]) free(pFlatTracks[0]);
		if (pFlatTracks[1]) free(pFlatTracks[1]);
		if (pTracks[0]) free(pTracks[0]);
		if (pTracks[1]) free(pTracks[1]);
		if (pPoints) free(pPoints);
		if (pStrips) free(pStrips);
		if (pCompressedBlock) free(pCompressedBlock);
		F.Close(); //CloseHandle(HFile);
		return IODATA_EXT_ERROR_UNKNOWN;		
		};
	Tks.Hdr.Type.HeaderFormat = BULK_HEADER_TYPE;
	*ppData = (IO_BULK_Tracks *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(IO_Data)) + QWORD_ALIGN((2 * Tks.Hdr.FCount)) + 
		QWORD_ALIGN(((Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1]) * sizeof(Track))) + 
		QWORD_ALIGN(((Tks.Hdr.BTCount[0] + Tks.Hdr.BTCount[1]) * sizeof(FlatTrack))) + 
		QWORD_ALIGN(((Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1] + Tks.Hdr.BTCount[0] + Tks.Hdr.BTCount[1]) * Tks.Hdr.FitCorrectionDataSize)) + QWORD_ALIGN(GSize) + QWORD_ALIGN(FSize));
	BYTE *p = (BYTE *)*ppData;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Tks, sizeof(Tks));
	(*ppData)->FieldHistory = p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pFields, 2 * Tks.Hdr.FCount);
	(*ppData)->pTracks[0] = (Track *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks[0], (Tks.Hdr.TCount[0] * sizeof(Track)));
	(*ppData)->pTracks[1] = (Track *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks[1], (Tks.Hdr.TCount[1] * sizeof(Track)));
	(*ppData)->pFlatTracks[0] = (FlatTrack *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pFlatTracks[0], (Tks.Hdr.BTCount[0] * sizeof(FlatTrack)));
	(*ppData)->pFlatTracks[1] = (FlatTrack *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pFlatTracks[1], (Tks.Hdr.BTCount[1] * sizeof(FlatTrack)));
	pC = p;	
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pCorrData, ((Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1] + Tks.Hdr.BTCount[0] + Tks.Hdr.BTCount[1]) * Tks.Hdr.FitCorrectionDataSize));
	if (Tks.Hdr.FitCorrectionDataSize)
	{
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.TCount[i]; j++)
			{
				(*ppData)->pTracks[i][j].pCorrection = pC;
				pC += Tks.Hdr.FitCorrectionDataSize;
				};
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.BTCount[i]; j++)
			{
				(*ppData)->pFlatTracks[i][j].pCorrection = pC;
				pC += Tks.Hdr.FitCorrectionDataSize;
				};
		}
	else
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				(*ppData)->pTracks[i][j].pCorrection = 0;
	if (C.InputData.SkipGrains)
	{
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.TCount[i]; j++)			
				(*ppData)->pTracks[i][j].pPoints = 0;
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.BTCount[i]; j++)			
				(*ppData)->pFlatTracks[i][j].pStrips = 0;
		}
	else
	{
		TVector *pG = (TVector *)p;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pPoints, GSize);	
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.TCount[i]; j++)
			{
				(*ppData)->pTracks[i][j].pPoints = pG;
				pG += (*ppData)->pTracks[i][j].PointsN;
				};
		p = (BYTE *)(void *)pG;
		FlatStrip *pF = (FlatStrip *)p;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pStrips, FSize);	
		for (i = 0; i < 2; i++)
			for (j = 0; j < Tks.Hdr.BTCount[i]; j++)
			{
				(*ppData)->pFlatTracks[i][j].pStrips = pF;
				pF += (*ppData)->pFlatTracks[i][j].StripsN;
				};
		};
	UINT RetCode = IODATA_EXT_ERROR_OK;
	if (Tks.Hdr.CodingMode == BULK_COMPRESSION_GRAINS_DIFF)
	{	try
		{
			if (!DecompressGrains_DifferentialMethod((*ppData), pCompressedBlock)) 
				throw IODATA_EXT_ERROR_DECOMPRESS + BULK_DATA_BYTES + BULK_GRAIN_SECTION;
			}
		catch (unsigned &err)
		{
			RetCode = err;
			}
		catch (...)
		{
			RetCode = IODATA_EXT_ERROR_UNKNOWN;
			};
		};
	if (pFields) free(pFields);
	if (pCorrData) free(pCorrData);
	if (pFlatTracks[0]) free(pFlatTracks[0]);
	if (pFlatTracks[1]) free(pFlatTracks[1]);
	if (pTracks[0]) free(pTracks[0]);
	if (pTracks[1]) free(pTracks[1]);
	if (pPoints) free(pPoints);
	if (pStrips) free(pStrips);
	if (pCompressedBlock) free(pCompressedBlock);
	F.Close(); //CloseHandle(HFile);
	return RetCode;
}

UINT CSySalDataIO::ReadBulkClustersDump(BufferedFile &F, IO_Header Hdr, IO_BULK_ClustersDump **ppData)
{
	IO_BULK_ClustersDump Cls;
	Cls.Hdr.Type = Hdr;
	DWORD br;
	BULK_ClustersPlaneInfo *pPlanes = 0;
	BYTE *p;
	int CSize = 0;
	int PSize;
	int i;
	try
	{
		if (Hdr.InfoType == (BULK_HEADER_BYTES | BULK_CLUSTERPRED_SECTION))
		{
			if (Cls.Hdr.Type.HeaderFormat == BULK_OLD_HEADER_TYPE)
			{
				TRY_READ_B(HFile, br, &Cls.Hdr.ID.Part[3], (sizeof(Cls.Hdr) - sizeof(Cls.Hdr.Type) - 
					3 * sizeof(Cls.Hdr.ID.Part[0])), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				Cls.Hdr.ID.Part[0] = Cls.Hdr.ID.Part[1] = Cls.Hdr.ID.Part[2] = 0;
				}
			else
			{
				TRY_READ_B(HFile, br, &Cls.Hdr.ID, (sizeof(Cls.Hdr) - sizeof(Cls.Hdr.Type)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				};
			}
		else
		{
			if (Cls.Hdr.Type.HeaderFormat == BULK_OLD_HEADER_TYPE)
			{
				TRY_READ_B(HFile, br, &Cls.Hdr.ID.Part[3], (sizeof(Cls.Hdr) - sizeof(Cls.Hdr.Type) - 
					3 * sizeof(Cls.Hdr.ID.Part[0]) - 2 * sizeof(float)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				Cls.Hdr.ID.Part[0] = Cls.Hdr.ID.Part[1] = Cls.Hdr.ID.Part[2] = 0;
				}
			else
			{
				TRY_READ_B(HFile, br, &Cls.Hdr.ID, (sizeof(Cls.Hdr) - sizeof(Cls.Hdr.Type) - 2 * sizeof(float)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				};
			Cls.Hdr.YCenter = Cls.Hdr.YPos;
			Cls.Hdr.ZCenter = Cls.Hdr.ZPos;
			}
		Cls.Hdr.Type.HeaderFormat = BULK_HEADER_TYPE;
		pPlanes = (BULK_ClustersPlaneInfo *)malloc(sizeof(BULK_ClustersPlaneInfo) * Cls.Hdr.Frames);
		for (i = 0; i < Cls.Hdr.Frames; i++) pPlanes[i].pClusters = 0;
		for (i = 0; i < Cls.Hdr.Frames; i++)
		{
			TRY_READ_B(HFile, br, &pPlanes[i].Z, sizeof(pPlanes[i].Z), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_CLUSTER_SECTION);
			TRY_READ_B(HFile, br, &pPlanes[i].Count, sizeof(pPlanes[i].Count), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_CLUSTER_SECTION);
			PSize = sizeof(Cluster) * pPlanes[i].Count;
			CSize += QWORD_ALIGN(PSize);
			pPlanes[i].pClusters = (Cluster *)malloc(PSize);
			TRY_READ_B(HFile, br, pPlanes[i].pClusters, PSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_CLUSTER_SECTION);
			};
		}
	catch (unsigned err)
	{
		if (pPlanes)
			for (i = 0; i < Cls.Hdr.Frames; i++)
				if (pPlanes[i].pClusters) free(pPlanes[i].pClusters);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};
	p = (BYTE *)(void *)(*ppData = (IO_BULK_ClustersDump *)CoTaskMemAlloc(CSize + QWORD_ALIGN(sizeof(Cls)) + QWORD_ALIGN((sizeof(BULK_ClustersPlaneInfo) * Cls.Hdr.Frames))));
	Cls.pPlanes = (BULK_ClustersPlaneInfo *)(p + QWORD_ALIGN(sizeof(Cls)));
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Cls, sizeof(Cls));
	BULK_ClustersPlaneInfo *pTemp = (BULK_ClustersPlaneInfo *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pPlanes, (sizeof(BULK_ClustersPlaneInfo) * Cls.Hdr.Frames));
	if (pPlanes)
	{
		for (i = 0; i < Cls.Hdr.Frames; i++)
		{
			pTemp[i].pClusters = (Cluster *)p;
			if (pPlanes[i].pClusters) 
			{
				QWORD_ALIGNED_COPY_AND_DISPLACE(p, pPlanes[i].pClusters, (sizeof(Cluster) * pPlanes[i].Count));
				free(pPlanes[i].pClusters);
				};
			};
		free(pPlanes);
		};	
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;	
}

UINT CSySalDataIO::ReadBulkPhotoSeq(BufferedFile &F, IO_Header Hdr, IO_BULK_PhotoSeq **ppData)
{
	IO_BULK_PhotoSeq Img;
	Img.Hdr.Type = Hdr;
	DWORD br;
	BULK_PhotoInfo *pFrames = 0;
	BYTE *pImages = 0;
	BYTE *p, *pData;
	int ImageSize;
	try
	{
		if (Hdr.InfoType == (BULK_HEADER_BYTES | BULK_IMAGEPRED_SECTION))
		{
			if (Img.Hdr.Type.HeaderFormat == BULK_OLD_HEADER_TYPE)
			{
				TRY_READ_B(HFile, br, &Img.Hdr.ID.Part[3], (sizeof(Img.Hdr) - sizeof(Img.Hdr.Type) - 
					3 * sizeof(Img.Hdr.ID.Part[0])), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				Img.Hdr.ID.Part[0] = Img.Hdr.ID.Part[1] = Img.Hdr.ID.Part[2] = 0;
				}
			else
			{
				TRY_READ_B(HFile, br, &Img.Hdr.ID, (sizeof(Img.Hdr) - sizeof(Img.Hdr.Type)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				};
			}
		else
		{
			if (Img.Hdr.Type.HeaderFormat == BULK_OLD_HEADER_TYPE)
			{
				TRY_READ_B(HFile, br, &Img.Hdr.ID.Part[3], (sizeof(Img.Hdr) - sizeof(Img.Hdr.Type) - 
					3 * sizeof(Img.Hdr.ID.Part[0]) - 2 * sizeof(float)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				Img.Hdr.ID.Part[0] = Img.Hdr.ID.Part[1] = Img.Hdr.ID.Part[2] = 0;
				}
			else
			{
				TRY_READ_B(HFile, br, &Img.Hdr.ID, (sizeof(Img.Hdr) - sizeof(Img.Hdr.Type) - 2 * sizeof(float)), IODATA_EXT_ERROR_READ + BULK_HEADER_BYTES);
				};
			Img.Hdr.YCenter = Img.Hdr.YPos;
			Img.Hdr.ZCenter = Img.Hdr.ZPos;
			};
		Img.Hdr.Type.HeaderFormat = BULK_HEADER_TYPE;
		ImageSize = Img.Hdr.Width * Img.Hdr.Height;
		p = pData = (BYTE *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(Img)) + QWORD_ALIGN(sizeof(BULK_PhotoInfo) * Img.Hdr.Frames) + QWORD_ALIGN(ImageSize) * Img.Hdr.Frames);
		Img.pImages = (BULK_PhotoInfo *)(p + QWORD_ALIGN(sizeof(Img)));
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Img, sizeof(Img));
		pFrames = (BULK_PhotoInfo *)p;
		pImages = p + QWORD_ALIGN(sizeof(BULK_PhotoInfo) * Img.Hdr.Frames);
		int i;
		for (i = 0; i < Img.Hdr.Frames; i++)
		{
			TRY_READ_B(HFile, br, &pFrames[i].Z, sizeof(pFrames[i].Z), IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_IMAGE_SECTION);
			pFrames[i].pImage = pImages;
			TRY_READ_B(HFile, br, pImages, ImageSize, IODATA_EXT_ERROR_READ + BULK_DATA_BYTES + BULK_IMAGE_SECTION);
			pImages += QWORD_ALIGN(ImageSize);
			};
		}
	catch (unsigned err)
	{
		CoTaskMemFree(pData);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};	
	*ppData = (IO_BULK_PhotoSeq *)pData;
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadCSSSTracks(BufferedFile &F, IO_Header Hdr, IO_CS_SS_Tracks **ppData)
{
	IO_CS_SS_Tracks Tks;
	Tks.Hdr.Type = Hdr;
	DWORD br;
	BYTE *pFields = 0;
	Track *pTracks[2] = {0, 0};
	CS_SS_LinkedTrack *pLinked = 0;
	int i, j;
	try
	{
		if (Tks.Hdr.Type.HeaderFormat == CS_SS_OLD_HEADER_TYPE)
		{
			TRY_READ_B(HFile, br, &Tks.Hdr.ID.Part[3], (sizeof(Tks.Hdr) - sizeof(Tks.Hdr.Type) - 
				3 * sizeof(Tks.Hdr.ID.Part[0])), IODATA_EXT_ERROR_READ + CS_SS_HEADER_BYTES);
			Tks.Hdr.ID.Part[0] = Tks.Hdr.ID.Part[1] = Tks.Hdr.ID.Part[2] = 0;
			}
		else
		{
			TRY_READ_B(HFile, br, &Tks.Hdr.ID, (sizeof(Tks.Hdr) - sizeof(Tks.Hdr.Type)), IODATA_EXT_ERROR_READ + CS_SS_HEADER_BYTES);
			if (Tks.Hdr.Type.HeaderFormat != OPERA_HEADER_TYPE && Tks.Hdr.Type.HeaderFormat != OPERA_HEADER_TYPE_WITH_INDEX)
			{
				Tks.Hdr.YSlope = Tks.Hdr.ZSlope = 0.0f;
				Tks.Hdr.GlobalSlopeTolerance = Tks.Hdr.SideSlopeTolerance = Tks.Hdr.GoodSlopeTolerance = 0.0f;
				};
			};
		pFields = (BYTE *)malloc(2 * Tks.Hdr.FCount);
		TRY_READ_B(HFile, br, pFields, (2 * Tks.Hdr.FCount), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_FIELD_SECTION);
		pTracks[0] = (Track *)malloc(Tks.Hdr.TCount[0] * sizeof(Track));
		pTracks[1] = (Track *)malloc(Tks.Hdr.TCount[1] * sizeof(Track));
		pLinked = (CS_SS_LinkedTrack *)malloc(Tks.Hdr.LCount * sizeof(CS_SS_LinkedTrack));
		if (Hdr.HeaderFormat == OPERA_HEADER_TYPE || Hdr.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX)
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					static int dummy;
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].PointsN, sizeof(pTracks[i][j].PointsN), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					pTracks[i][j].Valid = true;
					pTracks[i][j].InterceptErrors.X = pTracks[i][j].InterceptErrors.Y = pTracks[i][j].InterceptErrors.Z = 0.f;
					pTracks[i][j].SlopeErrors.X = pTracks[i][j].SlopeErrors.Y = pTracks[i][j].SlopeErrors.Z = 0.f;
					}
		else if (Hdr.HeaderFormat == CS_SS_HEADER_TYPE)
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].PointsN, sizeof(pTracks[i][j].PointsN), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].InterceptErrors, sizeof(pTracks[i][j].InterceptErrors), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].SlopeErrors, sizeof(pTracks[i][j].SlopeErrors), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					pTracks[i][j].Valid = true;
					}
		else
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].PointsN, sizeof(pTracks[i][j].PointsN), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION); // A way to skip 3 0 vectors...
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					pTracks[i][j].InterceptErrors.X = pTracks[i][j].InterceptErrors.Y = pTracks[i][j].InterceptErrors.Z = 0.f;
					pTracks[i][j].SlopeErrors.X = pTracks[i][j].SlopeErrors.Y = pTracks[i][j].SlopeErrors.Z = 0.f;
					pTracks[i][j].Valid = true;
					};
		if (Hdr.HeaderFormat == OPERA_HEADER_TYPE || Hdr.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX)
			for (i = 0; i < Tks.Hdr.LCount; i++)
			{
				static int dummy;
				TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].PointsN, sizeof(pLinked[i].PointsN), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Intercept, sizeof(pLinked[i].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Slope, sizeof(pLinked[i].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Sigma, sizeof(pLinked[i].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				pLinked[i].InterceptErrors.X = pLinked[i].InterceptErrors.Y = pLinked[i].InterceptErrors.Z = 0;
				pLinked[i].SlopeErrors.X = pLinked[i].SlopeErrors.Y = pLinked[i].SlopeErrors.Z = 0;
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[0], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[1], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				}
		else if (Hdr.HeaderFormat == CS_SS_HEADER_TYPE)
			for (i = 0; i < Tks.Hdr.LCount; i++)
			{
				TRY_READ_B(HFile, br, &pLinked[i].PointsN, sizeof(pLinked[i].PointsN), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Intercept, sizeof(pLinked[i].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Slope, sizeof(pLinked[i].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Sigma, sizeof(pLinked[i].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].InterceptErrors, sizeof(pLinked[i].InterceptErrors), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].SlopeErrors, sizeof(pLinked[i].SlopeErrors), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);			
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[0], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[1], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				}
		else
			for (i = 0; i < Tks.Hdr.LCount; i++)
			{
				TRY_READ_B(HFile, br, &pLinked[i].PointsN, sizeof(pLinked[i].PointsN), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Intercept, sizeof(pLinked[i].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Slope, sizeof(pLinked[i].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Sigma, sizeof(pLinked[i].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				pLinked[i].InterceptErrors.X = pLinked[i].InterceptErrors.Y = pLinked[i].InterceptErrors.Z = 0;
				pLinked[i].SlopeErrors.X = pLinked[i].SlopeErrors.Y = pLinked[i].SlopeErrors.Z = 0;
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[0], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[1], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				};
		}
	catch (unsigned err)
	{
		if (pFields) free(pFields);
		if (pTracks[0]) free(pTracks[0]);
		if (pTracks[1]) free(pTracks[1]);
		if (pLinked) free(pLinked);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};
	Tks.Hdr.Type.HeaderFormat = CS_SS_HEADER_TYPE;
	*ppData = (IO_CS_SS_Tracks *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(IO_Data)) + QWORD_ALIGN((2 * Tks.Hdr.FCount)) + 
		QWORD_ALIGN(((Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1]) * sizeof(Track))) + 
		QWORD_ALIGN((Tks.Hdr.LCount * sizeof(CS_SS_LinkedTrack))));
	BYTE *p = (BYTE *)*ppData;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Tks, sizeof(Tks));
	(*ppData)->FieldHistory = p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pFields, 2 * Tks.Hdr.FCount);
	(*ppData)->pTracks[0] = (Track *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks[0], (Tks.Hdr.TCount[0] * sizeof(Track)));
	(*ppData)->pTracks[1] = (Track *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks[1], (Tks.Hdr.TCount[1] * sizeof(Track)));
	(*ppData)->pLinked = (CS_SS_LinkedTrack *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pLinked, (Tks.Hdr.LCount * sizeof(CS_SS_LinkedTrack)));
	for (i = 0; i < Tks.Hdr.LCount; i++)
	{
		(*ppData)->pLinked[i].pTracks[0] = (int)(void *)(*ppData)->pLinked[i].pTracks[0] + (*ppData)->pTracks[0];
		(*ppData)->pLinked[i].pTracks[1] = (int)(void *)(*ppData)->pLinked[i].pTracks[1] + (*ppData)->pTracks[1];
		};			
	if (pTracks[0]) free(pTracks[0]);
	if (pTracks[1]) free(pTracks[1]);
	if (pLinked) free(pLinked);
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadOPERATracks(BufferedFile &F, IO_Header Hdr, IO_OPERA_Tracks **ppData)
{
	IO_OPERA_Tracks Tks;
	Tks.Hdr.Type = Hdr;
	DWORD br;
	BYTE *pFields = 0;
	Track2 *pTracks[2] = {0, 0};
	OPERA_LinkedTrack *pLinked = 0;
	OPERA_TrackIndexEntry *pTrackIndex[2] = {0, 0};
	int i, j;
	try
	{
		if (Tks.Hdr.Type.HeaderFormat == CS_SS_OLD_HEADER_TYPE)
		{
			TRY_READ_B(HFile, br, &Tks.Hdr.ID.Part[3], (sizeof(Tks.Hdr) - sizeof(Tks.Hdr.Type) - 
				3 * sizeof(Tks.Hdr.ID.Part[0])), IODATA_EXT_ERROR_READ + CS_SS_HEADER_BYTES);
			Tks.Hdr.ID.Part[0] = Tks.Hdr.ID.Part[1] = Tks.Hdr.ID.Part[2] = 0;
			Tks.Hdr.MinX = Tks.Hdr.MaxX = Tks.Hdr.XPos;
			Tks.Hdr.MinY = Tks.Hdr.MaxY = Tks.Hdr.YPos;
			}
		else 
		{
			TRY_READ_B(HFile, br, &Tks.Hdr.ID, (sizeof(Tks.Hdr) - sizeof(Tks.Hdr.Type)), IODATA_EXT_ERROR_READ + CS_SS_HEADER_BYTES);
			if (Tks.Hdr.Type.HeaderFormat != OPERA_HEADER_TYPE && Tks.Hdr.Type.HeaderFormat != OPERA_HEADER_TYPE_WITH_INDEX)
			{
				Tks.Hdr.MinX = Tks.Hdr.MaxX = Tks.Hdr.XPos;
				Tks.Hdr.MinY = Tks.Hdr.MaxY = Tks.Hdr.YPos;
				};
			}
		pFields = (BYTE *)malloc(2 * Tks.Hdr.FCount);
		TRY_READ_B(HFile, br, pFields, (2 * Tks.Hdr.FCount), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_FIELD_SECTION);
		pTracks[0] = (Track2 *)malloc(Tks.Hdr.TCount[0] * sizeof(Track2));
		pTracks[1] = (Track2 *)malloc(Tks.Hdr.TCount[1] * sizeof(Track2));
		pTrackIndex[0] = (OPERA_TrackIndexEntry *)malloc(Tks.Hdr.TCount[0] * sizeof(OPERA_TrackIndexEntry));
		pTrackIndex[1] = (OPERA_TrackIndexEntry *)malloc(Tks.Hdr.TCount[1] * sizeof(OPERA_TrackIndexEntry));
		pLinked = (OPERA_LinkedTrack *)malloc(Tks.Hdr.LCount * sizeof(OPERA_LinkedTrack));
		if (Hdr.HeaderFormat == OPERA_HEADER_TYPE || Hdr.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX)
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].AreaSum, sizeof(pTracks[i][j].AreaSum), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Grains, sizeof(pTracks[i][j].Grains), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					pTracks[i][j].Valid = true;
					pTracks[i][j].pGrains = 0;
					pTracks[i][j].pCorrection = 0;
					}
		else if (Hdr.HeaderFormat == CS_SS_HEADER_TYPE)
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					static TVector dummy;
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Grains, sizeof(pTracks[i][j].Grains), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					pTracks[i][j].Valid = true;
					pTracks[i][j].AreaSum = 0;
					pTracks[i][j].pGrains = 0;
					pTracks[i][j].pCorrection = 0;
					}
		else
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
				{
					static TVector dummy;
					TRY_READ_B(HFile, br, &pTracks[i][j].Field, sizeof(pTracks[i][j].Field), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Grains, sizeof(pTracks[i][j].Grains), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION); // A way to skip 3 0 vectors...
					TRY_READ_B(HFile, br, &pTracks[i][j].Intercept, sizeof(pTracks[i][j].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Slope, sizeof(pTracks[i][j].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].Sigma, sizeof(pTracks[i][j].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].FirstZ, sizeof(pTracks[i][j].FirstZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					TRY_READ_B(HFile, br, &pTracks[i][j].LastZ, sizeof(pTracks[i][j].LastZ), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_TRACK_SECTION);
					pTracks[i][j].Valid = true;
					pTracks[i][j].AreaSum = 0;
					pTracks[i][j].pGrains = 0;
					pTracks[i][j].pCorrection = 0;
					};
		if (Hdr.HeaderFormat == OPERA_HEADER_TYPE || Hdr.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX)
			for (i = 0; i < Tks.Hdr.LCount; i++)
			{
				TRY_READ_B(HFile, br, &pLinked[i].AreaSum, sizeof(pLinked[i].AreaSum), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Grains, sizeof(pLinked[i].Grains), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Intercept, sizeof(pLinked[i].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Slope, sizeof(pLinked[i].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Sigma, sizeof(pLinked[i].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[0], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[1], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				}
		else if (Hdr.HeaderFormat == CS_SS_HEADER_TYPE)
			for (i = 0; i < Tks.Hdr.LCount; i++)
			{
				static TVector dummy;
				TRY_READ_B(HFile, br, &pLinked[i].Grains, sizeof(pLinked[i].Grains), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Intercept, sizeof(pLinked[i].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Slope, sizeof(pLinked[i].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Sigma, sizeof(pLinked[i].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);			
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[0], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[1], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				pLinked[i].AreaSum = 0;
				}
		else
			for (i = 0; i < Tks.Hdr.LCount; i++)
			{
				TRY_READ_B(HFile, br, &pLinked[i].Grains, sizeof(pLinked[i].Grains), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Intercept, sizeof(pLinked[i].Intercept), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Slope, sizeof(pLinked[i].Slope), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].Sigma, sizeof(pLinked[i].Sigma), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[0], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				TRY_READ_B(HFile, br, &pLinked[i].pTracks[1], sizeof(int), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + CS_SS_LTRACK_SECTION);
				pLinked[i].AreaSum = 0;
				};
		if (Hdr.HeaderFormat == OPERA_HEADER_TYPE_WITH_INDEX)
		{
			TRY_READ_B(HFile, br, pTrackIndex[0], (sizeof(OPERA_TrackIndexEntry) * Tks.Hdr.TCount[0]), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + OPERA_INDEX_SECTION);
			TRY_READ_B(HFile, br, pTrackIndex[1], (sizeof(OPERA_TrackIndexEntry) * Tks.Hdr.TCount[1]), IODATA_EXT_ERROR_READ + CS_SS_DATA_BYTES + OPERA_INDEX_SECTION);
			}
		else
			for (i = 0; i < 2; i++)
				for (j = 0; j < Tks.Hdr.TCount[i]; j++)
					pTrackIndex[i][j].Fragment = pTrackIndex[i][j].View = pTrackIndex[i][j].Track = 0;
		}
	catch (unsigned err)
	{
		if (pFields) free(pFields);
		if (pTrackIndex[0]) free(pTrackIndex[0]);
		if (pTrackIndex[1]) free(pTrackIndex[1]);
		if (pTracks[0]) free(pTracks[0]);
		if (pTracks[1]) free(pTracks[1]);
		if (pLinked) free(pLinked);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};
	Tks.Hdr.Type.HeaderFormat = Hdr.HeaderFormat;
	*ppData = (IO_OPERA_Tracks *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(IO_Data)) + QWORD_ALIGN((2 * Tks.Hdr.FCount)) + 
		QWORD_ALIGN(((Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1]) * sizeof(Track2))) + 
		QWORD_ALIGN(((Tks.Hdr.TCount[0] + Tks.Hdr.TCount[1]) * sizeof(OPERA_TrackIndexEntry))) + 
		QWORD_ALIGN((Tks.Hdr.LCount * sizeof(OPERA_LinkedTrack))));
	BYTE *p = (BYTE *)*ppData;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Tks, sizeof(Tks));
	(*ppData)->FieldHistory = p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pFields, 2 * Tks.Hdr.FCount);
	(*ppData)->pTracks[0] = (Track2 *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks[0], (Tks.Hdr.TCount[0] * sizeof(Track2)));
	(*ppData)->pTracks[1] = (Track2 *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks[1], (Tks.Hdr.TCount[1] * sizeof(Track2)));
	(*ppData)->pLinked = (OPERA_LinkedTrack *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pLinked, (Tks.Hdr.LCount * sizeof(OPERA_LinkedTrack)));
	for (i = 0; i < Tks.Hdr.LCount; i++)
	{
		(*ppData)->pLinked[i].pTracks[0] = (int)(void *)(*ppData)->pLinked[i].pTracks[0] + (*ppData)->pTracks[0];
		(*ppData)->pLinked[i].pTracks[1] = (int)(void *)(*ppData)->pLinked[i].pTracks[1] + (*ppData)->pTracks[1];
		};			
	(*ppData)->pIndexEntries[0] = (OPERA_TrackIndexEntry *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTrackIndex[0], (Tks.Hdr.TCount[0] * sizeof(OPERA_TrackIndexEntry)));
	(*ppData)->pIndexEntries[1] = (OPERA_TrackIndexEntry *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTrackIndex[1], (Tks.Hdr.TCount[1] * sizeof(OPERA_TrackIndexEntry)));
	if (pTrackIndex[0]) free(pTrackIndex[0]);
	if (pTrackIndex[1]) free(pTrackIndex[1]);
	if (pTracks[0]) free(pTracks[0]);
	if (pTracks[1]) free(pTracks[1]);
	if (pLinked) free(pLinked);
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadBulkCSSSBatch(BufferedFile &F, IO_Header Hdr, IO_BS_Batch **ppData)
{
	IO_BS_Batch Batch;
	Batch.Hdr = Hdr;
	DWORD br;
	IO_BS_Batch *pData = 0;
	BYTE *p;
	try
	{
		TRY_READ_B(HFile, br, &Batch.CountOfTracks, sizeof(Batch.CountOfTracks), IODATA_EXT_ERROR_READ + BS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Batch.CountOfClusterDumps, sizeof(Batch.CountOfClusterDumps), IODATA_EXT_ERROR_READ + BS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Batch.CountOfPhotoSequences, sizeof(Batch.CountOfPhotoSequences), IODATA_EXT_ERROR_READ + BS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Batch.Reserved, sizeof(Batch.Reserved), IODATA_EXT_ERROR_READ + BS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Batch.pReserved, sizeof(Batch.pReserved), IODATA_EXT_ERROR_READ + BS_HEADER_BYTES);
		pData = (IO_BS_Batch *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(IO_BS_Batch)) + QWORD_ALIGN(sizeof(BS_Track) * Batch.CountOfTracks) + 
			QWORD_ALIGN(sizeof(BS_ClusterDump) * Batch.CountOfClusterDumps) + QWORD_ALIGN(sizeof(BS_PhotoSequence) * Batch.CountOfPhotoSequences));
		if (!pData) throw IODATA_EXT_ERROR_READ + BS_HEADER_BYTES;
		p = (BYTE *)pData;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Batch, sizeof(Batch));
		pData->pTracks = (BS_Track *)p;
		TRY_READ_B(HFile, br, p, (sizeof(BS_Track) * Batch.CountOfTracks), IODATA_EXT_ERROR_READ + BS_DATA_BYTES + BS_TRACK_SECTION);
		p += QWORD_ALIGN(sizeof(BS_Track) * Batch.CountOfTracks);
		pData->pClusterDumps = (BS_ClusterDump *)p;
		TRY_READ_B(HFile, br, p, (sizeof(BS_ClusterDump) * Batch.CountOfClusterDumps), IODATA_EXT_ERROR_READ + BS_DATA_BYTES + BS_CLUSTERPRED_SECTION);
		p += QWORD_ALIGN(sizeof(BS_ClusterDump) * Batch.CountOfClusterDumps);
		pData->pPhotoSequences = (BS_PhotoSequence *)p;
		TRY_READ_B(HFile, br, p, (sizeof(BS_PhotoSequence) * Batch.CountOfPhotoSequences), IODATA_EXT_ERROR_READ + BS_DATA_BYTES + BS_IMAGEPRED_SECTION);
		}
	catch (unsigned err)
	{
		if (pData) CoTaskMemFree(pData);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};	
	*ppData = (IO_BS_Batch *)pData;
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}


UINT CSySalDataIO::ReadTSReconstruction(BufferedFile &F, IO_Header Hdr, IO_TS_Reconstruction **ppData)
{
	IO_TS_Reconstruction TSRec;
	TSRec.Hdr.Type = Hdr;
	DWORD br;
	TS_Track *pTracks = 0;
	TS_Vertex *pVertices = 0;
	TS_RawTrack *pRawTracks = 0;
	TS_LayerInfo *pLayers = 0;
	TS_RawTrack ***pRawTracksIndex = 0;
	int MaxVTSTracksCount;
	t_TS_Vertex::t_VertexTSTrack *pVTSTracks = (t_TS_Vertex::t_VertexTSTrack *)malloc(sizeof(t_TS_Vertex::t_VertexTSTrack) * (MaxVTSTracksCount = READ_VERTEXTSTRACK_ALLOC_BLOCK));
	int i, s, j, k, h, segnum;
	int RawTracksSize = 0, SegmentsSize = 0, VertexTSTracksSize = 0;
	try
	{
		TRY_READ_B(HFile, br, &TSRec.Hdr.ID, (sizeof(TSRec.Hdr) - sizeof(TSRec.Hdr.Type)), IODATA_EXT_ERROR_READ + TS_HEADER_BYTES);
		pLayers = (TS_LayerInfo *)malloc(sizeof(TS_LayerInfo) * TSRec.Hdr.SheetsCount);
		pRawTracksIndex = (TS_RawTrack ***)malloc(sizeof(TS_RawTrack **) * 2 * TSRec.Hdr.SheetsCount);
		for (i = 0; i < TSRec.Hdr.SheetsCount; i++)
		{			
			pRawTracksIndex[2 * i] = pRawTracksIndex[2 * i + 1] = 0;
			pLayers[i].pRawTracks[0] = pLayers[i].pRawTracks[1] = 0;
			pLayers[i].pSegments[0] = pLayers[i].pSegments[1] = 0;
			};
		for (i = 0; i < TSRec.Hdr.SheetsCount; i++)
		{
			if (Hdr.HeaderFormat == TS_OLD_HEADER_TYPE)
			{
				TRY_READ_B(HFile, br, &pLayers[i].SheetId, sizeof(pLayers[i].SheetId), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].RelevantZs, sizeof(pLayers[i].RelevantZs), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].RefDepth, sizeof(pLayers[i].RefDepth), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].GlobalSideAlign[0], sizeof(pLayers[i].GlobalSideAlign[0]), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].GlobalSideAlign[1], sizeof(pLayers[i].GlobalSideAlign[1]), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].LocalSideAlign[0], sizeof(pLayers[i].LocalSideAlign[0]), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].LocalSideAlign[1], sizeof(pLayers[i].LocalSideAlign[1]), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].RTCount, sizeof(pLayers[i].RTCount), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].SCount, sizeof(pLayers[i].SCount), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				TRY_READ_B(HFile, br, &pLayers[i].Reserved, sizeof(pLayers[i].Reserved), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);
				int k;
				for (k = 2; k < 4; k++)
				{
					pLayers[i].LocalSideAlign[k].PosAlign.M[0][0] = pLayers[i].LocalSideAlign[k].PosAlign.M[0][1] = 
						pLayers[i].LocalSideAlign[k].PosAlign.M[1][0] = pLayers[i].LocalSideAlign[k].PosAlign.M[1][1] = 
						pLayers[i].LocalSideAlign[k].PosAlign.D[0] = pLayers[i].LocalSideAlign[k].PosAlign.D[1] = 
						pLayers[i].LocalSideAlign[k].PosAlign.D[2] = pLayers[i].LocalSideAlign[k].PosAlign.Reserved[0] = 
						pLayers[i].LocalSideAlign[k].PosAlign.Reserved[1] = pLayers[i].LocalSideAlign[k].PosAlign.Reserved[2] = 
						pLayers[i].LocalSideAlign[k].PosAlign.Reserved[3] = 0.;
					pLayers[i].LocalSideAlign[k].SlopeAlign.DShrink[0] = pLayers[i].LocalSideAlign[k].SlopeAlign.DShrink[1] = 
						pLayers[i].LocalSideAlign[k].SlopeAlign.DSlope[0] = pLayers[i].LocalSideAlign[k].SlopeAlign.DSlope[1] = 
						pLayers[i].LocalSideAlign[k].SlopeAlign.Reserved[0] = pLayers[i].LocalSideAlign[k].SlopeAlign.Reserved[1] = 
						pLayers[i].LocalSideAlign[k].SlopeAlign.Reserved[2] = pLayers[i].LocalSideAlign[k].SlopeAlign.Reserved[3] = 0.;
					};
				}
			else TRY_READ_B(HFile, br, &pLayers[i], (sizeof(pLayers[i]) - sizeof(pLayers[i].pRawTracks) - sizeof(pLayers[i].pSegments) - sizeof(pLayers[i].pReserved)), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_LAYER_SECTION);			
			RawTracksSize += (pLayers[i].RTCount[0] + pLayers[i].RTCount[1]);
			SegmentsSize += (pLayers[i].SCount[0] + pLayers[i].SCount[1]);
			pRawTracksIndex[2 * i] = (TS_RawTrack **)malloc(sizeof(TS_RawTrack *) * pLayers[i].SCount[0] * TSRec.Hdr.MaxTracksInSegment);
			pRawTracksIndex[2 * i + 1] = (TS_RawTrack **)malloc(sizeof(TS_RawTrack *) * pLayers[i].SCount[1] * TSRec.Hdr.MaxTracksInSegment);			
			for (s = 0; s < 2; s++)
			{
				pLayers[i].pRawTracks[s] = (TS_RawTrack *)malloc(pLayers[i].RTCount[s] * sizeof(TS_RawTrack));				
				TRY_READ_B(HFile, br, pLayers[i].pRawTracks[s], (sizeof(TS_RawTrack) * pLayers[i].RTCount[s]), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);				
				pLayers[i].pSegments[s] = (TS_Segment *)malloc(pLayers[i].SCount[s] * sizeof(TS_Segment));
				for (j = 0; j < pLayers[i].SCount[s]; j++)
				{
					pLayers[i].pSegments[s][j].pTracks = pRawTracksIndex[2 * i + s] + j * TSRec.Hdr.MaxTracksInSegment;
					for (k = 0; k < TSRec.Hdr.MaxTracksInSegment; k++)
						TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].pTracks[k], sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].pBottomLink, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].pTopLink, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);

					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].PointsN, sizeof(pLayers[i].pSegments[s][j].PointsN), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].Intercept, sizeof(pLayers[i].pSegments[s][j].Intercept), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].Slope, sizeof(pLayers[i].pSegments[s][j].Slope), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].Sigma, sizeof(pLayers[i].pSegments[s][j].Sigma), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].FirstZ, sizeof(pLayers[i].pSegments[s][j].FirstZ), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);
					TRY_READ_B(HFile, br, &pLayers[i].pSegments[s][j].LastZ, sizeof(pLayers[i].pSegments[s][j].LastZ), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION);					
					};
				};
			};
		
		pTracks = (TS_Track *)malloc(sizeof(TS_Track) * TSRec.Hdr.TSTracksCount);
		for (i = 0; i < TSRec.Hdr.TSTracksCount; i++)
		{	
			TRY_READ_B(HFile, br, &pTracks[i].Count, sizeof(pTracks[i].Count), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);

			TRY_READ_B(HFile, br, &pTracks[i].pBottomEnd, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].pTopEnd, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);

			TRY_READ_B(HFile, br, &pTracks[i].pBottomVertex, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].pTopVertex, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);

			TRY_READ_B(HFile, br, &pTracks[i].Flags, sizeof(pTracks[i].Flags), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].Intercept, sizeof(pTracks[i].Intercept), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].Slope, sizeof(pTracks[i].Slope), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].Sigma, sizeof(pTracks[i].Sigma), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].FirstZ, sizeof(pTracks[i].FirstZ), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].LastZ, sizeof(pTracks[i].LastZ), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			TRY_READ_B(HFile, br, &pTracks[i].Reserved, sizeof(pTracks[i].Reserved), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION);
			};

		pVertices = (TS_Vertex *)malloc(sizeof(TS_Vertex) * TSRec.Hdr.VerticesCount);
		for (i = 0; i < TSRec.Hdr.VerticesCount; i++)
		{
			TRY_READ_B(HFile, br, &pVertices[i].Count, sizeof(pVertices[i].Count), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
			TRY_READ_B(HFile, br, &pVertices[i].Pos, sizeof(pVertices[i].Pos), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
			TRY_READ_B(HFile, br, &pVertices[i].AvgDistance, sizeof(pVertices[i].AvgDistance), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
			if ((VertexTSTracksSize + pVertices[i].Count) >= MaxVTSTracksCount)
				pVTSTracks = (t_TS_Vertex::t_VertexTSTrack *)realloc(pVTSTracks, sizeof(t_TS_Vertex::t_VertexTSTrack) * (MaxVTSTracksCount += READ_VERTEXTSTRACK_ALLOC_BLOCK));			
			for (j = 0; j < pVertices[i].Count; j++)
			{
				TRY_READ_B(HFile, br, &pVTSTracks[VertexTSTracksSize].pTSTrack, sizeof(int), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
				TRY_READ_B(HFile, br, &pVTSTracks[VertexTSTracksSize].IsDownstream, sizeof(pVTSTracks[VertexTSTracksSize].IsDownstream), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
				TRY_READ_B(HFile, br, &pVTSTracks[VertexTSTracksSize].Reserved, sizeof(pVTSTracks[VertexTSTracksSize].Reserved), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
				VertexTSTracksSize++;
				};
			TRY_READ_B(HFile, br, &pVertices[i].Reserved, sizeof(pVertices[i].Reserved), IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_VERTEX_SECTION);
			};				
		}
	catch (unsigned err)
	{
		if (pRawTracksIndex) 
		{
			for (i = 0; i < (2 * TSRec.Hdr.SheetsCount); i++)
				if (pRawTracksIndex[i]) free(pRawTracksIndex[i]);			
			free(pRawTracksIndex);
			};
		if (pVTSTracks) free(pVTSTracks);
		if (pVertices) free(pVertices);
		if (pTracks) free(pTracks);
		if (pLayers) 
		{
			for (i = 0; i < TSRec.Hdr.SheetsCount; i++)
				for (s = 0; s < 2; s++)
				{
					if (pLayers[i].pRawTracks[s]) free(pLayers[i].pRawTracks[s]);
					if (pLayers[i].pSegments[s]) free(pLayers[i].pSegments[s]);
					};
			free(pLayers);
			};
		F.Close(); //CloseHandle(HFile);
		return err;		
		};
	TSRec.Hdr.Type.HeaderFormat = TS_HEADER_TYPE;
	*ppData = (IO_TS_Reconstruction *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(IO_Data)) + 
		QWORD_ALIGN((sizeof(TS_LayerInfo) * TSRec.Hdr.SheetsCount)) + 
		QWORD_ALIGN(sizeof(TS_RawTrack)) * RawTracksSize + 
		QWORD_ALIGN(sizeof(TS_Segment)) * SegmentsSize + 
		QWORD_ALIGN((sizeof(TS_RawTrack *) * SegmentsSize * TSRec.Hdr.MaxTracksInSegment)) + 
		QWORD_ALIGN((sizeof(TS_Track) * TSRec.Hdr.TSTracksCount)) + 
		QWORD_ALIGN((sizeof(TS_Vertex) * TSRec.Hdr.VerticesCount)) + 
		QWORD_ALIGN((sizeof(TS_Vertex::t_VertexTSTrack) * VertexTSTracksSize)));
		
	BYTE *p = (BYTE *)*ppData;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, &TSRec, sizeof(TSRec));
	TS_RawTrack **prawtrackindex = (TS_RawTrack **)(void *)p;
	p += QWORD_ALIGN((sizeof(TS_RawTrack *) * SegmentsSize * TSRec.Hdr.MaxTracksInSegment));
	TS_Vertex::t_VertexTSTrack *pvertextstracks = (TS_Vertex::t_VertexTSTrack *)(void *)p;
	p += QWORD_ALIGN((sizeof(TS_Vertex::t_VertexTSTrack) * VertexTSTracksSize));

	(*ppData)->pLayers = (TS_LayerInfo *)p;
	p += QWORD_ALIGN((sizeof(TS_LayerInfo) * TSRec.Hdr.SheetsCount));
	for (i = 0; i < TSRec.Hdr.SheetsCount; i++)
	{
		(*ppData)->pLayers[i] = pLayers[i];
		for (s = 0; s < 2; s++)
		{
			(*ppData)->pLayers[i].pRawTracks[s] = (TS_RawTrack *)(void *)p;
			QWORD_ALIGNED_COPY_AND_DISPLACE(p, pLayers[i].pRawTracks[s], (sizeof(TS_RawTrack) * pLayers[i].RTCount[s]));
			(*ppData)->pLayers[i].pSegments[s] = (TS_Segment *)(void *)p;
			QWORD_ALIGNED_COPY_AND_DISPLACE(p, pLayers[i].pSegments[s], (sizeof(TS_Segment) * pLayers[i].SCount[s]));
			for (j = 0; j < pLayers[i].SCount[s]; j++)
			{
				(*ppData)->pLayers[i].pSegments[s][j].pTracks = prawtrackindex;
				for (k = 0; k < TSRec.Hdr.MaxTracksInSegment; k++)
				{
					if (((int)(pLayers[i].pSegments[s][j].pTracks[k])) < 0)
						(*ppData)->pLayers[i].pSegments[s][j].pTracks[k] = 0;
					else
						(*ppData)->pLayers[i].pSegments[s][j].pTracks[k] = 
							(*ppData)->pLayers[i].pRawTracks[s] + ((int)(pLayers[i].pSegments[s][j].pTracks[k]));
					};
				prawtrackindex += TSRec.Hdr.MaxTracksInSegment;
				};
			};
		};
	for (i = 0; i < TSRec.Hdr.SheetsCount; i++)
		for (s = 0; s < 2; s++)
			for (j = 0; j < pLayers[i].SCount[s]; j++)
			{
				if (((int)(pLayers[i].pSegments[s][j].pBottomLink)) < 0)
					(*ppData)->pLayers[i].pSegments[s][j].pBottomLink = 0;
				else
				{
					segnum = (int)(pLayers[i].pSegments[s][j].pBottomLink);
					for (h = 0; (h < (2 * TSRec.Hdr.SheetsCount)) && (segnum >= pLayers[h / 2].SCount[h % 2]); h++)
						segnum -= pLayers[h / 2].SCount[h % 2];
					if (h >= (2 * TSRec.Hdr.SheetsCount))
					{
						CoTaskMemFree(*ppData);
						*ppData = 0;
						return IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION;
						};
					(*ppData)->pLayers[i].pSegments[s][j].pBottomLink = (*ppData)->pLayers[h / 2].pSegments[h % 2] + segnum;
					};

				if (((int)(pLayers[i].pSegments[s][j].pTopLink)) < 0)
					(*ppData)->pLayers[i].pSegments[s][j].pTopLink = 0;
				else
				{
					segnum = (int)(pLayers[i].pSegments[s][j].pTopLink);
					for (h = 0; (h < (2 * TSRec.Hdr.SheetsCount)) && (segnum >= pLayers[h / 2].SCount[h % 2]); h++)
						segnum -= pLayers[h / 2].SCount[h % 2];
					if (h >= (2 * TSRec.Hdr.SheetsCount))
					{
						CoTaskMemFree(*ppData);
						*ppData = 0;
						return IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_SEGMENT_SECTION;
						};
					(*ppData)->pLayers[i].pSegments[s][j].pTopLink = (*ppData)->pLayers[h / 2].pSegments[h % 2] + segnum;
					};
				};
	(*ppData)->pTracks = (TS_Track *)(void *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pTracks, (sizeof(TS_Track) * TSRec.Hdr.TSTracksCount));
	for (i = 0; i < TSRec.Hdr.TSTracksCount; i++)
	{
		segnum = (int)(pTracks[i].pBottomEnd);
		for (h = 0; (h < (2 * TSRec.Hdr.SheetsCount)) && (segnum >= pLayers[h / 2].SCount[h % 2]); h++)
		segnum -= pLayers[h / 2].SCount[h % 2];
		if (h >= (2 * TSRec.Hdr.SheetsCount))
		{
			CoTaskMemFree(*ppData);
			*ppData = 0;
			return IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION;
			};
		(*ppData)->pTracks[i].pBottomEnd = (*ppData)->pLayers[h / 2].pSegments[h % 2] + segnum;

		segnum = (int)(pTracks[i].pTopEnd);
		for (h = 0; (h < (2 * TSRec.Hdr.SheetsCount)) && (segnum >= pLayers[h / 2].SCount[h % 2]); h++)
		segnum -= pLayers[h / 2].SCount[h % 2];
		if (h >= (2 * TSRec.Hdr.SheetsCount))
		{
			CoTaskMemFree(*ppData);
			*ppData = 0;
			return IODATA_EXT_ERROR_READ + TS_DATA_BYTES + TS_TRACK_SECTION;
			};
		(*ppData)->pTracks[i].pTopEnd = (*ppData)->pLayers[h / 2].pSegments[h % 2] + segnum;
		};
	(*ppData)->pVertices = (TS_Vertex *)p;
	QWORD_ALIGNED_COPY_AND_DISPLACE(p, pVertices, (sizeof(TS_Vertex) * TSRec.Hdr.VerticesCount));
	for (i = 0; i < TSRec.Hdr.TSTracksCount; i++)
	{
		(*ppData)->pTracks[i].pBottomVertex = ((int)((*ppData)->pTracks[i].pBottomVertex) >= 0) ? ((int)((*ppData)->pTracks[i].pBottomVertex) + (*ppData)->pVertices) : 0;
		(*ppData)->pTracks[i].pTopVertex = ((int)((*ppData)->pTracks[i].pTopVertex) >= 0) ? ((int)((*ppData)->pTracks[i].pTopVertex) + (*ppData)->pVertices) : 0;
		};
	memcpy(pvertextstracks, pVTSTracks, sizeof(t_TS_Vertex::t_VertexTSTrack) * VertexTSTracksSize);
	for (i = 0; i < TSRec.Hdr.VerticesCount; i++)
	{
		(*ppData)->pVertices[i].pTSTracks = pvertextstracks;
		for (j = 0; j < pVertices[i].Count; j++)
		{			
			(*ppData)->pVertices[i].pTSTracks[j].pTSTrack = ((int)((*ppData)->pVertices[i].pTSTracks[j].pTSTrack)) + (*ppData)->pTracks;
			};
		pvertextstracks += (*ppData)->pVertices[i].Count;
		};

	if (pRawTracksIndex) 
	{
		for (i = 0; i < (2 * TSRec.Hdr.SheetsCount); i++)
			if (pRawTracksIndex[i]) free(pRawTracksIndex[i]);			
		free(pRawTracksIndex);
		};
	if (pVTSTracks) free(pVTSTracks);
	if (pVertices) free(pVertices);
	if (pTracks) free(pTracks);
	if (pLayers) 
	{
		for (i = 0; i < TSRec.Hdr.SheetsCount; i++)
			for (s = 0; s < 2; s++)
			{
				free(pLayers[i].pRawTracks[s]);
				free(pLayers[i].pSegments[s]);
				};
		free(pLayers);
		};
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadMarkMap(BufferedFile &F, IO_Header Hdr, IO_MarkMap **ppData)
{
	IO_MarkMap Map;
	Map.Hdr.Hdr = Hdr;
	DWORD br;
	MarkMap *pMarks = 0;
	BYTE *p = 0, *pData = 0;
	try
	{
		TRY_READ_B(HFile, br, &Map.Hdr.Id, sizeof(Map.Hdr.Id), IODATA_EXT_ERROR_READ + MAP_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Map.Map.Count, sizeof(Map.Map.Count), IODATA_EXT_ERROR_READ + MAP_HEADER_BYTES);

		p = pData = (BYTE *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(Map)) + QWORD_ALIGN(sizeof(Mark) * Map.Map.Count));
		if (!p) throw IODATA_EXT_ERROR_READ + MAP_DATA_BYTES + MAP_FULL_SECTION;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Map, sizeof(Map));
		((IO_MarkMap *)pData)->Map.pMarks = (Mark *)p;
		TRY_READ_B(HFile, br, p, (sizeof(Mark) * Map.Map.Count), IODATA_EXT_ERROR_READ + MAP_DATA_BYTES);
		}
	catch (unsigned err)
	{
		CoTaskMemFree(pData);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};	
	*ppData = (IO_MarkMap *)pData;
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadVertigoScanCatalog(BufferedFile &F, IO_Header Hdr, IO_VS_Catalog **ppData)
{
	IO_VS_Catalog Cat;
	Cat.Hdr.Type = Hdr;
	DWORD br;
	BYTE *p = 0, *pData = 0;
	VS_Config *pConfigs = 0;
	int *pFragmentIndices = 0;
	UINT PSpace = 0;
	int i;
	try
	{
		TRY_READ_B(HFile, br, &Cat.Hdr.ID, sizeof(Cat.Hdr.ID), IODATA_EXT_ERROR_READ + VS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Cat.Area, sizeof(Cat.Area), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CATALOG_SECTION);
		TRY_READ_B(HFile, br, &Cat.Config.CountOfConfigs, sizeof(Cat.Config.CountOfConfigs), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CATALOG_SECTION);

		pConfigs = (VS_Config *)malloc(sizeof(VS_Config) * Cat.Config.CountOfConfigs);
		if (!pConfigs && Cat.Config.CountOfConfigs) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CATALOG_SECTION;
		for (i = 0; i < Cat.Config.CountOfConfigs; i++)	pConfigs[i].Config.pItems = 0;
		for (i = 0; i < Cat.Config.CountOfConfigs; i++)
		{
			TRY_READ_B(HFile, br, &pConfigs[i].ClassName, sizeof(pConfigs[i].ClassName), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CONFIG_SECTION);
			TRY_READ_B(HFile, br, &pConfigs[i].Config.Name, sizeof(pConfigs[i].Config.Name), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CONFIG_SECTION);
			TRY_READ_B(HFile, br, &pConfigs[i].Config.CountOfItems, sizeof(pConfigs[i].Config.CountOfItems), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CONFIG_SECTION);
			pConfigs[i].Config.pItems = (char *)malloc((SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) * pConfigs[i].Config.CountOfItems);
			if (!pConfigs[i].Config.pItems && pConfigs[i].Config.CountOfItems) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CONFIG_SECTION;
			TRY_READ_B(HFile, br, pConfigs[i].Config.pItems, ((SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) * pConfigs[i].Config.CountOfItems), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CONFIG_SECTION);
			PSpace += QWORD_ALIGN(br);
			};

		pFragmentIndices = (int *)malloc(sizeof(int) * Cat.Area.XViews * Cat.Area.YViews);
		TRY_READ_B(HFile, br, pFragmentIndices, (sizeof(int) * Cat.Area.XViews * Cat.Area.YViews), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Cat.Reserved, sizeof(Cat.Reserved), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CATALOG_SECTION);

		p = pData = (BYTE *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(Cat)) + QWORD_ALIGN(sizeof(VS_Config)) * Cat.Config.CountOfConfigs + PSpace + QWORD_ALIGN(sizeof(int) * Cat.Area.XViews * Cat.Area.YViews));

		if (!p) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CATALOG_SECTION;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Cat, sizeof(Cat));
		((IO_VS_Catalog *)pData)->Config.pConfigs = (VS_Config *)p;
		for (i = 0; i < Cat.Config.CountOfConfigs; i++)
			QWORD_ALIGNED_COPY_AND_DISPLACE(p, &pConfigs[i], sizeof(pConfigs[i]));
		for (i = 0; i < Cat.Config.CountOfConfigs; i++)
		{
			((IO_VS_Catalog *)pData)->Config.pConfigs[i].Config.pItems = (char *)p;
			QWORD_ALIGNED_COPY_AND_DISPLACE(p, pConfigs[i].Config.pItems, (SYSAL_MAXCONFIG_ENTRY_NAME_LEN + SYSAL_MAXCONFIG_ENTRY_VALUE_LEN) * pConfigs[i].Config.CountOfItems);
			};
		((IO_VS_Catalog *)pData)->pFragmentIndices = (int *)p;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pFragmentIndices, (sizeof(int) * Cat.Area.XViews * Cat.Area.YViews));
		}
	catch (unsigned err)
	{
		if (pFragmentIndices) free(pFragmentIndices);
		if (pConfigs) 
		{
			for (i = 0; i < Cat.Config.CountOfConfigs; i++)
				if (pConfigs[i].Config.pItems) free(pConfigs[i].Config.pItems);
			free(pConfigs);
			};
		if (pData) CoTaskMemFree(pData);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};	
	if (pFragmentIndices) free(pFragmentIndices);
	if (pConfigs) 
	{
		for (i = 0; i < Cat.Config.CountOfConfigs; i++)
			if (pConfigs[i].Config.pItems) free(pConfigs[i].Config.pItems);
		free(pConfigs);
		};
	*ppData = (IO_VS_Catalog *)pData;
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadVertigoScanFragment(BufferedFile &F, IO_Header Hdr, IO_VS_Fragment **ppData)
{
	IO_VS_Fragment Frag;
	Frag.Hdr.Type.InfoType = Hdr.InfoType;
	Frag.Hdr.Type.HeaderFormat = VS_OLD_HEADER_TYPE;
	DWORD br;
	BYTE *p = 0, *pData = 0;
	VS_View *pViews = 0;
	BYTE *pCorr = 0;
	float *pZs = 0;
	int TSpace = 0;
	int GSpace = 0;
	int ZSpace = 0;
	int i, j, k;
	try
	{
		TRY_READ_B(HFile, br, &Frag.Hdr.ID, sizeof(Frag.Hdr.ID), IODATA_EXT_ERROR_READ + VS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Frag.Fragment.Index, sizeof(Frag.Fragment.Index), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.StartView, sizeof(Frag.Fragment.StartView), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.CountOfViews, sizeof(Frag.Fragment.CountOfViews), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.FitCorrectionDataSize, sizeof(Frag.Fragment.FitCorrectionDataSize), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.CodingMode, sizeof(Frag.Fragment.CodingMode), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Reserved, sizeof(Frag.Reserved), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);	

		switch (Frag.Fragment.CodingMode)
		{
			case VS_COMPRESSION_NULL:
			case VS_COMPRESSION_GRAIN_SUPPRESSION:	break;

			default:								throw MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKCOMPRESS, Frag.Hdr.Type);
			};

		pViews = (VS_View *)malloc(sizeof(VS_View) * Frag.Fragment.CountOfViews);
		if (!pViews && Frag.Fragment.CountOfViews) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION;
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			pViews[i].pTracks[0] = pViews[i].pTracks[1] = 0;
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
		{
			TRY_READ_B(HFile, br, &pViews[i].TileX, sizeof(pViews[i].TileX), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].TileY, sizeof(pViews[i].TileY), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].X, sizeof(pViews[i].X), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].Y, sizeof(pViews[i].Y), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].MapX, sizeof(pViews[i].MapX), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].MapY, sizeof(pViews[i].MapY), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].ImageMat, sizeof(pViews[i].ImageMat), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			for (j = 0; j < 2; j++)	TRY_READ_B(HFile, br, &pViews[i].Layers[j].Count, sizeof(pViews[i].Layers[j].Count), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
			TRY_READ_B(HFile, br, &pViews[i].RelevantZs, sizeof(pViews[i].RelevantZs), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].Status, sizeof(pViews[i].Status), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].TCount, sizeof(pViews[i].TCount), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
			ZSpace += pViews[i].Layers[0].Count + pViews[i].Layers[1].Count;
			TSpace += pViews[i].TCount[0] + pViews[i].TCount[1];
			};
		pZs = (float *)malloc(sizeof(float) * ZSpace);
		if (!pZs && (ZSpace > 0)) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION;
		if (Hdr.HeaderFormat == VS_HEADER_TYPE)
			for (j = 0; j < ZSpace; j++)	
			{
				static int dummy;
				TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
				TRY_READ_B(HFile, br, &pZs[j], sizeof(pZs[j]), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
				}
		else
			TRY_READ_B(HFile, br, pZs, (sizeof(float) * ZSpace), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
		if (Frag.Fragment.FitCorrectionDataSize)
		{
			pCorr = (BYTE *)malloc(Frag.Fragment.FitCorrectionDataSize * TSpace);
			if (!pCorr && (TSpace > 0)) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION;
			};
		if (Frag.Fragment.FitCorrectionDataSize) TSpace = 0;
		if (Hdr.HeaderFormat == VS_HEADER_TYPE)
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
				{
					pViews[i].pTracks[j] = (Track *)malloc(sizeof(Track) * pViews[i].TCount[j]);
					if (!pViews[i].pTracks[j] && pViews[i].TCount[j]) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION;
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						Track &t = pViews[i].pTracks[j][k];
						t.Field = i;
						static int dummy;
						TRY_READ_B(HFile, br, &dummy, sizeof(dummy), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
						TRY_READ_B(HFile, br, &t.PointsN, sizeof(t.PointsN), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Intercept, sizeof(t.Intercept), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Slope, sizeof(t.Slope), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Sigma, sizeof(t.Sigma), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.FirstZ, sizeof(t.FirstZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.LastZ, sizeof(t.LastZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						t.InterceptErrors.X = t.InterceptErrors.Y = t.InterceptErrors.Z = 
						t.SlopeErrors.X = t.SlopeErrors.Y = t.SlopeErrors.Z = 0.0f;
						t.Valid = true;
						if (Frag.Fragment.FitCorrectionDataSize)
						{
							TRY_READ_B(HFile, br, pCorr + TSpace * Frag.Fragment.FitCorrectionDataSize, Frag.Fragment.FitCorrectionDataSize, IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
							};
						TSpace++;
						GSpace += t.PointsN;
						};
					}
		else
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
				{
					pViews[i].pTracks[j] = (Track *)malloc(sizeof(Track) * pViews[i].TCount[j]);
					if (!pViews[i].pTracks[j] && pViews[i].TCount[j]) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION;
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						Track &t = pViews[i].pTracks[j][k];
						t.Field = i;
						TRY_READ_B(HFile, br, &t.PointsN, sizeof(t.PointsN), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Intercept, sizeof(t.Intercept), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Slope, sizeof(t.Slope), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Sigma, sizeof(t.Sigma), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.FirstZ, sizeof(t.FirstZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.LastZ, sizeof(t.LastZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.InterceptErrors, sizeof(t.InterceptErrors), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.SlopeErrors, sizeof(t.SlopeErrors), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						t.Valid = true;
						if (Frag.Fragment.FitCorrectionDataSize)
						{
							TRY_READ_B(HFile, br, pCorr + TSpace * Frag.Fragment.FitCorrectionDataSize, Frag.Fragment.FitCorrectionDataSize, IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
							};
						TSpace++;
						GSpace += t.PointsN;
						};
					};

		p = pData = (BYTE *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(Frag)) + QWORD_ALIGN(sizeof(VS_View)) * Frag.Fragment.CountOfViews + 
			QWORD_ALIGN(sizeof(float) * ZSpace) + QWORD_ALIGN(sizeof(Track)) * TSpace + QWORD_ALIGN(Frag.Fragment.FitCorrectionDataSize * TSpace) + 
			((C.InputData.SkipGrains || (Frag.Fragment.CodingMode == VS_COMPRESSION_GRAIN_SUPPRESSION)) ? 0 : QWORD_ALIGN(sizeof(TVector) * GSpace)));

		if (!p) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Frag, sizeof(Frag));
		((IO_VS_Fragment *)pData)->Fragment.pViews = (VS_View *)p;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pViews, (sizeof(VS_View) * Frag.Fragment.CountOfViews));
		ZSpace = 0;
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
			{
				((IO_VS_Fragment *)pData)->Fragment.pViews[i].Layers[j].pZs = (float *)p + ZSpace;
				ZSpace += ((IO_VS_Fragment *)pData)->Fragment.pViews[i].Layers[j].Count;
				};
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pZs, sizeof(float) * ZSpace);
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
			{
				((IO_VS_Fragment *)pData)->Fragment.pViews[i].pTracks[j] = (Track *)p;
				QWORD_ALIGNED_COPY_AND_DISPLACE(p, pViews[i].pTracks[j], sizeof(Track) * pViews[i].TCount[j]);
				};
		TSpace = 0;
		if (Frag.Fragment.FitCorrectionDataSize)
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						((IO_VS_Fragment *)pData)->Fragment.pViews[i].pTracks[j][k].pCorrection = p + TSpace * Frag.Fragment.FitCorrectionDataSize;
						TSpace++;					
						};
			QWORD_ALIGNED_COPY_AND_DISPLACE(p, pCorr, Frag.Fragment.FitCorrectionDataSize * TSpace);
			}
		else
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					for (k = 0; k < pViews[i].TCount[j]; k++)
						((IO_VS_Fragment *)pData)->Fragment.pViews[i].pTracks[j][k].pCorrection = 0;
			};
		
		if (!(C.InputData.SkipGrains || (Frag.Fragment.CodingMode == VS_COMPRESSION_GRAIN_SUPPRESSION)))
		{
			if (Hdr.HeaderFormat == VS_HEADER_TYPE)	
			{
				for (i = 0; i < GSpace; i++)
				{		
					Grain g;
					TRY_READ_B(HFile, br, &g, sizeof(g), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
					((TVector *)p + i)->X = g.X;
					((TVector *)p + i)->Y = g.Y;
					((TVector *)p + i)->Z = g.Z;
					};
				}
			else
			{
				TRY_READ_B(HFile, br, p, (sizeof(TVector) * GSpace), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_GRAIN_SECTION);
				}
		
			GSpace = 0;
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						((IO_VS_Fragment *)pData)->Fragment.pViews[i].pTracks[j][k].pPoints = (TVector *)(p + sizeof(TVector) * GSpace);
						GSpace += pViews[i].pTracks[j][k].PointsN;						
						};
			}
		else
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					for (k = 0; k < pViews[i].TCount[j]; k++)
						((IO_VS_Fragment *)pData)->Fragment.pViews[i].pTracks[j][k].pPoints = 0;					
			};
		}
	catch (unsigned err)
	{
		if (pViews) 
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					if (pViews[i].pTracks[j]) free(pViews[i].pTracks[j]);
			free(pViews);
			};
		if (pZs) free(pZs);
		if (pCorr) free(pCorr);
		if (pData) CoTaskMemFree(pData);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};	
	if (pViews) 
	{
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
				if (pViews[i].pTracks[j]) free(pViews[i].pTracks[j]);
		free(pViews);
		};
	if (pCorr) free(pCorr);
	if (pZs) free(pZs);
	*ppData = (IO_VS_Fragment *)pData;
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::ReadVertigoScanFragment2(BufferedFile &F, IO_Header Hdr, IO_VS_Fragment2 **ppData)
{
	IO_VS_Fragment2 Frag;
	Frag.Hdr.Type.InfoType = Hdr.InfoType;
	Frag.Hdr.Type.HeaderFormat = VS_HEADER_TYPE;
	DWORD br;
	BYTE *p = 0, *pData = 0;
	VS_View2 *pViews = 0;
	BYTE *pCorr = 0;
	VS_View2::t_LayerInfo *pLayers = 0;
	int TSpace = 0;
	int GSpace = 0;
	int LSpace = 0;
	int i, j, k;
	try
	{
		TRY_READ_B(HFile, br, &Frag.Hdr.ID, sizeof(Frag.Hdr.ID), IODATA_EXT_ERROR_READ + VS_HEADER_BYTES);
		TRY_READ_B(HFile, br, &Frag.Fragment.Index, sizeof(Frag.Fragment.Index), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.StartView, sizeof(Frag.Fragment.StartView), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.CountOfViews, sizeof(Frag.Fragment.CountOfViews), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Fragment.FitCorrectionDataSize, sizeof(Frag.Fragment.FitCorrectionDataSize), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		if (Hdr.HeaderFormat == VS_HEADER_TYPE && Frag.Fragment.FitCorrectionDataSize != 0) throw MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_OBSOLETE, Frag.Hdr.Type);
		TRY_READ_B(HFile, br, &Frag.Fragment.CodingMode, sizeof(Frag.Fragment.CodingMode), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);
		TRY_READ_B(HFile, br, &Frag.Reserved, sizeof(Frag.Reserved), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION);	

		switch (Frag.Fragment.CodingMode)
		{
			case VS_COMPRESSION_NULL:
			case VS_COMPRESSION_GRAIN_SUPPRESSION:	break;

			default:								throw MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKCOMPRESS, Frag.Hdr.Type);
			};

		pCorr = (BYTE *)malloc(Frag.Fragment.FitCorrectionDataSize);
		if (!pCorr && Frag.Fragment.FitCorrectionDataSize) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION;
		pViews = (VS_View2 *)malloc(sizeof(VS_View2) * Frag.Fragment.CountOfViews);
		if (!pViews && Frag.Fragment.CountOfViews) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_FRAGMENT_SECTION;
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			pViews[i].pTracks[0] = pViews[i].pTracks[1] = 0;
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
		{
			TRY_READ_B(HFile, br, &pViews[i].TileX, sizeof(pViews[i].TileX), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].TileY, sizeof(pViews[i].TileY), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].X, sizeof(pViews[i].X), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].Y, sizeof(pViews[i].Y), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].MapX, sizeof(pViews[i].MapX), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].MapY, sizeof(pViews[i].MapY), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].ImageMat, sizeof(pViews[i].ImageMat), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			for (j = 0; j < 2; j++)	TRY_READ_B(HFile, br, &pViews[i].Layers[j].Count, sizeof(pViews[i].Layers[j].Count), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].RelevantZs, sizeof(pViews[i].RelevantZs), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].Status, sizeof(pViews[i].Status), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);		
			TRY_READ_B(HFile, br, &pViews[i].TCount, sizeof(pViews[i].TCount), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
			LSpace += pViews[i].Layers[0].Count + pViews[i].Layers[1].Count;
			TSpace += pViews[i].TCount[0] + pViews[i].TCount[1];
			};
		pLayers = (VS_View2::t_LayerInfo *)malloc(sizeof(VS_View2::t_LayerInfo) * LSpace);
		if (!pLayers && (LSpace > 0)) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION;
		if (Hdr.HeaderFormat == VS_HEADER_TYPE) 
		{
			TRY_READ_B(HFile, br, pLayers, (sizeof(VS_View2::t_LayerInfo) * LSpace), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
			}
		else
			for (i = 0; i < LSpace; i++)
			{
				pLayers[i].Clusters = 0;
				TRY_READ_B(HFile, br, &pLayers[i].Z, sizeof(pLayers[i].Z), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
				}
		TSpace = 0;
		if (Hdr.HeaderFormat == VS_HEADER_TYPE)
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
				{
					pViews[i].pTracks[j] = (Track2 *)malloc(sizeof(Track2) * pViews[i].TCount[j]);
					if (!pViews[i].pTracks[j] && pViews[i].TCount[j]) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION;
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						Track2 &t = pViews[i].pTracks[j][k];
						t.Field = i;
						TRY_READ_B(HFile, br, &t.AreaSum, sizeof(t.AreaSum), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Grains, sizeof(t.Grains), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Intercept, sizeof(t.Intercept), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Slope, sizeof(t.Slope), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Sigma, sizeof(t.Sigma), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.FirstZ, sizeof(t.FirstZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.LastZ, sizeof(t.LastZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						t.Valid = true;
						TSpace++;
						GSpace += t.Grains;
						};
					};
			}
		else
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
				{
					pViews[i].pTracks[j] = (Track2 *)malloc(sizeof(Track2) * pViews[i].TCount[j]);
					if (!pViews[i].pTracks[j] && pViews[i].TCount[j]) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION;
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						Track2 &t = pViews[i].pTracks[j][k];
						t.Field = i;
						t.AreaSum = 0;
						static TVector Dummy;
						TRY_READ_B(HFile, br, &t.Grains, sizeof(t.Grains), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Intercept, sizeof(t.Intercept), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Slope, sizeof(t.Slope), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.Sigma, sizeof(t.Sigma), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.FirstZ, sizeof(t.FirstZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &t.LastZ, sizeof(t.LastZ), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &Dummy, sizeof(Dummy), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						TRY_READ_B(HFile, br, &Dummy, sizeof(Dummy), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_TRACK_SECTION);
						t.Valid = true;
						if (Frag.Fragment.FitCorrectionDataSize)
						{
							TRY_READ_B(HFile, br, pCorr, Frag.Fragment.FitCorrectionDataSize, IODATA_EXT_ERROR_WRITE + VS_DATA_BYTES + VS_TRACK_SECTION);
							};
						TSpace++;
						GSpace += t.Grains;
						};
					};
			}

		p = pData = (BYTE *)CoTaskMemAlloc(QWORD_ALIGN(sizeof(Frag)) + QWORD_ALIGN(sizeof(VS_View2)) * Frag.Fragment.CountOfViews + QWORD_ALIGN(sizeof(VS_View2::t_LayerInfo) * LSpace) + 
			QWORD_ALIGN(sizeof(Track2)) * TSpace + ((C.InputData.SkipGrains || (Frag.Fragment.CodingMode == VS_COMPRESSION_GRAIN_SUPPRESSION)) ? 
			0 : QWORD_ALIGN(sizeof(Grain) * GSpace)));

		if (!p) throw IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_CATALOG_SECTION;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, &Frag, sizeof(Frag));
		((IO_VS_Fragment2 *)pData)->Fragment.pViews = (VS_View2 *)p;
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pViews, (sizeof(VS_View2) * Frag.Fragment.CountOfViews));
		LSpace = 0;
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
			{
				((IO_VS_Fragment2 *)pData)->Fragment.pViews[i].Layers[j].pLayerInfo = (VS_View2::t_LayerInfo *)p + LSpace;
				LSpace += ((IO_VS_Fragment2 *)pData)->Fragment.pViews[i].Layers[j].Count;
				};
		QWORD_ALIGNED_COPY_AND_DISPLACE(p, pLayers, sizeof(VS_View2::t_LayerInfo) * LSpace);
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
			{
				((IO_VS_Fragment2 *)pData)->Fragment.pViews[i].pTracks[j] = (Track2 *)p;
				QWORD_ALIGNED_COPY_AND_DISPLACE(p, pViews[i].pTracks[j], sizeof(Track2) * pViews[i].TCount[j]);
				};
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
				for (k = 0; k < pViews[i].TCount[j]; k++)
					((IO_VS_Fragment2 *)pData)->Fragment.pViews[i].pTracks[j][k].pCorrection = 0;
		
		if (!(C.InputData.SkipGrains || (Frag.Fragment.CodingMode == VS_COMPRESSION_GRAIN_SUPPRESSION)))
		{
			if (Hdr.HeaderFormat == VS_HEADER_TYPE)	
			{
				TRY_READ_B(HFile, br, p, (sizeof(Grain) * GSpace), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_GRAIN_SECTION);
				}
			else
				for (i = 0; i < GSpace; i++)
				{
					((Grain *)p + i)->Area = 0;
					TRY_READ_B(HFile, br, &((Grain *)p + i)->X, sizeof(((Grain *)p + i)->X), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
					TRY_READ_B(HFile, br, &((Grain *)p + i)->Y, sizeof(((Grain *)p + i)->Y), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
					TRY_READ_B(HFile, br, &((Grain *)p + i)->Z, sizeof(((Grain *)p + i)->Z), IODATA_EXT_ERROR_READ + VS_DATA_BYTES + VS_VIEW_SECTION);
					};
			GSpace = 0;
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					for (k = 0; k < pViews[i].TCount[j]; k++)
					{
						((IO_VS_Fragment2 *)pData)->Fragment.pViews[i].pTracks[j][k].pGrains = (Grain *)(p + sizeof(Grain) * GSpace);
						GSpace += pViews[i].pTracks[j][k].Grains;						
						};
			}
		else
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					for (k = 0; k < pViews[i].TCount[j]; k++)
						((IO_VS_Fragment2 *)pData)->Fragment.pViews[i].pTracks[j][k].pGrains = 0;					
			};
		}
	catch (unsigned err)
	{
		if (pViews) 
		{
			for (i = 0; i < Frag.Fragment.CountOfViews; i++)
				for (j = 0; j < 2; j++)
					if (pViews[i].pTracks[j]) free(pViews[i].pTracks[j]);
			free(pViews);
			};
		if (pLayers) free(pLayers);
		if (pCorr) free(pCorr);
		if (pData) CoTaskMemFree(pData);
		F.Close(); //CloseHandle(HFile);
		return err;		
		};	
	if (pViews) 
	{
		for (i = 0; i < Frag.Fragment.CountOfViews; i++)
			for (j = 0; j < 2; j++)
				if (pViews[i].pTracks[j]) free(pViews[i].pTracks[j]);
		free(pViews);
		};
	if (pCorr) free(pCorr);
	if (pLayers) free(pLayers);
	*ppData = (IO_VS_Fragment2 *)pData;
	F.Close(); //CloseHandle(HFile);
	return IODATA_EXT_ERROR_OK;
}

UINT CSySalDataIO::Read(char *FileName, IO_Data **ppData, bool UseISySalDataIO)
{
/*
	HANDLE HFile = CreateFile(FileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (HFile == INVALID_HANDLE_VALUE) return IODATA_EXT_ERROR_OPEN;
*/
	BufferedFile F;
	if (!F.Open(FileName, false)) return IODATA_EXT_ERROR_OPEN;

	IO_Header Hdr;
	DWORD br;
	try
	{
		TRY_READ_B(HFile, br, &Hdr, sizeof(Hdr), IODATA_EXT_ERROR_READ);
		if ((Hdr.InfoType != (BULK_HEADER_BYTES | BULK_TRACK_SECTION) || (Hdr.HeaderFormat != BULK_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD2_HEADER_TYPE)) &&
			(Hdr.InfoType != (BULK_HEADER_BYTES | BULK_IMAGE_SECTION) || (Hdr.HeaderFormat != BULK_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD2_HEADER_TYPE)) &&
			(Hdr.InfoType != (BULK_HEADER_BYTES | BULK_CLUSTER_SECTION) || (Hdr.HeaderFormat != BULK_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD2_HEADER_TYPE)) &&
			(Hdr.InfoType != (BULK_HEADER_BYTES | BULK_IMAGEPRED_SECTION) || (Hdr.HeaderFormat != BULK_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD2_HEADER_TYPE)) &&
			(Hdr.InfoType != (BULK_HEADER_BYTES | BULK_CLUSTERPRED_SECTION) || (Hdr.HeaderFormat != BULK_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD_HEADER_TYPE && Hdr.HeaderFormat != BULK_OLD2_HEADER_TYPE)) &&
			(Hdr.InfoType != (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) || (Hdr.HeaderFormat != CS_SS_HEADER_TYPE && Hdr.HeaderFormat != CS_SS_OLD_HEADER_TYPE && Hdr.HeaderFormat != CS_SS_OLD2_HEADER_TYPE)) &&
			(Hdr.InfoType != (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) || (Hdr.HeaderFormat != OPERA_HEADER_TYPE)) &&
			(Hdr.InfoType != (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION) || (Hdr.HeaderFormat != OPERA_HEADER_TYPE_WITH_INDEX)) &&
			(Hdr.InfoType != (BS_HEADER_BYTES | BS_PRED_SECTION) || (Hdr.HeaderFormat != BS_HEADER_TYPE)) &&
			(Hdr.InfoType != (TS_HEADER_BYTES | TS_FULL_SECTION) || (Hdr.HeaderFormat != TS_HEADER_TYPE && Hdr.HeaderFormat != TS_OLD_HEADER_TYPE)) &&
			(Hdr.InfoType != (MAP_HEADER_BYTES | MAP_FULL_SECTION) || Hdr.HeaderFormat != MAP_MARK_HEADER_TYPE) && 
			(Hdr.InfoType != (VS_HEADER_BYTES | VS_CATALOG_SECTION) || Hdr.HeaderFormat != VS_OLD_HEADER_TYPE) && 
			(Hdr.InfoType != (VS_HEADER_BYTES | VS_CATALOG_SECTION) || Hdr.HeaderFormat != VS_HEADER_TYPE) && 
			(Hdr.InfoType != (VS_HEADER_BYTES | VS_FRAGMENT_SECTION) ||Hdr.HeaderFormat != VS_OLD_HEADER_TYPE) && 
			(Hdr.InfoType != (VS_HEADER_BYTES | VS_FRAGMENT_SECTION) ||Hdr.HeaderFormat != VS_HEADER_TYPE)) 
				throw MAKE_IO_EXT_ERROR(IODATA_EXT_ERROR_UNKFORMAT, Hdr);
		}
	catch (unsigned err)
	{
		//if (HFile != INVALID_HANDLE_VALUE) CloseHandle(HFile);
		F.Close();
		return err;
		};
	switch (Hdr.InfoType)
	{
		//case (BULK_HEADER_BYTES | BULK_TRACK_SECTION): the same as below
		case (CS_SS_HEADER_BYTES | CS_SS_TRACK_SECTION):	if (Hdr.HeaderFormat != CS_SS_HEADER_TYPE && Hdr.HeaderFormat != CS_SS_OLD_HEADER_TYPE && Hdr.HeaderFormat != CS_SS_OLD2_HEADER_TYPE && Hdr.HeaderFormat != OPERA_HEADER_TYPE && Hdr.HeaderFormat != OPERA_HEADER_TYPE_WITH_INDEX) 
																return ReadBulkTracks(F, Hdr, (IO_BULK_Tracks **)ppData);
															else return (UseISySalDataIO) ? ReadCSSSTracks(F, Hdr, (IO_CS_SS_Tracks **)ppData) : ReadOPERATracks(F, Hdr, (IO_OPERA_Tracks **)ppData);

		case (BULK_HEADER_BYTES | BULK_IMAGEPRED_SECTION):	
		case (BULK_HEADER_BYTES | BULK_IMAGE_SECTION):		return ReadBulkPhotoSeq(F, Hdr, (IO_BULK_PhotoSeq **)ppData);		

		case (BULK_HEADER_BYTES | BULK_CLUSTERPRED_SECTION):
		case (BULK_HEADER_BYTES | BULK_CLUSTER_SECTION):	return ReadBulkClustersDump(F, Hdr, (IO_BULK_ClustersDump **)ppData);

		case (BS_HEADER_BYTES | BS_PRED_SECTION):			return ReadBulkCSSSBatch(F, Hdr, (IO_BS_Batch **)ppData);

		case (TS_HEADER_BYTES | TS_FULL_SECTION):			return ReadTSReconstruction(F, Hdr, (IO_TS_Reconstruction **)ppData);

		case (MAP_HEADER_BYTES | MAP_FULL_SECTION):			return ReadMarkMap(F, Hdr, (IO_MarkMap **)ppData);

		case (VS_HEADER_BYTES | VS_CATALOG_SECTION):		return ReadVertigoScanCatalog(F, Hdr, (IO_VS_Catalog **)ppData);

		case (VS_HEADER_BYTES | VS_FRAGMENT_SECTION):		return (UseISySalDataIO) ? ReadVertigoScanFragment(F, Hdr, (IO_VS_Fragment **)ppData) : ReadVertigoScanFragment2(F, Hdr, (IO_VS_Fragment2 **)ppData);
		};
	//if (HFile != INVALID_HANDLE_VALUE) CloseHandle(HFile);
	F.Close();
	return IODATA_EXT_ERROR_UNKNOWN;
}


CSySalDataIO::BufferedFile::BufferedFile(int buffersize)
{
	pBuffer = (BYTE *)GlobalAlloc(GMEM_FIXED, BufferSize = buffersize);
	HFile = INVALID_HANDLE_VALUE;
	Write = false;
	FillLevel = 0;
	};

CSySalDataIO::BufferedFile::~BufferedFile()
{
	Close();
	if (pBuffer) GlobalFree(pBuffer);
	};

bool CSySalDataIO::BufferedFile::Open(char *path, bool write)
{
	HFile = CreateFile(path, write ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ, FILE_SHARE_READ, NULL, write ? CREATE_NEW : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	Write = write;
	FillLevel = 0;
	return (HFile != INVALID_HANDLE_VALUE && HFile != NULL);
	};

bool CSySalDataIO::BufferedFile::WriteBytes(BYTE *pbytes, int length)
{
	if (HFile == INVALID_HANDLE_VALUE || HFile == NULL || !Write) return false;
	DWORD byteswritten = 0;
	do
	{
		int tobufflen = (BufferSize - FillLevel);
		if (tobufflen > length) tobufflen = length;
		memcpy(pBuffer + FillLevel, pbytes, tobufflen);
		FillLevel += tobufflen;
		pbytes += tobufflen;
		length -= tobufflen;
		if (FillLevel >= BufferSize)
		{
			if ((!WriteFile(HFile, pBuffer, FillLevel, &byteswritten, NULL)) || (byteswritten != FillLevel)) return false;
			FillLevel = 0;
			};
		}
	while(length > 0);
	return true;
	};

bool CSySalDataIO::BufferedFile::ReadBytes(BYTE *pbytes, int length, int *pbytesread)
{
	if (HFile == INVALID_HANDLE_VALUE || HFile == NULL || Write) return false;
	DWORD bytesread = 0;
	*pbytesread = 0;
	while (length > 0)
	{
		if (FillLevel < length)
		{
			if (!ReadFile(HFile, pBuffer + FillLevel, BufferSize - FillLevel, &bytesread, NULL)) return false;
			FillLevel += bytesread;
			if ((FillLevel < BufferSize) && (FillLevel < length)) return false;
			}
		int tobufflen = (FillLevel > length) ? length : FillLevel;
		memcpy(pbytes, pBuffer, tobufflen);
		pbytes += tobufflen;
		length -= tobufflen;
		*pbytesread += tobufflen;
		memcpy(pBuffer, pBuffer + tobufflen, FillLevel - tobufflen);
		FillLevel -= tobufflen;
		}
	return true;
	};

bool CSySalDataIO::BufferedFile::Close()
{
	if (HFile == INVALID_HANDLE_VALUE || HFile == NULL) return false;
	DWORD dummy;
	if (Write && FillLevel > 0)
	{
		WriteFile(HFile, pBuffer, FillLevel, &dummy, NULL);
		}
	FillLevel = 0;
	CloseHandle(HFile);
	HFile = INVALID_HANDLE_VALUE;
	Write = false;
	return true;
	};

/*
	class AsyncBufferedFile
	{
		protected:

			HANDLE HWriteThread;
			char *pPath;
			char *pAlternatePath;

		public:
*/

CSySalDataIO::AsyncBufferedFile::AsyncBufferedFile(int attempts, int buffersize) : Attempts(attempts)
{
	BufferSize = buffersize;
	Write = true;
	HFile = INVALID_HANDLE_VALUE;
	pBuffer = 0;
	HWriteThread = INVALID_HANDLE_VALUE;
	pPath = 0;
	pAlternatePath = 0;
	};

CSySalDataIO::AsyncBufferedFile::~AsyncBufferedFile()
{
	if (HWriteThread != INVALID_HANDLE_VALUE)
	{
		WaitForSingleObject(HWriteThread, INFINITE);
		HWriteThread = INVALID_HANDLE_VALUE;
		}
	if (pPath) delete [] pPath;
	if (pAlternatePath) delete [] pAlternatePath;
	pBuffer = 0;
	}

bool CSySalDataIO::AsyncBufferedFile::Open(char *path, bool write)
{
	if (!write) return BufferedFile::Open(path, write);
	int l = strlen(path);
	pPath = new char[l + 1];
	strcpy(pPath, path);
	pAlternatePath = new char[strlen(path + l + 1) + 1];
	strcpy(pAlternatePath, path + l + 1);
	pBuffer = (BYTE *)GlobalAlloc(GMEM_FIXED, BufferSize);
	Write = write;
	FillLevel = 0;
	return true;
	}

bool CSySalDataIO::AsyncBufferedFile::Close()
{
	if (!Write) return BufferedFile::Close();
	return (_beginthread(WriteThreadHook, 0, this) != 0);
	}

bool CSySalDataIO::AsyncBufferedFile::WriteBytes(BYTE *pbytes, int length)
{
	if (!Write) return false;
	if (BufferSize - FillLevel < length) return false;
	memcpy(pBuffer + FillLevel, pbytes, length);
	FillLevel += length;
	return true;
	}

void CSySalDataIO::AsyncBufferedFile::WriteThreadHook(void *powner)
{
	((AsyncBufferedFile *)powner)->WriteThread();
	}

void CSySalDataIO::AsyncBufferedFile::WriteThread()
{
	int i;
	HANDLE HFile = INVALID_HANDLE_VALUE;
	for (i = 0; i < Attempts; i++)
	{
		HFile = CreateFile(pPath, GENERIC_READ, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
		if (HFile == INVALID_HANDLE_VALUE || HFile == 0)
		{
			HFile = INVALID_HANDLE_VALUE;
			Sleep(1000); 
			continue;
			}
		DWORD byteswritten = 0;
		if (!WriteFile(HFile, pBuffer, FillLevel, &byteswritten, NULL) || byteswritten < FillLevel)
		{
			CloseHandle(HFile);
			HFile = INVALID_HANDLE_VALUE;
			Sleep(1000);
			continue;
			}
		CloseHandle(HFile);
		if (pBuffer) GlobalFree(pBuffer);
		return;
		}	 
	for (i = 0; i < Attempts; i++)
	{
		HFile = CreateFile(pAlternatePath, GENERIC_READ, FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
		if (HFile == INVALID_HANDLE_VALUE || HFile == 0)
		{
			HFile = INVALID_HANDLE_VALUE;
			Sleep(1000); 
			continue;
			}
		DWORD byteswritten = 0;
		if (!WriteFile(HFile, pBuffer, FillLevel, &byteswritten, NULL) || byteswritten < FillLevel)
		{
			CloseHandle(HFile);
			HFile = INVALID_HANDLE_VALUE;
			Sleep(1000);
			continue;
			}
		CloseHandle(HFile);
		break;
		}	 
	if (pBuffer) GlobalFree(pBuffer);
	}
