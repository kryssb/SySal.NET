#ifndef _SYSAL_DATAIO_FILESEL_
#define _SYSAL_DATAIO_FILESEL_

bool SelectInputFile(CString &Path, CString Title, CString Ext, CString Desc, CWnd *pParent);

bool SelectOutputFile(CString &Path, CString Title, CString Ext, CString Desc, CWnd *pParent);

#endif