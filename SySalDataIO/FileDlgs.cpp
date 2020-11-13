#include "StdAfx.h"
#include "FileDlgs.h"
#include <afxdlgs.h>

bool SelectInputFile(CString &Path, CString Title, CString Ext, CString Desc, CWnd *pParent)
{
	CString Filter = "All Files|*.*||";
	if (Ext != "")
		Filter = ((Desc == "") ? (Ext + " Files") : Desc) + "|" + "*." + Ext + "|" + Filter;
	CFileDialog MyDlg(TRUE, Ext, Path, OFN_EXPLORER, Filter, pParent);
	MyDlg.m_ofn.lpstrTitle = Title;
	if (MyDlg.DoModal() == IDOK)
	{
		Path = MyDlg.GetPathName();
		return true;
		};	
	return false;
	};

bool SelectOutputFile(CString &Path, CString Title, CString Ext, CString Desc, CWnd *pParent)
{
	CString Filter = "All Files|*.*||";
	if (Ext != "")
		Filter = ((Desc == "") ? (Ext + " Files") : Desc) + "|" + "*." + Ext + "|" + Filter;
	CFileDialog MyDlg(FALSE, Ext, Path, OFN_EXPLORER | OFN_OVERWRITEPROMPT, Filter, pParent);
	MyDlg.m_ofn.lpstrTitle = Title;
	if (MyDlg.DoModal() == IDOK)
	{
		Path = MyDlg.GetPathName();
		return true;
		};	
	return false;
	};
