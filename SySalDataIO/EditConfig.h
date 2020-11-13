#if !defined(AFX_EDITCONFIG_H__5F8AF0C0_A008_11D3_A47D_D6B10EA8A430__INCLUDED_)
#define AFX_EDITCONFIG_H__5F8AF0C0_A008_11D3_A47D_D6B10EA8A430__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// EditConfig.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CEditConfig dialog

class CEditConfig : public CDialog
{
// Construction
public:
	CEditConfig(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CEditConfig)
	enum { IDD = IDD_EDIT_CONFIG };
	BOOL	m_AutoAppendUID;
	CString	m_DataDest;
	CString	m_DataSource;
	UINT	m_InputAttempts;
	BOOL	m_InputEnable;
	UINT	m_OutputAttempts;
	BOOL	m_OutputEnable;
	CString	m_InputDefDesc;
	CString	m_InputDefExt;
	CString	m_OutputDefDesc;
	CString	m_OutputDefExt;
	CString	m_RescueDest;
	BOOL	m_GrainCompressEnable;
	float	m_DepthPrecision;
	float	m_TransversePrecision;
	BOOL	m_SkipGrains;
	INT		m_AsyncWriteBufferMB;
	//}}AFX_DATA


// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CEditConfig)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:

	// Generated message map functions
	//{{AFX_MSG(CEditConfig)
	afx_msg void OnSelInputFile();
	afx_msg void OnSelOutputFile();
	afx_msg void OnSelOutRescueFile();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_EDITCONFIG_H__5F8AF0C0_A008_11D3_A47D_D6B10EA8A430__INCLUDED_)
