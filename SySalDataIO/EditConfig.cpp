// EditConfig.cpp : implementation file
//

#include "stdafx.h"
#include "resource.h"
#include "EditConfig.h"
#include "FileDlgs.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CEditConfig dialog


CEditConfig::CEditConfig(CWnd* pParent /*=NULL*/)
	: CDialog(CEditConfig::IDD, pParent)
{
	//{{AFX_DATA_INIT(CEditConfig)
	m_AutoAppendUID = FALSE;
	m_DataDest = _T("");
	m_DataSource = _T("");
	m_InputAttempts = 3;
	m_InputEnable = FALSE;
	m_SkipGrains = FALSE;
	m_OutputAttempts = 3;
	m_OutputEnable = FALSE;
	m_InputDefDesc = _T("");
	m_InputDefExt = _T("");
	m_OutputDefDesc = _T("");
	m_OutputDefExt = _T("");
	m_RescueDest = _T("");
	m_GrainCompressEnable = FALSE;
	m_DepthPrecision = 0.0f;
	m_TransversePrecision = 0.0f;
	m_AsyncWriteBufferMB = 0;
	//}}AFX_DATA_INIT
}


void CEditConfig::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CEditConfig)
	DDX_Check(pDX, IDC_EC_AUTOAPPENDUID, m_AutoAppendUID);
	DDX_Text(pDX, IDC_EC_DATADEST, m_DataDest);
	DDX_Text(pDX, IDC_EC_DATASOURCE, m_DataSource);
	DDX_Text(pDX, IDC_EC_INATTEMPTS, m_InputAttempts);
	DDV_MinMaxUInt(pDX, m_InputAttempts, 1, 10);
	DDX_Check(pDX, IDC_EC_INPUT, m_InputEnable);
	DDX_Text(pDX, IDC_EC_OUTATTEMPTS, m_OutputAttempts);
	DDV_MinMaxUInt(pDX, m_OutputAttempts, 1, 10);
	DDX_Check(pDX, IDC_EC_OUTPUT, m_OutputEnable);
	DDX_Text(pDX, IDC_EC_INDEFDESC, m_InputDefDesc);
	DDX_Text(pDX, IDC_EC_INDEFEXT, m_InputDefExt);
	DDX_Text(pDX, IDC_EC_OUTDEFDESC, m_OutputDefDesc);
	DDX_Text(pDX, IDC_EC_OUTDEFEXT, m_OutputDefExt);
	DDX_Text(pDX, IDC_EC_RESCUEDEST, m_RescueDest);
	DDX_Check(pDX, IDC_EC_O_GRAINCOMPRESS, m_GrainCompressEnable);
	DDX_Text(pDX, IDC_EC_O_DEPTHPREC, m_DepthPrecision);
	DDV_MinMaxFloat(pDX, m_DepthPrecision, 1.e-004f, 100.f);
	DDX_Text(pDX, IDC_EC_O_TRANSVPREC, m_TransversePrecision);
	DDV_MinMaxFloat(pDX, m_TransversePrecision, 1.e-004f, 100.f);
	DDX_Check(pDX, IDC_EC_SKIPGRAINS, m_SkipGrains);
	DDX_Text(pDX, IDC_EC_ASYNCWRITEBUFFERMB, m_AsyncWriteBufferMB);
	DDV_MinMaxInt(pDX, m_AsyncWriteBufferMB, 4, 64);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CEditConfig, CDialog)
	//{{AFX_MSG_MAP(CEditConfig)
	ON_BN_CLICKED(IDC_EC_SELINFILE, OnSelInputFile)
	ON_BN_CLICKED(IDC_EC_SELOUTFILE, OnSelOutputFile)
	ON_BN_CLICKED(IDC_EC_SELOUTRESCUEFILE, OnSelOutRescueFile)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CEditConfig message handlers

void CEditConfig::OnSelInputFile() 
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);
	if (SelectInputFile(m_DataSource, "Select Input File(set)", m_InputDefExt, m_InputDefDesc, this)) UpdateData(FALSE);
}

void CEditConfig::OnSelOutputFile() 
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);
	if (SelectOutputFile(m_DataDest, "Select Output File(set)", m_OutputDefExt, m_OutputDefDesc, this)) UpdateData(FALSE);
}

void CEditConfig::OnSelOutRescueFile() 
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);
	if (SelectOutputFile(m_RescueDest, "Select Rescue File(set)", m_OutputDefExt, m_OutputDefDesc, this)) UpdateData(FALSE);	
}
