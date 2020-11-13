// SySalDataIO.cpp : Implementation of DLL Exports.


// Note: Proxy/Stub Information
//      To build a separate proxy/stub DLL, 
//      run nmake -f SySalDataIOps.mk in the project directory.

#include "stdafx.h"
#include "resource.h"
#include <initguid.h>
#include "SySalDataIO.h"

#include "SySalDataIO_i.c"
#include "DataIO.h"


CComModule _Module;

BEGIN_OBJECT_MAP(ObjectMap)
OBJECT_ENTRY(CLSID_SySalDataIO, CSySalDataIO)
END_OBJECT_MAP()

class CSySalDataIOApp : public CWinApp
{
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CSySalDataIOApp)
	public:
    virtual BOOL InitInstance();
    virtual int ExitInstance();
	//}}AFX_VIRTUAL

	//{{AFX_MSG(CSySalDataIOApp)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

BEGIN_MESSAGE_MAP(CSySalDataIOApp, CWinApp)
	//{{AFX_MSG_MAP(CSySalDataIOApp)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

CSySalDataIOApp theApp;

BOOL CSySalDataIOApp::InitInstance()
{
    _Module.Init(ObjectMap, m_hInstance, &LIBID_SYSALDATAIOLib);
    return CWinApp::InitInstance();
}

int CSySalDataIOApp::ExitInstance()
{
    _Module.Term();
    return CWinApp::ExitInstance();
}

/////////////////////////////////////////////////////////////////////////////
// Used to determine whether the DLL can be unloaded by OLE

STDAPI DllCanUnloadNow(void)
{
    AFX_MANAGE_STATE(AfxGetStaticModuleState());
    return (AfxDllCanUnloadNow()==S_OK && _Module.GetLockCount()==0) ? S_OK : S_FALSE;
}

/////////////////////////////////////////////////////////////////////////////
// Returns a class factory to create an object of the requested type

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, LPVOID* ppv)
{
    return _Module.GetClassObject(rclsid, riid, ppv);
}

/////////////////////////////////////////////////////////////////////////////
// DllRegisterServer - Adds entries to the system registry

STDAPI DllRegisterServer(void)
{
    // registers object, typelib and all interfaces in typelib
	CRegKey MyKey;
	try
	{
		if (MyKey.Create(HKEY_LOCAL_MACHINE, "Software", REG_NONE, REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_READ) != ERROR_SUCCESS) throw 0;
		MyKey.Close();
		if (MyKey.Create(HKEY_LOCAL_MACHINE, "Software\\SySal2", REG_NONE, REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_READ) != ERROR_SUCCESS) throw 1;
		MyKey.Close();
		if (MyKey.Create(HKEY_LOCAL_MACHINE, "Software\\SySal2\\Classes", REG_NONE, REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_READ) != ERROR_SUCCESS) throw 2;
		MyKey.Close();
		if (MyKey.Create(HKEY_LOCAL_MACHINE, "Software\\SySal2\\Classes\\SySalDataIO", REG_NONE, REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_READ) != ERROR_SUCCESS) throw 3;
		MyKey.Close();
		MyKey.Open(HKEY_LOCAL_MACHINE, "Software\\SySal2\\Classes\\SySalDataIO", KEY_WRITE | KEY_READ);
		char CLSIDString[64];
		wsprintf(CLSIDString, "{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}", 
			CLSID_SySalDataIO.Data1, CLSID_SySalDataIO.Data2, CLSID_SySalDataIO.Data3, 
			CLSID_SySalDataIO.Data4[0], CLSID_SySalDataIO.Data4[1], CLSID_SySalDataIO.Data4[2], CLSID_SySalDataIO.Data4[3], 
			CLSID_SySalDataIO.Data4[4], CLSID_SySalDataIO.Data4[5], CLSID_SySalDataIO.Data4[6], CLSID_SySalDataIO.Data4[7]);
		MyKey.SetValue(CLSIDString, "CLSID");
		MyKey.Close();
		}
	catch (...)
	{
		MyKey.Close();
		MyKey.Open(HKEY_LOCAL_MACHINE, "", KEY_WRITE | KEY_READ);
		MyKey.RecurseDeleteKey("Software\\SySal2\\Classes\\SySalDataIO");
		MyKey.Close();
		};
    return _Module.RegisterServer(FALSE/*TRUE*/);
}

/////////////////////////////////////////////////////////////////////////////
// DllUnregisterServer - Removes entries from the system registry

STDAPI DllUnregisterServer(void)
{
	CRegKey MyKey;
	MyKey.Open(HKEY_LOCAL_MACHINE, "", KEY_WRITE | KEY_READ);
	MyKey.RecurseDeleteKey("Software\\SySal2\\Classes\\SySalDataIO");
	MyKey.Close();
    return _Module.UnregisterServer(TRUE);
}


