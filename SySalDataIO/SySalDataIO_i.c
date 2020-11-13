

/* this ALWAYS GENERATED file contains the IIDs and CLSIDs */

/* link this file in with the server and any clients */


 /* File created by MIDL compiler version 6.00.0366 */
/* at Thu Jan 26 13:30:47 2012
 */
/* Compiler settings for .\SySalDataIO.idl:
    Oicf, W1, Zp8, env=Win32 (32b run)
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
//@@MIDL_FILE_HEADING(  )

#pragma warning( disable: 4049 )  /* more than 64k source lines */


#ifdef __cplusplus
extern "C"{
#endif 


#include <rpc.h>
#include <rpcndr.h>

#ifdef _MIDL_USE_GUIDDEF_

#ifndef INITGUID
#define INITGUID
#include <guiddef.h>
#undef INITGUID
#else
#include <guiddef.h>
#endif

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        DEFINE_GUID(name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8)

#else // !_MIDL_USE_GUIDDEF_

#ifndef __IID_DEFINED__
#define __IID_DEFINED__

typedef struct _IID
{
    unsigned long x;
    unsigned short s1;
    unsigned short s2;
    unsigned char  c[8];
} IID;

#endif // __IID_DEFINED__

#ifndef CLSID_DEFINED
#define CLSID_DEFINED
typedef IID CLSID;
#endif // CLSID_DEFINED

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        const type name = {l,w1,w2,{b1,b2,b3,b4,b5,b6,b7,b8}}

#endif !_MIDL_USE_GUIDDEF_

MIDL_DEFINE_GUID(IID, IID_ISySalObject,0xC022EEAD,0x748A,0x11D3,0xA4,0x7B,0xE8,0x9C,0x0B,0xCE,0x97,0x20);


MIDL_DEFINE_GUID(IID, IID_ISySalDataIO,0x4B47E8CD,0x894C,0x11D3,0xA4,0x7C,0x9F,0x37,0x35,0x22,0x60,0x36);


MIDL_DEFINE_GUID(IID, IID_ISySalDataIO2,0x5892a1f5,0x5dd9,0x4fe3,0xa0,0x24,0xd8,0xb6,0x24,0x91,0x7c,0x1d);


MIDL_DEFINE_GUID(IID, LIBID_SYSALDATAIOLib,0x4B47E8C0,0x894C,0x11D3,0xA4,0x7C,0x9F,0x37,0x35,0x22,0x60,0x36);


MIDL_DEFINE_GUID(CLSID, CLSID_SySalDataIO,0x4B47E8CE,0x894C,0x11D3,0xA4,0x7C,0x9F,0x37,0x35,0x22,0x60,0x36);

#undef MIDL_DEFINE_GUID

#ifdef __cplusplus
}
#endif



