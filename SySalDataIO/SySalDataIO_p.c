

/* this ALWAYS GENERATED file contains the proxy stub code */


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

#if !defined(_M_IA64) && !defined(_M_AMD64)


#pragma warning( disable: 4049 )  /* more than 64k source lines */
#if _MSC_VER >= 1200
#pragma warning(push)
#endif
#pragma warning( disable: 4100 ) /* unreferenced arguments in x86 call */
#pragma warning( disable: 4211 )  /* redefine extent to static */
#pragma warning( disable: 4232 )  /* dllimport identity*/
#pragma optimize("", off ) 

#define USE_STUBLESS_PROXY


/* verify that the <rpcproxy.h> version is high enough to compile this file*/
#ifndef __REDQ_RPCPROXY_H_VERSION__
#define __REQUIRED_RPCPROXY_H_VERSION__ 475
#endif


#include "rpcproxy.h"
#ifndef __RPCPROXY_H_VERSION__
#error this stub requires an updated version of <rpcproxy.h>
#endif // __RPCPROXY_H_VERSION__


#include "SySalDataIO.h"

#define TYPE_FORMAT_STRING_SIZE   85                                
#define PROC_FORMAT_STRING_SIZE   937                               
#define TRANSMIT_AS_TABLE_SIZE    0            
#define WIRE_MARSHAL_TABLE_SIZE   0            

typedef struct _MIDL_TYPE_FORMAT_STRING
    {
    short          Pad;
    unsigned char  Format[ TYPE_FORMAT_STRING_SIZE ];
    } MIDL_TYPE_FORMAT_STRING;

typedef struct _MIDL_PROC_FORMAT_STRING
    {
    short          Pad;
    unsigned char  Format[ PROC_FORMAT_STRING_SIZE ];
    } MIDL_PROC_FORMAT_STRING;


static RPC_SYNTAX_IDENTIFIER  _RpcTransferSyntax = 
{{0x8A885D04,0x1CEB,0x11C9,{0x9F,0xE8,0x08,0x00,0x2B,0x10,0x48,0x60}},{2,0}};


extern const MIDL_TYPE_FORMAT_STRING __MIDL_TypeFormatString;
extern const MIDL_PROC_FORMAT_STRING __MIDL_ProcFormatString;


extern const MIDL_STUB_DESC Object_StubDesc;


extern const MIDL_SERVER_INFO ISySalObject_ServerInfo;
extern const MIDL_STUBLESS_PROXY_INFO ISySalObject_ProxyInfo;


extern const MIDL_STUB_DESC Object_StubDesc;


extern const MIDL_SERVER_INFO ISySalDataIO_ServerInfo;
extern const MIDL_STUBLESS_PROXY_INFO ISySalDataIO_ProxyInfo;


extern const MIDL_STUB_DESC Object_StubDesc;


extern const MIDL_SERVER_INFO ISySalDataIO2_ServerInfo;
extern const MIDL_STUBLESS_PROXY_INFO ISySalDataIO2_ProxyInfo;



#if !defined(__RPC_WIN32__)
#error  Invalid build platform for this stub.
#endif

#if !(TARGET_IS_NT50_OR_LATER)
#error You need a Windows 2000 or later to run this stub because it uses these features:
#error   /robust command line switch.
#error However, your C/C++ compilation flags indicate you intend to run this app on earlier systems.
#error This app will die there with the RPC_X_WRONG_STUB_VERSION error.
#endif


static const MIDL_PROC_FORMAT_STRING __MIDL_ProcFormatString =
    {
        0,
        {

	/* Procedure SetName */

			0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/*  2 */	NdrFcLong( 0x0 ),	/* 0 */
/*  6 */	NdrFcShort( 0x3 ),	/* 3 */
/*  8 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 10 */	NdrFcShort( 0x50 ),	/* 80 */
/* 12 */	NdrFcShort( 0x8 ),	/* 8 */
/* 14 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 16 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 18 */	NdrFcShort( 0x0 ),	/* 0 */
/* 20 */	NdrFcShort( 0x0 ),	/* 0 */
/* 22 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter name */

/* 24 */	NdrFcShort( 0xa ),	/* Flags:  must free, in, */
/* 26 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 28 */	NdrFcShort( 0x2 ),	/* Type Offset=2 */

	/* Return value */

/* 30 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 32 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 34 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetName */

/* 36 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 38 */	NdrFcLong( 0x0 ),	/* 0 */
/* 42 */	NdrFcShort( 0x4 ),	/* 4 */
/* 44 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 46 */	NdrFcShort( 0x50 ),	/* 80 */
/* 48 */	NdrFcShort( 0x8 ),	/* 8 */
/* 50 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 52 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 54 */	NdrFcShort( 0x0 ),	/* 0 */
/* 56 */	NdrFcShort( 0x0 ),	/* 0 */
/* 58 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter name */

/* 60 */	NdrFcShort( 0xa ),	/* Flags:  must free, in, */
/* 62 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 64 */	NdrFcShort( 0x2 ),	/* Type Offset=2 */

	/* Return value */

/* 66 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 68 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 70 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure SetConfig */

/* 72 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 74 */	NdrFcLong( 0x0 ),	/* 0 */
/* 78 */	NdrFcShort( 0x5 ),	/* 5 */
/* 80 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 82 */	NdrFcShort( 0x91 ),	/* 145 */
/* 84 */	NdrFcShort( 0x8 ),	/* 8 */
/* 86 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 88 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 90 */	NdrFcShort( 0x0 ),	/* 0 */
/* 92 */	NdrFcShort( 0x0 ),	/* 0 */
/* 94 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pConfig */

/* 96 */	NdrFcShort( 0x10a ),	/* Flags:  must free, in, simple ref, */
/* 98 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 100 */	NdrFcShort( 0xc ),	/* Type Offset=12 */

	/* Return value */

/* 102 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 104 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 106 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetConfig */

/* 108 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 110 */	NdrFcLong( 0x0 ),	/* 0 */
/* 114 */	NdrFcShort( 0x6 ),	/* 6 */
/* 116 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 118 */	NdrFcShort( 0x91 ),	/* 145 */
/* 120 */	NdrFcShort( 0x8 ),	/* 8 */
/* 122 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 124 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 126 */	NdrFcShort( 0x0 ),	/* 0 */
/* 128 */	NdrFcShort( 0x0 ),	/* 0 */
/* 130 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pConfig */

/* 132 */	NdrFcShort( 0x10a ),	/* Flags:  must free, in, simple ref, */
/* 134 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 136 */	NdrFcShort( 0xc ),	/* Type Offset=12 */

	/* Return value */

/* 138 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 140 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 142 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure EditConfig */

/* 144 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 146 */	NdrFcLong( 0x0 ),	/* 0 */
/* 150 */	NdrFcShort( 0x7 ),	/* 7 */
/* 152 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 154 */	NdrFcShort( 0x99 ),	/* 153 */
/* 156 */	NdrFcShort( 0x8 ),	/* 8 */
/* 158 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x3,		/* 3 */
/* 160 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 162 */	NdrFcShort( 0x0 ),	/* 0 */
/* 164 */	NdrFcShort( 0x0 ),	/* 0 */
/* 166 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pConfig */

/* 168 */	NdrFcShort( 0x10a ),	/* Flags:  must free, in, simple ref, */
/* 170 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 172 */	NdrFcShort( 0xc ),	/* Type Offset=12 */

	/* Parameter hWnd */

/* 174 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 176 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 178 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 180 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 182 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 184 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure EnableMonitor */

/* 186 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 188 */	NdrFcLong( 0x0 ),	/* 0 */
/* 192 */	NdrFcShort( 0x8 ),	/* 8 */
/* 194 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 196 */	NdrFcShort( 0x5 ),	/* 5 */
/* 198 */	NdrFcShort( 0x8 ),	/* 8 */
/* 200 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 202 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 204 */	NdrFcShort( 0x0 ),	/* 0 */
/* 206 */	NdrFcShort( 0x0 ),	/* 0 */
/* 208 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter bEnable */

/* 210 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 212 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 214 */	0x3,		/* FC_SMALL */
			0x0,		/* 0 */

	/* Return value */

/* 216 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 218 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 220 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetNumberOfConnections */

/* 222 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 224 */	NdrFcLong( 0x0 ),	/* 0 */
/* 228 */	NdrFcShort( 0x9 ),	/* 9 */
/* 230 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 232 */	NdrFcShort( 0x1c ),	/* 28 */
/* 234 */	NdrFcShort( 0x8 ),	/* 8 */
/* 236 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 238 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 240 */	NdrFcShort( 0x0 ),	/* 0 */
/* 242 */	NdrFcShort( 0x0 ),	/* 0 */
/* 244 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pNumber */

/* 246 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 248 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 250 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 252 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 254 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 256 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure SetConnection */

/* 258 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 260 */	NdrFcLong( 0x0 ),	/* 0 */
/* 264 */	NdrFcShort( 0xa ),	/* 10 */
/* 266 */	NdrFcShort( 0x54 ),	/* x86 Stack size/offset = 84 */
/* 268 */	NdrFcShort( 0x8 ),	/* 8 */
/* 270 */	NdrFcShort( 0x8 ),	/* 8 */
/* 272 */	0x46,		/* Oi2 Flags:  clt must size, has return, has ext, */
			0x3,		/* 3 */
/* 274 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 276 */	NdrFcShort( 0x0 ),	/* 0 */
/* 278 */	NdrFcShort( 0x0 ),	/* 0 */
/* 280 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter n */

/* 282 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 284 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 286 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter connection */

/* 288 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 290 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 292 */	NdrFcShort( 0x3a ),	/* Type Offset=58 */

	/* Return value */

/* 294 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 296 */	NdrFcShort( 0x50 ),	/* x86 Stack size/offset = 80 */
/* 298 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetConnection */

/* 300 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 302 */	NdrFcLong( 0x0 ),	/* 0 */
/* 306 */	NdrFcShort( 0xb ),	/* 11 */
/* 308 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 310 */	NdrFcShort( 0x8 ),	/* 8 */
/* 312 */	NdrFcShort( 0x8 ),	/* 8 */
/* 314 */	0x46,		/* Oi2 Flags:  clt must size, has return, has ext, */
			0x3,		/* 3 */
/* 316 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 318 */	NdrFcShort( 0x0 ),	/* 0 */
/* 320 */	NdrFcShort( 0x0 ),	/* 0 */
/* 322 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter n */

/* 324 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 326 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 328 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pConnection */

/* 330 */	NdrFcShort( 0x10b ),	/* Flags:  must size, must free, in, simple ref, */
/* 332 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 334 */	NdrFcShort( 0x3a ),	/* Type Offset=58 */

	/* Return value */

/* 336 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 338 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 340 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetIcon */

/* 342 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 344 */	NdrFcLong( 0x0 ),	/* 0 */
/* 348 */	NdrFcShort( 0xc ),	/* 12 */
/* 350 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 352 */	NdrFcShort( 0x24 ),	/* 36 */
/* 354 */	NdrFcShort( 0x8 ),	/* 8 */
/* 356 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x3,		/* 3 */
/* 358 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 360 */	NdrFcShort( 0x0 ),	/* 0 */
/* 362 */	NdrFcShort( 0x0 ),	/* 0 */
/* 364 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter n */

/* 366 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 368 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 370 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pHICON */

/* 372 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 374 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 376 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 378 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 380 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 382 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetClassName */

/* 384 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 386 */	NdrFcLong( 0x0 ),	/* 0 */
/* 390 */	NdrFcShort( 0xd ),	/* 13 */
/* 392 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 394 */	NdrFcShort( 0x50 ),	/* 80 */
/* 396 */	NdrFcShort( 0x8 ),	/* 8 */
/* 398 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 400 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 402 */	NdrFcShort( 0x0 ),	/* 0 */
/* 404 */	NdrFcShort( 0x0 ),	/* 0 */
/* 406 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter name */

/* 408 */	NdrFcShort( 0xa ),	/* Flags:  must free, in, */
/* 410 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 412 */	NdrFcShort( 0x2 ),	/* Type Offset=2 */

	/* Return value */

/* 414 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 416 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 418 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure SyncExec */

/* 420 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 422 */	NdrFcLong( 0x0 ),	/* 0 */
/* 426 */	NdrFcShort( 0xe ),	/* 14 */
/* 428 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 430 */	NdrFcShort( 0x0 ),	/* 0 */
/* 432 */	NdrFcShort( 0x8 ),	/* 8 */
/* 434 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x1,		/* 1 */
/* 436 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 438 */	NdrFcShort( 0x0 ),	/* 0 */
/* 440 */	NdrFcShort( 0x0 ),	/* 0 */
/* 442 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Return value */

/* 444 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 446 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 448 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure AsyncExec */

/* 450 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 452 */	NdrFcLong( 0x0 ),	/* 0 */
/* 456 */	NdrFcShort( 0xf ),	/* 15 */
/* 458 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 460 */	NdrFcShort( 0x8 ),	/* 8 */
/* 462 */	NdrFcShort( 0x8 ),	/* 8 */
/* 464 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 466 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 468 */	NdrFcShort( 0x0 ),	/* 0 */
/* 470 */	NdrFcShort( 0x0 ),	/* 0 */
/* 472 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter HNotifyStatusChangeEvent */

/* 474 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 476 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 478 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 480 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 482 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 484 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure AsyncCheckStatus */

/* 486 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 488 */	NdrFcLong( 0x0 ),	/* 0 */
/* 492 */	NdrFcShort( 0x10 ),	/* 16 */
/* 494 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 496 */	NdrFcShort( 0x1c ),	/* 28 */
/* 498 */	NdrFcShort( 0x8 ),	/* 8 */
/* 500 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 502 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 504 */	NdrFcShort( 0x0 ),	/* 0 */
/* 506 */	NdrFcShort( 0x0 ),	/* 0 */
/* 508 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pStatus */

/* 510 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 512 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 514 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 516 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 518 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 520 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure AsyncWait */

/* 522 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 524 */	NdrFcLong( 0x0 ),	/* 0 */
/* 528 */	NdrFcShort( 0x11 ),	/* 17 */
/* 530 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 532 */	NdrFcShort( 0x0 ),	/* 0 */
/* 534 */	NdrFcShort( 0x8 ),	/* 8 */
/* 536 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x1,		/* 1 */
/* 538 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 540 */	NdrFcShort( 0x0 ),	/* 0 */
/* 542 */	NdrFcShort( 0x0 ),	/* 0 */
/* 544 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Return value */

/* 546 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 548 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 550 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure AsyncPause */

/* 552 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 554 */	NdrFcLong( 0x0 ),	/* 0 */
/* 558 */	NdrFcShort( 0x12 ),	/* 18 */
/* 560 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 562 */	NdrFcShort( 0x0 ),	/* 0 */
/* 564 */	NdrFcShort( 0x8 ),	/* 8 */
/* 566 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x1,		/* 1 */
/* 568 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 570 */	NdrFcShort( 0x0 ),	/* 0 */
/* 572 */	NdrFcShort( 0x0 ),	/* 0 */
/* 574 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Return value */

/* 576 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 578 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 580 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure AsyncStop */

/* 582 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 584 */	NdrFcLong( 0x0 ),	/* 0 */
/* 588 */	NdrFcShort( 0x13 ),	/* 19 */
/* 590 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 592 */	NdrFcShort( 0x0 ),	/* 0 */
/* 594 */	NdrFcShort( 0x8 ),	/* 8 */
/* 596 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x1,		/* 1 */
/* 598 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 600 */	NdrFcShort( 0x0 ),	/* 0 */
/* 602 */	NdrFcShort( 0x0 ),	/* 0 */
/* 604 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Return value */

/* 606 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 608 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 610 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure AsyncResume */

/* 612 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 614 */	NdrFcLong( 0x0 ),	/* 0 */
/* 618 */	NdrFcShort( 0x14 ),	/* 20 */
/* 620 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 622 */	NdrFcShort( 0x0 ),	/* 0 */
/* 624 */	NdrFcShort( 0x8 ),	/* 8 */
/* 626 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x1,		/* 1 */
/* 628 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 630 */	NdrFcShort( 0x0 ),	/* 0 */
/* 632 */	NdrFcShort( 0x0 ),	/* 0 */
/* 634 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Return value */

/* 636 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 638 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 640 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure IsReady */

/* 642 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 644 */	NdrFcLong( 0x0 ),	/* 0 */
/* 648 */	NdrFcShort( 0x15 ),	/* 21 */
/* 650 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 652 */	NdrFcShort( 0x38 ),	/* 56 */
/* 654 */	NdrFcShort( 0x8 ),	/* 8 */
/* 656 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x3,		/* 3 */
/* 658 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 660 */	NdrFcShort( 0x0 ),	/* 0 */
/* 662 */	NdrFcShort( 0x0 ),	/* 0 */
/* 664 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pWrongConn */

/* 666 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 668 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 670 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pRetCode */

/* 672 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 674 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 676 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 678 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 680 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 682 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure Write */

/* 684 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 686 */	NdrFcLong( 0x0 ),	/* 0 */
/* 690 */	NdrFcShort( 0x3 ),	/* 3 */
/* 692 */	NdrFcShort( 0x18 ),	/* x86 Stack size/offset = 24 */
/* 694 */	NdrFcShort( 0x56 ),	/* 86 */
/* 696 */	NdrFcShort( 0x8 ),	/* 8 */
/* 698 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x5,		/* 5 */
/* 700 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 702 */	NdrFcShort( 0x0 ),	/* 0 */
/* 704 */	NdrFcShort( 0x0 ),	/* 0 */
/* 706 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter hWnd */

/* 708 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 710 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 712 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pData */

/* 714 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 716 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 718 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Parameter pExtErrorInfo */

/* 720 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 722 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 724 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter FileName */

/* 726 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 728 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 730 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Return value */

/* 732 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 734 */	NdrFcShort( 0x14 ),	/* x86 Stack size/offset = 20 */
/* 736 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure Read */

/* 738 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 740 */	NdrFcLong( 0x0 ),	/* 0 */
/* 744 */	NdrFcShort( 0x4 ),	/* 4 */
/* 746 */	NdrFcShort( 0x18 ),	/* x86 Stack size/offset = 24 */
/* 748 */	NdrFcShort( 0x56 ),	/* 86 */
/* 750 */	NdrFcShort( 0x8 ),	/* 8 */
/* 752 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x5,		/* 5 */
/* 754 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 756 */	NdrFcShort( 0x0 ),	/* 0 */
/* 758 */	NdrFcShort( 0x0 ),	/* 0 */
/* 760 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter hWnd */

/* 762 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 764 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 766 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pData */

/* 768 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 770 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 772 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Parameter pExtErrorInfo */

/* 774 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 776 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 778 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter FileName */

/* 780 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 782 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 784 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Return value */

/* 786 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 788 */	NdrFcShort( 0x14 ),	/* x86 Stack size/offset = 20 */
/* 790 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure GetStatus */

/* 792 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 794 */	NdrFcLong( 0x0 ),	/* 0 */
/* 798 */	NdrFcShort( 0x5 ),	/* 5 */
/* 800 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 802 */	NdrFcShort( 0x1c ),	/* 28 */
/* 804 */	NdrFcShort( 0x8 ),	/* 8 */
/* 806 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x2,		/* 2 */
/* 808 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 810 */	NdrFcShort( 0x0 ),	/* 0 */
/* 812 */	NdrFcShort( 0x0 ),	/* 0 */
/* 814 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pExtErrorInfo */

/* 816 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 818 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 820 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Return value */

/* 822 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 824 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 826 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure Write2 */

/* 828 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 830 */	NdrFcLong( 0x0 ),	/* 0 */
/* 834 */	NdrFcShort( 0x6 ),	/* 6 */
/* 836 */	NdrFcShort( 0x18 ),	/* x86 Stack size/offset = 24 */
/* 838 */	NdrFcShort( 0x56 ),	/* 86 */
/* 840 */	NdrFcShort( 0x8 ),	/* 8 */
/* 842 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x5,		/* 5 */
/* 844 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 846 */	NdrFcShort( 0x0 ),	/* 0 */
/* 848 */	NdrFcShort( 0x0 ),	/* 0 */
/* 850 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter hWnd */

/* 852 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 854 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 856 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pData */

/* 858 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 860 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 862 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Parameter pExtErrorInfo */

/* 864 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 866 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 868 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter FileName */

/* 870 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 872 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 874 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Return value */

/* 876 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 878 */	NdrFcShort( 0x14 ),	/* x86 Stack size/offset = 20 */
/* 880 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure Read2 */

/* 882 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 884 */	NdrFcLong( 0x0 ),	/* 0 */
/* 888 */	NdrFcShort( 0x7 ),	/* 7 */
/* 890 */	NdrFcShort( 0x18 ),	/* x86 Stack size/offset = 24 */
/* 892 */	NdrFcShort( 0x56 ),	/* 86 */
/* 894 */	NdrFcShort( 0x8 ),	/* 8 */
/* 896 */	0x44,		/* Oi2 Flags:  has return, has ext, */
			0x5,		/* 5 */
/* 898 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 900 */	NdrFcShort( 0x0 ),	/* 0 */
/* 902 */	NdrFcShort( 0x0 ),	/* 0 */
/* 904 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter hWnd */

/* 906 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 908 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 910 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter pData */

/* 912 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 914 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 916 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Parameter pExtErrorInfo */

/* 918 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 920 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 922 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter FileName */

/* 924 */	NdrFcShort( 0x148 ),	/* Flags:  in, base type, simple ref, */
/* 926 */	NdrFcShort( 0x10 ),	/* x86 Stack size/offset = 16 */
/* 928 */	0x1,		/* FC_BYTE */
			0x0,		/* 0 */

	/* Return value */

/* 930 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 932 */	NdrFcShort( 0x14 ),	/* x86 Stack size/offset = 20 */
/* 934 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

			0x0
        }
    };

static const MIDL_TYPE_FORMAT_STRING __MIDL_TypeFormatString =
    {
        0,
        {
			NdrFcShort( 0x0 ),	/* 0 */
/*  2 */	
			0x1d,		/* FC_SMFARRAY */
			0x0,		/* 0 */
/*  4 */	NdrFcShort( 0x40 ),	/* 64 */
/*  6 */	0x2,		/* FC_CHAR */
			0x5b,		/* FC_END */
/*  8 */	
			0x11, 0x0,	/* FC_RP */
/* 10 */	NdrFcShort( 0x2 ),	/* Offset= 2 (12) */
/* 12 */	
			0x16,		/* FC_PSTRUCT */
			0x3,		/* 3 */
/* 14 */	NdrFcShort( 0x48 ),	/* 72 */
/* 16 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 18 */	
			0x46,		/* FC_NO_REPEAT */
			0x5c,		/* FC_PAD */
/* 20 */	NdrFcShort( 0x44 ),	/* 68 */
/* 22 */	NdrFcShort( 0x44 ),	/* 68 */
/* 24 */	0x12, 0x8,	/* FC_UP [simple_pointer] */
/* 26 */	0x2,		/* FC_CHAR */
			0x5c,		/* FC_PAD */
/* 28 */	
			0x5b,		/* FC_END */

			0x4c,		/* FC_EMBEDDED_COMPLEX */
/* 30 */	0x0,		/* 0 */
			NdrFcShort( 0xffe3 ),	/* Offset= -29 (2) */
			0x8,		/* FC_LONG */
/* 34 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 36 */	
			0x11, 0x8,	/* FC_RP [simple_pointer] */
/* 38 */	0x8,		/* FC_LONG */
			0x5c,		/* FC_PAD */
/* 40 */	
			0x2f,		/* FC_IP */
			0x5a,		/* FC_CONSTANT_IID */
/* 42 */	NdrFcLong( 0x0 ),	/* 0 */
/* 46 */	NdrFcShort( 0x0 ),	/* 0 */
/* 48 */	NdrFcShort( 0x0 ),	/* 0 */
/* 50 */	0xc0,		/* 192 */
			0x0,		/* 0 */
/* 52 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 54 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 56 */	0x0,		/* 0 */
			0x46,		/* 70 */
/* 58 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 60 */	NdrFcShort( 0x48 ),	/* 72 */
/* 62 */	NdrFcShort( 0x0 ),	/* 0 */
/* 64 */	NdrFcShort( 0xc ),	/* Offset= 12 (76) */
/* 66 */	0x8,		/* FC_LONG */
			0x4c,		/* FC_EMBEDDED_COMPLEX */
/* 68 */	0x0,		/* 0 */
			NdrFcShort( 0xffbd ),	/* Offset= -67 (2) */
			0x4c,		/* FC_EMBEDDED_COMPLEX */
/* 72 */	0x0,		/* 0 */
			NdrFcShort( 0xffdf ),	/* Offset= -33 (40) */
			0x5b,		/* FC_END */
/* 76 */	
			0x11, 0x0,	/* FC_RP */
/* 78 */	NdrFcShort( 0xffec ),	/* Offset= -20 (58) */
/* 80 */	
			0x11, 0x8,	/* FC_RP [simple_pointer] */
/* 82 */	0x1,		/* FC_BYTE */
			0x5c,		/* FC_PAD */

			0x0
        }
    };


/* Object interface: IUnknown, ver. 0.0,
   GUID={0x00000000,0x0000,0x0000,{0xC0,0x00,0x00,0x00,0x00,0x00,0x00,0x46}} */


/* Object interface: ISySalObject, ver. 0.0,
   GUID={0xC022EEAD,0x748A,0x11D3,{0xA4,0x7B,0xE8,0x9C,0x0B,0xCE,0x97,0x20}} */

#pragma code_seg(".orpc")
static const unsigned short ISySalObject_FormatStringOffsetTable[] =
    {
    0,
    36,
    72,
    108,
    144,
    186,
    222,
    258,
    300,
    342,
    384,
    420,
    450,
    486,
    522,
    552,
    582,
    612,
    642
    };

static const MIDL_STUBLESS_PROXY_INFO ISySalObject_ProxyInfo =
    {
    &Object_StubDesc,
    __MIDL_ProcFormatString.Format,
    &ISySalObject_FormatStringOffsetTable[-3],
    0,
    0,
    0
    };


static const MIDL_SERVER_INFO ISySalObject_ServerInfo = 
    {
    &Object_StubDesc,
    0,
    __MIDL_ProcFormatString.Format,
    &ISySalObject_FormatStringOffsetTable[-3],
    0,
    0,
    0,
    0};
CINTERFACE_PROXY_VTABLE(22) _ISySalObjectProxyVtbl = 
{
    &ISySalObject_ProxyInfo,
    &IID_ISySalObject,
    IUnknown_QueryInterface_Proxy,
    IUnknown_AddRef_Proxy,
    IUnknown_Release_Proxy ,
    (void *) (INT_PTR) -1 /* ISySalObject::SetName */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::GetName */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::SetConfig */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::GetConfig */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::EditConfig */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::EnableMonitor */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::GetNumberOfConnections */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::SetConnection */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::GetConnection */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::GetIcon */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::GetClassName */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::SyncExec */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::AsyncExec */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::AsyncCheckStatus */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::AsyncWait */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::AsyncPause */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::AsyncStop */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::AsyncResume */ ,
    (void *) (INT_PTR) -1 /* ISySalObject::IsReady */
};

const CInterfaceStubVtbl _ISySalObjectStubVtbl =
{
    &IID_ISySalObject,
    &ISySalObject_ServerInfo,
    22,
    0, /* pure interpreted */
    CStdStubBuffer_METHODS
};


/* Object interface: ISySalDataIO, ver. 0.0,
   GUID={0x4B47E8CD,0x894C,0x11D3,{0xA4,0x7C,0x9F,0x37,0x35,0x22,0x60,0x36}} */

#pragma code_seg(".orpc")
static const unsigned short ISySalDataIO_FormatStringOffsetTable[] =
    {
    684,
    738,
    792
    };

static const MIDL_STUBLESS_PROXY_INFO ISySalDataIO_ProxyInfo =
    {
    &Object_StubDesc,
    __MIDL_ProcFormatString.Format,
    &ISySalDataIO_FormatStringOffsetTable[-3],
    0,
    0,
    0
    };


static const MIDL_SERVER_INFO ISySalDataIO_ServerInfo = 
    {
    &Object_StubDesc,
    0,
    __MIDL_ProcFormatString.Format,
    &ISySalDataIO_FormatStringOffsetTable[-3],
    0,
    0,
    0,
    0};
CINTERFACE_PROXY_VTABLE(6) _ISySalDataIOProxyVtbl = 
{
    &ISySalDataIO_ProxyInfo,
    &IID_ISySalDataIO,
    IUnknown_QueryInterface_Proxy,
    IUnknown_AddRef_Proxy,
    IUnknown_Release_Proxy ,
    (void *) (INT_PTR) -1 /* ISySalDataIO::Write */ ,
    (void *) (INT_PTR) -1 /* ISySalDataIO::Read */ ,
    (void *) (INT_PTR) -1 /* ISySalDataIO::GetStatus */
};

const CInterfaceStubVtbl _ISySalDataIOStubVtbl =
{
    &IID_ISySalDataIO,
    &ISySalDataIO_ServerInfo,
    6,
    0, /* pure interpreted */
    CStdStubBuffer_METHODS
};


/* Object interface: ISySalDataIO2, ver. 0.0,
   GUID={0x5892a1f5,0x5dd9,0x4fe3,{0xa0,0x24,0xd8,0xb6,0x24,0x91,0x7c,0x1d}} */

#pragma code_seg(".orpc")
static const unsigned short ISySalDataIO2_FormatStringOffsetTable[] =
    {
    684,
    738,
    792,
    828,
    882
    };

static const MIDL_STUBLESS_PROXY_INFO ISySalDataIO2_ProxyInfo =
    {
    &Object_StubDesc,
    __MIDL_ProcFormatString.Format,
    &ISySalDataIO2_FormatStringOffsetTable[-3],
    0,
    0,
    0
    };


static const MIDL_SERVER_INFO ISySalDataIO2_ServerInfo = 
    {
    &Object_StubDesc,
    0,
    __MIDL_ProcFormatString.Format,
    &ISySalDataIO2_FormatStringOffsetTable[-3],
    0,
    0,
    0,
    0};
CINTERFACE_PROXY_VTABLE(8) _ISySalDataIO2ProxyVtbl = 
{
    &ISySalDataIO2_ProxyInfo,
    &IID_ISySalDataIO2,
    IUnknown_QueryInterface_Proxy,
    IUnknown_AddRef_Proxy,
    IUnknown_Release_Proxy ,
    (void *) (INT_PTR) -1 /* ISySalDataIO::Write */ ,
    (void *) (INT_PTR) -1 /* ISySalDataIO::Read */ ,
    (void *) (INT_PTR) -1 /* ISySalDataIO::GetStatus */ ,
    (void *) (INT_PTR) -1 /* ISySalDataIO2::Write2 */ ,
    (void *) (INT_PTR) -1 /* ISySalDataIO2::Read2 */
};

const CInterfaceStubVtbl _ISySalDataIO2StubVtbl =
{
    &IID_ISySalDataIO2,
    &ISySalDataIO2_ServerInfo,
    8,
    0, /* pure interpreted */
    CStdStubBuffer_METHODS
};

static const MIDL_STUB_DESC Object_StubDesc = 
    {
    0,
    NdrOleAllocate,
    NdrOleFree,
    0,
    0,
    0,
    0,
    0,
    __MIDL_TypeFormatString.Format,
    1, /* -error bounds_check flag */
    0x50002, /* Ndr library version */
    0,
    0x600016e, /* MIDL Version 6.0.366 */
    0,
    0,
    0,  /* notify & notify_flag routine table */
    0x1, /* MIDL flag */
    0, /* cs routines */
    0,   /* proxy/server info */
    0   /* Reserved5 */
    };

const CInterfaceProxyVtbl * _SySalDataIO_ProxyVtblList[] = 
{
    ( CInterfaceProxyVtbl *) &_ISySalObjectProxyVtbl,
    ( CInterfaceProxyVtbl *) &_ISySalDataIOProxyVtbl,
    ( CInterfaceProxyVtbl *) &_ISySalDataIO2ProxyVtbl,
    0
};

const CInterfaceStubVtbl * _SySalDataIO_StubVtblList[] = 
{
    ( CInterfaceStubVtbl *) &_ISySalObjectStubVtbl,
    ( CInterfaceStubVtbl *) &_ISySalDataIOStubVtbl,
    ( CInterfaceStubVtbl *) &_ISySalDataIO2StubVtbl,
    0
};

PCInterfaceName const _SySalDataIO_InterfaceNamesList[] = 
{
    "ISySalObject",
    "ISySalDataIO",
    "ISySalDataIO2",
    0
};


#define _SySalDataIO_CHECK_IID(n)	IID_GENERIC_CHECK_IID( _SySalDataIO, pIID, n)

int __stdcall _SySalDataIO_IID_Lookup( const IID * pIID, int * pIndex )
{
    IID_BS_LOOKUP_SETUP

    IID_BS_LOOKUP_INITIAL_TEST( _SySalDataIO, 3, 2 )
    IID_BS_LOOKUP_NEXT_TEST( _SySalDataIO, 1 )
    IID_BS_LOOKUP_RETURN_RESULT( _SySalDataIO, 3, *pIndex )
    
}

const ExtendedProxyFileInfo SySalDataIO_ProxyFileInfo = 
{
    (PCInterfaceProxyVtblList *) & _SySalDataIO_ProxyVtblList,
    (PCInterfaceStubVtblList *) & _SySalDataIO_StubVtblList,
    (const PCInterfaceName * ) & _SySalDataIO_InterfaceNamesList,
    0, // no delegation
    & _SySalDataIO_IID_Lookup, 
    3,
    2,
    0, /* table of [async_uuid] interfaces */
    0, /* Filler1 */
    0, /* Filler2 */
    0  /* Filler3 */
};
#pragma optimize("", on )
#if _MSC_VER >= 1200
#pragma warning(pop)
#endif


#endif /* !defined(_M_IA64) && !defined(_M_AMD64)*/

