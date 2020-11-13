

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


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


/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __SySalDataIO_h__
#define __SySalDataIO_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __ISySalObject_FWD_DEFINED__
#define __ISySalObject_FWD_DEFINED__
typedef interface ISySalObject ISySalObject;
#endif 	/* __ISySalObject_FWD_DEFINED__ */


#ifndef __ISySalDataIO_FWD_DEFINED__
#define __ISySalDataIO_FWD_DEFINED__
typedef interface ISySalDataIO ISySalDataIO;
#endif 	/* __ISySalDataIO_FWD_DEFINED__ */


#ifndef __ISySalDataIO2_FWD_DEFINED__
#define __ISySalDataIO2_FWD_DEFINED__
typedef interface ISySalDataIO2 ISySalDataIO2;
#endif 	/* __ISySalDataIO2_FWD_DEFINED__ */


#ifndef __SySalDataIO_FWD_DEFINED__
#define __SySalDataIO_FWD_DEFINED__

#ifdef __cplusplus
typedef class SySalDataIO SySalDataIO;
#else
typedef struct SySalDataIO SySalDataIO;
#endif /* __cplusplus */

#endif 	/* __SySalDataIO_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "Connection.h"
#include "Config.h"
#include "GUIdefs.h"
#include "datastructs.h"

#ifdef __cplusplus
extern "C"{
#endif 

void * __RPC_USER MIDL_user_allocate(size_t);
void __RPC_USER MIDL_user_free( void * ); 

#ifndef __ISySalObject_INTERFACE_DEFINED__
#define __ISySalObject_INTERFACE_DEFINED__

/* interface ISySalObject */
/* [unique][helpstring][uuid][object] */ 


EXTERN_C const IID IID_ISySalObject;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("C022EEAD-748A-11D3-A47B-E89C0BCE9720")
    ISySalObject : public IUnknown
    {
    public:
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE SetName( 
            ConnectionName name) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetName( 
            ConnectionName name) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE SetConfig( 
            SySalConfig *pConfig) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetConfig( 
            SySalConfig *pConfig) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE EditConfig( 
            SySalConfig *pConfig,
            HSySalHANDLE hWnd) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE EnableMonitor( 
            boolean bEnable) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetNumberOfConnections( 
            int *pNumber) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE SetConnection( 
            int n,
            SySalConnection connection) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetConnection( 
            int n,
            SySalConnection *pConnection) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetIcon( 
            int n,
            HSySalHICON *pHICON) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetClassName( 
            ConnectionName name) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE SyncExec( void) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE AsyncExec( 
            HSySalHANDLE HNotifyStatusChangeEvent) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE AsyncCheckStatus( 
            long *pStatus) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE AsyncWait( void) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE AsyncPause( void) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE AsyncStop( void) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE AsyncResume( void) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE IsReady( 
            long *pWrongConn,
            long *pRetCode) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct ISySalObjectVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ISySalObject * This,
            /* [in] */ REFIID riid,
            /* [iid_is][out] */ void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ISySalObject * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ISySalObject * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ISySalObject * This,
            ConnectionName name);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetName )( 
            ISySalObject * This,
            ConnectionName name);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *SetConfig )( 
            ISySalObject * This,
            SySalConfig *pConfig);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetConfig )( 
            ISySalObject * This,
            SySalConfig *pConfig);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *EditConfig )( 
            ISySalObject * This,
            SySalConfig *pConfig,
            HSySalHANDLE hWnd);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *EnableMonitor )( 
            ISySalObject * This,
            boolean bEnable);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetNumberOfConnections )( 
            ISySalObject * This,
            int *pNumber);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *SetConnection )( 
            ISySalObject * This,
            int n,
            SySalConnection connection);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetConnection )( 
            ISySalObject * This,
            int n,
            SySalConnection *pConnection);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetIcon )( 
            ISySalObject * This,
            int n,
            HSySalHICON *pHICON);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetClassName )( 
            ISySalObject * This,
            ConnectionName name);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *SyncExec )( 
            ISySalObject * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *AsyncExec )( 
            ISySalObject * This,
            HSySalHANDLE HNotifyStatusChangeEvent);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *AsyncCheckStatus )( 
            ISySalObject * This,
            long *pStatus);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *AsyncWait )( 
            ISySalObject * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *AsyncPause )( 
            ISySalObject * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *AsyncStop )( 
            ISySalObject * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *AsyncResume )( 
            ISySalObject * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *IsReady )( 
            ISySalObject * This,
            long *pWrongConn,
            long *pRetCode);
        
        END_INTERFACE
    } ISySalObjectVtbl;

    interface ISySalObject
    {
        CONST_VTBL struct ISySalObjectVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ISySalObject_QueryInterface(This,riid,ppvObject)	\
    (This)->lpVtbl -> QueryInterface(This,riid,ppvObject)

#define ISySalObject_AddRef(This)	\
    (This)->lpVtbl -> AddRef(This)

#define ISySalObject_Release(This)	\
    (This)->lpVtbl -> Release(This)


#define ISySalObject_SetName(This,name)	\
    (This)->lpVtbl -> SetName(This,name)

#define ISySalObject_GetName(This,name)	\
    (This)->lpVtbl -> GetName(This,name)

#define ISySalObject_SetConfig(This,pConfig)	\
    (This)->lpVtbl -> SetConfig(This,pConfig)

#define ISySalObject_GetConfig(This,pConfig)	\
    (This)->lpVtbl -> GetConfig(This,pConfig)

#define ISySalObject_EditConfig(This,pConfig,hWnd)	\
    (This)->lpVtbl -> EditConfig(This,pConfig,hWnd)

#define ISySalObject_EnableMonitor(This,bEnable)	\
    (This)->lpVtbl -> EnableMonitor(This,bEnable)

#define ISySalObject_GetNumberOfConnections(This,pNumber)	\
    (This)->lpVtbl -> GetNumberOfConnections(This,pNumber)

#define ISySalObject_SetConnection(This,n,connection)	\
    (This)->lpVtbl -> SetConnection(This,n,connection)

#define ISySalObject_GetConnection(This,n,pConnection)	\
    (This)->lpVtbl -> GetConnection(This,n,pConnection)

#define ISySalObject_GetIcon(This,n,pHICON)	\
    (This)->lpVtbl -> GetIcon(This,n,pHICON)

#define ISySalObject_GetClassName(This,name)	\
    (This)->lpVtbl -> GetClassName(This,name)

#define ISySalObject_SyncExec(This)	\
    (This)->lpVtbl -> SyncExec(This)

#define ISySalObject_AsyncExec(This,HNotifyStatusChangeEvent)	\
    (This)->lpVtbl -> AsyncExec(This,HNotifyStatusChangeEvent)

#define ISySalObject_AsyncCheckStatus(This,pStatus)	\
    (This)->lpVtbl -> AsyncCheckStatus(This,pStatus)

#define ISySalObject_AsyncWait(This)	\
    (This)->lpVtbl -> AsyncWait(This)

#define ISySalObject_AsyncPause(This)	\
    (This)->lpVtbl -> AsyncPause(This)

#define ISySalObject_AsyncStop(This)	\
    (This)->lpVtbl -> AsyncStop(This)

#define ISySalObject_AsyncResume(This)	\
    (This)->lpVtbl -> AsyncResume(This)

#define ISySalObject_IsReady(This,pWrongConn,pRetCode)	\
    (This)->lpVtbl -> IsReady(This,pWrongConn,pRetCode)

#endif /* COBJMACROS */


#endif 	/* C style interface */



/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_SetName_Proxy( 
    ISySalObject * This,
    ConnectionName name);


void __RPC_STUB ISySalObject_SetName_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_GetName_Proxy( 
    ISySalObject * This,
    ConnectionName name);


void __RPC_STUB ISySalObject_GetName_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_SetConfig_Proxy( 
    ISySalObject * This,
    SySalConfig *pConfig);


void __RPC_STUB ISySalObject_SetConfig_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_GetConfig_Proxy( 
    ISySalObject * This,
    SySalConfig *pConfig);


void __RPC_STUB ISySalObject_GetConfig_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_EditConfig_Proxy( 
    ISySalObject * This,
    SySalConfig *pConfig,
    HSySalHANDLE hWnd);


void __RPC_STUB ISySalObject_EditConfig_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_EnableMonitor_Proxy( 
    ISySalObject * This,
    boolean bEnable);


void __RPC_STUB ISySalObject_EnableMonitor_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_GetNumberOfConnections_Proxy( 
    ISySalObject * This,
    int *pNumber);


void __RPC_STUB ISySalObject_GetNumberOfConnections_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_SetConnection_Proxy( 
    ISySalObject * This,
    int n,
    SySalConnection connection);


void __RPC_STUB ISySalObject_SetConnection_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_GetConnection_Proxy( 
    ISySalObject * This,
    int n,
    SySalConnection *pConnection);


void __RPC_STUB ISySalObject_GetConnection_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_GetIcon_Proxy( 
    ISySalObject * This,
    int n,
    HSySalHICON *pHICON);


void __RPC_STUB ISySalObject_GetIcon_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_GetClassName_Proxy( 
    ISySalObject * This,
    ConnectionName name);


void __RPC_STUB ISySalObject_GetClassName_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_SyncExec_Proxy( 
    ISySalObject * This);


void __RPC_STUB ISySalObject_SyncExec_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_AsyncExec_Proxy( 
    ISySalObject * This,
    HSySalHANDLE HNotifyStatusChangeEvent);


void __RPC_STUB ISySalObject_AsyncExec_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_AsyncCheckStatus_Proxy( 
    ISySalObject * This,
    long *pStatus);


void __RPC_STUB ISySalObject_AsyncCheckStatus_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_AsyncWait_Proxy( 
    ISySalObject * This);


void __RPC_STUB ISySalObject_AsyncWait_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_AsyncPause_Proxy( 
    ISySalObject * This);


void __RPC_STUB ISySalObject_AsyncPause_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_AsyncStop_Proxy( 
    ISySalObject * This);


void __RPC_STUB ISySalObject_AsyncStop_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_AsyncResume_Proxy( 
    ISySalObject * This);


void __RPC_STUB ISySalObject_AsyncResume_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalObject_IsReady_Proxy( 
    ISySalObject * This,
    long *pWrongConn,
    long *pRetCode);


void __RPC_STUB ISySalObject_IsReady_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);



#endif 	/* __ISySalObject_INTERFACE_DEFINED__ */


#ifndef __ISySalDataIO_INTERFACE_DEFINED__
#define __ISySalDataIO_INTERFACE_DEFINED__

/* interface ISySalDataIO */
/* [unique][helpstring][uuid][object] */ 


EXTERN_C const IID IID_ISySalDataIO;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4B47E8CD-894C-11D3-A47C-9F3735226036")
    ISySalDataIO : public IUnknown
    {
    public:
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE Write( 
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE Read( 
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE GetStatus( 
            UINT *pExtErrorInfo) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct ISySalDataIOVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ISySalDataIO * This,
            /* [in] */ REFIID riid,
            /* [iid_is][out] */ void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ISySalDataIO * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ISySalDataIO * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *Write )( 
            ISySalDataIO * This,
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *Read )( 
            ISySalDataIO * This,
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetStatus )( 
            ISySalDataIO * This,
            UINT *pExtErrorInfo);
        
        END_INTERFACE
    } ISySalDataIOVtbl;

    interface ISySalDataIO
    {
        CONST_VTBL struct ISySalDataIOVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ISySalDataIO_QueryInterface(This,riid,ppvObject)	\
    (This)->lpVtbl -> QueryInterface(This,riid,ppvObject)

#define ISySalDataIO_AddRef(This)	\
    (This)->lpVtbl -> AddRef(This)

#define ISySalDataIO_Release(This)	\
    (This)->lpVtbl -> Release(This)


#define ISySalDataIO_Write(This,hWnd,pData,pExtErrorInfo,FileName)	\
    (This)->lpVtbl -> Write(This,hWnd,pData,pExtErrorInfo,FileName)

#define ISySalDataIO_Read(This,hWnd,pData,pExtErrorInfo,FileName)	\
    (This)->lpVtbl -> Read(This,hWnd,pData,pExtErrorInfo,FileName)

#define ISySalDataIO_GetStatus(This,pExtErrorInfo)	\
    (This)->lpVtbl -> GetStatus(This,pExtErrorInfo)

#endif /* COBJMACROS */


#endif 	/* C style interface */



/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalDataIO_Write_Proxy( 
    ISySalDataIO * This,
    HSySalHANDLE hWnd,
    BYTE *pData,
    UINT *pExtErrorInfo,
    BYTE *FileName);


void __RPC_STUB ISySalDataIO_Write_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalDataIO_Read_Proxy( 
    ISySalDataIO * This,
    HSySalHANDLE hWnd,
    BYTE *pData,
    UINT *pExtErrorInfo,
    BYTE *FileName);


void __RPC_STUB ISySalDataIO_Read_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalDataIO_GetStatus_Proxy( 
    ISySalDataIO * This,
    UINT *pExtErrorInfo);


void __RPC_STUB ISySalDataIO_GetStatus_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);



#endif 	/* __ISySalDataIO_INTERFACE_DEFINED__ */


#ifndef __ISySalDataIO2_INTERFACE_DEFINED__
#define __ISySalDataIO2_INTERFACE_DEFINED__

/* interface ISySalDataIO2 */
/* [unique][helpstring][uuid][object] */ 


EXTERN_C const IID IID_ISySalDataIO2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5892a1f5-5dd9-4fe3-a024-d8b624917c1d")
    ISySalDataIO2 : public ISySalDataIO
    {
    public:
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE Write2( 
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName) = 0;
        
        virtual /* [helpstring] */ HRESULT STDMETHODCALLTYPE Read2( 
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct ISySalDataIO2Vtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ISySalDataIO2 * This,
            /* [in] */ REFIID riid,
            /* [iid_is][out] */ void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ISySalDataIO2 * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ISySalDataIO2 * This);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *Write )( 
            ISySalDataIO2 * This,
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *Read )( 
            ISySalDataIO2 * This,
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *GetStatus )( 
            ISySalDataIO2 * This,
            UINT *pExtErrorInfo);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *Write2 )( 
            ISySalDataIO2 * This,
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName);
        
        /* [helpstring] */ HRESULT ( STDMETHODCALLTYPE *Read2 )( 
            ISySalDataIO2 * This,
            HSySalHANDLE hWnd,
            BYTE *pData,
            UINT *pExtErrorInfo,
            BYTE *FileName);
        
        END_INTERFACE
    } ISySalDataIO2Vtbl;

    interface ISySalDataIO2
    {
        CONST_VTBL struct ISySalDataIO2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ISySalDataIO2_QueryInterface(This,riid,ppvObject)	\
    (This)->lpVtbl -> QueryInterface(This,riid,ppvObject)

#define ISySalDataIO2_AddRef(This)	\
    (This)->lpVtbl -> AddRef(This)

#define ISySalDataIO2_Release(This)	\
    (This)->lpVtbl -> Release(This)


#define ISySalDataIO2_Write(This,hWnd,pData,pExtErrorInfo,FileName)	\
    (This)->lpVtbl -> Write(This,hWnd,pData,pExtErrorInfo,FileName)

#define ISySalDataIO2_Read(This,hWnd,pData,pExtErrorInfo,FileName)	\
    (This)->lpVtbl -> Read(This,hWnd,pData,pExtErrorInfo,FileName)

#define ISySalDataIO2_GetStatus(This,pExtErrorInfo)	\
    (This)->lpVtbl -> GetStatus(This,pExtErrorInfo)


#define ISySalDataIO2_Write2(This,hWnd,pData,pExtErrorInfo,FileName)	\
    (This)->lpVtbl -> Write2(This,hWnd,pData,pExtErrorInfo,FileName)

#define ISySalDataIO2_Read2(This,hWnd,pData,pExtErrorInfo,FileName)	\
    (This)->lpVtbl -> Read2(This,hWnd,pData,pExtErrorInfo,FileName)

#endif /* COBJMACROS */


#endif 	/* C style interface */



/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalDataIO2_Write2_Proxy( 
    ISySalDataIO2 * This,
    HSySalHANDLE hWnd,
    BYTE *pData,
    UINT *pExtErrorInfo,
    BYTE *FileName);


void __RPC_STUB ISySalDataIO2_Write2_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring] */ HRESULT STDMETHODCALLTYPE ISySalDataIO2_Read2_Proxy( 
    ISySalDataIO2 * This,
    HSySalHANDLE hWnd,
    BYTE *pData,
    UINT *pExtErrorInfo,
    BYTE *FileName);


void __RPC_STUB ISySalDataIO2_Read2_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);



#endif 	/* __ISySalDataIO2_INTERFACE_DEFINED__ */



#ifndef __SYSALDATAIOLib_LIBRARY_DEFINED__
#define __SYSALDATAIOLib_LIBRARY_DEFINED__

/* library SYSALDATAIOLib */
/* [helpstring][version][uuid] */ 


EXTERN_C const IID LIBID_SYSALDATAIOLib;

EXTERN_C const CLSID CLSID_SySalDataIO;

#ifdef __cplusplus

class DECLSPEC_UUID("4B47E8CE-894C-11D3-A47C-9F3735226036")
SySalDataIO;
#endif
#endif /* __SYSALDATAIOLib_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


