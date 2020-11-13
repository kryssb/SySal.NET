////////////////////////////////////////////////////////////////////////////////
//! 
//! \file  mil3dmap.h
//! 
//! \brief Mil3dmap CAPI header (M3dmap...)
//! 
//! AUTHOR: Matrox Imaging dept.
//!
//! COPYRIGHT NOTICE:
//! Copyright © 2007 Matrox Electronic Systems Ltd. 
//! All Rights Reserved 
//  Revision:  9.01.0855
////////////////////////////////////////////////////////////////////////////////

#ifndef __MIL_3DMAP_H__
#define __MIL_3DMAP_H__

#if (!M_MIL_LITE) /* MIL FULL ONLY */

/* C++ directive if needed */
#ifdef __cplusplus
extern "C"
   {
#endif

////////////////////////////////////////////////////////////////////////////////
// M3dmapAlloc ContextTypes

#define M_LASER                           1L

// ControlFlags for M_LASER ContextType

#define M_CALIBRATED_CAMERA_LINEAR_MOTION 1L
#define M_DEPTH_CORRECTION                2L

////////////////////////////////////////////////////////////////////////////////
// M3dmapAllocResult ResultTypes

#define M_LASER_DATA                      1L

////////////////////////////////////////////////////////////////////////////////
// M3dmapControl/M3dmapInquire

// ControlTypes

#define M_SCAN_SPEED                      2L
#define M_FILL_MODE                       3L
#define M_FILL_SHARP_ELEVATION            4L
#define M_FILL_SHARP_ELEVATION_DEPTH      5L
#define M_ORIENTATION                     0x00002400L    // Already defined in mil.h, milpat.h, milmeas.h
#define M_PEAK_WIDTH                      11L
#define M_MIN_INTENSITY                   12L
#define M_MAX_FRAMES                      17L
#define M_CORRECTED_DEPTH                 18L

// Deprecated names, kept for backward compatibility
#define M_FILL_GAP_MODE                   M_FILL_SHARP_ELEVATION
#define M_FILL_GAP_DEPTH                  M_FILL_SHARP_ELEVATION_DEPTH

#define M_PIXEL_SIZE_X                    139L                             // Already defined in mildisplay.h
#define M_PIXEL_SIZE_Y                    140L                             // Already defined in mildisplay.h
#define M_GRAY_LEVEL_SIZE_Z               141L
#define M_WORLD_OFFSET_X                  142L
#define M_WORLD_OFFSET_Y                  143L
#define M_WORLD_OFFSET_Z                  144L

#define M_CALIBRATION_STATUS              159L           // Already defined in milcal.h

// Control Values for M_FILL_MODE

#define M_DISABLE                         -9999L         // Already defined in milblob.h, milcal.h, milmeas.h, mil.h, ...
#define M_X_THEN_Y                        1L

// Control Values for M_FILL_GAP_MODE

#define M_DISABLE                         -9999L         // Already defined in milblob.h, milcal.h, milmeas.h, mil.h, ...
#define M_MIN                             0x02000000L    // Already defined in mil.h, milim.h, ...
#define M_MAX                             0x04000000L    // Already defined in mil.h, milim.h, ...

// Control Values for M_FILL_GAP_DEPTH

#define M_INFINITE                        -1L            // Already defined in mil.h, milstr.h, milreg.h, milmetrol.h

// Control Values for M_ORIENTATION

#define M_1D_ROWS                         0x00000010L    // Already defined in milim.h
#define M_1D_COLUMNS                      0x00000020L    // Already defined in milim.h

// Control Values for M_CALIBRATION_STATUS
#define M_CALIBRATED                      0x0000300L     // Already defined in milcal.h, milmod.h
#define M_LASER_LINE_NOT_DETECTED         2L
#define M_NOT_INITIALIZED                 3L             // Already defined in milcal.h
#define M_NOT_ENOUGH_MEMORY               4L
#define M_INTERNAL_ERROR                  5L
#define M_MATHEMATICAL_EXCEPTION          8L             // Already defined in milcal.h

////////////////////////////////////////////////////////////////////////////////
// M3dmapExtract Operations

#define M_CORRECTED_DEPTH_MAP             1L

////////////////////////////////////////////////////////////////////////////////
// M3dmapGetResult ResultTypes

#define M_3D_POINTS_X                     1L
#define M_3D_POINTS_Y                     2L
#define M_3D_POINTS_Z                     3L
#define M_3D_POINTS_I                     4L
#define M_NUMBER_OF_3D_POINTS             5L
#define M_CORRECTED_DEPTH_MAP_SIZE_X      6L
#define M_CORRECTED_DEPTH_MAP_SIZE_Y      7L
#define M_CORRECTED_DEPTH_MAP_BUFFER_TYPE 8L
#define M_INTENSITY_MAP_BUFFER_TYPE       9L

////////////////////////////////////////////////////////////////////////////////
// M3dmapTriangulate ControlFlags

#define M_NO_INVALID_POINT                1L

////////////////////////////////////////////////////////////////////////////////
// CAPI function prototypes

#ifndef __midl // MIDL compiler used by ActiveMIL

MFTYPE32 MIL_ID MFTYPE M3dmapAlloc(MIL_ID  SystemId, 
                                   MIL_INT ContextType, 
                                   MIL_INT ControlFlag, 
                                   MIL_ID* ContextIdPtr);

MFTYPE32 MIL_ID MFTYPE M3dmapAllocResult(MIL_ID  SystemId,
                                         MIL_INT ResultType, 
                                         MIL_INT ControlFlag, 
                                         MIL_ID* ResultIdPtr);

MFTYPE32 void MFTYPE M3dmapFree(MIL_ID ContextOrResultId);

#if M_MIL_USE_64BIT
// Prototypes for 64 bits OSs
MFTYPE32 void MFTYPE M3dmapControlInt64(MIL_ID    ContextOrResultId,
                                        MIL_INT   Index,
                                        MIL_INT   ControlType,
                                        MIL_INT64 ControlValue);
MFTYPE32 void MFTYPE M3dmapControlDouble(MIL_ID   ContextOrResultId,
                                         MIL_INT  Index,
                                         MIL_INT  ControlType,
                                         double   ControlValue);
#else
// Prototypes for 32 bits OSs
#define M3dmapControlInt64  M3dmapControl
#define M3dmapControlDouble M3dmapControl
MFTYPE32 void MFTYPE M3dmapControl(MIL_ID  ContextOrResultId,
                                   MIL_INT Index,
                                   MIL_INT ControlType,
                                   double  ControlValue);
#endif

MFTYPE32 MIL_INT MFTYPE M3dmapInquire(MIL_ID  ContextOrResultId,
                                      MIL_INT Index,
                                      MIL_INT InquireType,
                                      void*   UserVarPtr);

MFTYPE32 void MFTYPE M3dmapAddScan(MIL_ID  ContextId,
                                   MIL_ID  ResultId,
                                   MIL_ID  ImageId, 
                                   MIL_ID  LineIntensityImageId,
                                   MIL_ID  ExtraInfoArrayId,
                                   MIL_INT Index,
                                   MIL_INT ControlFlag);

MFTYPE32 void MFTYPE M3dmapCalibrate(MIL_ID  ContextId,
                                     MIL_ID  ResultId,
                                     MIL_ID  CalibrationId,
                                     MIL_INT ControlFlag);

MFTYPE32 void MFTYPE M3dmapExtract(MIL_ID  ResultId,
                                   MIL_ID  DepthMapId,
                                   MIL_ID  IntensityMapId,
                                   MIL_INT Operation,
                                   MIL_INT Index,
                                   MIL_INT ControlFlag);

MFTYPE32 void MFTYPE M3dmapGetResult(MIL_ID  ResultId,
                                     MIL_INT Index,
                                     MIL_INT ResultType,
                                     void*   ResultArrayPtr);

MFTYPE32 void MFTYPE M3dmapTriangulate(const MIL_ID* CalibrationOrImageIdArrayPtr,
                                       const double* XPixelArrayPtr,
                                       const double* YPixelArrayPtr,
                                       double*       XWorldArrayPtr,
                                       double*       YWorldArrayPtr,
                                       double*       ZWorldArrayPtr,
                                       double*       RMSErrorArrayPtr,
                                       MIL_INT       NbCalibrations,
                                       MIL_INT       NbPoints,
                                       MIL_INT       CoordinateSystem,
                                       MIL_INT       ControlFlag);

#if M_MIL_USE_UNICODE
MFTYPE32 void MFTYPE M3dmapSaveA(const char* FileName,
                                 MIL_ID      ContextOrResultId,
                                 MIL_INT     ControlFlag);

MFTYPE32 MIL_ID MFTYPE M3dmapRestoreA(const char* FileName,
                                      MIL_ID      SystemId,
                                      MIL_INT     ControlFlag,
                                      MIL_ID*     ContextOrResultIdPtr);

MFTYPE32 void MFTYPE M3dmapStreamA(char*    MemPtrOrFileName,
                                   MIL_ID   SystemId,
                                   MIL_INT  Operation,
                                   MIL_INT  StreamType,
                                   double   Version,
                                   MIL_INT  ControlFlag,
                                   MIL_ID*  ContextOrResultIdPtr,
                                   MIL_INT* SizeByteVarPtr);

MFTYPE32 void MFTYPE M3dmapSaveW(MIL_CONST_TEXT_PTR FileName,
                                 MIL_ID             ContextOrResultId,
                                 MIL_INT            ControlFlag);

MFTYPE32 MIL_ID MFTYPE M3dmapRestoreW(MIL_CONST_TEXT_PTR FileName,
                                      MIL_ID             SystemId,
                                      MIL_INT            ControlFlag,
                                      MIL_ID*            ContextOrResultIdPtr);

MFTYPE32 void MFTYPE M3dmapStreamW(MIL_TEXT_PTR MemPtrOrFileName,
                                   MIL_ID       SystemId,
                                   MIL_INT      Operation,
                                   MIL_INT      StreamType,
                                   double       Version,
                                   MIL_INT      ControlFlag,
                                   MIL_ID*      ContextOrResultIdPtr,
                                   MIL_INT*     SizeByteVarPtr);

#if M_MIL_UNICODE_API
#define M3dmapSave           M3dmapSaveW
#define M3dmapRestore        M3dmapRestoreW
#define M3dmapStream         M3dmapStreamW
#else
#define M3dmapSave           M3dmapSaveA
#define M3dmapRestore        M3dmapRestoreA
#define M3dmapStream         M3dmapStreamA
#endif

#else
MFTYPE32 void MFTYPE M3dmapSave(MIL_CONST_TEXT_PTR FileName,
                                MIL_ID             ContextOrResultId,
                                MIL_INT            ControlFlag);

MFTYPE32 MIL_ID MFTYPE M3dmapRestore(MIL_CONST_TEXT_PTR FileName,
                                     MIL_ID             SystemId,
                                     MIL_INT            ControlFlag,
                                     MIL_ID*            ContextOrResultIdPtr);

MFTYPE32 void MFTYPE M3dmapStream(MIL_TEXT_PTR MemPtrOrFileName,
                                  MIL_ID       SystemId,
                                  MIL_INT      Operation,
                                  MIL_INT      StreamType,
                                  double       Version,
                                  MIL_INT      ControlFlag,
                                  MIL_ID*      ContextOrResultIdPtr,
                                  MIL_INT*     SizeByteVarPtr);
#endif

#endif /* #ifdef __midl */

/* C++ directive if needed */
#ifdef __cplusplus
}
#endif
////////////////////////////////////////////////////////////////////////////////

#if M_MIL_USE_64BIT
#ifdef __cplusplus
//////////////////////////////////////////////////////////////
// M3dmapControl function definition when compiling c++ files
//////////////////////////////////////////////////////////////
#if !M_MIL_USE_LINUX
inline void M3dmapControl(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT ControlType, int ControlValue)
   {
   M3dmapControlInt64(ContextOrResultId, Index, ControlType, ControlValue);
   };
#endif
inline void M3dmapControl(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT ControlType, MIL_INT32 ControlValue)
   {
   M3dmapControlInt64(ContextOrResultId, Index, ControlType, ControlValue);
   };

inline void M3dmapControl(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT ControlType, MIL_INT64 ControlValue)
   {
   M3dmapControlInt64(ContextOrResultId, Index, ControlType, ControlValue);
   };

inline void M3dmapControl(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT ControlType, double ControlValue)
   {
   M3dmapControlDouble(ContextOrResultId, Index, ControlType, ControlValue);
   };
#else
//////////////////////////////////////////////////////////////
// For C file, call the default function, i.e. Double one
//////////////////////////////////////////////////////////////
#define M3dmapControl  M3dmapControlDouble

#endif // __cplusplus
#endif // M_MIL_USE_64BIT

#if M_MIL_USE_SAFE_TYPE

//////////////////////////////////////////////////////////////
// See milos.h for explanation about these functions.
//////////////////////////////////////////////////////////////

//-------------------------------------------------------------------------------------
//  M3dmapGetResult

inline MFTYPE32 void MFTYPE M3dmapGetResultUnsafe  (MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, void          MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, int                   ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT8      MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT16     MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT32     MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT64     MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, float         MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_DOUBLE    MPTYPE *ResultArrayPtr);
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT8     MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT16    MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT32    MPTYPE *ResultArrayPtr);
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT64    MPTYPE *ResultArrayPtr);
#endif

// ----------------------------------------------------------
// M3dmapInquire

inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireUnsafe  (MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, void       MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, int                 UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT8   MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT16  MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT32  MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT64  MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, float      MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_DOUBLE MPTYPE * UserVarPtr);
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT8  MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT16 MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT32 MPTYPE * UserVarPtr);
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT64 MPTYPE * UserVarPtr);
#endif
#if M_MIL_SAFE_TYPE_ADD_WCHAR_T
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, wchar_t    MPTYPE * UserVarPtr);
#endif

// -------------------------------------------------------------------------
// M3dmapGetResult

inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType (MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, int ResultPtr)
   {
   if (ResultPtr != 0)
      SafeTypeError(MT("M3dmapGetResult"));

   M3dmapGetResult(ResultId, Index, ResultType, NULL);
   }

inline void M3dmapGetResultSafeTypeExecute (MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, void MPTYPE *ResultArrayPtr, MIL_INT GivenType)
   {
   MIL_INT RequiredType = (ResultType & (M_TYPE_DOUBLE | M_TYPE_FLOAT | M_TYPE_SHORT | M_TYPE_CHAR | M_TYPE_MIL_INT32 |
#if (BW_COMPATIBILITY != 0x80)
      M_TYPE_MIL_INT64 | 
#endif
      M_TYPE_MIL_ID ));
   if (RequiredType == 0)
      RequiredType = M_TYPE_DOUBLE;
   ReplaceTypeMilIdByTypeMilIntXX(&RequiredType);

   if (RequiredType != GivenType)
      SafeTypeError(MT("M3dmapGetResult"));

   M3dmapGetResult(ResultId, Index, ResultType, ResultArrayPtr);
   }


inline MFTYPE32 void MFTYPE M3dmapGetResultUnsafe  (MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, void       MPTYPE *ResultArrayPtr){M3dmapGetResult               (ResultId, Index, ResultType, ResultArrayPtr);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT8   MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_CHAR     );}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT16  MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_SHORT    );}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT32  MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_INT64  MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_MIL_INT64);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, float      MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_FLOAT);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_DOUBLE MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_DOUBLE);}
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT8  MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_CHAR);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT16 MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_SHORT);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT32 MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 void MFTYPE M3dmapGetResultSafeType(MIL_ID ResultId, MIL_INT Index, MIL_INT ResultType, MIL_UINT64 MPTYPE *ResultArrayPtr){M3dmapGetResultSafeTypeExecute(ResultId, Index, ResultType, ResultArrayPtr, M_TYPE_MIL_INT64);}
#endif

// ----------------------------------------------------------
// M3dmapInquire

inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, int UserVarPtr)
   {
   if (UserVarPtr != 0)
      SafeTypeError(MT("M3dmapInquire"));

   return M3dmapInquire(ContextOrResultId, Index, InquireType, NULL);
   }

inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeTypeExecute(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, void MPTYPE * UserVarPtr, MIL_INT GivenType)
   {
   MIL_INT RequiredType = (InquireType & (M_TYPE_MIL_INT32 | 
#if (BW_COMPATIBILITY != 0x80)
      M_TYPE_MIL_INT64 | 
#endif
      M_TYPE_DOUBLE | M_TYPE_FLOAT | M_TYPE_SHORT | M_TYPE_CHAR | M_TYPE_MIL_ID));
   if (RequiredType == 0)
      RequiredType = M_TYPE_DOUBLE;
   ReplaceTypeMilIdByTypeMilIntXX(&RequiredType);

   if (RequiredType != GivenType)
      SafeTypeError(MT("M3dmapInquire"));

   return M3dmapInquire(ContextOrResultId, Index, InquireType, UserVarPtr);
   }

inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireUnsafe  (MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, void       MPTYPE * UserVarPtr) {return M3dmapInquire       (ContextOrResultId, Index, InquireType, UserVarPtr                  );}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT8   MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_CHAR);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT16  MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_SHORT);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT32  MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_INT64  MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT64);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, float      MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_FLOAT   );}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_DOUBLE MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_DOUBLE   );}
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT8  MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_CHAR);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT16 MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_SHORT);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT32 MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, MIL_UINT64 MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT64);}
#endif
#if M_MIL_SAFE_TYPE_ADD_WCHAR_T
inline MFTYPE32 MIL_INT MFTYPE M3dmapInquireSafeType(MIL_ID ContextOrResultId, MIL_INT Index, MIL_INT InquireType, wchar_t    MPTYPE * UserVarPtr) {return M3dmapInquireSafeTypeExecute(ContextOrResultId, Index, InquireType, UserVarPtr, M_TYPE_SHORT);}
#endif

#define M3dmapGetResult        M3dmapGetResultSafeType
#define M3dmapInquire          M3dmapInquireSafeType

#else // #if M_MIL_USE_SAFE_TYPE

#define M3dmapGetResultUnsafe  M3dmapGetResult
#define M3dmapInquireUnsafe    M3dmapInquire

#endif // #if M_MIL_USE_SAFE_TYPE

#endif /* !M_MIL_LITE */

#endif
