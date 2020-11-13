/***************************************************************************/
/*

    Filename:  MILCAL.H
    Owner   :  Matrox Imaging dept.
    Revision:  9.01.0855
    Content :  This file contains the defines and the prototypes for the
               MIL calibration module. (Mcal...).

    COPYRIGHT (c) 1993-1997 Matrox Electronic Systems Ltd.
    All Rights Reserved

*/
/***************************************************************************/

#ifndef __MILCAL_H__
#define __MILCAL_H__

#if (!M_MIL_LITE) /* MIL FULL ONLY */

/* C++ directive if needed */
#ifdef __cplusplus
extern "C"
{
#endif


/***************************************************************************/
/*                      Calibration CAPI defines                           */
/***************************************************************************/

/***************************************************************************/
/* McalAlloc()                                                             */
/***************************************************************************/

/* Calibration mode */
#define M_LINEAR_INTERPOLATION                 0x01L
#define M_PERSPECTIVE_TRANSFORMATION           0x02L
#define M_TSAI_BASED                           0x08L
#define M_3D_ROBOTICS                          0x10L

/***************************************************************************/
/* McalGrid(), McalList()                                                  */
/***************************************************************************/

/* Grid Calibration mode */
#define M_DISPLACE_CAMERA_COORD               0x100L
#define M_FULL_CALIBRATION                    0x101L
#define M_DISPLACE_RELATIVE_COORD             0x102L
#define M_ACCUMULATE                          0x200L

/* Deprecated names, kept for backward compatibility */
#define M_LOCATE_CAMERA_ONLY                  M_DISPLACE_CAMERA_COORD
#define M_LOCATE_RELATIVE_ONLY                M_DISPLACE_RELATIVE_COORD

/* Calibration grid type */
#define M_CIRCLE_GRID                         0x001L
#define M_CHESSBOARD_GRID                     0x002L

/* Optional ModeFlag that can be
   added to the grid type */
#define M_Y_AXIS_UP                           0x010L
#define M_Y_AXIS_DOWN                         0x020L

#define M_FAST                           0x00002000L /* Also defined in milpat.h */

/***************************************************************************/
/* McalTransformCoordinate() and McalTransformResult()                       */
/***************************************************************************/

/* Transform type */
#define M_PIXEL_TO_WORLD                          1L
#define M_WORLD_TO_PIXEL                          2L
#define M_PACKED                         0x00020000L /* Also defined in mil.h */

/* Aspect ratio pre-correction flags */
#define M_PRE_CORRECTED                       0x100L

/* Data type */
#define M_LENGTH_X                              0x3L
#define M_LENGTH_Y                              0x4L
#define M_LENGTH                         0x00002000L  /* (Already defined in milmeas.h) */
#define M_AREA                                    2L  /* (Already defined in milblob.h) */
#define M_ANGLE						        0x00000800L  /* (Already defined in milmeas.h) */

/* Error in McalTransformCoordinate() and McalTransformCoordinate() */
#define M_INVALID_POINT                  1.7976931348623158e+308
#define M_ALLOW_INVALID_POINT_OUTPUT     0x00040000L

/***************************************************************************/
/* McalTransformCoordinate3dList()                                         */
/***************************************************************************/

/* Optional ModeFlag */
#define M_PLANE_INTERSECTION                    1L
#define M_UNIT_DIRECTION_VECTOR                 2L

/***************************************************************************/
/* McalControl() and/or McalInquire()                                      */
/***************************************************************************/

/* Control parameters. */

#define M_FOREGROUND_VALUE	                     4L
#define M_CAMERA_POSITION_X                     100L /* deprecated: use M_TOOL_POSITION_X */
#define M_CAMERA_POSITION_Y                     101L /* deprecated: use M_TOOL_POSITION_Y */
#define M_CAMERA_POSITION_Z                     102L /* deprecated: use M_TOOL_POSITION_Z */
#define M_PRINCIPAL_POINT_X                     103L
#define M_PRINCIPAL_POINT_Y                     104L
/* #define M_TSAI_CELL_SIZE_X                      105L : deprecated */
/* #define M_TSAI_CELL_SIZE_Y                      106L : deprecated */
/* #define M_TSAI_NUMBER_CELL_X                    107L : deprecated */
/* #define M_TSAI_NUMBER_PIXEL_X                   108L : deprecated */
#define M_GRID_ORIGIN_X                         109L
#define M_GRID_ORIGIN_Y                         110L
#define M_GRID_ORIGIN_Z                         111L
#define M_RELATIVE_ORIGIN_X                     112L
#define M_RELATIVE_ORIGIN_Y                     113L
#define M_RELATIVE_ORIGIN_Z                     114L
#define M_RELATIVE_ORIGIN_ANGLE                 115L
#define M_ROW_NUMBER                            116L
#define M_COLUMN_NUMBER                         117L
#define M_ROW_SPACING                           118L
#define M_COLUMN_SPACING                        119L
#define M_INPUT_UNITS                           121L
#define M_INPUT_COORDINATE_SYSTEM               M_INPUT_UNITS       /* deprecated name */
#define M_OUTPUT_UNITS                          122L
#define M_OUTPUT_COORDINATE_SYSTEM              M_OUTPUT_UNITS      /* deprecated name */
#define M_ASPECT_RATIO                         6001L /* (Already defined in mil.h) */
#define M_AUTO_ASPECT_RATIO                     123L
#define M_ASPECT_RATIO_AUTO_SETTING             M_AUTO_ASPECT_RATIO /* deprecated name */
#define M_ENHANCED_RESULTS_ACCURACY             124L
#define M_ASSOCIATED_CALIBRATION                125L
#define M_CORRECTION_STATE                      126L
#define M_PERSPECTIVE_EFFECT                    127L
#define M_PARTIALLY_ALIGNED                     128L
#define M_COMPLETELY_ALIGNED                    129L
#define M_CALIBRATION_SUCCESSFUL                130L
#define M_CALIBRATION_RESET                     245L
    
#define M_TRANSFORM_FILL_MODE                   131L /* deprecated name */
#define M_TRANSFORM_IMAGE_FILL_MODE             M_TRANSFORM_FILL_MODE
#define M_TRANSFORM_CACHE                       132L
#define M_TRANSFORM_AXIS_ALIGNMENT              133L
#define M_TRANSFORM_ACCURACY                    134L

#if ((M_MIL_USE_UNICODE && !M_MIL_UNICODE_API && !M_COMPILING_MILDLL)) && (BW_COMPATIBILITY != 0x80)
#define M_USER_STRING                           (136L|M_CLIENT_ASCII_MODE)
#else
#define M_USER_STRING                           136L
#endif
#define M_USER_STRING_LENGTH                    137L

#define M_IGNORE_MATHEMATICAL_ASSERTION         135L
#define M_DEBUG_SAVING                          138L

#define M_PIXEL_SIZE_X                          139L  /* deprecated */
#define M_PIXEL_SIZE_Y                          140L  /* deprecated */

#define M_CALIBRATION_CHILD_OFFSET_X            141L
#define M_CALIBRATION_CHILD_OFFSET_Y            142L

#define M_CORRECTED_OFFSET_X                    M_CALIBRATION_CHILD_OFFSET_X /* deprecated name */
#define M_CORRECTED_OFFSET_Y                    M_CALIBRATION_CHILD_OFFSET_Y /* deprecated name */

#define M_CHILD_CALIBRATION_OFFSET_X            145L /* deprecated */
#define M_CHILD_CALIBRATION_OFFSET_Y            146L /* deprecated */

#define M_FOCAL_LENGTH                          1009L /* already defined in milreg.h */
#define M_CCD_ASPECT_RATIO                      148L
#define M_CCD_SKEW                              149L
#define M_DISTORTION_RADIAL_1                   150L
#define M_DISTORTION_RADIAL_2                   151L
#define M_DISTORTION_RADIAL_3                   152L
#define M_DISTORTION_TANGENTIAL_1               153L
#define M_DISTORTION_TANGENTIAL_2               154L

#define M_LINK_TOOL_AND_CAMERA                  155L
#define M_TOOL_POSITION_X                       156L
#define M_TOOL_POSITION_Y                       157L
#define M_TOOL_POSITION_Z                       158L

#define M_CALIBRATION_STATUS                    159L
#define M_CALIBRATION_PLANE                     160L
#define M_AFFECTED_EXTRINSICS                   161L
#define M_GRID_CORNER_HINT_X                    162L
#define M_GRID_CORNER_HINT_Y                    163L

#define M_NUMBER_OF_CALIBRATION_POSES           199L
#define M_NUMBER_OF_CALIBRATION_POINTS          200L
#define M_CALIBRATION_IMAGE_POINTS_X            201L
#define M_CALIBRATION_IMAGE_POINTS_Y            202L
#define M_CALIBRATION_WORLD_POINTS_X            203L
#define M_CALIBRATION_WORLD_POINTS_Y            204L
#define M_CALIBRATION_MODE                      205L
#define M_GRID_TYPE                             206L

#define M_CORRECTED_WORLD_POS_X                 207L
#define M_CORRECTED_WORLD_POS_Y                 208L
#define M_CORRECTED_PIXEL_SIZE_X                M_PIXEL_SIZE_X
#define M_CORRECTED_PIXEL_SIZE_Y                M_PIXEL_SIZE_Y

#define M_TRANSFORM_IMAGE_WORLD_POS_X           209L
#define M_TRANSFORM_IMAGE_WORLD_POS_Y           210L
#define M_TRANSFORM_IMAGE_PIXEL_SIZE_X          211L
#define M_TRANSFORM_IMAGE_PIXEL_SIZE_Y          212L

#define M_CALIBRATION_WORLD_POINTS_Z            213L
#define M_CALIBRATION_CLASS                     214L


#define M_AVERAGE_PIXEL_ERROR                0x1000L
#define M_MAXIMUM_PIXEL_ERROR                0x2000L
#define M_GLOBAL_AVERAGE_PIXEL_ERROR         0x1001L
#define M_GLOBAL_MAXIMUM_PIXEL_ERROR         0x2002L
#define M_AVERAGE_WORLD_ERROR                0x4000L
#define M_MAXIMUM_WORLD_ERROR                0x8000L
#define M_GLOBAL_AVERAGE_WORLD_ERROR         0x4004L
#define M_GLOBAL_MAXIMUM_WORLD_ERROR         0x8008L

/* Data types for control parameters are no longer redefined... to avoid all the conditions and defines around MIL_INT variables (x64)*/

/* Control values. */
#define M_PIXEL                              0x1000L
#define M_WORLD                              0x2000L
#define M_CLIP                               0x0010L
#define M_FIT                                0x0020L
#define M_USER_DEFINED                           21L /* already defined in Mil.h */
#define M_ENABLE                              -9997L /* (Already defined in mil.h) */
#define M_DISABLE                             -9999L /* (Already defined in mil.h) */

/* M_CALIBRATION_SUCCESSFUL control values */
#define M_FALSE                                   0L /* already defined in mil.h */
#define M_TRUE                                    1L /* already defined in mil.h */

/* M_CALIBRATION_STATUS control values */
#define M_CALIBRATED                      0x0000300L /* already defined in milmod.h */
#define M_NOT_INITIALIZED                         3L
#define M_GRID_NOT_FOUND                          4L
#define M_PLANE_ANGLE_TOO_SMALL                   5L
#define M_CALIBRATING                             6L
#define M_INVALID_CALIBRATION_POINTS              7L
#define M_MATHEMATICAL_EXCEPTION                  8L

/* M_CALIBRATION_CLASS control values */
#define M_2D_CALIBRATION                          1L
#define M_3D_CALIBRATION                          2L

/* M_AFFECTED_EXTRINSICS control values */
#define M_EXTRINSICS_CURRENT                      0L
#define M_EXTRINSICS_AT_CALIBRATION_TIME          1L
#define M_EXTRINSICS_AT_FULL_CALIBRATION_TIME     2L
#define M_EXTRINSICS_AT_LOCATE_TIME               3L

/* M_GRID_CORNER_HINT_X and M_GRID_CORNER_HINT_Y control values */
#define M_NONE                           0x08000000L /* already defined in milstr.h, milcolor.h */

/* M_ASPECT_RATIO, M_PIXEL_SIZE_X and M_PIXEL_SIZE_Y can return this if invalid */
#define M_INVALID_SCALE                      -999999.0

/***************************************************************************/
/* McalRestore(), McalSave(), McalStream()                                 */
/***************************************************************************/
#define M_INTERACTIVE                         M_NULL /* Already defined in mil.h, milcode.h, miledge.h, milmeas.h, milocr.h, milpat.h, milmod.h */
#define M_INQUIRE_CALIBRATION_MODE             0x20L

/***************************************************************************/
/* McalTransformImage()                                                    */
/***************************************************************************/
// Operation Type //
#define M_FULL_CORRECTION                      0x001L
#define M_CORRECT_LENS_DISTORTION_ONLY         0x002L

// Control Flag //
#define M_WARP_IMAGE                           0x001L
#define M_EXTRACT_LUT_X                        0x002L
#define M_EXTRACT_LUT_Y                        0x004L

/***************************************************************************/
/* McalGetCoordinateSystem(), McalSetCoordinateSystem()                    */
/***************************************************************************/

#define M_IMAGE_COORDINATE_SYSTEM        0x02000000L
#define M_ABSOLUTE_COORDINATE_SYSTEM     0x01000000L
#define M_RELATIVE_COORDINATE_SYSTEM              0L
#define M_TOOL_COORDINATE_SYSTEM                  1L
#define M_CAMERA_COORDINATE_SYSTEM                2L
#define M_ROBOT_BASE_COORDINATE_SYSTEM            3L

#define M_HOMOGENEOUS_MATRIX                      0L
#define M_TRANSLATION                             1L
#define M_ROTATION_AXIS_ANGLE                     2L
#define M_ROTATION_QUATERNION                     3L
#define M_ROTATION_YXZ                            4L
#define M_ROTATION_MATRIX                         5L
#define M_IDENTITY                                6L
#define M_ROTATION_X                              7L
#define M_ROTATION_Y                              8L
#define M_ROTATION_Z                              9L
#define M_ROTATION_XYZ                           10L
#define M_ROTATION_XZY                           11L
#define M_ROTATION_YZX                           12L
#define M_ROTATION_ZXY                           13L
#define M_ROTATION_ZYX                           14L

#define M_NB_TRANSFORM_TYPES                     15L
#define M_TRANSFORM_TYPES_MASK           0x000000FFL
#define M_TRANSFORM_TYPES_SHIFT                   8L

/* Combination flags for McalSetCoordinateSystem(), must be > M_TRANSFORM_TYPES_MASK */
#define M_ASSIGN                                (1 << M_TRANSFORM_TYPES_SHIFT)
#define M_COMPOSE_WITH_CURRENT                  (2 << M_TRANSFORM_TYPES_SHIFT)

/***************************************************************************/
/* McalDraw()                                                              */
/***************************************************************************/

#define M_DRAW_IMAGE_POINTS                        1L
#define M_DRAW_WORLD_POINTS                        2L
#define M_DRAW_VALID_REGION                        3L
#define M_DRAW_VALID_REGION_FILLED                 4L

/***************************************************************************/
/*                 Calibration CAPI function prototype                     */
/***************************************************************************/

#ifndef __midl /* MIDL compiler used by ActiveMIL */

MFTYPE32 MIL_ID MFTYPE McalAlloc(MIL_ID SystemId,
                                 MIL_INT Mode,
                                 MIL_INT ModeFlag,
                                 MIL_ID *ContextIdPtr);

MFTYPE32 void MFTYPE McalFree(MIL_ID CalibrationId);

MFTYPE32 void MFTYPE McalAssociate(MIL_ID CalibrationId,
                                   MIL_ID AppOrMilObjectId,
                                   MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McalDraw(MIL_ID  GraphContId,
                              MIL_ID  CalibrationId,
                              MIL_ID  DestImageBufId, 
                              MIL_INT Operation,
                              MIL_INT Index,
                              MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McalGrid(MIL_ID CalibrationId,
                              MIL_ID SrcImageId,
                              double GridOffsetX,
                              double GridOffsetY,
                              double GridOffsetZ,
                              MIL_INT   RowNumber,
                              MIL_INT   ColumnNumber,
                              double RowSpacing,
                              double ColumnSpacing,
                              MIL_INT   Operation,
                              MIL_INT   GridType);

MFTYPE32 void MFTYPE McalList(MIL_ID CalibrationId,
                              const double * XPixArray,
                              const double * YPixArray,
                              const double * XWorldArray,
                              const double * YWorldArray,
                              const double * ZWorldArray,
                              MIL_INT   NumData,
                              MIL_INT   Operation,
                              MIL_INT   ControlFlag);

MFTYPE32 void MFTYPE McalRelativeOrigin(MIL_ID CalibrationID,
                                        double XOffset,
                                        double YOffset,
                                        double ZOffset,
                                        double AngularOffset,
                                        MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McalTransformCoordinate(MIL_ID CalibrationOrImageId,
                                             MIL_INT TransformType,
                                             double X,
                                             double Y,
                                             double * ResXPtr,
                                             double * ResYPtr);

MFTYPE32 void MFTYPE McalTransformCoordinateList(MIL_ID CalibrationOrImageId,
                                                 MIL_INT TransformType,
                                                 MIL_INT NumPoints,
                                                 const double *SrcXPtr,
                                                 const double *SrcYPtr,
                                                 double *ResXPtr,
                                                 double *ResYPtr);

MFTYPE32 void MFTYPE McalTransformCoordinate3dList(MIL_ID CalibrationOrImageId,
                                                   MIL_INT SrcCoordinateSystem,
                                                   MIL_INT DstCoordinateSystem,
                                                   MIL_INT NumPoints,
                                                   const double* XSrcArray,
                                                   const double* YSrcArray,
                                                   const double* ZSrcArray,
                                                   double* XDstArray,
                                                   double* YDstArray,
                                                   double* ZDstArray,
                                                   MIL_INT ModeFlag);

MFTYPE32 void MFTYPE McalTransformResult(MIL_ID CalibrationOrImageId,
                                         MIL_INT TransformType,
                                         MIL_INT DataType,
                                         double Data,
                                         double * ResData);

MFTYPE32 void MFTYPE McalTransformResultInRegion(MIL_ID CalibrationOrImageId,
                                                 MIL_INT TransformType,
                                                 MIL_INT DataType,
                                                 MIL_INT lBoxMinX,
                                                 MIL_INT lBoxMinY,
                                                 MIL_INT lBoxMaxX,
                                                 MIL_INT lBoxMaxY,
                                                 double Data,
                                                 double * ResData);

MFTYPE32 void MFTYPE McalTransformImage(MIL_ID SrcImageId,
                                        MIL_ID DestImageOrLutId,
                                        MIL_ID CalibrationId,
                                        MIL_INT InterpolationType,
                                        MIL_INT OperationType,
                                        MIL_INT ControlFlag);

#if M_MIL_USE_64BIT
// Prototypes for 64 bits OSs
MFTYPE32 void MFTYPE McalControlInt64(MIL_ID CalibrationOrImageId,
                                      MIL_INT   Parameter,
                                      MIL_INT64 Value);
MFTYPE32 void MFTYPE McalControlDouble(MIL_ID CalibrationOrImageId,
                                       MIL_INT   Parameter,
                                       double Value);
#else
// Prototypes for 32 bits OSs
#define McalControlInt64  McalControl
#define McalControlDouble McalControl
MFTYPE32 void MFTYPE McalControl(MIL_ID CalibrationOrImageId,
                                 MIL_INT   Parameter,
                                 double Value);
#endif

MFTYPE32 MIL_INT MFTYPE McalInquire(MIL_ID CalibrationOrMilId,
                                    MIL_INT   Parameter,
                                    void   * ValuePtr);

MFTYPE32 MIL_INT MFTYPE McalInquireSingle(MIL_ID CalibrationId,
                                          MIL_INT Index,
                                          MIL_INT Parameter,
                                          void   *ValuePtr);

MFTYPE32 void MFTYPE McalGetCoordinateSystem(MIL_ID  CalibrationOrImageId,
                                             MIL_INT TargetCoordinateSystem,
                                             MIL_INT ReferenceCoordinateSystem,
                                             MIL_INT InquireType,
                                             MIL_ID  MatrixId,
                                             double* Param1,
                                             double* Param2,
                                             double* Param3,
                                             double* Param4);

MFTYPE32 void MFTYPE McalSetCoordinateSystem(MIL_ID  CalibrationId,
                                             MIL_INT TargetCoordinateSystem,
                                             MIL_INT ReferenceCoordinateSystem,
                                             MIL_INT TransformType,
                                             MIL_ID  MatrixId,
                                             double  Param1,
                                             double  Param2,
                                             double  Param3,
                                             double  Param4);

#if M_MIL_USE_UNICODE
MFTYPE32 void MFTYPE McalSaveA(const char* FileName,
                               MIL_ID CalibrationId,
                               MIL_INT ControlFlag);

MFTYPE32 MIL_ID MFTYPE McalRestoreA(const char* FileName,
                                    MIL_ID SystemId,
                                    MIL_INT ControlFlag,
                                    MIL_ID * CalibrationIdPtr);

MFTYPE32 void MFTYPE McalStreamA(char* MemPtrOrFileName,
                                 MIL_ID SystemId,
                                 MIL_INT   Operation, MIL_INT StreamType,
                                 double Version,   MIL_INT ControlFlag,
                                 MIL_ID *ObjectIdPtr, MIL_INT   *SizeByteVarPtr   );
MFTYPE32 void MFTYPE McalSaveW(MIL_CONST_TEXT_PTR FileName,
                               MIL_ID CalibrationId,
                               MIL_INT ControlFlag);

MFTYPE32 MIL_ID MFTYPE McalRestoreW(MIL_CONST_TEXT_PTR FileName,
                                    MIL_ID SystemId,
                                    MIL_INT ControlFlag,
                                    MIL_ID * CalibrationIdPtr);

MFTYPE32 void MFTYPE McalStreamW(MIL_TEXT_PTR MemPtrOrFileName,
                                 MIL_ID SystemId,
                                 MIL_INT   Operation, MIL_INT StreamType,
                                 double Version,   MIL_INT ControlFlag,
                                 MIL_ID *ObjectIdPtr, MIL_INT   *SizeByteVarPtr   );

#if M_MIL_UNICODE_API
#define McalSave           McalSaveW
#define McalRestore        McalRestoreW
#define McalStream         McalStreamW
#else
#define McalSave           McalSaveA
#define McalRestore        McalRestoreA
#define McalStream         McalStreamA
#endif

#else
MFTYPE32 void MFTYPE McalSave(MIL_CONST_TEXT_PTR FileName,
                              MIL_ID CalibrationId,
                              MIL_INT ControlFlag);

MFTYPE32 MIL_ID MFTYPE McalRestore(MIL_CONST_TEXT_PTR FileName,
                                   MIL_ID SystemId,
                                   MIL_INT ControlFlag,
                                   MIL_ID * CalibrationIdPtr);

MFTYPE32 void MFTYPE McalStream( MIL_TEXT_PTR MemPtrOrFileName,
                                 MIL_ID SystemId,
                                 MIL_INT   Operation, MIL_INT StreamType,
                                 double Version,   MIL_INT ControlFlag,
                                 MIL_ID *ObjectIdPtr, MIL_INT   *SizeByteVarPtr   );
#endif

#endif /* #ifdef __midl */

/* C++ directive if needed */
#ifdef __cplusplus
}
#endif

#if M_MIL_USE_64BIT
#ifdef __cplusplus
//////////////////////////////////////////////////////////////
// McalControl function definition when compiling c++ files
//////////////////////////////////////////////////////////////
#if !M_MIL_USE_LINUX
inline void McalControl(MIL_ID CalibrationId, MIL_INT   Parameter, int Value)
   {
   McalControlInt64(CalibrationId, Parameter, Value);
   };
#endif
inline void McalControl(MIL_ID CalibrationId, MIL_INT   Parameter, MIL_INT32 Value)
   {
   McalControlInt64(CalibrationId, Parameter, Value);
   };

inline void McalControl(MIL_ID CalibrationId, MIL_INT   Parameter, MIL_INT64 Value)
   {
   McalControlInt64(CalibrationId, Parameter, Value);
   };

inline void McalControl(MIL_ID CalibrationId, MIL_INT   Parameter, double Value)
   {
   McalControlDouble(CalibrationId, Parameter, Value);
   };
#else
//////////////////////////////////////////////////////////////
// For C file, call the default function, i.e. Int64 one
//////////////////////////////////////////////////////////////
#define McalControl  McalControlDouble

#endif // __cplusplus
#endif // M_MIL_USE_64BIT

#if M_MIL_USE_SAFE_TYPE

//////////////////////////////////////////////////////////////
// See milos.h for explanation about these functions.
//////////////////////////////////////////////////////////////

// ----------------------------------------------------------
// McalInquire

inline MFTYPE32 MIL_INT MFTYPE McalInquireUnsafe  (MIL_ID CalibrationOrMilId, MIL_INT Parameter, void       MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, int          ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT8   MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT16  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT32  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT64  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, float      MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_DOUBLE MPTYPE * ValuePtr);
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT8  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT16 MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT32 MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT64 MPTYPE * ValuePtr);
#endif
#if M_MIL_SAFE_TYPE_ADD_WCHAR_T
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, wchar_t    MPTYPE * ValuePtr);
#endif

// ----------------------------------------------------------
// McalInquireSingle
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleUnsafe  (MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, void       MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, int          ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT8   MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT16  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT32  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT64  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, float      MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_DOUBLE MPTYPE * ValuePtr);
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT8  MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT16 MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT32 MPTYPE * ValuePtr);
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT64 MPTYPE * ValuePtr);
#endif
#if M_MIL_SAFE_TYPE_ADD_WCHAR_T
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, wchar_t    MPTYPE * ValuePtr);
#endif

// ----------------------------------------------------------
// McalInquire

inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, int ValuePtr)
   {
   if (ValuePtr)
      SafeTypeError(MT("McalInquire"));

   return McalInquire(CalibrationOrMilId, Parameter, NULL);
   }

inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, int ValuePtr)
   {
   if (ValuePtr)
      SafeTypeError(MT("McalInquireSingle"));

   return McalInquireSingle(CalibrationId, Index, Parameter, NULL);
   }


inline MFTYPE32 MIL_INT MFTYPE McalInquireExecute (MIL_ID CalibrationOrMilId, MIL_INT Parameter, void       MPTYPE *ValuePtr, MIL_INT GivenType)
   {
   MIL_INT RequiredType = (Parameter & (M_TYPE_MIL_INT32 | 
#if (BW_COMPATIBILITY != 0x80)
      M_TYPE_MIL_INT64 | 
#endif
      M_TYPE_DOUBLE | M_TYPE_FLOAT |M_TYPE_SHORT | M_TYPE_CHAR | M_TYPE_MIL_ID));
   if (RequiredType == 0)
      RequiredType = M_TYPE_DOUBLE;
   ReplaceTypeMilIdByTypeMilIntXX(&RequiredType);

   if (RequiredType != GivenType)
      SafeTypeError(MT("McalInquire"));

   return McalInquire(CalibrationOrMilId, Parameter, ValuePtr);
   }

inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleExecute (MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, void       MPTYPE *ValuePtr, MIL_INT GivenType)
   {
   MIL_INT RequiredType = (Parameter & (M_TYPE_MIL_INT32 | 
#if (BW_COMPATIBILITY != 0x80)
      M_TYPE_MIL_INT64 | 
#endif
      M_TYPE_DOUBLE | M_TYPE_FLOAT |M_TYPE_SHORT | M_TYPE_CHAR ));
   if (RequiredType == 0)
      RequiredType = M_TYPE_DOUBLE;

   if (RequiredType != GivenType)
      SafeTypeError(MT("McalInquireSingle"));

   return McalInquireSingle(CalibrationId, Index, Parameter, ValuePtr);
   }

inline MFTYPE32 MIL_INT MFTYPE McalInquireUnsafe  (MIL_ID CalibrationOrMilId, MIL_INT Parameter, void       MPTYPE * ValuePtr) {return McalInquire       (CalibrationOrMilId, Parameter, ValuePtr                  );}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT8   MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_CHAR);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT16  MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_SHORT);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT32  MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_INT64  MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_MIL_INT64);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, float      MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_FLOAT   );}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_DOUBLE MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_DOUBLE   );}
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT8  MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_CHAR);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT16 MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_SHORT);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT32 MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, MIL_UINT64 MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_MIL_INT64);}
#endif
#if M_MIL_SAFE_TYPE_ADD_WCHAR_T
inline MFTYPE32 MIL_INT MFTYPE McalInquireSafeType(MIL_ID CalibrationOrMilId, MIL_INT Parameter, wchar_t    MPTYPE * ValuePtr) {return McalInquireExecute(CalibrationOrMilId, Parameter, ValuePtr, M_TYPE_SHORT);}
#endif

inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleUnsafe  (MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, void       MPTYPE * ValuePtr) {return McalInquireSingle       (CalibrationId, Index, Parameter, ValuePtr                  );}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT8   MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_CHAR);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT16  MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_SHORT);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT32  MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_INT64  MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_MIL_INT64);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, float      MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_FLOAT   );}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_DOUBLE MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_DOUBLE   );}
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT8  MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_CHAR);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT16 MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_SHORT);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT32 MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_MIL_INT32);}
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, MIL_UINT64 MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_MIL_INT64);}
#endif
#if M_MIL_SAFE_TYPE_ADD_WCHAR_T
inline MFTYPE32 MIL_INT MFTYPE McalInquireSingleSafeType(MIL_ID CalibrationId, MIL_INT Index, MIL_INT Parameter, wchar_t    MPTYPE * ValuePtr) {return McalInquireSingleExecute(CalibrationId, Index, Parameter, ValuePtr, M_TYPE_SHORT);}
#endif

#define McalInquire          McalInquireSafeType
#define McalInquireSingle    McalInquireSingleSafeType

#else // #if M_MIL_USE_SAFE_TYPE

#define McalInquireUnsafe          McalInquire
#define McalInquireSingleUnsafe    McalInquireSingle

#endif // #if M_MIL_USE_SAFE_TYPE

#endif /* !M_MIL_LITE */

#endif /* __MILCAL_H__ */
