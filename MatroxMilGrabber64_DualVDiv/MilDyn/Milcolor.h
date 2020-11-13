////////////////////////////////////////////////////////////////////////////////
//! 
//! \file  milcolor.h 
//! 
//! \brief This file contains the defines and the prototypes for the
//!         MIL color module. (Mcol...). 
//! 
//! AUTHOR:  Matrox Imaging dept.
//!
//! COPYRIGHT NOTICE:
//! Copyright © 2006 - 2008 Matrox Electronic Systems Ltd. 
//! All Rights Reserved 
//  Revision:  9.01.0855
//////////////////////////////////////////////////////////////////////////////// 

#ifndef __MILCOLOR_H__
#define __MILCOLOR_H__

#if (!M_MIL_LITE) // MIL FULL ONLY

/* C++ directive if needed */
#ifdef __cplusplus
extern "C"
   {
#endif

/***************************************************************************/
/*                       MilColor CAPI defines                             */
/***************************************************************************/
#define M_CONTEXT                                  0x08000000L // Already defined in    milmod.h, milreg.h, milstr.h, milocr.h
#define M_GENERAL                                  0x20000000L // Already defined in    milmod.h, milreg.h, milstr.h, milocr.h
#define M_ALL                                      0x40000000L // Already defined in    milmod.h, milmeas.h, milocr.h,  milpat.h,  mil.h, miledge.h

// High level result types modifiers
#define M_SUPPORTED                                0x20000000L // Also defined in milmod.h, miledge.h, milstr.h, milmetrol.h, milcolor.h, utillib/milexch/dataexchangerdeclares.h
#define M_AVAILABLE                                0x40000000L // Also defined in milmod.h, miledge.h, milstr.h, milmetrol.h, milcolor.h, utillib/milexch/dataexchangerdeclares.h

// Allocation Macros
#define M_SAMPLE_LABEL_TAG                         0x01000000L  // must not clash with M_ALL, M_GENERAL, M_CONTEXT
#define M_SAMPLE_INDEX_TAG                         0x02000000L  // must not clash with M_ALL, M_GENERAL, M_CONTEXT

#define M_SAMPLE_LABEL(lbl)                        ((lbl) | M_SAMPLE_LABEL_TAG)
#define M_SAMPLE_INDEX(idx)                        ((idx) | M_SAMPLE_INDEX_TAG)

// Allocation Flag
#define M_COLOR_MATCHING_RESULT                    0x00001000L // used in McolAllocResult
#define M_COLOR_MATCHING                           0x00000100L // used in McolAlloc
#define M_COLOR_PROJECTION                         0x00000200L // used in McolAlloc

// Control Flags for Context
#define M_DISTANCE_TYPE                            1L          // used in matching context   (read-only)
#define M_PREPROCESSED                             14L         // used in matching context      *also defined in milpat.h, milmod.h, milmetrol.h, milstr.h
#define M_DISTANCE_TOLERANCE_MODE                  20L         // used in matching context
#define M_DISTANCE_TOLERANCE_METHOD                M_DISTANCE_TOLERANCE_MODE                 // we need this deprecated synonym because the early access of PP1 was released with it
#define M_OUTLIER_LABEL                            30L         // used in matching context
#define M_OUTLIER_DRAW_COLOR                       31L         // used in matching context
#define M_BACKGROUND_DRAW_COLOR                    32L         // used in matching context
#define M_NUMBER_OF_SAMPLES                        35L         // used in matching context
#define M_PRECONVERT_MODE                          40L         // used in matching context   (read-only)
#define M_MATCH_MODE                               50L         // used in matching context   (read-only)
#define M_MATCH_METHOD                             M_MATCH_MODE // we need this deprecated synonym because the early access of PP1 was released with it
#define M_BAND_MODE                                90L         // used in matching context
#define M_ACCEPTANCE                               200L        // used in matching context      *also used in milpat.h, milmod.h
#define M_ACCEPTANCE_RELEVANCE                     202L        // used in matching context
#define M_RELEVANCE_ACCEPTANCE                     M_ACCEPTANCE_RELEVANCE                    // we need this deprecated synonym because the early access of PP1 was released with it
#define M_TIE_EPSILON                              205L        // used in matching context
#define M_SAMPLE_BUFFER_FORMAT                     210L        // used in matching context      *undocumented internal flag
#define M_DISTANCE_IMAGE_NORMALIZE                 215L        // used in matching context and matching result
#define M_DISTANCE_PARAM_1                         221L        // used in matching context
#define M_DISTANCE_PARAM_2                         222L        // used in matching context
#define M_DISTANCE_PARAM_3                         223L        // used in matching context
#define M_DISTANCE_PARAM_4                         224L        // used in matching context
#define M_DISTANCE_PARAM_5                         225L        // used in matching context
#define M_ENCODING                          0x00008005L        // used in matching context     *also used in milstr.h, milcode.h
#define M_PRECONVERT_GAMMA                         226L        // used in matching context
// Encoding controls per-band
#define M_SCALE_BAND_0                             241L        // used in matching context
#define M_SCALE_BAND_1                             242L        // used in matching context
#define M_SCALE_BAND_2                             243L        // used in matching context
#define M_OFFSET_BAND_0                            244L        // used in matching context
#define M_OFFSET_BAND_1                            245L        // used in matching context
#define M_OFFSET_BAND_2                            246L        // used in matching context
// Internal controls
#define M_SCORE_MAX_DIST_PARAM                     250L        // used in matching context
#define M_SCORE_K_PARAM                            252L        // used in matching context

// Controls for output images (read-only)
#define M_LABEL_PIXEL_IMAGE_SIZE_BIT               352L        // used in matching result and matching context
#define M_LABEL_AREA_IMAGE_SIZE_BIT                372L        // used in matching result and matching context

// Alloc-time controls (read-only)
#define M_COLOR_SPACE                              230L        // used in matching context   (read-only)

// Supported color spaces
#define M_RGB                                      8L        // used in McolAlloc             *also defined in mil.h
#define M_CIELAB                                  41L        // used in McolAlloc
#define M_HSL                                      2L        // used in McolAlloc
#define M_YUV                                      4L        // used in McolAlloc             *also defined in mil.h

// Control/Inquire Flags for Samples
#define M_DISTANCE_TOLERANCE                       120L        // used in matching samples   
#define M_SAMPLE_TYPE                              121L        // used in matching samples
#define M_SAMPLE_IMAGE_SIZE_X                      122L        // used in matching samples
#define M_SAMPLE_IMAGE_SIZE_Y                      123L        // used in matching samples
#define M_SAMPLE_IMAGE_SIZE_BIT                    124L        // used in matching samples
#define M_SAMPLE_IMAGE_SIZE_BAND                   125L        // used in matching samples
#define M_SAMPLE_IMAGE_SIGN                        126L        // used in matching samples
#define M_SAMPLE_IMAGE_TYPE                        127L        // used in matching samples
#define M_SAMPLE_MASK_SIZE_X                       128L        // used in matching samples
#define M_SAMPLE_MASK_SIZE_Y                       129L        // used in matching samples
#define M_SAMPLE_MASK_SIZE_BIT                     130L        // used in matching samples
#define M_SAMPLE_MASK_SIZE_BAND                    131L        // used in matching samples
#define M_SAMPLE_MASK_SIGN                         132L        // used in matching samples
#define M_SAMPLE_MASK_TYPE                         133L        // used in matching samples
#define M_SAMPLE_MASKED                            135L        // used in matching samples
#define M_SAMPLE_COLOR_BAND_0                      140L        // used in matching samples
#define M_SAMPLE_COLOR_BAND_1                      141L        // used in matching samples
#define M_SAMPLE_COLOR_BAND_2                      142L        // used in matching samples
#define M_MATCH_SAMPLE_COLOR_BAND_0                145L        // used in matching samples
#define M_MATCH_SAMPLE_COLOR_BAND_1                146L        // used in matching samples
#define M_MATCH_SAMPLE_COLOR_BAND_2                147L        // used in matching samples
#define M_SAMPLE_LUT_SIZE_X                        150L        // used in matching context
#define M_SAMPLE_LUT_SIZE_Y                        151L        // used in matching context
#define M_SAMPLE_LUT_SIZE_BIT                      152L        // used in matching context
#define M_SAMPLE_LUT_SIZE_BAND                     153L        // used in matching context
#define M_SAMPLE_LUT_SIGN                          154L        // used in matching context
#define M_SAMPLE_LUT_TYPE                          155L        // used in matching context
#define M_SAMPLE_ABSOLUTE_TOLERANCE                160L        // used in matching context

// Control Values
#define M_NONE                                     0x08000000L // value for M_PRECONVERT_MODE  *also defined in milcal.h, 
#define M_EUCLIDEAN                                2L          // value for M_DISTANCE_TYPE
#define M_MANHATTAN                                3L          // value for M_DISTANCE_TYPE
#define M_DELTA_E                                  4L          // value for M_DISTANCE_TYPE
#define M_MAHALANOBIS_SAMPLE                       5L          // value for M_DISTANCE_TYPE
#define M_MAHALANOBIS                              M_MAHALANOBIS_SAMPLE
#define M_MAHALANOBIS_TARGET                       8L          // value for M_DISTANCE_TYPE
#define M_EUCLIDEAN_SQR                            22L         // value for M_DISTANCE_TYPE
#define M_MANHATTAN_SQR                            23L         // value for M_DISTANCE_TYPE
#define M_DELTA_E_SQR                              24L         // value for M_DISTANCE_TYPE
#define M_MAHALANOBIS_SAMPLE_SQR                   25L         // value for M_DISTANCE_TYPE
#define M_MAHALANOBIS_TARGET_SQR                   28L         // value for M_DISTANCE_TYPE
#define M_ABSOLUTE                                 1L          // value for M_DISTANCE_TOLERANCE_MODE  *also defined in milpat.h, 
#define M_RELATIVE                                 21L         // value for M_DISTANCE_TOLERANCE_MODE
#define M_SAMPLE_STDDEV                            33L         // value for M_DISTANCE_TOLERANCE_MODE
#define M_AUTO                                     444L        // value for M_DISTANCE_TOLERANCE *also defined in miledge.h, milmetrol.h, milmod.h, milocr.h
#define M_NONE                                     0x08000000L // value for M_PRECONVERT_MODE   *also defined in milstr.h, milcal.h
#define M_CIELAB                                   41L         // value for M_PRECONVERT_MODE
#define M_MIN_DIST_VOTE                            51L         // value for M_MATCH_MODE
#define M_STAT_MIN_DIST                            52L         // value for M_MATCH_MODE
#define M_ALL_BAND                                 -1L         // value for M_BAND_MODE              *also defined in MIL.H
#define M_ALL_BANDS                                M_ALL_BAND  // value for M_BAND_MODE              *also defined in MIL.H
#define M_TRANSPARENT                              0x01000059L // value for M_BACKGROUND_DRAW_COLOR  *also defined in MIL.H
// To work independently from RGB space we define new names for bands
#define M_COLOR_BAND_0                             0x00000100  // value for M_BAND_MODE & McolDistance
#define M_COLOR_BAND_1                             0x00000200  // value for M_BAND_MODE & McolDistance
#define M_COLOR_BAND_2                             0x00000400  // value for M_BAND_MODE & McolDistance
#define M_MAX_NORMALIZE                            0           // value for M_DISTANCE_IMAGE_NORMALIZE
#define M_NO_NORMALIZE                             -1.0        // value for M_DISTANCE_IMAGE_NORMALIZE
// Encodings
#define M_ENCODING_START                           60L
#define M_8BIT                                     M_ENCODING_START+8L          // used in matching context
#define M_10BIT                                    M_ENCODING_START+10L         // used in matching context
#define M_12BIT                                    M_ENCODING_START+12L         // used in matching context
#define M_14BIT                                    M_ENCODING_START+14L         // used in matching context
#define M_16BIT                                    M_ENCODING_START+16L         // used in matching context
#define M_USER_DEFINED                             21L                          // used in matching context      *also defined in MIL.H

// Control/Inquire Flags for Matching Result
#define M_GENERATE_LABEL_PIXEL_IMAGE               300L        // used in matching result
#define M_GENERATE_SAMPLE_COLOR_LUT                305L        // used in matching result
#define M_GENERATE_DISTANCE_IMAGE                  310L        // used in matching result
#define M_SAVE_AREA_IMAGE                          315L        // used in matching result
#define M_DRAW_RELATIVE_ORIGIN_X                   319L        // Already defined in miledge.h
#define M_DRAW_RELATIVE_ORIGIN_Y                   320L        // Already defined in miledge.h
#define M_DISTANCE_IMAGE_NORMALIZE                 215L        // used in matching context and macthing result

// Result type Flag for Matching Result
#define M_AREA_IMAGE_SIZE_X                        330L        // used in matching result
#define M_AREA_IMAGE_SIZE_Y                        331L        // used in matching result
#define M_AREA_IMAGE_SIZE_BIT                      332L        // used in matching result
#define M_AREA_IMAGE_SIZE_BAND                     333L        // used in matching result
#define M_AREA_IMAGE_SIGN                          334L        // used in matching result
#define M_AREA_IMAGE_TYPE                          335L        // used in matching result

#define M_DISTANCE_IMAGE_SIZE_X                    340L        // used in matching result
#define M_DISTANCE_IMAGE_SIZE_Y                    341L        // used in matching result
#define M_DISTANCE_IMAGE_SIZE_BIT                  342L        // used in matching result
#define M_DISTANCE_IMAGE_SIZE_BAND                 343L        // used in matching result
#define M_DISTANCE_IMAGE_SIGN                      344L        // used in matching result
#define M_DISTANCE_IMAGE_TYPE                      345L        // used in matching result

#define M_LABEL_PIXEL_IMAGE_SIZE_X                 350L        // used in matching result
#define M_LABEL_PIXEL_IMAGE_SIZE_Y                 351L        // used in matching result
#define M_LABEL_PIXEL_IMAGE_SIZE_BIT               352L        // used in matching result and matching context
#define M_LABEL_PIXEL_IMAGE_SIZE_BAND              353L        // used in matching result
#define M_LABEL_PIXEL_IMAGE_SIGN                   354L        // used in matching result
#define M_LABEL_PIXEL_IMAGE_TYPE                   355L        // used in matching result

#define M_COLORED_LABEL_PIXEL_IMAGE_SIZE_X         360L        // used in matching result
#define M_COLORED_LABEL_PIXEL_IMAGE_SIZE_Y         361L        // used in matching result
#define M_COLORED_LABEL_PIXEL_IMAGE_SIZE_BIT       362L        // used in matching result
#define M_COLORED_LABEL_PIXEL_IMAGE_SIZE_BAND      363L        // used in matching result
#define M_COLORED_LABEL_PIXEL_IMAGE_SIGN           364L        // used in matching result
#define M_COLORED_LABEL_PIXEL_IMAGE_TYPE           365L        // used in matching result

#define M_LABEL_AREA_IMAGE_SIZE_X                  370L        // used in matching result
#define M_LABEL_AREA_IMAGE_SIZE_Y                  371L        // used in matching result 
#define M_LABEL_AREA_IMAGE_SIZE_BIT                372L        // used in matching result and matching context
#define M_LABEL_AREA_IMAGE_SIZE_BAND               373L        // used in matching result
#define M_LABEL_AREA_IMAGE_SIGN                    374L        // used in matching result
#define M_LABEL_AREA_IMAGE_TYPE                    375L        // used in matching result

#define M_COLORED_LABEL_AREA_IMAGE_SIZE_X          380L        // used in matching result
#define M_COLORED_LABEL_AREA_IMAGE_SIZE_Y          381L        // used in matching result
#define M_COLORED_LABEL_AREA_IMAGE_SIZE_BIT        382L        // used in matching result
#define M_COLORED_LABEL_AREA_IMAGE_SIZE_BAND       383L        // used in matching result
#define M_COLORED_LABEL_AREA_IMAGE_SIGN            384L        // used in matching result
#define M_COLORED_LABEL_AREA_IMAGE_TYPE            385L        // used in matching result

#define M_MAX_DISTANCE                             12L         // used in matching result       *also define in thresholdcontext.h (milim)
#define M_LABEL_VALUE                              1L          // used in matching result       *also defined in miledge.h, milblob.h, milmetrol.h
#define M_SAMPLE_LABEL_VALUE                       M_LABEL_VALUE
#define M_AREA_LABEL_VALUE                         802L        // used in matching result
#define M_BEST_MATCH_INDEX                         804L        // used in matching result
#define M_BEST_MATCH_LABEL                         805L        // used in matching result
#define M_COLOR_DISTANCE                           806L        // used in matching result
#define M_SCORE                                    0x00001400L // used in matching result *also defined in milpat.h
#define M_NUMBER_OF_AREAS                          832L        // used in matching context
#define M_OUTLIER_COVERAGE                         808L        // used in matching result
#define M_STATUS                                   0x00008002L // used in matching result Already defined in milcode.h (in decimal: 32770)
#define M_SAMPLE_MATCH_STATUS                      809L        // used in matching result
#define M_NB_BEST_MATCH_SAMPLE                     810L        // used in matching result
#define M_NB_MATCH_SAMPLE                          811L        // used in matching result
#define M_NB_NO_MATCH_SAMPLE                       812L        // used in matching result
#define M_MATCH_INDEX                              820L        // used in matching result
#define M_MATCH_LABEL                              822L        // used in matching result
#define M_NO_MATCH_INDEX                           824L        // used in matching result
#define M_NO_MATCH_LABEL                           826L        // used in matching result
#define M_SAMPLE_COVERAGE                          830L        // used in matching result
#define M_SAMPLE_PIXEL_COUNT                       832L        // used in matching result
#define M_SCORE_RELEVANCE                          834L        // used in matching result
#define M_RELEVANCE_SCORE                          M_SCORE_RELEVANCE // we need this deprecated synonym because the early access of PP1 was released with it
#define M_AREA_PIXEL_COUNT                         838L        // used in matching result

// Result values
#define M_SUCCESS                                  0x00000000L // Already defined in milreg.h, used as status result
#define M_FAILURE                                  0x00000001L // Already defined in milreg.h, used as status result
#define M_TIE                                      0x00000002L // used as status result
#define M_MATCH                                    6L          // Already defined in milim.h, used as sample match status result
#define M_NO_MATCH                                 7L          // used as sample match status result

// Operation Flag
#define M_DELETE                                   3L          // used in McolDefine *also defined in milmod.h, miledge.h, milblob.h, milmetrol.h
#define M_IMAGE                                    0x00000004L // used in McolDefine *also defined in mil.h, milmod.h
#define M_TRIPLET                                  8L          // used in McolDefine

// Draw flags
#define M_DRAW_LABEL_PIXEL_IMAGE                   400L        // used in McolDraw
#define M_DRAW_COLORED_LABEL_PIXEL_IMAGE           402L        // used in McolDraw
#define M_DRAW_LABEL_AREA_IMAGE                    403L        // used in McolDraw
#define M_DRAW_COLORED_LABEL_AREA_IMAGE            404L        // used in McolDraw
#define M_DRAW_DISTANCE_IMAGE                      405L        // used in McolDraw
#define M_DRAW_AREA_IMAGE                          410L        // used in McolDraw
#define M_DRAW_SAMPLE_IMAGE                        415L        // used in McolDraw
#define M_DRAW_SAMPLE_MASK_IMAGE                   416L        // used in McolDraw
#define M_DRAW_SAMPLE_COLOR_LUT                    420L        // used in McolDraw

// Projection flags
#define M_COLOR_SEPARATION                         600L        // used in McolProject
#define M_PRINCIPAL_COMPONENT_PROJECTION           605L        // used in McolProject
#define M_COVARIANCE                               610L        // used in McolProject (McolSetMethod)
#define M_PRINCIPAL_COMPONENTS                     615L        // used in McolProject
#define M_MASK_CONTRAST_ENHANCEMENT                620L        // used in McolProject
#define M_BACKGROUND_LABEL                          43L        // used in McolProject
#define M_SELECTED_LABEL                           103L        // used in McolProject
#define M_REJECTED_LABEL                            73L        // used in McolProject
#define M_SOURCE_LABEL                             163L        // used in McolProject
#define M_BRIGHT_LABEL                             253L        // used in McolProject
#define M_DARK_LABEL                                83L        // used in McolProject
// McolProject result statuses
#define M_NO_BACKGROUND_DEFINED                    650L        // used in McolProject
#define M_NO_SELECTED_DEFINED                      652L        // used in McolProject
#define M_NO_REJECTED_DEFINED                      654L        // used in McolProject
#define M_REJECTED_EQUAL_SELECTED                  660L        // used in McolProject
#define M_REJECTED_EQUAL_BACKGROUND                662L        // used in McolProject
#define M_SELECTED_EQUAL_BACKGROUND                664L        // used in McolProject
#define M_3_COLORS_COLLINEAR                       668L        // used in McolProject
#define M_NO_SOURCE_DEFINED                        670L        // used in McolProject
#define M_UNSTABLE_POLARITY                        672L        // used in McolProject
#define M_UNSTABLE_PRINCIPAL_COMPONENT_2           680L        // used in McolProject
#define M_UNSTABLE_PRINCIPAL_COMPONENTS_12         682L        // used in McolProject
#define M_UNSTABLE_PRINCIPAL_COMPONENTS_012        684L        // used in McolProject
/***************************************************************************/
/*               MilColor CAPI function prototypes                         */
/***************************************************************************/

#ifndef __midl // MIDL compiler used by ActiveMIL

MFTYPE32 MIL_ID MFTYPE McolAlloc(MIL_ID SystemId, 
                                 MIL_INT ObjectType, 
                                 MIL_INT WorkingColorSpace, 
                                 MIL_ID  ColorProfileId,
                                 MIL_INT ControlFlag, 
                                 MIL_ID* ObjectPtr);
MFTYPE32 MIL_ID MFTYPE McolAllocResult(MIL_ID SystemId, MIL_INT ObjectType, MIL_INT ControlFlag, MIL_ID* ObjectPtr);
MFTYPE32 void   MFTYPE McolDefine(MIL_ID     ColorObjectContextID,
                                  MIL_ID     SrcImageID,
                                  MIL_INT    UserLabel,
                                  MIL_INT    ColorSampleType,
                                  MIL_DOUBLE Param1, 
                                  MIL_DOUBLE Param2,
                                  MIL_DOUBLE Param3,
                                  MIL_DOUBLE Param4);
MFTYPE32 void MFTYPE McolPreprocess(MIL_ID   ColorMatchingContextID,
                                    MIL_INT  ControlFlag);
MFTYPE32 void MFTYPE McolMatch(MIL_ID ColorMatchingContextID, 
                               MIL_ID TargetColorImageID, 
                               MIL_ID TargetColorProfileID,
                               MIL_ID AreaImageID, 
                               MIL_ID ColorResultorDestImageID,
                               MIL_INT controlFlag);
MFTYPE32 void MFTYPE McolFree(MIL_ID ColorObjectID);
MFTYPE32 MIL_INT MFTYPE McolInquire(MIL_ID   ContextId,
                                    MIL_INT  UserIndex,
                                    MIL_INT  InquireType,
                                    void*    UserVarPtr);
MFTYPE32 void MFTYPE McolGetResult(MIL_ID   ResultId,
                                   MIL_INT  AreaLabel,
                                   MIL_INT  SampleLabelOrIndex,
                                   MIL_INT  ResultType,
                                   void*    ResultArrayPtr);

MFTYPE32 void MFTYPE McolDistance(MIL_ID Src1, 
                                  MIL_ID Src2, 
                                  MIL_ID Dest, 
                                  MIL_ID Mask,
                                  MIL_ID Parameters,
                                  MIL_INT DistType,
                                  double  NormalizeFlag,
                                  MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McolDraw(MIL_ID   GraphContId,
                              MIL_ID   ColorObjectID,
                              MIL_ID   DestImageID,
                              MIL_INT  DrawOperation,
                              MIL_INT  AreaLabel,
                              MIL_INT  SampleIndexOrLabel,                              
                              MIL_INT  ControlFlag);

MFTYPE32 void MFTYPE McolMask(MIL_ID   ContextId,
                              MIL_INT  SampleIndexOrLabel,
                              MIL_ID   MaskBufferId,
                              MIL_INT  MaskType,
                              MIL_INT  ControlFlag);

MFTYPE32 void MFTYPE McolSetMethod(MIL_ID   ContextId,
                                   MIL_INT  OperationMode,
                                   MIL_INT  DistanceType, 
                                   MIL_INT  PreconvertMode,
                                   MIL_INT  ControlFlag);

MFTYPE32 void MFTYPE McolProject(MIL_ID Src,
                                 MIL_ID Colors,
                                 MIL_ID Dest,
                                 MIL_ID DestMask,
                                 MIL_INT Operation,
                                 MIL_INT ControlFlag,
                                 MIL_INT MPTYPE *ResultStatus);

#if M_MIL_USE_64BIT
// Prototypes for 64 bits OSs
MFTYPE32 void MFTYPE McolControlInt64    (MIL_ID      ContextId, 
                                          MIL_INT     Index,
                                          MIL_INT     ControlType, 
                                          MIL_INT64   ControlValue);

MFTYPE32 void MFTYPE McolControlDouble   (MIL_ID      ContextId, 
                                          MIL_INT     Index,
                                          MIL_INT     ControlType, 
                                          MIL_DOUBLE  ControlValue);

#else
// Prototypes for 32 bits OSs
#define McolControlInt64  McolControl
#define McolControlDouble McolControl
MFTYPE32 void MFTYPE McolControl         (MIL_ID      ContextId, 
                                          MIL_INT     Index,
                                          MIL_INT     ControlType, 
                                          MIL_DOUBLE  ControlValue);
#endif

#if M_MIL_USE_UNICODE

MFTYPE32 MIL_ID MFTYPE McolRestoreW(MIL_CONST_TEXT_PTR FileName, 
                                    MIL_ID SystemId, 
                                    MIL_INT ControlFlag, 
                                    MIL_ID *ContextIdPtr);

MFTYPE32 void MFTYPE McolSaveW(MIL_CONST_TEXT_PTR FileName, 
                               MIL_ID ContextId, 
                               MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McolStreamW(MIL_TEXT_PTR MemPtrOrFileName, 
                                 MIL_ID SystemId, 
                                 MIL_INT Operation, 
                                 MIL_INT StreamType, 
                                 double Version, 
                                 MIL_INT ControlFlag, 
                                 MIL_ID *ContextIdPtr, 
                                 MIL_INT *SizeByteVarPtr);

MFTYPE32 MIL_ID MFTYPE McolRestoreA(const char* FileName, 
                                    MIL_ID SystemId, 
                                    MIL_INT ControlFlag, 
                                    MIL_ID *ContextIdPtr);

MFTYPE32 void MFTYPE McolSaveA(const char* FileName, 
                               MIL_ID ContextId, 
                               MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McolStreamA(char* MemPtrOrFileName, 
                                 MIL_ID SystemId, 
                                 MIL_INT Operation, 
                                 MIL_INT StreamType, 
                                 double Version, 
                                 MIL_INT ControlFlag, 
                                 MIL_ID *ContextIdPtr, 
                                 MIL_INT *SizeByteVarPtr);

#if M_MIL_UNICODE_API
#define McolSave           McolSaveW
#define McolRestore        McolRestoreW
#define McolStream         McolStreamW
#else
#define McolSave           McolSaveA
#define McolRestore        McolRestoreA
#define McolStream         McolStreamA
#endif

#else

MFTYPE32 MIL_ID MFTYPE McolRestore(MIL_CONST_TEXT_PTR FileName, 
                                   MIL_ID SystemId, 
                                   MIL_INT ControlFlag, 
                                   MIL_ID *ContextIdPtr);

MFTYPE32 void MFTYPE McolSave(MIL_CONST_TEXT_PTR FileName, 
                              MIL_ID ContextId, 
                              MIL_INT ControlFlag);

MFTYPE32 void MFTYPE McolStream(MIL_TEXT_PTR MemPtrOrFileName, 
                                MIL_ID SystemId, 
                                MIL_INT Operation, 
                                MIL_INT StreamType, 
                                double Version, 
                                MIL_INT ControlFlag, 
                                MIL_ID *ContextIdPtr, 
                                MIL_INT *SizeByteVarPtr);
#endif

#endif /* #ifdef __midl */

   /* C++ directive if needed */
#ifdef __cplusplus
   }
#endif

#if M_MIL_USE_64BIT
#ifdef __cplusplus
//////////////////////////////////////////////////////////////
// McolControl function definition when compiling c++ files
//////////////////////////////////////////////////////////////
#if !M_MIL_USE_LINUX
inline void McolControl(MIL_ID   ContextId, 
                        MIL_INT  Index,
                        MIL_INT  ControlType, 
                        int      ControlValue)
   {
   McolControlInt64(ContextId, Index, ControlType, ControlValue);
   };
#endif

inline void McolControl(MIL_ID      ContextId, 
                        MIL_INT     Index,
                        MIL_INT     ControlType, 
                        MIL_INT32   ControlValue)
   {
   McolControlInt64(ContextId, Index, ControlType, ControlValue);
   }
inline void McolControl(MIL_ID      ContextId, 
                        MIL_INT     Index,
                        MIL_INT     ControlType, 
                        MIL_INT64   ControlValue)
   {
   McolControlInt64(ContextId, Index, ControlType, ControlValue);
   }
inline void McolControl(MIL_ID      ContextId, 
                        MIL_INT     Index,
                        MIL_INT     ControlType, 
                        MIL_DOUBLE  ControlValue)
   {
   McolControlDouble(ContextId, Index, ControlType, ControlValue);
   }

#else
//////////////////////////////////////////////////////////////
// For C file, call the default function, i.e. Double one
//////////////////////////////////////////////////////////////
#define McolControl  McolControlDouble

#endif // __cplusplus
#endif // M_MIL_USE_64BIT

#if M_MIL_USE_SAFE_TYPE

   //////////////////////////////////////////////////////////////
   // See milos.h for explanation about these functions.
   //////////////////////////////////////////////////////////////

   //-------------------------------------------------------------------------------------
   //  McolGetResult

   inline MFTYPE32 void MFTYPE McolGetResultUnsafe  (MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, void          MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, int                   ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_INT8      MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_INT16     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_INT32     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_INT64     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, float         MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_DOUBLE    MPTYPE *ResultArrayPtr);
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_UINT8     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_UINT16    MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_UINT32    MPTYPE *ResultArrayPtr);
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, MIL_UINT64    MPTYPE *ResultArrayPtr);
#endif

   // ----------------------------------------------------------
   // McolInquire

   inline MFTYPE32 MIL_INT MFTYPE McolInquireUnsafe  (MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, void          MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_INT8      MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_INT16     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_INT32     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_INT64     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, float         MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_DOUBLE    MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, int           ResultArrayPtr);
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_UINT8     MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_UINT16    MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_UINT32    MPTYPE *ResultArrayPtr);
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT InquireType, MIL_UINT64    MPTYPE *ResultArrayPtr);
#endif

   // -------------------------------------------------------------------------
   // McolGetResult

   inline MFTYPE32 void MFTYPE McolGetResultSafeType (MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT ColorSampleIndexOrLabel, MIL_INT ResultType, int ResultPtr)
      {
      if (ResultPtr != 0)
         SafeTypeError(MT("McolGetResult"));

      McolGetResult(ResultId, AreaLabel, ColorSampleIndexOrLabel, ResultType, NULL);
      }

   inline void McolGetResultSafeTypeExecute (MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, void MPTYPE *ResultArrayPtr, MIL_INT GivenType)
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
         SafeTypeError(MT("McolGetResult"));

      McolGetResult(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr);
      }

   inline MFTYPE32 void MFTYPE McolGetResultUnsafe  (MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, void       MPTYPE *ResultArrayPtr){McolGetResult               (ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr);}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_INT8   MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_CHAR     );}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_INT16  MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_SHORT    );}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_INT32  MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_MIL_INT32);}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_INT64  MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_MIL_INT64);}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, float      MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_FLOAT    );}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_DOUBLE MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_DOUBLE   );}
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_UINT8  MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_CHAR     );}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_UINT16 MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_SHORT    );}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_UINT32 MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_MIL_INT32);}
   inline MFTYPE32 void MFTYPE McolGetResultSafeType(MIL_ID ResultId, MIL_INT AreaLabel, MIL_INT SampleOrLabelIndex, MIL_INT ResultType, MIL_UINT64 MPTYPE *ResultArrayPtr){McolGetResultSafeTypeExecute(ResultId, AreaLabel, SampleOrLabelIndex, ResultType, ResultArrayPtr, M_TYPE_MIL_INT64);}
#endif

   // ----------------------------------------------------------
   // McolInquire

   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType  (MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, int UserVarPtr)
      {
      if(UserVarPtr != 0)
         SafeTypeError(MT("McolInquire"));

      return McolInquire(ContextId, Index, InquireType, NULL );
      }

   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeTypeExecute (MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, void *UserVarPtr, MIL_INT GivenType)
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
         SafeTypeError(MT("McolInquire"));

      return McolInquire(ContextId, Index, InquireType, UserVarPtr);
      }

   inline MFTYPE32 MIL_INT MFTYPE McolInquireUnsafe  (MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, void       *UserVarPtr) {return McolInquire               (ContextId, Index, InquireType, UserVarPtr                  );}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_INT8   *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_CHAR     );}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_INT16  *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_SHORT    );}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_INT32  *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT32);}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_INT64  *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT64);}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, float      *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_FLOAT    );}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_DOUBLE *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_DOUBLE   );}
#if M_MIL_SAFE_TYPE_SUPPORTS_UNSIGNED
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_UINT8  *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_CHAR     );}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_UINT16 *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_SHORT    );}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_UINT32 *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT32);}
   inline MFTYPE32 MIL_INT MFTYPE McolInquireSafeType(MIL_ID ContextId, MIL_INT Index, MIL_INT InquireType, MIL_UINT64 *UserVarPtr) {return McolInquireSafeTypeExecute(ContextId, Index, InquireType, UserVarPtr, M_TYPE_MIL_INT64);}
#endif

#define McolGetResult        McolGetResultSafeType
#define McolInquire          McolInquireSafeType

#else // #if M_MIL_USE_SAFE_TYPE

#define McolGetResultUnsafe        McolGetResult
#define McolInquireUnsafe          McolInquire

#endif // #if M_MIL_USE_SAFE_TYPE

#endif // !M_MIL_LITE

#endif /* __MILCOLOR_H__ */

