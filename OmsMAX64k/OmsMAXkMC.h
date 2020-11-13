/*
MODULE:
  OmsMAXkMC.h - Oregon Micro Systems MAXk motor controller
                Motion Control DLL header file.

PURPOSE:
  Contains declarations for use by C & C++ applications that link to the
  OmsMAXpMC DLL.

REVISIONS:
  30-Sep-2010  Release version 1.00
*/  

#ifndef INC_OmsMAXpMC_H
#define INC_OmsMAXpMC_H

#include <windows.h>

/*  Function error code definitions  */
#define SUCCESS                0
#define COMMAND_TIME_OUT       1
#define RESPONSE_TIME_OUT      2
#define INVALID_AXIS_SELECTION 3
#define MOVE_TIME_OUT          4
#define INVALID_PARAMETER      5
#define INVALID_BIT_NUMBER     6

/*  Axis status definitions  */
#define AXIS_DIRECTION   0x01
#define AXIS_DONE        0x02
#define AXIS_OVERTRAVEL  0x04
#define AXIS_HOME_SWITCH 0x08

/*  Encoder status definitions  */
#define SLIP_DETECT_ENABLED           0x01   
#define POSITION_MAINTENANCE_ENABLED  0x02
#define AXIS_SLIPPED                  0x04
#define AXIS_WITHIN_DEADBAND          0x08
#define ENCODER_AT_HOME               0x10

#define POSITIVE  0
#define NEGATIVE  1
#define MODE_OFF  0
#define MODE_ON   1
#define LOW       0
#define HIGH      1

/* Controller status definitions  */
#define COMMAND_ERROR       0x01

/*  Axis selection constants  */
#define OMS_X_AXIS 0x0001
#define OMS_Y_AXIS 0x0002
#define OMS_Z_AXIS 0x0004
#define OMS_T_AXIS 0x0008
#define OMS_U_AXIS 0x0010
#define OMS_V_AXIS 0x0020
#define OMS_R_AXIS 0x0040
#define OMS_S_AXIS 0x0080
#define OMS_W_AXIS 0x0100
#define OMS_K_AXIS 0x0200
#define OMS_ALL_AXES 0x03FF

/*  Axis over travel limit constants  */
#define OMS_X_AXIS_NEG_LIMIT 0x00001
#define OMS_Y_AXIS_NEG_LIMIT 0x00002
#define OMS_Z_AXIS_NEG_LIMIT 0x00004
#define OMS_T_AXIS_NEG_LIMIT 0x00008
#define OMS_U_AXIS_NEG_LIMIT 0x00010
#define OMS_V_AXIS_NEG_LIMIT 0x00020
#define OMS_R_AXIS_NEG_LIMIT 0x00040
#define OMS_S_AXIS_NEG_LIMIT 0x00080
#define OMS_W_AXIS_NEG_LIMIT 0x00100
#define OMS_K_AXIS_NEG_LIMIT 0x00200
#define OMS_X_AXIS_POS_LIMIT 0x00400
#define OMS_Y_AXIS_POS_LIMIT 0x00800
#define OMS_Z_AXIS_POS_LIMIT 0x01000
#define OMS_T_AXIS_POS_LIMIT 0x02000
#define OMS_U_AXIS_POS_LIMIT 0x04000
#define OMS_V_AXIS_POS_LIMIT 0x08000
#define OMS_R_AXIS_POS_LIMIT 0x10000
#define OMS_S_AXIS_POS_LIMIT 0x20000
#define OMS_W_AXIS_POS_LIMIT 0x40000
#define OMS_K_AXIS_POS_LIMIT 0x80000

/*  I/O bit selection constants  */
#define BIT0    0x0001
#define BIT1    0x0002
#define BIT2    0x0004
#define BIT3    0x0008
#define BIT4    0x0010
#define BIT5    0x0020
#define BIT6    0x0040
#define BIT7    0x0080
#define BIT8    0x0100
#define BIT9    0x0200
#define BIT10   0x0400
#define BIT11   0x0800
#define BIT12   0x1000
#define BIT13   0x2000
#define BIT14   0x4000
#define BIT15   0x8000

#define MIN_VELOCITY 1
#define MAX_VELOCITY 4000000

#define MIN_ACCELERATION 1
#define MAX_ACCELERATION 8000000

typedef struct
{
  long X;
  long Y;
  long Z;
  long T;
  long U;
  long V;
  long R;
  long S;
  long W;
  long K;
}AXES_DATA, *PAXES_DATA;

typedef struct
{
   long Motor;
   long Encoder;
}AXIS_DATA;

typedef struct
{
  AXIS_DATA X;
  AXIS_DATA Y;
  AXIS_DATA Z;
  AXIS_DATA T;
  AXIS_DATA U;
  AXIS_DATA V;
  AXIS_DATA R;
  AXIS_DATA S;
  AXIS_DATA W;
  AXIS_DATA K;
}POSITION_DATA, *PPOSITION_DATA;

typedef enum
{
  OMS_BIG_CMD_BUFFER_MODE = 0,
  OMS_NORMAL_CMD_BUFFER_MODE = 1
} eOMSBuffMode;


#ifndef __cplusplus
#define DLLIMPORT __declspec(dllimport)
#else
#define DLLIMPORT extern "C" __declspec(dllimport)
#endif

/*#############################################################################
  General Functions
#############################################################################*/
DLLIMPORT HANDLE _stdcall GetOmsHandle( LPSTR );
DLLIMPORT void _stdcall CloseOmsHandle( HANDLE);
DLLIMPORT long _stdcall ResetOmsController(HANDLE);
DLLIMPORT long _stdcall SendOmsQueryCommand(HANDLE, LPSTR, LPSTR);
DLLIMPORT long _stdcall SendAndGetString(HANDLE, LPSTR, LPSTR);
DLLIMPORT long _stdcall SendOmsTextCommand(HANDLE, LPSTR);
DLLIMPORT long _stdcall SendString(HANDLE, LPSTR);
DLLIMPORT long _stdcall GetCmdBufferFree(HANDLE);
DLLIMPORT long _stdcall GetOmsControllerDescription(HANDLE, LPSTR);
DLLIMPORT void _stdcall GetOmsDriverVersion(HANDLE, LPSTR);
DLLIMPORT void _stdcall GetOmsDllVersion(LPSTR, long);

/*#############################################################################
 Single Axis Move Functions
#############################################################################*/
DLLIMPORT long _stdcall MoveOmsAxisAbs(HANDLE, long , long);
DLLIMPORT long _stdcall MoveOmsAxisAbsWait(HANDLE, long , long, long );
DLLIMPORT long _stdcall MoveOmsAxisRel(HANDLE, long , long);
DLLIMPORT long _stdcall MoveOmsAxisRelWait(HANDLE, long , long, long );
DLLIMPORT long _stdcall MoveOmsAxisIndefinite(HANDLE, long , long );
DLLIMPORT long _stdcall MoveOmsAxisFractional(HANDLE, long , double );
DLLIMPORT long _stdcall MoveOmsAxisOneStep(HANDLE, long , long );

/*#############################################################################
  Multiple Axes Move Functions
#############################################################################*/

DLLIMPORT long _stdcall MoveOmsIndependentAbs(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsIndependentAbsWait(HANDLE, long , PAXES_DATA, long);
DLLIMPORT long _stdcall MoveOmsIndependentRel(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsIndependentRelWait(HANDLE, long , PAXES_DATA, long);

DLLIMPORT long _stdcall MoveOmsLinearAbs(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsLinearAbsWait(HANDLE, long , PAXES_DATA, long);
DLLIMPORT long _stdcall MoveOmsLinearRel(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsLinearRelWait(HANDLE, long , PAXES_DATA, long);

DLLIMPORT long _stdcall MoveOmsIndependentAbsMt(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsIndependentAbsMtWait(HANDLE, long , PAXES_DATA, long);
DLLIMPORT long _stdcall MoveOmsIndependentRelMt(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsIndependentRelMtWait(HANDLE, long , PAXES_DATA, long);

DLLIMPORT long _stdcall MoveOmsLinearAbsMt(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsLinearAbsMtWait(HANDLE, long , PAXES_DATA, long);
DLLIMPORT long _stdcall MoveOmsLinearRelMt(HANDLE, long , PAXES_DATA);
DLLIMPORT long _stdcall MoveOmsLinearRelMtWait(HANDLE, long , PAXES_DATA, long);

/*#############################################################################
  Stop Motion Functions
#############################################################################*/
DLLIMPORT long _stdcall StopOmsAxis(HANDLE, long );
DLLIMPORT long _stdcall StopAllOmsAxes(HANDLE );
DLLIMPORT long _stdcall KillAllOmsMotion(HANDLE);

/*#############################################################################
  Motor and Encoder Position
#############################################################################*/
DLLIMPORT long _stdcall GetOmsAxisMotorPosition(HANDLE, long , long *);
DLLIMPORT long _stdcall GetOmsAxisEncoderPosition(HANDLE, long , long *);
DLLIMPORT long _stdcall GetOmsAxisAbsoluteEncoderPosition(HANDLE, long , long *);
DLLIMPORT long _stdcall GetSelectedOmsMotorPositions(HANDLE, long, PAXES_DATA);
DLLIMPORT long _stdcall GetSelectedOmsEncoderPositions(HANDLE, long, PAXES_DATA);
DLLIMPORT long _stdcall GetMotorPositions (HANDLE, PAXES_DATA);
DLLIMPORT long _stdcall GetEncoderPositions (HANDLE, PAXES_DATA);
DLLIMPORT long _stdcall GetAbsoluteEncoderPositions (HANDLE, PAXES_DATA);
DLLIMPORT long _stdcall GetPositionData (HANDLE, PPOSITION_DATA);

/*#############################################################################
  Position Initialization
#############################################################################*/
DLLIMPORT long _stdcall DefineOmsHomeAsSwitchClosed(HANDLE, long);
DLLIMPORT long _stdcall DefineOmsHomeAsSwitchOpen(HANDLE, long);
DLLIMPORT long _stdcall FindOmsAxisFwdLimit(HANDLE, long, long, long);
DLLIMPORT long _stdcall FindOmsAxisRevLimit(HANDLE, long, long, long);

DLLIMPORT long _stdcall HomeOmsAxisFwdUseEncoder(HANDLE , long, long);
DLLIMPORT long _stdcall HomeOmsAxisFwdUseEncoderWait(HANDLE, long, long, long);
DLLIMPORT long _stdcall HomeOmsAxisFwdUseSwitch(HANDLE , long, long);
DLLIMPORT long _stdcall HomeOmsAxisFwdUseSwitchWait(HANDLE, long, long, long);

DLLIMPORT long _stdcall HomeOmsAxisRevUseEncoder(HANDLE, long, long);
DLLIMPORT long _stdcall HomeOmsAxisRevUseEncoderWait(HANDLE, long, long, long);
DLLIMPORT long _stdcall HomeOmsAxisRevUseSwitch(HANDLE, long, long);
DLLIMPORT long _stdcall HomeOmsAxisRevUseSwitchWait(HANDLE, long, long, long);

DLLIMPORT long _stdcall SetOmsAxisPosition(HANDLE, long , long);

/*#############################################################################
  Velocity Control/Reporting
#############################################################################*/
DLLIMPORT long _stdcall GetOmsAxisVelocity(HANDLE, long , long *);
DLLIMPORT long _stdcall GetSelectedOmsVelocities(HANDLE, long, PAXES_DATA);
DLLIMPORT long _stdcall SetOmsAxisBaseVelocity(HANDLE, long , long);
DLLIMPORT long _stdcall SetOmsAxisVelocity(HANDLE, long , long);

/*#############################################################################
  Acceleration Control/Reporting
#############################################################################*/
DLLIMPORT long _stdcall SetOmsAxisAcceleration(HANDLE, long , long);
DLLIMPORT long _stdcall GetOmsAxisAcceleration(HANDLE, long , long *);
DLLIMPORT long _stdcall GetSelectedOmsAccelerations(HANDLE, long, PAXES_DATA);
DLLIMPORT long _stdcall SelectOmsCosineRamp(HANDLE, long);
DLLIMPORT long _stdcall SelectOmsLinearRamp(HANDLE, long);
DLLIMPORT long _stdcall SelectOmsParabolicRamp(HANDLE, long, long);
DLLIMPORT long _stdcall SelectOmsSCurveRamp(HANDLE, long, long);

/*#############################################################################
  Axis/Axes Event Flags
#############################################################################*/
DLLIMPORT long _stdcall GetOmsAxisDoneFlag(HANDLE, long , BOOL*);
DLLIMPORT long _stdcall GetDoneFlags(HANDLE, long *);
DLLIMPORT long _stdcall GetAllOmsDoneFlags(HANDLE, long *);
DLLIMPORT long _stdcall ClrDoneFlags(HANDLE, long);
DLLIMPORT long _stdcall ClrOmsDoneFlags(HANDLE, long);

DLLIMPORT long _stdcall GetLimitFlags(HANDLE, long *);
DLLIMPORT long _stdcall GetAllOmsLimitFlags(HANDLE, long *);
DLLIMPORT long _stdcall GetOmsAxisLimitFlag(HANDLE, long , BOOL*);
DLLIMPORT long _stdcall ClrLimitFlags(HANDLE, long);
DLLIMPORT long _stdcall ClrOmsLimitFlags(HANDLE, long);

DLLIMPORT long _stdcall GetStatusFlags(HANDLE, long *);
DLLIMPORT long _stdcall GetOmsControllerFlags(HANDLE, long *);
DLLIMPORT long _stdcall ClrStatusFlags(HANDLE, long);
DLLIMPORT long _stdcall ClrOmsControllerFlags(HANDLE, long);
DLLIMPORT long _stdcall GetOmsAxisFlags(HANDLE, long , long*);

DLLIMPORT long _stdcall GetOmsLimitSensors(HANDLE, long *);
DLLIMPORT long _stdcall GetOmsHomeSensors(HANDLE, long *);

/*#############################################################################
  Encoder Feedback Specific Functions
#############################################################################*/
DLLIMPORT long _stdcall EnableOmsSlipDetection(HANDLE, long);
DLLIMPORT long _stdcall GetOmsAxisSlipFlag(HANDLE, long , BOOL*);
DLLIMPORT long _stdcall GetSlipFlags(HANDLE, long *);
DLLIMPORT long _stdcall GetAllOmsSlipFlags(HANDLE, long *);
DLLIMPORT long _stdcall ClrSlipFlags(HANDLE, long);
DLLIMPORT long _stdcall ClrOmsSlipFlags(HANDLE, long);


DLLIMPORT long _stdcall GetOmsEncoderFlags(HANDLE, long , long*);
DLLIMPORT long _stdcall SetOmsAxisPidEnable(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsEncoderHoldMode(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsEncoderRatio(HANDLE, long, long, long);
DLLIMPORT long _stdcall SetOmsHoldDeadBand(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsEncoderSlipTolerance(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsHoldGain(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsHoldVelocity(HANDLE, long, long);

/*#############################################################################
  Overtravel
#############################################################################*/
DLLIMPORT long _stdcall SetOmsAxisOvertravelDetect(HANDLE, long, long);
DLLIMPORT long _stdcall ConfigureOmsAxisLimitInput(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsSoftLimitsMode(HANDLE, long, long);

/*#############################################################################
  General Purpose I/O Bit Functions
#############################################################################*/
DLLIMPORT long _stdcall GetOmsIOBitConfig(HANDLE, long *);
DLLIMPORT long _stdcall ConfigureAllOmsIOBits(HANDLE, long);
DLLIMPORT long _stdcall GetAllOmsIOBits(HANDLE, long *);
DLLIMPORT long _stdcall SetAllOmsIOBits(HANDLE, long);
DLLIMPORT long _stdcall GetOmsIOBit(HANDLE, long, long *);
DLLIMPORT long _stdcall SetOmsIOBit(HANDLE, long, long);

/*#############################################################################
  Axis Auxiliary Output Bit Functions
#############################################################################*/
DLLIMPORT long _stdcall EnableOmsAxisAuxOutAutoMode(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsAxisAuxOutSettleTime(HANDLE, long, long);
DLLIMPORT long _stdcall SetOmsAxisAuxOutBit(HANDLE, long, long);
DLLIMPORT long _stdcall SetSelectedOmsAuxOutBits(HANDLE, long, long);

/*#############################################################################
  Wait Function
#############################################################################*/
DLLIMPORT long _stdcall OmsWait(long);

/*#############################################################################
  Big Command Buffer
#############################################################################*/
DLLIMPORT long _stdcall OMS_SetCommandBuffer(HANDLE hDevice, eOMSBuffMode eBuffSelection, long lBuffSize);
DLLIMPORT long _stdcall OMS_GetCommandBufferSelection(HANDLE hDevice, eOMSBuffMode* pBuffSelection);
DLLIMPORT long _stdcall OMS_BigBufferSendBlock(HANDLE hDevice, LPSTR pCmd, long lBlockSize, long *lNumSent);
DLLIMPORT long _stdcall OMS_GetBigBufferFreeSpace(HANDLE hDevice, long* pFree);
DLLIMPORT long _stdcall OMS_FlushBigBuffer(HANDLE hDevice);

#endif  /* INC_OmsMAXpMC_H  */
  
