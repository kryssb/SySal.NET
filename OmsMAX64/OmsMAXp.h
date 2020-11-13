//
//  MODULE:   OmsMAXp.h - Oregon Micro Systems PCI motor controller
//                       minimal function driver interface DLL header
//                       file.
//
//  PURPOSE:
//    Contains declarations for use by C & C++ applications that link to the
//    OmsMAXp DLL.
//
//  Revisions:
//  01-Oct-2003  Release version 1.00
//  09-Mar-2004  Added GetOmsLimitSensors & GetOmsHomeSensors functions.

#ifndef INC_OMSDLL_H
#define INC_OMSDLL_H

#include <windows.h>

//Axis selection constants
#define OMS_X_AXIS 0x01
#define OMS_Y_AXIS 0x02
#define OMS_Z_AXIS 0x04
#define OMS_T_AXIS 0x08
#define OMS_U_AXIS 0x10
#define OMS_V_AXIS 0x20
#define OMS_R_AXIS 0x40
#define OMS_S_AXIS 0x80

//Controller status flag defintions
#define  COMMAND_ERROR    0x01

//DLL status flag definitions
#define SUCCESS 0
#define TIME_OUT 3

/*Vector mode command definitions*/
#define MOVE_ABSOLUTE_CV     0x01
#define MOVE_RELATIVE_CV     0x02
#define MOVE_ABSOLUTE_RAMP   0x03
#define MOVE_RELATIVE_RAMP   0x04
#define VELOCITY_MODE_CV     0x05
#define VELOCITY_MODE_RAMP   0x06
#define END_VECTOR_MODE      0x10

#define INHIBIT_BIT_OUTPUT  0x00
#define ENABLE_BIT_OUTPUT   0x01

#define VECTOR_BUF_SIZE     10240    

//Vector mode function return codes:
#define INVALID_VECTOR_COMMAND         1
#define INVALID_AXIS_SELECTION         2
#define INVALID_VELOCITY_CONTROL_COUNT 3
#define INVALID_BIT_STATE_MASK         4
#define INVALID_VECTOR_QUEUE_SELECT    5
#define ALREADY_IN_VECTOR_MODE         6
#define QUEUE_FULL                     7
#define BUFFER_FULL                    8

typedef struct
{
   long AxisId;
   long MotorPos;
   long EncoderPos;
   long DacValue;
}PROFILE_SAMPLE, *PPROFILE_SAMPLE;

typedef struct
{
   long InsertIdx;
   long BufferFull;
   long SequenceNum;
   PROFILE_SAMPLE Buffer[1024];
}PROFILE_BUFFER, *PPROFILE_BUFFER;   

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
}MOTOR_POSITIONS, *PMOTOR_POSITIONS;

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
}ENCODER_POSITIONS, *PENCODER_POSITIONS;


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
}POSITION_DATA, *PPOSITION_DATA;

/*Vector definition structure.*/
typedef struct
{
  unsigned long AxisSelections;
  unsigned long VelocityCount;
  unsigned long OutputOption;
  unsigned long BitStates;
  unsigned long Command;
  long X;
  long Y;
  long Z;
  long T;
  long U;
  long V;
  long R;
  long S;
}VECTOR_DEFINITION, *PVECTOR_DEFINITION;

/*Vector command structure.*/
typedef struct
{
  long VectorCommand;
  long OutputCommand;
  long X;
  long Y;
  long Z;
  long T;
  long U;
  long V;
  long R;
  long S;
}VECTOR_TYPE, *PVECTOR_TYPE;

/*Vector transmission structure.*/
typedef struct
{
  long QueueSelection;
  long NextWord;
  long Word[VECTOR_BUF_SIZE];
}VECTOR_BUFFER, *PVECTOR_BUFFER;


#ifndef __cplusplus
#define DLLIMPORT __declspec(dllimport)
#else
#define DLLIMPORT extern "C" __declspec(dllimport)
#endif

DLLIMPORT void _stdcall   CloseOmsHandle (HANDLE);
DLLIMPORT void _stdcall   ClrDoneFlags (HANDLE, BYTE);
DLLIMPORT void _stdcall   ClrLimitFlags (HANDLE, BYTE);
DLLIMPORT void _stdcall   ClrSlipFlags (HANDLE, BYTE);
DLLIMPORT void _stdcall   ClrStatusFlags (HANDLE, BYTE);
DLLIMPORT void _stdcall   GetBufferStatus (HANDLE, long, long*);
DLLIMPORT long _stdcall   GetCmdBufferFree(HANDLE);
DLLIMPORT void _stdcall   GetDoneFlags (HANDLE, LPBYTE);
DLLIMPORT long _stdcall   GetEncoderPositions (HANDLE, PENCODER_POSITIONS);
DLLIMPORT void _stdcall   GetLimitFlags (HANDLE, LPBYTE);
DLLIMPORT long _stdcall   GetMotorPositions (HANDLE, PMOTOR_POSITIONS);
DLLIMPORT void _stdcall   GetOmsDriverVersion (HANDLE, LPSTR);
DLLIMPORT HANDLE _stdcall GetOmsHandle (LPSTR);
DLLIMPORT long _stdcall   GetPositionData (HANDLE, PPOSITION_DATA);
DLLIMPORT void _stdcall   GetProfileBuffer (HANDLE, long, PPROFILE_BUFFER);
DLLIMPORT void _stdcall   GetSlipFlags (HANDLE, LPBYTE);
DLLIMPORT void _stdcall   GetStatusFlags (HANDLE, LPBYTE);

DLLIMPORT void _stdcall GetOmsLimitSensors(HANDLE, long *);
DLLIMPORT void _stdcall GetOmsHomeSensors(HANDLE, long *);

DLLIMPORT long _stdcall   KillAllOmsMotion(HANDLE);
DLLIMPORT void _stdcall   OmsWait(long);
DLLIMPORT long _stdcall   ResetController(HANDLE);
DLLIMPORT void _stdcall   SendAndGetString (HANDLE, LPSTR, LPSTR);
DLLIMPORT long _stdcall   SendString (HANDLE, LPSTR);

//Vector mode functions:
DLLIMPORT long _stdcall ConstructVector(PVECTOR_DEFINITION, PVECTOR_TYPE);
DLLIMPORT long _stdcall EnterVectorMode(HANDLE, long);
DLLIMPORT long _stdcall ExitVectorMode(HANDLE, long);
DLLIMPORT long _stdcall GetFreeVectorSpace(HANDLE, long);
DLLIMPORT long _stdcall GetFreeVectorBuffWords(PVECTOR_BUFFER);
DLLIMPORT long _stdcall PackVector(PVECTOR_TYPE, PVECTOR_BUFFER);
DLLIMPORT long _stdcall SendVectors(HANDLE, long, PVECTOR_BUFFER);
DLLIMPORT long _stdcall VectorMode(HANDLE);
DLLIMPORT void _stdcall ZeroVectorBuffer(PVECTOR_BUFFER);

#endif  // INC_OMSDLL_H   

