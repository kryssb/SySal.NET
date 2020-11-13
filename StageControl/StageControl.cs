using System;
using SySal;
using System.Runtime.Serialization;
using System.Xml;
using System.Xml.Serialization;
using System.Security;
[assembly: AllowPartiallyTrustedCallers]


namespace SySal.StageControl
{
    /// <summary>
    /// Axis constants
    /// </summary>
    [Serializable]
    public enum Axis { X, Y, Z }

    /// <summary>
    /// Stage status constants
    /// </summary>
    [Serializable]
    [Flags]
    public enum AxisStatus { OK = 0, ReverseLimitActive = 1, ForwardLimitActive = 2, BothLimitsActive = 3, MotorOff = 4, UnknownError = 8 }

    /// <summary>
    /// A sample in a trajectory.
    /// </summary>
    [Serializable]
    public struct TrajectorySample
    {
        /// <summary>
        /// Time in milliseconds.
        /// </summary>
        public double TimeMS;
        /// <summary>
        /// Position.
        /// </summary>
        public SySal.BasicTypes.Vector Position;
    }

    /// <summary>
    /// Stage control interface.
    /// </summary>
    public interface IStageWithoutTimesource : IDisposable
    {
        /// <summary>
        /// Intensity of the illumination.
        /// </summary>
        ushort LightLevel
        {
            get;
            set;
        }
        /// <summary>
        /// Retrieves the position of an axis.
        /// </summary>
        /// <param name="ax">the axis whose position is sought.</param>
        /// <returns>the position of the axis in micron.</returns>
        double GetPos(Axis ax);
        /// <summary>
        /// Retrieves the status of an axis.
        /// </summary>
        /// <param name="ax">the axis whose status is sought.</param>
        /// <returns>the status of the axis.</returns>
        AxisStatus GetStatus(Axis ax);
        /// <summary>
        /// Resets an axis.
        /// </summary>
        /// <param name="ax">the axis to be reset.</param>
        void Reset(Axis ax);
        /// <summary>
        /// Stops an axis.
        /// </summary>
        /// <param name="ax">the axis to be stopped.</param>
        void Stop(Axis ax);
        /// <summary>
        /// Moves an axis to a specified position.
        /// </summary>
        /// <param name="ax">the axis to be moved.</param>
        /// <param name="pos">the position to reach.</param>
        /// <param name="speed">the travelling speed.</param>
        /// <param name="acc">the acceleration to be used.</param>
        /// <param name="dec">the deceleration to be used.</param>
        void PosMove(Axis ax, double pos, double speed, double acc, double dec);
        /// <summary>
        /// Moves an axis in sawtooth mode (between two positions).
        /// </summary>
        /// <param name="ax">the axis to be moved.</param>
        /// <param name="pos1">the first position to reach.</param>
        /// <param name="speed1">the first travelling speed.</param>
        /// <param name="acc1">the first acceleration to be used.</param>
        /// <param name="dec1">the first deceleration to be used.</param>
        /// <param name="checkpos">the checkpoint where motion is inverted to reach position #2.</param>
        /// <param name="pos2">the second position to reach.</param>
        /// <param name="speed2">the second travelling speed.</param>
        /// <param name="acc2">the second acceleration to be used.</param>
        /// <param name="dec2">the second deceleration to be used.</param>
        /// <remarks>The checkpoint should be in a position between the current and the first; if it is not reached, the command will never transition to the second target.</remarks>
        void SawToothPosMove(Axis ax, double pos1, double speed1, double acc1, double dec1, double checkpos, double pos2, double speed2, double acc2, double dec2);
        /// <summary>
        /// Moves an axis at a constant speed.
        /// </summary>
        /// <param name="ax">the axis to be moved.</param>
        /// <param name="speed">the travelling speed.</param>
        /// <param name="acc">the acceleration to use to reach the specified speed.</param>
        void SpeedMove(Axis ax, double speed, double acc);
        /// <summary>
        /// Starts recording a trajectory.
        /// </summary>
        /// <param name="mindeltams">minimum time interval in milliseconds between two samples.</param>
        /// <param name="totaltimems">the total time to be recorded.</param>
        void StartRecording(double mindeltams, double totaltimems);
        /// <summary>
        /// Stops recording a trajectory.
        /// </summary>
        void CancelRecording();
        /// <summary>
        /// The recorded trajectory.
        /// </summary>
        /// <remarks>If the trajectory is not complete, invoking this property returns only when recording is complete.</remarks>
        TrajectorySample[] Trajectory
        {
            get;
        }
        /// <summary>
        /// Retrieves the value of a reference position.
        /// </summary>
        /// <param name="name">the name of the reference position to be retrieved.</param>
        /// <returns>the value of the reference position.</returns>
        /// <remarks>If the reference position is not available or is undefined, an exception is thrown.</remarks>
        double GetNamedReferencePosition(string name);
    }
    /// <summary>
    /// Allows specifying a time source for trajectory sampling.
    /// </summary>
    public interface IStage : IStageWithoutTimesource
    {
        /// <summary>
        /// Sets the source of time sampling for recorded trajectories.
        /// </summary>
        System.Diagnostics.Stopwatch TimeSource
        {
            set;
        }
    }

    /// <summary>
    /// Enables functions with timer.
    /// </summary>
    public interface IStageWithTimer : IStage
    {
        /// <summary>
        /// Specifies that no functions will be called in a while, so internal management is allowed now.
        /// </summary>
        void Idle();
        /// <summary>
        /// Moves an axis to a specified position when the timesource reaches a specified time.
        /// </summary>
        /// <param name="timems">the time when motion has to start</param>
        /// <param name="ax">the axis to be moved.</param>
        /// <param name="pos">the position to reach.</param>
        /// <param name="speed">the travelling speed.</param>
        /// <param name="acc">the acceleration to be used.</param>
        /// <param name="dec">the deceleration to be used.</param>
        void AtTimePosMove(long timems, Axis ax, double pos, double speed, double acc, double dec);
        /// <summary>
        /// When the timesource reaches a specified time, begins a motion profile.
        /// </summary>
        /// <param name="timems">the time when motion has to start</param>
        /// <param name="ax">the axis to be moved.</param>
        /// <param name="ispos">the flags that specify if the move is a position move or a speed move. 
        /// <c>true</c> is used for position moves, <c>false</c> for speed moves.</param>
        /// <param name="pos">the positions to reach in position moves. Ignored for speed moves.</param>
        /// <param name="speed">the travelling speed in each segment. The sign is ignored in position moves. 
        /// In speed moves, the sign also encodes the direction of motion.</param>
        /// <param name="waitms">in position moves, the number of milliseconds to wait after each move; 
        /// in speed moves, the duration of the speed move.</param>
        /// <param name="acc">the acceleration to be used.</param>
        /// <param name="dec">the deceleration to be used.</param>
        /// <remarks>the numbers of flags, positions, speeds and wait times must be equal.</remarks>        
        void AtTimeMoveProfile(long timems, Axis ax, bool [] ispos, double [] pos, double [] speed, long [] waitms, double acc, double dec);
    }

    /// <summary>
    /// Enables direct access to trajectory data without allocating new memory.
    /// </summary>
    public interface IStageWithDirectTrajectoryData
    {
        /// <summary>
        /// Reads a sample in trajectory data.
        /// </summary>
        /// <param name="s">the index (0-based) of the sample to read.</param>
        /// <param name="ts">the variable where sample data are to be written.</param>
        /// <returns><c>true</c> if the trajectory sample is valid, <c>false</c> if the sample is beyond the last one.</returns>
        /// <remarks>If the sample sought is beyond the last, the return value and the result are both undefined.</remarks>
        bool GetTrajectoryData(uint s, ref StageControl.TrajectorySample ts);
    }
}
