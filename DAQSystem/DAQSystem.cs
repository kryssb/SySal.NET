using System;
using SySal;
using SySal.BasicTypes;
using System.Runtime.Serialization;
using System.Xml;
using System.Xml.Serialization;
using System.Security;
[assembly:AllowPartiallyTrustedCallers]

namespace SySal.DAQSystem
{
	/// <summary>
	/// The ports for Opera services.
	/// </summary>
	public enum OperaPort 
	{ 
		/// <summary>
		/// Scanning server port number.
		/// </summary>
		ScanServer = 1780, 
		/// <summary>
		/// Batch server port number.
		/// </summary>
		BatchServer = 1781, 
		/// <summary>
		/// Data processing server port number.
		/// </summary>
		DataProcessingServer = 1782
	};


	/// <summary>
	/// Types of reference frames.
	/// </summary>
	[Serializable]
	public enum Frame 		
	{ 
		/// <summary>
		/// Cartesian reference frame (X, Y).
		/// </summary>
		Cartesian, 
		/// <summary>
		/// Polar reference frame (Azimuth, Radius; also denoted as Transverse, Longitudinal).
		/// </summary>
		Polar 
	}

    namespace Scanning
    {
        /// <summary>
        /// A zone to be scanned
        /// </summary>
        [Serializable]
        public class ZoneDesc
        {
            /// <summary>
            /// Free tag.
            /// </summary>
            public long Series;
            /// <summary>
            /// Minimum X extent of the zone.
            /// </summary>
            public double MinX;
            /// <summary>
            /// Maximum X extent of the zone.
            /// </summary>
            public double MaxX;
            /// <summary>
            /// Minimum Y extent of the zone.
            /// </summary>
            public double MinY;
            /// <summary>
            /// Maximum Y extent of the zone.
            /// </summary>
            public double MaxY;
            /// <summary>
            /// Output path for the raw data files.
            /// </summary>
            public string Outname;
            /// <summary>
            /// If microtracks must to be acquired only in a slope window, this member is set to <c>true</c>; <c>false</c> otherwise.
            /// </summary>
            public bool UsePresetSlope;
            /// <summary>
            /// Preselected slope (ignored if <c>UsePresetSlope</c> is <c>false</c>).
            /// </summary>
            public SySal.BasicTypes.Vector2 PresetSlope;
            /// <summary>
            /// X and Y acceptance bands for preset slope (ignored if <c>UsePresetSlope</c> is <c>false</c>). 
            /// </summary>
            public SySal.BasicTypes.Vector2 PresetSlopeAcc;
        }

        /// <summary>
        /// Plate to be mounted on a microscope
        /// </summary>
        [Serializable]
        public class MountPlateDesc
        {
            /// <summary>
            /// Brick identifier.
            /// </summary>
            public long BrickId;
            /// <summary>
            /// Plate identifier.
            /// </summary>
            public long PlateId;
            /// <summary>
            /// Text description of the plate.
            /// </summary>
            public string TextDesc;
            /// <summary>
            /// Initialization string for the map. Can be a path to a map file or inline ASCII map string.
            /// </summary>
            public string MapInitString;
        }

        /// <summary>
        /// Intercalibration information.
        /// </summary>
        [Serializable]
        public struct IntercalibrationInfo
        {
            /// <summary>
            /// XX component of the affine transformation matrix.
            /// </summary>
            public double MXX;
            /// <summary>
            /// XY component of the affine transformation matrix.
            /// </summary>
            public double MXY;
            /// <summary>
            /// YX component of the affine transformation matrix.
            /// </summary>
            public double MYX;
            /// <summary>
            /// YY component of the affine transformation matrix.
            /// </summary>
            public double MYY;
            /// <summary>
            /// X component of the translation.
            /// </summary>
            public double TX;
            /// <summary>
            /// Y component of the translation.
            /// </summary>
            public double TY;
            /// <summary>
            /// Z component of the translation.
            /// </summary>
            public double TZ;
            /// <summary>
            /// X coordinate of the reference center.
            /// </summary>
            public double RX;
            /// <summary>
            /// Y coordinate of the reference center.
            /// </summary>
            public double RY;
            /// <summary>
            /// Transforms a point according to the affine transformation.
            /// </summary>
            /// <param name="inV">the input 2D point.</param>
            /// <returns>the transformed point.</returns>
            public SySal.BasicTypes.Vector2 Transform(SySal.BasicTypes.Vector2 inV)
            {
                SySal.BasicTypes.Vector2 v = new SySal.BasicTypes.Vector2();
                v.X = MXX * (inV.X - RX) + MXY * (inV.Y - RY) + TX + RX;
                v.Y = MYX * (inV.X - RX) + MYY * (inV.Y - RY) + TY + RY;
                return v;
            }
            /// <summary>
            /// Transforms a point according to the affine transformation.
            /// </summary>
            /// <param name="inV">the input 3D point.</param>
            /// <returns>the transformed point.</returns>
            public SySal.BasicTypes.Vector Transform(SySal.BasicTypes.Vector inV)
            {
                SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
                v.X = MXX * (inV.X - RX) + MXY * (inV.Y - RY) + TX + RX;
                v.Y = MYX * (inV.X - RX) + MYY * (inV.Y - RY) + TY + RY;
                v.Z = inV.Z + TZ;
                return v;
            }
            /// <summary>
            /// Deforms a vector using the linear deformation.
            /// </summary>
            /// <param name="inV">the input vector.</param>
            /// <returns>the transformed vector.</returns>
            public SySal.BasicTypes.Vector2 Deform(SySal.BasicTypes.Vector2 inV)
            {
                SySal.BasicTypes.Vector2 v = new SySal.BasicTypes.Vector2();
                v.X = MXX * inV.X + MXY * inV.Y;
                v.Y = MYX * inV.X + MYY * inV.Y;
                return v;
            }
            /// <summary>
            /// Deforms a vector using the linear deformation.
            /// </summary>
            /// <param name="inV">the input vector.</param>
            /// <returns>the transformed vector.</returns>
            public SySal.BasicTypes.Vector Deform(SySal.BasicTypes.Vector inV)
            {
                SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
                v.X = MXX * inV.X + MXY * inV.Y;
                v.Y = MYX * inV.X + MYY * inV.Y;
                v.Z = inV.Z;
                return v;
            }
        }

        /// <summary>
        /// This class hosts types and methods for manual check of tracks by a human operator.
        /// </summary>
        public class ManualCheck
        {
            /// <summary>
            /// The input for a manual check of a base track by a human operator.
            /// </summary>
            [Serializable]
            public struct InputBaseTrack
            {
                /// <summary>
                /// The Id of the track to be searched.
                /// </summary>
                public long Id;
                /// <summary>
                /// The position where the track is expected to be.
                /// </summary>
                public SySal.BasicTypes.Vector2 Position;
                /// <summary>
                /// The expected slope of the track.
                /// </summary>
                public SySal.BasicTypes.Vector2 Slope;
                /// <summary>
                /// Position tolerance.
                /// </summary>
                public double PositionTolerance;
                /// <summary>
                /// Slope tolerance.
                /// </summary>
                public double SlopeTolerance;
            }

            /// <summary>
            /// The result of a manual check of a base track by a human operator.
            /// </summary>
            [Serializable]
            public struct OutputBaseTrack
            {
                /// <summary>
                /// The Id of the track checked.
                /// </summary>
                public long Id;
                /// <summary>
                /// <c>true</c> if the check was successfully performed, <c>false</c> if an error occurred.
                /// </summary>
                public bool CheckDone;
                /// <summary>
                /// <c>true</c> if the track has been found, <c>false</c> otherwise. Ignored if <c>CheckDone</c> is <c>false</c>.
                /// </summary>
                public bool Found;
                /// <summary>
                /// Number of grains of the base track. Meaningful only if <c>CheckDone</c> and <c>Found</c> are both true.
                /// </summary>
                public int Grains;
                /// <summary>
                /// Position where the base track was found. Meaningful only if <c>CheckDone</c> and <c>Found</c> are both true.
                /// </summary>
                public SySal.BasicTypes.Vector2 Position;
                /// <summary>
                /// Slope of trhe base track found. Meaningful only if <c>CheckDone</c> and <c>Found</c> are both true.
                /// </summary>
                public SySal.BasicTypes.Vector2 Slope;
            }

        }

        /// <summary>
        /// Request to dump the sequence of images in a field of view.
        /// </summary>
        [Serializable]
        public struct ImageDumpRequest
        {
            /// <summary>
            /// Identifier of the dataset.
            /// </summary>
            public Identifier Id;
            /// <summary>
            /// Path where the output file should be stored.
            /// </summary>
            public string OutputPath;
            /// <summary>
            /// Center of the field of view.
            /// </summary>
            public SySal.BasicTypes.Vector2 Position;
            /// <summary>
            /// Slope of the track to be checked (if applicable).
            /// </summary>
            public SySal.BasicTypes.Vector2 Slope;
        }

        /// <summary>
        /// This class hosts types and methods for plate quality definition and monitoring.
        /// </summary>        
        public class PlateQuality
        {
            /// <summary>
            /// Set of quality data with fog and thickness.
            /// </summary>
            [Serializable]
            public struct FogThicknessSet
            {
                /// <summary>
                /// Average number of fog grains on top side in a volume of 1000 micron cubed.
                /// </summary>
                public double TopFogGrains_1000MicronCubed;
                /// <summary>
                /// Average number of fog grains on bottom side in a volume of 1000 micron cubed.
                /// </summary>
                public double BottomFogGrains_1000MicronCubed;
                /// <summary>
                /// Thickness of the top layer of emulsion in micron.
                /// </summary>
                public double TopThickness;
                /// <summary>
                /// Thickness of the base in micron.
                /// </summary>
                public double BaseThickness;
                /// <summary>
                /// Thickness of the bottom layer of emulsion in micron.
                /// </summary>
                public double BottomThickness;
            }
        }
    }

	namespace Drivers
	{
		/// <summary>
		/// Driver types.
		/// </summary>
		public enum DriverType 
		{ 
			/// <summary>
			/// The program is a computing module or a scanning setup.
			/// </summary>
			Lowest = 0,
			/// <summary>
			/// The program is a scanning driver.
			/// </summary>
			Scanning = 1, 
			/// <summary>
			/// The program is a volume driver.
			/// </summary>
			Volume = 2, 
			/// <summary>
			/// The program is a brick driver.
			/// </summary>
			Brick = 3,
			/// <summary>
			/// The program is a system driver.
			/// </summary>
			System = 4
		};

		/// <summary>
		/// The description of a driver.
		/// </summary>
		[Serializable]
		public class DriverInfo : ICloneable
		{
			/// <summary>
			/// The name of the driver process.
			/// </summary>
			public string Name = "";
			/// <summary>
			/// Description of the driver.
			/// </summary>
			public string Description = ""; 
			/// <summary>
			/// Type of the driver.
			/// </summary>
			public DriverType DriverType;
			/// <summary>
			/// Clones the DriverInfo.
			/// </summary>
			/// <returns>the object clone.</returns>
			public object Clone()
			{
				DriverInfo d = new DriverInfo();
				d.Name = (string)Name.Clone();
				d.Description = (string)Description.Clone();
				d.DriverType = DriverType;
				return d;
			}
		}

		/// <summary>
		/// Information needed by driver programs to perform their work.
		/// </summary>
		[Serializable]
		public class TaskStartupInfo : ICloneable
		{
			/// <summary>
			/// Opera Computing Infrastructure username associated to the process token.
			/// </summary>
			public string OPERAUsername;
			/// <summary>
			/// Opera Computing Infrastructure password.
			/// </summary>
			public string OPERAPassword;
			/// <summary>
			/// Opera DB User that the driver should impersonate.
			/// </summary>
			public string DBUserName = "";
			/// <summary>
			/// Opera DB password of the impersonated user.
			/// </summary>
			public string DBPassword = "";
			/// <summary>
			/// List of the possible DB servers to use.
			/// </summary>
			public string DBServers = "";
			/// <summary>
			/// Scratch directory for the driver.
			/// </summary>				
			public string ScratchDir = "";
			/// <summary>
			/// Repository for computing executables to be batch-launched if needed.
			/// </summary>
			public string ExeRepository = "";
			/// <summary>
			/// Full pathname of the file that states the progress of the task.
			/// </summary>
			public string ProgressFile = "";
			/// <summary>
			/// If true, the task is restarted where it was interrupted by using information from the progress file.
			/// </summary>
			public bool RecoverFromProgressFile = false;
			/// <summary>
			/// The process operation id of this task.
			/// </summary>
			public long ProcessOperationId = 0;
			/// <summary>
			/// Opera Computing Infrastructure Program Settings that the driver should use.
			/// </summary>
			public long ProgramSettingsId = 0;
			/// <summary>
			/// Id of the scanning machine associated with this task or -1 if none.
			/// </summary>
			public long MachineId = -1;
			/// <summary>
			/// Notes for the process operation.
			/// </summary>
			public string Notes = "";
			/// <summary>
			/// Full path (without the .rwc extension) where the raw data are to be stored.
			/// </summary>
			public string RawDataPath = "";
			/// <summary>
			/// Full path (without the .tlg extension) where the linked zone output is to be stored.
			/// </summary>
			public string LinkedZonePath = "";
			#region ICloneable Members
			/// <summary>
			/// Clones the object.
			/// </summary>
			/// <returns>the cloned object.</returns>
			public virtual object Clone()
			{
				TaskStartupInfo t = new TaskStartupInfo();
				t.DBPassword = this.DBPassword;
				t.DBServers = this.DBServers;
				t.DBUserName = this.DBUserName;
				t.ExeRepository = this.ExeRepository;
				t.LinkedZonePath = this.LinkedZonePath;
				t.MachineId = this.MachineId;				
				t.OPERAPassword = this.OPERAPassword;
				t.OPERAUsername = this.OPERAUsername;
				t.ProcessOperationId = this.ProcessOperationId;
				t.ProgramSettingsId = this.ProgramSettingsId;
				t.ProgressFile = this.ProgressFile;
				t.RawDataPath = this.RawDataPath;
				t.RecoverFromProgressFile = this.RecoverFromProgressFile;
				t.ScratchDir = this.ScratchDir;
				return t;
			}

			#endregion
		}

		/// <summary>
		/// Progress information for a task.
		/// </summary>
		[Serializable]
		public class TaskProgressInfo
		{
			/// <summary>
			/// Progress (fraction from 0 to 1) of the task.
			/// </summary>
			public double Progress = 0.0;
			/// <summary>
			/// Start time of the task.
			/// </summary>
			public System.DateTime StartTime;
			/// <summary>
			/// Finish time (expected if the task is not complete yet).
			/// </summary>
			public System.DateTime FinishTime;
			/// <summary>
			/// Additional information (depends on the specific driver).
			/// </summary>
			public string CustomInfo;
			/// <summary>
			/// True if the task is complete (with or without errors).
			/// </summary>
			public bool Complete;
			/// <summary>
			/// When Complete is true, if this is null the process completed successfully; if it is not null, the exception that terminated the process is saved here.
			/// If Complete is false, this is the exception that put the process in a paused state.
			/// </summary>
			public string ExitException;
			/// <summary>
			/// The Id of the last processed interrupt.
			/// </summary>
			public long LastProcessedInterruptId;
		}

		/// <summary>
		/// Prediction to drive the scanning.
		/// Extends ZoneDesc to incorporate the notion of a predicted track.
		/// </summary>
		[Serializable]	
		public class Prediction : SySal.DAQSystem.Scanning.ZoneDesc
		{
			/// <summary>
			/// Frame type of tolerances fro positions and slopes.
			/// </summary>
			public Frame ToleranceFrame;
			/// <summary>
			/// Predicted X coordinate.
			/// </summary>
			public double PredictedPosX;
			/// <summary>
			/// Predicted Y coordinate.
			/// </summary>
			public double PredictedPosY;
			/// <summary>
			/// First position tolerance to accept the candidate. Depending on the frame type, can be the Transverse (Azimuthal) or X coordinate.
			/// </summary>
			public double PositionTolerance1;
			/// <summary>
			/// Second position tolerance to accept the candidate. Depending on the frame type, can be the Longitudinal (Radial) or Y coordinate.
			/// </summary>
			public double PositionTolerance2;
			/// <summary>
			/// Slope X coordinate.
			/// </summary>
			public double PredictedSlopeX;
			/// <summary>
			/// Slope Y coordinate.
			/// </summary>
			public double PredictedSlopeY;
			/// <summary>
			/// First slope tolerance to accept the candidate. Depending on the frame type, can be the Transverse (Azimuthal) or X coordinate.
			/// </summary>
			public double SlopeTolerance1;
			/// <summary>
			/// Second slope tolerance to accept the candidate. Depending on the frame type, can be the Longitudinal (Radial) or Y coordinate.
			/// </summary>
			public double SlopeTolerance2;
			/// <summary>
			/// Minimum number of grains to accept the candidate.
			/// </summary>
			public uint MinGrains;
			/// <summary>
			/// Maximum sigma to accept the candidate.
			/// </summary>
			public double MaxSigma;
			/// <summary>
			/// Maximum scanning trials before giving up when no candidate is found.
			/// </summary>
			public uint MaxTrials;
			/// <summary>
			/// Index of the candidate track found.
			/// -1 if not found, >= 0 otherwise.
			/// </summary>
			public int CandidateIndex;
			/// <summary>
			/// Global parameters of the candidate track.
			/// </summary>
			public SySal.Tracking.MIPEmulsionTrackInfo CandidateInfo;
		}

        /// <summary>
        /// The shape and type of mark.
        /// </summary>
        [Flags]
        public enum MarkType 
        { 
            /// <summary>
            /// No mark.
            /// </summary>
            None = 0,
            /// <summary>
            /// Spot mark obtained by optical grid printing.
            /// </summary>
            SpotOptical = 1,
            /// <summary>
            /// Spot mark obtained by X-ray gun.
            /// </summary>
            SpotXRay = 2,
            /// <summary>
            /// Lateral X-ray line.
            /// </summary>
            LineXRay = 4
        }

        /// <summary>
        /// Contains definition of constant string for <c>MarkType</c> to <c>char</c> conversions.
        /// </summary>
        public class MarkChar
        {
            /// <summary>
            /// No mark.
            /// </summary>
            public const char None = ' ';
            /// <summary>
            /// Spot mark obtained by optical grid printing.
            /// </summary>
            public const char SpotOptical = 'S';
            /// <summary>
            /// Spot mark obtained by X-ray gun.
            /// </summary>
            public const char SpotXRay = 'X';
            /// <summary>
            /// Lateral X-ray line.
            /// </summary>
            public const char LineXRay = 'L';
        }

		/// <summary>
		/// Startup information for a scanning driver.
		/// </summary>
		[Serializable]
		[XmlInclude(typeof(Prediction))]
		public class ScanningStartupInfo : TaskStartupInfo, ICloneable
		{
			/// <summary>
			/// Plate to be scanned.
			/// </summary>
			public SySal.DAQSystem.Scanning.MountPlateDesc Plate;
			/// <summary>
			/// Zones to be scanned.
			/// </summary>
			public SySal.DAQSystem.Scanning.ZoneDesc [] Zones = new SySal.DAQSystem.Scanning.ZoneDesc[0];
			/// <summary>
			/// Id of the calibration to be used (zero or negative means NULL).
			/// </summary>
			public long CalibrationId;
            /// <summary>
            /// Type of marks to be used. This is only relevant if <c>CalibrationId</c> is negative or zero (NULL calibration). 
            /// When <c>CalibrationId</c> is positive, the type of mark is obtained implicitly from the calibration to be used.
            /// </summary>
            public MarkType MarkSet;

			#region ICloneable Members
			/// <summary>
			/// Clones the object.
			/// </summary>
			/// <returns>the cloned object.</returns>
			public override object Clone()
			{
				// TODO:  Add ScanningStartupInfo.Clone implementation
				ScanningStartupInfo t = new ScanningStartupInfo();
				t.DBPassword = this.DBPassword;
				t.DBServers = this.DBServers;
				t.DBUserName = this.DBUserName;
				t.ExeRepository = this.ExeRepository;
				t.LinkedZonePath = this.LinkedZonePath;
				t.MachineId = this.MachineId;		
				t.OPERAPassword = this.OPERAPassword;
				t.OPERAUsername = this.OPERAUsername;
				t.ProcessOperationId = this.ProcessOperationId;
				t.ProgramSettingsId = this.ProgramSettingsId;
				t.ProgressFile = this.ProgressFile;
				t.RawDataPath = this.RawDataPath;
				t.RecoverFromProgressFile = this.RecoverFromProgressFile;
				t.ScratchDir = this.ScratchDir;
				t.Plate = this.Plate;
				t.CalibrationId = CalibrationId;
                t.MarkSet = MarkSet;
				t.Zones = (Zones == null) ? null : (SySal.DAQSystem.Scanning.ZoneDesc [])(Zones.Clone());
				return t;
			}

			#endregion
		}

		/// <summary>
		/// Descriptor of a box in a brick.
		/// Scanback can also be initiated by this descriptor by setting TopPlate = BottomPlate.
		/// </summary>
		[Serializable]
		public class BoxDesc
		{
			/// <summary>
			/// Free tag.
			/// </summary>
			public long Series;
			/// <summary>
			/// Top plate (included) of the box.
			/// </summary>
			public int TopPlate;
			/// <summary>
			/// Bottom plate (included) of the box.
			/// </summary>
			public int BottomPlate;
			/// <summary>
			/// 2D extents of the Box intersection with the bottom plate.
			/// </summary>
			public Rectangle ExtentsOnBottom;
			/// <summary>
			/// "Slope" of the box w.r.t. the vertical axis.
			/// </summary>
			public Vector2 Slope;
			/// <summary>
			/// Center of the Box in the bottom plate.
			/// </summary>
			public Vector2 CenterOnBottom;
		}

		/// <summary>
		/// Startup information for a volume operation driver.
		/// </summary>
		[Serializable]
		public class VolumeOperationInfo : TaskStartupInfo, ICloneable
		{
			/// <summary>
			/// Boxes to be scanned in the volume.
			/// These can be scanback predictions as well.
			/// </summary>
			public BoxDesc [] Boxes;
			/// <summary>
			/// Id of the brick to be processed.
			/// </summary>
			public long BrickId = 0;
			#region ICloneable Members
			/// <summary>
			/// Clones the object.
			/// </summary>
			/// <returns>the object clone.</returns>
			public override object Clone()
			{
				VolumeOperationInfo t = new VolumeOperationInfo();
				t.DBPassword = this.DBPassword;
				t.DBServers = this.DBServers;
				t.DBUserName = this.DBUserName;
				t.ExeRepository = this.ExeRepository;
				t.LinkedZonePath = this.LinkedZonePath;
				t.MachineId = this.MachineId;				
				t.OPERAPassword = this.OPERAPassword;
				t.OPERAUsername = this.OPERAUsername;
				t.ProcessOperationId = this.ProcessOperationId;
				t.ProgramSettingsId = this.ProgramSettingsId;
				t.ProgressFile = this.ProgressFile;
				t.RawDataPath = this.RawDataPath;
				t.RecoverFromProgressFile = this.RecoverFromProgressFile;
				t.ScratchDir = this.ScratchDir;
				t.BrickId = this.BrickId;
				t.Boxes = (Boxes == null) ? null : ((BoxDesc [])(Boxes.Clone()));
				return t;
			}

			#endregion
		}

		/// <summary>
		/// Startup information for a brick operation driver.
		/// </summary>
		[Serializable]
		public class BrickOperationInfo : TaskStartupInfo, ICloneable
		{
			/// <summary>
			/// Id of the brick to be processed.
			/// </summary>
			public long BrickId = 0;

			#region ICloneable Members
			/// <summary>
			/// Clones the object.
			/// </summary>
			/// <returns>the object clone.</returns>
			public override object Clone()
			{
				BrickOperationInfo t = new BrickOperationInfo();
				t.DBPassword = this.DBPassword;
				t.DBServers = this.DBServers;
				t.DBUserName = this.DBUserName;
				t.ExeRepository = this.ExeRepository;
				t.LinkedZonePath = this.LinkedZonePath;
				t.MachineId = this.MachineId;				
				t.OPERAPassword = this.OPERAPassword;
				t.OPERAUsername = this.OPERAUsername;
				t.ProcessOperationId = this.ProcessOperationId;
				t.ProgramSettingsId = this.ProgramSettingsId;
				t.ProgressFile = this.ProgressFile;
				t.RawDataPath = this.RawDataPath;
				t.RecoverFromProgressFile = this.RecoverFromProgressFile;
				t.ScratchDir = this.ScratchDir;
				t.BrickId = this.BrickId;
				return t;
			}

			#endregion
		}

		/// <summary>
		/// Status information for a driver process.
		/// </summary>
		[Serializable]
		public enum Status
		{
			/// <summary>
			/// The process is unknown (it has never been scheduled).
			/// </summary>
			Unknown = 0,
			/// <summary>
			/// The process is running.
			/// </summary>
			Running = 1,
			/// <summary>
			/// The process is paused.
			/// </summary>
			Paused = 2,
			/// <summary>
			/// The process is completed.
			/// </summary>
			Completed = 3,
			/// <summary>
			/// The process failed.
			/// </summary>
			Failed = 4
		}

		/// <summary>
		/// Summarizes relevant information about a batch.
		/// </summary>
		[Serializable]
		public class BatchSummary
		{
			/// <summary>
			/// Id of the process operation.
			/// </summary>
			public long Id;
			/// <summary>
			/// The Id of the machine locked by the process operation.
			/// </summary>
			public long MachineId;
			/// <summary>
			/// Time when the operation started.
			/// </summary>
			public System.DateTime StartTime;
			/// <summary>
			/// The Id of the program settings used for the operation.
			/// </summary>
			public long ProgramSettingsId;
			/// <summary>
			/// The level of the process operation.
			/// </summary>
			public SySal.DAQSystem.Drivers.DriverType DriverLevel;
			/// <summary>
			/// The name of the executable module.
			/// </summary>
			public string Executable;
			/// <summary>
			/// The Id of the brick involved in the operation (if applicable; otherwise it is 0).
			/// </summary>
			public long BrickId;
			/// <summary>
			/// The Id of the plate involved in the operation (if applicable; otherwise it is 0).
			/// </summary>
			public long PlateId;
			/// <summary>
			/// The progress status of the operation.
			/// </summary>
			public double Progress;
			/// <summary>
			/// The expected finish time for the operation.
			/// </summary>
			public System.DateTime ExpectedFinishTime;
			/// <summary>
			/// The status of the process operation.
			/// </summary>
			public SySal.DAQSystem.Drivers.Status OpStatus;
		}

		/// <summary>
		/// An interrupt to a driver process.
		/// </summary>
		[Serializable]
		public class Interrupt
		{
			/// <summary>
			/// Interrupt id;
			/// </summary>
			public int Id;
			/// <summary>
			/// Interrupt data, in free format (depends on the driver).
			/// </summary>
			public string Data;
		}
	}
}
