using System;
using System.Collections;
using SySal;
using SySal.Tracking;
using NumericalTools;
using SySal.TotalScan;
using System.Xml.Serialization;
using System.Runtime.Serialization;
using System.Windows.Forms;


namespace SySal.Processing.VolumeGeneration
{
	class IntSegment: Segment
	{
		public IntSegment(SySal.Scanning.MIPBaseTrack tk, Layer layerowner, int posinlayer)
		{
			Info = tk.Info;
			m_LayerOwner = layerowner;
			m_PosInLayer = posinlayer; 
		}

		public IntSegment(SySal.Tracking.MIPEmulsionTrackInfo tk, Layer layerowner, int posinlayer)
		{
			Info = tk;
			m_LayerOwner = layerowner;
			m_PosInLayer = posinlayer; 
		}

		public IntSegment(Segment s)
		{
			Info = s.Info;
			m_LayerOwner = s.LayerOwner;
			m_PosInLayer = s.PosInLayer;
			m_TrackOwner = s.TrackOwner;
			//UpstreamLinked = s.UpstreamLinked;
			//DownstreamLinked = s.DownstreamLinked;
			
		}

		internal Layer IntLayer{ set {m_LayerOwner=value; } get {return m_LayerOwner;} }

		internal Segment UpstreamLinked = null;
		internal Segment DownstreamLinked = null;

		protected int m_Flag;
		public int Flag{  set {m_Flag=value;} get {return m_Flag;} }

	}

	/// <summary>
	/// Alignment data that can be initialized at will.
	/// </summary>
	public class AlignmentData : SySal.TotalScan.AlignmentData
	{
		/// <summary>
		/// Builds an empty AlignmentData class.
		/// </summary>
		public AlignmentData()
		{
			TranslationX = 0.0;
			TranslationY = 0.0;
			TranslationZ = 0.0;
			AffineMatrixXX = 1.0;
			AffineMatrixXY = 0.0;
			AffineMatrixYX = 0.0;
			AffineMatrixYY = 1.0;
			DShrinkX = 0.0;
			DShrinkY = 0.0;
			SAlignDSlopeX = 0.0;
			SAlignDSlopeY = 0.0;

		}
		/// <summary>
		/// Initializes a new AlignmentData instance.
		/// </summary>
		/// <param name="dShrink">2-component vector with shrink factors on X and Y.</param>
		/// <param name="sAlign_dSlope">2-component vector with linear distortion factors on X and Y.</param>
		/// <param name="Transl">3-component translations.</param>
		/// <param name="AffMat">2x2-matrix.</param>
		public AlignmentData(double[] dShrink, double[] sAlign_dSlope,
			double[] Transl, double[,] AffMat)
		{
			if (AffMat.GetLength(0)!=2 || AffMat.GetLength(1)!=2 || 
				Transl.Length!=3 || dShrink.Length !=2 || sAlign_dSlope.Length != 2) throw new Exception("....");
			TranslationX = Transl[0];
			TranslationY = Transl[1];
			TranslationZ = Transl[2];
			AffineMatrixXX = AffMat[0, 0];
			AffineMatrixXY = AffMat[0, 1];
			AffineMatrixYX = AffMat[1, 0];
			AffineMatrixYY = AffMat[1, 1];
			DShrinkX = dShrink[0];
			DShrinkY = dShrink[1];
			SAlignDSlopeX = sAlign_dSlope[0];
			SAlignDSlopeY = sAlign_dSlope[1];
		}

		public AlignmentData(double[] dShrink, double[] sAlign_dSlope, double[] Transformation)
		{
			if (dShrink.Length !=2 || 
				sAlign_dSlope.Length != 2) throw new Exception("....");
			TranslationX = Transformation[4];
			TranslationY = Transformation[5];
			TranslationZ = Transformation[6];
			AffineMatrixXX = Transformation[0];
			AffineMatrixXY = Transformation[1];
			AffineMatrixYX = Transformation[2];
			AffineMatrixYY = Transformation[3];
			DShrinkX = dShrink[0];
			DShrinkY = dShrink[1];
			SAlignDSlopeX = sAlign_dSlope[0];
			SAlignDSlopeY = sAlign_dSlope[1];
		}
	}

	/// <summary>
	/// Statistical distributions of simulated objects.
	/// </summary>
	public enum Distribution : int 
	{
		/// <summary>
		/// Custom distribution.
		/// </summary>
		Custom = 0, 
		/// <summary>
		/// Uniform distribution.
		/// </summary>
		Flat = 1, 
		/// <summary>
		/// Gaussian distribution.
		/// </summary>
		Gaussian = 2, 
		/// <summary>
		/// Delta distribution.
		/// </summary>
		SingleValue = 3
	}

	#region Configuration
	
	/// <summary>
	/// Configuration for VolumeGeneration.
	/// </summary>
	[Serializable]
	[XmlType("VolumeGeneration.Configuration")]
	public class Configuration : SySal.Management.Configuration, ICloneable, ISerializable
	{

		/// <summary>
		/// Reset to default parameters
		/// </summary>
		void DefaultParameters()
		{
			if(XSlopesDistrib == Distribution.Gaussian)
			{
				XSlopesDistribParameters[0] = 0;
				XSlopesDistribParameters[1] = 0.075;
			}
			else if(XSlopesDistrib == Distribution.Flat)
			{
				XSlopesDistribParameters[0] = -0.4;
				XSlopesDistribParameters[1] = 0.4;
			};

			if(YSlopesDistrib == Distribution.Gaussian)
			{
				YSlopesDistribParameters[0] = 0;
				YSlopesDistribParameters[1] = 0.075;
			}
			else if(YSlopesDistrib == Distribution.Flat)
			{
				YSlopesDistribParameters[0] = -0.4;
				YSlopesDistribParameters[1] = 0.4;
			};

			if(MomentumDistrib == Distribution.Gaussian)
			{
				MomentumDistribParameters[0] = 5;
				MomentumDistribParameters[1] = 0.5;
			}
			else if(MomentumDistrib == Distribution.Flat)
			{
				MomentumDistribParameters[0] = 0.5;
				MomentumDistribParameters[1] = 50;
			}
			else if(MomentumDistrib == Distribution.SingleValue)
			{
				MomentumDistribParameters[0] = 1;
			};

		}

		private Distribution m_XSlopesDistrib;

		/// <summary>
		/// Distribution for SlopesX.
		/// </summary>
		public Distribution XSlopesDistrib
		{
			get
			{
				return m_XSlopesDistrib;	
			}

			set
			{
				m_XSlopesDistrib = value;
				if (value == Distribution.Flat)
					XSlopesDistribParameters = new double[2];	
				else if (value == Distribution.Gaussian)
					XSlopesDistribParameters = new double[2];	
				else if (value == Distribution.SingleValue)
					XSlopesDistribParameters = new double[1];	
				DefaultParameters();
				
			}

		}

		/// <summary>
		/// Parameters for Distribution for SlopesX.
		/// </summary>
		public double[] XSlopesDistribParameters;

		private Distribution m_YSlopesDistrib;

		/// <summary>
		/// Distribution for SlopesY.
		/// </summary>
		public Distribution YSlopesDistrib
		{
			get
			{
				return m_YSlopesDistrib;	
			}

			set
			{
				m_YSlopesDistrib = value;
				if (value == Distribution.Flat)
					YSlopesDistribParameters = new double[2];	
				else if (value == Distribution.Gaussian)
					YSlopesDistribParameters = new double[2];	
				else if (value == Distribution.SingleValue)
					YSlopesDistribParameters = new double[1];	
				DefaultParameters();
				
			}

		}

		/// <summary>
		/// Parameters for Distribution for SlopesY.
		/// </summary>
		public double[] YSlopesDistribParameters;

		private Distribution m_MomentumDistrib;

		/// <summary>
		/// Distribution for Momentum.
		/// </summary>
		public Distribution MomentumDistrib
		{
			get
			{
				return m_MomentumDistrib;	
			}

			set
			{
				m_MomentumDistrib = value;
				if (value == Distribution.Flat)
					MomentumDistribParameters = new double[2];	
				else if (value == Distribution.Gaussian)
					MomentumDistribParameters = new double[2];	
				else if (value == Distribution.SingleValue)
					MomentumDistribParameters = new double[1];	
				DefaultParameters();
				
			}

		}
		/// <summary>
		/// Parameters for Distribution for Momentum.
		/// </summary>
		public double[] MomentumDistribParameters;

		/// <summary>
		/// Parameters for kinematic variables setting.
		/// </summary>
		public KinematicParameters KinePar;

		/// <summary>
		/// Parameters for error setting.
		/// </summary>
		public ErrorParameters ErrPar;

		/// <summary>
		/// Parameters for event.
		/// </summary>
		public EventParameters EvPar;

		/// <summary>
		/// Parameters for geometric variables setting.
		/// </summary>
		public GeometricParameters GeoPar;

		/// <summary>
		/// Parameters for geometric variables setting.
		/// </summary>
		public AffineParameters AffPar;

		/// <summary>
		/// High momentum tracks to generate in the out-of-bounds volume.
		/// </summary>
		public int HighMomentumTracks;

		/// <summary>
		/// Energy loss tracks to generate in the out-of-bounds volume.
		/// </summary>
		public int EnergyLossTracks;

		/// <summary>
		/// Junk tracks to generate in the out-of-bounds volume.
		/// </summary>
		public int JunkTracks;

		/// <summary>
		/// Builds an unitialized configuration.
		/// </summary>
		public Configuration() : base("") {}

		/// <summary>
		/// Builds a configuration with the specified name.
		/// </summary>
		/// <param name="name"></param>
		public Configuration(string name) : base(name) {}

		/// <summary>
		/// Yields a copy of the configuration.
		/// </summary>
		/// <returns></returns>
		public override object Clone()
		{
			Configuration C = new Configuration(Name);
			C.KinePar.MinimumEnergyForLoss = KinePar.MinimumEnergyForLoss;
			C.KinePar.RadiationLength = KinePar.RadiationLength;
			C.GeoPar.MostUpstreamPlane = GeoPar.MostUpstreamPlane;
			C.GeoPar.NotTrackingThickness = GeoPar.NotTrackingThickness;
			C.GeoPar.TrackingThickness = GeoPar.TrackingThickness;
			C.GeoPar.OutBoundsVolume.MaxX = GeoPar.OutBoundsVolume.MaxX;
			C.GeoPar.OutBoundsVolume.MaxY = GeoPar.OutBoundsVolume.MaxY;
			C.GeoPar.OutBoundsVolume.MaxZ = GeoPar.OutBoundsVolume.MaxZ;
			C.GeoPar.OutBoundsVolume.MinX = GeoPar.OutBoundsVolume.MinX;
			C.GeoPar.OutBoundsVolume.MinY = GeoPar.OutBoundsVolume.MinY;
			C.GeoPar.OutBoundsVolume.MinZ = GeoPar.OutBoundsVolume.MinZ;
			C.GeoPar.Volume.MaxX = GeoPar.Volume.MaxX;
			C.GeoPar.Volume.MinX = GeoPar.Volume.MinX;
			C.GeoPar.Volume.MaxY = GeoPar.Volume.MaxY;
			C.GeoPar.Volume.MinY = GeoPar.Volume.MinY;
			C.GeoPar.Volume.MaxZ = GeoPar.Volume.MaxZ;
			C.GeoPar.Volume.MinZ = GeoPar.Volume.MinZ;
			C.ErrPar.CoordinateAlignment.X = ErrPar.CoordinateAlignment.X;
			C.ErrPar.CoordinateAlignment.Y = ErrPar.CoordinateAlignment.Y;
			C.ErrPar.CoordinateErrors.X = ErrPar.CoordinateErrors.X;
			C.ErrPar.CoordinateErrors.Y = ErrPar.CoordinateErrors.Y;
			C.ErrPar.SlopeAlignment.X = ErrPar.SlopeAlignment.X;
			C.ErrPar.SlopeAlignment.Y = ErrPar.SlopeAlignment.Y;
			C.ErrPar.SlopeErrors.X = ErrPar.SlopeErrors.X;
			C.ErrPar.SlopeErrors.Y = ErrPar.SlopeErrors.Y;
			C.ErrPar.TrackFindingEfficiency = ErrPar.TrackFindingEfficiency;
			
			C.EvPar.LocalVertexDepth = EvPar.LocalVertexDepth;
			C.EvPar.OutgoingTracks = EvPar.OutgoingTracks;
			C.EvPar.PrimaryTrack = EvPar.PrimaryTrack;
			
			C.AffPar.AlignMaxShift = AffPar.AlignMaxShift;
			C.AffPar.AlignMinShift = AffPar.AlignMinShift;
			C.AffPar.DiagMaxTerm = AffPar.DiagMaxTerm;
			C.AffPar.DiagMinTerm = AffPar.DiagMinTerm;
			C.AffPar.LongAlignMaxShift = AffPar.LongAlignMaxShift;
			C.AffPar.LongAlignMinShift = AffPar.LongAlignMinShift;
			C.AffPar.OutDiagMaxTerm = AffPar.OutDiagMaxTerm;
			C.AffPar.OutDiagMinTerm = AffPar.OutDiagMinTerm;
			C.AffPar.SlopeMaxCoeff = AffPar.SlopeMaxCoeff;
			C.AffPar.SlopeMinCoeff = AffPar.SlopeMinCoeff;
			C.AffPar.SlopeMaxShift = AffPar.SlopeMaxShift;
			C.AffPar.SlopeMinShift = AffPar.SlopeMinShift;

			C.EnergyLossTracks = EnergyLossTracks;
			C.HighMomentumTracks = HighMomentumTracks;
			C.JunkTracks = JunkTracks;
			C.XSlopesDistrib = XSlopesDistrib;
			C.XSlopesDistribParameters = (double[])XSlopesDistribParameters.Clone();
			C.YSlopesDistrib = YSlopesDistrib;
			C.YSlopesDistribParameters = (double[])YSlopesDistribParameters.Clone();
			C.MomentumDistrib = MomentumDistrib;
			C.MomentumDistribParameters = (double[])MomentumDistribParameters.Clone();
			return C;
		}

		#region Serialization
		/// <summary>
		/// Saves the AOReconstruction parameters to a stream. This method is called by BinaryFormatter or SoapFormatter or other formatters.
		/// This method is overridden in derived classes to support specific serialization features.
		/// </summary>
		/// <param name="info">SerializationInfo data for serialization.</param>
		/// <param name="context">StreamingContext data for serialization.</param>
		public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			info.AddValue("KineticParameters_MinimumEnergyLoss", KinePar.MinimumEnergyForLoss);
			info.AddValue("KineticParameters_RadiationLength", KinePar.RadiationLength);

			info.AddValue("EventParameters_LocalVertexDepth", EvPar.LocalVertexDepth);
			info.AddValue("EventParameters_OutgoingTracks", EvPar.OutgoingTracks);
			info.AddValue("EventParameters_PrimaryTrack", EvPar.PrimaryTrack);

			info.AddValue("GeometricParameters_MostUpstreamPlane", GeoPar.MostUpstreamPlane);
			info.AddValue("GeometricParameters_NotTrackingThickness", GeoPar.NotTrackingThickness);
			info.AddValue("GeometricParameters_TrackingThickness", GeoPar.TrackingThickness);
			info.AddValue("GeometricParameters_OutBoundsVolume.MaxX", GeoPar.OutBoundsVolume.MaxX);
			info.AddValue("GeometricParameters_OutBoundsVolume.MinX", GeoPar.OutBoundsVolume.MinX);
			info.AddValue("GeometricParameters_OutBoundsVolume.MaxY", GeoPar.OutBoundsVolume.MaxY);
			info.AddValue("GeometricParameters_OutBoundsVolume.MinY", GeoPar.OutBoundsVolume.MinY);
			info.AddValue("GeometricParameters_OutBoundsVolume.MaxZ", GeoPar.OutBoundsVolume.MaxZ);
			info.AddValue("GeometricParameters_OutBoundsVolume.MinZ", GeoPar.OutBoundsVolume.MinZ);
			info.AddValue("GeometricParameters_Volume.MaxX", GeoPar.Volume.MaxX);
			info.AddValue("GeometricParameters_Volume.MinX", GeoPar.Volume.MinX);
			info.AddValue("GeometricParameters_Volume.MaxY", GeoPar.Volume.MaxY);
			info.AddValue("GeometricParameters_Volume.MinY", GeoPar.Volume.MinY);
			info.AddValue("GeometricParameters_Volume.MaxZ", GeoPar.Volume.MaxZ);
			info.AddValue("GeometricParameters_Volume.MinZ", GeoPar.Volume.MinZ);

			info.AddValue("ErrorParameters_CoordinateAlignment.X", ErrPar.CoordinateAlignment.X);
			info.AddValue("ErrorParameters_CoordinateAlignment.Y", ErrPar.CoordinateAlignment.Y);
			info.AddValue("ErrorParameters_CoordinateErrors.X", ErrPar.CoordinateErrors.X);
			info.AddValue("ErrorParameters_CoordinateErrors.Y", ErrPar.CoordinateErrors.Y);
			info.AddValue("ErrorParameters_SlopeAlignment.X", ErrPar.SlopeAlignment.X);
			info.AddValue("ErrorParameters_SlopeAlignment.Y", ErrPar.SlopeAlignment.Y);
			info.AddValue("ErrorParameters_SlopeErrors.X", ErrPar.SlopeErrors.X);
			info.AddValue("ErrorParameters_SlopeErrors.Y", ErrPar.SlopeErrors.Y);
			info.AddValue("ErrorParameters_Track_Finding_Efficiency", ErrPar.TrackFindingEfficiency);

			info.AddValue("AffineParameters_AlignMaxShift", AffPar.AlignMaxShift);
			info.AddValue("AffineParameters_AlignMinShift", AffPar.AlignMinShift);
			info.AddValue("AffineParameters_DiagMaxTerm", AffPar.DiagMaxTerm);
			info.AddValue("AffineParameters_DiagMinTerm", AffPar.DiagMinTerm);
			info.AddValue("AffineParameters_LongAlignMaxShift", AffPar.LongAlignMaxShift);
			info.AddValue("AffineParameters_LongAlignMinShift", AffPar.LongAlignMinShift);
			info.AddValue("AffineParameters_OutDiagMaxTerm", AffPar.OutDiagMaxTerm);
			info.AddValue("AffineParameters_OutDiagMinTerm", AffPar.OutDiagMinTerm);
			info.AddValue("AffineParameters_SlopeMaxCoeff", AffPar.SlopeMaxCoeff);
			info.AddValue("AffineParameters_SlopeMinCoeff", AffPar.SlopeMinCoeff);
			info.AddValue("AffineParameters_SlopeMaxShift", AffPar.SlopeMaxShift);
			info.AddValue("AffineParameters_SlopeMinShift", AffPar.SlopeMinShift);

			info.AddValue("Energy_Loss_Tracks", EnergyLossTracks);
			info.AddValue("High_Momentum_Tracks", HighMomentumTracks);
			info.AddValue("Junk_Tracks", JunkTracks);

			info.AddValue("XSlopes_Distribution", XSlopesDistrib);
			for(int i = 0; i < XSlopesDistribParameters.Length; i++) info.AddValue("XSlopes_Distribution_Parameter_" + (i+1), XSlopesDistribParameters[i]);
			info.AddValue("YSlopes_Distribution", YSlopesDistrib);
			for(int i = 0; i < YSlopesDistribParameters.Length; i++) info.AddValue("YSlopes_Distribution_Parameter_" + (i+1), YSlopesDistribParameters[i]);
			info.AddValue("Momentum_Distribution", MomentumDistrib);
			for(int i = 0; i < MomentumDistribParameters.Length; i++) info.AddValue("Momentum_Distribution_Parameter_" + (i+1), MomentumDistribParameters[i]);
		}

		/// <summary>
		/// Restores the AOReconstruction parameters to a stream. This constructor is called by BinaryFormatter or SoapFormatter or other formatters.
		/// </summary>
		/// <param name="info">SerializationInfo data for serialization.</param>
		/// <param name="context">StreamingContext data for serialization.</param>
		public Configuration(SerializationInfo info, StreamingContext context) : base(info.GetString("AlphaOmega_Configuration_Name")) 
		{
			//C = new Configuration(info.GetString("AlphaOmega_Configuration_Name"));
			KinePar.MinimumEnergyForLoss = info.GetDouble("KineticParameters_MinimumEnergyLoss");
			KinePar.RadiationLength = info.GetDouble("KineticParameters_RadiationLength");

			EvPar.LocalVertexDepth = info.GetDouble("EventParameters_LocalVertexDepth");
			EvPar.OutgoingTracks = info.GetInt32("EventParameters_OutgoingTracks");
			EvPar.PrimaryTrack = info.GetBoolean("EventParameters_PrimaryTrack");

			GeoPar.MostUpstreamPlane = info.GetInt32("GeometricParameters_MostUpstreamPlane");
			GeoPar.NotTrackingThickness = info.GetDouble("GeometricParameters_NotTrackingThickness");
			GeoPar.TrackingThickness = info.GetDouble("GeometricParameters_TrackingThickness");
			GeoPar.OutBoundsVolume.MaxX = info.GetDouble("GeometricParameters_OutBoundsVolume.MaxX");
			GeoPar.OutBoundsVolume.MinX = info.GetDouble("GeometricParameters_OutBoundsVolume.MinX");
			GeoPar.OutBoundsVolume.MaxY = info.GetDouble("GeometricParameters_OutBoundsVolume.MaxY");
			GeoPar.OutBoundsVolume.MinY = info.GetDouble("GeometricParameters_OutBoundsVolume.MinY");
			GeoPar.OutBoundsVolume.MaxZ = info.GetDouble("GeometricParameters_OutBoundsVolume.MaxZ");
			GeoPar.OutBoundsVolume.MinZ = info.GetDouble("GeometricParameters_OutBoundsVolume.MinZ");
			GeoPar.Volume.MaxX = info.GetDouble("GeometricParameters_Volume.MaxX");
			GeoPar.Volume.MinX = info.GetDouble("GeometricParameters_Volume.MinX");
			GeoPar.Volume.MaxY = info.GetDouble("GeometricParameters_Volume.MaxY");
			GeoPar.Volume.MinY = info.GetDouble("GeometricParameters_Volume.MinY");
			GeoPar.Volume.MaxZ = info.GetDouble("GeometricParameters_Volume.MaxZ");
			GeoPar.Volume.MinZ = info.GetDouble("GeometricParameters_Volume.MinZ");

			ErrPar.CoordinateAlignment.X = info.GetDouble("ErrorParameters_CoordinateAlignment.X");
			ErrPar.CoordinateAlignment.Y = info.GetDouble("ErrorParameters_CoordinateAlignment.Y");
			ErrPar.CoordinateErrors.X = info.GetDouble("ErrorParameters_CoordinateErrors.X");
			ErrPar.CoordinateErrors.Y = info.GetDouble("ErrorParameters_CoordinateErrors.Y");
			ErrPar.SlopeAlignment.X	 = info.GetDouble("ErrorParameters_SlopeAlignment.X");
			ErrPar.SlopeAlignment.Y = info.GetDouble("ErrorParameters_SlopeAlignment.Y");
			ErrPar.SlopeErrors.X = info.GetDouble("ErrorParameters_SlopeErrors.X");
			ErrPar.SlopeErrors.Y = info.GetDouble("ErrorParameters_SlopeErrors.Y");
			ErrPar.TrackFindingEfficiency = info.GetDouble("ErrorParameters_Track_Finding_Efficiency");

			AffPar.AlignMaxShift = info.GetDouble("AffineParameters_AlignMaxShift");
			AffPar.AlignMinShift = info.GetDouble("AffineParameters_AlignMinShift");
			AffPar.DiagMaxTerm = info.GetDouble("AffineParameters_DiagMaxTerm" );
			AffPar.DiagMinTerm = info.GetDouble("AffineParameters_DiagMinTerm" );
			AffPar.LongAlignMaxShift= info.GetDouble("AffineParameters_LongAlignMaxShift" );
			AffPar.LongAlignMinShift = info.GetDouble("AffineParameters_LongAlignMinShift" );
			AffPar.OutDiagMaxTerm = info.GetDouble("AffineParameters_OutDiagMaxTerm" );
			AffPar.OutDiagMinTerm = info.GetDouble("AffineParameters_OutDiagMinTerm" );
			AffPar.SlopeMaxCoeff = info.GetDouble("AffineParameters_SlopeMaxCoeff" );
			AffPar.SlopeMinCoeff = info.GetDouble("AffineParameters_SlopeMinCoeff" );
			AffPar.SlopeMaxShift = info.GetDouble("AffineParameters_SlopeMaxShift" );
			AffPar.SlopeMinShift = info.GetDouble("AffineParameters_SlopeMinShift" );

			EnergyLossTracks = info.GetInt32("Energy_Loss_Tracks");
			HighMomentumTracks = info.GetInt32("High_Momentum_Tracks");
			JunkTracks = info.GetInt32("Junk_Tracks");

			XSlopesDistrib = (VolumeGeneration.Distribution)info.GetInt32("XSlopes_Distribution");
			for(int i = 0; i < XSlopesDistribParameters.Length; i++) XSlopesDistribParameters[i] = info.GetDouble("XSlopes_Distribution_Parameter_" + (i+1));
			YSlopesDistrib = (VolumeGeneration.Distribution)info.GetInt32("YSlopes_Distribution");
			for(int i = 0; i < YSlopesDistribParameters.Length; i++) YSlopesDistribParameters[i] = info.GetDouble("YSlopes_Distribution_Parameter_" + (i+1));
			MomentumDistrib = (VolumeGeneration.Distribution)info.GetInt32("Momentum_Distribution");
			for(int i = 0; i < MomentumDistribParameters.Length; i++) MomentumDistribParameters[i] = info.GetDouble("Momentum_Distribution_Parameter_" + (i+1));
		}
		#endregion

	}

	#endregion

	#region Parameter Classes
	/// <summary>
	/// Kinematical parameters for simulation.
	/// </summary>
	public struct KinematicParameters
	{
		/// <summary>
		/// The radiation length of the material.
		/// </summary>
		public double RadiationLength;
		/// <summary>
		/// Minimum energy loss.
		/// </summary>
		public double MinimumEnergyForLoss;
		/// <summary>
		/// Initializes a new instance of the structure.
		/// </summary>
		/// <param name="radiation_length">the radiation length.</param>
		/// <param name="minimum_energy_for_loss">the minimum energy loss.</param>
		public KinematicParameters(double radiation_length, double minimum_energy_for_loss)
		{
			RadiationLength = radiation_length;
			MinimumEnergyForLoss = minimum_energy_for_loss;
		}

	}

	/// <summary>
	/// Simulated errors.
	/// </summary>
	public struct ErrorParameters
	{
		/// <summary>
		/// Errors on positions.
		/// </summary>
		public SySal.BasicTypes.Vector2 CoordinateErrors;
		/// <summary>
		/// Errors on slopes.
		/// </summary>
		public SySal.BasicTypes.Vector2 SlopeErrors;
		/// <summary>
		/// Alignment errors on positions.
		/// </summary>
		public SySal.BasicTypes.Vector2 CoordinateAlignment;
		/// <summary>
		/// Alignment errors on slopes.
		/// </summary>
		public SySal.BasicTypes.Vector2 SlopeAlignment;
		/// <summary>
		/// Tracking efficiency.
		/// </summary>
		public double TrackFindingEfficiency;
		/// <summary>
		/// Initializes a new instance of the structure.
		/// </summary>
		/// <param name="err_coord">the errors on positions.</param>
		/// <param name="err_slope">the errors on slopes.</param>
		/// <param name="coord_align">the alignment errors on positions.</param>
		/// <param name="slope_align">the alignment errors on slopes.</param>
		/// <param name="track_finding_efficiency">the tracking efficiency.</param>
		public ErrorParameters(SySal.BasicTypes.Vector2 err_coord, SySal.BasicTypes.Vector2 err_slope,
			SySal.BasicTypes.Vector2 coord_align, SySal.BasicTypes.Vector2 slope_align, double track_finding_efficiency)
		{
			CoordinateErrors = err_coord;
			SlopeErrors = err_slope;
			CoordinateAlignment = coord_align;
			SlopeAlignment = slope_align;
			TrackFindingEfficiency = track_finding_efficiency;
		}

	}
	/// <summary>
	/// Geometrical parameters.
	/// </summary>
	public struct GeometricParameters
	{
		/// <summary>
		/// The fiducial volume.
		/// </summary>
		public SySal.BasicTypes.Cuboid Volume;
		/// <summary>
		/// This volume contains the fiducial volume, and processes happening here appear partially in the fiducial volume too.
		/// </summary>
		public SySal.BasicTypes.Cuboid OutBoundsVolume;
		/// <summary>
		/// Tracking (sensitive) thickness of the volume.
		/// </summary>
		public double TrackingThickness;
		/// <summary>
		/// Insensitive thickness of the volume.
		/// </summary>
		public double NotTrackingThickness;
		/// <summary>
		/// The most upstream plane.
		/// </summary>
		public int MostUpstreamPlane;
		/// <summary>
		/// Initializes a new instance of the structure.
		/// </summary>
		/// <param name="tracking_thickness">the tracking thickness.</param>
		/// <param name="not_tracking_thickness">the insensitive thickness.</param>
		/// <param name="local_vertex_depth">the local vertex depth.</param>
		/// <param name="most_upstream_plane">the most upstream plane.</param>
		/// <param name="volume">the fiducial volume.</param>
		/// <param name="out_bounds_volume">the extra volume that contains the fiducial volume.</param>
			
		public GeometricParameters(double tracking_thickness, double not_tracking_thickness, 
			double local_vertex_depth, int most_upstream_plane,
			SySal.BasicTypes.Cuboid volume, SySal.BasicTypes.Cuboid out_bounds_volume)
		{
			TrackingThickness = tracking_thickness;
			NotTrackingThickness = not_tracking_thickness;
			Volume = volume;
			OutBoundsVolume = out_bounds_volume;
			MostUpstreamPlane = most_upstream_plane;
		}

	}
	/// <summary>
	/// Paramaters of an event.
	/// </summary>
	public struct EventParameters
	{
		/// <summary>
		/// Local depth of the vertex.
		/// </summary>
		public double LocalVertexDepth;
		/// <summary>
		/// Number of outgoing tracks.
		/// </summary>
		public int OutgoingTracks;
		/// <summary>
		/// If true, the vertex has a primary track.
		/// </summary>
		public bool PrimaryTrack;
		/// <summary>
		/// Initializes a new instance of the structure.
		/// </summary>
		/// <param name="local_vertex_depth">the local vertex depth.</param>
		/// <param name="outgoing_tracks">the number of outgoing tracks.</param>
		/// <param name="primary_track">flag for primary track presence.</param>
		public EventParameters(double local_vertex_depth, int outgoing_tracks, bool primary_track)
		{
			LocalVertexDepth = local_vertex_depth;
			OutgoingTracks = outgoing_tracks;
			PrimaryTrack = primary_track;
		}
	}

	/// <summary>
	/// Parameters for affine transformations.
	/// </summary>
	public struct AffineParameters
	{
		/// <summary>
		/// Maximum transverse alignment shift.
		/// </summary>
		public double AlignMaxShift;
		/// <summary>
		/// Minimum transverse alignment shift.
		/// </summary>
		public double AlignMinShift;
		/// <summary>
		/// Maximum transverse longitudinal shift.
		/// </summary>
		public double LongAlignMaxShift;
		/// <summary>
		/// Minimum transverse longitudinal shift.
		/// </summary>
		public double LongAlignMinShift;
		/// <summary>
		/// Maximum value for diagonal terms in the affine transformation.
		/// </summary>
		public double DiagMaxTerm;
		/// <summary>
		/// Minimum value for diagonal terms in the affine transformation.
		/// </summary>
		public double DiagMinTerm;
		/// <summary>
		/// Maximum value for off-diagonal terms in the affine transformation.
		/// </summary>
		public double OutDiagMaxTerm;
		/// <summary>
		/// Minimum value for off-diagonal terms in the affine transformation.
		/// </summary>
		public double OutDiagMinTerm;
		/// <summary>
		/// Maximum multiplicative coefficient for slopes.
		/// </summary>
		public double SlopeMaxCoeff;
		/// <summary>
		/// Minimum multiplicative coefficient for slopes.
		/// </summary>
		public double SlopeMinCoeff;
		/// <summary>
		/// Maximum slope deviation.
		/// </summary>
		public double SlopeMaxShift;
		/// <summary>
		/// Minimum slope deviation.
		/// </summary>
		public double SlopeMinShift;
		/// <summary>
		/// Initializes a new instance of the structure.
		/// </summary>
		/// <param name="Align_Max_Shift">the maximum transverse shift.</param>
		/// <param name="Align_Min_Shift">the minimum transverse shift.</param>
		/// <param name="LongAlign_Max_Shift">the maximum longitudinal shift.</param>
		/// <param name="LongAlign_Min_Shift">the minimum longitudinal shift.</param>
		/// <param name="Diag_Max_Term">the maximum value for diagonal terms.</param>
		/// <param name="Diag_Min_Term">the minimum value for diagonal terms.</param>
		/// <param name="OutDiag_Max_Term">the maximum value for off-diagonal terms.</param>
		/// <param name="OutDiag_Min_Term">the minimum value for off-diagonal terms.</param>
		/// <param name="Slope_Max_Coeff">the maximum value for slope multiplier.</param>
		/// <param name="Slope_Min_Coeff">the minimum value for slope multiplier.</param>
		/// <param name="Slope_Max_Shift">the maximum value for slope deviation.</param>
		/// <param name="Slope_Min_Shift">the minimum value for slope deviation.</param>
		public AffineParameters(double Align_Max_Shift, double Align_Min_Shift, double LongAlign_Max_Shift, double LongAlign_Min_Shift,
            double Diag_Max_Term, double Diag_Min_Term, double OutDiag_Max_Term, double OutDiag_Min_Term, double Slope_Max_Coeff,
            double Slope_Min_Coeff, double Slope_Max_Shift, double Slope_Min_Shift)
		{
			AlignMaxShift=Align_Max_Shift;
			AlignMinShift=Align_Min_Shift;
			LongAlignMaxShift=LongAlign_Max_Shift;
			LongAlignMinShift=LongAlign_Min_Shift;
			DiagMaxTerm=Diag_Max_Term;
			DiagMinTerm=Diag_Min_Term;
			OutDiagMaxTerm=OutDiag_Max_Term;
			OutDiagMinTerm=OutDiag_Min_Term;
			SlopeMaxCoeff=Slope_Max_Coeff;
			SlopeMinCoeff=Slope_Min_Coeff;
			SlopeMaxShift=Slope_Max_Shift;
			SlopeMinShift=Slope_Min_Shift;
		}

	}

	#endregion


	/// <summary>
	/// Volume Generator.
	/// </summary>
	/// <remarks>
	/// This class generates a volume according to the configuration specified.
	/// </remarks>
	[Serializable]
	[XmlType("VolumeGeneration.VolumeGenerator")]
	public class VolumeGenerator
	{
		[NonSerialized]
		private VolumeGeneration.Configuration C;
		/// <summary>
		/// Builds a new instance of the Volume Generator.
		/// </summary>
		public VolumeGenerator()
		{
			//
			// TODO: Add constructor logic here
			//
			C = new Configuration("Default VolumeGenerator Configuration");

			//Errors
			C.ErrPar.CoordinateAlignment.X = 0;
			C.ErrPar.CoordinateAlignment.Y = 0;
			C.ErrPar.CoordinateErrors.X = 2;
			C.ErrPar.CoordinateErrors.Y = 2;
			C.ErrPar.SlopeAlignment.X = 0;
			C.ErrPar.SlopeAlignment.Y = 0;
			C.ErrPar.SlopeErrors.X = 0.002;
			C.ErrPar.SlopeErrors.Y = 0.002;
			C.ErrPar.TrackFindingEfficiency = 0.99;

			//Geometry
			C.GeoPar.MostUpstreamPlane = 56;
			C.GeoPar.NotTrackingThickness = 1100;
			C.GeoPar.TrackingThickness = 200;
			C.GeoPar.OutBoundsVolume.MaxX = 5000;
			C.GeoPar.OutBoundsVolume.MinX = -5000;
			C.GeoPar.OutBoundsVolume.MaxY = 5000;
			C.GeoPar.OutBoundsVolume.MinY = -5000;
			C.GeoPar.OutBoundsVolume.MaxZ = 5500;
			C.GeoPar.OutBoundsVolume.MinZ = 0;
			C.GeoPar.Volume.MaxX = 2500;
			C.GeoPar.Volume.MinX = -2500;
			C.GeoPar.Volume.MaxY = 2500;
			C.GeoPar.Volume.MinY = -2500;
			C.GeoPar.Volume.MaxZ = 5500;
			C.GeoPar.Volume.MinZ = 0;

			//Event
			C.EvPar.LocalVertexDepth = 0;
			C.EvPar.OutgoingTracks = 3;
			C.EvPar.PrimaryTrack = false;
			
			//Kinematics
			C.KinePar.MinimumEnergyForLoss = 0.1;
			C.KinePar.RadiationLength = 5850;

			//Affine
			C.AffPar.AlignMaxShift = 500;
			C.AffPar.AlignMinShift = -500;
			C.AffPar.DiagMaxTerm = 1.005;
			C.AffPar.DiagMinTerm = 0.995;
			C.AffPar.LongAlignMaxShift = 20;
			C.AffPar.LongAlignMinShift = -20;
			C.AffPar.OutDiagMaxTerm = 0.005;
			C.AffPar.OutDiagMinTerm = -0.005;
			C.AffPar.SlopeMaxCoeff = 1.005;
			C.AffPar.SlopeMinCoeff = 0.995;
			C.AffPar.SlopeMaxShift = 0.005;
			C.AffPar.SlopeMinShift = -0.005;


			//Tracks Number
			C.EnergyLossTracks = 0;
			C.HighMomentumTracks = 1000;
			C.JunkTracks = 0;

			//Distributions
			C.XSlopesDistrib = VolumeGeneration.Distribution.Gaussian;
			C.YSlopesDistrib = VolumeGeneration.Distribution.Gaussian;
			C.MomentumDistrib = VolumeGeneration.Distribution.Flat;
			
		}

		private string intName;
		/// <summary>
		/// Name of the VolumeGenerator instance.
		/// </summary>
		public string Name
		{
			get
			{
				return intName;	
			}
			set
			{
				intName = value;	
			}
		}

		/// <summary>
		/// Accesses the VolumeGenerator's configuration.
		/// </summary>
		[XmlElement(typeof(VolumeGeneration.Configuration))]
		public SySal.Management.Configuration Config
		{
			get
			{
				return (VolumeGeneration.Configuration)C.Clone();
			}
			set
			{
				C = (VolumeGeneration.Configuration)value.Clone();
			}
		}

		/// <summary>
		/// Allows the user to edit the supplied configuration.
		/// </summary>
		/// <param name="c">the configuration to be edited.</param>
		/// <returns>the result of modified configuration.</returns>
		public bool EditConfiguration(ref SySal.Management.Configuration c)
		{
			bool ret;
			frmParameters myform = new frmParameters();
			myform.VConfig = (VolumeGeneration.Configuration)c.Clone();
			if ((ret = (myform.ShowDialog() == DialogResult.OK))) c = myform.VConfig;
			myform.Dispose();
			return ret;
		}

		/// <summary>
		/// Generates an event.
		/// </summary>
		/// <param name="StartingOrdinalID">the identifier for the event.</param>
		/// <param name="TrInfo">tracks to be injected.</param>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="TracksMomenta">tracks momenta. This array must have as many elements as the TrInfo parameter.</param>
		/// <returns>tracks generated according to the specified physics and simulation settings.</returns>
		public Track[] GenerateEvent(int StartingOrdinalID, SySal.Tracking.MIPEmulsionTrackInfo[] TrInfo, 
			VolumeGeneration.Configuration C , double[] TracksMomenta)
		{
			int i, plane;
			double zv, yv, xv, zp;
			double erry=0, errx=0, errsy=0, errsx=0, momy=0, momx=0, momsy=0, momsx=0;
			double[] y, z, x, Sx, Sy; 
			bool[] outofvolume;
			int ntracks = TrInfo.Length;
			double Theta0, r1=0, r2=0;
			double dz;

			ArrayList trcoll = new ArrayList();
			Track[] tr;
			IntSegment s;
			Layer l;
			int[] posinlayer = new int[C.GeoPar.MostUpstreamPlane];
			SySal.BasicTypes.Vector c = new SySal.BasicTypes.Vector();
			MIPEmulsionTrackInfo tk;

			outofvolume = new bool[ntracks];
			Sx = new double[ntracks];
			Sy = new double[ntracks];
			x = new double[ntracks];
			y = new double[ntracks];
			z = new double[ntracks];
			tr = new Track[ntracks];
			double cellsize = C.GeoPar.TrackingThickness+C.GeoPar.NotTrackingThickness;

			
			///ATTENZIONE AL 2*CellSize
			zv = C.GeoPar.Volume.MinZ + (2*cellsize-C.EvPar.LocalVertexDepth);
			yv = (C.GeoPar.Volume.MinY +C.GeoPar.Volume.MaxY) /2;
			xv = (C.GeoPar.Volume.MinX +C.GeoPar.Volume.MaxX) /2;

			for(i=0; i<ntracks;i++)
			{
				tr[i] = new Track(i+StartingOrdinalID);
				//tr[i].Id=i+StartingOrdinalID;
				tr[i].Comment="Out";
				y[i]=yv;
				z[i]=zv;
				x[i]=xv;
				Sx[i] = TrInfo[i].Slope.X;
				Sy[i] = TrInfo[i].Slope.Y;
			};
			
			for(i=0; i<ntracks;i++)
			{
				plane=0;
				zp = C.GeoPar.OutBoundsVolume.MinZ + 2*cellsize;
				dz = 2*cellsize - C.EvPar.LocalVertexDepth;
				while(zp<C.GeoPar.Volume.MaxZ)
				{
			
					//Simulazione Momento
					Theta0= (0.0136 / TracksMomenta[i]) * 
						Math.Sqrt(cellsize/C.KinePar.RadiationLength) * 
						(1 + 0.038 * Math.Log(cellsize/C.KinePar.RadiationLength));

					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
					momy = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
					momsy = r2 * Theta0;

					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
					momx = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
					momsx = r2 * Theta0;

					//sono le posizioni vere servono per il prossimo aggiornamento
					//a prescindere che ci sia la misura con relativo errore
					y[i] = (y[i] + dz*Sy[i] + momy);
					x[i] = (x[i] + dz*Sx[i] + momx);
					Sy[i] = (Sy[i] + momsy);
					Sx[i] = (Sx[i] + momsx);

					//Traccia all'interno del volume
					if (y[i]<C.GeoPar.Volume.MaxY && y[i]>C.GeoPar.Volume.MinY && 
						x[i]<C.GeoPar.Volume.MaxX && x[i]>C.GeoPar.Volume.MinX) 
					{

						//Check di Efficienza
						//se oltrepassato interviene la misura con i suoi errori
						MonteCarlo.Flat_Rnd_Number(0, 1, ref r1);
						if (r1 < C.ErrPar.TrackFindingEfficiency)
						{
							if(tr[i].Comment=="Out") tr[i].Comment="VertexTrack";

							//Errore
							//idx_al_plane = plane;
							//while(idx_al_plane > ep.CoordinateAlignment.Length) idx_al_plane -= ep.CoordinateAlignment.Length;
							MonteCarlo.Gaussian_Rnd_Number(/*ep.CoordinateAlignment[idx_al_plane].Y*/0, C.ErrPar.CoordinateErrors.Y, ref erry);
							MonteCarlo.Gaussian_Rnd_Number(/*ep.CoordinateAlignment[idx_al_plane].X*/0, C.ErrPar.CoordinateErrors.X, ref errx);

							//idx_al_plane = plane;
							//while(idx_al_plane > ep.SlopeAlignment.Length) idx_al_plane -= ep.SlopeAlignment.Length;
							MonteCarlo.Gaussian_Rnd_Number(/*ep.SlopeAlignment[idx_al_plane].Y*/0, C.ErrPar.SlopeErrors.Y, ref errsy);
							MonteCarlo.Gaussian_Rnd_Number(/*ep.SlopeAlignment[idx_al_plane].X*/0, C.ErrPar.SlopeErrors.X, ref errsx);

							tk = new MIPEmulsionTrackInfo(); 
							tk.Intercept.Z = (float)zp;
							//tk.Intercept.Y = (float)(y[i] + dz*TrInfo[i].Slope.Y + erry + momy);
							//tk.Intercept.X = (float)(x[i] + dz*TrInfo[i].Slope.X + errx + momx);
							tk.Intercept.Y = (float)(y[i] + erry);
							tk.Intercept.X = (float)(x[i] + errx);
							tk.TopZ = (float)zp;
							tk.BottomZ = (float)(zp - C.GeoPar.TrackingThickness);
							tk.Slope.X = (float)(Sx[i] + errsx);
							tk.Slope.Y = (float)(Sy[i] + errsy);
							
							l=new Layer(C.GeoPar.MostUpstreamPlane - plane,0,C.GeoPar.MostUpstreamPlane - plane,0,c);
							s = new IntSegment(tk, l, posinlayer[C.GeoPar.MostUpstreamPlane - plane]);
							posinlayer[C.GeoPar.MostUpstreamPlane - plane]++;
							//s = new Segment(tk, gp.MostUpstreamPlane - plane);
							//s = new Segment(tk);
							//Bisogna assegnare al segmento il  vecchio PosID (vedi riga commentata)
							tr[i].AddSegment(s);
						};
					};
					dz =cellsize;
					zp+=cellsize;
					plane++;
				};
				if (tr[i].Comment=="VertexTrack") trcoll.Add(tr[i]);				
			};
			return (Track[])(trcoll.ToArray(typeof(Track)));
		}

		/// <summary>
		/// Generates an event.
		/// </summary>
		/// <param name="layer">the tracking layers.</param>
		/// <param name="StartingOrdinalID">the Id for the event.</param>
		/// <param name="TrInfo">the tracks to be injected.</param>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="TracksMomenta">tracks momenta. This array must have as many elements as the TrInfo parameter.</param>
		/// <returns>tracks generated according to the specified physics and simulation settings.</returns>
		public Layer[] GenerateEvent(Layer[] layer, int StartingOrdinalID, SySal.Tracking.MIPEmulsionTrackInfo[] TrInfo, 
			VolumeGeneration.Configuration C , double[] TracksMomenta)
		{
			int i, plane;
			double zv, yv, xv, zp=0;
			double erry=0, errx=0, errsy=0, errsx=0, momy=0, momx=0, momsy=0, momsx=0;
			double[] y, z, x, Sx, Sy; 
			bool[] outofvolume;
			int ntracks = TrInfo.Length;
			double Theta0, r1=0, r2=0;
			double dz;

			ArrayList trcoll = new ArrayList();
			Track[] tr;
			IntSegment s;
			int[] posinlayer = new int[C.GeoPar.MostUpstreamPlane + 1];
			for(i=0; i< layer.Length; i++) posinlayer[C.GeoPar.MostUpstreamPlane - layer.Length + i + 1]= layer[i].Length;
			SySal.BasicTypes.Vector c = new SySal.BasicTypes.Vector();
			MIPEmulsionTrackInfo tk;

			outofvolume = new bool[ntracks];
			Sx = new double[ntracks];
			Sy = new double[ntracks];
			x = new double[ntracks];
			y = new double[ntracks];
			z = new double[ntracks];
			tr = new Track[ntracks];
			double cellsize = C.GeoPar.TrackingThickness+C.GeoPar.NotTrackingThickness;

			
			///ATTENZIONE AL 2*CellSize
			zv = C.EvPar.LocalVertexDepth;
			yv = (C.GeoPar.Volume.MinY +C.GeoPar.Volume.MaxY) /2;
			xv = (C.GeoPar.Volume.MinX +C.GeoPar.Volume.MaxX) /2;

			dz = cellsize;
			
			for(i=0; i<ntracks;i++)
			{
				plane=0;
				if (i!=0 || (i==0 && !C.EvPar.PrimaryTrack)) zp = TrInfo[i].Intercept.Z + Convert.ToInt32(/*1+*/0.5+C.EvPar.LocalVertexDepth/cellsize)*cellsize;
				if (i==0 && C.EvPar.PrimaryTrack) zp = TrInfo[i].Intercept.Z + cellsize;

				tr[i] = new Track(i+StartingOrdinalID);
				tr[i].Comment="Out";
				Sx[i] = TrInfo[i].Slope.X;
				Sy[i] = TrInfo[i].Slope.Y;
				int idlayer = layer.Length-1 - ((int)((zp-TrInfo[i].Intercept.Z)/cellsize)-1);
				//C'è bisogno di verificare le cond. iniziali:
				//la traccia potrebbe essere fuori dal volume
//				{
					//z[i] = layer[idlayer].DownstreamZ-dz;// + (i==PrimaryIndex?-dz:0);
					z[i] = layer[layer.Length-1].DownstreamZ + ((int)((zp-TrInfo[i].Intercept.Z)/cellsize)-1)*cellsize -dz;// + (i==PrimaryIndex?-dz:0);

					x[i] = /*(i==0 && C.EvPar.PrimaryTrack?*/(z[i]-zv)*Sx[i] + xv/*:(z[i]-zv)*Sx[i] + xv)*/;
					y[i] = /*(i==0 && C.EvPar.PrimaryTrack?*/(z[i]-zv)*Sy[i] + yv/*:(z[i]-zv)*Sy[i] + yv)*/;
//				}

				while((i!=0 && zp<C.GeoPar.Volume.MaxZ)||
					(i==0 && C.EvPar.PrimaryTrack && zp<C.EvPar.LocalVertexDepth) || 
					(i==0 && !C.EvPar.PrimaryTrack && zp<C.GeoPar.Volume.MaxZ) )
				{
			
					//Simulazione Momento
					Theta0= (0.0136 / TracksMomenta[i]) * 
						Math.Sqrt(cellsize/C.KinePar.RadiationLength) * 
						(1 + 0.038 * Math.Log(cellsize/C.KinePar.RadiationLength));

					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
					momy = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
					momsy = r2 * Theta0;

					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
					MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
					momx = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
					momsx = r2 * Theta0;

					//sono le posizioni vere servono per il prossimo aggiornamento
					//a prescindere che ci sia la misura con relativo errore
					y[i] = (y[i] + dz*Sy[i] + momy);
					x[i] = (x[i] + dz*Sx[i] + momx);
					Sy[i] = (Sy[i] + momsy);
					Sx[i] = (Sx[i] + momsx);

					//Traccia all'interno del volume
					if (y[i]<C.GeoPar.Volume.MaxY && y[i]>C.GeoPar.Volume.MinY && 
						x[i]<C.GeoPar.Volume.MaxX && x[i]>C.GeoPar.Volume.MinX) 
					{

						//Check di Efficienza
						//se oltrepassato interviene la misura con i suoi errori
						MonteCarlo.Flat_Rnd_Number(0, 1, ref r1);
						if (r1 < C.ErrPar.TrackFindingEfficiency)
						{
							if(tr[i].Comment=="Out") tr[i].Comment="VertexTrack";

							//Errore
							MonteCarlo.Gaussian_Rnd_Number(/*ep.CoordinateAlignment[idx_al_plane].Y*/0, C.ErrPar.CoordinateErrors.Y, ref erry);
							MonteCarlo.Gaussian_Rnd_Number(/*ep.CoordinateAlignment[idx_al_plane].X*/0, C.ErrPar.CoordinateErrors.X, ref errx);

							MonteCarlo.Gaussian_Rnd_Number(/*ep.SlopeAlignment[idx_al_plane].Y*/0, C.ErrPar.SlopeErrors.Y, ref errsy);
							MonteCarlo.Gaussian_Rnd_Number(/*ep.SlopeAlignment[idx_al_plane].X*/0, C.ErrPar.SlopeErrors.X, ref errsx);

							tk = new MIPEmulsionTrackInfo(); 
							tk.Intercept.Z = (float)zp;
							tk.Intercept.Y = (float)(y[i] + erry);
							tk.Intercept.X = (float)(x[i] + errx);
							tk.TopZ = (float)zp;
							tk.BottomZ = (float)(zp - C.GeoPar.TrackingThickness);
							tk.Slope.X = (float)(Sx[i] + errsx);
							tk.Slope.Y = (float)(Sy[i] + errsy);
							
							if(idlayer<layer.Length && idlayer>-1)
							{
								s = new IntSegment(tk, layer[idlayer], posinlayer[C.GeoPar.MostUpstreamPlane - plane]);
								posinlayer[C.GeoPar.MostUpstreamPlane - plane]++;
								//s = new Segment(tk, gp.MostUpstreamPlane - plane);
								//s = new Segment(tk);
								//Bisogna assegnare al segmento il  vecchio PosID (vedi riga commentata)
								//tr[i].AddSegment(s);
								layer[idlayer].AddSegment(s);
							}

						};
					};
					dz =cellsize;
					//zp+=cellsize;
					plane++;

					zp+=dz;
					idlayer = layer.Length-1 - ((int)((zp-TrInfo[i].Intercept.Z)/cellsize)-1);
				};
				if (tr[i].Comment=="VertexTrack") trcoll.Add(tr[i]);				
			};

			return layer;


		}

		private int mGeneratingLayers;
		/// <summary>
		/// Number of layers to generate.
		/// </summary>
		public int GeneratingLayers
		{
			get
			{
				return mGeneratingLayers;	
			}
			set
			{
				mGeneratingLayers = value;	
			}
		}
		
		/// <summary>
		/// Builds initial conditions for a volume.
		/// </summary>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="TrInfo">on exit, this array contains the positions and slopes of generated tracks.</param>
		/// <param name="TracksMomenta">on exit, this array contains the moment for generated tracks.</param>
		public void InitialConditions(VolumeGeneration.Configuration C, 
			out SySal.Tracking.MIPEmulsionTrackInfo[] TrInfo, out double[] TracksMomenta)
		{
			int ntracks = C.HighMomentumTracks + C.EnergyLossTracks + 
				C.JunkTracks*(int)((C.GeoPar.Volume.MaxZ-C.GeoPar.Volume.MinZ)/(C.GeoPar.TrackingThickness+C.GeoPar.NotTrackingThickness));

			TrInfo = new SySal.Tracking.MIPEmulsionTrackInfo[ntracks];
			TracksMomenta = new double[ntracks];

			//Incoming Conditions of Tracks
			for(int i=0; i<ntracks;i++)
			{
				//tr[i] = new Track(i+StartingTrackID);
				//tr[i].Comment="Out";
				TrInfo[i] = new SySal.Tracking.MIPEmulsionTrackInfo();
				TrInfo[i].Intercept.X = MonteCarlo.Flat_Rnd_Number(C.GeoPar.OutBoundsVolume.MinX,C.GeoPar.OutBoundsVolume.MaxX);
				TrInfo[i].Intercept.Y = MonteCarlo.Flat_Rnd_Number(C.GeoPar.OutBoundsVolume.MinY,C.GeoPar.OutBoundsVolume.MaxY);
				TrInfo[i].TopZ = TrInfo[i].Intercept.Z = C.GeoPar.OutBoundsVolume.MinZ;
				TrInfo[i].BottomZ = C.GeoPar.OutBoundsVolume.MinZ - C.GeoPar.TrackingThickness;
				if (C.XSlopesDistrib == Distribution.Flat) TrInfo[i].Slope.X = MonteCarlo.Flat_Rnd_Number(C.XSlopesDistribParameters[0],C.XSlopesDistribParameters[1]);
				else if (C.XSlopesDistrib == Distribution.Gaussian) TrInfo[i].Slope.X = MonteCarlo.Gaussian_Rnd_Number(C.XSlopesDistribParameters[0],C.XSlopesDistribParameters[1]);
				else if (C.XSlopesDistrib == Distribution.SingleValue) TrInfo[i].Slope.X = C.XSlopesDistribParameters[0];
				if (C.YSlopesDistrib == Distribution.Flat) TrInfo[i].Slope.Y = MonteCarlo.Flat_Rnd_Number(C.YSlopesDistribParameters[0],C.YSlopesDistribParameters[1]);
				else if (C.YSlopesDistrib == Distribution.Gaussian) TrInfo[i].Slope.Y = MonteCarlo.Gaussian_Rnd_Number(C.YSlopesDistribParameters[0],C.YSlopesDistribParameters[1]);
				else if (C.YSlopesDistrib == Distribution.SingleValue) TrInfo[i].Slope.Y = C.YSlopesDistribParameters[0];
				if (C.MomentumDistrib == Distribution.Flat) TracksMomenta[i] = MonteCarlo.Flat_Rnd_Number(C.MomentumDistribParameters[0],C.MomentumDistribParameters[1]);
				else if (C.MomentumDistrib == Distribution.Gaussian) TracksMomenta[i] = MonteCarlo.Gaussian_Rnd_Number(C.MomentumDistribParameters[0],C.MomentumDistribParameters[1]);
				else if (C.MomentumDistrib == Distribution.SingleValue) TracksMomenta[i] = C.MomentumDistribParameters[0];
			};
			
		}
		/// <summary>
		/// Builds initial conditions for an event.
		/// </summary>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="TrInfo">on exit, this array contains the positions and slopes of generated tracks.</param>
		/// <param name="TracksMomenta">on exit, this array contains the moment for generated tracks.</param>
		public void InitialEventConditions(VolumeGeneration.Configuration C, 
			out SySal.Tracking.MIPEmulsionTrackInfo[] TrInfo, out double[] TracksMomenta)
		{
			int ntracks = C.EvPar.OutgoingTracks;

			TrInfo = new SySal.Tracking.MIPEmulsionTrackInfo[ntracks + (C.EvPar.PrimaryTrack?1:0)];
			TracksMomenta = new double[ntracks + (C.EvPar.PrimaryTrack?1:0)];

			if(C.EvPar.PrimaryTrack)
			{
				TrInfo[0] = new SySal.Tracking.MIPEmulsionTrackInfo();
				TrInfo[0].Intercept.X = MonteCarlo.Flat_Rnd_Number(C.GeoPar.OutBoundsVolume.MinX,C.GeoPar.OutBoundsVolume.MaxX);
				TrInfo[0].Intercept.Y = MonteCarlo.Flat_Rnd_Number(C.GeoPar.OutBoundsVolume.MinY,C.GeoPar.OutBoundsVolume.MaxY);
				TrInfo[0].TopZ = TrInfo[0].Intercept.Z = C.GeoPar.OutBoundsVolume.MinZ;
				TrInfo[0].BottomZ = C.GeoPar.OutBoundsVolume.MinZ - C.GeoPar.TrackingThickness;
				if (C.XSlopesDistrib == Distribution.Flat) TrInfo[0].Slope.X = MonteCarlo.Flat_Rnd_Number(C.XSlopesDistribParameters[0],C.XSlopesDistribParameters[1]);
				else if (C.XSlopesDistrib == Distribution.Gaussian) TrInfo[0].Slope.X = MonteCarlo.Gaussian_Rnd_Number(C.XSlopesDistribParameters[0],C.XSlopesDistribParameters[1]);
				else if (C.XSlopesDistrib == Distribution.SingleValue) TrInfo[0].Slope.X = C.XSlopesDistribParameters[0];
				if (C.YSlopesDistrib == Distribution.Flat) TrInfo[0].Slope.Y = MonteCarlo.Flat_Rnd_Number(C.YSlopesDistribParameters[0],C.YSlopesDistribParameters[1]);
				else if (C.YSlopesDistrib == Distribution.Gaussian) TrInfo[0].Slope.Y = MonteCarlo.Gaussian_Rnd_Number(C.YSlopesDistribParameters[0],C.YSlopesDistribParameters[1]);
				else if (C.YSlopesDistrib == Distribution.SingleValue) TrInfo[0].Slope.Y = C.YSlopesDistribParameters[0];
				if (C.MomentumDistrib == Distribution.Flat) TracksMomenta[0] = MonteCarlo.Flat_Rnd_Number(C.MomentumDistribParameters[0],C.MomentumDistribParameters[1]);
				else if (C.MomentumDistrib == Distribution.Gaussian) TracksMomenta[0] = MonteCarlo.Gaussian_Rnd_Number(C.MomentumDistribParameters[0],C.MomentumDistribParameters[1]);
				else if (C.MomentumDistrib == Distribution.SingleValue) TracksMomenta[0] = C.MomentumDistribParameters[0];
			}

			//Conditions of OutGoing Tracks
			for(int i=(C.EvPar.PrimaryTrack?1:0); i<(C.EvPar.PrimaryTrack?1:0) + ntracks;i++)
			{
				TrInfo[i] = new SySal.Tracking.MIPEmulsionTrackInfo();
				TrInfo[i].Intercept.X = MonteCarlo.Flat_Rnd_Number(C.GeoPar.OutBoundsVolume.MinX,C.GeoPar.OutBoundsVolume.MaxX);
				TrInfo[i].Intercept.Y = MonteCarlo.Flat_Rnd_Number(C.GeoPar.OutBoundsVolume.MinY,C.GeoPar.OutBoundsVolume.MaxY);
				TrInfo[i].TopZ = TrInfo[i].Intercept.Z = C.GeoPar.OutBoundsVolume.MinZ;
				TrInfo[i].BottomZ = C.GeoPar.OutBoundsVolume.MinZ - C.GeoPar.TrackingThickness;
				if (C.XSlopesDistrib == Distribution.Flat) TrInfo[i].Slope.X = MonteCarlo.Flat_Rnd_Number(C.XSlopesDistribParameters[0],C.XSlopesDistribParameters[1]);
				else if (C.XSlopesDistrib == Distribution.Gaussian) TrInfo[i].Slope.X = MonteCarlo.Gaussian_Rnd_Number(C.XSlopesDistribParameters[0],C.XSlopesDistribParameters[1]);
				else if (C.XSlopesDistrib == Distribution.SingleValue) TrInfo[i].Slope.X = C.XSlopesDistribParameters[0];
				if (C.YSlopesDistrib == Distribution.Flat) TrInfo[i].Slope.Y = MonteCarlo.Flat_Rnd_Number(C.YSlopesDistribParameters[0],C.YSlopesDistribParameters[1]);
				else if (C.YSlopesDistrib == Distribution.Gaussian) TrInfo[i].Slope.Y = MonteCarlo.Gaussian_Rnd_Number(C.YSlopesDistribParameters[0],C.YSlopesDistribParameters[1]);
				else if (C.YSlopesDistrib == Distribution.SingleValue) TrInfo[i].Slope.Y = C.YSlopesDistribParameters[0];
				if (C.MomentumDistrib == Distribution.Flat) TracksMomenta[i] = MonteCarlo.Flat_Rnd_Number(C.MomentumDistribParameters[0],C.MomentumDistribParameters[1]);
				else if (C.MomentumDistrib == Distribution.Gaussian) TracksMomenta[i] = MonteCarlo.Gaussian_Rnd_Number(C.MomentumDistribParameters[0],C.MomentumDistribParameters[1]);
				else if (C.MomentumDistrib == Distribution.SingleValue) TracksMomenta[i] = C.MomentumDistribParameters[0];
			};			
		}

		/// <summary>
		/// Generates tracks.
		/// </summary>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="StartingTrackID">the starting Id.</param>
		/// <param name="TrInfo">the tracks to be injected.</param>
		/// <param name="TracksMomenta">the momenta of the tracks to be injected. This array must have as many elements as the TrInfo array.</param>
		/// <returns>the generated tracks.</returns>
		public Track[] GenerateTracks(VolumeGeneration.Configuration C, 
			int StartingTrackID, SySal.Tracking.MIPEmulsionTrackInfo[] TrInfo, double[] TracksMomenta)
		{
			int i, plane=0;
			double zv, zp; 
			double erry=0, errx=0, errsy=0, errsx=0, momy=0, momx=0, momsy=0, momsx=0;
			double[] y, z, x, Sx, Sy; 
			double Theta0, r1=0, r2=0;

			ArrayList trcoll = new ArrayList();
			Track[] tr;
			IntSegment s;
			Layer l;
			int[] posinlayer = new int[C.GeoPar.MostUpstreamPlane + 1];
			SySal.BasicTypes.Vector c = new SySal.BasicTypes.Vector();
			MIPEmulsionTrackInfo tk;

			int ntracks = TrInfo.Length;
			x = new double[ntracks];
			y = new double[ntracks];
			Sx = new double[ntracks];
			Sy = new double[ntracks];
			z = new double[ntracks];
			tr = new Track[ntracks];
			double cellsize = C.GeoPar.TrackingThickness + C.GeoPar.NotTrackingThickness;

			//Incoming Conditions of Tracks
			for(i=0; i<ntracks;i++)
			{
				tr[i] = new Track(i+StartingTrackID);
				//tr[i].Id=i+StartingOrdinalID;
				tr[i].Comment="Out";
				x[i] = TrInfo[i].Intercept.X;
				y[i] = TrInfo[i].Intercept.Y;
				z[i] = TrInfo[i].Intercept.Z;
				Sx[i] = TrInfo[i].Slope.X;
				Sy[i] = TrInfo[i].Slope.Y;
			};
			
			
			double dz = cellsize;

			for(i=0; i<ntracks;i++)
			{
				plane=0;
				zp = TrInfo[i].Intercept.Z + cellsize;
				while(zp<C.GeoPar.Volume.MaxZ)
				{
			
					//Simulazione Momento solo per tracce non junk
					if(i<C.HighMomentumTracks+C.EnergyLossTracks) 
					{
						double tmpMomentum;
						//Senza perdita o con perdita
						if(i<C.HighMomentumTracks) tmpMomentum = TracksMomenta[i];
						else tmpMomentum = TracksMomenta[i]*Math.Exp(-(zp-TrInfo[i].Intercept.Z)/C.KinePar.RadiationLength);
						if (tmpMomentum < C.KinePar.MinimumEnergyForLoss) break;
						Theta0 = (0.0136 / tmpMomentum) * 
							Math.Sqrt(cellsize/C.KinePar.RadiationLength) * 
							(1 + 0.038 * Math.Log(cellsize/C.KinePar.RadiationLength));

						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
						momy = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
						momsy = r2 * Theta0;

						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
						momx = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
						momsx = r2 * Theta0;
					}
					else
					{
						momy = momsy = momx = momsx = dz = 0;
					}
					//sono le posizioni vere servono per il prossimo aggiornamento
					y[i] = (y[i] + dz*Sy[i] + momy);
					x[i] = (x[i] + dz*Sx[i] + momx);
					Sy[i] = (Sy[i] + momsy);
					Sx[i] = (Sx[i] + momsx);

					//Traccia all'interno del volume
					if (y[i]<C.GeoPar.Volume.MaxY && y[i]>C.GeoPar.Volume.MinY && x[i]<C.GeoPar.Volume.MaxX && x[i]>C.GeoPar.Volume.MinX) 
					{

						//Check di Efficienza
						//se oltrepassato interviene la misura con i suoi errori
						MonteCarlo.Flat_Rnd_Number(0, 1, ref r1);
						if (r1 < C.ErrPar.TrackFindingEfficiency)
						{
							if(tr[i].Comment=="Out") tr[i].Comment="Passing";

							//Errore
							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.CoordinateErrors.Y, ref erry);
							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.CoordinateErrors.X, ref errx);

							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.SlopeErrors.Y, ref errsy);
							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.SlopeErrors.X, ref errsx);

							tk = new MIPEmulsionTrackInfo(); 
							tk.Intercept.Z = (float)zp;
							tk.Intercept.Y = (float)(y[i] + erry);
							tk.Intercept.X = (float)(x[i] + errx);
							tk.TopZ = (float)zp;
							tk.BottomZ = (float)(zp - C.GeoPar.TrackingThickness);
							tk.Slope.X = (float)(Sx[i] + errsx);
							tk.Slope.Y = (float)(Sy[i] + errsy);

							l=new Layer(C.GeoPar.MostUpstreamPlane - plane,0,C.GeoPar.MostUpstreamPlane - plane,0,c);
							s = new IntSegment(tk, l, posinlayer[C.GeoPar.MostUpstreamPlane - plane]);
							posinlayer[C.GeoPar.MostUpstreamPlane - plane]++;
							//s = new Segment(tk, C.GeoPar.MostUpstreamPlane - plane);
							//s = new Segment(tk);
							//Bisogna assegnare al segmento il  vecchio PosID (vedi riga commentata)
							tr[i].AddSegment(s);
						};
					};
					zp+=dz;
					plane++;
					//La junk track è solo su un side perciò esce
					if(i>=C.HighMomentumTracks+C.EnergyLossTracks) break;
				};
				if (tr[i].Comment=="Passing") trcoll.Add(tr[i]);				
			};
			return (Track[])(trcoll.ToArray(typeof(Track)));
		}


		/// <summary>
		/// Generates layers with tracking info.
		/// </summary>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="StartingTrackID">the starting Id.</param>
		/// <param name="TrInfo">the tracks to be injected.</param>
		/// <param name="TracksMomenta">the momenta of the tracks to be injected. This array must have as many elements as the TrInfo array.</param>
		/// <returns>the generated layers.</returns>
		public Layer[] GenerateLayers(VolumeGeneration.Configuration C, 
			int StartingTrackID, SySal.Tracking.MIPEmulsionTrackInfo[] TrInfo, double[] TracksMomenta)
		{
			int i, plane=0;
			double zv, zp; 
			double erry=0, errx=0, errsy=0, errsx=0, momy=0, momx=0, momsy=0, momsx=0;
			double[] y, z, x, Sx, Sy; 
			double Theta0, r1=0, r2=0;

			IntSegment s;
			int[] posinlayer = new int[C.GeoPar.MostUpstreamPlane + 1];
			SySal.BasicTypes.Vector c = new SySal.BasicTypes.Vector();
			MIPEmulsionTrackInfo tk;

			double cellsize = C.GeoPar.TrackingThickness + C.GeoPar.NotTrackingThickness;
			int nlayers = (int)((C.GeoPar.Volume.MaxZ-C.GeoPar.Volume.MinZ)/cellsize);
			Layer[] l = new Layer[nlayers];
			for(i=0; i<nlayers;i++) l[i] = new Layer(i,0,i,0,c);

			int ntracks = TrInfo.Length;
			x = new double[ntracks];
			y = new double[ntracks];
			z = new double[ntracks];
			Sx = new double[ntracks];
			Sy = new double[ntracks];

			//Incoming Conditions of Tracks
			for(i=0; i<ntracks;i++)
			{
				x[i] = TrInfo[i].Intercept.X;
				y[i] = TrInfo[i].Intercept.Y;
				z[i] = TrInfo[i].Intercept.Z;
				Sx[i] = TrInfo[i].Slope.X;
				Sy[i] = TrInfo[i].Slope.Y;
			};
			
			
			double dz = cellsize;

			for(i=0; i<ntracks;i++)
			{
				plane=0;
				zp = TrInfo[i].Intercept.Z + cellsize;
				int idlayer = nlayers-1 - ((int)((zp-TrInfo[i].Intercept.Z)/cellsize)-1);
				while(zp<C.GeoPar.Volume.MaxZ)
				{
			
					//Simulazione Momento solo per tracce non junk
					if(i<C.HighMomentumTracks+C.EnergyLossTracks) 
					{
						double tmpMomentum;
						//Senza perdita o con perdita
						if(i<C.HighMomentumTracks) tmpMomentum = TracksMomenta[i];
						else tmpMomentum = TracksMomenta[i]*Math.Exp(-(zp-TrInfo[i].Intercept.Z)/C.KinePar.RadiationLength);
						if (tmpMomentum < C.KinePar.MinimumEnergyForLoss) break;
						Theta0 = (0.0136 / tmpMomentum) * 
							Math.Sqrt(cellsize/C.KinePar.RadiationLength) * 
							(1 + 0.038 * Math.Log(cellsize/C.KinePar.RadiationLength));

						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
						momy = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
						momsy = r2 * Theta0;

						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r1);
						MonteCarlo.Gaussian_Rnd_Number(0, 1, ref r2);
						momx = (r1 * cellsize * Theta0 / Math.Sqrt(12)) + (r2 * cellsize * Theta0 / 2);
						momsx = r2 * Theta0;
					}
					else
					{
						momy = momsy = momx = momsx = dz = 0;
					}
					//sono le posizioni vere servono per il prossimo aggiornamento
					y[i] = (y[i] + dz*Sy[i] + momy);
					x[i] = (x[i] + dz*Sx[i] + momx);
					Sy[i] = (Sy[i] + momsy);
					Sx[i] = (Sx[i] + momsx);

					//Traccia all'interno del volume
					if (y[i]<C.GeoPar.Volume.MaxY && y[i]>C.GeoPar.Volume.MinY && x[i]<C.GeoPar.Volume.MaxX && x[i]>C.GeoPar.Volume.MinX) 
					{

						//Check di Efficienza
						//se oltrepassato interviene la misura con i suoi errori
						MonteCarlo.Flat_Rnd_Number(0, 1, ref r1);
						if (r1 < C.ErrPar.TrackFindingEfficiency)
						{
							//Errore
							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.CoordinateErrors.Y, ref erry);
							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.CoordinateErrors.X, ref errx);

							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.SlopeErrors.Y, ref errsy);
							MonteCarlo.Gaussian_Rnd_Number(0, C.ErrPar.SlopeErrors.X, ref errsx);

							tk = new MIPEmulsionTrackInfo(); 
							tk.Intercept.Z = (float)zp;
							tk.Intercept.Y = (float)(y[i] + erry);
							tk.Intercept.X = (float)(x[i] + errx);
							tk.TopZ = (float)zp;
							tk.BottomZ = (float)(zp - C.GeoPar.TrackingThickness);
							tk.Slope.X = (float)(Sx[i] + errsx);
							tk.Slope.Y = (float)(Sy[i] + errsy);

							s = new IntSegment(tk, l[idlayer], posinlayer[C.GeoPar.MostUpstreamPlane - plane]);
							posinlayer[C.GeoPar.MostUpstreamPlane - plane]++;
							//s = new Segment(tk, C.GeoPar.MostUpstreamPlane - plane);
							//s = new Segment(tk);
							//Bisogna assegnare al segmento il  vecchio PosID (vedi riga commentata)
							l[idlayer].AddSegment(s);
						};
					};
					zp+=dz;
					plane++;
					idlayer = nlayers-1 - ((int)((zp-TrInfo[i].Intercept.Z)/cellsize)-1);

					//La junk track è solo su un side perciò esce
					if(i>=C.HighMomentumTracks+C.EnergyLossTracks) break;
				};
			};
			return l;

		}

		/// <summary>
		/// Generates affine transformations to displace the layers.
		/// </summary>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="TransformationsNumber">number of transformations to generate.</param>
		/// <returns>the generated alignment parameters.</returns>
		public AlignmentData[] GenerateAffineTransformations(VolumeGeneration.Configuration C, int TransformationsNumber)
		{

			double[,] a = new double[2,2] {{1,0},{0,1}};
			double[] b = new double[3];
			double[] tmp = new double[2];
			double[] tmp2 = new double[2];

			AlignmentData[] alda = new AlignmentData[TransformationsNumber];
			int i, kSheet;
			

			for(kSheet = 0; kSheet < TransformationsNumber; kSheet++)
			{
				if(kSheet==0)
				{
					a = new double[2,2] {{1,0},{0,1}};
					b = new double[3];
					tmp = new double[2] {1,1};
					tmp2 = new double[2];
				}
				else
				{
					a[0,0] = MonteCarlo.Flat_Rnd_Number(C.AffPar.DiagMinTerm,C.AffPar.DiagMaxTerm); 
					a[1,1] = MonteCarlo.Flat_Rnd_Number(C.AffPar.DiagMinTerm,C.AffPar.DiagMaxTerm); 
					a[0,1] = MonteCarlo.Flat_Rnd_Number(C.AffPar.OutDiagMinTerm,C.AffPar.OutDiagMaxTerm); 
					a[1,0] = MonteCarlo.Flat_Rnd_Number(C.AffPar.OutDiagMinTerm,C.AffPar.OutDiagMaxTerm); 
					b[0] = MonteCarlo.Flat_Rnd_Number(C.AffPar.AlignMinShift,C.AffPar.AlignMaxShift); 
					b[1] = MonteCarlo.Flat_Rnd_Number(C.AffPar.AlignMinShift,C.AffPar.AlignMaxShift); 
					b[2] = MonteCarlo.Flat_Rnd_Number(C.AffPar.LongAlignMinShift,C.AffPar.LongAlignMaxShift); 
					//Come prima: commentare
					tmp[0] = tmp[1] = (MonteCarlo.Flat_Rnd_Number(C.AffPar.SlopeMinCoeff, C.AffPar.SlopeMaxCoeff));
					tmp2[0] = tmp2[1] = (MonteCarlo.Flat_Rnd_Number(C.AffPar.SlopeMinShift, C.AffPar.SlopeMaxShift));
				}
				alda[kSheet] = new AlignmentData(tmp, tmp2, b, a); 
			}

			return alda;
		}

		/// <summary>
		/// Applies affine transformations to a set of layers.
		/// </summary>
		/// <param name="C">the configuration for generation.</param>
		/// <param name="align">the alignment data. This array must have as many elements as the layers parameter.</param>
		/// <param name="layers">the layers to be transformed.</param>
		/// <returns>the transformed layers.</returns>
		public Layer[] ApplyAffineTransformations(VolumeGeneration.Configuration C, AlignmentData[] align, Layer[] layers)
		{

			int i, j, k, h, kSheet;
			int lnum = layers.Length;
			double tmpx, tmpy;
			SySal.BasicTypes.Vector c = new SySal.BasicTypes.Vector();
			IntSegment s;
			MIPEmulsionTrackInfo tk;
			ArrayList arrtk = new ArrayList();

			MIPEmulsionTrackInfo[] vectk;
			Layer[] l = new Layer[lnum];
			for(i=0; i<lnum;i++) l[i] = new Layer(i,0,i,0,c);

//			for(kSheet = 0; kSheet < lnum-1; kSheet++)
//			{
					
				/*
 				 * Implementazione II
				 * Ad ogni giro, ogni downstream sarà 
				 * disallineato di una trasformazione random
				 * RISPETTO AL FOGLIO PRIMA
				 */
					
			for(h=lnum-1; h> -1; h--)
			{
				for(i=0; i<layers[h].Length; i++)
				{
					tk = new MIPEmulsionTrackInfo(); 
					tk = (MIPEmulsionTrackInfo)layers[h][i].Info.Clone();
					for(k=0; k<= h; k++)
					{
						tmpx = tk.Intercept.X;
						tmpy = tk.Intercept.Y;

						
						tk.Intercept.Y = (align[k].AffineMatrixYX*tmpx + align[k].AffineMatrixYY*tmpy + align[k].TranslationY + tk.Slope.Y*align[k].TranslationZ);
						tk.Intercept.X = (align[k].AffineMatrixXX*tmpx + align[k].AffineMatrixXY*tmpy + align[k].TranslationX + tk.Slope.X*align[k].TranslationZ);
						//prima distorcere le posizioni trasv (anche in long con la slope esatta)
						//l[j][i].Info.Intercept.X = (align[j].AffineMatrixXX*tmpx + align[j].AffineMatrixXY*tmpy + align[j].TranslationX + layers[j][i].Info.Slope.X*align[j].TranslationZ);
						//l[j][i].Info.Intercept.Y = (align[j].AffineMatrixYX*tmpx + align[j].AffineMatrixYY*tmpy + align[j].TranslationY + layers[j][i].Info.Slope.Y*align[j].TranslationZ);


						//la pos long non va distorta, deve essere quella nominale assegnata al foglio di emu
						//dw[i].Info.Intercept.Z += alda[j].Traslation.Z;
						//dw[i].Info.TopZ += alda[j].Traslation.Z;
						//dw[i].Info.BottomZ += alda[j].Traslation.Z;
					};
					arrtk.Add(tk);
				};
			};
			int initidx	= 0;
				for(h=lnum-1; h> -1; h--)
				{
					vectk = new MIPEmulsionTrackInfo[layers[h].Length];
					arrtk.CopyTo(initidx, vectk, 0, layers[h].Length);
					initidx += layers[h].Length;
					for(i=0; i<layers[h].Length; i++)
					{
						tmpx = vectk[i].Slope.X;
						tmpy = vectk[i].Slope.Y;

						//poi distorcere le slopes
						//Come prima: commentare
						vectk[i].Slope.X = (align[h].DShrinkX *tmpx + align[h].SAlignDSlopeX);
						vectk[i].Slope.Y = (align[h].DShrinkY *tmpy + align[h].SAlignDSlopeY);

						//
						s = new IntSegment(vectk[i], l[h], i);
						l[h].AddSegment(s);

					};
				};
			return l;
							
		}


	}
}
