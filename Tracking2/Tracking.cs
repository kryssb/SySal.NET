using System;
using SySal;
using SySal.Imaging;
using SySal.BasicTypes;
using System.Security;
using System.Runtime.Serialization;
[assembly: AllowPartiallyTrustedCallers]

namespace SySal.Tracking
{
	/// <summary>
	/// Grain of a track
	/// </summary>
    [Serializable]
	public class Grain : ICloneable
	{
		/// <summary>
		/// 3D position of the grain.
		/// </summary>
		public Vector Position;
		/// <summary>
		/// Area in pixels.
		/// </summary>
		public uint Area;

		/// <summary>
		/// Yields a copy of this object.
		/// </summary>
		/// <returns></returns>
		public object Clone()
		{
			Grain g = new Grain();
			g.Position.X = Position.X;
			g.Position.Y = Position.Y;
			g.Position.Z = Position.Z;
			g.Area = Area;
			return g;
		}
	}

	/// <summary>
	/// Grain of a tomographic image
	/// </summary>
    [Serializable]
	public class Grain2
	{
		/// <summary>
		/// 2D position of the grain in a layer.
		/// </summary>
		public Vector2 Position;
		/// <summary>
		/// Area in pixels.
		/// </summary>
		public uint Area;
	}

	/// <summary>
	/// GrainPlane represents a plane of grains all at the same Z level
	/// </summary>
    [Serializable]
	public class GrainPlane
	{			
		/// <summary>
		/// Depth (Z) coordinate of the plane.
		/// </summary>
		public double Z;
		/// <summary>
		/// Grains in the plane.
		/// </summary>
		public Grain2 [] Grains;
	}

	/// <summary>
	/// MIPEmulsionTrackInfo stores information about a quasi-vertical track
	/// </summary>
    [Serializable]
	public class MIPEmulsionTrackInfo : ICloneable
	{
		/// <summary>
		/// The field of view where the track has been found.
		/// </summary>
		public uint Field;
		/// <summary>
		/// The number of grains in the track.
		/// </summary>
		public ushort Count;
		/// <summary>
		/// Sum of the areas of all grains.
		/// </summary>
		public uint AreaSum;
		/// <summary>
		/// 3D position of a point on the track trajectory.
		/// </summary>
		public Vector Intercept;
		/// <summary>
		/// 3D slope of the track trajectory.
		/// </summary>
		public Vector Slope;
		/// <summary>
		/// Quality of the track, usually expressed as alignment residuals or angular agreement.
		/// </summary>
		public double Sigma;
		/// <summary>
		/// Z of the top grain.
		/// </summary>
		public double TopZ;
		/// <summary>
		/// Z of the bottom grain.
		/// </summary>
		public double BottomZ;

		/// <summary>
		/// Yields a copy of this object.
		/// </summary>
		/// <returns></returns>
		public object Clone()
		{
			MIPEmulsionTrackInfo I = new MIPEmulsionTrackInfo();
			I.AreaSum = AreaSum;
			I.Count = Count;
			I.Field = Field;
			I.Intercept = Intercept;
			I.Slope = Slope;
			I.Sigma = Sigma;
			I.TopZ = TopZ;
			I.BottomZ = BottomZ;
			return I;
		}
	}

	/// <summary>
	/// MIPEmulsionTrack represents a track made of grains in the emulsion along with its global information
	/// </summary>
	public class MIPEmulsionTrack
	{
		/// <summary>
		/// Member data on which the Info property relies. Can be accessed by derived classes.
		/// </summary>
		protected MIPEmulsionTrackInfo m_Info;
		/// <summary>
		/// Retrieves global information about the track.
		/// </summary>
		public virtual MIPEmulsionTrackInfo Info { get { return (MIPEmulsionTrackInfo)m_Info.Clone(); } }
		/// <summary>
		/// Member data holding the array of the grains in the track. Can be accessed by derived classes.
		/// </summary>
		protected Grain [] m_Grains;
		/// <summary>
		/// Accesses grains in an array-like fashion.
		/// </summary>
		public virtual Grain this[int index] { get { return (Grain)(m_Grains[index].Clone());  } }
		/// <summary>
		/// Returns the number of grains in the track.
		/// </summary>
		public int Length { get { return (m_Grains == null) ? 0 : m_Grains.Length; } }
		/// <summary>
		/// Protected constructor. Prevents users from creating MIPEmulsionTrack objects without deriving the class. Is implicitly called in the constructors of derived classes.
		/// </summary>
		protected MIPEmulsionTrack() {}
		/// <summary>
		/// Protected accessor for quick access to the grain array. Can be used by derived classes.
		/// </summary>
		/// <param name="t"></param>
		/// <returns></returns>
		protected static Grain [] AccessGrains(MIPEmulsionTrack t) { return t.m_Grains; }
		/// <summary>
		/// Protected accessor for quick access to the track info. Can be used by derived classes.
		/// </summary>
		/// <param name="t"></param>
		/// <returns></returns>
		protected static MIPEmulsionTrackInfo AccessInfo(MIPEmulsionTrack t) { return t.m_Info; }
	}

	/// <summary>
	/// Tracker objects receive grains and retrieve tracks
	/// </summary>
	public interface IMIPTracker
	{
		/// <summary>
		/// Area where the tracker is supposed to operate.
		/// </summary>
		Rectangle TrackingArea
		{
			get;
			set;
		}

        /// <summary>
        /// Pixel-to-micron conversion factors.
        /// </summary>
        Vector2 Pixel2Micron
        {
            get;
            set;
        }

		/// <summary>
		/// Finds tracks as grain sequences.
		/// </summary>
        /// <param name="tomography">images as planes of grains.</param>
        /// <param name="istopside"><c>true</c> for top side, <c>false</c> for bottom side.</param>
        /// <param name="maxtracks">maximum number of tracks to produce.</param>
        /// <param name="enablepresetslope">if <c>true</c>, enables using a preset track slope, with limited slope acceptance.</param>
        /// <param name="presetslope">preselected slope of tracks to be found.</param>
        /// <param name="presetslopeacc">slope acceptances for preselected track slopes.</param>
		Grain [][] FindTracks(GrainPlane [] tomography, bool istopside, int maxtracks, bool enablepresetslope, Vector2 presetslope, Vector2 presetslopeacc);
	}

	/// <summary>
	/// Distortion correction.
	/// </summary>
	[Serializable]
	public abstract class DistortionCorrection
	{
		/// <summary>
		/// Order of the correction.
		/// </summary>
		public readonly uint Order;
		/// <summary>
		/// Coefficients of the correction.
		/// </summary>
		public readonly SySal.BasicTypes.Vector2 [] Coefficients;		
		/// <summary>
		/// Shrinkage factor.
		/// </summary>
		public readonly double Shrinkage;
		/// <summary>
		/// Z of the emulsion surface in contact with the plastic base of the plate.
		/// </summary>
		public readonly double ZBaseSurface;
		/// <summary>
		/// Z of the outer surface of the emulsion layer.
		/// </summary>
		public readonly double ZExternalSurface;

		/// <summary>
		/// Corrects the grain positions using known distortion information.
		/// </summary>
		/// <param name="grains"></param>
		public abstract void Correct(Grain [] grains);

		/// <summary>
		/// Protected constructor. Can only be called by derived classes. Users must derive a class from DistortionCorrection in order to use it.
		/// </summary>
		/// <param name="order"></param>
		/// <param name="coefficients"></param>
		/// <param name="shrinkage"></param>
		/// <param name="zbasesurface"></param>
		/// <param name="zexternalsurface"></param>
		protected DistortionCorrection(uint order, SySal.BasicTypes.Vector2 [] coefficients, double shrinkage, double zbasesurface, double zexternalsurface)
		{
			Order = order;
			Coefficients = coefficients;
			Shrinkage = shrinkage;
			ZBaseSurface = zbasesurface;
			ZExternalSurface = zexternalsurface;
		}
	}


	/// <summary>
	/// Performs post-tracking processing such as track fitting, distortion correction, unshrinking, etc.
	/// </summary>
	public interface IMIPPostProcessor
	{
		/// <summary>
		/// Retrieves the DistortionCorrection the PostProcessor used to correct the grains.
		/// </summary>
		DistortionCorrection DistortionInfo
		{
			get;
		}

		/// <summary>
		/// Processes the grain sequences supplied, optionally corrects them, and computes the geometrical parameters of each track.
		/// </summary>
		MIPEmulsionTrack [] Process(Grain [][] grainsequences, double zbasesurf, double zextsurf, double shrinkage, bool correctgrains);
	}
}
