using System;
using System.Collections;
using SySal;
using SySal.Tracking;
using NumericalTools;
using System.Runtime.Serialization;

namespace SySal.TotalScan
{

	/// <summary>
	/// Topological relationships.
	/// </summary>
	public enum Relationship : int {Mother=2, Daughter=1, Unknown=0}
	/// <summary>
	/// Topological intersection types.
	/// </summary>
	public enum IntersectionType : int {Unknown=0, V=1, Kink=2, Y=3, Lambda=4, X=5}
	/// <summary>
	/// Topological symmetries.
	/// </summary>
	public enum IntersectionSymmetry : int {Unknown=0, Element1Upstream = 1, Element2Upstream=2, Symmetric = 3}
	/// <summary>
	/// Computation results.
	/// </summary>
	public enum ComputationResult : int {OK =0, InsufficientMeasurements = 1, Incoherent =2}

	#region Kinematics Information
	public class KineInfo
	{
		public KineInfo()
		{

		}

		protected double m_RMS;

		public double RMS
		{
			get
			{
				return m_RMS;
			}

			set
			{
				m_RMS = value;	
			}

		}

		protected int m_NMeas;

		public int NMeas
		{
			get
			{
				return m_NMeas;
			}

			set
			{
				m_NMeas = value;	
			}
		}



		protected double m_LowerMomentum;

		public double LowerMomentum
		{
			get
			{
				return m_LowerMomentum;
			}

			set
			{
				m_LowerMomentum = value;	
			}

		}

		protected double m_UpperMomentum;

		public double UpperMomentum
		{
			get
			{
				return m_UpperMomentum;
			}

			set
			{
				m_UpperMomentum = value;	
			}

		}

		protected double m_Momentum;

		public double Momentum
		{
			get
			{
				return m_Momentum;
			}

			set
			{
				m_Momentum = value;	
			}

		}

	}
	#endregion

	#region Index

	/// <summary>
	/// Abstract class for object indexing.
	/// </summary>
	public abstract class Index : ICloneable
	{
		/// <summary>
		/// Creates an index from a BinaryReader. The index type must be known in advance.
		/// </summary>
		public delegate Index dCreateFromReader(System.IO.BinaryReader r);
		/// <summary>
		/// Saves an index to a BinaryWriter. The index type must be known in advance.
		/// </summary>
		public delegate void dSaveToWriter(Index i, System.IO.BinaryWriter w);

		/// <summary>
		/// Index class factory. Stores information to dynamically identify indices on the basis of their class signature.
		/// </summary>
		public class IndexFactory : ICloneable
		{
			/// <summary>
			/// Index type signature.
			/// </summary>
			public int Signature;
			/// <summary>
			/// Index size.
			/// </summary>
			public int Size;
			/// <summary>
			/// Reads an index of the class that has the signature stored in this index factory.
			/// </summary>
			public dCreateFromReader Reader;
			/// <summary>
			/// Saves an index of the class that has the signature stored in this index factory.
			/// </summary>
			public dSaveToWriter Writer;

			/// <summary>
			/// Public constructor.
			/// </summary>
			/// <param name="sgn">index type signature.</param>
			/// <param name="sz">index type size (when serialized).</param>
			/// <param name="rdr">index reader.</param>
			/// <param name="wrt">index writer.</param>
			public IndexFactory(int sgn, int sz, dCreateFromReader rdr, dSaveToWriter wrt)
			{
				Signature = sgn;
				Size = sz;
				Reader = rdr;
				Writer = wrt;
			}
			#region ICloneable Members
			/// <summary>
			/// Clones an IndexFactory.
			/// </summary>
			/// <returns>a clone of the IndexFactory object.</returns>
			public object Clone()
			{				
				return new IndexFactory(this.Signature, this.Size, this.Reader, this.Writer);
			}

			#endregion
		}

		static IndexFactory [] Factories = new IndexFactory[0];

		/// <summary>
		/// Registers a new IndexFactory. This function should be called before trying to read/write indices of each type. Normally, index classes should self-register automatically.
		/// </summary>
		/// <param name="f">the index factory to be registered.</param>
		public static void RegisterFactory(IndexFactory f)
		{
			int i;
			for (i = 0; i < Factories.Length; i++)			
				if (Factories[i].Signature == f.Signature) return;
			IndexFactory [] newfc = new IndexFactory[Factories.Length + 1];
			for (i = 0; i < Factories.Length; i++)			
				newfc[i] = Factories[i];
			newfc[i] = (IndexFactory)(f.Clone());
			Factories = newfc;
		}

		/// <summary>
		/// Searches for a registered IndexFactory with the specified signature.
		/// </summary>
		/// <param name="signature">IndexFactory signature to be searched for.</param>
		/// <returns>the IndexFactory with the specified signature or null if no registered IndexFactory is found.</returns>
		public static IndexFactory GetFactory(int signature)
		{			
			int i;
			for (i = 0; i < Factories.Length; i++)			
				if (Factories[i].Signature == signature)				
					return (IndexFactory)(Factories[i].Clone());
			return null;
		}

		/// <summary>
		/// Abstract property that returns the IndexFactory for this index.
		/// </summary>
		public abstract IndexFactory Factory { get; }

		/// <summary>
		/// Abstract method to save an Index to a BinaryWriter.
		/// </summary>
		/// <param name="b"></param>
		public abstract void Write(System.IO.BinaryWriter b);

		#region ICloneable Members

		/// <summary>
		/// Abstract method to clone an Index object.
		/// </summary>
		/// <returns>a clone of the Index.</returns>
		public abstract object Clone();

		#endregion
	}

	/// <summary>
	/// Null Index class for objects that have no index information.
	/// </summary>
	public class NullIndex : Index
	{
		/// <summary>
		/// The Index type signature.
		/// </summary>
		public static readonly int Signature = 0;

		/// <summary>
		/// Registers the Index factory for NullIndex.
		/// </summary>
		public static void RegisterFactory()
		{
			Index.RegisterFactory(new IndexFactory(Signature, 0, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)));			
		}

		/// <summary>
		/// Constructs a null index.
		/// </summary>
		public NullIndex()
		{
		}

		/// <summary>
		/// Saves a null index (does nothing).
		/// </summary>
		/// <param name="b"></param>
		public override void Write(System.IO.BinaryWriter b)
		{
		}

		/// <summary>
		/// Shows a string describing a null index.
		/// </summary>
		/// <returns>"NULL"</returns>
		public override string ToString()
		{
			return "NULL";
		}

		/// <summary>
		/// Saves a null index (does nothing).
		/// </summary>
		/// <param name="i">the index to be saved.</param>
		/// <param name="w">the BinaryWriter to be used for saving.</param>
		public static void SaveToWriter(Index i, System.IO.BinaryWriter w)
		{
		}

		/// <summary>
		/// Creates a null index from a BinaryReader. Actually, no byte is read.
		/// </summary>
		/// <param name="r">the BinaryReader to be used.</param>
		/// <returns>a null index.</returns>
		public static Index CreateFromReader(System.IO.BinaryReader r)
		{
			return new NullIndex();
		}

		/// <summary>
		/// Returns an IndexFactory for a NullIndex.
		/// </summary>
		public override IndexFactory Factory { get { return new IndexFactory(Signature, 0, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)); } }

		#region ICloneable Members

		/// <summary>
		/// Clones a NullIndex.
		/// </summary>
		/// <returns>a copy of the NullIndex.</returns>
		public override object Clone()
		{
			return new NullIndex();
		}

		#endregion

		public override bool Equals(object obj)
		{
			return obj.GetType() == this.GetType();
		}

		public override int GetHashCode()
		{
			return 0;
		}

	}
	#endregion

	#region Segment

	/// <summary>
	/// Index class for Segments related to base tracks. The Linked zone information is not specified. This class is automatically chosen for indices of old TSR files.
	/// </summary>
	public class BaseTrackIndex : Index
	{
		/// <summary>
		/// The signature of the BaseTrackIndex class.
		/// </summary>
		public static readonly int Signature = 1;

		/// <summary>
		/// Registers the Index factory for BaseTrackIndex.
		/// </summary>
		public static void RegisterFactory()
		{
			Index.RegisterFactory(new IndexFactory(Signature, 4, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)));			
		}

		/// <summary>
		/// Member data on which the Id property relies.
		/// </summary>
		protected int m_Id;

		/// <summary>
		/// BaseTrack Id of the Segment.
		/// </summary>
		public int Id { get { return m_Id; } }

		/// <summary>
		/// Constructs a BaseTrackIndex from the base track Id.
		/// </summary>
		/// <param name="id"></param>
		public BaseTrackIndex(int id)
		{
			m_Id = id;
		}

		/// <summary>
		/// Saves a BaseTrackIndex to a BinaryWriter.
		/// </summary>
		/// <param name="b">the BinaryWriter to be used for saving.</param>
		public override void Write(System.IO.BinaryWriter b)
		{
			b.Write(m_Id);
		}

		/// <summary>
		/// Converts the BaseTrackIndex to a text form.
		/// </summary>
		/// <returns>the Id in text form.</returns>
		public override string ToString()
		{
			return m_Id.ToString();
		}

		/// <summary>
		/// Saves a BaseTrackIndex to a BinaryWriter.
		/// </summary>
		/// <param name="i">the index to be saved. Must be a BaseTrackIndex.</param>
		/// <param name="w">the BinaryWriter to be used for saving.</param>
		public static void SaveToWriter(Index i, System.IO.BinaryWriter w)
		{
			((BaseTrackIndex)i).Write(w);
		}

		/// <summary>
		/// Reads a BaseTrackIndex from a BinaryReader.
		/// </summary>
		/// <param name="r">the BinaryReader to read from.</param>
		/// <returns>the BaseTrackIndex read from the stream.</returns>
		public static Index CreateFromReader(System.IO.BinaryReader r)
		{
			return new BaseTrackIndex(r.ReadInt32());
		}

		/// <summary>
		/// Returns the IndexFactory for a BaseTrackIndex.
		/// </summary>
		public override IndexFactory Factory { get { return new IndexFactory(Signature, 4, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)); } }

		#region ICloneable Members

		/// <summary>
		/// Clones this BaseTrackIndex.
		/// </summary>
		/// <returns>a clone of the BaseTrackIndex.</returns>
		public override object Clone()
		{
			return new BaseTrackIndex(m_Id);
		}

		#endregion

		public override bool Equals(object obj)
		{
			if (obj.GetType() != this.GetType()) return false;
			BaseTrackIndex x = (BaseTrackIndex)obj;
			return x.Id == this.m_Id;
		}

		public override int GetHashCode()
		{
			return m_Id;
		}
	}

	/// <summary>
	/// Index class for MIP Microtracks. Stores information about side and index within the side.
	/// </summary>
	public class MIPMicroTrackIndex : Index
	{
		/// <summary>
		/// The signature of the MIPMicroTrackIndex class.
		/// </summary>
		public static readonly int Signature = 2;

		/// <summary>
		/// Registers the Index factory for MIPMicroTrackIndex.
		/// </summary>
		public static void RegisterFactory()
		{
			Index.RegisterFactory(new IndexFactory(Signature, 6, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)));
		}

		/// <summary>
		/// Member data on which the Side property relies.
		/// </summary>
		protected short m_Side;
		/// <summary>
		/// The side of the MIP Microtrack.
		/// </summary>
		public short Side { get { return m_Side; } }
		/// <summary>
		/// Member data on which the Id property relies.
		/// </summary>
		protected int m_Id;
		/// <summary>
		/// The Id of the microtrack in its side.
		/// </summary>
		public int Id { get { return m_Id; } }

		/// <summary>
		/// Constructs a MIPMicroTrackIndex.
		/// </summary>
		/// <param name="side">the side of the microtrack.</param>
		/// <param name="id">the index of the microtrack.</param>
		public MIPMicroTrackIndex(short side, int id)
		{
			m_Side = side;
			m_Id = id;
		}

		/// <summary>
		/// Saves a MIPMicroTrackIndex to a BinaryWriter.
		/// </summary>
		/// <param name="b">the BinaryWriter to be used for saving.</param>
		public override void Write(System.IO.BinaryWriter b)
		{
			b.Write(m_Side);
			b.Write(m_Id);
		}

		/// <summary>
		/// Converts a MIPMicroTrackIndex to text form.
		/// </summary>
		/// <returns>a string of the form "side\id".</returns>
		public override string ToString()
		{
			return m_Side.ToString() + @"\" + m_Id.ToString();
		}

		/// <summary>
		/// Saves a MIPMicroTrackIndex to a BinaryWriter.
		/// </summary>
		/// <param name="i">the index to be saved.</param>
		/// <param name="w">the BinaryWriter to be used for writing.</param>
		public static void SaveToWriter(Index i, System.IO.BinaryWriter w)
		{
			((MIPMicroTrackIndex)i).Write(w);
		}

		/// <summary>
		/// Reads a MIPMicroTrackIndex from a BinaryReader.
		/// </summary>
		/// <param name="r">the BinaryReader to read the MIPMicroTrackIndex from.</param>
		/// <returns>the index read from the BinaryReader.</returns>
		public static Index CreateFromReader(System.IO.BinaryReader r)
		{
			return new MIPMicroTrackIndex(r.ReadInt16(), r.ReadInt32());
		}

		/// <summary>
		/// Returns the IndexFactory for the MIPMicroTrackIndex class.
		/// </summary>
		public override IndexFactory Factory { get { return new IndexFactory(Signature, 6, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)); } }

		#region ICloneable Members

		/// <summary>
		/// Clones a MIPMicroTrackIndex.
		/// </summary>
		/// <returns>the cloned object.</returns>
		public override object Clone()
		{
			return new MIPMicroTrackIndex(m_Side, m_Id);
		}

		#endregion

		public override bool Equals(object obj)
		{
			if (obj.GetType() != this.GetType()) return false;
			MIPMicroTrackIndex x = (MIPMicroTrackIndex)obj;
			return x.Id == this.m_Id && x.Side == this.m_Side;
		}

		public override int GetHashCode()
		{
			return m_Id + m_Side;
		}

	}

	/// <summary>
	/// Holds information about a base track in a TotalScan volume.
	/// </summary>
	public class Segment
	{
		/// <summary>
		/// Protected constructor. Prevents users from creating Segments without providing consistent information. Is implicitly called in derived classes.
		/// </summary>
		protected Segment()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		private static bool m_RegisteredIndexFactories = RegisterIndexFactories();

		private static bool RegisterIndexFactories()
		{
			NullIndex.RegisterFactory();
			BaseTrackIndex.RegisterFactory();
			MIPMicroTrackIndex.RegisterFactory();
			return true;
		}

		/// <summary>
		/// Builds a new segment.
		/// </summary>
		/// <param name="tk">the track to be copied.</param>
		/// <param name="ix">index of the segment.</param>
		public Segment(SySal.Scanning.MIPBaseTrack tk, Index ix)
		{
			//
			// TODO: Add constructor logic here
			//
			m_Info = tk.Info;
			m_LayerOwner = null;
			m_TrackOwner = null;
			m_PosInLayer = -1;
			m_PosInTrack = -1;
			m_Index = ix;
		}

		/// <summary>
		/// Builds a new segment.
		/// </summary>
		/// <param name="tk">the track to be copied.</param>
		/// <param name="ix">index of the segment.</param>
		public Segment(MIPEmulsionTrackInfo tk, Index ix)
		{
			//
			// TODO: Add constructor logic here
			//
			m_Info = tk;
			m_LayerOwner = null;
			m_TrackOwner = null;
			m_PosInLayer = -1;
			m_PosInTrack = -1;
			m_Index = ix;
		}

		/// <summary>
		/// Member data on which the Index property relies. Can be accessed by derived classes.
		/// </summary>
		protected Index m_Index;

		/// <summary>
		/// Retrieves the index of this segment with respect to the original data structure (e.g. microtrack, basetrack, etc.).
		/// </summary>
		public Index Index
		{
			get
			{
				return (Index)(m_Index.Clone());
			}
		}

		/// <summary>
		/// Member data on which the Info property relies. Can be accessed by derived classes.
		/// </summary>
		protected MIPEmulsionTrackInfo m_Info;
		/// <summary>
		/// Global geometrical information.
		/// </summary>
		public virtual MIPEmulsionTrackInfo Info
		{
			get
			{
				return (MIPEmulsionTrackInfo)m_Info.Clone();
			}

			set
			{
				if (value == null) throw new Exception("....");
				m_Info = value;
				if (m_TrackOwner != null) m_TrackOwner.NotifyChanged();
				if (m_LayerOwner != null) m_LayerOwner.NotifyChanged();
			}
		}

        /// <summary>
        /// Global geometrical information, represented in the frame before alignment.
        /// </summary>
        public virtual MIPEmulsionTrackInfo OriginalInfo
        {
            get
            {
                MIPEmulsionTrackInfo info = Info;
                info.Slope = m_LayerOwner.ToOriginalSlope(info.Slope);
                info.Intercept = m_LayerOwner.ToOriginalPoint(info.Intercept);
                SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
                v.X = v.Y = 0.0;
                v.Z = info.TopZ;
                v = m_LayerOwner.ToOriginalPoint(v);
                info.TopZ = v.Z;
                v.Z = info.BottomZ;
                v = m_LayerOwner.ToOriginalPoint(v);
                info.BottomZ = v.Z;
                return info;
            }
        }

        /// <summary>
        /// Yields text information about this segment.
        /// </summary>
        /// <returns>text information.</returns>
        public override string ToString()
        {
            SySal.Tracking.MIPEmulsionTrackInfo info = this.Info;
            string text = "SEGMENT INFO";
            text += "\r\nLayer: " + this.LayerOwner.Id;
            text += "\r\nSheet: " + this.LayerOwner.SheetId;
            text += "\r\nID: " + this.PosInLayer;
            text += "\r\nTrack: " + ((this.TrackOwner == null) ? -1 : this.TrackOwner.Id);
            text += "\r\nGrains: " + ((short)info.Count);
            text += "\r\nAreaSum: " + ((int)info.AreaSum);
            text += "\r\nIX/IY/IZ: " + info.Intercept.X + " / " + info.Intercept.Y + " / " + info.Intercept.Z;
            text += "\r\nSX/SY: " + info.Slope.X + " / " + info.Slope.Y;
            text += "\r\nSigma: " + info.Sigma;
            info = this.OriginalInfo;
            text += "\r\nOIX/OIY/OIZ: " + info.Intercept.X + " / " + info.Intercept.Y + " / " + info.Intercept.Z;
            text += "\r\nOSX/OSY: " + info.Slope.X + " / " + info.Slope.Y;
            if (m_Index != null) text += "\r\nIndex: " + m_Index.ToString();
            return text;
        }

		/// <summary>
		/// Member data on which the TrackOwner property relies. Can be accessed by derived classes.
		/// </summary>
		protected Track m_TrackOwner;
		/// <summary>
		/// The track this segment belongs to.
		/// </summary>
		public Track TrackOwner { get { return m_TrackOwner; } }

		/// <summary>
		/// Member data on which the PosInTrack property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_PosInTrack;
		/// <summary>
		/// The position of the segment in the track.
		/// </summary>
		public int PosInTrack {	get { return m_PosInTrack; } }
		internal void SetTrackOwner(Track owner, int posintrack) { m_TrackOwner = owner; m_PosInTrack = posintrack; } 
		
		protected static void SetTrackOwner(Segment s, Track t, int posintrack) { s.SetTrackOwner(t, posintrack); }

		/// <summary>
		/// Member data on which the LayerOwner property relies. Can be accessed by derived classes.
		/// </summary>
		protected Layer m_LayerOwner;
		/// <summary>
		/// The layer this segment belongs to.
		/// </summary>
		public Layer LayerOwner { get {	return m_LayerOwner; } }

		/// <summary>
		/// Member data on which the PosInLayer property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_PosInLayer;
		/// <summary>
		/// The position of the segment in the layer.
		/// </summary>
		public int PosInLayer {	get { return m_PosInLayer; } }
		internal void SetLayerOwner(Layer owner, int posinlayer) { m_LayerOwner = owner; m_PosInLayer = posinlayer; } 

		internal void SetIndex(Index ix)
		{
			m_Index = ix;
		}
	}

	/// <summary>
	/// A list of segment. Track and Layer are both derived from this base class.
	/// </summary>
	public class SegmentList
	{
		/// <summary>
		/// Member data holding the list of segments. Can be accessed by derived classes.
		/// </summary>
		protected Segment[] Segments = new Segment[0];

		/// <summary>
		/// Accesses the index-th segment.
		/// </summary>
		public Segment this[int index]
		{
			get { return Segments[index];  }
			set { ReplaceSegment(index, value); NotifyChanged(); }
		}

		/// <summary>
		/// Number of segments in the list.
		/// </summary>
		public int Length
		{
			get { return Segments.Length; }
		}

		/// <summary>
		/// Replaces a segment in the list.
		/// </summary>
		/// <param name="index"></param>
		/// <param name="s"></param>
		protected virtual void ReplaceSegment(int index, Segment s) {}

		/// <summary>
		/// Notifies derived classes that the list has changed.
		/// </summary>
		public virtual void NotifyChanged() {}
	}

	#endregion

	#region Attributes

	/// <summary>
	/// Index class for attributes identified only with a name.
	/// </summary>
	public class NamedAttributeIndex : Index
	{
		/// <summary>
		/// The signature of the NamedAttributeIndex class.
		/// </summary>
		public static readonly int Signature = 51;

		const int NameLen = 32;

		/// <summary>
		/// Registers the Index factory for a NamedAttributeIndex
		/// </summary>
		public static void RegisterFactory()
		{
			Index.RegisterFactory(new IndexFactory(Signature, NameLen, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)));
		}

		/// <summary>
		/// Member data on which the Name property relies.
		/// </summary>
		protected string m_Name;

		/// <summary>
		/// Name of the attribute.
		/// </summary>
		public string Name { get { return (string)(m_Name.Clone()); } }

		/// <summary>
		/// Constructs a NamedAttributeIndex from an attribute name (max 32 chars).
		/// </summary>
		/// <param name="name">the name to be assigned to the attribute</param>
		public NamedAttributeIndex(string name)
		{
			m_Name = name.PadRight(NameLen, ' ').Substring(0, NameLen).Trim();
			if (m_Name.Length == 0) throw new Exception("Name must be a non-null string. Spaces are trimmed, but are preserved between words.");
		}

		/// <summary>
		/// Saves a NamedAttributeIndex to a BinaryWriter.
		/// </summary>
		/// <param name="b">the BinaryWriter to be used for saving.</param>
		public override void Write(System.IO.BinaryWriter b)
		{
			char [] chars = m_Name.PadRight(NameLen, ' ').ToCharArray(0, NameLen);
			b.Write(chars);
		}

		/// <summary>
		/// Converts the NamedAttributeIndex to a text form.
		/// </summary>
		/// <returns>the Name in text form.</returns>
		public override string ToString()
		{
			return (string)(m_Name.Clone());
		}

		/// <summary>
		/// Saves a NamedAttributeIndex to a BinaryWriter.
		/// </summary>
		/// <param name="i">the index to be saved. Must be a NamedAttributeIndex.</param>
		/// <param name="w">the BinaryWriter to be used for saving.</param>
		public static void SaveToWriter(Index i, System.IO.BinaryWriter w)
		{
			((NamedAttributeIndex)i).Write(w);
		}

		/// <summary>
		/// Reads a NamedAttributeIndex from a BinaryReader.
		/// </summary>
		/// <param name="r">the BinaryReader to read from.</param>
		/// <returns>the NamedAttributeIndex read from the stream.</returns>
		public static Index CreateFromReader(System.IO.BinaryReader r)
		{
			return new NamedAttributeIndex(new string(r.ReadChars(NameLen)).Trim());
		}

		/// <summary>
		/// Returns the IndexFactory for a NamedAttributeIndex.
		/// </summary>
		public override IndexFactory Factory { get { return new IndexFactory(Signature, NameLen, new Index.dCreateFromReader(CreateFromReader), new Index.dSaveToWriter(SaveToWriter)); } }

		#region ICloneable Members

		/// <summary>
		/// Clones this NamedAttributeIndex.
		/// </summary>
		/// <returns>a clone of the NamedAttributeIndex.</returns>
		public override object Clone()
		{
			return new NamedAttributeIndex(m_Name);
		}

		#endregion

		public override bool Equals(object obj)
		{
			if (obj.GetType() != this.GetType()) return false;
			NamedAttributeIndex x = (NamedAttributeIndex)obj;
			return (String.Compare(x.Name, this.m_Name, true) == 0);
		}

		public override int GetHashCode()
		{
			return m_Name.GetHashCode();
		}

	}
	

	/// <summary>
	/// An attribute of a complex structure like a Track or Vertex.
	/// </summary>
	public class Attribute : ICloneable
	{
		/// <summary>
		/// Index of the attribute.
		/// </summary>
		public Index Index;
		/// <summary>
		/// Value of the attribute.
		/// </summary>
		public double Value;
		/// <summary>
		/// Public constructor that builds an attribute with index and value.
		/// </summary>
		/// <param name="ix">the index of the attribute.</param>
		/// <param name="v">the value of the attribute.</param>
		public Attribute(Index ix, double v) { Index = ix; Value = v; }

		#region ICloneable Members

		public object Clone()
		{
			return new Attribute((SySal.TotalScan.Index)(Index.Clone()), Value);
		}

		#endregion
	}

	/// <summary>
	/// A list of attributes.
	/// </summary>
	public interface IAttributeList
	{
		/// <summary>
		/// Sets an attribute.
		/// </summary>
		/// <param name="attributeindex">the attribute index.</param>
		/// <param name="attributevalue">the attribute value.</param>
		void SetAttribute(Index attributeindex, double attributevalue);
		/// <summary>
		/// Removes an attribute.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be removed.</param>
		void RemoveAttribute(Index attributeindex);
		/// <summary>
		/// Gets an attribute.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be read.</param>
		/// <returns>the value of the attribute.</returns>
		double GetAttribute(Index attributeindex);
		/// <summary>
		/// Lists all attributes.
		/// </summary>
		/// <returns>the list of the attributes.</returns>
		Attribute [] ListAttributes();
	}
	#endregion

	#region Track

	/// <summary>
	/// A long-range TotalScan track, made of connected segments in several emulsion plates.
	/// Derived from SegmentList, so its segments can be accessed in array-like fashion.
	/// </summary>
	public class Track : SegmentList, IAttributeList
	{
        /// <summary>
        /// Designates different ways to compute the extrapolation of a track.
        /// </summary>
        public enum ExtrapolationMode
        {
            /// <summary>
            /// Uses the last segments to fit the track.
            /// </summary>
            SegmentFit, 
            /// <summary>
            /// Uses the last base track.
            /// </summary>
            EndBaseTrack
        }

        /// <summary>
        /// Property backer for <c>TrackExtrapolationMode</c>.
        /// </summary>
        protected static ExtrapolationMode s_TrackExtrapolationMode;

        /// <summary>
        /// Gets/sets the way tracks are extrapolated to their ends.
        /// </summary>
        public static ExtrapolationMode TrackExtrapolationMode
        {
            get { return s_TrackExtrapolationMode; }
            set { s_TrackExtrapolationMode = value; }
        }

		/// <summary>
		/// Protected member that allows Id changing in derived classes.
		/// </summary>
		protected static void SetId(Track t, int newid)
		{
			t.m_Id = newid;
		}

		/// <summary>
		/// Protected constructor. Prevents users from creating Tracks without providing consistent information. Is implicitly called in derived classes.
		/// </summary>
		public Track()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		/// <summary>
		/// Builds a new track with the specified identifying number.
		/// </summary>
		/// <param name="id"></param>
		public Track(int id)
		{
			//
			// TODO: Add constructor logic here
			//
			m_Id = id;
		}

		/// <summary>
		/// Member data that signals whether the UpstreamLayer property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_Layer_Updated= false;
		/// <summary>
		/// Member data on which the UpstreamLayer property relies. Can be accessed by derived classes.
		/// </summary>
		protected Layer m_Upstream_Layer;
		/// <summary>
		/// The most upstream layer where the track has been seen.
		/// </summary>
		public Layer UpstreamLayer
		{
			get
			{
				if (!m_Upstream_Layer_Updated)
				{
					m_Upstream_Layer = (Segments.Length > 0) ? Segments[Segments.Length - 1].LayerOwner : null;
					m_Upstream_Layer_Updated= true;
				};
				return m_Upstream_Layer;
			}
		}

		/// <summary>
		/// Member data that signals whether the UpstreamLayerId property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_LayerId_Updated= false;
		/// <summary>
		/// Member data on which the UpstreamLayerId property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_Upstream_LayerId;
		/// <summary>
		/// The id of the most upstream layer where the track has been seen.
		/// </summary>
		public int UpstreamLayerId
		{
			get
			{
				if (!m_Upstream_LayerId_Updated)
				{
					m_Upstream_LayerId = (Segments.Length > 0) ? Segments[Segments.Length - 1].LayerOwner.Id : -1;
					m_Upstream_LayerId_Updated= true;
				};
				return m_Upstream_LayerId;
			}
		}

		/// <summary>
		/// Member data that signals whether the DownstreamLayer property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_Layer_Updated= false;
		/// <summary>
		/// Member data on which the DownstreamLayer property relies. Can be accessed by derived classes.
		/// </summary>
		protected Layer m_Downstream_Layer;
		/// <summary>
		/// The id of the most downstream layer where the track has been seen.
		/// </summary>
		public Layer DownstreamLayer
		{
			get
			{
				if (!m_Downstream_Layer_Updated)
				{
					m_Downstream_Layer = (Segments.Length > 0) ? Segments[0].LayerOwner : null;
					m_Downstream_Layer_Updated= true;
				};
				return m_Downstream_Layer;
			}
		}

		/// <summary>
		/// Member data that signals whether the DownstreamLayerId property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_LayerId_Updated= false;
		/// <summary>
		/// Member data on which the UpstreamLayerId property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_Downstream_LayerId;
		/// <summary>
		/// The id of the most downstream layer where the track has been seen.
		/// </summary>
		public int DownstreamLayerId
		{
			get
			{
				if (!m_Downstream_LayerId_Updated)
				{
					m_Downstream_LayerId = (Segments.Length > 0) ? Segments[0].LayerOwner.Id : -1;
					m_Downstream_LayerId_Updated= true;
				};
				return m_Downstream_LayerId;
			}
		}

		/// <summary>
		/// Member data on which the FittingSegments property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_FittingSegments=3;
		/// <summary>
		/// Gets / sets the number of segments used to compute the downstream / upstream track parameters.
		/// </summary>
		public int FittingSegments
		{
			get { return m_FittingSegments; }
			set	
			{
/*				if (value<1) 
					m_FittingSegments=1;
				else if(value > Length)
					m_FittingSegments=Length;
				else
 */
					m_FittingSegments = value;	
				NotifyChanged();
			}
		}

		/// <summary>
		/// Member data on which the Segment Id property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_Id;
		/// <summary>
		/// The id of the track, usually a sequential number in the volume.
		/// </summary>
		public int Id { get { return m_Id; } }
		internal void SetId(int id) { m_Id = id; }

		/// <summary>
		/// A user comment string accompanying the track.
		/// </summary>
		public string Comment = "";

		internal void SetSlopeAndPos(double dwslopex, double dwslopey, double dwposx, double dwposy, double upslopex, double upslopey, double upposx, double upposy)
		{
			m_Upstream_SlopeX = upslopex;
			m_Upstream_SlopeY = upslopey;
			m_Upstream_PosX = upposx;
			m_Upstream_PosY = upposy;
			m_Downstream_SlopeX = dwslopex;
			m_Downstream_SlopeY = dwslopey;
			m_Downstream_PosX = dwposx;
			m_Downstream_PosY = dwposy;
			m_Upstream_SlopeX_Updated = m_Upstream_SlopeY_Updated = m_Upstream_PosX_Updated = m_Upstream_PosY_Updated = 
				m_Downstream_SlopeX_Updated = m_Downstream_SlopeY_Updated = m_Downstream_PosX_Updated = m_Downstream_PosY_Updated = true;
		}

		/// <summary>
		/// Member data that signals whether the Upstream_SlopeY property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_SlopeY_Updated= false;
		/// <summary>
		/// Member data on which the Upstream_SlopeY property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Upstream_SlopeY;
		/// <summary>
		/// Upstream Y slope of the track.
		/// </summary>
		public double Upstream_SlopeY
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Upstream_SlopeY_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
            					if (m_FittingSegments>Segments.Length) m_FittingSegments = Segments.Length;
			            		Compute_Local_YCoord(Segments.Length-m_FittingSegments,out m_Upstream_SlopeY,out m_Upstream_PosY);
            					m_Upstream_SlopeY_Updated = true;
			            		m_Upstream_PosY_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeUpstreamBaseTrackExtrapolation(out p, out s);
                                m_Upstream_PosX = p.X;
                                m_Upstream_PosY = p.Y;
                                m_Upstream_SlopeX = s.X;
                                m_Upstream_SlopeY = s.Y;
            					m_Upstream_SlopeX_Updated = true;
            					m_Upstream_SlopeY_Updated = true;
			            		m_Upstream_PosX_Updated = true;
			            		m_Upstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
				};
				return m_Upstream_SlopeY;
			}
		}

		/// <summary>
		/// Member data that signals whether the Upstream_PosY property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_PosY_Updated= false;
		/// <summary>
		/// Member data on which the Upstream_PosY property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Upstream_PosY;
		/// <summary>
		/// Upstream Y position (extrapolated at Z = Upstream_PosZ, usually = 0) of the track.
		/// </summary>
		public double Upstream_PosY
		{
            get
            {
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Upstream_PosY_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
            					if (m_FittingSegments>Segments.Length) m_FittingSegments = Segments.Length;
			            		Compute_Local_YCoord(Segments.Length-m_FittingSegments,out m_Upstream_SlopeY,out m_Upstream_PosY);
            					m_Upstream_SlopeY_Updated = true;
			            		m_Upstream_PosY_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeUpstreamBaseTrackExtrapolation(out p, out s);
                                m_Upstream_PosX = p.X;
                                m_Upstream_PosY = p.Y;
                                m_Upstream_SlopeX = s.X;
                                m_Upstream_SlopeY = s.Y;
            					m_Upstream_SlopeX_Updated = true;
            					m_Upstream_SlopeY_Updated = true;
			            		m_Upstream_PosX_Updated = true;
			            		m_Upstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
				};
				return m_Upstream_PosY;
			}
		}

		/// <summary>
		/// Member data that signals whether the Upstream_SlopeX property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_SlopeX_Updated= false;
		/// <summary>
		/// Member data on which the Upstream_SlopeY property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Upstream_SlopeX;
		/// <summary>
		/// Upstream X slope of the track.
		/// </summary>
		public double Upstream_SlopeX
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Upstream_SlopeX_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
					            if (m_FittingSegments>Segments.Length) m_FittingSegments= Segments.Length;
					            Compute_Local_XCoord(Length-m_FittingSegments,out m_Upstream_SlopeX,out m_Upstream_PosX);
            					m_Upstream_SlopeX_Updated = true;
			            		m_Upstream_PosX_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeUpstreamBaseTrackExtrapolation(out p, out s);
                                m_Upstream_PosX = p.X;
                                m_Upstream_PosY = p.Y;
                                m_Upstream_SlopeX = s.X;
                                m_Upstream_SlopeY = s.Y;
            					m_Upstream_SlopeX_Updated = true;
            					m_Upstream_SlopeY_Updated = true;
			            		m_Upstream_PosX_Updated = true;
			            		m_Upstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
				};
				return m_Upstream_SlopeX;
			}

		}

		/// <summary>
		/// Member data that signals whether the Upstream_PosX property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_PosX_Updated= false;
		/// <summary>
		/// Member data on which the Upstream_PosX property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Upstream_PosX;
		/// <summary>
		/// Upstream X position (extrapolated at Z = Upstream_PosZ, usually = 0) of the track.
		/// </summary>
		public double Upstream_PosX
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Upstream_PosX_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
					            if (m_FittingSegments>Segments.Length) m_FittingSegments= Segments.Length;
					            Compute_Local_XCoord(Length-m_FittingSegments,out m_Upstream_SlopeX,out m_Upstream_PosX);
            					m_Upstream_SlopeX_Updated = true;
			            		m_Upstream_PosX_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeUpstreamBaseTrackExtrapolation(out p, out s);
                                m_Upstream_PosX = p.X;
                                m_Upstream_PosY = p.Y;
                                m_Upstream_SlopeX = s.X;
                                m_Upstream_SlopeY = s.Y;
            					m_Upstream_SlopeX_Updated = true;
            					m_Upstream_SlopeY_Updated = true;
			            		m_Upstream_PosX_Updated = true;
			            		m_Upstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
				};
				return m_Upstream_PosX;
			}

		}

		/// <summary>
		/// Most upstream Z where the track has been seen.
		/// </summary>
		public double Upstream_Z { get { return Segments[Segments.Length - 1].Info.BottomZ; } }

		/// <summary>
		/// Z where the upstream positions are extrapolated, usually = 0.
		/// </summary>
		public double Upstream_PosZ { get { return 0.0;}  }

		/// <summary>
		/// Member data that signals whether the Downstream_SlopeY property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_SlopeY_Updated= false;
		/// <summary>
		/// Member data on which the Downstream_SlopeY property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Downstream_SlopeY;
		/// <summary>
		/// Downstream Y slope of the track.
		/// </summary>
		public double Downstream_SlopeY
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Downstream_SlopeY_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
            					if (m_FittingSegments>Segments.Length) m_FittingSegments= Segments.Length;
			            		Compute_Local_YCoord(0, out m_Downstream_SlopeY, out m_Downstream_PosY);
            					m_Downstream_SlopeY_Updated = true;
			            		m_Downstream_PosY_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeDownstreamBaseTrackExtrapolation(out p, out s);
                                m_Downstream_PosX = p.X;
                                m_Downstream_PosY = p.Y;
                                m_Downstream_SlopeX = s.X;
                                m_Downstream_SlopeY = s.Y;
            					m_Downstream_SlopeX_Updated = true;
            					m_Downstream_SlopeY_Updated = true;
			            		m_Downstream_PosX_Updated = true;
			            		m_Downstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
                };
				return m_Downstream_SlopeY;
			}
		}

		/// <summary>
		/// Member data that signals whether the Downstream_PosY property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_PosY_Updated= false;
		/// <summary>
		/// Member data on which the Downstream_PosY property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Downstream_PosY;
		/// <summary>
		/// Downstream Y position (extrapolated at Z = Downstream_PosZ, usually = 0) of the track.
		/// </summary>
		public double Downstream_PosY
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Downstream_PosY_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
            					if (m_FittingSegments>Segments.Length) m_FittingSegments= Segments.Length;
			            		Compute_Local_YCoord(0, out m_Downstream_SlopeY, out m_Downstream_PosY);
            					m_Downstream_SlopeY_Updated = true;
			            		m_Downstream_PosY_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeDownstreamBaseTrackExtrapolation(out p, out s);
                                m_Downstream_PosX = p.X;
                                m_Downstream_PosY = p.Y;
                                m_Downstream_SlopeX = s.X;
                                m_Downstream_SlopeY = s.Y;
            					m_Downstream_SlopeX_Updated = true;
            					m_Downstream_SlopeY_Updated = true;
			            		m_Downstream_PosX_Updated = true;
			            		m_Downstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
                };				
                return m_Downstream_PosY;
			}
		}

		/// <summary>
		/// Most downstream Z where the track has been seen.
		/// </summary>
		public double Downstream_Z { get { return Segments[0].Info.TopZ; } }

		/// <summary>
		/// Z where the downstream positions are extrapolated, usually = 0.
		/// </summary>
		public double Downstream_PosZ { get { return 0.0; } }

		/// <summary>
		/// Member data that signals whether the Downstream_SlopeX property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_SlopeX_Updated= false;
		/// <summary>
		/// Member data on which the Downstream_SlopeX property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Downstream_SlopeX;
		/// <summary>
		/// Downstream X slope of the track.
		/// </summary>
		public double Downstream_SlopeX
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Downstream_SlopeX_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
            					if (m_FittingSegments>Segments.Length) m_FittingSegments= Segments.Length;
			            		Compute_Local_XCoord(0, out m_Downstream_SlopeX, out m_Downstream_PosX);
            					m_Downstream_SlopeX_Updated = true;
			            		m_Downstream_PosX_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeDownstreamBaseTrackExtrapolation(out p, out s);
                                m_Downstream_PosX = p.X;
                                m_Downstream_PosY = p.Y;
                                m_Downstream_SlopeX = s.X;
                                m_Downstream_SlopeY = s.Y;
            					m_Downstream_SlopeX_Updated = true;
            					m_Downstream_SlopeY_Updated = true;
			            		m_Downstream_PosX_Updated = true;
			            		m_Downstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
                };
				return m_Downstream_SlopeX;
			}
		}


		/// <summary>
		/// Member data that signals whether the Downstream_PosX property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_PosX_Updated= false;
		/// <summary>
		/// Member data on which the Downstream_PosX property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Downstream_PosX;
		/// <summary>
		/// Downstream X position (extrapolated at Z = Downstream_PosZ, usually = 0) of the track.
		/// </summary>
		public double Downstream_PosX
		{
			get
			{
				if (Segments.Length==0) throw new Exception("No segments available");
				if (!m_Downstream_PosX_Updated)
				{
                    switch (s_TrackExtrapolationMode)
                    {
                        case ExtrapolationMode.SegmentFit: 
                            {
            					if (m_FittingSegments>Segments.Length) m_FittingSegments= Segments.Length;
			            		Compute_Local_XCoord(0, out m_Downstream_SlopeX, out m_Downstream_PosX);
            					m_Downstream_SlopeX_Updated = true;
			            		m_Downstream_PosX_Updated = true;
                            }
                            break;

                        case ExtrapolationMode.EndBaseTrack:
                            {
                                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                                SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();
                                ComputeDownstreamBaseTrackExtrapolation(out p, out s);
                                m_Downstream_PosX = p.X;
                                m_Downstream_PosY = p.Y;
                                m_Downstream_SlopeX = s.X;
                                m_Downstream_SlopeY = s.Y;
            					m_Downstream_SlopeX_Updated = true;
            					m_Downstream_SlopeY_Updated = true;
			            		m_Downstream_PosX_Updated = true;
			            		m_Downstream_PosY_Updated = true;
                            }
                            break;

                        default: throw new Exception("Unknown extrapolation mode.");
                    }
				};
				return m_Downstream_PosX;
			}
		}

		/// <summary>
		/// Member data on which the Upstream_Vertex property relies. Can be accessed by derived classes.
		/// </summary>
		protected Vertex m_Upstream_Vertex = null;
		
		/// <summary>
		/// Upstream vertex, i.e. the vertex where the particle associated to the track is produced.
		/// </summary>
		public Vertex Upstream_Vertex { get { return m_Upstream_Vertex; } }
		//internal void SetUpstreamVertex(Vertex v) { m_Upstream_Vertex = v; }
		public void SetUpstreamVertex(Vertex v) { m_Upstream_Vertex = v; }

		/// <summary>
		/// Signals whether the Upstream_Impact_Parameter property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Upstream_Impact_Parameter_Updated = false;

		/// <summary>
		/// Impact parameter of this track w.r.t. the upstream vertex.
		/// </summary>
		protected double m_Upstream_Impact_Parameter;

		/// <summary>
		/// Impact parameter of the track w.r.t. the upstream vertex.
		/// </summary>
		public double Upstream_Impact_Parameter
		{
			get
			{
				if (m_Upstream_Vertex == null) throw new Exception("No upstream vertex");
				if (!m_Upstream_Impact_Parameter_Updated)
				{
					m_Upstream_Vertex.NotifyChanged();
                    VertexFit vf = m_Upstream_Vertex.GetVertexFit(0, null);
                    if (vf.Count == 1) m_Upstream_Impact_Parameter = 0.0;
                    else m_Upstream_Impact_Parameter = vf.TrackIP(vf.Track(new BaseTrackIndex(m_Id)));
                    m_Upstream_Impact_Parameter_Updated = true;
                    /*
					double dx = m_Upstream_Vertex.X - Upstream_PosX - Upstream_SlopeX * m_Upstream_Vertex.Z;
					double dy = m_Upstream_Vertex.Y - Upstream_PosY - Upstream_SlopeY * m_Upstream_Vertex.Z;
					m_Upstream_Impact_Parameter = Math.Sqrt(dx * dx + dy * dy);                    
					m_Upstream_Impact_Parameter_Updated = true;
                     */
				}
				return m_Upstream_Impact_Parameter;
			}
		}

        internal void SetUpstreamIP(double ip)
        {
            m_Upstream_Impact_Parameter_Updated = true;
            m_Upstream_Impact_Parameter = ip;
        }

        internal void SetDownstreamIP(double ip)
        {
            m_Downstream_Impact_Parameter_Updated = true;
            m_Downstream_Impact_Parameter = ip;
        }

        /// <summary>
		/// Member data on which the Downstream_Vertex property relies. Can be accessed by derived classes.
		/// </summary>
		protected Vertex m_Downstream_Vertex = null;

		/// <summary>
		/// Downstream vertex, i.e. the vertex where the particle associated to the track interacts or decays.
		/// </summary>
		public Vertex Downstream_Vertex { get { return m_Downstream_Vertex; } }
		//internal void SetDownstreamVertex(Vertex v) { m_Downstream_Vertex = v; }
		public void SetDownstreamVertex(Vertex v) { m_Downstream_Vertex = v; }

		/// <summary>
		/// Signals whether the Downstream_Impact_Parameter property needs recomputing. Can be accessed by derived classes.
		/// </summary>
		protected bool m_Downstream_Impact_Parameter_Updated = false;

		/// <summary>
		/// Member data on which the Downstream_Vertex property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Downstream_Impact_Parameter;

		/// <summary>
		/// Impact parameter of this track w.r.t. the downstream vertex.
		/// </summary>
		public double Downstream_Impact_Parameter
		{
			get
			{
				if (m_Downstream_Vertex == null) throw new Exception("No upstream vertex");
				if (!m_Downstream_Impact_Parameter_Updated)
				{
					m_Downstream_Vertex.NotifyChanged();
                    VertexFit vf = m_Downstream_Vertex.GetVertexFit(0, null);
                    if (vf.Count == 1) m_Downstream_Impact_Parameter = 0.0;
                    else m_Downstream_Impact_Parameter = vf.TrackIP(vf.Track(new BaseTrackIndex(m_Id)));
                    m_Downstream_Impact_Parameter_Updated = true;
                    /*
					double dx = m_Downstream_Vertex.X - Downstream_PosX - Downstream_SlopeX * m_Downstream_Vertex.Z;
					double dy = m_Downstream_Vertex.Y - Downstream_PosY - Downstream_SlopeY * m_Downstream_Vertex.Z;
					m_Downstream_Impact_Parameter = Math.Sqrt(dx * dx + dy * dy);
					m_Downstream_Impact_Parameter_Updated = true;
                    */
				}
				return m_Downstream_Impact_Parameter;
			}
		}

		/// <summary>
		/// Notifies the track that its parameters have changed and should be recomputed.
		/// </summary>
		public override void NotifyChanged()
		{
			m_Upstream_Layer_Updated = false;
			m_Upstream_LayerId_Updated = false;
			m_Downstream_Layer_Updated = false;
			m_Downstream_LayerId_Updated = false;
			m_Upstream_SlopeY_Updated = false;
			m_Upstream_SlopeX_Updated = false;
			m_Downstream_SlopeY_Updated = false;
			m_Downstream_SlopeX_Updated = false;
			m_Upstream_PosY_Updated = false;
			m_Upstream_PosX_Updated = false;
			m_Downstream_PosY_Updated = false;
			m_Downstream_PosX_Updated = false;
			m_Upstream_Impact_Parameter_Updated = false;
			m_Downstream_Impact_Parameter_Updated = false;
		}
			
		/// <summary>
		/// Adds a segment to the track.
		/// </summary>
		/// <param name="s"></param>
		public void AddSegment(Segment s)
		{
			int i, k=0;
			Segment[] tmp;
			if(Segments.Length != 0)
			{
				tmp = Segments;
				Segments = new Segment[Segments.Length +1];
				for (i=0; i < tmp.Length; i++) if (tmp[i].LayerOwner.Id<s.LayerOwner.Id) k++;
				if (i < tmp.Length && tmp[i].LayerOwner == s.LayerOwner) throw new Exception("A segment on this layer already belongs to this track!");

				Segments[k] = s;

				for (i = 0; i < k; i++) Segments[i] = tmp[i];
				for (; i < Segments.Length-1; i++) 
					(Segments[i + 1] = tmp[i]).SetTrackOwner(this, i + 1);
			}
			else Segments = new Segment[1] {s};
			if (s.TrackOwner != null) s.TrackOwner.RemoveSegment(s.LayerOwner.Id);
			s.SetTrackOwner(this, k);
			NotifyChanged();
		}

		/// <summary>
		/// Adds a segment to the track and check the consistency.
		/// </summary>
		/// <param name="s"></param>
		public void AddSegmentAndCheck(Segment s)
		{
			int i, k=0;
			Segment[] tmp;
			if(Segments.Length != 0)
			{
				tmp = Segments;
				Segments = new Segment[Segments.Length +1];
				for (i=0; i < tmp.Length; i++) if (tmp[i].LayerOwner.Id<s.LayerOwner.Id) k++;
				if (i < tmp.Length && tmp[i].LayerOwner == s.LayerOwner) throw new Exception("A segment on this layer already belongs to this track!");

				if ((k > 0 && tmp[k - 1].Info.BottomZ <= s.Info.BottomZ) ||
					(k < (tmp.Length - 1) && tmp[k + 1].Info.TopZ >= s.Info.TopZ))
					throw new Exception("Layer order is inconsistent with Z coordinate");
				
				Segments[k] = s;

				for (i = 0; i < k; i++) Segments[i] = tmp[i];
				for (; i < Segments.Length-1; i++) 
					(Segments[i + 1] = tmp[i]).SetTrackOwner(this, i + 1);
			}
			else Segments = new Segment[1] {s};
			if (s.TrackOwner != null) s.TrackOwner.RemoveSegment(s.LayerOwner.Id);
			s.SetTrackOwner(this, k);
			NotifyChanged();
		}

		/// <summary>
		/// Replaces a segment in the track.
		/// </summary>
		/// <param name="index"></param>
		/// <param name="s"></param>
		protected override void ReplaceSegment(int index, Segment s)
		{
			if ((index > 0 && Segments[index - 1].LayerOwner.Id >= s.LayerOwner.Id) ||
				(index < Segments.Length - 1 && Segments[index + 1].LayerOwner.Id <= s.LayerOwner.Id)) throw new Exception("Layer order violation");
			if ((index > 0 && Segments[index - 1].Info.BottomZ <= s.Info.TopZ) ||
				(index < (Segments.Length - 1) && Segments[index + 1].Info.TopZ >= s.Info.BottomZ))
				throw new Exception("Layer order is inconsistent with Z coordinate");
			Segments[index].SetTrackOwner(null, -1);
			if (s.TrackOwner != null) s.TrackOwner.RemoveSegment(s.LayerOwner.Id);
			Segments[index] = s;
			s.SetTrackOwner(this, index);
			NotifyChanged();
		}

		/// <summary>
		/// Removes a segment from the track.
		/// </summary>
		/// <param name="layerindex"></param>
		public void RemoveSegment(int layerindex)
		{
			int i,j;
			for (i = 0; i < Segments.Length && Segments[i].LayerOwner.Id != layerindex; i++);
			if (i == Segments.Length) throw new Exception("No segment to remove");
			Segments[i].SetTrackOwner(null, -1);
			Segment [] tmp = Segments;
			Segments = new Segment[tmp.Length - 1];
			for (j = 0; j < i; j++) Segments[j] = tmp[j]; 
			for (j = i; j < Segments.Length; j++)
				(Segments[j] = tmp[j + 1]).SetTrackOwner(this, j);
			NotifyChanged();
		}

        /// <summary>
        /// Computes fit information at a specified layer, projecting to a specified Z
        /// </summary>
        /// <param name="layerid">the id of the layer that is the center of the fit.</param>
        /// <param name="z">the Z to project position to.</param>
        /// <returns>fit information.</returns>
        public SySal.Tracking.MIPEmulsionTrackInfo Fit(int layerid, double z)
        {
            SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
            info.Count = (ushort)Length;
            info.AreaSum = 0;
            info.Sigma = 0;
            if (layerid < DownstreamLayerId)
            {
                info.Intercept.X = Downstream_PosX + (z - Downstream_PosZ) * Downstream_SlopeX;
                info.Intercept.Y = Downstream_PosY + (z - Downstream_PosZ) * Downstream_SlopeY;
                info.Slope.X = Downstream_SlopeX;
                info.Slope.Y = Downstream_SlopeY;
            }
            else if (layerid > UpstreamLayerId)
            {
                info.Intercept.X = Upstream_PosX + (z - Upstream_PosZ) * Upstream_SlopeX;
                info.Intercept.Y = Upstream_PosY + (z - Upstream_PosZ) * Upstream_SlopeY;
                info.Slope.X = Upstream_SlopeX;
                info.Slope.Y = Upstream_SlopeY;                
            }
            else
            {
                int i;
                for (i = 0; i < this.Segments.Length && this.Segments[i].LayerOwner.Id < layerid; i++);
                double s, p;
                i -= m_FittingSegments / 2;
                if (i > Segments.Length - m_FittingSegments) i = Segments.Length - m_FittingSegments;
                if (i < 0) i = 0;
                Compute_Local_XCoord(i, out s, out p);
                info.Intercept.X = p + z * s;
                info.Slope.X = s;
                Compute_Local_YCoord(i, out s, out p);
                info.Intercept.Y = p + z * s;
                info.Slope.Y = s;
            }
            info.Intercept.Z = z;                
            info.Slope.Z = 1.0;
            info.TopZ = 0.0;
            info.BottomZ = 0.0;
            return info;
        }

        /// <summary>
        /// Computes the upstream extrapolation using the nearest base track.
        /// </summary>
        /// <param name="pos">the fitted position.</param>
        /// <param name="slope">the fitted slope.</param>
        public void ComputeUpstreamBaseTrackExtrapolation(out SySal.BasicTypes.Vector pos, out SySal.BasicTypes.Vector slope)
        {
            pos = Segments[Segments.Length - 1].Info.Intercept;
            SySal.BasicTypes.Vector locslope = Segments[Segments.Length - 1].Info.Slope;
            int i;
            for (i = Segments.Length - 1; i >= 0 && 
                    (
                        (Segments[i].LayerOwner.Side == 0 && Segments[i].Info.Sigma >= 0.0) ||
                        (i > 0 && Segments[i].LayerOwner.Side == 2 && Segments[i - 1].LayerOwner.Side == 1 && Segments[i].LayerOwner.BrickId == Segments[i - 1].LayerOwner.BrickId && Segments[i].LayerOwner.SheetId == Segments[i - 1].LayerOwner.SheetId)
                    ) == false;                
                i--);
            if (i < 0)
            {
                SySal.Tracking.MIPEmulsionTrackInfo s = Segments[Segments.Length - 1].Info;
                pos = s.Intercept;
                slope = s.Slope;
                slope.Z = 1.0;
                pos.X -= pos.Z * slope.X;
                pos.Y -= pos.Z * slope.Y;
                pos.Z = 0.0;
            }
            else if (Segments[i].LayerOwner.Side == 0)
            {
                //pos = Segments[Segments.Length - 1].Info.Intercept;
                slope = Segments[i].Info.Slope;
                slope.Z = 1.0;
                double uz = Segments[Segments.Length - 1].LayerOwner.UpstreamZ;
                pos.X -= (pos.Z - uz) * locslope.X;
                pos.Y -= (pos.Z - uz) * locslope.Y;
                pos.X -= uz * slope.X;
                pos.Y -= uz * slope.Y;
                pos.Z = 0.0;
            }
            else
            {
                SySal.Tracking.MIPEmulsionTrackInfo s_top = Segments[i - 1].Info;
                SySal.Tracking.MIPEmulsionTrackInfo s_bottom = Segments[i].Info;
                //pos = Segments[0].Info.Intercept;
                slope.X = ((s_top.Intercept.X + (s_top.BottomZ - s_top.Intercept.Z) * s_top.Slope.X) - (s_bottom.Intercept.X + (s_bottom.TopZ - s_bottom.Intercept.Z) * s_bottom.Slope.X)) / (s_top.BottomZ - s_bottom.TopZ);
                slope.Y = ((s_top.Intercept.Y + (s_top.BottomZ - s_top.Intercept.Z) * s_top.Slope.Y) - (s_bottom.Intercept.Y + (s_bottom.TopZ - s_bottom.Intercept.Z) * s_bottom.Slope.Y)) / (s_top.BottomZ - s_bottom.TopZ);
                slope.Z = 1.0;
                pos.X -= pos.Z * slope.X;
                pos.Y -= pos.Z * slope.Y;
                pos.Z = 0.0;                
            }            
        }

        /// <summary>
        /// Computes the downstream extrapolation using the nearest base track.
        /// </summary>
        /// <param name="pos">the fitted position.</param>
        /// <param name="slope">the fitted slope.</param>
        public void ComputeDownstreamBaseTrackExtrapolation(out SySal.BasicTypes.Vector pos, out SySal.BasicTypes.Vector slope)
        {
            pos = Segments[0].Info.Intercept;
            SySal.BasicTypes.Vector locslope = Segments[0].Info.Slope;
            int i;
            for (i = 0; i < Segments.Length &&
                (
                    (Segments[i].LayerOwner.Side == 0 && Segments[i].Info.Sigma >= 0.0) ||
                    (i < (Segments.Length - 1) && Segments[i].LayerOwner.Side == 1 && Segments[i + 1].LayerOwner.Side == 2 && Segments[i].LayerOwner.BrickId == Segments[i + 1].LayerOwner.BrickId && Segments[i].LayerOwner.SheetId == Segments[i + 1].LayerOwner.SheetId)
                ) == false;
                i++) ;
            if (i >= Segments.Length)
            {
                SySal.Tracking.MIPEmulsionTrackInfo s = Segments[0].Info;
                pos = s.Intercept;
                slope = s.Slope;
                slope.Z = 1.0;
                pos.X -= pos.Z * slope.X;
                pos.Y -= pos.Z * slope.Y;
                pos.Z = 0.0;
            }
            else if (Segments[i].LayerOwner.Side == 0)
            {
                //pos = Segments[0].Info.Intercept;
                slope = Segments[i].Info.Slope;
                slope.Z = 1.0;
                double dz = Segments[0].LayerOwner.DownstreamZ;
                pos.X += (dz - pos.Z) * locslope.X;
                pos.Y += (dz - pos.Z) * locslope.Y;
                pos.X -= dz * slope.X;
                pos.Y -= dz * slope.Y;
                pos.Z = 0.0;
            }
            else
            {
                SySal.Tracking.MIPEmulsionTrackInfo s_top = Segments[i].Info;
                SySal.Tracking.MIPEmulsionTrackInfo s_bottom = Segments[i + 1].Info;
                //pos = Segments[0].Info.Intercept;
                slope.X = ((s_top.Intercept.X + (s_top.BottomZ - s_top.Intercept.Z) * s_top.Slope.X) - (s_bottom.Intercept.X + (s_bottom.TopZ - s_bottom.Intercept.Z) * s_bottom.Slope.X)) / (s_top.BottomZ - s_bottom.TopZ);
                slope.Y = ((s_top.Intercept.Y + (s_top.BottomZ - s_top.Intercept.Z) * s_top.Slope.Y) - (s_bottom.Intercept.Y + (s_bottom.TopZ - s_bottom.Intercept.Z) * s_bottom.Slope.Y)) / (s_top.BottomZ - s_bottom.TopZ);
                slope.Z = 1.0;
                pos.X -= pos.Z * slope.X;
                pos.Y -= pos.Z * slope.Y;
                pos.Z = 0.0;
            }
        }

		/// <summary>
		/// Computes the local X slope and position (extrapolated at Z = 0), using the local fit of aligned base track positions.
		/// </summary>
		/// <param name="StartingId">the first segment to use.</param>
		/// <param name="SlopeX">the fitted X slope.</param>
		/// <param name="PosX">the fitted X position.</param>
		public void Compute_Local_XCoord(int StartingId, out double SlopeX, out double PosX)
		{
			double[] ttmpz;
			double[] ttmpx;
			ArrayList tmpz = new ArrayList();
			ArrayList tmpx = new ArrayList();
			double a=0, b=0, dum=0;
			int j = 0;
			int fitsegs = m_FittingSegments;
            if (fitsegs > Segments.Length) fitsegs = Segments.Length;
			if(fitsegs > 1 && Segments.Length > 1)
			{
				if(StartingId + fitsegs>Segments.Length)
				{
					StartingId -= StartingId + fitsegs - Segments.Length;
					if (StartingId < 0) 
					{
						StartingId = 0;
						fitsegs = Segments.Length;
					}
				}
				for (int i = StartingId; i < StartingId + fitsegs; i++)
					if(i < Segments.Length)
					{
						tmpz.Add(Segments[i].Info.Intercept.Z);
						tmpx.Add(Segments[i].Info.Intercept.X);
						j++;
					};
				if(j > 1)
				{
					ttmpz = new double[j];
					ttmpx = new double[j];
					ttmpz = (double[])tmpz.ToArray(typeof(double));
					ttmpx = (double[])tmpx.ToArray(typeof(double));
					Fitting.LinearFitSE(ttmpz,ttmpx, ref a, ref b, ref dum, ref dum, ref dum, ref dum, ref dum);
				}
				else if (j == 1)
				{
					a = Segments[StartingId].Info.Slope.X;
					b = Segments[StartingId].Info.Intercept.X - a * Segments[StartingId].Info.Intercept.Z;
				}
			}
			else 
			{
				a = Segments[StartingId].Info.Slope.X;
				b = Segments[StartingId].Info.Intercept.X - a* Segments[StartingId].Info.Intercept.Z;
			};
			SlopeX=a;
			PosX=b;
		}

		/// <summary>
		/// Computes the local Y slope and position (extrapolated at Z = 0), using the local fit of aligned base track positions.
		/// </summary>
		/// <param name="StartingId">the first segment to use.</param>
		/// <param name="SlopeY">the fitted Y slope.</param>
		/// <param name="PosY">the fitted Y position.</param>
		public void Compute_Local_YCoord(int StartingId, out double SlopeY, out double PosY)
		{
			double[] ttmpz;
			double[] ttmpy;
			ArrayList tmpz = new ArrayList();
			ArrayList tmpy = new ArrayList();
			double a=0, b=0, dum=0;
			int j = 0;
			int fitsegs = m_FittingSegments;
            if (fitsegs > Segments.Length) fitsegs = Segments.Length;
			if(fitsegs > 1 && Segments.Length > 1)
			{
				if(StartingId + fitsegs>Segments.Length)
				{
					StartingId -= StartingId + fitsegs - Segments.Length;
					if (StartingId < 0) 
					{
						StartingId = 0;
						fitsegs = Segments.Length;
					}
				}
				for (int i = StartingId; i < StartingId + fitsegs; i++)
					if(i < Segments.Length)
					{
						tmpz.Add(Segments[i].Info.Intercept.Z);
						tmpy.Add(Segments[i].Info.Intercept.Y);
						j++;
					};
				if(j > 1)
				{
					ttmpz = new double[j];
					ttmpy = new double[j];
					ttmpz = (double[])tmpz.ToArray(typeof(double));
					ttmpy = (double[])tmpy.ToArray(typeof(double));
					Fitting.LinearFitSE(ttmpz,ttmpy, ref a, ref b, ref dum, ref dum, ref dum, ref dum, ref dum);
				}
				else if (j == 1)
				{
					a = Segments[StartingId].Info.Slope.Y;
					b = Segments[StartingId].Info.Intercept.Y - a * Segments[StartingId].Info.Intercept.Z;
				}
			}
			else 
			{
				a = Segments[StartingId].Info.Slope.Y;
				b = Segments[StartingId].Info.Intercept.Y - a* Segments[StartingId].Info.Intercept.Z;
			};
			SlopeY=a;
			PosY=b;
		}

		public override string ToString()
		{
			string text = "TRACK INFO";
			text += "\r\nID: " + Id.ToString();
			text += "\r\nSegments: " + Length.ToString();
			text += "\r\nComments: " + ((Comment == null) ? "" : Comment);
			text += "\r\nUpstream Z: " + Upstream_Z.ToString("F1");
			text += "\r\nUpstream Vtx: " + ((Upstream_Vertex == null) ? "" : Upstream_Vertex.Id.ToString());
			text += "\r\nUpstream IP: " + ((Upstream_Vertex == null) ? "" : Upstream_Impact_Parameter.ToString("F3"));
			text += "\r\nUpstream Slope X: " + (Upstream_SlopeX.ToString("F5"));
			text += "\r\nUpstream Slope Y: "+ (Upstream_SlopeY.ToString("F5"));
			text += "\r\nUpstream Pos X: "+ ((Upstream_SlopeX * (Upstream_Z - Upstream_PosZ) + Upstream_PosX).ToString("F1"));
			text += "\r\nUpstream Pos Y: " + ((Upstream_SlopeY * (Upstream_Z - Upstream_PosZ) + Upstream_PosY).ToString("F1"));
			text += "\r\nDownstream Z: "+ (Downstream_Z.ToString("F1"));
			text += "\r\nDownstream Vtx: " + ((Downstream_Vertex == null) ? "" : Downstream_Vertex.Id.ToString());
			text += "\r\nDownstream IP: " + ((Downstream_Vertex == null) ? "" : Downstream_Impact_Parameter.ToString("F3"));
			text += "\r\nDownstream Slope X: " + (Downstream_SlopeX.ToString("F5"));
			text += "\r\nDownstream Slope Y: " + (Downstream_SlopeY.ToString("F5"));
			text += "\r\nDownstream Pos X: " + ((Downstream_SlopeX * (Downstream_Z - Downstream_PosZ) + Downstream_PosX).ToString("F1"));
			text += "\r\nDownstream Pos Y: " + ((Downstream_SlopeY * (Downstream_Z - Downstream_PosZ) + Downstream_PosY).ToString("F1"));
			text += "\r\nSEGMENTS";
			text += "\r\nID\tGRAINS\tSX\tSY\tIX\tIY\tIZ\tTZ\tBZ\tLayerID\tSheetID";
			int i;
			for (i = 0; i < Length; i++)
			{
				SySal.Tracking.MIPEmulsionTrackInfo info = this[i].Info;
				text += "\r\n" + this[i].PosInLayer.ToString() + "\t" + info.Count.ToString() + "\t" + info.Slope.X.ToString("F5") + "\t" + info.Slope.Y.ToString("F5") + "\t" + info.Intercept.X.ToString("F1") + "\t" + info.Intercept.Y.ToString("F1") + "\t" + info.Intercept.Z.ToString("F1") + "\t" + this[i].LayerOwner.UpstreamZ.ToString("F1") + "\t" + this[i].LayerOwner.DownstreamZ.ToString("F1") + "\t" + this[i].LayerOwner.Id.ToString() + "\t" + this[i].LayerOwner.SheetId.ToString();
			}
			return text;
		}

		#region IAttributeList Members

		/// <summary>
		/// The list of the attributes of this track.
		/// </summary>
		protected Attribute [] m_AttributeList = new Attribute[0];

		/// <summary>
		/// Sets an attribute of the track.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be set.</param>
		/// <param name="attributevalue">the value of the attribute to be set.</param>
		public void SetAttribute(Index attributeindex, double attributevalue)
		{
			int i;
			for (i = 0; i < m_AttributeList.Length; i++)
			{
				if (m_AttributeList[i].Index.Equals(attributeindex))
				{
					m_AttributeList[i].Value = attributevalue;
					return;
				}
			}
			Attribute [] newlist = new Attribute[m_AttributeList.Length + 1];
			for (i = 0; i < m_AttributeList.Length; i++)
				newlist[i] = m_AttributeList[i];
			newlist[i] = new Attribute(attributeindex, attributevalue);
			m_AttributeList = newlist;
		}

		/// <summary>
		/// Removes an attribute from the track.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be removed.</param>
		public void RemoveAttribute(Index attributeindex)
		{
			int i;
			for (i = 0; i < m_AttributeList.Length && m_AttributeList[i].Index.Equals(attributeindex) == false; i++);
			if (i < m_AttributeList.Length)
			{
				int j;
				Attribute [] newlist = new Attribute[m_AttributeList.Length - 1];
				for (j = 0; j < i; j++)
					newlist[j] = m_AttributeList[j];
				for (; j < newlist.Length; j++)
					newlist[j] = m_AttributeList[j + 1];
				m_AttributeList = newlist;
			}
		}

		/// <summary>
		/// Gets an attribute of the track. An exception is thrown if the attribute is not found.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be read.</param>
		/// <returns>the value of the attribute.</returns>
		public double GetAttribute(Index attributeindex)
		{
			int i;
			for (i = 0; i < m_AttributeList.Length; i++)
				if (m_AttributeList[i].Index.Equals(attributeindex)) return m_AttributeList[i].Value;
			throw new Exception("Attribute " + attributeindex.ToString() + " cannot be found");
		}

		/// <summary>
		/// Returns the list of the attributes.
		/// </summary>
		/// <returns>the list of the attributes of the track.</returns>
		public Attribute[] ListAttributes()
		{
			Attribute [] newlist = new Attribute[m_AttributeList.Length];
			int i;
			for (i = 0; i < newlist.Length; i++)
				newlist[i] = m_AttributeList[i];
			return newlist;
		}

		#endregion
	}

    /// <summary>
    /// A shower, made of segmemnts (base tracks or microtracks).
    /// Derived from SegmentList, so its segments can be accessed in array-like fashion.
    /// </summary>
    public class Shower : SegmentList, IAttributeList
    {

        /// <summary>
        /// Member data on which the Segment Id property relies. Can be accessed by derived classes.
        /// </summary>
        protected int m_Id;
        /// <summary>
        /// The id of the shower, usually a sequential number in the volume.
        /// </summary>
        public int Id { get { return m_Id; } }
        internal void SetId(int id) { m_Id = id; }
        /// <summary>
		/// Protected member that allows Id changing in derived classes.
		/// </summary>
		protected static void SetId(Shower t, int newid)
		{
			t.m_Id = newid;
		}

		/// <summary>
        /// Protected constructor. Prevents users from creating Shower without providing consistent information. Is implicitly called in derived classes.
		/// </summary>
        public Shower()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		/// <summary>
		/// Builds a new shower with the specified identifying number.
		/// </summary>
		/// <param name="id"></param>
		public Shower(int id)
		{
			//
			// TODO: Add constructor logic here
			//
			m_Id = id;
		}

        /// <summary>
        /// Adds a segment to the shower.
        /// </summary>
        /// <param name="s">the segment to be added.</param>
        public void AddSegment(Segment s)
        {
            int i, k = 0;
            Segment[] tmp;
            if (Segments.Length != 0)
            {
                tmp = Segments;
                Segments = new Segment[Segments.Length + 1];
                for (i = 0; i < tmp.Length && tmp[i].LayerOwner.Id <= s.LayerOwner.Id; i++)                
                    if (tmp[i] == s) throw new Exception("The segment is already contained in the shower.");

                Segments[k] = s;

                for (i = 0; i < k; i++) Segments[i] = tmp[i];
                for (; i < Segments.Length - 1; i++)
                    Segments[i + 1] = tmp[i];
            }
            else Segments = new Segment[1] { s };
        }

		/// <summary>
		/// Removes a segment from the shower.
		/// </summary>
		/// <param name="s">the segment to be removed.</param>
		public void RemoveSegment(SySal.TotalScan.Segment s)
		{
			int i,j;
			for (i = 0; i < Segments.Length && Segments[i] != s; i++);
			if (i == Segments.Length) throw new Exception("No segment to remove");			
			Segment [] tmp = Segments;
			Segments = new Segment[tmp.Length - 1];
			for (j = 0; j < i; j++) Segments[j] = tmp[j]; 
			for (j = i; j < Segments.Length; j++)
				Segments[j] = tmp[j + 1];			
		}

        #region IAttributeList Members

        /// <summary>
        /// The list of the attributes of this track.
        /// </summary>
        protected Attribute[] m_AttributeList = new Attribute[0];

        /// <summary>
        /// Sets an attribute of the track.
        /// </summary>
        /// <param name="attributeindex">the index of the attribute to be set.</param>
        /// <param name="attributevalue">the value of the attribute to be set.</param>
        public void SetAttribute(Index attributeindex, double attributevalue)
        {
            int i;
            for (i = 0; i < m_AttributeList.Length; i++)
            {
                if (m_AttributeList[i].Index.Equals(attributeindex))
                {
                    m_AttributeList[i].Value = attributevalue;
                    return;
                }
            }
            Attribute[] newlist = new Attribute[m_AttributeList.Length + 1];
            for (i = 0; i < m_AttributeList.Length; i++)
                newlist[i] = m_AttributeList[i];
            newlist[i] = new Attribute(attributeindex, attributevalue);
            m_AttributeList = newlist;
        }

        /// <summary>
        /// Removes an attribute from the track.
        /// </summary>
        /// <param name="attributeindex">the index of the attribute to be removed.</param>
        public void RemoveAttribute(Index attributeindex)
        {
            int i;
            for (i = 0; i < m_AttributeList.Length && m_AttributeList[i].Index.Equals(attributeindex) == false; i++) ;
            if (i < m_AttributeList.Length)
            {
                int j;
                Attribute[] newlist = new Attribute[m_AttributeList.Length - 1];
                for (j = 0; j < i; j++)
                    newlist[j] = m_AttributeList[j];
                for (; j < newlist.Length; j++)
                    newlist[j] = m_AttributeList[j + 1];
                m_AttributeList = newlist;
            }
        }

        /// <summary>
        /// Gets an attribute of the track. An exception is thrown if the attribute is not found.
        /// </summary>
        /// <param name="attributeindex">the index of the attribute to be read.</param>
        /// <returns>the value of the attribute.</returns>
        public double GetAttribute(Index attributeindex)
        {
            int i;
            for (i = 0; i < m_AttributeList.Length; i++)
                if (m_AttributeList[i].Index.Equals(attributeindex)) return m_AttributeList[i].Value;
            throw new Exception("Attribute " + attributeindex.ToString() + " cannot be found");
        }

        /// <summary>
        /// Returns the list of the attributes.
        /// </summary>
        /// <returns>the list of the attributes of the track.</returns>
        public Attribute[] ListAttributes()
        {
            Attribute[] newlist = new Attribute[m_AttributeList.Length];
            int i;
            for (i = 0; i < newlist.Length; i++)
                newlist[i] = m_AttributeList[i];
            return newlist;
        }

        #endregion
    }

	#endregion

	#region Intersection

	public class BasicIntersection
	{
		public IntersectionType Type = IntersectionType.Unknown;
		public IntersectionSymmetry Symmetry = IntersectionSymmetry.Unknown;
		public double ClosestApproachDistance;
		public SySal.BasicTypes.Vector Pos;

	}
	
	public class SegmentIntersection: BasicIntersection
	{
		public Segment Segment1, Segment2;
	}

	public class TrackIntersection: BasicIntersection
	{
		public Track Track1, Track2;
	}

	#endregion

	#region Vertex

	/// <summary>
	/// A list of tracks. The vertex class is derived from this.
	/// </summary>
	
	public class TrackList
	{
		/// <summary>
		/// Member data holding the list of tracks.
		/// </summary>
		protected Track[] Tracks = new Track[0];

		/// <summary>
		/// Accesses the tracks in the list.
		/// </summary>
		public Track this[int index]
		{
			get { return Tracks[index];  }
			set { ReplaceTrack(index, value); NotifyChanged(); }
		}

		/// <summary>
		/// Returns the number of tracks in the list.
		/// </summary>
		public int Length
		{
			get { return Tracks.Length; }
		}

		/// <summary>
		/// Replaces a track in the list, notifying derived classes that the track has changed.
		/// </summary>
		/// <param name="index"></param>
		/// <param name="t"></param>
		protected virtual void ReplaceTrack(int index, Track t) {}

		/// <summary>
		/// Notifies derived classes that the list of tracks has changed.
		/// </summary>
		public virtual void NotifyChanged() {}
	}

	/// <summary>
	/// An intersection point of two or more tracks. It can be an interaction / decay vertex, as well as a kink point.
	/// </summary>
	public class Vertex : TrackList, IAttributeList
	{
		/// <summary>
		/// Protected constructor. Prevents users from creating Vertices without providing consistent information. Is implicitly called in derived classes.
		/// </summary>
		public Vertex()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		/// <summary>
		/// Builds a vertex with the specified id, usually the sequential number of the vertex in the volume.
		/// </summary>
		/// <param name="id"></param>
		public Vertex(int id)
		{
			//
			// TODO: Add constructor logic here
			//
			m_Id = id;
		}
	
		/// <summary>
		/// Member data on which the Id property relies. Can be accessed in derived classes.
		/// </summary>
		protected int m_Id;

		/// <summary>
		/// The vertex id number.
		/// </summary>
		public int Id { get { return m_Id; } }
		internal void SetId(int id) { m_Id = id; }

		/// <summary>
		/// Protected member that allows Id changing in derived classes.
		/// </summary>
		protected static void SetId(Vertex v, int newid)
		{
			v.m_Id = newid;
		}

		/// <summary>
		/// A user comment string accompanying the vertex.
		/// </summary>
		public string Comment;

		/// <summary>
		/// Signals whether the vertex coordinates should be updated. Can be accessed by derived classes.
		/// </summary>
		protected bool m_VertexCoordinatesUpdated = false;

        /// <summary>
        /// Notifies the vertex that its coordinates might have changed and need to be recomputed.
        /// </summary>
        public override void NotifyChanged()
        {
            m_VertexCoordinatesUpdated = false;
        }

		internal void SetPosDeltas(double x, double y, double z, double dx, double dy, double avgd)
		{
			m_X = x;
			m_Y = y;
			m_Z = z;
			m_DX = dx;
			m_DY = dy;
			m_AverageDistance = avgd;
			m_VertexCoordinatesUpdated = true;
		}

		/// <summary>
		/// Member data on which the X property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_X;

		/// <summary>
		/// X coordinate of the vertex.
		/// </summary>
		public double X
		{
			get
			{
				if (m_VertexCoordinatesUpdated==false) ComputeVertexCoordinates();
				return m_X;
			}
		}

		/// <summary>
		/// Member data on which the Y property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Y;

		/// <summary>
		/// Y coordinate of the vertex.
		/// </summary>
		public double Y
		{
			get
			{
				if (m_VertexCoordinatesUpdated==false) ComputeVertexCoordinates();
				return m_Y;
			}
		}

		/// <summary>
		/// Member data on which the Z property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_Z;

		/// <summary>
		/// Z coordinate of the vertex.
		/// </summary>
		public double Z
		{
			get
			{
				if (m_VertexCoordinatesUpdated == false) ComputeVertexCoordinates();
				return m_Z;
			}
		}

		/// <summary>
		/// Member data on which the DX property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_DX;

		/// <summary>
		/// Error on the X coordinate of the vertex.
		/// </summary>
		public double DX
		{
			get
			{
				if (m_VertexCoordinatesUpdated == false) ComputeVertexCoordinates();
				return m_DX;
			}
		}

		/// <summary>
		/// Member data on which the DY property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_DY;

		/// <summary>
		/// Error on the Y coordinate of the vertex.
		/// </summary>
		public double DY
		{
			get
			{
				if (m_VertexCoordinatesUpdated==false) ComputeVertexCoordinates();
				return m_DY;
			}
		}

		/// <summary>
		/// Member data on which the AverageDistance property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_AverageDistance;

		/// <summary>
		/// Average impact parameter of the tracks in the vertex.
		/// </summary>
		public double AverageDistance
		{
			get
			{
				if (m_VertexCoordinatesUpdated==false) ComputeVertexCoordinates();
				return m_AverageDistance;
			}
		}

		/// <summary>
		/// Adds a track to the vertex; if IsUpstream = true, the vertex is upstream of the track, otherwise it is downstream of the track.
		/// </summary>
		/// <param name="t"></param>
		/// <param name="IsUpstream"></param>
		public void AddTrack(Track t, bool IsUpstream)
		{
			int i;
			Track[] tmp;
			if (Tracks.Length == 0)
			{
				Tracks = new Track[1] {t};
			}
			else
			{
				for (i = 0; i < Tracks.Length; i++) if (Tracks[i] == t) throw new Exception("Cannot add the same track twice to this vertex.");
				tmp = Tracks;
				Tracks = new Track[tmp.Length + 1];
				for (i = 0; i < tmp.Length; i++) Tracks[i] = tmp[i];
				Tracks[i] = t;
			}
			m_VertexCoordinatesUpdated=false;
		}

		/// <summary>
		/// Removes a track from the vertex.
		/// </summary>
		/// <param name="t"></param>
		public void RemoveTrack(Track t)
		{
			int i, j;
			for (i = 0; i < Tracks.Length && Tracks[i] != t; i++);
			if (i == Tracks.Length) throw new Exception("No track to remove");
			if (t.Upstream_Vertex == this) t.SetUpstreamVertex(null);
			else if (t.Downstream_Vertex == this) t.SetDownstreamVertex(null);
			Track [] tmp = Tracks;
			Tracks = new Track[tmp.Length - 1];
			for (j=0; j < i; j++)
				Tracks[j] = tmp[j];
			for (j=i+1; j < tmp.Length; j++)
				Tracks[j-1] = tmp[j];
            m_VertexCoordinatesUpdated = false;
			NotifyChanged();
		}

        /// <summary>
        /// A function that returns a weight factor to use for a track in a vertex fit.
        /// </summary>
        /// <param name="t">the track for which a weight has to be computed.</param>
        /// <returns>the weight for the track fit.</returns>
        public delegate double dTrackWeightFunction(SySal.TotalScan.Track t);

        /// <summary>
        /// Property backer for <c>TrackWeightingFunction</c>.
        /// </summary>
        protected static dTrackWeightFunction s_TrackWeightingFunction = SlopeScatteringWeight;

        /// <summary>
        /// Gets/sets the function used to produce vertex fits.
        /// </summary>
        public static dTrackWeightFunction TrackWeightingFunction
        {
            get { return s_TrackWeightingFunction; }
            set { s_TrackWeightingFunction = value; }
        }

        /// <summary>
        /// Yields a weight that is inversely proportional to the RMS of slope in the track and directly to the square root of segments.
        /// </summary>
        /// <param name="t">the track for which a weight has to be computed.</param>
        /// <returns>the weight for the track fit.</returns>
        public static double SlopeScatteringWeight(SySal.TotalScan.Track t)
        {
            if (t.Length < 3) return 1.0e-3;
            double sx = 0.0;
            double sy = 0.0;
            double sx2 = 0.0;
            double sy2 = 0.0;
            int i;
            for (i = 0; i < t.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = t[i].Info;
                sx += info.Slope.X;
                sx2 += info.Slope.X * info.Slope.X;
                sy += info.Slope.Y;
                sy2 += info.Slope.Y * info.Slope.Y;
            }
            return (i * i) / ((i * sx2 - sx * sx) + (i * sy2 - sy * sy)) * Math.Sqrt(t.Length);
        }

        /// <summary>
        /// Yields a weight that is identical for all tracks.
        /// </summary>
        /// <param name="t">the track for which a weight has to be computed.</param>
        /// <returns>the weight for the track fit.</returns>
        public static double FlatWeight(SySal.TotalScan.Track t)
        {
            return 1.0;
        }

        public static readonly SySal.TotalScan.NamedAttributeIndex TrackWeightAttribute = new NamedAttributeIndex("VTXFITWEIGHT");

        /// <summary>
        /// Reads a weight from the VTXFITWEIGHT attribute. If missing, uses 1.0.
        /// </summary>
        /// <param name="t">the track for which a weight has to be computed.</param>
        /// <returns>the weight for the track fit.</returns>
        public static double AttributeWeight(SySal.TotalScan.Track t)
        {
            try
            {
                return t.GetAttribute(TrackWeightAttribute);
            }
            catch (Exception)
            {
                return 1.0;
            }
        }

        /// <summary>
        /// Returns a <see cref="VertexFit"/> object for advanced fitting functions.
        /// </summary>
        /// <param name="maxtrackextrapolation">the maximum extrapolation for a track fit. Negative numbers or zero default to 1000000.</param>
        /// <param name="weightfunction">the weighting function. If <c>null</c>, the default <see cref="SlopeScatteringWeight"/> is used.</param>
        public VertexFit GetVertexFit(double maxtrackextrapolation, dTrackWeightFunction weightfunction)
        {
            if (maxtrackextrapolation <= 0.0) maxtrackextrapolation = 1e6;
            if (weightfunction == null) weightfunction = SlopeScatteringWeight;
            if (Tracks.Length == 0) throw new Exception("Can't make a vertex with just one track.");
            VertexFit vf = new VertexFit();
            int i;
            for (i = 0; i < Tracks.Length; i++)
            {
                VertexFit.TrackFit tf = new VertexFit.TrackFit();
                Track tk = Tracks[i];
                if (tk.Upstream_Vertex == this)
                {
                    tf.Intercept.X = tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX;
                    tf.Intercept.Y = tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY;
                    tf.Intercept.Z = tk.Upstream_Z;
                    tf.Slope.X = tk.Upstream_SlopeX;
                    tf.Slope.Y = tk.Upstream_SlopeY;
                    tf.Slope.Z = 1.0;
                    tf.Weight = weightfunction(tk);
                    tf.MaxZ = tk.Upstream_Z;
                    tf.MinZ = tf.MaxZ - maxtrackextrapolation;
                }
                else
                {
                    tf.Intercept.X = tk.Downstream_PosX + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeX;
                    tf.Intercept.Y = tk.Downstream_PosY + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeY;
                    tf.Intercept.Z = tk.Downstream_Z;
                    tf.Slope.X = tk.Downstream_SlopeX;
                    tf.Slope.Y = tk.Downstream_SlopeY;
                    tf.Slope.Z = 1.0;
                    tf.Weight = weightfunction(tk);
                    tf.MinZ = tk.Downstream_Z;
                    tf.MaxZ = tf.MinZ + maxtrackextrapolation;
                }
                tf.Id = new BaseTrackIndex(tk.Id);
                vf.AddTrackFit(tf);
            }
            return vf;
        }
		
		/// <summary>
		/// Computes / refreshes the vertex coordinates.
		/// </summary>
		public void ComputeVertexCoordinates()
        {
#if true
            if (Tracks.Length == 0) throw new Exception("Can't make a vertex without tracks.");
            if (Tracks.Length == 1)
            {
                m_VertexCoordinatesUpdated = true;
                return;
            }
            VertexFit vf = new VertexFit();
            int i;
            for (i = 0; i < Tracks.Length; i++)
            {
                VertexFit.TrackFit tf = new VertexFit.TrackFit();
                Track tk = Tracks[i];
                if (tk.Upstream_Vertex == this)
                {
                    tf.Intercept.X = tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX;
                    tf.Intercept.Y = tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY;
                    tf.Intercept.Z = tk.Upstream_Z;
                    tf.Slope.X = tk.Upstream_SlopeX;
                    tf.Slope.Y = tk.Upstream_SlopeY;
                    tf.Slope.Z = 1.0;
                    tf.Weight = s_TrackWeightingFunction(tk);
                    tf.MaxZ = tk.Upstream_Z;
                    tf.MinZ = tf.MaxZ - 1e6;
                }
                else
                {
                    tf.Intercept.X = tk.Downstream_PosX + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeX;
                    tf.Intercept.Y = tk.Downstream_PosY + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeY;
                    tf.Intercept.Z = tk.Downstream_Z;
                    tf.Slope.X = tk.Downstream_SlopeX;
                    tf.Slope.Y = tk.Downstream_SlopeY;
                    tf.Slope.Z = 1.0;
                    tf.Weight = s_TrackWeightingFunction(tk);
                    tf.MinZ = tk.Downstream_Z;
                    tf.MaxZ = tf.MinZ + 1e6;
                }
                tf.Id = new BaseTrackIndex(i);
                vf.AddTrackFit(tf);
            }
            m_X = vf.X;
            m_Y = vf.Y;
            m_Z = vf.Z;
            m_AverageDistance = vf.AvgDistance;            
            m_VertexCoordinatesUpdated = true;
            m_DX = 0.0;
            m_DY = 0.0;
            for (i = 0; i < Tracks.Length; i++)
            {
                if (Tracks[i].Upstream_Vertex == this)
                {
                    Tracks[i].SetUpstreamIP(vf.TrackIP(vf.Track(i)));
                }
                else
                {
                    Tracks[i].SetDownstreamIP(vf.TrackIP(vf.Track(i)));
                }
            }
#else            
            int i, j;
			double Denom=0, Numer=0;
			double dum=0, X_Ver=0, Y_Ver=0, DY_Ver=0, Z_Ver=0, DX_Ver=0;
			double syi=0, sxi=0, syj=0, sxj=0;

			int n = Tracks.Length;
			double[] y = new double[n];
			double[] x = new double[n];
			double[] yv = new double[n];
			double[] xv = new double[n];

			for(i=0; i<n; i++)
			{ 
				if (Tracks[i].Downstream_Vertex == this)
				{
					y[i]=Tracks[i].Downstream_PosY;
					x[i]=Tracks[i].Downstream_PosX;
				}
				else
				{
					y[i]=Tracks[i].Upstream_PosY;
					x[i]=Tracks[i].Upstream_PosX;
				};
			};

			//coord longitudinali
			for(i = 0; i<n-1; i++)
				for(j = i+1; j<n; j++)
				{
					if (Tracks[i].Downstream_Vertex == this)
					{
						syi=Tracks[i].Downstream_SlopeY;
						sxi=Tracks[i].Downstream_SlopeX;
					}
					else
					{
						syi=Tracks[i].Upstream_SlopeY;
						sxi=Tracks[i].Upstream_SlopeX;
					};
					if (Tracks[j].Downstream_Vertex == this)
					{
						syj=Tracks[j].Downstream_SlopeY;
						sxj=Tracks[j].Downstream_SlopeX;
					}
					else
					{
						syj=Tracks[j].Upstream_SlopeY;
						sxj=Tracks[j].Upstream_SlopeX;
					};

					Denom += (syi - syj) *(syi - syj) + (sxi - sxj)*(sxi - sxj);
					Numer += (y[i] - y[j]) * (syi - syj) + (x[i] - x[j]) * (sxi - sxj);
				};
    
			Z_Ver = -(Numer / Denom);
    
			//coord trasverse
			for(i = 0; i<n; i++)
			{
				if (Tracks[i].Downstream_Vertex == this)
				{
					yv[i] = y[i] + Z_Ver * Tracks[i].Downstream_SlopeY;
					xv[i] = x[i] + Z_Ver * Tracks[i].Downstream_SlopeX;
				}
				else
				{
					yv[i] = y[i] + Z_Ver * Tracks[i].Upstream_SlopeY;
					xv[i] = x[i] + Z_Ver * Tracks[i].Upstream_SlopeX;
				};    
			};    
			
			Fitting.FindStatistics(yv, ref dum, ref dum, ref Y_Ver, ref DY_Ver);
			Fitting.FindStatistics(xv, ref dum, ref dum, ref X_Ver, ref DX_Ver);

			//Passaggio dei valori
			m_Z = Z_Ver;
			m_Y = Y_Ver;
			m_X = X_Ver;

			m_DY = DY_Ver;
			m_DX = DX_Ver;
			m_AverageDistance = 0;
			for (i = 0; i < Tracks.Length; i++)
			{
				double dx, dy;
				if (Tracks[i].Downstream_Vertex == this)
				{
					dx = Tracks[i].Downstream_PosX + m_Z * Tracks[i].Downstream_SlopeX - m_X;
					dy = Tracks[i].Downstream_PosY + m_Z * Tracks[i].Downstream_SlopeY - m_Y;
				}
				else
				{
					dx = Tracks[i].Upstream_PosX + m_Z * Tracks[i].Upstream_SlopeX - m_X;
					dy = Tracks[i].Upstream_PosY + m_Z * Tracks[i].Upstream_SlopeY - m_Y;
				}
				m_AverageDistance += Math.Sqrt(dx * dx + dy * dy);
			}
			m_AverageDistance /= Tracks.Length;

			m_VertexCoordinatesUpdated=true;
#endif
		}


        public override string ToString()
		{
			string text = "VERTEX INFO";
			text += "\r\nID: " + Id.ToString();
			text += "\r\nTracks: " + Length.ToString();
			text += "\r\nComment: " + ((Comment == null) ? "" : Comment);
			int i, c;
			for (i = c = 0; i < Length; i++)
				if (this[i].Upstream_Vertex == this) c++;
			text += "\r\nDownstream Tracks: " + c.ToString();
			for (i = c = 0; i < Length; i++)
				if (this[i].Downstream_Vertex == this) c++;
			text += "\r\nUpstream Tracks: " + c.ToString();
			text += "\r\nAverage Distance: " + AverageDistance.ToString("F3");
			text += "\r\nX: " + X.ToString("F3");
			text += "\r\nY: " + Y.ToString("F3");
			text += "\r\nZ: " + Z.ToString("F3");
			text += "\r\nTRACKS";
			text += "\r\nID\tUpVtx\tUpIP\tDownVtx\tDownIP\tSegments";
			for (i = 0; i < this.Length; i++)
			{
				SySal.TotalScan.Track tk = this[i];
				text += "\r\n" + tk.Id.ToString() + "\t" + ((tk.Upstream_Vertex == null) ? "-1" : tk.Upstream_Vertex.Id.ToString()) + "\t" + ((tk.Upstream_Vertex == null) ? "-1" : tk.Upstream_Impact_Parameter.ToString("F3")) + "\t" + ((tk.Downstream_Vertex == null) ? "-1" : tk.Downstream_Vertex.Id.ToString()) + "\t" + ((tk.Downstream_Vertex == null) ? "-1" : tk.Downstream_Impact_Parameter.ToString("F3")) + "\t" + tk.Length.ToString();
			}
			return text;
		}

		#region IAttributeList Members

		/// <summary>
		/// The list of the attributes of this vertex.
		/// </summary>
		protected Attribute [] m_AttributeList = new Attribute[0];

		/// <summary>
		/// Sets an attribute of the track.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be set.</param>
		/// <param name="attributevalue">the value of the attribute to be set.</param>
		public void SetAttribute(Index attributeindex, double attributevalue)
		{
			int i;
			for (i = 0; i < m_AttributeList.Length; i++)
			{
				if (m_AttributeList[i].Index.Equals(attributeindex))
				{
					m_AttributeList[i].Value = attributevalue;
					return;
				}
			}
			Attribute [] newlist = new Attribute[m_AttributeList.Length + 1];
			for (i = 0; i < m_AttributeList.Length; i++)
				newlist[i] = m_AttributeList[i];
			newlist[i] = new Attribute(attributeindex, attributevalue);
			m_AttributeList = newlist;
		}

		/// <summary>
		/// Removes an attribute from the vertex.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be removed.</param>
		public void RemoveAttribute(Index attributeindex)
		{
			int i;
			for (i = 0; i < m_AttributeList.Length && m_AttributeList[i].Index.Equals(attributeindex) == false; i++);
			if (i < m_AttributeList.Length)
			{
				int j;
				Attribute [] newlist = new Attribute[m_AttributeList.Length - 1];
				for (j = 0; j < i; j++)
					newlist[j] = m_AttributeList[j];
				for (; j < newlist.Length; j++)
					newlist[j] = m_AttributeList[j + 1];
				m_AttributeList = newlist;
			}
		}

		/// <summary>
		/// Gets an attribute of the track. An exception is thrown if the attribute is not found.
		/// </summary>
		/// <param name="attributeindex">the index of the attribute to be read.</param>
		/// <returns>the value of the attribute.</returns>
		public double GetAttribute(Index attributeindex)
		{
			int i;
			for (i = 0; i < m_AttributeList.Length; i++)
				if (m_AttributeList[i].Index.Equals(attributeindex)) return m_AttributeList[i].Value;
			throw new Exception("Attribute " + attributeindex.ToString() + " cannot be found");
		}

		/// <summary>
		/// Returns the list of the attributes.
		/// </summary>
		/// <returns>the list of the attributes of the vertex.</returns>
		public Attribute[] ListAttributes()
		{
			Attribute [] newlist = new Attribute[m_AttributeList.Length];
			int i;
			for (i = 0; i < newlist.Length; i++)
				newlist[i] = m_AttributeList[i];
			return newlist;
		}

		#endregion
	}
	#endregion

	#region Layer

	/// <summary>
	/// Alignment parameters for a layer (emulsion plate).
	/// </summary>
	public class AlignmentData : ICloneable
	{
		/// <summary>
		/// X component of the translation.
		/// </summary>
		public double TranslationX;

		/// <summary>
		/// Y component of the translation.
		/// </summary>
		public double TranslationY;

		/// <summary>
		/// Z component of the translation.
		/// </summary>
		public double TranslationZ;

		/// <summary>
		/// XX component of the affine deformation matrix.
		/// </summary>
		public double AffineMatrixXX;

		/// <summary>
		/// XY component of the affine deformation matrix.
		/// </summary>
		public double AffineMatrixXY;

		/// <summary>
		/// YX component of the affine deformation matrix.
		/// </summary>
		public double AffineMatrixYX;

		/// <summary>
		/// YY component of the affine deformation matrix.
		/// </summary>
		public double AffineMatrixYY;

		/// <summary>
		/// X slope multiplier.
		/// </summary>
		public double DShrinkX;

		/// <summary>
		/// Y slope multiplier.
		/// </summary>
		public double DShrinkY;

		/// <summary>
		/// X slope additive correction.
		/// </summary>
		public double SAlignDSlopeX;

		/// <summary>
		/// Y slope additive correction.
		/// </summary>
		public double SAlignDSlopeY;

		/// <summary>
		/// Creates a copy of this set of alignment data.
		/// </summary>
		/// <returns>a copy of the transformation.</returns>
		public object Clone()
		{
			AlignmentData A = new AlignmentData();
			A.TranslationX = TranslationX;
			A.TranslationY = TranslationY;
			A.TranslationZ = TranslationZ;
			A.AffineMatrixXX = AffineMatrixXX;
			A.AffineMatrixXY = AffineMatrixXY;
			A.AffineMatrixYX = AffineMatrixYX;
			A.AffineMatrixYY = AffineMatrixYY;
			A.DShrinkX = DShrinkX;
			A.DShrinkY = DShrinkY;
			A.SAlignDSlopeX = SAlignDSlopeX;
			A.SAlignDSlopeY = SAlignDSlopeY;
			return A;
		}

        /// <summary>
        /// Creates an object with the inverted transformation.
        /// </summary>
        /// <returns>the inverted transformation.</returns>
        public object Invert()
        {
            AlignmentData A = new AlignmentData();
            double det = 1.0 / (AffineMatrixXX * AffineMatrixYY - AffineMatrixXY * AffineMatrixYX);
            A.AffineMatrixXX = AffineMatrixYY * det;
            A.AffineMatrixYY = AffineMatrixXX * det;
            A.AffineMatrixXY = -AffineMatrixXY * det;
            A.AffineMatrixYX = -AffineMatrixYX * det;
            A.TranslationX = (AffineMatrixXY * TranslationY - AffineMatrixYY * TranslationX) * det;
            A.TranslationY = (AffineMatrixYX * TranslationX - AffineMatrixXX * TranslationY) * det;
            A.TranslationZ = -TranslationZ;
            A.DShrinkX = 1.0 / DShrinkX;
            A.DShrinkY = 1.0 / DShrinkY;
            A.SAlignDSlopeX = -SAlignDSlopeX / DShrinkX;
            A.SAlignDSlopeY = -SAlignDSlopeY / DShrinkY;
            return A;
        }
	}

	/// <summary>
	/// A layer of base tracks on an emulsion plate.
	/// Derived from SegmentList, so its segments can be accessed in an array-like fashion.
	/// </summary>
	public class Layer : SegmentList
	{
        /// <summary>
        /// Member data field on which <c>RadiationLength</c> relies.
        /// </summary>
        internal protected double m_RadiationLength = 0.0;

        /// <summary>
        /// Average radiation length in the layer.
        /// </summary>
        public double RadiationLengh
        {
            get { return m_RadiationLength; }
        }

        /// <summary>
        /// Member data field on which <c>DownstreamRadiationLength relies.</c>
        /// </summary>
        internal protected double m_DownstreamRadiationLength = 0.0;

        /// <summary>
        /// Average radiation length of the material downstream of this layer.
        /// </summary>
        public double DownstreamRadiationLength
        {
            get { return m_DownstreamRadiationLength; }
        }

        /// <summary>
        /// Member data field on which <c>UpstreamRadiationLength relies.</c>
        /// </summary>
        internal protected double m_UpstreamRadiationLength = 0.0;

        /// <summary>
        /// Average radiation length of the material upstream of this layer.
        /// </summary>
        public double UpstreamRadiationLength
        {
            get { return m_UpstreamRadiationLength; }
        }
        
        /// <summary>
		/// Signals whether the DownstreamZ property needs recomputing. Can be accessed in derived classes.
		/// </summary>
		protected bool m_DownstreamZ_Updated= false;

		/// <summary>
		/// Signals whether the UpstreamZ property needs recomputing. Can be accessed in derived classes.
		/// </summary>
		protected bool m_UpstreamZ_Updated= false;

        internal void ClearChanged() { m_DownstreamZ_Updated = m_UpstreamZ_Updated = true; }

		/// <summary>
		/// Signals whether the Z layer extents have to be extracted from segments. Can be accessed in derived classes.
		/// </summary>
		//protected bool m_UpdatingLayer= false;

		/// <summary>
		/// Protected constructor. Prevents users from creating Layers without providing consistent information. Is implicitly called in derived classes.
		/// </summary>
		protected Layer()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		/// <summary>
		/// Creates a new layer with a sequential id in the volume, a specified sheet id, a reference center, and fixed Z extents.
		/// </summary>
        /// <param name="id">the id of the layer in the sequence.</param>
        /// <param name="brickid">the brick the layer belongs to.</param>
        /// <param name="sheetid">the id of the plate.</param>
        /// <param name="side">the side identifier: 0 for base tracks, 1 for downstream, 2 for upstream.</param>
        /// <param name="Ref_Center">the reference center of the layer.</param>
        /// <param name="DownstreamZ">downstream Z extent of the layer.</param>
		/// <param name="UpstreamZ">upstream Z extent of the layer.</param>
		public Layer(int id, long brickid, int sheetid, short side, SySal.BasicTypes.Vector Ref_Center, double DownstreamZ, double UpstreamZ)
		{
			m_Id=id;
            m_BrickId = brickid;
            m_SheetId = sheetid;
            m_Side = side;
			m_DownstreamZ=DownstreamZ;
			m_UpstreamZ=UpstreamZ;
			m_DownstreamZ_Updated = true;
			m_UpstreamZ_Updated = true;
			m_RefCenter=Ref_Center;
			//m_UpdatingLayer=true;
			m_AlignmentData = new AlignmentData();
			m_AlignmentData.AffineMatrixXX = m_AlignmentData.AffineMatrixXY = m_AlignmentData.AffineMatrixYX = m_AlignmentData.AffineMatrixYY = 0.0;
			m_AlignmentData.TranslationX = m_AlignmentData.TranslationY = m_AlignmentData.TranslationZ = 0.0;
			m_AlignmentData.DShrinkX = m_AlignmentData.DShrinkY = 0.0;
			m_AlignmentData.SAlignDSlopeX = m_AlignmentData.SAlignDSlopeY = 0.0;
		}

		/// <summary>
		/// Creates a new layer with a sequential id in the volume, a specified sheet id, a reference center.
		/// The Z extents are computed when adding segments.
		/// </summary>
		/// <param name="id">the id of the layer in the sequence.</param>
        /// <param name="brickid">the brick the layer belongs to.</param>
		/// <param name="sheetid">the id of the plate.</param>
        /// <param name="side">the side identifier: 0 for base tracks, 1 for downstream, 2 for upstream.</param>
		/// <param name="Ref_Center">the reference center of the layer.</param>
		public Layer(int id, long brickid, int sheetid, short side, SySal.BasicTypes.Vector Ref_Center)
		{
			m_Id = id;
            m_BrickId = brickid;
            m_SheetId = sheetid;
            m_Side = side;
            m_RefCenter = Ref_Center;
			//m_UpdatingLayer=false;
			m_AlignmentData = new AlignmentData();
			m_AlignmentData.AffineMatrixXX = m_AlignmentData.AffineMatrixXY = m_AlignmentData.AffineMatrixYX = m_AlignmentData.AffineMatrixYY = 0.0;
			m_AlignmentData.TranslationX = m_AlignmentData.TranslationY = m_AlignmentData.TranslationZ = 0.0;
			m_AlignmentData.DShrinkX = m_AlignmentData.DShrinkY = 0.0;
			m_AlignmentData.SAlignDSlopeX = m_AlignmentData.SAlignDSlopeY = 0.0;			
		}


		/// <summary>
		/// Member data on which the Id property relies. Can be accessed by derived classes.
		/// </summary>
		protected short m_Side;

		/// <summary>
		/// Side identifier of the layer.
		/// Side = 0 -> base track layer.
		/// Side = 1 -> downstream emulsion layer.
		/// Side = 2 -> upstream emulsion layer.
		/// </summary>
		public short Side { get { return m_Side; } }

		/// <summary>
		/// Member data on which the Id property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_Id;

		/// <summary>
		/// Sequential id number of the layer in the volume.
		/// </summary>
		public int Id { get { return m_Id; } }
		internal void SetId(int id) { m_Id = id; }

		/// <summary>
		/// Member data on which the SheetId property relies. Can be accessed by derived classes.
		/// </summary>
		protected int m_SheetId;

		/// <summary>
		/// Sheet id number for the layer, following the experiment's conventions.
		/// </summary>
		public int SheetId { get { return m_SheetId; } }

        /// <summary>
        /// Member data on which the BrickId property relies. Can be accessed by derived classes.
        /// </summary>
        internal protected long m_BrickId;

        /// <summary>
        /// Sheet id number for the layer, following the experiment's conventions.
        /// </summary>
        public long BrickId { get { return m_BrickId; } }
        
        /// <summary>
		/// Member data on which the DownstreamZ property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_DownstreamZ;

		/// <summary>
		/// Downstream Z extent.
		/// </summary>
		public double DownstreamZ
		{
			get
			{
				if (!m_DownstreamZ_Updated /*|| !m_UpdatingLayer*/)
					m_DownstreamZ = UpdateDownstreamZ();
				return m_DownstreamZ;	
			}

		}

		/// <summary>
		/// Member data on which the Upstream property relies. Can be accessed by derived classes.
		/// </summary>
		protected double m_UpstreamZ;

		/// <summary>
		/// Upstream Z extent.
		/// </summary>
		public double UpstreamZ
		{
			get
			{
				if (!m_UpstreamZ_Updated /*|| !m_UpdatingLayer*/)
					m_UpstreamZ =  UpdateUpstreamZ();
				return m_UpstreamZ;	
			}
		}

        /// <summary>
        /// Transforms a vector using alignment data.
        /// </summary>
        /// <param name="inV">the input vector.</param>
        /// <returns>the vector after applying the alignment deformation.</returns>
        public SySal.BasicTypes.Vector2 ToAlignedVector(SySal.BasicTypes.Vector2 inV)
        {
            SySal.BasicTypes.Vector2 outV = new SySal.BasicTypes.Vector2();
            outV.X = m_AlignmentData.AffineMatrixXX * inV.X + m_AlignmentData.AffineMatrixXY * inV.Y;
            outV.Y = m_AlignmentData.AffineMatrixYX * inV.X + m_AlignmentData.AffineMatrixYY * inV.Y;
            return outV;
        }

        /// <summary>
        /// Transforms a slope using alignment data.
        /// </summary>
        /// <param name="inV">the input slope.</param>
        /// <returns>the slope after applying the alignment deformation.</returns>
        public SySal.BasicTypes.Vector ToAlignedSlope(SySal.BasicTypes.Vector inV)
        {
            SySal.BasicTypes.Vector outV = new SySal.BasicTypes.Vector();
            inV.X = m_AlignmentData.SAlignDSlopeX + inV.X * m_AlignmentData.DShrinkX;
            inV.Y = m_AlignmentData.SAlignDSlopeY + inV.Y * m_AlignmentData.DShrinkY;
            outV.X = m_AlignmentData.AffineMatrixXX * inV.X + m_AlignmentData.AffineMatrixXY * inV.Y;            
            outV.Y = m_AlignmentData.AffineMatrixYX * inV.X + m_AlignmentData.AffineMatrixYY * inV.Y;
            outV.Z = 1.0;
            return outV;
        }

        /// <summary>
        /// Transforms a point using alignment data.
        /// </summary>
        /// <param name="inV">the input point.</param>
        /// <returns>the point after applying the alignment transformation.</returns>
        public SySal.BasicTypes.Vector ToAlignedPoint(SySal.BasicTypes.Vector inV)
        {
            SySal.BasicTypes.Vector outV = new SySal.BasicTypes.Vector();
            outV.X = m_AlignmentData.AffineMatrixXX * (inV.X - m_RefCenter.X) + m_AlignmentData.AffineMatrixXY * (inV.Y - m_RefCenter.Y) + m_RefCenter.X + m_AlignmentData.TranslationX;
            outV.Y = m_AlignmentData.AffineMatrixYX * (inV.X - m_RefCenter.X) + m_AlignmentData.AffineMatrixYY * (inV.Y - m_RefCenter.Y) + m_RefCenter.Y + m_AlignmentData.TranslationY;
            outV.Z = inV.Z + m_AlignmentData.TranslationZ;
            return outV;
        }

        /// <summary>
        /// Transforms a vector back to the original reference using alignment data.
        /// </summary>
        /// <param name="inV">the input vector.</param>
        /// <returns>the vector transformed back to the original reference (pre-alignment).</returns>
        public SySal.BasicTypes.Vector2 ToOriginalVector(SySal.BasicTypes.Vector2 inV)
        {
            SySal.BasicTypes.Vector2 outV = new SySal.BasicTypes.Vector2();
            outV.X = m_InvAlignmentData.AffineMatrixXX * inV.X + m_InvAlignmentData.AffineMatrixXY * inV.Y;
            outV.Y = m_InvAlignmentData.AffineMatrixYX * inV.X + m_InvAlignmentData.AffineMatrixYY * inV.Y;
            return outV;
        }

        /// <summary>
        /// Transforms a point back to the original reference using alignment data.
        /// </summary>
        /// <param name="inV">the input point.</param>
        /// <returns>the point transformed back to the original reference (pre-alignment).</returns>
        public SySal.BasicTypes.Vector ToOriginalPoint(SySal.BasicTypes.Vector inV)
        {
            SySal.BasicTypes.Vector outV = new SySal.BasicTypes.Vector();
            outV.X = m_InvAlignmentData.AffineMatrixXX * (inV.X - m_RefCenter.X) + m_InvAlignmentData.AffineMatrixXY * (inV.Y - m_RefCenter.Y) + m_RefCenter.X + m_InvAlignmentData.TranslationX;
            outV.Y = m_InvAlignmentData.AffineMatrixYX * (inV.X - m_RefCenter.X) + m_InvAlignmentData.AffineMatrixYY * (inV.Y - m_RefCenter.Y) + m_RefCenter.Y + m_InvAlignmentData.TranslationY;
            outV.Z = inV.Z + m_InvAlignmentData.TranslationZ;
            return outV;
        }

        /// <summary>
        /// Transforms a slope back to the original reference using alignment data.
        /// </summary>
        /// <param name="inV">the input slope.</param>
        /// <returns>the slope transformed back to the original reference (pre-alignment).</returns>
        public SySal.BasicTypes.Vector ToOriginalSlope(SySal.BasicTypes.Vector inV)
        {
            SySal.BasicTypes.Vector outV = new SySal.BasicTypes.Vector();
            outV.X = m_InvAlignmentData.AffineMatrixXX * inV.X + m_InvAlignmentData.AffineMatrixXY * inV.Y;
            outV.Y = m_InvAlignmentData.AffineMatrixYX * inV.X + m_InvAlignmentData.AffineMatrixYY * inV.Y;
            outV.Z = 1.0;
            outV.X = m_InvAlignmentData.SAlignDSlopeX + outV.X * m_InvAlignmentData.DShrinkX;
            outV.Y = m_InvAlignmentData.SAlignDSlopeY + outV.Y * m_InvAlignmentData.DShrinkY;
            return outV;
        }


        /// <summary>
		/// Member data on which the RefCenter property relies. Can be accessed by derived classes.
		/// </summary>
		protected SySal.BasicTypes.Vector m_RefCenter;

		/// <summary>
		/// Reference center of the layer.
		/// </summary>
		public SySal.BasicTypes.Vector RefCenter
		{
			get
			{
/*				if (Segments.Length > 0)
				{
					m_RefCenter.Z = Segments[0].Info.Intercept.Z;
				}
*/				return m_RefCenter;
			}
		}

		/// <summary>
		/// Adds a segment to the layer.
		/// </summary>
		/// <param name="s"></param>
		public void AddSegment(Segment s)
		{
			int i;
			Segment[] tmp;
			tmp = Segments;
			Segments = new Segment[Segments.Length +1];
			for (i = 0; i < Segments.Length - 1; i++) Segments[i] = tmp[i];
			Segments[i] = s;
			s.SetLayerOwner(this, i);
			NotifyChanged();
		}

		/// <summary>
		/// Adds a set of segments to a layer.
		/// </summary>
		/// <param name="s"></param>
		public void AddSegments(Segment[] s)
		{
			int i;
			int n = Segments.Length;
			Segment [] tmp = Segments;
			Segments = new Segment[n + s.Length];
			for (i = 0; i < n; i++) Segments[i] = tmp[i];
			for (i = 0; i < s.Length; i++)
			{
				Segments[i + n] = s[i];
				s[i].SetLayerOwner(this, i + n);
			}
			NotifyChanged();
		}

		/// <summary>
		/// Replaces a segment in the layer.
		/// </summary>
		/// <param name="index"></param>
		/// <param name="s"></param>
		protected override void ReplaceSegment(int index, Segment s)
		{
			if (index < 0 || index > Segments.Length - 1) throw new IndexOutOfRangeException("Segment index out of bound.");
			Segments[index].SetLayerOwner(null, -1);
			Segments[index]= s;
			s.SetLayerOwner(this, index);
			NotifyChanged();
		}

		/// <summary>
		/// Notifies the layer that one or more of its layers have changed.
		/// </summary>
		public override void NotifyChanged()
		{
			//if (m_UpdatingLayer)
			//{
				m_DownstreamZ_Updated= false;
				m_UpstreamZ_Updated= false;
			//};
		}

		/// <summary>
		/// Updates the DownstreamZ property using segment information.
		/// </summary>
		/// <returns></returns>
		protected double UpdateDownstreamZ()
		{
			int n = Segments.Length;
			if(n==0) return 0;
			double[] x = new double[n];
			for(int i = 0; i<n;i++) x[i] = Segments[i].Info.TopZ;
			m_DownstreamZ_Updated= true;
			return m_DownstreamZ = Fitting.Average(x);
		}

		/// <summary>
		/// Updates the UpstreamZ property using segment information.
		/// </summary>
		/// <returns></returns>
		protected double UpdateUpstreamZ()
		{
			int n = Segments.Length;
			if(n==0) return 0;
			double[] x = new double[n];
			for(int i = 0; i<n;i++)  x[i] = Segments[i].Info.BottomZ;
			m_UpstreamZ_Updated= true;
			return m_UpstreamZ = Fitting.Average(x);
		}

		/// <summary>
		/// Member data on which the AlignmentData property relies. Can be accessed by derived classes.
		/// </summary>
		protected AlignmentData m_AlignmentData;

        /// <summary>
        /// Member data on which the InvAlignmentData property relies. Can be accessed by derived classes.
        /// </summary>
        protected AlignmentData m_InvAlignmentData;

		/// <summary>
		/// Returns a copy of the internal alignment data for the layer.
		/// </summary>
		public AlignmentData AlignData { get { return (AlignmentData)(m_AlignmentData.Clone()); } }
        /// <summary>
        /// Sets the alignment data (and their inverse).
        /// </summary>
        /// <param name="a">the set of alignment data to use.</param>
        protected internal void SetAlignmentData(AlignmentData a) { m_AlignmentData = a; m_InvAlignmentData = (AlignmentData)a.Invert(); }
	}
	#endregion

	#region Volume

	/// <summary>
	/// A TotalScan volume, containing segments, tracks, vertices and layers.
	/// </summary>
	public class Volume
	{
		/// <summary>
		/// File format information identifier. Useful for (de)serialization using files.
		/// </summary>
		protected enum FileFormatInfoType : byte { Normal = 0x49 }

		/// <summary>
		/// File format header identifier. Useful for (de)serialization using files.
		/// </summary>
		protected enum FileFormatHeader : ushort { NormalWithAttributes = 0x402, NormalDouble = 0x401, Normal = 0x400, SySal2000 = 0x300, SySal2000Old = 0x200, Old = 0x100 }

		private class TempLink
		{
			public int Layer;
			public int PosInLayer;
			public int LinkLayer;
			public int LinkPosInLayer;
			public int LinkId;

			public TempLink(int l, int p, int li) { Layer = l; PosInLayer = p; LinkId = li; LinkLayer = -1; LinkPosInLayer = -1; }
		}

        /// <summary>
        /// A list of showers in a TotalScan volume.
        /// </summary>
        public class ShowerList
        {
            /// <summary>
            /// Member data holding the list of showers.
            /// </summary>
            protected internal Shower[] Items;
            /// <summary>
            /// Accesses the list of showers in an array-like fashion.
            /// </summary>
            public Shower this[int index] { get { return Items[index]; } }
            /// <summary>
            /// The number of showers in the volume.
            /// </summary>
            public int Length { get { return Items.Length; } }
        }

		/// <summary>
		/// A list of tracks in a TotalScan volume.
		/// </summary>
		public class TrackList
		{
			/// <summary>
			/// Member data holding the list of tracks.
			/// </summary>
			protected internal Track [] Items;
			/// <summary>
			/// Accesses the list of tracks in an array-like fashion.
			/// </summary>
			public Track this[int index] { get { return Items[index]; } }
			/// <summary>
			/// The number of tracks in the volume.
			/// </summary>
			public int Length { get { return Items.Length; } }
		}

		/// <summary>
		/// A list of layers in a TotalScan volume.
		/// </summary>
		public class LayerList
		{
			/// <summary>
			/// Member data holding the list of layers.
			/// </summary>
			protected internal Layer [] Items;
			/// <summary>
			/// Accesses the list of layers in an array-like fashion.
			/// </summary>
			public Layer this[int index] { get { return Items[index]; } }
			/// <summary>
			/// The number of layers in the volume.
			/// </summary>
			public int Length { get { return Items.Length; } }
		}

		/// <summary>
		/// A list of vertices in a TotalScan volume.
		/// </summary>
		public class VertexList
		{
			/// <summary>
			/// Member data holding the list of Vertices.
			/// </summary>
			protected internal Vertex [] Items;
			/// <summary>
			/// Accesses the list of vertices in an array-like fashion.
			/// </summary>
			public Vertex this[int index] { get { return Items[index]; } }
			/// <summary>
			/// The number of vertices in the volume.
			/// </summary>
			public int Length { get { return Items.Length; } }
		}

		/// <summary>
		/// Protected constructor for derived classes.
		/// </summary>
		protected Volume() {}

        private void NormalDoubleWithAttributesFormatRead(System.IO.BinaryReader b)
        {
            m_Id.Part0 = b.ReadInt32();
            m_Id.Part1 = b.ReadInt32();
            m_Id.Part2 = b.ReadInt32();
            m_Id.Part3 = b.ReadInt32();
            m_Extents.MinX = b.ReadDouble();
            m_Extents.MinY = b.ReadDouble();
            m_Extents.MinZ = b.ReadDouble();
            m_Extents.MaxX = b.ReadDouble();
            m_Extents.MaxY = b.ReadDouble();
            m_Extents.MaxZ = b.ReadDouble();
            b.ReadBytes(96); // reserved bytes
            int attributeindexsignature = b.ReadInt32();
            int attributeindexsize = b.ReadInt32();
            Index.IndexFactory afc = Index.GetFactory(attributeindexsignature);
            if (afc != null && afc.Size != attributeindexsize) throw new Exception("Attribute index size " + attributeindexsize + " does not match expected size = " + afc.Size + " from index factory " + afc.GetType().Name);
            int sheetcount = b.ReadInt32();
            int trackcount = b.ReadInt32();
            int vertexcount = b.ReadInt32();
            int segmentindexsignature = b.ReadInt32();
            int segmentindexsize = b.ReadInt32();
            Index.IndexFactory ifc = Index.GetFactory(segmentindexsignature);
            if (ifc != null && ifc.Size != segmentindexsize) throw new Exception("Segment index size " + segmentindexsize + " does not match expected size = " + ifc.Size + " from index factory " + ifc.GetType().Name);
            m_RefCenter.X = b.ReadDouble();
            m_RefCenter.Y = b.ReadDouble();
            m_RefCenter.Z = b.ReadDouble();
            m_Layers = new LayerList();
            m_Tracks = new TrackList();
            m_Vertices = new VertexList();
            m_Showers = new ShowerList();
            m_Layers.Items = new Layer[sheetcount];
            m_Tracks.Items = new Track[trackcount];
            m_Vertices.Items = new Vertex[vertexcount];
            int i, j, h, k;
            for (i = 0; i < sheetcount; i++)
            {
                j = b.ReadInt32();
                double DownstreamExt = b.ReadDouble();
                b.ReadDouble(); b.ReadDouble();
                double UpstreamExt = b.ReadDouble();
                double RefZ = b.ReadDouble();
                SySal.BasicTypes.Vector V;
                V.X = m_RefCenter.X;
                V.Y = m_RefCenter.Y;
                V.Z = RefZ;
                AlignmentData a = new AlignmentData();

                a.AffineMatrixXX = b.ReadDouble();
                a.AffineMatrixXY = b.ReadDouble();
                a.AffineMatrixYX = b.ReadDouble();
                a.AffineMatrixYY = b.ReadDouble();
                a.TranslationX = b.ReadDouble();
                a.TranslationY = b.ReadDouble();
                a.TranslationZ = b.ReadDouble();
                b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes
                a.SAlignDSlopeX = b.ReadDouble();
                a.SAlignDSlopeY = b.ReadDouble();
                a.DShrinkX = b.ReadDouble();
                a.DShrinkY = b.ReadDouble();
                Layer l = m_Layers.Items[i] = new Layer(i, b.ReadInt64(),
                    (j < 0) ? (-j >> 16) : j,
                    (short)((j >= 0) ? 0 : (-j & 0xffff)),
                    V, DownstreamExt, UpstreamExt);
                l.m_RadiationLength = b.ReadDouble();
                l.m_DownstreamRadiationLength = b.ReadDouble();
                l.m_UpstreamRadiationLength = b.ReadDouble();

                l.SetAlignmentData(a);

                int segcount;
                segcount = b.ReadInt32();
                b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); // skip reserved bytes

                Segment[] segs = new Segment[segcount];

                for (j = 0; j < segcount; j++)
                {
                    MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                    //info.AreaSum = 0;
                    //info.Field = 0;
                    //info.Count = (ushort)b.ReadUInt32();
                    info.Count = (ushort)b.ReadUInt16();
                    info.AreaSum = (ushort)b.ReadUInt16();
                    info.Field = 0;
                    info.Intercept.X = b.ReadDouble();
                    info.Intercept.Y = b.ReadDouble();
                    info.Intercept.Z = b.ReadDouble();
                    info.Slope.X = b.ReadDouble();
                    info.Slope.Y = b.ReadDouble();
                    info.Slope.Z = b.ReadDouble();
                    info.Sigma = b.ReadDouble();
                    info.TopZ = b.ReadDouble();
                    info.BottomZ = b.ReadDouble();
                    if (ifc == null)
                    {
                        b.ReadBytes(segmentindexsize);
                        segs[j] = new Segment(info, new NullIndex());
                    }
                    else
                    {
                        segs[j] = new Segment(info, ifc.Reader(b));
                    }
                }
                l.AddSegments(segs);
                l.ClearChanged();
            }

            for (i = 0; i < trackcount; i++)
            {
                Track t = m_Tracks.Items[i] = new Track(i);
                k = b.ReadInt32();
                while (k-- > 0)
                    t.AddSegment(m_Layers.Items[b.ReadInt32()][b.ReadInt32()]);
                b.ReadInt32(); b.ReadInt32(); // skip vertex info; it will be recovered from vertex section
                //b.ReadInt32(); // skip flags
                if (b.ReadInt32() <= 0) t.Comment = null;
                else t.Comment = b.ReadString();
                SySal.BasicTypes.Vector2 ups, upp, dws, dwp;
                dwp.X = b.ReadDouble();
                dwp.Y = b.ReadDouble();
                dws.X = b.ReadDouble();
                dws.Y = b.ReadDouble();
                upp.X = b.ReadDouble();
                upp.Y = b.ReadDouble();
                ups.X = b.ReadDouble();
                ups.Y = b.ReadDouble();
                t.SetSlopeAndPos(dws.X, dws.Y, dwp.X, dwp.Y, ups.X, ups.Y, upp.X, upp.Y);
                int attrnum = b.ReadInt32();
                while (attrnum-- > 0)
                {
                    if (afc == null)
                    {
                        b.ReadBytes(attributeindexsize);
                        b.ReadDouble();
                    }
                    else
                    {
                        t.SetAttribute(afc.Reader(b), b.ReadDouble());
                    }
                }
            }

            for (i = 0; i < vertexcount; i++)
            {
                Vertex v = m_Vertices.Items[i] = new Vertex(i);
                h = b.ReadInt32();
                SySal.BasicTypes.Vector p;
                SySal.BasicTypes.Vector2 d;
                p.X = b.ReadDouble();
                p.Y = b.ReadDouble();
                p.Z = b.ReadDouble();
                double avgd = b.ReadDouble();
                for (j = 0; j < h; j++)
                {
                    k = b.ReadInt32();
                    bool isupstream = b.ReadBoolean();
                    if (isupstream) m_Tracks.Items[k].SetUpstreamVertex(v);
                    else m_Tracks.Items[k].SetDownstreamVertex(v);
                    v.AddTrack(m_Tracks.Items[k], isupstream);
                    b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); // skip reserved fields
                }
                d.X = b.ReadDouble();
                d.Y = b.ReadDouble();
                b.ReadDouble(); b.ReadDouble(); // skip reserved fields
                v.SetPosDeltas(p.X, p.Y, p.Z, d.X, d.Y, avgd);
                int attrnum = b.ReadInt32();
                while (attrnum-- > 0)
                {
                    if (afc == null)
                    {
                        b.ReadBytes(attributeindexsize);
                        b.ReadDouble();
                    }
                    else
                    {
                        v.SetAttribute(afc.Reader(b), b.ReadDouble());
                    }
                }
            }
        }

		private void NormalDoubleFormatRead(System.IO.BinaryReader b)
		{
			m_Id.Part0 = b.ReadInt32();
			m_Id.Part1 = b.ReadInt32();
			m_Id.Part2 = b.ReadInt32();
			m_Id.Part3 = b.ReadInt32();
			m_Extents.MinX = b.ReadDouble();
			m_Extents.MinY = b.ReadDouble();
			m_Extents.MinZ = b.ReadDouble();
			m_Extents.MaxX = b.ReadDouble();
			m_Extents.MaxY = b.ReadDouble();
			m_Extents.MaxZ = b.ReadDouble();
			b.ReadBytes(104); // reserved bytes
			int sheetcount = b.ReadInt32();
			int trackcount = b.ReadInt32();
			int vertexcount = b.ReadInt32();
			int maxtracksinsegment = b.ReadInt32(); // for back compatibility: should be 1 normally
			m_RefCenter.X = b.ReadDouble();
			m_RefCenter.Y = b.ReadDouble();
			m_RefCenter.Z = b.ReadDouble();
			m_Layers = new LayerList();
			m_Tracks = new TrackList();
			m_Vertices = new VertexList();
			m_Layers.Items = new Layer[sheetcount];
			m_Tracks.Items = new Track[trackcount];
			m_Vertices.Items = new Vertex[vertexcount];
			int i, j, h, k;
			for (i = 0; i < sheetcount; i++)
			{
				j = b.ReadInt32();
				double DownstreamExt = b.ReadDouble();
				b.ReadDouble(); b.ReadDouble();
				double UpstreamExt = b.ReadDouble();
				double RefZ = b.ReadDouble();
				SySal.BasicTypes.Vector V;
				V.X = m_RefCenter.X;
				V.Y = m_RefCenter.Y;
				V.Z = RefZ;
				AlignmentData a = new AlignmentData();

				a.AffineMatrixXX = b.ReadDouble();
				a.AffineMatrixXY = b.ReadDouble();
				a.AffineMatrixYX = b.ReadDouble();
				a.AffineMatrixYY = b.ReadDouble();
				a.TranslationX = b.ReadDouble();
				a.TranslationY = b.ReadDouble();
				a.TranslationZ = b.ReadDouble();
				b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes
				a.SAlignDSlopeX = b.ReadDouble();
				a.SAlignDSlopeY = b.ReadDouble();
				a.DShrinkX = b.ReadDouble();
				a.DShrinkY = b.ReadDouble();
                Layer l = m_Layers.Items[i] = new Layer(i, b.ReadInt64(),
                    (j < 0) ? (-j >> 16) : j,
                    (short)((j >= 0) ? 0 : (-j & 0xffff)), V, DownstreamExt, UpstreamExt);
                b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes

				l.SetAlignmentData(a);
				
				int segcount;
				segcount = b.ReadInt32();
				b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); // skip reserved bytes

				Segment [] segs = new Segment[segcount];

				for (j = 0; j < segcount; j++)
				{
					MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
					if (maxtracksinsegment > 0)
					{
						h = b.ReadInt32();
						for (k = 1; k < maxtracksinsegment; k++) b.ReadUInt32();
					}
					else h = -1;
					//info.AreaSum = 0;
					//info.Field = 0;
					//info.Count = (ushort)b.ReadUInt32();
					info.Count = (ushort)b.ReadUInt16();
					info.AreaSum = (ushort)b.ReadUInt16();
					info.Field = 0;					
					info.Intercept.X = b.ReadDouble();
					info.Intercept.Y = b.ReadDouble();
					info.Intercept.Z = b.ReadDouble();
					info.Slope.X = b.ReadDouble();
					info.Slope.Y = b.ReadDouble();
					info.Slope.Z = b.ReadDouble();
					info.Sigma = b.ReadDouble();
					info.TopZ = b.ReadDouble();
					info.BottomZ = b.ReadDouble();
					segs[j] = new Segment(info, new BaseTrackIndex(h));
				}
				l.AddSegments(segs);
			}

			for (i = 0; i < trackcount; i++)
			{
				Track t = m_Tracks.Items[i] = new Track(i);
				k = b.ReadInt32();
				while (k-- > 0)
					t.AddSegment(m_Layers.Items[b.ReadInt32()][b.ReadInt32()]);
				b.ReadInt32(); b.ReadInt32(); // skip vertex info; it will be recovered from vertex section
				//b.ReadInt32(); // skip flags
				if (b.ReadInt32() <= 0) t.Comment = null;
				else t.Comment = b.ReadString();
				SySal.BasicTypes.Vector2 ups, upp, dws, dwp;
				dwp.X = b.ReadDouble();
				dwp.Y = b.ReadDouble();
				dws.X = b.ReadDouble();
				dws.Y = b.ReadDouble();
				upp.X = b.ReadDouble();
				upp.Y = b.ReadDouble();
				ups.X = b.ReadDouble();
				ups.Y = b.ReadDouble();
				t.SetSlopeAndPos(dws.X, dws.Y, dwp.X, dwp.Y, ups.X, ups.Y, upp.X, upp.Y);
			}

			for (i = 0; i < vertexcount; i++)
			{
				Vertex v = m_Vertices.Items[i] = new Vertex(i);
				h = b.ReadInt32();
				SySal.BasicTypes.Vector p;
				SySal.BasicTypes.Vector2 d;
				p.X = b.ReadDouble();
				p.Y = b.ReadDouble();
				p.Z = b.ReadDouble();
				double avgd = b.ReadDouble();
				for (j = 0; j < h; j++)
				{
					k = b.ReadInt32();
					bool isupstream = b.ReadBoolean();
					if (isupstream) m_Tracks.Items[k].SetUpstreamVertex(v);
					else m_Tracks.Items[k].SetDownstreamVertex(v);
					v.AddTrack(m_Tracks.Items[k], isupstream);
					b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); // skip reserved fields
				}
				d.X = b.ReadDouble();
				d.Y = b.ReadDouble();					
				b.ReadDouble(); b.ReadDouble(); // skip reserved fields
				v.SetPosDeltas(p.X, p.Y, p.Z, d.X, d.Y, avgd);
			}
		}

		private void NormalDoubleFormatWithAttributesWrite(System.IO.BinaryWriter b)
		{
			int i, j;
			b.Write((byte)FileFormatInfoType.Normal);
			b.Write((ushort)FileFormatHeader.NormalWithAttributes);
			b.Write(m_Id.Part0);
			b.Write(m_Id.Part1);
			b.Write(m_Id.Part2);
			b.Write(m_Id.Part3);
			b.Write(m_Extents.MinX);
			b.Write(m_Extents.MinY);
			b.Write(m_Extents.MinZ);
			b.Write(m_Extents.MaxX);
			b.Write(m_Extents.MaxY);
			b.Write(m_Extents.MaxZ);
			for (i = 0; i < 96; i++) b.Write((byte)0);
			Index.IndexFactory afc = null;
			foreach (Track t in m_Tracks.Items)
			{
				SySal.TotalScan.Attribute [] attrs = t.ListAttributes();
				if (attrs != null && attrs.Length > 0)
				{
					afc = attrs[0].Index.Factory;
					break;
				}
			}
			if (afc == null)
			{
				foreach (Vertex v in m_Vertices.Items)
				{
					SySal.TotalScan.Attribute [] attrs = v.ListAttributes();
					if (attrs != null && attrs.Length > 0)
					{
						afc = attrs[0].Index.Factory;
						break;
					}
				}
			}
			b.Write((afc != null) ? afc.Signature : NullIndex.Signature);
			b.Write((afc != null) ? afc.Size : 0);
			b.Write(m_Layers.Items.Length);
			b.Write(m_Tracks.Items.Length);
			b.Write(m_Vertices.Items.Length);
			Index.IndexFactory ifc = null;
			foreach (Layer l in m_Layers.Items)
			{
				for (i = 0; i < l.Length; i++)
				{
					ifc = l[i].Index.Factory;
					if (ifc != null) break;
				}
				if (ifc != null) break;
			}
			b.Write((ifc != null) ? ifc.Signature : NullIndex.Signature);
			b.Write((ifc != null) ? ifc.Size : 0);
			b.Write(m_RefCenter.X);
			b.Write(m_RefCenter.Y);
			b.Write(m_RefCenter.Z);
			
			foreach (Layer l in m_Layers.Items)
			{
                if (l.Side == 0) b.Write(l.SheetId);
                else b.Write(-((l.SheetId << 16) + l.Side));                
				b.Write(l.DownstreamZ); b.Write(l.DownstreamZ); 
				b.Write(l.UpstreamZ); b.Write(l.UpstreamZ);
				b.Write(l.RefCenter.Z);
				
				AlignmentData a = l.AlignData;
				b.Write(a.AffineMatrixXX);
				b.Write(a.AffineMatrixXY);
				b.Write(a.AffineMatrixYX);
				b.Write(a.AffineMatrixYY);
				b.Write(a.TranslationX);
				b.Write(a.TranslationY);
				b.Write(a.TranslationZ);
				b.Write(0.0); b.Write(0.0); b.Write(0.0); b.Write(0.0); // skip reserved bytes
				b.Write(a.SAlignDSlopeX);
				b.Write(a.SAlignDSlopeY);
				b.Write(a.DShrinkX);
				b.Write(a.DShrinkY);
                b.Write(l.BrickId);
				b.Write(l.RadiationLengh); b.Write(l.DownstreamRadiationLength); b.Write(l.UpstreamRadiationLength);

				b.Write(l.Length);
				b.Write(0); b.Write(0); b.Write(0); // skip reserved bytes

				for (j = 0; j < l.Length; j++)
				{
					Segment s = l[j];
					MIPEmulsionTrackInfo info = s.Info;
					
					b.Write((ushort)info.Count);
					b.Write((ushort)info.AreaSum);
					b.Write(info.Intercept.X);
					b.Write(info.Intercept.Y);
					b.Write(info.Intercept.Z);
					b.Write(info.Slope.X);
					b.Write(info.Slope.Y);
					b.Write(info.Slope.Z);
					b.Write(info.Sigma);
					b.Write(info.TopZ);
					b.Write(info.BottomZ);
					if (s.Index.Factory.Signature == ifc.Signature) s.Index.Write(b);
					else throw new Exception("Only one segment index type can be allowed in stream serialization.");
				}
			}

			foreach (Track t in m_Tracks.Items)
			{
				b.Write(t.Length);
				for (i = 0; i < t.Length; i++)
				{
					b.Write(t[i].LayerOwner.Id);
					b.Write(t[i].PosInLayer);
				}
				b.Write((t.Upstream_Vertex != null) ? t.Upstream_Vertex.Id : -1);
				b.Write((t.Downstream_Vertex != null) ? t.Downstream_Vertex.Id : -1);
				if (t.Comment == null) b.Write(-1);
				else if (t.Comment.Length == 0) b.Write(0);
				else
				{
					b.Write(t.Comment.Length);
					b.Write(t.Comment);
				}
				b.Write(t.Downstream_PosX);
				b.Write(t.Downstream_PosY);
				b.Write(t.Downstream_SlopeX);
				b.Write(t.Downstream_SlopeY);
				b.Write(t.Upstream_PosX);
				b.Write(t.Upstream_PosY);
				b.Write(t.Upstream_SlopeX);
				b.Write(t.Upstream_SlopeY);
				Attribute [] attrs = t.ListAttributes();
				b.Write(attrs.Length);
				for (i = 0; i < attrs.Length; i++)
				{
					attrs[i].Index.Write(b);
					b.Write(attrs[i].Value);
				}
			}

			foreach (Vertex v in m_Vertices.Items)
			{
				b.Write(v.Length);
				b.Write(v.X);
				b.Write(v.Y);
				b.Write(v.Z);
				b.Write(v.AverageDistance);
				for (j = 0; j < v.Length; j++)
				{
					b.Write(v[j].Id);
					b.Write((bool)(v[j].Upstream_Vertex == v));
					b.Write(0); b.Write(0); b.Write(0); b.Write(0); // skip reserved fields 
				}
				b.Write(v.DX);
				b.Write(v.DY);
				b.Write(0.0); b.Write(0.0); // skip reserved fields
				Attribute [] attrs = v.ListAttributes();
				b.Write(attrs.Length);
				for (j = 0; j < attrs.Length; j++)
				{
					attrs[j].Index.Write(b);
					b.Write(attrs[j].Value);
				}
			}		
		}

		private void IntToIndex(int s, out int l, out int p)
		{
			int i;
			for (i = 0; i < m_Layers.Items.Length && s >= m_Layers.Items[i].Length; i++) s -= m_Layers.Items[i].Length;
			l = i;
			p = s;
		}

		private void OldFormatsRead(FileFormatHeader headerformat, System.IO.BinaryReader b)
		{
			m_Id.Part0 = b.ReadInt32();
			m_Id.Part1 = b.ReadInt32();
			m_Id.Part2 = b.ReadInt32();
			m_Id.Part3 = b.ReadInt32();
			b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); // skip the old Interesting Position
			if (headerformat == FileFormatHeader.Normal || headerformat == FileFormatHeader.SySal2000)
			{
				m_Extents.MinX = b.ReadSingle();
				m_Extents.MinY = b.ReadSingle();
				m_Extents.MinZ = b.ReadSingle();
				m_Extents.MaxX = b.ReadSingle();
				m_Extents.MaxY = b.ReadSingle();
				m_Extents.MaxZ = b.ReadSingle();
				b.ReadBytes(104); // reserved bytes
			}
			else b.ReadBytes(128); //reserved bytes
			int downstreamsheet = b.ReadInt32(); // for back compatibility
			int sheetcount = b.ReadInt32();
			int trackcount = b.ReadInt32();
			int vertexcount = b.ReadInt32();
			int maxtracksinsegment = b.ReadInt32(); // for back compatibility
			m_RefCenter.X = b.ReadSingle();
			m_RefCenter.Y = b.ReadSingle();
			m_RefCenter.Z = b.ReadSingle();
			m_Layers = new LayerList();
			m_Tracks = new TrackList();
			m_Vertices = new VertexList();
			m_Layers.Items = new Layer[sheetcount];
			m_Tracks.Items = new Track[trackcount];
			m_Vertices.Items = new Vertex[vertexcount];
			System.Collections.ArrayList upstreamlinks = new System.Collections.ArrayList();
			int i, j, h, k, ll;
			int [][] rawtrackcount = new int [sheetcount][];			
			for (i = 0; i < sheetcount; i++)
			{
				j = b.ReadInt32();
				double DownstreamExt = b.ReadSingle();
				b.ReadSingle(); b.ReadSingle();
				double UpstreamExt = b.ReadSingle();
				double RefZ = b.ReadSingle();
				SySal.BasicTypes.Vector V = new SySal.BasicTypes.Vector();
				V.X = m_RefCenter.X;
				V.Y = m_RefCenter.Y;
				V.Z = RefZ;
				Layer l = m_Layers.Items[i] = new Layer(i, 0, j, 0, V, DownstreamExt, UpstreamExt);
				b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); // skip global alignment
				b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); // skip the dummy side
				AlignmentData a = new AlignmentData();

				a.AffineMatrixXX = b.ReadDouble();
				a.AffineMatrixXY = b.ReadDouble();
				a.AffineMatrixYX = b.ReadDouble();
				a.AffineMatrixYY = b.ReadDouble();
				a.TranslationX = b.ReadDouble();
				a.TranslationY = b.ReadDouble();
				a.TranslationZ = b.ReadDouble();
				b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes
				a.SAlignDSlopeX = b.ReadDouble();
				a.SAlignDSlopeY = b.ReadDouble();
				a.DShrinkX = b.ReadDouble();
				a.DShrinkY = b.ReadDouble();
				b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes

				/* skip dummy side */
				b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); 
				b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); 
				b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();
				/* end skip */

				if (headerformat != FileFormatHeader.Old)
				{
					AlignmentData aa = new AlignmentData();
					aa.AffineMatrixXX = b.ReadDouble();
					aa.AffineMatrixXY = b.ReadDouble();
					aa.AffineMatrixYX = b.ReadDouble();
					aa.AffineMatrixYY = b.ReadDouble();
					aa.TranslationX = b.ReadDouble();
					aa.TranslationY = b.ReadDouble();
					aa.TranslationZ = b.ReadDouble();
					b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes
					aa.SAlignDSlopeX = b.ReadDouble();
					aa.SAlignDSlopeY = b.ReadDouble();
					aa.DShrinkX = b.ReadDouble();
					aa.DShrinkY = b.ReadDouble();
					b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();  // reserved bytes

					/* skip dummy side */
					b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); 
					b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); 
					b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble(); b.ReadDouble();
					/* end skip */

					/* the new "aa" transformation should be composed with the "a" transformation... 
					 * but for newly produced files we plan to store only the first transformation, and the second will be "identity"...
					 * */
				}

				l.SetAlignmentData(a);
				
				rawtrackcount[i] = new int[2];
				int [] segcount = new int[2];
				rawtrackcount[i][0] = b.ReadInt32();
				rawtrackcount[i][1] = b.ReadInt32();
				segcount[0] = b.ReadInt32();
				segcount[1] = b.ReadInt32();
				b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); b.ReadInt32();  // skip reserved bytes

				if (rawtrackcount[i][1] != 0 || segcount[1] != 0) throw new System.Exception("Dummy side is not empty.");

				for (j = 0; j < rawtrackcount[i][0]; j++)
				{
					b.ReadUInt32(); b.ReadUInt32();
					b.ReadSingle();	b.ReadSingle(); b.ReadSingle();
					b.ReadSingle();	b.ReadSingle(); b.ReadSingle();
					b.ReadSingle(); b.ReadSingle(); b.ReadSingle();
				}

				Segment [] segs = new Segment[segcount[0]];

				for (j = 0; j < segcount[0]; j++)
				{
					MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
					if (maxtracksinsegment > 0)
					{
						h = b.ReadInt32();
						for (k = 1; k < maxtracksinsegment; k++) b.ReadUInt32();
						if (headerformat != FileFormatHeader.Normal)
							for (k = 0; h >= rawtrackcount[i][0]; k++)
								h -= rawtrackcount[i][0];
					}
					else h = -1;
					if ((k = b.ReadInt32()) >= 0) upstreamlinks.Add(new TempLink(i, j, k));
					b.ReadInt32(); // skip downstream link
					info.AreaSum = 0;
					info.Field = 0;
					info.Count = (ushort)b.ReadUInt32();
					info.Intercept.X = b.ReadSingle();
					info.Intercept.Y = b.ReadSingle();
					info.Intercept.Z = b.ReadSingle();
					info.Slope.X = b.ReadSingle();
					info.Slope.Y = b.ReadSingle();
					info.Slope.Z = b.ReadSingle();
					info.Sigma = b.ReadSingle();
					info.TopZ = b.ReadSingle();
					info.BottomZ = b.ReadSingle();
					segs[j] = new Segment(info, new BaseTrackIndex(h));
				}
				l.AddSegments(segs);
			}

			for (i = 0; i < upstreamlinks.Count; i++)
			{
				TempLink tl = (TempLink)upstreamlinks[i];
				IntToIndex(tl.LinkId, out tl.LinkLayer, out tl.LinkPosInLayer);
			}

			for (i = 0; i < trackcount; i++)
			{
				Track t = m_Tracks.Items[i] = new Track(i);
				k = b.ReadInt32();
				b.ReadInt32(); // skip upstream end
				j = b.ReadInt32();
				IntToIndex(j, out ll, out h);
				do
				{
					t.AddSegment(m_Layers.Items[ll][h]);
					for (j = 0; j < upstreamlinks.Count; j++)
					{
						TempLink tl = (TempLink)upstreamlinks[j];
						if (tl.Layer == ll && tl.PosInLayer == h)
						{
							ll = tl.LinkLayer;
							h = tl.LinkPosInLayer;
							break;
						}
					}
				}
				while (j < upstreamlinks.Count);

				b.ReadInt32(); b.ReadInt32(); // skip vertex info; it will be recovered from vertex section
				b.ReadInt32(); // skip flags
				b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); 
				b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); 
				b.ReadSingle(); b.ReadSingle(); b.ReadSingle(); // skip global data
				SySal.BasicTypes.Vector2 ups, upp, dws, dwp;
				dwp.X = b.ReadSingle();
				dwp.Y = b.ReadSingle();
				dws.X = b.ReadSingle();
				dws.Y = b.ReadSingle();
				upp.X = b.ReadSingle();
				upp.Y = b.ReadSingle();
				ups.X = b.ReadSingle();
				ups.Y = b.ReadSingle();
				if (headerformat != FileFormatHeader.Old)
				{
					if (headerformat != FileFormatHeader.Normal)
					{
						dwp.X -= dws.X * RefCenter.Z;
						dwp.Y -= dws.Y * RefCenter.Z;
						upp.X -= ups.X * RefCenter.Z;
						upp.Y -= ups.Y * RefCenter.Z;
					}
					t.SetSlopeAndPos(dws.X, dws.Y, dwp.X, dwp.Y, ups.X, ups.Y, upp.X, upp.Y);
				}				
			}

			for (i = 0; i < vertexcount; i++)
			{
				Vertex v = m_Vertices.Items[i] = new Vertex(i);
				h = b.ReadInt32();
				SySal.BasicTypes.Vector p;
				SySal.BasicTypes.Vector2 d;
				p.X = b.ReadSingle();
				p.Y = b.ReadSingle();
				p.Z = b.ReadSingle();
				float avgd = b.ReadSingle();
				for (j = 0; j < h; j++)
				{
					k = b.ReadInt32();
					v.AddTrack(m_Tracks.Items[k], b.ReadBoolean());
					b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); b.ReadInt32(); // skip reserved fields
				}
				if (headerformat == FileFormatHeader.Normal)
				{
					d.X = b.ReadSingle();
					d.Y = b.ReadSingle();					
				}
				else
				{
					d.X = d.Y = 0.0f;
					b.ReadSingle(); b.ReadSingle();
				}
				b.ReadSingle(); b.ReadSingle(); // skip reserved fields
				v.SetPosDeltas(p.X, p.Y, p.Z, d.X, d.Y, avgd);
			}
		}


		/// <summary>
		/// Restores a TotalScan volume from a stream.
		/// </summary>
		/// <param name="r"></param>
		public Volume(System.IO.Stream r) 
		{
			System.IO.BinaryReader b = new System.IO.BinaryReader(r);
			FileFormatInfoType infotype = (FileFormatInfoType)b.ReadByte();
			FileFormatHeader headerformat = (FileFormatHeader)b.ReadUInt16();
			if (infotype != FileFormatInfoType.Normal || 
				(headerformat != FileFormatHeader.NormalWithAttributes &&
				headerformat != FileFormatHeader.NormalDouble && headerformat != FileFormatHeader.Normal && headerformat != FileFormatHeader.SySal2000 &&
				headerformat != FileFormatHeader.SySal2000Old && headerformat != FileFormatHeader.Old))
				throw new System.Exception("Unknown format");
			if (headerformat == FileFormatHeader.NormalWithAttributes) NormalDoubleWithAttributesFormatRead(b);
			else if (headerformat == FileFormatHeader.NormalDouble) NormalDoubleFormatRead(b);
			else OldFormatsRead(headerformat, b);
		}

		/// <summary>
		/// Saves a TotalScan volume to a stream.
		/// </summary>
		/// <param name="w">the stream to save to.</param>
		public virtual void Save(System.IO.Stream w) 
		{
			System.IO.BinaryWriter b = new System.IO.BinaryWriter(w);
			NormalDoubleFormatWithAttributesWrite(b);
		}

		/// <summary>
		/// Converts the Track List into an Array of Tracks for external uses.
		/// </summary>
		/// <returns></returns>
		public Track[] GetTracks()
		{
			return (Track[])m_Tracks.Items.Clone();
		}

		/// <summary>
		/// Member data on which the Tracks property relies. Can be accessed by derived classes.
		/// </summary>
		protected TrackList m_Tracks;
		/// <summary>
		/// Accesses the list of the tracks.
		/// </summary>
		public TrackList Tracks { get { return m_Tracks; } }

        /// <summary>
        /// Member data on which the Showers property relies. Can be accessed by derived classes.
        /// </summary>
        protected ShowerList m_Showers;
        /// <summary>
        /// Accesses the list of the showers.
        /// </summary>
        public ShowerList Showers { get { return m_Showers; } }

        /// <summary>
		/// Member data on which the Vertices property relies. Can be accessed by derived classes.
		/// </summary>
		protected VertexList m_Vertices;
		/// <summary>
		/// Accesses the list of the vertices.
		/// </summary>
		public VertexList Vertices { get { return m_Vertices; } }

		/// <summary>
		/// Member data on which the Layers property relies. Can be accessed by derived classes.
		/// </summary>
		protected LayerList m_Layers;
		/// <summary>
		/// Accesses the list of the layers.
		/// </summary>
		public LayerList Layers { get { return m_Layers; } }

		/// <summary>
		/// Member data on which the Id property relies. Can be accessed by derived classes.
		/// </summary>
		protected SySal.BasicTypes.Identifier m_Id;
		/// <summary>
		/// Volume Identifier.
		/// </summary>
		public SySal.BasicTypes.Identifier Id { get { return m_Id; } }

		/// <summary>
		/// Member data on which the Extents property relies. Can be accessed by derived classes.
		/// </summary>
		protected SySal.BasicTypes.Cuboid m_Extents;
		/// <summary>
		/// Extents of the volume (invalid if read from old files).
		/// </summary>
		public SySal.BasicTypes.Cuboid Extents { get { return m_Extents; } }

		/// <summary>
		/// Member data on which the RefCenter property relies. Can be accessed by derived classes.
		/// </summary>
		protected SySal.BasicTypes.Vector m_RefCenter;
		/// <summary>
		/// Reference center of the volume.
		/// </summary>
		public SySal.BasicTypes.Vector RefCenter { get { return m_RefCenter; } }

		/// <summary>
		/// Resets the id numbers of all objects in the volume to the right sequence for serialization.
		/// </summary>
		public void ResetIds()
		{
			int i, l;
			l = m_Tracks.Items.Length;
			for (i = 0; i < l; i++)
				m_Tracks.Items[i].SetId(i);
			l = m_Vertices.Items.Length;
			for (i = 0; i < l; i++)
				m_Vertices.Items[i].SetId(i);
			l = m_Layers.Items.Length;
			for (i = 0; i < l; i++)
			{
				Layer ll = m_Layers.Items[i];
				ll.SetId(i);
				int h, k;
				k = ll.Length;
				for (h = 0; h < k; h++)
					ll[h].SetLayerOwner(ll, h);
			}
		}
	}

	#endregion

	namespace PostProcessing
	{
		/// <summary>
		/// A generic interface for data analysis.
		/// </summary>
		public interface DataAnalyzer// : ISerializable
		{
			/// <summary>
			/// Feeds the data analyzer with a volume.
			/// </summary>
			/// <param name="volume">The volume to be analyzed.</param>
			void Feed(SySal.TotalScan.Volume volume);
		}


    }

    #region VertexFit
    /// <summary>
    /// Vertex fitting class.
    /// </summary>
    public class VertexFit
    {

        /// <summary>
        /// A TrackFit object is used to represent a track extrapolation towards a possible vertex. 
        /// </summary>
        public class TrackFit : SySal.Tracking.MIPEmulsionTrackInfo, System.ICloneable
        {
            /// <summary>
            /// Index of the track. It is used to distinguish several tracks in the same vertex.
            /// </summary>
            public SySal.TotalScan.Index Id;
            /// <summary>
            /// Weight of the track in the vertex fit.
            /// </summary>
            public double Weight;
            /// <summary>
            /// Minimum Z acceptable for this TrackFit.
            /// </summary>
            public double MinZ;
            /// <summary>
            /// Maximum Z acceptable for this TrackFit.
            /// </summary>
            public double MaxZ;
            /// <summary>
            /// Clones the TrackFit.
            /// </summary>
            /// <returns>a new TrackFit object.</returns>
            public virtual new object Clone()
            {
                SySal.TotalScan.VertexFit.TrackFit T = new TrackFit();
                T.AreaSum = this.AreaSum;
                T.BottomZ = this.BottomZ;
                T.Count = this.Count;
                T.Field = this.Field;
                T.Id = (SySal.TotalScan.Index)(this.Id.Clone());
                T.Intercept = this.Intercept;
                T.Sigma = this.Sigma;
                T.Slope = this.Slope;
                T.TopZ = this.TopZ;
                T.Weight = this.Weight;
                T.MinZ = this.MinZ;
                T.MaxZ = this.MaxZ;
                return T;
            }
        }

        /// <summary>
        /// Fit of a track with momentum information.
        /// </summary>
        public class TrackFitWithMomentum : TrackFit, System.ICloneable
        {
            /// <summary>
            /// The likelihood function for the track momentum.
            /// </summary>
            public NumericalTools.Likelihood PLikelihood;
            /// <summary>
            /// Most probable momentum of the particle.
            /// </summary>
            public double P
            {
                get { return PLikelihood.Best(0); }
            }
            /// <summary>
            /// Clones the object.
            /// </summary>
            /// <returns>a clone of the object.</returns>
            public override object Clone()
            {
                TrackFitWithMomentum T = new TrackFitWithMomentum();
                T.AreaSum = this.AreaSum;
                T.BottomZ = this.BottomZ;
                T.Count = this.Count;
                T.Field = this.Field;
                T.Id = (SySal.TotalScan.Index)(this.Id.Clone());
                T.Intercept = this.Intercept;
                T.Sigma = this.Sigma;
                T.Slope = this.Slope;
                T.TopZ = this.TopZ;
                T.Weight = this.Weight;
                T.MinZ = this.MinZ;
                T.MaxZ = this.MaxZ;
                T.PLikelihood = this.PLikelihood;
                return T;
            }
        }

        /// <summary>
        /// Base class to represent exceptions in the fitting procedure.
        /// </summary>
        public class FitException : System.Exception
        {
            /// <summary>
            /// Builds a fitting exception with an error message.
            /// </summary>
            /// <param name="t">the error message in the exception.</param>
            public FitException(string t) : base(t) { }
        }

        /// <summary>
        /// Represents a fitting error due to insufficient number of tracks in the vertex.
        /// </summary>
        public class NumberOfTracksException : FitException
        {
            /// <summary>
            /// Builds a fitting exception due to insufficient number of tracks in the vertex.
            /// </summary>
            /// <param name="t">the error message in the exception.</param>
            public NumberOfTracksException(string t) : base(t) { }
        }

        /// <summary>
        /// Signals no fit is available.
        /// </summary>
        public class NoFitReadyException : FitException
        {
            /// <summary>
            /// Builds a fitting exception to signal no fit is available.
            /// </summary>
            /// <param name="t">the error message in the exception.</param>
            public NoFitReadyException(string t) : base(t) { }
        }

        /// <summary>
        /// Signals the track number is wrong.
        /// </summary>
        public class WrongTrackNumberException : FitException
        {
            /// <summary>
            /// Builds an exception that signals the track number is wrong.
            /// </summary>
            /// <param name="t">the error message in the exception.</param>
            public WrongTrackNumberException(string t) : base(t) { }
        }

        /// <summary>
        /// Signals the Hessian in the vertex fit is non-positive.
        /// </summary>
        /// <remarks>This condition can happen when the vertex is topologically unlikely (e.g. tracks are diverging instead of converging), 
        /// or some limit mathematical situation has been reached (e.g. tracks are almost parallel).</remarks>
        public class NonPositiveHessianException : FitException
        {
            /// <summary>
            /// Builds an exception that signals the Hessian in the vertex fit is non-positive.
            /// </summary>
            /// <param name="t">the error message in the exception.</param>
            public NonPositiveHessianException(string t) : base(t) { }
        }

        /// <summary>
        /// Signals that the TrackFit limitations on Z are such that no Z range is allowed.
        /// </summary>
        public class NoZRangeAllowedException : FitException
        {
            /// <summary>
            /// Builds an exception that signals TrackFit limitations on Z are such that no Z range is allowed.
            /// </summary>
            /// <param name="t">the error message in the exception.</param>
            public NoZRangeAllowedException(string t) : base(t) { }
        }

        /// <summary>
        /// Builds a vertex fit.
        /// </summary>
        public VertexFit() { tList = new TrackFit[0]; }

        /// <summary>
        /// Adds a track to the fit.
        /// </summary>
        /// <param name="t">the TrackFit to be added.</param>
        /// <returns>the number of tracks in the vertex fit.</returns>
        /// <remarks><b>Notice: weight must be strictly positive.</b></remarks>
        public virtual int AddTrackFit(TrackFit t)
        {
            int i;
            for (i = 0; i < tList.Length; i++)
                if (t.Id.Equals(tList[i].Id)) throw new WrongTrackNumberException("Track already exists.");
            TrackFit[] nList = new TrackFit[tList.Length + 1];
            for (i = 0; i < tList.Length; i++)
                nList[i] = tList[i];
            nList[i] = (TrackFit)t.Clone();
            tList = nList;
            m_FitReady = false;
            return tList.Length;
        }

        /// <summary>
        /// Removes a track from a vertex fit.
        /// </summary>
        /// <param name="id">the index of the track to be removed.</param>
        /// <returns>the number of tracks remaining in the fit.</returns>
        /// <remarks><b>Notice: a removed track does no longer belong to the fit.</b> Do not use this method
        /// to temporarily exclude a track from a vertex fit (e.g. for computing the "disconnected IP"). Use
        /// <see cref="DisconnectedTrackIP"/> instead.</remarks>
        public virtual int RemoveTrackFit(SySal.TotalScan.Index id)
        {
            int j;
            for (j = 0; j < tList.Length && id.Equals(tList[j].Id) == false; j++) ;
            if (j == tList.Length) throw new WrongTrackNumberException("Wrong track number specified");
            TrackFit[] nList = new TrackFit[tList.Length - 1];
            tList[j] = tList[nList.Length];
            for (j = 0; j < nList.Length; j++)
                nList[j] = tList[j];
            tList = nList;
            m_FitReady = false;
            return tList.Length;
        }

        /// <summary>
        /// The X coordinate of the vertex.
        /// </summary>
        public virtual double X
        {
            get
            {
                if (m_FitReady == false)
                    FitVertex(out m_X, out m_Y, out m_Z, out m_AvgD, -1);
                return m_X;
            }
        }

        /// <summary>
        /// The Y coordinate of the vertex.
        /// </summary>
        public virtual double Y
        {
            get
            {
                if (m_FitReady == false)
                    FitVertex(out m_X, out m_Y, out m_Z, out m_AvgD, -1);
                return m_Y;
            }
        }

        /// <summary>
        /// The Z coordinate of the vertex.
        /// </summary>
        public virtual double Z
        {
            get
            {
                if (m_FitReady == false)
                    FitVertex(out m_X, out m_Y, out m_Z, out m_AvgD, -1);
                return m_Z;
            }
        }

        /// <summary>
        /// The average distance of tracks from the vertex position.
        /// </summary>
        public virtual double AvgDistance
        {
            get
            {
                if (m_FitReady == false)
                    FitVertex(out m_X, out m_Y, out m_Z, out m_AvgD, -1);
                return m_AvgD;
            }
        }

        /// <summary>
        /// The number of tracks in the vertex fit.
        /// </summary>
        public virtual int Count { get { return tList.Length; } }

        /// <summary>
        /// Gets the i-th TrackFit.
        /// </summary>
        /// <param name="i">the number of the TrackFit to be retrieved.</param>
        /// <returns>the i-th TrackFit.</returns>
        public virtual TrackFit Track(int i)
        {
            return (TrackFit)(tList[i].Clone());
        }

        /// <summary>
        /// Gets the TrackFit with the specified Id.
        /// </summary>
        /// <param name="id">the id of the TrackFit to be retrieved.</param>
        /// <returns>the specified TrackFit.</returns>
        public virtual TrackFit Track(Index id)
        {
            int j;
            for (j = 0; j < tList.Length && id.Equals(tList[j].Id) == false; j++) ;
            if (j == tList.Length) throw new WrongTrackNumberException("Wrong track number specified");
            return (TrackFit)(tList[j].Clone());
        }

        /// <summary>
        /// Computes the Impact Parameter of a track, without disconnection.
        /// </summary>
        /// <param name="t">the TrackFit to be used to compute the Impact Parameter.</param>
        /// <returns>the Impact Parameter.</returns>
        /// <remarks>The Impact Parameter is computed as the 3D distance of the vertex from the track. 
        /// Notice that no disconnection occurs, i.e. if the TrackFit belongs to the vertex, the vertex
        /// is not recomputed disconnecting this track.</remarks>
        public virtual double TrackIP(TrackFit t)
        {
            switch (tList.Length)
            {
                case 0: throw new NoFitReadyException("Vertex has no tracks.");

                case 1: return TwoTrackIP(tList[0], t);

                default:
                    {
                        if (m_FitReady == false)
                        {
                            FitVertex(out m_X, out m_Y, out m_Z, out m_AvgD, -1);
                            m_FitReady = true;
                        }
                        return PointTrackIP(t, m_X, m_Y, m_Z);
                    }
            }
        }

        /// <summary>
        /// Computes the Impact Parameter of a TrackFit belonging to the vertex, including track disconnection.
        /// </summary>
        /// <param name="id">the index of the track to be used to compute the Impact Parameter.</param>
        /// <returns>the Impact Parameter.</returns>
        /// <remarks>The Impact Parameter is computed as the 3D distance of the vertex from the track. 
        /// Notice that the track is temporarily disconnected from the vertex, then the vertex is recomputed without the track, 
        /// and the IP of the track with respect to the recomputed vertex is retrieved. X, Y, Z, AvgDistance are not affected,
        /// and "remember" the full-vertex fit.</remarks>
        public virtual double DisconnectedTrackIP(SySal.TotalScan.Index id)
        {
            int j;
            for (j = 0; j < tList.Length && id.Equals(tList[j].Id) == false; j++) ;
            if (j == tList.Length) throw new WrongTrackNumberException("Wrong track number specified");
            switch (tList.Length)
            {
                case 1: return 0.0;
                case 2: return TwoTrackIP(tList[j], tList[1 - j]);

                default:
                    {
                        double x, y, z, avgd;
                        TrackFit T = tList[j];
                        FitVertex(out x, out y, out z, out avgd, j);
                        return PointTrackIP(T, x, y, z);
                    }
            }
        }

        /// <summary>
        /// Signals whether the fit is ready or not.
        /// </summary>
        protected bool m_FitReady;
        /// <summary>
        /// Holds the fitted vertex X.
        /// </summary>
        protected double m_X;
        /// <summary>
        /// Holds the fitted vertex Y.
        /// </summary>
        protected double m_Y;
        /// <summary>
        /// Holds the fitted vertex Z.
        /// </summary>
        protected double m_Z;
        /// <summary>
        /// Holds the average distance of tracks from the fitted vertex.
        /// </summary>
        protected double m_AvgD;
        /// <summary>
        /// Holds the list of TrackFit objects belonging to the vertex.
        /// </summary>
        protected TrackFit[] tList;
        /// <summary>
        /// Computes the 3D distance between two straight lines.
        /// </summary>
        /// <param name="T">the first line.</param>
        /// <param name="L">the second line.</param>
        /// <returns>the Impact Parameter.</returns>
        protected static double TwoTrackIP(TrackFit T, TrackFit L)
        {
            double Vect_Prod_x, Vect_Prod_y, Vect_Prod_z;
            double mod;
            double PP_x, PP_y, PP_z;

            Vect_Prod_x = T.Slope.Y - L.Slope.Y;
            Vect_Prod_y = L.Slope.X - T.Slope.X;
            Vect_Prod_z = T.Slope.X * L.Slope.Y - T.Slope.Y * L.Slope.X;

            mod = (Vect_Prod_x * Vect_Prod_x) + (Vect_Prod_y * Vect_Prod_y) + (Vect_Prod_z * Vect_Prod_z);
            if (mod <= 0.0) throw new NonPositiveHessianException("Tracks are parallel.");
            mod = 1.0 / Math.Sqrt(mod);

            Vect_Prod_x *= mod;
            Vect_Prod_y *= mod;
            Vect_Prod_z *= mod;

            PP_x = T.Intercept.X - L.Intercept.X;
            PP_y = T.Intercept.Y - L.Intercept.Y;
            PP_z = T.Intercept.Z - L.Intercept.Z;

            return Math.Abs(PP_x * Vect_Prod_x + PP_y * Vect_Prod_y + PP_z * Vect_Prod_z);
        }

        /// <summary>
        /// Computes the 3D distance between a point and a straight line.
        /// </summary>
        /// <param name="T">the line.</param>
        /// <param name="x">the X coordinate of the point.</param>
        /// <param name="y">the Y coordinate of the point.</param>
        /// <param name="z">the Z coordinate of the point.</param>
        /// <returns></returns>
        protected static double PointTrackIP(TrackFit T, double x, double y, double z)
        {
            double Scal_Prod;
            double Num;
            double PP_x, PP_y, PP_z;

            Scal_Prod = 1.0 / (T.Slope.X * T.Slope.X + T.Slope.Y * T.Slope.Y + 1.0);

            PP_x = x - T.Intercept.X;
            PP_y = y - T.Intercept.Y;
            PP_z = z - T.Intercept.Z;

            Num = (PP_x * T.Slope.X + PP_y * T.Slope.Y + PP_z) * Scal_Prod;

            PP_x -= Num * T.Slope.X;
            PP_y -= Num * T.Slope.Y;
            PP_z -= Num;

            return Math.Sqrt((PP_x * PP_x) + (PP_y * PP_y) + (PP_z * PP_z));
        }

        /// <summary>
        /// Fits a vertex.
        /// </summary>
        /// <param name="x">the resulting X coordinate of the vertex.</param>
        /// <param name="y">the resulting Y coordinate of the vertex.</param>
        /// <param name="z">the resulting Z coordinate of the vertex.</param>
        /// <param name="avgd">the resulting AverageDistance of the vertex.</param>
        /// <param name="excludeone">the index of the track to be excluded from the vertex fit. Set this to a negative number to keep all tracks.</param>        
        protected void FitVertex(out double x, out double y, out double z, out double avgd, int excludeone)
        {
            if (tList.Length <= 0) throw new NumberOfTracksException("No tracks available to produce the vertex.");
            if (tList.Length == 1) throw new NumberOfTracksException("Cannot fit a vertex with just one track.");
            int i;
            double grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
            double den;            
            double denh;            
            double z_order = 0.0;
            double hess_xx = 0.0, hess_xy = 0.0, hess_xz = 0.0;
            double hess_yy = 0.0, hess_yz = 0.0;
            double hess_zz = 0.0;

            double ihess_xx = 0.0, ihess_xy = 0.0, ihess_xz = 0.0;
            double ihess_yy = 0.0, ihess_yz = 0.0;
            double ihess_zz = 0.0;
            double iden;
            double avgx = 0.0;
            double avgy = 0.0;
            double avgz = 0.0;            
            for (i = 0; i < tList.Length; i++)
            {
                avgx += tList[i].Intercept.X;
                avgy += tList[i].Intercept.Y;
                avgz += tList[i].Intercept.Z;
            }
            avgx /= tList.Length;
            avgy /= tList.Length;
            avgz /= tList.Length;
            double minz = tList[0].MinZ - avgz;
            double maxz = tList[0].MaxZ - avgz;
            double weight = 0.0;

            for (i = 0; i < tList.Length; i++)
            {
                if (i == excludeone) continue;
                SySal.Tracking.MIPEmulsionTrackInfo L = new SySal.Tracking.MIPEmulsionTrackInfo();
                L.Intercept = tList[i].Intercept;
                L.Slope = tList[i].Slope;
                L.Intercept.X -= avgx;
                L.Intercept.Y -= avgy;
                L.Intercept.Z -= avgz;
                minz = Math.Max(minz, (tList[i].MinZ - avgz));
                maxz = Math.Min(maxz, (tList[i].MaxZ - avgz));
                if (minz >= maxz) throw new NoZRangeAllowedException("No Z range allowed for this vertex.");
                weight += tList[i].Weight;
                den = (1 + (L.Slope.X * L.Slope.X) + (L.Slope.Y * L.Slope.Y));
                iden = tList[i].Weight / den;
                grad_x -= 2.0 * (L.Intercept.X + L.Intercept.X * L.Slope.Y * L.Slope.Y - L.Slope.X * L.Slope.Y * L.Intercept.Y - L.Intercept.Z * L.Slope.X) * iden;
                grad_y -= 2.0 * (L.Intercept.Y + L.Intercept.Y * L.Slope.X * L.Slope.X - L.Slope.X * L.Slope.Y * L.Intercept.X - L.Intercept.Z * L.Slope.Y) * iden;
                grad_z -= 2.0 * (-L.Slope.X * L.Intercept.X + L.Intercept.Z * L.Slope.X * L.Slope.X - L.Slope.Y * L.Intercept.Y + L.Slope.Y * L.Slope.Y * L.Intercept.Z) * iden;

                hess_xx += (1.0 + L.Slope.Y * L.Slope.Y) * iden;
                hess_xy -= L.Slope.X * L.Slope.Y * iden;
                hess_xz -= L.Slope.X * iden;
                hess_yy += (1.0 + L.Slope.X * L.Slope.X) * iden;
                hess_yz -= L.Slope.Y * iden;
                hess_zz += (L.Slope.X * L.Slope.X + L.Slope.Y * L.Slope.Y) * iden;

                z_order += ((1 + L.Slope.Y * L.Slope.Y) * (L.Intercept.X * L.Intercept.X) + (1 + L.Slope.X * L.Slope.X) * (L.Intercept.Y * L.Intercept.Y) - 2.0 * L.Slope.Y * L.Intercept.Y * L.Intercept.Z + (L.Slope.X * L.Slope.X + L.Slope.Y * L.Slope.Y) * (L.Intercept.Z * L.Intercept.Z) - 2.0 * (L.Slope.X * L.Intercept.X) * (L.Slope.Y * L.Intercept.Y + L.Intercept.Z)) * iden;
            }

            denh = (-hess_xz * hess_xz * hess_yy + 2.0 * hess_xy * hess_xz * hess_yz - hess_xx * hess_yz * hess_yz - hess_xy * hess_xy * hess_zz + hess_xx * hess_yy * hess_zz);
            if (denh <= 0.0) throw new NonPositiveHessianException("Determinant of Hessian is not positive.");
            denh = 0.125 / denh;

            hess_xx *= 2.0;
            hess_xy *= 2.0;
            hess_xz *= 2.0;
            hess_yy *= 2.0;
            hess_yz *= 2.0;
            hess_zz *= 2.0;

            ihess_xx = -(-hess_yz * hess_yz + hess_yy * hess_zz) * denh;
            ihess_xy = -(hess_xz * hess_yz - hess_xy * hess_zz) * denh;
            ihess_xz = -(-hess_xz * hess_yy + hess_xy * hess_yz) * denh;
            ihess_yy = -(-hess_xz * hess_xz + hess_xx * hess_zz) * denh;
            ihess_yz = -(hess_xy * hess_xz - hess_xx * hess_yz) * denh;
            ihess_zz = -(-hess_xy * hess_xy + hess_xx * hess_yy) * denh;

            z = ihess_xz * grad_x + ihess_yz * grad_y + ihess_zz * grad_z;
            if (z >= minz && z <= maxz)
            {
                x = ihess_xx * grad_x + ihess_xy * grad_y + ihess_xz * grad_z;
                y = ihess_xy * grad_x + ihess_yy * grad_y + ihess_yz * grad_z;

                avgd = Math.Sqrt((z_order +
                    (grad_x + 0.5 * (hess_xx * x + hess_xy * y + hess_xz * z)) * x +
                    (grad_y + 0.5 * (hess_xy * x + hess_yy * y + hess_yz * z)) * y +
                    (grad_z + 0.5 * (hess_xz * x + hess_yz * y + hess_zz * z)) * z) / weight);
            }
            else
            {
                if (z < minz) z = minz;
                else z = maxz;
                x = 0.0;
                y = 0.0;
                for (i = 0; i < tList.Length; i++)
                {
                    x += tList[i].Weight * (tList[i].Intercept.X - avgx + (z - (tList[i].Intercept.Z - avgz)) * tList[i].Slope.X);
                    y += tList[i].Weight * (tList[i].Intercept.Y - avgy + (z - (tList[i].Intercept.Z - avgz)) * tList[i].Slope.Y);
                }
                x /= weight;
                y /= weight;
                avgd = 0.0;
                double dx = 0.0;
                double dy = 0.0;
                for (i = 0; i < tList.Length; i++)
                {
                    dx = tList[i].Weight * (tList[i].Intercept.X - avgx + (z - (tList[i].Intercept.Z - avgz)) * tList[i].Slope.X) - x;
                    dy = tList[i].Weight * (tList[i].Intercept.Y - avgy + (z - (tList[i].Intercept.Z - avgz)) * tList[i].Slope.Y) - y;
                    avgd += tList[i].Weight * (dx * dx + dy * dy);
                }
                avgd = Math.Sqrt(avgd / weight);
            }

            x += avgx;
            y += avgy;
            z += avgz;
        }
    }

    #endregion

    #region General Purpose Functions

    public class GeneralPurposeFunction
	{
		#region Area Selection
		public static Track[] SelectTracksBySegmentsNumber(Track[] Tracks, int Minimum_Segments_Number)
		{
			int i,n;
			n = Tracks.GetLength(0);
			System.Collections.ArrayList SelTracks = new System.Collections.ArrayList(n);
			for (i=0; i<n; i++) if (Tracks[i].Length>=Minimum_Segments_Number) SelTracks.Add(Tracks[i]);
			return (Track[]) SelTracks.ToArray(typeof(Track));
		}

		public static Track[] SelectTracksBySlopeSpectrum(Track[] Tracks, double Center_Sx, double Center_Sy, double RadiusS, int SegNumberInCut)
		{
			int i,j,n, m, nr;
			double drs;
			n = Tracks.GetLength(0);
			System.Collections.ArrayList SelTracks = new System.Collections.ArrayList(n);
			
			for (i=0; i<n; i++)
			{
				m= Tracks[i].Length;
				nr=0;
				for (j=0; j<m;j++)
				{
					drs=Math.Sqrt((Tracks[i][j].Info.Slope.X-Center_Sx)*(Tracks[i][j].Info.Slope.X-Center_Sx)+
						(Tracks[i][j].Info.Slope.Y-Center_Sy)*(Tracks[i][j].Info.Slope.Y-Center_Sy));
					if(drs<RadiusS)
					{
						nr++;
						if(nr==SegNumberInCut && SegNumberInCut!=0)
						{
							SelTracks.Add(Tracks[i]);
							break;
						};
					};
				};
				if(nr==m && SegNumberInCut==0) SelTracks.Add(Tracks[i]);

			};

			return (Track[]) SelTracks.ToArray(typeof(Track));
		}

		public static Track[] SelectTracksBySlopeSpectrum(Track[] Tracks, double MinSx, double MaxSx, double MinSy, double MaxSy, int SegNumberInCut)
		{
			int i,j,n, m, nr;
			double syt,sxt;
			n = Tracks.GetLength(0);
			System.Collections.ArrayList SelTracks = new System.Collections.ArrayList(n);
			
			for (i=0; i<n; i++)
			{
				m= Tracks[i].Length;
				nr=0;
				for (j=0; j<m;j++)
				{
					syt=Tracks[i][j].Info.Slope.Y;
					sxt=Tracks[i][j].Info.Slope.X;
					if (syt< MaxSy && syt> MinSy && sxt < MaxSx && sxt > MinSx)
					{
						nr++;
						if(nr==SegNumberInCut && SegNumberInCut!=0)
						{
							SelTracks.Add(Tracks[i]);
							break;
						};
					};
				};
				if(nr==m && SegNumberInCut==0) SelTracks.Add(Tracks[i]);
			};
			return (Track[]) SelTracks.ToArray(typeof(Track));

		}

		public static Track[] SelectTracksByArea(Track[] Tracks, double CenterX, double CenterY, double Radius,
			double Sx, double Sy, double RadiusS, int SegNumberInCut)
		{
			int i,j,n, m, nr;
			double dr;
			double drs;
			n = Tracks.GetLength(0);
			System.Collections.ArrayList SelTracks = new System.Collections.ArrayList(n);
			
			for (i=0; i<n; i++)
			{
				nr=0;
				m= Tracks[i].Length;
				for (j=0; j<m;j++)
				{
					dr=Math.Sqrt((Tracks[i][j].Info.Intercept.X-CenterX)*(Tracks[i][j].Info.Intercept.X-CenterX)+
						(Tracks[i][j].Info.Intercept.Y-CenterY)*(Tracks[i][j].Info.Intercept.Y-CenterY));
					if (dr< Radius)
					{
						drs=Math.Sqrt((Tracks[i][j].Info.Slope.X-Sx)*(Tracks[i][j].Info.Slope.X-Sx)+
							(Tracks[i][j].Info.Slope.Y-Sy)*(Tracks[i][j].Info.Slope.Y-Sy));
						if(drs<RadiusS)
						{
							nr++;
							if(nr==SegNumberInCut && SegNumberInCut!=0)
							{
								SelTracks.Add(Tracks[i]);
								break;
							};
						};
					};
				};
				if(nr==m && SegNumberInCut==0) SelTracks.Add(Tracks[i]);
			};
			//			return (MultipleScattering.Track[]) SelTracks.ToArray(typeof(MultipleScattering.Track));
			return (Track[]) SelTracks.ToArray(typeof(Track));

		}
	
		public static Track[] SelectTracksByArea(Track[] Tracks, double MinX, double MaxX, double MinY,double MaxY,
			double MinSx, double MaxSx, double MinSy, double MaxSy, int SegNumberInCut)
		{
			int i,j,n, m, nr;
			double yt,xt;
			double syt,sxt;
			n = Tracks.GetLength(0);
			System.Collections.ArrayList SelTracks = new System.Collections.ArrayList(n);
			
			for (i=0; i<n; i++)
			{
				m= Tracks[i].Length;
				nr=0;
				for (j=0; j<m;j++)
				{
					yt=Tracks[i][j].Info.Intercept.Y;
					xt=Tracks[i][j].Info.Intercept.X;
					if (yt< MaxY && yt> MinY && xt < MaxX && xt > MinX)
					{
						syt=Tracks[i][j].Info.Slope.Y;
						sxt=Tracks[i][j].Info.Slope.X;
						if (syt< MaxSy && syt> MinSy && sxt < MaxSx && sxt > MinSx)
						{
							nr++;
							if(nr==SegNumberInCut && SegNumberInCut!=0)
							{
								SelTracks.Add(Tracks[i]);
								break;
							};
						};
					};
				};
				if(nr==m && SegNumberInCut==0) SelTracks.Add(Tracks[i]);
			};
			//			return (MultipleScattering.Track[]) SelTracks.ToArray(typeof(MultipleScattering.Track));
			return (Track[]) SelTracks.ToArray(typeof(Track));

		}

		#endregion

		#region Accuracy Computation
		public static ComputationResult ComputeAccuracy(Track[] Tracks, int NSegFit, int SheetPos, int MinTracksNum, ref double DX, ref double DY, 
			ref double RMSx, ref double RMSy, ref double DSX, ref double DSY, ref double RMSsx, ref double RMSsy)
		{
			int i,n, m;
			int jf,kf, Counts=0;
			double Sy=0, Sx=0, Ny=0, Nx=0, dum=0;

			if (NSegFit <= 0 && SheetPos<0) return ComputationResult.Incoherent; 

			NumericalTools.ComputationResult[] chk = new NumericalTools.ComputationResult[4];

			int[] Pos = new int[NSegFit];
			double[] tmpx = new double[NSegFit];
			double[] tmpy = new double[NSegFit];
			double[] tmpz = new double[NSegFit];
			double[] tmpsy = new double[NSegFit];
			double[] tmpsx = new double[NSegFit];

			n = Tracks.GetLength(0);
			System.Collections.ArrayList tmpDy = new System.Collections.ArrayList(n*NSegFit);
			System.Collections.ArrayList tmpDx = new System.Collections.ArrayList(n*NSegFit);
			System.Collections.ArrayList tmpDSy = new System.Collections.ArrayList(n*NSegFit);
			System.Collections.ArrayList tmpDSx = new System.Collections.ArrayList(n*NSegFit);

			for (i=0; i<n; i++)
			{
				m= Tracks[i].Length;
				for(jf = 0; jf<= m - NSegFit; jf++)
				{	
					for(kf = 0; kf< NSegFit;kf++)
					{        
						Pos[kf] = Tracks[i][jf + kf].LayerOwner.SheetId/*.PosID*/;
						tmpx[kf] = Tracks[i][jf + kf].Info.Intercept.X;
						tmpy[kf] = Tracks[i][jf + kf].Info.Intercept.Y;
						tmpz[kf] = Tracks[i][jf + kf].Info.Intercept.Z;
						tmpsy[kf] = Tracks[i][jf + kf].Info.Slope.Y;
						tmpsx[kf] = Tracks[i][jf + kf].Info.Slope.X;
                                    
					};
        
					// seleziona solo la misura sul foglio desiderato
					//se essa è presente
					for(kf = 0; kf< NSegFit; kf++)
						if (Pos[kf] == SheetPos)
						{
							//Fitta
							Fitting.LinearFitSE(tmpz, tmpy, ref Sy, ref Ny, 
								ref dum, ref dum, ref dum, ref dum, ref dum);
							Fitting.LinearFitSE(tmpz, tmpx, ref Sx, ref Nx, 
								ref dum, ref dum, ref dum, ref dum, ref dum);

							tmpDy.Add(tmpy[kf] - (Ny + Sy * tmpz[kf]));
							tmpDx.Add(tmpx[kf] - (Nx + Sx * tmpz[kf]));
							tmpDSy.Add(tmpsy[kf] - Sy);
							tmpDSx.Add(tmpsx[kf] - Sx);
							Counts++;
							break;
						};

				};
			};

			if(Counts >= MinTracksNum)
			{
				double[] tmp = (double[])tmpDx.ToArray(typeof(double));
				chk[0] = Fitting.FindStatistics(tmp, ref dum, ref dum, ref DX, ref RMSx);
				tmp = (double[])tmpDy.ToArray(typeof(double));
				chk[1] = Fitting.FindStatistics(tmp, ref dum, ref dum, ref DY, ref RMSy);
				tmp = (double[])tmpDSx.ToArray(typeof(double));
				chk[2] = Fitting.FindStatistics(tmp, ref dum, ref dum, ref DSX, ref RMSsx);
				tmp = (double[])tmpDSy.ToArray(typeof(double));
				chk[3] = Fitting.FindStatistics(tmp, ref dum, ref dum, ref DSY, ref RMSsy);
				for(i = 0; i< 4;i++) if(chk[i]!= NumericalTools.ComputationResult.OK) return ComputationResult.Incoherent;
				return ComputationResult.OK;
			}
			else
			{
				DY=0; DX=0;
				RMSy=0; RMSx=0; 
				DSY=0; DSX=0; 
				RMSsy=0; RMSsx=0;
				return ComputationResult.InsufficientMeasurements;
			};
	
		
		}

		#endregion
	}
	#endregion


	#region Reconstruction
	
	/// <summary>
	/// Checks whether the reconstruction process should be aborted.
	/// </summary>
	public delegate bool dShouldStop();

	/// <summary>
	/// Returns progress information about the reconstruction process.
	/// </summary>
	public delegate void dProgress(double percent);

	/// <summary>
	/// Returns text information about the reconstruction process.
	/// </summary>
	public delegate void dReport(string txtreport);

	/// <summary>
	/// Interface that every module that performs volume reconstruction should implement.
	/// </summary>
	public interface IVolumeReconstructor
	{
		/// <summary>
		/// Callback delegate that can be used to stop the reconstruction process.
		/// </summary>
		dShouldStop ShouldStop
		{
			get;
			set;
		}

		/// <summary>
		/// Callback delegate that monitors the reconstruction progress (ranging from 0 to 1).
		/// </summary>
		dProgress Progress
		{
			get;
			set;
		}

		/// <summary>
		/// Callback delegate that monitors the reconstruction report.
		/// </summary>
		dReport Report
		{
			get;
			set;
		}

		/// <summary>
		/// Clears the reconstructor of previously loaded layers.
		/// </summary>
		void Clear();

		/// <summary>
		/// Adds one layer to the set of layers to use for the reconstruction.
		/// The layer should have been previously filled up with segments.
		/// </summary>
		/// <param name="l"></param>
		void AddLayer(Layer l);

		/// <summary>
		/// Adds one layer to the set of layers to use for the reconstruction, filling it with segments whose geometrical parameters are given by a set of MIPEmulsionTrackInfo.
		/// </summary>
		/// <param name="l"></param>
		/// <param name="basetks"></param>
		void AddLayer(Layer l, MIPEmulsionTrackInfo [] basetks);

		/// <summary>
		/// Adds one layer to the set of layers to use for the reconstruction.
		/// The layer is filled up with tracks from the supplied scanning zone.
		/// This method is used to keep track of unassociated microtracks too, e.g. to search for kinks in the base.
		/// </summary>
		/// <param name="l"></param>
		/// <param name="zone"></param>
		void AddLayer(Layer l, SySal.Scanning.Plate.LinkedZone zone);

		/// <summary>
		/// Reconstructs volume tracks and optionally track intersections (vertices), using data that have been previously fed in through AddLayer.
		/// </summary>
		Volume Reconstruct();

		/// <summary>
		/// Recomputes vertices on an existing Volume. Yields a new volume with new vertices, and possibly, also new tracks. Does not recompute layer-to-layer alignment.
		/// </summary>
		Volume RecomputeVertices(Volume v);
	}


    /// <summary>
    /// The result of the momentum estimation.
    /// </summary>
    public struct MomentumResult
    {
        /// <summary>
        /// The estimated value of the momentum.
        /// </summary>
        public double Value;
        /// <summary>
        /// The Confidence Level for which bounds are computed.
        /// </summary>
        public double ConfidenceLevel;
        /// <summary>
        /// The lower bound of the momentum.
        /// </summary>
        public double LowerBound;
        /// <summary>
        /// The upper bound of the momentum.
        /// </summary>
        public double UpperBound;
    }

    /// <summary>
    /// Defines the geometry of the detector for what concerns Multiple Coulomb Scattering properties.
    /// </summary>
    /// <remarks>The geometry is defined as a set of X,Y planes (each having its own Z) where the radiation length of the material changes. Data must be fully contained within the Z interval spanned by the first and last surface.</remarks>
    public class Geometry
    {
        /// <summary>
        /// Defines an X,Y plane surface where the radiation length of the material changes.
        /// </summary>
        public struct LayerStart
        {
            /// <summary>
            /// The lower Z bound of the volume with a certain material.
            /// </summary>
            public double ZMin;
            /// <summary>
            /// The radiation length of the material.
            /// </summary>
            public double RadiationLength;
            /// <summary>
            /// Together with <c>Plate</c>, this field defines a plate in the detector. Ignored if <c>Plate</c> is zero or negative.
            /// </summary>
            public long Brick;
            /// <summary>
            /// The plate that has this layer as upstream end. If this layer is not an upstream plate boundary, <c>Plate</c> must be zero or negative.
            /// </summary>
            public int Plate;
        }
        /// <summary>
        /// The set of surfaces bounding material volumes traversed by scattering tracks.
        /// </summary>
        public LayerStart[] Layers;

        public class order : System.Collections.IComparer
        {
            public int Compare(object x, object y)
            {
                double c = ((LayerStart)x).ZMin - ((LayerStart)y).ZMin;
                if (c > 0.0) return 1;
                if (c < 0.0) return -1;
                return 0;
            }
        }

        /// <summary>
        /// Empty constructor.
        /// </summary>
        public Geometry() { }

        /// <summary>
        /// Creates a Geometry from a list of Layers (<see cref="SySal.TotalScan.Layer"/>).
        /// </summary>
        /// <param name="ll">the list of layers (usually a Volume.Layers object).</param>
        public Geometry(SySal.TotalScan.Volume.LayerList ll)
        {
            int len = ll.Length;
            int i;
            Layers = new LayerStart[2 * len];
            for (i = 0; i < len; i++)
            {
                SySal.TotalScan.Layer lay = ll[len - 1 - i];
                Layers[2 * i].Brick = lay.BrickId;
                Layers[2 * i].Plate = lay.SheetId;
                Layers[2 * i].RadiationLength = lay.RadiationLengh;
                Layers[2 * i].ZMin = lay.UpstreamZ;
                Layers[2 * i + 1].Brick = lay.BrickId;
                Layers[2 * i + 1].Plate = 0;
                Layers[2 * i + 1].RadiationLength = lay.DownstreamRadiationLength;
                Layers[2 * i + 1].ZMin = lay.DownstreamZ;
            }
        }
    }

    /// <summary>
    /// Interface for MCS-based momentum estimation.
    /// </summary>
    public interface IMCSMomentumEstimator
    {
        /// <summary>
        /// Computes the momentum and confidence limits using positions and slopes provided.
        /// </summary>
        /// <param name="data">the position and slopes of the track (even Z-unordered).</param>
        /// <returns>the momentum and confidence limits.</returns>
        MomentumResult ProcessData(SySal.Tracking.MIPEmulsionTrackInfo[] data);
    }
	#endregion
}
