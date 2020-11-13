using System;
using System.IO;
using System.Runtime.Serialization;
using SySal;
using SySal.Management;
using SySal.BasicTypes;
using SySal.Tracking;

namespace SySal.Scanning
{
	/// <summary>
	/// A track in an emulsion layer that can be identified through its number.
	/// </summary>
	public class MIPIndexedEmulsionTrack : MIPEmulsionTrack
	{
		/// <summary>
		/// Member data on which the Id property relies. Can be accessed by derived classes.
		/// </summary>
		protected internal int m_Id;
		/// <summary>
		/// Id of the track, e.g. its sequential number in the emulsion layer.
		/// </summary>
		public virtual int Id { get { return m_Id; } }
		/// <summary>
		/// Protected constructor. Prevents user from creating MIPIndexedEmulsionTrack objects without deriving the class. Is implicitly called in the constructors of derived classes.
		/// </summary>
		protected MIPIndexedEmulsionTrack() {}
		/// <summary>
		/// Builds a new MIPIndexedEmulsionTrack from a MIPEmulsionTrack with the specified id.
		/// </summary>
		/// <param name="track">the MIPEmulsionTrack that supplies the track information.</param>
		/// <param name="id">the track identification number.</param>
		public MIPIndexedEmulsionTrack(MIPEmulsionTrack track, int id)
		{
			m_Id = id;
			m_Info = MIPIndexedEmulsionTrack.AccessInfo(track);
			m_Grains = MIPIndexedEmulsionTrack.AccessGrains(track);
		}
		/// <summary>
		/// Builds a new MIPIndexedEmulsionTrack from a MIPEmulsionTrackInfo, an array of grains and with the specified id.
		/// </summary>
		/// <param name="info">the MIPEmulsionTrackInfo that supplies the track information.</param>
		/// <param name="grains">the array of Grains.</param>
		/// <param name="id">the track identification number.</param>
		public MIPIndexedEmulsionTrack(MIPEmulsionTrackInfo info, Grain [] grains, int id)
		{
			m_Id = id;
			m_Info = info;
			m_Grains = grains;
		}	

		internal Grain [] iGrains { get { return m_Grains; } }

		internal SySal.Tracking.MIPEmulsionTrackInfo iInfo { get { return m_Info; } }
	}

	/// <summary>
	/// Track built by linking two track segments in emulsion across the base
	/// </summary>
	public class MIPBaseTrack
	{
		/// <summary>
		/// Member data on which the Info property relies. Can be accessed by derived classes.
		/// </summary>
		protected internal MIPEmulsionTrackInfo m_Info;
		/// <summary>
		/// Global information about the base track.
		/// </summary>
		public virtual MIPEmulsionTrackInfo Info { get { return (MIPEmulsionTrackInfo)m_Info.Clone(); } }
		/// <summary>
		/// Member data on which the Id property relies. Can be accessed by derived classes.
		/// </summary>
		protected internal int m_Id;
		/// <summary>
		/// Id of the track, e.g. its sequential number in the scanning zone.
		/// </summary>
		public virtual int Id { get { return m_Id; } }
		/// <summary>
		/// Member data on which the Top property relies. Can be accessed by derived classes.
		/// </summary>		
		protected internal MIPIndexedEmulsionTrack m_Top;
		/// <summary>
		/// Top track connected to the base track.
		/// </summary>
		public virtual MIPIndexedEmulsionTrack Top { get { return m_Top; } }
		/// <summary>
		/// Member data on which the Bottom property relies. Can be accessed by derived classes.
		/// </summary>		
		protected internal MIPIndexedEmulsionTrack m_Bottom;
		/// <summary>
		/// Bottom track connected to the base track.
		/// </summary>
		public virtual MIPIndexedEmulsionTrack Bottom { get { return m_Bottom; } }
		/// <summary>
		/// Protected constructor. Prevents user from creating MIPBaseTrack objects without deriving the class. Is implicitly called in the constructors of derived classes.
		/// </summary>
		protected MIPBaseTrack() {}
		/// <summary>
		/// Builds a new MIPBaseTrack by connecting two MIPIndexedEmulsionTracks together.
		/// </summary>
		/// <param name="topseg">the emulsion track on the top side of the plate.</param>
		/// <param name="bottomseg">the emulsion track on the bottom side of the plate.</param>
		/// <param name="id">the unique identifier of the base track.</param>
		public MIPBaseTrack(MIPIndexedEmulsionTrack topseg, MIPIndexedEmulsionTrack bottomseg, int id)
		{
			m_Id = id;
			m_Top = topseg;
			m_Bottom = bottomseg;
			MIPEmulsionTrackInfo Ti = topseg.Info;
			MIPEmulsionTrackInfo Bi = bottomseg.Info;
			m_Info = new MIPEmulsionTrackInfo();
			m_Info.AreaSum = Ti.AreaSum + Bi.AreaSum;
			m_Info.Count = (ushort)(Ti.Count + Bi.Count);
			m_Info.TopZ = Ti.TopZ;
			m_Info.BottomZ = Bi.BottomZ;
			m_Info.Intercept = Ti.Intercept;
			double dztop = Ti.BottomZ - Ti.Intercept.Z;
			double dzbottom = Bi.TopZ - Bi.Intercept.Z; 
			double idz = 1.0 / (Ti.BottomZ - Bi.TopZ);
			m_Info.Slope.X = idz * (Ti.Intercept.X + dztop * Ti.Slope.X - Bi.Intercept.X - dzbottom * Bi.Slope.X);
			m_Info.Slope.Y = idz * (Ti.Intercept.Y + dztop * Ti.Slope.Y - Bi.Intercept.Y - dzbottom * Bi.Slope.Y);
			m_Info.Slope.Z = 1.0f;
			double dsxt = Info.Slope.X - Ti.Slope.X;
			double dsyt = Info.Slope.Y - Ti.Slope.Y;
			double dsxb = Info.Slope.X - Bi.Slope.X;
			double dsyb = Info.Slope.Y - Bi.Slope.Y;
			m_Info.Sigma = Math.Sqrt(dsxt * dsxt + dsyt * dsyt + dsxb * dsxb + dsyb * dsyb);
		}
		/// <summary>
		/// Protected accessor for quick access to Info. Can be used by derived classes.
		/// </summary>
		/// <param name="track">the MIPBaseTrack whose parameters are being requested.</param>
		/// <returns>the global parameters of the MIPBaseTrack.</returns>
		protected static MIPEmulsionTrackInfo AccessInfo(MIPBaseTrack track) { return track.m_Info; }
	}

	namespace Plate
	{
		/// <summary>
		/// A single emulsion layer
		/// </summary>
		public class Side
		{
			/// <summary>
			/// Protected constructor. Prevents users from creating instances of Side without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal Side() {}
			/// <summary>
			/// Member data on which the TopZ property relies. Can be accessed by derived classes.
			/// </summary>
			protected internal double m_TopZ;
			/// <summary>
			/// Top surface of the emulsion side.
			/// </summary>
			public double TopZ { get { return m_TopZ; } }
			/// <summary>
			/// Member data on which the BottomZ property relies. Can be accessed by derived classes.
			/// </summary>
			protected internal double m_BottomZ;
			/// <summary>
			/// Bottom surface of the emulsion side.
			/// </summary>
			public double BottomZ { get { return m_BottomZ; } }
			/// <summary>
			/// Protected member holding the array of MIPIndexedEmulsionTracks. Can be accessed by derived classes.
			/// </summary>
			protected internal MIPIndexedEmulsionTrack [] m_Tracks;
			/// <summary>
			/// Provides access to the tracks in an array-like fashion.
			/// </summary>
			public virtual MIPIndexedEmulsionTrack this[int index] { get { return m_Tracks[index]; } }
			/// <summary>
			/// Returns the number of tracks in the emulsion side.
			/// </summary>
			public virtual int Length { get { return m_Tracks.Length; } }
			/// <summary>
			/// Builds a new emulsion side using the supplied array of tracks and the specified surface depths.
			/// </summary>
			/// <param name="tracks">the track array to be attached to the side.</param>
			/// <param name="topz">the top surface of the emulsion side.</param>
			/// <param name="bottomz">the bottom surface of the emulsion side.</param>
			public Side(MIPIndexedEmulsionTrack [] tracks, double topz, double bottomz)
			{
				m_TopZ = topz;
				m_BottomZ = bottomz;
				int i;
				m_Tracks = tracks;
				for (i = 0; i < tracks.Length; i++) tracks[i].m_Id = i;
			}										
			/// <summary>
			/// Builds a new emulsion side using the supplied array of tracks and the specified surface depths.
			/// </summary>
			/// <param name="tracks">the array of MIPEmulsionTracks that provides track information.</param>
			/// <param name="topz">the top surface of the emulsion side.</param>
			/// <param name="bottomz">the bottom surface of the emulsion side.</param>
			public Side(MIPEmulsionTrack [] tracks, double topz, double bottomz)
			{
				m_TopZ = topz;
				m_BottomZ = bottomz;
				m_Tracks = new MIPIndexedEmulsionTrack[tracks.Length];
				int i;
				for (i = 0; i < tracks.Length; i++) m_Tracks[i] = new MIPIndexedEmulsionTrack(tracks[i], i);
			}
			/// <summary>
			/// Protected accessor for quick access to the array of tracks. Can be accessed by derived classes.
			/// </summary>
			/// <param name="s">side whose track array is to be retrieved.</param>
			/// <returns>the array of tracks in the side.</returns>
			protected static MIPEmulsionTrack [] AccessTracks(Side s) { return s.m_Tracks; }
            /// <summary>
            /// Protected setter for derived classes.
            /// </summary>
            /// <param name="s">side whose track array is to be set.</param>
            /// <param name="tks">the new array of tracks in the side.</param>
            protected static void SetTracks(Side s, SySal.Scanning.MIPIndexedEmulsionTrack [] tks) { s.m_Tracks = tks; }
            /// <summary>
            /// Protected setter for derived classes.
            /// </summary>
            /// <param name="s">side whose Zs have to be set.</param>
            /// <param name="ztop">the new Top Z.</param>
            /// <param name="zbottom">the new Bottom Z.</param>
            protected static void SetZs(Side s, double ztop, double zbottom) { s.m_TopZ = ztop; s.m_BottomZ = zbottom; }
		}

		/// <summary>
		/// A scanning zone, containing tracks in the emulsion and additional global information.
		/// </summary>
		public abstract class Zone
		{
			/// <summary>
			/// Member data on which the Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected Identifier m_Id;
			/// <summary>
			/// A unique identifier for the zone.
			/// </summary>
			public Identifier Id { get { return m_Id; } }
			/// <summary>
			/// Member data on which the Pos property relies. Can be accessed by derived classes.
			/// </summary>			
			protected Vector2 m_Center;
			/// <summary>
			/// The center of the scanning zone.
			/// </summary>
			public Vector2 Center { get { return m_Center; } }
			/// <summary>
			/// Member data on which the Top property relies. Can be accessed by derived classes.
			/// </summary>
			protected Side m_Top;
			/// <summary>
			/// The top side of the scanning zone.
			/// </summary>
			public Side Top { get { return m_Top; } }
			/// <summary>
			/// Member data on which the Bottom property relies. Can be accessed by derived classes.
			/// </summary>
			protected Side m_Bottom;
			/// <summary>
			/// The bottom side of the scanning zone.
			/// </summary>
			public Side Bottom { get { return m_Bottom; } }
			/// <summary>
			/// Retrieves the sides by index (0 = top, 1 = bottom). Useful to iterate over all tracks in the zone.
			/// </summary>
			/// <param name="iside">side index. 0 = top, 1 = bottom. Other values cause an exception to be thrown.</param>
			/// <returns>the selected side.</returns>
			public Side GetSide(int iside)
			{
				if (iside == 0) return m_Top;
				if (iside == 1) return m_Bottom;
				throw new Exception("The side index must be 0 (top) or 1 (bottom).");
			}
		}
		/// <summary>
		/// A Zone containing base tracks (i.e. tracks built by linking tracks across the base.
		/// The LinkedZone can be used in an array-like fashion to access the base tracks.
		/// </summary>
		public abstract class LinkedZone : Zone
		{
			/// <summary>
			/// Member data holding the array of base tracks. Can be accessed by derived classes.
			/// </summary>
			protected internal MIPBaseTrack [] m_Tracks;
			/// <summary>
			/// Accessor that allows array-like access to the array of base tracks.
			/// </summary>
			public virtual MIPBaseTrack this[int index] { get { return m_Tracks[index]; } }
			/// <summary>
			/// Retrieves the number of base tracks in the LinkedZone.
			/// </summary>
			public virtual int Length { get { return m_Tracks.Length; } }
			/// <summary>
			/// Provides quick access to the array of m_Tracks. Can be accessed by derived classes.
			/// </summary>
			/// <param name="lz">the LinkedZone whose tracks are being requested.</param>
			/// <returns>the array of MIPBaseTracks in the LinkedZone.</returns>
			protected static MIPBaseTrack [] AccessTracks(LinkedZone lz) { return lz.m_Tracks; }
			/// <summary>
			/// Saves the LinkedZone to a stream.
			/// </summary>
			/// <param name="s">stream to be used to save the LinkedZone.</param>
			public abstract void Save(System.IO.Stream s);			
		}

		namespace IO
		{
			namespace CHORUS
			{
				namespace Thick
				{
					internal enum Info : byte 
					{
						Track = 1, Field = 3, Cluster = 4,  
						Image = 5, Grain = 6, ImagePred = 7, ClusterPred = 8,
						BlackTrack = 10, BlackStrip = 11
					}

					internal enum Section : byte { Data = 0x20, Header = 0x40 }
	
					internal enum Format : ushort { Old = 0x10, Old2 = 0x20, Normal = 0x40 }
	
					internal enum Compression : ushort { Null = 0x0000, Differential = 0x101 }
				}

				namespace Thin
				{
					internal enum Info : byte {	Track = 1, BaseTrack = 2, Field = 3 }
	
					internal enum Section : byte { Data = 0x20, Header = 0x40 }
	
					internal enum Format : ushort { Old = 0x08, Old2 = 0x02, Normal = 0x01 }
				}

				/// <summary>
				/// Field history flags
				/// </summary>
				public enum FieldFlag : byte 
				{ 
					/// <summary>
					/// Scanning OK.
					/// </summary>
					OK = 0x00, 
					/// <summary>
					/// Focus not found on top side.
					/// </summary>
					NoTopFocus = 0x01, 
					/// <summary>
					/// Focus not found on bottom side.
					/// </summary>
					NoBottomFocus = 0x02, 
					/// <summary>
					/// Z limiter became active and interrupted the scanning on this field.
					/// </summary>
					ZLimiter = 0x04, 
					/// <summary>
					/// X limiter became active and interrupted the scanning on this field.
					/// </summary>
					XLimiter = 0x08, 
					/// <summary>
					/// Y limiter became active and interrupted the scanning on this field.
					/// </summary>
					YLimiter = 0x10, 
					/// <summary>
					/// The scanning was terminated by user request.
					/// </summary>
					Terminated = 0x80, 
					/// <summary>
					/// The field was not scanned.
					/// </summary>
					NotScanned = 0xFF } 

				/// <summary>
				/// Contains the scanning history for a field of view
				/// </summary>
				public struct FieldHistory
				{
					/// <summary>
					/// The result of the scanning on the top side of the plate.
					/// </summary>
					public FieldFlag Top;
					/// <summary>
					/// The result of the scanning on the bottom side of the plate.
					/// </summary>
					public FieldFlag Bottom;
				}
				
				/// <summary>
				/// A scanning zone of linked tracks on a CHORUS emulsion plate.
				/// </summary>
				public class LinkedZone : SySal.Scanning.Plate.LinkedZone
				{
					private class MIPIndexedEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
					{
						public MIPIndexedEmulsionTrack(MIPEmulsionTrackInfo info, Grain [] g, int id)
						{
							m_Info = info;
							m_Grains = g;
							m_Id = id;
						}

						public uint AreaSum() { return m_Info.AreaSum; }

						public static MIPEmulsionTrackInfo GetInfo(MIPEmulsionTrack t) { return MIPIndexedEmulsionTrack.AccessInfo(t); }
					}

					private class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
					{
						public MIPBaseTrack(MIPEmulsionTrackInfo info, SySal.Scanning.MIPIndexedEmulsionTrack topseg, SySal.Scanning.MIPIndexedEmulsionTrack bottomseg, int id)
						{
							m_Info = info;
							m_Top = topseg;
							m_Bottom = bottomseg;
							m_Id = id;
						}

						public static MIPEmulsionTrackInfo GetInfo(MIPBaseTrack t) { return MIPBaseTrack.AccessInfo(t); }
					}

					/// <summary>
					/// Member data on which the PredictedSlope property relies. Can be accessed by derived classes.
					/// </summary>
					protected Vector2 m_PredictedSlope;
					/// <summary>
					/// The predicted slope for this area.
					/// </summary>
					public Vector2 PredictedSlope { get { return m_PredictedSlope; } }
					/// <summary>
					/// Member data on which the SideSlopeTolerance relies. Can be accessed by derived classes.
					/// </summary>
					protected double m_SideSlopeTolerance;
					/// <summary>
					/// The slope tolerance for tracks on either emulsion side.
					/// </summary>
					public double SideSlopeTolerance { get { return m_SideSlopeTolerance; } }
					/// <summary>
					/// Member data on which the GlobalSlopeTolerance relies. Can be accessed by derived classes.
					/// </summary>
					protected double m_GlobalSlopeTolerance;
					/// <summary>
					/// The slope tolerance for base tracks.
					/// </summary>
					public double GlobalSlopeTolerance { get { return m_GlobalSlopeTolerance; } }
					/// <summary>
					/// Member data on which the GoodSlopeTolerance relies. Can be accessed by derived classes.
					/// </summary>
					protected double m_GoodSlopeTolerance;
					/// <summary>
					/// Base tracks that agree with the predicted slope better than GoodSlopeTolerance are enough to stop the scanning.
					/// </summary>
					public double GoodSlopeTolerance { get { return m_GoodSlopeTolerance; } }
					/// <summary>
					/// Member data on which the Fields property relies. Can be accessed by derived classes. 
					/// </summary>
					protected FieldHistory [] m_Fields;
					/// <summary>
					/// Documents the scanning quality for each field of view.
					/// </summary>
					public FieldHistory [] Fields { get { return (FieldHistory [])m_Fields.Clone(); } }
					/// <summary>
					/// Protected constructor. Prevents users from creating LinkedZones without deriving the class. Is implicitly called by constructors of derived classes.
					/// </summary>
					protected LinkedZone() {}
					/// <summary>
					/// Restores a LinkedZone from a stream where it has been previously saved.
					/// </summary>
					/// <param name="str">the input stream to be read.</param>
					public LinkedZone(System.IO.Stream str)
					{
						int i;

						System.IO.BinaryReader r = new System.IO.BinaryReader(str);

						byte infotype = r.ReadByte();
						ushort headerformat = r.ReadUInt16();
						if (infotype == ((byte)Thin.Info.Track | (byte)Thin.Section.Header))
							switch (headerformat)
							{
								case (ushort)Thin.Format.Normal: break;
								case (ushort)Thin.Format.Old: break;
								case (ushort)Thin.Format.Old2: break;
								default: throw new SystemException("Unknown format");
							};

						if (headerformat == (ushort)Thin.Format.Old)
						{
							m_Id.Part3 = r.ReadInt32();
							m_Id.Part0 = m_Id.Part1 = m_Id.Part2 = 0;
						}
						else
						{
							m_Id.Part0 = r.ReadInt32();
							m_Id.Part1 = r.ReadInt32();
							m_Id.Part2 = r.ReadInt32();
							m_Id.Part3 = r.ReadInt32();
						}
	
						m_Center.X = r.ReadSingle();
						m_Center.Y = r.ReadSingle();
						m_PredictedSlope.X = r.ReadSingle(); 
						m_PredictedSlope.Y = r.ReadSingle();
						m_SideSlopeTolerance = r.ReadSingle(); 
						m_GlobalSlopeTolerance = r.ReadSingle(); 
						m_GoodSlopeTolerance = r.ReadSingle();
	
						MIPEmulsionTrack [] toptracks = new MIPEmulsionTrack[r.ReadUInt32()];
						MIPEmulsionTrack [] bottomtracks = new MIPEmulsionTrack[r.ReadUInt32()];
						m_Tracks = new MIPBaseTrack[r.ReadUInt32()];

						m_Fields = new FieldHistory[r.ReadInt32()];

						float topext, topint, bottomint, bottomext;
						topext = r.ReadSingle();
						topint = r.ReadSingle();
						bottomint = r.ReadSingle();
						bottomext = r.ReadSingle();

						for (i = 0; i < m_Fields.Length; m_Fields[i++].Top = (FieldFlag)r.ReadByte());
						for (i = 0; i < m_Fields.Length; m_Fields[i++].Bottom = (FieldFlag)r.ReadByte());
					
						MIPEmulsionTrack [] tkarr = toptracks;
						do
						{
							if (headerformat == (ushort)Thin.Format.Normal)
								for (i = 0; i < tkarr.Length; i++)
								{
									MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
									info.AreaSum = 0;
									info.Field = r.ReadUInt32();
									info.Count = (ushort)r.ReadUInt32();
									info.Intercept.X = r.ReadSingle();
									info.Intercept.Y = r.ReadSingle();
									info.Intercept.Z = r.ReadSingle();
									info.Slope.X = r.ReadSingle();
									info.Slope.Y = r.ReadSingle();
									info.Slope.Z = r.ReadSingle();
									info.Sigma = r.ReadSingle();
									info.TopZ = r.ReadSingle();
									info.BottomZ = r.ReadSingle();
									r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // intercept errors
									r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // slope errors
									tkarr[i] = new MIPIndexedEmulsionTrack(info, null, i);									
								}
							else
								for (i = 0; i < tkarr.Length; i++)
								{
									MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
									info = new MIPEmulsionTrackInfo();
									info.AreaSum = 0;
									info.Field = r.ReadUInt32();
									info.Count = (ushort)r.ReadUInt32();
									r.ReadSingle(); r.ReadSingle(); r.ReadSingle();
									info.Intercept.X = r.ReadSingle();
									info.Intercept.Y = r.ReadSingle();
									info.Intercept.Z = r.ReadSingle();
									info.Slope.X = r.ReadSingle();
									info.Slope.Y = r.ReadSingle();
									info.Slope.Z = r.ReadSingle();
									info.Sigma = r.ReadSingle();
									info.TopZ = r.ReadSingle();
									info.BottomZ = r.ReadSingle();
									tkarr[i] = new MIPIndexedEmulsionTrack(info, null, i);
								}
							tkarr = (tkarr == toptracks) ? bottomtracks : null;
						}
						while (tkarr != null);

						m_Top = new Side(toptracks, topext, topint);
						m_Bottom = new Side(bottomtracks, bottomint, bottomext);

						if (headerformat == (ushort)Thin.Format.Normal)
						{
							for (i = 0; i < m_Tracks.Length; i++)
							{
								MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
								info = new MIPEmulsionTrackInfo();
								info.Count = (ushort)r.ReadUInt32();
								info.Intercept.X = r.ReadSingle();
								info.Intercept.Y = r.ReadSingle();
								info.Intercept.Z = r.ReadSingle();
								info.Slope.X = r.ReadSingle();
								info.Slope.Y = r.ReadSingle();
								info.Slope.Z = r.ReadSingle();
								info.Sigma = r.ReadSingle();
								r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // intercept errors
								r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // slope errors
								int topindex = r.ReadInt32();
								int bottomindex = r.ReadInt32();
								info.AreaSum = ((MIPIndexedEmulsionTrack)m_Top.m_Tracks[topindex]).AreaSum() + ((MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[bottomindex]).AreaSum();
								m_Tracks[i] = new MIPBaseTrack(info, m_Top.m_Tracks[topindex], m_Bottom.m_Tracks[bottomindex], i);
							}
						}
						else
						{
							for (i = 0; i < m_Tracks.Length; i++)
							{
								MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();

								info = new MIPEmulsionTrackInfo();
								info.Count = (ushort)r.ReadUInt32();
								info.Intercept.X = r.ReadSingle();
								info.Intercept.Y = r.ReadSingle();
								info.Intercept.Z = r.ReadSingle();
								info.Slope.X = r.ReadSingle();
								info.Slope.Y = r.ReadSingle();
								info.Slope.Z = r.ReadSingle();
								info.Sigma = r.ReadSingle();
								int topindex = r.ReadInt32();
								int bottomindex = r.ReadInt32();
								info.AreaSum = ((MIPIndexedEmulsionTrack)m_Top.m_Tracks[topindex]).AreaSum() + ((MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[bottomindex]).AreaSum();
								m_Tracks[i] = new MIPBaseTrack(info, m_Top.m_Tracks[topindex], m_Bottom.m_Tracks[bottomindex], i);
							}
						}
					}
					/// <summary>
					/// Saves a LinkedZone to a stream.
					/// </summary>
					/// <param name="str">the output stream where the LinkedZone is to be saved.</param>
					override public void Save(System.IO.Stream str)
					{
						System.IO.BinaryWriter w = new System.IO.BinaryWriter(str);
						w.Write((byte)((byte)Thin.Info.Track | (byte)Thin.Section.Header));
						w.Write((ushort)Thin.Format.Normal);

						w.Write(m_Id.Part0);
						w.Write(m_Id.Part1);
						w.Write(m_Id.Part2);
						w.Write(m_Id.Part3);

						w.Write((float)m_Center.X);
						w.Write((float)m_Center.Y);

						w.Write((float)m_PredictedSlope.X);
						w.Write((float)m_PredictedSlope.Y);
						w.Write((float)m_SideSlopeTolerance);
						w.Write((float)m_GlobalSlopeTolerance);
						w.Write((float)m_GoodSlopeTolerance);
	
						w.Write(m_Top.m_Tracks.Length);
						w.Write(m_Bottom.m_Tracks.Length);
						w.Write(m_Tracks.Length);
						w.Write(m_Fields.Length);

						w.Write((float)m_Top.m_TopZ);
						w.Write((float)m_Top.m_BottomZ);
						w.Write((float)m_Bottom.m_TopZ);
						w.Write((float)m_Bottom.m_BottomZ);

						foreach (FieldHistory f in Fields) w.Write((byte)f.Top);
						foreach (FieldHistory f in Fields) w.Write((byte)f.Bottom);
					
						MIPEmulsionTrack [] tkarr = m_Top.m_Tracks;
						do
						{
							foreach (MIPEmulsionTrack t in tkarr)
							{
								MIPEmulsionTrackInfo info = MIPIndexedEmulsionTrack.GetInfo(t);
								w.Write(info.Field);
								w.Write((uint)info.Count);
								w.Write((float)info.Intercept.X);
								w.Write((float)info.Intercept.Y);
								w.Write((float)info.Intercept.Z);
								w.Write((float)info.Slope.X);
								w.Write((float)info.Slope.Y);
								w.Write((float)info.Slope.Z);
								w.Write((float)info.Sigma);
								w.Write((float)info.TopZ);
								w.Write((float)info.BottomZ);
								w.Write((float)0.0f); w.Write((float)0.0f); w.Write((float)0.0f); // slope and intercept errors
								w.Write((float)0.0f); w.Write((float)0.0f); w.Write((float)0.0f);
							}
							tkarr = (tkarr == m_Top.m_Tracks) ? m_Bottom.m_Tracks : null;
						}
						while (tkarr != null);

						foreach (MIPBaseTrack l in m_Tracks)
						{
							MIPEmulsionTrackInfo info = MIPBaseTrack.GetInfo(l);
							w.Write((uint)info.Count);
							w.Write((float)info.Intercept.X);
							w.Write((float)info.Intercept.Y);
							w.Write((float)info.Intercept.Z);
							w.Write((float)info.Slope.X);
							w.Write((float)info.Slope.Y);
							w.Write((float)info.Slope.Z);
							w.Write((float)info.Sigma);
							w.Write((float)0.0f); w.Write((float)0.0f); w.Write((float)0.0f); // slope and intercept errors
							w.Write((float)0.0f); w.Write((float)0.0f); w.Write((float)0.0f);
							w.Write(l.Top.Id);
							w.Write(l.Bottom.Id);
						}
					}
				}
			}

			namespace OPERA
			{
				class File
				{
					public enum Info : byte { Track = 1, BaseTrack = 2, Field = 3 }
	
					public enum Section : byte { Data = 0x20, Header = 0x40 }
	
					public enum Format : ushort { Old = 0x08, Old2 = 0x02, NoExtents = 0x01, Normal = 0x03, NormalWithIndex = 0x04, NormalDoubleWithIndex = 0x05, Detailed = 0x06, MultiSection = 0x07 }
				}

				/// <summary>
				/// A scanning zone of linked tracks on a OPERA emulsion plate.
				/// </summary>
				public class LinkedZone : SySal.Scanning.Plate.LinkedZone
				{
                    /// <summary>
                    /// Section tag for LinkedZone in TLG files.
                    /// </summary>
                    public const byte SectionTag = 0x01;
					/// <summary>
					/// Member data on which the Extents property relies. Can be accessed by protected classes.
					/// </summary>
					protected Rectangle m_Extents;	
					/// <summary>
					/// The extents of the scanned area.
					/// </summary>
					public Rectangle Extents { get { return m_Extents; } }
					/// <summary>
					/// Member data on which the Transformation property relies. Can be accessed by protected classes.
					/// </summary>
					protected SySal.DAQSystem.Scanning.IntercalibrationInfo m_Transform;	
					/// <summary>
					/// The transformation applied to stage coordinates to obtain the plate coordinates.
					/// The RX and RY fields are not used.
					/// </summary>
					public SySal.DAQSystem.Scanning.IntercalibrationInfo Transform { get { return m_Transform; } }
					/// <summary>
					/// Stores information about the origin of an emulsion track in the Raw Data (RWD) files.
					/// </summary>					
					public struct TrackIndexEntry
					{
						/// <summary>
						/// The index of the fragment to which the original raw microtrack belongs.
						/// </summary>
						public int Fragment;
						/// <summary>
						/// The number of the view in the fragment to which the original raw microtrack belongs.
						/// </summary>
						public int View;
						/// <summary>
						/// The number of the track in the view to which the original raw microtrack belongs.
						/// </summary>
						public int Track;
					}

					/// <summary>
					/// Contains information about one scanning view.
					/// </summary>
					public class View
					{
						/// <summary>
						/// Member data on which the Side property relies.
						/// </summary>
						protected internal Side m_Side;
						/// <summary>
						/// The side this View belongs to.
						/// </summary>
						public Side Side { get { return m_Side; } }
						/// <summary>
						/// Member data on which the Id property relies.
						/// </summary>
						protected internal int m_Id;
						/// <summary>
						/// Id of the view.
						/// </summary>
						public int Id { get { return m_Id; } }
						/// <summary>
						/// Member data on which the Position property relies.
						/// </summary>
						protected internal SySal.BasicTypes.Vector2 m_Position;
						/// <summary>
						/// Position of the center of the view.
						/// </summary>
						public SySal.BasicTypes.Vector2 Position { get { return m_Position; } }
						/// <summary>
						/// Member data on which the TopZ property relies.
						/// </summary>
						protected internal double m_TopZ;
						/// <summary>
						/// Position of the top extent of the emulsion layer.						
						/// </summary>
						public double TopZ { get { return m_TopZ; } }
						/// <summary>
						/// Member data on which the BottomZ property relies.
						/// </summary>
						protected internal double m_BottomZ;
						/// <summary>
						/// Position of the bottom extent of the emulsion layer.
						/// </summary>
						public double BottomZ { get { return m_BottomZ; } }
						/// <summary>
						/// The number of tracks contained in the view.
						/// </summary>
						public int Length { get { return m_Tracks.Length; } }
						/// <summary>
						/// Provides access to the list of tracks contained in the view.
						/// </summary>
						public MIPIndexedEmulsionTrack this[int index] { get { return m_Tracks[index]; } }
						/// <summary>
						/// The list of tracks in the view. It is hidden, but can be used by derived classes.
						/// </summary>
						protected internal MIPIndexedEmulsionTrack [] m_Tracks;
						/// <summary>
						/// Protected constructor. Prevents constructing an invalid View and forces use of derived classes.
						/// </summary>
						protected internal View() 
						{
							m_Tracks = new MIPIndexedEmulsionTrack[0];
						}

						internal View(Side side, int id, double px, double py, double topz, double bottomz)
						{
							m_Side = side;
							m_Id = id;
							m_TopZ = topz;
							m_BottomZ = bottomz;
							m_Position.X = px;
							m_Position.Y = py;
							m_Tracks = new MIPIndexedEmulsionTrack[0];
						}
					};

					/// <summary>
					/// The Side of an OPERA Linked Zone. Contains View information in addition to standard Side data.
					/// </summary>
					public class Side : SySal.Scanning.Plate.Side
					{
						/// <summary>
						/// Array of views in this side.
						/// </summary>
						protected View [] m_Views;
						/// <summary>
						/// Returns view info for a view in this side.
						/// </summary>
						/// <param name="id">the id of the view info to be returned.</param>
						/// <returns>the specified view info</returns>
						public virtual View View(int id) { return m_Views[id]; }
						/// <summary>
						/// The number of view info in this side.
						/// </summary>
						public virtual int ViewCount { get { return m_Views.Length; } }

						internal void SetViews(View [] vw)
						{
							m_Views = vw;
						}
						/// <summary>
						/// Protected constructor. Prevents misuse of the class by constructing invalid Sides and forcing use of derived classes.
						/// </summary>
						protected Side() { m_Views = new View[0]; }
	
						internal Side(double topz, double bottomz) 
						{ 
							m_Views = new View[0]; 
							m_TopZ = topz;
							m_BottomZ = bottomz;
						}

						internal void SetTracks(MIPIndexedEmulsionTrack [] tks) { m_Tracks = tks; }
					}

					/// <summary>
					/// M.i.p. track in emulsion with an ordinal id and information about the raw data that originated it.
					/// </summary>
					public class MIPIndexedEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
					{
						/// <summary>
						/// Member data on which the OriginalRawData relies. Can be accessed in derived classes.
						/// </summary>
						protected internal TrackIndexEntry m_OriginalRawData;
						/// <summary>
						/// Retrieves information to trace the origin of this MIPEmulsionTrack in the raw data.
						/// </summary>
						public virtual TrackIndexEntry OriginalRawData { get { return m_OriginalRawData; } }			
						/// <summary>
						/// Protected constructor. Prevents users from creating MIPIndexedEmulsionTracks without deriving the class. Is implicitly called by constructors of derived classes.
						/// </summary>
						protected MIPIndexedEmulsionTrack() {}

						/// <summary>
						/// Protected member on which the View property relies. Can be accessed by derived classes.
						/// </summary>
						protected View m_View;
						/// <summary>
						/// The view this M.i.p. track belongs to.
						/// </summary>
						public virtual View View { get { return m_View; } }

						internal MIPIndexedEmulsionTrack(MIPEmulsionTrackInfo info, int id, View vw)
						{
							m_Info = info;
							m_Grains = null;
							m_Id = id;
							m_View = vw;
						}

						internal MIPIndexedEmulsionTrack(MIPEmulsionTrackInfo info, Grain [] m_Grains, int id, View vw)
						{
							m_Info = info;
							m_Grains = null;
							m_Id = id;
							m_View = vw;
						}

						internal void SetView(View vw)
						{
							m_View = vw;
						}

						internal static MIPEmulsionTrackInfo GetInfo(MIPEmulsionTrack t) { return MIPIndexedEmulsionTrack.AccessInfo(t); }

						internal static Grain [] GetGrains(MIPEmulsionTrack t) { return MIPIndexedEmulsionTrack.AccessGrains(t); }
					}

					private class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
					{
						public MIPBaseTrack(MIPEmulsionTrackInfo info, SySal.Scanning.MIPIndexedEmulsionTrack topseg, SySal.Scanning.MIPIndexedEmulsionTrack bottomseg, int id)
						{
							m_Info = info;
							m_Top = topseg;
							m_Bottom = bottomseg;
							m_Id = id;
						}

						public static MIPEmulsionTrackInfo GetInfo(SySal.Scanning.MIPBaseTrack t) { return MIPBaseTrack.AccessInfo(t); }
					}
					/// <summary>
					/// Protected constructor. Prevents users from creating LinkedZones without deriving the class. Is implicitly called by constructors of derived classes.
					/// </summary>
					protected LinkedZone() {}

					/// <summary>
					/// Restores a LinkedZone from a stream where it has been previously saved.
					/// </summary>
					/// <param name="str">the input stream to be read.</param>
					public LinkedZone(System.IO.Stream str)
					{
						System.IO.BinaryReader r = new System.IO.BinaryReader(str);
						byte infotype = r.ReadByte();
						ushort headerformat = r.ReadUInt16();
						if (infotype == ((byte)File.Info.Track | (byte)File.Section.Header))
							switch (headerformat)
							{
                                case (ushort)File.Format.MultiSection: break;
								case (ushort)File.Format.Detailed: break;
								case (ushort)File.Format.NormalDoubleWithIndex: break;
								case (ushort)File.Format.NormalWithIndex: break;
								case (ushort)File.Format.Normal: break;
								case (ushort)File.Format.NoExtents: break;
								case (ushort)File.Format.Old: break;
								case (ushort)File.Format.Old2: break;
	
								default: throw new SystemException("Unknown format");
							};

                        if (headerformat == (ushort)File.Format.MultiSection)
                        {
                            if (r.ReadByte() != SectionTag) throw new Exception("The first section in a TLG file must contain tracks!");
                            r.ReadInt64();
                        }
						
						int i;

						m_Transform.MXX = m_Transform.MYY = 1.0;
						m_Transform.MXY = m_Transform.MYX = 0.0;
						m_Transform.TX = m_Transform.TY = 0.0;
						m_Transform.RX = m_Transform.RY = 0.0;

						if (headerformat == (ushort)File.Format.Old)
						{
							m_Id.Part3 = r.ReadInt32();
							m_Id.Part0 = m_Id.Part1 = m_Id.Part2 = 0;
						}
						else
						{
							m_Id.Part0 = r.ReadInt32();
							m_Id.Part1 = r.ReadInt32();
							m_Id.Part2 = r.ReadInt32();
							m_Id.Part3 = r.ReadInt32();
						}
	
						if (headerformat == (ushort)File.Format.Detailed || headerformat == (ushort)File.Format.MultiSection)
						{
							m_Center.X = r.ReadDouble();
							m_Center.Y = r.ReadDouble();
							m_Extents.MinX = r.ReadDouble();
							m_Extents.MaxX = r.ReadDouble(); 
							m_Extents.MinY = r.ReadDouble();
							m_Extents.MaxY = r.ReadDouble(); 
							m_Transform.MXX = r.ReadDouble();
							m_Transform.MXY = r.ReadDouble();
							m_Transform.MYX = r.ReadDouble();
							m_Transform.MYY = r.ReadDouble();
							m_Transform.TX = r.ReadDouble();
							m_Transform.TY = r.ReadDouble();
							m_Transform.RX = r.ReadDouble();
							m_Transform.RY = r.ReadDouble();							
							
							View [] topviews = new View[r.ReadInt32()];
							View [] bottomviews = new View[r.ReadInt32()];
							m_Top = new Side(r.ReadDouble(), r.ReadDouble());
							m_Bottom = new Side(r.ReadDouble(), r.ReadDouble());
							((Side)m_Top).SetViews(topviews);
							((Side)m_Bottom).SetViews(bottomviews);
							System.Collections.ArrayList [] topvwtklist = new System.Collections.ArrayList[topviews.Length];
							System.Collections.ArrayList [] bottomvwtklist = new System.Collections.ArrayList[bottomviews.Length];
							View [] vwarr = topviews;
							System.Collections.ArrayList [] listarr = topvwtklist;
							Side side = (Side)m_Top;
							do
							{
								for (i = 0; i < vwarr.Length; i++)
								{
									vwarr[i] = new View(side, r.ReadInt32(), r.ReadDouble(), r.ReadDouble(), r.ReadDouble(), r.ReadDouble());
									listarr[i] = new System.Collections.ArrayList();
								}
								vwarr = (vwarr == topviews) ? bottomviews : null;
								listarr = (listarr == topvwtklist) ? bottomvwtklist : null;
								side = (side == m_Top) ? (Side)m_Bottom : null;								
							}
							while (vwarr != null);

							MIPIndexedEmulsionTrack [] toptracks = new MIPIndexedEmulsionTrack[r.ReadUInt32()];
							MIPIndexedEmulsionTrack [] bottomtracks = new MIPIndexedEmulsionTrack[r.ReadUInt32()];
							m_Tracks = new MIPBaseTrack[r.ReadUInt32()];

							MIPIndexedEmulsionTrack [] tkarr = toptracks;
							vwarr = topviews;
							listarr = topvwtklist;
							int viewid;
							MIPIndexedEmulsionTrack newtk;
							do
							{
								for (i = 0; i < tkarr.Length; i++)
								{
									MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
									info.Field = r.ReadUInt32();
									info.AreaSum = r.ReadUInt32();
									info.Count = (ushort)r.ReadUInt32();
									info.Intercept.X = r.ReadDouble();
									info.Intercept.Y = r.ReadDouble();
									info.Intercept.Z = r.ReadDouble();
									info.Slope.X = r.ReadDouble();
									info.Slope.Y = r.ReadDouble();
									info.Slope.Z = r.ReadDouble();
									info.Sigma = r.ReadDouble();
									info.TopZ = r.ReadDouble();
									info.BottomZ = r.ReadDouble();									
									tkarr[i] = newtk = new MIPIndexedEmulsionTrack(info, i, vwarr[viewid = r.ReadInt32()]);
									listarr[viewid].Add(newtk);
								}
								for (i = 0; i < vwarr.Length; i++)
								{
									vwarr[i].m_Tracks = (MIPIndexedEmulsionTrack [])listarr[i].ToArray(typeof(MIPIndexedEmulsionTrack));
									listarr[i] = null;
								}
								tkarr = (tkarr == toptracks) ? bottomtracks : null;
								vwarr = (vwarr == topviews) ? bottomviews : null;
								listarr = (listarr == topvwtklist) ? bottomvwtklist : null;
							}
							while (tkarr != null);

							topvwtklist = null;
							bottomvwtklist = null;
							((Side)m_Top).SetTracks(toptracks);
							((Side)m_Bottom).SetTracks(bottomtracks);							

							for (i = 0; i < m_Tracks.Length; i++)
							{
								MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
								info = new MIPEmulsionTrackInfo();
								info.AreaSum = (ushort)r.ReadUInt32();
								info.Count = (ushort)r.ReadUInt32();
								info.Intercept.X = r.ReadDouble();
								info.Intercept.Y = r.ReadDouble();
								info.Intercept.Z = r.ReadDouble();
								info.Slope.X = r.ReadDouble();
								info.Slope.Y = r.ReadDouble();
								info.Slope.Z = r.ReadDouble();
								info.Sigma = r.ReadDouble();
								MIPIndexedEmulsionTrack toptk = (MIPIndexedEmulsionTrack)m_Top.m_Tracks[r.ReadUInt32()];
								MIPIndexedEmulsionTrack bottomtk = (MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[r.ReadUInt32()];
								info.TopZ = MIPIndexedEmulsionTrack.GetInfo(toptk).TopZ;
								info.BottomZ = MIPIndexedEmulsionTrack.GetInfo(bottomtk).BottomZ;
								m_Tracks[i] = new MIPBaseTrack(info, toptk, bottomtk, i);
							}
							tkarr = (MIPIndexedEmulsionTrack [])m_Top.m_Tracks;
							while (true)
							{
								for (i = 0; i < tkarr.Length; i++)
								{
									tkarr[i].m_OriginalRawData.Fragment = r.ReadInt32();
									tkarr[i].m_OriginalRawData.View = r.ReadInt32();
									tkarr[i].m_OriginalRawData.Track = r.ReadInt32();
								}
								if (tkarr == (MIPIndexedEmulsionTrack [])m_Top.m_Tracks) tkarr = (MIPIndexedEmulsionTrack [])m_Bottom.m_Tracks; else break;
							}
						}
						else if (headerformat == (ushort)File.Format.NormalDoubleWithIndex)
						{
							m_Center.X = r.ReadDouble();
							m_Center.Y = r.ReadDouble();
							m_Extents.MinX = r.ReadDouble();
							m_Extents.MaxX = r.ReadDouble(); 
							m_Extents.MinY = r.ReadDouble();
							m_Extents.MaxY = r.ReadDouble(); 
							r.ReadSingle(); // for format compliance
							
							View topview = new View();
							View bottomview = new View();
							MIPIndexedEmulsionTrack [] toptracks = new MIPIndexedEmulsionTrack[r.ReadUInt32()];
							MIPIndexedEmulsionTrack [] bottomtracks = new MIPIndexedEmulsionTrack[r.ReadUInt32()];
							m_Tracks = new MIPBaseTrack[r.ReadUInt32()];

							i = r.ReadInt32(); // count of fields

							double topext, topint, bottomint, bottomext;
							topext = r.ReadDouble();
							topint = r.ReadDouble();
							bottomint = r.ReadDouble();
							bottomext = r.ReadDouble();
					
							r.ReadBytes(2 * i); // skip fields

							MIPIndexedEmulsionTrack [] tkarr = toptracks;
							View vw = topview;
							do
							{
								for (i = 0; i < tkarr.Length; i++)
								{
									MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
									info.Field = r.ReadUInt32();
									info.AreaSum = r.ReadUInt32();
									info.Count = (ushort)r.ReadUInt32();
									info.Intercept.X = r.ReadDouble();
									info.Intercept.Y = r.ReadDouble();
									info.Intercept.Z = r.ReadDouble();
									info.Slope.X = r.ReadDouble();
									info.Slope.Y = r.ReadDouble();
									info.Slope.Z = r.ReadDouble();
									info.Sigma = r.ReadDouble();
									info.TopZ = r.ReadDouble();
									info.BottomZ = r.ReadDouble();
									tkarr[i] = new MIPIndexedEmulsionTrack(info, i, vw);
								}
								tkarr = (tkarr == toptracks) ? bottomtracks : null;
								vw = (vw == topview) ? bottomview : null;
							}
							while (tkarr != null);

							m_Top = new Side(topext, topint);
							m_Bottom = new Side(bottomint, bottomext);
							((Side)m_Top).SetViews(new View[1] { topview });
							((Side)m_Bottom).SetViews(new View[1] { bottomview });
							((Side)m_Top).SetTracks(toptracks);
							((Side)m_Bottom).SetTracks(bottomtracks);
							topview.m_Side = (Side)m_Top;
							bottomview.m_Side = (Side)m_Bottom;
							topview.m_Tracks = toptracks;
							bottomview.m_Tracks = bottomtracks;
							topview.m_Id = 0;
							bottomview.m_Id = 0;
							topview.m_Position.X = bottomview.m_Position.X = m_Center.X;
							topview.m_Position.Y = bottomview.m_Position.Y = m_Center.Y;
							topview.m_TopZ = m_Top.TopZ;
							topview.m_BottomZ = m_Top.BottomZ;
							bottomview.m_TopZ = m_Bottom.TopZ;
							bottomview.m_BottomZ = m_Bottom.BottomZ;

							for (i = 0; i < m_Tracks.Length; i++)
							{
								MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
								info = new MIPEmulsionTrackInfo();
								info.AreaSum = (ushort)r.ReadUInt32();
								info.Count = (ushort)r.ReadUInt32();
								info.Intercept.X = r.ReadDouble();
								info.Intercept.Y = r.ReadDouble();
								info.Intercept.Z = r.ReadDouble();
								info.Slope.X = r.ReadDouble();
								info.Slope.Y = r.ReadDouble();
								info.Slope.Z = r.ReadDouble();
								info.Sigma = r.ReadDouble();
								if (headerformat == (ushort)File.Format.NoExtents)
								{
									r.ReadDouble(); r.ReadDouble(); r.ReadDouble(); // intercept errors
									r.ReadDouble(); r.ReadDouble(); r.ReadDouble(); // slope errors
								}
								MIPIndexedEmulsionTrack toptk = (MIPIndexedEmulsionTrack)m_Top.m_Tracks[r.ReadUInt32()];
								MIPIndexedEmulsionTrack bottomtk = (MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[r.ReadUInt32()];
								info.TopZ = MIPIndexedEmulsionTrack.GetInfo(toptk).TopZ;
								info.BottomZ = MIPIndexedEmulsionTrack.GetInfo(bottomtk).BottomZ;
								m_Tracks[i] = new MIPBaseTrack(info, toptk, bottomtk, i);
							}
							tkarr = (MIPIndexedEmulsionTrack [])m_Top.m_Tracks;
							while (true)
							{
								for (i = 0; i < tkarr.Length; i++)
								{
									tkarr[i].m_OriginalRawData.Fragment = r.ReadInt32();
									tkarr[i].m_OriginalRawData.View = r.ReadInt32();
									tkarr[i].m_OriginalRawData.Track = r.ReadInt32();
								}
								if (tkarr == (MIPIndexedEmulsionTrack [])m_Top.m_Tracks) tkarr = (MIPIndexedEmulsionTrack [])m_Bottom.m_Tracks; else break;
							}
						}
						else
						{
							m_Center.X = r.ReadSingle();
							m_Center.Y = r.ReadSingle();
							if (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalWithIndex)
							{
								m_Extents.MinX = r.ReadSingle();
								m_Extents.MaxX = r.ReadSingle(); 
								m_Extents.MinY = r.ReadSingle();
								m_Extents.MaxY = r.ReadSingle(); 
								r.ReadSingle(); //skip a meaningless float...
							}
							else
							{
								r.ReadSingle(); r.ReadSingle(); 
								r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); 
								m_Extents.MinX = m_Extents.MaxX = m_Center.X;
								m_Extents.MinY = m_Extents.MaxY = m_Center.Y;
							}
	
							View topview = new View();
							View bottomview = new View();
							MIPIndexedEmulsionTrack [] toptracks = new MIPIndexedEmulsionTrack[r.ReadUInt32()];
							MIPIndexedEmulsionTrack [] bottomtracks = new MIPIndexedEmulsionTrack[r.ReadUInt32()];
							m_Tracks = new MIPBaseTrack[r.ReadUInt32()];

							i = r.ReadInt32(); // count of fields

							double topext, topint, bottomint, bottomext;
							topext = r.ReadSingle();
							topint = r.ReadSingle();
							bottomint = r.ReadSingle();
							bottomext = r.ReadSingle();
					
							r.ReadBytes(2 * i); // skip fields

							MIPIndexedEmulsionTrack [] tkarr = toptracks;
							View vw = topview;
							do
							{
								if (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalWithIndex)
								{
									for (i = 0; i < tkarr.Length; i++)
									{
										MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
										info.Field = r.ReadUInt32();
										info.AreaSum = r.ReadUInt32();
										info.Count = (ushort)r.ReadUInt32();
										info.Intercept.X = r.ReadSingle();
										info.Intercept.Y = r.ReadSingle();
										info.Intercept.Z = r.ReadSingle();
										info.Slope.X = r.ReadSingle();
										info.Slope.Y = r.ReadSingle();
										info.Slope.Z = r.ReadSingle();
										info.Sigma = r.ReadSingle();
										info.TopZ = r.ReadSingle();
										info.BottomZ = r.ReadSingle();
										tkarr[i] = new MIPIndexedEmulsionTrack(info, i, vw);
									}
								}
								else
								{
									if (headerformat == (ushort)File.Format.NoExtents)
										for (i = 0; i < tkarr.Length; i++)
										{
											MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
											info.AreaSum = 0;
											info.Field = r.ReadUInt32();
											info.Count = (ushort)r.ReadUInt32();
											info.Intercept.X = r.ReadSingle();
											info.Intercept.Y = r.ReadSingle();
											info.Intercept.Z = r.ReadSingle();
											info.Slope.X = r.ReadSingle();
											info.Slope.Y = r.ReadSingle();
											info.Slope.Z = r.ReadSingle();
											info.Sigma = r.ReadSingle();
											info.TopZ = r.ReadSingle();
											info.BottomZ = r.ReadSingle();
											r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // intercept errors
											r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // slope errors
											tkarr[i] = new MIPIndexedEmulsionTrack(info, i, vw);
										}
									else
										for (i = 0; i < tkarr.Length; i++)
										{											
											MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
											info.AreaSum = 0;
											info.Field = r.ReadUInt32();
											info.Count = (ushort)r.ReadUInt32();
											r.ReadSingle(); r.ReadSingle(); r.ReadSingle();
											info.Intercept.X = r.ReadSingle();
											info.Intercept.Y = r.ReadSingle();
											info.Intercept.Z = r.ReadSingle();
											info.Slope.X = r.ReadSingle();
											info.Slope.Y = r.ReadSingle();
											info.Slope.Z = r.ReadSingle();
											info.Sigma = r.ReadSingle();
											info.TopZ = r.ReadSingle();
											info.BottomZ = r.ReadSingle();
											tkarr[i] = new MIPIndexedEmulsionTrack(info, i, vw);
										}
								}
								tkarr = (tkarr == toptracks) ? bottomtracks : null;
								vw = (vw == topview) ? bottomview : null;
							}
							while (tkarr != null);

							m_Top = new Side(topext, topint);
							m_Bottom = new Side(bottomint, bottomext);
							((Side)m_Top).SetViews(new View[1] { topview });
							((Side)m_Bottom).SetViews(new View[1] { bottomview });
							((Side)m_Top).SetTracks(toptracks);
							((Side)m_Bottom).SetTracks(bottomtracks);
							topview.m_Side = (Side)m_Top;
							bottomview.m_Side = (Side)m_Bottom;
							topview.m_Tracks = toptracks;
							bottomview.m_Tracks = bottomtracks;
							topview.m_Id = 0;
							bottomview.m_Id = 0;
							topview.m_Position.X = bottomview.m_Position.X = m_Center.X;
							topview.m_Position.Y = bottomview.m_Position.Y = m_Center.Y;
							topview.m_TopZ = m_Top.TopZ;
							topview.m_BottomZ = m_Top.BottomZ;
							bottomview.m_TopZ = m_Bottom.TopZ;
							bottomview.m_BottomZ = m_Bottom.BottomZ;

							if (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalWithIndex)
							{
								for (i = 0; i < m_Tracks.Length; i++)
								{
									MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
									info.AreaSum = (ushort)r.ReadUInt32();
									info.Count = (ushort)r.ReadUInt32();
									info.Intercept.X = r.ReadSingle();
									info.Intercept.Y = r.ReadSingle();
									info.Intercept.Z = r.ReadSingle();
									info.Slope.X = r.ReadSingle();
									info.Slope.Y = r.ReadSingle();
									info.Slope.Z = r.ReadSingle();
									info.Sigma = r.ReadSingle();
									if (headerformat == (ushort)File.Format.NoExtents)
									{
										r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // intercept errors
										r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // slope errors
									}
									MIPIndexedEmulsionTrack toptk = (MIPIndexedEmulsionTrack)m_Top.m_Tracks[r.ReadUInt32()];
									MIPIndexedEmulsionTrack bottomtk = (MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[r.ReadUInt32()];
									info.TopZ = MIPIndexedEmulsionTrack.GetInfo(toptk).TopZ;
									info.BottomZ = MIPIndexedEmulsionTrack.GetInfo(bottomtk).BottomZ;
									m_Tracks[i] = new MIPBaseTrack(info, toptk, bottomtk, i);
								}
							}
							else
							{
								if (headerformat == (ushort)File.Format.NoExtents)
								{
									for (i = 0; i < m_Tracks.Length; i++)
									{
										MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
										info.AreaSum = 0;
										info.Count = (ushort)r.ReadUInt32();
										info.Intercept.X = r.ReadSingle();
										info.Intercept.Y = r.ReadSingle();
										info.Intercept.Z = r.ReadSingle();
										info.Slope.X = r.ReadSingle();
										info.Slope.Y = r.ReadSingle();
										info.Slope.Z = r.ReadSingle();
										info.Sigma = r.ReadSingle();
										r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // intercept errors
										r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); // slope errors
										MIPIndexedEmulsionTrack toptk = (MIPIndexedEmulsionTrack)m_Top.m_Tracks[r.ReadUInt32()];
										MIPIndexedEmulsionTrack bottomtk = (MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[r.ReadUInt32()];
										info.TopZ = MIPIndexedEmulsionTrack.GetInfo(toptk).TopZ;
										info.BottomZ = MIPIndexedEmulsionTrack.GetInfo(bottomtk).BottomZ;										
										m_Tracks[i] = new MIPBaseTrack(info, toptk, bottomtk, i);
									}
								}
								else
								{
									for (i = 0; i < m_Tracks.Length; i++)
									{
										MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
										info.Count = (ushort)r.ReadUInt32();
										info.Intercept.X = r.ReadSingle();
										info.Intercept.Y = r.ReadSingle();
										info.Intercept.Z = r.ReadSingle();
										info.Slope.X = r.ReadSingle();
										info.Slope.Y = r.ReadSingle();
										info.Slope.Z = r.ReadSingle();
										info.Sigma = r.ReadSingle();										
										MIPIndexedEmulsionTrack toptk = (MIPIndexedEmulsionTrack)m_Top.m_Tracks[r.ReadUInt32()];
										MIPIndexedEmulsionTrack bottomtk = (MIPIndexedEmulsionTrack)m_Bottom.m_Tracks[r.ReadUInt32()];
										info.AreaSum = MIPIndexedEmulsionTrack.GetInfo(toptk).AreaSum + MIPIndexedEmulsionTrack.GetInfo(bottomtk).AreaSum;
										info.TopZ = MIPIndexedEmulsionTrack.GetInfo(toptk).TopZ;
										info.BottomZ = MIPIndexedEmulsionTrack.GetInfo(bottomtk).BottomZ;										
										m_Tracks[i] = new MIPBaseTrack(info, toptk, bottomtk, i);
									}
								}
							}
							if (headerformat == (ushort)File.Format.NormalWithIndex)
							{
								tkarr = (MIPIndexedEmulsionTrack [])m_Top.m_Tracks;
								while (true)
								{
									for (i = 0; i < tkarr.Length; i++)
									{
										tkarr[i].m_OriginalRawData.Fragment = r.ReadInt32();
										tkarr[i].m_OriginalRawData.View = r.ReadInt32();
										tkarr[i].m_OriginalRawData.Track = r.ReadInt32();
									}
									if (tkarr == (MIPIndexedEmulsionTrack [])m_Top.m_Tracks) tkarr = (MIPIndexedEmulsionTrack [])m_Bottom.m_Tracks; else break;
								}
							}
						}
					}

                    /// <summary>
                    /// Finds a section in a multi-section stream.
                    /// </summary>
                    /// <param name="str">the stream in which the section is to be searched.</param>
                    /// <param name="tag">the section tag to be searched.</param>
                    /// <param name="resetifmissing">if <c>true</c>, when the section is not found the stream is reset to the position it had at the beginning of the search; if <c>false</c>, when the section is not found the stream is set to its end (where a new section can be added). This parameter is ignored when the section is found.</param>
                    /// <returns><c>true</c> if the section has been found, <c>false</c> otherwise.</returns>
                    public static bool FindSection(System.IO.Stream str, byte tag, bool resetifmissing)
                    {
                        long startpos = str.Position;
                        str.Seek(0, SeekOrigin.End);
                        long endpos = str.Position;
                        str.Position = 0;                        

						System.IO.BinaryReader r = new System.IO.BinaryReader(str);
						byte infotype = r.ReadByte();
						ushort headerformat = r.ReadUInt16();
                        if ((infotype != ((byte)File.Info.Track | (byte)File.Section.Header)) || (headerformat != (ushort)File.Format.MultiSection))
                        {
                            throw new Exception("This stream does not support multi-section operation.");
                        }
                        long currentpos = str.Position;
                        while (currentpos < endpos)
                        {
                            str.Position = currentpos;
                            if (r.ReadByte() == tag)
                            {
                                r.ReadInt64();
                                return true;
                            }
                            currentpos = r.ReadInt64();
                        }
                        str.Position = (resetifmissing) ? startpos : endpos;
                        return false;
                    }                    

					/// <summary>
					/// Saves a LinkedZone to a stream.
					/// </summary>
					/// <param name="str">the output stream where the LinkedZone is to be saved.</param>
					override public void Save(System.IO.Stream str)
					{
                        long nextsectionpos = 0;
                        long nextsectionref;
						System.IO.BinaryWriter w = new System.IO.BinaryWriter(str);
						w.Write((byte)((byte)File.Info.Track | (byte)File.Section.Header));
						w.Write((ushort)File.Format.MultiSection);
                        w.Write(SectionTag);
                        nextsectionref = str.Position;
                        w.Write(nextsectionpos);

                        Identifier tId = Id;
						w.Write(tId.Part0);
						w.Write(tId.Part1);
						w.Write(tId.Part2);
						w.Write(tId.Part3);

                        Vector2 tCenter = Center;
						w.Write(tCenter.X);
						w.Write(tCenter.Y);

                        Rectangle tExtents = Extents;
						w.Write(tExtents.MinX);
						w.Write(tExtents.MaxX);
						w.Write(tExtents.MinY);
						w.Write(tExtents.MaxY);

                        SySal.DAQSystem.Scanning.IntercalibrationInfo tTransform = Transform;
						w.Write(tTransform.MXX);
						w.Write(tTransform.MXY);
						w.Write(tTransform.MYX);
						w.Write(tTransform.MYY);
						w.Write(tTransform.TX);
						w.Write(tTransform.TY);
						w.Write(tTransform.RX);
						w.Write(tTransform.RY);
						
						w.Write(((Side)Top).ViewCount);
						w.Write(((Side)Bottom).ViewCount);

						w.Write(Top.TopZ);
						w.Write(Top.BottomZ);
						w.Write(Bottom.TopZ);
						w.Write(Bottom.BottomZ);

						Side side = (Side)Top;
						do
						{
							int i;
							for (i = 0; i < side.ViewCount; i++)
							{
								View vw = side.View(i);
								w.Write(vw.Id);
                                Vector2 tPosition = vw.Position;
								w.Write(tPosition.X);
								w.Write(tPosition.Y);
								w.Write(vw.TopZ);
								w.Write(vw.BottomZ);
							}
							side = (side == (Side)Top) ? (Side)Bottom : null;
						}
						while (side != null);

						w.Write(Top.Length);
						w.Write(Bottom.Length);
						w.Write(Length);

                        int titer;
                        side = (Side)Top;
						do
						{
                            for (titer = 0; titer < side.Length; titer++)							
							{
                                MIPIndexedEmulsionTrack t = (MIPIndexedEmulsionTrack)side[titer];
								MIPEmulsionTrackInfo info = t.Info;
								w.Write((uint)info.Field);
								w.Write((uint)info.AreaSum);
								w.Write((uint)info.Count);
								w.Write(info.Intercept.X);
								w.Write(info.Intercept.Y);
								w.Write(info.Intercept.Z);
								w.Write(info.Slope.X);
								w.Write(info.Slope.Y);
								w.Write(info.Slope.Z);
								w.Write(info.Sigma);
								w.Write(info.TopZ);
								w.Write(info.BottomZ);
								w.Write(t.View.Id);
							}
							side = (side == (Side)Top) ? (Side)Bottom : null;
						}
						while (side != null);

                        for (titer = 0; titer < Length; titer++)
						{
                            SySal.Scanning.MIPBaseTrack b = this[titer];
							MIPEmulsionTrackInfo info = b.Info;
							w.Write((uint)info.AreaSum);
							w.Write((uint)info.Count);
							w.Write(info.Intercept.X);
							w.Write(info.Intercept.Y);
							w.Write(info.Intercept.Z);
							w.Write(info.Slope.X);
							w.Write(info.Slope.Y);
							w.Write(info.Slope.Z);
							w.Write(info.Sigma);
							w.Write(b.Top.Id);
							w.Write(b.Bottom.Id);
						}

                        side = (Side)Top;
						while (true)
						{
                            for (titer = 0; titer < side.Length; titer++)							
							{
                                SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry tie = ((MIPIndexedEmulsionTrack)side[titer]).OriginalRawData;
								w.Write(tie.Fragment);
								w.Write(tie.View);
								w.Write(tie.Track);
							}
                            if (side == (Side)Top) side = (Side)Bottom;
                            else break;
						}

                        w.Flush();
                        nextsectionpos = str.Position;
                        str.Position = nextsectionref;
                        w.Write(nextsectionpos);
                        w.Flush();
                        str.Position = nextsectionpos;
					}

                    /// <summary>
                    /// A class for serializing/deserializing lists of Ids of base tracks to be ignored in plate-to-plate alignment.
                    /// </summary>
                    public class BaseTrackIgnoreAlignment
                    {
                        /// <summary>
                        /// Section tag for this class in multi-section TLG files.
                        /// </summary>
                        public const byte SectionTag = 0x81;
                        /// <summary>
                        /// The list of Ids to be ignored in plate-to-plate alignment.
                        /// </summary>
                        public int[] Ids;
                        /// <summary>
                        /// Builds an empty list.
                        /// </summary>
                        public BaseTrackIgnoreAlignment()
                        {
                            Ids = new int[0];
                        }
                        /// <summary>
                        /// Saves a list of Ids of base tracks to be ignored in plate-to-plate alignment to a TLG file.
                        /// </summary>
                        /// <param name="str">the stream to be used for saving.</param>
                        public void Save(System.IO.Stream str)
                        {
                            if (SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, false)) throw new Exception("BaseTrackIgnoreAligment Section already exists in the stream!");
                            System.IO.BinaryWriter w = new System.IO.BinaryWriter(str);
                            long nextsectionpos = 0;
                            w.Write(SectionTag);
                            long nextsectionref = str.Position;
                            w.Write(nextsectionpos);
                            w.Write(Ids.Length);
                            foreach (int ix in Ids)
                                w.Write(ix);
                            w.Flush();
                            nextsectionpos = str.Position;
                            str.Position = nextsectionref;
                            w.Write(nextsectionpos);
                            w.Flush();
                            str.Position = nextsectionpos;
                        }
                        /// <summary>
                        /// Builds a list of Ids of base tracks to be ignored in plate-to-plate alignment from a TLG file. An exception is thrown if the section is not found in the TLG file.
                        /// </summary>
                        /// <param name="str">the stream to be used for reading.</param>
                        public BaseTrackIgnoreAlignment(System.IO.Stream str)
                        {
                            if (!SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, true)) throw new Exception("No BaseTrackIgnoreAligment section found in stream!");
                            System.IO.BinaryReader r = new System.IO.BinaryReader(str);
                            Ids = new int[r.ReadInt32()];
                            int i;
                            for (i = 0; i < Ids.Length; i++)
                                Ids[i] = r.ReadInt32();                            
                        }
                    }

                    /// <summary>
                    /// A class for serializing/deserializing an index of base tracks.
                    /// </summary>
                    public class BaseTrackIndex
                    {
                        /// <summary>
                        /// Section tag for this class in multi-section TLG files.
                        /// </summary>
                        public const byte SectionTag = 0x83;
                        /// <summary>
                        /// The list of base track Ids.
                        /// </summary>
                        public int[] Ids;
                        /// <summary>
                        /// Builds an empty list.
                        /// </summary>
                        public BaseTrackIndex()
                        {
                            Ids = new int[0];
                        }
                        /// <summary>
                        /// Saves a list of Ids of base tracks to a TLG file.
                        /// </summary>
                        /// <param name="str">the stream to be used for saving.</param>
                        public void Save(System.IO.Stream str)
                        {
                            if (SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, false)) throw new Exception("BaseTrackIndex Section already exists in the stream!");
                            System.IO.BinaryWriter w = new System.IO.BinaryWriter(str);
                            long nextsectionpos = 0;
                            w.Write(SectionTag);
                            long nextsectionref = str.Position;
                            w.Write(nextsectionpos);
                            w.Write(Ids.Length);
                            foreach (int ix in Ids)
                                w.Write(ix);
                            w.Flush();
                            nextsectionpos = str.Position;
                            str.Position = nextsectionref;
                            w.Write(nextsectionpos);
                            w.Flush();
                            str.Position = nextsectionpos;
                        }
                        /// <summary>
                        /// Builds a list of Ids of base tracks a TLG file. An exception is thrown if the section is not found in the TLG file.
                        /// </summary>
                        /// <param name="str">the stream to be used for reading.</param>
                        public BaseTrackIndex(System.IO.Stream str)
                        {
                            if (!SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, true)) throw new Exception("No BaseTrackIndex section found in stream!");
                            System.IO.BinaryReader r = new System.IO.BinaryReader(str);
                            Ids = new int[r.ReadInt32()];
                            int i;
                            for (i = 0; i < Ids.Length; i++)
                                Ids[i] = r.ReadInt32();
                        }
                    }
				}

                namespace RawData
				{
					class File
					{
						public enum Info : byte {	Catalog = 1, View = 2, Track = 3, Grain = 4, Config = 5, Fragment = 6 }
	
						public enum Section : byte { Data = 0x30, Header = 0x60 }
	
						public enum Format : ushort { Old = 0x0701, Normal = 0x0702, NormalDouble = 0x0703 }

						public enum Compression : uint { Null = 0, GrainSuppression = 0x0102 }
					}

					/// <summary>
					/// A scanning fragment : the basic unit of scanning tasks.
					/// Views are accessed in an array-like fashion.
					/// </summary>
					public class Fragment
					{
						/// <summary>
						/// If true, grains are not read from fragment files.
						/// </summary>
						static public bool SkipReadingGrains = false;

						private class MIPIndexedEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
						{
							public static MIPEmulsionTrackInfo GetInfo(MIPEmulsionTrack t) { return MIPIndexedEmulsionTrack.AccessInfo(t); }

							public static Grain [] GetGrains(MIPEmulsionTrack t) { return MIPIndexedEmulsionTrack.AccessGrains(t); }
						}

						/// <summary>
						/// Constants that define the coding mode for the fragment
						/// </summary>
						public enum FragmentCoding 
						{ 
							/// <summary>
							/// all grains are stored in plain format.
							/// </summary>
							Normal = 0x0, 
							/// <summary>
							/// grains are suppressed from the file to save space.
							/// </summary>
							GrainSuppression = 0x102 
						}
						/// <summary>
						/// Member data on which the Id property relies. Can be accessed by derived classes.
						/// </summary>
						protected Identifier m_Id;
						/// <summary>
						/// Scanning zone identifier. It is identical for all fragments in a scanning zone.
						/// </summary>
						public Identifier Id { get { return m_Id; } }
						/// <summary>
						/// Member data on which the Index property relies. Can be accessed by derived classes.
						/// </summary>
						protected uint m_Index;
						/// <summary>
						/// The fragment index. Ranges from 1 to the total number of fragments in the scanning zone. 0 is reserved to indicate null fragment.
						/// </summary>
						public uint Index { get { return m_Index; } }
						/// <summary>
						/// Member data on which the StartView property relies. Can be accessed by derived classes.
						/// </summary>
						protected uint m_StartView;
						/// <summary>
						/// The number of the first view in this fragment in the sequential order in which they have been scanned in the scanning zone.
						/// Is 0 for the first view of the first fragment, N at the second fragment, 2N at the third and so on, where N is the total number of views in a fragment.
						/// </summary>
						public uint StartView { get { return m_StartView; } }
						/// <summary>
						/// Member data on which the StartView property relies. Can be accessed by derived classes.
						/// </summary>
						protected FragmentCoding m_CodingMode;
						/// <summary>
						/// The coding mode of this fragment.
						/// </summary>
						public FragmentCoding CodingMode { get { return m_CodingMode; } }
						/// <summary>
						/// Protected member holding the array of Views. Can be accessed by derived classes.
						/// </summary>
						protected internal View [] m_Views;
						/// <summary>
						/// Provides access to the fragment views in an array-like fashion.
						/// </summary>
						public View this[int index] { get { return m_Views[index]; } }
						/// <summary>
						/// Returns the number of views.
						/// </summary>
						public int Length { get { return m_Views.Length; } }
						/// <summary>
						/// Provides quick access to the array of Views in the specified fragment. Can be accessed by derived classes.
						/// </summary>
						/// <param name="f">the fragment whose views are to be accessed.</param>
						/// <returns>the m_Views array in the specified fragment.</returns>
						protected static View [] AccessViews(Fragment f) { return f.m_Views; }
						/// <summary>
						/// Protected constructor. Prevents users from creating instances of Fragment without deriving the class. Is implicitly called by constructors in derived classes.
						/// </summary>
						protected Fragment() {}

						/// <summary>
						/// A field of view in a scanning fragment.
						/// </summary>
						public class View
						{
							/// <summary>
							/// Protected constructor. Prevents users from building instances of View without deriving the class. Is implicitly called by constructors in derived classes.
							/// </summary>
							protected internal View() {}

							/// <summary>
							/// Tiling position in a scanning zone tile grid arrangement.
							/// </summary>
							public struct TilePos
							{
								/// <summary>
								/// Grid X position.
								/// </summary>
								public int X;
								/// <summary>
								/// Grid Y position.
								/// </summary>
								public int Y;
							}
							/// <summary>
							/// Member data on which the Tile property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal TilePos m_Tile;
							/// <summary>
							/// The tile position of this view in the scanning zone tile grid.
							/// </summary>
							public TilePos Tile { get { return m_Tile; } }
							/// <summary>
							/// Member data on which the Top property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal Side m_Top;
							/// <summary>
							/// The top side of the field of view.
							/// </summary>
							public Side Top { get { return m_Top; } }
							/// <summary>
							/// Member data on which the Bottom property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal Side m_Bottom;
							/// <summary>
							/// The bottom side of the field of view.
							/// </summary>
							public Side Bottom { get { return m_Bottom; } }

							/// <summary>
							/// Contains information about each side of each view in a fragment.
							/// </summary>
							public class Side : SySal.Scanning.Plate.Side
							{
								internal double iTopZ { set { m_TopZ = value; } }
								internal double iBottomZ { set { m_BottomZ = value; } }

								/// <summary>
								/// Protected constructor. Prevents users from building instances of Side without deriving the class. Is implicitly called by constructors in derived classes.
								/// </summary>
								protected internal Side() : base() {}

								/// <summary>
								/// Flags that describe the scanning history on each side of each view.
								/// These flags can be combined together in a byte representation.
								/// </summary>
								public enum SideFlags : byte 
								{ 
									/// <summary>
									/// Successful scanning.
									/// </summary>
									OK = 0x00, 
									/// <summary>
									/// Focus not found on top side.
									/// </summary>
									NoTopFocus = 0x02, 
									/// <summary>
									/// Focus not found on bottom side.
									/// </summary>
									NoBottomFocus = 0x02, 
									/// <summary>
									/// The Z limiter triggered.
									/// </summary>
									ZLimiter = 0x04, 
									/// <summary>
									/// The X limiter triggered.
									/// </summary>
									XLimiter = 0x08, 
									/// <summary>
									/// The Y limiter triggered.
									/// </summary>
									YLimiter = 0x10, 
									/// <summary>
									/// The scanning was interrupted.
									/// </summary>
									Terminated = 0x80, 
									/// <summary>
									/// The side was not scanned (initial state for all sides of all views).
									/// </summary>
									NotScanned = 0xFF 
								} 

								/// <summary>
								/// General info on each layer of a tomographic sequence.
								/// </summary>
								public struct LayerInfo
								{
									/// <summary>
									/// Total number of grains in the layer.
									/// </summary>
									public uint Grains;
									/// <summary>
									/// Z coordinate of the layer.
									/// </summary>
									public double Z;
								}

								/// <summary>
								/// Member data on which the Pos property relies. Can be accessed by derived classes.
								/// </summary>
								protected internal Vector2 m_Pos;
								/// <summary>
								/// Stage position of the center of the field of view on this side.
								/// </summary>
								public Vector2 Pos { get { return m_Pos; } }
								/// <summary>
								/// Protected method that allows to set the Pos property from derived classes.
								/// </summary>
								/// <param name="s">the Side object to be changed.</param>
								/// <param name="pos">the new value of the Pos property.</param>
								protected static void SetPos(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, Vector2 pos) { s.m_Pos = pos; }
								/// <summary>
								/// Member data on which the MapPos property relies. Can be accessed by derived classes.
								/// </summary>
								protected internal Vector2 m_MapPos;
								/// <summary>
								/// Position of the center of the field of view in the experiment reference frame.
								/// </summary>
								public Vector2 MapPos { get { return m_MapPos; } }
								/// <summary>
								/// Protected method that allows to set the MapPos property from derived classes.
								/// </summary>
								/// <param name="s">the Side object to be changed.</param>
								/// <param name="mappos">the new value of the MapPos property.</param>
								protected static void SetMapPos(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, Vector2 mappos) { s.m_MapPos = mappos; }
								/// <summary>
								/// Member data on which the MXX, MXY, MYX, MYY property rely. Can be accessed by derived classes.
								/// </summary>
								protected internal double m_MXX, m_MXY, m_MYX, m_MYY;
								/// <summary>
								/// Member data on which the IMXX, IMXY, IMYX, IMYY property rely. Can be accessed by derived classes.
								/// </summary>
								protected internal double m_IMXX, m_IMXY, m_IMYX, m_IMYY;
								/// <summary>
								/// XX component of the matrix that transforms points and vector from the view reference frame to the experiment reference frame.
								/// </summary>
								public double MXX { get { return m_MXX; } }
								/// <summary>
								/// XY component of the matrix that transforms points and vector from the view reference frame to the experiment reference frame.
								/// </summary>								
								public double MXY { get { return m_MXY; } }
								/// <summary>
								/// YX component of the matrix that transforms points and vector from the view reference frame to the experiment reference frame.
								/// </summary>
								public double MYX { get { return m_MYX; } }
								/// <summary>
								/// YY component of the matrix that transforms points and vector from the view reference frame to the experiment reference frame.
								/// </summary>
								public double MYY { get { return m_MYY; } }
								/// <summary>
								/// XX component of the matrix that transforms points and vector from the experiment reference frame to the view reference frame.
								/// </summary>
								public double IMXX { get { return m_IMXX; } }
								/// <summary>
								/// XY component of the matrix that transforms points and vector from the experiment reference frame to the view reference frame.
								/// </summary>								
								public double IMXY { get { return m_IMXY; } }
								/// <summary>
								/// YX component of the matrix that transforms points and vector from the experiment reference frame to the view reference frame.
								/// </summary>
								public double IMYX { get { return m_IMYX; } }
								/// <summary>
								/// YY component of the matrix that transforms points and vector from the experiment reference frame to the view reference frame.
								/// </summary>
								public double IMYY { get { return m_IMYY; } }
								/// <summary>
								/// Protected method that allows to set the M... matrix properties from derived classes.
								/// </summary>
								/// <param name="s">the Side object to be changed.</param>
								/// <param name="mxx">the new value of the MXX matrix property.</param>
								/// <param name="mxy">the new value of the MXY matrix property.</param>
								/// <param name="myx">the new value of the MYX matrix property.</param>
								/// <param name="myy">the new value of the MYY matrix property.</param>
								protected static void SetM(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, double mxx, double mxy, double myx, double myy) 
								{ 
									s.m_MXX = mxx;
									s.m_MXY = mxy;
									s.m_MYX = myx;
									s.m_MYY = myy;
									double det = 1 / (s.m_MXX * s.m_MYY - s.m_MXY * s.m_MYX);
									s.m_IMXX = s.m_MYY * det;
									s.m_IMXY = -s.m_MXY * det;
									s.m_IMYX = -s.m_MYX * det;
									s.m_IMYY = s.m_MXX * det;

								}
								/// <summary>
								/// Transforms a 2d point from the view reference frame to the experiment reference frame.
								/// </summary>
								/// <param name="p">the 2d point to be transformed.</param>
								/// <returns>the transformed point.</returns>
								public Vector2 MapPoint(Vector2 p)
								{
									Vector2 v;
									v.X = m_MapPos.X + m_MXX * p.X + m_MXY * p.Y;
									v.Y = m_MapPos.Y + m_MYX * p.X + m_MYY * p.Y;
									return v;
								}
								/// <summary>
								/// Transforms a 3d point from the view reference frame to the experiment reference frame.
								/// </summary>
								/// <param name="p">the 3d point to be transformed.</param>
								/// <returns>the transformed point.</returns>
								public Vector MapPoint(Vector p)
								{
									Vector v;
									v.X = m_MapPos.X + m_MXX * p.X + m_MXY * p.Y;
									v.Y = m_MapPos.Y + m_MYX * p.X + m_MYY * p.Y;
									v.Z = p.Z;
									return v;
								}
								/// <summary>
								/// Transforms a 2d vector from the view reference frame to the experiment reference frame.
								/// </summary>
								/// <param name="p">the 2d vector to be transformed.</param>
								/// <returns>the transformed vector.</returns>
								public Vector2 MapVector(Vector2 p)
								{
									Vector2 v;
									v.X = m_MXX * p.X + m_MXY * p.Y;
									v.Y = m_MYX * p.X + m_MYY * p.Y;
									return v;
								}
								/// <summary>
								/// Transforms a 3d vector from the view reference frame to the experiment reference frame.
								/// </summary>
								/// <param name="p">the 3d vector to be transformed.</param>
								/// <returns>the transformed vector.</returns>
								public Vector MapVector(Vector p)
								{
									Vector v;
									v.X = m_MXX * p.X + m_MXY * p.Y;
									v.Y = m_MYX * p.X + m_MYY * p.Y;
									v.Z = p.Z;
									return v;
								}
								/// <summary>
								/// Transforms a 2d point from the experiment reference frame to the view reference frame.
								/// </summary>
								/// <param name="p">the 2d point to be transformed.</param>
								/// <returns>the transformed point.</returns>
								public Vector2 IMapPoint(Vector2 p)
								{
									Vector2 v;
									p.X -= m_MapPos.X;
									p.Y -= m_MapPos.Y;
									v.X = m_IMXX * p.X + m_IMXY * p.Y;
									v.Y = m_IMYX * p.X + m_IMYY * p.Y;
									return v;
								}
								/// <summary>
								/// Transforms a 3d point from the experiment reference frame to the view reference frame.
								/// </summary>
								/// <param name="p">the 3d point to be transformed.</param>
								/// <returns>the transformed point.</returns>
								public Vector IMapPoint(Vector p)
								{
									Vector v;
									p.X -= m_MapPos.X;
									p.Y -= m_MapPos.Y;
									v.X = m_IMXX * p.X + m_IMXY * p.Y;
									v.Y = m_IMYX * p.X + m_IMYY * p.Y;
									v.Z = p.Z;
									return v;
								}
								/// <summary>
								/// Transforms a 2d vector from the experiment reference frame to the view reference frame.
								/// </summary>
								/// <param name="p">the 2d vector to be transformed.</param>
								/// <returns>the transformed vector.</returns>
								public Vector2 IMapVector(Vector2 p)
								{
									Vector2 v;
									v.X = m_IMXX * p.X + m_IMXY * p.Y;
									v.Y = m_IMYX * p.X + m_IMYY * p.Y;
									return v;
								}
								/// <summary>
								/// Transforms a 3d vector from the experiment reference frame to the view reference frame.
								/// </summary>
								/// <param name="p">the 3d vector to be transformed.</param>
								/// <returns>the transformed vector.</returns>
								public Vector IMapVector(Vector p)
								{
									Vector v;
									v.X = m_IMXX * p.X + m_IMXY * p.Y;
									v.Y = m_IMYX * p.X + m_IMYY * p.Y;
									v.Z = p.Z;
									return v;
								}
		
								/// <summary>
								/// Member data on which the Flags property relies. Can be accessed by derived classes.
								/// </summary>
								protected internal SideFlags m_Flags;
								/// <summary>
								/// Scanning result flags.
								/// </summary>
								public SideFlags Flags { get { return m_Flags; } }

								/// <summary>
								/// A list of LayerInfo.
								/// </summary>
								public class LayerInfoList
								{
									/// <summary>
									/// Protected member data holding the array of LayerInfos. Can be accessed by derived classes.
									/// </summary>
									protected internal LayerInfo [] m_Layers;
									/// <summary>
									/// Provides access to LayerInfo in an array-like fashion.
									/// </summary>
									public LayerInfo this[int index] { get { return m_Layers[index]; } }
									/// <summary>
									/// Returns the number of LayerInfos in the list.
									/// </summary>
									public int Length { get { return m_Layers.Length; } }
									/// <summary>
									/// Initializes the LayerInfoList with an already allocated array.
									/// </summary>
									/// <param name="l">the array of LayerInfos to initialize the LayerInfoList.</param>
									public LayerInfoList(LayerInfo [] l) { m_Layers = l; }
								}
								/// <summary>
								/// Member data on which the Layers property relies. Can be accessed by derived classes.
								/// </summary>
								protected internal LayerInfoList m_Layers;
								/// <summary>
								/// Available layer information for this side.
								/// </summary>
								public LayerInfoList Layers { get { return m_Layers; } }
							}
						}

						/// <summary>
						/// Restores a saved fragment from a stream.
						/// </summary>
						/// <param name="s">the stream from which to read the fragment information.</param>
						public Fragment(System.IO.Stream s)
						{
							System.IO.BinaryReader r = new System.IO.BinaryReader(s);
							byte infotype = r.ReadByte();
							ushort headerformat = r.ReadUInt16();

							uint FitCorrectionDataSize;

							if (infotype != ((byte)File.Info.Fragment | (byte)File.Section.Header) ||
								(headerformat != (ushort)File.Format.NormalDouble && headerformat != (ushort)File.Format.Normal && 
								headerformat != (ushort)File.Format.Old)) throw new SystemException("Unknown data format");
							m_Id.Part0 = r.ReadInt32();
							m_Id.Part1 = r.ReadInt32();
							m_Id.Part2 = r.ReadInt32();
							m_Id.Part3 = r.ReadInt32();
							m_Index = r.ReadUInt32();
							m_StartView = r.ReadUInt32();
							m_Views = new View[r.ReadUInt32()];
							FitCorrectionDataSize = r.ReadUInt32();
							m_CodingMode = (FragmentCoding)r.ReadInt32();
							switch (m_CodingMode)
							{
								case FragmentCoding.Normal:				break;
								case FragmentCoding.GrainSuppression:	break;
								default:								throw new System.Exception("Unsupported fragment coding mode");
							}
							r.ReadBytes(256);

							int i, j, k;
							if (headerformat == (ushort)File.Format.NormalDouble)
							{
								double det;
								for (i = 0; i < m_Views.Length; i++)
								{
									View v = m_Views[i] = new View();
									v.m_Top = new RawData.Fragment.View.Side();
									v.m_Bottom = new RawData.Fragment.View.Side();
									v.m_Tile.X = r.ReadInt32();
									v.m_Tile.Y = r.ReadInt32();
									v.m_Top.m_Pos.X = r.ReadDouble();
									v.m_Bottom.m_Pos.X = r.ReadDouble();
									v.m_Top.m_Pos.Y = r.ReadDouble();
									v.m_Bottom.m_Pos.Y = r.ReadDouble();
									v.m_Top.m_MapPos.X = r.ReadDouble();
									v.m_Bottom.m_MapPos.X = r.ReadDouble();
									v.m_Top.m_MapPos.Y = r.ReadDouble();
									v.m_Bottom.m_MapPos.Y = r.ReadDouble();
									v.m_Top.m_MXX = r.ReadDouble();
									v.m_Top.m_MXY = r.ReadDouble();
									v.m_Top.m_MYX = r.ReadDouble();
									v.m_Top.m_MYY = r.ReadDouble();
									det = 1 / (v.m_Top.m_MXX * v.m_Top.m_MYY - v.m_Top.m_MXY * v.m_Top.m_MYX);
									v.m_Top.m_IMXX = v.m_Top.m_MYY * det;
									v.m_Top.m_IMXY = -v.m_Top.m_MXY * det;
									v.m_Top.m_IMYX = -v.m_Top.m_MYX * det;
									v.m_Top.m_IMYY = v.m_Top.m_MXX * det;
									v.m_Bottom.m_MXX = r.ReadDouble();
									v.m_Bottom.m_MXY = r.ReadDouble();
									v.m_Bottom.m_MYX = r.ReadDouble();
									v.m_Bottom.m_MYY = r.ReadDouble();
									det = 1 / (v.m_Bottom.m_MXX * v.m_Bottom.m_MYY - v.m_Bottom.m_MXY * v.m_Bottom.m_MYX);
									v.m_Bottom.m_IMXX = v.m_Bottom.m_MYY * det;
									v.m_Bottom.m_IMXY = -v.m_Bottom.m_MXY * det;
									v.m_Bottom.m_IMYX = -v.m_Bottom.m_MYX * det;
									v.m_Bottom.m_IMYY = v.m_Bottom.m_MXX * det;
									v.m_Top.m_Layers = new RawData.Fragment.View.Side.LayerInfoList(new RawData.Fragment.View.Side.LayerInfo[r.ReadInt32()]);
									v.m_Bottom.m_Layers = new RawData.Fragment.View.Side.LayerInfoList(new RawData.Fragment.View.Side.LayerInfo[r.ReadInt32()]);									
									v.m_Top.iTopZ = r.ReadDouble();
									v.m_Top.iBottomZ = r.ReadDouble();
									v.m_Bottom.iTopZ = r.ReadDouble();
									v.m_Bottom.iBottomZ = r.ReadDouble();
									v.m_Top.m_Flags = (RawData.Fragment.View.Side.SideFlags)r.ReadByte();
									v.m_Bottom.m_Flags = (RawData.Fragment.View.Side.SideFlags)r.ReadByte();
									v.m_Top.m_Tracks = new SySal.Scanning.MIPIndexedEmulsionTrack[r.ReadInt32()];
									v.m_Bottom.m_Tracks = new SySal.Scanning.MIPIndexedEmulsionTrack[r.ReadInt32()];
								}
								for (i = 0; i < m_Views.Length; i++)
								{
									View v = m_Views[i];
									for (j = 0; j < v.m_Top.m_Layers.m_Layers.Length; j++)
									{
										v.m_Top.m_Layers.m_Layers[j].Grains = (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalDouble) ? r.ReadUInt32() : 0;
										v.m_Top.m_Layers.m_Layers[j].Z = r.ReadDouble();
									}
									for (j = 0; j < v.m_Bottom.m_Layers.m_Layers.Length; j++)
									{
										v.m_Bottom.m_Layers.m_Layers[j].Grains = (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalDouble) ? r.ReadUInt32() : 0;
										v.m_Bottom.m_Layers.m_Layers[j].Z = r.ReadDouble();
									}
								}
								for (i = 0; i < m_Views.Length; i++)
								{
									View v = m_Views[i];
									Side h;
									for (h = v.m_Top; h != null; h = (h == v.m_Top) ? v.m_Bottom : null)
										for (j = 0; j < h.m_Tracks.Length; j++)
										{
											MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
											info.AreaSum = r.ReadUInt32();
											info.Count = (ushort)r.ReadUInt32();
											info.Intercept.X = r.ReadDouble();
											info.Intercept.Y = r.ReadDouble();
											info.Intercept.Z = r.ReadDouble();
											info.Slope.X = r.ReadDouble();
											info.Slope.Y = r.ReadDouble();
											info.Slope.Z = r.ReadDouble();
											info.Sigma = r.ReadDouble();
											info.TopZ = r.ReadDouble();
											info.BottomZ = r.ReadDouble();
											h.m_Tracks[j] = new SySal.Scanning.MIPIndexedEmulsionTrack(info, (CodingMode != FragmentCoding.GrainSuppression && SkipReadingGrains == false) ? new Grain[info.Count] : null, j);
										}
								}
								if (CodingMode == FragmentCoding.Normal)
								{
									if (SkipReadingGrains)
									{
										m_CodingMode = FragmentCoding.GrainSuppression;
									}
									else
									{
										for (i = 0; i < m_Views.Length; i++)
										{
											View v = m_Views[i];
											Side h;
											for (h = v.m_Top; h != null; h = (h == v.m_Top) ? v.m_Bottom : null)
												for (j = 0; j < h.m_Tracks.Length; j++)
												{
													Grain [] g = h.m_Tracks[j].iGrains;
													for (k = 0; k < g.Length; k++)
													{
														g[k] = new Grain();
														g[k].Area = (headerformat == (ushort)File.Format.NormalDouble) ? r.ReadUInt32() : 0;
														g[k].Position.X = r.ReadDouble();
														g[k].Position.Y = r.ReadDouble();
														g[k].Position.Z = r.ReadDouble();
													}
												}
										}
									}
								}
							}
							else
							{
								double det;
								for (i = 0; i < m_Views.Length; i++)
								{
									View v = m_Views[i] = new View();
									v.m_Top = new RawData.Fragment.View.Side();
									v.m_Bottom = new RawData.Fragment.View.Side();
									v.m_Tile.X = r.ReadInt32();
									v.m_Tile.Y = r.ReadInt32();
									v.m_Top.m_Pos.X = r.ReadSingle();
									v.m_Bottom.m_Pos.X = r.ReadSingle();
									v.m_Top.m_Pos.Y = r.ReadSingle();
									v.m_Bottom.m_Pos.Y = r.ReadSingle();
									v.m_Top.m_MapPos.X = r.ReadSingle();
									v.m_Bottom.m_MapPos.X = r.ReadSingle();
									v.m_Top.m_MapPos.Y = r.ReadSingle();
									v.m_Bottom.m_MapPos.Y = r.ReadSingle();
									v.m_Top.m_MXX = r.ReadSingle();
									v.m_Top.m_MXY = r.ReadSingle();
									v.m_Top.m_MYX = r.ReadSingle();
									v.m_Top.m_MYY = r.ReadSingle();
									det = 1 / (v.m_Top.m_MXX * v.m_Top.m_MYY - v.m_Top.m_MXY * v.m_Top.m_MYX);
									v.m_Top.m_IMXX = v.m_Top.m_MYY * det;
									v.m_Top.m_IMXY = -v.m_Top.m_MXY * det;
									v.m_Top.m_IMYX = -v.m_Top.m_MYX * det;
									v.m_Top.m_IMYY = v.m_Top.m_MXX * det;
									v.m_Bottom.m_MXX = r.ReadSingle();
									v.m_Bottom.m_MXY = r.ReadSingle();
									v.m_Bottom.m_MYX = r.ReadSingle();
									v.m_Bottom.m_MYY = r.ReadSingle();
									det = 1 / (v.m_Bottom.m_MXX * v.m_Bottom.m_MYY - v.m_Bottom.m_MXY * v.m_Bottom.m_MYX);
									v.m_Bottom.m_IMXX = v.m_Bottom.m_MYY * det;
									v.m_Bottom.m_IMXY = -v.m_Bottom.m_MXY * det;
									v.m_Bottom.m_IMYX = -v.m_Bottom.m_MYX * det;
									v.m_Bottom.m_IMYY = v.m_Bottom.m_MXX * det;
									v.m_Top.m_Layers = new RawData.Fragment.View.Side.LayerInfoList(new RawData.Fragment.View.Side.LayerInfo[r.ReadInt32()]);
									v.m_Bottom.m_Layers = new RawData.Fragment.View.Side.LayerInfoList(new RawData.Fragment.View.Side.LayerInfo[r.ReadInt32()]);
									v.m_Top.iTopZ = r.ReadSingle();
									v.m_Top.iBottomZ = r.ReadSingle();
									v.m_Bottom.iTopZ = r.ReadSingle();
									v.m_Bottom.iBottomZ = r.ReadSingle();
									v.m_Top.m_Flags = (RawData.Fragment.View.Side.SideFlags)r.ReadByte();
									v.m_Bottom.m_Flags = (RawData.Fragment.View.Side.SideFlags)r.ReadByte();
									v.m_Top.m_Tracks = new SySal.Scanning.MIPIndexedEmulsionTrack[r.ReadInt32()];
									v.m_Bottom.m_Tracks = new SySal.Scanning.MIPIndexedEmulsionTrack[r.ReadInt32()];
								}
								for (i = 0; i < m_Views.Length; i++)
								{
									View v = m_Views[i];
									for (j = 0; j < v.m_Top.m_Layers.m_Layers.Length; j++)
									{
										v.m_Top.m_Layers.m_Layers[j].Grains = (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalDouble) ? r.ReadUInt32() : 0;
										v.m_Top.m_Layers.m_Layers[j].Z = r.ReadSingle();
									}
									for (j = 0; j < v.m_Bottom.m_Layers.m_Layers.Length; j++)
									{
										v.m_Bottom.m_Layers.m_Layers[j].Grains = (headerformat == (ushort)File.Format.Normal || headerformat == (ushort)File.Format.NormalDouble) ? r.ReadUInt32() : 0;
										v.m_Bottom.m_Layers.m_Layers[j].Z = r.ReadSingle();
									}
								}
								for (i = 0; i < m_Views.Length; i++)
								{
									View v = m_Views[i];
									Side h;
									for (h = v.m_Top; h != null; h = (h == v.m_Top) ? v.m_Bottom : null)
										for (j = 0; j < h.m_Tracks.Length; j++)
										{
											MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();											
											info.AreaSum = (headerformat == (ushort)File.Format.Normal) ? r.ReadUInt32() : 0;
											info.Count = (ushort)r.ReadUInt32();
											info.Intercept.X = r.ReadSingle();
											info.Intercept.Y = r.ReadSingle();
											info.Intercept.Z = r.ReadSingle();
											info.Slope.X = r.ReadSingle();
											info.Slope.Y = r.ReadSingle();
											info.Slope.Z = r.ReadSingle();
											info.Sigma = r.ReadSingle();
											info.TopZ = r.ReadSingle();
											info.BottomZ = r.ReadSingle();
											if (headerformat == (ushort)File.Format.Old)
											{
												r.ReadSingle(); r.ReadSingle(); r.ReadSingle(); 
												r.ReadSingle(); r.ReadSingle(); r.ReadSingle();
												r.ReadBytes((int)FitCorrectionDataSize);
											};
											h.m_Tracks[j] = new SySal.Scanning.MIPIndexedEmulsionTrack(info, (CodingMode != FragmentCoding.GrainSuppression && SkipReadingGrains == false) ? new Grain[info.Count] : null, j);
										}
								}
								if (CodingMode == FragmentCoding.Normal)
								{
									if (SkipReadingGrains)
									{
										m_CodingMode = FragmentCoding.GrainSuppression;
									}
									else
									{
										for (i = 0; i < m_Views.Length; i++)
										{
											View v = m_Views[i];
											Side h;
											for (h = v.m_Top; h != null; h = (h == v.m_Top) ? v.m_Bottom : null)
												for (j = 0; j < h.m_Tracks.Length; j++)
												{
													Grain [] g = h.m_Tracks[j].iGrains;
													for (k = 0; k < g.Length; k++)
													{
														g[k] = new Grain();
														g[k].Area = (headerformat == (ushort)File.Format.Normal) ? r.ReadUInt32() : 0;
														g[k].Position.X = r.ReadSingle();
														g[k].Position.Y = r.ReadSingle();
														g[k].Position.Z = r.ReadSingle();
													}
												}
										}
									}
								}
							}
						}

						/// <summary>
						/// Saves the fragment to a stream.
						/// </summary>
						/// <param name="s">the stream where the fragment is to be saved.</param>
						public virtual void Save(System.IO.Stream s)
						{
							System.IO.BinaryWriter w = new System.IO.BinaryWriter(s);
							w.Write((byte)((byte)File.Info.Fragment | (byte)File.Section.Header));
							w.Write((ushort)File.Format.NormalDouble);

							w.Write(m_Id.Part0);
							w.Write(m_Id.Part1);
							w.Write(m_Id.Part2);
							w.Write(m_Id.Part3);

							w.Write(m_Index);
							w.Write(m_StartView);
							w.Write(m_Views.Length);
							w.Write(0); // no fit correction data
							w.Write((int)m_CodingMode);
							switch (m_CodingMode)
							{
								case FragmentCoding.Normal:				break;
								case FragmentCoding.GrainSuppression:	break;
								default:								throw new System.Exception("Unsupported fragment coding mode");
							}
							w.Write(new byte[256]);

							int j;
							foreach (View v in m_Views)
							{
								w.Write(v.m_Tile.X);
								w.Write(v.m_Tile.Y);
								w.Write(v.m_Top.m_Pos.X);
								w.Write(v.m_Bottom.m_Pos.X);
								w.Write(v.m_Top.m_Pos.Y);
								w.Write(v.m_Bottom.m_Pos.Y);
								w.Write(v.m_Top.m_MapPos.X);
								w.Write(v.m_Bottom.m_MapPos.X);
								w.Write(v.m_Top.m_MapPos.Y);
								w.Write(v.m_Bottom.m_MapPos.Y);
								w.Write(v.m_Top.m_MXX);
								w.Write(v.m_Top.m_MXY);
								w.Write(v.m_Top.m_MYX);
								w.Write(v.m_Top.m_MYY);
								w.Write(v.m_Bottom.m_MXX);
								w.Write(v.m_Bottom.m_MXY);
								w.Write(v.m_Bottom.m_MYX);
								w.Write(v.m_Bottom.m_MYY);

								w.Write(v.m_Top.m_Layers.m_Layers.Length);
								w.Write(v.m_Bottom.m_Layers.m_Layers.Length);
								w.Write(v.m_Top.m_TopZ);
								w.Write(v.m_Top.m_BottomZ);
								w.Write(v.m_Bottom.m_TopZ);
								w.Write(v.m_Bottom.m_BottomZ);
								w.Write((byte)v.m_Top.m_Flags);
								w.Write((byte)v.m_Bottom.m_Flags);
								w.Write(v.m_Top.m_Tracks.Length);
								w.Write(v.m_Bottom.m_Tracks.Length);								
							}
							foreach (View v in m_Views)
							{
								View.Side.LayerInfo [] l;
								l = v.Top.m_Layers.m_Layers;
								for (j = 0; j < l.Length; j++)
								{
									w.Write(l[j].Grains);
									w.Write(l[j].Z);
								}
								l = v.Bottom.m_Layers.m_Layers;
								for (j = 0; j < l.Length; j++)
								{
									w.Write(l[j].Grains);
									w.Write(l[j].Z);
								}
							}
							foreach (View v in m_Views)
							{
								Side h;
								for (h = v.m_Top; h != null; h = (h == v.m_Top) ? v.m_Bottom : null)
									foreach (SySal.Tracking.MIPEmulsionTrack t in h.m_Tracks)
									{
										MIPEmulsionTrackInfo info = MIPIndexedEmulsionTrack.GetInfo(t);
										w.Write(info.AreaSum);
										w.Write((uint)info.Count);
										w.Write(info.Intercept.X);
										w.Write(info.Intercept.Y);
										w.Write(info.Intercept.Z);
										w.Write(info.Slope.X);
										w.Write(info.Slope.Y);
										w.Write(info.Slope.Z);
										w.Write(info.Sigma);
										w.Write(info.TopZ);
										w.Write(info.BottomZ);
									}
							}
							if (CodingMode == FragmentCoding.Normal)
								foreach (View v in m_Views)
								{
									Side h;
									for (h = v.m_Top; h != null; h = (h == v.m_Top) ? v.m_Bottom : null)
										foreach (SySal.Tracking.MIPEmulsionTrack t in h.m_Tracks)
										{
											Grain [] gs = MIPIndexedEmulsionTrack.GetGrains(t);
											foreach (Grain g in gs)
											{
												w.Write(g.Area);
												w.Write(g.Position.X);
												w.Write(g.Position.Y);
												w.Write(g.Position.Z);
											}
										}
								}
						}
					}


					/// <summary>
					/// Contains overall information shared by all fragments in a scanning zone.
					/// </summary>
					public class Catalog
					{
						/// <summary>
						/// String representation of parameter name - parameter value pairs.
						/// </summary>
						public class KeyStringRepresentation
						{
							/// <summary>
							/// Member data on which the Name property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal string m_Name;
							/// <summary>
							/// Name of a parameter.
							/// </summary>
							public string Name { get { return m_Name; } }
							/// <summary>
							/// Member data on which the Value property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal string m_Value;
							/// <summary>
							/// Value of a parameter.
							/// </summary>
							public string Value { get { return m_Value; } }
							/// <summary>
							/// Protected constructor. Prevents users from creating instances of KeyStringRepresentation without deriving the class. Is implicitly called by constructors in derived classes.
							/// </summary>
							protected internal KeyStringRepresentation() {}
						}

						/// <summary>
						/// String representation of a whole configuration for an object.
						/// </summary>
						public class ConfigStringRepresentation
						{
							/// <summary>
							/// Member data on which the ClassName property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal string m_ClassName;
							/// <summary>
							/// Name of the class of the object.
							/// </summary>
							public string ClassName { get { return m_ClassName; } }
							/// <summary>
							/// Member data on which the Name property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal string m_Name;
							/// <summary>
							/// Name of a parameter.
							/// </summary>
							public string Name { get { return m_Name; } }
							/// <summary>
							/// Protected member data holding the array of KeyStringRepresentations. Can be accessed by derived classes.
							/// </summary>
							protected internal KeyStringRepresentation [] m_Keys;
							/// <summary>
							/// Provides access to the KeyStringRepresentations in an array-like fashion.
							/// </summary>
							public KeyStringRepresentation this[int index] { get { return m_Keys[index]; } }
							/// <summary>
							/// Returns the number of KeyStringRepresentations in the configuration.
							/// </summary>
							public int Length { get { return m_Keys.Length; } }
							/// <summary>
							/// Protected constructor. Prevents users from creating instances of ConfigStringRepresentation without deriving the class. Is implicitly called by constructors in derived classes.
							/// </summary>
							protected internal ConfigStringRepresentation() {}
						}

						/// <summary>
						/// String representation of a whole scanning setup.
						/// </summary>
						public class SetupStringRepresentation
						{
							/// <summary>
							/// Member data on which the Name property relies. Can be accessed by derived classes.
							/// </summary>
							protected internal string m_Name;
							/// <summary>
							/// Name of a parameter.
							/// </summary>
							public string Name { get { return m_Name; } }
							/// <summary>
							/// Protected member data holding the array of ConfigStringRepresentations. Can be accessed by derived classes.
							/// </summary>
							protected internal ConfigStringRepresentation [] m_Configs;
							/// <summary>
							/// Provides access to the ConfigStringRepresentations in an array-like fashion.
							/// </summary>							
							public ConfigStringRepresentation this[int index] { get { return m_Configs[index]; } }
							/// <summary>
							/// Returns the number of ConfigStringRepresentations.
							/// </summary>
							public int Length { get { return m_Configs.Length; } }
							/// <summary>
							/// Protected constructor. Prevents users from creating instances of SetupStringRepresentation without deriving the class. Is implicitly called by constructors in derived classes.
							/// </summary>
							protected internal SetupStringRepresentation() {}
						}

						/// <summary>
						/// Member data on which the Id property relies. Can be accessed by derived classes.
						/// </summary>
						protected Identifier m_Id;
						/// <summary>
						/// Catalog identifier. The same identifier value is shared by all fragments in the same scanning zone.
						/// </summary>
						public Identifier Id { get { return m_Id; } }
						/// <summary>
						/// Member data on which the Extents property relies. Can be accessed by derived classes.
						/// </summary>
						protected Rectangle m_Extents;
						/// <summary>
						/// The extents of the scanning zone.
						/// </summary>
						public Rectangle Extents { get { return m_Extents; } }
						/// <summary>
						/// Member data on which the Extents property relies. Can be accessed by derived classes.
						/// </summary>
						protected Vector2 m_Steps;
						/// <summary>
						/// Vector containing the distance between centers of adjacent fields of view.
						/// </summary>
						public Vector2 Steps { get { return m_Steps; } }
						/// <summary>
						/// Member data on which the Fragments property relies. Can be accessed by derived classes.
						/// </summary>
						protected uint m_Fragments;
						/// <summary>
						/// The total number of fragments in the scanning zone.
						/// </summary>
						public uint Fragments { get { return m_Fragments; } }
						/// <summary>
						/// Protected member data holding the array of fragment indices. Can be accessed by derived classes.
						/// </summary>
						protected uint [,] FragmentIndices;
						/// <summary>
						/// Provides access to the fragment indices in an array-like fashion.
						/// </summary>
						public uint this[int iy, int ix] { get { return FragmentIndices[iy, ix]; } }
						/// <summary>
						/// X size of the fragment index array (second component).
						/// </summary>
						public uint XSize { get { return (uint)FragmentIndices.GetLength(1); } }
						/// <summary>
						/// Y size of the fragment index array (first component).
						/// </summary>
						public uint YSize { get { return (uint)FragmentIndices.GetLength(0); } }
						/// <summary>
						/// Member data on which the SetupInfo property relies. Can be accessed by derived classes.
						/// </summary>
						protected SetupStringRepresentation m_SetupInfo;
						/// <summary>
						/// Information about the setup of the scanning program to scan this zone.
						/// </summary>
						public SetupStringRepresentation SetupInfo { get { return m_SetupInfo; } }
						/// <summary>
						/// Protected constructor. Prevents users from creating instances of Catalog without deriving the class. Is implicitly called by constructors in derived classes.
						/// </summary>
						protected Catalog() {}
						/// <summary>
						/// Restores a saved catalog from a stream.
						/// </summary>
						/// <param name="s">the stream from where the saved stream is to be loaded.</param>
						public Catalog(System.IO.Stream s)
						{
							System.IO.BinaryReader r = new System.IO.BinaryReader(s);
							byte infotype = r.ReadByte();
							ushort headerformat = r.ReadUInt16();

							if (infotype != ((byte)File.Info.Catalog | (byte)File.Section.Header) ||
								(headerformat != (ushort)File.Format.NormalDouble && headerformat != (ushort)File.Format.Normal && 
								headerformat != (ushort)File.Format.Old)) throw new SystemException("Unknown data format");

							m_Id.Part0 = r.ReadInt32();	
							m_Id.Part1 = r.ReadInt32();
							m_Id.Part2 = r.ReadInt32();
							m_Id.Part3 = r.ReadInt32();

							if (headerformat == (ushort)File.Format.NormalDouble)
							{
								m_Extents.MinX = r.ReadDouble();
								m_Extents.MaxX = r.ReadDouble();
								m_Extents.MinY = r.ReadDouble();
								m_Extents.MaxY = r.ReadDouble();
								m_Steps.X = r.ReadDouble();
								m_Steps.Y = r.ReadDouble();
								uint xviews, yviews;
								xviews = r.ReadUInt32();
								yviews = r.ReadUInt32();
								FragmentIndices = new uint[yviews, xviews];
								m_Fragments = r.ReadUInt32();

								m_SetupInfo = new SetupStringRepresentation();
								if (false /*headerformat == (ushort)Thin.Format.Normal*/)
									m_SetupInfo.m_Name = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64));
								else
									m_SetupInfo.m_Name = "Setup Info";
								m_SetupInfo.m_Configs = new ConfigStringRepresentation[r.ReadUInt32()];
								int i, j;
								for (i = 0; i < m_SetupInfo.m_Configs.Length; i++)
								{
									m_SetupInfo.m_Configs[i] = new ConfigStringRepresentation();
									m_SetupInfo.m_Configs[i].m_ClassName = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
									m_SetupInfo.m_Configs[i].m_Name = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
									m_SetupInfo.m_Configs[i].m_Keys = new KeyStringRepresentation[r.ReadUInt32()];
									for (j = 0; j < m_SetupInfo.m_Configs[i].m_Keys.Length; j++)
									{
										m_SetupInfo.m_Configs[i].m_Keys[j] = new KeyStringRepresentation();
										m_SetupInfo.m_Configs[i].m_Keys[j].m_Name = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
										m_SetupInfo.m_Configs[i].m_Keys[j].m_Value = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
									}
								}
								for (i = 0; i < FragmentIndices.GetLength(0); i++)
									for (j = 0; j < FragmentIndices.GetLength(1); j++)
										FragmentIndices[i, j] = r.ReadUInt32();
								r.ReadBytes(256);
							}
							else
							{
								m_Extents.MinX = r.ReadSingle();
								m_Extents.MaxX = r.ReadSingle();
								m_Extents.MinY = r.ReadSingle();
								m_Extents.MaxY = r.ReadSingle();
								m_Steps.X = r.ReadSingle();
								m_Steps.Y = r.ReadSingle();
								uint xviews, yviews;
								xviews = r.ReadUInt32();
								yviews = r.ReadUInt32();
								FragmentIndices = new uint[yviews, xviews];
								m_Fragments = r.ReadUInt32();

								m_SetupInfo = new SetupStringRepresentation();
								if (false /*headerformat == (ushort)Thin.Format.Normal*/)
									m_SetupInfo.m_Name = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64));
								else
									m_SetupInfo.m_Name = "Setup Info";
								m_SetupInfo.m_Configs = new ConfigStringRepresentation[r.ReadUInt32()];
								int i, j;
								for (i = 0; i < m_SetupInfo.m_Configs.Length; i++)
								{
									m_SetupInfo.m_Configs[i] = new ConfigStringRepresentation();
									m_SetupInfo.m_Configs[i].m_ClassName = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
									m_SetupInfo.m_Configs[i].m_Name = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
									m_SetupInfo.m_Configs[i].m_Keys = new KeyStringRepresentation[r.ReadUInt32()];
									for (j = 0; j < m_SetupInfo.m_Configs[i].m_Keys.Length; j++)
									{
										m_SetupInfo.m_Configs[i].m_Keys[j] = new KeyStringRepresentation();
										m_SetupInfo.m_Configs[i].m_Keys[j].m_Name = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
										m_SetupInfo.m_Configs[i].m_Keys[j].m_Value = System.Text.Encoding.ASCII.GetString(r.ReadBytes(64)).Split('\0')[0];
									}
								}
								for (i = 0; i < FragmentIndices.GetLength(0); i++)
									for (j = 0; j < FragmentIndices.GetLength(1); j++)
										FragmentIndices[i, j] = r.ReadUInt32();
								r.ReadBytes(256);
							}
						}
						/// <summary>
						/// Saves the catalog to a stream.
						/// </summary>
						/// <param name="s">the stream where the catalog is to be saved.</param>
						public virtual void Save(System.IO.Stream s)
						{
							System.IO.BinaryWriter w = new System.IO.BinaryWriter(s, System.Text.Encoding.ASCII);
							w.Write((byte)((byte)File.Info.Catalog | (byte)File.Section.Header));
							w.Write((ushort)File.Format.NormalDouble);

							w.Write(m_Id.Part0);
							w.Write(m_Id.Part1);
							w.Write(m_Id.Part2);
							w.Write(m_Id.Part3);

							w.Write(m_Extents.MinX);
							w.Write(m_Extents.MaxX);
							w.Write(m_Extents.MinY);
							w.Write(m_Extents.MaxY);
							w.Write(m_Steps.X);
							w.Write(m_Steps.Y);

							w.Write(FragmentIndices.GetLength(1));
							w.Write(FragmentIndices.GetLength(0));
							w.Write(m_Fragments);

							byte [] b = new byte[256];
							
							//	w.Write(System.Text.Encoding.ASCII.GetBytes(SetupInfo.Name));
							//	w.Write(b, 0, 64 - SetupInfo.Name.Length);							
							w.Write(m_SetupInfo.m_Configs.Length);
							int i, j;
							for (i = 0; i < m_SetupInfo.m_Configs.Length; i++)
							{
								w.Write(System.Text.Encoding.ASCII.GetBytes(m_SetupInfo.m_Configs[i].m_ClassName));
								w.Write(b, 0, 64 - m_SetupInfo.m_Configs[i].m_ClassName.Length);
								w.Write(System.Text.Encoding.ASCII.GetBytes(m_SetupInfo.m_Configs[i].m_Name));
								w.Write(b, 0, 64 - m_SetupInfo.m_Configs[i].m_Name.Length);
								w.Write(m_SetupInfo.m_Configs[i].m_Keys.Length);
								for (j = 0; j < m_SetupInfo.m_Configs[i].m_Keys.Length; j++)
								{
									w.Write(System.Text.Encoding.ASCII.GetBytes(m_SetupInfo.m_Configs[i].m_Keys[j].m_Name));
									w.Write(b, 0, 64 - m_SetupInfo.m_Configs[i].m_Keys[j].m_Name.Length);
									w.Write(System.Text.Encoding.ASCII.GetBytes(m_SetupInfo.m_Configs[i].m_Keys[j].m_Value));
									w.Write(b, 0, 64 - m_SetupInfo.m_Configs[i].m_Keys[j].m_Value.Length);
								}
							}
							for (i = 0; i < FragmentIndices.GetLength(0); i++)
								for (j = 0; j < FragmentIndices.GetLength(1); j++)
									w.Write(FragmentIndices[i, j]);
							w.Write(b, 0, 256);
						}
					}
				}
			}
		}
	}

	namespace PostProcessing
    {
        /// <summary>
        /// A class for serializing/deserializing slope corrections.
        /// </summary>
        public class SlopeCorrections
        {
            /// <summary>
            /// Section tag for this class in multi-section TLG files.
            /// </summary>
            public const byte SectionTag = 0x82;
            /// <summary>
            /// Thickness of the top emulsion layer.
            /// </summary>
            public double TopThickness;
            /// <summary>
            /// Thickness of the plastic base.
            /// </summary>
            public double BaseThickness;
            /// <summary>
            /// Thickness of the bottom emulsion layer;
            /// </summary>
            public double BottomThickness;
            /// <summary>
            /// Slope displacements (linear distortion) on top side.
            /// </summary>
            public SySal.BasicTypes.Vector2 TopDeltaSlope;
            /// <summary>
            /// Slope multipliers (shrinkage) on top side.
            /// </summary>
            public SySal.BasicTypes.Vector2 TopSlopeMultipliers;
            /// <summary>
            /// Slope displacements (linear distortion) on bottom side.
            /// </summary>
            public SySal.BasicTypes.Vector2 BottomDeltaSlope;
            /// <summary>
            /// Slope multipliers (shrinkage) on bottom side.
            /// </summary>
            public SySal.BasicTypes.Vector2 BottomSlopeMultipliers;
            /// <summary>
            /// Builds an identic slope correction set.            
            /// </summary>
            public SlopeCorrections()
            {
                TopDeltaSlope.X = TopDeltaSlope.Y = BottomDeltaSlope.X = BottomDeltaSlope.Y = 0.0;
                TopSlopeMultipliers.X = TopSlopeMultipliers.Y = BottomSlopeMultipliers.X = BottomSlopeMultipliers.Y = 1.0;
            }
            /// <summary>
            /// Saves a slope correction set to a TLG file.
            /// </summary>
            /// <param name="str">the stream to be used for saving.</param>
            public void Save(System.IO.Stream str)
            {
                if (SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, false)) throw new Exception("BaseTrackIgnoreAligment Section already exists in the stream!");
                System.IO.BinaryWriter w = new System.IO.BinaryWriter(str);
                long nextsectionpos = 0;
                w.Write(SectionTag);
                long nextsectionref = str.Position;
                w.Write(nextsectionpos);
                w.Write(TopThickness);
                w.Write(BaseThickness);
                w.Write(BottomThickness);
                w.Write(TopDeltaSlope.X);
                w.Write(TopDeltaSlope.Y);
                w.Write(TopSlopeMultipliers.X);
                w.Write(TopSlopeMultipliers.Y);
                w.Write(BottomDeltaSlope.X);
                w.Write(BottomDeltaSlope.Y);
                w.Write(BottomSlopeMultipliers.X);
                w.Write(BottomSlopeMultipliers.Y);                
                w.Flush();
                nextsectionpos = str.Position;
                str.Position = nextsectionref;
                w.Write(nextsectionpos);
                w.Flush();
                str.Position = nextsectionpos;
            }
            /// <summary>
            /// Builds a slope correction set from a TLG file. An exception is thrown if the section is not found in the TLG file.
            /// </summary>
            /// <param name="str">the stream to be used for reading.</param>
            public SlopeCorrections(System.IO.Stream str)
            {
                if (!SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, true)) throw new Exception("No SlopeCorrections section found in stream!");
                System.IO.BinaryReader r = new System.IO.BinaryReader(str);
                TopThickness = r.ReadDouble();
                BaseThickness = r.ReadDouble();
                BottomThickness = r.ReadDouble();
                TopDeltaSlope.X = r.ReadDouble();
                TopDeltaSlope.Y = r.ReadDouble();
                TopSlopeMultipliers.X = r.ReadDouble();
                TopSlopeMultipliers.Y = r.ReadDouble();
                BottomDeltaSlope.X = r.ReadDouble();
                BottomDeltaSlope.Y = r.ReadDouble();
                BottomSlopeMultipliers.X = r.ReadDouble();
                BottomSlopeMultipliers.Y = r.ReadDouble();
            }
        }


		namespace PatternMatching
		{
			/// <summary>
			/// Delegates needed to monitor the progress of pattern matching
			/// </summary>
			public delegate bool dShouldStop();
			public delegate void dProgress(double percent);

			/// <summary>
			/// Two tracks matched
			/// </summary>
			public class TrackPair
			{
				public struct IndexedTrack
				{
					public int Index;
					public object Track;
					public MIPEmulsionTrackInfo Info;
				}

				public IndexedTrack First, Second;

				public TrackPair() 
				{
					First.Index = Second.Index = -1;
					First.Track = Second.Track = null;
					First.Info = Second.Info = null;
				}

				public TrackPair(MIPEmulsionTrackInfo f, int firstindex, MIPEmulsionTrackInfo s, int secondindex)
				{
					First.Track = f;
					Second.Track = s;
					First.Index = firstindex;
					Second.Index = secondindex;
					First.Info = f;
					Second.Info = s;
				}

				public TrackPair(MIPEmulsionTrack f, int firstindex, MIPEmulsionTrack s, int secondindex)
				{
					First.Track = f;
					First.Info = f.Info;
					First.Index = firstindex;
					Second.Track = s;
					Second.Info = s.Info;
					Second.Index = secondindex;
				}

				public TrackPair(MIPBaseTrack f, int firstindex, MIPBaseTrack s, int secondindex)
				{
					First.Track = f;
					First.Info = f.Info;
					First.Index = firstindex;
					Second.Track = s;
					Second.Info = s.Info;
					Second.Index = secondindex;
				}
			}

			/// <summary>
			/// Pattern matching has its own exceptions
			/// </summary>
			public class PatternMatchingException : System.Exception
			{
				public PatternMatchingException(string message) : base(message) {}

				public PatternMatchingException() : base("Pattern matching procedure failed.") {}
			}

			/// <summary>
			/// Methods that any pattern matching library should implement
			/// </summary>
			public interface IPatternMatcher
			{
				dShouldStop ShouldStop
				{
					get;
					set;
				}

				dProgress Progress
				{
					get;
					set;
				}

				TrackPair [] Match(MIPEmulsionTrackInfo [] moreprecisepattern, MIPEmulsionTrackInfo [] secondpattern, double Zprojection, double maxoffsetx, double maxoffsety);

				TrackPair [] Match(MIPEmulsionTrack [] moreprecisepattern, MIPEmulsionTrack [] secondpattern, double Zprojection, double maxoffsetx, double maxoffsety);

				TrackPair [] Match(MIPBaseTrack [] moreprecisepattern, MIPBaseTrack [] secondpattern, double Zprojection, double maxoffsetx, double maxoffsety);
			}
		}

		/// <summary>
		/// Returns true if a data processing task has to be aborted.
		/// </summary>
		public delegate bool dShouldStop();
		/// <summary>
		/// Loads a fragment with the specified index from a raw data fragment group.
		/// </summary>
		public delegate SySal.Scanning.Plate.IO.OPERA.RawData.Fragment dLoadFragment(uint index);
		/// <summary>
		/// Notifies about the progress of a data processing operation in percentage (0 - 100.0).
		/// </summary>
		public delegate void dProgress(double percent);
		/// <summary>
		/// Notifies that a fragment has been completed.
		/// </summary>
		public delegate void dFragmentComplete(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag);

		/// <summary>
		/// Raw data processor interface.
		/// </summary>
		public interface IPrelinkProcessor
		{
			/// <summary>
			/// The delegate to be called to check user's request to abort the processing.
			/// </summary>
			dShouldStop ShouldStop
			{
				get;
				set;
			}
			/// <summary>
			/// The delegate that is called to notify the caller about the processing progress status.
			/// </summary>
			dProgress Progress
			{
				get;
				set;
			}
			/// <summary>
			/// The delegate that takes care of loading a fragment.
			/// </summary>
			dLoadFragment Load
			{
				get;
				set;
			}
			/// <summary>
			/// The delegate that is called when a fragment has been successfully processed.
			/// </summary>
			dFragmentComplete FragmentComplete
			{
				get;
				set;
			}
			/// <summary>
			/// Processes a group of raw data specified by a catalog.
			/// </summary>
			/// <param name="cat">the catalog that defines the raw data to be processed.</param>
			/// <returns>true if the raw data have been successfully processed, false otherwise.</returns>
			bool Process(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat);
		}

		/// <summary>
		/// Link processor interface.
		/// </summary>
		public interface ILinkProcessor
		{
			/// <summary>
			/// The delegate to be called to check user's request to abort the processing.
			/// </summary>
			dShouldStop ShouldStop
			{
				get;
				set;
			}
			/// <summary>
			/// The delegate that is called to notify the caller about the processing progress status.
			/// </summary>
			dProgress Progress
			{
				get;
				set;
			}
			/// <summary>
			/// The delegate that takes care of loading a fragment.
			/// </summary>
			dLoadFragment Load
			{
				get;
				set;
			}
			/// <summary>
			/// Links a group of raw data specified by a catalog.
			/// </summary>
			/// <param name="cat">the catalog that defines the raw data to be linked.</param>
			/// <returns>the LinkedZone that has been generated.</returns>
			SySal.Scanning.Plate.IO.OPERA.LinkedZone Link(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat);
		}

		/// <summary>
		/// Postlink processor interface.
		/// </summary>
		public interface IPostlinkProcessor
		{
			/// <summary>
			/// The delegate to be called to check user's request to abort the processing.
			/// </summary>
			dShouldStop ShouldStop
			{
				get;
				set;
			}
			/// <summary>
			/// The delegate that is called to notify the caller about the processing progress status.
			/// </summary>
			dProgress Progress
			{
				get;
				set;
			}
			/// <summary>
			/// Processes a linked zone.
			/// </summary>
			/// <param name="lz">the linked zone to be processed.</param>
			/// <returns>the LinkedZone that has been generated. Can also retrieve the input linked zone if no modification has been made.</returns>
			SySal.Scanning.Plate.IO.OPERA.LinkedZone Process(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz);
		}


		#region OldProcessingSteps

		namespace FieldShiftCorrection
		{
			public abstract class FragmentCorrection : ISerializable
			{
				public abstract void Correct(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag);
				public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
				{
					throw new System.Exception("Can't save information about an abstract class");	
				}
			}

			public struct FieldShift
			{
				public Vector2 Delta;
				public Vector2 DeltaErrors;
				public uint MatchCount;
				public uint FirstViewIndex, SecondViewIndex;
				
				public enum SideValue : byte { Top = 0, Bottom = 1, Both = 2 }

				public SideValue Side;
			}

			/// <summary>
			/// Delegates that are needed to monitor the asynchronous field shift correction process
			/// </summary>
			public delegate bool dShouldStop();
			public delegate SySal.Scanning.Plate.IO.OPERA.RawData.Fragment dLoad(uint index);
			public delegate void dProgress(double percent);
			public delegate void dFragmentComplete(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag);

			/// <summary>
			/// The field shift correction procedure has its own exceptions
			/// </summary>
			public class FieldShiftException : System.Exception
			{
				public FieldShiftException(string message) : base(message) {}

				public FieldShiftException() : base("Field Shift correction procedure failed") {}
			}

			public class NoFragmentLoaderException : FieldShiftException
			{
				public NoFragmentLoaderException(string message) : base(message) {}

				public NoFragmentLoaderException() : base("No loader delegate specified") {}
			}

			public class NoFragmentCompleteException : FieldShiftException
			{
				public NoFragmentCompleteException(string message) : base(message) {}

				public NoFragmentCompleteException() : base("No fragment complete delegate specified") {}
			}

			/// <summary>
			/// Methods that any field shift manager must implement
			/// </summary>
			public interface IFieldShiftManager
			{
				dShouldStop ShouldStop
				{
					get;
					set;
				}

				dProgress Progress
				{
					get;
					set;
				}

				dLoad Load
				{
					get;
					set;
				}

				dFragmentComplete FragmentComplete
				{
					get;
					set;
				}

				void Test(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag);

				void ComputeFragmentCorrection(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat, FieldShift.SideValue side, out FieldShift [] shifts, out FragmentCorrection corr);

				void AdjustDisplacedFragments(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat, FragmentCorrection corr);
			}			
		}

		namespace FragmentLinking
		{
			/// <summary>
			/// The fragment linking procedure has its own exceptions
			/// </summary>
			public class LinkException : System.Exception
			{
				public LinkException(string message) : base(message) {}

				public LinkException() : base("Fragment linking procedure failed") {}
			}

			public class NoFragmentLoaderException : LinkException
			{
				public NoFragmentLoaderException(string message) : base(message) {}

				public NoFragmentLoaderException() : base("No loader delegate specified") {}
			}

			/// <summary>
			/// Methods that any fragment linker must implement
			/// </summary>
			public interface IFragmentLinker
			{
				dShouldStop ShouldStop
				{
					get;
					set;
				}

				dProgress Progress
				{
					get;
					set;
				}

				dLoadFragment Load
				{
					get;
					set;
				}

				SySal.Scanning.Plate.LinkedZone Link(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat, System.Type outputtype);
			}
		}


		#endregion
	}
}
