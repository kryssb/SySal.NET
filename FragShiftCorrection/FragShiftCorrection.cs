using System;
using SySal;
using SySal.BasicTypes;
using SySal.Management;
using SySal.Imaging;
using SySal.Tracking;
using SySal.Scanning;
using SySal.Scanning.Plate;
using SySal.Scanning.PostProcessing.FieldShiftCorrection;
using System.Windows.Forms;
using System.Collections;
using System.Runtime.Serialization;
using SySal.Scanning.Plate.IO.OPERA.RawData;
using System.Xml.Serialization;
using NumericalTools;

namespace SySal.Processing.FragShiftCorrection
{
	/// <summary>
	/// Configuration for FragShiftCorrection.
	/// </summary>
	[Serializable]
	[XmlType("FragShiftCorrection.Configuration")]
	public class Configuration : SySal.Management.Configuration
	{
		/// <summary>
		/// Minimum number of grains to consider a track for systematic error computation.
		/// </summary>
		public int MinGrains;
		/// <summary>
		/// Minimum slope to consider a track for systematic error computation.
		/// </summary>
		public double MinSlope;
		/// <summary>
		/// Position tolerance to merge two tracks in the same field of view.
		/// </summary>
		public double MergePosTol;
		/// <summary>
		/// Slope tolerance to merge two tracks in the same field of view.
		/// </summary>
		public double MergeSlopeTol;
		/// <summary>
		/// Position tolerance to detect a cross-field doubly reconstructed track, the basis of systematic error correction.
		/// </summary>
		public double PosTol;
		/// <summary>
		/// Slope tolerance to detect a cross-field doubly reconstructed track, the basis of systematic error correction.
		/// </summary>
		public double SlopeTol;
		/// <summary>
		/// Minimum number of doubly reconstructed tracks in a pair of fields of view.
		/// </summary>
		public int MinMatches;
		/// <summary>
		/// Maximum dispersion of matching deviations (in position).
		/// </summary>
		public double MaxMatchError;	
		/// <summary>
		/// Minimum overlap in Z of two measurements of a doubly reconstructed track.
		/// </summary>
		public double GrainsOverlapRatio;
		/// <summary>
		/// Overlap tolerance in micron between two possible measurements of the same track.
		/// </summary>
		public double OverlapTol;
		/// <summary>
		/// Tolerance in Z for grains of two measurements of the same track.
		/// </summary>
		public double GrainZTol;
		/// <summary>
		/// If true, hysteresis is accounted for in a single step passing from forward to backward X axis motion, otherwise a sinusoidal profile is used.
		/// </summary>
		public bool IsStep;
		/// <summary>
		/// If true, mechanical hysteresis estimation is activated.
		/// </summary>
		public bool EnableHysteresis;

		/// <summary>
		/// Builds an empty configuration.
		/// </summary>
		public Configuration() : base("") {}

		/// <summary>
		/// Builds and empty configuration with a name.
		/// </summary>
		/// <param name="name">the configuration name.</param>
		public Configuration(string name) : base(name) {}

		/// <summary>
		/// Clones this configuration.
		/// </summary>
		/// <returns>the object clone.</returns>
		public override object Clone()
		{
			Configuration c = new Configuration(Name);
			c.MinGrains = MinGrains;
			c.MinSlope = MinSlope;
			c.MergePosTol = MergePosTol;
			c.MergeSlopeTol = MergeSlopeTol;
			c.PosTol = PosTol;
			c.SlopeTol = SlopeTol;
			c.MinMatches = MinMatches;
			c.MaxMatchError = MaxMatchError;
			c.GrainsOverlapRatio = GrainsOverlapRatio;
			c.OverlapTol = OverlapTol;
			c.GrainZTol = GrainZTol;
			c.IsStep = IsStep;
			c.EnableHysteresis = EnableHysteresis;
			return c;
		}
	}

	class MySide : SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side
	{
		static public void MySetPos(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, Vector2 pos) { SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.SetPos(s, pos); }
		static public void MySetMapPos(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, Vector2 mappos) { SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.SetMapPos(s, mappos); }
		static public void MySetM(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, double mxx, double mxy, double myx, double myy) { SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.SetM(s, mxx, mxy, myx, myy); }
	}

	class MyMIPEmulsionTrack : SySal.Tracking.MIPEmulsionTrack
	{
		static public SySal.Tracking.MIPEmulsionTrackInfo GetInfo(SySal.Tracking.MIPEmulsionTrack tk) { return SySal.Tracking.MIPEmulsionTrack.AccessInfo(tk); }
	}

	/// <summary>
	/// Linear correction of systematic shifts (mostly camera conversion factors and camera rotation) plus hysteretic contributions.
	/// </summary>
	[Serializable]
	public class LinearFragmentCorrectionWithHysteresis : SySal.Scanning.PostProcessing.FieldShiftCorrection.FragmentCorrection, ISerializable
	{
		/// <summary>
		/// The components of the deformation matrix.
		/// </summary>
		public double MXX, MXY, MYX, MYY;
		/// <summary>
		/// X mechanical hysteresis.
		/// </summary>
		public double XHysteresis;
		/// <summary>
		/// Period of X motion (computed from fragment analysis).
		/// </summary>
		public uint XPeriod;
		/// <summary>
		/// If true, the sequence of views is a zig-zag path with main motion along X; if false, the main motion is along Y.
		/// </summary>
		public bool IsHoriz;
		/// <summary>
		/// If true, the zig-zag path moves towards increasing Y.
		/// </summary>
		public bool IsUp;
		/// <summary>
		/// If true, step-correction is used for hysteresis.
		/// </summary>
		public bool IsStep;

		internal static double Step(int x, uint p)
		{
			if (x == 0 || x == p) return 0.0;
			if ((x % (2 * p)) < p) return 1.0;
			return -1.0;
		}

		/// <summary>
		/// Field-to-field shifts that extends the FieldShift found in SySal.Scanning.
		/// </summary>
		public struct FieldShift
		{
			/// <summary>
			/// Field-to-field shift.
			/// </summary>
			public SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShift FS;
			/// <summary>
			/// If true, the views share a vertical boundary (i.e., they are adjacent along the X direction).
			/// </summary>
			public bool IsHorizontalMatch;
			/// <summary>
			/// If true, the zig-zag path starts towards increasing X.
			/// </summary>
			public bool IsRightHeadingCoil;
			/// <summary>
			/// If true, the zig-zag path moves progressively towards increasing Y.
			/// </summary>
			public bool IsUpHeadingCoil;
			/// <summary>
			/// Distance between centers.
			/// </summary>
			public double CenterDistance;
			/// <summary>
			/// Initializes a FieldShift.
			/// </summary>
			/// <param name="fs">the value of the base FieldShift structure.</param>
			/// <param name="ishmatch">the value of IsHorizontalMatch.</param>
			/// <param name="isright">the value of IsRightHeadingCoil.</param>
			/// <param name="isup">the value of IsUpHeadingCoil.</param>
			/// <param name="fdist">the value of CenterDistance.</param>
			public FieldShift(SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShift fs, bool ishmatch, bool isright, bool isup, double fdist)
			{
				FS = fs;
				IsHorizontalMatch = ishmatch;
				IsRightHeadingCoil = isright;
				IsUpHeadingCoil = isup;
				CenterDistance = fdist;
			}
		}

		/// <summary>
		/// Applies the correction to a fragment.
		/// </summary>
		/// <param name="frag">the fragment to be corrected.</param>
		public override void Correct(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag)
		{
			double X, Y, DeltaXHist;
			int i, j, k;
			bool isright;

			for (i = 0; i < frag.Length; i++)
			{
				SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View v = frag[i];
				SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s = v.Top;
				
				
				while (s != null)
				{
					Vector2 p;
					if (IsStep)
					{
						isright = (frag[0].Tile.X < frag[1].Tile.X);
						DeltaXHist = (double)((isright ? 0.5 : -0.5) * XHysteresis * Step(i, XPeriod));
						p = s.Pos;
						p.X += DeltaXHist;
						MySide.MySetPos(s, p);
						p = s.MapPos;
						p.X += (s.MXX * DeltaXHist);
						p.Y += (s.MYX * DeltaXHist);
						MySide.MySetMapPos(s, p);
					}
					else
						if ((i % XPeriod) < (XPeriod / 2))
						{
							isright = (frag[0].Tile.X < frag[1].Tile.X);
							DeltaXHist = (double)((isright ? 1 : -1) * XHysteresis * Math.Sin(i * Math.PI / XPeriod));
							p = s.Pos;
							p.X += DeltaXHist;
							MySide.MySetPos(s, p);
							p = s.MapPos;
							p.X += (s.MXX * DeltaXHist);
							p.Y += (s.MYX * DeltaXHist);
							MySide.MySetMapPos(s, p);
						}

					double mxx, mxy, myx, myy;
					mxx = s.MXX * (1.0 + MXX) + s.MXY * MYX;
					mxy = s.MXX * MXY + s.MXY * (1.0 + MYY);
					myx = s.MYX * (1.0 + MXX) + s.MYY * MYX;
					myy = s.MYX * MXY + s.MYY * (1.0 + MYY);
					MySide.MySetM(s, mxx, mxy, myx, myy);

					s = (s == v.Top) ? v.Bottom : null;
				}				
			}
		}

		/// <summary>
		/// Used for serialization.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			info.AddValue("CorrectionMatrixXX", MXX);
			info.AddValue("CorrectionMatrixXY", MXY);
			info.AddValue("CorrectionMatrixYX", MYX);
			info.AddValue("CorrectionMatrixYY", MYY);
			info.AddValue("XHysteresis", XHysteresis);
			info.AddValue("XPeriod", XPeriod);
			info.AddValue("IsHorizontal",IsHoriz);
			info.AddValue("IsUp", IsUp);
			info.AddValue("IsStep", IsStep);
		}
		
		/// <summary>
		/// Creates an empty instance.
		/// </summary>
		public LinearFragmentCorrectionWithHysteresis() {}

		/// <summary>
		/// Initializes an instance with specific parameters.
		/// </summary>
		/// <param name="mat">the deformation matrix.</param>
		/// <param name="xhyst">the X hysteresis.</param>
		/// <param name="xperiod">the X period.</param>
		/// <param name="ishoriz">if true, the zig-zag path is horizontal.</param>
		/// <param name="isup">if true, the zig-zag path moves towards increasing Y.</param>
		/// <param name="isstep">if true, the hysteresis contributio is a step function.</param>
		public LinearFragmentCorrectionWithHysteresis(double [,] mat, double xhyst, 
										uint xperiod, bool ishoriz, bool isup, bool isstep)
		{
			if (mat.Rank != 2 || mat.GetLength(0) != 2 || mat.GetLength(1) != 2)
				throw new SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShiftException("Wrong matrix size.");
			if (xperiod<2)
				throw new SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShiftException("Wrong coil length.");
			if (ishoriz==false/* || isup== true*/)
				throw new SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShiftException("Correction for such a coil not yet implemented.");

			MXX = mat[0, 0];
			MXY = mat[0, 1];
			MYX = mat[1, 0];
			MYY = mat[1, 1];
			XHysteresis = xhyst;
			XPeriod = xperiod;
			IsHoriz = ishoriz;
			IsUp = isup;
			IsStep = isstep;
		}

		/// <summary>
		/// Used for serialization.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public LinearFragmentCorrectionWithHysteresis(SerializationInfo info, StreamingContext context)
		{
			MXX = info.GetDouble("CorrectionMatrixXX");
			MXY = info.GetDouble("CorrectionMatrixXY");
			MYX = info.GetDouble("CorrectionMatrixYX");
			MYY = info.GetDouble("CorrectionMatrixYY");
			XHysteresis = info.GetDouble("XHysteresis");
			XPeriod = info.GetUInt32("XPeriod");
			IsHoriz = info.GetBoolean("IsHorizontal");
			IsUp = info.GetBoolean("IsUp");
			IsStep = info.GetBoolean("IsStep");
		}
	}	

	/// <summary>
	/// FragmentShiftManager handles field shifts for whole fragments.
	/// </summary>
	/// <remarks>
	/// <para>
	/// The algorithm implemented to compute systematic error corrections works as follows:
	/// <list type="bullet">
	/// <item><term>For each pair of adjacent views, look for tracks in the overlap region that have been seen in both views.</term></item>
	/// <item><term>Measure the position discrepancy between the tracks.</term></item>
	/// <item><term>Build distributions of these discrepancies, so that the linear correlations can be extracted.</term></item>
	/// <item><term>Compute the deformation matrix, and, if required, the X hysteresis.</term></item>
	/// </list>
	/// The correlations considered are:
	/// <list type="table">
	/// <listheader><term>Correlation</term><description>Meaning</description></listheader>
	/// <item><term>DeltaX vs. X</term><description>Error in Pixel-to-micron X conversion factor.</description></item>
	/// <item><term>DeltaY vs. Y</term><description>Error in Pixel-to-micron X conversion factor.</description></item>
	/// <item><term>DeltaX vs. Y</term><description>Camera rotation.</description></item>
	/// <item><term>DeltaY vs. X</term><description>Camera rotation.</description></item>
	/// </list>
	/// Normally, the offset of the DeltaX vs. Y distribution is different for views that are in increasing X sequence and views that are in decreasing X sequence. This effect can be measured and is due to X mechanical hysteresis.
	/// </para>	
	/// </remarks>
	[Serializable]
	[XmlType("SySal.Processing.FragShiftCorrection.FragmentShiftManager")]
	public class FragmentShiftManager : IFieldShiftManager, IManageable
	{
		[NonSerialized]
		private FragShiftCorrection.Configuration C;

		[NonSerialized]
		private string intName;

		[NonSerialized]
		private dShouldStop intShouldStop;

		[NonSerialized]
		private dProgress intProgress;

		[NonSerialized]
		private dLoad intLoad;

		[NonSerialized]
		private dFragmentComplete intFragmentComplete;

		[NonSerialized]
		private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

		#region Management
		public FragmentShiftManager()
		{
			//
			// TODO: Add constructor logic here
			//
			C = new FragShiftCorrection.Configuration("Default Fragment Field Shift Manager Config");
			C.MinGrains = 6;
			C.PosTol = 50.0f;
			C.SlopeTol = 0.07f;
			C.MergePosTol = 20.0f;
			C.MergeSlopeTol = 0.02f;
			C.MinSlope = 0.010f;
			C.MinMatches = 2;
			C.MaxMatchError = 1.0f;
			C.GrainsOverlapRatio = 0.2f;
			C.OverlapTol = 40.0f;
			C.GrainZTol = 2.0f;
			C.IsStep = true;
			C.EnableHysteresis = false;

			intName = "Default Stripes FragmentShiftManager";
		}

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

		[XmlElement(typeof(FragShiftCorrection.Configuration))]
		public SySal.Management.Configuration Config
		{
			get
			{
				return C;
			}
			set
			{
				C = (FragShiftCorrection.Configuration)value;
			}
		}

		public bool EditConfiguration(ref SySal.Management.Configuration c)
		{
			bool ret;
			EditConfigForm myform = new EditConfigForm();
			myform.C = (FragShiftCorrection.Configuration)c.Clone();
			if ((ret = (myform.ShowDialog() == DialogResult.OK))) c = myform.C;
			myform.Dispose();
			return ret;
		}

		[XmlIgnore]
		public IConnectionList Connections
		{
			get
			{
				return EmptyConnectionList;
			}
		}

		[XmlIgnore]
		public bool MonitorEnabled
		{
			get
			{
				return false;
			}
			set
			{
				if (value != false) throw new System.Exception("This object has no monitor.");
			}
		}
		#endregion

		#region IFieldShiftManager

		#region Internals

		struct IntMatch
		{
			public IntTrack FirstT, SecondT;
			public IntView FirstV, SecondV;

			public IntMatch(IntTrack ft, IntTrack st, IntView fv, IntView sv) 
			{
				FirstT = ft;
				SecondT = st;
				FirstV = fv;
				SecondV = sv;
			}
		}
		
		class IntTrack
		{
			public bool Valid;

			public SySal.Tracking.MIPEmulsionTrackInfo Info;

			public SySal.Tracking.Grain [] Grains;

			public IntTrack() {}

			public IntTrack(MIPEmulsionTrack t)
			{
				Info = t.Info;
				Grains = new Grain[t.Length];
				int i;
				for (i = 0; i < Grains.Length; i++)
					Grains[i] = t[i];
				Valid = true;
			}
		}

		class IntCell
		{
			public int Fill;
			public IntTrack [] Tracks;
		}

		class IntSide
		{			
			public int Fill;
			public IntTrack [] Tracks;			
			public IntCell [] TopCells, BottomCells, LeftCells, RightCells;
			public double ZBase, ZExt;
			public Vector2 TopMin, BottomMin, LeftMin, RightMin;
			public Vector2 TopMax, BottomMax, LeftMax, RightMax;

			#region Side information
			public Vector2 Pos;
			public Vector2 MapPos;
			public double MXX, MXY, MYX, MYY;
			public SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.SideFlags Flags;
			public SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.LayerInfo [] Layers;
			#endregion

			public IntSide(Fragment.View.Side s, bool istop, double mergepostol, double mergepostol2, double mergeslopetol2, double overlaptol, double minslope2, int mincount, bool retainalltracks)
			{
				int i, j;
				Pos = s.Pos;
				MapPos = s.MapPos;
				MXX = s.MXX;
				MXY = s.MXY;
				MYX = s.MYX;
				MYY = s.MYY;
				Flags = s.Flags;
				Layers = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.LayerInfo[s.Layers.Length];
				for (i = 0; i < s.Layers.Length; i++)
					Layers[i] = s.Layers[i];

				IntCell [,] Cells;
				Vector2 Min, Max;

				ZBase = istop ? s.BottomZ : s.TopZ;
				ZExt = istop ? s.TopZ : s.BottomZ;
				Tracks = new IntTrack[s.Length];
				for (i = Fill = 0; i < Tracks.Length; i++)
					if (MyMIPEmulsionTrack.GetInfo(s[i]).Count >= mincount && (MyMIPEmulsionTrack.GetInfo(s[i]).Slope.X * MyMIPEmulsionTrack.GetInfo(s[i]).Slope.X + MyMIPEmulsionTrack.GetInfo(s[i]).Slope.Y * MyMIPEmulsionTrack.GetInfo(s[i]).Slope.Y) > minslope2)
					{
						IntTrack t = (Tracks[Fill] = new IntTrack(s[i]));
						Fill++;
					}
				if (Fill > 0)
				{
					Max.X = Min.X = Tracks[0].Info.Intercept.X;
					Max.Y = Min.Y = Tracks[0].Info.Intercept.Y;
					for (i = 1; i < Fill; i++)
					{
						if (Tracks[i].Info.Intercept.X < Min.X) Min.X = Tracks[i].Info.Intercept.X;
						else if (Tracks[i].Info.Intercept.X > Max.X) Max.X = Tracks[i].Info.Intercept.X;
						if (Tracks[i].Info.Intercept.Y < Min.Y) Min.Y = Tracks[i].Info.Intercept.Y; 
						else if (Tracks[i].Info.Intercept.Y > Max.Y) Max.Y = Tracks[i].Info.Intercept.Y; 
					}
					int xc, yc, ix, iy;
					xc = (int)Math.Floor((Max.X - Min.X) / mergepostol/* + 0.5f*/) + 3;
					yc = (int)Math.Floor((Max.Y - Min.Y) / mergepostol/* + 0.5f*/) + 3;
					LeftMin = BottomMin = Min;
					Min.X -= mergepostol;
					Min.Y -= mergepostol;
					Cells = new IntCell[yc, xc];
					for (iy = 0; iy < yc; iy++)
						for (ix = 0; ix < xc; ix++)
							Cells[iy, ix] = new IntCell();
					IntCell [] tempindex = new IntCell[Fill];
					for (i = 0; i < Fill; i++)
					{
						ix = (int)Math.Floor((Tracks[i].Info.Intercept.X - Min.X) / mergepostol/* + 0.5*/);
						iy = (int)Math.Floor((Tracks[i].Info.Intercept.Y - Min.Y) / mergepostol/* + 0.5*/);
						(tempindex[i] = Cells[iy, ix]).Fill++;
					}
					for (iy = 0; iy < yc; iy++)
						for (ix = 0; ix < xc; ix++)
						{
							Cells[iy, ix].Tracks = new IntTrack[Cells[iy, ix].Fill];
							Cells[iy, ix].Fill = 0;
						}
					for (i = 0; i < Fill; i++)
						tempindex[i].Tracks[tempindex[i].Fill++] = Tracks[i];
					
					Clean(Min, Cells, mergepostol, mergepostol2, mergeslopetol2);

					TopMin.X = LeftMin.X;
					TopMin.Y = Max.Y - overlaptol;
					RightMin.Y = LeftMin.Y;
					RightMin.X = Max.X - overlaptol;

					LeftMax.X = LeftMin.X + overlaptol;
					LeftMax.Y = Max.Y;
					RightMax.X = RightMin.X + overlaptol;
					RightMax.Y = Max.Y;
					TopMax.X = Max.X;
					TopMax.Y = TopMin.Y + overlaptol;
					BottomMax.X = Max.X;
					BottomMax.Y = BottomMin.Y + overlaptol;

					TopCells = new IntCell[(int)Math.Floor((Max.X - LeftMin.X) / overlaptol + 1.0f)];
					LeftCells = new IntCell[(int)Math.Floor((Max.Y - LeftMin.Y) / overlaptol + 1.0f)];
					BottomCells = new IntCell[(int)Math.Floor((Max.X - LeftMin.X) / overlaptol + 1.0f)];
					RightCells = new IntCell[(int)Math.Floor((Max.Y - LeftMin.Y) / overlaptol + 1.0f)];
					for (j = 0; j < TopCells.Length; TopCells[j++] = new IntCell());
					for (j = 0; j < BottomCells.Length; BottomCells[j++] = new IntCell());
					for (j = 0; j < LeftCells.Length; LeftCells[j++] = new IntCell());
					for (j = 0; j < RightCells.Length; RightCells[j++] = new IntCell());

					for (j = 0; j < 2; j++)
					{
						Vector2 min;
						Vector2 max;
						IntCell [] cells;
						if (j == 0)
						{
							min = LeftMin;
							max = LeftMax;
							cells = LeftCells;
						}
						else
						{
							min = RightMin;
							max = RightMax;
							cells = RightCells;
						}
						for (i = 0; i < Fill; i++)
							if (Tracks[i].Valid && Tracks[i].Info.Intercept.X >= min.X && Tracks[i].Info.Intercept.X <= max.X)
							{
								iy = (int)((Tracks[i].Info.Intercept.Y - min.Y) / overlaptol);
								if (iy < 0) iy = 0;
								else if (iy >= cells.Length) iy = cells.Length - 1;
								cells[iy].Fill++;
							}
						for (i = 0; i < cells.Length; i++)
						{
							cells[i].Tracks = new IntTrack[cells[i].Fill];
							cells[i].Fill = 0;
						}
						for (i = 0; i < Fill; i++)
							if (Tracks[i].Valid && Tracks[i].Info.Intercept.X >= min.X && Tracks[i].Info.Intercept.X <= max.X)
							{
								iy = (int)((Tracks[i].Info.Intercept.Y - min.Y) / overlaptol);
								if (iy < 0) iy = 0;
								else if (iy >= cells.Length) iy = cells.Length - 1;
								cells[iy].Tracks[cells[iy].Fill++] = Tracks[i];
							}
					}

					for (j = 0; j < 2; j++)
					{
						Vector2 min;
						Vector2 max;
						IntCell [] cells;
						if (j == 0)
						{
							min = BottomMin;
							max = BottomMax;
							cells = BottomCells;
						}
						else
						{
							min = TopMin;
							max = TopMax;
							cells = TopCells;
						}
						for (i = 0; i < Fill; i++)
							if (Tracks[i].Valid && Tracks[i].Info.Intercept.Y >= min.Y && Tracks[i].Info.Intercept.Y <= max.Y)
							{
								ix = (int)((Tracks[i].Info.Intercept.X - min.X) / overlaptol);
								if (ix < 0) ix = 0;
								else if (ix >= cells.Length) ix = cells.Length - 1;
								cells[ix].Fill++;
							}
						for (i = 0; i < cells.Length; i++)
						{
							cells[i].Tracks = new IntTrack[cells[i].Fill];
							cells[i].Fill = 0;
						}
						for (i = 0; i < Fill; i++)
							if (Tracks[i].Valid && Tracks[i].Info.Intercept.Y >= min.Y && Tracks[i].Info.Intercept.Y <= max.Y)
							{
								ix = (int)((Tracks[i].Info.Intercept.X - min.X) / overlaptol);
								if (ix < 0) ix = 0;
								else if (ix >= cells.Length) ix = cells.Length - 1;
								cells[ix].Tracks[cells[ix].Fill++] = Tracks[i];
							}
					}

				}
				else
				{
					TopCells = BottomCells = LeftCells = RightCells = new IntCell[0];
				}
				if (!retainalltracks) Tracks = null;
			}

			private void Clean(Vector2 Min, IntCell [,] Cells, double mergepostol, double mergepostol2, double mergeslopetol2)
			{
				int i, j, ix, iy, iix, iiy, sx, sy;
				double slx, sly, inx, iny;
				sy = Cells.GetLength(0);
				sx = Cells.GetLength(1);
				for (i = 0; i < Fill; i++)
				{
					IntTrack pt = Tracks[i];
					slx = pt.Info.Slope.X;
					sly = pt.Info.Slope.Y;
					ix = (int)(((inx = pt.Info.Intercept.X) - Min.X) / mergepostol/* + 0.5*/);
					iy = (int)(((iny = pt.Info.Intercept.Y) - Min.Y) / mergepostol/* + 0.5*/);
					for (iiy = iy - 1; iiy <= iy + 1; iiy++)
						if (iiy >= 0 && iiy < sy)
							for (iix = ix - 1; iix <= ix + 1; iix++)
								if (iix >= 0 && iix < sx)
								{
									IntCell c = Cells[iiy, iix];
									for (j = 0; j < c.Tracks.Length; j++)
									{
										IntTrack ptt = c.Tracks[j];
										if (pt != ptt)
										{
											double dslx = slx - ptt.Info.Slope.X;
											double dsly = sly - ptt.Info.Slope.Y;
											double dinx = inx - ptt.Info.Intercept.X;
											double diny = iny - ptt.Info.Intercept.Y;
											if ((dslx * dslx + dsly * dsly) < mergeslopetol2 &&
												(dinx * dinx + diny * diny) < mergepostol2)
											{
												if (pt.Info.Count >= ptt.Info.Count) ptt.Valid = false;
												else pt.Valid = false;
											}
										}
									}
								}
				}
			}

			public int GetShifts(IntSide s, bool isHorizontal, ref FieldShift shifts, double overlaptol, double matchtol, double slopetol, double grainsoverlapratio, double grainztol, 
				out ArrayList listofmatches, IntView f, IntView g)
			{
				listofmatches = new ArrayList();

				double field_dx = Pos.X - s.Pos.X;
				double field_dy = Pos.Y - s.Pos.Y;

				double dx = 0.0f;
				double dy = 0.0f;
				double dx2 = 0.0f;
				double dy2 = 0.0f;
				uint matches = 0;
				int totaloverlap = 0;


				IntCell [] ThisCells;
				IntCell [] OtherCells;
				double OtherMin;				

				if (isHorizontal)
				{
					ThisCells = RightCells;
					OtherCells = s.LeftCells;
					OtherMin = s.LeftMin.Y - field_dy;
				}
				else
				{
					ThisCells = TopCells;
					OtherCells = s.BottomCells;
					OtherMin = s.BottomMin.X - field_dx;
				}

				foreach (IntCell a in ThisCells)
					foreach (IntTrack at in a.Tracks)
					{
						int index = (int)(((isHorizontal ? at.Info.Intercept.Y : at.Info.Intercept.X) - OtherMin) / overlaptol);
						int iindex;
						for (iindex = 0; iindex < OtherCells.Length; iindex++)
							if (iindex < 0 || iindex >= OtherCells.Length) continue;
							else							
								foreach (IntTrack bt in OtherCells[iindex].Tracks)																		
									if (Math.Abs(at.Info.Slope.X - bt.Info.Slope.X) < slopetol &&
										Math.Abs(at.Info.Slope.Y - bt.Info.Slope.Y) < slopetol &&
										Math.Abs(field_dx + (at.Info.Intercept.X - bt.Info.Intercept.X)) < matchtol &&
										Math.Abs(field_dy + (at.Info.Intercept.Y - bt.Info.Intercept.Y)) < matchtol)
									{
										double t_dx = 0.0f, t_dy = 0.0f, t_dx2 = 0.0f, t_dy2 = 0.0f;
										int overlapcount = 0;
										int minoverlapgrains = (int)(grainsoverlapratio * ((at.Info.Count < bt.Info.Count) ? at.Info.Count : bt.Info.Count));
											
										foreach (Grain ag in at.Grains)
											foreach (Grain bg in bt.Grains)
												if (Math.Abs(ag.Position.Z - bg.Position.Z) < grainztol)
												{
													overlapcount++;
													double l_dx, l_dy;
													t_dx += (l_dx = (ag.Position.X - bg.Position.X) + field_dx);
													t_dy += (l_dy = (ag.Position.Y - bg.Position.Y) + field_dy);
													t_dx2 += l_dx * l_dx;
													t_dy2 += l_dy * l_dy;
													break;
												}
										if (overlapcount >= minoverlapgrains)
										{
											dx += t_dx;
											dy += t_dy;
											dx2 += t_dx2;
											dy2 += t_dy2;
											totaloverlap += overlapcount;
											matches++;
											listofmatches.Add(new IntMatch(at, bt, f, g));
										}
									}
					}			
			shifts.Delta.X += dx;
			shifts.Delta.Y += dy;
			shifts.DeltaErrors.X += dx2;
			shifts.DeltaErrors.Y += dy2;
			shifts.MatchCount += matches;
			return totaloverlap;
			}
		}

		class IntView
		{
			public int ViewIndex;
			public IntSide Top, Bottom;

			#region View information
			public SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.TilePos Tile;
			#endregion

			public IntView(Fragment.View v, int viewindex, double mergepostol, double mergepostol2, double mergeslopetol2, double overlaptol, double minslope2, int mincount, bool retainalltracks)
			{
				ViewIndex = viewindex;
				Tile = v.Tile;
				Top = new IntSide(v.Top, true, mergepostol, mergepostol2, mergeslopetol2, overlaptol, minslope2, mincount, retainalltracks);
				Bottom = new IntSide(v.Bottom, false, mergepostol, mergepostol2, mergeslopetol2, overlaptol, minslope2, mincount, retainalltracks);
			}

			public bool GetShifts(IntView v, FieldShift.SideValue side, ref FieldShift shifts, double overlaptol, double matchtol, double slopetol, double grainsoverlapratio, double grainztol, int minmatches, double maxerror, ArrayList listofmatches)
			{
				ArrayList tempmatches = new ArrayList();
				ArrayList ttemp;				
				shifts.Side = side;
				shifts.Delta.X = 0.0f;
				shifts.Delta.Y = 0.0f;
				shifts.DeltaErrors.X = 0.0f;
				shifts.DeltaErrors.Y = 0.0f;
				shifts.MatchCount = 0;

				int totaloverlap = 0;
				if (side == FieldShift.SideValue.Top || side == FieldShift.SideValue.Both) 
				{
					totaloverlap += Top.GetShifts(v.Top, v.Tile.X > Tile.X, ref shifts, overlaptol, matchtol, slopetol, grainsoverlapratio, grainztol, out ttemp, this, v);
					tempmatches.AddRange(ttemp);
				}
				// se ha raggiunto 12 può non fare il bottom
				if (side == FieldShift.SideValue.Bottom || side == FieldShift.SideValue.Both) 
				{
					totaloverlap += Bottom.GetShifts(v.Bottom, v.Tile.X > Tile.X, ref shifts, overlaptol, matchtol, slopetol, grainsoverlapratio, grainztol, out ttemp, this, v);
					tempmatches.AddRange(ttemp);
				}

				if (shifts.MatchCount >= minmatches)
				{
					listofmatches.AddRange(tempmatches);
					shifts.Delta.X /= totaloverlap;
					shifts.Delta.Y /= totaloverlap;
					shifts.DeltaErrors.X = (double)Math.Sqrt((shifts.DeltaErrors.X - shifts.Delta.X * shifts.Delta.X * totaloverlap) / totaloverlap);
					shifts.DeltaErrors.Y = (double)Math.Sqrt((shifts.DeltaErrors.Y - shifts.Delta.Y * shifts.Delta.Y * totaloverlap) / totaloverlap);
					return Math.Abs(shifts.DeltaErrors.X) < maxerror && Math.Abs(shifts.DeltaErrors.Y) < maxerror;
				}
				return false;
			}
		}

		class IntFragment
		{
			public IntView [] Views;
			public SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.TilePos Min, Max;
			public IntView [,] ViewMatrix;            

			#region Fragment information
			public Identifier Id;
			public uint Index;
			public uint StartView;
			public SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.FragmentCoding CodingMode;			
			#endregion

			public IntFragment(Fragment f, double mergepostol, double mergepostol2, double mergeslopetol2, double overlaptol, double minslope2, int mincount, bool retainalltracks)
			{
				CodingMode = f.CodingMode;
				StartView = f.StartView;
				Index = f.Index;
				Id = f.Id;

				int i;
				Views = new IntView [f.Length];				
				if (Views.Length == 0)
				{
					Min.X = Max.X = 0;
					Min.Y = Max.Y = 0;
					ViewMatrix = new IntView[0, 0];
				}
				else
				{
					Min = Max = f[0].Tile;
					if (retainalltracks)
						for (i = 0; i < f.Length; i++)
						{
							Views[i] = new IntView(f[i], i, mergepostol, mergepostol2, mergeslopetol2, overlaptol, minslope2, mincount, true);
							if (Views[i].Tile.X < Min.X) Min.X = Views[i].Tile.X;
							else if (Views[i].Tile.X > Max.X) Max.X = Views[i].Tile.X;
							if (Views[i].Tile.Y < Min.Y) Min.Y = Views[i].Tile.Y;
							else if (Views[i].Tile.Y > Max.Y) Max.Y = Views[i].Tile.Y;
						}
					else
					{
						int j;
						for (i = 0; i < f.Length; i++)
						{
							if (f[i].Tile.X < Min.X) Min.X = f[i].Tile.X;
							else if (f[i].Tile.X > Max.X) Max.X = f[i].Tile.X;
							if (f[i].Tile.Y < Min.Y) Min.Y = f[i].Tile.Y;
							else if (f[i].Tile.Y > Max.Y) Max.Y = f[i].Tile.Y;
						}
						for (i = j = 0; i < f.Length; i++)
							if (f[i].Tile.X == Min.X || f[i].Tile.X == Max.X ||
								f[i].Tile.Y == Min.Y || f[i].Tile.Y == Max.Y)
							{
								Views[i] = new IntView(f[i], i, mergepostol, mergepostol2, mergeslopetol2, overlaptol, minslope2, mincount, false);
								j++;
							}
						IntView [] newviews = new IntView[j];
						for (i = j = 0; i < Views.Length; i++)
							if (Views[i] != null)
								newviews[j++] = Views[i];
						Views = newviews;
					}
					ViewMatrix = new IntView[Max.Y - Min.Y + 1, Max.X - Min.X + 1];
					foreach (IntView v in Views)
						if (v != null) 
							ViewMatrix[v.Tile.Y - Min.Y, v.Tile.X - Min.X] = v;
				}
			}
		}


//		private ArrayList Compute(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag, FieldShift.SideValue side, out IntMatch [] matches)
		private ArrayList Compute(IntFragment ifrag, FieldShift.SideValue side, out IntMatch [] matches)
		{
			ArrayList listofmatches = new ArrayList();

			ArrayList pairs = new ArrayList(2 * ifrag.Views.Length);
			int xsize, ysize;			
            double cdist;

			// ora ifrag è passato da fuori
			//IntFragment ifrag = new IntFragment(frag, C.MergePosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.OverlapTol, C.MinSlope * C.MinSlope, C.MinGrains, true);
			ysize = ifrag.ViewMatrix.GetLength(0);
			xsize = ifrag.ViewMatrix.GetLength(1);
			bool isright, isup;
			isup = (ifrag.Views[0].Tile.Y < ifrag.Views[ifrag.Views.Length - 1].Tile.Y);
			isright = (xsize <= 1 || ifrag.Views[1].Tile.X > ifrag.Views[0].Tile.X);

			foreach (IntView v in ifrag.Views)
			{
				IntView w;
				FieldShift tempshift = new FieldShift();
				if ((v.Tile.X + 1 - ifrag.Min.X) < xsize && (w = ifrag.ViewMatrix[v.Tile.Y - ifrag.Min.Y, v.Tile.X + 1 - ifrag.Min.X]) != null) 
					if (v.GetShifts(w, side, ref tempshift, C.OverlapTol, C.PosTol, C.SlopeTol, C.GrainsOverlapRatio, C.GrainZTol, C.MinMatches, C.MaxMatchError, listofmatches))
					{
						tempshift.FirstViewIndex = (uint)v.ViewIndex;
						tempshift.SecondViewIndex = (uint)w.ViewIndex;
						cdist = (side == FieldShift.SideValue.Top)? v.Top.Pos.X-w.Top.Pos.X:v.Bottom.Pos.X-w.Bottom.Pos.X;			
						if (side == FieldShift.SideValue.Both) cdist =(cdist + v.Top.Pos.X-w.Top.Pos.X)/2;
						pairs.Add(new LinearFragmentCorrectionWithHysteresis.FieldShift(tempshift, true, isright, isup, cdist));
					}
				if ((v.Tile.Y + 1 - ifrag.Min.Y) < ysize && (w = ifrag.ViewMatrix[v.Tile.Y + 1 - ifrag.Min.Y, v.Tile.X - ifrag.Min.X]) != null) 
					if (v.GetShifts(w, side, ref tempshift, C.OverlapTol, C.PosTol, C.SlopeTol, C.GrainsOverlapRatio, C.GrainZTol, C.MinMatches, C.MaxMatchError, listofmatches))
					{
						tempshift.FirstViewIndex = (uint)v.ViewIndex;
						tempshift.SecondViewIndex = (uint)w.ViewIndex;
						cdist = (side == FieldShift.SideValue.Top)? v.Top.Pos.Y-w.Top.Pos.Y:v.Bottom.Pos.Y-w.Bottom.Pos.Y;			
						if (side == FieldShift.SideValue.Both) cdist =(cdist + v.Top.Pos.Y-w.Top.Pos.Y)/2;
						pairs.Add(new LinearFragmentCorrectionWithHysteresis.FieldShift(tempshift, false, isright, isup, cdist));
					}
			}

			matches = (IntMatch [])listofmatches.ToArray(typeof(IntMatch));			

			return pairs;
		}

		#endregion

		#region Callback properties
		public dShouldStop ShouldStop
		{
			get
			{
				return intShouldStop;
			}
			set
			{
				intShouldStop = value;
			}
		}
	
		public dLoad Load
		{
			get
			{
				return intLoad;
			}
			set
			{
				intLoad = value;
			}
		}
		
		public dProgress Progress
		{
			get
			{
				return intProgress;
			}
			set
			{
				intProgress = value;
			}
		}

		public dFragmentComplete FragmentComplete
		{
			get
			{
				return intFragmentComplete;
			}
			set
			{
				intFragmentComplete = value;
			}
		}
		#endregion

		public void Test(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag)
		{/*
			IntMatch [] m;

			System.IO.StreamWriter s = new System.IO.StreamWriter(@"c:\usr\temp.txt", true);
			FieldShift [] shifts = (FieldShift [])Compute(frag, FieldShift.SideValue.Top, out m).ToArray(typeof(FieldShift));
			//s.WriteLine("Side\tFirst\tSecond\tMatches\tDX\tDY\tEDX\tEDY");
			foreach (FieldShift f in shifts)
				s.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}", frag.Index, (frag.Views[0].Tile.X > frag.Views[1].Tile.X) ? 1 : -1, (frag.Views[0].Tile.Y > frag.Views[frag.Views.Length - 1].Tile.Y) ? 1 : -1,
					frag.Views[f.FirstViewIndex].Tile.X - frag.Views[f.SecondViewIndex].Tile.X, frag.Views[f.FirstViewIndex].Tile.Y - frag.Views[f.SecondViewIndex].Tile.Y, 
					f.Side, f.FirstViewIndex, f.SecondViewIndex, f.MatchCount, f.Delta.X, f.Delta.Y, f.DeltaErrors.X, f.DeltaErrors.Y);
			s.Flush();
			s.Close();

			s = new System.IO.StreamWriter(@"c:\usr\tempm.txt", true);
			foreach (IntMatch i in m)
				s.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}", 
					frag.Index, i.FirstV.Tile.X - i.SecondV.Tile.X, i.FirstV.Tile.Y - i.SecondV.Tile.Y,
					i.FirstT.Info.Intercept.X, i.FirstT.Info.Intercept.Y, i.FirstT.Info.Slope.X, i.FirstT.Info.Slope.Y, 
					i.SecondT.Info.Intercept.X, i.SecondT.Info.Intercept.Y, i.SecondT.Info.Slope.X, i.SecondT.Info.Slope.Y);
			s.Flush();
			s.Close();*/
		}

		#region EMILIANO
		public void ComputeFragmentCorrection(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat, FieldShift.SideValue side, out FieldShift [] shifts, out FragmentCorrection corr)
		{
			ArrayList pairs = new ArrayList();
			IntMatch [] m;
			uint xsize = 0, ysize = 0, nxsize, nysize;
			bool ishoriz = false, isup = false, isright = false;
			double [,] A;
			double B;
			IntFragment ifrag;

			if (intLoad == null) throw new NoFragmentLoaderException();
			int i;
			for (i = 0; i < cat.Fragments;)
			{
				if (intShouldStop != null)
					if (intShouldStop())
					{
						shifts = null;
						corr = null;
						return;
					}
				if (intProgress != null) intProgress((double)i / (double)cat.Fragments * 100.0f);
				SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag = intLoad((uint)++i);
				CheckRectCoil(frag, out ifrag, out nxsize, out nysize, out ishoriz, out isup, out isright);
				if (xsize == 0 || ysize == 0)
				{
					xsize = nxsize;
					ysize = nysize;
				}
				else if ((xsize != 0 || ysize != 0) && (nxsize !=xsize || nysize!=ysize))
					throw new SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShiftException("All fragments in the catalog must have the same size");
				pairs.AddRange(Compute(ifrag, side, out m)); 
			}
			if (intProgress != null) intProgress(100.0f);

			LinearFragmentCorrectionWithHysteresis.FieldShift [] newshifts = (LinearFragmentCorrectionWithHysteresis.FieldShift [])pairs.ToArray(typeof(LinearFragmentCorrectionWithHysteresis.FieldShift));			
			
			HysteresisFit(ref newshifts, out A, out B, xsize, ysize);

			shifts = new FieldShift[newshifts.Length];
			for (i = 0; i < newshifts.Length; i++)
				shifts[i] = newshifts[i].FS;
			
			corr = new LinearFragmentCorrectionWithHysteresis(A, B, xsize, ishoriz, isup, C.IsStep);
		}

		private void CheckRectCoil(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag, 
								out IntFragment	ifrag , out uint xsize, out uint ysize,
								out bool ishoriz, out bool isup, out bool isright)
		{

			uint i;
			bool IsX= false, IsY=false;
			int coillen=2;			
			ifrag = new IntFragment(frag, C.MergePosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.PosTol, C.MinSlope * C.MinSlope, C.MinGrains, true);
			ysize = (uint)ifrag.ViewMatrix.GetLength(0);
			xsize = (uint)ifrag.ViewMatrix.GetLength(1);
			int nViews = ifrag.Views.Length;


			/* Controllo se la serpentina è rettangolare: ci potrebbe anche essere 
			 * una serpentina che varia al variare delle righe o una interrotta in mezzo ad una riga
			 * */
			if ((Math.Abs(ifrag.Views[0].Tile.Y-ifrag.Views[nViews-1].Tile.Y)== ysize) &&
				((ifrag.Views[0].Tile.X==ifrag.Views[nViews-1].Tile.X) ||
				(Math.Abs(ifrag.Views[0].Tile.X-ifrag.Views[nViews-1].Tile.X)== xsize)))
				throw new System.Exception("Not a rectangular scanning area!");
			//Controllo del senso di percorrenza e check sui primi due
			if (ifrag.Views[0].Tile.X==ifrag.Views[1].Tile.X) IsY=true;
			/* Non posso dare per scontato un campo affiancato lungo Y se non è
			 * affiancato lungo X perchè il primo movimento potrebbe essere in diagonale!
			 * */
			if (ifrag.Views[0].Tile.Y==ifrag.Views[1].Tile.Y) IsX=true;
			//se i primi due campi sono in diagonale allora non è una serpentina
			ishoriz=IsX;
			isright = IsX && (ifrag.Views[0].Tile.X<ifrag.Views[1].Tile.X);
			isup = IsX && (ifrag.Views[0].Tile.Y<ifrag.Views[nViews-1].Tile.Y);
			

			if (!IsX && !IsY) throw new System.Exception("Not a coil from first 2 fields sequence!");

			for (i=1 ; i< nViews-1; i++ )
			{
				if ((IsX &&(ifrag.Views[i].Tile.X!=ifrag.Views[i+1].Tile.X)) || 
					(IsY &&(ifrag.Views[i].Tile.Y!=ifrag.Views[i+1].Tile.Y)))
				{
					coillen++;
				}
				else
				{
					if ((IsX && (coillen!=xsize)) || (IsY && (coillen!=ysize)))
						throw new System.Exception("Not a coil from fields sequence!");
					coillen=1;
				};
			};
			if ((IsX && (coillen!=xsize)) || (IsY && (coillen!=ysize)))
				throw new System.Exception("Not a coil from fields sequence!");
			
		}

		private void HysteresisFit(ref LinearFragmentCorrectionWithHysteresis.FieldShift[] shifts, out double[,] Amat, out double XHyst, uint xsize, uint ysize)
		{
#if false
			int i,j,k, n; //, serp;
			bool MaxAchieved;
			double [] dst;
			double [] ifid1; //, ifid2;
			bool [] coil_x;
			ArrayList sid = new ArrayList();
			ArrayList mtc = new ArrayList();
			ArrayList if1 = new ArrayList();
			ArrayList if2 = new ArrayList();
			ArrayList dsx = new ArrayList();
			ArrayList dsxh = new ArrayList();
			ArrayList ch = new ArrayList();
			ArrayList dsy = new ArrayList();
			ArrayList dsx_e = new ArrayList();
			ArrayList dsy_e = new ArrayList();
			ArrayList coilx = new ArrayList();
			ArrayList coily = new ArrayList();
			ArrayList ishoriz = new ArrayList();
			double [] xm,xv,xnv;
			double dx=0.004;	
			double maxx=0, dum=0;
			double xinf=0, xsup=0;
			double avg=0;
			double rms=0;
			Amat = new double[2,2];
			XHyst=0;

			j=0;
			//Selezionare i left
			foreach(LinearFragmentCorrectionWithHysteresis.FieldShift s in shifts)
			{
				if(s.IsHorizontalMatch)
				{
					j++;
					dsx.Add((double)(s.FS.Delta.X/s.CenterDistance));
					dsxh.Add((double)s.FS.Delta.X);
					ch.Add((double)s.CenterDistance);
					dsy.Add((double)(s.FS.Delta.Y/s.CenterDistance));
					dsx_e.Add((double)s.FS.DeltaErrors.X);
					dsy_e.Add((double)s.FS.DeltaErrors.Y);
					coilx.Add(s.IsRightHeadingCoil);
					coily.Add(s.IsUpHeadingCoil);
					ishoriz.Add(s.IsHorizontalMatch);
					if1.Add((double)s.FS.FirstViewIndex);
					if2.Add((double)s.FS.SecondViewIndex);
					sid.Add(s.FS.Side);
					mtc.Add(s.FS.MatchCount);
				}
			};
			
			//dst = new double[j];
			//dst = new double[shifts.Length];
			//Scarto distrib gaussiana
			for(k=0; k<2; k++)
			{
				dst = (double [])((k == 0) ? dsx : dsy).ToArray(typeof(double));
				Fitting.Prepare_Custom_Distribution(dst, 1, dx, 0, out xm, out xv, out xnv);
				Fitting.FindStatistics(xv, ref maxx, ref dum, ref dum, ref dum);
				
				n = xv.GetLength(0);
				MaxAchieved= false;
				for (i=0 ;i<n; i++)
				{
					if (xv[i]==maxx) MaxAchieved= true;
					if(xv[i]>0.1*maxx && !MaxAchieved) 
					{
						xinf=xm[i]-dx/2;
						for (j = i+1; j<n; j++)
						{
							if (xv[j]==maxx) MaxAchieved= true;
							if(xv[j]<0.1*maxx && MaxAchieved)
							{
								xsup=xm[j]+dx/2;
								break;
							};
						
						};
						break;
					};  
				};

				n = dst.GetLength(0);
				for (i=n-1 ;i>=0; i--)
				{
					//Bisogna scartare anche le y
					if (dst[i] > xsup || dst[i]<xinf)
					{ 
						coilx.RemoveAt(i);
						coily.RemoveAt(i);
						ishoriz.RemoveAt(i);
						dsx.RemoveAt(i);
						dsxh.RemoveAt(i);
						ch.RemoveAt(i);
						dsy.RemoveAt(i);
						dsx_e.RemoveAt(i);
						dsy_e.RemoveAt(i);
						sid.RemoveAt(i);
						mtc.RemoveAt(i);
						if1.RemoveAt(i);
						if2.RemoveAt(i);
					};
				};
			};

			//Calcolo Coeff A(0,0) A(1,0) cioè Axx Ayx
			for(k=0; k<2; k++)
			{
				dst = (double [])((k == 0) ? dsx : dsy).ToArray(typeof(double));

				Fitting.FindStatistics(dst, ref dum, ref dum, ref Amat[k,0], ref rms);
			};

			//resetta
			coilx.Clear();
			coily.Clear();
			ishoriz.Clear();
			dsx.Clear();
			dsxh.Clear();
			ch.Clear();
			dsy.Clear();
			dsx_e.Clear();
			dsy_e.Clear();
			if1.Clear();
			if2.Clear();
			sid.Clear();
			mtc.Clear();

			//Selezionare i bottom
			foreach(LinearFragmentCorrectionWithHysteresis.FieldShift s in shifts)
			{
				if(s.IsHorizontalMatch == false)
				{
					coilx.Add(s.IsRightHeadingCoil);
					coily.Add(s.IsUpHeadingCoil);
					ishoriz.Add(s.IsHorizontalMatch);
					dsx.Add((double)(s.FS.Delta.X/s.CenterDistance));
					dsxh.Add((double)s.FS.Delta.X);
					ch.Add((double)s.CenterDistance);
					dsy.Add((double)(s.FS.Delta.Y/s.CenterDistance));
					dsx_e.Add((double)s.FS.DeltaErrors.X);
					dsy_e.Add((double)s.FS.DeltaErrors.Y);
					if1.Add((double)s.FS.FirstViewIndex);
					if2.Add((double)s.FS.SecondViewIndex);
					sid.Add(s.FS.Side);
					mtc.Add(s.FS.MatchCount);
				}
			};

			//Scarto distrib gaussiana
			dst = (double [])dsy.ToArray(typeof(double));
			Fitting.Prepare_Custom_Distribution(dst, 1, dx, 0, out xm, out xv, out xnv);
			Fitting.FindStatistics(xv, ref maxx, ref dum, ref dum, ref dum);
				
			n = xv.GetLength(0);
			MaxAchieved= false;
			for (i=0 ;i<n; i++)
			{
				if (xv[i]==maxx) MaxAchieved= true;
				if(xv[i]>0.1*maxx && !MaxAchieved) 
				{
					xinf=xm[i]-dx/2;
					for (j = i+1; j<n; j++)
					{
						if (xv[j]==maxx) MaxAchieved= true;
						if(xv[j]<0.1*maxx && MaxAchieved)
						{
							xsup=xm[j]+dx/2;
							break;
						};
						
					};
					break;
				};  
			};

			n = dst.GetLength(0);
			for (i=n-1 ;i>=0; i--)
			{
				//Bisogna scartare anche le y
				if (dst[i] > xsup || dst[i]<xinf)
				{ 
					coilx.RemoveAt(i);
					coily.RemoveAt(i);
					ishoriz.RemoveAt(i);
					dsx.RemoveAt(i);
					dsxh.RemoveAt(i);
					ch.RemoveAt(i);
					dsy.RemoveAt(i);
					dsx_e.RemoveAt(i);
					dsy_e.RemoveAt(i);
					sid.RemoveAt(i);
					mtc.RemoveAt(i);
					if1.RemoveAt(i);
					if2.RemoveAt(i);
				};
			};

			//Scarto distrib non gaussiana
			dst = (double [])dsx.ToArray(typeof(double));
			Fitting.FindStatistics(dst,  ref dum, ref dum, ref avg, ref rms);
			//Modificare col max e la distrib
			xsup=avg+3.0*rms;
			xinf=avg-3.0*rms;
			n = dst.GetLength(0);
			for (i=n-1 ;i>=0; i--)
			{
				//Bisogna scartare anche le y
				if(dst[i]>xsup || dst[i]<xinf) 
				{
					coilx.RemoveAt(i);
					coily.RemoveAt(i);
					ishoriz.RemoveAt(i);
					dsx.RemoveAt(i);
					dsxh.RemoveAt(i);
					ch.RemoveAt(i);
					dsy.RemoveAt(i);
					dsx_e.RemoveAt(i);
					dsy_e.RemoveAt(i);
					sid.RemoveAt(i);
					mtc.RemoveAt(i);
					if1.RemoveAt(i);
					if2.RemoveAt(i);
				};
			};

			//Calcolo Coeff A(1,1) cioè Ayy
			dst = (double [])dsy.ToArray(typeof(double));
			Fitting.FindStatistics(dst,  ref dum, ref dum, ref Amat[1,1], ref rms);

			//y = A +/- B*Sin(Pi*X/Lserp) 
			//Calcolo Trasl A=Sinvec(0) e B=Sinvec(1)
//			IntFragment ifrag = new IntFragment(frag, C.MergePosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.PosTol, C.MinSlope * C.MinSlope, C.MinGrains);
//			ysize = ifrag.ViewMatrix.GetLength(0);
//			xsize = ifrag.ViewMatrix.GetLength(1);


			dst = (double [])dsxh.ToArray(typeof(double));
			ifid1 = (double [])if1.ToArray(typeof(double));
			coil_x = (bool [])coilx.ToArray(typeof(bool));
			n = dst.GetLength(0);

			for (i=0 ;i<n; i++)
			{
				//serp = (frag.Views[0].Tile.X > frag.Views[frag.Views.Length - 1].Tile.X) ? 1 : -1;
				//ifid1[i]=serp*Math.Sin(ifid1[i]*Math.PI/xsize);
				ifid1[i]=(coil_x[i] ? 1 : -1) * (C.IsStep ? LinearFragmentCorrectionWithHysteresis.Step((int)ifid1[i], xsize) : Math.Sin(ifid1[i]*Math.PI/xsize));
			};
			/* 
			* 4) fit dei punti (contributo di rotazione: media ed entità di isteresi
			* y = A + B*Sin(Pi*X/Lserp) (+ C*X) X: numero campo (A=Mxy shift in X per campi affiancati lungo Y)
			*/
			Fitting.LinearFitSE(ifid1, dst, ref XHyst,ref Amat[0,1], ref dum, ref dum, ref dum, ref dum, ref dum);
			//Calcolo Coeff A(0,1) cioè Axy

			XHyst = -XHyst;
			Amat[0, 1] = 0.0;
			for (i = 0; i < n; i++)
				Amat[0, 1] += (dst[i] - XHyst * ifid1[i]) / (double)ch[i];
			Amat[0, 1] /= n;

			//Bisogna prima sottrarre la sistematica a dst
			// NOTE: this is only for debug, and can be removed to improve performance.
			for (i=0 ;i<n; i++)
			{
				//serp = (frag.Views[0].Tile.X > frag.Views[frag.Views.Length - 1].Tile.X) ? 1 : -1;
				//ifid1[i]=serp*Math.Sin(ifid1[i]*Math.PI/xsize);
				//ifid1[i]=(coil_x[i] ? 1 : -1) * (C.IsStep ? LinearFragmentCorrectionWithHysteresis.Step((int)ifid1[i], xsize) : Math.Sin(ifid1[i]*Math.PI/xsize));
				dst[i]=dst[i]-Amat[0,1] - XHyst*ifid1[i];
			};
			rms = Fitting.RMS(dst);
#else
			int i,j,k, n; //, serp;
			bool MaxAchieved;
			double [] dst;
			double [] ifid1; //, ifid2;
			bool [] coil_x;
			double dxh;
			ArrayList sid = new ArrayList();
			ArrayList mtc = new ArrayList();
			ArrayList if1 = new ArrayList();
			ArrayList if2 = new ArrayList();
			ArrayList dsx = new ArrayList();
			ArrayList dsxh = new ArrayList();
			ArrayList ch = new ArrayList();
			ArrayList dsy = new ArrayList();
			ArrayList dsx_e = new ArrayList();
			ArrayList dsy_e = new ArrayList();
			ArrayList coilx = new ArrayList();
			ArrayList coily = new ArrayList();
			ArrayList ishoriz = new ArrayList();
			double [] xm,xv,xnv;
			double dx=0.004;	
			double maxx=0, dum=0;
			double xinf=0, xsup=0;
			double avg=0;
			double rms=0;
			Amat = new double[2,2];
			XHyst=0;

			//Selezionare i bottom
			foreach(LinearFragmentCorrectionWithHysteresis.FieldShift s in shifts)
			{
				if(s.IsHorizontalMatch == false)
				{
					coilx.Add(s.IsRightHeadingCoil);
					coily.Add(s.IsUpHeadingCoil);
					ishoriz.Add(s.IsHorizontalMatch);
					dsx.Add((double)(s.FS.Delta.X/s.CenterDistance));
					dsxh.Add((double)s.FS.Delta.X);
					ch.Add((double)s.CenterDistance);
					dsy.Add((double)(s.FS.Delta.Y/s.CenterDistance));
					dsx_e.Add((double)s.FS.DeltaErrors.X);
					dsy_e.Add((double)s.FS.DeltaErrors.Y);
					if1.Add((double)s.FS.FirstViewIndex);
					if2.Add((double)s.FS.SecondViewIndex);
					sid.Add(s.FS.Side);
					mtc.Add(s.FS.MatchCount);
				}
			};

			//Scarto distrib gaussiana
			dst = (double [])dsy.ToArray(typeof(double));
			Fitting.Prepare_Custom_Distribution(dst, 1, dx, 0, out xm, out xv, out xnv);
			Fitting.FindStatistics(xv, ref maxx, ref dum, ref dum, ref dum);
				
			n = xv.GetLength(0);
			/*
			MaxAchieved= false;
			for (i=0; i<n; i++)
			{
				if (xv[i]==maxx) MaxAchieved= true;
				if(xv[i]>0.1*maxx && !MaxAchieved) 
				{
					xinf=xm[i]-dx/2;
					for (j = i+1; j<n; j++)
					{
						if (xv[j]==maxx) MaxAchieved= true;
						if(xv[j]<0.1*maxx && MaxAchieved)
						{
							xsup=xm[j]+dx/2;
							break;
						};
						
					};
					break;
				};  
			};
			*/
			for (i = 0; i < n && xv[i] < maxx * 0.1; i++);
			if (i < n)
			{
				xinf = xm[i] - dx/2;
				for (; i < n && xv[i] > maxx * 0.1; i++);
				if (i < n)
					xsup = xm[i - 1] + dx/2;
			}

			n = dst.GetLength(0);
			for (i=n-1 ;i>=0; i--)
			{
				//Bisogna scartare anche le y
				if (dst[i] > xsup || dst[i]<xinf)
				{ 
					coilx.RemoveAt(i);
					coily.RemoveAt(i);
					ishoriz.RemoveAt(i);
					dsx.RemoveAt(i);
					dsxh.RemoveAt(i);
					ch.RemoveAt(i);
					dsy.RemoveAt(i);
					dsx_e.RemoveAt(i);
					dsy_e.RemoveAt(i);
					sid.RemoveAt(i);
					mtc.RemoveAt(i);
					if1.RemoveAt(i);
					if2.RemoveAt(i);
				};
			};

			//Scarto distrib non gaussiana
			dst = (double [])dsx.ToArray(typeof(double));
			Fitting.FindStatistics(dst,  ref dum, ref dum, ref avg, ref rms);
			//Modificare col max e la distrib
			xsup=avg+3.0*rms;
			xinf=avg-3.0*rms;
			n = dst.GetLength(0);
			for (i=n-1 ;i>=0; i--)
			{
				//Bisogna scartare anche le y
				if(dst[i]>xsup || dst[i]<xinf) 
				{
					coilx.RemoveAt(i);
					coily.RemoveAt(i);
					ishoriz.RemoveAt(i);
					dsx.RemoveAt(i);
					dsxh.RemoveAt(i);
					ch.RemoveAt(i);
					dsy.RemoveAt(i);
					dsx_e.RemoveAt(i);
					dsy_e.RemoveAt(i);
					sid.RemoveAt(i);
					mtc.RemoveAt(i);
					if1.RemoveAt(i);
					if2.RemoveAt(i);
				};
			};

			//Calcolo Coeff A(1,1) cioè Ayy
			dst = (double [])dsy.ToArray(typeof(double));
			Fitting.FindStatistics(dst,  ref dum, ref dum, ref Amat[1,1], ref rms);

			//y = A +/- B*Sin(Pi*X/Lserp) 
			//Calcolo Trasl A=Sinvec(0) e B=Sinvec(1)
			//			IntFragment ifrag = new IntFragment(frag, C.MergePosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.PosTol, C.MinSlope * C.MinSlope, C.MinGrains);
			//			ysize = ifrag.ViewMatrix.GetLength(0);
			//			xsize = ifrag.ViewMatrix.GetLength(1);


			dst = (double [])dsxh.ToArray(typeof(double));
			ifid1 = (double [])if1.ToArray(typeof(double));
			coil_x = (bool [])coilx.ToArray(typeof(bool));
			n = dst.GetLength(0);

			for (i=0 ;i<n; i++)
			{
				//serp = (frag.Views[0].Tile.X > frag.Views[frag.Views.Length - 1].Tile.X) ? 1 : -1;
				//ifid1[i]=serp*Math.Sin(ifid1[i]*Math.PI/xsize);
				ifid1[i]=(coil_x[i] ? 1 : -1) * (C.IsStep ? LinearFragmentCorrectionWithHysteresis.Step((int)ifid1[i], xsize) : Math.Sin(ifid1[i]*Math.PI/xsize));
			};
			/* 
			* 4) fit dei punti (contributo di rotazione: media ed entità di isteresi
			* y = A + B*Sin(Pi*X/Lserp) (+ C*X) X: numero campo (A=Mxy shift in X per campi affiancati lungo Y)
			*/
			if (C.EnableHysteresis)
				Fitting.LinearFitSE(ifid1, dst, ref XHyst,ref Amat[0,1], ref dum, ref dum, ref dum, ref dum, ref dum);
			else XHyst = 0.0;
			//Calcolo Coeff A(0,1) cioè Axy

			XHyst = -XHyst;
			Amat[0, 1] = 0.0;
			for (i = 0; i < n; i++)
				Amat[0, 1] += (dst[i] - XHyst * ifid1[i]) / (double)ch[i];
			Amat[0, 1] /= n;

			//resetta
			coilx.Clear();
			coily.Clear();
			ishoriz.Clear();
			dsx.Clear();
			dsxh.Clear();
			ch.Clear();
			dsy.Clear();
			dsx_e.Clear();
			dsy_e.Clear();
			if1.Clear();
			if2.Clear();
			sid.Clear();
			mtc.Clear();

			//Selezionare i left
			foreach(LinearFragmentCorrectionWithHysteresis.FieldShift s in shifts)
			{
				if(s.IsHorizontalMatch)
				{

					if (C.IsStep)
						dxh = XHyst * LinearFragmentCorrectionWithHysteresis.Step((int)s.FS.FirstViewIndex, (uint)xsize);
					else
						dxh = XHyst * Math.Sin(s.FS.FirstViewIndex * Math.PI / xsize);

					/*
					{
						System.IO.StreamWriter fdeb = new System.IO.StreamWriter(@"d:\kryss\mio2.txt", true);
						fdeb.WriteLine("{0} {1} {2} {3} {4} {5}", s.IsRightHeadingCoil ? 1 : -1, s.IsUpHeadingCoil ? 1 : -1, s.FS.FirstViewIndex, dxh, s.FS.Delta.X, s.CenterDistance);
						fdeb.Flush();
						fdeb.Close();
					}
					*/

					dsx.Add((double)((s.FS.Delta.X + dxh) /s.CenterDistance));
					dsxh.Add((double)(s.FS.Delta.X + dxh));
					ch.Add((double)s.CenterDistance);
					dsy.Add((double)(s.FS.Delta.Y/s.CenterDistance));
					dsx_e.Add((double)s.FS.DeltaErrors.X);
					dsy_e.Add((double)s.FS.DeltaErrors.Y);
					coilx.Add(s.IsRightHeadingCoil);
					coily.Add(s.IsUpHeadingCoil);
					ishoriz.Add(s.IsHorizontalMatch);
					if1.Add((double)s.FS.FirstViewIndex);
					if2.Add((double)s.FS.SecondViewIndex);
					sid.Add(s.FS.Side);
					mtc.Add(s.FS.MatchCount);
				}
			};
			
			//dst = new double[j];
			//dst = new double[shifts.Length];
			//Scarto distrib gaussiana
			for(k=0; k<2; k++)
			{
				dst = (double [])((k == 0) ? dsx : dsy).ToArray(typeof(double));
				Fitting.Prepare_Custom_Distribution(dst, 1, dx, 0, out xm, out xv, out xnv);
				Fitting.FindStatistics(xv, ref maxx, ref dum, ref dum, ref dum);
				
				n = xv.GetLength(0);
				for (i = 0; i < n && xv[i] < maxx * 0.1; i++);
				if (i < n)
				{
					xinf = xm[i] - dx/2;
					for (; i < n && xv[i] > maxx * 0.1; i++);
					if (i < n)
						xsup = xm[i - 1] + dx/2;
				}

/*				MaxAchieved= false;
				for (i=0 ;i<n; i++)
				{
					if (xv[i]==maxx) MaxAchieved= true;
					if(xv[i]>0.1*maxx && !MaxAchieved) 
					{
						xinf=xm[i]-dx/2;
						for (j = i + 1; j < n; j++)
						{
							if (xv[j]==maxx) MaxAchieved= true;
							if(xv[j]<0.1*maxx && MaxAchieved)
							{
								xsup=xm[j]+dx/2;
								break;
							};
						
						};
						break;
					};  
				};
*/
				n = dst.GetLength(0);
				for (i=n-1 ;i>=0; i--)
				{
					//Bisogna scartare anche le y
					if (dst[i] > xsup || dst[i]<xinf)
					{ 
						coilx.RemoveAt(i);
						coily.RemoveAt(i);
						ishoriz.RemoveAt(i);
						dsx.RemoveAt(i);
						dsxh.RemoveAt(i);
						ch.RemoveAt(i);
						dsy.RemoveAt(i);
						dsx_e.RemoveAt(i);
						dsy_e.RemoveAt(i);
						sid.RemoveAt(i);
						mtc.RemoveAt(i);
						if1.RemoveAt(i);
						if2.RemoveAt(i);
					};
				};
			};

			//Calcolo Coeff A(0,0) A(1,0) cioè Axx Ayx
			for(k=0; k<2; k++)
			{
				dst = (double [])((k == 0) ? dsx : dsy).ToArray(typeof(double));

				Fitting.FindStatistics(dst, ref dum, ref dum, ref Amat[k,0], ref rms);
			};

#endif
		}
		#endregion

		#region CRISTIANO
		private void FieldShiftAverage(ArrayList shifts, out Vector2 AvgShift, out double Weight)
		{
			Vector2 LSum;
			double iWeight = 0;
			LSum.X = LSum.Y = 0;
			bool discarded;

			foreach (object obj in shifts)
			{
				Vector2 delta = ((FieldShift)obj).Delta;
				double wh = (double)((FieldShift)obj).MatchCount;
				iWeight += wh;
				LSum.X += wh * delta.X;
				LSum.Y += wh * delta.Y;
			}

			do
			{
				int i;
				discarded = false;
				for (i = 0; i < shifts.Count; i++)
				{
					Vector2 delta = ((FieldShift)shifts[i]).Delta;
					double wh = (double)((FieldShift)shifts[i]).MatchCount;
					Vector2 PLSum;
					double PWeight = iWeight - wh;

					if (PWeight > 0)
					{
						PLSum.X = LSum.X - wh * delta.X;
						PLSum.Y = LSum.Y - wh * delta.Y;
						Vector2 PAvg;
						PAvg.X = PLSum.X / PWeight;
						PAvg.Y = PLSum.Y / PWeight;
						if ((Math.Abs(delta.X - PAvg.X) > 2 * C.MaxMatchError) ||
							(Math.Abs(delta.Y - PAvg.Y) > 2 * C.MaxMatchError))
						{
							shifts.RemoveAt(i);
							discarded = true;
							LSum = PLSum;
							iWeight = PWeight;
							i--;
						}
					}
				}
			}
			while (discarded);
			LSum.X /= iWeight;
			LSum.Y /= iWeight;
			AvgShift = LSum;
			Weight = iWeight;
		}

		public void AdjustDisplacedFragments(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog cat, FragmentCorrection corr)
		{
			try
			{
				int i, j, index;
				int xsize, ysize;

				xsize = (int)cat.XSize;
				ysize = (int)cat.YSize;

				Fragment.View.TilePos [] tiledisp = new Fragment.View.TilePos[2];
				tiledisp[0].X = -1;
				tiledisp[0].Y = 0;
				tiledisp[1].X = 0;
				tiledisp[1].Y = -1;

				if (intLoad == null) throw new NoFragmentLoaderException();
				if (intFragmentComplete == null) throw new NoFragmentCompleteException();

				int [] fragmentorder = new int[cat.Fragments];
				{
					bool [] used = new bool[cat.Fragments];
					ArrayList [] dependencies = new ArrayList[cat.Fragments];
					for (i = 0; i < cat.Fragments; dependencies[i++] = new ArrayList());
					Fragment.View.TilePos t;
					for (t.Y = 1; t.Y < ysize; t.Y++)
						for (t.X = 1; t.X < xsize; t.X++)
							for (j = 0; j < 2; j++)
								if (cat[t.X, t.Y] != cat[t.X + tiledisp[j].X, t.Y + tiledisp[j].Y])
									dependencies[cat[t.X, t.Y] - 1].Add(cat[t.X + tiledisp[j].X, t.Y + tiledisp[j].Y]);

					for (i = 0; i < cat.Fragments; i++)
					{
						for (index = 0; used[index] || dependencies[index].Count > 0; index++);
						used[index++] = true;
						fragmentorder[i] = index;
						for (j = 0; j < cat.Fragments; j++)
							if (!used[j])
							{
								int k;
								for (k = 0; k < dependencies[j].Count; k++)
									if ((uint)dependencies[j][k] == index) 
										dependencies[j].RemoveAt(k--);
							}
					}
				}

				Fragment frag = null;
				IntFragment [] intfrags = new IntFragment[cat.Fragments];
				int [] usagecount = new int[cat.Fragments];

				if (intProgress != null) intProgress(0);

				for (j = 0; j < cat.Fragments; j++)
				{
					index = fragmentorder[j];
					i = index - 1;
					if (intShouldStop != null)
						if (intShouldStop()) return;
					frag = intLoad((uint)index);
					/*
					 * KRYSS: o si modifica la chiamata o si calcola tutto dentro
					 */
					if (corr != null) corr.Correct(frag);
					if (intfrags[i] == null)
					{
						intfrags[i] = new IntFragment(frag, C.MergePosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.OverlapTol, C.MinSlope * C.MinSlope, C.MinGrains, false);
						usagecount[i] = 0;
					}
					IntFragment intfrag = intfrags[i];
					bool [] usedfragment = new bool[cat.Fragments];
					ArrayList shifts = new ArrayList();
					ArrayList matches = new ArrayList();
					foreach (IntView v in intfrag.Views)
						foreach (Fragment.View.TilePos d in tiledisp)
						{
							Fragment.View.TilePos t;
							t.X = v.Tile.X + d.X;
							t.Y = v.Tile.Y + d.Y;
							if (t.X >= 0 && t.Y >= 0)
							{
								int index2 = (int)cat[t.X, t.Y];
								if (index2 == index) continue;
								if (intfrags[index2 - 1] == null)
								{
									Fragment tempfrag = intLoad((uint)index2);
									if (corr != null) corr.Correct(tempfrag);
									intfrags[index2 - 1] = new IntFragment(tempfrag, C.MergePosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.OverlapTol, C.MinSlope * C.MinSlope, C.MinGrains, false);
									usagecount[index2 - 1] = 0;
								}
								usedfragment[index2 - 1] = true;
								IntFragment intfrag2 = intfrags[index2 - 1];
								foreach (IntView w in intfrag2.Views)
									if (w.Tile.X == t.X && w.Tile.Y == t.Y)
									{
										FieldShift fs = new FieldShift();
										if (w.GetShifts(v, SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShift.SideValue.Both, ref fs, C.OverlapTol, C.PosTol, C.SlopeTol, C.GrainsOverlapRatio, C.GrainZTol, 1, C.MaxMatchError, matches))
											shifts.Add(fs);
										break;
									}
							}
						}

					Vector2 AvgShift;
					double Weight;
					FieldShiftAverage(shifts, out AvgShift, out Weight);

					if (Weight > (double)C.MinMatches)
					{
						AvgShift.X /= Weight;
						AvgShift.Y /= Weight;
						int vi;
						for (vi = 0; vi < frag.Length; vi++)
						{
							Vector2 p;
							Fragment.View v = frag[vi];
							p = v.Top.Pos;
							p.X += AvgShift.X;
							p.Y += AvgShift.Y;
							MySide.MySetPos(v.Top, p);
							p = v.Top.MapPos;
							p.X += (v.Top.MXX * AvgShift.X + v.Top.MXY * AvgShift.Y);
							p.Y += (v.Top.MYX * AvgShift.X + v.Top.MYY * AvgShift.Y);
							MySide.MySetMapPos(v.Top, p);
							p = v.Bottom.Pos;
							p.X += AvgShift.X;
							p.Y += AvgShift.Y;
							MySide.MySetPos(v.Bottom, p);
							p = v.Bottom.MapPos;
							p.X += (v.Bottom.MXX * AvgShift.X + v.Bottom.MXY * AvgShift.Y);
							p.Y += (v.Bottom.MYX * AvgShift.X + v.Bottom.MYY * AvgShift.Y);
							MySide.MySetMapPos(v.Bottom, p);
						}
						foreach (IntView v in intfrag.Views)
						{
							v.Top.Pos.X += AvgShift.X;
							v.Top.Pos.Y += AvgShift.Y;
							v.Top.MapPos.X += (v.Top.MXX * AvgShift.X + v.Top.MXY * AvgShift.Y);
							v.Top.MapPos.Y += (v.Top.MYX * AvgShift.X + v.Top.MYY * AvgShift.Y);
							
							v.Top.BottomMin.X += AvgShift.X;
							v.Top.BottomMax.X += AvgShift.X;
							v.Top.TopMin.X += AvgShift.X;
							v.Top.TopMax.X += AvgShift.X;
							v.Top.LeftMin.X += AvgShift.X;
							v.Top.LeftMax.X += AvgShift.X;
							v.Top.RightMin.X += AvgShift.X;
							v.Top.RightMax.X += AvgShift.X;

							v.Top.BottomMin.Y += AvgShift.Y;
							v.Top.BottomMax.Y += AvgShift.Y;
							v.Top.TopMin.Y += AvgShift.Y;
							v.Top.TopMax.Y += AvgShift.Y;
							v.Top.LeftMin.Y += AvgShift.Y;
							v.Top.LeftMax.Y += AvgShift.Y;
							v.Top.RightMin.Y += AvgShift.Y;
							v.Top.RightMax.Y += AvgShift.Y;
							
							v.Bottom.Pos.X += AvgShift.X;
							v.Bottom.Pos.Y += AvgShift.Y;
							v.Bottom.MapPos.X += (v.Bottom.MXX * AvgShift.X + v.Bottom.MXY * AvgShift.Y);
							v.Bottom.MapPos.Y += (v.Bottom.MYX * AvgShift.X + v.Bottom.MYY * AvgShift.Y);

							v.Bottom.BottomMin.X += AvgShift.X;
							v.Bottom.BottomMax.X += AvgShift.X;
							v.Bottom.TopMin.X += AvgShift.X;
							v.Bottom.TopMax.X += AvgShift.X;
							v.Bottom.LeftMin.X += AvgShift.X;
							v.Bottom.LeftMax.X += AvgShift.X;
							v.Bottom.RightMin.X += AvgShift.X;
							v.Bottom.RightMax.X += AvgShift.X;

							v.Bottom.BottomMin.Y += AvgShift.Y;
							v.Bottom.BottomMax.Y += AvgShift.Y;
							v.Bottom.TopMin.Y += AvgShift.Y;
							v.Bottom.TopMax.Y += AvgShift.Y;
							v.Bottom.LeftMin.Y += AvgShift.Y;
							v.Bottom.LeftMax.Y += AvgShift.Y;
							v.Bottom.RightMin.Y += AvgShift.Y;
							v.Bottom.RightMax.Y += AvgShift.Y;
						}
					}

					for (i = 0; i < cat.Fragments; i++)
						if (usedfragment[i])
							if (++usagecount[i] == 2) 
							{
								intfrags[i] = null;
								usagecount[i] = 0;
							}

					intFragmentComplete(frag);
					if (intProgress != null) intProgress((double)j / (double)cat.Fragments * 100.0f);
				}
			}
			catch (Exception x)
			{
				MessageBox.Show(x.ToString(), x.Message);
			}
			if (intProgress != null) intProgress(100.0f);
			GC.Collect();
		}
		#endregion

		#endregion

	}
}