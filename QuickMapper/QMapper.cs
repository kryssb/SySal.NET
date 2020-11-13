using System;
using System.Collections;
using System.Runtime.Serialization;
using SySal.BasicTypes;
using SySal.Management;
using SySal.Tracking;
using SySal.Scanning;
using SySal.Scanning.PostProcessing.PatternMatching;
using System.Xml.Serialization;

namespace SySal.Processing.QuickMapping
{
	/// <summary>
	/// QuickMapping configuration.
	/// </summary>
	[Serializable]
	[XmlType("QuickMapping.Configuration")]
	public class Configuration : SySal.Management.Configuration
	{
		/// <summary>
		/// Slope tolerance for track matching.
		/// </summary>
		public double SlopeTol;
		/// <summary>
		/// Position tolerance for track matching.
		/// </summary>
		public double PosTol;
		/// <summary>
		/// If <c>true</c>, absolute coordinates are used for mapping; otherwise, only relative track-to-track mapping matters.
		/// </summary>
		public bool UseAbsoluteReference;
		/// <summary>
		/// If <c>true</c>, full statistics is used and no statistical shortcut is applied to speed up the search.
		/// </summary>
		public bool FullStatistics;
		/// <summary>
		/// Builds an empty configuration.
		/// </summary>		
		public Configuration() : base("") {}
		/// <summary>
		/// Builds an empty configuration with the specified name.
		/// </summary>
		/// <param name="name">the name to be assigned to the configuration.</param>
		public Configuration(string name) : base(name) {}
		/// <summary>
		/// Clones the configuration.
		/// </summary>
		/// <returns>the object clone.</returns>
		public override object Clone()
		{
			Configuration c = new Configuration(Name);
			c.SlopeTol = SlopeTol;
			c.PosTol = PosTol;
			c.FullStatistics = FullStatistics;
			c.UseAbsoluteReference = UseAbsoluteReference;
			return c;
		}
	}

    /// <summary>
    /// Delegate called to log the mapping process.
    /// </summary>
    /// <param name="dx">Trial displacement in X.</param>
    /// <param name="dy">Trial displacement in Y.</param>
    /// <param name="fraction">Fraction of total sample used.</param>
    /// <param name="matches">Number of matches found.</param>
    public delegate void dMapLogger(double dx, double dy, double fraction, int matches);

	/// <summary>
	/// Quick pattern matching class.
	/// </summary>
	/// <remarks>
	/// <para>The QuickMapping algorithm takes two maps of tracks and searches for the translation that optimizes the number of matches.</para>
	/// <para>The algorithm works in relative coordinates; the origin of the mapping procedure is set by overlapping the centers of the two maps. 
	/// Therefore, absolute translations do not affect the ability of the algorithm to find the optimum matching conditions, but just the value of the translations found.</para>
	/// <para>This implementation works by dividing one map in cells to speed up the search for track matches.</para>
	/// <para>In order to speed up the search, <i>statistical boost</i> can be applied: the basic idea is that a background match has always much fewer tracks than the optimum match, 
	/// so it's useless to check for all tracks. In practice, after checking one quarter of the tracks, the number of matches found is compared with the number of matches obtained with the current best trial. 
	/// If it is worse (off a proper tolerance), that trial translation is given up, and the algorithm goes on with the next trial. The same comparison is done at one-half and at 
	/// 3/4 of the total track sample. After the good combination is found, all other combinations are very quickly discarded.</para>
	/// </remarks>
	[Serializable]
	[XmlType("QuickMapping.QuickMapper")]
	public class QuickMapper : IPatternMatcher, IManageable
	{
		[NonSerialized]
		private Configuration C;

		[NonSerialized]
		private string intName;

		[NonSerialized]
		private dShouldStop intShouldStop;

		[NonSerialized]
		private dProgress intProgress;

		[NonSerialized]
		private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        [NonSerialized]
        private dMapLogger intMapLogger;

		#region Management
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

		[XmlElement(typeof(QuickMapping.Configuration))]
		public SySal.Management.Configuration Config
		{
			get
			{
				return C;
			}
			set
			{
				C = (QuickMapping.Configuration)value;
			}
		}

		public bool EditConfiguration(ref SySal.Management.Configuration c)
		{
			bool ret;
			EditConfigForm myform = new EditConfigForm();
			myform.C = (QuickMapping.Configuration)c.Clone();
			if ((ret = (myform.ShowDialog() == System.Windows.Forms.DialogResult.OK))) c = myform.C;
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

		public QuickMapper()
		{
			C = new Configuration();
			C.SlopeTol = 0.02f;
			C.PosTol = 20.0f;
			intName = "Default Quick Mapper";
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

        public dMapLogger MapLogger
        {
            get
            {
                return intMapLogger;
            }
            set
            {
                intMapLogger = value;
            }
        }
		#endregion

		private struct TrackIndex
		{
			public int Index;
			public MIPEmulsionTrackInfo Info;

			public TrackIndex(int i, MIPEmulsionTrackInfo m)
			{
				Index = i;
				Info = m;
			}
		}

		private struct CellIndex
		{
			public int X, Y;
		}

		private TrackPair [] Match(MIPEmulsionTrackInfo [] first, MIPEmulsionTrackInfo [] second, double maxoffsetx, double maxoffsety, Vector2 minf, Vector2 maxf, Vector2 mins, Vector2 maxs)
		{
			int i;
			int bix, biy, ix, iy, iix, iiy, dix, diy, lix, liy;
			int bestix, bestiy;
			ArrayList BestList, NewList;			
			Vector2 fsize, ssize;
			fsize.X = maxf.X - minf.X;
			fsize.Y = maxf.Y - minf.Y;
			ssize.X = maxs.X - mins.X;
			ssize.Y = maxs.Y - mins.Y;
			if (C.UseAbsoluteReference == false)
			{
				if (maxoffsetx > (fsize.X + ssize.X)) maxoffsetx = fsize.X + ssize.X;
				if (maxoffsety > (fsize.Y + ssize.Y)) maxoffsety = fsize.Y + ssize.Y;
			}
			int sxcells, sycells;
			CellIndex [] FIndices = new CellIndex[first.Length];
			if (C.UseAbsoluteReference)
			{
				for (i = 0; i < first.Length; i++)
				{
					MIPEmulsionTrackInfo f = first[i];
					FIndices[i].X = (int)Math.Floor((f.Intercept.X - mins.X) / C.PosTol);
					FIndices[i].Y = (int)Math.Floor((f.Intercept.Y - mins.Y) / C.PosTol);				
				}
			}
			else
			{
				for (i = 0; i < first.Length; i++)
				{
					MIPEmulsionTrackInfo f = first[i];
					FIndices[i].X = (int)Math.Floor((f.Intercept.X - minf.X + (ssize.X - fsize.X) * 0.5) / C.PosTol);
					FIndices[i].Y = (int)Math.Floor((f.Intercept.Y - minf.Y + (ssize.Y - fsize.Y) * 0.5) / C.PosTol);				
				}
			}
			sxcells = (int)Math.Floor(ssize.X / C.PosTol + 1.0f);
			sycells = (int)Math.Floor(ssize.Y / C.PosTol + 1.0f);
			dix = (int)Math.Ceiling(maxoffsetx / C.PosTol);
			diy = (int)Math.Ceiling(maxoffsety / C.PosTol);
			BestList = new ArrayList();
			//Console.WriteLine("{0} tracks in first zone - {1} tracks in second zone - {2} cells - {3} trials", first.Length, second.Length, sycells * sxcells, (2 * dix + 1) * (2 * diy + 1));
			System.DateTime Start = System.DateTime.Now;
			const int statboost = 4;
			int [] checkpoints = new int[statboost];
			int k;

			ArrayList [,] TSCells = new ArrayList[sycells, sxcells];
			for (i = 0; i < second.Length; i++)
			{
				MIPEmulsionTrackInfo s = second[i];
				ix = (int)Math.Floor((s.Intercept.X - mins.X) / C.PosTol);
				iy = (int)Math.Floor((s.Intercept.Y - mins.Y) / C.PosTol);
/*				for (iiy = iy - 1; iiy <= iy + 1; iiy++)
					for (iix = ix - 1; iix <= ix + 1; iix++)
						if (iiy >= 0 && iiy < sycells && iix >= 0 && iix < sxcells)
						{
							if (TSCells[iiy, iix] == null) TSCells[iiy, iix] = new ArrayList();
							TSCells[iiy, iix].Add(new TrackIndex(i, s));				
						}
*/
				if (iy >= 0 && iy < sycells && ix >= 0 && ix < sxcells)
				{
					if (TSCells[iy, ix] == null) TSCells[iy, ix] = new ArrayList();
					TSCells[iy, ix].Add(new TrackIndex(i, s));				
				}

			}
			TrackIndex [,][] SCells = new TrackIndex[sycells, sxcells][];
			for (iy = 0; iy < sycells; iy++)
				for (ix = 0; ix < sxcells; ix++)
					SCells[iy, ix] = (TSCells[iy, ix] == null) ? new TrackIndex[0] : (TrackIndex [])TSCells[iy, ix].ToArray(typeof(TrackIndex));
			bestix = bestiy = -1;

			System.Random Rnd = new System.Random();
			int background = 0;
			for (i = 0; i < FIndices.Length; i++)
			{
				iix = Rnd.Next(sxcells);
				iiy = Rnd.Next(sycells);
				MIPEmulsionTrackInfo fi = first[i];
				foreach (TrackIndex sti in SCells[iiy, iix])
				{
					MIPEmulsionTrackInfo si = sti.Info;
					if (Math.Abs(fi.Slope.X - si.Slope.X) < C.SlopeTol && 
						Math.Abs(fi.Slope.Y - si.Slope.Y) < C.SlopeTol)
						background++;
				}
			}
			if (C.FullStatistics == false)
			{
				for (k = 0; k < statboost; k++)
				{
					checkpoints[k] = background * (k + 1) / statboost;
					checkpoints[k] -= 1 * (int)Math.Sqrt(checkpoints[k]);
					Console.WriteLine("Checkpoint {0}: {1}", k, checkpoints[k]);
				}
			}
			Console.WriteLine("Background: {0}", background);

			for (biy = -diy; biy <= diy; biy++)
			{
				if (intProgress != null)
					intProgress((diy > 0) ? ((biy + diy) / diy * 50.0f) : 0.0);
				for (bix = -dix; bix <= dix; bix++)				
				{
					if (intShouldStop != null && intShouldStop()) return null;
					NewList = new ArrayList();
					for (k = 0; k < statboost; k++)
					{
						for (i = k; i < FIndices.Length; i += statboost)
							//for (i = 0; i < FIndices.Length; i++)
						{
							iix = FIndices[i].X + bix;// if (iix < 0 || iix >= sxcells) continue;
							iiy = FIndices[i].Y + biy;// if (iiy < 0 || iiy >= sycells) continue;
							MIPEmulsionTrackInfo fi = first[i];
							for (liy = iiy - 1; liy <= iiy + 1; liy++)
								if (liy >= 0 && liy < sycells)
								{
									for (lix = iix - 1; lix <= iix + 1; lix++)
										if (lix >= 0 && lix < sxcells)
										{		
											TrackIndex [] s = SCells[liy, lix];							
											foreach (TrackIndex sti in s)
											{
												MIPEmulsionTrackInfo si = sti.Info;
												if (Math.Abs(fi.Slope.X - si.Slope.X) < C.SlopeTol && 
													Math.Abs(fi.Slope.Y - si.Slope.Y) < C.SlopeTol)
												{
													NewList.Add(new TrackPair(fi, i, si, sti.Index));
												}
											}
										}
								}
						}
						if (NewList.Count < checkpoints[k]) break;
					}
                    if (intMapLogger != null)
                        intMapLogger(bix * C.PosTol, biy * C.PosTol, ((double)(k + 1)) / statboost, NewList.Count);
					if (NewList.Count > BestList.Count) 
					{
						BestList = NewList;
						bestix = bix;
						bestiy = biy;
						for (k = 0; k < statboost; k++)
						{
							checkpoints[k] = BestList.Count * (k + 1) / statboost;
							checkpoints[k] -= 1 * (int)Math.Sqrt(checkpoints[k]);
						}
					}
				}
			}
			Console.WriteLine("Time elapsed: {0}", System.DateTime.Now - Start);
			return (TrackPair [])BestList.ToArray(typeof(TrackPair));
		}

		
		private TrackPair [] PrepareAndMatch(MIPEmulsionTrackInfo [] first, MIPEmulsionTrackInfo [] second, double maxoffsetx, double maxoffsety)
		{
			if (first.Length == 0 || second.Length == 0) throw new PatternMatchingException("Null pattern passed. Both track patterns must contain at least one track.");

			Vector2 MinFirst, MaxFirst, MinSec, MaxSec;
			MinFirst.X = MinFirst.Y = MaxFirst.X = MaxFirst.Y = MinSec.X = MinSec.Y = MaxSec.X = MaxSec.Y = 0.0f;
			int mapind;
			MIPEmulsionTrackInfo [] arr;
			int i;
			for (mapind = 0; mapind < 2; mapind++)
			{
				arr = (mapind == 0) ? first : second;
				Vector2 Min, Max;
				Min.X = Max.X = arr[0].Intercept.X;
				Min.Y = Max.Y = arr[0].Intercept.Y;
				for (i = 1; i < arr.Length; i++)
				{
					if (arr[i].Intercept.X < Min.X) Min.X = arr[i].Intercept.X;
					else if (arr[i].Intercept.X > Max.X) Max.X = arr[i].Intercept.X;
					if (arr[i].Intercept.Y < Min.Y) Min.Y = arr[i].Intercept.Y;
					else if (arr[i].Intercept.Y > Max.Y) Max.Y = arr[i].Intercept.Y;
				}
				if (mapind == 0)
				{
					MinFirst = Min;
					MaxFirst = Max;
				}
				else
				{
					MinSec = Min;
					MaxSec = Max;
				}
			};

			return Match(first, second, maxoffsetx, maxoffsety, MinFirst, MaxFirst, MinSec, MaxSec);
		}

		public TrackPair [] Match(MIPEmulsionTrackInfo [] moreprecisepattern, MIPEmulsionTrackInfo []  secondpattern, double Zprojection, double maxoffsetx, double maxoffsety)
		{
			MIPEmulsionTrackInfo [] proj = new MIPEmulsionTrackInfo[moreprecisepattern.Length];
			int i;
			for (i = 0; i < proj.Length; i++)
			{
				MIPEmulsionTrackInfo p = proj[i] = new MIPEmulsionTrackInfo();
				MIPEmulsionTrackInfo m = moreprecisepattern[i];
				p.AreaSum = m.AreaSum;
				p.Field = m.Field;
				p.Count = m.Count;
				p.TopZ = m.TopZ + Zprojection;
				p.BottomZ = m.BottomZ + Zprojection;
				p.Slope = m.Slope;
				p.Sigma = m.Sigma;
				p.Intercept.X = m.Intercept.X + Zprojection * p.Slope.X;
				p.Intercept.Y = m.Intercept.Y + Zprojection * p.Slope.Y;
				p.Intercept.Z = m.Intercept.Z + Zprojection;
			}
			TrackPair [] pairs = PrepareAndMatch(proj, secondpattern, maxoffsetx, maxoffsety);
			for (i = 0; i < pairs.Length; i++)
				pairs[i].First.Track = moreprecisepattern[pairs[i].First.Index];				
			return pairs;
		}

		public TrackPair [] Match(MIPEmulsionTrack [] moreprecisepattern, MIPEmulsionTrack []  secondpattern, double Zprojection, double maxoffsetx, double maxoffsety)
		{
			MIPEmulsionTrackInfo [] proj = new MIPEmulsionTrackInfo[moreprecisepattern.Length];
			int i;
			for (i = 0; i < proj.Length; i++)
			{
				MIPEmulsionTrackInfo p = proj[i] = new MIPEmulsionTrackInfo();
				MIPEmulsionTrackInfo m = moreprecisepattern[i].Info;
				p.AreaSum = m.AreaSum;
				p.Field = m.Field;
				p.Count = m.Count;
				p.TopZ = m.TopZ + Zprojection;
				p.BottomZ = m.BottomZ + Zprojection;
				p.Slope = m.Slope;
				p.Sigma = m.Sigma;
				p.Intercept.X = m.Intercept.X + Zprojection * p.Slope.X;
				p.Intercept.Y = m.Intercept.Y + Zprojection * p.Slope.Y;
				p.Intercept.Z = m.Intercept.Z + Zprojection;
			}
			MIPEmulsionTrackInfo [] fix = new MIPEmulsionTrackInfo[secondpattern.Length];
			for (i = 0; i < fix.Length; i++)
				fix[i] = secondpattern[i].Info;
			TrackPair [] pairs = PrepareAndMatch(proj, fix, maxoffsetx, maxoffsety);
			for (i = 0; i < pairs.Length; i++)
			{
				pairs[i].First.Track = moreprecisepattern[pairs[i].First.Index];
				pairs[i].Second.Track = moreprecisepattern[pairs[i].Second.Index];
			}
			return pairs;	
		}

		public TrackPair [] Match(MIPBaseTrack [] moreprecisepattern, MIPBaseTrack [] secondpattern, double Zprojection, double maxoffsetx, double maxoffsety)
		{
			MIPEmulsionTrackInfo [] proj = new MIPEmulsionTrackInfo[moreprecisepattern.Length];
			int i;
			for (i = 0; i < proj.Length; i++)
			{
				MIPEmulsionTrackInfo p = proj[i] = new MIPEmulsionTrackInfo();
				MIPEmulsionTrackInfo m = moreprecisepattern[i].Info;
				p.AreaSum = m.AreaSum;
				p.Field = m.Field;
				p.Count = m.Count;
				p.TopZ = m.TopZ + Zprojection;
				p.BottomZ = m.BottomZ + Zprojection;
				p.Slope = m.Slope;
				p.Sigma = m.Sigma;
				p.Intercept.X = m.Intercept.X + Zprojection * p.Slope.X;
				p.Intercept.Y = m.Intercept.Y + Zprojection * p.Slope.Y;
				p.Intercept.Z = m.Intercept.Z + Zprojection;
			}
			MIPEmulsionTrackInfo [] fix = new MIPEmulsionTrackInfo[secondpattern.Length];
			for (i = 0; i < fix.Length; i++)
				fix[i] = secondpattern[i].Info;
			TrackPair [] pairs = PrepareAndMatch(proj, fix, maxoffsetx, maxoffsety);
			for (i = 0; i < pairs.Length; i++)
			{
				pairs[i].First.Track = moreprecisepattern[pairs[i].First.Index];
				pairs[i].Second.Track = secondpattern[pairs[i].Second.Index];
			}
			return pairs;	
		}

	}
}
