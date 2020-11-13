using System;
using SySal.Scanning.Plate.IO.OPERA;
using SySal.Scanning.PostProcessing.PatternMatching;
using NumericalTools;
using SySal.Processing.QuickMapping;

namespace SySal.Executables.QMap
{
	class MIPEmulsionTrack : SySal.Tracking.MIPEmulsionTrack
	{
		public static SySal.Tracking.MIPEmulsionTrackInfo GetInfo(SySal.Tracking.MIPEmulsionTrack t) { return MIPEmulsionTrack.AccessInfo(t); }
	}

	class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
	{
		public static SySal.Tracking.MIPEmulsionTrackInfo GetInfo(SySal.Scanning.MIPBaseTrack t) { return MIPBaseTrack.AccessInfo(t); }

        public MIPBaseTrack(double px, double py, double sx, double sy)
        {
            m_Info = new SySal.Tracking.MIPEmulsionTrackInfo();
            m_Info.Intercept.X = px;
            m_Info.Intercept.Y = py;
            m_Info.Slope.X = sx;
            m_Info.Slope.Y = sy;
        }
	}

	class LinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
	{
		public static SySal.Scanning.MIPBaseTrack [] GetTracks(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz) 
		{ 
			int l = lz.Length;
			int i;
			SySal.Scanning.MIPBaseTrack [] tks = new SySal.Scanning.MIPBaseTrack[l];
			for (i = 0; i < l; i++)
				tks[i] = lz[i];
			return tks; 
		
		}
	}

	class Side : SySal.Scanning.Plate.Side
	{
		public static SySal.Tracking.MIPEmulsionTrack [] GetTracks(SySal.Scanning.Plate.Side s) { return Side.AccessTracks(s); }
	}

	/// <summary>
	/// QMap - Command line tool for pattern matching between Linked Zones.
	/// </summary>
	/// <remarks>
	/// <para>QMap uses <see cref="SySal.Processing.QuickMapping.QuickMapper">QuickMapper</see> to perform pattern matching between two maps of base-tracks from LinkedZones.</para>
	/// <para>One LinkedZone is called the <i>fixed</i> linked zone, whereas the other is the <i>projected</i> linked zone. The latter is actually projected along Z 
	/// (the longitudinal coordinate) and then pattern matching is performed. Optionally, the parameters of an affine transformation that optimizes the mapping can be computed, 
	/// and the projected map is rewritten after transformation.</para>
	/// <para>Usage: <c>QMap.exe &lt;projmap&gt; &lt;fixed&gt; &lt;output&gt; &lt;zproj&gt; &lt;slopetol&gt; &lt;postol&gt; &lt;maxoffset&gt; &lt;useabsolutereference&gt; &lt;fullstatistics&gt; [&lt;aligned tlg&gt;]</c></para>
	/// <para>Both the fixed map and the projected map are read from the <see cref="SySal.OperaPersistence">GUFS</see>, so they can be files as well as DB records.</para>
	/// <para>If <c>useabsolutereference</c> is <c>false</c>, the <c>maxoffset</c> is the maximum displacement between the maps in <b>relative</b> coordinates: even if the absolute coordinates are very different, 
	/// QMap internally represents the track patterns as if they had the same origin. However, the output differences are in the absolute reference frame. This behaviour automatically frees the user from rescaling the maps to the same origin.</para>
	/// <para>If <c>useabsolutereference</c> is <c>true</c>, QMap takes absolute distances into account.</para>
	/// <para>In order to speed up computations, if <c>fullstatistics</c> is <c>false</c>, only a subsample is evaluated at each step. In case of very low track densities, it is recommended to set <c>fullstatistics</c> to <c>true</c>, since the good solution might be missed because of a statistical fluctuation.</para>
	/// <para>The output ASCII file is made of n-tuples with a header. The column list with the meaning is shown below:
	/// <list type="table">
	/// <listheader><term>Name</term><description>Description</description></listheader>
	/// <item><term>PID</term><description>Zero-based Id of the base track in the projected map.</description></item>
	/// <item><term>PN</term><description>Number of grains of the base track in the projected map.</description></item>
	/// <item><term>PA</term><description>Area Sum of the base track in the projected map.</description></item>
	/// <item><term>PPX</term><description>Original X component of the position (before projection and mapping) of the base track in the projected map.</description></item>
	/// <item><term>PPY</term><description>Original Y component of the position (before projection and mapping) of the base track in the projected map.</description></item>
	/// <item><term>PSX</term><description>Original X component of the slope (before projection and mapping) of the base track in the projected map.</description></item>
	/// <item><term>PSY</term><description>Original Y component of the slope (before projection and mapping) of the base track in the projected map.</description></item>
	/// <item><term>PS</term><description>Original sigma of the base track in the projected map.</description></item>
	/// <item><term>FID</term><description>Zero-based Id of the base track in the fixed map.</description></item>
	/// <item><term>FN</term><description>Number of grains of the base track in the fixed map.</description></item>
	/// <item><term>FA</term><description>Area Sum of the base track in the fixed map.</description></item>
	/// <item><term>FPX</term><description>Original X component of the position (before projection and mapping) of the base track in the fixed map.</description></item>
	/// <item><term>FPY</term><description>Original Y component of the position (before projection and mapping) of the base track in the fixed map.</description></item>
	/// <item><term>FSX</term><description>Original X component of the slope (before projection and mapping) of the base track in the fixed map.</description></item>
	/// <item><term>FSY</term><description>Original Y component of the slope (before projection and mapping) of the base track in the fixed map.</description></item>
	/// <item><term>FS</term><description>Original sigma of the base track in the fixed map.</description></item>
	/// <item><term>DPX</term><description>X component of the position difference (projected - fixed, after projection) of the fixed base track and projected base track.</description></item>
	/// <item><term>DPY</term><description>Y component of the position difference (projected - fixed, after projection) of the fixed base track and projected base track.</description></item>
	/// <item><term>DSX</term><description>X component of the slope difference (projected - fixed) of the fixed base track and projected base track.</description></item>
	/// <item><term>DSY</term><description>Y component of the slope difference (projected - fixed) of the fixed base track and projected base track.</description></item>
	/// </list>
	/// The optional output TLG is in the <see cref="SySal.OperaPersistence">GUFS</see> as well as other paths. Optimization is only obtained by linear fits with no iterations, so it can be improved. This facility is only intended to be a rough hint of the real transformation to apply.
	/// </para>
	/// </remarks>
	public class QMapClass
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			if (args.Length != 9 && args.Length != 10 && args.Length != 11 && args.Length != 12)
			{
				Console.WriteLine("usage: qmap <projmap> <fixed> <output> <zproj> <slopetol> <postol> <maxoffset> <absolutereference> <fullstatistics> [/log <logfile>] [<aligned tlg>]");
                Console.WriteLine("map files can be TLG files or TXT files.");
                Console.WriteLine("In case of TXT files, the syntax to be used is the following:");
                Console.WriteLine("  px|py|sx|sy|myfile.txt");
                Console.WriteLine("  px,py,sx,sy are zero-based numbers identifying the column corresponding in each row of the TXT file to positions (x,y) and slope (x,y).");
				return;
			}
            
			float zproj, maxoffset;
			zproj = Convert.ToSingle(args[3], System.Globalization.CultureInfo.InvariantCulture);
			maxoffset = Convert.ToSingle(args[6], System.Globalization.CultureInfo.InvariantCulture);
			SySal.Processing.QuickMapping.Configuration C = new SySal.Processing.QuickMapping.Configuration();
			C.SlopeTol = Convert.ToSingle(args[4], System.Globalization.CultureInfo.InvariantCulture);
			C.PosTol = Convert.ToSingle(args[5], System.Globalization.CultureInfo.InvariantCulture);
			C.UseAbsoluteReference = Convert.ToBoolean(args[7]);
			C.FullStatistics = Convert.ToBoolean(args[8]);
			QuickMapper Q = new QuickMapper();
			Q.Progress = new dProgress(Progress);
			Q.Config = C;
            string logfile = null;
            if (args.Length >= 11 && string.Compare(args[9], "/log") == 0)
            {
                logfile = args[10];
                string[] argv1 = new string[args.Length - 2];
                int i;
                for (i = 0; i < argv1.Length; i++)
                    argv1[i] = args[i];
                argv1[argv1.Length - 1] = args[args.Length - 1];
                args = argv1;
                System.IO.File.WriteAllText(logfile, "DX\tDY\tFraction\tMatches");
                Q.MapLogger = new dMapLogger((double dx, double dy, double fract, int nmatches) => System.IO.File.AppendAllText(logfile, "\r\n" +
                    dx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + dy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                    fract.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" + nmatches));
            }
            SySal.OperaPersistence.LinkedZoneDetailLevel = SySal.OperaDb.Scanning.LinkedZone.DetailLevel.BaseGeom;
			SySal.Scanning.Plate.IO.OPERA.LinkedZone proj = null;
            SySal.Scanning.MIPBaseTrack[] ptracks = null;
            if (args[0].ToLower().EndsWith(".tlg"))
            {
                proj = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[0], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
                ptracks = LinkedZone.GetTracks(proj);
            }
            else ptracks = LoadTracksFromASCIIFile(args[0]);
            SySal.Scanning.Plate.IO.OPERA.LinkedZone second = null;
            SySal.Scanning.MIPBaseTrack[] stracks = null;
            if (args[1].ToLower().EndsWith(".tlg"))
            {
                second = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[1], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
                stracks = LinkedZone.GetTracks(second);
            }
            else stracks = LoadTracksFromASCIIFile(args[1]);
			TrackPair [] pairs = Q.Match(ptracks, stracks, zproj, maxoffset, maxoffset);
			System.IO.StreamWriter w = new System.IO.StreamWriter(args[2], false);
			w.WriteLine("PID\tPN\tPA\tPPX\tPPY\tPSX\tPSY\tPS\tFID\tFN\tFA\tFPX\tFPY\tFSX\tFSY\tFS\tDPX\tDPY\tDSX\tDSY");
			foreach (TrackPair p in pairs)
				w.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}",
					p.First.Index, p.First.Info.Count, p.First.Info.AreaSum, p.First.Info.Intercept.X, p.First.Info.Intercept.Y,
					p.First.Info.Slope.X, p.First.Info.Slope.Y, p.First.Info.Sigma,
					p.Second.Index, p.Second.Info.Count, p.Second.Info.AreaSum, p.Second.Info.Intercept.X, p.Second.Info.Intercept.Y,
					p.Second.Info.Slope.X, p.Second.Info.Slope.Y, p.Second.Info.Sigma,
					p.First.Info.Intercept.X - p.Second.Info.Intercept.X, p.First.Info.Intercept.Y - p.Second.Info.Intercept.Y, 
					p.First.Info.Slope.X - p.Second.Info.Slope.X, p.First.Info.Slope.Y - p.Second.Info.Slope.Y);
			w.Flush();
			w.Close();
			Console.WriteLine();
			Console.WriteLine("{0} matches found.", pairs.Length);
			if (args.Length == 10 && proj != null && second != null)
			{
				double [] xfit = new double[3];
				double [] yfit = new double[3];
				double ccorr = 0.0;
				double [,] Indeps = new double[2, pairs.Length];
				double [] Deps = new double[pairs.Length];
				int i;

				for (i = 0; i < pairs.Length; i++)
				{
					SySal.Scanning.MIPBaseTrack t = (SySal.Scanning.MIPBaseTrack)pairs[i].First.Track;
					Indeps[0, i] = t.Info.Intercept.X;
					Indeps[1, i] = t.Info.Intercept.Y;
					Deps[i] = pairs[i].Second.Info.Intercept.X - pairs[i].First.Info.Intercept.X;
				}
				Fitting.MultipleLinearRegression(Indeps, Deps, ref xfit, ref ccorr);

				for (i = 0; i < pairs.Length; i++)
				{
					SySal.Scanning.MIPBaseTrack t = (SySal.Scanning.MIPBaseTrack)pairs[i].First.Track;
					Deps[i] = pairs[i].Second.Info.Intercept.Y - pairs[i].First.Info.Intercept.Y;
				}
				Fitting.MultipleLinearRegression(Indeps, Deps, ref yfit, ref ccorr);

				Console.WriteLine("Dx={0} Mxx={1} Mxy={2}", xfit[0], xfit[1], xfit[2]);
				Console.WriteLine("Dy={0} Myx={1} Myy={2}", yfit[0], yfit[1], yfit[2]);

				xfit[1] += 1.0;
				yfit[2] += 1.0;

				float tempx;
				SySal.Scanning.MIPBaseTrack [] basetks = LinkedZone.GetTracks(proj);
				foreach (SySal.Scanning.MIPBaseTrack t in basetks)
				{
					SySal.Tracking.MIPEmulsionTrackInfo info = MIPBaseTrack.GetInfo(t);
					tempx = (float)(info.Intercept.X * xfit[1] + info.Intercept.Y * xfit[2] + xfit[0]);
					info.Intercept.Y = (float)(info.Intercept.X * yfit[1] + info.Intercept.Y * yfit[2] + yfit[0]);
					info.Intercept.X = tempx;
					tempx = (float)(info.Slope.X * xfit[1] + info.Slope.Y * xfit[2]);
					info.Slope.Y = (float)(info.Slope.X * yfit[1] + info.Slope.Y * yfit[2]);
					info.Slope.X = tempx;
				}
				SySal.Tracking.MIPEmulsionTrack [] toptks = Side.GetTracks(proj.Top);
				foreach (SySal.Tracking.MIPEmulsionTrack t in toptks)
				{
					SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.GetInfo(t);
					tempx = (float)(info.Intercept.X * xfit[1] + info.Intercept.Y * xfit[2] + xfit[0]);
					info.Intercept.Y = (float)(info.Intercept.X * yfit[1] + info.Intercept.Y * yfit[2] + yfit[0]);
					info.Intercept.X = tempx;
					tempx = (float)(info.Slope.X * xfit[1] + info.Slope.Y * xfit[2]);
					info.Slope.Y = (float)(info.Slope.X * yfit[1] + info.Slope.Y * yfit[2]);
					info.Slope.X = tempx;
				}
				SySal.Tracking.MIPEmulsionTrack [] bottomtks = Side.GetTracks(proj.Bottom);
				foreach (SySal.Tracking.MIPEmulsionTrack t in bottomtks)
				{
					SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.GetInfo(t);
					tempx = (float)(info.Intercept.X * xfit[1] + info.Intercept.Y * xfit[2] + xfit[0]);
					info.Intercept.Y = (float)(info.Intercept.X * yfit[1] + info.Intercept.Y * yfit[2] + yfit[0]);
					info.Intercept.X = tempx;
					tempx = (float)(info.Slope.X * xfit[1] + info.Slope.Y * xfit[2]);
					info.Slope.Y = (float)(info.Slope.X * yfit[1] + info.Slope.Y * yfit[2]);
					info.Slope.X = tempx;
				}
				Console.WriteLine("Results written to: " + SySal.OperaPersistence.Persist(args[9], proj));
			}
		}

		public static void Progress(double f)
		{
			Console.Write("#");
		}

        static SySal.Scanning.MIPBaseTrack[] LoadTracksFromASCIIFile(string fname)
        {
            string[] tokens = fname.Split('|');
            if (tokens.Length != 5) throw new Exception("ASCII file paths must have px|py|sx|sy|filepath syntax.");
            int pxcol = int.Parse(tokens[0]);
            int pycol = int.Parse(tokens[1]);
            int sxcol = int.Parse(tokens[2]);
            int sycol = int.Parse(tokens[3]);
            System.Collections.ArrayList tks_arr = new System.Collections.ArrayList();
            System.IO.StreamReader file = new System.IO.StreamReader(tokens[4]);
            string line;
            int allrows = 0, skippedrows = 0;
            while ((line = file.ReadLine()) != null)
            {
                allrows++;
                tokens = line.Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length < 4)
                {
                    skippedrows++;
                    continue;
                }
                try
                {
                    tks_arr.Add(
                        new MIPBaseTrack(
                            (pxcol >= 0 ? double.Parse(tokens[pxcol], System.Globalization.CultureInfo.InvariantCulture) : 0.0),
                            (pycol >= 0 ? double.Parse(tokens[pycol], System.Globalization.CultureInfo.InvariantCulture) : 0.0),
                            (sxcol >= 0 ? double.Parse(tokens[sxcol], System.Globalization.CultureInfo.InvariantCulture) : 0.0),
                            (sycol >= 0 ? double.Parse(tokens[sycol], System.Globalization.CultureInfo.InvariantCulture) : 0.0)
                        )
                );
                }
                catch (Exception)
                {
                    skippedrows++;
                }                
            }
            file.Close();
            Console.WriteLine("Read " + allrows + " lines, skipped " + skippedrows + " lines.");
            return (SySal.Scanning.MIPBaseTrack[])tks_arr.ToArray(typeof(SySal.Scanning.MIPBaseTrack));
        }
	}
}
