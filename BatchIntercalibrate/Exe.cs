using System;

namespace SySal.Executables.BatchIntercalibrate
{
	class MyTransformation
	{
		public static void Transform(double X, double Y, ref double tX, ref double tY)
		{
			tX = Transformation.MXX * (X - Transformation.RX) + Transformation.MXY * (Y - Transformation.RY) + Transformation.TX + Transformation.RX;
			tY = Transformation.MYX * (X - Transformation.RX) + Transformation.MYY * (Y - Transformation.RY) + Transformation.TY + Transformation.RY;
		}

		public static void Deform(double X, double Y, ref double dX, ref double dY)
		{
			dX = Transformation.MXX * X + Transformation.MXY * Y;
			dY = Transformation.MYX * X + Transformation.MYY * Y;
		}

		public static SySal.DAQSystem.Scanning.IntercalibrationInfo Transformation = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
	}

	class LinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
	{
		public class tMIPEmulsionTrack : SySal.Tracking.MIPEmulsionTrack
		{
			public static void ApplyTransformation(SySal.Tracking.MIPEmulsionTrack tk)
			{
				SySal.Tracking.MIPEmulsionTrackInfo info = tMIPEmulsionTrack.AccessInfo(tk);
				MyTransformation.Transform(info.Intercept.X, info.Intercept.Y, ref info.Intercept.X, ref info.Intercept.Y);
				MyTransformation.Deform(info.Slope.X, info.Slope.Y, ref info.Slope.X, ref info.Slope.Y);
				SySal.Tracking.Grain [] grains = tMIPEmulsionTrack.AccessGrains(tk);
				if (grains != null)
					foreach (SySal.Tracking.Grain g in grains)
						MyTransformation.Transform(g.Position.X, g.Position.Y, ref g.Position.X, ref g.Position.Y);
			}
		}								 

		public class tMIPBaseTrack : SySal.Scanning.MIPBaseTrack
		{
			public static void ApplyTransformation(SySal.Scanning.MIPBaseTrack tk)
			{
				SySal.Tracking.MIPEmulsionTrackInfo info = tMIPBaseTrack.AccessInfo(tk);
				MyTransformation.Transform(info.Intercept.X, info.Intercept.Y, ref info.Intercept.X, ref info.Intercept.Y);
				MyTransformation.Deform(info.Slope.X, info.Slope.Y, ref info.Slope.X, ref info.Slope.Y);

			}
		}

		public class Side : SySal.Scanning.Plate.Side
		{
			public static SySal.Tracking.MIPEmulsionTrack [] GetTracks(SySal.Scanning.Plate.Side s) { return Side.AccessTracks(s); }
		}

		public LinkedZone(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz)
		{
			MyTransformation.Transform(lz.Extents.MinX, lz.Extents.MinY, ref m_Extents.MinX, ref m_Extents.MinY);
			MyTransformation.Transform(lz.Extents.MaxX, lz.Extents.MaxY, ref m_Extents.MaxX, ref m_Extents.MaxY);
			MyTransformation.Transform(lz.Center.X, lz.Center.Y, ref m_Center.X, ref m_Center.Y);
			m_Id = lz.Id;
			m_Tracks = LinkedZone.AccessTracks(lz);
			m_Top = lz.Top;
			m_Bottom = lz.Bottom;
			foreach (SySal.Scanning.MIPBaseTrack btk in m_Tracks)
				tMIPBaseTrack.ApplyTransformation(btk);
			SySal.Tracking.MIPEmulsionTrack [] mutks;
			mutks = BatchIntercalibrate.LinkedZone.Side.GetTracks(m_Top);
			foreach (SySal.Scanning.MIPIndexedEmulsionTrack mutk in mutks)
				tMIPEmulsionTrack.ApplyTransformation(mutk);
			mutks = BatchIntercalibrate.LinkedZone.Side.GetTracks(m_Bottom);
			foreach (SySal.Scanning.MIPIndexedEmulsionTrack mutk in mutks)
				tMIPEmulsionTrack.ApplyTransformation(mutk);
		}

	}


	/// <summary>
	/// BatchIntercalibrate - performs intercalibration of two plates by one or three reference zones.	
	/// </summary>
	/// <remarks>
	/// <para>The number of zones is extracted from the mode parameter.</para>
	/// <para>usage (1 zone): <c>BatchIntercalibrate.exe &lt;mode&gt; &lt;outmap&gt; &lt;refzone&gt; &lt;calibzone&gt; &lt;zproj&gt; &lt;postol&gt; &lt;slopetol&gt; &lt;maxoffset&gt; &lt;leverarm&gt; &lt;minmatches&gt; [&lt;refcenterX&gt; &lt;refcenterY&gt;]</c></para>
	/// <para>usage (3 zones): <c>BatchIntercalibrate &lt;mode&gt; &lt;outmap&gt; &lt;refzone1&gt; &lt;calibzone1&gt; &lt;refzone2&gt; &lt;calibzone2&gt; &lt;refzone3&gt; &lt;calibzone3&gt; &lt;zproj&gt; &lt;postol&gt; &lt;slopetol&gt; &lt;maxoffset&gt; &lt;minmatches&gt; [&lt;refcenterX&gt; &lt;refcenterY&gt;]</c></para>
	/// <para>
	/// Available modes:
	/// <list type="table">
	/// <listheader><term>Code</term><description>Meaning</description></listheader>
	/// <item><term><c>1</c></term><description>1 zone, just compute calibration.</description></item>
	/// <item><term><c>1r</c></term><description>1 zone, compute calibration and rewrite the zone file using the transformation found.</description></item>
	/// <item><term><c>3</c></term><description>3 zones, just compute calibration.</description></item>
	/// <item><term><c>3r</c></term><description>3 zones, compute calibration and rewrite the zone files using the transformation found.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// Meaning of the other parameters:
	/// <list type="table">
	/// <listheader><term>Parameter</term><description>Meaning</description></listheader>
	/// <item><term><c>outmap</c></term><description>path to the output file (map file in text format)</description></item>
	/// <item><term><c>refzone(#)</c></term><description>Opera persistence path of the #th linkedzone with the reference zone</description></item>
	/// <item><term><c>calibzone(#)</c></term><description>Opera persistence path of the #th linkedzone on the sheet to be calibrated</description></item>
	/// <item><term><c>zproj</c></term><description>Z of the sheet to be calibrated - Z of the reference sheet</description></item>
	/// <item><term><c>postol</c></term><description>position tolerance for pattern matching</description></item>
	/// <item><term><c>slopetol</c></term><description>slope tolerance for pattern matching</description></item>
	/// <item><term><c>leverarm</c></term><description>maximum expected variation of position match due to rotation (1-zone version only)</description></item>
	/// <item><term><c>maxoffset</c></term><description>maximum offset</description></item>	
	/// <item><term><c>minmatches</c></term><description>minimum number of matching tracks</description></item>	
	/// <item><term><c>refcenterX</c></term><description>(optional) X coordinate of the reference center</description></item>	
	/// <item><term><c>refcenterY</c></term><description>(optional) Y coordinate of the reference center</description></item>	
	/// </list>
	/// BatchIntercalibrate uses QuickMapper for pattern matching computations. See <see cref="SySal.Processing.QuickMapping.QuickMapper"/> for more information.
	/// </para>
	/// </remarks>
	public class Exe
	{
        /// <summary>
        /// Checks the intercalibration.
        /// </summary>
        /// <param name="info">the intercalibration to be checked.</param>
		public static void CheckIntercalibration(SySal.DAQSystem.Scanning.IntercalibrationInfo info)
		{
			double det = info.MXX * info.MYY - info.MXY * info.MYX;
			Console.WriteLine();
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Scanning.IntercalibrationInfo));
			xmls.Serialize(Console.Out, info);
//			if (Math.Abs(det - 1.0) > 0.1) throw new Exception("Transformation is a reflection or requires excessive contraction/expansion.");
//			if (Math.Abs(info.MXY + info.MYX) / det > 0.005) throw new Exception("Transformation has excessive shear.");
		}

		private static void ShowUsage()
		{
			Console.WriteLine();
			Console.WriteLine("BatchIntercalibrate - performs intercalibration of two plates by one or three reference zones.");
			Console.WriteLine("The number of zones is extracted from the mode parameter.");
			Console.WriteLine("usage (1 zone): BatchIntercalibrate <mode> <outmap> <refzone> <calibzone> <zproj> <postol> <slopetol> <maxoffset> <leverarm> <minmatches> [<refcenterX> <refcenterY>]");
			Console.WriteLine("usage (3 zones): BatchIntercalibrate <mode> <outmap> <refzone1> <calibzone1> <refzone2> <calibzone2> <refzone3> <calibzone3> <zproj> <postol> <slopetol> <maxoffset> <minmatches> [<refcenterX> <refcenterY>]");
			Console.WriteLine("mode = 1 -> 1 zone");
			Console.WriteLine("mode = 1r -> 1 zone, zone file is rewritten");
			Console.WriteLine("mode = 3 -> 3 zones");
			Console.WriteLine("mode = 3r -> 3 zones, zone files are rewritten");
			Console.WriteLine("outmap = path to the output file (map file in text format)");
			Console.WriteLine("refzone(#) = Opera persistence path of the #th linkedzone with the reference zone");
			Console.WriteLine("calibzone(#) = Opera persistence path of the #th linkedzone on the sheet to be calibrated");
			Console.WriteLine("zproj = Z of the sheet to be calibrated - Z of the reference sheet");
			Console.WriteLine("postol = position tolerance for pattern matching");
			Console.WriteLine("slopetol = slope tolerance for pattern matching");
			Console.WriteLine("leverarm = maximum expected variation of position match due to rotation (1-zone version only)");
			Console.WriteLine("maxoffset = maximum offset");
			Console.WriteLine("minmatches = minimum number of matching tracks");
			Console.WriteLine("refcenterX = (optional) X coordinate of the reference center");
			Console.WriteLine("refcenterY = (optional) Y coordinate of the reference center");
		}

		private struct MapPos
		{
			public double X, Y;
			public double DX, DY;

			public void SetFromPairs(SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs)
			{
				X = Y = DX = DY = 0;
				foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair p in pairs)
				{
					X += ((SySal.Tracking.MIPEmulsionTrackInfo)(p.Second.Track)).Intercept.X;
					Y += ((SySal.Tracking.MIPEmulsionTrackInfo)(p.Second.Track)).Intercept.Y;
					DX += p.First.Info.Intercept.X - p.Second.Info.Intercept.X;
					DY += p.First.Info.Intercept.Y - p.Second.Info.Intercept.Y;
				}
				X /= pairs.Length;
				Y /= pairs.Length;
				DX /= pairs.Length;
				DY /= pairs.Length;					
			}
		}

		private static void ThreeZoneIntercalibration(string [] args, ref SySal.DAQSystem.Scanning.IntercalibrationInfo intercal, bool rewrite)
		{
			int i, n;
			int MinMatches = Convert.ToInt32(args[11]);
			SySal.Processing.QuickMapping.QuickMapper QMap = new SySal.Processing.QuickMapping.QuickMapper();
			SySal.Processing.QuickMapping.Configuration C = (SySal.Processing.QuickMapping.Configuration)QMap.Config;
			C.PosTol = Convert.ToDouble(args[8], System.Globalization.CultureInfo.InvariantCulture);
			C.SlopeTol = Convert.ToDouble(args[9], System.Globalization.CultureInfo.InvariantCulture);
			QMap.Config = C;
			double ZProj = Convert.ToDouble(args[7], System.Globalization.CultureInfo.InvariantCulture);
			double maxoffset = Convert.ToDouble(args[10], System.Globalization.CultureInfo.InvariantCulture);
			MapPos [] mappositions = new MapPos[3];

			SySal.Scanning.Plate.IO.OPERA.LinkedZone [] savezones = new SySal.Scanning.Plate.IO.OPERA.LinkedZone[3];
			for (n = 0; n < 3; n++)
			{
				SySal.Scanning.Plate.IO.OPERA.LinkedZone reflz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[1 + n * 2], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
				SySal.Tracking.MIPEmulsionTrackInfo [] refzone = new SySal.Tracking.MIPEmulsionTrackInfo[reflz.Length];
				for (i = 0; i < refzone.Length; i++)
					refzone[i] = reflz[i].Info;
				reflz = null;
				SySal.Scanning.Plate.IO.OPERA.LinkedZone caliblz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[2 + n * 2], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
				if (rewrite) savezones[n] = caliblz;
				SySal.Tracking.MIPEmulsionTrackInfo [] calibzone = new SySal.Tracking.MIPEmulsionTrackInfo[caliblz.Length];
				for (i = 0; i < calibzone.Length; i++)
					calibzone[i] = caliblz[i].Info;
				caliblz = null;
				GC.Collect();
				Console.Write("Zone #" + n.ToString() + " ");
				SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs = QMap.Match(refzone, calibzone, ZProj, maxoffset, maxoffset);
				if (pairs.Length < MinMatches)
				{
					Console.Error.WriteLine("Too few matching tracks: " + MinMatches.ToString() + " required, " + pairs.Length + " obtained. Aborting.");
					return;
				}
				Console.WriteLine("Matches: " + pairs.Length);
				mappositions[n].SetFromPairs(pairs); 
				mappositions[n].X -= intercal.RX;
				mappositions[n].Y -= intercal.RY;
			}
			double x20 = mappositions[2].X - mappositions[0].X;
			double x10 = mappositions[1].X - mappositions[0].X;
			double y20 = mappositions[2].Y - mappositions[0].Y;
			double y10 = mappositions[1].Y - mappositions[0].Y;
			double det = 1.0 / (x10 * y20 - x20 * y10);
			double u20 = mappositions[2].DX - mappositions[0].DX;
			double v20 = mappositions[2].DY - mappositions[0].DY;
			double u10 = mappositions[1].DX - mappositions[0].DX;
			double v10 = mappositions[1].DY - mappositions[0].DY;
			intercal.MXX = (u10 * y20 - u20 * y10) * det;
			intercal.MXY = (u20 * x10 - u10 * x20) * det;
			intercal.MYX = (v10 * y20 - v20 * y10) * det;
			intercal.MYY = (v20 * x10 - v10 * x20) * det;
			intercal.TX = mappositions[0].DX - intercal.MXX * mappositions[0].X - intercal.MXY * mappositions[0].Y;
			intercal.TY = mappositions[0].DY - intercal.MYX * mappositions[0].X - intercal.MYY * mappositions[0].Y;
			intercal.MXX += 1.0;
			intercal.MYY += 1.0;
			CheckIntercalibration(intercal);
			if (rewrite)
			{
				MyTransformation.Transformation = intercal;
				for (n = 0; n < 3; n++)
					SySal.OperaPersistence.Persist(args[2 + n * 2], new LinkedZone(savezones[n]));
			}
		}
			
			
		private static void OneZoneIntercalibration(string [] args, ref SySal.DAQSystem.Scanning.IntercalibrationInfo intercal, bool rewrite)
		{
			int i, j, n;
			int MinMatches = Convert.ToInt32(args[8]);
			SySal.Scanning.Plate.IO.OPERA.LinkedZone reflz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[1], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
			SySal.Tracking.MIPEmulsionTrackInfo [] refzone = new SySal.Tracking.MIPEmulsionTrackInfo[reflz.Length];
			for (i = 0; i < refzone.Length; i++)
				refzone[i] = reflz[i].Info;
			reflz = null;
			GC.Collect();
			SySal.Scanning.Plate.IO.OPERA.LinkedZone caliblz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[2], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
			SySal.Tracking.MIPEmulsionTrackInfo [] calibzone = new SySal.Tracking.MIPEmulsionTrackInfo[caliblz.Length];
			for (i = 0; i < calibzone.Length; i++)
				calibzone[i] = caliblz[i].Info;
			if (!rewrite) caliblz = null;
			GC.Collect();
			SySal.Processing.QuickMapping.QuickMapper QMap = new SySal.Processing.QuickMapping.QuickMapper();
			SySal.Processing.QuickMapping.Configuration C = (SySal.Processing.QuickMapping.Configuration)QMap.Config;
			double postol = Convert.ToDouble(args[4], System.Globalization.CultureInfo.InvariantCulture);
			double leverarm = Convert.ToDouble(args[6], System.Globalization.CultureInfo.InvariantCulture);
			C.PosTol = postol + leverarm;
			C.SlopeTol = Convert.ToDouble(args[5], System.Globalization.CultureInfo.InvariantCulture);
			QMap.Config = C;
			double ZProj = Convert.ToDouble(args[3], System.Globalization.CultureInfo.InvariantCulture);
			double maxoffset = Convert.ToDouble(args[7], System.Globalization.CultureInfo.InvariantCulture);
			SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs = QMap.Match(refzone, calibzone, ZProj, maxoffset, maxoffset); 
			if (pairs.Length < MinMatches)
			{
				Console.Error.WriteLine("Too few matching tracks: " + MinMatches.ToString() + " required, " + pairs.Length + " obtained. Aborting.");
				return;
			}
			Console.WriteLine("Matches: " + pairs.Length);

			n = pairs.Length;
			double [,] indep = new double[2, n];
			double [] dep = new double[n];
			double [] res = new double[3];
			double dummy = 0.0;
			for (i = 0; i < n; i++)
			{
				indep[0, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(pairs[i].Second.Track)).Intercept.X - intercal.RX;
				indep[1, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(pairs[i].Second.Track)).Intercept.Y - intercal.RY;
				dep[i] = pairs[i].First.Info.Intercept.X - intercal.RX;
			}
			NumericalTools.Fitting.MultipleLinearRegression(indep, dep, ref res, ref dummy);
			Console.WriteLine("{0}\t{1}\t{2}\t{3}", res[0], res[1], res[2], dummy);
			intercal.TX = res[0]; intercal.MXX = res[1]; intercal.MXY = res[2];
			for (i = 0; i < n; i++)
			{
				indep[0, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(pairs[i].Second.Track)).Intercept.X - intercal.RX;
				indep[1, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(pairs[i].Second.Track)).Intercept.Y - intercal.RY;
				dep[i] = pairs[i].First.Info.Intercept.Y - intercal.RY;
			}
			NumericalTools.Fitting.MultipleLinearRegression(indep, dep, ref res, ref dummy);
			Console.WriteLine("{0}\t{1}\t{2}\t{3}", res[0], res[1], res[2], dummy);
			intercal.TY = res[0]; intercal.MYX = res[1]; intercal.MYY = res[2];
			
			System.Collections.ArrayList a_goodpairs = new System.Collections.ArrayList();
			foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair p in pairs)
			{
				dummy = Math.Abs(intercal.TX + intercal.MXX * (p.Second.Info.Intercept.X - intercal.RX) + intercal.MXY * (p.Second.Info.Intercept.Y - intercal.RY) - p.First.Info.Intercept.X + intercal.RX);
				if (dummy > postol) continue;
				dummy = Math.Abs(intercal.TY + intercal.MYX * (p.Second.Info.Intercept.X - intercal.RX) + intercal.MYY * (p.Second.Info.Intercept.Y - intercal.RY) - p.First.Info.Intercept.Y + intercal.RY);
				if (dummy > postol) continue;
				a_goodpairs.Add(p);
			}
			Console.WriteLine("remaining " + a_goodpairs.Count);

			SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] goodpairs = (SySal.Scanning.PostProcessing.PatternMatching.TrackPair [])a_goodpairs.ToArray(typeof(SySal.Scanning.PostProcessing.PatternMatching.TrackPair));
			n = goodpairs.Length;
			indep = new double[2, n];
			dep = new double[n];
			for (i = 0; i < n; i++)
			{
				indep[0, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(goodpairs[i].Second.Track)).Intercept.X - intercal.RX;
				indep[1, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(goodpairs[i].Second.Track)).Intercept.Y - intercal.RY;
				dep[i] = goodpairs[i].First.Info.Intercept.X - intercal.RX;
			}
			NumericalTools.Fitting.MultipleLinearRegression(indep, dep, ref res, ref dummy);
			Console.WriteLine("{0}\t{1}\t{2}\t{3}", res[0], res[1], res[2], dummy);
			intercal.TX = res[0]; intercal.MXX = res[1]; intercal.MXY = res[2];
			for (i = 0; i < n; i++)
			{
				indep[0, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(goodpairs[i].Second.Track)).Intercept.X - intercal.RX;
				indep[1, i] = ((SySal.Tracking.MIPEmulsionTrackInfo)(goodpairs[i].Second.Track)).Intercept.Y - intercal.RY;
				dep[i] = goodpairs[i].First.Info.Intercept.Y - intercal.RY;
			}
			NumericalTools.Fitting.MultipleLinearRegression(indep, dep, ref res, ref dummy);
			Console.WriteLine("{0}\t{1}\t{2}\t{3}", res[0], res[1], res[2], dummy);
			intercal.TY = res[0]; intercal.MYX = res[1]; intercal.MYY = res[2];
			CheckIntercalibration(intercal);
			if (rewrite)
			{
				MyTransformation.Transformation = intercal;
				SySal.OperaPersistence.Persist(args[2], new LinkedZone(caliblz));
			}
		}

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			try
			{
				int nzones = 0;
				bool rewrite = false;
				if (args.Length < 1) 
				{
					ShowUsage();
					return;
				}
				if (args[0] == "1" || args[0].ToLower() == "1r")
				{
					nzones = 1;
					if (args[0].ToLower() == "1r") rewrite = true;
				}
				else if (args[0] == "3" || args[0].ToLower() == "3r")
				{
					nzones = 3;
					if (args[0].ToLower() == "3r") rewrite = true;
				}
				else
				{
					ShowUsage();
					return;
				}
				string [] oldargs = args;
				args = new string[oldargs.Length - 1];
				int i;
				for (i = 1; i < oldargs.Length; i++) args[i - 1] = oldargs[i];
				if ((nzones == 1 && (args.Length != 9 && args.Length != 11)) ||
					(nzones == 3 && (args.Length != 12 && args.Length != 14)))
				{
					ShowUsage();
					return;
				}
				SySal.DAQSystem.Scanning.IntercalibrationInfo intercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
				if (nzones == 1 && args.Length == 11)
				{
					intercal.RX = Convert.ToDouble(args[9], System.Globalization.CultureInfo.InvariantCulture);
					intercal.RY = Convert.ToDouble(args[10], System.Globalization.CultureInfo.InvariantCulture);
				}
				else if (nzones == 3 && args.Length == 14)
				{
					intercal.RX = Convert.ToDouble(args[12], System.Globalization.CultureInfo.InvariantCulture);
					intercal.RY = Convert.ToDouble(args[13], System.Globalization.CultureInfo.InvariantCulture);
				}

				if (nzones == 1) OneZoneIntercalibration(args, ref intercal, rewrite);
				else ThreeZoneIntercalibration(args, ref intercal, rewrite);

				System.IO.StreamWriter wr = new System.IO.StreamWriter(args[0]);
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Scanning.IntercalibrationInfo));
				xmls.Serialize(wr, intercal);
				wr.Flush();
				wr.Close();
			}
			catch (Exception x)
			{
				Console.Error.WriteLine(x);
			}
		}

		
	}
}
