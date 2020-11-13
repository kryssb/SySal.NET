using System;
using SySal;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal.DAQSystem.Drivers;
using System.Xml;
using System.Xml.Serialization;


namespace SySal.DAQSystem.Drivers.Intercalibration3Driver
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
			mutks = LinkedZone.Side.GetTracks(m_Top);
			foreach (SySal.Scanning.MIPIndexedEmulsionTrack mutk in mutks)
				tMIPEmulsionTrack.ApplyTransformation(mutk);
			mutks = LinkedZone.Side.GetTracks(m_Bottom);
			foreach (SySal.Scanning.MIPIndexedEmulsionTrack mutk in mutks)
				tMIPEmulsionTrack.ApplyTransformation(mutk);
			SySal.DAQSystem.Scanning.IntercalibrationInfo otr = lz.Transform;
			SySal.DAQSystem.Scanning.IntercalibrationInfo tr = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
			tr.MXX = MyTransformation.Transformation.MXX * otr.MXX + MyTransformation.Transformation.MXY * otr.MYX;
			tr.MXY = MyTransformation.Transformation.MXX * otr.MXY + MyTransformation.Transformation.MXY * otr.MYY;
			tr.MYX = MyTransformation.Transformation.MYX * otr.MXX + MyTransformation.Transformation.MYY * otr.MYX;
			tr.MYY = MyTransformation.Transformation.MYX * otr.MXY + MyTransformation.Transformation.MYY * otr.MYY;
			tr.RX = MyTransformation.Transformation.RX;
			tr.RY = MyTransformation.Transformation.RY;
			tr.TZ = MyTransformation.Transformation.TZ + otr.TZ;
			tr.TX = (tr.MXX - MyTransformation.Transformation.MXX) * (MyTransformation.Transformation.RX - otr.RX) + (tr.MXY - MyTransformation.Transformation.MXY) * (MyTransformation.Transformation.RY - otr.RY) + MyTransformation.Transformation.MXX * otr.TX + MyTransformation.Transformation.MXY * otr.TY + MyTransformation.Transformation.TX;
			tr.TY = (tr.MYX - MyTransformation.Transformation.MYX) * (MyTransformation.Transformation.RX - otr.RX) + (tr.MYY - MyTransformation.Transformation.MYY) * (MyTransformation.Transformation.RY - otr.RY) + MyTransformation.Transformation.MYX * otr.TX + MyTransformation.Transformation.MYY * otr.TY + MyTransformation.Transformation.TY;
			m_Transform = tr;
		}

	}

	/// <summary>
	/// Defines if reference plates for intercalibration must be upstream or downstream of the current plate.
	/// </summary>
	[Serializable]
	public enum ReferenceDirection
	{
		/// <summary>
		/// The reference plate must be upstream of the plate being intercalibrated.
		/// </summary>
		Upstream,
		/// <summary>
		/// The reference plate must be downstream of the plate being intercalibrated.
		/// </summary>
		Downstream
	}

	/// <summary>
	/// Settings for Intercalibration3Driver.
	/// </summary>
	[Serializable]
	public class Intercalibration3Settings
	{
		/// <summary>
		/// The Id of the scanning program settings.
		/// </summary>
		public long ScanningConfigId;
		/// <summary>
		/// The Id of the linking program settings.
		/// </summary>
		public long LinkConfigId;
		/// <summary>
		/// The Id for quality cut.
		/// </summary>
		public long QualityCutId;
		/// <summary>
		/// Position tolerance for pattern matching (at the last iteration).
		/// </summary>
		public double PositionTolerance;
		/// <summary>
		/// Slope tolerance for pattern matching.
		/// </summary>
		public double SlopeTolerance;
		/// <summary>
		/// Minimum number of matching tracks to accept the pattern matching.
		/// </summary>
		public int MinMatches;
		/// <summary>
		/// Maximum absolute offset between zones.
		/// </summary>
		public double MaxOffset;
		/// <summary>
		/// Minimum distance spanned by zones in the X direction.
		/// </summary>
		public double XZoneDistance;
		/// <summary>
		/// Minimum distance spanned by zones in the Y direction.
		/// </summary>
		public double YZoneDistance;
		/// <summary>
		/// Size of each zone in micron.
		/// </summary>
		public double ZoneSize;
		/// <summary>
		/// Number of iterations for intercalibration refinement (minimum is 1).
		/// </summary>
		public int Iterations;
		/// <summary>
		/// Position tolerance for pattern matching at the first iteration. The actual position tolerance that is used at each iteration decreases linearly from this value to the value of <c>PositionTolerance</c>. If this is set to zero or a value that is lower than PositionTolerance, it is reset to be equal to PositionTolerance.
		/// </summary>
		public double InitialPositionTolerance;
		/// <summary>
		/// Defines whether reference plates are to be found among upstream or downstream plates.
		/// </summary>
		public ReferenceDirection ReferenceDirection;
	}

	/// <summary>
	/// Mapping Position.
	/// </summary>
	[Serializable]
	public class MapPos
	{
		/// <summary>
		/// X center of the zone.
		/// </summary>
		public double X;
		/// <summary>
		/// Y center of the zone.
		/// </summary>
		public double Y;
		/// <summary>
		/// X translation for the zone.
		/// </summary>
		public double DX;
		/// <summary>
		/// Y translation for the zone.
		/// </summary>
		public double DY;
		/// <summary>
		/// Sets the zone translation from pairs of matched tracks.
		/// </summary>
		/// <param name="pairs">the list of track pairs.</param>
		public void SetFromPairs(SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs)
		{
			X = Y = DX = DY = 0;
			double EqXX = 0.0, EqYX = 0.0, EqYY = 0.0, EqDX = 0.0, EqDY = 0.0;
			double TkDX, TkDY, TkDS, Det;
			SySal.Tracking.MIPEmulsionTrackInfo Tk0, Tk1;
            rP = new SySal.BasicTypes.Vector2[pairs.Length];
            rS = new SySal.BasicTypes.Vector2[pairs.Length];
            cP = new SySal.BasicTypes.Vector2[pairs.Length];
            int i = 0;            
			foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair p in pairs)
			{	                
				//Tk0 = ((SySal.Tracking.MIPEmulsionTrackInfo)(p.First.Track));
                //Tk1 = ((SySal.Tracking.MIPEmulsionTrackInfo)(p.Second.Track));				
                Tk0 = ((SySal.Tracking.MIPEmulsionTrackInfo)(p.First.Info));
				Tk1 = ((SySal.Tracking.MIPEmulsionTrackInfo)(p.Second.Info));				
				X += cP[i].X = Tk1.Intercept.X;
				Y += cP[i].Y = Tk1.Intercept.Y;
				TkDX = (rP[i].X = Tk0.Intercept.X) - Tk1.Intercept.X;
				TkDY = (rP[i].Y = Tk0.Intercept.Y) - Tk1.Intercept.Y;
				TkDS = (rS[i].Y = Tk0.Slope.Y) * TkDX - (rS[i].X = Tk0.Slope.X) * TkDY;
				EqDX += Tk0.Slope.Y * TkDS;
				EqDY += Tk0.Slope.X * TkDS;
				EqXX += Tk0.Slope.Y * Tk0.Slope.Y;
				EqYX += Tk0.Slope.X * Tk0.Slope.Y;
				EqYY -= Tk0.Slope.X * Tk0.Slope.X;
                i++;
			}
			Det = 1.0 / (EqYY * EqXX + EqYX * EqYX);
			X /= pairs.Length;
			Y /= pairs.Length;
			DX = (EqDX * EqYY + EqDY * EqYX) * Det;
			DY = (EqDY * EqXX - EqDX * EqYX) * Det;					
		}
        /// <summary>
        /// Positions of reference tracks
        /// </summary>
        public SySal.BasicTypes.Vector2[] rP;
        /// <summary>
        /// Slopes of reference tracks
        /// </summary>
        public SySal.BasicTypes.Vector2[] rS;
        /// <summary>
        /// Positions of calibration tracks
        /// </summary>
        public SySal.BasicTypes.Vector2[] cP;
	}

	/// <summary>
	/// Pair of indices of matching tracks.
	/// </summary>
	[Serializable]
	public class IndexPair
	{
		/// <summary>
		/// Index of track on the reference zone.
		/// </summary>
		public long IdRef;
		/// <summary>
		/// Index of track on the zone to be calibrated.
		/// </summary>
		public long IdCal;
	}

	/// <summary>
	/// Zone Mapping Information.
	/// </summary>
	[Serializable]
	public class ZoneMapInfo
	{
		/// <summary>
		/// Mapping Position for the zone.
		/// </summary>
		public MapPos Info;		
		/// <summary>
		/// Pairs of matching tracks.
		/// </summary>
		public IndexPair [] Pairs;
	}

	/// <summary>
	/// Intercalibration3Driver executor.
	/// </summary>
	/// <remarks>
	/// <para>Intercalibration3Driver performs plate calibration.</para>
	/// <para>Longitudinal offsets are accounted for in translation computations, but they are not written to the DB to change the plate Z.</para>
	/// <para>Results of pattern matching are written to TB_PATTERN_MATCH and the calibration obtained is recorded in VW_PLATES.</para>
	/// <para>Intercalibration parameters and pattern matching are performed iteratively, linearly decreasing the mapping tolerances.</para>
	/// <para>
	/// The following substitutions apply:
	/// <list type="table">
	/// <item><term><c>%EXEREP%</c></term><description>Executable repository path specified in the Startup file.</description></item>
	/// <item><term><c>%SCRATCH%</c></term><description>Scratch directory specified in the Startup file.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// A sample XML configuration for Intercalibration3Driver follows:
	/// <example>
	/// <code>
	/// &lt;Intercalibration3Settings&gt;
	///  &lt;ScanningConfigId&gt;1003892834&lt;/ScanningConfigId&gt;
	///  &lt;LinkConfigId&gt;1008832388&lt;/LinkConfigId&gt;
	///  &lt;QualityCutId&gt;1003892838&lt;/QualityCutId&gt;
	///  &lt;PositionTolerance&gt;20&lt;/PositionTolerance&gt;
	///  &lt;SlopeTolerance&gt;0.04&lt;/SlopeTolerance&gt;
	///  &lt;MinMatches&gt;20&lt;/MinMatches&gt;
	///  &lt;MaxOffset&gt;3000&lt;/MaxOffset&gt;
	///  &lt;XZoneDistance&gt;80000&lt;/XZoneDistance&gt;
	///  &lt;YZoneDistance&gt;60000&lt;/YZoneDistance&gt;
	///  &lt;ZoneSize&gt;6000&lt;/ZoneSize&gt;
	///  &lt;Iterations&gt;3&lt;/Iterations&gt;
	///  &lt;InitialPositionTolerance&gt;40&lt;/InitialPositionTolerance&gt;
	/// &lt;/Intercalibration3Settings&gt;
	/// </code>
	/// </example>
	/// </para>
	/// <para><b>NOTICE: If the quality cut id is identical to the linker id, no quality cut is applied (unless the linker applies its own quality cuts).</b></para>
	/// </remarks>	
	public class Exe
	{
		static void ShowExplanation()
		{
			ExplanationForm EF = new ExplanationForm();
			System.IO.StringWriter strw = new System.IO.StringWriter();
			strw.WriteLine("");
			strw.WriteLine("");
			strw.WriteLine("Intercalibration3Driver");
			strw.WriteLine("--------------");
			strw.WriteLine("Intercalibration3Driver performs plate calibration.");
			strw.WriteLine("Longitudinal offsets are accounted for in translation computations, but they are not written to the DB to change the plate Z.");
			strw.WriteLine("Intercalibration parameters and pattern matching are performed iteratively, linearly decreasing the mapping tolerances.");
			strw.WriteLine("--------------");
			strw.WriteLine("The following substitutions apply (case is disregarded):");
			strw.WriteLine("%EXEREP% = Executable repository path specified in the Startup file.");
			strw.WriteLine("%SCRATCH% = Scratch directory specified in the Startup file.");			
			strw.WriteLine("--------------");
			strw.WriteLine("The program settings should have the following structure:");
			Intercalibration3Settings iset = new Intercalibration3Settings();
			iset.LinkConfigId = 1008832388;
			iset.QualityCutId = 1003892838;
			iset.ScanningConfigId = 1003892834;
			iset.MaxOffset = 3000.0;
			iset.MinMatches = 20;
			iset.PositionTolerance = 20.0;
			iset.SlopeTolerance = 0.04;
			iset.XZoneDistance = 80000.0;
			iset.YZoneDistance = 60000.0;
			iset.ZoneSize = 6000.0;
			iset.Iterations = 3;
			iset.InitialPositionTolerance = 40.0;
			new System.Xml.Serialization.XmlSerializer(typeof(Intercalibration3Settings)).Serialize(strw, iset);
			EF.RTFOut.Text = strw.ToString();
			EF.ShowDialog();			
		}

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[MTAThread]
		internal static void Main(string[] args)
		{
			HE = SySal.DAQSystem.Drivers.HostEnv.Own;
			if (HE == null)
			{
				ShowExplanation();
				return;
			}

			Execute();
		}

		private static SySal.DAQSystem.Drivers.HostEnv HE = null;

		private static SySal.OperaDb.OperaDbConnection Conn = null;

		private static string BasePath = null;

		private static SySal.DAQSystem.Drivers.ScanningStartupInfo StartupInfo;

		private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = null;

		private static Intercalibration3Settings ProgSettings;

		private static string QualityCut;

		private static string LinkConfig;

		private static long ReferencePlate;

		private static double ReferenceDeltaZ;

		private static long ReferenceCalibrationOperation;

		private static long [] ReferenceZones;

        private static char UsedMarkSet;

		private static SySal.DAQSystem.ScanServer ScanSrv;

		private static SySal.DAQSystem.IDataProcessingServer DataProcSrv;

		private static System.Collections.Queue ScanQueue = new System.Collections.Queue();

		private static int TotalZones = 0;

		private static System.Threading.ManualResetEvent ProcessEvent = new System.Threading.ManualResetEvent(false);

		private static System.Threading.Thread ThisThread = null;

        private static System.Threading.Thread DBKeepAliveThread = null;

        private static void DBKeepAliveThreadExec()
        {
            try
            {
                SySal.OperaDb.OperaDbCommand keepalivecmd = null;
                lock (Conn)
                    keepalivecmd = new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM DUAL", Conn);
                while (Conn != null)
                {
                    keepalivecmd.ExecuteScalar();
                    System.Threading.Thread.Sleep(10000);
                }
            }
            catch (System.Threading.ThreadAbortException)
            {
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception) { }
        }

        private static System.Threading.Thread WorkerThread = new System.Threading.Thread(new System.Threading.ThreadStart(WorkerThreadExec));

		private static System.Collections.Queue WorkQueue = new System.Collections.Queue();

		private static void WorkerThreadExec()
		{
            try
            {
                while (true)
                {
                    int qc;
                    try
                    {
                        System.Threading.Thread.Sleep(System.Threading.Timeout.Infinite);
                    }
                    catch (System.Threading.ThreadInterruptedException) { }
                    lock (WorkQueue)
                        if ((qc = WorkQueue.Count) == 0) return;
                    while (qc > 0)
                    {
                        object obj = null;
                        lock (WorkQueue)
                        {
                            obj = WorkQueue.Dequeue();
                            qc = WorkQueue.Count;
                        }
                        PostProcess(obj);
                    }
                }
            }
            catch (System.Threading.ThreadAbortException)
            {
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception) { }
		}

		private static SySal.DAQSystem.Scanning.IntercalibrationInfo Intercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();

		private static System.Exception ThisException = null;

		private delegate void dPostProcess(bool init);


		static string corrstring = null;

		static ZoneMapInfo [] zmi = new ZoneMapInfo[3];

		static SySal.Scanning.Plate.IO.OPERA.LinkedZone [] reflz = new SySal.Scanning.Plate.IO.OPERA.LinkedZone[3];
		
		static SySal.Scanning.Plate.IO.OPERA.LinkedZone [] caliblz = new SySal.Scanning.Plate.IO.OPERA.LinkedZone[3];

		static System.DateTime [] starttime = new System.DateTime[3];

		static System.DateTime [] endtime = new System.DateTime[3];

		static object LinkerExe = null;

		static object QualityCutExe = null;

		static SySal.Processing.QuickMapping.QuickMapper QM = null;

		static SySal.OperaDb.OperaDbTransaction trans = null;

		static SySal.Processing.QuickMapping.Configuration C = new SySal.Processing.QuickMapping.Configuration();

        private static SySal.DAQSystem.Scanning.IntercalibrationInfo IntercalFromZones(ZoneMapInfo[] zmi, SySal.BasicTypes.Vector2 center)
        {
            int i, j;
            int totalentries;
            for (i = totalentries = 0; i < zmi.Length; i++)
                totalentries += zmi[i].Info.rP.Length;
            double[] rDX = new double[totalentries];
            double[] rDY = new double[totalentries];
            double[] rSX = new double[totalentries];
            double[] rSY = new double[totalentries];
            double[] cPX = new double[totalentries];
            double[] cPY = new double[totalentries];
            for (i = j = totalentries = 0; i < zmi.Length; i++)
            {
                MapPos m = zmi[i].Info;
                for (j = 0; j < m.rP.Length; j++)
                {
                    rDX[totalentries] = m.rP[j].X - center.X - (cPX[totalentries] = m.cP[j].X - center.X);
                    rDY[totalentries] = m.rP[j].Y - center.Y - (cPY[totalentries] = m.cP[j].Y - center.Y);
                    rSX[totalentries] = m.rS[j].X;
                    rSY[totalentries] = m.rS[j].Y;
                    totalentries++;
                }
            }
            double [] align_params = new double[7];
            NumericalTools.Fitting.Affine_Focusing(rDX, rDY, cPX, cPY, rSX, rSY, ref align_params);
            SySal.DAQSystem.Scanning.IntercalibrationInfo intercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
            intercal.RX = center.X;
            intercal.RY = center.Y;
            intercal.MXX = 1.0 + align_params[0];
            intercal.MXY = align_params[1];
            intercal.MYX = align_params[2];
            intercal.MYY = 1.0 + align_params[3];
            intercal.TX = align_params[4];
            intercal.TY = align_params[5];
            intercal.TZ = align_params[6];
            return intercal;
        }

        private static void PostProcess(object obj)
		{
			bool init;
			if (obj.GetType() == typeof(bool)) init = true;
			else init = false;
			try
			{
				int i, zone;

				if (init)
				{
					if (ProgSettings.Iterations <= 1) ProgSettings.Iterations = 1;
					if (ProgSettings.InitialPositionTolerance <= ProgSettings.PositionTolerance || ProgSettings.Iterations == 1) ProgSettings.InitialPositionTolerance = ProgSettings.PositionTolerance;

					if (System.IO.File.Exists(StartupInfo.ScratchDir + @"\fragmentshiftcorrection_" + StartupInfo.MachineId + ".xml"))
					{
						System.IO.StreamReader rcorr = null;
						try
						{
							rcorr = new System.IO.StreamReader(StartupInfo.ScratchDir + @"\fragmentshiftcorrection_" + StartupInfo.MachineId + ".xml");
							corrstring = rcorr.ReadToEnd();
							rcorr.Close();
						}
						catch (Exception) 
						{
							if (rcorr != null) rcorr.Close();
						}
					}
		
					if (ReferenceZones != null)
						for (i = 0; i < 3; i++)
							reflz[i] = new SySal.OperaDb.Scanning.LinkedZone(Conn, null, StartupInfo.Plate.BrickId, ReferenceZones[i], SySal.OperaDb.Scanning.LinkedZone.DetailLevel.BaseGeom);

					LinkerExe = System.Activator.CreateInstanceFrom(StartupInfo.ExeRepository + @"\BatchLink.exe", "SySal.Executables.BatchLink.Exe").Unwrap();
					if (ProgSettings.LinkConfigId == ProgSettings.QualityCutId) QualityCutExe = null;
					else QualityCutExe = System.Activator.CreateInstanceFrom(StartupInfo.ExeRepository + @"\TLGSel.exe", "SySal.Executables.TLGSel.Exe").Unwrap();

					if (ReferenceZones != null)
					{
						QM = new SySal.Processing.QuickMapping.QuickMapper();						
						C.Name = "Intercalibration3 configuration";
						C.PosTol = ProgSettings.InitialPositionTolerance;
						C.SlopeTol = ProgSettings.SlopeTolerance;
						QM.Config = C;
					}
				
                    lock(Conn)
    					trans = Conn.BeginTransaction();
					return;
				}
				SySal.DAQSystem.Scanning.ZoneDesc zd = (SySal.DAQSystem.Scanning.ZoneDesc)obj;

				System.Globalization.CultureInfo InvC = System.Globalization.CultureInfo.InvariantCulture;
				zone = (int)(zd.Series - 101);
				starttime[zone] = System.IO.File.GetCreationTime(zd.Outname + ".rwc");
				string [] rwds = System.IO.Directory.GetFiles(zd.Outname.Substring(0, zd.Outname.LastIndexOf("\\")), zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
				foreach (string s in rwds)
				{
					System.DateTime modtime = System.IO.File.GetLastWriteTime(s);
					if (modtime > endtime[zone]) endtime[zone] = modtime;
				}
				caliblz[zone] = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)(LinkerExe.GetType().InvokeMember("ProcessData", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, LinkerExe, new object[4] {zd.Outname + ".rwc", null, LinkConfig, corrstring}));
				if (QualityCutExe != null) caliblz[zone] = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)(QualityCutExe.GetType().InvokeMember("ProcessData", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, QualityCutExe, new object[3] {caliblz[zd.Series - 101], QualityCut, true}));
				SySal.OperaPersistence.Persist(StartupInfo.ScratchDir + @"\intercal3_" + StartupInfo.ProcessOperationId + "_" + zone + ".tlg", caliblz[zone]);

				if (ReferenceZones != null)
				{
					SySal.Tracking.MIPEmulsionTrackInfo [] refzone = new SySal.Tracking.MIPEmulsionTrackInfo[reflz[zone].Length];
					for (i = 0; i < refzone.Length; i++)
						refzone[i] = reflz[zone][i].Info;							
					SySal.Tracking.MIPEmulsionTrackInfo [] calibzone = new SySal.Tracking.MIPEmulsionTrackInfo[caliblz[zone].Length];
					for (i = 0; i < calibzone.Length; i++)
						calibzone[i] = caliblz[zone][i].Info;
					SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs = QM.Match(refzone, calibzone, ReferenceDeltaZ, ProgSettings.MaxOffset, ProgSettings.MaxOffset);

					if (pairs.Length < ProgSettings.MinMatches)
					{
						throw new Exception("Too few matching tracks: " + ProgSettings.MinMatches.ToString() + " required, " + pairs.Length + " obtained. Aborting.");
					}
					HE.WriteLine("Matches: " + pairs.Length);
					zmi[zone] = new ZoneMapInfo();							
					zmi[zone].Info = new MapPos();
					zmi[zone].Info.SetFromPairs(pairs); 
					zmi[zone].Pairs = new IndexPair[pairs.Length];							
					for (i = 0; i < pairs.Length; i++)
					{
						zmi[zone].Pairs[i] = new IndexPair();
						zmi[zone].Pairs[i].IdRef = pairs[i].First.Index;
						zmi[zone].Pairs[i].IdCal = pairs[i].Second.Index;
					}
					HE.WriteLine("X: " + zmi[zone].Info.X);
					HE.WriteLine("Y: " + zmi[zone].Info.Y);
					HE.WriteLine("DX: " + zmi[zone].Info.DX);
					HE.WriteLine("DY: " + zmi[zone].Info.DY);
					refzone = null;
					calibzone = null;
//					reflz[zone] = null;
					}
				else
				{
					SySal.OperaDb.Scanning.LinkedZone.Save(caliblz[zone], StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, zd.Series, zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1), starttime[zone], endtime[zone], Conn, trans);
					caliblz[zone] = null;
				}

				if (zone == 2)
				{
					if (ReferenceZones != null)
					{                                                
						double x20 = zmi[2].Info.X - zmi[0].Info.X;
						double x10 = zmi[1].Info.X - zmi[0].Info.X;
						double y20 = zmi[2].Info.Y - zmi[0].Info.Y;
						double y10 = zmi[1].Info.Y - zmi[0].Info.Y;
						double det = 1.0 / (x10 * y20 - x20 * y10);
						double u20 = zmi[2].Info.DX - zmi[0].Info.DX;
						double v20 = zmi[2].Info.DY - zmi[0].Info.DY;
						double u10 = zmi[1].Info.DX - zmi[0].Info.DX;
						double v10 = zmi[1].Info.DY - zmi[0].Info.DY;
						Intercal.MXX = 1.0 + (u10 * y20 - u20 * y10) * det;
						Intercal.MXY = (u20 * x10 - u10 * x20) * det;
						Intercal.MYX = (v10 * y20 - v20 * y10) * det;
						Intercal.MYY = 1.0 + (v20 * x10 - v10 * x20) * det;
						Intercal.TX = zmi[0].Info.DX + zmi[0].Info.X - Intercal.MXX * (zmi[0].Info.X - Intercal.RX) - Intercal.MXY * (zmi[0].Info.Y - Intercal.RY) - Intercal.RX;
						Intercal.TY = zmi[0].Info.DY + zmi[0].Info.Y - Intercal.MYX * (zmi[0].Info.X - Intercal.RX) - Intercal.MYY * (zmi[0].Info.Y - Intercal.RY) - Intercal.RY;

                        SySal.BasicTypes.Vector2 refcenter = new SySal.BasicTypes.Vector2();
                        refcenter.X = Intercal.RX;
                        refcenter.Y = Intercal.RY;
                        Intercal = IntercalFromZones(zmi, refcenter);

						HE.WriteLine("Intercalibration computed");
						HE.WriteLine("MXX: " + Intercal.MXX.ToString(InvC));
						HE.WriteLine("MXY: " + Intercal.MXY.ToString(InvC));
						HE.WriteLine("MYX: " + Intercal.MYX.ToString(InvC));
						HE.WriteLine("MYY: " + Intercal.MYY.ToString(InvC));
						HE.WriteLine("TX " + Intercal.TX.ToString(InvC));
						HE.WriteLine("TY " + Intercal.TY.ToString(InvC));
                        HE.WriteLine("TZ " + Intercal.TZ.ToString(InvC));

// FROM HERE
						int CurrentIteration = 0;
						while (++CurrentIteration < ProgSettings.Iterations)
						{
							double lambda = (float)CurrentIteration / (ProgSettings.Iterations - 1);
							C.PosTol = ProgSettings.PositionTolerance * lambda + (1.0 - lambda) * ProgSettings.InitialPositionTolerance;
							QM.Config = C;

							for (zone = 0; zone < 3; zone++)
							{
								SySal.Tracking.MIPEmulsionTrackInfo [] refzone = new SySal.Tracking.MIPEmulsionTrackInfo[reflz[zone].Length];
								for (i = 0; i < refzone.Length; i++)
									refzone[i] = reflz[zone][i].Info;							
								SySal.Tracking.MIPEmulsionTrackInfo [] calibzone = new SySal.Tracking.MIPEmulsionTrackInfo[caliblz[zone].Length];
								for (i = 0; i < calibzone.Length; i++)
								{
									SySal.Tracking.MIPEmulsionTrackInfo info1;
									SySal.Tracking.MIPEmulsionTrackInfo info2 = new SySal.Tracking.MIPEmulsionTrackInfo();
									info1 = caliblz[zone][i].Info;
									info2.Count = info1.Count;
									info2.AreaSum = info1.AreaSum;
									info2.Intercept.X = Intercal.MXX * (info1.Intercept.X - Intercal.RX) + Intercal.MXY * (info1.Intercept.Y - Intercal.RY) + Intercal.TX + Intercal.RX;
									info2.Intercept.Y = Intercal.MYX * (info1.Intercept.X - Intercal.RX) + Intercal.MYY * (info1.Intercept.Y - Intercal.RY) + Intercal.TY + Intercal.RY;
									info2.Sigma = info1.Sigma;
									info2.Slope.X = Intercal.MXX * info1.Slope.X + Intercal.MXY * info1.Slope.Y;
									info2.Slope.Y = Intercal.MYX * info1.Slope.X + Intercal.MYY * info1.Slope.Y;
									info2.TopZ = info1.TopZ;
									info2.BottomZ = info1.BottomZ;
									calibzone[i] = info2;
								}
								SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs = QM.Match(refzone, calibzone, ReferenceDeltaZ - Intercal.TZ, ProgSettings.MaxOffset, ProgSettings.MaxOffset);
								if (pairs.Length < ProgSettings.MinMatches)
								{
									throw new Exception("Too few matching tracks: " + ProgSettings.MinMatches.ToString() + " required, " + pairs.Length + " obtained. Aborting.");
								}
								HE.WriteLine("Matches: " + pairs.Length);
								zmi[zone] = new ZoneMapInfo();							
								zmi[zone].Info = new MapPos();
								zmi[zone].Info.SetFromPairs(pairs); 
								zmi[zone].Pairs = new IndexPair[pairs.Length];							
								for (i = 0; i < pairs.Length; i++)
								{
									zmi[zone].Pairs[i] = new IndexPair();
									zmi[zone].Pairs[i].IdRef = pairs[i].First.Index;
									zmi[zone].Pairs[i].IdCal = pairs[i].Second.Index;
								}
								HE.WriteLine("X: " + zmi[zone].Info.X);
								HE.WriteLine("Y: " + zmi[zone].Info.Y);
								HE.WriteLine("DX: " + zmi[zone].Info.DX);
								HE.WriteLine("DY: " + zmi[zone].Info.DY);
							}

							x20 = zmi[2].Info.X - zmi[0].Info.X;
							x10 = zmi[1].Info.X - zmi[0].Info.X;
							y20 = zmi[2].Info.Y - zmi[0].Info.Y;
							y10 = zmi[1].Info.Y - zmi[0].Info.Y;
							det = 1.0 / (x10 * y20 - x20 * y10);
							u20 = zmi[2].Info.DX - zmi[0].Info.DX;
							v20 = zmi[2].Info.DY - zmi[0].Info.DY;
							u10 = zmi[1].Info.DX - zmi[0].Info.DX;
							v10 = zmi[1].Info.DY - zmi[0].Info.DY;

							SySal.DAQSystem.Scanning.IntercalibrationInfo tempInt = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
							SySal.DAQSystem.Scanning.IntercalibrationInfo tempOut = new SySal.DAQSystem.Scanning.IntercalibrationInfo();

							tempInt.RX = Intercal.RX;
							tempInt.RY = Intercal.RY;
							tempInt.MXX = 1.0 + (u10 * y20 - u20 * y10) * det;
							tempInt.MXY = (u20 * x10 - u10 * x20) * det;
							tempInt.MYX = (v10 * y20 - v20 * y10) * det;
							tempInt.MYY = 1.0 + (v20 * x10 - v10 * x20) * det;
							tempInt.TX = zmi[0].Info.DX + zmi[0].Info.X - tempInt.MXX * (zmi[0].Info.X - tempInt.RX) - tempInt.MXY * (zmi[0].Info.Y - tempInt.RY) - tempInt.RX;
							tempInt.TY = zmi[0].Info.DY + zmi[0].Info.Y - tempInt.MYX * (zmi[0].Info.X - tempInt.RX) - tempInt.MYY * (zmi[0].Info.Y - tempInt.RY) - tempInt.RY;

                            tempInt = IntercalFromZones(zmi, refcenter);

							tempOut.RX = tempInt.RX;
							tempOut.RY = tempInt.RY;
							tempOut.MXX = tempInt.MXX * Intercal.MXX + tempInt.MXY * Intercal.MYX;
							tempOut.MXY = tempInt.MXX * Intercal.MXY + tempInt.MXY * Intercal.MYY;
							tempOut.MYX = tempInt.MYX * Intercal.MXX + tempInt.MYY * Intercal.MYX;
							tempOut.MYY = tempInt.MYX * Intercal.MXY + tempInt.MYY * Intercal.MYY;
							tempOut.TX = tempInt.MXX * Intercal.TX + tempInt.MXY * Intercal.TY + tempInt.TX;
							tempOut.TY = tempInt.MYX * Intercal.TX + tempInt.MYY * Intercal.TY + tempInt.TY;
                            tempOut.TZ = tempInt.TZ + Intercal.TZ;
							Intercal = tempOut;

							HE.WriteLine("Intercalibration computed - Iteration #" + CurrentIteration);
							HE.WriteLine("MXX: " + Intercal.MXX.ToString(InvC));
							HE.WriteLine("MXY: " + Intercal.MXY.ToString(InvC));
							HE.WriteLine("MYX: " + Intercal.MYX.ToString(InvC));
							HE.WriteLine("MYY: " + Intercal.MYY.ToString(InvC));
							HE.WriteLine("TX " + Intercal.TX.ToString(InvC));
							HE.WriteLine("TY " + Intercal.TY.ToString(InvC));
                            HE.WriteLine("TZ " + Intercal.TZ.ToString(InvC));

						}
						

// TO HERE
					}
					else
					{
						Intercal.MXX = Intercal.MYY = 1.0;
						Intercal.MXY = Intercal.MYX = 0.0;
						Intercal.TX = Intercal.TY = 0.0;
                        Intercal.TZ = 0.0;
					}
					HE.WriteLine("Writing information to DB");
					double z = Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, Conn, trans).ExecuteScalar());
                    new SySal.OperaDb.OperaDbCommand("CALL PC_CALIBRATE_PLATE(" + StartupInfo.Plate.BrickId + ", " + StartupInfo.Plate.PlateId + ", " + StartupInfo.ProcessOperationId + ", '" + UsedMarkSet + "', " + (z + Intercal.TZ).ToString(InvC) + ", " + 
						Intercal.MXX.ToString(InvC) + ", " + Intercal.MXY.ToString(InvC) + ", " + Intercal.MYX.ToString(InvC) + ", " + Intercal.MYY.ToString(InvC) + 
						", " + Intercal.TX.ToString(InvC) + ", " + Intercal.TY.ToString(InvC) + ")", Conn, trans).ExecuteNonQuery();
					if (ReferenceZones != null)
					{
						MyTransformation.Transformation = Intercal;
						long [] newzoneids = new long[3];
						const int BatchSize = 50;
						long [] a_idfirst = new long[BatchSize];
						long [] a_idnew = new long[BatchSize];
						long [] a_idinfirst = new long[BatchSize];
						long [] a_idinnew = new long[BatchSize];
						SySal.OperaDb.OperaDbCommand mapcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO " + (Conn.HasBufferTables ? "OPERA.LZ_PATTERN_MATCH" : " TB_PATTERN_MATCH") + " (ID_EVENTBRICK, ID_FIRSTZONE, ID_SECONDZONE, ID_INFIRSTZONE, ID_INSECONDZONE, ID_PROCESSOPERATION) VALUES (" + StartupInfo.Plate.BrickId + ", :ifirstzone, :newzoneid, :idinfirst, :idinnew, " + StartupInfo.ProcessOperationId + ")", Conn, trans);
						mapcmd.Parameters.Add("ifirstzone", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_idfirst;
						mapcmd.Parameters.Add("newzoneid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_idnew;
						mapcmd.Parameters.Add("idinfirst", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_idinfirst;
						mapcmd.Parameters.Add("idinnew", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_idinnew;			
						mapcmd.Prepare();
						for (zone = 0; zone < 3; zone++)
						{				
							caliblz[zone] = new LinkedZone(caliblz[zone]);
							newzoneids[zone] = SySal.OperaDb.Scanning.LinkedZone.Save(caliblz[zone], StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, zone + 101, BasePath.Substring(0, BasePath.LastIndexOf("\\")) + "_z" + (zone + 1).ToString(), starttime[zone], endtime[zone], Conn, trans);
							int b, n;
							n = zmi[zone].Pairs.Length;
							for (b = 0; b < BatchSize; b++)
							{
								a_idfirst[b] = ReferenceZones[zone];
								a_idnew[b] = newzoneids[zone];
							}
							for (i = 0; i < n; i++)
							{
								for (b = 0; b < BatchSize && i < n; b++)
								{
									a_idinfirst[b] = zmi[zone].Pairs[i].IdRef + 1;
									a_idinnew[b] = zmi[zone].Pairs[i].IdCal + 1;
									i++;
								}
								mapcmd.ArrayBindCount = b;
								mapcmd.ExecuteNonQuery();
							}		
							caliblz[zone] = null;						
						}			
					}
                    lock(Conn)
    					trans.Commit();
					caliblz = null;
					reflz = null;
					zmi = null;
				}
				ThisException = null;
				ProcessEvent.Set();
			}
			catch (Exception x)
			{
				try
				{
					HE.WriteLine("Exception:\r\n" + x.ToString());
				}
				catch (Exception) {}
				ThisException = x;
				ProcessEvent.Set();
			}
		}

		private static void DeleteFile(string filename)
		{
			try
			{
				System.IO.File.Delete(filename);
			}
			catch (Exception x)
			{
				if (System.IO.File.Exists(filename)) throw x;					
			}
		}

		private static void Execute()
		{
			ProgressInfo = HE.ProgressInfo;
			int i;

			StartupInfo = (SySal.DAQSystem.Drivers.ScanningStartupInfo)HE.StartupInfo;
			Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
			Conn.Open();
            (DBKeepAliveThread = new System.Threading.Thread(DBKeepAliveThreadExec)).Start();
			ScanSrv = HE.ScanSrv;

			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Intercalibration3Settings));
			ProgSettings = (Intercalibration3Settings)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
			xmls = null;
				
			if (StartupInfo.ExeRepository.EndsWith("\\")) StartupInfo.ExeRepository = StartupInfo.ExeRepository.Remove(StartupInfo.ExeRepository.Length - 1, 1);
			if (StartupInfo.ScratchDir.EndsWith("\\")) StartupInfo.ScratchDir = StartupInfo.ScratchDir.Remove(StartupInfo.ScratchDir.Length - 1, 1);

			if (new SySal.OperaDb.OperaDbCommand("SELECT DAMAGED FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, Conn, null).ExecuteScalar().ToString() != "N")
				throw new Exception("Plate #" + StartupInfo.Plate.PlateId + ", Brick #" + StartupInfo.Plate.BrickId + " is damaged!");

			SySal.OperaDb.OperaDbDataAdapter da = null;
			System.Data.DataSet ds = new System.Data.DataSet();
			da = new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK, ID, Z FROM TB_PLATES WHERE(ID_EVENTBRICK=" + StartupInfo.Plate.BrickId + " AND ID=" + StartupInfo.Plate.PlateId + ")", Conn, null);
			da.Fill(ds);
			if (ds.Tables[0].Rows.Count != 1) throw new Exception("Plate does not exist in DataBase!");
			long idplate = StartupInfo.Plate.PlateId;
			long idbrick = StartupInfo.Plate.BrickId;
			double plateZ = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][2]);			

			ds = new System.Data.DataSet();
			da = new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, Z, CALIBRATION, ABS(Z - " + plateZ.ToString("F0") + ") AS ZDIST FROM TB_PLATES INNER JOIN (SELECT ID AS IDPL, CALIBRATION FROM VW_PLATES WHERE (ID_EVENTBRICK = " + idbrick + " AND DAMAGED = 'N' AND CALIBRATION IS NOT NULL)) ON (ID_EVENTBRICK = " + idbrick + " AND ID = IDPL AND ID <> " + idplate + " AND Z " + ((ProgSettings.ReferenceDirection == ReferenceDirection.Upstream) ? "<" : ">") + plateZ.ToString("F0") + ") ORDER BY ZDIST ASC", Conn, null);			
			da.Fill(ds);
			if (ds.Tables[0].Rows.Count == 0)
			{
				ReferencePlate = idplate;
				ReferenceDeltaZ = 0;
				ReferenceZones = null;
				ReferenceCalibrationOperation = 0;
			}
			else
			{
				ReferencePlate = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][0]);
				ReferenceDeltaZ = plateZ - SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][1]);
				ReferenceCalibrationOperation = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][2]);
				ds = new System.Data.DataSet();
				da = new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_ZONES.ID FROM TB_ZONES WHERE(TB_ZONES.ID_EVENTBRICK = " + idbrick + " AND TB_ZONES.ID_PLATE = " + ReferencePlate + " AND TB_ZONES.ID_PROCESSOPERATION = " + ReferenceCalibrationOperation + ") ORDER BY TB_ZONES.SERIES ASC", Conn, null);
				da.Fill(ds);
				if (ds.Tables[0].Rows.Count != 3) 
				{
					ProgressInfo.Complete = true;
					ProgressInfo.ExitException = new Exception("Ambiguity in intercalibration zones for plate=" + idplate + " brick=" + idbrick + " idplate=" + ReferencePlate + ": " + ds.Tables[0].Rows.Count + " zones found!").ToString();
					HE.ProgressInfo = ProgressInfo;
					return;
				}
				ReferenceZones = new long[3];
				for (i = 0; i < 3; i++)
					ReferenceZones[i] = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[i][0]);
			}
			TotalZones = 3;

			long calibrationid;            
			StartupInfo.Plate.MapInitString = SySal.OperaDb.Scanning.Utilities.GetMapString(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, false, 
                SySal.OperaDb.Scanning.Utilities.CharToMarkType(UsedMarkSet = Convert.ToChar(new SySal.OperaDb.OperaDbCommand("SELECT MARKSET FROM TB_PROGRAMSETTINGS WHERE ID = " + StartupInfo.ProgramSettingsId, Conn).ExecuteScalar())), 
                out calibrationid, Conn, null);
		
			if (StartupInfo.RecoverFromProgressFile)
			{
				try
				{
					System.Xml.XmlDocument xmldoc = new XmlDocument();
					xmldoc.LoadXml(ProgressInfo.CustomInfo.Replace("[", "<").Replace("]", ">"));
					System.Xml.XmlNode xmln = xmldoc.FirstChild;
					double sizeoverride = Convert.ToDouble(xmln["SizeOverride"].InnerText,System.Globalization.CultureInfo.InvariantCulture);
					if (sizeoverride > 0.0) ProgSettings.ZoneSize = sizeoverride;
				}
				catch (Exception) {}
			}

			LinkConfig = new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.LinkConfigId + ")", Conn, null).ExecuteScalar().ToString();
			QualityCut = new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.QualityCutId + ")", Conn, null).ExecuteScalar().ToString().Trim();
			if (QualityCut.StartsWith("\"") && QualityCut.EndsWith("\"")) QualityCut = QualityCut.Substring(1, QualityCut.Length - 2);

			WorkerThread.Start();

			SySal.BasicTypes.Vector2 Center;
			ds = new System.Data.DataSet();
			da = new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_EVENTBRICKS.MINX - ZEROX, TB_EVENTBRICKS.MAXX - ZEROX, TB_EVENTBRICKS.MINY - ZEROY, TB_EVENTBRICKS.MAXY - ZEROY FROM TB_EVENTBRICKS WHERE(ID = " + idbrick + ")", Conn, null);
			da.Fill(ds);
			Center.X = 0.5 * (Convert.ToDouble(ds.Tables[0].Rows[0][0]) + Convert.ToDouble(ds.Tables[0].Rows[0][1]));
			Center.Y = 0.5 * (Convert.ToDouble(ds.Tables[0].Rows[0][2]) + Convert.ToDouble(ds.Tables[0].Rows[0][3]));				
			Intercal.RX = Center.X;
			Intercal.RY = Center.Y;

			int trial = 0;
			HE.WriteLine(HE.ProgramSettings);
			StartupInfo.Plate.TextDesc = "Plate #" + idplate + ", Brick #" + idbrick;
			StartupInfo.Zones = new SySal.DAQSystem.Scanning.ZoneDesc[3];
			BasePath = ReplaceStrings(@"%SCRATCH%\intercalibration_" + idbrick + "_" + idplate + "_" + StartupInfo.ProcessOperationId, 0);

			StartupInfo.Zones[0] = new SySal.DAQSystem.Scanning.ZoneDesc();
			StartupInfo.Zones[0].Series = 101;
			StartupInfo.Zones[0].Outname = BasePath + "_z1";

			StartupInfo.Zones[1] = new SySal.DAQSystem.Scanning.ZoneDesc();
			StartupInfo.Zones[1].Series = 102;
			StartupInfo.Zones[1].Outname = BasePath + "_z2";

			StartupInfo.Zones[2] = new SySal.DAQSystem.Scanning.ZoneDesc();
			StartupInfo.Zones[2].Series = 103;
			StartupInfo.Zones[2].Outname = BasePath + "_z3";

			ScanSrv = HE.ScanSrv;
			DataProcSrv = HE.DataProcSrv;
			//new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.DataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.DataProcessingServer), "tcp://localhost:" + ((int)SySal.DAQSystem.OperaPort.BatchServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(30000));

			WorkQueue.Enqueue(true);
			WorkerThread.Interrupt();

			if (ScanSrv.SetScanLayout(new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE (ID = " + ProgSettings.ScanningConfigId + ")", Conn, null).ExecuteScalar().ToString()) == false)
			{
				ProgressInfo.Complete = false;
				ProgressInfo.ExitException = new Exception("Scan Server configuration refused!").ToString();
				HE.ProgressInfo = ProgressInfo;
				return;
			}
			int z;
			bool PlateLoaded = false;
			if ((PlateLoaded = ScanSrv.LoadPlate(StartupInfo.Plate)) == false)
			{
				ProgressInfo.Complete = false;
				ProgressInfo.ExitException = new Exception("Plate not loaded!").ToString();
				HE.ProgressInfo = ProgressInfo;
				return;
			}			
			while (++trial < 3)
			{
				StartupInfo.Zones[0].MinX = Center.X - (ProgSettings.XZoneDistance + ProgSettings.ZoneSize * trial) * 0.5;
				StartupInfo.Zones[0].MaxX = StartupInfo.Zones[0].MinX + ProgSettings.ZoneSize * trial;
				StartupInfo.Zones[0].MinY = Center.Y - (ProgSettings.YZoneDistance + ProgSettings.ZoneSize * trial) * 0.5;
				StartupInfo.Zones[0].MaxY = StartupInfo.Zones[0].MinY + ProgSettings.ZoneSize * trial;
				StartupInfo.Zones[1].MinX = Center.X + (ProgSettings.XZoneDistance - ProgSettings.ZoneSize * trial) * 0.5;
				StartupInfo.Zones[1].MaxX = StartupInfo.Zones[1].MinX + ProgSettings.ZoneSize * trial;
				StartupInfo.Zones[1].MinY = Center.Y - (ProgSettings.YZoneDistance + ProgSettings.ZoneSize * trial) * 0.5;
				StartupInfo.Zones[1].MaxY = StartupInfo.Zones[1].MinY + ProgSettings.ZoneSize * trial;
				StartupInfo.Zones[2].MinX = Center.X - (ProgSettings.XZoneDistance + ProgSettings.ZoneSize * trial) * 0.5;
				StartupInfo.Zones[2].MaxX = StartupInfo.Zones[2].MinX + ProgSettings.ZoneSize * trial;
				StartupInfo.Zones[2].MinY = Center.Y + (ProgSettings.YZoneDistance - ProgSettings.ZoneSize * trial) * 0.5;
				StartupInfo.Zones[2].MaxY = StartupInfo.Zones[2].MinY + ProgSettings.ZoneSize * trial;

				ProgressInfo = new SySal.DAQSystem.Drivers.TaskProgressInfo();					
				ProgressInfo.StartTime = System.DateTime.Now;
				ProgressInfo.FinishTime = ProgressInfo.StartTime;
				ProgressInfo.Progress = 0.0;
				foreach (SySal.DAQSystem.Scanning.ZoneDesc zd in StartupInfo.Zones)
				{
					string dir = zd.Outname.Substring(0, zd.Outname.LastIndexOf("\\"));
					string name = zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1) + ".*";
					string [] fnames = System.IO.Directory.GetFiles(zd.Outname.Substring(0, zd.Outname.LastIndexOf("\\")), zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1) + ".*");
					foreach (string fname in fnames)
						DeleteFile(fname);
					ScanQueue.Enqueue(zd);
				}		
				HE.ProgressInfo = ProgressInfo;

				z = 0;

				while (true)
				{
					if (ScanQueue.Count <= 0) 
					{
						ProcessEvent.WaitOne();						
						if (ThisException != null) throw ThisException;
						if (ScanQueue.Count <= 0) break;
					}
					SySal.DAQSystem.Scanning.ZoneDesc zd = null;
					lock (ScanQueue)
						zd = (SySal.DAQSystem.Scanning.ZoneDesc)ScanQueue.Peek();						
					ProgressInfo.Progress = ((double)TotalZones - ScanQueue.Count) / ((double)(TotalZones));
					HE.ProgressInfo = ProgressInfo;

					if (ScanSrv.Scan(zd))
					{
						lock(ScanQueue)
						{
							ScanQueue.Dequeue();
							WorkQueue.Enqueue(zd);
							if (ProcessEvent.WaitOne(0, false) == true)
							{
								ProcessEvent.Reset();
							}
							WorkerThread.Interrupt();
						}
					}
					else 
					{
						ProgressInfo.Complete = false;
						ProgressInfo.ExitException = new Exception("Scanning failed for zone " + 
							StartupInfo.Plate.BrickId + "/" + StartupInfo.Plate.PlateId + "/" + zd.Series).ToString();
						HE.ProgressInfo = ProgressInfo;
						return;
					}
					System.TimeSpan timeelapsed = System.DateTime.Now - ProgressInfo.StartTime;
					ProgressInfo.FinishTime = ProgressInfo.StartTime.AddMilliseconds(timeelapsed.TotalMilliseconds / (z + 1) * StartupInfo.Zones.Length);
					HE.ProgressInfo = ProgressInfo;	
				}

				ProgressInfo.Progress = 1.0;
				HE.ProgressInfo = ProgressInfo;
				trial = 10;

				foreach (SySal.DAQSystem.Scanning.ZoneDesc zd in StartupInfo.Zones)
				{
					string dir = zd.Outname.Substring(0, zd.Outname.LastIndexOf("\\"));
					string name = zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1) + ".*";
					string [] fnames = System.IO.Directory.GetFiles(zd.Outname.Substring(0, zd.Outname.LastIndexOf("\\")), zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1) + ".*");
					foreach (string fname in fnames)
						DeleteFile(fname);
				}

				DeleteFile(BasePath + "_r1.tlg");
				DeleteFile(BasePath + "_r2.tlg");
				DeleteFile(BasePath + "_r3.tlg");
				DeleteFile(BasePath + "_m1.xml");
				DeleteFile(BasePath + "_m2.xml");
				DeleteFile(BasePath + "_m3.xml");					
			}

			DeleteFile(BasePath + "_linkconfig.xml");
			lock(WorkQueue)
			{
				WorkQueue.Clear();
				WorkerThread.Interrupt();
			}
			WorkerThread.Join();

			if (trial < 10) 
			{
				ProgressInfo.Complete = false;
				ProgressInfo.ExitException = new Exception("Intercalibration3 failed!").ToString();
				HE.ProgressInfo = ProgressInfo;
                lock (Conn)
                {
                    Conn.Close();
                    Conn = null;
                }
				return;
			}
			else 
			{
				ProgressInfo.Complete = true;
				ProgressInfo.ExitException = null;
				ProgressInfo.FinishTime = System.DateTime.Now;
				ProgressInfo.Progress = 1.0;
				HE.ProgressInfo = ProgressInfo;
                lock (Conn)
                {
                    Conn.Close();
                    Conn = null;
                }
				return;
			}
		}

		private static string ReplaceStrings(string s, long zoneid)
		{
			string ns = (string)s.Clone();
			ns = ns.Replace("%EXEREP%", StartupInfo.ExeRepository);
			ns = ns.Replace("%SCRATCH%", StartupInfo.ScratchDir);
			return ns;
		}

	}
}
