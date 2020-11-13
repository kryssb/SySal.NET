using System;
using SySal;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal.DAQSystem.Drivers;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.DAQSystem.Drivers.TestPredictionDriver
{
	[Serializable]
	public enum DataSource
	{
		/// <summary>
		/// Data come from random generation.
		/// </summary>
		Random = 1,
		/// <summary>
		/// Data come from a TLG of real tracks.
		/// </summary>
		SingleLinkedZone = 2
	}

	/// <summary>
	/// Configuration for TestPredictionDriver.
	/// </summary>
	[Serializable]
	public class Config
	{
		/// <summary>
		/// Data source for candidates.
		/// </summary>
		public DataSource DataSource;
		/// <summary>
		/// If random data are generated, the efficiency to find a candidate is set by RandomEfficiency.
		/// </summary>
		public double RandomEfficiency;
		/// <summary>
		/// if random data are generated, this is the RMS of position residuals for generated data.
		/// </summary>
		public double RandomPositionRMS;
		/// <summary>
		/// if random data are generated, this is the RMS of slope residuals for generated data.
		/// </summary>
		public double RandomSlopeRMS;
	}

	/// <summary>
	/// TestPredictionDriver executor.
	/// </summary>
	/// <remarks>
	/// <para>TestPredictionDriver simulates prediction-driven scanning.</para>
	/// <para>All input and output live in the DB. Predictions are read from TB_SCANBACK_PREDICTIONS and results are written to the same table.</para>
	/// <para>
	/// Candidates for scanning can come from a TLG file or they can be generated at random.
	/// </para>
	/// <para>
	/// Supported Interrupts:
	/// <list type="bullet">
	/// <item>
	/// <description><c>TLG &lt;tlgpath&gt;</c> loads the TLG from which candidates will be taken.</description>
	/// </item>
	/// </list>
	/// </para>
	/// <para>
	/// A sample XML configuration for TestPredictionDriver follows:
	/// <example>
	/// <code>
	/// &lt;Config&gt;
	///  &lt;DataSource&gt;Random&lt;/DataSource&gt;
	///  &lt;RandomEfficiency&gt;0.9&lt;/RandomEfficiency&gt;
	///  &lt;RandomPositionRMS&gt;10&lt;/RandomPositionRMS&gt;
	///  &lt;RandomSlopeRMS&gt;0.006&lt;/RandomSlopeRMS&gt;
	/// &lt;/Config&gt;
	/// </code>
	/// </example>
	/// </para>
	/// <para>
	/// The possible sources are:
	/// <list type="table">
	/// <item><term>Random</term><description>the candidates are generated randomly, simulating the specified efficiency and errors.</description></item>
	/// <item><term>SingleLinkedZone</term><description>the candidates are taken from a TLG file.</description></item>
	/// </list>
	/// </para>
	/// </remarks>	
	public class Exe : MarshalByRefObject
	{
		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>the lifetime service object or null.</returns>
		public override object InitializeLifetimeService()
		{
			return null;	
		}

		static SySal.DAQSystem.Drivers.HostEnv HE = null;

		static SySal.OperaDb.OperaDbConnection DB = null;

		static SySal.DAQSystem.Drivers.ScanningStartupInfo StartupInfo;

		static Config C;
		
		static void ShowExplanation()
		{
			ExplanationForm EF = new ExplanationForm();
			System.IO.StringWriter strw = new System.IO.StringWriter();
			strw.WriteLine("TestPredictionDriver");
			strw.WriteLine("--------------");
			strw.WriteLine("TestPredictionDriver helps developing higher level tools that need prediction scan.");
			strw.WriteLine("It is intended as a diagnostic and debugging tool.");
			strw.WriteLine("--------------");
			strw.WriteLine("Sample configuration:");
			TestPredictionDriver.Config C = new TestPredictionDriver.Config();
			C.DataSource = DataSource.Random;
			C.RandomEfficiency = 0.9;
			C.RandomPositionRMS = 10.0;
			C.RandomSlopeRMS = 0.006;
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TestPredictionDriver.Config));			
			xmls.Serialize(strw, C);
			strw.WriteLine("");
			strw.WriteLine("--------------");
			strw.WriteLine("DataSource can be " + DataSource.SingleLinkedZone + " or " + DataSource.Random + ".");
			EF.RTFOut.Text = strw.ToString();
			EF.ShowDialog();			
		}

		static System.Text.RegularExpressions.Regex tlgex = new System.Text.RegularExpressions.Regex(@"\s*tlg\s+");

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			HE = SySal.DAQSystem.Drivers.HostEnv.Own;
			if (HE == null)
			{
				ShowExplanation();
				return;
			}
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TestPredictionDriver.Config));
			C = (Config)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
			HE.Progress = 0.0;			
			System.Data.DataSet ds = new System.Data.DataSet();
			StartupInfo = (SySal.DAQSystem.Drivers.ScanningStartupInfo)(HE.StartupInfo);
			DB = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
			DB.Open();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PATH, POSX, POSY, SLOPEX, SLOPEY, FRAME, POSTOL1, POSTOL2, SLOPETOL1, SLOPETOL2 FROM TB_SCANBACK_PREDICTIONS WHERE " + 
				"ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID_PLATE = " + StartupInfo.Plate.PlateId + " AND ID_PATH IN (SELECT ID FROM TB_SCANBACK_PATHS WHERE " + 
				"ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID_PROCESSOPERATION = " + 
				new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_PROC_OPERATIONS PK_PROC_OPERATIONS) */ ID_PARENT_OPERATION FROM TB_PROC_OPERATIONS WHERE ID = " + StartupInfo.ProcessOperationId, DB, null).ExecuteScalar().ToString() + 
				")", DB, null).Fill(ds);
			if (C.DataSource == DataSource.SingleLinkedZone)
			{
				HE.WriteLine("Waiting for a TLG file path to be sent through an interrupt.");
				HE.WriteLine("Interrupt syntax:");
				HE.WriteLine("TLG <filename>");
				HE.WriteLine("The filename can contain white spaces.");
				HE.WriteLine("Examples:");
				HE.WriteLine(@"TLG c:\mydir\plate_01.tlg");
				HE.WriteLine(@"TLG \\mymachine.mydomain.eu\mydir\my nice plate 05.tlg");		
				bool done = false;
				while (!done)
				{
					SySal.DAQSystem.Drivers.Interrupt newint = HE.NextInterrupt;
					if (newint != null)
					{
						System.Text.RegularExpressions.Match m = tlgex.Match(newint.Data.ToLower());
						if (m.Success)
						{
							string tlgpath = newint.Data.Substring(m.Length);
							HE.LastProcessedInterruptId = newint.Id;
							SingleLinkedZone(ds.Tables[0], tlgpath);
							done = true;
						}
						else
						{
							HE.LastProcessedInterruptId = newint.Id;
							System.Threading.Thread.Sleep(1000);
						}
					}
				}
				HE.Complete = true;
				return;
			}
			else if (C.DataSource == DataSource.Random)
			{
				HE.WriteLine("Generating dummy base tracks");
				try
				{
					GenerateRandomScan(ds.Tables[0]);
				}
				catch (Exception x)
				{
					HE.ExitException = x.Message;
					HE.Complete = false;
					return;
				}
			}
			else
			{
				HE.ExitException = "Unknown DataSource mode!";
				return;
			}
			HE.ExitException = "";
			HE.Complete = true;
		}

		static System.Random Rnd = new System.Random();

		static void SingleLinkedZone(System.Data.DataTable dt, string tlgpath)
		{
			if (dt.Columns.Count != 10) throw new Exception("The predictions are expected to be in 10 column format!");
			
			if (String.Compare(dt.Columns[0].ColumnName, "ID_PATH", true) != 0 ||
				String.Compare(dt.Columns[1].ColumnName, "POSX", true) != 0 ||
				String.Compare(dt.Columns[2].ColumnName, "POSY", true) != 0 ||
				String.Compare(dt.Columns[3].ColumnName, "SLOPEX", true) != 0 ||
				String.Compare(dt.Columns[4].ColumnName, "SLOPEY", true) != 0 ||
				String.Compare(dt.Columns[5].ColumnName, "FRAME", true) != 0 ||
				String.Compare(dt.Columns[6].ColumnName, "POSTOL1", true) != 0 ||
				String.Compare(dt.Columns[7].ColumnName, "POSTOL2", true) != 0 ||
				String.Compare(dt.Columns[8].ColumnName, "SLOPETOL1", true) != 0 ||
				String.Compare(dt.Columns[9].ColumnName, "SLOPETOL2", true) != 0)
				throw new Exception("The expected column list is:\nID_PATH POSX POSY SLOPEX SLOPEY FRAME POSTOL1 POSTOL2 SLOPETOL1 SLOPETOL2");

			SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = null;

			try
			{
				lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(tlgpath, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
			}
			catch (Exception x)
			{
				HE.ExitException = x.Message;
				throw x;
			}

			SySal.OperaDb.Schema.DB = DB;

			SySal.OperaDb.OperaDbTransaction trans = DB.BeginTransaction();
 
			int searched, found;

			double Z = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, DB, null).ExecuteScalar());

			searched = dt.Rows.Count;
			found = 0;
			int row;
			for (row = 0; row < dt.Rows.Count; row++)
			{
				System.Data.DataRow dr = dt.Rows[row];
				double px = SySal.OperaDb.Convert.ToDouble(dr[1]);
				double py = SySal.OperaDb.Convert.ToDouble(dr[2]);
				double sx = SySal.OperaDb.Convert.ToDouble(dr[3]);
				double sy = SySal.OperaDb.Convert.ToDouble(dr[4]);				
				SySal.DAQSystem.Frame frame = (dr[5].ToString() == "P") ? SySal.DAQSystem.Frame.Polar : SySal.DAQSystem.Frame.Cartesian;
				double p1 = SySal.OperaDb.Convert.ToDouble(dr[6]);
				double p2 = SySal.OperaDb.Convert.ToDouble(dr[7]);
				double s1 = SySal.OperaDb.Convert.ToDouble(dr[8]);
				double s2 = SySal.OperaDb.Convert.ToDouble(dr[9]);

				double nx, ny, slope;

				System.Collections.ArrayList tksinzone = new System.Collections.ArrayList();

				long zoneid = SySal.OperaDb.Schema.TB_ZONES.Insert(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, 0, px - 100.0, px + 100.0, 
					py - 100.0, py + 100.0, tlgpath, System.DateTime.Now, System.DateTime.Now.AddSeconds(1), SySal.OperaDb.Convert.ToInt64(dr[0]), 1, 0, 0, 1, 0, 0);

				SySal.OperaDb.Schema.TB_VIEWS.Insert(StartupInfo.Plate.BrickId, zoneid, 1, 1, Z + 43.0, Z, px, py);
				SySal.OperaDb.Schema.TB_VIEWS.Insert(StartupInfo.Plate.BrickId, zoneid, 2, 1, Z - 200.0, Z - 243.0, px, py);
				SySal.OperaDb.Schema.TB_VIEWS.Flush();

				int i;
				for (i = 0; i < lz.Length; i++)
				{
					SySal.Scanning.MIPBaseTrack bt = lz[i];
					SySal.Tracking.MIPEmulsionTrackInfo info = bt.Info;
					if (Math.Abs(info.Intercept.X - px) > 100.0 || Math.Abs(info.Intercept.Y - py) > 100.0) continue;
					tksinzone.Add(bt);					
				}
				for (i = 0; i < tksinzone.Count; i++)
				{
					SySal.Scanning.MIPBaseTrack bt = (SySal.Scanning.MIPBaseTrack)(tksinzone[i]);
					SySal.Tracking.MIPEmulsionTrackInfo info = bt.Info;
					SySal.Tracking.MIPEmulsionTrackInfo tinfo = bt.Top.Info;
					SySal.Tracking.MIPEmulsionTrackInfo binfo = bt.Bottom.Info;
					SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, zoneid, 1, tksinzone.Count, tinfo.Intercept.X, tinfo.Intercept.Y, tinfo.Slope.X, tinfo.Slope.Y, 12, 100, System.DBNull.Value, 0.1, 1);
					SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, zoneid, 2, tksinzone.Count, binfo.Intercept.X, binfo.Intercept.Y, binfo.Slope.X, binfo.Slope.Y, 12, 100, System.DBNull.Value, 0.1, 1);
				}
				SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Flush();
				for (i = 0; i < tksinzone.Count; i++)
				{
					SySal.Scanning.MIPBaseTrack bt = (SySal.Scanning.MIPBaseTrack)(tksinzone[i]);
					SySal.Tracking.MIPEmulsionTrackInfo info = bt.Info;
					SySal.OperaDb.Schema.TB_MIPBASETRACKS.Insert(StartupInfo.Plate.BrickId, zoneid, tksinzone.Count, info.Intercept.X, info.Intercept.Y, info.Slope.X, info.Slope.Y, 24, 120, System.DBNull.Value, info.Sigma, 1, tksinzone.Count, 2, tksinzone.Count);
				}				
				SySal.OperaDb.Schema.TB_MIPBASETRACKS.Flush();
				if (tksinzone.Count == 0)
					SySal.OperaDb.Schema.PC_SCANBACK_NOCANDIDATE.Call(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, SySal.OperaDb.Convert.ToInt64(dr[0]), zoneid);
				else
				{
					int bestcand = -1;
					double bestd = s1;
					double destd = s1;
					for (i = 0; i < tksinzone.Count; i++)
					{
						SySal.Scanning.MIPBaseTrack bt = (SySal.Scanning.MIPBaseTrack)(tksinzone[i]);
						SySal.Tracking.MIPEmulsionTrackInfo info = bt.Info;
						switch(frame)
						{
							case SySal.DAQSystem.Frame.Cartesian:
							{
								if (Math.Abs(px - info.Intercept.X) > p1) continue;
								if (Math.Abs(py - info.Intercept.Y) > p2) continue;
								if (Math.Abs(sx - info.Slope.X) > s1) continue;
								if (Math.Abs(sy - info.Slope.Y) > s2) continue;
								destd = Math.Sqrt((info.Slope.X - sx) * (info.Slope.X - sx) + (info.Slope.Y - sy) * (info.Slope.Y - sy));
								break;
							}
							case SySal.DAQSystem.Frame.Polar:
							{
								slope = Math.Sqrt(sx * sx + sy * sy);
								nx = sx / slope;
								ny = sy / slope;
								if ((destd = Math.Abs(Math.Abs(px - info.Intercept.X) * ny - Math.Abs(py - info.Intercept.Y) * nx)) > p1) continue;
								if (Math.Abs(Math.Abs(px - info.Intercept.X) * nx + Math.Abs(py - info.Intercept.Y) * ny) > p2) continue;
								if (Math.Abs(Math.Abs(sx - info.Intercept.X) * ny - Math.Abs(sy - info.Slope.Y) * nx) > s1) continue;
								if (Math.Abs(Math.Abs(sx - info.Intercept.X) * nx + Math.Abs(sy - info.Slope.Y) * ny) > s2) continue;								
								break;
							}
						}
						if (destd < bestd)
						{
							bestd = destd;
							bestcand = i;
						}
					}
					if (bestd >= 0)
					{
						SySal.OperaDb.Schema.PC_SCANBACK_CANDIDATE.Call(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, bestcand + 1, zoneid, 1, 0);
						found++;
					}
					else SySal.OperaDb.Schema.PC_SCANBACK_NOCANDIDATE.Call(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, bestcand + 1, zoneid);
					HE.Progress = ((double)row)/(dt.Rows.Count);
				}
				
			}	
			trans.Commit();		
			HE.WriteLine("Searched = " + searched + ", Found = " + found + ", Eff = " + ((double)found/(double)searched * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
		}

		static void GenerateRandomScan(System.Data.DataTable dt)
		{
			if (dt.Columns.Count != 10) throw new Exception("The predictions are expected to be in 10 column format!");
			
			if (String.Compare(dt.Columns[0].ColumnName, "ID_PATH", true) != 0 ||
				String.Compare(dt.Columns[1].ColumnName, "POSX", true) != 0 ||
				String.Compare(dt.Columns[2].ColumnName, "POSY", true) != 0 ||
				String.Compare(dt.Columns[3].ColumnName, "SLOPEX", true) != 0 ||
				String.Compare(dt.Columns[4].ColumnName, "SLOPEY", true) != 0 ||
				String.Compare(dt.Columns[5].ColumnName, "FRAME", true) != 0 ||
				String.Compare(dt.Columns[6].ColumnName, "POSTOL1", true) != 0 ||
				String.Compare(dt.Columns[7].ColumnName, "POSTOL2", true) != 0 ||
				String.Compare(dt.Columns[8].ColumnName, "SLOPETOL1", true) != 0 ||
				String.Compare(dt.Columns[9].ColumnName, "SLOPETOL2", true) != 0)
				throw new Exception("The expected column list is:\nID_PATH POSX POSY SLOPEX SLOPEY FRAME POSTOL1 POSTOL2 SLOPETOL1 SLOPETOL2");

			SySal.OperaDb.Schema.DB = DB;

			SySal.OperaDb.OperaDbTransaction trans = DB.BeginTransaction();
 
			int searched, found;

			double Z = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, DB, null).ExecuteScalar());

			searched = dt.Rows.Count;
			found = 0;
			foreach (System.Data.DataRow dr in dt.Rows)
			{
				double px = SySal.OperaDb.Convert.ToDouble(dr[1]);
				double py = SySal.OperaDb.Convert.ToDouble(dr[2]);
				double sx = SySal.OperaDb.Convert.ToDouble(dr[3]);
				double sy = SySal.OperaDb.Convert.ToDouble(dr[4]);
				double fx = px + (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomPositionRMS;
				double fy = py + (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomPositionRMS;				
				double f2x, f2y;
				long zoneid = SySal.OperaDb.Schema.TB_ZONES.Insert(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, 0, px - 100.0, px + 100.0, 
					py - 100.0, py + 100.0, "null.tlg", System.DateTime.Now, System.DateTime.Now.AddSeconds(1), SySal.OperaDb.Convert.ToInt64(dr[0]), 1, 0, 0, 1, 0, 0);

				SySal.OperaDb.Schema.TB_VIEWS.Insert(StartupInfo.Plate.BrickId, zoneid, 1, 1, Z + 43.0, Z, px, py);
				SySal.OperaDb.Schema.TB_VIEWS.Insert(StartupInfo.Plate.BrickId, zoneid, 2, 1, Z - 200.0, Z - 243.0, px, py);

				if (Rnd.NextDouble() <= C.RandomEfficiency)
				{
					SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, zoneid, 1, 1, fx, fy, sx + (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomSlopeRMS, sy + (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomSlopeRMS, 10, 100, System.DBNull.Value, 0.1, 1);
					SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, zoneid, 2, 1, f2x = (fx - 200 * (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomSlopeRMS), f2y = (fy - 200 * (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomSlopeRMS), sx + (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomSlopeRMS, sy + (Rnd.NextDouble() * 2.0 - 1.0) * C.RandomSlopeRMS, 10, 100, System.DBNull.Value, 0.1, 1);
					SySal.OperaDb.Schema.TB_MIPBASETRACKS.Insert(StartupInfo.Plate.BrickId, zoneid, 1, fx, fy, (fx - f2x) / 200.0, (fy - f2y) / 200.0, 20, 200, System.DBNull.Value, .1, 1, 1, 2, 1);
					SySal.OperaDb.Schema.TB_VIEWS.Flush();
					SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Flush();
					SySal.OperaDb.Schema.TB_MIPBASETRACKS.Flush();
					SySal.OperaDb.Schema.PC_SCANBACK_CANDIDATE.Call(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, SySal.OperaDb.Convert.ToInt64(dr[0]), zoneid, 1, 0);
					found++;
				}
				else
				{
					SySal.OperaDb.Schema.PC_SCANBACK_NOCANDIDATE.Call(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, SySal.OperaDb.Convert.ToInt64(dr[0]), zoneid);
				}
			}	
			trans.Commit();		
			HE.WriteLine("Searched = " + searched + ", Found = " + found + ", Eff = " + ((double)found/(double)searched * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
		}		
	}	
}
