using System;
using System.Collections;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using System.Xml;
using System.Xml.Serialization;
using SySal;
using SySal.BasicTypes;
using SySal.Scanning.PostProcessing.PatternMatching;
using SySal.DAQSystem.Drivers;

namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    /// <summary>
    /// Settings from CSScanDriver.
    /// </summary>
    [Serializable]
	public class CSScanDriverSettings
	{
        /// <summary>
        /// Position tolerance X.
        /// </summary>
        public double PositionToleranceX;
        /// <summary>
        /// Position tolerance Y.
        /// </summary>
        public double PositionToleranceY;
        /// <summary>
        /// Slope tolerance X.
        /// </summary>
        public double SlopeToleranceX;
        /// <summary>
        /// Slope tolerance Y.
        /// </summary>
        public double SlopeToleranceY;
        /// <summary>
        /// If true skip measured areas.
        /// </summary>
        public bool SkipMeasuredAreas = false;
        /// <summary>
        /// Program settings Id for the Wide Area Scanning driver.
        /// </summary>
        public long WideAreaConfigId;
        /// <summary>
        /// Program settings Id for the CSMap.
        /// </summary>
        public long CSMapConfigId;
        /// <summary>
        /// If set to <c>true</c>, the driver starts in <c>Halted</c> state, 
        /// waiting for a <c>Zone</c> interrupt. 
        /// </summary>
        public bool WaitForScanningArea = false;
    }
	/// <summary>
	/// Main executor.
	/// </summary>
    public class Exe : MarshalByRefObject, IInterruptNotifier
	{
		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>the lifetime service object or null.</returns>
		public override object InitializeLifetimeService()
		{
			return null;	
		}

		static void ShowExplanation()
		{
			ExplanationForm EF = new ExplanationForm();
			System.IO.StringWriter strw = new System.IO.StringWriter();
			strw.WriteLine("");
			strw.WriteLine("CSScanDriver");
			strw.WriteLine("--------------");
			strw.WriteLine("CSScanDriver scans a Plate doublet");
			strw.WriteLine();
            //strw.WriteLine("Type: CSScanDriver /Interrupt <batchmanager> <process operation id> <interrupt string>");
            //strw.WriteLine("to send an interrupt message to a running CSScanDriver process operation.");
            //strw.WriteLine("SUPPORTED INTERRUPTS:");
            //strw.WriteLine("IgnoreScanFailure False|True - instructs CSScanDriver to stop on failed zones or skip them and go on.");
            //strw.WriteLine("IgnoreRecalFailure False|True - instructs CSScanDriver to stop on failed recalibration tracks or skip them and go on.");
            //strw.WriteLine("Type: CSScanDriver /EasyInterrupt for a graphical user interface to send interrupts.");			
            //strw.WriteLine("--------------");
            //strw.WriteLine("The following substitutions apply (case is disregarded):");
            //strw.WriteLine("%EXEREP% = Executable repository path specified in the Startup file.");
            //strw.WriteLine("%RWDDIR% = Output directory for Raw Data.");
            //strw.WriteLine("%TLGDIR% = Output directory for linked zones.");
            //strw.WriteLine("%RWD% = Scanning output file name (not including extension).");
            //strw.WriteLine("%TLG% = Linked zone file name (not including extension).");
            //strw.WriteLine("%SCRATCH% = Scratch directory specified in the Startup file.");			
            //strw.WriteLine("%ZONEID% = Hexadecimal file name for a zone.");
            //strw.WriteLine("--------------");
			strw.WriteLine("The program settings should have the following structure:");
			CSScanDriverSettings pset = new CSScanDriverSettings();
			new System.Xml.Serialization.XmlSerializer(typeof(CSScanDriverSettings)).Serialize(strw, pset);
            strw.WriteLine("");
            EF.RTFOut.Text = strw.ToString();
            EF.ShowDialog();			
		}
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
        [STAThread]
        internal static void Main(string[] args)
        {
            if (args.Length == 1)
            {
                if (args[0].ToLower() == "/easyschedule")
                {
                    EasySchedule();
                    return;
                }
                else if (args[0].ToLower() == "/easyconfig")
                {
                    return;
                }
            }

            HE = SySal.DAQSystem.Drivers.HostEnv.Own;
            if (HE == null)
            {
                if (args.Length == 1 && String.Compare(args[0].Trim().ToLower(), "/easyinterrupt", true) == 0) EasyInterrupt();
                else ShowExplanation();
                return;
            }
            Execute();
        }

        private static void EasySchedule()
        {
            new frmScheduleCSScanDriver().ShowDialog();
            return;
        }

        private static long WaitingOnId;

        private static bool CS1WideAreaScanDone;

        private static bool CS2WideAreaScanDone;

        private static long CS1WideAreaScanProcOperationId;

        private static long CS2WideAreaScanProcOperationId;
        		
		private static bool CSMappingDone;
		
		private static bool CSCandidatesDone;

        private static bool ComputeScanArea = true;

        private static double MinX = 0;

        private static double MaxX = 0;

        private static double MinY = 0;

        private static double MaxY = 0;

        private static SySal.DAQSystem.Drivers.HostEnv HE = null;

		private static SySal.OperaDb.OperaDbConnection Conn = null;

		private static CSScanDriverSettings ProgSettings;

		private static SySal.DAQSystem.Drivers.VolumeOperationInfo StartupInfo;

		private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = null;

        private static SySal.DAQSystem.IDataProcessingServer DataProcSrv;

		private static System.Threading.Thread ThisThread = null;

        private static System.Exception ThisException = null;

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
                    System.Threading.Thread.Sleep(5 * 60 * 1000); //5 minutes
                }
            }
            catch (System.Threading.ThreadAbortException)
            {
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception) { }
        }

        private static void EasyInterrupt()
        {
            (new frmEasyInterrupt()).ShowDialog();
        }
        
        private static void Execute()
        {
            try
            {
                //System.Windows.Forms.MessageBox.Show("Eccomi");
                ThisThread = System.Threading.Thread.CurrentThread;

                StartupInfo = (SySal.DAQSystem.Drivers.VolumeOperationInfo)HE.StartupInfo;
                Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                Conn.Open();
                (DBKeepAliveThread = new System.Threading.Thread(DBKeepAliveThreadExec)).Start();

                System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(CSScanDriverSettings));
                ProgSettings = (CSScanDriverSettings)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
                xmls = null;

                if (StartupInfo.ExeRepository.EndsWith("\\")) StartupInfo.ExeRepository = StartupInfo.ExeRepository.Remove(StartupInfo.ExeRepository.Length - 1, 1);
                if (StartupInfo.ScratchDir.EndsWith("\\")) StartupInfo.ScratchDir = StartupInfo.ScratchDir.Remove(StartupInfo.ScratchDir.Length - 1, 1);
                if (StartupInfo.LinkedZonePath.EndsWith("\\")) StartupInfo.LinkedZonePath = StartupInfo.LinkedZonePath.Remove(StartupInfo.LinkedZonePath.Length - 1, 1);
                if (StartupInfo.RawDataPath.EndsWith("\\")) StartupInfo.RawDataPath = StartupInfo.RawDataPath.Remove(StartupInfo.RawDataPath.Length - 1, 1);

                //create a directory where to put all acquisition files
                if (StartupInfo.RawDataPath.IndexOf(System.Convert.ToString(StartupInfo.ProcessOperationId)) < 0)
                {
                    StartupInfo.RawDataPath = StartupInfo.RawDataPath + "\\cssd_" + StartupInfo.BrickId + "_" + StartupInfo.ProcessOperationId;
                    if (!System.IO.Directory.Exists(StartupInfo.RawDataPath)) System.IO.Directory.CreateDirectory(StartupInfo.RawDataPath);
                }

                ProgressInfo = HE.ProgressInfo;
                if (StartupInfo.RecoverFromProgressFile)
                {
                    try
                    {
                        XmlDocument xmldoc = new XmlDocument();
                        xmldoc.LoadXml(ProgressInfo.CustomInfo.Replace('[', '<').Replace(']', '>'));
                        System.Xml.XmlNode xmlprog = xmldoc.FirstChild;

                        WaitingOnId = Convert.ToInt64(xmlprog["WaitingOnId"].InnerText);
                        CS1WideAreaScanDone = Convert.ToBoolean(xmlprog["CS1WideAreaScanDone"].InnerText);
                        CS2WideAreaScanDone = Convert.ToBoolean(xmlprog["CS2WideAreaScanDone"].InnerText);
                        CS1WideAreaScanProcOperationId = Convert.ToInt64(xmlprog["CS1WideAreaScanProcOperationId"].InnerText);
                        CS2WideAreaScanProcOperationId = Convert.ToInt64(xmlprog["CS2WideAreaScanProcOperationId"].InnerText);
                        CSMappingDone = Convert.ToBoolean(xmlprog["CSMappingDone"].InnerText);
                        CSCandidatesDone = Convert.ToBoolean(xmlprog["CSCandidatesDone"].InnerText);
                        ComputeScanArea = Convert.ToBoolean(xmlprog["ComputeScanArea"].InnerText);
                        MinX = Convert.ToDouble(xmlprog["MinX"].InnerText, System.Globalization.CultureInfo.InvariantCulture);
                        MaxX = Convert.ToDouble(xmlprog["MaxX"].InnerText, System.Globalization.CultureInfo.InvariantCulture);
                        MinY = Convert.ToDouble(xmlprog["MinY"].InnerText, System.Globalization.CultureInfo.InvariantCulture);
                        MaxY = Convert.ToDouble(xmlprog["MaxY"].InnerText, System.Globalization.CultureInfo.InvariantCulture);

                        ProgressInfo.ExitException = null;
                        HE.WriteLine("Restarting complete");
                    }
                    catch (Exception ex)
                    {
                        HE.WriteLine("Restarting failed - proceeding to re-initialize process.");
                        ProgressInfo = HE.ProgressInfo;
                        ProgressInfo.Progress = 0.0;
                        ProgressInfo.StartTime = System.DateTime.Now;
                        ProgressInfo.FinishTime = ProgressInfo.StartTime.AddYears(1);
                        HE.WriteLine(ex.Message);
                    }
                }
                else
                {
                    CS1WideAreaScanDone = false;
                    CS2WideAreaScanDone = false;
                    CS1WideAreaScanProcOperationId = 0;
                    CS2WideAreaScanProcOperationId = 0;
                    CSMappingDone = false;
                    CSCandidatesDone = false;
                    ComputeScanArea = true;
                    ProgressInfo = new TaskProgressInfo();
                    ProgressInfo.Complete = false;
                    ProgressInfo.ExitException = null;
                    ProgressInfo.Progress = 0.0;
                    ProgressInfo.StartTime = System.DateTime.Now;
                    ProgressInfo.FinishTime = ProgressInfo.StartTime.AddYears(1);
                }

                UpdateProgress();

                HE.InterruptNotifier = new Exe();

                if (ProgSettings.WaitForScanningArea)
                {
                    while (ComputeScanArea == true)
                    {
                        System.Threading.Thread.Sleep(1000);
                    }
                }

                UpdateProgress();

                for (int i = 0; i < 2; i++)
                {
                    bool WideAreaScanDone = (i == 0) ? CS1WideAreaScanDone : CS2WideAreaScanDone;

                    while (WideAreaScanDone == false)
                    {
                        if (WaitingOnId == 0)
                        {
                            SySal.DAQSystem.Drivers.ScanningStartupInfo wasdstartupinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
                            wasdstartupinfo.DBPassword = StartupInfo.DBPassword;
                            wasdstartupinfo.DBServers = StartupInfo.DBServers;
                            wasdstartupinfo.DBUserName = StartupInfo.DBUserName;
                            wasdstartupinfo.ExeRepository = StartupInfo.ExeRepository;
                            wasdstartupinfo.LinkedZonePath = StartupInfo.LinkedZonePath;
                            wasdstartupinfo.MachineId = StartupInfo.MachineId;
                            wasdstartupinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
                            wasdstartupinfo.Plate.BrickId = StartupInfo.BrickId;
                            wasdstartupinfo.Plate.PlateId = i + 1;
                            long calibrationId;
                            wasdstartupinfo.MarkSet = MarkType.SpotXRay;
                            wasdstartupinfo.Plate.MapInitString = SySal.OperaDb.Scanning.Utilities.GetMapString(wasdstartupinfo.Plate.BrickId, wasdstartupinfo.Plate.PlateId, false, MarkType.SpotXRay, out calibrationId, Conn, null);
                            wasdstartupinfo.Plate.TextDesc = "Brick #" + wasdstartupinfo.Plate.BrickId + " Plate #" + wasdstartupinfo.Plate.PlateId;
                            wasdstartupinfo.ProcessOperationId = 0;
                            wasdstartupinfo.ProgramSettingsId = ProgSettings.WideAreaConfigId;
                            wasdstartupinfo.ProgressFile = "";
                            wasdstartupinfo.RawDataPath = StartupInfo.RawDataPath;
                            wasdstartupinfo.RecoverFromProgressFile = false;
                            wasdstartupinfo.ScratchDir = StartupInfo.ScratchDir;
                            if (ComputeScanArea == true)
                                wasdstartupinfo.Zones = CalculateZones(wasdstartupinfo.Plate.BrickId, wasdstartupinfo.Plate.PlateId);
                            else
                            {
                                wasdstartupinfo.Zones = new SySal.DAQSystem.Scanning.ZoneDesc[1];
                                wasdstartupinfo.Zones[0] = new SySal.DAQSystem.Scanning.ZoneDesc();
                                wasdstartupinfo.Zones[0].MinX = MinX;
                                wasdstartupinfo.Zones[0].MaxX = MaxX;
                                wasdstartupinfo.Zones[0].MinY = MinY;
                                wasdstartupinfo.Zones[0].MaxY = MaxY;
                            }
                            WaitingOnId = HE.Start(wasdstartupinfo);
                            UpdateProgress();

                            HE.WriteLine("Starting wasd " + WaitingOnId);
                        }

                        SySal.DAQSystem.Drivers.Status status;
                        if (WaitingOnId != 0)
                        {
                            status = HE.Wait(WaitingOnId);
                            if (status == SySal.DAQSystem.Drivers.Status.Failed)
                            {
                                WaitingOnId = 0;
                                throw new Exception("Scan of the changeable sheet " + (i + 1) + " failed!\n");
                            }

                            HE.WriteLine("Waiting wasd " + WaitingOnId);
                        }

                        if (WaitingOnId != 0)
                        {
                            status = HE.GetStatus(WaitingOnId);
                            if (status == SySal.DAQSystem.Drivers.Status.Completed)
                            {
                                if (i == 0)
                                {
                                    CS1WideAreaScanDone = true;
                                    CS1WideAreaScanProcOperationId = WaitingOnId;
                                }
                                else
                                {
                                    CS2WideAreaScanDone = true;
                                    CS2WideAreaScanProcOperationId = WaitingOnId;
                                }
                                ProgressInfo.Progress = 0.45 * (i + 1);
                                WaitingOnId = 0;
                                break;
                            }

                            HE.WriteLine("Checking status wasd " + WaitingOnId);
                        }
                        UpdateProgress();
                    }
                }

                if (CSMappingDone == false)
                {
                    HE.WriteLine("Start mapping");

                    DataProcSrv = HE.DataProcSrv;

                    SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
                    dbd.AliasUsername = StartupInfo.DBUserName;
                    dbd.AliasPassword = StartupInfo.DBPassword;
                    dbd.Description = "Plate doublet mapping B#" + StartupInfo.BrickId;
                    dbd.Id = DataProcSrv.SuggestId;
                    dbd.Token = HE.Token;
                    dbd.MachinePowerClass = 5;
                    dbd.Filename = StartupInfo.ExeRepository + @"\CSMap.exe";
//                    dbd.CommandLineArguments = StartupInfo.ProcessOperationId + " db:\\" + ProgSettings.CSMapConfigId + ".xml CSMap_" + StartupInfo.BrickId + "_" + StartupInfo.ProcessOperationId + " false " + StartupInfo.ProcessOperationId;
                    dbd.CommandLineArguments =
                        "db:\\" + CS1WideAreaScanProcOperationId + ".tlg " +
                        "db:\\" + CS2WideAreaScanProcOperationId + ".tlg " +
                        "db:\\" + StartupInfo.BrickId + " " +
                        "db:\\" + ProgSettings.CSMapConfigId + ".xml " +
                        StartupInfo.RawDataPath + @"\CSSD " +
                        StartupInfo.ProcessOperationId;

                    if (!DataProcSrv.Enqueue(dbd)) throw new Exception("Cannot schedule CSd mapping batch " + dbd.Id + " for brick " + StartupInfo.BrickId + ". Aborting.");
                    while (DataProcSrv.DoneWith(dbd.Id) == false) System.Threading.Thread.Sleep(100);
                    dbd = DataProcSrv.Result(dbd.Id);

                    if (System.IO.File.Exists(StartupInfo.RawDataPath + @"\CHECK") == true)
                        CSMappingDone = true;
                    else
                    {
                        throw new Exception("Plate Mapping failure");
                    }
                }

                //DUMP MARK FILE
                System.IO.StreamWriter markFile = null;
                try
                {
                    long CalibrationId;
                    string markString = SySal.OperaDb.Scanning.Utilities.GetMapString(StartupInfo.BrickId, 1, false, MarkType.SpotXRay, out CalibrationId, Conn, null);
                    markFile = new System.IO.StreamWriter(StartupInfo.RawDataPath + @"\marks.txt");
                    markFile.WriteLine(markString);
                    markFile.Flush();
                }
                catch { }
                finally
                {
                    if (markFile != null) markFile.Close();
                }

                WaitingOnId = 0;
                ProgressInfo.Progress = 1.0;
                ProgressInfo.Complete = true;
                UpdateProgress();
            }
            catch (Exception ex)
            {
                if (ProgressInfo == null) ProgressInfo = new SySal.DAQSystem.Drivers.TaskProgressInfo();
                HE.WriteLine(ex.Message);
                ProgressInfo.Complete = false;
                ProgressInfo.ExitException = ex.ToString();
                UpdateProgress();
            }
        }
        
        private static void UpdateProgress()
		{
            string xmlstr = "\r\n\t\t[InfoContainer]\r\n\t\t\t[WaitingOnId]" + WaitingOnId + "[/WaitingOnId]" +
                "\r\n\t\t\t[CS1WideAreaScanDone]" + CS1WideAreaScanDone + "[/CS1WideAreaScanDone]" +
                "\r\n\t\t\t[CS1WideAreaScanProcOperationId]" + CS1WideAreaScanProcOperationId + "[/CS1WideAreaScanProcOperationId]" +
                "\r\n\t\t\t[CS2WideAreaScanDone]" + CS2WideAreaScanDone + "[/CS2WideAreaScanDone]" +
                "\r\n\t\t\t[CS2WideAreaScanProcOperationId]" + CS2WideAreaScanProcOperationId + "[/CS2WideAreaScanProcOperationId]" +
                "\r\n\t\t\t[CSMappingDone]" + CSMappingDone + "[/CSMappingDone]" +
                "\r\n\t\t\t[CSCandidatesDone]" + CSCandidatesDone + "[/CSCandidatesDone]" +
                "\r\n\t\t\t[ComputeScanArea]" + ComputeScanArea + "[/ComputeScanArea]" +
                "\r\n\t\t\t[MinX]" + MinX + "[/MinX]" +
                "\r\n\t\t\t[MaxX]" + MaxX + "[/MaxX]" +
                "\r\n\t\t\t[MinY]" + MinY + "[/MinY]" +
                "\r\n\t\t\t[MaxY]" + MaxY + "[/MaxY]";
            xmlstr += "\r\n\t\t[/InfoContainer]\r\n";

			ProgressInfo.CustomInfo = xmlstr;
			HE.ProgressInfo = ProgressInfo;
		}

        public static SySal.DAQSystem.Scanning.ZoneDesc[] CalculateZones(long brickid, long plateid)
        {
            System.Data.DataSet ds = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("select minx-zerox, maxx-zerox, miny-zeroy, maxy-zeroy from tb_eventbricks where id = " + brickid, Conn).Fill(ds);

            double local_brick_minx = Convert.ToDouble(ds.Tables[0].Rows[0][0], System.Globalization.CultureInfo.InvariantCulture);
            double local_brick_maxx = Convert.ToDouble(ds.Tables[0].Rows[0][1], System.Globalization.CultureInfo.InvariantCulture);
            double local_brick_miny = Convert.ToDouble(ds.Tables[0].Rows[0][2], System.Globalization.CultureInfo.InvariantCulture);
            double local_brick_maxy = Convert.ToDouble(ds.Tables[0].Rows[0][3], System.Globalization.CultureInfo.InvariantCulture);

            System.Collections.ArrayList newZoneArray = new ArrayList();
            string EventType = Convert.ToString(new SySal.OperaDb.OperaDbCommand("select DISTINCT type from vw_local_predictions where id_cs_eventbrick = " + brickid, Conn).ExecuteScalar());

            //TODO: improve this part: multi-zone for CC-like events

            if (EventType.Equals("CC") || EventType.Equals("NC"))
            {
                ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("select pred_localx, pred_localy, POSTOL1, POSTOL2 from vw_local_predictions where id_cs_eventbrick = " + brickid, Conn).Fill(ds);

                double posx, posy, postolx, postoly;
                SySal.DAQSystem.Scanning.ZoneDesc zone = new SySal.DAQSystem.Scanning.ZoneDesc();
                posx = Convert.ToDouble(ds.Tables[0].Rows[0][0], System.Globalization.CultureInfo.InvariantCulture);
                posy = Convert.ToDouble(ds.Tables[0].Rows[0][1], System.Globalization.CultureInfo.InvariantCulture);
                postolx = (ds.Tables[0].Rows[0][2] != DBNull.Value) ? Convert.ToDouble(ds.Tables[0].Rows[0][2], System.Globalization.CultureInfo.InvariantCulture) : ProgSettings.PositionToleranceX;
                postoly = (ds.Tables[0].Rows[0][3] != DBNull.Value) ? Convert.ToDouble(ds.Tables[0].Rows[0][3], System.Globalization.CultureInfo.InvariantCulture) : ProgSettings.PositionToleranceY;

                zone.MinX = posx - postolx;
                zone.MaxX = posx + postolx;
                zone.MinY = posy - postoly;
                zone.MaxY = posy + postoly;

                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                {
                    posx = Convert.ToDouble(dr[0], System.Globalization.CultureInfo.InvariantCulture);
                    posy = Convert.ToDouble(dr[1], System.Globalization.CultureInfo.InvariantCulture);
                    postolx = (dr[2] != DBNull.Value) ? Convert.ToDouble(dr[2], System.Globalization.CultureInfo.InvariantCulture) : ProgSettings.PositionToleranceX;
                    postoly = (dr[3] != DBNull.Value) ? Convert.ToDouble(dr[3], System.Globalization.CultureInfo.InvariantCulture) : ProgSettings.PositionToleranceY;

                    zone.MinX = Math.Min(zone.MinX, posx - postolx);
                    zone.MaxX = Math.Max(zone.MaxX, posx + postolx);
                    zone.MinY = Math.Min(zone.MinY, posy - postoly);
                    zone.MaxY = Math.Max(zone.MaxY, posy + postoly);
                }
                newZoneArray.Add(zone);
            }

            foreach (SySal.DAQSystem.Scanning.ZoneDesc z in newZoneArray)
            {
                if (z.MinX < local_brick_minx) z.MinX = local_brick_minx;
                if (z.MaxX > local_brick_maxx) z.MaxX = local_brick_maxx;
                if (z.MinY < local_brick_miny) z.MinY = local_brick_miny;
                if (z.MaxY > local_brick_maxy) z.MaxY = local_brick_maxy;
            }

            System.Collections.ArrayList ZoneArray = null;

            if (ProgSettings.SkipMeasuredAreas == true)
            {
                ZoneArray = new ArrayList();

                ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT a.id_processoperation, MIN(a.minx) as minx, MAX(a.maxx) as maxx, MIN(a.miny) as miny, MAX(a.maxy) as maxy FROM tb_zones a INNER JOIN tb_proc_operations b ON(b.id = a.id_processoperation) WHERE a.id_eventbrick = " + brickid + " AND b.id_plate = " + plateid + " AND b.success = 'Y' GROUP BY a.id_processoperation", Conn).Fill(ds);

                if (ds.Tables[0].Rows.Count > 0)
                {
                    SySal.DAQSystem.Scanning.ZoneDesc[] oldZones = new SySal.DAQSystem.Scanning.ZoneDesc[ds.Tables[0].Rows.Count];
                    for (int i = 0; i < ds.Tables[0].Rows.Count; i++)
                    {
                        oldZones[i] = new SySal.DAQSystem.Scanning.ZoneDesc();
                        oldZones[i].Series = Convert.ToInt64(ds.Tables[0].Rows[0][0]);
                        oldZones[i].MinX = Convert.ToDouble(ds.Tables[0].Rows[0][1], System.Globalization.CultureInfo.InvariantCulture);
                        oldZones[i].MaxX = Convert.ToDouble(ds.Tables[0].Rows[0][2], System.Globalization.CultureInfo.InvariantCulture);
                        oldZones[i].MinY = Convert.ToDouble(ds.Tables[0].Rows[0][3], System.Globalization.CultureInfo.InvariantCulture);
                        oldZones[i].MaxY = Convert.ToDouble(ds.Tables[0].Rows[0][4], System.Globalization.CultureInfo.InvariantCulture);
                    }

                    foreach (SySal.DAQSystem.Scanning.ZoneDesc n in newZoneArray)
                    {
                        foreach (SySal.DAQSystem.Scanning.ZoneDesc o in oldZones)
                        {
                        }

                    }
                }
            }
            else
                ZoneArray = newZoneArray;

            return (SySal.DAQSystem.Scanning.ZoneDesc[])ZoneArray.ToArray(typeof(SySal.DAQSystem.Scanning.ZoneDesc));
        }

        //minx maxx miny maxy
        private static System.Text.RegularExpressions.Regex ZoneEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*");

#region IInterruptNotifier Members

        public void NotifyInterrupt(Interrupt nextint)
        {
            lock (StartupInfo)
            {
                try
                {
                    if (nextint.Data != null && nextint.Data.Length > 0)
                    {
                        string[] lines = nextint.Data.Split(',');
                        foreach (string line in lines)
                        {
                            System.Text.RegularExpressions.Match mz = ZoneEx.Match(line);
                            if (mz.Success == true && ComputeScanArea == true)
                            {
                                MinX = Convert.ToDouble(mz.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                                MaxX = Convert.ToDouble(mz.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                                MinY = Convert.ToDouble(mz.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                                MaxY = Convert.ToDouble(mz.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                                ComputeScanArea = false;
                            }
                        }
                    }
                    HE.LastProcessedInterruptId = nextint.Id;
                }
                catch (Exception x)
                {
                    HE.WriteLine("Error processing interrupt: " + x.Message);
                }
                HE.LastProcessedInterruptId = nextint.Id;
            }
        }
#endregion
    }
}
