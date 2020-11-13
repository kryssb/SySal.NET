using System;
using System.Collections;
using System.Runtime.Serialization;
using System.Xml;
using System.Xml.Serialization;
using System.Data;
using System.Data.Common;
using ZoneStatus;

namespace SySal.DAQSystem.Drivers.WideAreaScanDriver
{
    [Serializable]
    public class WideAreaScanSettings
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
        /// Apply shrinkage correction.
        /// </summary>
        public bool CorrectShrinkage = false;
        /// <summary>
        /// The Id for quality cut.
        /// </summary>
        public long QualityCutId;
        /// <summary>
        /// Fiducial volume: to take into account for the vacuum channel
        /// </summary>
        public double FiducialVolume = 3000;
        /// <summary>
        /// X strip width.
        /// </summary>
        public double XStep;
        /// <summary>
        /// Y strip width.
        /// </summary>
        public double YStep;
        /// <summary>
        /// X Superposition among different strips
        /// </summary>
        public double XOverLapping;
        /// <summary>
        /// Y Superposition among different strips
        /// </summary>
        public double YOverLapping;
        /// <summary>
        /// Maximum number of trials to search the candidate.
        /// </summary>
        public uint MaxTrials;
        /// <summary>
        /// Minimum density of base-tracks.
        /// </summary>
        public double MinDensityBase;
        /// <summary>
        /// Maximum percentage empty views per strip.
        /// </summary>
        public double MaxPercentageEmptyViews;
    }
    /// <summary>
    /// Strip comparer to make the snake path.
    /// </summary>
    public class StripComparer : System.Collections.IComparer
    {
        int IComparer.Compare(Object x, Object y)
        {
            SySal.DAQSystem.Drivers.Prediction strip1 = (SySal.DAQSystem.Drivers.Prediction)x;
            SySal.DAQSystem.Drivers.Prediction strip2 = (SySal.DAQSystem.Drivers.Prediction)y;
            return System.Convert.ToInt32(strip1.Series - strip2.Series);
        }
    }
    
    public class Exe : MarshalByRefObject, IInterruptNotifier
    {
        private static System.Exception ThisException = null;

        private static SySal.DAQSystem.Drivers.HostEnv HE = null;

        private static SySal.OperaDb.OperaDbConnection Conn = null;

        private static SySal.OperaDb.OperaDbTransaction Trans = null;

        private static WideAreaScanSettings ProgSettings;

        private static string QualityCut;

        private static string LinkConfigPath;

        private static SySal.DAQSystem.Drivers.ScanningStartupInfo StartupInfo;

        private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = null;

        private static SySal.DAQSystem.ScanServer ScanSrv;

        private static SySal.DAQSystem.IDataProcessingServer DataProcSrv;

        private static string StartupFile = null;

        private static string ProgressFile = null;

        private static int TotalStrips;

        private static int TotalCompletedStrips;

        private static System.Collections.ArrayList TotalStripList = new System.Collections.ArrayList();

        private static System.Collections.ArrayList ScanQueue = new System.Collections.ArrayList();

        private static StripStatusArrayClass StripStatusArray = null;

        private static System.Threading.Thread ThisThread = null;

        private static System.Threading.Thread WorkerThread = new System.Threading.Thread(new System.Threading.ThreadStart(WorkerThreadExec));

        private static System.Threading.Thread DBKeepAliveThread = null;

        private static System.Collections.Queue WorkQueue = new System.Collections.Queue();

        private static double DefaultEmuThickness = 44.0;

        //        private static int MaxProcessingQueue = 10000;

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
                        PostProcess();
                        lock (WorkQueue) qc = WorkQueue.Count;
                    }
                }
            }
            catch (System.Threading.ThreadAbortException x)
            {
                HE.WriteLine("ThreadAbortException: " + x.Message);
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception x)
            {
                HE.WriteLine("WorkerThreadExec " + x.Message);
            }
        }

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

        static void ShowExplanation()
        {
            ExplanationForm EF = new ExplanationForm();
            System.IO.StringWriter strw = new System.IO.StringWriter();
            strw.WriteLine("");
            strw.WriteLine("WideAreaScanDriver");
            strw.WriteLine("--------------");
            strw.WriteLine("WideAreaScanDriver scans a CS");
            strw.WriteLine("--------------");
            strw.WriteLine("The program settings should have the following structure:");
            WideAreaScanSettings pset = new WideAreaScanSettings();
            pset.XStep = 5000;
            pset.YStep = 5000;
            pset.XOverLapping = 300;
            pset.YOverLapping = 300;
            pset.MaxTrials = 2;
            pset.MinDensityBase = 0;
            new System.Xml.Serialization.XmlSerializer(typeof(WideAreaScanSettings)).Serialize(strw, pset);
            strw.WriteLine("");
            strw.WriteLine("");
            strw.WriteLine("NOTICE: If the quality cut id is identical to the linker id, no quality cut is applied (unless the linker applies its own quality cuts).");
            EF.RTFOut.Text = strw.ToString();
            EF.ShowDialog();
        }
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        internal static void Main(string[] args)
        {
            HE = SySal.DAQSystem.Drivers.HostEnv.Own;
            if (HE == null)
            {
                if (args.Length == 1 && String.Compare(args[0].Trim(), "/EasyInterrupt", true) == 0) EasyInterrupt();
                else ShowExplanation();
                return;
            }
            Execute();
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

                StartupInfo = (SySal.DAQSystem.Drivers.ScanningStartupInfo)HE.StartupInfo;
                Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                Conn.Open();
                (DBKeepAliveThread = new System.Threading.Thread(DBKeepAliveThreadExec)).Start();

                SySal.OperaDb.Schema.DB = Conn;
                
                System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(WideAreaScanSettings));
                ProgSettings = (WideAreaScanSettings)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
                xmls = null;

                HE.WriteLine("WideAreaScanDriver starting");

                if (StartupInfo.ExeRepository.EndsWith("\\")) StartupInfo.ExeRepository = StartupInfo.ExeRepository.Remove(StartupInfo.ExeRepository.Length - 1, 1);
                if (StartupInfo.ScratchDir.EndsWith("\\")) StartupInfo.ScratchDir = StartupInfo.ScratchDir.Remove(StartupInfo.ScratchDir.Length - 1, 1);
                if (StartupInfo.LinkedZonePath.EndsWith("\\")) StartupInfo.LinkedZonePath = StartupInfo.LinkedZonePath.Remove(StartupInfo.LinkedZonePath.Length - 1, 1);
                if (StartupInfo.RawDataPath.EndsWith("\\")) StartupInfo.RawDataPath = StartupInfo.RawDataPath.Remove(StartupInfo.RawDataPath.Length - 1, 1);

                long parentopid = System.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_PROC_OPERATIONS PK_PROC_OPERATIONS) */ ID_PARENT_OPERATION FROM TB_PROC_OPERATIONS WHERE ID = " + StartupInfo.ProcessOperationId, Conn, null).ExecuteScalar());
                if (StartupInfo.RawDataPath.IndexOf(parentopid.ToString()) < 0)
                {
                    StartupInfo.RawDataPath = StartupInfo.RawDataPath + "\\cssd_" + StartupInfo.Plate.BrickId + "_" + parentopid;
                    if (!System.IO.Directory.Exists(StartupInfo.RawDataPath)) throw new Exception(StartupInfo.RawDataPath + " not exist!!");
                }

                try
                {
                    StartupFile = StartupInfo.RawDataPath + "\\bmt_" + StartupInfo.ProcessOperationId + ".startup";
                    System.Xml.Serialization.XmlSerializer StartupInfoXml = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.ScanningStartupInfo));
                    System.IO.StreamWriter sw = new System.IO.StreamWriter(StartupFile);
                    StartupInfoXml.Serialize(sw, StartupInfo);
                    sw.Flush();
                    sw.Close();
                }
                catch (Exception) { }

                try
                {
                    ProgressFile = StartupInfo.RawDataPath + "\\bmt_" + StartupInfo.ProcessOperationId + ".progress";
                    System.IO.File.Copy(StartupInfo.ProgressFile, ProgressFile, true);
                }
                catch (Exception) { }

                if (System.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT DAMAGED FROM tb_plate_damagenotices WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID_PLATE = " + StartupInfo.Plate.PlateId, Conn, null).ExecuteScalar()) != 0)
                    throw new Exception("Plate #" + StartupInfo.Plate.PlateId + ", Brick #" + StartupInfo.Plate.BrickId + " is damaged!");

                string scanSettings = (string)new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_PROGRAMSETTINGS PK_PROGRAMSETTINGS) */ SETTINGS FROM TB_PROGRAMSETTINGS WHERE (ID = " + ProgSettings.ScanningConfigId + ")", Conn, Trans).ExecuteScalar();
                string linkConfig = (string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.LinkConfigId + ")", Conn, null).ExecuteScalar();
                xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Executables.BatchLink.Config));
                SySal.Executables.BatchLink.Config C = (SySal.Executables.BatchLink.Config)xmls.Deserialize(new System.IO.StringReader(linkConfig));

                if (ProgSettings.CorrectShrinkage == true)
                {
                    double topStep = Convert.ToDouble(GetParameterFromXml("TopStep", scanSettings), System.Globalization.CultureInfo.InvariantCulture);
                    double botStep = Convert.ToDouble(GetParameterFromXml("BottomStep", scanSettings), System.Globalization.CultureInfo.InvariantCulture);

                    C.TopMultSlopeX = topStep / DefaultEmuThickness;
                    C.TopMultSlopeY = topStep / DefaultEmuThickness;
                    C.BottomMultSlopeX = botStep / DefaultEmuThickness;
                    C.BottomMultSlopeY = botStep / DefaultEmuThickness;
                }

                LinkConfigPath = StartupInfo.RawDataPath + "\\linkconfig_" + StartupInfo.ProcessOperationId + ".xml";
                System.IO.StreamWriter w = new System.IO.StreamWriter(LinkConfigPath);
                xmls.Serialize(w, C);
                w.Flush();
                w.Close();
                xmls = null;

                if (ProgSettings.QualityCutId == ProgSettings.LinkConfigId)
                {
                    QualityCut = null;
                }
                else
                {
                    QualityCut = (string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.QualityCutId + ")", Conn, null).ExecuteScalar();
                    if (QualityCut.StartsWith("\"") && QualityCut.EndsWith("\"")) QualityCut = QualityCut.Substring(1, QualityCut.Length - 2);
                }

                
                System.Data.DataSet ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("select MINX-ZEROX, MAXX-ZEROX, MINY-ZEROY, MAXY-ZEROY from TB_EVENTBRICKS where ID = " + StartupInfo.Plate.BrickId, Conn).Fill(ds);

                double PlateMinX = Convert.ToDouble(ds.Tables[0].Rows[0][0], System.Globalization.CultureInfo.InvariantCulture);
                double PlateMaxX = Convert.ToDouble(ds.Tables[0].Rows[0][1], System.Globalization.CultureInfo.InvariantCulture);
                double PlateMinY = Convert.ToDouble(ds.Tables[0].Rows[0][2], System.Globalization.CultureInfo.InvariantCulture);
                double PlateMaxY = Convert.ToDouble(ds.Tables[0].Rows[0][3], System.Globalization.CultureInfo.InvariantCulture);

                //to take into account for the vacuum channel
                PlateMinX += ProgSettings.FiducialVolume;
                PlateMaxX -= ProgSettings.FiducialVolume;
                PlateMinY += ProgSettings.FiducialVolume;
                PlateMaxY -= ProgSettings.FiducialVolume;

                foreach (SySal.DAQSystem.Scanning.ZoneDesc zone in StartupInfo.Zones)
                {
                    SySal.BasicTypes.Rectangle area = new SySal.BasicTypes.Rectangle();
                    area.MinX = (zone.MinX > PlateMinX) ? zone.MinX : PlateMinX;
                    area.MaxX = (zone.MaxX < PlateMaxX) ? zone.MaxX : PlateMaxX;
                    area.MinY = (zone.MinY > PlateMinY) ? zone.MinY : PlateMinY;
                    area.MaxY = (zone.MaxY < PlateMaxY) ? zone.MaxY : PlateMaxY;

                    //check on the area
                    double XStep = Convert.ToDouble(GetParameterFromXml("XStep", scanSettings), System.Globalization.CultureInfo.InvariantCulture);//365
                    double YStep = Convert.ToDouble(GetParameterFromXml("YStep", scanSettings), System.Globalization.CultureInfo.InvariantCulture);
                    int XFields = Convert.ToInt32(GetParameterFromXml("XFields", scanSettings));//7
                    int YFields = Convert.ToInt32(GetParameterFromXml("YFields", scanSettings));

                    int XFrag = (int)System.Math.Ceiling(ProgSettings.XStep / XStep / XFields);//5000/365/7
                    int YFrag = (int)System.Math.Ceiling(ProgSettings.YStep / YStep / YFields);

                    double XAreaStep = XStep * XFields * XFrag;//365*7*2 = 5110
                    double YAreaStep = YStep * YFields * YFrag;

                    int nXsteps = (int)(System.Math.Ceiling((area.MaxX - area.MinX) / (XAreaStep - ProgSettings.XOverLapping)));
                    int nYsteps = (int)(System.Math.Ceiling((area.MaxY - area.MinY) / (YAreaStep - ProgSettings.YOverLapping)));

                    if (nXsteps == 0) nXsteps++;
                    if (nYsteps == 0) nYsteps++;

                    //Exit from fiducial area
                    //if (area.MinX + nXsteps * XAreaStep - (nXsteps - 1) * ProgSettings.XOverLapping > area.MaxX)
                    //    nXsteps--;
                    //if (area.MinY + nYsteps * YAreaStep - (nYsteps - 1) * ProgSettings.YOverLapping > area.MaxY)
                    //    nYsteps--;
                    bool WarningDx = ((nXsteps * XAreaStep - (nXsteps - 1) * ProgSettings.XOverLapping) > (PlateMaxX - PlateMinX)) ? true : false;
                    bool WarningDy = ((nYsteps * YAreaStep - (nYsteps - 1) * ProgSettings.YOverLapping) > (PlateMaxY - PlateMinY)) ? true : false;

                    if (WarningDx == true)
                        nXsteps--;

                    if (WarningDy == true)
                        nYsteps--;

                    double dx1 = area.MinX - PlateMinX;
                    double dx2 = Math.Max(0, area.MinX + nXsteps * XAreaStep - (nXsteps - 1) * ProgSettings.XOverLapping - area.MaxX);
                    double dx = Math.Min(dx1, dx2);

                    double dy2 = Math.Max(0, area.MinY + nYsteps * YAreaStep - (nYsteps - 1) * ProgSettings.YOverLapping - area.MaxY);
                    double dy1 = area.MinY - PlateMinY;
                    double dy = Math.Min(dy1, dy2);

                    int count = TotalStripList.Count;
                    System.Collections.ArrayList StripList = new ArrayList();
                    for (int iy = 0; iy < nYsteps; iy++)
                    {
                        for (int ix = 0; ix < nXsteps; ix++)
                        {
                            SySal.DAQSystem.Drivers.Prediction strip = new SySal.DAQSystem.Drivers.Prediction();

                            strip.Series = (iy % 2 == 0) ? iy * nXsteps + ix + 1 : (iy + 1) * nXsteps - ix;
                            strip.Series += count;
                            strip.MinX = area.MinX + ix * (XAreaStep - ProgSettings.XOverLapping) - dx;
                            strip.MaxX = strip.MinX + XAreaStep;
                            strip.MinY = area.MinY + iy * (YAreaStep - ProgSettings.YOverLapping) - dy;
                            strip.MaxY = strip.MinY + YAreaStep;
                            strip.MaxTrials = ProgSettings.MaxTrials;
                            strip.UsePresetSlope = false;
                            strip.Outname = StartupInfo.RawDataPath + "\\wideareascan_" + StartupInfo.ProcessOperationId + "_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.Plate.PlateId + "_" + strip.Series;

                            //if (ix == nXsteps - 1) strip.MaxX = area.MaxX;
                            //if (iy == nYsteps - 1) strip.MaxY = area.MaxY;

                            StripList.Add(strip);
                        }
                    }

                    if (StripList.Count > 0)
                    {
                        System.Collections.IComparer stripComparer = new StripComparer();
                        StripList.Sort(stripComparer);
                        TotalStripList.AddRange(StripList);
                    }
                }

                if (TotalStripList.Count == 0)
                    throw new Exception("Number of strips is zero!!");
                TotalStrips = TotalStripList.Count;

                StripStatusArray = new StripStatusArrayClass(TotalStrips, ProgSettings.MaxTrials);

                ProgressInfo = HE.ProgressInfo;

                HE.WriteLine("Start recoverery from ProgressFile");
                if (StartupInfo.RecoverFromProgressFile == false)
                {
                    ProgressInfo.StartTime = System.DateTime.Now;
                    ProgressInfo.FinishTime = ProgressInfo.StartTime;
                    ProgressInfo.Progress = 0.0;
                }
                else
                {
                    XmlDocument xmldoc = new XmlDocument();
                    xmldoc.LoadXml(ProgressInfo.CustomInfo.Replace('[', '<').Replace(']', '>'));
                    System.Xml.XmlNode xmlprog = xmldoc.FirstChild;
                    System.Xml.XmlNode xmlelem = xmlprog["ZoneInfos"];
                    XmlNode xn = xmlelem.FirstChild;
                    int Id;
                    while (xn != null)
                    {
                        Id = System.Convert.ToInt32(xn["ID"].InnerText);
                        StripStatusArray[Id - 1].Id = Id;
                        StripStatusArray[Id - 1].MaxTrials = System.Convert.ToUInt32(xn["MaxTrials"].InnerText);
                        StripStatusArray[Id - 1].Scanned = System.Convert.ToBoolean(xn["Scanned"].InnerText);
                        StripStatusArray[Id - 1].Monitored = System.Convert.ToBoolean(xn["Monitored"].InnerText);
                        StripStatusArray[Id - 1].MonitoringFile = System.Convert.ToString(xn["MonitoringFile"].InnerText);

                        try
                        {
                            StripStatusArray[Id - 1].Processed = System.Convert.ToBoolean(xn["Processed"].InnerText);
                        }
                        catch
                        {
                            HE.WriteLine("Processed set as false");
                            StripStatusArray[Id - 1].Processed = false;
                        }
                        StripStatusArray[Id - 1].Completed = System.Convert.ToBoolean(xn["Completed"].InnerText);

                        xn = xn.NextSibling;
                    }
                    ProgressInfo.ExitException = null;
                }

                HE.WriteLine("End recoverery from ProgressFile");
                ProgressInfo.Progress = (double)TotalCompletedStrips / (double)TotalStrips;
                ProgressInfo.Complete = false;
                ProgressInfo.ExitException = null;
                HE.ProgressInfo = ProgressInfo;

                DataProcSrv = HE.DataProcSrv;

                UpdateProgress();

                HE.WriteLine("Make scanning zone");
                TotalCompletedStrips = 0;
                for (int i = 0; i < TotalStrips; i++)
                {
                    SySal.DAQSystem.Drivers.Prediction strip = (SySal.DAQSystem.Drivers.Prediction)TotalStripList[i];

                    try
                    {
                        strip.MaxTrials = StripStatusArray[i].MaxTrials;
                        if (StripStatusArray[i].Completed == true)
                        {
                            TotalCompletedStrips++;
                            continue;
                        }
                        else
                        {
                            if (StripStatusArray[i].Scanned == false)
                                ScanQueue.Add(strip);
                            else if (ScanningCompleted(strip.Outname) == true)
                            {
                                StripStatusArray[i].Scanned = true;
                                WorkQueue.Enqueue(strip);
                            }
                            else ScanQueue.Add(strip);
                        }
                    }
                    catch (Exception x)
                    {
                        HE.WriteLine("Fill queues: " + x.Message);
                        StripStatusArray[i] = new StripStatus(i + 1, ProgSettings.MaxTrials);
                        ScanQueue.Add(strip);
                    }
                }

                if (ScanQueue.Count > 0)
                {
                    ScanSrv = HE.ScanSrv;
                    if (ScanSrv.SetScanLayout(scanSettings) == false)
                        throw new Exception("Scan Server configuration refused!");
                    StartupInfo.Plate.MapInitString = SySal.OperaDb.Scanning.Utilities.GetMapString(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, false, StartupInfo.MarkSet, out StartupInfo.CalibrationId, Conn, null);
                    if (ScanSrv.LoadPlate(StartupInfo.Plate) == false) throw new Exception("Can't load plate " + StartupInfo.Plate.PlateId + " + brick " + StartupInfo.Plate.BrickId);
                }

                if (ThisException != null) throw ThisException;

                WorkerThread.Start();
                if (WorkQueue.Count > 0) WorkerThread.Interrupt();

                HE.WriteLine("Start scanning cycle");
                try
                {
                    while (ScanQueue.Count + WorkQueue.Count > 0)
                    {
                        //TODO
                        //try
                        //{
                        //    MaxProcessingQueue = System.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select value from lz_sitevars where name='MaxProcessingQueue'", Conn, null).ExecuteScalar());
                        //}
                        //catch
                        //{
                        //    MaxProcessingQueue = 1000;
                        //}

                        //if (WorkQueue.Count > MaxProcessingQueue)
                        //{
                        //    ThisException = new Exception("Processing too slow. WorkQueue has " + WorkQueue.Count + " jobs in queue");
                        //    throw ThisException;
                        //}

                        if (ScanQueue.Count <= 0 && WorkQueue.Count <= 0)
                        {
                            if (ProgressInfo.ExitException != null) return;
                            if (ThisException != null)
                            {
                                HE.WriteLine("ThisException: " + ThisException.Message);
                                throw ThisException;
                            }
                            System.Threading.Thread.Sleep(1000);
                            if (ScanQueue.Count <= 0 && WorkQueue.Count <= 0) break;
                        }
                        if (ScanQueue.Count > 0)
                        {
                            SySal.DAQSystem.Drivers.Prediction pred = null;
                            SySal.DAQSystem.Scanning.ZoneDesc zd = null;
                            try
                            {
                                pred = (SySal.DAQSystem.Drivers.Prediction)ScanQueue[0];
                                zd = new SySal.DAQSystem.Scanning.ZoneDesc();

                                zd.MinX = pred.MinX;
                                zd.MaxX = pred.MaxX;
                                zd.MinY = pred.MinY;
                                zd.MaxY = pred.MaxY;
                                zd.Outname = pred.Outname;
                                zd.Series = pred.Series;

                                HE.WriteLine("Scanning zone " + zd.Series + " MinX: " + zd.MinX.ToString("F1") + " MaxX: " + zd.MaxX.ToString("F1") + " MinY: " + zd.MinY.ToString("F1") + " MaxY: " + zd.MaxY.ToString("F1"));
                            }
                            catch (Exception x)
                            {
                                HE.WriteLine("Define scanning zone: " + x.Message);
                                throw x;
                            }

                            string[] rwds = System.IO.Directory.GetFiles(pred.Outname.Substring(0, pred.Outname.LastIndexOf("\\")), pred.Outname.Substring(pred.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
                            foreach (string rwd in rwds)
                            {
                                try
                                {
                                    if (System.IO.File.Exists(rwd) == true)
                                        System.IO.File.Delete(rwd);
                                }
                                catch (Exception)
                                {
                                    HE.WriteLine("Deleting " + rwd);
                                }
                            }
                            try
                            {
                                if (System.IO.File.Exists(pred.Outname + ".rwc") == true)
                                    System.IO.File.Delete(pred.Outname + ".rwc");
                            }
                            catch (Exception)
                            {
                                HE.WriteLine("Deleting rwc");
                            }

                            HE.WriteLine("Starting scanning zone " + zd.Series);
                            if (ScanSrv.Scan(zd))
                            {
                                HE.WriteLine("Scanning zone " + zd.Series + " completed");
                                while (true)
                                {
                                    if (ScanningCompleted(pred.Outname) == true) break;
                                    try
                                    {
                                        System.Threading.Thread.Sleep(3000);
                                    }
                                    catch (System.Threading.ThreadInterruptedException) { }
                                    catch (Exception x)
                                    {
                                        HE.WriteLine("Check scanning completion: " + x.Message);
                                        throw x;
                                    }
                                }
                                int index = (int)zd.Series - 1;
                                StripStatusArray[index].Scanned = true;
                                lock (WorkQueue)
                                {
                                    WorkQueue.Enqueue(pred);
                                    WorkerThread.Interrupt();
                                }
                            }
                            else
                                throw new Exception("Scanning failed for zone " + zd.Series + " plate " + StartupInfo.Plate.PlateId + " brick " + StartupInfo.Plate.BrickId);

                            try
                            {
                                lock (ScanQueue)
                                    ScanQueue.RemoveAt(0);
                            }
                            catch (Exception x)
                            {
                                HE.WriteLine("Remove zone from ScanQueue: " + x.Message);
                                throw x;
                            }
                            UpdateProgress();
                        }
                    }

                    HE.WriteLine("Clear WorkQueue");
                    try
                    {
                        lock (WorkQueue)
                        {
                            WorkQueue.Clear();
                            WorkerThread.Interrupt();
                        }
                        //TODO: dopo il commento il processo termina 
                        //WorkerThread.Join();
                    }
                    catch (Exception x)
                    {
                        HE.WriteLine("Clear WorkQueue: " + x.Message);
                        throw x;
                    }
                }
                catch (Exception x)
                {
                    HE.WriteLine("Scanning loop: " + x.Message);
                    ThisException = x;
                    throw ThisException;
                }
            }
            catch (Exception x)
            {
                HE.WriteLine("Execute: " + x.Message);
                ThisException = x;
                throw ThisException;
            }

            bool KeepRawData = Convert.ToBoolean(new SySal.OperaDb.OperaDbCommand("select value from lz_sitevars where name='KeepRawData'", Conn).ExecuteScalar());

            if (KeepRawData == false)
            {
                HE.WriteLine("RWD cancellation");
                string[] files = System.IO.Directory.GetFiles(StartupInfo.RawDataPath, "*.rwd.*");
                foreach (string f in files)
                {
                    try
                    {
                        if (System.IO.File.Exists(f) == true)
                            System.IO.File.Delete(f);
                    }
                    catch (Exception x)
                    {
                        HE.WriteLine("RWD cancellation: " + x.Message);
                    }
                }
            }

            HE.Progress = 1.0;
            HE.InterruptNotifier = null;
            ProgressInfo.Complete = true;
            ProgressInfo.ExitException = null;
            ProgressInfo.Progress = 1.0;
            ProgressInfo.FinishTime = System.DateTime.Now;
            Conn.Close();
            UpdateProgress();
            UpdatePlots(true);
        }

        private static bool ScanningCompleted(string outname)
        {
            System.IO.FileStream f = null;
            try
            {
                f = new System.IO.FileStream(outname + ".rwc", System.IO.FileMode.Open, System.IO.FileAccess.Read);
            }
            catch (Exception x)
            {
                HE.WriteLine("RWC not exist " + x.Message);
                return false;
            }

            SySal.Scanning.Plate.IO.OPERA.RawData.Catalog Cat = new SySal.Scanning.Plate.IO.OPERA.RawData.Catalog(f);
            f.Close();

            uint start = Cat[0, 0];
            uint end = start;
            for (int ix = 0; ix < Cat.XSize; ix++)
                for (int iy = 0; iy < Cat.YSize; iy++)
                    end = Math.Max(end, Cat[iy, ix]);

            for (uint j = start; j <= end; j++)
            {
                if (System.IO.File.Exists(outname + ".rwd." + System.Convert.ToString(j, 16).PadLeft(8, '0')) == false)
                {
                    return false;
                }
            }
            return true;
        }

        private static string GetParameterFromXml(string paramName, string scanSettings)
        {
            System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
            doc.LoadXml(scanSettings);
            System.Xml.XmlNodeList yfieldXml = doc.GetElementsByTagName(paramName);
            if (yfieldXml.Count != 1) throw new Exception("Unknown or ambigious " + paramName + " settings!");
            return yfieldXml[0].InnerText;
        }

        private static void SetParameterFromXml(string newValue, string paramName, ref string scanSettings)
        {
            System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
            doc.LoadXml(scanSettings);
            System.Xml.XmlNodeList node = doc.GetElementsByTagName(paramName);
            if (node.Count != 1) throw new Exception("Unknown or ambigious " + paramName + " settings!");

            node[0].InnerText = newValue;
        }

        public Exe() { }

        private static void PostProcess()
        {
            try
            {
                while (true)
                {
                    SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
                    string corrfile = "";
                    if (System.IO.File.Exists(StartupInfo.RawDataPath + @"\fragmentshiftcorrection_" + StartupInfo.MachineId + ".xml"))
                        corrfile = " " + StartupInfo.RawDataPath + @"\fragmentshiftcorrection_" + StartupInfo.MachineId + ".xml";

                    SySal.DAQSystem.Drivers.Prediction pred = null;
                    lock (WorkQueue)
                    {
                        if (WorkQueue.Count == 0)
                        {
                            ThisException = null;
                            break;
                        }
                        pred = (SySal.DAQSystem.Drivers.Prediction)WorkQueue.Peek();
                    }

                    int index = (int)(pred.Series - 1);

                    HE.WriteLine("Processing " + pred.Series);
                    string tlgpath = pred.Outname + ".tlg";

                    if (StripStatusArray[index].Processed == false)
                    {
                        //                    SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = null;
                        try
                        {
                            if (System.IO.File.Exists(tlgpath) == true)
                                System.IO.File.Delete(tlgpath);
                        }
                        catch { }

                        dbd.AliasUsername = StartupInfo.DBUserName;
                        dbd.AliasPassword = StartupInfo.DBPassword;
                        dbd.Description = "Link for WideAreaScanDriver B#" + StartupInfo.Plate.BrickId + " P#" + StartupInfo.Plate.PlateId + " V#" + pred.Series;
                        dbd.Id = DataProcSrv.SuggestId;
                        dbd.Token = HE.Token;
                        dbd.MachinePowerClass = 5;
                        dbd.Filename = StartupInfo.ExeRepository + @"\BatchLink.exe";
//                        dbd.CommandLineArguments = pred.Outname + ".rwc " + tlgpath + " db:\\" + ProgSettings.LinkConfigId + ".xml " + corrfile;
                        dbd.CommandLineArguments = pred.Outname + ".rwc " + tlgpath + " " + LinkConfigPath + " " + corrfile;
                        if (!DataProcSrv.Enqueue(dbd)) throw new Exception("Cannot schedule linking batch " + dbd.Id + " for " + pred.Outname + ". Aborting.");
                        while (DataProcSrv.DoneWith(dbd.Id) == false)
                            try
                            {
                                System.Threading.Thread.Sleep(1000);
                            }
                            catch (System.Threading.ThreadInterruptedException) { };
                        dbd = DataProcSrv.Result(dbd.Id);

                        if (ProgSettings.LinkConfigId != ProgSettings.QualityCutId)
                        {
                            dbd.AliasUsername = StartupInfo.DBUserName;
                            dbd.AliasPassword = StartupInfo.DBPassword;
                            dbd.Description = "Quality cut for WideAreaScanDriver B#" + StartupInfo.Plate.BrickId + " P#" + StartupInfo.Plate.PlateId + " V#" + pred.Series;
                            dbd.Id = DataProcSrv.SuggestId;
                            dbd.Token = HE.Token;
                            dbd.MachinePowerClass = 5;
                            dbd.Filename = StartupInfo.ExeRepository + @"\TLGSel.exe";
                            dbd.CommandLineArguments = pred.Outname + ".tlg " + tlgpath + " \"" + QualityCut + "\"";
                            if (!DataProcSrv.Enqueue(dbd)) throw new Exception("Cannot schedule TLGsel " + dbd.Id + " for " + pred.Outname + ". Aborting.");
                            while (DataProcSrv.DoneWith(dbd.Id) == false)
                                try
                                {
                                    System.Threading.Thread.Sleep(1000);
                                }
                                catch (System.Threading.ThreadInterruptedException) { };
                            dbd = DataProcSrv.Result(dbd.Id);
                        }

                        StripLinkStatusInfo status = new StripLinkStatusInfo();

                        string qualityCheckFile = pred.Outname + "_check.xml";
                        string monitoringFile = StripStatusArray[index].MonitoringFile = pred.Outname + "_monitoring.xml";

                        dbd.AliasUsername = StartupInfo.DBUserName;
                        dbd.AliasPassword = StartupInfo.DBPassword;
                        dbd.Description = "Monitoring for WideAreaScanDriver B#" + StartupInfo.Plate.BrickId + " P#" + StartupInfo.Plate.PlateId + " V#" + pred.Series;
                        dbd.Id = DataProcSrv.SuggestId;
                        dbd.Token = HE.Token;
                        dbd.MachinePowerClass = 5;
                        dbd.Filename = StartupInfo.ExeRepository + @"\ZoneCheck.exe";
                        dbd.CommandLineArguments = pred.Series + " " + pred.Outname + ".rwc " + tlgpath + " " + qualityCheckFile + " " + monitoringFile;
                        if (!DataProcSrv.Enqueue(dbd)) throw new Exception("Cannot schedule linking batch " + dbd.Id + " for " + pred.Outname + ". Aborting.");
                        while (DataProcSrv.DoneWith(dbd.Id) == false)
                            try
                            {
                                System.Threading.Thread.Sleep(1000);
                            }
                            catch (System.Threading.ThreadInterruptedException) { };
                        dbd = DataProcSrv.Result(dbd.Id);

                        if (StripLinkStatusInfo.IsScannedStripGood(qualityCheckFile, ProgSettings.MinDensityBase))
                        {
                            HE.WriteLine("Zone completed: " + pred.Series);
                            StripStatusArray[index].MonitoringFile = monitoringFile;
                            StripStatusArray[index].Monitored = true;
                            StripStatusArray[index].Processed = true;
                            StripStatusArray[index].MaxTrials = pred.MaxTrials;
                            UpdateProgress();
                        }
                        else if (--pred.MaxTrials > 0)
                        {
                            HE.WriteLine("Zone " + pred.Series + " trials to go: " + pred.MaxTrials);
                            string[] rwds = System.IO.Directory.GetFiles(pred.Outname.Substring(0, pred.Outname.LastIndexOf("\\")), pred.Outname.Substring(pred.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
                            foreach (string rwd in rwds)
                            {
                                try
                                {
                                    if (System.IO.File.Exists(rwd) == true)
                                        System.IO.File.Delete(rwd);
                                }
                                catch (Exception)
                                {
                                    HE.WriteLine("Deleting " + rwd);
                                }
                            }

                            try
                            {
                                if (System.IO.File.Exists(pred.Outname + ".rwc") == true)
                                    System.IO.File.Delete(pred.Outname + ".rwc");
                            }
                            catch (Exception)
                            {
                                HE.WriteLine("Deleting rwc");
                            }

                            StripStatusArray[index].Scanned = false;
                            StripStatusArray[index].MaxTrials = pred.MaxTrials;

                            lock (ScanQueue)
                            {
                                if (ScanQueue.Count >= 2)
                                    ScanQueue.Insert(2, pred);
                                else
                                    ScanQueue.Add(pred);
                            }
                            UpdateProgress();
                        }
                        else
                        {
                            HE.WriteLine("Zone : " + pred.Series + " inserted after " + ProgSettings.MaxTrials);
                            StripStatusArray[index].MonitoringFile = monitoringFile;
                            StripStatusArray[index].Monitored = true;
                            StripStatusArray[index].Processed = true;
                            StripStatusArray[index].MaxTrials = pred.MaxTrials;
                            UpdateProgress();
                        }
                    }

                    if (StripStatusArray[index].Processed == true)
                    {
                        long idzone = 0;
                        SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(tlgpath, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));

                        WriteZone(pred.Outname + ".rwc", tlgpath, pred.Series);
                        //idzone = DumpZone(lz, StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, pred.Series, tlgpath, System.IO.File.GetCreationTime(pred.Outname + ".rwc"), System.DateTime.Now, Conn, null);
                        StripStatusArray[index].Completed = true;

                        TotalCompletedStrips++;
                        lock (WorkQueue)
                            WorkQueue.Dequeue();

                        UpdatePlots(false);
                    }

                    ProgressInfo.Progress = (double)TotalCompletedStrips / (double)TotalStrips;
                    UpdateProgress();
                }
            }
            catch (Exception x)
            {
                try
                {
                    HE.WriteLine("PostProcess:\r\n" + x.Message);
                }
                catch (Exception) { }
                ThisException = x;
            }
        }

        private static void UpdateProgress()
        {
            string xmlstr = "\r\n\t\t[InfoContainer]";
            xmlstr += "\r\n\t\t\t[ZoneInfos]\r\n";

            for (int i = 0; i < StripStatusArray.Length; i++)
            {
                xmlstr += "\t\t\t\t[Zone]\r\n";
                xmlstr += "\t\t\t\t\t[ID]" + StripStatusArray[i].Id + "[/ID]\r\n";
                xmlstr += "\t\t\t\t\t[MaxTrials]" + StripStatusArray[i].MaxTrials + "[/MaxTrials]\r\n";
                xmlstr += "\t\t\t\t\t[Scanned]" + StripStatusArray[i].Scanned + "[/Scanned]\r\n";
                xmlstr += "\t\t\t\t\t[Monitored]" + StripStatusArray[i].Monitored + "[/Monitored]\r\n";
                xmlstr += "\t\t\t\t\t[MonitoringFile]" + StripStatusArray[i].MonitoringFile + "[/MonitoringFile]\r\n";
                xmlstr += "\t\t\t\t\t[Processed]" + StripStatusArray[i].Processed + "[/Processed]\r\n";
                xmlstr += "\t\t\t\t\t[Completed]" + StripStatusArray[i].Completed + "[/Completed]\r\n";
                xmlstr += "\t\t\t\t[/Zone]\r\n";
            }
            xmlstr += "\t\t\t[/ZoneInfos]\r\n\t\t[/InfoContainer]\r\n";
            ProgressInfo.CustomInfo = xmlstr;
            HE.ProgressInfo = ProgressInfo;
        }

        private static void UpdatePlots(bool forceUpdate)
        {
            if (forceUpdate == false && ScanQueue.Count == 0) return;

            HE.WriteLine("UpdatePlots");
            try
            {
                ProgressFile = StartupInfo.RawDataPath + "\\bmt_" + StartupInfo.ProcessOperationId + ".progress";
                System.IO.File.Copy(StartupInfo.ProgressFile, ProgressFile, true);
            }
            catch (Exception) { }

            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = StartupInfo.DBUserName;
            dbd.AliasPassword = StartupInfo.DBPassword;
            dbd.Description = "UpdatePlots for WideAreaScanDriver Proc#" + StartupInfo.ProcessOperationId + " B#" + StartupInfo.Plate.BrickId + " P#" + StartupInfo.Plate.PlateId;
            dbd.Id = DataProcSrv.SuggestId;
            dbd.Token = HE.Token;
            dbd.MachinePowerClass = 5;
            dbd.Filename = StartupInfo.ExeRepository + @"\UpdatePlots.exe";
            dbd.CommandLineArguments = StartupFile + " " + ProgressFile;
            if (!DataProcSrv.Enqueue(dbd)) throw new Exception("Cannot schedule monitoring batch " + dbd.Id + " for process " + StartupInfo.ProcessOperationId + ". Aborting.");
            while (DataProcSrv.DoneWith(dbd.Id) == false)
                try
                {
                    System.Threading.Thread.Sleep(1000);
                }
                catch (System.Threading.ThreadInterruptedException) { };
            dbd = DataProcSrv.Result(dbd.Id);
        }

        private static long DumpZone(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, long db_brick_id, long db_plate_id, long db_procop_id, long series, string rawdatapath, DateTime starttime, DateTime endtime, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
        {
            lock (Conn)
                Trans = Conn.BeginTransaction();

            long db_zone_id = 0;
            try
            {
                SySal.DAQSystem.Scanning.IntercalibrationInfo transform = lz.Transform;
                double TDX = transform.TX - transform.MXX * transform.RX - transform.MXY * transform.RY;
                double TDY = transform.TY - transform.MYX * transform.RX - transform.MYY * transform.RY;

                //zone
                db_zone_id = SySal.OperaDb.Schema.TB_ZONES.Insert(db_brick_id, db_plate_id, db_procop_id, db_zone_id,
                    lz.Extents.MinX, lz.Extents.MaxX, lz.Extents.MinY, lz.Extents.MaxY,
                    rawdatapath, starttime, endtime, series,
                    transform.MXX, transform.MXY, transform.MYX, transform.MYY, TDX, TDY);

                Trans.Commit();
            }
            catch (Exception x)
            {
                db_brick_id = 0;
                if (Trans != null) Trans.Rollback();
                HE.WriteLine("DumpZone: " + x.Message);
            }
            return db_brick_id;
        }

        private static void WriteZone(string rwcpath, string tlgpath, long series)
        {
            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = StartupInfo.DBUserName;
            dbd.AliasPassword = StartupInfo.DBPassword;
            dbd.Description = "DumpZone for WideAreaScanDriver B#" + StartupInfo.Plate.BrickId + " P#" + StartupInfo.Plate.PlateId + " V#" + series;
            dbd.Id = DataProcSrv.SuggestId;
            dbd.Token = HE.Token;
            dbd.MachinePowerClass = 5;
            dbd.Filename = StartupInfo.ExeRepository + @"\DumpZone.exe";
            dbd.CommandLineArguments = StartupFile + " " + rwcpath + " " + tlgpath + " " + series + " " + LinkConfigPath;
            if (!DataProcSrv.Enqueue(dbd)) throw new Exception("Cannot schedule DumpZone batch " + dbd.Id + " for process " + StartupInfo.ProcessOperationId + ". Aborting.");
            while (DataProcSrv.DoneWith(dbd.Id) == false)
                try
                {
                    System.Threading.Thread.Sleep(1000);
                }
                catch (System.Threading.ThreadInterruptedException) { };
            dbd = DataProcSrv.Result(dbd.Id);
        }

        #region IInterruptNotifier Members

        /// <summary>
        /// Notifies incoming interrupts.
        /// </summary>
        /// <param name="nextint">the next interrupt to be processed.</param>
        void IInterruptNotifier.NotifyInterrupt(Interrupt nextint)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        #endregion
    }
}