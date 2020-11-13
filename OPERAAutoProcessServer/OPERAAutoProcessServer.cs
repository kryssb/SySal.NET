using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.ServiceProcess;
using System.Text;
using System.Net;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using System.Xml.Serialization;

namespace SySal.Services.OperaAutoProcessServer
{
    public partial class OperaAutoProcessServer : ServiceBase, SySal.Web.IWebApplication
    {        
        internal static string LogFile;

        internal static string MonitorPage;

        internal static SySal.Web.WebServer WebSrv;

        internal static void Log(string x)
        {
            if (LogFile != null && LogFile != "")
                try
                {
                    System.IO.File.AppendAllText(LogFile, "\r\n" + x);
                }
                catch (Exception) { }
        }

        [Serializable]
        public class VolumeOpRecord : IComparable
        {
            public long Id;
            
            public int BrickId;

            public long LinkConfigId;

            public long AlignIgnoreConfigId;

            public long GraphPlateSelId;

            public long RecConfigId;

            public bool Completed;

            [XmlIgnore]
            public bool Paused;

            [XmlIgnore]
            public string ExceptionText;

            public class AtomicOpRecord : IComparable
            {
                public long VolumeId;

                public int PlateId;

                public enum OpType { Link, GraphPlateSel, AlignIgnore, Reconstruct, MicrotrackExpand, CSMap };

                public OpType Activity;

                public ulong BatchId;

                #region IComparable Members

                public int CompareTo(object obj)
                {
                    AtomicOpRecord a = (AtomicOpRecord)obj;
                    if (a.VolumeId != VolumeId) return (a.VolumeId - VolumeId > 0) ? 1 : -1;
                    if (a.PlateId != PlateId) return a.PlateId - PlateId;
                    return (int)a.Activity - (int)Activity;
                }

                #endregion

                public AtomicOpRecord() { }

                public AtomicOpRecord(long volid, int plate, OpType act)
                {
                    VolumeId = volid;
                    PlateId = plate;
                    Activity = act;
                }

                public override string ToString()
                {
                    return "VolumeId " + VolumeId + " PlateId " + PlateId + " Activity " + Activity + " BatchId " + BatchId.ToString("X16");
                }

                public override bool Equals(object obj)
                {
                    AtomicOpRecord ar = (AtomicOpRecord)obj;
                    return VolumeId == ar.VolumeId && PlateId == ar.PlateId && Activity == ar.Activity;
                }
            }

            #region IComparable Members

            public int CompareTo(object obj)
            {
                VolumeOpRecord v = (VolumeOpRecord)obj;
                if (v.BrickId != BrickId) return (v.BrickId - BrickId > 0) ? -1 : 1;
                if (v.Id != Id) return (v.Id - Id > 0) ? -1 : 1;
                return 0;
            }

            #endregion

            [XmlIgnore]
            public System.Collections.ArrayList AtomicOps = new System.Collections.ArrayList();

            public VolumeOpRecord() { }

            public VolumeOpRecord(long id, int bkid, bool hascosmics, bool istrackfollow)
            {
                Id = id;
                BrickId = bkid;
                LinkConfigId = OperaAutoProcessServer.LinkConfigId;
                AlignIgnoreConfigId = OperaAutoProcessServer.IgnoreSelId;
                GraphPlateSelId = OperaAutoProcessServer.GraphPlateSelId;
                RecConfigId = istrackfollow ? OperaAutoProcessServer.RecTrackFollowConfigId :
                    (hascosmics ? OperaAutoProcessServer.RecWithCosmicRaysConfigId : OperaAutoProcessServer.RecNoCosmicRaysConfigId);
            }

            public override string ToString()
            {
                string s = "";
                foreach (object o in AtomicOps) s += "\r\n" + o.ToString();
                return "Id " + Id + "\r\n" + "BrickId " + BrickId + "\r\nAtomicOps\r\n" + s;
            }

            public override bool Equals(object obj)
            {
                VolumeOpRecord vr = (VolumeOpRecord)obj;
                return Id == vr.Id && BrickId == vr.BrickId;
            }
        }

        internal static System.Collections.ArrayList TotalScanOps = new System.Collections.ArrayList();

        internal static System.Collections.ArrayList TotalScanOpsCopy = new System.Collections.ArrayList();

        internal string DataProcSrvAddress;

        /// <summary>
        /// Name of the file where operations to be added are put.
        /// </summary>
        internal static string AddQueueName;

        /// <summary>
        /// Name of the working directory ("slashless").
        /// </summary>
        internal static string WorkDirName;

        /// <summary>
        /// Scratch directory.
        /// </summary>
        internal static string ScratchDir;

        /// <summary>
        /// Repository of executables.
        /// </summary>
        internal static string ExeRep;

        /// <summary>
        /// Progress file.
        /// </summary>
        internal static string ProgressFile;

        /// <summary>
        /// DB connection.
        /// </summary>
        internal static SySal.OperaDb.OperaDbConnection DBConn;

        /// <summary>
        /// DB connection for WWW services.
        /// </summary>
        internal static SySal.OperaDb.OperaDbConnection DBConnWWW;

        /// <summary>
        /// Connection String for the DB.
        /// </summary>
        internal static string DBServer;

        /// <summary>
        /// User that the service shall impersonate.
        /// </summary>
        internal static string DBUserName;

        /// <summary>
        /// Password to access the DB.
        /// </summary>
        internal static string DBPassword;

        /// <summary>
        /// Computing Infrastructure User.
        /// </summary>
        internal static string OPERAUserName;

        /// <summary>
        /// Computing Infrastructure Password.
        /// </summary>
        internal static string OPERAPassword;

        /// <summary>
        /// Site identifier read from the DB.
        /// </summary>
        internal static long IdSite;

        /// <summary>
        /// Site name read from the DB.
        /// </summary>
        internal static string SiteName;

        /// <summary>
        /// Machine identifier read from the DB.
        /// </summary>
        internal static long IdMachine;

        /// <summary>
        /// Machine address that matches the DB registration entry.
        /// </summary>
        internal static string MachineAddress;

        /// <summary>
        /// Machine name read from the DB.
        /// </summary>
        internal static string MachineName;

        /// <summary>
        /// Link configuration.
        /// </summary>
        internal static long LinkConfigId;

        /// <summary>
        /// String to extract microtracks from DB.
        /// </summary>
        internal static string LinkString;

        /// <summary>
        /// Id of the selection to ignore base tracks for alignment. 
        /// </summary>
        internal static long IgnoreSelId;

        /// <summary>
        /// Id of the selection to select in-plate tracks for graphics.
        /// </summary>
        internal static long GraphPlateSelId;

        /// <summary>
        /// Reconstruction configuration with cosmic rays.
        /// </summary>
        internal static long RecWithCosmicRaysConfigId;

        /// <summary>
        /// Reconstruction configuration without cosmic rays.
        /// </summary>
        internal static long RecNoCosmicRaysConfigId;

        /// <summary>
        /// Reconstruction configuration for quick track following (Scanback replacement).
        /// </summary>
        internal static long RecTrackFollowConfigId;

        /// <summary>
        /// String to be used in a WHERE clause to detect track following operations.
        /// </summary>
        internal static string TrackFollowSelection;

        /// <summary>
        /// Maximum number of batches to enqueue to the Data Processing Servers.
        /// </summary>
        internal static int MaxBatches = 2;

        object ReadOverride(string name, SySal.OperaDb.OperaDbConnection conn)
        {
            object o = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_MACHINEVARS WHERE ID_MACHINE = " + IdMachine + " AND NAME = '" + name + "'", conn, null).ExecuteScalar();
            if (o != null && o != System.DBNull.Value) return o;
            o = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = '" + name + "'", conn, null).ExecuteScalar();
            if (o == null) o = System.DBNull.Value;
            return o;
        }

        public OperaAutoProcessServer()
        {
            InitializeComponent();
            this.AutoLog = false;
        }

        const string EventSource = "OperaAutoProcessServer";

        protected override void OnStart(string[] args)
        {
            CheckTimer.Stop();
            try
            {
                System.Configuration.AppSettingsReader asr = new System.Configuration.AppSettingsReader();
                LogFile = (string)asr.GetValue("LogFile", typeof(string));                
                
                if (!EventLog.Exists("Opera")) EventLog.CreateEventSource(EventSource, "Opera");
                if (EventLog.SourceExists(EventSource) == false) EventLog.CreateEventSource(EventSource, "Opera");
                EventLog.Source = EventSource;
                EventLog.Log = "Opera";                

                EventLog.WriteEntry("Service starting.");

                try
                {
                    DBServer = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBServer", typeof(string)));
                    DBUserName = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBUserName", typeof(string)));
                    DBPassword = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBPassword", typeof(string)));
                    OPERAUserName = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("OPERAUserName", typeof(string)));
                    OPERAPassword = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("OPERAPassword", typeof(string)));
                }
                catch (Exception x)
                {
                    throw new Exception("Encryption error in credentials.\r\nPlease fill in valid encrypted data (you can use OperaDbGUILogin, for instance), or run the service as the appropriate user.");
                }
                DBConnWWW = new SySal.OperaDb.OperaDbConnection(DBServer, DBUserName, DBPassword);
                DBConn = new SySal.OperaDb.OperaDbConnection(DBServer, DBUserName, DBPassword);
                DBConn.Open();

                IPHostEntry iph = Dns.Resolve(Dns.GetHostName());
                string[] idstr = new string[iph.Aliases.Length + iph.AddressList.Length];
                idstr[0] = iph.HostName;
                int i;
                for (i = 0; i < iph.Aliases.Length; i++)
                    idstr[i] = iph.Aliases[i];
                for (i = 0; i < iph.AddressList.Length; i++)
                    idstr[i + iph.Aliases.Length] = iph.AddressList[i].ToString();
                string selstr = "LOWER(TB_MACHINES.ADDRESS)='" + iph.HostName.ToLower() + "'";
                foreach (string s in idstr)
                    selstr += (" OR ADDRESS='" + s + "'");
                DataSet ds = new DataSet();
                SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_SITES.ID, TB_SITES.NAME, TB_MACHINES.ID, TB_MACHINES.NAME, TB_MACHINES.ADDRESS FROM TB_SITES INNER JOIN TB_MACHINES ON (TB_MACHINES.ID_SITE = TB_SITES.ID AND TB_MACHINES.ISDATAPROCESSINGSERVER = 1 AND (" + selstr + "))", DBConn, null);
                da.Fill(ds);
                if (ds.Tables[0].Rows.Count < 1) throw new Exception("Can't find myself in OperaDb registered machines. This service is made unavailable.");
                IdSite = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][0]);
                SiteName = ds.Tables[0].Rows[0][1].ToString();
                IdMachine = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][2]);
                MachineName = ds.Tables[0].Rows[0][3].ToString();
                MachineAddress = ds.Tables[0].Rows[0][4].ToString();

                DataProcSrvAddress = new SySal.OperaDb.OperaDbCommand("SELECT ADDRESS FROM TB_MACHINES WHERE ID_SITE = " + IdSite + " AND ISBATCHSERVER = 1", DBConn).ExecuteScalar().ToString();

                object val;
                val = ReadOverride("APS_CheckTimeInterval", DBConn);
                if (val != System.DBNull.Value) CheckTimer.Interval = Convert.ToInt32(val.ToString()) * 1000;
                else CheckTimer.Interval = (int)asr.GetValue("CheckTimeInterval", typeof(int)) * 1000;

                val = ReadOverride("APS_WorkDirName", DBConn);
                WorkDirName = val.ToString();

                val = ReadOverride("APS_MonitorPage", DBConn);
                if (val != System.DBNull.Value) MonitorPage = val.ToString();
                else MonitorPage = (string)asr.GetValue("MonitorPage", typeof(string));

                ScratchDir = ReadOverride("ScratchDir", DBConn).ToString();
                if (ScratchDir.EndsWith("\\") || ScratchDir.EndsWith("/")) ScratchDir = ScratchDir.Remove(ScratchDir.Length - 1);
                ProgressFile = ((string)ScratchDir.Clone()) + "\\" + MachineName + ".app";

                val = ReadOverride("APS_AddQueueName", DBConn);
                if (val != System.DBNull.Value) AddQueueName = val.ToString();
                else AddQueueName = ScratchDir + "\\addqueue.aps";                

                ExeRep = ReadOverride("ExeRepository", DBConn).ToString();
                if (ExeRep.EndsWith("\\") || ExeRep.EndsWith("/")) ExeRep = ExeRep.Remove(ExeRep.Length - 1);

                val = ReadOverride("APS_ExeRepSubDir", DBConn);
                if (val != System.DBNull.Value) ExeRep += "\\" + val.ToString();

                val = ReadOverride("APS_LinkConfigId", DBConn);
                if (val != System.DBNull.Value) LinkConfigId = Convert.ToInt64(val.ToString());
                else LinkConfigId = (long)asr.GetValue("LinkConfigId", typeof(long));

                val = ReadOverride("APS_LinkString", DBConn);
                if (val != System.DBNull.Value) LinkString = val.ToString();
                else LinkString = (string)asr.GetValue("LinkString", typeof(string));

                val = ReadOverride("APS_AlignIgnoreSelId", DBConn);
                if (val != System.DBNull.Value) IgnoreSelId = Convert.ToInt64(val.ToString());
                else IgnoreSelId = (long)asr.GetValue("AlignIgnoreSelId", typeof(long));

                val = ReadOverride("APS_GraphPlateSelId", DBConn);
                if (val != System.DBNull.Value) GraphPlateSelId = Convert.ToInt64(val.ToString());
                else GraphPlateSelId = (long)asr.GetValue("GraphPlateSelId", typeof(long));

                val = ReadOverride("APS_RecWithCosmicRaysConfigId", DBConn);
                if (val != System.DBNull.Value) RecWithCosmicRaysConfigId = Convert.ToInt64(val.ToString());
                else RecWithCosmicRaysConfigId = (long)asr.GetValue("RecWithCosmicRaysConfigId", typeof(long));

                val = ReadOverride("APS_RecNoCosmicRaysConfigId", DBConn);
                if (val != System.DBNull.Value) RecNoCosmicRaysConfigId = Convert.ToInt64(val.ToString());
                else RecNoCosmicRaysConfigId = (long)asr.GetValue("RecNoCosmicRaysConfigId", typeof(long));

                val = ReadOverride("APS_RecTrackFollowConfigId", DBConn);
                if (val != System.DBNull.Value) RecTrackFollowConfigId = Convert.ToInt64(val.ToString());
                else RecTrackFollowConfigId = (long)asr.GetValue("RecTrackFollowConfigId", typeof(long));

                val = ReadOverride("APS_TrackFollowSelection", DBConn);
                if (val != System.DBNull.Value) TrackFollowSelection = val.ToString();
                else TrackFollowSelection = (string)asr.GetValue("TrackFollowSelection", typeof(string));

                int wwwport = 0;
                val = ReadOverride("APS_WWWPort", DBConn);
                if (val != System.DBNull.Value) wwwport = Convert.ToInt32(val.ToString());
                else wwwport = (int)asr.GetValue("WWWPort", typeof(int));
                OperaAutoProcessServer aps = null;
                if (wwwport > 0) WebSrv = new SySal.Web.WebServer(wwwport, aps = new OperaAutoProcessServer());

                bool showexc = false;
                val = ReadOverride("APS_WWWShowExceptions", DBConn);
                if (val != System.DBNull.Value) showexc = Convert.ToBoolean(val.ToString());
                else showexc = (bool)asr.GetValue("WWWShowExceptions", typeof(bool));
                if (aps != null) aps.SetShowExceptions(showexc);

                val = ReadOverride("APS_MaxBatches", DBConn);
                if (val != System.DBNull.Value) MaxBatches = Convert.ToInt32(val.ToString());
                else MaxBatches = (int)asr.GetValue("MaxBatches", typeof(int));

                //ChannelServices.RegisterChannel(new TcpChannel((int)SySal.DAQSystem.OperaPort.DataProcessingServer));
//                DPS = new SySal.DAQSystem.MyDataProcessingServer(EventLog);
  //              RemotingServices.Marshal(DPS, "AutoProcessServer.rem");

                Log("CheckTimeInterval: " + CheckTimer.Interval);
                Log("WorkDirName: " + WorkDirName);
                Log("LinkConfigId: " + LinkConfigId);
                Log("RecWithCosmicRaysConfigId: " + RecWithCosmicRaysConfigId);
                Log("RecNoCosmicRaysConfigId: " + RecNoCosmicRaysConfigId);
                Log("MaxBatches: " + MaxBatches);

                ReadProgressFile();

                /*
                foreach (VolumeOpRecord v_1 in TotalScanOps)
                    Log(v_1.ToString());
                */

//KRYSS                DBConn.Close();
                
                CleanBatches();

                CheckTimer.AutoReset = true;                
                CheckTimer.Elapsed += new System.Timers.ElapsedEventHandler(CheckTimer_Tick);
                CheckTimer.Enabled = true;
                CheckTimer.Start();

                Log("Started");
            }
            catch (Exception x)
            {
                Log("Exception while starting: ");
                Log(x.ToString());
                EventLog.WriteEntry("Service startup failure:\r\n" + x.ToString(), EventLogEntryType.Error);
                throw x;
            }
        }

        void CleanBatches()
        {
            SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)SySal.DAQSystem.OperaPort.BatchServer + "/DataProcessingServer.rem");
            SySal.DAQSystem.DataProcessingBatchDesc[] openbatches = Srv.Queue;
            foreach (SySal.DAQSystem.DataProcessingBatchDesc dbd in openbatches)
            {
                if (dbd.Description.StartsWith("APS_" + MachineName))
                {
                    try
                    {
                        Srv.Remove(dbd.Id, null, OPERAUserName, OPERAPassword);
                    }
                    catch (Exception) { }
                }
            }
        }

        static System.Timers.Timer CheckTimer = new System.Timers.Timer();

        void ReadProgressFile()
        {
            System.IO.StreamReader r = null;
            try
            {
                r = new System.IO.StreamReader(ProgressFile);
                VolumeOpRecord[] vr = (VolumeOpRecord[])XmlS.Deserialize(r);
                foreach (VolumeOpRecord v in vr) TotalScanOps.Add(v);
                r.Close();
            }
            catch (Exception x)
            {
                EventLog.WriteEntry("Cannot read progress file:\r\n" + x.ToString(), EventLogEntryType.Warning);
            }
            finally
            {
                if (r != null)
                {
                    r.Close();
                    r = null;
                }
            }

        }

        static System.Xml.Serialization.XmlSerializer XmlS = new XmlSerializer(typeof(VolumeOpRecord []));

        const string cmdStop = "stop";
        const string cmdAutoPause = "autopause";
        const string cmdPause = "pause";
        const string cmdPrioritize = "prioritize";
        const string cmdTempRec = "temprec";       

        string HtmlProgress(bool sort, bool forweb)
        {
            string ticks = System.DateTime.Now.Ticks.ToString();
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>OperaAutoProcess Monitor - " + MachineName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Verdana,Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Verdana,Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Verdana,Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    body {font-family: Verdana,Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "<p><b>OperaAutoProcess Monitor (" + MachineName + ")<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></p>\r\n" +
                            (forweb ?
                            "<p><form method=\"get\"><input id=\"addbtn\" type=\"button\" value=\"Add to queue\" onclick=\"location='?" + AddQueueCmd + "=0_' + addop.value; \"/><input id=\"addop\" maxlength=\"30\" name=\"addop\" size=\"30\" type=\"text\" /></form></p><br><a href=\"?sort=true&t=" + ticks + "\">sort</a>&nbsp;<a href=\"?sort=false&t=" + ticks + "\">don't sort</a>" 
                            : 
                            "<p>Command files accepted: <b>" + cmdStop + ".aps</b>, <b>" + cmdPause + ".aps</b>, <b>" + cmdTempRec + ".aps</b>, <b>" + cmdPrioritize + ".aps</b></p>\r\n" +
                            "<p>Put additional process operations to monitor in: <b>" + AddQueueName.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;") + "</b></p>\r\n"
                            ) +
                            "<table align=\"center\" width=\"100%\" border=\"1\">\r\n" +
                            "<tr><th>Brick</th><th>Operation</th><th>ConfigIds</th><th>Processing</th></tr>\r\n";

            System.Collections.ArrayList ops = new System.Collections.ArrayList();
            lock (TotalScanOpsCopy)
                foreach (VolumeOpRecord vr in TotalScanOpsCopy) ops.Add(vr);

            if (sort) ops.Sort();

            foreach (VolumeOpRecord vr in ops)
            {
                html += "<tr valign=\"top\"><td>" + vr.BrickId;
                if (vr.Completed) html += "<br><font color=\"green\"><b>COMPLETED</b></font>";
                else if (vr.Paused) html += "<br><b>PAUSED</b>";
                if (vr.ExceptionText != null) html += "<br><font color=\"red\"><b>" + vr.ExceptionText.Replace("<", "&lt;").Replace(">", "gt;").Replace("&", "&amp;") + "</b></font>";
                if (forweb)
                {
                    string v = vr.BrickId + "_" + vr.Id;
                    html += "<br><a href=\"?" + TempRecCmd + "=" + v + "&t=" + ticks + "\">temprec</a>&nbsp;<a href=\"?" + (vr.Paused ? ResumeCmd : PauseCmd) + "=" + v + "&t=" + ticks + "\">" + (vr.Paused ? "resume" : "pause") + "</a><br><a href=\"?" + PrioritizeCmd + "=" + v + "&t=" + ticks + "\">prioritize</a><br><a href=\"?" + StopCmd + "=" + v + "&t=" + ticks + "\"><font color=\"red\"><i>stop<i></font></a>" + (forweb ? ("<br><a href=\"" + GraphSelPage + "?" + GraphProcOpCmd + "=" + vr.Id + "&" + GraphRefreshCmd + "=" + ticks + "\">graph</a>") : "");
                }
                html += "</td><td>" + vr.Id + "</td><td><table border=\"1\" width=\"100%\" align=\"center\"><tr><th>Activity</th><th>Id</th><tr><tr><td>Link</td><td>" + vr.LinkConfigId + "</td></tr><tr><td>AlignIgnore</td><td>" + vr.AlignIgnoreConfigId + "</td></tr><tr><td>Reconstruction</td><td>" + vr.RecConfigId + "</td></tr></table></td><td>";
                html += "<table border=\"1\" align=\"center\" width=\"100%\"><tr><th>Volume</th><th>Plate</th><th>Activity</th></tr>\r\n";
                foreach (VolumeOpRecord.AtomicOpRecord ar in vr.AtomicOps)
                    html += "<tr><td>" + ar.VolumeId + "</td><td>" + ar.PlateId + "</td><td>" + ar.Activity + "</td></tr>\r\n";
                html += "</table>";

                html += "</td></tr>\r\n";
            }

            html += "</table>\r\n</body>\r\n</html>\r\n";
            return html;
        }

        void UpdateProgressFile()
        {            
            System.IO.StreamWriter w = null;
            try
            {                
                w = new System.IO.StreamWriter(ProgressFile);
                XmlS.Serialize(w, (VolumeOpRecord [])TotalScanOps.ToArray(typeof(VolumeOpRecord)));
                w.Flush();
                w.Close();
            }
            catch (Exception x)
            {
                EventLog.WriteEntry("Cannot update progress file:\r\n" + x.ToString(), EventLogEntryType.Error);
            }
            finally
            {
                if (w != null) 
                {
                    w.Close();
                    w = null;
                }
            }
            try
            {
                if (MonitorPage != null && MonitorPage != "")
                {
                    System.IO.File.WriteAllText(MonitorPage, HtmlProgress(true, false));
                }
            }
            catch (Exception x)
            {
                EventLog.WriteEntry("Cannot update monitor page:\r\n" + x.ToString(), EventLogEntryType.Error);
            }            
        }

        bool ScheduleMicrotrackExpand(long opid, int bkid, long volid, VolumeOpRecord vr)
        {            
            System.Data.DataSet ds1 = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PLATE FROM TB_VOLUME_SLICES WHERE ID_EVENTBRICK = " + bkid + " AND ID_VOLUME = " + volid, DBConn).Fill(ds1);
            System.IO.StreamWriter tw = new System.IO.StreamWriter(ListMTkExpandFilePath(vr.BrickId, opid, volid));
            foreach (System.Data.DataRow dr in ds1.Tables[0].Rows)
                tw.WriteLine("0 " + dr[0].ToString() + " " + TLGName(opid, bkid, volid, SySal.OperaDb.Convert.ToInt32(dr[0])));
            tw.Flush();
            tw.Close();
            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = DBUserName;
            dbd.AliasPassword = DBPassword;
            dbd.Description = "APS_" + MachineName + " microtrack expansion for brick " + bkid + " op " + opid + " vol " + volid;
            dbd.Filename = ExeRep + "\\TSRMuExpand.exe";
            dbd.MachinePowerClass = 2;
            dbd.Username = OPERAUserName;
            dbd.Password = OPERAPassword;
            dbd.CommandLineArguments = "\"" + TSRName(opid, bkid, volid) + "\" \"" + ListMTkExpandFilePath(vr.BrickId, opid, volid) + "\" \"" + MTkTSRName(opid, bkid, volid) + "\"";
            dbd.MaxOutputText = 65536;
            dbd.OutputTextSaveFile = MTkTSRName(opid, bkid, volid) + ".log";
            SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)SySal.DAQSystem.OperaPort.BatchServer + "/DataProcessingServer.rem");
            dbd.Id = Srv.SuggestId;
            if (Srv.Enqueue(dbd))
            {
                VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(volid, 0, VolumeOpRecord.AtomicOpRecord.OpType.MicrotrackExpand);
                ar.BatchId = dbd.Id;
                vr.AtomicOps.Add(ar);
                return true;
            }
            return false;
        }

        bool ScheduleRec(long opid, int bkid, long volid, VolumeOpRecord vr, bool temporary)
        {           
            System.Data.DataSet ds = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT EXECUTABLE, SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + RecWithCosmicRaysConfigId, DBConn).Fill(ds);
            if (ds.Tables[0].Rows.Count == 0) return false;
            System.Data.DataSet ds1 = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, Z FROM TB_PLATES INNER JOIN (SELECT ID_EVENTBRICK AS IDB, ID_PLATE FROM TB_VOLUME_SLICES WHERE ID_EVENTBRICK = " + bkid + " AND ID_VOLUME = " + volid + ") ON (ID_EVENTBRICK = IDB AND ID = ID_PLATE) ORDER BY Z DESC", DBConn).Fill(ds1);
            System.IO.StreamWriter tw = new System.IO.StreamWriter(ListFilePath(vr.BrickId, opid, volid));
            tw.WriteLine("<Input><Zones>");
            foreach (System.Data.DataRow dr in ds1.Tables[0].Rows)
                if (temporary == false || (System.IO.File.Exists(TLGName(opid, bkid, volid, SySal.OperaDb.Convert.ToInt32(dr[0]))) && System.IO.File.Exists(IgnoreName(opid, bkid, volid, SySal.OperaDb.Convert.ToInt32(dr[0])))))
                    tw.WriteLine(" <Zone><Source>" + TLGName(opid, bkid, volid, SySal.OperaDb.Convert.ToInt32(dr[0])) + "</Source><SheetId>" + dr[0].ToString() + "</SheetId><Z>" + SySal.OperaDb.Convert.ToDouble(dr[1]).ToString(System.Globalization.CultureInfo.InvariantCulture) + 
                        "</Z><AlignmentIgnoreListPath>" + IgnoreName(opid, bkid, volid, SySal.OperaDb.Convert.ToInt32(dr[0])) + "</AlignmentIgnoreListPath></Zone>");
            tw.WriteLine("</Zones></Input>");
            tw.Flush();
            tw.Close();
            /*
            bool hascosmics;
            try
            {
                hascosmics = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select decode(RESULT_STATUS,'BLACK_CS_DEVELOP',0,'NO_COSMIC_RAYS_DEVELOP',0,1) as hascosmics from tv_cs_results where id_cs_eventbrick = (select max(id_cs_eventbrick) as lastcs from tv_cs_results where mod(id_cs_eventbrick,1000000) = " + (bkid - 1000000) + " and id_cs_eventbrick between 3000000 and 9999999)", DBConn).ExecuteScalar()) != 0;
            }
            catch (Exception x)
            {
                vr.ExceptionText = "Cannot retrieve cosmic ray information for this brick.";
                return false;
            }
             */
            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = DBUserName;
            dbd.AliasPassword = DBPassword;
            dbd.Description = "APS_" + MachineName + " reconstruction for brick " + bkid + " op " + opid + " vol " + volid;
            dbd.Filename = ExeRep + "\\" + ds.Tables[0].Rows[0][0].ToString();
            dbd.MachinePowerClass = 2;
            dbd.Username = OPERAUserName;
            dbd.Password = OPERAPassword;
            dbd.CommandLineArguments = "\"" + ListFilePath(vr.BrickId, opid, volid) + "\" \"" + (temporary ? TempTSRName(opid, bkid, volid) : TSRName(opid, bkid, volid)) + "\" db:\\" + /*(hascosmics ? RecWithCosmicRaysConfigId : RecNoCosmicRaysConfigId)*/ vr.RecConfigId + ".xml";
            dbd.MaxOutputText = 65536;
            dbd.OutputTextSaveFile = (temporary ? TempTSRName(opid, bkid, volid) : TSRName(opid, bkid, volid)) + ".log";
            SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)SySal.DAQSystem.OperaPort.BatchServer + "/DataProcessingServer.rem");
            dbd.Id = Srv.SuggestId;
            if (Srv.Enqueue(dbd) && temporary == false)
            {
                VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(volid, 0, VolumeOpRecord.AtomicOpRecord.OpType.Reconstruct);
                ar.BatchId = dbd.Id;
                vr.AtomicOps.Add(ar);
                return true;
            }
            return false;
        }

        bool ScheduleLink(long opid, int bkid, long volid, int plate, VolumeOpRecord vr)
        {            
            System.Data.DataSet ds = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT EXECUTABLE, SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + LinkConfigId, DBConn).Fill(ds);
            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = DBUserName;
            dbd.AliasPassword = DBPassword;
            dbd.Description = "APS_" + MachineName + " link for brick " + bkid + " op " + opid + " vol " + volid + " plate " + plate;
            dbd.Filename = ExeRep + "\\" + ds.Tables[0].Rows[0][0].ToString();
            dbd.MachinePowerClass = 5;
            dbd.Username = OPERAUserName;
            dbd.Password = OPERAPassword;
            dbd.CommandLineArguments = "/dbquery \"" + LinkString.Replace("_BRICK_", bkid.ToString()).Replace("_PLATE_", plate.ToString()).Replace("_VOL_", volid.ToString()) + "\" \"" + TLGName(opid, bkid, volid, plate) + "\" db:\\" + vr.LinkConfigId + ".xml";
            dbd.OutputTextSaveFile = TLGName(opid, bkid, volid, plate) + ".log";
            SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)SySal.DAQSystem.OperaPort.BatchServer + "/DataProcessingServer.rem");
            dbd.Id = Srv.SuggestId;
            if (Srv.Enqueue(dbd))
            {
                VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(volid, plate, VolumeOpRecord.AtomicOpRecord.OpType.Link);
                ar.BatchId = dbd.Id;
                vr.AtomicOps.Add(ar);
                return true;
            }
            return false;
        }

        bool ScheduleIgnore(long opid, int bkid, long volid, int plate, VolumeOpRecord vr)
        {            
            System.Data.DataSet ds = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT EXECUTABLE, SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + vr.AlignIgnoreConfigId, DBConn).Fill(ds);            
            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = DBUserName;
            dbd.AliasPassword = DBPassword;
            dbd.Description = "APS_" + MachineName + " align ignore selection for brick " + bkid + " op " + opid + " vol " + volid + " plate " + plate;
            dbd.Filename = ExeRep + "\\" + ds.Tables[0].Rows[0][0].ToString();
            dbd.MachinePowerClass = 5;
            dbd.Username = OPERAUserName;
            dbd.Password = OPERAPassword;
            dbd.CommandLineArguments = "\"" + TLGName(opid, bkid, volid, plate) + "\" \"" + IgnoreName(opid, bkid, volid, plate) + "\" \"" + ds.Tables[0].Rows[0][1].ToString().Replace("_BRICK_", vr.BrickId.ToString()).Replace("_PLATE_", plate.ToString()).Replace("_VOLUME_", volid.ToString()).Replace("_OPERATION_", opid.ToString()) + "\"";
            dbd.OutputTextSaveFile = IgnoreName(opid, bkid, volid, plate) + ".log";
            SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)SySal.DAQSystem.OperaPort.BatchServer + "/DataProcessingServer.rem");
            dbd.Id = Srv.SuggestId;
            if (Srv.Enqueue(dbd))
            {
                VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(volid, plate, VolumeOpRecord.AtomicOpRecord.OpType.AlignIgnore);
                ar.BatchId = dbd.Id;
                vr.AtomicOps.Add(ar);
                return true;
            }
            return false;
        }

        bool ScheduleGraphPlateSel(long opid, int bkid, long volid, int plate, VolumeOpRecord vr)
        {
            System.Data.DataSet ds = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT EXECUTABLE, SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + vr.GraphPlateSelId, DBConn).Fill(ds);
            SySal.DAQSystem.DataProcessingBatchDesc dbd = new SySal.DAQSystem.DataProcessingBatchDesc();
            dbd.AliasUsername = DBUserName;
            dbd.AliasPassword = DBPassword;
            dbd.Description = "APS_" + MachineName + " graph selection for brick " + bkid + " op " + opid + " vol " + volid + " plate " + plate;
            dbd.Filename = ExeRep + "\\" + ds.Tables[0].Rows[0][0].ToString();
            dbd.MachinePowerClass = 5;
            dbd.Username = OPERAUserName;
            dbd.Password = OPERAPassword;
            dbd.CommandLineArguments = "\"" + TLGName(opid, bkid, volid, plate) + "\" \"" + GraphPlateSelName(opid, bkid, volid, plate) + "\" " + ds.Tables[0].Rows[0][1].ToString().Replace("_BRICK_", vr.BrickId.ToString()).Replace("_PLATE_", plate.ToString()).Replace("_VOLUME_", volid.ToString()).Replace("_OPERATION_", opid.ToString());
            dbd.OutputTextSaveFile = GraphPlateSelName(opid, bkid, volid, plate) + ".log";
            SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)SySal.DAQSystem.OperaPort.BatchServer + "/DataProcessingServer.rem");
            dbd.Id = Srv.SuggestId;
            if (Srv.Enqueue(dbd))
            {
                VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(volid, plate, VolumeOpRecord.AtomicOpRecord.OpType.GraphPlateSel);
                ar.BatchId = dbd.Id;
                vr.AtomicOps.Add(ar);
                return true;
            }            
            return false;
        }

        void CheckTimer_Tick(object sender, EventArgs e)
        {
            bool reopen = false;
            try
            {
                new SySal.OperaDb.OperaDbCommand("SELECT * FROM DUAL", DBConn).ExecuteScalar();
            }
            catch (Exception)
            {
                DBConn.Close();
                reopen = true;
            }
            try
            {
                if (reopen) DBConn.Open();

                object val = null;

                System.Configuration.AppSettingsReader asr = new System.Configuration.AppSettingsReader();

                val = ReadOverride("APS_LinkConfigId", DBConn);
                if (val != System.DBNull.Value) LinkConfigId = Convert.ToInt64(val.ToString());
                else LinkConfigId = (long)asr.GetValue("LinkConfigId", typeof(long));

                val = ReadOverride("APS_LinkString", DBConn);
                if (val != System.DBNull.Value) LinkString = val.ToString();
                else LinkString = (string)asr.GetValue("LinkString", typeof(string));

                val = ReadOverride("APS_AlignIgnoreSelId", DBConn);
                if (val != System.DBNull.Value) IgnoreSelId = Convert.ToInt64(val.ToString());
                else IgnoreSelId = (long)asr.GetValue("AlignIgnoreSelId", typeof(long));

                val = ReadOverride("APS_GraphPlateSelId", DBConn);
                if (val != System.DBNull.Value) GraphPlateSelId = Convert.ToInt64(val.ToString());
                else GraphPlateSelId = (long)asr.GetValue("GraphPlateSelId", typeof(long));

                val = ReadOverride("APS_RecWithCosmicRaysConfigId", DBConn);
                if (val != System.DBNull.Value) RecWithCosmicRaysConfigId = Convert.ToInt64(val.ToString());
                else RecWithCosmicRaysConfigId = (long)asr.GetValue("RecWithCosmicRaysConfigId", typeof(long));

                val = ReadOverride("APS_RecNoCosmicRaysConfigId", DBConn);
                if (val != System.DBNull.Value) RecNoCosmicRaysConfigId = Convert.ToInt64(val.ToString());
                else RecNoCosmicRaysConfigId = (long)asr.GetValue("RecNoCosmicRaysConfigId", typeof(long));

                val = ReadOverride("APS_RecTrackFollowConfigId", DBConn);
                if (val != System.DBNull.Value) RecTrackFollowConfigId = Convert.ToInt64(val.ToString());
                else RecTrackFollowConfigId = (long)asr.GetValue("RecTrackFollowConfigId", typeof(long));

                val = ReadOverride("APS_TrackFollowSelection", DBConn);
                if (val != System.DBNull.Value) TrackFollowSelection = val.ToString();
                else TrackFollowSelection = (string)asr.GetValue("TrackFollowSelection", typeof(string));

                val = ReadOverride("APS_MaxBatches", DBConn);
                if (val != System.DBNull.Value) MaxBatches = Convert.ToInt32(val.ToString());
                else MaxBatches = (int)asr.GetValue("MaxBatches", typeof(int));
            }
            catch (Exception x) 
            {
                try
                {
                    EventLog.WriteEntry("Unable to refresh parameters:\r\n" + x.ToString(), EventLogEntryType.Error);
                }
                catch (Exception y)
                {
                    Log(x.ToString());
                }
            }
            finally
            {
//KRYSS                DBConn.Close();
            }
            try
            {
                System.Collections.ArrayList enqlist = new System.Collections.ArrayList();
                if (System.IO.File.Exists(AddQueueName))
                    try
                    {
                        System.Collections.ArrayList polist = new System.Collections.ArrayList();                        
                        string[] procops = System.IO.File.ReadAllText(AddQueueName).Split(' ', '\r', '\n', ',', '\t', ';');
                        foreach (string po in procops)
                            try
                            {
                                polist.Add(System.Convert.ToInt64(po));
                            }
                            catch (Exception) { }
                        foreach (long po in polist)
                        {
//KRYSS                            DBConn.Open();
                            try
                            {
                                int bkid = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT ID_EVENTBRICK FROM TB_PROC_OPERATIONS WHERE ID = " + po.ToString(), DBConn).ExecuteScalar());
                                enqlist.Add(new object[2] { po, bkid });
                            }
                            catch (Exception) { }
//KRYSS                            DBConn.Close();
                        }
                    }
                    catch (Exception x)
                    {
                        Log(x.ToString());
                    }
                    finally
                    {
                        if (System.IO.File.Exists(AddQueueName))
                            try
                            {
                                System.IO.File.Delete(AddQueueName);
                            }
                            catch (Exception)
                            {
                                Log("Can't delete AddQueueName file: " + AddQueueName);
                            }
                    }
//KRYSS                DBConn.Open();
                System.Data.DataSet ds = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, ID_EVENTBRICK FROM TB_PROC_OPERATIONS WHERE SUCCESS = 'R' AND EXISTS (SELECT * FROM TB_VOLUMES WHERE ID_PROCESSOPERATION = TB_PROC_OPERATIONS.ID)", DBConn).Fill(ds);
//KRYSS                DBConn.Close();
                lock (TotalScanOps)
                {
                    lock (TotalScanOpsCopy)
                    {
                        TotalScanOpsCopy.Clear();
                        foreach (VolumeOpRecord vxz in TotalScanOps)
                            TotalScanOpsCopy.Add(vxz);
                    }
                    /*
                    foreach (VolumeOpRecord v_1 in TotalScanOps)
                        Log(v_1.ToString());
                     */
                    VolumeOpRecord opid = null;
                    foreach (object[] enqi in enqlist)
                        ds.Tables[0].Rows.Add(enqi);
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    {
                        opid = new VolumeOpRecord(SySal.OperaDb.Convert.ToInt64(dr[0]), SySal.OperaDb.Convert.ToInt32(dr[1]), false, false);
                        if (TotalScanOps.Contains(opid) == false/* && System.IO.Directory.Exists(DirName(opid.BrickId, opid.Id)) == false*/)
                        {
                            try
                            {
//KRYSS                                DBConn.Open();
                                bool istrackfollow = false;
                                string sqltext = "select count(*) as trackfollowcheck from tb_programsettings where id = (select id_programsettings from tb_proc_operations where id = " + opid.Id + ") and ";
                                if (TrackFollowSelection != null && TrackFollowSelection.Trim().Length > 0)                                
                                    try
                                    {
                                        sqltext += TrackFollowSelection;
                                        istrackfollow = (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand(sqltext, DBConn).ExecuteScalar()) > 0);
                                    }
                                    catch (Exception x)
                                    {
                                        throw new Exception("Unable to use TrackFollow selection\r\nSQL text:\r\n" + sqltext + "\r\n" + x.ToString());
                                    }     
                                bool hascosmics;
                                try
                                {
                                    hascosmics = istrackfollow ? false :
                                        (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select decode(RESULT_STATUS,'BLACK_CS_DEVELOP',0,'NO_COSMIC_RAYS_DEVELOP',0,'CS_CAND_OK_FAST_UNPACK',0,1) as hascosmics from tv_cs_results where id_cs_eventbrick = (select max(id_cs_eventbrick) as lastcs from tv_cs_results where mod(id_cs_eventbrick,1000000) = " + (opid.BrickId % 1000000) + " and id_cs_eventbrick between 3000000 and 9999999)", DBConn).ExecuteScalar()) != 0);
                                }
                                catch (Exception x)
                                {
                                    throw new Exception("Cannot retrieve cosmic rays information for brick " + opid.BrickId);
                                }
                                opid = new VolumeOpRecord(opid.Id, opid.BrickId, hascosmics, istrackfollow);
                            }
                            catch (Exception x)
                            {                                
                                Log(x.ToString());
                            }
                            finally
                            {
//KRYSS                                DBConn.Close();
                            }                            
                            if (System.IO.Directory.Exists(DirName(opid.BrickId, opid.Id)) == false)
                                System.IO.Directory.CreateDirectory(DirName(opid.BrickId, opid.Id));
                            TotalScanOps.Add(opid);                            
                        }
                        UpdateProgressFile();
                    }
                }
//KRYSS                DBConn.Open();
                SySal.DAQSystem.IDataProcessingServer Srv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcSrvAddress + ":" + (int)(SySal.DAQSystem.OperaPort.BatchServer) + "/DataProcessingServer.rem");
                lock (TotalScanOps)
                {
                    int RunningBatches = 0;
                    foreach (VolumeOpRecord vr in TotalScanOps)
                        if (vr.Paused) vr.AtomicOps.Clear();
                        else foreach (VolumeOpRecord.AtomicOpRecord ar in vr.AtomicOps)                            
                            if (ar.BatchId != 0) RunningBatches++;

                    ds = null;
                    SySal.OperaDb.OperaDbCommand cmd1 = new SySal.OperaDb.OperaDbCommand("SELECT DECODE(SUCCESS,'N',1,0) as ABORTED FROM TB_PROC_OPERATIONS WHERE ID = :id", DBConn);
                    cmd1.Parameters.Add("id", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                    int i;

                    for (i = 0; i < TotalScanOps.Count; i++)
                    {
                        if (System.IO.File.Exists(PrioritizeFileName(((VolumeOpRecord)TotalScanOps[i]).BrickId, ((VolumeOpRecord)TotalScanOps[i]).Id)))
                        {
                            try
                            {
                                System.IO.File.Delete(PrioritizeFileName(((VolumeOpRecord)TotalScanOps[i]).BrickId, ((VolumeOpRecord)TotalScanOps[i]).Id));
                            }
                            catch (Exception x)
                            {
                                Log("Error deleting prioritize file:\r\n" + x.ToString());
                            }
                            VolumeOpRecord vr = (VolumeOpRecord)TotalScanOps[i];
                            TotalScanOps.RemoveAt(i);
                            TotalScanOps.Insert(0, vr);
                            break;
                        }
                    }

                    for (i = 0; i < TotalScanOps.Count; i++)
                    {
                        long id = ((VolumeOpRecord)TotalScanOps[i]).Id;
                        int bkid = ((VolumeOpRecord)TotalScanOps[i]).BrickId;
                        ((VolumeOpRecord)TotalScanOps[i]).Paused = false;
                        ((VolumeOpRecord)TotalScanOps[i]).ExceptionText = null;
                        cmd1.Parameters[0].Value = id;
                        try
                        {
                            if (System.IO.File.Exists(StopFileName(bkid, id)))
                            {
                                TotalScanOps.RemoveAt(i--);
                                UpdateProgressFile();
                            }
                            else if ((SySal.OperaDb.Convert.ToInt32(cmd1.ExecuteScalar()) == 1 && System.IO.File.Exists(AutoPauseFileName(bkid, id)) == false) || System.IO.File.Exists(PauseFileName(bkid, id)))
                            {
                                try
                                {
                                    System.IO.File.AppendAllText(AutoPauseFileName(bkid, id), "process seen aborted");
                                }
                                catch (Exception) { }
                                ((VolumeOpRecord)TotalScanOps[i]).Paused = true;
                                continue;
                            }
                            else
                            {                                
                                VolumeOpRecord vr = (VolumeOpRecord)TotalScanOps[i];                                
                                int j;
                                for (j = 0; j < vr.AtomicOps.Count; j++)
                                {
                                    VolumeOpRecord.AtomicOpRecord ar = (VolumeOpRecord.AtomicOpRecord)vr.AtomicOps[j];
                                    if (ar.BatchId == 0) continue;
                                    try
                                    {
                                        if (Srv.DoneWith(ar.BatchId)) vr.AtomicOps.RemoveAt(j--);
                                    }
                                    catch (Exception)
                                    {
                                        vr.AtomicOps.RemoveAt(j--);
                                    }
                                }
                                System.Data.DataSet dsv = new DataSet();
                                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM TB_VOLUMES WHERE ID_EVENTBRICK = " + vr.BrickId + " AND ID_PROCESSOPERATION = " + vr.Id, DBConn).Fill(dsv);
                                int voltodo = dsv.Tables[0].Rows.Count;
                                int voltoexpand = dsv.Tables[0].Rows.Count;
                                foreach (System.Data.DataRow drv in dsv.Tables[0].Rows)
                                {
                                    if (System.IO.File.Exists(MTkTSRName(vr.Id, vr.BrickId, SySal.OperaDb.Convert.ToInt64(drv[0]))))
                                    {
                                        voltoexpand--;
                                        voltodo--;
                                        continue;
                                    }
                                    if (System.IO.File.Exists(TSRName(vr.Id, vr.BrickId, SySal.OperaDb.Convert.ToInt64(drv[0]))))
                                    {
                                        voltodo--;
                                        VolumeOpRecord.AtomicOpRecord chk_x = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(drv[0]), 0, VolumeOpRecord.AtomicOpRecord.OpType.MicrotrackExpand);
                                        if (vr.AtomicOps.Contains(chk_x) == false && RunningBatches < MaxBatches)
                                        {                                            
                                            if (ScheduleMicrotrackExpand(vr.Id, vr.BrickId, chk_x.VolumeId, vr)) RunningBatches++;                                            
                                        }
                                        continue;
                                    }
                                    ds = new DataSet();
                                    new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PLATE, ID_ZONE FROM TB_VOLUME_SLICES WHERE ID_EVENTBRICK = " + vr.BrickId + " AND ID_VOLUME = " + drv[0].ToString(), DBConn).Fill(ds);

                                    if (ds.Tables[0].Rows.Count == 0)
                                    {
                                        voltodo--;
                                        continue;
                                    }
                                    bool canrec = true;
                                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                                    {
                                        bool chk1;
                                        VolumeOpRecord.AtomicOpRecord chk_l = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(drv[0]), SySal.OperaDb.Convert.ToInt32(dr[0]), VolumeOpRecord.AtomicOpRecord.OpType.Link);
                                        VolumeOpRecord.AtomicOpRecord chk_i = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(drv[0]), SySal.OperaDb.Convert.ToInt32(dr[0]), VolumeOpRecord.AtomicOpRecord.OpType.AlignIgnore);
                                        VolumeOpRecord.AtomicOpRecord chk_g = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(drv[0]), SySal.OperaDb.Convert.ToInt32(dr[0]), VolumeOpRecord.AtomicOpRecord.OpType.GraphPlateSel);
                                        if (dr[1] == System.DBNull.Value) { canrec = false; }
                                        else if ((chk1 = vr.AtomicOps.Contains(chk_l)) || System.IO.File.Exists(TLGName(vr.Id, vr.BrickId, SySal.OperaDb.Convert.ToInt64(drv[0]), SySal.OperaDb.Convert.ToInt32(dr[0]))) == false)
                                        {
                                            canrec = false;
                                            if (chk1 == false)
                                            {
                                                //VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(dr[0]), SySal.OperaDb.Convert.ToInt32(dr[1]), VolumeOpRecord.AtomicOpRecord.OpType.Link);
                                                //if (vr.AtomicOps.Contains(ar) == false) 
                                                if (RunningBatches < MaxBatches)
                                                {
                                                    if (ScheduleLink(vr.Id, vr.BrickId, chk_l.VolumeId, chk_l.PlateId, vr)) RunningBatches++;                                                    
                                                }
                                            }
                                        }
                                        else if (vr.GraphPlateSelId > 0 && ((chk1 = vr.AtomicOps.Contains(chk_g)) || System.IO.File.Exists(GraphPlateSelName(vr.Id, vr.BrickId, SySal.OperaDb.Convert.ToInt64(drv[0]), SySal.OperaDb.Convert.ToInt32(dr[0]))) == false))
                                        {
                                            canrec = false;
                                            if (chk1 == false)
                                            {
                                                //VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(dr[0]), SySal.OperaDb.Convert.ToInt32(dr[1]), VolumeOpRecord.AtomicOpRecord.OpType.AlignIgnore);
                                                //if (vr.AtomicOps.Contains(ar) == false) 
                                                if (RunningBatches < MaxBatches)
                                                {
                                                    if (ScheduleGraphPlateSel(vr.Id, vr.BrickId, chk_g.VolumeId, chk_g.PlateId, vr)) RunningBatches++;
                                                }
                                            }
                                        }
                                        else if ((chk1 = vr.AtomicOps.Contains(chk_i)) || System.IO.File.Exists(IgnoreName(vr.Id, vr.BrickId, SySal.OperaDb.Convert.ToInt64(drv[0]), SySal.OperaDb.Convert.ToInt32(dr[0]))) == false)
                                        {
                                            canrec = false;
                                            if (chk1 == false)
                                            {
                                                //VolumeOpRecord.AtomicOpRecord ar = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(dr[0]), SySal.OperaDb.Convert.ToInt32(dr[1]), VolumeOpRecord.AtomicOpRecord.OpType.AlignIgnore);
                                                //if (vr.AtomicOps.Contains(ar) == false) 
                                                if (RunningBatches < MaxBatches)
                                                {
                                                    if (ScheduleIgnore(vr.Id, vr.BrickId, chk_i.VolumeId, chk_i.PlateId, vr)) RunningBatches++;                                                    
                                                }
                                            }
                                        }
                                    }
                                    VolumeOpRecord.AtomicOpRecord chk_r = new VolumeOpRecord.AtomicOpRecord(SySal.OperaDb.Convert.ToInt64(drv[0]), 0, VolumeOpRecord.AtomicOpRecord.OpType.Reconstruct);
                                    if (canrec)
                                    {                                        
                                        if (vr.AtomicOps.Contains(chk_r) == false && RunningBatches < MaxBatches)
                                        {   
                                            if (ScheduleRec(vr.Id, vr.BrickId, chk_r.VolumeId, vr, false)) RunningBatches++;                                            
                                        }
                                    }
                                    else if (System.IO.File.Exists(TemporaryRecName(vr.BrickId, vr.Id)))
                                    {
                                        ScheduleRec(vr.Id, vr.BrickId, chk_r.VolumeId, vr, true);
                                    }
                                }
                                if (voltoexpand == 0)
                                {
                                    vr.Completed = true;
                                    UpdateProgressFile();
                                }
                            }                            
                        }
                        catch (Exception x)
                        {                            
                            Log(x.ToString());
                        }

                    }
                    for (i = 0; i < TotalScanOps.Count; i++)
                    {
                        VolumeOpRecord vr = (VolumeOpRecord)TotalScanOps[i]; 
                        if (System.IO.File.Exists(TemporaryRecName(vr.BrickId, vr.Id)))
                            try
                            {
                                System.IO.File.Delete(TemporaryRecName(vr.BrickId, vr.Id));
                            }
                            catch (Exception x)
                            {
                                Log("Can't delete " + cmdTempRec + " file:\r\n" + x.ToString());
                            }
                    }
                }
//KRYSS                DBConn.Close();
            }
            catch (Exception x)
            {                
                Log(x.ToString());
            }
            finally
            {
//KRYSS                DBConn.Close();
             
            }
        }
        
        internal static string ListFilePath(int bkid, long opid, long volid)
        {
            return DirName(bkid, opid) + "\\list_" + volid + ".txt";
        }

        internal static string ListMTkExpandFilePath(int bkid, long opid, long volid)
        {
            return DirName(bkid, opid) + "\\mtkx_" + volid + ".txt";
        }

        internal static string DirName(int bkid, long opid)
        {
            return ScratchDir + "\\" + WorkDirName + bkid + "_" + opid;
        }

        internal static string MTkTSRName(long opid, int bkid, long volid)
        {
            return DirName(bkid, opid) + "\\mtkvol_" + bkid + "_" + volid + ".tsr";
        }

        internal static string TSRName(long opid, int bkid, long volid)
        {
            return DirName(bkid, opid) + "\\volume_" + bkid + "_" + volid + ".tsr";
        }

        internal static string TempTSRName(long opid, int bkid, long volid)
        {
            return DirName(bkid, opid) + "\\temprec_" + bkid + "_" + volid + ".tsr";
        }

        internal static string TLGName(long opid, int bkid, long volid, int plateid)
        {
            return DirName(bkid, opid) + "\\plate_" + bkid + "_" + volid + "_" + plateid + ".tlg";
        }

        internal static string IgnoreName(long opid, int bkid, long volid, int plateid)
        {
            return DirName(bkid, opid) + "\\plate_" + bkid + "_" + volid + "_" + plateid + ".tlg.bi";
        }

        internal static string GraphPlateSelName(long opid, int bkid, long volid, int plateid)
        {
            return DirName(bkid, opid) + "\\graphsel_" + bkid + "_" + volid + "_" + plateid + ".txt";
        }

        internal static string TemporaryRecName(int bkid, long opid)
        {
            return DirName(bkid, opid) + "\\" + cmdTempRec + ".aps";
        }

        internal static string StopFileName(int bkid, long opid)
        {
            return DirName(bkid, opid) + "\\" + cmdStop + ".aps";
        }

        internal static string PrioritizeFileName(int bkid, long opid)
        {
            return DirName(bkid, opid) + "\\" + cmdPrioritize + ".aps";
        }

        internal static string PauseFileName(int bkid, long opid)
        {
            return DirName(bkid, opid) + "\\" + cmdPause + ".aps";
        }

        internal static string AutoPauseFileName(int bkid, long opid)
        {
            return DirName(bkid, opid) + "\\" + cmdAutoPause + ".aps";
        }

        protected override void OnStop()
        {
            EventLog.WriteEntry("Service stopping.");
            CleanBatches();
            DBConn.Close();
        }

        #region IWebApplication Members

        /// <summary>
        /// Name of the web application.
        /// </summary>
        public string ApplicationName
        {
            get { return "OperaAutoProcessServer"; }
        }

        const string StopCmd = "stop";
        const string PauseCmd = "pause";
        const string ResumeCmd = "resume";
        const string PrioritizeCmd = "prioritize";
        const string TempRecCmd = "temprec";
        const string AddQueueCmd = "addqueue";        

        const string UserIdCmd = "uid";
        const string PasswordIdCmd = "pwd";

        class SessionData
        {
            public long UserId;
            public int BrickId;
            public long ProcOp;
            public int MinGrains = 20;
            public double MaxSigma = 2;
            public double MaxDPos = 5000;
            public double MaxDSlope = 0.5;
            public int AzimuthDeg = 45;
            public int SlopeDeg = 165;
            public double Zoom = 0.002;
            public int SelZones = 0;
            public int TotalZones = 0;
            public bool ShowZones;
            public System.IO.MemoryStream PlotStream;
            public GDI3D.Plot.Plot Plot;
        }

        /// <summary>
        /// Processes POST methods, such as login information.
        /// </summary>
        /// <param name="sess">the Web Session.</param>
        /// <param name="page">the page requested (ignored).</param>
        /// <param name="postfields">the parameters posted.</param>
        /// <returns>The requested information in HTML format.</returns>
        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
        {
            if (sess.UserData == null)
            {
                string user = "";
                string pwd = "";
                if (postfields != null)
                    foreach (string q in postfields)
                    {
                        if (q == null) Log("NULL PASSED");
                        else Log(q);
                        if (q.ToLower().StartsWith(UserIdCmd + "=")) user = q.Substring(UserIdCmd.Length + 1);
                        else if (q.ToLower().StartsWith(PasswordIdCmd + "=")) pwd = q.Substring(PasswordIdCmd.Length + 1);
                    }
                if (user.Length > 0)
                {
                    lock (DBConnWWW)
                    {
                        try
                        {
                            DBConnWWW.Open();
                            long userid = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(user, pwd, DBConnWWW, null);
                            SySal.OperaDb.ComputingInfrastructure.UserPermission perm = new SySal.OperaDb.ComputingInfrastructure.UserPermission();
                            perm.DB_Site_Id = IdSite;
                            perm.Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.ProcessData;
                            perm.Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
                            if (SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(userid, new SySal.OperaDb.ComputingInfrastructure.UserPermission[1] { perm }, true, DBConnWWW, null) == false) throw new Exception();
                            SessionData sd = new SessionData();
                            sd.UserId = userid;
                            sess.UserData = sd;
                        }
                        catch (Exception x)
                        {
                            sess.UserData = null;
                        }
                        finally
                        {
                            DBConnWWW.Close();
                        }
                    }
                }
            }
            if (sess.UserData == null)
            {
                return new SySal.Web.HTMLResponse(
                    "<html><head><meta http-equiv=\"pragma\" content=\"no-cache\"><meta http-equiv=\"EXPIRES\" content=\"0\"><title>OperaAutoProcessServer Login</title></head><body>\r\n" +
                    " <div align=\"center\">\r\n" +
                    "  <h1>OperaAutoProcessServer Login</h1>\r\n" +
                    "  <form action=\"/\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n" +
                    "  <table align=\"center\" border=\"0\">\r\n" +
                    "  <tr><td>Username</td><td><input id=\"" + UserIdCmd + "\" maxlength=\"30\" name=\"" + UserIdCmd + "\" size=\"30\" type=\"text\" /></td></tr>\r\n" +
                    "  <tr><td>Password</td><td><input id=\"" + PasswordIdCmd + "\" name=\"" + PasswordIdCmd + "\" size=\"30\" type=\"password\" /></td></tr>\r\n" +
                    "  </table>\r\n" +
                    "  <input id=\"Button1\" type=\"submit\" value=\"Login\" />\r\n" +
                    "  </form>\r\n" +
                    " </div>\r\n" +
                    "</body></html>\r\n"
                    );
            }
            else return HttpGet(sess, page, postfields);
        }

        bool m_ShowExceptions = false;

        internal void SetShowExceptions(bool sh) { m_ShowExceptions = sh; }

        /// <summary>
        /// Defines whether exceptions should be shown.
        /// </summary>
        public bool ShowExceptions
        {
            get { return m_ShowExceptions; }
        }

        const string GraphSelPage = "/gsel.htm";
        const string GraphSelCmd = "gsel";
        const string PlotPage = "/plot.png";

        /// <summary>
        /// Processes GET methods, such as the status of processing batches.
        /// </summary>
        /// <param name="sess">the Web Session.</param>
        /// <param name="page">the page requested (ignored).</param>
        /// <param name="queryget">the parameters in the URL.</param>
        /// <returns>The requested information in HTML format.</returns>
        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            if (String.Compare(GraphSelPage, page, true) == 0) return HttpGetGraphSel(sess, page, queryget);
            if (String.Compare(PlotPage, page, true) == 0) return HttpGetPlot(sess, page, queryget);
            return HttpGetRoot(sess, page, queryget);
        }

        private SySal.Web.ChunkedResponse HttpGetPlot(SySal.Web.Session sess, string page, params string[] queryget)
        {
            SessionData sd = (SessionData)sess.UserData;
            if (sd.PlotStream == null)
            {
                sd.PlotStream = new System.IO.MemoryStream();
                System.Drawing.Image im = new System.Drawing.Bitmap(50, 50);
                im.Save(sd.PlotStream, System.Drawing.Imaging.ImageFormat.Png);                
            }
            sd.PlotStream.Seek(0, System.IO.SeekOrigin.Begin);
            return new SySal.Web.ByteArrayResponse(65536, sd.PlotStream.GetBuffer(), "image/png");
        }

        const string GraphProcOpCmd = "op";
        const string GraphRotLeftCmd = "rl";
        const string GraphRotRightCmd = "rr";
        const string GraphRotUpCmd = "ru";
        const string GraphRotDownCmd = "rd";
        const string GraphZoomInCmd = "zi";
        const string GraphZoomOutCmd = "zo";
        const string GraphRefreshCmd = "refresh";
        const string GraphDblClickCmd = "dbck";
        const string GraphMinGrains = "mg";
        const string GraphMaxSigma = "ms";
        const string GraphMaxDSlope = "mds";
        const string GraphMaxDPos = "mdp";
        const string GraphCheckZonesCmd = "shz";
        const string GraphFilterCmd = "xf";

        private SySal.Web.ChunkedResponse HttpGetGraphSel(SySal.Web.Session sess, string page, params string[] queryget)
        {
            if (sess.UserData == null) return HttpPost(sess, page, queryget);
            SessionData sd = (SessionData)sess.UserData;
            bool refresh = false;
            string owner = null;
            int tempint;
            double tempdouble;
            bool showzones = false;
            bool filter = false;
            foreach (string s in queryget)
            {
                if (s.StartsWith(GraphFilterCmd)) filter = true;
                else if (s.StartsWith(GraphRefreshCmd)) refresh = true;
                else if (s.StartsWith(GraphMinGrains) && sd.Plot != null)
                    try
                    {
                        tempint = Convert.ToInt32(s.Substring(GraphMinGrains.Length + 1));
                        if (tempint != sd.MinGrains)
                        {
                            sd.MinGrains = tempint;                            
                        }
                    }
                    catch (Exception) { }
                else if (s.StartsWith(GraphMaxSigma) && sd.Plot != null)
                    try
                    {
                        tempdouble = Convert.ToDouble(s.Substring(GraphMaxSigma.Length + 1));
                        if (tempdouble != sd.MaxSigma)
                        {
                            sd.MaxSigma = tempdouble;                            
                        }
                    }
                    catch (Exception) { }
                else if (s.StartsWith(GraphMaxDSlope) && sd.Plot != null)
                    try
                    {
                        tempdouble = Convert.ToDouble(s.Substring(GraphMaxDSlope.Length + 1));
                        if (tempdouble != sd.MaxDSlope)
                        {
                            sd.MaxDSlope = tempdouble;                            
                        }
                    }
                    catch (Exception) { }
                else if (s.StartsWith(GraphMaxDPos) && sd.Plot != null)
                    try
                    {
                        tempdouble = Convert.ToDouble(s.Substring(GraphMaxDPos.Length + 1));
                        if (tempdouble != sd.MaxDPos)
                        {
                            sd.MaxDPos = tempdouble;                            
                        }
                    }
                    catch (Exception) { }
                else if (s.StartsWith(GraphCheckZonesCmd))
                    showzones = true;
            }
            if (filter)
            {
                sd.ShowZones = showzones;
                refresh = true;
            }
            foreach (string s in queryget)
            {
                if (s.StartsWith(GraphProcOpCmd + "="))
                    try
                    {
                        long newop = Convert.ToInt64(s.Substring(GraphProcOpCmd.Length + 1));
                        if (newop != sd.ProcOp)
                        {
                            lock (DBConnWWW)
                                try
                                {
                                    DBConnWWW.Open();
                                    sd.BrickId = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT ID_EVENTBRICK FROM TB_PROC_OPERATIONS WHERE ID = " + newop, DBConnWWW).ExecuteScalar());
                                    sd.ProcOp = newop;
                                    refresh = true;
                                }
                                catch (Exception x)
                                {
                                    sd.BrickId = 0;
                                    sd.ProcOp = 0;
                                    sd.PlotStream = null;
                                    sd.Plot = null;
                                    Log("GraphException:\r\n" + x.ToString());
                                }
                                finally
                                {
                                    DBConnWWW.Close();
                                }
                        }
                    }
                    catch (Exception)
                    {
                        sd.ProcOp = 0;
                    }
                else if (s.StartsWith(GraphRotLeftCmd) && sd.Plot != null) { sd.AzimuthDeg += 15; UpdatePlot(sd); }
                else if (s.StartsWith(GraphRotRightCmd) && sd.Plot != null) { sd.AzimuthDeg -= 15; UpdatePlot(sd); }
                else if (s.StartsWith(GraphRotUpCmd) && sd.Plot != null) { sd.SlopeDeg += 15; UpdatePlot(sd); }
                else if (s.StartsWith(GraphRotDownCmd) && sd.Plot != null) { sd.SlopeDeg -= 15; UpdatePlot(sd); }
                else if (s.StartsWith(GraphZoomInCmd) && sd.Plot != null) { sd.Zoom *= 1.25; UpdatePlot(sd); }
                else if (s.StartsWith(GraphZoomOutCmd) && sd.Plot != null) { sd.Zoom *= 0.8; UpdatePlot(sd); }                
                else if (s.StartsWith(GraphDblClickCmd) && sd.Plot != null)
                    try
                    {
                        UpdatePlot(sd);
                        string[] tokens = s.Substring(GraphDblClickCmd.Length + 1).Split('_');
                        owner = sd.Plot.FindNearestObject(Convert.ToInt32(tokens[0]), Convert.ToInt32(tokens[1]), false);
                    }
                    catch (Exception)
                    {
                        owner = null;
                    }
            }
            if (refresh && sd.ProcOp != 0)
            {
                VolumeOpRecord vr = null;
                lock (TotalScanOpsCopy)
                {
                    vr = new VolumeOpRecord(sd.ProcOp, sd.BrickId, false, true);
                    vr = (VolumeOpRecord)TotalScanOpsCopy[TotalScanOpsCopy.IndexOf(vr)];
                }
                lock (DBConnWWW)
                    try
                    {
                        DBConnWWW.Open();
                        MakePlot(sd, vr);
                    }
                    catch (Exception x)
                    {
                        Log("GraphException: \r\n" + x.ToString());
                    }
                    finally
                    {
                        DBConnWWW.Close();
                    }
            }

            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>OperaAutoProcess Graphical Monitor - " + MachineName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Verdana,Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Verdana,Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Verdana,Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    body {font-family: Verdana,Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "<p><b>OperaAutoProcess Graphical Monitor (" + MachineName + ")<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></p>\r\n" +
                            "<p>Brick: " + sd.BrickId + " ProcOp: " + sd.ProcOp + " Zones: " + sd.SelZones + "/" + sd.TotalZones + "</p>\r\n" +
                            "  <form action=\"" + GraphSelPage + "\" method=\"get\" enctype=\"application/x-www-form-urlencoded\">\r\n" +
                            "  <table align=\"center\" border=\"0\" width=\"1\">\r\n";
            if (sd.ProcOp > 0)
            {
                string ticks = System.DateTime.Now.Ticks.ToString();
                html +=
                    "  <hr>\r\n" +
                    "   <div align=\"center\">\r\n" +
                    "    <a href=\"" + GraphSelPage + "?" + GraphRotLeftCmd + "=" + ticks + "\">left</a>&nbsp;<a href=\"" + GraphSelPage + "?" + GraphRotRightCmd + "=" + ticks + "\">right</a>&nbsp;<a href=\"" + GraphSelPage + "?" + GraphRotUpCmd + "=" + ticks + 
                          "\">up</a>&nbsp;<a href=\"" + GraphSelPage + "?" + GraphRotDownCmd + "=" + ticks + "\">down</a>&nbsp;<a href=\"" + GraphSelPage + "?" + GraphZoomInCmd + "=" + ticks + "\">in</a>&nbsp;<a href=\"" + GraphSelPage + "?" + GraphZoomOutCmd +
                          "=" + ticks + "\">out</a>&nbsp;<a href=\"" + GraphSelPage + "?" + GraphRefreshCmd + "=" + ticks + "\">refresh</a>\r\n" +
                    "   </div>\r\n" +
                    "   <table width=\"1\" align=\"center\">\r\n" +
                    "    <tr>\r\n" +
                    "     <td><input id=\"" + GraphFilterCmd + "\" name=\"" + GraphFilterCmd + "\" type=\"submit\" value=\"Filter\" />\r\n" +
                    "     <td rowspan=6><img width=\"" + GraphPlotWidth + "\" height=\"" + GraphPlotHeight + "\" src=\"" + PlotPage + "?t=" + ticks + "\" ondblclick=\"document.parentWindow.location='" + GraphSelPage + "?t=" + ticks + "&" + GraphDblClickCmd + "=' + window.event.offsetX + '_' + window.event.offsetY;\"><td>\r\n" +
                    "    </tr>\r\n" +
                    "    <tr><td>Min. Grains <input type=\"text\" size=\"6\" name=\"" + GraphMinGrains + "\" id=\"" + GraphMinGrains + "\" value=\"" + sd.MinGrains + "\" /></td></tr>" +
                    "    <tr><td>Max. Sigma <input type=\"text\" size=\"6\" name=\"" + GraphMaxSigma + "\" id=\"" + GraphMaxSigma + "\" value=\"" + sd.MaxSigma + "\" /></td></tr>\r\n" +
                    "    <tr><td>Max. &Delta;Slope <input type=\"text\" size=\"6\" name=\"" + GraphMaxDSlope + "\" id=\"" + GraphMaxDSlope + "\" value=\"" + sd.MaxDSlope + "\" /></td></tr>\r\n" +
                    "    <tr><td>Max. &Delta;Pos <input type=\"text\" size=\"6\" name=\"" + GraphMaxDPos + "\" id=\"" + GraphMaxDPos + "\" value=\"" + sd.MaxDPos + "\" /></td></tr>\r\n" +
                    "    <tr><td>Show Zones<input type=\"checkbox\" name=\"" + GraphCheckZonesCmd + "\" id=\"" + GraphCheckZonesCmd + "\" " + (sd.ShowZones ? " CHECKED " : "") + " /></td></tr>\r\n" + 
                    "   </table>\r\n";
                if (owner != null)
                    html += "  <p><font color=\"#009900\">" + SySal.Web.WebServer.HtmlFormat(owner) + "</font></p>\r\n";
            }
            html +=
                    "  </table>\r\n" +
                    "  </form>\r\n" +
                    " </div>\r\n" +
                    " <div align=\"center\"><a href=\"/\">Main page</a></div>\r\n" +
                    "</body></html>\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        private static void GetSpottingVectors(SessionData sd, out SySal.BasicTypes.Vector D, out SySal.BasicTypes.Vector N)
        {
            double tha = Math.PI * sd.AzimuthDeg / 180.0;
            double ths = Math.PI * sd.SlopeDeg / 180.0;
            double ca = Math.Cos(tha);
            double sa = Math.Sin(tha);
            double cs = Math.Cos(ths);
            double ss = Math.Sin(ths);
            D = new SySal.BasicTypes.Vector();
            D.Z = -ss;
            D.X = -ca * cs;
            D.Y = -sa * cs;
            N = new SySal.BasicTypes.Vector();
            N.Z = cs;
            N.X = -ca * ss;
            N.Y = -sa * ss;
        }

        const int GraphPlotWidth = 800;
        const int GraphPlotHeight = 600;

        private static void MakePlot(SessionData sd, VolumeOpRecord vr)
        {
            SySal.OperaDb.OperaDbDataReader rd1 = null;
            try
            {
                rd1 = new SySal.OperaDb.OperaDbCommand("SELECT (MINX - ZEROX), (MAXX - ZEROX), (MINY - ZEROY), (MAXY - ZEROY) FROM TB_EVENTBRICKS WHERE ID = " + vr.BrickId, DBConnWWW).ExecuteReader();
                rd1.Read();
                SySal.BasicTypes.Cuboid qbe = new SySal.BasicTypes.Cuboid();
                qbe.MinX = rd1.GetDouble(0);
                qbe.MaxX = rd1.GetDouble(1);
                qbe.MinY = rd1.GetDouble(2);
                qbe.MaxY = rd1.GetDouble(3);
                rd1.Close();
                rd1 = new SySal.OperaDb.OperaDbCommand("SELECT MIN(Z), MAX(Z) FROM TB_PLATES WHERE ID_EVENTBRICK = " + vr.BrickId, DBConnWWW).ExecuteReader();
                rd1.Read();
                qbe.MinZ = rd1.GetDouble(0) - 255.0;
                qbe.MaxZ = rd1.GetDouble(1) + 45.0;
                rd1.Close();
                System.Collections.ArrayList plarr = new System.Collections.ArrayList();
                rd1 = new SySal.OperaDb.OperaDbCommand("SELECT ID_PLATE, ID_VOLUME FROM TB_VOLUME_SLICES WHERE (ID_EVENTBRICK, ID_VOLUME) IN (SELECT ID_EVENTBRICK, ID FROM TB_VOLUMES WHERE ID_EVENTBRICK = " + vr.BrickId + " AND ID_PROCESSOPERATION = " + vr.Id + ") AND ID_ZONE IS NOT NULL", DBConnWWW).ExecuteReader();
                while (rd1.Read())
                    plarr.Add(new object[] { rd1.GetInt32(0), rd1.GetInt64(1) } );                
                rd1.Close();

                GDI3D.Plot.Plot gdiPlot = new GDI3D.Plot.Plot(GraphPlotWidth, GraphPlotHeight);

                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MinX, qbe.MinY, qbe.MaxZ, qbe.MaxX, qbe.MinY, qbe.MaxZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MaxX, qbe.MinY, qbe.MaxZ, qbe.MaxX, qbe.MaxY, qbe.MaxZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MaxX, qbe.MaxY, qbe.MaxZ, qbe.MinX, qbe.MaxY, qbe.MaxZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MinX, qbe.MaxY, qbe.MaxZ, qbe.MinX, qbe.MinY, qbe.MaxZ, null, 64, 64, 192));

                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MinX, qbe.MinY, qbe.MinZ, qbe.MaxX, qbe.MinY, qbe.MinZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MaxX, qbe.MinY, qbe.MinZ, qbe.MaxX, qbe.MaxY, qbe.MinZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MaxX, qbe.MaxY, qbe.MinZ, qbe.MinX, qbe.MaxY, qbe.MinZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MinX, qbe.MaxY, qbe.MinZ, qbe.MinX, qbe.MinY, qbe.MinZ, null, 64, 64, 192));

                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MinX, qbe.MinY, qbe.MinZ, qbe.MinX, qbe.MinY, qbe.MaxZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MinX, qbe.MaxY, qbe.MinZ, qbe.MinX, qbe.MaxY, qbe.MaxZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MaxX, qbe.MaxY, qbe.MinZ, qbe.MaxX, qbe.MaxY, qbe.MaxZ, null, 64, 64, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(qbe.MaxX, qbe.MinY, qbe.MinZ, qbe.MaxX, qbe.MinY, qbe.MaxZ, null, 64, 64, 192));

                gdiPlot.Add(new GDI3D.Plot.Line(-10000, -10000, 0, 0, -10000, 0, "X", 192, 192, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(-10000, -10000, 0, -10000, 0, 0, "Y", 192, 192, 192));
                gdiPlot.Add(new GDI3D.Plot.Line(-10000, -10000, 0, -10000, -10000, 10000, "Z", 192, 192, 192));

                int grains;
                SySal.BasicTypes.Vector pos = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector2 slope = new SySal.BasicTypes.Vector2();
                double sigma;
                SySal.BasicTypes.Vector2 dpos = new SySal.BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 dslope = new SySal.BasicTypes.Vector2();
                sd.TotalZones = plarr.Count;
                sd.SelZones = 0;

                foreach (object [] volslice in plarr)
                {
                    if (System.IO.File.Exists(GraphPlateSelName(vr.Id, vr.BrickId, (long)volslice[1], (int)volslice[0])))
                        try
                        {
                            sd.SelZones++;
                            string[] lines = System.IO.File.ReadAllLines(GraphPlateSelName(vr.Id, vr.BrickId, (long)volslice[1], (int)volslice[0]));
                            foreach (string line in lines)
                            {
                                string[] tokens = line.Split(' ', '\t');
                                if (tokens.Length == 13)
                                {
                                    grains = Convert.ToInt32(tokens[2]);
                                    pos.X = Convert.ToDouble(tokens[3]);
                                    pos.Y = Convert.ToDouble(tokens[4]);
                                    pos.Z = Convert.ToDouble(tokens[5]);
                                    slope.X = Convert.ToDouble(tokens[6]);
                                    slope.Y = Convert.ToDouble(tokens[7]);
                                    sigma = Convert.ToDouble(tokens[8]);
                                    dpos.X = Convert.ToDouble(tokens[9]);
                                    dpos.Y = Convert.ToDouble(tokens[10]);
                                    dslope.X = Convert.ToDouble(tokens[11]);
                                    dslope.Y = Convert.ToDouble(tokens[12]);
                                    if (grains < sd.MinGrains || sigma > sd.MaxSigma) continue;
                                    if (dslope.X * dslope.X + dslope.Y * dslope.Y > sd.MaxDSlope * sd.MaxDSlope) continue;
                                    if (dpos.X * dpos.X + dpos.Y * dpos.Y > sd.MaxDPos * sd.MaxDPos) continue;

                                    double lev = (pos.Z - qbe.MinZ) / (qbe.MaxZ - qbe.MinZ);
                        			if (lev < 0.0) lev = 0.0;
                        			else if (lev > 1.0) lev = 1.0;

                                    int r = (int)(255 * (1.0 - lev * lev) * (1.0 - lev * lev));
                                    int g = (int)(255 * (4.0 * lev * (1.0 - lev)));
                                    int b = (int)(255 * lev * (2.0 - lev) * lev * (2.0 - lev));

                                    gdiPlot.Add(new GDI3D.Plot.Line(
                                        pos.X + 545.0 * slope.X, pos.Y + 545.0 * slope.Y, pos.Z + 545.0,
                                        pos.X - 755.0 * slope.X, pos.Y - 755.0 * slope.Y, pos.Z - 755.0,
                                        "Pl " + tokens[1] + " G " + tokens[2] + " PX " + pos.X.ToString("F1") + " PY " + pos.Y.ToString("F1") + " PZ " + pos.Z.ToString("F1") +
                                        " SX " + slope.X.ToString("F4") + " SY " + slope.Y.ToString("F4") + " S " + sigma.ToString("F4") +
                                        " DPX " + dpos.X.ToString("F1") + " DPY " + dpos.Y.ToString("F1") + " DSX " + dslope.X.ToString("F4") + " DSY " + dslope.Y.ToString("F4"),
                                        r, g, b));
                                }
                            }
                        }
                        catch (Exception)
                        {

                        }
                }

                if (sd.ShowZones)
                {
                    rd1 = new SySal.OperaDb.OperaDbCommand("SELECT Z, MINX, MAXX, MINY, MAXY, ZD FROM TB_PLATES INNER JOIN " +
                        "(SELECT ID_EVENTBRICK AS IDB, ID_PLATE, MINX, MAXX, MINY, MAXY, NVL2(ID_ZONE, 1, 0) AS ZD FROM TB_VOLUME_SLICES WHERE (ID_EVENTBRICK, ID_VOLUME) IN (SELECT ID_EVENTBRICK, ID FROM TB_VOLUMES WHERE ID_EVENTBRICK = " +
                        sd.BrickId + " AND ID_PROCESSOPERATION = " + sd.ProcOp + ")) ON (ID_EVENTBRICK = IDB AND ID_PLATE = ID)", DBConnWWW).ExecuteReader();
                    while (rd1.Read())
                    {
                        bool done = (rd1.GetInt32(5) > 0);

                        SySal.BasicTypes.CCuboid cbe = new SySal.BasicTypes.CCuboid();
                        cbe.Q.MinX = rd1.GetDouble(1);
                        cbe.Q.MaxX = rd1.GetDouble(2);
                        cbe.Q.MinY = rd1.GetDouble(3);
                        cbe.Q.MaxY = rd1.GetDouble(4);
                        cbe.Q.MaxZ = rd1.GetDouble(0) + 45.0;
                        cbe.Q.MinZ = cbe.Q.MaxZ - 300.0;

                        if (done)
                        {
                            cbe.C.Red = 128;
                            cbe.C.Green = 224;
                            cbe.C.Blue = 128;
                        }
                        else
                        {
                            cbe.C.Red = 160;
                            cbe.C.Green = 160;
                            cbe.C.Blue = 160;
                        }

                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MinX, cbe.Q.MinY, cbe.Q.MaxZ, cbe.Q.MaxX, cbe.Q.MinY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MaxX, cbe.Q.MinY, cbe.Q.MaxZ, cbe.Q.MaxX, cbe.Q.MaxY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MaxX, cbe.Q.MaxY, cbe.Q.MaxZ, cbe.Q.MinX, cbe.Q.MaxY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MinX, cbe.Q.MaxY, cbe.Q.MaxZ, cbe.Q.MinX, cbe.Q.MinY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));

                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MinX, cbe.Q.MinY, cbe.Q.MinZ, cbe.Q.MaxX, cbe.Q.MinY, cbe.Q.MinZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MaxX, cbe.Q.MinY, cbe.Q.MinZ, cbe.Q.MaxX, cbe.Q.MaxY, cbe.Q.MinZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MaxX, cbe.Q.MaxY, cbe.Q.MinZ, cbe.Q.MinX, cbe.Q.MaxY, cbe.Q.MinZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MinX, cbe.Q.MaxY, cbe.Q.MinZ, cbe.Q.MinX, cbe.Q.MinY, cbe.Q.MinZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));

                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MinX, cbe.Q.MinY, cbe.Q.MinZ, cbe.Q.MinX, cbe.Q.MinY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MinX, cbe.Q.MaxY, cbe.Q.MinZ, cbe.Q.MinX, cbe.Q.MaxY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MaxX, cbe.Q.MaxY, cbe.Q.MinZ, cbe.Q.MaxX, cbe.Q.MaxY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));
                        gdiPlot.Add(new GDI3D.Plot.Line(cbe.Q.MaxX, cbe.Q.MinY, cbe.Q.MinZ, cbe.Q.MaxX, cbe.Q.MinY, cbe.Q.MaxZ, null, (int)cbe.C.Red, (int)cbe.C.Green, (int)cbe.C.Blue));

                    }
                    rd1.Close();
                }
                
                gdiPlot.SetCameraSpotting(0.5 * (qbe.MinX + qbe.MaxX), 0.5 * (qbe.MinY + qbe.MaxY), 0.5 * (qbe.MinZ + qbe.MaxZ));
                gdiPlot.LineWidth = 2;
                gdiPlot.Alpha = 0.75;
                sd.Plot = gdiPlot;

                UpdatePlot(sd);
            }
            finally
            {
                if (rd1 != null) rd1.Close();
            }
        }

        private static void UpdatePlot(SessionData sd)
        {            
            SySal.BasicTypes.Vector P = new SySal.BasicTypes.Vector();
            SySal.BasicTypes.Vector D, N;
            sd.Plot.GetCameraSpotting(ref P.X, ref P.Y, ref P.Z);
            GetSpottingVectors(sd, out D, out N);
            sd.Plot.SetCameraOrientation(D.X, D.Y, D.Z, N.X, N.Y, N.Z);
            sd.Plot.SetCameraSpotting(P.X, P.Y, P.Z);
            sd.Plot.Distance = 1000000;
            sd.Plot.Infinity = true;
            sd.Plot.Zoom = sd.Zoom;
            sd.Plot.Transform();
            sd.PlotStream = new System.IO.MemoryStream();
            sd.Plot.Save(sd.PlotStream, System.Drawing.Imaging.ImageFormat.Gif);
        }

        private SySal.Web.ChunkedResponse HttpGetRoot(SySal.Web.Session sess, string page, params string[] queryget)
        {
            if (sess.UserData == null) return HttpPost(sess, page, queryget);
            bool sort = true;
            bool refresh = false;
            if (queryget != null)
                foreach (string q in queryget)
                {
                    if (q.ToLower() == "sort=false")
                        sort = false;
                    else
                    {
                        int checkeq = q.IndexOf("=");
                        if (checkeq >= 0)
                        {
                            long opid = 0;
                            int bkid = 0;
                            string p = q.Substring(0, checkeq).ToLower();
                            string[] v = q.Substring(checkeq + 1).Split('_');
                            try
                            {
                                opid = System.Convert.ToInt64(v[1]);
                                bkid = System.Convert.ToInt32(v[0]);
                            }
                            catch (Exception)
                            {
                                opid = 0;
                                bkid = 0;
                            }
                            switch (p)
                            {
                                case StopCmd:
                                    try
                                    {
                                        System.IO.File.WriteAllText(StopFileName(bkid, opid), "www");
                                    }
                                    catch (Exception x) { Log(StopCmd); Log(x.ToString()); }
                                    refresh = true;
                                    break;

                                case PauseCmd:
                                    try
                                    {
                                        System.IO.File.WriteAllText(PauseFileName(bkid, opid), "www");
                                    }
                                    catch (Exception x) { Log(PauseCmd); Log(x.ToString()); }
                                    refresh = true;
                                    break;

                                case ResumeCmd:
                                    try
                                    {
                                        System.IO.File.Delete(PauseFileName(bkid, opid));
                                    }
                                    catch (Exception x) { Log(ResumeCmd); Log(x.ToString()); }
                                    refresh = true;
                                    break;

                                case PrioritizeCmd:
                                    try
                                    {
                                        System.IO.File.WriteAllText(PrioritizeFileName(bkid, opid), "www");
                                    }
                                    catch (Exception x) { Log(PrioritizeCmd); Log(x.ToString()); }
                                    refresh = true;
                                    break;

                                case TempRecCmd:
                                    try
                                    {
                                        System.IO.File.WriteAllText(TemporaryRecName(bkid, opid), "www");
                                    }
                                    catch (Exception x) { Log(TempRecCmd); Log(x.ToString()); }
                                    refresh = true;
                                    break;

                                case AddQueueCmd:
                                    try
                                    {
                                        System.IO.File.AppendAllText(AddQueueName, "\r\n" + opid);
                                    }
                                    catch (Exception x) { Log(AddQueueCmd); Log(x.ToString()); }
                                    refresh = true;
                                    break;
                            }
                        }
                    }
                }            
            if (refresh) return new SySal.Web.HTMLResponse("<html><head><meta http-equiv=\"pragma\" content=\"no-cache\"><meta http-equiv=\"EXPIRES\" content=\"0\"><meta http-equiv=\"refresh\" content=\"0; URL=?sort=" + sort + "\"></head><body><a href=\"?sort=" + sort + "\">OperaAutoProcessServer - Refresh</a></body></html>\r\n");
            return new SySal.Web.HTMLResponse(HtmlProgress(sort, true));
        }

        #endregion
    }
}
