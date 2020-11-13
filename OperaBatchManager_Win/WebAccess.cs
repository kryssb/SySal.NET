using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Services.OperaBatchManager_Win
{
    internal class WebAccess : SySal.Web.IWebApplication
    {
        System.Collections.Specialized.OrderedDictionary m_MachineNames = new System.Collections.Specialized.OrderedDictionary();

        internal string GetMachineName(long id)
        {
            if (m_MachineNames.Contains(id)) return m_MachineNames[(object)id].ToString();
            SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);            
            try
            {
                conn.Open();
                SySal.OperaDb.ComputingInfrastructure.Machine m = new SySal.OperaDb.ComputingInfrastructure.Machine(id, conn, null);
                m_MachineNames.Add((object)id, m.Name);
                return m.Name;
            }
            catch (Exception)
            {
                return id.ToString();
            }
            finally
            {
                if (conn != null) conn.Close();
            }
        }

        #region IWebApplication Members       

        bool m_ShowExceptions = false;

        internal void SetShowExceptions(bool sh) { m_ShowExceptions = sh; }

        /// <summary>
        /// Defines whether exceptions should be shown.
        /// </summary>
        public bool ShowExceptions
        {
            get { return m_ShowExceptions; }
        }

        /// <summary>
        /// The name of the Web Application.
        /// </summary>
        public string ApplicationName
        {
            get { return "OperaBatchManager"; }
        }

        /// <summary>
        /// Routes GET methods as POST methods.
        /// </summary>
        /// <param name="sess">Session information.</param>
        /// <param name="page">ignored.</param>
        /// <param name="queryget">the action parameters passed.</param>
        /// <returns>the status page.</returns>
        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        static System.Text.RegularExpressions.Regex rx_drivers = new System.Text.RegularExpressions.Regex(@"/(\d+)");

        /// <summary>
        /// Handles POST methods.
        /// </summary>
        /// <param name="sess">Session information.</param>
        /// <param name="page">ignored.</param>
        /// <param name="postfields">the action parameters passed.</param>
        /// <returns>the status page.</returns>
        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
        {
            if (page.ToLower().StartsWith(DPSPage)) return HttpDPSPage(sess, page, postfields);
            if (page.ToLower().StartsWith(BMPage)) return HttpBMPage(sess, page, postfields);
            if (page.ToLower().StartsWith(ProcStartPage)) return HttpProcStartPage(sess, page, postfields);
            if (page.ToLower().StartsWith(MonitorPage) && MainForm.MonitoringFile != null) return HttpMonPage(sess, page, postfields);
            if (page.ToLower().StartsWith(AutoStartPage)) return HttpAutoStartPage(sess, page, postfields);
            System.Text.RegularExpressions.Match rxm = rx_drivers.Match(page);
            if (rxm.Success)
                try
                {
                    long drvid = Convert.ToInt64(rxm.Groups[1].Value);
                    return ((SySal.Web.IWebApplication)MainForm.BM.GetTaskInfo(drvid).Domain.GetData(SySal.DAQSystem.Drivers.HostEnv.WebAccessString)).HttpPost(sess, page, postfields);
                }
                catch (Exception) {}

            return new SySal.Web.HTMLResponse(
                    "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                    "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                    "<head>\r\n" +
                    "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                    "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                    "    <title>OperaBatchManager/DPS Monitor - " + MainForm.MachineName + "</title>\r\n" +
                    "    <style type=\"text/css\">\r\n" +
                    "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                    "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                    "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                    "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                    "    </style>\r\n" +
                    "</head>\r\n" +
                    "<body>\r\n" +
                    "<div><b>OperaBatchManager Monitor (" + MainForm.MachineName + ")</b></div>\r\n" +
                    "<p><a href=\"" + BMPage + "\">BatchManager page</a></p>\r\n" +
                    "<p><a href=\"" + DPSPage + "\">DataProcessingServer page</a></p>\r\n" +
                    "<p><a href=\"" + MonitorPage + "\">Monitoring page</a></p>\r\n" +
                    "<p><a href=\"" + AutoStartPage + "\">AutoStart page</a></p>\r\n" +
                    "</body>\r\n"
                    );                                                                  
        }

        const string MonitorPage = "/mon";

        private SySal.Web.ChunkedResponse HttpMonPage(SySal.Web.Session sess, string page, params string[] postfields)
        {
            System.DateTime pagestart = System.DateTime.Now;
            System.DateTime start = new DateTime();            
            SySal.OperaDb.OperaDbConnection conn = null;
            SySal.OperaDb.OperaDbDataReader dr = null;
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>OperaBatchManager/Monitor</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "     th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold; border-right-width: 1px; border-bottom-width: 1px; border-top-width: 0px; border-left-width: 0px; border-color: black; border-style: solid }\r\n" +
                            "     td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal; border-right-width: 1px; border-bottom-width: 1px; border-top-width: 0px; border-left-width: 0px; border-color: black; border-style: solid }\r\n" +
                            "     .td1 { border-left-width: thin; border-right-width: thin; border-top-width: thin }\r\n" +
                            "     .noborder { border: none }\r\n" +
                            "     table { border-left-width: 1px; border-top-width: 1px; border-right-width: 0px; border-bottom-width: 0px; border-color: black; border-style: solid }\r\n" +
                            "     p { font-family: Trebuchet MS,Arial,Helvetica; font-size: 14; color: navy; background-color: white; text-align: left; font-weight: bold }\r\n" +
                            "     div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            " <div><b>OperaBatchManager/Monitor<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div><br>\r\n" +
                            " <div>\r\n";

            try
            {
                start = System.DateTime.Now;
                string monfile = System.IO.File.ReadAllText(MainForm.MonitoringFile);
                string[] qvws = monfile.Split(' ', '\t', '\r', '\n');
                System.TimeSpan filespan = System.DateTime.Now - start;
                start = System.DateTime.Now;
                conn = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
                conn.Open();
                System.TimeSpan connspan = System.DateTime.Now - start;
                html += " </div><div>file time: " + SySal.Web.WebServer.HtmlFormat(filespan.TotalMilliseconds.ToString()) + "ms, dbconn time: " + SySal.Web.WebServer.HtmlFormat(connspan.TotalMilliseconds.ToString()) + "ms</div><div>\r\n";
                foreach (string q in qvws)
                {
                    start = System.DateTime.Now;
                    if (q.Trim().Length <= 0) continue;
                    try
                    {
                        dr = new SySal.OperaDb.OperaDbCommand("select * from " + q, conn).ExecuteReader();
                        html += " <table align=\"left\" cellpadding=\"0\" cellspacing=\"0\" class=\"noborder\">\r\n" +
                                "  <tr><td class=\"td1\"><p>" + SySal.Web.WebServer.HtmlFormat(q) + "</p></td><td class=\"noborder\" width=\"5px\">&nbsp;</td></tr>\r\n" +
                                "  <tr><td><table align=\"left\" cellspacing=\"0\">\r\n";
                        html += "  <tr>";
                        int j;
                        for (j = 0; j < dr.FieldCount; j++)
                            html += "<th>" + SySal.Web.WebServer.HtmlFormat(dr.GetName(j)) + "</th>";
                        html += "</tr>\r\n";
                        while (dr.Read())
                        {
                            html += "  <tr>";
                            for (j = 0; j < dr.FieldCount; j++)
                            {
                                object v = dr.GetValue(j);
                                if (v == System.DBNull.Value) v = null;
                                html += "<td>" + ((v == null) ? "&nbsp;" : SySal.Web.WebServer.HtmlFormat(v.ToString())) + "</td>";
                            }
                            html += "</tr>\r\n";
                        }
                    }
                    catch (Exception x)
                    {
                        html += "<font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(x.ToString()) + "</font>";
                    }
                    finally
                    {
                        html += "  </table>\r\n" + "  <tr><td>Time: " + SySal.Web.WebServer.HtmlFormat((System.DateTime.Now - start).TotalMilliseconds.ToString()) + "ms</td></tr>\r\n" +
                            "  </td></tr>\r\n" +
                            " </table>\r\n";
                        if (dr != null)
                        {
                            dr.Close();
                            dr = null;
                        }
                    }
                }
            }
            catch (Exception x)
            {
                html += "<font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(x.ToString()) + "</font>";
            }
            finally
            {
                if (dr != null) dr.Close();
                if (conn != null) conn.Close();
                html += " </div>\r\n<div>Total time: " + SySal.Web.WebServer.HtmlFormat((System.DateTime.Now - pagestart).TotalMilliseconds.ToString()) + "ms </div>\r\n</body>";
            }
            return new SySal.Web.HTMLResponse(html);
        }

        const string DPSPage = "/dps";

        private SySal.Web.ChunkedResponse HttpDPSPage(SySal.Web.Session sess, string page, params string[] postfields)
        {
            string user = "";
            string pwd = "";
            string dbuser = "";
            string dbpwd = "";
            string exepath = "";
            string cmdargs = "";
            string desc = "";
            string outsavefile = null;
            bool enq = false;
            bool rem = false;
            string xctext = "";
            ulong expid = 0;
            uint powerclass = 5;
            System.Collections.ArrayList chk = new System.Collections.ArrayList();
            try
            {
                if (postfields != null)
                {
                    foreach (string s in postfields)
                        if (s.StartsWith(ExpandCmd + "="))
                            try
                            {
                                expid = Convert.ToUInt64(s.Substring(ExpandCmd.Length + 1));
                            }
                            catch (Exception) { }
                    foreach (string s in postfields)
                    {
                        int eq = s.IndexOf("=");
                        if (eq >= 0)
                        {
                            string t = s.Substring(0, eq).ToLower();
                            string v = SySal.Web.WebServer.URLDecode(s.Substring(eq + 1));
                            switch (t)
                            {
                                case PowerClassCmd: try
                                    {
                                        powerclass = Convert.ToUInt32(v);
                                    }
                                    catch (Exception) { } break;
                                case CmdArgsCmd: cmdargs = v; break;
                                case DescCmd: desc = v; break;
                                case ExePathCmd: exepath = v; break;
                                case UserIdCmd: user = v; break;
                                case PasswordIdCmd: pwd = v; break;
                                case DBUserIdCmd: dbuser = v; break;
                                case DBPasswordIdCmd: dbpwd = v; break;
                                case OutSaveFileCmd: outsavefile = v; break;
                                case EnqBtn: enq = true; break;
                                case RemBtn: rem = true; break;                                    
                                default: if (s.StartsWith(CheckCmd))
                                        try
                                        {
                                            chk.Add(System.Convert.ToUInt64(t.Substring(CheckCmd.Length)));
                                        }
                                        catch (Exception) { }
                                    break;
                            }
                        }
                    }
                }
                if (enq)
                {
                    try
                    {
                        SySal.DAQSystem.DataProcessingBatchDesc bd = new SySal.DAQSystem.DataProcessingBatchDesc();
                        bd.Id = MainForm.DPS.SuggestId;
                        bd.Filename = exepath;
                        bd.CommandLineArguments = cmdargs;
                        bd.Username = user;
                        bd.Password = pwd;
                        bd.Token = null;
                        bd.AliasUsername = dbuser;
                        bd.AliasPassword = dbpwd;
                        bd.Description = desc;                        
                        bd.MachinePowerClass = powerclass;
                        bd.OutputTextSaveFile = outsavefile;
                        if (MainForm.DPS.Enqueue(bd) == false) throw new Exception("Batch refused.");
                    }
                    catch (Exception x)
                    {
                        xctext = x.ToString();
                    }
                }
                if (rem)
                {
                    foreach (ulong u in chk)
                    {
                        try
                        {
                            MainForm.DPS.Remove(u, null, user, pwd);
                        }
                        catch (Exception x) 
                        { 
                            xctext = x.ToString();
                        }
                    }
                }                
            }
            catch (Exception x)
            {
                xctext = x.ToString();
            }       
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>OperaBatchManager/DPS Monitor - " + MainForm.MachineName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "<div><b>OperaBatchManager/DPS Monitor (" + MainForm.MachineName + ")<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div>\r\n" +
                            "<div>Parallel jobs: " + MainForm.DPS.ParallelJobs + "</div>\r\n" +
                            "<br><a href=\"" + page + "\">Refresh</a><br>\r\n" +
                            "<form action=\"" + page + "\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n";
            if (xctext.Length > 0)
                html += "<p><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "</font></p>\r\n";
            if (MainForm.DPS != null)
            {
                SySal.DAQSystem.DataProcessingBatchDesc [] batches = MainForm.DPS.Queue;
                html += "<table border=\"1\" align=\"center\" width=\"100%\">\r\n" +
                        " <tr><th width=\"10%\">Batch</th><th width=\"5%\">PowerClass</th><th width=\"65%\">Description</th><th width=\"10%\">Owner</th><th width=\"10%\">Started</th></tr>\r\n";
                foreach (SySal.DAQSystem.DataProcessingBatchDesc b in batches)
                    html += " <tr><td><input id=\"" + CheckCmd + b.Id + "\" name=\"" + CheckCmd + b.Id + "\" type=\"checkbox\" />" + b.Id.ToString("X16") + "</td><td>" + b.MachinePowerClass + "</td><td>" + SySal.Web.WebServer.HtmlFormat(b.Description) +
                        ((expid == b.Id) ? ("<br><div align=\"left\"><font face=\"Courier\"><c>" + SySal.Web.WebServer.HtmlFormat(b.Filename + " " + b.CommandLineArguments) + "</c></font></div>&nbsp;<a href=\"" + page + "?" + ExpandCmd + "=0\"><i>Shrink</i></a>") : ("&nbsp;<a href=\"" + page + "?" + ExpandCmd + "=" + b.Id + "\"><i>Expand</i></a>")) +
                        "</td><td>&nbsp;" + SySal.Web.WebServer.HtmlFormat((b.Username == null || b.Username == "") ? "N/A" : b.Username) + "</td><td>&nbsp;" + b.Started.ToString() + "</td></tr>\r\n";

                html += "</table>\r\n" +
                        "<p><input id=\"" + EnqBtn + "\" name=\"" + EnqBtn + "\" type=\"submit\" value=\"Enqueue\"/>&nbsp;<input id=\"" + RemBtn + "\" name=\"" + RemBtn + "\" type=\"submit\" value=\"Remove Selected\"/></p>\r\n" +
                        "<p>Description <input id=\"" + DescCmd + "\" maxlength=\"1024\" name=\"" + DescCmd + "\" size=\"50\" type=\"text\" /></p>\r\n" +
                        "<p>Executable <input id=\"" + ExePathCmd + "\" maxlength=\"1024\" name=\"" + ExePathCmd + "\" size=\"50\" type=\"text\" value=\"" + SySal.Web.WebServer.HtmlFormat(MainForm.ExeRepository) + "\" /></p>\r\n" +
                        "<p>Command line arguments <input id=\"" + CmdArgsCmd + "\" maxlength=\"10240\" name=\"" + CmdArgsCmd + "\" size=\"50\" type=\"text\" /></p>\r\n" +
                        "<p>Machine power class <input id=\"" + PowerClassCmd + "\" maxlength=\"5\" name=\"" + PowerClassCmd + "\" size=\"5\" type=\"text\" /></p>\r\n" +
                        "<p>Output save file <input id=\"" + OutSaveFileCmd + "\" maxlength=\"1024\" name=\"" + OutSaveFileCmd + "\" size=\"50\" type=\"text\" /></p>\r\n" +
                        "<table align=\"left\" border=\"0\">\r\n" +
                        " <tr><td align=\"left\" width=\"50%\"><p>Username</p></td><td align=\"right\" width=\"50%\"><input id=\"" + UserIdCmd + "\" maxlength=\"30\" name=\"" + UserIdCmd + "\" size=\"30\" type=\"text\" /></td></tr>\r\n" +
                        " <tr><td align=\"left\" width=\"50%\"><p>Password</p></td><td align=\"right\" width=\"50%\"><input id=\"" + PasswordIdCmd + "\" name=\"" + PasswordIdCmd + "\" size=\"30\" type=\"password\" /></td></tr>\r\n" +
                        " <tr><td align=\"left\" width=\"50%\"><p>DB User</p></td><td align=\"right\" width=\"50%\"><input id=\"" + DBUserIdCmd + "\" maxlength=\"30\" name=\"" + DBUserIdCmd + "\" size=\"30\" type=\"text\" /></td></tr>\r\n" +
                        " <tr><td align=\"left\" width=\"50%\"><p>DB Password</p></td><td align=\"right\" width=\"50%\"><input id=\"" + DBPasswordIdCmd + "\" name=\"" + DBPasswordIdCmd + "\" size=\"30\" type=\"password\" /></td></tr>\r\n" +
                        "</table>\r\n" +                        
                        "</form>\r\n";
            }
            html += "</body>\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        const string OutSaveFileCmd = "osf";
        const string PowerClassCmd = "pwc";
        const string CheckCmd = "chk";
        const string ExpandCmd = "exp";
        const string EnqBtn = "enq";
        const string RemBtn = "rem";
        const string DescCmd = "dsc";
        const string ExePathCmd = "exe";
        const string CmdArgsCmd = "cmd";
        const string UserIdCmd = "uid";
        const string PasswordIdCmd = "pwd";
        const string DBUserIdCmd = "dbu";
        const string DBPasswordIdCmd = "dbp";

        const string BMPage = "/bm";        

        const string PauseCmd = "pause";
        const string ResumeCmd = "resume";
        const string AbortCmd = "abort";
        const string SetProgressCmd = "setprogress";
        const string ProgressText = "progresstext";
        const string AutoStartToggleCmd = "autostart";

        static System.Xml.Serialization.XmlSerializer progxmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.TaskProgressInfo));

        private SySal.Web.ChunkedResponse HttpBMPage(SySal.Web.Session sess, string page, params string[] postfields)
        {
            bool processpagemissing = false;
            string ppxtext = "";
            if (page.ToLower().StartsWith(BMPage + "/"))
            {
                try
                {
                    long reqid = Convert.ToInt64(page.Substring(BMPage.Length + 1));
                    return ((SySal.Web.IWebApplication)MainForm.BM.GetTaskInfo(reqid).Domain.GetData(SySal.DAQSystem.Drivers.HostEnv.WebAccessString)).HttpPost(sess, page, postfields);
                }
                catch (Exception ppx)
                {
                    ppxtext = ppx.ToString();
                    processpagemissing = true;
                }
            }
            long expid = 0;
            string xctext = null;
            string usr = "";
            string pwd = "";
            string progresstext = "";
            if (postfields != null)
            {
                foreach (string p in postfields)
                    if (p.StartsWith(UserIdCmd + "="))
                        usr = p.Substring(UserIdCmd.Length + 1);
                    else if (p.StartsWith(PasswordIdCmd + "="))
                        pwd = p.Substring(PasswordIdCmd.Length + 1);
                    else if (p.StartsWith(ProgressText + "="))
                        progresstext = p.Substring(ProgressText.Length + 1);
            }

            if (usr.Length == 0 || pwd.Length == 0)
            {
                if (sess.UserData != null)
                {
                    usr = ((SySal.DAQSystem.Drivers.HostEnv.WebUserData)sess.UserData).Usr;
                    pwd = ((SySal.DAQSystem.Drivers.HostEnv.WebUserData)sess.UserData).Pwd;
                }
            }
            else
            {
                SySal.DAQSystem.Drivers.HostEnv.WebUserData u = new SySal.DAQSystem.Drivers.HostEnv.WebUserData();
                u.Usr = usr;
                u.Pwd = pwd;
                sess.UserData = u;
            }

            bool setprogressenable = false;

            string postfieldsdump = "";

            if (postfields != null)
            {
                foreach (string p in postfields)
                    try
                    {
                        postfieldsdump += "<br>" + SySal.Web.WebServer.HtmlFormat(p);
                        if (p.StartsWith(PauseCmd + "="))
                            MainForm.BM.Pause(Convert.ToInt64(p.Substring(PauseCmd.Length + 1)), usr, pwd);
                        else if (p.StartsWith(ResumeCmd + "="))
                            MainForm.BM.Resume(Convert.ToInt64(p.Substring(ResumeCmd.Length + 1)), usr, pwd);
                        else if (p.StartsWith(AbortCmd + "="))
                            MainForm.BM.Abort(Convert.ToInt64(p.Substring(AbortCmd.Length + 1)), usr, pwd);
                        else if (p.StartsWith(ExpandCmd + "="))
                            expid = Convert.ToInt64(p.Substring(ExpandCmd.Length + 1));
                        else if (p.StartsWith(SetProgressCmd + "="))
                            setprogressenable = true;
                        else if (p.StartsWith(AutoStartToggleCmd + "="))
                            MainForm.TheMainForm.AutoStartEnabled = Convert.ToBoolean(p.Substring(AutoStartToggleCmd.Length + 1));
                    }
                    catch (Exception x)
                    {
                        xctext = x.ToString();
                    }
                if (setprogressenable)
                {
                    System.IO.StringReader rd = new System.IO.StringReader(progresstext);
                    SySal.DAQSystem.Drivers.HostEnv[] tasks = MainForm.BM.Tasks;
                    foreach (SySal.DAQSystem.Drivers.HostEnv h in tasks)
                        if (h.StartupInfo.ProcessOperationId == expid)
                            h.ProgressInfo = (SySal.DAQSystem.Drivers.TaskProgressInfo)progxmls.Deserialize(rd);

                }
            }

            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>OperaBatchManager/Process Monitor - " + MainForm.MachineName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" + //"<br>usr=" + usr + "<br>pwd=" + pwd + "<br>" + //"<br />" + postfieldsdump + "<br />" +
                            //(processpagemissing ? "<div><font color=red>Process page " + SySal.Web.WebServer.HtmlFormat(page) + " missing, switching to main.</font></div>" : "") + 
                            (processpagemissing ? "<div><font color=red>" + SySal.Web.WebServer.HtmlFormat(ppxtext) + "</font></div>" : "") + 
                            "<div><b>OperaBatchManager/Process Monitor (" + MainForm.MachineName + ")<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div>\r\n" +
                            "<div align=\"left\">" + ((usr.Length == 0) ? "Not logged on" : ("Logged on as <b>" + usr.ToLower() + "</b>")) + "</div>\r\n" +
                            "<br><a href=\"" + BMPage + "\">Refresh</a><br>\r\n" +
                            (xctext != null ? ("<font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "</font><br>\r\n") : "") +
                            "<div align=\"left\">AutoStart is " + (MainForm.TheMainForm.AutoStartEnabled ? "<font color=\"green\">ON</font>" : "<font color=\"red\">OFF</font>") + " <a href=?" + AutoStartToggleCmd + "=" + (MainForm.TheMainForm.AutoStartEnabled ? "false" : "true") + ">Toggle</a></div>\r\n" + 
                            "<form action=\"" + BMPage + "\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n" +
                            "<table border=\"1\" align=\"center\" width=\"100%\">\r\n" +
                            " <tr><th>Id</th><th>Brick</th><th>Plate</th><th>Executable</th><th>Machine</th><th>Type</th><th>Started</th><th>Progress</th></tr>\r\n";
            System.Collections.ArrayList arr = new System.Collections.ArrayList();
            arr.AddRange(MainForm.BM.Operations);
            arr.Sort();
            long[] ids = (long[])arr.ToArray(typeof(long));
            SySal.DAQSystem.Drivers.TaskStartupInfo[] ops = new SySal.DAQSystem.Drivers.TaskStartupInfo[ids.Length];
            SySal.DAQSystem.Drivers.TaskProgressInfo[] prgi = new SySal.DAQSystem.Drivers.TaskProgressInfo[ids.Length];
            int i;
            for (i = 0; i < ids.Length; i++)
                try
                {
                    SySal.DAQSystem.Drivers.BatchSummary bms = MainForm.BM.GetSummary(ids[i]);
                    long brick = 0;
                    ops[i] = MainForm.BM.GetOperationStartupInfo(ids[i]);
                    if (ops[i] is SySal.DAQSystem.Drivers.BrickOperationInfo) brick = ((SySal.DAQSystem.Drivers.BrickOperationInfo)ops[i]).BrickId;
                    else if (ops[i] is SySal.DAQSystem.Drivers.VolumeOperationInfo) brick = ((SySal.DAQSystem.Drivers.VolumeOperationInfo)ops[i]).BrickId;
                    else if (ops[i] is SySal.DAQSystem.Drivers.ScanningStartupInfo) brick = ((SySal.DAQSystem.Drivers.ScanningStartupInfo)ops[i]).Plate.BrickId;
                    prgi[i] = MainForm.BM.GetProgressInfo(ids[i]);
                    string pinfo = "";
                    if (bms.Id == expid)
                    {
                        System.IO.StringWriter wr = new System.IO.StringWriter();
                        progxmls.Serialize(wr, MainForm.BM.GetProgressInfo(bms.Id));
                        pinfo = SySal.Web.WebServer.HtmlFormat(wr.ToString());
                    }
                    html += " <tr><td><a href=\"" + BMPage + "/" + bms.Id + "\">" + bms.Id + "</a>&nbsp;<a href=\"" + BMPage + "?" + (bms.OpStatus == SySal.DAQSystem.Drivers.Status.Paused ? ResumeCmd : PauseCmd) + "=" + bms.Id + "\">" + 
                        (bms.OpStatus == SySal.DAQSystem.Drivers.Status.Paused ? "resume" : "pause") + "</a>&nbsp;<a href=\"" + BMPage + "?" + AbortCmd + "=" + bms.Id + "\">abort</a></td><td>" + 
                        (bms.BrickId <= 0 ? "&nbsp;" : (bms.BrickId.ToString())) + "</td><td>" + ((bms.PlateId <= 0) ? "&nbsp;" : bms.PlateId.ToString()) + "</td><td>" + SySal.Web.WebServer.HtmlFormat(bms.Executable) + "</td><td>" + SySal.Web.WebServer.HtmlFormat(GetMachineName(bms.MachineId)) + "</td><td>" + 
                        bms.DriverLevel + "</td><td>" + SySal.Web.WebServer.HtmlFormat(bms.StartTime.ToString("HH:mm:ss dd/MM/yy")) + "</td><td>" + 
                        SySal.Web.WebServer.HtmlFormat((bms.Progress * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%") + 
                        ((expid == bms.Id) ? (" <a href=\"" + BMPage + "?" + ExpandCmd + "=0\">shrink</a><br><textarea rows=\"20\" cols=\"40\">" + // name=\"" + ProgressText + "\" id=\"" + ProgressText + "\">" + 
                        pinfo + "</textarea><input type=\"hidden\" id=\"" + ExpandCmd + "\" name=\"" + ExpandCmd + "\" value=\"" + bms.Id + "\"/>" 
                        /*+ "<input type=\"submit\" id=\"" + SetProgressCmd + "\" name=\"" + SetProgressCmd + "\" value=\"Set\" />"*/) : (" <a href=\"" + BMPage + "?" + ExpandCmd + "=" + bms.Id + 
                        "\">expand</a>")) + "</td></tr>\r\n";
                }
                catch (Exception) { }

            html +=
                "<table align=\"left\" border=\"0\">\r\n" +
                " <tr><td align=\"left\" width=\"50%\"><p>Username</p></td><td align=\"right\" width=\"50%\"><input id=\"" + UserIdCmd + "\" maxlength=\"30\" name=\"" + UserIdCmd + "\" size=\"30\" type=\"text\" /></td></tr>\r\n" +
                " <tr><td align=\"left\" width=\"50%\"><p>Password</p></td><td align=\"right\" width=\"50%\"><input id=\"" + PasswordIdCmd + "\" name=\"" + PasswordIdCmd + "\" size=\"30\" type=\"password\" /></td></tr>\r\n" +
                " <tr><td align=\"center\" colspan=\"2\"><input type=\"submit\" value=\"Log on\" /></td></tr>\r\n" +
                " <tr><td align=\"center\" colspan=\"2\"><a href=\"" + ProcStartPage + "\">Start process</a></td></tr>\r\n" +
                "</table>\r\n" +
                "</form>\r\n" +
                "</body>\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        const string ProcStartPage = "/ps";        
        const string ProcStartShowAllProgSetCmd = "sa";
        const string ProcStartProgSetCmd = "pg";
        const string DriverLevelCmd = "dl";
        const string ProcStartBrickCmd = "bk";
        const string ProcStartPlateCmd = "pl";
        const string ProcStartMachineCmd = "ma";
        const string ProcStartNotesCmd = "notes";

        private SySal.Web.ChunkedResponse HttpProcStartPage(SySal.Web.Session sess, string page, params string[] postfields)
        {
            SySal.OperaDb.OperaDbConnection DBConn = null;

            try
            {
                long ConfigId = 0;
                int BrickId = 0;
                int PlateId = 0;
                int DriverLevel = -1;
                long MachineId = 0;
                string Notes = null;
                bool ShowAllConfigs = false;

                string xctext = null;
                string usr = "";
                string pwd = "";
                string debuglog = "";
                if (postfields != null)
                {
                    debuglog += "A";
                    foreach (string p in postfields)
                        if (p.StartsWith(UserIdCmd + "="))
                            usr = p.Substring(UserIdCmd.Length + 1);
                        else if (p.StartsWith(PasswordIdCmd.Length + "="))
                            pwd = p.Substring(PasswordIdCmd.Length + 1);
                    debuglog += "B";
                    if (usr.Length == 0 || pwd.Length == 0)
                    {
                        debuglog += "C";
                        if (sess.UserData != null)
                        {
                            debuglog += "D";
                            usr = ((SySal.DAQSystem.Drivers.HostEnv.WebUserData)sess.UserData).Usr;
                            pwd = ((SySal.DAQSystem.Drivers.HostEnv.WebUserData)sess.UserData).Pwd;
                        }
                    }
                    else
                    {
                        debuglog += "E";
                        SySal.DAQSystem.Drivers.HostEnv.WebUserData u = new SySal.DAQSystem.Drivers.HostEnv.WebUserData();
                        u.Usr = usr;
                        u.Pwd = pwd;
                        sess.UserData = u;
                    }
                    debuglog += "F";

                    foreach (string p in postfields)
                        try
                        {
                            if (p.StartsWith(ProcStartShowAllProgSetCmd + "="))
                                ShowAllConfigs = (p.Substring(ProcStartShowAllProgSetCmd.Length + 1).ToLower() == "y");
                            else if (p.StartsWith(ProcStartProgSetCmd + "="))
                                ConfigId = Convert.ToInt64(p.Substring(ProcStartProgSetCmd.Length + 1));
                            else if (p.StartsWith(DriverLevelCmd + "="))
                                DriverLevel = Convert.ToInt32(p.Substring(DriverLevelCmd.Length + 1));
                            else if (p.StartsWith(ProcStartBrickCmd + "="))
                                BrickId = Convert.ToInt32(p.Substring(ProcStartBrickCmd.Length + 1));
                            else if (p.StartsWith(ProcStartPlateCmd + "="))
                                PlateId = Convert.ToInt32(p.Substring(ProcStartPlateCmd.Length + 1));
                            else if (p.StartsWith(ProcStartMachineCmd + "="))
                                MachineId = Convert.ToInt64(p.Substring(ProcStartMachineCmd.Length + 1));
                            else if (p.StartsWith(ProcStartNotesCmd + "="))
                                Notes = SySal.Web.WebServer.URLDecode(p.Substring(ProcStartNotesCmd.Length + 1));
                        }
                        catch (Exception x)
                        {
                            xctext = x.ToString();
                        }
                }
                string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                                "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                                "<head>\r\n" +
                                "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                                "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                                "    <title>OperaBatchManager/Process Start - " + MainForm.MachineName + "</title>\r\n" +
                                "    <style type=\"text/css\">\r\n" +
                                "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                                "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                                "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                                "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                                "    </style>\r\n" +
                                "</head>\r\n" +
                                "<body>\r\n" + //"<br>usr=" + usr + "<br>pwd=" + pwd + "<br>debug " + debuglog + "<br>cookie: " + sess.Cookie + "<br>sess: " + sess.GetHashCode() + "<br>" + 
                                "<div><b>OperaBatchManager/Process Start (" + MainForm.MachineName + ")</b></div>\r\n" +                                
                                "<form action=\"" + ProcStartPage + " method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n";

                DBConn = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
                DBConn.Open();

                if (ConfigId == 0)
                {                    

                    string favprogset = "inner join (SELECT VALUE, NAME FROM OPERA.LZ_SITEVARS WHERE upper(NAME) like 'PROGSET %') on (ID = VALUE and DRIVERLEVEL > 0)";
                    string progsetwhere = ShowAllConfigs ? "WHERE DRIVERLEVEL > 0" : favprogset;

                    System.Data.DataSet ds = new System.Data.DataSet();
                    new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, EXECUTABLE, " +
                        (ShowAllConfigs ? "DESCRIPTION" : "trim(substr(name, instr(name, 'PROGSET') + 7)) as DESCRIPTION") +
                        ", DRIVERLEVEL FROM TB_PROGRAMSETTINGS " + progsetwhere + " ORDER BY ID ASC", DBConn, null).Fill(ds);

                    html +=
                        " <div>Step 1: <b>Select process type</b></div>\r\n" +
                        " <table width=\"100%\" border=\"1\" align=\"center\">\r\n" +
                        "  <tr><th>ID</th><th>DriverLevel</th><th>Executable</th><th>Description</th></tr>\r\n";
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                        html += "  <tr><td><input type=\"RADIO\" id=\"" + ProcStartProgSetCmd + "\" name=\"" + ProcStartProgSetCmd + "\" value=\"" + dr[0] + "\" />" + dr[0] + "</td><td>" + dr[3] + "</td><td>" + dr[1] + "</td><td>" + dr[2] + "</td></tr>\r\n";
                    html +=
                        " </table><br>\r\n" +
                        (ShowAllConfigs
                            ? (" <a href=\"" + ProcStartPage + "?" + ProcStartShowAllProgSetCmd + "=n\">Show only favorite configurations</a><br>\r\n")
                            : (" <a href=\"" + ProcStartPage + "?" + ProcStartShowAllProgSetCmd + "=y\">Show all configurations</a><br>\r\n")
                        );
                }
                else
                {
                    if (DriverLevel < 0)
                        DriverLevel = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT DRIVERLEVEL FROM TB_PROGRAMSETTINGS WHERE ID = " + ConfigId, DBConn).ExecuteScalar());
                    html += 
                        " <input type=\"HIDDEN\" name=\"" + ProcStartProgSetCmd + "\" id=\"" + ProcStartProgSetCmd + "\" value=\"" + ConfigId + "\" />\r\n" + 
                        " <input type=\"HIDDEN\" name=\"" + DriverLevelCmd + "\" id=\"" + DriverLevelCmd + "\" value=\"" + DriverLevel + "\" />\r\n";

                    if (DriverLevel >= (int)SySal.DAQSystem.Drivers.DriverType.Scanning && DriverLevel <= (int)SySal.DAQSystem.Drivers.DriverType.Brick && BrickId <= 0)
                    {
                        html += 
                            " <div>Step 2.1: <b>Select brick</b></div>\r\n" +
                            " <select size=\"20\" multiple=\"false\" name=\"" + ProcStartBrickCmd + "\" id=\"" + ProcStartBrickCmd + "\">\r\n";

                        System.Data.DataSet ds = new System.Data.DataSet();
                        new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM TB_EVENTBRICKS ORDER BY ID ASC", DBConn, null).Fill(ds);
                        foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                            html += " <option value=\"" + dr[0] + "\">" + dr[0] + "</option>\r\n";
                        
                        html += " </select>\r\n";
                    }
                    else 
                    {
                        html += " <input type=\"HIDDEN\" name=\"" + ProcStartBrickCmd + "\" id=\"" + ProcStartBrickCmd + "\" value=\"" + BrickId + "\" />\r\n";
                        if (DriverLevel == (int)SySal.DAQSystem.Drivers.DriverType.Scanning && PlateId <= 0)
                        {
                            html +=
                                " <div>Step 2.2: <b>Select plate</b></div>\r\n" +
                                " <select size=\"20\" multiple=\"false\" name=\"" + ProcStartPlateCmd + "\" id=\"" + ProcStartPlateCmd + "\">\r\n";

                            System.Data.DataSet ds = new System.Data.DataSet();
                            new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM VW_PLATES WHERE ID_EVENTBRICK = " + BrickId + " ORDER BY ID ASC", DBConn, null).Fill(ds);
                            foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                                html += " <option value=\"" + dr[0] + "\">" + dr[0] + "</option>\r\n";

                            html += " </select>\r\n";
                        }
                        else 
                        {
                            html += " <input type=\"HIDDEN\" name=\"" + ProcStartPlateCmd + "\" id=\"" + ProcStartPlateCmd + "\" value=\"" + PlateId + "\" />\r\n";
                            if (MachineId <= 0)
                            {
                                html +=
                                    " <div>Step 3: <b>Select machine</b></div>\r\n" +
                                    " <table width=\"100%\" border=\"1\" align=\"center\">\r\n" +
                                    "  <tr><th>ID</th><th>name</th><th>Address</th></tr>\r\n";
                    			System.Data.DataSet ds = new System.Data.DataSet();
                                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, NAME, ADDRESS FROM TB_MACHINES WHERE ID_SITE IN (SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE') ORDER BY ID ASC", DBConn, null).Fill(ds);

                                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                                    html += "  <tr><td><input type=\"RADIO\" id=\"" + ProcStartMachineCmd + "\" name=\"" + ProcStartMachineCmd + "\" value=\"" + dr[0] + "\" />" + dr[0] + "</td><td>" + dr[1] + "</td><td>" + dr[2] + "</td></tr>\r\n";

                                html += " </table><br>\r\n";
                            }
                            else
                            {
                                html += " <input type=\"HIDDEN\" name=\"" + ProcStartMachineCmd + "\" id=\"" + ProcStartMachineCmd + "\" value=\"" + MachineId + "\" />\r\n";
                                if (Notes == null)
                                {
                                    html +=
                                        " <div>Step 4: <b>Notes</b> <i>(optional)</i></div>\r\n" +
                                        " <input type=\"text\" name=\"" + ProcStartNotesCmd + "\" id=\"" + ProcStartNotesCmd + "\" value=\"\" size=\"50\" />\r\n";
                                }
                                else
                                {
                                    SySal.DAQSystem.Drivers.TaskStartupInfo tsinfo = null;
                                    switch ((SySal.DAQSystem.Drivers.DriverType)DriverLevel)
                                    {
                                        case SySal.DAQSystem.Drivers.DriverType.Scanning:
                                            {
                                                SySal.DAQSystem.Drivers.ScanningStartupInfo xinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
                                                xinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
                                                xinfo.Plate.BrickId = BrickId;
                                                xinfo.Plate.PlateId = PlateId;
                                                tsinfo = xinfo;
                                                break;
                                            }

                                        case SySal.DAQSystem.Drivers.DriverType.Volume:
                                            {
                                                SySal.DAQSystem.Drivers.VolumeOperationInfo xinfo = new SySal.DAQSystem.Drivers.VolumeOperationInfo();
                                                xinfo.BrickId = BrickId;
                                                tsinfo = xinfo;
                                                break;
                                            }

                                        case SySal.DAQSystem.Drivers.DriverType.Brick:
                                            {
                                                SySal.DAQSystem.Drivers.BrickOperationInfo xinfo = new SySal.DAQSystem.Drivers.BrickOperationInfo();
                                                xinfo.BrickId = BrickId;
                                                tsinfo = xinfo;
                                                break;
                                            }

                                        case SySal.DAQSystem.Drivers.DriverType.System:
                                            {
                                                tsinfo = new SySal.DAQSystem.Drivers.TaskStartupInfo();
                                                break;
                                            }
                                    }
                                    tsinfo.MachineId = MachineId;
                                    tsinfo.ProgramSettingsId = ConfigId;
                                    tsinfo.OPERAUsername = usr;
                                    tsinfo.OPERAPassword = pwd;
                                    tsinfo.Notes = Notes;
                                    try
                                    {
                                        long opid = MainForm.BM.Start(0, tsinfo);
                                        html += "<div><font color=\"green\">Operation " + opid + " started.</font><br><a href=\"" + BMPage + "\">Main page</a></div>\r\n";
                                    }
                                    catch (Exception x)
                                    {
                                        html += "<div><font color=\"green\">Error starting operation!</font><br>" + SySal.Web.WebServer.HtmlFormat(x.ToString()) + "<br><a href=\"" + BMPage + "\">Main page</a></div>\r\n";
                                    }
                                }

                            }
                        }
                    }
                    

                }

                html +=
                    " <input type=\"submit\" value=\"Next\"/> <a href=\"" + ProcStartPage + "\">Reset</a> <a href=\"" + BMPage + "\">Main page</a>\r\n" + 
                    "</form>\r\n" +
                    "</body>\r\n";
                return new SySal.Web.HTMLResponse(html);
            }
            catch (Exception x) 
            { 
                return new SySal.Web.HTMLResponse("<html><head><title>Error</title></head><body>" + SySal.Web.WebServer.HtmlFormat(x.ToString()) + "<br><a href=\"" + BMPage + "\">Main page</a></body></html>");
            }
            finally
            {
                if (DBConn != null) DBConn.Close();
                DBConn = null;
            }
        }

        const string AutoStartPage = "/as";

        private SySal.Web.ChunkedResponse HttpAutoStartPage(SySal.Web.Session sess, string page, params string[] postfields)
        {
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>OperaBatchManager/AutoStart - " + MainForm.MachineName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "<div><b>OperaBatchManager/AutoStart queue (" + MainForm.MachineName + ")</b></div>\r\n";
            string[] queue = new string[0];
            try
            {
                html +=     " <table width=\"100%\" border=\"1\">\r\n" +
                            "  <tr><th>ProgramSettings</th><th>Machine</th><th>Brick</th><th>Plate</th><th>Notes</th><th>Interrupt</th></tr>\r\n";
                queue = System.IO.File.ReadAllText(MainForm.AutoStartFile).Split('\r', '\n');
                foreach (string q in queue)
                {
                    string[] qt = q.Split('$');
                    if (qt.Length == 6)
                        html += "  <tr><td>" + SySal.Web.WebServer.HtmlFormat(qt[0].Trim()) + "</td><td>" + SySal.Web.WebServer.HtmlFormat(qt[1].Trim()) +
                            "</td><td>" + SySal.Web.WebServer.HtmlFormat(qt[2].Trim()) + "</td><td>" + SySal.Web.WebServer.HtmlFormat(qt[3].Trim()) +
                            "</td><td>" + SySal.Web.WebServer.HtmlFormat(qt[4].Trim()) + "</td><td>" + SySal.Web.WebServer.HtmlFormat(qt[5].Trim()) + "</td></tr>\r\n";
                    else if (q.Trim().Length > 0)
                        html += "  <tr><td colspan=\"6\"><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(q.Trim()) + "</font></td></tr>\r\n";
                }
                html += " </table>\r\n";
            }
            catch (Exception) 
            {
                html += "  <font color=\"red\">Error reading &quote;" + SySal.Web.WebServer.HtmlFormat(MainForm.AutoStartFile) + "&quote;.<br>Please retry.</font>\r\n";
            }
            html += "</body>\r\n</html>\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        #endregion
    }
}
