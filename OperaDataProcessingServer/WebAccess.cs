using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Services.OperaDataProcessingServer
{
    internal class WebAccess : SySal.Web.IWebApplication
    {
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
            get { return "OperaDataProcessingServer"; }
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

        /// <summary>
        /// Handles POST methods.
        /// </summary>
        /// <param name="sess">Session information.</param>
        /// <param name="page">ignored.</param>
        /// <param name="postfields">the action parameters passed.</param>
        /// <returns>the status page.</returns>
        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
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
                        bd.Id = OperaDataProcessingServer.DPS.SuggestId;
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
                        if (OperaDataProcessingServer.DPS.Enqueue(bd) == false) throw new Exception("Batch refused.");
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
                            OperaDataProcessingServer.DPS.Remove(u, null, user, pwd);
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
                            "    <title>OperaDataProcessingServer Monitor - " + OperaDataProcessingServer.MachineName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "<div><b>OperaDataProcessingServer Monitor (" + OperaDataProcessingServer.MachineName + ")<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div>\r\n" +
                            "<br><a href=\"/\">Refresh</a><br>\r\n" +
                            "<form action=\"/\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n";
            if (xctext.Length > 0)
                html += "<p><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "</font></p>\r\n";
            if (OperaDataProcessingServer.DPS != null)
            {
                SySal.DAQSystem.DataProcessingBatchDesc [] batches = OperaDataProcessingServer.DPS.Queue;
                html += "<table border=\"1\" align=\"center\" width=\"100%\">\r\n" +
                        " <tr><th width=\"10%\">Batch</th><th width=\"5%\">PowerClass</th><th width=\"65%\">Description</th><th width=\"10%\">Owner</th><th width=\"10%\">Started</th></tr>\r\n";
                foreach (SySal.DAQSystem.DataProcessingBatchDesc b in batches)
                    html += " <tr><td><input id=\"" + CheckCmd + b.Id + "\" name=\"" + CheckCmd + b.Id + "\" type=\"checkbox\" />" + b.Id.ToString("X16") + "</td><td>" + b.MachinePowerClass + "</td><td>" + SySal.Web.WebServer.HtmlFormat(b.Description) +
                        ((expid == b.Id) ? ("<br><div align=\"left\"><font face=\"Courier\"><c>" + SySal.Web.WebServer.HtmlFormat(b.Filename + " " + b.CommandLineArguments) + "</c></font></div>&nbsp;<a href=\"/?" + ExpandCmd + "=0\"><i>Shrink</i></a>") : ("&nbsp;<a href=\"/?" + ExpandCmd + "=" + b.Id + "\"><i>Expand</i></a>")) +
                        "</td><td>&nbsp;" + SySal.Web.WebServer.HtmlFormat((b.Username == null || b.Username == "") ? "N/A" : b.Username) + "</td><td>&nbsp;" + b.Started.ToString() + "</td></tr>\r\n";

                html += "</table>\r\n" +
                        "<p><input id=\"" + EnqBtn + "\" name=\"" + EnqBtn + "\" type=\"submit\" value=\"Enqueue\"/>&nbsp;<input id=\"" + RemBtn + "\" name=\"" + RemBtn + "\" type=\"submit\" value=\"Remove Selected\"/></p>\r\n" +
                        "<p>Description <input id=\"" + DescCmd + "\" maxlength=\"1024\" name=\"" + DescCmd + "\" size=\"50\" type=\"text\" /></p>\r\n" +
                        "<p>Executable <input id=\"" + ExePathCmd + "\" maxlength=\"1024\" name=\"" + ExePathCmd + "\" size=\"50\" type=\"text\" value=\"" + SySal.Web.WebServer.HtmlFormat(OperaDataProcessingServer.ExeRepository) + "\" /></p>\r\n" +
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

        #endregion
    }
}
