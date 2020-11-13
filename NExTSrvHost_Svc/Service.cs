using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.ServiceProcess;
using System.Text;
using System.Net;

namespace SySal.Services.NExTSrvHost_Svc
{
    public partial class NExTSrvHost : ServiceBase, SySal.Web.IWebApplication
    {
        static SySal.Web.WebServer WS;

        static internal Configuration TheConfig = new Configuration();

        public NExTSrvHost()
        {
            InitializeComponent();
        }

        protected override void OnStart(string[] args)
        {
            try
            {
                EventLog.CreateEventSource("NExTSrvHost", "Application");
            }
            catch (Exception) { }
            EventLog.Source = "NExTSrvHost";
            EventLog.Log = "Application";

            string configfile = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName + ".xml";
            System.Xml.Serialization.XmlSerializer xmlc = new System.Xml.Serialization.XmlSerializer(typeof(Configuration));
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(NExT.NExTConfiguration));
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                TheConfig = (Configuration)xmlc.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(configfile)));
                System.Collections.ArrayList dynsrvparams = new System.Collections.ArrayList();
                if (TheConfig.DBServer.Length > 0)
                {
                    NExT.NExTConfiguration.ServerParameter sp = new SySal.NExT.NExTConfiguration.ServerParameter();
                    sp.Name = "DBServer";
                    sp.Value = SySal.OperaDb.OperaDbCredentials.Decode(TheConfig.DBServer);
                    dynsrvparams.Add(sp);
                    sp = new SySal.NExT.NExTConfiguration.ServerParameter();
                    sp.Name = "DBUsr";
                    sp.Value = SySal.OperaDb.OperaDbCredentials.Decode(TheConfig.DBUserName);
                    sp = new SySal.NExT.NExTConfiguration.ServerParameter();
                    sp.Name = "DBPwd";
                    sp.Value = SySal.OperaDb.OperaDbCredentials.Decode(TheConfig.DBPassword);
                    conn = new SySal.OperaDb.OperaDbConnection(
                        SySal.OperaDb.OperaDbCredentials.Decode(TheConfig.DBServer),
                        SySal.OperaDb.OperaDbCredentials.Decode(TheConfig.DBUserName),
                        SySal.OperaDb.OperaDbCredentials.Decode(TheConfig.DBPassword)
                        );
                    conn.Open();

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
                    SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_SITES.ID, TB_SITES.NAME, TB_MACHINES.ID, TB_MACHINES.NAME, TB_MACHINES.ADDRESS FROM TB_SITES INNER JOIN TB_MACHINES ON (TB_MACHINES.ID_SITE = TB_SITES.ID AND TB_MACHINES.ISDATAPROCESSINGSERVER = 1 AND (" + selstr + "))", conn, null);
                    da.Fill(ds);
                    if (ds.Tables[0].Rows.Count < 1) throw new Exception("Can't find myself in OperaDb registered machines. This service is made unavailable.");
                    long IdSite = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][0]);
                    string SiteName = ds.Tables[0].Rows[0][1].ToString();
                    long IdMachine = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][2]);
                    string MachineName = ds.Tables[0].Rows[0][3].ToString();
                    string MachineAddress = ds.Tables[0].Rows[0][4].ToString();
                    object o = (TheConfig.DBConfigQuery.Length <= 0) ? null : new SySal.OperaDb.OperaDbCommand(TheConfig.DBConfigQuery.Replace("_ID_MACHINE_", IdMachine.ToString()), conn, null).ExecuteScalar();                     
                    if (o != null && o != System.DBNull.Value)
                        TheConfig.Services = (NExT.NExTConfiguration)xmls.Deserialize(new System.IO.StringReader(o.ToString()));
                    conn.Close();
                    conn = null;
                }
                SySal.NExT.NExTServer.SetupConfiguration(TheConfig.Services, (SySal.NExT.NExTConfiguration.ServerParameter[])dynsrvparams.ToArray(typeof(SySal.NExT.NExTConfiguration.ServerParameter)));
                WS = new SySal.Web.WebServer(TheConfig.WWWPort, this);
            }
            catch (Exception x)
            {
                EventLog.WriteEntry("Cannot read/apply server configuration file \"" + configfile + "\".\r\nAttempting to write a sample file.\r\n\r\nError:\r\n" + x.ToString(), EventLogEntryType.Error);
                try
                {
                    System.IO.StringWriter sw = new System.IO.StringWriter();
                    xmlc.Serialize(sw, TheConfig);
                    sw.WriteLine();
                    sw.WriteLine("------------");
                    sw.WriteLine();
                    xmls.Serialize(sw, TheConfig.Services);
                    System.IO.File.WriteAllText(configfile, sw.ToString());
                }
                catch (Exception) { }
                throw x;
            }
            finally
            {
                if (conn != null) conn.Close();
            }            
        }

        const string EvSource = "NExTSrvHost";

        protected override void OnStop()
        {
            NExT.NExTServer.Cleanup();
        }

        #region IWebApplication Members

        public string ApplicationName
        {
            get { return "NExTSrvHost_Svc"; }
        }

        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
        {
            SySal.Web.IWebApplication iwa = null;
            if (page.Length > 1)
                try
                {
                    iwa = (SySal.Web.IWebApplication)SySal.NExT.NExTServer.NExTServerFromURI(page.Substring(1));
                }
                catch (Exception) { };
            if (iwa != null) return iwa.HttpPost(sess, page, postfields);
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>NExTSrvHost Monitor</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "   <div><b>NExTSrvHost Monitor<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div>\r\n" +
                            "   <div align=\"center\">\r\n" +
                            "   <table width=\"100%\" border=\"1\">\r\n" +
                            "       <tr><th>Service</th><th>Status</th></tr>\r\n";
            foreach (SySal.NExT.NExTConfiguration.ServiceEntry se in TheConfig.Services.ServiceEntries)
            {
                foreach (string name in se.Names)
                {
                    SySal.NExT.INExTServer ins = SySal.NExT.NExTServer.NExTServerFromURI(name);
                    SySal.NExT.ServerMonitorGauge[] g = ins.MonitorGauges;
                    string mtext = "";
                    foreach (SySal.NExT.ServerMonitorGauge g1 in g)
                        mtext += SySal.Web.WebServer.HtmlFormat(g1.Name) + "=" + SySal.Web.WebServer.HtmlFormat(g1.Value.ToString()) + "; ";
                    html += "       <tr><td><a href=\"" + SySal.Web.WebServer.HtmlFormat(name) + "\">" + SySal.Web.WebServer.HtmlFormat(name) + "</a></td><td>" + mtext + "</td></tr>\r\n";
                }
            }
            html += "   <table>\r\n</body>\r\n\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        public bool ShowExceptions
        {
            get { return true; }
        }
        #endregion
    }
}
