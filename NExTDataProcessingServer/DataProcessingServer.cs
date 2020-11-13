using System;
using System.Collections.Generic;
using System.Text;
using System.Security;
[assembly: AllowPartiallyTrustedCallers]

namespace SySal.NExT
{
    /// <summary>
    /// Data event containing a data processing batch request.
    /// </summary>
    [Serializable]
    public class DataProcessingBatchDescEvent : SySal.NExT.DataEvent
    {
        /// <summary>
        /// Descriptor of a data processing batch.
        /// </summary>
        public SySal.DAQSystem.DataProcessingBatchDesc Desc;
        /// <summary>
        /// Builds a new DataProcessingBatchDescEvent, wrapping a DataProcessingBatchDesc.
        /// </summary>
        /// <param name="desc">the descriptor of the data processing batch.</param>
        /// <param name="emitter">the URI of the emitter.</param>
        /// <remarks>the token is automatically set to the batch Id.</remarks>
        public DataProcessingBatchDescEvent(SySal.DAQSystem.DataProcessingBatchDesc desc, string emitter)
        {
            Emitter = emitter;
            EventId = (long)desc.Id;
            Desc = desc;
        }
        /// <summary>
        /// Builds an empty event.
        /// </summary>
        public DataProcessingBatchDescEvent()
        { }
    }
    /// <summary>
    /// Data event containing information about completion of a batch.
    /// </summary>
    [Serializable]
    public class DataProcessingCompleteEvent : SySal.NExT.DataEvent
    {
        /// <summary>
        /// Id of the Batch;
        /// </summary>
        public ulong Id;
        /// <summary>
        /// Total processor time used by the process.
        /// </summary>
        public System.TimeSpan TotalProcessorTime;
        /// <summary>
        /// The maximum requirement of virtual memory the process has had during execution.
        /// </summary>
        public int PeakVirtualMemorySize;
        /// <summary>
        /// The maximum working set the process has had during execution.
        /// </summary>
        public int PeakWorkingSet;
        /// <summary>
        /// The exception that terminated the job; <c>null</c> if no exception was thrown.
        /// </summary>
        public System.Exception FinalException;
    }
    /// <summary>
    /// Data event notifying that a batch must be escalated because it consumes too many resources on a worker.
    /// </summary>
    [Serializable]
    public class DataProcessingEscalateEvent : SySal.NExT.DataEvent
    {
        /// <summary>
        /// Id of the Batch;
        /// </summary>
        public ulong Id;
    }
    /// <summary>
    /// Data processing server worker server. One <see cref="SySal.NExT.DataProcessingServer"/> uses one or more workers.
    /// </summary>
    /// <remarks>This version relies on the SySal.NExT technology. It is intended to work behind a DataProcessingServer, 
    /// so it processes only one batch at a time.</remarks>
    public class DataProcessingServerWorker : SySal.NExT.NExTServer, SySal.Web.IWebApplication
    {
        string m_TempDirBase;
        /// <summary>
        /// This server can send results to a data collector.
        /// </summary>
        public override string[] DataConsumerGroups
        {
            get { return new string[] { "ResultCollector" }; }
        }
        /// <summary>
        /// Gets the status of the worker.
        /// </summary>
        public override ServerMonitorGauge[] MonitorGauges
        {
            get 
            {
                ServerMonitorGauge[] g = new ServerMonitorGauge[]
                {
                    new ServerMonitorGauge(),
                    new ServerMonitorGauge(),
                    new ServerMonitorGauge(),
                    new ServerMonitorGauge(),
                    new ServerMonitorGauge(),
                    new ServerMonitorGauge(),
                    new ServerMonitorGauge()
                };
                g[0].Name = "MachinePowerClass";
                g[0].Value = MachinePowerClass;
                g[1].Name = "Batch";        
                g[2].Name = "Exe";
                g[3].Name = "Arguments";
                g[4].Name = "Description";
                g[5].Name = "PID";
                g[6].Name = "Start Time";
                g[1].Value = g[2].Value = g[3].Value = g[4].Value = g[5].Value = g[6].Value = "";
                DAQSystem.DataProcessingBatchDesc d = m_Exec;
                if (d != null)
                {
                    g[1].Value = d.Id.ToString("X016");
                    g[2].Value = d.Filename;
                    g[3].Value = d.CommandLineArguments;
                    g[4].Value = d.Description;
                    System.Diagnostics.Process p = m_ExecProc;
                    g[5].Value = (p == null) ? "" : p.Id.ToString();
                    g[6].Value = (p == null) ? "" : p.StartTime.ToString();
                }
                return g;
            }
        }

        SySal.DAQSystem.DataProcessingBatchDesc m_Exec;
        System.Diagnostics.Process m_ExecProc;
        bool m_ExecProcKilled;        
        uint ResultTimeoutSeconds = 600;
        uint OutputUpdateSeconds = 10;
        uint MaxOutputText = 65536;
        string DBServer = "";
        uint MachinePowerClass = 5;

        void ExecThread(object de)
        {
            System.Exception retX = null;
            string textout = "";
            string retXstr = "";
            SySal.DAQSystem.DataProcessingBatchDesc desc = ((DataProcessingBatchDescEvent)de).Desc;
            m_ExecProcKilled = false;
            m_ExecProc = new System.Diagnostics.Process();
            m_Exec = desc;
            m_ExecProc.StartInfo.Arguments = desc.CommandLineArguments;
            m_ExecProc.StartInfo.FileName = desc.Filename;
            m_ExecProc.StartInfo.UseShellExecute = false;
            m_ExecProc.StartInfo.RedirectStandardError = true;
            m_ExecProc.StartInfo.RedirectStandardOutput = true;
            desc.Started = desc.Finished = System.DateTime.Now;
            SySal.OperaDb.OperaDbCredentials cred = new SySal.OperaDb.OperaDbCredentials();
            try
            {
                cred.DBUserName = (desc.AliasUsername == null) ? "" : desc.AliasUsername;
                cred.DBPassword = (desc.AliasPassword == null) ? "" : desc.AliasPassword;
                cred.DBServer = DBServer;
                cred.OPERAUserName = (desc.Username == null) ? "" : desc.Username;
                cred.OPERAPassword = (desc.Password == null) ? "" : desc.Password;
                cred.RecordToEnvironment(m_ExecProc.StartInfo.EnvironmentVariables);
                m_ExecProc.StartInfo.EnvironmentVariables["TEMP"] = m_TempDirBase;
                m_ExecProc.StartInfo.EnvironmentVariables["TMP"] = m_TempDirBase;
                CleanTemp();
                desc.Started = System.DateTime.Now;
                m_ExecProc.Start();
            }
            catch (Exception x)
            {
                retX = new SySal.DAQSystem.DataProcessingException("Internal error occurred during process start.", x);
            }
            try
            {
                m_ExecProc.PriorityClass = System.Diagnostics.ProcessPriorityClass.BelowNormal;
            }
            catch (Exception) { }

            Log("Process prepared for batch: " + desc.Id.ToString("X016"));
            int c = -1;
            if (retX == null)
            {
                Log("Enter outer process waiter for batch " + desc.Id.ToString("X016"));
                do
                {
                    try
                    {
                        m_ExecProc.Refresh();
                        System.DateTime nextw = System.DateTime.Now.AddSeconds(OutputUpdateSeconds);
                        Log("Enter inner process waiter for batch " + desc.Id.ToString("X016"));
                        while (m_ExecProc.WaitForExit(0) == false && (c = m_ExecProc.StandardOutput.Read()) >= 0)
                        {
                            textout += (char)c;
                            if (textout.Length > MaxOutputText)
                                textout = textout.Remove(0, textout.Length - (int)MaxOutputText);
                            if (System.DateTime.Now >= nextw)
                                try
                                {
                                    if (desc.OutputTextSaveFile != null && desc.OutputTextSaveFile.Length > 0)
                                        System.IO.File.WriteAllText(desc.OutputTextSaveFile, textout);
                                }
                                catch (Exception) { }
                                finally
                                {
                                    nextw = System.DateTime.Now.AddSeconds(OutputUpdateSeconds);
                                }
                        }
                        Log("Exit inner process waiter for batch " + desc.Id.ToString("X016"));
                        while ((c = m_ExecProc.StandardError.Read()) >= 0)
                            retXstr += (char)c;
                        m_ExecProc.Refresh();
                        desc.TotalProcessorTime = m_ExecProc.TotalProcessorTime;
                        desc.PeakVirtualMemorySize = m_ExecProc.PeakVirtualMemorySize;
                        desc.PeakWorkingSet = m_ExecProc.PeakWorkingSet;
                        if (desc.OutputTextSaveFile != null && desc.OutputTextSaveFile.Length > 0)
                            System.IO.File.WriteAllText(desc.OutputTextSaveFile, textout);
                    }
                    catch (Exception) { }
                }
                while (m_ExecProc.WaitForExit(1000) == false);
                Log("Exit outer process waiter for batch " + desc.Id.ToString("X016"));
                textout += (char)c;
                if (textout.Length > MaxOutputText)
                    textout.Remove(0, textout.Length - (int)MaxOutputText);
                if (desc.OutputTextSaveFile != null && desc.OutputTextSaveFile.Length > 0)
                    try
                    {
                        System.IO.File.WriteAllText(desc.OutputTextSaveFile, textout);
                    }
                    catch (Exception) { }
                desc.Finished = System.DateTime.Now;
                if (retXstr == null || retXstr.Length == 0)
                    retX = null;
                else
                    retX = new SySal.DAQSystem.DataProcessingException(retXstr);
                if (m_ExecProcKilled) retX = new Exception("Process has been killed.");
            }
            else
            {
                try
                {
                    retXstr += m_ExecProc.StandardError.ReadToEnd();
                }
                catch (Exception) { }
            }
            try
            {
                m_ExecProc.Close();
            }
            catch (Exception) { }
            DataEvent dde = null;
            DataProcessingCompleteEvent dce = new DataProcessingCompleteEvent();
            dde = dce;
            dce.Emitter = NExTName;
            dce.EventId = ((DataProcessingBatchDescEvent)de).EventId;
            dce.Id = desc.Id;
            dce.PeakVirtualMemorySize = desc.PeakVirtualMemorySize;
            dce.PeakWorkingSet = desc.PeakWorkingSet;
            dce.TotalProcessorTime = desc.TotalProcessorTime;
            if (retX != null)
            {
                Log("Exception=" + retX.ToString());

                string low = retXstr.ToLower();
                if (low.Contains("outofmemory") || low.Contains("out of memory") || low.Contains("memoryexception"))
                {
                    DataProcessingEscalateEvent des = new DataProcessingEscalateEvent();
                    des.Emitter = NExTName;
                    des.EventId = ((DataProcessingBatchDescEvent)de).EventId;
                    des.Id = desc.Id;
                    dde = des;
                }
                dce.FinalException = new Exception(retXstr);
            }
            CleanTemp();
            ConsumerGroupRouters[0].RouteDataEvent(dde);
            lock (LockObj)
            {
                m_ExecProcKilled = false;
                m_ExecProc = null;
                m_Exec = null;
                DequeueDataEventAsCompleted(retX, dde, (int)ResultTimeoutSeconds * 1000);
            }
        }

        /// <summary>
        /// Processes a job.
        /// </summary>
        /// <param name="de">the data event must be a <see cref="DataProcessingBatchDescEvent"/>.</param>
        /// <returns></returns>
        public override bool OnDataEvent(DataEvent de)
        {
            Log(de.GetType().ToString() + "\r\nEventId=" + de.EventId + "\r\nEmitter=" + de.Emitter);
            if (de is AbortEvent)
            {
                SySal.DAQSystem.DataProcessingBatchDesc d = m_Exec;
                if (d == null || (long)d.Id != ((AbortEvent)de).StopId) throw new SySal.DAQSystem.DataProcessingException("Unknown batch.");
                m_ExecProcKilled = true;                
                try
                {
                    m_ExecProc.Kill();                    
                }
                catch (Exception) { }
                return true;
            }
            if (de is DataProcessingBatchDescEvent)
            {
                if (((DataProcessingBatchDescEvent)de).Desc.MachinePowerClass > MachinePowerClass) return false;
                lock (LockObj)
                {
                    if (m_ExecProc != null) return false;
                    RegisterDataEvent(de);
                }
                System.Threading.Thread thr = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(ExecThread));
                thr.Start(de);
                return true;
            }
            throw new NExTException(de.EventId, NExTName, "Unsupported event type \"" + de.GetType() + "\"; " + typeof(DataProcessingBatchDescEvent) + " expected.");
        }

        static DataProcessingServerWorker()
        {
            SySal.NExT.NExTConfiguration.ServerParameterDescriptor[] d = new NExTConfiguration.ServerParameterDescriptor[6];
            d[0] = new NExTConfiguration.ServerParameterDescriptor();
            d[0].Name = "DBSrv";
            d[0].Value = "";
            d[0].ValueType = typeof(string);
            d[0].CanBeStatic = d[0].CanBeDynamic = true;
            d[0].Description = "The Database server to be used to run batches. This overrides the setting in the batch.";
            d[1] = new NExTConfiguration.ServerParameterDescriptor();
            d[1].Name = "MaxOutputText";
            d[1].Value = "65536";
            d[1].ValueType = typeof(uint);
            d[1].CanBeStatic = d[1].CanBeDynamic = true;
            d[1].Description = "Maximum length of the output text.";
            d[2] = new NExTConfiguration.ServerParameterDescriptor();
            d[2].Name = "OutputUpdateSeconds";
            d[2].Value = "10";
            d[2].ValueType = typeof(uint);
            d[2].CanBeStatic = d[2].CanBeDynamic = true;
            d[2].Description = "Polling interval for the output text.";
            d[3] = new NExTConfiguration.ServerParameterDescriptor();
            d[3].Name = "MachinePowerClass";
            d[3].Value = "5";
            d[3].ValueType = typeof(uint);
            d[3].CanBeStatic = d[3].CanBeDynamic = true;
            d[3].Description = "Power class of the machine.";
            d[4] = new NExTConfiguration.ServerParameterDescriptor();
            d[4].Name = "ResultTimeoutSeconds";
            d[4].Value = "600";
            d[4].ValueType = typeof(uint);
            d[4].CanBeStatic = d[4].CanBeDynamic = true;
            d[4].Description = "Storage time of results in seconds.";
            d[5] = new NExTConfiguration.ServerParameterDescriptor();
            d[5].Name = "LogFile";
            d[5].Value = "";
            d[5].ValueType = typeof(string);
            d[5].CanBeStatic = d[5].CanBeDynamic = true;
            d[5].Description = "Log file for the server.";
            s_KnownParameters = d;
        }
        /// <summary>
        /// Bulds a new worker.
        /// </summary>
        /// <param name="name">the name of the worker.</param>
        /// <param name="publish">this parameter is ignored.</param>
        /// <param name="staticparams">static parameters:
        /// <list type="table">
        /// <listheader><term>Name</term><description>Description</description></listheader>
        /// <item><term>DBSrv</term><description>the DB server to pass to processing batches.</description></item>
        /// <item><term>MaxOutputText</term><description>the maximum size (in characters) of output text.</description></item>
        /// <item><term>OutputUpdateSeconds</term><description>the interval to poll the status of the process.</description></item>
        /// <item><term>MachinePowerClass</term><description>the power class of the machine.</description></item>
        /// <item><term>ResultTimeoutSeconds</term><description>maximum time to retain a result in seconds.</description></item>
        /// <item><term>LogFile</term><description>the file name to use as template for log files. If <c>null</c>, or pointing to a non-existing path, no log is generated.</description></item>
        /// </list>
        /// </param>
        /// <param name="dynparams">dynamic parameters:
        /// <list type="table">
        /// <listheader><term>Name</term><description>Description</description></listheader>
        /// <item><term>DBSrv</term><description>the DB server to pass to processing batches.</description></item>
        /// <item><term>MaxOutputText</term><description>the maximum size (in characters) of output text.</description></item>
        /// <item><term>OutputUpdateSeconds</term><description>the interval to poll the status of the process.</description></item>
        /// <item><term>MachinePowerClass</term><description>the power class of the machine.</description></item>
        /// <item><term>ResultTimeoutSeconds</term><description>maximum time to retain a result in seconds.</description></item>
        /// </list>
        /// </param>
        /// <remarks>
        /// <para>Results are sent to the "ResultCollector" consumer group.</para>
        /// </remarks>        
        public DataProcessingServerWorker(string name, bool publish, SySal.NExT.NExTConfiguration.ServerParameter[] staticparams, SySal.NExT.NExTConfiguration.ServerParameter[] dynparams)
            : base(name, publish, staticparams, dynparams) 
        {
            //ConsumerGroupRouters[0].WaitForCompletion = true;
            System.Collections.Generic.Dictionary<string, object> d = InterpretParameters(staticparams, dynparams);
            DBServer = (string)d["DBSrv"];
            MaxOutputText = (uint)d["MaxOutputText"];
            OutputUpdateSeconds = (uint)d["OutputUpdateSeconds"];
            MachinePowerClass = (uint)d["MachinePowerClass"];
            ResultTimeoutSeconds = (uint)d["ResultTimeoutSeconds"];            
            m_TempDirBase = System.Environment.GetEnvironmentVariable("TEMP");
            if (m_TempDirBase.EndsWith("/") == false && m_TempDirBase.EndsWith("\\") == false)
                m_TempDirBase += "/";
            m_TempDirBase += NExTName;
        }

        private void CleanTemp()
        {
            if (System.IO.Directory.Exists(m_TempDirBase)) System.IO.Directory.Delete(m_TempDirBase, true);
            System.IO.Directory.CreateDirectory(m_TempDirBase);
        }

        int[] LockObj = new int[0];

        #region IWebApplication Members

        public string ApplicationName
        {
            get { return NExTName; }
        }

        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
        {
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                            "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                            "<head>\r\n" +
                            "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                            "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                            "    <title>NExT Data Processing Server " + NExTName + "</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "   <div><b>NExT Data Processing Server \"" + NExTName + "\"<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div>\r\n" +
                            "   <div align=\"center\">\r\n" +
                            "   <table width=\"100%\" border=\"1\">\r\n" +
                            "       <tr><th>Parameter</th><th>Value</th></tr>\r\n";
            SySal.NExT.ServerMonitorGauge[] g = MonitorGauges;
            foreach (SySal.NExT.ServerMonitorGauge g1 in g)
                html += "       <tr><td>" + SySal.Web.WebServer.HtmlFormat(g1.Name) + "</td><td>" + SySal.Web.WebServer.HtmlFormat(g1.Value.ToString()) + "</td></tr>\r\n";
            html += "   <table>\r\n</body>\r\n\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        public bool ShowExceptions
        {
            get { return true; }
        }

        #endregion
    }
    /// <summary>
    /// Data processing server. It uses one or more <see cref="SySal.NExT.DataProcessingServerWorker"/>.
    /// </summary>
    public class DataProcessingServer : SySal.NExT.NExTServer, SySal.DAQSystem.IDataProcessingServer, SySal.Web.IWebApplication
    {
        class DataProcessingBatchResult : SySal.NExT.DataProcessingCompleteEvent
        {
            public System.DateTime ExpiryTime;
            public System.DateTime Started;
            public System.DateTime Finished;
        }
        /// <summary>
        /// Checks affinity between a worker and a batch.
        /// </summary>
        /// <param name="uri">the worker to be checked.</param>
        /// <param name="de">the DataEvent that contains the batch descriptor.</param>
        /// <returns><c>true</c> if the MachinePowerClass of the worker is not less than required by the batch descriptor.</returns>
        protected bool WorkerAffinityCheck(string uri, DataEvent de)
        {
            Log("Enter WorkerAffinityCheck");
            string[] uris = ConsumerGroupRouters[0].URIs;
            Log("uris.length = " + uris.Length + " m_MachinePowerClass" + ((m_MachinePowerClass == null) ? "NULL" : m_MachinePowerClass.Length.ToString()));
            if (m_MachinePowerClass == null || uris.Length != m_MachinePowerClass.Length) return false;
            int i;
            for (i = 0; i < uris.Length && String.Compare(uris[i], uri, true) != 0; i++) ;
            if (i == uris.Length) return false;
            Log("WorkerAffinityCheck " + m_MachinePowerClass[i] + "/" + ((DataProcessingBatchDescEvent)de).Desc.MachinePowerClass);
            try
            {
                return (m_MachinePowerClass[i] > 0) && (m_MachinePowerClass[i] >= ((DataProcessingBatchDescEvent)de).Desc.MachinePowerClass);
            }
            catch (Exception)
            {
                return false;
            }
        }
        /// <summary>
        /// Checks whether a job should be rerouted.
        /// </summary>
        /// <param name="ded">the data event completed.</param>
        /// <returns>always returns <c>false</c>.</returns>
        protected bool WorkerShouldReroute(DataEventDone ded)
        {
            DataEvent de = (DataEvent)ded.Info;
            Log("DataEventDone: " + ded.GetType());
            if (de is DataProcessingEscalateEvent)
            {
                SySal.NExT.DataProcessingEscalateEvent dce = (SySal.NExT.DataProcessingEscalateEvent)de;
                SySal.DAQSystem.DataProcessingBatchDesc d = null;
                lock (m_BatchList)
                    if (m_BatchList.ContainsKey(dce.Id))
                    {
                        Log("Increase PowerClass " + dce.Id.ToString("X16"));
                        d = m_BatchList[dce.Id];
                        d.MachinePowerClass++;
                    }
                if (ConsumerGroupRouters[0].RouteDataEvent(new DataProcessingBatchDescEvent(d, NExTName)) == false)
                    lock (m_BatchList)
                    {
                        DataProcessingBatchResult r = new DataProcessingBatchResult();
                        r.Emitter = NExTName;
                        r.EventId = (long)d.Id;
                        r.ExpiryTime = System.DateTime.Now;
                        r.ExpiryTime = r.ExpiryTime.AddSeconds((int)ResultTimeoutSeconds);
                        r.FinalException = new NExTException(dce.EventId, NExTName, "Unable to escalate the powerclass of this batch.\r\nBatch refused at powerclass = " + d.MachinePowerClass + ".");
                        r.Finished = System.DateTime.Now;
                        r.Id = d.Id;
                        r.PeakVirtualMemorySize = 0;
                        r.PeakWorkingSet = 0;
                        r.Started = d.Started;
                        r.TotalProcessorTime = new TimeSpan(0);
                        Log("Add result " + r.Id.ToString("X16"));
                        m_ResultList.Add(r.Id, r);
                        Log("Remove batch " + dce.Id.ToString("X16"));
                        m_BatchList.Remove(dce.Id);
                    }
            }
            else if (de is DataProcessingCompleteEvent)
            {
                SySal.NExT.DataProcessingCompleteEvent dce = (SySal.NExT.DataProcessingCompleteEvent)de;
                lock (m_BatchList)
                    if (m_BatchList.ContainsKey(dce.Id))
                    {
                        SySal.DAQSystem.DataProcessingBatchDesc d = m_BatchList[dce.Id];
                        DataProcessingBatchResult r = new DataProcessingBatchResult();
                        r.Emitter = dce.Emitter;
                        r.EventId = dce.EventId;
                        r.ExpiryTime = System.DateTime.Now;
                        r.ExpiryTime = r.ExpiryTime.AddSeconds((int)ResultTimeoutSeconds);
                        r.FinalException = dce.FinalException;
                        r.Finished = System.DateTime.Now;
                        r.Id = dce.Id;
                        r.PeakVirtualMemorySize = dce.PeakVirtualMemorySize;
                        r.PeakWorkingSet = dce.PeakWorkingSet;
                        r.Started = d.Started;
                        r.TotalProcessorTime = dce.TotalProcessorTime;
                        Log("Add result complete " + r.Id.ToString("X16"));
                        m_ResultList.Add(r.Id, r);
                        Log("Remove complete batch " + dce.Id.ToString("X16"));
                        m_BatchList.Remove(dce.Id);
                    }                
            }
            return false;
        }
        /// <summary>
        /// Returns the set of data consumers: in this case there is only one group, named "Workers".
        /// </summary>
        public override string[] DataConsumerGroups
        {
            get { return new string[] { "Workers" }; }
        }
        /// <summary>
        /// Processes data events.
        /// </summary>
        /// <param name="de">the event to be processed.</param>
        /// <returns><c>true</c> if the event is accepted, <c>false</c> otherwise.</returns>
        /// <remarks>This implementation always returns <c>true</c>.</remarks>
        public override bool OnDataEvent(DataEvent de)
        {
            return true;
        }
        /// <summary>
        /// Provides a representation of the internal status fo the DataProcessingServer.
        /// </summary>
        public override ServerMonitorGauge[] MonitorGauges
        {
            get 
            {
                SySal.NExT.ServerMonitorGauge[] g = new ServerMonitorGauge[]
                {
                    new SySal.NExT.ServerMonitorGauge(),
                    new SySal.NExT.ServerMonitorGauge(),                    
                    new SySal.NExT.ServerMonitorGauge(),
                    new SySal.NExT.ServerMonitorGauge()
                };
                g[0].Name = "MachinePowerClass";
                g[0].Value = MachinePowerClass;
                g[1].Name = "MaxQueueLength";
                g[1].Value = ConsumerGroupRouters[0].MaxQueueLength;
                g[2].Name = "QueueLength";
                g[2].Value = QueueLength;
                g[3].Name = "Workers";
                g[3].Value = ConsumerGroupRouters[0].URIs.Length;
                return g;
            }
        }

        static DataProcessingServer()
        {
            NExTConfiguration.ServerParameterDescriptor[] d = new NExTConfiguration.ServerParameterDescriptor[6];
            d[0] = new NExTConfiguration.ServerParameterDescriptor();
            d[0].Name = "DBSrv";
            d[0].Value = "";
            d[0].ValueType = typeof(string);
            d[0].CanBeDynamic = d[0].CanBeStatic = true;
            d[0].Description = "The Database server to be accessed.";
            d[1] = new NExTConfiguration.ServerParameterDescriptor();
            d[1].Name = "DBUsr";
            d[1].Value = "";
            d[1].ValueType = typeof(string);
            d[1].CanBeDynamic = d[1].CanBeStatic = true;
            d[1].Description = "Username to access the DB. This will be used to validate batch users.";
            d[2] = new NExTConfiguration.ServerParameterDescriptor();
            d[2].Name = "DBPwd";
            d[2].Value = "";
            d[2].ValueType = typeof(string);
            d[2].CanBeDynamic = d[2].CanBeStatic = true;
            d[2].Description = "Password to access the DB.";
            d[3] = new NExTConfiguration.ServerParameterDescriptor();
            d[3].Name = "ResultTimeoutSeconds";
            d[3].Value = "600";
            d[3].ValueType = typeof(uint);
            d[3].CanBeDynamic = d[3].CanBeStatic = true;
            d[3].Description = "Storage time for batch results.";
            d[4] = new NExTConfiguration.ServerParameterDescriptor();
            d[4].Name = "MaxQueueLength";
            d[4].Value = "100";
            d[4].ValueType = typeof(uint);
            d[4].CanBeStatic = d[4].CanBeDynamic = true;
            d[4].Description = "Maximum number of batches in the queue.";
            d[5] = new NExTConfiguration.ServerParameterDescriptor();
            d[5].Name = "LogFile";
            d[5].Value = "";
            d[5].ValueType = typeof(string);
            d[5].CanBeStatic = d[5].CanBeDynamic = true;
            d[5].Description = "Log file for the server.";
            s_KnownParameters = d;
        }
        /// <summary>
        /// Builds a new DataProcessingServer.
        /// </summary>
        /// <param name="name">the name of the data processing server object.</param>
        /// <param name="publish"><c>true</c> to publish, <c>false</c> otherwise.</param>
        /// <param name="staticparams">the list of static parameters.</param>
        /// <param name="dynparams">the list of dynamic parameters.</param>
        /// <remarks>The following parameters are currently understood both as static and dynamic, in addition to default <see cref="NExtServer"/> parameters:
        /// <list type="table">
        /// <listheader><term>Parameter</term><description>Meaning</description></listheader>
        /// <item><term>ResultTimeoutSeconds</term><description>The duration in seconds of a result.</description></item>
        /// <item><term>DBSrv</term><description>Database server.</description></item>
        /// <item><term>DBUsr</term><description>Database username.</description></item>
        /// <item><term>DBPwd</term><description>Database password.</description></item>
        /// </list>
        /// </remarks>
        public DataProcessingServer(string name, bool publish, SySal.NExT.NExTConfiguration.ServerParameter[] staticparams, SySal.NExT.NExTConfiguration.ServerParameter[] dynparams)
            : base(name, publish, staticparams, dynparams)
        {
            m_MonitorTimer.Elapsed += new System.Timers.ElapsedEventHandler(m_MonitorTimer_Elapsed);            
            ConsumerGroupRouters[0].AffinityChecker = new ConsumerGroupRouter.dCheckAffinity(WorkerAffinityCheck);
            ConsumerGroupRouters[0].ShouldReroute = new ConsumerGroupRouter.dShouldReroute(WorkerShouldReroute);
            ConsumerGroupRouters[0].WaitForCompletion = true;
            System.Collections.Generic.Dictionary<string, object> d = InterpretParameters(staticparams, dynparams);
            ResultTimeoutSeconds = (uint)d["ResultTimeoutSeconds"];
            DBSrv = (string)d["DBSrv"];
            DBUsr = (string)d["DBUsr"];
            DBPwd = (string)d["DBPwd"];
            m_MonitorTimer.Enabled = true;
            m_MonitorTimer.Start();
        }

        long m_IdSite = 0;
        long IdSite
        {
            get
            {
                if (m_IdSite != 0) return m_IdSite;
                SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(DBSrv, DBUsr, DBPwd);
                try
                {
                    conn.Open();
                    m_IdSite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT TO_NUMBER(VALUE) FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE'", conn).ExecuteScalar());
                    return m_IdSite;
                }
                finally
                {
                    conn.Close();
                }                
            }
        }

        uint ResultTimeoutSeconds = 600;
        string DBSrv = "";
        string DBUsr = "";
        string DBPwd = "";        

        void m_MonitorTimer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            string[] workers = ConsumerGroupRouters[0].URIs;
            uint[] mp = new uint[workers.Length];
            int i;
            for (i = 0; i < workers.Length; i++)            
            {
                string w = workers[i];
                try
                {
                    SySal.NExT.INExTServer ins = SySal.NExT.NExTServer.NExTServerFromURI(w);
                    SySal.NExT.ServerMonitorGauge[] g = ins.MonitorGauges;
                    foreach (SySal.NExT.ServerMonitorGauge g1 in g)
                        if (String.Compare(g1.Name, "MachinePowerClass", true) == 0)
                            mp[i] = Convert.ToUInt32(g1.Value);
                }
                catch (Exception) 
                {
                    mp[i] = 0;
                }
            }
            m_MachinePowerClass = mp;
            int j;
            for (j = 0; j < m_MachinePowerClass.Length; j++)
                Log("Worker " + j + ": " + m_MachinePowerClass[j]);
            lock (m_ResultList)
            {
                System.DateTime now = System.DateTime.Now;
                foreach (ulong key in m_ResultList.Keys)
                    if (m_ResultList[key].ExpiryTime < now)
                    {
                        Log("Expired batch " + key.ToString("X16") + " - expiry time: " + m_ResultList[key].ExpiryTime);
                        m_ResultList.Remove(key);
                    }
            }
        }

        #region IDataProcessingServer Members
        /// <summary>
        /// Returns the number of jobs that can be executed in parallel.
        /// </summary>
        /// <remarks>this number is identical to the number of workers, in this implementation.</remarks>
        public uint ParallelJobs
        {
            get { return (uint)ConsumerGroupRouters[0].URIs.Length; }
        }

        System.Collections.Generic.Dictionary<ulong, SySal.DAQSystem.DataProcessingBatchDesc> m_BatchList = new Dictionary<ulong,SySal.DAQSystem.DataProcessingBatchDesc>();

        System.Collections.Generic.Dictionary<ulong, DataProcessingBatchResult> m_ResultList = new Dictionary<ulong, DataProcessingBatchResult>();

        /// <summary>
        /// The list of batch jobs in the processing queue.
        /// </summary>
        public SySal.DAQSystem.DataProcessingBatchDesc[] Queue
        {
            get 
            {
                SySal.DAQSystem.DataProcessingBatchDesc[] darr = new SySal.DAQSystem.DataProcessingBatchDesc[m_BatchList.Count];
                int i = 0;
                foreach (SySal.DAQSystem.DataProcessingBatchDesc de in m_BatchList.Values)
                    darr[i++] = de;
                return darr;
            }
        }
        /// <summary>
        /// The length of the processing queue.
        /// </summary>
        public int QueueLength
        {
            get { return m_BatchList.Count; }
        }

        System.Timers.Timer m_MonitorTimer = new System.Timers.Timer(10000);

        uint[] m_MachinePowerClass;
        /// <summary>
        /// Retrieves the power class of this cluster.
        /// </summary>
        /// <remarks>In this implementation, this number is the largest MachinePowerClass of all workers.</remarks>
        public int MachinePowerClass
        {
            get 
            {
                if (m_MachinePowerClass == null) return 0;
                int mp = 0;
                foreach (int mp1 in m_MachinePowerClass)
                    mp = Math.Max(mp1, mp);
                return mp;
            }
        }
        /// <summary>
        /// Removes a batch from the queue or aborts it if it is already being executed.
        /// A non-null token or a username/password pair must be supplied that matches the one with which the batch was started.
        /// If the token is supplied, the username/password pair is ignored.
        /// </summary>
        /// <param name="id">identifier of the batch to be removed.</param>
        /// <param name="token">the process token to be used.</param>
        /// <param name="user">username of the user that started the batch. Ignored if <c>token</c> is non-null.</param>
        /// <param name="password">password of the user that started the batch. Ignored if <c>token</c> is non-null.</param>
        public void Remove(ulong id, string token, string user, string password)
        {
            lock (m_BatchList)
            {
                Log("Remove batch " + id.ToString("X16"));
                if (m_BatchList.ContainsKey(id) == false) throw new Exception("Unknown batch " + id.ToString("X16") + ".");
                SySal.DAQSystem.DataProcessingBatchDesc desc = m_BatchList[id];
                bool OkToRemove = false;
                if (token != null)
                {
                    if (desc.Token != null)
                    {
                        if (desc.Token == token) OkToRemove = true;
                        else throw new Exception("A process operation cannot remove a batch of another process operation.");
                    }
                    else throw new Exception("A process operation cannot remove a batch that has been started with a specific user request.");
                }
                else
                {
                    if (DBSrv != null && DBSrv.Length > 0)
                    {
                        long id_user = 0;
                        SySal.OperaDb.OperaDbConnection conn = null;
                        try
                        {
                            conn = new SySal.OperaDb.OperaDbConnection(DBSrv, DBUsr, DBPwd);
                            conn.Open();
                            id_user = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(user, password, conn, null);
                            if (desc.Token != null)
                            {
                                try
                                {
                                    SySal.OperaDb.ComputingInfrastructure.User.CheckTokenOwnership(desc.Token, id_user, null, null, conn, null);
                                    OkToRemove = true;
                                }
                                catch (Exception)
                                {
                                    throw new Exception("A user cannot remove a batch started by an operation of another user.");
                                }
                            }
                            else
                            {
                                if (String.Compare(desc.Username, user, true) != 0)
                                    throw new Exception("A user cannot stop a batch scheduled by another user!");
                                else OkToRemove = true;
                            }
                        }
                        catch (Exception)
                        {
                            if (conn != null) conn.Close();
                        }
                    }
                    else OkToRemove = true;
                }
                if (OkToRemove == false) throw new Exception("Cannot remove the batch.");
                else
                {
                    if (ConsumerGroupRouters[0].CancelRouting((long)id))
                    {
                        Log("Requested remove batch " + id.ToString("X16"));
                        m_BatchList.Remove(id);
                        DataProcessingBatchResult res = new DataProcessingBatchResult();
                        res.FinalException = new Exception("The batch was removed from the queue.");
                        res.Id = desc.Id;
                        res.Started = desc.Started;
                        res.Finished = System.DateTime.Now;
                        res.ExpiryTime = System.DateTime.Now;
                        res.ExpiryTime = res.ExpiryTime.AddSeconds((int)ResultTimeoutSeconds);
                        Log("Added removed result " + id.ToString("X16"));
                        m_ResultList.Add(id, res);
                    }
                }
            }
        }
        /// <summary>
        /// Enqueues a batch without waiting for its execution.
        /// </summary>
        /// <param name="desc">the descriptor of the batch. If the batch is rejected because another batch in the queue already has the same id, the Id member is set to 0.</param>
        /// <returns>true if the batch has been accepted, false otherwise.</returns>
        public bool Enqueue(SySal.DAQSystem.DataProcessingBatchDesc desc)
        {
            if (DBSrv != null && DBSrv.Length > 0)
            {
                SySal.OperaDb.OperaDbConnection conn = null;
                try
                {
                    conn = new SySal.OperaDb.OperaDbConnection(DBSrv, DBUsr, DBPwd);
                    conn.Open();
                    SySal.OperaDb.ComputingInfrastructure.UserPermission[] rights = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
                    rights[0].DB_Site_Id = IdSite;
                    rights[0].Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.ProcessData;
                    rights[0].Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
                    if (desc.Token != null)
                    {
                        if (!SySal.OperaDb.ComputingInfrastructure.User.CheckTokenAccess(desc.Token, rights, conn, null)) throw new Exception("The user does not own the permission to process data in this site");
                    }
                    else SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(desc.Username, desc.Password, conn, null), rights, true, conn, null);
                    conn.Close();
                }
                catch (System.Exception)
                {
                    if (conn != null) conn.Close();
                    return false;
                }
            }
            lock (m_BatchList)
            {
                desc.Started = System.DateTime.Now;
                Log("Add batch " + desc.Id.ToString("X16"));
                m_BatchList.Add(desc.Id, desc);
                try
                {
                    if (ConsumerGroupRouters[0].RouteDataEvent(new DataProcessingBatchDescEvent(desc, NExTName)) == false)
                    {
                        Log("Failed, remove batch " + desc.Id.ToString("X16"));
                        m_BatchList.Remove(desc.Id);
                        return false;
                    }
                }
                catch (Exception x)
                {
                    Log("X, remove batch " + desc.Id.ToString("X16") + "\r\n" + x.ToString());
                    m_BatchList.Remove(desc.Id);
                    throw x;
                }
            }
            return true;
        }
        /// <summary>
        /// Checks for execution completion.
        /// </summary>
        /// <param name="id">the id of the batch.</param>
        /// <returns>true if the batch has been completed, false if it is in progress.</returns>
        public bool DoneWith(ulong id)
        {
            lock (m_BatchList)
            {
                if (m_BatchList.ContainsKey(id)) return false;
                if (m_ResultList.ContainsKey(id)) return true;
                throw new SySal.DAQSystem.DataProcessingException("Unknown batch " + id.ToString("X16") + ".");
            }
        }
        /// <summary>
        /// Gets the result for a batch.
        /// </summary>
        /// <param name="id">the id of the batch.</param>
        /// <returns>the batch descriptor. It is modified to reflect the batch output. An exception is thrown if the batch terminated with an exception.</returns>
        public SySal.DAQSystem.DataProcessingBatchDesc Result(ulong id)
        {
            lock (m_BatchList)
            {
                DataProcessingBatchResult res = null;
                try
                {
                    Log("Check result " + id.ToString("X16"));
                    res = m_ResultList[id];
                }
                catch (Exception)
                {
                    throw new SySal.DAQSystem.DataProcessingException("Unknown batch " + id.ToString("X16") + ".");
                }
                if (res.FinalException != null) throw res.FinalException;
                SySal.DAQSystem.DataProcessingBatchDesc desc = new SySal.DAQSystem.DataProcessingBatchDesc();
                desc.Id = res.Id;
                desc.AliasPassword = desc.AliasUsername = desc.CommandLineArguments = desc.Description = desc.Filename = 
                    desc.OutputTextSaveFile = desc.Password = desc.Token = desc.Username = "";
                desc.Finished = res.Finished;
                desc.PeakVirtualMemorySize = res.PeakVirtualMemorySize;
                desc.PeakWorkingSet = res.PeakWorkingSet;
                desc.Started = res.Started;
                desc.TotalProcessorTime = res.TotalProcessorTime;
                return desc;
            }
        }
        /// <summary>
        /// Suggests a unique Id.
        /// </summary>
        /// <remarks>In this implementation, the Id runs with time.</remarks>
        public ulong SuggestId
        {
            get { return (ulong)System.DateTime.Now.Ticks; }
        }
        /// <summary>
        /// Tells whether this DataProcessingServer is saturated or not.
        /// </summary>
        public bool IsWillingToProcess
        {
            get { return ConsumerGroupRouters[0].DataEventQueue.Length < ConsumerGroupRouters[0].MaxQueueLength; }
        }
        /// <summary>
        /// Tests the communication.
        /// </summary>
        /// <param name="commpar">the communication parameter.</param>
        /// <returns>2 * commpar - 1.</returns>
        public int TestComm(int commpar)
        {
            return 2 * commpar - 1;
        }

        #endregion

        #region IWebApplication Members

        public string ApplicationName
        {
            get { return "DataProcessingServer(" + NExTName + ")"; }
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
                        bd.Id = SuggestId;
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
                        if (Enqueue(bd) == false) throw new Exception("Batch refused.");
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
                            Remove(u, null, user, pwd);
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
                            "    <title>DataProcessingServer</title>\r\n" +
                            "    <style type=\"text/css\">\r\n" +
                            "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                            "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                            "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                            "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                            "    </style>\r\n" +
                            "</head>\r\n" +
                            "<body>\r\n" +
                            "<div><b>DataProcessingServer " + SySal.Web.WebServer.HtmlFormat(NExTName) + "<br>Last Update: " + System.DateTime.Now.ToLongTimeString() + "</b></div>\r\n" +
                            "<br><a href=\"" + page + "\">Refresh</a><br>\r\n" +
                            "<form action=\"" + page + "\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n";
            if (xctext.Length > 0)
                html += "<p><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "</font></p>\r\n";
            SySal.DAQSystem.DataProcessingBatchDesc[] batches = Queue;
            html += "<table border=\"1\" align=\"center\" width=\"100%\">\r\n" +
                    " <tr><th width=\"10%\">Batch</th><th width=\"5%\">PowerClass</th><th width=\"65%\">Description</th><th width=\"10%\">Owner</th><th width=\"10%\">Started</th></tr>\r\n";
            foreach (SySal.DAQSystem.DataProcessingBatchDesc b in batches)
                html += " <tr><td><input id=\"" + CheckCmd + b.Id + "\" name=\"" + CheckCmd + b.Id + "\" type=\"checkbox\" />" + b.Id.ToString("X16") + "</td><td>" + b.MachinePowerClass + "</td><td>" + SySal.Web.WebServer.HtmlFormat(b.Description) +
                    ((expid == b.Id) ? ("<br><div align=\"left\"><font face=\"Courier\"><c>" + SySal.Web.WebServer.HtmlFormat(b.Filename + " " + b.CommandLineArguments) + "</c></font></div>&nbsp;<a href=\"" + page + "?" + ExpandCmd + "=0\"><i>Shrink</i></a>") : ("&nbsp;<a href=\"" + page + "?" + ExpandCmd + "=" + b.Id + "\"><i>Expand</i></a>")) +
                    "</td><td>&nbsp;" + SySal.Web.WebServer.HtmlFormat((b.Username == null || b.Username == "") ? "N/A" : b.Username) + "</td><td>&nbsp;" + b.Started.ToString() + "</td></tr>\r\n";

            html += "</table>\r\n" +
                    "<p><input id=\"" + EnqBtn + "\" name=\"" + EnqBtn + "\" type=\"submit\" value=\"Enqueue\"/>&nbsp;<input id=\"" + RemBtn + "\" name=\"" + RemBtn + "\" type=\"submit\" value=\"Remove Selected\"/></p>\r\n" +
                    "<p>Description <input id=\"" + DescCmd + "\" maxlength=\"1024\" name=\"" + DescCmd + "\" size=\"50\" type=\"text\" /></p>\r\n" +
                    "<p>Executable <input id=\"" + ExePathCmd + "\" maxlength=\"1024\" name=\"" + ExePathCmd + "\" size=\"50\" type=\"text\" value=\"\" /></p>\r\n" +
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
            html += "</body>\r\n";
            return new SySal.Web.HTMLResponse(html);
        }

        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        /// <summary>
        /// Exceptions are shown.
        /// </summary>
        public bool ShowExceptions
        {
            get { return true; }
        }

        #endregion
    }
}
