using System;
using SySal.DAQSystem;
using System.Runtime.Serialization;
using SySal.Services.OperaBatchManager_Win;

namespace SySal.DAQSystem
{
    /// <summary>
    /// Data Processing Server (Manager). 
    /// </summary>
    /// <remarks>
    /// This class implements a DataProcessingServer cluster manager. The list of available worker DataProcessingServer machines is obtained from the DB.
    /// </remarks>
    public class MyDataProcessingServer2 : MarshalByRefObject, SySal.DAQSystem.IDataProcessingServer
    {

        /// <summary>
        /// The number of jobs that can be performed in parallel. 
        /// </summary>
        public uint ParallelJobs
        {
            get
            {
                return 1;
            }
        }

        /// <summary>
        /// Class holding the results of the processing of a batch.
        /// </summary>
        internal class DataProcessingResult
        {
            /// <summary>
            /// Id of the batch.
            /// </summary>
            public DataProcessingBatchDesc Desc;
            /// <summary>
            /// Exception generated during the batch execution.
            /// </summary>
            public System.Exception X;
            /// <summary>
            /// Tells whether the batch has been processed or is still awaiting execution.
            /// </summary>
            public bool Processed;
            /// <summary>
            /// Expiration time of this result.
            /// </summary>
            public System.DateTime ExpirationTime;
            /// <summary>
            /// Public constructor.
            /// </summary>
            /// <param name="desc">Descriptor of the batch.</param>
            /// <param name="x">Resulting exception.</param>
            /// <param name="resultlivetime">Time to live of the result.</param>
            public DataProcessingResult(DataProcessingBatchDesc desc, System.Exception x, System.TimeSpan resultlivetime)
            {
                Desc = desc;
                X = x;
                Processed = false;
                ExpirationTime = System.DateTime.Now + resultlivetime;
            }
        }
        /// <summary>
        /// A thread that handles a slave DPS.
        /// </summary>
        internal class ExeThread
        {
            /// <summary>
            /// DataProcessingServer that is handling the batch.
            /// </summary>
            public string ServerName;

            /// <summary>
            /// Batch being executed.
            /// </summary>
            public SySal.DAQSystem.DataProcessingBatchDesc Desc;

            /// <summary>
            /// The queue of batches to be executed.
            /// </summary>
            public System.Collections.ArrayList JobQueue;

            /// <summary>
            /// List of the batches being executed.
            /// </summary>
            public System.Collections.ArrayList ExeList;

            /// <summary>
            /// List of results.
            /// </summary>
            public System.Collections.ArrayList ResultList;

            /// <summary>
            /// Fires when the job queue is not empty.
            /// </summary>
            public System.Threading.ManualResetEvent DataReady;

            /// <summary>
            /// Power class of the associated machine.
            /// </summary>
            public int MachinePowerClass;

            /// <summary>
            /// The execution thread.
            /// </summary>
            public System.Threading.Thread XThread;

            /// <summary>
            /// Builds a new execution thread.
            /// </summary>
            /// <param name="sname">the server this thread monitors.</param>
            public ExeThread(string sname, System.Collections.ArrayList jq, System.Collections.ArrayList el, System.Threading.ManualResetEvent dr, System.Collections.ArrayList rs)
            {
                ServerName = sname;
                Desc = null;
                JobQueue = jq;
                ExeList = el;
                ResultList = rs;
                DataReady = dr;
                XThread = new System.Threading.Thread(new System.Threading.ThreadStart(Execute));
                XThread.Start();
            }

            public System.Threading.AutoResetEvent KillEvent = new System.Threading.AutoResetEvent(false);

            /// <summary>
            /// The execution method, ran by the execution thread.
            /// </summary>
            public void Execute()
            {
                SySal.DAQSystem.SyncDataProcessingServerWrapper DPSW = null;
                int test = 0;
                string milestone = "A";
                try
                {
                    while (true)
                    {
                        try
                        {
                            if (DPSW == null)
                            {
                                DPSW = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + ServerName + ":" + ((int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), new TimeSpan(0, 1, 0));
                                if (DPSW.TestComm(++test) != 2 * test - 1) throw new Exception();
                                MachinePowerClass = DPSW.MachinePowerClass;
                            }
                        }
                        catch (Exception) 
                        { 
                            DPSW = null; 
                            continue; 
                        }
                        milestone = "B";
                        try
                        {
                            Desc = null;
                            int i;
                            lock (JobQueue)
                            {
                                for (i = 0; i < JobQueue.Count; i++)
                                    if (((SySal.DAQSystem.DataProcessingBatchDesc)JobQueue[i]).MachinePowerClass <= MachinePowerClass)
                                    {
                                        KillEvent.Reset();
                                        Desc = (SySal.DAQSystem.DataProcessingBatchDesc)JobQueue[i];
                                        JobQueue.RemoveAt(i);
                                        ExeList.Add(Desc);
                                        break;
                                    }
                                if (JobQueue.Count == 0) DataReady.Reset();
                            }
                            milestone = "C";
                            if (Desc == null)
                            {
                                DataReady.WaitOne();
                                continue;
                            }
                            else
                                try
                                {
                                    SySal.DAQSystem.DataProcessingBatchDesc mbd = (SySal.DAQSystem.DataProcessingBatchDesc)Desc.Clone();
                                    milestone = "D";
                                    mbd.Id = DPSW.SuggestId;
                                    milestone = "E";
                                    mbd.Description = Desc.Id.ToString("X16") + " _DPS_REMAP_ " + Desc.Description;
                                    if (MainForm.ImpersonateBatchUser == false)
                                    {
                                        mbd.Username = MainForm.OPERAUserName;
                                        mbd.Password = MainForm.OPERAPassword;
                                        mbd.Token = null;
                                    }
                                    milestone = "E1";
                                    if (DPSW.Enqueue(mbd) == false)
                                        lock (JobQueue)
                                        {
                                            ExeList.Remove(Desc);
                                            JobQueue.Add(Desc);
                                            Desc = null;
                                            continue;
                                        }
                                    milestone = "F";
                                    bool killed = false;
                                    while (DPSW.DoneWith(mbd.Id) == false)
                                    {
                                        if (KillEvent.WaitOne(MainForm.DataProcSrvMonitorInterval * 1000))
                                        {
                                            milestone = "F1";
                                            lock (JobQueue)
                                            {
                                                milestone = "F2";
                                                ResultList.Add(new DataProcessingResult(Desc, new Exception("The batch was removed from the queue."), MainForm.ResultLiveTime));
                                                ExeList.Remove(Desc);
                                                Desc = null;
                                                milestone = "F3";
                                                try
                                                {
                                                    DPSW.Remove(mbd.Id, mbd.Token, mbd.Username, mbd.Password);
                                                }
                                                catch (Exception) { }
                                                milestone = "F4";
                                                killed = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (killed == false)
                                    {
                                        milestone = "G";
                                        mbd = DPSW.Result(mbd.Id);
                                        milestone = "H";
                                        Desc.PeakVirtualMemorySize = mbd.PeakVirtualMemorySize;
                                        Desc.PeakWorkingSet = mbd.PeakWorkingSet;
                                        Desc.TotalProcessorTime = mbd.TotalProcessorTime;
                                        Desc.Finished = mbd.Finished;
                                        lock (JobQueue)
                                        {
                                            ResultList.Add(new DataProcessingResult(Desc, null, MainForm.ResultLiveTime));
                                            ExeList.Remove(Desc);
                                            Desc = null;
                                        }
                                    }
                                    milestone = "I";
                                }
                                catch (SySal.DAQSystem.DataProcessingException retx)
                                {                                    
                                    lock (JobQueue)
                                    {
                                        ResultList.Add(new DataProcessingResult(Desc, retx, MainForm.ResultLiveTime));
                                        ExeList.Remove(Desc);
                                        Desc = null;                                        
                                    }
                                }
                                catch (Exception x)
                                {
                                    DPSW = null;
                                    MachinePowerClass = 0;
                                    try
                                    {
                                        EventLog.WriteEntry("Error handling batch " + Desc.Id.ToString("X16") + " - DPS: " + ServerName + "\r\nMilestone: " + milestone + "\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Warning);
                                    }
                                    catch (Exception) { }
                                    lock (JobQueue)
                                    {                                        
                                        ExeList.Remove(Desc);
                                        JobQueue.Add(Desc);
                                        Desc = null;
                                    }
                                }
                        }
                        catch (Exception)
                        {
                            Desc = null;
                            DPSW = null;
                            MachinePowerClass = 0;
                        }
                    }
                }
                catch (System.Threading.ThreadAbortException a)
                {
                    System.Threading.Thread.ResetAbort();
                    return;
                }
            }
        }

        /// <summary>
        /// The internal queue of batches to be executed.
        /// </summary>
        internal System.Collections.ArrayList m_Queue = new System.Collections.ArrayList();
        /// <summary>
        /// The internal queue of batches being executed.
        /// </summary>
        internal System.Collections.ArrayList m_ExeList = new System.Collections.ArrayList();
        /// <summary>
        /// The internal list of completed batches.
        /// </summary>
        internal System.Collections.ArrayList m_ResultList = new System.Collections.ArrayList();
        /// <summary>
        /// Threads that handle the slave DPS.
        /// </summary>
        internal ExeThread[] m_ExeThreads;
        /// <summary>
        /// Event firing when the execution queue is not empty.
        /// </summary>
        internal System.Threading.ManualResetEvent m_DataReady = new System.Threading.ManualResetEvent(false);
        
        /// <summary>
        /// Event logger.
        /// </summary>
        static internal System.Diagnostics.EventLog EventLog;
        /// <summary>
        /// Cleaner method.
        /// </summary>
        protected void CleanResults()
        {
            lock (m_Queue)
            {
                System.DateTime now = System.DateTime.Now;
                int i;
                for (i = 0; i < m_ResultList.Count; i++)
                    if (((DataProcessingResult)m_ResultList[i]).ExpirationTime < now)
                        m_ResultList.RemoveAt(i--);
            }
        }
        /// <summary>
        /// Creates a new data processing server.
        /// </summary>
        /// <param name="evlog">The system event log to write events to.</param>
        public MyDataProcessingServer2(System.Diagnostics.EventLog evlog)
        {
            EventLog = evlog;
            OperaDb.OperaDbConnection conn = new OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
            conn.Open();
            System.Data.DataSet ds = new System.Data.DataSet();
            new OperaDb.OperaDbDataAdapter("SELECT ADDRESS FROM TB_MACHINES WHERE (ISDATAPROCESSINGSERVER = 1 AND ID_SITE = " + MainForm.IdSite.ToString() + ")", conn, null).Fill(ds);
            conn.Close();
            m_ExeThreads = new ExeThread[ds.Tables[0].Rows.Count];
            int i;
            for (i = 0; i < ds.Tables[0].Rows.Count; i++)
                m_ExeThreads[i] = new ExeThread(ds.Tables[0].Rows[i][0].ToString(), m_Queue, m_ExeList, m_DataReady, m_ResultList);
        }

        /// <summary>
        /// Checks whether the machine is willing to accept new requests of batch data processing.
        /// </summary>
        public bool IsWillingToProcess
        {
            get
            {
                foreach (ExeThread x in m_ExeThreads)
                    if (x.MachinePowerClass > 0) return true;
                return false;
            }
        }

        /// <summary>
        /// Gets the number of data processing batches to be executed.
        /// Notice that in case of quick transitions, a subsequent Queue query might return an inconsistent result.
        /// </summary>
        public int QueueLength
        {
            get
            {
                lock (m_Queue)
                    return m_Queue.Count + m_ExeList.Count;
            }
        }

        /// <summary>
        /// Gets the queue of data processing batches to be executed. 
        /// Notice that in case of quick transitions, a subsequent QueueLength query might return an inconsistent result.
        /// </summary>
        public DataProcessingBatchDesc[] Queue
        {
            get
            {
                lock (m_Queue)
                {
                    System.Collections.ArrayList lqueue = new System.Collections.ArrayList();
                    foreach (SySal.DAQSystem.DataProcessingBatchDesc desc in m_ExeList)
                    {
                        SySal.DAQSystem.DataProcessingBatchDesc ddesc = new SySal.DAQSystem.DataProcessingBatchDesc();
                        ddesc.CommandLineArguments = desc.CommandLineArguments;
                        ddesc.Description = desc.Description;
                        ddesc.Enqueued = desc.Enqueued;
                        ddesc.Filename = desc.Filename;
                        ddesc.Finished = desc.Finished;
                        ddesc.Id = desc.Id;
                        ddesc.MachinePowerClass = desc.MachinePowerClass;
                        ddesc.PeakVirtualMemorySize = desc.PeakVirtualMemorySize;
                        ddesc.PeakWorkingSet = desc.PeakWorkingSet;
                        ddesc.Started = desc.Started;
                        ddesc.TotalProcessorTime = desc.TotalProcessorTime;
                        ddesc.Username = desc.Username;
                        lqueue.Add(ddesc);
                    }
                    foreach (SySal.DAQSystem.DataProcessingBatchDesc desc in m_Queue)
                    {
                        SySal.DAQSystem.DataProcessingBatchDesc ddesc = new SySal.DAQSystem.DataProcessingBatchDesc();
                        ddesc.CommandLineArguments = desc.CommandLineArguments;
                        ddesc.Description = desc.Description;
                        ddesc.Enqueued = desc.Enqueued;
                        ddesc.Filename = desc.Filename;
                        ddesc.Finished = desc.Finished;
                        ddesc.Id = desc.Id;
                        ddesc.MachinePowerClass = desc.MachinePowerClass;
                        ddesc.PeakVirtualMemorySize = desc.PeakVirtualMemorySize;
                        ddesc.PeakWorkingSet = desc.PeakWorkingSet;
                        ddesc.Started = desc.Started;
                        ddesc.TotalProcessorTime = desc.TotalProcessorTime;
                        ddesc.Username = desc.Username;
                        lqueue.Add(ddesc);
                    }
                    return (DataProcessingBatchDesc[])lqueue.ToArray(typeof(DataProcessingBatchDesc));
                }
            }
        }

        /// <summary>
        /// The power class of the machine, computed as the highest value of machine power class presently supplied by the DataProcessingServer machines.
        /// </summary>
        public int MachinePowerClass
        {
            get
            {
                int mpc = 0;
                foreach (ExeThread x in m_ExeThreads)
                    mpc = Math.Max(mpc, x.MachinePowerClass);
                return mpc;
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
            lock (m_Queue)
            {
                int i;
                for (i = 0; i < m_Queue.Count + m_ExeList.Count; i++)
                {
                    SySal.DAQSystem.DataProcessingBatchDesc desc = (SySal.DAQSystem.DataProcessingBatchDesc)(i < m_Queue.Count ? m_Queue[i] : m_ExeList[i - m_Queue.Count]);
                    bool OkToRemove = false;
                    if (desc.Id == id)
                    {
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
                            long id_user = 0;
                            SySal.OperaDb.OperaDbConnection conn = null;
                            try
                            {
                                conn = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
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
                        if (OkToRemove == false) throw new Exception("Cannot remove the batch.");
                        else
                        {
                            if (i < m_Queue.Count)
                            {
                                DataProcessingResult dpr = new DataProcessingResult(desc, new Exception("The batch was removed from the queue."), MainForm.ResultLiveTime);
                                dpr.Processed = true;
                                m_ResultList.Add(dpr);
                                m_Queue.RemoveAt(i);
                                return;
                            }
                            else
                                for (i = 0; i < m_ExeThreads.Length; i++)
                                    if (m_ExeThreads[i].Desc != null && m_ExeThreads[i].Desc.Id == desc.Id)
                                    {
                                        m_ExeThreads[i].KillEvent.Set();
                                        return;
                                    }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Checks for execution completion.
        /// </summary>
        /// <param name="id">the id of the batch.</param>
        /// <returns>true if the batch has been completed, false if it is in progress.</returns>
        public bool DoneWith(ulong id)
        {
            lock (m_Queue)
            {
                foreach (DataProcessingResult res in m_ResultList)
                    if (res.Desc.Id == id) return true;
                foreach (DataProcessingBatchDesc desc in m_Queue)
                    if (desc.Id == id) return false;
                foreach (DataProcessingBatchDesc desc in m_ExeList)
                    if (desc.Id == id) return false;
                throw new Exception("Unknown batch " + id.ToString("X16") + ". The batch was never scheduled ot its result might have expired.");
            }
        }


        /// <summary>
        /// Gets the result for a batch.
        /// </summary>
        /// <param name="id">the id of the batch.</param>
        /// <returns>the batch descriptor. It is modified to reflect the batch output. An exception is thrown if the batch terminated with an exception.</returns>
        public DataProcessingBatchDesc Result(ulong id)
        {
            lock (m_Queue)
            {
                foreach (DataProcessingResult res in m_ResultList)
                    if (res.Desc.Id == id)
                    {
                        if (res.X != null) throw res.X;
                        return res.Desc;
                    }
                throw new Exception("Unknown batch " + id.ToString("X16") + ". The batch was never scheduled ot its result might have expired.");
            }
        }

        /// <summary>
        /// Enqueues a batch without waiting for its execution.
        /// </summary>
        /// <param name="desc">the descriptor of the batch. If the batch is rejected because another batch in the queue already has the same id, the Id member is set to 0.</param>
        /// <returns>true if the batch has been accepted, false otherwise.</returns>
        public bool Enqueue(DataProcessingBatchDesc desc)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
                conn.Open();
                SySal.OperaDb.ComputingInfrastructure.UserPermission[] rights = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
                rights[0].DB_Site_Id = MainForm.IdSite;
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
            lock (m_Queue)
            {
                foreach (SySal.DAQSystem.DataProcessingBatchDesc d in m_Queue)
                    if (d.Id == desc.Id)
                    {
                        desc.Id = 0;
                        return false;
                    }
                foreach (SySal.DAQSystem.DataProcessingBatchDesc d in m_ExeList)
                    if (d.Id == desc.Id)
                    {
                        desc.Id = 0;
                        return false;
                    }
                desc.Finished = desc.Started = desc.Enqueued = System.DateTime.Now;
                m_Queue.Add(desc);
                m_DataReady.Set();
            }
            return true;
        }

        /// <summary>
        /// Creates a new DataProcessingServer.
        /// </summary>
        public MyDataProcessingServer2()
        {
            throw new Exception("Implemented only for conformance to DataProcessingServer scheme.");
        }

        /// <summary>
        /// Initializes the Lifetime Service.
        /// </summary>
        /// <returns>the lifetime service object or null.</returns>
        public override object InitializeLifetimeService()
        {
            return null;
        }

        /// <summary>
        /// Tests the communication with the DataProcessingServer.
        /// </summary>
        /// <param name="commpar">communication parameter.</param>
        /// <returns>2 * commpar - 1 if the DataProcessingServer object and the communication are working properly.</returns>
        public int TestComm(int commpar)
        {
            return 2 * commpar - 1;
        }

        private static ulong[] IdHigh = new ulong[1];

        /// <summary>
        /// Provides an Id for a new batch to be enqueued.
        /// Batch Id clashing is a reason for rejection of well-formed batch descriptors.
        /// Use of this property does not completely guarantee that the batch id does not clash with another Id in the queue, because another process could schedule another batch with the same Id.
        /// However, the Ids generated by this property all come from the same sequence and are very likely not to be duplicated within a reasonable amount of time.
        /// </summary>
        public ulong SuggestId
        {
            get
            {
                ulong l = (ulong)System.DateTime.Now.Ticks;
                lock (IdHigh)
                {
                    return (++IdHigh[0] << 32) + (ulong)(l & 0xffffffffL);
                }
            }
        }

        internal void AbortAllBatches()
        {
            foreach (ExeThread x in m_ExeThreads)
            {
                x.XThread.Abort();
                x.XThread.Join();
            }
        }
    }
}
