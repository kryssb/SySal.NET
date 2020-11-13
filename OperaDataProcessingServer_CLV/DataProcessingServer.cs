using System;
using SySal.DAQSystem;
using System.Runtime.Serialization;
using SySal.Services.OperaDataProcessingServer_CLV;

namespace SySal.DAQSystem
{
	/// <summary>
	/// Implementation of the Data Processing Server.
	/// </summary>
	public class MyDataProcessingServer : MarshalByRefObject, SySal.DAQSystem.IDataProcessingServer
	{		
		/// <summary>
		/// A lockable boolean telling whether the DataProcessingServer is willing to process new batches.
		/// </summary>
		protected bool [] m_IsWillingToProcess = new bool[1];
		/// <summary>
		/// Checks whether the machine is willing to accept new requests of batch data processing.
		/// </summary>
		public bool IsWillingToProcess { get { return m_IsWillingToProcess[0]; } }
		/// <summary>
		/// Changes the state of willingness to process.
		/// </summary>
		/// <param name="iswilling">the state the DataProcessingServer must enter.</param>
		internal void SetIsWillingToProcess(bool iswilling)
		{
			m_IsWillingToProcess[0] = iswilling;
		}
		/// <summary>
		/// Tells whether the process is terminating. It's used to abort the execution thread.
		/// </summary>
		protected bool Terminate = false;
		/// <summary>
		/// Execution thread.
		/// </summary>
		protected System.Threading.Thread m_ExecThread;
		/// <summary>
		/// Thread that maintains the result list, cleaning old batches.
		/// </summary>
		protected System.Threading.Thread m_ResultCleanerThread;
		/// <summary>
		/// Time duration of a result in the result list.
		/// </summary>
		protected System.TimeSpan m_ResultLiveTime;
		/// <summary>
		/// Signals that there is a new entry in the processing queue.
		/// </summary>
		protected System.Threading.AutoResetEvent m_QueueNotEmpty;
		/// <summary>
		/// The queue of data processing batches to be executed.
		/// </summary>
		protected System.Collections.ArrayList m_Queue;		
		/// <summary>
		/// The list of batches for which result information is sought.
		/// </summary>
		protected System.Collections.ArrayList m_ResultList;
		/// <summary>
		/// The process in which the current batch is being executed.
		/// </summary>
		protected System.Diagnostics.Process m_ExecProc = null; 
		/// <summary>
		/// Tells whether the process has been terminated by a kill command.
		/// </summary>
		protected bool m_ExecProcKilled = false;
		/// <summary>
		/// Result of execution for a data processing batch.
		/// </summary>
		class DataProcessingResult
		{
			/// <summary>
			/// The batch descriptor that initiated the processing.
			/// </summary>
			public DataProcessingBatchDesc Desc;
			/// <summary>
			/// Possible exception that terminates the processing.
			/// </summary>
			public System.Exception X;
			/// <summary>
			/// Tells whether the batch has been processed.
			/// </summary>
			public bool Processed;
			/// <summary>
			/// Time when this result will expire.
			/// </summary>
			public System.DateTime ExpirationTime;
			/// <summary>
			/// Public constructor.
			/// </summary>
			/// <param name="desc">the batch descriptor that initiates the processing.</param>		
			/// <param name="resultduration">time duration of this result.</param>	
			public DataProcessingResult(DataProcessingBatchDesc desc, System.TimeSpan resultduration)
			{
				Desc = desc;
				X = null;
				Processed = false;
				ExpirationTime = System.DateTime.Now + resultduration;
			}
		}

		/// <summary>
		/// Creates a new data processing server.
		/// </summary>
		/// <param name="evlog">The system event log to write events to.</param>
		public MyDataProcessingServer(System.Diagnostics.EventLog evlog)
		{
			EventLog = evlog;
			m_ResultLiveTime = OperaDataProcessingServer.ResultLiveTime;
			m_IsWillingToProcess[0] = true;
			m_Queue = new System.Collections.ArrayList();
			m_ResultList = new System.Collections.ArrayList();
			m_QueueNotEmpty = new System.Threading.AutoResetEvent(false);
			m_ResultCleanerThread = new System.Threading.Thread(new System.Threading.ThreadStart(ResultCleanerThread));
			m_ResultCleanerThread.Priority = System.Threading.ThreadPriority.BelowNormal;
			m_ResultCleanerThread.Start();
			m_ExecThread = new System.Threading.Thread(new System.Threading.ThreadStart(ExecThread));
			m_ExecThread.Priority = OperaDataProcessingServer.LowPriority ? System.Threading.ThreadPriority.BelowNormal : System.Threading.ThreadPriority.Normal;
			m_ExecThread.Start();
		}

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
		/// The event log to be used to record anomalous behaviours.
		/// </summary>
		internal System.Diagnostics.EventLog EventLog;

		/// <summary>
		/// Gets the queue of data processing batches to be executed. 
		/// Notice that in case of quick transitions, a subsequent QueueLength query might return an inconsistent result.
		/// </summary>
		public DataProcessingBatchDesc [] Queue { get { return (DataProcessingBatchDesc [])m_Queue.ToArray(typeof(DataProcessingBatchDesc)); } }

		/// <summary>
		/// Gets the number of data processing batches to be executed.
		/// Notice that in case of quick transitions, a subsequent Queue query might return an inconsistent result.
		/// </summary>
		public int QueueLength { get { return m_Queue.Count; } }

		/// <summary>
		/// Draws a batch out ouf the queue or aborts it if it is already being executed.
		/// A non-null token or a username/password pair must be supplied that matches the one with which the batch was started.
		/// If the token is supplied, the username/password pair is ignored.
		/// </summary>
		/// <param name="id">identifier of the batch to be removed.</param>
		/// <param name="token">the process token to be used.</param>
		/// <param name="user">username of the user that started the batch. Ignored if <c>token</c> is non-null.</param>
		/// <param name="password">password of the user that started the batch. Ignored if <c>token</c> is non-null.</param>
		public void Remove(ulong id, string token, string user, string password)
		{ 			
			lock(m_Queue)
			{
				int i;
				for (i = 0; i < m_Queue.Count && ((DataProcessingBatchDesc)m_Queue[i]).Id != id; i++);
				if (i == m_Queue.Count) throw new Exception("Batch not present in processing queue.");
				DataProcessingBatchDesc dpb = (DataProcessingBatchDesc)m_Queue[i];
				bool OkToRemove = false;
				if (token != null)
				{
					if (dpb.Token != null)
					{
						if (dpb.Token == token) OkToRemove = true;
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
						conn = new SySal.OperaDb.OperaDbConnection(OperaDataProcessingServer.DBServer, OperaDataProcessingServer.DBUserName, OperaDataProcessingServer.DBPassword);
						conn.Open();
						id_user = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(user, password, conn, null);						
						if (dpb.Token != null)
						{
							try
							{
								SySal.OperaDb.ComputingInfrastructure.User.CheckTokenOwnership(dpb.Token, id_user, null, null, conn, null);
								OkToRemove = true;
							}
							catch (Exception)
							{
								throw new Exception("A user cannot remove a batch started by an operation of another user.");
							}
						}
						else 
						{
							if (String.Compare(dpb.Username, user, true) != 0) 
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
				if (i == 0)
				{
					try
					{
						m_ExecProc.Kill();
						m_ExecProcKilled = true;
					}
					catch (Exception) {}
				}
				else 
				{					
					lock(m_ResultList)
					{
						m_Queue.RemoveAt(i);
						DataProcessingResult dpr = null;
						for (i = 0; i < m_ResultList.Count; i++)
						{
							dpr = (DataProcessingResult)m_ResultList[i];
							if (dpr.Desc.Id == id) return;
						}						
						dpr = new DataProcessingResult(dpb, m_ResultLiveTime);
						dpr.X = new Exception("The batch was removed from the queue.");
						dpr.Processed = true;								
						m_ResultList.Add(dpr);						
					}
				}
			}
		}

		/// <summary>
		/// Gets the power class of the machine.
		/// </summary>
		public int MachinePowerClass { get { return OperaDataProcessingServer.MachinePowerClass; } }

		/// <summary>
		/// Enqueues a batch.
		/// </summary>
		/// <param name="desc">the descriptor of the batch. If the batch is rejected because another batch in the queue already has the same id, the Id member is set to 0.</param>
		/// <returns>true if the batch has been accepted, false otherwise.</returns>
		public bool Enqueue(DataProcessingBatchDesc desc)
		{ 	
			if (m_IsWillingToProcess[0] == false) return false;
			SySal.OperaDb.OperaDbConnection conn = null;
			try
			{
				conn = new SySal.OperaDb.OperaDbConnection(OperaDataProcessingServer.DBServer, OperaDataProcessingServer.DBUserName, OperaDataProcessingServer.DBPassword);
				conn.Open();
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] rights = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
				rights[0].DB_Site_Id = OperaDataProcessingServer.IdSite;
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
			lock(m_Queue)
			{
				foreach (SySal.DAQSystem.DataProcessingBatchDesc d in m_Queue)
					if (d.Id == desc.Id)
					{
						desc.Id = 0;
						return false;
					}
				desc.Finished = desc.Started = desc.Enqueued = System.DateTime.Now;
				m_Queue.Add(desc);
				m_QueueNotEmpty.Set();
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
			lock(m_Queue)
				lock(m_ResultList)
				{
					foreach (DataProcessingBatchDesc desc in m_Queue)
						if (desc.Id == id) return false;
					foreach (DataProcessingResult res in m_ResultList)
						if (res.Desc.Id == id) return true;
					throw new Exception("Unknown batch " + id + ". The batch was never scheduled ot its result might have expired.");
				}
		}

		/// <summary>
		/// Gets the result for a batch.
		/// </summary>
		/// <param name="id">the id of the batch.</param>
		/// <returns>the batch descriptor. It is modified to reflect the batch output. An exception is thrown if the batch terminated with an exception.</returns>
		public DataProcessingBatchDesc Result(ulong id)
		{ 			
			lock(m_ResultList)
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

		private static ulong [] IdHigh = new ulong[1];

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
				lock(IdHigh)
				{
					return (++IdHigh[0] << 32) + (ulong)(l & 0xffffffffL);
				}
			}
		}

		/// <summary>
		/// Creates a new DataProcessingServer.
		/// </summary>
		public MyDataProcessingServer()
		{
			throw new Exception("Implemented only for conformance to DataProcessingServer scheme.");
		}

		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>the lifetime service object or null.</returns>
		public object InitializeLifetimeService()
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

		/// <summary>
		/// Aborts all processing batches and terminates the execution thread.
		/// </summary>
		internal void AbortAllBatches()
		{
			Terminate = true;
			m_ExecThread.Abort();
			m_ResultCleanerThread.Abort();
		}

		/// <summary>
		/// Result cleaner thread method.
		/// </summary>
		protected void ResultCleanerThread()
		{
			while (true)
			{
				lock(m_ResultList)
				{
					System.DateTime now = System.DateTime.Now;
					int i;
					for (i = 0; i < m_ResultList.Count; i++)
						if (((DataProcessingResult)m_ResultList[i]).ExpirationTime < now)
							m_ResultList.RemoveAt(i--);					
				}
				System.Threading.Thread.Sleep(1000);
			}
		}

		/// <summary>
		/// Execution thread method.
		/// </summary>
		protected void ExecThread()
		{
			while (m_QueueNotEmpty.WaitOne())
				while (m_Queue.Count > 0)
				{
					SySal.OperaDb.OperaDbCredentials cred = new SySal.OperaDb.OperaDbCredentials();
					DataProcessingBatchDesc desc = null;
					System.Exception retX = null;
					string retXstr = "";
					m_ExecProc = new System.Diagnostics.Process();
					lock(m_ExecProc)
					{
						m_ExecProcKilled = false;
						lock(m_Queue)
						{
							desc = (DataProcessingBatchDesc)m_Queue[0];
						}
						m_ExecProc.StartInfo.Arguments = desc.CommandLineArguments;
						m_ExecProc.StartInfo.FileName = desc.Filename;
						m_ExecProc.StartInfo.UseShellExecute = false;
						m_ExecProc.StartInfo.RedirectStandardError = true;
						desc.Started = desc.Finished = System.DateTime.Now;
                        try
                        {
                            cred.DBUserName = (desc.AliasUsername == null) ? "" : desc.AliasUsername;
                            cred.DBPassword = (desc.AliasPassword == null) ? "" : desc.AliasPassword;
                            cred.DBServer = OperaDataProcessingServer.DBServer;
                            cred.OPERAUserName = (desc.Username == null) ? "" : desc.Username;
                            cred.OPERAPassword = (desc.Password == null) ? "" : desc.Password;
                            cred.RecordToEnvironment(m_ExecProc.StartInfo.EnvironmentVariables);
                            desc.Started = System.DateTime.Now;
                            m_ExecProc.Start();
                        }
                        catch (Exception x)
                        {
                            retX = new DataProcessingException("Internal error occurred during process start.", x);
                        }
                        try
                        {
                            m_ExecProc.PriorityClass = OperaDataProcessingServer.LowPriority ? System.Diagnostics.ProcessPriorityClass.BelowNormal : System.Diagnostics.ProcessPriorityClass.Normal;
                            //m_ExecProc.MaxWorkingSet = new System.IntPtr(OperaDataProcessingServer.PeakWorkingSetMB * 1048576);							
                        }
                        catch (Exception) { }
                    }
					if (retX == null)
					{
						//do
					{
						try
						{
							m_ExecProc.Refresh();
							retXstr += m_ExecProc.StandardError.ReadToEnd();
							desc.TotalProcessorTime = m_ExecProc.TotalProcessorTime;
							desc.PeakVirtualMemorySize = m_ExecProc.PeakVirtualMemorySize;
							desc.PeakWorkingSet = m_ExecProc.PeakWorkingSet;									
						}
						catch(Exception) {}
					}
						//while (m_ExecProc.WaitForExit(1000) == false);
						desc.Finished = System.DateTime.Now;
						lock(m_ExecProc)
						{/*
							try
							{
								retXstr += m_ExecProc.StandardError.ReadToEnd();
							}
							catch (Exception) {}*/
							if (retXstr == null || retXstr.Length == 0) 
								retX = null;
							else
								retX = new DataProcessingException(retXstr);
							if (m_ExecProcKilled) retX = new Exception("Process has been killed.");							
						}
					}
					else
					{
						try
						{
							retXstr += m_ExecProc.StandardError.ReadToEnd();
						}
						catch (Exception) {}
					}
					m_ExecProc = null;
					SySal.OperaDb.OperaDbConnection conn = null;
					lock(m_ResultList)
						lock(m_Queue)
						{
							DataProcessingResult dpr = new DataProcessingResult(desc, m_ResultLiveTime);
							dpr.Processed = true;
							dpr.X = retX;
							m_ResultList.Add(dpr);
							m_Queue.RemoveAt(0);
						}						
				}
		}
	}
}
