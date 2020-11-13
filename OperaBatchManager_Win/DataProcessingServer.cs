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
	public class MyDataProcessingServer : MarshalByRefObject, SySal.DAQSystem.IDataProcessingServer
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
		/// A batch being executed.
		/// </summary>
		protected class ExeBatch
		{
			/// <summary>
			/// Original description of the batch.
			/// </summary>
			public SySal.DAQSystem.DataProcessingBatchDesc Desc;
			/// <summary>
			/// Description of the batch being executed by the DataProcessingServer.
			/// </summary>
			public SySal.DAQSystem.DataProcessingBatchDesc MappedDesc;
			/// <summary>
			/// DataProcessingServer that is handling the batch.
			/// </summary>
			public SySal.DAQSystem.SyncDataProcessingServerWrapper DPSW;
		}
		/// <summary>
		/// List of the batches being executed.
		/// </summary>
		protected System.Collections.ArrayList m_ExeList = new System.Collections.ArrayList();
		/// <summary>
		/// Thread that controls execution on DataProcessingServers.
		/// </summary>
		protected System.Threading.Thread m_ExeThread;
		/// <summary>
		/// Checks each DataProcessingServer for connection.
		/// </summary>
		protected void ExploreConnections()
		{
			int i;
			for (i = 0; i < m_DPSHandlers.Length; i++)
			{
				if (m_DPSHandlers[i].Srv == null)
				{
					try
					{
						m_DPSHandlers[i].Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + m_DPSHandlers[i].m_Address + ":" + ((int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromSeconds(10));
						int rndpar = DataProcSrvHandler.Rnd.Next();
						if (m_DPSHandlers[i].Srv.TestComm(rndpar) != 2 * rndpar - 1) throw new Exception();
						m_DPSHandlers[i].IsAvailable = true;
						m_DPSHandlers[i].MachinePowerClass = m_DPSHandlers[i].Srv.MachinePowerClass;
					}
					catch (Exception)
					{
						m_DPSHandlers[i].Srv = null;
						m_DPSHandlers[i].IsAvailable = false;
						m_DPSHandlers[i].MachinePowerClass = 0;
					}
				}
				else
				{
					try
					{
						int rndpar = DataProcSrvHandler.Rnd.Next();
						if (m_DPSHandlers[i].Srv.TestComm(rndpar) != 2 * rndpar - 1) throw new Exception();
						m_DPSHandlers[i].IsAvailable = true;
						m_DPSHandlers[i].MachinePowerClass = m_DPSHandlers[i].Srv.MachinePowerClass;
					}
					catch (Exception)
					{
						m_DPSHandlers[i].Srv = null;
						m_DPSHandlers[i].IsAvailable = false;
						m_DPSHandlers[i].MachinePowerClass = 0;
					}
				}
			}
		}
		/// <summary>
		/// Checks the execution status of all batches.
		/// </summary>
		protected void MonitorBatches()
		{
			System.Collections.ArrayList l_ExeList = (System.Collections.ArrayList)m_ExeList.Clone();				
			foreach (ExeBatch exe in l_ExeList)
			{
				try
				{
					if (exe.DPSW.DoneWith(exe.MappedDesc.Id))
					{
						try
						{
							exe.MappedDesc = exe.DPSW.Result(exe.MappedDesc.Id);
							lock(m_Queue)								
							{
								m_ResultList.Add(new DataProcessingResult(exe.Desc, null, MainForm.ResultLiveTime));
								m_Queue.Remove(exe.Desc);
								m_ExeList.Remove(exe);
							}
						}
						catch (SySal.DAQSystem.DataProcessingException retx)
						{
							lock(m_Queue)
							{
								m_ResultList.Add(new DataProcessingResult(exe.Desc, retx, MainForm.ResultLiveTime));
								m_Queue.Remove(exe.Desc);
								m_ExeList.Remove(exe);
							}
						}
						catch (Exception x)
						{
                            m_ExeList.Remove(exe);
                            try
                            {
                                EventLog.WriteEntry("Error handling batch " + exe.Desc.Id.ToString("X16") + "\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Warning);
                            }
                            catch (Exception) { }
						}
					}
				}
				catch (Exception x)
				{
                    m_ExeList.Remove(exe);
                    try
                    {
                        EventLog.WriteEntry("Error handling batch " + exe.Desc.Id.ToString("X16") + "\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Warning);
                    }
                    catch (Exception) { }
				}
			}
		}
		/// <summary>
		/// Schedules batches onto DataProcessingServers.
		/// </summary>
		protected void FeedDataProcessingServers()
		{
			if (m_ExeList.Count < m_DPSHandlers.Length)
			{
				lock(m_Queue)
				{
					if (m_Queue.Count > m_ExeList.Count)
					{
						DataProcSrvHandler [] l_AvDPSHandlers = (DataProcSrvHandler [])m_DPSHandlers.Clone();							
						int i, j;
						for (j = 0; j < m_ExeList.Count; j++)
						{
							for (i = 0; i < l_AvDPSHandlers.Length; i++)
								if (l_AvDPSHandlers[i] != null && l_AvDPSHandlers[i].Srv == ((ExeBatch)m_ExeList[j]).DPSW)
								{
									l_AvDPSHandlers[i] = null;
									break;
								}
						}
						for (i = 0; i < m_Queue.Count; i++)
						{
							SySal.DAQSystem.DataProcessingBatchDesc desc = (SySal.DAQSystem.DataProcessingBatchDesc)m_Queue[i];
							for (j = 0; j < m_ExeList.Count; j++)
								if (((ExeBatch)m_ExeList[j]).Desc == desc) break;
							if (j == m_ExeList.Count)
							{
								for (j = 0; j < l_AvDPSHandlers.Length; j++)
								{
									try
									{
										if (l_AvDPSHandlers[j] != null && l_AvDPSHandlers[j].IsAvailable && l_AvDPSHandlers[j].MachinePowerClass >= desc.MachinePowerClass)
										{
											ExeBatch exe = new ExeBatch();
											exe.Desc = desc;
											exe.MappedDesc = (SySal.DAQSystem.DataProcessingBatchDesc)desc.Clone();
											exe.MappedDesc.Id = l_AvDPSHandlers[j].Srv.SuggestId;
											exe.MappedDesc.Description = exe.Desc.Id.ToString("X16") + " _DPS_REMAP_ " + exe.Desc.Description; 
											if (MainForm.ImpersonateBatchUser == false)
											{
												exe.MappedDesc.Username = MainForm.OPERAUserName;
												exe.MappedDesc.Password = MainForm.OPERAPassword;
												exe.MappedDesc.Token = null;
											}
											exe.DPSW = l_AvDPSHandlers[j].Srv;
											if (exe.DPSW.Enqueue(exe.MappedDesc) == false)
											{
												long ticks = System.DateTime.Now.Ticks;
												if (ticks < 0) ticks = -ticks;
												exe.MappedDesc.Id = (ulong)ticks;
												exe.MappedDesc.Description = exe.MappedDesc.Id.ToString("X16") + " _OWN_REMAP_ " + exe.Desc.Description; 
												if (l_AvDPSHandlers[j].Srv.Enqueue(exe.MappedDesc) == false)
													m_ResultList.Add(new DataProcessingResult(exe.Desc, new Exception("Unknown error!"), MainForm.ResultLiveTime));
												else 
												{
													m_ExeList.Add(exe);
													l_AvDPSHandlers[j] = null;
													break;
												}
											}
											else 
											{
												m_ExeList.Add(exe);
												l_AvDPSHandlers[j] = null;
												break;
											}
										}
									}
									catch (Exception)
									{
										lock(m_DPSHandlers)
										{
											l_AvDPSHandlers[j].Srv = null;
											l_AvDPSHandlers[j].IsAvailable = false;
											l_AvDPSHandlers[j].MachinePowerClass = 0;
										}
									}
								}
							}
						}
					}
				}
			}

		}

		/// <summary>
		/// Execution method.
		/// </summary>
		protected void ExeThread()
		{
			MainForm.ThreadLogStart("ExeThread");
			while (!Terminate)
			{
				try
				{

					ExploreConnections();
					FeedDataProcessingServers();
					MonitorBatches();
					CleanResults();
					System.Threading.Thread.Sleep(1000);
				}
				catch (Exception x)
				{
					System.IO.StreamWriter wr = new System.IO.StreamWriter(MainForm.ScratchDir + @"\operabatchmanager_win.dpxthrderr.txt");
					wr.WriteLine(x);
					wr.Flush();
					wr.Close();
				}
			}
			MainForm.ThreadLogEnd();
		}

		/// <summary>
		/// The list of handlers of remote DataProcessingServers.
		/// </summary>
		protected DataProcSrvHandler [] m_DPSHandlers;

		/// <summary>
		/// Class that handles one remote Data Processing Server.
		/// </summary>
		protected class DataProcSrvHandler
		{
			/// <summary>
			/// Random generator. It's currently used to generate TestComm numbers.
			/// </summary>
			internal static System.Random Rnd = new System.Random();
			/// <summary>
			/// Network address of the DataProcessingServer machine.
			/// </summary>
			internal string m_Address;
			/// <summary>
			/// Tells whether the machine is available.
			/// </summary>
			internal bool IsAvailable = false;
			/// <summary>
			/// Power class of the machine. This number is read upon first connection and subsequent reconnections, and then it's cached.
			/// Quick variation of the DataProcessingServer side could result in an inconsistent value. In this case, it suffices to turn off the DataProcessingServer for a time twice as long as the reconnection interval, so the cache is flushed.
			/// </summary>
			internal int MachinePowerClass = 0;
			/// <summary>
			/// The DataProcessingServer that actually performs the batches.
			/// </summary>
			internal SySal.DAQSystem.SyncDataProcessingServerWrapper Srv = null;
		}
		/// <summary>
		/// Signals that all execution threads must terminate for server shutdown and must not respawn.
		/// </summary>
		protected static bool Terminate = false;
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
		/// The internal queue of batches to be executed.
		/// </summary>
		internal System.Collections.ArrayList m_Queue = new System.Collections.ArrayList();
		/// <summary>
		/// The internal list of completed batches.
		/// </summary>
		internal System.Collections.ArrayList m_ResultList = new System.Collections.ArrayList();
		/// <summary>
		/// Time duration of each result in the result list.
		/// </summary>
		internal System.TimeSpan m_ResultLiveTime;
		/// <summary>
		/// Event logger.
		/// </summary>
		static internal System.Diagnostics.EventLog EventLog;
		/// <summary>
		/// Cleaner method.
		/// </summary>
		protected void CleanResults()
		{
			lock(m_Queue)
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
		public MyDataProcessingServer(System.Diagnostics.EventLog evlog)
		{
			EventLog = evlog;
			m_ResultLiveTime = MainForm.ResultLiveTime;
			m_ExeThread = new System.Threading.Thread(new System.Threading.ThreadStart(ExeThread));
			m_ExeThread.Priority = System.Threading.ThreadPriority.BelowNormal;			
			OperaDb.OperaDbConnection conn = new OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
			conn.Open();
			System.Data.DataSet ds = new System.Data.DataSet();
			OperaDb.OperaDbDataAdapter da = new OperaDb.OperaDbDataAdapter("SELECT ADDRESS FROM TB_MACHINES WHERE (ISDATAPROCESSINGSERVER = 1 AND ID_SITE = " + MainForm.IdSite.ToString() + ")", conn, null);
			da.Fill(ds);
			m_DPSHandlers = new DataProcSrvHandler[ds.Tables[0].Rows.Count];
			int i;
			for (i = 0; i < ds.Tables[0].Rows.Count; i++)
			{
				m_DPSHandlers[i] = new DataProcSrvHandler();
				m_DPSHandlers[i].m_Address = ds.Tables[0].Rows[i][0].ToString();
				m_DPSHandlers[i].IsAvailable = false;
				m_DPSHandlers[i].Srv = null;
			}
			conn.Close();
			m_ExeThread.Start();
		}

		/// <summary>
		/// Restarts the execution thread if it is stopped.
		/// </summary>
		internal void RestartExeThread()
		{
			try
			{
				if (m_ExeThread == null || m_ExeThread.IsAlive == false || (m_ExeThread.ThreadState != System.Threading.ThreadState.WaitSleepJoin && m_ExeThread.ThreadState != System.Threading.ThreadState.Running))
				{
					m_ExeThread = new System.Threading.Thread(new System.Threading.ThreadStart(ExeThread));
					m_ExeThread.Priority = System.Threading.ThreadPriority.BelowNormal;			
					m_ExeThread.Start();
				}
			}
			catch(Exception x)
			{
				System.Windows.Forms.MessageBox.Show(x.Message, "Error restarting execution thread");
			}
		}

		/// <summary>
		/// Checks whether the machine is willing to accept new requests of batch data processing.
		/// </summary>
		public bool IsWillingToProcess 
		{ 
			get 
			{ 
				foreach (DataProcSrvHandler h in m_DPSHandlers)
					if (!h.IsAvailable) return false;
				return true;
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
				return m_Queue.Count;
			}
		}

		/// <summary>
		/// Gets the queue of data processing batches to be executed. 
		/// Notice that in case of quick transitions, a subsequent QueueLength query might return an inconsistent result.
		/// </summary>
		public DataProcessingBatchDesc [] Queue 
		{ 
			get 
			{ 
				lock(m_Queue)
				{
					System.Collections.ArrayList lqueue = new System.Collections.ArrayList();
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
					return (DataProcessingBatchDesc [])lqueue.ToArray(typeof(DataProcessingBatchDesc)); 
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
				foreach (DataProcSrvHandler h in m_DPSHandlers)
					if (h.IsAvailable && h.MachinePowerClass > mpc) mpc = h.MachinePowerClass;
				return mpc;
			}
		}

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
				for (i = 0; i < m_Queue.Count; i++)
				{
					SySal.DAQSystem.DataProcessingBatchDesc desc = (SySal.DAQSystem.DataProcessingBatchDesc)m_Queue[i];
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
							DataProcessingResult dpr = new DataProcessingResult(desc, new Exception("The batch was removed from the queue."), m_ResultLiveTime);
							dpr.Processed = true;
							m_ResultList.Add(dpr);
							m_Queue.RemoveAt(i);
							for (i = 0; i < m_ExeList.Count; i++)
								if (((ExeBatch)m_ExeList[i]).Desc.Id == id)
								{
									ExeBatch exe = (ExeBatch)m_ExeList[i];
									try
									{
										exe.DPSW.Remove(exe.MappedDesc.Id, exe.MappedDesc.Token, exe.MappedDesc.Username, exe.MappedDesc.Password);
									}
									catch (Exception) {}
									m_ExeList.RemoveAt(i);
								}
							return;
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
			lock(m_Queue)		
			{
				foreach (DataProcessingResult res in m_ResultList)
					if (res.Desc.Id == id) return true;
				foreach (DataProcessingBatchDesc desc in m_Queue)
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
			lock(m_Queue)
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
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] rights = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
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
			}
			return true;
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

		internal void AbortAllBatches()
		{
			Terminate = true;
			m_ExeThread.Join();
		}
	}
}
