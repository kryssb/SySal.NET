using System;
using SySal.BasicTypes;
using SySal.DAQSystem;
using SySal.DAQSystem.Scanning;
using SySal.DAQSystem.Drivers;
using System.Runtime.Serialization;
using System.Xml.Serialization;
using SySal.Services.OperaBatchManager_Win;

namespace SySal.DAQSystem
{
	/// <summary>
	/// IProcessEventNotifier defines methods for process monitoring.
	/// </summary>
	public interface IProcessEventNotifier
	{
		/// <summary>
		/// Signals process startup.
		/// </summary>
		/// <param name="h">the HostEnv that hosts the process operation.</param>
		/// <param name="description">description of the process operation driver.</param>
		/// <param name="machinename">the name of the machine on which the process operation runs.</param>
		/// <param name="notes">notes to the process operation.</param>
		void ProcessStart(BatchManager.HostEnv h, string description, string machinename, string notes);
		/// <summary>
		/// Signals process termination.
		/// </summary>
		/// <param name="id">the id of the process operation that is terminated.</param>
		void ProcessEnd(long id);
	}

	/// <summary>
	/// Batch manager implementation for OperaBatchManager_Win.
	/// </summary>
	public class BatchManager : MarshalByRefObject
	{		
		internal System.Timers.Timer CleanupTimer = null;

		internal static void CleanupHook(object source, System.Timers.ElapsedEventArgs e)
		{            
            TheInstance.CleanupTimer.Stop();
			TheInstance.Cleanup();
            TheInstance.CleanupTimer.Start();
		}

		internal void Cleanup()
		{
			lock(DBConn)
			{
                try
                {
                    DBConn.Open();
                    new SySal.OperaDb.OperaDbCommand("CALL LP_CLEAN_ORPHAN_TOKENS()", DBConn, null).ExecuteNonQuery();
                }
                catch (Exception x)
                {
                    this.m_EventLog.WriteEntry("Error during cleanup!\r\n" + x.Message, System.Diagnostics.EventLogEntryType.Error);
                }
                finally
                {
                    try
                    {
                        DBConn.Close();
                    }
                    catch (Exception) { };
                }
			}
		}

		internal static BatchManager TheInstance = null;
		/// <summary>
		/// The event logger.
		/// </summary>
		private System.Diagnostics.EventLog m_EventLog;

		private IProcessEventNotifier m_ProcessEventNotifier;		

		/// <summary>
		/// Called when a process in the task list exits.
		/// </summary>
		/// <param name="h">the HostEnv of the process operation that completed.</param>
		private void OnDriverExit(HostEnv h)
		{
			h.WriteProgress();
			h.WriteInterruptQueue();
			lock(DBConn)
			{
				int i;
				for (i = 0; i < m_TaskList.Count; i++)
				{
					if (((HostEnv)m_TaskList[i]) == h)
					{						
						bool success = false;
                        if (h.m_ProgressInfo != null && h.m_ProgressInfo.Complete == true)
                        {
                            if (h.m_ProgressInfo.ExitException == null || h.m_ProgressInfo.ExitException.Length == 0) success = true;
                            else
                            {
                                success = false;
                                m_EventLog.WriteEntry("Process operation #" + h.m_StartupInfo.ProcessOperationId + " suspended on error:\r\n\r\n" + h.m_ProgressInfo.ExitException, System.Diagnostics.EventLogEntryType.Error);
                            }
                            try
                            {
                                lock (DBConn)
                                {
                                    try
                                    {
                                        DBConn.Open();
                                        SySal.OperaDb.ComputingInfrastructure.ProcessOperation.EndTokenized(h.m_StartupInfo.ProcessOperationId, success, DBConn, null);
                                        try
                                        {
                                            m_TaskList.RemoveAt(i);
                                            h.DeleteFiles();
                                            h.CompletionEvent.Set();
                                        }
                                        catch (Exception) { }
                                    }
                                    finally
                                    {
                                        DBConn.Close();
                                    }
                                }
                                new System.Threading.Thread(new ProcessEndNotify(m_ProcessEventNotifier, h.m_StartupInfo.ProcessOperationId).Exec).Start();
                                //h.CF.Close();
                                return;
                            }
                            catch (Exception x)
                            {
                                m_EventLog.WriteEntry("Error in process removal!\r\nProcess Id = " + h.m_StartupInfo.ProcessOperationId.ToString() + "\r\n\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
                            }

                        }				
						return;
					}
				}
				h.DeleteFiles();
				//h.CF.Close();
			}
		}

		/// <summary>
		/// Retrieves a ScanServer that is not used by any HostEnv. This actually realizes an implicit lock.
		/// An exception is thrown if the ScanServer is locked or the machine is not a ScanServer.
		/// </summary>
		/// <param name="machineid">the id of the machine that should be used as ScanServer</param>
		/// <returns>the ScanServer for the specified machine.</returns>
		internal SySal.DAQSystem.ScanServer LockedScanSrv(long machineid)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
					if (h.m_StartupInfo.MachineId == machineid && h.m_ScanSrv != null) 
						throw new Exception("The requested ScanServer is already locked by process operation " + h.m_StartupInfo.ProcessOperationId);			
				lock(DBConn)
				{
					ScanServer srv = null;
                    string addr = null;
					try
					{
						DBConn.Open();
                        addr = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_MACHINES PK_MACHINES) */ ADDRESS FROM TB_MACHINES WHERE ID = " + machineid + " AND ISSCANNINGSERVER = 1", DBConn, null).ExecuteScalar().ToString();
					}									
					finally
					{
						DBConn.Close();
					}
                    srv = (ScanServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(ScanServer), "tcp://" + addr + ":" + ((int)SySal.DAQSystem.OperaPort.ScanServer).ToString() + "/ScanServer.rem");
					return srv;
				}
			}
		}

        internal void UnlockScanSrv(HostEnv h)
        {
            lock (DBConn)
                h.m_ScanSrv = null;
        }

		/// <summary>
		/// Host environment for a process driver.
		/// </summary>
		public class HostEnv : SySal.DAQSystem.Drivers.HostEnv
		{
			/// <summary>
			/// Initializes the Lifetime Service.
			/// </summary>
			/// <returns>null to obtain an everlasting HostEnv.</returns>
			public override object InitializeLifetimeService()
			{
				return null;	
			}
			/// <summary>
			/// Thread that routes interrupt notifications.
			/// </summary>
			internal System.Threading.Thread InterruptNotificationThread;

			/// <summary>
			/// Event that signals a stop.
			/// </summary>
			internal System.Threading.ManualResetEvent StopEvent = new System.Threading.ManualResetEvent(false);

			/// <summary>
			/// Thread that handles GUI windows.
			/// </summary>
			internal System.Threading.Thread CF_thread;

			bool ThreadsShouldStop = false;

			private void InterruptNotificationThreadStart()
			{
				MainForm.ThreadLogStart("InterruptNotificationThreadStart");
                try
                {
                    while (!ThreadsShouldStop)
                    {
                        try
                        {
                            System.Threading.Thread.Sleep(System.Threading.Timeout.Infinite);
                        }
                        catch (System.Threading.ThreadInterruptedException) { }
                        while (ThreadsShouldStop == false && m_InterruptNotifier != null && m_Interrupts.Count > 0)
                            try
                            {
                                m_InterruptNotifier.NotifyInterrupt((SySal.DAQSystem.Drivers.Interrupt)m_Interrupts.Peek());
                            }
/*
                            catch (System.Threading.ThreadAbortException)
                            {
                                //System.Threading.Thread.ResetAbort();
                                MainForm.ThreadLogEnd();
                                return;
                            }
*/
                            catch (Exception x)
                            {
                                try
                                {
                                    TheInstance.m_EventLog.WriteEntry("Failed to route interrupt to process " + this.m_StartupInfo.ProcessOperationId + " because:\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Warning);
                                    m_Interrupts.Dequeue();
                                }
                                catch (Exception) { }
                            }
                    }
                }
                finally
                {
                    MainForm.ThreadLogEnd();
                }
			}
			/// <summary>
			/// Completion status of the process.
			/// </summary>
			internal SySal.DAQSystem.Drivers.Status FinalStatus = SySal.DAQSystem.Drivers.Status.Unknown;
			/// <summary>
			/// Event that marks completion of the task.
			/// </summary>
			internal System.Threading.ManualResetEvent CompletionEvent = new System.Threading.ManualResetEvent(false);
            /// <summary>
            /// Event that signals freeing of microscope.
            /// </summary>
            internal System.Threading.ManualResetEvent ScanServerFreeEvent = new System.Threading.ManualResetEvent(false);
            /// <summary>
			/// The console associated to the process operation.
			/// </summary>
			internal ConsoleForm CF;
			/// <summary>
			/// The domain in which the process operation runs.
			/// </summary>
			internal System.AppDomain Domain;
			/// <summary>
			/// The HostEnv of the parent process operation, if any.
			/// </summary>
			internal HostEnv m_Parent;
			/// <summary>
			/// The token associated to the process operation.
			/// </summary>
			internal string m_Token;
			/// <summary>
			/// Gets the Computing Infrastructure security token associated to this process operation.
			/// </summary>			
			public override string Token { get { return (string)m_Token.Clone(); } }

			internal string m_Exe;			

			internal void Execute()
			{
                try
                {
                    MainForm.ThreadLogStart("Execute");
                    StopEvent.Reset();
                    lock (m_StartupInfo)
                    {
                        m_ProgressInfo.ExitException = "";
                        if (InterruptNotificationThread == null)
                        {
                            ThreadsShouldStop = false;
                            if (Domain == null)
                            {
                                AppDomainSetup asu = new AppDomainSetup();
                                asu = AppDomain.CurrentDomain.SetupInformation;
                                asu.ShadowCopyFiles = "true";
                                asu.ShadowCopyDirectories = MainForm.DriverDir;
                                Domain = AppDomain.CreateDomain("Op " + m_StartupInfo.ProcessOperationId, AppDomain.CurrentDomain.Evidence, asu);
                                Domain.SetData("HostEnv", this);
                                if (m_Parent != null) m_Parent.m_FastResult = null;
                            }
                            InterruptNotificationThread = new System.Threading.Thread(new System.Threading.ThreadStart(InterruptNotificationThreadStart));
                            InterruptNotificationThread.Start();
                        }
                        if (CF_thread == null)
                        {
                            CF_thread = new System.Threading.Thread(new System.Threading.ThreadStart(CF.ShowDlg));
                            CF_thread.Start();
                        }
                    }
                    try
                    {
                        Domain.ExecuteAssembly(m_Exe);
                    }
                    catch (Exception x)
                    {
                        if (x is System.Threading.ThreadAbortException) System.Threading.Thread.ResetAbort();
                        if (m_Parent != null) m_Parent.m_FastResult = null;
                        m_ProgressInfo.ExitException = x.Message;
                        WriteProgress();
                    }
                    lock (m_StartupInfo)
                    {
                        if (Domain != null)
                            try
                            {
                                AppDomain.Unload(Domain);
                            }
                            catch (Exception) { };
                        m_ScanSrv = null;
                        m_InterruptNotifier = null;
                        Domain = null;
                        ThreadsShouldStop = true;
                        InterruptNotificationThread.Interrupt();
                        InterruptNotificationThread.Join();
                        InterruptNotificationThread = null;
                        //CF.Visible = false;
                        CF.SetVisible(false);
                        if (m_ProgressInfo.Complete == true)
                        {
                            try
                            {
                                CF.Close();
                                CF.Dispose();
                                CF = null;
                                CF_thread.Join();
                                CF_thread = null;
                            }
                            catch (Exception) { }
                        }
                    }
                    try
                    {
                        BatchManager.TheInstance.OnDriverExit(this);
                    }
                    catch (Exception) { }
                    MainForm.ThreadLogEnd();
                }
                catch (Exception) { }
                finally
                {
                    m_ExecuteThread = null;
                    Domain = null;
                }
			}

			internal void Stop()
			{
				lock(m_StartupInfo)
				{
					try
					{                        					
                        try
                        {
                            StopEvent.Set();                                                        
                        }
                        catch (Exception) {};
                        if (m_ExecuteThread != null) m_ExecuteThread.Abort();
					}
					catch (Exception) {}
                    try
                    {
                        AppDomain.Unload(Domain);
                    }
                    catch (Exception) { };                    
                    Domain = null;
					if (m_Parent != null) m_Parent.m_FastResult = null;
				}
			}
						
			internal void Run()
			{				
				lock(m_StartupInfo)
                    try
                    {
                        if (Domain == null)
                            (m_ExecuteThread = new System.Threading.Thread(new System.Threading.ThreadStart(Execute))).Start();
                    }
                    catch (Exception)
                    {                        
                        if (Domain != null)
                        {
                            try
                            {
                                AppDomain.Unload(Domain);
                            }
                            catch (Exception)
                            {
                                Domain = null;
                            }
                        }
                        if (m_ExecuteThread != null)
                            try
                            {
                                m_ExecuteThread.Abort();
                            }
                            catch (Exception)
                            {
                                m_ExecuteThread = null;
                            }
                    }
			}

            private System.Threading.Thread m_ExecuteThread = null;

			/// <summary>
			/// Cached information about task startup.
			/// </summary>
			internal SySal.DAQSystem.Drivers.TaskStartupInfo m_StartupInfo;
			/// <summary>
			/// Startup file.
			/// </summary>
			string m_StartupFile;
			/// <summary>
			/// Cached information about task progress.
			/// </summary>
			internal SySal.DAQSystem.Drivers.TaskProgressInfo m_ProgressInfo;

			internal HostEnv(SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo, string startupfile, string exe, string programsettings, HostEnv parent, object fastinput, ConsoleForm cf, string token)
			{
				m_StartupFile = startupfile;
				m_StartupInfo = startupinfo;
                m_StartupInfo.DBServers = MainForm.DBServer;
                m_StartupInfo.DBUserName = MainForm.DBUserName;
                m_StartupInfo.DBPassword = MainForm.DBPassword;
				m_ProgramSettings = programsettings;				
				m_Exe = exe;
				m_Token = token;
				CF = cf;				
				CF.Text = "Operation " + m_StartupInfo.ProcessOperationId + " - " + exe.Remove(0, exe.LastIndexOf('\\'));
				m_FastInput = fastinput;
				m_Parent = parent;
				if (m_StartupInfo.RecoverFromProgressFile)
				{
					LoadProgress();
					LoadInterruptQueue();
				}
				else
				{
					WriteStartupFile();
					ResetProgressInfo();
					WriteProgress();
					WriteInterruptQueue();
				}
				if (m_Parent != null) m_Parent.m_FastResult = null;
				Domain = null;
				InterruptNotificationThread = null;
				CF_thread = null;
			}

			/// <summary>
			/// Destroys the HostEnv after stopping the associated threads.
			/// </summary>
			~HostEnv()
			{
				ThreadsShouldStop = true;
				if (InterruptNotificationThread != null)
				{
					InterruptNotificationThread.Interrupt();
					InterruptNotificationThread.Join();
					InterruptNotificationThread = null;
				}
                if (CF_thread != null)
                {
                    try
                    {
                        if (CF != null)
                        {
                            CF.Close();
                            CF.Dispose();
                            CF = null;
                        }
                        CF_thread.Join();
                        CF_thread = null;
                    }
                    catch (Exception) { }
                }
            }

			static private System.Xml.Serialization.XmlSerializer ProgressXmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.TaskProgressInfo));

			static private System.Xml.Serialization.XmlSerializer InterruptXmls = new System.Xml.Serialization.XmlSerializer(typeof(Interrupt []));

			/// <summary>
			/// Loads the progress information for this task.
			/// </summary>
			public void LoadProgress()
			{				
				System.IO.StreamReader r = null;
				try
				{
					r = new System.IO.StreamReader(m_StartupInfo.ProgressFile);
					string progress = r.ReadToEnd();
					r.Close();					
					m_ProgressInfo = (SySal.DAQSystem.Drivers.TaskProgressInfo)ProgressXmls.Deserialize(new System.IO.StringReader(progress));
				}
				catch(Exception)
				{
					if (r != null) r.Close();
					r = null;
					try
					{
						r = new System.IO.StreamReader(m_StartupInfo.ProgressFile + "_backup");
						string progress = r.ReadToEnd();
						r.Close();
						m_ProgressInfo = (SySal.DAQSystem.Drivers.TaskProgressInfo)ProgressXmls.Deserialize(new System.IO.StringReader(progress));
					}
					catch (Exception)
					{
						if (r != null) r.Close();
						m_ProgressInfo = null;
					}
				}
				if (m_ProgressInfo.LastProcessedInterruptId > 0)
				{
					if (m_Interrupts.Count > 0 && ((Interrupt)(m_Interrupts.Peek())).Id == m_ProgressInfo.LastProcessedInterruptId)
						m_Interrupts.Dequeue();
				}
			}

			/// <summary>
			/// Loads the interrupt queue.
			/// </summary>
			public void LoadInterruptQueue()
			{
				System.IO.StreamReader r = null;
				Interrupt [] intlist = null;
				try
				{
					r = new System.IO.StreamReader(m_StartupInfo.ProgressFile + "_interrupts");
					string intliststr = r.ReadToEnd();
					r.Close();					
					intlist = (Interrupt [])InterruptXmls.Deserialize(new System.IO.StringReader(intliststr));
					m_Interrupts.Clear();
					foreach (Interrupt ie in intlist)
						m_Interrupts.Enqueue(ie);
					if (intlist.Length == 0) NextInterruptId = 1;
					else
					{
						NextInterruptId = 1 + intlist[intlist.Length - 1].Id;
						if (NextInterruptId <= 0) NextInterruptId = 1;
					}
				}
				catch(Exception)
				{
					if (r != null) r.Close();
					m_Interrupts.Clear();
					NextInterruptId = 1;
				}
			}

			/// <summary>
			/// Writes progress info to the progress file.
			/// </summary>
			public void WriteProgress()
			{				
				try
				{
					System.IO.StringWriter strw = new System.IO.StringWriter();
					ProgressXmls.Serialize(strw, m_ProgressInfo);
					strw.Flush();
					System.IO.StreamWriter w = new System.IO.StreamWriter(m_StartupInfo.ProgressFile);
					w.Write(strw.ToString());
					w.Flush();
					w.Close();
					System.IO.File.Copy(m_StartupInfo.ProgressFile, m_StartupInfo.ProgressFile + "_backup", true);
				}
				catch (Exception) {}
			}

			/// <summary>
			/// Writes interrupt queue to the interrupt file.
			/// </summary>
			public void WriteInterruptQueue()
			{
				System.IO.StreamWriter w = null;
				object [] objlist = m_Interrupts.ToArray();
				Interrupt [] intlist = new Interrupt[objlist.Length];
				int i;
				for (i = 0; i < objlist.Length; i++)
					intlist[i] = (Interrupt)(objlist[i]);
				try
				{					
					w = new System.IO.StreamWriter(m_StartupInfo.ProgressFile + "_interrupts");
					InterruptXmls.Serialize(w, intlist);
					w.Flush();
					w.Close();
					w = null;
				}
				catch(Exception)
				{
					if (w != null) w.Close();
				}
			}

			/// <summary>
			/// Writes startup info to the startup file.
			/// </summary>
			public void WriteStartupFile()
			{
				System.Xml.Serialization.XmlSerializer sxmls = new System.Xml.Serialization.XmlSerializer(m_StartupInfo.GetType());
				System.IO.StreamWriter w = new System.IO.StreamWriter(m_StartupFile);
				sxmls.Serialize(w, m_StartupInfo);
				w.Flush();
				w.Close();
			}

			/// <summary>
			/// Deletes process files.
			/// </summary>
			public void DeleteFiles()
			{
				try
				{
					if (MainForm.ArchivedTaskDir.Length > 0) System.IO.File.Move(m_StartupFile, MainForm.ArchivedTaskDir + "\\" + m_StartupFile.Remove(0, m_StartupFile.LastIndexOf('\\') + 1));
					else System.IO.File.Delete(m_StartupFile);
				}
				catch (Exception) 
				{
					try
					{
						System.IO.File.Delete(m_StartupFile);
					}
					catch (Exception) {}
				}
				try
				{
					if (MainForm.ArchivedTaskDir.Length > 0) System.IO.File.Move(m_StartupInfo.ProgressFile, MainForm.ArchivedTaskDir + "\\" + m_StartupInfo.ProgressFile.Remove(0, m_StartupInfo.ProgressFile.LastIndexOf('\\') + 1));
					else System.IO.File.Delete(m_StartupInfo.ProgressFile);
				}
				catch (Exception) 
				{
					try
					{
						System.IO.File.Delete(m_StartupInfo.ProgressFile);
					}
					catch (Exception) {}
				}
				try
				{
					if (MainForm.ArchivedTaskDir.Length > 0) System.IO.File.Move(m_StartupInfo.ProgressFile + "_backup", MainForm.ArchivedTaskDir + "\\" + m_StartupInfo.ProgressFile.Remove(0, m_StartupInfo.ProgressFile.LastIndexOf('\\') + 1) + "_backup");
					else System.IO.File.Delete(m_StartupInfo.ProgressFile + "_backup");
				}
				catch (Exception)
				{
					try
					{
						System.IO.File.Delete(m_StartupInfo.ProgressFile + "_backup");
					}
					catch (Exception) {}
				}
				try
				{
					if (MainForm.ArchivedTaskDir.Length > 0) System.IO.File.Move(m_StartupInfo.ProgressFile + "_interrupts", MainForm.ArchivedTaskDir + "\\" + m_StartupInfo.ProgressFile.Remove(0, m_StartupInfo.ProgressFile.LastIndexOf('\\') + 1) + "_interrupts");
					else System.IO.File.Delete(m_StartupInfo.ProgressFile + "_interrupts");
				}
				catch (Exception) 
				{
					try
					{
						System.IO.File.Delete(m_StartupInfo.ProgressFile + "_interrupts");
					}
					catch (Exception) {}				
				}
			}

			internal int NextInterruptId = 1;

			internal System.Collections.Queue m_Interrupts = new System.Collections.Queue();

			SySal.DAQSystem.Drivers.IInterruptNotifier m_InterruptNotifier = null;

			internal void QueueInterrupt(string intdata)
			{
				lock(m_StartupInfo)
				{
					Interrupt newint = new Interrupt();
					newint.Id = NextInterruptId++;
					newint.Data = intdata;
					if (NextInterruptId == 0) NextInterruptId = 1;
					m_Interrupts.Enqueue(newint);
					WriteInterruptQueue();
					InterruptNotificationThread.Interrupt();
				}
			}

			#region BatchManager functions
			/// <summary>
			/// The machine ids handled by this BatchManager.
			/// </summary>
			public override long [] Machines  {  get { return TheInstance.Machines; } }
			/// <summary>
			/// The ids of the process operations currently handled by this BatchManager.
			/// </summary>
			public override long [] Operations { get { return TheInstance.Operations; } }
			/// <summary>
			/// Retrieves the startup information (except password and alias credentials) for the specified process operation.
			/// </summary>
			/// <param name="id">id of the process operation for which startup information is required.</param>
			/// <returns>the startup information of the process operation.</returns>
			public override SySal.DAQSystem.Drivers.TaskStartupInfo GetOperationStartupInfo(long id) { return TheInstance.GetOperationStartupInfo(id); }
			/// <summary>
			/// Retrieves the progress information for the specified process operation.
			/// </summary>
			/// <param name="id">id of the process operation for which progress information is required.</param>
			/// <returns>the progress information of the process operation.</returns>
			public override SySal.DAQSystem.Drivers.TaskProgressInfo GetProgressInfo(long id) { return TheInstance.GetProgressInfo(id); }
			/// <summary>
			/// Starts a new process operation, which will automatically be a child operation of the current one.
			/// </summary>
			/// <param name="startupinfo">startup information for the process operation.</param>		
			/// <returns>the process operation id that has been allocated to this process operation.</returns>
			public override long Start(SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo)
			{
				startupinfo.DBServers = m_StartupInfo.DBServers;
				startupinfo.DBUserName = m_StartupInfo.DBUserName;
				startupinfo.DBPassword = m_StartupInfo.DBPassword;
				startupinfo.OPERAUsername = m_StartupInfo.OPERAUsername;				
				return TheInstance.Start(m_StartupInfo.ProcessOperationId, startupinfo);				
			}
			/// <summary>
			/// Starts a new process operation, which will automatically be a child operation of the current one, adding fast input. Prepared input may be used to avoid querying the DB, but the callee retains complete responsibility about the correctness of the prepared input.
			/// </summary>
			/// <param name="startupinfo">startup information for the process operation.</param>
			/// <param name="fastinput">the fast input for the process operation. Correctness and consistency of this input cannot be guaranteed.</param>			
			/// <returns>the process operation id that has been allocated to this process operation.</returns>
			public override long Start(SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo, object fastinput)
			{
				startupinfo.DBServers = m_StartupInfo.DBServers;
				startupinfo.DBUserName = m_StartupInfo.DBUserName;
				startupinfo.DBPassword = m_StartupInfo.DBPassword;
				startupinfo.OPERAUsername = m_StartupInfo.OPERAUsername;
				return TheInstance.Start(m_StartupInfo.ProcessOperationId, startupinfo, fastinput, this);
			}

			/// <summary>
			/// Stops execution of the current driver until the specified process operation returns.
			/// </summary>
			/// <param name="procopid">the Id of the process operation whose completion is being awaited.</param>
			/// <returns>the status of the operation after completion.</returns>
			public override SySal.DAQSystem.Drivers.Status Wait(long procopid)
			{
				HostEnv hw = null;
				lock(TheInstance.DBConn)
				{					
					foreach (HostEnv h in TheInstance.m_TaskList)
						if (h.m_StartupInfo.ProcessOperationId == procopid)
						{
							hw = h;
							break;
						}
				}
				if (hw != null)
				{
                    try
                    {                      
                        if (System.Threading.WaitHandle.WaitAny(new System.Threading.WaitHandle[2] { hw.CompletionEvent, StopEvent }) == 1) throw new Exception("Driver is being stopped.");
                        //hw.CompletionEvent.WaitOne();
                        return hw.FinalStatus;
                    }
                    catch (Exception)
                    {
                        return TheInstance.GetStatus(procopid);
                    }        
				}
				return TheInstance.GetStatus(procopid);
			}

            /// <summary>
            /// Stops execution of the current driver until the specified process operation returns or frees its scan server.
            /// </summary>
            /// <param name="procopid">the Id of the process operation whose completion is being awaited.</param>
            /// <returns>the status of the operation after completion.</returns>
            public override SySal.DAQSystem.Drivers.Status WaitForOpOrScanServer(long procopid)
            {
                HostEnv hw = null;
                lock (TheInstance.DBConn)
                {
                    foreach (HostEnv h in TheInstance.m_TaskList)
                        if (h.m_StartupInfo.ProcessOperationId == procopid)
                        {
                            hw = h;
                            break;
                        }
                }
                if (hw != null)
                {
                    try
                    {
                        if (System.Threading.WaitHandle.WaitAny(new System.Threading.WaitHandle[3] { hw.CompletionEvent, hw.ScanServerFreeEvent, StopEvent }) == 2) throw new Exception("Driver is being stopped.");
                        //hw.CompletionEvent.WaitOne();
                        return hw.FinalStatus;
                    }
                    catch (Exception)
                    {
                        return TheInstance.GetStatus(procopid);
                    }
                }
                return TheInstance.GetStatus(procopid);
            }

            /// <summary>
			/// Pauses a process operation using the credentials of the current process operation.
			/// </summary>
			/// <param name="id">the id of the process operation to be paused.</param>
			/// <returns>the status of the process operation.</returns>
			public override SySal.DAQSystem.Drivers.Status Pause(long id)
			{
				return TheInstance.Pause(id, Token);
			}
			/// <summary>
			/// Resumes a paused process operation using the credentials of the current process operation..
			/// </summary>
			/// <param name="id">the id of the process operation to be resumed.</param>
			/// <returns>the status of the process operation.</returns>
			public override SySal.DAQSystem.Drivers.Status Resume(long id)
			{
				return TheInstance.Resume(id, Token);
			}
			/// <summary>
			/// Aborts a process operation using the credentials of the current process operation..
			/// </summary>
			/// <param name="id">the id of the process operation to be aborted.</param>
			/// <returns>the status of the process operation.</returns>
			public override SySal.DAQSystem.Drivers.Status Abort(long id)
			{
				return TheInstance.Abort(id, Token);
			}

			/// <summary>
			/// Retrieves the status of the specified process operation.
			/// </summary>
			/// <param name="id">the id of the process operation for which execution information is required.</param>
			public override SySal.DAQSystem.Drivers.Status GetStatus(long id)
			{
				return TheInstance.GetStatus(id);
			}
			/// <summary>
			/// Generates a summary of the specified process operation.
			/// </summary>
			/// <param name="id">the id of the process operation for which the summary is required.</param>
			public override SySal.DAQSystem.Drivers.BatchSummary GetSummary(long id)
			{
				return TheInstance.GetSummary(id);
			}
			/// <summary>
			/// Adds an interrupt to the interrupt list of the process using the credentials of the current process operation. Interrupt data can be passed.
			/// </summary>
			/// <param name="id">the id of the process to be interrupted.</param>
			/// <param name="interruptdata">interrupt data to be passed to the process; their format and content depend on the specific executable driving the process.</param>
			public override void Interrupt(long id, string interruptdata)
			{
				TheInstance.Interrupt(id, interruptdata, Token);
			}
			#endregion

			#region Assistance to Driver
			/// <summary>
			/// Startup information for the process.
			/// </summary>
			public override SySal.DAQSystem.Drivers.TaskStartupInfo StartupInfo 
			{ 
				get 
				{ 
					return (TaskStartupInfo)(m_StartupInfo.Clone()); 
				} 
			}

			object m_FastInput;
			/// <summary>
			/// Reads the prepared input for this process operation. Consistency with the general logic (in particular, with the OperaDB) is not guaranteed: it depends on the caller, but the responsibility to accept the prepared input depends on the callee.
			/// </summary>
			public override object FastInput { get { return m_FastInput; } }

			/// <summary>
			/// Sets the prepared output for this process operation. Consistency with the general logic (in particular, with the OperaDB) is not guaranteed: it depends on the callee, but the responsibility to accept the prepared output depends on the caller.
			/// </summary>
			public override object FastOutput { set { if (m_Parent != null) m_Parent.m_FastResult = value; } }

			object m_FastResult;
			/// <summary>
			/// Gets the prepared output from the child operation of this process operation. Consistency with the general logic (in particular, with the OperaDB) is not guaranteed: it depends on the callee, but the responsibility to accept the prepared output depends on the caller.
			/// </summary>
			public override object FastResult { get { return m_FastResult; } }
			/// <summary>
			/// Cached copy of program settings.
			/// </summary>
			internal string m_ProgramSettings;
			/// <summary>
			/// Program settings for the process operation.
			/// </summary>
			public override string ProgramSettings { get { return ((string)m_ProgramSettings.Clone()); } }

			void ResetProgressInfo()
			{
				lock(m_StartupInfo)
				{
					m_ProgressInfo = new SySal.DAQSystem.Drivers.TaskProgressInfo();
					m_ProgressInfo.Complete = false;
					m_ProgressInfo.StartTime = System.DateTime.Now;
					m_ProgressInfo.FinishTime = m_ProgressInfo.StartTime.AddDays(1.0);
					m_ProgressInfo.LastProcessedInterruptId = 0;
					m_ProgressInfo.Progress = 0.0;
					m_ProgressInfo.ExitException = "";
					m_ProgressInfo.CustomInfo = null;
				}
			}

			/// <summary>
			/// Progress information about the process operation.
			/// </summary>
			public override SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo
			{ 
				get { return m_ProgressInfo; }				
				set
				{
					lock(m_StartupInfo)
					{
						if (value != null) m_ProgressInfo = value;
						else ResetProgressInfo();
						WriteProgress();
					}
				}			
			}

			/// <summary>
			/// Provides quick write access to the Progress field of the progress info.
			/// </summary>
			public override double Progress 
			{ 
				set 
				{
					lock(m_StartupInfo)
					{
						if (m_ProgressInfo == null) ResetProgressInfo();
						m_ProgressInfo.Progress = value;
						WriteProgress();
					}
				}
			}

			/// <summary>
			/// Provides quick write access to the CustomInfo field of the progress info.
			/// </summary>
			public override string CustomInfo 
			{ 
				set
				{
					lock(m_StartupInfo)
					{
						if (m_ProgressInfo == null) ResetProgressInfo();
						m_ProgressInfo.CustomInfo = value;
						WriteProgress();
					}
				}			
			}

			/// <summary>
			/// Provides quick write access to the Complete field of the progress info.
			/// </summary>
			public override bool Complete 
			{ 
				set
				{
					lock(m_StartupInfo)
					{
						if (m_ProgressInfo == null) ResetProgressInfo();
						m_ProgressInfo.Complete = value;
						WriteProgress();
					}
				}			
			}

			/// <summary>
			/// Provides quick write access to the LastProcessedInterruptId of the progress info.
			/// </summary>
			public override int LastProcessedInterruptId 
			{ 
				set
				{
					lock(m_StartupInfo)
					{
						if (m_ProgressInfo == null) ResetProgressInfo();
					}
					if (m_Interrupts.Count == 0) lock(m_StartupInfo) m_ProgressInfo.LastProcessedInterruptId = 0;
					else
						lock(m_StartupInfo)
						{
							if (((Interrupt)(m_Interrupts.Peek())).Id == value)
							{
								m_ProgressInfo.LastProcessedInterruptId = value;
								m_Interrupts.Dequeue();
								WriteInterruptQueue();								
								InterruptNotificationThread.Interrupt();
							}
						}
					lock(m_StartupInfo) WriteProgress();	
				}		
			}

			/// <summary>
			/// Provides quick write access to the ExitException of the progress info.
			/// </summary>
			public override string ExitException
			{ 
				set
				{
					lock(m_StartupInfo)
					{
						if (m_ProgressInfo == null) ResetProgressInfo();
						m_ProgressInfo.ExitException = value;			
						WriteProgress();
					}
				}
			}

			/// <summary>
			/// Gets the next interrupt for the specified process.
			/// </summary>
			/// <returns>the next unprocessed interrupt. Null is returned if no unprocessed interrupt exists.</returns>
			public override SySal.DAQSystem.Drivers.Interrupt NextInterrupt 
			{ 
				get
				{
					lock(m_StartupInfo)
					{
						if (m_Interrupts.Count == 0) return null;
						return ((Interrupt)(m_Interrupts.Peek()));						
					}
				}			
			}

			/// <summary>
			/// Writes text to the host environment logger.
			/// </summary>
			/// <param name="text">the text to be written.</param>
			public override void Write(string text)
			{
				CF.Write(text.Replace("\n", "\r\n"));
			}

			/// <summary>
			/// Writes text to the host environment logger and advances to the next line.
			/// </summary>
			/// <param name="text">the text to be written.</param>
			public override void WriteLine(string text)
			{
				CF.Write(text.Replace("\n", "\r\n") + "\r\n");
			}

			/// <summary>
			/// Registers an interrupt notifier interface for the driver process.
			/// Upon registration the driver process should be sent notifications about the first interrupt, if any.
			/// </summary>
			public override SySal.DAQSystem.Drivers.IInterruptNotifier InterruptNotifier 
			{ 
				set
				{
					lock(m_StartupInfo)
					{
						m_InterruptNotifier = value;
						InterruptNotificationThread.Interrupt();					
					}		
				}
			}

			/// <summary>
			/// Gets the DataProcessingServer (usually hosted by the BatchManager) that serves process running on the current BatchManager.
			/// </summary>
			public override IDataProcessingServer DataProcSrv 
			{ 
				get
				{
					return MainForm.DPS;
				}
			}

			internal ScanServer m_ScanSrv = null;
			/// <summary>
			/// Gets the ScanServer associated to this process operation. 
			/// The ScanServer is locked when the process operation starts and is in a running state. No other process referring to the same ScanServer can be running at the same moment.
			/// </summary>
			public override ScanServer ScanSrv 
			{ 
				get
				{
					if (m_ScanSrv != null) return m_ScanSrv;
					m_ScanSrv = TheInstance.LockedScanSrv(m_StartupInfo.MachineId);
                    ScanServerFreeEvent.Reset();
					return m_ScanSrv;
				}
                set
                {
                    if (value != null) throw new Exception("A ScanServer can only be set to null (to unlock it).");
                    TheInstance.UnlockScanSrv(this);
                    ScanServerFreeEvent.Set();
                }
			}
			#endregion

		}

		/// <summary>
		/// The list of tasks currently being executed.
		/// </summary>
		protected System.Collections.ArrayList m_TaskList;

		internal HostEnv [] Tasks
		{
			get 
			{
				lock(DBConn)
					return (HostEnv [])(m_TaskList.ToArray(typeof(HostEnv)));
			}
		}

		/// <summary>
		/// The unique DB connection.
		/// </summary>
		protected internal SySal.OperaDb.OperaDbConnection DBConn = new OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
 
		/// <summary>
		/// Creates a new BatchManager.
		/// </summary>
		/// <param name="eventlog">the event logger to use.</param>
		/// <param name="pev">the interface to be called on process events.</param>
		internal BatchManager(System.Diagnostics.EventLog eventlog, IProcessEventNotifier pev)
		{
			TheInstance = this;
			m_ProcessEventNotifier = pev;
			m_EventLog = eventlog;
			m_TaskList = new System.Collections.ArrayList();
			System.Xml.Serialization.XmlSerializer xmls1 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.ScanningStartupInfo));
			System.Xml.Serialization.XmlSerializer xmls2 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.VolumeOperationInfo));
			System.Xml.Serialization.XmlSerializer xmls3 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.BrickOperationInfo));
			System.Xml.Serialization.XmlSerializer xmls4 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.TaskStartupInfo));
            Cleanup();
			lock(DBConn)
				lock(DBConn)
				{				
					DBConn.Open();
					System.Data.DataSet ds = new System.Data.DataSet();
					SySal.OperaDb.OperaDbDataAdapter da = new OperaDb.OperaDbDataAdapter("SELECT /*+INDEX (TB_MACHINES PK_MACHINES) INDEX_ASC (TB_PROC_OPERATIONS IX_PROC_OPERATIONS_START) */ TB_PROC_OPERATIONS.ID, TB_PROGRAMSETTINGS.EXECUTABLE, TB_PROC_OPERATIONS.DRIVERLEVEL, TB_PROC_OPERATIONS.ID_PARENT_OPERATION, TB_PROC_OPERATIONS.NOTES FROM TB_PROGRAMSETTINGS INNER JOIN (TB_PROC_OPERATIONS INNER JOIN TB_MACHINES ON (TB_PROC_OPERATIONS.ID_MACHINE = TB_MACHINES.ID)) ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID) WHERE (TB_MACHINES.ID_SITE = " + MainForm.IdSite + " AND TB_PROC_OPERATIONS.STARTTIME IS NOT NULL AND TB_PROC_OPERATIONS.FINISHTIME IS NULL) ORDER BY ID ASC", DBConn, null);
					da.Fill(ds);
                    string xcxtext = "";
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    {                        
                        try
                        {
                            long id = SySal.OperaDb.Convert.ToInt64(dr[0]);
                            string startupfile = MainForm.TaskDir + "\\bmt_" + id.ToString() + ".startup";
                            if (System.IO.File.Exists(startupfile))
                            {
                                System.IO.StreamReader r = new System.IO.StreamReader(startupfile);
                                string xmlstr = r.ReadToEnd();
                                r.Close();
                                SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo = null;
                                try
                                {
                                    startupinfo = (SySal.DAQSystem.Drivers.TaskStartupInfo)xmls1.Deserialize(new System.IO.StringReader(xmlstr));
                                }
                                catch (Exception)
                                {
                                    try
                                    {
                                        startupinfo = (SySal.DAQSystem.Drivers.TaskStartupInfo)xmls2.Deserialize(new System.IO.StringReader(xmlstr));
                                    }
                                    catch (Exception)
                                    {
                                        try
                                        {
                                            startupinfo = (SySal.DAQSystem.Drivers.TaskStartupInfo)xmls3.Deserialize(new System.IO.StringReader(xmlstr));
                                        }
                                        catch (Exception)
                                        {
                                            try
                                            {
                                                startupinfo = (SySal.DAQSystem.Drivers.TaskStartupInfo)xmls4.Deserialize(new System.IO.StringReader(xmlstr));
                                            }
                                            catch (Exception)
                                            {
                                                throw new Exception("Corrupt startup file for Process Operation Id = " + id + "!");
                                            }
                                        }
                                    }
                                }
                                if (startupinfo.ProcessOperationId != id) throw new Exception("Startup file for id = " + id + " has non-matching Process Id = " + startupinfo.ProcessOperationId + "!");
                                startupinfo.RecoverFromProgressFile = System.IO.File.Exists(startupinfo.ProgressFile);
                                System.Data.DataSet dsp = new System.Data.DataSet();
                                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX (TB_PROGRAMSETTINGS PK_PROGRAMSETTINGS) */ EXECUTABLE, DESCRIPTION, SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + startupinfo.ProgramSettingsId, DBConn, null).Fill(dsp);
                                HostEnv parent = null;
                                if (dr[3] != System.DBNull.Value)
                                {
                                    long parentid = SySal.OperaDb.Convert.ToInt64(dr[3]);
                                    foreach (HostEnv h in m_TaskList)
                                        if (h.m_StartupInfo.ProcessOperationId == parentid)
                                        {
                                            parent = h;
                                            break;
                                        }
                                }
                                string token = "";
                                int xtoken = 0;
                                while (xtoken < 2 && token.Length == 0)
                                    try
                                    {
                                        token = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (LZ_TOKENS IX_TOKENS_PROCOP) */ ID FROM OPERA.LZ_TOKENS WHERE ID_PROCESSOPERATION = " + startupinfo.ProcessOperationId, DBConn, null).ExecuteScalar().ToString();
                                    }
                                    catch (Exception xcx)
                                    {
                                        if (++xtoken >= 2)
                                        {
                                            xcxtext += "\r\n\r\nCannot recreate token for " + startupinfo.ProcessOperationId + ".\r\n" + xcx.ToString();
                                            break;
                                        }
                                        xcxtext += "\r\n\r\nCannot retrieve process token for " + startupinfo.ProcessOperationId + ".\r\n" + xcx.ToString();
                                        new SySal.OperaDb.OperaDbCommand("CALL OPERA.LP_RECREATE_TOKEN(" + startupinfo.ProcessOperationId + ")", DBConn, null).ExecuteNonQuery();
                                    }
                                if (xtoken >= 2) continue;
                                string machinename = "";
                                try
                                {
                                    machinename = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_MACHINES AK_ID_MACHINES) */ NAME FROM TB_MACHINES WHERE ID = " + startupinfo.MachineId, DBConn, null).ExecuteScalar().ToString();
                                }
                                catch (Exception xcx)
                                {
                                    xcxtext += "\r\nUnknown machine " + startupinfo.MachineId + ".";
                                    continue;
                                }
                                HostEnv newtask = new HostEnv(startupinfo, startupfile, MainForm.DriverDir + "\\" + dsp.Tables[0].Rows[0][0].ToString(), dsp.Tables[0].Rows[0][2].ToString(), parent, null, new ConsoleForm(), token);
                                m_TaskList.Add(newtask);
                                string notes = null;
                                if (dr[4] != System.DBNull.Value) notes = dr[4].ToString();
                                new System.Threading.Thread(new System.Threading.ThreadStart(new ProcessStartNotify(m_ProcessEventNotifier, newtask, dsp.Tables[0].Rows[0][1].ToString(), machinename, notes).Exec)).Start();
                            }
                            else
                            {
                                xcxtext += "\r\nMissing startup information for Process Operation ID = " + id + ".";
                                continue;
                            }
                        }
                        catch (Exception xcx)
                        {
                            xcxtext += "\r\nError resuming process " + dr[0].ToString() + "\r\n" + xcx.ToString();
                        }
                    }
                    DBConn.Close();
                    if (xcxtext.Length > 0)
                    {
                        eventlog.WriteEntry(xcxtext, System.Diagnostics.EventLogEntryType.Error);
                        System.Windows.Forms.MessageBox.Show("Errors occurred while resuming processes.\r\nPlease check the event log for further details.", "Errors found", System.Windows.Forms.MessageBoxButtons.OK, System.Windows.Forms.MessageBoxIcon.Warning);
                    }
				}	
            CleanupTimer = new System.Timers.Timer(60000); //(600000);
			CleanupTimer.AutoReset = true;
			CleanupTimer.Elapsed += new System.Timers.ElapsedEventHandler(CleanupHook);
			CleanupTimer.Start();
		}

		/// <summary>
		/// Creates a new BatchManager. This is only for conformance to the BatchManagerScheme.
		/// </summary>
		public BatchManager() {}

		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>the lifetime service object or null.</returns>
		public override object InitializeLifetimeService()
		{
			return null;	
		}

		/// <summary>
		/// The internal member on which Machines relies.
		/// </summary>
		protected long [] MachineIds;

		/// <summary>
		/// The machine ids handled by this BatchManager.
		/// </summary>
		public virtual long [] Machines 
		{ 
			get 
			{ 
				return (long [])MachineIds.Clone();
			} 
		}

		/// <summary>
		/// The ids of the process operations currently handled by this BatchManager.
		/// </summary>
		public virtual long [] Operations 
		{ 			
			get 
			{ 
				lock(DBConn)
				{
					long [] ids = new long[m_TaskList.Count];
					int i;
					for (i = 0; i < m_TaskList.Count; i++)
						ids[i] = ((HostEnv)m_TaskList[i]).m_StartupInfo.ProcessOperationId;
					return ids;
				}
			} 
		}

		/// <summary>
		/// Retrieves the startup information (except password and alias credentials) for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which startup information is required.</param>
		/// <returns>the startup information of the process operation. If the process operation is complete or unknown, returns null.</returns>
		public virtual SySal.DAQSystem.Drivers.TaskStartupInfo GetOperationStartupInfo(long id)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
					if (h.m_StartupInfo.ProcessOperationId == id)
						return h.StartupInfo;
				return null;
			}
		}

		/// <summary>
		/// Retrieves the task information for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which task information is required.</param>
		/// <returns>the task information of the process operation. If the process operation is complete or unknown, returns null.</returns>
		internal HostEnv GetTaskInfo(long id)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
					if (h.m_StartupInfo.ProcessOperationId == id)
						return h;
				return null;
			}
		}

		/// <summary>
		/// Retrieves the progress information for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which progress information is required.</param>
		/// <returns>the progress information of the process operation. If the process operation is complete or unknown, returns null.</returns>
		public virtual SySal.DAQSystem.Drivers.TaskProgressInfo GetProgressInfo(long id)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						return h.ProgressInfo;
					};
				return null;
			}
		}

		/// <summary>
		/// Starts a new process operation.
		/// </summary>
		/// <param name="parentid">id of the parent process operation; if zero or negative, it is treated as NULL.</param>
		/// <param name="startupinfo">startup information for the process operation.</param>		
		/// <returns>the process operation id that has been allocated to this process operation.</returns>		
		public virtual long Start(long parentid, SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo)
		{
			return Start(parentid, startupinfo, null, null);
		}

		/// <summary>
		/// Starts a new process operation, including possible fast input.
		/// </summary>
		/// <param name="parentid">id of the parent process operation; if zero or negative, it is treated as NULL.</param>
		/// <param name="startupinfo">startup information for the process operation.</param>		
		/// <param name="fastinput">the fast input object.</param>
		/// <param name="parent">the parent process operation, if any.</param>
		/// <returns>the process operation id that has been allocated to this process operation.</returns>		
		internal virtual long Start(long parentid, SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo, object fastinput, HostEnv parent)
		{
            lock (DBConn)
            {
                System.Data.DataSet ds = null;
                OperaDb.OperaDbDataAdapter da = null;
                long userid = -1;
                HostEnv newtask = null;
                string machinename = "";
                lock (DBConn)
                {
                    try
                    {
                        DBConn.Open();
                        ds = new System.Data.DataSet();
                        da = new OperaDb.OperaDbDataAdapter("SELECT /*+INDEX (TB_PROGRAMSETTINGS PK_PROGRAMSETTINGS) */ EXECUTABLE, DRIVERLEVEL, SETTINGS, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE (ID = " + startupinfo.ProgramSettingsId + ")", DBConn, null);
                        da.Fill(ds);
                    }
                    catch (Exception x)
                    {
                        m_EventLog.WriteEntry("Error in process startup!\r\nProgram settings Id = " + startupinfo.ProgramSettingsId + "\r\n\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
                        throw x;
                    }
                    if (ds.Tables.Count != 1 || ds.Tables[0].Rows.Count != 1)
                    {
                        DBConn.Close();
                        throw new Exception("Unknown or ambiguous program settings! Cannot start process operation.");
                    }
                    if (startupinfo.MachineId < 0) startupinfo.MachineId = MainForm.IdMachine;
                    startupinfo.DBServers = MainForm.DBServer;
                    startupinfo.DBUserName = MainForm.DBUserName;
                    startupinfo.DBPassword = MainForm.DBPassword;
                    startupinfo.ExeRepository = MainForm.ExeRepository;
                    startupinfo.LinkedZonePath = MainForm.ScratchDir;
                    startupinfo.RawDataPath = MainForm.RawDataDir;
                    startupinfo.ScratchDir = MainForm.ScratchDir;
                    startupinfo.RecoverFromProgressFile = false;

                    string token;

                    switch ((SySal.DAQSystem.Drivers.DriverType)SySal.OperaDb.Convert.ToInt32(ds.Tables[0].Rows[0][1]))
                    {
                        case SySal.DAQSystem.Drivers.DriverType.Scanning: startupinfo.ProcessOperationId = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.StartTokenized(parentid, ((SySal.DAQSystem.Drivers.ScanningStartupInfo)startupinfo).CalibrationId, startupinfo.MachineId, startupinfo.ProgramSettingsId, startupinfo.OPERAUsername, startupinfo.OPERAPassword, out token, out userid, ((SySal.DAQSystem.Drivers.ScanningStartupInfo)startupinfo).Plate.BrickId, ((SySal.DAQSystem.Drivers.ScanningStartupInfo)startupinfo).Plate.PlateId, startupinfo.Notes, DBConn, null); break;
                        case SySal.DAQSystem.Drivers.DriverType.Volume: startupinfo.ProcessOperationId = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.StartTokenized(parentid, startupinfo.MachineId, startupinfo.ProgramSettingsId, startupinfo.OPERAUsername, startupinfo.OPERAPassword, out token, out userid, ((SySal.DAQSystem.Drivers.VolumeOperationInfo)startupinfo).BrickId, startupinfo.Notes, DBConn, null); break;
                        case SySal.DAQSystem.Drivers.DriverType.Brick: startupinfo.ProcessOperationId = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.StartTokenized(parentid, startupinfo.MachineId, startupinfo.ProgramSettingsId, startupinfo.OPERAUsername, startupinfo.OPERAPassword, out token, out userid, ((SySal.DAQSystem.Drivers.BrickOperationInfo)startupinfo).BrickId, startupinfo.Notes, DBConn, null); break;
                        case SySal.DAQSystem.Drivers.DriverType.System: startupinfo.ProcessOperationId = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.StartTokenized(parentid, startupinfo.MachineId, startupinfo.ProgramSettingsId, startupinfo.OPERAUsername, startupinfo.OPERAPassword, out token, out userid, startupinfo.Notes, DBConn, null); break;
                        default: throw new Exception("This BatchManager cannot manage operations with DriverLevel = " + SySal.OperaDb.Convert.ToInt32(ds.Tables[0].Rows[0][1].ToString()));
                    }

                    string startupfile = MainForm.TaskDir + "\\bmt_" + startupinfo.ProcessOperationId.ToString() + ".startup";
                    startupinfo.ProgressFile = MainForm.TaskDir + "\\bmt_" + startupinfo.ProcessOperationId.ToString() + ".progress";

                    newtask = new HostEnv(startupinfo, startupfile, MainForm.DriverDir + "\\" + ds.Tables[0].Rows[0][0].ToString(), ds.Tables[0].Rows[0][2].ToString(), parent, fastinput, new ConsoleForm(), token);

                    m_TaskList.Add(newtask);                    
                    try
                    {
                        machinename = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_MACHINES AK_ID_MACHINES) */ NAME FROM TB_MACHINES WHERE ID = " + newtask.m_StartupInfo.MachineId, DBConn, null).ExecuteScalar().ToString();
                    }
                    catch (Exception)
                    {
                        machinename = "Unknown";
                    }
                    DBConn.Close();
                }
                new System.Threading.Thread(new System.Threading.ThreadStart(new ProcessStartNotify(m_ProcessEventNotifier, newtask, ds.Tables[0].Rows[0][3].ToString(), machinename, startupinfo.Notes).Exec)).Start();
                newtask.Run();

                return newtask.m_StartupInfo.ProcessOperationId;
            }
		}

		internal class ProcessStartNotify
		{
			private IProcessEventNotifier IPEN;
			private HostEnv H;
			private string Description;
			private string MachineName;
			private string Notes;
			internal ProcessStartNotify(IProcessEventNotifier ipen, HostEnv h, string description, string machinename, string notes)
			{
				IPEN = ipen;
				H = h;
				Description = description;
				MachineName = machinename;
				Notes = notes;
			}
			internal void Exec() 
			{ 
				MainForm.ThreadLogStart("ProcessStartNotify");
				IPEN.ProcessStart(H, Description, MachineName, Notes); 
				MainForm.ThreadLogEnd();
			}
		}

        internal class ProcessEndNotify
        {
            private IProcessEventNotifier IPEN;
            private long ProcessOperationId;

            internal ProcessEndNotify(IProcessEventNotifier ipen, long procid)
            {
                IPEN = ipen;
                ProcessOperationId = procid;
            }
            internal void Exec()
            {
                MainForm.ThreadLogStart("ProcessEndNotify");
                IPEN.ProcessEnd(ProcessOperationId);
                MainForm.ThreadLogEnd();
            }
        }

		/// <summary>
		/// Pauses a process operation.
		/// </summary>
		/// <param name="id">the id of the process operation to be paused.</param>
		/// <param name="username">username to pause the process operation; must match the one used to start the process operation.</param>
		/// <param name="password">password to pause the process operation.</param>
		/// <returns>the status of the process operation.</returns>
		public virtual SySal.DAQSystem.Drivers.Status Pause(long id, string username, string password)
		{
			lock(DBConn)
			{
				SySal.DAQSystem.Drivers.Status ret = SySal.DAQSystem.Drivers.Status.Unknown;
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						lock(DBConn)
						{
							DBConn.Open();
							try
							{
								SySal.OperaDb.ComputingInfrastructure.User.CheckTokenOwnership(h.Token, 0, username, password, DBConn, null);
							}
							catch (Exception) 
							{
								throw new Exception("The username/password pair supplied does not match with the one used to start the process operation.");
							}
							finally
							{
								DBConn.Close();
							}
						}
                        if (h.Domain != null) h.Stop();
                        ret = SySal.DAQSystem.Drivers.Status.Paused;
						//ret = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.Status(id, DBConn, null);
					}
				}
				return ret;
			}
		}

		/// <summary>
		/// Internal method to reload the configuration for a process operation.
		/// </summary>
		/// <param name="id">the process operation to be reconfigured.</param>
		internal void Reconfig(long id)
		{
            HostEnv h = null;
			lock(DBConn)
			{				
				foreach (HostEnv x in m_TaskList)
				{
					if (x.m_StartupInfo.ProcessOperationId == id)
					{
                        h = x;
                        break;
                    }
                }
            }
            if (h != null)
            {
                try
                {
                    if (h.Domain != null) h.Stop();
                }
                catch (Exception x)
                {
                    try
                    {
                        m_EventLog.WriteEntry("Unable to stop Process operation #" + h.m_StartupInfo.ProcessOperationId + " to reconfigure with configuration " + h.m_StartupInfo.ProgramSettingsId + "\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
                    }
                    catch (Exception) { };
                    return;
                }
                lock(DBConn)
                    try
                    {
                        DBConn.Open();
                        h.m_ProgramSettings = new OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + h.m_StartupInfo.ProgramSettingsId, DBConn).ExecuteScalar().ToString();
                    }
                    catch (Exception x)
                    {
                        try
                        {
                            m_EventLog.WriteEntry("Unable to reconfigure Process operation #" + h.m_StartupInfo.ProcessOperationId + " with configuration " + h.m_StartupInfo.ProgramSettingsId + "\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
                        }
                        catch (Exception) { }
                    }
                    finally
                    {
                        DBConn.Close();
                    }
            }		
		}

		/// <summary>
		/// Internal method to pause an operation from the Windows interface or from a HostEnv.
		/// </summary>
		/// <param name="id">the process operation to be paused.</param>
		/// <param name="token">the token of the calling process operation.</param>
		internal SySal.DAQSystem.Drivers.Status Pause(long id, string token)
		{
			lock(DBConn)
			{				
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						if (token != null && token != h.Token) throw new Exception("A process operation can only pause one of its descendants.");						
						if (h.Domain != null) h.Stop();
						return SySal.DAQSystem.Drivers.Status.Paused;
					}
				}
				return SySal.DAQSystem.Drivers.Status.Unknown;
			}			
		}

		/// <summary>
		/// Resumes a paused process operation.
		/// </summary>
		/// <param name="id">the id of the process operation to be resumed.</param>
		/// <param name="username">username to resume the process operation; must match the one used to start the process operation.</param>
		/// <param name="password">password to resume the process operation; must match the one used to start the process operation.</param>
		/// <returns>the status of the process operation.</returns>
		public virtual SySal.DAQSystem.Drivers.Status Resume(long id, string username, string password)
		{
			lock(DBConn)
			{
				SySal.DAQSystem.Drivers.Status ret = SySal.DAQSystem.Drivers.Status.Unknown;
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						lock(DBConn)
						{
							DBConn.Open();
							try 
							{
								SySal.OperaDb.ComputingInfrastructure.User.CheckTokenOwnership(h.Token, 0, username, password, DBConn, null);
							}
							catch (Exception) 
							{
								throw new Exception("The username/password pair supplied does not match with the one used to start the process operation.");
							}
							finally
							{
								DBConn.Close();
							}
                            if (h.Domain != null) ret = SySal.DAQSystem.Drivers.Status.Running;
                            else
                            {
                                h.m_StartupInfo.RecoverFromProgressFile = true;
                                try
                                {
                                    h.Run();
                                    return SySal.DAQSystem.Drivers.Status.Running;
                                }
                                catch (Exception x)
                                {
                                    m_EventLog.WriteEntry("Error in process resume!\r\nProcess Id = " + h.m_StartupInfo.ProcessOperationId + "\r\n\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
                                    return SySal.DAQSystem.Drivers.Status.Paused;
                                }
                            }
                            //ret = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.Status(id, DBConn, null);
						}
					}
				}
				return ret;
			}
		}

		/// <summary>
		/// Internal method to resume an operation from the Windows interface or from a HostEnv.
		/// </summary>
		/// <param name="id">the process operation to be resumed.</param>
		/// <param name="token">the token of the calling process operation.</param>
		internal SySal.DAQSystem.Drivers.Status Resume(long id, string token)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						if (token != null && token != h.Token) throw new Exception("A process operation can only resume one of its descendants.");
						if (h.Domain != null) return SySal.DAQSystem.Drivers.Status.Running;
						h.m_StartupInfo.RecoverFromProgressFile = true;
						try
						{
							h.Run();
							return SySal.DAQSystem.Drivers.Status.Running;
						}
						catch (Exception) {}
					}
				}
				return SySal.DAQSystem.Drivers.Status.Unknown;
			}			
		}

		/// <summary>
		/// Shows a hidden console form.
		/// </summary>
		/// <param name="id">the id of the process whose console form must be shown.</param>		
		internal void Show(long id)
		{
			HostEnv h = null;
			lock(DBConn)
			{
				int i;
				for (i = 0; i < m_TaskList.Count; i++)
				{
					h = (HostEnv)m_TaskList[i];
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						if (h.CF != null) 
							try
							{
                                //h.CF.Show();
                                //h.CF.Visible = true;
                                h.CF.SetVisible(true);
                            }
							catch (Exception) {}
						break;
					}
				}
			}									
		}


		/// <summary>
		/// Aborts a process operation.
		/// </summary>
		/// <param name="id">the id of the process operation to be aborted.</param>
		/// <param name="username">username to abort the process operation; must match the one used to start the process operation.</param>
		/// <param name="password">password to abort the process operation; must match the one used to start the process operation.</param>
		/// <returns>the status of the process operation.</returns>
		public virtual SySal.DAQSystem.Drivers.Status Abort(long id, string username, string password)
		{
			HostEnv h = null;
			lock(DBConn)
			{
				int i;
				for (i = 0; i < m_TaskList.Count; i++)
				{
					h = (HostEnv)m_TaskList[i];
					if (h.m_StartupInfo.ProcessOperationId == id) break;
				}
				if (h == null) return SySal.DAQSystem.Drivers.Status.Unknown;
				else
				{
					lock (DBConn)
                        try
                        {
                            DBConn.Open();
                            SySal.OperaDb.ComputingInfrastructure.User.CheckTokenOwnership(h.Token, 0, username, password, DBConn, null);
                            SySal.OperaDb.ComputingInfrastructure.ProcessOperation.EndTokenized(id, false, DBConn, null);
                        }
                        catch (Exception x)
                        {
                            m_EventLog.WriteEntry("Error in process abort!\r\nProcess Id = " + h.m_StartupInfo.ProcessOperationId + "\r\n\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
                            return SySal.DAQSystem.Drivers.Status.Unknown;
                        }
                        finally
                        {
                            DBConn.Close();		
                        }
					m_TaskList.RemoveAt(i);
				}
			}
			try
			{
                new System.Threading.Thread(new ProcessEndNotify(m_ProcessEventNotifier, h.m_StartupInfo.ProcessOperationId).Exec).Start();
                do
                {
                    try
                    {
                        h.Stop();
                    }
                    catch (Exception) { }
                }
                while (h.Domain != null);
				h.CompletionEvent.Set();
				h.DeleteFiles();						
			}
			catch (Exception x)
			{
				m_EventLog.WriteEntry("Error in process abort!\r\nProcess Id = " + h.m_StartupInfo.ProcessOperationId + "\r\n\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);							
				DBConn.Close();
				return SySal.DAQSystem.Drivers.Status.Unknown;
			}

			SySal.DAQSystem.Drivers.Status ret = SySal.DAQSystem.Drivers.Status.Unknown;
			lock(DBConn)
			{
				DBConn.Open();
				try
				{
					// ret = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.Status(id, DBConn, null);
				}
				catch (Exception) {}
				DBConn.Close();
			}
			return ret;						
		}

		/// <summary>
		/// Internal method to abort an operation from the Windows interface or from a HostEnv.
		/// </summary>
		/// <param name="id">the process operation to be aborted..</param>
		/// <param name="token">the token of the calling process operation.</param>
		internal SySal.DAQSystem.Drivers.Status Abort(long id, string token)
		{
			HostEnv h = null;
			lock(DBConn)
			{
				int i;
				for (i = 0; i < m_TaskList.Count; i++)
				{
					h = (HostEnv)m_TaskList[i];
					if (h.m_StartupInfo.ProcessOperationId == id) break;
				}
				if (h == null) return SySal.DAQSystem.Drivers.Status.Unknown;
				else if (token != null && token != h.Token) throw new Exception("A process operation can only abort one of its descendants."); 
				else m_TaskList.RemoveAt(i);
			}
            try
            {
                lock (DBConn)
                {
                    try
                    {
                        DBConn.Open();
                        SySal.OperaDb.ComputingInfrastructure.ProcessOperation.EndTokenized(id, false, DBConn, null);
                        do
                        {
                            try
                            {
                                h.Stop();
                            }
                            catch (Exception) { }
                        }
                        while (h.Domain != null);
                        h.CompletionEvent.Set();
                        h.DeleteFiles();
                    }
                    finally
                    {
                        DBConn.Close();
                    }
                }
                new System.Threading.Thread(new ProcessEndNotify(m_ProcessEventNotifier, h.m_StartupInfo.ProcessOperationId).Exec).Start();
            }
            catch (Exception x)
            {                
                m_EventLog.WriteEntry("Error in process abort!\r\nProcess Id = " + h.m_StartupInfo.ProcessOperationId + "\r\n\r\n" + x.ToString(), System.Diagnostics.EventLogEntryType.Error);
            }
			SySal.DAQSystem.Drivers.Status ret = SySal.DAQSystem.Drivers.Status.Unknown;
			lock(DBConn)
			{
				DBConn.Open();
				try
				{
					// ret = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.Status(id, DBConn, null);
				}
				catch (Exception) {}
				DBConn.Close();
			}
			return ret;						
		}


		/// <summary>
		/// Retrieves the status of the specified process operation.
		/// </summary>
		/// <param name="id">the id of the process operation for which execution information is required.</param>
		public virtual SySal.DAQSystem.Drivers.Status GetStatus(long id)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
					if (h.m_StartupInfo.ProcessOperationId == id)					
					{
						return (h.Domain != null) ? SySal.DAQSystem.Drivers.Status.Running : SySal.DAQSystem.Drivers.Status.Paused;
					}
				SySal.DAQSystem.Drivers.Status ret = SySal.DAQSystem.Drivers.Status.Unknown;
				lock(DBConn)
				{
					DBConn.Open();
					try
					{
						ret = SySal.OperaDb.ComputingInfrastructure.ProcessOperation.Status(id, DBConn, null);
					}
					catch (Exception) {}
					DBConn.Close();
				}
				return ret;
			}
		}

        /// <summary>
        /// The list of busy machines.
        /// </summary>
        public long [] BusyMachines 
        {
            get
            {
                System.Collections.ArrayList ar = new System.Collections.ArrayList();
                lock (DBConn)
                {
                    foreach (HostEnv h in m_TaskList)
                        if (h.Domain != null)
                        {
                            long machineid = h.StartupInfo.MachineId;
                            if (ar.Contains(machineid) == false)
                                ar.Add(machineid);
                        }
                    return (long[])ar.ToArray(typeof(long));
                }
            }
        }
		
		/// <summary>
		/// Generates a summary of the specified process operation.
		/// </summary>
		/// <param name="id">the id of the process operation for which the summary is required.</param>
		public virtual SySal.DAQSystem.Drivers.BatchSummary GetSummary(long id)
		{
			SySal.DAQSystem.Drivers.BatchSummary bsm = new SySal.DAQSystem.Drivers.BatchSummary();
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
					if (h.m_StartupInfo.ProcessOperationId == id)			
					{
						bsm.Id = id;
						bsm.MachineId = h.m_StartupInfo.MachineId;
						Type stype = h.m_StartupInfo.GetType();
						if (stype == typeof(SySal.DAQSystem.Drivers.ScanningStartupInfo))
							bsm.DriverLevel = SySal.DAQSystem.Drivers.DriverType.Scanning;
						else if (stype == typeof(SySal.DAQSystem.Drivers.VolumeOperationInfo))
							bsm.DriverLevel = SySal.DAQSystem.Drivers.DriverType.Volume;
						else if (stype == typeof(SySal.DAQSystem.Drivers.BrickOperationInfo))
							bsm.DriverLevel = SySal.DAQSystem.Drivers.DriverType.Brick;
						else 
							bsm.DriverLevel = SySal.DAQSystem.Drivers.DriverType.System;
				
					
						bsm.Executable = (string)(h.m_Exe.Clone());
						bsm.Executable = bsm.Executable.Remove(0, bsm.Executable.LastIndexOf('\\') + 1);							
						bsm.ProgramSettingsId = h.m_StartupInfo.ProgramSettingsId;
						SySal.DAQSystem.Drivers.TaskProgressInfo pinfo = h.m_ProgressInfo;
						if (pinfo == null)
						{
							bsm.StartTime = System.DateTime.Now;
							bsm.ExpectedFinishTime = System.DateTime.Now;
							bsm.Progress = 0.0;
						}
						else
						{
							bsm.StartTime = pinfo.StartTime;
							bsm.ExpectedFinishTime = pinfo.FinishTime;
							bsm.Progress = pinfo.Progress;
						}
						bsm.BrickId = bsm.PlateId = 0;						
						switch (bsm.DriverLevel)
						{
							case SySal.DAQSystem.Drivers.DriverType.Scanning:	
								bsm.BrickId = ((SySal.DAQSystem.Drivers.ScanningStartupInfo)h.m_StartupInfo).Plate.BrickId;
								bsm.PlateId = ((SySal.DAQSystem.Drivers.ScanningStartupInfo)h.m_StartupInfo).Plate.PlateId;
								break;

							case SySal.DAQSystem.Drivers.DriverType.Volume:	
								bsm.BrickId = ((SySal.DAQSystem.Drivers.VolumeOperationInfo)h.m_StartupInfo).BrickId;
								break;

							case SySal.DAQSystem.Drivers.DriverType.Brick:	
								bsm.BrickId = ((SySal.DAQSystem.Drivers.BrickOperationInfo)h.m_StartupInfo).BrickId;
								break;
						}
						bsm.OpStatus = (h.Domain != null) ? SySal.DAQSystem.Drivers.Status.Running : SySal.DAQSystem.Drivers.Status.Paused;
						return bsm;
					}
			}
			return null;				
			
		}

		/// <summary>
		/// Tests the communication with the BatchManager.
		/// </summary>
		/// <param name="commpar">communication parameter.</param>
		/// <returns>2 * commpar - 1 if the BatchManager object and the communication are working properly.</returns>
		public virtual int TestComm(int commpar)
		{
			return 2 * commpar - 1;
		}

		/// <summary>
		/// Adds an interrupt to the interrupt list of the process. Interrupt data can be passed.
		/// </summary>
		/// <param name="id">the id of the process to be interrupted.</param>
		/// <param name="username">username to interrupt the process operation; must match the one used to start the process operation.</param>
		/// <param name="password">password to interrupt the process operation; must match the one used to start the process operation.</param>
		/// <param name="interruptdata">interrupt data to be passed to the process; their format and content depend on the specific executable driving the process.</param>
		public virtual void Interrupt(long id, string username, string password, string interruptdata)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						try
						{
							DBConn.Open();
							SySal.OperaDb.ComputingInfrastructure.User.CheckTokenOwnership(h.Token, 0, username, password, DBConn, null);
							DBConn.Close();
						}
						catch (Exception x) 
						{
							DBConn.Close();
							throw x;
						}
						h.QueueInterrupt(interruptdata);
						return;
					}
				}
			}
		}

		/// <summary>
		/// Internal method to interrupt an operation from the Windows interface or from a HostEnv.
		/// </summary>
		/// <param name="id">the process operation to be interrupt.</param>
		/// <param name="intdata">the interrupt data associated to the interrupt.</param>
		/// <param name="token">the token of the calling process operation.</param>
		internal void Interrupt(long id, string intdata, string token)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id) 
					{
						if (token != null && token != h.Token) throw new Exception("An operation can only send interrupts to one of its descendants.");
						h.QueueInterrupt(intdata);
						return;
					}
				}
			}			
		}

		/// <summary>
		/// Gets the next interrupt id for the specified process.
		/// </summary>
		/// <param name="id">the id of the process whose interrupt list is to be searched.</param>
		/// <returns>the id of the next unprocessed interrupt.</returns>
		public virtual Interrupt NextInterrupt(long id)
		{
			lock(DBConn)
			{
				foreach (HostEnv h in m_TaskList)
				{
					if (h.m_StartupInfo.ProcessOperationId == id)
					{
						return h.NextInterrupt;
					}
				}
			}
			throw new Exception("No process exists with the specified id.");
		}
	}
}
