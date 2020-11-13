using System;

namespace SySal.DAQSystem.Drivers
{
	/// <summary>
	/// This interface can be registered by driver processes for quick interrupt notifications.
	/// When the host environment detects an interrupt for the process, this interface is called to avoid polling.
	/// </summary>
	public interface IInterruptNotifier
	{
		/// <summary>
		/// Notifies the driver process of the new incoming interrupt.
		/// </summary>
		/// <param name="nextint">the next interrupt to be processed.</param>
		void NotifyInterrupt(Interrupt nextint);
	}

	/// <summary>
	/// Host environment for a driver process.
	/// </summary>
	public abstract class HostEnv : MarshalByRefObject
	{
		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>null to obtain an everlasting HostEnv.</returns>
		public override object InitializeLifetimeService()
		{
			return null;	
		}

		#region BatchManager functions
		/// <summary>
		/// The machine ids handled by this BatchManager.
		/// </summary>
		public abstract long [] Machines { get; }

		/// <summary>
		/// The ids of the process operations currently handled by this BatchManager.
		/// </summary>
		public abstract long [] Operations { get; }

		/// <summary>
		/// Retrieves the startup information (except password and alias credentials) for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which startup information is required.</param>
		/// <returns>the startup information of the process operation.</returns>
		public abstract SySal.DAQSystem.Drivers.TaskStartupInfo GetOperationStartupInfo(long id);

		/// <summary>
		/// Retrieves the progress information for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which progress information is required.</param>
		/// <returns>the progress information of the process operation.</returns>
		public abstract SySal.DAQSystem.Drivers.TaskProgressInfo GetProgressInfo(long id);

		/// <summary>
		/// Starts a new process operation, which will automatically be a child operation of the current one.
		/// </summary>
		/// <param name="startupinfo">startup information for the process operation.</param>		
		/// <returns>the process operation id that has been allocated to this process operation.</returns>
		/// <remarks>The child operation will inherit the token of the parent operation.</remarks>
		public abstract long Start(SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo);

		/// <summary>
		/// Starts a new process operation, which will automatically be a child operation of the current one, adding prepared input. Fast input may be used to avoid querying the DB, but the callee retains complete responsibility about the correctness of the fast input.
		/// </summary>
		/// <param name="startupinfo">startup information for the process operation.</param>
		/// <param name="fastinput">the fast input for the process operation. Correctness and consistency of this input cannot be guaranteed.</param>
		/// <returns>the process operation id that has been allocated to this process operation.</returns>
		/// <remarks>The child operation will inherit the token of the parent operation.</remarks>
		public abstract long Start(SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo, object fastinput);

		/// <summary>
		/// Stops execution of the current driver until the specified process operation returns.
		/// </summary>
		/// <param name="procopid">the Id of the process operation whose completion is being awaited.</param>
		/// <returns>the status of the operation after completion.</returns>
		public abstract SySal.DAQSystem.Drivers.Status Wait(long procopid);

        /// <summary>
        /// Stops execution of the current driver until the specified process operation returns or frees its scan server.
        /// </summary>
        /// <param name="procopid">the Id of the process operation whose completion is being awaited.</param>
        /// <returns>the status of the operation after completion.</returns>
        public abstract SySal.DAQSystem.Drivers.Status WaitForOpOrScanServer(long procopid);

        /// <summary>
		/// Pauses a process operation using the credentials of the current process operation.
		/// </summary>
		/// <param name="id">the id of the process operation to be paused.</param>
		/// <returns>the status of the process operation.</returns>
		public abstract SySal.DAQSystem.Drivers.Status Pause(long id);

		/// <summary>
		/// Resumes a paused process operation using the credentials of the current process operation..
		/// </summary>
		/// <param name="id">the id of the process operation to be resumed.</param>
		/// <returns>the status of the process operation.</returns>
		public abstract SySal.DAQSystem.Drivers.Status Resume(long id);

		/// <summary>
		/// Aborts a process operation using the credentials of the current process operation..
		/// </summary>
		/// <param name="id">the id of the process operation to be aborted.</param>
		/// <returns>the status of the process operation.</returns>
		public abstract SySal.DAQSystem.Drivers.Status Abort(long id);

		/// <summary>
		/// Retrieves the status of the specified process operation.
		/// </summary>
		/// <param name="id">the id of the process operation for which execution information is required.</param>
		public abstract SySal.DAQSystem.Drivers.Status GetStatus(long id);

		/// <summary>
		/// Generates a summary of the specified process operation.
		/// </summary>
		/// <param name="id">the id of the process operation for which the summary is required.</param>
		public abstract SySal.DAQSystem.Drivers.BatchSummary GetSummary(long id);

		/// <summary>
		/// Adds an interrupt to the interrupt list of the process using the credentials of the current process operation. Interrupt data can be passed.
		/// </summary>
		/// <param name="id">the id of the process to be interrupted.</param>
		/// <param name="interruptdata">interrupt data to be passed to the process; their format and content depend on the specific executable driving the process.</param>
		public abstract void Interrupt(long id, string interruptdata);

		#endregion

		#region Assistance to driver
		/// <summary>
		/// Startup information for the process.
		/// </summary>
		public abstract TaskStartupInfo StartupInfo { get; }

		/// <summary>
		/// Reads the fast input for this process operation. Consistency with the general logic (in particular, with the OperaDB) is not guaranteed: it depends on the caller, but the responsibility to accept the prepared input depends on the callee.
		/// </summary>
		public abstract object FastInput { get; }

		/// <summary>
		/// Sets the fast output for this process operation. Consistency with the general logic (in particular, with the OperaDB) is not guaranteed: it depends on the callee, but the responsibility to accept the prepared output depends on the caller.
		/// </summary>
		public abstract object FastOutput { set; }

		/// <summary>
		/// Gets the fast output from the child operation of this process operation. Consistency with the general logic (in particular, with the OperaDB) is not guaranteed: it depends on the callee, but the responsibility to accept the prepared output depends on the caller.
		/// </summary>
		public abstract object FastResult { get; }

		/// <summary>
		/// Program settings for the process operation.
		/// </summary>
		public abstract string ProgramSettings { get; }

		/// <summary>
		/// Progress information about the process operation.
		/// </summary>
		public abstract TaskProgressInfo ProgressInfo{ get; set; }

		/// <summary>
		/// Provides quick write access to the Progress field of the progress info.
		/// </summary>
		public abstract double Progress { set; }

		/// <summary>
		/// Provides quick write access to the CustomInfo field of the progress info.
		/// </summary>
		public abstract string CustomInfo { set; }

		/// <summary>
		/// Provides quick write access to the Complete field of the progress info.
		/// </summary>
		public abstract bool Complete { set; }

		/// <summary>
		/// Provides quick write access to the LastProcessedInterruptId of the progress info.
		/// </summary>
		public abstract int LastProcessedInterruptId { set; }

		/// <summary>
		/// Provides quick write access to the ExitException of the progress info.
		/// </summary>
		public abstract string ExitException { set; }

		/// <summary>
		/// Gets the next interrupt for the specified process.
		/// </summary>
		/// <returns>the next unprocessed interrupt. Null is returned if no unprocessed interrupt exists.</returns>
		public abstract Interrupt NextInterrupt { get; }

		/// <summary>
		/// Writes text to the host environment logger.
		/// </summary>
		/// <param name="text">the text to be written.</param>
		public abstract void Write(string text);

		/// <summary>
		/// Writes text to the host environment logger and advances to the next line.
		/// </summary>
		/// <param name="text">the text to be written.</param>
		public abstract void WriteLine(string text);

		/// <summary>
		/// Registers an interrupt notifier interface for the driver process.
		/// Upon registration the driver process should be sent notification about the first interrupt, if any.
		/// </summary>
		public abstract IInterruptNotifier InterruptNotifier { set; }

		/// <summary>
		/// Gets the DataProcessingServer (usually hosted by the BatchManager) that serves process running on the current BatchManager.
		/// </summary>
		public abstract IDataProcessingServer DataProcSrv { get; }

		/// <summary>
		/// Gets the ScanServer associated to this process operation. 
		/// The ScanServer is locked when the process operation starts and is in a running state. No other process referring to the same ScanServer can be running at the same moment.
		/// </summary>
        public abstract ScanServer ScanSrv { get; set; }

		/// <summary>
		/// Gets the Computing Infrastructure security token associated to this process operation.
		/// </summary>
		public abstract string Token { get; }

		#endregion

		/// <summary>
		/// The current environment of the calling driver process. Used by driver programs to gain access to their own host environment.
		/// </summary>
		public static HostEnv Own 
		{ 
			get 
			{ 
				try
				{
					return (HostEnv)(System.AppDomain.CurrentDomain.GetData("HostEnv")); 
				}
				catch (Exception)
				{
					return null;
				}
			} 
		}

        /// <summary>
        /// The name that is given to the Web application provided by each driver.
        /// </summary>
        public static readonly string WebAccessString = "WebAccess";

        /// <summary>
        /// User data that are passed to the Web application of each driver.
        /// </summary>
        [Serializable]
        public class WebUserData
        {
            /// <summary>
            /// Computing infrastructure user name.
            /// </summary>
            public string Usr;
            /// <summary>
            /// Computing infrastructure password.
            /// </summary>
            public string Pwd;
        }
	}
}
