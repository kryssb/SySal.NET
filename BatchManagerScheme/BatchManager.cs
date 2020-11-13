using System;
using SySal.BasicTypes;
using SySal.DAQSystem;
using SySal.DAQSystem.Scanning;
using System.Runtime.Serialization;
using System.Xml.Serialization;

namespace SySal.DAQSystem
{
	/// <summary>
	/// Batch manager definitions.
	/// </summary>
	public class BatchManager : MarshalByRefObject
	{		
		/// <summary>
		/// Creates a new BatchManager. 
		/// </summary>
		public BatchManager()
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
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
		/// The machine ids handled by this BatchManager.
		/// </summary>
		public virtual long [] Machines { get { throw new System.Exception("This is only a stub. Use a remote instance instead."); } }

		/// <summary>
		/// The ids of the process operations currently handled by this BatchManager.
		/// </summary>
		public virtual long [] Operations { get { throw new System.Exception("This is only a stub. Use a remote instance instead."); } }

		/// <summary>
		/// Retrieves the startup information (except password and alias credentials) for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which startup information is required.</param>
		/// <returns>the startup information of the process operation.</returns>
		public virtual SySal.DAQSystem.Drivers.TaskStartupInfo GetOperationStartupInfo(long id)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Retrieves the progress information for the specified process operation.
		/// </summary>
		/// <param name="id">id of the process operation for which progress information is required.</param>
		/// <returns>the progress information of the process operation.</returns>
		public virtual SySal.DAQSystem.Drivers.TaskProgressInfo GetProgressInfo(long id)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Starts a new process operation.
		/// </summary>
		/// <param name="parentid">id of the parent process operation; if zero or negative, it is treated as NULL.</param>
		/// <param name="startupinfo">startup information for the process operation.</param>		
		/// <returns>the process operation id that has been allocated to this process operation.</returns>
		public virtual long Start(long parentid, SySal.DAQSystem.Drivers.TaskStartupInfo startupinfo)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Pauses a process operation.
		/// </summary>
		/// <param name="id">the id of the process operation to be paused.</param>
		/// <param name="username">username to pause the process operation; must match the one used to start the process operation.</param>
		/// <param name="password">password to pause the process operation; must match the one used to start the process operation.</param>
		/// <returns>the status of the process operation.</returns>
		public virtual SySal.DAQSystem.Drivers.Status Pause(long id, string username, string password)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
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
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
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
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Retrieves the status of the specified process operation.
		/// </summary>
		/// <param name="id">the id of the process operation for which execution information is required.</param>
		public virtual SySal.DAQSystem.Drivers.Status GetStatus(long id)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Generates a summary of the specified process operation.
		/// </summary>
		/// <param name="id">the id of the process operation for which the summary is required.</param>
		public virtual SySal.DAQSystem.Drivers.BatchSummary GetSummary(long id)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Tests the communication with the BatchManager.
		/// </summary>
		/// <param name="i">communication parameter.</param>
		/// <returns>2 * commpar - 1 if the BatchManager object and the communication are working properly.</returns>
		public virtual int TestComm(int commpar)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
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
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}

		/// <summary>
		/// Gets the next interrupt for the specified process.
		/// </summary>
		/// <param name="id">the id of the process whose interrupt list is to be searched.</param>
		/// <returns>the next unprocessed interrupt. Null is returned if no unprocessed interrupt exist.</returns>
		public virtual SySal.DAQSystem.Drivers.Interrupt NextInterrupt(long id)
		{
			throw new System.Exception("This is only a stub. Use a remote instance instead.");
		}
	}
}
