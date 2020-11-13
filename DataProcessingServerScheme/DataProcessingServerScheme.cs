using System;
using SySal.DAQSystem;
using System.Runtime.Serialization;
using System.Security;
[assembly:AllowPartiallyTrustedCallers]

namespace SySal.DAQSystem
{
	/// <summary>
	/// Descriptor of a data processing batch.
	/// </summary>
	[Serializable]
	public class DataProcessingBatchDesc : ICloneable
	{
		/// <summary>
		/// The Identifier of the batch assigned by the the Data Processing Server.
		/// </summary>
		public ulong Id;
		/// <summary>
		/// Optional description of the data processing task.
		/// </summary>
		public string Description;
		/// <summary>
		/// Machine processing power class needed to run the batch, or 0 if it does not matter.
		/// </summary>
		public uint MachinePowerClass;
		/// <summary>
		/// Opera Computing Infrastructure process token that the driver should use. The token contains implicitly the user identification and the privileges at the time of token creation.
		/// If this string is null, the Username and Password fields are used to authenticate the user.
		/// </summary>
		public string Token;
		/// <summary>
		/// Username of the user requesting the batch. Used only if Token is a null string.
		/// </summary>
		public string Username;
		/// <summary>
		/// Password of the user requesting the batch. Used only if Token is a null string.
		/// </summary>
		public string Password;
		/// <summary>
		/// Alternate username of the user requesting the batch for special services (e.g. DB access).
		/// </summary>
		public string AliasUsername;
		/// <summary>
		/// Alternate password of the user requesting the batch for special services (e.g. DB access).
		/// </summary>
		public string AliasPassword;
		/// <summary>
		/// string specifying the command line arguments.
		/// </summary>
		public string CommandLineArguments;
		/// <summary>
		/// Full path name of the process file to be executed.
		/// </summary>
		public string Filename;
		/// <summary>
		/// Date/time when the batch was enqueued.
		/// </summary>
		public System.DateTime Enqueued;
		/// <summary>
		/// Date/time when the batch was started.
		/// </summary>
		public System.DateTime Started;
		/// <summary>
		/// Date/time when the batch was finished/terminated.
		/// </summary>
		public System.DateTime Finished;
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
        /// Maximum size of output text.
        /// </summary>
        public int MaxOutputText = 20480;
        /// <summary>
        /// File where the output text is to be saved. Set to <c>null</c> or to an empty string to avoid saving the output.
        /// </summary>
        public string OutputTextSaveFile;
		/// <summary>
		/// Clones the DataProcessingBatchDesc object.
		/// </summary>
		/// <returns>a new object identical to this DataProcessingBatchDesc.</returns>
		public object Clone()
		{
			DataProcessingBatchDesc dps = new DataProcessingBatchDesc();
			dps.Id = Id;
			if (Username != null) dps.Username = (string)Username.Clone();
			if (Password != null) dps.Password = (string)Password.Clone();
			if (Token != null) dps.Token = (string)Token.Clone();
			dps.AliasUsername = (string)AliasUsername.Clone();
			dps.AliasPassword = (string)AliasPassword.Clone();
			dps.MachinePowerClass = MachinePowerClass;
			dps.CommandLineArguments = CommandLineArguments;
			dps.Filename = Filename;
			dps.Enqueued = Enqueued;
			dps.Started = Started;
			dps.Finished = Finished;
			dps.TotalProcessorTime = TotalProcessorTime;
			dps.PeakVirtualMemorySize = PeakVirtualMemorySize;
			dps.PeakWorkingSet = PeakWorkingSet;
			dps.Description = (string)Description.Clone();
            dps.MaxOutputText = MaxOutputText;
            dps.OutputTextSaveFile = OutputTextSaveFile;
			return (object)dps;
		}
	}

	/// <summary>
	/// Data processing exception. Wraps exceptions born in the context of data processing.
	/// This class is useful in discriminating whether a generic exception is due to a computing exception or to difficulties in connections or remoting management: all computing exceptions are wrapped in DataProcessingExceptions, whereas connection errors, abortions, etc., are not.
	/// </summary>
	[Serializable]
	public class DataProcessingException : System.Exception
	{
		public DataProcessingException() : base() {}

		public DataProcessingException(string message) : base(message) {}		

		public DataProcessingException(string message, System.Exception innerException) : base(message, innerException)  {}

		protected DataProcessingException(SerializationInfo info, StreamingContext ctx) : base(info, ctx) {}
	}

	/// <summary>
	/// Data Processing Server definitions.
	/// </summary>
	public interface IDataProcessingServer
	{
        /// <summary>
        /// The number of jobs that can be performed in parallel. 
        /// </summary>
        uint ParallelJobs { get; }

		/// <summary>
		/// The queue of data processing batches to be executed.
		/// </summary>
		DataProcessingBatchDesc [] Queue { get; }

		/// <summary>
		/// The number of data processing batches to be executed.
		/// </summary>
		int QueueLength { get; }

		/// <summary>
		/// The power class of the machine.
		/// </summary>
		int MachinePowerClass { get; }

		/// <summary>
		/// Draws a batch out ouf the queue or aborts it if it is already being executed.
		/// A non-null token or a username/password pair must be supplied that matches the one with which the batch was started.
		/// If the token is supplied, the username/password pair is ignored.
		/// </summary>
		/// <param name="id">identifier of the batch to be removed.</param>
		/// <param name="token">the process token to be used.</param>
		/// <param name="user">username of the user that started the batch. Ignored if <c>token</c> is non-null.</param>
		/// <param name="password">password of the user that started the batch. Ignored if <c>token</c> is non-null.</param>
		void Remove(ulong id, string token, string user, string password);

		/// <summary>
		/// Enqueues a batch. If a non-null token is supplied, it is used; otherwise, the username/password pair is used to authenticate the user.
		/// </summary>
		/// <param name="desc">the descriptor of the batch.</param>
		/// <returns>true if the batch has been accepted, false otherwise.</returns>
		bool Enqueue(DataProcessingBatchDesc desc);

		/// <summary>
		/// Checks for execution completion.
		/// </summary>
		/// <param name="id">the id of the batch.</param>
		/// <returns>true if the batch has been completed, false if it is in progress.</returns>
		bool DoneWith(ulong id);

		/// <summary>
		/// Gets the result for a batch.
		/// </summary>
		/// <param name="id">the id of the batch.</param>
		/// <returns>the batch descriptor. It is modified to reflect the batch output. An exception is thrown if the batch terminated with an exception.</returns>
		DataProcessingBatchDesc Result(ulong id);

		/// <summary>
		/// Provides an Id for a new batch to be enqueued.
		/// Batch Id clashing is a reason for rejection of well-formed batch descriptors.
		/// Use of this property does not completely guarantee that the batch id does not clash with another Id in the queue, because another process could schedule another batch with the same Id.
		/// However, the Ids generated by this property all come from the same sequence and are very likely not to be duplicated within a reasonable amount of time.
		/// </summary>
		ulong SuggestId { get; }

		/// <summary>
		/// Checks whether the machine is willing to accept new requests of batch data processing.
		/// </summary>
		bool IsWillingToProcess { get; }

		/// <summary>
		/// Tests the communication with the DataProcessingServer.
		/// </summary>
		/// <param name="i">communication parameter.</param>
		/// <returns>2 * commpar - 1 if the DataProcessingServer object and the communication are working properly.</returns>
		int TestComm(int commpar);
	}

	/// <summary>
	/// Wraps a Data Processing Server to made synchronous calls easier with automatic detection of network errors.
	/// </summary>
	public class SyncDataProcessingServerWrapper : IDataProcessingServer
	{
		/// <summary>
		/// The contained DataProcessingServer object.
		/// </summary>
		protected IDataProcessingServer m_Srv;
		/// <summary>
		/// Communication timeout.
		/// </summary>
		protected System.TimeSpan m_Timeout;
		/// <summary>
		/// The generic object returned by an asynchronous method call.
		/// </summary>
		protected object m_ReturnObj;
		/// <summary>
		/// The generic exception returned by an asynchronous method call.
		/// </summary>
		protected Exception m_ReturnException;
		/// <summary>
		/// The thread in which the asynchronous call is executed.
		/// </summary>
		protected System.Threading.Thread m_Thread;

        /// <summary>
        /// The number of jobs that can be performed in parallel. 
        /// </summary>
        public uint ParallelJobs 
        { 
            get 
            {
                m_ReturnException = null;
                m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                    delegate()
                    {
                        try
                        {
                            m_ReturnObj = m_Srv.ParallelJobs;
                        }
                        catch (Exception x)
                        {
                            m_ReturnException = x;
                        }
                    }
                    ));
                m_Thread.Start();
                if (m_Thread.Join(m_Timeout) == false)
                {
                    try
                    {
                        m_Thread.Abort();
                    }
                    catch (Exception) { }
                    m_Srv = null;
                    throw new Exception("Communication timeout!");
                }
                if (m_ReturnException != null) throw m_ReturnException;
                return (uint)m_ReturnObj;
            } 
        }

		/// <summary>
		/// Checks whether the machine is willing to accept new requests of batch data processing.
		/// </summary>
		public bool IsWillingToProcess
		{
			get
			{
				m_ReturnException = null;
				m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                    delegate()
                    {
                        try
                        {
                            m_ReturnObj = m_Srv.IsWillingToProcess;
                        }
                        catch (Exception x)
                        {
                            m_ReturnException = x;
                        }
                    }
                    ));
				m_Thread.Start();
				if (m_Thread.Join(m_Timeout) == false)
				{
					try
					{
						m_Thread.Abort();
					}
					catch (Exception) {}
					m_Srv = null;
					throw new Exception("Communication timeout!");
				}
				if (m_ReturnException != null) throw m_ReturnException;
				return (bool)m_ReturnObj;
			}
		}

		/// <summary>
		/// Provides an Id for a new batch to be enqueued. 
		/// </summary>
		public ulong SuggestId
		{
			get
			{
				m_ReturnException = null;
				m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                    delegate()
                    {
                        try
                        {
                            m_ReturnObj = m_Srv.SuggestId;
                        }
                        catch (Exception x)
                        {
                            m_ReturnException = x;
                        }
                    }
                    ));
				m_Thread.Start();
				if (m_Thread.Join(m_Timeout) == false)
				{
					try
					{
						m_Thread.Abort();
					}
					catch (Exception) {}
					m_Srv = null;
					throw new Exception("Communication timeout!");
				}
				if (m_ReturnException != null) throw m_ReturnException;
				return (ulong)m_ReturnObj;
			}
		}

		/// <summary>
		/// The queue of data processing batches to be executed.
		/// </summary>
		public DataProcessingBatchDesc [] Queue 
		{ 
			get
			{
				m_ReturnException = null;
				m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                    delegate()
                    {
                        try
                        {
                            m_ReturnObj = m_Srv.Queue;
                        }
                        catch (Exception x)
                        {
                            m_ReturnException = x;
                        }
                    }
                    ));
				m_Thread.Start();
				if (m_Thread.Join(m_Timeout) == false)
				{
					try
					{
						m_Thread.Abort();
					}
					catch (Exception) {}
					m_Srv = null;
					throw new Exception("Communication timeout!");
				}
				if (m_ReturnException != null) throw m_ReturnException;
				return (DataProcessingBatchDesc [])m_ReturnObj;
			}
		}

		/// <summary>
		/// The number of data processing batches to be executed.
		/// </summary>
		public int QueueLength 
		{ 
			get
			{
				m_ReturnException = null;
				m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                    delegate()
                    {
                        try
                        {
                            m_ReturnObj = m_Srv.QueueLength;
                        }
                        catch (Exception x)
                        {
                            m_ReturnException = x;
                        }
                    }                    
                    ));
				m_Thread.Start();
				if (m_Thread.Join(m_Timeout) == false)
				{
					try
					{
						m_Thread.Abort();
					}
					catch (Exception) {}
					m_Srv = null;
					throw new Exception("Communication timeout!");
				}
				if (m_ReturnException != null) throw m_ReturnException;
				return (int)m_ReturnObj;
			}
		}

		/// <summary>
		/// The power class of the machine.
		/// </summary>
		public int MachinePowerClass 
		{ 
			get
			{
				m_ReturnException = null;
				m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                    delegate()
                    {
                        try
                        {
                            m_ReturnObj = m_Srv.MachinePowerClass;
                        }
                        catch (Exception x)
                        {
                            m_ReturnException = x;
                        }
                    }                    
                    ));
				m_Thread.Start();
				if (m_Thread.Join(m_Timeout) == false)
				{
					try
					{
						m_Thread.Abort();
					}
					catch (Exception) {}
					m_Srv = null;
					throw new Exception("Communication timeout!");
				}
				if (m_ReturnException != null) throw m_ReturnException;
				return (int)m_ReturnObj;
			}				
		}

		/// <summary>
		/// Input slot for aRemove and aDoneWith.
		/// </summary>
		protected ulong m_pId;

		/// <summary>
		/// Input slot for aRemove.
		/// </summary>
		protected string m_pUser;

		/// <summary>
		/// Input slot for aRemove.
		/// </summary>
		protected string m_pPassword;

		/// <summary>
		/// Input slot for aRemove.
		/// </summary>
		protected string m_pToken;

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
			m_pId = id;
			m_pUser = user;
			m_pPassword = password;
			m_ReturnException = null;
			m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                delegate()
                {
                    try
                    {
                        m_Srv.Remove(m_pId, m_pToken, m_pUser, m_pPassword);
                    }
                    catch (Exception x)
                    {
                        m_ReturnException = x;
                    }
                }
                ));
			m_Thread.Start();
			if (m_Thread.Join(m_Timeout) == false)
			{
				try
				{
					m_Thread.Abort();
				}
				catch (Exception) {}
				m_Srv = null;
				throw new Exception("Communication timeout!");
			}
			if (m_ReturnException != null) throw m_ReturnException;
		}

		/// <summary>
		/// Input slot for aEnqueue and aResult.
		/// </summary>
		protected DataProcessingBatchDesc m_pDesc;

		/// <summary>
		/// Enqueues a batch.
		/// </summary>
		/// <param name="desc">the descriptor of the batch.</param>
		/// <returns>true if the batch has been accepted, false otherwise.</returns>
		public bool Enqueue(DataProcessingBatchDesc desc)
		{ 			
			m_ReturnException = null;
			m_pDesc = desc;
			m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                delegate()
                {
                    try
                    {
                        m_ReturnObj = m_Srv.Enqueue(m_pDesc);
                    }
                    catch (Exception x)
                    {
                        m_ReturnException = x;
                    }
                }                
                ));
			m_Thread.Start();
			if (m_Thread.Join(m_Timeout) == false)
			{
				try
				{
					m_Thread.Abort();
				}
				catch (Exception) {}
				m_Srv = null;
				throw new Exception("Communication timeout!");
			}
			if (m_ReturnException != null) throw m_ReturnException;
			return (bool)m_ReturnObj;
		}

		/// <summary>
		/// Checks for execution completion.
		/// </summary>
		/// <param name="id">the id of the batch.</param>
		/// <returns>true if the batch has been completed, false if it is in progress.</returns>
		public bool DoneWith(ulong id)
		{ 			
			m_ReturnException = null;
			m_pId = id;
			m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                delegate()
                {
                    try
                    {
                        m_ReturnObj = m_Srv.DoneWith(m_pId);
                    }
                    catch (Exception x)
                    {
                        m_ReturnException = x;
                    }
                }                                
                ));
			m_Thread.Start();
			if (m_Thread.Join(m_Timeout) == false)
			{
				try
				{
					m_Thread.Abort();
				}
				catch (Exception) {}
				m_Srv = null;
				throw new Exception("Communication timeout!");
			}
			if (m_ReturnException != null) throw m_ReturnException;
			return (bool)m_ReturnObj;
		}

		/// <summary>
		/// Gets the result for a batch.
		/// </summary>
		/// <param name="id">the id of the batch.</param>
		/// <returns>the batch descriptor. It is modified to reflect the batch output. An exception is thrown if the batch terminated with an exception.</returns>
		public DataProcessingBatchDesc Result(ulong id)
		{ 			
			m_ReturnException = null;
			m_pId = id;
			m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                delegate()
                {
                    try
                    {
                        m_ReturnObj = m_Srv.Result(m_pId);
                    }
                    catch (Exception x)
                    {
                        m_ReturnException = x;
                    }
                }                                
                ));
			m_Thread.Start();
			if (m_Thread.Join(m_Timeout) == false)
			{
				try
				{
					m_Thread.Abort();
				}
				catch (Exception) {}
				m_Srv = null;
				throw new Exception("Communication timeout!");
			}
			if (m_ReturnException != null) throw m_ReturnException;
			return (DataProcessingBatchDesc)m_ReturnObj;
		}
		/// <summary>
		/// Builds a new SyncDataProcessingServerWrapper around a DataProcessingServer.
		/// </summary>
		/// <param name="srv">the DataProcessingServer to be wrapped.</param>
		/// <param name="timeout">the communication timeout to be used.</param>
		public SyncDataProcessingServerWrapper(IDataProcessingServer srv, System.TimeSpan timeout)
		{
			m_Srv = srv;
			m_Timeout = timeout;
		}

		/// <summary>
		/// Input slot for TestComm.
		/// </summary>
		protected int m_pCommpar;

        /// <summary>
		/// Tests the communication with the DataProcessingServer.
		/// </summary>
		/// <param name="i">communication parameter.</param>
		/// <returns>2 * commpar - 1 if the DataProcessingServer object and the communication are working properly.</returns>
		public int TestComm(int commpar)
		{
			m_pCommpar = commpar;
			m_ReturnException = null;
			m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(
                delegate()
                {
                    try
                    {
                        m_ReturnObj = m_Srv.TestComm(m_pCommpar);
                    }
                    catch (Exception x)
                    {
                        m_ReturnException = x;
                    }
                }                                
                ));
			m_Thread.Start();
			if (m_Thread.Join(m_Timeout) == false)
			{
				try
				{
					m_Thread.Abort();
				}
				catch (Exception) {}
				m_Srv = null;
				throw new Exception("Communication timeout!");
			}
			if (m_ReturnException != null) throw m_ReturnException;
			return (int)m_ReturnObj;
		}

	}
}
