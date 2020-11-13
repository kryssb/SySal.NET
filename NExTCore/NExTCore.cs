using System;
using System.Collections.Generic;
using System.Text;
using SySal;
using System.Xml.Serialization;
using System.Runtime.Remoting.Channels;
using System.Security;
[assembly: AllowPartiallyTrustedCallers]

namespace SySal.NExT
{
    /// <summary>
    /// An event, possibly containing data.
    /// </summary>
    [Serializable]
    public class DataEvent
    {        
        /// <summary>
        /// The token that uniquely identifies this data set. A default value is provided, using the system time.
        /// </summary>
        public long EventId = System.DateTime.Now.Ticks;
        /// <summary>
        /// The URI of the object that emitted this event; exceptions must be notified to this URI.
        /// </summary>
        public string Emitter;
    }
    /// <summary>
    /// Request to abort processing a <see cref="DataEvent"/>.
    /// </summary>
    [Serializable]
    public class AbortEvent : DataEvent
    {
        /// <summary>
        /// The id of the event to be aborted.
        /// </summary>
        public long StopId;
    }
    /// <summary>
    /// Information from the completion of processing for a data event.
    /// </summary>
    [Serializable]
    public class DataEventDone
    {
        /// <summary>
        /// Id of the event.
        /// </summary>
        public long EventId;
        /// <summary>
        /// <c>true</c> if the event processing is complete, <c>false</c> otherwise.
        /// </summary>
        public bool Done;
        /// <summary>
        /// Final exception raised in processing the event; if <see cref="Done"/> is <c>false</c>, this is <c>null</c>.
        /// </summary>
        public System.Exception FinalException;
        /// <summary>
        /// Additional information; if <see cref="Done"/> is <c>false</c>, this is <c>null</c>.
        /// </summary>
        public object Info;
    }
    /// <summary>
    /// An exception arising in the context of a NExT processing task.
    /// </summary>
    [Serializable]
    public class NExTException : System.Exception, System.Runtime.Serialization.ISerializable
    {
        /// <summary>
        /// Token of the data set that caused the exception.
        /// </summary>
        public long Token;
        /// <summary>
        /// The URI of the object that raised the exception.
        /// </summary>
        public string RaisedByURI;
        /// <summary>
        /// Builds a new NExTException.
        /// </summary>
        /// <param name="token">token of the dataset.</param>
        /// <param name="raisedby">URI of the object that raised the exception.</param>
        /// <param name="message">exception message.</param>
        public NExTException(long token, string raisedby, string message)
            : base(message)
        {
            Token = token;
            RaisedByURI = raisedby;
        }
        /// <summary>
        /// Builds an empty exception.
        /// </summary>
        public NExTException() { }
        /// <summary>
        /// Deserialization constructor for remoting.
        /// </summary>
        /// <param name="info">serialization information.</param>
        /// <param name="context">the streaming context.</param>
        protected NExTException(System.Runtime.Serialization.SerializationInfo info, System.Runtime.Serialization.StreamingContext context)
            : base(info, context)
        {
            Token = info.GetInt64("Token");
            RaisedByURI = info.GetString("RaisedByURI");
        }


        #region ISerializable Members
        /// <summary>
        /// Serializes the NExTException.
        /// </summary>
        /// <param name="info">serialization information.</param>
        /// <param name="context">the streaming context.</param>
        void System.Runtime.Serialization.ISerializable.GetObjectData(System.Runtime.Serialization.SerializationInfo info, System.Runtime.Serialization.StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("Token", Token);
            info.AddValue("RaisedByURI", RaisedByURI);
        }

        #endregion
    }

    /// <summary>
    /// Gauge of a server monitor. The control used to display the value depends on the type of the value.
    /// </summary>
    [Serializable]
    public struct ServerMonitorGauge
    {
        /// <summary>
        /// The name of the parameter.
        /// </summary>
        public string Name;
        /// <summary>
        /// The value of the parameter.
        /// </summary>
        /// <remarks>The following conversion table applies between value types and controls:
        /// <list type="table">
        /// <listheader><term>Value type</term><description>Control type</description></listheader>
        /// <item><term>Bool</term><description>Check Box</description></item>
        /// <item><term>String</term><description>Edit Box</description></item>
        /// <item><term>Int32</term><description>Edit Box</description></item>
        /// <item><term>Double</term><description>Progress Bar</description></item>
        /// </list>
        /// </remarks>
        public object Value;
    }

    public interface INExTServer
    {
        /// <summary>
        /// Retrieves the list of supported data consumer groups.
        /// </summary>
        string[] DataConsumerGroups { get; }
        /// <summary>
        /// Registers a group of data consumers.
        /// </summary>
        /// <param name="groupname">the name of the group.</param>
        /// <param name="uris">URI's of the group members.</param>
        void RegisterDataConsumerGroup(string groupname, string [] uris);
        /// <summary>
        /// Unregisters a group of data consumers.
        /// </summary>
        /// <param name="groupname">the name of the group to be deleted.</param>
        void UnregisterDataConsumerGroup(string groupname);
        /// <summary>
        /// Called when new data are available to consume.
        /// </summary>
        /// <param name="de">the DataEvent containing the new information.</param>
        /// <returns><c>true</c> if the data can be processed, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>Both immediate and delayed processing may be implemented. In the case of immediate processing, the call 
        /// will not return until processing is complete, and exceptions might be immediately reported. In the case of delayed 
        /// processing, <c>true</c> will be returned if data are only <i><b>formally</b></i> correct, and the actual result will
        /// come from a call to <see cref="DoneWith"/>. Even in the case of immediate processing, calling DoneWith will work.
        /// A robust implementation will anyway switch to DoneWith after a proper time delay elapses, to avoid getting stuck
        /// with a server that failed or crashed or was turned off.
        /// </para>
        /// <para>
        /// In case an exception is thrown, it should be checked whether it is a <see cref="NExTException"/> or not: in the
        /// former case, the exception was due to a problem with the communication or to the NExT infrastructure, so data
        /// correctness is not questioned, and they can be sent to another machine for proper processing; in the latter
        /// case, data have a problem, and it will be pointless to broadcast them again.
        /// </para>
        /// </remarks>
        bool OnDataEvent(DataEvent de);
        /// <summary>
        /// Checks that a data set has been processed. 
        /// </summary>
        /// <param name="Id">the id of the data set.</param>
        /// <returns>information about the completion status of the processing for a data set.</returns>
        DataEventDone DoneWith(long Id);
        /// <summary>
        /// Provides information about the status of the server object.
        /// </summary>
        ServerMonitorGauge[] MonitorGauges { get; }
    }

    /// <summary>
    /// Wraps a remote INExTServer to handle possible disconnections due to network errors or to remote server unavailability.
    /// </summary>
    public sealed class NExTServerSyncWrapper : INExTServer
    {
        INExTServer m_INxS;

        string m_URI;

        int m_TimeOut;

        string m_LogFile;

        /// <summary>
        /// Sets the file where the sync wrapper logs its operations.
        /// </summary>
        public string LogFile
        {
            set { m_LogFile = value; }
        }

        void Log(string text)
        {
            if (m_LogFile != null && m_LogFile.Length > 0)
                try
                {
                    System.IO.File.AppendAllText(m_LogFile, "\r\nSyncWrap(" + m_URI + ") " + System.DateTime.Now + " -> " + text);
                }
                catch (Exception) { }
        }

        /// <summary>
        /// Wraps a remote INExTServer with a specified URI and timeout in milliseconds.
        /// </summary>
        /// <param name="uri">URI of the server to wrap.</param>
        /// <param name="timeout">timeout in milliseconds.</param>
        public NExTServerSyncWrapper(string uri, int timeout)
        {
            m_URI = uri;
            m_INxS = null;
            m_TimeOut = timeout;
        }

        #region INExTServer Members

        /// <summary>
        /// Wraps <see cref="INExTServer.DataConsumerGroups"./>
        /// </summary>
        public string[] DataConsumerGroups 
        {
            get
            {
                System.Exception exc = null;
                string[] ret = null;
                do
                {
                    exc = null;
                    if (m_INxS == null)
                        try
                        {
                            Log("Attempt DataConsumerGroups Connect");
                            m_INxS = (INExTServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(INExTServer), m_URI);
                            Log("OK DataConsumerGroups Connect");
                            System.Threading.Thread ths = new System.Threading.Thread(delegate()
                            {
                                try
                                {
                                    Log("Attempt DataConsumerGroups");
                                    ret = m_INxS.DataConsumerGroups;
                                    Log("OK DataConsumerGroups");
                                }
                                catch (Exception x)
                                {
                                    Log("X DataConsumerGroups: " + x.ToString());
                                    if (x is System.Threading.ThreadAbortException == false)
                                        exc = x;
                                }
                            }
                            );
                            ths.Start();
                            if (ths.Join(m_TimeOut) == false)
                            {
                                Log("Abort DataConsumerGroups");
                                try
                                {
                                    ths.Abort();
                                }
                                catch (Exception) { }
                                m_INxS = null;
                                exc = new Exception();
                            }
                        }
                        catch (Exception x)
                        {
                            Log("X DataConsumerGroups Connect: " + x.ToString());
                            m_INxS = null;
                            exc = x;
                            System.Threading.Thread.Sleep(m_TimeOut);
                            continue;
                        }
                }
                while (exc != null);
                Log("Exit DataConsumerGroups with exc = " + ((exc == null) ? "NULL" : exc.ToString()));
                return ret;
            }
        }
        /// <summary>
        /// Wraps <see cref="INExTServer.RegisterDataConsumerGroup"/>
        /// </summary>
        /// <param name="groupname">the name of the group.</param>
        /// <param name="uris">URI's of the group members.</param>
        public void RegisterDataConsumerGroup(string groupname, string[] uris)
        {
            System.Exception exc = null;
            do
            {
                exc = null;
                if (m_INxS == null)
                    try
                    {
                        m_INxS = (INExTServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(INExTServer), m_URI);
                        System.Threading.Thread ths = new System.Threading.Thread(delegate()
                        {
                            try
                            {
                                m_INxS.RegisterDataConsumerGroup(groupname, uris);
                            }
                            catch (Exception x)
                            {
                                if (x is System.Threading.ThreadAbortException == false)
                                    exc = x;
                            }
                        }
                        );
                        ths.Start();
                        if (ths.Join(m_TimeOut) == false)
                        {
                            try
                            {
                                ths.Abort();
                            }
                            catch (Exception) { }
                            m_INxS = null;
                            exc = new Exception();
                        }
                    }
                    catch (Exception x)
                    {
                        m_INxS = null;
                        exc = x;
                        System.Threading.Thread.Sleep(m_TimeOut);
                        continue;
                    }
            }
            while (exc != null);
        }

        /// <summary>
        /// Unregisters a group of data consumers.
        /// </summary>
        /// <param name="groupname">the name of the group to be deleted.</param>
        public void UnregisterDataConsumerGroup(string groupname)
        {
            System.Exception exc = null;
            do
            {
                exc = null;
                if (m_INxS == null)
                    try
                    {
                        m_INxS = (INExTServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(INExTServer), m_URI);
                        System.Threading.Thread ths = new System.Threading.Thread(delegate()
                        {
                            try
                            {
                                m_INxS.UnregisterDataConsumerGroup(groupname);
                            }
                            catch (Exception x)
                            {
                                if (x is System.Threading.ThreadAbortException == false)
                                    exc = x;
                            }
                        }
                        );
                        ths.Start();
                        if (ths.Join(m_TimeOut) == false)
                        {
                            try
                            {
                                ths.Abort();
                            }
                            catch (Exception) { }
                            m_INxS = null;
                            exc = new Exception();
                        }
                    }
                    catch (Exception x)
                    {
                        m_INxS = null;
                        exc = x;
                        System.Threading.Thread.Sleep(m_TimeOut);
                        continue;
                    }
            }
            while (exc != null);
        }
        
        /// <summary>
        /// Wraps <see cref="INExTServer.OnDataEvent"/>
        /// </summary>
        /// <param name="de">Data to be sent.</param>
        /// <returns><c>true</c> if the data can be processed, <c>false</c> otherwise.</returns>
        public bool OnDataEvent(DataEvent de)
        {
            System.Exception exc = null;
            bool ret = false;
            exc = null;
            if (m_INxS == null)
                try
                {
                    Log("Attempt OnDataEvent Connect");
                    m_INxS = (INExTServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(INExTServer), m_URI);
                    Log("OK OnDataEvent Connect");
                }
                catch (Exception x)
                {
                    Log("X OnDataEvent Connect: " + x.ToString());
                    m_INxS = null;
                    throw new NExTException(de.EventId, m_URI, x.ToString());
                }
            System.Threading.Thread ths = new System.Threading.Thread(delegate()
                    {
                        try
                        {
                            Log("Attempt OnDataEvent");
                            ret = m_INxS.OnDataEvent(de);
                            Log("OK OnDataEvent");
                        }
                        catch (Exception x)
                        {
                            Log("X OnDataEvent: " + x.ToString());
                            if (x is System.Threading.ThreadAbortException == false)
                                exc = x;
                        }
                    }
                    );
            ths.Start();
            if (ths.Join(m_TimeOut) == false)
            {
                try
                {
                    Log("Abort OnDataEvent");
                    ths.Abort();
                }
                catch (Exception) { }
                m_INxS = null;
                throw new NExTException(de.EventId, m_URI, "Timeout (" + m_TimeOut + ")");
            }
            if (exc != null)
            {
                Log("X OnDataEvent: " + exc.ToString());
                if (exc is System.Runtime.Remoting.RemotingException) exc = new NExTException(de.EventId, m_URI, exc.ToString());
                if (exc is NExTException) m_INxS = null;
                throw exc;
            }            
            Log("Exit OnDataEvent with exc=" + ((exc == null) ? "NULL" : exc.ToString()));
            return ret;
        }

        /// <summary>
        /// Wraps <see cref="INExTServer.DoneWith>"/>
        /// </summary>
        /// <param name="Id">the data set to be checked.</param>
        /// <returns>information about the completion status of the processing for a data set.</returns>
        public DataEventDone DoneWith(long Id)
        {
            System.Exception exc = null;
            DataEventDone ret = null;
            exc = null;
            if (m_INxS == null)
                try
                {
                    Log("Attempt DoneWith Connect");
                    m_INxS = (INExTServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(INExTServer), m_URI);
                    Log("OK DoneWith Connect");
                }
                catch (Exception x)
                {
                    Log("X DoneWith Connect: " + x.ToString());
                    m_INxS = null;
                    throw new NExTException(Id, m_URI, x.ToString());
                }
            System.Threading.Thread ths = new System.Threading.Thread(delegate()
                    {
                        try
                        {
                            Log("Attempt DoneWith");
                            ret = m_INxS.DoneWith(Id);
                            Log("OK DoneWith");
                        }
                        catch (Exception x)
                        {
                            Log("X DoneWith: " + x.ToString());
                            if (x is System.Threading.ThreadAbortException == false)
                                exc = x;
                        }
                    }
                    );
            ths.Start();
            if (ths.Join(m_TimeOut) == false)
            {
                try
                {
                    Log("Abort DoneWith");
                    ths.Abort();
                }
                catch (Exception) { }
                m_INxS = null;
                throw new NExTException(Id, "", "Timeout (" + m_TimeOut + ")");
            }
            if (exc != null)
            {
                Log("X DoneWith: " + exc.ToString());
                if (exc is System.Runtime.Remoting.RemotingException) exc = new NExTException(Id, m_URI, exc.ToString());
                if (exc is NExTException) m_INxS = null;
                throw exc;
            }
            Log("Exit DoneWith");
            return ret;
        }

        /// <summary>
        /// Wraps <see cref="INExTServer.MonitorGauges>"/>
        /// </summary>        
        public ServerMonitorGauge[] MonitorGauges
        {
            get {
                System.Exception exc = null;
                ServerMonitorGauge[] ret = null;
                do
                {
                    exc = null;
                    if (m_INxS == null)
                        try
                        {
                            Log("Attempt MonitorGauges Connect");
                            m_INxS = (INExTServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(INExTServer), m_URI);
                            Log("OK MonitorGauges Connect");
                            System.Threading.Thread ths = new System.Threading.Thread(delegate()
                            {
                                try
                                {
                                    Log("Attempt MonitorGauges");
                                    ret = m_INxS.MonitorGauges;
                                    Log("OK MonitorGauges");
                                }
                                catch (Exception x)
                                {
                                    Log("X MonitorGauges: " + x.ToString());
                                    if (x is System.Threading.ThreadAbortException == false)
                                        exc = x;
                                }
                            }
                            );
                            ths.Start();
                            if (ths.Join(m_TimeOut) == false)
                            {
                                try
                                {
                                    Log("Abort MonitorGauges");
                                    ths.Abort();
                                }
                                catch (Exception) { }
                                m_INxS = null;
                                exc = new Exception();
                            }
                        }
                        catch (Exception x)
                        {
                            m_INxS = null;
                            exc = x;
                            System.Threading.Thread.Sleep(m_TimeOut);
                            continue;
                        }
                }
                while (exc != null);
                Log("Exit MonitorGauges with exc=" + ((exc == null) ? "NULL" : exc.ToString()));
                return ret;
            }
        }

        #endregion
    }

    /// <summary>
    /// Local configuration of the host for NExT services.
    /// </summary>
    [Serializable]
    public class NExTConfiguration
    {
        /// <summary>
        /// Parameter to be used in the creation of a server.
        /// </summary>
        [Serializable]
        public class ServerParameter
        {
            /// <summary>
            /// Name of the parameter.
            /// </summary>
            public string Name;
            /// <summary>
            /// Value of the parameter.
            /// </summary>
            public string Value;
        }

        /// <summary>
        /// Describes a <see cref="ServerParameter"/>.
        /// </summary>
        /// <remarks>The <c>Value</c> field defines the default value.</remarks>
        [Serializable]
        public class ServerParameterDescriptor : ServerParameter
        {
            /// <summary>
            /// Describes the meaning of the parameter.
            /// </summary>
            public string Description;
            /// <summary>
            /// Defines the type of the value.
            /// </summary>
            public Type ValueType;
            /// <summary>
            /// Flag that tells whether the parameter can appear as a static parameter.
            /// </summary>
            public bool CanBeStatic;
            /// <summary>
            /// Flag that tells whether the parameter can appear as a dynamic parameter.
            /// </summary>
            public bool CanBeDynamic;
        }

        /// <summary>
        /// A group of data consumers.
        /// </summary>
        [Serializable]
        public class DataConsumerGroup
        {
            /// <summary>
            /// Name of the group.
            /// </summary>
            public string Name;
            /// <summary>
            /// URI's of the data consumers.
            /// </summary>
            public string[] URIs;
            /// <summary>
            /// If <c>true</c>, data are sent and the router waits for batch completion; if <c>false</c>, data are transmitted and forgotten.
            /// </summary>
            public bool WaitForCompletion;
            /// <summary>
            /// In case a data event is refused or a transmission error occurs, retransmission is not attempted before this interval elapses.
            /// </summary>
            public uint RetryIntervalMS;
        }

        /// <summary>
        /// Information about a service.
        /// </summary>
        [Serializable]
        public class ServiceEntry
        {
            /// <summary>
            /// Names of the services (used to form URI's).
            /// </summary>
            /// <remarks>Multiple services with identical parameters can be easily created in place of just one. 
            /// This makes sense when several processors or cores are available.</remarks>
            public string [] Names;
            /// <summary>
            /// <c>true</c> to make the service publicly available on the local netwrok, <c>false</c> otherwise.
            /// </summary>
            public bool Publish;
            /// <summary>
            /// File containing the class that implements the service.
            /// </summary>
            public string CodeFile;
            /// <summary>
            /// The class that implements the service.
            /// </summary>
            public string TypeName;
            /// <summary>
            /// Static server creation parameters.
            /// </summary>
            public ServerParameter[] StaticParameters;
            /// <summary>
            /// List of data consumers for this service.
            /// </summary>
            /// <remarks>Data are broadcast to all groups, and within each group, datasets are sent only to one 
            /// (randomly selected) member of the group. </remarks>
            public DataConsumerGroup [] DataConsumerGroups;
        }
        /// <summary>
        /// Name of the configuration.
        /// </summary>
        public string ConfigurationName;
        /// <summary>
        /// List of the services to be implemented.
        /// </summary>
        public ServiceEntry[] ServiceEntries;
        /// <summary>
        /// Port to use to publish URI's of publicly available services.
        /// </summary>
        public int TCPIPPort;
        /// <summary>
        /// Timeout in milliseconds for remote calls.
        /// </summary>
        public int TimeoutMS;
    }
    
    /// <summary>
    /// Any NExTServer is derived from this class, which provides useful basic functionalities.
    /// </summary>
    public abstract class NExTServer : MarshalByRefObject, INExTServer
    {
        /// <summary>
        /// The log file for this server.
        /// </summary>
        protected string LogFile = "";
        /// <summary>
        /// Writes a string to the log file.
        /// </summary>
        /// <param name="text">the message to be written.</param>
        /// <remarks>If the message cannot be written, no exception is raised.</remarks>             
        protected void Log(string text)
        {
            if (LogFile.Length > 0)
                try
                {
                    System.IO.File.AppendAllText(LogFile + "_" + NExTName + ".log", "\r\n" + System.DateTime.Now + " -> " + text);
                }
                catch (Exception) { }
        }
        /// <summary>
        /// Timeout for automatically generated sync wrappers.
        /// </summary>
        protected static int s_Timeout = 10000;
        /// <summary>
        /// The queue of events to be processed.
        /// </summary>
        protected System.Collections.Generic.Queue<DataEvent> m_DataEvents = new Queue<DataEvent>();
        /// <summary>
        /// Registers a data event in the queue of events to be processed.
        /// </summary>
        /// <param name="de">the data event to register.</param>
        protected void RegisterDataEvent(DataEvent de)
        {
            lock (m_DataEvents)
                m_DataEvents.Enqueue(de);
        }
        /// <summary>
        /// The result of a data event.
        /// </summary>
        protected class DataEventResult 
        {
            /// <summary>
            /// Information about the DataEvent.
            /// </summary>
            public DataEventDone Info;
            /// <summary>
            /// Time when this result expires.
            /// </summary>
            public System.DateTime ExpiryTime;
        }
        /// <summary>
        /// Declares a data event completed.
        /// </summary>
        /// <param name="finalexception">final exception of the data event that is first in the data processing queue.</param>
        /// <param name="additionalinfo">additional information for the data event.</param>
        /// <param name="timeoutms">result duration in ms</param>
        protected void DequeueDataEventAsCompleted(System.Exception finalexception, object additionalinfo, int timeoutms)
        {
            DataEvent de;
            lock (m_DataEvents)
                de = m_DataEvents.Peek();
            DataEventResult der = new DataEventResult();
            der.Info = new DataEventDone();
            der.Info.EventId = de.EventId;
            der.Info.Done = true;
            der.Info.FinalException = finalexception;
            der.Info.Info = additionalinfo;            
            der.ExpiryTime = System.DateTime.Now.AddMilliseconds(timeoutms);
            lock (m_DataEventResults)
                m_DataEventResults.Add(de.EventId, der);
            lock (m_DataEvents)
                m_DataEvents.Dequeue();
        }
        /// <summary>
        /// The list of data event results.
        /// </summary>
        protected System.Collections.Generic.Dictionary<long, DataEventResult> m_DataEventResults = new Dictionary<long, DataEventResult>();
        /// <summary>
        /// Timer that triggers the cleanup of <see cref="m_DataEventResults"/>.
        /// </summary>
        protected System.Timers.Timer m_DataEventKeeper = new System.Timers.Timer(60000);
        /// <summary>
        /// Local NExT server registry.
        /// </summary>
        internal static System.Collections.Generic.Dictionary<string, NExTServer> LocalRegistry = new Dictionary<string, NExTServer>();
        /// <summary>
        /// Initializes the lifetime service.
        /// </summary>
        /// <returns>always <c>null</c> (unlimited lifetime).</returns>
        public override object InitializeLifetimeService()
        {
            return null;
        }
        /// <summary>
        /// Property backer for <see cref="NExTName"/>.
        /// </summary>
        protected string m_NExTName;
        /// <summary>
        /// The name from which the object is known to NExT services.
        /// </summary>
        public string NExTName { get { return (string)m_NExTName.Clone(); } }
        /// <summary>
        /// Static data backer for the <see cref="KnownParameters"/> property.
        /// </summary>
        protected static NExTConfiguration.ServerParameterDescriptor[] s_KnownParameters;
        /// <summary>
        /// Lists the known parameters.
        /// </summary>
        public static NExTConfiguration.ServerParameterDescriptor[] KnownParameters
        {
            get
            {
                NExTConfiguration.ServerParameterDescriptor[] d = new NExTConfiguration.ServerParameterDescriptor[s_KnownParameters.Length];
                s_KnownParameters.CopyTo(d, 0);
                return d;
            }
        }
        /// <summary>
        /// Performs initialization tasks.
        /// </summary>
        static NExTServer() 
        {
            s_KnownParameters = new NExTConfiguration.ServerParameterDescriptor[]
            {
                new NExTConfiguration.ServerParameterDescriptor(),
                new NExTConfiguration.ServerParameterDescriptor()
            };
            s_KnownParameters[0].Name = "MaxQueueLength";
            s_KnownParameters[0].Value = "2";
            s_KnownParameters[0].ValueType = typeof(uint);
            s_KnownParameters[0].CanBeDynamic = s_KnownParameters[0].CanBeStatic = true;
            s_KnownParameters[0].Description = "The maximum length of the queue in consumer groups.";
            s_KnownParameters[1].Name = "LogFile";
            s_KnownParameters[1].Value = "";
            s_KnownParameters[1].ValueType = typeof(string);
            s_KnownParameters[1].CanBeDynamic = s_KnownParameters[1].CanBeStatic = true;
            s_KnownParameters[1].Description = "The file where the server logs its actions.";
        }
        /// <summary>
        /// Converts a string to its value defined by a specified type.
        /// </summary>
        /// <param name="v">the value to be converted.</param>
        /// <param name="t">the type of the parameter.</param>
        /// <returns>the value corresponding to the string.</returns>
        protected static object ConvertValue(string v, Type t)
        {
            if (t == typeof(string)) return v;
            if (t == typeof(int)) return Convert.ToInt32(v, System.Globalization.CultureInfo.InvariantCulture);
            if (t == typeof(uint)) return Convert.ToUInt32(v, System.Globalization.CultureInfo.InvariantCulture);
            if (t == typeof(long)) return Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture);
            if (t == typeof(ulong)) return Convert.ToUInt64(v, System.Globalization.CultureInfo.InvariantCulture);
            if (t == typeof(double)) return Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture);
            throw new Exception("Unsupported data type \"" + t.ToString() + "\".");
        }
        /// <summary>
        /// Interprets the lists of static and dynamic parameters and produces a unique list with known parameters.
        /// </summary>
        /// <param name="staticparams">the list of static parameters.</param>
        /// <param name="dynamicparams">the list of dynamic parameters.</param>
        /// <returns>a list of values for the known parameters.</returns>
        protected static System.Collections.Generic.Dictionary<string, object> InterpretParameters(NExTConfiguration.ServerParameter[] staticparams, NExTConfiguration.ServerParameter[] dynamicparams)
        {
            NExTConfiguration.ServerParameterDescriptor[] descp = KnownParameters;
            System.Collections.Generic.Dictionary<string, object> outl = new Dictionary<string, object>();
            foreach (NExTConfiguration.ServerParameterDescriptor dep in descp)
                outl.Add(dep.Name, ConvertValue(dep.Value, dep.ValueType));
            foreach (NExTConfiguration.ServerParameterDescriptor dep in descp)
                if (dep.CanBeStatic)
                    foreach (NExTConfiguration.ServerParameter sp in staticparams)
                        if (String.Compare(sp.Name, dep.Name, true) == 0)
                            outl[dep.Name] = ConvertValue(sp.Value, dep.ValueType);
            foreach (NExTConfiguration.ServerParameterDescriptor dep in descp)
                if (dep.CanBeDynamic)
                    foreach (NExTConfiguration.ServerParameter sp in dynamicparams)
                        if (String.Compare(sp.Name, dep.Name, true) == 0)
                            outl[dep.Name] = ConvertValue(sp.Value, dep.ValueType);
            return outl;
        }
        /// <summary>
        /// Protected constructor that performs common NExT initialization.
        /// </summary>
        /// <param name="name">the name of the server.</param>
        /// <param name="publish"><c>true</c> if the server is to be made publicly available over the network, <c>false</c> otherwise.</param>        
        /// <param name="staticserverparams">the list of static server creation parameters.</param>
        /// <param name="dynamicserverparams">the list of dynamic server creation parameters.</param>
        /// <remarks>The following parameters are currently understood both as static and dynamic:
        /// <list type="table">
        /// <listheader><term>Parameter</term><description>Meaning</description></listheader>
        /// <item><term>MaxQueueLength</term><description>The maximum number of data events that can be in the queue.</description></item>
        /// </list>
        /// </remarks>
        protected NExTServer(string name, bool publish, NExTConfiguration.ServerParameter [] staticserverparams, NExTConfiguration.ServerParameter [] dynamicserverparams)
        {
#if !(DEBUG)
            try
            {
#endif
                lock (LocalRegistry)
                {
                    System.Collections.Generic.Dictionary<string, object> sp = InterpretParameters(staticserverparams, dynamicserverparams);
                    LogFile = (string)sp["LogFile"];
                    uint maxqueuelength = 2;
                    try
                    {
                        maxqueuelength = (uint)sp["MaxQueueLength"];
                    }
                    catch (Exception) { }
                    if (LocalRegistry.ContainsKey(name)) throw new Exception("Server already exists.");
                    m_NExTName = name;
                    string[] dgs = this.DataConsumerGroups;                    
                    ConsumerGroupRouters = new ConsumerGroupRouter[dgs.Length];
                    int i;
                    for (i = 0; i < ConsumerGroupRouters.Length; i++)
                        ConsumerGroupRouters[i] = new ConsumerGroupRouter(dgs[i], (int)maxqueuelength, LogFile);
                    LocalRegistry.Add(name, this);
                    if (publish) System.Runtime.Remoting.RemotingServices.Marshal(this, m_NExTName);                    
                }
                m_DataEventKeeper.Elapsed += new System.Timers.ElapsedEventHandler(m_DataEventKeeper_Elapsed);
                m_DataEventKeeper.Start();
#if !(DEBUG)
            }
            catch (Exception x)
            {
                lock (LocalRegistry)
                {
                    if (LocalRegistry.ContainsKey(name)) LocalRegistry.Remove(m_NExTName);
                }
                throw x;
            }
#endif

        }

        void m_DataEventKeeper_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            System.DateTime now = System.DateTime.Now;
            lock (m_DataEventResults)
                foreach (System.Collections.Generic.KeyValuePair<long, DataEventResult> der in m_DataEventResults)
                    if (der.Value.ExpiryTime < now)
                        m_DataEventResults.Remove(der.Key);
        }

        ~NExTServer()
        {
            lock (LocalRegistry)
            {
                try
                {
                    System.Runtime.Remoting.RemotingServices.Disconnect(this);
                }
                catch (Exception) { }
                if (LocalRegistry.ContainsKey(m_NExTName)) LocalRegistry.Remove(m_NExTName);
            }
        }

        /// <summary>
        /// The first characters of remote URI's.
        /// </summary>
        public const string URITcpString = "tcp:";
        /// <summary>
        /// Gets a NExT server from its URI.
        /// </summary>
        /// <param name="uri">the URI of the server being sought.</param>
        /// <param name="dataconsumergroups">groups of data consumers.</param>
        /// <returns>the <see cref="INExTServer"/> interface for the server, or an exception if it cannot be found.</returns>        
        public static INExTServer NExTServerFromURI(string uri)
        {
            if (uri.StartsWith(URITcpString))
            {
                return new NExTServerSyncWrapper(uri, s_Timeout);
            }
            else
                lock (LocalRegistry)
                {
                    return (INExTServer)LocalRegistry[uri];
                }
        }
        /// <summary>
        /// Deletes all objects from the local registrar; normally this should be enough to destroy them.
        /// </summary>
        public static void Cleanup()
        {
            lock (LocalRegistry)
            {
                foreach (INExTServer nxe in LocalRegistry.Values)
                {
                    string[] dg = nxe.DataConsumerGroups;
                    foreach (string s in dg)
                        nxe.UnregisterDataConsumerGroup(s);
                }
                LocalRegistry.Clear();
            }
        }

        /// <summary>
        /// Sets up the servers and connections listed in a configuration.
        /// </summary>
        /// <param name="cfg">the configuration to be realized.</param>
        /// <param name="dynamicserverparams">the list of the dynamic server parameters.</param>
        public static void SetupConfiguration(NExTConfiguration cfg, NExTConfiguration.ServerParameter [] dynamicserverparams)
        {
            Cleanup();
            s_Timeout = cfg.TimeoutMS;
            if (dynamicserverparams == null) dynamicserverparams = new NExTConfiguration.ServerParameter[0];
            if (cfg.TCPIPPort > 0)
            {
                System.Runtime.Remoting.Channels.IChannel[] iregs = System.Runtime.Remoting.Channels.ChannelServices.RegisteredChannels;
                bool registered = false;
                foreach (System.Runtime.Remoting.Channels.IChannel ich in iregs)
                    if (String.Compare(ich.ChannelName, "tcp", true) == 0)
                        registered = true;
                if (registered == false) System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new System.Runtime.Remoting.Channels.Tcp.TcpChannel(cfg.TCPIPPort));
            }
            string codefiledir = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
            codefiledir = codefiledir.Remove(Math.Max(codefiledir.LastIndexOf('\\'), codefiledir.LastIndexOf('/')) + 1);            
            lock (LocalRegistry)
            {
                foreach (NExTConfiguration.ServiceEntry se in cfg.ServiceEntries)
                {
                    if (se.StaticParameters == null) se.StaticParameters = new NExTConfiguration.ServerParameter[0];
                    foreach (string name in se.Names)
                        System.Activator.CreateInstanceFrom(codefiledir + se.CodeFile, se.TypeName, false, System.Reflection.BindingFlags.CreateInstance, null, new object[] { name, se.Publish, se.StaticParameters, dynamicserverparams }, System.Globalization.CultureInfo.InvariantCulture, new object[0], null);
                }
                foreach (NExTConfiguration.ServiceEntry se in cfg.ServiceEntries)
                {
                    foreach (string name in se.Names)
                    {
                        INExTServer srv = NExTServerFromURI(name);
                        foreach (NExTConfiguration.DataConsumerGroup dcg in se.DataConsumerGroups)
                            srv.RegisterDataConsumerGroup(dcg.Name, (dcg.URIs == null) ? new string[0] : dcg.URIs);
                    }
                }
            }
        }

        #region INExTServer Members
        /// <summary>
        /// Lists the data consumer groups.
        /// </summary>
        public abstract string[] DataConsumerGroups { get; }
        /// <summary>
        /// Sets the list of servers of a data consumer group.
        /// </summary>
        /// <param name="groupname">the name of the data consumer group.</param>
        /// <param name="uris">the list of servers to be registered.</param>
        public virtual void RegisterDataConsumerGroup(string groupname, string[] uris)
        {
            foreach (ConsumerGroupRouter cbg in ConsumerGroupRouters)
                if (String.Compare(cbg.GroupName, groupname, true) == 0)
                {
                    cbg.URIs = uris;
                    return;
                }
            throw new Exception("No group named \"" + groupname + "\" supported.");
        }
        /// <summary>
        /// Unregisters all servers in a data consumer group.
        /// </summary>
        /// <param name="groupname">the group to be reset.</param>
        public virtual void UnregisterDataConsumerGroup(string groupname)
        {
            foreach (ConsumerGroupRouter cbg in ConsumerGroupRouters)
                if (String.Compare(cbg.GroupName, groupname, true) == 0)
                {
                    cbg.URIs = new string[0];
                    return;
                }
            throw new Exception("No group named \"" + groupname + "\" supported.");
        }
        /// <summary>
        /// Processes data.
        /// </summary>
        public abstract bool OnDataEvent(DataEvent de);
        /// <summary>
        /// Checks whether a dataset event is done.
        /// </summary>
        public virtual DataEventDone DoneWith(long id)
        {
            DataEventDone ded = new DataEventDone();
            ded.EventId = id;
            ded.Done = false;
            try
            {
                lock (m_DataEvents)
                    foreach (DataEvent de in m_DataEvents)
                        if (de.EventId == id)
                            return ded;
                lock (m_DataEventResults)
                    foreach (System.Collections.Generic.KeyValuePair<long, DataEventResult> der in m_DataEventResults)
                        if (der.Key == id)
                        {
                            return der.Value.Info;
                        }
                throw new NExTException(id, this.m_NExTName, "Unknown data event.");
            }
            catch (Exception xc)
            {
                Log("X Inner DoneWith: " + xc.ToString());
                throw xc;
            }
        }
        /// <summary>
        /// Accesses monitoring information.
        /// </summary>
        public abstract ServerMonitorGauge[] MonitorGauges { get; }

        #endregion
        /// <summary>
        /// Routers for the consumer groups.
        /// </summary>
        protected ConsumerGroupRouter [] ConsumerGroupRouters;
        /// <summary>
        /// Basic services to broadcast data to a consumer group.
        /// </summary>
        protected class ConsumerGroupRouter
        {
            /// <summary>
            /// Writes a string to the log file.
            /// </summary>
            /// <param name="text">the message to be written.</param>
            /// <remarks>If the message cannot be written, no exception is raised.</remarks>             
            protected void Log(string text)
            {
                if (m_LogFile.Length > 0)
                    try
                    {
                        System.IO.File.AppendAllText(m_LogFile + "_" + GroupName + ".log", "\r\n" + System.DateTime.Now + " -> " + text);
                    }
                    catch (Exception) { }
            }
            /// <summary>
            /// Delegate to a method that checks the affinity between URI's and data events.
            /// </summary>
            /// <param name="uri">the URI of the consumer that is expected to receive the data event.</param>
            /// <param name="de">the data event to be sent.</param>
            /// <returns><c>true</c> if a data event can be sent to a specific URI, <c>false</c> otherwise.</returns>
            public delegate bool dCheckAffinity(string uri, DataEvent de);
            /// <summary>
            /// Property backer of <see cref="AffinityChecker"/>.
            /// </summary>
            protected dCheckAffinity m_AffinityChecker = null;
            /// <summary>
            /// Delegate to the method that checks the affinity between data events and URI's.
            /// </summary>
            /// <remarks>If this is <c>null</c>, all URI's are considered eligible for all data events.</remarks>
            public dCheckAffinity AffinityChecker
            {                
                get { return m_AffinityChecker; }
                set { m_AffinityChecker = value; }                
            }
            /// <summary>
            /// Delegate to a method that checks whether a data event should be rerouted.
            /// </summary>
            /// <param name="ded">data event completion information.</param>            
            /// <returns><c>true</c> if the data event should be sent again, <c>false</c> otherwise.</returns>
            public delegate bool dShouldReroute(DataEventDone ded);
            /// <summary>
            /// Property backer for <see cref="ShouldReroute"/>.
            /// </summary>
            protected dShouldReroute m_ShouldReroute = null;
            /// <summary>
            /// Delegate to the method that checks whether a data event should be rerouted.
            /// </summary>
            /// <remarks>If this is <c>null</c>, only events <c>DoneWith</c> fails with an exception are rerouted.</remarks>
            public dShouldReroute ShouldReroute
            {
                get { return m_ShouldReroute; }
                set { m_ShouldReroute = value; }
            }
            /// <summary>
            /// The name of the group.
            /// </summary>
            public string GroupName;

            string[] m_URIs;
            INExTServer [] m_Servers;
            DataEvent[] m_ServerSlots;
            System.Collections.ArrayList m_DataQueue;
            string m_LogFile = "";
            
            System.Threading.Thread[] m_ExecThreads;
            System.Threading.AutoResetEvent m_DataReady;
            /// <summary>
            /// Property backer for <see cref="WaitForCompletion"/>.
            /// </summary>
            protected bool m_WaitForCompletion;
            /// <summary>
            /// If <c>true</c>, the router waits for completion; otherwise, data are transmitted and then forgotten.
            /// </summary>
            public bool WaitForCompletion
            {
                get { return m_WaitForCompletion; }
                set { m_WaitForCompletion = value; }
            }
            /// <summary>
            /// Property backer for <see cref="MaxQueueLength"/>.
            /// </summary>
            protected int m_MaxQueueLength = 0;
            /// <summary>
            /// Max length of the data distribution queue.
            /// </summary>
            public int MaxQueueLength { get { return m_MaxQueueLength; } }
            /// <summary>
            /// Builds a new router with the specified group name and queue length.
            /// </summary>
            /// <param name="groupname">the name of the consumer group.</param>
            /// <param name="maxqueuelength">the maximum queue length.</param>
            /// <param name="logfile">the file to log actions to.</param>
            public ConsumerGroupRouter(string groupname, int maxqueuelength, string logfile)
            {
                GroupName = groupname;
                m_MaxQueueLength = maxqueuelength;
                m_Servers = new INExTServer[0];
                m_ServerSlots = new DataEvent[0];
                m_DataQueue = new System.Collections.ArrayList();
                m_ExecThreads = new System.Threading.Thread[0];
                m_DataReady = new System.Threading.AutoResetEvent(false);
                m_LogFile = logfile;
                m_WaitForCompletion = false;
            }

            /// <summary>
            /// The list of URI's to provide data to.
            /// </summary>
            public string[] URIs
            {
                get { return m_URIs; }
                set
                {
                    foreach (System.Threading.Thread ths in m_ExecThreads)
                        try
                        {
                            ths.Abort();
                            ths.Join();
                        }
                        catch (Exception) { }
                    m_DataQueue.Clear();
                    m_Servers = new INExTServer[0];
                    m_ServerSlots = new DataEvent[0];
                    m_ExecThreads = new System.Threading.Thread[0];
                    m_DataReady.Reset();
                    m_URIs = new string[0];

                    m_URIs = value;
                    m_Servers = new INExTServer[m_URIs.Length];
                    m_ServerSlots = new DataEvent[m_URIs.Length];
                    int i;
                    for (i = 0; i < m_Servers.Length; i++)
                    {
                        m_Servers[i] = NExTServerFromURI(m_URIs[i]);
                        if (m_Servers[i] is NExTServerSyncWrapper)
                            (m_Servers[i] as NExTServerSyncWrapper).LogFile = m_LogFile;
                    }
                    m_ExecThreads = new System.Threading.Thread[m_Servers.Length];
                    for (i = 0; i < m_ExecThreads.Length; i++)
                        (m_ExecThreads[i] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(ExecThread))).Start(i);
                }
            }

            void ExecThread(object o)
            {
                int Index = (int)o;
                INExTServer W = (INExTServer)m_Servers[Index];
                while (true)
                {
                    Log("DataReady wait start");
                    m_DataReady.WaitOne(1000);
                    DataEvent de = null;
                    Log("DataReady wait end");
                    lock (m_DataQueue)
                    {
                        Log("DataQueue count: " + m_DataQueue.Count);
                        if (m_DataQueue.Count > 0)
                        {
                            int i;
                            for (i = 0; i < m_DataQueue.Count; i++)
                            {
                                Log("CheckAffinity: " + ((DataEvent)m_DataQueue[i]).EventId + " URI: " + m_URIs[Index]);
                                Log("m_AffinityChecker == null? " + (m_AffinityChecker == null));
                                if ((m_AffinityChecker == null) || m_AffinityChecker(m_URIs[Index], (DataEvent)m_DataQueue[i]))
                                {
                                    Log("CheckAffinity OK");
                                    de = (DataEvent)m_DataQueue[i];
                                    m_ServerSlots[Index] = de;
                                    m_DataQueue.RemoveAt(i);
                                    Log("DataQueue count: " + m_DataQueue.Count);
                                    if (m_DataQueue.Count > 0)
                                        m_DataReady.Set();
                                    break;
                                }
                            }
                        }
                    }
                    if (de != null)
                    {
                        DataEventDone ded = null;
                        bool reroute = false;
                        try
                        {
                            Log("OnDataEvent");
                            if (W.OnDataEvent(de) == false) throw new NExTException();
                            Log("DoneWith (WaitForCompletion = " + m_WaitForCompletion + ")");
                            if (m_WaitForCompletion)
                                while ((ded = W.DoneWith(de.EventId)).Done == false)
                                    System.Threading.Thread.Sleep(s_Timeout);
                            m_ServerSlots[Index] = null;
                        }
                        catch (Exception x)
                        {
                            Log("X type: " + x.GetType().ToString() + "\r\n" + x.ToString());
                            Log("ServerSlots[" + Index + "] = " + ((m_ServerSlots[Index] == null) ? "NULL" : m_ServerSlots[Index].EventId.ToString()));
                            if (m_ServerSlots[Index] == null) continue; // aborted events are not rerouted.
                            reroute = true;
                        }
                        if (reroute == false) reroute = m_WaitForCompletion && ((ded == null) || ((m_ShouldReroute != null) && m_ShouldReroute(ded)));
                        if (reroute)
                            lock (m_DataQueue)
                            {
                                Log("Reroute");
                                m_ServerSlots[Index] = null;
                                m_DataQueue.Add(de);
                                m_DataReady.Set();
                            }
                        else
                            lock (m_DataQueue)
                            {
                                m_ServerSlots[Index] = null;
                            }
                    }
                }
            }

            /// <summary>
            /// Routes a data event to all URI's in this group.
            /// </summary>
            /// <param name="de">the data event to be routed.</param>
            /// <returns><c>true</c> if successful, <c>false</c> otherwise.</returns>
            /// <remarks>If the queue is full, this method waits until it gets empty, hence it may take a long time to complete.</remarks>
            public bool RouteDataEvent(DataEvent de)
            {
                if (m_URIs == null || m_URIs.Length == 0) return true;
                while (true)
                    lock (m_DataQueue)
                    {
                        if (m_DataQueue.Count < m_MaxQueueLength)
                        {
                            m_DataQueue.Add(de);
                            m_DataReady.Set();
                            return true;
                        }
                        else System.Threading.Thread.Sleep(1000);
                    }
            }
            /// <summary>
            /// Retrieves the list of the data events that are being routed.
            /// </summary>
            public long[] DataEventQueue
            {
                get
                {
                    System.Collections.ArrayList arr = new System.Collections.ArrayList();
                    lock (DataEventQueue)
                    {
                        int i;
                        for (i = 0; i < m_ServerSlots.Length; i++)
                            if (m_ServerSlots[i] != null)
                                arr.Add(m_ServerSlots[i].EventId);
                        long[] ret = new long[arr.Count + DataEventQueue.Length];
                        for (i = 0; i < arr.Count; i++)
                            ret[i] = (long)arr[i];
                        DataEventQueue.CopyTo(ret, arr.Count);
                        return ret;
                    }

                }
            }
            /// <summary>
            /// Cancels routing a data event.
            /// </summary>
            /// <param name="eventid">the id of the data event that must be removed from the queue.</param>            
            /// <returns><c>true</c> if the event was successfully removed from the queue, <c>false</c> otherwise.</returns>
            public bool CancelRouting(long eventid)
            {
                AbortEvent StopEvent = new AbortEvent();
                StopEvent.Emitter = "";
                StopEvent.StopId = eventid;
                lock (m_DataQueue)
                {
                    int i;
                    for (i = 0; i < m_DataQueue.Count; i++)
                        if (((DataEvent)m_DataQueue[i]).EventId == eventid)
                        {
                            m_DataQueue.RemoveAt(i--);
                            return true;
                        }
                    for (i = 0; i < m_ServerSlots.Length; i++)
                        if (m_ServerSlots[i] != null && m_ServerSlots[i].EventId == eventid)
                        {
                            try
                            {
                                m_Servers[i].OnDataEvent(StopEvent);
                                m_ServerSlots[i] = null;
                                return true;
                            }
                            catch (Exception)
                            {
                                continue;
                            }
                        }
                    if (m_DataQueue.Count > 0) m_DataReady.Set();
                    return true;
                }
            }
        }
    }

    namespace Test
    {
        [Serializable]
        public class CountDataEvent : DataEvent
        {
            public int Value;
            public CountDataEvent(int val) { Value = val; }
        }

        public class SumServer : NExTServer
        {            
            int m_Sum;

            static SumServer()
            {
                NExTConfiguration.ServerParameterDescriptor[] d = new NExTConfiguration.ServerParameterDescriptor[2];
                d[0] = s_KnownParameters[0];
                d[1] = new NExTConfiguration.ServerParameterDescriptor();
                d[1].Name = "StartValue";
                d[1].Value = "0";
                d[1].ValueType = typeof(int);
                d[1].Description = "The start value of the sum.";
                d[1].CanBeDynamic = d[1].CanBeStatic = true;
                s_KnownParameters = d;
            }

            public SumServer(string name, bool publish, NExTConfiguration.ServerParameter[] staticparams, NExTConfiguration.ServerParameter[] dynparams)
                : base(name, publish, staticparams, dynparams)
            {
                m_Sum = (int)(InterpretParameters(staticparams, dynparams)["StartValue"]);
            }

            public override ServerMonitorGauge[] MonitorGauges
            {
                get
                {
                    ServerMonitorGauge g1 = new ServerMonitorGauge();
                    g1.Name = "Sum";
                    g1.Value = m_Sum;
                    ServerMonitorGauge g2 = new ServerMonitorGauge();
                    g2.Name = "ProcQueue";
                    g2.Value = m_DataEvents.Count;
                    ServerMonitorGauge g3 = new ServerMonitorGauge();
                    g3.Name = "ResQueue";
                    g3.Value = m_DataEventResults.Count;
                    return new ServerMonitorGauge[] { g1, g2, g3 };
                }
            }

            public override bool OnDataEvent(DataEvent de)
            {
                if (de is CountDataEvent)
                {
                    RegisterDataEvent(de);
                    m_Sum += ((CountDataEvent)de).Value;
                    DequeueDataEventAsCompleted(null, null, 10000);
                    return true;
                }
                return false;
            }

            public override string[] DataConsumerGroups
            {
                get
                {
                    return new string[] { "Even", "Odd" };
                }
            }
        }

        public class TriggerServer : NExTServer
        {
            System.Random m_Rnd = new Random();

            int m_Count = 0;

            System.Timers.Timer m_Timer = new System.Timers.Timer();

            public TriggerServer(string name, bool publish, NExTConfiguration.ServerParameter [] staticparams, NExTConfiguration.ServerParameter [] dynparams) : base(name, publish, staticparams, dynparams) 
            {
                m_Timer = new System.Timers.Timer(5000.0);
                m_Timer.Elapsed += new System.Timers.ElapsedEventHandler(m_Timer_Elapsed);
                m_Timer.Start();
            }

            void m_Timer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
            {
                m_Count++;
                if (ConsumerGroupRouters != null && ConsumerGroupRouters[0] != null)
                {                    
                    ConsumerGroupRouters[0].RouteDataEvent(new CountDataEvent(m_Rnd.Next()));
                }
            }

            public override ServerMonitorGauge[] MonitorGauges
            {
                get
                {
                    ServerMonitorGauge g = new ServerMonitorGauge();
                    g.Name = "Triggers";
                    g.Value = m_Count;
                    return new ServerMonitorGauge[] { g };
                }
            }

            public override bool OnDataEvent(DataEvent de)
            {
                return false;
            }

            public override string[] DataConsumerGroups
            {
                get
                {
                    return new string[] { "TriggerUsers" };
                }
            }
        }
    }
}
