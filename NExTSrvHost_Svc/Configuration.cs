using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace SySal.Services.NExTSrvHost_Svc
{
    /// <summary>
    /// Configuration of the NExT Service Host.
    /// </summary>
    [Serializable]
    public class Configuration
    {
        /// <summary>
        /// Encrypted DB server name. If this is a zero-length string, other DB-related fields are ignored, and the configuration is set from the file.
        /// </summary>
        public string DBServer = "";
        /// <summary>
        /// Encrypted DB user name.
        /// </summary>
        public string DBUserName = "";
        /// <summary>
        /// Encrypted DB password.
        /// </summary>
        public string DBPassword = "";
        /// <summary>
        /// If empty, the configuration is read from the file; otherwise, it specifies the scalar query that retrieves the configuration from the DB.
        /// </summary>
        /// <remarks>The string _ID_MACHINE_ is replaced with the id of the machine running the query.</remarks>
        public string DBConfigQuery = "";
        /// <summary>
        /// The status of all services is checked periodically after this time interval (in milliseconds).
        /// </summary>
        public int PollingIntervalMS = 1000;
        /// <summary>
        /// Port for the Web interface.
        /// </summary>
        public int WWWPort;
        /// <summary>
        /// Configuration on NExT services.
        /// </summary>        
        public SySal.NExT.NExTConfiguration Services = new SySal.NExT.NExTConfiguration();
    }

}
