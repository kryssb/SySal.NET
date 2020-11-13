using System;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using System.Net;

namespace SySal.Services.OperaDataProcessingServer_CLV
{
	/// <summary>
	/// OperaDataProcessingServer - Command Line Version implementation.
	/// </summary>
	/// <remarks>
	/// <para>The DataProcessingServer is at the foundation of the Computing Infrastructure. Data Processing Servers provide the computing power that is needed for lengthy calculations on a well defined cluster of interchangeable machines.</para>
	/// <para>Some time is required for permission checks and calculation setup. Therefore, use of Data Processing Servers for quick, small units of computing is discouraged.</para>
	/// <para>Data Processing Servers are very well suited for heavy computations, such as fragment linking on large areas or volume reconstructions.</para>
	/// <para>The service stores the credentials for each computation in the default credential record for the user account in which it runs. Therefore, it is strongly recommended to create a user account to run this executable. 
	/// Every computation batch launched will alter the credentials, so it would be very uncomfortable (and prone to security breaches) to run the service in an administrative account, or in a common user account.</para>
	/// <para>The service must be accompanied by a configuration file with the name <c>OperaDataProcessingServer.exe.config</c> in the same directory. This file stores the access credentials to access the DB to validate user requests. 
	/// The configuration file should be shaped as shown in the following example:
	/// <example><code>
	/// &lt;?xml version="1.0" encoding="utf-8" ?&gt; 
	/// &lt;configuration&gt;
	///  &lt;appSettings&gt;
	///   &lt;add key="DBServer" value="OPERADB.SCANNING.MYSITE.EU" /&gt;
	///   &lt;add key="DBUserName" value="DPSMGR" /&gt;
	///   &lt;add key="DBPassword" value="DPSPWD" /&gt;
	///  &lt;/appSettings&gt;
	/// &lt;/configuration&gt;
	/// </code></example>
	/// More parameters are needed for complete configuration of the OperaDataProcessingServer. They can be put in this file, but they can also be stored in the DB site configuration, and this is the recommended practice,
	/// since it allows unified management of the Computing Infrastructure site. These parameters can be put into <c>LZ_SITEVARS</c>, <c>LZ_MACHINEVARS</c>, or in the configuration file. The configuration file overrides any other
	/// setting (and is deprecated, unless used for debugging purposes) and <c>LZ_MACHINEVARS</c> overrides <c>LZ_SITEVARS</c>. Here follows the list of the remaining parameters:
	/// <list type="table">
	/// <listheader><term>Name</term><description>Description</description></listheader>
	/// <item><term>PeakWorkingSetMB</term><description>The maximum working set allowed for a process, in MB. If the server is a dedicated machine, set this number as high as possible, even as high as the total available virtual memory. If the machine is used for other purposes, a recommended value for this parameter is 128 (meaning 128 MB), but this might be too small for some large computation batch.</description></item>
	/// <item><term>MachinePowerClass</term><description>The power class of the machine, i.e. a number in an arbitrary scale of computing power, starting from 0. This is actually used to avoid launching batches that will need too much memory on machines with little RAM, whereas other machine are available. For example, running a batch with a peak working set of 2.3 GB on a machine with 512 MB RAM will take an unacceptable time to complete because of disk swap, even if the processor is very powerful; a machine with 4 GB RAM will take much shorter time. Set this parameter to <c>5</c> if you have not yet defined a power class scale for your site.</description></item>
	/// <item><term>LowPriority</term><description>If set to <c>true</c> (the recommended value), the processes on the DataProcessingServer, and the Server itself, run in low priority, so that interactive use of the machine is still reasonable. Turning this parameter to <c>false</c> is recommended only if this is a dedicated machine with very large RAM.</description></item>
	/// <item><term>ResultLiveSeconds</term><description>The time interval in seconds during which the results of a computation are kept available (this only means that completion information is available; any output files would not be deleted in any case). A recommended value for this parameter to 600 (seconds), corresponding to 10 minutes. The result is forgotten upon expiration of this time.</description></item>
	/// </list>
	/// The recommended place for these settings is the DB machine configuration table (<c>LZ_MACHINEVARS</c>).
	/// </para>
	/// <para>The OperaDataProcessingServer is typically used in a cluster configuration, where it serves as a working machine for a central manager.</para>
	/// <para>Since the OperaDataProcessingServer has no memory of the computations it performs, stopping the service or shutting down the machine results in a loss of data. This is acceptable if the DataProcessingServer is a worker server, because the manager is expected to detect the unavailability of the server and reschedule the computation on another machine.</para>
	/// <para><b>NOTICE: if a computation executable creates temporary files, it is its own responsibility to clean them. OperaDataProcessingServer does not perform disk/disk space maintenance.</b> A full disk is a very common source of problems for DataProcessingServers that continuously hang. If this happens, clean your disk of orphaned temporary files.</para>
	/// <para><b>Installation hints for the OperaDataProcessingServer_CLV executable.</b>
	/// <list type="bullet">
	/// <item><term>Login as an administrator and start the service. Review possible errors in the Event Viewer.</term></item>
	/// <item><term>Ensure the Opera log exists in the Event Viewer. Repeat the previous step if the Opera log does not exist.</term></item>
	/// <item><term>Login as the user that will run the service.</term></item>
	/// <item><term>Try and start the service and ensure it runs normally.</term></item>
	/// <item><term>If you have a firewall installed, unblock the service from gaining access to the network. If you have a DMZ (De-Militarized Zone), the OperaDataProcessingServer should be seen without restrictions in the DMZ and completely hidden on other network interface cards.</term></item>
	/// </list>
	/// </para>
	/// </remarks>
	public class OperaDataProcessingServer
	{
		object ReadOverride(string name, SySal.OperaDb.OperaDbConnection conn)
		{
			object o = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_MACHINEVARS WHERE ID_MACHINE = " + IdMachine + " AND NAME = '" + name + "'", conn, null).ExecuteScalar();
			if (o != null && o != System.DBNull.Value) return o;
			o = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = '" + name + "'", conn, null).ExecuteScalar();
			if (o == null) o = System.DBNull.Value;
			return o;
		}

		/// <summary>
		/// The time duration of a result.
		/// </summary>
		internal static System.TimeSpan ResultLiveTime;

		/// <summary>
		/// Tells whether the computing thread must be in a priority below normal.
		/// </summary>
		internal static bool LowPriority;

		/// <summary>
		/// Connection String for the DB.
		/// </summary>
		internal static string DBServer;

		/// <summary>
		/// User that the OperaBatchServer shall impersonate.
		/// </summary>
		internal static string DBUserName;

		/// <summary>
		/// Password to access the DB.
		/// </summary>
		internal static string DBPassword;

		/// <summary>
		/// Site identifier read from the DB.
		/// </summary>
		internal static long IdSite;

		/// <summary>
		/// Site name read from the DB.
		/// </summary>
		internal static string SiteName;

		/// <summary>
		/// Machine identifier read from the DB.
		/// </summary>
		internal static long IdMachine;

		/// <summary>
		/// Machine address that matches the DB registration entry.
		/// </summary>
		internal static string MachineAddress;

		/// <summary>
		/// Machine name read from the DB.
		/// </summary>
		internal static string MachineName;

		/// <summary>
		/// Peak working set in MB.
		/// </summary>
		internal static int PeakWorkingSetMB;

		/// <summary>
		/// Machine power class.
		/// </summary>
		internal static int MachinePowerClass;

		/// <summary>
		/// The Data Processing Server instance.
		/// </summary>
		protected SySal.DAQSystem.MyDataProcessingServer DPS = null;

		/// <summary>
		/// The event logger.
		/// </summary>
		System.Diagnostics.EventLog EventLog;

		/// <summary>
		/// Creates a new OperaDataProcessingServer service.
		/// </summary>
		public OperaDataProcessingServer()
		{
			EventLog = new EventLog();
		}


		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[MTAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			Console.WriteLine("Opera Data Processing Server - Command Line Version for Debug");
			Console.WriteLine("Service starting.");
			OperaDataProcessingServer MyExe = new OperaDataProcessingServer();
			MyExe.OnStart(args);
			Console.WriteLine("Service started.");
			Console.WriteLine("Self-identification yields:");
			Console.WriteLine("Site: {0} Site Id: {1} Machine: {2} Machine Id: {3} Machine Address: {4}", 
				SiteName, IdSite, MachineName, IdMachine, MachineAddress);
			Console.WriteLine("Type 'exit' and press <return> to stop.");
			while (Console.ReadLine().ToLower() != "exit");
			Console.WriteLine("Service stopping.");
			MyExe.OnStop();
			Console.WriteLine("Service stopped. Bye.");
		}

		/// <summary>
		/// Set things in motion.
		/// </summary>
		protected void OnStart(string[] args)
		{
			try
			{
				int i;
				if (!EventLog.Exists("Opera")) EventLog.CreateEventSource("OperaDataProcServer", "Opera");
				EventLog.Source = "OperaDataProcServer";
				EventLog.Log = "Opera";
             
				System.Configuration.AppSettingsReader asr = new System.Configuration.AppSettingsReader();
                try
                {
                    DBServer = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBServer", typeof(string)));
                    DBUserName = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBUserName", typeof(string)));
                    DBPassword = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBPassword", typeof(string)));
                }
                catch (Exception x)
                {
                    throw new Exception("Encryption error in credentials.\r\nPlease fill in valid encrypted data (you can use OperaDbGUILogin, for instance), or run the service as the appropriate user.");
                }
                SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(DBServer, DBUserName, DBPassword);
				conn.Open();

				IPHostEntry iph = Dns.Resolve(Dns.GetHostName());
				string [] idstr = new string[iph.Aliases.Length + iph.AddressList.Length];
				idstr[0] = iph.HostName;
				for (i = 0; i < iph.Aliases.Length; i++)
					idstr[i] = iph.Aliases[i];
				for (i = 0; i < iph.AddressList.Length; i++)
					idstr[i + iph.Aliases.Length] = iph.AddressList[i].ToString();
				string selstr = "LOWER(TB_MACHINES.ADDRESS)='" + iph.HostName.ToLower() + "'";
				foreach (string s in idstr)
					selstr += (" OR ADDRESS='" + s + "'");
				DataSet ds = new DataSet();
				SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_SITES.ID, TB_SITES.NAME, TB_MACHINES.ID, TB_MACHINES.NAME, TB_MACHINES.ADDRESS FROM TB_SITES INNER JOIN TB_MACHINES ON (TB_MACHINES.ID_SITE = TB_SITES.ID AND TB_MACHINES.ISDATAPROCESSINGSERVER = 1 AND (" + selstr + "))", conn, null);
				da.Fill(ds);
				if (ds.Tables[0].Rows.Count < 1) throw new Exception("Can't find myself in OperaDb registered machines. This service is made unavailable.");
				IdSite = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][0]);
				SiteName = ds.Tables[0].Rows[0][1].ToString();
				IdMachine = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][2]);
				MachineName = ds.Tables[0].Rows[0][3].ToString();
				MachineAddress = ds.Tables[0].Rows[0][4].ToString();

				object val;
				val = ReadOverride("DPS_MachinePowerClass", conn); 
				if (val != System.DBNull.Value) MachinePowerClass = Convert.ToInt32(val.ToString());
				else MachinePowerClass = (int)asr.GetValue("MachinePowerClass", typeof(int));

				val = ReadOverride("DPS_PeakWorkingSetMB", conn); 
				if (val != System.DBNull.Value) PeakWorkingSetMB = Convert.ToInt32(val.ToString());
				else PeakWorkingSetMB = (int)asr.GetValue("PeakWorkingSetMB", typeof(int));

				val = ReadOverride("DPS_ResultLiveSeconds", conn); 
				if (val != System.DBNull.Value) ResultLiveTime = System.TimeSpan.FromSeconds(Convert.ToInt32(val.ToString()));
				else ResultLiveTime = System.TimeSpan.FromSeconds((int)asr.GetValue("ResultLiveSeconds", typeof(int)));
				
				val = ReadOverride("DPS_LowPriority", conn); 
				if (val != System.DBNull.Value) LowPriority = Convert.ToBoolean(val.ToString());
				else LowPriority = (bool)asr.GetValue("LowPriority", typeof(bool));				

				ChannelServices.RegisterChannel(new TcpChannel((int)SySal.DAQSystem.OperaPort.DataProcessingServer));
				DPS = new SySal.DAQSystem.MyDataProcessingServer(EventLog);
				RemotingServices.Marshal(DPS, "DataProcessingServer.rem");
				conn.Close();

				EventLog.WriteEntry("Service starting\r\nThread Priority: " + (LowPriority ? "Below normal" : "Normal") + "\r\nPeak working set (MB): " + PeakWorkingSetMB + "\r\nMachine power class: " + MachinePowerClass + "\r\nSelf-identification yields:\r\nSite: " + SiteName + "\r\nSite Id: " + IdSite + "\r\nMachine: " + MachineName + "\r\nMachine Id: " + IdMachine + "\r\nMachine Address: " + MachineAddress, EventLogEntryType.Information);
			}
			catch (System.Exception x)
			{
				EventLog.WriteEntry("Service startup failure:\n" + x.ToString(), EventLogEntryType.Error);
				throw x;
			}
		}
 
		/// <summary>
		/// Stop this service.
		/// </summary>
		protected void OnStop()
		{
			// TODO: Add code here to perform any tear-down necessary to stop your service.
			EventLog.WriteEntry("Service stopping", EventLogEntryType.Information);
			if (DPS != null)
			{
				RemotingServices.Disconnect(DPS);
				DPS.AbortAllBatches();
			}
		}
	}
}

