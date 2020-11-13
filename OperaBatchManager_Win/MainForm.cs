using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using System.Diagnostics;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using System.Net;
using System.Text.RegularExpressions;


namespace SySal.Services.OperaBatchManager_Win
{
	/// <summary>
	/// OperaBatchManager_Win - GUI implementation of the BatchManager and cluster DataProcessing services.
	/// </summary>
	/// <remarks>
	/// <para>This executable hosts both an implementation of BatchManager and a manager of DataProcessing servers. The manager itself does no processing, but it allocates work to the worker DataProcessingServers, so if one of them is taken offline its work is reallocated to other available machines.</para>
	/// <para>
	/// The MainForm is the startup object. It creates and launches the BatchManager and DataProcessingServer services. MainForm supports the following actions on process operations:
	/// <list type="table">
	/// <listheader><term>Action</term><description>Explanation</description></listheader>
	/// <item><term>Start</term><description>creates a new process operation. Opens <see cref="SySal.Services.OperaBatchManager_Win.StartForm">StartForm</see> to set the needed parameters.</description></item>
	/// <item><term>Pause</term><description>pauses the currently selected process operation. No specific credentials are needed when accessing this BatchManager directly from its console. <b>NOTICE: if you want to suspend a process operation even for a long time, but you think it will be resumed in the future, use "Pause", not "Abort".</b></description></item>
	/// <item><term>Resume</term><description>resumes the currently selected process operation. No specific credentials are needed when accessing this BatchManager directly from its console.</description></item>
	/// <item><term>Interrupt</term><description>opens the <see cref="SySal.Services.OperaBatchManager_Win.InterruptForm">InterruptForm</see> to enter a string of data to be sent as an interrupt to the currently selected process operation. No specific credentials are needed when accessing this BatchManager directly from its console.</description></item>
	/// <item><term>Abort</term><description>aborts the currently selected process operation. <b>CAUTION: an aborted operation cannot be resumed in the future: it is closed forever. If you mean to continue a process operation, use "Pause" instead.</b></description></item>
	/// <item><term>Reconfig</term><description>Reloads the configuration for a process operation. It is intended for use in testing environments, when configurations can be changed in the DB.</description></item>
	/// <item><term>DPX Restart</term><description>Restarts the Execute thread of the DataProcessingServer. The thread may be closed in severe malfunction cases, when the Event Log is full and limited disk space is available. Using this function makes sense only after the cause of malfunction is detected and removed. Clicking this button when it's not really needed makes no harm: if the thread is properly running, no action is taken.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// The MainForm takes care of reading the configuration file and/or retrieving settings from the DB. Here is a sample configuration file.
	/// <example>
	/// <code>
	/// &lt;?xml version="1.0" encoding="utf-8" ?&gt; 
	/// &lt;configuration&gt;
	///  &lt;appSettings&gt;
	///   &lt;add key="DBServer" value="OPERADB" /&gt;
	///   &lt;add key="DBUserName" value="BATCHMGR" /&gt;
	///   &lt;add key="DBPassword" value="MGRPWD" /&gt;
	///   &lt;add key="OPERAUserName" value="batchsrv" /&gt;
	///   &lt;add key="OPERAPassword" value="opmgrpwd" /&gt;
	///  &lt;/appSettings&gt;
	/// &lt;/configuration&gt;
	/// </code>
	/// </example>
	/// </para>
	/// <para>
	/// There are other parameters that can be read from global site variables in the DB (<c>LZ_SITEVARS</c>), machine variables in the DB (<c>LZ_MACHINEVARS</c>) or from the application configuration file. 
	/// As a rule, the more local setting overrides the others. In general, the preferred practice should be to use <c>LZ_SITEVARS</c>, then to use <c>LZ_MACHINEVARS</c>, and the application configuration file should only be used for debugging purposes: 
	/// indeed, storing the settings into the DB allows the administrator to manage them in a unified way; scattering them over local files makes it difficult to track changes and settings, and may lead to puzzling situations.
	/// </para>
	/// <para>
	/// Here follows the list of the parameters needed for complete configuration of this BatchManager and DataProcessingServer.
	/// <list type="table">
	/// <listheader><term>Parameter</term><description>Explanation</description></listheader>
	/// <item><term><c>BM_DataProcSrvMonitorInterval</c></term><description>Time interval in seconds between two polling queries to a worker DataProcessingServer in the cluster managed by this DataProcessingServer.</description></item>
	/// <item><term><c>BM_ImpersonateBatchUser</c></term><description>If true, the DataProcessingServer uses the calling user's credentials to run the processing tasks; if false, the BatchManager's credentials are used. <b>CAUTION: since the BatchManager credentials allow processes to write to the OPERA DB, it is recommended that this variable be always set to true, except for temporary debugging purposes.</b></description></item>
	/// <item><term><c>BM_ResultLiveSeconds</c></term><description>Time interval in seconds that defines how long a processing result will be kept in memory before being forgotten.</description></item>
	/// <item><term><c>ExeRepository</c></term><description>Full path (usually a network path) to the directory where processing executables and commonly used assemblies are hosted.</description></item>
	/// <item><term><c>ScratchDir</c></term><description>Full path (usually a network path) to the directory to be used as a scratch area for temporary files.</description></item>
	/// <item><term><c>RawDataDir</c></term><description>Full path (usually a network path) to the directory where raw data are stored.</description></item>
	/// <item><term><c>TaskDir</c></term><description>Full path (<b>it is strongly recommended to use a local, non-shared path</b>) to the directory where task files are stored. See below for a discussion of task files.</description></item>
	/// <item><term><c>ArchivedTaskDir</c></term><description>Full path to the directory where task files of completed/aborted tasks are archived.</description></item>	
	/// <item><term><c>DriverDir</c></term><description>Full path (<b>it is strongly recommended to use a local, non-shared path</b>) where driver executables are found.</description></item>
	/// </list>
	/// In common practice, <c>ExeRepository</c>, <c>ScratchDir</c> and <c>RawDataDir</c> are network paths. For obvious safety and security reasons, <c>TaskDir</c>, <c>ArchivedTaskDir</c> and <c>DriverDir</c> should be local paths, not reachable through network access.
	/// </para>
	/// <para><b>Task files</b></para>
	/// <para>For each process operation that is running or paused, the BatchManager keeps track of the current status of the process by means of 4 files:
	/// <list type="bullet">
	/// <item><term>the <c>.startup</c> file</term></item>
	/// <item><term>the <c>.progress</c> file</term></item>
	/// <item><term>the <c>.progress_backup</c> file</term></item>
	/// <item><term>the <c>.interrupts</c> file</term></item>
	/// </list>
	/// </para>
	/// <para>
	/// The startup file (<c>.startup</c>) holds startup information about the process operation. Most of this information is duplicated in the DB, but some specific driver
	/// might need additional parameters that do not fit in the current DB schema. 
	/// This file is cached in the HostEnv for the process operation driver, so it is written when the process operation is created, and read only if the BatchManager restarts after shutdown.
	/// If this file is missing, the process operation cannot be restarted, and the BatchManager restart process is aborted. It should <b><u>never</u></b> be changed by hand.
	/// </para>
	/// <para>
	/// The progress files (<c>.progress</c> and <c>.progress_backup</c>) are a pair of files that hold the same information. They are duplicated because they are continuously overwritten,
	/// and sometimes even the BatchManager administrator might modify them by hand (this is a deprecated practice, but it is the only way to "help" some old drivers that do not support interrupts). 
	/// They are cached in the BatchManager memory, and are actually read only when the OperaBatchManager restarts. When doing so, if the <c>.progress</c> file is unreadable, the <c>.progress_backup</c>
	/// is tried.
	/// </para>
	/// <para>
	/// The interrupt file (<c>.interrupts</c>) holds the interrupt queue of the process operation. It is a copy of the BatchManager interrupt queue, written every time the queue changes;
	/// the BatchManager reads this file on restart.
	/// </para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form, SySal.DAQSystem.IProcessEventNotifier
	{
		internal class ThreadLogEntry : IComparer
		{
			public string Id;
			public string Function;
			#region IComparer Members

			public int Compare(object x, object y)
			{				
				return String.Compare(((ThreadLogEntry)x).Id, ((ThreadLogEntry)y).Id);
			}

			#endregion
		}

		internal static System.Collections.ArrayList ThreadLog = new ArrayList();

		internal static void ThreadLogStart(string function)
		{
			lock(ThreadLog)
			{                
                ThreadLogEntry thrl = new ThreadLogEntry();
                thrl.Id = System.Threading.Thread.CurrentThread.GetHashCode().ToString();
                thrl.Function = function;
                int insertpoint = ~ThreadLog.BinarySearch(thrl, thrl);
                if (insertpoint < 0)
                    ThreadLog.Insert(insertpoint, thrl);
			}
		}

		internal static void ThreadLogEnd()
		{
			lock(ThreadLog)
			{
				ThreadLogEntry thrl = new ThreadLogEntry();
				thrl.Id = System.Threading.Thread.CurrentThread.GetHashCode().ToString();
                int findpoint = ThreadLog.BinarySearch(thrl, thrl);
				if (findpoint >= 0)
                    ThreadLog.RemoveAt(findpoint);
			}
		}

		internal static void ThreadLogDump()
		{
			lock(ThreadLog)
			{
				System.IO.StreamWriter w = null;
				try
				{
					w = new System.IO.StreamWriter(ScratchDir + @"\operabatchmanager_win.threaddump.txt");
					foreach (ThreadLogEntry thrl in ThreadLog)
						w.WriteLine(thrl.Id + "\t" + thrl.Function);
					w.Flush();
				}
				catch(Exception) {}
				if (w != null) w.Close();
			}
		}

		object ReadOverride(string name, SySal.OperaDb.OperaDbConnection conn)
		{
			object o = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_MACHINEVARS WHERE ID_MACHINE = " + IdMachine + " AND NAME = '" + name + "'", conn, null).ExecuteScalar();
			if (o != null && o != System.DBNull.Value) return o;
			o = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = '" + name + "'", conn, null).ExecuteScalar();
			if (o == null) o = System.DBNull.Value;
			return o;
		}

		private System.Diagnostics.EventLog EventLog;
		private System.Windows.Forms.ListView ProcessList;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.ColumnHeader columnHeader4;
		private System.Windows.Forms.ColumnHeader columnHeader5;
		private System.Windows.Forms.ColumnHeader columnHeader6;
		private System.Windows.Forms.ColumnHeader columnHeader7;
		private System.Windows.Forms.ColumnHeader columnHeader8;
		private System.Windows.Forms.Button PauseButton;
		private System.Windows.Forms.Button ResumeButton;
		private System.Windows.Forms.Button AbortButton;
		private System.Windows.Forms.Button InterruptButton;
		private System.Windows.Forms.Button StartButton;
		private System.Windows.Forms.Timer MonitorTimer;
		private System.Windows.Forms.ColumnHeader columnHeader9;
		private System.Windows.Forms.ColumnHeader columnHeader10;
		private System.ComponentModel.IContainer components;

		private Size ProcessListDeflate;
		private System.Windows.Forms.ColumnHeader columnHeader11;
		private System.Windows.Forms.ColumnHeader columnHeader12;
		private System.Windows.Forms.Button ProgressButton;
		private System.Windows.Forms.TextBox ThreadsText;
        private System.Windows.Forms.Button ConfigReloadButton;
        internal Timer AutoStartTimer;
        private Button AutoStartButton;
		private Size ButtonLineMargin;

		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			ProcessListDeflate.Width = this.Width - ProcessList.Width;
			ProcessListDeflate.Height = this.Height - ProcessList.Height;
			ButtonLineMargin.Width = this.Width - AbortButton.Right; 
			ButtonLineMargin.Height = this.Height - AbortButton.Bottom;
			EventLog = new EventLog();
			OnStart();			
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if (components != null) 
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
            this.components = new System.ComponentModel.Container();
            this.ProcessList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader12 = new System.Windows.Forms.ColumnHeader();
            this.PauseButton = new System.Windows.Forms.Button();
            this.ResumeButton = new System.Windows.Forms.Button();
            this.AbortButton = new System.Windows.Forms.Button();
            this.InterruptButton = new System.Windows.Forms.Button();
            this.StartButton = new System.Windows.Forms.Button();
            this.MonitorTimer = new System.Windows.Forms.Timer(this.components);
            this.ProgressButton = new System.Windows.Forms.Button();
            this.ThreadsText = new System.Windows.Forms.TextBox();
            this.ConfigReloadButton = new System.Windows.Forms.Button();
            this.AutoStartTimer = new System.Windows.Forms.Timer(this.components);
            this.AutoStartButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // ProcessList
            // 
            this.ProcessList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader8,
            this.columnHeader9,
            this.columnHeader10,
            this.columnHeader5,
            this.columnHeader11,
            this.columnHeader6,
            this.columnHeader7,
            this.columnHeader12});
            this.ProcessList.FullRowSelect = true;
            this.ProcessList.GridLines = true;
            this.ProcessList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.ProcessList.HideSelection = false;
            this.ProcessList.Location = new System.Drawing.Point(8, 8);
            this.ProcessList.Name = "ProcessList";
            this.ProcessList.Size = new System.Drawing.Size(904, 176);
            this.ProcessList.TabIndex = 1;
            this.ProcessList.UseCompatibleStateImageBehavior = false;
            this.ProcessList.View = System.Windows.Forms.View.Details;
            this.ProcessList.DoubleClick += new System.EventHandler(this.OnProcessDoubleClick);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Operation Id";
            this.columnHeader1.Width = 100;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Type";
            this.columnHeader2.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Machine";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Description";
            this.columnHeader4.Width = 115;
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "Executable";
            this.columnHeader8.Width = 100;
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "Brick";
            this.columnHeader9.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader9.Width = 73;
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "Plate";
            this.columnHeader10.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader10.Width = 40;
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "Status";
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "Progress";
            this.columnHeader11.Width = 59;
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Start Time";
            this.columnHeader6.Width = 100;
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Finish Time";
            this.columnHeader7.Width = 91;
            // 
            // columnHeader12
            // 
            this.columnHeader12.Text = "Notes";
            this.columnHeader12.Width = 200;
            // 
            // PauseButton
            // 
            this.PauseButton.Location = new System.Drawing.Point(231, 192);
            this.PauseButton.Name = "PauseButton";
            this.PauseButton.Size = new System.Drawing.Size(64, 24);
            this.PauseButton.TabIndex = 2;
            this.PauseButton.Text = "Pause";
            this.PauseButton.Click += new System.EventHandler(this.PauseButton_Click);
            // 
            // ResumeButton
            // 
            this.ResumeButton.Location = new System.Drawing.Point(303, 192);
            this.ResumeButton.Name = "ResumeButton";
            this.ResumeButton.Size = new System.Drawing.Size(64, 24);
            this.ResumeButton.TabIndex = 3;
            this.ResumeButton.Text = "Resume";
            this.ResumeButton.Click += new System.EventHandler(this.ResumeButton_Click);
            // 
            // AbortButton
            // 
            this.AbortButton.Location = new System.Drawing.Point(848, 192);
            this.AbortButton.Name = "AbortButton";
            this.AbortButton.Size = new System.Drawing.Size(64, 24);
            this.AbortButton.TabIndex = 4;
            this.AbortButton.Text = "Abort";
            this.AbortButton.Click += new System.EventHandler(this.AbortButton_Click);
            // 
            // InterruptButton
            // 
            this.InterruptButton.Location = new System.Drawing.Point(375, 192);
            this.InterruptButton.Name = "InterruptButton";
            this.InterruptButton.Size = new System.Drawing.Size(64, 24);
            this.InterruptButton.TabIndex = 5;
            this.InterruptButton.Text = "Interrupt";
            this.InterruptButton.Click += new System.EventHandler(this.InterruptButton_Click);
            // 
            // StartButton
            // 
            this.StartButton.Location = new System.Drawing.Point(8, 192);
            this.StartButton.Name = "StartButton";
            this.StartButton.Size = new System.Drawing.Size(64, 24);
            this.StartButton.TabIndex = 7;
            this.StartButton.Text = "Start";
            this.StartButton.Click += new System.EventHandler(this.StartButton_Click);
            // 
            // MonitorTimer
            // 
            this.MonitorTimer.Interval = 500;
            this.MonitorTimer.Tick += new System.EventHandler(this.OnMonitorTimerTick);
            // 
            // ProgressButton
            // 
            this.ProgressButton.Location = new System.Drawing.Point(447, 192);
            this.ProgressButton.Name = "ProgressButton";
            this.ProgressButton.Size = new System.Drawing.Size(64, 24);
            this.ProgressButton.TabIndex = 8;
            this.ProgressButton.Text = "Progress";
            this.ProgressButton.Click += new System.EventHandler(this.ProgressButton_Click);
            // 
            // ThreadsText
            // 
            this.ThreadsText.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.ThreadsText.ForeColor = System.Drawing.Color.Navy;
            this.ThreadsText.Location = new System.Drawing.Point(710, 194);
            this.ThreadsText.Name = "ThreadsText";
            this.ThreadsText.ReadOnly = true;
            this.ThreadsText.Size = new System.Drawing.Size(96, 20);
            this.ThreadsText.TabIndex = 9;
            this.ThreadsText.Text = "Threads: 0";
            this.ThreadsText.DoubleClick += new System.EventHandler(this.OnThreadDblClick);
            // 
            // ConfigReloadButton
            // 
            this.ConfigReloadButton.Location = new System.Drawing.Point(519, 192);
            this.ConfigReloadButton.Name = "ConfigReloadButton";
            this.ConfigReloadButton.Size = new System.Drawing.Size(64, 24);
            this.ConfigReloadButton.TabIndex = 10;
            this.ConfigReloadButton.Text = "Reconfig";
            this.ConfigReloadButton.Click += new System.EventHandler(this.ConfigReloadButton_Click);
            // 
            // AutoStartTimer
            // 
            this.AutoStartTimer.Interval = 30000;
            this.AutoStartTimer.Tick += new System.EventHandler(this.OnAutoStartTick);
            // 
            // AutoStartButton
            // 
            this.AutoStartButton.Location = new System.Drawing.Point(608, 192);
            this.AutoStartButton.Name = "AutoStartButton";
            this.AutoStartButton.Size = new System.Drawing.Size(89, 24);
            this.AutoStartButton.TabIndex = 11;
            this.AutoStartButton.Text = "AutoStart ON";
            this.AutoStartButton.Click += new System.EventHandler(this.AutoStartButton_Click);
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(922, 224);
            this.Controls.Add(this.AutoStartButton);
            this.Controls.Add(this.ConfigReloadButton);
            this.Controls.Add(this.ThreadsText);
            this.Controls.Add(this.ProgressButton);
            this.Controls.Add(this.StartButton);
            this.Controls.Add(this.InterruptButton);
            this.Controls.Add(this.AbortButton);
            this.Controls.Add(this.ResumeButton);
            this.Controls.Add(this.PauseButton);
            this.Controls.Add(this.ProcessList);
            this.MinimumSize = new System.Drawing.Size(724, 152);
            this.Name = "MainForm";
            this.Text = "Opera Batch Manager - Windows Version";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Closing += new System.ComponentModel.CancelEventHandler(this.OnClosing);
            this.Resize += new System.EventHandler(this.OnResize);
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{			
			Application.Run(TheMainForm = new MainForm());
		}

		internal static MainForm TheMainForm;

        private delegate SySal.DAQSystem.Drivers.Status AsyncCall1(long procid, string token);
        private delegate void AsyncCall2(long procid);
        private delegate void AsyncCall3(SySal.DAQSystem.BatchManager.HostEnv h, string description, string machinename, string notes);

		/// <summary>
		/// Scratch directory.
		/// </summary>
		public static string ScratchDir;

		/// <summary>
		/// Raw data directory.
		/// </summary>
		public static string RawDataDir;

		/// <summary>
		/// Directory where task startup and progress files are put.
		/// </summary>
		public static string TaskDir;

		/// <summary>
		/// Directory where task startup and progress files are archived upon completion.
		/// </summary>
		public static string ArchivedTaskDir;

		/// <summary>
		/// Directory of driver executables.
		/// </summary>
		public static string DriverDir;

		/// <summary>
		/// Directory of executables for the DataProcessingServer.
		/// </summary>
		public static string ExeRepository;

		/// <summary>
		/// Time duration of each result in the result list.
		/// </summary>
		public static System.TimeSpan ResultLiveTime;

		/// <summary>
		/// If true, the Batch Manager impersonates the user that originally requests the batch when scheduling it on Data Processing Servers.
		/// </summary>
		public static bool ImpersonateBatchUser;

		/// <summary>
		/// Data processing server monitoring interval in seconds.
		/// </summary>
		public static int DataProcSrvMonitorInterval;

		/// <summary>
		/// Connection String for the DB.
		/// </summary>
		public static string DBServer;

		/// <summary>
		/// DB User that the OperaBatchServer shall impersonate.
		/// </summary>
		public static string DBUserName;

		/// <summary>
		/// DB Password to access the DB.
		/// </summary>
		public static string DBPassword;

		/// <summary>
		/// OPERA User that the OperaBatchServer shall impersonate.
		/// </summary>
		public static string OPERAUserName;

		/// <summary>
		/// OPERA Password to access the DB.
		/// </summary>
		public static string OPERAPassword;

		/// <summary>
		/// Site identifier read from the DB.
		/// </summary>
		public static long IdSite;

		/// <summary>
		/// Site name read from the DB.
		/// </summary>
		public static string SiteName;

		/// <summary>
		/// Machine identifier read from the DB.
		/// </summary>
		public static long IdMachine;

		/// <summary>
		/// Machine address that matches the DB registration entry.
		/// </summary>
		public static string MachineAddress;

		/// <summary>
		/// Machine name read from the DB.
		/// </summary>
		public static string MachineName;

		/// <summary>
		/// The Batch Manager instance.
		/// </summary>
		protected internal static SySal.DAQSystem.BatchManager BM = null;

		/// <summary>
		/// The Data Processing Server instance.
		/// </summary>
		protected internal static SySal.DAQSystem.IDataProcessingServer DPS = null;

        /// <summary>
        /// The Web Access provider.
        /// </summary>
        internal static SySal.Web.WebServer WA = null;

        /// <summary>
        /// The port to be used for Web access.
        /// </summary>
        internal int WebPort = 0;

        /// <summary>
        /// The file containing a queue of processes to be started.
        /// </summary>
        internal static string AutoStartFile;

        /// <summary>
        /// File hosting the list of tables to be queried for the monitoring page.
        /// </summary>
        internal static string MonitoringFile;

		/// <summary>
		/// Set things in motion so your service can do its work.
		/// </summary>
		protected void OnStart()
		{
			try
			{
				if (!EventLog.Exists("Opera")) EventLog.CreateEventSource("OperaBatchManager", "Opera");
				EventLog.Source = "OperaBatchManager";
				EventLog.Log = "Opera";

				System.Configuration.AppSettingsReader asr = new System.Configuration.AppSettingsReader();

                try
                {
                    DBServer = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBServer", typeof(string)));
                    DBUserName = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBUserName", typeof(string)));
                    DBPassword = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("DBPassword", typeof(string)));
                    OPERAUserName = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("OPERAUserName", typeof(string)));
                    OPERAPassword = SySal.OperaDb.OperaDbCredentials.Decode((string)asr.GetValue("OPERAPassword", typeof(string)));
                }
                catch (Exception x)
                {
                    throw new Exception("Encryption error in credentials.\r\nPlease fill in valid encrypted data (you can use OperaDbGUILogin, for instance), or run the service as the appropriate user.");
                }
                
                SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(DBServer, DBUserName, DBPassword);
				conn.Open();

				int i;
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
				SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_SITES.ID, TB_SITES.NAME, TB_MACHINES.ID, TB_MACHINES.NAME, TB_MACHINES.ADDRESS FROM TB_SITES INNER JOIN TB_MACHINES ON (TB_MACHINES.ID_SITE = TB_SITES.ID AND TB_MACHINES.ISBATCHSERVER = 1 AND (" + selstr + "))", conn, null);
				da.Fill(ds);
				if (ds.Tables[0].Rows.Count < 1) throw new Exception("Can't find myself in OperaDb registered machines. This service is made unavailable.");
				IdSite = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][0]);
				SiteName = ds.Tables[0].Rows[0][1].ToString();
				IdMachine = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[0][2]);
				MachineName = ds.Tables[0].Rows[0][3].ToString();
				MachineAddress = ds.Tables[0].Rows[0][4].ToString();

				object val;
				val = ReadOverride("BM_DataProcSrvMonitorInterval", conn); 
				if (val != System.DBNull.Value) DataProcSrvMonitorInterval = Convert.ToInt32(val.ToString());
				else DataProcSrvMonitorInterval = (int)asr.GetValue("BM_DataProcSrvMonitorInterval", typeof(int));

				val = ReadOverride("BM_ImpersonateBatchUser", conn); 
				if (val != System.DBNull.Value) ImpersonateBatchUser = Convert.ToBoolean(val.ToString());
				else ImpersonateBatchUser = (bool)asr.GetValue("BM_ImpersonateBatchUser", typeof(bool));

				val = ReadOverride("BM_ResultLiveSeconds", conn); 
				if (val != System.DBNull.Value) ResultLiveTime = System.TimeSpan.FromSeconds(Convert.ToInt32(val.ToString()));
				else ResultLiveTime = System.TimeSpan.FromSeconds((int)asr.GetValue("BM_ResultLiveSeconds", typeof(int)));		

				val = ReadOverride("ExeRepository", conn); 
				if (val != System.DBNull.Value) ExeRepository = val.ToString();
				else ExeRepository = (string)asr.GetValue("ExeRepository", typeof(string)); 
				if (ExeRepository.EndsWith("\\")) ExeRepository = ExeRepository.Remove(ExeRepository.Length - 1, 1);

				val = ReadOverride("ScratchDir", conn); 
				if (val != System.DBNull.Value) ScratchDir = val.ToString();
				else ScratchDir = (string)asr.GetValue("ScratchDir", typeof(string)); 
				if (ScratchDir.EndsWith("\\")) ScratchDir = ScratchDir.Remove(ScratchDir.Length - 1, 1);

				val = ReadOverride("RawDataDir", conn); 
				if (val != System.DBNull.Value) RawDataDir = val.ToString();
				else RawDataDir = (string)asr.GetValue("RawDataDir", typeof(string)); 
				if (RawDataDir.EndsWith("\\")) RawDataDir = RawDataDir.Remove(RawDataDir.Length - 1, 1);

				val = ReadOverride("TaskDir", conn); 
				if (val != System.DBNull.Value) TaskDir = val.ToString();
				else TaskDir = (string)asr.GetValue("TaskDir", typeof(string)); 
				if (TaskDir.EndsWith("\\")) TaskDir = TaskDir.Remove(TaskDir.Length - 1, 1);

				val = ReadOverride("ArchivedTaskDir", conn); 
				if (val != System.DBNull.Value) ArchivedTaskDir = val.ToString();
				else ArchivedTaskDir = (string)asr.GetValue("ArchivedTaskDir", typeof(string)); 
				if (ArchivedTaskDir.EndsWith("\\")) ArchivedTaskDir = ArchivedTaskDir.Remove(ArchivedTaskDir.Length - 1, 1);

				val = ReadOverride("DriverDir", conn); 
				if (val != System.DBNull.Value) DriverDir = val.ToString();
				else DriverDir = (string)asr.GetValue("DriverDir", typeof(string)); 
				if (DriverDir.EndsWith("\\")) DriverDir = DriverDir.Remove(DriverDir.Length - 1, 1);

                try
                {
                    val = ReadOverride("BM_WWWPort", conn);
                    if (val != System.DBNull.Value) WebPort = Convert.ToInt32(val.ToString());
                    else WebPort = (int)asr.GetValue("WWWPort", typeof(int));
                }
                catch (Exception)
                {
                    WebPort = 8080;
                }

                bool showexc = false;
                try
                {                    
                    val = ReadOverride("BM_WWWShowExceptions", conn);
                    if (val != System.DBNull.Value) showexc = Convert.ToBoolean(val.ToString());
                    else showexc = (bool)asr.GetValue("WWWShowExceptions", typeof(bool));
                }
                catch (Exception) { }

                WebAccess wa = null;
                if (WebPort > 0)
                    WA = new SySal.Web.WebServer(WebPort, wa = new WebAccess());
                if (wa != null) wa.SetShowExceptions(showexc);

                val = ReadOverride("BM_AutoStartFile", conn);
                if (val != System.DBNull.Value) AutoStartFile = val.ToString();
                else AutoStartFile = (string)asr.GetValue("AutoStartFile", typeof(string));

                val = ReadOverride("BM_MonitoringFile", conn);
                if (val != System.DBNull.Value) MonitoringFile = val.ToString();
                else
                    try
                    {
                        MonitoringFile = (string)asr.GetValue("MonitoringFile", typeof(string));
                    }
                    catch (Exception) { }

                ChannelServices.RegisterChannel(new TcpChannel((int)SySal.DAQSystem.OperaPort.BatchServer));
				BM = new SySal.DAQSystem.BatchManager(EventLog, this);
				RemotingServices.Marshal(BM, "BatchManager.rem");
                /*
				DPS = new SySal.DAQSystem.MyDataProcessingServer2(EventLog);
				RemotingServices.Marshal((MarshalByRefObject)DPS, "DataProcessingServer.rem");
                 */

                NExT.NExTConfiguration nxconfig = new SySal.NExT.NExTConfiguration();
                nxconfig.ConfigurationName = "DataProcessingServer";
                nxconfig.TCPIPPort = 0;
                nxconfig.TimeoutMS = 120000;
                nxconfig.ServiceEntries = new SySal.NExT.NExTConfiguration.ServiceEntry[] { new SySal.NExT.NExTConfiguration.ServiceEntry() };                
                nxconfig.ServiceEntries[0].Names = new string[] { "DataProcessingServer.rem" };
                nxconfig.ServiceEntries[0].Publish = true;
                nxconfig.ServiceEntries[0].CodeFile = "NExTDataProcessingServer.dll";
                nxconfig.ServiceEntries[0].TypeName = "SySal.NExT.DataProcessingServer";
                nxconfig.ServiceEntries[0].StaticParameters = new SySal.NExT.NExTConfiguration.ServerParameter[] 
                { 
                    new SySal.NExT.NExTConfiguration.ServerParameter(),
                    new SySal.NExT.NExTConfiguration.ServerParameter(),
                    new SySal.NExT.NExTConfiguration.ServerParameter(),
                    new SySal.NExT.NExTConfiguration.ServerParameter(),
                    new SySal.NExT.NExTConfiguration.ServerParameter(),
                    new SySal.NExT.NExTConfiguration.ServerParameter()
                };
                nxconfig.ServiceEntries[0].StaticParameters[0].Name = "DBSrv";
                nxconfig.ServiceEntries[0].StaticParameters[0].Value = MainForm.DBServer;
                nxconfig.ServiceEntries[0].StaticParameters[1].Name = "DBUsr";
                nxconfig.ServiceEntries[0].StaticParameters[1].Value = MainForm.DBUserName;
                nxconfig.ServiceEntries[0].StaticParameters[2].Name = "DBPwd";
                nxconfig.ServiceEntries[0].StaticParameters[2].Value = MainForm.DBPassword;
                nxconfig.ServiceEntries[0].StaticParameters[3].Name = "ResultTimeoutSeconds";
                nxconfig.ServiceEntries[0].StaticParameters[3].Value = MainForm.ResultLiveTime.TotalSeconds.ToString();
                nxconfig.ServiceEntries[0].StaticParameters[4].Name = "MaxQueueLength";
                nxconfig.ServiceEntries[0].StaticParameters[4].Value = "1000";
                nxconfig.ServiceEntries[0].StaticParameters[5].Name = "LogFile";
                nxconfig.ServiceEntries[0].StaticParameters[5].Value = ScratchDir + "\\dps.log";                

                System.Collections.ArrayList wrklist = new ArrayList();
                System.Data.DataSet w_ds = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter(
                    "select id, address, to_number(value) as workers from " +
                    "(select id, address from tb_machines where id_site = " +
                    "(select value from opera.lz_sitevars where name = 'ID_SITE') " +
                    " and isdataprocessingserver > 0 " +
                    ") inner join opera.lz_machinevars on (id = id_machine and name = 'DPS_Workers')",
                    conn).Fill(w_ds);
                foreach (System.Data.DataRow dwr in w_ds.Tables[0].Rows)
                {
                    for (i = 0; i < Convert.ToInt32(dwr[2]); i++)
                        wrklist.Add("tcp://" + dwr[1].ToString() + ":" + ((int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DPSWorker_" + i.ToString());
                }
                nxconfig.ServiceEntries[0].DataConsumerGroups = new SySal.NExT.NExTConfiguration.DataConsumerGroup[]
                {
                    new SySal.NExT.NExTConfiguration.DataConsumerGroup()
                };
                nxconfig.ServiceEntries[0].DataConsumerGroups[0] = new SySal.NExT.NExTConfiguration.DataConsumerGroup();
                nxconfig.ServiceEntries[0].DataConsumerGroups[0].Name = "Workers";
                nxconfig.ServiceEntries[0].DataConsumerGroups[0].RetryIntervalMS = 10000;
                nxconfig.ServiceEntries[0].DataConsumerGroups[0].URIs = (string [])wrklist.ToArray(typeof(string));
                NExT.NExTServer.SetupConfiguration(nxconfig, new SySal.NExT.NExTConfiguration.ServerParameter[0]);
                DPS = (SySal.DAQSystem.IDataProcessingServer)NExT.NExTServer.NExTServerFromURI("DataProcessingServer.rem");

				conn.Close();
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
			if (BM != null)
			{
				RemotingServices.Disconnect(BM);
				//((SySal.DAQSystem.MyDataProcessingServer2)DPS).AbortAllBatches();
				System.Diagnostics.Process.GetCurrentProcess().Kill();
			}
		}

		private void StopButton_Click(object sender, System.EventArgs e)
		{
			if (MessageBox.Show("Are you really sure you want to stop this BatchManager?\r\nAll processes will be paused.\r\nAll data processing results will be lost.\r\nAll scanning tasks will be lost.", "Warning", MessageBoxButtons.OKCancel, MessageBoxIcon.Asterisk) == DialogResult.OK)
				OnStop();
		}

		private void OnClosing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			if (MessageBox.Show("Are you really sure you want to stop this BatchManager?\r\nAll processes will be paused.\r\nAll data processing results will be lost.\r\nAll scanning tasks will be lost.", "Warning", MessageBoxButtons.OKCancel, MessageBoxIcon.Asterisk) == DialogResult.OK)
				OnStop();
			else e.Cancel = true;
		}

		private void OnMonitorTimerTick(object sender, System.EventArgs e)
		{
			SySal.DAQSystem.BatchManager.HostEnv [] tasks = BM.Tasks;
			lock(SySal.DAQSystem.BatchManager.TheInstance.DBConn)
			{
				foreach (SySal.DAQSystem.BatchManager.HostEnv h in tasks)								
					foreach (ListViewItem lvi in ProcessList.Items)
					{
						if ((long)lvi.Tag == h.m_StartupInfo.ProcessOperationId)
						{							
							lvi.SubItems[7].Text = (h.Domain == null) ? "Paused" : "Running";
							lvi.SubItems[8].Text = (h.m_ProgressInfo.Progress * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%";
							lvi.SubItems[9].Text = h.m_ProgressInfo.StartTime.ToString("HH:mm:ss dd/MM/yy");
							lvi.SubItems[10].Text = h.m_ProgressInfo.FinishTime.ToString("HH:mm:ss dd/MM/yy");
						}
					}
			}
			ThreadsText.Text = "Threads: " + System.Diagnostics.Process.GetCurrentProcess().Threads.Count;
		}

		private void PauseButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                long procid = 0;
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    if (ProcessList.SelectedItems.Count == 1)
                        procid = (long)(ProcessList.SelectedItems[0].Tag);
                if (procid > 0) new AsyncCall1(BM.Pause).BeginInvoke(procid, null, null, null);
            }
            catch (Exception x) 
            {
                MessageBox.Show(x.Message, "Internal error", MessageBoxButtons.OK);
            }
		}

		private void ResumeButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                long procid = 0;
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    if (ProcessList.SelectedItems.Count == 1)
                        procid = (long)(ProcessList.SelectedItems[0].Tag);
                if (procid > 0) BM.Resume(procid, null);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Internal error", MessageBoxButtons.OK);
            }
		}

		internal class ExecInterrupt
		{
			private long Id;
			private InterruptForm Ifd;
			private SySal.DAQSystem.BatchManager BM;
			internal ExecInterrupt(long id, InterruptForm ifd, SySal.DAQSystem.BatchManager bm)
			{
				Id = id;
				Ifd = ifd;
				BM = bm;
			}
			internal void Exec() 
			{
				ThreadLogStart("ExecInterrupt");
				if (Ifd.ShowDialog() == DialogResult.OK)
					BM.Interrupt(Id, Ifd.RTFIn.Text, null);
				ThreadLogEnd();
			}
		}

		private void InterruptButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                long procid = 0;
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    if (ProcessList.SelectedItems.Count == 1)
                        procid = (long)(ProcessList.SelectedItems[0].Tag);
                if (procid > 0) new System.Threading.Thread(new System.Threading.ThreadStart(new ExecInterrupt(procid, new InterruptForm(), BM).Exec)).Start();
            }
            catch (Exception x) 
            { 
                MessageBox.Show(x.Message, "Internal error", MessageBoxButtons.OK);
            }
		}

		private void ProgressButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    if (ProcessList.SelectedItems.Count == 1)
                    {
                        long id = (long)(ProcessList.SelectedItems[0].Tag);
                        SySal.DAQSystem.Drivers.HostEnv h = null;
                        foreach (SySal.DAQSystem.Drivers.HostEnv he in BM.Tasks)
                        {
                            if (he.StartupInfo.ProcessOperationId == id)
                            {
                                h = he;
                                break;
                            }
                        }
                        if (h != null)
                            new System.Threading.Thread(new System.Threading.ThreadStart(new ViewEditProgress(h).Exec)).Start();
                    }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Internal error", MessageBoxButtons.OK);
            }
		}

		static System.Xml.Serialization.XmlSerializer progxmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.TaskProgressInfo));

		internal class ViewEditProgress
		{
			private SySal.DAQSystem.Drivers.HostEnv H;
			internal ViewEditProgress(SySal.DAQSystem.Drivers.HostEnv h) { H = h; }
			internal void Exec()
			{
				ThreadLogStart("ViewEditProgress");
				SySal.DAQSystem.Drivers.TaskProgressInfo proginfo = H.ProgressInfo;
				System.IO.StringWriter wr = new System.IO.StringWriter();
				progxmls.Serialize(wr, proginfo);
				ProgressForm pf = new ProgressForm();
				pf.Text = "Progress Viewer/Editor for operation " + H.StartupInfo.ProcessOperationId;
				pf.RTFOut.Text = wr.ToString();
				if (pf.ShowDialog() == DialogResult.OK)
				{
					try
					{
						System.IO.StringReader rd = new System.IO.StringReader(pf.RTFOut.Text);
						H.ProgressInfo = (SySal.DAQSystem.Drivers.TaskProgressInfo)progxmls.Deserialize(rd);
					}
					catch (Exception x) 
					{
						MessageBox.Show(x.Message, "Error applying changes to progress file.", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
				}
				ThreadLogEnd();
			}
		}

		private void StartButton_Click(object sender, System.EventArgs e)
		{
			new System.Threading.Thread(new System.Threading.ThreadStart(ExecStart)).Start();
		}

		private void ExecStart()
		{
			ThreadLogStart("ExecStart");
            DialogResult dr = new DialogResult();
            StartForm sf = null;
            try
            {
    			sf = new StartForm();
                dr = sf.ShowDialog();
            }
            catch (Exception xc)
            {
                MessageBox.Show(xc.ToString(), "Invalid Input");
                ThreadLogEnd();
                return;
            }
			if (dr == DialogResult.OK)
			{
				SySal.DAQSystem.Drivers.TaskStartupInfo tsinfo = null;
				switch (sf.sel_DriverType)
				{
					case SySal.DAQSystem.Drivers.DriverType.Scanning:
					{
						SySal.DAQSystem.Drivers.ScanningStartupInfo xinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
						xinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
						xinfo.Plate.BrickId = sf.sel_BrickId;
						xinfo.Plate.PlateId = sf.sel_PlateId;
						tsinfo = xinfo;
						break;
					}

					case SySal.DAQSystem.Drivers.DriverType.Volume:
					{
						SySal.DAQSystem.Drivers.VolumeOperationInfo xinfo = new SySal.DAQSystem.Drivers.VolumeOperationInfo();
						xinfo.BrickId = sf.sel_BrickId;
						tsinfo = xinfo;
						break;
					}

					case SySal.DAQSystem.Drivers.DriverType.Brick:
					{
						SySal.DAQSystem.Drivers.BrickOperationInfo xinfo = new SySal.DAQSystem.Drivers.BrickOperationInfo();
						xinfo.BrickId = sf.sel_BrickId;
						tsinfo = xinfo;
						break;
					}
					
					case SySal.DAQSystem.Drivers.DriverType.System:
					{
						tsinfo = new SySal.DAQSystem.Drivers.TaskStartupInfo();
						break;
					}					
				}
				tsinfo.MachineId = sf.sel_MachineId;
				tsinfo.ProgramSettingsId = sf.sel_ProgramSettingsId;
				tsinfo.OPERAUsername = sf.sel_Username;
				tsinfo.OPERAPassword = sf.sel_Password;
				tsinfo.Notes = sf.NotesText.Text;
				try
				{
					BM.Start(0, tsinfo);
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "Error starting process operation", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
			}
			ThreadLogEnd();
		}        

		private void AbortButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                long procid = 0;
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    if (ProcessList.SelectedItems.Count == 1)
                        procid = (long)(ProcessList.SelectedItems[0].Tag);
                if (procid > 0) new AsyncCall1(BM.Abort).BeginInvoke(procid, null, null, null);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Internal error", MessageBoxButtons.OK);
            }
		}

		private void OnResize(object sender, System.EventArgs e)
		{			
			ProcessList.Width = this.Width - ProcessListDeflate.Width;
			ProcessList.Height = this.Height - ProcessListDeflate.Height;
			AbortButton.Location = new Point(this.Width - AbortButton.Width - ButtonLineMargin.Width, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			PauseButton.Location = new Point(PauseButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			ConfigReloadButton.Location = new Point(ConfigReloadButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			ResumeButton.Location = new Point(ResumeButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			StartButton.Location = new Point(StartButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			InterruptButton.Location = new Point(InterruptButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			ProgressButton.Location = new Point(ProgressButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
			ThreadsText.Location = new Point(ThreadsText.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
            AutoStartButton.Location = new Point(AutoStartButton.Left, this.Height - AbortButton.Height - ButtonLineMargin.Height);
		}
		#region IProcessEventNotifier Members



		/// <summary>
		/// Notifies the GUI interface that a new process operation starts.
		/// </summary>
		/// <param name="h">the HostEnv of the process operation.</param>
		/// <param name="description">the description of the process operation.</param>
		/// <param name="machinename">the name of the machine that runs the operation.</param>
		/// <param name="notes">notes about the process operation.</param>
		public void ProcessStart(SySal.DAQSystem.BatchManager.HostEnv h, string description, string machinename, string notes)
		{
            if (ProcessList.InvokeRequired) { ProcessList.Invoke(new AsyncCall3(ProcessStart), h, description, machinename, notes); }
            else
            {
                ListViewItem lvi = new ListViewItem(h.m_StartupInfo.ProcessOperationId.ToString());
                lvi.SubItems.Add("");
                Type t = h.m_StartupInfo.GetType();
                int brick = -1;
                int plate = -1;
                if (t == typeof(SySal.DAQSystem.Drivers.ScanningStartupInfo))
                {
                    lvi.SubItems[1].Text = "Scanning";
                    brick = (int)((SySal.DAQSystem.Drivers.ScanningStartupInfo)(h.m_StartupInfo)).Plate.BrickId;
                    plate = (int)((SySal.DAQSystem.Drivers.ScanningStartupInfo)(h.m_StartupInfo)).Plate.PlateId;
                }
                else if (t == typeof(SySal.DAQSystem.Drivers.VolumeOperationInfo))
                {
                    lvi.SubItems[1].Text = "Volume";
                    brick = (int)((SySal.DAQSystem.Drivers.VolumeOperationInfo)(h.m_StartupInfo)).BrickId;
                }
                else if (t == typeof(SySal.DAQSystem.Drivers.BrickOperationInfo))
                {
                    lvi.SubItems[1].Text = "Brick";
                    brick = (int)((SySal.DAQSystem.Drivers.BrickOperationInfo)(h.m_StartupInfo)).BrickId;
                }
                else
                {
                    lvi.SubItems[1].Text = "System";
                }
                lvi.SubItems.Add(machinename);
                lvi.SubItems.Add(description);
                string exe = (string)(h.m_Exe.Clone());
                lvi.SubItems.Add(exe.Remove(0, exe.LastIndexOf('\\') + 1));
                lvi.SubItems.Add((brick <= 0) ? "" : brick.ToString());
                lvi.SubItems.Add((plate <= 0) ? "" : plate.ToString());
                lvi.SubItems.Add("Starting/Paused");
                lvi.SubItems.Add("");
                lvi.SubItems.Add("");
                lvi.SubItems.Add("");
                lvi.SubItems.Add((notes == null) ? "" : notes);
                lvi.Tag = h.m_StartupInfo.ProcessOperationId;
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    ProcessList.Items.Add(lvi);
            }
		}        

		/// <summary>
		/// Notifies the GUI interface that a process operation has ended.
		/// </summary>
		/// <param name="id">the Id of the process operation that has ended.</param>
		public void ProcessEnd(long id)
		{
            if (ProcessList.InvokeRequired) ProcessList.Invoke(new AsyncCall2(ProcessEnd), id);
            else
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                {
                    int i;
                    for (i = 0; i < ProcessList.Items.Count; i++)
                        if ((long)ProcessList.Items[i].Tag == id)
                        {
                            ProcessList.Items.RemoveAt(i);
                            return;
                        }
                }
		}

		#endregion

		private void OnLoad(object sender, System.EventArgs e)
		{
			MonitorTimer.Start();
            AutoStartTimer.Start();
            AutoStartEnabled = false;            
            AutoStartButton.Text = AutoStartStartText;
		}

		private void OnThreadDblClick(object sender, System.EventArgs e)
		{
			ThreadLogDump();
		}

		private void ConfigReloadButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                long procid = 0;
                lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
                    if (ProcessList.SelectedItems.Count == 1)
                        procid = (long)(ProcessList.SelectedItems[0].Tag);
                if (procid > 0) new AsyncCall2(BM.Reconfig).BeginInvoke(procid, null, null);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Internal error", MessageBoxButtons.OK);
            }
        }

		private void OnProcessDoubleClick(object sender, System.EventArgs e)
		{
			long procid = 0;
            lock (SySal.DAQSystem.BatchManager.TheInstance.DBConn)
				if (ProcessList.SelectedItems.Count == 1)
					procid = (long)(ProcessList.SelectedItems[0].Tag);
			try
			{
				if (procid > 0) BM.Show(procid);							
			}
			catch (Exception) {}
		}        

        private void OnAutoStartTick(object sender, EventArgs e)
        {
            AutoStartButton.Text = AutoStartEnabled ? AutoStartStopText : AutoStartStartText;
            if (AutoStartEnabled == false) return;
            SySal.OperaDb.OperaDbConnection db = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);
            try
            {
                long[] busymachines = BM.BusyMachines;
                string[] startlist = System.IO.File.ReadAllText(AutoStartFile).Split('\r', '\n');
                int i;
                for (i = 0; i < startlist.Length; i++)
                {
                    string[] args = startlist[i].Split('$');
                    if (args.Length != 6) continue;
                    try
                    {
                        long machineid = 0;
                        try
                        {
                            machineid = Convert.ToInt64(args[1]);
                        }
                        catch (Exception)
                        {
                            try
                            {
                                db.Open();
                                machineid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_MACHINES WHERE NAME = '" + args[1].Trim() + "' AND ID_SITE = (SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE')", db).ExecuteScalar());
                            }
                            catch (Exception)
                            {
                                machineid = 0;
                            }
                            finally
                            {
                                db.Close();
                            }
                        }
                        foreach (long l in busymachines)
                            if (l == machineid)
                            {
                                machineid = 0;
                                break;
                            }
                        if (machineid <= 0) continue;
                        db.Open();
                        long progid = Convert.ToInt64(args[0]);   
                        long brickid = Convert.ToInt64(args[2]);
                        int plateid = Convert.ToInt32(args[3]);
                        string notes = args[4].Trim();
                        string interrupt = args[5].Trim();
                        int driverlevel = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT DRIVERLEVEL FROM TB_PROGRAMSETTINGS WHERE ID = " + progid, db).ExecuteScalar());
                        SySal.DAQSystem.Drivers.TaskStartupInfo tsinfo = null;
                        switch (driverlevel)
                        {
                            case (int)SySal.DAQSystem.Drivers.DriverType.System:
                                {                                    
                                    tsinfo = new SySal.DAQSystem.Drivers.TaskStartupInfo();                                    
                                }
                                break;

                            case (int)SySal.DAQSystem.Drivers.DriverType.Brick:
                                {
                                    SySal.DAQSystem.Drivers.BrickOperationInfo binfo = new SySal.DAQSystem.Drivers.BrickOperationInfo();
                                    binfo.BrickId = brickid;
                                    tsinfo = binfo;
                                }
                                break;

                            case (int)SySal.DAQSystem.Drivers.DriverType.Volume:
                                {
                                    SySal.DAQSystem.Drivers.VolumeOperationInfo vinfo = new SySal.DAQSystem.Drivers.VolumeOperationInfo();
                                    vinfo.BrickId = brickid;
                                    vinfo.Boxes = new SySal.DAQSystem.Drivers.BoxDesc[0];
                                    tsinfo = vinfo;
                                }
                                break;

                            case (int)SySal.DAQSystem.Drivers.DriverType.Scanning:
                                {
                                    SySal.DAQSystem.Drivers.ScanningStartupInfo sinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
                                    sinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
                                    sinfo.Plate.BrickId = brickid;
                                    sinfo.Plate.PlateId = plateid;
                                    tsinfo = sinfo;
                                }
                                break;

                            default: throw new Exception("Unsupported operation type.");
                        }
                        tsinfo.MachineId = machineid;
                        tsinfo.Notes = notes;
                        tsinfo.ProgramSettingsId = progid;
                        tsinfo.OPERAUsername = MainForm.OPERAUserName;
                        tsinfo.OPERAPassword = MainForm.OPERAPassword;
                        long procopid = BM.Start(0, tsinfo);
                        long[] nb = new long[busymachines.Length + 1];
                        int j;
                        for (j = 0; j < busymachines.Length; j++)
                            nb[j] = busymachines[j];
                        nb[j] = machineid;
                        busymachines = nb;
                        BM.Interrupt(procopid, MainForm.OPERAUserName, MainForm.OPERAPassword, args[5]);                        
                        startlist[i] = null;
                        db.Close();
                    }
                    catch (Exception) 
                    { 
                        db.Close();
                    }
                }
                string newtext = "";
                for (i = 0; i < startlist.Length; i++)
                    if (startlist[i] != null && startlist[i].Trim().Length > 0)
                    {
                        if (newtext.Length > 0) newtext += "\r\n";
                        newtext += startlist[i];
                    }
                System.IO.File.WriteAllText(AutoStartFile, newtext);
            }
            catch (Exception)
            {

            }
            finally
            {
                db.Close();
            }
        }

        const string AutoStartStopText = "AutoStart OFF";

        const string AutoStartStartText = "AutoStart ON";

        internal bool AutoStartEnabled = false;

        private void AutoStartButton_Click(object sender, EventArgs e)
        {
            AutoStartEnabled = !AutoStartEnabled;
            AutoStartButton.Text = AutoStartEnabled ? AutoStartStopText : AutoStartStartText;
        }
	}
}