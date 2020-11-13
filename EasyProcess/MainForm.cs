using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.EasyProcess
{
	/// <summary>
	/// EasyProcess - GUI application to manage processes running on a remote OperaBatchManager.
	/// </summary>
	/// <remarks>
	/// <para>EasyProcess runs with the credentials specified in the user record. The Computing Infrastructure credentials can be changed, but not the DB access ones. The DB access credentials can be recorded by using OperaDbGUILogin (See <see cref="SySal.Executables.OperaDbGUILogin.MainForm"/>) or OperaDbTextLogin (<see cref="SySal.Executables.OperaDbTextLogin.Exe"/>).</para>
	/// <para>
	/// The upper tree view shows hierarchies of process operations. It is initially empty, and is filled or updated each time one of the following buttons is pressed:
	/// <list type="table">
	/// <listheader><term>Button</term><description>Action</description></listheader>
	/// <item><term>Days Ago</term><description>Shows the root process operations started not earlier than the number of days specified in the adjacent box, with their descendants.</description></item>
	/// <item><term>In Progress</term><description>Shows the root process operations that are currently running, with their descendants (both running and completed).</description></item>
	/// <item><term>All</term><description>Shows all root process operations, with their descendants. Use of this button is discouraged, since it involves downloading the whole TB_PROC_OPERATIONS table, with additional joins, which may take a long time.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// A root process operation is a process operation with no parent operation.
	/// </para>
	/// <para>The Info button gets detailed information about the currently selected process operation, and opens a ProcOpInfo form (See <see cref="SySal.Executables.EasyProcess.ProcOpInfo"/>).</para>
	/// <para>In the lower window, running information of the BatchManager specified by its IP or DNS name are shown.</para>
	/// <para>
	/// Actions of the various buttons are explained as follows:	
	/// <list type="table">
	/// <listheader><term>Button</term><description>Action</description></listheader>
	/// <item><term>Start/Stop</term><description>Starts or stops polling the BatchManager status with the interval specified in seconds in the adjacent box.</description></item>
	/// <item><term>Pause</term><description>Pauses the currently selected process operation.</description></item>
	/// <item><term>Resume</term><description>Resumes the currently selected process operation.</description></item>
	/// <item><term>Abort</term><description>Aborts the currently selected process operation. <b>NOTICE: an aborted process operation can never be resumed and is lost forever.</b> Use Pause to suspend a process operation that you want to resume, even if you want to do so several days later.</description></item>
	/// </list>
	/// </para>
	/// <para>If the user double-clicks on one process operation in the BatchManager window (the lower window), EasyProcess attempts to open the HTML progress information page for it. It is expected to be in the directory specified as "Scratch Directory". If the directory is unspecified, EasyProcess attempts to download it from the DB settings for the current site, but this is not necessarily the right scratch directory for the BatchManager currently under use (in principle, every BatchManager can have its own scratch directory). The Scratch Directory can be set explicitly by the user, in which case the DB is not accessed.</para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TreeView ProcOpTree;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Button ProcOpRefreshButton;
		private System.Windows.Forms.ImageList ProcOpStatusImageList;
		private System.Windows.Forms.Button ProcOpRefreshDaysAgoButton;
		private System.Windows.Forms.TextBox DaysAgoText;
		private System.Windows.Forms.Button ProcOpRefreshInProgressButton;
		private System.Windows.Forms.ListView BatchManagerList;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.Button BMPauseButton;
		private System.Windows.Forms.Button BMResumeButton;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox OPERAUsernameText;
		private System.Windows.Forms.TextBox OPERAPasswordText;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox BatchManagerText;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button StartStopButton;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox RefreshSecondsText;
		private System.Windows.Forms.Timer BMRefreshTimer;
		private System.Windows.Forms.Button BMAbortButton;
		private System.Windows.Forms.Button ProcOpInfoButton;
		private System.Windows.Forms.TextBox ScratchDirText;
		private System.Windows.Forms.Label label4;
        private TextBox OnBrickText;
        private Button ProcOpRefreshOnBrick;
		private System.ComponentModel.IContainer components;

		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			DaysAgoText.Text = DaysAgo.ToString();
			int totalwidth = BatchManagerList.Width - 2 * SystemInformation.Border3DSize.Width;
			BatchManagerList.Columns.Add("Process Operation ID", (int)(totalwidth * 0.2), System.Windows.Forms.HorizontalAlignment.Left);
			BatchManagerList.Columns.Add("Executable", (int)(totalwidth * 0.2), System.Windows.Forms.HorizontalAlignment.Left);
			BatchManagerList.Columns.Add("Machine ID", (int)(totalwidth * 0.15), System.Windows.Forms.HorizontalAlignment.Left);
			BatchManagerList.Columns.Add("Brick", (int)(totalwidth * 0.075), System.Windows.Forms.HorizontalAlignment.Left);
			BatchManagerList.Columns.Add("Plate", (int)(totalwidth * 0.075), System.Windows.Forms.HorizontalAlignment.Left);
			BatchManagerList.Columns.Add("Level", (int)(totalwidth * 0.1), System.Windows.Forms.HorizontalAlignment.Left);
			BatchManagerList.Columns.Add("Status", (int)(totalwidth * 0.2), System.Windows.Forms.HorizontalAlignment.Left);

			StartStopButton.Text = "Start";
			BMRefreshTimer.Interval = 20 * 1000;
			BMRefreshTimer.Tick += new System.EventHandler(BMRefresh);
			System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new System.Runtime.Remoting.Channels.Tcp.TcpChannel());
			Credentials = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			OPERAUsernameText.Text = Credentials.OPERAUserName;
			OPERAPasswordText.Text = Credentials.OPERAPassword;
			BMPauseButton.Enabled = false;
			BMResumeButton.Enabled = false;
			BMAbortButton.Enabled = false;
		}

		SySal.OperaDb.OperaDbConnection Conn = null;

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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.ProcOpTree = new System.Windows.Forms.TreeView();
            this.ProcOpStatusImageList = new System.Windows.Forms.ImageList(this.components);
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.OnBrickText = new System.Windows.Forms.TextBox();
            this.ProcOpRefreshOnBrick = new System.Windows.Forms.Button();
            this.ProcOpInfoButton = new System.Windows.Forms.Button();
            this.DaysAgoText = new System.Windows.Forms.TextBox();
            this.ProcOpRefreshInProgressButton = new System.Windows.Forms.Button();
            this.ProcOpRefreshButton = new System.Windows.Forms.Button();
            this.ProcOpRefreshDaysAgoButton = new System.Windows.Forms.Button();
            this.BatchManagerList = new System.Windows.Forms.ListView();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.ScratchDirText = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.RefreshSecondsText = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.StartStopButton = new System.Windows.Forms.Button();
            this.BatchManagerText = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.OPERAPasswordText = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.OPERAUsernameText = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.BMAbortButton = new System.Windows.Forms.Button();
            this.BMResumeButton = new System.Windows.Forms.Button();
            this.BMPauseButton = new System.Windows.Forms.Button();
            this.BMRefreshTimer = new System.Windows.Forms.Timer(this.components);
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // ProcOpTree
            // 
            this.ProcOpTree.ForeColor = System.Drawing.SystemColors.Highlight;
            this.ProcOpTree.FullRowSelect = true;
            this.ProcOpTree.HideSelection = false;
            this.ProcOpTree.ImageIndex = 0;
            this.ProcOpTree.ImageList = this.ProcOpStatusImageList;
            this.ProcOpTree.Location = new System.Drawing.Point(16, 24);
            this.ProcOpTree.Name = "ProcOpTree";
            this.ProcOpTree.SelectedImageIndex = 0;
            this.ProcOpTree.Size = new System.Drawing.Size(648, 256);
            this.ProcOpTree.TabIndex = 1;
            this.ProcOpTree.BeforeExpand += new System.Windows.Forms.TreeViewCancelEventHandler(this.OnProcOpBeforeExpand);
            // 
            // ProcOpStatusImageList
            // 
            this.ProcOpStatusImageList.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("ProcOpStatusImageList.ImageStream")));
            this.ProcOpStatusImageList.TransparentColor = System.Drawing.Color.Transparent;
            this.ProcOpStatusImageList.Images.SetKeyName(0, "");
            this.ProcOpStatusImageList.Images.SetKeyName(1, "");
            this.ProcOpStatusImageList.Images.SetKeyName(2, "");
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.OnBrickText);
            this.groupBox1.Controls.Add(this.ProcOpRefreshOnBrick);
            this.groupBox1.Controls.Add(this.ProcOpInfoButton);
            this.groupBox1.Controls.Add(this.DaysAgoText);
            this.groupBox1.Controls.Add(this.ProcOpRefreshInProgressButton);
            this.groupBox1.Controls.Add(this.ProcOpRefreshButton);
            this.groupBox1.Location = new System.Drawing.Point(8, 8);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(664, 312);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "All process operations";
            // 
            // OnBrickText
            // 
            this.OnBrickText.Location = new System.Drawing.Point(455, 283);
            this.OnBrickText.Name = "OnBrickText";
            this.OnBrickText.Size = new System.Drawing.Size(58, 20);
            this.OnBrickText.TabIndex = 8;
            this.OnBrickText.Text = "1000000";
            this.OnBrickText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.OnBrickText.Leave += new System.EventHandler(this.OnBrickLeave);
            // 
            // ProcOpRefreshOnBrick
            // 
            this.ProcOpRefreshOnBrick.Location = new System.Drawing.Point(369, 280);
            this.ProcOpRefreshOnBrick.Name = "ProcOpRefreshOnBrick";
            this.ProcOpRefreshOnBrick.Size = new System.Drawing.Size(80, 24);
            this.ProcOpRefreshOnBrick.TabIndex = 7;
            this.ProcOpRefreshOnBrick.Text = "On &Brick";
            this.ProcOpRefreshOnBrick.Click += new System.EventHandler(this.ProcOpRefreshOnBrick_Click);
            // 
            // ProcOpInfoButton
            // 
            this.ProcOpInfoButton.Location = new System.Drawing.Point(8, 280);
            this.ProcOpInfoButton.Name = "ProcOpInfoButton";
            this.ProcOpInfoButton.Size = new System.Drawing.Size(64, 24);
            this.ProcOpInfoButton.TabIndex = 2;
            this.ProcOpInfoButton.Text = "I&nfo";
            this.ProcOpInfoButton.Click += new System.EventHandler(this.ProcOpInfoButton_Click);
            // 
            // DaysAgoText
            // 
            this.DaysAgoText.Location = new System.Drawing.Point(208, 280);
            this.DaysAgoText.Name = "DaysAgoText";
            this.DaysAgoText.Size = new System.Drawing.Size(40, 20);
            this.DaysAgoText.TabIndex = 4;
            this.DaysAgoText.Text = "3";
            this.DaysAgoText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.DaysAgoText.Leave += new System.EventHandler(this.OnDaysAgoTextLeave);
            // 
            // ProcOpRefreshInProgressButton
            // 
            this.ProcOpRefreshInProgressButton.Location = new System.Drawing.Point(272, 280);
            this.ProcOpRefreshInProgressButton.Name = "ProcOpRefreshInProgressButton";
            this.ProcOpRefreshInProgressButton.Size = new System.Drawing.Size(80, 24);
            this.ProcOpRefreshInProgressButton.TabIndex = 5;
            this.ProcOpRefreshInProgressButton.Text = "&In progress";
            this.ProcOpRefreshInProgressButton.Click += new System.EventHandler(this.ProcOpRefreshInProgressButton_Click);
            // 
            // ProcOpRefreshButton
            // 
            this.ProcOpRefreshButton.Location = new System.Drawing.Point(608, 280);
            this.ProcOpRefreshButton.Name = "ProcOpRefreshButton";
            this.ProcOpRefreshButton.Size = new System.Drawing.Size(48, 24);
            this.ProcOpRefreshButton.TabIndex = 6;
            this.ProcOpRefreshButton.Text = "All";
            this.ProcOpRefreshButton.Click += new System.EventHandler(this.OnProcOpRefreshButton_Click);
            // 
            // ProcOpRefreshDaysAgoButton
            // 
            this.ProcOpRefreshDaysAgoButton.Location = new System.Drawing.Point(128, 288);
            this.ProcOpRefreshDaysAgoButton.Name = "ProcOpRefreshDaysAgoButton";
            this.ProcOpRefreshDaysAgoButton.Size = new System.Drawing.Size(80, 24);
            this.ProcOpRefreshDaysAgoButton.TabIndex = 3;
            this.ProcOpRefreshDaysAgoButton.Text = "Days &ago";
            this.ProcOpRefreshDaysAgoButton.Click += new System.EventHandler(this.ProcOpRefreshDaysAgoButton_Click);
            // 
            // BatchManagerList
            // 
            this.BatchManagerList.ForeColor = System.Drawing.SystemColors.Highlight;
            this.BatchManagerList.FullRowSelect = true;
            this.BatchManagerList.GridLines = true;
            this.BatchManagerList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.BatchManagerList.HideSelection = false;
            this.BatchManagerList.Location = new System.Drawing.Point(16, 376);
            this.BatchManagerList.MultiSelect = false;
            this.BatchManagerList.Name = "BatchManagerList";
            this.BatchManagerList.Size = new System.Drawing.Size(648, 184);
            this.BatchManagerList.TabIndex = 13;
            this.BatchManagerList.UseCompatibleStateImageBehavior = false;
            this.BatchManagerList.View = System.Windows.Forms.View.Details;
            this.BatchManagerList.DoubleClick += new System.EventHandler(this.OnBatchList_DblClick);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.ScratchDirText);
            this.groupBox2.Controls.Add(this.label4);
            this.groupBox2.Controls.Add(this.RefreshSecondsText);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Controls.Add(this.StartStopButton);
            this.groupBox2.Controls.Add(this.BatchManagerText);
            this.groupBox2.Controls.Add(this.label3);
            this.groupBox2.Controls.Add(this.OPERAPasswordText);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.OPERAUsernameText);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.BMAbortButton);
            this.groupBox2.Controls.Add(this.BMResumeButton);
            this.groupBox2.Controls.Add(this.BMPauseButton);
            this.groupBox2.Location = new System.Drawing.Point(8, 328);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(664, 304);
            this.groupBox2.TabIndex = 7;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Batch Manager view";
            // 
            // ScratchDirText
            // 
            this.ScratchDirText.Location = new System.Drawing.Point(280, 240);
            this.ScratchDirText.Name = "ScratchDirText";
            this.ScratchDirText.Size = new System.Drawing.Size(312, 20);
            this.ScratchDirText.TabIndex = 22;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(136, 240);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(136, 24);
            this.label4.TabIndex = 21;
            this.label4.Text = "Scratch directory";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // RefreshSecondsText
            // 
            this.RefreshSecondsText.Location = new System.Drawing.Point(624, 16);
            this.RefreshSecondsText.Name = "RefreshSecondsText";
            this.RefreshSecondsText.Size = new System.Drawing.Size(32, 20);
            this.RefreshSecondsText.TabIndex = 12;
            this.RefreshSecondsText.Text = "20";
            this.RefreshSecondsText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.RefreshSecondsText.Leave += new System.EventHandler(this.OnLeaveRefreshSeconds);
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(544, 16);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(72, 24);
            this.label5.TabIndex = 11;
            this.label5.Text = "Refresh (s)";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // StartStopButton
            // 
            this.StartStopButton.Location = new System.Drawing.Point(472, 16);
            this.StartStopButton.Name = "StartStopButton";
            this.StartStopButton.Size = new System.Drawing.Size(64, 24);
            this.StartStopButton.TabIndex = 10;
            this.StartStopButton.Text = "&Start/Stop";
            this.StartStopButton.Click += new System.EventHandler(this.StartStopButton_Click);
            // 
            // BatchManagerText
            // 
            this.BatchManagerText.Location = new System.Drawing.Point(152, 16);
            this.BatchManagerText.Name = "BatchManagerText";
            this.BatchManagerText.Size = new System.Drawing.Size(312, 20);
            this.BatchManagerText.TabIndex = 9;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(8, 16);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(136, 24);
            this.label3.TabIndex = 8;
            this.label3.Text = "Batch Manager address";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // OPERAPasswordText
            // 
            this.OPERAPasswordText.Location = new System.Drawing.Point(424, 272);
            this.OPERAPasswordText.Name = "OPERAPasswordText";
            this.OPERAPasswordText.PasswordChar = '*';
            this.OPERAPasswordText.Size = new System.Drawing.Size(136, 20);
            this.OPERAPasswordText.TabIndex = 20;
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(296, 272);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 24);
            this.label2.TabIndex = 19;
            this.label2.Text = "OPERA &Password";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // OPERAUsernameText
            // 
            this.OPERAUsernameText.Location = new System.Drawing.Point(136, 272);
            this.OPERAUsernameText.Name = "OPERAUsernameText";
            this.OPERAUsernameText.Size = new System.Drawing.Size(136, 20);
            this.OPERAUsernameText.TabIndex = 18;
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(8, 272);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(120, 24);
            this.label1.TabIndex = 17;
            this.label1.Text = "OPERA &Username";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // BMAbortButton
            // 
            this.BMAbortButton.Location = new System.Drawing.Point(600, 240);
            this.BMAbortButton.Name = "BMAbortButton";
            this.BMAbortButton.Size = new System.Drawing.Size(56, 24);
            this.BMAbortButton.TabIndex = 16;
            this.BMAbortButton.Text = "Abort";
            this.BMAbortButton.Click += new System.EventHandler(this.BMAbortButton_Click);
            // 
            // BMResumeButton
            // 
            this.BMResumeButton.Location = new System.Drawing.Point(72, 240);
            this.BMResumeButton.Name = "BMResumeButton";
            this.BMResumeButton.Size = new System.Drawing.Size(56, 24);
            this.BMResumeButton.TabIndex = 15;
            this.BMResumeButton.Text = "Resume";
            this.BMResumeButton.Click += new System.EventHandler(this.BMResumeButton_Click);
            // 
            // BMPauseButton
            // 
            this.BMPauseButton.Location = new System.Drawing.Point(8, 240);
            this.BMPauseButton.Name = "BMPauseButton";
            this.BMPauseButton.Size = new System.Drawing.Size(56, 24);
            this.BMPauseButton.TabIndex = 14;
            this.BMPauseButton.Text = "Pause";
            this.BMPauseButton.Click += new System.EventHandler(this.BMPauseButton_Click);
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(680, 638);
            this.Controls.Add(this.BatchManagerList);
            this.Controls.Add(this.ProcOpRefreshDaysAgoButton);
            this.Controls.Add(this.ProcOpTree);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.groupBox2);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.Text = "Easy Process";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new MainForm());
		}

		uint DaysAgo = 1;

		private void OnProcOpRefreshButton_Click(object sender, System.EventArgs e)
		{
			ProcOpRefresh("");
		}

		SySal.OperaDb.OperaDbCredentials Credentials = null;

		void ProcOpRefresh(string selection)
		{
			//SySal.OperaDb.OperaDbConnection Conn = null;
			System.Windows.Forms.Cursor oldcursor = Cursor;
			Cursor = Cursors.WaitCursor;
			try
			{
				if (Conn == null)
				{
					Conn = new SySal.OperaDb.OperaDbConnection(Credentials.DBServer, Credentials.DBUserName, Credentials.DBPassword);
					Conn.Open();
				}
				ProcOpTree.BeginUpdate();
				ProcOpTree.Nodes.Clear();
				System.Data.DataSet ds = new System.Data.DataSet();
				SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_RANGE (TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT) */ TB_PROC_OPERATIONS.ID, TB_PROGRAMSETTINGS.DESCRIPTION, TB_PROGRAMSETTINGS.EXECUTABLE, TB_PROC_OPERATIONS.STARTTIME, TB_PROC_OPERATIONS.SUCCESS, TB_PROC_OPERATIONS.ID_EVENTBRICK, TB_PROC_OPERATIONS.ID_PLATE FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID) WHERE (TB_PROC_OPERATIONS.ID_PARENT_OPERATION IS NULL " + selection + ") ORDER BY STARTTIME DESC", Conn, null);
				da.Fill(ds);				
				System.Collections.Stack nodestack = new System.Collections.Stack();
				foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				{
					int image = 2;
					if (dr[4].ToString() == "N") image = 1;
					else if (dr[4].ToString() == "Y") image = 0;
					string addinfo = " ";
					if (dr[5] != System.DBNull.Value) addinfo += "B#" + dr[5].ToString() + " ";
					if (dr[6] != System.DBNull.Value) addinfo += "P#" + dr[6].ToString() + " ";
					TreeNode tn = new TreeNode("#" + dr[0].ToString() + ": " + dr[1].ToString() + addinfo + "(" + dr[2].ToString() + ") - (" + Convert.ToDateTime(dr[3]).ToString("dd/MM/yyyy HH:mm:ss") + ")", image, image);
					tn.Tag = Convert.ToInt64(dr[0]);
					ProcOpTree.Nodes.Add(tn);
					nodestack.Push(tn);
				}
				TreeNode nextnode = null;
				while (nodestack.Count > 0)
				{
					nextnode = (TreeNode)nodestack.Pop();
					ds = new System.Data.DataSet();
					da = new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT) */ TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS WHERE (TB_PROC_OPERATIONS.ID_PARENT_OPERATION = " + nextnode.Tag.ToString() + ") ORDER BY STARTTIME DESC", Conn, null);
					da.Fill(ds);
					foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
					{						
						TreeNode tn = new TreeNode("");
						tn.Tag = Convert.ToInt64(dr[0]);
						nextnode.Nodes.Add(tn);
						nodestack.Push(tn);
					}
				}
				ProcOpTree.EndUpdate();
				//Conn.Close();
			}
			catch (Exception x)
			{
				if (Conn != null) 
				{
					Conn.Close();
					Conn = null;
				}
				ProcOpTree.EndUpdate();
				MessageBox.Show(x.ToString(), "Error refreshing process operation information");
			}
			Cursor = oldcursor;
		}

		private void OnProcOpBeforeExpand(object sender, System.Windows.Forms.TreeViewCancelEventArgs e)
		{
			ProcOpTree.BeginUpdate();
			//SySal.OperaDb.OperaDbConnection Conn = null;
			try
			{
				foreach (TreeNode tn in e.Node.Nodes)
				{
					if (tn.Text.Length == 0)
					{
						if (Conn == null)
						{						
							Conn = new SySal.OperaDb.OperaDbConnection(Credentials.DBServer, Credentials.DBUserName, Credentials.DBPassword);
							Conn.Open();
						}
						System.Data.DataSet ds = new System.Data.DataSet();
						new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX(TB_PROC_OPERATIONS PK_PROC_OPERATIONS) */ DESCRIPTION, EXECUTABLE, STARTTIME, SUCCESS, ID_EVENTBRICK, ID_PLATE FROM TB_PROGRAMSETTINGS INNER JOIN TB_PROC_OPERATIONS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID) WHERE (TB_PROC_OPERATIONS.ID = " + tn.Tag.ToString() + ")", Conn, null).Fill(ds);
						int image = 2;
						System.Data.DataRow dr = ds.Tables[0].Rows[0];
						if (dr[3].ToString() == "N") image = 1;
						else if (dr[3].ToString() == "Y") image = 0;
						string addinfo = " ";
						if (dr[4] != System.DBNull.Value) addinfo += "B#" + dr[4].ToString() + " ";
						if (dr[5] != System.DBNull.Value) addinfo += "P#" + dr[5].ToString() + " ";
						tn.Text = "#" + tn.Tag.ToString() + ": " + dr[0].ToString() + addinfo + "(" + dr[1].ToString() + ") - (" + Convert.ToDateTime(dr[2]).ToString("dd/MM/yyyy HH:mm:ss") + ")";
						tn.ImageIndex = tn.SelectedImageIndex = image;
					}
				}
				//if (Conn != null) Conn.Close();
				ProcOpTree.EndUpdate();
			}
			catch (Exception x)
			{
				if (Conn != null) 
				{
					Conn.Close();
					Conn = null;
				}
				ProcOpTree.EndUpdate();
				MessageBox.Show(x.ToString(), "Error refreshing process operation information");
			}
		}

		private void OnDaysAgoTextLeave(object sender, System.EventArgs e)
		{
			try
			{
				DaysAgo = Convert.ToUInt32(DaysAgoText.Text);
			}
			catch (Exception)
			{
				DaysAgoText.Text = (DaysAgo = 1).ToString();
			}
		}

		private void ProcOpRefreshDaysAgoButton_Click(object sender, System.EventArgs e)
		{
			System.DateTime start = System.DateTime.Now.AddDays(-DaysAgo);
			ProcOpRefresh("AND STARTTIME >= TO_TIMESTAMP('" + SySal.OperaDb.OperaDbConnection.ToTimeFormat(start) + "', " + SySal.OperaDb.OperaDbConnection.TimeFormat + ")");
		}

		private void ProcOpRefreshInProgressButton_Click(object sender, System.EventArgs e)
		{
			ProcOpRefresh("AND SUCCESS = 'R'");
		}

		private void ProcOpMenuInfo_Click(object sender, System.EventArgs e)
		{
		}

		private void ProcOpInfoButton_Click(object sender, System.EventArgs e)
		{
			if (ProcOpTree.SelectedNode != null)
			{
				ProcOpInfo showinfo = new ProcOpInfo(Credentials, Convert.ToInt64(ProcOpTree.SelectedNode.Tag), ref Conn);
				showinfo.ShowDialog();
				showinfo.Dispose();
			}				
		}

		SySal.DAQSystem.BatchManager BMSrv = null;

		private void StartStopButton_Click(object sender, System.EventArgs e)
		{
			if (StartStopButton.Text == "Start")
			{
				try
				{					
					BMSrv = (SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + BatchManagerText.Text + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer) + "/BatchManager.rem");
				}
				catch (Exception x)
				{
					BMSrv = null;
					MessageBox.Show(x.ToString(), "Can't connect to the specified Batch Manager!", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}
				BatchManagerText.Enabled = false;
				RefreshSecondsText.Enabled = false;
				BMPauseButton.Enabled = true;
				BMResumeButton.Enabled = true;
				BMAbortButton.Enabled = true;
				StartStopButton.Text = "Stop";
				BMRefreshTimer.Start();
			}
			else
			{
				BMRefreshTimer.Stop();
				BatchManagerText.Enabled = true;
				RefreshSecondsText.Enabled = true;
				BMPauseButton.Enabled = false;
				BMResumeButton.Enabled = false;
				BMAbortButton.Enabled = false;
				StartStopButton.Text = "Start";
			}
		}

		private void BMRefresh(object o, EventArgs e)
		{
			lock(BatchManagerList)
			{
				BatchManagerList.BeginUpdate();
				long selectedid = 0;
				if (BatchManagerList.SelectedItems.Count == 1)
					selectedid = Convert.ToInt64(BatchManagerList.SelectedItems[0].Tag);
				try
				{
					BatchManagerList.Items.Clear();
					long [] ids = BMSrv.Operations;
					foreach (long id in ids)
					{
						try
						{
							ListViewItem li = BatchManagerList.Items.Add(id.ToString());
							li.Tag = id;
							SySal.DAQSystem.Drivers.BatchSummary binfo = BMSrv.GetSummary(id);
							if (binfo == null) continue;
							li.SubItems.Add(binfo.Executable);
							li.SubItems.Add(binfo.MachineId.ToString());
							li.SubItems.Add(binfo.BrickId == 0 ? "" : binfo.BrickId.ToString());
							li.SubItems.Add(binfo.PlateId == 0 ? "" : binfo.PlateId.ToString());
							li.SubItems.Add(binfo.DriverLevel.ToString());
							if (binfo.OpStatus == SySal.DAQSystem.Drivers.Status.Running)
								li.SubItems.Add((binfo.Progress * 100.0).ToString("F1") + "% ETR: " + binfo.ExpectedFinishTime.ToString("HH:mm:ss"));
							else 
								li.SubItems.Add(binfo.OpStatus.ToString());
							if (id == selectedid) li.Selected = true;
						}
						catch (Exception x) 
						{
							MessageBox.Show(x.ToString(), x.Message);
						}
					}
				}
				catch(Exception)
				{
				
				}
				BatchManagerList.EndUpdate();
			}
		}

		private void OnLeaveRefreshSeconds(object sender, System.EventArgs e)
		{
			try
			{
				BMRefreshTimer.Interval = 1000 * (int)Convert.ToUInt32(RefreshSecondsText.Text);
			}
			catch (Exception)
			{
				RefreshSecondsText.Text = ((BMRefreshTimer.Interval = 20 * 1000) / 1000).ToString();
			}		
		}

		private void BMPauseButton_Click(object sender, System.EventArgs e)
		{
			lock(BatchManagerList)
			{
				long selectedid = 0;
				if (BatchManagerList.SelectedItems.Count == 1)
				{
					selectedid = Convert.ToInt64(BatchManagerList.SelectedItems[0].Tag);
					if (MessageBox.Show("Are you sure you want to pause process operation #" + selectedid + "?", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
					{
						try
						{
							BMSrv.Pause(selectedid, OPERAUsernameText.Text, OPERAPasswordText.Text);
						}
						catch (Exception x)
						{
							MessageBox.Show(x.ToString(), x.Message, MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
						}
					}
				}
			}
		}

		private void BMResumeButton_Click(object sender, System.EventArgs e)
		{
			lock(BatchManagerList)
			{
				long selectedid = 0;
				if (BatchManagerList.SelectedItems.Count == 1)
				{
					selectedid = Convert.ToInt64(BatchManagerList.SelectedItems[0].Tag);
					try
					{
						BMSrv.Resume(selectedid, OPERAUsernameText.Text, OPERAPasswordText.Text);
					}
					catch (Exception x)
					{
						MessageBox.Show(x.ToString(), x.Message, MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
					}
				}
			}		
		}

		private void BMAbortButton_Click(object sender, System.EventArgs e)
		{
			lock(BatchManagerList)
			{
				long selectedid = 0;
				if (BatchManagerList.SelectedItems.Count == 1)
				{
					selectedid = Convert.ToInt64(BatchManagerList.SelectedItems[0].Tag);
					if (MessageBox.Show("Are you sure you want to abort process operation #" + selectedid + "?", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
					{
						try
						{
							BMSrv.Abort(selectedid, OPERAUsernameText.Text, OPERAPasswordText.Text);
						}
						catch (Exception x)
						{
							MessageBox.Show(x.ToString(), x.Message, MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
						}
					}
				}
			}		
		}

		private void OnBatchList_DblClick(object sender, System.EventArgs e)
		{
			try
			{
				long selectedid = 0;
				lock(BatchManagerList)
				{
					if (BatchManagerList.SelectedItems.Count != 1) return;
					selectedid = Convert.ToInt64(BatchManagerList.SelectedItems[0].Tag);
				}
				if (ScratchDirText.Text.Length == 0)
				{
					if (Conn == null)
					{
						Conn = new SySal.OperaDb.OperaDbConnection(Credentials.DBServer, Credentials.DBUserName, Credentials.DBPassword);
						Conn.Open();
					}
					ScratchDirText.Text = new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = 'ScratchDir'", Conn, null).ExecuteScalar().ToString();					
				}
				if (!ScratchDirText.Text.EndsWith(@"\")) ScratchDirText.Text += @"\";
				System.Diagnostics.Process.Start(ScratchDirText.Text + selectedid + "_progress.htm");
			}
			catch (Exception x)
			{
				MessageBox.Show("Can't retrieve progress page path because of the following error:\r\n" + x.ToString(), "DB connection error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
			}
			return;
#if false
			// TODO: Find a way to deserialize CustomInfo XML elements...
			SySal.DAQSystem.Drivers.TaskProgressInfo progrinfo = null;
			long selectedid = 0;
			lock(BatchManagerList)
			{
				if (BatchManagerList.SelectedItems.Count != 1) return;
				selectedid = Convert.ToInt64(BatchManagerList.SelectedItems[0].Tag);
				try
				{
					progrinfo = BMSrv.GetProgressInfo(selectedid);
				}
				catch (Exception x)
				{
					MessageBox.Show(x.ToString(), x.Message, MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
				}				
			}
			if (progrinfo != null)
			{
				ProgressInfo progrdlg = new ProgressInfo(selectedid, progrinfo);
				progrdlg.ShowDialog();
				progrdlg.Dispose();
			}
#endif
		}

        private void ProcOpRefreshOnBrick_Click(object sender, EventArgs e)
        {
            System.DateTime start = System.DateTime.Now.AddDays(-DaysAgo);
            ProcOpRefresh("AND ID_EVENTBRICK = " + OnBrickText.Text);
        }

        private void OnBrickLeave(object sender, EventArgs e)
        {
            try
            {
                System.Convert.ToInt32(OnBrickText.Text);
            }
            catch (Exception)
            {
                OnBrickText.Text = "1000000";
            }
        }
	}
}
