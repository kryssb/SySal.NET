using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// The MainForm of the OperaPublicationManager is where all actions start.
	/// </summary>
	/// <remarks>The form is divided in five main sections:
	/// <para><list type="table">
	/// <item><term>Login area</term><description>Here the DB and the login credentials are selected. The DB connection is established after the <i>Connect</i> button is pressed.</description></item>
	/// <item><term>General area</term><description>This area is circled by the <i>General</i> group box. DB links and queues are set up here.
	/// <para>A <i><b>DB Link</b></i> is the fundamental pillar of Oracle DB-based publication. The local DB communicates with other DBs identified as DB Links. The local DB logs onto the remote DB identifying itself as a specific user on the remote DB.
	/// DB Links can be added/deleted, tested (checking access of the TB_SITES table on remote DBs) and their list exported to ASCII files.</para>
	/// <para>A <i><b>queue</b></i> is an entity associated to a specific DB Link, and containing a time-ordered list of jobs to be executed against that DB. No job can be created without a related queue. Queue processing starts automatically at regular time intervals.
	/// Queues can be added/deleted, their next execution time can be changed, and their list can be exported to ASCII files.</para></description></item>
	/// <item><term>Job Management</term><description>This area is circled by the <i>Job Management</i> group box. Jobs can be created/deleted/monitored by tools in this area. In order to create a job of a certain type (<i>System</i>, <i>Brick</i>, <i>Operation</i>), 
	/// the corresponding button must be pressed. The job action can be one of the following:
	/// <para><b>Compare</b> compares the number of rows related, in each table, directly or indirectly, to the specified operation or brick; the publication GUID is also checked. Row content is <b><u>not checked</u></b>.</para>
	/// <para><b>Publish</b> uploads the current local system settings, a brick or an operation (along with all its children) to the remote DB.</para>
	/// <para><b>Unpublish</b> removes a brick or an operation (along with all its children) from the remote DB, using the local representation of the brick/operation.</para>
	/// <para><b>Copy</b> downloads the current local system settings, a brick or an operation (along with all its children) from the remote DB.</para>
	/// <para><b>Delete</b> deletes a brick or an operation (along with all its children) from the local DB.</para>
    /// <para><b>CS Candidates</b> downloads CS candidates for a CS doublet along with the geometrical parameters of the doublet. Notice this is a <b>Brick</b> job (the <c>New Brick Job</c> button must be pressed).</para>
	/// <para>A new job is always created with the currently selected queue. If no queue is selected, the job is not created.</para>
	/// <para>When a new job is created, it is in a <i>TODO</i> status. It must be explicitly scheduled (by means of the proper button). This additional requirement avoids that data be published without explicit human approval. Jobs in an <i>ABORTED</i> status can be 
	/// re-scheduled; however, it's important to check the log before rescheduling: if the reason for job abortion is not removed, it will probably fail again. If a <i>COPY</i> or <i>PUBLISH</i> job was interrupted, and some data had been already written by the time of
	/// interruption, in most cases manual removal of the relics are needed before the job can be started again; deregistration of the object from the publication subsystem is also needed in this case.</para>
	/// <para>The other buttons in this area allow to look at the log entries for the highlighted job (<see cref="SySal.Executables.OperaPublicationManager.LogForm"/>), at its current activity (<see cref="SySal.Executables.OperaPublicationManager.ActivityForm"/>) 
	/// or at execution statistics (<see cref="SySal.Executables.OperaPublicationManager.StatsForm"/>).</para>
	/// <para>The list of jobs can be exported to ASCII files.</para>	
	/// </description></item>
	/// <item><term>Object Management</term><description><para>Every object that is handled at least once by the publication subsystem is recorded there with its ID and a GUID. The GUID is a 32-digit hexadecimal code that is generated in such a way that it is statistically 
	/// unique all over the world. The Object Management area takes care of these records.</para>
	/// <para>The GUID changes every time the object is registered again in the publication system, i.e. at every version change. The <i>Version Check</i> button checks that the local GUID and the remote GUID match. The DB link used for the match is the currently selected DB link.
	/// If no DB link is selected, no check is performed.</para>
	/// <para>An object can be explicitly deregistered from the local publication system. However, if it has been registered in other DBs publication subsystems, it will no be deleted from there. Valid objects are not eligible for deregistration, nor can they be deregistered 
	/// if they're undergoing a write or delete operation. If the user wants to explicitly deregister an object, he/she must put the object in an <i>INVALID</i> status (by the <i>INVALIDATE</i> button); then the object can be deleted (by pressing <i>DEREGISTER</i>). This is a common
	/// situation when a very long job is interrupted by network or disk errors; then the job is stopped, but the object status is still in a <i>WRITING</i> or <i>DELETING</i> status, and it cannot be deleted nor overwritten until it is eliminated from the publication subsystem. 
	/// In such a situation, the user would mark the object as <i>INVALID</i>, then he/she will deregister it.</para>
	/// <para>Finally, the object list can be exported to ASCII files.</para></description></item>
	/// <item><term>Miscellanea</term><description>The last area, in the lower-right corner, contains several command buttons. The user can look at aggregated statistics for all jobs, or view all logs. Finally, the log can be purged of old entries (<see cref="SySal.Executables.OperaPublicationManager.PurgeLogForm"/>).</description></item>
	/// </list></para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.DataGrid gridJobs;
		private System.Windows.Forms.DataGrid gridDBLinks;
		private System.Windows.Forms.DataGrid gridQueues;
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private SySal.Controls.GroupBox groupBox1;
		private System.Windows.Forms.TextBox textPassword;
		private System.Windows.Forms.TextBox textUsername;
		private System.Windows.Forms.TextBox textDBServer;
		private SySal.Controls.GroupBox groupBox2;
		private SySal.Controls.StaticText staticText2;
		private SySal.Controls.StaticText staticText1;
		private SySal.Controls.StaticText staticText3;
		private SySal.Controls.StaticText staticText4;
		private SySal.Controls.StaticText staticText5;
		private SySal.Controls.GroupBox groupBox3;
		private System.Windows.Forms.DataGrid gridObjs;
		private SySal.Controls.Button btnConnect;
		private SySal.Controls.Button btnLinkAdd;
		private SySal.Controls.Button btnLinkDelete;
		private SySal.Controls.Button btnNewSystemJob;
		private SySal.Controls.Button btnNewBrickJob;
		private SySal.Controls.Button btnLinkTest;
		private SySal.Controls.Button btnLinkExport;
		private SySal.Controls.Button btnQueueExport;
		private SySal.Controls.Button btnQueueDelete;
		private SySal.Controls.Button btnQueueAdd;
		private SySal.Controls.Button btnNewOperationJob;
		private SySal.Controls.Button btnJobSchedule;
		private SySal.Controls.Button btnJobViewLog;
		private SySal.Controls.Button btnJobViewStats;
		private SySal.Controls.Button btnJobExport;
		private SySal.Controls.Button btnViewLog;
		private SySal.Controls.Button btnObjVersionCheck;
		private SySal.Controls.Button btnExit;
		private SySal.Controls.Button btnObjExport;
		private SySal.Controls.Button btnPurgeLog;
		private SySal.Controls.RadioButton radioNJPublish;
		private SySal.Controls.RadioButton radioNJCompare;
		private SySal.Controls.RadioButton radioNJCopy;
		private SySal.Controls.RadioButton radioNJUnpublish;
		private SySal.Controls.RadioButton radioNJDelete;
		private System.Windows.Forms.Panel TitlePanel;
		private SySal.Controls.StaticText staticText6;
		private SySal.Controls.StaticText staticText7;
		private SySal.Controls.StaticText staticText8;
		private SySal.Controls.Button btnViewStats;
		private SySal.Controls.Button btnJobViewActivity;
		private SySal.Controls.Button btnQueueChangeTime;
		private SySal.Controls.Button btnObjDeregister;
		private SySal.Controls.Button btnObjInvalidate;
		private SySal.Controls.RadioButton radioNJCSCands;
		private SySal.Controls.StaticText staticText9;		
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//		
			groupBox1.AdoptChild(gridDBLinks);
			groupBox1.AdoptChild(btnLinkAdd);
			groupBox1.AdoptChild(btnLinkTest);
			groupBox1.AdoptChild(btnLinkExport);
			groupBox1.AdoptChild(btnLinkDelete);
			groupBox1.AdoptChild(gridQueues);
			groupBox1.AdoptChild(btnQueueAdd);
			groupBox1.AdoptChild(btnQueueExport);
			groupBox1.AdoptChild(btnQueueDelete);	
			groupBox1.AdoptChild(btnQueueChangeTime);

			groupBox2.AdoptChild(gridJobs);
			groupBox2.AdoptChild(btnJobSchedule);
			groupBox2.AdoptChild(btnJobViewStats);
			groupBox2.AdoptChild(btnJobViewLog);
			groupBox2.AdoptChild(btnJobExport);
			groupBox2.AdoptChild(btnNewSystemJob);
			groupBox2.AdoptChild(btnNewBrickJob);
			groupBox2.AdoptChild(btnNewOperationJob);
			groupBox2.AdoptChild(radioNJCompare);
			groupBox2.AdoptChild(radioNJPublish);
			groupBox2.AdoptChild(radioNJUnpublish);
			groupBox2.AdoptChild(radioNJCopy);
			groupBox2.AdoptChild(radioNJDelete);
			groupBox2.AdoptChild(radioNJCSCands);
			groupBox2.AdoptChild(staticText1);
			groupBox2.AdoptChild(staticText2);
			groupBox2.AdoptChild(staticText3);
			groupBox2.AdoptChild(staticText4);
			groupBox2.AdoptChild(staticText5);

			groupBox3.AdoptChild(gridObjs);
			groupBox3.AdoptChild(btnObjVersionCheck);
			groupBox3.AdoptChild(btnObjExport);
			groupBox3.AdoptChild(btnObjDeregister);
			groupBox3.AdoptChild(btnObjInvalidate);
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
			this.gridJobs = new System.Windows.Forms.DataGrid();
			this.gridDBLinks = new System.Windows.Forms.DataGrid();
			this.gridQueues = new System.Windows.Forms.DataGrid();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.groupBox1 = new SySal.Controls.GroupBox();
			this.btnConnect = new SySal.Controls.Button();
			this.btnLinkAdd = new SySal.Controls.Button();
			this.btnLinkDelete = new SySal.Controls.Button();
			this.textPassword = new System.Windows.Forms.TextBox();
			this.textUsername = new System.Windows.Forms.TextBox();
			this.textDBServer = new System.Windows.Forms.TextBox();
			this.groupBox2 = new SySal.Controls.GroupBox();
			this.btnNewSystemJob = new SySal.Controls.Button();
			this.btnNewBrickJob = new SySal.Controls.Button();
			this.btnNewOperationJob = new SySal.Controls.Button();
			this.radioNJPublish = new SySal.Controls.RadioButton();
			this.radioNJCompare = new SySal.Controls.RadioButton();
			this.staticText2 = new SySal.Controls.StaticText();
			this.staticText1 = new SySal.Controls.StaticText();
			this.radioNJCopy = new SySal.Controls.RadioButton();
			this.staticText3 = new SySal.Controls.StaticText();
			this.radioNJUnpublish = new SySal.Controls.RadioButton();
			this.staticText4 = new SySal.Controls.StaticText();
			this.radioNJDelete = new SySal.Controls.RadioButton();
			this.staticText5 = new SySal.Controls.StaticText();
			this.btnJobSchedule = new SySal.Controls.Button();
			this.btnJobViewLog = new SySal.Controls.Button();
			this.btnViewLog = new SySal.Controls.Button();
			this.btnJobViewStats = new SySal.Controls.Button();
			this.btnLinkTest = new SySal.Controls.Button();
			this.groupBox3 = new SySal.Controls.GroupBox();
			this.gridObjs = new System.Windows.Forms.DataGrid();
			this.btnObjVersionCheck = new SySal.Controls.Button();
			this.btnExit = new SySal.Controls.Button();
			this.btnLinkExport = new SySal.Controls.Button();
			this.btnQueueExport = new SySal.Controls.Button();
			this.btnQueueDelete = new SySal.Controls.Button();
			this.btnQueueAdd = new SySal.Controls.Button();
			this.btnJobExport = new SySal.Controls.Button();
			this.btnObjExport = new SySal.Controls.Button();
			this.btnPurgeLog = new SySal.Controls.Button();
			this.TitlePanel = new System.Windows.Forms.Panel();
			this.staticText6 = new SySal.Controls.StaticText();
			this.staticText7 = new SySal.Controls.StaticText();
			this.staticText8 = new SySal.Controls.StaticText();
			this.btnViewStats = new SySal.Controls.Button();
			this.btnJobViewActivity = new SySal.Controls.Button();
			this.btnQueueChangeTime = new SySal.Controls.Button();
			this.btnObjDeregister = new SySal.Controls.Button();
			this.btnObjInvalidate = new SySal.Controls.Button();
			this.radioNJCSCands = new SySal.Controls.RadioButton();
			this.staticText9 = new SySal.Controls.StaticText();
			((System.ComponentModel.ISupportInitialize)(this.gridJobs)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.gridDBLinks)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.gridQueues)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.gridObjs)).BeginInit();
			this.SuspendLayout();
			// 
			// gridJobs
			// 
			this.gridJobs.AlternatingBackColor = System.Drawing.Color.White;
			this.gridJobs.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridJobs.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridJobs.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridJobs.CaptionForeColor = System.Drawing.Color.White;
			this.gridJobs.CaptionText = "Jobs (double-click to refresh)";
			this.gridJobs.DataMember = "";
			this.gridJobs.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridJobs.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridJobs.Location = new System.Drawing.Point(24, 320);
			this.gridJobs.Name = "gridJobs";
			this.gridJobs.PreferredColumnWidth = 120;
			this.gridJobs.ReadOnly = true;
			this.gridJobs.Size = new System.Drawing.Size(696, 160);
			this.gridJobs.TabIndex = 0;
			this.gridJobs.DoubleClick += new System.EventHandler(this.gridJobs_DoubleClick);
			// 
			// gridDBLinks
			// 
			this.gridDBLinks.AlternatingBackColor = System.Drawing.Color.White;
			this.gridDBLinks.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridDBLinks.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridDBLinks.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridDBLinks.CaptionForeColor = System.Drawing.Color.White;
			this.gridDBLinks.CaptionText = "DB Links (double-click to refresh)";
			this.gridDBLinks.DataMember = "";
			this.gridDBLinks.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridDBLinks.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridDBLinks.Location = new System.Drawing.Point(224, 96);
			this.gridDBLinks.Name = "gridDBLinks";
			this.gridDBLinks.ReadOnly = true;
			this.gridDBLinks.Size = new System.Drawing.Size(376, 144);
			this.gridDBLinks.TabIndex = 4;
			this.gridDBLinks.DoubleClick += new System.EventHandler(this.gridDBLink_DoubleClick);
			// 
			// gridQueues
			// 
			this.gridQueues.AlternatingBackColor = System.Drawing.Color.White;
			this.gridQueues.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridQueues.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridQueues.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridQueues.CaptionForeColor = System.Drawing.Color.White;
			this.gridQueues.CaptionText = "Queues (double-click to refresh)";
			this.gridQueues.DataMember = "";
			this.gridQueues.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridQueues.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridQueues.Location = new System.Drawing.Point(608, 96);
			this.gridQueues.Name = "gridQueues";
			this.gridQueues.PreferredColumnWidth = 120;
			this.gridQueues.ReadOnly = true;
			this.gridQueues.Size = new System.Drawing.Size(376, 144);
			this.gridQueues.TabIndex = 6;
			this.gridQueues.DoubleClick += new System.EventHandler(this.gridQueues_DoubleClick);
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 48);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(1008, 672);
			this.backgroundPanel1.TabIndex = 9;
			// 
			// groupBox1
			// 
			this.groupBox1.BackColor = System.Drawing.Color.White;
			this.groupBox1.ClosedPosition = new System.Drawing.Rectangle(8, 64, 992, 32);
			this.groupBox1.IsOpen = true;
			this.groupBox1.IsStatic = true;
			this.groupBox1.LabelText = "General";
			this.groupBox1.Location = new System.Drawing.Point(208, 64);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.OpenPosition = new System.Drawing.Rectangle(8, 64, 992, 256);
			this.groupBox1.Size = new System.Drawing.Size(792, 216);
			this.groupBox1.TabIndex = 10;
			this.groupBox1.OpenCloseEvent += new SySal.Controls.GroupBox.dOpenCloseEvent(this.OnOpenCloseGeneral);
			// 
			// btnConnect
			// 
			this.btnConnect.BackColor = System.Drawing.Color.White;
			this.btnConnect.ButtonText = "Connect to DB";
			this.btnConnect.Location = new System.Drawing.Point(16, 72);
			this.btnConnect.Name = "btnConnect";
			this.btnConnect.Size = new System.Drawing.Size(112, 24);
			this.btnConnect.TabIndex = 11;
			this.btnConnect.Click += new System.EventHandler(this.btnConnect_Click);
			this.btnConnect.DoubleClick += new System.EventHandler(this.btnConnect_Click);
			// 
			// btnLinkAdd
			// 
			this.btnLinkAdd.BackColor = System.Drawing.Color.White;
			this.btnLinkAdd.ButtonText = "Add";
			this.btnLinkAdd.Location = new System.Drawing.Point(224, 248);
			this.btnLinkAdd.Name = "btnLinkAdd";
			this.btnLinkAdd.Size = new System.Drawing.Size(56, 24);
			this.btnLinkAdd.TabIndex = 12;
			this.btnLinkAdd.Click += new System.EventHandler(this.btnDBLinkAdd_Click);
			this.btnLinkAdd.DoubleClick += new System.EventHandler(this.btnDBLinkAdd_Click);
			// 
			// btnLinkDelete
			// 
			this.btnLinkDelete.BackColor = System.Drawing.Color.White;
			this.btnLinkDelete.ButtonText = "Delete";
			this.btnLinkDelete.Location = new System.Drawing.Point(464, 248);
			this.btnLinkDelete.Name = "btnLinkDelete";
			this.btnLinkDelete.Size = new System.Drawing.Size(64, 24);
			this.btnLinkDelete.TabIndex = 13;
			this.btnLinkDelete.Click += new System.EventHandler(this.btnDBLinkDel_Click);
			this.btnLinkDelete.DoubleClick += new System.EventHandler(this.btnDBLinkDel_Click);
			// 
			// textPassword
			// 
			this.textPassword.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textPassword.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textPassword.ForeColor = System.Drawing.Color.Navy;
			this.textPassword.Location = new System.Drawing.Point(96, 160);
			this.textPassword.Name = "textPassword";
			this.textPassword.PasswordChar = '*';
			this.textPassword.Size = new System.Drawing.Size(104, 20);
			this.textPassword.TabIndex = 18;
			this.textPassword.Text = "";
			// 
			// textUsername
			// 
			this.textUsername.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textUsername.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textUsername.ForeColor = System.Drawing.Color.Navy;
			this.textUsername.Location = new System.Drawing.Point(96, 136);
			this.textUsername.Name = "textUsername";
			this.textUsername.Size = new System.Drawing.Size(104, 20);
			this.textUsername.TabIndex = 17;
			this.textUsername.Text = "";
			// 
			// textDBServer
			// 
			this.textDBServer.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textDBServer.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textDBServer.ForeColor = System.Drawing.Color.Navy;
			this.textDBServer.Location = new System.Drawing.Point(96, 112);
			this.textDBServer.Name = "textDBServer";
			this.textDBServer.Size = new System.Drawing.Size(104, 20);
			this.textDBServer.TabIndex = 16;
			this.textDBServer.Text = "";
			// 
			// groupBox2
			// 
			this.groupBox2.BackColor = System.Drawing.Color.White;
			this.groupBox2.ClosedPosition = new System.Drawing.Rectangle(8, 328, 992, 32);
			this.groupBox2.IsOpen = true;
			this.groupBox2.IsStatic = true;
			this.groupBox2.LabelText = "Job Management";
			this.groupBox2.Location = new System.Drawing.Point(8, 288);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.OpenPosition = new System.Drawing.Rectangle(8, 328, 992, 192);
			this.groupBox2.Size = new System.Drawing.Size(992, 232);
			this.groupBox2.TabIndex = 19;
			this.groupBox2.OpenCloseEvent += new SySal.Controls.GroupBox.dOpenCloseEvent(this.OnOpenCloseJobs);
			// 
			// btnNewSystemJob
			// 
			this.btnNewSystemJob.BackColor = System.Drawing.Color.White;
			this.btnNewSystemJob.ButtonText = "New System Job";
			this.btnNewSystemJob.Location = new System.Drawing.Point(728, 320);
			this.btnNewSystemJob.Name = "btnNewSystemJob";
			this.btnNewSystemJob.Size = new System.Drawing.Size(136, 24);
			this.btnNewSystemJob.TabIndex = 20;
			this.btnNewSystemJob.Click += new System.EventHandler(this.btnNewSystemJob_Click);
			this.btnNewSystemJob.DoubleClick += new System.EventHandler(this.btnNewSystemJob_Click);
			// 
			// btnNewBrickJob
			// 
			this.btnNewBrickJob.BackColor = System.Drawing.Color.White;
			this.btnNewBrickJob.ButtonText = "New Brick Job";
			this.btnNewBrickJob.Location = new System.Drawing.Point(728, 352);
			this.btnNewBrickJob.Name = "btnNewBrickJob";
			this.btnNewBrickJob.Size = new System.Drawing.Size(136, 24);
			this.btnNewBrickJob.TabIndex = 21;
			this.btnNewBrickJob.Click += new System.EventHandler(this.btnNewBrickJob_Click);
			this.btnNewBrickJob.DoubleClick += new System.EventHandler(this.btnNewBrickJob_Click);
			// 
			// btnNewOperationJob
			// 
			this.btnNewOperationJob.BackColor = System.Drawing.Color.White;
			this.btnNewOperationJob.ButtonText = "New Operation Job";
			this.btnNewOperationJob.Location = new System.Drawing.Point(728, 384);
			this.btnNewOperationJob.Name = "btnNewOperationJob";
			this.btnNewOperationJob.Size = new System.Drawing.Size(136, 24);
			this.btnNewOperationJob.TabIndex = 22;
			this.btnNewOperationJob.Click += new System.EventHandler(this.btnNewOperationJob_Click);
			this.btnNewOperationJob.DoubleClick += new System.EventHandler(this.btnNewOperationJob_Click);
			// 
			// radioNJPublish
			// 
			this.radioNJPublish.BackColor = System.Drawing.Color.White;
			this.radioNJPublish.Checked = false;
			this.radioNJPublish.Location = new System.Drawing.Point(880, 344);
			this.radioNJPublish.Name = "radioNJPublish";
			this.radioNJPublish.Size = new System.Drawing.Size(16, 16);
			this.radioNJPublish.TabIndex = 56;
			// 
			// radioNJCompare
			// 
			this.radioNJCompare.BackColor = System.Drawing.Color.White;
			this.radioNJCompare.Checked = false;
			this.radioNJCompare.Location = new System.Drawing.Point(880, 320);
			this.radioNJCompare.Name = "radioNJCompare";
			this.radioNJCompare.Size = new System.Drawing.Size(16, 16);
			this.radioNJCompare.TabIndex = 55;
			// 
			// staticText2
			// 
			this.staticText2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText2.LabelText = "Publish";
			this.staticText2.Location = new System.Drawing.Point(896, 344);
			this.staticText2.Name = "staticText2";
			this.staticText2.Size = new System.Drawing.Size(72, 24);
			this.staticText2.TabIndex = 54;
			this.staticText2.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// staticText1
			// 
			this.staticText1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText1.LabelText = "Compare";
			this.staticText1.Location = new System.Drawing.Point(896, 320);
			this.staticText1.Name = "staticText1";
			this.staticText1.Size = new System.Drawing.Size(72, 24);
			this.staticText1.TabIndex = 53;
			this.staticText1.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// radioNJCopy
			// 
			this.radioNJCopy.BackColor = System.Drawing.Color.White;
			this.radioNJCopy.Checked = false;
			this.radioNJCopy.Location = new System.Drawing.Point(880, 368);
			this.radioNJCopy.Name = "radioNJCopy";
			this.radioNJCopy.Size = new System.Drawing.Size(16, 16);
			this.radioNJCopy.TabIndex = 58;
			// 
			// staticText3
			// 
			this.staticText3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText3.LabelText = "Copy from Link";
			this.staticText3.Location = new System.Drawing.Point(896, 368);
			this.staticText3.Name = "staticText3";
			this.staticText3.Size = new System.Drawing.Size(96, 24);
			this.staticText3.TabIndex = 57;
			this.staticText3.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// radioNJUnpublish
			// 
			this.radioNJUnpublish.BackColor = System.Drawing.Color.White;
			this.radioNJUnpublish.Checked = false;
			this.radioNJUnpublish.Location = new System.Drawing.Point(880, 392);
			this.radioNJUnpublish.Name = "radioNJUnpublish";
			this.radioNJUnpublish.Size = new System.Drawing.Size(16, 16);
			this.radioNJUnpublish.TabIndex = 60;
			// 
			// staticText4
			// 
			this.staticText4.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText4.LabelText = "Unpublish";
			this.staticText4.Location = new System.Drawing.Point(896, 392);
			this.staticText4.Name = "staticText4";
			this.staticText4.Size = new System.Drawing.Size(96, 24);
			this.staticText4.TabIndex = 59;
			this.staticText4.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// radioNJDelete
			// 
			this.radioNJDelete.BackColor = System.Drawing.Color.White;
			this.radioNJDelete.Checked = false;
			this.radioNJDelete.Location = new System.Drawing.Point(880, 416);
			this.radioNJDelete.Name = "radioNJDelete";
			this.radioNJDelete.Size = new System.Drawing.Size(16, 16);
			this.radioNJDelete.TabIndex = 62;
			// 
			// staticText5
			// 
			this.staticText5.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText5.LabelText = "Delete local";
			this.staticText5.Location = new System.Drawing.Point(896, 416);
			this.staticText5.Name = "staticText5";
			this.staticText5.Size = new System.Drawing.Size(96, 24);
			this.staticText5.TabIndex = 61;
			this.staticText5.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// btnJobSchedule
			// 
			this.btnJobSchedule.BackColor = System.Drawing.Color.White;
			this.btnJobSchedule.ButtonText = "Schedule";
			this.btnJobSchedule.Location = new System.Drawing.Point(24, 488);
			this.btnJobSchedule.Name = "btnJobSchedule";
			this.btnJobSchedule.Size = new System.Drawing.Size(88, 24);
			this.btnJobSchedule.TabIndex = 65;
			this.btnJobSchedule.Click += new System.EventHandler(this.btnJobSchedule_Click);
			this.btnJobSchedule.DoubleClick += new System.EventHandler(this.btnJobSchedule_Click);
			// 
			// btnJobViewLog
			// 
			this.btnJobViewLog.BackColor = System.Drawing.Color.White;
			this.btnJobViewLog.ButtonText = "View related log";
			this.btnJobViewLog.Location = new System.Drawing.Point(136, 488);
			this.btnJobViewLog.Name = "btnJobViewLog";
			this.btnJobViewLog.Size = new System.Drawing.Size(128, 24);
			this.btnJobViewLog.TabIndex = 66;
			this.btnJobViewLog.Click += new System.EventHandler(this.btnJobViewLog_Click);
			this.btnJobViewLog.DoubleClick += new System.EventHandler(this.btnJobViewLog_Click);
			// 
			// btnViewLog
			// 
			this.btnViewLog.BackColor = System.Drawing.Color.White;
			this.btnViewLog.ButtonText = "View full log";
			this.btnViewLog.Location = new System.Drawing.Point(880, 576);
			this.btnViewLog.Name = "btnViewLog";
			this.btnViewLog.Size = new System.Drawing.Size(112, 24);
			this.btnViewLog.TabIndex = 67;
			this.btnViewLog.Click += new System.EventHandler(this.btnViewLog_Click);
			this.btnViewLog.DoubleClick += new System.EventHandler(this.btnViewLog_Click);
			// 
			// btnJobViewStats
			// 
			this.btnJobViewStats.BackColor = System.Drawing.Color.White;
			this.btnJobViewStats.ButtonText = "View related statistics";
			this.btnJobViewStats.Location = new System.Drawing.Point(288, 488);
			this.btnJobViewStats.Name = "btnJobViewStats";
			this.btnJobViewStats.Size = new System.Drawing.Size(152, 24);
			this.btnJobViewStats.TabIndex = 68;
			this.btnJobViewStats.Click += new System.EventHandler(this.btnJobViewStats_Click);
			this.btnJobViewStats.DoubleClick += new System.EventHandler(this.btnJobViewStats_Click);
			// 
			// btnLinkTest
			// 
			this.btnLinkTest.BackColor = System.Drawing.Color.White;
			this.btnLinkTest.ButtonText = "Test";
			this.btnLinkTest.Location = new System.Drawing.Point(288, 248);
			this.btnLinkTest.Name = "btnLinkTest";
			this.btnLinkTest.Size = new System.Drawing.Size(56, 24);
			this.btnLinkTest.TabIndex = 69;
			this.btnLinkTest.Click += new System.EventHandler(this.btnDBLinkTest_Click);
			this.btnLinkTest.DoubleClick += new System.EventHandler(this.btnDBLinkTest_Click);
			// 
			// groupBox3
			// 
			this.groupBox3.BackColor = System.Drawing.Color.White;
			this.groupBox3.ClosedPosition = new System.Drawing.Rectangle(8, 520, 992, 32);
			this.groupBox3.IsOpen = true;
			this.groupBox3.IsStatic = true;
			this.groupBox3.LabelText = "Object Management";
			this.groupBox3.Location = new System.Drawing.Point(8, 528);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.OpenPosition = new System.Drawing.Rectangle(8, 520, 992, 144);
			this.groupBox3.Size = new System.Drawing.Size(776, 184);
			this.groupBox3.TabIndex = 70;
			this.groupBox3.OpenCloseEvent += new SySal.Controls.GroupBox.dOpenCloseEvent(this.OnOpenCloseObjs);
			// 
			// gridObjs
			// 
			this.gridObjs.AlternatingBackColor = System.Drawing.Color.White;
			this.gridObjs.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridObjs.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridObjs.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridObjs.CaptionForeColor = System.Drawing.Color.White;
			this.gridObjs.CaptionText = "Published Objects (double-click to refresh)";
			this.gridObjs.DataMember = "";
			this.gridObjs.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridObjs.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridObjs.Location = new System.Drawing.Point(24, 560);
			this.gridObjs.Name = "gridObjs";
			this.gridObjs.PreferredColumnWidth = 120;
			this.gridObjs.ReadOnly = true;
			this.gridObjs.Size = new System.Drawing.Size(600, 144);
			this.gridObjs.TabIndex = 71;
			this.gridObjs.DoubleClick += new System.EventHandler(this.gridObjs_DoubleClick);
			// 
			// btnObjVersionCheck
			// 
			this.btnObjVersionCheck.BackColor = System.Drawing.Color.White;
			this.btnObjVersionCheck.ButtonText = "Version check";
			this.btnObjVersionCheck.Location = new System.Drawing.Point(632, 560);
			this.btnObjVersionCheck.Name = "btnObjVersionCheck";
			this.btnObjVersionCheck.Size = new System.Drawing.Size(136, 24);
			this.btnObjVersionCheck.TabIndex = 72;
			this.btnObjVersionCheck.Click += new System.EventHandler(this.btnObjsVersionCheck_Click);
			this.btnObjVersionCheck.DoubleClick += new System.EventHandler(this.btnObjsVersionCheck_Click);
			// 
			// btnExit
			// 
			this.btnExit.BackColor = System.Drawing.Color.White;
			this.btnExit.ButtonText = "Exit";
			this.btnExit.Location = new System.Drawing.Point(880, 680);
			this.btnExit.Name = "btnExit";
			this.btnExit.Size = new System.Drawing.Size(112, 24);
			this.btnExit.TabIndex = 74;
			this.btnExit.Click += new System.EventHandler(this.btnExit_Click);
			this.btnExit.DoubleClick += new System.EventHandler(this.btnExit_Click);
			// 
			// btnLinkExport
			// 
			this.btnLinkExport.BackColor = System.Drawing.Color.White;
			this.btnLinkExport.ButtonText = "Export";
			this.btnLinkExport.Location = new System.Drawing.Point(352, 248);
			this.btnLinkExport.Name = "btnLinkExport";
			this.btnLinkExport.Size = new System.Drawing.Size(64, 24);
			this.btnLinkExport.TabIndex = 75;
			this.btnLinkExport.Click += new System.EventHandler(this.btnDBLinkExport_Click);
			this.btnLinkExport.DoubleClick += new System.EventHandler(this.btnDBLinkExport_Click);
			// 
			// btnQueueExport
			// 
			this.btnQueueExport.BackColor = System.Drawing.Color.White;
			this.btnQueueExport.ButtonText = "Export";
			this.btnQueueExport.Location = new System.Drawing.Point(672, 248);
			this.btnQueueExport.Name = "btnQueueExport";
			this.btnQueueExport.Size = new System.Drawing.Size(64, 24);
			this.btnQueueExport.TabIndex = 79;
			this.btnQueueExport.Click += new System.EventHandler(this.btnQueuesExport_Click);
			this.btnQueueExport.DoubleClick += new System.EventHandler(this.btnQueuesExport_Click);
			// 
			// btnQueueDelete
			// 
			this.btnQueueDelete.BackColor = System.Drawing.Color.White;
			this.btnQueueDelete.ButtonText = "Delete";
			this.btnQueueDelete.Location = new System.Drawing.Point(920, 248);
			this.btnQueueDelete.Name = "btnQueueDelete";
			this.btnQueueDelete.Size = new System.Drawing.Size(64, 24);
			this.btnQueueDelete.TabIndex = 77;
			this.btnQueueDelete.Click += new System.EventHandler(this.btnQueueDel_Click);
			this.btnQueueDelete.DoubleClick += new System.EventHandler(this.btnQueueDel_Click);
			// 
			// btnQueueAdd
			// 
			this.btnQueueAdd.BackColor = System.Drawing.Color.White;
			this.btnQueueAdd.ButtonText = "Add";
			this.btnQueueAdd.Location = new System.Drawing.Point(608, 248);
			this.btnQueueAdd.Name = "btnQueueAdd";
			this.btnQueueAdd.Size = new System.Drawing.Size(56, 24);
			this.btnQueueAdd.TabIndex = 76;
			this.btnQueueAdd.Click += new System.EventHandler(this.btnQueueAdd_Click);
			this.btnQueueAdd.DoubleClick += new System.EventHandler(this.btnQueueAdd_Click);
			// 
			// btnJobExport
			// 
			this.btnJobExport.BackColor = System.Drawing.Color.White;
			this.btnJobExport.ButtonText = "Export";
			this.btnJobExport.Location = new System.Drawing.Point(656, 488);
			this.btnJobExport.Name = "btnJobExport";
			this.btnJobExport.Size = new System.Drawing.Size(64, 24);
			this.btnJobExport.TabIndex = 80;
			this.btnJobExport.Click += new System.EventHandler(this.btnJobExport_Click);
			this.btnJobExport.DoubleClick += new System.EventHandler(this.btnJobExport_Click);
			// 
			// btnObjExport
			// 
			this.btnObjExport.BackColor = System.Drawing.Color.White;
			this.btnObjExport.ButtonText = "Export";
			this.btnObjExport.Location = new System.Drawing.Point(632, 592);
			this.btnObjExport.Name = "btnObjExport";
			this.btnObjExport.Size = new System.Drawing.Size(136, 24);
			this.btnObjExport.TabIndex = 81;
			this.btnObjExport.Click += new System.EventHandler(this.btnObjExport_Click);
			this.btnObjExport.DoubleClick += new System.EventHandler(this.btnObjExport_Click);
			// 
			// btnPurgeLog
			// 
			this.btnPurgeLog.BackColor = System.Drawing.Color.White;
			this.btnPurgeLog.ButtonText = "Purge log";
			this.btnPurgeLog.Location = new System.Drawing.Point(880, 608);
			this.btnPurgeLog.Name = "btnPurgeLog";
			this.btnPurgeLog.Size = new System.Drawing.Size(112, 24);
			this.btnPurgeLog.TabIndex = 82;
			this.btnPurgeLog.Click += new System.EventHandler(this.btnPurgeLog_Click);
			this.btnPurgeLog.DoubleClick += new System.EventHandler(this.btnPurgeLog_Click);
			// 
			// TitlePanel
			// 
			this.TitlePanel.Location = new System.Drawing.Point(0, 0);
			this.TitlePanel.Name = "TitlePanel";
			this.TitlePanel.Size = new System.Drawing.Size(1008, 48);
			this.TitlePanel.TabIndex = 83;
			this.TitlePanel.Paint += new System.Windows.Forms.PaintEventHandler(this.OnTitlePaint);
			this.TitlePanel.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnTitleMouseMove);
			this.TitlePanel.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnTitleMouseDown);
			// 
			// staticText6
			// 
			this.staticText6.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText6.LabelText = "DB Server";
			this.staticText6.Location = new System.Drawing.Point(16, 112);
			this.staticText6.Name = "staticText6";
			this.staticText6.Size = new System.Drawing.Size(72, 24);
			this.staticText6.TabIndex = 84;
			this.staticText6.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// staticText7
			// 
			this.staticText7.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText7.LabelText = "Username";
			this.staticText7.Location = new System.Drawing.Point(16, 136);
			this.staticText7.Name = "staticText7";
			this.staticText7.Size = new System.Drawing.Size(72, 24);
			this.staticText7.TabIndex = 85;
			this.staticText7.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// staticText8
			// 
			this.staticText8.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText8.LabelText = "Password";
			this.staticText8.Location = new System.Drawing.Point(16, 160);
			this.staticText8.Name = "staticText8";
			this.staticText8.Size = new System.Drawing.Size(72, 24);
			this.staticText8.TabIndex = 86;
			this.staticText8.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// btnViewStats
			// 
			this.btnViewStats.BackColor = System.Drawing.Color.White;
			this.btnViewStats.ButtonText = "View statistics";
			this.btnViewStats.Location = new System.Drawing.Point(880, 544);
			this.btnViewStats.Name = "btnViewStats";
			this.btnViewStats.Size = new System.Drawing.Size(112, 24);
			this.btnViewStats.TabIndex = 87;
			this.btnViewStats.Click += new System.EventHandler(this.btnViewStats_Click);
			this.btnViewStats.DoubleClick += new System.EventHandler(this.btnViewStats_Click);
			// 
			// btnJobViewActivity
			// 
			this.btnJobViewActivity.BackColor = System.Drawing.Color.White;
			this.btnJobViewActivity.ButtonText = "View job activity";
			this.btnJobViewActivity.Location = new System.Drawing.Point(472, 488);
			this.btnJobViewActivity.Name = "btnJobViewActivity";
			this.btnJobViewActivity.Size = new System.Drawing.Size(120, 24);
			this.btnJobViewActivity.TabIndex = 88;
			this.btnJobViewActivity.Click += new System.EventHandler(this.btnJobViewActivity_Click);
			this.btnJobViewActivity.DoubleClick += new System.EventHandler(this.btnJobViewActivity_Click);
			// 
			// btnQueueChangeTime
			// 
			this.btnQueueChangeTime.BackColor = System.Drawing.Color.White;
			this.btnQueueChangeTime.ButtonText = "Change time";
			this.btnQueueChangeTime.Location = new System.Drawing.Point(752, 248);
			this.btnQueueChangeTime.Name = "btnQueueChangeTime";
			this.btnQueueChangeTime.Size = new System.Drawing.Size(104, 24);
			this.btnQueueChangeTime.TabIndex = 89;
			this.btnQueueChangeTime.Click += new System.EventHandler(this.btnQueueChangeTime_Click);
			this.btnQueueChangeTime.DoubleClick += new System.EventHandler(this.btnQueueChangeTime_Click);
			// 
			// btnObjDeregister
			// 
			this.btnObjDeregister.BackColor = System.Drawing.Color.White;
			this.btnObjDeregister.ButtonText = "Deregister";
			this.btnObjDeregister.Location = new System.Drawing.Point(632, 624);
			this.btnObjDeregister.Name = "btnObjDeregister";
			this.btnObjDeregister.Size = new System.Drawing.Size(136, 24);
			this.btnObjDeregister.TabIndex = 90;
			this.btnObjDeregister.Click += new System.EventHandler(this.btnObjDeregister_Click);
			this.btnObjDeregister.DoubleClick += new System.EventHandler(this.btnObjDeregister_Click);
			// 
			// btnObjInvalidate
			// 
			this.btnObjInvalidate.BackColor = System.Drawing.Color.White;
			this.btnObjInvalidate.ButtonText = "Invalidate";
			this.btnObjInvalidate.Location = new System.Drawing.Point(632, 656);
			this.btnObjInvalidate.Name = "btnObjInvalidate";
			this.btnObjInvalidate.Size = new System.Drawing.Size(136, 24);
			this.btnObjInvalidate.TabIndex = 91;
			this.btnObjInvalidate.Click += new System.EventHandler(this.btnObjInvalidate_Click);
			this.btnObjInvalidate.DoubleClick += new System.EventHandler(this.btnObjInvalidate_Click);
			// 
			// radioNJCSCands
			// 
			this.radioNJCSCands.BackColor = System.Drawing.Color.White;
			this.radioNJCSCands.Checked = false;
			this.radioNJCSCands.Location = new System.Drawing.Point(880, 440);
			this.radioNJCSCands.Name = "radioNJCSCands";
			this.radioNJCSCands.Size = new System.Drawing.Size(16, 16);
			this.radioNJCSCands.TabIndex = 93;
			// 
			// staticText9
			// 
			this.staticText9.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText9.LabelText = "CS Candidates";
			this.staticText9.Location = new System.Drawing.Point(896, 440);
			this.staticText9.Name = "staticText9";
			this.staticText9.Size = new System.Drawing.Size(96, 24);
			this.staticText9.TabIndex = 92;
			this.staticText9.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// MainForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(1006, 718);
			this.ControlBox = false;
			this.Controls.Add(this.radioNJCSCands);
			this.Controls.Add(this.staticText9);
			this.Controls.Add(this.btnObjInvalidate);
			this.Controls.Add(this.btnObjDeregister);
			this.Controls.Add(this.btnQueueChangeTime);
			this.Controls.Add(this.btnJobViewActivity);
			this.Controls.Add(this.btnViewStats);
			this.Controls.Add(this.staticText8);
			this.Controls.Add(this.staticText7);
			this.Controls.Add(this.btnExit);
			this.Controls.Add(this.btnPurgeLog);
			this.Controls.Add(this.btnViewLog);
			this.Controls.Add(this.staticText6);
			this.Controls.Add(this.btnLinkExport);
			this.Controls.Add(this.btnQueueExport);
			this.Controls.Add(this.btnQueueDelete);
			this.Controls.Add(this.btnQueueAdd);
			this.Controls.Add(this.gridDBLinks);
			this.Controls.Add(this.gridQueues);
			this.Controls.Add(this.btnConnect);
			this.Controls.Add(this.btnLinkAdd);
			this.Controls.Add(this.btnLinkDelete);
			this.Controls.Add(this.textPassword);
			this.Controls.Add(this.textUsername);
			this.Controls.Add(this.btnLinkTest);
			this.Controls.Add(this.textDBServer);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.btnObjExport);
			this.Controls.Add(this.btnJobExport);
			this.Controls.Add(this.btnObjVersionCheck);
			this.Controls.Add(this.gridObjs);
			this.Controls.Add(this.groupBox3);
			this.Controls.Add(this.btnJobViewStats);
			this.Controls.Add(this.btnJobViewLog);
			this.Controls.Add(this.btnJobSchedule);
			this.Controls.Add(this.gridJobs);
			this.Controls.Add(this.radioNJDelete);
			this.Controls.Add(this.staticText5);
			this.Controls.Add(this.radioNJUnpublish);
			this.Controls.Add(this.staticText4);
			this.Controls.Add(this.radioNJCopy);
			this.Controls.Add(this.staticText3);
			this.Controls.Add(this.radioNJPublish);
			this.Controls.Add(this.radioNJCompare);
			this.Controls.Add(this.staticText2);
			this.Controls.Add(this.staticText1);
			this.Controls.Add(this.btnNewOperationJob);
			this.Controls.Add(this.btnNewBrickJob);
			this.Controls.Add(this.btnNewSystemJob);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.backgroundPanel1);
			this.Controls.Add(this.TitlePanel);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "MainForm";
			this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
			((System.ComponentModel.ISupportInitialize)(this.gridJobs)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.gridDBLinks)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.gridQueues)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.gridObjs)).EndInit();
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

		SySal.OperaDb.OperaDbCredentials Cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();

		SySal.OperaDb.OperaDbConnection Conn = null;

		LoginForm LogForm = new LoginForm();

/*
		private void menuLogRefresh_Click(object sender, System.EventArgs e)
		{
			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			try
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, DBLINK, OBJID, TO_CHAR(TIMESTAMP, 'DD-MM-YYYY hh24:mi:ss') as TIMESTAMP, TEXT FROM PT_LOG ORDER BY TIMESTAMP DESC", Conn, null).Fill(ds);
				gridLog.DataSource = ds.Tables[0];
			}
			finally
			{
				this.Cursor = oldc;
			}
		}
*/
		DBLinkForm DBLinkCreateForm = new DBLinkForm();

		QueueForm QueueTimeInfo = new QueueForm();

		AddBricksForm BricksForm = new AddBricksForm();

		AddOperationsForm OperationForm = new AddOperationsForm();

        CSCandsForm CSForm = new CSCandsForm();

		static internal void ExportToFile(System.Data.DataTable dt, string filename)
		{
			System.IO.StreamWriter wr = null;
			try
			{
				wr = new System.IO.StreamWriter(filename);
				int i;
				for (i = 0; i < dt.Columns.Count; i++)
				{	
					if (i > 0) wr.Write("\t");
					wr.Write(dt.Columns[i].ColumnName);
				}
				foreach (System.Data.DataRow dr in dt.Rows)
				{
					wr.WriteLine();
					for (i = 0; i < dt.Columns.Count; i++)
					{	
						if (i > 0) wr.Write("\t");
						wr.Write(dr[i].ToString());
					}
				}
				wr.Close();
			}
			catch (Exception x)
			{
				if (wr != null) wr.Close();
				wr = null;
				MessageBox.Show(x.Message, "Export failed!", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}	
		}

		PurgeLogForm LogPForm = new PurgeLogForm();

		private void btnConnect_Click(object sender, System.EventArgs e)
		{
			Cred.DBServer = textDBServer.Text;
			Cred.DBUserName = textUsername.Text;
			Cred.DBPassword = textPassword.Text;
			if (Conn != null)
			{
				Conn.Close();
				Conn = null;
			}				
			try
			{
				Conn = new SySal.OperaDb.OperaDbConnection(Cred.DBServer, Cred.DBUserName, Cred.DBPassword);
				Conn.Open();
				gridDBLink_DoubleClick(this, null);				
				gridQueues_DoubleClick(this, null);
				System.Data.DataTable dt = new DataTable();
				gridJobs.DataSource = dt;
				gridObjs.DataSource = dt;
			}
			catch (Exception x)
			{
				if (Conn != null)
				{
					Conn.Close();
					Conn = null;
				}
				MessageBox.Show(x.Message, "Connection error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				System.Data.DataTable dt = new DataTable();
				gridDBLinks.DataSource = dt;
				gridQueues.DataSource = dt;
				gridJobs.DataSource = dt;
				gridObjs.DataSource = dt;
			}
			
		
		}

		private void gridDBLink_DoubleClick(object sender, System.EventArgs e)
		{
			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			try
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT DB_LINK as DBLINK, HOST, OWNER, USERNAME, TO_CHAR(CREATED, 'DD-MM-YYYY hh:mi:ss') as CREATIONTIME FROM ALL_DB_LINKS ORDER BY DB_LINK ASC", Conn, null).Fill(ds);
				gridDBLinks.DataSource = ds.Tables[0];
			}
			finally
			{
				this.Cursor = oldc;
			}		
		}

		private void gridJobs_DoubleClick(object sender, System.EventArgs e)
		{
			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			try
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, DBLINK, OBJID, TYPE, STATUS, nvl(TO_CHAR(BEGINTIME, 'DD-MM-YYYY hh24:mi:ss'), '') as BEGINTIME, nvl(TO_CHAR(ENDTIME, 'DD-MM-YYYY hh24:mi:ss'), '') as ENDTIME, PROGRESS FROM PT_JOBS ORDER BY ID DESC", Conn, null).Fill(ds);
				gridJobs.DataSource = ds.Tables[0];			
			}
			finally
			{
				this.Cursor = oldc;
			}		
		}

		private void gridQueues_DoubleClick(object sender, System.EventArgs e)
		{
			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			try
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT DBLINK, JOB as QUEUE, LOG_USER, PRIV_USER, SCHEMA_USER, TO_CHAR(NEXT_DATE, 'DD-MM-YYYY hh24:mi:ss') as NEXTDATE, INTERVAL FROM DBA_JOBS INNER JOIN PT_GENERAL ON (JOB = QUEUEID) ORDER BY DBLINK ASC", Conn, null).Fill(ds);
				gridQueues.DataSource = ds.Tables[0];
			}
			finally
			{
				this.Cursor = oldc;
			}		
		}

		private void gridObjs_DoubleClick(object sender, System.EventArgs e)
		{
			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			try
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT TYPE, OBJID, TOKEN, STATUS FROM PT_OBJECTS ORDER BY STATUS ASC", Conn, null).Fill(ds);
				gridObjs.DataSource = ds.Tables[0];
			}
			finally
			{
				this.Cursor = oldc;
			}		
		}

		private void btnDBLinkExport_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile((System.Data.DataTable)gridDBLinks.DataSource, sdlg.FileName);						
		}

		private void btnQueuesExport_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile((System.Data.DataTable)gridQueues.DataSource, sdlg.FileName);		
		}

		private void btnJobExport_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile((System.Data.DataTable)gridJobs.DataSource, sdlg.FileName);				
		}

		private void btnObjExport_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile((System.Data.DataTable)gridObjs.DataSource, sdlg.FileName);						
		}

		private void btnDBLinkAdd_Click(object sender, System.EventArgs e)
		{
			if (DBLinkCreateForm.ShowDialog() == DialogResult.OK)
			{
				try
				{
					new SySal.OperaDb.OperaDbCommand("CREATE DATABASE LINK " + DBLinkCreateForm.textLinkName.Text + " CONNECT TO " + DBLinkCreateForm.textUsername.Text + " IDENTIFIED BY \"" + DBLinkCreateForm.textPassword.Text + "\" USING '" + DBLinkCreateForm.textDBServer.Text + "'", Conn, null).ExecuteNonQuery();
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "DB Link creation error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				gridDBLink_DoubleClick(null, null);
			}		
		}

		private void btnDBLinkTest_Click(object sender, System.EventArgs e)
		{
			if (gridDBLinks.CurrentRowIndex < 0) return;
			string dblink = gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString();
			try
			{
				new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_SITES@" + dblink, Conn, null).ExecuteScalar();
				MessageBox.Show("Successfully tested access to TB_SITES@" + dblink, "DB Link OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
			catch(Exception x)
			{
				MessageBox.Show(x.Message, "DB Link testing failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		
		}

		private void btnDBLinkDel_Click(object sender, System.EventArgs e)
		{
			if (gridDBLinks.CurrentRowIndex < 0) return;
			try
			{				
				new SySal.OperaDb.OperaDbCommand("DROP DATABASE LINK " + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString(), Conn, null).ExecuteNonQuery();
			}
			catch(Exception x)
			{
				MessageBox.Show(x.Message, "DB Link deletion error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			gridDBLink_DoubleClick(null, null);
		}

		private void btnQueueAdd_Click(object sender, System.EventArgs e)
		{
			if (gridDBLinks.CurrentRowIndex < 0) return;
			string dblink = gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString();
			QueueTimeInfo.textDBLink.Text = dblink;
			QueueTimeInfo.timeFirstSchedule.Value = System.Convert.ToDateTime(new SySal.OperaDb.OperaDbCommand("SELECT SYSTIMESTAMP FROM DUAL", Conn, null).ExecuteScalar()).AddSeconds(20);
			if (QueueTimeInfo.ShowDialog() == DialogResult.OK)
				try
				{				
					SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_QUEUE('" + dblink + "', :firstdate, " + QueueTimeInfo.comboInterval.Text + ", :qid)", Conn, null);
					cmd.Parameters.Add("firstdate", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input).Value = QueueTimeInfo.timeFirstSchedule.Value;
					cmd.Parameters.Add("qid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output).Value = 0;
					cmd.ExecuteNonQuery();
					new SySal.OperaDb.OperaDbCommand("COMMIT", Conn, null).ExecuteNonQuery();
					gridQueues_DoubleClick(null, null);
				}
				catch(Exception x)
				{
					MessageBox.Show(x.Message, "Queue creation error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}							
		}

		private void btnQueueDel_Click(object sender, System.EventArgs e)
		{
			if (gridQueues.CurrentRowIndex < 0) return;
			try
			{
				new SySal.OperaDb.OperaDbCommand("call PP_DEL_PUBLICATION_QUEUE('" + gridQueues[gridQueues.CurrentRowIndex, 0].ToString() + "')", Conn, null).ExecuteNonQuery();
				new SySal.OperaDb.OperaDbCommand("COMMIT", Conn, null).ExecuteNonQuery();
				gridQueues_DoubleClick(null, null);
			}
			catch(Exception x)
			{
				MessageBox.Show(x.Message, "Queue deletion error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void btnQueueChangeTime_Click(object sender, System.EventArgs e)
		{
			if (gridQueues.CurrentRowIndex < 0) return;
			string dblink = gridQueues[gridQueues.CurrentRowIndex, 0].ToString();
			QueueTimeInfo.textDBLink.Text = dblink;
			QueueTimeInfo.timeFirstSchedule.Value = System.Convert.ToDateTime(new SySal.OperaDb.OperaDbCommand("SELECT SYSTIMESTAMP FROM DUAL", Conn, null).ExecuteScalar()).AddSeconds(20);		
			if (QueueTimeInfo.ShowDialog() == DialogResult.OK)
				try
				{				
					SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("call PP_CHANGE_QUEUE_TIME(" + gridQueues[gridQueues.CurrentRowIndex, 1].ToString() + ", :firstdate, " + QueueTimeInfo.comboInterval.Text + ")", Conn, null);
					cmd.Parameters.Add("firstdate", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input).Value = QueueTimeInfo.timeFirstSchedule.Value;
					cmd.ExecuteNonQuery();
					gridQueues_DoubleClick(null, null);
				}
				catch(Exception x)
				{
					MessageBox.Show(x.Message, "Error changing queue time", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}							


		}

		private void btnJobSchedule_Click(object sender, System.EventArgs e)
		{
			try
			{
				int i;
				int len = ((System.Data.DataTable)gridJobs.DataSource).Rows.Count;
				for (i = 0; i < len; i++)
					if (gridJobs.IsSelected(i))
						new SySal.OperaDb.OperaDbCommand("call PP_SCHEDULE_PUBLICATION_JOB(" + gridJobs[i, 0] + ")", Conn, null).ExecuteNonQuery();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Scheduling failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			gridJobs_DoubleClick(null, null);								
		}

		internal enum JobType { Null, Compare, Publish, CopyFromLink, Unpublish, DeleteLocal, CSCandidates }

		private JobType CurrentJobType 
		{
			get
			{
				if (radioNJCompare.Checked) return JobType.Compare;
				if (radioNJPublish.Checked) return JobType.Publish;
				if (radioNJCopy.Checked) return JobType.CopyFromLink;
				if (radioNJUnpublish.Checked) return JobType.Unpublish;
				if (radioNJDelete.Checked) return JobType.DeleteLocal;
				if (radioNJCSCands.Checked) return JobType.CSCandidates;
				return JobType.Null;
			}
		}

		private void btnNewSystemJob_Click(object sender, System.EventArgs e)
		{			
			if (gridDBLinks.CurrentRowIndex < 0) return;
			try
			{
				SySal.OperaDb.OperaDbCommand cmd = null;
				switch (CurrentJobType)
				{
					case JobType.Publish: cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(" + 
											  new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn, null).ExecuteScalar() + 
											  ", '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'PUB_SYSTEM', :qid)", 
											  Conn, null);
						break;

					case JobType.CopyFromLink: cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(" + 
												   new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn, null).ExecuteScalar() + 
												   ", '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'COPY_SYSTEM', :qid)", 
												   Conn, null);
                        break;

                    default: return;
				}
				cmd.Parameters.Add("qid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Job creation failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			gridJobs_DoubleClick(null, null);
		}

		private void btnNewBrickJob_Click(object sender, System.EventArgs e)
		{
			if (gridDBLinks.CurrentRowIndex < 0) return;
			JobType jt = CurrentJobType;
            if (jt == JobType.CSCandidates)
            {
                CSForm.ShowDialog(Conn, gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString(), jt);
                return;
            }
			if (BricksForm.ShowDialog(Conn, gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString(), jt) == DialogResult.OK)
			{
				SySal.OperaDb.OperaDbCommand cmd = null;
				switch (CurrentJobType)
				{
					case JobType.Compare:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'CMP_BRICK', :jid)", Conn, null); break;
					case JobType.Publish:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'PUB_BRICK', :jid)", Conn, null); break;
					case JobType.Unpublish:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'UNPUB_BRICK', :jid)", Conn, null); break;
					case JobType.CopyFromLink:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'COPY_BRICK', :jid)", Conn, null); break;
					case JobType.DeleteLocal:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'DEL_BRICK', :jid)", Conn, null); break;
				}

				cmd.Parameters.Add("xid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				cmd.Parameters.Add("jid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				foreach (string s in BricksForm.Ids)
				{
					try
					{
						cmd.Parameters[0].Value = SySal.OperaDb.Convert.ToInt64(s);
						cmd.ExecuteNonQuery();
					}
					catch(Exception x)
					{
						MessageBox.Show(x.Message, "Job creation failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
				}
			}
			gridJobs_DoubleClick(null, null);
		}

		private void btnNewOperationJob_Click(object sender, System.EventArgs e)
		{
			if (gridDBLinks.CurrentRowIndex < 0) return;
			JobType jt = CurrentJobType;
            if (jt == JobType.CSCandidates) return;
			if (OperationForm.ShowDialog(Conn, gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString(), jt) == DialogResult.OK)
			{
				SySal.OperaDb.OperaDbCommand cmd = null;
				switch (jt)
				{
					case JobType.Compare:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'CMP_OPERATION', :jid)", Conn, null); break;
					case JobType.Publish:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'PUB_OPERATION', :jid)", Conn, null); break;
					case JobType.Unpublish:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'UNPUB_OPERATION', :jid)", Conn, null); break;
					case JobType.CopyFromLink:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'COPY_OP', :jid)", Conn, null); break;
					case JobType.DeleteLocal:	cmd = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "', 'DEL_OPERATION', :jid)", Conn, null); break;                    
				}
				cmd.Parameters.Add("xid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				cmd.Parameters.Add("jid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				foreach (string s in OperationForm.Ids)
				{
					try
					{
						cmd.Parameters[0].Value = SySal.OperaDb.Convert.ToInt64(s);
						cmd.ExecuteNonQuery();
					}
					catch(Exception x)
					{
						MessageBox.Show(x.Message, "Scheduling failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
				}
			}
			gridJobs_DoubleClick(null, null);
		
		}

		private Point DragPoint;

		private System.Drawing.Image ImTitle = LoadImage("Title2.bmp");

		private System.Drawing.Image ImTitleBand = LoadImage("Title2bnd.bmp");

		static Image LoadImage(string resname)
		{			
			System.IO.Stream myStream;
			System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
			myStream = myAssembly.GetManifestResourceStream("SySal.Executables.OperaPublicationManager." + resname);
			Image im = new Bitmap(myStream);
			myStream.Close();
			return im;
		}

		private void OnTitleMouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			DragPoint = System.Windows.Forms.Control.MousePosition;			
		}

		private void OnTitleMouseMove(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				Point newp = System.Windows.Forms.Control.MousePosition;
				Point currl = this.Location;
				currl.X += (newp.X - DragPoint.X);
				currl.Y += (newp.Y - DragPoint.Y);
				this.Location = currl;
				DragPoint = newp;
			}
		}

		private void OnTitlePaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			System.Drawing.Graphics g = e.Graphics;
			g.DrawImage(ImTitle, 0, 0, ImTitle.Width, ImTitle.Height);
			int x;
			for (x = ImTitle.Width; x < this.Width; x += ImTitleBand.Width)
				g.DrawImage(ImTitleBand, x, 0, ImTitleBand.Width, ImTitleBand.Height);
		}

		private void OnGeneralGroupDoubleClick(object sender, System.EventArgs e)
		{
			if (groupBox1.IsOpen)			
			{
				groupBox2.IsOpen = false;
				groupBox3.IsOpen = false;
			}
			groupBox2.Top = groupBox1.Top + groupBox1.Height;
			groupBox3.Top = groupBox2.Top + groupBox2.Height;
		}

		private void OnJobsDoubleClick(object sender, System.EventArgs e)
		{
			if (groupBox2.IsOpen) groupBox2.IsOpen = false;
			else
			{
				groupBox1.IsOpen = false;
				groupBox3.IsOpen = false;
			}
		}

		private void OnObjsDoubleClick(object sender, System.EventArgs e)
		{
			if (groupBox3.IsOpen) groupBox3.IsOpen = false;
			else
			{
				groupBox1.IsOpen = false;
				groupBox2.IsOpen = false;
			}
			groupBox2.Top = groupBox1.Top + groupBox1.Height;
			groupBox3.Top = groupBox2.Top + groupBox2.Height;
		}

		private void OnOpenCloseGeneral(SySal.Controls.GroupBox sender, bool isopen)
		{
			if (isopen)
			{
				groupBox2.IsOpen = false;
				groupBox3.IsOpen = false;
			}
			groupBox2.Top = groupBox1.Top + groupBox1.Height;
			groupBox3.Top = groupBox2.Top + groupBox2.Height;
		}

		private void OnOpenCloseJobs(SySal.Controls.GroupBox sender, bool isopen)
		{
			if (isopen)
			{
				groupBox1.IsOpen = false;
				groupBox3.IsOpen = false;
			}
			groupBox2.Top = groupBox1.Top + groupBox1.Height;
			groupBox3.Top = groupBox2.Top + groupBox2.Height;		
		}

		private void OnOpenCloseObjs(SySal.Controls.GroupBox sender, bool isopen)
		{
			if (isopen)
			{
				groupBox1.IsOpen = false;
				groupBox2.IsOpen = false;
			}
			groupBox2.Top = groupBox1.Top + groupBox1.Height;
			groupBox3.Top = groupBox2.Top + groupBox2.Height;		
		}

		private void btnExit_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		LogForm ViewLogForm = null;

		private void btnJobViewLog_Click(object sender, System.EventArgs e)
		{
			bool oneselected = false;
			long jobid = 0;
			string dblink = "";
			int i;
			int len = ((System.Data.DataTable)gridJobs.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridJobs.IsSelected(i))
				{
					if (oneselected) 
					{
						MessageBox.Show("Only one job must be selected.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
						return;
					}
					dblink = gridJobs[i, 1].ToString();
					jobid = System.Convert.ToInt64(gridJobs[i, 0].ToString());
					oneselected = true;
				}
			if (oneselected == false) return;
			if (ViewLogForm != null)
			{
				ViewLogForm.Close();
				ViewLogForm.Dispose();
				ViewLogForm = null;
			}
			ViewLogForm = new LogForm(jobid, dblink, Conn);
			ViewLogForm.Show();
		}

		StatsForm Stats = new StatsForm();

		private void btnJobViewStats_Click(object sender, System.EventArgs e)
		{
			bool oneselected = false;
			long jobid = 0;
			int i;
			int len = ((System.Data.DataTable)gridJobs.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridJobs.IsSelected(i))
				{
					if (oneselected) 
					{
						MessageBox.Show("Only one job must be selected.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
						return;
					}
					jobid = System.Convert.ToInt64(gridJobs[i, 0].ToString());
					oneselected = true;
				}
			if (oneselected == false) return;
			Stats.ShowDialog(jobid, Conn);		
		}

		ActivityForm Activity = new ActivityForm();

		private void btnJobViewActivity_Click(object sender, System.EventArgs e)
		{
			bool oneselected = false;
			long jobid = 0;
			int i;
			int len = ((System.Data.DataTable)gridJobs.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridJobs.IsSelected(i))
				{
					if (oneselected) 
					{
						MessageBox.Show("Only one job must be selected.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
						return;
					}
					jobid = System.Convert.ToInt64(gridJobs[i, 0].ToString());
					oneselected = true;
				}
			if (oneselected == false) return;
			Activity.ShowDialog(jobid, Conn);				
		}

		private void btnViewLog_Click(object sender, System.EventArgs e)
		{
			if (ViewLogForm != null)
			{
				ViewLogForm.Close();
				ViewLogForm.Dispose();
				ViewLogForm = null;
			}
			ViewLogForm = new LogForm(0, "", Conn);
			ViewLogForm.Show();		
		}

		PurgeLogForm PurgeLog = new PurgeLogForm();

		private void btnPurgeLog_Click(object sender, System.EventArgs e)
		{
			PurgeLog.ShowDialog(Conn);
		}		

		VersionCheckForm VersionCheck = new VersionCheckForm();

		private void btnObjsVersionCheck_Click(object sender, System.EventArgs e)
		{
			if (gridDBLinks.CurrentRowIndex < 0) return;
			Cursor oldc = Cursor;
			Cursor = Cursors.WaitCursor;
			System.Data.DataTable dt = new System.Data.DataTable();
			dt.Columns.Add("TYPE");
			dt.Columns.Add("OBJID");
			dt.Columns.Add("VERSION CHECK RESULT");
			SySal.OperaDb.OperaDbCommand checkCmd = new SySal.OperaDb.OperaDbCommand("CALL PP_COMPARE_OBJECTS@" + gridDBLinks[gridDBLinks.CurrentRowIndex, 0].ToString() + "(:jtype, :jobj, :jtoken, :jstatus)", Conn, null);
			checkCmd.Parameters.Add("jtype", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			checkCmd.Parameters.Add("jobj", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			checkCmd.Parameters.Add("jtoken", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			checkCmd.Parameters.Add("jstatus", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			int i;
			int len = ((System.Data.DataTable)gridObjs.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridObjs.IsSelected(i))
				{
					string status = "OK";
					try
					{
						checkCmd.Parameters[0].Value = gridObjs[i, 0].ToString();
						checkCmd.Parameters[1].Value = gridObjs[i, 1].ToString();
						checkCmd.Parameters[2].Value = gridObjs[i, 2].ToString();
						checkCmd.Parameters[3].Value = gridObjs[i, 3].ToString();
						checkCmd.ExecuteNonQuery();
					}
					catch (Exception x)
					{
						status = "Failed: " + x.Message;
						status = status.Replace(@"\n", " ");
						status = status.Replace(@"\r", " ");
						status = status.Replace(@"\t", " ");
					}
					dt.Rows.Add(new object [3] { gridObjs[i, 0], gridObjs[i, 1], status } );
				}						
			Cursor = oldc;
			VersionCheck.ShowDialog(dt);
		}

		private void btnViewStats_Click(object sender, System.EventArgs e)
		{
			Stats.ShowDialog(0, Conn);		
		}

		private void btnObjDeregister_Click(object sender, System.EventArgs e)
		{
			Cursor oldc = Cursor;
			Cursor = Cursors.WaitCursor;
			SySal.OperaDb.OperaDbCommand checkCmd = new SySal.OperaDb.OperaDbCommand("CALL PP_DEREGISTER_OBJECT(:jtype, :jobj)", Conn, null);
			checkCmd.Parameters.Add("jtype", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			checkCmd.Parameters.Add("jobj", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			int i;
			int len = ((System.Data.DataTable)gridObjs.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridObjs.IsSelected(i))				
					try
					{
						checkCmd.Parameters[0].Value = gridObjs[i, 0].ToString();
						checkCmd.Parameters[1].Value = gridObjs[i, 1].ToString();
						checkCmd.ExecuteNonQuery();
					}
					catch (Exception x)
					{
						MessageBox.Show(x.Message, "Error deregistering object", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}			
			gridObjs_DoubleClick(null, null);
			Cursor = oldc;
		}

		private void btnObjInvalidate_Click(object sender, System.EventArgs e)
		{
			Cursor oldc = Cursor;
			Cursor = Cursors.WaitCursor;
			SySal.OperaDb.OperaDbCommand checkCmd = new SySal.OperaDb.OperaDbCommand("CALL PP_INVALIDATE_OBJECT(:jtype, :jobj)", Conn, null);
			checkCmd.Parameters.Add("jtype", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			checkCmd.Parameters.Add("jobj", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
			int i;
			int len = ((System.Data.DataTable)gridObjs.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridObjs.IsSelected(i))				
					try
					{
						checkCmd.Parameters[0].Value = gridObjs[i, 0].ToString();
						checkCmd.Parameters[1].Value = gridObjs[i, 1].ToString();
						checkCmd.ExecuteNonQuery();
					}
					catch (Exception x)
					{
						MessageBox.Show(x.Message, "Error deregistering object", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}			
			gridObjs_DoubleClick(null, null);
			Cursor = oldc;		
		}
	}
}
