using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// Shows log information for a specified job.
	/// </summary>
	public class LogForm : System.Windows.Forms.Form
	{
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private System.Windows.Forms.DataGrid gridLog;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private long JobId;

		private string DBLink;

		private SySal.OperaDb.OperaDbConnection DBConn;

		/// <summary>
		/// Creates a new LogForm.
		/// </summary>
		/// <param name="jobid">the ID of the job whose log information must be displayed.</param>
		/// <param name="dblink">the DB link involved.</param>
		/// <param name="dbconn">the DB connection to be used.</param>
		public LogForm(long jobid, string dblink, SySal.OperaDb.OperaDbConnection dbconn)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			JobId = jobid;
			DBLink = dblink;
			if (JobId > 0)
				this.Text = "Log View for Job #" + JobId;
			DBConn = dbconn;
			OnRefreshLog(this, null);
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
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
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.gridLog = new System.Windows.Forms.DataGrid();
			((System.ComponentModel.ISupportInitialize)(this.gridLog)).BeginInit();
			this.SuspendLayout();
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(552, 328);
			this.backgroundPanel1.TabIndex = 10;
			// 
			// gridLog
			// 
			this.gridLog.AlternatingBackColor = System.Drawing.Color.White;
			this.gridLog.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridLog.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridLog.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridLog.CaptionForeColor = System.Drawing.Color.White;
			this.gridLog.CaptionText = "Log (double-click to refresh)";
			this.gridLog.DataMember = "";
			this.gridLog.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridLog.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridLog.Location = new System.Drawing.Point(16, 16);
			this.gridLog.Name = "gridLog";
			this.gridLog.PreferredColumnWidth = 120;
			this.gridLog.ReadOnly = true;
			this.gridLog.Size = new System.Drawing.Size(520, 296);
			this.gridLog.TabIndex = 11;
			this.gridLog.DoubleClick += new System.EventHandler(this.OnRefreshLog);
			// 
			// LogForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(552, 324);
			this.Controls.Add(this.gridLog);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "LogForm";
			this.Text = "Log View";
			this.Resize += new System.EventHandler(this.OnResize);
			((System.ComponentModel.ISupportInitialize)(this.gridLog)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void OnRefreshLog(object sender, System.EventArgs e)
		{
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, DBLINK, TIMESTAMP, OBJID, TEXT FROM PT_LOG" + ((JobId <= 0) ? " ORDER BY ID DESC" : (" WHERE OBJID = " + JobId + " AND DBLINK = '" + DBLink + "' ORDER BY ID DESC")), DBConn, null).Fill(ds);
			gridLog.DataSource = ds.Tables[0];
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			backgroundPanel1.Width = this.Width - 8;
			backgroundPanel1.Height = this.Height - 24;
			gridLog.Width = this.Width - 40;
			gridLog.Height = this.Height - 56;
		}
	}
}
