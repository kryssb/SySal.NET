using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// The ActivityForm for a job shows its current activity.
	/// </summary>
	/// <remarks>
	/// <para><i>Job Type</i> shows the type of job (Brick/Operation/System Publishing/Copying/Deleting/etc.)</para>
	/// <para><i>Object ID</i> is the ID of the object being handled by the job.</para>
	/// <para><i>Status</i> shows the process status (<c>TODO</c>/<c>SCHEDULED</c>/<c>RUNNING</c>/<c>ABORTED</c>/<c>DONE</c>)</para>
	/// <para><i>Progress</i> displays detailed information about the tables being accessed and the percentage of work done.</para>
	/// <para>The progress bar expresses the fraction of work done in a graphical way.</para>
	/// </remarks>
	public class ActivityForm : System.Windows.Forms.Form
	{
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private System.ComponentModel.IContainer components;

		private long JobId;

		private SySal.OperaDb.OperaDbConnection Conn;

		private SySal.OperaDb.OperaDbDataAdapter daGetInfo;

		private string DBLink;
		private System.Windows.Forms.TextBox textJobType;
		private SySal.Controls.StaticText staticText1;
		private SySal.Controls.StaticText staticText2;
		private System.Windows.Forms.TextBox textObjId;
		private SySal.Controls.StaticText staticText3;
		private System.Windows.Forms.TextBox textStatus;
		private SySal.Controls.StaticText staticText4;
		private System.Windows.Forms.TextBox textProgress;
		private SySal.Controls.StaticText lblLev1;
		private SySal.Controls.ProgressBar progressBar1;
		private SySal.Controls.ProgressBar progressBar2;
		private SySal.Controls.StaticText lblLev2;
		private SySal.Controls.ProgressBar progressBar3;
		private SySal.Controls.StaticText lblLev3;
		private System.Windows.Forms.Timer RefTimer;

		private SySal.OperaDb.OperaDbConnection DBConn;

		/// <summary>
		/// Builds the ActivityForm.
		/// </summary>
		public ActivityForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//			
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
			this.components = new System.ComponentModel.Container();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.textJobType = new System.Windows.Forms.TextBox();
			this.staticText1 = new SySal.Controls.StaticText();
			this.staticText2 = new SySal.Controls.StaticText();
			this.textObjId = new System.Windows.Forms.TextBox();
			this.staticText3 = new SySal.Controls.StaticText();
			this.textStatus = new System.Windows.Forms.TextBox();
			this.staticText4 = new SySal.Controls.StaticText();
			this.textProgress = new System.Windows.Forms.TextBox();
			this.lblLev1 = new SySal.Controls.StaticText();
			this.progressBar1 = new SySal.Controls.ProgressBar();
			this.progressBar2 = new SySal.Controls.ProgressBar();
			this.lblLev2 = new SySal.Controls.StaticText();
			this.progressBar3 = new SySal.Controls.ProgressBar();
			this.lblLev3 = new SySal.Controls.StaticText();
			this.RefTimer = new System.Windows.Forms.Timer(this.components);
			this.SuspendLayout();
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(480, 208);
			this.backgroundPanel1.TabIndex = 10;
			this.backgroundPanel1.Load += new System.EventHandler(this.OnLoad);
			// 
			// textJobType
			// 
			this.textJobType.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textJobType.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textJobType.ForeColor = System.Drawing.Color.Navy;
			this.textJobType.Location = new System.Drawing.Point(144, 24);
			this.textJobType.Name = "textJobType";
			this.textJobType.ReadOnly = true;
			this.textJobType.Size = new System.Drawing.Size(152, 20);
			this.textJobType.TabIndex = 18;
			this.textJobType.Text = "";
			// 
			// staticText1
			// 
			this.staticText1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText1.LabelText = "Job type";
			this.staticText1.Location = new System.Drawing.Point(24, 24);
			this.staticText1.Name = "staticText1";
			this.staticText1.Size = new System.Drawing.Size(80, 24);
			this.staticText1.TabIndex = 19;
			this.staticText1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText2
			// 
			this.staticText2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText2.LabelText = "Object ID";
			this.staticText2.Location = new System.Drawing.Point(24, 48);
			this.staticText2.Name = "staticText2";
			this.staticText2.Size = new System.Drawing.Size(80, 24);
			this.staticText2.TabIndex = 21;
			this.staticText2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textObjId
			// 
			this.textObjId.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textObjId.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textObjId.ForeColor = System.Drawing.Color.Navy;
			this.textObjId.Location = new System.Drawing.Point(144, 48);
			this.textObjId.Name = "textObjId";
			this.textObjId.ReadOnly = true;
			this.textObjId.Size = new System.Drawing.Size(152, 20);
			this.textObjId.TabIndex = 20;
			this.textObjId.Text = "";
			// 
			// staticText3
			// 
			this.staticText3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText3.LabelText = "Status";
			this.staticText3.Location = new System.Drawing.Point(24, 72);
			this.staticText3.Name = "staticText3";
			this.staticText3.Size = new System.Drawing.Size(80, 24);
			this.staticText3.TabIndex = 23;
			this.staticText3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textStatus
			// 
			this.textStatus.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textStatus.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textStatus.ForeColor = System.Drawing.Color.Navy;
			this.textStatus.Location = new System.Drawing.Point(144, 72);
			this.textStatus.Name = "textStatus";
			this.textStatus.ReadOnly = true;
			this.textStatus.Size = new System.Drawing.Size(152, 20);
			this.textStatus.TabIndex = 22;
			this.textStatus.Text = "";
			// 
			// staticText4
			// 
			this.staticText4.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText4.LabelText = "Progress";
			this.staticText4.Location = new System.Drawing.Point(24, 96);
			this.staticText4.Name = "staticText4";
			this.staticText4.Size = new System.Drawing.Size(80, 24);
			this.staticText4.TabIndex = 25;
			this.staticText4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textProgress
			// 
			this.textProgress.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textProgress.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textProgress.ForeColor = System.Drawing.Color.Navy;
			this.textProgress.Location = new System.Drawing.Point(144, 96);
			this.textProgress.Name = "textProgress";
			this.textProgress.ReadOnly = true;
			this.textProgress.Size = new System.Drawing.Size(296, 20);
			this.textProgress.TabIndex = 24;
			this.textProgress.Text = "";
			// 
			// lblLev1
			// 
			this.lblLev1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.lblLev1.LabelText = "Activity1";
			this.lblLev1.Location = new System.Drawing.Point(24, 128);
			this.lblLev1.Name = "lblLev1";
			this.lblLev1.Size = new System.Drawing.Size(104, 24);
			this.lblLev1.TabIndex = 26;
			this.lblLev1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// progressBar1
			// 
			this.progressBar1.BackColor = System.Drawing.Color.White;
			this.progressBar1.Location = new System.Drawing.Point(144, 128);
			this.progressBar1.Name = "progressBar1";
			this.progressBar1.Percent = 0;
			this.progressBar1.Size = new System.Drawing.Size(296, 16);
			this.progressBar1.TabIndex = 27;
			// 
			// progressBar2
			// 
			this.progressBar2.BackColor = System.Drawing.Color.White;
			this.progressBar2.Location = new System.Drawing.Point(144, 152);
			this.progressBar2.Name = "progressBar2";
			this.progressBar2.Percent = 0;
			this.progressBar2.Size = new System.Drawing.Size(296, 16);
			this.progressBar2.TabIndex = 29;
			// 
			// lblLev2
			// 
			this.lblLev2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.lblLev2.LabelText = "Activity2";
			this.lblLev2.Location = new System.Drawing.Point(24, 152);
			this.lblLev2.Name = "lblLev2";
			this.lblLev2.Size = new System.Drawing.Size(104, 24);
			this.lblLev2.TabIndex = 28;
			this.lblLev2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// progressBar3
			// 
			this.progressBar3.BackColor = System.Drawing.Color.White;
			this.progressBar3.Location = new System.Drawing.Point(144, 176);
			this.progressBar3.Name = "progressBar3";
			this.progressBar3.Percent = 0;
			this.progressBar3.Size = new System.Drawing.Size(296, 16);
			this.progressBar3.TabIndex = 31;
			// 
			// lblLev3
			// 
			this.lblLev3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.lblLev3.LabelText = "Activity3";
			this.lblLev3.Location = new System.Drawing.Point(24, 176);
			this.lblLev3.Name = "lblLev3";
			this.lblLev3.Size = new System.Drawing.Size(104, 24);
			this.lblLev3.TabIndex = 30;
			this.lblLev3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// RefTimer
			// 
			this.RefTimer.Interval = 500;
			this.RefTimer.Tick += new System.EventHandler(this.OnRefresh);
			// 
			// ActivityForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(480, 204);
			this.Controls.Add(this.progressBar3);
			this.Controls.Add(this.lblLev3);
			this.Controls.Add(this.progressBar2);
			this.Controls.Add(this.lblLev2);
			this.Controls.Add(this.progressBar1);
			this.Controls.Add(this.lblLev1);
			this.Controls.Add(this.staticText4);
			this.Controls.Add(this.textProgress);
			this.Controls.Add(this.staticText3);
			this.Controls.Add(this.textStatus);
			this.Controls.Add(this.staticText2);
			this.Controls.Add(this.textObjId);
			this.Controls.Add(this.staticText1);
			this.Controls.Add(this.textJobType);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "ActivityForm";
			this.Text = "Activity of Job #";
			this.Closing += new System.ComponentModel.CancelEventHandler(this.OnClosing);
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// Sets the job for which the dialog is to be displayed and shows it.
		/// </summary>
		/// <param name="jobid">the ID of the job for which information is to be displayed.</param>
		/// <param name="conn">the DB connection to be used.</param>
		internal void ShowDialog(long jobid, SySal.OperaDb.OperaDbConnection conn)
		{
			Conn = conn;
			JobId = jobid;
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT TYPE, OBJID FROM PT_JOBS WHERE ID = " + JobId, Conn, null).Fill(ds);
			textJobType.Text = ds.Tables[0].Rows[0][0].ToString();
			textObjId.Text = ds.Tables[0].Rows[0][1].ToString();
			lblLev1.Visible = false; progressBar1.Visible = false;
			lblLev2.Visible = false; progressBar2.Visible = false;
			lblLev3.Visible = false; progressBar3.Visible = false;
			daGetInfo = new SySal.OperaDb.OperaDbDataAdapter("SELECT STATUS, PROGRESS FROM PT_JOBS WHERE ID = " + JobId, Conn, null);
			Text = "Activity of Job #" + JobId;			
			ShowDialog();
		}

		private bool DoRefresh = false;

		private void OnRefresh(object sender, System.EventArgs e)
		{
			if (DoRefresh == false) return;
			System.Data.DataSet ds = new System.Data.DataSet();
			daGetInfo.Fill(ds);
			string progress;
			textStatus.Text = ds.Tables[0].Rows[0][0].ToString();
			textProgress.Text = progress = ds.Tables[0].Rows[0][1].ToString();
			progress = progress.Remove(0, progress.IndexOf("#") + 1);
			string [] proglevs = progress.Split('%');
			int i;
			for (i = 0; i < 3 && i < proglevs.Length; i++)
			{
				if (proglevs[i] == null || proglevs[i].Trim().Length == 0) continue;
				switch(i)
				{
					case 0: lblLev1.LabelText = proglevs[i] + "%"; 
						try
						{
							progressBar1.Percent = Convert.ToDouble(proglevs[i].Substring(proglevs[i].LastIndexOf(' ') + 1)) / 100.0; 
						}
						catch (Exception)
						{
							progressBar1.Percent = 1.0;	
						}
						lblLev1.Visible = true; 
						progressBar1.Visible = true; 
						break;

					case 1: lblLev2.LabelText = proglevs[i] + "%"; 
						try
						{
							progressBar2.Percent = Convert.ToDouble(proglevs[i].Substring(proglevs[i].LastIndexOf(' ') + 1)) / 100.0; 
						}
						catch (Exception)
						{
							progressBar2.Percent = 1.0;
						}
						lblLev2.Visible = true; 
						progressBar2.Visible = true; 
						break;

					case 2: lblLev3.LabelText = proglevs[i] + "%"; 
						try
						{
							progressBar3.Percent = Convert.ToDouble(proglevs[i].Substring(proglevs[i].LastIndexOf(' ') + 1)) / 100.0; 
						}
						catch (Exception)
						{
							progressBar3.Percent = 1.0;
						}
						lblLev3.Visible = true; 
						progressBar3.Visible = true; 
						break;
				}
			}
			for (; i < 3; i++)
			{
				switch(i)
				{
					case 0: lblLev1.Visible = false; progressBar1.Visible = false; break;
					case 1: lblLev2.Visible = false; progressBar2.Visible = false; break;
					case 2: lblLev3.Visible = false; progressBar3.Visible = false; break;
				}
			}		
		}

		private void OnLoad(object sender, System.EventArgs e)
		{
			RefTimer.Enabled = true;
			DoRefresh = true;
		}

		private void OnClosing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			DoRefresh = false;
			RefTimer.Enabled = false;
		}
	}
}
