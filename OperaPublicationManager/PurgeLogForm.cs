using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// The PurgeLogForm is used to clean the OPERAPUB log of useless log entries. 
	/// </summary>
	/// <remarks>
	/// <para>The number of current entries in the log is shown. The log can be purged up to a specified time (<i>Earliest survival time</i>); 
	/// before purging the log, the number of survivor entries can be estimated by clicking on the <i>Estimate survivors after purge</i> button.</para>
	/// </remarks>
	public class PurgeLogForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox textLogEntries;
		internal System.Windows.Forms.DateTimePicker timePurge;
		private System.Windows.Forms.TextBox textSurvivors;
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private SySal.Controls.StaticText staticText1;
		private SySal.Controls.StaticText staticText2;
		private SySal.Controls.Button buttonCancel;
		private SySal.Controls.Button buttonEstimateSurvivors;
		private SySal.Controls.Button buttonProceed;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		/// <summary>
		/// Creates a new PurgeLogForm.
		/// </summary>
		public PurgeLogForm()
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
			this.textLogEntries = new System.Windows.Forms.TextBox();
			this.timePurge = new System.Windows.Forms.DateTimePicker();
			this.textSurvivors = new System.Windows.Forms.TextBox();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.buttonProceed = new SySal.Controls.Button();
			this.staticText1 = new SySal.Controls.StaticText();
			this.staticText2 = new SySal.Controls.StaticText();
			this.buttonCancel = new SySal.Controls.Button();
			this.buttonEstimateSurvivors = new SySal.Controls.Button();
			this.SuspendLayout();
			// 
			// textLogEntries
			// 
			this.textLogEntries.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textLogEntries.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textLogEntries.ForeColor = System.Drawing.Color.Navy;
			this.textLogEntries.Location = new System.Drawing.Point(264, 16);
			this.textLogEntries.Name = "textLogEntries";
			this.textLogEntries.ReadOnly = true;
			this.textLogEntries.Size = new System.Drawing.Size(64, 20);
			this.textLogEntries.TabIndex = 1;
			this.textLogEntries.Text = "";
			// 
			// timePurge
			// 
			this.timePurge.CalendarForeColor = System.Drawing.Color.Navy;
			this.timePurge.CustomFormat = "dd-MM-yyyy hh:mm:ss";
			this.timePurge.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.timePurge.Format = System.Windows.Forms.DateTimePickerFormat.Custom;
			this.timePurge.Location = new System.Drawing.Point(264, 40);
			this.timePurge.Name = "timePurge";
			this.timePurge.Size = new System.Drawing.Size(160, 20);
			this.timePurge.TabIndex = 3;
			// 
			// textSurvivors
			// 
			this.textSurvivors.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textSurvivors.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textSurvivors.ForeColor = System.Drawing.Color.Navy;
			this.textSurvivors.Location = new System.Drawing.Point(264, 72);
			this.textSurvivors.Name = "textSurvivors";
			this.textSurvivors.ReadOnly = true;
			this.textSurvivors.Size = new System.Drawing.Size(64, 20);
			this.textSurvivors.TabIndex = 5;
			this.textSurvivors.Text = "";
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(440, 152);
			this.backgroundPanel1.TabIndex = 9;
			// 
			// buttonProceed
			// 
			this.buttonProceed.BackColor = System.Drawing.Color.White;
			this.buttonProceed.ButtonText = "Proceed";
			this.buttonProceed.Location = new System.Drawing.Point(360, 112);
			this.buttonProceed.Name = "buttonProceed";
			this.buttonProceed.Size = new System.Drawing.Size(64, 24);
			this.buttonProceed.TabIndex = 11;
			this.buttonProceed.Click += new System.EventHandler(this.buttonProceed_Click);
			this.buttonProceed.DoubleClick += new System.EventHandler(this.buttonProceed_Click);
			// 
			// staticText1
			// 
			this.staticText1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText1.LabelText = "Current entries in log";
			this.staticText1.Location = new System.Drawing.Point(16, 16);
			this.staticText1.Name = "staticText1";
			this.staticText1.Size = new System.Drawing.Size(143, 24);
			this.staticText1.TabIndex = 10;
			this.staticText1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText2
			// 
			this.staticText2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText2.LabelText = "Earliest survival timestamp after purge";
			this.staticText2.Location = new System.Drawing.Point(16, 40);
			this.staticText2.Name = "staticText2";
			this.staticText2.Size = new System.Drawing.Size(242, 24);
			this.staticText2.TabIndex = 12;
			this.staticText2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// buttonCancel
			// 
			this.buttonCancel.BackColor = System.Drawing.Color.White;
			this.buttonCancel.ButtonText = "Cancel";
			this.buttonCancel.Location = new System.Drawing.Point(16, 112);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(64, 24);
			this.buttonCancel.TabIndex = 13;
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			this.buttonCancel.DoubleClick += new System.EventHandler(this.buttonCancel_Click);
			// 
			// buttonEstimateSurvivors
			// 
			this.buttonEstimateSurvivors.BackColor = System.Drawing.Color.White;
			this.buttonEstimateSurvivors.ButtonText = "Estimate survivors after purge";
			this.buttonEstimateSurvivors.Location = new System.Drawing.Point(16, 72);
			this.buttonEstimateSurvivors.Name = "buttonEstimateSurvivors";
			this.buttonEstimateSurvivors.Size = new System.Drawing.Size(200, 24);
			this.buttonEstimateSurvivors.TabIndex = 14;
			this.buttonEstimateSurvivors.Click += new System.EventHandler(this.buttonEstimateSurvivors_Click);
			this.buttonEstimateSurvivors.DoubleClick += new System.EventHandler(this.buttonEstimateSurvivors_Click);
			// 
			// PurgeLogForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(442, 152);
			this.Controls.Add(this.buttonEstimateSurvivors);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.staticText2);
			this.Controls.Add(this.buttonProceed);
			this.Controls.Add(this.staticText1);
			this.Controls.Add(this.textSurvivors);
			this.Controls.Add(this.timePurge);
			this.Controls.Add(this.textLogEntries);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "PurgeLogForm";
			this.Text = "Purge Log";
			this.ResumeLayout(false);

		}
		#endregion
		
		SySal.OperaDb.OperaDbConnection Conn = null;

		/// <summary>
		/// Shows the dialog.
		/// </summary>
		/// <param name="conn">the DB connection to be used.</param>
		/// <returns><c>DialogResult.OK</c> if the purge log operation has been accepted; other codes otherwise.</returns>
		public DialogResult ShowDialog(SySal.OperaDb.OperaDbConnection conn)
		{
			Conn = conn;
			DialogResult res = DialogResult.Cancel;
			try
			{
				textLogEntries.Text = new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM PT_LOG", Conn, null).ExecuteScalar().ToString();
				textSurvivors.Text = "";
				timePurge.Value = System.DateTime.Now.AddDays(-7);
				res = ShowDialog();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error executing \"Purge Log\" form", MessageBoxButtons.OK, MessageBoxIcon.Error);
				res = DialogResult.Cancel;
			}
			Conn = null;			
			return res;
		}

		private void buttonEstimateSurvivors_Click(object sender, System.EventArgs e)
		{
			textSurvivors.Text = new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM PT_LOG WHERE TIMESTAMP >= TO_TIMESTAMP('" + timePurge.Value.ToString("dd-MM-yyyy hh:mm:ss") + "', 'dd-mm-yyyy hh24:mi:ss')", Conn, null).ExecuteScalar().ToString();
		}

		private void buttonProceed_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();
		}

		private void buttonCancel_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.OK;
			Close();		
		}
	}
}
