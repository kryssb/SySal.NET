using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// The QueueForm allows a user enter information to set up or reset a job queue (see <see cref="SySal.Executables.OperaPublicationManager.MainForm"/> for job queues).
	/// </summary>
	/// <remarks>
	/// <para>The <i>DB Link</i> parameter specifies the DB link the queue refers to. The <i>First time</i> is the first time when the queue will be executed. <i>Repeat after days</i> specifies the time interval, in days, before the next execution.</para>
	/// <para>When the QueueForm is opened, <i>First time</i> is initialized to 30s after the currnt time; therefore, in order to schedule a queue to run almost immediately, one just needs to open the QueueForm on that queue and click <i>OK</i>.</para>
	/// </remarks>
	public class QueueForm : System.Windows.Forms.Form
	{
		internal System.Windows.Forms.TextBox textDBLink;
		internal System.Windows.Forms.DateTimePicker timeFirstSchedule;
		internal System.Windows.Forms.ComboBox comboInterval;
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private SySal.Controls.StaticText staticText4;
		private SySal.Controls.StaticText staticText1;
		private SySal.Controls.StaticText staticText2;
		private SySal.Controls.Button buttonCancel;
		private SySal.Controls.Button buttonOK;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		/// <summary>
		/// Creates a new QueueForm.
		/// </summary>
		public QueueForm()
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
			this.textDBLink = new System.Windows.Forms.TextBox();
			this.timeFirstSchedule = new System.Windows.Forms.DateTimePicker();
			this.comboInterval = new System.Windows.Forms.ComboBox();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.staticText4 = new SySal.Controls.StaticText();
			this.staticText1 = new SySal.Controls.StaticText();
			this.staticText2 = new SySal.Controls.StaticText();
			this.buttonCancel = new SySal.Controls.Button();
			this.buttonOK = new SySal.Controls.Button();
			this.SuspendLayout();
			// 
			// textDBLink
			// 
			this.textDBLink.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textDBLink.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textDBLink.ForeColor = System.Drawing.Color.Navy;
			this.textDBLink.Location = new System.Drawing.Point(144, 16);
			this.textDBLink.Name = "textDBLink";
			this.textDBLink.ReadOnly = true;
			this.textDBLink.Size = new System.Drawing.Size(200, 20);
			this.textDBLink.TabIndex = 2;
			this.textDBLink.Text = "";
			// 
			// timeFirstSchedule
			// 
			this.timeFirstSchedule.CalendarFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.timeFirstSchedule.CalendarForeColor = System.Drawing.Color.Navy;
			this.timeFirstSchedule.CustomFormat = "dd-MM-yyyy HH:mm:ss";
			this.timeFirstSchedule.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.timeFirstSchedule.Format = System.Windows.Forms.DateTimePickerFormat.Custom;
			this.timeFirstSchedule.Location = new System.Drawing.Point(144, 40);
			this.timeFirstSchedule.MaxDate = new System.DateTime(2050, 12, 31, 0, 0, 0, 0);
			this.timeFirstSchedule.MinDate = new System.DateTime(2006, 1, 1, 0, 0, 0, 0);
			this.timeFirstSchedule.Name = "timeFirstSchedule";
			this.timeFirstSchedule.TabIndex = 4;
			// 
			// comboInterval
			// 
			this.comboInterval.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.comboInterval.ForeColor = System.Drawing.Color.Navy;
			this.comboInterval.ItemHeight = 14;
			this.comboInterval.Items.AddRange(new object[] {
															   "1",
															   "2",
															   "3",
															   "4",
															   "5",
															   "6",
															   "7",
															   "10",
															   "15",
															   "20",
															   "30"});
			this.comboInterval.Location = new System.Drawing.Point(144, 64);
			this.comboInterval.Name = "comboInterval";
			this.comboInterval.Size = new System.Drawing.Size(200, 22);
			this.comboInterval.TabIndex = 6;
			this.comboInterval.Text = "1";
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(360, 136);
			this.backgroundPanel1.TabIndex = 0;
			// 
			// staticText4
			// 
			this.staticText4.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText4.LabelText = "DB Link";
			this.staticText4.Location = new System.Drawing.Point(16, 16);
			this.staticText4.Name = "staticText4";
			this.staticText4.Size = new System.Drawing.Size(72, 24);
			this.staticText4.TabIndex = 1;
			this.staticText4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText1
			// 
			this.staticText1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText1.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.staticText1.ForeColor = System.Drawing.Color.Navy;
			this.staticText1.LabelText = "First time";
			this.staticText1.Location = new System.Drawing.Point(16, 40);
			this.staticText1.Name = "staticText1";
			this.staticText1.Size = new System.Drawing.Size(72, 24);
			this.staticText1.TabIndex = 3;
			this.staticText1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText2
			// 
			this.staticText2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText2.LabelText = "Repeat after days:";
			this.staticText2.Location = new System.Drawing.Point(16, 64);
			this.staticText2.Name = "staticText2";
			this.staticText2.Size = new System.Drawing.Size(120, 24);
			this.staticText2.TabIndex = 5;
			this.staticText2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// buttonCancel
			// 
			this.buttonCancel.BackColor = System.Drawing.Color.White;
			this.buttonCancel.ButtonText = "Cancel";
			this.buttonCancel.Location = new System.Drawing.Point(16, 96);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(56, 24);
			this.buttonCancel.TabIndex = 7;
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			this.buttonCancel.DoubleClick += new System.EventHandler(this.buttonCancel_Click);
			// 
			// buttonOK
			// 
			this.buttonOK.BackColor = System.Drawing.Color.White;
			this.buttonOK.ButtonText = "OK";
			this.buttonOK.Location = new System.Drawing.Point(288, 96);
			this.buttonOK.Name = "buttonOK";
			this.buttonOK.Size = new System.Drawing.Size(56, 24);
			this.buttonOK.TabIndex = 8;
			this.buttonOK.Click += new System.EventHandler(this.buttonOK_Click);
			this.buttonOK.DoubleClick += new System.EventHandler(this.buttonOK_Click);
			// 
			// QueueForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(362, 136);
			this.Controls.Add(this.buttonOK);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.staticText2);
			this.Controls.Add(this.staticText1);
			this.Controls.Add(this.staticText4);
			this.Controls.Add(this.comboInterval);
			this.Controls.Add(this.timeFirstSchedule);
			this.Controls.Add(this.textDBLink);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "QueueForm";
			this.ShowInTaskbar = false;
			this.Text = "Queue Information";
			this.ResumeLayout(false);

		}
		#endregion

		private void buttonOK_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.OK;
			Close();
		}

		private void buttonCancel_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();		
		}
	}
}
