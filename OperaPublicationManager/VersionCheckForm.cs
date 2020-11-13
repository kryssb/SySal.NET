using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// Shows the results of a version check.
	/// </summary>
	public class VersionCheckForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.DataGrid gridVCheck;
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		/// <summary>
		/// Creates a new VersionCheckForm.
		/// </summary>
		public VersionCheckForm()
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
			this.gridVCheck = new System.Windows.Forms.DataGrid();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			((System.ComponentModel.ISupportInitialize)(this.gridVCheck)).BeginInit();
			this.SuspendLayout();
			// 
			// gridVCheck
			// 
			this.gridVCheck.AlternatingBackColor = System.Drawing.Color.White;
			this.gridVCheck.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridVCheck.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridVCheck.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridVCheck.CaptionForeColor = System.Drawing.Color.White;
			this.gridVCheck.DataMember = "";
			this.gridVCheck.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridVCheck.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridVCheck.Location = new System.Drawing.Point(16, 16);
			this.gridVCheck.Name = "gridVCheck";
			this.gridVCheck.PreferredColumnWidth = 120;
			this.gridVCheck.ReadOnly = true;
			this.gridVCheck.Size = new System.Drawing.Size(520, 296);
			this.gridVCheck.TabIndex = 11;
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(552, 328);
			this.backgroundPanel1.TabIndex = 12;
			// 
			// VersionCheckForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(552, 324);
			this.Controls.Add(this.gridVCheck);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "VersionCheckForm";
			this.Text = "Version Check Results";
			this.Resize += new System.EventHandler(this.OnResize);
			((System.ComponentModel.ISupportInitialize)(this.gridVCheck)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void OnResize(object sender, System.EventArgs e)
		{
			backgroundPanel1.Width = this.Width - 8;
			backgroundPanel1.Height = this.Height - 24;
			gridVCheck.Width = this.Width - 40;
			gridVCheck.Height = this.Height - 56;
		}

		/// <summary>
		/// Shows the dialog, using a specified DataTable as the data source.
		/// </summary>
		/// <param name="dt">the DataTable to be used as a data source.</param>
		public void ShowDialog(System.Data.DataTable dt)
		{
			gridVCheck.DataSource = dt;
			ShowDialog();
		}
	}
}
