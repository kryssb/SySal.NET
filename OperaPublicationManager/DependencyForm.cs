using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// Shows the process operation from which a process operation depends directly or indirectly (e.g. through its children).
	/// </summary>
	public class DependencyForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button buttonClose;
		private System.Windows.Forms.Button buttonExport;
		private System.Windows.Forms.DataGrid gridDependencies;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public DependencyForm()
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
			this.gridDependencies = new System.Windows.Forms.DataGrid();
			this.buttonClose = new System.Windows.Forms.Button();
			this.buttonExport = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.gridDependencies)).BeginInit();
			this.SuspendLayout();
			// 
			// gridDependencies
			// 
			this.gridDependencies.AlternatingBackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.gridDependencies.BackColor = System.Drawing.Color.White;
			this.gridDependencies.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridDependencies.CaptionForeColor = System.Drawing.Color.White;
			this.gridDependencies.CaptionText = "Dependency walk results";
			this.gridDependencies.DataMember = "";
			this.gridDependencies.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.gridDependencies.Location = new System.Drawing.Point(8, 8);
			this.gridDependencies.Name = "gridDependencies";
			this.gridDependencies.PreferredColumnWidth = 150;
			this.gridDependencies.ReadOnly = true;
			this.gridDependencies.Size = new System.Drawing.Size(400, 344);
			this.gridDependencies.TabIndex = 0;
			// 
			// buttonClose
			// 
			this.buttonClose.Location = new System.Drawing.Point(320, 360);
			this.buttonClose.Name = "buttonClose";
			this.buttonClose.Size = new System.Drawing.Size(88, 24);
			this.buttonClose.TabIndex = 1;
			this.buttonClose.Text = "Close";
			this.buttonClose.Click += new System.EventHandler(this.buttonClose_Click);
			// 
			// buttonExport
			// 
			this.buttonExport.Location = new System.Drawing.Point(8, 360);
			this.buttonExport.Name = "buttonExport";
			this.buttonExport.Size = new System.Drawing.Size(88, 24);
			this.buttonExport.TabIndex = 2;
			this.buttonExport.Text = "Export to file";
			this.buttonExport.Click += new System.EventHandler(this.buttonExport_Click);
			// 
			// DependencyForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(418, 392);
			this.Controls.Add(this.buttonExport);
			this.Controls.Add(this.buttonClose);
			this.Controls.Add(this.gridDependencies);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "DependencyForm";
			this.ShowInTaskbar = false;
			this.Text = "Dependency Report";
			((System.ComponentModel.ISupportInitialize)(this.gridDependencies)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void buttonExport_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) MainForm.ExportToFile((System.Data.DataTable)gridDependencies.DataSource, sdlg.FileName);
		}

		private void buttonClose_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		public DialogResult ShowDialog(System.Data.DataTable dt)
		{
			gridDependencies.DataSource = dt;
			return ShowDialog();
		}
	}
}
