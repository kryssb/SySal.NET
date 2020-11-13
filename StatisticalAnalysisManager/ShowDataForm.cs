using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace NumericalTools
{
	/// <summary>
	/// This form shows explicitly the data in a Dataset.
	/// </summary>
	/// <remarks>
	/// <para>Normally, the <i>Locked</i> flag (lower-left corner) is enabled, thus protecting
	/// your data for unwanted changes. If you clear the flag, you can edit the values, and this
	/// change is immediately reflected to the <see cref="NumericalTools.AnalysisControl">AnalysisForm</see>.</para>
	/// </remarks>
	public class ShowDataForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.DataGrid DataGrid;
		private System.Windows.Forms.CheckBox LockedCheck;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		internal ShowDataForm(string name, System.Data.DataSet ds)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			DataGrid.SetDataBinding(ds, "Data");			
			Text = "Data for dataset " + name;
			LockedCheck.Checked = true;
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
			this.DataGrid = new System.Windows.Forms.DataGrid();
			this.LockedCheck = new System.Windows.Forms.CheckBox();
			((System.ComponentModel.ISupportInitialize)(this.DataGrid)).BeginInit();
			this.SuspendLayout();
			// 
			// DataGrid
			// 
			this.DataGrid.AlternatingBackColor = System.Drawing.SystemColors.InactiveCaptionText;
			this.DataGrid.CaptionVisible = false;
			this.DataGrid.DataMember = "";
			this.DataGrid.HeaderForeColor = System.Drawing.SystemColors.ControlText;
			this.DataGrid.Location = new System.Drawing.Point(8, 8);
			this.DataGrid.Name = "DataGrid";
			this.DataGrid.ReadOnly = true;
			this.DataGrid.Size = new System.Drawing.Size(512, 248);
			this.DataGrid.TabIndex = 0;
			// 
			// LockedCheck
			// 
			this.LockedCheck.Location = new System.Drawing.Point(8, 264);
			this.LockedCheck.Name = "LockedCheck";
			this.LockedCheck.Size = new System.Drawing.Size(208, 24);
			this.LockedCheck.TabIndex = 1;
			this.LockedCheck.Text = "Locked";
			this.LockedCheck.CheckedChanged += new System.EventHandler(this.OnCheckedChanged);
			// 
			// ShowDataForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(530, 288);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.LockedCheck,
																		  this.DataGrid});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "ShowDataForm";
			this.Text = "Data for dataset";
			((System.ComponentModel.ISupportInitialize)(this.DataGrid)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void OnCheckedChanged(object sender, System.EventArgs e)
		{
			DataGrid.ReadOnly = LockedCheck.Checked;
		}
	}
}
