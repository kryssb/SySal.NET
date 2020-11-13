using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.AreaScan2Driver
{
	/// <summary>
	/// Summary description for frmReuseSettings.
	/// </summary>
	internal class frmReuseSettings : System.Windows.Forms.Form
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.ListBox lstAvailable;
		private System.Windows.Forms.ListBox lstSelected;
		private System.Windows.Forms.CheckBox chkDisplayAll;
		private System.Windows.Forms.Button btnSelect;
		private System.Windows.Forms.Label lblAvailable;
		private System.Windows.Forms.Label lblSelected;
		private System.Windows.Forms.Button btnUnselect;
		private System.Windows.Forms.Button btnSelectAll;
		private System.Windows.Forms.Button btnUnselectAll;
		private System.Windows.Forms.Button btnAccept;
		private System.Windows.Forms.Button btnCancel;

		SySal.OperaDb.OperaDbConnection _conn;
		public long[] SelectedIds;
		

		public static long[] Get(SySal.OperaDb.OperaDbConnection conn, long[] ids)
		{
			frmReuseSettings form = new frmReuseSettings(conn, ids);
			form.ShowDialog();
			return form.SelectedIds;
		}

		public frmReuseSettings(SySal.OperaDb.OperaDbConnection conn, long[] ids)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			_conn = conn;
			System.Data.DataSet ds = new System.Data.DataSet();
			SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='AreaScan2Driver.exe'", _conn);
			da.Fill(ds);
			//dataGrid1.SetDataBinding(ds, "Table");
			string qry = "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE = 'AreaScan2Driver.exe'";
			if (chkDisplayAll.Checked == true) qry = "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE DRIVERLEVEL=1";
			Utilities.FillListBox(lstAvailable, qry, _conn);						
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
			this.lstAvailable = new System.Windows.Forms.ListBox();
			this.lstSelected = new System.Windows.Forms.ListBox();
			this.chkDisplayAll = new System.Windows.Forms.CheckBox();
			this.btnSelect = new System.Windows.Forms.Button();
			this.btnUnselect = new System.Windows.Forms.Button();
			this.lblAvailable = new System.Windows.Forms.Label();
			this.lblSelected = new System.Windows.Forms.Label();
			this.btnSelectAll = new System.Windows.Forms.Button();
			this.btnUnselectAll = new System.Windows.Forms.Button();
			this.btnAccept = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// lstAvailable
			// 
			this.lstAvailable.Location = new System.Drawing.Point(8, 72);
			this.lstAvailable.Name = "lstAvailable";
			this.lstAvailable.Size = new System.Drawing.Size(240, 251);
			this.lstAvailable.TabIndex = 2;
			// 
			// lstSelected
			// 
			this.lstSelected.Location = new System.Drawing.Point(328, 72);
			this.lstSelected.Name = "lstSelected";
			this.lstSelected.Size = new System.Drawing.Size(256, 251);
			this.lstSelected.TabIndex = 3;
			// 
			// chkDisplayAll
			// 
			this.chkDisplayAll.Location = new System.Drawing.Point(16, 40);
			this.chkDisplayAll.Name = "chkDisplayAll";
			this.chkDisplayAll.Size = new System.Drawing.Size(240, 24);
			this.chkDisplayAll.TabIndex = 4;
			this.chkDisplayAll.Text = "Display all level-1 driver settings";
			this.chkDisplayAll.CheckedChanged += new System.EventHandler(this.chkDisplayAll_CheckedChanged);
			// 
			// btnSelect
			// 
			this.btnSelect.Location = new System.Drawing.Point(272, 72);
			this.btnSelect.Name = "btnSelect";
			this.btnSelect.Size = new System.Drawing.Size(32, 24);
			this.btnSelect.TabIndex = 5;
			this.btnSelect.Text = ">";
			this.btnSelect.Click += new System.EventHandler(this.btnSelect_Click);
			// 
			// btnUnselect
			// 
			this.btnUnselect.Location = new System.Drawing.Point(272, 112);
			this.btnUnselect.Name = "btnUnselect";
			this.btnUnselect.Size = new System.Drawing.Size(32, 24);
			this.btnUnselect.TabIndex = 6;
			this.btnUnselect.Text = "<";
			this.btnUnselect.Click += new System.EventHandler(this.btnUnselect_Click);
			// 
			// lblAvailable
			// 
			this.lblAvailable.Location = new System.Drawing.Point(32, 8);
			this.lblAvailable.Name = "lblAvailable";
			this.lblAvailable.TabIndex = 7;
			this.lblAvailable.Text = "Available settings:";
			// 
			// lblSelected
			// 
			this.lblSelected.Location = new System.Drawing.Point(344, 16);
			this.lblSelected.Name = "lblSelected";
			this.lblSelected.TabIndex = 8;
			this.lblSelected.Text = "Selected settings:";
			// 
			// btnSelectAll
			// 
			this.btnSelectAll.Location = new System.Drawing.Point(272, 248);
			this.btnSelectAll.Name = "btnSelectAll";
			this.btnSelectAll.Size = new System.Drawing.Size(32, 24);
			this.btnSelectAll.TabIndex = 9;
			this.btnSelectAll.Text = ">>";
			this.btnSelectAll.Click += new System.EventHandler(this.btnSelectAll_Click);
			// 
			// btnUnselectAll
			// 
			this.btnUnselectAll.Location = new System.Drawing.Point(272, 288);
			this.btnUnselectAll.Name = "btnUnselectAll";
			this.btnUnselectAll.Size = new System.Drawing.Size(32, 24);
			this.btnUnselectAll.TabIndex = 10;
			this.btnUnselectAll.Text = "<<";
			this.btnUnselectAll.Click += new System.EventHandler(this.btnUnselectAll_Click);
			// 
			// btnAccept
			// 
			this.btnAccept.Location = new System.Drawing.Point(424, 336);
			this.btnAccept.Name = "btnAccept";
			this.btnAccept.TabIndex = 11;
			this.btnAccept.Text = "Accept";
			this.btnAccept.Click += new System.EventHandler(this.btnAccept_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(512, 336);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 12;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// frmReuseSettings
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(592, 366);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.btnAccept);
			this.Controls.Add(this.btnUnselectAll);
			this.Controls.Add(this.btnSelectAll);
			this.Controls.Add(this.lblSelected);
			this.Controls.Add(this.lblAvailable);
			this.Controls.Add(this.btnUnselect);
			this.Controls.Add(this.btnSelect);
			this.Controls.Add(this.chkDisplayAll);
			this.Controls.Add(this.lstSelected);
			this.Controls.Add(this.lstAvailable);
			this.Name = "frmReuseSettings";
			this.Text = "Choose reuse settings";
			this.ResumeLayout(false);

		}
		#endregion

		private void chkDisplayAll_CheckedChanged(object sender, System.EventArgs e)
		{
			string qry = "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE = 'AreaScan2Driver.exe'";
			if (chkDisplayAll.Checked == true) qry = "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE DRIVERLEVEL=1";
			Utilities.FillListBox(lstAvailable, qry, _conn);
		}

		private bool IsAlreadySelected(long id)
		{			
			foreach (Utilities.ConfigItem item in lstSelected.Items)
			{
				if (id == item.Id) return true;
			}
			return false;
		}

		private bool IsAlreadySelected(Utilities.ConfigItem o)
		{
			return IsAlreadySelected(o.Id);
		}

		private void btnSelect_Click(object sender, System.EventArgs e)
		{	
			Utilities.ConfigItem item = (Utilities.ConfigItem) lstAvailable.SelectedItem;
			if (item == null) return;
			if (IsAlreadySelected(item)) return;
			lstSelected.Items.Add(item);
		}

		private void btnUnselect_Click(object sender, System.EventArgs e)
		{
			Utilities.ConfigItem item = (Utilities.ConfigItem) lstSelected.SelectedItem;
			if (item == null) return;
			lstSelected.Items.Remove(item);
		}

		private void btnSelectAll_Click(object sender, System.EventArgs e)
		{
			lstSelected.Items.Clear();
			foreach (object item in lstAvailable.Items)
			{
				lstSelected.Items.Add(item);
				//lstAvailable.Items.Remove(item);
			}
			lstAvailable.Items.Clear();
			/*foreach (object item in lstAvailable.Items)
			{
			//	lstSelected.Items.Add(item);
				lstAvailable.Items.Remove(item);
			}*/
		}

		private void btnUnselectAll_Click(object sender, System.EventArgs e)
		{			
			lstSelected.Items.Clear();
			chkDisplayAll_CheckedChanged(sender, e);
		}

		private void btnAccept_Click(object sender, System.EventArgs e)
		{			
			int i=0;
			SelectedIds = new long[lstSelected.Items.Count];
			foreach (object item in lstSelected.Items)
			{				
				SelectedIds[i++] = ((Utilities.ConfigItem) item).Id;
			}
			Close();
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		
	}
}
