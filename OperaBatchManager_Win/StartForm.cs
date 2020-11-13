using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Services.OperaBatchManager_Win
{
	/// <summary>
	/// StartForm is used to start a new process operation.	
	/// </summary>
	/// <remarks>
	/// <para>
	/// The process operation must start with a definite set of credentials. 
	/// The OPERA Computing Infrastructure credentials are set in this form. 
	/// The DB access credentials are null (empty boxes), since the BatchManager will replace them with its own DB access credentials.
	/// </para>
	/// <para>
	/// This form shows four list boxes to select:
	/// <list type="bullet">
	/// <item><term>the driver executable</term></item>
	/// <item><term>the machine to run the process operation (for drivers of level 1,2,3 select an appropriate ScanningServer; for higher levels, select a BatchManager)</term></item>
	/// <item><term>the brick involved in the process operation (if applicable)</term></item>
	/// <item><term>the plate involved in the process operation (if applicable)</term></item>
	/// </list>
	/// </para>
	/// <para>Depending on the operation driver selected, the brick and plate selections may or may not be applicable.</para>
	/// <para>For a level-1 operation (<c>Scanning</c> level), both brick and plate must be selected.</para>
	/// <para>For a level-2 operation (<c>Volume</c> level), the brick must be selected.</para>
	/// <para>For a level-3 operation (<c>Brick</c> level), the brick must be selected.</para>
	/// <para>For a level-4 or higher operation (<c>System</c> level), neither the brick nor the plate must be selected.</para>
	/// </remarks>
	public class StartForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.ColumnHeader columnHeader4;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.ColumnHeader columnHeader5;
		private System.Windows.Forms.ColumnHeader columnHeader6;
		private System.Windows.Forms.ColumnHeader columnHeader7;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.ColumnHeader columnHeader8;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox OPERAUsernameText;
		private System.Windows.Forms.TextBox OPERAPasswordText;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Button m_OKButton;
		private System.Windows.Forms.Button m_CancelButton;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.ListView ProgramList;
		private System.Windows.Forms.ListView BrickList;
		private System.Windows.Forms.ListView PlateList;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.ListView MachineList;
		private System.Windows.Forms.ColumnHeader columnHeader9;
		private System.Windows.Forms.ColumnHeader columnHeader10;
		private System.Windows.Forms.ColumnHeader columnHeader11;
		internal System.Windows.Forms.TextBox NotesText;
		private System.Windows.Forms.Label label3;
        private CheckBox ShowFavoriteOnlyCheck;

		SySal.OperaDb.OperaDbConnection DBConn;

		public StartForm()
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.ShowFavoriteOnlyCheck = new System.Windows.Forms.CheckBox();
            this.ProgramList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.BrickList = new System.Windows.Forms.ListView();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.PlateList = new System.Windows.Forms.ListView();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.label1 = new System.Windows.Forms.Label();
            this.OPERAUsernameText = new System.Windows.Forms.TextBox();
            this.OPERAPasswordText = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.m_OKButton = new System.Windows.Forms.Button();
            this.m_CancelButton = new System.Windows.Forms.Button();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.MachineList = new System.Windows.Forms.ListView();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.NotesText = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.ShowFavoriteOnlyCheck);
            this.groupBox1.Controls.Add(this.ProgramList);
            this.groupBox1.Location = new System.Drawing.Point(8, 8);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(608, 264);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Program settings";
            // 
            // ShowFavoriteOnlyCheck
            // 
            this.ShowFavoriteOnlyCheck.AutoSize = true;
            this.ShowFavoriteOnlyCheck.Checked = true;
            this.ShowFavoriteOnlyCheck.CheckState = System.Windows.Forms.CheckState.Checked;
            this.ShowFavoriteOnlyCheck.Location = new System.Drawing.Point(10, 16);
            this.ShowFavoriteOnlyCheck.Name = "ShowFavoriteOnlyCheck";
            this.ShowFavoriteOnlyCheck.Size = new System.Drawing.Size(169, 17);
            this.ShowFavoriteOnlyCheck.TabIndex = 1;
            this.ShowFavoriteOnlyCheck.Text = "Show \"favorite\" programs only";
            this.ShowFavoriteOnlyCheck.UseVisualStyleBackColor = true;
            this.ShowFavoriteOnlyCheck.CheckedChanged += new System.EventHandler(this.OnFavoriteChanged);
            // 
            // ProgramList
            // 
            this.ProgramList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4});
            this.ProgramList.FullRowSelect = true;
            this.ProgramList.GridLines = true;
            this.ProgramList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.ProgramList.HideSelection = false;
            this.ProgramList.Location = new System.Drawing.Point(8, 38);
            this.ProgramList.MultiSelect = false;
            this.ProgramList.Name = "ProgramList";
            this.ProgramList.Size = new System.Drawing.Size(592, 215);
            this.ProgramList.TabIndex = 0;
            this.ProgramList.UseCompatibleStateImageBehavior = false;
            this.ProgramList.View = System.Windows.Forms.View.Details;
            this.ProgramList.SelectedIndexChanged += new System.EventHandler(this.OnSelectedProgramChanged);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "ID";
            this.columnHeader1.Width = 120;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Executable";
            this.columnHeader2.Width = 120;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Description";
            this.columnHeader3.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader3.Width = 200;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Type";
            this.columnHeader4.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader4.Width = 120;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.BrickList);
            this.groupBox2.Location = new System.Drawing.Point(8, 280);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(232, 264);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Brick";
            // 
            // BrickList
            // 
            this.BrickList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader5,
            this.columnHeader6,
            this.columnHeader7});
            this.BrickList.FullRowSelect = true;
            this.BrickList.GridLines = true;
            this.BrickList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.BrickList.HideSelection = false;
            this.BrickList.Location = new System.Drawing.Point(8, 16);
            this.BrickList.MultiSelect = false;
            this.BrickList.Name = "BrickList";
            this.BrickList.Size = new System.Drawing.Size(216, 240);
            this.BrickList.TabIndex = 1;
            this.BrickList.UseCompatibleStateImageBehavior = false;
            this.BrickList.View = System.Windows.Forms.View.Details;
            this.BrickList.SelectedIndexChanged += new System.EventHandler(this.OnSelectedBrickChanged);
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "ID";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Set";
            this.columnHeader6.Width = 91;
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Brick ID";
            this.columnHeader7.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.PlateList);
            this.groupBox3.Location = new System.Drawing.Point(248, 280);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(88, 264);
            this.groupBox3.TabIndex = 2;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Plate";
            // 
            // PlateList
            // 
            this.PlateList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader8});
            this.PlateList.FullRowSelect = true;
            this.PlateList.GridLines = true;
            this.PlateList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.PlateList.HideSelection = false;
            this.PlateList.Location = new System.Drawing.Point(8, 16);
            this.PlateList.MultiSelect = false;
            this.PlateList.Name = "PlateList";
            this.PlateList.Size = new System.Drawing.Size(72, 240);
            this.PlateList.TabIndex = 2;
            this.PlateList.UseCompatibleStateImageBehavior = false;
            this.PlateList.View = System.Windows.Forms.View.Details;
            this.PlateList.SelectedIndexChanged += new System.EventHandler(this.OnSelectedPlateChanged);
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "ID";
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(8, 584);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(104, 24);
            this.label1.TabIndex = 3;
            this.label1.Text = "OPERA Username";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // OPERAUsernameText
            // 
            this.OPERAUsernameText.Location = new System.Drawing.Point(112, 584);
            this.OPERAUsernameText.Name = "OPERAUsernameText";
            this.OPERAUsernameText.Size = new System.Drawing.Size(128, 20);
            this.OPERAUsernameText.TabIndex = 5;
            // 
            // OPERAPasswordText
            // 
            this.OPERAPasswordText.Location = new System.Drawing.Point(352, 584);
            this.OPERAPasswordText.Name = "OPERAPasswordText";
            this.OPERAPasswordText.PasswordChar = '*';
            this.OPERAPasswordText.Size = new System.Drawing.Size(128, 20);
            this.OPERAPasswordText.TabIndex = 6;
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(248, 584);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(104, 24);
            this.label2.TabIndex = 5;
            this.label2.Text = "OPERA Password";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // m_OKButton
            // 
            this.m_OKButton.Location = new System.Drawing.Point(528, 552);
            this.m_OKButton.Name = "m_OKButton";
            this.m_OKButton.Size = new System.Drawing.Size(88, 24);
            this.m_OKButton.TabIndex = 7;
            this.m_OKButton.Text = "OK";
            this.m_OKButton.Click += new System.EventHandler(this.m_OKButton_Click);
            // 
            // m_CancelButton
            // 
            this.m_CancelButton.Location = new System.Drawing.Point(528, 584);
            this.m_CancelButton.Name = "m_CancelButton";
            this.m_CancelButton.Size = new System.Drawing.Size(88, 24);
            this.m_CancelButton.TabIndex = 8;
            this.m_CancelButton.Text = "Cancel";
            this.m_CancelButton.Click += new System.EventHandler(this.m_CancelButton_Click);
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.MachineList);
            this.groupBox4.Location = new System.Drawing.Point(344, 280);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(272, 264);
            this.groupBox4.TabIndex = 3;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Machine";
            // 
            // MachineList
            // 
            this.MachineList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader9,
            this.columnHeader10,
            this.columnHeader11});
            this.MachineList.FullRowSelect = true;
            this.MachineList.GridLines = true;
            this.MachineList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.MachineList.HideSelection = false;
            this.MachineList.Location = new System.Drawing.Point(8, 16);
            this.MachineList.MultiSelect = false;
            this.MachineList.Name = "MachineList";
            this.MachineList.Size = new System.Drawing.Size(256, 240);
            this.MachineList.TabIndex = 3;
            this.MachineList.UseCompatibleStateImageBehavior = false;
            this.MachineList.View = System.Windows.Forms.View.Details;
            this.MachineList.SelectedIndexChanged += new System.EventHandler(this.OnSelectedMachineChanged);
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "ID";
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "Name";
            this.columnHeader10.Width = 90;
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "Address";
            this.columnHeader11.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader11.Width = 100;
            // 
            // NotesText
            // 
            this.NotesText.Location = new System.Drawing.Point(112, 552);
            this.NotesText.Name = "NotesText";
            this.NotesText.Size = new System.Drawing.Size(368, 20);
            this.NotesText.TabIndex = 4;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(8, 552);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 24);
            this.label3.TabIndex = 10;
            this.label3.Text = "Notes";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // StartForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(626, 616);
            this.Controls.Add(this.NotesText);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.m_CancelButton);
            this.Controls.Add(this.m_OKButton);
            this.Controls.Add(this.OPERAPasswordText);
            this.Controls.Add(this.OPERAUsernameText);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "StartForm";
            this.Text = "Start Process Operation";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Closing += new System.ComponentModel.CancelEventHandler(this.OnClose);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		internal long sel_ProgramSettingsId;
		internal SySal.DAQSystem.Drivers.DriverType sel_DriverType;
		internal long sel_BrickId;
		internal int sel_PlateId;
		internal long sel_MachineId;
		internal string sel_Username;
		internal string sel_Password;

		private void OnLoad(object sender, System.EventArgs e)
		{
			DBConn = new SySal.OperaDb.OperaDbConnection(MainForm.DBServer, MainForm.DBUserName, MainForm.DBPassword);		
			DBConn.Open();
			System.Data.DataSet ds;
			ds = new System.Data.DataSet();
            ProgramList.Items.Clear();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, EXECUTABLE, " +
                ((progsetwhere.Length == 0) ? "DESCRIPTION" : "trim(substr(name, instr(name, 'PROGSET') + 7)) as DESCRIPTION") +
                ", DRIVERLEVEL FROM TB_PROGRAMSETTINGS " + progsetwhere + " ORDER BY ID ASC", DBConn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				ListViewItem lvi = ProgramList.Items.Add(dr[0].ToString());
				lvi.SubItems.Add(dr[1].ToString());
				lvi.SubItems.Add(dr[2].ToString());
				lvi.SubItems.Add(dr[3].ToString());
			}
            BrickList.Items.Clear();
			ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, ID_SET, ID_BRICK FROM TB_EVENTBRICKS ORDER BY ID ASC", DBConn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				ListViewItem lvi = BrickList.Items.Add(dr[0].ToString());
				lvi.SubItems.Add(dr[1].ToString());
				lvi.SubItems.Add(dr[2].ToString());
			}
            MachineList.Items.Clear();
			ds = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, NAME, ADDRESS FROM TB_MACHINES WHERE ID_SITE IN (SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE') ORDER BY ID ASC", DBConn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				ListViewItem lvi = MachineList.Items.Add(dr[0].ToString());
				lvi.SubItems.Add(dr[1].ToString());
				lvi.SubItems.Add(dr[2].ToString());
			}
			NotesText.Text = "";
		}

		private void OnClose(object sender, System.ComponentModel.CancelEventArgs e)
		{
			if (DBConn != null) DBConn.Close();
		}

		private void OnSelectedProgramChanged(object sender, System.EventArgs e)
		{
            try
            {
                if (ProgramList.SelectedItems.Count == 0) return;
                sel_ProgramSettingsId = Convert.ToInt64(ProgramList.SelectedItems[0].SubItems[0].Text);
                int driverlevel = Convert.ToInt32(ProgramList.SelectedItems[0].SubItems[3].Text);
                if (driverlevel == 1)
                {
                    BrickList.Enabled = true;
                    PlateList.Enabled = true;
                    sel_DriverType = SySal.DAQSystem.Drivers.DriverType.Scanning;
                    if (BrickList.SelectedItems.Count == 0 && BrickList.Items.Count > 0) BrickList.Items[0].Selected = true;
                }
                else if (driverlevel == 2 || driverlevel == 3)
                {
                    BrickList.Enabled = true;
                    PlateList.Items.Clear();
                    PlateList.Enabled = false;
                    sel_DriverType = (driverlevel == 2) ? SySal.DAQSystem.Drivers.DriverType.Volume : SySal.DAQSystem.Drivers.DriverType.Brick;
                    if (BrickList.SelectedItems.Count == 0 && BrickList.Items.Count > 0) BrickList.Items[0].Selected = true;
                }
                else
                {
                    BrickList.Enabled = false;
                    PlateList.Items.Clear();
                    PlateList.Enabled = false;
                    sel_DriverType = SySal.DAQSystem.Drivers.DriverType.System;
                }
            }
            catch (Exception) { }
		}

		private void OnSelectedBrickChanged(object sender, System.EventArgs e)
		{
            try
            {
                if (BrickList.SelectedItems.Count == 0) return;
                sel_BrickId = Convert.ToInt64(BrickList.SelectedItems[0].SubItems[0].Text);
                System.Data.DataSet ds = new System.Data.DataSet();
                PlateList.Items.Clear();
                if (sel_DriverType == SySal.DAQSystem.Drivers.DriverType.Scanning)
                {
                    new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM VW_PLATES WHERE ID_EVENTBRICK = " + BrickList.SelectedItems[0].SubItems[0].Text + " AND DAMAGED = 'N' ORDER BY ID ASC", DBConn, null).Fill(ds);
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                        PlateList.Items.Add(dr[0].ToString());
                    if (PlateList.Items.Count > 0)
                        PlateList.Items[0].Selected = true;
                    PlateList.Enabled = true;
                }
                else PlateList.Enabled = false;
            }
            catch (Exception) { }
		}

		private void OnSelectedPlateChanged(object sender, System.EventArgs e)
		{
            try
            {
                if (PlateList.SelectedItems.Count > 0)
                    sel_PlateId = Convert.ToInt32(PlateList.SelectedItems[0].SubItems[0].Text);
            }
            catch (Exception) { }
		}

		private void OnSelectedMachineChanged(object sender, System.EventArgs e)
		{
            try
            {
                sel_MachineId = Convert.ToInt64(MachineList.SelectedItems[0].SubItems[0].Text);
            }
            catch (Exception) { }
		}

		private void m_OKButton_Click(object sender, System.EventArgs e)
		{
            try
            {
                DialogResult = DialogResult.OK;
                sel_Username = OPERAUsernameText.Text;
                sel_Password = OPERAPasswordText.Text;
                Close();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
		}

		private void m_CancelButton_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();		
		}

        private string progsetwhere = favprogset;

        //private const string favprogset = "where ID in (SELECT VALUE FROM OPERA.LZ_SITEVARS WHERE upper(NAME) like 'PROGSET %')";
        private const string favprogset = "inner join (SELECT VALUE, NAME FROM OPERA.LZ_SITEVARS WHERE upper(NAME) like 'PROGSET %') on (ID = VALUE)";

        private void OnFavoriteChanged(object sender, EventArgs e)
        {
            progsetwhere = ShowFavoriteOnlyCheck.Checked ? favprogset : "";
            OnLoad(sender, e);
        }
	}
}
