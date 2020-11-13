using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.QualityCutConfig
{
	/// <summary>
	/// GUI tool to configure options for TLGSel.
	/// </summary>
	public class frmConfig : System.Windows.Forms.Form
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private long _configId;
		private System.Windows.Forms.Label lblConfigName;
		private System.Windows.Forms.TextBox txtConfigName;
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.TextBox txtCut;
		private System.Windows.Forms.Label lblCut;
		private string _originalConfigName;		
		private SySal.OperaDb.OperaDbConnection _connection;
		public  SySal.OperaDb.OperaDbConnection Connection { get{return _connection;} set {_connection = value;} }
		public long ConfigId { get{return _configId;} set {_configId = value;} }

		private long WriteToDb(string desc, string exe, int driverlevel, int marks, string settings)
		{		
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();			
			
			long authorid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM VW_USERS WHERE UPPER(USERNAME) = UPPER('" + cred.OPERAUserName + "') ", _connection, null).ExecuteScalar());
			SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROGRAMSETTINGS(:description, :exe, :authorid, :driverlevel, :marks, :settings, :newid)", _connection);
			//SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_TEST(:description)", _connection);
		
			cmd.Parameters.Add("description", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = desc;
			cmd.Parameters.Add("exe", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = exe;
			cmd.Parameters.Add("authorid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = authorid;
			cmd.Parameters.Add("driverlevel", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = driverlevel;
			cmd.Parameters.Add("marks", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = marks;
			cmd.Parameters.Add("settings", SySal.OperaDb.OperaDbType.CLOB, System.Data.ParameterDirection.Input).Value = settings;
			cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
			

			try
			{
				cmd.ExecuteNonQuery();
				return (long) cmd.Parameters["newid"].Value;
			//	return 1;
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message);
				return 0;
			}			
		
		}

		public static long GetConfig(long id, SySal.OperaDb.OperaDbConnection conn)
		{
			frmConfig configForm = new frmConfig(id, conn);
			configForm.ShowDialog();
			return configForm.ConfigId;
		}
		public long Get(long id, SySal.OperaDb.OperaDbConnection conn)
		{
			frmConfig configForm = new frmConfig(id, conn);
			//configForm.ConfigId = id;
			//configForm.Connection = conn;
			configForm.ShowDialog();
			return configForm.ConfigId;
		}

		public frmConfig(long id, SySal.OperaDb.OperaDbConnection conn)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			//_configId = id;
			//textBox1.DataBindings.Add("Text", this, "ConfigId");	
			_connection = conn;
			if (id != 0)
			{
				SySal.OperaDb.ComputingInfrastructure.ProgramSettings ps = new SySal.OperaDb.ComputingInfrastructure.ProgramSettings(id, conn, null);
				_originalConfigName = ps.Description;
				txtConfigName.Text = ps.Description;
				txtCut.Text = ps.Settings;
			}
			else
			{
				_originalConfigName = "";
				txtConfigName.Text = "";
				txtCut.Text = "";
				btnCreate.Enabled = false;
			}
			//txtConfigName.DataBindings.Add("Text", ps, "Description");
			//txtCut.DataBindings.Add("Text", ps, "Settings");
			
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
			System.Resources.ResourceManager resources = new System.Resources.ResourceManager(typeof(frmConfig));
			this.lblConfigName = new System.Windows.Forms.Label();
			this.txtConfigName = new System.Windows.Forms.TextBox();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.txtCut = new System.Windows.Forms.TextBox();
			this.lblCut = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// lblConfigName
			// 
			this.lblConfigName.Location = new System.Drawing.Point(16, 16);
			this.lblConfigName.Name = "lblConfigName";
			this.lblConfigName.Size = new System.Drawing.Size(200, 16);
			this.lblConfigName.TabIndex = 0;
			this.lblConfigName.Text = "Configuration name:";
			// 
			// txtConfigName
			// 
			this.txtConfigName.Location = new System.Drawing.Point(8, 40);
			this.txtConfigName.Name = "txtConfigName";
			this.txtConfigName.Size = new System.Drawing.Size(376, 20);
			this.txtConfigName.TabIndex = 1;
			this.txtConfigName.Text = "";
			this.txtConfigName.TextChanged += new System.EventHandler(this.txtConfigName_TextChanged);
			// 
			// btnCreate
			// 
			this.btnCreate.Location = new System.Drawing.Point(8, 136);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 2;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(312, 136);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 3;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// txtCut
			// 
			this.txtCut.Location = new System.Drawing.Point(8, 96);
			this.txtCut.Name = "txtCut";
			this.txtCut.Size = new System.Drawing.Size(376, 20);
			this.txtCut.TabIndex = 4;
			this.txtCut.Text = "";
			this.txtCut.TextChanged += new System.EventHandler(this.txtCut_TextChanged);
			// 
			// lblCut
			// 
			this.lblCut.Location = new System.Drawing.Point(16, 72);
			this.lblCut.Name = "lblCut";
			this.lblCut.Size = new System.Drawing.Size(100, 16);
			this.lblCut.TabIndex = 5;
			this.lblCut.Text = "Cut:";
			// 
			// frmConfig
			// 
			this.AcceptButton = this.btnCreate;
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(392, 166);
			this.Controls.Add(this.lblCut);
			this.Controls.Add(this.txtCut);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.txtConfigName);
			this.Controls.Add(this.lblConfigName);
			this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
			this.Name = "frmConfig";
			this.Text = "Quality cut";
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{			
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			SySal.OperaDb.OperaDbConnection connection = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
			connection.Open();
			//Application.Run(new frmConfig((long)5e15+600106, connection));
			Application.Run(new frmConfig(0, connection));
		}

		private void txtConfigName_TextChanged(object sender, System.EventArgs e)
		{
			//bool test1 = (_originalConfigName.Trim() != txtConfigName.Text.Trim());
			//bool test2 = txtCut.Text.Trim() != "";
			btnCreate.Enabled = (_originalConfigName.Trim() != txtConfigName.Text.Trim() & 
				txtCut.Text.Trim() != "");
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			ConfigId = WriteToDb(txtConfigName.Text, "TLGSel.exe", 0, 0, txtCut.Text);
			if (ConfigId != 0) MessageBox.Show("Created new quality cut with id " + ConfigId);
			Close();
		}

		private void txtCut_TextChanged(object sender, System.EventArgs e)
		{			
			btnCreate.Enabled = (_originalConfigName.Trim() != txtConfigName.Text.Trim() && txtCut.Text.Trim() != "");
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}
	}
}
