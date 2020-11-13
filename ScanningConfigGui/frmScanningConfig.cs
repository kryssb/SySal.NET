using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using SySal.OperaDb;

namespace SySal.Executables.ScanningConfigGui
{
	/// <summary>
	/// GUI tool to set up a scanning configuration.
	/// </summary>
	public class frmConfig : System.Windows.Forms.Form
	{
		private System.Windows.Forms.OpenFileDialog openFileDialog1;
		private System.Windows.Forms.TextBox txtConfigName;
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.Button btnLoad;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private long _configId;
		private OperaDbConnection _connection;
		private string _scanningSettingsString;
		private string _configName;
		private System.Windows.Forms.TextBox txtSettings;
		private string _originalConfigName = "";
		
		public static long Get(ref long id, OperaDbConnection conn)
		{
			//MessageBox.Show("tada!");
			frmConfig form =  new frmConfig(id, conn);
			form.ShowDialog();
			return form._configId;
		}

		public frmConfig(long id, OperaDbConnection conn)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			_configId = id;
			_connection = conn;
			
			if (id != 0)
			{
				_scanningSettingsString = (string) (new OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + id, _connection, null)).ExecuteScalar();
				_configName = ((string) (new OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + id, _connection, null)).ExecuteScalar()).Trim();
				txtSettings.Text = _scanningSettingsString;
				txtConfigName.Text = _configName;
				_originalConfigName = _configName;
			}
			btnCreate.Enabled = false;
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
			this.txtConfigName = new System.Windows.Forms.TextBox();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
			this.btnLoad = new System.Windows.Forms.Button();
			this.txtSettings = new System.Windows.Forms.TextBox();
			this.SuspendLayout();
			// 
			// txtConfigName
			// 
			this.txtConfigName.Location = new System.Drawing.Point(16, 16);
			this.txtConfigName.Name = "txtConfigName";
			this.txtConfigName.Size = new System.Drawing.Size(296, 20);
			this.txtConfigName.TabIndex = 0;
			this.txtConfigName.Text = "";
			this.txtConfigName.TextChanged += new System.EventHandler(this.txtConfigName_TextChanged);
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(16, 432);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 2;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(320, 432);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 3;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// btnLoad
			// 
			this.btnLoad.Location = new System.Drawing.Point(320, 16);
			this.btnLoad.Name = "btnLoad";
			this.btnLoad.TabIndex = 4;
			this.btnLoad.Text = "Load";
			this.btnLoad.Click += new System.EventHandler(this.btnLoad_Click);
			// 
			// txtSettings
			// 
			this.txtSettings.Location = new System.Drawing.Point(16, 48);
			this.txtSettings.Multiline = true;
			this.txtSettings.Name = "txtSettings";
			this.txtSettings.Size = new System.Drawing.Size(376, 376);
			this.txtSettings.TabIndex = 5;
			this.txtSettings.Text = "";
			// 
			// frmConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(408, 462);
			this.Controls.Add(this.txtSettings);
			this.Controls.Add(this.btnLoad);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.txtConfigName);
			this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
			this.Name = "frmConfig";
			this.Text = "Edit scanning configuration";
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			OperaDbCredentials cred = OperaDbCredentials.CreateFromRecord();
			OperaDbConnection conn = new OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
			conn.Open();
			//Application.Run(new frmConfig((long)5e15+600275, conn));
			Application.Run(new frmConfig(0, conn));
		}

		private void btnLoad_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog openFileDialog1 = new OpenFileDialog();
			openFileDialog1.InitialDirectory = @"c:\Sysal.NET.beta2\Settings" ;
			openFileDialog1.Filter = "Xml files (*.xml)|*.xml" ;			
			openFileDialog1.RestoreDirectory = true ;
			System.IO.StreamReader sr;
			if (openFileDialog1.ShowDialog() == DialogResult.OK)
			{		
				sr = System.IO.File.OpenText(openFileDialog1.FileName);			
				txtSettings.Text = sr.ReadToEnd();				
			}
		}

		private void txtConfigName_TextChanged(object sender, System.EventArgs e)
		{
			_configName = txtConfigName.Text.Trim();
			btnCreate.Enabled = _originalConfigName != _configName 
				&& _configName != "";
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			long newid = WriteToDb(txtConfigName.Text, "ScanServer.exe", 0, 0, txtSettings.Text);
			if (newid != 0) _configId = newid;
			MessageBox.Show("Scanning configuration with id " + newid.ToString() + " created");
			Close();
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private long WriteToDb(string desc, string exe, int driverlevel, int marks, string settings)
		{		
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();						
			long authorid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM VW_USERS WHERE UPPER(USERNAME) = UPPER('" + cred.OPERAUserName + "') ", _connection, null).ExecuteScalar());
			SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROGRAMSETTINGS(:description, :exe, :authorid, :driverlevel, :marks, :settings, :newid)", _connection);					
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

	}
}
