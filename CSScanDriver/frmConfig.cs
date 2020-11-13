using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.CSScanDriver
{
	/// <summary>
	/// Summary description for frmConfig.
	/// </summary>
	public class frmConfig : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button areascanEditButton;
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private static long _settingsId;
		private SySal.OperaDb.OperaDbConnection _connection;
		private string _currentConfigName = "";		
		private string _originalConfigName = "";		
		private long _wasdSettingsId;
		//private long _predSettingsId;
		private System.Windows.Forms.ComboBox cmbWasd;
		private System.Windows.Forms.Label lblWasdScan;		
		private string _subDriverName = "WideAreaScanDriver.exe";
		//private string _predDriverName = "PredictionScanDriver.exe";
		public CSScanDriverSettings _progSettings;

		public static long GetConfig(long config, SySal.OperaDb.OperaDbConnection connection)
		{
			new frmConfig(config, connection).ShowDialog();
			return _settingsId;
		}

		public frmConfig(long settingsId, SySal.OperaDb.OperaDbConnection connection)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			_connection = connection;
			Utilities.FillComboBox(cmbWasd, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='"+ _subDriverName + "'", _connection);

			if (settingsId != 0)
			{
				string settings = (string)(new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, _connection)).ExecuteScalar();								
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(CSScanDriverSettings));
				_progSettings = (CSScanDriverSettings) xmls.Deserialize(new System.IO.StringReader(settings));
				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, _connection)).ExecuteScalar();								
			}
			else
			{
				_progSettings = new CSScanDriverSettings();			
			}
			_wasdSettingsId = _progSettings.WideAreaConfigId;
			//_predSettingsId = _progSettings.PredScanConfigId;
			//_progSettings.InitialQuery;
			Utilities.SelectId(cmbWasd, _wasdSettingsId);		
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
			this.areascanEditButton = new System.Windows.Forms.Button();
			this.cmbWasd = new System.Windows.Forms.ComboBox();
			this.lblWasdScan = new System.Windows.Forms.Label();
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.configNameLabel = new System.Windows.Forms.Label();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// areascanEditButton
			// 
			this.areascanEditButton.Location = new System.Drawing.Point(544, 48);
			this.areascanEditButton.Name = "areascanEditButton";
			this.areascanEditButton.TabIndex = 109;
			this.areascanEditButton.Text = "View/edit";
			this.areascanEditButton.Click += new System.EventHandler(this.areascanEditButton_Click);
			// 
			// cmbWasd
			// 
			this.cmbWasd.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbWasd.Location = new System.Drawing.Point(152, 48);
			this.cmbWasd.Name = "cmbWasd";
			this.cmbWasd.Size = new System.Drawing.Size(376, 21);
			this.cmbWasd.TabIndex = 108;
			this.cmbWasd.SelectedIndexChanged += new System.EventHandler(this.cmbWasd_SelectedIndexChanged);
			// 
			// lblWasdScan
			// 
			this.lblWasdScan.Location = new System.Drawing.Point(16, 48);
			this.lblWasdScan.Name = "lblWasdScan";
			this.lblWasdScan.Size = new System.Drawing.Size(88, 23);
			this.lblWasdScan.TabIndex = 107;
			this.lblWasdScan.Text = "WASD settings:";
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(152, 8);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(464, 20);
			this.configNameTextBox.TabIndex = 105;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(16, 8);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(128, 23);
			this.configNameLabel.TabIndex = 106;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(16, 80);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 104;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(544, 80);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 103;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// frmConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(624, 109);
			this.Controls.Add(this.areascanEditButton);
			this.Controls.Add(this.cmbWasd);
			this.Controls.Add(this.lblWasdScan);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.configNameLabel);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.btnCancel);
			this.Name = "frmConfig";
			this.Text = "CSScanDriver config";
			this.ResumeLayout(false);

		}
		#endregion

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void areascanEditButton_Click(object sender, System.EventArgs e)
		{
			string path = (string)(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME='ExeRepository'", _connection, null)).ExecuteScalar();
			object WideAreaScanDriverExe = System.Activator.CreateInstanceFrom(path + @"\WideAreaScanDriver.exe", 
				"WideAreaScanDriver.frmWideAreaScanConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.WideAreaConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) WideAreaScanDriverExe.GetType().InvokeMember("GetConfig", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
				null, WideAreaScanDriverExe, new object[] {_progSettings.WideAreaConfigId, _connection}
				);
			if (newid != 0) _progSettings.WideAreaConfigId = newid;
			Utilities.FillComboBox(cmbWasd, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='WideAreaScanDriver.exe'", _connection);
			Utilities.SelectId(cmbWasd, _progSettings.WideAreaConfigId);
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		public bool ConfigNameIsValid()
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			return _currentConfigName.Trim() != _originalConfigName.Trim() & _currentConfigName.Trim() != ""; 
		}

		private bool IsFormFilled()
		{			
			return _progSettings.WideAreaConfigId != 0;
            //return _progSettings.WideAreaConfigId != 0 & _progSettings.PredScanConfigId != 0;
			//return _progSettings.WideAreaConfigId != 0 && _progSettings.InitialQuery != "";
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{			
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(CSScanDriverSettings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, _progSettings);
			sw.Flush();
			_settingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "CSScanDriver.exe", 2, 0, sw.ToString());
			Close();		
		}

		private void cmbWasd_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.WideAreaConfigId  = ((Utilities.ConfigItem)cmbWasd.SelectedItem).Id;
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

//		private void predscanEditButton_Click(object sender, System.EventArgs e)
//		{
//			string path = (string)(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME='ExeRepository'", _connection, null)).ExecuteScalar();
//			object PredictionScanDriverExe = System.Activator.CreateInstanceFrom(path + @"\PredictionScanDriver.exe", 
//				"PredictionScanDriver.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.PredScanConfigId, _connection}, null, null, null).Unwrap();		
//			long newid = (long) PredictionScanDriverExe.GetType().InvokeMember("GetConfig", 
//				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
//				null, PredictionScanDriverExe, new object[] {_progSettings.PredScanConfigId, _connection}
//				);
//			if (newid != 0) _progSettings.PredScanConfigId = newid;
//		}

//		private void cmbPredScan_SelectedIndexChanged(object sender, System.EventArgs e)
//		{
//			_progSettings.PredScanConfigId   = ((Utilities.ConfigItem)cmbPredScan.SelectedItem).Id;
//			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
//		}
	}
}
