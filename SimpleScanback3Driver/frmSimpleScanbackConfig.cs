using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.SimpleScanback3Driver
{
	/// <summary>
	/// Summary description for frmConfig.
	/// </summary>
	internal class frmConfig : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Label maxMissedPlatesLabel;
		private System.Windows.Forms.Label directionLabel;
		private System.Windows.Forms.Label configNameLabel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private static long _settingsId;
		private long _intercalibrationSettingsId;
		private long _predscanSettingsId;
		private int _maxMissedPlates;
		private SySal.OperaDb.OperaDbConnection _connection;
		private string _currentConfigName = "";
		private System.Windows.Forms.ComboBox cmbIntercalibration;
		private System.Windows.Forms.ComboBox cmbPredscan;
		private System.Windows.Forms.ComboBox cmbDirection;
		private string _originalConfigName = "";
		private ScanDirection _direction;
		private System.Windows.Forms.TextBox txtMaxMissingPlates;
		private System.Windows.Forms.Button intercalibrationEditButton;
		private System.Windows.Forms.Button predscanEditButton;	
		private SimpleScanback3Settings _progSettings;
	
		public static long GetConfig(long settingsId, SySal.OperaDb.OperaDbConnection conn)
		{
			new frmConfig(settingsId, conn).ShowDialog();
			return _settingsId;
		}

		public frmConfig(long settingsId, SySal.OperaDb.OperaDbConnection conn)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			_connection = conn;
			
			//Utilities.FillComboBox(
			cmbDirection.Items.Add(ScanDirection.Upstream);			
			cmbDirection.Items.Add(ScanDirection.Downstream);	
			Utilities.FillComboBox(cmbIntercalibration, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='IntercalibrationDriver.exe'", _connection);
			Utilities.FillComboBox(cmbPredscan, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='PredictionScan2Driver.exe'", _connection);
			
			if (settingsId != 0)
			{
				string settings = (string)(new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();								
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SimpleScanback3Settings));
				_progSettings = (SimpleScanback3Settings) xmls.Deserialize(new System.IO.StringReader(settings));
				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
								
			}
			else
			{
				_progSettings = new SimpleScanback3Settings();
			}
			txtMaxMissingPlates.Text = _progSettings.MaxMissingPlates.ToString();
			Utilities.SelectId(cmbIntercalibration, _progSettings.IntercalibrationConfigId);
			Utilities.SelectId(cmbPredscan, _progSettings.PredictionScanConfigId);
			if (_progSettings.Direction == ScanDirection.Upstream) cmbDirection.SelectedIndex = 0;
			else if (_progSettings.Direction == ScanDirection.Downstream) cmbDirection.SelectedIndex = 1;
			else throw new Exception("Wrong direction: " + _progSettings.Direction);
			_intercalibrationSettingsId = _progSettings.IntercalibrationConfigId;
			_predscanSettingsId  = _progSettings.PredictionScanConfigId;
			_direction = _progSettings.Direction;
			_maxMissedPlates = _progSettings.MaxMissingPlates;
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
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.cmbIntercalibration = new System.Windows.Forms.ComboBox();
			this.cmbPredscan = new System.Windows.Forms.ComboBox();
			this.label2 = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.txtMaxMissingPlates = new System.Windows.Forms.TextBox();
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.maxMissedPlatesLabel = new System.Windows.Forms.Label();
			this.directionLabel = new System.Windows.Forms.Label();
			this.cmbDirection = new System.Windows.Forms.ComboBox();
			this.configNameLabel = new System.Windows.Forms.Label();
			this.intercalibrationEditButton = new System.Windows.Forms.Button();
			this.predscanEditButton = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(16, 192);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 77;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(544, 192);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 76;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// cmbIntercalibration
			// 
			this.cmbIntercalibration.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbIntercalibration.Location = new System.Drawing.Point(152, 104);
			this.cmbIntercalibration.Name = "cmbIntercalibration";
			this.cmbIntercalibration.Size = new System.Drawing.Size(376, 21);
			this.cmbIntercalibration.TabIndex = 87;
			this.cmbIntercalibration.SelectedIndexChanged += new System.EventHandler(this.cmbIntercalibration_SelectedIndexChanged);
			// 
			// cmbPredscan
			// 
			this.cmbPredscan.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbPredscan.Location = new System.Drawing.Point(152, 144);
			this.cmbPredscan.Name = "cmbPredscan";
			this.cmbPredscan.Size = new System.Drawing.Size(376, 21);
			this.cmbPredscan.TabIndex = 86;
			this.cmbPredscan.SelectedIndexChanged += new System.EventHandler(this.cmbPredscan_SelectedIndexChanged);
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(16, 144);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(128, 23);
			this.label2.TabIndex = 85;
			this.label2.Text = "Predscan2 settings:";
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(16, 104);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(128, 16);
			this.label1.TabIndex = 84;
			this.label1.Text = "Intercalibration settings:";
			// 
			// txtMaxMissingPlates
			// 
			this.txtMaxMissingPlates.Location = new System.Drawing.Point(552, 64);
			this.txtMaxMissingPlates.Name = "txtMaxMissingPlates";
			this.txtMaxMissingPlates.Size = new System.Drawing.Size(64, 20);
			this.txtMaxMissingPlates.TabIndex = 83;
			this.txtMaxMissingPlates.Text = "";
			this.txtMaxMissingPlates.Leave += new System.EventHandler(this.txtMaxMissingPlates_Leave);
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(152, 24);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(464, 20);
			this.configNameTextBox.TabIndex = 78;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// maxMissedPlatesLabel
			// 
			this.maxMissedPlatesLabel.Location = new System.Drawing.Point(408, 64);
			this.maxMissedPlatesLabel.Name = "maxMissedPlatesLabel";
			this.maxMissedPlatesLabel.Size = new System.Drawing.Size(112, 23);
			this.maxMissedPlatesLabel.TabIndex = 82;
			this.maxMissedPlatesLabel.Text = "Max missed plates:";
			// 
			// directionLabel
			// 
			this.directionLabel.Location = new System.Drawing.Point(16, 64);
			this.directionLabel.Name = "directionLabel";
			this.directionLabel.TabIndex = 81;
			this.directionLabel.Text = "Direction:";
			// 
			// cmbDirection
			// 
			this.cmbDirection.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbDirection.Location = new System.Drawing.Point(152, 64);
			this.cmbDirection.Name = "cmbDirection";
			this.cmbDirection.Size = new System.Drawing.Size(176, 21);
			this.cmbDirection.TabIndex = 80;
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(16, 24);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(128, 23);
			this.configNameLabel.TabIndex = 79;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// intercalibrationEditButton
			// 
			this.intercalibrationEditButton.Location = new System.Drawing.Point(544, 104);
			this.intercalibrationEditButton.Name = "intercalibrationEditButton";
			this.intercalibrationEditButton.TabIndex = 89;
			this.intercalibrationEditButton.Text = "View/edit";
			this.intercalibrationEditButton.Click += new System.EventHandler(this.intercalibrationEditButton_Click);
			// 
			// predscanEditButton
			// 
			this.predscanEditButton.Location = new System.Drawing.Point(544, 144);
			this.predscanEditButton.Name = "predscanEditButton";
			this.predscanEditButton.TabIndex = 88;
			this.predscanEditButton.Text = "View/edit";
			this.predscanEditButton.Click += new System.EventHandler(this.predscanEditButton_Click);
			// 
			// frmConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(632, 222);
			this.Controls.Add(this.intercalibrationEditButton);
			this.Controls.Add(this.predscanEditButton);
			this.Controls.Add(this.cmbIntercalibration);
			this.Controls.Add(this.cmbPredscan);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.txtMaxMissingPlates);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.maxMissedPlatesLabel);
			this.Controls.Add(this.directionLabel);
			this.Controls.Add(this.cmbDirection);
			this.Controls.Add(this.configNameLabel);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.btnCancel);
			this.Name = "frmConfig";
			this.Text = "View/edit SimpleScanbackDriver config";
			this.ResumeLayout(false);

		}
		#endregion

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			if (cmbDirection.SelectedIndex == 0) _progSettings.Direction = ScanDirection.Upstream;
			else if (cmbDirection.SelectedIndex == 1) _progSettings.Direction  = ScanDirection.Downstream;
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SimpleScanback3Settings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, _progSettings);
			sw.Flush();
			_settingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "SimpleScanback3Driver.exe", 2, 0, sw.ToString());
			Close();
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void cmbDirection_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			if (cmbDirection.SelectedIndex == 0) _direction = ScanDirection.Upstream;
			else if (cmbDirection.SelectedIndex == 1) _direction = ScanDirection.Downstream;
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		public bool ConfigNameIsValid()
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			return _currentConfigName.Trim() != _originalConfigName.Trim() & _currentConfigName.Trim() != ""; 
		}

		private bool IsFormFilled()
		{
			bool scanDirUndefined = (_progSettings.Direction != ScanDirection.Upstream) & (_progSettings.Direction != ScanDirection.Downstream); 
			return !(scanDirUndefined | _progSettings.IntercalibrationConfigId == 0 | _progSettings.PredictionScanConfigId == 0);
		}

		private void cmbIntercalibration_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.IntercalibrationConfigId = ((Utilities.ConfigItem)cmbIntercalibration.SelectedItem).Id;			
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void cmbPredscan_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.PredictionScanConfigId = ((Utilities.ConfigItem)cmbPredscan.SelectedItem).Id;			
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void txtMaxMissingPlates_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.MaxMissingPlates = Convert.ToInt32(txtMaxMissingPlates.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMaxMissingPlates.Focus();
			}
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void intercalibrationEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
			object IntercalibrationDriverExe = System.Activator.CreateInstanceFrom(path + @"\IntercalibrationDriver.exe", 
				"IntercalibrationDriver.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.IntercalibrationConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) IntercalibrationDriverExe.GetType().InvokeMember("GetConfig", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
				null, IntercalibrationDriverExe, new object[] {_progSettings.IntercalibrationConfigId, _connection}
				);
			if (newid != 0) _progSettings.IntercalibrationConfigId = newid;
			Utilities.FillComboBox(cmbIntercalibration, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='IntercalibrationDriver.exe'", _connection);
			Utilities.SelectId(cmbIntercalibration, _progSettings.IntercalibrationConfigId);
		}

		private void predscanEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
			object PredictionScan2DriverExe = System.Activator.CreateInstanceFrom(path + @"\PredictionScan2Driver.exe", 
				"PredictionScan2Driver.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.PredictionScanConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) PredictionScan2DriverExe.GetType().InvokeMember("GetConfig", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
				null, PredictionScan2DriverExe, new object[] {_progSettings.PredictionScanConfigId, _connection}
				);
			if (newid != 0) _progSettings.PredictionScanConfigId = newid;
			Utilities.FillComboBox(cmbPredscan, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='PredictionScan2Driver.exe'", _connection);
			Utilities.SelectId(cmbPredscan, _progSettings.PredictionScanConfigId);
		}

		
	}
}
