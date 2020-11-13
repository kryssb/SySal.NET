using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.TotalScanDriver
{
	/// <summary>
	/// Configuration Form.
	/// </summary>
	internal class frmConfig : System.Windows.Forms.Form
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.Button intercalibrationEditButton;
		private System.Windows.Forms.ComboBox cmbIntercalibration;
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Label directionLabel;
		private System.Windows.Forms.ComboBox cmbDirection;
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.Button areascanEditButton;
		private System.Windows.Forms.Label lblAreaScan;
		private System.Windows.Forms.Label lblIntercalibration;

		private string _originalConfigName = "";
		private ScanDirection _direction;

		private static long _settingsId;
		private long _intercalibrationSettingsId;
		private long _areascanSettingsId;		
		private SySal.OperaDb.OperaDbConnection _connection;
		private string _currentConfigName = "";
		private System.Windows.Forms.ComboBox cmbAreascan;
		private System.Windows.Forms.GroupBox grpVolumeCreation;
		private System.Windows.Forms.ComboBox cmbInputSource;
		private System.Windows.Forms.Label lblInputSource;
		private System.Windows.Forms.TextBox txtDownstreamPlates;
		private System.Windows.Forms.Label lblDownstreamPlates;
		private System.Windows.Forms.TextBox txtUpstreamPlates;
		private System.Windows.Forms.Label lblUpstreamPlates;
		private System.Windows.Forms.Label lblWidthFormula;
		private System.Windows.Forms.TextBox txtWidthFormula;
		private System.Windows.Forms.TextBox txtPrimarySlopeX;
		private System.Windows.Forms.TextBox txtPrimarySlopeY;
		private System.Windows.Forms.Label lblPrimarySlopeX;
		private System.Windows.Forms.Label lblPrimarySlopeY;

		private TotalScanSettings _progSettings;


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
						
			cmbDirection.Items.Add(ScanDirection.Upstream);			
			cmbDirection.Items.Add(ScanDirection.Downstream);	
			Utilities.FillComboBox(cmbIntercalibration, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='IntercalibrationDriver.exe'", _connection);
			Utilities.FillComboBox(cmbAreascan, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='AreaScan2Driver.exe'", _connection);

			if (settingsId != 0)
			{
				string settings = (string)(new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, _connection)).ExecuteScalar();								
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TotalScanSettings));
				_progSettings = (TotalScanSettings) xmls.Deserialize(new System.IO.StringReader(settings));
				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, _connection)).ExecuteScalar();
								
			}
			else
			{
				_progSettings = new TotalScanSettings();
				_progSettings.VolumeCreationMode = new VolumeCreation();
			}

			Utilities.SelectId(cmbIntercalibration, _progSettings.IntercalibrationConfigId);
			Utilities.SelectId(cmbAreascan, _progSettings.AreaScanConfigId);

			if (_progSettings.Direction == ScanDirection.Upstream) cmbDirection.SelectedIndex = 0;
			else if (_progSettings.Direction == ScanDirection.Downstream) cmbDirection.SelectedIndex = 1;
			switch (_progSettings.VolumeCreationMode.Source) 
			{
				case InputSource.ScanbackPath:
				cmbInputSource.SelectedIndex = 0;
				break;
				case InputSource.ScanbackPathFixedPrimarySlope:
				cmbInputSource.SelectedIndex = 1;
				break;
				case InputSource.VolumeTrack:
				cmbInputSource.SelectedIndex = 2;
				break;
			}
			txtDownstreamPlates.Text = _progSettings.VolumeCreationMode.DownstreamPlates.ToString();
			txtUpstreamPlates.Text = _progSettings.VolumeCreationMode.UpstreamPlates.ToString();
			txtPrimarySlopeX.Text = _progSettings.VolumeCreationMode.PrimarySlope.X.ToString();
			txtPrimarySlopeY.Text = _progSettings.VolumeCreationMode.PrimarySlope.Y.ToString();
			txtWidthFormula.Text = _progSettings.VolumeCreationMode.WidthFormula;
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
			this.intercalibrationEditButton = new System.Windows.Forms.Button();
			this.areascanEditButton = new System.Windows.Forms.Button();
			this.cmbIntercalibration = new System.Windows.Forms.ComboBox();
			this.cmbAreascan = new System.Windows.Forms.ComboBox();
			this.lblAreaScan = new System.Windows.Forms.Label();
			this.lblIntercalibration = new System.Windows.Forms.Label();
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.directionLabel = new System.Windows.Forms.Label();
			this.cmbDirection = new System.Windows.Forms.ComboBox();
			this.configNameLabel = new System.Windows.Forms.Label();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.grpVolumeCreation = new System.Windows.Forms.GroupBox();
			this.lblPrimarySlopeY = new System.Windows.Forms.Label();
			this.lblPrimarySlopeX = new System.Windows.Forms.Label();
			this.txtPrimarySlopeY = new System.Windows.Forms.TextBox();
			this.txtPrimarySlopeX = new System.Windows.Forms.TextBox();
			this.txtWidthFormula = new System.Windows.Forms.TextBox();
			this.lblWidthFormula = new System.Windows.Forms.Label();
			this.txtUpstreamPlates = new System.Windows.Forms.TextBox();
			this.lblUpstreamPlates = new System.Windows.Forms.Label();
			this.txtDownstreamPlates = new System.Windows.Forms.TextBox();
			this.lblDownstreamPlates = new System.Windows.Forms.Label();
			this.lblInputSource = new System.Windows.Forms.Label();
			this.cmbInputSource = new System.Windows.Forms.ComboBox();
			this.grpVolumeCreation.SuspendLayout();
			this.SuspendLayout();
			// 
			// intercalibrationEditButton
			// 
			this.intercalibrationEditButton.Location = new System.Drawing.Point(544, 248);
			this.intercalibrationEditButton.Name = "intercalibrationEditButton";
			this.intercalibrationEditButton.TabIndex = 103;
			this.intercalibrationEditButton.Text = "View/edit";
			this.intercalibrationEditButton.Click += new System.EventHandler(this.intercalibrationEditButton_Click);
			// 
			// areascanEditButton
			// 
			this.areascanEditButton.Location = new System.Drawing.Point(544, 288);
			this.areascanEditButton.Name = "areascanEditButton";
			this.areascanEditButton.TabIndex = 102;
			this.areascanEditButton.Text = "View/edit";
			this.areascanEditButton.Click += new System.EventHandler(this.areascanEditButton_Click);
			// 
			// cmbIntercalibration
			// 
			this.cmbIntercalibration.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbIntercalibration.Location = new System.Drawing.Point(152, 248);
			this.cmbIntercalibration.Name = "cmbIntercalibration";
			this.cmbIntercalibration.Size = new System.Drawing.Size(376, 21);
			this.cmbIntercalibration.TabIndex = 101;
			this.cmbIntercalibration.SelectedIndexChanged += new System.EventHandler(this.cmbIntercalibration_SelectedIndexChanged);
			// 
			// cmbAreascan
			// 
			this.cmbAreascan.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbAreascan.Location = new System.Drawing.Point(152, 288);
			this.cmbAreascan.Name = "cmbAreascan";
			this.cmbAreascan.Size = new System.Drawing.Size(376, 21);
			this.cmbAreascan.TabIndex = 100;
			this.cmbAreascan.SelectedIndexChanged += new System.EventHandler(this.cmbAreascan_SelectedIndexChanged);
			// 
			// lblAreaScan
			// 
			this.lblAreaScan.Location = new System.Drawing.Point(16, 288);
			this.lblAreaScan.Name = "lblAreaScan";
			this.lblAreaScan.Size = new System.Drawing.Size(128, 23);
			this.lblAreaScan.TabIndex = 99;
			this.lblAreaScan.Text = "AreaScan2 settings:";
			// 
			// lblIntercalibration
			// 
			this.lblIntercalibration.Location = new System.Drawing.Point(16, 248);
			this.lblIntercalibration.Name = "lblIntercalibration";
			this.lblIntercalibration.Size = new System.Drawing.Size(128, 16);
			this.lblIntercalibration.TabIndex = 98;
			this.lblIntercalibration.Text = "Intercalibration settings:";
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(151, 16);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(464, 20);
			this.configNameTextBox.TabIndex = 92;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// directionLabel
			// 
			this.directionLabel.Location = new System.Drawing.Point(15, 56);
			this.directionLabel.Name = "directionLabel";
			this.directionLabel.TabIndex = 95;
			this.directionLabel.Text = "Direction:";
			// 
			// cmbDirection
			// 
			this.cmbDirection.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbDirection.Location = new System.Drawing.Point(151, 56);
			this.cmbDirection.Name = "cmbDirection";
			this.cmbDirection.Size = new System.Drawing.Size(176, 21);
			this.cmbDirection.TabIndex = 94;
			this.cmbDirection.SelectedIndexChanged += new System.EventHandler(this.cmbDirection_SelectedIndexChanged);
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(15, 16);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(128, 23);
			this.configNameLabel.TabIndex = 93;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(16, 320);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 91;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(544, 320);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 90;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// grpVolumeCreation
			// 
			this.grpVolumeCreation.Controls.Add(this.lblPrimarySlopeY);
			this.grpVolumeCreation.Controls.Add(this.lblPrimarySlopeX);
			this.grpVolumeCreation.Controls.Add(this.txtPrimarySlopeY);
			this.grpVolumeCreation.Controls.Add(this.txtPrimarySlopeX);
			this.grpVolumeCreation.Controls.Add(this.txtWidthFormula);
			this.grpVolumeCreation.Controls.Add(this.lblWidthFormula);
			this.grpVolumeCreation.Controls.Add(this.txtUpstreamPlates);
			this.grpVolumeCreation.Controls.Add(this.lblUpstreamPlates);
			this.grpVolumeCreation.Controls.Add(this.txtDownstreamPlates);
			this.grpVolumeCreation.Controls.Add(this.lblDownstreamPlates);
			this.grpVolumeCreation.Controls.Add(this.lblInputSource);
			this.grpVolumeCreation.Controls.Add(this.cmbInputSource);
			this.grpVolumeCreation.Location = new System.Drawing.Point(8, 88);
			this.grpVolumeCreation.Name = "grpVolumeCreation";
			this.grpVolumeCreation.Size = new System.Drawing.Size(616, 144);
			this.grpVolumeCreation.TabIndex = 104;
			this.grpVolumeCreation.TabStop = false;
			this.grpVolumeCreation.Text = "Volume creation";
			// 
			// lblPrimarySlopeY
			// 
			this.lblPrimarySlopeY.Location = new System.Drawing.Point(336, 112);
			this.lblPrimarySlopeY.Name = "lblPrimarySlopeY";
			this.lblPrimarySlopeY.Size = new System.Drawing.Size(112, 23);
			this.lblPrimarySlopeY.TabIndex = 107;
			this.lblPrimarySlopeY.Text = "Primary Slope Y:";
			// 
			// lblPrimarySlopeX
			// 
			this.lblPrimarySlopeX.Location = new System.Drawing.Point(16, 112);
			this.lblPrimarySlopeX.Name = "lblPrimarySlopeX";
			this.lblPrimarySlopeX.Size = new System.Drawing.Size(112, 23);
			this.lblPrimarySlopeX.TabIndex = 106;
			this.lblPrimarySlopeX.Text = "Primary slope X:";
			// 
			// txtPrimarySlopeY
			// 
			this.txtPrimarySlopeY.Location = new System.Drawing.Point(488, 112);
			this.txtPrimarySlopeY.Name = "txtPrimarySlopeY";
			this.txtPrimarySlopeY.Size = new System.Drawing.Size(64, 20);
			this.txtPrimarySlopeY.TabIndex = 105;
			this.txtPrimarySlopeY.Text = "";
			this.txtPrimarySlopeY.Leave += new System.EventHandler(this.txtPrimarySlopeY_Leave);
			// 
			// txtPrimarySlopeX
			// 
			this.txtPrimarySlopeX.Location = new System.Drawing.Point(160, 112);
			this.txtPrimarySlopeX.Name = "txtPrimarySlopeX";
			this.txtPrimarySlopeX.Size = new System.Drawing.Size(64, 20);
			this.txtPrimarySlopeX.TabIndex = 104;
			this.txtPrimarySlopeX.Text = "";
			this.txtPrimarySlopeX.Leave += new System.EventHandler(this.txtPrimarySlopeX_Leave);
			// 
			// txtWidthFormula
			// 
			this.txtWidthFormula.Location = new System.Drawing.Point(224, 80);
			this.txtWidthFormula.Name = "txtWidthFormula";
			this.txtWidthFormula.Size = new System.Drawing.Size(328, 20);
			this.txtWidthFormula.TabIndex = 103;
			this.txtWidthFormula.Text = "";
			this.txtWidthFormula.Leave += new System.EventHandler(this.txtWidthFormula_Leave);
			// 
			// lblWidthFormula
			// 
			this.lblWidthFormula.Location = new System.Drawing.Point(112, 80);
			this.lblWidthFormula.Name = "lblWidthFormula";
			this.lblWidthFormula.TabIndex = 102;
			this.lblWidthFormula.Text = "Width formula:";
			// 
			// txtUpstreamPlates
			// 
			this.txtUpstreamPlates.Location = new System.Drawing.Point(488, 48);
			this.txtUpstreamPlates.Name = "txtUpstreamPlates";
			this.txtUpstreamPlates.Size = new System.Drawing.Size(64, 20);
			this.txtUpstreamPlates.TabIndex = 101;
			this.txtUpstreamPlates.Text = "";
			this.txtUpstreamPlates.Leave += new System.EventHandler(this.txtUpstreamPlates_Leave);
			// 
			// lblUpstreamPlates
			// 
			this.lblUpstreamPlates.Location = new System.Drawing.Point(336, 48);
			this.lblUpstreamPlates.Name = "lblUpstreamPlates";
			this.lblUpstreamPlates.Size = new System.Drawing.Size(112, 23);
			this.lblUpstreamPlates.TabIndex = 100;
			this.lblUpstreamPlates.Text = "Upstream plates:";
			// 
			// txtDownstreamPlates
			// 
			this.txtDownstreamPlates.Location = new System.Drawing.Point(160, 48);
			this.txtDownstreamPlates.Name = "txtDownstreamPlates";
			this.txtDownstreamPlates.Size = new System.Drawing.Size(64, 20);
			this.txtDownstreamPlates.TabIndex = 99;
			this.txtDownstreamPlates.Text = "";
			this.txtDownstreamPlates.Leave += new System.EventHandler(this.txtDownstreamPlates_Leave);
			// 
			// lblDownstreamPlates
			// 
			this.lblDownstreamPlates.Location = new System.Drawing.Point(16, 48);
			this.lblDownstreamPlates.Name = "lblDownstreamPlates";
			this.lblDownstreamPlates.Size = new System.Drawing.Size(112, 23);
			this.lblDownstreamPlates.TabIndex = 98;
			this.lblDownstreamPlates.Text = "Downstream plates:";
			// 
			// lblInputSource
			// 
			this.lblInputSource.Location = new System.Drawing.Point(136, 16);
			this.lblInputSource.Name = "lblInputSource";
			this.lblInputSource.Size = new System.Drawing.Size(80, 23);
			this.lblInputSource.TabIndex = 96;
			this.lblInputSource.Text = "InputSource:";
			// 
			// cmbInputSource
			// 
			this.cmbInputSource.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbInputSource.Items.AddRange(new object[] {
																"Scanback path",
																"Scanback path, fixed primary slope",
																"Volume track"});
			this.cmbInputSource.Location = new System.Drawing.Point(224, 16);
			this.cmbInputSource.Name = "cmbInputSource";
			this.cmbInputSource.Size = new System.Drawing.Size(328, 21);
			this.cmbInputSource.TabIndex = 95;
			this.cmbInputSource.SelectedIndexChanged += new System.EventHandler(this.cmbInputSource_SelectedIndexChanged);
			// 
			// frmConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(632, 350);
			this.Controls.Add(this.grpVolumeCreation);
			this.Controls.Add(this.intercalibrationEditButton);
			this.Controls.Add(this.areascanEditButton);
			this.Controls.Add(this.cmbIntercalibration);
			this.Controls.Add(this.cmbAreascan);
			this.Controls.Add(this.lblAreaScan);
			this.Controls.Add(this.lblIntercalibration);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.directionLabel);
			this.Controls.Add(this.cmbDirection);
			this.Controls.Add(this.configNameLabel);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.btnCancel);
			this.Name = "frmConfig";
			this.Text = "TotalScan configuration";
			this.grpVolumeCreation.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

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

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void areascanEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\'));
			object AreaScan2DriverExe = System.Activator.CreateInstanceFrom(path + @"\AreaScan2Driver.exe", 
				"AreaScan2Driver.frmAreaScan2Config", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.AreaScanConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) AreaScan2DriverExe.GetType().InvokeMember("GetConfig", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
				null, AreaScan2DriverExe, new object[] {_progSettings.AreaScanConfigId, _connection}
				);
			if (newid != 0) _progSettings.AreaScanConfigId = newid;
			Utilities.FillComboBox(cmbAreascan, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='AreaScan2Driver.exe'", _connection);
			Utilities.SelectId(cmbAreascan, _progSettings.AreaScanConfigId);
		}

		private void cmbIntercalibration_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.IntercalibrationConfigId = ((Utilities.ConfigItem)cmbIntercalibration.SelectedItem).Id;			
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void cmbAreascan_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.AreaScanConfigId = ((Utilities.ConfigItem)cmbAreascan.SelectedItem).Id;			
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		public bool ConfigNameIsValid()
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			return _currentConfigName.Trim() != _originalConfigName.Trim() & _currentConfigName.Trim() != ""; 
		}

		private bool IsFormFilled()
		{
			bool scanDirUndefined = (_progSettings.Direction != ScanDirection.Upstream) & (_progSettings.Direction != ScanDirection.Downstream); 
			bool sourceUndefined = (_progSettings.VolumeCreationMode.Source != InputSource.ScanbackPath) & (_progSettings.VolumeCreationMode.Source != InputSource.ScanbackPathFixedPrimarySlope)
				& (_progSettings.VolumeCreationMode.Source != InputSource.VolumeTrack);
			return !(scanDirUndefined | sourceUndefined | 
				_progSettings.IntercalibrationConfigId == 0 | _progSettings.AreaScanConfigId == 0);
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			if (cmbDirection.SelectedIndex == 0) _progSettings.Direction = ScanDirection.Upstream;
			else if (cmbDirection.SelectedIndex == 1) _progSettings.Direction  = ScanDirection.Downstream;
			switch (cmbInputSource.SelectedIndex)
			{
				case 0:
					_progSettings.VolumeCreationMode.Source = InputSource.ScanbackPath;
					break;					
				case 1:
					_progSettings.VolumeCreationMode.Source = InputSource.ScanbackPathFixedPrimarySlope;
					break;
				case 2:
					_progSettings.VolumeCreationMode.Source = InputSource.VolumeTrack;
					break;				
			}
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TotalScanSettings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, _progSettings);
			sw.Flush();
			_settingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "TotalScanDriver.exe", 2, 0, sw.ToString());
			Close();		
		}

		private void cmbInputSource_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			switch (cmbInputSource.SelectedIndex)
			{
				case 0:
					_progSettings.VolumeCreationMode.Source = InputSource.ScanbackPath;
					break;					
				case 1:
					_progSettings.VolumeCreationMode.Source = InputSource.ScanbackPathFixedPrimarySlope;
					break;
				case 2:
					_progSettings.VolumeCreationMode.Source = InputSource.VolumeTrack;
					break;				
			}			
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void txtDownstreamPlates_Leave(object sender, System.EventArgs e)
		{			
			try
			{
				_progSettings.VolumeCreationMode.DownstreamPlates = Convert.ToUInt16(txtDownstreamPlates.Text);				
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtDownstreamPlates.Focus();
			}
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void txtUpstreamPlates_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.VolumeCreationMode.UpstreamPlates = Convert.ToUInt16(txtUpstreamPlates.Text);				
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtUpstreamPlates.Focus();
			}
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void txtWidthFormula_Leave(object sender, System.EventArgs e)
		{			
			_progSettings.VolumeCreationMode.WidthFormula = txtWidthFormula.Text;					
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void txtPrimarySlopeX_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.VolumeCreationMode.PrimarySlope.X = Convert.ToDouble(txtPrimarySlopeX.Text);				
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtPrimarySlopeX.Focus();
			}
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void txtPrimarySlopeY_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.VolumeCreationMode.PrimarySlope.Y = Convert.ToDouble(txtPrimarySlopeY.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtPrimarySlopeY.Focus();
			}			
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void cmbDirection_SelectedIndexChanged(object sender, System.EventArgs e)
		{
		
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}




	}
}
