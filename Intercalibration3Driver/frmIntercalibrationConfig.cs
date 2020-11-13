using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.Intercalibration3Driver
{
	/// <summary>
	/// Summary description for frmConfig.
	/// </summary>
	internal class frmConfig : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox zoneSizeTextBox;
		private System.Windows.Forms.Label zoneSizeLabel;
		private System.Windows.Forms.Button qualityEditButton;
		private System.Windows.Forms.ComboBox scanningComboBox;
		private System.Windows.Forms.ComboBox qualityComboBox;
		private System.Windows.Forms.TextBox slopeTolTextBox;
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Label posToleranceLabel;
		private System.Windows.Forms.Label slopeTolLabel;
		private System.Windows.Forms.TextBox posTolTextBox;
		private System.Windows.Forms.TextBox minMatchesTextBox;
		private System.Windows.Forms.Label minMatchesLabel;
		private System.Windows.Forms.Label maxOffsetLabel;
		private System.Windows.Forms.TextBox maxOffsetTextBox;
		private System.Windows.Forms.Label yZoneDistanceLabel;
		private System.Windows.Forms.TextBox xZoneDistanceTextBox;
		private System.Windows.Forms.Label xZoneDistanceLabel;
		private System.Windows.Forms.TextBox yZoneDistanceTextBox;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.ComboBox linkingComboBox;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.Button linkEditButton;
		private System.Windows.Forms.Button scanEditButton;
		private System.Windows.Forms.Button btnCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private Intercalibration3Settings _progSettings;
		private static long _settingsId;
		private long _scanningConfigId;
		private long _linkingConfigId;
		private long _qualityCutId;		
		private System.Windows.Forms.Button btnCreate;
		private string _currentConfigName = "";
		private string _originalConfigName = "";
		private SySal.OperaDb.OperaDbConnection _connection;
		//private Intercalibration3Settings _progSettings;

		public static long GetConfig(long settingsId, SySal.OperaDb.OperaDbConnection conn)
		{
			new frmConfig(settingsId, conn).ShowDialog();
			return _settingsId;
		}
		public frmConfig()
		{
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
			Utilities.FillComboBox(linkingComboBox, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE 'BatchLink%.exe'", conn);
			Utilities.FillComboBox(scanningComboBox, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE 'ScanServer%.exe'", conn);
			Utilities.FillComboBox(qualityComboBox, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE '%Sel.exe'", conn);
			
			if (settingsId != 0)
			{
				string settings = (string)(new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
				
				
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Intercalibration3Settings));
				_progSettings = (Intercalibration3Settings) xmls.Deserialize(new System.IO.StringReader(settings));

				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
				
				/*linkingComboBox.SelectedValue = _linkingConfigId;
				scanningComboBox.SelectedValue = _scanningConfigId;
				qualityComboBox.SelectedValue = _qualityCutId;*/
			}
			else
			{
				_progSettings = new Intercalibration3Settings();
			}
			posTolTextBox.Text = _progSettings.PositionTolerance.ToString();
			slopeTolTextBox.Text = _progSettings.SlopeTolerance.ToString();
			minMatchesTextBox.Text = _progSettings.MinMatches.ToString();
			maxOffsetTextBox.Text = _progSettings.MaxOffset.ToString();
			xZoneDistanceTextBox.Text = _progSettings.XZoneDistance.ToString();
			yZoneDistanceTextBox.Text = _progSettings.YZoneDistance.ToString();
			zoneSizeTextBox.Text = _progSettings.ZoneSize.ToString();
			_linkingConfigId = _progSettings.LinkConfigId;				
			_scanningConfigId = _progSettings.ScanningConfigId;
			_qualityCutId = _progSettings.QualityCutId;
			/*for (int i=0; i<scanningComboBox.Items.Count; i++)
				{
					if ( ((Utilities.ConfigItem)scanningComboBox.Items[i]).Id == _scanningConfigId ) 
					{
						scanningComboBox.SelectedIndex = i;
						break;
					}
				}*/
			Utilities.SelectId(scanningComboBox, _scanningConfigId);
			Utilities.SelectId(linkingComboBox, _linkingConfigId);
			Utilities.SelectId(qualityComboBox, _qualityCutId);

			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		
		public bool ConfigNameIsValid()
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			return _currentConfigName.Trim() != _originalConfigName.Trim() & _currentConfigName.Trim() != ""; 
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
			this.zoneSizeTextBox = new System.Windows.Forms.TextBox();
			this.zoneSizeLabel = new System.Windows.Forms.Label();
			this.qualityEditButton = new System.Windows.Forms.Button();
			this.scanningComboBox = new System.Windows.Forms.ComboBox();
			this.qualityComboBox = new System.Windows.Forms.ComboBox();
			this.slopeTolTextBox = new System.Windows.Forms.TextBox();
			this.configNameLabel = new System.Windows.Forms.Label();
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.posToleranceLabel = new System.Windows.Forms.Label();
			this.slopeTolLabel = new System.Windows.Forms.Label();
			this.posTolTextBox = new System.Windows.Forms.TextBox();
			this.minMatchesTextBox = new System.Windows.Forms.TextBox();
			this.minMatchesLabel = new System.Windows.Forms.Label();
			this.maxOffsetLabel = new System.Windows.Forms.Label();
			this.maxOffsetTextBox = new System.Windows.Forms.TextBox();
			this.yZoneDistanceLabel = new System.Windows.Forms.Label();
			this.xZoneDistanceTextBox = new System.Windows.Forms.TextBox();
			this.xZoneDistanceLabel = new System.Windows.Forms.Label();
			this.yZoneDistanceTextBox = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.label8 = new System.Windows.Forms.Label();
			this.linkingComboBox = new System.Windows.Forms.ComboBox();
			this.label9 = new System.Windows.Forms.Label();
			this.linkEditButton = new System.Windows.Forms.Button();
			this.scanEditButton = new System.Windows.Forms.Button();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// zoneSizeTextBox
			// 
			this.zoneSizeTextBox.Location = new System.Drawing.Point(280, 168);
			this.zoneSizeTextBox.Name = "zoneSizeTextBox";
			this.zoneSizeTextBox.TabIndex = 61;
			this.zoneSizeTextBox.Text = "";
			this.zoneSizeTextBox.Leave += new System.EventHandler(this.zoneSizeTextBox_Leave);
			// 
			// zoneSizeLabel
			// 
			this.zoneSizeLabel.Location = new System.Drawing.Point(208, 168);
			this.zoneSizeLabel.Name = "zoneSizeLabel";
			this.zoneSizeLabel.Size = new System.Drawing.Size(56, 23);
			this.zoneSizeLabel.TabIndex = 60;
			this.zoneSizeLabel.Text = "Zone size:";
			// 
			// qualityEditButton
			// 
			this.qualityEditButton.Location = new System.Drawing.Point(464, 216);
			this.qualityEditButton.Name = "qualityEditButton";
			this.qualityEditButton.TabIndex = 57;
			this.qualityEditButton.Text = "View/edit";
			this.qualityEditButton.Click += new System.EventHandler(this.qualityEditButton_Click);
			// 
			// scanningComboBox
			// 
			this.scanningComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.scanningComboBox.Location = new System.Drawing.Point(112, 280);
			this.scanningComboBox.Name = "scanningComboBox";
			this.scanningComboBox.Size = new System.Drawing.Size(344, 21);
			this.scanningComboBox.TabIndex = 54;
			this.scanningComboBox.SelectedIndexChanged += new System.EventHandler(this.scanningComboBox_SelectedIndexChanged);
			// 
			// qualityComboBox
			// 
			this.qualityComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.qualityComboBox.Location = new System.Drawing.Point(112, 216);
			this.qualityComboBox.Name = "qualityComboBox";
			this.qualityComboBox.Size = new System.Drawing.Size(344, 21);
			this.qualityComboBox.TabIndex = 52;
			this.qualityComboBox.SelectedIndexChanged += new System.EventHandler(this.qualityComboBox_SelectedIndexChanged);
			// 
			// slopeTolTextBox
			// 
			this.slopeTolTextBox.Location = new System.Drawing.Point(448, 64);
			this.slopeTolTextBox.Name = "slopeTolTextBox";
			this.slopeTolTextBox.TabIndex = 51;
			this.slopeTolTextBox.Text = "";
			this.slopeTolTextBox.Leave += new System.EventHandler(this.slopeTolTextBox_TextChanged);
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(40, 24);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(128, 23);
			this.configNameLabel.TabIndex = 44;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(176, 16);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(376, 20);
			this.configNameTextBox.TabIndex = 35;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// posToleranceLabel
			// 
			this.posToleranceLabel.Location = new System.Drawing.Point(40, 64);
			this.posToleranceLabel.Name = "posToleranceLabel";
			this.posToleranceLabel.Size = new System.Drawing.Size(128, 23);
			this.posToleranceLabel.TabIndex = 45;
			this.posToleranceLabel.Text = "Position tolerance:";
			// 
			// slopeTolLabel
			// 
			this.slopeTolLabel.Location = new System.Drawing.Point(336, 64);
			this.slopeTolLabel.Name = "slopeTolLabel";
			this.slopeTolLabel.Size = new System.Drawing.Size(128, 23);
			this.slopeTolLabel.TabIndex = 38;
			this.slopeTolLabel.Text = "Slope tolerance:";
			// 
			// posTolTextBox
			// 
			this.posTolTextBox.Location = new System.Drawing.Point(176, 64);
			this.posTolTextBox.Name = "posTolTextBox";
			this.posTolTextBox.TabIndex = 50;
			this.posTolTextBox.Text = "";
			this.posTolTextBox.Leave += new System.EventHandler(this.posTolTextBox_Leave);
			// 
			// minMatchesTextBox
			// 
			this.minMatchesTextBox.Location = new System.Drawing.Point(176, 96);
			this.minMatchesTextBox.Name = "minMatchesTextBox";
			this.minMatchesTextBox.TabIndex = 49;
			this.minMatchesTextBox.Text = "";
			this.minMatchesTextBox.Leave += new System.EventHandler(this.minMatchesTextBox_Leave);
			// 
			// minMatchesLabel
			// 
			this.minMatchesLabel.Location = new System.Drawing.Point(40, 96);
			this.minMatchesLabel.Name = "minMatchesLabel";
			this.minMatchesLabel.Size = new System.Drawing.Size(128, 23);
			this.minMatchesLabel.TabIndex = 37;
			this.minMatchesLabel.Text = "Min matches:";
			// 
			// maxOffsetLabel
			// 
			this.maxOffsetLabel.Location = new System.Drawing.Point(336, 96);
			this.maxOffsetLabel.Name = "maxOffsetLabel";
			this.maxOffsetLabel.Size = new System.Drawing.Size(80, 23);
			this.maxOffsetLabel.TabIndex = 36;
			this.maxOffsetLabel.Text = "Max offset:";
			// 
			// maxOffsetTextBox
			// 
			this.maxOffsetTextBox.Location = new System.Drawing.Point(448, 96);
			this.maxOffsetTextBox.Name = "maxOffsetTextBox";
			this.maxOffsetTextBox.TabIndex = 46;
			this.maxOffsetTextBox.Text = "";
			this.maxOffsetTextBox.Leave += new System.EventHandler(this.maxOffsetTextBox_Leave);
			// 
			// yZoneDistanceLabel
			// 
			this.yZoneDistanceLabel.Location = new System.Drawing.Point(336, 128);
			this.yZoneDistanceLabel.Name = "yZoneDistanceLabel";
			this.yZoneDistanceLabel.Size = new System.Drawing.Size(88, 23);
			this.yZoneDistanceLabel.TabIndex = 39;
			this.yZoneDistanceLabel.Text = "Y zone distance:";
			// 
			// xZoneDistanceTextBox
			// 
			this.xZoneDistanceTextBox.Location = new System.Drawing.Point(176, 128);
			this.xZoneDistanceTextBox.Name = "xZoneDistanceTextBox";
			this.xZoneDistanceTextBox.TabIndex = 48;
			this.xZoneDistanceTextBox.Text = "";
			this.xZoneDistanceTextBox.Leave += new System.EventHandler(this.xZoneDistanceTextBox_Leave);
			// 
			// xZoneDistanceLabel
			// 
			this.xZoneDistanceLabel.Location = new System.Drawing.Point(40, 128);
			this.xZoneDistanceLabel.Name = "xZoneDistanceLabel";
			this.xZoneDistanceLabel.Size = new System.Drawing.Size(128, 23);
			this.xZoneDistanceLabel.TabIndex = 42;
			this.xZoneDistanceLabel.Text = "X zone distance:";
			// 
			// yZoneDistanceTextBox
			// 
			this.yZoneDistanceTextBox.Location = new System.Drawing.Point(448, 128);
			this.yZoneDistanceTextBox.Name = "yZoneDistanceTextBox";
			this.yZoneDistanceTextBox.TabIndex = 47;
			this.yZoneDistanceTextBox.Text = "";
			this.yZoneDistanceTextBox.Leave += new System.EventHandler(this.yZoneDistanceTextBox_Leave);
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(8, 216);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(88, 23);
			this.label7.TabIndex = 43;
			this.label7.Text = "Quality cut:";
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(8, 280);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(104, 23);
			this.label8.TabIndex = 40;
			this.label8.Text = "Scanning Config:";
			// 
			// linkingComboBox
			// 
			this.linkingComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.linkingComboBox.Location = new System.Drawing.Point(112, 248);
			this.linkingComboBox.Name = "linkingComboBox";
			this.linkingComboBox.Size = new System.Drawing.Size(344, 21);
			this.linkingComboBox.TabIndex = 53;
			this.linkingComboBox.SelectedIndexChanged += new System.EventHandler(this.linkingComboBox_SelectedIndexChanged);
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(8, 248);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(88, 23);
			this.label9.TabIndex = 41;
			this.label9.Text = "Linking Config:";
			// 
			// linkEditButton
			// 
			this.linkEditButton.Location = new System.Drawing.Point(464, 248);
			this.linkEditButton.Name = "linkEditButton";
			this.linkEditButton.TabIndex = 55;
			this.linkEditButton.Text = "View/edit";
			this.linkEditButton.Click += new System.EventHandler(this.linkEditButton_Click);
			// 
			// scanEditButton
			// 
			this.scanEditButton.Location = new System.Drawing.Point(464, 280);
			this.scanEditButton.Name = "scanEditButton";
			this.scanEditButton.TabIndex = 56;
			this.scanEditButton.Text = "View/edit";
			this.scanEditButton.Click += new System.EventHandler(this.scanEditButton_Click);
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(16, 336);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 63;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(464, 336);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 62;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// frmConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(560, 374);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.zoneSizeTextBox);
			this.Controls.Add(this.zoneSizeLabel);
			this.Controls.Add(this.qualityEditButton);
			this.Controls.Add(this.scanningComboBox);
			this.Controls.Add(this.qualityComboBox);
			this.Controls.Add(this.slopeTolTextBox);
			this.Controls.Add(this.configNameLabel);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.posToleranceLabel);
			this.Controls.Add(this.slopeTolLabel);
			this.Controls.Add(this.posTolTextBox);
			this.Controls.Add(this.minMatchesTextBox);
			this.Controls.Add(this.minMatchesLabel);
			this.Controls.Add(this.maxOffsetLabel);
			this.Controls.Add(this.maxOffsetTextBox);
			this.Controls.Add(this.yZoneDistanceLabel);
			this.Controls.Add(this.xZoneDistanceTextBox);
			this.Controls.Add(this.xZoneDistanceLabel);
			this.Controls.Add(this.yZoneDistanceTextBox);
			this.Controls.Add(this.label7);
			this.Controls.Add(this.label8);
			this.Controls.Add(this.linkingComboBox);
			this.Controls.Add(this.label9);
			this.Controls.Add(this.linkEditButton);
			this.Controls.Add(this.scanEditButton);
			this.Name = "frmConfig";
			this.Text = "View/edit intercalibration config";
			this.ResumeLayout(false);

		}
		#endregion

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Intercalibration3Settings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, _progSettings);
			sw.Flush();
			_settingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "Intercalibration3Driver.exe", 1, 1, sw.ToString());
			Close();
		}

		private void posTolTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.PositionTolerance = Convert.ToDouble(posTolTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				posTolTextBox.Focus();
			}
		}

		private void minMatchesTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.MinMatches = Convert.ToInt16(minMatchesTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				minMatchesTextBox.Focus();
			}

		}

		private void xZoneDistanceTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.XZoneDistance = Convert.ToDouble(xZoneDistanceTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				xZoneDistanceTextBox.Focus();
			}
		}

		private void slopeTolTextBox_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.SlopeTolerance = Convert.ToDouble(slopeTolTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				slopeTolTextBox.Focus();
			}

		}

		private void maxOffsetTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.MaxOffset = Convert.ToDouble(maxOffsetTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				maxOffsetTextBox.Focus();
			}

		}

		private void yZoneDistanceTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.YZoneDistance = Convert.ToDouble(yZoneDistanceTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				yZoneDistanceTextBox.Focus();
			}
		}

		private void zoneSizeTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.ZoneSize = Convert.ToDouble(zoneSizeTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				zoneSizeTextBox.Focus();
			}		
		}

		private void linkEditButton_Click(object sender, System.EventArgs e)
		{
			
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
			object LinkingConfigExe = System.Activator.CreateInstanceFrom(path + @"\LinkingConfig.exe", 
				"LinkingConfig.frmLinkingConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.LinkConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) LinkingConfigExe.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, 
				null, LinkingConfigExe, new object[] {_progSettings.LinkConfigId, _connection}
				);
			if (newid != 0) _progSettings.LinkConfigId = newid;
			Utilities.FillComboBox(linkingComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='BatchLink.exe'", _connection);
			Utilities.SelectId(linkingComboBox, _progSettings.LinkConfigId);
			//linkingComboBox.SelectedValue = _progSettings.LinkConfigId;

		}

		private void qualityEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
			object QualityCutExe = System.Activator.CreateInstanceFrom(path + @"\QualityCutConfig.exe", 
				"QualityCutConfig.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.QualityCutId, _connection}, null, null, null).Unwrap();		
			long newid = (long) QualityCutExe.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, 
				null, QualityCutExe, new object[] {_progSettings.QualityCutId, _connection}
				);
			if (newid != 0) _progSettings.QualityCutId = newid;
			Utilities.FillComboBox(qualityComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE '%Sel.exe'", _connection);
			Utilities.SelectId(qualityComboBox, _progSettings.QualityCutId);
		}

		private void scanEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
			object ScanningConfigGui = System.Activator.CreateInstanceFrom(path + @"\ScanningConfigGui.exe", 
				"ScanningConfigGui.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {_progSettings.ScanningConfigId, _connection}, null, null, null).Unwrap();		
			//object[] test = {ref long id};
			long newid = (long) ScanningConfigGui.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
				null, ScanningConfigGui, new object[] {_progSettings.ScanningConfigId, _connection}
				);
			//MessageBox.Show("Got " + newid);
			if (newid != 0) _progSettings.ScanningConfigId = newid;
			Utilities.FillComboBox(scanningComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE 'ScanServer%.exe'", _connection);
			Utilities.SelectId(scanningComboBox, _progSettings.ScanningConfigId);
		}


		/*private void button2_Click(object sender, System.EventArgs e)
		{
		
		}*/

	

		private bool IsFormFilled()
		{
			return !(_progSettings.LinkConfigId == 0 | _progSettings.QualityCutId == 0 | _progSettings.ScanningConfigId == 0);
		}

		private void qualityComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.QualityCutId = ((Utilities.ConfigItem)qualityComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void linkingComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.LinkConfigId = ((Utilities.ConfigItem)linkingComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void scanningComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.ScanningConfigId = ((Utilities.ConfigItem)scanningComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		
		
	}
}
