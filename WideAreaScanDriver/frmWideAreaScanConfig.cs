using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.WideAreaScanDriver
{
	/// <summary>
	/// Summary description for frmWideAreaScanConfig.
	/// </summary>
	public class frmWideAreaScanConfig : System.Windows.Forms.Form
	{
		public static long SettingsId;
		private SySal.OperaDb.OperaDbConnection _connection;
		private string _currentConfigName = "";
		private string _originalConfigName = "";
		private WideAreaScanSettings ProgSettings;
		private long _scanningConfigId;
		private long _linkingConfigId;
		private long _qualityCutId;


		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.ComboBox qualityComboBox;
		private System.Windows.Forms.Button qualityEditButton;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Button linkEditButton;
		private System.Windows.Forms.Button scanEditButton;
		private System.Windows.Forms.ComboBox scanningComboBox;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.ComboBox linkingComboBox;
		private System.Windows.Forms.Label label9;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public static long GetConfig(long settingsId, SySal.OperaDb.OperaDbConnection conn)
		{
			new frmWideAreaScanConfig(settingsId, conn).ShowDialog();
			return SettingsId;
		}

		public frmWideAreaScanConfig(long settingsId, SySal.OperaDb.OperaDbConnection conn)
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
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(WideAreaScanSettings));
				ProgSettings = (WideAreaScanSettings) xmls.Deserialize(new System.IO.StringReader(settings));
				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
				
				/*linkingComboBox.SelectedValue = _linkingConfigId;
				scanningComboBox.SelectedValue = _scanningConfigId;
				qualityComboBox.SelectedValue = _qualityCutId;*/
			}
			else
			{
				ProgSettings = new WideAreaScanSettings();				
			}
//Luillo			txtInitQuery.Text = ProgSettings.InitSql;	
		
			_linkingConfigId = ProgSettings.LinkConfigId;				
			_scanningConfigId = ProgSettings.ScanningConfigId;
			_qualityCutId = ProgSettings.QualityCutId;

			Utilities.SelectId(scanningComboBox, _scanningConfigId);
			Utilities.SelectId(linkingComboBox, _linkingConfigId);		
			Utilities.SelectId(qualityComboBox, _qualityCutId);
		}

		public bool ConfigNameIsValid()
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			return _currentConfigName.Trim() != _originalConfigName.Trim() & _currentConfigName.Trim() != ""; 
		}

		private bool IsFormFilled()
		{
			return !(ProgSettings.LinkConfigId == 0 | ProgSettings.QualityCutId == 0 | ProgSettings.ScanningConfigId == 0);
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
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.configNameLabel = new System.Windows.Forms.Label();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.qualityComboBox = new System.Windows.Forms.ComboBox();
			this.qualityEditButton = new System.Windows.Forms.Button();
			this.label7 = new System.Windows.Forms.Label();
			this.linkEditButton = new System.Windows.Forms.Button();
			this.scanEditButton = new System.Windows.Forms.Button();
			this.scanningComboBox = new System.Windows.Forms.ComboBox();
			this.label8 = new System.Windows.Forms.Label();
			this.linkingComboBox = new System.Windows.Forms.ComboBox();
			this.label9 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(120, 16);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(392, 20);
			this.configNameTextBox.TabIndex = 116;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(16, 16);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(112, 23);
			this.configNameLabel.TabIndex = 115;
			this.configNameLabel.Text = "Configuration name:";
			this.configNameLabel.Click += new System.EventHandler(this.configNameLabel_Click);
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(16, 152);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 118;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(448, 152);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 117;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// qualityComboBox
			// 
			this.qualityComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.qualityComboBox.Location = new System.Drawing.Point(120, 56);
			this.qualityComboBox.Name = "qualityComboBox";
			this.qualityComboBox.Size = new System.Drawing.Size(312, 21);
			this.qualityComboBox.TabIndex = 132;
			this.qualityComboBox.SelectedIndexChanged += new System.EventHandler(this.qualityComboBox_SelectedIndexChanged);
			// 
			// qualityEditButton
			// 
			this.qualityEditButton.Location = new System.Drawing.Point(448, 56);
			this.qualityEditButton.Name = "qualityEditButton";
			this.qualityEditButton.TabIndex = 131;
			this.qualityEditButton.Text = "View/edit";
			this.qualityEditButton.Click += new System.EventHandler(this.qualityEditButton_Click);
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(8, 56);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(88, 23);
			this.label7.TabIndex = 130;
			this.label7.Text = "Quality cut:";
			// 
			// linkEditButton
			// 
			this.linkEditButton.Location = new System.Drawing.Point(448, 88);
			this.linkEditButton.Name = "linkEditButton";
			this.linkEditButton.TabIndex = 129;
			this.linkEditButton.Text = "View/edit";
			this.linkEditButton.Click += new System.EventHandler(this.linkEditButton_Click);
			// 
			// scanEditButton
			// 
			this.scanEditButton.Location = new System.Drawing.Point(448, 120);
			this.scanEditButton.Name = "scanEditButton";
			this.scanEditButton.TabIndex = 128;
			this.scanEditButton.Text = "View/edit";
			this.scanEditButton.Click += new System.EventHandler(this.scanEditButton_Click);
			// 
			// scanningComboBox
			// 
			this.scanningComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.scanningComboBox.Location = new System.Drawing.Point(120, 120);
			this.scanningComboBox.Name = "scanningComboBox";
			this.scanningComboBox.Size = new System.Drawing.Size(312, 21);
			this.scanningComboBox.TabIndex = 127;
			this.scanningComboBox.SelectedIndexChanged += new System.EventHandler(this.scanningComboBox_SelectedIndexChanged);
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(8, 120);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(104, 23);
			this.label8.TabIndex = 125;
			this.label8.Text = "Scanning Config:";
			// 
			// linkingComboBox
			// 
			this.linkingComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.linkingComboBox.Location = new System.Drawing.Point(120, 88);
			this.linkingComboBox.Name = "linkingComboBox";
			this.linkingComboBox.Size = new System.Drawing.Size(312, 21);
			this.linkingComboBox.TabIndex = 126;
			this.linkingComboBox.SelectedIndexChanged += new System.EventHandler(this.linkingComboBox_SelectedIndexChanged);
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(8, 88);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(88, 23);
			this.label9.TabIndex = 124;
			this.label9.Text = "StripLink config:";
			// 
			// frmWideAreaScanConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(528, 181);
			this.Controls.Add(this.qualityComboBox);
			this.Controls.Add(this.qualityEditButton);
			this.Controls.Add(this.label7);
			this.Controls.Add(this.linkEditButton);
			this.Controls.Add(this.scanEditButton);
			this.Controls.Add(this.scanningComboBox);
			this.Controls.Add(this.label8);
			this.Controls.Add(this.linkingComboBox);
			this.Controls.Add(this.label9);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.configNameLabel);
			this.Name = "frmWideAreaScanConfig";
			this.Text = "WideAreaScanDriver configuration";
			this.ResumeLayout(false);

		}
		#endregion

		private void configNameLabel_Click(object sender, System.EventArgs e)
		{
		
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(WideAreaScanSettings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, ProgSettings);
			sw.Flush();
			SettingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "WideAreaScanDriver.exe", 1, 1, sw.ToString());
			Close();
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void qualityComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			ProgSettings.QualityCutId = ((Utilities.ConfigItem)qualityComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void linkingComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			ProgSettings.LinkConfigId = ((Utilities.ConfigItem)linkingComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void scanningComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			ProgSettings.ScanningConfigId = ((Utilities.ConfigItem)scanningComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

//Luillo
//		private void txtInitQuery_Leave(object sender, System.EventArgs e)
//		{
//			ProgSettings.InitSql  = txtInitQuery.Text;
//		}

		private void qualityEditButton_Click(object sender, System.EventArgs e)
		{
			string path = (string)(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME='ExeRepository'", _connection, null)).ExecuteScalar();
			object QualityCutExe = System.Activator.CreateInstanceFrom(path + @"\QualityCutConfig.exe", 
				"QualityCutConfig.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {ProgSettings.QualityCutId, _connection}, null, null, null).Unwrap();		
			long newid = (long) QualityCutExe.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, 
				null, QualityCutExe, new object[] {ProgSettings.QualityCutId, _connection}
				);
			if (newid != 0) ProgSettings.QualityCutId = newid;
			Utilities.FillComboBox(qualityComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE '%Sel.exe'", _connection);
			Utilities.SelectId(qualityComboBox, ProgSettings.QualityCutId);		
		}

		private void linkEditButton_Click(object sender, System.EventArgs e)
		{
			string path = (string)(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME='ExeRepository'", _connection, null)).ExecuteScalar();
			object LinkingConfigExe = System.Activator.CreateInstanceFrom(path + @"\StripLink.exe", 
				"LinkingConfig.frmLinkingConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {ProgSettings.LinkConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) LinkingConfigExe.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, 
				null, LinkingConfigExe, new object[] {ProgSettings.LinkConfigId, _connection}
				);
			if (newid != 0) ProgSettings.LinkConfigId = newid;
			Utilities.FillComboBox(linkingComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='StripLink.exe'", _connection);
			Utilities.SelectId(linkingComboBox, ProgSettings.LinkConfigId);
		}

		private void scanEditButton_Click(object sender, System.EventArgs e)
		{
			string path = (string)(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME='ExeRepository'", _connection, null)).ExecuteScalar();
			object ScanningConfigGui = System.Activator.CreateInstanceFrom(path + @"\ScanningConfigGui.exe", 
				"ScanningConfigGui.frmConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {ProgSettings.ScanningConfigId, _connection}, null, null, null).Unwrap();		
			//object[] test = {ref long id};
			long newid = (long) ScanningConfigGui.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static, 
				null, ScanningConfigGui, new object[] {ProgSettings.ScanningConfigId, _connection}
				);
			//MessageBox.Show("Got " + newid);
			if (newid != 0) ProgSettings.ScanningConfigId = newid;
			Utilities.FillComboBox(scanningComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE 'ScanServer%.exe'", _connection);
			Utilities.SelectId(scanningComboBox, ProgSettings.ScanningConfigId);
		}
	}
}
