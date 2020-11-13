using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.AreaScan2Driver
{
	/// <summary>
	/// Summary description for frmAreaScan2Config.
	/// </summary>
	internal class frmAreaScan2Config : System.Windows.Forms.Form
	{
		public static long SettingsId;
		private SySal.OperaDb.OperaDbConnection _connection;
		private string _currentConfigName = "";
		private string _originalConfigName = "";
		public AreaScan2Settings ProgSettings;
		private long _scanningConfigId;
		private long _linkingConfigId;
		private long _qualityCutId;

		private System.Windows.Forms.TextBox txtMinYDistance;
		private System.Windows.Forms.TextBox txtMinXDistance;
		private System.Windows.Forms.Label lblMinXDistance;
		private System.Windows.Forms.Label lblMinYDistance;
		private System.Windows.Forms.TextBox txtMinDensityBase;
		private System.Windows.Forms.Label lblMinDensityBase;
		private System.Windows.Forms.TextBox txtMinTracks;
		private System.Windows.Forms.Label lblMinTracks;
		private System.Windows.Forms.Label lblRecalibrationSelection;
		private System.Windows.Forms.TextBox txtRecalibrationSelection;
		private System.Windows.Forms.ComboBox qualityComboBox;
		private System.Windows.Forms.Button qualityEditButton;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Button btnCreate;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.TextBox baseThicknessTextBox;
		private System.Windows.Forms.TextBox maxTrialsTextBox;
		private System.Windows.Forms.TextBox slopeTolIncTextBox;
		private System.Windows.Forms.TextBox posTolIncTextBox;
		private System.Windows.Forms.TextBox slopeTolTextBox;
		private System.Windows.Forms.TextBox posTolTextBox;
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.Button linkEditButton;
		private System.Windows.Forms.Button scanEditButton;
		private System.Windows.Forms.Label maxTrialsLabel;
		private System.Windows.Forms.Label baseThicknessLabel;
		private System.Windows.Forms.Label posTolIncLabel;
		private System.Windows.Forms.Label slopeTolIncLabel;
		private System.Windows.Forms.ComboBox scanningComboBox;
		private System.Windows.Forms.Label posToleranceLabel;
		private System.Windows.Forms.Label slopeTolLabel;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.ComboBox linkingComboBox;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.ListBox lstReuseIds;
		private System.Windows.Forms.GroupBox grpReuseSettings;
		private System.Windows.Forms.TextBox txtMinOverlapArea;
		private System.Windows.Forms.CheckBox chkEnableReuse;
		private System.Windows.Forms.TextBox txtMinOverlapFraction;
		private System.Windows.Forms.Label lblMinOverlapFraction;
		private System.Windows.Forms.Label lblMinOverlapArea;
		private System.Windows.Forms.Label lblMaxMissingArea;
		private System.Windows.Forms.TextBox txtMaxMissingArea;
		private System.Windows.Forms.Button btnEditReuseIds;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public static long GetConfig(long settingsId, SySal.OperaDb.OperaDbConnection conn)
		{
			new frmAreaScan2Config(settingsId, conn).ShowDialog();
			return SettingsId;
		}

		public frmAreaScan2Config(long settingsId, SySal.OperaDb.OperaDbConnection conn)
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
				
				
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(AreaScan2Settings));
				ProgSettings = (AreaScan2Settings) xmls.Deserialize(new System.IO.StringReader(settings));

				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
				
				/*linkingComboBox.SelectedValue = _linkingConfigId;
				scanningComboBox.SelectedValue = _scanningConfigId;
				qualityComboBox.SelectedValue = _qualityCutId;*/
			}
			else
			{
				ProgSettings = new AreaScan2Settings();
				ProgSettings.Quality = new QualitySettings();
				ProgSettings.Recalibration = new RecalibrationSettings();
				ProgSettings.Reuse = new ReuseSettings();
			}
			
			posTolTextBox.Text = ProgSettings.Recalibration.PositionTolerance.ToString();
			posTolIncTextBox.Text = ProgSettings.Recalibration.PositionToleranceIncreaseWithSlope.ToString();
			slopeTolTextBox.Text = ProgSettings.Recalibration.SlopeTolerance.ToString();
			slopeTolIncTextBox.Text = ProgSettings.Recalibration.SlopeToleranceIncreaseWithSlope.ToString();
			maxTrialsTextBox.Text = ProgSettings.Quality.MaxTrials.ToString();
			baseThicknessTextBox.Text = ProgSettings.Recalibration.BaseThickness.ToString();
			txtRecalibrationSelection.Text = ProgSettings.Recalibration.SelectionText;
			txtMinXDistance.Text = ProgSettings.Recalibration.MinXDistance.ToString();
			txtMinYDistance.Text = ProgSettings.Recalibration.MinYDistance.ToString();
			txtMinTracks.Text = ProgSettings.Recalibration.MinTracks.ToString();
			txtMinDensityBase.Text = ProgSettings.Quality.MinDensityBase.ToString();

			txtMinOverlapArea.Text = ProgSettings.Reuse.MinOverlapArea.ToString();
			txtMinOverlapFraction.Text = ProgSettings.Reuse.MinOverlapFraction.ToString();
			txtMaxMissingArea.Text = ProgSettings.Reuse.MaxMissingArea.ToString();

			chkEnableReuse.Checked = ProgSettings.Reuse.Enable;
			chkEnableReuse_CheckedChanged(this, null);

			//			txtPositionTolerance.Text = ProgSettings.PositionTolerance.ToString();
			_linkingConfigId = ProgSettings.LinkConfigId;				
			_scanningConfigId = ProgSettings.ScanningConfigId;
			_qualityCutId = ProgSettings.QualityCutId;

			Utilities.SelectId(scanningComboBox, _scanningConfigId);
			Utilities.SelectId(linkingComboBox, _linkingConfigId);		
			Utilities.SelectId(qualityComboBox, _qualityCutId);
			
			if (ProgSettings.Reuse.ProgramSettingsIds != null )
				foreach(long id in ProgSettings.Reuse.ProgramSettingsIds)
			{
				lstReuseIds.Items.Add(new Utilities.ConfigItem(Utilities.LookupItem(id, _connection), id));
			}

			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
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
			this.txtMinYDistance = new System.Windows.Forms.TextBox();
			this.txtMinXDistance = new System.Windows.Forms.TextBox();
			this.lblMinXDistance = new System.Windows.Forms.Label();
			this.lblMinYDistance = new System.Windows.Forms.Label();
			this.txtMinDensityBase = new System.Windows.Forms.TextBox();
			this.lblMinDensityBase = new System.Windows.Forms.Label();
			this.txtMinTracks = new System.Windows.Forms.TextBox();
			this.lblMinTracks = new System.Windows.Forms.Label();
			this.lblRecalibrationSelection = new System.Windows.Forms.Label();
			this.txtRecalibrationSelection = new System.Windows.Forms.TextBox();
			this.qualityComboBox = new System.Windows.Forms.ComboBox();
			this.qualityEditButton = new System.Windows.Forms.Button();
			this.label7 = new System.Windows.Forms.Label();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.baseThicknessTextBox = new System.Windows.Forms.TextBox();
			this.maxTrialsTextBox = new System.Windows.Forms.TextBox();
			this.slopeTolIncTextBox = new System.Windows.Forms.TextBox();
			this.posTolIncTextBox = new System.Windows.Forms.TextBox();
			this.slopeTolTextBox = new System.Windows.Forms.TextBox();
			this.posTolTextBox = new System.Windows.Forms.TextBox();
			this.configNameLabel = new System.Windows.Forms.Label();
			this.linkEditButton = new System.Windows.Forms.Button();
			this.scanEditButton = new System.Windows.Forms.Button();
			this.maxTrialsLabel = new System.Windows.Forms.Label();
			this.baseThicknessLabel = new System.Windows.Forms.Label();
			this.posTolIncLabel = new System.Windows.Forms.Label();
			this.slopeTolIncLabel = new System.Windows.Forms.Label();
			this.scanningComboBox = new System.Windows.Forms.ComboBox();
			this.posToleranceLabel = new System.Windows.Forms.Label();
			this.slopeTolLabel = new System.Windows.Forms.Label();
			this.label8 = new System.Windows.Forms.Label();
			this.linkingComboBox = new System.Windows.Forms.ComboBox();
			this.label9 = new System.Windows.Forms.Label();
			this.lstReuseIds = new System.Windows.Forms.ListBox();
			this.btnEditReuseIds = new System.Windows.Forms.Button();
			this.grpReuseSettings = new System.Windows.Forms.GroupBox();
			this.lblMaxMissingArea = new System.Windows.Forms.Label();
			this.txtMaxMissingArea = new System.Windows.Forms.TextBox();
			this.lblMinOverlapArea = new System.Windows.Forms.Label();
			this.lblMinOverlapFraction = new System.Windows.Forms.Label();
			this.txtMinOverlapFraction = new System.Windows.Forms.TextBox();
			this.chkEnableReuse = new System.Windows.Forms.CheckBox();
			this.txtMinOverlapArea = new System.Windows.Forms.TextBox();
			this.grpReuseSettings.SuspendLayout();
			this.SuspendLayout();
			// 
			// txtMinYDistance
			// 
			this.txtMinYDistance.Location = new System.Drawing.Point(456, 144);
			this.txtMinYDistance.Name = "txtMinYDistance";
			this.txtMinYDistance.TabIndex = 129;
			this.txtMinYDistance.Text = "";
			this.txtMinYDistance.Leave += new System.EventHandler(this.txtMinYDistance_Leave);
			// 
			// txtMinXDistance
			// 
			this.txtMinXDistance.Location = new System.Drawing.Point(184, 144);
			this.txtMinXDistance.Name = "txtMinXDistance";
			this.txtMinXDistance.TabIndex = 128;
			this.txtMinXDistance.Text = "";
			this.txtMinXDistance.Leave += new System.EventHandler(this.txtMinXDistance_Leave);
			// 
			// lblMinXDistance
			// 
			this.lblMinXDistance.Location = new System.Drawing.Point(40, 144);
			this.lblMinXDistance.Name = "lblMinXDistance";
			this.lblMinXDistance.Size = new System.Drawing.Size(128, 23);
			this.lblMinXDistance.TabIndex = 127;
			this.lblMinXDistance.Text = "Min X distance:";
			// 
			// lblMinYDistance
			// 
			this.lblMinYDistance.Location = new System.Drawing.Point(304, 144);
			this.lblMinYDistance.Name = "lblMinYDistance";
			this.lblMinYDistance.Size = new System.Drawing.Size(128, 23);
			this.lblMinYDistance.TabIndex = 126;
			this.lblMinYDistance.Text = "Min Y distance:";
			// 
			// txtMinDensityBase
			// 
			this.txtMinDensityBase.Location = new System.Drawing.Point(456, 336);
			this.txtMinDensityBase.Name = "txtMinDensityBase";
			this.txtMinDensityBase.TabIndex = 125;
			this.txtMinDensityBase.Text = "";
			this.txtMinDensityBase.Leave += new System.EventHandler(this.txtMinDensityBase_Leave);
			// 
			// lblMinDensityBase
			// 
			this.lblMinDensityBase.Location = new System.Drawing.Point(304, 336);
			this.lblMinDensityBase.Name = "lblMinDensityBase";
			this.lblMinDensityBase.Size = new System.Drawing.Size(128, 23);
			this.lblMinDensityBase.TabIndex = 124;
			this.lblMinDensityBase.Text = "Min density base:";
			// 
			// txtMinTracks
			// 
			this.txtMinTracks.Location = new System.Drawing.Point(184, 112);
			this.txtMinTracks.Name = "txtMinTracks";
			this.txtMinTracks.TabIndex = 123;
			this.txtMinTracks.Text = "";
			this.txtMinTracks.Leave += new System.EventHandler(this.txtMinTracks_Leave);
			// 
			// lblMinTracks
			// 
			this.lblMinTracks.Location = new System.Drawing.Point(40, 112);
			this.lblMinTracks.Name = "lblMinTracks";
			this.lblMinTracks.Size = new System.Drawing.Size(144, 23);
			this.lblMinTracks.TabIndex = 122;
			this.lblMinTracks.Text = "Min tracks:";
			// 
			// lblRecalibrationSelection
			// 
			this.lblRecalibrationSelection.Location = new System.Drawing.Point(40, 192);
			this.lblRecalibrationSelection.Name = "lblRecalibrationSelection";
			this.lblRecalibrationSelection.Size = new System.Drawing.Size(152, 23);
			this.lblRecalibrationSelection.TabIndex = 121;
			this.lblRecalibrationSelection.Text = "Recalibration selection text:";
			// 
			// txtRecalibrationSelection
			// 
			this.txtRecalibrationSelection.Location = new System.Drawing.Point(32, 216);
			this.txtRecalibrationSelection.Multiline = true;
			this.txtRecalibrationSelection.Name = "txtRecalibrationSelection";
			this.txtRecalibrationSelection.Size = new System.Drawing.Size(528, 96);
			this.txtRecalibrationSelection.TabIndex = 120;
			this.txtRecalibrationSelection.Text = "";
			this.txtRecalibrationSelection.Leave += new System.EventHandler(this.txtRecalibrationSelection_Leave);
			// 
			// qualityComboBox
			// 
			this.qualityComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.qualityComboBox.Location = new System.Drawing.Point(144, 560);
			this.qualityComboBox.Name = "qualityComboBox";
			this.qualityComboBox.Size = new System.Drawing.Size(312, 21);
			this.qualityComboBox.TabIndex = 119;
			this.qualityComboBox.SelectedIndexChanged += new System.EventHandler(this.qualityComboBox_SelectedIndexChanged);
			// 
			// qualityEditButton
			// 
			this.qualityEditButton.Location = new System.Drawing.Point(472, 560);
			this.qualityEditButton.Name = "qualityEditButton";
			this.qualityEditButton.TabIndex = 118;
			this.qualityEditButton.Text = "View/edit";
			this.qualityEditButton.Click += new System.EventHandler(this.qualityEditButton_Click);
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(32, 560);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(88, 23);
			this.label7.TabIndex = 117;
			this.label7.Text = "Quality cut:";
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(32, 664);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 116;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(480, 664);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 115;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(184, 8);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(376, 20);
			this.configNameTextBox.TabIndex = 114;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// baseThicknessTextBox
			// 
			this.baseThicknessTextBox.Location = new System.Drawing.Point(458, 112);
			this.baseThicknessTextBox.Name = "baseThicknessTextBox";
			this.baseThicknessTextBox.TabIndex = 110;
			this.baseThicknessTextBox.Text = "";
			this.baseThicknessTextBox.Leave += new System.EventHandler(this.baseThicknessTextBox_Leave);
			// 
			// maxTrialsTextBox
			// 
			this.maxTrialsTextBox.Location = new System.Drawing.Point(184, 336);
			this.maxTrialsTextBox.Name = "maxTrialsTextBox";
			this.maxTrialsTextBox.TabIndex = 109;
			this.maxTrialsTextBox.Text = "";
			this.maxTrialsTextBox.Leave += new System.EventHandler(this.maxTrialsTextBox_Leave);
			// 
			// slopeTolIncTextBox
			// 
			this.slopeTolIncTextBox.Location = new System.Drawing.Point(458, 80);
			this.slopeTolIncTextBox.Name = "slopeTolIncTextBox";
			this.slopeTolIncTextBox.TabIndex = 106;
			this.slopeTolIncTextBox.Text = "";
			this.slopeTolIncTextBox.Leave += new System.EventHandler(this.slopeTolIncTextBox_Leave);
			// 
			// posTolIncTextBox
			// 
			this.posTolIncTextBox.Location = new System.Drawing.Point(186, 80);
			this.posTolIncTextBox.Name = "posTolIncTextBox";
			this.posTolIncTextBox.TabIndex = 105;
			this.posTolIncTextBox.Text = "";
			this.posTolIncTextBox.Leave += new System.EventHandler(this.posTolIncTextBox_Leave);
			// 
			// slopeTolTextBox
			// 
			this.slopeTolTextBox.Location = new System.Drawing.Point(458, 48);
			this.slopeTolTextBox.Name = "slopeTolTextBox";
			this.slopeTolTextBox.TabIndex = 100;
			this.slopeTolTextBox.Text = "";
			this.slopeTolTextBox.Leave += new System.EventHandler(this.slopeTolTextBox_Leave);
			// 
			// posTolTextBox
			// 
			this.posTolTextBox.Location = new System.Drawing.Point(184, 48);
			this.posTolTextBox.Name = "posTolTextBox";
			this.posTolTextBox.TabIndex = 99;
			this.posTolTextBox.Text = "";
			this.posTolTextBox.Leave += new System.EventHandler(this.posTolTextBox_Leave);
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(42, 11);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(112, 23);
			this.configNameLabel.TabIndex = 113;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// linkEditButton
			// 
			this.linkEditButton.Location = new System.Drawing.Point(472, 592);
			this.linkEditButton.Name = "linkEditButton";
			this.linkEditButton.TabIndex = 112;
			this.linkEditButton.Text = "View/edit";
			this.linkEditButton.Click += new System.EventHandler(this.linkEditButton_Click);
			// 
			// scanEditButton
			// 
			this.scanEditButton.Location = new System.Drawing.Point(472, 624);
			this.scanEditButton.Name = "scanEditButton";
			this.scanEditButton.TabIndex = 111;
			this.scanEditButton.Text = "View/edit";
			this.scanEditButton.Click += new System.EventHandler(this.scanEditButton_Click);
			// 
			// maxTrialsLabel
			// 
			this.maxTrialsLabel.Location = new System.Drawing.Point(40, 336);
			this.maxTrialsLabel.Name = "maxTrialsLabel";
			this.maxTrialsLabel.Size = new System.Drawing.Size(88, 23);
			this.maxTrialsLabel.TabIndex = 108;
			this.maxTrialsLabel.Text = "Max trials:";
			// 
			// baseThicknessLabel
			// 
			this.baseThicknessLabel.Location = new System.Drawing.Point(306, 112);
			this.baseThicknessLabel.Name = "baseThicknessLabel";
			this.baseThicknessLabel.Size = new System.Drawing.Size(142, 23);
			this.baseThicknessLabel.TabIndex = 107;
			this.baseThicknessLabel.Text = "Base thickness:";
			// 
			// posTolIncLabel
			// 
			this.posTolIncLabel.Location = new System.Drawing.Point(42, 80);
			this.posTolIncLabel.Name = "posTolIncLabel";
			this.posTolIncLabel.Size = new System.Drawing.Size(144, 23);
			this.posTolIncLabel.TabIndex = 104;
			this.posTolIncLabel.Text = "Position tolerance increase:";
			// 
			// slopeTolIncLabel
			// 
			this.slopeTolIncLabel.Location = new System.Drawing.Point(306, 80);
			this.slopeTolIncLabel.Name = "slopeTolIncLabel";
			this.slopeTolIncLabel.Size = new System.Drawing.Size(144, 23);
			this.slopeTolIncLabel.TabIndex = 103;
			this.slopeTolIncLabel.Text = "Slope tolerance increase:";
			// 
			// scanningComboBox
			// 
			this.scanningComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.scanningComboBox.Location = new System.Drawing.Point(144, 624);
			this.scanningComboBox.Name = "scanningComboBox";
			this.scanningComboBox.Size = new System.Drawing.Size(312, 21);
			this.scanningComboBox.TabIndex = 102;
			this.scanningComboBox.SelectedIndexChanged += new System.EventHandler(this.scanningComboBox_SelectedIndexChanged);
			// 
			// posToleranceLabel
			// 
			this.posToleranceLabel.Location = new System.Drawing.Point(42, 48);
			this.posToleranceLabel.Name = "posToleranceLabel";
			this.posToleranceLabel.Size = new System.Drawing.Size(128, 23);
			this.posToleranceLabel.TabIndex = 96;
			this.posToleranceLabel.Text = "Position tolerance:";
			// 
			// slopeTolLabel
			// 
			this.slopeTolLabel.Location = new System.Drawing.Point(306, 48);
			this.slopeTolLabel.Name = "slopeTolLabel";
			this.slopeTolLabel.Size = new System.Drawing.Size(128, 23);
			this.slopeTolLabel.TabIndex = 95;
			this.slopeTolLabel.Text = "Slope tolerance:";
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(32, 624);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(104, 23);
			this.label8.TabIndex = 98;
			this.label8.Text = "Scanning Config:";
			// 
			// linkingComboBox
			// 
			this.linkingComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.linkingComboBox.Location = new System.Drawing.Point(144, 592);
			this.linkingComboBox.Name = "linkingComboBox";
			this.linkingComboBox.Size = new System.Drawing.Size(312, 21);
			this.linkingComboBox.TabIndex = 101;
			this.linkingComboBox.SelectedIndexChanged += new System.EventHandler(this.linkingComboBox_SelectedIndexChanged);
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(32, 592);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(88, 23);
			this.label9.TabIndex = 97;
			this.label9.Text = "Linking Config:";
			// 
			// lstReuseIds
			// 
			this.lstReuseIds.Location = new System.Drawing.Point(16, 40);
			this.lstReuseIds.Name = "lstReuseIds";
			this.lstReuseIds.Size = new System.Drawing.Size(432, 95);
			this.lstReuseIds.TabIndex = 131;
			// 
			// btnEditReuseIds
			// 
			this.btnEditReuseIds.Location = new System.Drawing.Point(456, 72);
			this.btnEditReuseIds.Name = "btnEditReuseIds";
			this.btnEditReuseIds.Size = new System.Drawing.Size(104, 24);
			this.btnEditReuseIds.TabIndex = 134;
			this.btnEditReuseIds.Text = "Edit";
			this.btnEditReuseIds.Click += new System.EventHandler(this.button1_Click);
			// 
			// grpReuseSettings
			// 
			this.grpReuseSettings.Controls.Add(this.lblMaxMissingArea);
			this.grpReuseSettings.Controls.Add(this.txtMaxMissingArea);
			this.grpReuseSettings.Controls.Add(this.lblMinOverlapArea);
			this.grpReuseSettings.Controls.Add(this.lblMinOverlapFraction);
			this.grpReuseSettings.Controls.Add(this.txtMinOverlapFraction);
			this.grpReuseSettings.Controls.Add(this.chkEnableReuse);
			this.grpReuseSettings.Controls.Add(this.txtMinOverlapArea);
			this.grpReuseSettings.Controls.Add(this.btnEditReuseIds);
			this.grpReuseSettings.Controls.Add(this.lstReuseIds);
			this.grpReuseSettings.Location = new System.Drawing.Point(16, 368);
			this.grpReuseSettings.Name = "grpReuseSettings";
			this.grpReuseSettings.Size = new System.Drawing.Size(568, 176);
			this.grpReuseSettings.TabIndex = 135;
			this.grpReuseSettings.TabStop = false;
			this.grpReuseSettings.Text = "Reuse settings";
			// 
			// lblMaxMissingArea
			// 
			this.lblMaxMissingArea.Location = new System.Drawing.Point(384, 144);
			this.lblMaxMissingArea.Name = "lblMaxMissingArea";
			this.lblMaxMissingArea.Size = new System.Drawing.Size(96, 23);
			this.lblMaxMissingArea.TabIndex = 141;
			this.lblMaxMissingArea.Text = "Max missing area:";
			// 
			// txtMaxMissingArea
			// 
			this.txtMaxMissingArea.Location = new System.Drawing.Point(480, 144);
			this.txtMaxMissingArea.Name = "txtMaxMissingArea";
			this.txtMaxMissingArea.Size = new System.Drawing.Size(56, 20);
			this.txtMaxMissingArea.TabIndex = 140;
			this.txtMaxMissingArea.Text = "";
			this.txtMaxMissingArea.Leave += new System.EventHandler(this.txtMaxMissingArea_Leave);
			// 
			// lblMinOverlapArea
			// 
			this.lblMinOverlapArea.Location = new System.Drawing.Point(8, 144);
			this.lblMinOverlapArea.Name = "lblMinOverlapArea";
			this.lblMinOverlapArea.Size = new System.Drawing.Size(96, 23);
			this.lblMinOverlapArea.TabIndex = 139;
			this.lblMinOverlapArea.Text = "Min overlap area:";
			// 
			// lblMinOverlapFraction
			// 
			this.lblMinOverlapFraction.Location = new System.Drawing.Point(184, 144);
			this.lblMinOverlapFraction.Name = "lblMinOverlapFraction";
			this.lblMinOverlapFraction.Size = new System.Drawing.Size(112, 23);
			this.lblMinOverlapFraction.TabIndex = 138;
			this.lblMinOverlapFraction.Text = "Min overlap fraction:";
			// 
			// txtMinOverlapFraction
			// 
			this.txtMinOverlapFraction.Location = new System.Drawing.Point(304, 144);
			this.txtMinOverlapFraction.Name = "txtMinOverlapFraction";
			this.txtMinOverlapFraction.Size = new System.Drawing.Size(56, 20);
			this.txtMinOverlapFraction.TabIndex = 137;
			this.txtMinOverlapFraction.Text = "";
			this.txtMinOverlapFraction.Leave += new System.EventHandler(this.txtMinOverlapFraction_Leave);
			// 
			// chkEnableReuse
			// 
			this.chkEnableReuse.Location = new System.Drawing.Point(152, 8);
			this.chkEnableReuse.Name = "chkEnableReuse";
			this.chkEnableReuse.TabIndex = 136;
			this.chkEnableReuse.Text = "Enable";
			this.chkEnableReuse.CheckedChanged += new System.EventHandler(this.chkEnableReuse_CheckedChanged);
			// 
			// txtMinOverlapArea
			// 
			this.txtMinOverlapArea.Location = new System.Drawing.Point(104, 144);
			this.txtMinOverlapArea.Name = "txtMinOverlapArea";
			this.txtMinOverlapArea.Size = new System.Drawing.Size(56, 20);
			this.txtMinOverlapArea.TabIndex = 135;
			this.txtMinOverlapArea.Text = "";
			this.txtMinOverlapArea.Leave += new System.EventHandler(this.txtMinOverlapArea_Leave);
			// 
			// frmAreaScan2Config
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(592, 694);
			this.Controls.Add(this.grpReuseSettings);
			this.Controls.Add(this.txtMinYDistance);
			this.Controls.Add(this.txtMinXDistance);
			this.Controls.Add(this.lblMinXDistance);
			this.Controls.Add(this.lblMinYDistance);
			this.Controls.Add(this.txtMinDensityBase);
			this.Controls.Add(this.lblMinDensityBase);
			this.Controls.Add(this.txtMinTracks);
			this.Controls.Add(this.lblMinTracks);
			this.Controls.Add(this.lblRecalibrationSelection);
			this.Controls.Add(this.txtRecalibrationSelection);
			this.Controls.Add(this.qualityComboBox);
			this.Controls.Add(this.qualityEditButton);
			this.Controls.Add(this.label7);
			this.Controls.Add(this.btnCreate);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.baseThicknessTextBox);
			this.Controls.Add(this.maxTrialsTextBox);
			this.Controls.Add(this.slopeTolIncTextBox);
			this.Controls.Add(this.posTolIncTextBox);
			this.Controls.Add(this.slopeTolTextBox);
			this.Controls.Add(this.posTolTextBox);
			this.Controls.Add(this.configNameLabel);
			this.Controls.Add(this.linkEditButton);
			this.Controls.Add(this.scanEditButton);
			this.Controls.Add(this.maxTrialsLabel);
			this.Controls.Add(this.baseThicknessLabel);
			this.Controls.Add(this.posTolIncLabel);
			this.Controls.Add(this.slopeTolIncLabel);
			this.Controls.Add(this.scanningComboBox);
			this.Controls.Add(this.posToleranceLabel);
			this.Controls.Add(this.slopeTolLabel);
			this.Controls.Add(this.label8);
			this.Controls.Add(this.linkingComboBox);
			this.Controls.Add(this.label9);
			this.Name = "frmAreaScan2Config";
			this.Text = "AreaScan2Driver configuration";
			this.grpReuseSettings.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(AreaScan2Settings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, ProgSettings);
			sw.Flush();
			SettingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "AreaScan2Driver.exe", 1, 0, sw.ToString());
			Close();
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void posTolIncTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.PositionToleranceIncreaseWithSlope = Convert.ToDouble(posTolIncTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				posTolIncTextBox.Focus();
			}
		}

		private void posTolTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.PositionTolerance = Convert.ToDouble(posTolTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				posTolTextBox.Focus();
			}
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			_currentConfigName = configNameTextBox.Text.Trim();
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
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

		private void slopeTolTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.SlopeTolerance = Convert.ToDouble(slopeTolTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				slopeTolTextBox.Focus();
			}
		}

		private void slopeTolIncTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.SlopeToleranceIncreaseWithSlope = Convert.ToDouble(slopeTolIncTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				slopeTolIncTextBox.Focus();
			}
		}

		private void txtMinTracks_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.MinTracks = Convert.ToInt32(txtMinTracks.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinTracks.Focus();
			}
		}

		private void baseThicknessTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.BaseThickness = Convert.ToDouble(baseThicknessTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				baseThicknessTextBox.Focus();
			}
		}

		private void txtMinXDistance_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.MinXDistance = Convert.ToDouble(txtMinXDistance.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinXDistance.Focus();
			}
		}

		private void txtMinYDistance_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Recalibration.MinYDistance = Convert.ToDouble(txtMinYDistance.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinYDistance.Focus();
			}
		}

		private void txtRecalibrationSelection_Leave(object sender, System.EventArgs e)
		{
				ProgSettings.Recalibration.SelectionText  = txtRecalibrationSelection.Text;
		}

		private void maxTrialsTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Quality.MaxTrials = (uint) Convert.ToInt16(maxTrialsTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				maxTrialsTextBox.Focus();
			}
		}

		private void txtMinDensityBase_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Quality.MinDensityBase = (double) Convert.ToDouble(txtMinDensityBase.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinDensityBase.Focus();
			}
		}

		private void btnReuseSettings_Click(object sender, System.EventArgs e)
		{
			//(new frmReuseSettings(_connection)).ShowDialog();
			ProgSettings.Reuse.ProgramSettingsIds = frmReuseSettings.Get(_connection, ProgSettings.Reuse.ProgramSettingsIds);
		}

		private void button1_Click(object sender, System.EventArgs e)
		{
			ProgSettings.Reuse.ProgramSettingsIds = frmReuseSettings.Get(_connection, ProgSettings.Reuse.ProgramSettingsIds);
			lstReuseIds.Items.Clear();
			if (ProgSettings.Reuse.ProgramSettingsIds != null )
				foreach(long id in ProgSettings.Reuse.ProgramSettingsIds)
			{
				lstReuseIds.Items.Add(new Utilities.ConfigItem(Utilities.LookupItem(id, _connection), id));
			}
		}

		private void button2_Click(object sender, System.EventArgs e)
		{			
		}

		private void txtMinOverlapArea_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Reuse.MinOverlapArea = Convert.ToDouble(txtMinOverlapArea.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinOverlapArea.Focus();
			}
		}

		private void txtMinOverlapFraction_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Reuse.MinOverlapFraction = Convert.ToDouble(txtMinOverlapFraction.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinOverlapFraction.Focus();
			}
		}

		private void txtMaxMissingArea_Leave(object sender, System.EventArgs e)
		{
			try
			{
				ProgSettings.Reuse.MaxMissingArea = Convert.ToDouble(txtMaxMissingArea.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMaxMissingArea.Focus();
			}
		}

		private void chkEnableReuse_CheckedChanged(object sender, System.EventArgs e)
		{
			ProgSettings.Reuse.Enable = chkEnableReuse.Checked;
			if (chkEnableReuse.Checked) 
			{
				lstReuseIds.Enabled = true;
				txtMinOverlapArea.Enabled = true;
				txtMinOverlapFraction.Enabled = true;
				txtMaxMissingArea.Enabled = true;
				btnEditReuseIds.Enabled = true;
			}
			else
			{
				lstReuseIds.Enabled = false;
				txtMinOverlapArea.Enabled = false;
				txtMinOverlapFraction.Enabled = false;
				txtMaxMissingArea.Enabled = false;
				btnEditReuseIds.Enabled = false;
			}
		}

		private void qualityEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
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
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\'));
			object LinkingConfigExe = System.Activator.CreateInstanceFrom(path + @"\LinkingConfig.exe", 
				"LinkingConfig.frmLinkingConfig", false, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance, null, new object[2] {ProgSettings.LinkConfigId, _connection}, null, null, null).Unwrap();		
			long newid = (long) LinkingConfigExe.GetType().InvokeMember("Get", 
				System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, 
				null, LinkingConfigExe, new object[] {ProgSettings.LinkConfigId, _connection}
				);
			if (newid != 0) ProgSettings.LinkConfigId = newid;
			Utilities.FillComboBox(linkingComboBox, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE='BatchLink.exe'", _connection);
			Utilities.SelectId(linkingComboBox, ProgSettings.LinkConfigId);
		}

		private void scanEditButton_Click(object sender, System.EventArgs e)
		{
			string path = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName; path = path.Remove(path.LastIndexOf('\\'), path.Length - path.LastIndexOf('\\')); 
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

		/*private void lstReuseIds_SelectedIndexChanged(object sender, System.EventArgs e)
		{
		
		}*/

		



	}
}
