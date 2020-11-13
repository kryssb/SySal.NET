using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
	/// <summary>
	/// Summary description for frmConfig.
	/// </summary>
	internal class frmConfig : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.Button linkEditButton;
		private System.Windows.Forms.Button scanEditButton;
		private System.Windows.Forms.TextBox baseThicknessTextBox;
		private System.Windows.Forms.Label maxTrialsLabel;
		private System.Windows.Forms.Label baseThicknessLabel;
		private System.Windows.Forms.TextBox maxTrialsTextBox;
		private System.Windows.Forms.TextBox slopeTolIncTextBox;
		private System.Windows.Forms.Label posTolIncLabel;
		private System.Windows.Forms.Label slopeTolIncLabel;
		private System.Windows.Forms.TextBox posTolIncTextBox;
		private System.Windows.Forms.ComboBox scanningComboBox;
		private System.Windows.Forms.TextBox slopeTolTextBox;
		private System.Windows.Forms.Label posToleranceLabel;
		private System.Windows.Forms.Label slopeTolLabel;
		private System.Windows.Forms.TextBox posTolTextBox;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.ComboBox linkingComboBox;
		private System.Windows.Forms.Label label9;
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
		private PredictionScan3Settings _progSettings;
		private long _scanningConfigId;
		private long _linkingConfigId;
		private long _qualityCutId;
		private System.Windows.Forms.Button qualityEditButton;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TextBox txtRecalibrationSelection;
		private System.Windows.Forms.Label lblRecalibrationSelection;
		private System.Windows.Forms.TextBox txtPositionTolerance;
		private System.Windows.Forms.TextBox txtMinTracks;
		private System.Windows.Forms.TextBox txtMinYDistance;
		private System.Windows.Forms.TextBox txtMinXDistance;
		private System.Windows.Forms.Label lblMinTracks;
		private System.Windows.Forms.Label lblPositionTolerance;
		private System.Windows.Forms.Label lblMinXDistance;
		private System.Windows.Forms.Label lblMinYDistance;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox txtSelectionFunction;
		private System.Windows.Forms.TextBox txtMaxSelFVal;
		private System.Windows.Forms.TextBox txtMinSelFVal;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button btnSyntaxHelp;
		private System.Windows.Forms.ComboBox qualityComboBox;
		

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
			Utilities.FillComboBox(linkingComboBox, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE 'BatchLink%.exe'", conn);
			Utilities.FillComboBox(scanningComboBox, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE 'ScanServer%.exe'", conn);
			Utilities.FillComboBox(qualityComboBox, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE '%Sel.exe'", conn);

			if (settingsId != 0)
			{
				string settings = (string)(new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
				
				
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(PredictionScan3Settings));
				_progSettings = (PredictionScan3Settings) xmls.Deserialize(new System.IO.StringReader(settings));

				_originalConfigName = _currentConfigName =  configNameTextBox.Text = (string)(new SySal.OperaDb.OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID = " + settingsId, conn)).ExecuteScalar();
				
				/*linkingComboBox.SelectedValue = _linkingConfigId;
				scanningComboBox.SelectedValue = _scanningConfigId;
				qualityComboBox.SelectedValue = _qualityCutId;*/
			}
			else
			{
				_progSettings = new PredictionScan3Settings();
			}
			
			posTolTextBox.Text = _progSettings.PositionTolerance.ToString(System.Globalization.CultureInfo.InvariantCulture);
			posTolIncTextBox.Text = _progSettings.PositionToleranceIncreaseWithSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
			slopeTolTextBox.Text = _progSettings.SlopeTolerance.ToString(System.Globalization.CultureInfo.InvariantCulture);
			slopeTolIncTextBox.Text = _progSettings.SlopeToleranceIncreaseWithSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
			maxTrialsTextBox.Text = _progSettings.MaxTrials.ToString(System.Globalization.CultureInfo.InvariantCulture);
			baseThicknessTextBox.Text = _progSettings.BaseThickness.ToString(System.Globalization.CultureInfo.InvariantCulture);
			txtRecalibrationSelection.Text = _progSettings.RecalibrationSelectionText;
			txtMinXDistance.Text = _progSettings.RecalibrationMinXDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
			txtMinYDistance.Text = _progSettings.RecalibrationMinYDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
			txtMinTracks.Text = _progSettings.RecalibrationMinTracks.ToString(System.Globalization.CultureInfo.InvariantCulture);
			txtPositionTolerance.Text = _progSettings.PositionTolerance.ToString(System.Globalization.CultureInfo.InvariantCulture);
			txtSelectionFunction.Text = _progSettings.SelectionFunction;
			txtMinSelFVal.Text = _progSettings.SelectionFunctionMin.ToString(System.Globalization.CultureInfo.InvariantCulture);
			txtMaxSelFVal.Text = _progSettings.SelectionFunctionMax.ToString(System.Globalization.CultureInfo.InvariantCulture);
			_linkingConfigId = _progSettings.LinkConfigId;				
			_scanningConfigId = _progSettings.ScanningConfigId;
			_qualityCutId = _progSettings.QualityCutId;

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

		private bool IsFormFilled()
		{
			return !(_progSettings.LinkConfigId == 0 | _progSettings.QualityCutId == 0 | _progSettings.ScanningConfigId == 0);
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
			this.linkEditButton = new System.Windows.Forms.Button();
			this.scanEditButton = new System.Windows.Forms.Button();
			this.baseThicknessTextBox = new System.Windows.Forms.TextBox();
			this.maxTrialsLabel = new System.Windows.Forms.Label();
			this.baseThicknessLabel = new System.Windows.Forms.Label();
			this.maxTrialsTextBox = new System.Windows.Forms.TextBox();
			this.slopeTolIncTextBox = new System.Windows.Forms.TextBox();
			this.posTolIncLabel = new System.Windows.Forms.Label();
			this.slopeTolIncLabel = new System.Windows.Forms.Label();
			this.posTolIncTextBox = new System.Windows.Forms.TextBox();
			this.scanningComboBox = new System.Windows.Forms.ComboBox();
			this.slopeTolTextBox = new System.Windows.Forms.TextBox();
			this.posToleranceLabel = new System.Windows.Forms.Label();
			this.slopeTolLabel = new System.Windows.Forms.Label();
			this.posTolTextBox = new System.Windows.Forms.TextBox();
			this.label8 = new System.Windows.Forms.Label();
			this.linkingComboBox = new System.Windows.Forms.ComboBox();
			this.label9 = new System.Windows.Forms.Label();
			this.btnCreate = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.qualityEditButton = new System.Windows.Forms.Button();
			this.label7 = new System.Windows.Forms.Label();
			this.qualityComboBox = new System.Windows.Forms.ComboBox();
			this.txtRecalibrationSelection = new System.Windows.Forms.TextBox();
			this.lblRecalibrationSelection = new System.Windows.Forms.Label();
			this.txtPositionTolerance = new System.Windows.Forms.TextBox();
			this.txtMinTracks = new System.Windows.Forms.TextBox();
			this.txtMinYDistance = new System.Windows.Forms.TextBox();
			this.txtMinXDistance = new System.Windows.Forms.TextBox();
			this.lblMinTracks = new System.Windows.Forms.Label();
			this.lblPositionTolerance = new System.Windows.Forms.Label();
			this.lblMinXDistance = new System.Windows.Forms.Label();
			this.lblMinYDistance = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.txtSelectionFunction = new System.Windows.Forms.TextBox();
			this.txtMaxSelFVal = new System.Windows.Forms.TextBox();
			this.txtMinSelFVal = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.label3 = new System.Windows.Forms.Label();
			this.btnSyntaxHelp = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(160, 8);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(376, 20);
			this.configNameTextBox.TabIndex = 2;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(18, 11);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(112, 23);
			this.configNameLabel.TabIndex = 1;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// linkEditButton
			// 
			this.linkEditButton.Location = new System.Drawing.Point(456, 464);
			this.linkEditButton.Name = "linkEditButton";
			this.linkEditButton.TabIndex = 37;
			this.linkEditButton.Text = "View/edit";
			this.linkEditButton.Click += new System.EventHandler(this.linkEditButton_Click);
			// 
			// scanEditButton
			// 
			this.scanEditButton.Location = new System.Drawing.Point(456, 496);
			this.scanEditButton.Name = "scanEditButton";
			this.scanEditButton.TabIndex = 40;
			this.scanEditButton.Text = "View/edit";
			this.scanEditButton.Click += new System.EventHandler(this.scanEditButton_Click);
			// 
			// baseThicknessTextBox
			// 
			this.baseThicknessTextBox.Location = new System.Drawing.Point(434, 123);
			this.baseThicknessTextBox.Name = "baseThicknessTextBox";
			this.baseThicknessTextBox.TabIndex = 14;
			this.baseThicknessTextBox.Text = "";
			this.baseThicknessTextBox.Leave += new System.EventHandler(this.baseThicknessTextBox_Leave);
			// 
			// maxTrialsLabel
			// 
			this.maxTrialsLabel.Location = new System.Drawing.Point(18, 123);
			this.maxTrialsLabel.Name = "maxTrialsLabel";
			this.maxTrialsLabel.Size = new System.Drawing.Size(88, 23);
			this.maxTrialsLabel.TabIndex = 11;
			this.maxTrialsLabel.Text = "Max trials:";
			// 
			// baseThicknessLabel
			// 
			this.baseThicknessLabel.Location = new System.Drawing.Point(282, 123);
			this.baseThicknessLabel.Name = "baseThicknessLabel";
			this.baseThicknessLabel.Size = new System.Drawing.Size(142, 23);
			this.baseThicknessLabel.TabIndex = 13;
			this.baseThicknessLabel.Text = "Base thickness:";
			// 
			// maxTrialsTextBox
			// 
			this.maxTrialsTextBox.Location = new System.Drawing.Point(162, 123);
			this.maxTrialsTextBox.Name = "maxTrialsTextBox";
			this.maxTrialsTextBox.TabIndex = 12;
			this.maxTrialsTextBox.Text = "";
			this.maxTrialsTextBox.Leave += new System.EventHandler(this.maxTrialsTextBox_Leave);
			// 
			// slopeTolIncTextBox
			// 
			this.slopeTolIncTextBox.Location = new System.Drawing.Point(434, 91);
			this.slopeTolIncTextBox.Name = "slopeTolIncTextBox";
			this.slopeTolIncTextBox.TabIndex = 10;
			this.slopeTolIncTextBox.Text = "";
			this.slopeTolIncTextBox.TextChanged += new System.EventHandler(this.slopeTolIncTextBox_TextChanged);
			// 
			// posTolIncLabel
			// 
			this.posTolIncLabel.Location = new System.Drawing.Point(18, 91);
			this.posTolIncLabel.Name = "posTolIncLabel";
			this.posTolIncLabel.Size = new System.Drawing.Size(144, 23);
			this.posTolIncLabel.TabIndex = 5;
			this.posTolIncLabel.Text = "Position tolerance increase:";
			// 
			// slopeTolIncLabel
			// 
			this.slopeTolIncLabel.Location = new System.Drawing.Point(282, 91);
			this.slopeTolIncLabel.Name = "slopeTolIncLabel";
			this.slopeTolIncLabel.Size = new System.Drawing.Size(144, 23);
			this.slopeTolIncLabel.TabIndex = 9;
			this.slopeTolIncLabel.Text = "Slope tolerance increase:";
			// 
			// posTolIncTextBox
			// 
			this.posTolIncTextBox.Location = new System.Drawing.Point(162, 91);
			this.posTolIncTextBox.Name = "posTolIncTextBox";
			this.posTolIncTextBox.TabIndex = 6;
			this.posTolIncTextBox.Text = "";
			this.posTolIncTextBox.Leave += new System.EventHandler(this.posTolIncTextBox_Leave);
			// 
			// scanningComboBox
			// 
			this.scanningComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.scanningComboBox.Location = new System.Drawing.Point(120, 496);
			this.scanningComboBox.Name = "scanningComboBox";
			this.scanningComboBox.Size = new System.Drawing.Size(312, 21);
			this.scanningComboBox.TabIndex = 39;
			this.scanningComboBox.SelectedIndexChanged += new System.EventHandler(this.scanningComboBox_SelectedIndexChanged);
			// 
			// slopeTolTextBox
			// 
			this.slopeTolTextBox.Location = new System.Drawing.Point(434, 59);
			this.slopeTolTextBox.Name = "slopeTolTextBox";
			this.slopeTolTextBox.TabIndex = 8;
			this.slopeTolTextBox.Text = "";
			this.slopeTolTextBox.Leave += new System.EventHandler(this.slopeTolTextBox_Leave);
			// 
			// posToleranceLabel
			// 
			this.posToleranceLabel.Location = new System.Drawing.Point(18, 59);
			this.posToleranceLabel.Name = "posToleranceLabel";
			this.posToleranceLabel.Size = new System.Drawing.Size(128, 23);
			this.posToleranceLabel.TabIndex = 3;
			this.posToleranceLabel.Text = "Position tolerance:";
			// 
			// slopeTolLabel
			// 
			this.slopeTolLabel.Location = new System.Drawing.Point(282, 59);
			this.slopeTolLabel.Name = "slopeTolLabel";
			this.slopeTolLabel.Size = new System.Drawing.Size(128, 23);
			this.slopeTolLabel.TabIndex = 7;
			this.slopeTolLabel.Text = "Slope tolerance:";
			// 
			// posTolTextBox
			// 
			this.posTolTextBox.Location = new System.Drawing.Point(162, 59);
			this.posTolTextBox.Name = "posTolTextBox";
			this.posTolTextBox.TabIndex = 4;
			this.posTolTextBox.Text = "";
			this.posTolTextBox.Leave += new System.EventHandler(this.posTolTextBox_Leave);
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(8, 496);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(104, 23);
			this.label8.TabIndex = 38;
			this.label8.Text = "Scanning Config:";
			// 
			// linkingComboBox
			// 
			this.linkingComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.linkingComboBox.Location = new System.Drawing.Point(120, 464);
			this.linkingComboBox.Name = "linkingComboBox";
			this.linkingComboBox.Size = new System.Drawing.Size(312, 21);
			this.linkingComboBox.TabIndex = 36;
			this.linkingComboBox.SelectedIndexChanged += new System.EventHandler(this.linkingComboBox_SelectedIndexChanged);
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(8, 464);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(88, 23);
			this.label9.TabIndex = 35;
			this.label9.Text = "Linking Config:";
			// 
			// btnCreate
			// 
			this.btnCreate.Enabled = false;
			this.btnCreate.Location = new System.Drawing.Point(8, 528);
			this.btnCreate.Name = "btnCreate";
			this.btnCreate.TabIndex = 41;
			this.btnCreate.Text = "Create";
			this.btnCreate.Click += new System.EventHandler(this.btnCreate_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.Location = new System.Drawing.Point(456, 528);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 42;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// qualityEditButton
			// 
			this.qualityEditButton.Location = new System.Drawing.Point(456, 432);
			this.qualityEditButton.Name = "qualityEditButton";
			this.qualityEditButton.TabIndex = 34;
			this.qualityEditButton.Text = "View/edit";
			this.qualityEditButton.Click += new System.EventHandler(this.qualityEditButton_Click);
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(8, 432);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(88, 23);
			this.label7.TabIndex = 32;
			this.label7.Text = "Quality cut:";
			// 
			// qualityComboBox
			// 
			this.qualityComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.qualityComboBox.Location = new System.Drawing.Point(120, 432);
			this.qualityComboBox.Name = "qualityComboBox";
			this.qualityComboBox.Size = new System.Drawing.Size(312, 21);
			this.qualityComboBox.TabIndex = 33;
			this.qualityComboBox.SelectedIndexChanged += new System.EventHandler(this.qualityComboBox_SelectedIndexChanged);
			// 
			// txtRecalibrationSelection
			// 
			this.txtRecalibrationSelection.Location = new System.Drawing.Point(8, 176);
			this.txtRecalibrationSelection.Multiline = true;
			this.txtRecalibrationSelection.Name = "txtRecalibrationSelection";
			this.txtRecalibrationSelection.Size = new System.Drawing.Size(528, 96);
			this.txtRecalibrationSelection.TabIndex = 16;
			this.txtRecalibrationSelection.Text = "";
			this.txtRecalibrationSelection.Leave += new System.EventHandler(this.txtRecalibrationSelection_Leave);
			// 
			// lblRecalibrationSelection
			// 
			this.lblRecalibrationSelection.Location = new System.Drawing.Point(16, 152);
			this.lblRecalibrationSelection.Name = "lblRecalibrationSelection";
			this.lblRecalibrationSelection.Size = new System.Drawing.Size(152, 23);
			this.lblRecalibrationSelection.TabIndex = 15;
			this.lblRecalibrationSelection.Text = "Recalibration selection text:";
			// 
			// txtPositionTolerance
			// 
			this.txtPositionTolerance.Location = new System.Drawing.Point(432, 320);
			this.txtPositionTolerance.Name = "txtPositionTolerance";
			this.txtPositionTolerance.Size = new System.Drawing.Size(104, 20);
			this.txtPositionTolerance.TabIndex = 24;
			this.txtPositionTolerance.Text = "";
			this.txtPositionTolerance.Leave += new System.EventHandler(this.txtPositionTolerance_Leave);
			// 
			// txtMinTracks
			// 
			this.txtMinTracks.Location = new System.Drawing.Point(160, 320);
			this.txtMinTracks.Name = "txtMinTracks";
			this.txtMinTracks.TabIndex = 22;
			this.txtMinTracks.Text = "";
			this.txtMinTracks.Leave += new System.EventHandler(this.txtMinTracks_Leave);
			// 
			// txtMinYDistance
			// 
			this.txtMinYDistance.Location = new System.Drawing.Point(432, 288);
			this.txtMinYDistance.Name = "txtMinYDistance";
			this.txtMinYDistance.Size = new System.Drawing.Size(104, 20);
			this.txtMinYDistance.TabIndex = 20;
			this.txtMinYDistance.Text = "";
			this.txtMinYDistance.Leave += new System.EventHandler(this.txtMinYDistance_Leave);
			// 
			// txtMinXDistance
			// 
			this.txtMinXDistance.Location = new System.Drawing.Point(160, 288);
			this.txtMinXDistance.Name = "txtMinXDistance";
			this.txtMinXDistance.TabIndex = 18;
			this.txtMinXDistance.Text = "";
			this.txtMinXDistance.Leave += new System.EventHandler(this.txtMinXDistance_Leave);
			// 
			// lblMinTracks
			// 
			this.lblMinTracks.Location = new System.Drawing.Point(16, 320);
			this.lblMinTracks.Name = "lblMinTracks";
			this.lblMinTracks.Size = new System.Drawing.Size(144, 23);
			this.lblMinTracks.TabIndex = 21;
			this.lblMinTracks.Text = "Min tracks:";
			// 
			// lblPositionTolerance
			// 
			this.lblPositionTolerance.Location = new System.Drawing.Point(280, 320);
			this.lblPositionTolerance.Name = "lblPositionTolerance";
			this.lblPositionTolerance.Size = new System.Drawing.Size(144, 23);
			this.lblPositionTolerance.TabIndex = 23;
			this.lblPositionTolerance.Text = "Position tolerance:";
			// 
			// lblMinXDistance
			// 
			this.lblMinXDistance.Location = new System.Drawing.Point(16, 288);
			this.lblMinXDistance.Name = "lblMinXDistance";
			this.lblMinXDistance.Size = new System.Drawing.Size(128, 23);
			this.lblMinXDistance.TabIndex = 17;
			this.lblMinXDistance.Text = "Min X distance:";
			// 
			// lblMinYDistance
			// 
			this.lblMinYDistance.Location = new System.Drawing.Point(280, 288);
			this.lblMinYDistance.Name = "lblMinYDistance";
			this.lblMinYDistance.Size = new System.Drawing.Size(128, 23);
			this.lblMinYDistance.TabIndex = 19;
			this.lblMinYDistance.Text = "Min Y distance:";
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(16, 352);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(128, 23);
			this.label1.TabIndex = 25;
			this.label1.Text = "Selection function:";
			// 
			// txtSelectionFunction
			// 
			this.txtSelectionFunction.Location = new System.Drawing.Point(160, 352);
			this.txtSelectionFunction.Name = "txtSelectionFunction";
			this.txtSelectionFunction.Size = new System.Drawing.Size(296, 20);
			this.txtSelectionFunction.TabIndex = 26;
			this.txtSelectionFunction.Text = "(DST / 0.003)^2 + (DSL / (0.003 + PSL * 0.3))^2";
			this.txtSelectionFunction.Leave += new System.EventHandler(this.txtSelectionFunction_Leave);
			// 
			// txtMaxSelFVal
			// 
			this.txtMaxSelFVal.Location = new System.Drawing.Point(432, 384);
			this.txtMaxSelFVal.Name = "txtMaxSelFVal";
			this.txtMaxSelFVal.Size = new System.Drawing.Size(104, 20);
			this.txtMaxSelFVal.TabIndex = 31;
			this.txtMaxSelFVal.Text = "";
			this.txtMaxSelFVal.Leave += new System.EventHandler(this.txtMaxSelFVal_Leave);
			// 
			// txtMinSelFVal
			// 
			this.txtMinSelFVal.Location = new System.Drawing.Point(160, 384);
			this.txtMinSelFVal.Name = "txtMinSelFVal";
			this.txtMinSelFVal.TabIndex = 29;
			this.txtMinSelFVal.Text = "";
			this.txtMinSelFVal.Leave += new System.EventHandler(this.txtMinSelFVal_Leave);
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(16, 384);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(128, 23);
			this.label2.TabIndex = 28;
			this.label2.Text = "Min value of sel. func.:";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(276, 384);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(128, 23);
			this.label3.TabIndex = 30;
			this.label3.Text = "Max value of sel. func.:";
			// 
			// btnSyntaxHelp
			// 
			this.btnSyntaxHelp.Location = new System.Drawing.Point(464, 352);
			this.btnSyntaxHelp.Name = "btnSyntaxHelp";
			this.btnSyntaxHelp.TabIndex = 27;
			this.btnSyntaxHelp.Text = "Syntax Help";
			this.btnSyntaxHelp.Click += new System.EventHandler(this.btnSyntaxHelp_Click);
			// 
			// frmConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(544, 558);
			this.Controls.Add(this.btnSyntaxHelp);
			this.Controls.Add(this.txtMaxSelFVal);
			this.Controls.Add(this.txtMinSelFVal);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.txtSelectionFunction);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.txtPositionTolerance);
			this.Controls.Add(this.txtMinTracks);
			this.Controls.Add(this.txtMinYDistance);
			this.Controls.Add(this.txtMinXDistance);
			this.Controls.Add(this.lblMinTracks);
			this.Controls.Add(this.lblPositionTolerance);
			this.Controls.Add(this.lblMinXDistance);
			this.Controls.Add(this.lblMinYDistance);
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
			this.Name = "frmConfig";
			this.Text = "View/edit PredictionScan3Driver config";
			this.ResumeLayout(false);

		}
		#endregion

		private void qualityComboBox_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			_progSettings.QualityCutId = ((Utilities.ConfigItem)qualityComboBox.SelectedItem).Id;
			btnCreate.Enabled = IsFormFilled() & ConfigNameIsValid();
		}

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			btnCreate.Enabled = ConfigNameIsValid() & IsFormFilled();
		}

		private void posTolTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.PositionTolerance = Convert.ToDouble(posTolTextBox.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				posTolTextBox.Focus();
			}
		}

		private void slopeTolTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.SlopeTolerance = Convert.ToDouble(slopeTolTextBox.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				slopeTolTextBox.Focus();
			}
		}

		private void posTolIncTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.PositionToleranceIncreaseWithSlope = Convert.ToDouble(posTolIncTextBox.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				posTolIncTextBox.Focus();
			}
		}

		private void slopeTolIncTextBox_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.SlopeToleranceIncreaseWithSlope = Convert.ToDouble(slopeTolIncTextBox.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				slopeTolIncTextBox.Focus();
			}

		}

		private void maxTrialsTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.MaxTrials = (uint) Convert.ToInt32(maxTrialsTextBox.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				maxTrialsTextBox.Focus();
			}

		}

		private void baseThicknessTextBox_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.BaseThickness = Convert.ToDouble(baseThicknessTextBox.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				baseThicknessTextBox.Focus();
			}

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

		private void btnCreate_Click(object sender, System.EventArgs e)
		{
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(PredictionScan3Settings));
			System.IO.StringWriter sw = new System.IO.StringWriter();
			xmls.Serialize(sw, _progSettings);
			sw.Flush();
			_settingsId = Utilities.WriteSettingsToDb(_connection, configNameTextBox.Text, "PredictionScanDriver.exe", 1, 0, sw.ToString());
			Close();
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

		private void txtRecalibrationSelection_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.RecalibrationSelectionText = txtRecalibrationSelection.Text;
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtRecalibrationSelection.Focus();
			}
		}

		private void txtMinXDistance_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.RecalibrationMinXDistance = Convert.ToDouble(txtMinXDistance.Text, System.Globalization.CultureInfo.InvariantCulture);
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
				_progSettings.RecalibrationMinYDistance = Convert.ToDouble(txtMinYDistance.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinYDistance.Focus();
			}			
		}

		private void txtMinTracks_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.RecalibrationMinTracks = Convert.ToInt32(txtMinTracks.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinTracks.Focus();
			}
		}

		private void txtPositionTolerance_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.RecalibrationPosTolerance = Convert.ToDouble(txtPositionTolerance.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtPositionTolerance.Focus();
			}
		}

		private void txtMinSelFVal_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.SelectionFunctionMin = Convert.ToDouble(txtMinSelFVal.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMinSelFVal.Focus();
			}		
		}

		private void txtMaxSelFVal_Leave(object sender, System.EventArgs e)
		{
			try
			{
				_progSettings.SelectionFunctionMax = Convert.ToDouble(txtMaxSelFVal.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtMaxSelFVal.Focus();
			}				
		}

		private void txtSelectionFunction_Leave(object sender, System.EventArgs e)
		{
			try
			{
				Exe.MakeFunctionAndCheck(_progSettings.SelectionFunction = txtSelectionFunction.Text);
			}
			catch(Exception ex)
			{
				MessageBox.Show(ex.Message);
				txtSelectionFunction.Focus();				
			}
		}

		private void btnSyntaxHelp_Click(object sender, System.EventArgs e)
		{
			MessageBox.Show(Exe.SyntaxHelp(), "Syntax help");		
		}



	}
}
