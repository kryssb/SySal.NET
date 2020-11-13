using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using SySal.OperaDb;
using System.Xml.Serialization;


namespace SySal.Executables.LinkingConfig
{
	/// <summary>
	/// GUI tool to configure options for BatchLink.
	/// </summary>
	/// <remarks><b>NOTICE: This tool does not support the newest options in BatchLink and should be updated.</b></remarks>
	public class frmLinkingConfig : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label configNameLabel;
		private System.Windows.Forms.TextBox configNameTextBox;
		private System.Windows.Forms.Button cancelButton;
		private System.Windows.Forms.Button okButton;
		private System.Windows.Forms.TextBox maskPeakHeightTextBox;
		private System.Windows.Forms.Label maskPeakHeightLabel;
		private System.Windows.Forms.TextBox maskBinningTextBox;
		private System.Windows.Forms.Label topMultSlopeXLabel;
		private System.Windows.Forms.Label topMultSlopeYLabel;
		private System.Windows.Forms.TextBox topMultSlopeYTextBox;
		private System.Windows.Forms.Label botMultSlopeYLabel;
		private System.Windows.Forms.TextBox botMultSlopeYTextBox;
		private System.Windows.Forms.Label botMultSlopeXLabel;
		private System.Windows.Forms.TextBox botMultSlopeXTextBox;
		private System.Windows.Forms.Label topDeltaSlopeXlabel;
		private System.Windows.Forms.TextBox topDeltaSlopeXTextBox;
		private System.Windows.Forms.TextBox botDeltaSlopeXTextBox;
		private System.Windows.Forms.Label botDeltaSlopeXlabel;
		private System.Windows.Forms.TextBox botDeltaSlopeYTextBox;
		private System.Windows.Forms.Label botDeltaSlopeYlabel;
		private System.Windows.Forms.Label topDeltaSlopeYlabel;
		private System.Windows.Forms.TextBox topDeltaSlopeYTextBox;
		private System.Windows.Forms.Label maskBinningLabel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private OperaDbConnection _connection;
		private OperaDbCredentials _credentials;
		private long _configId;
		private BatchLink.Config _batchLinkConfig = new BatchLink.Config();
		private SySal.OperaDb.ComputingInfrastructure.ProgramSettings _programSettings;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TextBox topMultSlopeXTextBox;
		private System.Windows.Forms.Label autoCorrectMaxSlopeLabel;
		private System.Windows.Forms.Label autoCorrectMinSlopeLabel;
		private System.Windows.Forms.TextBox autocorrectMaxSlopeTextBox;
		private System.Windows.Forms.TextBox autocorrectMinSlopeTextBox;
		private System.Windows.Forms.Label autoCorrectMultLabel;
		private System.Windows.Forms.CheckBox autocorrectMultipliersCheckBox;
		private System.Windows.Forms.TextBox minGrainsTextBox;
		private System.Windows.Forms.Label minGrainsLabel;
		private System.Windows.Forms.Label mergeSlopeTolLabel;
		private System.Windows.Forms.TextBox mergeSlopeTolTextBox;
		private System.Windows.Forms.Label slopeTolLabel;
		private System.Windows.Forms.TextBox slopeTolTextBox;
		private System.Windows.Forms.Label minSlopeLabel;
		private System.Windows.Forms.TextBox minSlopeTextBox;
		private System.Windows.Forms.Label mergePosTolLabel;
		private System.Windows.Forms.TextBox mergePosTolTextBox;
		private System.Windows.Forms.Label slopeTolIncWithSlopeLabel;
		private System.Windows.Forms.TextBox slopeTolIncWithSlopeTextBox;
		private System.Windows.Forms.Label memorySavingLabel;
		private System.Windows.Forms.TextBox memorySavingTextBox;
		private string _originalConfigName;
		
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

		public static void Get(ref long id, OperaDbConnection conn)
		{
			frmLinkingConfig form = new frmLinkingConfig(id, conn);
			form.ShowDialog();
			id = form._configId;
		}

		public frmLinkingConfig(long id, OperaDbConnection conn)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			_connection = conn;
			if (id != 0) 
			{
				_programSettings = new SySal.OperaDb.ComputingInfrastructure.ProgramSettings(id, _connection, null);
				System.Xml.Serialization.XmlSerializer xmls = new XmlSerializer(_batchLinkConfig.GetType());				
				_batchLinkConfig = (BatchLink.Config) xmls.Deserialize(new System.IO.StringReader(_programSettings.Settings.Replace("LinkingSettings", "BatchLink.Config")));
				_originalConfigName =  _programSettings.Description;
				configNameTextBox.Text = _programSettings.Description;
			}
			else
			{
				//_batchLinkConfig = new BatchLink.Config();
				_originalConfigName = "";
				configNameTextBox.Text = "";
			}			
			topMultSlopeXTextBox.DataBindings.Add("Text", _batchLinkConfig, "TopMultSlopeX");
			topMultSlopeYTextBox.DataBindings.Add("Text", _batchLinkConfig, "TopMultSlopeY");
			botMultSlopeXTextBox.DataBindings.Add("Text", _batchLinkConfig, "BottomMultSlopeX");
			botMultSlopeYTextBox.DataBindings.Add("Text", _batchLinkConfig, "BottomMultSlopeY");
			topDeltaSlopeXTextBox.DataBindings.Add("Text", _batchLinkConfig, "TopDeltaSlopeX");
			topDeltaSlopeYTextBox.DataBindings.Add("Text", _batchLinkConfig, "TopDeltaSlopeY");
			botDeltaSlopeXTextBox.DataBindings.Add("Text", _batchLinkConfig, "BottomDeltaSlopeX");
			botDeltaSlopeYTextBox.DataBindings.Add("Text", _batchLinkConfig, "BottomDeltaSlopeY");
			maskBinningTextBox.DataBindings.Add("Text", _batchLinkConfig, "MaskBinning");
			maskPeakHeightTextBox.DataBindings.Add("Text", _batchLinkConfig, "MaskPeakHeightMultiplier");
			autocorrectMultipliersCheckBox.DataBindings.Add("Checked", _batchLinkConfig, "AutoCorrectMultipliers");
			autocorrectMinSlopeTextBox.DataBindings.Add("Text", _batchLinkConfig, "AutoCorrectMinSlope");
			autocorrectMaxSlopeTextBox.DataBindings.Add("Text", _batchLinkConfig, "AutoCorrectMaxSlope");
			minGrainsTextBox.DataBindings.Add("Text", _batchLinkConfig, "MinGrains");
			minSlopeTextBox.DataBindings.Add("Text", _batchLinkConfig, "MinSlope");
			mergePosTolTextBox.DataBindings.Add("Text", _batchLinkConfig, "MergePosTol");
			mergeSlopeTolTextBox.DataBindings.Add("Text", _batchLinkConfig, "MergeSlopeTol");
			slopeTolTextBox.DataBindings.Add("Text", _batchLinkConfig, "SlopeTol");
			slopeTolIncWithSlopeTextBox.DataBindings.Add("Text", _batchLinkConfig, "SlopeTolIncreaseWithSlope");
			memorySavingTextBox.DataBindings.Add("Text", _batchLinkConfig, "MemorySaving");
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

		public long Get(long id, OperaDbConnection conn)
		{			
			frmLinkingConfig configForm = new frmLinkingConfig(id, conn);
			configForm.ShowDialog();
			return configForm._configId;
		}

		public static long GetConfig(long id, OperaDbConnection conn)
		{			
			frmLinkingConfig configForm = new frmLinkingConfig(id, conn);
			configForm.ShowDialog();
			return configForm._configId;
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			System.Resources.ResourceManager resources = new System.Resources.ResourceManager(typeof(frmLinkingConfig));
			this.configNameLabel = new System.Windows.Forms.Label();
			this.configNameTextBox = new System.Windows.Forms.TextBox();
			this.cancelButton = new System.Windows.Forms.Button();
			this.okButton = new System.Windows.Forms.Button();
			this.maskPeakHeightTextBox = new System.Windows.Forms.TextBox();
			this.maskPeakHeightLabel = new System.Windows.Forms.Label();
			this.maskBinningTextBox = new System.Windows.Forms.TextBox();
			this.topMultSlopeXLabel = new System.Windows.Forms.Label();
			this.topMultSlopeYLabel = new System.Windows.Forms.Label();
			this.topMultSlopeYTextBox = new System.Windows.Forms.TextBox();
			this.botMultSlopeYLabel = new System.Windows.Forms.Label();
			this.botMultSlopeYTextBox = new System.Windows.Forms.TextBox();
			this.botMultSlopeXLabel = new System.Windows.Forms.Label();
			this.botMultSlopeXTextBox = new System.Windows.Forms.TextBox();
			this.topDeltaSlopeXlabel = new System.Windows.Forms.Label();
			this.topDeltaSlopeXTextBox = new System.Windows.Forms.TextBox();
			this.botDeltaSlopeXTextBox = new System.Windows.Forms.TextBox();
			this.botDeltaSlopeXlabel = new System.Windows.Forms.Label();
			this.botDeltaSlopeYTextBox = new System.Windows.Forms.TextBox();
			this.botDeltaSlopeYlabel = new System.Windows.Forms.Label();
			this.topDeltaSlopeYlabel = new System.Windows.Forms.Label();
			this.topDeltaSlopeYTextBox = new System.Windows.Forms.TextBox();
			this.maskBinningLabel = new System.Windows.Forms.Label();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.topMultSlopeXTextBox = new System.Windows.Forms.TextBox();
			this.autoCorrectMaxSlopeLabel = new System.Windows.Forms.Label();
			this.autoCorrectMinSlopeLabel = new System.Windows.Forms.Label();
			this.autocorrectMaxSlopeTextBox = new System.Windows.Forms.TextBox();
			this.autocorrectMinSlopeTextBox = new System.Windows.Forms.TextBox();
			this.autoCorrectMultLabel = new System.Windows.Forms.Label();
			this.autocorrectMultipliersCheckBox = new System.Windows.Forms.CheckBox();
			this.minGrainsTextBox = new System.Windows.Forms.TextBox();
			this.minGrainsLabel = new System.Windows.Forms.Label();
			this.mergeSlopeTolLabel = new System.Windows.Forms.Label();
			this.mergeSlopeTolTextBox = new System.Windows.Forms.TextBox();
			this.slopeTolLabel = new System.Windows.Forms.Label();
			this.slopeTolTextBox = new System.Windows.Forms.TextBox();
			this.minSlopeLabel = new System.Windows.Forms.Label();
			this.minSlopeTextBox = new System.Windows.Forms.TextBox();
			this.mergePosTolLabel = new System.Windows.Forms.Label();
			this.mergePosTolTextBox = new System.Windows.Forms.TextBox();
			this.slopeTolIncWithSlopeLabel = new System.Windows.Forms.Label();
			this.slopeTolIncWithSlopeTextBox = new System.Windows.Forms.TextBox();
			this.memorySavingLabel = new System.Windows.Forms.Label();
			this.memorySavingTextBox = new System.Windows.Forms.TextBox();
			this.groupBox1.SuspendLayout();
			this.SuspendLayout();
			// 
			// configNameLabel
			// 
			this.configNameLabel.Location = new System.Drawing.Point(48, 16);
			this.configNameLabel.Name = "configNameLabel";
			this.configNameLabel.Size = new System.Drawing.Size(112, 23);
			this.configNameLabel.TabIndex = 44;
			this.configNameLabel.Text = "Configuration name:";
			// 
			// configNameTextBox
			// 
			this.configNameTextBox.Location = new System.Drawing.Point(160, 16);
			this.configNameTextBox.Name = "configNameTextBox";
			this.configNameTextBox.Size = new System.Drawing.Size(256, 20);
			this.configNameTextBox.TabIndex = 43;
			this.configNameTextBox.Text = "";
			this.configNameTextBox.TextChanged += new System.EventHandler(this.configNameTextBox_TextChanged);
			// 
			// cancelButton
			// 
			this.cancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.cancelButton.Location = new System.Drawing.Point(304, 520);
			this.cancelButton.Name = "cancelButton";
			this.cancelButton.Size = new System.Drawing.Size(88, 32);
			this.cancelButton.TabIndex = 42;
			this.cancelButton.Text = "Cancel";
			this.cancelButton.Click += new System.EventHandler(this.cancelButton_Click);
			// 
			// okButton
			// 
			this.okButton.Enabled = false;
			this.okButton.Location = new System.Drawing.Point(48, 520);
			this.okButton.Name = "okButton";
			this.okButton.Size = new System.Drawing.Size(88, 32);
			this.okButton.TabIndex = 41;
			this.okButton.Text = "Create";
			this.okButton.Click += new System.EventHandler(this.okButton_Click);
			// 
			// maskPeakHeightTextBox
			// 
			this.maskPeakHeightTextBox.Location = new System.Drawing.Point(352, 232);
			this.maskPeakHeightTextBox.Name = "maskPeakHeightTextBox";
			this.maskPeakHeightTextBox.Size = new System.Drawing.Size(48, 20);
			this.maskPeakHeightTextBox.TabIndex = 34;
			this.maskPeakHeightTextBox.Text = "";
			// 
			// maskPeakHeightLabel
			// 
			this.maskPeakHeightLabel.Location = new System.Drawing.Point(208, 232);
			this.maskPeakHeightLabel.Name = "maskPeakHeightLabel";
			this.maskPeakHeightLabel.Size = new System.Drawing.Size(152, 23);
			this.maskPeakHeightLabel.TabIndex = 33;
			this.maskPeakHeightLabel.Text = "Mask peak height multiplier:";
			// 
			// maskBinningTextBox
			// 
			this.maskBinningTextBox.Location = new System.Drawing.Point(144, 232);
			this.maskBinningTextBox.Name = "maskBinningTextBox";
			this.maskBinningTextBox.Size = new System.Drawing.Size(56, 20);
			this.maskBinningTextBox.TabIndex = 32;
			this.maskBinningTextBox.Text = "";
			// 
			// topMultSlopeXLabel
			// 
			this.topMultSlopeXLabel.Location = new System.Drawing.Point(40, 72);
			this.topMultSlopeXLabel.Name = "topMultSlopeXLabel";
			this.topMultSlopeXLabel.Size = new System.Drawing.Size(96, 23);
			this.topMultSlopeXLabel.TabIndex = 17;
			this.topMultSlopeXLabel.Text = "Top Mult Slope X:";
			// 
			// topMultSlopeYLabel
			// 
			this.topMultSlopeYLabel.Location = new System.Drawing.Point(232, 72);
			this.topMultSlopeYLabel.Name = "topMultSlopeYLabel";
			this.topMultSlopeYLabel.Size = new System.Drawing.Size(96, 23);
			this.topMultSlopeYLabel.TabIndex = 18;
			this.topMultSlopeYLabel.Text = "Top Mult Slope Y:";
			// 
			// topMultSlopeYTextBox
			// 
			this.topMultSlopeYTextBox.Location = new System.Drawing.Point(336, 72);
			this.topMultSlopeYTextBox.Name = "topMultSlopeYTextBox";
			this.topMultSlopeYTextBox.Size = new System.Drawing.Size(64, 20);
			this.topMultSlopeYTextBox.TabIndex = 26;
			this.topMultSlopeYTextBox.Text = "";
			// 
			// botMultSlopeYLabel
			// 
			this.botMultSlopeYLabel.Location = new System.Drawing.Point(232, 112);
			this.botMultSlopeYLabel.Name = "botMultSlopeYLabel";
			this.botMultSlopeYLabel.Size = new System.Drawing.Size(96, 23);
			this.botMultSlopeYLabel.TabIndex = 15;
			this.botMultSlopeYLabel.Text = "Bot Mult Slope Y:";
			// 
			// botMultSlopeYTextBox
			// 
			this.botMultSlopeYTextBox.Location = new System.Drawing.Point(336, 112);
			this.botMultSlopeYTextBox.Name = "botMultSlopeYTextBox";
			this.botMultSlopeYTextBox.Size = new System.Drawing.Size(64, 20);
			this.botMultSlopeYTextBox.TabIndex = 25;
			this.botMultSlopeYTextBox.Text = "";
			// 
			// botMultSlopeXLabel
			// 
			this.botMultSlopeXLabel.Location = new System.Drawing.Point(40, 112);
			this.botMultSlopeXLabel.Name = "botMultSlopeXLabel";
			this.botMultSlopeXLabel.Size = new System.Drawing.Size(96, 23);
			this.botMultSlopeXLabel.TabIndex = 16;
			this.botMultSlopeXLabel.Text = "Bot Mult Slope X:";
			// 
			// botMultSlopeXTextBox
			// 
			this.botMultSlopeXTextBox.Location = new System.Drawing.Point(144, 112);
			this.botMultSlopeXTextBox.Name = "botMultSlopeXTextBox";
			this.botMultSlopeXTextBox.Size = new System.Drawing.Size(56, 20);
			this.botMultSlopeXTextBox.TabIndex = 27;
			this.botMultSlopeXTextBox.Text = "";
			// 
			// topDeltaSlopeXlabel
			// 
			this.topDeltaSlopeXlabel.Location = new System.Drawing.Point(40, 152);
			this.topDeltaSlopeXlabel.Name = "topDeltaSlopeXlabel";
			this.topDeltaSlopeXlabel.Size = new System.Drawing.Size(104, 23);
			this.topDeltaSlopeXlabel.TabIndex = 19;
			this.topDeltaSlopeXlabel.Text = "Top Delta Slope X:";
			// 
			// topDeltaSlopeXTextBox
			// 
			this.topDeltaSlopeXTextBox.Location = new System.Drawing.Point(144, 152);
			this.topDeltaSlopeXTextBox.Name = "topDeltaSlopeXTextBox";
			this.topDeltaSlopeXTextBox.Size = new System.Drawing.Size(56, 20);
			this.topDeltaSlopeXTextBox.TabIndex = 29;
			this.topDeltaSlopeXTextBox.Text = "";
			// 
			// botDeltaSlopeXTextBox
			// 
			this.botDeltaSlopeXTextBox.Location = new System.Drawing.Point(144, 192);
			this.botDeltaSlopeXTextBox.Name = "botDeltaSlopeXTextBox";
			this.botDeltaSlopeXTextBox.Size = new System.Drawing.Size(56, 20);
			this.botDeltaSlopeXTextBox.TabIndex = 28;
			this.botDeltaSlopeXTextBox.Text = "";
			// 
			// botDeltaSlopeXlabel
			// 
			this.botDeltaSlopeXlabel.Location = new System.Drawing.Point(40, 192);
			this.botDeltaSlopeXlabel.Name = "botDeltaSlopeXlabel";
			this.botDeltaSlopeXlabel.Size = new System.Drawing.Size(96, 23);
			this.botDeltaSlopeXlabel.TabIndex = 22;
			this.botDeltaSlopeXlabel.Text = "Bot Delta Slope X:";
			// 
			// botDeltaSlopeYTextBox
			// 
			this.botDeltaSlopeYTextBox.Location = new System.Drawing.Point(336, 192);
			this.botDeltaSlopeYTextBox.Name = "botDeltaSlopeYTextBox";
			this.botDeltaSlopeYTextBox.Size = new System.Drawing.Size(64, 20);
			this.botDeltaSlopeYTextBox.TabIndex = 30;
			this.botDeltaSlopeYTextBox.Text = "";
			// 
			// botDeltaSlopeYlabel
			// 
			this.botDeltaSlopeYlabel.Location = new System.Drawing.Point(232, 192);
			this.botDeltaSlopeYlabel.Name = "botDeltaSlopeYlabel";
			this.botDeltaSlopeYlabel.Size = new System.Drawing.Size(96, 23);
			this.botDeltaSlopeYlabel.TabIndex = 23;
			this.botDeltaSlopeYlabel.Text = "Bot Delta Slope Y:";
			// 
			// topDeltaSlopeYlabel
			// 
			this.topDeltaSlopeYlabel.Location = new System.Drawing.Point(232, 152);
			this.topDeltaSlopeYlabel.Name = "topDeltaSlopeYlabel";
			this.topDeltaSlopeYlabel.Size = new System.Drawing.Size(104, 23);
			this.topDeltaSlopeYlabel.TabIndex = 20;
			this.topDeltaSlopeYlabel.Text = "Top Delta Slope Y:";
			// 
			// topDeltaSlopeYTextBox
			// 
			this.topDeltaSlopeYTextBox.Location = new System.Drawing.Point(336, 152);
			this.topDeltaSlopeYTextBox.Name = "topDeltaSlopeYTextBox";
			this.topDeltaSlopeYTextBox.Size = new System.Drawing.Size(64, 20);
			this.topDeltaSlopeYTextBox.TabIndex = 24;
			this.topDeltaSlopeYTextBox.Text = "";
			// 
			// maskBinningLabel
			// 
			this.maskBinningLabel.Location = new System.Drawing.Point(40, 232);
			this.maskBinningLabel.Name = "maskBinningLabel";
			this.maskBinningLabel.Size = new System.Drawing.Size(96, 23);
			this.maskBinningLabel.TabIndex = 21;
			this.maskBinningLabel.Text = "Mask binning:";
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.memorySavingTextBox);
			this.groupBox1.Controls.Add(this.memorySavingLabel);
			this.groupBox1.Controls.Add(this.minGrainsTextBox);
			this.groupBox1.Controls.Add(this.minGrainsLabel);
			this.groupBox1.Controls.Add(this.mergeSlopeTolLabel);
			this.groupBox1.Controls.Add(this.mergeSlopeTolTextBox);
			this.groupBox1.Controls.Add(this.slopeTolLabel);
			this.groupBox1.Controls.Add(this.slopeTolTextBox);
			this.groupBox1.Controls.Add(this.minSlopeLabel);
			this.groupBox1.Controls.Add(this.minSlopeTextBox);
			this.groupBox1.Controls.Add(this.mergePosTolLabel);
			this.groupBox1.Controls.Add(this.mergePosTolTextBox);
			this.groupBox1.Controls.Add(this.slopeTolIncWithSlopeLabel);
			this.groupBox1.Controls.Add(this.slopeTolIncWithSlopeTextBox);
			this.groupBox1.Location = new System.Drawing.Point(24, 352);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(408, 160);
			this.groupBox1.TabIndex = 45;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Stripes Linker Config";
			// 
			// topMultSlopeXTextBox
			// 
			this.topMultSlopeXTextBox.Location = new System.Drawing.Point(144, 72);
			this.topMultSlopeXTextBox.Name = "topMultSlopeXTextBox";
			this.topMultSlopeXTextBox.Size = new System.Drawing.Size(56, 20);
			this.topMultSlopeXTextBox.TabIndex = 31;
			this.topMultSlopeXTextBox.Text = "";
			// 
			// autoCorrectMaxSlopeLabel
			// 
			this.autoCorrectMaxSlopeLabel.Location = new System.Drawing.Point(224, 312);
			this.autoCorrectMaxSlopeLabel.Name = "autoCorrectMaxSlopeLabel";
			this.autoCorrectMaxSlopeLabel.Size = new System.Drawing.Size(128, 23);
			this.autoCorrectMaxSlopeLabel.TabIndex = 51;
			this.autoCorrectMaxSlopeLabel.Text = "AutoCorrect max slope:";
			// 
			// autoCorrectMinSlopeLabel
			// 
			this.autoCorrectMinSlopeLabel.Location = new System.Drawing.Point(40, 312);
			this.autoCorrectMinSlopeLabel.Name = "autoCorrectMinSlopeLabel";
			this.autoCorrectMinSlopeLabel.Size = new System.Drawing.Size(120, 23);
			this.autoCorrectMinSlopeLabel.TabIndex = 50;
			this.autoCorrectMinSlopeLabel.Text = "AutoCorrect min slope:";
			// 
			// autocorrectMaxSlopeTextBox
			// 
			this.autocorrectMaxSlopeTextBox.Location = new System.Drawing.Point(360, 312);
			this.autocorrectMaxSlopeTextBox.Name = "autocorrectMaxSlopeTextBox";
			this.autocorrectMaxSlopeTextBox.Size = new System.Drawing.Size(32, 20);
			this.autocorrectMaxSlopeTextBox.TabIndex = 49;
			this.autocorrectMaxSlopeTextBox.Text = "";
			// 
			// autocorrectMinSlopeTextBox
			// 
			this.autocorrectMinSlopeTextBox.Location = new System.Drawing.Point(176, 312);
			this.autocorrectMinSlopeTextBox.Name = "autocorrectMinSlopeTextBox";
			this.autocorrectMinSlopeTextBox.Size = new System.Drawing.Size(32, 20);
			this.autocorrectMinSlopeTextBox.TabIndex = 48;
			this.autocorrectMinSlopeTextBox.Text = "";
			// 
			// autoCorrectMultLabel
			// 
			this.autoCorrectMultLabel.Location = new System.Drawing.Point(216, 272);
			this.autoCorrectMultLabel.Name = "autoCorrectMultLabel";
			this.autoCorrectMultLabel.Size = new System.Drawing.Size(120, 23);
			this.autoCorrectMultLabel.TabIndex = 47;
			this.autoCorrectMultLabel.Text = "AutoCorrect multipliers";
			// 
			// autocorrectMultipliersCheckBox
			// 
			this.autocorrectMultipliersCheckBox.Location = new System.Drawing.Point(184, 272);
			this.autocorrectMultipliersCheckBox.Name = "autocorrectMultipliersCheckBox";
			this.autocorrectMultipliersCheckBox.Size = new System.Drawing.Size(16, 16);
			this.autocorrectMultipliersCheckBox.TabIndex = 46;
			this.autocorrectMultipliersCheckBox.Text = "checkBox1";
			// 
			// minGrainsTextBox
			// 
			this.minGrainsTextBox.Location = new System.Drawing.Point(128, 17);
			this.minGrainsTextBox.Name = "minGrainsTextBox";
			this.minGrainsTextBox.Size = new System.Drawing.Size(56, 20);
			this.minGrainsTextBox.TabIndex = 43;
			this.minGrainsTextBox.Text = "";
			// 
			// minGrainsLabel
			// 
			this.minGrainsLabel.Location = new System.Drawing.Point(24, 17);
			this.minGrainsLabel.Name = "minGrainsLabel";
			this.minGrainsLabel.Size = new System.Drawing.Size(96, 23);
			this.minGrainsLabel.TabIndex = 34;
			this.minGrainsLabel.Text = "Min Grains";
			// 
			// mergeSlopeTolLabel
			// 
			this.mergeSlopeTolLabel.Location = new System.Drawing.Point(216, 17);
			this.mergeSlopeTolLabel.Name = "mergeSlopeTolLabel";
			this.mergeSlopeTolLabel.Size = new System.Drawing.Size(96, 23);
			this.mergeSlopeTolLabel.TabIndex = 35;
			this.mergeSlopeTolLabel.Text = "Merge Slope Tol";
			// 
			// mergeSlopeTolTextBox
			// 
			this.mergeSlopeTolTextBox.Location = new System.Drawing.Point(320, 17);
			this.mergeSlopeTolTextBox.Name = "mergeSlopeTolTextBox";
			this.mergeSlopeTolTextBox.Size = new System.Drawing.Size(64, 20);
			this.mergeSlopeTolTextBox.TabIndex = 40;
			this.mergeSlopeTolTextBox.Text = "";
			// 
			// slopeTolLabel
			// 
			this.slopeTolLabel.Location = new System.Drawing.Point(216, 56);
			this.slopeTolLabel.Name = "slopeTolLabel";
			this.slopeTolLabel.Size = new System.Drawing.Size(64, 23);
			this.slopeTolLabel.TabIndex = 32;
			this.slopeTolLabel.Text = "Slope Tol";
			// 
			// slopeTolTextBox
			// 
			this.slopeTolTextBox.Location = new System.Drawing.Point(320, 57);
			this.slopeTolTextBox.Name = "slopeTolTextBox";
			this.slopeTolTextBox.Size = new System.Drawing.Size(64, 20);
			this.slopeTolTextBox.TabIndex = 39;
			this.slopeTolTextBox.Text = "";
			// 
			// minSlopeLabel
			// 
			this.minSlopeLabel.Location = new System.Drawing.Point(24, 57);
			this.minSlopeLabel.Name = "minSlopeLabel";
			this.minSlopeLabel.Size = new System.Drawing.Size(96, 23);
			this.minSlopeLabel.TabIndex = 33;
			this.minSlopeLabel.Text = "Min Slope";
			// 
			// minSlopeTextBox
			// 
			this.minSlopeTextBox.Location = new System.Drawing.Point(128, 57);
			this.minSlopeTextBox.Name = "minSlopeTextBox";
			this.minSlopeTextBox.Size = new System.Drawing.Size(56, 20);
			this.minSlopeTextBox.TabIndex = 41;
			this.minSlopeTextBox.Text = "";
			// 
			// mergePosTolLabel
			// 
			this.mergePosTolLabel.Location = new System.Drawing.Point(24, 97);
			this.mergePosTolLabel.Name = "mergePosTolLabel";
			this.mergePosTolLabel.Size = new System.Drawing.Size(104, 23);
			this.mergePosTolLabel.TabIndex = 36;
			this.mergePosTolLabel.Text = "Merge Pos Tol";
			// 
			// mergePosTolTextBox
			// 
			this.mergePosTolTextBox.Location = new System.Drawing.Point(128, 97);
			this.mergePosTolTextBox.Name = "mergePosTolTextBox";
			this.mergePosTolTextBox.Size = new System.Drawing.Size(56, 20);
			this.mergePosTolTextBox.TabIndex = 42;
			this.mergePosTolTextBox.Text = "";
			// 
			// slopeTolIncWithSlopeLabel
			// 
			this.slopeTolIncWithSlopeLabel.Location = new System.Drawing.Point(192, 97);
			this.slopeTolIncWithSlopeLabel.Name = "slopeTolIncWithSlopeLabel";
			this.slopeTolIncWithSlopeLabel.Size = new System.Drawing.Size(120, 23);
			this.slopeTolIncWithSlopeLabel.TabIndex = 37;
			this.slopeTolIncWithSlopeLabel.Text = "SlopeTolIncWithSlope";
			// 
			// slopeTolIncWithSlopeTextBox
			// 
			this.slopeTolIncWithSlopeTextBox.Location = new System.Drawing.Point(320, 97);
			this.slopeTolIncWithSlopeTextBox.Name = "slopeTolIncWithSlopeTextBox";
			this.slopeTolIncWithSlopeTextBox.Size = new System.Drawing.Size(64, 20);
			this.slopeTolIncWithSlopeTextBox.TabIndex = 38;
			this.slopeTolIncWithSlopeTextBox.Text = "";
			// 
			// memorySavingLabel
			// 
			this.memorySavingLabel.Location = new System.Drawing.Point(104, 128);
			this.memorySavingLabel.Name = "memorySavingLabel";
			this.memorySavingLabel.Size = new System.Drawing.Size(120, 23);
			this.memorySavingLabel.TabIndex = 49;
			this.memorySavingLabel.Text = "Memory Saving";
			// 
			// memorySavingTextBox
			// 
			this.memorySavingTextBox.Location = new System.Drawing.Point(240, 128);
			this.memorySavingTextBox.Name = "memorySavingTextBox";
			this.memorySavingTextBox.Size = new System.Drawing.Size(56, 20);
			this.memorySavingTextBox.TabIndex = 50;
			this.memorySavingTextBox.Text = "";
			// 
			// frmLinkingConfig
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(440, 558);
			this.Controls.Add(this.autoCorrectMaxSlopeLabel);
			this.Controls.Add(this.autoCorrectMinSlopeLabel);
			this.Controls.Add(this.autocorrectMaxSlopeTextBox);
			this.Controls.Add(this.autocorrectMinSlopeTextBox);
			this.Controls.Add(this.autoCorrectMultLabel);
			this.Controls.Add(this.autocorrectMultipliersCheckBox);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.configNameLabel);
			this.Controls.Add(this.configNameTextBox);
			this.Controls.Add(this.cancelButton);
			this.Controls.Add(this.okButton);
			this.Controls.Add(this.maskPeakHeightTextBox);
			this.Controls.Add(this.maskPeakHeightLabel);
			this.Controls.Add(this.maskBinningTextBox);
			this.Controls.Add(this.topMultSlopeXTextBox);
			this.Controls.Add(this.topMultSlopeXLabel);
			this.Controls.Add(this.topMultSlopeYLabel);
			this.Controls.Add(this.topMultSlopeYTextBox);
			this.Controls.Add(this.botMultSlopeYLabel);
			this.Controls.Add(this.botMultSlopeYTextBox);
			this.Controls.Add(this.botMultSlopeXLabel);
			this.Controls.Add(this.botMultSlopeXTextBox);
			this.Controls.Add(this.topDeltaSlopeXlabel);
			this.Controls.Add(this.topDeltaSlopeXTextBox);
			this.Controls.Add(this.botDeltaSlopeXTextBox);
			this.Controls.Add(this.botDeltaSlopeXlabel);
			this.Controls.Add(this.botDeltaSlopeYTextBox);
			this.Controls.Add(this.botDeltaSlopeYlabel);
			this.Controls.Add(this.topDeltaSlopeYlabel);
			this.Controls.Add(this.topDeltaSlopeYTextBox);
			this.Controls.Add(this.maskBinningLabel);
			this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
			this.Name = "frmLinkingConfig";
			this.Text = "BatchLink configuration";
			this.groupBox1.ResumeLayout(false);
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
			/*frmLinkingConfig configForm = new frmLinkingConfig((long)5e15+600107, conn);
			configForm.ShowDialog();*/
			
			//long id = (new frmLinkingConfig((long)5e15+100012, conn)).Get((long)5e15+100012, conn);
			long id = (new frmLinkingConfig(0, conn)).Get(0, conn);

		//	return configForm._configId;
		
			//Application.Run(new frmLinkingConfig((long)5e15+600107, conn));			
			//Get((long)5e15+600107, conn);
		}

		private void cancelButton_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void okButton_Click(object sender, System.EventArgs e)
		{
			_configId = WriteToDb(configNameTextBox.Text, "BatchLink.exe", 0, 0, _batchLinkConfig.ToXml());
			if (_configId != 0) MessageBox.Show("Created linking config with id " + _configId);
			Close();
		}

		private void configNameTextBox_TextChanged(object sender, System.EventArgs e)
		{
			okButton.Enabled = (configNameTextBox.Text.Trim() !=_originalConfigName.Trim() &&
					configNameTextBox.Text.Trim() != ""
				);
		}
	}
}
