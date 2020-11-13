using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using SySal.Processing.AlphaOmegaReconstruction;
using NumericalTools;
using SySal.Processing.VolumeGeneration;
using SySal.OperaDb;
using SySal.TotalScan;

namespace SySal.Executables.EasyReconstruct
{
	/// <summary>
	/// EasyReconstruct - GUI tool to perform reconstructions by AlphaOmegaReconstruction.
	/// </summary>
	/// <remarks>
	/// <para>EasyReconstruct can work on already reconstructed TSR files, or on data that come from files/DB.</para>
	/// <para>To load a TSR file, press the Load button in the Reconstruction Input File group.</para>
	/// <para>    
	/// To load data from an ASCII list file that specifies the zones, do the following steps:
	/// <list type="bullet">
	/// <item><term>select the file and load the list by clicking on the Load List button</term></item>
	/// <item><term>change/reset/set the selection according to your needs</term></item>
	/// <item><term>click on Load Zones to actually load the set of zones.</term></item>    
	/// </list>    
	/// If you want to save your selection, you can do by clicking on the Save List File button.
	/// The selection string is a mathematical expression that deselects a track when its value is zero, and selects the track otherwise. The available quantities are shown in the combobox adjacent to the selection text. Click on the Insert button to insert the variable selected in the combobox, or type it manually.
	/// </para>
    /// <para>The list file should carry be formatted as follows:
    /// <list type="bullet">
    /// <item><term>On the first line, either <c>FileSystem</c> or <c>Database</c> must appear, to indicate which is the data source.</term></item>
    /// <item><term>Zones must be listed in order of decreasing Z, in lines following the first.</term></item>
    /// <item><term>Each line must contain the following fields (those in brackets are optional): <c>zone_file_path|dbzone Z [sheet_id] [alignment_ignore_list_file_path]</c>. If data come from files, <c>zone_file_path</c> is the full path to each TLG file.
    /// If the source is a DB, dbzone is a string like <c>8\35</c> (meaning brick 8, zoneid = 35). The <c>alignment_ignore_list_file_path</c> is an optional file that contains the list of Ids of tracks to be ignored in alignment.
    /// If the extension of the path is ".TLG", the list is extracted from the BaseTrackIgnoreAlignment section of the TLG; if the extension is not ".TLG", the file is assumed to be 
    /// an ASCII file, with word <c>Index</c> on the first line, and zero-based Ids of tracks to be ignored during alignment computation on the following lines (one per each line).</term></item>
    /// <item><term>The last line must not be empty. It can optionally contain a text beginning with <c>Selection:</c> and followed by a selection string to select/deselect tracks for volume reconstruction.</term></item>
    /// </list>
    /// <example><code>
    /// FileSystem
    /// c:\mydata\plate_1.tlg 
    /// </code>
    /// </example>
    /// </para>
	/// <para>Loading data can take a long time. In the meanwhile, you can load/edit/save configurations by the associated buttons.</para>
	/// <para>Output results can be saved to a TSR file by clicking on the Save button of the Reconstruction Output File group.</para>
	/// <para>
	/// The action buttons are described as follows:
	/// <list type="table">
	/// <listheader><term>Button</term><description>Action</description></listheader>
	/// <item><term>Process</term><description>processes data from scratch, performing alignment, track connection and topological/vertexing analysis, as defined by the configuration. If you want to reprocess data that have been already processed, reload the zones or the original TSR file.</description></item>
	/// <item><term>Vertexing</term><description>performs topological reconstruction only.</description></item>    
	/// <item><term>Stop</term><description>interrupts the current processing operation. This command might take some time to execute.</description></item>
	/// <item><term>Show data</term><description>shows alignment data for each pair of adjacent plates. The fits are obtained from as many segments as defined in the Fitting Segments box, if available. An <see cref="SySal.Executables.EasyReconstruct.AnalysisForm">Analysis Form</see> is opened.</description></item>
	/// <item><term>Clear report</term><description>clears the log panel.</description></item>
    /// <item><term>Toggle Info</term><description>shows/hides the right panel with internal information about execution.</description></item>
	/// <item><term>Efficiency</term><description>starts a computation module that computes the efficiency. <b>This function should be considered obsolete.</b></description></item>
	/// <item><term>Display</term><description>opens a <see cref="SySal.Executables.EasyReconstruct.DisplayForm">DisplayGDI form</see> to show tracks. Only the tracks that pass the selection defined by the box on the left (if not empty) are used for display. If the box is empty, all tracks are used.</description></item>
	/// </list>
	/// </para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		private enum DataMode : int {Files=0, Database=1}

		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Button cmdLoad;
		private System.Windows.Forms.Button cmdInput;
		private System.Windows.Forms.TextBox txtFileList;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.TextBox txtConfigFile;
		private System.Windows.Forms.Button cmdLoadConfig;
		private System.Windows.Forms.Button cmdInputConfig;
        private System.Windows.Forms.Button cmdEditConfig;
        private IContainer components;
		private System.Windows.Forms.Button cmdSave;
		private System.Windows.Forms.TextBox txtSaveConfigFile;
		private System.Windows.Forms.Button cmdSaveConfig;
		private System.Windows.Forms.Button cmdLoadFiles;


		/// <summary>
		/// Local Variables
		/// </summary>
		private SySal.Processing.AlphaOmegaReconstruction.AlphaOmegaReconstructor AORec = new AlphaOmegaReconstructor();
        private SySal.Processing.MCSLikelihood.MomentumEstimator MCSLikelihood = new SySal.Processing.MCSLikelihood.MomentumEstimator();
		private SySal.TotalScan.Volume v;
		private SySal.Processing.AlphaOmegaReconstruction.AlignmentData[] AORecAlignData = new SySal.Processing.AlphaOmegaReconstruction.AlignmentData[0];
		
		private string[] InputFileList;
		private double[] InputZCoordList;
		private int[] InputIdList;
        private long[] InputBrickList;
        private string[] InputAlignIgnoreList;

		private System.Windows.Forms.ProgressBar progressBar1;
		private System.Windows.Forms.CheckBox chkSelection;
		private System.Windows.Forms.TextBox txtSelection;
		private System.Windows.Forms.ComboBox cmbSelection;
		private System.Windows.Forms.Button cmdSelection;
		private System.Windows.Forms.TextBox txtFileListSelection;
		private System.Windows.Forms.Button cmdSaveSelection;
		private System.Windows.Forms.Button cmdSaveList;
		private System.Windows.Forms.Button cmdProcess;
		private System.Windows.Forms.Button cmdShowData;
		private System.Windows.Forms.ComboBox cmbAlignedSheet;
		private System.Windows.Forms.Button cmdStop;
		private bool StopThread = false;
		private System.Windows.Forms.RichTextBox rtxtReport;
		private System.Windows.Forms.Button cmdClearReport;
		private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Button cmdEditMCSMomentumConfig;
		private System.Windows.Forms.Button cmdOutputMCSMomentumConfig;
		private System.Windows.Forms.TextBox txtSaveMCSMomentumConfigFile;
		private System.Windows.Forms.Button cmdSaveMCSMomentumConfig;
		private System.Windows.Forms.TextBox txtMCSMomentumConfigFile;
		private System.Windows.Forms.Button cmdLoadMCSMomentumConfig;
		private System.Windows.Forms.Button cmdInputMCSMomentumConfig;
        private System.Windows.Forms.Button cmdEfficiency;
		private int LayersAdded = 0;
		private DataMode inmode;
		private DataMode outmode;
		public string databasename;
		public string username;
		public string pwd;
		private System.Windows.Forms.Button cmdDBSelect;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.Button cmdSaveOutputFile;
		private System.Windows.Forms.Button cmdOutputFile;
		private System.Windows.Forms.TextBox txtRecOutputFile;
		private System.Windows.Forms.GroupBox groupBox5;
		private System.Windows.Forms.TextBox txtRecInputFile;
		private System.Windows.Forms.Button cmdLoadRec;
		private System.Windows.Forms.Button button2;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox txtFitSeg;
		private System.Windows.Forms.Button cmdVertexRec;
		private System.Windows.Forms.TextBox txtShowSelection;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.ComboBox cmbShowSelection;
		private System.Windows.Forms.Button DisplayButton;
        private GroupBox groupBoxInfoPanel;
        private Button cmdXpRefresh;
        private ListView lvExposedInfo;
        private Timer XpRefreshTimer;
        private ColumnHeader columnHeader1;
        private Button SetCommentButton;
        private Label label3;
        private TextBox txtTrackComment;
        private Button cmdMCSMomentum;
        private TextBox txtTSRDSName;
        private Label label4;
        private TextBox txtTSRDSBrick;
        private Button btnBrowseDB;
        private ListBox lvImportedInfo;
        private Button btnImport;
        private Button btnReset;
        private CheckBox chkSetCSZ;
        private CheckBox chkSetBrickDownZ;
        private CheckBox chkNormBrick;
        private Button SaveImportedInfoButton;
        private Button btnOpenFile;
        private Button btnFileFormats;
        private ComboBox cmbMCSAlgo;
        private ComboBox cmbEvents;
        private Label label5;
        private CheckBox chkResetSlopeCorrections;
        private ComboBox cmbTrackExtrapolationMode;
        private Label label6;
        private ComboBox cmbVtxTrackWeighting;
        private Label label7;
        private TabControl tabControl1;
        private TabPage tabPage1;
        private TabPage tabPage2;
        private TabPage tabPage3;
        private Button btnVolMergeConfig;
        private Button btnMapMergeFilterVars;
        private TextBox txtResetDS;
        private Label label9;
        private TextBox txtImportedDS;
        private Label label8;
        private ComboBox cmbMapMergeFilter;
        private Button btnFilterDel;
        private Button btnFilterAdd;
        private TreeView clAvailableInfo;
		SySal.OperaDb.OperaDbConnection dbconn;

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
            Application.EnableVisualStyles();
			Application.Run(new MainForm());
		}

		public void Progress(double percent)
		{
			progressBar1.Value = (int)(percent*100);
		}

		public void Report(string textstring)
		{
			rtxtReport.AppendText(textstring);
		}

		public bool ShouldStop()
		{
			return StopThread;
		}

		private void cmdInputConfig_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog of = new OpenFileDialog();
			of.CheckFileExists = true;
			of.Filter = "xml files (*.xml)|*.xml|All files (*.*)|*.*";
			of.FilterIndex = 0;
			if (of.ShowDialog() == DialogResult.OK)
				txtConfigFile.Text = of.FileName;
		
		}

		private void cmdLoadConfig_Click(object sender, System.EventArgs e)
		{
			System.IO.FileStream f = null;
			try
			{
				f = new System.IO.FileStream(txtConfigFile.Text, System.IO.FileMode.Open, System.IO.FileAccess.Read);
				System.Runtime.Serialization.Formatters.Soap.SoapFormatter fmt = new System.Runtime.Serialization.Formatters.Soap.SoapFormatter();
				AORec.Config = (SySal.Processing.AlphaOmegaReconstruction.Configuration)fmt.Deserialize(f);
				f.Close();
			}
			catch(Exception exc)
			{
				if (f != null) f.Close();
				System.Windows.Forms.MessageBox.Show("Load Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void cmdEditConfig_Click(object sender, System.EventArgs e)
		{
			SySal.Management.Configuration tmpc = (SySal.Management.Configuration)AORec.Config.Clone();
			if (AORec.EditConfiguration(ref tmpc)) AORec.Config = (SySal.Management.Configuration)tmpc.Clone();
		}

		private void cmdSave_Click(object sender, System.EventArgs e)
		{
			System.IO.FileStream f;
			try
			{
				System.Runtime.Serialization.Formatters.Soap.SoapFormatter fmt = new System.Runtime.Serialization.Formatters.Soap.SoapFormatter();
				f = new System.IO.FileStream(txtSaveConfigFile.Text, System.IO.FileMode.Create, System.IO.FileAccess.Write);
				fmt.Serialize(f, AORec.Config);
				f.Flush();
				f.Close();
			}
			catch(Exception exc)
			{
				//f.Close();
				System.Windows.Forms.MessageBox.Show("Save Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void cmdSaveConfig_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sf = new SaveFileDialog();
			sf.Filter = "xml files (*.xml)|*.xml|All files (*.*)|*.*";
			sf.FilterIndex = 0;
			sf.CheckPathExists = true;
			sf.OverwritePrompt = true;
			if (sf.ShowDialog() == DialogResult.OK)
				txtSaveConfigFile.Text = sf.FileName;
		
		}

		private void cmdInput_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog of = new OpenFileDialog();
			of.CheckFileExists = true;
			of.Filter = "text document files (*.txt)|*.txt|All files (*.*)|*.*";
			of.FilterIndex = 0;
			if (of.ShowDialog() == DialogResult.OK)
				txtFileList.Text = of.FileName;
		
		}

		private void cmdLoad_Click(object sender, System.EventArgs e)
		{
			
			ArrayList inputfilelist = new ArrayList();
			ArrayList inputzcoordlist = new ArrayList();
            ArrayList inputbricklist = new ArrayList();
			ArrayList inputidlist = new ArrayList();
            ArrayList alignignorelist = new ArrayList();
			string [] tokens;
			string line;
			string dbname = "";
			System.IO.StreamReader r = null;
			this.chkSelection.Checked = false;
			this.txtSelection.Text = "";
			int i=-1;
			try
			{
				r = new System.IO.StreamReader(txtFileList.Text);
				tokens = ManageSpaces(r.ReadLine().Trim()).Split(' ','\t');
				if(tokens[0].ToLower() == "database")
				{
					inmode = DataMode.Database;
					dbname = tokens[1];
				}
				else if (tokens[0].ToLower() == "filesystem") inmode = DataMode.Files;
				else throw new Exception("Wrong File Format.\r\n First line must contain DataMode (Filesystem or Database)");
				while ((line = r.ReadLine()) != null)
				{
					tokens = ManageSpaces(line.Trim()).Split(' ','\t');
					if(tokens[0].ToLower() == "selection:")
					{
						txtSelection.Text = ManageSpaces(line.Remove(0,10).Trim());
						chkSelection.Checked = true;
					}
					else
					{
						int n = tokens.Length; 
						if (n!=2 && n != 4 && n != 5) 
						{
							throw new Exception("Wrong File Format.	\r\n File Format must be: filepath, long_coord (desc. order) [, sheet id] [,file with list of tracks to be ignored in alignment]. \r\n At the end (optional): Selection: [Math Expression]");
						}
						else
						{
							inputfilelist.Add(tokens[0]);
							inputzcoordlist.Add(System.Convert.ToDouble(tokens[1]));
                            if (n == 5)
                            {
                                inputbricklist.Add(System.Convert.ToInt64(tokens[2]));
                                inputidlist.Add(System.Convert.ToInt32(tokens[3]));
                                alignignorelist.Add(tokens[4]);
                            }
                            else if (n == 4)
                            {
                                try
                                {
                                    inputbricklist.Add(System.Convert.ToInt64(tokens[2]));
                                    inputidlist.Add(System.Convert.ToInt32(tokens[3]));
                                    alignignorelist.Add(null);
                                }
                                catch (Exception)
                                {
                                    inputbricklist.Add(0);
                                    inputidlist.Add(++i);
                                    alignignorelist.Add(tokens[2]);
                                }
                            }
                            else
                            {
                                inputbricklist.Add(0);
                                inputidlist.Add(++i);
                                alignignorelist.Add(null);
                            };
						};
					};
				}
				InputFileList = (string[])inputfilelist.ToArray(typeof(string));
				InputZCoordList = (double[])inputzcoordlist.ToArray(typeof(double));
                InputBrickList = (long[])inputbricklist.ToArray(typeof(long));
				InputIdList = (int[])inputidlist.ToArray(typeof(int));
                InputAlignIgnoreList = (string[])alignignorelist.ToArray(typeof(string));
				r.Close();
				if(inmode == DataMode.Database)
				{
					EasyReconstruct.DBAccessForm dbfrm = new EasyReconstruct.DBAccessForm();
					dbfrm.feed(dbname);
					dbfrm.ShowDialog();
					if (dbfrm.DialogResult == DialogResult.OK)
					{
						databasename = dbfrm.dbname;
						username = dbfrm.userid;
						pwd = dbfrm.pwd;

						dbconn = new SySal.OperaDb.OperaDbConnection(databasename, username, pwd);
						dbconn.Open();
						dbconn.Close();
					};
				}
			}
			catch(Exception exc)
			{
				if(inmode == DataMode.Database)
				{
					dbconn.Close();
				}
				else if(inmode == DataMode.Files)
				{
					r.Close();
				}
				System.Windows.Forms.MessageBox.Show("Load Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			v = null;
			GC.Collect();

		}

		private string ManageSpaces(string s)
		{	
			int newlen, len;
			string n;
			newlen = s.Length;
			do
			{
				len = newlen;
				n = s.Replace("  ", " ");
				newlen = n.Length;
				s = n;
			}
			while (newlen < len);
			return s;
		}

		private void cmdLoadFiles_Click(object sender, System.EventArgs e)
		{
			v = null;
			GC.Collect();
			new dLoadFiles(this.LoadFiles).BeginInvoke(null, null);
		}

		private void LoadFiles()
		{
			LayersAdded = 0;
			int i, n = InputFileList.Length;
			int j, m, k;
			double[] dz = new double[2];
			SySal.BasicTypes.Vector c = new SySal.BasicTypes.Vector();
			SySal.Tracking.MIPEmulsionTrackInfo tmpTR = new SySal.Tracking.MIPEmulsionTrackInfo();
			SySal.Tracking.MIPEmulsionTrackInfo tmpTRtop = new SySal.Tracking.MIPEmulsionTrackInfo();
			SySal.Tracking.MIPEmulsionTrackInfo tmpTRbot = new SySal.Tracking.MIPEmulsionTrackInfo();
			System.IO.FileStream file = null;

			bool applysel = chkSelection.Checked;
			int npar;
			Function f;
			bool chk = true;

			if(applysel)
			{
				f = new CStyleParsedFunction(txtSelection.Text.ToLower());
				npar = f.ParameterList.Length;
			}
			else
			{
				f = null;
				npar = 0;
			};

			try
			{
                int[] remaplist = null;
				rtxtReport.Text = "Loading\r\n";
				AORec.Clear();
				if (n==0) throw new Exception("File List is empty.");
				progressBar1.Maximum = n;

				if(inmode == DataMode.Database) dbconn.Open();

				for (i=0; i<n; i++)
				{
					SySal.Scanning.Plate.IO.OPERA.LinkedZone SySalInput;
					if (inmode == DataMode.Files) 
					{
						//SySalInput = new SySal.Scanning.Plate.IO.OPERA.LinkedZone(file = new System.IO.FileStream(InputFileList[i], System.IO.FileMode.Open, System.IO.FileAccess.Read));
                        SySalInput = SySal.DataStreams.OPERALinkedZone.FromFile(InputFileList[i]);
					}
					else
					{
						SySalInput = new SySal.OperaDb.Scanning.LinkedZone(dbconn, null, System.Convert.ToInt64(InputFileList[i].Substring(0, InputFileList[i].IndexOf('\\'))), System.Convert.ToInt64(InputFileList[i].Substring(InputFileList[i].IndexOf('\\') + 1)), SySal.OperaDb.Scanning.LinkedZone.DetailLevel.BaseFull);						
					}
                    c.Z = InputZCoordList[i];
					m = SySalInput.Length;
                    if (applysel) remaplist = new int[m];
					//SySal.Tracking.MIPEmulsionTrackInfo[] up = new SySal.Tracking.MIPEmulsionTrackInfo[m];
					rtxtReport.Text += "File #" + (i+1) + " (Brick/Id=" + InputBrickList[i] + "/" + InputIdList[i] + "): " + m + " tracks loaded\r\n";
					ArrayList ArrTr = new ArrayList();
					
					double[] zcor = new double[m]; 
					for(j = 0; j < m; j++)
						zcor[j] = SySalInput[j].Info.Intercept.Z;

					double zmean = NumericalTools.Fitting.Average(zcor);
					double dgap;
					for(j = 0; j < m; j++)
					{
						tmpTR = SySalInput[j].Info;
						dgap = zmean - tmpTR.Intercept.Z;
						tmpTR.Intercept.Z = zmean;
						tmpTR.Intercept.X += tmpTR.Slope.X*dgap;
						tmpTR.Intercept.Y += tmpTR.Slope.Y*dgap;
						tmpTR.TopZ += dgap;
						tmpTR.BottomZ += dgap;

						tmpTR.Intercept.Z = InputZCoordList[i];
						double tmptopz = tmpTR.TopZ;
						double tmpbotz = tmpTR.BottomZ;
						dgap = zmean - tmptopz;
						tmpTR.TopZ = InputZCoordList[i] - dgap; // - dz[0];
						//tmpTR.BottomZ = InputZCoordList[i] - 200; //- dz[1]; 
						//tmpTR.BottomZ = InputZCoordList[i] -(SySalInput.Top.BottomZ - SySalInput.Bottom.TopZ); //- dz[1]; 
						tmpTR.BottomZ = InputZCoordList[i] - (tmptopz - tmpbotz) - dgap; //- dz[1]; 

						if(applysel)
						{
							//tmpTRtop = new SySal.Tracking.MIPEmulsionTrackInfo();
							tmpTRtop = SySalInput.Top[j].Info;
							//tmpTRbot = new SySal.Tracking.MIPEmulsionTrackInfo();
							tmpTRbot = SySalInput.Bottom[j].Info;

							for(k = 0; k < npar; k++)
							{
								if(f.ParameterList[k].ToLower() == "px") f[k] = tmpTR.Intercept.X;
								else if(f.ParameterList[k].ToLower() == "py") f[k] = tmpTR.Intercept.Y;
								else if(f.ParameterList[k].ToLower() == "pz") f[k] = tmpTR.Intercept.Z;
								else if(f.ParameterList[k].ToLower() == "sx") f[k] = tmpTR.Slope.X;
								else if(f.ParameterList[k].ToLower() == "sy") f[k] = tmpTR.Slope.Y;
								else if(f.ParameterList[k].ToLower() == "a") f[k] = tmpTR.AreaSum;
								else if(f.ParameterList[k].ToLower() == "sigma") f[k] = tmpTR.Sigma;
								else if(f.ParameterList[k].ToLower() == "topz") f[k] = tmpTR.TopZ;
								else if(f.ParameterList[k].ToLower() == "bottomz") f[k] = tmpTR.BottomZ;
								else if(f.ParameterList[k].ToLower() == "n") f[k] = tmpTR.Count;
								else if(f.ParameterList[k].ToLower() == "field") f[k] = tmpTR.Field;
								else if(f.ParameterList[k].ToLower() == "tsx") f[k] = tmpTRtop.Slope.X;
								else if(f.ParameterList[k].ToLower() == "tsy") f[k] = tmpTRtop.Slope.Y;
								else if(f.ParameterList[k].ToLower() == "bsx") f[k] = tmpTRbot.Slope.X;
								else if(f.ParameterList[k].ToLower() == "bsy") f[k] = tmpTRbot.Slope.Y;
								else throw new Exception("Unknown parameter in selection: " + f.ParameterList[k]);

							}

							chk = System.Convert.ToBoolean(f.Evaluate());
						}
                        if (chk)
                        {
                            if (applysel) remaplist[j] = ArrTr.Count;
                            ArrTr.Add(new SySal.TotalScan.Segment(tmpTR, new SySal.TotalScan.BaseTrackIndex(j)));
                        }
                        else if (applysel) remaplist[j] = -1;
					}                    

					SySal.TotalScan.Layer tmpLayer = new SySal.TotalScan.Layer(i, InputBrickList[i], InputIdList[i], 0, c);	
					tmpLayer.AddSegments((SySal.TotalScan.Segment [])ArrTr.ToArray(typeof(SySal.TotalScan.Segment)));
					AORec.AddLayer(tmpLayer);
                    if (InputAlignIgnoreList[i] != null)
                    {
                        if (InputAlignIgnoreList[i].ToLower().EndsWith(".tlg"))
                        {
                            System.IO.FileStream r = new System.IO.FileStream(InputAlignIgnoreList[i], System.IO.FileMode.Open, System.IO.FileAccess.Read);
                            SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment ai = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment(r);
                            r.Close();
                            AORec.SetAlignmentIgnoreList(i, ai.Ids);
                        }
                        else
                        {
                            System.IO.StreamReader r = new System.IO.StreamReader(InputAlignIgnoreList[i]);
                            System.Collections.ArrayList tmpignorelist = new ArrayList();
                            string line;
                            while ((line = r.ReadLine()) != null)
                                try
                                {
                                    int ix = System.Convert.ToInt32(line);
                                    if (applysel) ix = remaplist[ix];
                                    if (ix >= 0) tmpignorelist.Add(ix);
                                }
                                catch (Exception) { };
                            r.Close();
                            AORec.SetAlignmentIgnoreList(i, (int[])tmpignorelist.ToArray(typeof(int)));
                        }
                    }
					progressBar1.Value = i+1;
					rtxtReport.Text += "File #" + (i+1) + " (Id=" + InputIdList[i] + "): " + ArrTr.Count + " tracks selected\r\n";
					LayersAdded++;
				};

				if(inmode == DataMode.Database) dbconn.Close();
				chkSelection.Enabled = true;
				rtxtReport.Text += "Stand By\r\n";
			}
			catch(Exception exc)
			{
				if(inmode == DataMode.Database) dbconn.Close();
				System.Windows.Forms.MessageBox.Show("Load Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				if(inmode == DataMode.Files && file!=null) file.Close();
			}
		
		}

		private void cmdSelection_Click(object sender, System.EventArgs e)
		{
			txtSelection.Text += cmbSelection.Text;
		}

		private void OnLoad(object sender, System.EventArgs e)
		{
            RedirectSaveTSR2Close = false;
            SySal.TotalScan.BaseTrackIndex.RegisterFactory();
            SySal.TotalScan.NamedAttributeIndex.RegisterFactory();
            SySal.TotalScan.NullIndex.RegisterFactory();
            SySal.TotalScan.MIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBNamedAttributeIndex.RegisterFactory();
			cmbSelection.Items.Add("PX");
			cmbSelection.Items.Add("PY");
			cmbSelection.Items.Add("PZ");
			cmbSelection.Items.Add("SX");
			cmbSelection.Items.Add("SY");
			cmbSelection.Items.Add("A");
			cmbSelection.Items.Add("Sigma");
			cmbSelection.Items.Add("TopZ");
			cmbSelection.Items.Add("BottomZ");
			cmbSelection.Items.Add("N");
			cmbSelection.Items.Add("Field");
			cmbSelection.Items.Add("TSX");
			cmbSelection.Items.Add("TSY");
			cmbSelection.Items.Add("BSX");
			cmbSelection.Items.Add("BSY");

            cmbShowSelection.Items.Add("ID");
            cmbShowSelection.Items.Add("USX");
			cmbShowSelection.Items.Add("USY");
			cmbShowSelection.Items.Add("UPX");
			cmbShowSelection.Items.Add("UPY");
            cmbShowSelection.Items.Add("UPZ");
			cmbShowSelection.Items.Add("UZ");
            cmbShowSelection.Items.Add("UVID");
            cmbShowSelection.Items.Add("UIP");
			cmbShowSelection.Items.Add("N");
			cmbShowSelection.Items.Add("DSX");
			cmbShowSelection.Items.Add("DSY");
			cmbShowSelection.Items.Add("DPX");
			cmbShowSelection.Items.Add("DPY");
            cmbShowSelection.Items.Add("DPZ");
			cmbShowSelection.Items.Add("DZ");
            cmbShowSelection.Items.Add("DVID");
            cmbShowSelection.Items.Add("DIP");

            m_TSRDS.DataType = "TSR";
            m_TSRDS.DataId = 0;

            txtTSRDSName.Text = m_TSRDS.DataType;
            txtTSRDSBrick.Text = m_TSRDS.DataId.ToString();

            cmbMCSAlgo.Items.Add(new SySal.Processing.MCSAnnecy.MomentumEstimator());
            cmbMCSAlgo.Items.Add(MCSLikelihood/* new SySal.Processing.MCSLikelihood.MomentumEstimator()*/);            
            cmbMCSAlgo.SelectedIndex = 0;

            cmbEvents.Items.Add(0L);
            cmbEvents.SelectedIndex = 0;

            cmbTrackExtrapolationMode.Items.Add(SySal.TotalScan.Track.ExtrapolationMode.EndBaseTrack);
            cmbTrackExtrapolationMode.Items.Add(SySal.TotalScan.Track.ExtrapolationMode.SegmentFit);
            cmbTrackExtrapolationMode.SelectedIndex = 0;

            cmbVtxTrackWeighting.Items.Add("Attribute-driven Weight");
            cmbVtxTrackWeighting.Items.Add("Flat Weight");
            cmbVtxTrackWeighting.Items.Add("Slope Scattering Weight");
            cmbVtxTrackWeighting.SelectedIndex = 0;

            UserProfileInfo.Load();
            cmbMapMergeFilter.Items.Clear();            
            foreach (string s in UserProfileInfo.ThisProfileInfo.MapMergeFilters)
                cmbMapMergeFilter.Items.Add(s);
            if (cmbMapMergeFilter.Items.Count > 0) cmbMapMergeFilter.SelectedIndex = 0;

            this.AORec.Expose = true;
		}

        bool XpEnableTimer = false;

		private void cmdSaveList_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sf = new SaveFileDialog();
			sf.Filter = "text document files (*.txt)|*.txt|All files (*.*)|*.*";
			sf.FilterIndex = 0;
			sf.CheckPathExists = true;
			sf.OverwritePrompt = true;
			if (sf.ShowDialog() == DialogResult.OK)
				txtFileListSelection.Text = sf.FileName;

		}

		private void cmdSaveSelection_Click(object sender, System.EventArgs e)
		{
			int i, n = InputFileList.Length;
			System.IO.StreamWriter w;
			try
			{
				w = new System.IO.StreamWriter(txtFileListSelection.Text);
				for(i=0; i<n; i++)
					w.WriteLine("{0}\t{1}", InputFileList[i], InputZCoordList[i]);
				w.WriteLine("{0}", "Selection: " + txtSelection.Text);
				w.Close();
			}
			catch(Exception exc)
			{
				//w.Close();
				System.Windows.Forms.MessageBox.Show("Load Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}


		}

		private void cmdProcess_Click(object sender, System.EventArgs e)
		{            
#if (ALIGN_DEBUG)
			Process();
#else
			new dProcess(this.Process).BeginInvoke(null, null);
#endif
		}

		private void Process()
		{



//			System.IO.StreamWriter w = null;
#if (!ALIGN_DEBUG)
			try
#endif
			{
                XpEnableTimer = true;
				//w = new System.IO.StreamWriter(txtFileList.Text.Split('.')[0] + "out.txt");

				StopThread = false;
				progressBar1.Maximum = 100;//00*(LayersAdded-1);
				rtxtReport.Text += "Processing...\r\n";

				AORec.ShouldStop = new SySal.TotalScan.dShouldStop(ShouldStop);
				AORec.Progress = new SySal.TotalScan.dProgress(Progress);
				AORec.Report = new SySal.TotalScan.dReport(Report);
				//bool vtxrec = true;
				v = AORec.Reconstruct();

                AORecAlignData = null;
                System.Collections.ArrayList xinfo = AORec.ExposedInfo;
                foreach (object o in xinfo)
                {
                    if (o != null && o.GetType() == typeof(SySal.Processing.AlphaOmegaReconstruction.AlignmentData[]))
                    {
                        AORecAlignData = (SySal.Processing.AlphaOmegaReconstruction.AlignmentData[])o;
                        break;
                    }
                }                

				cmbAlignedSheet.Items.Clear();
                if (AORecAlignData != null)
                    for (int kSheet = 0; kSheet < LayersAdded - 1; kSheet++)
                        if (AORecAlignData[kSheet].Result == MappingResult.OK) cmbAlignedSheet.Items.Add("Sheets: " + v.Layers[kSheet].SheetId + " - " + v.Layers[kSheet + 1].SheetId);
/*
				for(int i = 0; i< AORecAlignData.Length-1; i++)
				w.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}", 
					i, AORecAlignData[i].TranslationX, AORecAlignData[i].TranslationY, AORecAlignData[i].TranslationZ,
					AORecAlignData[i].AffineMatrixXX, AORecAlignData[i].AffineMatrixXY, AORecAlignData[i].AffineMatrixYX, AORecAlignData[i].AffineMatrixYY);
*/				
				rtxtReport.Text += "Stand by\r\n";

			}
#if (!ALIGN_DEBUG)
			catch(Exception exc)
			{                
				System.Windows.Forms.MessageBox.Show("Processing Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
#endif
            XpEnableTimer = false;
/*
 			if (w != null) 
			{
				w.Flush();
				w.Close();
			}
*/	
		}

		private void cmdShowData_Click(object sender, System.EventArgs e)
		{
			string[] tokens = cmbAlignedSheet.Text.Split(' ');
			if (tokens.Length==1 && tokens[0]=="") 
			{
				System.Windows.Forms.MessageBox.Show("Insert a selection in the combo box.", "Undefined data", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
				return;
			}
			int[] ids = new int[2] {System.Convert.ToInt32(tokens[1]), System.Convert.ToInt32(tokens[3])};
			int n = v.Tracks.Length, m;
			int i, j;
			int Count=0;

			EasyReconstruct.AnalysisForm frmAna = new EasyReconstruct.AnalysisForm();
			frmAna.Text = cmbAlignedSheet.Text;

			double refz=0;
			bool checkout = false;
			for(i=0; i<n; i++)
				if((m = v.Tracks[i].Length)>1)
				{
					for(j=0; j<m-1; j++)
						if(v.Tracks[i][j].LayerOwner.SheetId == ids[0] && 
							v.Tracks[i][j+1].LayerOwner.SheetId == ids[1])
						{
							refz = 0.5*(v.Tracks[i][j].LayerOwner.UpstreamZ + v.Tracks[i][j+1].LayerOwner.DownstreamZ);
							checkout = true;
							break;
						}
					if(checkout) break;
				}

			Count++;
			for(i++; i<n; i++)
				if((m = v.Tracks[i].Length)>1)
					for(j=0; j<m-1; j++)
						if(v.Tracks[i][j].LayerOwner.SheetId == ids[0] && 
							v.Tracks[i][j+1].LayerOwner.SheetId == ids[1]) Count++;

			double[] sx = new double[Count];
			double[] sy = new double[Count];
			double[] dsx = new double[Count];
			double[] dsy = new double[Count];
			double[] dsxfit = new double[Count];
			double[] dsyfit = new double[Count];
			double[] x = new double[Count];
			double[] y = new double[Count];
			double[] dx = new double[Count];
			double[] dy = new double[Count];
			double[] dxfit = new double[Count];
			double[] dyfit = new double[Count];
			double[] count = new double[Count];
			double tmpfitx, tmpfity, tmpfitsx, tmpfitsy, dummy;
			
			Count = 0;
			int fitseg = System.Convert.ToInt32(txtFitSeg.Text);
			for(i=0; i<n; i++)
			{
				v.Tracks[i].FittingSegments = fitseg;
				if((m = v.Tracks[i].Length)>1)
					for(j=0; j<m-1; j++)
					{
						if(v.Tracks[i][j].LayerOwner.SheetId == ids[0] && 
							v.Tracks[i][j+1].LayerOwner.SheetId == ids[1])
						{
							count[Count] = v.Tracks[i].Length;
							sx[Count] = v.Tracks[i][j].Info.Slope.X;
							sy[Count] = v.Tracks[i][j].Info.Slope.Y;
							dsx[Count] = sx[Count] - v.Tracks[i][j+1].Info.Slope.X;
							dsy[Count] = sy[Count] - v.Tracks[i][j+1].Info.Slope.Y;

							v.Tracks[i].Compute_Local_XCoord(j, out tmpfitsx, out tmpfitx);	
							//v.Tracks[i].Compute_Local_XCoord(0, out tmpfitsx, out tmpfitx);	
							//v.Tracks[i].Compute_Local_XCoord(j, out tmpfitx, out dummy);	
							dsxfit[Count] = sx[Count] - tmpfitsx;
							dxfit[Count] = v.Tracks[i][j].Info.Intercept.X - (tmpfitx + tmpfitsx * v.Tracks[i][j].Info.Intercept.Z);
							v.Tracks[i].Compute_Local_YCoord(j, out tmpfitsy, out tmpfity);	
							//v.Tracks[i].Compute_Local_YCoord(0, out tmpfitsy, out tmpfity);	
							//v.Tracks[i].Compute_Local_YCoord(j, out tmpfity, out dummy);	
							dsyfit[Count] = sy[Count] - tmpfitsy;
							dyfit[Count] = v.Tracks[i][j].Info.Intercept.Y - (tmpfity + tmpfitsy * v.Tracks[i][j].Info.Intercept.Z);
							x[Count] = v.Tracks[i][j].Info.Intercept.X + (refz - v.Tracks[i][j].Info.Intercept.Z) * sx[Count];
							y[Count] = v.Tracks[i][j].Info.Intercept.Y + (refz - v.Tracks[i][j].Info.Intercept.Z) * sy[Count];
							dx[Count] = x[Count] - (v.Tracks[i][j+1].Info.Intercept.X + (refz - v.Tracks[i][j+1].Info.Intercept.Z) * v.Tracks[i][j+1].Info.Slope.X);
							dy[Count] = y[Count] - (v.Tracks[i][j+1].Info.Intercept.Y + (refz - v.Tracks[i][j+1].Info.Intercept.Z) * v.Tracks[i][j+1].Info.Slope.Y);
							Count++;
						}
					}
			}

			frmAna.analysisControl1.AddDataSet("General");
			frmAna.analysisControl1.AddVariable(sx, "sx", "");
			frmAna.analysisControl1.AddVariable(sy, "sy", "");
			frmAna.analysisControl1.AddVariable(dsx, "dsx", "");
			frmAna.analysisControl1.AddVariable(dsy, "dsy", "");
			frmAna.analysisControl1.AddVariable(dsxfit, "dsxfit", "");
			frmAna.analysisControl1.AddVariable(dsyfit, "dsyfit", "");
			frmAna.analysisControl1.AddVariable(x, "x", "micron");
			frmAna.analysisControl1.AddVariable(y, "y", "micron");
			frmAna.analysisControl1.AddVariable(dx, "dx", "micron");
			frmAna.analysisControl1.AddVariable(dy, "dy", "micron");
			frmAna.analysisControl1.AddVariable(dxfit, "dxfit", "micron");
			frmAna.analysisControl1.AddVariable(dyfit, "dyfit", "micron");
			frmAna.analysisControl1.AddVariable(count, "n", "");
			frmAna.Show();

		}

		private void cmdStop_Click(object sender, System.EventArgs e)
		{
			if (System.Windows.Forms.MessageBox.Show("Are you sure you want to stop processing?" , "Stop Processing", MessageBoxButtons.YesNo, MessageBoxIcon.Exclamation) == DialogResult.Yes)	StopThread = true;
		}

		private void cmdClearReport_Click(object sender, System.EventArgs e)
		{
			rtxtReport.Text = "";
		}

		private void cmdInputMCSMomentumConfig_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog of = new OpenFileDialog();
			of.CheckFileExists = true;
			of.Filter = "xml files (*.xml)|*.xml|All files (*.*)|*.*";
			of.FilterIndex = 0;
			if (of.ShowDialog() == DialogResult.OK)
				txtMCSMomentumConfigFile.Text = of.FileName;

		}

		private void cmdOutputMCSMomentumConfig_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sf = new SaveFileDialog();
			sf.Filter = "xml files (*.xml)|*.xml|All files (*.*)|*.*";
			sf.FilterIndex = 0;
			sf.CheckPathExists = true;
			sf.OverwritePrompt = true;
			if (sf.ShowDialog() == DialogResult.OK)
				txtSaveMCSMomentumConfigFile.Text = sf.FileName;

		}

		private void cmdLoadMCSMomentumConfig_Click(object sender, System.EventArgs e)
		{
			System.IO.FileStream f;
			try
			{
				f = new System.IO.FileStream(txtMCSMomentumConfigFile.Text, System.IO.FileMode.Open, System.IO.FileAccess.Read);
				System.Runtime.Serialization.Formatters.Soap.SoapFormatter fmt = new System.Runtime.Serialization.Formatters.Soap.SoapFormatter();
				((SySal.Management.IManageable)cmbMCSAlgo.SelectedItem).Config = (SySal.Management.Configuration)fmt.Deserialize(f);
				f.Close();
			}
			catch(Exception exc)
			{
				//f.Close();
				System.Windows.Forms.MessageBox.Show("Load Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

        private void cmdSaveMCSMomentumConfig_Click(object sender, EventArgs e)
        {
            System.IO.FileStream f;
            try
            {
                System.Runtime.Serialization.Formatters.Soap.SoapFormatter fmt = new System.Runtime.Serialization.Formatters.Soap.SoapFormatter();
                f = new System.IO.FileStream(txtSaveMCSMomentumConfigFile.Text, System.IO.FileMode.Create, System.IO.FileAccess.Write);
                fmt.Serialize(f, ((SySal.Management.IManageable)cmbMCSAlgo.SelectedItem).Config);
                f.Flush();
                f.Close();
            }
            catch (Exception exc)
            {
                //f.Close();
                System.Windows.Forms.MessageBox.Show("Save Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

		private void cmdEditMCSMomentumConfig_Click(object sender, System.EventArgs e)
		{
			SySal.Management.Configuration tmpc = (SySal.Management.Configuration)(((SySal.Management.IManageable)cmbMCSAlgo.SelectedItem).Config.Clone());
            if (((SySal.Management.IManageable)cmbMCSAlgo.SelectedItem).EditConfiguration(ref tmpc)) ((SySal.Management.IManageable)cmbMCSAlgo.SelectedItem).Config = (SySal.Management.Configuration)tmpc.Clone();

		}

		private void cmdEfficiency_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog ofn = new OpenFileDialog();
			ofn.Title = "Load postprocessing assembly";
			ofn.Filter = "Assembly files (*.dll, *.exe)|*.exe;*.dll|All files (*.*)|*.*";
			try
			{
				if (ofn.ShowDialog() == DialogResult.OK)
				{
					System.Reflection.Assembly ass = System.Reflection.Assembly.LoadFrom(ofn.FileName);
					SySal.TotalScan.PostProcessing.DataAnalyzer da = null;
					foreach (System.Type t in ass.GetExportedTypes())
					{
						try
						{
							da = (SySal.TotalScan.PostProcessing.DataAnalyzer)ass.CreateInstance(t.FullName);
							break;
						}
						catch (Exception) {}
					}
					if (da == null) throw new Exception("No DataAnalyzer found in selected assembly.");
					da.Feed(v);
				}
			}
			catch (Exception x)
			{
				MessageBox.Show(x.ToString(), "Can't perform postprocessing task");
			}
		}

		private void cmdOutputFile_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sf = new SaveFileDialog();
			sf.Filter = "tsr files (*.tsr)|*.tsr|All files (*.*)|*.*";
			sf.FilterIndex = 0;
			sf.CheckPathExists = true;
			sf.OverwritePrompt = true;
			if (sf.ShowDialog() == DialogResult.OK)
				txtRecOutputFile.Text = sf.FileName;
		
		}

		private void cmdSaveOutputFile_Click(object sender, System.EventArgs e)
		{
			System.IO.FileStream w = null;
			try
			{
				w = new System.IO.FileStream(txtRecOutputFile.Text, System.IO.FileMode.Create);
				if(v==null) throw new Exception("Volume not set to an object:\n\rNo volume to dump");
				v.Save(w);
				w.Flush();
				w.Close();
				w = null;
			}
			catch
			{
				if (w != null) w.Close();
			}
	
		}

		private void button2_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog of = new OpenFileDialog();
			of.Filter = "tsr files (*.tsr)|*.tsr|All files (*.*)|*.*";
			of.FilterIndex = 0;
			of.CheckPathExists = true;
			if (of.ShowDialog() == DialogResult.OK)
				txtRecInputFile.Text = of.FileName;
		
		}

		private void cmdLoadRec_Click(object sender, System.EventArgs e)
		{
			System.IO.FileStream r = null;

			try
			{                
				AORec.Clear();
				rtxtReport.Text = "Loading...\r\n";
				r = new System.IO.FileStream(txtRecInputFile.Text, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                v = null;
                GC.Collect();
                v = new SySal.TotalScan.Flexi.Volume();
                ((SySal.TotalScan.Flexi.Volume)v).ImportVolume(m_TSRDS, new SySal.TotalScan.Volume(r));
                int i;
                for (i = 0; i < v.Layers.Length && (v.Layers[i].BrickId < 1000000 || v.Layers[i].BrickId > 2999999); i++) ;
                if (i < v.Layers.Length)
                {
                    txtTSRDSBrick.Text = v.Layers[i].BrickId.ToString();
                    OnTSRBrickLeave(this, e);
                }
                for (i = 0; i < v.Vertices.Length; i++)
                    try
                    {
                        long ev = (long)v.Vertices[i].GetAttribute(VertexBrowser.FBEventIndex);
                        if (ev > 0)
                        {
                            cmbEvents.SelectedIndex = cmbEvents.Items.Add(ev.ToString());
                            break;
                        }
                    }
                    catch (Exception) { }
                if (i == v.Vertices.Length)
                    for (i = 0; i < v.Tracks.Length; i++)
                        try
                        {
                            long ev = (long)v.Tracks[i].GetAttribute(TrackBrowser.FBEventIndex);
                            if (ev > 0)
                            {
                                cmbEvents.SelectedIndex = cmbEvents.Items.Add(ev.ToString());
                                break;
                            }
                        }
                        catch (Exception) { }
				rtxtReport.Text += "File successfully loaded\r\n";
				rtxtReport.Text += "Stand by\r\n";
				cmbAlignedSheet.Items.Clear();
				int LayersAdded = v.Layers.Length;
				for(int kSheet=0; kSheet<LayersAdded; kSheet++) 
				{
					AORec.AddLayer(v.Layers[kSheet]);
					if (kSheet > 0) cmbAlignedSheet.Items.Add("Sheets: " + v.Layers[kSheet-1].SheetId + " - " + v.Layers[kSheet].SheetId);
				}
				r.Close();
				r = null;
			}
			catch (Exception x)
			{
				if (r != null) r.Close();
				MessageBox.Show(x.ToString(), "Can't open file");
			}
		}
	
		internal delegate void dLoadFiles();
		internal delegate void dProcess();
		internal delegate void dGenerate();
		internal delegate void dVertexRec();

		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
            System.Windows.Forms.Control.CheckForIllegalCrossThreadCalls = false;
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.cmdDBSelect = new System.Windows.Forms.Button();
            this.cmdSaveList = new System.Windows.Forms.Button();
            this.cmdSaveSelection = new System.Windows.Forms.Button();
            this.txtFileListSelection = new System.Windows.Forms.TextBox();
            this.cmdSelection = new System.Windows.Forms.Button();
            this.cmbSelection = new System.Windows.Forms.ComboBox();
            this.txtSelection = new System.Windows.Forms.TextBox();
            this.chkSelection = new System.Windows.Forms.CheckBox();
            this.cmdLoadFiles = new System.Windows.Forms.Button();
            this.txtFileList = new System.Windows.Forms.TextBox();
            this.cmdLoad = new System.Windows.Forms.Button();
            this.cmdInput = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.cmdSaveConfig = new System.Windows.Forms.Button();
            this.txtSaveConfigFile = new System.Windows.Forms.TextBox();
            this.cmdSave = new System.Windows.Forms.Button();
            this.cmdEditConfig = new System.Windows.Forms.Button();
            this.txtConfigFile = new System.Windows.Forms.TextBox();
            this.cmdLoadConfig = new System.Windows.Forms.Button();
            this.cmdInputConfig = new System.Windows.Forms.Button();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.cmdProcess = new System.Windows.Forms.Button();
            this.cmdShowData = new System.Windows.Forms.Button();
            this.cmbAlignedSheet = new System.Windows.Forms.ComboBox();
            this.cmdStop = new System.Windows.Forms.Button();
            this.rtxtReport = new System.Windows.Forms.RichTextBox();
            this.cmdClearReport = new System.Windows.Forms.Button();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.cmbMCSAlgo = new System.Windows.Forms.ComboBox();
            this.cmdOutputMCSMomentumConfig = new System.Windows.Forms.Button();
            this.txtSaveMCSMomentumConfigFile = new System.Windows.Forms.TextBox();
            this.cmdSaveMCSMomentumConfig = new System.Windows.Forms.Button();
            this.cmdEditMCSMomentumConfig = new System.Windows.Forms.Button();
            this.txtMCSMomentumConfigFile = new System.Windows.Forms.TextBox();
            this.cmdLoadMCSMomentumConfig = new System.Windows.Forms.Button();
            this.cmdInputMCSMomentumConfig = new System.Windows.Forms.Button();
            this.cmdEfficiency = new System.Windows.Forms.Button();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.txtRecOutputFile = new System.Windows.Forms.TextBox();
            this.cmdSaveOutputFile = new System.Windows.Forms.Button();
            this.cmdOutputFile = new System.Windows.Forms.Button();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.txtRecInputFile = new System.Windows.Forms.TextBox();
            this.cmdLoadRec = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.txtFitSeg = new System.Windows.Forms.TextBox();
            this.cmdVertexRec = new System.Windows.Forms.Button();
            this.txtShowSelection = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.cmbShowSelection = new System.Windows.Forms.ComboBox();
            this.DisplayButton = new System.Windows.Forms.Button();
            this.groupBoxInfoPanel = new System.Windows.Forms.GroupBox();
            this.cmdXpRefresh = new System.Windows.Forms.Button();
            this.lvExposedInfo = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.XpRefreshTimer = new System.Windows.Forms.Timer(this.components);
            this.SetCommentButton = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.txtTrackComment = new System.Windows.Forms.TextBox();
            this.cmdMCSMomentum = new System.Windows.Forms.Button();
            this.txtTSRDSName = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtTSRDSBrick = new System.Windows.Forms.TextBox();
            this.btnBrowseDB = new System.Windows.Forms.Button();
            this.cmbVtxTrackWeighting = new System.Windows.Forms.ComboBox();
            this.label7 = new System.Windows.Forms.Label();
            this.cmbTrackExtrapolationMode = new System.Windows.Forms.ComboBox();
            this.label6 = new System.Windows.Forms.Label();
            this.chkResetSlopeCorrections = new System.Windows.Forms.CheckBox();
            this.cmbEvents = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.btnFileFormats = new System.Windows.Forms.Button();
            this.btnOpenFile = new System.Windows.Forms.Button();
            this.SaveImportedInfoButton = new System.Windows.Forms.Button();
            this.chkNormBrick = new System.Windows.Forms.CheckBox();
            this.chkSetBrickDownZ = new System.Windows.Forms.CheckBox();
            this.chkSetCSZ = new System.Windows.Forms.CheckBox();
            this.btnReset = new System.Windows.Forms.Button();
            this.lvImportedInfo = new System.Windows.Forms.ListBox();
            this.btnImport = new System.Windows.Forms.Button();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.btnFilterDel = new System.Windows.Forms.Button();
            this.btnFilterAdd = new System.Windows.Forms.Button();
            this.cmbMapMergeFilter = new System.Windows.Forms.ComboBox();
            this.txtResetDS = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.txtImportedDS = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.btnMapMergeFilterVars = new System.Windows.Forms.Button();
            this.btnVolMergeConfig = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.clAvailableInfo = new System.Windows.Forms.TreeView();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.groupBoxInfoPanel.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cmdDBSelect);
            this.groupBox1.Controls.Add(this.cmdSaveList);
            this.groupBox1.Controls.Add(this.cmdSaveSelection);
            this.groupBox1.Controls.Add(this.txtFileListSelection);
            this.groupBox1.Controls.Add(this.cmdSelection);
            this.groupBox1.Controls.Add(this.cmbSelection);
            this.groupBox1.Controls.Add(this.txtSelection);
            this.groupBox1.Controls.Add(this.chkSelection);
            this.groupBox1.Controls.Add(this.cmdLoadFiles);
            this.groupBox1.Controls.Add(this.txtFileList);
            this.groupBox1.Controls.Add(this.cmdLoad);
            this.groupBox1.Controls.Add(this.cmdInput);
            this.groupBox1.Location = new System.Drawing.Point(6, 62);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(625, 116);
            this.groupBox1.TabIndex = 3;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Scanning File List";
            // 
            // cmdDBSelect
            // 
            this.cmdDBSelect.Location = new System.Drawing.Point(531, 86);
            this.cmdDBSelect.Name = "cmdDBSelect";
            this.cmdDBSelect.Size = new System.Drawing.Size(88, 24);
            this.cmdDBSelect.TabIndex = 14;
            this.cmdDBSelect.Text = "DB Select";
            // 
            // cmdSaveList
            // 
            this.cmdSaveList.Location = new System.Drawing.Point(8, 84);
            this.cmdSaveList.Name = "cmdSaveList";
            this.cmdSaveList.Size = new System.Drawing.Size(88, 24);
            this.cmdSaveList.TabIndex = 13;
            this.cmdSaveList.Text = "Output List File";
            this.cmdSaveList.Click += new System.EventHandler(this.cmdSaveList_Click);
            // 
            // cmdSaveSelection
            // 
            this.cmdSaveSelection.Location = new System.Drawing.Point(436, 85);
            this.cmdSaveSelection.Name = "cmdSaveSelection";
            this.cmdSaveSelection.Size = new System.Drawing.Size(88, 24);
            this.cmdSaveSelection.TabIndex = 12;
            this.cmdSaveSelection.Text = "Save List File";
            this.cmdSaveSelection.Click += new System.EventHandler(this.cmdSaveSelection_Click);
            // 
            // txtFileListSelection
            // 
            this.txtFileListSelection.Location = new System.Drawing.Point(104, 84);
            this.txtFileListSelection.Name = "txtFileListSelection";
            this.txtFileListSelection.Size = new System.Drawing.Size(326, 20);
            this.txtFileListSelection.TabIndex = 11;
            // 
            // cmdSelection
            // 
            this.cmdSelection.Location = new System.Drawing.Point(531, 55);
            this.cmdSelection.Name = "cmdSelection";
            this.cmdSelection.Size = new System.Drawing.Size(88, 24);
            this.cmdSelection.TabIndex = 10;
            this.cmdSelection.Text = "Insert";
            this.cmdSelection.Click += new System.EventHandler(this.cmdSelection_Click);
            // 
            // cmbSelection
            // 
            this.cmbSelection.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbSelection.Location = new System.Drawing.Point(436, 56);
            this.cmbSelection.Name = "cmbSelection";
            this.cmbSelection.Size = new System.Drawing.Size(88, 21);
            this.cmbSelection.TabIndex = 9;
            // 
            // txtSelection
            // 
            this.txtSelection.Location = new System.Drawing.Point(104, 56);
            this.txtSelection.Name = "txtSelection";
            this.txtSelection.Size = new System.Drawing.Size(326, 20);
            this.txtSelection.TabIndex = 8;
            // 
            // chkSelection
            // 
            this.chkSelection.Location = new System.Drawing.Point(8, 56);
            this.chkSelection.Name = "chkSelection";
            this.chkSelection.Size = new System.Drawing.Size(104, 16);
            this.chkSelection.TabIndex = 7;
            this.chkSelection.Text = "Apply Selection";
            // 
            // cmdLoadFiles
            // 
            this.cmdLoadFiles.Location = new System.Drawing.Point(531, 19);
            this.cmdLoadFiles.Name = "cmdLoadFiles";
            this.cmdLoadFiles.Size = new System.Drawing.Size(88, 24);
            this.cmdLoadFiles.TabIndex = 6;
            this.cmdLoadFiles.Text = "Load Zones";
            this.cmdLoadFiles.Click += new System.EventHandler(this.cmdLoadFiles_Click);
            // 
            // txtFileList
            // 
            this.txtFileList.Location = new System.Drawing.Point(104, 21);
            this.txtFileList.Name = "txtFileList";
            this.txtFileList.Size = new System.Drawing.Size(326, 20);
            this.txtFileList.TabIndex = 5;
            // 
            // cmdLoad
            // 
            this.cmdLoad.Location = new System.Drawing.Point(436, 19);
            this.cmdLoad.Name = "cmdLoad";
            this.cmdLoad.Size = new System.Drawing.Size(88, 24);
            this.cmdLoad.TabIndex = 4;
            this.cmdLoad.Text = "Load List";
            this.cmdLoad.Click += new System.EventHandler(this.cmdLoad_Click);
            // 
            // cmdInput
            // 
            this.cmdInput.Location = new System.Drawing.Point(8, 19);
            this.cmdInput.Name = "cmdInput";
            this.cmdInput.Size = new System.Drawing.Size(88, 24);
            this.cmdInput.TabIndex = 3;
            this.cmdInput.Text = "Input List File";
            this.cmdInput.Click += new System.EventHandler(this.cmdInput_Click);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.cmdSaveConfig);
            this.groupBox2.Controls.Add(this.txtSaveConfigFile);
            this.groupBox2.Controls.Add(this.cmdSave);
            this.groupBox2.Controls.Add(this.cmdEditConfig);
            this.groupBox2.Controls.Add(this.txtConfigFile);
            this.groupBox2.Controls.Add(this.cmdLoadConfig);
            this.groupBox2.Controls.Add(this.cmdInputConfig);
            this.groupBox2.Location = new System.Drawing.Point(6, 182);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(625, 80);
            this.groupBox2.TabIndex = 4;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Configuration File";
            // 
            // cmdSaveConfig
            // 
            this.cmdSaveConfig.Location = new System.Drawing.Point(16, 48);
            this.cmdSaveConfig.Name = "cmdSaveConfig";
            this.cmdSaveConfig.Size = new System.Drawing.Size(104, 24);
            this.cmdSaveConfig.TabIndex = 9;
            this.cmdSaveConfig.Text = "Output Config File";
            this.cmdSaveConfig.Click += new System.EventHandler(this.cmdSaveConfig_Click);
            // 
            // txtSaveConfigFile
            // 
            this.txtSaveConfigFile.Location = new System.Drawing.Point(128, 48);
            this.txtSaveConfigFile.Name = "txtSaveConfigFile";
            this.txtSaveConfigFile.Size = new System.Drawing.Size(302, 20);
            this.txtSaveConfigFile.TabIndex = 8;
            // 
            // cmdSave
            // 
            this.cmdSave.Location = new System.Drawing.Point(436, 47);
            this.cmdSave.Name = "cmdSave";
            this.cmdSave.Size = new System.Drawing.Size(88, 24);
            this.cmdSave.TabIndex = 7;
            this.cmdSave.Text = "Save";
            this.cmdSave.Click += new System.EventHandler(this.cmdSave_Click);
            // 
            // cmdEditConfig
            // 
            this.cmdEditConfig.Location = new System.Drawing.Point(531, 17);
            this.cmdEditConfig.Name = "cmdEditConfig";
            this.cmdEditConfig.Size = new System.Drawing.Size(88, 24);
            this.cmdEditConfig.TabIndex = 6;
            this.cmdEditConfig.Text = "Edit";
            this.cmdEditConfig.Click += new System.EventHandler(this.cmdEditConfig_Click);
            // 
            // txtConfigFile
            // 
            this.txtConfigFile.Location = new System.Drawing.Point(128, 21);
            this.txtConfigFile.Name = "txtConfigFile";
            this.txtConfigFile.Size = new System.Drawing.Size(302, 20);
            this.txtConfigFile.TabIndex = 5;
            // 
            // cmdLoadConfig
            // 
            this.cmdLoadConfig.Location = new System.Drawing.Point(436, 17);
            this.cmdLoadConfig.Name = "cmdLoadConfig";
            this.cmdLoadConfig.Size = new System.Drawing.Size(88, 24);
            this.cmdLoadConfig.TabIndex = 4;
            this.cmdLoadConfig.Text = "Load";
            this.cmdLoadConfig.Click += new System.EventHandler(this.cmdLoadConfig_Click);
            // 
            // cmdInputConfig
            // 
            this.cmdInputConfig.Location = new System.Drawing.Point(16, 19);
            this.cmdInputConfig.Name = "cmdInputConfig";
            this.cmdInputConfig.Size = new System.Drawing.Size(104, 24);
            this.cmdInputConfig.TabIndex = 3;
            this.cmdInputConfig.Text = "Input Config File";
            this.cmdInputConfig.Click += new System.EventHandler(this.cmdInputConfig_Click);
            // 
            // progressBar1
            // 
            this.progressBar1.Location = new System.Drawing.Point(11, 227);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(576, 8);
            this.progressBar1.TabIndex = 5;
            // 
            // cmdProcess
            // 
            this.cmdProcess.Location = new System.Drawing.Point(396, 12);
            this.cmdProcess.Name = "cmdProcess";
            this.cmdProcess.Size = new System.Drawing.Size(89, 24);
            this.cmdProcess.TabIndex = 7;
            this.cmdProcess.Text = "Process";
            this.cmdProcess.Click += new System.EventHandler(this.cmdProcess_Click);
            // 
            // cmdShowData
            // 
            this.cmdShowData.Location = new System.Drawing.Point(504, 76);
            this.cmdShowData.Name = "cmdShowData";
            this.cmdShowData.Size = new System.Drawing.Size(80, 24);
            this.cmdShowData.TabIndex = 8;
            this.cmdShowData.Text = "Show Data";
            this.cmdShowData.Click += new System.EventHandler(this.cmdShowData_Click);
            // 
            // cmbAlignedSheet
            // 
            this.cmbAlignedSheet.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbAlignedSheet.Location = new System.Drawing.Point(396, 76);
            this.cmbAlignedSheet.Name = "cmbAlignedSheet";
            this.cmbAlignedSheet.Size = new System.Drawing.Size(96, 21);
            this.cmbAlignedSheet.TabIndex = 9;
            // 
            // cmdStop
            // 
            this.cmdStop.Location = new System.Drawing.Point(396, 136);
            this.cmdStop.Name = "cmdStop";
            this.cmdStop.Size = new System.Drawing.Size(79, 24);
            this.cmdStop.TabIndex = 10;
            this.cmdStop.Text = "Stop";
            this.cmdStop.Click += new System.EventHandler(this.cmdStop_Click);
            // 
            // rtxtReport
            // 
            this.rtxtReport.DetectUrls = false;
            this.rtxtReport.Location = new System.Drawing.Point(12, 14);
            this.rtxtReport.Name = "rtxtReport";
            this.rtxtReport.Size = new System.Drawing.Size(376, 150);
            this.rtxtReport.TabIndex = 11;
            this.rtxtReport.Text = "";
            // 
            // cmdClearReport
            // 
            this.cmdClearReport.Location = new System.Drawing.Point(396, 106);
            this.cmdClearReport.Name = "cmdClearReport";
            this.cmdClearReport.Size = new System.Drawing.Size(89, 24);
            this.cmdClearReport.TabIndex = 12;
            this.cmdClearReport.Text = "Clear Report";
            this.cmdClearReport.Click += new System.EventHandler(this.cmdClearReport_Click);
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.cmbMCSAlgo);
            this.groupBox3.Controls.Add(this.cmdOutputMCSMomentumConfig);
            this.groupBox3.Controls.Add(this.txtSaveMCSMomentumConfigFile);
            this.groupBox3.Controls.Add(this.cmdSaveMCSMomentumConfig);
            this.groupBox3.Controls.Add(this.cmdEditMCSMomentumConfig);
            this.groupBox3.Controls.Add(this.txtMCSMomentumConfigFile);
            this.groupBox3.Controls.Add(this.cmdLoadMCSMomentumConfig);
            this.groupBox3.Controls.Add(this.cmdInputMCSMomentumConfig);
            this.groupBox3.Location = new System.Drawing.Point(6, 266);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(625, 80);
            this.groupBox3.TabIndex = 14;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "MCS Momentum Configuration File";
            // 
            // cmbMCSAlgo
            // 
            this.cmbMCSAlgo.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbMCSAlgo.FormattingEnabled = true;
            this.cmbMCSAlgo.Location = new System.Drawing.Point(476, 49);
            this.cmbMCSAlgo.Name = "cmbMCSAlgo";
            this.cmbMCSAlgo.Size = new System.Drawing.Size(143, 21);
            this.cmbMCSAlgo.TabIndex = 10;
            // 
            // cmdOutputMCSMomentumConfig
            // 
            this.cmdOutputMCSMomentumConfig.Location = new System.Drawing.Point(16, 48);
            this.cmdOutputMCSMomentumConfig.Name = "cmdOutputMCSMomentumConfig";
            this.cmdOutputMCSMomentumConfig.Size = new System.Drawing.Size(104, 24);
            this.cmdOutputMCSMomentumConfig.TabIndex = 9;
            this.cmdOutputMCSMomentumConfig.Text = "Output Config File";
            this.cmdOutputMCSMomentumConfig.Click += new System.EventHandler(this.cmdOutputMCSMomentumConfig_Click);
            // 
            // txtSaveMCSMomentumConfigFile
            // 
            this.txtSaveMCSMomentumConfigFile.Location = new System.Drawing.Point(128, 48);
            this.txtSaveMCSMomentumConfigFile.Name = "txtSaveMCSMomentumConfigFile";
            this.txtSaveMCSMomentumConfigFile.Size = new System.Drawing.Size(248, 20);
            this.txtSaveMCSMomentumConfigFile.TabIndex = 8;
            // 
            // cmdSaveMCSMomentumConfig
            // 
            this.cmdSaveMCSMomentumConfig.Location = new System.Drawing.Point(385, 48);
            this.cmdSaveMCSMomentumConfig.Name = "cmdSaveMCSMomentumConfig";
            this.cmdSaveMCSMomentumConfig.Size = new System.Drawing.Size(88, 24);
            this.cmdSaveMCSMomentumConfig.TabIndex = 7;
            this.cmdSaveMCSMomentumConfig.Text = "Save";
            this.cmdSaveMCSMomentumConfig.Click += new System.EventHandler(this.cmdSaveMCSMomentumConfig_Click);
            // 
            // cmdEditMCSMomentumConfig
            // 
            this.cmdEditMCSMomentumConfig.Location = new System.Drawing.Point(531, 18);
            this.cmdEditMCSMomentumConfig.Name = "cmdEditMCSMomentumConfig";
            this.cmdEditMCSMomentumConfig.Size = new System.Drawing.Size(88, 24);
            this.cmdEditMCSMomentumConfig.TabIndex = 6;
            this.cmdEditMCSMomentumConfig.Text = "Edit";
            this.cmdEditMCSMomentumConfig.Click += new System.EventHandler(this.cmdEditMCSMomentumConfig_Click);
            // 
            // txtMCSMomentumConfigFile
            // 
            this.txtMCSMomentumConfigFile.Location = new System.Drawing.Point(128, 21);
            this.txtMCSMomentumConfigFile.Name = "txtMCSMomentumConfigFile";
            this.txtMCSMomentumConfigFile.Size = new System.Drawing.Size(248, 20);
            this.txtMCSMomentumConfigFile.TabIndex = 5;
            // 
            // cmdLoadMCSMomentumConfig
            // 
            this.cmdLoadMCSMomentumConfig.Location = new System.Drawing.Point(385, 18);
            this.cmdLoadMCSMomentumConfig.Name = "cmdLoadMCSMomentumConfig";
            this.cmdLoadMCSMomentumConfig.Size = new System.Drawing.Size(88, 24);
            this.cmdLoadMCSMomentumConfig.TabIndex = 4;
            this.cmdLoadMCSMomentumConfig.Text = "Load";
            this.cmdLoadMCSMomentumConfig.Click += new System.EventHandler(this.cmdLoadMCSMomentumConfig_Click);
            // 
            // cmdInputMCSMomentumConfig
            // 
            this.cmdInputMCSMomentumConfig.Location = new System.Drawing.Point(16, 19);
            this.cmdInputMCSMomentumConfig.Name = "cmdInputMCSMomentumConfig";
            this.cmdInputMCSMomentumConfig.Size = new System.Drawing.Size(104, 24);
            this.cmdInputMCSMomentumConfig.TabIndex = 3;
            this.cmdInputMCSMomentumConfig.Text = "Input Config File";
            this.cmdInputMCSMomentumConfig.Click += new System.EventHandler(this.cmdInputMCSMomentumConfig_Click);
            // 
            // cmdEfficiency
            // 
            this.cmdEfficiency.Location = new System.Drawing.Point(499, 173);
            this.cmdEfficiency.Name = "cmdEfficiency";
            this.cmdEfficiency.Size = new System.Drawing.Size(88, 24);
            this.cmdEfficiency.TabIndex = 15;
            this.cmdEfficiency.Text = "Efficiency";
            this.cmdEfficiency.Visible = false;
            this.cmdEfficiency.Click += new System.EventHandler(this.cmdEfficiency_Click);
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.txtRecOutputFile);
            this.groupBox4.Controls.Add(this.cmdSaveOutputFile);
            this.groupBox4.Controls.Add(this.cmdOutputFile);
            this.groupBox4.Location = new System.Drawing.Point(6, 348);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(625, 56);
            this.groupBox4.TabIndex = 19;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Reconstruction Output File";
            // 
            // txtRecOutputFile
            // 
            this.txtRecOutputFile.Location = new System.Drawing.Point(104, 21);
            this.txtRecOutputFile.Name = "txtRecOutputFile";
            this.txtRecOutputFile.Size = new System.Drawing.Size(421, 20);
            this.txtRecOutputFile.TabIndex = 5;
            // 
            // cmdSaveOutputFile
            // 
            this.cmdSaveOutputFile.Location = new System.Drawing.Point(531, 19);
            this.cmdSaveOutputFile.Name = "cmdSaveOutputFile";
            this.cmdSaveOutputFile.Size = new System.Drawing.Size(88, 24);
            this.cmdSaveOutputFile.TabIndex = 4;
            this.cmdSaveOutputFile.Text = "Save";
            this.cmdSaveOutputFile.Click += new System.EventHandler(this.cmdSaveOutputFile_Click);
            // 
            // cmdOutputFile
            // 
            this.cmdOutputFile.Location = new System.Drawing.Point(8, 19);
            this.cmdOutputFile.Name = "cmdOutputFile";
            this.cmdOutputFile.Size = new System.Drawing.Size(88, 24);
            this.cmdOutputFile.TabIndex = 3;
            this.cmdOutputFile.Text = "Output File";
            this.cmdOutputFile.Click += new System.EventHandler(this.cmdOutputFile_Click);
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.txtRecInputFile);
            this.groupBox5.Controls.Add(this.cmdLoadRec);
            this.groupBox5.Controls.Add(this.button2);
            this.groupBox5.Location = new System.Drawing.Point(6, 6);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(625, 56);
            this.groupBox5.TabIndex = 20;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Reconstruction Input File";
            // 
            // txtRecInputFile
            // 
            this.txtRecInputFile.Location = new System.Drawing.Point(104, 21);
            this.txtRecInputFile.Name = "txtRecInputFile";
            this.txtRecInputFile.Size = new System.Drawing.Size(420, 20);
            this.txtRecInputFile.TabIndex = 5;
            // 
            // cmdLoadRec
            // 
            this.cmdLoadRec.Location = new System.Drawing.Point(531, 19);
            this.cmdLoadRec.Name = "cmdLoadRec";
            this.cmdLoadRec.Size = new System.Drawing.Size(88, 24);
            this.cmdLoadRec.TabIndex = 4;
            this.cmdLoadRec.Text = "Load";
            this.cmdLoadRec.Click += new System.EventHandler(this.cmdLoadRec_Click);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(8, 19);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(88, 24);
            this.button2.TabIndex = 3;
            this.button2.Text = "Input File";
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(501, 112);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(88, 13);
            this.label1.TabIndex = 21;
            this.label1.Text = "Fitting Segments:";
            // 
            // txtFitSeg
            // 
            this.txtFitSeg.Location = new System.Drawing.Point(557, 139);
            this.txtFitSeg.Name = "txtFitSeg";
            this.txtFitSeg.Size = new System.Drawing.Size(24, 20);
            this.txtFitSeg.TabIndex = 22;
            this.txtFitSeg.Text = "4";
            this.txtFitSeg.Leave += new System.EventHandler(this.txtFitSeg_Leave);
            // 
            // cmdVertexRec
            // 
            this.cmdVertexRec.Location = new System.Drawing.Point(498, 12);
            this.cmdVertexRec.Name = "cmdVertexRec";
            this.cmdVertexRec.Size = new System.Drawing.Size(89, 24);
            this.cmdVertexRec.TabIndex = 23;
            this.cmdVertexRec.Text = "Vertexing";
            this.cmdVertexRec.Click += new System.EventHandler(this.cmdVertexRec_Click);
            // 
            // txtShowSelection
            // 
            this.txtShowSelection.Location = new System.Drawing.Point(68, 173);
            this.txtShowSelection.Name = "txtShowSelection";
            this.txtShowSelection.Size = new System.Drawing.Size(320, 20);
            this.txtShowSelection.TabIndex = 24;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 175);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(54, 13);
            this.label2.TabIndex = 25;
            this.label2.Text = "Selection:";
            // 
            // cmbShowSelection
            // 
            this.cmbShowSelection.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbShowSelection.Location = new System.Drawing.Point(396, 173);
            this.cmbShowSelection.Name = "cmbShowSelection";
            this.cmbShowSelection.Size = new System.Drawing.Size(88, 21);
            this.cmbShowSelection.TabIndex = 26;
            // 
            // DisplayButton
            // 
            this.DisplayButton.Location = new System.Drawing.Point(421, 333);
            this.DisplayButton.Name = "DisplayButton";
            this.DisplayButton.Size = new System.Drawing.Size(87, 24);
            this.DisplayButton.TabIndex = 27;
            this.DisplayButton.Text = "Display";
            this.DisplayButton.Click += new System.EventHandler(this.DisplayButton_Click);
            // 
            // groupBoxInfoPanel
            // 
            this.groupBoxInfoPanel.Controls.Add(this.cmdXpRefresh);
            this.groupBoxInfoPanel.Controls.Add(this.lvExposedInfo);
            this.groupBoxInfoPanel.Location = new System.Drawing.Point(15, 241);
            this.groupBoxInfoPanel.Name = "groupBoxInfoPanel";
            this.groupBoxInfoPanel.Size = new System.Drawing.Size(572, 237);
            this.groupBoxInfoPanel.TabIndex = 28;
            this.groupBoxInfoPanel.TabStop = false;
            this.groupBoxInfoPanel.Text = "Exposed information from inner computation code";
            // 
            // cmdXpRefresh
            // 
            this.cmdXpRefresh.Location = new System.Drawing.Point(477, 208);
            this.cmdXpRefresh.Name = "cmdXpRefresh";
            this.cmdXpRefresh.Size = new System.Drawing.Size(89, 24);
            this.cmdXpRefresh.TabIndex = 28;
            this.cmdXpRefresh.Text = "Refresh";
            this.cmdXpRefresh.Click += new System.EventHandler(this.cmdXpRefresh_Click);
            // 
            // lvExposedInfo
            // 
            this.lvExposedInfo.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1});
            this.lvExposedInfo.GridLines = true;
            this.lvExposedInfo.Location = new System.Drawing.Point(11, 19);
            this.lvExposedInfo.Name = "lvExposedInfo";
            this.lvExposedInfo.Size = new System.Drawing.Size(555, 183);
            this.lvExposedInfo.TabIndex = 0;
            this.lvExposedInfo.UseCompatibleStateImageBehavior = false;
            this.lvExposedInfo.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Info";
            this.columnHeader1.Width = 600;
            // 
            // XpRefreshTimer
            // 
            this.XpRefreshTimer.Enabled = true;
            this.XpRefreshTimer.Interval = 10000;
            this.XpRefreshTimer.Tick += new System.EventHandler(this.XpRefreshTick);
            // 
            // SetCommentButton
            // 
            this.SetCommentButton.Location = new System.Drawing.Point(396, 197);
            this.SetCommentButton.Name = "SetCommentButton";
            this.SetCommentButton.Size = new System.Drawing.Size(89, 24);
            this.SetCommentButton.TabIndex = 30;
            this.SetCommentButton.Text = "Set Comment";
            this.SetCommentButton.Click += new System.EventHandler(this.SetCommentButton_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(12, 203);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(54, 13);
            this.label3.TabIndex = 32;
            this.label3.Text = "Comment:";
            // 
            // txtTrackComment
            // 
            this.txtTrackComment.Location = new System.Drawing.Point(68, 200);
            this.txtTrackComment.Name = "txtTrackComment";
            this.txtTrackComment.Size = new System.Drawing.Size(320, 20);
            this.txtTrackComment.TabIndex = 31;
            // 
            // cmdMCSMomentum
            // 
            this.cmdMCSMomentum.Location = new System.Drawing.Point(396, 42);
            this.cmdMCSMomentum.Name = "cmdMCSMomentum";
            this.cmdMCSMomentum.Size = new System.Drawing.Size(89, 24);
            this.cmdMCSMomentum.TabIndex = 33;
            this.cmdMCSMomentum.Text = "Momentum";
            this.cmdMCSMomentum.Click += new System.EventHandler(this.cmdMCSMomentum_Click);
            // 
            // txtTSRDSName
            // 
            this.txtTSRDSName.Location = new System.Drawing.Point(240, 53);
            this.txtTSRDSName.Name = "txtTSRDSName";
            this.txtTSRDSName.Size = new System.Drawing.Size(60, 20);
            this.txtTSRDSName.TabIndex = 35;
            this.txtTSRDSName.Leave += new System.EventHandler(this.OnTSRNameLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(4, 56);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(224, 13);
            this.label4.TabIndex = 34;
            this.label4.Text = "TSR DataSet name + brick/process operation";
            // 
            // txtTSRDSBrick
            // 
            this.txtTSRDSBrick.Location = new System.Drawing.Point(306, 53);
            this.txtTSRDSBrick.Name = "txtTSRDSBrick";
            this.txtTSRDSBrick.Size = new System.Drawing.Size(109, 20);
            this.txtTSRDSBrick.TabIndex = 36;
            this.txtTSRDSBrick.Leave += new System.EventHandler(this.OnTSRBrickLeave);
            // 
            // btnBrowseDB
            // 
            this.btnBrowseDB.Location = new System.Drawing.Point(9, 81);
            this.btnBrowseDB.Name = "btnBrowseDB";
            this.btnBrowseDB.Size = new System.Drawing.Size(99, 24);
            this.btnBrowseDB.TabIndex = 43;
            this.btnBrowseDB.Text = "Browse DB";
            this.btnBrowseDB.Click += new System.EventHandler(this.btnBrowseDB_Click);
            // 
            // cmbVtxTrackWeighting
            // 
            this.cmbVtxTrackWeighting.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbVtxTrackWeighting.FormattingEnabled = true;
            this.cmbVtxTrackWeighting.Location = new System.Drawing.Point(374, 9);
            this.cmbVtxTrackWeighting.Name = "cmbVtxTrackWeighting";
            this.cmbVtxTrackWeighting.Size = new System.Drawing.Size(134, 21);
            this.cmbVtxTrackWeighting.TabIndex = 63;
            this.cmbVtxTrackWeighting.SelectedIndexChanged += new System.EventHandler(this.OnVertexTrackWeightingChanged);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(249, 12);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(112, 13);
            this.label7.TabIndex = 64;
            this.label7.Text = "Vertex track weighting";
            // 
            // cmbTrackExtrapolationMode
            // 
            this.cmbTrackExtrapolationMode.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbTrackExtrapolationMode.FormattingEnabled = true;
            this.cmbTrackExtrapolationMode.Location = new System.Drawing.Point(116, 9);
            this.cmbTrackExtrapolationMode.Name = "cmbTrackExtrapolationMode";
            this.cmbTrackExtrapolationMode.Size = new System.Drawing.Size(117, 21);
            this.cmbTrackExtrapolationMode.TabIndex = 46;
            this.cmbTrackExtrapolationMode.SelectedIndexChanged += new System.EventHandler(this.OnTrackExtrapolationModeChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(6, 12);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(98, 13);
            this.label6.TabIndex = 62;
            this.label6.Text = "Track extrapolation";
            // 
            // chkResetSlopeCorrections
            // 
            this.chkResetSlopeCorrections.AutoSize = true;
            this.chkResetSlopeCorrections.Checked = true;
            this.chkResetSlopeCorrections.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkResetSlopeCorrections.Location = new System.Drawing.Point(421, 99);
            this.chkResetSlopeCorrections.Name = "chkResetSlopeCorrections";
            this.chkResetSlopeCorrections.Size = new System.Drawing.Size(82, 30);
            this.chkResetSlopeCorrections.TabIndex = 61;
            this.chkResetSlopeCorrections.Text = "Reset slope\r\ncorrections";
            this.chkResetSlopeCorrections.UseVisualStyleBackColor = true;
            // 
            // cmbEvents
            // 
            this.cmbEvents.FormattingEnabled = true;
            this.cmbEvents.Location = new System.Drawing.Point(4, 277);
            this.cmbEvents.Name = "cmbEvents";
            this.cmbEvents.Size = new System.Drawing.Size(101, 21);
            this.cmbEvents.TabIndex = 60;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(4, 254);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(35, 13);
            this.label5.TabIndex = 59;
            this.label5.Text = "Event";
            // 
            // btnFileFormats
            // 
            this.btnFileFormats.Location = new System.Drawing.Point(9, 149);
            this.btnFileFormats.Name = "btnFileFormats";
            this.btnFileFormats.Size = new System.Drawing.Size(99, 24);
            this.btnFileFormats.TabIndex = 58;
            this.btnFileFormats.Text = "File Formats";
            this.btnFileFormats.Click += new System.EventHandler(this.btnFileFormats_Click);
            // 
            // btnOpenFile
            // 
            this.btnOpenFile.Location = new System.Drawing.Point(9, 111);
            this.btnOpenFile.Name = "btnOpenFile";
            this.btnOpenFile.Size = new System.Drawing.Size(99, 24);
            this.btnOpenFile.TabIndex = 57;
            this.btnOpenFile.Text = "Open File";
            this.btnOpenFile.Click += new System.EventHandler(this.btnOpenFile_Click);
            // 
            // SaveImportedInfoButton
            // 
            this.SaveImportedInfoButton.Location = new System.Drawing.Point(421, 303);
            this.SaveImportedInfoButton.Name = "SaveImportedInfoButton";
            this.SaveImportedInfoButton.Size = new System.Drawing.Size(87, 24);
            this.SaveImportedInfoButton.TabIndex = 56;
            this.SaveImportedInfoButton.Text = "ASCII dump";
            this.SaveImportedInfoButton.Click += new System.EventHandler(this.SaveImportedInfoButton_Click);
            // 
            // chkNormBrick
            // 
            this.chkNormBrick.AutoSize = true;
            this.chkNormBrick.Checked = true;
            this.chkNormBrick.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkNormBrick.Location = new System.Drawing.Point(421, 55);
            this.chkNormBrick.Name = "chkNormBrick";
            this.chkNormBrick.Size = new System.Drawing.Size(87, 30);
            this.chkNormBrick.TabIndex = 55;
            this.chkNormBrick.Text = "Normalize\r\nbrick number";
            this.chkNormBrick.UseVisualStyleBackColor = true;
            // 
            // chkSetBrickDownZ
            // 
            this.chkSetBrickDownZ.AutoSize = true;
            this.chkSetBrickDownZ.Checked = true;
            this.chkSetBrickDownZ.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkSetBrickDownZ.Location = new System.Drawing.Point(7, 190);
            this.chkSetBrickDownZ.Name = "chkSetBrickDownZ";
            this.chkSetBrickDownZ.Size = new System.Drawing.Size(122, 17);
            this.chkSetBrickDownZ.TabIndex = 54;
            this.chkSetBrickDownZ.Text = "Set Brick Down Z=0";
            this.chkSetBrickDownZ.UseVisualStyleBackColor = true;
            // 
            // chkSetCSZ
            // 
            this.chkSetCSZ.AutoSize = true;
            this.chkSetCSZ.Checked = true;
            this.chkSetCSZ.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkSetCSZ.Location = new System.Drawing.Point(7, 223);
            this.chkSetCSZ.Name = "chkSetCSZ";
            this.chkSetCSZ.Size = new System.Drawing.Size(99, 17);
            this.chkSetCSZ.TabIndex = 53;
            this.chkSetCSZ.Text = "Set CS Z=4550";
            this.chkSetCSZ.UseVisualStyleBackColor = true;
            // 
            // btnReset
            // 
            this.btnReset.Location = new System.Drawing.Point(421, 223);
            this.btnReset.Name = "btnReset";
            this.btnReset.Size = new System.Drawing.Size(87, 24);
            this.btnReset.TabIndex = 52;
            this.btnReset.Text = "Reset";
            this.btnReset.Click += new System.EventHandler(this.btnReset_Click);
            // 
            // lvImportedInfo
            // 
            this.lvImportedInfo.FormattingEnabled = true;
            this.lvImportedInfo.Location = new System.Drawing.Point(116, 223);
            this.lvImportedInfo.Name = "lvImportedInfo";
            this.lvImportedInfo.Size = new System.Drawing.Size(299, 134);
            this.lvImportedInfo.TabIndex = 51;
            // 
            // btnImport
            // 
            this.btnImport.Font = new System.Drawing.Font("Symbol", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(2)));
            this.btnImport.Location = new System.Drawing.Point(261, 185);
            this.btnImport.Name = "btnImport";
            this.btnImport.Size = new System.Drawing.Size(69, 24);
            this.btnImport.TabIndex = 50;
            this.btnImport.Text = "¯";
            this.btnImport.Click += new System.EventHandler(this.btnImport_Click);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Location = new System.Drawing.Point(1, 2);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(645, 514);
            this.tabControl1.TabIndex = 47;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.groupBox5);
            this.tabPage1.Controls.Add(this.groupBox1);
            this.tabPage1.Controls.Add(this.groupBox2);
            this.tabPage1.Controls.Add(this.groupBox3);
            this.tabPage1.Controls.Add(this.groupBox4);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(637, 488);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "File management";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.clAvailableInfo);
            this.tabPage2.Controls.Add(this.btnFilterDel);
            this.tabPage2.Controls.Add(this.btnFilterAdd);
            this.tabPage2.Controls.Add(this.cmbMapMergeFilter);
            this.tabPage2.Controls.Add(this.txtResetDS);
            this.tabPage2.Controls.Add(this.label9);
            this.tabPage2.Controls.Add(this.txtImportedDS);
            this.tabPage2.Controls.Add(this.label8);
            this.tabPage2.Controls.Add(this.btnMapMergeFilterVars);
            this.tabPage2.Controls.Add(this.btnVolMergeConfig);
            this.tabPage2.Controls.Add(this.cmbVtxTrackWeighting);
            this.tabPage2.Controls.Add(this.label7);
            this.tabPage2.Controls.Add(this.label6);
            this.tabPage2.Controls.Add(this.cmbTrackExtrapolationMode);
            this.tabPage2.Controls.Add(this.DisplayButton);
            this.tabPage2.Controls.Add(this.txtTSRDSBrick);
            this.tabPage2.Controls.Add(this.chkResetSlopeCorrections);
            this.tabPage2.Controls.Add(this.btnBrowseDB);
            this.tabPage2.Controls.Add(this.cmbEvents);
            this.tabPage2.Controls.Add(this.txtTSRDSName);
            this.tabPage2.Controls.Add(this.label5);
            this.tabPage2.Controls.Add(this.label4);
            this.tabPage2.Controls.Add(this.btnFileFormats);
            this.tabPage2.Controls.Add(this.btnOpenFile);
            this.tabPage2.Controls.Add(this.btnImport);
            this.tabPage2.Controls.Add(this.SaveImportedInfoButton);
            this.tabPage2.Controls.Add(this.lvImportedInfo);
            this.tabPage2.Controls.Add(this.chkNormBrick);
            this.tabPage2.Controls.Add(this.btnReset);
            this.tabPage2.Controls.Add(this.chkSetBrickDownZ);
            this.tabPage2.Controls.Add(this.chkSetCSZ);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(637, 488);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Display";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // btnFilterDel
            // 
            this.btnFilterDel.Location = new System.Drawing.Point(587, 393);
            this.btnFilterDel.Name = "btnFilterDel";
            this.btnFilterDel.Size = new System.Drawing.Size(40, 28);
            this.btnFilterDel.TabIndex = 75;
            this.btnFilterDel.Text = "-";
            this.btnFilterDel.UseVisualStyleBackColor = true;
            this.btnFilterDel.Click += new System.EventHandler(this.btnFilterDel_Click);
            // 
            // btnFilterAdd
            // 
            this.btnFilterAdd.Location = new System.Drawing.Point(543, 393);
            this.btnFilterAdd.Name = "btnFilterAdd";
            this.btnFilterAdd.Size = new System.Drawing.Size(40, 28);
            this.btnFilterAdd.TabIndex = 74;
            this.btnFilterAdd.Text = "+";
            this.btnFilterAdd.UseVisualStyleBackColor = true;
            this.btnFilterAdd.Click += new System.EventHandler(this.btnFilterAdd_Click);
            // 
            // cmbMapMergeFilter
            // 
            this.cmbMapMergeFilter.FormattingEnabled = true;
            this.cmbMapMergeFilter.Location = new System.Drawing.Point(288, 398);
            this.cmbMapMergeFilter.Name = "cmbMapMergeFilter";
            this.cmbMapMergeFilter.Size = new System.Drawing.Size(243, 21);
            this.cmbMapMergeFilter.TabIndex = 73;
            // 
            // txtResetDS
            // 
            this.txtResetDS.Location = new System.Drawing.Point(270, 433);
            this.txtResetDS.Name = "txtResetDS";
            this.txtResetDS.Size = new System.Drawing.Size(60, 20);
            this.txtResetDS.TabIndex = 72;
            this.txtResetDS.Text = "SBSF";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(236, 436);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(16, 13);
            this.label9.TabIndex = 71;
            this.label9.Text = "to";
            // 
            // txtImportedDS
            // 
            this.txtImportedDS.Location = new System.Drawing.Point(165, 433);
            this.txtImportedDS.Name = "txtImportedDS";
            this.txtImportedDS.Size = new System.Drawing.Size(60, 20);
            this.txtImportedDS.TabIndex = 70;
            this.txtImportedDS.Text = "TSR";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(9, 436);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(144, 13);
            this.label8.TabIndex = 69;
            this.label8.Text = "Reset/filter imported DataSet";
            // 
            // btnMapMergeFilterVars
            // 
            this.btnMapMergeFilterVars.Location = new System.Drawing.Point(183, 393);
            this.btnMapMergeFilterVars.Name = "btnMapMergeFilterVars";
            this.btnMapMergeFilterVars.Size = new System.Drawing.Size(99, 28);
            this.btnMapMergeFilterVars.TabIndex = 68;
            this.btnMapMergeFilterVars.Text = "Filter Variables";
            this.btnMapMergeFilterVars.Click += new System.EventHandler(this.btnMapMergeFilterVars_Click);
            // 
            // btnVolMergeConfig
            // 
            this.btnVolMergeConfig.Location = new System.Drawing.Point(9, 393);
            this.btnVolMergeConfig.Name = "btnVolMergeConfig";
            this.btnVolMergeConfig.Size = new System.Drawing.Size(165, 28);
            this.btnVolMergeConfig.TabIndex = 65;
            this.btnVolMergeConfig.Text = "Volume merging parameters";
            this.btnVolMergeConfig.Click += new System.EventHandler(this.btnVolMergeConfig_Click);
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.rtxtReport);
            this.tabPage3.Controls.Add(this.groupBoxInfoPanel);
            this.tabPage3.Controls.Add(this.cmdMCSMomentum);
            this.tabPage3.Controls.Add(this.progressBar1);
            this.tabPage3.Controls.Add(this.label3);
            this.tabPage3.Controls.Add(this.cmdProcess);
            this.tabPage3.Controls.Add(this.txtTrackComment);
            this.tabPage3.Controls.Add(this.cmdShowData);
            this.tabPage3.Controls.Add(this.SetCommentButton);
            this.tabPage3.Controls.Add(this.cmbAlignedSheet);
            this.tabPage3.Controls.Add(this.cmdStop);
            this.tabPage3.Controls.Add(this.cmbShowSelection);
            this.tabPage3.Controls.Add(this.cmdClearReport);
            this.tabPage3.Controls.Add(this.label2);
            this.tabPage3.Controls.Add(this.cmdEfficiency);
            this.tabPage3.Controls.Add(this.txtShowSelection);
            this.tabPage3.Controls.Add(this.label1);
            this.tabPage3.Controls.Add(this.cmdVertexRec);
            this.tabPage3.Controls.Add(this.txtFitSeg);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(637, 488);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Reconstruction";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // clAvailableInfo
            // 
            this.clAvailableInfo.CheckBoxes = true;
            this.clAvailableInfo.FullRowSelect = true;
            this.clAvailableInfo.HideSelection = false;
            this.clAvailableInfo.Location = new System.Drawing.Point(116, 79);
            this.clAvailableInfo.Name = "clAvailableInfo";
            this.clAvailableInfo.Size = new System.Drawing.Size(299, 97);
            this.clAvailableInfo.TabIndex = 76;
            this.clAvailableInfo.AfterCheck += new System.Windows.Forms.TreeViewEventHandler(this.OnNodeAfterCheck);
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(649, 516);
            this.Controls.Add(this.tabControl1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.Text = "Easy Reconstruct";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.groupBoxInfoPanel.ResumeLayout(false);
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.ResumeLayout(false);

		}
		#endregion

		private void txtFitSeg_Leave(object sender, System.EventArgs e)
		{
			int tmp;
			string tmpstr = txtFitSeg.Text;
			try
			{
				tmp = System.Convert.ToInt32(txtFitSeg.Text);
				if (tmp<2) throw new Exception("Fitting Segments Number must be greater than 2");
				txtFitSeg.Text = System.Convert.ToString(tmp);
			}
			catch(Exception exc)
			{
				System.Windows.Forms.MessageBox.Show(exc.ToString(), "Easy Reconstruct",
					System.Windows.Forms.MessageBoxButtons.OK, System.Windows.Forms.MessageBoxIcon.Error);
				txtFitSeg.Text = tmpstr;
			}

		}

		private void cmdVertexRec_Click(object sender, System.EventArgs e)
		{
			new dVertexRec(this.VertexRec).BeginInvoke(null, null);
		}

		private void VertexRec()
		{

			try
			{

				StopThread = false;
				//progressBar1.Maximum = v.Tracks.Length;
				rtxtReport.Text += "Reconstructing Vertices...\r\n";

				AORec.ShouldStop = new SySal.TotalScan.dShouldStop(ShouldStop);
				//AORec.Progress = new SySal.TotalScan.dProgress(Progress);
				AORec.Report = new SySal.TotalScan.dReport(Report);
				v = AORec.RecomputeVertices(v);

				rtxtReport.Text += "Stand by\r\n";

			}
			catch(Exception exc)
			{
				System.Windows.Forms.MessageBox.Show("Vertex Reconstruction Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

        private static bool SelectTrack(SySal.TotalScan.Track tmpTR, NumericalTools.Function f)
        {
            if (f == null) return true;
            int npar, k;
            npar = f.ParameterList.Length;
            for (k = 0; k < npar; k++)
            {
                if (f.ParameterList[k].ToLower() == "id") f[k] = tmpTR.Id;
                else if (f.ParameterList[k].ToLower() == "usx") f[k] = tmpTR.Upstream_SlopeX;
                else if (f.ParameterList[k].ToLower() == "usy") f[k] = tmpTR.Upstream_SlopeY;
                else if (f.ParameterList[k].ToLower() == "upx") f[k] = tmpTR.Upstream_PosX;
                else if (f.ParameterList[k].ToLower() == "upy") f[k] = tmpTR.Upstream_PosY;
                else if (f.ParameterList[k].ToLower() == "upz") f[k] = tmpTR.Upstream_PosZ;
                else if (f.ParameterList[k].ToLower() == "uz") f[k] = tmpTR.Upstream_Z;
                else if (f.ParameterList[k].ToLower() == "uvid") f[k] = (tmpTR.Upstream_Vertex == null) ? -1 : tmpTR.Upstream_Vertex.Id;
                else if (f.ParameterList[k].ToLower() == "uip") f[k] = (tmpTR.Upstream_Vertex == null) ? -1 : tmpTR.Upstream_Impact_Parameter;
                else if (f.ParameterList[k].ToLower() == "n") f[k] = tmpTR.Length;
                else if (f.ParameterList[k].ToLower() == "dsx") f[k] = tmpTR.Downstream_SlopeX;
                else if (f.ParameterList[k].ToLower() == "dsy") f[k] = tmpTR.Downstream_SlopeY;
                else if (f.ParameterList[k].ToLower() == "dpx") f[k] = tmpTR.Downstream_PosX;
                else if (f.ParameterList[k].ToLower() == "dpy") f[k] = tmpTR.Downstream_PosY;
                else if (f.ParameterList[k].ToLower() == "dpz") f[k] = tmpTR.Downstream_PosZ;
                else if (f.ParameterList[k].ToLower() == "dz") f[k] = tmpTR.Downstream_Z;
                else if (f.ParameterList[k].ToLower() == "dvid") f[k] = (tmpTR.Downstream_Vertex == null) ? -1 : tmpTR.Downstream_Vertex.Id;
                else if (f.ParameterList[k].ToLower() == "dip") f[k] = (tmpTR.Downstream_Vertex == null) ? -1 : tmpTR.Downstream_Impact_Parameter;
                else throw new Exception("Unknown parameter in selection: " + f.ParameterList[k]);
            }
            return System.Convert.ToBoolean(f.Evaluate());
        }

        internal static bool RedirectSaveTSR2Close = true;

        public static SySal.TotalScan.Volume EditedVolume = null;

        static System.Text.RegularExpressions.Regex rx_fieldsvalues = new System.Text.RegularExpressions.Regex(@"([^:]+):([IDSids]):([^:]+)");

        public static void SetSegmentExtendedFields(string s)
        {
            string [] fields = s.Split('/');
            object [] values = new object[fields.Length];
            int i;
            for (i = 0; i < fields.Length; i++)
            {
                System.Text.RegularExpressions.Match m = rx_fieldsvalues.Match(fields[i]);
                if (m.Success == false) throw new Exception("Cannot understand string " + i + ": \"" + fields[i] + "\".");
                switch (m.Groups[2].Value.ToUpper()[0])
                {
                    case 'I': values[i] = System.Convert.ToInt32(m.Groups[3].Value); break;
                    case 'D': values[i] = System.Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture); break;
                    case 'S': values[i] = m.Groups[3].Value; break;
                }
                fields[i] = m.Groups[1].Value.Trim();
                if (fields[i].Length <= 0) throw new Exception("Zero-length field (" + i + ").");
            }
            TrackBrowser.XSegInfo.SetExtendedFieldsList(fields, values);
            DisplayForm.InstallExtendedFieldFilters();
        }
        /// <summary>
        /// Displays tracks in a volume, using a specified selection.
        /// </summary>
        /// <param name="v">the volume from which tracks have to be picked.</param>
        /// <param name="selection">the selection string.</param>
        public static void RunDisplay(SySal.TotalScan.Flexi.Volume v, string selection)
        {
            try
            {              
                SySal.TotalScan.Track.TrackExtrapolationMode = SySal.TotalScan.Track.ExtrapolationMode.EndBaseTrack;
                SySal.TotalScan.Vertex.TrackWeightingFunction = new Vertex.dTrackWeightFunction(SySal.TotalScan.Vertex.AttributeWeight);

                if (MomentumFitForm.MCSLikelihood != null)
                    foreach (Geometry.LayerStart ls in ((SySal.Processing.MCSLikelihood.Configuration)MomentumFitForm.MCSLikelihood.Config).Geometry.Layers)
                        if (ls.RadiationLength <= 0.0)
                        {
                            MomentumFitForm.MCSLikelihood = null;
                            break;
                        }
                if (TrackBrowser.MCSAlgorithms == null || TrackBrowser.MCSAlgorithms.Length < 1)
                {
                    TrackBrowser.MCSAlgorithms = new IMCSMomentumEstimator[1] { new SySal.Processing.MCSAnnecy.MomentumEstimator() };
                }
                
                EasyReconstruct.DisplayForm frmView = new EasyReconstruct.DisplayForm((SySal.TotalScan.Flexi.Volume)v);
                frmView.SelectedEvent = 1;
                int i, j;
                int n = v.Tracks.Length, m;
                Function f = null;
                
                SySal.TotalScan.Track tmpTR = new SySal.TotalScan.Track();
                ArrayList ArrTr = new ArrayList();

                if (selection != null && selection.Length > 0) f = new CStyleParsedFunction(selection.ToLower());
                
                for (j = 0; j < n; j++)
                    if (SelectTrack(v.Tracks[j], f))
                        ArrTr.Add(v.Tracks[j]);
                
                SySal.TotalScan.Track tmpT;
                float MaxX, MinX, MaxZ, MinZ, MaxY, MinY;
                SySal.TotalScan.Track[] tmpArrayT = (SySal.TotalScan.Track[])ArrTr.ToArray(typeof(SySal.TotalScan.Track));
                int nt = tmpArrayT.Length;

                if (nt > 0)
                {
                    MaxX = (float)tmpArrayT[0][0].Info.Intercept.X;
                    MinX = MaxX;
                    MaxY = (float)tmpArrayT[0][0].Info.Intercept.Y;
                    MinY = MaxY;
                    MaxZ = (float)tmpArrayT[0][0].Info.Intercept.Z;
                    MinZ = MaxZ;

                    for (i = 0; i < nt; i++)
                    {
                        tmpT = tmpArrayT[i];
                        m = tmpT.Length;
                        for (j = 0; j < m; j++)
                        {
                            if (tmpT[j].Info.Intercept.X > MaxX) MaxX = (float)tmpT[j].Info.Intercept.X;
                            if ((float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X > MaxX) MaxX = (float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X;

                            if (tmpT[j].Info.Intercept.X < MinX) MinX = (float)tmpT[j].Info.Intercept.X;
                            if ((float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X < MinX) MinX = (float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X;

                            if (tmpT[j].Info.Intercept.Y > MaxY) MaxY = (float)tmpT[j].Info.Intercept.Y;
                            if ((float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y > MaxY) MaxY = (float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y;

                            if (tmpT[j].Info.Intercept.Y < MinY) MinY = (float)tmpT[j].Info.Intercept.Y;
                            if ((float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y < MinY) MinY = (float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y;

                            if (tmpT[j].Info.Intercept.Z > MaxZ) MaxZ = (float)tmpT[j].Info.Intercept.Z;
                            if ((float)tmpT[j].Info.Intercept.Z - 200f > MaxZ) MaxZ = (float)tmpT[j].Info.Intercept.Z - 200f;

                            if (tmpT[j].Info.Intercept.Z < MinZ) MinZ = (float)tmpT[j].Info.Intercept.Z;
                            if ((float)tmpT[j].Info.Intercept.Z - 200f < MinZ) MinZ = (float)tmpT[j].Info.Intercept.Z - 200f;
                        }
                    }
                }
                else
                {
                    for (i = 0; i < v.Layers.Length && v.Layers[i].Length == 0; i++) Console.WriteLine("Layer " + i + " segments " + v.Layers[i].Length);                    
                    if (i == v.Layers.Length)
                    {
                        throw new Exception("All layers are empty!");
                    }                    
                    SySal.Tracking.MIPEmulsionTrackInfo info = v.Layers[i][0].Info;
                    MinX = MaxX = (float)info.Intercept.X;
                    MinY = MaxY = (float)info.Intercept.Y;
                    MinZ = MaxZ = (float)info.Intercept.Z;

                    for (i = 0; i < v.Layers.Length; i++)
                    {
                        SySal.TotalScan.Layer lay = v.Layers[i];
                        nt = lay.Length;
                        for (j = 0; j < nt; j++)
                        {
                            info = lay[j].Info;

                            if (info.Intercept.X > MaxX) MaxX = (float)info.Intercept.X;
                            if ((float)info.Intercept.X - 200f * (float)info.Slope.X > MaxX) MaxX = (float)info.Intercept.X - 200f * (float)info.Slope.X;

                            if (info.Intercept.X < MinX) MinX = (float)info.Intercept.X;
                            if ((float)info.Intercept.X - 200f * (float)info.Slope.X < MinX) MinX = (float)info.Intercept.X - 200f * (float)info.Slope.X;

                            if (info.Intercept.Y > MaxY) MaxY = (float)info.Intercept.Y;
                            if ((float)info.Intercept.Y - 200f * (float)info.Slope.Y > MaxY) MaxY = (float)info.Intercept.Y - 200f * (float)info.Slope.Y;

                            if (info.Intercept.Y < MinY) MinY = (float)info.Intercept.Y;
                            if ((float)info.Intercept.Y - 200f * (float)info.Slope.Y < MinY) MinY = (float)info.Intercept.Y - 200f * (float)info.Slope.Y;

                            if (info.Intercept.Z > MaxZ) MaxZ = (float)info.Intercept.Z;
                            if ((float)info.Intercept.Z - 200f > MaxZ) MaxZ = (float)info.Intercept.Z - 200f;

                            if (info.Intercept.Z < MinZ) MinZ = (float)info.Intercept.Z;
                            if ((float)info.Intercept.Z - 200f < MinZ) MinZ = (float)info.Intercept.Z - 200f;
                        }                        
                    }                    
                }

                float gapx = (float)(MaxX - MinX) * 0.1f + 1000f;
                float gapy = (float)(MaxY - MinY) * 0.1f + 1000f;
                float gapz = (float)(MaxZ - MinZ) * 0.1f + 1000f;
                MaxX += gapx; MinX -= gapx;
                MaxY += gapy; MinY -= gapy;
                MaxZ += gapz; MinZ -= gapz;

                frmView.gdiDisplay1.Clear();
                //					.SetBounds3D(MinX, MaxX, MinY, MaxY, MinZ, MaxZ);
                frmView.gdiDisplay1.SetCameraSpotting(0.5 * (MinX + MaxX), 0.5 * (MinY + MaxY), 0.5 * (MinZ + MaxZ));
                frmView.gdiDisplay1.Distance = Math.Max(Math.Max(MaxY - MinY, MaxX - MinX), MaxZ - MinZ) * 5.0;
                frmView.gdiDisplay1.SetCameraOrientation(0, 0, -1.0, 0, 1.0, 0);

                frmView.B_MaxX = MaxX; frmView.B_MinX = MinX;
                frmView.B_MaxY = MaxY; frmView.B_MinY = MinY;
                frmView.B_MaxZ = MaxZ; frmView.B_MinZ = MinZ;
                frmView.MinimumSegmentsNumber = 1;
                frmView.MinimumTracksNumber = 2;
                frmView.mTracks = tmpArrayT;
                Application.Run(frmView);
//                frmView.Show();                           
            }
            catch (Exception exc)
            {
                MessageBox.Show(exc.ToString(), "Display error");
            }
        }

		private void DisplayButton_Click(object sender, System.EventArgs e)
		{
            Cursor oldc = Cursor;
            Cursor = Cursors.WaitCursor;
#if !DEBUG
            try
            {
#endif
                if (v == null) throw new Exception("Volume not set to an object");
                int i, j;
                TrackBrowser.MCSAlgorithms = new SySal.TotalScan.IMCSMomentumEstimator[cmbMCSAlgo.Items.Count];
                for (i = 0; i < TrackBrowser.MCSAlgorithms.Length; i++)
                    TrackBrowser.MCSAlgorithms[i] = (SySal.TotalScan.IMCSMomentumEstimator)cmbMCSAlgo.Items[i];
                SySal.TotalScan.Flexi.Volume myvol = PrepareVolume();
                if (myvol == null) throw new Exception("Error in display data import");
                MomentumFitForm.MCSLikelihood = MCSLikelihood;
                foreach (Geometry.LayerStart ls in ((SySal.Processing.MCSLikelihood.Configuration)MCSLikelihood.Config).Geometry.Layers)
                    if (ls.RadiationLength <= 0.0)
                    {
                        MomentumFitForm.MCSLikelihood = null;
                        break;
                    }
                EasyReconstruct.DisplayForm frmView = new EasyReconstruct.DisplayForm(myvol);
                frmView.SelectedEvent = System.Convert.ToInt64(cmbEvents.Text);
                int n = myvol.Tracks.Length, m;
                Function f = null;

                SySal.TotalScan.Track tmpTR = new SySal.TotalScan.Track();
                ArrayList ArrTr = new ArrayList();

                if (txtShowSelection.Text.Trim().Length > 0) f = new CStyleParsedFunction(txtShowSelection.Text.ToLower());

                for (j = 0; j < n; j++)
                    if (SelectTrack(myvol.Tracks[j], f))
                        ArrTr.Add(myvol.Tracks[j]);

                SySal.TotalScan.Track tmpT;
                float MaxX, MinX, MaxZ, MinZ, MaxY, MinY;
                SySal.TotalScan.Track[] tmpArrayT = (SySal.TotalScan.Track[])ArrTr.ToArray(typeof(SySal.TotalScan.Track));
                int nt = tmpArrayT.Length;

                MaxX = (float)tmpArrayT[0][0].Info.Intercept.X;
                MinX = MaxX;
                MaxY = (float)tmpArrayT[0][0].Info.Intercept.Y;
                MinY = MaxY;
                MaxZ = (float)tmpArrayT[0][0].Info.Intercept.Z;
                MinZ = MaxZ;


                for (i = 0; i < nt; i++)
                {
                    tmpT = tmpArrayT[i];
                    m = tmpT.Length;
                    for (j = 0; j < m; j++)
                    {
                        if (tmpT[j].Info.Intercept.X > MaxX) MaxX = (float)tmpT[j].Info.Intercept.X;
                        if ((float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X > MaxX) MaxX = (float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X;

                        if (tmpT[j].Info.Intercept.X < MinX) MinX = (float)tmpT[j].Info.Intercept.X;
                        if ((float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X < MinX) MinX = (float)tmpT[j].Info.Intercept.X - 200f * (float)tmpT[j].Info.Slope.X;

                        if (tmpT[j].Info.Intercept.Y > MaxY) MaxY = (float)tmpT[j].Info.Intercept.Y;
                        if ((float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y > MaxY) MaxY = (float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y;

                        if (tmpT[j].Info.Intercept.Y < MinY) MinY = (float)tmpT[j].Info.Intercept.Y;
                        if ((float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y < MinY) MinY = (float)tmpT[j].Info.Intercept.Y - 200f * (float)tmpT[j].Info.Slope.Y;

                        if (tmpT[j].Info.Intercept.Z > MaxZ) MaxZ = (float)tmpT[j].Info.Intercept.Z;
                        if ((float)tmpT[j].Info.Intercept.Z - 200f > MaxZ) MaxZ = (float)tmpT[j].Info.Intercept.Z - 200f;

                        if (tmpT[j].Info.Intercept.Z < MinZ) MinZ = (float)tmpT[j].Info.Intercept.Z;
                        if ((float)tmpT[j].Info.Intercept.Z - 200f < MinZ) MinZ = (float)tmpT[j].Info.Intercept.Z - 200f;
                    }
                }

                float gapx = (float)(MaxX - MinX) * 0.1f + 1000f;
                float gapy = (float)(MaxY - MinY) * 0.1f + 1000f;
                float gapz = (float)(MaxZ - MinZ) * 0.1f + 1000f;
                MaxX += gapx; MinX -= gapx;
                MaxY += gapy; MinY -= gapy;
                MaxZ += gapz; MinZ -= gapz;
                frmView.gdiDisplay1.Clear();
                //					.SetBounds3D(MinX, MaxX, MinY, MaxY, MinZ, MaxZ);
                frmView.gdiDisplay1.SetCameraSpotting(0.5 * (MinX + MaxX), 0.5 * (MinY + MaxY), 0.5 * (MinZ + MaxZ));
                frmView.gdiDisplay1.Distance = Math.Max(Math.Max(MaxY - MinY, MaxX - MinX), MaxZ - MinZ) * 5.0; ;
                frmView.gdiDisplay1.SetCameraOrientation(0, 0, -1.0, 0, 1.0, 0);

                frmView.B_MaxX = MaxX; frmView.B_MinX = MinX;
                frmView.B_MaxY = MaxY; frmView.B_MinY = MinY;
                frmView.B_MaxZ = MaxZ; frmView.B_MinZ = MinZ;
                frmView.MinimumSegmentsNumber = 1;
                frmView.MinimumTracksNumber = 2;
                frmView.Show();
                frmView.mTracks = tmpArrayT;
                Cursor = oldc;
#if !DEBUG
            }
            catch (Exception exc)
            {
                Cursor = oldc;                
                MessageBox.Show(exc.Message, "Selection error");
            }
#endif
            }

        private void cmdXpRefresh_Click(object sender, EventArgs e)
        {
            XpRefreshExecute();
        }

        private void XpRefreshTick(object sender, EventArgs e)
        {
            if (XpEnableTimer)
                XpRefreshExecute();
        }

        private void XpRefreshExecute()
        {
            lock (this)
            {
                if (AORec != null)
                {
                    System.Collections.ArrayList xi = AORec.ExposedInfo;
                    lvExposedInfo.BeginUpdate();
                    lvExposedInfo.Items.Clear();
                    if (xi != null)
                        foreach (object o in xi)
                            if (o != null && o.GetType() == typeof(string))
                                lvExposedInfo.Items.Add(o.ToString());
                    lvExposedInfo.EndUpdate();
                }
            }
        }

        private void SetCommentButton_Click(object sender, EventArgs e)
        {
            if (v == null) throw new Exception("Volume not set to an object");
            Function f = null;
            if (txtShowSelection.Text.Trim().Length > 0) f = new CStyleParsedFunction(txtShowSelection.Text.ToLower());
            int i;
            for (i = 0; i < v.Tracks.Length; i++)
                if (SelectTrack(v.Tracks[i], f))
                    v.Tracks[i].Comment = txtTrackComment.Text;
        }

        private void cmdMCSMomentum_Click(object sender, EventArgs e)
        {
            if (v != null)
            {
                int i, j, k;
                int measured = 0;
                int exc = 0;
                Cursor oldc = Cursor;
                try
                {
                    Cursor = Cursors.WaitCursor;
                    for (i = 0; i < v.Tracks.Length; i++)
                        try
                        {
                            MCSMomentumHelper.ProcessData((IMCSMomentumEstimator)cmbMCSAlgo.SelectedItem, v.Tracks[i]);
                            measured++;
                        }
                        catch (Exception x)
                        {
                            if (++exc == 1)
                                MessageBox.Show(x.ToString(), "Momentum fit error.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    if (exc > 1)
                        MessageBox.Show("Several exceptions (" + exc +") were thrown, only the first has been shown.", "Multiple errors.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    Cursor = oldc;
                }
                if (v.Tracks.Length > 0)
                    MessageBox.Show("Tracks measured: " + measured + "/" + v.Tracks.Length + " (" + ((double)measured * 100.0/(double)v.Tracks.Length).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)", "Momentum estimation complete", MessageBoxButtons.OK);
            }
        }

        SySal.TotalScan.Flexi.DataSet m_TSRDS = new SySal.TotalScan.Flexi.DataSet();

        private void OnTSRBrickLeave(object sender, EventArgs e)
        {
            try
            {
                m_TSRDS.DataId = System.Convert.ToInt64(txtTSRDSBrick.Text);
                if (chkNormBrick.Checked && m_TSRDS.DataId < 1000000)
                {
                    m_TSRDS.DataId += 1000000;
                    txtTSRDSBrick.Text = m_TSRDS.DataId.ToString();
                }
            }
            catch (Exception)
            {
                txtTSRDSBrick.Text = m_TSRDS.DataId.ToString();
            }
        }

        private void OnTSRNameLeave(object sender, EventArgs e)
        {
            if (txtTSRDSName.Text.Trim().Length == 0)
            {
                txtTSRDSName.Text = m_TSRDS.DataType;
                return;
            }
            m_TSRDS.DataType = txtTSRDSName.Text.Trim();
        }

        private void btnBrowseDB_Click(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                clAvailableInfo.BeginUpdate();
                clAvailableInfo.Nodes.Clear();
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                TreeNode node;

                System.Data.DataSet ds1 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM TB_EVENTBRICKS WHERE MOD(ID,1000000) = " + (m_TSRDS.DataId % 1000000).ToString() + " ORDER BY ID ASC", conn).Fill(ds1);
                foreach (System.Data.DataRow dr1 in ds1.Tables[0].Rows)
                    clAvailableInfo.Nodes.Add(DBGeomString + dr1[0].ToString()).Checked = true;

                node = null;
                System.Data.DataSet ds2 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK, ID_PROCESSOPERATION, PATH, row_number() over (partition by id_eventbrick, id_processoperation order by path) as rnum FROM TB_SCANBACK_PATHS WHERE MOD(ID_EVENTBRICK,1000000) = " + (m_TSRDS.DataId % 1000000).ToString() + " ORDER BY ID_EVENTBRICK, ID_PROCESSOPERATION, PATH ASC", conn).Fill(ds2);              
                foreach (System.Data.DataRow dr2 in ds2.Tables[0].Rows)
                {
                    if (SySal.OperaDb.Convert.ToInt32(dr2[3]) == 1)
                    {
                        node = clAvailableInfo.Nodes.Add(DBSBSFString + dr2[0].ToString() + " " + dr2[1].ToString());
                        node.Checked = true;                        
                    }
                    node.Nodes.Add(dr2[2].ToString()).Checked = true;
                }

                node = null;
                System.Data.DataSet ds3 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK, ID_PROCESSOPERATION, CANDIDATE, row_number() over (partition by ID_EVENTBRICK, ID_PROCESSOPERATION order by CANDIDATE) as rnum FROM TB_CS_CANDIDATES WHERE MOD(ID_EVENTBRICK,1000000) = " + (m_TSRDS.DataId % 1000000).ToString() + " ORDER BY ID_EVENTBRICK, ID_PROCESSOPERATION, CANDIDATE ASC", conn).Fill(ds3);
                foreach (System.Data.DataRow dr3 in ds3.Tables[0].Rows)
                {
                    if (SySal.OperaDb.Convert.ToInt32(dr3[3]) == 1)
                    {
                        node = clAvailableInfo.Nodes.Add(DBCSString + dr3[0].ToString() + " " + dr3[1].ToString());
                        node.Checked = true;
                    }
                    node.Nodes.Add(dr3[2].ToString()).Checked = true;
                }

                cmbEvents.Items.Clear();
                System.Data.DataSet ds4 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT DISTINCT EVENT FROM TB_PREDICTED_EVENTS WHERE ID IN (SELECT ID_EVENT FROM TV_PREDTRACK_BRICK_ASSOC WHERE MOD(ID_CS_EVENTBRICK,1000000) = " + (m_TSRDS.DataId % 1000000).ToString() + ") ORDER BY EVENT", conn).Fill(ds4);
                foreach (System.Data.DataRow dr4 in ds4.Tables[0].Rows)
                    cmbEvents.Items.Add(SySal.OperaDb.Convert.ToInt64(dr4[0]));
                cmbEvents.Items.Add(0);
                if (cmbEvents.Items.Count > 0) cmbEvents.SelectedIndex = 0;

                node = null;
                System.Data.DataSet ds5 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("select idev, track, row_number() over (partition by idev order by track) as rnum from tb_predicted_tracks " +
                    "inner join (select id_event as idev, track as idtk, pdgid, type as tktyp, momentum from tv_predtrack_brick_assoc where mod(id_cs_eventbrick, 1000000) = " + (m_TSRDS.DataId % 1000000).ToString() + ")" +
                    "on (id_event = idev and track = idtk)", conn).Fill(ds5);
                foreach (System.Data.DataRow dr5 in ds5.Tables[0].Rows)
                {
                    if (SySal.OperaDb.Convert.ToInt32(dr5[2]) == 1)
                    {
                        node = clAvailableInfo.Nodes.Add(DBTTString + dr5[0].ToString());
                        node.Checked = true;
                    }
                    node.Nodes.Add(dr5[1].ToString()).Checked = true;
                }
            }
            catch (Exception x)
            {

                MessageBox.Show(x.Message, "DB connection error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
                clAvailableInfo.EndUpdate();
            }
        }

        private const string DBGeomString = "DB Geom ";
        private const string DBSBSFString = "DB SB/SF ";
        private const string DBCSString = "DB CS ";
        private const string DBTTString = "DB TT ";
        private const string IMPVOLString = "IMPVOL ";

        System.Collections.ArrayList m_InteractiveDisplayImports = new ArrayList();

        private void btnImport_Click(object sender, EventArgs e)
        {
            UseWaitCursor = true;
            System.Collections.ArrayList check = new ArrayList();
            foreach (TreeNode node in clAvailableInfo.Nodes)
            {
                if (node.Nodes.Count == 0)
                {
                    if (node.Checked) check.Add(node.Text);
                }
                else 
                    foreach (TreeNode node1 in node.Nodes)                    
                        if (node1.Checked) check.Add(node.Text + " " + node1.Text);
            }
            foreach (string text in check)
            {
                if (text.StartsWith(DBGeomString)) DBImportGeom(text.Remove(0, DBGeomString.Length));
                else if (text.StartsWith(DBSBSFString)) DBImportSBSF(text.Remove(0, DBSBSFString.Length));
                else if (text.StartsWith(DBCSString)) DBImportCS(text.Remove(0, DBCSString.Length));
                else if (text.StartsWith(DBTTString)) DBImportTT(text.Remove(0, DBTTString.Length));
            }
            UseWaitCursor = false;
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            m_InteractiveDisplayImports.Clear();
            lvImportedInfo.Items.Clear();
        }

        private class DataDesc
        {
            public string Text;
            public object O;

            public DataDesc(string t, object o) { Text = t; O = o; }
        }

        private void DBImportGeom(string text)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();

                System.Data.DataSet ds1 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK, ID, Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + text + " ORDER BY Z ASC", conn).Fill(ds1);
                if (ds1.Tables[0].Rows.Count <= 0) return;
                double downz = SySal.OperaDb.Convert.ToDouble(ds1.Tables[0].Rows[ds1.Tables[0].Rows.Count - 1][2]);
                Geometry g = new Geometry();
                g.Layers = new Geometry.LayerStart[2 * ds1.Tables[0].Rows.Count];
                int i;
                for (i = 0; i < ds1.Tables[0].Rows.Count; i++)
                {
                    System.Data.DataRow dr = ds1.Tables[0].Rows[i];
                    g.Layers[2 * i].Brick = SySal.OperaDb.Convert.ToInt64(dr[0]);
                    g.Layers[2 * i].Plate = SySal.OperaDb.Convert.ToInt32(dr[1]);
                    g.Layers[2 * i].ZMin = (g.Layers[2 * i].Brick >= 2000000 && chkSetCSZ.Checked) 
                        ? ((1 - g.Layers[2 * i].Plate) * 300.0 + 4850.0 - 255.0)
                        : (SySal.OperaDb.Convert.ToDouble(dr[2]) - 255.0 - downz);
                    g.Layers[2 * i].RadiationLength = 29000.0;
                    g.Layers[2 * i + 1].Brick = g.Layers[2 * i].Brick;
                    g.Layers[2 * i + 1].Plate = 0;
                    g.Layers[2 * i + 1].ZMin = g.Layers[2 * i].ZMin + 300.0;
                    g.Layers[2 * i + 1].RadiationLength = 5600.0;
                }
                g.Layers[2 * i - 1].RadiationLength = 1e9;
                DataDesc nd = new DataDesc("Geom " + text, g);
                foreach (DataDesc d in m_InteractiveDisplayImports)
                    if (d.Text == nd.Text)
                        throw new Exception("Dataset already imported.");
                m_InteractiveDisplayImports.Add(nd);
                lvImportedInfo.Items.Add(nd.Text);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB import error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
            }
        }

        private void DBImportSBSF(string text)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                string [] dstext = text.Split(' ');
                SySal.TotalScan.Flexi.DataSet dsinfo = new SySal.TotalScan.Flexi.DataSet();
                dsinfo.DataId = System.Convert.ToInt64(dstext[1]);                
                dsinfo.DataType = "SBSF";
                System.Data.DataSet ds1 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter(
                    "select id_plate, Z, grains, areasum, posx, posy, slopex, slopey, sigma from tb_plates inner join " +
                    "(select idb, id_plate, grains, areasum, posx, posy, slopex, slopey, sigma from tb_mipbasetracks inner join " +
                    " (select id_eventbrick as idb, id_plate, id_candidate, id_zone as idz from tb_scanback_predictions where (id_eventbrick, id_path) in " +
                    "  (select id_eventbrick + 0, id + 0 from tb_scanback_paths where id_eventbrick = " + dstext[0] + " and id_processoperation = " + dstext[1] + " and path = " + dstext[2] + ") " +
                    " ) on (id_eventbrick = idb and id_zone = idz and id = id_candidate) " +
                    ") on (id_eventbrick = idb and id = id_plate) ORDER BY Z DESC", 
                    conn).Fill(ds1);
                if (ds1.Tables[0].Rows.Count <= 0) return;
                SySal.TotalScan.Flexi.Track tk = new SySal.TotalScan.Flexi.Track(dsinfo, 0);
                int i;
                long bk = System.Convert.ToInt64(dstext[0]);
                for (i = 0; i < ds1.Tables[0].Rows.Count; i++)
                {
                    System.Data.DataRow dr = ds1.Tables[0].Rows[i];
                    SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    info.Field = (uint)SySal.OperaDb.Convert.ToInt32(dr[0]);
                    info.Count = (ushort)SySal.OperaDb.Convert.ToInt32(dr[2]);
                    info.AreaSum = (uint)SySal.OperaDb.Convert.ToInt32(dr[3]);
                    info.Intercept.Z = SySal.OperaDb.Convert.ToDouble(dr[1]);
                    info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[4]);
                    info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[5]);
                    info.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[6]);
                    info.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[7]);
                    info.Slope.Z = 1.0;
                    info.Sigma = SySal.OperaDb.Convert.ToDouble(dr[8]);
                    info.BottomZ = info.Intercept.Z - 255.0;
                    info.TopZ = info.Intercept.Z + 45.0;
                    SySal.TotalScan.Flexi.Segment sg = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(info, new SySal.TotalScan.NullIndex()), dsinfo);
                    sg.SetLayer(new SySal.TotalScan.Flexi.Layer(i, bk, (int)info.Field, 0), 0);
                    tk.AddSegments(new SySal.TotalScan.Flexi.Segment[1] { sg });
                }
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("SBSFProcOp"), (double)System.Convert.ToInt64(dstext[1]));
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("SBSFPath"), (double)System.Convert.ToInt64(dstext[2]));
                DataDesc nd = new DataDesc("SBSF " + dstext[0] + " " + dstext[1], new SySal.TotalScan.Flexi.Track[1] { tk });
                for (i = 0; i < m_InteractiveDisplayImports.Count; i++)
                {
                    DataDesc d = (DataDesc)m_InteractiveDisplayImports[i];
                    if (d.Text == nd.Text)
                    {
                        SySal.TotalScan.Flexi.Track [] oa = (SySal.TotalScan.Flexi.Track [])d.O;
                        SySal.TotalScan.Flexi.Track [] na = new SySal.TotalScan.Flexi.Track[oa.Length + 1];
                        int j;
                        for (j = 0; j < oa.Length; j++) na[j] = oa[j];
                        tk.SetId(j);
                        na[j] = tk;
                        d.O = na;
                        break;
                    }
                }
                if (i == m_InteractiveDisplayImports.Count)
                {
                    m_InteractiveDisplayImports.Add(nd);
                    lvImportedInfo.Items.Add("SBSF " + dstext[0] + " " + dstext[1]);
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB import error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
            }
        }

        private void DBImportCS(string text)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                string[] dstext = text.Split(' ');
                SySal.TotalScan.Flexi.DataSet dsinfo = new SySal.TotalScan.Flexi.DataSet();
                dsinfo.DataId = System.Convert.ToInt64(dstext[1]);               
                dsinfo.DataType = "CS";
                System.Data.DataSet ds1 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter(
                    "select least(g11, g12), g11 + g12, least(g21, g22), g21 + g22, a11 + a12, a21 + a22, z1, z2," +
                    " round(decode(g11,0,px12 + (z11-z12) * sx11, px11),1) as px1, round(decode(g11,0,py12 + (z11-z12) * sy11, py11),1) as py1, round((px11 - px12) / (z11 - z12),4) as slopex1, round((py11 - py12) / (z11 - z12),4) as slopey1, " +
                    " round(decode(g21,0,px22 + (z21-z22) * sx21, px21),1) as px2, round(decode(g21,0,py22 + (z21-z22) * sx21, py21),1) as py2, round((px21 - px22) / (z21 - z22),4) as slopex2, round((py21 - py22) / (z21 - z22),4) as slopey2 from " +
                    "(select sum(sx * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as sx11, sum(sy * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as sy11, " +
                    " sum(sx * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as sx12, sum(sy * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as sy12, " +
                    " sum(sx * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as sx21, sum(sy * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as sy21, " +
                    " sum(sx * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as sx22, sum(sy * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as sy22, " +
                    " sum(Z * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as z1, sum(Z * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as z2, " +
                    " sum(grains * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as g11, sum(areasum * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as a11, " +
                    " sum(px * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as px11, sum(py * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as py11, sum(upz * decode(abs(side - 1) + abs(id_plate - 1),0,1,0)) as z11, " +
                    " sum(grains * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as g12, sum(areasum * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as a12, " +
                    " sum(px * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as px12, sum(py * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as py12, sum(downz * decode(abs(side - 2) + abs(id_plate - 1),0,1,0)) as z12, " +
                    " sum(grains * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as g21, sum(areasum * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as a21, " + 
                    " sum(px * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as px21, sum(py * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as py21, sum(upz * decode(abs(side - 1) + abs(id_plate - 2),0,1,0)) as z21, " +
                    " sum(grains * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as g22, sum(areasum * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as a22, " +
                    " sum(px * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as px22, sum(py * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as py22, sum(downz * decode(abs(side - 2) + abs(id_plate - 2),0,1,0)) as z22 from " + 
                    "(select id_plate, Z, side, grains, areasum, px, py, sx, sy, upz, downz from tb_views right join " +
                    " (select idb, id_plate, idz, Z, sd, grains, areasum, px, py, sx, sy, id_view from tb_plates inner join " +
                    "  (select idb, id_plate, idz, side as sd, grains, areasum, px, py, sx, sy, id_view from tb_zones inner join " +
                    "   (select /*+index(tb_mipmicrotracks pk_mipmicrotracks)*/ idb, idz, side, grains, areasum, posx as px, posy as py, slopex as sx, slopey as sy, id_view from tb_mipmicrotracks inner join " + 
                    "    (select id_eventbrick as idb, id_zone as idz, side as sd, id_microtrack from tb_cs_candidate_tracks where (id_eventbrick, id_candidate) in " + 
                    "     (select id_eventbrick, id from tb_cs_candidates where id_eventbrick = " + dstext[0] + " and id_processoperation = " + dstext[1] + " and candidate = " + dstext[2] + ") " + 
                    "    ) on (id_eventbrick = idb and id_zone = idz and side = sd and id = id_microtrack) " + 
                    "   ) on (id_eventbrick = idb and id = idz) " +
                    "  ) on (id_eventbrick = idb and id = id_plate) " +
                    " ) on (id_eventbrick = idb and id_zone = idz and side = sd and id_view = id)) " +
                    ")", conn).Fill(ds1);
                if (ds1.Tables[0].Rows.Count != 1) return;
                long bk = System.Convert.ToInt64(dstext[0]);
                SySal.TotalScan.Flexi.Track tk = new SySal.TotalScan.Flexi.Track(dsinfo, 0);
                System.Data.DataRow dr = ds1.Tables[0].Rows[0];
                SySal.Tracking.MIPEmulsionTrackInfo info1 = new SySal.Tracking.MIPEmulsionTrackInfo();
                info1.Field = 1;
                info1.Count = SySal.OperaDb.Convert.ToUInt16(dr[1]);
                info1.AreaSum = SySal.OperaDb.Convert.ToUInt32(dr[4]);
                info1.Intercept.Z = SySal.OperaDb.Convert.ToDouble(dr[6]);
                info1.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[8]);
                info1.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[9]);
                info1.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[10]);
                info1.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[11]);
                info1.Slope.Z = 1.0;
                info1.Sigma = (SySal.OperaDb.Convert.ToInt32(dr[0]) != 0) ? 0.1 : -1.0;
                info1.TopZ = info1.Intercept.Z + 45.0;
                info1.BottomZ = info1.Intercept.Z - 255.0;
                SySal.Tracking.MIPEmulsionTrackInfo info2 = new SySal.Tracking.MIPEmulsionTrackInfo();
                info2.Field = 2;
                info2.Count = SySal.OperaDb.Convert.ToUInt16(dr[3]);
                info2.AreaSum = SySal.OperaDb.Convert.ToUInt32(dr[5]);
                info2.Intercept.Z = SySal.OperaDb.Convert.ToDouble(dr[7]);
                info2.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[12]);
                info2.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[13]);
                info2.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[14]);
                info2.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[15]);
                info2.Slope.Z = 1.0;
                info2.Sigma = (SySal.OperaDb.Convert.ToInt32(dr[2]) != 0) ? 0.1 : -1.0;
                info2.TopZ = info2.Intercept.Z + 45.0;
                info2.BottomZ = info2.Intercept.Z - 255.0;
                if (info1.Sigma < 0.0) info1.Slope = info2.Slope;
                else if (info2.Sigma < 0.0) info2.Slope = info1.Slope;                
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("CSProcOp"), (double)System.Convert.ToInt64(dstext[1]));
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("CSCand"), (double)System.Convert.ToInt64(dstext[2]));
                SySal.TotalScan.Flexi.Segment sg1 = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(info1, new SySal.TotalScan.NullIndex()), dsinfo);
                sg1.SetLayer(new SySal.TotalScan.Flexi.Layer(1, bk, 1, 0), 0);
                SySal.TotalScan.Flexi.Segment sg2 = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(info2, new SySal.TotalScan.NullIndex()), dsinfo);
                sg2.SetLayer(new SySal.TotalScan.Flexi.Layer(2, bk, 2, 0), 0);
                tk.AddSegments(new SySal.TotalScan.Flexi.Segment[2] { sg1, sg2 });
                DataDesc nd = new DataDesc("CS " + dstext[0] + " " + dstext[1], new SySal.TotalScan.Flexi.Track[1] { tk });
                int i;
                for (i = 0; i < m_InteractiveDisplayImports.Count; i++)
                {
                    DataDesc d = (DataDesc)m_InteractiveDisplayImports[i];
                    if (d.Text == nd.Text)
                    {
                        SySal.TotalScan.Flexi.Track[] oa = (SySal.TotalScan.Flexi.Track[])d.O;
                        SySal.TotalScan.Flexi.Track[] na = new SySal.TotalScan.Flexi.Track[oa.Length + 1];
                        int j;
                        for (j = 0; j < oa.Length; j++) na[j] = oa[j];
                        tk.SetId(j);
                        na[j] = tk;
                        d.O = na;
                        break;
                    }
                }
                if (i == m_InteractiveDisplayImports.Count)
                {
                    m_InteractiveDisplayImports.Add(nd);
                    lvImportedInfo.Items.Add("CS " + dstext[0] + " " + dstext[1]);
                }

            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB import error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
            }
        }

        private void DBImportTT(string text)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();                
                string[] dstext = text.Split(' ');
                SySal.TotalScan.Flexi.DataSet dsinfo = new SySal.TotalScan.Flexi.DataSet();
                dsinfo.DataType = "TT";
                dsinfo.DataId = System.Convert.ToInt64(dstext[0]);                
                System.Data.DataSet ds1 = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter(
                    "select id_cs_eventbrick, idev, track, posx * 1000 as px, posy * 1000 as py, slopex, slopey, tktyp, pdgid, momentum from tb_predicted_tracks " +
                    "inner join (select id_cs_eventbrick, id_event as idev, track as idtk, pdgid, type as tktyp, momentum from tv_predtrack_brick_assoc where mod(id_cs_eventbrick, 1000000) = " + (m_TSRDS.DataId % 1000000).ToString() + ")" +
                    "on (id_event = idev and track = idtk and track = " + dstext[1] + ")", conn).Fill(ds1);                
                if (ds1.Tables[0].Rows.Count != 1) return;                
                System.Data.DataRow dr = ds1.Tables[0].Rows[0];
                long bk = System.Convert.ToInt64(dr[0]);
                SySal.TotalScan.Flexi.Track tk = new SySal.TotalScan.Flexi.Track(dsinfo, 0);
                SySal.Tracking.MIPEmulsionTrackInfo info1 = new SySal.Tracking.MIPEmulsionTrackInfo();
                info1.Field = 1;
                info1.Count = 0;
                info1.AreaSum = 0;
                info1.Intercept.Z = 4850.0;
                info1.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[3]);
                info1.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[4]);
                info1.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[5]);
                info1.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[6]);
                info1.Slope.Z = 1.0;
                info1.Sigma = -1.0;
                info1.TopZ = 4850.0;
                info1.BottomZ = 4300.0;
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("TTEvent"), (double)System.Convert.ToInt64(dstext[0]));
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PDGID"), (double)System.Convert.ToInt64(dr[8]));
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("TYPE_" + dr[7].ToString()), 1.0);
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("TTMomentum"), System.Convert.ToDouble(dr[9]));
                SySal.TotalScan.Flexi.Segment sg1 = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(info1, new SySal.TotalScan.NullIndex()), dsinfo);
                sg1.SetLayer(new SySal.TotalScan.Flexi.Layer(1, bk, 1, 0), 0);
                tk.AddSegments(new SySal.TotalScan.Flexi.Segment[1] { sg1 });
                DataDesc nd = new DataDesc("TT " + dstext[0] + " " + dstext[1], new SySal.TotalScan.Flexi.Track[1] { tk });
                int i;
                for (i = 0; i < m_InteractiveDisplayImports.Count; i++)
                {
                    DataDesc d = (DataDesc)m_InteractiveDisplayImports[i];
                    if (d.Text == nd.Text)
                    {
                        SySal.TotalScan.Flexi.Track[] oa = (SySal.TotalScan.Flexi.Track[])d.O;
                        SySal.TotalScan.Flexi.Track[] na = new SySal.TotalScan.Flexi.Track[oa.Length + 1];
                        int j;
                        for (j = 0; j < oa.Length; j++) na[j] = oa[j];
                        tk.SetId(j);
                        na[j] = tk;
                        d.O = na;
                        break;
                    }
                }
                if (i == m_InteractiveDisplayImports.Count)
                {
                    m_InteractiveDisplayImports.Add(nd);
                    lvImportedInfo.Items.Add(DBTTString + text);
                }

            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB import error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
            }
        }

        private SySal.TotalScan.Flexi.Volume PrepareVolume()
        {
            int i, j;
            SySal.TotalScan.Flexi.Volume myvol = new SySal.TotalScan.Flexi.Volume();
            {
                SySal.TotalScan.Flexi.DataSet ds0 = new SySal.TotalScan.Flexi.DataSet();
                ds0.DataType = m_TSRDS.DataType;
                ds0.DataId = m_TSRDS.DataId;
                myvol.ImportVolume(ds0, v);
                for (i = 0; i < myvol.Layers.Length; i++)
                {
                    SySal.TotalScan.AlignmentData al = myvol.Layers[i].AlignData;
                    if (chkResetSlopeCorrections.Checked)
                    {
                        al.SAlignDSlopeX = al.SAlignDSlopeY = 0.0;
                        al.DShrinkX = al.DShrinkY = 1.0;
                    }
                    al.TranslationZ = 0.0;
                    ((SySal.TotalScan.Flexi.Layer)myvol.Layers[i]).SetAlignment(al);
                }
            }
            System.Collections.ArrayList pgeom = new ArrayList();
            foreach (DataDesc d in m_InteractiveDisplayImports)
                if (d.Text.StartsWith("Geom"))
                {
                    Geometry g = (Geometry)d.O;
                    foreach (Geometry.LayerStart ls in g.Layers)
                    {
                        pgeom.Add(ls);
                        if (ls.Plate <= 0) continue;                        
                        for (i = 0; i < myvol.Layers.Length; i++)
                            if ((myvol.Layers[i].BrickId == ls.Brick || myvol.Layers[i].BrickId == 0) && myvol.Layers[i].SheetId == ls.Plate) break;
                        if (i < myvol.Layers.Length)
                        {
                            SySal.TotalScan.Flexi.Layer lay = (SySal.TotalScan.Flexi.Layer)myvol.Layers[i];
                            lay.DisplaceAndClampZ(ls.ZMin, ls.ZMin + 300.0);
                            lay.SetRadiationLength(ls.RadiationLength);
                            SySal.BasicTypes.Vector r = lay.RefCenter;
                            r.Z = ls.ZMin + 255.0;
                            lay.SetRefCenter(r);
                            if (lay.BrickId == 0) lay.SetBrickId(m_TSRDS.DataId);
                        }
                    }
                    foreach (Geometry.LayerStart ls in g.Layers)
                    {
                        if (ls.Plate <= 0) continue;                        
                        for (i = 0; i < myvol.Layers.Length; i++)
                            if ((myvol.Layers[i].BrickId == ls.Brick || myvol.Layers[i].BrickId == 0) && myvol.Layers[i].SheetId == ls.Plate) break;
                        if (i == myvol.Layers.Length)
                        {
                            SySal.TotalScan.Flexi.Layer bestlay = (SySal.TotalScan.Flexi.Layer)myvol.Layers[0];
                            double bestdz = Math.Abs(bestlay.UpstreamZ - ls.ZMin);
                            double dz;
                            for (i = 1; i < myvol.Layers.Length; i++)
                                if ((dz = Math.Abs(myvol.Layers[i].UpstreamZ - ls.ZMin)) < bestdz)
                                {
                                    bestdz = dz;
                                    bestlay = (SySal.TotalScan.Flexi.Layer)myvol.Layers[i];
                                }
                            SySal.TotalScan.Flexi.DataSet myds = new SySal.TotalScan.Flexi.DataSet();
                            myds.DataId = ls.Brick;
                            myds.DataType = "Geom";
                            SySal.TotalScan.Flexi.Layer newlay = new SySal.TotalScan.Flexi.Layer(0, ls.Brick, ls.Plate, 0);
                            newlay.SetUpstreamZ(ls.ZMin);
                            newlay.SetDownstreamZ(ls.ZMin + 300.0);
                            SySal.BasicTypes.Vector r = bestlay.RefCenter;
                            r.Z = ls.ZMin + 255.0;
                            newlay.SetRefCenter(r);
                            newlay.SetAlignment(bestlay.AlignData);
                            newlay.SetRadiationLength(ls.RadiationLength);
                            ((SySal.TotalScan.Flexi.Volume.LayerList)myvol.Layers).Insert(newlay);
                        }
                    }
                    int il, jl;
                    for (il = 0; il < g.Layers.Length - 1; il++)
                    {
                        if (g.Layers[il].Plate > 0)
                            if (g.Layers[il + 1].Plate <= 0)
                            {
                                for (jl = 0; jl < myvol.Layers.Length && (myvol.Layers[jl].SheetId != g.Layers[il].Plate || myvol.Layers[jl].BrickId != g.Layers[il].Brick); jl++) ;
                                if (jl < myvol.Layers.Length)
                                {
                                    ((SySal.TotalScan.Flexi.Layer)myvol.Layers[jl]).SetDownstreamRadiationLength(g.Layers[il + 1].RadiationLength);
                                    if (jl > 0) ((SySal.TotalScan.Flexi.Layer)myvol.Layers[jl - 1]).SetUpstreamRadiationLength(g.Layers[il + 1].RadiationLength);
                                }
                            }
                    }
                }
            foreach (DataDesc d in m_InteractiveDisplayImports)
                if (d.O is SySal.TotalScan.Flexi.Track[])
                {
                    SySal.TotalScan.Flexi.DataSet rds = null;
                    SySal.TotalScan.Flexi.Track[] ta = (SySal.TotalScan.Flexi.Track[])d.O;
                    ((SySal.TotalScan.Flexi.Volume.TrackList)myvol.Tracks).Insert(ta);
                    foreach (SySal.TotalScan.Flexi.Track tk in ta)
                    {
                        if (rds == null) rds = tk.DataSet;
                        else tk.DataSet = rds;                        
                        for (i = 0; i < tk.Length; i++)
                        {
                            SySal.TotalScan.Flexi.Segment seg = (SySal.TotalScan.Flexi.Segment)tk[i];
                            for (j = 0; j < myvol.Layers.Length; j++)
                                if (myvol.Layers[j].SheetId == seg.LayerOwner.SheetId && myvol.Layers[j].BrickId == seg.LayerOwner.BrickId)
                                    break;
                            if (j == myvol.Layers.Length)
                            {
                                MessageBox.Show("Segments found on layers that had not been imported.\r\nPlease import all plates relevant to the data.", "Inconsistent geometry", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                return null;
                            }
                            SySal.TotalScan.Flexi.Layer fl = (SySal.TotalScan.Flexi.Layer)myvol.Layers[j];
                            SySal.Tracking.MIPEmulsionTrackInfo info = seg.Info;
                            SySal.BasicTypes.Vector i2 = fl.ToAlignedPoint(info.Intercept);
                            info.Intercept.X = i2.X;
                            info.Intercept.Y = i2.Y;
                            double dz = fl.RefCenter.Z - info.Intercept.Z;
                            info.TopZ += dz;
                            info.BottomZ += dz;
                            info.Intercept.Z += dz;
                            info.Slope = fl.ToAlignedSlope(info.Slope);
                            seg.SetInfo(info);
                            seg.DataSet = rds;
                            fl.Add(new SySal.TotalScan.Flexi.Segment[1] { seg });
                        }
                    }
                }
                else if (d.O is SySal.TotalScan.Flexi.Volume)
                {
                    SySal.TotalScan.Flexi.DataSet ids = new SySal.TotalScan.Flexi.DataSet();
                    ids.DataId = m_TSRDS.DataId;
                    ids.DataType = txtImportedDS.Text;
                    SySal.TotalScan.Flexi.DataSet rds = new SySal.TotalScan.Flexi.DataSet();
                    rds.DataId = m_TSRDS.DataId;
                    rds.DataType = txtResetDS.Text;
                    System.IO.StringWriter tw = new System.IO.StringWriter();
#if !DEBUG
                    try
                    {
#endif
                        SySal.Processing.MapMerge.MapManager.dMapFilter dflt = null;
                        if (cmbMapMergeFilter.Text.Length > 0) dflt = new SySal.Processing.MapMerge.MapManager.dMapFilter(new MapMergeFilterAdapter(cmbMapMergeFilter.Text).Filter);
                        m_MapMerger.AddToVolume(myvol, (SySal.TotalScan.Flexi.Volume)d.O, rds, ids, dflt, tw);
#if !DEBUG
                    }
                    catch (Exception x)
                    {
                        Report(x.ToString());
                    }
                    finally
                    {
                        Report(tw.ToString());
                    }
#endif
                }

            for (i = 0; i < myvol.Tracks.Length; i++) myvol.Tracks[i].NotifyChanged();
            int vtxkilled = 0;
            for (i = 0; i < myvol.Vertices.Length; i++)
                try
                {
                    myvol.Vertices[i].NotifyChanged();
                    if (myvol.Vertices[i].AverageDistance >= 0.0)
                        ((SySal.TotalScan.Flexi.Vertex)(myvol.Vertices[i])).SetId(i);
                }
                catch (Exception)
                {
                    vtxkilled++;
                    SySal.TotalScan.Vertex vtxk = myvol.Vertices[i];
                    for (j = 0; j < vtxk.Length; j++)
                    {
                        SySal.TotalScan.Track tk = vtxk[j];
                        if (tk.Upstream_Vertex == vtxk) tk.SetUpstreamVertex(null);
                        else tk.SetDownstreamVertex(null);
                    }
                    ((SySal.TotalScan.Flexi.Volume.VertexList)myvol.Vertices).Remove(new int[1] { i });
                    i--;
                }
            if (vtxkilled > 0)
                MessageBox.Show(((vtxkilled == 1) ? "1 vertex was invalid with the new geometry and has been killed." : 
                    (vtxkilled + " vertices were invalid with the new geometry and have been killed.")) + 
                    "\r\nVertex IDs have been renumbered.", "Reconstruction modified", MessageBoxButtons.OK, MessageBoxIcon.Warning);

            pgeom.Sort(new LayerComparer());
            SySal.Processing.MCSLikelihood.Configuration mcsg = (SySal.Processing.MCSLikelihood.Configuration)MCSLikelihood.Config;
            mcsg.Geometry = new Geometry(myvol.Layers);                       
            //mcsg.Geometry.Layers = (Geometry.LayerStart [])pgeom.ToArray(typeof(Geometry.LayerStart));
            MCSLikelihood.Config = mcsg;
            return myvol;
        }

        private class LayerComparer : IComparer
        {

            #region IComparer Members

            public int Compare(object x, object y)
            {
                double a = ((Geometry.LayerStart)x).ZMin - ((Geometry.LayerStart)y).ZMin;
                if (a < 0.0) return -1;
                if (a > 0.0) return 1;
                return 0;
            }

            #endregion
        }

        private void SaveImportedInfoButton_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select ASCII file to dump imported information.";
            sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            System.IO.StreamWriter w = null;
            if (sdlg.ShowDialog() == DialogResult.OK)
                try
                {
                    w = new System.IO.StreamWriter(sdlg.FileName);
                    foreach (DataDesc d in m_InteractiveDisplayImports)
                    {
                        if (d.O is SySal.TotalScan.Flexi.Track[])
                            ASCIIDumpTracks(w, (SySal.TotalScan.Flexi.Track[])d.O);
                        else if (d.O is Geometry)
                            ASCIIDumpGeometry(w, (Geometry)d.O);
                        else
                            throw new Exception("Dump not supported for datatype " + d.O.GetType());
                    }
                    w.Flush();
                    w.Close();
                    w = null;
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (w != null)
                    {
                        w.Close();
                        w = null;
                    }
                }
        }

        private void ASCIIDumpTracks(System.IO.StreamWriter w, SySal.TotalScan.Flexi.Track[] tks)
        {
            foreach (SySal.TotalScan.Flexi.Track tk in tks)
            {
                SySal.TotalScan.Flexi.DataSet ds = tk.DataSet;
                SySal.TotalScan.Attribute[] a = tk.ListAttributes();
                w.WriteLine("TRACK\t" + tk.Id + "\t" + tk.DataSet.DataType + "\t" + tk.DataSet.DataId + "\t" + tk.DataSet.DataId.ToString("G28", System.Globalization.CultureInfo.InvariantCulture) + "\tSEGMENTS\t" + tk.Length + "\tATTRIBUTES\t" + a.Length);
                w.WriteLine("PLATE\tSIDE\tGRAINS\tAREASUM\tPOSX\tPOSY\tPOSZ\tSLOPEX\tSLOPEY\tSIGMA");
                int i;
                for (i = 0; i < tk.Length; i++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = tk[i].Info;
                    w.WriteLine(tk[i].LayerOwner.SheetId + "\t" + tk[i].LayerOwner.Side + "\t" + info.Count + "\t" + info.AreaSum + "\t" +
                        info.Intercept.X.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        info.Intercept.Y.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        info.Intercept.Z.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        info.Slope.X.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        info.Slope.Y.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        info.Sigma.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
                w.WriteLine("INDEX\tTYPE\tVALUE");
                foreach (SySal.TotalScan.Attribute attr in a)
                    w.WriteLine(attr.Index.ToString() + "\t" + attr.Index.GetType() + "\t" + attr.Value.ToString("G20", System.Globalization.CultureInfo.InvariantCulture));
            }
            w.WriteLine();
            w.Flush();            
        }

        private void ASCIIDumpGeometry(System.IO.StreamWriter w, Geometry g)
        {
            w.WriteLine("GEOM\t" + g.Layers.Length);
            w.WriteLine("BRICK\tPLATE\tSIDE\tZMIN\tRADLEN");
            foreach (Geometry.LayerStart ls in g.Layers)
                w.WriteLine(ls.Brick + "\t" + ls.Plate + "\t0\t" + ls.ZMin.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" + ls.RadiationLength.ToString(System.Globalization.CultureInfo.InvariantCulture));
            w.WriteLine();
            w.Flush();
        }

        private void btnOpenFile_Click(object sender, EventArgs e)
        {
            System.IO.StreamReader r = null;
            System.IO.FileStream rv = null;
            OpenFileDialog odlg = new OpenFileDialog();
            odlg.Title = "Select file to open.";
            odlg.Filter = "Text files (*.txt)|*.txt|TSR files (*.tsr)|*.tsr|All files (*.*)|*.*";
            if (odlg.ShowDialog() == DialogResult.OK)
                try
                {
                    if (odlg.FileName.ToLower().EndsWith(".tsr"))
                    {
                        rv = new System.IO.FileStream(odlg.FileName, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                        SySal.TotalScan.Flexi.Volume iv = new SySal.TotalScan.Flexi.Volume();
                        SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
                        ds.DataId = m_TSRDS.DataId;
                        ds.DataType = txtImportedDS.Text;
                        iv.ImportVolume(ds, new Volume(rv));
                        rv.Close();
                        rv = null;
                        DataDesc nd = new DataDesc(IMPVOLString + odlg.FileName, iv);
                        m_InteractiveDisplayImports.Add(nd);
                        lvImportedInfo.Items.Add(nd.Text);
                    }
                    else
                    {
                        r = new System.IO.StreamReader(odlg.FileName);
                        FileImportManualCheck(r);
                        r.Close();
                        r = null;
                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    if (r != null) r.Close();
                    if (rv != null) rv.Close();
                }
        }

        private void btnFileFormats_Click(object sender, EventArgs e)
        {
            new QBrowser("Supported file formats",
                "TSR format (for data merge)\r\n" + 
                "Formats for manual checks:\r\n" +
                "\r\nType #1:\r\n" +
                "\r\nTrackId\tPX\tPY\tSX\tSY\tPlate\tFlag\tZ1\tZ2\r\n" +
                "\r\nTrackId = Track to which the segment belongs (ignored)\r\n" +
                "\r\nPX = X position (at downstream plastic surface for base tracks and downstream microtracks, at upstream plastic surface for upstream microtracks)\r\n" +
                "\r\nPY = Y position (at downstream plastic surface for base tracks and downstream microtracks, at upstream plastic surface for upstream microtracks)\r\n" +
                "\r\nSX = X slope\r\n" +
                "\r\nSY = Y slope\r\n" + 
                "\r\nPlate = Plate id\r\n" + 
                "\r\nFlag = 0 (Base track), 1 (Downstream microtrack), 2 (Upstream microtrack), < 0 (not found)" +
                "\r\nZ1 = Downstream emulsion surface for downstream microtracks, downstream plastic surface for base tracks, upstream plastic surface for upstream microtracks\r\n" + 
                "\r\nZ2 = Downstream plastic surface for downstream microtracks, upstream plastic surface for base tracks, upstream emulsion surface for upstream microtracks\r\n" +                 
                "\r\nType #2:\r\nBrick\tPlate\tSide\tTrack\tSurfZ1\tSurfZ2\tSurfZ3\tSurfZ4\tX1\tY1\tZ1\tX2\tY2\tZ2\tGrains\tOK" +
                "\r\nBrick = Brick the plate belongs to" +
                "\r\nPlate = Plate where measurements are done" +
                "\r\nSide = 0 for base tracks, 1 for downstream, 2 for upstream" +
                "\r\nTrack = ID of the track to be measured" +
                "\r\nSurfZ1/2/3/4 = Z of emulsion/plastic/air surfaces" +
                "\r\nX/Y/Z 1/2 = coordinates of the two points measured" +
                "\r\nGrains = number of grains in the microtrack" +
                "\r\nOK = 1 if the track is found, 0 if not found").ShowDialog();
        }

        static System.Text.RegularExpressions.Regex manchk1_rx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");

        static System.Text.RegularExpressions.Regex manchk2_rx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s*");

        private void FileImportManualCheck(System.IO.StreamReader r)
        {            
            string line = null;
            System.Text.RegularExpressions.Match m = null;
            System.Collections.ArrayList sl = new ArrayList();
            while ((line = r.ReadLine()) != null)
                if (line.Trim().Length <= 0) continue;
                else if ((m = manchk1_rx.Match(line)).Success == true && m.Length == line.Length)
                    try
                    {
                        SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
                        ds.DataId = m_TSRDS.DataId;
                        ds.DataType = "MANCHECK";
                        SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                        info.Field = System.Convert.ToUInt32(m.Groups[6].Value);
                        int datacode = System.Convert.ToInt32(m.Groups[7].Value);
                        if (datacode < 0) continue;
                        int darkness = 0;
                        int side = datacode % 3;
                        darkness = datacode / 3;
                        int trackid = System.Convert.ToInt32(m.Groups[1].Value);                        
                        info.Intercept.X = System.Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Intercept.Y = System.Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);                        
                        info.Slope.X = System.Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Slope.Y = System.Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Slope.Z = 1.0;
                        info.TopZ = System.Convert.ToDouble(m.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.BottomZ = System.Convert.ToDouble(m.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                        switch (side)
                        {
                            case 0: info.Intercept.Z = 0.0; info.BottomZ -= info.TopZ; info.TopZ = 45.0; info.Sigma = 0.1; break;
                            case 1: info.Intercept.Z = 0.0; info.TopZ -= info.BottomZ; info.BottomZ = 0.0;
                                info.Intercept.X -= 45.0 * info.Slope.X; info.Intercept.Y -= 45.0 * info.Slope.Y; info.Sigma = -1.0; break;
                            case 2: info.Intercept.Z = 0.0; info.BottomZ -= info.TopZ; info.TopZ = -210.0; info.BottomZ += info.TopZ;
                                info.Intercept.X += 210.0 * info.Slope.X; info.Intercept.Y += 210.0 * info.Slope.Y; info.Sigma = -1.0; break;
                        }
                        info.Count = 0;
                        info.AreaSum = 0;                                                               
                        SySal.TotalScan.Flexi.Track tk = new SySal.TotalScan.Flexi.Track(ds, trackid);
                        sl.Add(tk);                        
                        SySal.TotalScan.Flexi.Segment sg = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(info, new SySal.TotalScan.NullIndex()), ds);
                        sg.SetLayer(new SySal.TotalScan.Flexi.Layer(0, ds.DataId, (int)info.Field, (short)datacode), 0);
                        tk.AddSegments(new SySal.TotalScan.Flexi.Segment[1] { sg });
                        if (darkness <= (int)TrackBrowser.FBDarkness.Black) tk.SetAttribute(TrackBrowser.FBDarknessIndex, darkness * 0.5);
                        else
                        {
                            tk.SetAttribute(TrackBrowser.FBParticleIndex, (double)TrackBrowser.FBParticleType.EPair);
                            tk.SetAttribute(TrackBrowser.FBDecaySearchIndex, (double)TrackBrowser.FBDecaySearch.EPlusEMinus);
                        }                        
                    }
                    catch (Exception x)
                    {
                        MessageBox.Show("Error (Trying type 1):\r\n" + x.Message + "\r\nAt line:\r\n" + line, "File import error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        continue;
                    }
                else if ((m = manchk2_rx.Match(line)).Success == true && m.Length == line.Length)
                    try
                    {
                        SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
                        ds.DataId = System.Convert.ToInt64(m.Groups[1].Value);
                        ds.DataType = "MANCHECK";                        
                        SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                        info.Field = System.Convert.ToUInt32(m.Groups[2].Value);
                        int side = System.Convert.ToInt32(m.Groups[3].Value);
                        int trackid = System.Convert.ToInt32(m.Groups[4].Value);                        
                        if (side != 0) throw new Exception("Only Side = 0 is currently supported");
                        System.Collections.ArrayList zs = new ArrayList(4);
                        zs.Add(System.Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture));
                        zs.Add(System.Convert.ToDouble(m.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture));
                        zs.Add(System.Convert.ToDouble(m.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture));
                        zs.Add(System.Convert.ToDouble(m.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture));
                        zs.Sort();
                        info.Intercept.X = System.Convert.ToDouble(m.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Intercept.Y = System.Convert.ToDouble(m.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Intercept.Z = System.Convert.ToDouble(m.Groups[11].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Slope.X = System.Convert.ToDouble(m.Groups[12].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Slope.Y = System.Convert.ToDouble(m.Groups[13].Value, System.Globalization.CultureInfo.InvariantCulture);
                        info.Slope.Z = System.Convert.ToDouble(m.Groups[14].Value, System.Globalization.CultureInfo.InvariantCulture);
                        if (info.Slope.Z > info.Intercept.Z)
                        {
                            SySal.BasicTypes.Vector swap = info.Intercept;
                            info.Intercept = info.Slope;
                            info.Slope = swap;
                        }                        
                        info.TopZ = info.Intercept.Z;
                        info.BottomZ = info.Slope.Z;
                        info.Slope.X = (info.Intercept.X - info.Slope.X) / (info.Intercept.Z - info.Slope.Z);
                        info.Slope.Y = (info.Intercept.Y - info.Slope.Y) / (info.Intercept.Z - info.Slope.Z);
                        info.Slope.Z = 1.0;
                        info.Count = System.Convert.ToUInt16(m.Groups[15].Value);
                        if (System.Convert.ToInt32(m.Groups[16].Value) <= 0) continue;
                        info.Sigma = 0.1;
                        info.AreaSum = 0;
                        SySal.TotalScan.Flexi.Track tk = null;
                        foreach (SySal.TotalScan.Flexi.Track itk in sl)
                            if (itk.Id == trackid)
                            {
                                tk = itk;
                                break;
                            }
                        if (tk == null)
                        {
                            tk = new SySal.TotalScan.Flexi.Track(ds, trackid);
                            sl.Add(tk);
                        }
                        SySal.TotalScan.Flexi.Layer nl = new SySal.TotalScan.Flexi.Layer(0, ds.DataId, (int)info.Field, 0);
                        nl.SetDownstreamZ(45.0);
                        nl.SetUpstreamZ(-255.0);
                        SySal.TotalScan.Flexi.Segment sg = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(info, new SySal.TotalScan.NullIndex()), ds);
                        sg.SetLayer(nl, 0);
                        tk.AddSegments(new SySal.TotalScan.Flexi.Segment [1] { sg });
                    }
                    catch (Exception x) 
                    {
                        MessageBox.Show("Error (Trying type 2):\r\n" + x.Message + "\r\nAt line:\r\n" + line, "File import error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        continue;                     
                    }
            foreach (SySal.TotalScan.Flexi.Track tk in sl)
            {
                int i;
                DataDesc nd = new DataDesc("MANCHK " + tk.DataSet.DataId, new SySal.TotalScan.Flexi.Track[1] { tk });
                for (i = 0; i < m_InteractiveDisplayImports.Count; i++)
                {
                    DataDesc d = (DataDesc)m_InteractiveDisplayImports[i];
                    if (d.Text == nd.Text)
                    {
                        SySal.TotalScan.Flexi.Track[] oa = (SySal.TotalScan.Flexi.Track[])d.O;
                        SySal.TotalScan.Flexi.Track[] na = new SySal.TotalScan.Flexi.Track[oa.Length + 1];
                        int j;
                        for (j = 0; j < oa.Length; j++) na[j] = oa[j];
                        tk.SetId(j);
                        na[j] = tk;
                        d.O = na;
                        break;
                    }
                }
                if (i == m_InteractiveDisplayImports.Count)
                {
                    m_InteractiveDisplayImports.Add(nd);
                    lvImportedInfo.Items.Add(nd.Text);
                }
            }
        }

        private void SaveAllToTSRButton_Click(object sender, EventArgs e)
        {
            System.IO.FileStream ws = null;
            try
            {
                if (v == null) throw new Exception("Volume not set to an object");
                SySal.TotalScan.Flexi.Volume myvol = PrepareVolume();
                if (myvol == null) throw new Exception("Error in data import");
                SaveFileDialog sdlg = new SaveFileDialog();
                sdlg.Title = "Save all to TSR format";
                sdlg.Filter = "TSR files (*.tsr)|*.tsr|All files (*.*)|*.*";
                if (sdlg.ShowDialog() == DialogResult.OK)
                {
                    ws = new System.IO.FileStream(sdlg.FileName, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                    myvol.Save(ws);
                    ws.Flush();
                    ws.Close();
                    ws = null;
                }
            }
            catch (Exception exc)
            {
                MessageBox.Show(exc.Message, "Selection error");
            }
            finally
            {
                if (ws != null)
                {
                    ws.Close();
                    ws = null;
                }
            }
        }

        private void OnTrackExtrapolationModeChanged(object sender, EventArgs e)
        {
            SySal.TotalScan.Track.TrackExtrapolationMode = (SySal.TotalScan.Track.ExtrapolationMode)cmbTrackExtrapolationMode.SelectedItem;
        }

        private void OnVertexTrackWeightingChanged(object sender, EventArgs e)
        {
            switch (cmbVtxTrackWeighting.SelectedIndex)
            {
                case 0: SySal.TotalScan.Vertex.TrackWeightingFunction = new SySal.TotalScan.Vertex.dTrackWeightFunction(SySal.TotalScan.Vertex.AttributeWeight); break;
                case 1: SySal.TotalScan.Vertex.TrackWeightingFunction = new SySal.TotalScan.Vertex.dTrackWeightFunction(SySal.TotalScan.Vertex.FlatWeight); break;
                case 2: SySal.TotalScan.Vertex.TrackWeightingFunction = new SySal.TotalScan.Vertex.dTrackWeightFunction(SySal.TotalScan.Vertex.SlopeScatteringWeight); break;
                default: throw new Exception("Unsupported weighting function.");
            }
        }

        SySal.Processing.MapMerge.MapMerger m_MapMerger = new SySal.Processing.MapMerge.MapMerger();

        static System.Xml.Serialization.XmlSerializer xmlMapMerger = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MapMerge.Configuration));

        class MapMergeFilterAdapter
        {
            SySal.Executables.EasyReconstruct.DisplayForm.ObjFilter m_ObjF;
            public MapMergeFilterAdapter(string t)
            {
                m_ObjF = new DisplayForm.ObjFilter(DisplayForm.SegmentFilterFunctions, t);
            }
            public bool Filter(object o)
            {
                return m_ObjF.Value(o) != 0.0;
            }
        }

        private void btnVolMergeConfig_Click(object sender, EventArgs e)
        {
            DialogResult dr = MessageBox.Show("Do you want to load a saved configuration?", "Information required", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);
            if (dr == DialogResult.Cancel) return;
            if (dr == DialogResult.Yes)
            {
                OpenFileDialog odlg = new OpenFileDialog();
                odlg.Title = "Select map merging configuration";
                odlg.Filter = "XML files (*.xml)|*.xml|All files (*.*)|*.*";
                if (odlg.ShowDialog() != DialogResult.OK) return;
                try
                {
                    System.IO.StringReader sr = new System.IO.StringReader(System.IO.File.ReadAllText(odlg.FileName));                    
                    SySal.Processing.MapMerge.Configuration cfg = (SySal.Processing.MapMerge.Configuration)xmlMapMerger.Deserialize(sr);
                    m_MapMerger.Config = cfg;
                }
                catch (Exception xcx)
                {
                    MessageBox.Show(xcx.ToString(), "Error loading Map Merging configuration", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            else
            {
                SySal.Management.Configuration cfg = m_MapMerger.Config;
                if (m_MapMerger.EditConfiguration(ref cfg))
                {
                    m_MapMerger.Config = cfg;
                    if (MessageBox.Show("Do you want to save the new configuration?", "Information required", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                        try
                        {
                            SaveFileDialog sdlg = new SaveFileDialog();
                            sdlg.Title = "Select a file to save the map merging configuration";
                            sdlg.Filter = "XML files (*.xml)|*.xml|All files (*.*)|*.*";
                            if (sdlg.ShowDialog() == DialogResult.OK)
                            {
                                System.IO.StringWriter sw = new System.IO.StringWriter();
                                xmlMapMerger.Serialize(sw, cfg);
                                System.IO.File.WriteAllText(sdlg.FileName, sw.ToString());
                            }
                        }
                        catch (Exception xcx)
                        {
                            MessageBox.Show(xcx.ToString(), "Error saving Map Merging configuration", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                }
            }
        }

        private void btnMapMergeFilterVars_Click(object sender, EventArgs e)
        {
            string helpstr = "";
            foreach (SySal.Executables.EasyReconstruct.DisplayForm.FilterF f in SySal.Executables.EasyReconstruct.DisplayForm.SegmentFilterFunctions)
                helpstr += "\r\n" + f.Name + " -> " + f.HelpText;
            MessageBox.Show(helpstr, "Map Merging filter variables", MessageBoxButtons.OK);
        }

        internal static void SaveProfile()
        {
            try
            {
                UserProfileInfo.Save();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error saving profile information", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void btnFilterAdd_Click(object sender, EventArgs e)
        {
            string s = cmbMapMergeFilter.Text.Trim();
            if (s.Length > 0)
            {
                foreach (string s1 in UserProfileInfo.ThisProfileInfo.MapMergeFilters)
                    if (String.Compare(s1, s, true) == 0) return;
                cmbMapMergeFilter.Items.Insert(0, s);
                string[] os = UserProfileInfo.ThisProfileInfo.MapMergeFilters;
                UserProfileInfo.ThisProfileInfo.MapMergeFilters = new string[os.Length + 1];
                UserProfileInfo.ThisProfileInfo.MapMergeFilters[0] = s;
                os.CopyTo(UserProfileInfo.ThisProfileInfo.MapMergeFilters, 1);
                MainForm.SaveProfile();
            }
        }

        private void btnFilterDel_Click(object sender, EventArgs e)
        {
            string s = cmbMapMergeFilter.Text.Trim();
            if (s.Length > 0)
            {
                int i, j;
                for (i = 0; i < UserProfileInfo.ThisProfileInfo.MapMergeFilters.Length && String.Compare(s, UserProfileInfo.ThisProfileInfo.MapMergeFilters[i], true) != 0; i++) ;
                if (i < UserProfileInfo.ThisProfileInfo.MapMergeFilters.Length)
                {
                    cmbMapMergeFilter.Items.RemoveAt(i);
                    string[] os = UserProfileInfo.ThisProfileInfo.MapMergeFilters;
                    UserProfileInfo.ThisProfileInfo.MapMergeFilters = new string[os.Length - 1];
                    for (j = 0; j < i; j++) UserProfileInfo.ThisProfileInfo.MapMergeFilters[j] = os[j];
                    for (++j; j < os.Length; j++) UserProfileInfo.ThisProfileInfo.MapMergeFilters[j - 1] = os[j];
                    MainForm.SaveProfile();
                }
            }
        }

        private void OnNodeAfterCheck(object sender, TreeViewEventArgs e)
        {
            foreach (TreeNode n in e.Node.Nodes)
                n.Checked = e.Node.Checked;
        }
    }
}