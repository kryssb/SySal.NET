using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using NumericalTools;
using NumericalTools.Scripting;

namespace SySal.Executables.QuickDataCheck
{
	/// <summary>
	/// QuickDataCheck - GUI program for data check and analysis.
	/// </summary>
	/// <remarks>
	/// <para>QuickDataCheck is able to read data files in the following formats:
	/// <list type="bullet">
	/// <item><term>SySal RWD</term></item>
	/// <item><term>SySal TLG</term></item>
	/// <item><term>SySal TSR</term></item>
	/// <item><term>Generic ASCII n-tuple file</term></item>
	/// </list>
	/// In addition, datasets from DB queries can be handled.
	/// </para>
	/// <para>QuickDataCheck supports all features from the <see cref="NumericalTools.AnalysisControl">StatisticalAnalysisManager</see>. Actually, the main form is an instance of that control.</para>	
	/// <para>A scripting engine allows editing and execution of analysis scripts with C-like syntax or Pascal-like syntax.</para>	
	/// </remarks>
	public class ExeForm : System.Windows.Forms.Form
	{
		private bool DBConnOwner = true;
		private SySal.OperaDb.OperaDbConnection m_DBConn = null;
		public SySal.OperaDb.OperaDbConnection DBConn
		{
			get { return m_DBConn; }
			set 
			{ 
				DBConnOwner = false; 
				if (m_DBConn != null) m_DBConn.Close();
				m_DBConn = value; 
				mnuConnect.Enabled = false;
				mnuQuickQuery.Enabled = true;
			}
		}

		private QuickDataCheck.ScriptEditor ScriptEd = null;


		private SySal.OperaDb.OperaDbCredentials m_DBCred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();

		private System.Windows.Forms.MainMenu mainMenu1;
		private System.Windows.Forms.OpenFileDialog openFileDialog1;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		private System.Windows.Forms.MenuItem menuItemFile;
		private System.Windows.Forms.MenuItem menuItemExit;
		private System.Windows.Forms.MenuItem mnuOpenTLG;
		private System.Windows.Forms.MenuItem menuItemDatabase;
		private System.Windows.Forms.MenuItem mnuOpenRWD;
		private System.Windows.Forms.MenuItem mnuOpenASCII;
		private System.Windows.Forms.MenuItem mnuSavePlot;
		private System.Windows.Forms.MenuItem mnuSaveDataSet;
        private System.Windows.Forms.MenuItem mnuConnect;
        private IContainer components;

		private System.Windows.Forms.MenuItem mnuQuickQuery;
		private System.Windows.Forms.MenuItem menuItemScript;
		private NumericalTools.AnalysisControl analysisControl1;
        private MenuItem menuItemView;
        private MenuItem menuViewAsTracks;		
		private System.Windows.Forms.MenuItem mnuOpenTSR;

		private static double [] allocdouble(double [] arraytopreserve, int sizetoadd)
		{
			if (arraytopreserve == null) return new double[sizetoadd];
			int i, l = arraytopreserve.Length;
			double [] newarray = new double[l + sizetoadd];
			for (i = 0; i < l; i++)
				newarray[i] = arraytopreserve[i];
			return newarray;
		}

		public ExeForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.SkipReadingGrains = true;
			mnuQuickQuery.Enabled = false;
			NumericalTools.Scripting.Script.ResetEngine();
			ScriptEd = new QuickDataCheck.ScriptEditor();
			dAddOutput = new AddOutput(ScriptEd.AddOutput);
			Script.AddFunctionDescriptor(new FunctionDescriptor("dbconnect", "Opens a connection to a DB server.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "DBServer"), new ParameterDescriptor(ParameterType.String, "Username"), new ParameterDescriptor(ParameterType.String, "Password") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnDBCONNECT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("dbdataset", "Fills a new dataset with the result of a DB query.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "DataSetName"), new ParameterDescriptor(ParameterType.String, "SQLText") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnDBDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("tlgdataset", "Fills a new dataset with data from one or more TLG file(s).", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "FileName") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnTLGDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("tsrdataset", "Fills a new dataset with data from one or more TSR file(s).", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "FileName") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnTSRDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("rwddataset", "Fills a new dataset with data from one or more RWD file(s).", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "FileName") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnRWDDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("asciidataset", "Fills a new dataset with data from an ASCII file.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "FileName") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnASCIIDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("print", "Prints a message.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Message") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnPRINT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setdataset", "Sets the current dataset. Both the dataset name or its zero-based index can be passed.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Name_or_Number") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("removedataset", "Remove a dataset. Both the dataset name or its zero-based index can be passed.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Name_or_Number") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnREMOVEDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("getdataset", "Gets the current dataset name.", new ParameterDescriptor [] {}, ParameterType.String, new NumericalTools.Scripting.Function(fnGETDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("getvariables", "Retrieves the number of variables available in the current dataset.", new ParameterDescriptor [] {}, ParameterType.Int32, new NumericalTools.Scripting.Function(fnGETVARIABLES)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("getvariable", "Retrieves the name of a variable in the current dataset.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Int32, "Variable_Number") }, ParameterType.String, new NumericalTools.Scripting.Function(fnGETVARIABLE)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("removevariable", "Removes a variable from the current dataset.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Variable_Name") }, ParameterType.String, new NumericalTools.Scripting.Function(fnREMOVEVARIABLE)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setxvar", "Selects the X variable for plots.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Variable_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETXVAR)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setxaxis", "Configures the X axis for plots.\r\nSets the variable and the axis scale and binning.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Variable_Name"), new ParameterDescriptor(ParameterType.Double, "Min"), new ParameterDescriptor(ParameterType.Double, "Max"), new ParameterDescriptor(ParameterType.Double, "BinSize") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETXAXIS)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setyvar", "Selects the Y variable for plots.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Variable_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETYVAR)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setyaxis", "Configures the Y axis for plots.\r\nSets the variable and the axis scale and binning.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Variable_Name"), new ParameterDescriptor(ParameterType.Double, "Min"), new ParameterDescriptor(ParameterType.Double, "Max"), new ParameterDescriptor(ParameterType.Double, "BinSize") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETYAXIS)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setzvar", "Selects the Z variable for plots.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Variable_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETZVAR)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("plot", "Draws a plot and an optional fit.\r\nThe second parameter (Fit_Type) is optional; if it is not specified, no fit is performed.\r\nValid plot options are:\r\nhisto -> histogram\r\nglent -> grey level entries\r\nhueent -> hue entries\r\nglquant -> grey level quantities\r\nhuequant -> hue quantities\r\ngscatter -> group scatter\r\nlego -> 3D LEGO\r\nscatter -> 2D scatter plot\r\nscatter3d -> 3D scatter plot\r\nValid fit types are:\r\ngauss -> adds a Gaussian curve fit\r\nigauss -> adds an inverse Gaussian curve\r\nn -> a polynomial fit of order n.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Plot_Type"), new ParameterDescriptor(ParameterType.String, "Fit_Type") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnPLOT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("exportplot", "Exports the current plot to a file.\r\nThe format depends on the file extension. Supported formats are:\r\nGIF, JPG/JPEG, BMP, PNG, WMF, EMF.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "File_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnEXPORTPLOT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("exportdata", "Exports the current dataset to an ASCII file.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "File_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnEXPORTDATA)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("overlay", "Overlays a function to the current display window.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Expression"), new ParameterDescriptor(ParameterType.String, "Function_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnOVERLAY)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("cutdataset", "Cuts a dataset using the specified criterion.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Expression"), new ParameterDescriptor(ParameterType.String, "Cut_Function") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnCUTDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("newdataset", "Creates a new dataset from scratch.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Expression"), new ParameterDescriptor(ParameterType.String, "DataSet_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnNEWDATASET)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("newvariable", "Creates a new variable in the current dataset.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Expression"), new ParameterDescriptor(ParameterType.String, "Variable_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnNEWVARIABLE)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setplotcolor", "Sets the plot color.\r\n3 parameters are expected as R, G, B values ranging from 0 to 1.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "R"), new ParameterDescriptor(ParameterType.Double, "G"), new ParameterDescriptor(ParameterType.Double, "B") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPLOTCOLOR)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setlabelfont", "Sets the label font face, size and style.\r\nStyle string can be null or a combination of:\r\nB -> Bold\r\nI -> Italic\r\nS -> Strikeout\r\nU -> Underline",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Font_Face"), new ParameterDescriptor(ParameterType.Int32, "Font_Size"), new ParameterDescriptor(ParameterType.String, "Font_Style") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETLABELFONT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setpanelfont", "Sets the panel font face, size and style.\r\nStyle string can be null or a combination of:\r\nB -> Bold\r\nI -> Italic\r\nS -> Strikeout\r\nU -> Underline",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Font_Face"), new ParameterDescriptor(ParameterType.Int32, "Font_Size"), new ParameterDescriptor(ParameterType.String, "Font_Style") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPANELFONT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setpanel", "Sets the panel text.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Panel_Text") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPANEL)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setpanelformat", "Sets the panel numerical format.\r\nAvailable formats are:\r\nFn -> Fixed n digits\r\nGn -> n significant digits\r\nEn -> Exponential notation with n digits.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Panel_Format") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPANELFORMAT)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setpanelx", "Sets the X position of the panel.\r\n0 = left, 1 = right, non-integer values are allowed.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "Panel_X") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPANELX)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setpanely", "Sets the Y position of the panel.\r\n0 = left, 1 = right, non-integer values are allowed.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "Panel_Y") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPANELY)));
			Script.AddFunctionDescriptor(new FunctionDescriptor("setpalette", "Sets the color palette.\r\nValid names are:\r\nRGBCont -> RGB continuous.\r\nFlat16 -> Flat palette with 16 colors.\r\nGreyCont -> Grey continuous.\r\nGrey16 -> 16 grey levels.",  new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "Palette_Name") }, ParameterType.Void, new NumericalTools.Scripting.Function(fnSETPALETTE)));
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ExeForm));
            this.mainMenu1 = new System.Windows.Forms.MainMenu(this.components);
            this.menuItemFile = new System.Windows.Forms.MenuItem();
            this.mnuOpenTSR = new System.Windows.Forms.MenuItem();
            this.mnuOpenTLG = new System.Windows.Forms.MenuItem();
            this.mnuOpenRWD = new System.Windows.Forms.MenuItem();
            this.mnuOpenASCII = new System.Windows.Forms.MenuItem();
            this.mnuSavePlot = new System.Windows.Forms.MenuItem();
            this.mnuSaveDataSet = new System.Windows.Forms.MenuItem();
            this.menuItemExit = new System.Windows.Forms.MenuItem();
            this.menuItemDatabase = new System.Windows.Forms.MenuItem();
            this.mnuConnect = new System.Windows.Forms.MenuItem();
            this.mnuQuickQuery = new System.Windows.Forms.MenuItem();
            this.menuItemScript = new System.Windows.Forms.MenuItem();
            this.menuItemView = new System.Windows.Forms.MenuItem();
            this.menuViewAsTracks = new System.Windows.Forms.MenuItem();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.analysisControl1 = new NumericalTools.AnalysisControl();
            this.SuspendLayout();
            // 
            // mainMenu1
            // 
            this.mainMenu1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
            this.menuItemFile,
            this.menuItemDatabase,
            this.menuItemScript,
            this.menuItemView});
            // 
            // menuItemFile
            // 
            this.menuItemFile.Index = 0;
            this.menuItemFile.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
            this.mnuOpenTSR,
            this.mnuOpenTLG,
            this.mnuOpenRWD,
            this.mnuOpenASCII,
            this.mnuSavePlot,
            this.mnuSaveDataSet,
            this.menuItemExit});
            this.menuItemFile.Text = "File";
            // 
            // mnuOpenTSR
            // 
            this.mnuOpenTSR.Index = 0;
            this.mnuOpenTSR.Text = "Open TSR(s)";
            this.mnuOpenTSR.Click += new System.EventHandler(this.mnuOpenTSR_Click);
            // 
            // mnuOpenTLG
            // 
            this.mnuOpenTLG.Index = 1;
            this.mnuOpenTLG.Text = "Open TLG(s)";
            this.mnuOpenTLG.Click += new System.EventHandler(this.mnuOpenTLG_Click);
            // 
            // mnuOpenRWD
            // 
            this.mnuOpenRWD.Index = 2;
            this.mnuOpenRWD.Text = "Open RWD(s)";
            this.mnuOpenRWD.Click += new System.EventHandler(this.mnuOpenRWD_Click);
            // 
            // mnuOpenASCII
            // 
            this.mnuOpenASCII.Index = 3;
            this.mnuOpenASCII.Text = "Open  ASCII";
            this.mnuOpenASCII.Click += new System.EventHandler(this.mnuOpenASCII_Click);
            // 
            // mnuSavePlot
            // 
            this.mnuSavePlot.Index = 4;
            this.mnuSavePlot.Text = "Save Plot";
            this.mnuSavePlot.Click += new System.EventHandler(this.mnuSavePlot_Click);
            // 
            // mnuSaveDataSet
            // 
            this.mnuSaveDataSet.Index = 5;
            this.mnuSaveDataSet.Text = "Save DataSet";
            this.mnuSaveDataSet.Click += new System.EventHandler(this.mnuSaveDataSet_Click);
            // 
            // menuItemExit
            // 
            this.menuItemExit.Index = 6;
            this.menuItemExit.Text = "Exit";
            this.menuItemExit.Click += new System.EventHandler(this.mnuExit_Click);
            // 
            // menuItemDatabase
            // 
            this.menuItemDatabase.Index = 1;
            this.menuItemDatabase.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
            this.mnuConnect,
            this.mnuQuickQuery});
            this.menuItemDatabase.Text = "Database";
            // 
            // mnuConnect
            // 
            this.mnuConnect.Index = 0;
            this.mnuConnect.Text = "Connect";
            this.mnuConnect.Click += new System.EventHandler(this.mnuConnect_Click);
            // 
            // mnuQuickQuery
            // 
            this.mnuQuickQuery.Index = 1;
            this.mnuQuickQuery.Text = "Quick Query";
            this.mnuQuickQuery.Click += new System.EventHandler(this.mnuQuickQuery_Click);
            // 
            // menuItemScript
            // 
            this.menuItemScript.Index = 2;
            this.menuItemScript.Text = "Script";
            this.menuItemScript.Click += new System.EventHandler(this.mnuScript_Click);
            // 
            // menuItemView
            // 
            this.menuItemView.Index = 3;
            this.menuItemView.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
            this.menuViewAsTracks});
            this.menuItemView.Text = "View";
            // 
            // menuViewAsTracks
            // 
            this.menuViewAsTracks.Index = 0;
            this.menuViewAsTracks.Text = "as Tracks (X3LView required)";
            this.menuViewAsTracks.Click += new System.EventHandler(this.menuViewAsTracks_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.Multiselect = true;
            // 
            // analysisControl1
            // 
            this.analysisControl1.CurrentDataSet = -1;
            this.analysisControl1.LabelFont = new System.Drawing.Font("Comic Sans MS", 9F, System.Drawing.FontStyle.Bold);
            this.analysisControl1.Location = new System.Drawing.Point(0, 0);
            this.analysisControl1.Name = "analysisControl1";
            this.analysisControl1.Palette = NumericalTools.Plot.PaletteType.RGBContinuous;
            this.analysisControl1.Panel = null;
            this.analysisControl1.PanelFont = new System.Drawing.Font("Comic Sans MS", 9F, System.Drawing.FontStyle.Bold);
            this.analysisControl1.PanelFormat = null;
            this.analysisControl1.PanelX = 1;
            this.analysisControl1.PanelY = 0;
            this.analysisControl1.PlotColor = System.Drawing.Color.Red;
            this.analysisControl1.Size = new System.Drawing.Size(962, 672);
            this.analysisControl1.TabIndex = 0;
            // 
            // ExeForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(965, 672);
            this.Controls.Add(this.analysisControl1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.Menu = this.mainMenu1;
            this.Name = "ExeForm";
            this.Text = "Quick Data Check";
            this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
            Application.EnableVisualStyles();
			Application.Run(new ExeForm());
		}

		private void x_OpenTLG(string [] fnames)
		{
			while (analysisControl1.DataSetNumber>0) analysisControl1.RemoveDSet(0);

			int i;

			analysisControl1.AddDataSet("Linked");

			analysisControl1.AddVariable(new double[0], "SX", "");
			analysisControl1.AddVariable(new double[0], "SY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "A", "");
			analysisControl1.AddVariable(new double[0], "Sigma", "");
			analysisControl1.AddVariable(new double[0], "Slope", "");

			analysisControl1.AddVariable(new double[0], "TSX", "");
			analysisControl1.AddVariable(new double[0], "TSY", "");
			analysisControl1.AddVariable(new double[0], "TPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TPY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TTZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TBZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TN", "");
			analysisControl1.AddVariable(new double[0], "TA", "");
			analysisControl1.AddVariable(new double[0], "TSigma", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TSlope", "");
			analysisControl1.AddVariable(new double[0], "TF", "");
			analysisControl1.AddVariable(new double[0], "TV", "");

			analysisControl1.AddVariable(new double[0], "BSX", "");
			analysisControl1.AddVariable(new double[0], "BSY", "");
			analysisControl1.AddVariable(new double[0], "BPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BPY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BTZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BBZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BN", "");
			analysisControl1.AddVariable(new double[0], "BA", "");
			analysisControl1.AddVariable(new double[0], "BSigma", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BSlope", "");
			analysisControl1.AddVariable(new double[0], "BF", "");
			analysisControl1.AddVariable(new double[0], "BV", "");

			analysisControl1.AddDataSet("TopTracks");

			analysisControl1.AddVariable(new double[0], "ID", "");
			analysisControl1.AddVariable(new double[0], "SX", "");
			analysisControl1.AddVariable(new double[0], "SY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "A", "");
			analysisControl1.AddVariable(new double[0], "Sigma", "");
			analysisControl1.AddVariable(new double[0], "Slope", "");
			analysisControl1.AddVariable(new double[0], "ViewID", "");

			analysisControl1.AddDataSet("BottomTracks");

			analysisControl1.AddVariable(new double[0], "ID", "");
			analysisControl1.AddVariable(new double[0], "SX", "");
			analysisControl1.AddVariable(new double[0], "SY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "A", "");
			analysisControl1.AddVariable(new double[0], "Sigma", "");
			analysisControl1.AddVariable(new double[0], "Slope", "");
			analysisControl1.AddVariable(new double[0], "ViewID", "");

			analysisControl1.AddDataSet("Views");

			analysisControl1.AddVariable(new double[0], "ID", "");
			analysisControl1.AddVariable(new double[0], "Side", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "N", "");

            analysisControl1.AddDataSet("SlopeCorrections");

            analysisControl1.AddVariable(new double[0], "TDSX", "");
            analysisControl1.AddVariable(new double[0], "TDSY", "");
            analysisControl1.AddVariable(new double[0], "TMSX", "");
            analysisControl1.AddVariable(new double[0], "TMSY", "");
            analysisControl1.AddVariable(new double[0], "BDSX", "");
            analysisControl1.AddVariable(new double[0], "BDSY", "");
            analysisControl1.AddVariable(new double[0], "BMSX", "");
            analysisControl1.AddVariable(new double[0], "BMSY", "");

            analysisControl1.AddDataSet("TopTksIndex");

            analysisControl1.AddVariable(new double[0], "IDZONE", "");
            analysisControl1.AddVariable(new double[0], "Side", "");
            analysisControl1.AddVariable(new double[0], "IDTRACK", "");

            analysisControl1.AddDataSet("BottomTksIndex");

            analysisControl1.AddVariable(new double[0], "IDZONE", "");
            analysisControl1.AddVariable(new double[0], "Side", "");
            analysisControl1.AddVariable(new double[0], "IDTRACK", "");

            analysisControl1.AddDataSet("AlignmentIgnore");

            analysisControl1.AddVariable(new double[0], "ID", "");

            analysisControl1.AddDataSet("BaseTrackIndex");

            analysisControl1.AddVariable(new double[0], "ID", "");

            double[] tmpdat = new double[33];
			double[] tmptks = new double[13];
			double[] tmpvw = new double[7];
            double[] tmpix = new double[3];
            double[] tmpds = new double[8];
            double[] tmpiid = new double[1];

			int total = 0;
			foreach (string fname in fnames)
			{
				System.IO.FileStream file = new System.IO.FileStream(fname,System.IO.FileMode.Open,System.IO.FileAccess.Read,System.IO.FileShare.Read);
				//SySal.Scanning.Plate.IO.OPERA.LinkedZone sysFile = new SySal.Scanning.Plate.IO.OPERA.LinkedZone(file);
                SySal.Scanning.Plate.IO.OPERA.LinkedZone sysFile = SySal.DataStreams.OPERALinkedZone.FromStream(file);
                SySal.OperaDb.Scanning.DBMIPMicroTrackIndex dbmi = null;
                try
                {
                    dbmi = new SySal.OperaDb.Scanning.DBMIPMicroTrackIndex(file);
                }
                catch (Exception) { dbmi = null;  }
                SySal.Scanning.PostProcessing.SlopeCorrections sc = null;
                try
                {
                    sc = new SySal.Scanning.PostProcessing.SlopeCorrections(file);
                }
                catch (Exception) { sc = null; }
                SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex bi = null;
                try
                {
                    bi = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex(file);
                }
                catch (Exception) { bi = null; }
                SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment ai = null;
                try
                {
                    ai = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment(file);
                }
                catch (Exception) { ai = null; }
                if (!(sysFile is SySal.DataStreams.OPERALinkedZone)) file.Close();

				int n = sysFile.Length;
				analysisControl1.SelectDataSet("Linked");
				for (i = 0; i < n; i++)
				{
					SySal.Tracking.MIPEmulsionTrackInfo info = IntMIPBaseTrack.GetInfo(sysFile[i]);

					tmpdat[0] = info.Slope.X;
					tmpdat[1] = info.Slope.Y;
					tmpdat[2] = info.Intercept.X;
					tmpdat[3] = info.Intercept.Y;
					tmpdat[4] = info.Intercept.Z;
					tmpdat[5] = info.Count;
					tmpdat[6] = info.AreaSum;
					tmpdat[7] = info.Sigma;
					tmpdat[8] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);

					info = IntMIPIndexedEmulsionTrack.GetInfo(sysFile[i].Top);
					tmpdat[9] = info.Slope.X;
					tmpdat[10] = info.Slope.Y;
					tmpdat[11] = info.Intercept.X;
					tmpdat[12] = info.Intercept.Y;
					tmpdat[13] = info.TopZ;
					tmpdat[14] = info.BottomZ;
					tmpdat[15] = info.Count;
					tmpdat[16] = info.AreaSum;
					tmpdat[17] = info.Sigma;
					tmpdat[18] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
					tmpdat[19] = (double)((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)sysFile[i].Top).OriginalRawData.Fragment;
					tmpdat[20] = (double)((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)sysFile[i].Top).OriginalRawData.View;

					info = IntMIPIndexedEmulsionTrack.GetInfo(sysFile[i].Bottom);
					tmpdat[21] = info.Slope.X;
					tmpdat[22] = info.Slope.Y;
					tmpdat[23] = info.Intercept.X;
					tmpdat[24] = info.Intercept.Y;
					tmpdat[25] = info.TopZ;
					tmpdat[26] = info.BottomZ;
					tmpdat[27] = info.Count;
					tmpdat[28] = info.AreaSum;
					tmpdat[29] = info.Sigma;
					tmpdat[30] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
					tmpdat[31] = (double)((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)sysFile[i].Bottom).OriginalRawData.Fragment;
					tmpdat[32] = (double)((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)sysFile[i].Bottom).OriginalRawData.View;

					analysisControl1.AddRow(tmpdat);
				}

				n = sysFile.Top.Length;
				analysisControl1.SelectDataSet("TopTracks");
				for (i = 0; i < n; i++)
				{
					SySal.Tracking.MIPEmulsionTrackInfo info = IntMIPIndexedEmulsionTrack.GetInfo(sysFile.Top[i]);
					tmptks[0] = i;
					tmptks[1] = info.Slope.X;
					tmptks[2] = info.Slope.Y;
					tmptks[3] = info.Intercept.X;
					tmptks[4] = info.Intercept.Y;
					tmptks[5] = info.Intercept.Z;
					tmptks[6] = info.TopZ;
					tmptks[7] = info.BottomZ;
					tmptks[8] = info.Count;
					tmptks[9] = info.AreaSum;
					tmptks[10] = info.Sigma;
					tmptks[11] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
					tmptks[12] = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)sysFile.Top[i]).View.Id;
					analysisControl1.AddRow(tmptks);
				}

				n = sysFile.Bottom.Length;
				analysisControl1.SelectDataSet("BottomTracks");
				for (i = 0; i < n; i++)
				{
					SySal.Tracking.MIPEmulsionTrackInfo info = IntMIPIndexedEmulsionTrack.GetInfo(sysFile.Bottom[i]);
					tmptks[0] = i;
					tmptks[1] = info.Slope.X;
					tmptks[2] = info.Slope.Y;
					tmptks[3] = info.Intercept.X;
					tmptks[4] = info.Intercept.Y;
					tmptks[5] = info.Intercept.Z;
					tmptks[6] = info.TopZ;
					tmptks[7] = info.BottomZ;
					tmptks[8] = info.Count;
					tmptks[9] = info.AreaSum;
					tmptks[10] = info.Sigma;
					tmptks[11] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
					tmptks[12] = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)sysFile.Bottom[i]).View.Id;
					analysisControl1.AddRow(tmptks);
				}

				n = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)sysFile.Top).ViewCount;
				analysisControl1.SelectDataSet("Views");
				for (i = 0; i < n; i++)
				{
					SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)sysFile.Top).View(i);
					tmpvw[0] = vw.Id;
					tmpvw[1] = 1;
					tmpvw[2] = vw.Position.X;
					tmpvw[3] = vw.Position.Y;
					tmpvw[4] = vw.TopZ;
					tmpvw[5] = vw.BottomZ;
					tmpvw[6] = vw.Length;
					analysisControl1.AddRow(tmpvw);
				}

				n = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)sysFile.Bottom).ViewCount;
				analysisControl1.SelectDataSet("Views");
				for (i = 0; i < n; i++)
				{
					SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)sysFile.Bottom).View(i);
					tmpvw[0] = vw.Id;
					tmpvw[1] = 2;
					tmpvw[2] = vw.Position.X;
					tmpvw[3] = vw.Position.Y;
					tmpvw[4] = vw.TopZ;
					tmpvw[5] = vw.BottomZ;
					tmpvw[6] = vw.Length;
					analysisControl1.AddRow(tmpvw);
				}

                if (dbmi != null)
                {
                    n = dbmi.TopTracksIndex.Length;
                    analysisControl1.SelectDataSet("TopTksIndex");
                    for (i = 0; i < n; i++)
                    {
                        tmpix[0] = dbmi.TopTracksIndex[i].ZoneId;
                        tmpix[1] = dbmi.TopTracksIndex[i].Side;
                        tmpix[2] = dbmi.TopTracksIndex[i].Id;
                        analysisControl1.AddRow(tmpix);
                    }
                }

                if (dbmi != null)
                {
                    n = dbmi.BottomTracksIndex.Length;
                    analysisControl1.SelectDataSet("BottomTksIndex");
                    for (i = 0; i < n; i++)
                    {
                        tmpix[0] = dbmi.BottomTracksIndex[i].ZoneId;
                        tmpix[1] = dbmi.BottomTracksIndex[i].Side;
                        tmpix[2] = dbmi.BottomTracksIndex[i].Id;
                        analysisControl1.AddRow(tmpix);
                    }
                }

                if (sc != null)
                {
                    analysisControl1.SelectDataSet("SlopeCorrections");
                    tmpds[0] = sc.TopDeltaSlope.X;
                    tmpds[1] = sc.TopDeltaSlope.Y;
                    tmpds[2] = sc.TopSlopeMultipliers.X;
                    tmpds[3] = sc.TopSlopeMultipliers.Y;
                    tmpds[4] = sc.BottomDeltaSlope.X;
                    tmpds[5] = sc.BottomDeltaSlope.Y;
                    tmpds[6] = sc.BottomSlopeMultipliers.X;
                    tmpds[7] = sc.BottomSlopeMultipliers.Y;
                    analysisControl1.AddRow(tmpds);
                }

                if (bi != null)
                {
                    n = bi.Ids.Length;
                    analysisControl1.SelectDataSet("BaseTrackIndex");
                    for (i = 0; i < n; i++)
                    {
                        tmpiid[0] = bi.Ids[i];
                        analysisControl1.AddRow(tmpiid);
                    }
                }

                if (ai != null)
                {
                    n = ai.Ids.Length;
                    analysisControl1.SelectDataSet("AlignmentIgnore");
                    for (i = 0; i < n; i++)
                    {
                        tmpiid[0] = ai.Ids[i];
                        analysisControl1.AddRow(tmpiid);
                    }
                }

                sysFile = null;
				GC.Collect();
				total += n;
			}

			analysisControl1.SelectDataSet("Linked");
			for (i = 0; i < 33; i++)
				analysisControl1.AutoVariableStatistics(i, null);
			analysisControl1.SelectDataSet("TopTracks");
			for (i = 0; i < 13; i++)
				analysisControl1.AutoVariableStatistics(i, null);
			analysisControl1.SelectDataSet("BottomTracks");
			for (i = 0; i < 13; i++)
				analysisControl1.AutoVariableStatistics(i, null);
			analysisControl1.SelectDataSet("Views");
			for (i = 0; i < 7; i++)
				analysisControl1.AutoVariableStatistics(i, null);
		}

		private void mnuOpenTLG_Click(object sender, System.EventArgs e)
		{
			//File Tlg
			openFileDialog1.CheckFileExists=true;
			openFileDialog1.CheckPathExists=true;
			openFileDialog1.Multiselect = true;
			openFileDialog1.Filter = "Track Linked Grain (*.tlg)|*.tlg";
			System.Windows.Forms.DialogResult dr = openFileDialog1.ShowDialog();
			if(dr == System.Windows.Forms.DialogResult.Cancel) return;

			string [] fnames = openFileDialog1.FileNames;
			if (fnames.Length == 0) return;

			try
			{
				x_OpenTLG(fnames);
			}
			catch(Exception e1)
			{
				System.Windows.Forms.MessageBox.Show("Error Reading File!\n\r" + 
					e1.ToString(), 
					Application.ProductName);
			};		
		}

		private void x_OpenTSR(string [] fnames)
		{
			while (analysisControl1.DataSetNumber>0) analysisControl1.RemoveDSet(0);

			int i, j;

			double [] tmply = new double[13];
			double [] tmpbtk = new double[15];
			double [] tmpvtk = new double[23];
			double [] tmpvtx = new double[9];

			analysisControl1.AddDataSet("Layers");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "REFZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "MXX", "");
			analysisControl1.AddVariable(new double[0], "MXY", "");
			analysisControl1.AddVariable(new double[0], "MYX", "");
			analysisControl1.AddVariable(new double[0], "MYY", "");
			analysisControl1.AddVariable(new double[0], "DX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "MSX", "");
			analysisControl1.AddVariable(new double[0], "MSY", "");
			analysisControl1.AddVariable(new double[0], "DSX", "");
			analysisControl1.AddVariable(new double[0], "DSY", "");
			analysisControl1.AddDataSet("BaseTracks");
			analysisControl1.AddVariable(new double[0], "SX", "");
			analysisControl1.AddVariable(new double[0], "SY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "A", "");
			analysisControl1.AddVariable(new double[0], "Sigma", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Slope", "");
			analysisControl1.AddVariable(new double[0], "NT", "");
			analysisControl1.AddVariable(new double[0], "BaseId", "");
			analysisControl1.AddVariable(new double[0], "LayerId", "");
			analysisControl1.AddVariable(new double[0], "TrackId", "");
			analysisControl1.AddVariable(new double[0], "PosInLayerId", "");
			analysisControl1.AddVariable(new double[0], "PosInTrackId", "");
			analysisControl1.AddDataSet("VolumeTracks");
			analysisControl1.AddVariable(new double[0], "DSX", "");
			analysisControl1.AddVariable(new double[0], "DSY", "");
			analysisControl1.AddVariable(new double[0], "DPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DPY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DSlope", "");
			analysisControl1.AddVariable(new double[0], "DVID", "");
			analysisControl1.AddVariable(new double[0], "DVN", "");
			analysisControl1.AddVariable(new double[0], "DIP", "\xb5m");
			analysisControl1.AddVariable(new double[0], "USX", "");
			analysisControl1.AddVariable(new double[0], "USY", "");
			analysisControl1.AddVariable(new double[0], "UPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "UPY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "UZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "USlope", "");
			analysisControl1.AddVariable(new double[0], "UVID", "");
			analysisControl1.AddVariable(new double[0], "UVN", "");
			analysisControl1.AddVariable(new double[0], "UIP", "\xb5m");
			analysisControl1.AddVariable(new double[0], "N", "");
            analysisControl1.AddVariable(new double[0], "DLayer", "");
            analysisControl1.AddVariable(new double[0], "DSheetID", "");
            analysisControl1.AddVariable(new double[0], "ULayer", "");
            analysisControl1.AddVariable(new double[0], "USheetID", "");
            analysisControl1.AddDataSet("Vertices");			
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "ND", "");
			analysisControl1.AddVariable(new double[0], "NU", "");
			analysisControl1.AddVariable(new double[0], "AvgD", "\xb5m");
			analysisControl1.AddVariable(new double[0], "X", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Y", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Z", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "DY", "\xb5m");

			foreach (string fname in fnames)
			{
				System.IO.FileStream file = new System.IO.FileStream(fname,System.IO.FileMode.Open,System.IO.FileAccess.Read,System.IO.FileShare.Read);
				SySal.TotalScan.Volume sysFile = new SySal.TotalScan.Volume(file);
				file.Close();				
				int ilayer, itrack, ibasetrack, ivertex;	
				analysisControl1.CurrentDataSet = 0;
				for (ilayer = 0; ilayer < sysFile.Layers.Length; ilayer++)
				{
					SySal.TotalScan.Layer ly = sysFile.Layers[ilayer];
					tmply[0] = ly.Length;
					tmply[1] = ly.RefCenter.Z;
					SySal.TotalScan.AlignmentData al = sysFile.Layers[ilayer].AlignData;
					tmply[2] = al.AffineMatrixXX;
					tmply[3] = al.AffineMatrixXY;
					tmply[4] = al.AffineMatrixYX;
					tmply[5] = al.AffineMatrixYY;
					tmply[6] = al.TranslationX;
					tmply[7] = al.TranslationY;
					tmply[8] = al.TranslationZ;
					tmply[9] = al.DShrinkX;
					tmply[10] = al.DShrinkY;
					tmply[11] = al.SAlignDSlopeX;
					tmply[12] = al.SAlignDSlopeY;
					analysisControl1.AddRow(tmply);
				}

				analysisControl1.CurrentDataSet = 1;
				for (ilayer = 0; ilayer < sysFile.Layers.Length; ilayer++)
				{
					SySal.TotalScan.Layer ly = sysFile.Layers[ilayer];
					for (ibasetrack = 0; ibasetrack < ly.Length; ibasetrack++)
					{
						SySal.TotalScan.Segment seg = sysFile.Layers[ilayer][ibasetrack];
						SySal.Tracking.MIPEmulsionTrackInfo info = seg.Info;
						tmpbtk[0] = info.Slope.X; 
						tmpbtk[1] = info.Slope.Y; 
						tmpbtk[2] = info.Intercept.X; 
						tmpbtk[3] = info.Intercept.Y; 
						tmpbtk[4] = info.Intercept.Z; 
						tmpbtk[5] = (double)info.Count; 
						tmpbtk[6] = (double)info.AreaSum;
						tmpbtk[7] = info.Sigma; 
						tmpbtk[8] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y); 
						tmpbtk[9] = (seg.TrackOwner == null) ? 0 : seg.TrackOwner.Length; 
						tmpbtk[10] = -1;//seg.BaseTrackId; 
						tmpbtk[11] = seg.LayerOwner.Id;
						tmpbtk[12] = (seg.TrackOwner == null) ? -1 : seg.TrackOwner.Id;
						tmpbtk[13] = seg.PosInLayer;
						tmpbtk[14] = (seg.TrackOwner == null) ? -1 : seg.PosInTrack;
						analysisControl1.AddRow(tmpbtk);	
					}					
				}

				analysisControl1.CurrentDataSet = 2;
				for (itrack = 0; itrack < sysFile.Tracks.Length; itrack++)
				{
					SySal.TotalScan.Track t = sysFile.Tracks[itrack];
					
					tmpvtk[0] = t.Downstream_SlopeX;
					tmpvtk[1] = t.Downstream_SlopeY;
					tmpvtk[2] = t.Downstream_PosX;
					tmpvtk[3] = t.Downstream_PosY;
					tmpvtk[4] = t.Downstream_Z;
					tmpvtk[5] = Math.Sqrt(tmpvtk[0] * tmpvtk[0] + tmpvtk[1] * tmpvtk[1]);
					tmpvtk[6] = (t.Downstream_Vertex == null) ? -1 : t.Downstream_Vertex.Id;
					tmpvtk[7] = (t.Downstream_Vertex == null) ? 0 : t.Downstream_Vertex.Length;
					tmpvtk[8] = (t.Downstream_Vertex == null) ? -1 : t.Downstream_Impact_Parameter;					

					tmpvtk[9] = t.Upstream_SlopeX;
					tmpvtk[10] = t.Upstream_SlopeY;
					tmpvtk[11] = t.Upstream_PosX;
					tmpvtk[12] = t.Upstream_PosY;
					tmpvtk[13] = t.Upstream_Z;
					tmpvtk[14] = Math.Sqrt(tmpvtk[9] * tmpvtk[9] + tmpvtk[10] * tmpvtk[10]);
					tmpvtk[15] = (t.Upstream_Vertex == null) ? -1 : t.Upstream_Vertex.Id;
					tmpvtk[16] = (t.Upstream_Vertex == null) ? 0 : t.Upstream_Vertex.Length;
					tmpvtk[17] = (t.Upstream_Vertex == null) ? -1 : t.Upstream_Impact_Parameter;					

					tmpvtk[18] = t.Length;

                    tmpvtk[19] = t.DownstreamLayer.Id;
                    tmpvtk[20] = t.DownstreamLayer.SheetId;
                    tmpvtk[21] = t.UpstreamLayer.Id;
                    tmpvtk[22] = t.UpstreamLayer.SheetId;
					
					analysisControl1.AddRow(tmpvtk);
				}

				analysisControl1.CurrentDataSet = 3;
				for (ivertex = 0; ivertex < sysFile.Vertices.Length; ivertex++)
				{
					SySal.TotalScan.Vertex v = sysFile.Vertices[ivertex];
					tmpvtx[0] = v.Length;
					int dc, uc;
					for (i = dc = uc = 0; i < v.Length; i++)
						if (v[i].Upstream_Vertex == v) dc++;
						else uc++;
					tmpvtx[1] = (double)dc;
					tmpvtx[2] = (double)uc;
					tmpvtx[3] = v.AverageDistance;
					tmpvtx[4] = v.X;
					tmpvtx[5] = v.Y;
					tmpvtx[6] = v.Z;
					tmpvtx[7] = v.DX;
					tmpvtx[8] = v.DY;
					analysisControl1.AddRow(tmpvtx);
				}
				sysFile = null;
				GC.Collect();
			}
			for (i = 3; i >= 0; i--)
			{
				analysisControl1.CurrentDataSet = i;
				for (j = 0; j < analysisControl1.Variables; j++)
					analysisControl1.AutoVariableStatistics(j, null);
			}
		}

		private void mnuOpenTSR_Click(object sender, System.EventArgs e)
		{
			//File TSR
			int i, j;
			openFileDialog1.CheckFileExists=true;
			openFileDialog1.CheckPathExists=true;
			openFileDialog1.Multiselect = true;
			openFileDialog1.Filter = "TotalScan Reconstruction (*.tsr)|*.tsr";
			System.Windows.Forms.DialogResult dr = openFileDialog1.ShowDialog();
			if(dr == System.Windows.Forms.DialogResult.Cancel) return;

			string [] fnames = openFileDialog1.FileNames;
			if (fnames.Length == 0) return;

			try
			{
				x_OpenTSR(fnames);
			}
			catch(Exception e1)
			{
				System.Windows.Forms.MessageBox.Show("Error Reading File!\n\r" + 
					e1.ToString(), 
					Application.ProductName);
			};			
		}

		private void x_OpenRWD(string [] fnames)
		{
			int i, j;

			while (analysisControl1.DataSetNumber > 0) analysisControl1.RemoveDSet(0);
				
			analysisControl1.AddDataSet("TopTracks");
			analysisControl1.AddVariable(new double[0], "SX", "");
			analysisControl1.AddVariable(new double[0], "SY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "FPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "FPY", "\xb5m");
            analysisControl1.AddVariable(new double[0], "FPZ", "\xb5m");
            analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
            analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
            analysisControl1.AddVariable(new double[0], "Slope", "");
			analysisControl1.AddVariable(new double[0], "Sigma", "");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "A", "");
			analysisControl1.AddDataSet("BottomTracks");
			analysisControl1.AddVariable(new double[0], "SX", "");
			analysisControl1.AddVariable(new double[0], "SY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "FPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "FPY", "\xb5m");
            analysisControl1.AddVariable(new double[0], "FPZ", "\xb5m");
            analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
            analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
            analysisControl1.AddVariable(new double[0], "Slope", "");
			analysisControl1.AddVariable(new double[0], "Sigma", "");
			analysisControl1.AddVariable(new double[0], "N", "");
			analysisControl1.AddVariable(new double[0], "A", "");
			analysisControl1.AddDataSet("TopViews");
			analysisControl1.AddVariable(new double[0], "TX", "");
			analysisControl1.AddVariable(new double[0], "TY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "MPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "MPY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "IXX", "");
			analysisControl1.AddVariable(new double[0], "IXY", "");
			analysisControl1.AddVariable(new double[0], "IYX", "");
			analysisControl1.AddVariable(new double[0], "IYY", "");
			analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Base", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Tracks", "");
			analysisControl1.AddDataSet("BottomViews");
			analysisControl1.AddVariable(new double[0], "TX", "");
			analysisControl1.AddVariable(new double[0], "TY", "");
			analysisControl1.AddVariable(new double[0], "PX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "PY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "MPX", "\xb5m");
			analysisControl1.AddVariable(new double[0], "MPY", "\xb5m");
			analysisControl1.AddVariable(new double[0], "IXX", "");
			analysisControl1.AddVariable(new double[0], "IXY", "");
			analysisControl1.AddVariable(new double[0], "IYX", "");
			analysisControl1.AddVariable(new double[0], "IYY", "");
			analysisControl1.AddVariable(new double[0], "TZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "BZ", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Base", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Tracks", "");
			analysisControl1.AddDataSet("TopLayers");
			analysisControl1.AddVariable(new double[0], "Layer", "");
			analysisControl1.AddVariable(new double[0], "Z", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Grains", "");
			analysisControl1.AddDataSet("BottomLayers");
			analysisControl1.AddVariable(new double[0], "Layer", "");
			analysisControl1.AddVariable(new double[0], "Z", "\xb5m");
			analysisControl1.AddVariable(new double[0], "Grains", "");

			double [] tmptk = new double[13];
			double [] tmpvw = new double[14];
			double [] tmply = new Double[3];

			foreach (string fname in fnames)
			{
				System.IO.FileStream file = new System.IO.FileStream(fname,System.IO.FileMode.Open,System.IO.FileAccess.Read,System.IO.FileShare.Read);
				SySal.Scanning.Plate.IO.OPERA.RawData.Fragment sysFile = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment(file);
				file.Close();
				int nv = sysFile.Length;

				SySal.Tracking.MIPEmulsionTrackInfo info;

				analysisControl1.CurrentDataSet = 4;
				for (i = 0; i < nv; i++) 									
					for (j = 0; j < sysFile[i].Top.Layers.Length; j++)
					{
						tmply[0] = j;
						tmply[1] = sysFile[i].Top.Layers[j].Z;
						tmply[2] = sysFile[i].Top.Layers[j].Grains;
						analysisControl1.AddRow(tmply);
					}
				analysisControl1.CurrentDataSet = 5;
				for (i = 0; i < nv; i++) 									
					for (j = 0; j < sysFile[i].Bottom.Layers.Length; j++)
					{
						tmply[0] = j;
						tmply[1] = sysFile[i].Bottom.Layers[j].Z;
						tmply[2] = sysFile[i].Bottom.Layers[j].Grains;
						analysisControl1.AddRow(tmply);
					}

				analysisControl1.CurrentDataSet = 2;
				for (i = 0; i < nv; i++) 									
				{
					tmpvw[0] = sysFile[i].Tile.X;
					tmpvw[1] = sysFile[i].Tile.Y;
					tmpvw[2] = sysFile[i].Top.Pos.X;
					tmpvw[3] = sysFile[i].Top.Pos.Y;
					tmpvw[4] = sysFile[i].Top.MapPos.X;
					tmpvw[5] = sysFile[i].Top.MapPos.Y;
					tmpvw[6] = sysFile[i].Top.MXX;
					tmpvw[7] = sysFile[i].Top.MXY;
					tmpvw[8] = sysFile[i].Top.MYX;
					tmpvw[9] = sysFile[i].Top.MYY;
					tmpvw[10] = sysFile[i].Top.TopZ;
					tmpvw[11] = sysFile[i].Top.BottomZ;
					tmpvw[12] = sysFile[i].Top.BottomZ - sysFile[i].Bottom.TopZ;
					tmpvw[13] = sysFile[i].Top.Length;
					analysisControl1.AddRow(tmpvw);
				}
				analysisControl1.CurrentDataSet = 3;
				for (i = 0; i < nv; i++) 									
				{
					tmpvw[0] = sysFile[i].Tile.X;
					tmpvw[1] = sysFile[i].Tile.Y;
					tmpvw[2] = sysFile[i].Bottom.Pos.X;
					tmpvw[3] = sysFile[i].Bottom.Pos.Y;
					tmpvw[4] = sysFile[i].Bottom.MapPos.X;
					tmpvw[5] = sysFile[i].Bottom.MapPos.Y;
					tmpvw[6] = sysFile[i].Bottom.MXX;
					tmpvw[7] = sysFile[i].Bottom.MXY;
					tmpvw[8] = sysFile[i].Bottom.MYX;
					tmpvw[9] = sysFile[i].Bottom.MYY;
					tmpvw[10] = sysFile[i].Bottom.TopZ;
					tmpvw[11] = sysFile[i].Bottom.BottomZ;
					tmpvw[12] = sysFile[i].Top.BottomZ - sysFile[i].Bottom.TopZ;
					tmpvw[13] = sysFile[i].Bottom.Length;
					analysisControl1.AddRow(tmpvw);
				}

				analysisControl1.CurrentDataSet = 0;
				for (i = 0; i < nv; i++) 									
					for (j = 0; j < sysFile[i].Top.Length; j++)
					{
						info = IntMIPIndexedEmulsionTrack.GetInfo(sysFile[i].Top[j]);
						SySal.BasicTypes.Vector v = sysFile[i].Top.MapPoint(info.Intercept);
						tmptk[0] = info.Slope.X; 
						tmptk[1] = info.Slope.Y; 
						tmptk[2] = v.X;  
						tmptk[3] = v.Y; 
						tmptk[4] = info.Intercept.X;  
						tmptk[5] = info.Intercept.Y;
                        tmptk[6] = info.Intercept.Z;
                        tmptk[7] = info.TopZ;
                        tmptk[8] = info.BottomZ;
                        tmptk[9] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y); 
						tmptk[10] = info.Sigma; 						
						tmptk[11] = (double)info.Count; 
						tmptk[12] = (double)info.AreaSum; 
						analysisControl1.AddRow(tmptk);						
					}
				analysisControl1.CurrentDataSet = 1;
				for (i = 0; i < nv; i++) 									
					for (j = 0; j < sysFile[i].Bottom.Length; j++)
					{
						info = IntMIPIndexedEmulsionTrack.GetInfo(sysFile[i].Bottom[j]);
						SySal.BasicTypes.Vector v = sysFile[i].Bottom.MapPoint(info.Intercept);
						tmptk[0] = info.Slope.X; 
						tmptk[1] = info.Slope.Y; 
						tmptk[2] = v.X;  
						tmptk[3] = v.Y; 
						tmptk[4] = info.Intercept.X;  
						tmptk[5] = info.Intercept.Y;
                        tmptk[6] = info.Intercept.Z;
                        tmptk[7] = info.TopZ;
                        tmptk[8] = info.BottomZ;
                        tmptk[9] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
                        tmptk[10] = info.Sigma;
                        tmptk[11] = (double)info.Count;
                        tmptk[12] = (double)info.AreaSum;
                        analysisControl1.AddRow(tmptk);						
					}
				sysFile = null;
				GC.Collect();
			}
			for (i = 5; i >= 0; i--)
			{
				analysisControl1.CurrentDataSet = i;
				for (j = 0; j < analysisControl1.Variables; j++)
					analysisControl1.AutoVariableStatistics(j, null);
			}
		}

		private void mnuOpenRWD_Click(object sender, System.EventArgs e)
		{
			//File RWD
			openFileDialog1.CheckFileExists=true;
			openFileDialog1.CheckPathExists=true;
			openFileDialog1.Multiselect = true;
			openFileDialog1.Filter = "RaW Data (*.rwd.*)|*.rwd.*";
			System.Windows.Forms.DialogResult dr = openFileDialog1.ShowDialog();
			if(dr == System.Windows.Forms.DialogResult.Cancel) return;

			string [] fnames = openFileDialog1.FileNames;
			if (fnames.Length == 0) return;

			try
			{
				x_OpenRWD(fnames);
			}
			catch(Exception e2)
			{
				System.Windows.Forms.MessageBox.Show("Error Reading File!\n\r" + 
					e2.ToString(), 
					Application.ProductName);
			};

		}

		private void x_SavePlot(string fname)
		{
			if(fname=="") return;
            try
            {

                Bitmap b = analysisControl1.BitmapPlot;
                string lfname = fname.ToLower();
                if (lfname.EndsWith(".gif"))
                    b.Save(fname, System.Drawing.Imaging.ImageFormat.Gif);
                else if (lfname.EndsWith(".jpg") || lfname.EndsWith(".jpeg"))
                    b.Save(fname, System.Drawing.Imaging.ImageFormat.Jpeg);
                else if (lfname.EndsWith(".bmp"))
                    b.Save(fname, System.Drawing.Imaging.ImageFormat.Bmp);
                else if (lfname.EndsWith(".png"))
                    b.Save(fname, System.Drawing.Imaging.ImageFormat.Png);
                /*			
                    else if (lfname.EndsWith(".wmf"))
                                b.Save(fname, System.Drawing.Imaging.ImageFormat.Wmf);
                 */
                else if (lfname.EndsWith(".emf"))
                    //b.Save(fname, System.Drawing.Imaging.ImageFormat.Emf);
                    analysisControl1.SaveMetafile(lfname);

            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error saving file", MessageBoxButtons.OK);
            }
		}

		private void mnuSavePlot_Click(object sender, System.EventArgs e)
		{

			saveFileDialog1.CheckPathExists=true;
			saveFileDialog1.Filter= "Graphics Interchange Format (*.gif)|*.gif|" +
				"Joint Photographic Experts Group (*.jpeg)|*.jpeg|" +
				"Bitmap (*.bmp)|*.bmp|" +
				"Portable Network Graphics (*.png)|*.png|" +
				//"Windows Metafile (*.wmf)|*.wmf|" + 
				"Enhanced Metafile (*.emf)|*.emf" ;

			//saveFileDialog1.FilterIndex=1;
			if (saveFileDialog1.ShowDialog() != DialogResult.OK) return;

			x_SavePlot(saveFileDialog1.FileName);
		}
/*
		private void menuItem6_Click(object sender, System.EventArgs e)
		{
			menuItem6.Checked = true;
			menuItem7.Checked = false;
		}

		private void menuItem7_Click(object sender, System.EventArgs e)
		{
			menuItem6.Checked = false;
			menuItem7.Checked = true;
		}
*/
		private void mnuSaveDataSet_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sf = new SaveFileDialog();
			sf.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
			sf.FilterIndex = 0;
			sf.CheckPathExists = true;
			sf.OverwritePrompt = true;
			if (sf.ShowDialog() == DialogResult.OK)	analysisControl1.DumpCurrentDataSetIntoFile(sf.FileName);
		}

		private void menuItemExit_Click(object sender, System.EventArgs e)
		{
			Application.Exit();
		}

		private void x_OpenASCII(string fname)
		{
			//ASCII
			int n, k;
			int i, j;

			double[] tmpdat;
			string[] varnm;
			string[] varum;
			string line;
			if(fname=="") return;
			int nline=0;

			while (analysisControl1.DataSetNumber > 0) analysisControl1.RemoveDSet(0);

			string [] tokens;
			System.IO.StreamReader r;
			r = new System.IO.StreamReader(fname);
			tokens = ManageSpaces(r.ReadLine().Trim()).Split(' ','\t');
			nline++;
			n = tokens.Length;
			tmpdat = new double[n];
			varnm = new string[n];
			varum = new string[n];
			for(i = 0; i < n; i++)
				varnm[i] = varum[i] = "";

			try
			{
				for(i = 0; i < n; i++)
				{
					tmpdat[i] = Convert.ToDouble(tokens[i]);
					varnm[i] = "var" + i;
					varum[i] = "";
				}

			}
			catch(Exception)
			{
				//Error thrown: first line contains header
				tmpdat = new double[0];
				for(i = 0; i < n; i++)
				{
					string[] tmparr = tokens[i].Split(new char[] {'('}, 2);
					varnm[i] = tmparr[0];
					if (tmparr.Length > 1) varum[i] = "(" + tmparr[1];
				}
			};
			analysisControl1.AddDataSet("General");
			for (i = 0; i < n; i++)
				analysisControl1.AddVariable(new double[0], varnm[i], varum[i]);
			if (tmpdat.Length > 0)
				analysisControl1.AddRow(tmpdat);
			else 
				tmpdat = new double[n];
		
			try
			{
				while ((line = r.ReadLine()) != null)
				{
					nline++;
					tokens = ManageSpaces(line.Trim()).Split(' ','\t');
					if (line.Length < 1) continue;
					if (n != tokens.Length) throw new Exception("Format error at line " + nline);
					for (i = 0; i < n; i++) tmpdat[i] = Convert.ToDouble(tokens[i]);
					analysisControl1.AddRow(tmpdat);
				}
				for (i = 0; i < n; i++)
                    analysisControl1.AutoVariableStatistics(i, null);
			}
			catch(Exception x)
			{
				System.Windows.Forms.MessageBox.Show("Error Reading File!\n\r" + 
					x.ToString(), 
					Application.ProductName);
			}
			r.Close();
			GC.Collect();
		}

		private void mnuOpenASCII_Click(object sender, System.EventArgs e)
		{
			openFileDialog1.CheckFileExists=true;
			openFileDialog1.CheckPathExists=true;
			openFileDialog1.Multiselect = false;
			openFileDialog1.Filter = "ASCII file (*.txt)|*.txt|ASCII file (*.dat)|*.dat";
			System.Windows.Forms.DialogResult dr = openFileDialog1.ShowDialog();
			if(dr == System.Windows.Forms.DialogResult.Cancel) return;
			x_OpenASCII(openFileDialog1.FileName);
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

		private void mnuExit_Click(object sender, System.EventArgs e)
		{
			Application.Exit();
		}

		private void mnuConnect_Click(object sender, System.EventArgs e)
		{

			try
			{
				QuickDataCheck.DBAccessForm dbfrm = new QuickDataCheck.DBAccessForm();
				dbfrm.feed(m_DBCred);
				dbfrm.ShowDialog();
				if (dbfrm.DialogResult == DialogResult.OK)
				{
					m_DBCred = dbfrm.newDBCred;
					if (m_DBConn != null) 
					{
						m_DBConn.Close();
						m_DBConn = null;
						mnuQuickQuery.Enabled = false;
					}

					m_DBConn = new SySal.OperaDb.OperaDbConnection(m_DBCred.DBServer, m_DBCred.DBUserName, m_DBCred.DBPassword);
					m_DBConn.Open();
					mnuQuickQuery.Enabled = true;
				}
			}
			catch(Exception exc)
			{
				System.Windows.Forms.MessageBox.Show("Connection Error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}

		}

		private void x_QuickQuery(string datasetname, string querytext)
		{
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter(querytext, m_DBConn, null).Fill(ds);
			analysisControl1.AddDataSet(datasetname);
			analysisControl1.CurrentDataSet = analysisControl1.DataSetNumber - 1;
			System.Data.DataColumnCollection dcc = ds.Tables[0].Columns;
			System.Data.DataRowCollection drc = ds.Tables[0].Rows;
			double [] a_d = new double[drc.Count];
			int c;
			for (c = 0; c < dcc.Count; c++)
			{
				System.Data.DataColumn dc = dcc[c];
				try
				{
					int i;
					for (i = 0; i < drc.Count; i++)
						a_d[i] = Convert.ToDouble(drc[i][c]);
					analysisControl1.AddVariable(a_d, dc.ColumnName, "");
				}
				catch (Exception) {}
			}
			GC.Collect();
		}

		private void mnuQuickQuery_Click(object sender, System.EventArgs e)
		{
			Cursor oldc = Cursor;
			try
			{
				QuickDataCheck.QuickQueryForm qqfrm = new QuickDataCheck.QuickQueryForm();
                qqfrm.dbConn = m_DBConn;
				qqfrm.ShowDialog();
				if (qqfrm.DialogResult == DialogResult.OK)
				{
					Cursor = Cursors.WaitCursor;
					x_QuickQuery(qqfrm.DataSetName, qqfrm.QueryText);
				}
			}
			catch(Exception exc)
			{
				System.Windows.Forms.MessageBox.Show("DB error: \r\n" + exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			Cursor = oldc;
		
		}

		private void mnuScript_Click(object sender, System.EventArgs e)
		{
			ScriptEd.Show();
		}

		#region scripting functions
		private AddOutput dAddOutput;

		private void fnCUTDATASET(ref object ret, object [] connpars)
		{
			if (connpars.Length != 2) throw new Exception("Dataset name, new name and selection string are required!!");
			this.analysisControl1.ApplyFunction((string)connpars[0], NumericalTools.AnalysisControl.FunctionUse.Cut, (string)connpars[1]);
			ret = null;
		}

		private void fnNEWDATASET(ref object ret, object [] connpars)
		{
			if (connpars.Length != 2) throw new Exception("Dataset name, new name and selection string are required!!");
			this.analysisControl1.ApplyFunction((string)connpars[0], NumericalTools.AnalysisControl.FunctionUse.CutNew, (string)connpars[1]);
			ret = null;
		}

		private void fnNEWVARIABLE(ref object ret, object [] connpars)
		{
			if (connpars.Length != 2) throw new Exception("Dataset name, new name and selection string are required!!");
			this.analysisControl1.ApplyFunction((string)connpars[0], NumericalTools.AnalysisControl.FunctionUse.AddVariable, (string)connpars[1]);
			ret = null;
		}

		private void fnOVERLAY(ref object ret, object [] connpars)
		{
			if (connpars.Length != 2) throw new Exception("Dataset name, new name and overlay expression are required!!");
			this.analysisControl1.ApplyFunction((string)connpars[0], NumericalTools.AnalysisControl.FunctionUse.AddVariable, (string)connpars[1]);
			ret = null;
		}

		private void fnDBCONNECT(ref object ret, object [] connpars)
		{
			if (connpars.Length != 3) throw new Exception("DB server, username and password are required as parameters!");
			if (DBConnOwner)
			{
				if (m_DBConn != null) 
				{
					m_DBConn.Close();
					m_DBConn = null;
				}
				m_DBConn = new SySal.OperaDb.OperaDbConnection(connpars[0].ToString(), connpars[1].ToString(), connpars[2].ToString());
				m_DBConn.Open();
				if (dAddOutput != null) dAddOutput("Connected to " + connpars[0].ToString());
				ret = null;
			}
		}

		private void fnDBDATASET(ref object ret, object [] connpars)
		{
			if (connpars.Length != 2) throw new Exception("Dataset name and query text are required as parameters!");
			x_QuickQuery(connpars[0].ToString(), connpars[1].ToString());
			ret = null;
		}

		private void fnTLGDATASET(ref object ret, object [] pars)
		{
			if (pars.Length < 1) throw new Exception("File names required as parameters!");
			int i;
			string [] fnames = new String[pars.Length];
			for (i = 0; i < pars.Length; i++)
				fnames[i] = pars[i].ToString();
			x_OpenTLG(fnames);
			ret = null;
		}

		private void fnTSRDATASET(ref object ret, object [] pars)
		{
			if (pars.Length < 1) throw new Exception("File names required as parameters!");
			int i;
			string [] fnames = new String[pars.Length];
			for (i = 0; i < pars.Length; i++)
				fnames[i] = pars[i].ToString();
			x_OpenTSR(fnames);
			ret = null;
		}

		private void fnRWDDATASET(ref object ret, object [] pars)
		{
			if (pars.Length < 1) throw new Exception("File names required as parameters!");
			int i;
			string [] fnames = new String[pars.Length];
			for (i = 0; i < pars.Length; i++)
				fnames[i] = pars[i].ToString();
			x_OpenRWD(fnames);
			ret = null;
		}

		private void fnASCIIDATASET(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("File name required as parameters!");
			x_OpenASCII(pars[0].ToString());
			ret = null;
		}

		private void fnPRINT(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("One parameter required!");
			if (dAddOutput != null) dAddOutput(pars[0].ToString());
			ret = null;
		}

		private void fnSETDATASET(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Dataset name or number required!");
			try
			{
				int d = (int)pars[0];
				analysisControl1.CurrentDataSet = d;
			}
			catch(Exception)
			{
				try
				{
					int d = (int)(double)pars[0];
					analysisControl1.CurrentDataSet = d;
				}
				catch(Exception)
				{
					analysisControl1.SelectDataSet(pars[0].ToString());
				}				
			}
			if (dAddOutput != null) dAddOutput("Current Dataset is \"" + analysisControl1.DataSetName(analysisControl1.CurrentDataSet) + "\"");
			ret = null;
		}

		private void fnGETDATASET(ref object ret, object [] pars)
		{
			if (pars.Length != 0) throw new Exception("No parameters allowed!");
			ret = (string)analysisControl1.DataSetName(analysisControl1.CurrentDataSet).Clone();
		}

		private void fnREMOVEDATASET(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Dataset name or number required!");
			try
			{
				int d = (int)pars[0];
				analysisControl1.RemoveDSet(d);
			}
			catch(Exception)
			{
				try
				{
					int d = (int)(double)pars[0];
					analysisControl1.RemoveDSet(d);
				}
				catch(Exception)
				{
					int d;
					for (d = 0; d < analysisControl1.DataSetNumber && String.Compare(analysisControl1.DataSetName(d), pars[0].ToString(), true) != 0; d++);
					analysisControl1.RemoveDSet(d);
				}				
			}
			if (dAddOutput != null) dAddOutput("Dataset \"" + analysisControl1.DataSetName(analysisControl1.CurrentDataSet) + "\" removed.");
			GC.Collect();
			ret = null;
		}

		private void fnGETVARIABLES(ref object ret, object [] pars)
		{
			if (pars.Length != 0) throw new Exception("No parameters allowed!");
			ret = (int)analysisControl1.Variables;
		}

		private void fnGETVARIABLE(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Variable number is required!");
			ret = analysisControl1.VariableName(Convert.ToInt32(pars[0]));
		}

		private void fnREMOVEVARIABLE(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Variable name is required!");
			ret = analysisControl1.RemoveVariable((string)pars[0].ToString());
		}

		private void fnSETXVAR(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Variable name is required!");
			analysisControl1.SetX(pars[0].ToString());
			ret = null;
		}

		private void fnSETXAXIS(ref object ret, object [] pars)
		{			
			if (pars.Length != 4) throw new Exception("Variable name, min, max and binning are required!");
			analysisControl1.SetX(pars[0].ToString(), (double)pars[1], (double)pars[2], (double)pars[3]);
			ret = null;
		}

		private void fnSETYVAR(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Variable name is required!");
			analysisControl1.SetY(pars[0].ToString());
			ret = null;
		}

		private void fnSETYAXIS(ref object ret, object [] pars)
		{			
			if (pars.Length != 4) throw new Exception("Variable name, min, max and binning are required!");
			analysisControl1.SetY(pars[0].ToString(), (double)pars[1], (double)pars[2], (double)pars[3]);
			ret = null;
		}

		private void fnSETZVAR(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("Variable name is required!");
			analysisControl1.SetZ(pars[0].ToString());
			ret = null;
		}

		private void fnPLOT(ref object ret, object [] pars)
		{
			if (pars.Length != 2) throw new Exception("Plot type and fit type are required!");
			analysisControl1.Plot(pars[0].ToString(), pars[1].ToString());
			ret = null;
		}

		private void fnEXPORTPLOT(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("File name is required!");
			x_SavePlot(pars[0].ToString());
			ret = null;
		}

		private void fnEXPORTDATA(ref object ret, object [] pars)
		{
			if (pars.Length != 1) throw new Exception("File name is required!");
			analysisControl1.DumpCurrentDataSetIntoFile(pars[0].ToString());
			ret = null;
		}

		private void fnSETPALETTE(ref object ret, object [] palettepars)
		{
			if (palettepars.Length != 1) throw new Exception("palette name is required as parameter!");
			string ps = palettepars[0].ToString();
			if (String.Compare(ps, "RGBCont", true) == 0) analysisControl1.Palette = Plot.PaletteType.RGBContinuous;
			else if (String.Compare(ps, "Flat16", true) == 0) analysisControl1.Palette = Plot.PaletteType.Flat16;
			else if (String.Compare(ps, "GreyCont", true) == 0) analysisControl1.Palette = Plot.PaletteType.GreyContinuous;
			else if (String.Compare(ps, "Grey16", true) == 0) analysisControl1.Palette = Plot.PaletteType.Grey16;
			else throw new Exception("Unknown palette type!");				
			ret = null;
		}

		private void fnSETPLOTCOLOR(ref object ret, object [] plotpars)
		{
			if (plotpars.Length != 3) throw new Exception("3 numbers (R, G, B) must be specified for the plot color!");
			Color c = Color.FromArgb(Convert.ToInt32(Convert.ToDouble(plotpars[0]) * 255), Convert.ToInt32(Convert.ToDouble(plotpars[1]) * 255), Convert.ToInt32(Convert.ToDouble(plotpars[2]) * 255));
			analysisControl1.PlotColor = c;
			ret = null;
		}

		private void fnSETLABELFONT(ref object ret, object [] fontpars)
		{
			if (fontpars.Length != 3) throw new Exception("Font face, size and style are required as parameters!");
			string fs = fontpars[2].ToString().ToUpper();
			Font f = new Font(fontpars[0].ToString(), Convert.ToInt32(fontpars[1]), ((fs.IndexOf("B") >= 0) ? FontStyle.Bold : 0) | ((fs.IndexOf("I") >= 0) ? FontStyle.Italic : 0) | ((fs.IndexOf("U") >= 0) ? FontStyle.Underline : 0) | ((fs.IndexOf("S") >= 0) ? FontStyle.Strikeout : 0));
			analysisControl1.LabelFont = f;			
			ret = null;
		}

		private void fnSETPANELFONT(ref object ret, object [] fontpars)
		{
			if (fontpars.Length != 3) throw new Exception("Font face, size and style are required as parameters!");
			string fs = fontpars[2].ToString().ToUpper();
			Font f = new Font(fontpars[0].ToString(), Convert.ToInt32(fontpars[1]), ((fs.IndexOf("B") >= 0) ? FontStyle.Bold : 0) | ((fs.IndexOf("I") >= 0) ? FontStyle.Italic : 0) | ((fs.IndexOf("U") >= 0) ? FontStyle.Underline : 0) | ((fs.IndexOf("S") >= 0) ? FontStyle.Strikeout : 0));
			analysisControl1.PanelFont = f;			
			ret = null;
		}

		private void fnSETPANEL(ref object ret, object [] fontpars)
		{
			if (fontpars.Length != 1) throw new Exception("Panel text is required as parameter!");
			analysisControl1.Panel = fontpars[0].ToString();
			ret = null;
		}

		private void fnSETPANELFORMAT(ref object ret, object [] fontpars)
		{
			if (fontpars.Length != 1) throw new Exception("Panel format is required as parameter!");
			analysisControl1.PanelFormat = fontpars[0].ToString();
			ret = null;
		}

		private void fnSETPANELX(ref object ret, object [] fontpars)
		{
			if (fontpars.Length != 1) throw new Exception("Panel X position is required as parameter!");
			analysisControl1.PanelX = Convert.ToDouble(fontpars[0]);
			ret = null;
		}

		private void fnSETPANELY(ref object ret, object [] fontpars)
		{
			if (fontpars.Length != 1) throw new Exception("Panel Y position is required as parameter!");
			analysisControl1.PanelY = Convert.ToDouble(fontpars[0]);
			ret = null;
		}

		#endregion

        private void menuViewAsTracks_Click(object sender, EventArgs e)
        {
            if (analysisControl1.Variables <= 0) return;
            string datasetname = analysisControl1.DataSetName(analysisControl1.CurrentDataSet);
            if (analysisControl1.CurrentDataRows <= 0) return;
            string[] vars = new string[analysisControl1.Variables];
            int i;
            System.Collections.Specialized.OrderedDictionary vardict = new System.Collections.Specialized.OrderedDictionary();
            for (i = 0; i < vars.Length; i++)
            {
                vars[i] = analysisControl1.VariableName(i);
                vardict.Add(vars[i], i);
            }            
            ViewAsTracksDataSelector vwdlg = new ViewAsTracksDataSelector();
            vwdlg.Variables = vars;
            if (vwdlg.ShowDialog() == DialogResult.OK)
            {                
                System.Collections.Specialized.OrderedDictionary seldict = vwdlg.Selection;
                for (i = 0; i < seldict.Count; i++)
                    if (seldict[i] is string)
                        seldict[i] = vardict[seldict[i]];
                SySal.BasicTypes.Cuboid qbe = new SySal.BasicTypes.Cuboid();                
                GDI3D.Scene scene = new GDI3D.Scene();
                scene.Lines = new GDI3D.Line[analysisControl1.CurrentDataRows];
                scene.OwnerSignatures = new string[analysisControl1.CurrentDataRows];
                for (i = 0; i < scene.Lines.Length; i++)
                {
                    GDI3D.Line line = new GDI3D.Line();
                    line.XS = (seldict["Xstart"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Xstart"]) : (double)seldict["Xstart"];
                    line.YS = (seldict["Ystart"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Ystart"]) : (double)seldict["Ystart"];
                    line.ZS = (seldict["Zstart"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Zstart"]) : (double)seldict["Zstart"];
                    if (i == 0)
                    {
                        qbe.MinX = qbe.MaxX = line.XS;
                        qbe.MinY = qbe.MaxY = line.YS;
                        qbe.MinZ = qbe.MaxZ = line.ZS;
                    }
                    else
                    {
                        if (line.XS < qbe.MinX) qbe.MinX = line.XS;
                        else if (line.XS > qbe.MaxX) qbe.MaxX = line.XS;
                        if (line.YS < qbe.MinY) qbe.MinY = line.YS;
                        else if (line.YS > qbe.MaxY) qbe.MaxY = line.YS;
                        if (line.ZS < qbe.MinZ) qbe.MinZ = line.ZS;
                        else if (line.ZS > qbe.MaxZ) qbe.MaxZ = line.ZS;
                    }
                    switch (vwdlg.Mode)
                    {
                        case ViewAsTracksDataSelector.SegmentMode.StartEnd:
                            line.XF = (seldict["Xend"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Xend"]) : (double)seldict["Xend"];
                            line.YF = (seldict["Yend"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Yend"]) : (double)seldict["Yend"];
                            line.ZF = (seldict["Zend"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Zend"]) : (double)seldict["Zend"];
                            break;

                        case ViewAsTracksDataSelector.SegmentMode.StartSlopeLength:
                            line.ZF = (seldict["Length"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Length"]) : (double)seldict["Length"];
                            line.XF = ((seldict["Xslope"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Xslope"]) : (double)seldict["Xslope"]) * line.ZF + line.XS;
                            line.YF = ((seldict["Yslope"] is int) ? analysisControl1.GetDouble(i, (int)seldict["Yslope"]) : (double)seldict["Yslope"]) * line.ZF + line.YS;
                            line.ZF += line.ZS;                            
                            break;
                    }
                    Color c;
                    if (seldict["Hue"] is int)
                    {
                        c = NumericalTools.Plot.Hue(analysisControl1.GetDouble(i, (int)seldict["Hue"]));
                    }
                    else if (seldict["Hue"] is double)
                    {
                        c = NumericalTools.Plot.Hue((double)seldict["Hue"]);
                    }
                    else
                    {
                        c = NumericalTools.Plot.Hue(0.5);
                    }
                    line.R = c.R;
                    line.G = c.G;
                    line.B = c.B;
                    line.Owner = i;
                    scene.OwnerSignatures[i] = datasetname + " " + i;
                    scene.Lines[i] = line;
                }                
                scene.Points = new GDI3D.Point[0];
                scene.CameraSpottingX = 0.5 * (qbe.MinX + qbe.MaxX);
                scene.CameraSpottingY = 0.5 * (qbe.MinY + qbe.MaxY);
                scene.CameraSpottingZ = 0.5 * (qbe.MinZ + qbe.MaxZ);
                scene.CameraDirectionX = 0.0;
                scene.CameraDirectionY = 0.0;
                scene.CameraDirectionZ = -1.0;
                scene.CameraNormalX = 0.0;
                scene.CameraNormalY = 1.0;
                scene.CameraNormalZ = 0.0;
                scene.CameraDistance = 2.0 * Math.Max(qbe.MaxX - qbe.MinX, qbe.MaxY - qbe.MinY) + 1.0 + qbe.MaxZ - qbe.MinZ;
                scene.Zoom = 1.0;
                string codebase = System.Reflection.Assembly.GetExecutingAssembly().CodeBase;
                object o = System.Activator.CreateInstanceFrom(codebase.Remove(codebase.LastIndexOf("QuickDataCheck")) + "X3LView.exe", "SySal.Executables.X3LView.X3LView").Unwrap();
                o.GetType().InvokeMember("SetScene", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, o, new object[1] { scene });
                o.GetType().InvokeMember("Show", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, o, new object[0] {}); 
            }
        }
    }

    delegate void AddOutput(string s);

    class IntMIPBaseTrack : SySal.Scanning.MIPBaseTrack
    {
        public static SySal.Tracking.MIPEmulsionTrackInfo GetInfo(SySal.Scanning.MIPBaseTrack t) { return IntMIPBaseTrack.AccessInfo(t); }
    }

    class IntMIPIndexedEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
    {
        public static SySal.Tracking.MIPEmulsionTrackInfo GetInfo(SySal.Scanning.MIPIndexedEmulsionTrack t) { return IntMIPIndexedEmulsionTrack.AccessInfo(t); }
    }
}
