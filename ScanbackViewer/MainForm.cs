using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using SySal.TotalScan;

namespace SySal.Executables.ScanbackViewer
{
	/// <summary>
	/// ScanbackViewer - GUI tool to study scanback/scanforth results.
	/// </summary>
	/// <remarks>
	/// <para>
	/// ScanbackViewer is oriented to give a detailed account of scanback/scanforth procedures 
	/// on a path-by-path basis rather than from a statistical point of view.
	/// </para>
	/// <para>
	/// The standard usage of this tool requires the following steps:
	/// <list type="number">
	/// <item><term>connect to a DB (using <see cref="SySal.Executables.ScanbackViewer.DBLoginForm">DBLoginForm</see>);</term></item>
	/// <item><term>select a brick;</term></item>
	/// <item><term>select a scanback/scanforth operation;</term></item>
	/// <item><term>select one or more available paths whose history is to be studied;</term></item>
	/// <item><term>click on the "Add" button to select a color for the paths and finally have them added to the 3D display.</term></item>
	/// </list>
	/// The paths displayed can be studied track-by-track. The selected color is used for base-track candidates, and a darker segment 
	/// connects base-tracks candidates found on different plates.
	/// </para>
	/// <para>Clicking on a plate marker shows information about the plate.</para>
	/// <para>Clicking on a base-track marker shows information about that base track, in the format: <c>PATH ID_PLATE FPX FPY FSX FSY</c>.</para>
	/// <para>Clicking on a base-track connector marker shows the full history of the corresponding path, in the format: <c>PATH ID_PLATE PPX PPY PSX PSY GRAINS FPX FPY FSX FSY Z</c>.</para>
	/// <para><c>F/PPX/Y</c> = [Found | Predicted] Position [X | Y]</para> 
	/// <para><c>F/PSX/Y</c> = [Found | Predicted] Slope [X | Y]</para> 
	/// <para>The path view is a 3D view provided by <see cref="GDI3D.Scene">GDI3D</see> and its <see cref="GDI3D.Control.GDIDisplay">Display Control</see>.</para>
	/// <para>By dragging the image with the <u>right mouse button</u> pressed, one can rotate/pan it.</para>
	/// <para>Default points of view are available (XY, YZ, XZ).</para>
	/// <para>To center the view on a particularly interesting point, click on <i>Set Focus</i> and then <u>left-click</u> on the interesting point.</para>
	/// <para>The image obtained can be saved to an <see cref="GDI3D.Scene">X3L</see> file, and/or it can be merged with an X3L file. This is typically used to overlap the scanback/scanoforth history with TotalScan results.</para>	
    /// <para>When the <c>Enable Detailed View</c> button is clicked, the next left-click near a scanback path will open an X3LView slave window with the details of all tracks seen in the scanback procedure, with the selected candidates enhanced by lighter colors. 
    /// A rainbow color code ranging from magenta (upstream) through red (downstream) will denote the Z level of tracks. Clicking on <c>Disable Detailed View will disable this function and switch back to the normal behaviour of left-click.</c></para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		const double BaseThickness = 210.0;

		private System.Windows.Forms.Button buttonConnect;
		private System.Windows.Forms.ComboBox comboBricks;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.ComboBox comboOps;
		private System.Windows.Forms.Button buttonAdd;
		private GDI3D.Control.GDIDisplay gdiDisp;
		private System.Windows.Forms.TextBox textInfo;
		private System.Windows.Forms.ColorDialog colorSelector;
		private System.Windows.Forms.CheckedListBox listPaths;
		private System.Windows.Forms.Button buttonSelAll;
		private System.Windows.Forms.Button buttonSelNone;
		private System.Windows.Forms.Button buttonIn;
		private System.Windows.Forms.Button buttonOut;
		private System.Windows.Forms.Button buttonXZ;
		private System.Windows.Forms.Button buttonYZ;
		private System.Windows.Forms.Button buttonXY;
		private System.Windows.Forms.Button buttonPan;
		private System.Windows.Forms.Button buttonRot;
		private System.Windows.Forms.Button buttonSavePlot;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		private System.Windows.Forms.Button buttonSetFocus;
		private System.Windows.Forms.Button buttonMergePlot;
		private System.Windows.Forms.OpenFileDialog openMergePlotFileDialog;
		private System.Windows.Forms.Button buttonBackground;
		private System.Windows.Forms.ColorDialog backgroundSelector;
        private Button buttonRefresh;
        private Button buttonDetail;
        private Button buttonRequestVolume;
        private Label label3;
        private ComboBox comboPlates;
        private Button buttonMomentum;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//			
			gdiDisp.DoubleClickSelect = new GDI3D.Control.SelectObject(ShowInfo);

            m_MCSForm = new MomentumForm();
            m_MCSForm.MCS = new SySal.Processing.MCSLikelihood.MomentumEstimator();            
		}

        void ShowDetail(object o)
        {
            if (o is ScanbackPath)
                try
                {
                    Conn.Open();
                    ScanbackPath sbp = (ScanbackPath)o;
                    System.Data.DataSet ds = new System.Data.DataSet();
                    new SySal.OperaDb.OperaDbDataAdapter("SELECT IDPL as ID_PLATE, ID_ZONE, ID, GRAINS, AREASUM, POSX, POSY, SLOPEX, SLOPEY, SIGMA, Z, DECODE(ID_CANDIDATE - ID, 0, 1, 0) AS ISCANDIDATE FROM TB_MIPBASETRACKS INNER JOIN " +
                        "(SELECT idb, idpl, idz, ID_CANDIDATE, Z FROM" +
                        "(SELECT ID_EVENTBRICK as idb, ID_PLATE as idpl, ID_ZONE as idz, ID_CANDIDATE FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PATH) IN " +
                        "(SELECT ID_EVENTBRICK, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + sbp.Id_Eventbrick + " AND ID_PROCESSOPERATION = " + sbp.Id_Processoperation + " AND PATH = " + sbp.Path + "))" +
                        "INNER JOIN TB_PLATES ON (ID_EVENTBRICK = idb AND ID = idpl)) ON (ID_EVENTBRICK = idb AND ID_ZONE = idz) ORDER BY Z DESC", Conn, null).Fill(ds);
                    if (ds.Tables[0].Rows.Count <= 0) return;
                    double minZ, maxZ, Z;
                    int i;
                    maxZ = minZ = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][10]);
                    for (i = 1; i < ds.Tables[0].Rows.Count; i++)
                    {
                        Z = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][10]);
                        if (Z < minZ) minZ = Z;
                        else if (Z > maxZ) maxZ = Z;
                    }
                    if (maxZ == minZ) maxZ = minZ + 1.0;
                    GDI3D.Scene scene = new GDI3D.Scene();
                    scene.Lines = new GDI3D.Line[ds.Tables[0].Rows.Count];
                    scene.OwnerSignatures = new string[scene.Lines.Length];
                    scene.Points = new GDI3D.Point[0];
                    for (i = 0; i < scene.Lines.Length; i++)
                    {
                        System.Data.DataRow dr = ds.Tables[0].Rows[i];
                        double x = SySal.OperaDb.Convert.ToDouble(dr[5]);
                        double y = SySal.OperaDb.Convert.ToDouble(dr[6]);
                        double z = SySal.OperaDb.Convert.ToDouble(dr[10]);
                        Color c = NumericalTools.Plot.Hue((z - minZ) / (maxZ - minZ));
                        bool iscand = (SySal.OperaDb.Convert.ToInt32(dr[11]) != 0);
                        int r = c.R / 2;
                        int g = c.G / 2;
                        int b = c.B / 2;
                        if (iscand)
                        {
                            r += 127;
                            g += 127;
                            b += 127;
                        }
                        scene.Lines[i] = new GDI3D.Line(x, y, z, x - 200.0 * SySal.OperaDb.Convert.ToDouble(dr[7]), y - 200.0 * SySal.OperaDb.Convert.ToDouble(dr[8]), z - 200.0, i, r, g, b);
                        scene.OwnerSignatures[i] = "Plate " + dr[0].ToString() + "\r\nZone " + dr[1] + "\r\nId " + dr[2] + "\r\nGrains " + dr[3] + "\r\nAreasum " + dr[4] + "\r\nPosX " + dr[5].ToString() + "\r\nPosY " + dr[6].ToString() + "\r\nSlopeX " + dr[7].ToString() + "\r\nSlopeY " + dr[8].ToString() + "\r\nSigma " + dr[9].ToString() + "\r\nZ " + dr[10].ToString() + "\r\nIsCandidate " + dr[11].ToString();
                    }
                    scene.CameraSpottingX = scene.Lines[0].XS;
                    scene.CameraSpottingY = scene.Lines[0].YS;
                    scene.CameraSpottingZ = scene.Lines[0].ZS;
                    scene.CameraDirectionX = 0.0;
                    scene.CameraDirectionY = 0.0;
                    scene.CameraDirectionZ = -1.0;
                    scene.CameraNormalX = 0.0;
                    scene.CameraNormalY = 1.0;
                    scene.CameraNormalZ = 0.0;
                    scene.CameraDistance = Math.Abs(scene.Lines[0].ZS - scene.Lines[scene.Lines.Length - 1].ZF) * 2.0 + 1000.0;
                    scene.Zoom = 1.0;
                    string codebase = System.Reflection.Assembly.GetExecutingAssembly().CodeBase;
                    object v = System.Activator.CreateInstanceFrom(codebase.Remove(codebase.LastIndexOf("ScanbackViewer")) + "X3LView.exe", "SySal.Executables.X3LView.X3LView").Unwrap();
                    v.GetType().InvokeMember("SetScene", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, v, new object[1] { scene });
                    v.GetType().InvokeMember("Show", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, v, new object[0] { });
                }
                catch (Exception) { }
                finally
                {
                    Conn.Close();
                }
            else ShowInfo(o);
        }

		void ShowInfo(object o)
		{
            m_SelectedScanbackPath = null;
            comboPlates.Items.Clear();
			textInfo.Text = o.ToString();
            if (o is ScanbackPath)
            {
                m_SelectedScanbackPath = (ScanbackPath)o;
                foreach (int p in m_SelectedScanbackPath.PlatesFound)
                    comboPlates.Items.Add(p.ToString());
            }
		}

        ScanbackPath m_SelectedScanbackPath = null;

		DBLoginForm DBLF = new DBLoginForm();

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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.gdiDisp = new GDI3D.Control.GDIDisplay();
            this.buttonConnect = new System.Windows.Forms.Button();
            this.comboBricks = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.comboOps = new System.Windows.Forms.ComboBox();
            this.buttonAdd = new System.Windows.Forms.Button();
            this.textInfo = new System.Windows.Forms.TextBox();
            this.colorSelector = new System.Windows.Forms.ColorDialog();
            this.listPaths = new System.Windows.Forms.CheckedListBox();
            this.buttonSelAll = new System.Windows.Forms.Button();
            this.buttonSelNone = new System.Windows.Forms.Button();
            this.buttonIn = new System.Windows.Forms.Button();
            this.buttonOut = new System.Windows.Forms.Button();
            this.buttonXZ = new System.Windows.Forms.Button();
            this.buttonYZ = new System.Windows.Forms.Button();
            this.buttonXY = new System.Windows.Forms.Button();
            this.buttonPan = new System.Windows.Forms.Button();
            this.buttonRot = new System.Windows.Forms.Button();
            this.buttonSavePlot = new System.Windows.Forms.Button();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.buttonSetFocus = new System.Windows.Forms.Button();
            this.buttonMergePlot = new System.Windows.Forms.Button();
            this.openMergePlotFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.buttonBackground = new System.Windows.Forms.Button();
            this.backgroundSelector = new System.Windows.Forms.ColorDialog();
            this.buttonRefresh = new System.Windows.Forms.Button();
            this.buttonDetail = new System.Windows.Forms.Button();
            this.buttonRequestVolume = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.comboPlates = new System.Windows.Forms.ComboBox();
            this.buttonMomentum = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // gdiDisp
            // 
            this.gdiDisp.Alpha = 0.50196078431372548;
            this.gdiDisp.AutoRender = true;
            this.gdiDisp.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))));
            this.gdiDisp.BorderWidth = 1;
            this.gdiDisp.ClickSelect = null;
            this.gdiDisp.Distance = 10;
            this.gdiDisp.DoubleClickSelect = null;
            this.gdiDisp.Infinity = true;
            this.gdiDisp.LabelFontName = "Arial";
            this.gdiDisp.LabelFontSize = 12;
            this.gdiDisp.LineWidth = 2;
            this.gdiDisp.Location = new System.Drawing.Point(8, 40);
            this.gdiDisp.MouseMode = GDI3D.Control.MouseMotion.Rotate;
            this.gdiDisp.MouseMultiplier = 0.01;
            this.gdiDisp.Name = "gdiDisp";
            this.gdiDisp.NextClickSetsCenter = false;
            this.gdiDisp.PointSize = 5;
            this.gdiDisp.Size = new System.Drawing.Size(712, 480);
            this.gdiDisp.TabIndex = 0;
            this.gdiDisp.Zoom = 100;
            // 
            // buttonConnect
            // 
            this.buttonConnect.Location = new System.Drawing.Point(8, 8);
            this.buttonConnect.Name = "buttonConnect";
            this.buttonConnect.Size = new System.Drawing.Size(128, 24);
            this.buttonConnect.TabIndex = 1;
            this.buttonConnect.Text = "Connect to DB";
            this.buttonConnect.Click += new System.EventHandler(this.buttonConnect_Click);
            // 
            // comboBricks
            // 
            this.comboBricks.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBricks.Location = new System.Drawing.Point(200, 8);
            this.comboBricks.Name = "comboBricks";
            this.comboBricks.Size = new System.Drawing.Size(80, 21);
            this.comboBricks.TabIndex = 2;
            this.comboBricks.SelectionChangeCommitted += new System.EventHandler(this.OnBricksSelected);
            this.comboBricks.SelectedIndexChanged += new System.EventHandler(this.OnBricksSelected);
            this.comboBricks.ValueMemberChanged += new System.EventHandler(this.OnBricksSelected);
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(144, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(48, 24);
            this.label1.TabIndex = 3;
            this.label1.Text = "Bricks";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(304, 8);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(80, 24);
            this.label2.TabIndex = 5;
            this.label2.Text = "Operations";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // comboOps
            // 
            this.comboOps.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboOps.Location = new System.Drawing.Point(392, 8);
            this.comboOps.Name = "comboOps";
            this.comboOps.Size = new System.Drawing.Size(136, 21);
            this.comboOps.TabIndex = 4;
            this.comboOps.SelectionChangeCommitted += new System.EventHandler(this.OnOpsSelected);
            this.comboOps.SelectedIndexChanged += new System.EventHandler(this.OnOpsSelected);
            // 
            // buttonAdd
            // 
            this.buttonAdd.Location = new System.Drawing.Point(656, 8);
            this.buttonAdd.Name = "buttonAdd";
            this.buttonAdd.Size = new System.Drawing.Size(64, 24);
            this.buttonAdd.TabIndex = 6;
            this.buttonAdd.Text = "Add paths";
            this.buttonAdd.Click += new System.EventHandler(this.buttonAdd_Click);
            // 
            // textInfo
            // 
            this.textInfo.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.textInfo.Location = new System.Drawing.Point(8, 528);
            this.textInfo.Multiline = true;
            this.textInfo.Name = "textInfo";
            this.textInfo.ReadOnly = true;
            this.textInfo.Size = new System.Drawing.Size(582, 96);
            this.textInfo.TabIndex = 23;
            // 
            // colorSelector
            // 
            this.colorSelector.AllowFullOpen = false;
            this.colorSelector.Color = System.Drawing.Color.Red;
            this.colorSelector.SolidColorOnly = true;
            // 
            // listPaths
            // 
            this.listPaths.CheckOnClick = true;
            this.listPaths.Location = new System.Drawing.Point(728, 8);
            this.listPaths.Name = "listPaths";
            this.listPaths.Size = new System.Drawing.Size(125, 154);
            this.listPaths.TabIndex = 8;
            this.listPaths.ThreeDCheckBoxes = true;
            // 
            // buttonSelAll
            // 
            this.buttonSelAll.Location = new System.Drawing.Point(728, 176);
            this.buttonSelAll.Name = "buttonSelAll";
            this.buttonSelAll.Size = new System.Drawing.Size(62, 24);
            this.buttonSelAll.TabIndex = 9;
            this.buttonSelAll.Text = "All";
            this.buttonSelAll.Click += new System.EventHandler(this.buttonSelAll_Click);
            // 
            // buttonSelNone
            // 
            this.buttonSelNone.Location = new System.Drawing.Point(791, 176);
            this.buttonSelNone.Name = "buttonSelNone";
            this.buttonSelNone.Size = new System.Drawing.Size(62, 24);
            this.buttonSelNone.TabIndex = 10;
            this.buttonSelNone.Text = "None";
            this.buttonSelNone.Click += new System.EventHandler(this.buttonSelNone_Click);
            // 
            // buttonIn
            // 
            this.buttonIn.Location = new System.Drawing.Point(728, 224);
            this.buttonIn.Name = "buttonIn";
            this.buttonIn.Size = new System.Drawing.Size(62, 24);
            this.buttonIn.TabIndex = 11;
            this.buttonIn.Text = "In";
            this.buttonIn.Click += new System.EventHandler(this.buttonIn_Click);
            // 
            // buttonOut
            // 
            this.buttonOut.Location = new System.Drawing.Point(791, 224);
            this.buttonOut.Name = "buttonOut";
            this.buttonOut.Size = new System.Drawing.Size(62, 24);
            this.buttonOut.TabIndex = 12;
            this.buttonOut.Text = "Out";
            this.buttonOut.Click += new System.EventHandler(this.buttonOut_Click);
            // 
            // buttonXZ
            // 
            this.buttonXZ.Location = new System.Drawing.Point(728, 256);
            this.buttonXZ.Name = "buttonXZ";
            this.buttonXZ.Size = new System.Drawing.Size(62, 24);
            this.buttonXZ.TabIndex = 13;
            this.buttonXZ.Text = "XZ";
            this.buttonXZ.Click += new System.EventHandler(this.buttonXZ_Click);
            // 
            // buttonYZ
            // 
            this.buttonYZ.Location = new System.Drawing.Point(728, 288);
            this.buttonYZ.Name = "buttonYZ";
            this.buttonYZ.Size = new System.Drawing.Size(62, 24);
            this.buttonYZ.TabIndex = 14;
            this.buttonYZ.Text = "YZ";
            this.buttonYZ.Click += new System.EventHandler(this.buttonYZ_Click);
            // 
            // buttonXY
            // 
            this.buttonXY.Location = new System.Drawing.Point(728, 320);
            this.buttonXY.Name = "buttonXY";
            this.buttonXY.Size = new System.Drawing.Size(62, 24);
            this.buttonXY.TabIndex = 15;
            this.buttonXY.Text = "XY";
            this.buttonXY.Click += new System.EventHandler(this.buttonXY_Click);
            // 
            // buttonPan
            // 
            this.buttonPan.Location = new System.Drawing.Point(791, 256);
            this.buttonPan.Name = "buttonPan";
            this.buttonPan.Size = new System.Drawing.Size(62, 24);
            this.buttonPan.TabIndex = 16;
            this.buttonPan.Text = "Pan";
            this.buttonPan.Click += new System.EventHandler(this.buttonPan_Click);
            // 
            // buttonRot
            // 
            this.buttonRot.Location = new System.Drawing.Point(791, 288);
            this.buttonRot.Name = "buttonRot";
            this.buttonRot.Size = new System.Drawing.Size(62, 24);
            this.buttonRot.TabIndex = 17;
            this.buttonRot.Text = "Rot";
            this.buttonRot.Click += new System.EventHandler(this.buttonRot_Click);
            // 
            // buttonSavePlot
            // 
            this.buttonSavePlot.Location = new System.Drawing.Point(728, 600);
            this.buttonSavePlot.Name = "buttonSavePlot";
            this.buttonSavePlot.Size = new System.Drawing.Size(125, 24);
            this.buttonSavePlot.TabIndex = 22;
            this.buttonSavePlot.Text = "Save Plot";
            this.buttonSavePlot.Click += new System.EventHandler(this.buttonSavePlot_Click);
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.Filter = "XML 3D files (*.x3l)|*.x3l|All files (*.*)|*.*";
            this.saveFileDialog1.Title = "Select output file";
            // 
            // buttonSetFocus
            // 
            this.buttonSetFocus.Location = new System.Drawing.Point(728, 352);
            this.buttonSetFocus.Name = "buttonSetFocus";
            this.buttonSetFocus.Size = new System.Drawing.Size(125, 24);
            this.buttonSetFocus.TabIndex = 19;
            this.buttonSetFocus.Text = "Set Focus";
            this.buttonSetFocus.Click += new System.EventHandler(this.buttonSetFocus_Click);
            // 
            // buttonMergePlot
            // 
            this.buttonMergePlot.Location = new System.Drawing.Point(728, 564);
            this.buttonMergePlot.Name = "buttonMergePlot";
            this.buttonMergePlot.Size = new System.Drawing.Size(125, 24);
            this.buttonMergePlot.TabIndex = 21;
            this.buttonMergePlot.Text = "Merge Plot";
            this.buttonMergePlot.Click += new System.EventHandler(this.buttonMergePlot_Click);
            // 
            // openMergePlotFileDialog
            // 
            this.openMergePlotFileDialog.Filter = "XML 3D files (*.x3l)|*.x3l|All files (*.*)|*.*";
            this.openMergePlotFileDialog.Multiselect = true;
            this.openMergePlotFileDialog.Title = "Select plot to merge";
            // 
            // buttonBackground
            // 
            this.buttonBackground.Location = new System.Drawing.Point(728, 496);
            this.buttonBackground.Name = "buttonBackground";
            this.buttonBackground.Size = new System.Drawing.Size(125, 24);
            this.buttonBackground.TabIndex = 20;
            this.buttonBackground.Text = "Background";
            this.buttonBackground.Click += new System.EventHandler(this.buttonBackground_Click);
            // 
            // backgroundSelector
            // 
            this.backgroundSelector.AllowFullOpen = false;
            this.backgroundSelector.Color = System.Drawing.Color.White;
            this.backgroundSelector.SolidColorOnly = true;
            // 
            // buttonRefresh
            // 
            this.buttonRefresh.Location = new System.Drawing.Point(728, 528);
            this.buttonRefresh.Name = "buttonRefresh";
            this.buttonRefresh.Size = new System.Drawing.Size(125, 24);
            this.buttonRefresh.TabIndex = 24;
            this.buttonRefresh.Text = "Refresh";
            this.buttonRefresh.Click += new System.EventHandler(this.buttonRefresh_Click);
            // 
            // buttonDetail
            // 
            this.buttonDetail.Location = new System.Drawing.Point(728, 382);
            this.buttonDetail.Name = "buttonDetail";
            this.buttonDetail.Size = new System.Drawing.Size(125, 24);
            this.buttonDetail.TabIndex = 25;
            this.buttonDetail.Text = "Enable detail view";
            this.buttonDetail.Click += new System.EventHandler(this.buttonDetail_Click);
            // 
            // buttonRequestVolume
            // 
            this.buttonRequestVolume.Location = new System.Drawing.Point(595, 528);
            this.buttonRequestVolume.Name = "buttonRequestVolume";
            this.buttonRequestVolume.Size = new System.Drawing.Size(125, 24);
            this.buttonRequestVolume.TabIndex = 26;
            this.buttonRequestVolume.Text = "Request Volume";
            this.buttonRequestVolume.Click += new System.EventHandler(this.buttonRequestVolume_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(597, 567);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(31, 13);
            this.label3.TabIndex = 27;
            this.label3.Text = "Plate";
            // 
            // comboPlates
            // 
            this.comboPlates.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboPlates.FormattingEnabled = true;
            this.comboPlates.Location = new System.Drawing.Point(638, 564);
            this.comboPlates.Name = "comboPlates";
            this.comboPlates.Size = new System.Drawing.Size(81, 21);
            this.comboPlates.TabIndex = 28;
            // 
            // buttonMomentum
            // 
            this.buttonMomentum.Location = new System.Drawing.Point(594, 600);
            this.buttonMomentum.Name = "buttonMomentum";
            this.buttonMomentum.Size = new System.Drawing.Size(125, 24);
            this.buttonMomentum.TabIndex = 29;
            this.buttonMomentum.Text = "Momentum";
            this.buttonMomentum.Click += new System.EventHandler(this.buttonMomentum_Click);
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(865, 628);
            this.Controls.Add(this.buttonMomentum);
            this.Controls.Add(this.comboPlates);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.buttonRequestVolume);
            this.Controls.Add(this.buttonDetail);
            this.Controls.Add(this.buttonRefresh);
            this.Controls.Add(this.buttonBackground);
            this.Controls.Add(this.buttonMergePlot);
            this.Controls.Add(this.buttonSetFocus);
            this.Controls.Add(this.buttonSavePlot);
            this.Controls.Add(this.buttonRot);
            this.Controls.Add(this.buttonPan);
            this.Controls.Add(this.buttonXZ);
            this.Controls.Add(this.buttonYZ);
            this.Controls.Add(this.buttonXY);
            this.Controls.Add(this.buttonOut);
            this.Controls.Add(this.buttonIn);
            this.Controls.Add(this.buttonSelNone);
            this.Controls.Add(this.buttonSelAll);
            this.Controls.Add(this.listPaths);
            this.Controls.Add(this.textInfo);
            this.Controls.Add(this.buttonAdd);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboOps);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.comboBricks);
            this.Controls.Add(this.buttonConnect);
            this.Controls.Add(this.gdiDisp);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Scanback Viewer";
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
            Application.EnableVisualStyles();
			Application.Run(new MainForm());
		}

		SySal.OperaDb.OperaDbConnection Conn = null;		

		private void buttonConnect_Click(object sender, System.EventArgs e)
		{
			if (DBLF.ShowDialog() == DialogResult.OK)
			{
				if (Conn != null) Conn.Close();
				Conn = null;
                try
                {
                    Conn = new SySal.OperaDb.OperaDbConnection(DBLF.textDB.Text, DBLF.textUser.Text, DBLF.textPwd.Text);
                    Conn.Open();
                    SySal.OperaDb.Schema.DB = Conn;
                    System.Data.DataSet ds = new System.Data.DataSet();
                    new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM TB_EVENTBRICKS ORDER BY ID ASC", Conn, null).Fill(ds);
                    comboBricks.Items.Clear();
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                        comboBricks.Items.Add(dr[0].ToString());
                    comboOps.Items.Clear();
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Connection error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    if (Conn != null) Conn.Close();
                    Conn = null;
                }
                finally
                {
                    Conn.Close();
                }
			}
		}

        private Color LastColor = Color.White;

        class ScanbackPath
        {
            public string Id_Eventbrick;

            public string Id_Processoperation;

            public string Path;

            public string DisplayString;

            public int[] PlatesFound;

            public ScanbackPath(string _ideventbrick_, string _idprocessoperation_, string _path_, string _displaystring_, int [] _platesfound_, SySal.Tracking.MIPEmulsionTrackInfo [] _measurements_)
            {
                Id_Eventbrick = _ideventbrick_;
                Id_Processoperation = _idprocessoperation_;
                Path = _path_;
                DisplayString = _displaystring_;
                PlatesFound = _platesfound_;
                Measurements = _measurements_;
            }

            public override string ToString() 
            {
                return (string)(DisplayString.Clone());
            }

            public SySal.Tracking.MIPEmulsionTrackInfo[] Measurements;
        }

		private void buttonAdd_Click(object sender, System.EventArgs e)
		{
			if (comboOps.Text == null || comboOps.Text.Length == 0) return;
			gdiDisp.AutoRender = false;
            try
            {
                Conn.Open();
                Color col;
                if (e == null)
                {
                    col = LastColor;
                }
                else
                {
                    if (colorSelector.ShowDialog() != DialogResult.OK) return;
                    col = colorSelector.Color;
                    LastColor = col;
                }
                foreach (object o in listPaths.CheckedItems)
                {
                    System.Data.DataSet ds = new System.Data.DataSet();
                    new SySal.OperaDb.OperaDbDataAdapter("SELECT PATH, ID_PLATE, PPX, PPY, PSX, PSY, GRAINS, FPX, FPY, FSX, FSY, Z FROM VW_SCANBACK_HISTORY WHERE ID_EVENTBRICK = " + comboBricks.Text + " AND ID_PROCESSOPERATION = " + comboOps.Text + " AND PATH = " + o.ToString() + " ORDER BY Z DESC", Conn, null).Fill(ds);
                    string s = "PATH\tPLATE\tPPX\tPPY\tPSX\tPSY\tGRAINS\tFPX\tFPY\tFSX\tFSY";
                    System.Collections.ArrayList platesfound_a = new ArrayList();
                    System.Collections.ArrayList measfound_a = new ArrayList();
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    {
                        if (dr[6] != System.DBNull.Value)
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                            info.Field = SySal.OperaDb.Convert.ToUInt32(dr[1]);
                            info.Count = SySal.OperaDb.Convert.ToUInt16(dr[6]);
                            info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[7]);
                            info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[8]);
                            info.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[9]);
                            info.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[10]);
                            info.Intercept.Z = SySal.OperaDb.Convert.ToDouble(dr[11]);
                            measfound_a.Add(info);
                            platesfound_a.Add(SySal.OperaDb.Convert.ToInt32(dr[1]));
                        }
                        s += "\r\n" + dr[0].ToString() + "\t" + dr[1].ToString() + "\t" + SySal.OperaDb.Convert.ToDouble(dr[2]).ToString("F2") + "\t" + SySal.OperaDb.Convert.ToDouble(dr[3]).ToString("F2") +
                            "\t" + SySal.OperaDb.Convert.ToDouble(dr[4]).ToString("F5") + "\t" + SySal.OperaDb.Convert.ToDouble(dr[5]).ToString("F5") + "\t";
                        if (dr[6] == System.DBNull.Value)
                        {
                            s += "0\t0\t0\t0\t0";
                        }
                        else
                        {
                            s += dr[6].ToString() + "\t" + SySal.OperaDb.Convert.ToDouble(dr[7]).ToString("F2") + "\t" + SySal.OperaDb.Convert.ToDouble(dr[8]).ToString("F2") +
                                "\t" + SySal.OperaDb.Convert.ToDouble(dr[9]).ToString("F5") + "\t" + SySal.OperaDb.Convert.ToDouble(dr[10]).ToString("F5");
                        }
                    }
                    SySal.Tracking.MIPEmulsionTrackInfo[] meas = (SySal.Tracking.MIPEmulsionTrackInfo[])(measfound_a.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo)));
                    int[] platesfound = (int[])platesfound_a.ToArray(typeof(int));
                    bool isfirst = true;
                    double lastx = 0.0, lasty = 0.0, lastz = 0.0, x = 0.0, y = 0.0, z = 0.0, sx = 0.0, sy = 0.0;
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    {
                        if (dr[6] == System.DBNull.Value) continue;
                        x = SySal.OperaDb.Convert.ToDouble(dr[7]);
                        y = SySal.OperaDb.Convert.ToDouble(dr[8]);
                        sx = SySal.OperaDb.Convert.ToDouble(dr[9]);
                        sy = SySal.OperaDb.Convert.ToDouble(dr[10]);
                        z = SySal.OperaDb.Convert.ToDouble(dr[11]);

                        if (isfirst == false)
                        {
                            gdiDisp.Add(new GDI3D.Control.Line(lastx, lasty, lastz, x, y, z, new ScanbackPath(comboBricks.Text, comboOps.Text, dr[0].ToString(), s, platesfound, meas), col.R / 2, col.G / 2, col.B / 2));
                        }
                        gdiDisp.Add(new GDI3D.Control.Line(x, y, z, lastx = x - sx * BaseThickness, lasty = y - sy * BaseThickness, lastz = z - BaseThickness, new ScanbackPath(comboBricks.Text, comboOps.Text, dr[0].ToString(), /*dr[0].ToString() + " " + dr[1].ToString() + " " + x.ToString("F2") + " " + y.ToString("F2") + " " + sx.ToString("F5") + " " + sy.ToString("F5")*/ s, platesfound, meas), col.R, col.G, col.B));
                        isfirst = false;
                    }

                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                Conn.Close();
            }
			gdiDisp.AutoRender = true;
			gdiDisp.Render();
		}

		SySal.BasicTypes.Cuboid Extents;

        private GDI3D.Control.Line[] BrickLines = null;

		private void OnBricksSelected(object sender, System.EventArgs e)
		{
            m_SelectedScanbackPath = null;
			if (Conn == null) return;
            try
            {
                Conn.Open();
                gdiDisp.AutoRender = false;
                int id_brick = SySal.OperaDb.Convert.ToInt32(comboBricks.Text.ToString());
                System.Data.DataSet ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT MINX - ZEROX, MINY - ZEROY, MINZ - ZEROZ, MAXX - ZEROX, MAXY - ZEROY, MAXZ - ZEROZ FROM TB_EVENTBRICKS WHERE ID = " + id_brick, Conn, null).Fill(ds);
                Extents.MinX = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][0]);
                Extents.MinY = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][1]);
                Extents.MinZ = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][2]);
                Extents.MaxX = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][3]);
                Extents.MaxY = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][4]);
                Extents.MaxZ = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][5]);
                ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, Z, DAMAGED FROM OPERA.VW_PLATES WHERE ID_EVENTBRICK = " + id_brick + " ORDER BY Z ASC", Conn, null).Fill(ds);
                gdiDisp.Clear();
                double Z;
                System.Collections.ArrayList bricklines = new ArrayList();
                GDI3D.Control.Line line;
                Geometry.LayerStart[] mcslys = new Geometry.LayerStart[ds.Tables[0].Rows.Count * 2];
                int ly = 0;
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                {
                    mcslys[ly].ZMin = SySal.OperaDb.Convert.ToDouble(dr[1]) - 250.0;
                    mcslys[ly++].RadiationLength = 29000.0;
                    mcslys[ly].ZMin = mcslys[ly - 1].ZMin + 300.0;
                    mcslys[ly++].RadiationLength = 5600.0;
                    Z = SySal.OperaDb.Convert.ToDouble(dr[1]);
                    string s = "Plate #" + dr[0] + " Z = " + dr[1] + " DAMAGED = " + dr[2];
                    int coloff = (String.Compare(dr[2].ToString(), "N", true) == 0) ? 0 : 32;
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MinX, Extents.MinY, Z, Extents.MaxX, Extents.MinY, Z, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MinX, Extents.MaxY, Z, Extents.MaxX, Extents.MaxY, Z, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MinX, Extents.MinY, Z, Extents.MinX, Extents.MaxY, Z, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MaxX, Extents.MinY, Z, Extents.MaxX, Extents.MaxY, Z, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MinX, Extents.MinY, Z - BaseThickness, Extents.MaxX, Extents.MinY, Z - BaseThickness, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MinX, Extents.MaxY, Z - BaseThickness, Extents.MaxX, Extents.MaxY, Z - BaseThickness, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MinX, Extents.MinY, Z - BaseThickness, Extents.MinX, Extents.MaxY, Z - BaseThickness, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                    gdiDisp.Add(line = new GDI3D.Control.Line(Extents.MaxX, Extents.MinY, Z - BaseThickness, Extents.MaxX, Extents.MaxY, Z - BaseThickness, s, 32 + coloff, 48, 48)); bricklines.Add(line);
                }
                SySal.Processing.MCSLikelihood.Configuration c = (SySal.Processing.MCSLikelihood.Configuration)m_MCSForm.MCSLikelihood.Config;
                c.Geometry.Layers = mcslys;
                m_MCSForm.MCSLikelihood.Config = c;
                BrickLines = (GDI3D.Control.Line[])bricklines.ToArray(typeof(GDI3D.Control.Line));
                gdiDisp.SetCameraSpotting(0.5 * (Extents.MinX + Extents.MaxX), 0.5 * (Extents.MinY + Extents.MaxY), 0.5 * (Extents.MinZ + Extents.MaxZ));
                gdiDisp.SetCameraOrientation(0, 0, -1, 0, 1, 0);
                gdiDisp.Distance = 30 * (Extents.MaxZ - Extents.MinZ);
                //gdiDisp.Zoom = 5000;
                gdiDisp.Zoom = .001;
                ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT DISTINCT ID_PROCESSOPERATION FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + id_brick + " ORDER BY ID_PROCESSOPERATION ASC", Conn, null).Fill(ds);
                comboOps.Items.Clear();
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    comboOps.Items.Add(dr[0]);
                gdiDisp.AutoRender = true;
                gdiDisp.Render();
            }
            catch (Exception)
            {
                gdiDisp.AutoRender = false;
                return;
            }
            finally
            {
                Conn.Close();
            }
		}

		private void OnOpsSelected(object sender, System.EventArgs e)
		{
			if (Conn == null) return;
            try
            {
                Conn.Open();
                listPaths.Items.Clear();
                int id_brick = SySal.OperaDb.Convert.ToInt32(comboBricks.Text.ToString());
                System.Data.DataSet ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT DISTINCT PATH FROM VW_SCANBACK_HISTORY WHERE ID_EVENTBRICK = " + id_brick + " AND ID_PROCESSOPERATION = " + comboOps.Text + " ORDER BY PATH", Conn, null).Fill(ds);
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    listPaths.Items.Add(dr[0].ToString());
            }
            catch (Exception) { }
            finally
            {
                Conn.Close();
            }
		}

		private void buttonSelAll_Click(object sender, System.EventArgs e)
		{
			int i;
			for (i = 0; i < listPaths.Items.Count; i++)
				listPaths.SetItemChecked(i, true);
		}

		private void buttonSelNone_Click(object sender, System.EventArgs e)
		{
			int i;
			for (i = 0; i < listPaths.Items.Count; i++)
				listPaths.SetItemChecked(i, false);		
		}

		private void buttonIn_Click(object sender, System.EventArgs e)
		{
			gdiDisp.Zoom *= 1.1;
		}

		private void buttonOut_Click(object sender, System.EventArgs e)
		{
			gdiDisp.Zoom /= 1.1;
		}

		private void buttonPan_Click(object sender, System.EventArgs e)
		{
			gdiDisp.MouseMode = GDI3D.Control.MouseMotion.Pan;
		}

		private void buttonRot_Click(object sender, System.EventArgs e)
		{
			gdiDisp.MouseMode = GDI3D.Control.MouseMotion.Rotate;
		}

		private void buttonXZ_Click(object sender, System.EventArgs e)
		{
			gdiDisp.SetCameraOrientation(0, 1, 0, 0, 0, -1);	
			gdiDisp.SetCameraSpotting(0.5 * (Extents.MinX + Extents.MaxX), 0.5 * (Extents.MinY + Extents.MaxY), 0.5 * (Extents.MinZ + Extents.MaxZ));
			gdiDisp.Transform();
			gdiDisp.Render();
		}

		private void buttonYZ_Click(object sender, System.EventArgs e)
		{
			gdiDisp.SetCameraOrientation(-1, 0, 0, 0, 0, -1);
			gdiDisp.SetCameraSpotting(0.5 * (Extents.MinX + Extents.MaxX), 0.5 * (Extents.MinY + Extents.MaxY), 0.5 * (Extents.MinZ + Extents.MaxZ));
			gdiDisp.Transform();
			gdiDisp.Render();
		}

		private void buttonXY_Click(object sender, System.EventArgs e)
		{
			gdiDisp.SetCameraOrientation(0, 0, -1, 0, 1, 0);
			gdiDisp.SetCameraSpotting(0.5 * (Extents.MinX + Extents.MaxX), 0.5 * (Extents.MinY + Extents.MaxY), 0.5 * (Extents.MinZ + Extents.MaxZ));
			gdiDisp.Transform();
			gdiDisp.Render();		
		}

		private void buttonSavePlot_Click(object sender, System.EventArgs e)
		{
			if (saveFileDialog1.ShowDialog() == DialogResult.OK)
			{
				try
				{
					gdiDisp.Save(saveFileDialog1.FileName);
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "Can't save file", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
			}
		}

		private void buttonSetFocus_Click(object sender, System.EventArgs e)
		{
			gdiDisp.NextClickSetsCenter = true;
		}

		private void buttonMergePlot_Click(object sender, System.EventArgs e)
		{
			if (openMergePlotFileDialog.ShowDialog() == DialogResult.OK)
			{
				foreach (string s in openMergePlotFileDialog.FileNames)
					try
					{
						gdiDisp.LoadMergeScene(s);
					}
					catch (Exception x)
					{
						MessageBox.Show(x.Message, "Can't load & merge file", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
			}		
		}

		private void buttonBackground_Click(object sender, System.EventArgs e)
		{
			if (backgroundSelector.ShowDialog() == DialogResult.OK)
			{
				gdiDisp.BackColor = backgroundSelector.Color;
				gdiDisp.Render();
			}					
		}

        private void buttonRefresh_Click(object sender, EventArgs e)
        {
            if (BrickLines == null) return;
            Cursor oldcursor = Cursor;
            Cursor = Cursors.WaitCursor;
            try
            {
                GDI3D.Scene scene = gdiDisp.GetScene();
                scene.Points = new GDI3D.Point[0];
                scene.Lines = new GDI3D.Line[0];
                scene.OwnerSignatures = new string[0];
                gdiDisp.Clear();
                gdiDisp.AutoRender = false;
                gdiDisp.SetScene(scene);
                foreach (GDI3D.Control.Line line in BrickLines)
                    gdiDisp.Add(line);
                buttonAdd_Click(this, null);
                gdiDisp.AutoRender = true;
            }
            finally
            {
                Cursor = oldcursor;
            }
        }

        bool m_OpenDetail = false;

        private void buttonDetail_Click(object sender, EventArgs e)
        {
            m_OpenDetail = !m_OpenDetail;
            if (m_OpenDetail)
            {
                buttonDetail.Text = "Disable detail view";
                gdiDisp.DoubleClickSelect = new GDI3D.Control.SelectObject(ShowDetail);
            }
            else
            {
                buttonDetail.Text = "Enable detail view";
                gdiDisp.DoubleClickSelect = new GDI3D.Control.SelectObject(ShowInfo);
            }
        }

        private void buttonRequestVolume_Click(object sender, EventArgs e)
        {
            if (comboPlates.SelectedItem != null)
            {
                try
                {
                    Conn.Open();
                    SySal.OperaDb.Schema.TB_B_SBPATHS_VOLUMES.Insert(System.Convert.ToInt64(m_SelectedScanbackPath.Id_Eventbrick), System.Convert.ToInt64(m_SelectedScanbackPath.Id_Processoperation), System.Convert.ToInt64(m_SelectedScanbackPath.Path), null, null, Convert.ToInt32(comboPlates.SelectedItem.ToString()));
                    SySal.OperaDb.Schema.TB_B_SBPATHS_VOLUMES.Flush();
                    MessageBox.Show("Volume request inserted.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error entering volume request in TB_B_SBPATHS_VOLUMES", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    Conn.Close();
                }
            }
        }

        private void buttonMomentum_Click(object sender, EventArgs e)
        {
            if (m_SelectedScanbackPath != null)
            {
                m_MCSForm.Measurements = m_SelectedScanbackPath.Measurements;
                m_MCSForm.ShowDialog();
            }
        }

        MomentumForm m_MCSForm;
	}
}
