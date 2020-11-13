using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
	/// <summary>
	/// Displays 3D views of tracks.
	/// </summary>
	/// <remarks>
	/// <para>
	/// The buttons in the Display group perform the following actions:
	/// <list type="table">
	/// <listheader><term>Button</term><description>Action</description></listheader>
	/// <item><term>Default view</term><description>sets the viewing point to its default position.</description></item>
	/// <item><term>Zoom +,-</term><description>zooms in/out the view.</description></item>
	/// <item><term>Rotation/Panning</term><description>when Rotation is enabled, dragging the image with the right mouse button pressed rotates it; otherwise, it is panned (transverse translation).</description></item>
	/// <item><term>Isometric</term><description>when enabled, the view is not in perspective (objects do not shrink with distance), but isometric. Usually, in isometric view, zoom factors must increase in order to have a good image.</description></item>
	/// <item><term>Background</term><description>selects the background color.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// The buttons in the Selection group perform the following actions:
	/// <list type="table">
	/// <listheader><term>Button</term><description>Action</description></listheader>
	/// <item><term>Segments</term><description>shows all segments. This is called "Segment display" mode.</description></item>
	/// <item><term>Tracks</term><description>shows all tracks (with some constraints, see below). This is called "Track display" mode.</description></item>
	/// <item><term>Vertices</term><description>shows all vertices (with some constraints, see below). This is called "Vertex display" mode.</description></item>
	/// <item><term>Graph from Track #</term><description>shows the graph starting from the specified track number. This is called "Graph display" mode.</description></item>
	/// <item><term>Graph from Vertex #</term><description>shows the graph starting from the specified vertex number. This is called "Graph display" mode.</description></item>
	/// <item><term>Show segments</term><description>turns on segment viewing when showing tracks/vertices/graphs.</description></item>
	/// <item><term>Min segments</term><description>minimum number of segments to show a track in "Track display" mode.</description></item>
	/// <item><term>Min tracks</term><description>minimum number of attached tracks to show a vertex in "Vertex display" mode.</description></item>
	/// <item><term>Plot</term><description>refreshes the display.</description></item>
	/// <item><term>Save</term><description>saves the plot in a standard 2D format or in the X3L format for 3D viewing and analysis.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// The sliders in the Graphics area tune the following properties:
	/// <list type="table">
	/// <listheader><term>Slider</term><description>Property</description></listheader>
	/// <item><term>Alpha</term><description>the opaqueness of lines and points. 0 means transparent, 1 means opaque.</description></item>
	/// <item><term>Lines</term><description>the thickness of lines.</description></item>
	/// <item><term>Points</term><description>the size of points.</description></item>
	/// </list>
	/// </para>
	/// <para>
	/// A double click with the left mouse button near a graphical element (line or point) opens the corresponding browser (<see cref="SySal.Executables.EasyReconstruct.TrackBrowser">TrackBrowser</see> for tracks, <see cref="SySal.Executables.EasyReconstruct.VertexBrowser">VertexBrowser</see> for vertices).
	/// </para>
	/// </remarks>
	public class DisplayForm : System.Windows.Forms.Form, IPSelector, TrackSelector, IDecaySearchAutomation
	{
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Button cmdDefault;
		private System.Windows.Forms.Button cmdZoomMinus;
		private System.Windows.Forms.Button cmdZoomPlus;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.RadioButton radRotation;
		private System.Windows.Forms.RadioButton radPanning;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox txtMinSegments;
		private System.Windows.Forms.Button cmdPlot;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.RadioButton radVtx;
		private System.Windows.Forms.RadioButton radSeg;
		private System.Windows.Forms.RadioButton radTrk;
		public SySal.TotalScan.Track[] mTracks;		
		public double B_MaxX, B_MinX, B_MaxY, B_MinY, B_MaxZ, B_MinZ;
		public int MinimumSegmentsNumber;
		public int MinimumTracksNumber;
		public int GraphStartTrackId;
		public int GraphStartVtxId;
		private System.Windows.Forms.CheckBox chkShowSegments;
		private System.Windows.Forms.TextBox txtMinTracks;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.RadioButton radioTrackGraph;
		private System.Windows.Forms.TextBox textTrackGraphStart;
		private System.Windows.Forms.TextBox textVtxGraphStart;
		private System.Windows.Forms.RadioButton radioVtxGraph;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.TrackBar AlphaTrackBar;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.TrackBar LinesTrackBar;
		private System.Windows.Forms.Label label16;
		private System.Windows.Forms.TrackBar PointsTrackBar;		
		private System.Windows.Forms.CheckBox checkIsometric;
		private System.Windows.Forms.Button cmdBkgnd;
		private System.Windows.Forms.ColorDialog colorDialog1;
        private Button cmdClear;
        private CheckBox checkTrackVtxColor;
		private System.Windows.Forms.Button SaveBtn;
        private GroupBox groupBox4;
        private TextBox textDIP;
        private Label label5;
        private TextBox textNDIP;
        private Label label6;
        private TextBox textSecondTrackVtx;
        private Label label4;
        private TextBox textIPFirstTrack;
        private Label label3;
        private TextBox txtPThruDown;
        private Label label8;
        private TextBox txtPThruUp;
        private Label label7;
        private Label label10;
        private TextBox txtSlopeTol;
        private TextBox txtSlopeY;
        private TextBox txtSlopeX;
        private Label label9;
        private Label label12;
        private TextBox txtPosTol;
        private TextBox txtPosY;
        private TextBox txtPosX;
        private Label label11;
        private Label label17;
        private Button cmdPlateFrame;
        private Button cmdPlateColor;

        SySal.TotalScan.Volume.LayerList m_Layers;
        private Button cmdRot;
        private TextBox txtYRot;
        private Label label19;
        private TextBox txtXRot;
        private Label label18;
        private RadioButton radAllSegs;
        private Button btn400;
        private Button btn600;
        private Button btn800;
        private Button btn1000;
        private RadioButton radioTaggedSegs;
        private Button cmdTagColor;
        private RadioButton radioUntaggedSegs;
        private Button cmdSetFocus;
        private Button cmdVertexBrowser;
        private Button cmdTrackBrowser;
        private Button cmdFont;
        private FontDialog fontDialog1;
        private Button cmdVertexFit;
        private TextBox txtVertexFitName;
        private CheckedListBox clDataSets;
        SySal.BasicTypes.Cuboid m_Extents;
        private TextBox txtSegExtend;
        private Label label20;
        private Button cmdSaveToTSR;
        private TextBox txtAttrSearch;
        private ComboBox cmbAttrFilter;
        private Button cmdRemoveByOwner;

        private SySal.TotalScan.Flexi.Volume m_V;
        private Button cmdExportToOperaFeedback;
        private Button btnFilterVars;
        private Button cmdShowGlobalData;
        private Button btnDecaySearchAssistant;
        private CheckBox chkGraphDownstream;
        private CheckBox chkGraphUpstream;
        private Button cmdRecStart;
        private Button cmdRecStop;
        private TextBox txtMovieKB;
        private Label label21;
        private Button cmdMovie;
        private ComboBox cmbFilter;
        private Button btnFilterDel;
        private Button btnFilterAdd;

        long m_SelectedEvent;

        public long SelectedEvent
        {
            get { return m_SelectedEvent; }
            set 
            { 
                m_SelectedEvent = value;
                if (MVForm != null)
                {
                    MVForm.Close();
                    MVForm = null;
                }
                MVForm = new MovieForm(gdiDisplay1, m_SelectedEvent, (int)m_V.Layers[m_V.Layers.Length - 1].BrickId);                
            }
        }

		public DisplayForm(SySal.TotalScan.Flexi.Volume v)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

            DSAForm = new DecaySearchAssistantForm(this);            
            m_V = v;
            SySal.TotalScan.Volume.LayerList ll = m_V.Layers;
            m_Layers = ll;
            int i, j;
            for (i = 0; i < ll.Length; i++)
            {
                SySal.TotalScan.Layer lay = ll[i];
                if (lay.Length == 0) continue;                
                SySal.BasicTypes.Vector info = lay[0].Info.Intercept;
                m_Extents.MinX = m_Extents.MaxX = info.X;
                m_Extents.MinY = m_Extents.MaxY = info.Y;
                m_Extents.MinZ = lay.UpstreamZ;
                m_Extents.MaxZ = lay.DownstreamZ;
            }
            for (i = 0; i < ll.Length; i++)
            {
                SySal.TotalScan.Layer lay = ll[i];
                for (j = 0; j < lay.Length; j++)
                {
                    SySal.BasicTypes.Vector info = lay[j].Info.Intercept;
                    if (m_Extents.MinX > info.X) m_Extents.MinX = info.X;
                    else if (m_Extents.MaxX < info.X) m_Extents.MaxX = info.X;
                    if (m_Extents.MinY > info.Y) m_Extents.MinY = info.Y;
                    else if (m_Extents.MaxY < info.Y) m_Extents.MaxY = info.Y;
                    if (lay.UpstreamZ < m_Extents.MinZ) m_Extents.MinZ = lay.UpstreamZ;
                    if (lay.DownstreamZ > m_Extents.MaxZ) m_Extents.MaxZ = lay.DownstreamZ;
                    m_Extents.MinZ = lay.UpstreamZ;
                    m_Extents.MaxZ = lay.DownstreamZ;
                }
            }
            m_Extents.MinX -= 500.0;
            m_Extents.MinY -= 500.0;
            m_Extents.MinZ -= 500.0;
            m_Extents.MaxX += 500.0;
            m_Extents.MaxY += 500.0;
            m_Extents.MaxZ += 500.0;
            PosX = ll[ll.Length / 2].RefCenter.X;
            PosY = ll[ll.Length / 2].RefCenter.Y;
			textTrackGraphStart.Text = GraphStartTrackId.ToString();
			textVtxGraphStart.Text = GraphStartVtxId.ToString();
            m_Panel = new GDIPanel(this);
            gdiDisplay1 = m_Panel.gdiDisplay1;
            gdiDisplay1.DoubleClickSelect = new GDI3D.Control.SelectObject(OnSelectObject);
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.cmdMovie = new System.Windows.Forms.Button();
            this.txtMovieKB = new System.Windows.Forms.TextBox();
            this.label21 = new System.Windows.Forms.Label();
            this.cmdRecStop = new System.Windows.Forms.Button();
            this.cmdRecStart = new System.Windows.Forms.Button();
            this.cmdSetFocus = new System.Windows.Forms.Button();
            this.cmdRot = new System.Windows.Forms.Button();
            this.txtYRot = new System.Windows.Forms.TextBox();
            this.label19 = new System.Windows.Forms.Label();
            this.txtXRot = new System.Windows.Forms.TextBox();
            this.label18 = new System.Windows.Forms.Label();
            this.cmdPlateColor = new System.Windows.Forms.Button();
            this.cmdPlateFrame = new System.Windows.Forms.Button();
            this.cmdBkgnd = new System.Windows.Forms.Button();
            this.checkIsometric = new System.Windows.Forms.CheckBox();
            this.radPanning = new System.Windows.Forms.RadioButton();
            this.radRotation = new System.Windows.Forms.RadioButton();
            this.label1 = new System.Windows.Forms.Label();
            this.cmdZoomMinus = new System.Windows.Forms.Button();
            this.cmdZoomPlus = new System.Windows.Forms.Button();
            this.cmdDefault = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.chkGraphUpstream = new System.Windows.Forms.CheckBox();
            this.chkGraphDownstream = new System.Windows.Forms.CheckBox();
            this.btnFilterVars = new System.Windows.Forms.Button();
            this.txtAttrSearch = new System.Windows.Forms.TextBox();
            this.cmbAttrFilter = new System.Windows.Forms.ComboBox();
            this.txtSegExtend = new System.Windows.Forms.TextBox();
            this.clDataSets = new System.Windows.Forms.CheckedListBox();
            this.label20 = new System.Windows.Forms.Label();
            this.cmdVertexBrowser = new System.Windows.Forms.Button();
            this.radioUntaggedSegs = new System.Windows.Forms.RadioButton();
            this.cmdTagColor = new System.Windows.Forms.Button();
            this.radioTaggedSegs = new System.Windows.Forms.RadioButton();
            this.radAllSegs = new System.Windows.Forms.RadioButton();
            this.label17 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.txtPosTol = new System.Windows.Forms.TextBox();
            this.txtPosY = new System.Windows.Forms.TextBox();
            this.txtPosX = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.txtSlopeTol = new System.Windows.Forms.TextBox();
            this.txtSlopeY = new System.Windows.Forms.TextBox();
            this.txtSlopeX = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.txtPThruDown = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.txtPThruUp = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.cmdClear = new System.Windows.Forms.Button();
            this.checkTrackVtxColor = new System.Windows.Forms.CheckBox();
            this.SaveBtn = new System.Windows.Forms.Button();
            this.textVtxGraphStart = new System.Windows.Forms.TextBox();
            this.radioVtxGraph = new System.Windows.Forms.RadioButton();
            this.textTrackGraphStart = new System.Windows.Forms.TextBox();
            this.radioTrackGraph = new System.Windows.Forms.RadioButton();
            this.txtMinTracks = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.chkShowSegments = new System.Windows.Forms.CheckBox();
            this.cmdPlot = new System.Windows.Forms.Button();
            this.txtMinSegments = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.radVtx = new System.Windows.Forms.RadioButton();
            this.radSeg = new System.Windows.Forms.RadioButton();
            this.radTrk = new System.Windows.Forms.RadioButton();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.cmdFont = new System.Windows.Forms.Button();
            this.label16 = new System.Windows.Forms.Label();
            this.PointsTrackBar = new System.Windows.Forms.TrackBar();
            this.label15 = new System.Windows.Forms.Label();
            this.LinesTrackBar = new System.Windows.Forms.TrackBar();
            this.label14 = new System.Windows.Forms.Label();
            this.AlphaTrackBar = new System.Windows.Forms.TrackBar();
            this.colorDialog1 = new System.Windows.Forms.ColorDialog();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.txtVertexFitName = new System.Windows.Forms.TextBox();
            this.cmdVertexFit = new System.Windows.Forms.Button();
            this.textDIP = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.textNDIP = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.textSecondTrackVtx = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.textIPFirstTrack = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.btn400 = new System.Windows.Forms.Button();
            this.btn600 = new System.Windows.Forms.Button();
            this.btn800 = new System.Windows.Forms.Button();
            this.btn1000 = new System.Windows.Forms.Button();
            this.cmdTrackBrowser = new System.Windows.Forms.Button();
            this.fontDialog1 = new System.Windows.Forms.FontDialog();
            this.cmdSaveToTSR = new System.Windows.Forms.Button();
            this.cmdRemoveByOwner = new System.Windows.Forms.Button();
            this.cmdExportToOperaFeedback = new System.Windows.Forms.Button();
            this.cmdShowGlobalData = new System.Windows.Forms.Button();
            this.btnDecaySearchAssistant = new System.Windows.Forms.Button();
            this.cmbFilter = new System.Windows.Forms.ComboBox();
            this.btnFilterAdd = new System.Windows.Forms.Button();
            this.btnFilterDel = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.PointsTrackBar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.LinesTrackBar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.AlphaTrackBar)).BeginInit();
            this.groupBox4.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cmdMovie);
            this.groupBox1.Controls.Add(this.txtMovieKB);
            this.groupBox1.Controls.Add(this.label21);
            this.groupBox1.Controls.Add(this.cmdRecStop);
            this.groupBox1.Controls.Add(this.cmdRecStart);
            this.groupBox1.Controls.Add(this.cmdSetFocus);
            this.groupBox1.Controls.Add(this.cmdRot);
            this.groupBox1.Controls.Add(this.txtYRot);
            this.groupBox1.Controls.Add(this.label19);
            this.groupBox1.Controls.Add(this.txtXRot);
            this.groupBox1.Controls.Add(this.label18);
            this.groupBox1.Controls.Add(this.cmdPlateColor);
            this.groupBox1.Controls.Add(this.cmdPlateFrame);
            this.groupBox1.Controls.Add(this.cmdBkgnd);
            this.groupBox1.Controls.Add(this.checkIsometric);
            this.groupBox1.Controls.Add(this.radPanning);
            this.groupBox1.Controls.Add(this.radRotation);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.cmdZoomMinus);
            this.groupBox1.Controls.Add(this.cmdZoomPlus);
            this.groupBox1.Controls.Add(this.cmdDefault);
            this.groupBox1.Location = new System.Drawing.Point(215, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(201, 194);
            this.groupBox1.TabIndex = 4;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Display";
            // 
            // cmdMovie
            // 
            this.cmdMovie.Location = new System.Drawing.Point(134, 16);
            this.cmdMovie.Name = "cmdMovie";
            this.cmdMovie.Size = new System.Drawing.Size(61, 20);
            this.cmdMovie.TabIndex = 53;
            this.cmdMovie.Text = "Movie ";
            this.cmdMovie.Click += new System.EventHandler(this.cmdMovie_Click);
            // 
            // txtMovieKB
            // 
            this.txtMovieKB.Location = new System.Drawing.Point(150, 77);
            this.txtMovieKB.Name = "txtMovieKB";
            this.txtMovieKB.Size = new System.Drawing.Size(44, 20);
            this.txtMovieKB.TabIndex = 52;
            this.txtMovieKB.Leave += new System.EventHandler(this.OnMovieKBLeave);
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(118, 81);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(24, 13);
            this.label21.TabIndex = 51;
            this.label21.Text = "KB:";
            // 
            // cmdRecStop
            // 
            this.cmdRecStop.Location = new System.Drawing.Point(134, 40);
            this.cmdRecStop.Name = "cmdRecStop";
            this.cmdRecStop.Size = new System.Drawing.Size(61, 20);
            this.cmdRecStop.TabIndex = 52;
            this.cmdRecStop.Text = "RecStop";
            this.cmdRecStop.Visible = false;
            this.cmdRecStop.Click += new System.EventHandler(this.cmdRecStop_Click);
            // 
            // cmdRecStart
            // 
            this.cmdRecStart.Location = new System.Drawing.Point(134, 16);
            this.cmdRecStart.Name = "cmdRecStart";
            this.cmdRecStart.Size = new System.Drawing.Size(61, 20);
            this.cmdRecStart.TabIndex = 51;
            this.cmdRecStart.Text = "RecStart";
            this.cmdRecStart.Visible = false;
            this.cmdRecStart.Click += new System.EventHandler(this.cmdRecStart_Click);
            // 
            // cmdSetFocus
            // 
            this.cmdSetFocus.Location = new System.Drawing.Point(80, 102);
            this.cmdSetFocus.Name = "cmdSetFocus";
            this.cmdSetFocus.Size = new System.Drawing.Size(115, 22);
            this.cmdSetFocus.TabIndex = 35;
            this.cmdSetFocus.Text = "Set Focus";
            this.cmdSetFocus.Click += new System.EventHandler(this.cmdSetFocus_Click);
            // 
            // cmdRot
            // 
            this.cmdRot.Location = new System.Drawing.Point(140, 138);
            this.cmdRot.Name = "cmdRot";
            this.cmdRot.Size = new System.Drawing.Size(55, 20);
            this.cmdRot.TabIndex = 18;
            this.cmdRot.Text = "Rot";
            this.cmdRot.Click += new System.EventHandler(this.cmdRot_Click);
            // 
            // txtYRot
            // 
            this.txtYRot.Location = new System.Drawing.Point(150, 165);
            this.txtYRot.Name = "txtYRot";
            this.txtYRot.Size = new System.Drawing.Size(45, 20);
            this.txtYRot.TabIndex = 17;
            this.txtYRot.Leave += new System.EventHandler(this.txtYRot_Leave);
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(102, 167);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(42, 13);
            this.label19.TabIndex = 16;
            this.label19.Text = "Y rot (°)";
            // 
            // txtXRot
            // 
            this.txtXRot.Location = new System.Drawing.Point(54, 164);
            this.txtXRot.Name = "txtXRot";
            this.txtXRot.Size = new System.Drawing.Size(45, 20);
            this.txtXRot.TabIndex = 15;
            this.txtXRot.Leave += new System.EventHandler(this.txtXRot_Leave);
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(8, 167);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(42, 13);
            this.label18.TabIndex = 14;
            this.label18.Text = "X rot (°)";
            // 
            // cmdPlateColor
            // 
            this.cmdPlateColor.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(255)))), ((int)(((byte)(255)))));
            this.cmdPlateColor.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.cmdPlateColor.Location = new System.Drawing.Point(105, 36);
            this.cmdPlateColor.Name = "cmdPlateColor";
            this.cmdPlateColor.Size = new System.Drawing.Size(24, 22);
            this.cmdPlateColor.TabIndex = 11;
            this.cmdPlateColor.UseVisualStyleBackColor = false;
            this.cmdPlateColor.Click += new System.EventHandler(this.cmdPlateColor_Click);
            // 
            // cmdPlateFrame
            // 
            this.cmdPlateFrame.Location = new System.Drawing.Point(8, 39);
            this.cmdPlateFrame.Name = "cmdPlateFrame";
            this.cmdPlateFrame.Size = new System.Drawing.Size(94, 22);
            this.cmdPlateFrame.TabIndex = 9;
            this.cmdPlateFrame.Text = "+ Plate Frame";
            this.cmdPlateFrame.Click += new System.EventHandler(this.cmdPlateFrame_Click);
            // 
            // cmdBkgnd
            // 
            this.cmdBkgnd.Location = new System.Drawing.Point(80, 138);
            this.cmdBkgnd.Name = "cmdBkgnd";
            this.cmdBkgnd.Size = new System.Drawing.Size(55, 20);
            this.cmdBkgnd.TabIndex = 8;
            this.cmdBkgnd.Text = "Bkgnd";
            this.cmdBkgnd.Click += new System.EventHandler(this.cmdBkgnd_Click);
            // 
            // checkIsometric
            // 
            this.checkIsometric.Location = new System.Drawing.Point(13, 103);
            this.checkIsometric.Name = "checkIsometric";
            this.checkIsometric.Size = new System.Drawing.Size(72, 16);
            this.checkIsometric.TabIndex = 7;
            this.checkIsometric.Text = "Isometric";
            this.checkIsometric.CheckedChanged += new System.EventHandler(this.checkIsometric_CheckedChanged);
            // 
            // radPanning
            // 
            this.radPanning.Location = new System.Drawing.Point(8, 142);
            this.radPanning.Name = "radPanning";
            this.radPanning.Size = new System.Drawing.Size(64, 16);
            this.radPanning.TabIndex = 4;
            this.radPanning.Text = "Pan";
            this.radPanning.CheckedChanged += new System.EventHandler(this.radPanning_CheckedChanged);
            // 
            // radRotation
            // 
            this.radRotation.Checked = true;
            this.radRotation.Location = new System.Drawing.Point(8, 123);
            this.radRotation.Name = "radRotation";
            this.radRotation.Size = new System.Drawing.Size(64, 16);
            this.radRotation.TabIndex = 3;
            this.radRotation.TabStop = true;
            this.radRotation.Text = "Rotate";
            this.radRotation.CheckedChanged += new System.EventHandler(this.radRotation_CheckedChanged);
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(10, 79);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(40, 16);
            this.label1.TabIndex = 6;
            this.label1.Text = "Zoom:";
            // 
            // cmdZoomMinus
            // 
            this.cmdZoomMinus.Location = new System.Drawing.Point(80, 77);
            this.cmdZoomMinus.Name = "cmdZoomMinus";
            this.cmdZoomMinus.Size = new System.Drawing.Size(24, 20);
            this.cmdZoomMinus.TabIndex = 2;
            this.cmdZoomMinus.Text = "-";
            this.cmdZoomMinus.Click += new System.EventHandler(this.cmdZoomMinus_Click);
            // 
            // cmdZoomPlus
            // 
            this.cmdZoomPlus.Location = new System.Drawing.Point(56, 77);
            this.cmdZoomPlus.Name = "cmdZoomPlus";
            this.cmdZoomPlus.Size = new System.Drawing.Size(24, 20);
            this.cmdZoomPlus.TabIndex = 1;
            this.cmdZoomPlus.Text = "+";
            this.cmdZoomPlus.Click += new System.EventHandler(this.cmdZoomPlus_Click);
            // 
            // cmdDefault
            // 
            this.cmdDefault.Location = new System.Drawing.Point(8, 16);
            this.cmdDefault.Name = "cmdDefault";
            this.cmdDefault.Size = new System.Drawing.Size(94, 20);
            this.cmdDefault.TabIndex = 0;
            this.cmdDefault.Text = "Default View";
            this.cmdDefault.Click += new System.EventHandler(this.cmdDefault_Click);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.btnFilterDel);
            this.groupBox2.Controls.Add(this.btnFilterAdd);
            this.groupBox2.Controls.Add(this.cmbFilter);
            this.groupBox2.Controls.Add(this.chkGraphUpstream);
            this.groupBox2.Controls.Add(this.chkGraphDownstream);
            this.groupBox2.Controls.Add(this.btnFilterVars);
            this.groupBox2.Controls.Add(this.txtAttrSearch);
            this.groupBox2.Controls.Add(this.cmbAttrFilter);
            this.groupBox2.Controls.Add(this.txtSegExtend);
            this.groupBox2.Controls.Add(this.clDataSets);
            this.groupBox2.Controls.Add(this.label20);
            this.groupBox2.Controls.Add(this.cmdVertexBrowser);
            this.groupBox2.Controls.Add(this.radioUntaggedSegs);
            this.groupBox2.Controls.Add(this.cmdTagColor);
            this.groupBox2.Controls.Add(this.radioTaggedSegs);
            this.groupBox2.Controls.Add(this.radAllSegs);
            this.groupBox2.Controls.Add(this.label17);
            this.groupBox2.Controls.Add(this.label12);
            this.groupBox2.Controls.Add(this.txtPosTol);
            this.groupBox2.Controls.Add(this.txtPosY);
            this.groupBox2.Controls.Add(this.txtPosX);
            this.groupBox2.Controls.Add(this.label11);
            this.groupBox2.Controls.Add(this.label10);
            this.groupBox2.Controls.Add(this.txtSlopeTol);
            this.groupBox2.Controls.Add(this.txtSlopeY);
            this.groupBox2.Controls.Add(this.txtSlopeX);
            this.groupBox2.Controls.Add(this.label9);
            this.groupBox2.Controls.Add(this.txtPThruDown);
            this.groupBox2.Controls.Add(this.label8);
            this.groupBox2.Controls.Add(this.txtPThruUp);
            this.groupBox2.Controls.Add(this.label7);
            this.groupBox2.Controls.Add(this.cmdClear);
            this.groupBox2.Controls.Add(this.checkTrackVtxColor);
            this.groupBox2.Controls.Add(this.SaveBtn);
            this.groupBox2.Controls.Add(this.textVtxGraphStart);
            this.groupBox2.Controls.Add(this.radioVtxGraph);
            this.groupBox2.Controls.Add(this.textTrackGraphStart);
            this.groupBox2.Controls.Add(this.radioTrackGraph);
            this.groupBox2.Controls.Add(this.txtMinTracks);
            this.groupBox2.Controls.Add(this.label13);
            this.groupBox2.Controls.Add(this.chkShowSegments);
            this.groupBox2.Controls.Add(this.cmdPlot);
            this.groupBox2.Controls.Add(this.txtMinSegments);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.radVtx);
            this.groupBox2.Controls.Add(this.radSeg);
            this.groupBox2.Controls.Add(this.radTrk);
            this.groupBox2.Location = new System.Drawing.Point(8, 12);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(201, 611);
            this.groupBox2.TabIndex = 5;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Selection";
            // 
            // chkGraphUpstream
            // 
            this.chkGraphUpstream.AutoSize = true;
            this.chkGraphUpstream.Location = new System.Drawing.Point(103, 207);
            this.chkGraphUpstream.Name = "chkGraphUpstream";
            this.chkGraphUpstream.Size = new System.Drawing.Size(71, 17);
            this.chkGraphUpstream.TabIndex = 52;
            this.chkGraphUpstream.Text = "Upstream";
            this.chkGraphUpstream.UseVisualStyleBackColor = true;
            // 
            // chkGraphDownstream
            // 
            this.chkGraphDownstream.AutoSize = true;
            this.chkGraphDownstream.Checked = true;
            this.chkGraphDownstream.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkGraphDownstream.Location = new System.Drawing.Point(8, 207);
            this.chkGraphDownstream.Name = "chkGraphDownstream";
            this.chkGraphDownstream.Size = new System.Drawing.Size(85, 17);
            this.chkGraphDownstream.TabIndex = 51;
            this.chkGraphDownstream.Text = "Downstream";
            this.chkGraphDownstream.UseVisualStyleBackColor = true;
            // 
            // btnFilterVars
            // 
            this.btnFilterVars.Location = new System.Drawing.Point(6, 226);
            this.btnFilterVars.Name = "btnFilterVars";
            this.btnFilterVars.Size = new System.Drawing.Size(89, 24);
            this.btnFilterVars.TabIndex = 36;
            this.btnFilterVars.Text = "Filter Vars";
            this.btnFilterVars.UseVisualStyleBackColor = true;
            this.btnFilterVars.Click += new System.EventHandler(this.btnFilterVars_Click);
            // 
            // txtAttrSearch
            // 
            this.txtAttrSearch.Location = new System.Drawing.Point(101, 278);
            this.txtAttrSearch.Name = "txtAttrSearch";
            this.txtAttrSearch.Size = new System.Drawing.Size(91, 20);
            this.txtAttrSearch.TabIndex = 49;
            // 
            // cmbAttrFilter
            // 
            this.cmbAttrFilter.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbAttrFilter.FormattingEnabled = true;
            this.cmbAttrFilter.Items.AddRange(new object[] {
            "With attribute",
            "Without attribute "});
            this.cmbAttrFilter.Location = new System.Drawing.Point(6, 277);
            this.cmbAttrFilter.Name = "cmbAttrFilter";
            this.cmbAttrFilter.Size = new System.Drawing.Size(89, 21);
            this.cmbAttrFilter.TabIndex = 48;
            // 
            // txtSegExtend
            // 
            this.txtSegExtend.Location = new System.Drawing.Point(147, 19);
            this.txtSegExtend.Name = "txtSegExtend";
            this.txtSegExtend.Size = new System.Drawing.Size(44, 20);
            this.txtSegExtend.TabIndex = 47;
            this.txtSegExtend.Leave += new System.EventHandler(this.OnSegExtendLeave);
            // 
            // clDataSets
            // 
            this.clDataSets.CheckOnClick = true;
            this.clDataSets.FormattingEnabled = true;
            this.clDataSets.Location = new System.Drawing.Point(7, 549);
            this.clDataSets.Name = "clDataSets";
            this.clDataSets.Size = new System.Drawing.Size(185, 49);
            this.clDataSets.TabIndex = 47;
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(98, 23);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(43, 13);
            this.label20.TabIndex = 46;
            this.label20.Text = "Extend:";
            // 
            // cmdVertexBrowser
            // 
            this.cmdVertexBrowser.Location = new System.Drawing.Point(166, 185);
            this.cmdVertexBrowser.Name = "cmdVertexBrowser";
            this.cmdVertexBrowser.Size = new System.Drawing.Size(28, 19);
            this.cmdVertexBrowser.TabIndex = 46;
            this.cmdVertexBrowser.Text = ">";
            this.cmdVertexBrowser.Click += new System.EventHandler(this.cmdVertexBrowser_Click);
            // 
            // radioUntaggedSegs
            // 
            this.radioUntaggedSegs.Location = new System.Drawing.Point(8, 66);
            this.radioUntaggedSegs.Name = "radioUntaggedSegs";
            this.radioUntaggedSegs.Size = new System.Drawing.Size(127, 24);
            this.radioUntaggedSegs.TabIndex = 3;
            this.radioUntaggedSegs.Text = "Untagged Segments";
            // 
            // cmdTagColor
            // 
            this.cmdTagColor.BackColor = System.Drawing.Color.White;
            this.cmdTagColor.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.cmdTagColor.Location = new System.Drawing.Point(137, 43);
            this.cmdTagColor.Name = "cmdTagColor";
            this.cmdTagColor.Size = new System.Drawing.Size(24, 22);
            this.cmdTagColor.TabIndex = 12;
            this.cmdTagColor.UseVisualStyleBackColor = false;
            this.cmdTagColor.Click += new System.EventHandler(this.OnTagColorClick);
            // 
            // radioTaggedSegs
            // 
            this.radioTaggedSegs.Location = new System.Drawing.Point(8, 42);
            this.radioTaggedSegs.Name = "radioTaggedSegs";
            this.radioTaggedSegs.Size = new System.Drawing.Size(127, 24);
            this.radioTaggedSegs.TabIndex = 2;
            this.radioTaggedSegs.Text = "Tagged Segments";
            // 
            // radAllSegs
            // 
            this.radAllSegs.Location = new System.Drawing.Point(8, 16);
            this.radAllSegs.Name = "radAllSegs";
            this.radAllSegs.Size = new System.Drawing.Size(80, 26);
            this.radAllSegs.TabIndex = 1;
            this.radAllSegs.Text = "Segments";
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(4, 460);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(71, 13);
            this.label17.TabIndex = 44;
            this.label17.Text = "(Central layer)";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(78, 461);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(22, 13);
            this.label12.TabIndex = 43;
            this.label12.Text = "Tol";
            // 
            // txtPosTol
            // 
            this.txtPosTol.Location = new System.Drawing.Point(105, 458);
            this.txtPosTol.Name = "txtPosTol";
            this.txtPosTol.Size = new System.Drawing.Size(51, 20);
            this.txtPosTol.TabIndex = 27;
            this.txtPosTol.Leave += new System.EventHandler(this.OnPosTolLeave);
            // 
            // txtPosY
            // 
            this.txtPosY.Location = new System.Drawing.Point(101, 432);
            this.txtPosY.Name = "txtPosY";
            this.txtPosY.Size = new System.Drawing.Size(55, 20);
            this.txtPosY.TabIndex = 26;
            this.txtPosY.Leave += new System.EventHandler(this.OnPosYLeave);
            // 
            // txtPosX
            // 
            this.txtPosX.Location = new System.Drawing.Point(40, 432);
            this.txtPosX.Name = "txtPosX";
            this.txtPosX.Size = new System.Drawing.Size(55, 20);
            this.txtPosX.TabIndex = 25;
            this.txtPosX.Leave += new System.EventHandler(this.OnPosXLeave);
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(3, 435);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(25, 13);
            this.label11.TabIndex = 40;
            this.label11.Text = "Pos";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(117, 407);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(22, 13);
            this.label10.TabIndex = 38;
            this.label10.Text = "Tol";
            // 
            // txtSlopeTol
            // 
            this.txtSlopeTol.Location = new System.Drawing.Point(143, 404);
            this.txtSlopeTol.Name = "txtSlopeTol";
            this.txtSlopeTol.Size = new System.Drawing.Size(32, 20);
            this.txtSlopeTol.TabIndex = 24;
            this.txtSlopeTol.Leave += new System.EventHandler(this.OnSlopeTolLeave);
            // 
            // txtSlopeY
            // 
            this.txtSlopeY.Location = new System.Drawing.Point(78, 404);
            this.txtSlopeY.Name = "txtSlopeY";
            this.txtSlopeY.Size = new System.Drawing.Size(38, 20);
            this.txtSlopeY.TabIndex = 23;
            this.txtSlopeY.Leave += new System.EventHandler(this.OnSlopeYLeave);
            // 
            // txtSlopeX
            // 
            this.txtSlopeX.Location = new System.Drawing.Point(40, 404);
            this.txtSlopeX.Name = "txtSlopeX";
            this.txtSlopeX.Size = new System.Drawing.Size(38, 20);
            this.txtSlopeX.TabIndex = 22;
            this.txtSlopeX.Leave += new System.EventHandler(this.OnSlopeXLeave);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(3, 407);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(34, 13);
            this.label9.TabIndex = 21;
            this.label9.Text = "Slope";
            // 
            // txtPThruDown
            // 
            this.txtPThruDown.Location = new System.Drawing.Point(132, 379);
            this.txtPThruDown.Name = "txtPThruDown";
            this.txtPThruDown.Size = new System.Drawing.Size(27, 20);
            this.txtPThruDown.TabIndex = 20;
            this.txtPThruDown.Leave += new System.EventHandler(this.OnMaxPThruDownLeave);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(3, 382);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(117, 13);
            this.label8.TabIndex = 19;
            this.label8.Text = "Max downstream layers";
            // 
            // txtPThruUp
            // 
            this.txtPThruUp.Location = new System.Drawing.Point(132, 355);
            this.txtPThruUp.Name = "txtPThruUp";
            this.txtPThruUp.Size = new System.Drawing.Size(27, 20);
            this.txtPThruUp.TabIndex = 18;
            this.txtPThruUp.Leave += new System.EventHandler(this.OnMaxPThruUpLeave);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(3, 358);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(103, 13);
            this.label7.TabIndex = 17;
            this.label7.Text = "Max upstream layers";
            // 
            // cmdClear
            // 
            this.cmdClear.Location = new System.Drawing.Point(86, 491);
            this.cmdClear.Name = "cmdClear";
            this.cmdClear.Size = new System.Drawing.Size(72, 24);
            this.cmdClear.TabIndex = 29;
            this.cmdClear.Text = "Clear";
            this.cmdClear.Click += new System.EventHandler(this.cmdClear_Click);
            // 
            // checkTrackVtxColor
            // 
            this.checkTrackVtxColor.AutoSize = true;
            this.checkTrackVtxColor.Location = new System.Drawing.Point(87, 526);
            this.checkTrackVtxColor.Name = "checkTrackVtxColor";
            this.checkTrackVtxColor.Size = new System.Drawing.Size(63, 17);
            this.checkTrackVtxColor.TabIndex = 31;
            this.checkTrackVtxColor.Text = "Colorize";
            this.checkTrackVtxColor.UseVisualStyleBackColor = true;
            // 
            // SaveBtn
            // 
            this.SaveBtn.Location = new System.Drawing.Point(7, 521);
            this.SaveBtn.Name = "SaveBtn";
            this.SaveBtn.Size = new System.Drawing.Size(72, 24);
            this.SaveBtn.TabIndex = 30;
            this.SaveBtn.Text = "Save";
            this.SaveBtn.Click += new System.EventHandler(this.SaveButton_Click);
            // 
            // textVtxGraphStart
            // 
            this.textVtxGraphStart.Location = new System.Drawing.Point(120, 184);
            this.textVtxGraphStart.Name = "textVtxGraphStart";
            this.textVtxGraphStart.Size = new System.Drawing.Size(40, 20);
            this.textVtxGraphStart.TabIndex = 10;
            this.textVtxGraphStart.Leave += new System.EventHandler(this.OnVtxGraphStartLeave);
            // 
            // radioVtxGraph
            // 
            this.radioVtxGraph.Location = new System.Drawing.Point(8, 186);
            this.radioVtxGraph.Name = "radioVtxGraph";
            this.radioVtxGraph.Size = new System.Drawing.Size(112, 16);
            this.radioVtxGraph.TabIndex = 9;
            this.radioVtxGraph.Text = "Graph from Vtx#";
            // 
            // textTrackGraphStart
            // 
            this.textTrackGraphStart.Location = new System.Drawing.Point(120, 160);
            this.textTrackGraphStart.Name = "textTrackGraphStart";
            this.textTrackGraphStart.Size = new System.Drawing.Size(40, 20);
            this.textTrackGraphStart.TabIndex = 8;
            this.textTrackGraphStart.Leave += new System.EventHandler(this.OnTrackGraphStartLeave);
            // 
            // radioTrackGraph
            // 
            this.radioTrackGraph.Location = new System.Drawing.Point(8, 162);
            this.radioTrackGraph.Name = "radioTrackGraph";
            this.radioTrackGraph.Size = new System.Drawing.Size(112, 16);
            this.radioTrackGraph.TabIndex = 7;
            this.radioTrackGraph.Text = "Graph from Tk#";
            // 
            // txtMinTracks
            // 
            this.txtMinTracks.Location = new System.Drawing.Point(82, 328);
            this.txtMinTracks.Name = "txtMinTracks";
            this.txtMinTracks.Size = new System.Drawing.Size(27, 20);
            this.txtMinTracks.TabIndex = 16;
            this.txtMinTracks.TextChanged += new System.EventHandler(this.txtMinTracks_TextChanged);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(4, 330);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(63, 13);
            this.label13.TabIndex = 15;
            this.label13.Text = "Min Tracks:";
            // 
            // chkShowSegments
            // 
            this.chkShowSegments.Location = new System.Drawing.Point(89, 138);
            this.chkShowSegments.Name = "chkShowSegments";
            this.chkShowSegments.Size = new System.Drawing.Size(105, 16);
            this.chkShowSegments.TabIndex = 11;
            this.chkShowSegments.Text = "Show segments";
            // 
            // cmdPlot
            // 
            this.cmdPlot.Location = new System.Drawing.Point(7, 491);
            this.cmdPlot.Name = "cmdPlot";
            this.cmdPlot.Size = new System.Drawing.Size(72, 24);
            this.cmdPlot.TabIndex = 28;
            this.cmdPlot.Text = "Plot";
            this.cmdPlot.Click += new System.EventHandler(this.cmdPlot_Click);
            // 
            // txtMinSegments
            // 
            this.txtMinSegments.Location = new System.Drawing.Point(82, 304);
            this.txtMinSegments.Name = "txtMinSegments";
            this.txtMinSegments.Size = new System.Drawing.Size(27, 20);
            this.txtMinSegments.TabIndex = 14;
            this.txtMinSegments.TextChanged += new System.EventHandler(this.txtMinSegments_TextChanged);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(4, 308);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(77, 13);
            this.label2.TabIndex = 13;
            this.label2.Text = "Min Segments:";
            // 
            // radVtx
            // 
            this.radVtx.Checked = true;
            this.radVtx.Location = new System.Drawing.Point(8, 140);
            this.radVtx.Name = "radVtx";
            this.radVtx.Size = new System.Drawing.Size(96, 16);
            this.radVtx.TabIndex = 6;
            this.radVtx.TabStop = true;
            this.radVtx.Text = "Vertices";
            // 
            // radSeg
            // 
            this.radSeg.Location = new System.Drawing.Point(8, 90);
            this.radSeg.Name = "radSeg";
            this.radSeg.Size = new System.Drawing.Size(127, 28);
            this.radSeg.TabIndex = 4;
            this.radSeg.Text = "Track Segments";
            // 
            // radTrk
            // 
            this.radTrk.Location = new System.Drawing.Point(8, 118);
            this.radTrk.Name = "radTrk";
            this.radTrk.Size = new System.Drawing.Size(96, 16);
            this.radTrk.TabIndex = 5;
            this.radTrk.Text = "Tracks";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.cmdFont);
            this.groupBox3.Controls.Add(this.label16);
            this.groupBox3.Controls.Add(this.PointsTrackBar);
            this.groupBox3.Controls.Add(this.label15);
            this.groupBox3.Controls.Add(this.LinesTrackBar);
            this.groupBox3.Controls.Add(this.label14);
            this.groupBox3.Controls.Add(this.AlphaTrackBar);
            this.groupBox3.Location = new System.Drawing.Point(215, 216);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(201, 110);
            this.groupBox3.TabIndex = 7;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Graphics";
            // 
            // cmdFont
            // 
            this.cmdFont.Location = new System.Drawing.Point(140, 13);
            this.cmdFont.Name = "cmdFont";
            this.cmdFont.Size = new System.Drawing.Size(55, 20);
            this.cmdFont.TabIndex = 46;
            this.cmdFont.Text = "Font";
            this.cmdFont.Click += new System.EventHandler(this.cmdFont_Click);
            // 
            // label16
            // 
            this.label16.Location = new System.Drawing.Point(91, 16);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(40, 16);
            this.label16.TabIndex = 5;
            this.label16.Text = "Points";
            // 
            // PointsTrackBar
            // 
            this.PointsTrackBar.LargeChange = 1;
            this.PointsTrackBar.Location = new System.Drawing.Point(92, 30);
            this.PointsTrackBar.Minimum = 1;
            this.PointsTrackBar.Name = "PointsTrackBar";
            this.PointsTrackBar.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.PointsTrackBar.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.PointsTrackBar.Size = new System.Drawing.Size(45, 71);
            this.PointsTrackBar.TabIndex = 4;
            this.PointsTrackBar.TickStyle = System.Windows.Forms.TickStyle.None;
            this.PointsTrackBar.Value = 5;
            this.PointsTrackBar.ValueChanged += new System.EventHandler(this.OnPointSizeChanged);
            // 
            // label15
            // 
            this.label15.Location = new System.Drawing.Point(49, 16);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(40, 16);
            this.label15.TabIndex = 3;
            this.label15.Text = "Lines";
            // 
            // LinesTrackBar
            // 
            this.LinesTrackBar.LargeChange = 1;
            this.LinesTrackBar.Location = new System.Drawing.Point(51, 30);
            this.LinesTrackBar.Maximum = 4;
            this.LinesTrackBar.Minimum = 1;
            this.LinesTrackBar.Name = "LinesTrackBar";
            this.LinesTrackBar.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.LinesTrackBar.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.LinesTrackBar.Size = new System.Drawing.Size(45, 71);
            this.LinesTrackBar.TabIndex = 2;
            this.LinesTrackBar.TickStyle = System.Windows.Forms.TickStyle.None;
            this.LinesTrackBar.Value = 1;
            this.LinesTrackBar.ValueChanged += new System.EventHandler(this.OnLineWidthChanged);
            // 
            // label14
            // 
            this.label14.Location = new System.Drawing.Point(10, 16);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(40, 16);
            this.label14.TabIndex = 1;
            this.label14.Text = "Alpha";
            // 
            // AlphaTrackBar
            // 
            this.AlphaTrackBar.LargeChange = 1;
            this.AlphaTrackBar.Location = new System.Drawing.Point(10, 30);
            this.AlphaTrackBar.Maximum = 8;
            this.AlphaTrackBar.Name = "AlphaTrackBar";
            this.AlphaTrackBar.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.AlphaTrackBar.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.AlphaTrackBar.Size = new System.Drawing.Size(45, 71);
            this.AlphaTrackBar.TabIndex = 0;
            this.AlphaTrackBar.TickStyle = System.Windows.Forms.TickStyle.None;
            this.AlphaTrackBar.Value = 4;
            this.AlphaTrackBar.ValueChanged += new System.EventHandler(this.OnAlphaChanged);
            // 
            // colorDialog1
            // 
            this.colorDialog1.AnyColor = true;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.txtVertexFitName);
            this.groupBox4.Controls.Add(this.cmdVertexFit);
            this.groupBox4.Controls.Add(this.textDIP);
            this.groupBox4.Controls.Add(this.label5);
            this.groupBox4.Controls.Add(this.textNDIP);
            this.groupBox4.Controls.Add(this.label6);
            this.groupBox4.Controls.Add(this.textSecondTrackVtx);
            this.groupBox4.Controls.Add(this.label4);
            this.groupBox4.Controls.Add(this.textIPFirstTrack);
            this.groupBox4.Controls.Add(this.label3);
            this.groupBox4.Location = new System.Drawing.Point(215, 332);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(201, 169);
            this.groupBox4.TabIndex = 30;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Impact Parameter";
            // 
            // txtVertexFitName
            // 
            this.txtVertexFitName.Location = new System.Drawing.Point(111, 137);
            this.txtVertexFitName.Name = "txtVertexFitName";
            this.txtVertexFitName.Size = new System.Drawing.Size(84, 20);
            this.txtVertexFitName.TabIndex = 46;
            // 
            // cmdVertexFit
            // 
            this.cmdVertexFit.Location = new System.Drawing.Point(11, 136);
            this.cmdVertexFit.Name = "cmdVertexFit";
            this.cmdVertexFit.Size = new System.Drawing.Size(83, 20);
            this.cmdVertexFit.TabIndex = 47;
            this.cmdVertexFit.Text = "Vertex Fit";
            this.cmdVertexFit.Click += new System.EventHandler(this.cmdVertexFit_Click);
            // 
            // textDIP
            // 
            this.textDIP.Location = new System.Drawing.Point(111, 110);
            this.textDIP.Name = "textDIP";
            this.textDIP.ReadOnly = true;
            this.textDIP.Size = new System.Drawing.Size(53, 20);
            this.textDIP.TabIndex = 7;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(8, 111);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(73, 13);
            this.label5.TabIndex = 6;
            this.label5.Text = "Disconnected";
            // 
            // textNDIP
            // 
            this.textNDIP.Location = new System.Drawing.Point(111, 84);
            this.textNDIP.Name = "textNDIP";
            this.textNDIP.ReadOnly = true;
            this.textNDIP.Size = new System.Drawing.Size(53, 20);
            this.textNDIP.TabIndex = 5;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(8, 85);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(94, 13);
            this.label6.TabIndex = 4;
            this.label6.Text = "Non-disconnected";
            // 
            // textSecondTrackVtx
            // 
            this.textSecondTrackVtx.Location = new System.Drawing.Point(96, 49);
            this.textSecondTrackVtx.Name = "textSecondTrackVtx";
            this.textSecondTrackVtx.ReadOnly = true;
            this.textSecondTrackVtx.Size = new System.Drawing.Size(68, 20);
            this.textSecondTrackVtx.TabIndex = 3;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(8, 50);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(86, 13);
            this.label4.TabIndex = 2;
            this.label4.Text = "2nd track/vertex";
            // 
            // textIPFirstTrack
            // 
            this.textIPFirstTrack.Location = new System.Drawing.Point(96, 23);
            this.textIPFirstTrack.Name = "textIPFirstTrack";
            this.textIPFirstTrack.ReadOnly = true;
            this.textIPFirstTrack.Size = new System.Drawing.Size(68, 20);
            this.textIPFirstTrack.TabIndex = 1;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(8, 24);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(48, 13);
            this.label3.TabIndex = 0;
            this.label3.Text = "1st track";
            // 
            // btn400
            // 
            this.btn400.Location = new System.Drawing.Point(7, 629);
            this.btn400.Name = "btn400";
            this.btn400.Size = new System.Drawing.Size(81, 20);
            this.btn400.TabIndex = 31;
            this.btn400.Text = "400x400";
            this.btn400.Click += new System.EventHandler(this.btn400_Click);
            // 
            // btn600
            // 
            this.btn600.Location = new System.Drawing.Point(120, 629);
            this.btn600.Name = "btn600";
            this.btn600.Size = new System.Drawing.Size(81, 20);
            this.btn600.TabIndex = 32;
            this.btn600.Text = "600x600";
            this.btn600.Click += new System.EventHandler(this.btn600_Click);
            // 
            // btn800
            // 
            this.btn800.Location = new System.Drawing.Point(227, 629);
            this.btn800.Name = "btn800";
            this.btn800.Size = new System.Drawing.Size(81, 20);
            this.btn800.TabIndex = 33;
            this.btn800.Text = "800x800";
            this.btn800.Click += new System.EventHandler(this.btn800_Click);
            // 
            // btn1000
            // 
            this.btn1000.Location = new System.Drawing.Point(334, 629);
            this.btn1000.Name = "btn1000";
            this.btn1000.Size = new System.Drawing.Size(81, 20);
            this.btn1000.TabIndex = 34;
            this.btn1000.Text = "1000x1000";
            this.btn1000.Click += new System.EventHandler(this.btn1000_Click);
            // 
            // cmdTrackBrowser
            // 
            this.cmdTrackBrowser.Location = new System.Drawing.Point(174, 172);
            this.cmdTrackBrowser.Name = "cmdTrackBrowser";
            this.cmdTrackBrowser.Size = new System.Drawing.Size(28, 20);
            this.cmdTrackBrowser.TabIndex = 45;
            this.cmdTrackBrowser.Text = ">";
            this.cmdTrackBrowser.Click += new System.EventHandler(this.cmdTrackBrowser_Click);
            // 
            // cmdSaveToTSR
            // 
            this.cmdSaveToTSR.Location = new System.Drawing.Point(214, 599);
            this.cmdSaveToTSR.Name = "cmdSaveToTSR";
            this.cmdSaveToTSR.Size = new System.Drawing.Size(202, 24);
            this.cmdSaveToTSR.TabIndex = 46;
            this.cmdSaveToTSR.Text = "Save to TSR";
            this.cmdSaveToTSR.Click += new System.EventHandler(this.cmdSaveToTSR_Click);
            // 
            // cmdRemoveByOwner
            // 
            this.cmdRemoveByOwner.Location = new System.Drawing.Point(214, 541);
            this.cmdRemoveByOwner.Name = "cmdRemoveByOwner";
            this.cmdRemoveByOwner.Size = new System.Drawing.Size(202, 24);
            this.cmdRemoveByOwner.TabIndex = 47;
            this.cmdRemoveByOwner.Text = "Remove object from plot";
            this.cmdRemoveByOwner.Click += new System.EventHandler(this.cmdRemoveByOwner_Click);
            // 
            // cmdExportToOperaFeedback
            // 
            this.cmdExportToOperaFeedback.Location = new System.Drawing.Point(214, 570);
            this.cmdExportToOperaFeedback.Name = "cmdExportToOperaFeedback";
            this.cmdExportToOperaFeedback.Size = new System.Drawing.Size(202, 24);
            this.cmdExportToOperaFeedback.TabIndex = 48;
            this.cmdExportToOperaFeedback.Text = "Export to OperaFeedback";
            this.cmdExportToOperaFeedback.Click += new System.EventHandler(this.cmdExportToOperaFeedback_Click);
            // 
            // cmdShowGlobalData
            // 
            this.cmdShowGlobalData.Location = new System.Drawing.Point(215, 510);
            this.cmdShowGlobalData.Name = "cmdShowGlobalData";
            this.cmdShowGlobalData.Size = new System.Drawing.Size(104, 24);
            this.cmdShowGlobalData.TabIndex = 49;
            this.cmdShowGlobalData.Text = "Global data";
            this.cmdShowGlobalData.Click += new System.EventHandler(this.cmdShowGlobalData_Click);
            // 
            // btnDecaySearchAssistant
            // 
            this.btnDecaySearchAssistant.Location = new System.Drawing.Point(326, 510);
            this.btnDecaySearchAssistant.Name = "btnDecaySearchAssistant";
            this.btnDecaySearchAssistant.Size = new System.Drawing.Size(90, 24);
            this.btnDecaySearchAssistant.TabIndex = 50;
            this.btnDecaySearchAssistant.Text = "Decay Search";
            this.btnDecaySearchAssistant.Click += new System.EventHandler(this.btnDecaySearchAssistant_Click);
            // 
            // cmbFilter
            // 
            this.cmbFilter.FormattingEnabled = true;
            this.cmbFilter.Location = new System.Drawing.Point(6, 251);
            this.cmbFilter.Name = "cmbFilter";
            this.cmbFilter.Size = new System.Drawing.Size(185, 21);
            this.cmbFilter.TabIndex = 53;
            // 
            // btnFilterAdd
            // 
            this.btnFilterAdd.Location = new System.Drawing.Point(101, 226);
            this.btnFilterAdd.Name = "btnFilterAdd";
            this.btnFilterAdd.Size = new System.Drawing.Size(40, 24);
            this.btnFilterAdd.TabIndex = 54;
            this.btnFilterAdd.Text = "+";
            this.btnFilterAdd.UseVisualStyleBackColor = true;
            this.btnFilterAdd.Click += new System.EventHandler(this.btnFilterAdd_Click);
            // 
            // btnFilterDel
            // 
            this.btnFilterDel.Location = new System.Drawing.Point(151, 226);
            this.btnFilterDel.Name = "btnFilterDel";
            this.btnFilterDel.Size = new System.Drawing.Size(40, 24);
            this.btnFilterDel.TabIndex = 55;
            this.btnFilterDel.Text = "-";
            this.btnFilterDel.UseVisualStyleBackColor = true;
            this.btnFilterDel.Click += new System.EventHandler(this.btnFilterDel_Click);
            // 
            // DisplayForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(425, 664);
            this.Controls.Add(this.btnDecaySearchAssistant);
            this.Controls.Add(this.cmdShowGlobalData);
            this.Controls.Add(this.cmdExportToOperaFeedback);
            this.Controls.Add(this.cmdRemoveByOwner);
            this.Controls.Add(this.cmdSaveToTSR);
            this.Controls.Add(this.cmdTrackBrowser);
            this.Controls.Add(this.btn1000);
            this.Controls.Add(this.btn800);
            this.Controls.Add(this.btn600);
            this.Controls.Add(this.btn400);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.MaximizeBox = false;
            this.Name = "DisplayForm";
            this.Text = "Display";
            this.Load += new System.EventHandler(this.OnLoad);
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.OnClose);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.PointsTrackBar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.LinesTrackBar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.AlphaTrackBar)).EndInit();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.ResumeLayout(false);

		}
		#endregion


		private void cmdZoomPlus_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Zoom *= 2.0;
		}

		private void cmdZoomMinus_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Zoom *= 0.5;
		}

		private void radRotation_CheckedChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
		}

		private void radPanning_CheckedChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Pan;
		}


		private void cmdDefault_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.SetCameraOrientation(0, 0, -1, 0, -1, 0);
		}

		private void txtMinSegments_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				MinimumSegmentsNumber = Convert.ToInt32(txtMinSegments.Text);
			}
			catch(Exception exc)
			{
			}
		}

		private void SpawnGraph(SySal.TotalScan.Track tk, System.Collections.ArrayList tklist)
		{
			SySal.TotalScan.Vertex v;			
			int i, j;
			for (i = 0; i < tklist.Count && tk != tklist[i]; i++);
			if (i < tklist.Count) return;
			tklist.Add(tk);
            if (chkGraphDownstream.Checked)
    			if ((v = tk.Downstream_Vertex) != null)			
	    			for (j = 0; j < v.Length; j++)
		    			SpawnGraph(v[j], tklist);
            if (chkGraphUpstream.Checked)
    			if ((v = tk.Upstream_Vertex) != null)			
	    			for (j = 0; j < v.Length; j++)
		    			SpawnGraph(v[j], tklist);			
		}

        SySal.Executables.EasyReconstruct.dShow tkFilter = null;

        protected int m_TrackSelection = 0;

        public void ShowTracks(dShow filter, bool show, bool makedataset)
        {
            if (show)
            {
                tkFilter = filter;
                cmdPlot_Click(this, null);
                tkFilter = null;
            }
            if (makedataset)
            {
                AnalysisForm af = null;
                System.Collections.ArrayList darr = new ArrayList();
                af = new AnalysisForm();
                af.Text = "Track selection #" + (++m_TrackSelection);
                int i, n, di;
                n = mTracks.Length;
                for (i = 0; i < n; i++)
                {
                    SySal.TotalScan.Track tk = mTracks[i];
                    if (filter(tk))
                    {
                        double[] d = new double[TrackFilterFunctions.Length];
                        for (di = 0; di < TrackFilterFunctions.Length; di++)
                            d[di] = TrackFilterFunctions[di].F(tk);
                        darr.Add(d);
                    }
                }
                af.analysisControl1.AddDataSet("TkSel" + m_TrackSelection);                
                for (di = 0; di < TrackFilterFunctions.Length; di++)
                    af.analysisControl1.AddVariable(ExtractDoubleArray(darr, di), TrackFilterFunctions[di].Name, "");
                af.Show();
            } 
        }

        protected int m_SegmentSelection = 0;

        public void ShowSegments(dShowSeg filter, bool show, bool makedataset)
        {
            const int WarningLimit = 1000;
            AnalysisForm af = null;
            System.Collections.ArrayList darr = new ArrayList();
            if (makedataset)
            {
                af = new AnalysisForm();
                af.Text = "Segment selection #" + (++m_SegmentSelection);
            }
            int i, j, p, n;
            string fstr = cmbFilter.Text.Trim();
            ObjFilter f = null;
            try
            {
                f = (fstr.Length == 0) ? null : f = new ObjFilter(SegmentFilterFunctions, fstr);
            }
            catch (Exception x)
            {
                MessageBox.Show("Please specify a valid filter function.\r\n" + x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            gdiDisplay1.AutoRender = false;

            p = 0;
            for (i = 0; i < m_Layers.Length && p < WarningLimit; i++)
            {
                SySal.TotalScan.Layer lay = m_Layers[i];
                n = lay.Length;
                for (j = 0; j < n && p < WarningLimit; j++)
                {
                    SySal.TotalScan.Segment lays = lay[j];
                    if (filter(lays) && (f == null || f.Value(lays) != 0.0)) p++;
                }
            }
            if (p >= WarningLimit && MessageBox.Show("More than " + WarningLimit + " segments selected; proceed or cancel?", "Confirmation required", MessageBoxButtons.OKCancel, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Cancel) return;
            
            for (i = 0; i < m_Layers.Length; i++)
            {
                SySal.TotalScan.Layer lay = m_Layers[i];
                n = lay.Length;
                for (j = 0; j < n; j++)
                {
                    SySal.TotalScan.Segment lays = lay[j];
                    if (filter(lays) && (f == null || f.Value(lays) != 0.0))
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = lays.Info;
                        if (show)
                            AddSegmentAutoColor(info.Intercept.X + info.Slope.X * (info.TopZ - info.Intercept.Z), info.Intercept.Y + info.Slope.Y * (info.TopZ - info.Intercept.Z), info.TopZ,
                                info.Intercept.X + info.Slope.X * (info.BottomZ - info.Intercept.Z), info.Intercept.Y + info.Slope.Y * (info.BottomZ - info.Intercept.Z), info.BottomZ, lay[j], null);
                        if (makedataset)
                        {
                            double[] d = new double[SegmentFilterFunctions.Length];
                            int di;
                            for (di = 0; di < SegmentFilterFunctions.Length; di++)
                                d[di] = SegmentFilterFunctions[di].F(lays);
                            darr.Add(d);
                        }
                    }
                }
            }
            if (show)
            {
                gdiDisplay1.AutoRender = true;
                gdiDisplay1.Render();
            }
            if (makedataset)
            {
                af.analysisControl1.AddDataSet("SegSel" + m_SegmentSelection);
                int di;
                for (di = 0; di < SegmentFilterFunctions.Length; di++)
                    af.analysisControl1.AddVariable(ExtractDoubleArray(darr, di), SegmentFilterFunctions[di].Name, "");
                af.Show();
            }
        }

        static double[] ExtractDoubleArray(System.Collections.ArrayList darr, int pos)
        {
            double[] a = new double[darr.Count];
            int i;
            for (i = 0; i < a.Length; i++)
                a[i] = ((double[])darr[i])[pos];
            return a;
        }

        public void ShowSegments(SySal.TotalScan.Segment[] segs, object owner)
        {
            gdiDisplay1.AutoRender = false;
            SySal.Tracking.MIPEmulsionTrackInfo lastinfo = null;
            foreach (SySal.TotalScan.Segment s in segs)
            {
                if (s == null) continue;
                SySal.Tracking.MIPEmulsionTrackInfo info = s.Info;
                AddSegmentAutoColor(info.Intercept.X + info.Slope.X * (info.TopZ - info.Intercept.Z), info.Intercept.Y + info.Slope.Y * (info.TopZ - info.Intercept.Z), info.TopZ,
                    info.Intercept.X + info.Slope.X * (info.BottomZ - info.Intercept.Z), info.Intercept.Y + info.Slope.Y * (info.BottomZ - info.Intercept.Z), info.BottomZ, s, null);
                if (lastinfo != null)
                    AddSegment(info.Intercept.X + info.Slope.X * (info.TopZ - info.Intercept.Z), info.Intercept.Y + info.Slope.Y * (info.TopZ - info.Intercept.Z), info.TopZ,
                        lastinfo.Intercept.X + lastinfo.Slope.X * (lastinfo.BottomZ - lastinfo.Intercept.Z), lastinfo.Intercept.Y + lastinfo.Slope.Y * (lastinfo.BottomZ - lastinfo.Intercept.Z), lastinfo.BottomZ,
                        owner, 255, 160, 255, null);
                lastinfo = info;
            }
            gdiDisplay1.AutoRender = true;
            gdiDisplay1.Render();
        }

        public void ToggleAddReplaceSegments(GDI3D.Control.SelectObject so)
        {
            if (so == null)
            {
                gdiDisplay1.Cursor = Cursors.Arrow;
                gdiDisplay1.DoubleClickSelect = new GDI3D.Control.SelectObject(OnSelectObject);
            }
            else 
            {
                gdiDisplay1.Cursor = Cursors.Hand;
                gdiDisplay1.DoubleClickSelect = so;
            }
        }

        public SySal.TotalScan.Segment[] Follow(SySal.TotalScan.Segment start, bool downstream, int mingrains, double postol, double slopetol, double slopetolincrease, int maxmiss)
        {
            int misses = 0;
            int layer;
            CheckedListBox.CheckedItemCollection e_dss = clDataSets.CheckedItems;
            SySal.TotalScan.Segment [] segs = new SySal.TotalScan.Segment[m_Layers.Length];
            segs[start.LayerOwner.Id] = start;
            SySal.TotalScan.Segment lastseg = start;
            for (layer = start.LayerOwner.Id + (downstream ? -1 : 1); misses <= maxmiss && layer >= 0 && layer <= m_Layers.Length - 1; layer += (downstream ? -1 : 1))
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = lastseg.Info;
                SySal.TotalScan.Layer ls = m_Layers[layer];
                double px = info.Intercept.X + (ls.RefCenter.Z - info.Intercept.Z) * info.Slope.X;
                double py = info.Intercept.Y + (ls.RefCenter.Z - info.Intercept.Z) * info.Slope.Y;
                double sx = info.Slope.X;
                double sy = info.Slope.Y;
                double s2 = sx * sx + sy * sy;
                double nx = 1.0, ny = 0.0;
                if (s2 > 0.0)
                {
                    s2 = Math.Sqrt(s2);
                    nx = sx / s2;
                    ny = sy / s2;
                }
                double sltol = slopetol + s2 * slopetolincrease;
                int i;
                SySal.TotalScan.Segment nextseg = null;
                double dsq = 0.0, bestdsq = 0.0;
                for (i = 0; i < ls.Length; i++)
                {
                    if (SySal.TotalScan.Flexi.Volume.IsIn(e_dss, ((SySal.TotalScan.Flexi.Segment)ls[i]).DataSet) == false) continue;
                    SySal.Tracking.MIPEmulsionTrackInfo ninfo = ls[i].Info;
                    if (ninfo.Count < mingrains) continue;
                    if (Math.Abs(ninfo.Intercept.X - px) > postol) continue;
                    if (Math.Abs(ninfo.Intercept.Y - py) > postol) continue;
                    double dsx = ninfo.Slope.X - sx;
                    double dsy = ninfo.Slope.Y - sy;
                    double dst = Math.Abs(dsx * ny - dsy * nx);
                    if (dst > slopetol) continue;
                    double dsl = Math.Abs(dsx * nx + dsy * ny);
                    if (dsl > sltol) continue;
                    dst /= slopetol;
                    dsl /= sltol;
                    dsq = dst * dst + dsl * dsl;
                    if (bestdsq > dsq || nextseg == null)
                    {
                        nextseg = ls[i];
                        bestdsq = dsq;
                    }
                }
                if (nextseg == null) misses++;
                else
                {
                    misses = 0;
                    segs[layer] = nextseg;
                    lastseg = nextseg;
                }
            }
            return segs;
        }

        public void Add(SySal.TotalScan.Track tk)
        {
            foreach (SySal.TotalScan.Track t in mTracks)
                if (tk == t) return;
            int i;
            SySal.TotalScan.Track[] newtks = new SySal.TotalScan.Track[mTracks.Length + 1];
            for (i = 0; i < mTracks.Length; i++) newtks[i] = mTracks[i];
            newtks[i] = tk;
            mTracks = newtks;
        }

        public void Remove(SySal.TotalScan.Track tk)
        {
            gdiDisplay1.DeleteWithOwner(tk);
        }

        public void Remove(SySal.TotalScan.Vertex vx)
        {
            gdiDisplay1.DeleteWithOwner(vx);
        }

        private void cmdPlot_Click(object sender, System.EventArgs e)
		{
            ObjFilter ff = null;
            string fstr = cmbFilter.Text.Trim();
			try
			{
				gdiDisplay1.AutoRender = false;
                CheckedListBox.CheckedItemCollection e_dss = clDataSets.CheckedItems;
				System.Collections.ArrayList VerticesShown = new System.Collections.ArrayList();
				int vsindex;                
				if (MinimumSegmentsNumber<1) throw new Exception("Minimum Segment Number below 1.");
				SySal.TotalScan.Track [] displayTracks = null;
				int i, j;
                bool usetagged = this.radioTaggedSegs.Checked;
                bool useuntagged = this.radioUntaggedSegs.Checked;
                int rtag = cmdTagColor.BackColor.R;
                int gtag = cmdTagColor.BackColor.G;
                int btag = cmdTagColor.BackColor.B;
                if (this.radAllSegs.Checked || usetagged || useuntagged)
                {
                    ff = (fstr.Length > 0) ? new ObjFilter(SegmentFilterFunctions, fstr) : null;
                    gdiDisplay1.AutoRender = false;
                    for (i = 0; i < this.m_Layers.Length; i++)
                    {
                        SySal.TotalScan.Layer lay = this.m_Layers[i];
                        for (j = 0; j < lay.Length; j++)
                        {
                            SySal.TotalScan.Flexi.Segment sj = (SySal.TotalScan.Flexi.Segment)lay[j];
                            if (SySal.TotalScan.Flexi.Volume.IsIn(e_dss, sj.DataSet) == false) continue;
                            if (usetagged && (sj == null || sj.Index is SySal.TotalScan.NullIndex)) continue;
                            if (useuntagged && !(sj.Index != null && sj.Index is SySal.TotalScan.NullIndex)) continue;
                            if (ff != null && ff.Value(sj) == 0.0) continue;
                            SySal.Tracking.MIPEmulsionTrackInfo info = lay[j].Info;
                            double[] x = new double[2] { info.Intercept.X + (lay.DownstreamZ - info.Intercept.Z + m_SegExtend) * info.Slope.X, info.Intercept.X + (lay.UpstreamZ - info.Intercept.Z - m_SegExtend) * info.Slope.X };
                            double[] y = new double[2] { info.Intercept.Y + (lay.DownstreamZ - info.Intercept.Z + m_SegExtend) * info.Slope.Y, info.Intercept.Y + (lay.UpstreamZ - info.Intercept.Z - m_SegExtend) * info.Slope.Y };
                            double[] z = new double[2] { lay.DownstreamZ + m_SegExtend, lay.UpstreamZ - m_SegExtend };
                            {
                                int r = rtag, g = gtag, b = btag;
                                //string segtag = sj.Index.ToString();
                                if (!usetagged)
                                {
                                    AutoColor(ref r, ref g, ref b, info.Intercept.Z, -2);
                                    r = (r + 255) / 2;
                                    g = (g + 255) / 2;
                                    b = (b + 255) / 2;
                                    //segtag = null;
                                }
                                AddSegment(x[0], y[0], z[0], x[1], y[1], z[1], sj, r, g, b, sj.ToString());
                            }
                        }
                    }
                    gdiDisplay1.Render();
                    gdiDisplay1.AutoRender = true;
                    return;
                }
				else if (tkFilter == null && this.radioTrackGraph.Checked)
				{                    
					for (i = 0; i < mTracks.Length && mTracks[i].Id != this.GraphStartTrackId; i++);
					if (i < mTracks.Length)
					{
						System.Collections.ArrayList tklist = new System.Collections.ArrayList();
						SpawnGraph(mTracks[i], tklist);
						displayTracks = (SySal.TotalScan.Track [])tklist.ToArray(typeof(SySal.TotalScan.Track));
					}
					else
					{
						MessageBox.Show("Can't find the track with Id = " + this.GraphStartTrackId + ".", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        gdiDisplay1.AutoRender = true;
						return;
					}
				}
                else if (tkFilter == null && this.radioVtxGraph.Checked)
				{
					if (this.GraphStartVtxId >= 0 && this.GraphStartVtxId < m_V.Vertices.Length)
					{
						System.Collections.ArrayList tklist = new System.Collections.ArrayList();
                        foreach (SySal.TotalScan.Track iti in mTracks)
                            if ((iti.Upstream_Vertex != null && iti.Upstream_Vertex.Id == this.GraphStartVtxId) ||
                                (iti.Downstream_Vertex != null && iti.Downstream_Vertex.Id == this.GraphStartVtxId))
                                SpawnGraph(iti, tklist);
                        displayTracks = (SySal.TotalScan.Track[])tklist.ToArray(typeof(SySal.TotalScan.Track));
					}
					else
					{
						MessageBox.Show("Can't find the vertex with Id = " + this.GraphStartVtxId + ".", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        gdiDisplay1.AutoRender = true;
						return;
					}
				}
				else displayTracks = mTracks;
                int n = displayTracks.Length, m;
                double CentralZ = m_Layers[m_Layers.Length / 2].RefCenter.Z;
				//gdiDisplay1.Clear();				
                ff = (fstr.Length > 0) ? new ObjFilter(TrackFilterFunctions, fstr) : null;
				for (i=0; i<n; i++)
				{
					SySal.TotalScan.Flexi.Track tmpT = (SySal.TotalScan.Flexi.Track)displayTracks[i];
                    if (SySal.TotalScan.Flexi.Volume.IsIn(e_dss, tmpT.DataSet) == false) continue;
                    if (CheckAttributes(tmpT, txtAttrSearch.Text) == false) continue;
                    if (ff != null && ff.Value(tmpT) == 0.0) continue;
                    string tklabel = "T " + tmpT.Id;
                    if (tkFilter != null && tkFilter(tmpT) == false) continue;
                    if (tmpT[0].LayerOwner.Id < MaxPThruDown) continue;
                    if (tmpT[tmpT.Length - 1].LayerOwner.Id >= (m_Layers.Length - MaxPThruUp)) continue;
                    if (SlopeTol > 0.0)
                    {
                        if ((Math.Abs(tmpT.Upstream_SlopeX - SlopeX) > SlopeTol || Math.Abs(tmpT.Upstream_SlopeY - SlopeY) > SlopeTol) &&
                            (Math.Abs(tmpT.Downstream_SlopeX - SlopeX) > SlopeTol || Math.Abs(tmpT.Downstream_SlopeY - SlopeY) > SlopeTol)) continue;
                    }
                    if (PosTol > 0.0)
                    {
                        if ((Math.Abs(tmpT.Upstream_PosX + (CentralZ - tmpT.Upstream_PosZ) * tmpT.Upstream_SlopeX - PosX) > PosTol || Math.Abs(tmpT.Upstream_PosY + (CentralZ - tmpT.Upstream_PosZ) * tmpT.Upstream_SlopeY - PosY) > PosTol) &&
                            (Math.Abs(tmpT.Downstream_PosX + (CentralZ - tmpT.Downstream_PosZ) * tmpT.Downstream_SlopeX - PosX) > PosTol || Math.Abs(tmpT.Downstream_PosY + (CentralZ - tmpT.Downstream_PosZ) * tmpT.Downstream_SlopeY - PosY) > PosTol)) continue;
                    }
					m = tmpT.Length;
					if(radSeg.Checked && m >= MinimumSegmentsNumber)
					{
						for(j=0; j<m; j++)
						{
                            SySal.Tracking.MIPEmulsionTrackInfo info = tmpT[j].Info;
                            SySal.TotalScan.Layer lay = tmpT[j].LayerOwner;
                            double[] x = new double[2] { info.Intercept.X + (lay.DownstreamZ - info.Intercept.Z + m_SegExtend) * info.Slope.X, info.Intercept.X + (lay.UpstreamZ - info.Intercept.Z - m_SegExtend) * info.Slope.X };
                            double[] y = new double[2] { info.Intercept.Y + (lay.DownstreamZ - info.Intercept.Z + m_SegExtend) * info.Slope.Y, info.Intercept.Y + (lay.UpstreamZ - info.Intercept.Z - m_SegExtend) * info.Slope.Y };
                            double[] z = new double[2] { lay.DownstreamZ + m_SegExtend, lay.UpstreamZ - m_SegExtend };
							{
								AddSegmentAutoColor(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, tklabel);
							}
						}
					}
					else if(radTrk.Checked && m >= MinimumSegmentsNumber)
					{
						if(chkShowSegments.Checked)
						{
							double[] x = new double[2];
							double[] y = new double[2];
							double[] z = new double[2];
							for(int h=0; h<tmpT.Length; h++)
							{
                                x[1] = tmpT[h].Info.Intercept.X + tmpT[h].Info.Slope.X * (tmpT[h].Info.BottomZ - tmpT[h].Info.Intercept.Z - m_SegExtend);
                                x[0] = tmpT[h].Info.Intercept.X + tmpT[h].Info.Slope.X * (tmpT[h].Info.TopZ - tmpT[h].Info.Intercept.Z + m_SegExtend);
                                y[1] = tmpT[h].Info.Intercept.Y + tmpT[h].Info.Slope.Y * (tmpT[h].Info.BottomZ - tmpT[h].Info.Intercept.Z - m_SegExtend);
                                y[0] = tmpT[h].Info.Intercept.Y + tmpT[h].Info.Slope.Y * (tmpT[h].Info.TopZ - tmpT[h].Info.Intercept.Z + m_SegExtend);
                                z[1] = tmpT[h].Info.BottomZ - m_SegExtend;
                                z[0] = tmpT[h].Info.TopZ + m_SegExtend;
                                AddSegmentAutoColor(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, (h == tmpT.Length / 2) ? tklabel : null);
							}
						}
						else
						{
							double[] x = null;
							double[] y = null;
							double[] z = null;
							for(j = 0; j < 2 * m - 1; j++)
							{
								if (j % 2 == 0)
								{
                                    x = new double[2] { tmpT[j / 2].Info.Intercept.X + (tmpT[j / 2].LayerOwner.DownstreamZ - tmpT[j / 2].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2].Info.Slope.X, tmpT[j / 2].Info.Intercept.X + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.X };
                                    y = new double[2] { tmpT[j / 2].Info.Intercept.Y + (tmpT[j / 2].LayerOwner.DownstreamZ - tmpT[j / 2].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2].Info.Slope.Y, tmpT[j / 2].Info.Intercept.Y + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.Y };
                                    z = new double[2] { tmpT[j / 2].LayerOwner.DownstreamZ + m_SegExtend, tmpT[j / 2].LayerOwner.UpstreamZ - m_SegExtend };
									AddSegment(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, 240, 240, 240, null);
								}
								else
								{
                                    x = new double[2] { tmpT[j / 2 + 1].Info.Intercept.X + (tmpT[j / 2 + 1].LayerOwner.DownstreamZ - tmpT[j / 2 + 1].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2 + 1].Info.Slope.X, tmpT[j / 2].Info.Intercept.X + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.X };
                                    y = new double[2] { tmpT[j / 2 + 1].Info.Intercept.Y + (tmpT[j / 2 + 1].LayerOwner.DownstreamZ - tmpT[j / 2 + 1].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2 + 1].Info.Slope.Y, tmpT[j / 2].Info.Intercept.Y + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.Y };
                                    z = new double[2] { tmpT[j / 2 + 1].LayerOwner.DownstreamZ + m_SegExtend, tmpT[j / 2].LayerOwner.UpstreamZ - m_SegExtend };
                                    AddSegmentAutoColor(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, (j / 2 == m / 2) ? tklabel : null);
								}
							}
							if (tmpT.Downstream_Vertex != null)
                                try
                                {
                                    AddSegment(tmpT[0].Info.Intercept.X + (tmpT[0].LayerOwner.DownstreamZ - tmpT[0].Info.Intercept.Z) * tmpT[0].Info.Slope.X, tmpT[0].Info.Intercept.Y + (tmpT[0].LayerOwner.DownstreamZ - tmpT[0].Info.Intercept.Z) * tmpT[0].Info.Slope.Y, tmpT[0].LayerOwner.DownstreamZ, tmpT.Downstream_Vertex.X, tmpT.Downstream_Vertex.Y, tmpT.Downstream_Vertex.Z, tmpT, 64, 64, 96, null);
                                    vsindex = VerticesShown.BinarySearch(tmpT.Downstream_Vertex.Id);
                                    if (vsindex < 0)
                                    {
                                        AddPointAutoColor(tmpT.Downstream_Vertex.X, tmpT.Downstream_Vertex.Y, tmpT.Downstream_Vertex.Z, tmpT.Downstream_Vertex, "V " + tmpT.Downstream_Vertex.Id);
                                        VerticesShown.Insert(~vsindex, tmpT.Downstream_Vertex.Id);
                                    }
                                }
                                catch (Exception) { }
							if (tmpT.Upstream_Vertex != null)
                                try
                                {
                                    AddSegment(tmpT[m - 1].Info.Intercept.X + (tmpT[m - 1].LayerOwner.UpstreamZ - tmpT[m - 1].Info.Intercept.Z) * tmpT[m - 1].Info.Slope.X, tmpT[m - 1].Info.Intercept.Y + (tmpT[m - 1].LayerOwner.UpstreamZ - tmpT[m - 1].Info.Intercept.Z) * tmpT[m - 1].Info.Slope.Y, tmpT[m - 1].LayerOwner.UpstreamZ, tmpT.Upstream_Vertex.X, tmpT.Upstream_Vertex.Y, tmpT.Upstream_Vertex.Z, tmpT, 96, 64, 64, null);
                                    vsindex = VerticesShown.BinarySearch(tmpT.Upstream_Vertex.Id);
                                    if (vsindex < 0)
                                    {
                                        AddPointAutoColor(tmpT.Upstream_Vertex.X, tmpT.Upstream_Vertex.Y, tmpT.Upstream_Vertex.Z, tmpT.Upstream_Vertex, "V " + tmpT.Upstream_Vertex.Id);
                                        VerticesShown.Insert(~vsindex, tmpT.Upstream_Vertex.Id);
                                    }
                                }
                                catch (Exception) { }
						}
					}
					else if(this.radioTrackGraph.Checked || this.radioVtxGraph.Checked || 
						(this.radVtx.Checked && m >= MinimumSegmentsNumber && 
						    ((tmpT.Upstream_Vertex != null && tmpT.Upstream_Vertex.Length >= MinimumTracksNumber) ||
						     (tmpT.Downstream_Vertex != null && tmpT.Downstream_Vertex.Length >= MinimumTracksNumber))))
					{
						
						if(chkShowSegments.Checked)
						{
							double[] x = new double[2];
							double[] y = new double[2];
							double[] z = new double[2];
							for(int h=0; h<tmpT.Length; h++)
							{
                                x[1] = tmpT[h].Info.Intercept.X + tmpT[h].Info.Slope.X * (tmpT[h].Info.BottomZ - tmpT[h].Info.Intercept.Z - m_SegExtend);
								x[0] = tmpT[h].Info.Intercept.X + tmpT[h].Info.Slope.X*(tmpT[h].Info.TopZ-tmpT[h].Info.Intercept.Z + m_SegExtend);
                                y[1] = tmpT[h].Info.Intercept.Y + tmpT[h].Info.Slope.Y * (tmpT[h].Info.BottomZ - tmpT[h].Info.Intercept.Z - m_SegExtend);
                                y[0] = tmpT[h].Info.Intercept.Y + tmpT[h].Info.Slope.Y * (tmpT[h].Info.TopZ - tmpT[h].Info.Intercept.Z + m_SegExtend);
                                z[1] = tmpT[h].Info.BottomZ - m_SegExtend;
                                z[0] = tmpT[h].Info.TopZ + m_SegExtend;
                                AddSegmentAutoColor(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, (h == tmpT.Length / 2) ? tklabel : null);
							}
						}
						else
						{
							double[] x = null;
							double[] y = null;
							double[] z = null;
							for(j = 0; j < 2 * m - 1; j++)
							{
								if (j % 2 == 0)
								{
                                    x = new double[2] { tmpT[j / 2].Info.Intercept.X + (tmpT[j / 2].LayerOwner.DownstreamZ - tmpT[j / 2].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2].Info.Slope.X, tmpT[j / 2].Info.Intercept.X + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.X };
                                    y = new double[2] { tmpT[j / 2].Info.Intercept.Y + (tmpT[j / 2].LayerOwner.DownstreamZ - tmpT[j / 2].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2].Info.Slope.Y, tmpT[j / 2].Info.Intercept.Y + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.Y };
                                    z = new double[2] { tmpT[j / 2].LayerOwner.DownstreamZ + m_SegExtend, tmpT[j / 2].LayerOwner.UpstreamZ - m_SegExtend };
									AddSegment(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, 240, 240, 240, null);
								}
								else
								{
                                    x = new double[2] { tmpT[j / 2 + 1].Info.Intercept.X + (tmpT[j / 2 + 1].LayerOwner.DownstreamZ - tmpT[j / 2 + 1].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2 + 1].Info.Slope.X, tmpT[j / 2].Info.Intercept.X + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.X };
                                    y = new double[2] { tmpT[j / 2 + 1].Info.Intercept.Y + (tmpT[j / 2 + 1].LayerOwner.DownstreamZ - tmpT[j / 2 + 1].Info.Intercept.Z + m_SegExtend) * tmpT[j / 2 + 1].Info.Slope.Y, tmpT[j / 2].Info.Intercept.Y + (tmpT[j / 2].LayerOwner.UpstreamZ - tmpT[j / 2].Info.Intercept.Z - m_SegExtend) * tmpT[j / 2].Info.Slope.Y };
                                    z = new double[2] { tmpT[j / 2 + 1].LayerOwner.DownstreamZ + m_SegExtend, tmpT[j / 2].LayerOwner.UpstreamZ - m_SegExtend };
                                    AddSegmentAutoColor(x[0], y[0], z[0], x[1], y[1], z[1], tmpT, (j / 2 == m / 2) ? tklabel : null);
								}
							}
							if (tmpT.Downstream_Vertex != null)
                                try
                                {
                                    AddSegment(tmpT[0].Info.Intercept.X + (tmpT[0].LayerOwner.DownstreamZ - tmpT[0].Info.Intercept.Z) * tmpT[0].Info.Slope.X, tmpT[0].Info.Intercept.Y + (tmpT[0].LayerOwner.DownstreamZ - tmpT[0].Info.Intercept.Z) * tmpT[0].Info.Slope.Y, tmpT[0].LayerOwner.DownstreamZ, tmpT.Downstream_Vertex.X, tmpT.Downstream_Vertex.Y, tmpT.Downstream_Vertex.Z, tmpT, 64, 64, 96, null);
                                    vsindex = VerticesShown.BinarySearch(tmpT.Downstream_Vertex.Id);
                                    if (vsindex < 0)
                                    {
                                        AddPointAutoColor(tmpT.Downstream_Vertex.X, tmpT.Downstream_Vertex.Y, tmpT.Downstream_Vertex.Z, tmpT.Downstream_Vertex, "V " + tmpT.Downstream_Vertex.Id);
                                        VerticesShown.Insert(~vsindex, tmpT.Downstream_Vertex.Id);
                                    }
                                }
                                catch (Exception) { }
							if (tmpT.Upstream_Vertex != null)
                                try
                                {
                                    AddSegment(tmpT[m - 1].Info.Intercept.X + (tmpT[m - 1].LayerOwner.UpstreamZ - tmpT[m - 1].Info.Intercept.Z) * tmpT[m - 1].Info.Slope.X, tmpT[m - 1].Info.Intercept.Y + (tmpT[m - 1].LayerOwner.UpstreamZ - tmpT[m - 1].Info.Intercept.Z) * tmpT[m - 1].Info.Slope.Y, tmpT[m - 1].LayerOwner.UpstreamZ, tmpT.Upstream_Vertex.X, tmpT.Upstream_Vertex.Y, tmpT.Upstream_Vertex.Z, tmpT, 96, 64, 64, null);
                                    vsindex = VerticesShown.BinarySearch(tmpT.Upstream_Vertex.Id);
                                    if (vsindex < 0)
                                    {
                                        AddPointAutoColor(tmpT.Upstream_Vertex.X, tmpT.Upstream_Vertex.Y, tmpT.Upstream_Vertex.Z, tmpT.Upstream_Vertex, "V " + tmpT.Upstream_Vertex.Id);
                                        VerticesShown.Insert(~vsindex, tmpT.Upstream_Vertex.Id);
                                    }
                                }
                                catch (Exception) { }
						}
					}
				}
				gdiDisplay1.Render();
				gdiDisplay1.AutoRender = true;
			}
			catch(Exception exc)
			{
                MessageBox.Show(exc.ToString(), "Painting error.");
			}
		}

        GDIPanel m_Panel = null;

        internal GDI3D.Control.GDIDisplay gdiDisplay1;

		private void OnLoad(object sender, System.EventArgs e)
		{
            if (MainForm.RedirectSaveTSR2Close) cmdSaveToTSR.Text = "Save to ROOT file and close";
            txtSegExtend.Text = m_SegExtend.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMinSegments.Text = MinimumSegmentsNumber.ToString();
            txtMinTracks.Text = MinimumTracksNumber.ToString();
            txtPThruUp.Text = MaxPThruUp.ToString();
            txtPThruDown.Text = MaxPThruDown.ToString();
            txtSlopeY.Text = SlopeY.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtSlopeX.Text = SlopeX.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtSlopeTol.Text = SlopeTol.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtPosY.Text = PosY.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtPosX.Text = PosX.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtPosTol.Text = PosTol.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtXRot.Text = m_XRotDeg.ToString();
            txtYRot.Text = m_YRotDeg.ToString();
            txtMovieKB.Text = m_MovieKB.ToString();
            SySal.TotalScan.Flexi.DataSet[] dss = m_V.DataSets;
            clDataSets.Items.Clear();
            cmbAttrFilter.SelectedIndex = 0;
            cmdRemoveByOwner_Click(sender, e);
            foreach (SySal.TotalScan.Flexi.DataSet ds1 in dss)
                clDataSets.SetItemChecked(clDataSets.Items.Add(ds1), true);
            //gdiDisplay1.MouseMultiplier = (0.001 * Math.Pow((MaxX - MinX) * (MaxY - MinY) * (MaxZ - MinZ) + 1.0, 1.0 / 3.0));
            cmdDefault_Click(sender, e);
            cmbFilter.Items.Clear();
            foreach (string s in UserProfileInfo.ThisProfileInfo.DisplayFilters)
                cmbFilter.Items.Add(s);
            m_Panel.Show();
            m_Panel.Location = new Point(this.Right, this.Top);
		}

		private void txtMinTracks_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				MinimumTracksNumber = Convert.ToInt32(txtMinTracks.Text);
			}
			catch(Exception exc)
			{                
			}
	
		}

		private void OnTrackGraphStartLeave(object sender, System.EventArgs e)
		{
			try
			{
				GraphStartTrackId = Convert.ToInt32(textTrackGraphStart.Text);
			}
			catch (Exception)
			{
				MessageBox.Show("Track graph start id must be a number", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				textTrackGraphStart.Focus();
			}
		}

		private void OnVtxGraphStartLeave(object sender, System.EventArgs e)
		{
			try
			{
				GraphStartVtxId = Convert.ToInt32(textVtxGraphStart.Text);
			}
			catch (Exception)
			{
				MessageBox.Show("Vertex graph start id must be a number", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				textVtxGraphStart.Focus();
			}		
		}

        static int[][] autocolors = new int [][] {
            new int [] {0,0,255},
            new int [] {0,255,0},
            new int [] {255,0,0},
            new int [] {255,255,0},
            new int [] {0,255,255},
            new int [] {255,0,255}
        };


		void AutoColor(ref int r, ref int g, ref int b, double z, int ownerid)
		{
			double lev = (z - B_MinZ) / (B_MaxZ - B_MinZ);
			if (lev < 0.0) lev = 0.0;
			else if (lev > 1.0) lev = 1.0;
            if (ownerid < 0)
            {
                if (ownerid == -2)
                {
                    r = (int)(255 * (1.0 - lev * lev) * (1.0 - lev * lev));
                    g = (int)(255 * (4.0 * lev * (1.0 - lev)));
                    b = (int)(255 * lev * (2.0 - lev) * lev * (2.0 - lev));
                }
                else
                {
                    r = (int)(64 * lev) + 128;
                    g = (int)(64 * lev) + 128;
                    b = (int)(64 * lev) + 128;
                }
            }
            else
            {
                ownerid = ownerid % autocolors.Length;
                r = autocolors[ownerid][0];
                g = autocolors[ownerid][1];
                b = autocolors[ownerid][2];
                if (lev < 0.5)
                {
                    r = (int)(r * (lev + 0.5));
                    g = (int)(g * (lev + 0.5));
                    b = (int)(b * (lev + 0.5));
                }
                else
                {
                    r = Math.Min((int)(r + 127 * (1 - lev)), 255);
                    g = Math.Min((int)(g + 127 * (1 - lev)), 255);
                    b = Math.Min((int)(b + 127 * (1 - lev)), 255);
                }
            }
		}

		void AddSegmentAutoColor(double xf, double yf, double zf, double xs, double ys, double zs, object owner, string label)
		{
			int r = 0, g = 0, b = 0;
            if (checkTrackVtxColor.Checked == false)
            {
                AutoColor(ref r, ref g, ref b, 0.5 * (zf + zs), -2);
            }
            else
            {
                int ownerid = -1;
                SySal.TotalScan.Track tk = null;
                try
                {
                    tk = (SySal.TotalScan.Track)owner;
                    if (tk.Upstream_Vertex != null)
                    {
                        SySal.TotalScan.Vertex vtx = tk.Upstream_Vertex;
                        double myslope = (tk.Upstream_SlopeX * tk.Upstream_SlopeX + tk.Upstream_SlopeY * tk.Upstream_SlopeY);
                        int i, pos;
                        for (i = pos = 0; i < vtx.Length; i++)
                        {
                            SySal.TotalScan.Track ntk = vtx[i];
                            if (ntk.Upstream_Vertex == vtx)
                                if ((ntk.Upstream_SlopeX * ntk.Upstream_SlopeX + ntk.Upstream_SlopeY * ntk.Upstream_SlopeY) < myslope) pos++;
                        }
                        ownerid = vtx.Id + pos;
                    }
                }
                catch (Exception) { }
                AutoColor(ref r, ref g, ref b, 0.5 * (zf + zs), ownerid);                
            }
            GDI3D.Control.Line l = new GDI3D.Control.Line(xf, yf, zf, xs, ys, zs, owner, r, g, b);
            l.Label = label;
			gdiDisplay1.Add(l);
		}

		void AddSegment(double xf, double yf, double zf, double xs, double ys, double zs, object owner, int r, int g, int b, string label)
		{
            GDI3D.Control.Line l = new GDI3D.Control.Line(xf, yf, zf, xs, ys, zs, owner, r, g, b);
            l.Label = label;
			gdiDisplay1.Add(l);
		}

		void AddPointAutoColor(double x, double y, double z, object owner, string label)
		{
			int r = 0, g = 0, b = 0;
            int ownerid = -1;
            try
            {
                ownerid = ((SySal.TotalScan.Vertex)owner).Id;
            }
            catch(Exception) {};
			AutoColor(ref r, ref g, ref b, z, -1);
            GDI3D.Control.Point p = new GDI3D.Control.Point(x, y, z, owner, r, g, b);
            p.Label = label;
			gdiDisplay1.Add(p);
		}

		void AddPoint(double x, double y, double z, object owner, int r, int g, int b, string label)
		{
            GDI3D.Control.Point p = new GDI3D.Control.Point(x, y, z, owner, r, g, b);
            p.Label = label;
			gdiDisplay1.Add(p);
		}

		private void OnAlphaChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.Alpha = AlphaTrackBar.Value / 8.0;
		}

		private void OnLineWidthChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.LineWidth = LinesTrackBar.Value;
		}

		private void OnPointSizeChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.PointSize = PointsTrackBar.Value;
		}

		private void SaveButton_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sd = new SaveFileDialog();
			sd.Title = "Select file and format";
			sd.Filter = "Windows Bitmap (*.bmp)|*.bmp|Joint Photographic Experts Group (*.jpg)|*.jpg|Graphics Interexchange Format (*.GIF)|*.gif|Portable Network Graphics (*.png)|*.png|3D XML Scene file (*.x3l)|*.x3l";
			if (sd.ShowDialog() == DialogResult.OK)
			{
				gdiDisplay1.Save(sd.FileName);
			}
		}

		private void OnSelectObject(object obj)
		{
            if (obj == null) return;
            if (m_DoubleClickRemoves)
            {
                gdiDisplay1.DeleteWithOwner(obj);
                gdiDisplay1.Render();
                return;
            }
			if (obj is SySal.TotalScan.Track)
			{
				SySal.TotalScan.Track tk = (SySal.TotalScan.Track)obj;
				TrackBrowser.Browse(tk, m_Layers, this, this, m_SelectedEvent, m_V);
			}
            else if (obj is SySal.TotalScan.Vertex)
            {
                SySal.TotalScan.Vertex vtx = (SySal.TotalScan.Vertex)obj;
                VertexBrowser.Browse(vtx, m_Layers, this, this, m_SelectedEvent, m_V);
            }
            else
            {
                new QBrowser(obj.GetType().ToString() + " Info", obj.ToString()).ShowDialog();
                if (obj is SySal.TotalScan.Flexi.Segment)
                {
                    SySal.TotalScan.Flexi.Segment seg = (SySal.TotalScan.Flexi.Segment)obj;
                    if (seg.TrackOwner == null)
                    {
                        if (MessageBox.Show("Segment is not associated to any track.\r\nPromote to volume track now?", "User input", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                        {
                            SySal.TotalScan.Flexi.Track newtk = new SySal.TotalScan.Flexi.Track(seg.DataSet, m_V.Tracks.Length);
                            newtk.AddSegments(new SySal.TotalScan.Flexi.Segment[1] { seg } );
                            ((SySal.TotalScan.Flexi.Volume.TrackList)m_V.Tracks).Insert(new SySal.TotalScan.Flexi.Track[1] { newtk } );
                            SySal.TotalScan.Track[] newtks = new SySal.TotalScan.Track[mTracks.Length + 1];
                            int i;
                            for (i = 0; i < mTracks.Length; i++) newtks[i] = mTracks[i];
                            newtks[i] = newtk;
                            mTracks = newtks;
                            TrackBrowser.Browse(newtk, m_V.Layers, this, this, m_SelectedEvent, m_V);
                        }
                    }
                }
            }
		}

		private void checkIsometric_CheckedChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.Infinity = checkIsometric.Checked;
		}

		private void cmdBkgnd_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() == DialogResult.OK)
			{
				gdiDisplay1.BackColor = colorDialog1.Color;
				gdiDisplay1.Render();
			}			
		}

        private void cmdClear_Click(object sender, EventArgs e)
        {
            gdiDisplay1.Clear();
            gdiDisplay1.Render();
        }

        #region IPSelector Members

        SySal.TotalScan.Track m_IPFirst;
        object m_IPSecond;

        void IPSelector.SelectTrack(SySal.TotalScan.Track tk, bool isfirst)
        {
            if (isfirst)
            {
                if (m_IPFirst != null) gdiDisplay1.Highlight(m_IPFirst, false);
                m_IPFirst = tk;                
                textIPFirstTrack.Text = "Tk " + tk.Id;
                gdiDisplay1.Highlight(m_IPFirst, true);
            }
            else
            {
                if (m_IPSecond != null) gdiDisplay1.Highlight(m_IPSecond, false);
                m_IPSecond = tk;
                textSecondTrackVtx.Text = "Tk " + tk.Id;
                gdiDisplay1.Highlight(m_IPSecond, true);
            }
            ComputeIP();
        }

        void IPSelector.SelectVertex(SySal.TotalScan.Vertex vtx)
        {
            if (m_IPSecond != null) gdiDisplay1.Highlight(m_IPSecond, false);
            m_IPSecond = vtx;
            textSecondTrackVtx.Text = "Vtx " + vtx.Id;
            gdiDisplay1.Highlight(m_IPSecond, true);
            ComputeIP();
        }

        void IPSelector.SetLabel(object owner, string label)
        {
            gdiDisplay1.SetLabel(owner, label);
        }

        void IPSelector.EnableLabel(object owner, bool enable)
        {
            gdiDisplay1.EnableLabel(owner, enable);
        }

        void IPSelector.Highlight(object owner, bool hlstatus)
        {
            gdiDisplay1.Highlight(owner, hlstatus);
        }

        bool IPSelector.DeleteWithOwner(object owner)
        {
            return gdiDisplay1.DeleteWithOwner(owner);
        }

        void IPSelector.Plot(object owner)
        {
            ShowTracks(new dShow(new TrackBrowser.TrackFilterWithTrack((SySal.TotalScan.Track)owner).Filter), true, false);
        }        

        void ComputeIP()
        {
            if (m_IPFirst == null || m_IPSecond == null)
            {
                textDIP.Text = textNDIP.Text = "";
            }
            else
                try
                {
                    if (m_IPSecond is SySal.TotalScan.Track)
                    {
                        SySal.TotalScan.Track second = (SySal.TotalScan.Track)m_IPSecond;
                        SySal.TotalScan.VertexFit vf = new SySal.TotalScan.VertexFit();
                        SySal.TotalScan.VertexFit.TrackFit tf1 = new SySal.TotalScan.VertexFit.TrackFit();
                        SySal.TotalScan.VertexFit.TrackFit tf2 = new SySal.TotalScan.VertexFit.TrackFit();
                        tf1.Id = new SySal.TotalScan.BaseTrackIndex(m_IPFirst.Id);
                        tf2.Id = new SySal.TotalScan.BaseTrackIndex(second.Id);
                        if (m_IPFirst.Upstream_Z > second.Downstream_Z)
                        {                            
                            tf1.Intercept.X = m_IPFirst.Upstream_PosX + (m_IPFirst.Upstream_Z - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeX;
                            tf1.Intercept.Y = m_IPFirst.Upstream_PosY + (m_IPFirst.Upstream_Z - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeY;
                            tf1.Intercept.Z = m_IPFirst.Upstream_Z;
                            tf1.Slope.X = m_IPFirst.Upstream_SlopeX;
                            tf1.Slope.Y = m_IPFirst.Upstream_SlopeY;
                            tf1.Slope.Z = 1.0;
                            tf1.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_IPFirst);
                            tf1.MaxZ = m_IPFirst.Upstream_Z;
                            tf1.MinZ = tf1.MaxZ - 1e6;
                            vf.AddTrackFit(tf1);                            
                            tf2.Intercept.X = second.Downstream_PosX + (second.Downstream_Z - second.Downstream_PosZ) * second.Downstream_SlopeX;
                            tf2.Intercept.Y = second.Downstream_PosY + (second.Downstream_Z - second.Downstream_PosZ) * second.Downstream_SlopeY;
                            tf2.Intercept.Z = second.Downstream_Z;
                            tf2.Slope.X = second.Downstream_SlopeX;
                            tf2.Slope.Y = second.Downstream_SlopeY;
                            tf2.Slope.Z = 1.0;
                            tf2.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(second);
                            tf2.MinZ = second.Downstream_Z;
                            tf2.MaxZ = tf2.MinZ + 1e6;
                            vf.AddTrackFit(tf2);
                        }
                        else if (m_IPFirst.Downstream_Z < second.Upstream_Z)
                        {
                            tf1.Intercept.X = second.Upstream_PosX + (second.Upstream_Z - second.Upstream_PosZ) * second.Upstream_SlopeX;
                            tf1.Intercept.Y = second.Upstream_PosY + (second.Upstream_Z - second.Upstream_PosZ) * second.Upstream_SlopeY;
                            tf1.Intercept.Z = second.Upstream_Z;
                            tf1.Slope.X = second.Upstream_SlopeX;
                            tf1.Slope.Y = second.Upstream_SlopeY;
                            tf1.Slope.Z = 1.0;
                            tf1.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(second);
                            tf1.MaxZ = second.Upstream_Z;
                            tf1.MinZ = tf1.MaxZ - 1e6;
                            vf.AddTrackFit(tf1);
                            tf2.Intercept.X = m_IPFirst.Downstream_PosX + (m_IPFirst.Downstream_Z - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeX;
                            tf2.Intercept.Y = m_IPFirst.Downstream_PosY + (m_IPFirst.Downstream_Z - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeY;
                            tf2.Intercept.Z = m_IPFirst.Downstream_Z;
                            tf2.Slope.X = m_IPFirst.Downstream_SlopeX;
                            tf2.Slope.Y = m_IPFirst.Downstream_SlopeY;
                            tf2.Slope.Z = 1.0;
                            tf2.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_IPFirst);
                            tf2.MinZ = m_IPFirst.Downstream_Z;
                            tf2.MaxZ = tf2.MinZ + 1e6;
                            vf.AddTrackFit(tf2);
                        }
                        else
                        {
                            double maxz = Math.Max(m_IPFirst.Downstream_Z, second.Downstream_Z);
                            double minz = Math.Min(m_IPFirst.Upstream_Z, second.Upstream_Z);
                            double dx, dy, ddownz, dupz;
                            dx = (m_IPFirst.Downstream_PosX + (maxz - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeX) -
                                (second.Downstream_PosX + (maxz - second.Downstream_PosZ) * second.Downstream_SlopeX);
                            dy = (m_IPFirst.Downstream_PosY + (maxz - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeY) -
                                (second.Downstream_PosY + (maxz - second.Downstream_PosZ) * second.Downstream_SlopeY);
                            ddownz = dx * dx + dy * dy;
                            dx = (m_IPFirst.Upstream_PosX + (minz - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeX) -
                                (second.Upstream_PosX + (minz - second.Upstream_PosZ) * second.Upstream_SlopeX);
                            dy = (m_IPFirst.Upstream_PosY + (minz - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeY) -
                                (second.Upstream_PosY + (minz - second.Upstream_PosZ) * second.Upstream_SlopeY);
                            dupz = dx * dx + dy * dy;
                            if (ddownz > dupz)
                            {
                                tf1.Intercept.X = m_IPFirst.Upstream_PosX + (m_IPFirst.Upstream_Z - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeX;
                                tf1.Intercept.Y = m_IPFirst.Upstream_PosY + (m_IPFirst.Upstream_Z - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeY;
                                tf1.Intercept.Z = m_IPFirst.Upstream_Z;
                                tf1.Slope.X = m_IPFirst.Upstream_SlopeX;
                                tf1.Slope.Y = m_IPFirst.Upstream_SlopeY;
                                tf1.Slope.Z = 1.0;
                                tf1.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_IPFirst);
                                tf1.MaxZ = m_IPFirst.Upstream_Z;
                                tf1.MinZ = tf1.MaxZ - 1e6;
                                vf.AddTrackFit(tf1);
                                tf2.Intercept.X = second.Upstream_PosX + (second.Upstream_Z - second.Upstream_PosZ) * second.Upstream_SlopeX;
                                tf2.Intercept.Y = second.Upstream_PosY + (second.Upstream_Z - second.Upstream_PosZ) * second.Upstream_SlopeY;
                                tf2.Intercept.Z = second.Upstream_Z;
                                tf2.Slope.X = second.Upstream_SlopeX;
                                tf2.Slope.Y = second.Upstream_SlopeY;
                                tf2.Slope.Z = 1.0;
                                tf2.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(second);
                                tf2.MinZ = second.Upstream_Z;
                                tf2.MaxZ = tf2.MinZ + 1e6;
                                vf.AddTrackFit(tf2);
                            }
                            else
                            {
                                tf1.Intercept.X = m_IPFirst.Downstream_PosX + (m_IPFirst.Downstream_Z - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeX;
                                tf1.Intercept.Y = m_IPFirst.Downstream_PosY + (m_IPFirst.Downstream_Z - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeY;
                                tf1.Intercept.Z = m_IPFirst.Downstream_Z;
                                tf1.Slope.X = m_IPFirst.Downstream_SlopeX;
                                tf1.Slope.Y = m_IPFirst.Downstream_SlopeY;
                                tf1.Slope.Z = 1.0;
                                tf1.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_IPFirst);
                                tf1.MaxZ = m_IPFirst.Downstream_Z;
                                tf1.MinZ = tf1.MaxZ - 1e6;
                                vf.AddTrackFit(tf1);
                                tf2.Intercept.X = second.Downstream_PosX + (second.Downstream_Z - second.Downstream_PosZ) * second.Downstream_SlopeX;
                                tf2.Intercept.Y = second.Downstream_PosY + (second.Downstream_Z - second.Downstream_PosZ) * second.Downstream_SlopeY;
                                tf2.Intercept.Z = second.Downstream_Z;
                                tf2.Slope.X = second.Downstream_SlopeX;
                                tf2.Slope.Y = second.Downstream_SlopeY;
                                tf2.Slope.Z = 1.0;
                                tf2.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(second);
                                tf2.MinZ = second.Downstream_Z;
                                tf2.MaxZ = tf2.MinZ + 1e6;
                                vf.AddTrackFit(tf2);
                            }
                        }
                        double dip = vf.DisconnectedTrackIP(tf1.Id);
                        textDIP.Text = dip.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                        textNDIP.Text = "";
                        try
                        {
                            double ndip = vf.TrackIP(tf1);
                            textNDIP.Text = ndip.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                        }
                        catch (Exception) { }
                    }
                    else
                    {
                        SySal.TotalScan.Vertex vtx = (SySal.TotalScan.Vertex)m_IPSecond;
                        SySal.TotalScan.VertexFit vf = new SySal.TotalScan.VertexFit();
                        int i;
                        for (i = 0; i < vtx.Length; i++)
                        {
                            SySal.TotalScan.Track tk = vtx[i];
                            SySal.TotalScan.VertexFit.TrackFit tf = new SySal.TotalScan.VertexFit.TrackFit();
                            if (tk.Upstream_Vertex == vtx)
                            {
                                tf.Intercept.X = tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX;
                                tf.Intercept.Y = tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY;
                                tf.Intercept.Z = tk.Upstream_Z;
                                tf.Slope.X = tk.Upstream_SlopeX;
                                tf.Slope.Y = tk.Upstream_SlopeY;
                                tf.Slope.Z = 1.0;
                                tf.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(tk);
                                tf.MaxZ = tk.Upstream_Z;
                                tf.MinZ = tf.MaxZ - 1e6;                                                                        
                            }
                            else
                            {
                                tf.Intercept.X = tk.Downstream_PosX + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeX;
                                tf.Intercept.Y = tk.Downstream_PosY + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeY;
                                tf.Intercept.Z = tk.Downstream_Z;
                                tf.Slope.X = tk.Downstream_SlopeX;
                                tf.Slope.Y = tk.Downstream_SlopeY;
                                tf.Slope.Z = 1.0;
                                tf.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(tk);
                                tf.MinZ = tk.Downstream_Z;
                                tf.MaxZ = tf.MinZ + 1e6;
                            }
                            tf.Id = new SySal.TotalScan.BaseTrackIndex(tk.Id);
                            vf.AddTrackFit(tf);
                        }
                        SySal.TotalScan.VertexFit.TrackFit tf1 = new SySal.TotalScan.VertexFit.TrackFit();
                        if (Math.Abs(vf.Z - m_IPFirst.Downstream_Z) < Math.Abs(vf.Z - m_IPFirst.Upstream_Z))
                        {
                            tf1.Intercept.X = m_IPFirst.Downstream_PosX + (m_IPFirst.Downstream_Z - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeX;
                            tf1.Intercept.Y = m_IPFirst.Downstream_PosY + (m_IPFirst.Downstream_Z - m_IPFirst.Downstream_PosZ) * m_IPFirst.Downstream_SlopeY;
                            tf1.Intercept.Z = m_IPFirst.Downstream_Z;
                            tf1.Slope.X = m_IPFirst.Downstream_SlopeX;
                            tf1.Slope.Y = m_IPFirst.Downstream_SlopeY;
                            tf1.Slope.Z = 1.0;
                            tf1.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_IPFirst);
                            tf1.MinZ = m_IPFirst.Downstream_Z;
                            tf1.MaxZ = tf1.MinZ + 1e6;
                        }
                        else
                        {
                            tf1.Intercept.X = m_IPFirst.Upstream_PosX + (m_IPFirst.Upstream_Z - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeX;
                            tf1.Intercept.Y = m_IPFirst.Upstream_PosY + (m_IPFirst.Upstream_Z - m_IPFirst.Upstream_PosZ) * m_IPFirst.Upstream_SlopeY;
                            tf1.Intercept.Z = m_IPFirst.Upstream_Z;
                            tf1.Slope.X = m_IPFirst.Upstream_SlopeX;
                            tf1.Slope.Y = m_IPFirst.Upstream_SlopeY;
                            tf1.Slope.Z = 1.0;
                            tf1.Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_IPFirst);
                            tf1.MaxZ = m_IPFirst.Upstream_Z;
                            tf1.MinZ = tf1.MaxZ - 1e6;                                                                        
                        }
                        tf1.Id = new SySal.TotalScan.BaseTrackIndex(m_IPFirst.Id);
                        double ndip = vf.TrackIP(tf1);
                        textNDIP.Text = ndip.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                        textDIP.Text = "";
                        try
                        {
                            double dip = vf.DisconnectedTrackIP(tf1.Id);
                            textDIP.Text = dip.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                        }
                        catch (Exception) { }
                    }
                }
                catch (Exception)
                {
                    textDIP.Text = textNDIP.Text = "";
                }
        }

        #endregion

        private int MaxPThruUp = 0;

        private void OnMaxPThruUpLeave(object sender, EventArgs e)
        {
            try
            {
                MaxPThruUp = System.Convert.ToInt32(txtPThruUp.Text);
            }
            catch (Exception)
            {
                txtPThruUp.Text = MaxPThruUp.ToString();
            }
        }

        private int MaxPThruDown = 0;

        private void OnMaxPThruDownLeave(object sender, EventArgs e)
        {
            try
            {
                MaxPThruDown = System.Convert.ToInt32(txtPThruDown.Text);
            }
            catch (Exception)
            {
                txtPThruDown.Text = MaxPThruDown.ToString();
            }
        }

        private double SlopeX = 0.0;

        private void OnSlopeXLeave(object sender, EventArgs e)
        {
            try
            {
                SlopeX = System.Convert.ToDouble(txtSlopeX.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtSlopeX.Text = SlopeX.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private double SlopeY = 0.0;

        private void OnSlopeYLeave(object sender, EventArgs e)
        {
            try
            {
                SlopeY = System.Convert.ToDouble(txtSlopeY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtSlopeY.Text = SlopeY.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }

        }

        private double SlopeTol = -1.0;

        private void OnSlopeTolLeave(object sender, EventArgs e)
        {
            try
            {
                SlopeTol = System.Convert.ToDouble(txtSlopeTol.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtSlopeTol.Text = SlopeTol.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private double PosX = 0.0;

        private void OnPosXLeave(object sender, EventArgs e)
        {
            try
            {
                PosX = System.Convert.ToDouble(txtPosX.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtPosX.Text = PosX.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private double PosY = 0.0;

        private void OnPosYLeave(object sender, EventArgs e)
        {
            try
            {
                PosY = System.Convert.ToDouble(txtPosY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtPosY.Text = PosY.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private double PosTol = -1.0;

        private void OnPosTolLeave(object sender, EventArgs e)
        {
            try
            {
                PosTol = System.Convert.ToDouble(txtPosTol.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtPosTol.Text = PosTol.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void cmdPlateFrame_Click(object sender, EventArgs e)
        {
            try
            {
                gdiDisplay1.AutoRender = false;
                int i, p, q;
                for (i = 0; i < m_Layers.Length; i++)
                {
                    SySal.TotalScan.Layer lay = m_Layers[i];
                    SySal.BasicTypes.Vector [] v = new SySal.BasicTypes.Vector[4];
                    v[0].X = m_Extents.MinX;
                    v[0].Y = m_Extents.MinY;
                    v[0].Z = lay.DownstreamZ;
                    v[1].X = m_Extents.MinX;
                    v[1].Y = m_Extents.MaxY;
                    v[1].Z = lay.DownstreamZ;
                    v[2].X = m_Extents.MaxX;
                    v[2].Y = m_Extents.MaxY;
                    v[2].Z = lay.DownstreamZ;
                    v[3].X = m_Extents.MaxX;
                    v[3].Y = m_Extents.MinY;
                    v[3].Z = lay.DownstreamZ;
                    for (p = 0; p < v.Length; p++)
                    {
                        v[p] = lay.ToAlignedPoint(v[p]);
                        v[p].Z = lay.DownstreamZ;
                    }                    
                    for (p = 0; p < v.Length; p++)
                    {
                        q = (p + v.Length - 1) % v.Length;
                        gdiDisplay1.Add(new GDI3D.Control.Line(v[p].X, v[p].Y, v[p].Z, v[q].X, v[q].Y, v[q].Z, null, cmdPlateColor.BackColor.R, cmdPlateColor.BackColor.G, cmdPlateColor.BackColor.B));
                    }                    
                    for (p = 0; p < v.Length; p++)
                        v[p].Z = lay.UpstreamZ;
                    for (p = 0; p < v.Length; p++)
                    {
                        q = (p + v.Length - 1) % v.Length;
                        gdiDisplay1.Add(new GDI3D.Control.Line(v[p].X, v[p].Y, v[p].Z, v[q].X, v[q].Y, v[q].Z, null, cmdPlateColor.BackColor.R, cmdPlateColor.BackColor.G, cmdPlateColor.BackColor.B));
                    }                    
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Graphics error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                gdiDisplay1.Transform();
                gdiDisplay1.Render();
                gdiDisplay1.AutoRender = true;                
            }
        }

        private void cmdPlateColor_Click(object sender, EventArgs e)
        {
            colorDialog1.Color = cmdPlateColor.BackColor;
            if (colorDialog1.ShowDialog() == DialogResult.OK)
            {
                cmdPlateColor.BackColor = colorDialog1.Color;
            }
        }

        int m_XRotDeg = 0;

        int m_YRotDeg = 0;

        private void cmdRot_Click(object sender, EventArgs e)
        {
            SySal.BasicTypes.Vector w = new SySal.BasicTypes.Vector();
            w.X = Math.PI / 180.0 * m_XRotDeg;
            w.Y = Math.PI / 180.0 * m_YRotDeg;
            w.Z = 0.0;
            SySal.BasicTypes.Vector T = new SySal.BasicTypes.Vector();
            T.X = 0;
            T.Y = 0;
            T.Z = -1;
            SySal.BasicTypes.Vector D = Rotate(w, T);
            T.X = 0;
            T.Y = 1;
            T.Z = 0;
            SySal.BasicTypes.Vector N = Rotate(w, T);
            gdiDisplay1.SetCameraOrientation(D.X, D.Y, D.Z, N.X, N.Y, N.Z);
            gdiDisplay1.Distance = gdiDisplay1.Distance;
            gdiDisplay1.Render();            
        }

        private static SySal.BasicTypes.Vector Rotate(SySal.BasicTypes.Vector theta, SySal.BasicTypes.Vector v)
        {
            double norm = theta.X * theta.X + theta.Y * theta.Y + theta.Z * theta.Z;
            if (norm <= 0.0) return v;
            norm = Math.Sqrt(norm);
            double inorm = 1.0 / norm;
            double wx = v.X * theta.X * inorm;
            double wy = v.Y * theta.Y * inorm;
            double wz = v.Z * theta.Z * inorm;
            double qx = v.X - wx;
            double qy = v.Y - wy;
            double qz = v.Z - wz;
            double ux = (theta.Y * v.Z - theta.Z * v.Y) * inorm;
            double uy = (theta.Z * v.X - theta.X * v.Z) * inorm;
            double uz = (theta.X * v.Y - theta.Y * v.X) * inorm;
            double c = Math.Cos(norm);
            double s = Math.Sin(norm);
            v.X = wx + c * qx + s * ux;
            v.Y = wy + c * qy + s * uy;
            v.Z = wz + c * qz + s * uz;
            return v;
        }

        private void txtXRot_Leave(object sender, EventArgs e)
        {
            try
            {
                m_XRotDeg = System.Convert.ToInt32(txtXRot.Text);
            }
            catch (Exception)
            {
                txtXRot.Text = m_XRotDeg.ToString();
            }
        }

        private void txtYRot_Leave(object sender, EventArgs e)
        {
            try
            {
                m_YRotDeg = System.Convert.ToInt32(txtYRot.Text);
            }
            catch (Exception)
            {
                txtYRot.Text = m_YRotDeg.ToString();
            }
        }

        bool m_Closing = false;

        private void OnClose(object sender, FormClosingEventArgs e)
        {
            if (!m_Closing)
            {
                if (MVForm != null) MVForm.Close();
                DSAForm.Close();
                TrackBrowser.CloseAll();
                VertexBrowser.CloseAll();
                VertexFitForm.CloseAll();
                m_Closing = true;
                m_Panel.Close();
            }
        }

        private void btn400_Click(object sender, EventArgs e)
        {
            m_Panel.SetSize(0);
        }

        private void btn600_Click(object sender, EventArgs e)
        {
            m_Panel.SetSize(200);
        }

        private void btn800_Click(object sender, EventArgs e)
        {
            m_Panel.SetSize(400);
        }

        private void btn1000_Click(object sender, EventArgs e)
        {
            m_Panel.SetSize(600);
        }

        private void OnTagColorClick(object sender, EventArgs e)
        {
            colorDialog1.Color = cmdTagColor.BackColor;
            if (colorDialog1.ShowDialog() == DialogResult.OK)
            {
                cmdTagColor.BackColor = colorDialog1.Color;
            }
        }

        private void cmdSetFocus_Click(object sender, EventArgs e)
        {
            gdiDisplay1.NextClickSetsCenter = true;
        }

        private void cmdTrackBrowser_Click(object sender, EventArgs e)
        {
            if (GraphStartTrackId >= 0 && GraphStartTrackId < mTracks.Length)
                TrackBrowser.Browse(mTracks[GraphStartTrackId], m_Layers, this, this, m_SelectedEvent, m_V);
        }

        private void cmdVertexBrowser_Click(object sender, EventArgs e)
        {
            if (GraphStartVtxId < 0) return;
            foreach (SySal.TotalScan.Track tk in mTracks)            
            {
                if (tk.Upstream_Vertex != null && tk.Upstream_Vertex.Id == GraphStartVtxId) VertexBrowser.Browse(tk.Upstream_Vertex, m_Layers, this, this, m_SelectedEvent, m_V);
                if (tk.Downstream_Vertex != null && tk.Downstream_Vertex.Id == GraphStartVtxId) VertexBrowser.Browse(tk.Downstream_Vertex, m_Layers, this, this, m_SelectedEvent, m_V);       
            }
        }

        private void cmdFont_Click(object sender, EventArgs e)
        {
            if (fontDialog1.ShowDialog() == DialogResult.OK)
            {
                gdiDisplay1.AutoRender = false;
                gdiDisplay1.LabelFontName = fontDialog1.Font.Name;
                gdiDisplay1.LabelFontSize = (int)fontDialog1.Font.Size;
                gdiDisplay1.AutoRender = true;
            }
        }

        private void cmdVertexFit_Click(object sender, EventArgs e)
        {
            if (txtVertexFitName.Text.Trim().Length == 0)
            {
                MessageBox.Show("Please specify a name for the new vertex fit.", "Missing input", MessageBoxButtons.OK);
                return;
            }
            VertexFitForm.Browse(txtVertexFitName.Text, gdiDisplay1, this, m_V);
            if (OnAddFit != null) OnAddFit(this, null);
        }

        private event dGenericEvent OnAddFit;

        public void SubscribeOnAddFit(dGenericEvent ge)
        {
            OnAddFit += ge;
        }

        public void UnsubscribeOnAddFit(dGenericEvent ge)
        {
            OnAddFit -= ge;
        }

        public void RaiseAddFit(object sender, EventArgs e)
        {
            if (OnAddFit != null) OnAddFit(sender, e);
        }

        private double m_SegExtend = 100.0;

        private void OnSegExtendLeave(object sender, EventArgs e)
        {
            try
            {
                if ((m_SegExtend = System.Convert.ToDouble(txtSegExtend.Text, System.Globalization.CultureInfo.InvariantCulture)) < 0.0)
                {
                    m_SegExtend = 0.0;
                    throw new Exception();
                }
            }
            catch (Exception)
            {
                txtSegExtend.Text = m_SegExtend.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void cmdSaveToTSR_Click(object sender, EventArgs e)
        {
            System.IO.FileStream ws = null;
            try
            {
                SaveFileDialog sdlg = new SaveFileDialog();
                sdlg.Title = "Save all to TSR format";
                sdlg.Filter = "TSR files (*.tsr)|*.tsr|All files (*.*)|*.*";
                if (MainForm.RedirectSaveTSR2Close || sdlg.ShowDialog() == DialogResult.OK)
                {
                    int tkkilled = 0;
                    int vtxkilled = 0;
                    string killstr = "";
                    int i;
                    for (i = 0; i < m_V.Tracks.Length; i++)
                        try
                        {
                            if (m_V.Tracks[i].Length == 0) throw new Exception();
                            m_V.Tracks[i].NotifyChanged();
                            ((SySal.TotalScan.Flexi.Track)m_V.Tracks[i]).SetId(i);
                        }
                        catch (Exception)
                        {
                            tkkilled++;
                            SySal.TotalScan.Track tk = m_V.Tracks[i];
                            if (tk.Upstream_Vertex != null) tk.Upstream_Vertex.RemoveTrack(tk);
                            if (tk.Downstream_Vertex != null) tk.Downstream_Vertex.RemoveTrack(tk);
                            ((SySal.TotalScan.Flexi.Volume.TrackList)m_V.Tracks).Remove(new int[1] { i });
                            i--;
                        }
                    if (tkkilled > 0)
                        killstr = ((tkkilled == 1) ? "1 track was invalid and has been killed." :
                            (tkkilled + " tracks were invalid and have been killed.")) +
                            "\r\nTrack IDs have been renumbered.\r\n";                    
                    for (i = 0; i < m_V.Vertices.Length; i++)
                        try
                        {
                            m_V.Vertices[i].NotifyChanged();
                            if (m_V.Vertices[i].AverageDistance >= 0.0)
                                ((SySal.TotalScan.Flexi.Vertex)(m_V.Vertices[i])).SetId(i);
                        }
                        catch (Exception)
                        {
                            vtxkilled++;
                            SySal.TotalScan.Vertex vtxk = m_V.Vertices[i];
                            int j;
                            for (j = 0; j < vtxk.Length; j++)
                            {
                                SySal.TotalScan.Track tk = vtxk[j];
                                if (tk.Upstream_Vertex == vtxk) tk.SetUpstreamVertex(null);
                                else tk.SetDownstreamVertex(null);
                            }
                            ((SySal.TotalScan.Flexi.Volume.VertexList)m_V.Vertices).Remove(new int[1] { i });
                            i--;
                        }
                    if (vtxkilled > 0)
                        killstr += ((vtxkilled == 1) ? "1 vertex was invalid and has been killed." :
                            (vtxkilled + " vertices were invalid and have been killed.")) +
                            "\r\nVertex IDs have been renumbered.";
                    if (killstr.Length > 0)
                        MessageBox.Show(killstr, "Reconstruction modified", MessageBoxButtons.OK, MessageBoxIcon.Warning);

                    if (MainForm.RedirectSaveTSR2Close)
                    {
                        MainForm.EditedVolume = m_V;
                        Close();
                    }
                    else
                    {
                        ws = new System.IO.FileStream(sdlg.FileName, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                        m_V.Save(ws);
                        ws.Flush();
                        ws.Close();
                        ws = null;
                    }
                }
            }
            catch (Exception exc)
            {
                MessageBox.Show(exc.Message, "File error");
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

        private bool CheckAttributes(SySal.TotalScan.IAttributeList attrlist, string attrtext)
        {
            string st = attrtext.Trim().ToUpper();
            if (st.Length == 0) return true;
            bool useinclude = (cmbAttrFilter.SelectedIndex == 0);
            SySal.TotalScan.Attribute[] attrs = attrlist.ListAttributes();
            foreach (SySal.TotalScan.Attribute a in attrs)
                if (a.Index.ToString().ToUpper().IndexOf(st) >= 0)
                {
                    if (useinclude) return true;
                    else return false;
                }
            return !useinclude;
        }

        bool m_DoubleClickRemoves = true;

        private void cmdRemoveByOwner_Click(object sender, EventArgs e)
        {
            m_DoubleClickRemoves = !m_DoubleClickRemoves;
            cmdRemoveByOwner.Text = m_DoubleClickRemoves ? "Stop removing objects" : "Remove object from plot";            
        }

        private void cmdExportToOperaFeedback_Click(object sender, EventArgs e)
        {
            System.Collections.ArrayList ar_v = new ArrayList();
            System.Collections.ArrayList ar_t = new ArrayList();
            int i;
            for (i = 0; i < m_V.Vertices.Length; i++)
                try
                {
                    if (m_V.Vertices[i].GetAttribute(VertexBrowser.FBEventIndex) >= 0.0) ar_v.Add(m_V.Vertices[i]);
                }
                catch (Exception) { }
            for (i = 0; i < m_V.Tracks.Length; i++)
                try
                {
                    if (m_V.Tracks[i].GetAttribute(TrackBrowser.FBEventIndex) >= 0.0) ar_t.Add(m_V.Tracks[i]);
                }
                catch (Exception) { }
#if (DEBUG)
#else
            try
            {
#endif
                SySal.Executables.OperaFeedback.MainForm ofb = new SySal.Executables.OperaFeedback.MainForm();
                int brick = 0;
                for (i = 0; i < m_Layers.Length && (m_Layers[i].BrickId < 1000000 || m_Layers[i].BrickId >= 3000000); i++);
                if (i < m_Layers.Length) brick = (int)m_Layers[i].BrickId;
                ofb.Preset(brick, (SySal.TotalScan.Vertex[])ar_v.ToArray(typeof(SySal.TotalScan.Vertex)), (SySal.TotalScan.Track[])ar_t.ToArray(typeof(SySal.TotalScan.Track)));
                ofb.ShowDialog();
#if (DEBUG)
#else
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Export error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
#endif
        }

        internal static string DefaultProjFileName = "";

        private void btnFilterVars_Click(object sender, EventArgs e)
        {
            string helpstr = "Segment filter variables:";
            foreach (FilterF f in SegmentFilterFunctions)
                helpstr += "\r\n" + f.Name + " -> " + f.HelpText;
            helpstr += "\r\nTrack filter variables:";
            foreach (FilterF f in TrackFilterFunctions)
                helpstr += "\r\n" + f.Name + " -> " + f.HelpText;
            MessageBox.Show(helpstr, "Help on filter variables");
        }

        public delegate double dFilterF(object o);

        public class FilterF
        {
            public string Name;
            public dFilterF F;
            public string HelpText;
            public FilterF(string n, dFilterF f, string h) { Name = n; F = f; HelpText = h; }
        }

        public class ExtendedFieldF : FilterF
        {
            protected double XF(object o)
            {
                TrackBrowser.XSegInfo.Segment = o as SySal.TotalScan.Segment;
                return Convert.ToDouble(TrackBrowser.XSegInfo.ExtendedField(Name));
            }
            public ExtendedFieldF(string n) : base(n, null, "Extended field " + n) 
            {
                F = new dFilterF(XF);
            }
        }

        public static void InstallExtendedFieldFilters()
        {
            string[] fields = TrackBrowser.XSegInfo.ExtendedFields;
            System.Collections.ArrayList gf = new ArrayList();
            foreach (string s in fields)
            {
                System.Type ty = TrackBrowser.XSegInfo.ExtendedFieldType(s);
                if (ty == typeof(int) || ty == typeof(double))
                    gf.Add(s);
            }
            if (gf.Count > 0)
            {
                FilterF[] sf = new FilterF[SegmentFilterFunctions.Length + gf.Count];
                SegmentFilterFunctions.CopyTo(sf, 0);   
                int i;
                for (i = 0; i < gf.Count; i++)
                    sf[SegmentFilterFunctions.Length + i] = new ExtendedFieldF(gf[i] as string);
                SegmentFilterFunctions = sf;
            }
        }

        internal static double fSegN(object o) { return (double)(short)((SySal.TotalScan.Segment)o).Info.Count; }
        internal static double fSegA(object o) { return (double)(int)((SySal.TotalScan.Segment)o).Info.AreaSum; }
        internal static double fSegS(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Sigma; }
        internal static double fSegSX(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Slope.X; }
        internal static double fSegSY(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Slope.Y; }
        internal static double fSegPX(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Intercept.X; }
        internal static double fSegPY(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Intercept.Y; }
        internal static double fSegPZ(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Intercept.Z; }
        internal static double fSegLayer(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.Id; }
        internal static double fSegBrickId(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.BrickId; }
        internal static double fSegSheetId(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.SheetId; }
        internal static double fSegSide(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.Side; }
        internal static double fSegLayerPos(object o) { return (double)((SySal.TotalScan.Segment)o).PosInLayer; }
        internal static double fSegTrack(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null ? -1.0 : (double)s.TrackOwner.Id; }
        internal static double fSegTrackPos(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null ? -1.0 : (double)s.PosInTrack; }
        internal static double fSegTrackN(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null ? 0.0 : (double)s.TrackOwner.Length; }
        internal static double fSegUpVN(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Upstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Upstream_Vertex.Length; }
        internal static double fSegUpVID(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Upstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Upstream_Vertex.Id; }
        internal static double fSegUpVIP(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Upstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Upstream_Impact_Parameter; }
        internal static double fSegDownVN(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Downstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Downstream_Vertex.Length; }
        internal static double fSegDownVID(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Downstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Downstream_Vertex.Id; }
        internal static double fSegDownVIP(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Downstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Downstream_Impact_Parameter; }

        static internal FilterF[] SegmentFilterFunctions = new FilterF[]
            { 
                new FilterF("N", fSegN, "Number of grains"),
                new FilterF("A", fSegA, "Total sum of the area in pixel"),
                new FilterF("S", fSegS, "Sigma"),
                new FilterF("SX", fSegSX, "X slope"),
                new FilterF("SY", fSegSY, "Y slope"),
                new FilterF("PX", fSegPX, "X position"),
                new FilterF("PY", fSegPY, "Y position"),
                new FilterF("PZ", fSegPZ, "Z position"),
                new FilterF("Brick", fSegBrickId, "Brick id"),
                new FilterF("Sheet", fSegSheetId, "Sheet id"),
                new FilterF("Side", fSegSide, "Side"),
                new FilterF("Layer", fSegLayer, "Layer id"),
                new FilterF("LPos", fSegLayerPos, "Position in layer"),
                new FilterF("Track", fSegTrack, "Track id"),
                new FilterF("TrackPos", fSegTrackPos, "Position in track"),
                new FilterF("NT", fSegTrackN, "Segments in the track"),
                new FilterF("UVID", fSegUpVID, "Id of the upstream vertex"),
                new FilterF("UVN", fSegUpVN, "Tracks at the upstream vertex"),
                new FilterF("UVIP", fSegUpVIP, "Upstream Impact Parameter of the owner track"),
                new FilterF("DVID", fSegDownVID, "Id of the downstream vertex"),
                new FilterF("DVN", fSegDownVN, "Tracks at the downstream vertex"),
                new FilterF("DVIP", fSegDownVIP, "Downstream Impact Parameter of the owner track")
            };        

        internal static double fTkN(object o) { return (double)((SySal.TotalScan.Track)o).Length; }
        internal static double fTkG(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; int g = 0; int i; for (i = 0; i < t.Length; i++) g += t[i].Info.Count; return (double)g; }
        internal static double fTkB(object o) { return (double)((SySal.TotalScan.Flexi.Track)o).BaseTracks.Length; }
        internal static double fTkM(object o) { SySal.TotalScan.Flexi.Track t = (SySal.TotalScan.Flexi.Track)o; return (double)(t.Length - t.BaseTracks.Length); }
        internal static double fTkID(object o) { return (double)((SySal.TotalScan.Track)o).Id; }
        internal static double fTkDSX(object o) { return (double)((SySal.TotalScan.Track)o).Downstream_SlopeX; }
        internal static double fTkDSY(object o) { return (double)((SySal.TotalScan.Track)o).Downstream_SlopeY; }
        internal static double fTkDPX(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return (t.Downstream_PosX + (t.Downstream_Z - t.Downstream_PosZ) * t.Downstream_SlopeX); }
        internal static double fTkDPY(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return (t.Downstream_PosY + (t.Downstream_Z - t.Downstream_PosZ) * t.Downstream_SlopeY); }
        internal static double fTkDPZ(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Downstream_Z; }
        internal static double fTkUSX(object o) { return (double)((SySal.TotalScan.Track)o).Upstream_SlopeX; }
        internal static double fTkUSY(object o) { return (double)((SySal.TotalScan.Track)o).Upstream_SlopeY; }
        internal static double fTkUPX(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return (t.Upstream_PosX + (t.Upstream_Z - t.Upstream_PosZ) * t.Upstream_SlopeX); }
        internal static double fTkUPY(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return (t.Upstream_PosY + (t.Upstream_Z - t.Upstream_PosZ) * t.Upstream_SlopeY); }
        internal static double fTkUPZ(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Upstream_Z; }
        internal static double fTkUpVN(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Upstream_Vertex == null ? -1.0 : (double)t.Upstream_Vertex.Length; }
        internal static double fTkUpVID(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Upstream_Vertex == null ? -1.0 : (double)t.Upstream_Vertex.Id; }
        internal static double fTkUpVIP(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Upstream_Vertex == null ? -1.0 : (double)t.Upstream_Impact_Parameter; }
        internal static double fTkDownVN(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Downstream_Vertex == null ? -1.0 : (double)t.Downstream_Vertex.Length; }
        internal static double fTkDownVID(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Downstream_Vertex == null ? -1.0 : (double)t.Downstream_Vertex.Id; }
        internal static double fTkDownVIP(object o) { SySal.TotalScan.Track t = (SySal.TotalScan.Track)o; return t.Downstream_Vertex == null ? -1.0 : (double)t.Downstream_Impact_Parameter; }


        static FilterF[] TrackFilterFunctions = new FilterF[]
            {                 
                new FilterF("N", fTkN, "Number of segments"),
                new FilterF("G", fTkG, "Total number of grains"),
                new FilterF("B", fTkB, "Number of base tracks"),
                new FilterF("M", fTkM, "Number of microtracks"),
                new FilterF("ID", fTkID, "Id of the track"),
                new FilterF("DSX", fTkDSX, "Downstream X slope"),
                new FilterF("DSY", fTkDSY, "Downstream Y slope"),
                new FilterF("DPX", fTkDPX, "Downstream X position"),
                new FilterF("DPY", fTkDPY, "Downstream Y position"),
                new FilterF("DPZ", fTkDPZ, "Downstream Z position"),
                new FilterF("USX", fTkUSX, "Upstream X slope"),
                new FilterF("USY", fTkUSY, "Upstream Y slope"),
                new FilterF("UPX", fTkUPX, "Upstream X position"),
                new FilterF("UPY", fTkUPY, "Upstream Y position"),
                new FilterF("UPZ", fTkUPZ, "Upstream Z position"),
                new FilterF("UVID", fTkUpVID, "Id of the upstream vertex"),
                new FilterF("UVN", fTkUpVN, "Tracks at the upstream vertex"),
                new FilterF("UVIP", fTkUpVIP, "Upstream Impact Parameter"),
                new FilterF("DVID", fTkDownVID, "Id of the downstream vertex"),
                new FilterF("DVN", fTkDownVN, "Tracks at the downstream vertex"),
                new FilterF("DVIP", fTkDownVIP, "Downstream Impact Parameter")
            };
        
        internal class ObjFilter
        {
            NumericalTools.Function F;
            FilterF[] FMap;

            public ObjFilter(FilterF[] flist, string fstr)
            {
                int i;
            
                    F = new NumericalTools.CStyleParsedFunction(fstr);
                    FMap = new FilterF[F.ParameterList.Length];
                    
                    for (i = 0; i < FMap.Length; i++)
                    {
                        string z = F.ParameterList[i];
                        foreach (FilterF ff1 in flist)
                            if (String.Compare(ff1.Name, z, true) == 0)
                            {
                                FMap[i] = ff1;
                                break;
                            }
                        if (FMap[i] == null) throw new Exception("Unknown parameter \"" + z + "\".");
                    }
                
            }

            public double Value(object o)
            {
                int p;
                for (p = 0; p < FMap.Length; p++)
                    F[p] = FMap[p].F(o);
                return F.Evaluate();
            }

        }

        private void cmdShowGlobalData_Click(object sender, EventArgs e)
        {
            int i, j, nl;
            string str = "Layers: " + m_V.Layers.Length + "\r\n";
            for (i = nl = 0; i < m_V.Layers.Length; i++) nl += m_V.Layers[i].Length;
            str += "Segments: " + nl + "\r\nTracks: " + m_V.Tracks.Length + "\r\nVertices: " + m_V.Vertices.Length;
            str += "\r\n\r\nAlignment data:\r\nLAYER BRICK SHEET SIDE MXX MXY MYX MYY DX DY DZ MSX MSY DSX DSY MAPN";            
            for (i = 0; i < m_V.Layers.Length; i++)
            {
                SySal.TotalScan.Layer ly = m_V.Layers[i];
                SySal.TotalScan.AlignmentData ad = ly.AlignData;
                str += "\r\n" + ly.Id + " " + ly.BrickId + " " + ly.SheetId + " " + ly.Side + " " + ad.AffineMatrixXX.ToString("F6") + " " + ad.AffineMatrixXY.ToString("F6") + " " + ad.AffineMatrixYX.ToString("F6") + " " + ad.AffineMatrixYY.ToString("F6") + " " + ad.TranslationX.ToString("F2") + " " + ad.TranslationY.ToString("F2") + " " + ad.TranslationZ.ToString("F2") + " " + ad.DShrinkX.ToString("F4") + " " + ad.DShrinkY.ToString("F4") + " " + ad.SAlignDSlopeX.ToString("F6") + " " + ad.SAlignDSlopeY.ToString("F6");
                for (j = nl = 0; j < ly.Length; j++) if (ly[j].TrackOwner != null) nl++;
                str += " " + nl;
            }
            new QBrowser("Global Data", str).ShowDialog();
        }

        DecaySearchAssistantForm DSAForm;

        private void btnDecaySearchAssistant_Click(object sender, EventArgs e)
        {
            if (DSAForm.Visible) DSAForm.Hide(); else DSAForm.Show();
        }

        #region IDecaySearchAutomation Members

        public static SySal.TotalScan.NamedAttributeIndex DSCurrentStepIndex = new SySal.TotalScan.NamedAttributeIndex("-DSA-Step-");

        public int CurrentStep
        {
            get
            {
                int[] v = PrimaryVertex;
                if (v.Length != 1) return 0;
                try
                {
                    return (int)m_V.Vertices[v[0]].GetAttribute(DSCurrentStepIndex);
                }
                catch (Exception) { return 0; }
            }
            set
            {
                int[] v = PrimaryVertex;
                if (v.Length != 1) return;
                m_V.Vertices[v[0]].SetAttribute(DSCurrentStepIndex, value);
            }
        }

        public int[] PrimaryVertex
        {
            get 
            {
                System.Collections.ArrayList ar = new ArrayList();
                int i, n;
                n = m_V.Vertices.Length;                
                for (i = 0; i < n; i++)
                    try
                    {
                        if (m_V.Vertices[i].GetAttribute(VertexBrowser.FBIsPrimaryIndex) > 0.0) ar.Add(i);
                    }
                    catch (Exception) { }
                return (int[])ar.ToArray(typeof(int));
            }
        }

        public string ErrorsInFeedbackTracks
        {
            get
            {
                string vstr = "";
                int i, n;
                n = m_V.Tracks.Length;
                bool vtxup, vtxdown, hasds;
                for (i = 0; i < n; i++)
                {
                    SySal.TotalScan.Flexi.Track tk = (SySal.TotalScan.Flexi.Track)m_V.Tracks[i];
                    try
                    {
                        if (tk.GetAttribute(TrackBrowser.FBEventIndex) > 0.0)
                        {
                            vtxup = vtxdown = hasds = false;
                            try
                            {
                                if (tk.Upstream_Vertex.GetAttribute(VertexBrowser.FBEventIndex) > 0.0) vtxup = true;
                            }
                            catch (Exception) { }
                            try
                            {
                                if (tk.Downstream_Vertex.GetAttribute(VertexBrowser.FBEventIndex) > 0.0) vtxdown = true;
                            }
                            catch (Exception) { }
                            if (vtxup == false && vtxdown == false) vstr += "\r\nTrack " + tk.Id + " has no vertex attached.";
                            try
                            {
                                if ((TrackBrowser.FBDecaySearch)(int)tk.GetAttribute(TrackBrowser.FBDecaySearchIndex) != TrackBrowser.FBDecaySearch.Null) hasds = true;
                            }
                            catch (Exception) { }
                            if (hasds == false)
                            {
                                vstr += "\r\nTrack " + tk.Id + " has no Decay Search information.";
                                BrowseTrack(tk.Id);
                            }
                        }
                    }
                    catch (Exception) { }
                }
                if (vstr.Length > 0) vstr = "Errors found:" + vstr;
                return vstr;
            }
        }

        public void FindPrimaryTracks(string postols, string slopetols)
        {
            int i, n;
            double postol = 80.0;
            double slopetol = 0.05;
            try
            {
                postol = Convert.ToDouble(postols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            try
            {
                slopetol = Convert.ToDouble(slopetols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            n = m_V.Tracks.Length;
            for (i = 0; i < n; i++)
            {
                SySal.TotalScan.Flexi.Track tk = ((SySal.TotalScan.Flexi.Track)m_V.Tracks[i]);
                if (tk.DataSet.DataType.StartsWith("SBSF"))
                {
                    SySal.TotalScan.Flexi.Track[] ptks = TrackBrowser.xMatchInOtherDataSets(m_V, tk, slopetol, postol);
                    foreach (SySal.TotalScan.Flexi.Track ptk in ptks)
                        if (ptk.DataSet.DataType.StartsWith("TSR"))                            
                            ShowTracks(new dShow(new TrackBrowser.TrackFilterWithTrack(ptk).Filter), true, false);
                }
            }            
        }

        SySal.Processing.TagPrimary.PrimaryVertexTagger m_PrimaryVtxTagger = new SySal.Processing.TagPrimary.PrimaryVertexTagger();

        public void FindPrimaryVertex(string postols, string slopetols)
        {
            int i, n;
            double postol = 80.0;
            double slopetol = 0.05;
            try
            {
                postol = Convert.ToDouble(postols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            try
            {
                slopetol = Convert.ToDouble(slopetols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }

            SySal.Processing.TagPrimary.Configuration cfg = (SySal.Processing.TagPrimary.Configuration)m_PrimaryVtxTagger.Config;
            cfg.AngularToleranceSB = slopetol;
            cfg.PositionToleranceSB = postol;
            m_PrimaryVtxTagger.Config = cfg;            

            n = m_V.Vertices.Length;
            for (i = 0; i < n; i++)            
                m_V.Vertices[i].RemoveAttribute(VertexBrowser.FBIsPrimaryIndex);
            foreach (VertexBrowser vbw in VertexBrowser.AvailableBrowsers)
                vbw.RefreshAttributeList();
            n = m_V.Tracks.Length;
            ArrayList ar = new ArrayList();
            try
            {
                int vtxid = m_PrimaryVtxTagger.ProcessData(m_V);
                if (vtxid >= 0)
                {
                    if (MessageBox.Show("One vertex found by automatic procedure: proceed to flagging?", "Automatic procedure success", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
                    {
                        VertexBrowser.xAddToFeedback((SySal.TotalScan.Flexi.Vertex)m_V.Vertices[vtxid], m_SelectedEvent);
                        m_V.Vertices[vtxid].SetAttribute(VertexBrowser.FBIsPrimaryIndex, 1.0);
                        foreach (VertexBrowser vbw in VertexBrowser.AvailableBrowsers)
                            vbw.RefreshAttributeList();
                    }
                }
                else MessageBox.Show("No primary vertex identified.", "Automatic procedure failure", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            catch (Exception xcx) 
            {
                MessageBox.Show(xcx.ToString(), "Error in automatic vertex tagging", MessageBoxButtons.OK, MessageBoxIcon.Error);                
            }            
            /*
            for (i = 0; i < n; i++)
            {
                SySal.TotalScan.Flexi.Track tk = ((SySal.TotalScan.Flexi.Track)m_V.Tracks[i]);
                if (tk.DataSet.DataType.StartsWith("SBSF"))
                {
                    SySal.TotalScan.Flexi.Track[] ptks = TrackBrowser.xMatchInOtherDataSets(m_V, tk, slopetol, postol);
                    foreach (SySal.TotalScan.Flexi.Track ptk in ptks)
                        if (ptk.DataSet.DataType.StartsWith("TSR") && ptk.Upstream_Vertex != null)
                            if (ar.Contains(ptk.Upstream_Vertex.Id) == false)
                                ar.Add(ptk.Upstream_Vertex.Id);
                }
            }
            int i1;
            for (i = 0; i < ar.Count; i++)
            {
                SySal.TotalScan.Vertex v1 = m_V.Vertices[(int)ar[i]];
                for (i1 = 0; i1 < v1.Length; i1++)
                    if (v1[i1].Upstream_Vertex == v1)
                        if (v1[i1].Downstream_Vertex != null && ar.Contains(v1[i1].Downstream_Vertex.Id))
                        {
                            ar.Remove(v1[i1].Downstream_Vertex.Id);
                            break;
                        }
                if (i1 < v1.Length)
                {
                    i = -1;
                    continue;
                }
                for (i1 = 0; i1 < v1.Length; i1++)
                    if (v1[i1].Downstream_Vertex == v1)
                        if (v1[i1].Upstream_Vertex != null && ar.Contains(v1[i1].Upstream_Vertex.Id))
                        {
                            ar.Remove(v1.Id);
                            i = -1;
                            break;
                        }
            }
            if (ar.Count == 0)
            {
                MessageBox.Show("No vertex found.", "Automatic procedure failed", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            string vf = "Id of vertices found:";
            foreach (int vi in ar)
            {
                vf += ("\r\n" + vi);
                ShowTracks(new dShow(new TrackBrowser.TrackFilterWithVertex(m_V.Vertices[vi]).Filter));
            }
            new QBrowser("Automatic procedure result", vf).ShowDialog();
            if (ar.Count == 1)
            {
                if (MessageBox.Show("One vertex found by automatic procedure: proceed to flagging?", "Automatic procedure success", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
                {
                    VertexBrowser.xAddToFeedback((SySal.TotalScan.Flexi.Vertex)m_V.Vertices[(int)ar[0]], m_SelectedEvent);
                    m_V.Vertices[(int)ar[0]].SetAttribute(VertexBrowser.FBIsPrimaryIndex, 1.0);
                    foreach (VertexBrowser vbw in VertexBrowser.AvailableBrowsers)
                        vbw.RefreshAttributeList();
                }
            }
            else MessageBox.Show("Ambiguous result from automatic vertex search.\r\nSorry, I can't help.", "Automatic procedure failed", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            */
        }

        public void BrowseTrack(int id)
        {
            TrackBrowser.Browse(m_V.Tracks[id], m_Layers, this, this, m_SelectedEvent, m_V);
        }

        public int[] CheckPrimaryTracksHoles()
        {
            System.Collections.ArrayList ar = new ArrayList();
            SySal.TotalScan.Vertex pv = null;
            int i;
            for (i = 0; i < m_V.Tracks.Length; i++)
                try
                {
                    if (m_V.Tracks[i].Upstream_Vertex.GetAttribute(VertexBrowser.FBIsPrimaryIndex) > 0.0)
                    {
                        SySal.TotalScan.Flexi.Track ftk = (SySal.TotalScan.Flexi.Track)m_V.Tracks[i];
                        if (SySal.Processing.DecaySearchVSept09.KinkSearchResult.TrackHasTooManyHoles(ftk)) ar.Add(~i);
                        else if (SySal.Processing.DecaySearchVSept09.KinkSearchResult.TrackHasHoles(ftk)) ar.Add(i);
                    }
                }
                catch (Exception) { }
            return (int[])ar.ToArray(typeof(int));
        }

        public void ComputeMomenta(string postols, string slopetols)
        {
            int i, n;
            double postol = 80.0;
            double slopetol = 0.05;
            try
            {
                postol = Convert.ToDouble(postols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            try
            {
                slopetol = Convert.ToDouble(slopetols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            n = m_V.Tracks.Length;
            for (i = 0; i < n; i++)
            {
                SySal.TotalScan.Flexi.Track tk = ((SySal.TotalScan.Flexi.Track)m_V.Tracks[i]);
                if (tk.DataSet.DataType.StartsWith("SBSF"))
                {
                    TrackBrowser.Browse(tk, m_V.Layers, this, this, m_SelectedEvent, m_V).xAutoComputeMomentum();
                    SySal.TotalScan.Flexi.Track[] ptks = TrackBrowser.xMatchInOtherDataSets(m_V, tk, slopetol, postol);
                    foreach (SySal.TotalScan.Flexi.Track ptk in ptks)
                        if (ptk.DataSet.DataType.StartsWith("TSR"))
                        {
                            double p;
                            try
                            {
                                if ((p = tk.GetAttribute(TrackBrowser.FBPIndex)) > 0.0)
                                    ptk.SetAttribute(TrackBrowser.FBPIndex, p);
                            }
                            catch (Exception) { }
                            try
                            {
                                if ((p = tk.GetAttribute(TrackBrowser.FBPMinIndex)) > 0.0)
                                    ptk.SetAttribute(TrackBrowser.FBPMinIndex, p);
                            }
                            catch (Exception) { }
                            try
                            {
                                if ((p = tk.GetAttribute(TrackBrowser.FBPMaxIndex)) > 0.0)
                                    ptk.SetAttribute(TrackBrowser.FBPMaxIndex, p);
                            }
                            catch (Exception) { }
                        }
                }
            }
            foreach (TrackBrowser tbw in TrackBrowser.AvailableBrowsers)
                tbw.RefreshAttributeList();
        }

        public void MergeManualTracks(string postols, string slopetols)
        {
            double mergeslopetol = 0.05;
            double mergepostol = 50.0;
            try
            {
                mergepostol = Convert.ToDouble(postols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            try
            {
                mergeslopetol = Convert.ToDouble(slopetols, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            ArrayList mchktks = new ArrayList();
            ArrayList tsrtks = new ArrayList();
            int i, j, n;
            n = m_V.Tracks.Length;
            for (i = 0; i < n; i++)
            {
                SySal.TotalScan.Flexi.Track tk = ((SySal.TotalScan.Flexi.Track)m_V.Tracks[i]);
                if (tk.DataSet.DataType.StartsWith("TSR"))
                    try
                    {
                        if (tk.GetAttribute(TrackBrowser.FBEventIndex) > 0.0) tsrtks.Add(tk);
                    }
                    catch (Exception) {};
            }
            for (i = 0; i < m_V.Layers.Length; i++)
            {                
                SySal.TotalScan.Layer ly = m_V.Layers[i];
                n = m_V.Layers[i].Length;
                for (j = 0; j < n; j++)
                {
                    SySal.TotalScan.Flexi.Segment s = (SySal.TotalScan.Flexi.Segment)ly[j];
                    if (s.DataSet.DataType.StartsWith("MAN")) mchktks.Add(s);
                }
            }            
            SySal.TotalScan.Flexi.Track [] mergeassoc = new SySal.TotalScan.Flexi.Track[mchktks.Count];
            double[] mergeq = new double[mchktks.Count];
            double dx, dy, ds;
            for (i = 0; i < mergeassoc.Length; i++)            
                mergeq[i] = mergeslopetol;
            int changed;
            do
            {
                changed = 0;
                for (i = 0; i < mergeq.Length; i++)
                {
                    SySal.TotalScan.Flexi.Segment s = (SySal.TotalScan.Flexi.Segment)mchktks[i];
                    SySal.Tracking.MIPEmulsionTrackInfo sinfo = s.Info;
                    foreach (SySal.TotalScan.Flexi.Track tk in tsrtks)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo finfo = tk.Fit(s.LayerOwner.Id, s.LayerOwner.RefCenter.Z);
                        dx = finfo.Slope.X - sinfo.Slope.X;
                        if (Math.Abs(dx) > mergeq[i]) continue;
                        dy = finfo.Slope.Y - sinfo.Slope.Y;
                        if (Math.Abs(dy) > mergeq[i]) continue;
                        ds = dx * dx + dy * dy;
                        if (ds > mergeq[i] * mergeq[i]) continue;
                        ds = Math.Sqrt(ds);
                        if (mergeq[i] <= ds) continue;
                        dx = finfo.Intercept.X - sinfo.Intercept.X;
                        if (Math.Abs(dx) > mergepostol) continue;
                        dy = finfo.Intercept.Y - sinfo.Intercept.Y;
                        if (Math.Abs(dy) > mergepostol) continue;
                        if (dx * dx + dy * dy > mergepostol * mergepostol) continue;
                        bool stop = false;
                        if (mergeassoc[i] != tk)
                        {
                            for (j = 0; j < mergeq.Length; j++)
                            {
                                SySal.TotalScan.Segment s1 = (SySal.TotalScan.Flexi.Segment)mchktks[j];
                                if (s1 != s && s1.LayerOwner.Id == s.LayerOwner.Id && mergeassoc[j] == tk)
                                    if (mergeq[j] <= ds)
                                    {
                                        stop = true;
                                        break;
                                    }
                                    else
                                    {
                                        changed++;
                                        mergeassoc[j] = null;
                                        mergeq[j] = mergeslopetol;
                                        break;
                                    }
                            }
                            if (stop) continue;
                            changed++;
                            mergeassoc[i] = tk;
                            mergeq[i] = ds;
                            break;
                        }
                    }
                }
            }
            while (changed > 0);
            string vstr;
            string[] vs = new string[mergeq.Length];
            for (i = 0; i < mergeq.Length; i++)
            {
                SySal.TotalScan.Flexi.Segment s = (SySal.TotalScan.Flexi.Segment)mchktks[i];                
                if (mergeassoc[i] == null) vs[i] = s.TrackOwner.Id + " " + s.Info.Slope.X.ToString("F4") + " " + s.Info.Slope.Y.ToString("F4");
                else
                {
                    SySal.Tracking.MIPEmulsionTrackInfo sinfo = s.Info;
                    SySal.TotalScan.Flexi.Track tk = mergeassoc[i];
                    SySal.Tracking.MIPEmulsionTrackInfo finfo = tk.Fit(s.LayerOwner.Id, s.LayerOwner.RefCenter.Z);
                    for (j = 0; j < tk.Length && tk[j].LayerOwner != s.LayerOwner; j++) ;
                    if (j < tk.Length) vs[i] = s.TrackOwner.Id + " " + s.Info.Slope.X.ToString("F4") + " " + s.Info.Slope.Y.ToString("F4") + " " + tk.Id + " " + s.LayerOwner.Id + " " + mergeq[i].ToString("F4") + " " + (sinfo.Intercept.X - finfo.Intercept.X).ToString("F1") + " " + (sinfo.Intercept.Y - finfo.Intercept.Y).ToString("F1") + " " + tk[j].Info.Slope.X.ToString("F4") + " " + tk[j].Info.Slope.Y.ToString("F4") + " " + tk[j].Info.Count;
                    else vs[i] = s.TrackOwner.Id + " " + s.Info.Slope.X.ToString("F4") + " " + s.Info.Slope.Y.ToString("F4") + " " + tk.Id + " " + s.LayerOwner.Id + " " + mergeq[i].ToString("F4") + " " + (sinfo.Intercept.X - finfo.Intercept.X).ToString("F1") + " " + (sinfo.Intercept.Y - finfo.Intercept.Y).ToString("F1");
                }
            }
            SegReplaceForm srf = new SegReplaceForm();
            srf.Replacements = vs;
            if (srf.ShowDialog() == DialogResult.OK)
            {
                for (i = 0; i < mergeq.Length; i++)
                {
                    if (srf.Confirmed[i] && mergeassoc[i] != null)
                    {
                        SySal.TotalScan.Flexi.Segment s = (SySal.TotalScan.Flexi.Segment)mchktks[i];
                        SySal.TotalScan.Track mtk = s.TrackOwner;
                        SySal.TotalScan.Flexi.Track tk = mergeassoc[i];
                        for (j = 0; j < tk.Length && tk[j].LayerOwner != s.LayerOwner; j++) ;
                        if (j < tk.Length) tk.RemoveSegments(new int[1] { j });
                        SySal.TotalScan.Flexi.Segment ns = new SySal.TotalScan.Flexi.Segment(s, ((SySal.TotalScan.Flexi.Segment)s).DataSet);
                        ((SySal.TotalScan.Flexi.Layer)s.LayerOwner).Add(new SySal.TotalScan.Flexi.Segment[1] { ns });
                        tk.AddSegment(ns);
                        if (mtk != null)
                        {
                            SySal.TotalScan.Attribute[] attrl = mtk.ListAttributes();
                            foreach (SySal.TotalScan.Attribute a in attrl)
                            {
                                double attribval = 0.0;
                                try
                                {
                                    if (tk.GetAttribute(a.Index) > 0.0) continue;
                                }
                                catch (Exception) { }
                                tk.SetAttribute(a.Index, a.Value);
                            }
                        }
                        tk.NotifyChanged();                        
                        if (tk.Upstream_Vertex != null) tk.Upstream_Vertex.NotifyChanged();
                        if (tk.Downstream_Vertex != null) tk.Downstream_Vertex.NotifyChanged();
                        foreach (TrackBrowser tb in TrackBrowser.AvailableBrowsers)
                            if (tb.Track == tk)
                                tb.Track = tk;
                    }
                }
                MessageBox.Show("Replacements done.\r\nPlease regenerate the plot to see the changes.", "Success");

                if (MessageBox.Show("Remove from feedback all the tracks that had no matching manual segment?\r\nThis operation cannot be undone.", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
                {
                    int removed = 0;
                    vstr = "Tracks removed:";
                    ArrayList va = new ArrayList();
                    foreach (SySal.TotalScan.Flexi.Track tk in tsrtks)
                    {
                        for (i = 0; i < mergeassoc.Length && mergeassoc[i] != tk; i++) ;
                        if (i == mergeassoc.Length)
                            try
                            {
                                tk.RemoveAttribute(TrackBrowser.FBEventIndex);
                                if (tk.Upstream_Vertex != null)
                                {
                                    if (va.Contains(tk.Upstream_Vertex) == false) va.Add(tk.Upstream_Vertex);
                                    tk.Upstream_Vertex.RemoveTrack(tk);
                                }
                                if (tk.Downstream_Vertex != null)
                                {
                                    if (va.Contains(tk.Downstream_Vertex) == false) va.Add(tk.Downstream_Vertex);
                                    tk.Downstream_Vertex.RemoveTrack(tk);
                                }
                                vstr += "\r\n" + tk.Id;
                                removed++;
                            }
                            catch (Exception) { }
                    }
                    vstr += "\r\n" + removed + " track(s) removed.";
                    if (va.Count > 0)
                    {
                        vstr += "\r\nVertices altered:";
                        foreach (SySal.TotalScan.Vertex vtx in va)
                            vstr += "\r\n" + vtx.Id;
                        vstr += "\r\nPlease regenerate the plot to see the results.";
                    }
                    new QBrowser("Track removal", vstr).ShowDialog();
                }
            }
        }

        public void IsolatedTrackEventExtraTrackSearch()
        {
            int [] pv = PrimaryVertex;
            if (pv.Length != 1)
            {
                MessageBox.Show("Primary vertex not defined!\r\nYou might need to define a fictitious vertex upstream of an isolated muon or hadron track.", "Required information missing", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                return;
            }
            ShowTracks(new SySal.Processing.DecaySearchVSept09.IsolatedTrackEventExtraTrackFilter(m_V.Vertices[pv[0]]).Filter, true, false);
        }

        public void OneMuOrMultiProngZeroMuEventExtraTrackSearch()
        {
            int[] pv = PrimaryVertex;
            if (pv.Length != 1)
            {
                MessageBox.Show("Primary vertex not defined!", "Required information missing", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                return;
            }
            ShowTracks(new SySal.Processing.DecaySearchVSept09.OneMuOrMultiProngZeroEventExtraTrackFilter(m_V.Vertices[pv[0]]).Filter, true, false);
        }

        public void ZeroMu123ProngEventExtraTrackSearch()
        {
            int[] pv = PrimaryVertex;
            if (pv.Length != 1)
            {
                MessageBox.Show("Primary vertex not defined!", "Required information missing", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                return;
            }
            ShowTracks(new SySal.Processing.DecaySearchVSept09.ZeroMu123ProngEventExtraTrackFilter(m_V.Vertices[pv[0]]).Filter, true, false);
        }

        #endregion

        private void cmdRecStart_Click(object sender, EventArgs e)
        {
            gdiDisplay1.StartMovie(10, m_MovieKB);
        }

        private void cmdRecStop_Click(object sender, EventArgs e)
        {
            GDI3D.Movie mv = gdiDisplay1.StopMovie(null);
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Please select file to save to.";
            sdlg.Filter = "Animated GIF files (*.gif)|*.gif";
            if (sdlg.ShowDialog() == DialogResult.OK)
            {
                mv.Save(sdlg.FileName);
            }
        }

        int m_MovieKB = 128;

        private void OnMovieKBLeave(object sender, EventArgs e)
        {
            try
            {
                m_MovieKB = Convert.ToInt32(txtMovieKB.Text);
                if (m_MovieKB <= 0) throw new Exception();
            }
            catch (Exception)
            {
                txtMovieKB.Text = m_MovieKB.ToString();
            }
        }

        MovieForm MVForm;

        private void cmdMovie_Click(object sender, EventArgs e)
        {
            MVForm.Visible = !MVForm.Visible;
        }

        private void btnFilterAdd_Click(object sender, EventArgs e)
        {
            string s = cmbFilter.Text.Trim();
            if (s.Length > 0)
            {
                foreach (string s1 in UserProfileInfo.ThisProfileInfo.DisplayFilters)
                    if (String.Compare(s1, s, true) == 0) return;
                cmbFilter.Items.Insert(0, s);
                string [] os = UserProfileInfo.ThisProfileInfo.DisplayFilters;
                UserProfileInfo.ThisProfileInfo.DisplayFilters = new string [os.Length + 1];
                UserProfileInfo.ThisProfileInfo.DisplayFilters[0] = s;
                os.CopyTo(UserProfileInfo.ThisProfileInfo.DisplayFilters, 1);
                MainForm.SaveProfile();
            }
        }

        private void btnFilterDel_Click(object sender, EventArgs e)
        {
            string s = cmbFilter.Text.Trim();
            if (s.Length > 0)
            {
                int i, j;
                for (i = 0; i < UserProfileInfo.ThisProfileInfo.DisplayFilters.Length && String.Compare(s, UserProfileInfo.ThisProfileInfo.DisplayFilters[i], true) != 0; i++) ;
                if (i < UserProfileInfo.ThisProfileInfo.DisplayFilters.Length)
                {
                    cmbFilter.Items.RemoveAt(i);
                    string[] os = UserProfileInfo.ThisProfileInfo.DisplayFilters;
                    UserProfileInfo.ThisProfileInfo.DisplayFilters = new string[os.Length - 1];
                    for (j = 0; j < i; j++) UserProfileInfo.ThisProfileInfo.DisplayFilters[j] = os[j];
                    for (++j; j < os.Length; j++) UserProfileInfo.ThisProfileInfo.DisplayFilters[j - 1] = os[j];
                    MainForm.SaveProfile();
                }
            }
        }
    }
}
