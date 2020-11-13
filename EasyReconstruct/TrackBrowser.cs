using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
	/// <summary>
	/// TrackBrowser - shows information about a volume track.
	/// </summary>
	/// <remarks>
	/// <para>The upper group shows the track parameters, which can be dumped to an ASCII file.</para>
	/// <para>The lower group shows the segments, and they can be dumped to an ASCII file.</para>
	/// <para>A comment can be set to a track. <b>NOTICE: if <c>NOVERTEX</c> is set, the track is excluded from any possible vertex the next time topological reconstruction is started.</b></para>
	/// <para>Two buttons allow one to navigate from a track to its upstream/downstream vertex (if any).</para>
	/// </remarks>
	public class TrackBrowser : System.Windows.Forms.Form
	{
        public static SySal.TotalScan.IMCSMomentumEstimator[] MCSAlgorithms;

        public static ExtendedSegInfoProvider XSegInfo = new ExtendedSegInfoProvider();

		private System.Windows.Forms.ListView GeneralList;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ListView SegmentList;
		private System.Windows.Forms.ColumnHeader columnHeader5;
		private System.Windows.Forms.ColumnHeader columnHeader6;
		private System.Windows.Forms.ColumnHeader columnHeader7;
		private System.Windows.Forms.ColumnHeader columnHeader8;
		private System.Windows.Forms.ColumnHeader columnHeader9;
		private System.Windows.Forms.ColumnHeader columnHeader10;
		private System.Windows.Forms.ColumnHeader columnHeader11;
		private System.Windows.Forms.ColumnHeader columnHeader12;
		private System.Windows.Forms.Button SetCommentButton;
		private System.Windows.Forms.TextBox CommentText;
		private System.Windows.Forms.Button GoUpVtxButton;
		private System.Windows.Forms.Button GoDownVtxButton;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.Button GeneralSelButton;
		private System.Windows.Forms.TextBox GeneralDumpFileText;
		private System.Windows.Forms.Button GeneralDumpFileButton;
		private System.Windows.Forms.Button SegDumpButton;
		private System.Windows.Forms.TextBox SegDumpFileText;
		private System.Windows.Forms.Button SegDumpSelButton;
		private System.Windows.Forms.ColumnHeader columnHeader4;
		private System.Windows.Forms.ColumnHeader columnHeader13;
		private System.Windows.Forms.ColumnHeader columnHeader14;
        private ColumnHeader columnHeader15;
        private ColumnHeader columnHeader16;
        private ColumnHeader columnHeader17;
        private ColumnHeader columnHeader18;
        private ColumnHeader columnHeader19;
        private ListBox LayerList;
        private Label label1;
        private TextBox InfoText;
        private Label label2;
        private TextBox OrigInfoText;
        private Label label3;

		private SySal.TotalScan.Track m_Track = null;
        private Button SelectIPFirstButton;
        private Button SelectIPSecondButton;

        private SySal.Executables.EasyReconstruct.IPSelector SelectForIP;
        private Button btnShowRelatedTracks;
        private Label label4;
        private TextBox txtOpening;
        private TextBox txtRadius;
        private Label label5;
        private TextBox txtDeltaSlope;
        private Label label6;
        private TextBox txtDeltaZ;
        private Label label7;

        private SySal.Executables.EasyReconstruct.TrackSelector SelectForGraph;
        private RadioButton radioUpstream;
        private RadioButton radioDownstream;
        private GroupBox groupBox4;
        private GroupBox groupBox5;
        private RadioButton radioUpstreamDir;
        private RadioButton radioDownstreamDir;
        private Button btnShowRelatedSegments;
        private Button btnScanbackScanforth;
        private TextBox txtMaxMisses;
        private Label label8;
        private TextBox LabelText;
        private Button SetLabelButton;
        private CheckBox HighlightCheck;
        private CheckBox EnableLabelCheck;
        private TabControl tabControl1;
        private TabPage tabPage1;
        private TabPage tabPage2;
        private TabPage tabPage3;
        private TabPage tabPage4;
        private TabPage tabPage5;
        private TextBox txtExtrapolationDist;
        private Button AddUpButton;
        private Button AddDownButton;
        private ListBox ListFits;
        private Label label9;
        private RadioButton radioUseLastSeg;
        private RadioButton radioUseFit;
        private Button WeightFromQualityButton;
        private Label label10;
        private TextBox txtWeight;
        private Button MomentumFromAttrButton;
        private Label label11;
        private TextBox txtMomentum;
        private TextBox txtDataSet;
        private Label label12;
        private TabPage tabPage6;
        private TextBox textIgnoreDeltaSlope;
        private TextBox textMomentumResult;
        private Button ExportButton;
        private Button buttonIgnoreDeltaSlope;
        private Label label13;
        private TextBox textMeasIgnoreGrains;
        private ListView SlopeList;
        private ColumnHeader columnHeader26;
        private ColumnHeader columnHeader27;
        private ColumnHeader columnHeader28;
        private ColumnHeader columnHeader29;
        private Button buttonMeasIgnoreGrains;
        private Button buttonMeasSelAll;
        private ColumnHeader columnHeader20;
        private ColumnHeader columnHeader21;
        private ColumnHeader columnHeader22;
        private ColumnHeader columnHeader23;
        private ColumnHeader columnHeader24;
        private ListBox MomentumFitterList;
        private ColumnHeader columnHeader30;
        private ColumnHeader columnHeader31;
        private ColumnHeader columnHeader32;
        private ComboBox AlgoCombo;
        private Button MCSAnnecyComputeButton;
        private Button SegRemoveButton;
        private TabPage tabPage7;
        private Button cmdAddSetAttribute;
        private Button cmdRemoveAttributes;
        private ListView AttributeList;
        private ColumnHeader columnHeader25;
        private ColumnHeader columnHeader34;
        private TextBox txtAttrValue;
        private TextBox txtAttrName;
        private TabPage tabPage8;
        private Label label17;
        private Label label16;
        private ComboBox cmbFBOutOfBrick;
        private Label label15;
        private ComboBox cmbFBDarkness;
        private Label label14;
        private ComboBox cmbFBParticle;
        private CheckBox chkFBScanback;
        private CheckBox chkFBEvent;
        private ComboBox cmbFBLastPlate;
        private Button btnFBFindSBSFTrack;
        private ComboBox cmbFBTkImportList;
        private TextBox txtFBSlopeTol;
        private Label label18;
        private CheckBox chkFBManual;
        private Button btnFBImportAttributes;
        private Button btnFBSingleProngVertex;
        private TextBox txtFBSingleProngVertexZ;
        private Button AppendToSelButton;
        private TextBox AppendToFileText;
        private Button AppendToButton;
        private TextBox DSlopeRText;
        private Label label20;
        private TextBox DSlopeRMSText;
        private Label label19;
        private TextBox KinkPlateText;
        private Label label21;

        private SySal.Processing.DecaySearchVSept09.KinkSearchResult m_KinkSearchResult;
        private Label label22;
        private TextBox txtIP;
        private TextBox txtAttribImportTk;
        private ComboBox cmbAttribImport;
        private Button cmdImportAttribute;
        private TextBox txtAttribImportValue;
        private Button SegAddReplaceButton;
        private Label label23;
        private ComboBox cmbFBDecaySearch;
        private Button btnBrowseTrack;
        private ComboBox cmbMatchOtherDS;
        private Button btnMatchInOtherDatasets;
        private Button PlotButton;
        private Label label24;
        private TextBox UpStopsText;
        private Label label25;
        private TextBox DownStopsText;
        private Label label26;
        private TextBox UpSegsText;
        private Label label27;
        private TextBox DownSegsText;
        private Button AppendUpButton;
        private Button AppendDownButton;
        private TextBox KinkSearchXText;
        private Label label28;
        private Button SplitTrackButton;
        private CheckBox chkSplitWithVertex;
        private Button CheckAllLayers;
        private Button CheckMissingBasetracksButton;
        private Button cmdSpecialAttributesReference;
        private Button LaunchScanButton;
        private CheckBox chkBroadcastAction;
        private RadioButton rdBoth;
        private RadioButton rdDataset;
        private RadioButton rdShow;
        private ColumnHeader columnHeader33;

		public SySal.TotalScan.Track Track
		{
			get { return m_Track; }
			set 
			{
                int i;
				m_Track = value;
                /*
                SySal.Tracking.MIPEmulsionTrackInfo[] bts = ((SySal.TotalScan.Flexi.Track)m_Track).BaseTracks;
                SySal.TotalScan.Segment[] btssegs = new SySal.TotalScan.Segment[bts.Length];
                for (i = 0; i < bts.Length; i++)                    
                    btssegs[i] = new SySal.TotalScan.Segment(bts[i], new SySal.TotalScan.NullIndex());                
                m_KinkSearchResult = new SySal.Processing.AlphaOmegaReconstruction.KinkSearchResult(btssegs);
                 */
                SySal.Tracking.MIPEmulsionTrackInfo[] ksbt = ((SySal.TotalScan.Flexi.Track)m_Track).BaseTracks;
                if (m_Track.Upstream_Vertex == null) m_KinkSearchResult = new SySal.Processing.DecaySearchVSept09.KinkSearchResult();
                else
                {
                    m_KinkSearchResult = new SySal.Processing.DecaySearchVSept09.KinkSearchResult((SySal.TotalScan.Flexi.Track)m_Track);
                    if (m_KinkSearchResult.KinkIndex >= 0) m_KinkSearchResult.KinkIndex = m_Layers[m_KinkSearchResult.KinkIndex].SheetId;
                }
                //if (m_KinkSearchResult.KinkIndex >= 0) m_KinkSearchResult.KinkIndex = (int)(bts[m_KinkSearchResult.KinkIndex].Field);                
                DSlopeRMSText.Text = m_KinkSearchResult.TransverseSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "/" + m_KinkSearchResult.LongitudinalSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture);
                DSlopeRText.Text = m_KinkSearchResult.TransverseMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "/" + m_KinkSearchResult.LongitudinalMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture);
                KinkPlateText.Text = m_KinkSearchResult.KinkDelta.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "/" + m_KinkSearchResult.KinkIndex;
                KinkPlateText.BackColor = (m_KinkSearchResult.KinkDelta >= 3.0) ? Color.Coral : DSlopeRMSText.BackColor;
                KinkSearchXText.Text = (m_KinkSearchResult.ExceptionMessage == null) ? "" : m_KinkSearchResult.ExceptionMessage;
                ListViewItem lvw = null;
				this.Text = "TrackBrowser #" + m_Track.Id;
                txtDataSet.Text = ((SySal.TotalScan.Flexi.Track)m_Track).DataSet.ToString();
				CommentText.Text = (m_Track.Comment == null) ? "" : m_Track.Comment;
				GeneralList.Items.Clear();
				GeneralList.Items.Add("ID").SubItems.Add(m_Track.Id.ToString());
				GeneralList.Items.Add("Segments").SubItems.Add(m_Track.Length.ToString());
				GeneralList.Items.Add("Comment").SubItems.Add((m_Track.Comment == null) ? "" : m_Track.Comment);
				GeneralList.Items.Add("Upstream Z").SubItems.Add(m_Track.Upstream_Z.ToString("F1"));
                GeneralList.Items.Add("Upstream Vtx").SubItems.Add((m_Track.Upstream_Vertex == null) ? "" : m_Track.Upstream_Vertex.Id.ToString());
                lvw = GeneralList.Items.Add("Upstream IP");
                try
                {
                    lvw.SubItems.Add((m_Track.Upstream_Vertex == null) ? "" : m_Track.Upstream_Impact_Parameter.ToString("F3"));
                }
                catch (Exception)
                {
                    lvw.SubItems.Add("N/A");
                }				
				GeneralList.Items.Add("Upstream Slope X").SubItems.Add(m_Track.Upstream_SlopeX.ToString("F5"));
				GeneralList.Items.Add("Upstream Slope Y").SubItems.Add(m_Track.Upstream_SlopeY.ToString("F5"));
				GeneralList.Items.Add("Upstream Pos X").SubItems.Add((m_Track.Upstream_SlopeX * (m_Track.Upstream_Z - m_Track.Upstream_PosZ) + m_Track.Upstream_PosX).ToString("F1"));
				GeneralList.Items.Add("Upstream Pos Y").SubItems.Add((m_Track.Upstream_SlopeY * (m_Track.Upstream_Z - m_Track.Upstream_PosZ) + m_Track.Upstream_PosY).ToString("F1"));
				GeneralList.Items.Add("Downstream Z").SubItems.Add(m_Track.Downstream_Z.ToString("F1"));
				GeneralList.Items.Add("Downstream Vtx").SubItems.Add((m_Track.Downstream_Vertex == null) ? "" : m_Track.Downstream_Vertex.Id.ToString());                
                lvw = GeneralList.Items.Add("Downstream IP");
                try
                {
                    lvw.SubItems.Add((m_Track.Downstream_Vertex == null) ? "" : m_Track.Downstream_Impact_Parameter.ToString("F3"));
                }
                catch (Exception)
                {
                    lvw.SubItems.Add("N/A");
                }
				GeneralList.Items.Add("Downstream Slope X").SubItems.Add(m_Track.Downstream_SlopeX.ToString("F5"));
				GeneralList.Items.Add("Downstream Slope Y").SubItems.Add(m_Track.Downstream_SlopeY.ToString("F5"));
				GeneralList.Items.Add("Downstream Pos X").SubItems.Add((m_Track.Downstream_SlopeX * (m_Track.Downstream_Z - m_Track.Downstream_PosZ) + m_Track.Downstream_PosX).ToString("F1"));
				GeneralList.Items.Add("Downstream Pos Y").SubItems.Add((m_Track.Downstream_SlopeY * (m_Track.Downstream_Z - m_Track.Downstream_PosZ) + m_Track.Downstream_PosY).ToString("F1"));
                foreach (SySal.TotalScan.Attribute attr in m_Track.ListAttributes())                
                    GeneralList.Items.Add("@ " + attr.Index.ToString()).SubItems.Add(attr.Value.ToString());                    
				GoUpVtxButton.Enabled = (m_Track.Upstream_Vertex != null);
				GoDownVtxButton.Enabled = (m_Track.Downstream_Vertex != null);
				SegmentList.Items.Clear();
                SlopeList.Items.Clear();
				for (i = 0; i < m_Track.Length; i++)
				{
                    XSegInfo.Segment = m_Track[i];
					SySal.Tracking.MIPEmulsionTrackInfo info = m_Track[i].Info;
					System.Windows.Forms.ListViewItem lvi = SegmentList.Items.Add(i.ToString());
					lvi.SubItems.Add(m_Track[i].PosInLayer.ToString());
					lvi.SubItems.Add(info.Count.ToString());
					lvi.SubItems.Add(info.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(info.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(info.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(info.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(info.Intercept.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(m_Track[i].LayerOwner.UpstreamZ.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(m_Track[i].LayerOwner.DownstreamZ.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
					lvi.SubItems.Add(m_Track[i].LayerOwner.Id.ToString());
					lvi.SubItems.Add(m_Track[i].LayerOwner.SheetId.ToString());
                    lvi.SubItems.Add((m_Track[i].Index == null) ? "-" : (m_Track[i].Index.ToString()));
                    info = m_Track[i].OriginalInfo;
                    lvi.SubItems.Add(info.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(info.Intercept.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(info.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(info.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(m_Track[i].LayerOwner.BrickId.ToString());
                    lvi.SubItems.Add(m_Track[i].LayerOwner.Side.ToString());
                    lvi.SubItems.Add(m_Track[i].Info.Sigma.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
                    info.Field = (uint)m_Track[i].LayerOwner.Id;
                    foreach (string s in XSegInfo.ExtendedFields)
                        lvi.SubItems.Add(XSegInfo.ExtendedField(s).ToString());
                }
                SySal.Tracking.MIPEmulsionTrackInfo[] btslopes = ((SySal.TotalScan.Flexi.Track)m_Track).BaseTracks;
                for (i = 0; i < btslopes.Length; i++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = btslopes[i];                    
                    System.Windows.Forms.ListViewItem lvi1 = SlopeList.Items.Add(info.Field.ToString());
                    lvi1.SubItems.Add(info.Intercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    lvi1.SubItems.Add(info.Count.ToString());
                    lvi1.SubItems.Add(info.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi1.SubItems.Add(info.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));                    
                    lvi1.Checked = true;
                    lvi1.Tag = info;                    
				}
                LayerList.Items.Clear();
                for (i = 0; i < m_Layers.Length; i++)
                    LayerList.Items.Add("L " + i + " B " + m_Layers[i].BrickId + " P " + m_Layers[i].SheetId + " Z " + m_Layers[i].RefCenter.Z.ToString("F1"));
                if (m_Track.Upstream_Vertex == null || m_Track.Downstream_Vertex == null)
                {
                    btnFBSingleProngVertex.Visible = true;
                    txtFBSingleProngVertexZ.Visible = true;
                    if (m_Track.Upstream_Vertex == null) m_SingleProngVertexZ = -750.0;
                    else m_SingleProngVertexZ = +550.0;
                    txtFBSingleProngVertexZ.Text = m_SingleProngVertexZ.ToString(System.Globalization.CultureInfo.InvariantCulture);
                }
                RefreshAttributeList();
			}
		}

        NumericalTools.Likelihood m_MomentumLikelihood = null;

        internal void SetMomentum(NumericalTools.Likelihood lk, double pmin, double pmax, double cl)
        {
            m_Track.SetAttribute(FBPIndex, lk.Best(0));
            m_Track.SetAttribute(FBPMinIndex, pmin);
            m_Track.SetAttribute(FBPMaxIndex, pmax);
            RefreshAttributeList();
            string pstr = "@ P";
            foreach (ListViewItem lvi in GeneralList.Items)
                if (String.Compare(lvi.SubItems[0].Text, pstr, true) == 0)
                {
                    lvi.SubItems[1].Text = lk.Best(0).ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                    pstr = null;
                    break;
                }
            if (pstr != null)
                GeneralList.Items.Add(pstr).SubItems.Add(lk.Best(0).ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
            m_Momentum = lk.Best(0);
            txtMomentum.Text = m_Momentum.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            m_MomentumLikelihood = lk;
            textMomentumResult.Text = m_Momentum.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "; " +
                pmin.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "; " +
                pmax.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "; " +
                cl.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
        }

        SySal.TotalScan.Volume.LayerList m_Layers;

        SySal.TotalScan.Volume m_V;

        long m_Event;

		protected TrackBrowser(SySal.TotalScan.Volume.LayerList ll, long ev, SySal.TotalScan.Volume v)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//            
            m_Layers = ll;
            m_Event = ev;
            m_V = v;
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
            this.GeneralList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.SegmentList = new System.Windows.Forms.ListView();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader12 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader14 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader13 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader15 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader16 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader17 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader18 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader19 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader31 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader32 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader33 = new System.Windows.Forms.ColumnHeader();
            this.GeneralDumpFileButton = new System.Windows.Forms.Button();
            this.GeneralDumpFileText = new System.Windows.Forms.TextBox();
            this.GeneralSelButton = new System.Windows.Forms.Button();
            this.SegDumpButton = new System.Windows.Forms.Button();
            this.SetCommentButton = new System.Windows.Forms.Button();
            this.CommentText = new System.Windows.Forms.TextBox();
            this.GoUpVtxButton = new System.Windows.Forms.Button();
            this.GoDownVtxButton = new System.Windows.Forms.Button();
            this.SegDumpFileText = new System.Windows.Forms.TextBox();
            this.SegDumpSelButton = new System.Windows.Forms.Button();
            this.OrigInfoText = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.InfoText = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.LayerList = new System.Windows.Forms.ListBox();
            this.label1 = new System.Windows.Forms.Label();
            this.SelectIPFirstButton = new System.Windows.Forms.Button();
            this.SelectIPSecondButton = new System.Windows.Forms.Button();
            this.btnShowRelatedTracks = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.txtOpening = new System.Windows.Forms.TextBox();
            this.txtRadius = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtDeltaSlope = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtDeltaZ = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.radioUpstream = new System.Windows.Forms.RadioButton();
            this.radioDownstream = new System.Windows.Forms.RadioButton();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.radioUpstreamDir = new System.Windows.Forms.RadioButton();
            this.radioDownstreamDir = new System.Windows.Forms.RadioButton();
            this.btnShowRelatedSegments = new System.Windows.Forms.Button();
            this.btnScanbackScanforth = new System.Windows.Forms.Button();
            this.txtMaxMisses = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.LabelText = new System.Windows.Forms.TextBox();
            this.SetLabelButton = new System.Windows.Forms.Button();
            this.HighlightCheck = new System.Windows.Forms.CheckBox();
            this.EnableLabelCheck = new System.Windows.Forms.CheckBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.KinkSearchXText = new System.Windows.Forms.TextBox();
            this.label28 = new System.Windows.Forms.Label();
            this.PlotButton = new System.Windows.Forms.Button();
            this.KinkPlateText = new System.Windows.Forms.TextBox();
            this.label21 = new System.Windows.Forms.Label();
            this.DSlopeRText = new System.Windows.Forms.TextBox();
            this.label20 = new System.Windows.Forms.Label();
            this.DSlopeRMSText = new System.Windows.Forms.TextBox();
            this.label19 = new System.Windows.Forms.Label();
            this.txtDataSet = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.chkSplitWithVertex = new System.Windows.Forms.CheckBox();
            this.SplitTrackButton = new System.Windows.Forms.Button();
            this.SegAddReplaceButton = new System.Windows.Forms.Button();
            this.SegRemoveButton = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.LaunchScanButton = new System.Windows.Forms.Button();
            this.CheckAllLayers = new System.Windows.Forms.Button();
            this.CheckMissingBasetracksButton = new System.Windows.Forms.Button();
            this.AppendUpButton = new System.Windows.Forms.Button();
            this.AppendDownButton = new System.Windows.Forms.Button();
            this.label24 = new System.Windows.Forms.Label();
            this.UpStopsText = new System.Windows.Forms.TextBox();
            this.label25 = new System.Windows.Forms.Label();
            this.DownStopsText = new System.Windows.Forms.TextBox();
            this.label26 = new System.Windows.Forms.Label();
            this.UpSegsText = new System.Windows.Forms.TextBox();
            this.label27 = new System.Windows.Forms.Label();
            this.DownSegsText = new System.Windows.Forms.TextBox();
            this.AppendToSelButton = new System.Windows.Forms.Button();
            this.AppendToFileText = new System.Windows.Forms.TextBox();
            this.AppendToButton = new System.Windows.Forms.Button();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.rdBoth = new System.Windows.Forms.RadioButton();
            this.rdDataset = new System.Windows.Forms.RadioButton();
            this.rdShow = new System.Windows.Forms.RadioButton();
            this.chkBroadcastAction = new System.Windows.Forms.CheckBox();
            this.btnBrowseTrack = new System.Windows.Forms.Button();
            this.cmbMatchOtherDS = new System.Windows.Forms.ComboBox();
            this.btnMatchInOtherDatasets = new System.Windows.Forms.Button();
            this.label22 = new System.Windows.Forms.Label();
            this.txtIP = new System.Windows.Forms.TextBox();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.MomentumFromAttrButton = new System.Windows.Forms.Button();
            this.label11 = new System.Windows.Forms.Label();
            this.txtMomentum = new System.Windows.Forms.TextBox();
            this.WeightFromQualityButton = new System.Windows.Forms.Button();
            this.label10 = new System.Windows.Forms.Label();
            this.txtWeight = new System.Windows.Forms.TextBox();
            this.radioUseLastSeg = new System.Windows.Forms.RadioButton();
            this.radioUseFit = new System.Windows.Forms.RadioButton();
            this.ListFits = new System.Windows.Forms.ListBox();
            this.label9 = new System.Windows.Forms.Label();
            this.AddDownButton = new System.Windows.Forms.Button();
            this.txtExtrapolationDist = new System.Windows.Forms.TextBox();
            this.AddUpButton = new System.Windows.Forms.Button();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.MCSAnnecyComputeButton = new System.Windows.Forms.Button();
            this.AlgoCombo = new System.Windows.Forms.ComboBox();
            this.MomentumFitterList = new System.Windows.Forms.ListBox();
            this.textIgnoreDeltaSlope = new System.Windows.Forms.TextBox();
            this.textMomentumResult = new System.Windows.Forms.TextBox();
            this.ExportButton = new System.Windows.Forms.Button();
            this.buttonIgnoreDeltaSlope = new System.Windows.Forms.Button();
            this.label13 = new System.Windows.Forms.Label();
            this.textMeasIgnoreGrains = new System.Windows.Forms.TextBox();
            this.SlopeList = new System.Windows.Forms.ListView();
            this.columnHeader30 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader26 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader27 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader28 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader29 = new System.Windows.Forms.ColumnHeader();
            this.buttonMeasIgnoreGrains = new System.Windows.Forms.Button();
            this.buttonMeasSelAll = new System.Windows.Forms.Button();
            this.tabPage7 = new System.Windows.Forms.TabPage();
            this.cmdSpecialAttributesReference = new System.Windows.Forms.Button();
            this.txtAttribImportValue = new System.Windows.Forms.TextBox();
            this.txtAttribImportTk = new System.Windows.Forms.TextBox();
            this.cmbAttribImport = new System.Windows.Forms.ComboBox();
            this.cmdImportAttribute = new System.Windows.Forms.Button();
            this.txtAttrValue = new System.Windows.Forms.TextBox();
            this.txtAttrName = new System.Windows.Forms.TextBox();
            this.cmdAddSetAttribute = new System.Windows.Forms.Button();
            this.cmdRemoveAttributes = new System.Windows.Forms.Button();
            this.AttributeList = new System.Windows.Forms.ListView();
            this.columnHeader25 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader34 = new System.Windows.Forms.ColumnHeader();
            this.tabPage8 = new System.Windows.Forms.TabPage();
            this.label23 = new System.Windows.Forms.Label();
            this.cmbFBDecaySearch = new System.Windows.Forms.ComboBox();
            this.btnFBSingleProngVertex = new System.Windows.Forms.Button();
            this.txtFBSingleProngVertexZ = new System.Windows.Forms.TextBox();
            this.chkFBManual = new System.Windows.Forms.CheckBox();
            this.btnFBImportAttributes = new System.Windows.Forms.Button();
            this.cmbFBTkImportList = new System.Windows.Forms.ComboBox();
            this.txtFBSlopeTol = new System.Windows.Forms.TextBox();
            this.label18 = new System.Windows.Forms.Label();
            this.btnFBFindSBSFTrack = new System.Windows.Forms.Button();
            this.cmbFBLastPlate = new System.Windows.Forms.ComboBox();
            this.label17 = new System.Windows.Forms.Label();
            this.label16 = new System.Windows.Forms.Label();
            this.cmbFBOutOfBrick = new System.Windows.Forms.ComboBox();
            this.label15 = new System.Windows.Forms.Label();
            this.cmbFBDarkness = new System.Windows.Forms.ComboBox();
            this.label14 = new System.Windows.Forms.Label();
            this.cmbFBParticle = new System.Windows.Forms.ComboBox();
            this.chkFBScanback = new System.Windows.Forms.CheckBox();
            this.chkFBEvent = new System.Windows.Forms.CheckBox();
            this.columnHeader20 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader21 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader22 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader23 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader24 = new System.Windows.Forms.ColumnHeader();
            this.groupBox4.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.tabPage5.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.tabPage7.SuspendLayout();
            this.tabPage8.SuspendLayout();
            this.SuspendLayout();
            // 
            // GeneralList
            // 
            this.GeneralList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.GeneralList.FullRowSelect = true;
            this.GeneralList.GridLines = true;
            this.GeneralList.Location = new System.Drawing.Point(5, 36);
            this.GeneralList.Name = "GeneralList";
            this.GeneralList.Size = new System.Drawing.Size(477, 104);
            this.GeneralList.TabIndex = 0;
            this.GeneralList.UseCompatibleStateImageBehavior = false;
            this.GeneralList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Parameter";
            this.columnHeader1.Width = 200;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Value";
            this.columnHeader2.Width = 100;
            // 
            // SegmentList
            // 
            this.SegmentList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader5,
            this.columnHeader3,
            this.columnHeader12,
            this.columnHeader10,
            this.columnHeader11,
            this.columnHeader8,
            this.columnHeader9,
            this.columnHeader4,
            this.columnHeader14,
            this.columnHeader13,
            this.columnHeader6,
            this.columnHeader7,
            this.columnHeader15,
            this.columnHeader16,
            this.columnHeader17,
            this.columnHeader18,
            this.columnHeader19,
            this.columnHeader31,
            this.columnHeader32,
            this.columnHeader33});
            this.SegmentList.FullRowSelect = true;
            this.SegmentList.GridLines = true;
            this.SegmentList.Location = new System.Drawing.Point(6, 6);
            this.SegmentList.Name = "SegmentList";
            this.SegmentList.Size = new System.Drawing.Size(476, 168);
            this.SegmentList.TabIndex = 1;
            this.SegmentList.UseCompatibleStateImageBehavior = false;
            this.SegmentList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "TrackID";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "LayerID";
            // 
            // columnHeader12
            // 
            this.columnHeader12.Text = "Grains";
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "SX";
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "SY";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "IX";
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "IY";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "IZ";
            // 
            // columnHeader14
            // 
            this.columnHeader14.Text = "UpZ";
            // 
            // columnHeader13
            // 
            this.columnHeader13.Text = "DownZ";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "LayerID";
            // 
            // columnHeader7
            // 
            this.columnHeader7.DisplayIndex = 12;
            this.columnHeader7.Text = "SheetID";
            // 
            // columnHeader15
            // 
            this.columnHeader15.DisplayIndex = 15;
            this.columnHeader15.Text = "Index";
            // 
            // columnHeader16
            // 
            this.columnHeader16.DisplayIndex = 16;
            this.columnHeader16.Text = "OIX";
            // 
            // columnHeader17
            // 
            this.columnHeader17.DisplayIndex = 17;
            this.columnHeader17.Text = "OIY";
            // 
            // columnHeader18
            // 
            this.columnHeader18.DisplayIndex = 18;
            this.columnHeader18.Text = "OSX";
            // 
            // columnHeader19
            // 
            this.columnHeader19.DisplayIndex = 19;
            this.columnHeader19.Text = "OSY";
            // 
            // columnHeader31
            // 
            this.columnHeader31.DisplayIndex = 11;
            this.columnHeader31.Text = "Brick";
            // 
            // columnHeader32
            // 
            this.columnHeader32.DisplayIndex = 14;
            this.columnHeader32.Text = "Side";
            this.columnHeader32.Width = 20;
            // 
            // columnHeader33
            // 
            this.columnHeader33.DisplayIndex = 13;
            this.columnHeader33.Text = "Sigma";
            // 
            // GeneralDumpFileButton
            // 
            this.GeneralDumpFileButton.Location = new System.Drawing.Point(434, 146);
            this.GeneralDumpFileButton.Name = "GeneralDumpFileButton";
            this.GeneralDumpFileButton.Size = new System.Drawing.Size(48, 24);
            this.GeneralDumpFileButton.TabIndex = 2;
            this.GeneralDumpFileButton.Text = "Dump";
            this.GeneralDumpFileButton.Click += new System.EventHandler(this.GeneralDumpFileButton_Click);
            // 
            // GeneralDumpFileText
            // 
            this.GeneralDumpFileText.Location = new System.Drawing.Point(49, 146);
            this.GeneralDumpFileText.Name = "GeneralDumpFileText";
            this.GeneralDumpFileText.Size = new System.Drawing.Size(379, 20);
            this.GeneralDumpFileText.TabIndex = 1;
            // 
            // GeneralSelButton
            // 
            this.GeneralSelButton.Location = new System.Drawing.Point(6, 146);
            this.GeneralSelButton.Name = "GeneralSelButton";
            this.GeneralSelButton.Size = new System.Drawing.Size(32, 24);
            this.GeneralSelButton.TabIndex = 0;
            this.GeneralSelButton.Text = "...";
            this.GeneralSelButton.Click += new System.EventHandler(this.GeneralSelButton_Click);
            // 
            // SegDumpButton
            // 
            this.SegDumpButton.Location = new System.Drawing.Point(434, 180);
            this.SegDumpButton.Name = "SegDumpButton";
            this.SegDumpButton.Size = new System.Drawing.Size(48, 24);
            this.SegDumpButton.TabIndex = 10;
            this.SegDumpButton.Text = "Dump";
            this.SegDumpButton.Click += new System.EventHandler(this.SegDumpButton_Click);
            // 
            // SetCommentButton
            // 
            this.SetCommentButton.Location = new System.Drawing.Point(9, 172);
            this.SetCommentButton.Name = "SetCommentButton";
            this.SetCommentButton.Size = new System.Drawing.Size(90, 24);
            this.SetCommentButton.TabIndex = 4;
            this.SetCommentButton.Text = "Set Comment";
            this.SetCommentButton.Click += new System.EventHandler(this.SetCommentButton_Click);
            // 
            // CommentText
            // 
            this.CommentText.Location = new System.Drawing.Point(145, 172);
            this.CommentText.Name = "CommentText";
            this.CommentText.Size = new System.Drawing.Size(337, 20);
            this.CommentText.TabIndex = 5;
            // 
            // GoUpVtxButton
            // 
            this.GoUpVtxButton.Location = new System.Drawing.Point(9, 204);
            this.GoUpVtxButton.Name = "GoUpVtxButton";
            this.GoUpVtxButton.Size = new System.Drawing.Size(146, 24);
            this.GoUpVtxButton.TabIndex = 6;
            this.GoUpVtxButton.Text = "Go to Upstream Vertex";
            this.GoUpVtxButton.Click += new System.EventHandler(this.GoUpVtxButton_Click);
            // 
            // GoDownVtxButton
            // 
            this.GoDownVtxButton.Location = new System.Drawing.Point(336, 204);
            this.GoDownVtxButton.Name = "GoDownVtxButton";
            this.GoDownVtxButton.Size = new System.Drawing.Size(146, 24);
            this.GoDownVtxButton.TabIndex = 7;
            this.GoDownVtxButton.Text = "Go to Downstream Vertex";
            this.GoDownVtxButton.Click += new System.EventHandler(this.GoDownVtxButton_Click);
            // 
            // SegDumpFileText
            // 
            this.SegDumpFileText.Location = new System.Drawing.Point(48, 180);
            this.SegDumpFileText.Name = "SegDumpFileText";
            this.SegDumpFileText.Size = new System.Drawing.Size(380, 20);
            this.SegDumpFileText.TabIndex = 9;
            // 
            // SegDumpSelButton
            // 
            this.SegDumpSelButton.Location = new System.Drawing.Point(8, 180);
            this.SegDumpSelButton.Name = "SegDumpSelButton";
            this.SegDumpSelButton.Size = new System.Drawing.Size(32, 24);
            this.SegDumpSelButton.TabIndex = 8;
            this.SegDumpSelButton.Text = "...";
            this.SegDumpSelButton.Click += new System.EventHandler(this.SegDumpSelButton_Click);
            // 
            // OrigInfoText
            // 
            this.OrigInfoText.Location = new System.Drawing.Point(117, 134);
            this.OrigInfoText.Name = "OrigInfoText";
            this.OrigInfoText.ReadOnly = true;
            this.OrigInfoText.Size = new System.Drawing.Size(366, 20);
            this.OrigInfoText.TabIndex = 5;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(6, 136);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(102, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "OIX/OIY/OSX/OSY";
            this.label3.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // InfoText
            // 
            this.InfoText.Location = new System.Drawing.Point(117, 108);
            this.InfoText.Name = "InfoText";
            this.InfoText.ReadOnly = true;
            this.InfoText.Size = new System.Drawing.Size(366, 20);
            this.InfoText.TabIndex = 3;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(38, 111);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(70, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "IX/IY/SX/SY";
            // 
            // LayerList
            // 
            this.LayerList.FormattingEnabled = true;
            this.LayerList.Location = new System.Drawing.Point(45, 12);
            this.LayerList.Name = "LayerList";
            this.LayerList.Size = new System.Drawing.Size(438, 82);
            this.LayerList.TabIndex = 1;
            this.LayerList.SelectedIndexChanged += new System.EventHandler(this.OnProjectLayerChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(4, 19);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(33, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Layer";
            // 
            // SelectIPFirstButton
            // 
            this.SelectIPFirstButton.Location = new System.Drawing.Point(3, 13);
            this.SelectIPFirstButton.Name = "SelectIPFirstButton";
            this.SelectIPFirstButton.Size = new System.Drawing.Size(168, 24);
            this.SelectIPFirstButton.TabIndex = 12;
            this.SelectIPFirstButton.Text = "Select for IP (1st track)";
            this.SelectIPFirstButton.Click += new System.EventHandler(this.SelectIPFirstButton_Click);
            // 
            // SelectIPSecondButton
            // 
            this.SelectIPSecondButton.Location = new System.Drawing.Point(315, 13);
            this.SelectIPSecondButton.Name = "SelectIPSecondButton";
            this.SelectIPSecondButton.Size = new System.Drawing.Size(168, 24);
            this.SelectIPSecondButton.TabIndex = 13;
            this.SelectIPSecondButton.Text = "Select for IP (2nd track)";
            this.SelectIPSecondButton.Click += new System.EventHandler(this.SelectIPSecondButton_Click);
            // 
            // btnShowRelatedTracks
            // 
            this.btnShowRelatedTracks.Location = new System.Drawing.Point(3, 43);
            this.btnShowRelatedTracks.Name = "btnShowRelatedTracks";
            this.btnShowRelatedTracks.Size = new System.Drawing.Size(144, 24);
            this.btnShowRelatedTracks.TabIndex = 14;
            this.btnShowRelatedTracks.Text = "Related tracks";
            this.btnShowRelatedTracks.Click += new System.EventHandler(this.btnShowRelatedTracks_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(370, 48);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(47, 13);
            this.label4.TabIndex = 19;
            this.label4.Text = "Opening";
            // 
            // txtOpening
            // 
            this.txtOpening.Location = new System.Drawing.Point(438, 44);
            this.txtOpening.Name = "txtOpening";
            this.txtOpening.Size = new System.Drawing.Size(45, 20);
            this.txtOpening.TabIndex = 20;
            this.txtOpening.Leave += new System.EventHandler(this.OnOpeningLeave);
            // 
            // txtRadius
            // 
            this.txtRadius.Location = new System.Drawing.Point(438, 67);
            this.txtRadius.Name = "txtRadius";
            this.txtRadius.Size = new System.Drawing.Size(45, 20);
            this.txtRadius.TabIndex = 22;
            this.txtRadius.Leave += new System.EventHandler(this.OnRadiusLeave);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(370, 71);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(40, 13);
            this.label5.TabIndex = 21;
            this.label5.Text = "Radius";
            // 
            // txtDeltaSlope
            // 
            this.txtDeltaSlope.Location = new System.Drawing.Point(438, 90);
            this.txtDeltaSlope.Name = "txtDeltaSlope";
            this.txtDeltaSlope.Size = new System.Drawing.Size(45, 20);
            this.txtDeltaSlope.TabIndex = 24;
            this.txtDeltaSlope.Leave += new System.EventHandler(this.OnDeltaSlopeLeave);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(370, 94);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(59, 13);
            this.label6.TabIndex = 23;
            this.label6.Text = "DeltaSlope";
            // 
            // txtDeltaZ
            // 
            this.txtDeltaZ.Location = new System.Drawing.Point(438, 114);
            this.txtDeltaZ.Name = "txtDeltaZ";
            this.txtDeltaZ.Size = new System.Drawing.Size(45, 20);
            this.txtDeltaZ.TabIndex = 26;
            this.txtDeltaZ.Leave += new System.EventHandler(this.OnDeltaZLeave);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(370, 118);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(39, 13);
            this.label7.TabIndex = 25;
            this.label7.Text = "DeltaZ";
            // 
            // radioUpstream
            // 
            this.radioUpstream.AutoSize = true;
            this.radioUpstream.Location = new System.Drawing.Point(6, 19);
            this.radioUpstream.Name = "radioUpstream";
            this.radioUpstream.Size = new System.Drawing.Size(70, 17);
            this.radioUpstream.TabIndex = 27;
            this.radioUpstream.Text = "Upstream";
            this.radioUpstream.UseVisualStyleBackColor = true;
            // 
            // radioDownstream
            // 
            this.radioDownstream.AutoSize = true;
            this.radioDownstream.Checked = true;
            this.radioDownstream.Location = new System.Drawing.Point(6, 41);
            this.radioDownstream.Name = "radioDownstream";
            this.radioDownstream.Size = new System.Drawing.Size(84, 17);
            this.radioDownstream.TabIndex = 28;
            this.radioDownstream.TabStop = true;
            this.radioDownstream.Text = "Downstream";
            this.radioDownstream.UseVisualStyleBackColor = true;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.radioUpstream);
            this.groupBox4.Controls.Add(this.radioDownstream);
            this.groupBox4.Location = new System.Drawing.Point(156, 43);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(96, 69);
            this.groupBox4.TabIndex = 29;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "End";
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.radioUpstreamDir);
            this.groupBox5.Controls.Add(this.radioDownstreamDir);
            this.groupBox5.Location = new System.Drawing.Point(260, 43);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(104, 69);
            this.groupBox5.TabIndex = 30;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Direction";
            // 
            // radioUpstreamDir
            // 
            this.radioUpstreamDir.AutoSize = true;
            this.radioUpstreamDir.Location = new System.Drawing.Point(6, 19);
            this.radioUpstreamDir.Name = "radioUpstreamDir";
            this.radioUpstreamDir.Size = new System.Drawing.Size(70, 17);
            this.radioUpstreamDir.TabIndex = 27;
            this.radioUpstreamDir.Text = "Upstream";
            this.radioUpstreamDir.UseVisualStyleBackColor = true;
            // 
            // radioDownstreamDir
            // 
            this.radioDownstreamDir.AutoSize = true;
            this.radioDownstreamDir.Checked = true;
            this.radioDownstreamDir.Location = new System.Drawing.Point(6, 41);
            this.radioDownstreamDir.Name = "radioDownstreamDir";
            this.radioDownstreamDir.Size = new System.Drawing.Size(84, 17);
            this.radioDownstreamDir.TabIndex = 28;
            this.radioDownstreamDir.TabStop = true;
            this.radioDownstreamDir.Text = "Downstream";
            this.radioDownstreamDir.UseVisualStyleBackColor = true;
            // 
            // btnShowRelatedSegments
            // 
            this.btnShowRelatedSegments.Location = new System.Drawing.Point(2, 73);
            this.btnShowRelatedSegments.Name = "btnShowRelatedSegments";
            this.btnShowRelatedSegments.Size = new System.Drawing.Size(145, 24);
            this.btnShowRelatedSegments.TabIndex = 31;
            this.btnShowRelatedSegments.Text = "Related segments";
            this.btnShowRelatedSegments.Click += new System.EventHandler(this.btnShowRelatedSegments_Click);
            // 
            // btnScanbackScanforth
            // 
            this.btnScanbackScanforth.Location = new System.Drawing.Point(2, 146);
            this.btnScanbackScanforth.Name = "btnScanbackScanforth";
            this.btnScanbackScanforth.Size = new System.Drawing.Size(145, 24);
            this.btnScanbackScanforth.TabIndex = 32;
            this.btnScanbackScanforth.Text = "Scanback/Scanforth";
            this.btnScanbackScanforth.Click += new System.EventHandler(this.btnScanbackScanforth_Click);
            // 
            // txtMaxMisses
            // 
            this.txtMaxMisses.Location = new System.Drawing.Point(236, 150);
            this.txtMaxMisses.Name = "txtMaxMisses";
            this.txtMaxMisses.Size = new System.Drawing.Size(45, 20);
            this.txtMaxMisses.TabIndex = 34;
            this.txtMaxMisses.Leave += new System.EventHandler(this.OnMaxMissesLeave);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(157, 153);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(62, 13);
            this.label8.TabIndex = 33;
            this.label8.Text = "Max Misses";
            // 
            // LabelText
            // 
            this.LabelText.Location = new System.Drawing.Point(263, 337);
            this.LabelText.Name = "LabelText";
            this.LabelText.Size = new System.Drawing.Size(237, 20);
            this.LabelText.TabIndex = 37;
            // 
            // SetLabelButton
            // 
            this.SetLabelButton.Location = new System.Drawing.Point(168, 334);
            this.SetLabelButton.Name = "SetLabelButton";
            this.SetLabelButton.Size = new System.Drawing.Size(89, 24);
            this.SetLabelButton.TabIndex = 38;
            this.SetLabelButton.Text = "Set Label";
            this.SetLabelButton.Click += new System.EventHandler(this.SetLabelButton_Click);
            // 
            // HighlightCheck
            // 
            this.HighlightCheck.AutoSize = true;
            this.HighlightCheck.Location = new System.Drawing.Point(11, 339);
            this.HighlightCheck.Name = "HighlightCheck";
            this.HighlightCheck.Size = new System.Drawing.Size(61, 17);
            this.HighlightCheck.TabIndex = 39;
            this.HighlightCheck.Text = "Higlight";
            this.HighlightCheck.UseVisualStyleBackColor = true;
            this.HighlightCheck.CheckedChanged += new System.EventHandler(this.HighlightCheck_CheckedChanged);
            // 
            // EnableLabelCheck
            // 
            this.EnableLabelCheck.AutoSize = true;
            this.EnableLabelCheck.Location = new System.Drawing.Point(78, 339);
            this.EnableLabelCheck.Name = "EnableLabelCheck";
            this.EnableLabelCheck.Size = new System.Drawing.Size(82, 17);
            this.EnableLabelCheck.TabIndex = 40;
            this.EnableLabelCheck.Text = "Show Label";
            this.EnableLabelCheck.UseVisualStyleBackColor = true;
            this.EnableLabelCheck.CheckedChanged += new System.EventHandler(this.EnableLabelCheck_CheckedChanged);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Controls.Add(this.tabPage6);
            this.tabControl1.Controls.Add(this.tabPage7);
            this.tabControl1.Controls.Add(this.tabPage8);
            this.tabControl1.Location = new System.Drawing.Point(8, 6);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(496, 322);
            this.tabControl1.TabIndex = 41;
            // 
            // tabPage1
            // 
            this.tabPage1.BackColor = System.Drawing.Color.Transparent;
            this.tabPage1.Controls.Add(this.KinkSearchXText);
            this.tabPage1.Controls.Add(this.label28);
            this.tabPage1.Controls.Add(this.PlotButton);
            this.tabPage1.Controls.Add(this.KinkPlateText);
            this.tabPage1.Controls.Add(this.label21);
            this.tabPage1.Controls.Add(this.DSlopeRText);
            this.tabPage1.Controls.Add(this.label20);
            this.tabPage1.Controls.Add(this.DSlopeRMSText);
            this.tabPage1.Controls.Add(this.label19);
            this.tabPage1.Controls.Add(this.txtDataSet);
            this.tabPage1.Controls.Add(this.label12);
            this.tabPage1.Controls.Add(this.GeneralDumpFileButton);
            this.tabPage1.Controls.Add(this.GeneralList);
            this.tabPage1.Controls.Add(this.GeneralDumpFileText);
            this.tabPage1.Controls.Add(this.GeneralSelButton);
            this.tabPage1.Controls.Add(this.CommentText);
            this.tabPage1.Controls.Add(this.SetCommentButton);
            this.tabPage1.Controls.Add(this.GoUpVtxButton);
            this.tabPage1.Controls.Add(this.GoDownVtxButton);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(488, 296);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "General";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // KinkSearchXText
            // 
            this.KinkSearchXText.Location = new System.Drawing.Point(281, 265);
            this.KinkSearchXText.Name = "KinkSearchXText";
            this.KinkSearchXText.ReadOnly = true;
            this.KinkSearchXText.Size = new System.Drawing.Size(201, 20);
            this.KinkSearchXText.TabIndex = 18;
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.Location = new System.Drawing.Point(163, 268);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(112, 13);
            this.label28.TabIndex = 17;
            this.label28.Text = "Kink search exception";
            // 
            // PlotButton
            // 
            this.PlotButton.Location = new System.Drawing.Point(434, 7);
            this.PlotButton.Name = "PlotButton";
            this.PlotButton.Size = new System.Drawing.Size(48, 24);
            this.PlotButton.TabIndex = 16;
            this.PlotButton.Text = "Plot";
            this.PlotButton.Click += new System.EventHandler(this.PlotButton_Click);
            // 
            // KinkPlateText
            // 
            this.KinkPlateText.Location = new System.Drawing.Point(84, 265);
            this.KinkPlateText.Name = "KinkPlateText";
            this.KinkPlateText.ReadOnly = true;
            this.KinkPlateText.Size = new System.Drawing.Size(64, 20);
            this.KinkPlateText.TabIndex = 15;
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(13, 268);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(62, 13);
            this.label21.TabIndex = 14;
            this.label21.Text = "Kink/ Layer";
            // 
            // DSlopeRText
            // 
            this.DSlopeRText.Location = new System.Drawing.Point(376, 239);
            this.DSlopeRText.Name = "DSlopeRText";
            this.DSlopeRText.ReadOnly = true;
            this.DSlopeRText.Size = new System.Drawing.Size(106, 20);
            this.DSlopeRText.TabIndex = 13;
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(296, 242);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(69, 13);
            this.label20.TabIndex = 12;
            this.label20.Text = "Rtransv/long";
            // 
            // DSlopeRMSText
            // 
            this.DSlopeRMSText.Location = new System.Drawing.Point(146, 239);
            this.DSlopeRMSText.Name = "DSlopeRMSText";
            this.DSlopeRMSText.ReadOnly = true;
            this.DSlopeRMSText.Size = new System.Drawing.Size(119, 20);
            this.DSlopeRMSText.TabIndex = 11;
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(11, 242);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(126, 13);
            this.label19.TabIndex = 10;
            this.label19.Text = "DSlope RMS transv/long";
            // 
            // txtDataSet
            // 
            this.txtDataSet.Location = new System.Drawing.Point(66, 10);
            this.txtDataSet.Name = "txtDataSet";
            this.txtDataSet.ReadOnly = true;
            this.txtDataSet.Size = new System.Drawing.Size(365, 20);
            this.txtDataSet.TabIndex = 9;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(6, 13);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(46, 13);
            this.label12.TabIndex = 8;
            this.label12.Text = "DataSet";
            // 
            // tabPage2
            // 
            this.tabPage2.BackColor = System.Drawing.Color.Transparent;
            this.tabPage2.Controls.Add(this.chkSplitWithVertex);
            this.tabPage2.Controls.Add(this.SplitTrackButton);
            this.tabPage2.Controls.Add(this.SegAddReplaceButton);
            this.tabPage2.Controls.Add(this.SegRemoveButton);
            this.tabPage2.Controls.Add(this.SegDumpButton);
            this.tabPage2.Controls.Add(this.SegmentList);
            this.tabPage2.Controls.Add(this.SegDumpFileText);
            this.tabPage2.Controls.Add(this.SegDumpSelButton);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(488, 296);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Segments";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // chkSplitWithVertex
            // 
            this.chkSplitWithVertex.AutoSize = true;
            this.chkSplitWithVertex.Location = new System.Drawing.Point(208, 266);
            this.chkSplitWithVertex.Name = "chkSplitWithVertex";
            this.chkSplitWithVertex.Size = new System.Drawing.Size(90, 17);
            this.chkSplitWithVertex.TabIndex = 42;
            this.chkSplitWithVertex.Text = "Create Vertex";
            this.chkSplitWithVertex.UseVisualStyleBackColor = true;
            // 
            // SplitTrackButton
            // 
            this.SplitTrackButton.Location = new System.Drawing.Point(8, 261);
            this.SplitTrackButton.Name = "SplitTrackButton";
            this.SplitTrackButton.Size = new System.Drawing.Size(186, 24);
            this.SplitTrackButton.TabIndex = 41;
            this.SplitTrackButton.Text = "Split track at selected segment";
            this.SplitTrackButton.Click += new System.EventHandler(this.SplitTrackButton_Click);
            // 
            // SegAddReplaceButton
            // 
            this.SegAddReplaceButton.Location = new System.Drawing.Point(8, 222);
            this.SegAddReplaceButton.Name = "SegAddReplaceButton";
            this.SegAddReplaceButton.Size = new System.Drawing.Size(186, 24);
            this.SegAddReplaceButton.TabIndex = 40;
            this.SegAddReplaceButton.Text = "Start adding/replacing segments";
            this.SegAddReplaceButton.Click += new System.EventHandler(this.SegAddReplaceButton_Click);
            // 
            // SegRemoveButton
            // 
            this.SegRemoveButton.Location = new System.Drawing.Point(320, 222);
            this.SegRemoveButton.Name = "SegRemoveButton";
            this.SegRemoveButton.Size = new System.Drawing.Size(162, 24);
            this.SegRemoveButton.TabIndex = 39;
            this.SegRemoveButton.Text = "Remove selected segments";
            this.SegRemoveButton.Click += new System.EventHandler(this.SegRemoveButton_Click);
            // 
            // tabPage3
            // 
            this.tabPage3.BackColor = System.Drawing.Color.Transparent;
            this.tabPage3.Controls.Add(this.LaunchScanButton);
            this.tabPage3.Controls.Add(this.CheckAllLayers);
            this.tabPage3.Controls.Add(this.CheckMissingBasetracksButton);
            this.tabPage3.Controls.Add(this.AppendUpButton);
            this.tabPage3.Controls.Add(this.AppendDownButton);
            this.tabPage3.Controls.Add(this.label24);
            this.tabPage3.Controls.Add(this.UpStopsText);
            this.tabPage3.Controls.Add(this.label25);
            this.tabPage3.Controls.Add(this.DownStopsText);
            this.tabPage3.Controls.Add(this.label26);
            this.tabPage3.Controls.Add(this.UpSegsText);
            this.tabPage3.Controls.Add(this.label27);
            this.tabPage3.Controls.Add(this.DownSegsText);
            this.tabPage3.Controls.Add(this.AppendToSelButton);
            this.tabPage3.Controls.Add(this.AppendToFileText);
            this.tabPage3.Controls.Add(this.AppendToButton);
            this.tabPage3.Controls.Add(this.OrigInfoText);
            this.tabPage3.Controls.Add(this.LayerList);
            this.tabPage3.Controls.Add(this.label3);
            this.tabPage3.Controls.Add(this.label1);
            this.tabPage3.Controls.Add(this.InfoText);
            this.tabPage3.Controls.Add(this.label2);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(488, 296);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Projection";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // LaunchScanButton
            // 
            this.LaunchScanButton.Location = new System.Drawing.Point(394, 258);
            this.LaunchScanButton.Name = "LaunchScanButton";
            this.LaunchScanButton.Size = new System.Drawing.Size(89, 24);
            this.LaunchScanButton.TabIndex = 69;
            this.LaunchScanButton.Text = "Launch Scan";
            this.LaunchScanButton.Click += new System.EventHandler(this.LaunchScanButton_Click);
            // 
            // CheckAllLayers
            // 
            this.CheckAllLayers.Location = new System.Drawing.Point(203, 258);
            this.CheckAllLayers.Name = "CheckAllLayers";
            this.CheckAllLayers.Size = new System.Drawing.Size(123, 24);
            this.CheckAllLayers.TabIndex = 68;
            this.CheckAllLayers.Text = "Check all layers";
            this.CheckAllLayers.Click += new System.EventHandler(this.CheckAllLayers_Click);
            // 
            // CheckMissingBasetracksButton
            // 
            this.CheckMissingBasetracksButton.Location = new System.Drawing.Point(19, 258);
            this.CheckMissingBasetracksButton.Name = "CheckMissingBasetracksButton";
            this.CheckMissingBasetracksButton.Size = new System.Drawing.Size(178, 24);
            this.CheckMissingBasetracksButton.TabIndex = 67;
            this.CheckMissingBasetracksButton.Text = "Check all missing basetracks";
            this.CheckMissingBasetracksButton.Click += new System.EventHandler(this.CheckMissingBasetracksButton_Click);
            // 
            // AppendUpButton
            // 
            this.AppendUpButton.Location = new System.Drawing.Point(19, 223);
            this.AppendUpButton.Name = "AppendUpButton";
            this.AppendUpButton.Size = new System.Drawing.Size(201, 24);
            this.AppendUpButton.TabIndex = 66;
            this.AppendUpButton.Text = "Append Upstream segments";
            this.AppendUpButton.Click += new System.EventHandler(this.AppendUpButton_Click);
            // 
            // AppendDownButton
            // 
            this.AppendDownButton.Location = new System.Drawing.Point(19, 191);
            this.AppendDownButton.Name = "AppendDownButton";
            this.AppendDownButton.Size = new System.Drawing.Size(201, 24);
            this.AppendDownButton.TabIndex = 65;
            this.AppendDownButton.Text = "Append Downstream segments";
            this.AppendDownButton.Click += new System.EventHandler(this.AppendDownButton_Click);
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(422, 228);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(57, 13);
            this.label24.TabIndex = 64;
            this.label24.Text = "stop layers";
            // 
            // UpStopsText
            // 
            this.UpStopsText.Location = new System.Drawing.Point(366, 225);
            this.UpStopsText.Name = "UpStopsText";
            this.UpStopsText.Size = new System.Drawing.Size(50, 20);
            this.UpStopsText.TabIndex = 63;
            this.UpStopsText.Leave += new System.EventHandler(this.OnUpStopsLeave);
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.Location = new System.Drawing.Point(422, 196);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(57, 13);
            this.label25.TabIndex = 62;
            this.label25.Text = "stop layers";
            // 
            // DownStopsText
            // 
            this.DownStopsText.Location = new System.Drawing.Point(366, 193);
            this.DownStopsText.Name = "DownStopsText";
            this.DownStopsText.Size = new System.Drawing.Size(50, 20);
            this.DownStopsText.TabIndex = 61;
            this.DownStopsText.Leave += new System.EventHandler(this.OnDownStopsLeave);
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(282, 228);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(61, 13);
            this.label26.TabIndex = 60;
            this.label26.Text = "track layers";
            // 
            // UpSegsText
            // 
            this.UpSegsText.Location = new System.Drawing.Point(226, 225);
            this.UpSegsText.Name = "UpSegsText";
            this.UpSegsText.Size = new System.Drawing.Size(50, 20);
            this.UpSegsText.TabIndex = 58;
            this.UpSegsText.Leave += new System.EventHandler(this.OnUpSegsLeave);
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.Location = new System.Drawing.Point(282, 196);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(61, 13);
            this.label27.TabIndex = 57;
            this.label27.Text = "track layers";
            // 
            // DownSegsText
            // 
            this.DownSegsText.Location = new System.Drawing.Point(226, 193);
            this.DownSegsText.Name = "DownSegsText";
            this.DownSegsText.Size = new System.Drawing.Size(50, 20);
            this.DownSegsText.TabIndex = 55;
            this.DownSegsText.Leave += new System.EventHandler(this.OnDowsSegsLeave);
            // 
            // AppendToSelButton
            // 
            this.AppendToSelButton.Location = new System.Drawing.Point(450, 162);
            this.AppendToSelButton.Name = "AppendToSelButton";
            this.AppendToSelButton.Size = new System.Drawing.Size(33, 24);
            this.AppendToSelButton.TabIndex = 41;
            this.AppendToSelButton.Text = "...";
            this.AppendToSelButton.Click += new System.EventHandler(this.AppendToSelButton_Click);
            // 
            // AppendToFileText
            // 
            this.AppendToFileText.Location = new System.Drawing.Point(117, 164);
            this.AppendToFileText.Name = "AppendToFileText";
            this.AppendToFileText.Size = new System.Drawing.Size(327, 20);
            this.AppendToFileText.TabIndex = 40;
            // 
            // AppendToButton
            // 
            this.AppendToButton.Location = new System.Drawing.Point(19, 161);
            this.AppendToButton.Name = "AppendToButton";
            this.AppendToButton.Size = new System.Drawing.Size(89, 24);
            this.AppendToButton.TabIndex = 39;
            this.AppendToButton.Text = "Append to";
            this.AppendToButton.Click += new System.EventHandler(this.AppendToButton_Click);
            // 
            // tabPage4
            // 
            this.tabPage4.BackColor = System.Drawing.Color.Transparent;
            this.tabPage4.Controls.Add(this.rdBoth);
            this.tabPage4.Controls.Add(this.rdDataset);
            this.tabPage4.Controls.Add(this.rdShow);
            this.tabPage4.Controls.Add(this.chkBroadcastAction);
            this.tabPage4.Controls.Add(this.btnBrowseTrack);
            this.tabPage4.Controls.Add(this.cmbMatchOtherDS);
            this.tabPage4.Controls.Add(this.btnMatchInOtherDatasets);
            this.tabPage4.Controls.Add(this.label22);
            this.tabPage4.Controls.Add(this.txtIP);
            this.tabPage4.Controls.Add(this.SelectIPFirstButton);
            this.tabPage4.Controls.Add(this.SelectIPSecondButton);
            this.tabPage4.Controls.Add(this.btnShowRelatedTracks);
            this.tabPage4.Controls.Add(this.label4);
            this.tabPage4.Controls.Add(this.txtOpening);
            this.tabPage4.Controls.Add(this.txtMaxMisses);
            this.tabPage4.Controls.Add(this.label5);
            this.tabPage4.Controls.Add(this.label8);
            this.tabPage4.Controls.Add(this.txtRadius);
            this.tabPage4.Controls.Add(this.btnScanbackScanforth);
            this.tabPage4.Controls.Add(this.label6);
            this.tabPage4.Controls.Add(this.btnShowRelatedSegments);
            this.tabPage4.Controls.Add(this.txtDeltaSlope);
            this.tabPage4.Controls.Add(this.groupBox5);
            this.tabPage4.Controls.Add(this.label7);
            this.tabPage4.Controls.Add(this.groupBox4);
            this.tabPage4.Controls.Add(this.txtDeltaZ);
            this.tabPage4.Location = new System.Drawing.Point(4, 22);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(488, 296);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "Neighborhood";
            this.tabPage4.UseVisualStyleBackColor = true;
            // 
            // rdBoth
            // 
            this.rdBoth.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdBoth.AutoSize = true;
            this.rdBoth.Location = new System.Drawing.Point(108, 114);
            this.rdBoth.Name = "rdBoth";
            this.rdBoth.Size = new System.Drawing.Size(39, 23);
            this.rdBoth.TabIndex = 54;
            this.rdBoth.Text = "Both";
            this.rdBoth.UseVisualStyleBackColor = true;
            // 
            // rdDataset
            // 
            this.rdDataset.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdDataset.AutoSize = true;
            this.rdDataset.Location = new System.Drawing.Point(50, 114);
            this.rdDataset.Name = "rdDataset";
            this.rdDataset.Size = new System.Drawing.Size(54, 23);
            this.rdDataset.TabIndex = 53;
            this.rdDataset.Text = "Dataset";
            this.rdDataset.UseVisualStyleBackColor = true;
            // 
            // rdShow
            // 
            this.rdShow.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdShow.AutoSize = true;
            this.rdShow.Checked = true;
            this.rdShow.Location = new System.Drawing.Point(0, 114);
            this.rdShow.Name = "rdShow";
            this.rdShow.Size = new System.Drawing.Size(44, 23);
            this.rdShow.TabIndex = 52;
            this.rdShow.TabStop = true;
            this.rdShow.Text = "Show";
            this.rdShow.UseVisualStyleBackColor = true;
            // 
            // chkBroadcastAction
            // 
            this.chkBroadcastAction.AutoSize = true;
            this.chkBroadcastAction.Location = new System.Drawing.Point(3, 179);
            this.chkBroadcastAction.Name = "chkBroadcastAction";
            this.chkBroadcastAction.Size = new System.Drawing.Size(203, 17);
            this.chkBroadcastAction.TabIndex = 41;
            this.chkBroadcastAction.Text = "Broadcast action to all open browsers";
            this.chkBroadcastAction.UseVisualStyleBackColor = true;
            // 
            // btnBrowseTrack
            // 
            this.btnBrowseTrack.Location = new System.Drawing.Point(351, 207);
            this.btnBrowseTrack.Name = "btnBrowseTrack";
            this.btnBrowseTrack.Size = new System.Drawing.Size(132, 24);
            this.btnBrowseTrack.TabIndex = 39;
            this.btnBrowseTrack.Text = "Browse";
            this.btnBrowseTrack.Click += new System.EventHandler(this.btnBrowseTrack_Click);
            // 
            // cmbMatchOtherDS
            // 
            this.cmbMatchOtherDS.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbMatchOtherDS.FormattingEnabled = true;
            this.cmbMatchOtherDS.Location = new System.Drawing.Point(141, 209);
            this.cmbMatchOtherDS.Name = "cmbMatchOtherDS";
            this.cmbMatchOtherDS.Size = new System.Drawing.Size(204, 21);
            this.cmbMatchOtherDS.TabIndex = 38;
            // 
            // btnMatchInOtherDatasets
            // 
            this.btnMatchInOtherDatasets.Location = new System.Drawing.Point(3, 207);
            this.btnMatchInOtherDatasets.Name = "btnMatchInOtherDatasets";
            this.btnMatchInOtherDatasets.Size = new System.Drawing.Size(132, 24);
            this.btnMatchInOtherDatasets.TabIndex = 37;
            this.btnMatchInOtherDatasets.Text = "Match in other datasets";
            this.btnMatchInOtherDatasets.Click += new System.EventHandler(this.btnMatchInOtherDatasets_Click);
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(370, 144);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(17, 13);
            this.label22.TabIndex = 35;
            this.label22.Text = "IP";
            // 
            // txtIP
            // 
            this.txtIP.Location = new System.Drawing.Point(438, 140);
            this.txtIP.Name = "txtIP";
            this.txtIP.Size = new System.Drawing.Size(45, 20);
            this.txtIP.TabIndex = 36;
            this.txtIP.Leave += new System.EventHandler(this.OnIPLeave);
            // 
            // tabPage5
            // 
            this.tabPage5.BackColor = System.Drawing.Color.Transparent;
            this.tabPage5.Controls.Add(this.MomentumFromAttrButton);
            this.tabPage5.Controls.Add(this.label11);
            this.tabPage5.Controls.Add(this.txtMomentum);
            this.tabPage5.Controls.Add(this.WeightFromQualityButton);
            this.tabPage5.Controls.Add(this.label10);
            this.tabPage5.Controls.Add(this.txtWeight);
            this.tabPage5.Controls.Add(this.radioUseLastSeg);
            this.tabPage5.Controls.Add(this.radioUseFit);
            this.tabPage5.Controls.Add(this.ListFits);
            this.tabPage5.Controls.Add(this.label9);
            this.tabPage5.Controls.Add(this.AddDownButton);
            this.tabPage5.Controls.Add(this.txtExtrapolationDist);
            this.tabPage5.Controls.Add(this.AddUpButton);
            this.tabPage5.Location = new System.Drawing.Point(4, 22);
            this.tabPage5.Name = "tabPage5";
            this.tabPage5.Size = new System.Drawing.Size(488, 296);
            this.tabPage5.TabIndex = 4;
            this.tabPage5.Text = "Vertex fit";
            this.tabPage5.UseVisualStyleBackColor = true;
            // 
            // MomentumFromAttrButton
            // 
            this.MomentumFromAttrButton.Location = new System.Drawing.Point(129, 173);
            this.MomentumFromAttrButton.Name = "MomentumFromAttrButton";
            this.MomentumFromAttrButton.Size = new System.Drawing.Size(85, 24);
            this.MomentumFromAttrButton.TabIndex = 15;
            this.MomentumFromAttrButton.Text = "From attribute";
            this.MomentumFromAttrButton.Click += new System.EventHandler(this.MomentumFromAttribute_Click);
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(12, 179);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(55, 13);
            this.label11.TabIndex = 14;
            this.label11.Text = "P (GeV/c)";
            // 
            // txtMomentum
            // 
            this.txtMomentum.Location = new System.Drawing.Point(73, 176);
            this.txtMomentum.Name = "txtMomentum";
            this.txtMomentum.Size = new System.Drawing.Size(50, 20);
            this.txtMomentum.TabIndex = 13;
            this.txtMomentum.Leave += new System.EventHandler(this.OnMomentumLeave);
            // 
            // WeightFromQualityButton
            // 
            this.WeightFromQualityButton.Location = new System.Drawing.Point(129, 144);
            this.WeightFromQualityButton.Name = "WeightFromQualityButton";
            this.WeightFromQualityButton.Size = new System.Drawing.Size(85, 24);
            this.WeightFromQualityButton.TabIndex = 12;
            this.WeightFromQualityButton.Text = "From quality";
            this.WeightFromQualityButton.Click += new System.EventHandler(this.WeightFromQualityButton_Click);
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(12, 150);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(41, 13);
            this.label10.TabIndex = 11;
            this.label10.Text = "Weight";
            // 
            // txtWeight
            // 
            this.txtWeight.Location = new System.Drawing.Point(73, 147);
            this.txtWeight.Name = "txtWeight";
            this.txtWeight.Size = new System.Drawing.Size(50, 20);
            this.txtWeight.TabIndex = 10;
            this.txtWeight.Leave += new System.EventHandler(this.OnWeightLeave);
            // 
            // radioUseLastSeg
            // 
            this.radioUseLastSeg.AutoSize = true;
            this.radioUseLastSeg.Location = new System.Drawing.Point(15, 116);
            this.radioUseLastSeg.Name = "radioUseLastSeg";
            this.radioUseLastSeg.Size = new System.Drawing.Size(106, 17);
            this.radioUseLastSeg.TabIndex = 9;
            this.radioUseLastSeg.Text = "Use last segment";
            this.radioUseLastSeg.UseVisualStyleBackColor = true;
            // 
            // radioUseFit
            // 
            this.radioUseFit.AutoSize = true;
            this.radioUseFit.Checked = true;
            this.radioUseFit.Location = new System.Drawing.Point(15, 93);
            this.radioUseFit.Name = "radioUseFit";
            this.radioUseFit.Size = new System.Drawing.Size(55, 17);
            this.radioUseFit.TabIndex = 8;
            this.radioUseFit.TabStop = true;
            this.radioUseFit.Text = "Use fit";
            this.radioUseFit.UseVisualStyleBackColor = true;
            // 
            // ListFits
            // 
            this.ListFits.FormattingEnabled = true;
            this.ListFits.Location = new System.Drawing.Point(228, 24);
            this.ListFits.Name = "ListFits";
            this.ListFits.Size = new System.Drawing.Size(248, 173);
            this.ListFits.Sorted = true;
            this.ListFits.TabIndex = 6;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(12, 25);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(111, 13);
            this.label9.TabIndex = 5;
            this.label9.Text = "Extrapolation distance";
            // 
            // AddDownButton
            // 
            this.AddDownButton.Location = new System.Drawing.Point(129, 54);
            this.AddDownButton.Name = "AddDownButton";
            this.AddDownButton.Size = new System.Drawing.Size(85, 24);
            this.AddDownButton.TabIndex = 4;
            this.AddDownButton.Text = "Downstream";
            this.AddDownButton.Click += new System.EventHandler(this.AddDownButton_Click);
            // 
            // txtExtrapolationDist
            // 
            this.txtExtrapolationDist.Location = new System.Drawing.Point(129, 25);
            this.txtExtrapolationDist.Name = "txtExtrapolationDist";
            this.txtExtrapolationDist.Size = new System.Drawing.Size(85, 20);
            this.txtExtrapolationDist.TabIndex = 3;
            this.txtExtrapolationDist.Leave += new System.EventHandler(this.OnExtrapDistLeave);
            // 
            // AddUpButton
            // 
            this.AddUpButton.Location = new System.Drawing.Point(15, 54);
            this.AddUpButton.Name = "AddUpButton";
            this.AddUpButton.Size = new System.Drawing.Size(85, 24);
            this.AddUpButton.TabIndex = 2;
            this.AddUpButton.Text = "Upstream";
            this.AddUpButton.Click += new System.EventHandler(this.AddUpButton_Click);
            // 
            // tabPage6
            // 
            this.tabPage6.BackColor = System.Drawing.Color.Transparent;
            this.tabPage6.Controls.Add(this.MCSAnnecyComputeButton);
            this.tabPage6.Controls.Add(this.AlgoCombo);
            this.tabPage6.Controls.Add(this.MomentumFitterList);
            this.tabPage6.Controls.Add(this.textIgnoreDeltaSlope);
            this.tabPage6.Controls.Add(this.textMomentumResult);
            this.tabPage6.Controls.Add(this.ExportButton);
            this.tabPage6.Controls.Add(this.buttonIgnoreDeltaSlope);
            this.tabPage6.Controls.Add(this.label13);
            this.tabPage6.Controls.Add(this.textMeasIgnoreGrains);
            this.tabPage6.Controls.Add(this.SlopeList);
            this.tabPage6.Controls.Add(this.buttonMeasIgnoreGrains);
            this.tabPage6.Controls.Add(this.buttonMeasSelAll);
            this.tabPage6.Location = new System.Drawing.Point(4, 22);
            this.tabPage6.Name = "tabPage6";
            this.tabPage6.Size = new System.Drawing.Size(488, 296);
            this.tabPage6.TabIndex = 5;
            this.tabPage6.Text = "Momentum";
            this.tabPage6.UseVisualStyleBackColor = true;
            // 
            // MCSAnnecyComputeButton
            // 
            this.MCSAnnecyComputeButton.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.MCSAnnecyComputeButton.Location = new System.Drawing.Point(3, 166);
            this.MCSAnnecyComputeButton.Name = "MCSAnnecyComputeButton";
            this.MCSAnnecyComputeButton.Size = new System.Drawing.Size(157, 27);
            this.MCSAnnecyComputeButton.TabIndex = 42;
            this.MCSAnnecyComputeButton.Text = "Compute momentum";
            this.MCSAnnecyComputeButton.UseVisualStyleBackColor = true;
            this.MCSAnnecyComputeButton.Visible = false;
            this.MCSAnnecyComputeButton.Click += new System.EventHandler(this.MCSAnnecyComputeButton_Click);
            // 
            // AlgoCombo
            // 
            this.AlgoCombo.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.AlgoCombo.FormattingEnabled = true;
            this.AlgoCombo.Location = new System.Drawing.Point(339, 138);
            this.AlgoCombo.Name = "AlgoCombo";
            this.AlgoCombo.Size = new System.Drawing.Size(144, 21);
            this.AlgoCombo.TabIndex = 18;
            this.AlgoCombo.SelectedIndexChanged += new System.EventHandler(this.OnMCSAlgoSelChanged);
            // 
            // MomentumFitterList
            // 
            this.MomentumFitterList.FormattingEnabled = true;
            this.MomentumFitterList.Location = new System.Drawing.Point(166, 163);
            this.MomentumFitterList.Name = "MomentumFitterList";
            this.MomentumFitterList.Size = new System.Drawing.Size(317, 69);
            this.MomentumFitterList.TabIndex = 17;
            // 
            // textIgnoreDeltaSlope
            // 
            this.textIgnoreDeltaSlope.Location = new System.Drawing.Point(292, 138);
            this.textIgnoreDeltaSlope.Name = "textIgnoreDeltaSlope";
            this.textIgnoreDeltaSlope.Size = new System.Drawing.Size(39, 20);
            this.textIgnoreDeltaSlope.TabIndex = 16;
            this.textIgnoreDeltaSlope.Leave += new System.EventHandler(this.OnIgnoreDeltaSlopeLeave);
            // 
            // textMomentumResult
            // 
            this.textMomentumResult.Location = new System.Drawing.Point(6, 211);
            this.textMomentumResult.Name = "textMomentumResult";
            this.textMomentumResult.ReadOnly = true;
            this.textMomentumResult.Size = new System.Drawing.Size(154, 20);
            this.textMomentumResult.TabIndex = 9;
            // 
            // ExportButton
            // 
            this.ExportButton.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ExportButton.Location = new System.Drawing.Point(3, 166);
            this.ExportButton.Name = "ExportButton";
            this.ExportButton.Size = new System.Drawing.Size(157, 27);
            this.ExportButton.TabIndex = 8;
            this.ExportButton.Text = "Export to momentum fitter";
            this.ExportButton.UseVisualStyleBackColor = true;
            this.ExportButton.Click += new System.EventHandler(this.ExportButton_Click);
            // 
            // buttonIgnoreDeltaSlope
            // 
            this.buttonIgnoreDeltaSlope.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonIgnoreDeltaSlope.Location = new System.Drawing.Point(211, 134);
            this.buttonIgnoreDeltaSlope.Name = "buttonIgnoreDeltaSlope";
            this.buttonIgnoreDeltaSlope.Size = new System.Drawing.Size(75, 27);
            this.buttonIgnoreDeltaSlope.TabIndex = 15;
            this.buttonIgnoreDeltaSlope.Text = "DSlope <";
            this.buttonIgnoreDeltaSlope.UseVisualStyleBackColor = true;
            this.buttonIgnoreDeltaSlope.Click += new System.EventHandler(this.buttonIgnoreDeltaSlope_Click);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(3, 196);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(112, 13);
            this.label13.TabIndex = 12;
            this.label13.Text = "P ;  P min ; P max ; CL";
            // 
            // textMeasIgnoreGrains
            // 
            this.textMeasIgnoreGrains.Location = new System.Drawing.Point(166, 138);
            this.textMeasIgnoreGrains.Name = "textMeasIgnoreGrains";
            this.textMeasIgnoreGrains.Size = new System.Drawing.Size(39, 20);
            this.textMeasIgnoreGrains.TabIndex = 14;
            this.textMeasIgnoreGrains.Leave += new System.EventHandler(this.OnIgnoreGrainsLeave);
            // 
            // SlopeList
            // 
            this.SlopeList.CheckBoxes = true;
            this.SlopeList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader30,
            this.columnHeader26,
            this.columnHeader27,
            this.columnHeader28,
            this.columnHeader29});
            this.SlopeList.FullRowSelect = true;
            this.SlopeList.GridLines = true;
            this.SlopeList.Location = new System.Drawing.Point(3, 3);
            this.SlopeList.Name = "SlopeList";
            this.SlopeList.Size = new System.Drawing.Size(480, 127);
            this.SlopeList.TabIndex = 13;
            this.SlopeList.UseCompatibleStateImageBehavior = false;
            this.SlopeList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader30
            // 
            this.columnHeader30.Text = "Sheet";
            // 
            // columnHeader26
            // 
            this.columnHeader26.Text = "Z";
            // 
            // columnHeader27
            // 
            this.columnHeader27.Text = "Grains";
            // 
            // columnHeader28
            // 
            this.columnHeader28.Text = "SlopeX";
            // 
            // columnHeader29
            // 
            this.columnHeader29.Text = "SlopeY";
            // 
            // buttonMeasIgnoreGrains
            // 
            this.buttonMeasIgnoreGrains.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonMeasIgnoreGrains.Location = new System.Drawing.Point(86, 134);
            this.buttonMeasIgnoreGrains.Name = "buttonMeasIgnoreGrains";
            this.buttonMeasIgnoreGrains.Size = new System.Drawing.Size(74, 27);
            this.buttonMeasIgnoreGrains.TabIndex = 11;
            this.buttonMeasIgnoreGrains.Text = "Grains >=";
            this.buttonMeasIgnoreGrains.UseVisualStyleBackColor = true;
            this.buttonMeasIgnoreGrains.Click += new System.EventHandler(this.buttonMeasIgnoreGrains_Click);
            // 
            // buttonMeasSelAll
            // 
            this.buttonMeasSelAll.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonMeasSelAll.Location = new System.Drawing.Point(3, 134);
            this.buttonMeasSelAll.Name = "buttonMeasSelAll";
            this.buttonMeasSelAll.Size = new System.Drawing.Size(74, 27);
            this.buttonMeasSelAll.TabIndex = 10;
            this.buttonMeasSelAll.Text = "Select all";
            this.buttonMeasSelAll.UseVisualStyleBackColor = true;
            this.buttonMeasSelAll.Click += new System.EventHandler(this.buttonMeasSelAll_Click);
            // 
            // tabPage7
            // 
            this.tabPage7.Controls.Add(this.cmdSpecialAttributesReference);
            this.tabPage7.Controls.Add(this.txtAttribImportValue);
            this.tabPage7.Controls.Add(this.txtAttribImportTk);
            this.tabPage7.Controls.Add(this.cmbAttribImport);
            this.tabPage7.Controls.Add(this.cmdImportAttribute);
            this.tabPage7.Controls.Add(this.txtAttrValue);
            this.tabPage7.Controls.Add(this.txtAttrName);
            this.tabPage7.Controls.Add(this.cmdAddSetAttribute);
            this.tabPage7.Controls.Add(this.cmdRemoveAttributes);
            this.tabPage7.Controls.Add(this.AttributeList);
            this.tabPage7.Location = new System.Drawing.Point(4, 22);
            this.tabPage7.Name = "tabPage7";
            this.tabPage7.Size = new System.Drawing.Size(488, 296);
            this.tabPage7.TabIndex = 6;
            this.tabPage7.Text = "Attributes";
            this.tabPage7.UseVisualStyleBackColor = true;
            // 
            // cmdSpecialAttributesReference
            // 
            this.cmdSpecialAttributesReference.Location = new System.Drawing.Point(14, 211);
            this.cmdSpecialAttributesReference.Name = "cmdSpecialAttributesReference";
            this.cmdSpecialAttributesReference.Size = new System.Drawing.Size(149, 24);
            this.cmdSpecialAttributesReference.TabIndex = 48;
            this.cmdSpecialAttributesReference.Text = "Special attributes";
            this.cmdSpecialAttributesReference.Click += new System.EventHandler(this.cmdSpecialAttributesReference_Click);
            // 
            // txtAttribImportValue
            // 
            this.txtAttribImportValue.Location = new System.Drawing.Point(363, 256);
            this.txtAttribImportValue.Name = "txtAttribImportValue";
            this.txtAttribImportValue.ReadOnly = true;
            this.txtAttribImportValue.Size = new System.Drawing.Size(120, 20);
            this.txtAttribImportValue.TabIndex = 47;
            // 
            // txtAttribImportTk
            // 
            this.txtAttribImportTk.Location = new System.Drawing.Point(169, 256);
            this.txtAttribImportTk.Name = "txtAttribImportTk";
            this.txtAttribImportTk.Size = new System.Drawing.Size(82, 20);
            this.txtAttribImportTk.TabIndex = 46;
            this.txtAttribImportTk.Leave += new System.EventHandler(this.OnSelAttribChanged);
            // 
            // cmbAttribImport
            // 
            this.cmbAttribImport.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbAttribImport.FormattingEnabled = true;
            this.cmbAttribImport.Location = new System.Drawing.Point(257, 256);
            this.cmbAttribImport.Name = "cmbAttribImport";
            this.cmbAttribImport.Size = new System.Drawing.Size(100, 21);
            this.cmbAttribImport.TabIndex = 45;
            this.cmbAttribImport.SelectedIndexChanged += new System.EventHandler(this.OnSelAttribChanged);
            // 
            // cmdImportAttribute
            // 
            this.cmdImportAttribute.Location = new System.Drawing.Point(14, 253);
            this.cmdImportAttribute.Name = "cmdImportAttribute";
            this.cmdImportAttribute.Size = new System.Drawing.Size(149, 24);
            this.cmdImportAttribute.TabIndex = 44;
            this.cmdImportAttribute.Text = "Import attribute from track";
            this.cmdImportAttribute.Click += new System.EventHandler(this.cmdImportAttribute_Click);
            // 
            // txtAttrValue
            // 
            this.txtAttrValue.Location = new System.Drawing.Point(257, 170);
            this.txtAttrValue.Name = "txtAttrValue";
            this.txtAttrValue.Size = new System.Drawing.Size(226, 20);
            this.txtAttrValue.TabIndex = 43;
            // 
            // txtAttrName
            // 
            this.txtAttrName.Location = new System.Drawing.Point(88, 170);
            this.txtAttrName.Name = "txtAttrName";
            this.txtAttrName.Size = new System.Drawing.Size(163, 20);
            this.txtAttrName.TabIndex = 41;
            // 
            // cmdAddSetAttribute
            // 
            this.cmdAddSetAttribute.Location = new System.Drawing.Point(14, 170);
            this.cmdAddSetAttribute.Name = "cmdAddSetAttribute";
            this.cmdAddSetAttribute.Size = new System.Drawing.Size(68, 24);
            this.cmdAddSetAttribute.TabIndex = 40;
            this.cmdAddSetAttribute.Text = "Add/Set";
            this.cmdAddSetAttribute.Click += new System.EventHandler(this.cmdAddSetAttribute_Click);
            // 
            // cmdRemoveAttributes
            // 
            this.cmdRemoveAttributes.Location = new System.Drawing.Point(311, 211);
            this.cmdRemoveAttributes.Name = "cmdRemoveAttributes";
            this.cmdRemoveAttributes.Size = new System.Drawing.Size(172, 24);
            this.cmdRemoveAttributes.TabIndex = 39;
            this.cmdRemoveAttributes.Text = "Remove selected attribute(s)";
            this.cmdRemoveAttributes.Click += new System.EventHandler(this.cmdRemoveAttributes_Click);
            // 
            // AttributeList
            // 
            this.AttributeList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader25,
            this.columnHeader34});
            this.AttributeList.FullRowSelect = true;
            this.AttributeList.GridLines = true;
            this.AttributeList.Location = new System.Drawing.Point(14, 12);
            this.AttributeList.Name = "AttributeList";
            this.AttributeList.Size = new System.Drawing.Size(469, 152);
            this.AttributeList.TabIndex = 0;
            this.AttributeList.UseCompatibleStateImageBehavior = false;
            this.AttributeList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader25
            // 
            this.columnHeader25.Text = "Index";
            this.columnHeader25.Width = 280;
            // 
            // columnHeader34
            // 
            this.columnHeader34.Text = "Value";
            this.columnHeader34.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader34.Width = 120;
            // 
            // tabPage8
            // 
            this.tabPage8.Controls.Add(this.label23);
            this.tabPage8.Controls.Add(this.cmbFBDecaySearch);
            this.tabPage8.Controls.Add(this.btnFBSingleProngVertex);
            this.tabPage8.Controls.Add(this.txtFBSingleProngVertexZ);
            this.tabPage8.Controls.Add(this.chkFBManual);
            this.tabPage8.Controls.Add(this.btnFBImportAttributes);
            this.tabPage8.Controls.Add(this.cmbFBTkImportList);
            this.tabPage8.Controls.Add(this.txtFBSlopeTol);
            this.tabPage8.Controls.Add(this.label18);
            this.tabPage8.Controls.Add(this.btnFBFindSBSFTrack);
            this.tabPage8.Controls.Add(this.cmbFBLastPlate);
            this.tabPage8.Controls.Add(this.label17);
            this.tabPage8.Controls.Add(this.label16);
            this.tabPage8.Controls.Add(this.cmbFBOutOfBrick);
            this.tabPage8.Controls.Add(this.label15);
            this.tabPage8.Controls.Add(this.cmbFBDarkness);
            this.tabPage8.Controls.Add(this.label14);
            this.tabPage8.Controls.Add(this.cmbFBParticle);
            this.tabPage8.Controls.Add(this.chkFBScanback);
            this.tabPage8.Controls.Add(this.chkFBEvent);
            this.tabPage8.Location = new System.Drawing.Point(4, 22);
            this.tabPage8.Name = "tabPage8";
            this.tabPage8.Size = new System.Drawing.Size(488, 296);
            this.tabPage8.TabIndex = 7;
            this.tabPage8.Text = "Feedback";
            this.tabPage8.UseVisualStyleBackColor = true;
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.Location = new System.Drawing.Point(10, 240);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(75, 13);
            this.label23.TabIndex = 25;
            this.label23.Text = "Decay Search";
            // 
            // cmbFBDecaySearch
            // 
            this.cmbFBDecaySearch.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbFBDecaySearch.FormattingEnabled = true;
            this.cmbFBDecaySearch.Location = new System.Drawing.Point(100, 237);
            this.cmbFBDecaySearch.Name = "cmbFBDecaySearch";
            this.cmbFBDecaySearch.Size = new System.Drawing.Size(166, 21);
            this.cmbFBDecaySearch.TabIndex = 24;
            this.cmbFBDecaySearch.SelectedIndexChanged += new System.EventHandler(this.OnDecaySearchChanged);
            // 
            // btnFBSingleProngVertex
            // 
            this.btnFBSingleProngVertex.Location = new System.Drawing.Point(212, 8);
            this.btnFBSingleProngVertex.Name = "btnFBSingleProngVertex";
            this.btnFBSingleProngVertex.Size = new System.Drawing.Size(175, 25);
            this.btnFBSingleProngVertex.TabIndex = 23;
            this.btnFBSingleProngVertex.Text = "Add single-prong vertex at Z = ";
            this.btnFBSingleProngVertex.UseVisualStyleBackColor = true;
            this.btnFBSingleProngVertex.Visible = false;
            this.btnFBSingleProngVertex.Click += new System.EventHandler(this.btnFBSingleProngVertex_Click);
            // 
            // txtFBSingleProngVertexZ
            // 
            this.txtFBSingleProngVertexZ.Location = new System.Drawing.Point(393, 11);
            this.txtFBSingleProngVertexZ.Name = "txtFBSingleProngVertexZ";
            this.txtFBSingleProngVertexZ.Size = new System.Drawing.Size(75, 20);
            this.txtFBSingleProngVertexZ.TabIndex = 22;
            this.txtFBSingleProngVertexZ.Visible = false;
            this.txtFBSingleProngVertexZ.TextChanged += new System.EventHandler(this.OnFBSingleProngVertexZChanged);
            // 
            // chkFBManual
            // 
            this.chkFBManual.AutoSize = true;
            this.chkFBManual.Location = new System.Drawing.Point(100, 201);
            this.chkFBManual.Name = "chkFBManual";
            this.chkFBManual.Size = new System.Drawing.Size(119, 17);
            this.chkFBManual.TabIndex = 20;
            this.chkFBManual.Text = "Manually recovered";
            this.chkFBManual.UseVisualStyleBackColor = true;
            this.chkFBManual.CheckedChanged += new System.EventHandler(this.OnFBManualChecked);
            // 
            // btnFBImportAttributes
            // 
            this.btnFBImportAttributes.Location = new System.Drawing.Point(13, 158);
            this.btnFBImportAttributes.Name = "btnFBImportAttributes";
            this.btnFBImportAttributes.Size = new System.Drawing.Size(206, 25);
            this.btnFBImportAttributes.TabIndex = 19;
            this.btnFBImportAttributes.Text = "Import attributes from track";
            this.btnFBImportAttributes.UseVisualStyleBackColor = true;
            this.btnFBImportAttributes.Click += new System.EventHandler(this.btnFBImportAttributes_Click);
            // 
            // cmbFBTkImportList
            // 
            this.cmbFBTkImportList.FormattingEnabled = true;
            this.cmbFBTkImportList.Location = new System.Drawing.Point(228, 161);
            this.cmbFBTkImportList.Name = "cmbFBTkImportList";
            this.cmbFBTkImportList.Size = new System.Drawing.Size(129, 21);
            this.cmbFBTkImportList.TabIndex = 18;
            this.cmbFBTkImportList.TextChanged += new System.EventHandler(this.OnFBTkImportListTextChanged);
            // 
            // txtFBSlopeTol
            // 
            this.txtFBSlopeTol.Location = new System.Drawing.Point(338, 127);
            this.txtFBSlopeTol.Name = "txtFBSlopeTol";
            this.txtFBSlopeTol.Size = new System.Drawing.Size(75, 20);
            this.txtFBSlopeTol.TabIndex = 17;
            this.txtFBSlopeTol.Leave += new System.EventHandler(this.OnFBSlopeTolLeave);
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(225, 133);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(81, 13);
            this.label18.TabIndex = 16;
            this.label18.Text = "Slope tolerance";
            // 
            // btnFBFindSBSFTrack
            // 
            this.btnFBFindSBSFTrack.Location = new System.Drawing.Point(13, 127);
            this.btnFBFindSBSFTrack.Name = "btnFBFindSBSFTrack";
            this.btnFBFindSBSFTrack.Size = new System.Drawing.Size(206, 25);
            this.btnFBFindSBSFTrack.TabIndex = 15;
            this.btnFBFindSBSFTrack.Text = "Fill list of compatible SB/SF tracks";
            this.btnFBFindSBSFTrack.UseVisualStyleBackColor = true;
            this.btnFBFindSBSFTrack.Click += new System.EventHandler(this.btnFBFindSBSFTrack_Click);
            // 
            // cmbFBLastPlate
            // 
            this.cmbFBLastPlate.FormattingEnabled = true;
            this.cmbFBLastPlate.Location = new System.Drawing.Point(338, 100);
            this.cmbFBLastPlate.Name = "cmbFBLastPlate";
            this.cmbFBLastPlate.Size = new System.Drawing.Size(130, 21);
            this.cmbFBLastPlate.TabIndex = 14;
            this.cmbFBLastPlate.SelectedIndexChanged += new System.EventHandler(this.OnFBLastPlateChanged);
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(272, 103);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(53, 13);
            this.label17.TabIndex = 13;
            this.label17.Text = "Last plate";
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(10, 103);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(62, 13);
            this.label16.TabIndex = 12;
            this.label16.Text = "Out of brick";
            // 
            // cmbFBOutOfBrick
            // 
            this.cmbFBOutOfBrick.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbFBOutOfBrick.FormattingEnabled = true;
            this.cmbFBOutOfBrick.Location = new System.Drawing.Point(80, 100);
            this.cmbFBOutOfBrick.Name = "cmbFBOutOfBrick";
            this.cmbFBOutOfBrick.Size = new System.Drawing.Size(186, 21);
            this.cmbFBOutOfBrick.TabIndex = 11;
            this.cmbFBOutOfBrick.SelectedIndexChanged += new System.EventHandler(this.OnFBOutOfBrickChanged);
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(10, 76);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(52, 13);
            this.label15.TabIndex = 10;
            this.label15.Text = "Darkness";
            // 
            // cmbFBDarkness
            // 
            this.cmbFBDarkness.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbFBDarkness.FormattingEnabled = true;
            this.cmbFBDarkness.Location = new System.Drawing.Point(80, 73);
            this.cmbFBDarkness.Name = "cmbFBDarkness";
            this.cmbFBDarkness.Size = new System.Drawing.Size(186, 21);
            this.cmbFBDarkness.TabIndex = 9;
            this.cmbFBDarkness.SelectedIndexChanged += new System.EventHandler(this.OnFBDarknessChanged);
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(10, 49);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(42, 13);
            this.label14.TabIndex = 8;
            this.label14.Text = "Particle";
            // 
            // cmbFBParticle
            // 
            this.cmbFBParticle.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbFBParticle.FormattingEnabled = true;
            this.cmbFBParticle.Location = new System.Drawing.Point(80, 46);
            this.cmbFBParticle.Name = "cmbFBParticle";
            this.cmbFBParticle.Size = new System.Drawing.Size(186, 21);
            this.cmbFBParticle.TabIndex = 7;
            this.cmbFBParticle.SelectedIndexChanged += new System.EventHandler(this.OnFBParticleTypeChanged);
            // 
            // chkFBScanback
            // 
            this.chkFBScanback.AutoSize = true;
            this.chkFBScanback.Location = new System.Drawing.Point(13, 201);
            this.chkFBScanback.Name = "chkFBScanback";
            this.chkFBScanback.Size = new System.Drawing.Size(75, 17);
            this.chkFBScanback.TabIndex = 6;
            this.chkFBScanback.Text = "Scanback";
            this.chkFBScanback.UseVisualStyleBackColor = true;
            this.chkFBScanback.CheckedChanged += new System.EventHandler(this.OnFBScanbackChecked);
            // 
            // chkFBEvent
            // 
            this.chkFBEvent.AutoSize = true;
            this.chkFBEvent.Location = new System.Drawing.Point(13, 13);
            this.chkFBEvent.Name = "chkFBEvent";
            this.chkFBEvent.Size = new System.Drawing.Size(150, 17);
            this.chkFBEvent.TabIndex = 5;
            this.chkFBEvent.Text = "Include track in Feedback";
            this.chkFBEvent.UseVisualStyleBackColor = true;
            this.chkFBEvent.CheckedChanged += new System.EventHandler(this.OnFBEventChecked);
            // 
            // columnHeader20
            // 
            this.columnHeader20.Text = "Plate";
            // 
            // columnHeader21
            // 
            this.columnHeader21.Text = "Z";
            // 
            // columnHeader22
            // 
            this.columnHeader22.Text = "Grains";
            // 
            // columnHeader23
            // 
            this.columnHeader23.Text = "SlopeX";
            // 
            // columnHeader24
            // 
            this.columnHeader24.Text = "SlopeY";
            // 
            // TrackBrowser
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(507, 360);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.EnableLabelCheck);
            this.Controls.Add(this.HighlightCheck);
            this.Controls.Add(this.SetLabelButton);
            this.Controls.Add(this.LabelText);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "TrackBrowser";
            this.Text = "TrackBrowser";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Closed += new System.EventHandler(this.OnClose);
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.tabPage5.ResumeLayout(false);
            this.tabPage5.PerformLayout();
            this.tabPage6.ResumeLayout(false);
            this.tabPage6.PerformLayout();
            this.tabPage7.ResumeLayout(false);
            this.tabPage7.PerformLayout();
            this.tabPage8.ResumeLayout(false);
            this.tabPage8.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		private void SetCommentButton_Click(object sender, System.EventArgs e)
		{
			m_Track.Comment = CommentText.Text;
			GeneralList.Items[2].SubItems[1].Text = m_Track.Comment;
		}

		private void GeneralSelButton_Click(object sender, System.EventArgs e)
		{
			System.Windows.Forms.SaveFileDialog ddlg = new System.Windows.Forms.SaveFileDialog();
			ddlg.Title = "Select file to dump info of track #" + m_Track.Id.ToString();
			ddlg.FileName = (GeneralDumpFileText.Text.Length == 0) ? ("TkInfo_" + m_Track.Id.ToString() + ".txt") : GeneralDumpFileText.Text;
			ddlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (ddlg.ShowDialog() == DialogResult.OK) GeneralDumpFileText.Text = ddlg.FileName;
		}

		private void GeneralDumpFileButton_Click(object sender, System.EventArgs e)
		{
			System.IO.StreamWriter w = null;
			try
			{
				w = new System.IO.StreamWriter(GeneralDumpFileText.Text);
				foreach (System.Windows.Forms.ListViewItem lvi in GeneralList.Items)				
					w.WriteLine(lvi.SubItems[0].Text + "\t" + lvi.SubItems[1].Text);				
				w.Flush();
				w.Close();
				MessageBox.Show("Dump OK", "OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
			catch (Exception x)
			{
				if (w != null) w.Close();
				MessageBox.Show(x.ToString(), "Error dumping info", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void SegDumpSelButton_Click(object sender, System.EventArgs e)
		{
			System.Windows.Forms.SaveFileDialog sdlg = new System.Windows.Forms.SaveFileDialog();
			sdlg.Title = "Select file to dump segments of track #" + m_Track.Id.ToString();
			sdlg.FileName = (SegDumpFileText.Text.Length == 0) ? ("TkSeg_" + m_Track.Id.ToString() + ".txt") : SegDumpFileText.Text;
			sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";		
			if (sdlg.ShowDialog() == DialogResult.OK) SegDumpFileText.Text = sdlg.FileName;
		}

		private void SegDumpButton_Click(object sender, System.EventArgs e)
		{
			System.IO.StreamWriter w = null;
			try
			{
				w = new System.IO.StreamWriter(SegDumpFileText.Text);
				int i, c;
				c = SegmentList.Columns.Count;
				for (i = 0; i < c; i++)
					if (i == 0) w.Write(SegmentList.Columns[i].Text);
					else w.Write("\t" + SegmentList.Columns[i].Text);
				w.WriteLine();
				foreach (System.Windows.Forms.ListViewItem lvi in SegmentList.Items)				
				{
					for (i = 0; i < lvi.SubItems.Count; i++)
						if (i == 0) w.Write(lvi.SubItems[i].Text);
						else w.Write("\t" + lvi.SubItems[i].Text);
					w.WriteLine();
				}
				w.Flush();
				w.Close();
				MessageBox.Show("Dump OK", "OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
			catch (Exception x)
			{
				if (w != null) w.Close();
				MessageBox.Show(x.ToString(), "Error dumping info", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}		
		}

		private void GoUpVtxButton_Click(object sender, System.EventArgs e)
		{
			VertexBrowser.Browse(m_Track.Upstream_Vertex, m_Layers,SelectForIP, SelectForGraph, m_Event, m_V);
		}

		private void GoDownVtxButton_Click(object sender, System.EventArgs e)
		{
			VertexBrowser.Browse(m_Track.Downstream_Vertex, m_Layers, SelectForIP, SelectForGraph, m_Event, m_V);
		}

		internal static System.Collections.ArrayList AvailableBrowsers = new System.Collections.ArrayList();

        public static void CloseAll()
        {
            while (AvailableBrowsers.Count > 0) ((TrackBrowser)AvailableBrowsers[0]).Close();
        }

        public static void RefreshAll()
        {
            foreach (TrackBrowser b in AvailableBrowsers) b.Track = b.Track;
        }

		public static TrackBrowser Browse(SySal.TotalScan.Track tk, SySal.TotalScan.Volume.LayerList ll, SySal.Executables.EasyReconstruct.IPSelector ipsel, SySal.Executables.EasyReconstruct.TrackSelector tksel, long ev, SySal.TotalScan.Volume v)
		{
			foreach (TrackBrowser b in AvailableBrowsers)
			{
				if (b.Track == tk)
				{
					b.BringToFront();
					return b;
				}
			}
			TrackBrowser newb = new TrackBrowser(ll, ev, v);
			newb.Track = tk;
            newb.SelectForIP = ipsel;
            newb.SelectForGraph = tksel;
			newb.Show();
			AvailableBrowsers.Add(newb);
            tksel.SubscribeOnAddFit(new dGenericEvent(newb.RefreshFitList_Click));
            newb.RefreshFitList_Click(tksel, null);
			return newb;
		}

		private void OnClose(object sender, System.EventArgs e)
		{
            SelectForGraph.ToggleAddReplaceSegments(null);
            try
            {
                SelectForGraph.UnsubscribeOnAddFit(new dGenericEvent(this.RefreshFitList_Click));
                MomentumFitForm.UnsubscribeOnUpdateBrowsers(new dGenericEvent(OnUpdateMomentumFitterList));
                AvailableBrowsers.Remove(this);
            }
            catch (Exception) { }
        }

        private void OnProjectLayerChanged(object sender, EventArgs e)
        {
            if (LayerList.SelectedIndex >= 0)
            {
                SySal.TotalScan.Layer l = m_Layers[LayerList.SelectedIndex];
                SySal.Tracking.MIPEmulsionTrackInfo info = m_Track.Fit(l.Id, l.RefCenter.Z);
                InfoText.Text = info.Intercept.X.ToString("F1") + " " + info.Intercept.Y.ToString("F1") + " " + info.Slope.X.ToString("F5") + " " + info.Slope.Y.ToString("F5");
                info.Intercept = l.ToOriginalPoint(info.Intercept);
                info.Slope = l.ToOriginalSlope(info.Slope);
                OrigInfoText.Text = info.Intercept.X.ToString("F1") + " " + info.Intercept.Y.ToString("F1") + " " + info.Slope.X.ToString("F5") + " " + info.Slope.Y.ToString("F5");
            }
        }

        private void SelectIPFirstButton_Click(object sender, EventArgs e)
        {
            SelectForIP.SelectTrack(m_Track, true);
        }

        private void SelectIPSecondButton_Click(object sender, EventArgs e)
        {
            SelectForIP.SelectTrack(m_Track, false);
        }

        private double m_Opening = 0.05;

        private void OnOpeningLeave(object sender, EventArgs e)
        {
            try
            {
                m_Opening = System.Convert.ToDouble(txtOpening.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtOpening.Text = m_Opening.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtOpening.Focus();
            }
        }

        private double m_Radius = 100.0;

        private void OnRadiusLeave(object sender, EventArgs e)
        {
            try
            {
                m_Radius = System.Convert.ToDouble(txtRadius.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtRadius.Text = m_Radius.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtRadius.Focus();
            }
        }

        private double m_DeltaSlope = 0.05;

        private void OnDeltaSlopeLeave(object sender, EventArgs e)
        {
            try
            {
                m_DeltaSlope = System.Convert.ToDouble(txtDeltaSlope.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtDeltaSlope.Text = m_DeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtDeltaSlope.Focus();
            }
        }

        private double m_DeltaZ = 100000.0;

        private void OnDeltaZLeave(object sender, EventArgs e)
        {
            try
            {
                m_DeltaZ = System.Convert.ToDouble(txtDeltaZ.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtDeltaZ.Text = m_DeltaZ.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtDeltaZ.Focus();
            }
        }

        private double m_IP = 20.0;

        private void OnIPLeave(object sender, EventArgs e)
        {
            try
            {
                m_IP = System.Convert.ToDouble(txtIP.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtIP.Text = m_IP.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtIP.Focus();
            }
        }

        public enum FBParticleType { Unknown = 0, Muon = 13, Electron = 11, EPair = 22, Tauon = 15, Charm = 4 }

        public enum FBDarkness { MIP = 0, Grey = 1, Black = 2 }

        public enum FBOutOfBrick { N = 0, PassingThrough = 1, EdgeOut = 2 }

        internal struct PlateDesc
        {
            public long Brick;
            public int Plate;
            public override string ToString()
            {
                return Plate + " (" + Brick + ")";
            }
        }

        public enum FBDecaySearch { Null = 0, PrimaryVertexTrack = 1, EPlusEMinus = 2, LowMomentum = 3, ScanForthToDo = 4, ScanForthDone = 5 }   

        private void OnLoad(object sender, EventArgs e)
        {
            foreach (string s in XSegInfo.ExtendedFields)            
                SegmentList.Columns.Add(s, 60);            
            m_IsRefreshing = true;
            if (MCSAlgorithms != null && MCSAlgorithms.Length > 0)
            {
                foreach (SySal.TotalScan.IMCSMomentumEstimator algo in MCSAlgorithms)
                    AlgoCombo.Items.Add(algo);
                if (AlgoCombo.Items.Count > 0) AlgoCombo.SelectedIndex = 0;
                if (MomentumFitForm.MCSLikelihood == null)
                {
                    ExportButton.Enabled = false;
                    ExportButton.Text = "Bad MCS geometry";
                }
                OnMCSAlgoSelChanged(sender, e);
            }
            cmbAttribImport.Items.Add(ImportableAttribute.UpstreamIP);
            cmbAttribImport.Items.Add(ImportableAttribute.DownstreamIP);
            cmbAttribImport.SelectedIndex = 0;
            cmbFBDecaySearch.Items.Add(FBDecaySearch.Null);
            cmbFBDecaySearch.Items.Add(FBDecaySearch.PrimaryVertexTrack);
            cmbFBDecaySearch.Items.Add(FBDecaySearch.EPlusEMinus);
            cmbFBDecaySearch.Items.Add(FBDecaySearch.LowMomentum);
            cmbFBDecaySearch.Items.Add(FBDecaySearch.ScanForthToDo);
            cmbFBDecaySearch.Items.Add(FBDecaySearch.ScanForthDone);
            cmbFBDecaySearch.SelectedIndex = 0;
            cmbFBParticle.Items.Add(FBParticleType.Unknown);
            cmbFBParticle.Items.Add(FBParticleType.Muon);
            cmbFBParticle.Items.Add(FBParticleType.Electron);
            cmbFBParticle.Items.Add(FBParticleType.EPair);
            cmbFBParticle.Items.Add(FBParticleType.Charm);
            cmbFBParticle.Items.Add(FBParticleType.Tauon);
            cmbFBParticle.SelectedIndex = 0;
            cmbFBDarkness.Items.Add(FBDarkness.MIP);
            cmbFBDarkness.Items.Add(FBDarkness.Grey);
            cmbFBDarkness.Items.Add(FBDarkness.Black);
            cmbFBDarkness.SelectedIndex = 0;
            cmbFBOutOfBrick.Items.Add(FBOutOfBrick.N);
            cmbFBOutOfBrick.Items.Add(FBOutOfBrick.PassingThrough);
            cmbFBOutOfBrick.Items.Add(FBOutOfBrick.EdgeOut);
            cmbFBOutOfBrick.SelectedIndex = 0;
            System.Collections.ArrayList pl = new ArrayList();
            int i;
            for (i = 0; i < m_Layers.Length; i++)
            {
                SySal.TotalScan.Layer lay = m_Layers[i];
                PlateDesc pld = new PlateDesc();
                pld.Brick = lay.BrickId;
                pld.Plate = lay.SheetId;
                if (pl.Contains(pld) == false) pl.Add(pld);
            }
            foreach (PlateDesc pld in pl)
                cmbFBLastPlate.Items.Add(pld);
            if (cmbFBLastPlate.Items.Count > 0) cmbFBLastPlate.SelectedIndex = cmbFBLastPlate.Items.Count - 1;
            textIgnoreDeltaSlope.Text = m_IgnoreDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            textMeasIgnoreGrains.Text = m_IgnoreGrains.ToString();
            txtOpening.Text = m_Opening.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtRadius.Text = m_Radius.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDeltaSlope.Text = m_DeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDeltaZ.Text = m_DeltaZ.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtIP.Text = m_IP.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtExtrapolationDist.Text = m_ExtrapDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMaxMisses.Text = m_MaxMisses.ToString();
            txtFBSlopeTol.Text = m_FBSlopeTol.ToString(System.Globalization.CultureInfo.InvariantCulture);
            DownSegsText.Text = m_DownSegs.ToString();
            UpSegsText.Text = m_UpSegs.ToString();
            DownStopsText.Text = m_DownStops.ToString();
            UpStopsText.Text = m_UpStops.ToString();
            //WeightFromQualityButton_Click(this, null);
            m_Weight = 1.0;
            txtWeight.Text = m_Weight.ToString(System.Globalization.CultureInfo.InvariantCulture);
            try
            {
                m_Momentum = m_Track.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("P"));
            }
            catch (Exception)
            {
                m_Momentum = 0.0;
            }
            m_MomentumLikelihood = new NumericalTools.OneParamLogLikelihood(m_Momentum, m_Momentum, new double[1] { 1.0 }, "P");
            txtMomentum.Text = m_Momentum.ToString(System.Globalization.CultureInfo.InvariantCulture);
            OnUpdateMomentumFitterList(this, null);
            MomentumFitForm.SubscribeOnUpdateBrowsers(new dGenericEvent(OnUpdateMomentumFitterList));
            RefreshFeedbackStatus();
            AppendToFileText.Text = DisplayForm.DefaultProjFileName;
            m_IsRefreshing = false;
        }

        public class TrackFilter
        {
            SySal.BasicTypes.Vector m_Start;

            SySal.BasicTypes.Vector m_Slope;

            bool m_IsDownstreamDir;            

            double m_Opening;

            double m_Radius;

            double m_DeltaSlope;

            double m_DeltaZ;

            double m_IP;

            SySal.TotalScan.VertexFit m_VF;

            public TrackFilter(SySal.TotalScan.Track tk, bool isdownstream, bool isdownstreamdir, double opening, double radius, double deltaslope, double deltaz, double ip)
            {
                if (isdownstream)
                {
                    m_Start.X = tk.Downstream_PosX + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeX;
                    m_Start.Y = tk.Downstream_PosY + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeY;
                    m_Start.Z = tk.Downstream_Z;
                }
                else
                {
                    m_Start.X = tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX;
                    m_Start.Y = tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY;
                    m_Start.Z = tk.Upstream_Z;                    
                }
                if (isdownstreamdir)
                {
                    m_Slope.X = tk.Downstream_SlopeX;
                    m_Slope.Y = tk.Downstream_SlopeY;
                }
                else
                {
                    m_Slope.X = tk.Upstream_SlopeX;
                    m_Slope.Y = tk.Upstream_SlopeY;
                }
                m_Slope.Z = 1.0;
                m_IsDownstreamDir = isdownstreamdir;
                m_Opening = opening;
                m_Radius = radius;
                m_DeltaSlope = deltaslope;
                m_DeltaZ = deltaz;
                m_IP = ip;
                m_VF = new SySal.TotalScan.VertexFit();
                SySal.TotalScan.VertexFit.TrackFit tf = new SySal.TotalScan.VertexFit.TrackFit();
                if (isdownstream)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = tk[0].Info;
                    tf.Intercept = info.Intercept;
                    tf.Slope = info.Slope;
                    tf.MinZ = tk.Downstream_Z;
                    tf.MaxZ = tf.MinZ + m_DeltaZ;                    
                }
                else
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = tk[tk.Length - 1].Info;
                    tf.Intercept = info.Intercept;
                    tf.Slope = info.Slope;                    
                    tf.MaxZ = tk.Upstream_Z;
                    tf.MinZ = tf.MaxZ - m_DeltaZ;
                }   
                tf.Weight = 1.0;
                tf.Id = new SySal.TotalScan.BaseTrackIndex(tk.Id);
                m_VF.AddTrackFit(tf);
            }            

            public bool Filter(SySal.TotalScan.Track t)
            {                
                SySal.BasicTypes.Vector end;
                SySal.BasicTypes.Vector slope;
                if (m_IsDownstreamDir)
                {
                    //if (t.Upstream_Z < m_Start.Z) return false;
                    //if (t.Downstream_Z + m_DeltaZ < m_Start.Z) return false;
                    end.X = t.Downstream_PosX + (t.Downstream_Z - t.Downstream_PosZ) * t.Downstream_SlopeX;
                    end.Y = t.Downstream_PosY + (t.Downstream_Z - t.Downstream_PosZ) * t.Downstream_SlopeY;
                    end.Z = t.Downstream_Z;
                    slope.X = t.Downstream_SlopeX;
                    slope.Y = t.Downstream_SlopeY;
                    slope.Z = 1.0;
                }
                else
                {
                    //if (t.Downstream_Z > m_Start.Z) return false;
                    //if (t.Upstream_Z - m_DeltaZ > m_Start.Z) return false;
                    end.X = t.Upstream_PosX + (t.Upstream_Z - t.Upstream_PosZ) * t.Upstream_SlopeX;
                    end.Y = t.Upstream_PosY + (t.Upstream_Z - t.Upstream_PosZ) * t.Upstream_SlopeY;
                    end.Z = t.Upstream_Z;
                    slope.X = t.Upstream_SlopeX;
                    slope.Y = t.Upstream_SlopeY;
                    slope.Z = 1.0;
                }
                if (m_DeltaZ >= 0.0 && Math.Abs(end.Z - m_Start.Z) > m_DeltaZ) return false;
                if (m_DeltaSlope >= 0.0)
                    if ((slope.X - m_Slope.X) * (slope.X - m_Slope.X) + (slope.Y - m_Slope.Y) * (slope.Y - m_Slope.Y) > m_DeltaSlope * m_DeltaSlope) return false;
                if (m_Radius >= 0.0)
                {
                    double dx = m_Start.X + (end.Z - m_Start.Z) * m_Slope.X - end.X;
                    double dy = m_Start.Y + (end.Z - m_Start.Z) * m_Slope.Y - end.Y;
                    if (dx * dx + dy * dy > m_Radius * m_Radius) return false;
                }
                if (m_Opening >= 0.0)
                {
                    double dz = end.Z - m_Start.Z;
                    if (dz < 0.0 == m_IsDownstreamDir) return false;
                    if (dz != 0.0)
                    {
                        double dx = (end.X - m_Start.X) / dz - m_Slope.X;
                        double dy = (end.Y - m_Start.Y) / dz - m_Slope.Y;
                        if (dx * dx + dy * dy > m_Opening * m_Opening) return false;
                    }
                }
                if (m_IP >= 0.0 && m_DeltaZ >= 0.0)
                {
                    SySal.TotalScan.VertexFit.TrackFit tf = new SySal.TotalScan.VertexFit.TrackFit();
                    tf.Id = new SySal.TotalScan.BaseTrackIndex(-1);
                    try
                    {
                        if (m_IsDownstreamDir)
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo info = t[0].Info;
                            tf.Intercept = info.Intercept;
                            tf.Slope = info.Slope;
                            tf.MinZ = t.Downstream_Z;
                            tf.MaxZ = tf.MinZ + m_DeltaZ;
                        }
                        else
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo info = t[t.Length - 1].Info;
                            tf.Intercept = info.Intercept;
                            tf.Slope = info.Slope;
                            tf.MaxZ = t.Upstream_Z;
                            tf.MinZ = tf.MaxZ - m_DeltaZ;
                        }
                        tf.Weight = 1.0;
                        m_VF.AddTrackFit(tf);
                        return m_VF.AvgDistance <= 0.5 * m_IP;
                    }
                    catch (Exception) { return false; }
                    finally
                    {
                        if (m_VF.Count == 2)
                            m_VF.RemoveTrackFit(tf.Id);
                    }
                }
                return true;
            }

            public bool FilterSeg(SySal.TotalScan.Segment s)
            {
                SySal.BasicTypes.Vector end;
                SySal.BasicTypes.Vector slope;
                SySal.Tracking.MIPEmulsionTrackInfo info = s.Info;
                if (m_IsDownstreamDir)
                {
                    //if (info.BottomZ < m_Start.Z) return false;
                    end.X = info.Intercept.X + (info.TopZ - info.Intercept.Z) * info.Slope.X;
                    end.Y = info.Intercept.Y + (info.TopZ - info.Intercept.Z) * info.Slope.Y;
                    end.Z = info.TopZ;
                    slope.X = info.Slope.X;
                    slope.Y = info.Slope.Y;
                    slope.Z = 1.0;
                }
                else
                {
                    //if (info.TopZ > m_Start.Z) return false;
                    end.X = info.Intercept.X + (info.BottomZ - info.Intercept.Z) * info.Slope.X;
                    end.Y = info.Intercept.Y + (info.BottomZ - info.Intercept.Z) * info.Slope.Y;
                    end.Z = info.BottomZ;
                    slope.X = info.Slope.X;
                    slope.Y = info.Slope.Y;
                    slope.Z = 1.0;
                }
                if (m_DeltaZ >= 0.0 && Math.Abs(end.Z - m_Start.Z) > m_DeltaZ) return false;
                if (m_DeltaSlope >= 0.0)
                    if ((slope.X - m_Slope.X) * (slope.X - m_Slope.X) + (slope.Y - m_Slope.Y) * (slope.Y - m_Slope.Y) > m_DeltaSlope * m_DeltaSlope) return false;
                if (m_Radius >= 0.0)
                {
                    double dx = m_Start.X + (end.Z - m_Start.Z) * m_Slope.X - end.X;
                    double dy = m_Start.Y + (end.Z - m_Start.Z) * m_Slope.Y - end.Y;
                    if (dx * dx + dy * dy > m_Radius * m_Radius) return false;
                }
                if (m_Opening >= 0.0)
                {
                    double dz = end.Z - m_Start.Z;
                    if (dz < 0.0 == m_IsDownstreamDir) return false;
                    if (dz != 0.0)
                    {
                        double dx = (end.X - m_Start.X) / dz - m_Slope.X;
                        double dy = (end.Y - m_Start.Y) / dz - m_Slope.Y;
                        if (dx * dx + dy * dy > m_Opening * m_Opening) return false;
                    }
                }
                if (m_IP >= 0.0 && m_DeltaZ >= 0.0)
                {
                    SySal.TotalScan.VertexFit.TrackFit tf = new SySal.TotalScan.VertexFit.TrackFit();
                    tf.Id = new SySal.TotalScan.BaseTrackIndex(-1);
                    try
                    {
                        tf.Intercept = info.Intercept;
                        tf.Slope = info.Slope;
                        if (m_IsDownstreamDir)
                        {
                            tf.MinZ = info.TopZ;
                            tf.MaxZ = info.TopZ + m_DeltaZ;
                        }
                        else
                        {
                            tf.MaxZ = info.BottomZ;
                            tf.MinZ = info.BottomZ - m_DeltaZ;
                        }
                        tf.Weight = 1.0;
                        m_VF.AddTrackFit(tf);
                        return m_VF.AvgDistance <= 0.5 * m_IP;
                    }
                    catch (Exception) { return false; }
                    finally
                    {
                        if (m_VF.Count == 2)
                            m_VF.RemoveTrackFit(tf.Id);
                    }
                }
                return true;
            }
        }

        public class TrackFilterWithTrack
        {
            SySal.TotalScan.Track m_Track;

            public TrackFilterWithTrack(SySal.TotalScan.Track tk)
            {
                m_Track = tk;
            }

            public bool Filter(SySal.TotalScan.Track t)
            {
                return t == m_Track;
            }

            public bool FilterSeg(SySal.TotalScan.Segment s)
            {
                return s.TrackOwner == m_Track;
            }
        }

        public class TrackFilterWithVertex
        {
            SySal.TotalScan.Vertex m_Vertex;

            public TrackFilterWithVertex(SySal.TotalScan.Vertex vtx)
            {
                m_Vertex = vtx;
            }

            public bool Filter(SySal.TotalScan.Track t)
            {
                return t.Upstream_Vertex == m_Vertex || t.Downstream_Vertex == m_Vertex;
            }

            public bool FilterSeg(SySal.TotalScan.Segment s)
            {
                SySal.TotalScan.Track tk = s.TrackOwner;
                if (tk == null) return m_Vertex == null;
                return tk.Upstream_Vertex == m_Vertex || tk.Downstream_Vertex == m_Vertex;
            }
        }

        private void btnShowRelatedTracks_Click(object sender, EventArgs e)
        {
            object[] o;
            if (chkBroadcastAction.Checked == false) o = new object[1] { m_Track };
            else
            {
                o = new object[AvailableBrowsers.Count];
                int i;
                for (i = 0; i < o.Length; i++)
                    o[i] = ((TrackBrowser)AvailableBrowsers[i]).Track;
            }
            foreach (SySal.TotalScan.Track ot in o)
                SelectForGraph.ShowTracks(new TrackFilter(ot, radioDownstream.Checked, radioDownstreamDir.Checked, m_Opening, m_Radius, m_DeltaSlope, m_DeltaZ, m_IP).Filter, rdShow.Checked || rdBoth.Checked, rdDataset.Checked || rdBoth.Checked);
        }

        private void btnShowRelatedSegments_Click(object sender, EventArgs e)
        {
            object[] o;
            if (chkBroadcastAction.Checked == false) o = new object[1] { m_Track };
            else
            {
                o = new object[AvailableBrowsers.Count];
                int i;
                for (i = 0; i < o.Length; i++)
                    o[i] = ((TrackBrowser)AvailableBrowsers[i]).Track;
            }
            foreach (SySal.TotalScan.Track ot in o)
                SelectForGraph.ShowSegments(new TrackFilter(ot, radioDownstream.Checked, radioDownstreamDir.Checked, m_Opening, m_Radius, m_DeltaSlope, m_DeltaZ, m_IP).FilterSeg, rdShow.Checked || rdBoth.Checked, rdDataset.Checked || rdBoth.Checked);
        }

        private void btnScanbackScanforth_Click(object sender, EventArgs e)
        {
            SySal.TotalScan.Segment[] segs = SelectForGraph.Follow(m_Track[radioDownstream.Checked ? 0 : (m_Track.Length - 1)], radioDownstreamDir.Checked, 0, m_Radius, m_DeltaSlope, 0.0, m_MaxMisses);
            string text = "TRACK\tID_PLATE\tZ\tGRAINS\tFPX\tFPY\tFSX\tFSY\tOFPX\tOFPY\tOFSX\tOFSY";
            bool hasnewsegs = false;
            foreach (SySal.TotalScan.Segment s in segs)
                if (s != null)
                {
                    if (s.TrackOwner != m_Track) hasnewsegs = true;
                    SySal.Tracking.MIPEmulsionTrackInfo info = s.Info;
                    text += "\r\n" + (s.TrackOwner == null ? "-1" : s.TrackOwner.Id.ToString()) + "\t" + s.LayerOwner.SheetId + "\t" + info.Intercept.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + info.Count + "\t" + info.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + info.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) +
                        "\t" + info.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\t" + info.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture);
                    info = s.OriginalInfo;
                    text += "\t" + info.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + info.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) +
                        "\t" + info.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\t" + info.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture);
                }
            SelectForGraph.ShowSegments(segs, text);
            new QBrowser("Segments found following path", text).ShowDialog();
            if (hasnewsegs && MessageBox.Show("Extend the track using new segments?", "Reconstruction editing", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                foreach (SySal.TotalScan.Segment s in segs)
                    if (s != null && s.TrackOwner != m_Track)                    
                        if (s.TrackOwner == null) m_Track.AddSegment(s);
                        else 
                        {
                            SySal.TotalScan.Flexi.Segment ns = new SySal.TotalScan.Flexi.Segment(s, ((SySal.TotalScan.Flexi.Segment)s).DataSet);                            
                            ((SySal.TotalScan.Flexi.Layer)s.LayerOwner).Add(new SySal.TotalScan.Flexi.Segment[1] { ns });
                            m_Track.AddSegment(ns);
                        }
                Track = m_Track;
            }
        }

        private int m_MaxMisses = 3;

        private void OnMaxMissesLeave(object sender, EventArgs e)
        {
            try
            {
                m_MaxMisses = System.Convert.ToInt32(txtMaxMisses.Text);
            }
            catch (Exception)
            {
                txtMaxMisses.Text = m_MaxMisses.ToString();
                txtMaxMisses.Focus();
            }
        }

        private void HighlightCheck_CheckedChanged(object sender, EventArgs e)
        {
            SelectForIP.Highlight(m_Track, HighlightCheck.Checked);
        }

        private void EnableLabelCheck_CheckedChanged(object sender, EventArgs e)
        {
            SelectForIP.EnableLabel(m_Track, EnableLabelCheck.Checked);
        }

        private void SetLabelButton_Click(object sender, EventArgs e)
        {
            SelectForIP.SetLabel(m_Track, LabelText.Text);
        }

        private void RefreshFitList_Click(object sender, EventArgs e)
        {
            ListFits.BeginUpdate();
            ListFits.Items.Clear();
            foreach (VertexFitForm vff in VertexFitForm.AvailableBrowsers)
                ListFits.Items.Add(vff.FitName);
            ListFits.EndUpdate();
        }

        private void AddUpButton_Click(object sender, EventArgs e)
        {
            int i;
            tf.PLikelihood = m_MomentumLikelihood;
            tf.Field = 0;
            tf.Id = new SySal.TotalScan.BaseTrackIndex(m_Track.Id);
            tf.Sigma = 0;
            tf.Weight = m_Weight;
            for (i = 0; i < m_Track.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = m_Track[i].Info;
                tf.AreaSum += info.AreaSum;
                tf.Count += info.Count;
            }
            tf.MaxZ = m_Track.Upstream_Z;
            tf.MinZ = m_Track.Upstream_Z - m_ExtrapDistance;
            tf.TopZ = m_Track.Downstream_Z;
            tf.BottomZ = m_Track.Upstream_Z;
            SySal.Tracking.MIPEmulsionTrackInfo f_info = null;
            if (radioUseFit.Checked)
            {
                f_info = new SySal.Tracking.MIPEmulsionTrackInfo();
                f_info.Intercept.X = m_Track.Upstream_PosX;
                f_info.Intercept.Y = m_Track.Upstream_PosY;
                f_info.Intercept.Z = m_Track.Upstream_PosZ;
                f_info.Slope.X = m_Track.Upstream_SlopeX;
                f_info.Slope.Y = m_Track.Upstream_SlopeY;
                f_info.Slope.Z = 1.0;
            }
            else if (radioUseLastSeg.Checked) f_info = m_Track[m_Track.Length - 1].Info;
            tf.Intercept = f_info.Intercept;
            tf.Slope = f_info.Slope;
            tf.Slope.Z = 1.0;
            AddTrackFit(tf);
        }

        private SySal.TotalScan.VertexFit.TrackFitWithMomentum tf = new SySal.TotalScan.VertexFit.TrackFitWithMomentum();

        private void AddDownButton_Click(object sender, EventArgs e)
        {            
            int i;
            tf.PLikelihood = m_MomentumLikelihood;
            tf.Field = 0;
            tf.Id = new SySal.TotalScan.BaseTrackIndex(m_Track.Id);
            tf.Sigma = 0;
            tf.Weight = m_Weight;
            for (i = 0; i < m_Track.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = m_Track[i].Info;
                tf.AreaSum += info.AreaSum;
                tf.Count += info.Count;
            }
            tf.MaxZ = m_Track.Downstream_Z + m_ExtrapDistance;
            tf.MinZ = m_Track.Downstream_Z;
            tf.TopZ = m_Track.Downstream_Z;
            tf.BottomZ = m_Track.Upstream_Z;
            SySal.Tracking.MIPEmulsionTrackInfo f_info = null;
            if (radioUseFit.Checked)
            {
                f_info = new SySal.Tracking.MIPEmulsionTrackInfo();
                f_info.Intercept.X = m_Track.Downstream_PosX;
                f_info.Intercept.Y = m_Track.Downstream_PosY;
                f_info.Intercept.Z = m_Track.Downstream_PosZ;
                f_info.Slope.X = m_Track.Downstream_SlopeX;
                f_info.Slope.Y = m_Track.Downstream_SlopeY;
                f_info.Slope.Z = 1.0;
            }
            else if (radioUseLastSeg.Checked) f_info = m_Track[0].Info;
            tf.Intercept = f_info.Intercept;
            tf.Slope = f_info.Slope;
            tf.Slope.Z = 1.0;
            AddTrackFit(tf);
        }

        private void AddTrackFit(SySal.TotalScan.VertexFit.TrackFit tf)
        {
            if (ListFits.SelectedIndex < 0) return;
            string fitname = ListFits.Items[ListFits.SelectedIndex].ToString();
            int i;
            for (i = 0; i < VertexFitForm.AvailableBrowsers.Count; i++)
            {
                VertexFitForm vf = (VertexFitForm)VertexFitForm.AvailableBrowsers[i];
                if (String.Compare(vf.FitName, fitname, true) == 0)
                    vf.AddTrackFit(tf);
            }
        }

        double m_ExtrapDistance = 3900.0;

        private void OnExtrapDistLeave(object sender, EventArgs e)
        {
            try
            {
                m_ExtrapDistance = System.Convert.ToDouble(txtExtrapolationDist.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtExtrapolationDist.Text = m_ExtrapDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void WeightFromQualityButton_Click(object sender, EventArgs e)
        {
            m_Weight = SySal.TotalScan.Vertex.SlopeScatteringWeight(m_Track);
            txtWeight.Text = m_Weight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        double m_Weight = 0.1;

        private void OnWeightLeave(object sender, EventArgs e)
        {
            try
            {
                m_Weight = System.Convert.ToDouble(txtWeight.Text, System.Globalization.CultureInfo.InvariantCulture);
                if (m_Weight <= 0.0) WeightFromQualityButton_Click(this, null);
            }
            catch (Exception)
            {
                txtWeight.Text = m_Weight.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        double m_Momentum = 0.0;

        private void OnMomentumLeave(object sender, EventArgs e)
        {
            try
            {
                m_Momentum = System.Convert.ToDouble(txtMomentum.Text, System.Globalization.CultureInfo.InvariantCulture);
                m_MomentumLikelihood = new NumericalTools.OneParamLogLikelihood(m_Momentum, m_Momentum, new double[1] { 1.0 }, "P");
                if (m_Momentum < 0.0) { m_Momentum = 0.0; throw new Exception(); }
            }
            catch (Exception)
            {
                txtMomentum.Text = m_Momentum.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void MomentumFromAttribute_Click(object sender, EventArgs e)
        {
            try
            {
                m_Momentum = m_Track.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("P"));
                txtMomentum.Text = m_Momentum.ToString(System.Globalization.CultureInfo.InvariantCulture);
                m_MomentumLikelihood = new NumericalTools.OneParamLogLikelihood(m_Momentum, m_Momentum, new double[1] { 1.0 }, "P");
            }
            catch (Exception)
            {
                MessageBox.Show("No valid momentum attribute found.", "Data error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void ExportButton_Click(object sender, EventArgs e)
        {
            if (MomentumFitterList.SelectedIndex < 0) return;
            string sel = MomentumFitterList.SelectedItem.ToString();
            MomentumFitForm mf = null;
            if (String.Compare(sel, NewFitterString, true) == 0) mf = new MomentumFitForm("Tk " + m_Track.Id);
            else
            {
                foreach (MomentumFitForm mfx in MomentumFitForm.AvailableBrowsers)
                    if (String.Compare(mfx.FitName, sel, true) == 0)
                    {
                        mf = mfx;
                        break;
                    }
                if (mf == null)
                {
                    MessageBox.Show("Internal inconsistency found!", "Debug Info", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            int i;
            SySal.Tracking.MIPEmulsionTrackInfo[] tks = new SySal.Tracking.MIPEmulsionTrackInfo[SlopeList.CheckedItems.Count];
            for (i = 0; i < SlopeList.CheckedItems.Count; i++)
                tks[i] = (SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.CheckedItems[i].Tag;
            mf.Add(new MomentumFitForm.FitSet(tks, this));
            mf.Show();
        }

        private void buttonMeasSelAll_Click(object sender, EventArgs e)
        {
            SlopeList.BeginUpdate();
            foreach (ListViewItem lvi in SlopeList.Items)
                lvi.Checked = true;
            SlopeList.EndUpdate();
        }

        private void buttonMeasIgnoreGrains_Click(object sender, EventArgs e)
        {
            SlopeList.BeginUpdate();
            foreach (ListViewItem lvi in SlopeList.Items)
                if (((SySal.Tracking.MIPEmulsionTrackInfo)(lvi.Tag)).Count < m_IgnoreGrains)
                    lvi.Checked = false;
            SlopeList.EndUpdate();
        }

        private void buttonIgnoreDeltaSlope_Click(object sender, EventArgs e)
        {
            SlopeList.BeginUpdate();
            int i;
            double dsx, dsy;
            for (i = 0; i < SlopeList.Items.Count; i++)
            {
                ListViewItem lvi = SlopeList.Items[i];
                if (i == 0 && SlopeList.Items.Count >= 2)
                {
                    dsx = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[0].Tag).Slope.X - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[1].Tag).Slope.X;
                    dsy = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[0].Tag).Slope.Y - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[1].Tag).Slope.Y;
                    if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                        lvi.Checked = false;
                }
                else if (i == SlopeList.Items.Count - 1 && SlopeList.Items.Count >= 2)
                {
                    dsx = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i].Tag).Slope.X - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i - 1].Tag).Slope.X;
                    dsy = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i].Tag).Slope.Y - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i - 1].Tag).Slope.Y;
                    if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                        lvi.Checked = false;
                }
                else if (SlopeList.Items.Count >= 3)
                {
                    dsx = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i].Tag).Slope.X - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i - 1].Tag).Slope.X;
                    dsy = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i].Tag).Slope.Y - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i - 1].Tag).Slope.Y;
                    if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                    {
                        dsx = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i].Tag).Slope.X - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i + 1].Tag).Slope.X;
                        dsy = ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i].Tag).Slope.Y - ((SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.Items[i + 1].Tag).Slope.Y;
                        if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                            lvi.Checked = false;
                    }
                }
            }
            SlopeList.EndUpdate();        
        }

        private void OnIgnoreGrainsLeave(object sender, EventArgs e)
        {
            try
            {
                m_IgnoreGrains = System.Convert.ToInt32(textMeasIgnoreGrains.Text);
            }
            catch (Exception)
            {
                textMeasIgnoreGrains.Text = m_IgnoreGrains.ToString();
            }
        }

        private void OnIgnoreDeltaSlopeLeave(object sender, EventArgs e)
        {
            try
            {
                m_IgnoreDeltaSlope = System.Convert.ToDouble(textIgnoreDeltaSlope.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                textIgnoreDeltaSlope.Text = m_IgnoreDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        int m_IgnoreGrains = 18;

        double m_IgnoreDeltaSlope = 0.01;

        const string NewFitterString = "{...new fitter...}";

        private void OnUpdateMomentumFitterList(object sender, EventArgs e)
        {
            MomentumFitterList.BeginUpdate();
            MomentumFitterList.Items.Clear();
            foreach (MomentumFitForm mf in MomentumFitForm.AvailableBrowsers)
                MomentumFitterList.Items.Add(mf.FitName);
            MomentumFitterList.Items.Add(NewFitterString);
            MomentumFitterList.EndUpdate();            
        }

        private void OnMCSAlgoSelChanged(object sender, EventArgs e)
        {
            if (AlgoCombo.SelectedItem is SySal.Processing.MCSLikelihood.MomentumEstimator)
            {
                MomentumFitterList.Visible = true;
                ExportButton.Visible = true;
                MCSAnnecyComputeButton.Visible = false;
            }
            else if (AlgoCombo.SelectedItem is SySal.Processing.MCSAnnecy.MomentumEstimator)
            {
                MomentumFitterList.Visible = false;
                ExportButton.Visible = false;
                MCSAnnecyComputeButton.Visible = true;
            }
        }

        private void MCSAnnecyComputeButton_Click(object sender, EventArgs e)
        {
            try
            {
                int i;
                SySal.Tracking.MIPEmulsionTrackInfo[] tks = new SySal.Tracking.MIPEmulsionTrackInfo[SlopeList.CheckedItems.Count];
                for (i = 0; i < SlopeList.CheckedItems.Count; i++)
                    tks[i] = (SySal.Tracking.MIPEmulsionTrackInfo)SlopeList.CheckedItems[i].Tag;                                
                SySal.TotalScan.MomentumResult result = ((SySal.Processing.MCSAnnecy.MomentumEstimator)AlgoCombo.SelectedItem).ProcessData(tks);
                SetMomentum(new NumericalTools.OneParamLogLikelihood(result.Value, result.Value, new double[1] { 1.0 }, "P"), result.LowerBound, result.UpperBound, result.ConfidenceLevel);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Momentum fit error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        public void xAutoComputeMomentum()
        {
            foreach (TabPage tbp in tabControl1.TabPages)
                if (tbp.Text == "Momentum")
                {
                    tabControl1.SelectedTab = tbp;
                    break;
                }
            m_IgnoreGrains = 18;
            m_IgnoreDeltaSlope = 0.01;
            textIgnoreDeltaSlope.Text = m_IgnoreDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            textMeasIgnoreGrains.Text = m_IgnoreGrains.ToString();
            buttonMeasIgnoreGrains_Click(this, null);
            buttonIgnoreDeltaSlope_Click(this, null);
            int i;
            int excludesheet = -1;
            for (i = 0; i < m_Layers.Length; i++)
            {
                SySal.TotalScan.Flexi.Layer ly = (SySal.TotalScan.Flexi.Layer)m_Layers[i];
                if (ly.BrickId >= 1000000 && ly.BrickId < 3000000)
                {
                    excludesheet = ly.Id;
                    break;
                }
            }
            foreach (ListViewItem lvi in SlopeList.Items)
                if (Convert.ToInt32(lvi.SubItems[0].Text) == excludesheet)
                    lvi.Checked = false;
            foreach (object o in AlgoCombo.Items)
                if (o is SySal.Processing.MCSAnnecy.MomentumEstimator)
                {
                    AlgoCombo.SelectedItem = o;
                    break;
                }
            MCSAnnecyComputeButton_Click(this, null);
        }

        private void SegRemoveButton_Click(object sender, EventArgs e)
        {
            if (SegmentList.SelectedItems.Count > 0 && SegmentList.SelectedItems.Count < m_Track.Length &&
                MessageBox.Show("Are you sure you want to delete " + SegmentList.SelectedItems.Count + " segment" + ((SegmentList.SelectedItems.Count == 1) ? "?" : "s?") + "\r\nPlease note this change cannot be undone.", "Segment removal warning", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                int[] segrem = new int[SegmentList.SelectedIndices.Count];
                int i;
                for (i = 0; i < segrem.Length; i++)
                    segrem[i] = SegmentList.SelectedIndices[i];
                ((SySal.TotalScan.Flexi.Track)m_Track).RemoveSegments(segrem);
                m_Track.NotifyChanged();
                Track = m_Track;
                if (SelectForIP.DeleteWithOwner(m_Track))
                    SelectForIP.Plot(m_Track);
            }
        }

        internal void RefreshAttributeList()
        {
            AttributeList.BeginUpdate();
            AttributeList.Items.Clear();
            SySal.TotalScan.Attribute [] attrlist = m_Track.ListAttributes();
            foreach (SySal.TotalScan.Attribute attr in attrlist)
            {
                ListViewItem lvi = new ListViewItem(attr.Index.ToString());
                lvi.SubItems.Add(attr.Value.ToString(System.Globalization.CultureInfo.InvariantCulture));
                lvi.Tag = attr.Index;
                AttributeList.Items.Add(lvi);
            }
            RefreshFeedbackStatus();
            AttributeList.EndUpdate();
        }

        public static SySal.TotalScan.NamedAttributeIndex FBEventIndex = new SySal.TotalScan.NamedAttributeIndex("EVENT");
        public static SySal.TotalScan.NamedAttributeIndex FBParticleIndex = new SySal.TotalScan.NamedAttributeIndex("PARTICLE");
        public static SySal.TotalScan.NamedAttributeIndex FBDarknessIndex = new SySal.TotalScan.NamedAttributeIndex("DARKNESS");
        public static SySal.TotalScan.NamedAttributeIndex FBOutOfBrickIndex = new SySal.TotalScan.NamedAttributeIndex("OUTOFBRICK");
        public static SySal.TotalScan.NamedAttributeIndex FBLastPlateIndex = new SySal.TotalScan.NamedAttributeIndex("LASTPLATE");
        public static SySal.TotalScan.NamedAttributeIndex FBScanbackIndex = new SySal.TotalScan.NamedAttributeIndex("SCANBACK");
        public static SySal.TotalScan.NamedAttributeIndex FBManualIndex = new SySal.TotalScan.NamedAttributeIndex("MANUAL");
        public static SySal.TotalScan.NamedAttributeIndex FBPIndex = new SySal.TotalScan.NamedAttributeIndex("P");
        public static SySal.TotalScan.NamedAttributeIndex FBPMinIndex = new SySal.TotalScan.NamedAttributeIndex("PMIN");
        public static SySal.TotalScan.NamedAttributeIndex FBPMaxIndex = new SySal.TotalScan.NamedAttributeIndex("PMAX");
        public static SySal.TotalScan.NamedAttributeIndex FBDecaySearchIndex = new SySal.TotalScan.NamedAttributeIndex("DECAYSEARCH");

        bool m_IsRefreshing = false;

        internal void RefreshFeedbackStatus()
        {
            m_IsRefreshing = true;
            try
            {
                if (System.Convert.ToInt64(m_Track.GetAttribute(FBEventIndex)) >= 0) chkFBEvent.Checked = true;
            }
            catch (Exception) { chkFBEvent.Checked = false; }
            try
            {
                chkFBScanback.Checked = (System.Convert.ToInt32(m_Track.GetAttribute(FBScanbackIndex)) > 0);
            }
            catch (Exception) { chkFBScanback.Checked = false; }
            try
            {
                chkFBManual.Checked = (System.Convert.ToInt32(m_Track.GetAttribute(FBManualIndex)) > 0);
            }
            catch (Exception) { chkFBManual.Checked = false; }
            try
            {
                cmbFBParticle.SelectedItem = (FBParticleType)System.Convert.ToInt32(m_Track.GetAttribute(FBParticleIndex));
            }
            catch (Exception) { if (cmbFBParticle.Items.Count > 0) cmbFBParticle.SelectedIndex = 0; }
            try
            {
                cmbFBDarkness.SelectedItem = (FBDarkness)System.Convert.ToInt32(m_Track.GetAttribute(FBDarknessIndex) * 2.0);
            }
            catch (Exception) { if (cmbFBDarkness.Items.Count > 0) cmbFBDarkness.SelectedIndex = 0; }
            try
            {
                cmbFBOutOfBrick.SelectedItem = (FBOutOfBrick)System.Convert.ToInt32(m_Track.GetAttribute(FBOutOfBrickIndex));
            }
            catch (Exception) { if (cmbFBOutOfBrick.Items.Count > 0) cmbFBOutOfBrick.SelectedIndex = 0; }
            try
            {
                cmbFBDecaySearch.SelectedItem = (FBDecaySearch)System.Convert.ToInt32(m_Track.GetAttribute(FBDecaySearchIndex));
            }
            catch (Exception) { if (cmbFBDecaySearch.Items.Count > 0) cmbFBDecaySearch.SelectedIndex = 0; }
            try
            {
                PlateDesc pl = new PlateDesc();
                pl.Brick = ((SySal.TotalScan.Flexi.Track)m_Track).DataSet.DataId;
                pl.Plate = System.Convert.ToInt32(m_Track.GetAttribute(FBLastPlateIndex));
                cmbFBLastPlate.SelectedItem = pl;
            }
            catch (Exception) { if (cmbFBLastPlate.Items.Count > 0) cmbFBLastPlate.SelectedIndex = cmbFBLastPlate.Items.Count - 1; }
            m_IsRefreshing = false;
        }

        private void cmdAddSetAttribute_Click(object sender, EventArgs e)
        {
            try
            {
                if (txtAttrName.Text.Trim().Length == 0)
                {
                    txtAttrName.Focus();
                    return;
                }
                m_Track.SetAttribute(new SySal.TotalScan.NamedAttributeIndex(txtAttrName.Text.Trim()),
                    System.Convert.ToDouble(txtAttrValue.Text, System.Globalization.CultureInfo.InvariantCulture));
                RefreshAttributeList();
            }
            catch (Exception)
            {
                txtAttrValue.Focus();
                return;
            }
        }

        private void cmdRemoveAttributes_Click(object sender, EventArgs e)
        {
            if (AttributeList.SelectedIndices.Count > 0 && MessageBox.Show("Are you sure you want to remove " + AttributeList.SelectedIndices.Count + " attribute" +
                ((AttributeList.SelectedIndices.Count == 1) ? "?" : "s?") + "\r\nThis operation cannot be undone.", "Attribute removal", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
                foreach (ListViewItem lvi in AttributeList.SelectedItems)
                    m_Track.RemoveAttribute((SySal.TotalScan.Index)lvi.Tag);
            RefreshAttributeList();
        }

        private void OnFBEventChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBEvent.Checked) m_Track.SetAttribute(FBEventIndex, System.Convert.ToDouble(m_Event));
            else m_Track.RemoveAttribute(FBEventIndex);
            RefreshAttributeList();
        }

        private void OnFBParticleTypeChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            FBParticleType p = (FBParticleType)cmbFBParticle.SelectedItem;
            if (p == FBParticleType.Unknown) m_Track.RemoveAttribute(FBParticleIndex);
            else m_Track.SetAttribute(FBParticleIndex, (double)(int)p);
            RefreshAttributeList();
        }

        private void OnFBDarknessChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            m_Track.SetAttribute(FBDarknessIndex, (double)(int)(FBDarkness)cmbFBDarkness.SelectedItem * 0.5);
            RefreshAttributeList();
        }

        private void OnFBOutOfBrickChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            FBOutOfBrick f = (FBOutOfBrick)cmbFBOutOfBrick.SelectedItem;
            if (f == FBOutOfBrick.N) m_Track.RemoveAttribute(FBLastPlateIndex);
            else m_Track.SetAttribute(FBLastPlateIndex, (double)(((PlateDesc)cmbFBLastPlate.SelectedItem).Plate));
            m_Track.SetAttribute(FBOutOfBrickIndex, (double)(int)f);
            RefreshAttributeList();
        }

        private void OnFBLastPlateChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            OnFBOutOfBrickChanged(sender, e);
        }

        internal class TrackComparer : IComparer
        {
            internal SySal.TotalScan.Track Target;
            public TrackComparer(SySal.TotalScan.Track t) { Target = t; }

            #region IComparer Members

            public int Compare(object x, object y)
            {
                SySal.TotalScan.Track xt = (SySal.TotalScan.Track)x;
                SySal.TotalScan.Track yt = (SySal.TotalScan.Track)y;
                double dx = xt.Upstream_SlopeX - Target.Upstream_SlopeX;
                double dy = xt.Upstream_SlopeY - Target.Upstream_SlopeY;
                double xd = dx * dx + dy * dy;
                dx = xt.Downstream_SlopeX - Target.Downstream_SlopeX;
                dy = xt.Downstream_SlopeY - Target.Downstream_SlopeY;
                xd = Math.Min(xd, dx * dx + dy * dy);

                dx = yt.Upstream_SlopeX - Target.Upstream_SlopeX;
                dy = yt.Upstream_SlopeY - Target.Upstream_SlopeY;
                double yd = dx * dx + dy * dy;
                dx = yt.Downstream_SlopeX - Target.Downstream_SlopeX;
                dy = yt.Downstream_SlopeY - Target.Downstream_SlopeY;
                yd = Math.Min(yd, dx * dx + dy * dy);
                if (xd < yd) return -1;
                if (xd > yd) return 1;
                return 0;
            }

            #endregion
        }

        private void btnFBFindSBSFTrack_Click(object sender, EventArgs e)
        {
            System.Collections.ArrayList ar = new ArrayList();
            int i;
            for (i = 0; i < m_V.Tracks.Length; i++)
                if (m_Track != m_V.Tracks[i] && String.Compare(((SySal.TotalScan.Flexi.Track)m_V.Tracks[i]).DataSet.DataType, "SBSF", true) == 0)
                {
                    SySal.TotalScan.Track t = m_V.Tracks[i];
                    double dx = m_Track.Upstream_SlopeX - t.Upstream_SlopeX;
                    double dy = m_Track.Upstream_SlopeY - t.Upstream_SlopeY;
                    if (dx * dx + dy * dy <= m_FBSlopeTol * m_FBSlopeTol)
                        ar.Add(t);
                    else
                    {
                        dx = m_Track.Downstream_SlopeX - t.Downstream_SlopeX;
                        dy = m_Track.Downstream_SlopeY - t.Downstream_SlopeY;
                        if (dx * dx + dy * dy <= m_FBSlopeTol * m_FBSlopeTol)
                            ar.Add(t);
                    }
                }
            ar.Sort(new TrackComparer(m_Track));
            cmbFBTkImportList.Items.Clear();
            cmbFBTkImportList.SelectedText = "";
            foreach (SySal.TotalScan.Track tk in ar)
                cmbFBTkImportList.Items.Add(tk.Id);
            if (cmbFBTkImportList.Items.Count > 0)
                cmbFBTkImportList.Text = cmbFBTkImportList.Items[0].ToString();
        }

        double m_FBSlopeTol = 0.05;

        private void OnFBSlopeTolLeave(object sender, EventArgs e)
        {
            try
            {
                m_FBSlopeTol = System.Convert.ToDouble(txtFBSlopeTol, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtFBSlopeTol.Text = m_FBSlopeTol.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtFBSlopeTol.Focus();
            }
        }

        private void btnFBImportAttributes_Click(object sender, EventArgs e)
        {
            try
            {
                SySal.TotalScan.Track t = m_V.Tracks[System.Convert.ToInt32(cmbFBTkImportList.Text)];
                SySal.TotalScan.Attribute[] attrlist = t.ListAttributes();
                foreach (SySal.TotalScan.Attribute a in attrlist)
                    m_Track.SetAttribute(a.Index, a.Value);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Attribute import error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            RefreshAttributeList();
            RefreshFeedbackStatus();
        }

        private void OnFBScanbackChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBScanback.Checked) m_Track.SetAttribute(FBScanbackIndex, 1.0);
            else m_Track.RemoveAttribute(FBScanbackIndex);
            RefreshAttributeList();
        }

        private void OnFBManualChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBManual.Checked) m_Track.SetAttribute(FBManualIndex, 1.0);
            else m_Track.RemoveAttribute(FBManualIndex);
            RefreshAttributeList();
        }

        SySal.TotalScan.Track m_LastHighlighted = null;

        private void OnFBTkImportListTextChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            try
            {
                int i = System.Convert.ToInt32(cmbFBTkImportList.Text);
                if (m_LastHighlighted != null) SelectForIP.Highlight(m_LastHighlighted, false);
                //SelectForGraph.ShowTracks(new TrackFilterWithTrack(m_V.Tracks[i]));
                SelectForIP.Highlight(m_LastHighlighted = m_V.Tracks[i], true);
            }
            catch (Exception) { }
        }

        internal class SingleProngVertex : SySal.TotalScan.Flexi.Vertex
        {
            public SingleProngVertex(SySal.BasicTypes.Vector w, double avgd, SySal.TotalScan.Flexi.DataSet ds, int id, SySal.TotalScan.Track tk) : base(ds, id)
            {
                m_X = w.X;
                m_Y = w.Y;
                m_Z = w.Z;
                m_DX = m_DY = 0.0;
                m_AverageDistance = 0.0;
                m_VertexCoordinatesUpdated = true;
                Tracks = new SySal.TotalScan.Track[1] { tk };
            }            
        }

        private void btnFBSingleProngVertex_Click(object sender, EventArgs e)
        {
            SySal.BasicTypes.Vector w = new SySal.BasicTypes.Vector();
            if (m_SingleProngVertexZ < 0)
            {
                w.X = m_Track.Upstream_PosX + (m_Track.Upstream_Z - m_Track.Upstream_PosZ + m_SingleProngVertexZ) * m_Track.Upstream_SlopeX;
                w.Y = m_Track.Upstream_PosY + (m_Track.Upstream_Z - m_Track.Upstream_PosZ + m_SingleProngVertexZ) * m_Track.Upstream_SlopeY;
                w.Z = m_Track.Upstream_Z + m_SingleProngVertexZ;                
            }
            else
            {
                w.X = m_Track.Downstream_PosX + (m_Track.Downstream_Z - m_Track.Downstream_PosZ + m_SingleProngVertexZ) * m_Track.Downstream_SlopeX;
                w.Y = m_Track.Downstream_PosY + (m_Track.Downstream_Z - m_Track.Downstream_PosZ + m_SingleProngVertexZ) * m_Track.Downstream_SlopeY;
                w.Z = m_Track.Downstream_Z + m_SingleProngVertexZ;
            }
            SySal.TotalScan.Flexi.Vertex newvtx = new SingleProngVertex(w, 0.0, ((SySal.TotalScan.Flexi.Track)m_Track).DataSet, m_V.Vertices.Length, m_Track);
            if (m_SingleProngVertexZ < 0) m_Track.SetUpstreamVertex(newvtx);
            else m_Track.SetDownstreamVertex(newvtx);
            ((SySal.TotalScan.Flexi.Volume.VertexList)m_V.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { newvtx });
            try
            {
                double ev = m_Track.GetAttribute(FBEventIndex);
                if (ev >= 0.0) newvtx.SetAttribute(FBEventIndex, ev);
            }
            catch (Exception) { }
            VertexBrowser.Browse(newvtx, m_Layers, SelectForIP, SelectForGraph, m_Event, m_V);
            Track = m_Track;
        }

        private double m_SingleProngVertexZ = -500.0;

        private void OnFBSingleProngVertexZChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            try
            {
                double z = System.Convert.ToDouble(txtFBSingleProngVertexZ.Text, System.Globalization.CultureInfo.InvariantCulture);
                if (z < 0 && m_Track.Upstream_Vertex != null) throw new Exception();
                else if (z > 0 && m_Track.Downstream_Vertex != null) throw new Exception();
                m_SingleProngVertexZ = z;
            }
            catch (Exception)
            {
                txtFBSingleProngVertexZ.Text = m_SingleProngVertexZ.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void AppendToSelButton_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select ASCII file to dump file for manual checks";
            sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            sdlg.OverwritePrompt = false;
            if (sdlg.ShowDialog() == DialogResult.OK)
                AppendToFileText.Text = sdlg.FileName;            
        }

        private void EnsureInFeedback()
        {
            try
            {
                if (m_Track.GetAttribute(FBEventIndex) > 0.0) return;
            }
            catch (Exception) { }
            if (MessageBox.Show("Since this track is not included in feedback, it will not be eligible for automatic matching to manual checks.\r\nDo you want to add it to feedback?", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button1) == DialogResult.Yes)
            {
                m_Track.SetAttribute(FBEventIndex, (double)m_Event);
            }
        }

        private void AppendToButton_Click(object sender, EventArgs e)
        {
            if (LayerList.SelectedIndex >= 0)
            {
                EnsureInFeedback();
                AppendLayer(LayerList.SelectedIndex);
            }
        }

        private void AppendLayer(int layerid)
        {            
            try
            {
                if (System.IO.File.Exists(AppendToFileText.Text) == false)
                    System.IO.File.WriteAllText(AppendToFileText.Text, "IDTRACK\tPOSX\tPOSY\tSLOPEX\tSLOPEY\tPLATE");
                SySal.TotalScan.Layer l = m_Layers[layerid];
                LayerList.SelectedIndex = layerid;
                System.IO.File.AppendAllText(AppendToFileText.Text, ("\r\n" + ((layerid >= m_Track[0].LayerOwner.Id && layerid <= m_Track[m_Track.Length - 1].LayerOwner.Id) ? m_Track.Id : -1) + "\t" + OrigInfoText.Text + "\t" + l.SheetId).Replace(" ", "\t"));
                MessageBox.Show("Data saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                DisplayForm.DefaultProjFileName = AppendToFileText.Text;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        internal enum ImportableAttribute
        {
            UpstreamIP = 1, DownstreamIP = 2
        }

        private void OnSelAttribChanged(object sender, EventArgs e)
        {
            try
            {
                double v = 0.0;
                SySal.TotalScan.Track tk = m_V.Tracks[System.Convert.ToInt32(txtAttribImportTk.Text)];
                switch ((ImportableAttribute)cmbAttribImport.SelectedItem)
                {
                    case ImportableAttribute.UpstreamIP: v = (tk.Upstream_Vertex == null) ? -1.0 : tk.Upstream_Impact_Parameter; break;
                    case ImportableAttribute.DownstreamIP: v = (tk.Downstream_Vertex == null) ? -1.0 : tk.Downstream_Impact_Parameter; break;
                    default: throw new Exception();
                }
                txtAttribImportValue.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {                
                txtAttribImportValue.Text = "";
                if (sender == cmbAttribImport) txtAttribImportTk.Focus();                
            }
        }

        private void cmdImportAttribute_Click(object sender, EventArgs e)
        {
            try
            {
                string name = "";
                double v = 0.0;
                SySal.TotalScan.Track tk = m_V.Tracks[System.Convert.ToInt32(txtAttribImportTk.Text)];
                switch ((ImportableAttribute)cmbAttribImport.SelectedItem)
                {
                    case ImportableAttribute.UpstreamIP: v = (tk.Upstream_Vertex == null) ? -1.0 : tk.Upstream_Impact_Parameter; name = "UpIP"; break;
                    case ImportableAttribute.DownstreamIP: v = (tk.Downstream_Vertex == null) ? -1.0 : tk.Downstream_Impact_Parameter; name = "DownIP"; break;
                    default: throw new Exception();
                }
                txtAttribImportValue.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                m_Track.SetAttribute(new SySal.TotalScan.NamedAttributeIndex(name), v);
                RefreshAttributeList();
            }
            catch (Exception)
            {
                txtAttribImportValue.Text = "";
                txtAttribImportTk.Focus();
            }
        }

        internal void SelectSegment(object owner)
        {
            if (owner is SySal.TotalScan.Flexi.Segment)
            {
                SySal.TotalScan.Flexi.Segment s = (SySal.TotalScan.Flexi.Segment)owner;
                if (s.TrackOwner != null)
                {
                    SySal.TotalScan.Flexi.Layer l = (SySal.TotalScan.Flexi.Layer)s.LayerOwner;
                    s = new SySal.TotalScan.Flexi.Segment(s, s.DataSet);
                    l.Add(new SySal.TotalScan.Flexi.Segment[1] { s });
                }                
                int i;
                for (i = 0; i < m_Track.Length; i++)
                    if (m_Track[i].LayerOwner == s.LayerOwner)
                    {
                        m_Track.RemoveSegment(s.LayerOwner.Id);
                        break;
                    }
                m_Track.AddSegment(s);
                Track = m_Track;
            }
        }

        private void SegAddReplaceButton_Click(object sender, EventArgs e)
        {
            string t = SegAddReplaceButton.Text;
            if (t.StartsWith("Start"))
            {
                SelectForGraph.ToggleAddReplaceSegments(new GDI3D.Control.SelectObject(SelectSegment));
                SegAddReplaceButton.Text = t.Replace("Start", "Stop");
            }
            else
            {
                SelectForGraph.ToggleAddReplaceSegments(null);
                SegAddReplaceButton.Text = t.Replace("Stop", "Start");
            }
        }

        private void OnDecaySearchChanged(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            m_Track.SetAttribute(FBDecaySearchIndex, (double)(int)(FBDecaySearch)cmbFBDecaySearch.SelectedItem);
            RefreshAttributeList();
        }

        private void btnBrowseTrack_Click(object sender, EventArgs e)
        {
            if (cmbMatchOtherDS.SelectedItem != null) 
                TrackBrowser.Browse(m_V.Tracks[Convert.ToInt32(cmbMatchOtherDS.SelectedItem.ToString().Split(' ')[0])], m_Layers, this.SelectForIP, SelectForGraph, m_Event, m_V);
        }

        private void btnMatchInOtherDatasets_Click(object sender, EventArgs e)
        {
            cmbMatchOtherDS.Items.Clear();
            SySal.TotalScan.Flexi.Track[] thetks = xMatchInOtherDataSets(m_V, (SySal.TotalScan.Flexi.Track)m_Track, m_DeltaSlope, m_Radius);
            foreach (SySal.TotalScan.Flexi.Track tk in thetks)
                cmbMatchOtherDS.Items.Add(tk.Id + " (" + tk.DataSet.ToString() + ")");
            if (cmbMatchOtherDS.Items.Count > 0) cmbMatchOtherDS.SelectedItem = cmbMatchOtherDS.Items[0];
        }

        static internal SySal.TotalScan.Flexi.Track[] xMatchInOtherDataSets(SySal.TotalScan.Volume v, SySal.TotalScan.Flexi.Track thetk, double deltaslope, double radius)
        {
            int i, n, j;
            double dx, dy;
            double ds2 = deltaslope * deltaslope;
            double dp2 = radius * radius;
            SySal.TotalScan.Flexi.DataSet theds = ((SySal.TotalScan.Flexi.Track)thetk).DataSet;
            ArrayList ar = new ArrayList();
            for (i = thetk[thetk.Length - 1].LayerOwner.Id; i >= thetk[0].LayerOwner.Id; i--)
            {
                SySal.TotalScan.Layer ly = v.Layers[i];
                SySal.Tracking.MIPEmulsionTrackInfo info = thetk.Fit(i, ly.RefCenter.Z);
                n = ly.Length;
                for (j = 0; j < n; j++)
                {
                    SySal.TotalScan.Flexi.Segment s = (SySal.TotalScan.Flexi.Segment)ly[j];
                    if (s.TrackOwner == null) continue;
                    SySal.TotalScan.Flexi.Track tk = (SySal.TotalScan.Flexi.Track)s.TrackOwner;
                    if (SySal.TotalScan.Flexi.DataSet.AreEqual(theds, tk.DataSet)) continue;
                    SySal.Tracking.MIPEmulsionTrackInfo jnfo = s.Info;
                    if (deltaslope >= 0.0)
                    {
                        dx = info.Slope.X - jnfo.Slope.X;
                        if (Math.Abs(dx) > deltaslope) continue;
                        dy = info.Slope.Y - jnfo.Slope.Y;
                        if (Math.Abs(dy) > deltaslope) continue;
                        if (dx * dx + dy * dy > ds2) continue;
                    }
                    if (radius >= 0.0)
                    {
                        dx = info.Intercept.X - jnfo.Intercept.X;
                        if (Math.Abs(dx) > radius) continue;
                        dy = info.Intercept.Y - jnfo.Intercept.Y;
                        if (Math.Abs(dy) > radius) continue;
                        if (dx * dx + dy * dy > dp2) continue;
                    }
                    if (ar.Contains(tk) == false) ar.Add(tk);
                }
            }
            return (SySal.TotalScan.Flexi.Track[])ar.ToArray(typeof(SySal.TotalScan.Flexi.Track));
        }

        private void PlotButton_Click(object sender, EventArgs e)
        {
            SelectForGraph.ShowTracks(new TrackFilterWithTrack(m_Track).Filter, true, false);
        }

        uint m_DownSegs = 1;

        private void OnDowsSegsLeave(object sender, EventArgs e)
        {
            try
            {
                m_DownSegs = Convert.ToUInt32(DownSegsText.Text);
            }
            catch (Exception)
            {
                DownSegsText.Text = m_DownSegs.ToString();
            }
        }

        uint m_UpSegs = 1;

        private void OnUpSegsLeave(object sender, EventArgs e)
        {
            try
            {
                m_UpSegs = Convert.ToUInt32(UpSegsText.Text);
            }
            catch (Exception)
            {
                UpSegsText.Text = m_UpSegs.ToString();
            }
        }

        uint m_DownStops = 5;

        private void OnDownStopsLeave(object sender, EventArgs e)
        {
            try
            {
                m_DownStops = Convert.ToUInt32(DownStopsText.Text);
            }
            catch (Exception)
            {
                DownStopsText.Text = m_DownStops.ToString();
            }
        }

        uint m_UpStops = 5;

        private void OnUpStopsLeave(object sender, EventArgs e)
        {
            try
            {
                m_UpStops = Convert.ToUInt32(UpStopsText.Text);
            }
            catch (Exception)
            {
                UpStopsText.Text = m_UpStops.ToString();
            }
        }

        private void AppendDownButton_Click(object sender, EventArgs e)
        {
            EnsureInFeedback();
            int layerstart = (int)(m_Track[0].LayerOwner.Id - m_DownStops);
            int layerstop = (int)(m_Track[0].LayerOwner.Id - 1 + m_DownSegs);
            int i;
            for (i = layerstart; i <= layerstop; i++) AppendLayer(i);
        }

        private void AppendUpButton_Click(object sender, EventArgs e)
        {
            EnsureInFeedback();
            int layerstart = (int)(m_Track[m_Track.Length - 1].LayerOwner.Id + 1 - m_UpSegs);
            int layerstop = (int)(m_Track[m_Track.Length - 1].LayerOwner.Id + m_UpStops);
            int i;
            for (i = layerstart; i <= layerstop; i++) AppendLayer(i);
        }

        private void SplitTrackButton_Click(object sender, EventArgs e)
        {
            if (SegmentList.SelectedIndices.Count != 2 || (SegmentList.SelectedIndices[1] != SegmentList.SelectedIndices[0] + 1))
            {
                MessageBox.Show("Please select the pair of consecutive segments where you want to split the track.", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            SySal.TotalScan.Flexi.Track ntk = new SySal.TotalScan.Flexi.Track(((SySal.TotalScan.Flexi.Track)m_Track).DataSet, m_V.Tracks.Length);
            int i = SegmentList.SelectedIndices[1];
            while (i < m_Track.Length)
                ntk.AddSegment(m_Track[i]);            
            ((SySal.TotalScan.Flexi.Volume.TrackList)m_V.Tracks).Insert(new SySal.TotalScan.Flexi.Track[1] { ntk });
            if (m_Track.Upstream_Vertex != null)
            {
                m_Track.Upstream_Vertex.AddTrack(ntk, true);
                ntk.SetUpstreamVertex(m_Track.Upstream_Vertex);
                m_Track.Upstream_Vertex.RemoveTrack(m_Track);
                m_Track.SetUpstreamVertex(null);
                
            }
            if (chkSplitWithVertex.Checked)
            {
                SySal.TotalScan.Flexi.Vertex nvtx = new SySal.TotalScan.Flexi.Vertex(((SySal.TotalScan.Flexi.Track)m_Track).DataSet, m_V.Vertices.Length);
                nvtx.AddTrack(m_Track, true);
                m_Track.SetUpstreamVertex(nvtx);
                nvtx.AddTrack(ntk, false);
                ntk.SetDownstreamVertex(nvtx);  
                ((SySal.TotalScan.Flexi.Volume.VertexList)m_V.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { nvtx });                             
            }
            Track = m_Track;
            Browse(ntk, m_Layers, SelectForIP, SelectForGraph, m_Event, m_V);
            SelectForGraph.Add(ntk);
            MessageBox.Show("Track split. Please regenerate the plot to see the changes.", "Success", MessageBoxButtons.OK);
        }

        private void CheckMissingBasetracksButton_Click(object sender, EventArgs e)
        {
            EnsureInFeedback();
            SySal.Tracking.MIPEmulsionTrackInfo[] btks = ((SySal.TotalScan.Flexi.Track)m_Track).BaseTracks;
            int layerid;
            for (layerid = m_Track[0].LayerOwner.Id; layerid <= m_Track[m_Track.Length - 1].LayerOwner.Id; layerid++)
            {                
                SySal.Tracking.MIPEmulsionTrackInfo f = null;
                foreach (SySal.Tracking.MIPEmulsionTrackInfo b in btks)
                    if (layerid == (int)b.Field)
                    {
                        f = b;
                        break;
                    }
                if (f == null)
                    AppendLayer(layerid);
            }
        }

        private void CheckAllLayers_Click(object sender, EventArgs e)
        {
            EnsureInFeedback();
            int layerid;
            for (layerid = m_Track[0].LayerOwner.Id; layerid <= m_Track[m_Track.Length - 1].LayerOwner.Id; layerid++)
                AppendLayer(layerid);            
        }

        private void cmdSpecialAttributesReference_Click(object sender, EventArgs e)
        {
            new QBrowser("EVENT\t\t-> Track will be included in feedback\r\nVTXFITWEIGHT\t-> Track weighting factor to be used for vertex fitting.\r\nP\t\t-> Best momentum estimate.\r\nPMIN\t\t-> Lower bound of 90% CL for momentum.\r\nPMAX\t\t-> Upper bound of 90% CL for momentum.\r\n" + 
                "PARTICLE\t-> Particle type.\r\nDARKNESS\t-> Darkness of the track.\r\nOUTOFBRICK\t-> Track exits the brick.\r\nLASTPLATE\t-> Last plate where the track is found before exiting the brick.\r\n" +
                "SCANBACK\t-> Track found during scanback.\r\nMANUAL\t-> Track manually recovered (seen only during operator check).\r\nDECAYSEARCH\t-> Result of the Decay Search.",
                "Reference of special attributes").ShowDialog();
        }

        private void LaunchScanButton_Click(object sender, EventArgs e)
        {
            if (LayerList.SelectedItems.Count == 1)
            {
                EnqueueOpForm eqof = new EnqueueOpForm();
                SySal.TotalScan.Layer l = m_Layers[LayerList.SelectedIndex];
                eqof.BrickId = l.BrickId;
                eqof.VolumeStartsFromTrack = true;
                eqof.VolStart.Id = m_Track.Id;
                eqof.VolStart.Plate = l.SheetId;
                SySal.Tracking.MIPEmulsionTrackInfo info = m_Track.Fit(l.Id, l.RefCenter.Z);
                info.Intercept = l.ToOriginalPoint(info.Intercept);
                info.Slope = l.ToOriginalSlope(info.Slope);
                eqof.VolStart.Position.X = info.Intercept.X;
                eqof.VolStart.Position.Y = info.Intercept.Y;
                eqof.VolStart.Slope.X = info.Slope.X;
                eqof.VolStart.Slope.Y = info.Slope.Y;
                try
                {
                    eqof.ShowDialog();
                }
                catch (Exception) { }
            }
        }
    }
}
