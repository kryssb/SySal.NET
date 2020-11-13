using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
	/// <summary>
	/// VertexBrowser - shows information about a vertex.
	/// </summary>
	/// <remarks>
	/// <para>The upper group shows the vertex parameters, which can be dumped to an ASCII file.</para>
	/// <para>The lower group shows the tracks, and they can be dumped to an ASCII file.</para>
	/// <para>A comment can be set to a vertex.</para>
	/// <para>One can navigate from a vertex to any of its tracks.</para>
	/// </remarks>
	public class VertexBrowser : System.Windows.Forms.Form
	{
		private System.Windows.Forms.ListView GeneralList;
		private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
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
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.Button GeneralSelButton;
		private System.Windows.Forms.TextBox GeneralDumpFileText;
		private System.Windows.Forms.Button GeneralDumpFileButton;
		private System.Windows.Forms.Button TrackDumpButton;
		private System.Windows.Forms.TextBox TrackDumpFileText;
		private System.Windows.Forms.Button TrackDumpSelButton;
		private System.Windows.Forms.Button GoToSelTrackButton;
		private System.Windows.Forms.ListView TrackList;
		private System.Windows.Forms.ColumnHeader columnHeader14;
		private System.Windows.Forms.ColumnHeader columnHeader15;
		private System.Windows.Forms.ColumnHeader columnHeader16;
		private System.Windows.Forms.ColumnHeader columnHeader17;
		private System.Windows.Forms.ColumnHeader columnHeader18;
		private System.Windows.Forms.ColumnHeader columnHeader19;

		private SySal.TotalScan.Vertex m_Vertex = null;
        private Button SelectIPButton;

        private SySal.Executables.EasyReconstruct.IPSelector SelectForIP;
        private GroupBox groupBox5;
        private RadioButton radioUpstreamDir;
        private RadioButton radioDownstreamDir;
        private TextBox txtDeltaZ;
        private Label label7;
        private TextBox txtDeltaSlope;
        private Label label6;
        private TextBox txtRadius;
        private Label label5;
        private TextBox txtOpening;
        private Label label4;
        private Button btnShowRelatedTracks;
        private SySal.Executables.EasyReconstruct.TrackSelector SelectForGraph;
        private Button btnShowRelatedSegments;
        private CheckBox EnableLabelCheck;
        private CheckBox HighlightCheck;
        private Button SetLabelButton;
        private TextBox LabelText;
        private TabControl tabControl1;
        private TabPage tabPage1;
        private TabPage tabPage2;
        private TabPage tabPage3;
        private TabPage tabPage4;
        private ListBox ListFits;
        private Button AddTracksButton;
        private Label label9;
        private TextBox txtExtrapolationDist;
        private RadioButton radioUseLastSeg;
        private RadioButton radioUseFit;
        private TextBox txtDataSet;
        private Label label12;
        private TabPage tabPage5;
        private TextBox txtAttrValue;
        private TextBox txtAttrName;
        private Button cmdAddSetAttribute;
        private Button cmdRemoveAttributes;
        private ListView AttributeList;
        private ColumnHeader columnHeader4;
        private ColumnHeader columnHeader13;
        private ColumnHeader columnHeader25;
        private ColumnHeader columnHeader34;
        private Button btnDownDecays;
        private Button btnUpTracks;
        private Button btnDownTracks;
        private Button btnUpDecays;
        private TabPage tabPage6;
        private CheckBox chkFBTauDecay;
        private CheckBox chkFBCharm;
        private CheckBox chkFBPrimary;
        private CheckBox chkFBEvent;
        private CheckBox chkFBDeadMaterial;

		public SySal.TotalScan.Vertex Vertex
		{
			get { return m_Vertex; }
			set 
			{
				m_Vertex = value;
				this.Text = "VertexBrowser #" + m_Vertex.Id;
                txtDataSet.Text = ((SySal.TotalScan.Flexi.Vertex)m_Vertex).DataSet.ToString();
				CommentText.Text = (m_Vertex.Comment == null) ? "" : m_Vertex.Comment;
				GeneralList.Items.Clear();
				GeneralList.Items.Add("ID").SubItems.Add(m_Vertex.Id.ToString());
				GeneralList.Items.Add("Tracks").SubItems.Add(m_Vertex.Length.ToString());
				GeneralList.Items.Add("Comment").SubItems.Add((m_Vertex.Comment == null) ? "" : m_Vertex.Comment);
				int i, c;
				for (i = c = 0; i < m_Vertex.Length; i++)
					if (m_Vertex[i].Upstream_Vertex == m_Vertex) c++;
				GeneralList.Items.Add("Downstream Tracks").SubItems.Add(c.ToString());
				for (i = c = 0; i < m_Vertex.Length; i++)
					if (m_Vertex[i].Downstream_Vertex == m_Vertex) c++;
				GeneralList.Items.Add("Upstream Tracks").SubItems.Add(c.ToString());
                try
                {
                    GeneralList.Items.Add("Average Distance").SubItems.Add(m_Vertex.AverageDistance.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    GeneralList.Items.Add("X").SubItems.Add(m_Vertex.X.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    GeneralList.Items.Add("Y").SubItems.Add(m_Vertex.Y.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    GeneralList.Items.Add("Z").SubItems.Add(m_Vertex.Z.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                }
                catch (Exception)
                {
                    GeneralList.Items.Add("Average Distance").SubItems.Add("N/A");
                    GeneralList.Items.Add("X").SubItems.Add("N/A");
                    GeneralList.Items.Add("Y").SubItems.Add("N/A");
                    GeneralList.Items.Add("Z").SubItems.Add("N/A");
                }
                foreach (SySal.TotalScan.Attribute attr in m_Vertex.ListAttributes())
                {
                    GeneralList.Items.Add("@ " + attr.Index.ToString()).SubItems.Add(attr.Value.ToString());
                }
				TrackList.Items.Clear();
				for (i = 0; i < m_Vertex.Length; i++)
				{
					SySal.TotalScan.Track tk = m_Vertex[i];
					System.Windows.Forms.ListViewItem lvi = TrackList.Items.Add(tk.Id.ToString());
					lvi.SubItems.Add((tk.Upstream_Vertex == null) ? "" : tk.Upstream_Vertex.Id.ToString());
                    try
                    {
                        lvi.SubItems.Add((tk.Upstream_Vertex == null) ? "" : tk.Upstream_Impact_Parameter.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    catch (Exception) { lvi.SubItems.Add("N/A"); }
					lvi.SubItems.Add((tk.Downstream_Vertex == null) ? "" : tk.Downstream_Vertex.Id.ToString());
                    try
                    {
                        lvi.SubItems.Add((tk.Downstream_Vertex == null) ? "" : tk.Downstream_Impact_Parameter.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    catch (Exception) { lvi.SubItems.Add("N/A"); }
					lvi.SubItems.Add(tk.Length.ToString());
					lvi.Tag = tk;
				}
                RefreshAttributeList();
			}
		}

        SySal.TotalScan.Volume.LayerList m_Layers;

        long m_Event;
        private Button btnUpPartners;
        private Button btnDownPartners;
        private CheckBox chkPartner;
        private Button KillButton;
        private TabPage tabPage7;
        private Button AppendToSelButton;
        private TextBox AppendToFileText;
        private Button AppendToButton;
        private TextBox DownSegsText;
        private Label label1;
        private CheckBox DownTracksCheck;
        private Label label2;
        private CheckBox UpTracksCheck;
        private TextBox UpSegsText;
        private Label label3;
        private TextBox UpStopsText;
        private Label label8;
        private TextBox DownStopsText;
        private CheckBox ConnToVtxCheck;
        private TextBox ConnToVtxText;
        private Button LaunchScanButton;
        private RadioButton rdBoth;
        private RadioButton rdDataset;
        private RadioButton rdShow;

        SySal.TotalScan.Volume m_V;

		protected VertexBrowser(SySal.TotalScan.Volume.LayerList ll, long ev, SySal.TotalScan.Volume v)
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
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader12 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.GeneralDumpFileButton = new System.Windows.Forms.Button();
            this.GeneralDumpFileText = new System.Windows.Forms.TextBox();
            this.GeneralSelButton = new System.Windows.Forms.Button();
            this.TrackList = new System.Windows.Forms.ListView();
            this.columnHeader14 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader16 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader17 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader18 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader19 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader15 = new System.Windows.Forms.ColumnHeader();
            this.GoToSelTrackButton = new System.Windows.Forms.Button();
            this.TrackDumpButton = new System.Windows.Forms.Button();
            this.SetCommentButton = new System.Windows.Forms.Button();
            this.CommentText = new System.Windows.Forms.TextBox();
            this.TrackDumpFileText = new System.Windows.Forms.TextBox();
            this.TrackDumpSelButton = new System.Windows.Forms.Button();
            this.SelectIPButton = new System.Windows.Forms.Button();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.chkPartner = new System.Windows.Forms.CheckBox();
            this.radioUpstreamDir = new System.Windows.Forms.RadioButton();
            this.radioDownstreamDir = new System.Windows.Forms.RadioButton();
            this.txtDeltaZ = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.txtDeltaSlope = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtRadius = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtOpening = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.btnShowRelatedTracks = new System.Windows.Forms.Button();
            this.btnShowRelatedSegments = new System.Windows.Forms.Button();
            this.EnableLabelCheck = new System.Windows.Forms.CheckBox();
            this.HighlightCheck = new System.Windows.Forms.CheckBox();
            this.SetLabelButton = new System.Windows.Forms.Button();
            this.LabelText = new System.Windows.Forms.TextBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.KillButton = new System.Windows.Forms.Button();
            this.txtDataSet = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.tabPage7 = new System.Windows.Forms.TabPage();
            this.LaunchScanButton = new System.Windows.Forms.Button();
            this.ConnToVtxText = new System.Windows.Forms.TextBox();
            this.ConnToVtxCheck = new System.Windows.Forms.CheckBox();
            this.label3 = new System.Windows.Forms.Label();
            this.UpStopsText = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.DownStopsText = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.UpTracksCheck = new System.Windows.Forms.CheckBox();
            this.UpSegsText = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.DownTracksCheck = new System.Windows.Forms.CheckBox();
            this.DownSegsText = new System.Windows.Forms.TextBox();
            this.AppendToSelButton = new System.Windows.Forms.Button();
            this.AppendToFileText = new System.Windows.Forms.TextBox();
            this.AppendToButton = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.btnUpPartners = new System.Windows.Forms.Button();
            this.btnDownPartners = new System.Windows.Forms.Button();
            this.btnUpTracks = new System.Windows.Forms.Button();
            this.btnDownTracks = new System.Windows.Forms.Button();
            this.btnUpDecays = new System.Windows.Forms.Button();
            this.btnDownDecays = new System.Windows.Forms.Button();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.radioUseLastSeg = new System.Windows.Forms.RadioButton();
            this.radioUseFit = new System.Windows.Forms.RadioButton();
            this.label9 = new System.Windows.Forms.Label();
            this.txtExtrapolationDist = new System.Windows.Forms.TextBox();
            this.ListFits = new System.Windows.Forms.ListBox();
            this.AddTracksButton = new System.Windows.Forms.Button();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.txtAttrValue = new System.Windows.Forms.TextBox();
            this.txtAttrName = new System.Windows.Forms.TextBox();
            this.cmdAddSetAttribute = new System.Windows.Forms.Button();
            this.cmdRemoveAttributes = new System.Windows.Forms.Button();
            this.AttributeList = new System.Windows.Forms.ListView();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader13 = new System.Windows.Forms.ColumnHeader();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.chkFBDeadMaterial = new System.Windows.Forms.CheckBox();
            this.chkFBTauDecay = new System.Windows.Forms.CheckBox();
            this.chkFBCharm = new System.Windows.Forms.CheckBox();
            this.chkFBPrimary = new System.Windows.Forms.CheckBox();
            this.chkFBEvent = new System.Windows.Forms.CheckBox();
            this.columnHeader25 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader34 = new System.Windows.Forms.ColumnHeader();
            this.rdShow = new System.Windows.Forms.RadioButton();
            this.rdDataset = new System.Windows.Forms.RadioButton();
            this.rdBoth = new System.Windows.Forms.RadioButton();
            this.groupBox5.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage7.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.tabPage5.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.SuspendLayout();
            // 
            // GeneralList
            // 
            this.GeneralList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.GeneralList.FullRowSelect = true;
            this.GeneralList.GridLines = true;
            this.GeneralList.Location = new System.Drawing.Point(15, 34);
            this.GeneralList.Name = "GeneralList";
            this.GeneralList.Size = new System.Drawing.Size(396, 159);
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
            // columnHeader6
            // 
            this.columnHeader6.Text = "LayerID";
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "SheetID";
            // 
            // GeneralDumpFileButton
            // 
            this.GeneralDumpFileButton.Location = new System.Drawing.Point(364, 202);
            this.GeneralDumpFileButton.Name = "GeneralDumpFileButton";
            this.GeneralDumpFileButton.Size = new System.Drawing.Size(48, 24);
            this.GeneralDumpFileButton.TabIndex = 2;
            this.GeneralDumpFileButton.Text = "Dump";
            this.GeneralDumpFileButton.Click += new System.EventHandler(this.GeneralDumpFileButton_Click);
            // 
            // GeneralDumpFileText
            // 
            this.GeneralDumpFileText.Location = new System.Drawing.Point(56, 202);
            this.GeneralDumpFileText.Name = "GeneralDumpFileText";
            this.GeneralDumpFileText.Size = new System.Drawing.Size(302, 20);
            this.GeneralDumpFileText.TabIndex = 1;
            // 
            // GeneralSelButton
            // 
            this.GeneralSelButton.Location = new System.Drawing.Point(15, 202);
            this.GeneralSelButton.Name = "GeneralSelButton";
            this.GeneralSelButton.Size = new System.Drawing.Size(32, 24);
            this.GeneralSelButton.TabIndex = 0;
            this.GeneralSelButton.Text = "...";
            this.GeneralSelButton.Click += new System.EventHandler(this.GeneralSelButton_Click);
            // 
            // TrackList
            // 
            this.TrackList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader14,
            this.columnHeader16,
            this.columnHeader17,
            this.columnHeader18,
            this.columnHeader19,
            this.columnHeader15});
            this.TrackList.FullRowSelect = true;
            this.TrackList.GridLines = true;
            this.TrackList.Location = new System.Drawing.Point(8, 15);
            this.TrackList.Name = "TrackList";
            this.TrackList.Size = new System.Drawing.Size(407, 168);
            this.TrackList.TabIndex = 6;
            this.TrackList.UseCompatibleStateImageBehavior = false;
            this.TrackList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader14
            // 
            this.columnHeader14.Text = "ID";
            // 
            // columnHeader16
            // 
            this.columnHeader16.Text = "UpVtx";
            // 
            // columnHeader17
            // 
            this.columnHeader17.Text = "UpIP";
            // 
            // columnHeader18
            // 
            this.columnHeader18.Text = "DownVtx";
            // 
            // columnHeader19
            // 
            this.columnHeader19.Text = "DownIP";
            // 
            // columnHeader15
            // 
            this.columnHeader15.Text = "Segments";
            // 
            // GoToSelTrackButton
            // 
            this.GoToSelTrackButton.Location = new System.Drawing.Point(8, 189);
            this.GoToSelTrackButton.Name = "GoToSelTrackButton";
            this.GoToSelTrackButton.Size = new System.Drawing.Size(407, 24);
            this.GoToSelTrackButton.TabIndex = 5;
            this.GoToSelTrackButton.Text = "Go to selected track";
            this.GoToSelTrackButton.Click += new System.EventHandler(this.GoToSelTrackButton_Click);
            // 
            // TrackDumpButton
            // 
            this.TrackDumpButton.Location = new System.Drawing.Point(366, 218);
            this.TrackDumpButton.Name = "TrackDumpButton";
            this.TrackDumpButton.Size = new System.Drawing.Size(48, 24);
            this.TrackDumpButton.TabIndex = 10;
            this.TrackDumpButton.Text = "Dump";
            this.TrackDumpButton.Click += new System.EventHandler(this.TrackDumpButton_Click);
            // 
            // SetCommentButton
            // 
            this.SetCommentButton.Location = new System.Drawing.Point(15, 232);
            this.SetCommentButton.Name = "SetCommentButton";
            this.SetCommentButton.Size = new System.Drawing.Size(108, 24);
            this.SetCommentButton.TabIndex = 4;
            this.SetCommentButton.Text = "Set Comment";
            this.SetCommentButton.Click += new System.EventHandler(this.SetCommentButton_Click);
            // 
            // CommentText
            // 
            this.CommentText.Location = new System.Drawing.Point(131, 232);
            this.CommentText.Name = "CommentText";
            this.CommentText.Size = new System.Drawing.Size(280, 20);
            this.CommentText.TabIndex = 5;
            // 
            // TrackDumpFileText
            // 
            this.TrackDumpFileText.Location = new System.Drawing.Point(48, 218);
            this.TrackDumpFileText.Name = "TrackDumpFileText";
            this.TrackDumpFileText.Size = new System.Drawing.Size(312, 20);
            this.TrackDumpFileText.TabIndex = 9;
            // 
            // TrackDumpSelButton
            // 
            this.TrackDumpSelButton.Location = new System.Drawing.Point(8, 218);
            this.TrackDumpSelButton.Name = "TrackDumpSelButton";
            this.TrackDumpSelButton.Size = new System.Drawing.Size(32, 24);
            this.TrackDumpSelButton.TabIndex = 8;
            this.TrackDumpSelButton.Text = "...";
            this.TrackDumpSelButton.Click += new System.EventHandler(this.TrackDumpSelButton_Click);
            // 
            // SelectIPButton
            // 
            this.SelectIPButton.Location = new System.Drawing.Point(3, 14);
            this.SelectIPButton.Name = "SelectIPButton";
            this.SelectIPButton.Size = new System.Drawing.Size(410, 24);
            this.SelectIPButton.TabIndex = 13;
            this.SelectIPButton.Text = "Select for IP";
            this.SelectIPButton.Click += new System.EventHandler(this.SelectIPButton_Click);
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.chkPartner);
            this.groupBox5.Controls.Add(this.radioUpstreamDir);
            this.groupBox5.Controls.Add(this.radioDownstreamDir);
            this.groupBox5.Location = new System.Drawing.Point(166, 46);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(120, 87);
            this.groupBox5.TabIndex = 41;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Direction";
            // 
            // chkPartner
            // 
            this.chkPartner.AutoSize = true;
            this.chkPartner.Location = new System.Drawing.Point(6, 64);
            this.chkPartner.Name = "chkPartner";
            this.chkPartner.Size = new System.Drawing.Size(60, 17);
            this.chkPartner.TabIndex = 29;
            this.chkPartner.Text = "Partner";
            this.chkPartner.UseVisualStyleBackColor = true;
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
            // txtDeltaZ
            // 
            this.txtDeltaZ.Location = new System.Drawing.Point(366, 117);
            this.txtDeltaZ.Name = "txtDeltaZ";
            this.txtDeltaZ.Size = new System.Drawing.Size(45, 20);
            this.txtDeltaZ.TabIndex = 39;
            this.txtDeltaZ.Leave += new System.EventHandler(this.OnDeltaZLeave);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(297, 120);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(39, 13);
            this.label7.TabIndex = 38;
            this.label7.Text = "DeltaZ";
            // 
            // txtDeltaSlope
            // 
            this.txtDeltaSlope.Location = new System.Drawing.Point(366, 93);
            this.txtDeltaSlope.Name = "txtDeltaSlope";
            this.txtDeltaSlope.Size = new System.Drawing.Size(45, 20);
            this.txtDeltaSlope.TabIndex = 37;
            this.txtDeltaSlope.Leave += new System.EventHandler(this.OnDeltaSlopeLeave);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(297, 96);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(59, 13);
            this.label6.TabIndex = 36;
            this.label6.Text = "DeltaSlope";
            // 
            // txtRadius
            // 
            this.txtRadius.Location = new System.Drawing.Point(366, 70);
            this.txtRadius.Name = "txtRadius";
            this.txtRadius.Size = new System.Drawing.Size(45, 20);
            this.txtRadius.TabIndex = 35;
            this.txtRadius.Leave += new System.EventHandler(this.OnRadiusLeave);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(297, 73);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(40, 13);
            this.label5.TabIndex = 34;
            this.label5.Text = "Radius";
            // 
            // txtOpening
            // 
            this.txtOpening.Location = new System.Drawing.Point(366, 47);
            this.txtOpening.Name = "txtOpening";
            this.txtOpening.Size = new System.Drawing.Size(45, 20);
            this.txtOpening.TabIndex = 33;
            this.txtOpening.Leave += new System.EventHandler(this.OnOpeningLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(297, 50);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(47, 13);
            this.label4.TabIndex = 32;
            this.label4.Text = "Opening";
            // 
            // btnShowRelatedTracks
            // 
            this.btnShowRelatedTracks.Location = new System.Drawing.Point(3, 44);
            this.btnShowRelatedTracks.Name = "btnShowRelatedTracks";
            this.btnShowRelatedTracks.Size = new System.Drawing.Size(157, 24);
            this.btnShowRelatedTracks.TabIndex = 31;
            this.btnShowRelatedTracks.Text = "Related tracks";
            this.btnShowRelatedTracks.Click += new System.EventHandler(this.btnShowRelatedTracks_Click);
            // 
            // btnShowRelatedSegments
            // 
            this.btnShowRelatedSegments.Location = new System.Drawing.Point(3, 73);
            this.btnShowRelatedSegments.Name = "btnShowRelatedSegments";
            this.btnShowRelatedSegments.Size = new System.Drawing.Size(157, 24);
            this.btnShowRelatedSegments.TabIndex = 42;
            this.btnShowRelatedSegments.Text = "Related segments";
            this.btnShowRelatedSegments.Click += new System.EventHandler(this.btnShowRelatedSegments_Click);
            // 
            // EnableLabelCheck
            // 
            this.EnableLabelCheck.AutoSize = true;
            this.EnableLabelCheck.Location = new System.Drawing.Point(71, 309);
            this.EnableLabelCheck.Name = "EnableLabelCheck";
            this.EnableLabelCheck.Size = new System.Drawing.Size(82, 17);
            this.EnableLabelCheck.TabIndex = 46;
            this.EnableLabelCheck.Text = "Show Label";
            this.EnableLabelCheck.UseVisualStyleBackColor = true;
            this.EnableLabelCheck.CheckedChanged += new System.EventHandler(this.EnableLabelCheck_CheckedChanged);
            // 
            // HighlightCheck
            // 
            this.HighlightCheck.AutoSize = true;
            this.HighlightCheck.Location = new System.Drawing.Point(4, 309);
            this.HighlightCheck.Name = "HighlightCheck";
            this.HighlightCheck.Size = new System.Drawing.Size(61, 17);
            this.HighlightCheck.TabIndex = 45;
            this.HighlightCheck.Text = "Higlight";
            this.HighlightCheck.UseVisualStyleBackColor = true;
            this.HighlightCheck.CheckedChanged += new System.EventHandler(this.HighlightCheck_CheckedChanged);
            // 
            // SetLabelButton
            // 
            this.SetLabelButton.Location = new System.Drawing.Point(161, 304);
            this.SetLabelButton.Name = "SetLabelButton";
            this.SetLabelButton.Size = new System.Drawing.Size(89, 24);
            this.SetLabelButton.TabIndex = 44;
            this.SetLabelButton.Text = "Set Label";
            this.SetLabelButton.Click += new System.EventHandler(this.SetLabelButton_Click);
            // 
            // LabelText
            // 
            this.LabelText.Location = new System.Drawing.Point(256, 307);
            this.LabelText.Name = "LabelText";
            this.LabelText.Size = new System.Drawing.Size(173, 20);
            this.LabelText.TabIndex = 43;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage7);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Controls.Add(this.tabPage6);
            this.tabControl1.Location = new System.Drawing.Point(4, 2);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(429, 296);
            this.tabControl1.TabIndex = 47;
            // 
            // tabPage1
            // 
            this.tabPage1.BackColor = System.Drawing.Color.Transparent;
            this.tabPage1.Controls.Add(this.KillButton);
            this.tabPage1.Controls.Add(this.txtDataSet);
            this.tabPage1.Controls.Add(this.label12);
            this.tabPage1.Controls.Add(this.GeneralDumpFileButton);
            this.tabPage1.Controls.Add(this.GeneralList);
            this.tabPage1.Controls.Add(this.GeneralDumpFileText);
            this.tabPage1.Controls.Add(this.GeneralSelButton);
            this.tabPage1.Controls.Add(this.CommentText);
            this.tabPage1.Controls.Add(this.SetCommentButton);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(421, 270);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "General";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // KillButton
            // 
            this.KillButton.Location = new System.Drawing.Point(364, 6);
            this.KillButton.Name = "KillButton";
            this.KillButton.Size = new System.Drawing.Size(48, 24);
            this.KillButton.TabIndex = 12;
            this.KillButton.Text = "Kill";
            this.KillButton.Click += new System.EventHandler(this.KillButton_Click);
            // 
            // txtDataSet
            // 
            this.txtDataSet.Location = new System.Drawing.Point(74, 6);
            this.txtDataSet.Name = "txtDataSet";
            this.txtDataSet.ReadOnly = true;
            this.txtDataSet.Size = new System.Drawing.Size(284, 20);
            this.txtDataSet.TabIndex = 11;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(15, 9);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(46, 13);
            this.label12.TabIndex = 10;
            this.label12.Text = "DataSet";
            // 
            // tabPage2
            // 
            this.tabPage2.BackColor = System.Drawing.Color.Transparent;
            this.tabPage2.Controls.Add(this.GoToSelTrackButton);
            this.tabPage2.Controls.Add(this.TrackDumpButton);
            this.tabPage2.Controls.Add(this.TrackList);
            this.tabPage2.Controls.Add(this.TrackDumpFileText);
            this.tabPage2.Controls.Add(this.TrackDumpSelButton);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(421, 270);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Tracks";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // tabPage7
            // 
            this.tabPage7.Controls.Add(this.LaunchScanButton);
            this.tabPage7.Controls.Add(this.ConnToVtxText);
            this.tabPage7.Controls.Add(this.ConnToVtxCheck);
            this.tabPage7.Controls.Add(this.label3);
            this.tabPage7.Controls.Add(this.UpStopsText);
            this.tabPage7.Controls.Add(this.label8);
            this.tabPage7.Controls.Add(this.DownStopsText);
            this.tabPage7.Controls.Add(this.label2);
            this.tabPage7.Controls.Add(this.UpTracksCheck);
            this.tabPage7.Controls.Add(this.UpSegsText);
            this.tabPage7.Controls.Add(this.label1);
            this.tabPage7.Controls.Add(this.DownTracksCheck);
            this.tabPage7.Controls.Add(this.DownSegsText);
            this.tabPage7.Controls.Add(this.AppendToSelButton);
            this.tabPage7.Controls.Add(this.AppendToFileText);
            this.tabPage7.Controls.Add(this.AppendToButton);
            this.tabPage7.Location = new System.Drawing.Point(4, 22);
            this.tabPage7.Name = "tabPage7";
            this.tabPage7.Size = new System.Drawing.Size(421, 270);
            this.tabPage7.TabIndex = 6;
            this.tabPage7.Text = "Projections";
            this.tabPage7.UseVisualStyleBackColor = true;
            // 
            // LaunchScanButton
            // 
            this.LaunchScanButton.Location = new System.Drawing.Point(314, 225);
            this.LaunchScanButton.Name = "LaunchScanButton";
            this.LaunchScanButton.Size = new System.Drawing.Size(89, 24);
            this.LaunchScanButton.TabIndex = 70;
            this.LaunchScanButton.Text = "Launch Scan";
            this.LaunchScanButton.Click += new System.EventHandler(this.LaunchScanButton_Click);
            // 
            // ConnToVtxText
            // 
            this.ConnToVtxText.Location = new System.Drawing.Point(275, 68);
            this.ConnToVtxText.Name = "ConnToVtxText";
            this.ConnToVtxText.Size = new System.Drawing.Size(50, 20);
            this.ConnToVtxText.TabIndex = 56;
            // 
            // ConnToVtxCheck
            // 
            this.ConnToVtxCheck.AutoSize = true;
            this.ConnToVtxCheck.Checked = true;
            this.ConnToVtxCheck.CheckState = System.Windows.Forms.CheckState.Checked;
            this.ConnToVtxCheck.Location = new System.Drawing.Point(13, 70);
            this.ConnToVtxCheck.Name = "ConnToVtxCheck";
            this.ConnToVtxCheck.Size = new System.Drawing.Size(162, 17);
            this.ConnToVtxCheck.TabIndex = 55;
            this.ConnToVtxCheck.Text = "Tracks connecting to vertex:";
            this.ConnToVtxCheck.UseVisualStyleBackColor = true;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(331, 42);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(57, 13);
            this.label3.TabIndex = 54;
            this.label3.Text = "stop layers";
            // 
            // UpStopsText
            // 
            this.UpStopsText.Location = new System.Drawing.Point(275, 39);
            this.UpStopsText.Name = "UpStopsText";
            this.UpStopsText.Size = new System.Drawing.Size(50, 20);
            this.UpStopsText.TabIndex = 53;
            this.UpStopsText.Leave += new System.EventHandler(this.OnUpStopsLeave);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(331, 15);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(57, 13);
            this.label8.TabIndex = 52;
            this.label8.Text = "stop layers";
            // 
            // DownStopsText
            // 
            this.DownStopsText.Location = new System.Drawing.Point(275, 12);
            this.DownStopsText.Name = "DownStopsText";
            this.DownStopsText.Size = new System.Drawing.Size(50, 20);
            this.DownStopsText.TabIndex = 51;
            this.DownStopsText.Leave += new System.EventHandler(this.OnDownStopsLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(191, 42);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(61, 13);
            this.label2.TabIndex = 50;
            this.label2.Text = "track layers";
            // 
            // UpTracksCheck
            // 
            this.UpTracksCheck.AutoSize = true;
            this.UpTracksCheck.Checked = true;
            this.UpTracksCheck.CheckState = System.Windows.Forms.CheckState.Checked;
            this.UpTracksCheck.Location = new System.Drawing.Point(13, 41);
            this.UpTracksCheck.Name = "UpTracksCheck";
            this.UpTracksCheck.Size = new System.Drawing.Size(103, 17);
            this.UpTracksCheck.TabIndex = 49;
            this.UpTracksCheck.Text = "Upstream tracks";
            this.UpTracksCheck.UseVisualStyleBackColor = true;
            // 
            // UpSegsText
            // 
            this.UpSegsText.Location = new System.Drawing.Point(135, 39);
            this.UpSegsText.Name = "UpSegsText";
            this.UpSegsText.Size = new System.Drawing.Size(50, 20);
            this.UpSegsText.TabIndex = 48;
            this.UpSegsText.Leave += new System.EventHandler(this.OnUpSegsLeave);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(191, 15);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(61, 13);
            this.label1.TabIndex = 47;
            this.label1.Text = "track layers";
            // 
            // DownTracksCheck
            // 
            this.DownTracksCheck.AutoSize = true;
            this.DownTracksCheck.Checked = true;
            this.DownTracksCheck.CheckState = System.Windows.Forms.CheckState.Checked;
            this.DownTracksCheck.Location = new System.Drawing.Point(13, 14);
            this.DownTracksCheck.Name = "DownTracksCheck";
            this.DownTracksCheck.Size = new System.Drawing.Size(117, 17);
            this.DownTracksCheck.TabIndex = 46;
            this.DownTracksCheck.Text = "Downstream tracks";
            this.DownTracksCheck.UseVisualStyleBackColor = true;
            // 
            // DownSegsText
            // 
            this.DownSegsText.Location = new System.Drawing.Point(135, 12);
            this.DownSegsText.Name = "DownSegsText";
            this.DownSegsText.Size = new System.Drawing.Size(50, 20);
            this.DownSegsText.TabIndex = 45;
            this.DownSegsText.Leave += new System.EventHandler(this.OnDownSegsLeave);
            // 
            // AppendToSelButton
            // 
            this.AppendToSelButton.Location = new System.Drawing.Point(370, 123);
            this.AppendToSelButton.Name = "AppendToSelButton";
            this.AppendToSelButton.Size = new System.Drawing.Size(33, 24);
            this.AppendToSelButton.TabIndex = 44;
            this.AppendToSelButton.Text = "...";
            this.AppendToSelButton.Click += new System.EventHandler(this.AppendToSelButton_Click);
            // 
            // AppendToFileText
            // 
            this.AppendToFileText.Location = new System.Drawing.Point(108, 126);
            this.AppendToFileText.Name = "AppendToFileText";
            this.AppendToFileText.Size = new System.Drawing.Size(256, 20);
            this.AppendToFileText.TabIndex = 43;
            // 
            // AppendToButton
            // 
            this.AppendToButton.Location = new System.Drawing.Point(13, 123);
            this.AppendToButton.Name = "AppendToButton";
            this.AppendToButton.Size = new System.Drawing.Size(89, 24);
            this.AppendToButton.TabIndex = 42;
            this.AppendToButton.Text = "Append to";
            this.AppendToButton.Click += new System.EventHandler(this.AppendToButton_Click);
            // 
            // tabPage3
            // 
            this.tabPage3.BackColor = System.Drawing.Color.Transparent;
            this.tabPage3.Controls.Add(this.rdBoth);
            this.tabPage3.Controls.Add(this.rdDataset);
            this.tabPage3.Controls.Add(this.rdShow);
            this.tabPage3.Controls.Add(this.btnUpPartners);
            this.tabPage3.Controls.Add(this.btnDownPartners);
            this.tabPage3.Controls.Add(this.btnUpTracks);
            this.tabPage3.Controls.Add(this.btnDownTracks);
            this.tabPage3.Controls.Add(this.btnUpDecays);
            this.tabPage3.Controls.Add(this.btnDownDecays);
            this.tabPage3.Controls.Add(this.SelectIPButton);
            this.tabPage3.Controls.Add(this.btnShowRelatedTracks);
            this.tabPage3.Controls.Add(this.label4);
            this.tabPage3.Controls.Add(this.txtOpening);
            this.tabPage3.Controls.Add(this.label5);
            this.tabPage3.Controls.Add(this.btnShowRelatedSegments);
            this.tabPage3.Controls.Add(this.txtRadius);
            this.tabPage3.Controls.Add(this.groupBox5);
            this.tabPage3.Controls.Add(this.label6);
            this.tabPage3.Controls.Add(this.txtDeltaZ);
            this.tabPage3.Controls.Add(this.txtDeltaSlope);
            this.tabPage3.Controls.Add(this.label7);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(421, 270);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Neighborhood";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // btnUpPartners
            // 
            this.btnUpPartners.Location = new System.Drawing.Point(166, 237);
            this.btnUpPartners.Name = "btnUpPartners";
            this.btnUpPartners.Size = new System.Drawing.Size(157, 24);
            this.btnUpPartners.TabIndex = 48;
            this.btnUpPartners.Text = "Quick Upstream Partners";
            this.btnUpPartners.Click += new System.EventHandler(this.btnUpPartners_Click);
            // 
            // btnDownPartners
            // 
            this.btnDownPartners.Location = new System.Drawing.Point(166, 207);
            this.btnDownPartners.Name = "btnDownPartners";
            this.btnDownPartners.Size = new System.Drawing.Size(157, 24);
            this.btnDownPartners.TabIndex = 47;
            this.btnDownPartners.Text = "Quick Downstream Partners";
            this.btnDownPartners.Click += new System.EventHandler(this.btnDownPartners_Click);
            // 
            // btnUpTracks
            // 
            this.btnUpTracks.Location = new System.Drawing.Point(3, 174);
            this.btnUpTracks.Name = "btnUpTracks";
            this.btnUpTracks.Size = new System.Drawing.Size(157, 24);
            this.btnUpTracks.TabIndex = 46;
            this.btnUpTracks.Text = "Quick Upstream Tracks";
            this.btnUpTracks.Click += new System.EventHandler(this.btnUpTracks_Click);
            // 
            // btnDownTracks
            // 
            this.btnDownTracks.Location = new System.Drawing.Point(3, 147);
            this.btnDownTracks.Name = "btnDownTracks";
            this.btnDownTracks.Size = new System.Drawing.Size(157, 24);
            this.btnDownTracks.TabIndex = 45;
            this.btnDownTracks.Text = "Quick Downstream Tracks";
            this.btnDownTracks.Click += new System.EventHandler(this.btnDownTracks_Click);
            // 
            // btnUpDecays
            // 
            this.btnUpDecays.Location = new System.Drawing.Point(166, 177);
            this.btnUpDecays.Name = "btnUpDecays";
            this.btnUpDecays.Size = new System.Drawing.Size(157, 24);
            this.btnUpDecays.TabIndex = 44;
            this.btnUpDecays.Text = "Quick Upstream Decays";
            this.btnUpDecays.Click += new System.EventHandler(this.btnUpDecays_Click);
            // 
            // btnDownDecays
            // 
            this.btnDownDecays.Location = new System.Drawing.Point(166, 147);
            this.btnDownDecays.Name = "btnDownDecays";
            this.btnDownDecays.Size = new System.Drawing.Size(157, 24);
            this.btnDownDecays.TabIndex = 43;
            this.btnDownDecays.Text = "Quick Downstream Decays";
            this.btnDownDecays.Click += new System.EventHandler(this.btnDownDecays_Click);
            // 
            // tabPage4
            // 
            this.tabPage4.BackColor = System.Drawing.Color.Transparent;
            this.tabPage4.Controls.Add(this.radioUseLastSeg);
            this.tabPage4.Controls.Add(this.radioUseFit);
            this.tabPage4.Controls.Add(this.label9);
            this.tabPage4.Controls.Add(this.txtExtrapolationDist);
            this.tabPage4.Controls.Add(this.ListFits);
            this.tabPage4.Controls.Add(this.AddTracksButton);
            this.tabPage4.Location = new System.Drawing.Point(4, 22);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(421, 270);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "Add to vertex fit";
            this.tabPage4.UseVisualStyleBackColor = true;
            // 
            // radioUseLastSeg
            // 
            this.radioUseLastSeg.AutoSize = true;
            this.radioUseLastSeg.Location = new System.Drawing.Point(14, 96);
            this.radioUseLastSeg.Name = "radioUseLastSeg";
            this.radioUseLastSeg.Size = new System.Drawing.Size(106, 17);
            this.radioUseLastSeg.TabIndex = 14;
            this.radioUseLastSeg.Text = "Use last segment";
            this.radioUseLastSeg.UseVisualStyleBackColor = true;
            // 
            // radioUseFit
            // 
            this.radioUseFit.AutoSize = true;
            this.radioUseFit.Checked = true;
            this.radioUseFit.Location = new System.Drawing.Point(14, 73);
            this.radioUseFit.Name = "radioUseFit";
            this.radioUseFit.Size = new System.Drawing.Size(55, 17);
            this.radioUseFit.TabIndex = 13;
            this.radioUseFit.TabStop = true;
            this.radioUseFit.Text = "Use fit";
            this.radioUseFit.UseVisualStyleBackColor = true;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(11, 12);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(111, 13);
            this.label9.TabIndex = 12;
            this.label9.Text = "Extrapolation distance";
            // 
            // txtExtrapolationDist
            // 
            this.txtExtrapolationDist.Location = new System.Drawing.Point(139, 12);
            this.txtExtrapolationDist.Name = "txtExtrapolationDist";
            this.txtExtrapolationDist.Size = new System.Drawing.Size(85, 20);
            this.txtExtrapolationDist.TabIndex = 11;
            this.txtExtrapolationDist.Leave += new System.EventHandler(this.OnExtrapDistanceLeave);
            // 
            // ListFits
            // 
            this.ListFits.FormattingEnabled = true;
            this.ListFits.Location = new System.Drawing.Point(240, 12);
            this.ListFits.Name = "ListFits";
            this.ListFits.Size = new System.Drawing.Size(178, 173);
            this.ListFits.Sorted = true;
            this.ListFits.TabIndex = 9;
            // 
            // AddTracksButton
            // 
            this.AddTracksButton.Location = new System.Drawing.Point(139, 38);
            this.AddTracksButton.Name = "AddTracksButton";
            this.AddTracksButton.Size = new System.Drawing.Size(85, 24);
            this.AddTracksButton.TabIndex = 8;
            this.AddTracksButton.Text = "Add tracks";
            this.AddTracksButton.Click += new System.EventHandler(this.AddTracksButton_Click);
            // 
            // tabPage5
            // 
            this.tabPage5.Controls.Add(this.txtAttrValue);
            this.tabPage5.Controls.Add(this.txtAttrName);
            this.tabPage5.Controls.Add(this.cmdAddSetAttribute);
            this.tabPage5.Controls.Add(this.cmdRemoveAttributes);
            this.tabPage5.Controls.Add(this.AttributeList);
            this.tabPage5.Location = new System.Drawing.Point(4, 22);
            this.tabPage5.Name = "tabPage5";
            this.tabPage5.Size = new System.Drawing.Size(421, 270);
            this.tabPage5.TabIndex = 4;
            this.tabPage5.Text = "Attributes";
            this.tabPage5.UseVisualStyleBackColor = true;
            // 
            // txtAttrValue
            // 
            this.txtAttrValue.Location = new System.Drawing.Point(226, 175);
            this.txtAttrValue.Name = "txtAttrValue";
            this.txtAttrValue.Size = new System.Drawing.Size(180, 20);
            this.txtAttrValue.TabIndex = 48;
            // 
            // txtAttrName
            // 
            this.txtAttrName.Location = new System.Drawing.Point(87, 175);
            this.txtAttrName.Name = "txtAttrName";
            this.txtAttrName.Size = new System.Drawing.Size(129, 20);
            this.txtAttrName.TabIndex = 47;
            // 
            // cmdAddSetAttribute
            // 
            this.cmdAddSetAttribute.Location = new System.Drawing.Point(13, 175);
            this.cmdAddSetAttribute.Name = "cmdAddSetAttribute";
            this.cmdAddSetAttribute.Size = new System.Drawing.Size(68, 24);
            this.cmdAddSetAttribute.TabIndex = 46;
            this.cmdAddSetAttribute.Text = "Add/Set";
            this.cmdAddSetAttribute.Click += new System.EventHandler(this.cmdAddSetAttribute_Click);
            // 
            // cmdRemoveAttributes
            // 
            this.cmdRemoveAttributes.Location = new System.Drawing.Point(13, 206);
            this.cmdRemoveAttributes.Name = "cmdRemoveAttributes";
            this.cmdRemoveAttributes.Size = new System.Drawing.Size(172, 24);
            this.cmdRemoveAttributes.TabIndex = 45;
            this.cmdRemoveAttributes.Text = "Remove selected attribute(s)";
            this.cmdRemoveAttributes.Click += new System.EventHandler(this.cmdRemoveAttributes_Click);
            // 
            // AttributeList
            // 
            this.AttributeList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader4,
            this.columnHeader13});
            this.AttributeList.FullRowSelect = true;
            this.AttributeList.GridLines = true;
            this.AttributeList.Location = new System.Drawing.Point(13, 17);
            this.AttributeList.Name = "AttributeList";
            this.AttributeList.Size = new System.Drawing.Size(393, 152);
            this.AttributeList.TabIndex = 44;
            this.AttributeList.UseCompatibleStateImageBehavior = false;
            this.AttributeList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Index";
            this.columnHeader4.Width = 252;
            // 
            // columnHeader13
            // 
            this.columnHeader13.Text = "Value";
            this.columnHeader13.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader13.Width = 120;
            // 
            // tabPage6
            // 
            this.tabPage6.Controls.Add(this.chkFBDeadMaterial);
            this.tabPage6.Controls.Add(this.chkFBTauDecay);
            this.tabPage6.Controls.Add(this.chkFBCharm);
            this.tabPage6.Controls.Add(this.chkFBPrimary);
            this.tabPage6.Controls.Add(this.chkFBEvent);
            this.tabPage6.Location = new System.Drawing.Point(4, 22);
            this.tabPage6.Name = "tabPage6";
            this.tabPage6.Size = new System.Drawing.Size(421, 270);
            this.tabPage6.TabIndex = 5;
            this.tabPage6.Text = "Feedback";
            this.tabPage6.UseVisualStyleBackColor = true;
            // 
            // chkFBDeadMaterial
            // 
            this.chkFBDeadMaterial.AutoSize = true;
            this.chkFBDeadMaterial.Location = new System.Drawing.Point(13, 203);
            this.chkFBDeadMaterial.Name = "chkFBDeadMaterial";
            this.chkFBDeadMaterial.Size = new System.Drawing.Size(143, 17);
            this.chkFBDeadMaterial.TabIndex = 4;
            this.chkFBDeadMaterial.Text = "Vertex is in dead material";
            this.chkFBDeadMaterial.UseVisualStyleBackColor = true;
            this.chkFBDeadMaterial.CheckedChanged += new System.EventHandler(this.OnFBDeadMaterialChecked);
            // 
            // chkFBTauDecay
            // 
            this.chkFBTauDecay.AutoSize = true;
            this.chkFBTauDecay.Location = new System.Drawing.Point(13, 158);
            this.chkFBTauDecay.Name = "chkFBTauDecay";
            this.chkFBTauDecay.Size = new System.Drawing.Size(109, 17);
            this.chkFBTauDecay.TabIndex = 3;
            this.chkFBTauDecay.Text = "Tau decay vertex";
            this.chkFBTauDecay.UseVisualStyleBackColor = true;
            this.chkFBTauDecay.CheckedChanged += new System.EventHandler(this.OnFBTauChecked);
            // 
            // chkFBCharm
            // 
            this.chkFBCharm.AutoSize = true;
            this.chkFBCharm.Location = new System.Drawing.Point(13, 113);
            this.chkFBCharm.Name = "chkFBCharm";
            this.chkFBCharm.Size = new System.Drawing.Size(120, 17);
            this.chkFBCharm.TabIndex = 2;
            this.chkFBCharm.Text = "Charm decay vertex";
            this.chkFBCharm.UseVisualStyleBackColor = true;
            this.chkFBCharm.CheckedChanged += new System.EventHandler(this.OnFBCharmChecked);
            // 
            // chkFBPrimary
            // 
            this.chkFBPrimary.AutoSize = true;
            this.chkFBPrimary.Location = new System.Drawing.Point(13, 68);
            this.chkFBPrimary.Name = "chkFBPrimary";
            this.chkFBPrimary.Size = new System.Drawing.Size(92, 17);
            this.chkFBPrimary.TabIndex = 1;
            this.chkFBPrimary.Text = "Primary vertex";
            this.chkFBPrimary.UseVisualStyleBackColor = true;
            this.chkFBPrimary.CheckedChanged += new System.EventHandler(this.OnFBPrimaryChecked);
            // 
            // chkFBEvent
            // 
            this.chkFBEvent.AutoSize = true;
            this.chkFBEvent.Location = new System.Drawing.Point(13, 16);
            this.chkFBEvent.Name = "chkFBEvent";
            this.chkFBEvent.Size = new System.Drawing.Size(155, 17);
            this.chkFBEvent.TabIndex = 0;
            this.chkFBEvent.Text = "Include vertex in Feedback";
            this.chkFBEvent.UseVisualStyleBackColor = true;
            this.chkFBEvent.CheckedChanged += new System.EventHandler(this.OnFBEventChecked);
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
            // rdShow
            // 
            this.rdShow.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdShow.AutoSize = true;
            this.rdShow.Checked = true;
            this.rdShow.Location = new System.Drawing.Point(4, 103);
            this.rdShow.Name = "rdShow";
            this.rdShow.Size = new System.Drawing.Size(44, 23);
            this.rdShow.TabIndex = 49;
            this.rdShow.Text = "Show";
            this.rdShow.UseVisualStyleBackColor = true;
            // 
            // rdDataset
            // 
            this.rdDataset.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdDataset.AutoSize = true;
            this.rdDataset.Location = new System.Drawing.Point(54, 103);
            this.rdDataset.Name = "rdDataset";
            this.rdDataset.Size = new System.Drawing.Size(54, 23);
            this.rdDataset.TabIndex = 50;
            this.rdDataset.Text = "Dataset";
            this.rdDataset.UseVisualStyleBackColor = true;
            // 
            // rdBoth
            // 
            this.rdBoth.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdBoth.AutoSize = true;
            this.rdBoth.Location = new System.Drawing.Point(112, 103);
            this.rdBoth.Name = "rdBoth";
            this.rdBoth.Size = new System.Drawing.Size(39, 23);
            this.rdBoth.TabIndex = 51;
            this.rdBoth.Text = "Both";
            this.rdBoth.UseVisualStyleBackColor = true;
            // 
            // VertexBrowser
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(435, 337);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.EnableLabelCheck);
            this.Controls.Add(this.HighlightCheck);
            this.Controls.Add(this.SetLabelButton);
            this.Controls.Add(this.LabelText);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "VertexBrowser";
            this.Text = "VertexBrowser";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Closed += new System.EventHandler(this.OnClose);
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage7.ResumeLayout(false);
            this.tabPage7.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.tabPage5.ResumeLayout(false);
            this.tabPage5.PerformLayout();
            this.tabPage6.ResumeLayout(false);
            this.tabPage6.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		private void SetCommentButton_Click(object sender, System.EventArgs e)
		{
			m_Vertex.Comment = CommentText.Text;
			GeneralList.Items[2].SubItems[1].Text = m_Vertex.Comment;
		}

		private void GeneralSelButton_Click(object sender, System.EventArgs e)
		{
			System.Windows.Forms.SaveFileDialog ddlg = new System.Windows.Forms.SaveFileDialog();
			ddlg.Title = "Select file to dump info of vertex #" + m_Vertex.Id.ToString();
			ddlg.FileName = (GeneralDumpFileText.Text.Length == 0) ? ("VtxInfo_" + m_Vertex.Id.ToString() + ".txt") : GeneralDumpFileText.Text;
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

		private void TrackDumpSelButton_Click(object sender, System.EventArgs e)
		{
			System.Windows.Forms.SaveFileDialog sdlg = new System.Windows.Forms.SaveFileDialog();
			sdlg.Title = "Select file to dump tracks of vertex #" + m_Vertex.Id.ToString();
			sdlg.FileName = (TrackDumpFileText.Text.Length == 0) ? ("VtxTracks_" + m_Vertex.Id.ToString() + ".txt") : TrackDumpFileText.Text;
			sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";		
			if (sdlg.ShowDialog() == DialogResult.OK) TrackDumpFileText.Text = sdlg.FileName;
		}

		private void TrackDumpButton_Click(object sender, System.EventArgs e)
		{
			System.IO.StreamWriter w = null;
			try
			{
				w = new System.IO.StreamWriter(TrackDumpFileText.Text);
				int i, c;
				c = TrackList.Columns.Count;
				for (i = 0; i < c; i++)
					if (i == 0) w.Write(TrackList.Columns[i].Text);
					else w.Write("\t" + TrackList.Columns[i].Text);
				w.WriteLine();
				foreach (System.Windows.Forms.ListViewItem lvi in TrackList.Items)				
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

		private void GoToSelTrackButton_Click(object sender, System.EventArgs e)
		{
			if (TrackList.SelectedItems.Count != 1) return;
			TrackBrowser.Browse((SySal.TotalScan.Track)TrackList.SelectedItems[0].Tag, m_Layers, SelectForIP, SelectForGraph, m_Event, m_V);
		}

		protected internal static System.Collections.ArrayList AvailableBrowsers = new System.Collections.ArrayList();

        public static void CloseAll()
        {
            while (AvailableBrowsers.Count > 0) ((VertexBrowser)AvailableBrowsers[0]).Close();
        }

        public static void RefreshAll()
        {
            foreach (VertexBrowser b in AvailableBrowsers) b.Vertex = b.Vertex;
        }

		public static VertexBrowser Browse(SySal.TotalScan.Vertex vtx, SySal.TotalScan.Volume.LayerList ll, SySal.Executables.EasyReconstruct.IPSelector ipsel, SySal.Executables.EasyReconstruct.TrackSelector tksel, long ev, SySal.TotalScan.Volume v)
		{
			foreach (VertexBrowser b in AvailableBrowsers)
			{
				if (b.Vertex == vtx)
				{
					b.BringToFront();
					return b;
				}
			}
			VertexBrowser newb = new VertexBrowser(ll, ev, v);
			newb.Vertex = vtx;
            newb.SelectForIP = ipsel;
            newb.SelectForGraph = tksel;
			newb.Show();
			AvailableBrowsers.Add(newb);
            tksel.SubscribeOnAddFit(new dGenericEvent(newb.RefreshFitListButton_Click));
            newb.RefreshFitListButton_Click(tksel, null);
			return newb;
		}

		private void OnClose(object sender, System.EventArgs e)
		{
            try
            {
                AvailableBrowsers.Remove(this);
            }
            catch (Exception) { }
        }

        private void SelectIPButton_Click(object sender, EventArgs e)
        {
            SelectForIP.SelectVertex(m_Vertex);
        }

        private double m_Opening = 2.0;

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

        private double m_Radius = -100.0;

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

        private void RefreshParameters()
        {
            txtOpening.Text = m_Opening.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtRadius.Text = m_Radius.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDeltaSlope.Text = m_DeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDeltaZ.Text = m_DeltaZ.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtExtrapolationDist.Text = m_ExtrapDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
            DownSegsText.Text = m_DownSegs.ToString();
            UpSegsText.Text = m_UpSegs.ToString();
            DownStopsText.Text = m_DownStops.ToString();
            UpStopsText.Text = m_UpStops.ToString();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            RefreshParameters();
            AppendToFileText.Text = DisplayForm.DefaultProjFileName;
        }

        public class TrackFilter
        {
            SySal.BasicTypes.Vector m_Start;

            bool m_IsDownstreamDir;

            double m_Opening;

            double m_Radius;

            double m_DeltaSlope;

            double m_DeltaZ;

            bool m_IsParent;

            public TrackFilter(SySal.TotalScan.Vertex vtx, bool isdownstreamdir, double opening, double radius, double deltaslope, double deltaz, bool isparent)
            {
                m_Start.X = vtx.X;
                m_Start.Y = vtx.Y;
                m_Start.Z = vtx.Z;
                m_IsDownstreamDir = isdownstreamdir;
                m_Opening = opening;
                m_Radius = radius;
                m_DeltaSlope = deltaslope;
                m_DeltaZ = deltaz;
                m_IsParent = isparent;
            }

            public bool Filter(SySal.TotalScan.Track t)
            {
                SySal.BasicTypes.Vector end;
                SySal.BasicTypes.Vector slope;
                if (m_IsDownstreamDir)
                {
                    if ((t.Upstream_Z < m_Start.Z) ^ m_IsParent) return false;
                    end.X = t.Upstream_PosX + (t.Upstream_Z - t.Upstream_PosZ) * t.Upstream_SlopeX;
                    end.Y = t.Upstream_PosY + (t.Upstream_Z - t.Upstream_PosZ) * t.Upstream_SlopeY;
                    end.Z = t.Upstream_Z;
                    slope.X = t.Upstream_SlopeX;
                    slope.Y = t.Upstream_SlopeY;
                    slope.Z = 1.0;
                }
                else
                {
                    if ((t.Downstream_Z > m_Start.Z) ^ m_IsParent) return false;
                    end.X = t.Downstream_PosX + (t.Downstream_Z - t.Downstream_PosZ) * t.Downstream_SlopeX;
                    end.Y = t.Downstream_PosY + (t.Downstream_Z - t.Downstream_PosZ) * t.Downstream_SlopeY;
                    end.Z = t.Downstream_Z;
                    slope.X = t.Downstream_SlopeX;
                    slope.Y = t.Downstream_SlopeY;
                    slope.Z = 1.0;
                }
                if (m_DeltaZ >= 0.0 && Math.Abs(end.Z - m_Start.Z) > m_DeltaZ) return false;
                if (m_Radius >= 0.0)
                {
                    double dx = m_Start.X - end.X;
                    double dy = m_Start.Y - end.Y;
                    if (dx * dx + dy * dy > m_Radius * m_Radius) return false;
                    if (m_DeltaSlope < 0.0) return true;
                    else
                    {
                        double dz = end.Z - m_Start.Z;
                        dx = (end.X - m_Start.X) / dz - slope.X;
                        dy = (end.Y - m_Start.Y) / dz - slope.Y;
                        return (dx * dx + dy * dy < m_DeltaSlope * m_DeltaSlope);
                    }

                }
                if (m_Opening >= 0.0)
                {
                    double dz = end.Z - m_Start.Z;
                    if (dz != 0.0)
                    {
                        double dx = (end.X - m_Start.X) / dz;
                        double dy = (end.Y - m_Start.Y) / dz;
                        if (dx * dx + dy * dy > m_Opening * m_Opening) return false;
                    }
                    if (m_DeltaSlope < 0.0) return true;
                    else
                    {
                        double dx = (end.X - m_Start.X) / dz - slope.X;
                        double dy = (end.Y - m_Start.Y) / dz - slope.Y;
                        return (dx * dx + dy * dy < m_DeltaSlope * m_DeltaSlope);
                    }
                }
                return false;
            }

            public bool FilterSeg(SySal.TotalScan.Segment s)
            {
                SySal.BasicTypes.Vector end;
                SySal.BasicTypes.Vector slope;
                SySal.Tracking.MIPEmulsionTrackInfo info = s.Info;
                if (m_IsDownstreamDir)
                {
                    if (info.BottomZ < m_Start.Z) return false;
                    end.X = info.Intercept.X + (info.BottomZ - info.Intercept.Z) * info.Slope.X;
                    end.Y = info.Intercept.Y + (info.BottomZ - info.Intercept.Z) * info.Slope.Y;
                    end.Z = info.BottomZ;
                    slope.X = info.Slope.X;
                    slope.Y = info.Slope.Y;
                    slope.Z = 1.0;
                }
                else
                {
                    if (info.TopZ > m_Start.Z) return false;
                    end.X = info.Intercept.X + (info.TopZ - info.Intercept.Z) * info.Slope.X;
                    end.Y = info.Intercept.Y + (info.TopZ - info.Intercept.Z) * info.Slope.Y;
                    end.Z = info.TopZ;
                    slope.X = info.Slope.X;
                    slope.Y = info.Slope.Y;
                    slope.Z = 1.0;
                }
                if (m_DeltaZ >= 0.0 && Math.Abs(end.Z - m_Start.Z) > m_DeltaZ) return false;
                if (m_Radius >= 0.0)
                {
                    double dx = m_Start.X - end.X;
                    double dy = m_Start.Y - end.Y;
                    if (dx * dx + dy * dy > m_Radius * m_Radius) return false;
                    if (m_DeltaSlope < 0.0) return true;
                    else
                    {
                        double dz = end.Z - m_Start.Z;
                        dx = (end.X - m_Start.X) / dz - slope.X;
                        dy = (end.Y - m_Start.Y) / dz - slope.Y;
                        return (dx * dx + dy * dy < m_DeltaSlope * m_DeltaSlope);
                    }

                }
                if (m_Opening >= 0.0)
                {
                    double dz = end.Z - m_Start.Z;
                    if (dz != 0.0)
                    {
                        double dx = (end.X - m_Start.X) / dz;
                        double dy = (end.Y - m_Start.Y) / dz;
                        if (dx * dx + dy * dy > m_Opening * m_Opening) return false;
                    }
                    if (m_DeltaSlope < 0.0) return true;
                    else
                    {
                        double dx = (end.X - m_Start.X) / dz - slope.X;
                        double dy = (end.Y - m_Start.Y) / dz - slope.Y;
                        return (dx * dx + dy * dy < m_DeltaSlope * m_DeltaSlope);
                    }
                }
                return false;
            }        
        }

        private void btnShowRelatedTracks_Click(object sender, EventArgs e)
        {
            SelectForGraph.ShowTracks(new TrackFilter(m_Vertex, radioDownstreamDir.Checked, m_Opening, m_Radius, m_DeltaSlope, m_DeltaZ, chkPartner.Checked).Filter, rdShow.Checked || rdBoth.Checked, rdDataset.Checked || rdBoth.Checked);
        }

        private void btnShowRelatedSegments_Click(object sender, EventArgs e)
        {
            SelectForGraph.ShowSegments(new TrackFilter(m_Vertex, radioDownstreamDir.Checked, m_Opening, m_Radius, m_DeltaSlope, m_DeltaZ, chkPartner.Checked).FilterSeg, rdShow.Checked || rdBoth.Checked, rdDataset.Checked || rdBoth.Checked);
        }

        private void HighlightCheck_CheckedChanged(object sender, EventArgs e)
        {
            SelectForIP.Highlight(m_Vertex, HighlightCheck.Checked);
        }

        private void EnableLabelCheck_CheckedChanged(object sender, EventArgs e)
        {
            SelectForIP.EnableLabel(m_Vertex, EnableLabelCheck.Checked);
        }

        private void SetLabelButton_Click(object sender, EventArgs e)
        {
            SelectForIP.SetLabel(m_Vertex, LabelText.Text);
        }

        private void AddTracksButton_Click(object sender, EventArgs e)
        {
            int j;
            for (j = 0; j < m_Vertex.Length; j++)
            {
                SySal.TotalScan.Track m_Track = m_Vertex[j];
                if (m_Track.Upstream_Vertex == m_Vertex)
                {
                    SySal.TotalScan.VertexFit.TrackFitWithMomentum tf = new SySal.TotalScan.VertexFit.TrackFitWithMomentum();
                    int i;
                    tf.Field = 0;
                    tf.Id = new SySal.TotalScan.BaseTrackIndex(m_Track.Id);
                    tf.Sigma = 0;
                    tf.Weight = 1.0;
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
                    double p = 0.0;
                    try
                    {
                        p = m_Track.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("P"));
                    }
                    catch (Exception) {}
                    tf.PLikelihood = new NumericalTools.OneParamLogLikelihood(p, p, new double[1] { 0 }, "P");
                    AddTrackFit(tf);
                }
                else
                {
                    SySal.TotalScan.VertexFit.TrackFit tf = new SySal.TotalScan.VertexFit.TrackFit();
                    int i;
                    tf.Field = 0;
                    tf.Id = new SySal.TotalScan.BaseTrackIndex(m_Track.Id);
                    tf.Sigma = 0;
                    tf.Weight = 1.0;
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
            }
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

        private void RefreshFitListButton_Click(object sender, EventArgs e)        
        {
            ListFits.BeginUpdate();
            ListFits.Items.Clear();
            foreach (VertexFitForm vff in VertexFitForm.AvailableBrowsers)
                ListFits.Items.Add(vff.FitName);
            ListFits.EndUpdate();
        }

        double m_ExtrapDistance = 3900.0;

        private void OnExtrapDistanceLeave(object sender, EventArgs e)
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

        internal void RefreshAttributeList()
        {
            AttributeList.BeginUpdate();
            AttributeList.Items.Clear();
            SySal.TotalScan.Attribute[] attrlist = m_Vertex.ListAttributes();
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
        public static SySal.TotalScan.NamedAttributeIndex FBIsPrimaryIndex = new SySal.TotalScan.NamedAttributeIndex("PRIMARY");
        public static SySal.TotalScan.NamedAttributeIndex FBIsCharmIndex = new SySal.TotalScan.NamedAttributeIndex("CHARM");
        public static SySal.TotalScan.NamedAttributeIndex FBIsTauIndex = new SySal.TotalScan.NamedAttributeIndex("TAU");
        public static SySal.TotalScan.NamedAttributeIndex FBIsDeadMaterialIndex = new SySal.TotalScan.NamedAttributeIndex("OUTOFBRICK");

        bool m_IsRefreshing = false;

        private void RefreshFeedbackStatus()
        {
            m_IsRefreshing = true;
            try
            {
                if (System.Convert.ToInt64(m_Vertex.GetAttribute(FBEventIndex)) >= 0) chkFBEvent.Checked = true;
            }
            catch (Exception) { chkFBEvent.Checked = false; }
            try
            {
                chkFBPrimary.Checked = (System.Convert.ToInt32(m_Vertex.GetAttribute(FBIsPrimaryIndex)) > 0);
            }
            catch (Exception) { chkFBPrimary.Checked = false; }
            try
            {
                chkFBCharm.Checked = (System.Convert.ToInt32(m_Vertex.GetAttribute(FBIsCharmIndex)) > 0);
            }
            catch (Exception) { chkFBCharm.Checked = false; }
            try
            {
                chkFBTauDecay.Checked = (System.Convert.ToInt32(m_Vertex.GetAttribute(FBIsTauIndex)) > 0);
            }
            catch (Exception) { chkFBTauDecay.Checked = false; }
            try
            {
                chkFBDeadMaterial.Checked = (System.Convert.ToInt32(m_Vertex.GetAttribute(FBIsDeadMaterialIndex)) > 0);
            }
            catch (Exception) { chkFBDeadMaterial.Checked = false; }
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
                m_Vertex.SetAttribute(new SySal.TotalScan.NamedAttributeIndex(txtAttrName.Text.Trim()),
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
                    m_Vertex.RemoveAttribute((SySal.TotalScan.Index)lvi.Tag);
            RefreshAttributeList();
        }

        private void btnDownTracks_Click(object sender, EventArgs e)
        {
            m_Opening = 2.0;
            m_Radius = -1.0;
            m_DeltaSlope = 0.05;
            m_DeltaZ = 100000.0;
            chkPartner.Checked = false;
            radioDownstreamDir.Checked = true;
            radioUpstreamDir.Checked = false;
            RefreshParameters();
            btnShowRelatedTracks_Click(sender, e);
        }

        private void btnUpTracks_Click(object sender, EventArgs e)
        {
            m_Opening = 2.0;
            m_Radius = -1.0;
            m_DeltaSlope = 0.05;
            m_DeltaZ = 100000.0;
            chkPartner.Checked = false;
            radioDownstreamDir.Checked = false;
            radioUpstreamDir.Checked = true;
            RefreshParameters();
            btnShowRelatedTracks_Click(sender, e);
        }

        private void btnDownDecays_Click(object sender, EventArgs e)
        {
            m_Opening = -1.0;
            m_Radius = 1000.0;
            m_DeltaSlope = -1.0;
            m_DeltaZ = 6500.0;            
            chkPartner.Checked = false;
            radioDownstreamDir.Checked = true;
            radioUpstreamDir.Checked = false;
            RefreshParameters();
            btnShowRelatedTracks_Click(sender, e);
        }

        private void btnUpDecays_Click(object sender, EventArgs e)
        {
            m_Opening = -1.0;
            m_Radius = 1000.0;
            m_DeltaSlope = -1.0;
            m_DeltaZ = 6500.0;
            chkPartner.Checked = false;
            radioDownstreamDir.Checked = false;
            radioUpstreamDir.Checked = true;
            RefreshParameters();
            btnShowRelatedTracks_Click(sender, e);
        }

        private void OnFBEventChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBEvent.Checked)
            {
                xAddToFeedback((SySal.TotalScan.Flexi.Vertex)m_Vertex, m_Event);
                MessageBox.Show(((m_Vertex.Length == 1) ? "1 track" : (m_Vertex.Length + " tracks")) + " included with this vertex.", "Flag extension", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else m_Vertex.RemoveAttribute(FBEventIndex);
            RefreshAttributeList();
        }

        static internal void xAddToFeedback(SySal.TotalScan.Flexi.Vertex v, long eventid)
        {
            v.SetAttribute(FBEventIndex, System.Convert.ToDouble(eventid));
            int i;
            for (i = 0; i < v.Length; i++)
            {
                v[i].SetAttribute(FBEventIndex, System.Convert.ToDouble(eventid));
                foreach (TrackBrowser tbw in TrackBrowser.AvailableBrowsers)
                    if (tbw.Track == v[i])
                    {
                        tbw.RefreshAttributeList();
                        break;
                    }
            }
        }

        private void OnFBPrimaryChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBPrimary.Checked) m_Vertex.SetAttribute(FBIsPrimaryIndex, 1.0);
            else m_Vertex.RemoveAttribute(FBIsPrimaryIndex);
            RefreshAttributeList();
        }

        private void OnFBCharmChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBCharm.Checked) m_Vertex.SetAttribute(FBIsCharmIndex, 1.0);
            else m_Vertex.RemoveAttribute(FBIsCharmIndex);
            RefreshAttributeList();
        }

        private void OnFBTauChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBTauDecay.Checked) m_Vertex.SetAttribute(FBIsTauIndex, 1.0);
            else m_Vertex.RemoveAttribute(FBIsTauIndex);
            RefreshAttributeList();
        }

        private void OnFBDeadMaterialChecked(object sender, EventArgs e)
        {
            if (m_IsRefreshing) return;
            if (chkFBDeadMaterial.Checked) m_Vertex.SetAttribute(FBIsDeadMaterialIndex, 1.0);
            else m_Vertex.RemoveAttribute(FBIsDeadMaterialIndex);
            RefreshAttributeList();
        }

        private void btnDownPartners_Click(object sender, EventArgs e)
        {
            m_Opening = -1.0;
            m_Radius = 1000.0;
            m_DeltaSlope = -1.0;
            m_DeltaZ = 6500.0;
            chkPartner.Checked = true;
            radioDownstreamDir.Checked = true;
            radioUpstreamDir.Checked = false;
            RefreshParameters();
            btnShowRelatedTracks_Click(sender, e);
        }

        private void btnUpPartners_Click(object sender, EventArgs e)
        {
            m_Opening = -1.0;
            m_Radius = 1000.0;
            m_DeltaSlope = -1.0;
            m_DeltaZ = 6500.0;
            chkPartner.Checked = true;
            radioDownstreamDir.Checked = false;
            radioUpstreamDir.Checked = true;
            RefreshParameters();
            btnShowRelatedTracks_Click(sender, e);
        }

        private void KillButton_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Are you sure you want to kill this vertex?\r\nThis operation cannot be undone.", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                int i;
                for (i = 0; i < m_Vertex.Length; i++)
                {
                    SySal.TotalScan.Track tk = m_Vertex[i];
                    if (tk.Upstream_Vertex == m_Vertex) tk.SetUpstreamVertex(null);
                    if (tk.Downstream_Vertex == m_Vertex) tk.SetDownstreamVertex(null);
                    foreach (TrackBrowser tb in TrackBrowser.AvailableBrowsers)
                        if (tb.Track == tk)
                            tb.Track = tk;
                }
                ((SySal.TotalScan.Flexi.Volume.VertexList)m_V.Vertices).Remove(new int[1] { m_Vertex.Id });
                SelectForGraph.Remove(m_Vertex);
                TrackBrowser.RefreshAll();
                Close();
                
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

        private uint m_DownSegs = 1;

        private void OnDownSegsLeave(object sender, EventArgs e)
        {
            try
            {
                m_DownSegs = Convert.ToUInt32(DownSegsText.Text);
            }
            catch (Exception)
            {
                DownSegsText.Text = m_DownSegs.ToString();
                DownSegsText.Focus();
            }
        }

        private uint m_UpSegs = 1;

        private void OnUpSegsLeave(object sender, EventArgs e)
        {
            try
            {
                m_UpSegs = Convert.ToUInt32(UpSegsText.Text);
            }
            catch (Exception)
            {
                UpSegsText.Text = m_UpSegs.ToString();
                UpSegsText.Focus();
            }
        }

        private uint m_DownStops = 1;

        private void OnDownStopsLeave(object sender, EventArgs e)
        {
            try
            {
                m_DownStops = Convert.ToUInt32(DownStopsText.Text);
            }
            catch (Exception)
            {
                DownStopsText.Text = m_DownStops.ToString();
                DownStopsText.Focus();
            }
        }

        private uint m_UpStops = 1;

        private void OnUpStopsLeave(object sender, EventArgs e)
        {
            try
            {
                m_UpStops = Convert.ToUInt32(UpStopsText.Text);
            }
            catch (Exception)
            {
                UpStopsText.Text = m_UpStops.ToString();
                UpStopsText.Focus();
            } 
        }

        private void AppendToButton_Click(object sender, EventArgs e)
        {
            try
            {
                int i, lydown, lyup, lyid, j;
                SySal.BasicTypes.Vector2 s = new SySal.BasicTypes.Vector2();
                SySal.BasicTypes.Vector sl = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector vl = new SySal.BasicTypes.Vector();
                for (i = 0; i < m_Vertex.Length + (ConnToVtxCheck.Checked ? 1 : 0); i++)
                {
                    SySal.TotalScan.Track tk = (i < m_Vertex.Length) ? m_Vertex[i] : null;
                    if (tk == null)
                    {
                        tk = new SySal.TotalScan.Track(-1);
                        SySal.TotalScan.Vertex ovx = null;
                        try
                        {
                            ovx = m_V.Vertices[Convert.ToInt32(ConnToVtxText.Text)];
                            if (ovx == m_Vertex) continue;
                            if (m_Vertex.Z > ovx.Z)
                            {
                                v.X = m_Vertex.X;
                                v.Y = m_Vertex.Y;
                                v.Z = m_Vertex.Z;
                                vl.X = ovx.X;
                                vl.Y = ovx.Y;
                                vl.Z = ovx.Z;
                            }
                            else
                            {
                                vl.X = m_Vertex.X;
                                vl.Y = m_Vertex.Y;
                                vl.Z = m_Vertex.Z;
                                v.X = ovx.X;
                                v.Y = ovx.Y;
                                v.Z = ovx.Z;
                            }
                            if (v.Z - vl.Z <= 0.0) continue;
                            s.X = (v.X - vl.X) / (v.Z - vl.Z);
                            s.Y = (v.Y - vl.Y) / (v.Z - vl.Z);
                            for (lydown = 0; lydown < m_Layers.Length && m_Layers[lydown].RefCenter.Z > v.Z; lydown++) ;
                            if (lydown == m_Layers.Length) continue;
                            for (lyup = lydown; lyup < m_Layers.Length && m_Layers[lyup].RefCenter.Z >= vl.Z; lyup++) ;
                            if (--lyup < lydown) continue;                            
                        }
                        catch (Exception) { continue;  };
                    }
                    else if (tk.Upstream_Vertex == m_Vertex)
                    {
                        v = tk[tk.Length - 1].Info.Intercept;
                        if (Math.Abs(m_Vertex.Z - v.Z) < 1.0)
                        {
                            s.X = tk.Upstream_SlopeX;
                            s.Y = tk.Upstream_SlopeY;
                        }
                        else
                        {
                            s.X = (v.X - m_Vertex.X) / (v.Z - m_Vertex.Z);
                            s.Y = (v.Y - m_Vertex.Y) / (v.Z - m_Vertex.Z);
                        }
                        lyid = tk[tk.Length - 1].LayerOwner.Id;
                        lydown = Math.Max(0, lyid - (int)m_DownSegs);
                        for (j = lyid; j < m_Layers.Length && m_Vertex.Z < m_Layers[j].RefCenter.Z; j++) ;
                        lyup = Math.Min(j + (int)m_DownStops, m_V.Layers.Length - 1);
                    }
                    else
                    {
                        v = tk[0].Info.Intercept;
                        if (Math.Abs(m_Vertex.Z - v.Z) < 1.0)
                        {
                            s.X = tk.Downstream_SlopeX;
                            s.Y = tk.Downstream_SlopeY;
                        }
                        else
                        {
                            s.X = (v.X - m_Vertex.X) / (v.Z - m_Vertex.Z);
                            s.Y = (v.Y - m_Vertex.Y) / (v.Z - m_Vertex.Z);
                        }
                        lyid = tk[0].LayerOwner.Id;
                        lyup = Math.Min(m_Layers.Length, lyid + (int)m_UpSegs);
                        for (j = lyup; j >= 0 && m_Vertex.Z > m_Layers[j].RefCenter.Z; j--) ;
                        lydown = Math.Max(j - (int)m_UpStops, 0);
                    }
                    for (lyid = lydown; lyid <= lyup; lyid++)
                    {
                        if (System.IO.File.Exists(AppendToFileText.Text) == false)
                            System.IO.File.WriteAllText(AppendToFileText.Text, "IDTRACK\tPOSX\tPOSY\tSLOPEX\tSLOPEY\tPLATE");
                        SySal.TotalScan.Layer l = m_Layers[lyid];
                        vl.Z = l.RefCenter.Z;
                        vl.X = m_Vertex.X + (vl.Z - m_Vertex.Z) * s.X;
                        vl.Y = m_Vertex.Y + (vl.Z - m_Vertex.Z) * s.Y;
                        vl = l.ToOriginalPoint(vl);
                        sl.Z = 1.0;
                        sl.X = s.X;
                        sl.Y = s.Y;
                        sl = l.ToOriginalSlope(sl);
                        System.IO.File.AppendAllText(AppendToFileText.Text, "\r\n" + tk.Id +
                            "\t" + vl.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) +
                            "\t" + vl.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) +
                            "\t" + sl.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) +
                            "\t" + sl.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) +
                            "\t" + l.SheetId);
                    }
                }
                MessageBox.Show("Data saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                DisplayForm.DefaultProjFileName = AppendToFileText.Text;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void LaunchScanButton_Click(object sender, EventArgs e)
        {
            EnqueueOpForm eqof = new EnqueueOpForm();
            int bestid;
            double bestz = 1.0 + m_Layers[0].RefCenter.Z;
            eqof.BrickId = ((SySal.TotalScan.Flexi.Vertex)m_Vertex).DataSet.DataId;
            eqof.VolumeStartsFromTrack = false;
            eqof.VolStart.Id = m_Vertex.Id;            
            int id = 0;
            int i;            
            SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
            v.X = m_Vertex.X;
            v.Y = m_Vertex.Y;
            v.Z = m_Vertex.Z;
            for (i = 0; i < m_Layers.Length; i++)
                if (m_Layers[i].UpstreamZ > v.Z)
                    if (m_Layers[i].UpstreamZ < bestz)
                    {
                        id = i;
                        bestz = m_Layers[i].UpstreamZ;
                    }
            SySal.TotalScan.Layer l = m_Layers[id];
            eqof.VolStart.Plate = l.SheetId;
            v = l.ToOriginalPoint(v);
            eqof.VolStart.Position.X = v.X;
            eqof.VolStart.Position.Y = v.Y;
            eqof.VolStart.Slope.X = 0.0;
            eqof.VolStart.Slope.Y = 0.0;
            try
            {
                eqof.ShowDialog();
            }
            catch (Exception) { }

        }
    }
}
