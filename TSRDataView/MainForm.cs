using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.TSRDataView
{
	/// <summary>
	/// TSRDataView is an application to browse TSR files and performing common searches/computations.    
	/// </summary>
    /// <remarks>
    /// After opening one or more TSR file(s) (<c>Load</c> button), data can be searched for the following objects:
    /// <list type="table">
    /// <item><term>Layer</term><description>A layer is searched for on the basis of its ID/SheetID</description></item>
    /// <item><term>Segment</term><description>A segment is searched for on the basis of its ID, the ID of its layer (or the corresponding SheetID), the ID of its track and its position/slope.</description></item>
    /// <item><term>Track</term><description>A track is searched for on the basis of its ID, the Up/Downstream vertex ID, and its position/slope (projected to a certain layer, also transforming its geometry back to the original reference of the TLG file).</description></item>
    /// <item><term>Vertex</term><description>A vertex is searched for on the basis of its ID and its position.</description></item>
    /// <item><term>Volume</term><description>Useful when multiple TSR files are opened to search for a specific volume.</description></item>
    /// </list>
    /// The meaning of the search field changes slightly depending on the search being performed. The ID is always the sequential identifier of the object, so it means a layer ordinal number for layers, a sequential
    /// number within the layer for segments, the ordinal number of the track, and so on. The result of the search has a number of columns that depends on the object searched.
    /// <para>
    /// <b>Layer</b>:
    /// <list type="table">
    /// <item><term>ID</term><description>The ID of the layer (sequential number, 0 for the most downstream one).</description></item>
    /// <item><term>SheetID</term><description>The ID of the sheet.</description></item>
    /// <item><term>RefZ</term><description>The reference Z for the layer.</description></item>
    /// <item><term>DownZ</term><description>The downstream Z for the layer.</description></item>
    /// <item><term>UpZ</term><description>The upstream Z for the layer.</description></item>
    /// <item><term>Segments</term><description>The number of segments contained.</description></item>
    /// <item><term>MXX,MXY,MYX,MYY</term><description>The XX,XY,YX,YY component of the transformation matrix.</description></item>
    /// <item><term>TX,TY,TZ</term><description>The X,Y,Z component of the translation vector.</description></item>
    /// <item><term>MSX,MSY</term><description>The multiplication constant for the X,Y component of slopes.</description></item>
    /// <item><term>DSX,DSY</term><description>The additive constant for the X,Y component of slopes.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Segment</b>:
    /// <list type="table">
    /// <item><term>LyrID</term><description>The ID of the layer of the segment (sequential number, 0 for the most downstream one).</description></item>
    /// <item><term>SheetID</term><description>The ID of the sheet containing the segment.</description></item>
    /// <item><term>TkID</term><description>The ID of the track containing the segment.</description></item>
    /// <item><term>ID</term><description>The ID of the segment within its layer.</description></item>
    /// <item><term>BaseID</term><description>The original ID of the base track in the TLG file (if applicable).</description></item>
    /// <item><term>N</term><description>The number of grains of the segment.</description></item>
    /// <item><term>PX,PY</term><description>The (aligned) X,Y component of the position.</description></item>
    /// <item><term>SX,SY</term><description>The (aligned) X,Y component of the slope.</description></item>
    /// <item><term>S</term><description>The <c>Sigma</c> parameter for segment (basetrack/microtrack) quality.</description></item>
    /// <item><term>OPX,OPY</term><description>The original X,Y component of the position.</description></item>
    /// <item><term>OSX,OSY</term><description>The original X,Y component of the slope.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Track</b>:
    /// <list type="table">
    /// <item><term>ID</term><description>The ID of the track.</description></item>
    /// <item><term>N</term><description>The number of segments in the track.</description></item>
    /// <item><term>DwVID/UpVID</term><description>The ID of the downstream/upstream vertex.</description></item>
    /// <item><term>DwIP/UpIP</term><description>The IP of the track w.r.t. the downstream/upstream vertex.</description></item>
    /// <item><term>DwZX,DwZY,UpZX,UpZY</term><description>The X,Y component of the track position, taken from the downstream/upstream end, computed at <c>Z = 0</c>.</description></item>
    /// <item><term>DwX,DwY,DwZ,UpX,UpY,UpZ</term><description>The X,Y,Z component of the track position, taken from the downstream/upstream end.</description></item>
    /// <item><term>DwSX,DwSY,UpSX,UpSY</term><description>The X,Y component of the slope, taken from the downstream/upstream end.</description></item>
    /// <item><term>APX,APY</term><description>The X,Y component of the position, computed by interpolation/extrapolation at the specified layer number (Layer ID apply, not Sheet IDs).</description></item>
    /// <item><term>ASX,ASY</term><description>The X,Y component of the slope, computed by interpolation/extrapolation at the specified layer number (Layer ID apply, not Sheet IDs).</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Vertex</b>:
    /// <list type="table">
    /// <item><term>ID</term><description>The ID of the vertex.</description></item>
    /// <item><term>N</term><description>The number of tracks attached to the vertex.</description></item>
    /// <item><term>AvgD</term><description>Average distance of tracks at the vertex point.</description></item>    
    /// <item><term>X,Y,Z</term><description>The X,Y,Z component of the vertex position.</description></item>
    /// <item><term>DownID</term><description>The ID of the layer immediately downstream of this vertex.</description></item>
    /// <item><term>DownSheetID</term><description>The Sheet ID of the layer immediately downstream of this vertex.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
	public class TSRDataViewForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.RadioButton radioLayers;
		private System.Windows.Forms.RadioButton radioSegments;
		private System.Windows.Forms.RadioButton radioTracks;
		private System.Windows.Forms.RadioButton radioVertices;
		private System.Windows.Forms.Button buttonChooseFile;
		private System.Windows.Forms.TextBox textFilePath;
		private System.Windows.Forms.Button buttonLoad;
		private System.Windows.Forms.RadioButton radioVolume;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox textID;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox textMinX;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox textMaxX;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox textMaxY;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox textMinY;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.TextBox textMaxSY;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TextBox textMinSY;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.TextBox textMaxSX;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.TextBox textMinSX;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.Button buttonSearchAndShow;
		private System.Windows.Forms.TextBox textBaseID;
		private System.Windows.Forms.ListView listResults;
		private System.Windows.Forms.TextBox textLayerID;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.TextBox textSheetID;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.TextBox textTrackID;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.TextBox textUpVID;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.TextBox textDownVID;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.Button buttonSave;
		private System.Windows.Forms.TextBox textProjectToLayer;
		private System.Windows.Forms.Label label16;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public TSRDataViewForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
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
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.radioLayers = new System.Windows.Forms.RadioButton();
			this.radioSegments = new System.Windows.Forms.RadioButton();
			this.radioTracks = new System.Windows.Forms.RadioButton();
			this.radioVertices = new System.Windows.Forms.RadioButton();
			this.radioVolume = new System.Windows.Forms.RadioButton();
			this.buttonChooseFile = new System.Windows.Forms.Button();
			this.textFilePath = new System.Windows.Forms.TextBox();
			this.buttonLoad = new System.Windows.Forms.Button();
			this.listResults = new System.Windows.Forms.ListView();
			this.label1 = new System.Windows.Forms.Label();
			this.textID = new System.Windows.Forms.TextBox();
			this.textBaseID = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.textMinX = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.textMaxX = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.textMaxY = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.textMinY = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.textMaxSY = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.textMinSY = new System.Windows.Forms.TextBox();
			this.label8 = new System.Windows.Forms.Label();
			this.textMaxSX = new System.Windows.Forms.TextBox();
			this.label9 = new System.Windows.Forms.Label();
			this.textMinSX = new System.Windows.Forms.TextBox();
			this.label10 = new System.Windows.Forms.Label();
			this.buttonSearchAndShow = new System.Windows.Forms.Button();
			this.textLayerID = new System.Windows.Forms.TextBox();
			this.label11 = new System.Windows.Forms.Label();
			this.textSheetID = new System.Windows.Forms.TextBox();
			this.label12 = new System.Windows.Forms.Label();
			this.textTrackID = new System.Windows.Forms.TextBox();
			this.label13 = new System.Windows.Forms.Label();
			this.textUpVID = new System.Windows.Forms.TextBox();
			this.label14 = new System.Windows.Forms.Label();
			this.textDownVID = new System.Windows.Forms.TextBox();
			this.label15 = new System.Windows.Forms.Label();
			this.buttonSave = new System.Windows.Forms.Button();
			this.textProjectToLayer = new System.Windows.Forms.TextBox();
			this.label16 = new System.Windows.Forms.Label();
			this.groupBox1.SuspendLayout();
			this.SuspendLayout();
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.radioLayers,
																					this.radioSegments,
																					this.radioTracks,
																					this.radioVertices,
																					this.radioVolume});
			this.groupBox1.Location = new System.Drawing.Point(8, 8);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(128, 152);
			this.groupBox1.TabIndex = 0;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Object type";
			// 
			// radioLayers
			// 
			this.radioLayers.Location = new System.Drawing.Point(8, 24);
			this.radioLayers.Name = "radioLayers";
			this.radioLayers.Size = new System.Drawing.Size(112, 24);
			this.radioLayers.TabIndex = 1;
			this.radioLayers.Text = "Layers";
			// 
			// radioSegments
			// 
			this.radioSegments.Location = new System.Drawing.Point(8, 48);
			this.radioSegments.Name = "radioSegments";
			this.radioSegments.Size = new System.Drawing.Size(112, 24);
			this.radioSegments.TabIndex = 2;
			this.radioSegments.Text = "Segments";
			// 
			// radioTracks
			// 
			this.radioTracks.Location = new System.Drawing.Point(8, 72);
			this.radioTracks.Name = "radioTracks";
			this.radioTracks.Size = new System.Drawing.Size(112, 24);
			this.radioTracks.TabIndex = 3;
			this.radioTracks.Text = "Tracks";
			// 
			// radioVertices
			// 
			this.radioVertices.Location = new System.Drawing.Point(8, 96);
			this.radioVertices.Name = "radioVertices";
			this.radioVertices.Size = new System.Drawing.Size(112, 24);
			this.radioVertices.TabIndex = 4;
			this.radioVertices.Text = "Vertices";
			// 
			// radioVolume
			// 
			this.radioVolume.Location = new System.Drawing.Point(8, 120);
			this.radioVolume.Name = "radioVolume";
			this.radioVolume.Size = new System.Drawing.Size(112, 24);
			this.radioVolume.TabIndex = 5;
			this.radioVolume.Text = "Volume";
			// 
			// buttonChooseFile
			// 
			this.buttonChooseFile.Location = new System.Drawing.Point(144, 16);
			this.buttonChooseFile.Name = "buttonChooseFile";
			this.buttonChooseFile.Size = new System.Drawing.Size(96, 24);
			this.buttonChooseFile.TabIndex = 6;
			this.buttonChooseFile.Text = "Choose file";
			this.buttonChooseFile.Click += new System.EventHandler(this.buttonChooseFile_Click);
			// 
			// textFilePath
			// 
			this.textFilePath.Location = new System.Drawing.Point(248, 16);
			this.textFilePath.Name = "textFilePath";
			this.textFilePath.Size = new System.Drawing.Size(448, 20);
			this.textFilePath.TabIndex = 7;
			this.textFilePath.Text = "";
			// 
			// buttonLoad
			// 
			this.buttonLoad.Location = new System.Drawing.Point(704, 16);
			this.buttonLoad.Name = "buttonLoad";
			this.buttonLoad.Size = new System.Drawing.Size(88, 24);
			this.buttonLoad.TabIndex = 8;
			this.buttonLoad.Text = "Load";
			this.buttonLoad.Click += new System.EventHandler(this.buttonLoad_Click);
			// 
			// listResults
			// 
			this.listResults.FullRowSelect = true;
			this.listResults.GridLines = true;
			this.listResults.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
			this.listResults.Location = new System.Drawing.Point(8, 184);
			this.listResults.Name = "listResults";
			this.listResults.Size = new System.Drawing.Size(784, 344);
			this.listResults.TabIndex = 9;
			this.listResults.View = System.Windows.Forms.View.Details;
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(144, 48);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(32, 23);
			this.label1.TabIndex = 10;
			this.label1.Text = "ID=";
			// 
			// textID
			// 
			this.textID.Location = new System.Drawing.Point(208, 48);
			this.textID.Name = "textID";
			this.textID.Size = new System.Drawing.Size(72, 20);
			this.textID.TabIndex = 11;
			this.textID.Text = "";
			this.textID.Leave += new System.EventHandler(this.textIDLeave);
			// 
			// textBaseID
			// 
			this.textBaseID.Location = new System.Drawing.Point(352, 48);
			this.textBaseID.Name = "textBaseID";
			this.textBaseID.Size = new System.Drawing.Size(72, 20);
			this.textBaseID.TabIndex = 13;
			this.textBaseID.Text = "";
			this.textBaseID.Leave += new System.EventHandler(this.textBaseIDLeave);
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(288, 48);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(56, 23);
			this.label2.TabIndex = 12;
			this.label2.Text = "Base ID=";
			// 
			// textMinX
			// 
			this.textMinX.Location = new System.Drawing.Point(208, 96);
			this.textMinX.Name = "textMinX";
			this.textMinX.Size = new System.Drawing.Size(72, 20);
			this.textMinX.TabIndex = 15;
			this.textMinX.Text = "";
			this.textMinX.Leave += new System.EventHandler(this.textMinXLeave);
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(144, 96);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(56, 23);
			this.label3.TabIndex = 14;
			this.label3.Text = "MinX=";
			// 
			// textMaxX
			// 
			this.textMaxX.Location = new System.Drawing.Point(352, 96);
			this.textMaxX.Name = "textMaxX";
			this.textMaxX.Size = new System.Drawing.Size(72, 20);
			this.textMaxX.TabIndex = 17;
			this.textMaxX.Text = "";
			this.textMaxX.Leave += new System.EventHandler(this.textMaxXLeave);
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(288, 96);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(56, 23);
			this.label4.TabIndex = 16;
			this.label4.Text = "MaxX=";
			// 
			// textMaxY
			// 
			this.textMaxY.Location = new System.Drawing.Point(640, 96);
			this.textMaxY.Name = "textMaxY";
			this.textMaxY.Size = new System.Drawing.Size(72, 20);
			this.textMaxY.TabIndex = 21;
			this.textMaxY.Text = "";
			this.textMaxY.Leave += new System.EventHandler(this.textMaxYLeave);
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(576, 96);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(56, 23);
			this.label5.TabIndex = 20;
			this.label5.Text = "MaxY=";
			// 
			// textMinY
			// 
			this.textMinY.Location = new System.Drawing.Point(496, 96);
			this.textMinY.Name = "textMinY";
			this.textMinY.Size = new System.Drawing.Size(72, 20);
			this.textMinY.TabIndex = 19;
			this.textMinY.Text = "";
			this.textMinY.Leave += new System.EventHandler(this.textMinYLeave);
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(432, 96);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(56, 23);
			this.label6.TabIndex = 18;
			this.label6.Text = "MinY=";
			// 
			// textMaxSY
			// 
			this.textMaxSY.Location = new System.Drawing.Point(640, 120);
			this.textMaxSY.Name = "textMaxSY";
			this.textMaxSY.Size = new System.Drawing.Size(72, 20);
			this.textMaxSY.TabIndex = 29;
			this.textMaxSY.Text = "";
			this.textMaxSY.Leave += new System.EventHandler(this.textMaxSYLeave);
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(576, 120);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(56, 23);
			this.label7.TabIndex = 28;
			this.label7.Text = "MaxSY=";
			// 
			// textMinSY
			// 
			this.textMinSY.Location = new System.Drawing.Point(496, 120);
			this.textMinSY.Name = "textMinSY";
			this.textMinSY.Size = new System.Drawing.Size(72, 20);
			this.textMinSY.TabIndex = 27;
			this.textMinSY.Text = "";
			this.textMinSY.Leave += new System.EventHandler(this.textMinSYLeave);
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(432, 120);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(56, 23);
			this.label8.TabIndex = 26;
			this.label8.Text = "MinSY=";
			// 
			// textMaxSX
			// 
			this.textMaxSX.Location = new System.Drawing.Point(352, 120);
			this.textMaxSX.Name = "textMaxSX";
			this.textMaxSX.Size = new System.Drawing.Size(72, 20);
			this.textMaxSX.TabIndex = 25;
			this.textMaxSX.Text = "";
			this.textMaxSX.Leave += new System.EventHandler(this.textMaxSXLeave);
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(288, 120);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(56, 23);
			this.label9.TabIndex = 24;
			this.label9.Text = "MaxSX=";
			// 
			// textMinSX
			// 
			this.textMinSX.Location = new System.Drawing.Point(208, 120);
			this.textMinSX.Name = "textMinSX";
			this.textMinSX.Size = new System.Drawing.Size(72, 20);
			this.textMinSX.TabIndex = 23;
			this.textMinSX.Text = "";
			this.textMinSX.Leave += new System.EventHandler(this.textMinSXLeave);
			// 
			// label10
			// 
			this.label10.Location = new System.Drawing.Point(144, 120);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(56, 23);
			this.label10.TabIndex = 22;
			this.label10.Text = "MinSX=";
			// 
			// buttonSearchAndShow
			// 
			this.buttonSearchAndShow.Location = new System.Drawing.Point(144, 152);
			this.buttonSearchAndShow.Name = "buttonSearchAndShow";
			this.buttonSearchAndShow.Size = new System.Drawing.Size(152, 24);
			this.buttonSearchAndShow.TabIndex = 30;
			this.buttonSearchAndShow.Text = "Search and show";
			this.buttonSearchAndShow.Click += new System.EventHandler(this.buttonSearchAndShow_Click);
			// 
			// textLayerID
			// 
			this.textLayerID.Location = new System.Drawing.Point(496, 48);
			this.textLayerID.Name = "textLayerID";
			this.textLayerID.Size = new System.Drawing.Size(72, 20);
			this.textLayerID.TabIndex = 32;
			this.textLayerID.Text = "";
			this.textLayerID.Leave += new System.EventHandler(this.textLayerIDLeave);
			// 
			// label11
			// 
			this.label11.Location = new System.Drawing.Point(432, 48);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(56, 23);
			this.label11.TabIndex = 31;
			this.label11.Text = "Lyr ID=";
			// 
			// textSheetID
			// 
			this.textSheetID.Location = new System.Drawing.Point(640, 48);
			this.textSheetID.Name = "textSheetID";
			this.textSheetID.Size = new System.Drawing.Size(72, 20);
			this.textSheetID.TabIndex = 34;
			this.textSheetID.Text = "";
			this.textSheetID.Leave += new System.EventHandler(this.textSheetIDLeave);
			// 
			// label12
			// 
			this.label12.Location = new System.Drawing.Point(576, 48);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(56, 23);
			this.label12.TabIndex = 33;
			this.label12.Text = "Sheet ID=";
			// 
			// textTrackID
			// 
			this.textTrackID.Location = new System.Drawing.Point(208, 72);
			this.textTrackID.Name = "textTrackID";
			this.textTrackID.Size = new System.Drawing.Size(72, 20);
			this.textTrackID.TabIndex = 36;
			this.textTrackID.Text = "";
			this.textTrackID.Leave += new System.EventHandler(this.textTrackIDLeave);
			// 
			// label13
			// 
			this.label13.Location = new System.Drawing.Point(144, 72);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(56, 23);
			this.label13.TabIndex = 35;
			this.label13.Text = "Track ID=";
			// 
			// textUpVID
			// 
			this.textUpVID.Location = new System.Drawing.Point(352, 72);
			this.textUpVID.Name = "textUpVID";
			this.textUpVID.Size = new System.Drawing.Size(72, 20);
			this.textUpVID.TabIndex = 38;
			this.textUpVID.Text = "";
			this.textUpVID.Leave += new System.EventHandler(this.textUpVIDLeave);
			// 
			// label14
			// 
			this.label14.Location = new System.Drawing.Point(288, 72);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(56, 23);
			this.label14.TabIndex = 37;
			this.label14.Text = "UpV ID=";
			// 
			// textDownVID
			// 
			this.textDownVID.Location = new System.Drawing.Point(496, 72);
			this.textDownVID.Name = "textDownVID";
			this.textDownVID.Size = new System.Drawing.Size(72, 20);
			this.textDownVID.TabIndex = 40;
			this.textDownVID.Text = "";
			this.textDownVID.Leave += new System.EventHandler(this.textDownVIDLeave);
			// 
			// label15
			// 
			this.label15.Location = new System.Drawing.Point(432, 72);
			this.label15.Name = "label15";
			this.label15.Size = new System.Drawing.Size(56, 23);
			this.label15.TabIndex = 39;
			this.label15.Text = "DwV ID=";
			// 
			// buttonSave
			// 
			this.buttonSave.Location = new System.Drawing.Point(304, 152);
			this.buttonSave.Name = "buttonSave";
			this.buttonSave.Size = new System.Drawing.Size(152, 24);
			this.buttonSave.TabIndex = 41;
			this.buttonSave.Text = "Save to file";
			this.buttonSave.Click += new System.EventHandler(this.buttonSave_Click);
			// 
			// textProjectToLayer
			// 
			this.textProjectToLayer.Location = new System.Drawing.Point(640, 152);
			this.textProjectToLayer.Name = "textProjectToLayer";
			this.textProjectToLayer.Size = new System.Drawing.Size(72, 20);
			this.textProjectToLayer.TabIndex = 43;
			this.textProjectToLayer.Text = "";
			this.textProjectToLayer.Leave += new System.EventHandler(this.textProjectToLayerLeave);
			// 
			// label16
			// 
			this.label16.Location = new System.Drawing.Point(528, 152);
			this.label16.Name = "label16";
			this.label16.Size = new System.Drawing.Size(104, 23);
			this.label16.TabIndex = 42;
			this.label16.Text = "Project to layer";
			this.label16.TextAlign = System.Drawing.ContentAlignment.TopRight;
			// 
			// TSRDataViewForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(800, 534);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.textProjectToLayer,
																		  this.label16,
																		  this.buttonSave,
																		  this.textDownVID,
																		  this.label15,
																		  this.textUpVID,
																		  this.label14,
																		  this.textTrackID,
																		  this.label13,
																		  this.textSheetID,
																		  this.label12,
																		  this.textLayerID,
																		  this.label11,
																		  this.buttonSearchAndShow,
																		  this.textMaxSY,
																		  this.label7,
																		  this.textMinSY,
																		  this.label8,
																		  this.textMaxSX,
																		  this.label9,
																		  this.textMinSX,
																		  this.label10,
																		  this.textMaxY,
																		  this.label5,
																		  this.textMinY,
																		  this.label6,
																		  this.textMaxX,
																		  this.label4,
																		  this.textMinX,
																		  this.label3,
																		  this.textBaseID,
																		  this.label2,
																		  this.textID,
																		  this.label1,
																		  this.listResults,
																		  this.buttonLoad,
																		  this.textFilePath,
																		  this.buttonChooseFile,
																		  this.groupBox1});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.Name = "TSRDataViewForm";
			this.Text = "TSR Data View";
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
			Application.Run(new TSRDataViewForm());
		}

		private void buttonChooseFile_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog ofn = new OpenFileDialog();
			ofn.Filter = "TSR files (*.tsr)|*.tsr";
			if (ofn.ShowDialog() == DialogResult.OK)
			{
				textFilePath.Text = ofn.FileName;
			}
		}

		private SySal.TotalScan.Volume Vol = null;

		private void buttonLoad_Click(object sender, System.EventArgs e)
		{
			try
			{
				Vol = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(textFilePath.Text, typeof(SySal.TotalScan.Volume));
			}
			catch (System.Exception x)
			{
				MessageBox.Show(x.Message, "Loading error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private bool CheckID;
		private int vID;
		
		private bool CheckBaseID;
		private int vBaseID;

		private bool CheckLayerID;
		private int vLayerID;

		private bool CheckSheetID;
		private int vSheetID;

		private bool CheckTrackID;
		private int vTrackID;

		private bool CheckUpVID;
		private int vUpVID;

		private bool CheckDownVID;
		private int vDownVID;

		private bool CheckMinX;
		private double vMinX;

		private bool CheckMaxX;
		private double vMaxX;

		private bool CheckMinY;
		private double vMinY;

		private bool CheckMaxY;
		private double vMaxY;

		private bool CheckMinSX;
		private double vMinSX;

		private bool CheckMaxSX;
		private double vMaxSX;

		private bool CheckMinSY;
		private double vMinSY;

		private bool CheckMaxSY;
		private double vMaxSY;

		private bool CheckProjectToLayer;
		private int vProjectToLayer;

		private void textIDLeave(object sender, System.EventArgs e)
		{
			if (CheckID = (textID.Text.Length != 0))
				try
				{
					vID = Convert.ToInt32(textID.Text);
				}
				catch (Exception)
				{
					textID.Focus();
				}
		}

		private void textBaseIDLeave(object sender, System.EventArgs e)
		{
			if (CheckBaseID = (textBaseID.Text.Length != 0))
				try
				{
					vBaseID = Convert.ToInt32(textBaseID.Text);
				}
				catch (Exception)
				{
					textBaseID.Focus();
				}		
		}

		private void textLayerIDLeave(object sender, System.EventArgs e)
		{
			if (CheckLayerID = (textLayerID.Text.Length != 0))
				try
				{
					vLayerID = Convert.ToInt32(textLayerID.Text);
				}
				catch (Exception)
				{
					textLayerID.Focus();
				}		
		}

		private void textSheetIDLeave(object sender, System.EventArgs e)
		{
			if (CheckSheetID = (textSheetID.Text.Length != 0))
				try
				{
					vSheetID = Convert.ToInt32(textSheetID.Text);
				}
				catch (Exception)
				{
					textSheetID.Focus();
				}		
		}

		private void textTrackIDLeave(object sender, System.EventArgs e)
		{
			if (CheckTrackID = (textTrackID.Text.Length != 0))
				try
				{
					vTrackID = Convert.ToInt32(textTrackID.Text);
				}
				catch (Exception)
				{
					textTrackID.Focus();
				}				
		}

		private void textUpVIDLeave(object sender, System.EventArgs e)
		{
			if (CheckUpVID = (textUpVID.Text.Length != 0))
				try
				{
					vUpVID = Convert.ToInt32(textUpVID.Text);
				}
				catch (Exception)
				{
					textUpVID.Focus();
				}						
		}

		private void textDownVIDLeave(object sender, System.EventArgs e)
		{
			if (CheckDownVID = (textDownVID.Text.Length != 0))
				try
				{
					vDownVID = Convert.ToInt32(textDownVID.Text);
				}
				catch (Exception)
				{
					textDownVID.Focus();
				}								
		}

		private void textMinXLeave(object sender, System.EventArgs e)
		{
			if (CheckMinX = (textMinX.Text.Length != 0))
				try
				{
					vMinX = Convert.ToDouble(textMinX.Text);
				}
				catch (Exception)
				{
					textMinX.Focus();
				}				
		}

		private void textMaxXLeave(object sender, System.EventArgs e)
		{
			if (CheckMaxX = (textMaxX.Text.Length != 0))
				try
				{
					vMaxX = Convert.ToDouble(textMaxX.Text);
				}
				catch (Exception)
				{
					textMaxX.Focus();
				}						
		}

		private void textMinYLeave(object sender, System.EventArgs e)
		{
			if (CheckMinY = (textMinY.Text.Length != 0))
				try
				{
					vMinY = Convert.ToDouble(textMinY.Text);
				}
				catch (Exception)
				{
					textMinY.Focus();
				}						
		}

		private void textMaxYLeave(object sender, System.EventArgs e)
		{
			if (CheckMaxY = (textMaxY.Text.Length != 0))
				try
				{
					vMaxY = Convert.ToDouble(textMaxY.Text);
				}
				catch (Exception)
				{
					textMaxY.Focus();
				}						
		}

		private void textMinSXLeave(object sender, System.EventArgs e)
		{
			if (CheckMinSX = (textMinSX.Text.Length != 0))
				try
				{
					vMinSX = Convert.ToDouble(textMinSX.Text);
				}
				catch (Exception)
				{
					textMinSX.Focus();
				}						
		}

		private void textMaxSXLeave(object sender, System.EventArgs e)
		{
			if (CheckMaxSX = (textMaxSX.Text.Length != 0))
				try
				{
					vMaxSX = Convert.ToDouble(textMaxSX.Text);
				}
				catch (Exception)
				{
					textMaxSX.Focus();
				}								
		}

		private void textMinSYLeave(object sender, System.EventArgs e)
		{
			if (CheckMinSY = (textMinSY.Text.Length != 0))
				try
				{
					vMinSY = Convert.ToDouble(textMinSY.Text);
				}
				catch (Exception)
				{
					textMinSY.Focus();
				}								
		}

		private void textMaxSYLeave(object sender, System.EventArgs e)
		{
			if (CheckMaxSY = (textMaxSY.Text.Length != 0))
				try
				{
					vMaxSY = Convert.ToDouble(textMaxSY.Text);
				}
				catch (Exception)
				{
					textMaxSY.Focus();
				}										
		}

		private void textProjectToLayerLeave(object sender, System.EventArgs e)
		{
			if (CheckProjectToLayer = (textProjectToLayer.Text.Length != 0))
				try
				{
					vProjectToLayer = Convert.ToInt32(textProjectToLayer.Text);
				}
				catch (Exception)
				{
					textProjectToLayer.Focus();
				}										
		}

		private void buttonSearchAndShow_Click(object sender, System.EventArgs e)		
		{
			if (Vol == null) 
			{
				MessageBox.Show("Null volume. Please load a volume.");
				return;
			}
			if (radioLayers.Checked) SS_Layers();
			else if (radioSegments.Checked) SS_Segments();
			else if (radioTracks.Checked) SS_Tracks();
			else if (radioVertices.Checked) SS_Vertices();
			else if (radioVolume.Checked) SS_Volume();
		}

		private void AdjustColumns(params string [] array)
		{
			listResults.Clear();
			foreach (string s in array)
				listResults.Columns.Add(s, 1, HorizontalAlignment.Right);
			foreach (ColumnHeader c in listResults.Columns)
				c.Width = (listResults.Width - 16) / listResults.Columns.Count;
		}

		private void AddData(params string [] data)
		{
			ListViewItem lvi = listResults.Items.Add(" ");
			lvi.SubItems[0].Text = data[0];
			int i;
			for (i = 1; i < data.Length; i++)
				lvi.SubItems.Add(data[i]);
		}

		private void SS_Volume()
		{
			listResults.BeginUpdate();
			AdjustColumns("ID", "RefX", "RefY", "RefZ", "Layers", "Tracks", "Vertices");
			AddData(Vol.Id.Part0 + "/" + Vol.Id.Part1 + "/" + Vol.Id.Part2 + "/" + Vol.Id.Part3,
				Vol.RefCenter.X.ToString("F1"), Vol.RefCenter.Y.ToString("F1"), Vol.RefCenter.Z.ToString("F1"),
				Vol.Layers.Length.ToString(), Vol.Tracks.Length.ToString(), Vol.Vertices.Length.ToString());
			listResults.EndUpdate();
		}

		private void SS_Layers()
		{
			listResults.BeginUpdate();
			AdjustColumns("ID", "SheetID", "RefZ", "DownZ", "UpZ", "Segments", "MXX", "MXY", "MYX", "MYY", "TX", "TY", "TZ", "MSX", "MSY", "MDX", "MDY");
			int i;
			for (i = 0; i < Vol.Layers.Length; i++)
			{
				SySal.TotalScan.Layer l = Vol.Layers[i];
				if (CheckID && l.Id != vID) continue;
				if (CheckSheetID && l.SheetId != vSheetID) continue;
				SySal.TotalScan.AlignmentData a = l.AlignData;
				AddData(l.Id.ToString(), l.SheetId.ToString(), l.RefCenter.Z.ToString("F1"), l.DownstreamZ.ToString("F1"), l.UpstreamZ.ToString("F1"), l.Length.ToString(),
					a.AffineMatrixXX.ToString("F5"), a.AffineMatrixXY.ToString("F5"), a.AffineMatrixYX.ToString("F5"), a.AffineMatrixYY.ToString("F5"), 
					a.TranslationX.ToString("F1"), a.TranslationY.ToString("F1"), a.TranslationZ.ToString("F1"), 
					a.DShrinkX.ToString("F4"), a.DShrinkY.ToString("F4"), a.SAlignDSlopeX.ToString("F4"), a.SAlignDSlopeY.ToString("F4"));
			}
			listResults.EndUpdate();
		}

		private void SS_Segments()
		{
			listResults.BeginUpdate();
			AdjustColumns("LyrID", "SheetID", "TkID", "ID", "BaseID", "N", "PX", "PY", "SX", "SY", "S", "OPX", "OPY", "OSX", "OSY");
			int i, j;
			double rx = Vol.RefCenter.X;
			double ry = Vol.RefCenter.Y;
			for (i = 0; i < Vol.Layers.Length; i++)
			{
				SySal.TotalScan.Layer l = Vol.Layers[i];
				SySal.TotalScan.AlignmentData a = l.AlignData;
				if (CheckSheetID && l.SheetId != vSheetID) continue;
				if (CheckLayerID && l.Id != vLayerID) continue;
				for (j = 0; j < l.Length; j++)
				{
					SySal.TotalScan.Segment s = l[j];
					if (CheckID && j != vID) continue;
                    try
                    {
                        if (CheckBaseID && ((SySal.TotalScan.BaseTrackIndex)s.Index).Id != vBaseID) continue;
                    }
                    catch (Exception) { }
					if (CheckTrackID && (s.TrackOwner == null || s.TrackOwner.Id != vTrackID)) continue;
					SySal.Tracking.MIPEmulsionTrackInfo info = s.Info;

					if (CheckMinX && info.Intercept.X < vMinX) continue;
					if (CheckMaxX && info.Intercept.X > vMaxX) continue;
					if (CheckMinY && info.Intercept.Y < vMinY) continue;
					if (CheckMaxY && info.Intercept.Y > vMaxY) continue;

					if (CheckMinSX && info.Slope.X < vMinSX) continue;
					if (CheckMaxSX && info.Slope.X > vMaxSX) continue;
					if (CheckMinSY && info.Slope.Y < vMinSY) continue;
					if (CheckMaxSY && info.Slope.Y > vMaxSY) continue;

					double det = 1.0 / (a.AffineMatrixXX * a.AffineMatrixYY - a.AffineMatrixXY * a.AffineMatrixYX);
					double ixx, ixy, iyx, iyy;
					ixx = a.AffineMatrixYY * det;
					iyy = a.AffineMatrixXX * det;
					ixy = - a.AffineMatrixXY * det;
					iyx = - a.AffineMatrixYX * det;
					double osx, osy;
					osx = ixx * info.Slope.X + ixy * info.Slope.Y;
					osy = iyx * info.Slope.X + iyy * info.Slope.Y;
					AddData(l.Id.ToString(), l.SheetId.ToString(), (s.TrackOwner == null) ? "" : s.TrackOwner.Id.ToString(), j.ToString(), s.Index.ToString(), 
						info.Count.ToString(), info.Intercept.X.ToString("F1"), info.Intercept.Y.ToString("F1"),
						info.Slope.X.ToString("F4"), info.Slope.Y.ToString("F4"), info.Sigma.ToString("F3"),
						(ixx * (info.Intercept.X - rx - a.TranslationX) + ixy * (info.Intercept.Y - ry - a.TranslationY) + rx).ToString("F1"),
						(iyx * (info.Intercept.X - rx - a.TranslationX) + iyy * (info.Intercept.Y - ry - a.TranslationY) + ry).ToString("F1"),
						((osx - a.SAlignDSlopeX) / a.DShrinkX).ToString("F4"), ((osy - a.SAlignDSlopeY) / a.DShrinkY).ToString("F4"));
				}
			}			
			listResults.EndUpdate();
		}

		private void SS_Tracks()
		{
			listResults.BeginUpdate();
			AdjustColumns("ID", "N", "DwVID", "DwIP", "UpVID", "UpIP", "DwZX", "DwZY", "DwX", "DwY", "DwZ", "DwSX", "DwSY", "UpZX", "UpZY", "UpX", "UpY", "UpZ", "UpSX", "UpSY", "APX", "APY", "ASX", "ASY");
			int i;
			double rx = Vol.RefCenter.X;
			double ry = Vol.RefCenter.Y;
			for (i = 0; i < Vol.Tracks.Length; i++)
			{
				SySal.TotalScan.Track t = Vol.Tracks[i];
				if (CheckID && t.Id != vID) continue;
				if (CheckDownVID && (t.Downstream_Vertex == null || t.Downstream_Vertex.Id != vDownVID)) continue;
				if (CheckUpVID && (t.Upstream_Vertex == null || t.Upstream_Vertex.Id != vUpVID)) continue;
				
				double dwsx = t.Downstream_SlopeX, dwsy = t.Downstream_SlopeY, upsx = t.Upstream_SlopeX, upsy = t.Upstream_SlopeY;
				
				if (CheckMinSX && dwsx < vMinSX && upsx < vMinSX) continue;
				if (CheckMaxSX && dwsx > vMaxSX && upsx > vMaxSX) continue;
				if (CheckMinSY && dwsy < vMinSY && upsy < vMinSY) continue;
				if (CheckMaxSY && dwsy > vMaxSY && upsy > vMaxSY) continue;

				double dwzx = t.Downstream_PosX, dwzy = t.Downstream_PosY, upzx = t.Upstream_PosX, upzy = t.Upstream_PosY;
				double dwx, dwy, upx, upy;
				
				dwx = dwzx + t.Downstream_Z * t.Downstream_SlopeX;
				dwy = dwzy + t.Downstream_Z * t.Downstream_SlopeY;

				upx = upzx + t.Upstream_Z * t.Upstream_SlopeX;
				upy = upzy + t.Upstream_Z * t.Upstream_SlopeY;

				if (CheckMinX && dwx < vMinX && upx < vMinX && dwzx < vMinX && upzx < vMinX) continue;
				if (CheckMaxX && dwx > vMaxX && upx > vMaxX && dwzx > vMaxX && upzx > vMaxX) continue;
				if (CheckMinY && dwy < vMinY && upy < vMinY && dwzy < vMinY && upzy < vMinY) continue;
				if (CheckMaxY && dwy > vMaxY && upy > vMaxY && dwzy > vMaxY && upzy > vMaxY) continue;

				double apx = 0.0, apy = 0.0, asx = 0.0, asy = 0.0;
				if (CheckProjectToLayer)
				{
					int k;
					if (vProjectToLayer < 0) vProjectToLayer = 0;
					else if (vProjectToLayer >= Vol.Layers.Length) vProjectToLayer = Vol.Layers.Length - 1;
					SySal.TotalScan.Layer l = Vol.Layers[vProjectToLayer];
					SySal.TotalScan.AlignmentData a = l.AlignData;
					t.FittingSegments = 3;
					for (k = 0; k < t.Length - 2 && t[k].LayerOwner.Id < vProjectToLayer; k++);
					if (t.Length >= 1 /* was 3! */)
					{
						k--; if (k < 0) k = 0;
						double px, py, sx, sy;
						t.Compute_Local_XCoord(k, out sx, out px);
						t.Compute_Local_YCoord(k, out sy, out py);
						px += sx * l.RefCenter.Z;
						py += sy * l.RefCenter.Z;

						double det = 1.0 / (a.AffineMatrixXX * a.AffineMatrixYY - a.AffineMatrixXY * a.AffineMatrixYX);
						double ixx, ixy, iyx, iyy;
						ixx = a.AffineMatrixYY * det;
						iyy = a.AffineMatrixXX * det;
						ixy = - a.AffineMatrixXY * det;
						iyx = - a.AffineMatrixYX * det;
						apx = ixx * (px - rx - a.TranslationX) + ixy * (py - ry - a.TranslationY) + rx;
						apy = iyx * (px - rx - a.TranslationX) + iyy * (py - ry - a.TranslationY) + ry;
						asx = ixx * sx + ixy * sy;
						asy = iyx * sx + iyy * sy;
					}
				}

				AddData(t.Id.ToString(), t.Length.ToString(), (t.Downstream_Vertex == null) ? "" : t.Downstream_Vertex.Id.ToString(), (t.Downstream_Vertex == null) ? "" : t.Downstream_Impact_Parameter.ToString("F2"),
					(t.Upstream_Vertex == null) ? "" : t.Upstream_Vertex.Id.ToString(), (t.Upstream_Vertex == null) ? "" : t.Upstream_Impact_Parameter.ToString("F2"),
					dwzx.ToString("F1"), dwzy.ToString("F1"), dwx.ToString("F1"), dwy.ToString("F1"), t.Downstream_Z.ToString("F1"), dwsx.ToString("F4"), dwsy.ToString("F4"),
					upzx.ToString("F1"), upzy.ToString("F1"), upx.ToString("F1"), upy.ToString("F1"), t.Upstream_Z.ToString("F1"), upsx.ToString("F4"), upsy.ToString("F4"),
					CheckProjectToLayer ? apx.ToString("F1") : "", CheckProjectToLayer ? apy.ToString("F1") : "", CheckProjectToLayer ? asx.ToString("F4") : "", CheckProjectToLayer ? asy.ToString("F4") : "");
			}
			listResults.EndUpdate();
		}

		private void SS_Vertices()
		{
			listResults.BeginUpdate();
			AdjustColumns("ID", "N", "AvgD", "X", "Y", "Z", "DownID", "DownSheetID");
			int i;
			for (i = 0; i < Vol.Vertices.Length; i++)
			{
				SySal.TotalScan.Vertex v = Vol.Vertices[i];
				if (CheckID && v.Id != vID) continue;
				if (CheckMinX && v.X < vMinX) continue;
				if (CheckMaxX && v.X > vMaxX) continue;
				if (CheckMinY && v.Y < vMinY) continue;
				if (CheckMaxY && v.Y > vMaxY) continue;
				int k;
				for (k = 0; k < Vol.Layers.Length && Vol.Layers[k].UpstreamZ > v.Z; k++);
				if (k > 0) k--;
				AddData(v.Id.ToString(), v.Length.ToString(), v.AverageDistance.ToString("F2"), v.X.ToString("F1"), v.Y.ToString("F1"), v.Z.ToString("F1"), Vol.Layers[k].Id.ToString(), Vol.Layers[k].SheetId.ToString());
			}
			listResults.EndUpdate();
		}

		private void buttonSave_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select file to dump the current data set";
			sdlg.Filter = "Text files (*.txt)|*.txt|Data files (*.dat)|*.dat|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK)
			{
				int i, j;
				System.IO.StreamWriter w = new System.IO.StreamWriter(sdlg.FileName);
				for (j = 0; j < listResults.Columns.Count; j++)
				{
					w.Write(listResults.Columns[j].Text);
					if (j < listResults.Columns.Count - 1) w.Write("\t");
				}
				w.WriteLine();
				for (i = 0; i < listResults.Items.Count; i++)
				{
					ListViewItem lvi = listResults.Items[i];
					for (j = 0; j < lvi.SubItems.Count; j++)
					{
						if (lvi.SubItems[j].Text.Length > 0) w.Write(lvi.SubItems[j].Text); else w.Write("-1");
						if (j < lvi.SubItems.Count - 1) w.Write("\t");
					}
					w.WriteLine();
				}
				w.Flush();
				w.Close();
			}
		}

	}
}
