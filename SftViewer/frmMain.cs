using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using GDI3D;
using SySal.OperaDb.ComputingInfrastructure;
using System.Xml.Serialization;

namespace SySal.Executables.SftViewer
{
		
	/// <summary>
	/// SftViewer is a GUI tool to display PEANUT events
	/// using the OPERA database.
	/// </summary>
	/// <remarks>
	/// The standard usage of this tool requires the following steps:
	/// <list type="number">
	/// <item><term>connect to a DB (using <see cref="SySal.Executables.SftViewer.frmLogin">frmLogin</see>);</term></item>
	/// <item>select a process operation;</item>
	/// <item>select the run;</item>
	/// <item>type in the event # and push Enter;</item>
	/// <item>select the elements to display and choose the appropriate colors.</item> 
	/// </list>
	/// Double-clicking on an element displays information about it.
	/// <para>The path view is a 3D view provided by <see cref="GDI3D.Scene">GDI3D</see> and its <see cref="GDI3D.Control.GDIDisplay">Display Control</see>.</para>
	/// <para>By dragging the image with the <u>right mouse button</u> pressed, one can rotate/pan it.</para>
	/// <para>Default points of view are available (XY, YZ, XZ).</para>
	/// <para>To center the view on a particularly interesting point, click on <i>Set Focus</i> and then <u>left-click</u> on the interesting point.</para>
	/// <para>The image obtained can be saved to an <see cref="GDI3D.Scene">X3L</see> file, and/or it can be merged with an X3L file. This is typically used to overlap the scanback/scanoforth history with TotalScan results.</para> 
	/// </remarks>
	public class frmMain : System.Windows.Forms.Form
	{
		public long CurrEvent; 
		
		[Serializable]
			public class AppSettings
		{
			public bool Draw3DTracks;
			public bool DrawProjTracks;
			public bool DrawProjHits;
			public bool DrawExposed;
			public bool DrawBricks;
		}
		public static AppSettings Settings;
		public class ThreeDTrack
		{
			int Index;
			public ThreeDTrack(int idx)
			{
				Index = idx;
			}
			public override string ToString()
			{
				if (ThreeDTracks == null) return null;
				else 
				{
					ThreeDTracks.Row = Index;
					return String.Format("3D track, id = {0}, trk_id_x = {1}, trk_id_y = {2}", ThreeDTracks._TRACK_ID, ThreeDTracks._TRK_ID_X, ThreeDTracks._TRK_ID_Y);
				}
			}
		}

		public class ProjTrack
		{
			int Index;
			public ProjTrack(int idx)
			{
				Index = idx;
			}
			public override string ToString()
			{
				if (ProjTracks == null) return null;
				else
				{
					ProjTracks.Row = Index;
					return String.Format("{0}-track, id = {3}, a{0}={1}, b{0}={2}, nhits=4", ProjTracks._PROJ_ID, ProjTracks._ACOORD, ProjTracks._BCOORD, ProjTracks._TRACK_ID, ProjTracks._NHITS);
				}
			}
		}

		public class Hit
		{
			int Index;
			public Hit(int idx)
			{
				Index = idx;
			}
			public override string ToString()
			{
				if (Hits == null) return null;
				else
				{
					Hits.Row = Index;
					return String.Format("{0}-hit, id={1}, {0}={2}, brightness={3}", 
						Hits._PROJ_ID, Hits._HIT_ID, Hits._TCOORD, Hits._BRIGHTNESS);
					//return String.Format("{0}-hit, ID={1}", Hits._PROJ_ID);
				}
			}

		}

		public Box[,,] Boxes = new Box[3,4,4];
		public class Box: ICloneable
		{
			//public string Description;
			public GDI3D.Control.Line[] Lines = new GDI3D.Control.Line[12];
			public double[] Xs = new double[2];
			public double[] Ys = new double[2];
			public double[] Zs = new double[2];
			public ArrayList BrickIndices;
			public int PosCode;
			public System.Drawing.Color Color;
			

			public Box(double x0, double y0, double z0, double x1, double y1, double z1, int poscode, ArrayList brickIdx, System.Drawing.Color color)
			{
				PosCode = poscode;
				BrickIndices = new ArrayList();
				int iline = 0;
				Xs[0] = x0;
				Xs[1] = x1;
				Ys[0] = y0;
				Ys[1] = y1;
				Zs[0] = z0;
				Zs[1] = z1;
				Color = color;
				
				for (int ix=0; ix<2; ix++)
					for (int iy=0; iy<2; iy++)
						for (int iz=0; iz<2; iz++)
						{
							int iix = ix+1;
							int iiy = iy+1;
							int iiz = iz+1;
							if (iix<2) Lines[iline++] = new GDI3D.Control.Line(Xs[ix], Ys[iy], Zs[iz], Xs[iix], Ys[iy], Zs[iz], this, color.R, color.G, color.B);
							if (iiy<2) Lines[iline++] = new GDI3D.Control.Line(Xs[ix], Ys[iy], Zs[iz], Xs[ix], Ys[iiy], Zs[iz], this, color.R, color.G, color.B);
							if (iiz<2) Lines[iline++] = new GDI3D.Control.Line(Xs[ix], Ys[iy], Zs[iz], Xs[ix], Ys[iy], Zs[iiz], this, color.R, color.G, color.B);
							
						}
				//		Console.WriteLine("Done");
			}
			public object Clone()
			{
				return new Box(Xs[0], Ys[0], Zs[0], Xs[1], Ys[1], Zs[1], PosCode, BrickIndices, Color);
			}

			public override string ToString()
			{
				string result = String.Format("Brick(s): poscode = {0} ", PosCode);
				
				for (int i=0; i<BrickIndices.Count; i++)
				{
					Bricks.Row = (int) BrickIndices[i];
					result += Bricks._ID_EVENTBRICK.ToString() + ' ';
				}					
				
				return result;				
			}

		}

		private GDI3D.Control.GDIDisplay gdiDisplay1;
		private System.Windows.Forms.Button btnXY;
		private System.Windows.Forms.Button btnXZ;
		
		public string Password;
		public string DbServer;
		public string Username;

		public static double Xmin = -58.2 - 128.1;
		public static double Xmax = 198.0;
		public static double Ymin = -210;
		public static double Ymax = 210;
		public static double Zmin = -252.4 - 79.6;
		public static double Zmax = 134.1;

		public class PeanutRun
		{
			public int Id;
			public int MaxEvent;
		}

		public PeanutRun[] Runs;
		public static SySal.OperaDb.OperaDbConnection Conn;
		long SelectedProcopId;
		long SelectedEvent;

		public static SySal.OperaDb.Schema.TB_PEANUT_HITS Hits;
		public static SySal.OperaDb.Schema.TB_PEANUT_PREDTRACKS ProjTracks;
		public static SySal.OperaDb.Schema.TB_PEANUT_PREDTRACKS ThreeDTracks;
		public static SySal.OperaDb.Schema.TB_PEANUT_BRICKINFO Bricks;
		public static SySal.OperaDb.Schema.TB_PEANUT_BRICKINFO ExposedBricks;

		public static System.Drawing.Color ProjHitColor = System.Drawing.Color.Aquamarine;
		public static System.Drawing.Color ProjTrackColor = System.Drawing.Color.Purple;
		public static System.Drawing.Color ThreeDTrackColor = System.Drawing.Color.Brown;
		public static System.Drawing.Color SftColor = System.Drawing.Color.LightGray;
		public static System.Drawing.Color ExposedColor = System.Drawing.Color.Black;


		private System.Windows.Forms.Label lblProcopId;
		private System.Windows.Forms.Label lblRun;
		private System.Windows.Forms.ComboBox cmbProcopid;
		private System.Windows.Forms.ComboBox cmbRun;
		private System.Windows.Forms.TextBox txtMaxEvent;
		private System.Windows.Forms.Label lblMaxEvent;
		private System.Windows.Forms.Button btnZoomIn;
		private System.Windows.Forms.Button btnZoomOut;
		private System.Windows.Forms.Button btnPan;
		private System.Windows.Forms.Button btnRotate;
		private System.Windows.Forms.Button btnFocus;
		private System.Windows.Forms.Button btnConnect;
		private System.Windows.Forms.Button btnDraw;
		private System.Windows.Forms.Panel panel1;
		private System.Windows.Forms.ColorDialog colorDialog1;
		private System.Windows.Forms.Button btn3dTracksColor;
		private System.Windows.Forms.Button btnProjHitsColor;
		private System.Windows.Forms.Button btnProjTracksColor;
		private System.Windows.Forms.CheckBox ckbProjHits;
		private System.Windows.Forms.CheckBox ckbProjTracks;
		private System.Windows.Forms.CheckBox ckb3dTracks;
		private System.Windows.Forms.StatusBar statusBar1;
		private System.Windows.Forms.CheckBox ckbBricks;
		private System.Windows.Forms.Button btnBricks;
		private System.Windows.Forms.Button btnSave;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		private System.Windows.Forms.Button btnExposed;
		private System.Windows.Forms.CheckBox ckbExposed;
		private System.Windows.Forms.TextBox txtDetails;
		private System.Windows.Forms.Button btnYZ;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public frmMain()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		
			/*SySal.OperaDb.OperaDbCredentials cred = new SySal.OperaDb.OperaDbCredentials();
			cred.DBPassword = "KARAMUL";
			cred.DBUserName = "NIKOLAY";
			cred.DBServer = "opita";
			Conn = cred.Connect();*/

			CurrEvent = 0;

			SySal.OperaDb.Schema.DB = Conn;

			btn3dTracksColor.BackColor = ThreeDTrackColor;
			btnProjTracksColor.BackColor = ProjTrackColor;
			btnProjHitsColor.BackColor = ProjHitColor;
			btnBricks.BackColor = SftColor;
			btnExposed.BackColor = ExposedColor;

			gdiDisplay1.SetCameraSpotting(0.5,0.5,0.5);
			gdiDisplay1.Infinity = true;
			gdiDisplay1.Distance = 1000000;
			gdiDisplay1.Zoom = 1e-1;
			gdiDisplay1.SetCameraOrientation(0,1,0,0,0,-1);
			gdiDisplay1.Render();
			gdiDisplay1.Refresh();
			gdiDisplay1.AutoRender = true;

			gdiDisplay1.DoubleClickSelect = new GDI3D.Control.SelectObject(ShowInfo);
			
			
			/*Settings.Password = "a";
			Settings.UserName = "b";
			Settings.DbServer = "c";

		*/

			
			//DrawSft();
		}

		public void ReadSettings()
		{
			try
			{
				System.Xml.Serialization.XmlSerializer xmls = new XmlSerializer(Settings.GetType());
				System.Xml.XmlTextReader r = new System.Xml.XmlTextReader(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Personal) + "\\sftviewer.config");
				Settings = (AppSettings) xmls.Deserialize(r);				
				ckb3dTracks.Checked = Settings.Draw3DTracks;
				ckbBricks.Checked = Settings.DrawBricks;
				ckbExposed.Checked = Settings.DrawExposed;
				ckbProjHits.Checked = Settings.DrawProjHits;
				ckbProjTracks.Checked = Settings.DrawProjTracks;
				r.Close();
			}
			catch
			{				
			}
		}

		public void SaveSettings()
		{
			
			try
			{
				Settings.Draw3DTracks = ckb3dTracks.Checked;
				Settings.DrawBricks = ckbBricks.Checked;
				Settings.DrawExposed = ckbExposed.Checked;
				Settings.DrawProjHits = ckbProjHits.Checked;
				Settings.DrawProjTracks = ckbProjTracks.Checked;
				System.Xml.Serialization.XmlSerializer xmls = new XmlSerializer(Settings.GetType());
				System.Xml.XmlTextWriter w = new System.Xml.XmlTextWriter(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Personal) + "\\sftviewer.config", System.Text.Encoding.ASCII);
				xmls.Serialize(w, Settings);
				w.Close();
			}
			catch (Exception ex)
			{
				string temp = ex.ToString();
			}
		}

		public void ShowInfo(object o)
		{
			//if (o!=null) statusBar1.Text = o.ToString();
			if (o!=null) txtDetails.Text = o.ToString();
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
			this.gdiDisplay1 = new GDI3D.Control.GDIDisplay();
			this.btnXY = new System.Windows.Forms.Button();
			this.btnXZ = new System.Windows.Forms.Button();
			this.cmbProcopid = new System.Windows.Forms.ComboBox();
			this.lblProcopId = new System.Windows.Forms.Label();
			this.lblRun = new System.Windows.Forms.Label();
			this.cmbRun = new System.Windows.Forms.ComboBox();
			this.lblMaxEvent = new System.Windows.Forms.Label();
			this.txtMaxEvent = new System.Windows.Forms.TextBox();
			this.btnZoomIn = new System.Windows.Forms.Button();
			this.btnZoomOut = new System.Windows.Forms.Button();
			this.btnPan = new System.Windows.Forms.Button();
			this.btnRotate = new System.Windows.Forms.Button();
			this.btnFocus = new System.Windows.Forms.Button();
			this.btnConnect = new System.Windows.Forms.Button();
			this.btnDraw = new System.Windows.Forms.Button();
			this.panel1 = new System.Windows.Forms.Panel();
			this.btnExposed = new System.Windows.Forms.Button();
			this.ckbExposed = new System.Windows.Forms.CheckBox();
			this.btnBricks = new System.Windows.Forms.Button();
			this.ckbBricks = new System.Windows.Forms.CheckBox();
			this.btn3dTracksColor = new System.Windows.Forms.Button();
			this.btnProjTracksColor = new System.Windows.Forms.Button();
			this.btnProjHitsColor = new System.Windows.Forms.Button();
			this.ckb3dTracks = new System.Windows.Forms.CheckBox();
			this.ckbProjTracks = new System.Windows.Forms.CheckBox();
			this.ckbProjHits = new System.Windows.Forms.CheckBox();
			this.colorDialog1 = new System.Windows.Forms.ColorDialog();
			this.statusBar1 = new System.Windows.Forms.StatusBar();
			this.btnSave = new System.Windows.Forms.Button();
			this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
			this.txtDetails = new System.Windows.Forms.TextBox();
			this.btnYZ = new System.Windows.Forms.Button();
			this.panel1.SuspendLayout();
			this.SuspendLayout();
			// 
			// gdiDisplay1
			// 
			this.gdiDisplay1.Alpha = 0.50196078431372548;
			this.gdiDisplay1.AutoRender = true;
			this.gdiDisplay1.BackColor = System.Drawing.Color.White;
			this.gdiDisplay1.BorderWidth = 1;
			this.gdiDisplay1.ClickSelect = null;
			this.gdiDisplay1.Distance = 10;
			this.gdiDisplay1.DoubleClickSelect = null;
			this.gdiDisplay1.Infinity = true;
			this.gdiDisplay1.LineWidth = 1;
			this.gdiDisplay1.Location = new System.Drawing.Point(8, 8);
			this.gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
			this.gdiDisplay1.MouseMultiplier = 0.01;
			this.gdiDisplay1.Name = "gdiDisplay1";
			this.gdiDisplay1.NextClickSetsCenter = false;
			this.gdiDisplay1.PointSize = 5;
			this.gdiDisplay1.Size = new System.Drawing.Size(744, 528);
			this.gdiDisplay1.TabIndex = 0;
			this.gdiDisplay1.Zoom = 20;
			// 
			// btnXY
			// 
			this.btnXY.Location = new System.Drawing.Point(792, 72);
			this.btnXY.Name = "btnXY";
			this.btnXY.Size = new System.Drawing.Size(48, 23);
			this.btnXY.TabIndex = 4;
			this.btnXY.Text = "XY";
			this.btnXY.Click += new System.EventHandler(this.btnXY_Click);
			// 
			// btnXZ
			// 
			this.btnXZ.Location = new System.Drawing.Point(848, 72);
			this.btnXZ.Name = "btnXZ";
			this.btnXZ.Size = new System.Drawing.Size(48, 23);
			this.btnXZ.TabIndex = 5;
			this.btnXZ.Text = "XZ";
			this.btnXZ.Click += new System.EventHandler(this.btnXZ_Click);
			// 
			// cmbProcopid
			// 
			this.cmbProcopid.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbProcopid.Location = new System.Drawing.Point(768, 320);
			this.cmbProcopid.Name = "cmbProcopid";
			this.cmbProcopid.Size = new System.Drawing.Size(184, 21);
			this.cmbProcopid.TabIndex = 9;
			this.cmbProcopid.SelectedValueChanged += new System.EventHandler(this.comboBox1_SelectedValueChanged);
			// 
			// lblProcopId
			// 
			this.lblProcopId.Location = new System.Drawing.Point(768, 296);
			this.lblProcopId.Name = "lblProcopId";
			this.lblProcopId.Size = new System.Drawing.Size(144, 23);
			this.lblProcopId.TabIndex = 10;
			this.lblProcopId.Text = "Select process operation:";
			// 
			// lblRun
			// 
			this.lblRun.Location = new System.Drawing.Point(768, 352);
			this.lblRun.Name = "lblRun";
			this.lblRun.TabIndex = 11;
			this.lblRun.Text = "Select run:";
			this.lblRun.Click += new System.EventHandler(this.lblRun_Click);
			// 
			// cmbRun
			// 
			this.cmbRun.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cmbRun.Location = new System.Drawing.Point(872, 352);
			this.cmbRun.Name = "cmbRun";
			this.cmbRun.Size = new System.Drawing.Size(80, 21);
			this.cmbRun.TabIndex = 12;
			this.cmbRun.SelectedValueChanged += new System.EventHandler(this.cmbRun_SelectedValueChanged);
			// 
			// lblMaxEvent
			// 
			this.lblMaxEvent.Location = new System.Drawing.Point(768, 376);
			this.lblMaxEvent.Name = "lblMaxEvent";
			this.lblMaxEvent.Size = new System.Drawing.Size(80, 32);
			this.lblMaxEvent.TabIndex = 13;
			this.lblMaxEvent.Text = "Type event # (up to):";
			// 
			// txtMaxEvent
			// 
			this.txtMaxEvent.Location = new System.Drawing.Point(872, 384);
			this.txtMaxEvent.Name = "txtMaxEvent";
			this.txtMaxEvent.Size = new System.Drawing.Size(80, 20);
			this.txtMaxEvent.TabIndex = 14;
			this.txtMaxEvent.Text = "";
			this.txtMaxEvent.KeyDown += new System.Windows.Forms.KeyEventHandler(this.frmMain_KeyDown);
			this.txtMaxEvent.Leave += new System.EventHandler(this.txtMaxEvent_Leave);
			// 
			// btnZoomIn
			// 
			this.btnZoomIn.Location = new System.Drawing.Point(824, 112);
			this.btnZoomIn.Name = "btnZoomIn";
			this.btnZoomIn.Size = new System.Drawing.Size(48, 23);
			this.btnZoomIn.TabIndex = 4;
			this.btnZoomIn.Text = "In";
			this.btnZoomIn.Click += new System.EventHandler(this.btnZoomIn_Click);
			// 
			// btnZoomOut
			// 
			this.btnZoomOut.Location = new System.Drawing.Point(888, 112);
			this.btnZoomOut.Name = "btnZoomOut";
			this.btnZoomOut.Size = new System.Drawing.Size(48, 23);
			this.btnZoomOut.TabIndex = 15;
			this.btnZoomOut.Text = "Out";
			this.btnZoomOut.Click += new System.EventHandler(this.btnZoomOut_Click);
			// 
			// btnPan
			// 
			this.btnPan.Location = new System.Drawing.Point(824, 24);
			this.btnPan.Name = "btnPan";
			this.btnPan.Size = new System.Drawing.Size(48, 23);
			this.btnPan.TabIndex = 16;
			this.btnPan.Text = "Pan";
			this.btnPan.Click += new System.EventHandler(this.btnPan_Click);
			// 
			// btnRotate
			// 
			this.btnRotate.Location = new System.Drawing.Point(880, 24);
			this.btnRotate.Name = "btnRotate";
			this.btnRotate.Size = new System.Drawing.Size(48, 23);
			this.btnRotate.TabIndex = 17;
			this.btnRotate.Text = "Rotate";
			this.btnRotate.Click += new System.EventHandler(this.btnRotate_Click);
			// 
			// btnFocus
			// 
			this.btnFocus.Location = new System.Drawing.Point(936, 24);
			this.btnFocus.Name = "btnFocus";
			this.btnFocus.Size = new System.Drawing.Size(48, 23);
			this.btnFocus.TabIndex = 18;
			this.btnFocus.Text = "Focus";
			this.btnFocus.Click += new System.EventHandler(this.btnFocus_Click);
			// 
			// btnConnect
			// 
			this.btnConnect.Location = new System.Drawing.Point(760, 24);
			this.btnConnect.Name = "btnConnect";
			this.btnConnect.Size = new System.Drawing.Size(56, 23);
			this.btnConnect.TabIndex = 19;
			this.btnConnect.Text = "Connect ";
			this.btnConnect.Click += new System.EventHandler(this.btnConnect_Click);
			// 
			// btnDraw
			// 
			this.btnDraw.Enabled = false;
			this.btnDraw.Location = new System.Drawing.Point(872, 416);
			this.btnDraw.Name = "btnDraw";
			this.btnDraw.Size = new System.Drawing.Size(80, 23);
			this.btnDraw.TabIndex = 20;
			this.btnDraw.Text = "Draw";
			this.btnDraw.Click += new System.EventHandler(this.btnDraw_Click);
			// 
			// panel1
			// 
			this.panel1.Controls.Add(this.btnExposed);
			this.panel1.Controls.Add(this.ckbExposed);
			this.panel1.Controls.Add(this.btnBricks);
			this.panel1.Controls.Add(this.ckbBricks);
			this.panel1.Controls.Add(this.btn3dTracksColor);
			this.panel1.Controls.Add(this.btnProjTracksColor);
			this.panel1.Controls.Add(this.btnProjHitsColor);
			this.panel1.Controls.Add(this.ckb3dTracks);
			this.panel1.Controls.Add(this.ckbProjTracks);
			this.panel1.Controls.Add(this.ckbProjHits);
			this.panel1.Location = new System.Drawing.Point(776, 144);
			this.panel1.Name = "panel1";
			this.panel1.Size = new System.Drawing.Size(184, 144);
			this.panel1.TabIndex = 21;
			// 
			// btnExposed
			// 
			this.btnExposed.BackColor = System.Drawing.SystemColors.ActiveBorder;
			this.btnExposed.ForeColor = System.Drawing.SystemColors.ControlText;
			this.btnExposed.Location = new System.Drawing.Point(128, 104);
			this.btnExposed.Name = "btnExposed";
			this.btnExposed.Size = new System.Drawing.Size(32, 23);
			this.btnExposed.TabIndex = 14;
			this.btnExposed.Click += new System.EventHandler(this.btnExposed_Click);
			// 
			// ckbExposed
			// 
			this.ckbExposed.Location = new System.Drawing.Point(24, 104);
			this.ckbExposed.Name = "ckbExposed";
			this.ckbExposed.TabIndex = 13;
			this.ckbExposed.Text = "Exposed bricks";
			this.ckbExposed.CheckedChanged += new System.EventHandler(this.ckbExposed_CheckedChanged);
			// 
			// btnBricks
			// 
			this.btnBricks.BackColor = System.Drawing.SystemColors.ActiveBorder;
			this.btnBricks.ForeColor = System.Drawing.SystemColors.ControlText;
			this.btnBricks.Location = new System.Drawing.Point(128, 80);
			this.btnBricks.Name = "btnBricks";
			this.btnBricks.Size = new System.Drawing.Size(32, 23);
			this.btnBricks.TabIndex = 12;
			this.btnBricks.Click += new System.EventHandler(this.btnBricks_Click);
			// 
			// ckbBricks
			// 
			this.ckbBricks.Checked = true;
			this.ckbBricks.CheckState = System.Windows.Forms.CheckState.Checked;
			this.ckbBricks.Location = new System.Drawing.Point(24, 80);
			this.ckbBricks.Name = "ckbBricks";
			this.ckbBricks.TabIndex = 11;
			this.ckbBricks.Text = "All bricks";
			this.ckbBricks.CheckedChanged += new System.EventHandler(this.ckbBricks_CheckedChanged);
			// 
			// btn3dTracksColor
			// 
			this.btn3dTracksColor.BackColor = System.Drawing.SystemColors.ActiveBorder;
			this.btn3dTracksColor.ForeColor = System.Drawing.SystemColors.ControlText;
			this.btn3dTracksColor.Location = new System.Drawing.Point(128, 56);
			this.btn3dTracksColor.Name = "btn3dTracksColor";
			this.btn3dTracksColor.Size = new System.Drawing.Size(32, 23);
			this.btn3dTracksColor.TabIndex = 10;
			this.btn3dTracksColor.Click += new System.EventHandler(this.btn3dTracksColor_Click);
			// 
			// btnProjTracksColor
			// 
			this.btnProjTracksColor.BackColor = System.Drawing.SystemColors.ActiveBorder;
			this.btnProjTracksColor.ForeColor = System.Drawing.SystemColors.ControlText;
			this.btnProjTracksColor.Location = new System.Drawing.Point(128, 32);
			this.btnProjTracksColor.Name = "btnProjTracksColor";
			this.btnProjTracksColor.Size = new System.Drawing.Size(32, 23);
			this.btnProjTracksColor.TabIndex = 8;
			this.btnProjTracksColor.Click += new System.EventHandler(this.btnProjTracksColor_Click);
			// 
			// btnProjHitsColor
			// 
			this.btnProjHitsColor.BackColor = System.Drawing.SystemColors.ActiveBorder;
			this.btnProjHitsColor.ForeColor = System.Drawing.SystemColors.ControlText;
			this.btnProjHitsColor.Location = new System.Drawing.Point(128, 8);
			this.btnProjHitsColor.Name = "btnProjHitsColor";
			this.btnProjHitsColor.Size = new System.Drawing.Size(32, 23);
			this.btnProjHitsColor.TabIndex = 7;
			this.btnProjHitsColor.Click += new System.EventHandler(this.btnProjHitsColor_Click);
			// 
			// ckb3dTracks
			// 
			this.ckb3dTracks.Location = new System.Drawing.Point(24, 56);
			this.ckb3dTracks.Name = "ckb3dTracks";
			this.ckb3dTracks.TabIndex = 4;
			this.ckb3dTracks.Text = "3D tracks";
			this.ckb3dTracks.CheckedChanged += new System.EventHandler(this.ckb3dTracks_CheckedChanged);
			// 
			// ckbProjTracks
			// 
			this.ckbProjTracks.Location = new System.Drawing.Point(24, 32);
			this.ckbProjTracks.Name = "ckbProjTracks";
			this.ckbProjTracks.TabIndex = 3;
			this.ckbProjTracks.Text = "Projected tracks";
			this.ckbProjTracks.CheckedChanged += new System.EventHandler(this.ckbProjTracks_CheckedChanged);
			// 
			// ckbProjHits
			// 
			this.ckbProjHits.Location = new System.Drawing.Point(24, 8);
			this.ckbProjHits.Name = "ckbProjHits";
			this.ckbProjHits.TabIndex = 2;
			this.ckbProjHits.Text = "Projected hits";
			this.ckbProjHits.CheckedChanged += new System.EventHandler(this.ckbProjHits_CheckedChanged);
			// 
			// statusBar1
			// 
			this.statusBar1.Location = new System.Drawing.Point(0, 552);
			this.statusBar1.Name = "statusBar1";
			this.statusBar1.Size = new System.Drawing.Size(1000, 22);
			this.statusBar1.TabIndex = 22;
			// 
			// btnSave
			// 
			this.btnSave.Location = new System.Drawing.Point(768, 416);
			this.btnSave.Name = "btnSave";
			this.btnSave.Size = new System.Drawing.Size(80, 23);
			this.btnSave.TabIndex = 23;
			this.btnSave.Text = "Save";
			this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
			// 
			// saveFileDialog1
			// 
			this.saveFileDialog1.Filter = "XML 3D files (*.x3l)|*.x3l|All files (*.*)|*.*";
			this.saveFileDialog1.Title = "Select output file";
			// 
			// txtDetails
			// 
			this.txtDetails.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(255)), ((System.Byte)(192)));
			this.txtDetails.Location = new System.Drawing.Point(768, 456);
			this.txtDetails.Multiline = true;
			this.txtDetails.Name = "txtDetails";
			this.txtDetails.ReadOnly = true;
			this.txtDetails.Size = new System.Drawing.Size(216, 80);
			this.txtDetails.TabIndex = 24;
			this.txtDetails.Text = "";
			// 
			// btnYZ
			// 
			this.btnYZ.Location = new System.Drawing.Point(904, 72);
			this.btnYZ.Name = "btnYZ";
			this.btnYZ.Size = new System.Drawing.Size(48, 23);
			this.btnYZ.TabIndex = 25;
			this.btnYZ.Text = "YZ";
			this.btnYZ.Click += new System.EventHandler(this.btnYZ_Click);
			// 
			// frmMain
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(1000, 574);
			this.Controls.Add(this.btnYZ);
			this.Controls.Add(this.txtDetails);
			this.Controls.Add(this.txtMaxEvent);
			this.Controls.Add(this.gdiDisplay1);
			this.Controls.Add(this.btnSave);
			this.Controls.Add(this.statusBar1);
			this.Controls.Add(this.panel1);
			this.Controls.Add(this.btnDraw);
			this.Controls.Add(this.btnConnect);
			this.Controls.Add(this.btnFocus);
			this.Controls.Add(this.btnRotate);
			this.Controls.Add(this.btnPan);
			this.Controls.Add(this.btnZoomOut);
			this.Controls.Add(this.lblMaxEvent);
			this.Controls.Add(this.cmbRun);
			this.Controls.Add(this.lblRun);
			this.Controls.Add(this.lblProcopId);
			this.Controls.Add(this.cmbProcopid);
			this.Controls.Add(this.btnXZ);
			this.Controls.Add(this.btnXY);
			this.Controls.Add(this.btnZoomIn);
			this.Name = "frmMain";
			this.Text = "SFT event viewer";
			this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.frmMain_KeyDown);
			this.Load += new System.EventHandler(this.frmMain_Load);
			this.Closed += new System.EventHandler(this.frmMain_Closed);
			this.panel1.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{						
			Application.Run(new frmMain());
		}

		/*	private void frmMain_Load(object sender, System.EventArgs e)
			{

			}*/

		private void DrawSft()
		{
			if (!ckbBricks.Checked) return;
			gdiDisplay1.AutoRender = false;
			double[] xi = new double[3]{198, 69.9, -58.2};
			double[] xf = new double[3]{69.9, -58.2, -186.3};
			double[] yi = new double[4]{210, 105, 0, -105};
			double[] yf = new double[4]{105, 0, -105, -210};
			double[] zi = new double[4]{-332, -208, -66.5, 54.5};
			double[] zf = new double[4]{-252.4, -128.4, 13.1, 134.1};

			//Box[,,] boxes = new Box[3,4,4];

			for (int ix=0; ix<3; ix++)
				for (int iy=0; iy<4; iy++)
					for (int iz=0; iz<4; iz++)
					{
						Box box = new Box(xi[ix], yi[iy], zi[iz], xf[ix], yf[iy], zf[iz], (iz+1)*100 + (iy+1)+10 + ix+1, null, SftColor);
						//box.PosCode = (iz+1)*100 + (iy+1)+10 + ix+1;
						//box.Description =  ; 
						Boxes[ix, iy, iz] = (Box) box.Clone();
						/*foreach (GDI3D.Control.Line l in box.Lines)
						{
							gdiDisplay1.Add(l);
						}*/												
					}

			if (Bricks != null)
			{
				int nbricks = Bricks.Count;
				for (int i=0; i<nbricks; i++)
				{
					Bricks.Row = i;
					int poscode = (int) Bricks._POSITIONCODE;
					int ix, iy, iz;
					int[] temp = Decompose(poscode);
					iz = temp[0]; iy = temp[1]; ix = temp[2];
					Boxes[ix-1,iy-1,iz-1].BrickIndices.Add(i);
				}
			}

			for (int ix=0; ix<3; ix++)
				for (int iy=0; iy<4; iy++)
					for (int iz=0; iz<4; iz++)
					{
						foreach (GDI3D.Control.Line l in Boxes[ix, iy, iz].Lines)
						{
							gdiDisplay1.Add(l);
						}												
					}

			gdiDisplay1.AutoRender = true;
			gdiDisplay1.Distance = 1e6;
		}

		private void DrawExposed()
		{
			long id_event = 0;
			try
			{
				SelectedEvent = Convert.ToInt32(txtMaxEvent.Text);
				int run = Convert.ToInt32(cmbRun.SelectedItem);
				SelectedEvent += ((long) run*10000000);
				id_event = Convert.ToInt64((new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_PREDICTED_EVENTS WHERE EVENT = " + SelectedEvent, Conn).ExecuteScalar()));
				object temp =  (new SySal.OperaDb.OperaDbCommand("SELECT TO_CHAR(TIME, 'hh24:mi:ss dd-mm-yyyy') FROM TB_PREDICTED_EVENTS WHERE ID=" + id_event, Conn).ExecuteScalar());
				//ExposedBricks.SelectWhere("EX
				string time = "TO_DATE('" + temp.ToString() + "', 'hh24:mi:ss dd-mm-yyyy')";
				ExposedBricks = SySal.OperaDb.Schema.TB_PEANUT_BRICKINFO.SelectWhere("EXPOSURESTART<" + time + " AND EXPOSUREFINISH>" + time, "1");

			}
			catch(Exception ex)
			{
				string temp2 = ex.ToString();
				return;
			}
			if (!ckbExposed.Checked) return;
			gdiDisplay1.AutoRender = false;
			double[] xi = new double[3]{198, 69.9, -58.2};
			double[] xf = new double[3]{69.9, -58.2, -186.3};
			double[] yi = new double[4]{210, 105, 0, -105};
			double[] yf = new double[4]{105, 0, -105, -210};
			double[] zi = new double[4]{-332, -208, -66.5, 54.5};
			double[] zf = new double[4]{-252.4, -128.4, 13.1, 134.1};

			//Box[,,] boxes = new Box[3,4,4];

			
			if (Bricks != null)
			{
				int nbricks = ExposedBricks.Count;
				Box[] boxes = new Box[nbricks];
				for (int i=0; i<nbricks; i++)
				{
					ExposedBricks.Row = i;
					ArrayList lst = new ArrayList();
					lst.Add(i);
					long id_eventbrick = ExposedBricks._ID_EVENTBRICK;
					long poscode = ExposedBricks._POSITIONCODE;
					SySal.OperaDb.Schema.TB_EVENTBRICKS ev = null;
					try
					{
						ev = SySal.OperaDb.Schema.TB_EVENTBRICKS.SelectWhere("ID=" + id_eventbrick, "1");
					}
					catch(Exception ex)
					{
						string temp = ex.ToString();
					}
					ev.Row = 0;
					Box box = new Box(1E-3*ev._MINX, 1E-3*ev._MINY, 1E-3*ev._MINZ, 1E-3*ev._MAXX, 1E-3*ev._MAXY, 1E-3*ev._MAXZ, (int) poscode, lst, ExposedColor);
					
					//Box box = new Box(x0, y0, z1, x1, y1, z1, poscode, new ArrayList(i));
					
					foreach (GDI3D.Control.Line l in box.Lines)
					{
						l.R = ExposedColor.R;
						l.G = ExposedColor.G;
						l.B = ExposedColor.B;
						gdiDisplay1.Add(l);
					}												
					
				}
			}

			
			gdiDisplay1.AutoRender = true;
			gdiDisplay1.Distance = 1e6;
		}


		public int[] Decompose(int poscode)
		{
			if (poscode<=0 || poscode>1000) throw new Exception("Wrong argument for Decompose: " + poscode);
			int[] res = null;
			int hundreds = poscode/100;
			int rest = poscode - hundreds*100;
			int tens = rest/10;
			int ones = rest-tens*10;
			res = new int[3]{hundreds, tens, ones};
			return res;
		}


		private void btnReset_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.SetCameraSpotting(0.5,0.5,0.5);
			gdiDisplay1.Infinity = true;
			gdiDisplay1.Distance = 1000000;
			gdiDisplay1.Zoom = 1e-1;
			gdiDisplay1.SetCameraOrientation(0,1,0,0,0,-1);			
			gdiDisplay1.Render();
			gdiDisplay1.Refresh();
		}

		private void btnXY_Click(object sender, System.EventArgs e)
		{			
			gdiDisplay1.SetCameraOrientation(0, 0, 1, 0, 1, 0);			
			gdiDisplay1.Distance = 1e9;
			gdiDisplay1.Render();			
		}

		private void btnXZ_Click(object sender, System.EventArgs e)
		{		
			gdiDisplay1.SetCameraOrientation(0, -1, 0, 1, 0, 0);			
			gdiDisplay1.Distance = 1e9;
			gdiDisplay1.Render();						
		}

		

		private void comboBox1_SelectedValueChanged(object sender, System.EventArgs e)
		{
			SelectedProcopId = Convert.ToInt64(cmbProcopid.SelectedItem);
			SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT DISTINCT FLOOR(EVENT/1E7) FROM TB_PREDICTED_EVENTS WHERE ID_PROCESSOPERATION = " + SelectedProcopId + " ORDER BY 1", Conn);
			System.Data.DataSet ds = new DataSet();
			da.Fill(ds);
			int count = 0;
			SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("SELECT MAX(EVENT-1E7*:runId) as maxev FROM TB_PREDICTED_EVENTS WHERE ID_PROCESSOPERATION = " + SelectedProcopId + " AND EVENT>:runId*1E7 AND EVENT<(:runId+1)*1E7", Conn);
			//SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("SELECT MAX(EVENT)-1E7*3 FROM TB_PREDICTED_EVENTS WHERE ID_PROCESSOPERATION = " + SelectedProcopId + " AND EVENT>3*1E7 AND EVENT<(4)*1E7", Conn);
			cmd.Parameters.Add("runId", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
			Runs = new PeanutRun[ds.Tables[0].Rows.Count];
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				Runs[count] = new PeanutRun();
				Runs[count].Id = Convert.ToInt32(dr[0]);
				cmd.Parameters[0].Value = Runs[count].Id;
				Runs[count].MaxEvent = SySal.OperaDb.Convert.ToInt32(cmd.ExecuteScalar());
				cmbRun.Items.Add(Runs[count].Id);
				count++;
			}
			CurrEvent = 0;
			txtMaxEvent.Text = "";
		}

		private void cmbRun_SelectedValueChanged(object sender, System.EventArgs e)
		{
			lblMaxEvent.Text = "Type in event# (up to " + Runs[cmbRun.SelectedIndex].MaxEvent.ToString() + ")";
			CurrEvent = 0;
			txtMaxEvent.Text = "";
		}

		

		private void txtMaxEvent_Leave(object sender, System.EventArgs e)
		{
			if (cmbRun.SelectedIndex == -1) return;
			if (txtMaxEvent.Text.Trim() == "") return;
			gdiDisplay1.AutoRender = false;
			try
			{
				SelectedEvent = Convert.ToInt32(txtMaxEvent.Text);
			}
			catch
			{
				MessageBox.Show("Wrong input");
				return;
			}
			int run = Convert.ToInt32(cmbRun.SelectedItem);
			if (SelectedEvent > Runs[cmbRun.SelectedIndex].MaxEvent)
			{
				MessageBox.Show("Event # exceeds the maximum for this run");
				return;
			}
			SelectedEvent += ((long) run*10000000);
			
			CurrEvent = Convert.ToInt64((new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_PREDICTED_EVENTS WHERE EVENT = " + SelectedEvent, Conn).ExecuteScalar()));

			
			Hits = SySal.OperaDb.Schema.TB_PEANUT_HITS.SelectWhere("ID_EVENT =" + CurrEvent, "1");
			ProjTracks = SySal.OperaDb.Schema.TB_PEANUT_PREDTRACKS.SelectWhere("trk_id_x is null and ID_EVENT =" + CurrEvent, "1");
			ThreeDTracks = SySal.OperaDb.Schema.TB_PEANUT_PREDTRACKS.SelectWhere("trk_id_x is not null and ID_EVENT =" + CurrEvent, "1");

			/*gdiDisplay1.Clear();
			DrawSft();
			DrawHits();
			DrawProjTracks();
			DrawThreeDTracks();
			gdiDisplay1.AutoRender = true;
			btnDraw.Enabled = true;*/
			btnDraw.Enabled = true;
			DrawAll();
		}
		
		public void DrawHits()
		{
			if (!ckbProjHits.Checked) return;
			int nhits = Hits.Count;
			for (int i=0; i<nhits; i++)
			{
				Hits.Row = i;
				double xmin = Xmin;
				double xmax = Xmax;
				double ymin = Ymin;
				double ymax = Ymax;
				double z = 1E-3*Hits._Z;
				if (Hits._PROJ_ID == 'X') xmin =  xmax = 1E-3*Hits._TCOORD;
				else if (Hits._PROJ_ID == 'Y') ymin = ymax = 1E-3*Hits._TCOORD;
				else continue;
				Hit h = new Hit(i);
				gdiDisplay1.Add(new GDI3D.Control.Line(xmin, ymin, z, xmax, ymax, z, h, ProjHitColor.R, ProjHitColor.G, ProjHitColor.B));
				gdiDisplay1.Add(new GDI3D.Control.Point(xmin, ymin, z, h, ProjHitColor.R, ProjHitColor.G, ProjHitColor.B));
				gdiDisplay1.Add(new GDI3D.Control.Point(xmax, ymax, z, h, ProjHitColor.R, ProjHitColor.G, ProjHitColor.B));
				gdiDisplay1.Distance = 1e9;
			}
		}

		public void DrawProjTracks()
		{
			gdiDisplay1.AutoRender = false;
			if (!ckbProjTracks.Checked) return;
			int ntracks = ProjTracks.Count;
			for (int i=0; i<ntracks; i++)
			{
				ProjTracks.Row = i;
				char proj_id = ProjTracks._PROJ_ID;
				if (proj_id == 'U') return;
				double a = 1E-3*ProjTracks._ACOORD;
				double b = 1E-3*ProjTracks._BCOORD;
				GDI3D.Control.Line[] lines = new GDI3D.Control.Line[4];
				if (proj_id == 'X') lines = GetXProjTrack(a, b, (int) ProjTracks._TRACK_ID);
				else if (proj_id == 'Y') lines = GetYProjTrack(a, b, (int) ProjTracks._TRACK_ID);
				if (lines == null) continue;
				foreach (GDI3D.Control.Line l in lines)
				{
					l.Owner = new ProjTrack(i);
					gdiDisplay1.Add(l);
				}
			}
			gdiDisplay1.AutoRender = true;
		}

		public void DrawThreeDTracks()
		{
			gdiDisplay1.AutoRender = false;
			if (!ckb3dTracks.Checked) return;
			int ntracks = ThreeDTracks.Count;
			//SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("SELECT ACOORD, BCOORD FROM TB_PEANUT_PREDTRACKS WHERE TRACK_ID=:track_id AND ID_EVENT=:id_event", Conn);
			SySal.OperaDb.OperaDbDataAdapter dax, day;
			//double 
						
			for (int i=0; i<ntracks; i++)
			{
				ThreeDTracks.Row = i;
				dax = new SySal.OperaDb.OperaDbDataAdapter("SELECT ACOORD, BCOORD FROM TB_PEANUT_PREDTRACKS WHERE PROJ_ID='X' AND ID_EVENT=" + ThreeDTracks._ID_EVENT + " AND TRACK_ID=" + ThreeDTracks._TRK_ID_X, Conn);
				day = new SySal.OperaDb.OperaDbDataAdapter("SELECT ACOORD, BCOORD FROM TB_PEANUT_PREDTRACKS WHERE PROJ_ID='Y' AND ID_EVENT=" + ThreeDTracks._ID_EVENT + " AND TRACK_ID=" + ThreeDTracks._TRK_ID_Y, Conn);
				System.Data.DataSet dsx = new DataSet();
				System.Data.DataSet dsy = new DataSet();
				dax.Fill(dsx);
				day.Fill(dsy);
				int ntrk = dsx.Tables[0].Rows.Count;
				for (int j=0; j<ntrk; j++)
				{
					double ax = 1E-3*Convert.ToDouble(dsx.Tables[0].Rows[j][0]);
					double bx = 1E-3*Convert.ToDouble(dsx.Tables[0].Rows[j][1]);
					double ay = 1E-3*Convert.ToDouble(dsy.Tables[0].Rows[j][0]);
					double by = 1E-3*Convert.ToDouble(dsy.Tables[0].Rows[j][1]);
					GDI3D.Control.Line l = GetLine(ax, ay, bx, by, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);
					if (l!=null) 
					{
						l.Owner = new ThreeDTrack(i);
						gdiDisplay1.Add(l);
					}
				}
				/*foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				{

				}*/
				//gdiDisplay1.Add(GetLine(ax, ay, bx, by, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax));
				//cmd.Ex
			}

			gdiDisplay1.AutoRender = true;
		}

		public GDI3D.Control.Line GetLine(double ax, double ay, double bx, double by, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
		{
			GDI3D.Control.Point[] points = new GDI3D.Control.Point[6];
			GDI3D.Control.Point[] result = new GDI3D.Control.Point[2];
			int k = 0;
			for (int i=0; i<6; i++) points[i] = new GDI3D.Control.Point(0, 0, 0, null, 0, 0, 0);
			for (int i=0; i<2; i++) result[i] = new GDI3D.Control.Point(0, 0, 0, null, 0, 0, 0);
			points[0].X = xmin; points[0].Y = ay*(xmin-bx)/ax + by; points[0].Z = (xmin-bx)/ax;
			points[1].X = xmax; points[1].Y = ay*(xmax-bx)/ax + by; points[1].Z = (xmax-bx)/ax;
			points[2].X = ax*(ymin-by)/ay + bx; points[2].Y = ymin; points[2].Z = (ymin-by)/ay;
			points[3].X = ax*(ymax-by)/ay + bx; points[3].Y = ymax; points[3].Z = (ymax-by)/ay;
			points[4].X = ax*zmin+bx; points[4].Y = ay*zmin+by; points[4].Z = zmin;
			points[5].X = ax*zmax+bx; points[5].Y = ay*zmax+by; points[5].Z = zmax;
			foreach (GDI3D.Control.Point p in points)
			{
				if (p.X >= xmin && p.X <= xmax && p.Y >= ymin && p.Y <= ymax && p.Z >= zmin && p.Z <= zmax) result[k++] = p;
			}
			if (k<2) return null;
			else if (k>2) {throw new Exception("cannot have more than 2 intersections of a ray with a cube!"); return null;}
			else return new GDI3D.Control.Line(result[0].X, result[0].Y, result[0].Z, result[1].X, result[1].Y, result[1].Z, null, ThreeDTrackColor.R, ThreeDTrackColor.G, ThreeDTrackColor.B);
		}

		/*public GDI3D.Control.Line GetLine(double ax, double ay, double bx, double by, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
		{
			GDI3D.Control.Point[] points = new GDI3D.Control.Point[6];
			GDI3D.Control.Point[] result = new GDI3D.Control.Point[2];
			int k = 0;
			for (int i=0; i<6; i++) points[i] = new GDI3D.Control.Point(0, 0, 0, null, 0, 0, 0);
			for (int i=0; i<2; i++) result[i] = new GDI3D.Control.Point(0, 0, 0, null, 0, 0, 0);
			points[0].X = (zmin - bx)/ax; points[0].Y = (zmin - by)/ay; points[0].Z = zmin;
			points[1].X = (zmax - bx)/ax; points[1].Y = (zmax - by)/ay; points[1].Z = zmax;
			points[2].X = xmin; points[2].Y = (ax*xmin + bx - by)/ay; points[2].Z = ax*xmin + bx;
			points[3].X = xmax; points[3].Y = (ax*xmax + bx - by)/ay; points[3].Z = ax*xmax + bx;
			points[4].X = (ay*ymin + by - bx)/ax; points[4].Y = ymin; points[4].Z = ay*ymin + by;
			points[5].X = (ay*ymax + by - bx)/ax; points[5].Y = ymax; points[5].Z = ay*ymax + by;
			foreach (GDI3D.Control.Point p in points)
			{
				if (p.X >= xmin && p.X <= xmax && p.Y >= ymin && p.Y <= ymax && p.Z >= zmin && p.Z <= zmax) result[k++] = p;
			}
			if (k<2) return null;
			else if (k>2) {throw new Exception("cannot have more than 2 intersections of a ray with a cube!"); return null;}
			else return new GDI3D.Control.Line(result[0].X, result[0].Y, result[0].Z, result[1].X, result[1].Y, result[1].Z, null, ThreeDTrackColor.R, ThreeDTrackColor.G, ThreeDTrackColor.B);
		}*/

		/*public GDI3D.Control.Line[] GetXProjTrack(double ax, double bx, int index)
		{
			GDI3D.Control.Line[] result = new GDI3D.Control.Line[4];

			double y1 = Ymin;
			double y2 = Ymax;

			double z1 = ax*Xmin + bx;
			double z2 = ax*Xmax + bx;
			double x1 = (Zmin-bx)/ax;
			double x2 = (Zmax-bx)/ax;

			double[] x = new double[2];
			double[] y = new double[2]{Ymin, Ymax};
			double[] z = new double[2];

			SySal.BasicTypes.Vector2[] intersec = new SySal.BasicTypes.Vector2[2];
			for (int i=0; i<2; i++) intersec[i] = new SySal.BasicTypes.Vector2();
			int k=0;

			
			if (x1 > Xmin && x1 < Xmax) 
			{	
				x[k] = x1; z[k] = Zmin;
				k++;
			}
			if (x2 > Xmin && x2 < Xmax)
			{
				x[k] = x2; z[k] = Zmax;
				k++;
			}
			if (z1 > Zmin && z1 < Zmax)
			{
				x[k] = Xmin; z[k] = z1;
				k++;
			}
			if (z2 > Zmin && z2 < Zmax)
			{
				x[k] = Xmax; z[k] = z2;
				k++;
			}
			if (k<2) return null;
			else if (k>2) throw new Exception("Cannot have more than 2 intersections of a line with a square!");
			else 
			{
				//else throw new Exception("no intersection!");
				result[0] = new GDI3D.Control.Line(x[0], y[0], z[0], x[0], y[1], z[0], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				result[1] = new GDI3D.Control.Line(x[1], y[0], z[1], x[1], y[1], z[1], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				result[2] = new GDI3D.Control.Line(x[0], y[0], z[0], x[1], y[0], z[1], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				result[3] = new GDI3D.Control.Line(x[0], y[1], z[0], x[1], y[1], z[1], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				return result;
			}
		}*/

		public GDI3D.Control.Line[] GetXProjTrack(double ax, double bx, int index)
		{
			GDI3D.Control.Line[] result = new GDI3D.Control.Line[4];

			double y1 = Ymin;
			double y2 = Ymax;

			double z1 = (Xmin-bx)/ax;
			double z2 = (Xmax-bx)/ax;
			double x1 = ax*Zmin + bx;
			double x2 = ax*Zmax + bx;

			double[] x = new double[2];
			double[] y = new double[2]{Ymin, Ymax};
			double[] z = new double[2];

			SySal.BasicTypes.Vector2[] intersec = new SySal.BasicTypes.Vector2[2];
			for (int i=0; i<2; i++) intersec[i] = new SySal.BasicTypes.Vector2();
			int k=0;

			
			if (x1 > Xmin && x1 < Xmax) 
			{	
				x[k] = x1; z[k] = Zmin;
				k++;
			}
			if (x2 > Xmin && x2 < Xmax)
			{
				x[k] = x2; z[k] = Zmax;
				k++;
			}
			if (z1 > Zmin && z1 < Zmax)
			{
				x[k] = Xmin; z[k] = z1;
				k++;
			}
			if (z2 > Zmin && z2 < Zmax)
			{
				x[k] = Xmax; z[k] = z2;
				k++;
			}
			if (k<2) return null;
			else if (k>2) throw new Exception("Cannot have more than 2 intersections of a line with a square!");
			else 
			{
				//else throw new Exception("no intersection!");
				result[0] = new GDI3D.Control.Line(x[0], y[0], z[0], x[0], y[1], z[0], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				result[1] = new GDI3D.Control.Line(x[1], y[0], z[1], x[1], y[1], z[1], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				result[2] = new GDI3D.Control.Line(x[0], y[0], z[0], x[1], y[0], z[1], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				result[3] = new GDI3D.Control.Line(x[0], y[1], z[0], x[1], y[1], z[1], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				return result;
			}
		}

/*		public GDI3D.Control.Line[] GetYProjTrack(double ay, double by, int index)
		{
			GDI3D.Control.Line[] result = new GDI3D.Control.Line[4];

			double x1 = Xmin;
			double x2 = Xmax;

			double z1 = ay*Ymin + by;
			double z2 = ay*Ymax + by;
			double y1 = (Zmin-by)/ay;
			double y2 = (Zmax-by)/ay;

			double[] x = new double[2]{Xmin, Xmax};
			double[] y = new double[2];
			double[] z = new double[2];

			int k=0;
			
			if (y1 > Ymin && y1 < Ymax) 
			{	
				y[k] = y1; z[k] = Zmin;
				k++;
			}
			if (y2 > Ymin && y2 < Ymax)
			{
				y[k] = y2; z[k] = Zmax;
				k++;
			}
			if (z1 > Zmin && z1 < Zmax)
			{
				y[k] = Ymin; z[k] = z1;
				k++;
			}
			if (z2 > Zmin && z2 < Zmax)
			{
				y[k] = Ymax; z[k] = z2;
				k++;
			}
			if (k<2) return null;
			else if (k>2) throw new Exception("Cannot have more than 2 intersections of a line with a square!");
			else 
			{
				//else throw new Exception("no intersection!");
				result[0] = new GDI3D.Control.Line(x[0], y[0], z[0], x[1], y[0], z[0], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				result[1] = new GDI3D.Control.Line(x[0], y[1], z[1], x[1], y[1], z[1], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				result[2] = new GDI3D.Control.Line(x[0], y[0], z[0], x[0], y[1], z[1], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				result[3] = new GDI3D.Control.Line(x[1], y[0], z[0], x[1], y[1], z[1], index, ProjTrackColor.R/2, ProjTrackColor.G/2, ProjTrackColor.B/2);
				return result;
			}
		}
 */

		public GDI3D.Control.Line[] GetYProjTrack(double ay, double by, int index)
		{
			GDI3D.Control.Line[] result = new GDI3D.Control.Line[4];

			double x1 = Xmin;
			double x2 = Xmax;

			double y1 = ay*Zmin + by;
			double y2 = ay*Zmax + by;
			double z1 = (Ymin-by)/ay;
			double z2 = (Ymax-by)/ay;

			double[] x = new double[2]{Xmin, Xmax};
			double[] y = new double[2];
			double[] z = new double[2];

			int k=0;
			
			if (y1 > Ymin && y1 < Ymax) 
			{	
				y[k] = y1; z[k] = Zmin;
				k++;
			}
			if (y2 > Ymin && y2 < Ymax)
			{
				y[k] = y2; z[k] = Zmax;
				k++;
			}
			if (z1 > Zmin && z1 < Zmax)
			{
				y[k] = Ymin; z[k] = z1;
				k++;
			}
			if (z2 > Zmin && z2 < Zmax)
			{
				y[k] = Ymax; z[k] = z2;
				k++;
			}
			if (k<2) return null;
			else if (k>2) throw new Exception("Cannot have more than 2 intersections of a line with a square!");
			else 
			{
				//else throw new Exception("no intersection!");
				result[0] = new GDI3D.Control.Line(x[0], y[0], z[0], x[1], y[0], z[0], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				result[1] = new GDI3D.Control.Line(x[0], y[1], z[1], x[1], y[1], z[1], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				result[2] = new GDI3D.Control.Line(x[0], y[0], z[0], x[0], y[1], z[1], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				result[3] = new GDI3D.Control.Line(x[1], y[0], z[0], x[1], y[1], z[1], index, ProjTrackColor.R, ProjTrackColor.G, ProjTrackColor.B);
				return result;
			}
		}

		private void btnZoomIn_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Zoom *= 1.4;
		}

		private void btnZoomOut_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Zoom /= 1.4;
		}

		private void lblRun_Click(object sender, System.EventArgs e)
		{
		
		}

		private void btnPan_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Pan;
		}

		private void btnRotate_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
		}

		private void btnFocus_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.NextClickSetsCenter = true;
		}

		private void btnConnect_Click(object sender, System.EventArgs e)
		{
			frmLogin loginForm = new frmLogin();
			System.Windows.Forms.DialogResult res = loginForm.ShowDialog();
			if (res != System.Windows.Forms.DialogResult.OK) return;
			FillCombo();
			
		}

		public void FillCombo()
		{
			cmbProcopid.Items.Clear();
			SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter(@"SELECT DISTINCT ID_PROCESSOPERATION FROM TB_PREDICTED_EVENTS", Conn);
			System.Data.DataSet ds = new System.Data.DataSet();
			da.Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				cmbProcopid.Items.Add(dr[0]);
			}
			Bricks = SySal.OperaDb.Schema.TB_PEANUT_BRICKINFO.SelectWhere("1=1", "1");
		}

		private void btnDraw_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Distance = 1e9;
			gdiDisplay1.Render();
		}

		private void btnProjHitsColor_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() != DialogResult.OK) return;
			ProjHitColor = colorDialog1.Color;
			btnProjHitsColor.BackColor = ProjHitColor;
			DrawAll();
		}

		private void btnProjTracksColor_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() != DialogResult.OK) return;
			ProjTrackColor = colorDialog1.Color;
			btnProjTracksColor.BackColor = ProjTrackColor;
			DrawAll();
		}

		private void btn3dTracksColor_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() != DialogResult.OK) return;
			ThreeDTrackColor = colorDialog1.Color;
			btn3dTracksColor.BackColor = ThreeDTrackColor;
			DrawAll();
		}

		private void ckbProjHits_CheckedChanged(object sender, System.EventArgs e)
		{
			/*gdiDisplay1.Clear();
			gdiDisplay1.AutoRender = false;
			DrawAll();
			gdiDisplay1.AutoRender = true;
			gdiDisplay1.Distance = 1e9;*/
			DrawAll();
		}

		public void DrawAll()
		{
			if (CurrEvent == 0) return;
			gdiDisplay1.Clear();
			gdiDisplay1.AutoRender = false;
			DrawSft();
			DrawHits();
			DrawProjTracks();
			DrawThreeDTracks();
			DrawExposed();
			gdiDisplay1.AutoRender = true;
			gdiDisplay1.Distance = 1e9;
			
		}

		private void ckbProjTracks_CheckedChanged(object sender, System.EventArgs e)
		{
			//gdiDisplay1.Clear();
			DrawAll();
			//gdiDisplay1.Distance = 1e9;
		}

		private void ckb3dTracks_CheckedChanged(object sender, System.EventArgs e)
		{
			DrawAll();
		}

		private void ckbBricks_CheckedChanged(object sender, System.EventArgs e)
		{
			DrawAll();
		}

		private void frmMain_Load(object sender, System.EventArgs e)
		{
			Settings = new AppSettings();
			ReadSettings();
			if (Settings != null)
			{
				try
				{
					SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
					/*SySal.OperaDb.OperaDbCredentials cred = new SySal.OperaDb.OperaDbCredentials();
					cred.DBPassword = Settings.Password;
					cred.DBServer = Settings.DbServer;
					cred.DBUserName = Settings.UserName;*/
					Conn = cred.Connect();
					Conn.Open();
					SySal.OperaDb.Schema.DB = Conn;
					FillCombo();
					statusBar1.Text = "Connected";
				}
				catch(Exception ex)
				{
					string temp = ex.ToString();
				}
			}
		}

		private void btnSave_Click(object sender, System.EventArgs e)
		{
			if (saveFileDialog1.ShowDialog() == DialogResult.OK)
			{
				try
				{
					gdiDisplay1.Save(saveFileDialog1.FileName);
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "Can't save file", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
			}
		}

		private void frmMain_KeyDown(object sender, System.Windows.Forms.KeyEventArgs e)
		{
			if (e.KeyValue == 13) txtMaxEvent_Leave(sender, null);
		}

		private void ckbExposed_CheckedChanged(object sender, System.EventArgs e)
		{
			DrawAll();
		}

		private void btnBricks_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() != DialogResult.OK) return;
			SftColor = colorDialog1.Color;
			btnBricks.BackColor = SftColor;
			DrawAll();		
		}

		private void btnExposed_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() != DialogResult.OK) return;
			ExposedColor = colorDialog1.Color;
			btnExposed.BackColor = ExposedColor;
			DrawAll();		
			
		}

		private void btnYZ_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.SetCameraOrientation(1, 0, 0, 0, 1, 0);			
			gdiDisplay1.Distance = 1e9;
			gdiDisplay1.Render();						
		
		}

		private void frmMain_Closed(object sender, System.EventArgs e)
		{
			SaveSettings();
		}

		/*	private void txtMaxEvent_KeyPress(object sender, System.Windows.Forms.KeyPressEventArgs e)
			{
				if (e.KeyChar == '\n') txtMaxEvent_Leave(sender, null);
			}*/

		/*private void txtMaxEvent_Enter(object sender, System.EventArgs e)
		{
			txtMaxEvent_Leave(sender, e);
		}*/

		
		
	}


}
