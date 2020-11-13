using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.X3LView
{
	/// <summary>
	/// X3LView - Viewing application for generic X3L files.
	/// </summary>
	/// <remarks>
	/// <para>X3LView is a thin viewing application to work with graphic files in the <see cref="GDI3D.Scene">X3L</see> format.</para>
	/// <para>X3L files contain analysis information together with the graphics, so they are a powerful help to work in the absence of a connected DB or other data source.</para>
	/// <para>The following actions are available:
	/// <list type="table">
	/// <listheader><term>Control</term><description>Action</description></listheader>
	/// <item><term><b>Load</b> (button)</term><description>Loads a new X3L file.</description></item>
	/// <item><term><b>Merge</b> (button)</term><description>Loads a new X3L file and merges its contents with the current contents.</description></item>
	/// <item><term><b>Save</b> (button)</term><description>Saves the contents to a new X3L file or to common 2D graphics formats.</description></item>
	/// <item><term><b>+</b> (button)</term><description>Zooms in.</description></item>
	/// <item><term><b>-</b> (button)</term><description>Zooms out.</description></item>
	/// <item><term><b>Isometric</b> (check flag)</term><description>If checked, the view is isometric and distance shrinking is not applied; if false, perspective viewing with distance shrinking is applied. Normally, the zoom factors needed to obtain a reasonable picture are very different between isometric and perspective viewing.</description></item>
	/// <item><term><b>Set Focus</b> (button)</term><description>After clicking this button, the next <u>left click</u> with the mouse buttons will set the rotation center on the object being clicked.</description></item>
	/// <item><term><b>Pan</b> (button)</term><description>After clicking this button, dragging the view with the <u>right mouse button</u> will translate the content.</description></item>
	/// <item><term><b>Rot</b> (button)</term><description>After clicking this button, dragging the view with the <u>right mouse button</u> will rotate the content.</description></item>
	/// <item><term><b>Alpha</b> (slider)</term><description>This slider control the level of opaqueness of lines and points.</description></item>
	/// <item><term><b>Lines</b> (slider)</term><description>This slider control the thickness of lines.</description></item>
	/// <item><term><b>Points</b> (slider)</term><description>This slider control the size of points.</description></item>
	/// <item><term><b>Bkgnd</b> (button)</term><description>Sets the background color.</description></item>
    /// <item><term><b>Kill by owner</b> (button)</term><description>When clicked, it enables "killing" mode: every subsequent double-click will kill the nearest graphical object with non-null owner. Clicking again will switch to normal mode (see below).</description></item>
	/// </list>
	/// Double-clicking with the <u>left mouse button</u> on an object will display its owner information, if available.
	/// </para>
	/// </remarks>
	public class X3LView : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button LoadButton;
		private System.Windows.Forms.Button SaveButton;
		private System.Windows.Forms.Button ZoomInButton;
		private System.Windows.Forms.Button ZoomOutButton;
		private GDI3D.Control.GDIDisplay gdiDisplay1;
		private System.Windows.Forms.TrackBar AlphaTrack;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TrackBar LinesTrack;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.TrackBar PointsTrack;
		private System.Windows.Forms.Button MergeButton;
		private System.Windows.Forms.CheckBox IsometricButton;
		private System.Windows.Forms.Button SetFocusButton;
		private System.Windows.Forms.Button PanButton;
		private System.Windows.Forms.Button RotButton;
		private System.Windows.Forms.Button BkgndButton;
		private System.Windows.Forms.ColorDialog colorDialog1;
        private Button XYbutton;
        private Button XZbutton;
        private Button YZbutton;
        private Button RYZbutton;
        private Button RXZbutton;
        private Button RXYbutton;
        private Button KillByOwnerButton;
        private Button QuickDisplayButton;
        private Button RecolorButton;
		
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public X3LView()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
            gdiDisplay1.DoubleClickSelect = new GDI3D.Control.SelectObject(OnDoubleClickSelect);
			gdiDisplay1.Alpha = (AlphaTrack.Value = 4) / 8.0;
			gdiDisplay1.LineWidth = LinesTrack.Value = 1;
			gdiDisplay1.PointSize = PointsTrack.Value = 5;
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(X3LView));
            this.LoadButton = new System.Windows.Forms.Button();
            this.SaveButton = new System.Windows.Forms.Button();
            this.ZoomInButton = new System.Windows.Forms.Button();
            this.ZoomOutButton = new System.Windows.Forms.Button();
            this.AlphaTrack = new System.Windows.Forms.TrackBar();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.LinesTrack = new System.Windows.Forms.TrackBar();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.PointsTrack = new System.Windows.Forms.TrackBar();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.MergeButton = new System.Windows.Forms.Button();
            this.IsometricButton = new System.Windows.Forms.CheckBox();
            this.SetFocusButton = new System.Windows.Forms.Button();
            this.PanButton = new System.Windows.Forms.Button();
            this.RotButton = new System.Windows.Forms.Button();
            this.BkgndButton = new System.Windows.Forms.Button();
            this.colorDialog1 = new System.Windows.Forms.ColorDialog();
            this.XYbutton = new System.Windows.Forms.Button();
            this.XZbutton = new System.Windows.Forms.Button();
            this.YZbutton = new System.Windows.Forms.Button();
            this.RYZbutton = new System.Windows.Forms.Button();
            this.RXZbutton = new System.Windows.Forms.Button();
            this.RXYbutton = new System.Windows.Forms.Button();
            this.KillByOwnerButton = new System.Windows.Forms.Button();
            this.gdiDisplay1 = new GDI3D.Control.GDIDisplay();
            this.QuickDisplayButton = new System.Windows.Forms.Button();
            this.RecolorButton = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.AlphaTrack)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.LinesTrack)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PointsTrack)).BeginInit();
            this.SuspendLayout();
            // 
            // LoadButton
            // 
            this.LoadButton.Location = new System.Drawing.Point(4, 12);
            this.LoadButton.Name = "LoadButton";
            this.LoadButton.Size = new System.Drawing.Size(80, 24);
            this.LoadButton.TabIndex = 1;
            this.LoadButton.Text = "Load";
            this.LoadButton.Click += new System.EventHandler(this.LoadButton_Click);
            // 
            // SaveButton
            // 
            this.SaveButton.Location = new System.Drawing.Point(4, 76);
            this.SaveButton.Name = "SaveButton";
            this.SaveButton.Size = new System.Drawing.Size(80, 24);
            this.SaveButton.TabIndex = 3;
            this.SaveButton.Text = "Save";
            this.SaveButton.Click += new System.EventHandler(this.SaveButton_Click);
            // 
            // ZoomInButton
            // 
            this.ZoomInButton.Location = new System.Drawing.Point(4, 108);
            this.ZoomInButton.Name = "ZoomInButton";
            this.ZoomInButton.Size = new System.Drawing.Size(24, 24);
            this.ZoomInButton.TabIndex = 4;
            this.ZoomInButton.Text = "+";
            this.ZoomInButton.Click += new System.EventHandler(this.ZoomInButton_Click);
            // 
            // ZoomOutButton
            // 
            this.ZoomOutButton.Location = new System.Drawing.Point(44, 108);
            this.ZoomOutButton.Name = "ZoomOutButton";
            this.ZoomOutButton.Size = new System.Drawing.Size(24, 24);
            this.ZoomOutButton.TabIndex = 5;
            this.ZoomOutButton.Text = "-";
            this.ZoomOutButton.Click += new System.EventHandler(this.ZoomOutButton_Click);
            // 
            // AlphaTrack
            // 
            this.AlphaTrack.LargeChange = 2;
            this.AlphaTrack.Location = new System.Drawing.Point(20, 342);
            this.AlphaTrack.Maximum = 8;
            this.AlphaTrack.Name = "AlphaTrack";
            this.AlphaTrack.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.AlphaTrack.Size = new System.Drawing.Size(45, 48);
            this.AlphaTrack.TabIndex = 10;
            this.AlphaTrack.TickFrequency = 0;
            this.AlphaTrack.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.AlphaTrack.Value = 4;
            this.AlphaTrack.ValueChanged += new System.EventHandler(this.OnAlphaChanged);
            // 
            // groupBox1
            // 
            this.groupBox1.Location = new System.Drawing.Point(12, 324);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(56, 72);
            this.groupBox1.TabIndex = 7;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Alpha";
            // 
            // LinesTrack
            // 
            this.LinesTrack.LargeChange = 2;
            this.LinesTrack.Location = new System.Drawing.Point(20, 416);
            this.LinesTrack.Maximum = 4;
            this.LinesTrack.Minimum = 1;
            this.LinesTrack.Name = "LinesTrack";
            this.LinesTrack.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.LinesTrack.Size = new System.Drawing.Size(45, 48);
            this.LinesTrack.TabIndex = 11;
            this.LinesTrack.TickFrequency = 0;
            this.LinesTrack.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.LinesTrack.Value = 1;
            this.LinesTrack.ValueChanged += new System.EventHandler(this.OnLinesChanged);
            // 
            // groupBox2
            // 
            this.groupBox2.Location = new System.Drawing.Point(12, 399);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(56, 72);
            this.groupBox2.TabIndex = 9;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Lines";
            // 
            // PointsTrack
            // 
            this.PointsTrack.LargeChange = 2;
            this.PointsTrack.Location = new System.Drawing.Point(20, 491);
            this.PointsTrack.Minimum = 1;
            this.PointsTrack.Name = "PointsTrack";
            this.PointsTrack.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.PointsTrack.Size = new System.Drawing.Size(45, 48);
            this.PointsTrack.TabIndex = 12;
            this.PointsTrack.TickFrequency = 0;
            this.PointsTrack.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.PointsTrack.Value = 5;
            this.PointsTrack.ValueChanged += new System.EventHandler(this.OnPointsChanged);
            // 
            // groupBox3
            // 
            this.groupBox3.Location = new System.Drawing.Point(12, 475);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(56, 72);
            this.groupBox3.TabIndex = 11;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Points";
            // 
            // MergeButton
            // 
            this.MergeButton.Location = new System.Drawing.Point(4, 44);
            this.MergeButton.Name = "MergeButton";
            this.MergeButton.Size = new System.Drawing.Size(80, 24);
            this.MergeButton.TabIndex = 2;
            this.MergeButton.Text = "Merge";
            this.MergeButton.Click += new System.EventHandler(this.MergeButton_Click);
            // 
            // IsometricButton
            // 
            this.IsometricButton.Location = new System.Drawing.Point(4, 138);
            this.IsometricButton.Name = "IsometricButton";
            this.IsometricButton.Size = new System.Drawing.Size(72, 24);
            this.IsometricButton.TabIndex = 6;
            this.IsometricButton.Text = "Isometric";
            this.IsometricButton.CheckedChanged += new System.EventHandler(this.IsometricButton_CheckedChanged);
            // 
            // SetFocusButton
            // 
            this.SetFocusButton.Location = new System.Drawing.Point(4, 164);
            this.SetFocusButton.Name = "SetFocusButton";
            this.SetFocusButton.Size = new System.Drawing.Size(80, 24);
            this.SetFocusButton.TabIndex = 7;
            this.SetFocusButton.Text = "Set Focus";
            this.SetFocusButton.Click += new System.EventHandler(this.SetFocusButton_Click);
            // 
            // PanButton
            // 
            this.PanButton.Location = new System.Drawing.Point(4, 196);
            this.PanButton.Name = "PanButton";
            this.PanButton.Size = new System.Drawing.Size(40, 24);
            this.PanButton.TabIndex = 8;
            this.PanButton.Text = "Pan";
            this.PanButton.Click += new System.EventHandler(this.PanButton_Click);
            // 
            // RotButton
            // 
            this.RotButton.Location = new System.Drawing.Point(45, 196);
            this.RotButton.Name = "RotButton";
            this.RotButton.Size = new System.Drawing.Size(40, 24);
            this.RotButton.TabIndex = 9;
            this.RotButton.Text = "Rot";
            this.RotButton.Click += new System.EventHandler(this.RotButton_Click);
            // 
            // BkgndButton
            // 
            this.BkgndButton.Location = new System.Drawing.Point(4, 555);
            this.BkgndButton.Name = "BkgndButton";
            this.BkgndButton.Size = new System.Drawing.Size(80, 24);
            this.BkgndButton.TabIndex = 14;
            this.BkgndButton.Text = "Bkgnd";
            this.BkgndButton.Click += new System.EventHandler(this.BkgndButton_Click);
            // 
            // colorDialog1
            // 
            this.colorDialog1.AnyColor = true;
            this.colorDialog1.Color = System.Drawing.Color.White;
            // 
            // XYbutton
            // 
            this.XYbutton.Location = new System.Drawing.Point(4, 229);
            this.XYbutton.Name = "XYbutton";
            this.XYbutton.Size = new System.Drawing.Size(40, 24);
            this.XYbutton.TabIndex = 15;
            this.XYbutton.Text = "XY";
            this.XYbutton.Click += new System.EventHandler(this.XYbutton_Click);
            // 
            // XZbutton
            // 
            this.XZbutton.Location = new System.Drawing.Point(4, 259);
            this.XZbutton.Name = "XZbutton";
            this.XZbutton.Size = new System.Drawing.Size(40, 24);
            this.XZbutton.TabIndex = 16;
            this.XZbutton.Text = "XZ";
            this.XZbutton.Click += new System.EventHandler(this.XZbutton_Click);
            // 
            // YZbutton
            // 
            this.YZbutton.Location = new System.Drawing.Point(4, 289);
            this.YZbutton.Name = "YZbutton";
            this.YZbutton.Size = new System.Drawing.Size(40, 24);
            this.YZbutton.TabIndex = 17;
            this.YZbutton.Text = "YZ";
            this.YZbutton.Click += new System.EventHandler(this.YZbutton_Click);
            // 
            // RYZbutton
            // 
            this.RYZbutton.Location = new System.Drawing.Point(45, 289);
            this.RYZbutton.Name = "RYZbutton";
            this.RYZbutton.Size = new System.Drawing.Size(40, 24);
            this.RYZbutton.TabIndex = 20;
            this.RYZbutton.Text = "-YZ";
            this.RYZbutton.Click += new System.EventHandler(this.RYZbutton_Click);
            // 
            // RXZbutton
            // 
            this.RXZbutton.Location = new System.Drawing.Point(45, 259);
            this.RXZbutton.Name = "RXZbutton";
            this.RXZbutton.Size = new System.Drawing.Size(40, 24);
            this.RXZbutton.TabIndex = 19;
            this.RXZbutton.Text = "-XZ";
            this.RXZbutton.Click += new System.EventHandler(this.RXZbutton_Click);
            // 
            // RXYbutton
            // 
            this.RXYbutton.Location = new System.Drawing.Point(45, 229);
            this.RXYbutton.Name = "RXYbutton";
            this.RXYbutton.Size = new System.Drawing.Size(40, 24);
            this.RXYbutton.TabIndex = 18;
            this.RXYbutton.Text = "-XY";
            this.RXYbutton.Click += new System.EventHandler(this.RXYbutton_Click);
            // 
            // KillByOwnerButton
            // 
            this.KillByOwnerButton.Location = new System.Drawing.Point(4, 613);
            this.KillByOwnerButton.Name = "KillByOwnerButton";
            this.KillByOwnerButton.Size = new System.Drawing.Size(80, 24);
            this.KillByOwnerButton.TabIndex = 22;
            this.KillByOwnerButton.Text = "Kill by owner";
            this.KillByOwnerButton.Click += new System.EventHandler(this.KillByOwnerButton_Click);
            // 
            // gdiDisplay1
            // 
            this.gdiDisplay1.Alpha = 0.50196078431372548;
            this.gdiDisplay1.AutoRender = true;
            this.gdiDisplay1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))));
            this.gdiDisplay1.BorderWidth = 1;
            this.gdiDisplay1.ClickSelect = null;
            this.gdiDisplay1.Distance = 100;
            this.gdiDisplay1.DoubleClickSelect = null;
            this.gdiDisplay1.Infinity = false;
            this.gdiDisplay1.LabelFontName = "Arial";
            this.gdiDisplay1.LabelFontSize = 12;
            this.gdiDisplay1.LineWidth = 2;
            this.gdiDisplay1.Location = new System.Drawing.Point(91, 12);
            this.gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
            this.gdiDisplay1.MouseMultiplier = 0.01;
            this.gdiDisplay1.Name = "gdiDisplay1";
            this.gdiDisplay1.NextClickSetsCenter = false;
            this.gdiDisplay1.PointSize = 5;
            this.gdiDisplay1.Size = new System.Drawing.Size(897, 665);
            this.gdiDisplay1.TabIndex = 13;
            this.gdiDisplay1.Zoom = 2000;
            // 
            // QuickDisplayButton
            // 
            this.QuickDisplayButton.Location = new System.Drawing.Point(5, 646);
            this.QuickDisplayButton.Name = "QuickDisplayButton";
            this.QuickDisplayButton.Size = new System.Drawing.Size(80, 24);
            this.QuickDisplayButton.TabIndex = 23;
            this.QuickDisplayButton.Text = "Quick D!";
            this.QuickDisplayButton.Click += new System.EventHandler(this.QuickDisplayButton_Click);
            // 
            // RecolorButton
            // 
            this.RecolorButton.Location = new System.Drawing.Point(4, 584);
            this.RecolorButton.Name = "RecolorButton";
            this.RecolorButton.Size = new System.Drawing.Size(80, 24);
            this.RecolorButton.TabIndex = 24;
            this.RecolorButton.Text = "Recolor";
            this.RecolorButton.Click += new System.EventHandler(this.RecolorButton_Click);
            // 
            // X3LView
            // 
            this.AllowDrop = true;
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(993, 682);
            this.Controls.Add(this.RecolorButton);
            this.Controls.Add(this.QuickDisplayButton);
            this.Controls.Add(this.KillByOwnerButton);
            this.Controls.Add(this.RYZbutton);
            this.Controls.Add(this.RXZbutton);
            this.Controls.Add(this.RXYbutton);
            this.Controls.Add(this.YZbutton);
            this.Controls.Add(this.XZbutton);
            this.Controls.Add(this.XYbutton);
            this.Controls.Add(this.BkgndButton);
            this.Controls.Add(this.RotButton);
            this.Controls.Add(this.PanButton);
            this.Controls.Add(this.SetFocusButton);
            this.Controls.Add(this.IsometricButton);
            this.Controls.Add(this.MergeButton);
            this.Controls.Add(this.PointsTrack);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.LinesTrack);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.AlphaTrack);
            this.Controls.Add(this.gdiDisplay1);
            this.Controls.Add(this.ZoomOutButton);
            this.Controls.Add(this.ZoomInButton);
            this.Controls.Add(this.SaveButton);
            this.Controls.Add(this.LoadButton);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "X3LView";
            this.Text = "X3L Viewer";
            this.Load += new System.EventHandler(this.OnLoad);
            this.DragDrop += new System.Windows.Forms.DragEventHandler(this.OnDragDrop);
            ((System.ComponentModel.ISupportInitialize)(this.AlphaTrack)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.LinesTrack)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PointsTrack)).EndInit();
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
			Application.Run(new X3LView());
		}

        /// <summary>
        /// Sets the scene to be shown. This method can be called when X3LView is used as a module rather than an independent program.
        /// </summary>
        /// <param name="scene">the scene to be shown.</param>
        public void SetScene(GDI3D.Scene scene)
        {
            gdiDisplay1.SetScene(scene);
            gdiDisplay1.Transform();
            gdiDisplay1.Render();
        }

		void OnDoubleClickSelect(object sel)
		{
			OwnerView ow = new OwnerView();
			ow.SetText(sel.ToString());
			ow.Show();
		}

        void KillByOwner(object el)
        {
            gdiDisplay1.DeleteWithOwner(el);
            gdiDisplay1.Render();
        }

        void Recolor(object el)
        {
            ColorDialog dlg = new ColorDialog();
            dlg.FullOpen = false;
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                gdiDisplay1.RecolorWithOwner(el, dlg.Color.R, dlg.Color.G, dlg.Color.B, m_Recolor == RecolorMode.Hue);
                gdiDisplay1.Render();
            }
        }

        private void LoadButton_Click(object sender, System.EventArgs e)
		{
			System.Windows.Forms.OpenFileDialog od = new System.Windows.Forms.OpenFileDialog();
			od.Title = "Select X3L file";
			od.Filter = "X3L XML Scene file (*.x3l)|*.x3l";
			if (od.ShowDialog() == DialogResult.OK)
				gdiDisplay1.LoadScene(od.FileName);
		}

		private void SaveButton_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sd = new SaveFileDialog();
			sd.Title = "Select file and format";
			sd.Filter = "Windows Bitmap (*.bmp)|*.bmp|Joint Photographic Experts Group (*.jpg)|*.jpg|Graphics Interexchange Format (*.gif)|*.gif|Portable Network Graphics (*.png)|*.png|Enhanced Windows Metafile (*.emf)|*.emf|3D XML Scene file (*.x3l)|*.x3l";
			if (sd.ShowDialog() == DialogResult.OK)
			{
				gdiDisplay1.Save(sd.FileName);
			}		
		}

		private void ZoomInButton_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Zoom *= 1.25;
		}

		private void ZoomOutButton_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.Zoom *= 0.8;
		}

		private void OnDragDrop(object sender, System.Windows.Forms.DragEventArgs e)
		{
            MessageBox.Show(e.ToString(), "DragDrop Debug");
		}

		private void OnAlphaChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.Alpha = AlphaTrack.Value / 8.0;
		}

		private void OnLinesChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.LineWidth = LinesTrack.Value;
		}

		private void OnPointsChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.PointSize = PointsTrack.Value;
		}

		private void MergeButton_Click(object sender, System.EventArgs e)
		{
			System.Windows.Forms.OpenFileDialog od = new System.Windows.Forms.OpenFileDialog();
			od.Title = "Select X3L file to merge";
			od.Filter = "X3L XML Scene file (*.x3l)|*.x3l";
			od.Multiselect = true;
			if (od.ShowDialog() == DialogResult.OK)
				foreach (string s in od.FileNames)
					try
					{
						gdiDisplay1.LoadMergeScene(s);
					}
					catch (Exception x)
					{
						MessageBox.Show(x.Message, "Can't load & merge file", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
		}

		private void IsometricButton_CheckedChanged(object sender, System.EventArgs e)
		{
			gdiDisplay1.Infinity = IsometricButton.Checked;			
		}

		private void SetFocusButton_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.NextClickSetsCenter = true;
		}

		private void PanButton_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Pan;
		}

		private void RotButton_Click(object sender, System.EventArgs e)
		{
			gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
		}

		private void BkgndButton_Click(object sender, System.EventArgs e)
		{
			if (colorDialog1.ShowDialog() == DialogResult.OK)
			{
				gdiDisplay1.BackColor = colorDialog1.Color;
				gdiDisplay1.Render();
			}
		}

        private void XYbutton_Click(object sender, EventArgs e)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        private void XZbutton_Click(object sender, EventArgs e)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        private void YZbutton_Click(object sender, EventArgs e)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        private void RXYbutton_Click(object sender, EventArgs e)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        private void RXZbutton_Click(object sender, EventArgs e)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        private void RYZbutton_Click(object sender, EventArgs e)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        bool m_EnableKill = false;
        enum RecolorMode { None, Hue, Color };
        RecolorMode m_Recolor = RecolorMode.None;

        private void UpdateButtonText()
        {
            KillByOwnerButton.Text = m_EnableKill ? "Stop killing" : "Kill by owner";
            RecolorButton.Text = (m_Recolor == RecolorMode.None) ? "Change hue" : ((m_Recolor == RecolorMode.Hue) ? "Change color" : "No recolor");
            gdiDisplay1.DoubleClickSelect = (m_Recolor != RecolorMode.None) ? new GDI3D.Control.SelectObject(Recolor) : (m_EnableKill ? new GDI3D.Control.SelectObject(KillByOwner) : new GDI3D.Control.SelectObject(OnDoubleClickSelect));
        }

        private void KillByOwnerButton_Click(object sender, EventArgs e)
        {
            m_EnableKill = !m_EnableKill;
            m_Recolor = RecolorMode.None;
            UpdateButtonText();
        }


        private void RecolorButton_Click(object sender, EventArgs e)
        {
            switch (m_Recolor)
            {
                case RecolorMode.None: m_Recolor = RecolorMode.Hue; break;
                case RecolorMode.Hue: m_Recolor = RecolorMode.Color; break;
                case RecolorMode.Color: m_Recolor = RecolorMode.None; break;
            }
            m_EnableKill = false;
            UpdateButtonText();
        }

        private void QuickDisplayButton_Click(object sender, EventArgs e)
        {
            QuickDisplayChoose qdc = new QuickDisplayChoose();
            if (qdc.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(GDI3D.Scene));
                    GDI3D.Scene sc = (GDI3D.Scene)xmls.Deserialize(new System.IO.StringReader(qdc.m_Scene));                    
                    SetScene(sc);
                    gdiDisplay1.Render();
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "XML Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void OnLoad(object sender, EventArgs e)
        {
            UpdateButtonText();
        }
	}
}
