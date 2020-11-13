using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using SySal.Management;

namespace SySal.Executables.NExTScanner
{
    public partial class CameraDisplay : Form, IMachineSettingsEditor
    {
        SySal.Imaging.IImageProcessor m_ImageProcessor;

        public SySal.Imaging.IImageProcessor ImageProcessor
        {
            get
            {
                return m_ImageProcessor;
            }
            set
            {
                m_ImageProcessor = value;
                if (m_ImageProcessor != null)
                {
                    HostedFormat = m_ImageProcessor.ImageFormat;
                    m_ImageProcessor.OutputFeatures = m_ImageProcessor.OutputFeatures | SySal.Imaging.ImageProcessingFeatures.BinarizedImage | SySal.Imaging.ImageProcessingFeatures.Clusters;
                }
            }
        }

        protected SySal.Imaging.ImageInfo m_HostedFormat;

        public SySal.Imaging.ImageInfo HostedFormat
        {
            get
            {
                return m_HostedFormat;
            }
            set
            {
                SuspendLayout();
                try
                {
                    m_HostedFormat = value;
                    double f = Math.Min(1.0, Math.Min((double)(Width - pnRight.MinimumSize.Width) / (double)m_HostedFormat.Width, (double)Height / (double)m_HostedFormat.Height));
                    pnBottom.Height = Height - (int)(f * m_HostedFormat.Height);
                    pnRight.Width = Width - (int)(f * m_HostedFormat.Width);
                }
                catch (Exception) { }
                ResumeLayout();
            }
        }

        SySal.Executables.NExTScanner.ImagingConfiguration m_ImagingConfiguration = SySal.Executables.NExTScanner.ImagingConfiguration.Default;

        public SySal.Executables.NExTScanner.ImagingConfiguration ImagingConfiguration
        {
            set
            {
                m_ImagingConfiguration = value;
            }
        }

        public IMapProvider iMap;

        public CameraDisplay()
        {
            InitializeComponent();
            CameraDisplaySettings c = (CameraDisplaySettings)SySal.Management.MachineSettings.GetSettings(typeof(CameraDisplaySettings));
            if (c == null)
            {
                c = new CameraDisplaySettings();
                c.PanelWidth = 640;
                c.PanelHeight = 480;
                c.PanelTop = 0;
                c.PanelLeft = 0;
            }
            StartPosition = FormStartPosition.Manual;
            Left = c.PanelLeft;
            Top = c.PanelTop;
            Width = c.PanelWidth;
            Height = c.PanelHeight;
            SySal.Imaging.ImageInfo info = new SySal.Imaging.ImageInfo();
            info.BitsPerPixel = 8;
            info.PixelFormat = SySal.Imaging.PixelFormatType.GrayScale8;
            info.Width = (ushort)(Width - pnRight.MinimumSize.Width);
            info.Height = (ushort)(Height - pnBottom.MinimumSize.Height);
            pnRight.Width = pnRight.MinimumSize.Width;
            pnBottom.Height = pnBottom.MinimumSize.Height;
            HostedFormat = info;
            System.Drawing.Bitmap z = new Bitmap(8, 8, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            GrayScalePalette = z.Palette;
            int i;
            for (i = 0; i < GrayScalePalette.Entries.Length; i++)
                GrayScalePalette.Entries[i] = Color.FromArgb(i, i, i);
        }

        System.Drawing.Imaging.ColorPalette GrayScalePalette;

        private void DisplayClusters(System.Drawing.Graphics g, SySal.Imaging.Cluster[] clusters, double xedge, double yedge, double fzoom)
        {
            //float f = (float)pbScreen.Width / (float)m_HostedFormat.Width;            
            System.Drawing.Pen cpen = new Pen(Color.Coral, 1);
            foreach (SySal.Imaging.Cluster c in clusters)
            {
                int side = (int)Math.Sqrt(c.Area) / 2 + 1;
                g.DrawEllipse(cpen, (float)(fzoom * (c.X - xedge - side)), (float)(fzoom * (c.Y - yedge - side)), (float)(2 * side * fzoom), (float)(2 * side * fzoom));
            }
        }

        bool m_EnableCross = false;

        public bool EnableCross
        {
            get { return m_EnableCross; }
            set 
            { 
                m_EnableCross = value;
                m_OverlayBmp = null;
            }
        }

        private void DisplayOverlay(double fzoom)
        {
            if (m_OverlayScene != null || m_EnableCross)
            {
                if (m_OverlayBmp == null) m_OverlayBmp = new Bitmap(pbScreen.Width, pbScreen.Height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(m_OverlayBmp);
                g.Clear(Color.Black);
                Pen axispen = new Pen(Color.LightCoral);
                g.DrawLine(axispen, 0, pbScreen.Height / 2, pbScreen.Width, pbScreen.Height / 2);
                g.DrawLine(axispen, pbScreen.Width / 2, 0, pbScreen.Width / 2, pbScreen.Height);
                if (m_OverlayScene != null)
                {
                    BasicTypes.Vector2 S = new BasicTypes.Vector2();
                    BasicTypes.Vector2 F = new BasicTypes.Vector2();
                    foreach (GDI3D.Line ln in m_OverlayScene.Lines)
                    {
                        S.X = ln.XS;
                        S.Y = ln.YS;
                        F.X = ln.XF;
                        F.Y = ln.YF;
                        var vS = iMap.PlateMap.Transform(S);
                        var vF = iMap.PlateMap.Transform(F);
                        float x1 = (float)((vS.X - m_StagePos.X) / m_ImagingConfiguration.Pixel2Micron.X * fzoom) + pbScreen.Width / 2;
                        float y1 = (float)((vS.Y - m_StagePos.Y) / m_ImagingConfiguration.Pixel2Micron.Y * fzoom) + pbScreen.Height / 2;
                        float x2 = (float)((vF.X - m_StagePos.X) / m_ImagingConfiguration.Pixel2Micron.X * fzoom) + pbScreen.Width / 2;
                        float y2 = (float)((vF.Y - m_StagePos.Y) / m_ImagingConfiguration.Pixel2Micron.Y * fzoom) + pbScreen.Height / 2;
                        g.DrawLine(new Pen(Color.FromArgb(255, ln.R, ln.G, ln.B)), x1, y1, x2, y2);
                    }
                    foreach (GDI3D.Point pn in m_OverlayScene.Points)
                    {
                        S.X = pn.X;
                        S.Y = pn.Y;
                        var vS = iMap.PlateMap.Transform(S);
                        float x = (float)((vS.X - m_StagePos.X) / m_ImagingConfiguration.Pixel2Micron.X * fzoom) + pbScreen.Width / 2;
                        float y = (float)((vS.Y - m_StagePos.Y) / m_ImagingConfiguration.Pixel2Micron.Y * fzoom) + pbScreen.Height / 2;
                        g.DrawEllipse(new Pen(Color.FromArgb(255, pn.R, pn.G, pn.B)), x - 2, y - 2, 5, 5);
                    }
                }
                g.Dispose();                
            }
        }

        System.Drawing.Bitmap m_OriginalBmp = null;

        System.Drawing.Bitmap m_Bmp = null;

        System.Drawing.Bitmap m_OverlayBmp = null;

        System.Drawing.Graphics m_G = null;

        private delegate void dVoid();

        private delegate void dSetText(TextBox tb, string tx);

        private delegate void dSetTextBackColor(TextBox tb, string tx, Color bk);

        void SetText(TextBox tb, string tx)
        {
            if (InvokeRequired) Invoke(new dSetText(SetText), new object[] { tb, tx });
            tb.Text = tx;
        }

        void SetTextBackColor(TextBox tb, string tx, Color bk)
        {
            if (InvokeRequired) Invoke(new dSetTextBackColor(SetTextBackColor), new object[] { tb, tx, bk });
            tb.Text = tx;
            tb.BackColor = bk;
        }

        public void SetNonreusableImage(SySal.Imaging.LinearMemoryImage lmi)
        {
            lock (this)
            {
                m_OriginalBmp = new Bitmap(lmi.Info.Width, lmi.Info.Height, lmi.Info.Width, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, ImageAccessor.Scan(lmi));
                m_OriginalBmp.Palette = GrayScalePalette;
                m_Bmp = new Bitmap(pbScreen.Width, pbScreen.Height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            }
            this.Invoke(new dVoid(delegate()
                {
                    lock (this)
                    {
                        pbScreen.Image = m_Bmp;
                        m_G = System.Drawing.Graphics.FromImage(m_Bmp);
                        m_G.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                        m_G.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighSpeed;
                        m_G.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                        double fzoom = m_Bmp.Width * m_Zoom / m_OriginalBmp.Width;
                        m_G.DrawImage(m_OriginalBmp, new Rectangle(0, 0, m_Bmp.Width, m_Bmp.Height), new Rectangle((int)((m_OriginalBmp.Width - m_Bmp.Width / fzoom) * 0.5), (int)((m_OriginalBmp.Height - m_Bmp.Height / fzoom) * 0.5), (int)(m_Bmp.Width / fzoom), (int)(m_Bmp.Height / fzoom)), GraphicsUnit.Pixel);
                    }
                }
            ));
        }

        GDI3D.Scene m_OverlayScene = null;

        SySal.StageControl.IStageWithoutTimesource m_Stage = null;

        public SySal.StageControl.IStageWithoutTimesource Stage
        {
            set
            {
                m_Stage = value;
                try
                {
                    m_StagePos.X = m_Stage.GetPos(SySal.StageControl.Axis.X);
                    m_StagePos.Y = m_Stage.GetPos(SySal.StageControl.Axis.Y);
                    m_StagePos.Z = m_Stage.GetPos(SySal.StageControl.Axis.Z);
                }
                catch (Exception) { }
            }
        }

        SySal.BasicTypes.Vector m_StagePos;

        public SySal.Imaging.LinearMemoryImage ImageShown
        {
            set
            {                
                System.DateTime t1, t2, t3, t4, t5, t6, t7, t8;
                try
                {
                    string errorstring = "";
                    SySal.Imaging.LinearMemoryImage lmi = value;

                    bool processed = false;
#if DEBUG_DISPLAY_TIME
                    t1 = System.DateTime.Now;
                    t2 = t3 = t4 = t5 = t6 = t7 = t8 = t1;
#endif
                    try
                    {                            
                        if (chkProcess.Checked && m_ImageProcessor != null && m_ImageProcessor.IsReady)
                        {                            
                            m_ImageProcessor.Input = value;
#if DEBUG_DISPLAY_TIME
                            t2 = System.DateTime.Now;
#endif
                            if (m_ShowBinary) lmi = m_ImageProcessor.BinarizedImages;
                            SetText(txtGreyLevelMedian, m_ImageProcessor.GreyLevelMedian.ToString());
                            SetTextBackColor(txtClusters, m_ImageProcessor.Clusters[0].Length.ToString(), (m_ImageProcessor.Warnings.Length > 0) ? Color.Coral : SystemColors.Control);
                            processed = true;
#if DEBUG_DISPLAY_TIME
                            t3 = System.DateTime.Now;
#endif
                        }
                        else
                        {
                            SetText(txtGreyLevelMedian, "");
                            SetTextBackColor(txtClusters, "", SystemColors.Control);
#if DEBUG_DISPLAY_TIME
                            t2 = t3 = System.DateTime.Now;
#endif
                        }
                    }
                    catch (Exception exc) { errorstring = exc.ToString(); }                    
                    
                    m_OriginalBmp = new Bitmap(lmi.Info.Width, lmi.Info.Height, lmi.Info.Width, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, ImageAccessor.Scan(lmi));                    
                    m_OriginalBmp.Palette = GrayScalePalette;
#if DEBUG_DISPLAY_TIME
                    t4 = System.DateTime.Now;
#endif
                    if (m_Bmp == null || m_Bmp.Width != pbScreen.Width || m_Bmp.Height != pbScreen.Height) 
                    {
                        pbScreen.Image = m_Bmp = new Bitmap(pbScreen.Width, pbScreen.Height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                        m_G = System.Drawing.Graphics.FromImage(m_Bmp);
                        m_G.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                        m_G.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighSpeed;
                        m_G.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceOver;
                    }
#if DEBUG_DISPLAY_TIME
                    t5 = System.DateTime.Now;
#endif
                    double xedge, yedge;
                    double fzoom = m_Bmp.Width * m_Zoom / m_OriginalBmp.Width;                    
                    m_G.DrawImage(m_OriginalBmp, new Rectangle(0, 0, m_Bmp.Width, m_Bmp.Height), new Rectangle((int)(xedge = (m_OriginalBmp.Width - m_Bmp.Width / fzoom) * 0.5), (int)(yedge = (m_OriginalBmp.Height - m_Bmp.Height / fzoom) * 0.5), (int)(m_Bmp.Width / fzoom), (int)(m_Bmp.Height / fzoom)), GraphicsUnit.Pixel);
#if DEBUG_DISPLAY_TIME
                    t6 = System.DateTime.Now;
#endif
                    if (errorstring.Length > 0) m_G.DrawString(errorstring, new Font("Segoe UI", 12), new SolidBrush(Color.Red), 0.0f, 0.0f);
                    if (processed && chkShowClusters.Checked) DisplayClusters(m_G, m_ImageProcessor.Clusters[0], xedge, yedge, fzoom);
                    if (m_OverlayScene != null || m_EnableCross)
                    {
                        if (m_OverlayBmp == null) DisplayOverlay(fzoom);
                        System.Drawing.Imaging.ImageAttributes imgattr = new System.Drawing.Imaging.ImageAttributes();
                        imgattr.SetColorKey(Color.FromArgb(0, 0, 0), Color.FromArgb(0, 0, 0));
                        m_G.DrawImage(m_OverlayBmp, new Rectangle(0, 0, m_Bmp.Width, m_Bmp.Height), 0, 0, m_Bmp.Width, m_Bmp.Height, GraphicsUnit.Pixel, imgattr);                        
                    }
#if DEBUG_DISPLAY_TIME
                    t7 = System.DateTime.Now;
#endif
                    //pbScreen.Image = m_Bmp;
                    Invoke(new dVoid(pbScreen.Refresh));
                    //pbScreen.Refresh();
#if DEBUG_DISPLAY_TIME
                    t8 = System.DateTime.Now;                    
                    System.Drawing.Graphics debug_G = pbScreen.CreateGraphics();
                    debug_G.DrawString(
                        "Delta12 " + (t2 - t1).TotalMilliseconds.ToString() +
                        "\r\nDelta23 " + (t3 - t2).TotalMilliseconds.ToString() +
                        "\r\nDelta34 " + (t4 - t3).TotalMilliseconds.ToString() +
                        "\r\nDelta45 " + (t5 - t4).TotalMilliseconds.ToString() +
                        "\r\nDelta56 " + (t6 - t5).TotalMilliseconds.ToString() +
                        "\r\nDelta67 " + (t7 - t6).TotalMilliseconds.ToString() +
                        "\r\nDelta78 " + (t8 - t7).TotalMilliseconds.ToString(),
                        new Font("Arial", 12), new SolidBrush(Color.Blue), 32.0f, 200.0f);
#endif                     
                }
                catch (Exception xc)
                {                    
//                    MessageBox.Show(xc.ToString(), "Display error");
                }
            }
        }

        #region IMachineSettingsEditor Members

        public bool EditMachineSettings(Type t)
        {
            CameraDisplaySettings C = (CameraDisplaySettings)SySal.Management.MachineSettings.GetSettings(t);
            if (C == null)
            {
                MessageBox.Show("No valid configuration found, switching to default", "Configuration warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                C = new CameraDisplaySettings();
                C.Name = "Default CameraDisplaySettings configuration";
                C.PanelLeft = 0;
                C.PanelTop = 0;
                C.PanelWidth = 640;
                C.PanelHeight = 480;
            }
            EditCameraDisplaySettingsForm ef = new EditCameraDisplaySettingsForm(C);
            if (ef.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    SySal.Management.MachineSettings.SetSettings(t, ef.C);
                    MessageBox.Show("Configuration saved", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    HostedFormat = m_HostedFormat;
                    return true;
                }
                catch (Exception x)
                {
                    MessageBox.Show("Error saving configuration\r\n\r\n" + x.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
            }
            return false;
        }

        #endregion

        private void btnConfigure_Click(object sender, EventArgs e)
        {
            if (EditMachineSettings(typeof(CameraDisplaySettings)))
            {
                CameraDisplaySettings c = (CameraDisplaySettings)SySal.Management.MachineSettings.GetSettings(typeof(CameraDisplaySettings));
                Left = c.PanelLeft;
                Top = c.PanelTop;
                Width = Math.Max(100, c.PanelWidth);
                Height = Math.Max(100, c.PanelHeight);
                HostedFormat = m_HostedFormat;
            }
        }

        bool m_ShowBinary = false;

        private void OnSourceCheckedChanged(object sender, EventArgs e)
        {
            m_ShowBinary = rdBinary.Checked;
        }

        private void OnBinaryCheckedChanged(object sender, EventArgs e)
        {
            if (rdBinary.Checked)
            {
                if (m_ImageProcessor != null)
                {
                    m_ShowBinary = true;
                }
                else
                {
                    m_ShowBinary = false;
                    rdSource.Checked = true;
                }
            }
            else m_ShowBinary = false;
        }

        class ImageAccessor : SySal.Imaging.LinearMemoryImage
        {
            static public IntPtr Scan(SySal.Imaging.LinearMemoryImage lm)
            {
                return SySal.Imaging.LinearMemoryImage.AccessMemoryAddress(lm);
            }

            public ImageAccessor() : base(new SySal.Imaging.ImageInfo(), new IntPtr(), 0, null) { }

            public override SySal.Imaging.Image SubImage(uint i)
            {
                return null;
            }

            public override void Dispose()
            {
            }

        }

        double m_Zoom = 1.0;

        private void btnZoomIn_Click(object sender, EventArgs e)
        {
            m_Zoom *= 1.25;
            if (m_Zoom >= 8.0) m_Zoom = 8.0;
            txtZoom.Text = m_Zoom.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            m_OverlayBmp = null;
        }

        private void btnZoomOut_Click(object sender, EventArgs e)
        {
            m_Zoom *= 0.8;
            if (m_Zoom <= 1.0) m_Zoom = 1.0;
            txtZoom.Text = m_Zoom.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            m_OverlayBmp = null;
        }

        private void OnImagePaint(object sender, PaintEventArgs e)
        {
            if (m_Bmp != null) e.Graphics.DrawImageUnscaled(m_Bmp, 0, 0);
            else e.Graphics.FillRectangle(new SolidBrush(Color.Black), pbScreen.ClientRectangle);
            this.Validate();
        }

        private void btnSaveImage_Click(object sender, EventArgs e)
        {            
            if (m_Bmp != null)
            {
                Bitmap savebmp = (Bitmap)m_OriginalBmp.Clone();
                if (dlgSaveImage.ShowDialog() == DialogResult.OK)
                {
                    string fname = dlgSaveImage.FileName;
                    try
                    {
                        if (fname.ToLower().EndsWith(".b64"))
                        {
                            System.IO.File.WriteAllText(fname, SySal.Imaging.Base64ImageEncoding.ImageToBase64(new SySalImageFromImage(savebmp)));
                        }
                        else if (fname.ToLower().EndsWith(".jpg") || fname.ToLower().EndsWith(".jpeg"))
                        {
                            savebmp.Save(fname, System.Drawing.Imaging.ImageFormat.Jpeg);
                        }
                        else 
                        {
                            savebmp.Save(fname, System.Drawing.Imaging.ImageFormat.Bmp);
                        }
                        MessageBox.Show("File saved to \"" + fname + "\"", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    catch (Exception xc)
                    {
                        MessageBox.Show("Error saving file:\r\n" + xc.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private void btnSendToBack_Click(object sender, EventArgs e)
        {
            this.SendToBack();
        }

        static System.Xml.Serialization.XmlSerializer xmlscene = new System.Xml.Serialization.XmlSerializer(typeof(GDI3D.Scene));

        const double EmulThickness = 45.0;

        const double BaseThickness = 205.0;

        public string m_DefaultDirectory = "";

        public static System.Text.RegularExpressions.Regex rx_RWD = new System.Text.RegularExpressions.Regex(@"(.+)\.rwd\.([0-9a-f]+)");

        private void btnLoadOverlay_Click(object sender, EventArgs e)
        {            
            dlgLoadOverlay.InitialDirectory = m_DefaultDirectory;
            if (dlgLoadOverlay.ShowDialog() == DialogResult.OK)
            {
                System.IO.BinaryReader r = null;
                System.IO.FileStream fr = null;
                try
                {
                    if (dlgLoadOverlay.FileName.ToLower().EndsWith(".x3l"))
                    {
                        m_OverlayScene = (GDI3D.Scene)xmlscene.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(dlgLoadOverlay.FileName)));
                    }
                    else if (dlgLoadOverlay.FileName.ToLower().EndsWith(".rwd") || rx_RWD.Match(dlgLoadOverlay.FileName.ToLower()).Success)
                    {
                        fr = new System.IO.FileStream(dlgLoadOverlay.FileName, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read);
                        SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag = new Scanning.Plate.IO.OPERA.RawData.Fragment(fr);
                        fr.Close();
                        fr = null;
                        GDI3D.Scene scene = new GDI3D.Scene();
                        int total = 0;
                        int v;
                        for (v = 0; v < frag.Length; v++)
                            total += frag[v].Top.Length + frag[v].Bottom.Length;
                        scene.Lines = new GDI3D.Line[2 * total];
                        scene.Points = new GDI3D.Point[0];
                        scene.OwnerSignatures = new string[total];
                        total = 0;
                        for (v = 0; v < frag.Length; v++)
                        {
                            int i;
                            for (i = 0; i < frag[v].Top.Length; i++)
                            {
                                SySal.Tracking.MIPEmulsionTrackInfo info = frag[v].Top[i].Info;
                                info.Intercept = frag[v].Top.MapPoint(info.Intercept);
                                info.Slope = frag[v].Top.MapVector(info.Slope);
                                scene.OwnerSignatures[total / 2] = v + "T" + i + ": " + info.Count + " " + info.AreaSum + " " + info.Intercept.X.ToString("F1") + " " + info.Intercept.Y.ToString("F1") + " " + info.Slope.X.ToString("F4") + " " + info.Slope.Y.ToString("F4") + " " + info.Sigma.ToString("F3");
                                scene.Lines[total++] = new GDI3D.Line(info.Intercept.X, info.Intercept.Y, 0, info.Intercept.X + EmulThickness * info.Slope.X, info.Intercept.Y + EmulThickness * info.Slope.Y, EmulThickness, i, 224, 0, 0);
                                scene.Lines[total++] = new GDI3D.Line(info.Intercept.X, info.Intercept.Y, 0, info.Intercept.X - BaseThickness * info.Slope.X, info.Intercept.Y - BaseThickness * info.Slope.Y, -BaseThickness, i, 0, 224, 0);                                
                            }
                            for (i = 0; i < frag[v].Bottom.Length; i++)
                            {
                                SySal.Tracking.MIPEmulsionTrackInfo info = frag[v].Bottom[i].Info;
                                info.Intercept = frag[v].Bottom.MapPoint(info.Intercept);
                                info.Slope = frag[v].Bottom.MapVector(info.Slope);
                                scene.OwnerSignatures[total / 2] = v + "B" + i + ": " + info.Count + " " + info.AreaSum + " " + info.Intercept.X.ToString("F1") + " " + info.Intercept.Y.ToString("F1") + " " + info.Slope.X.ToString("F4") + " " + info.Slope.Y.ToString("F4") + " " + info.Sigma.ToString("F3");
                                scene.Lines[total++] = new GDI3D.Line(info.Intercept.X, info.Intercept.Y, -BaseThickness, info.Intercept.X - EmulThickness * info.Slope.X, info.Intercept.Y - EmulThickness * info.Slope.Y, -BaseThickness - EmulThickness, i, 0, 0, 224);
                                scene.Lines[total++] = new GDI3D.Line(info.Intercept.X, info.Intercept.Y, -BaseThickness, info.Intercept.X + BaseThickness * info.Slope.X, info.Intercept.Y + BaseThickness * info.Slope.Y, 0, i, 0, 224, 0);
                            }                            
                        }                        
                        m_OverlayScene = scene;
                    }
                    else if (dlgLoadOverlay.FileName.ToLower().EndsWith(".reader"))
                    {
                        r = new System.IO.BinaryReader(new System.IO.FileStream(dlgLoadOverlay.FileName, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read));
                        int total = r.ReadInt32();
                        GDI3D.Scene scene = new GDI3D.Scene();
                        scene.Lines = new GDI3D.Line[2 * total];
                        scene.Points = new GDI3D.Point[0];
                        scene.OwnerSignatures = new string[total];
                        r.ReadDouble(); r.ReadDouble(); r.ReadDouble(); r.ReadDouble();
                        int i;
                        for (i = 0; i < total; i++)
                        {
                            Int64 zone = r.ReadInt64();
                            short side = r.ReadInt16();
                            int id = r.ReadInt32();
                            short count = r.ReadInt16();
                            int areasum = r.ReadInt32();
                            double x = r.ReadDouble();
                            double y = r.ReadDouble();
                            double z = (side == 1) ? 0 : -BaseThickness;
                            double sx = r.ReadDouble();
                            double sy = r.ReadDouble();
                            double sigma = r.ReadDouble();
                            int view = r.ReadInt32();
                            double vx = r.ReadDouble();
                            double vy = r.ReadDouble();
                            scene.OwnerSignatures[i] = zone + " " + side + " " + id + ": " + count + " " + areasum + " " + x.ToString("F1") + " " + y.ToString("F1") + " " + sx.ToString("F4") + " " + sy.ToString("F4") + " " + sigma.ToString("F3") + " in " + view + ": " + vx.ToString("F1") + " " + vy.ToString("F1");
                            double dz = ((side == 1) ? EmulThickness : -EmulThickness);
                            double dzbase = ((side == 1) ? -BaseThickness : BaseThickness);
                            scene.Lines[2 * i] = new GDI3D.Line(x, y, z, x + dz * sx, y + dz * sy, z + dz, i, (side == 1) ? 255 : 0, 0, (side == 2) ? 255 : 0);
                            scene.Lines[2 * i + 1] = new GDI3D.Line(x, y, z, x + dzbase * sx, y + dzbase * sy, z + dzbase, i, (side == 1) ? 255 : 0, 192, (side == 2) ? 255 : 0);
                        }
                        r.Close();
                        r = null;
                        m_OverlayScene = scene;
                    }
                    else if (dlgLoadOverlay.FileName.ToLower().EndsWith(".tlg"))
                    {
                        SySal.DataStreams.OPERALinkedZone lzd = new DataStreams.OPERALinkedZone(dlgLoadOverlay.FileName);
                        int total = lzd.Length;
                        GDI3D.Scene scene = new GDI3D.Scene();
                        scene.Lines = new GDI3D.Line[3 * total];
                        scene.Points = new GDI3D.Point[0];
                        scene.OwnerSignatures = new string[total];
                        int i;
                        for (i = 0; i < total; i++)
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo info = lzd[i].Info;
                            scene.OwnerSignatures[i] = i + ": " + info.Count + " " + info.AreaSum + " " + info.Intercept.X.ToString("F1") + " " + info.Intercept.Y.ToString("F1") + " " + info.Slope.X.ToString("F4") + " " + info.Slope.Y.ToString("F4") + " " + info.Sigma.ToString("F3");
                            scene.Lines[3 * i] = new GDI3D.Line(info.Intercept.X, info.Intercept.Y, 0, info.Intercept.X + EmulThickness * info.Slope.X, info.Intercept.Y + EmulThickness * info.Slope.Y, EmulThickness, i, 224, 0, 0);
                            scene.Lines[3 * i + 1] = new GDI3D.Line(info.Intercept.X, info.Intercept.Y, 0, info.Intercept.X - BaseThickness * info.Slope.X, info.Intercept.Y - BaseThickness * info.Slope.Y, -BaseThickness, i, 0, 224, 0);
                            scene.Lines[3 * i + 2] = new GDI3D.Line(info.Intercept.X - BaseThickness * info.Slope.X, info.Intercept.Y - BaseThickness * info.Slope.Y, -BaseThickness, info.Intercept.X - (BaseThickness + EmulThickness) * info.Slope.X, info.Intercept.Y - (BaseThickness + EmulThickness) * info.Slope.Y, -BaseThickness - EmulThickness, i, 0, 0, 224);
                        }
                        lzd.Dispose();
                        m_OverlayScene = scene;
                    }
                    else MessageBox.Show("Unsupported format", "File error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (r != null)
                    {
                        r.Close();
                        r = null;
                    }
                    if (fr != null)
                    {
                        fr.Close();
                        fr = null;
                    }
                    m_OverlayBmp = null;
                }
            }
        }

        private void btnSyncStage_Click(object sender, EventArgs e)
        {
            try
            {
                m_StagePos.X = m_Stage.GetPos(SySal.StageControl.Axis.X);
                m_StagePos.Y = m_Stage.GetPos(SySal.StageControl.Axis.Y);
                m_StagePos.Z = m_Stage.GetPos(SySal.StageControl.Axis.Z);
                m_OverlayBmp = null;
            }
            catch (Exception) { }            
        }

        private void btnClearOverlay_Click(object sender, EventArgs e)
        {
            m_OverlayScene = null;            
        }

        private void OnScreenMouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == System.Windows.Forms.MouseButtons.Right)
            {
                if (m_Bmp != null && m_OriginalBmp != null)
                {
                    double fzoom = m_Bmp.Width * m_Zoom / m_OriginalBmp.Width;
                    txtMouseX.Text = ((e.X - pbScreen.Width / 2) / fzoom * m_ImagingConfiguration.Pixel2Micron.X + m_StagePos.X).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                    txtMouseY.Text = ((e.Y - pbScreen.Height / 2) / fzoom * m_ImagingConfiguration.Pixel2Micron.Y + m_StagePos.Y).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                }
            }
        }
    }
}
