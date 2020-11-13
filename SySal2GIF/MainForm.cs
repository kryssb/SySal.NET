using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.SySal2GIF
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        System.Collections.ArrayList m_FileSequence = new System.Collections.ArrayList();

        System.IO.MemoryStream m_FogFile = null;

        Size m_FogImageSize = new Size(0, 0);

        long[] m_FogFileImagePointers = null;

        Image AniGIF = null;

        SySal.Imaging.ImageSequenceInfo ImSeqInfo;

        void DefaultImSeqInfo()
        {
            ImSeqInfo.Comment = "";
            ImSeqInfo.Info2D.Center.X = ImSeqInfo.Info2D.Center.Y = 0.0;
            ImSeqInfo.Info2D.PixelToMicronXX = ImSeqInfo.Info2D.PixelToMicronYY = 1.0;
            ImSeqInfo.Info2D.PixelToMicronXY = ImSeqInfo.Info2D.PixelToMicronYX = 0.0;
            ImSeqInfo.EmulsionLayers = new SySal.Imaging.EmulsionLayerImageDepthInfo[0];
        }

        static string AniGIFComment(System.IO.Stream gifs)
        {
            string s = "";
            long pos = gifs.Position;
            try
            {
                int b, b1;
                gifs.Position += 781;
                while ((b = gifs.ReadByte()) >= 0)
                {
                    switch (b)
                    {
                        case 33:
                            b = gifs.ReadByte();
                            if (b < 0) throw new Exception("Malformed file.");
                            switch (b)
                            {
                                case 254:
                                    do
                                    {
                                        b = gifs.ReadByte();
                                        if (b < 0) throw new Exception("Malformed file.");
                                        b1 = b;
                                        while (b1-- > 0)
                                            s += (char)gifs.ReadByte();
                                    }
                                    while (b != 0);
                                    return s;
                                    break;
                                
                                default:
                                    do
                                    {
                                        b = gifs.ReadByte();
                                        if (b < 0) throw new Exception("Malformed file.");
                                        gifs.Position += b;
                                    }
                                    while (b != 0);
                                    continue;
                            }
                            break;

                        case 44:
                            gifs.Position += 10;
                            do
                            {
                                b = gifs.ReadByte();
                                if (b < 0) throw new Exception("Malformed file.");
                                gifs.Position += b;
                            }
                            while (b != 0);
                            continue;
                            break;

                        case 0: continue;

                        case ';': return null;

                        default: throw new Exception("Unexpected bytecode " + b.ToString("X2") + " in GIF file at position " + gifs.Position + ".");
                    }
                }
                return null;
            }
            finally
            {
                gifs.Position = pos;
            }
        }

        int Frames
        {
            get
            {
                if (m_FileSequence != null && m_FileSequence.Count > 0) return m_FileSequence.Count;
                if (AniGIF != null) return AniGIF.GetFrameCount(new System.Drawing.Imaging.FrameDimension(AniGIF.FrameDimensionsList[0]));
                if (m_FogFileImagePointers != null) return m_FogFileImagePointers.Length;
                return 0;
            }
        }

        Rectangle m_SelRect;

        int m_CurrentImage;

        int m_Zoom = 0;

        int Zoom
        {
            get { return m_Zoom; }
            set
            {
                m_Zoom = value;
                if (m_Image == null) return;                
                pictureBox1.Size = new Size((int)(m_Image.Width / ZoomFactor), (int)(m_Image.Height / ZoomFactor));
                pictureBox1.Refresh();
            }
        }

        double ZoomFactor
        {
            get { return Math.Pow(2.0, m_Zoom); }
        }

        static byte[,] SySalColorTable = new byte[16, 3]
            {
                {0, 0, 0},
				{	0,	0,	192},
				{	0,	192,	0},
				{	0,	192,	192},
				{	192,	0,	0},
				{	192,	0,	192},
				{	192,	192,	0},
				{	192,	192,	192},
				{	224,	224,	224},
				{	0,	0,	255},
				{	0,	255,	0},
				{	0,	255,	255},
				{	255,	0,	0},
				{	255,	0,	255},
				{	255,	255,	0},
				{	255,	255,	255}
            };

        Image m_Image;

        int CurrentImage
        {
            get { return m_CurrentImage; }
            set
            {
                m_CurrentImage = value;
                try
                {
                    if (m_FogFileImagePointers != null)
                    {
                        m_FogFile.Position = m_FogFileImagePointers[m_CurrentImage];                        
                        int x, y;
                        System.IO.MemoryStream ms = BMP8Stream(m_FogImageSize.Width, m_FogImageSize.Height);
                        byte[] bys = new byte[m_FogImageSize.Width * m_FogImageSize.Height];
                        m_FogFile.Read(bys, 0, m_FogImageSize.Width * m_FogImageSize.Height);
                        ms.Write(bys, 0, m_FogImageSize.Width * m_FogImageSize.Height);
                        ms.Position = 0;
                        m_Image = Image.FromStream(ms);
                        pictureBox1.Invalidate();
                    }
                    else if (AniGIF != null)
                    {
                        AniGIF.SelectActiveFrame(new System.Drawing.Imaging.FrameDimension(AniGIF.FrameDimensionsList[0]), m_CurrentImage);                        
                        m_Image = new Bitmap(AniGIF);
                        pictureBox1.Invalidate();
                    }
                    else
                    {
                        byte[] bys = System.IO.File.ReadAllBytes((string)m_FileSequence[m_CurrentImage]);
                        System.IO.MemoryStream ms = new System.IO.MemoryStream(bys);
                        Image im = Image.FromStream(ms);
                        if (im.PixelFormat == System.Drawing.Imaging.PixelFormat.Format8bppIndexed && m_ApplySySalColorTable)
                        {
                            int i;
                            for (i = 0; i < SySalColorTable.GetLength(0); i++)
                            {
                                ms.Seek(54 + 4 * i, System.IO.SeekOrigin.Begin);
                                ms.WriteByte(SySalColorTable[i, 2]);
                                ms.WriteByte(SySalColorTable[i, 1]);
                                ms.WriteByte(SySalColorTable[i, 0]);
                            }
                        }
                        ms.Seek(0, System.IO.SeekOrigin.Begin);
                        im = Image.FromStream(ms);
                        m_Image = im;
                        pictureBox1.Invalidate();                        
                    }
                }
                catch (Exception x)
                {
                    m_Image = null;
                    pictureBox1.Refresh();
                    statusImageN.Text = "Image # ???";
                    return;
                }
                if (m_Image != null)
                {
                    if (pictureBox1.Width != m_Image.Width || pictureBox1.Height != m_Image.Height)
                        pictureBox1.Size = new Size((int)(m_Image.Width / ZoomFactor), (int)(m_Image.Height / ZoomFactor));
                    pictureBox1.Refresh();
                    statusImageN.Text = "Image # " + m_CurrentImage;
                    statusImageSize.Text = "WxH = " + m_Image.Width + "x" + m_Image.Height;
                }
            }
        }

        void Reset()
        {
            m_FileSequence.Clear();
            AniGIF = null;
            m_FogFile = null;
            m_FogFileImagePointers = null;
            DefaultImSeqInfo();
            m_SelRect.X = 0;
            m_SelRect.Y = 0;
            m_SelRect.Width = 0;
            m_SelRect.Height = 0;
            m_Image = null;
            m_CurrentImage = -1;
        }

        private void openSequenceToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofn = new OpenFileDialog();
            ofn.Filter = "SySal bitmap files (*.bmp)|*.bmp";
            ofn.Title = "Select SySal image dump files";
            ofn.Multiselect = true;
            if (ofn.ShowDialog() == DialogResult.OK)
            {
                Reset();
                AniGIF = null;
                m_FileSequence.AddRange(ofn.FileNames);
                reverseToolStripMenuItem.Enabled = true;
                recodeToAnimatedGIFToolStripMenuItem.Enabled = true;
                extractBMPImagesToolStripMenuItem.Enabled = false;
                sharpenImageToolStripMenuItem.Enabled = false;
                CurrentImage = 0;
            }
        }

        private void openAnimatedGIFToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofn = new OpenFileDialog();
            ofn.Filter = "SySal animated GIF files (*.gif)|*.gif";
            ofn.Title = "Select SySal animated GIF files";
            ofn.Multiselect = false;
            if (ofn.ShowDialog() == DialogResult.OK)
            {                
                System.IO.FileStream fs = null;
                TimerPieForm pf = null;
                try
                {
                    Reset();
                    fs = new System.IO.FileStream(ofn.FileName, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                    m_Comment = AniGIFComment(fs);
                    fs.Close();
                    fs = null;                    
                    AniGIF = Image.FromFile(ofn.FileName);
                    m_FogImageSize.Width = AniGIF.Width;
                    m_FogImageSize.Height = AniGIF.Height;
                    System.Drawing.Imaging.FrameDimension dim = new System.Drawing.Imaging.FrameDimension(AniGIF.FrameDimensionsList[0]);
                    m_FogFileImagePointers = new long[AniGIF.GetFrameCount(dim)];
                    m_FogFile = new System.IO.MemoryStream(m_FogFileImagePointers.Length * m_FogImageSize.Width * m_FogImageSize.Height);
                    int i;
                    pf = new TimerPieForm(new TimerPieForm.dNotifyStop(NotifyStop));
                    pf.Progress = 0.0;
                    pf.Show();
                    for (i = 0; i < m_FogFileImagePointers.Length; i++)
                    {
                        Application.DoEvents();
                        if (m_Stop) throw new Exception("Loading aborted.");
                        m_FogFileImagePointers[i] = m_FogFile.Position;
                        AniGIF.SelectActiveFrame(dim, i);
                        System.IO.MemoryStream tms = new System.IO.MemoryStream();
                        
                        BMP8Encode(AniGIF, new System.IO.BinaryWriter(tms));
                        byte[] tbys = tms.ToArray();
                        m_FogFile.Write(tbys, 1078, m_FogImageSize.Height *m_FogImageSize.Width);
                        pf.Progress = (i + 1.0) / m_FogFileImagePointers.Length;
                    }
                    reverseToolStripMenuItem.Enabled = true;
                    recodeToAnimatedGIFToolStripMenuItem.Enabled = true;
                    extractBMPImagesToolStripMenuItem.Enabled = true;
                    sharpenImageToolStripMenuItem.Enabled = true;
                    AniGIF = null;
                    pf.Close();
                    pf = null;
                    try
                    {
                        System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Imaging.ImageSequenceInfo));
                        ImSeqInfo = (SySal.Imaging.ImageSequenceInfo)xmls.Deserialize(new System.IO.StringReader(m_Comment));
                    }
                    catch (Exception)
                    {
                        MessageBox.Show("Cannot understand comment as image information.\r\nDefaulting to nominal 2D information.", "File warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                    CurrentImage = 0;
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error opening animated GIF", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (fs != null) fs.Close();
                    if (pf != null) pf.Close();
                }
            }
        }

        private void frameUpToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (Frames <= 0) CurrentImage = -1;
            else CurrentImage = (CurrentImage + 1) % Frames;
        }

        private void frameRevToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (Frames <= 0) CurrentImage = -1;
            else CurrentImage = (CurrentImage + Frames - 1) % Frames;
        }

        private void OnResize(object sender, EventArgs e)
        {
            Image im = pictureBox1.Image;
            if (im != null)
            {
                int h = im.Height + pictureBox1.Location.Y + statusStrip1.Height;
                if (Width > im.Width) Width = im.Width;
                if (Height > h) Height = h;
            }
        }

        private void OnMouseEnter(object sender, EventArgs e)
        {
            Point m = pictureBox1.PointToClient(Cursor.Position);
            if (m_RegionLock == false) FirstPoint = m;
            double x, y, z;
            int g;
            GetWorldCoordinates(m, out x, out y, out z, out g);
            statusXY.Text = "X;Y;Z;G = " + x.ToString("F1") + ";" + y.ToString("F1") + ";" + z.ToString("F1") + ";" + g.ToString();
        }

        private void OnMouseLeave(object sender, EventArgs e)
        {
            statusXY.Text = "X;Y;Z;G =";
        }

        bool m_RegionLock = false;

        void GetWorldCoordinates(Point m, out double x, out double y, out double z, out int g)
        {
            if (m_Image == null)
            {
                x = m.X;
                y = m.Y;
                z = 0.0;
                g = -1;
            }
            else
            {
                int i, ci;
                for (i = ci = 0; i < ImSeqInfo.EmulsionLayers.Length && ci + ImSeqInfo.EmulsionLayers[i].DepthInfo.Length < CurrentImage; i++) ci += ImSeqInfo.EmulsionLayers[i].DepthInfo.Length;
                if (i < ImSeqInfo.EmulsionLayers.Length)
                    z = ImSeqInfo.EmulsionLayers[i].DepthInfo[CurrentImage - ci].Z;
                else z = 0.0;
                double zf = ZoomFactor;
                double mx, my;
                mx = Math.Round(m.X * zf);
                my = Math.Round(m.Y * zf);

                x = ImSeqInfo.Info2D.Center.X + ImSeqInfo.Info2D.PixelToMicronXX * (mx - m_Image.Width * 0.5) + ImSeqInfo.Info2D.PixelToMicronXY * (m_Image.Height * 0.5 - my);
                y = ImSeqInfo.Info2D.Center.Y + ImSeqInfo.Info2D.PixelToMicronYX * (mx - m_Image.Width * 0.5) + ImSeqInfo.Info2D.PixelToMicronYY * (m_Image.Height * 0.5 - my);
                if (mx >= 0 && mx < m_Image.Width && my >= 0 && my <= m_Image.Height)
                {
                    Color c = ((Bitmap)m_Image).GetPixel((int)mx, (int)my);
                    g = (c.R + c.G + c.B) / 3;
                }
                else g = -1;
            }
        }

        private void OnMouseMove(object sender, MouseEventArgs e)
        {
            Point m = pictureBox1.PointToClient(Cursor.Position);            
            double x, y, z;
            int g;
            GetWorldCoordinates(m, out x, out y, out z, out g);
            statusXY.Text = "X;Y;Z;G = " + x.ToString("F1") + ";" + y.ToString("F1") + ";" + z.ToString("F1") + ";" + g.ToString();
            if (e.Button == MouseButtons.Left)
            {
                SecPoint = m;
                double zf = ZoomFactor;
                SecPoint.X = (int)(SecPoint.X * zf);
                SecPoint.Y = (int)(SecPoint.Y * zf);
                Rectangle rc = new Rectangle((int)(Math.Min(FirstPoint.X, SecPoint.X) / zf), (int)(Math.Min(FirstPoint.Y, SecPoint.Y) / zf), (int)((Math.Abs(SecPoint.X - FirstPoint.X) + 1) / zf), (int)((Math.Abs(SecPoint.Y - FirstPoint.Y) + 1) / zf));
                pictureBox1.Refresh();                
                pictureBox1.CreateGraphics().DrawRectangle(new Pen(Color.Red), rc);                
            }
            else if (m_RegionLock == false) FirstPoint = m;
        }

        private void zoomInToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Zoom = Zoom - 1;
        }

        private void zoomOutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Zoom = Zoom + 1;
        }

        private void commentToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (m_FogFileImagePointers != null)
            {
                System.Collections.ArrayList ar = new System.Collections.ArrayList();
                ar.AddRange(m_FogFileImagePointers);
                ar.Reverse();
                m_FogFileImagePointers = (long[])ar.ToArray(typeof(long));
            }
            else m_FileSequence.Reverse();
            CurrentImage = Frames - 1 - CurrentImage;
        }

        Point FirstPoint;
        Point SecPoint;

        private void OnMouseDown(object sender, MouseEventArgs e)
        {
            FirstPoint = pictureBox1.PointToClient(Cursor.Position);
            double zf = ZoomFactor;
            FirstPoint.X = (int)(FirstPoint.X * zf);
            FirstPoint.Y = (int)(FirstPoint.Y * zf);
        }

        private void OnMouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                m_RegionLock = true;
                Refresh();
            }
            else if (e.Button == MouseButtons.Right)
            {
                m_RegionLock = false;
                Refresh();
            }
        }

        private void OnPaint(object sender, PaintEventArgs e)
        {
            if (Frames == 0 || m_Image == null) base.OnPaint(e);
            else
            {
                double zf = ZoomFactor;                               
                e.Graphics.DrawImage(m_Image, pictureBox1.ClientRectangle, 0, 0, (float)(pictureBox1.Width * zf), (float)(pictureBox1.Height * zf), GraphicsUnit.Pixel);                
                if (m_RegionLock)
                {
                    Rectangle rc = new Rectangle((int)(Math.Min(FirstPoint.X, SecPoint.X) / zf), (int)(Math.Min(FirstPoint.Y, SecPoint.Y) / zf), (int)((Math.Abs(SecPoint.X - FirstPoint.X) + 1) / zf), (int)((Math.Abs(SecPoint.Y - FirstPoint.Y) + 1) / zf));
                    e.Graphics.DrawRectangle(new Pen(Color.LightGreen), rc);                    
                }
            }
        }

        private void unlockRegionToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_RegionLock = false;
            Refresh();
        }

        bool m_Stop = false;

        void NotifyStop()
        {
            m_Stop = true;
        }

        private void recodeToAnimatedGIFToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int minx = 0, maxx = 0, miny = 0, maxy = 0;
            int width, height, newwidth = 0, newheight = 0, offset;
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Filter = "Animated GIF files (*.gif)|*.gif|All files (*.*)|*.*";
            sdlg.Title = "Select output animated GIF file";
            if (sdlg.ShowDialog() != DialogResult.OK) return;
            TimerPieForm pf = null;
            try
            {
                if (m_RegionLock)
                {
                    minx = (Math.Min(FirstPoint.X, SecPoint.X) >> 2) << 2;
                    miny = (Math.Min(FirstPoint.Y, SecPoint.Y) >> 2) << 2;
                    maxx = Math.Max(FirstPoint.X, SecPoint.X) | 3;
                    maxy = Math.Max(FirstPoint.Y, SecPoint.Y) | 3;                    
                }
                GDI3D.Movie movie = new GDI3D.Movie(1);
                SySal.Imaging.ImageSequenceInfo i2info = ImSeqInfo;
                int cx = ((minx + maxx) - m_FogImageSize.Width) / 2;
                int cy = ((miny + maxy) - m_FogImageSize.Height) / 2;
                i2info.Info2D.Center.X += i2info.Info2D.PixelToMicronXX * cx + i2info.Info2D.PixelToMicronXY * cy;
                i2info.Info2D.Center.Y += i2info.Info2D.PixelToMicronYX * cx + i2info.Info2D.PixelToMicronYY * cy;
                System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(i2info.GetType());
                System.IO.StringWriter sw = new System.IO.StringWriter();
                xmls.Serialize(sw, i2info);
                movie.Comment = sw.ToString();
                m_Stop = false;
                pf = new TimerPieForm(new TimerPieForm.dNotifyStop(NotifyStop));
                pf.Progress = 0.0;                
                pf.Show();
                int frame;
                for (frame = 0; frame < Frames; frame++)                
                {
                    if (m_Stop)
                    {
                        MessageBox.Show("Stopped", "No file generated", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        pf.Close();
                        return;
                    }
                    byte[] bys;
                    if (m_FogFileImagePointers == null)
                    {
                        bys = System.IO.File.ReadAllBytes(m_FileSequence[frame].ToString());
                    }
                    else
                    {
                        System.IO.MemoryStream mst = BMP8Stream(m_FogImageSize.Width, m_FogImageSize.Height);                        
                        m_FogFile.Position = m_FogFileImagePointers[frame];
                        int s = m_FogImageSize.Width * m_FogImageSize.Height;
                        while (s-- >= 0) mst.WriteByte((byte)m_FogFile.ReadByte());
                        bys = mst.ToArray();
                    }
                    System.IO.MemoryStream ms = new System.IO.MemoryStream(bys);
                    System.IO.BinaryReader msr = new System.IO.BinaryReader(ms);
                    ms.Position = 10;
                    offset = msr.ReadInt32();
                    ms.Position = 18;
                    width = msr.ReadInt32();
                    height = msr.ReadInt32();
                    if (m_RegionLock == false)
                    {
                        minx = 0;
                        maxx = width - 1;
                        miny = 0;
                        maxy = height - 1;
                    }
                    newwidth = maxx - minx + 1;
                    newheight = maxy - miny + 1;
                    System.IO.MemoryStream msnew = new System.IO.MemoryStream(offset + newwidth * newheight);
                    System.IO.BinaryWriter msneww = new System.IO.BinaryWriter(msnew);
                    msnew.Write(bys, 0, offset);                    
                    msnew.Position = 18;
                    msneww.Write(newwidth);
                    msneww.Write(newheight);
                    msneww.Flush();
                    msnew.Position = offset;
                    int y;
                    for (y = maxy; y >= miny; y--)
                        msneww.Write(bys, offset + (height - 1 - y) * width + minx, newwidth);
                    int i;
                    msneww.Flush();

                    msnew.Seek(0, System.IO.SeekOrigin.Begin);
                    for (i = 0; i < SySalColorTable.GetLength(0); i++)
                    {
                        msnew.Seek(54 + 4 * i, System.IO.SeekOrigin.Begin);
                        msnew.WriteByte(SySalColorTable[i, 2]);
                        msnew.WriteByte(SySalColorTable[i, 1]);
                        msnew.WriteByte(SySalColorTable[i, 0]);                        
                    }
                    msnew.Seek(0, System.IO.SeekOrigin.Begin);
                    System.Drawing.Image im = System.Drawing.Image.FromStream(msnew);
                    movie.AddFrame(im);
                    pf.Progress = (double)(frame + 1.0)/Frames;
                    Application.DoEvents();
                }
                pf.Progress = 1.0;
                Application.DoEvents();
                pf.Close();
                movie.Save(sdlg.FileName);
                MessageBox.Show("Movie created", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                if (pf != null) pf.Close();
                MessageBox.Show(x.ToString(), "Error creating movie", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        bool m_ApplySySalColorTable = true;

        private void applySySalColorTToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_ApplySySalColorTable = !m_ApplySySalColorTable;
            applySySalColorTToolStripMenuItem.Checked = m_ApplySySalColorTable;
            CurrentImage = CurrentImage;
        }

        private void OnLoad(object sender, EventArgs e)
        {
            DefaultImSeqInfo();
            applySySalColorTToolStripMenuItem.Checked = m_ApplySySalColorTable;
            pictureBox1.Cursor = Cursors.UpArrow;           
        }

        CommentForm m_CommentForm = new CommentForm();

        string m_Comment = "";

        private void commentToolStripMenuItem_Click_1(object sender, EventArgs e)
        {
            m_CommentForm.txtComment.Text = m_Comment;
            if (m_CommentForm.ShowDialog() == DialogResult.OK)
            {
                m_Comment = m_CommentForm.txtComment.Text;
            }
        }

        private void extractBMPImagesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Filter = "SySal BMP files (*.bmp)|*.bmp";
            sdlg.Title = "Select base path of files to be generated";
            if (sdlg.ShowDialog() == DialogResult.OK)
            {
                TimerPieForm pf = null;
                int ci = CurrentImage;
                string basepath = sdlg.FileName;
                if (basepath.ToLower().EndsWith(".bmp")) basepath = basepath.Remove(basepath.Length - 4);
                try
                {
                    pf = new TimerPieForm(new TimerPieForm.dNotifyStop(NotifyStop));
                    pf.Progress = 0.0;
                    pf.Show();
                    m_Stop = false;
                    int f = Frames;
                    int i;
                    int digits = (int)Math.Max(Math.Ceiling(Math.Log10((double)Frames)),1.0);
                    for (i = 0; i < f && !m_Stop; i++)
                    {
                        CurrentImage = i;
                        System.IO.FileStream fs = new System.IO.FileStream(basepath + i.ToString("D" + "".PadRight(digits, '0') + digits) + ".bmp", System.IO.FileMode.Create);
                        BMP8Encode(m_Image, new System.IO.BinaryWriter(fs));
                        fs.Flush();
                        fs.Close();                        
                        pf.Progress = ((double)i) / ((double)f);
                        Application.DoEvents();
                    }
                    if (m_Stop) throw new Exception("Stopped");
                    pf.Progress = 1.0;
                    MessageBox.Show("File sequence saved", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                CurrentImage = ci;
                if (pf != null) pf.Close();                              
            }
        }

        static System.IO.MemoryStream BMP8Stream(int width, int height)
        {
            System.IO.MemoryStream ms = new System.IO.MemoryStream(1078 + width * height);
            System.IO.BinaryWriter w = new System.IO.BinaryWriter(ms);
            w.Write('B');
            w.Write('M');
            w.Write(1078 + width * height);
            w.Write((int)0);
            w.Write((int)1078);
            w.Write((int)40);
            w.Write((int)width);
            w.Write((int)height);
            w.Write((short)1);
            w.Write((short)8);
            w.Write((int)0);
            w.Write((int)0);
            w.Write((int)10000);
            w.Write((int)10000);
            w.Write((int)0);
            w.Write((int)0);
            int i;
            for (i = 0; i < SySalColorTable.GetLength(0); i++)
            {
                w.Write((byte)SySalColorTable[i, 2]);
                w.Write((byte)SySalColorTable[i, 1]);
                w.Write((byte)SySalColorTable[i, 0]);
                w.Write((byte)0);
            }
            while (i++ < 256)
            {
                w.Write((byte)i);
                w.Write((byte)i);
                w.Write((byte)i);
                w.Write((byte)0);
            }
            w.Flush();
            return ms;
        }

        static void BMP8Encode(Image im, System.IO.BinaryWriter w)
        {
            Bitmap bmp = new Bitmap(im);
            w.Write('B');
            w.Write('M');
            w.Write(1078 + im.Width * im.Height);
            w.Write((int)0);
            w.Write((int)1078);
            w.Write((int)40);
            w.Write((int)im.Width);
            w.Write((int)im.Height);
            w.Write((short)1);
            w.Write((short)8);
            w.Write((int)0); 
            w.Write((int)0);
            w.Write((int)10000);
            w.Write((int)10000);
            w.Write((int)0);
            w.Write((int)0);
            int i;
            for (i = 0; i < SySalColorTable.GetLength(0); i++)
            {
                w.Write((byte)SySalColorTable[i, 2]);
                w.Write((byte)SySalColorTable[i, 1]);
                w.Write((byte)SySalColorTable[i, 0]);
                w.Write((byte)0);
            }
            while (i++ < 256)
            {
                w.Write((byte)i);
                w.Write((byte)i);
                w.Write((byte)i);
                w.Write((byte)0);
            }
            System.IO.MemoryStream ms = new System.IO.MemoryStream();
            im.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
            ms.Position = 54;
            int s = im.Width * im.Height;
            byte cR, cG, cB, cA;
            while (s-- >= 0)
                {
                    cB = (byte)ms.ReadByte();
                    cG = (byte)ms.ReadByte();
                    cR = (byte)ms.ReadByte();
                    cA = (byte)ms.ReadByte();
                    byte bc = 0;
                    if (cR == cG && cG == cB)
                    {
                        bc = cR;
                        if (bc < SySalColorTable.GetLength(0)) bc = (byte)SySalColorTable.GetLength(0);
                    }
                    else
                    {
                        int dist = 0;
                        int best = -1;
                        int bestdist = 256;
                        for (i = 0; i < SySalColorTable.GetLength(0); i++)
                        {
                            dist = Math.Max(Math.Abs((int)(uint)cR - (int)SySalColorTable[i, 0]), 
                                Math.Max(Math.Abs((int)(uint)cG - (int)SySalColorTable[i, 1]),
                                    Math.Abs((int)(uint)cB - (int)SySalColorTable[i, 2])));
                            if (dist < bestdist)
                            {
                                best = i;
                                bestdist = dist;
                            }
                        }
                        bc = (byte)best;
                    }
                    w.Write(bc);
                }
        }

        private void openFogImageSequenceFileToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofn = new OpenFileDialog();
            ofn.Filter = "Fog image sequence file (*.mic)|*.mic";
            ofn.Title = "Select Fog image sequence file";
            ofn.Multiselect = false;
            if (ofn.ShowDialog() == DialogResult.OK)
            {
                Reset();
                AniGIF = null;
                GDI3D.Movie movie = new GDI3D.Movie(1);
                System.IO.FileStream rf = null;
                System.IO.BinaryReader rb = null;
                m_FogFile = new System.IO.MemoryStream();
                TimerPieForm pf = null;
                System.Collections.ArrayList lpos = new System.Collections.ArrayList();
                try
                {
                    pf = new TimerPieForm(new TimerPieForm.dNotifyStop(NotifyStop));
                    pf.Progress = 0.0;
                    pf.Show();
                    rf = new System.IO.FileStream(ofn.FileName, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                    rb = new System.IO.BinaryReader(rf);
                    string header = new string((rb.ReadChars(16)));
                    if (String.Compare(header, "Opera Mic View ", true) != 0) throw new Exception("Unknown file format - check header.");                    
                    int version = rb.ReadInt32();
                    if (version == 1)
                    {                        
                        int sides = 1 + rb.ReadInt32();
                        int[] frames = new int[2] { rb.ReadInt32(), rb.ReadInt32() };
                        rb.ReadSingle(); rb.ReadSingle();
                        ImSeqInfo.Info2D.Center.X = (double)rb.ReadSingle();
                        ImSeqInfo.Info2D.Center.Y = (double)rb.ReadSingle();
                        ImSeqInfo.Info2D.PixelToMicronXX = (double)rb.ReadDouble();
                        ImSeqInfo.Info2D.PixelToMicronXY = (double)rb.ReadDouble();
                        ImSeqInfo.Info2D.PixelToMicronYX = (double)rb.ReadDouble();
                        ImSeqInfo.Info2D.PixelToMicronYY = (double)rb.ReadDouble();
                        rb.ReadSingle(); rb.ReadSingle();
                        rf.Position = 168;
                        int topgrains = rb.ReadInt32();
                        int toptracks = rb.ReadInt32();
                        rf.Position = 232;
                        ImSeqInfo.Info2D.Id.Part0 = rb.ReadInt32();
                        ImSeqInfo.Info2D.Id.Part1 = rb.ReadInt32();
                        ImSeqInfo.Info2D.Id.Part2 = ImSeqInfo.Info2D.Id.Part3 = 0;
                        ImSeqInfo.EmulsionLayers = new SySal.Imaging.EmulsionLayerImageDepthInfo[2];
                        rf.Position = 272;
                        ImSeqInfo.EmulsionLayers[0].TopZ = (double)rb.ReadSingle();
                        ImSeqInfo.EmulsionLayers[0].BottomZ = (double)rb.ReadSingle();
                        rf.Position += 14;
                        int i, y, x;
                        ImSeqInfo.EmulsionLayers[0].DepthInfo = new SySal.Imaging.ImageDepthInfo[frames[0]];
                        for (i = 0; i < frames[0]; i++)
                            ImSeqInfo.EmulsionLayers[0].DepthInfo[i].Z = (double)rb.ReadSingle();
                        rf.Position += 12 * topgrains;
                        for (i = 0; i < toptracks; i++)
                        {
                            rf.Position += 48;
                            int grains = rb.ReadInt32();
                            rf.Position += 44 + 28 * grains;
                        }
                        m_FogImageSize.Width = 1280;
                        m_FogImageSize.Height = 1024;
                        for (i = 0; i < frames[0]; i++)
                        {
                            Application.DoEvents();
                            if (m_Stop) throw new Exception("Loading aborted.");
                            lpos.Add(m_FogFile.Position);
                            long basepos = rf.Position;
                            for (y = m_FogImageSize.Height - 1; y >= 0; y--)
                            {
                                rf.Position = basepos + y * m_FogImageSize.Width;
                                for (x = 0; x < m_FogImageSize.Width; x++)
                                {
                                    byte b = rb.ReadByte();
                                    if (b < SySalColorTable.Length) b = (byte)SySalColorTable.Length;
                                    m_FogFile.WriteByte(b);
                                }
                            }
                            rf.Position = basepos + m_FogImageSize.Width * m_FogImageSize.Height;
                            pf.Progress = (0.5 * (i + 1)) / frames[0];
                        }
                        rf.Position += 80;
                        int bottomgrains = rb.ReadInt32();
                        int bottomtracks = rb.ReadInt32();
                        rf.Position += 56;
                        ImSeqInfo.EmulsionLayers[1].TopZ = (double)rb.ReadSingle();
                        ImSeqInfo.EmulsionLayers[1].BottomZ = (double)rb.ReadSingle();
                        rf.Position += 14;
                        ImSeqInfo.EmulsionLayers[1].DepthInfo = new SySal.Imaging.ImageDepthInfo[frames[1]];
                        for (i = 0; i < frames[1]; i++)
                            ImSeqInfo.EmulsionLayers[1].DepthInfo[i].Z = (double)rb.ReadSingle();
                        rf.Position += 12 * bottomgrains;
                        for (i = 0; i < bottomtracks; i++)
                        {
                            rf.Position += 48;
                            int grains = rb.ReadInt32();
                            rf.Position += 44 + 28 * grains;
                        }
                        for (i = 0; i < frames[1]; i++)
                        {
                            Application.DoEvents();
                            if (m_Stop) throw new Exception("Loading aborted.");
                            lpos.Add(m_FogFile.Position);
                            long basepos = rf.Position;
                            for (y = m_FogImageSize.Height - 1; y >= 0; y--)
                            {
                                rf.Position = basepos + y * m_FogImageSize.Width;
                                for (x = 0; x < m_FogImageSize.Width; x++)
                                {
                                    byte b = rb.ReadByte();
                                    if (b < SySalColorTable.Length) b = (byte)SySalColorTable.Length;
                                    m_FogFile.WriteByte(b);
                                }
                            }
                            rf.Position = basepos + m_FogImageSize.Width * m_FogImageSize.Height;
                            pf.Progress = 0.5 + (0.5 * (i + 1)) / frames[1];
                        }
                        System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(ImSeqInfo.GetType());
                        System.IO.StringWriter sw = new System.IO.StringWriter();
                        xmls.Serialize(sw, ImSeqInfo);
                        m_Comment = sw.ToString();
                        m_FogFileImagePointers = (long[])lpos.ToArray(typeof(long));
                        reverseToolStripMenuItem.Enabled = true;
                        recodeToAnimatedGIFToolStripMenuItem.Enabled = true;
                        extractBMPImagesToolStripMenuItem.Enabled = true;
                        sharpenImageToolStripMenuItem.Enabled = true;
                        CurrentImage = 0;
                    }
                    else
                    {
                        throw new Exception("Unsupported Fog file version " + version + ".");
                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error reading file.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    m_FogFile = null;
                    m_FogFileImagePointers = null;
                }
                finally
                {
                    if (rf != null) rf.Close();
                    rb = null;
                    rf = null;
                    if (pf != null) pf.Close();
                }
            }
        }

        Size RemapCellSize = new Size(32, 32);

        int RemapGreyLevels = 16;        

        int[,] FilterKernel = new int[3, 3]
            {
                {1,   2, 1},
                {2, -12, 2},
                {1,   2, 1}
            };

        FilterControlForm m_FC = new FilterControlForm();

        private void sharpenImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (m_FC.ShowDialog() != DialogResult.OK) return;
            int i, x, y, ix, iy;
            TimerPieForm pf = null;
            pf = new TimerPieForm(new TimerPieForm.dNotifyStop(NotifyStop));
            pf.Progress = 0.0;
            pf.Show();
            int s = m_FogImageSize.Width * m_FogImageSize.Height;            
            System.IO.MemoryStream nms = new System.IO.MemoryStream((int)m_FogFile.Length);
            double k = m_FC.m_FilterMult;
            double z = m_FC.m_FilterZeroThresh;
            double m = m_FC.m_ImageMult;
            double p = m_FC.m_ImageOffset;
            foreach (long pos in m_FogFileImagePointers)
            {
                Application.DoEvents();
                if (m_Stop)
                {
                    pf.Close();
                    return;
                }
                m_FogFile.Position = pos;
                byte[,] imb = new byte[m_FogImageSize.Height, m_FogImageSize.Width];
                for (i = 0; i < s; i++)
                {
                    y = i / m_FogImageSize.Width;
                    x = i - y * m_FogImageSize.Width;
                    imb[y, x] = (byte)m_FogFile.ReadByte();
                }
                int[,] fb = new int[m_FogImageSize.Height, m_FogImageSize.Width];
                for (y = 1; y < m_FogImageSize.Height - 1; y++)
                    for (x = 1; x < m_FogImageSize.Width -1; x++)                    
                    {
                        int v = 0;
                        for (iy = -1; iy <= 1; iy++)
                            for (ix = -1; ix <= 1; ix++)                            
                                v += (int)imb[y + iy, x + ix] * FilterKernel[iy + 1, ix + 1];
                        fb[y, x] = (int)Math.Min(z, v * k);                        
                    }
                for (y = 0; y < m_FogImageSize.Height; y++)
                    for (x = 0; x < m_FogImageSize.Width; x++)
                    {
                        int v = fb[y, x] + (int)(imb[y, x] * m + p);
                        if (v < SySalColorTable.Length) v = SySalColorTable.Length;
                        else if (v > 255) v = 255;
                        nms.WriteByte((byte)v);
                    }
                pf.Progress += 1.0 / m_FogFileImagePointers.Length;
            }
            m_FogFile = nms;
            CurrentImage = CurrentImage;
            pf.Close();
        }
    }
}