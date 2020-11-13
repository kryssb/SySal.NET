using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using SySal.Management;

namespace SySal.ImageProcessorDisplay
{
    public partial class ImageDisplayForm : Form, IMachineSettingsEditor
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

        public ImageDisplayForm()
        {
            InitializeComponent();
            SySal.ImageProcessorDisplay.Configuration c = (Configuration)SySal.Management.MachineSettings.GetSettings(typeof(Configuration));
            if (c == null)
            {
                c = new Configuration();
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
            System.Drawing.Bitmap z = new Bitmap(8,8, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            GrayScalePalette = z.Palette;
            int i;
            for (i = 0; i < GrayScalePalette.Entries.Length; i++)
                GrayScalePalette.Entries[i] = Color.FromArgb(i, i, i);
        }

        System.Drawing.Imaging.ColorPalette GrayScalePalette;

        private void DisplayClusters(System.Drawing.Graphics g, SySal.Imaging.Cluster[] clusters)
        {
            //float f = (float)pbScreen.Width / (float)m_HostedFormat.Width;            
            float f = 1.0f;
            System.Drawing.Pen cpen = new Pen(Color.Coral, 1);
            foreach (SySal.Imaging.Cluster c in clusters)
            {
                int side = (int)Math.Sqrt(c.Area) / 2 + 1;
                g.DrawEllipse(cpen, (float)(f * (c.X - side)), (float)(f * (c.Y - side)), (float)(2 * side), (float)(2 * side));
            }
        }

        public SySal.Imaging.LinearMemoryImage ImageShown
        {
            set
            {
                string errorstring = "";
                SySal.Imaging.LinearMemoryImage lmi = value;
                try
                {
                    m_ImageProcessor.Input = value;
                    if (m_ImageProcessor != null)
                    {
                        if (m_ShowBinary) lmi = m_ImageProcessor.BinarizedImages;
                        txtGreyLevelMedian.Text = m_ImageProcessor.GreyLevelMedian.ToString();
                        txtClusters.Text = m_ImageProcessor.Clusters[0].Length.ToString();
                        txtClusters.BackColor = (m_ImageProcessor.Warnings.Length > 0) ? Color.Coral : SystemColors.Control;                        
                    }
                    else
                    {
                        txtGreyLevelMedian.Text = "";
                        txtClusters.Text = "";
                        txtClusters.BackColor = SystemColors.Control;
                    }
                }
                catch (Exception exc) { errorstring = exc.ToString(); }
                System.Drawing.Bitmap bmp = new Bitmap(lmi.Info.Width, lmi.Info.Height, lmi.Info.Width, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, ImageAccessor.Scan(lmi));
                bmp.Palette = GrayScalePalette;
                System.Drawing.Bitmap mbmp = new Bitmap(lmi.Info.Width, lmi.Info.Height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(mbmp);
                g.DrawImageUnscaled(bmp, 0, 0);
                if (errorstring.Length > 0) g.DrawString(errorstring, new Font("Arial", 12), new SolidBrush(Color.Red), 0.0f, 0.0f);
                if (m_ImageProcessor != null && chkShowClusters.Checked) DisplayClusters(g, m_ImageProcessor.Clusters[0]);
                pbScreen.Image = mbmp;                
            }
        }

        #region IMachineSettingsEditor Members

        public bool EditMachineSettings(Type t)
        {
            Configuration C = (Configuration)SySal.Management.MachineSettings.GetSettings(t);
            if (C == null)
            {
                MessageBox.Show("No valid configuration found, switching to default", "Configuration warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                C = new Configuration();
                C.Name = "Default ImageProcessorDisplay configuration";
                C.PanelLeft = 0;
                C.PanelTop = 0;
                C.PanelWidth = 640;
                C.PanelHeight = 480;
            }
            EditConfigForm ef = new EditConfigForm(C);
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
            if (EditMachineSettings(typeof(SySal.ImageProcessorDisplay.Configuration)))
            {
                SySal.ImageProcessorDisplay.Configuration c = (Configuration)SySal.Management.MachineSettings.GetSettings(typeof(SySal.ImageProcessorDisplay.Configuration));
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
    }
}