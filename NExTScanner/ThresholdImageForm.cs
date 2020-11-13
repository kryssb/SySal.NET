using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class ThresholdImageForm : SySal.SySalNExTControls.SySalDialog
    {
        public class ThresholdImageComputationSettings : SySal.Management.Configuration
        {
            public uint MinimumThreshold;
            public uint MaximumThreshold;
            public uint ThresholdSteps;
            public uint XWaves;
            public uint YWaves;
            public uint XCells;
            public uint YCells;
            public uint CellWidth;
            public uint CellHeight;
            public uint MinClusterSize;
            public uint MaxClusterSize;

            public ThresholdImageComputationSettings()
                : base("")
            {
                MinimumThreshold = 100;
                MaximumThreshold = 200;
                ThresholdSteps = 16;
                MinClusterSize = 4;
                MaxClusterSize = 64;
                XWaves = 4;
                YWaves = 4;
                XCells = 11;
                YCells = 11;
                CellWidth = 160;
                CellHeight = 128;
            }

            public override object Clone()
            {
                ThresholdImageComputationSettings s = new ThresholdImageComputationSettings();
                s.Name = Name;
                s.MinClusterSize = MinClusterSize;
                s.MaxClusterSize = MaxClusterSize;
                s.MinimumThreshold = MinimumThreshold;
                s.MaximumThreshold = MaximumThreshold;
                s.ThresholdSteps = ThresholdSteps;
                s.XWaves = XWaves;
                s.YWaves = YWaves;
                s.XCells = XCells;
                s.YCells = YCells;
                s.CellWidth = CellWidth;
                s.CellHeight = CellHeight;
                return s;
            }

            internal static ThresholdImageComputationSettings Stored
            {
                get
                {
                    ThresholdImageComputationSettings c = SySal.Management.MachineSettings.GetSettings(typeof(ThresholdImageComputationSettings)) as ThresholdImageComputationSettings;
                    if (c == null) c = new ThresholdImageComputationSettings();
                    return c;
                }
            }
        }

        public ThresholdImageForm()
        {
            InitializeComponent();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        public SySal.Imaging.ImageInfo m_ImageFormat;

        ThresholdImageComputationSettings S = ThresholdImageComputationSettings.Stored;

        public SySal.Imaging.IImageProcessor[] iGPU;

        string[] m_Files = new string[0];

        public string m_Result = "";

        private void btnSelImages_Click(object sender, EventArgs e)
        {
            if (OpenImageFileDlg.ShowDialog() == DialogResult.OK)
            {
                if (OpenImageFileDlg.FileNames.Length <= 0)
                {
                    MessageBox.Show("At least one image must be selected.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                m_Files = OpenImageFileDlg.FileNames;
                txtSampleImages.Text = (m_Files.Length).ToString() + " image(s)";
            }
        }

        void ShowSettings()
        {
            txtMinThreshold.Text = S.MinimumThreshold.ToString();
            txtMaxThreshold.Text = S.MaximumThreshold.ToString();
            txtThresholdSteps.Text = S.ThresholdSteps.ToString();
            txtMinSize.Text = S.MinClusterSize.ToString();
            txtMaxSize.Text = S.MaxClusterSize.ToString();            
            txtXDCTWavelets.Text = S.XWaves.ToString();
            txtYDCTWavelets.Text = S.YWaves.ToString();
            txtXCells.Text = S.XCells.ToString();
            txtYCells.Text = S.YCells.ToString();
            txtCellWidth.Text = S.CellWidth.ToString();
            txtCellHeight.Text = S.CellHeight.ToString();
        }

        private void OnMinThresholdLeave(object sender, EventArgs e)
        {
            try
            {
                S.MinimumThreshold = uint.Parse(txtMinThreshold.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnMaxThresholdLeave(object sender, EventArgs e)
        {
            try
            {
                S.MaximumThreshold = uint.Parse(txtMaxThreshold.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnThresholdStepsLeave(object sender, EventArgs e)
        {
            try
            {
                S.ThresholdSteps = uint.Parse(txtThresholdSteps.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnMinClusterSizeLeave(object sender, EventArgs e)
        {
            try
            {
                S.MinClusterSize = uint.Parse(txtMinSize.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnMaxClusterSizeLeave(object sender, EventArgs e)
        {
            try
            {
                S.MaxClusterSize = uint.Parse(txtMaxSize.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnXWavesLeave(object sender, EventArgs e)
        {
            try
            {
                S.XWaves = uint.Parse(txtXDCTWavelets.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnYWavesLeave(object sender, EventArgs e)
        {
            try
            {
                S.YWaves = uint.Parse(txtYDCTWavelets.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnCellWidthLeave(object sender, EventArgs e)
        {
            try
            {
                S.CellWidth = uint.Parse(txtCellWidth.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnCellHeightLeave(object sender, EventArgs e)
        {
            try
            {
                S.CellHeight = uint.Parse(txtCellHeight.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnXCellsLeave(object sender, EventArgs e)
        {
            try
            {
                S.XCells = uint.Parse(txtXCells.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnYCellsLeave(object sender, EventArgs e)
        {
            try
            {
                S.YCells = uint.Parse(txtYCells.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }
        }

        private void OnLoad(object sender, EventArgs e)
        {            
            ShowSettings();
        }

        const int HiDensValues = 4;

        private void btnCompute_Click(object sender, EventArgs e)
        {
            if (m_Running)
            {
                m_Running = false;
                return;
            }
            if (S.MinimumThreshold == S.MaximumThreshold)
            {
                SySal.Imaging.DCTInterpolationImage dct = null;
                try
                {
                    SySal.Imaging.ImageInfo info = m_ImageFormat;
                    info.BitsPerPixel = 16;
                    SySal.Imaging.DCTInterpolationImage.PointValue pval = new SySal.Imaging.DCTInterpolationImage.PointValue();
                    pval.X = (ushort)(info.Width / 2);
                    pval.Y = (ushort)(info.Height / 2);
                    pval.Value = (int)S.MaximumThreshold;
                    dct = new SySal.Imaging.DCTInterpolationImage(info, 1, 1, new SySal.Imaging.DCTInterpolationImage.PointValue [] { pval });
                    m_Result = txtResult.Text = dct.ToString();
                    MessageBox.Show("Constant threshold image built.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                catch (Exception x)
                {
                    MessageBox.Show("Invalid DCT built.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }       
            }
            if (m_Files.Length <= 0) 
            {
                MessageBox.Show("Please choose sample image files.", "Input missing", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            double[][,] densities = null;
            SySal.Imaging.Image[] thresholdimgs = null;
            int[] cellmeanx = new int[S.XCells];
            int[] cellmeany = new int[S.YCells];
            int[] cellminx = new int[S.XCells];
            int[] cellmaxx = new int[S.YCells];
            int[] cellminy = new int[S.XCells];
            int[] cellmaxy = new int[S.YCells];
            m_Running = true;
            EnableButtons();
            pbProgress.Minimum = 0.0;
            pbProgress.Maximum = (double)m_Files.Length;
            pbProgress.Value = 0.0;
            System.Threading.Thread execthread = new System.Threading.Thread(new System.Threading.ThreadStart(delegate()
                {
                    thresholdimgs = new SySal.Imaging.Image[S.ThresholdSteps + 1];
                    int i, j, ix, iy;
                    SySal.Imaging.ImageInfo m_info = m_ImageFormat;
                    m_info.BitsPerPixel = 16;
                    int bestgpu = 0;
                    int maximages = iGPU[bestgpu].MaxImages;
                    for (i = 1; i < iGPU.Length; i++)
                        if (iGPU[i].MaxImages > maximages)
                        {
                            bestgpu = i;
                            maximages = iGPU[i].MaxImages;
                        };
                    int xstep = (int)(m_ImageFormat.Width / S.XCells);
                    int ystep = (int)(m_ImageFormat.Height / S.YCells);
                    for (i = 0; i < S.XCells; i++)
                    {
                        cellmeanx[i] = (int)(xstep * (i + 0.5));
                        cellminx[i] = cellmeanx[i] - (int)S.CellWidth / 2;
                        cellmaxx[i] = cellmeanx[i] + (int)S.CellWidth / 2;
                    }
                    for (i = 0; i < S.YCells; i++)
                    {
                        cellmeany[i] = (int)(ystep * (i + 0.5));
                        cellminy[i] = cellmeany[i] - (int)S.CellHeight / 2;
                        cellmaxy[i] = cellmeany[i] + (int)S.CellHeight / 2;
                    }
                    densities = new double[thresholdimgs.Length][,];
                    int[] files = new int[thresholdimgs.Length];
                    for (j = 0; j <= S.ThresholdSteps; j++)
                    {
                        densities[j] = new double[S.XCells, S.YCells];
                        thresholdimgs[j] = new SySal.Imaging.Image(m_info, new ConstThresholdImagePixels((short)(S.MinimumThreshold + (j * (S.MaximumThreshold - S.MinimumThreshold)) / S.ThresholdSteps)));
                    }
                    for (i = 0; i < m_Files.Length; i++)
                    {
                        SySal.Imaging.LinearMemoryImage im = null;
                        try
                        {
                            im = iGPU[bestgpu].ImageFromFile(m_Files[i]);
                            for (j = 0; j < thresholdimgs.Length; j++)
                            {
                                iGPU[bestgpu].ThresholdImage = thresholdimgs[j];
                                iGPU[bestgpu].Input = im;
                                if (iGPU[bestgpu].Warnings == null || iGPU[bestgpu].Warnings.Length == 0)
                                {
                                    SySal.Imaging.Cluster[] clusters = iGPU[bestgpu].Clusters[0];
                                    foreach (SySal.Imaging.Cluster cls in clusters)
                                        if (cls.Area >= S.MinClusterSize && cls.Area <= S.MaxClusterSize)
                                            for (ix = 0; ix < S.XCells; ix++)
                                                if (cls.X >= cellminx[ix] && cls.X <= cellmaxx[ix])
                                                    for (iy = 0; iy < S.YCells; iy++)
                                                        if (cls.Y >= cellminy[iy] && cls.Y <= cellmaxy[iy])
                                                            densities[j][ix, iy]++;
                                    files[j]++;
                                }
                                if (m_Running == false) break;
                            }
                        }
                        catch (Exception) 
                        {
                            if (im != null) ((SySal.Imaging.LinearMemoryImage)im).Dispose();
                        }
                        this.Invoke(new dSetValue(SetValue), new object[] { (double)(i + 1) });
                        if (m_Running == false) break;
                    }
                    for (j = 0; j < files.Length; j++)
                        if (files[j] > 0)
                            for (ix = 0; ix < S.XCells; ix++)
                                for (iy = 0; iy < S.YCells; iy++)
                                    densities[j][ix, iy] /= (files[j] * m_ImageFormat.Width * m_ImageFormat.Height);
                }));
            execthread.Start();
            while (execthread.Join(100) == false) Application.DoEvents();
            System.Collections.ArrayList hidens = new System.Collections.ArrayList();
            foreach (double d in densities[densities.Length - 1]) hidens.Add(d);
            hidens.Sort();
            if (hidens.Count < HiDensValues)
            {
                MessageBox.Show("Too few clusters:\r\nthe threshold was set too high, or the images were not taken in emulsion.\r\nPlease correct and retry.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                m_Running = false;
                EnableButtons();
                return;
            }
            double target = 0.0;
            int t;
            for (t = 1; t <= HiDensValues; t++) target += (double)hidens[hidens.Count - t];
            target /= HiDensValues;
            System.Collections.ArrayList pvals = new System.Collections.ArrayList();
            {
                int ix, iy;
                for (ix = 0; ix < S.XCells; ix++)                
                    for (iy = 0; iy < S.YCells; iy++)
                    {
                        for (t = 1; t < densities.Length && (densities[t - 1][ix, iy] < target || densities[t][ix, iy] > target); t++);
                        if (t < densities.Length)
                        {
                            SySal.Imaging.DCTInterpolationImage.PointValue pval = new SySal.Imaging.DCTInterpolationImage.PointValue();
                            pval.X = (ushort)cellmeanx[ix];
                            pval.Y = (ushort)cellmeanx[iy];
                            pval.Value = (int)((target - densities[t - 1][ix, iy]) / (densities[t][ix, iy] - densities[t - 1][ix, iy]) * 
                                (((ConstThresholdImagePixels)thresholdimgs[t].Pixels).m_Threshold - ((ConstThresholdImagePixels)thresholdimgs[t - 1].Pixels).m_Threshold) + 
                                ((ConstThresholdImagePixels)thresholdimgs[t - 1].Pixels).m_Threshold);
                            pvals.Add(pval);
                        }
                    }
                SySal.Imaging.DCTInterpolationImage dct = null;
                try
                {
                    SySal.Imaging.ImageInfo info = m_ImageFormat;
                    info.BitsPerPixel = 16;
                    dct = new SySal.Imaging.DCTInterpolationImage(info, (int)S.XWaves, (int)S.YWaves,
                        (SySal.Imaging.DCTInterpolationImage.PointValue [])pvals.ToArray(typeof(SySal.Imaging.DCTInterpolationImage.PointValue)));
                    m_Result = txtResult.Text = dct.ToString();
                }
                catch (Exception x)
                {
                    MessageBox.Show("Invalid DCT built.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    m_Running = false;
                    EnableButtons();
                    return;
                }       
            }
            m_Running = false;
            EnableButtons();
            {
                System.IO.StringWriter sw = new System.IO.StringWriter();
                sw.WriteLine("THRESHOLD\tX\tY\tDENSITY");
                int j, ix, iy;
                for (j = 0; j < thresholdimgs.Length; j++)
                    for (ix = 0; ix < S.XCells; ix++)
                        for (iy = 0; iy < S.YCells; iy++)
                            sw.WriteLine(((ConstThresholdImagePixels)thresholdimgs[j].Pixels).m_Threshold + "\t" + cellmeanx[ix] + "\t" + cellmeany[iy] + "\t" + densities[j][ix, iy].ToString(System.Globalization.CultureInfo.InvariantCulture));
                sw.Flush();
                sw.Close();
                InfoPanel panel = new InfoPanel();
                panel.TopLevel = true;
                panel.SetContent("Density vs. Threshold", sw.ToString());
                panel.ShowDialog();
            }
        }

        delegate void dSetValue(double v);

        private void SetValue(double v)
        {
            pbProgress.Value = v;
        }

        void EnableButtons()
        {
            btnViewResult.Enabled = !m_Running;
            btnOK.Enabled = !m_Running;
            btnCancel.Enabled = !m_Running;
            btnSelImages.Enabled = !m_Running;
            btnCompute.Text = m_Running ? "Stop" : "Compute";
        }

        bool m_Running = false;

        class ConstThresholdImagePixels : SySal.Imaging.IImagePixels
        {
            public short m_Threshold;

            public ConstThresholdImagePixels(short threshold) { m_Threshold = threshold; }

            #region IImagePixels Members

            public ushort Channels
            {
                get { return 2; }
            }

            public byte this[ushort x, ushort y, ushort channel]
            {
                get
                {
                    switch (channel)
                    {
                        case 0: return (byte)((m_Threshold & 0xff00) >> 8);
                        case 1: return (byte)(m_Threshold & 0xff);
                        default: throw new Exception("Unknown channel.");
                    }
                }
                set
                {
                    throw new Exception("The method or operation is not implemented.");
                }
            }

            public byte this[uint index]
            {
                get
                {
                    switch (index % 2)
                    {
                        case 0: return (byte)((m_Threshold & 0xff00) >> 8);
                        case 1: return (byte)(m_Threshold & 0xff);
                        default: throw new Exception("Unknown channel.");
                    }
                }
                set
                {
                    throw new Exception("The method or operation is not implemented.");
                }
            }

            #endregion
        }

        private void btnViewResult_Click(object sender, EventArgs e)
        {
            if (m_Result.Length <= 0) return;
            try
            {
                SySal.Imaging.DCTInterpolationImage dct = Scanner.DCTInterpolationImageFromString(m_Result);
                InfoPanel panel = new InfoPanel();
                panel.SetContent("Threshold image", SySalImageFromImage.ThresholdImage(dct));
                panel.TopLevel = true;
                panel.ShowDialog();                
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error creating interpolated image.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        private void btnDefaults_Click(object sender, EventArgs e)
        {
            S = new ThresholdImageComputationSettings();
            ShowSettings();
        }

        private void btnRemember_Click(object sender, EventArgs e)
        {
            try
            {
                SySal.Management.MachineSettings.SetSettings(typeof(ThresholdImageComputationSettings), S);
                MessageBox.Show("Settings saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}

