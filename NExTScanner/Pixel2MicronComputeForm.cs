using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class Pixel2MicronComputeForm : SySal.SySalNExTControls.SySalDialog
    {
        public class Pixel2MicronComputationSettings : SySal.Management.Configuration
        {
            public uint MinArea = 4;
            public uint MaxArea = 36;
            public uint MinMatches = 100;
            public double PosTolerance = 2.0;
            public double MinConv = 0.1;
            public double MaxConv = 1.0;

            public Pixel2MicronComputationSettings()
                : base("")
            {
            }

            public override object Clone()
            {
                Pixel2MicronComputationSettings s = new Pixel2MicronComputationSettings();
                s.Name = Name;
                s.MinArea = MinArea;
                s.MaxArea = MaxArea;
                s.MinConv = MinConv;
                s.MaxConv = MaxConv;
                s.MinMatches = MinMatches;
                s.PosTolerance = PosTolerance;
                return s;
            }

            internal static Pixel2MicronComputationSettings Stored
            {
                get
                {
                    Pixel2MicronComputationSettings c = SySal.Management.MachineSettings.GetSettings(typeof(Pixel2MicronComputationSettings)) as Pixel2MicronComputationSettings;
                    if (c == null) c = new Pixel2MicronComputationSettings();
                    return c;
                }
            }
        }

        private Pixel2MicronComputationSettings S = Pixel2MicronComputationSettings.Stored;

        public SySal.Executables.NExTScanner.ISySalLog iLog;

        public Pixel2MicronComputeForm()
        {
            InitializeComponent();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        private void ShowSettings()
        {
            txtMinArea.Text = S.MinArea.ToString();
            txtMaxArea.Text = S.MaxArea.ToString();
            txtPosTolerance.Text = S.PosTolerance.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMinConv.Text = S.MinConv.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMaxConv.Text = S.MaxConv.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMinMatches.Text = S.MinMatches.ToString();
        }

        private void OnMinAreaLeave(object sender, EventArgs e)
        {
            try
            {
                S.MinArea = uint.Parse(txtMinArea.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }                
        }

        private void OnMaxAreaLeave(object sender, EventArgs e)
        {
            try
            {
                S.MaxArea = uint.Parse(txtMaxArea.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }                
        }

        private void OnPosTolLeave(object sender, EventArgs e)
        {
            try
            {
                double c = double.Parse(txtPosTolerance.Text);
                if (c <= 0.0) throw new Exception();
                S.PosTolerance = c;
            }
            catch (Exception)
            {
                ShowSettings();
            }                
        }

        private void OnMinConvLeave(object sender, EventArgs e)
        {
            try
            {
                double c = double.Parse(txtMinConv.Text);
                if (c <= 0.0) throw new Exception();
                S.MinConv = c;
            }
            catch (Exception)
            {
                ShowSettings();
            }                
        }

        private void OnMaxConvLeave(object sender, EventArgs e)
        {
            try
            {
                double c = double.Parse(txtMaxConv.Text);
                if (c <= 0.0) throw new Exception();
                S.MaxConv = c;
            }
            catch (Exception)
            {
                ShowSettings();
            }                
        }

        private void OnMinMatchesLeave(object sender, EventArgs e)
        {
            try
            {
                S.MinMatches = uint.Parse(txtMinMatches.Text);
            }
            catch (Exception)
            {
                ShowSettings();
            }                
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        public string[] SummaryFiles;

        bool m_Stop = false;

        bool m_Running = false;
        private void EnableButtons(object btn)
        {
            btnCancel.Enabled = !m_Running;
            btnOK.Enabled = !m_Running;
            if (m_Running == false)
            {
                btnComputeX.Enabled = true;
                btnComputeX.Text = "Compute X";
                btnComputeY.Enabled = true;
                btnComputeY.Text = "Compute Y";
                btnComputeBoth.Enabled = true;
                btnComputeBoth.Text = "Compute Both";
            }
            else
            {
                if (btnComputeX == btn)
                {
                    btnComputeX.Text = "Stop";
                    btnComputeX.Enabled = true;
                }
                else btnComputeX.Enabled = false;
                if (btnComputeY == btn)
                {
                    btnComputeY.Text = "Stop";
                    btnComputeY.Enabled = true;
                }
                else btnComputeY.Enabled = false;
                if (btnComputeBoth == btn)
                {
                    btnComputeBoth.Text = "Stop";
                    btnComputeBoth.Enabled = true;
                }
                else btnComputeBoth.Enabled = false;
            }
        }

        private delegate void dEnableButtons(object btn);

        private SySal.Imaging.Cluster[] FilterClusters(SySal.Imaging.Cluster[] clist)
        {
            System.Collections.ArrayList tarr = new System.Collections.ArrayList();
            foreach (SySal.Imaging.Cluster c in clist)
                if (c.Area >= S.MinArea && c.Area <= S.MaxArea)
                    tarr.Add(c);
            return (SySal.Imaging.Cluster[])tarr.ToArray(typeof(SySal.Imaging.Cluster));
        }

        public double XConv = 1.0;

        public double YConv = 1.0;

        private void Compute(bool comp_x, bool comp_y, object btn)
        {
            m_Running = true;
            EnableButtons(btn);
            m_Stop = false;
            pbProgress.Value = 0.0;
            System.Threading.Thread thread = new System.Threading.Thread(new System.Threading.ThreadStart(delegate()
                {
                    try
                    {
                        SySal.Processing.QuickMapping.QuickMapper QM = new SySal.Processing.QuickMapping.QuickMapper();
                        SySal.Processing.QuickMapping.Configuration qmc = (SySal.Processing.QuickMapping.Configuration)QM.Config;
                        qmc.FullStatistics = false;
                        qmc.UseAbsoluteReference = true;
                        qmc.PosTol = S.PosTolerance;
                        qmc.SlopeTol = 1.0;
                        QM.Config = qmc;
                        System.Collections.ArrayList xconv = new System.Collections.ArrayList();
                        System.Collections.ArrayList yconv = new System.Collections.ArrayList();
                        int sfi;
                        for (sfi = 0; sfi < SummaryFiles.Length; sfi++)
                        {
                            QuasiStaticAcquisition Q = new QuasiStaticAcquisition(SummaryFiles[sfi]);
                            foreach (QuasiStaticAcquisition.Sequence seq in Q.Sequences)
                            {
                                int ly;
                                SySal.Imaging.Cluster[] prevC = FilterClusters(seq.Layers[0].ReadClusters());
                                SySal.Imaging.Cluster[] nextC = null;
                                SySal.BasicTypes.Vector prevP = seq.Layers[0].Position;
                                SySal.BasicTypes.Vector nextP = new SySal.BasicTypes.Vector();
                                for (ly = 1; ly < seq.Layers.Length; ly++)
                                {
                                    if (m_Stop) throw new Exception("Stopped.");
                                    nextC = FilterClusters(seq.Layers[ly].ReadClusters());
                                    nextP = seq.Layers[ly].Position;
                                    this.Invoke(new dSetValue(SetValue), 100.0 * (((double)seq.Id + (double)(ly - 1) / (double)(seq.Layers.Length - 1)) / (double)(Q.Sequences.Length) + sfi) / SummaryFiles.Length);
                                    SySal.BasicTypes.Vector2 dp = new SySal.BasicTypes.Vector2(seq.Layers[ly].Position - seq.Layers[ly - 1].Position);
                                    SySal.BasicTypes.Vector2 dp1 = dp; dp1.X /= S.MinConv; dp1.Y /= S.MinConv;
                                    SySal.BasicTypes.Vector2 dp2 = dp; dp2.X /= S.MaxConv; dp2.Y /= S.MaxConv;
                                    SySal.BasicTypes.Vector2 da = new SySal.BasicTypes.Vector2(); da.X = 0.5 * (dp1.X + dp2.X); da.Y = 0.5 * (dp1.Y + dp2.Y);
                                    SySal.BasicTypes.Vector2 dext = new SySal.BasicTypes.Vector2(); dext.X = Math.Abs(dp1.X - dp2.X); dext.Y = Math.Abs(dp1.Y - dp2.Y);
                                    SySal.Tracking.MIPEmulsionTrackInfo[] prevmap = new SySal.Tracking.MIPEmulsionTrackInfo[prevC.Length];
                                    int i;
                                    for (i = 0; i < prevC.Length; i++)
                                    {
                                        SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                                        info.Intercept.X = prevC[i].X;
                                        info.Intercept.Y = prevC[i].Y;
                                        prevmap[i] = info;
                                    }
                                    SySal.Tracking.MIPEmulsionTrackInfo[] nextmap = new SySal.Tracking.MIPEmulsionTrackInfo[nextC.Length];
                                    for (i = 0; i < nextC.Length; i++) nextmap[i] = new SySal.Tracking.MIPEmulsionTrackInfo();
                                    double[,] convopt = new double[,] { { 1.0, 1.0 }, { -1.0, 1.0 }, { -1.0, -1.0 }, { 1.0, -1.0 } };
                                    int o;
                                    SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] bestpairs = new SySal.Scanning.PostProcessing.PatternMatching.TrackPair[0];
                                    SySal.BasicTypes.Vector2 bestda = new SySal.BasicTypes.Vector2();
                                    for (o = 0; o < convopt.GetLength(0); o++)
                                        try
                                        {
                                            for (i = 0; i < nextC.Length; i++)
                                            {
                                                SySal.Tracking.MIPEmulsionTrackInfo info = nextmap[i];
                                                info.Intercept.X = nextC[i].X + da.X * convopt[o, 0];
                                                info.Intercept.Y = nextC[i].Y + da.Y * convopt[o, 1];
                                                nextmap[i] = info;
                                            }
                                            SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] prs = QM.Match(prevmap, nextmap, 0.0, dext.X, dext.Y);
                                            if (prs.Length > bestpairs.Length)
                                            {
                                                bestda.X = da.X * convopt[o, 0];
                                                bestda.Y = da.Y * convopt[o, 1];
                                                bestpairs = prs;
                                            }
                                        }
                                        catch (Exception xc) { }

                                    if (bestpairs.Length >= S.MinMatches)
                                    {
                                        double[] deltas = new double[bestpairs.Length];
                                        for (i = 0; i < bestpairs.Length; i++)
                                            deltas[i] = bestpairs[i].First.Info.Intercept.X - nextC[bestpairs[i].Second.Index].X;
                                        bestda.X = NumericalTools.Fitting.Quantiles(deltas, new double[] { 0.5 })[0];
                                        if (bestda.X != 0.0)
                                        {
                                            double v = dp.X / bestda.X;
                                            int pos = xconv.BinarySearch(v);
                                            if (pos < 0) pos = ~pos;
                                            xconv.Insert(pos, v);
                                        }
                                        for (i = 0; i < bestpairs.Length; i++)
                                            deltas[i] = bestpairs[i].First.Info.Intercept.Y - nextC[bestpairs[i].Second.Index].Y;
                                        bestda.Y = NumericalTools.Fitting.Quantiles(deltas, new double[] { 0.5 })[0];
                                        if (bestda.Y != 0.0)
                                        {
                                            double v = dp.Y / bestda.Y;
                                            int pos = yconv.BinarySearch(v);
                                            if (pos < 0) pos = ~pos;
                                            yconv.Insert(pos, v);
                                        }
                                        if (comp_x && xconv.Count > 0)
                                        {
                                            int bmin, bmax;
                                            bmin = (int)Math.Ceiling(xconv.Count * 0.16);
                                            bmax = (int)Math.Floor(xconv.Count * 0.84);
                                            if (bmax < bmin) bmin = bmax;
                                            double[] sample1s = (double[])xconv.GetRange(bmin, bmax - bmin + 1).ToArray(typeof(double));
                                            XConv = NumericalTools.Fitting.Average(sample1s);
                                            this.Invoke(new dSetConv(SetConv), new object[] { true, XConv, 0.5 * (sample1s[sample1s.Length - 1] - sample1s[0]) / Math.Sqrt(sample1s.Length) });
                                        }
                                        if (comp_y && yconv.Count > 0)
                                        {
                                            int bmin, bmax;
                                            bmin = (int)Math.Ceiling(yconv.Count * 0.16);
                                            bmax = (int)Math.Floor(yconv.Count * 0.84);
                                            if (bmax < bmin) bmin = bmax;
                                            double[] sample1s = (double[])yconv.GetRange(bmin, bmax - bmin + 1).ToArray(typeof(double));
                                            YConv = NumericalTools.Fitting.Average(sample1s);
                                            this.Invoke(new dSetConv(SetConv), new object[] { false, YConv, 0.5 * (sample1s[sample1s.Length - 1] - sample1s[0]) / Math.Sqrt(sample1s.Length) });
                                        }
                                    }
                                    prevP = nextP;
                                    prevC = nextC;
                                }
                            }
                        }
                    }
                    catch (Exception xc1)
                    {
                        iLog.Log("Compute", xc1.ToString());
                    }
                    m_Running = false;
                    this.Invoke(new dEnableButtons(EnableButtons), btn);
                }));
            thread.Start();
        }

        delegate void dSetConv(bool isx, double v, double bounds);

        private void SetConv(bool isx, double v, double bounds)
        {
            if (isx)
            {
                txtPix2MiX.Text = v.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
                txtPixMiXErr.Text = bounds.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
            }
            else
            {
                txtPix2MiY.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtPixMiYErr.Text = bounds.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        delegate void dSetValue(double v);

        private void SetValue(double v)
        {
            pbProgress.Value = v;
        }

        private void btnComputeX_Click(object sender, EventArgs e)
        {
            if (m_Running) m_Stop = true;
            else Compute(true, false, btnComputeX);
        }

        private void btnComputeY_Click(object sender, EventArgs e)
        {
            if (m_Running) m_Stop = true;
            else Compute(false, true, btnComputeY);
        }

        private void btnComputeBoth_Click(object sender, EventArgs e)
        {
            if (m_Running) m_Stop = true;
            else Compute(true, true, btnComputeBoth);
        }

        private void btnDefaults_Click(object sender, EventArgs e)
        {
            S = new Pixel2MicronComputationSettings();
            ShowSettings();
        }

        private void btnRemember_Click(object sender, EventArgs e)
        {
            try
            {
                SySal.Management.MachineSettings.SetSettings(typeof(Pixel2MicronComputationSettings), S);
                MessageBox.Show("Settings saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void OnLoad(object sender, EventArgs e)
        {
            ShowSettings();
        }
    }
}

