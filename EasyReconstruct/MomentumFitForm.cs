using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    public partial class MomentumFitForm : Form
    {
        public static SySal.Processing.MCSLikelihood.MomentumEstimator MCSLikelihood;

        internal static System.Collections.ArrayList AvailableBrowsers = new System.Collections.ArrayList();

        private static event dGenericEvent OnAvailableBrowsersChanged;

        static internal void SubscribeOnUpdateBrowsers(dGenericEvent ge)
        {
            OnAvailableBrowsersChanged += ge;
        }

        static internal void UnsubscribeOnUpdateBrowsers(dGenericEvent ge)
        {
            OnAvailableBrowsersChanged -= ge;
        }

        static internal void RaiseOnUpdateBrowsers(object sender, EventArgs e)
        {
            if (OnAvailableBrowsersChanged != null) OnAvailableBrowsersChanged(sender, e);
        }

        internal MomentumFitForm(string name)
        {
            InitializeComponent();
            pbLkDisplay.Image = new Bitmap(pbLkDisplay.Width, pbLkDisplay.Height);
            m_FitName = name;
        }

        private string m_FitName;

        internal string FitName { get { return m_FitName; } }

        internal NumericalTools.Likelihood Likelihood;

        private void btnCompute_Click(object sender, EventArgs e)
        {
            if (clSlopeSets.CheckedItems.Count < 1)
            {
                MessageBox.Show("At least one data set must be enabled.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            NumericalTools.Likelihood[] lk = new NumericalTools.Likelihood[clSlopeSets.CheckedItems.Count];
            int i;
            for (i = 0; i < clSlopeSets.CheckedItems.Count; i++)
            {
                FitSet fs = (FitSet)clSlopeSets.CheckedItems[i];
                if (fs.Likelihood == null)
                    try
                    {
                        MCSLikelihood.ProcessData(fs.Tracks, out fs.Likelihood);
                    }
                    catch (Exception x)
                    {
                        MessageBox.Show("Error computing likelihood for set \"" + fs.ToString() + "\":\r\n" + x.Message, "Computation Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                lk[i] = fs.Likelihood;
            }
            Likelihood = null;
            txtResults.Text = "";
            try
            {
                Likelihood = new NumericalTools.OneParamLogLikelihood(0.05, lk);
                string t = Likelihood.Best(0).ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + ";";
                double cl = ((SySal.Processing.MCSLikelihood.Configuration)MCSLikelihood.Config).ConfidenceLevel;
                double[] bounds = Likelihood.ConfidenceRegions(0, cl);                
                for (i = 0; i < bounds.Length; i += 2)
                    t += " [" + bounds[i].ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "," + bounds[i + 1].ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "]";
                t += "; " + cl.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                txtResults.Text = t;
            }
            catch (Exception x)
            {
                MessageBox.Show("Error computing combined likelihood:\r\n" + x.Message, "Computation Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            PlotLikelihood("LogL (combined)", Likelihood);
        }

        private void OnClose(object sender, FormClosedEventArgs e)
        {
            AvailableBrowsers.Remove(this);
            RaiseOnUpdateBrowsers(this, null);
        }

        private void OnLoad(object sender, EventArgs e)
        {
            this.Text = "Momentum fit \"" + m_FitName + "\"";
            AvailableBrowsers.Add(this);
            RaiseOnUpdateBrowsers(this, null);
        }

        internal void Add(FitSet fs)
        {
            try
            {
                if (fs.Tracks.Length < 2) throw new Exception("At least two slopes must be measured.");
                if (clSlopeSets.Items.Contains(fs)) throw new Exception("Slope set is already contained and will not be added again.");
                foreach (object o in clSlopeSets.Items)
                {
                    FitSet fso = (FitSet)o;
                    if (fso.Source == fs.Source) throw new Exception("Slope set is already contained and will not be added again.");
                    int i, j;
                    i = j = 0;
                    while (i < fso.Tracks.Length && j < fs.Tracks.Length)
                        if (fso.Tracks[i].Field < fs.Tracks[j].Field) i++;
                        else if (fso.Tracks[i].Field > fs.Tracks[j].Field) j++;
                        else throw new Exception("The new data set has at least one plate in common with \"" + fso.ToString() + "\".\r\nCombined fit requires that no data be in common.\r\nDuplicate found on layer " + 
                            fso.Tracks[i].Field + " Grains " + fso.Tracks[i].Count + " SX/Y " + 
                            fso.Tracks[i].Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "/" + 
                            fso.Tracks[i].Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));                    
                }
                clSlopeSets.SetItemChecked(clSlopeSets.Items.Add(fs), true);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Data Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        internal class FitSet
        {
            public SySal.Tracking.MIPEmulsionTrackInfo[] Tracks;
            public TrackBrowser Source;           
            public NumericalTools.Likelihood Likelihood;
            public override string ToString()
            {
                SySal.TotalScan.Flexi.DataSet ds = ((SySal.TotalScan.Flexi.Track)Source.Track).DataSet;
                return "Tk " + Source.Track.Id + " " + ds.DataType + " " + ds.DataId;
            }
            public FitSet(SySal.Tracking.MIPEmulsionTrackInfo[] tks, TrackBrowser src)
            {
                Tracks = tks;
                Source = src;               
                Likelihood = null;
            }
        }

        private void OnSelectedSlopeSetChanged(object sender, EventArgs e)
        {
            if (clSlopeSets.SelectedIndex >= 0)
            {
                FitSet fs = (FitSet)clSlopeSets.SelectedItem;
                PlotLikelihood("LogL \"" + fs.ToString() + "\"", fs.Likelihood);
            }
        }

        private void PlotLikelihood(string text, NumericalTools.Likelihood lk)
        {
            if (lk == null)
            {
                Image im = pbLkDisplay.Image;
                Graphics g = Graphics.FromImage(im);
                g.FillRectangle(new SolidBrush(Color.White), 0, 0, im.Width, im.Height);
                g.DrawString("Likelihood not available - press Compute", new Font("Lucida", 10), new SolidBrush(Color.Black), 0, 0);
                pbLkDisplay.Refresh();
            }
            else
            {
                Image im = pbLkDisplay.Image;
                Graphics g = Graphics.FromImage(im);
                g.FillRectangle(new SolidBrush(Color.White), 0, 0, im.Width, im.Height);
                double[] p = new double[(int)Math.Floor((lk.MaxBound(0) - lk.MinBound(0)) / 0.05 + 1.0)];
                double[] logl = new double[p.Length];
                int i;                                
                for (i = 0; i < p.Length; i++)
                {
                    p[i] = lk.MinBound(0) + i * 0.05;
                    logl[i] = lk.LogValue(p[i]);
                }
                NumericalTools.Plot pl = new NumericalTools.Plot();
                pl.VecX = p;
                pl.VecY = logl;
                double dummy = 0.0;                
                double miny = 0.0, maxy = 0.0;
                NumericalTools.Fitting.FindStatistics(logl, ref maxy, ref miny, ref dummy, ref dummy);
                maxy += 0.2 * (maxy - miny);
                pl.JoinPenThickness = 2;
                pl.MinX = lk.MinBound(0) - 0.05;
                pl.MaxX = lk.MaxBound(0) + 0.05;
                pl.SetXDefaultLimits = false;
                pl.MinY = miny;
                pl.MaxY = maxy;
                pl.SetYDefaultLimits = false;
                pl.XTitle = "P";
                pl.YTitle = text;
                pl.LabelFont = new Font("Lucida", 10);
                pl.PanelFont = new Font("Lucida", 10);
                pl.PanelY = 3;
                pl.Scatter(g, im.Width, im.Height);
                pbLkDisplay.Refresh();
            }
        }

        private void btnRemove_Click(object sender, EventArgs e)
        {
            if (clSlopeSets.SelectedIndex >= 0)
                clSlopeSets.Items.RemoveAt(clSlopeSets.SelectedIndex);
        }

        private void btnDumpLikelihood_Click(object sender, EventArgs e)
        {
            if (Likelihood == null) return;
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select file to save likelihood function points.";
            sdlg.Filter = "Text files(*.txt)|*.txt|All files (*.*)|*.*";
            System.IO.StreamWriter w = null;
            if (sdlg.ShowDialog() == DialogResult.OK)
                try
                {
                    w = new System.IO.StreamWriter(sdlg.FileName);
                    w.Write("P\tLogL");
                    double p;
                    for (p = Likelihood.MinBound(0); p <= Likelihood.MaxBound(0); p += 0.05)
                        w.Write("\r\n" + p.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" + Likelihood.LogValue(p).ToString(System.Globalization.CultureInfo.InvariantCulture));
                    w.Flush();
                    w.Close();
                }
                catch (Exception x)
                {
                    MessageBox.Show("Can't write file \"" + sdlg.FileName + "\".", "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    if (w != null)
                    {
                        w.Close();
                        w = null;
                    }
                }
            if (w != null)
            {
                w.Close();
                w = null;
            }            
        }

        private void btnExport_Click(object sender, EventArgs e)
        {
            if (Likelihood == null) 
            {
                MessageBox.Show("No momentum available.", "Missing fit", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            double cl = ((SySal.Processing.MCSLikelihood.Configuration)MCSLikelihood.Config).ConfidenceLevel;
            double[] bounds = Likelihood.ConfidenceRegions(0, cl);
            foreach (FitSet fs in clSlopeSets.Items)            
                fs.Source.SetMomentum(Likelihood, bounds[0], bounds[bounds.Length - 1], cl);
        }
    }
}