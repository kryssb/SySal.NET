using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.ScanbackViewer
{
    internal partial class MomentumForm : Form
    {
        public MomentumForm()
        {
            InitializeComponent();            
        }

        public SySal.TotalScan.IMCSMomentumEstimator MCS;

        public SySal.Processing.MCSLikelihood.MomentumEstimator MCSLikelihood = new SySal.Processing.MCSLikelihood.MomentumEstimator();

        public SySal.Processing.MCSAnnecy.MomentumEstimator MCSAnnecy = new SySal.Processing.MCSAnnecy.MomentumEstimator();

        public SySal.Tracking.MIPEmulsionTrackInfo[] Measurements;

        private void OnLoad(object sender, EventArgs e)
        {
            MCS = MCSLikelihood;
            rdAlgLikelihood.Checked = true;
            textResult.Text = "";
            textIgnoreDeltaSlope.Text = m_IgnoreDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            textMeasIgnoreGrains.Text = m_IgnoreGrains.ToString();
            lvSlopes.BeginUpdate();
            lvSlopes.Items.Clear();
            try
            {
                foreach (SySal.Tracking.MIPEmulsionTrackInfo info in Measurements)
                {
                    ListViewItem lvi = new ListViewItem(info.Field.ToString());
                    lvi.SubItems.Add(info.Intercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(info.Count.ToString());
                    lvi.SubItems.Add(info.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(info.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.Checked = true;
                    lvi.Tag = info;
                    lvSlopes.Items.Add(lvi);
                }
            }
            catch (Exception x)
            {
                MessageBox.Show("Error inserting measurement " + lvSlopes.Items.Count + "\r\n" + x.ToString(), "Debug Info");
            }
            lvSlopes.EndUpdate();
        }

        private void OnGrainsLeave(object sender, EventArgs e)
        {
            try
            {
                m_IgnoreGrains = System.Convert.ToInt32(textMeasIgnoreGrains.Text);
            }
            catch (Exception)
            {
                textMeasIgnoreGrains.Text = m_IgnoreGrains.ToString();
            }
        }

        private void OnDeltaSlopeLeave(object sender, EventArgs e)
        {
            try
            {
                m_IgnoreDeltaSlope = System.Convert.ToDouble(textIgnoreDeltaSlope.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                textIgnoreDeltaSlope.Text = m_IgnoreDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        int m_IgnoreGrains = 20;

        double m_IgnoreDeltaSlope = 0.01;

        private void buttonConfigFileSel_Click(object sender, EventArgs e)
        {
            sfn.FileName = textConfigFile.Text;
            sfn.OverwritePrompt = false;
            if (sfn.ShowDialog() == DialogResult.OK)
            {
                textConfigFile.Text = sfn.FileName;
            }
        }

        static System.Xml.Serialization.XmlSerializer xmls_likelihood = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MCSLikelihood.Configuration));

        static System.Xml.Serialization.XmlSerializer xmls_annecy = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MCSAnnecy.Configuration));

        private void buttonLoadConfig_Click(object sender, EventArgs e)
        {
            System.IO.StreamReader r = null;
            try
            {
                r = new System.IO.StreamReader(textConfigFile.Text);               
                if (MCS == MCSLikelihood) MCSLikelihood.Config = (SySal.Processing.MCSLikelihood.Configuration)xmls_likelihood.Deserialize(r);
                else MCSAnnecy.Config = (SySal.Processing.MCSAnnecy.Configuration)xmls_annecy.Deserialize(r);
            }
            catch (Exception x)
            {
                MessageBox.Show("Can't load file \"" + textConfigFile.Text + "\"!", "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (r != null) r.Close();
            }
        }

        private void buttonSaveConfig_Click(object sender, EventArgs e)
        {
            System.IO.StreamWriter w = null;
            try
            {
                w = new System.IO.StreamWriter(textConfigFile.Text);
                if (MCS == MCSLikelihood) xmls_likelihood.Serialize(w, MCSLikelihood.Config);
                else xmls_annecy.Serialize(w, MCSAnnecy.Config);
            }
            catch (Exception x)
            {
                MessageBox.Show("Can't save file \"" + textConfigFile.Text + "\"!", "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (w != null)
                {
                    w.Flush();
                    w.Close();
                }
            }
        }

        private void buttonEditConfig_Click(object sender, EventArgs e)
        {
            if (MCS == MCSLikelihood)
            {
                SySal.Management.Configuration c = MCSLikelihood.Config;
                if (MCSLikelihood.EditConfiguration(ref c))
                    MCSLikelihood.Config = c;
            }
            else
            {
                SySal.Management.Configuration c = MCSAnnecy.Config;
                if (MCSAnnecy.EditConfiguration(ref c))
                    MCSAnnecy.Config = c;
            }
        }

        private void buttonCompute_Click(object sender, EventArgs e)
        {
            try
            {
                int good = 0;
                foreach (ListViewItem lvi in lvSlopes.Items)
                    if (lvi.Checked)
                        good++;
                SySal.Tracking.MIPEmulsionTrackInfo[] meas = new SySal.Tracking.MIPEmulsionTrackInfo[good];
                good = 0;
                foreach (ListViewItem lvi in lvSlopes.Items)
                    if (lvi.Checked)
                        meas[good++] = (SySal.Tracking.MIPEmulsionTrackInfo)lvi.Tag;                        
                SySal.TotalScan.MomentumResult res = MCS.ProcessData(meas);
                textResult.Text = res.Value.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " ; " +
                    res.LowerBound.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " ; " +
                    res.UpperBound.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " ; " +
                    (res.ConfidenceLevel * 100.0).ToString("F0", System.Globalization.CultureInfo.InvariantCulture) + "%";
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Computation error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                textResult.Text = "";
            }
        }

        private void buttonMeasSelAll_Click(object sender, EventArgs e)
        {
            lvSlopes.BeginUpdate();
            foreach (ListViewItem lvi in lvSlopes.Items)
                lvi.Checked = true;
            lvSlopes.EndUpdate();
        }

        private void buttonMeasIgnoreGrains_Click(object sender, EventArgs e)
        {
            lvSlopes.BeginUpdate();
            foreach (ListViewItem lvi in lvSlopes.Items)
                if (((SySal.Tracking.MIPEmulsionTrackInfo)(lvi.Tag)).Count < m_IgnoreGrains)
                    lvi.Checked = false;
            lvSlopes.EndUpdate();
        }

        private void buttonIgnoreDeltaSlope_Click(object sender, EventArgs e)
        {
            lvSlopes.BeginUpdate();
            int i;
            double dsx, dsy;
            for (i = 0; i < Measurements.Length; i++)
            {
                ListViewItem lvi = lvSlopes.Items[i];
                if (i == 0 && Measurements.Length >= 2)
                {
                    dsx = Measurements[0].Slope.X - Measurements[1].Slope.X;
                    dsy = Measurements[0].Slope.Y - Measurements[1].Slope.Y;
                    if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                        lvi.Checked = false;
                }
                else if (i == Measurements.Length - 1 && Measurements.Length >= 2)
                {
                    dsx = Measurements[i].Slope.X - Measurements[i - 1].Slope.X;
                    dsy = Measurements[i].Slope.Y - Measurements[i - 1].Slope.Y;
                    if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                        lvi.Checked = false;
                }
                else if (Measurements.Length >= 3)
                {
                    dsx = Measurements[i].Slope.X - Measurements[i - 1].Slope.X;
                    dsy = Measurements[i].Slope.Y - Measurements[i - 1].Slope.Y;
                    if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                    {
                        dsx = Measurements[i].Slope.X - Measurements[i + 1].Slope.X;
                        dsy = Measurements[i].Slope.Y - Measurements[i + 1].Slope.Y;
                        if (dsx * dsx + dsy * dsy >= m_IgnoreDeltaSlope)
                            lvi.Checked = false;
                    }
                }
            }
            lvSlopes.EndUpdate();
        }

        private void rdAlgLikelihood_CheckedChanged(object sender, EventArgs e)
        {
            MCS = MCSLikelihood;
        }

        private void rdAlgAnnecy_CheckedChanged(object sender, EventArgs e)
        {
            MCS = MCSAnnecy;
        }
    }
}