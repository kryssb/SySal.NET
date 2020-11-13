using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    public partial class VertexFitForm : Form
    {
        internal static System.Collections.ArrayList AvailableBrowsers = new System.Collections.ArrayList();

        public static void CloseAll()
        {
            while (AvailableBrowsers.Count > 0) ((VertexFitForm)AvailableBrowsers[0]).Close();
        }

        internal TrackSelector m_TrackSelector;

        SySal.TotalScan.Flexi.Volume m_V;

        public static VertexFitForm Browse(string name, GDI3D.Control.GDIDisplay gdidisp, TrackSelector updatefits, SySal.TotalScan.Flexi.Volume v)
        {            
            foreach (VertexFitForm b in AvailableBrowsers)
            {
                if (String.Compare(b.FitName, name, true) == 0)
                {
                    b.BringToFront();
                    return b;
                }
            }
            VertexFitForm newb = new VertexFitForm(name, gdidisp, v);            
            newb.Show();
            AvailableBrowsers.Add(newb);
            newb.m_TrackSelector = updatefits;
            return newb;
        }

        public VertexFitForm(string name, GDI3D.Control.GDIDisplay gdidisp, SySal.TotalScan.Flexi.Volume v)
        {
            InitializeComponent();
            m_V = v;
            m_FitName = name;
            m_gdiDisplay = gdidisp;
            m_ParentSlopes.X = 0.0;
            m_ParentSlopes.Y = 0.059;
            txtParentSX.Text = m_ParentSlopes.X.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtParentSY.Text = m_ParentSlopes.Y.ToString(System.Globalization.CultureInfo.InvariantCulture);
            Text = "VertexFit \"" + name + "\"";
        }

        private void OnClose(object sender, System.EventArgs e)
        {
            try
            {
                AvailableBrowsers.Remove(this);
                m_gdiDisplay.RemoveOwned(m_VF);
                m_TrackSelector.RaiseAddFit(this, null);
            }
            catch (Exception) { }
        }

        private SySal.TotalScan.VertexFit m_VF = new SySal.TotalScan.VertexFit();

        public void AddTrackFit(SySal.TotalScan.VertexFit.TrackFit tf)
        {
            try
            {
                m_VF.AddTrackFit(tf);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error adding track fit", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            UpdateFit();
        }

        private string m_FitName;

        internal string FitName { get { return m_FitName; } }

        SySal.BasicTypes.Vector m_TotalP = new SySal.BasicTypes.Vector();

        private void UpdateFit()
        {
            m_gdiDisplay.AutoRender = false;            
            TrackList.BeginUpdate();            
            try
            {
                m_TotalP.X = m_TotalP.Y = m_TotalP.Z = 0.0;
                TrackList.Items.Clear();
                int i;
                for (i = 0; i < m_VF.Count; i++)
                {
                    SySal.BasicTypes.Vector s = new SySal.BasicTypes.Vector();                                        
                    SySal.TotalScan.VertexFit.TrackFit tf = m_VF.Track(i);
                    s.Z = 1.0 / Math.Sqrt(1.0 + tf.Slope.X * tf.Slope.X + tf.Slope.Y * tf.Slope.Y);
                    s.X = tf.Slope.X * s.Z;
                    s.Y = tf.Slope.Y * s.Z;
                    ListViewItem lvi = new ListViewItem(tf.Id.ToString());
                    lvi.Tag = tf.Id;
                    lvi.SubItems.Add(tf.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Intercept.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Weight.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    if (tf is SySal.TotalScan.VertexFit.TrackFitWithMomentum)
                    {
                        double p = ((SySal.TotalScan.VertexFit.TrackFitWithMomentum)tf).P;
                        lvi.SubItems.Add(p.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                        m_TotalP.X += s.X * p;
                        m_TotalP.Y += s.Y * p;
                        m_TotalP.Z += s.Z * p;
                    }
                    else
                        lvi.SubItems.Add("");
                    lvi.SubItems.Add(m_VF.TrackIP(tf).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(m_VF.DisconnectedTrackIP(tf.Id).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(Math.Min(Math.Abs(m_VF.Z - tf.MaxZ),Math.Abs(m_VF.Z - tf.MinZ)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    TrackList.Items.Add(lvi);
                }
                textX.Text = m_VF.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                textY.Text = m_VF.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                textZ.Text = m_VF.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                textX.Text = "";
                textY.Text = "";
                textZ.Text = "";                
                TrackList.Items.Clear();
                int i;
                for (i = 0; i < m_VF.Count; i++)
                {
                    SySal.TotalScan.VertexFit.TrackFit tf = m_VF.Track(i);
                    ListViewItem lvi = new ListViewItem(tf.Id.ToString());
                    lvi.Tag = tf.Id;
                    lvi.SubItems.Add(tf.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Intercept.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    lvi.SubItems.Add(tf.Weight.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    if (tf is SySal.TotalScan.VertexFit.TrackFitWithMomentum)
                        lvi.SubItems.Add(((SySal.TotalScan.VertexFit.TrackFitWithMomentum)tf).P.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    else
                        lvi.SubItems.Add("");
                    lvi.SubItems.Add("");
                    lvi.SubItems.Add("");
                    lvi.SubItems.Add("");
                    TrackList.Items.Add(lvi);
                }                
            }
            txtPx.Text = m_TotalP.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtPy.Text = m_TotalP.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtPz.Text = m_TotalP.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtP.Text = Math.Sqrt(m_TotalP.X * m_TotalP.X + m_TotalP.Y * m_TotalP.Y + m_TotalP.Z * m_TotalP.Z).ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            UpdatePT();
            TrackList.EndUpdate();
            if (m_gdiDisplay.RemoveOwned(m_VF) > 0) cmdAddToPlot_Click(this, null);
            m_gdiDisplay.AutoRender = true;
            m_gdiDisplay.Render();
        }

        private void UpdatePT()
        {
            SySal.BasicTypes.Vector pb = new SySal.BasicTypes.Vector();
            pb.Z = 1.0 / Math.Sqrt(m_ParentSlopes.X * m_ParentSlopes.X + m_ParentSlopes.Y * m_ParentSlopes.Y + 1.0);
            pb.X = pb.Z * m_ParentSlopes.X;
            pb.Y = pb.Z * m_ParentSlopes.Y;
            double n = pb.X * m_TotalP.X + pb.Y * m_TotalP.Y + pb.Z * m_TotalP.Z;
            double n2 = Math.Sqrt(m_TotalP.X * m_TotalP.X + m_TotalP.Y * m_TotalP.Y + m_TotalP.Z * m_TotalP.Z - n * n);
            txtPT.Text = n2.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
        }

        private void cmdRemove_Click(object sender, EventArgs e)
        {
            if (TrackList.SelectedItems.Count != 1) return;
            try
            {
                m_VF.RemoveTrackFit((SySal.TotalScan.Index)(TrackList.SelectedItems[0].Tag));
                //TrackList.Items.Remove(TrackList.SelectedItems[0]);
                UpdateFit();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error removing fit", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void DumpSelButton_Click(object sender, EventArgs e)
        {
            System.Windows.Forms.SaveFileDialog sdlg = new System.Windows.Forms.SaveFileDialog();
            sdlg.Title = "Select file to dump fit \"" + m_FitName + "\"";
            sdlg.FileName = DumpFileText.Text;
            sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            if (sdlg.ShowDialog() == DialogResult.OK) DumpFileText.Text = sdlg.FileName;
        }

        private void DumpFileButton_Click(object sender, EventArgs e)
        {
            System.IO.StreamWriter w = null;
            try
            {
                w = new System.IO.StreamWriter(DumpFileText.Text);
                w.WriteLine(m_FitName + "\t" + m_VF.Count + "\t" + textX.Text + "\t" + textY.Text + "\t" + textZ.Text);
                foreach (ListViewItem lvi in TrackList.Items)
                {
                    int i;
                    w.Write(lvi.SubItems[0].Text);
                    for (i = 1; i < lvi.SubItems.Count; i++)
                        w.Write("\t" + lvi.SubItems[i].Text);
                    w.WriteLine();
                }
                w.Flush();
                w.Close();
            }
            catch (Exception x)
            {
                if (w != null)
                {
                    w.Close();
                    w = null;
                }
                MessageBox.Show(x.ToString(), "Error dumping fit info", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        GDI3D.Control.GDIDisplay m_gdiDisplay;

        private void cmdAddToPlot_Click(object sender, EventArgs e)
        {
            try
            {
                m_gdiDisplay.AutoRender = false;
                int i;
                for (i = 0; i < m_VF.Count; i++)
                {
                    SySal.TotalScan.VertexFit.TrackFit tf = m_VF.Track(i);
                    GDI3D.Control.Line l = new GDI3D.Control.Line(
                        tf.Intercept.X + tf.Slope.X * (tf.MaxZ - tf.Intercept.Z),
                        tf.Intercept.Y + tf.Slope.Y * (tf.MaxZ - tf.Intercept.Z),
                        tf.MaxZ,
                        tf.Intercept.X + tf.Slope.X * (tf.MinZ - tf.Intercept.Z),
                        tf.Intercept.Y + tf.Slope.Y * (tf.MinZ - tf.Intercept.Z),
                        tf.MinZ,
                        m_VF, cmdTagColor.BackColor.R, cmdTagColor.BackColor.G, cmdTagColor.BackColor.B
                        );
                    l.Dashed = true;
                    l.Label = "TF " + tf.Id.ToString();
                    m_gdiDisplay.Add(l);
                }
                GDI3D.Control.Point p = new GDI3D.Control.Point(m_VF.X, m_VF.Y, m_VF.Z, m_VF, cmdTagColor.BackColor.R, cmdTagColor.BackColor.G, cmdTagColor.BackColor.B);
                p.Label = "VF " + m_FitName;
                m_gdiDisplay.Add(p);
            }
            catch (Exception) { };
            m_gdiDisplay.AutoRender = true;
            m_gdiDisplay.Render();
        }

        private void OnHighlightChanged(object sender, EventArgs e)
        {
            m_gdiDisplay.Highlight(m_VF, CheckHighight.Checked);
        }

        private void OnShowLabelChanged(object sender, EventArgs e)
        {
            m_gdiDisplay.EnableLabel(m_VF, CheckShowLabels.Checked);
        }

        private void cmdTagColor_Click(object sender, EventArgs e)
        {
            ColorDialog colorDialog1 = new ColorDialog();
            colorDialog1.Color = cmdTagColor.BackColor;
            if (colorDialog1.ShowDialog() == DialogResult.OK)
            {
                cmdTagColor.BackColor = colorDialog1.Color;
            }
        }

        SySal.BasicTypes.Vector2 m_ParentSlopes = new SySal.BasicTypes.Vector2();

        private void OnParentSXLeave(object sender, EventArgs e)
        {
            try
            {
                m_ParentSlopes.X = System.Convert.ToDouble(txtParentSX.Text, System.Globalization.CultureInfo.InvariantCulture);                
            }
            catch (Exception)
            {
                txtParentSX.Text = m_ParentSlopes.X.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
            UpdatePT();
        }

        private void OnParentSYLeave(object sender, EventArgs e)
        {
            try
            {
                m_ParentSlopes.Y = System.Convert.ToDouble(txtParentSY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtParentSY.Text = m_ParentSlopes.Y.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
            UpdatePT();
        }

        private void cmdToVertex_Click(object sender, EventArgs e)
        {
            if (m_VF.Count <= 0) return;
            SySal.TotalScan.Track [] tklist = new SySal.TotalScan.Track[m_VF.Count];
            int i;
            System.Collections.ArrayList vtxaltered = new System.Collections.ArrayList();
            for (i = 0; i < tklist.Length; i++)
            {
                SySal.TotalScan.VertexFit.TrackFit tf = m_VF.Track(i);
                tklist[i] = m_V.Tracks[((SySal.TotalScan.BaseTrackIndex)tf.Id).Id];
                tklist[i].SetAttribute(SySal.TotalScan.Vertex.TrackWeightAttribute, tf.Weight);
                bool isupstream = (tklist[i].Downstream_Z + tklist[i].Upstream_Z) > (tf.MaxZ + tf.MinZ);
                SySal.TotalScan.Vertex vtxa = isupstream ? tklist[i].Upstream_Vertex : tklist[i].Downstream_Vertex;
                if (vtxa != null)
                {
                    if (vtxaltered.Contains(vtxa) == false) vtxaltered.Add(vtxa);
                    vtxa.RemoveTrack(tklist[i]);
                }
            }
            int[] ids = new int[vtxaltered.Count];
            for (i = 0; i < ids.Length; i++)
                ids[i] = ((SySal.TotalScan.Vertex)vtxaltered[i]).Id;
            System.Collections.ArrayList vtxremove = new System.Collections.ArrayList();
            foreach (SySal.TotalScan.Vertex vtx in vtxaltered)
                try
                {
                    vtx.NotifyChanged();
                    if (vtx.AverageDistance >= 0.0) continue;
                }
                catch (Exception)
                {
                    vtxremove.Add(vtx.Id);
                }
            
            ((SySal.TotalScan.Flexi.Volume.VertexList)m_V.Vertices).Remove((int[])vtxremove.ToArray(typeof(int)));
            SySal.TotalScan.Flexi.Vertex nv = new SySal.TotalScan.Flexi.Vertex(((SySal.TotalScan.Flexi.Track)tklist[0]).DataSet, m_V.Vertices.Length);
            nv.SetId(m_V.Vertices.Length);
            for (i = 0; i < tklist.Length; i++)
            {
                SySal.TotalScan.VertexFit.TrackFit tf = m_VF.Track(i);
                if ((tklist[i].Downstream_Z + tklist[i].Upstream_Z) > (tf.MaxZ + tf.MinZ))
                {
                    nv.AddTrack(tklist[i], false);
                    tklist[i].SetUpstreamVertex(nv);
                }
                else
                {
                    nv.AddTrack(tklist[i], true);
                    tklist[i].SetDownstreamVertex(nv);
                }
            }
            try
            {
                nv.NotifyChanged();
                if (nv.AverageDistance >= 0.0)
                    ((SySal.TotalScan.Flexi.Volume.VertexList)m_V.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { nv });
            }
            catch (Exception x)
            {
                MessageBox.Show("Can't create new vertex - check geometry/topology.", "Vertex error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                nv.SetId(-1);
            }
            if (nv.Id >= 0)
            {
                string s = "";
                if (ids.Length > 0)
                {
                    s += "\r\nVertices altered: {";
                    for (i = 0; i < ids.Length; i++)
                        if (i == 0) s += ids[i].ToString();
                        else s += ", " + ids[i].ToString();
                    s += "}";
                }
                if (vtxremove.Count > 0)
                {
                    s += "\r\nVertices removed: {";
                    for (i = 0; i < vtxremove.Count; i++)
                        if (i == 0) s += vtxremove[i].ToString();
                        else s += ", " + vtxremove[i].ToString();
                    s += "}";
                }
                MessageBox.Show("New vertex " + nv.Id + " created." + s + "\r\nPlease regenerate the plot to see the changes.", "Vertex created", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            TrackBrowser.RefreshAll();
            VertexBrowser.RefreshAll();
        }
    }
}