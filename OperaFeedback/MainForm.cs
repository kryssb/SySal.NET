using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.OperaFeedback
{
    /// <summary>
    /// OperaFeedback interface.
    /// It can be used in two ways:
    /// 1) to insert new feedback into a scanning DB;
    /// 2) to download feedback information from a scanning DB.
    /// <para>
    /// In the first case, OperaFeedback should be used a follows the steps below:
    /// <list type="number">
    /// <item><description>Load a TSR file.</description></item>
    /// <item><description>Type the id of the vertices that are relevant to vertex location feedback, then click "Add Vertex" for each of them.</description></item>
    /// <item><description>Set the attributes for the vertices, by selecting each vertex, modifying the flags, and then clicking on "Set attributes".</description></item>
    /// <item><description>Set the attributes for the tracks, by selecting each track, modifying the flags, and then clicking on "Set attributes".</description></item>
    /// <item><description>If there are one-prong vertices, you need to add the related track by clicking on the "Add Track" button, with a meaningful id, and keeping the "1-prong" checkbox checked.</description></item>
    /// <item><description>Add vertices detected manually.</description></item>
    /// <item><description>Add tracks detected manually. Each track must either be connected to an existing vertex, or it is the only outgoing track of a single-prong vertex; in the latter case, check "1-prong".</description></item>
    /// <item><description>Tracks/vertices can be selected also by clicking on the display. Apply any correction you need.</description></item>
    /// <item><description>Select a machine to record data to the DB.</description></item>
    /// <item><description>Select a program setting configuration to record data to the DB.</description></item>
    /// <item><description>Select a valid brick number (including the leading "1000000").</description></item>
    /// <item><description>Click on "Write to DB".</description></item>
    /// </list>
    /// Please notice that the "author" of the feedback will be the OPERA Computing Infrastructure user currently logged on (see OperaDbGUILogin/OperaDbTextLogin and <see cref="SySal.OperaDb.OperaDbCredentials"/>).
    /// </para>
    /// <para>
    /// In order to use OperaFeedback to view/check feedback information from the DB, take the following steps:
    /// <list type="number">
    /// <item><description>Select a valid brick number (including the leading "1000000").</description></item>
    /// <item><description>Click on ">" to get the available feedback data sets.</description></item>
    /// <item><description>Click on "Read from DB" to get the selected feedback data set.</description></item>
    /// </list>
    /// </para>
    /// </summary>
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        private void btnSel_Click(object sender, EventArgs e)
        {
            ofd.FileName = txtFileName.Text;
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                txtFileName.Text = ofd.FileName;
            }
        }

        SySal.TotalScan.Volume vol = null;

        private void btnLoad_Click(object sender, EventArgs e)
        {
            lvVertices.Items.Clear();
            lvTracks.Items.Clear();
            try
            {
                vol = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(txtFileName.Text, typeof(SySal.TotalScan.Volume));
            }
            catch (Exception x)
            {
                MessageBox.Show("Can't load TSR file.\r\n" + x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                vol = null;
            }
        }

        int iVertex = -1;
        int iTrack = -1;

        private void OnVertexLeave(object sender, EventArgs e)
        {
            try
            {
                iVertex = System.Convert.ToInt32(txtVertex.Text);
            }
            catch (Exception)
            {
                txtVertex.Text = iVertex.ToString();
                txtVertex.Focus();
            }
        }

        private void OnTrackLeave(object sender, EventArgs e)
        {
            try
            {
                iTrack = System.Convert.ToInt32(txtTrack.Text);
            }
            catch (Exception)
            {
                txtTrack.Text = iTrack.ToString();
                txtTrack.Focus();
            }
        }

        private void btnAddVertex_Click(object sender, EventArgs e)
        {
            if (iVertex < 0 || iVertex >= vol.Vertices.Length) return;
            int newid = AddVertex(iVertex);
        }

        int AddVertex(int iv)
        {
            int newid = lvVertices.Items.Count + 1;
            ListViewItem lvi = new ListViewItem(newid.ToString());
            SySal.TotalScan.Vertex v = vol.Vertices[iv];
            lvi.SubItems.Add(v.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(v.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(v.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(chkVtxPrimary.Checked ? "\xD7" : "");
            lvi.SubItems.Add(chkVtxCharm.Checked ? "\xD7" : "");
            lvi.SubItems.Add(chkVtxTau.Checked ? "\xD7" : "");
            lvi.SubItems.Add(chkVtxDeadMaterial.Checked ? "\xD7" : "");
            int i;
            if (chkVtxPrimary.Checked)
            {
                for (i = 0; i < lvVertices.Items.Count; i++)
                    if (lvVertices.Items[i].SubItems[4].Text == "\xD7")
                        lvVertices.Items[i].SubItems[4].Text = "";
            }
            lvi.Tag = iv;
            lvVertices.Items.Add(lvi);
            for (i = 0; i < v.Length; i++)
                AddTrack(v[i].Id);
            DrawTopology();
            CenterTracks();
            return newid;
        }

        int FindVertex(int tag)
        {
            int j;
            for (j = 0; j < lvVertices.Items.Count; j++)
                if ((int)lvVertices.Items[j].Tag == tag)
                    return System.Convert.ToInt32(lvVertices.Items[j].Text);
            return -1;
        }

        int AddTrack(int it)
        {
            int i;
            int jv;
            for (i = 0; i < lvTracks.Items.Count; i++)
                if ((int)lvTracks.Items[i].Tag == it)
                {
                    jv = -1;
                    if (vol.Tracks[it].Upstream_Vertex != null)
                        jv = FindVertex(vol.Tracks[it].Upstream_Vertex.Id);
                    if (jv > 0) lvTracks.Items[i].SubItems[1].Text = jv.ToString();
                    else lvTracks.Items[i].SubItems[1].Text = "";
                    jv = -1;
                    if (vol.Tracks[it].Downstream_Vertex != null)
                        jv = FindVertex(vol.Tracks[it].Downstream_Vertex.Id);
                    if (jv > 0) lvTracks.Items[i].SubItems[2].Text = jv.ToString();
                    else lvTracks.Items[i].SubItems[2].Text = "";
                    break;
                }
            int newid = lvTracks.Items.Count + 1;
            ListViewItem lvi = new ListViewItem(newid.ToString());
            SySal.TotalScan.Track tk = vol.Tracks[it];
            jv = -1;
            if (tk.Upstream_Vertex != null) jv = FindVertex(tk.Upstream_Vertex.Id);
            lvi.SubItems.Add((jv > 0) ? jv.ToString() : "");
            jv = -1;
            if (tk.Downstream_Vertex != null) jv = FindVertex(tk.Downstream_Vertex.Id);
            lvi.SubItems.Add((jv > 0) ? jv.ToString() : "");
            lvi.SubItems.Add((tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add((tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(tk.Upstream_Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(tk.Upstream_SlopeX.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(tk.Upstream_SlopeY.ToString("F5", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add("");
            lvi.SubItems.Add(cmbParticle.Text);
            lvi.SubItems.Add(chkTkScanback.Checked ? "\xD7" : "");
            lvi.SubItems.Add(cmbDarkness.Text);
            lvi.SubItems.Add("");
            lvi.SubItems.Add("");
            lvi.SubItems.Add("");
            lvi.SubItems.Add((tk.Upstream_Vertex == null) ? "" : tk.Upstream_Impact_Parameter.ToString("F1"));
            lvi.SubItems.Add((tk.Downstream_Vertex == null) ? "" : tk.Downstream_Impact_Parameter.ToString("F1"));
            lvi.SubItems.Add("");
            SySal.Processing.DecaySearchVSept09.KinkSearchResult kr = new SySal.Processing.DecaySearchVSept09.KinkSearchResult((SySal.TotalScan.Flexi.Track)tk);
            lvi.SubItems.Add(kr.TransverseSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(kr.LongitudinalSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(kr.TransverseMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(kr.LongitudinalMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            if (kr.KinkDelta > 3.0)
            {
                lvi.SubItems.Add(tk[kr.KinkIndex].LayerOwner.SheetId.ToString()).BackColor = Color.PaleVioletRed;
            }
            else
            {
                lvi.SubItems.Add("");
            }
            try
            {
                i = (int)tk.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("DECAYSEARCH"));
            }
            catch (Exception)
            {
                i = 0;
            }
            lvi.SubItems.Add(cmbDecaySearchFlag.Items[i].ToString());
            AutoColorSubItems(lvi);
            SySal.Tracking.MIPEmulsionTrackInfo [] segs = new SySal.Tracking.MIPEmulsionTrackInfo[tk.Length];
            for (i = 0; i < segs.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = tk[i].Info;
                info.Field = (uint)tk[i].LayerOwner.SheetId;
                segs[i] = info;
            }
            lvi.Tag = segs;
            lvTracks.Items.Add(lvi);
            return newid;
        }

        void DrawTopology()
        {
            gdiDisplayFdbck.Clear();
            gdiDisplayFdbck.AutoRender = false;
            foreach (ListViewItem lvi in lvVertices.Items)
            {
                SySal.BasicTypes.Vector vp = new SySal.BasicTypes.Vector();
                vp.X = System.Convert.ToDouble(lvi.SubItems[1].Text);
                vp.Y = System.Convert.ToDouble(lvi.SubItems[2].Text);
                vp.Z = System.Convert.ToDouble(lvi.SubItems[3].Text);
                int r = 127, g = 192, b = 255;
                if (lvi.SubItems[4].Text.Length > 0)
                {
                    r = g = b = 255;
                    if (lvi.SubItems[5].Text.Length > 0)
                    {
                        r = b = 63;
                        g = 255;
                    }
                    if (lvi.SubItems[6].Text.Length > 0)
                    {
                        r = g = 63;
                        b = 255;
                    }
                }
                gdiDisplayFdbck.Add(new GDI3D.Control.Point(vp.X, vp.Y, vp.Z, lvi, r, g, b));
            }
            foreach (ListViewItem lvi in lvTracks.Items)
            {
                SySal.BasicTypes.Vector tp = new SySal.BasicTypes.Vector();
                tp.X = System.Convert.ToDouble(lvi.SubItems[3].Text);
                tp.Y = System.Convert.ToDouble(lvi.SubItems[4].Text);
                tp.Z = System.Convert.ToDouble(lvi.SubItems[5].Text);
                SySal.BasicTypes.Vector2 td = new SySal.BasicTypes.Vector2();
                td.X = System.Convert.ToDouble(lvi.SubItems[6].Text);
                td.Y = System.Convert.ToDouble(lvi.SubItems[7].Text);
                SySal.BasicTypes.Vector ts = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector tf = new SySal.BasicTypes.Vector();
                int uv = -1, dv = -1;
                try
                {
                    uv = System.Convert.ToInt32(lvi.SubItems[1].Text);
                }
                catch (Exception)
                {
                    uv = -1;
                }
                try
                {
                    dv = System.Convert.ToInt32(lvi.SubItems[2].Text);
                }
                catch (Exception)
                {
                    dv = -1;
                }
                if (uv >= 0 && dv >= 0)
                {
                    ListViewItem vs = lvVertices.Items[uv - 1];
                    ts.X = System.Convert.ToDouble(vs.SubItems[1].Text);
                    ts.Y = System.Convert.ToDouble(vs.SubItems[2].Text);
                    ts.Z = System.Convert.ToDouble(vs.SubItems[3].Text);
                    ListViewItem vf = lvVertices.Items[dv - 1];
                    tf.X = System.Convert.ToDouble(vf.SubItems[1].Text);
                    tf.Y = System.Convert.ToDouble(vf.SubItems[2].Text);
                    tf.Z = System.Convert.ToDouble(vf.SubItems[3].Text);
                }
                else if (uv >= 0)
                {
                    ListViewItem vs = lvVertices.Items[uv - 1];
                    ts.X = System.Convert.ToDouble(vs.SubItems[1].Text);
                    ts.Y = System.Convert.ToDouble(vs.SubItems[2].Text);
                    ts.Z = System.Convert.ToDouble(vs.SubItems[3].Text);
                    tf.X = 5000.0 * td.X + ts.X;
                    tf.Y = 5000.0 * td.Y + ts.Y;
                    tf.Z = 5000.0 + ts.Z;
                }
                else if (dv >= 0)
                {
                    ListViewItem vs = lvVertices.Items[dv - 1];
                    tf.X = System.Convert.ToDouble(vs.SubItems[1].Text);
                    tf.Y = System.Convert.ToDouble(vs.SubItems[2].Text);
                    tf.Z = System.Convert.ToDouble(vs.SubItems[3].Text);
                    ts.X = tf.X - 5000.0 * td.X;
                    ts.Y = tf.Y - 5000.0 * td.Y;
                    ts.Z = tf.Z - 5000.0;
                }
                else
                {
                    ts.X = tp.X;
                    ts.Y = tp.Y;
                    ts.Z = tp.Z;
                    tf.X = (-tp.Z) * td.X + ts.X;
                    tf.Y = (-tp.Z) * td.Y + ts.Y;
                    tf.Z = 0.0;
                }
                int r = 63;
                int g = 128;
                int b = 255;
                if (lvi.SubItems[8].Text.Length > 0)
                {
                    r = 192;
                    g = 192;
                    b = 192;
                }
                if (lvi.SubItems[10].Text.Length > 0)
                {
                    r = 255;
                    g = 255;
                    b = 63;
                }
                if (lvi.SubItems[9].Text == "MUON")
                {
                    r = 255;
                    g = 64;
                    b = 64;
                }
                else if (lvi.SubItems[9].Text == "ELECTRON")
                {
                    r = 208;
                    g = 64;
                    b = 208;
                }
                if (lvi.Tag != null && ((SySal.Tracking.MIPEmulsionTrackInfo[])lvi.Tag).Length > 0)
                {
                    double dz1, dz2;
                    foreach (SySal.Tracking.MIPEmulsionTrackInfo info in (SySal.Tracking.MIPEmulsionTrackInfo[])lvi.Tag)
                    {
                        switch (info.AreaSum)
                        {
                            case 1: dz1 = 545.0; dz2 = 0.0; break;
                            case 2: dz1 = -210.0; dz2 = -755.0; break;
                            default: dz1 = 545.0; dz2 = -755.0; break;
                        }
                        gdiDisplayFdbck.Add(new GDI3D.Control.Line(
                            info.Intercept.X + dz1 * info.Slope.X,
                            info.Intercept.Y + dz1 * info.Slope.Y,
                            info.Intercept.Z + dz1,
                            info.Intercept.X + dz2 * info.Slope.X,
                            info.Intercept.Y + dz2 * info.Slope.Y,
                            info.Intercept.Z + dz2,
                            lvi, r, g, b
                            ));
                    }
                    if (uv >= 0)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = ((SySal.Tracking.MIPEmulsionTrackInfo[])lvi.Tag)[((SySal.Tracking.MIPEmulsionTrackInfo[])lvi.Tag).Length - 1];
                        gdiDisplayFdbck.Add(new GDI3D.Control.Line(ts.X, ts.Y, ts.Z, info.Intercept.X, info.Intercept.Y, info.Intercept.Z, lvi, r / 3 * 2, g / 3 * 2, b / 3 * 2));
                    }
                    if (dv >= 0)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = ((SySal.Tracking.MIPEmulsionTrackInfo[])lvi.Tag)[0];
                        gdiDisplayFdbck.Add(new GDI3D.Control.Line(tf.X, tf.Y, tf.Z, info.Intercept.X, info.Intercept.Y, info.Intercept.Z, lvi, r / 3 * 2, g / 3 * 2, b / 3 * 2));
                    }
                }
                else gdiDisplayFdbck.Add(new GDI3D.Control.Line(ts.X, ts.Y, ts.Z, tf.X, tf.Y, tf.Z, lvi, r, g, b));
            }
            gdiDisplayFdbck.AutoRender = true;
        }

        void CenterTracks()
        {
            GDI3D.Line [] lines = gdiDisplayFdbck.GetScene().Lines;
            if (lines.Length == 0) return;
            SySal.BasicTypes.Vector min = new SySal.BasicTypes.Vector();
            SySal.BasicTypes.Vector max = new SySal.BasicTypes.Vector();
            max.X = min.X = lines[0].XF;
            max.Y = min.Y = lines[0].YF;
            max.Z = min.Z = lines[0].ZF;            
            foreach (GDI3D.Line li in lines)
            {
                min.X = Math.Min(Math.Min(li.XF, li.XS), min.X);
                max.X = Math.Max(Math.Max(li.XF, li.XS), max.X);
                min.Y = Math.Min(Math.Min(li.YF, li.YS), min.Y);
                max.Y = Math.Max(Math.Max(li.YF, li.YS), max.Y);
                min.Z = Math.Min(Math.Min(li.ZF, li.ZS), min.Z);
                max.Z = Math.Max(Math.Max(li.ZF, li.ZS), max.Z);
            }
            gdiDisplayFdbck.AutoRender = false;
            gdiDisplayFdbck.SetCameraSpotting((min.X + max.X) * 0.5, (min.Y + max.Y) * 0.5, (min.Z + max.Z) * 0.5);
            gdiDisplayFdbck.Distance = 100000.0;
            gdiDisplayFdbck.Infinity = true;
            gdiDisplayFdbck.Zoom = 200.0 / (Math.Max(max.X - min.X, Math.Max(max.Y - min.Y, max.Z - min.Z) + 1));
            gdiDisplayFdbck.AutoRender = true;
            gdiDisplayFdbck.Transform();
            gdiDisplayFdbck.Render();
        }

        private void btnSetTrackAttr_Click(object sender, EventArgs e)
        {
            foreach (int i in lvTracks.SelectedIndices)
            {
                ListViewItem lvi = lvTracks.Items[i];
                lvi.SubItems[9].Text = cmbParticle.Text;
                lvi.SubItems[10].Text = chkTkScanback.Checked ? "\xD7" : "";
                lvi.SubItems[11].Text = cmbDarkness.Text;
                try
                {
                    lvi.SubItems[12].Text = System.Convert.ToDouble(txtP.Text).ToString();
                    lvi.SubItems[13].Text = System.Convert.ToDouble(txtPmin.Text).ToString();
                    lvi.SubItems[14].Text = System.Convert.ToDouble(txtPmax.Text).ToString();
                }
                catch (Exception)
                {
                    lvi.SubItems[12].Text = lvi.SubItems[13].Text = lvi.SubItems[14].Text = "";
                }
                try
                {
                    lvi.SubItems[18].Text = System.Convert.ToDouble(txtRMSDSlopeT.Text).ToString();
                    lvi.SubItems[19].Text = System.Convert.ToDouble(txtRMSDSlopeL.Text).ToString();
                    lvi.SubItems[20].Text = System.Convert.ToDouble(txtRDSlopeT.Text).ToString();
                    lvi.SubItems[21].Text = System.Convert.ToDouble(txtRDSlopeL.Text).ToString();                    
                }
                catch (Exception)
                {
                    lvi.SubItems[18].Text = lvi.SubItems[19].Text = lvi.SubItems[20].Text = lvi.SubItems[21].Text = "";
                }
                try
                {
                    lvi.SubItems[22].Text = System.Convert.ToInt32(txtKinkPlate.Text).ToString();
                }
                catch (Exception)
                {
                    lvi.SubItems[22].Text = "";
                }
                int lp = System.Convert.ToInt32(txtLastPlate.Text);
                if (chkTkOut.Checked ^ (lp <= 0))
                {
                    MessageBox.Show("Track " + lvi.SubItems[0].Text + " \'Out\' field is not consistent with LASTPLATE;\r\nthe latter must be 0 for contained tracks, positive for exiting tracks.", "Input Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    lvi.SubItems[17].Text = lp.ToString();
                }
                lvi.SubItems[23].Text = cmbDecaySearchFlag.Text;
                AutoColorSubItems(lvi);
            }
            DrawTopology();
            CenterTracks();
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void lvTracks_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lvTracks.SelectedItems.Count == 1)
            {
                ListViewItem lvi = lvTracks.SelectedItems[0];
                txtTkX.Text = lvi.SubItems[3].Text;
                txtTkY.Text = lvi.SubItems[4].Text;
                txtTkZ.Text = lvi.SubItems[5].Text;
                txtSX.Text = lvi.SubItems[6].Text;
                txtSY.Text = lvi.SubItems[7].Text;
                txtUpVtx.Text = lvi.SubItems[1].Text;
                txtDownVtx.Text = lvi.SubItems[2].Text;
                txtUpIP.Text = "{" + lvi.SubItems[15].Text + "}";
                txtDownIP.Text = "{" + lvi.SubItems[16].Text + "}";
                txtP.Text = lvi.SubItems[12].Text;
                txtPmin.Text = lvi.SubItems[13].Text;
                txtPmax.Text = lvi.SubItems[14].Text;
                cmbParticle.Text = lvi.SubItems[9].Text;
                chkTkScanback.Checked = lvi.SubItems[10].Text.Length > 0;
                cmbDarkness.Text = lvi.SubItems[11].Text;
                //chkTkOut.Checked = (String.Compare(lvi.SubItems[1].Text.Trim(), "OUT", true) == 0);
                txtLastPlate.Text = lvi.SubItems[17].Text;
                chkTkOut.Checked = (txtLastPlate.Text.Trim().Length == 0 || (SySal.OperaDb.Convert.ToInt32(txtLastPlate.Text) > 0));
                txtRMSDSlopeT.Text = lvi.SubItems[18].Text;
                txtRMSDSlopeL.Text = lvi.SubItems[19].Text;
                txtRDSlopeT.Text = lvi.SubItems[20].Text;
                txtRDSlopeL.Text = lvi.SubItems[21].Text;
                txtKinkPlate.Text = lvi.SubItems[22].Text;
                cmbDecaySearchFlag.Text = lvi.SubItems[23].Text;
            }
        }

        private void AddTrack(int uvid, int dvid, double upip, double dwip, bool manual, SySal.BasicTypes.Vector Pos, SySal.BasicTypes.Vector2 Slope, bool isout)
        {
            double P = -1.0, Pmin = -1.0, Pmax = -1.0;
            try
            {
                P = System.Convert.ToDouble(txtP.Text);
                Pmin = System.Convert.ToDouble(txtPmin.Text);
                Pmax = System.Convert.ToDouble(txtPmax.Text);
            }
            catch (Exception)
            {
                P = Pmin = Pmax = -1.0;
            }
            double rmsT = 0.0, rmsL = 0.0, maxT = 0.0, maxL = 0.0;
            int kp = 0;
            try
            {
                rmsT = System.Convert.ToDouble(txtRMSDSlopeT.Text);
                rmsL = System.Convert.ToDouble(txtRMSDSlopeL.Text);
                maxT = System.Convert.ToDouble(txtRDSlopeT.Text);
                maxL = System.Convert.ToDouble(txtRDSlopeL.Text);                
            }
            catch (Exception)
            {
                rmsT = rmsL = maxT = maxL = 0.0;                
            }
            try
            {
                kp = System.Convert.ToInt32(txtKinkPlate.Text);
            }
            catch (Exception)
            {
                kp = 0;
            }
            if (uvid == dvid && uvid == -1)
            {
                if (chkTkOneProng.Checked == false && isout == false)
                {
                    MessageBox.Show("Can't add vertex-less track", "Data error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                else
                {
                    if (isout == false)
                    {
                        int i;
                        uvid = 0;
                        for (i = 0; i < lvVertices.Items.Count; i++)
                            uvid = Math.Max(uvid, System.Convert.ToInt32(lvVertices.Items[i].SubItems[0].Text));
                        uvid++;
                        lvVertices.Items.Add(new ListViewItem(new string[] { uvid.ToString(), (Pos.X - 750.0 * Slope.X).ToString(), (Pos.Y - 750.0 * Slope.Y).ToString(), (Pos.Z - 750.0).ToString(), chkVtxPrimary.Checked ? "\xD7" : "", chkVtxCharm.Checked ? "\xD7" : "", chkVtxTau.Checked ? "\xD7" : "", chkVtxDeadMaterial.Checked ? "\xD7" : "" }));
                    }
                }
            }
            int lastplate = -1;
            if (isout)
            {
                try
                {
                    lastplate = System.Convert.ToInt32(txtLastPlate.Text);
                }
                catch (Exception)
                {
                    MessageBox.Show("An outbound track should have its LastPlate field defined.", "Data error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            int tkid = 0;
            foreach (ListViewItem xtk in lvTracks.Items)
                tkid = Math.Max(System.Convert.ToInt32(xtk.SubItems[0].Text), tkid);
            tkid++;
            ListViewItem lvi = new ListViewItem(new string[] { tkid.ToString(), (uvid >= 0) ? uvid.ToString() : (isout ? "Out" : ""), (dvid >= 0) ? dvid.ToString() : "", Pos.X.ToString(), Pos.Y.ToString(), Pos.Z.ToString(), Slope.X.ToString(), Slope.Y.ToString(), manual ? "\xD7" : "", cmbParticle.Text, 
                chkTkScanback.Checked ? "\xD7" : "", cmbDarkness.Text, P.ToString(), Pmin.ToString(), Pmax.ToString(), (uvid >= 0) ? upip.ToString() : "", (dvid >= 0) ? dwip.ToString() : "" , isout ? lastplate.ToString() :  "", 
                rmsT.ToString(), rmsL.ToString(), maxT.ToString(), maxL.ToString(), (kp > 0) ? kp.ToString() : "", cmbDecaySearchFlag.SelectedItem.ToString()});
            lvi.Tag = new SySal.Tracking.MIPEmulsionTrackInfo[0]; 
            lvTracks.Items.Add(lvi);            
            DrawTopology();
            CenterTracks();            
        }

        private void btnTkManual_Click(object sender, EventArgs e)
        {
            int uvid = -1, dvid = -1;
            bool isout = chkTkOut.Checked;
            double upip = 0.0, dwip = 0.0;
            SySal.BasicTypes.Vector Pos = new SySal.BasicTypes.Vector();
            SySal.BasicTypes.Vector2 Slope = new SySal.BasicTypes.Vector2();
            try
            {
                Pos.X = System.Convert.ToDouble(txtTkX.Text);
                Pos.Y = System.Convert.ToDouble(txtTkY.Text);
                Pos.Z = System.Convert.ToDouble(txtTkZ.Text);
                Slope.X = System.Convert.ToDouble(txtSX.Text);
                Slope.Y = System.Convert.ToDouble(txtSY.Text);

            }
            catch (Exception) 
            {
                MessageBox.Show("Bad position/slope data for track.", "Data Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            try
            {
                if (isout)
                {
                    uvid = -1;
                    upip = 0.0;
                }
                else
                {
                    uvid = System.Convert.ToInt32(txtUpVtx.Text);
                    upip = System.Convert.ToDouble(txtUpIP.Text);
                    string uvs = uvid.ToString();
                    bool found = false;
                    foreach (ListViewItem xuv in lvVertices.Items)
                        if (xuv.SubItems[0].Text == uvs)
                        {
                            found = true;
                            break;
                        }
                    if (found == false)
                    {
                        MessageBox.Show("Non-existing upstream vertex", "Data error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                }
            }
            catch (Exception) 
            {
                uvid = -1;
                upip = 0.0;
            }
            try
            {
                dvid = System.Convert.ToInt32(txtDownVtx.Text);
                dwip = System.Convert.ToDouble(txtDownIP.Text);
                string dvs = dvid.ToString();
                bool found = false;
                foreach (ListViewItem xdv in lvVertices.Items)
                    if (xdv.SubItems[0].Text == dvs)
                    {
                        found = true;
                        break;
                    }
                if (found == false)
                {
                    MessageBox.Show("Non-existing downstream vertex", "Data error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            catch (Exception) 
            {
                dvid = -1;
                dwip = 0.0;
            }
            AddTrack(uvid, dvid, upip, dwip, true, Pos, Slope, isout);
        }

        private void btnXY_Click(object sender, EventArgs e)
        {
            gdiDisplayFdbck.SetCameraOrientation(0, 0, -1, 0, 1, 0);
            gdiDisplayFdbck.Distance = gdiDisplayFdbck.Distance;
            gdiDisplayFdbck.Transform();
            gdiDisplayFdbck.Render();
        }

        private void btnXZ_Click(object sender, EventArgs e)
        {
            gdiDisplayFdbck.SetCameraOrientation(0, 1, 0, 0, 0, -1);
            gdiDisplayFdbck.Distance = gdiDisplayFdbck.Distance;
            gdiDisplayFdbck.Transform();
            gdiDisplayFdbck.Render();
        }

        private void btnYZ_Click(object sender, EventArgs e)
        {
            gdiDisplayFdbck.SetCameraOrientation(1, 0, 0, 0, 0, -1);
            gdiDisplayFdbck.Distance = gdiDisplayFdbck.Distance;
            gdiDisplayFdbck.Transform();
            gdiDisplayFdbck.Render();
        }

        string ReportQuery(string sql, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
        {
            System.Data.DataSet ds = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter(sql, conn, trans).Fill(ds);
            string outstr = "";
            foreach (System.Data.DataColumn dc in ds.Tables[0].Columns)
                outstr += (dc.Ordinal > 0 ? "\t" : "") + dc.ColumnName;
            outstr += "\r\n";
            foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
            {
                object [] o = dr.ItemArray;
                int i;
                for (i = 0; i < o.Length; i++)
                    outstr += (i > 0 ? "\t" : "") + o[i].ToString();
                outstr += "\r\n";
            }
            return outstr;
        }

        const double PositionRounding = 0.01;
        const double SlopeRounding = 0.00001;

        public void Preset(int brickid, SySal.TotalScan.Vertex[] vtxlist, SySal.TotalScan.Track[] tklist)
        {
            int i;
            double a;
            lvVertices.BeginUpdate();
            lvTracks.BeginUpdate();
            lvVertices.Items.Clear();
            lvTracks.Items.Clear();
            txtBrick.Text = brickid.ToString();
            OnBrickLeave(this, null);
            foreach (SySal.TotalScan.Vertex w in vtxlist)
            {
                ListViewItem lvi = new ListViewItem((lvVertices.Items.Count + 1).ToString());
                lvi.SubItems.Add(w.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(w.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(w.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY")); } catch (Exception) { a = 0.0; };
                lvi.SubItems.Add(a != 0.0 ? "\xD7" : "");
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("CHARM")); } catch (Exception) { a = 0.0; };
                lvi.SubItems.Add(a != 0.0 ? "\xD7" : "");
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("TAU")); } catch (Exception) { a = 0.0; };
                lvi.SubItems.Add(a != 0.0 ? "\xD7" : "");
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("OUTOFBRICK")); } catch (Exception) { a = 0.0; };
                lvi.SubItems.Add(a != 0.0 ?"\xD7" : "");
                lvVertices.Items.Add(lvi);
            }
            try
            {
                txtEvent.Text = ((long)tklist[0].GetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"))).ToString();
            }
            catch (Exception) { txtEvent.Text = ""; }
            foreach (SySal.TotalScan.Track w in tklist)
            {
                ListViewItem lvi = new ListViewItem((lvTracks.Items.Count + 1).ToString());
                string upipstr = "";
                string downipstr = "";
                if (w.Upstream_Vertex == null)
                {
                    lvi.SubItems.Add("");
                    upipstr = "";
                }
                else
                {
                    for (i = 0; i < vtxlist.Length && vtxlist[i] != w.Upstream_Vertex; i++) ;
                    if (i == vtxlist.Length)
                    {
                        lvi.SubItems.Add("");
                        upipstr = "";
                    }
                    else
                    {
                        lvi.SubItems.Add((i + 1).ToString());
                        try
                        {
                            upipstr = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("UpIP")).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                        }
                        catch (Exception)
                        {
                            upipstr = w.Upstream_Impact_Parameter.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                }
                if (w.Downstream_Vertex == null)
                {
                    lvi.SubItems.Add("");
                    downipstr = "";
                }
                else
                {
                    for (i = 0; i < vtxlist.Length && vtxlist[i] != w.Downstream_Vertex; i++) ;
                    if (i == vtxlist.Length)
                    {
                        lvi.SubItems.Add("");
                        downipstr = "";
                    }
                    else
                    {
                        lvi.SubItems.Add((i + 1).ToString());
                        try
                        {
                            downipstr = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("DownIP")).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                        }
                        catch (Exception)
                        {
                            downipstr = w.Downstream_Impact_Parameter.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                }
                SySal.BasicTypes.Vector p = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector2 s = new SySal.BasicTypes.Vector2();
                if (w.Upstream_Vertex != null)
                {
                    p.X = w.Upstream_PosX + (w.Upstream_Z - w.Upstream_PosZ) * w.Upstream_SlopeX;
                    p.Y = w.Upstream_PosY + (w.Upstream_Z - w.Upstream_PosZ) * w.Upstream_SlopeY;
                    p.Z = w.Upstream_Z;
                    s.X = w.Upstream_SlopeX;
                    s.Y = w.Upstream_SlopeY;
                }
                else
                {
                    p.X = w.Downstream_PosX + (w.Downstream_Z - w.Downstream_PosZ) * w.Downstream_SlopeX;
                    p.Y = w.Downstream_PosY + (w.Downstream_Z - w.Downstream_PosZ) * w.Downstream_SlopeY;
                    p.Z = w.Downstream_Z;
                    s.X = w.Downstream_SlopeX;
                    s.Y = w.Downstream_SlopeY;
                }
                lvi.SubItems.Add(p.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(p.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(p.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(s.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(s.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("MANUAL")); } catch (Exception) { a = 0.0; };
                lvi.SubItems.Add(a != 0.0 ? "\xD7" : "");
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE")); } catch (Exception) { a = 0.0; };
                switch ((int)a)
                {
                    case 13: lvi.SubItems.Add("MUON"); break;
                    case 11: lvi.SubItems.Add("ELECTRON"); break;
                    case 22: lvi.SubItems.Add("EPAIR"); break;
                    case 15: lvi.SubItems.Add("TAUON"); break;
                    case 4: lvi.SubItems.Add("CHARM"); break;
                    default: lvi.SubItems.Add(""); break;
                }
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("SCANBACK")); } catch (Exception) { a = 0.0; };
                lvi.SubItems.Add(a != 0.0 ? "\xD7" : "");
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("DARKNESS")); } catch (Exception) { a = 0.0; };
                if (a == 0.0) lvi.SubItems.Add("MIP");
                else if (a == 1.0) lvi.SubItems.Add("BLACK");
                else lvi.SubItems.Add("GREY");
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("P")); } catch (Exception) { a = -1.0; };
                lvi.SubItems.Add((a < 0.0) ? "" : a.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMIN")); } catch (Exception) { a = -1.0; };
                lvi.SubItems.Add((a < 0.0) ? "" : a.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMAX")); } catch (Exception) { a = -1.0; };
                lvi.SubItems.Add((a < 0.0) ? "" : a.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(upipstr);
                lvi.SubItems.Add(downipstr);
                try { a = w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("LASTPLATE")); } catch (Exception) { a = -1.0; };
                lvi.SubItems.Add((a < 0.0) ? "0" : a.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
                SySal.Processing.DecaySearchVSept09.KinkSearchResult kr = new SySal.Processing.DecaySearchVSept09.KinkSearchResult();
                if (w.Upstream_Vertex != null)
                    try
                    {
                        if (w.Upstream_Vertex.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY")) > 0.0) kr = new SySal.Processing.DecaySearchVSept09.KinkSearchResult((SySal.TotalScan.Flexi.Track)w);
                    }
                    catch (Exception) { }
                lvi.SubItems.Add(kr.TransverseMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(kr.LongitudinalMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(kr.TransverseSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(kr.LongitudinalSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                if (kr.KinkIndex >= 0)
                {
                    for (i = 0; i < w.Length && w[i].LayerOwner.Id != kr.KinkIndex; i++) ;
                    if (i == w.Length) throw new Exception("Wrong kink index.");
                    kr.KinkIndex = w[i].LayerOwner.SheetId;
                }
                lvi.SubItems.Add(kr.KinkIndex.ToString());
                if (kr.KinkDelta > 3.0)
                {
                    lvi.BackColor = Color.PaleVioletRed;
                }
                SySal.Tracking.MIPEmulsionTrackInfo[] segs = new SySal.Tracking.MIPEmulsionTrackInfo[w.Length];
                for (i = 0; i < segs.Length; i++)
                {
                    SySal.TotalScan.Segment ws = w[i];
                    segs[i] = ws.Info;
                    segs[i].Field = (uint)ws.LayerOwner.SheetId;
                    if (segs[i].Sigma < 0.0) segs[i].AreaSum = (uint)((Math.Abs(ws.LayerOwner.DownstreamZ - segs[i].TopZ) < Math.Abs(ws.LayerOwner.UpstreamZ - segs[i].BottomZ)) ? SegmentFlags.Up : SegmentFlags.Down);
                    else segs[i].AreaSum = (uint)SegmentFlags.Base;
                    if (segs[i].Count == 0) segs[i].AreaSum = (uint)((SegmentFlags)segs[i].AreaSum | SegmentFlags.Manual);
                    else segs[i].AreaSum = (uint)((SegmentFlags)segs[i].AreaSum | SegmentFlags.Auto);                    
                    segs[i].Intercept.X = Math.Round(segs[i].Intercept.X / PositionRounding) * PositionRounding;
                    segs[i].Intercept.Y = Math.Round(segs[i].Intercept.Y / PositionRounding) * PositionRounding;
                    segs[i].Intercept.Z = Math.Round(segs[i].Intercept.Z / PositionRounding) * PositionRounding;
                    segs[i].Slope.X = Math.Round(segs[i].Slope.X / SlopeRounding) * SlopeRounding;
                    segs[i].Slope.Y = Math.Round(segs[i].Slope.Y / SlopeRounding) * SlopeRounding;
                    segs[i].Slope.Z = 1.0;
                    segs[i].TopZ = Math.Round(segs[i].TopZ / PositionRounding) * PositionRounding;
                    segs[i].BottomZ = Math.Round(segs[i].BottomZ / PositionRounding) * PositionRounding;
                    segs[i].Sigma = Math.Round(segs[i].Sigma / SlopeRounding) * SlopeRounding;
                    if (ws is SySal.TotalScan.Flexi.Segment)
                    {
                        SySal.TotalScan.Flexi.DataSet ds = ((SySal.TotalScan.Flexi.Segment)ws).DataSet;
                        string dt = ds.DataType.ToUpper();
                        if (dt.StartsWith("MAN") || segs[i].Count == 0)
                        {
                            segs[i].Sigma = 0.0;
                            segs[i].AreaSum = (uint)(((SegmentFlags)segs[i].AreaSum & (~SegmentFlags.Mode)) | SegmentFlags.Manual);
                            segs[i].Count = 0;
                        }
                        else if (dt.StartsWith("SB"))
                        {
                            segs[i].AreaSum = (uint)(((SegmentFlags)segs[i].AreaSum & (~SegmentFlags.Mode)) | SegmentFlags.SBSF);
                        }
                    }
                }
                try
                {
                    i = (int)w.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("DECAYSEARCH"));
                }
                catch (Exception)
                {
                    i = 0;
                }
                lvi.SubItems.Add(cmbDecaySearchFlag.Items[i].ToString());
                AutoColorSubItems(lvi);
                lvi.Tag = segs;
                lvTracks.Items.Add(lvi);
            }
            lvTracks.EndUpdate();
            lvVertices.EndUpdate();
            DrawTopology();
            CenterTracks();
        }

        void SetInTrackDecaySearchParams(ListViewItem lvi, SySal.Tracking.MIPEmulsionTrackInfo[] minfo)
        {
            bool[] allowedkink = new bool[minfo.Length - 1];
            uint minplate = minfo[0].Field;
            foreach (SySal.Tracking.MIPEmulsionTrackInfo info in minfo)
                if (info.Field < minplate)
                    minplate = info.Field;
            int i;
            SySal.TotalScan.Segment[] segs = new SySal.TotalScan.Segment[minfo.Length];
            for (i = 0; i < minfo.Length; i++)
            {
                if (i < minfo.Length - 1) allowedkink[i] = (minfo[i].Field <= minplate + 4);
                segs[i] = new SySal.TotalScan.Segment(minfo[i], new SySal.TotalScan.NullIndex());
            }
            SySal.Processing.DecaySearchVSept09.KinkSearchResult kr = new SySal.Processing.DecaySearchVSept09.KinkSearchResult();
            kr = new SySal.Processing.DecaySearchVSept09.KinkSearchResult(segs, allowedkink);
            lvi.SubItems[18].Text = (kr.TransverseMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems[19].Text = (kr.LongitudinalMaxDeltaSlopeRatio.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems[20].Text = (kr.TransverseSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems[21].Text = (kr.LongitudinalSlopeRMS.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            if (kr.KinkIndex >= 0)
            {
                kr.KinkIndex = (int)minfo[kr.KinkIndex].Field;
            }
            if (kr.ExceptionMessage != null && kr.ExceptionMessage.Length > 0)
                MessageBox.Show(kr.ExceptionMessage, "In-track Decay Search failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            lvi.SubItems[22].Text = (kr.KinkIndex.ToString());
            if (kr.KinkDelta > 3.0)
            {
                lvi.BackColor = Color.PaleVioletRed;
            }
            else lvi.BackColor = lvTracks.BackColor;
        }

        void AutoColorSubItems(ListViewItem lvi)
        {
            Color c = lvTracks.BackColor;
            double t = 0.0;
            try
            {
                t = 0.0;
                t = System.Convert.ToDouble(lvi.SubItems[15].Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            if (t > 10.0) c = Color.Coral;
            try
            {
                t = 0.0;
                t = System.Convert.ToDouble(lvi.SubItems[16].Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception) { }
            if (t > 10.0) c = Color.Coral;            
            try
            {
                double rt = System.Convert.ToDouble(lvi.SubItems[20].Text);
                double rl = System.Convert.ToDouble(lvi.SubItems[21].Text);
                if (rt * rt + rl * rl > 3.0) c = Color.Coral;
            }
            catch (Exception) { }            
            lvi.BackColor = c;
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            gdiDisplayFdbck.Infinity = true;
            gdiDisplayFdbck.SetCameraOrientation(0, 0, -1, 0, 1, 0);
            cmbParticle.Text = "other";
            cmbDarkness.Text = "MIP";
            cmbDecaySearchFlag.SelectedIndex = 0;
            gdiDisplayFdbck.DoubleClickSelect = new GDI3D.Control.SelectObject(ItemSelect);
            SySal.OperaDb.OperaDbConnection Conn = null;
            try
            {
                Conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                Conn.Open();
                System.Data.DataSet ds1 = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT NAME FROM TB_MACHINES WHERE ID_SITE = (SELECT TO_NUMBER(VALUE) FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE') ORDER BY NAME", Conn).Fill(ds1);
                foreach (System.Data.DataRow dr1 in ds1.Tables[0].Rows)
                    cmbMachines.Items.Add(dr1[0].ToString());
                cmbMachines.SelectedIndex = 0;
                System.Data.DataSet ds2 = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT DESCRIPTION FROM (SELECT DESCRIPTION, row_number() over (order by decode(instr(DESCRIPTION,'Decay'),0,2,1) asc, decode(instr(EXECUTABLE, 'OperaFeedback.exe'),0,2,1) asc, id) as rnum FROM TB_PROGRAMSETTINGS WHERE DESCRIPTION LIKE '%Feedback%') ORDER BY RNUM", Conn).Fill(ds2);
                foreach (System.Data.DataRow dr2 in ds2.Tables[0].Rows)
                    cmbSettings.Items.Add(dr2[0].ToString());
                cmbSettings.SelectedIndex = 0;
            }
            catch (Exception x)
            {
                MessageBox.Show("DB services will not be available:\r\n" + x.Message, "DB Connection Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                //Close();
            }
            finally
            {
                if (Conn != null) Conn.Close();
            }
        }

        private void ItemSelect(object target)
        {
            ListViewItem lv = (ListViewItem)target;
            lv.Selected = true;
        }

        private void btnDelVertex_Click(object sender, EventArgs e)
        {
            if (lvVertices.SelectedItems.Count == 1)
            {
                ListViewItem lvi = lvVertices.SelectedItems[0];                
                int vid = System.Convert.ToInt32(lvi.SubItems[0].Text);
                lvVertices.Items.Remove(lvi);
                int i, nv;
                for (i = 0; i < lvVertices.Items.Count; i++)
                    if ((nv = System.Convert.ToInt32(lvVertices.Items[i].SubItems[0].Text)) > vid)
                        lvVertices.Items[i].SubItems[0].Text = (nv - 1).ToString();
                for (i = 0; i < lvTracks.Items.Count; i++)
                {
                    ListViewItem lvt = lvTracks.Items[i];
                    int uvid = -1, dvid = -1;
                    try
                    {
                        uvid = System.Convert.ToInt32(lvt.SubItems[1].Text);
                        if (uvid > vid) lvt.SubItems[1].Text = (uvid - 1).ToString();
                        else if (uvid == vid)
                        {
                            lvt.SubItems[1].Text = "";
                            uvid = -1;
                        }
                    }
                    catch (Exception) { }
                    try
                    {
                        dvid = System.Convert.ToInt32(lvt.SubItems[2].Text);
                        if (dvid > vid) lvt.SubItems[2].Text = (dvid - 1).ToString();
                        else if (dvid == vid)
                        {
                            lvt.SubItems[2].Text = "";
                            dvid = -1;
                        }
                    }
                    catch (Exception) { }
                    if (uvid == dvid && uvid == -1)
                    {
                        int j;
                        for (j = lvt.Index + 1; j < lvTracks.Items.Count; j++)
                            lvTracks.Items[j].SubItems[0].Text = (System.Convert.ToInt32(lvTracks.Items[j].SubItems[0].Text) - 1).ToString();
                        lvTracks.Items.Remove(lvt);
                        i--;
                    }
                }
                DrawTopology();
                CenterTracks();
            }
        }

        private void btnDelTrack_Click(object sender, EventArgs e)
        {
            if (lvTracks.SelectedItems.Count == 1)
            {
                lvTracks.Items.Remove(lvTracks.SelectedItems[0]);
                bool[] touched = new bool[lvVertices.Items.Count];
                int i;
                foreach (ListViewItem lvt in lvTracks.Items)
                {
                    try
                    {
                        touched[System.Convert.ToInt32(lvt.SubItems[1].Text) - 1] = true;
                    }
                    catch (Exception) { }
                    try
                    {
                        touched[System.Convert.ToInt32(lvt.SubItems[2].Text) - 1] = true;
                    }
                    catch (Exception) { }
                }
                for (i = 0; i < touched.Length; i++)
                    if (touched[i] == false)
                        MessageBox.Show("Warning: Vertex " + lvVertices.Items[i].SubItems[0].Text + " now has no tracks.", "Data warning");
                DrawTopology();
                CenterTracks();
            }
        }

        private void btnAddTrack_Click(object sender, EventArgs e)
        {
            if (iTrack < 0 || iTrack >= vol.Tracks.Length) return;
            SySal.TotalScan.Track tk = vol.Tracks[iTrack];
            if (tk.Upstream_Vertex != null) AddVertex(tk.Upstream_Vertex.Id);
            if (tk.Downstream_Vertex != null) AddVertex(tk.Downstream_Vertex.Id);
            if (tk.Upstream_Vertex == null && tk.Downstream_Vertex == null && chkTkOneProng.Checked)
            {
                SySal.BasicTypes.Vector pos = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector2 slope = new SySal.BasicTypes.Vector2();
                pos.X = tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX;
                pos.Y = tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY;
                pos.Z = tk.Upstream_Z;
                slope.X = tk.Upstream_SlopeX;
                slope.Y = tk.Upstream_SlopeY;
                AddTrack(-1, -1, 0.0, 0.0, false, pos, slope, chkTkOut.Checked);
            }
        }

        private void btnZoomIn_Click(object sender, EventArgs e)
        {
            gdiDisplayFdbck.Zoom *= 1.1;
        }

        private void btnZoomOut_Click(object sender, EventArgs e)
        {
            gdiDisplayFdbck.Zoom /= 1.1;
        }

        private void btnVtxManual_Click(object sender, EventArgs e)
        {
            int newid = lvVertices.Items.Count + 1;
            try
            {
                ListViewItem lvi = new ListViewItem(new string[] {newid.ToString(), System.Convert.ToDouble(txtVX.Text).ToString("F1"), System.Convert.ToDouble(txtVY.Text).ToString("F1"), System.Convert.ToDouble(txtVZ.Text).ToString("F1"),
                    chkVtxPrimary.Checked ? "\xD7" : "", chkVtxCharm.Checked ? "\xD7" : "", chkVtxTau.Checked ? "\xD7" : "", chkVtxDeadMaterial.Checked ? "\xD7" : ""});
                if (chkVtxPrimary.Checked)
                {
                    int i;
                    for (i = 0; i < lvVertices.Items.Count; i++)
                        if (lvVertices.Items[i].SubItems[4].Text == "\xD7")
                            lvVertices.Items[i].SubItems[4].Text = "";
                }
                lvVertices.Items.Add(lvi);
            }
            catch (Exception)
            {
                MessageBox.Show("Incorrect vertex position provided.", "Data Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            DrawTopology();
            CenterTracks();            
        }

        int iBrick = -1;

        private void OnBrickLeave(object sender, EventArgs e)
        {
            try
            {
                iBrick = System.Convert.ToInt32(txtBrick.Text);
            }
            catch (Exception) 
            {
                iBrick = -1;
                txtBrick.Text = "";
            }
        }

        private void btnSelFeedback_Click(object sender, EventArgs e)
        {
            System.Data.DataSet ds = new DataSet();
            SySal.OperaDb.OperaDbConnection Conn = null;
            try
            {
                Conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                Conn.Open();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_RECONSTRUCTION FROM VW_FEEDBACK_RECONSTRUCTIONS WHERE ID_EVENTBRICK = " + iBrick + " ORDER BY ID_RECONSTRUCTION ASC", Conn).Fill(ds);
                cmbOp.Items.Clear();
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    cmbOp.Items.Add(dr[0].ToString());                
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (Conn != null) Conn.Close();
            }
        }

        private void btnWrite_Click(object sender, EventArgs e)
        {
            long idproc = 0;
            long idrec = 0;
            long idset = 0;
            long idmachine = 0;
            long iduser = 0;
            SegmentFlags st = new SegmentFlags();
            SegmentFlags sm = new SegmentFlags();
            SySal.OperaDb.OperaDbConnection Conn = null;
            SySal.OperaDb.OperaDbTransaction Trans = null;
            SySal.OperaDb.OperaDbCredentials cred = null;
            string step = "";
            if (txtEvent.Text.Length == 0)
            {
                MessageBox.Show("Please fill in the event number", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                txtEvent.Focus();
                return;
            }
            try
            {
                step = "Connection open.";
                Conn = (cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord()).Connect();
                Conn.Open();
                Trans = Conn.BeginTransaction();

                iduser = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(cred.OPERAUserName, cred.OPERAPassword, Conn, Trans);
                idmachine = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_MACHINES WHERE NAME = '" + cmbMachines.Text + "' AND ID_SITE = (SELECT TO_NUMBER(VALUE) FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE')", Conn, Trans).ExecuteScalar());
                idset = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_PROGRAMSETTINGS WHERE DESCRIPTION = '" + cmbSettings.Text + "'", Conn, Trans).ExecuteScalar());

                step = "Create operation.";
                SySal.OperaDb.OperaDbCommand cmd1 = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROC_OPERATION_BRICK(" + idmachine + "," + idset + "," + iduser + "," + iBrick + ",NULL,systimestamp,NULL,:newid)", Conn, Trans);
                cmd1.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Output);
                cmd1.ExecuteNonQuery();
                idproc = SySal.OperaDb.Convert.ToInt64(cmd1.Parameters[0].Value);

                step = "Insert Reconstruction.";
                new SySal.OperaDb.OperaDbCommand("INSERT INTO VW_FEEDBACK_RECONSTRUCTIONS (ID_EVENTBRICK, ID_PROCESSOPERATION) VALUES (" + iBrick + "," + idproc + ")", Conn, Trans).ExecuteNonQuery();

                step = "Get Reconstruction.";
                idrec = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID_RECONSTRUCTION FROM VW_FEEDBACK_RECONSTRUCTIONS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_PROCESSOPERATION = " + idproc, Conn, Trans).ExecuteScalar());

                SySal.OperaDb.OperaDbCommand cmd3 = new SySal.OperaDb.OperaDbCommand("INSERT INTO VW_FEEDBACK_VERTICES (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_VERTEX, POSX, POSY, POSZ, ISPRIMARY, ISCHARM, ISTAU, OUTOFBRICK) VALUES (:idb,:idr,:idv,:x,:y,:z,:isprim,:ischarm,:istau,:outofbrick)", Conn, Trans);
                cmd3.Parameters.Add("idb", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                cmd3.Parameters.Add("idr", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                cmd3.Parameters.Add("idv", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd3.Parameters.Add("x", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd3.Parameters.Add("y", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd3.Parameters.Add("z", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd3.Parameters.Add("isprim", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd3.Parameters.Add("ischarm", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd3.Parameters.Add("istau", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd3.Parameters.Add("outofbrick", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                foreach (ListViewItem lvv in lvVertices.Items)
                {
                    cmd3.Parameters[0].Value = iBrick;
                    cmd3.Parameters[1].Value = idrec;
                    cmd3.Parameters[2].Value = System.Convert.ToInt32(lvv.SubItems[0].Text);
                    cmd3.Parameters[3].Value = System.Convert.ToDouble(lvv.SubItems[1].Text);
                    cmd3.Parameters[4].Value = System.Convert.ToDouble(lvv.SubItems[2].Text);
                    cmd3.Parameters[5].Value = System.Convert.ToDouble(lvv.SubItems[3].Text);
                    cmd3.Parameters[6].Value = (lvv.SubItems[4].Text.Length > 0) ? "Y" : "N";
                    cmd3.Parameters[7].Value = (lvv.SubItems[5].Text.Length > 0) ? "Y" : "N";
                    cmd3.Parameters[8].Value = (lvv.SubItems[6].Text.Length > 0) ? "Y" : "N";
                    cmd3.Parameters[9].Value = (lvv.SubItems[7].Text.Length > 0) ? "DEAD_MATERIAL" : "N";
                    step = "Insert Vertex.";
                    cmd3.ExecuteNonQuery();
                }

                SySal.OperaDb.OperaDbCommand cmd4 = new SySal.OperaDb.OperaDbCommand("INSERT INTO VW_FEEDBACK_TRACKS (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_TRACK, ID_UPVTX, ID_DOWNVTX, X, Y, Z, SX, SY, MANUAL, PARTICLE, SCANBACK, DARKNESS, P, PMIN, PMAX, UPIP, DOWNIP, OUTOFBRICK, LASTPLATE, RSLOPET, RSLOPEL, RMSSLOPET, RMSSLOPEL, KINKPLATEDOWN, KINKPLATEUP, DECAYSEARCH, EVENT) VALUES " +
                    "(:idb,:idr,:idt,:iduv,:iddv,:x,:y,:z,:sx,:sy,:manual,:particle,:sb,:darkness,:p,:pmin,:pmax,:upip,:dwip,:outofbrick,:lastplate,:rslopet,:rslopel,:rt,:rl,:kpd,:kpu,:ds, " + txtEvent.Text + ")", Conn, Trans);
                cmd4.Parameters.Add("idb", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                cmd4.Parameters.Add("idr", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                cmd4.Parameters.Add("idt", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd4.Parameters.Add("iduv", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd4.Parameters.Add("iddv", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd4.Parameters.Add("x", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("y", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("z", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("sx", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("sy", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("manual", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd4.Parameters.Add("particle", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd4.Parameters.Add("sb", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd4.Parameters.Add("darkness", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd4.Parameters.Add("p", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("pmin", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("pmax", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("upip", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("dwip", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("outofbrick", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd4.Parameters.Add("lastplate", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd4.Parameters.Add("rslopet", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("rslopel", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input); 
                cmd4.Parameters.Add("rmst", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("rmsl", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd4.Parameters.Add("kpd", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd4.Parameters.Add("kpu", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd4.Parameters.Add("ds", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                SySal.OperaDb.OperaDbCommand cmd5 = new SySal.OperaDb.OperaDbCommand("INSERT INTO VW_FEEDBACK_SEGMENTS (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_TRACK, ID_PLATE, TRACKMODE, TRACKTYPE, GRAINS, POSX, POSY, POSZ, SLOPEX, SLOPEY) VALUES " +
                    "(:idb,:idr,:idt,:idp,:tkm,:tkt,:g,:x,:y,:z,:sx,:sy)", Conn, Trans);
                cmd5.Parameters.Add("idb", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                cmd5.Parameters.Add("idr", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                cmd5.Parameters.Add("idt", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd5.Parameters.Add("idp", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd5.Parameters.Add("tkm", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd5.Parameters.Add("tkt", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input);
                cmd5.Parameters.Add("g", SySal.OperaDb.OperaDbType.Int, ParameterDirection.Input);
                cmd5.Parameters.Add("x", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd5.Parameters.Add("y", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd5.Parameters.Add("z", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd5.Parameters.Add("sx", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                cmd5.Parameters.Add("sy", SySal.OperaDb.OperaDbType.Double, ParameterDirection.Input);
                foreach (ListViewItem lvt in lvTracks.Items)
                {
                    int idt;
                    cmd4.Parameters[0].Value = iBrick;
                    cmd4.Parameters[1].Value = idrec;
                    cmd4.Parameters[2].Value = (idt = System.Convert.ToInt32(lvt.SubItems[0].Text));
                    if (lvt.SubItems[1].Text.Length == 0)
                    {
                        cmd4.Parameters[3].Value = System.DBNull.Value;
                        cmd4.Parameters[19].Value = System.DBNull.Value;
                        cmd4.Parameters[20].Value = System.DBNull.Value;
                    }
                    else
                    {
                        int lastplate = System.Convert.ToInt32(lvt.SubItems[17].Text);
                        if (lastplate > 0)
                        {
                            cmd4.Parameters[19].Value = (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_PLATES WHERE (ID_EVENTBRICK, Z) = (SELECT ID_EVENTBRICK, MIN(Z) FROM TB_PLATES WHERE ID_EVENTBRICK = " + iBrick + " GROUP BY ID_EVENTBRICK)", Conn).ExecuteScalar()) == lastplate) ? "PASSING_THROUGH" : "EDGE_OUT";
                            cmd4.Parameters[20].Value = lastplate;
                        }
                        else
                        {
                            cmd4.Parameters[3].Value = System.Convert.ToInt32(lvt.SubItems[1].Text);
                            cmd4.Parameters[19].Value = System.DBNull.Value;
                            cmd4.Parameters[20].Value = 0;
                        }
                    }
                    if (lvt.SubItems[2].Text.Length == 0) cmd4.Parameters[4].Value = System.DBNull.Value;
                    else cmd4.Parameters[4].Value = System.Convert.ToInt32(lvt.SubItems[2].Text);
                    cmd4.Parameters[5].Value = System.Convert.ToDouble(lvt.SubItems[3].Text);
                    cmd4.Parameters[6].Value = System.Convert.ToDouble(lvt.SubItems[4].Text);
                    cmd4.Parameters[7].Value = System.Convert.ToDouble(lvt.SubItems[5].Text);
                    cmd4.Parameters[8].Value = System.Convert.ToDouble(lvt.SubItems[6].Text);
                    cmd4.Parameters[9].Value = System.Convert.ToDouble(lvt.SubItems[7].Text);
                    cmd4.Parameters[10].Value = (lvt.SubItems[8].Text.Length > 0) ? "Y" : "N";
                    cmd4.Parameters[11].Value = lvt.SubItems[9].Text;
                    cmd4.Parameters[12].Value = (lvt.SubItems[10].Text.Length > 0) ? "Y" : "N";
                    cmd4.Parameters[13].Value = lvt.SubItems[11].Text;
                    if (lvt.SubItems[12].Text.Length == 0) cmd4.Parameters[14].Value = System.DBNull.Value;
                    else cmd4.Parameters[14].Value = System.Convert.ToDouble(lvt.SubItems[12].Text);
                    if (lvt.SubItems[13].Text.Length == 0) cmd4.Parameters[15].Value = System.DBNull.Value;
                    else cmd4.Parameters[15].Value = System.Convert.ToDouble(lvt.SubItems[13].Text);
                    if (lvt.SubItems[14].Text.Length == 0) cmd4.Parameters[16].Value = System.DBNull.Value;
                    else cmd4.Parameters[16].Value = System.Convert.ToDouble(lvt.SubItems[14].Text);
                    if (lvt.SubItems[15].Text.Length == 0) cmd4.Parameters[17].Value = System.DBNull.Value;
                    else cmd4.Parameters[17].Value = System.Convert.ToDouble(lvt.SubItems[15].Text);
                    if (lvt.SubItems[16].Text.Length == 0) cmd4.Parameters[18].Value = System.DBNull.Value;
                    else cmd4.Parameters[18].Value = System.Convert.ToDouble(lvt.SubItems[16].Text);
                    
                    cmd4.Parameters[21].Value = System.Convert.ToDouble(lvt.SubItems[18].Text);
                    cmd4.Parameters[22].Value = System.Convert.ToDouble(lvt.SubItems[19].Text);
                    cmd4.Parameters[23].Value = System.Convert.ToDouble(lvt.SubItems[20].Text);
                    cmd4.Parameters[24].Value = System.Convert.ToDouble(lvt.SubItems[21].Text);
                    if (lvt.SubItems[22].Text.Length == 0)
                    {
                        cmd4.Parameters[25].Value = System.DBNull.Value;
                        cmd4.Parameters[26].Value = System.DBNull.Value;
                    }
                    else
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo[] segs = (SySal.Tracking.MIPEmulsionTrackInfo[])lvt.Tag;
                        int kplj = Convert.ToInt32(lvt.SubItems[22].Text);
                        int kpli;
                        if (kplj < 0) kpli = kplj;
                        else
                        {
                            if (segs.Length < 2) kpli = kplj - 1;
                            else
                            {
                                int j;
                                for (j = 0; j < segs.Length - 1 && kplj != (int)segs[j].Field; j++) ;
                                if (j + 1 >= segs.Length) throw new Exception("Kink plate cannot be the most upstream one (track " + lvt.SubItems[0].Text + ")");
                                kpli = (int)segs[j + 1].Field;
                            }
                        }
                        cmd4.Parameters[25].Value = kplj;
                        cmd4.Parameters[26].Value = kpli;
                    }
                    cmd4.Parameters[27].Value = lvt.SubItems[23].Text;
                    step = "Insert Track.";
                    cmd4.ExecuteNonQuery();                    
                    foreach (SySal.Tracking.MIPEmulsionTrackInfo info in (SySal.Tracking.MIPEmulsionTrackInfo [])lvt.Tag)
                    {
                        cmd5.Parameters[0].Value = iBrick;
                        cmd5.Parameters[1].Value = idrec;
                        cmd5.Parameters[2].Value = idt;
                        cmd5.Parameters[3].Value = (int)info.Field;
                        st = (SegmentFlags)info.AreaSum & SegmentFlags.Type;
                        sm = (SegmentFlags)info.AreaSum & SegmentFlags.Mode;
                        cmd5.Parameters[4].Value = (sm == SegmentFlags.SBSF) ? "S" : ((sm == SegmentFlags.Auto) ? "A" : "M");
                        cmd5.Parameters[5].Value = (st == SegmentFlags.Base) ? "B" : ((st == SegmentFlags.Down) ? "D" : "U");
                        /*
                        cmd5.Parameters[4].Value = (info.Sigma < 0.0) ? "S" : (info.Sigma > 0.0 ? "A" : "M");
                        cmd5.Parameters[5].Value = (info.AreaSum == 0) ? "B" : (info.AreaSum == 1 ? "D" : "U");
                         */
                        cmd5.Parameters[6].Value = (int)info.Count;
                        cmd5.Parameters[7].Value = info.Intercept.X;
                        cmd5.Parameters[8].Value = info.Intercept.Y;
                        cmd5.Parameters[9].Value = info.Intercept.Z;
                        cmd5.Parameters[10].Value = info.Slope.X;
                        cmd5.Parameters[11].Value = info.Slope.Y;
                        step = "Insert Segment.";
                        cmd5.ExecuteNonQuery();
                    }
                }

                step = "Close Operation.";
                new SySal.OperaDb.OperaDbCommand("CALL PC_SUCCESS_OPERATION(" + idproc + ", systimestamp)", Conn, Trans).ExecuteNonQuery();
                if (chkTestWrite.Checked)
                {
                    string report = ReportQuery("SELECT * FROM VW_FEEDBACK_RECONSTRUCTIONS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + idrec, Conn, Trans);
                    report += ReportQuery("SELECT * FROM VW_FEEDBACK_VERTICES WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + idrec + " ORDER BY ID_VERTEX", Conn, Trans);
                    report += ReportQuery("SELECT * FROM VW_FEEDBACK_TRACKS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + idrec + " ORDER BY ID_TRACK", Conn, Trans);
                    report += ReportQuery("SELECT * FROM VW_FEEDBACK_SEGMENTS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + idrec + " ORDER BY ID_TRACK, ID_PLATE", Conn, Trans);
                    SQLReportForm sqlr = new SQLReportForm();
                    sqlr.rtSQL.Text = report;
                    sqlr.ShowDialog();
                    Trans.Rollback();
                    MessageBox.Show("DB insertion test OK", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else
                {
                    Trans.Commit();
                    MessageBox.Show("Data successfully written to DB", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(step + "\r\n" + x.ToString(), "Data Insertion Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (Conn != null) Conn.Close();
            }
        }

        private void btnSetVertexAttr_Click(object sender, EventArgs e)
        {
            foreach (int i in lvVertices.SelectedIndices)
            {
                ListViewItem lvi = lvVertices.Items[i];
                lvi.SubItems[5].Text = chkVtxCharm.Checked ? "\xD7" : "";
                lvi.SubItems[6].Text = chkVtxTau.Checked ? "\xD7" : "";
                lvi.SubItems[7].Text = chkVtxDeadMaterial.Checked ? "\xD7" : "";
                if (chkVtxPrimary.Checked)
                {
                    int j;
                    for (j = 0; j < lvVertices.Items.Count; j++)
                        lvVertices.Items[j].SubItems[4].Text = "";
                }
                lvi.SubItems[4].Text = chkVtxPrimary.Checked ? "\xD7" : "";
            }
            DrawTopology();
            CenterTracks();
        }

        private static string[] RowStrings(System.Data.DataRow dr)
        {
            object [] o = dr.ItemArray;
            string[] s = new string[o.Length];
            int i;
            for (i = 0; i < s.Length; i++)
                s[i] = (String.Compare(o[i].ToString(),"X",true) == 0) ? "\xD7" : o[i].ToString();
            return s;
        }

        private void btnReadFromDB_Click(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbConnection Conn = null;
            try
            {
                Conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                Conn.Open();
                lvVertices.Items.Clear();
                lvTracks.Items.Clear();
                txtEvent.Text = new SySal.OperaDb.OperaDbCommand("SELECT EVENT FROM VW_FEEDBACK_RECONSTRUCTIONS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + cmbOp.Text, Conn).ExecuteScalar().ToString();
                System.Data.DataSet dsv = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_VERTEX, POSX, POSY, POSZ, DECODE(ISPRIMARY, 'Y', 'X', ''), DECODE(ISCHARM, 'Y', 'X', ''), DECODE(ISTAU, 'Y', 'X', ''), DECODE(OUTOFBRICK, 'DEAD_MATERIAL', 'X', '') FROM VW_FEEDBACK_VERTICES WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + cmbOp.Text + " ORDER BY ID_VERTEX", Conn).Fill(dsv);
                foreach (System.Data.DataRow dr in dsv.Tables[0].Rows)
                    lvVertices.Items.Add(new ListViewItem(RowStrings(dr)));
                System.Data.DataSet dst = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_TRACK, ID_UPVTX, ID_DOWNVTX, X, Y, Z, SX, SY, DECODE(MANUAL, 'Y', 'X', ''), PARTICLE, DECODE(SCANBACK, 'Y', 'X', ''), DARKNESS, P, PMIN, PMAX, UPIP, DOWNIP, LASTPLATE, RSLOPET, RSLOPEL, RMSSLOPET, RMSSLOPEL, KINKPLATEDOWN, DECAYSEARCH FROM VW_FEEDBACK_TRACKS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + cmbOp.Text + " ORDER BY ID_TRACK", Conn).Fill(dst);
                foreach (System.Data.DataRow dr in dst.Tables[0].Rows)
                    lvTracks.Items.Add(new ListViewItem(RowStrings(dr)));
                System.Data.DataSet dss = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_TRACK, ID_PLATE, TRACKMODE, TRACKTYPE, GRAINS, POSX, POSY, POSZ, SLOPEX, SLOPEY, TKNUM FROM (SELECT ID_TRACK, ID_PLATE, TRACKMODE, TRACKTYPE, GRAINS, POSX, POSY, POSZ, SLOPEX, SLOPEY, ROW_NUMBER() OVER (PARTITION BY ID_TRACK ORDER BY POSZ ASC) AS TKNUM FROM VW_FEEDBACK_SEGMENTS WHERE ID_EVENTBRICK = " + iBrick + " AND ID_RECONSTRUCTION = " + cmbOp.Text + ") ORDER BY ID_TRACK ASC, TKNUM DESC, ID_PLATE DESC", Conn).Fill(dss);
                SySal.Tracking.MIPEmulsionTrackInfo [] segs = null;
                foreach (System.Data.DataRow dr in dss.Tables[0].Rows)
                {
                    if (segs == null)
                        segs = new SySal.Tracking.MIPEmulsionTrackInfo[SySal.OperaDb.Convert.ToInt32(dr[10])];
                    SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    info.Field = SySal.OperaDb.Convert.ToUInt32(dr[1]);
                    info.AreaSum = 0;
                    switch (dr[2].ToString().ToUpper())
                    {
                        case "A": info.AreaSum = (uint)((SegmentFlags)info.AreaSum | SegmentFlags.Auto); info.Sigma = 1.0; break;
                        case "M": info.AreaSum = (uint)((SegmentFlags)info.AreaSum | SegmentFlags.Manual); info.Sigma = 0.0; break;
                        case "S": info.AreaSum = (uint)((SegmentFlags)info.AreaSum | SegmentFlags.SBSF); info.Sigma = -1.0; break;
                        default: throw new Exception("Unsupported trackmode \"" + dr[2].ToString() + "\".");
                    }                    
                    switch (dr[3].ToString().ToUpper())
                    {
                        case "B": info.AreaSum = (uint)((SegmentFlags)info.AreaSum | SegmentFlags.Base); break;
                        case "D": info.AreaSum = (uint)((SegmentFlags)info.AreaSum | SegmentFlags.Down); break;
                        case "U": info.AreaSum = (uint)((SegmentFlags)info.AreaSum | SegmentFlags.Up); break;
                        default: throw new Exception("Unsupported tracktype \"" + dr[3].ToString() + "\".");
                    }                    
                    info.Count = SySal.OperaDb.Convert.ToUInt16(dr[4]);
                    info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[5]);
                    info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[6]);
                    info.Intercept.Z = SySal.OperaDb.Convert.ToDouble(dr[7]);
                    info.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[8]);
                    info.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[9]);
                    info.Slope.Z = 1.0;
                    segs[segs.Length - SySal.OperaDb.Convert.ToInt32(dr[10])] = info;
                    if (segs[segs.Length - 1] != null)
                    {
                        lvTracks.Items[SySal.OperaDb.Convert.ToInt32(dr[0]) - 1].Tag = segs;
                        segs = null;
                    }
                }
                DrawTopology();
                CenterTracks();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB Read Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (Conn != null) Conn.Close();
            }
        }

        private void btnAccessSegments_Click(object sender, EventArgs e)
        {
            if (lvTracks.SelectedItems.Count == 1)
            {
                SegmentForm sf = new SegmentForm();
                sf.Text = "Segments of track #" + lvTracks.SelectedItems[0].SubItems[0].Text;
                sf.Segments = (SySal.Tracking.MIPEmulsionTrackInfo[])lvTracks.SelectedItems[0].Tag;
                sf.ShowDialog();
                lvTracks.SelectedItems[0].Tag = sf.Segments;
                DrawTopology();
                CenterTracks();
            }
        }

        private void OnEventLeave(object sender, EventArgs e)
        {
            try
            {
                System.Convert.ToInt64(txtEvent.Text);
            }
            catch (Exception) { txtEvent.Text = ""; }            
        }

        private void btnExport_Click(object sender, EventArgs e)
        {
            SaveFileDialog sd = new SaveFileDialog();
            sd.Title = "Select ASCII feedback file";
            sd.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            string b = "000000" + txtBrick.Text;
            sd.FileName = "B" + b.Substring(b.Length - 6, 6) + "_e" + txtEvent.Text + ".feedback.txt";
            if (sd.ShowDialog() == DialogResult.OK)
#if (DEBUG)
#else
                try
#endif
                {
                    string ot = "";
                    foreach (ListViewItem lvi1 in lvVertices.Items)
                    {
                        int nd = 0;
                        int nu = 0;
                        foreach (ListViewItem lvi2 in lvTracks.Items)
                        {
                            if (lvi2.SubItems[1].Text == lvi1.SubItems[0].Text) nd++;
                            if (lvi2.SubItems[2].Text == lvi1.SubItems[0].Text) nu++;
                        }
                        ot += ("\r\n" + lvi1.SubItems[0].Text + " " + lvi1.SubItems[1].Text + " " + lvi1.SubItems[2].Text + " " + lvi1.SubItems[3].Text + " " + (lvi1.SubItems[4].Text.Length > 0 ? "1" : "0") + " " + (lvi1.SubItems[5].Text.Length > 0 ? "1" : "0") + " " + (lvi1.SubItems[6].Text.Length > 0 ? "1" : "0") + " " + nd + " " + nu + " " + (lvi1.SubItems[7].Text.Length > 0 ? "1" : "0"));
                    }
                    foreach (ListViewItem lvi2 in lvTracks.Items)
                    {                        
                        int j;
                        int ptype;
                        for (ptype = 0; ptype < cmbParticle.Items.Count && String.Compare(cmbParticle.Items[ptype].ToString(), lvi2.SubItems[9].Text, true) != 0; ptype++) ;
                        if (ptype == cmbParticle.Items.Count) throw new Exception("Unsupported particle type.");
                        int dkns;
                        for (dkns = 0; dkns < cmbDarkness.Items.Count && String.Compare(cmbDarkness.Items[dkns].ToString(), lvi2.SubItems[11].Text, true) != 0; dkns++) ;
                        if (dkns == cmbDarkness.Items.Count) throw new Exception("Unsupported darkness.");
                        SySal.Tracking.MIPEmulsionTrackInfo[] segs = new SySal.Tracking.MIPEmulsionTrackInfo[0];                        
                        if (lvi2.Tag != null) segs = ((SySal.Tracking.MIPEmulsionTrackInfo[])lvi2.Tag);                       
                        int kplj = Convert.ToInt32(lvi2.SubItems[22].Text);                        
                        int kpli;
                        if (kplj < 0) kpli = kplj;
                        else
                        {
                            if (segs.Length < 2) kpli = kplj - 1;
                            else
                            {                                
                                for (j = 0; j < segs.Length - 1 && kplj != (int)segs[j].Field; j++) ;
                                if (j + 1 >= segs.Length) throw new Exception("Kink plate cannot be the most upstream one (track " + lvi2.SubItems[0].Text +")");
                                kpli = (int)segs[j + 1].Field;
                            }
                        }
                        int dsflag;
                        for (dsflag = 0; dsflag < cmbDecaySearchFlag.Items.Count && String.Compare(cmbDecaySearchFlag.Items[dsflag].ToString(), lvi2.SubItems[23].Text, true) != 0; dsflag++) ;
                        if (dsflag == cmbDecaySearchFlag.Items.Count) throw new Exception("Unsupported Decay Search flag.");
                        ot += ("\r\n" + lvi2.SubItems[0].Text + " " + FillNull(lvi2.SubItems[1].Text, "-1") + " " + FillNull(lvi2.SubItems[2].Text, "-1") + " " + lvi2.SubItems[3].Text + " " + lvi2.SubItems[4].Text + " " + lvi2.SubItems[5].Text + " " + lvi2.SubItems[6].Text + " " + lvi2.SubItems[7].Text + " " + FillNull(lvi2.SubItems[15].Text, "-1") + " " + FillNull(lvi2.SubItems[16].Text, "-1") + " " + FillNull(lvi2.SubItems[12].Text, "0") + " " + FillNull(lvi2.SubItems[13].Text, "0") + " " +
                            FillNull(lvi2.SubItems[14].Text, "0") + " " + (lvi2.SubItems[8].Text.Length > 0 ? "1" : "0") + " " + ptype + " " + (lvi2.SubItems[10].Text.Length > 0 ? "1" : "0") + " " + (dkns * 0.5) + " " + (lvi2.SubItems[17].Text.Length > 0 ? ("1 " + lvi2.SubItems[17].Text) : "0 0") + " " + segs.Length + " " + lvi2.SubItems[18].Text + " " + lvi2.SubItems[19].Text + " " + lvi2.SubItems[20].Text + " " + lvi2.SubItems[21].Text + " " + 
                            kpli + " " + kplj + " " + dsflag);                        
                        for (j = segs.Length - 1; j >= 0; j--)
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo info = segs[j];
                            SegmentFlags st = (SegmentFlags)info.AreaSum & SegmentFlags.Type;
                            SegmentFlags sm = (SegmentFlags)info.AreaSum & SegmentFlags.Mode;
                            ot += "\r\n" + info.Field + " " + info.Intercept.X + " " + info.Intercept.Y + " " + info.Intercept.Z + " " + info.Slope.X + " " + info.Slope.Y + " " + 
                                ((st == SegmentFlags.Base) ? "0" : ((st == SegmentFlags.Down) ? "1" : "2")) +
                                " " +
                                ((sm == SegmentFlags.Auto) ? "0" : ((sm == SegmentFlags.SBSF) ? "1" : "2")) +
                                " " + info.Count;
                        }                        
                    }
                    if (ot.Length > 0) ot = ot.Substring(2);
                    System.IO.File.WriteAllText(sd.FileName, ot);
                    MessageBox.Show("File exported", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
#if (DEBUG)
#else
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error exporting file", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
#endif
            }

        private static string FillNull(string s, string defaultval)
        {
            return (s.Length == 0) ? defaultval : s;
        }

        private void btnRedo_Click(object sender, EventArgs e)
        {
            if (lvTracks.SelectedItems.Count == 1)
            {                
                SySal.Tracking.MIPEmulsionTrackInfo[] segs = (SySal.Tracking.MIPEmulsionTrackInfo[])lvTracks.SelectedItems[0].Tag;
                SetInTrackDecaySearchParams(lvTracks.SelectedItems[0], segs);
            }
        }

        private void btnSegAutoFix_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lvTracks.Items)
                try
                {
                    bool fixproblem = false;
                    SySal.Tracking.MIPEmulsionTrackInfo [] infos = (SySal.Tracking.MIPEmulsionTrackInfo [])lvi.Tag;
                    foreach (SySal.Tracking.MIPEmulsionTrackInfo info in infos)
                    {
                        SegmentFlags st = ((SegmentFlags)info.AreaSum) & SegmentFlags.Type;
                        SegmentFlags sm = ((SegmentFlags)info.AreaSum) & SegmentFlags.Mode;
                        if (info.Count == 0 && sm != SegmentFlags.Manual) sm = SegmentFlags.Manual;
                        else if (info.Count > 0 && sm == SegmentFlags.Manual) sm = SegmentFlags.Auto;
                        info.AreaSum = (uint)(st | sm);
                    }
                    if (fixproblem) throw new Exception("Ambiguity found fixing automatically segments of track " + lvi.SubItems[0].Text);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error fixing track " + lvi.SubItems[0].Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
        }
    }

    [Flags]
    internal enum SegmentFlags : uint
    {
        Down = 0x01,
        Up = 0x02,
        Base = 0x03,
        Type = 0x03,
        Auto = 0x10,
        SBSF = 0x20,
        Manual = 0x00,
        Mode = 0x30
    }
}