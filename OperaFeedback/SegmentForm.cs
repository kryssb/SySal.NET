using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.OperaFeedback
{
    public partial class SegmentForm : Form
    {
        public SegmentForm()
        {
            InitializeComponent();
        }

        public SySal.Tracking.MIPEmulsionTrackInfo[] Segments = new SySal.Tracking.MIPEmulsionTrackInfo[0];

        bool m_Modified = false;

        private void btnSegAdd_Click(object sender, EventArgs e)
        {
            try
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                string [] tokens = txtSegAdd.Text.Split(';');
                if (tokens.Length != 9) throw new Exception("The number of fields must match the number of columns.");
                if ((info.Field = uint.Parse(tokens[0])) <= 0) throw new Exception("Plate must be positive.");
                switch (tokens[1].ToUpper())
                {
                    case "B": info.AreaSum = (uint)(((SegmentFlags)info.AreaSum & (~SegmentFlags.Type)) | SegmentFlags.Base); break;
                    case "D": info.AreaSum = (uint)(((SegmentFlags)info.AreaSum & (~SegmentFlags.Type)) | SegmentFlags.Down); break;
                    case "U": info.AreaSum = (uint)(((SegmentFlags)info.AreaSum & (~SegmentFlags.Type)) | SegmentFlags.Up); break;
                    default: throw new Exception("Track Type = D(ownstream microtrack), U(pstream microtrack) or B(ase).");
                }
                switch (tokens[2].ToUpper())
                {
                    case "M": info.AreaSum = (uint)(((SegmentFlags)info.AreaSum & (~SegmentFlags.Mode)) | SegmentFlags.Manual); break;
                    case "A": info.AreaSum = (uint)(((SegmentFlags)info.AreaSum & (~SegmentFlags.Mode)) | SegmentFlags.Auto); break;
                    case "S": info.AreaSum = (uint)(((SegmentFlags)info.AreaSum & (~SegmentFlags.Mode)) | SegmentFlags.SBSF); break;
/*
                    case "M": info.Sigma = 0; break;
                    case "A": info.Sigma = 1; break;
                    case "S": info.Sigma = -1; break;
 */
                    default: throw new Exception("Track Mode = M(anual), A(uto), S(canback/scanforth).");
                }
                info.Count = ushort.Parse(tokens[3]);
                info.Intercept.X = double.Parse(tokens[4], System.Globalization.CultureInfo.InvariantCulture);
                info.Intercept.Y = double.Parse(tokens[5], System.Globalization.CultureInfo.InvariantCulture);
                info.Intercept.Z = double.Parse(tokens[6], System.Globalization.CultureInfo.InvariantCulture);
                info.Slope.X = double.Parse(tokens[7], System.Globalization.CultureInfo.InvariantCulture);
                info.Slope.Y = double.Parse(tokens[8], System.Globalization.CultureInfo.InvariantCulture);
                int i;
                for (i = 0; i < lvSegments.Items.Count && ((SySal.Tracking.MIPEmulsionTrackInfo)lvSegments.Items[i].Tag).Intercept.Z > info.Intercept.Z; i++) ;
                if (i < lvSegments.Items.Count && ((SySal.Tracking.MIPEmulsionTrackInfo)lvSegments.Items[i].Tag).Intercept.Z == info.Intercept.Z)
                    lvSegments.Items.RemoveAt(i);
                lvSegments.Items.Insert(i, TrackToItem(info));
                m_Modified = true;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void btnSegRemove_Click(object sender, EventArgs e)
        {
            lvSegments.BeginUpdate();
            while (lvSegments.SelectedItems.Count > 0)
            {
                m_Modified = true;
                lvSegments.Items.Remove(lvSegments.SelectedItems[0]);
            }
            lvSegments.EndUpdate();
        }

        private void OnLoad(object sender, EventArgs e)
        {            
            lvSegments.Items.Clear();
            foreach (SySal.Tracking.MIPEmulsionTrackInfo info in Segments)
            {
                int i;
                for (i = 0; i < lvSegments.Items.Count && ((SySal.Tracking.MIPEmulsionTrackInfo)lvSegments.Items[i].Tag).Intercept.Z > info.Intercept.Z; i++) ;
                lvSegments.Items.Insert(i, TrackToItem(info));                
            }
        }

        static ListViewItem TrackToItem(SySal.Tracking.MIPEmulsionTrackInfo info)
        {
            ListViewItem lvi = new ListViewItem();
            lvi.Text = info.Field.ToString();
            SegmentFlags st = (SegmentFlags)info.AreaSum & SegmentFlags.Type;
            SegmentFlags sm = (SegmentFlags)info.AreaSum & SegmentFlags.Mode;            
            lvi.SubItems.Add((st == SegmentFlags.Base) ? "B" : ((st == SegmentFlags.Down) ? "D" : "U"));
            lvi.SubItems.Add((sm == SegmentFlags.Auto) ? "A" : ((sm == SegmentFlags.SBSF) ? "S" : "M"));
            /*
            lvi.SubItems.Add(info.AreaSum == 0 ? "B" : (info.AreaSum == 1 ? "D" : "U"));
            lvi.SubItems.Add(((info.Sigma < 0.0) ? "S" : (info.Sigma > 0.0 ? "A" : "M")));
             */
            lvi.SubItems.Add(info.Count.ToString());
            lvi.SubItems.Add(info.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(info.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(info.Intercept.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(info.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.SubItems.Add(info.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
            lvi.Tag = info;
            return lvi;
        }

        private void OnClose(object sender, FormClosingEventArgs e)
        {
            if (m_Modified && MessageBox.Show("Segment list modified, commit changes?", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Information, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                Segments = new SySal.Tracking.MIPEmulsionTrackInfo[lvSegments.Items.Count];
                int i;
                for (i = 0; i < Segments.Length; i++)
                    Segments[i] = (SySal.Tracking.MIPEmulsionTrackInfo)lvSegments.Items[i].Tag;
            }
        }

        private void lvSegments_DoubleClick(object sender, EventArgs e)
        {
            if (lvSegments.SelectedItems.Count == 1)
            {
                ListViewItem lvi = lvSegments.SelectedItems[0];
                int i;
                string s = "";
                for (i = 0; i < lvi.SubItems.Count; i++)
                {
                    if (i > 0) s += ";";
                    s += lvi.SubItems[i].Text;
                }
                txtSegAdd.Text = s;
            }
        }
    }
}