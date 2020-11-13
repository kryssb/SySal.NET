using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.QuickDataCheck
{
    partial class ViewAsTracksDataSelector : Form
    {
        public ViewAsTracksDataSelector()
        {
            InitializeComponent();
        }

        private void buttonSetVar_Click(object sender, EventArgs e)
        {
            if (listVars.SelectedIndices.Count != 1) return;
            if (comboVars.SelectedIndex < 0) return;
            listVars.Items[listVars.SelectedIndices[0]].SubItems[1].Text = comboVars.SelectedItem.ToString();
            listVars.Items[listVars.SelectedIndices[0]].SubItems[1].Tag = comboVars.SelectedItem.ToString();
        }

        private void buttonRemove_Click(object sender, EventArgs e)
        {
            if (listVars.SelectedIndices.Count != 1) return;
            listVars.Items[listVars.SelectedIndices[0]].SubItems[1].Text = "";
            listVars.Items[listVars.SelectedIndices[0]].SubItems[1].Tag = null;
        }

        private void buttonSetConst_Click(object sender, EventArgs e)
        {
            if (listVars.SelectedIndices.Count != 1) return;
            try
            {
                double d = Convert.ToDouble(textConstant.Text, System.Globalization.CultureInfo.InvariantCulture);
                listVars.Items[listVars.SelectedIndices[0]].SubItems[1].Text = d.ToString(System.Globalization.CultureInfo.InvariantCulture);
                listVars.Items[listVars.SelectedIndices[0]].SubItems[1].Tag = d;
            }
            catch (Exception)
            {
                textConstant.Text = "";
            }
        }

        private void buttonCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        public string[] Variables
        {
            set
            {
                comboVars.Items.Clear();
                foreach (string s in value)
                    comboVars.Items.Add(s);
            }
        }

        private System.Collections.Specialized.OrderedDictionary m_Selection = null;

        public System.Collections.Specialized.OrderedDictionary Selection
        {
            get { return m_Selection; }
        }

        public enum SegmentMode { StartEnd, StartSlopeLength }

        private SegmentMode m_SegmentMode = SegmentMode.StartEnd;

        public SegmentMode Mode { get { return m_SegmentMode; } }

        private void buttonOk_Click(object sender, EventArgs e)
        {
            try
            {
                System.Collections.Specialized.OrderedDictionary dict = new System.Collections.Specialized.OrderedDictionary();
                int i;
                for (i = 0; i < listVars.Items.Count; i++)
                    dict.Add(listVars.Items[i].Text, listVars.Items[i].SubItems[1].Tag);
                if (dict["Xstart"] == null || dict["Ystart"] == null || dict["Zstart"] == null) throw new Exception("Missing information for starting point.");
                if (dict["Xend"] != null && dict["Yend"] != null && dict["Zend"] != null)
                {
                    dict["Length"] = null;
                    dict["Xslope"] = null;
                    dict["Yslope"] = null;
                    m_Selection = dict;
                    DialogResult = DialogResult.OK;
                    m_SegmentMode = SegmentMode.StartEnd;
                    Close();
                    return;
                }
                if (dict["Xslope"] != null && dict["Yslope"] != null && dict["Length"] != null)
                {
                    dict["Xend"] = null;
                    dict["Yend"] = null;
                    dict["Zend"] = null;
                    m_Selection = dict;
                    m_SegmentMode = SegmentMode.StartSlopeLength;
                    DialogResult = DialogResult.OK;
                    Close();
                    return;
                }
                throw new Exception("No clear end point specified and no slope/length specified.");
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error in selection", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
        }
    }
}