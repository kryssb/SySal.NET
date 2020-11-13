using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    public partial class SegReplaceForm : Form
    {
        public SegReplaceForm()
        {
            InitializeComponent();
        }

        public string[] Replacements;

        public bool[] Confirmed;

        private void OnLoad(object sender, EventArgs e)
        {
            Confirmed = new bool[Replacements.Length];
            lvReplacements.Items.Clear();
            foreach (string s in Replacements)
            {                
                ListViewItem lvi = new ListViewItem(s.Split(' '));
                lvi.Checked = true;
                lvReplacements.Items.Add(lvi);
            }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            int i;
            for (i = 0; i < lvReplacements.Items.Count; i++)
                Confirmed[i] = lvReplacements.Items[i].Checked;
            Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }
    }
}