using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    internal partial class GPUSettingsForm : SySal.SySalNExTControls.SySalDialog
    {
        public GPUSettingsForm()
        {
            InitializeComponent();
        }

        public Scanner.SImageProcessor[] iGPUs = null;

        private void OnLoad(object sender, EventArgs e)
        {
            clvEnabledGPUs.Items.Clear();
            foreach (Scanner.SImageProcessor s in iGPUs)
            {
                string n = s.IProc.ToString();
                n = n.Substring(0, n.IndexOfAny(new char[] { '\n', '\r' }));
                clvEnabledGPUs.Items.Add(n, s.Enabled);
            }
            btnGPUSetup.Visible = (iGPUs.Length > 0 && ((object)iGPUs[0].IProc) is SySal.Management.IMachineSettingsEditor);                            
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (clvEnabledGPUs.CheckedItems.Count > 0 || MessageBox.Show("No GPU selected for processing.\r\nYou will not be able to process any image.\r\nProceed?", "Warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                foreach (Scanner.SImageProcessor s in iGPUs) s.Enabled = false;
                foreach (int i in clvEnabledGPUs.CheckedIndices) iGPUs[i].Enabled = true;
                DialogResult = DialogResult.OK;
                Close();
            }
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }
    }
}

