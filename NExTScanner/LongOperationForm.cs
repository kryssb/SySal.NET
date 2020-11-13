using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class LongOperationForm : SySal.SySalNExTControls.SySalDialog
    {
        public LongOperationForm()
        {
            InitializeComponent();
        }

        public double m_Minimum = 0.0;

        public double m_Maximum = 100.0;

        public double Value
        {
            get { return pbProgress.Value; }
            set { pbProgress.Value = value; }
        }

        public delegate void dStopCallback();

        public dStopCallback m_StopCallback;

        private delegate void dClose();

        private delegate void dSetValue(double v);

        private void SetValue(double v)
        {
            Value = v;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            if (m_StopCallback != null) m_StopCallback();
            Close();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            pbProgress.Minimum = m_Minimum;
            pbProgress.Maximum = m_Maximum;
            pbProgress.Value = m_Minimum;
            if (m_StopCallback == null) btnStop.Visible = false;
        }

        public void InvokeClose()
        {
            try
            {
                this.Invoke(new dClose(Close));
            }
            catch (Exception) { }
        }

        public void InvokeSetValue(double v)
        {
            try
            {
                this.Invoke(new dSetValue(SetValue), new object[] { v });
            }
            catch (Exception) { }
        }
    }
}

