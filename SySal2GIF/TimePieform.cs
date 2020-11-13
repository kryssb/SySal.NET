using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.SySal2GIF
{
    public partial class TimerPieForm : Form
    {
        public delegate void dNotifyStop();

        public TimerPieForm(dNotifyStop ns)
        {
            InitializeComponent();
            m_NotifyStop = ns;
        }

        dNotifyStop m_NotifyStop;

        double m_Progress;

        public double Progress
        {
            get
            {
                return m_Progress;
            }
            set
            {
                m_Progress = value;
                Graphics g = pieBox.CreateGraphics();
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                g.FillPie(new SolidBrush(Color.Blue), 2, 2, pieBox.Width - 4, pieBox.Height - 4, 0, (int)(360.0 * m_Progress));
            }
        }

        private void StopButton_Click(object sender, EventArgs e)
        {
            if (m_NotifyStop != null) m_NotifyStop();
        }
    }
}