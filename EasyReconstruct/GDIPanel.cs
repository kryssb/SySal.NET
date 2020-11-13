using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    public partial class GDIPanel : Form
    {
        public GDIPanel(System.Windows.Forms.Form owner)
        {
            InitializeComponent();
            m_Owner = owner;
        }

        private System.Windows.Forms.Form m_Owner;

        private void OnClose(object sender, FormClosingEventArgs e)
        {
            m_Owner.Close();
        }

        internal void SetSize(uint sizeincrease)
        {
            int XBorder = this.Width - gdiDisplay1.Width;
            int YBorder = this.Height - gdiDisplay1.Height;
            this.Size = new Size((int)(400 + sizeincrease + XBorder), (int)(400 + sizeincrease + YBorder));
            gdiDisplay1.Size = new Size((int)(400 + sizeincrease), (int)(400 + sizeincrease));
            gdiDisplay1.Transform();
            gdiDisplay1.Render();
        }
    }
}